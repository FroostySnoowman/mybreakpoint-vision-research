"""Point segmentation model: audio + ball/motion features -> in-play probability.

Inputs for a chunk of `T` steps (`STEP_SECONDS` each):
- `audio`: mono float waveform at `SAMPLE_RATE`, exactly `T * HOP_PER_STEP *
  HOP` samples (zero padded at the end of a video).
- `feats`: (T, N_FEATURES) from `features.step_features`.

The log-mel front end is part of the network (fixed convolution weights), so a
caller feeds raw PCM and gets exactly the spectrogram the model was trained on,
with no separate DSP implementation to drift.

Output: per-step logits for "a point is in play". Two auxiliary heads predict
point starts and ends; they sharpen boundaries during training.

Long videos are processed in overlapping chunks of `CHUNK_STEPS` steps
(`chunk_bounds`).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
from torch import nn

from .features import N_FEATURES, STEP_SECONDS

SAMPLE_RATE = 16000
N_FFT = 400
HOP = 160
N_MELS = 64
HOP_PER_STEP = int(round(STEP_SECONDS * SAMPLE_RATE / HOP))
SAMPLES_PER_STEP = HOP_PER_STEP * HOP
CHUNK_STEPS = 1200
CHUNK_OVERLAP = 320


def mel_filterbank(n_mels=N_MELS, n_fft=N_FFT, sr=SAMPLE_RATE, fmin=50.0, fmax=8000.0) -> np.ndarray:
    def hz_to_mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10 ** (m / 2595.0) - 1.0)

    n_bins = n_fft // 2 + 1
    freqs = np.linspace(0, sr / 2, n_bins)
    mels = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz = mel_to_hz(mels)
    fb = np.zeros((n_mels, n_bins), np.float32)
    for i in range(n_mels):
        lo, mid, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / max(mid - lo, 1e-6)
        down = (hi - freqs) / max(hi - mid, 1e-6)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


class LogMel(nn.Module):
    """STFT magnitude via fixed conv1d (real and imaginary DFT kernels), then mel and log."""

    def __init__(self):
        super().__init__()
        n_bins = N_FFT // 2 + 1
        window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
        k = np.arange(n_bins)[:, None]
        n = np.arange(N_FFT)[None, :]
        real = (np.cos(2 * np.pi * k * n / N_FFT) * window).astype(np.float32)
        imag = (-np.sin(2 * np.pi * k * n / N_FFT) * window).astype(np.float32)
        self.register_buffer('dft', torch.from_numpy(np.concatenate([real, imag], 0)[:, None, :]))
        self.register_buffer('mel', torch.from_numpy(mel_filterbank()))
        self.n_bins = n_bins

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = audio[:, None, :]
        x = nn.functional.pad(x, (N_FFT // 2 - HOP // 2, N_FFT // 2 + HOP // 2 - HOP))
        spec = nn.functional.conv1d(x, self.dft, stride=HOP)
        re, im = spec[:, : self.n_bins], spec[:, self.n_bins:]
        power = re * re + im * im
        mel = torch.matmul(self.mel, power)
        return torch.log(mel + 1e-6)


class AudioEncoder(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(N_MELS, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(),
            nn.Conv1d(96, 96, 5, padding=2), nn.BatchNorm1d(96), nn.ReLU(),
            nn.Conv1d(96, dim, HOP_PER_STEP, stride=HOP_PER_STEP), nn.BatchNorm1d(dim), nn.ReLU(),
        )

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        logmel = (logmel + 8.0) / 4.0
        return self.net(logmel)


THUMB_H, THUMB_W = 144, 256
VISUAL_STRIDE = 2
TEMPORAL_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 1, 4, 16)


class VisualEncoder(nn.Module):
    """Per-step embedding of a small gray frame and its change from the previous step.

    Input `thumbs` is (B, T, THUMB_H, THUMB_W) in 0..1. Both channels are divided
    by the frame's mean brightness, so a dim night frame and a sunny one look
    alike to the network; the coarse 3x5 pooled layout keeps far court, net
    and near court apart, which is where serves and split steps happen.
    """

    def __init__(self, dim=48):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(2, 16, 5, stride=2, padding=2), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 48, 3, stride=2, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 5)),
        )
        self.proj = nn.Sequential(nn.Linear(64 * 15, dim), nn.ReLU())

    def forward(self, thumbs: torch.Tensor) -> torch.Tensor:
        """Embeds every second step (5 Hz) and repeats each embedding for the next step."""
        b, t, h, w = thumbs.shape
        prev = torch.cat([thumbs[:, :1], thumbs[:, :-1]], 1)
        cur, prev = thumbs[:, ::VISUAL_STRIDE], prev[:, ::VISUAL_STRIDE]
        n = cur.shape[1]
        scale = cur.mean(dim=(2, 3), keepdim=True) + 0.01
        x = torch.stack([cur / scale, (cur - prev).abs() / scale], 2).reshape(b * n, 2, h, w)
        y = self.proj(self.net(x).reshape(b * n, -1)).reshape(b, n, -1)
        y = y.repeat_interleave(VISUAL_STRIDE, dim=1)[:, :t]
        return y.transpose(1, 2)


class TemporalBlock(nn.Module):
    def __init__(self, ch: int, dilation: int):
        super().__init__()
        self.conv = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm = nn.BatchNorm1d(ch)
        self.mix = nn.Conv1d(ch, ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mix(torch.relu(self.norm(self.conv(x))))


MOTION_SLICE = slice(5, 8)
MOTION_GAIN = 1000.0


def compress_motion(feats: torch.Tensor) -> torch.Tensor:
    """Log-compress the banded motion features (the last three columns).

    Raw band motion has a median near 1e-3 and rare scene-cut spikes near 0.5,
    so a batch norm scaled by those spikes flattens ordinary player movement to
    zero. log1p(gain * x) / log1p(gain) keeps both in a usable 0..1 range.
    """
    motion = torch.log1p(feats[..., MOTION_SLICE].clamp(min=0) * MOTION_GAIN) / math.log1p(MOTION_GAIN)
    return torch.cat([feats[..., :MOTION_SLICE.start], motion], dim=-1)


class PointSegmenter(nn.Module):
    def __init__(self, audio_dim=48, feat_dim=32, hidden=64, visual_dim=0, compress=True):
        super().__init__()
        self.visual_dim = visual_dim
        self.compress = compress
        self.visual = VisualEncoder(visual_dim) if visual_dim else None
        self.logmel = LogMel()
        self.audio = AudioEncoder(audio_dim)
        self.feat_norm = nn.BatchNorm1d(N_FEATURES)
        self.feats = nn.Sequential(nn.Linear(N_FEATURES, feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim), nn.ReLU())
        self.local = nn.Sequential(nn.Conv1d(audio_dim + feat_dim + visual_dim, hidden, 5, padding=2), nn.ReLU())
        self.temporal = nn.Sequential(*[TemporalBlock(hidden, d) for d in TEMPORAL_DILATIONS])
        self.head = nn.Conv1d(hidden, 3, 1)

    def forward(self, audio: torch.Tensor, feats: torch.Tensor, thumbs: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.forward_logmel(self.logmel(audio), feats, thumbs)

    def forward_logmel(self, logmel: torch.Tensor, feats: torch.Tensor, thumbs: Optional[torch.Tensor] = None) -> torch.Tensor:
        a = self.audio(logmel)
        if self.compress:
            feats = compress_motion(feats)
        f = self.feats(self.feat_norm(feats.transpose(1, 2)).transpose(1, 2)).transpose(1, 2)
        parts = [a, f]
        if self.visual is not None:
            parts.append(self.visual(thumbs))
        x = self.temporal(self.local(torch.cat(parts, 1)))
        return self.head(x).transpose(1, 2)


def load_segmenter(path: str, map_location: str = 'cpu') -> PointSegmenter:
    """Load a trained segmenter with the input handling it was trained with.

    Checkpoints from before motion compression carry no 'compress' key and were
    trained on raw motion, so they load with compression off.
    """
    ckpt = torch.load(path, map_location=map_location)
    model = PointSegmenter(visual_dim=ckpt.get('visual_dim', 0), compress=ckpt.get('compress', False))
    model.load_state_dict(ckpt['model'])
    return model.eval()


def chunk_bounds(n_steps: int, chunk: int, overlap: int):
    """Overlapping chunks covering `n_steps`, as (start, stop, keep_lo, keep_hi).

    Predictions near a chunk edge see less context, so each step is taken from
    exactly one chunk: the split between neighbors sits in the middle of their
    overlap. `keep_lo:keep_hi` is relative to the chunk start.
    """
    if n_steps <= chunk:
        return [(0, n_steps, 0, n_steps)]
    spans, start, stride = [], 0, chunk - overlap
    while True:
        stop = min(start + chunk, n_steps)
        start = max(0, stop - chunk)
        spans.append((start, stop))
        if stop == n_steps:
            break
        start += stride
    out = []
    for i, (a, b) in enumerate(spans):
        lo = 0 if i == 0 else (spans[i - 1][1] + a) // 2
        hi = n_steps if i == len(spans) - 1 else (b + spans[i + 1][0]) // 2
        out.append((a, b, lo - a, hi - a))
    return out
