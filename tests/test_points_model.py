import numpy as np
import torch

from breakpoint_vision.points.features import N_FEATURES
from breakpoint_vision.points.model import HOP, N_FFT, SAMPLE_RATE, SAMPLES_PER_STEP, LogMel, PointSegmenter, chunk_bounds, mel_filterbank


def test_logmel_produces_ten_frames_per_step():
    audio = torch.zeros(2, 30 * SAMPLES_PER_STEP)
    assert LogMel()(audio).shape == (2, 64, 300)


def test_logmel_matches_a_numpy_reference():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 0.1, 4 * SAMPLES_PER_STEP).astype(np.float32)
    ours = LogMel()(torch.from_numpy(y)[None]).numpy()[0]
    pad_l = N_FFT // 2 - HOP // 2
    yp = np.pad(y, (pad_l, N_FFT - HOP - pad_l))
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)
    frames = np.stack([yp[i * HOP:i * HOP + N_FFT] * window for i in range(ours.shape[1])])
    power = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    ref = np.log(mel_filterbank() @ power.T + 1e-6)
    assert np.max(np.abs(ours - ref)) < 1e-3


def test_a_pure_tone_lands_in_the_matching_mel_band():
    t = np.arange(2 * SAMPLES_PER_STEP) / SAMPLE_RATE
    y = (0.5 * np.sin(2 * np.pi * 3000 * t)).astype(np.float32)
    m = LogMel()(torch.from_numpy(y)[None]).numpy()[0]
    band = int(m[:, 5:15].mean(1).argmax())
    centers = np.argmax(mel_filterbank(), axis=1) * (SAMPLE_RATE / 2) / (N_FFT // 2)
    assert abs(centers[band] - 3000) < 250


def test_segmenter_outputs_three_logits_per_step():
    m = PointSegmenter().eval()
    out = m(torch.zeros(1, 50 * SAMPLES_PER_STEP), torch.zeros(1, 50, N_FEATURES))
    assert out.shape == (1, 50, 3)


def test_chunks_cover_every_step_exactly_once():
    for n in (1, 599, 1200, 1201, 4321):
        kept = np.zeros(n, int)
        for start, stop, lo, hi in chunk_bounds(n, 1200, 200):
            assert 0 <= start < stop <= n and stop - start <= 1200
            kept[start + lo:start + hi] += 1
        assert np.all(kept == 1), n


def test_visual_branch_takes_one_thumbnail_per_step():
    m = PointSegmenter(visual_dim=16).eval()
    out = m(torch.zeros(1, 20 * SAMPLES_PER_STEP), torch.zeros(1, 20, N_FEATURES), torch.rand(1, 20, 144, 256))
    assert out.shape == (1, 20, 3)


def test_visual_branch_ignores_overall_brightness():
    from breakpoint_vision.points.model import VisualEncoder

    enc = VisualEncoder(8).eval()
    x = torch.rand(1, 5, 144, 256) * 0.8 + 0.1
    a = enc(x)
    b = enc(x * 0.25)
    assert torch.allclose(a, b, atol=0.05)


def test_motion_compression_spreads_ordinary_motion_and_keeps_order():
    from breakpoint_vision.points.model import compress_motion
    feats = torch.zeros(1, 4, N_FEATURES)
    feats[0, :, 5:8] = torch.tensor([0.0, 0.001, 0.01, 0.5]).unsqueeze(1)
    feats[0, :, 0] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    out = compress_motion(feats)
    m = out[0, :, 5]
    assert torch.equal(out[0, :, :5], feats[0, :, :5])
    assert m[0] == 0 and m[1] > 0.08 and m[2] > 0.3 and m[3] < 1.0
    assert torch.all(m[1:] > m[:-1])


def test_load_segmenter_honors_the_training_input_handling(tmp_path):
    from breakpoint_vision.points.model import PointSegmenter, load_segmenter
    old = PointSegmenter(compress=False)
    torch.save({'model': old.state_dict(), 'visual_dim': 0}, tmp_path / 'old.pt')
    assert load_segmenter(str(tmp_path / 'old.pt')).compress is False
    new = PointSegmenter(visual_dim=16)
    torch.save({'model': new.state_dict(), 'visual_dim': 16, 'compress': True}, tmp_path / 'new.pt')
    loaded = load_segmenter(str(tmp_path / 'new.pt'))
    assert loaded.compress is True and loaded.visual_dim == 16
