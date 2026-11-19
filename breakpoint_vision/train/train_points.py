"""Train the point segmentation model on per-video feature files.

Each `data/features/<vid>.npz` holds `feats` (T, N_FEATURES), `audio`
(T * SAMPLES_PER_STEP), and labels: `silver` (automatic) and optionally `gold`
with `gold_window`. Gold videos are never used for training.

Augmentation targets the picture getting worse while the play does not change.
Ball features are weakened, dropped for stretches, or polluted with false
detections (what a dark court or a busy neighboring court does to the tracker),
and audio is rescaled and mixed with noise. The label always comes from the clean footage, so the model learns to
lean on whatever signal survives.
"""

from __future__ import annotations

import json
import os
import random
import time
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch import nn

from ..points.decode import DecodeConfig, decode, evaluate
from ..points.features import STEP_SECONDS
from ..points.model import CHUNK_OVERLAP, CHUNK_STEPS, HOP_PER_STEP, SAMPLES_PER_STEP, LogMel, PointSegmenter, chunk_bounds


@torch.no_grad()
def compute_logmel(audio: np.ndarray, chunk_steps: int = 3000) -> np.ndarray:
    """Log-mel for a whole video, (N_MELS, steps * HOP_PER_STEP), computed in chunks."""
    lm = LogMel().eval()
    steps = len(audio) // SAMPLES_PER_STEP
    out = []
    for a in range(0, steps, chunk_steps):
        b = min(steps, a + chunk_steps)
        x = torch.from_numpy(audio[a * SAMPLES_PER_STEP:b * SAMPLES_PER_STEP].astype(np.float32))[None]
        out.append(lm(x)[0].numpy())
    return np.concatenate(out, 1).astype(np.float16)


def cut_mask(thumbs: np.ndarray, factor: float = 20.0, pad_steps: int = 10) -> np.ndarray:
    """Steps within `pad_steps` of an editing cut (a frame change far above the video's norm).

    Fixed-camera recordings never contain cuts; edited public videos do. Masking the
    seconds around a cut keeps the clips useful for learning what play looks
    like without teaching the model that a cut means a point boundary.
    """
    t = np.asarray(thumbs[:, ::4, ::4], np.float32)
    diff = np.zeros(len(t), np.float32)
    diff[1:] = np.abs(t[1:] - t[:-1]).mean(axis=(1, 2))
    cuts = np.flatnonzero(diff > factor * max(float(np.median(diff)), 0.5))
    mask = np.zeros(len(t), bool)
    for c in cuts:
        mask[max(0, c - pad_steps):c + pad_steps + 1] = True
    return mask


def boost_short_segments(label: np.ndarray, weight: np.ndarray, max_steps: int, boost: float, margin: int = 5) -> np.ndarray:
    """Multiply the weight of short reviewed segments (and a few steps around them).

    Serve faults and unreturned serves last one to three seconds, so per-step
    losses barely notice them next to long rallies. Only steps already at full
    (reviewed) weight are boosted, so silver labels stay down-weighted.
    """
    out = weight.copy()
    y = np.concatenate([[0], (label > 0.5).astype(np.int8), [0]])
    starts = np.flatnonzero(np.diff(y) == 1)
    ends = np.flatnonzero(np.diff(y) == -1)
    for a, b in zip(starts, ends):
        if b - a <= max_steps and weight[a:b].min() >= 1.0:
            lo, hi = max(0, a - margin), min(len(out), b + margin)
            out[lo:hi] = np.where(weight[lo:hi] >= 1.0, weight[lo:hi] * boost, weight[lo:hi])
    return out


def load_video(path: str, thumbs_dir: str = 'data/thumbs', silver_weight: float = 0.3, label_names: Sequence[str] = ('train_gold',), short_seconds: float = 4.0, short_boost: float = 1.0) -> Dict[str, np.ndarray]:
    d = np.load(path)
    v = {k: d[k] for k in d.files}
    vid = os.path.basename(path)[:-4]
    tp = f'{thumbs_dir}/{vid}.npy'
    if os.path.exists(tp):
        v['thumbs'] = np.load(tp, mmap_mode='r')
    T = len(v['feats'])
    label = v['silver'].astype(np.float32).copy() if 'silver' in v else np.zeros(T, np.float32)
    weight = np.full(T, silver_weight, np.float32)
    for name in label_names:
        if name in v and f'{name}_window' in v:
            a, b = np.round(np.asarray(v[f'{name}_window']) / STEP_SECONDS).astype(int)
            label[a:b] = v[name][a:b]
            weight[a:b] = 1.0
    if 'thumbs' in v:
        cuts = cut_mask(v['thumbs'])[:T]
        weight[:len(cuts)][cuts] = 0.0
    if short_boost != 1.0:
        weight = boost_short_segments(label, weight, int(round(short_seconds / STEP_SECONDS)), short_boost)
    v['label'] = label
    v['weight'] = weight
    v['boundaries'] = target_boundaries(label)
    cache = path[:-4] + '.logmel.npy'
    if os.path.exists(cache):
        v['logmel'] = np.load(cache)
    else:
        v['logmel'] = compute_logmel(v['audio'])
        np.save(cache, v['logmel'])
    del v['audio']
    return v


def target_boundaries(y: np.ndarray, sigma_steps: float = 2.0) -> np.ndarray:
    starts = np.flatnonzero(np.diff(np.concatenate([[0], y])) > 0)
    ends = np.flatnonzero(np.diff(np.concatenate([y, [0]])) < 0) + 1
    idx = np.arange(len(y))
    out = np.zeros((len(y), 2), np.float32)
    for s in starts:
        out[:, 0] = np.maximum(out[:, 0], np.exp(-0.5 * ((idx - s) / sigma_steps) ** 2))
    for e in ends:
        out[:, 1] = np.maximum(out[:, 1], np.exp(-0.5 * ((idx - e) / sigma_steps) ** 2))
    return out


def degrade_feats(f: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    f = f.copy()
    T = len(f)
    mode = rng.random()
    if mode < 0.35:
        scale = rng.uniform(0.0, 0.7)
        f[:, 0] *= scale
        f[:, 1] *= scale * rng.uniform(0.3, 1.0)
        f[:, 2] *= rng.uniform(0.0, 1.0)
    if rng.random() < 0.5:
        for _ in range(int(rng.integers(1, 6))):
            a = int(rng.integers(0, T))
            b = min(T, a + int(rng.integers(20, 300)))
            f[a:b, 0:3] = 0.0
            f[a:b, 3:5] = 0.5
    if rng.random() < 0.3:
        n = int(T * rng.uniform(0.01, 0.08))
        idx = rng.integers(0, T, n)
        f[idx, 0] = rng.uniform(0.5, 0.95, n)
        f[idx, 1] = rng.uniform(0.2, 1.0, n)
        f[idx, 2] = rng.uniform(0.0, 1.5, n)
        f[idx, 3] = rng.uniform(0, 1, n)
        f[idx, 4] = rng.uniform(0, 1, n)
    if rng.random() < 0.3:
        f[:, 5:8] *= rng.uniform(0.3, 2.0)
    return f


def degrade_logmel(m: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Gain, background noise and dropouts applied in the log-mel power domain."""
    m = m.astype(np.float32) + float(np.log(10 ** (rng.uniform(-18, 6) / 10)))
    if rng.random() < 0.5:
        floor = float(np.log(10 ** rng.uniform(-6.0, -2.0)))
        m = np.logaddexp(m, floor + rng.normal(0, 0.3, size=(m.shape[0], 1)).astype(np.float32))
    if rng.random() < 0.1:
        m = np.full_like(m, np.log(1e-6))
    return m


def degrade_thumbs(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Global exposure, gamma, contrast, noise and blur on a (T, H, W) window in 0..1."""
    t = t * float(np.exp(rng.uniform(np.log(0.15), np.log(1.8))))
    t = np.clip(t, 0, 1) ** float(rng.uniform(0.6, 1.6))
    if rng.random() < 0.6:
        t = t + rng.normal(0, rng.uniform(0.005, 0.06), size=t.shape).astype(np.float32)
    if rng.random() < 0.3:
        k = np.array([0.25, 0.5, 0.25], np.float32)
        t = (t[:, :-2] * k[0] + t[:, 1:-1] * k[1] + t[:, 2:] * k[2])
        t = np.pad(t, ((0, 0), (1, 1), (0, 0)), mode='edge')
    return np.clip(t, 0, 1).astype(np.float32)


class Windows(torch.utils.data.Dataset):
    def __init__(self, videos: Sequence[Dict[str, np.ndarray]], steps: int, n: int, seed: int, augment: bool, gold_fraction: float = 0.7):
        self.videos, self.steps, self.n, self.seed, self.augment, self.gold_fraction = videos, steps, n, seed, augment, gold_fraction
        self.gold_spans = []
        for i, v in enumerate(videos):
            idx = np.flatnonzero(v['weight'] >= 1.0)
            if len(idx):
                self.gold_spans.append((i, int(idx[0]), int(idx[-1])))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng(self.seed * 1_000_003 + i)
        if self.gold_spans and rng.random() < self.gold_fraction:
            vi, ga, gb = self.gold_spans[int(rng.integers(0, len(self.gold_spans)))]
            v = self.videos[vi]
            T = len(v['feats'])
            a = int(np.clip(rng.integers(ga - self.steps // 2, max(ga - self.steps // 2 + 1, gb - self.steps // 2)), 0, max(0, T - self.steps)))
        else:
            v = self.videos[int(rng.integers(0, len(self.videos)))]
            T = len(v['feats'])
            a = int(rng.integers(0, max(0, T - self.steps) + 1))
        b = a + self.steps
        f = v['feats'][a:b]
        y = v['label'][a:b]
        w = v['weight'][a:b]
        tb = v['boundaries'][a:b]
        th = (np.asarray(v['thumbs'][a:b], np.float32) / 255.0) if 'thumbs' in v else np.zeros((len(f), 144, 256), np.float32)
        m = v['logmel'][:, a * HOP_PER_STEP:b * HOP_PER_STEP].astype(np.float32)
        if len(f) < self.steps:
            pad = self.steps - len(f)
            f = np.pad(f, ((0, pad), (0, 0)))
            f[-pad:, 3:5] = 0.5
            y = np.pad(y, (0, pad))
            tb = np.pad(tb, ((0, pad), (0, 0)))
            w = np.pad(w, (0, pad))
            th = np.pad(th, ((0, pad), (0, 0), (0, 0)), mode='edge')
            m = np.pad(m, ((0, 0), (0, pad * HOP_PER_STEP)), constant_values=np.log(1e-6))
        if self.augment:
            if rng.random() < 0.7:
                f = degrade_feats(f, rng)
            if rng.random() < 0.6:
                m = degrade_logmel(m, rng)
            if rng.random() < 0.7:
                th = degrade_thumbs(th, rng)
        return torch.from_numpy(m), torch.from_numpy(f.astype(np.float32)), torch.from_numpy(th), torch.from_numpy(y.astype(np.float32)), torch.from_numpy(w.astype(np.float32)), torch.from_numpy(tb)


@torch.no_grad()
def predict(model: PointSegmenter, v: Dict[str, np.ndarray], device: str, chunk: int = CHUNK_STEPS, overlap: int = CHUNK_OVERLAP) -> np.ndarray:
    model.eval()
    T = len(v['feats'])
    prob = np.zeros(T, np.float32)
    for start, stop, lo, hi in chunk_bounds(T, chunk, overlap):
        f = torch.from_numpy(v['feats'][start:stop].astype(np.float32))[None].to(device)
        m = torch.from_numpy(v['logmel'][:, start * HOP_PER_STEP:stop * HOP_PER_STEP].astype(np.float32))[None].to(device)
        th = torch.from_numpy(np.asarray(v['thumbs'][start:stop], np.float32) / 255.0)[None].to(device) if 'thumbs' in v else None
        out = torch.sigmoid(model.forward_logmel(m, f, th)[0, :, 0]).cpu().numpy()
        prob[start + lo:start + hi] = out[lo:hi]
    return prob


def segments_from_labels(y: np.ndarray) -> List[dict]:
    segs, inside, s = [], False, 0
    for i, v in enumerate(y):
        if v and not inside:
            inside, s = True, i
        elif not v and inside:
            segs.append({'start': s * STEP_SECONDS, 'end': i * STEP_SECONDS})
            inside = False
    if inside:
        segs.append({'start': s * STEP_SECONDS, 'end': len(y) * STEP_SECONDS})
    return segs


def score_video(model, v, device, label='silver', window=None, cfg=None):
    prob = predict(model, v, device)
    pred = decode(prob, cfg or DecodeConfig.tuned())
    gold = segments_from_labels(v[label])
    win = window if window is not None else (0.0, len(prob) * STEP_SECONDS)
    return evaluate(pred, gold, win)


def has_train_gold(v: Dict[str, np.ndarray]) -> bool:
    return 'train_gold' in v and 'train_gold_window' in v


def evaluate_window(model, v, device, label: str, cfg=None) -> dict:
    prob = predict(model, v, device)
    window = tuple(np.asarray(v[f'{label}_window']).tolist())
    return evaluate(decode(prob, cfg or DecodeConfig.tuned()), segments_from_labels(v[label]), window)


def train(feature_dir: str, out_dir: str, gold_ids: Sequence[str], val_ids: Sequence[str] = (), steps: int = 4000, batch: int = 16, window_steps: int = 600, lr: float = 2e-3, device: str = 'cpu', seed: int = 0, visual_dim: int = 0, eval_every: int = 250):
    """Fit on every non-test video; validate on the train-gold windows of `val_ids`.

    Test videos (`gold_ids`) are never loaded here; scoring them is a separate,
    final step.
    """
    gold = set(gold_ids)
    names = sorted(f[:-4] for f in os.listdir(feature_dir) if f.endswith('.npz') and not f.endswith('.logmel.npy'))
    train_names = [n for n in names if n not in gold and (not visual_dim or os.path.exists(f'data/thumbs/{n}.npy'))]
    val_set = set(val_ids)
    fit = [load_video(f'{feature_dir}/{n}.npz') for n in train_names if n not in val_set]
    val = {n: load_video(f'{feature_dir}/{n}.npz') for n in train_names if n in val_set}
    val = {n: v for n, v in val.items() if has_train_gold(v)}
    print(f'fit {len(fit)} videos ({sum(has_train_gold(v) for v in fit)} with train-gold), val {sorted(val)}', flush=True)
    torch.manual_seed(seed)
    model = PointSegmenter(visual_dim=visual_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    bce = nn.BCEWithLogitsLoss(reduction='none')
    loader = torch.utils.data.DataLoader(Windows(fit, window_steps, steps * batch, seed, True), batch_size=batch, num_workers=4, persistent_workers=True)
    os.makedirs(out_dir, exist_ok=True)
    best, t0, history = -1e9, time.time(), []
    for step, (m, f, th, y, w, tb) in enumerate(loader, 1):
        model.train()
        m, f, th, y, w, tb = m.to(device), f.to(device), th.to(device), y.to(device), w.to(device), tb.to(device)
        out = model.forward_logmel(m, f, th if visual_dim else None)
        main = (bce(out[..., 0], y) * w).sum() / w.sum().clamp(min=1.0)
        bound = ((bce(out[..., 1:], tb) * (1 + 4 * tb)) * w[..., None]).sum() / (2 * w.sum().clamp(min=1.0))
        loss = main + 0.5 * bound
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        if step % 50 == 0:
            print(f'step {step} loss {loss.item():.4f} {time.time() - t0:.0f}s', flush=True)
        if val and (step % eval_every == 0 or step == steps):
            scores = {n: evaluate_window(model, v, device, 'train_gold') for n, v in val.items()}
            summary = {k: float(np.mean([m[k] for m in scores.values() if m[k] is not None])) for k in ('recall', 'precision', 'play_time_kept', 'dead_time_removed', 'start_err_median', 'end_err_median')}
            history.append({'step': step, 'val': summary, 'per_video': scores})
            print('eval', step, json.dumps({k: round(v, 3) for k, v in summary.items()}), flush=True)
            score = summary['recall'] + summary['precision'] + 2 * summary['play_time_kept'] + summary['dead_time_removed']
            if score > best:
                best = score
                torch.save({'model': model.state_dict(), 'step': step, 'val': summary, 'visual_dim': visual_dim, 'compress': model.compress}, f'{out_dir}/points_best.pt')
            json.dump(history, open(f'{out_dir}/history.json', 'w'), indent=1)
        if step >= steps:
            break
    torch.save({'model': model.state_dict(), 'step': steps, 'visual_dim': visual_dim, 'compress': model.compress}, f'{out_dir}/points_last.pt')
    return model
