"""Turn per-step in-play probabilities into play segments, and score them.

`decode` applies an optional per-video rescale, median smoothing, hysteresis
thresholds, gap merging and a minimum duration. Its settings are tuned operating
values: build them with `DecodeConfig.tuned()` or pass them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..params import tuned
from .features import STEP_SECONDS


@dataclass
class DecodeConfig:
    on: float
    off: float
    smooth_steps: int
    min_seconds: float
    merge_gap_seconds: float
    scale_quantile: float
    scale_floor: float

    @classmethod
    def tuned(cls, **overrides) -> "DecodeConfig":
        return cls(**{**tuned("decode"), **overrides})


def rescale(p: np.ndarray, quantile: float, floor: float) -> np.ndarray:
    """Divide by the video's own `quantile` probability (never by less than
    `floor`) so a model that is under-confident on one video still clears the
    thresholds. A floor of 1 leaves the probabilities unchanged."""
    if floor >= 1.0 or len(p) == 0:
        return p
    ref = max(np.float32(np.quantile(p, quantile)), np.float32(floor))
    return np.minimum(np.float32(1.0), p / ref).astype(np.float32)


def smooth(p: np.ndarray, k: int) -> np.ndarray:
    if k <= 1:
        return p
    pad = k // 2
    padded = np.pad(p, (pad, pad), mode='edge')
    return np.array([np.median(padded[i:i + k]) for i in range(len(p))], dtype=np.float32)


def decode(prob: Sequence[float], cfg: Optional[DecodeConfig] = None) -> List[dict]:
    cfg = cfg or DecodeConfig.tuned()
    p = smooth(rescale(np.asarray(prob, np.float32), cfg.scale_quantile, cfg.scale_floor), cfg.smooth_steps)
    segs, inside, start = [], False, 0
    for i, v in enumerate(p):
        if not inside and v >= cfg.on:
            inside, start = True, i
            while start > 0 and p[start - 1] >= cfg.off:
                start -= 1
        elif inside and v < cfg.off:
            segs.append([start, i])
            inside = False
    if inside:
        segs.append([start, len(p)])
    merged: List[List[int]] = []
    for a, b in segs:
        if merged and (a - merged[-1][1]) * STEP_SECONDS < cfg.merge_gap_seconds:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    out = []
    for a, b in merged:
        if (b - a) * STEP_SECONDS >= cfg.min_seconds:
            out.append({'start': round(a * STEP_SECONDS, 2), 'end': round(b * STEP_SECONDS, 2), 'confidence': float(np.mean(p[a:b]))})
    return out


def _iou(a, b):
    inter = max(0.0, min(a['end'], b['end']) - max(a['start'], b['start']))
    union = max(a['end'], b['end']) - min(a['start'], b['start'])
    return inter / union if union > 0 else 0.0


def match_segments(pred: Sequence[dict], gold: Sequence[dict], min_iou: float = 0.3):
    """Greedy one-to-one matching by IoU, best pairs first."""
    pairs = sorted(((_iou(p, g), i, j) for i, p in enumerate(pred) for j, g in enumerate(gold)), reverse=True)
    used_p, used_g, matches = set(), set(), []
    for iou, i, j in pairs:
        if iou < min_iou:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i)
        used_g.add(j)
        matches.append((i, j, iou))
    return matches


def evaluate(pred: Sequence[dict], gold: Sequence[dict], window: Sequence[float]) -> dict:
    """Segment and time metrics inside `window` (seconds)."""
    t0, t1 = window
    clip = lambda segs: [{'start': max(s['start'], t0), 'end': min(s['end'], t1)} for s in segs if s['end'] > t0 and s['start'] < t1]
    pred, gold = clip(pred), clip(gold)
    matches = match_segments(pred, gold)
    step = 0.05
    grid = np.arange(t0, t1, step) + step / 2
    covered = lambda segs: np.any([(grid >= s['start']) & (grid < s['end']) for s in segs], axis=0) if segs else np.zeros_like(grid, bool)
    pg, gg = covered(pred), covered(gold)
    start_err = [pred[i]['start'] - gold[j]['start'] for i, j, _ in matches]
    end_err = [pred[i]['end'] - gold[j]['end'] for i, j, _ in matches]
    return {
        'gold_points': len(gold),
        'pred_points': len(pred),
        'matched': len(matches),
        'recall': len(matches) / max(len(gold), 1),
        'precision': len(matches) / max(len(pred), 1),
        'play_time_kept': float((pg & gg).sum() / max(gg.sum(), 1)),
        'dead_time_removed': float((~pg & ~gg).sum() / max((~gg).sum(), 1)),
        'start_err_median': float(np.median(np.abs(start_err))) if start_err else None,
        'end_err_median': float(np.median(np.abs(end_err))) if end_err else None,
        'start_err_p90': float(np.percentile(np.abs(start_err), 90)) if start_err else None,
        'end_err_p90': float(np.percentile(np.abs(end_err), 90)) if end_err else None,
    }
