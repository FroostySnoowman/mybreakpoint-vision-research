"""Score decoded play segments as a points-only playback would use them.

Playback pads every segment (lead-in before, tail after) and joins padded
segments whose gap is under `merge_gap_seconds`. A real point counts as kept
whole when at least `WHOLE_FRACTION` of it is played.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from .decode import DecodeConfig, decode, evaluate

GRID_SECONDS = 0.05
WHOLE_FRACTION = 0.95
PLAY_KINDS = ('point', 'fault', 'let')


@dataclass
class Playback:
    lead_in_seconds: float = 1.5
    tail_seconds: float = 2.0
    merge_gap_seconds: float = 2.5


def playback_plan(segments: Sequence[dict], playback: Playback = Playback()) -> List[List[float]]:
    out: List[List[float]] = []
    for s in sorted(segments, key=lambda s: s['start']):
        a, b = max(0.0, s['start'] - playback.lead_in_seconds), s['end'] + playback.tail_seconds
        if out and a - out[-1][1] < playback.merge_gap_seconds:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _covered(ranges: Sequence[Sequence[float]], grid: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(grid, bool)
    for a, b in ranges:
        mask |= (grid >= a) & (grid < b)
    return mask


def product_metrics(pred: Sequence[dict], gold: Sequence[dict], window: Sequence[float], playback: Playback = Playback()) -> dict:
    """points_kept_whole, play_time_kept and dead_time_removed inside `window`."""
    t0, t1 = window
    grid = np.arange(t0, t1, GRID_SECONDS) + GRID_SECONDS / 2
    kept = _covered(playback_plan(pred, playback), grid)
    play = _covered([(g['start'], g['end']) for g in gold], grid)
    whole = []
    for g in gold:
        m = (grid >= max(g['start'], t0)) & (grid < min(g['end'], t1))
        if m.any():
            whole.append(kept[m].mean() >= WHOLE_FRACTION)
    return {
        'points_kept_whole': float(np.mean(whole)) if whole else None,
        'play_time_kept': float((kept & play).sum() / max(play.sum(), 1)),
        'dead_time_removed': float((~kept & ~play).sum() / max((~play).sum(), 1)),
        'n_points': len(whole),
    }


def play_segments(label: Mapping) -> List[dict]:
    return [s for s in label['segments'] if s.get('kind', 'point') in PLAY_KINDS]


def score_videos(probs: Mapping[str, np.ndarray], golds: Mapping[str, Tuple[Sequence[dict], Sequence[float]]], cfg: DecodeConfig, playback: Playback = Playback()) -> Dict[str, dict]:
    rows = {}
    for vid, p in probs.items():
        gold, window = golds[vid]
        pred = decode(p, cfg)
        rows[vid] = {**evaluate(pred, gold, window), **product_metrics(pred, gold, window, playback)}
    return rows


def summarize(rows: Mapping[str, dict]) -> dict:
    """Per-video means, plus points kept whole pooled over every point."""
    keys = ('points_kept_whole', 'play_time_kept', 'dead_time_removed', 'recall', 'precision', 'start_err_median', 'end_err_median')
    out = {}
    for k in keys:
        vals = [r[k] for r in rows.values() if r.get(k) is not None]
        out[k] = round(float(np.mean(vals)), 3) if vals else None
    total = sum(r['n_points'] for r in rows.values())
    out['points_kept_whole_pooled'] = round(sum((r['points_kept_whole'] or 0) * r['n_points'] for r in rows.values()) / max(total, 1), 3)
    return out
