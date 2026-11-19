"""Ball tracking metrics against human-checked ball positions.

A labeled frame is one of three kinds. `visible: true` frames have a ball
position: the tracker's output counts as a hit within 8 px (and, separately,
within 3 px) in 512x288 model pixels, a miss when it is farther, and "no output"
when the tracker returns nothing. `visible: false` and `"not_in_play"` frames are
negatives: any output is a false alarm. Every other label (`"uncertain"`) is
excluded from both.

The tracker is `breakpoint_vision.ball_track.BallTracker`, fed candidate lists
frame by frame; its output at the labeled frame is what gets scored.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from breakpoint_vision.ball_track import BallTracker, BallTrackerConfig, Candidate

W, H = 512, 288
HIT_PX = 8.0
TIGHT_PX = 3.0
NEGATIVE_LABELS = (False, 'not_in_play')
COUNT_KEYS = ('vis', 'hit3', 'hit8', 'none', 'neg', 'fa', 'uncertain')


def tracker_config(path: Optional[Path] = None) -> BallTrackerConfig:
    """Explicit values from a JSON file, else `BallTrackerConfig.tuned()`."""
    if path is None:
        return BallTrackerConfig.tuned()
    with open(path) as fh:
        return BallTrackerConfig(**json.load(fh))


def new_counts() -> Dict[str, int]:
    return {k: 0 for k in COUNT_KEYS}


def count_frame(g: Dict[str, int], label: Mapping, pos: Optional[Sequence[float]]) -> None:
    """Add one labeled frame and the tracker's output to the counts `g`."""
    visible = label['visible']
    if visible is True:
        g['vis'] += 1
        if pos is None:
            g['none'] += 1
        else:
            d = float(np.hypot(pos[0] - label['x'], pos[1] - label['y']))
            g['hit3'] += int(d <= TIGHT_PX)
            g['hit8'] += int(d <= HIT_PX)
    elif visible in NEGATIVE_LABELS:
        g['neg'] += 1
        g['fa'] += int(pos is not None)
    else:
        g['uncertain'] += 1


def is_hit(label: Mapping, pos: Optional[Sequence[float]], tol: float = HIT_PX) -> bool:
    return pos is not None and math.hypot(pos[0] - label['x'], pos[1] - label['y']) <= tol


def rates(g: Mapping[str, int]) -> dict:
    v = max(g['vis'], 1)
    return {
        **{k: int(g[k]) for k in COUNT_KEYS},
        'track@8px': round(g['hit8'] / v, 4),
        'track@3px': round(g['hit3'] / v, 4),
        'no_output': round(g['none'] / v, 4),
        'wrong': round((g['vis'] - g['hit8'] - g['none']) / v, 4),
        'false_alarm': f"{g['fa']}/{g['neg']}",
    }


def summarize_counts(per_video: Mapping[str, Mapping[str, int]], groups: Mapping[str, str]) -> dict:
    """Totals, per-group and per-video rates from per-video counts."""
    total, by_group = new_counts(), {}
    for vid, g in per_video.items():
        grp = by_group.setdefault(groups.get(vid, 'other'), new_counts())
        for k in COUNT_KEYS:
            total[k] += g[k]
            grp[k] += g[k]
    return {
        'total': rates(total),
        'by_group': {k: rates(v) for k, v in sorted(by_group.items())},
        'per_video': {k: rates(v) for k, v in sorted(per_video.items())},
    }


def to_candidates(cands: Iterable[Sequence[float]]) -> List[Candidate]:
    return [Candidate(float(c[0]), float(c[1]), float(c[2])) for c in cands]


def track_last(frames: Iterable[Iterable[Sequence[float]]], cfg: BallTrackerConfig, width: int = W) -> Optional[Tuple[float, float]]:
    """Run a fresh tracker over per-frame candidate lists; return its last output."""
    tracker = BallTracker(width, cfg)
    pos = None
    for cands in frames:
        pos = tracker.update(to_candidates(cands))
    return pos


def load_labels(data: Path) -> Dict[str, dict]:
    return {Path(p).stem: json.load(open(p)) for p in sorted(glob.glob(f'{data}/ball_gt/*.json'))}


def load_cached_rows(data: Path) -> List[Tuple[str, dict, dict]]:
    """(video, label frame, cached row) for every cached labeled frame."""
    out = []
    for p in sorted(glob.glob(f'{data}/ball_gt_cache/*.json')):
        vid = Path(p).stem
        gt = {f['k']: f for f in json.load(open(f'{data}/ball_gt/{vid}.json'))['frames']}
        for row in json.load(open(p))['frames']:
            out.append((vid, gt[row['k']], row))
    return out


def wilson_interval(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def exact_mcnemar_p(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from the two discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n)


def paired_counts(a: Mapping, b: Mapping) -> Tuple[int, int]:
    """Discordant pairs over keys present in both: (only a hit, only b hit)."""
    only_a = sum(1 for k in a if k in b and a[k] and not b[k])
    only_b = sum(1 for k in a if k in b and b[k] and not a[k])
    return only_a, only_b
