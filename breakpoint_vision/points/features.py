"""Per-step visual features for point segmentation.

Every definition here is deliberately small and fully specified so it can be
reimplemented exactly from this description; `tests/test_points_features.py`
pins the behavior.

Time is cut into fixed steps of `STEP_SECONDS`. Every tracker sample (one per
decoded frame) belongs to the step `floor(t / STEP_SECONDS)`. For each step:

0. `ball_max`: highest ball confidence in the step (0 when there are no samples).
1. `ball_seen`: fraction of the step's samples with confidence above
   `SEEN_CONF`.
2. `ball_speed`: highest ball speed in the step, in frame widths per second,
   measured between consecutive seen samples less than `MAX_GAP` seconds apart,
   clipped to `MAX_SPEED`. The previous sample may belong to the previous step.
3. `ball_x`, 4. `ball_y`: mean position of seen samples, normalized to 0..1 by
   frame width and height (0.5 when nothing was seen).
5, 6, 7. `motion_top`, `motion_mid`, `motion_bottom`: mean over the step's
   frames of the motion score in three horizontal bands of the frame (rows 0-11,
   12-23 and 24-35 of a 64x36 gray thumbnail): mean absolute difference of
   consecutive thumbnails in the band divided by the band's mean brightness in
   the two thumbnails plus 0.01, clipped to `MAX_MOTION`. Bands separate the far
   court, the net area and the near court, so a far-side serve still shows up
   when the ball is too small to track. Dividing by brightness makes the score
   comparable between a floodlit night court and midday sun.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

STEP_SECONDS = 0.1
SEEN_CONF = 0.5
MAX_GAP = 0.12
MAX_SPEED = 3.0
MAX_MOTION = 1.0
FEATURE_NAMES = ('ball_max', 'ball_seen', 'ball_speed', 'ball_x', 'ball_y', 'motion_top', 'motion_mid', 'motion_bottom')
MOTION_BANDS = ((0, 12), (12, 24), (24, 36))
N_FEATURES = len(FEATURE_NAMES)


def motion_score(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    """Exposure-normalized motion between two gray images (0..1 or 0..255)."""
    a = prev_gray.astype(np.float32)
    b = gray.astype(np.float32)
    if a.max() > 1.0 or b.max() > 1.0:
        a, b = a / 255.0, b / 255.0
    diff = float(np.abs(b - a).mean())
    level = float((a.mean() + b.mean()) / 2.0)
    return diff / (level + 0.01)


def band_motion(prev_gray: np.ndarray, gray: np.ndarray) -> np.ndarray:
    """Motion score per horizontal band of two 64x36 gray thumbnails."""
    return np.array([motion_score(prev_gray[a:b], gray[a:b]) for a, b in MOTION_BANDS], np.float32)


def step_features(
    times: Sequence[float],
    xs: Sequence[float],
    ys: Sequence[float],
    confs: Sequence[float],
    width: float,
    height: float,
    duration: float,
    motion: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Return an array of shape (n_steps, N_FEATURES), float32."""
    n_steps = max(1, int(math.ceil(duration / STEP_SECONDS)))
    out = np.zeros((n_steps, N_FEATURES), np.float32)
    out[:, 3] = 0.5
    out[:, 4] = 0.5
    count = np.zeros(n_steps, np.int32)
    seen = np.zeros(n_steps, np.int32)
    sx = np.zeros(n_steps, np.float64)
    sy = np.zeros(n_steps, np.float64)
    mot_sum = np.zeros((n_steps, len(MOTION_BANDS)), np.float64)
    prev_t = prev_x = prev_y = None
    for i, t in enumerate(times):
        k = int(math.floor(t / STEP_SECONDS))
        if k < 0 or k >= n_steps:
            continue
        c = float(confs[i])
        count[k] += 1
        out[k, 0] = max(out[k, 0], c)
        if motion is not None:
            mot_sum[k] += np.minimum(np.asarray(motion[i], np.float64), MAX_MOTION)
        if c > SEEN_CONF:
            seen[k] += 1
            sx[k] += xs[i] / width
            sy[k] += ys[i] / height
            if prev_t is not None and 0 < t - prev_t <= MAX_GAP:
                speed = math.hypot(xs[i] - prev_x, ys[i] - prev_y) / width / (t - prev_t)
                out[k, 2] = max(out[k, 2], min(speed, MAX_SPEED))
            prev_t, prev_x, prev_y = t, xs[i], ys[i]
        else:
            prev_t = prev_x = prev_y = None
    has = count > 0
    out[has, 1] = seen[has] / count[has]
    got = seen > 0
    out[got, 3] = sx[got] / seen[got]
    out[got, 4] = sy[got] / seen[got]
    if motion is not None:
        out[has, 5:8] = mot_sum[has] / count[has][:, None]
    return out


def step_labels(segments: Sequence[dict], duration: float, kinds=('point', 'fault', 'let')) -> np.ndarray:
    """1 for steps whose center lies inside a labeled play segment, else 0."""
    n_steps = max(1, int(math.ceil(duration / STEP_SECONDS)))
    centers = (np.arange(n_steps) + 0.5) * STEP_SECONDS
    y = np.zeros(n_steps, np.float32)
    for s in segments:
        if s.get('kind', 'point') not in kinds:
            continue
        y[(centers >= s['start']) & (centers < s['end'])] = 1.0
    return y
