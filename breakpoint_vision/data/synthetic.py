"""Synthetic tennis clips for smoke-testing the training path end-to-end.

This does not replace real training data. It lets a smoke run prove the
model/dataset/optimizer/device/checkpoint path works with no downloaded data, and
gives trajectory code something realistic to run against in tests.

A synthetic sample is a small stack of frames with a bright blob (the "ball")
moving on a noisy background, plus the ground-truth ball position per frame.
Pure numpy; torch is only used by the Dataset wrapper that consumes these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class SyntheticClip:
    frames: np.ndarray  # (T, H, W, 3) uint8
    positions: List[Optional[Tuple[float, float]]]  # per-frame (x, y) or None


def make_clip(
    length: int = 8,
    height: int = 288,
    width: int = 512,
    ball_radius: int = 5,
    seed: int = 0,
    invisible_prob: float = 0.05,
) -> SyntheticClip:
    rng = np.random.default_rng(seed)

    bg_color = rng.integers(30, 90, size=3)
    frames = np.empty((length, height, width, 3), dtype=np.uint8)

    # Ball follows a parabolic arc with constant horizontal velocity.
    x0 = rng.uniform(0.1, 0.4) * width
    vx = rng.uniform(0.05, 0.12) * width
    y0 = rng.uniform(0.2, 0.5) * height
    vy = rng.uniform(-0.15, -0.05) * height
    g = rng.uniform(0.02, 0.05) * height

    positions: List[Optional[Tuple[float, float]]] = []
    for t in range(length):
        frame = np.tile(bg_color.astype(np.uint8), (height, width, 1))
        frame = frame + rng.integers(-10, 10, size=(height, width, 3), dtype=np.int16)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

        tt = t / max(1, length - 1) * 4.0
        x = x0 + vx * tt
        y = y0 + vy * tt + g * tt * tt

        visible = (0 <= x < width) and (0 <= y < height) and (rng.random() > invisible_prob)
        if visible:
            _draw_blob(frame, x, y, ball_radius, rng)
            positions.append((float(x), float(y)))
        else:
            positions.append(None)

        frames[t] = frame

    return SyntheticClip(frames=frames, positions=positions)


def _draw_blob(frame: np.ndarray, cx: float, cy: float, radius: int, rng) -> None:
    h, w = frame.shape[:2]
    # Motion blur: stretch the blob slightly along x to mimic a fast ball.
    stretch = rng.uniform(1.0, 2.5)
    x0, x1 = max(0, int(cx - radius * stretch)), min(w, int(cx + radius * stretch) + 1)
    y0, y1 = max(0, int(cy - radius)), min(h, int(cy + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return
    ys, xs = np.mgrid[y0:y1, x0:x1]
    d = ((xs - cx) / stretch) ** 2 + (ys - cy) ** 2
    mask = np.exp(-d / (2 * (radius / 2.0) ** 2))
    for c in range(3):
        base = frame[y0:y1, x0:x1, c].astype(np.float32)
        frame[y0:y1, x0:x1, c] = np.clip(base + mask * (230 - base), 0, 255).astype(np.uint8)


def clip_to_trajectory(clip: SyntheticClip):
    """Ground-truth trajectory as (frame, x, y, visible) tuples, the form `track_ball` emits."""
    traj = []
    for i, pos in enumerate(clip.positions):
        if pos is None:
            traj.append((i, 0.0, 0.0, False))
        else:
            traj.append((i, pos[0], pos[1], True))
    return traj
