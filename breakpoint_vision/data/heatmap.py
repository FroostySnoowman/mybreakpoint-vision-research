"""Gaussian heatmap rendering, shared by ball and court targets.

Pure numpy so it can be unit-tested without torch.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def render_gaussian(height: int, width: int, center: Optional[Tuple[float, float]], sigma: float) -> np.ndarray:
    """A (H, W) float32 heatmap with a Gaussian blob at center=(x, y).

    center=None (ball not visible) yields an all-zero heatmap, which is the
    correct target: the network should predict "nothing here".
    """
    hm = np.zeros((height, width), dtype=np.float32)
    if center is None:
        return hm
    cx, cy = center
    if not (0 <= cx < width and 0 <= cy < height):
        return hm

    radius = int(3 * sigma)
    x0, x1 = max(0, int(cx - radius)), min(width, int(cx + radius) + 1)
    y0, y1 = max(0, int(cy - radius)), min(height, int(cy + radius) + 1)
    if x0 >= x1 or y0 >= y1:
        return hm

    ys, xs = np.mgrid[y0:y1, x0:x1]
    g = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma**2))
    hm[y0:y1, x0:x1] = g.astype(np.float32)
    return hm


def render_multi(height: int, width: int, centers, sigma: float) -> np.ndarray:
    """(K, H, W) stack, one Gaussian channel per (x,y) center (None allowed)."""
    return np.stack([render_gaussian(height, width, c, sigma) for c in centers], axis=0)
