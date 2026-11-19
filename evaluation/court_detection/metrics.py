"""Court plane error: how far a predicted court homography is from the labeled one.

For each labeled frame, a 12x24 grid of points covering the doubles court is
projected into the image through the labeled homography (fitted exactly to the
labeled keypoints). The grid points that land inside the 512x288 frame are
mapped back to the court plane through the inverse of the predicted homography,
and the frame's error is the median distance, in cm, between where they land
and where they started. A frame passes when that median is within 15 cm. A frame
with no predicted homography, or whose prediction cannot be inverted, gets an
error of 1e9 cm, so it counts as a failure in every rate and in the median.

The predicted homography is a RANSAC fit (3 px) to the keypoints whose
confidence is at least 0.3. The temporal variant keeps, per keypoint, the median
of its confident detections over the last 12 frames of a video and fits the
court to those (static camera).
"""

from __future__ import annotations

import json
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

from breakpoint_vision.court import CourtKeypoints

W, H = 512, 288
KP = CourtKeypoints()
NAMES = list(KP.names)
WORLD = np.array(KP.ordered_world_array(), np.float64)
GX, GY = np.meshgrid(np.linspace(-5.485, 5.485, 12), np.linspace(-11.885, 11.885, 24))
COURT_GRID = np.c_[GX.ravel(), GY.ravel()]
CONF_THRESHOLD = 0.3
RANSAC_PX = 3.0
PASS_CM = 15.0
FAILURE_CM = 1e9
TEMPORAL_WINDOW = 12
TEMPORAL_MIN = 3
AGGREGATED_THRESHOLD = 0.5


def load_rows(dirs: Iterable[str]) -> List[dict]:
    rows = []
    for d in dirs:
        with open(f'{d}/labels.json') as fh:
            for r in json.load(fh)['images']:
                rows.append(dict(r, path=f"{d}/{r['file']}"))
    return rows


def points_array(row: dict) -> np.ndarray:
    pts = np.full((len(NAMES), 2), np.nan, np.float32)
    for i, n in enumerate(NAMES):
        if n in row['keypoints']:
            pts[i] = row['keypoints'][n]
    return pts


def label_homography(row: dict) -> np.ndarray:
    import cv2

    pts = points_array(row)
    ok = ~np.isnan(pts[:, 0])
    Hm, _ = cv2.findHomography(WORLD[ok], pts[ok].astype(np.float64), 0)
    return Hm


def project(Hm: np.ndarray, pts) -> tuple:
    p = np.c_[np.asarray(pts, np.float64), np.ones(len(pts))] @ Hm.T
    return p[:, :2] / p[:, 2:3], p[:, 2]


def plane_error_cm(Hm: Optional[np.ndarray], row: dict) -> float:
    img, w = project(label_homography(row), COURT_GRID)
    vis = (w > 0) & (img[:, 0] >= 0) & (img[:, 0] < W) & (img[:, 1] >= 0) & (img[:, 1] < H)
    if Hm is None or vis.sum() == 0:
        return FAILURE_CM
    try:
        back, _ = project(np.linalg.inv(Hm), img[vis])
    except np.linalg.LinAlgError:
        return FAILURE_CM
    return float(np.median(np.linalg.norm(back - COURT_GRID[vis], axis=1)) * 100)


def points_only_fit(pk: np.ndarray, thr: float = CONF_THRESHOLD) -> Optional[np.ndarray]:
    import cv2

    ok = pk[:14, 2] >= thr
    if ok.sum() < 4:
        return None
    Hm, _ = cv2.findHomography(WORLD[ok], pk[:14][ok, :2].astype(np.float64), cv2.RANSAC, RANSAC_PX)
    return Hm


def peaks(hm: np.ndarray, r: int = 2) -> np.ndarray:
    """Per channel: argmax refined by the centroid of the (2r+1)^2 window inside the frame, and its value."""
    k, h, w = hm.shape
    flat = hm.reshape(k, -1)
    idx = flat.argmax(1)
    conf = flat[np.arange(k), idx]
    ys, xs = idx // w, idx % w
    out = np.zeros((k, 3), np.float32)
    for i in range(k):
        y0, y1, x0, x1 = max(0, ys[i] - r), min(h, ys[i] + r + 1), max(0, xs[i] - r), min(w, xs[i] + r + 1)
        win = hm[i, y0:y1, x0:x1]
        gy, gx = np.mgrid[y0:y1, x0:x1]
        s = win.sum()
        out[i] = (float((win * gx).sum() / s) if s > 1e-9 else xs[i], float((win * gy).sum() / s) if s > 1e-9 else ys[i], conf[i])
    return out


def in_frame(pts: np.ndarray) -> np.ndarray:
    out = pts.copy()
    bad = (pts[:, 0] < 0) | (pts[:, 0] > W - 1) | (pts[:, 1] < 0) | (pts[:, 1] > H - 1)
    out[bad] = np.nan
    return out


def full_projection(row: dict) -> np.ndarray:
    return project(label_homography(row), WORLD)[0].astype(np.float32)


def calib_error(pk: np.ndarray, truth: np.ndarray, thr: float = CONF_THRESHOLD):
    """Largest pixel error of the 14 projected keypoints that lie in frame, and the confident count."""
    ok = pk[:14, 2] >= thr
    Hm = points_only_fit(pk, thr)
    if Hm is None:
        return None, int(ok.sum())
    proj = project(Hm, WORLD)[0]
    vis = (truth[:, 0] >= 0) & (truth[:, 0] < W) & (truth[:, 1] >= 0) & (truth[:, 1] < H)
    return float(np.max(np.linalg.norm(proj[vis] - truth[vis], axis=1))), int(ok.sum())


class TemporalMedian:
    """Running per-keypoint median of confident detections over the last `window` frames."""

    def __init__(self, window: int = TEMPORAL_WINDOW, min_frames: int = TEMPORAL_MIN, thr: float = CONF_THRESHOLD):
        self.window, self.min_frames, self.thr = window, min_frames, thr
        self.hist: List[np.ndarray] = []

    def add(self, pk: np.ndarray) -> Optional[np.ndarray]:
        """Add one frame's 14 keypoints; return the aggregate once `min_frames` are held."""
        self.hist.append(np.asarray(pk)[:14])
        self.hist = self.hist[-self.window:]
        if len(self.hist) < self.min_frames:
            return None
        stack = np.stack(self.hist)
        agg = np.zeros((14, 3), np.float32)
        for i in range(14):
            good = stack[:, i, 2] >= self.thr
            if good.sum() >= max(2, len(self.hist) // 3):
                agg[i, :2] = np.median(stack[good, i, :2], 0)
                agg[i, 2] = 1.0
        return agg


def plane_summary(errors: Sequence[float]) -> Dict[str, Optional[float]]:
    e = np.asarray(errors, np.float64)
    if len(e) == 0:
        return {'frames': 0, 'within_15cm': None, 'within_30cm': None, 'p50_cm': None, 'failures': 0}
    return {
        'frames': int(len(e)),
        'within_15cm': round(float((e <= PASS_CM).mean()), 3),
        'within_30cm': round(float((e <= 30).mean()), 3),
        'p50_cm': round(float(np.median(e)), 1),
        'failures': int((e >= FAILURE_CM).sum()),
    }
