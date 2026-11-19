"""Planar geometry: homography estimation and reprojection error.

Pure numpy so it is testable without OpenCV or torch.

A homography H (3x3) maps image pixels to the world court plane defined in
`court`. We solve it with the normalized Direct Linear Transform (DLT) from >=4
point correspondences. Normalization (Hartley) matters a lot for numerical
stability, so it is not optional here.
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


def _to_array(points: Sequence[Point]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"expected an (N,2) array of points, got shape {arr.shape}")
    return arr


def _normalize(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley normalization: center at origin, mean distance sqrt(2)."""
    centroid = points.mean(axis=0)
    centered = points - centroid
    mean_dist = np.sqrt((centered**2).sum(axis=1)).mean()
    if mean_dist < 1e-12:
        scale = 1.0
    else:
        scale = np.sqrt(2.0) / mean_dist
    t = np.array(
        [
            [scale, 0.0, -scale * centroid[0]],
            [0.0, scale, -scale * centroid[1]],
            [0.0, 0.0, 1.0],
        ]
    )
    homo = np.hstack([points, np.ones((points.shape[0], 1))])
    normed = (t @ homo.T).T
    return normed[:, :2], t


def estimate_homography(src: Sequence[Point], dst: Sequence[Point]) -> np.ndarray:
    """Homography mapping src points to dst points (both (N,2), N>=4)."""
    s = _to_array(src)
    d = _to_array(dst)
    if s.shape[0] != d.shape[0]:
        raise ValueError("src and dst must have the same number of points")
    if s.shape[0] < 4:
        raise ValueError("need at least 4 correspondences for a homography")

    s_n, t_s = _normalize(s)
    d_n, t_d = _normalize(d)

    rows: List[np.ndarray] = []
    for (x, y), (u, v) in zip(s_n, d_n):
        rows.append([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u])
        rows.append([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v])
    a = np.asarray(rows, dtype=np.float64)

    _, _, vh = np.linalg.svd(a)
    h_n = vh[-1].reshape(3, 3)

    # Undo normalization: H = T_d^-1 @ H_n @ T_s
    h = np.linalg.inv(t_d) @ h_n @ t_s
    if abs(h[2, 2]) > 1e-12:
        h = h / h[2, 2]
    return h


def estimate_homography_robust(
    src: Sequence[Point],
    dst: Sequence[Point],
    threshold_m: float = 0.30,
    min_inliers: int = 6,
) -> Tuple[np.ndarray, List[int]]:
    """Homography plus the inlier set, fitted so one bad keypoint cannot poison it.

    `estimate_homography` is a least-squares fit, so every correspondence pulls on
    the result in proportion to its error, and a keypoint model does not fail
    gently: one misplaced keypoint spreads error onto every other point while the
    fit's own reprojection error rises only modestly.

    Two stages, so the common case stays cheap. First `_trim_worst`: fit on
    everything, drop the single worst residual, refit, repeat. That is O(n) solves
    and settles the clean and one-outlier cases. Its weakness is that with several
    gross outliers the initial fit is dragged so far that the true inliers look
    like the outliers.

    So when trimming cannot keep at least `n - 1` points, fall back to exhaustive
    4-point RANSAC: every C(n,4) subset (at most 1001 for 14 keypoints), scored by
    consensus within `threshold_m` on the court plane, refit on the largest
    consensus set. Exhaustive rather than sampled because the subset count is tiny
    and determinism matters: the same frame must calibrate identically every run.

    Falls back to the plain fit over all points when no subset reaches
    `min_inliers`; the caller still sees a reprojection error, and `court_gate`
    rejects it if it is implausible.
    """
    s = _to_array(src)
    d = _to_array(dst)
    if s.shape[0] != d.shape[0]:
        raise ValueError("src and dst must have the same number of points")
    n = s.shape[0]
    if n < 4:
        raise ValueError("need at least 4 correspondences for a homography")

    all_idx = list(range(n))
    if n == 4:
        return estimate_homography(s, d), all_idx

    trimmed, trim_inliers = _trim_worst(s, d, threshold_m, max(4, min_inliers))
    if len(trim_inliers) >= n - 1:
        return trimmed, trim_inliers

    best_inliers: List[int] = []
    for subset in combinations(all_idx, 4):
        idx = list(subset)
        try:
            candidate = estimate_homography(s[idx], d[idx])
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not np.all(np.isfinite(candidate)):
            continue
        mapped = apply_homography(candidate, s)
        errors = np.sqrt(((mapped - d) ** 2).sum(axis=1))
        count = int((errors <= threshold_m).sum())
        if count > len(best_inliers):
            best_inliers = [i for i in all_idx if errors[i] <= threshold_m]
            if count == n:
                break

    if len(best_inliers) < max(4, min_inliers):
        return estimate_homography(s, d), all_idx

    refit = estimate_homography(s[best_inliers], d[best_inliers])
    return refit, best_inliers


def _trim_worst(
    s: np.ndarray,
    d: np.ndarray,
    threshold_m: float,
    min_keep: int,
) -> Tuple[np.ndarray, List[int]]:
    """Iteratively drop the single worst-fitting correspondence and refit.

    Returns as soon as every kept point is within `threshold_m`, or once dropping
    another would go below `min_keep`. Cheap (O(n) solves) and enough for the clean
    and single-outlier cases; `estimate_homography_robust` escalates when it is not.
    """
    idx = list(range(s.shape[0]))
    while True:
        h = estimate_homography(s[idx], d[idx])
        errors = np.sqrt(((apply_homography(h, s[idx]) - d[idx]) ** 2).sum(axis=1))
        worst = int(np.argmax(errors))
        if errors[worst] <= threshold_m or len(idx) <= min_keep:
            return h, idx
        idx = idx[:worst] + idx[worst + 1 :]


def apply_homography(h: np.ndarray, points: Sequence[Point]) -> np.ndarray:
    """Map (N,2) points through homography H, returning (N,2)."""
    pts = _to_array(points)
    homo = np.hstack([pts, np.ones((pts.shape[0], 1))])
    mapped = (h @ homo.T).T
    w = mapped[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return mapped[:, :2] / w


def reprojection_error(h: np.ndarray, src: Sequence[Point], dst: Sequence[Point]) -> float:
    """Mean Euclidean error (in dst units) of mapping src->dst through H."""
    mapped = apply_homography(h, src)
    d = _to_array(dst)
    return float(np.sqrt(((mapped - d) ** 2).sum(axis=1)).mean())
