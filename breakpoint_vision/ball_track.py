"""Turn a per-frame ball heatmap into a trajectory.

Taking the argmax of each heatmap and accepting it above a fixed confidence
discards two things the network already provides. When the argmax is wrong (a
line marking, a shoe, a distant ball), the real ball is often still a local
maximum, just not the tallest one. And a tennis ball travels a smooth,
near-ballistic path, so its recent positions say a great deal about where it is
now: a tracker that predicts forward can accept a faint peak because it continues
the trajectory, while rejecting a strong peak that teleports.

So: take the top-N local maxima, refine each to sub-pixel by local centroid, and
let a small constant-velocity tracker choose between them. A track must be
confirmed by consecutive strong detections before its prediction is trusted to
admit weak peaks, so a single fluke cannot open a low-threshold window.

Distances are fractions of the model input width, not pixels, so the same config
applies at any input resolution. The same tracker also runs on device, and every
tie-break is deterministic so the two implementations make identical decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .params import tuned


@dataclass
class BallTrackerConfig:
    """Candidate extraction and tracking parameters.

    `seed_confidence` is the bar for a peak trusted on its own. `keep_confidence`
    is the lower bar for a peak that the track's own prediction vouches for.
    `gate_fraction` is how far from the predicted position such a peak may be, as
    a fraction of input width; `coast_frames` is how long a track survives with
    nothing accepted, and `gate_growth_fraction` widens the gate for each coasted
    frame because uncertainty grows while blind. The thresholds and gates are
    tuned operating values: build with `BallTrackerConfig.tuned()` or pass them
    explicitly.
    """

    min_candidate_confidence: float
    seed_confidence: float
    keep_confidence: float
    gate_fraction: float
    gate_growth_fraction: float
    coast_frames: int
    confirm_detections: int
    max_candidates: int = 4
    peak_separation_px: int = 5
    subpixel_radius_px: int = 2

    @classmethod
    def tuned(cls, **overrides) -> "BallTrackerConfig":
        return cls(**{**tuned("ball_tracker"), **overrides})


@dataclass
class Candidate:
    x: float
    y: float
    confidence: float


def heatmap_candidates(
    heatmap: np.ndarray,
    cfg: Optional[BallTrackerConfig] = None,
) -> List[Candidate]:
    """Top local maxima of a (H, W) heatmap, sub-pixel refined, strongest first.

    Local maxima rather than a global argmax with suppression: a true peak next to
    a taller one survives (see module docstring).

    `_local_maxima` returns one maximum per tied plateau, so a saturated blob cannot
    spend every candidate slot on itself and hide a genuine second peak. Ordering
    is `(-confidence, y, x)`: ties are broken by position so the same heatmap
    always yields the same candidate list rather than depending on scan order.
    """
    cfg = cfg or BallTrackerConfig.tuned()
    hm = np.asarray(heatmap, dtype=np.float32)
    if hm.ndim != 2:
        raise ValueError(f"expected a 2-D heatmap, got shape {hm.shape}")

    peaks = _local_maxima(hm, cfg.peak_separation_px, cfg.min_candidate_confidence)
    peaks.sort(key=lambda p: (-p[2], p[1], p[0]))

    keep = max(1, cfg.max_candidates)
    out: List[Candidate] = []
    for (x, y, conf) in peaks:
        if len(out) >= keep:
            break
        rx, ry = _subpixel(hm, x, y, cfg.subpixel_radius_px)
        out.append(Candidate(x=rx, y=ry, confidence=conf))
    return out


def _local_maxima(hm: np.ndarray, separation: int, min_conf: float) -> List[Tuple[int, int, float]]:
    """Maxima of `hm`, one per tied plateau, at least as tall as every neighbor.

    The `>=` test is what lets a peak beside a taller one survive, but it also
    reports every pixel of a flat region as a maximum. A confident blob saturates
    once sigmoid rounds it to 1.0, and those pixels would fill every candidate slot
    with the same ball, so each connected group of equal-valued maxima is
    collapsed to the pixel nearest its centroid.
    """
    h, w = hm.shape
    k = max(1, int(separation))
    padded = np.pad(hm, k, mode="constant", constant_values=-np.inf)
    window_max = np.full_like(hm, -np.inf)
    for dy in range(-k, k + 1):
        for dx in range(-k, k + 1):
            shifted = padded[k + dy : k + dy + h, k + dx : k + dx + w]
            np.maximum(window_max, shifted, out=window_max)
    ys, xs = np.nonzero((hm >= window_max) & (hm >= min_conf))
    return _collapse_plateaus([(int(x), int(y), float(hm[y, x])) for x, y in zip(xs, ys)])


def _collapse_plateaus(peaks: List[Tuple[int, int, float]]) -> List[Tuple[int, int, float]]:
    """One representative per 8-connected group of equal-valued maxima.

    The representative is the group pixel closest to the group centroid, with ties
    broken by (y, x) rather than by traversal order: a plateau that is symmetric
    about its centroid has two equally-close pixels, and picking the first-visited
    one would make the answer depend on scan order.
    """
    if len(peaks) < 2:
        return peaks

    by_point = {(x, y): conf for (x, y, conf) in peaks}
    seen = set()
    merged: List[Tuple[int, int, float]] = []
    for (x, y, conf) in peaks:
        if (x, y) in seen:
            continue
        group = [(x, y)]
        seen.add((x, y))
        queue = [(x, y)]
        while queue:
            cx, cy = queue.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (cx + dx, cy + dy)
                    if neighbor in seen or neighbor not in by_point:
                        continue
                    if by_point[neighbor] != conf:
                        continue
                    seen.add(neighbor)
                    group.append(neighbor)
                    queue.append(neighbor)
        mx = sum(p[0] for p in group) / len(group)
        my = sum(p[1] for p in group) / len(group)
        rep = min(group, key=lambda p: ((p[0] - mx) ** 2 + (p[1] - my) ** 2, p[1], p[0]))
        merged.append((rep[0], rep[1], conf))
    return merged


def _subpixel(hm: np.ndarray, x: int, y: int, radius: int) -> Tuple[float, float]:
    """Intensity-weighted centroid of the patch around an integer peak.

    The heatmap blob is a Gaussian several pixels wide, so its center of mass is a
    better estimate of the ball than the single tallest pixel.

    Out-of-bounds window positions are clamped to the edge rather than dropped:
    dropping them leaves an asymmetric window whose centroid is pulled inward for a
    ball at the frame edge. `keypoint_peaks` clamps the same way.
    """
    h, w = hm.shape
    r = max(0, int(radius))
    ys = np.clip(np.arange(y - r, y + r + 1), 0, h - 1)
    xs = np.clip(np.arange(x - r, x + r + 1), 0, w - 1)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    patch = hm[gy, gx]
    total = float(patch.sum())
    if total <= 1e-9:
        return float(x), float(y)
    return float((patch * gx).sum() / total), float((patch * gy).sum() / total)


class BallTracker:
    """Constant-velocity association over per-frame heatmap candidates.

    Feed `update` the candidates for each frame in order; it returns the accepted
    ball position in heatmap coordinates, or None for "no ball this frame". Call
    `reset` whenever continuity breaks (the court gate dropping, a seek, a new
    clip), because a stale prediction is worse than no prediction.
    """

    def __init__(self, input_width: int, cfg: Optional[BallTrackerConfig] = None):
        self.cfg = cfg or BallTrackerConfig.tuned()
        self.input_width = float(max(1, input_width))
        self.reset()

    def reset(self) -> None:
        self._position: Optional[Tuple[float, float]] = None
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self._missed = 0
        self._confirmed = 0
        self._confidence = 0.0

    @property
    def confirmed(self) -> bool:
        return self._confirmed >= self.cfg.confirm_detections

    @property
    def confidence(self) -> float:
        """Confidence of the candidate accepted by the last successful `update`.

        This is the accepted peak's own score, which is not always the strongest
        peak in the frame: for a faint peak vouched for by the track's prediction,
        reporting the leader's confidence instead would overstate it.
        """
        return self._confidence

    def update(self, candidates: Sequence[Candidate]) -> Optional[Tuple[float, float]]:
        cfg = self.cfg
        if not candidates:
            return self._miss()

        if self._position is not None and self.confirmed:
            predicted = self._predict()
            budget = (cfg.gate_fraction + cfg.gate_growth_fraction * self._missed) * self.input_width
            best: Optional[Tuple[float, Candidate]] = None
            for cand in candidates:
                if cand.confidence < cfg.keep_confidence:
                    continue
                dist = float(np.hypot(cand.x - predicted[0], cand.y - predicted[1]))
                if dist <= budget and (best is None or dist < best[0]):
                    best = (dist, cand)
            if best is not None:
                return self._accept(best[1], strong=best[1].confidence >= cfg.seed_confidence)

        leader = candidates[0]
        if leader.confidence >= cfg.seed_confidence:
            return self._accept(leader, strong=True)
        return self._miss()

    def _predict(self) -> Tuple[float, float]:
        steps = self._missed + 1
        return (
            self._position[0] + self._velocity[0] * steps,
            self._position[1] + self._velocity[1] * steps,
        )

    def _accept(self, cand: Candidate, strong: bool) -> Tuple[float, float]:
        if self._position is not None:
            steps = self._missed + 1
            self._velocity = (
                (cand.x - self._position[0]) / steps,
                (cand.y - self._position[1]) / steps,
            )
        self._position = (cand.x, cand.y)
        self._missed = 0
        self._confidence = cand.confidence
        self._confirmed = self._confirmed + 1 if (strong or self.confirmed) else 0
        return self._position

    def _miss(self) -> None:
        self._missed += 1
        if self._missed > self.cfg.coast_frames:
            self.reset()
        return None
