"""Gate ball tracking on a validated court detection.

This module decides, for a given moment, whether a court is believed to be in
view, and whether a candidate ball detection is plausible given that court. It is
pure numpy so it is unit-testable without torch.

Three checks, in increasing cost:

1. `validate_calibration`: a homography solved from >=4 argmax keypoint peaks
   can be geometrically absurd (self-intersecting, microscopic, off-screen) and
   still "succeed". We require enough keypoints, a small reprojection error, an
   invertible homography, and a convex court quad covering a sane share of the
   frame.
2. Hysteresis over time: N consecutive valid detections to acquire the court,
   and a bounded age after which a stale calibration is dropped rather than
   reused forever. A single dropped detection does not blank the court, but a
   camera that leaves the court does. Streaming use gets this from
   `CourtPresence`; batch use, which knows the whole timeline up front, gets it
   from `confirmed_timeline` plus `TimelineBallGate`'s age bound.
3. `ball_region`: with a court held, a ball is only accepted inside the projected
   court and its run-off, swept upward to cover the airspace a lofted ball
   occupies (a ball above the court plane always projects toward the top of an
   upright frame, while nothing legitimate appears far below the near baseline).
   The region is projected rather than padded in pixels so it follows perspective.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import numpy as np

from .config import CourtGateConfig
from .court import (
    HALF_DOUBLES,
    HALF_LENGTH,
    HALF_SINGLES,
    SERVICE_LINE_FROM_NET,
    doubles_court_polygon,
    singles_court_polygon,
)
from .geometry import apply_homography

Segment = Tuple[float, float, float, float]


def court_world_segments() -> List[Tuple[Tuple[float, float], Tuple[float, float]]]:
    """All court lines in world coordinates (meters): the two boundary
    rectangles, both service lines, the center service line, and the net."""
    segs: List[Tuple[Tuple[float, float], Tuple[float, float]]] = []
    for poly in (doubles_court_polygon(), singles_court_polygon()):
        for i in range(len(poly)):
            segs.append((poly[i], poly[(i + 1) % len(poly)]))
    svc = SERVICE_LINE_FROM_NET
    segs.append(((-HALF_SINGLES, -svc), (HALF_SINGLES, -svc)))
    segs.append(((-HALF_SINGLES, svc), (HALF_SINGLES, svc)))
    segs.append(((0.0, -svc), (0.0, svc)))
    segs.append(((-HALF_SINGLES, 0.0), (HALF_SINGLES, 0.0)))
    return segs


def project_court_segments(homography_img_to_world: np.ndarray, width: int, height: int) -> Optional[List[Segment]]:
    """Project the world court lines into normalized image coords via the inverse
    of the image->world homography. Returns None if the homography is singular."""
    try:
        inv = np.linalg.inv(homography_img_to_world)
    except np.linalg.LinAlgError:
        return None
    out: List[Segment] = []
    for (a, b) in court_world_segments():
        pa = apply_homography(inv, [a])[0]
        pb = apply_homography(inv, [b])[0]
        out.append((float(pa[0] / width), float(pa[1] / height), float(pb[0] / width), float(pb[1] / height)))
    return out


def court_pixel_quad(homography_img_to_world: np.ndarray) -> Optional[np.ndarray]:
    """The four doubles corners in image pixels, or None if unusable."""
    try:
        inv = np.linalg.inv(np.asarray(homography_img_to_world, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    quad = apply_homography(inv, doubles_court_polygon())
    if not np.all(np.isfinite(quad)):
        return None
    return quad


def _polygon_area(quad: np.ndarray) -> float:
    x = quad[:, 0]
    y = quad[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def _is_convex(quad: np.ndarray, eps: float = 1e-6) -> bool:
    n = len(quad)
    signs: List[float] = []
    for i in range(n):
        a = quad[i]
        b = quad[(i + 1) % n]
        c = quad[(i + 2) % n]
        signs.append(float((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
    if any(abs(s) <= eps for s in signs):
        return False
    return all(s > 0 for s in signs) or all(s < 0 for s in signs)


def validate_calibration(
    calib,
    width: int,
    height: int,
    cfg: Optional[CourtGateConfig] = None,
) -> bool:
    """Is this calibration good enough to believe a court is in view?

    Rejects the failure modes that let a court "detection" through on a frame
    with no court in it: too few keypoints, a large reprojection error, and a
    quad that is non-convex, degenerate, absurdly sized, or nowhere near frame.
    """
    cfg = cfg or CourtGateConfig.tuned()
    if calib is None:
        return False
    if width <= 0 or height <= 0:
        return False
    if calib.used_keypoints < cfg.min_keypoints:
        return False
    err = float(calib.reprojection_error_m)
    if not np.isfinite(err) or err > cfg.max_reprojection_error_m:
        return False

    quad = court_pixel_quad(calib.homography)
    if quad is None or not _is_convex(quad):
        return False

    frame_area = float(width) * float(height)
    area_fraction = _polygon_area(quad) / frame_area
    if area_fraction < cfg.min_court_frame_fraction or area_fraction > cfg.max_court_frame_fraction:
        return False

    cx, cy = quad.mean(axis=0)
    if not (-width <= cx <= 2 * width and -height <= cy <= 2 * height):
        return False
    return True


def ball_region(calib, cfg: Optional[CourtGateConfig] = None) -> Optional[np.ndarray]:
    """Pixel polygon a ball detection must fall inside, given a calibration.

    Built by projecting the court plus its run-off, the apron a real venue
    leaves beyond the lines, and then sweeping that polygon upward to cover the
    airspace a lofted ball occupies. Projecting the region rather than padding the
    court's bounding box is what makes it respect perspective: near the camera the
    same run-off is hundreds of pixels wide and can fall outside the frame, while
    at the far baseline it is a thin sliver. A fixed pixel margin gets both ends
    wrong at once.

    Returns None if the region cannot be projected, in which case no ball is
    accepted at all.
    """
    cfg = cfg or CourtGateConfig.tuned()
    try:
        inv = np.linalg.inv(np.asarray(calib.homography, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None

    side = HALF_DOUBLES + max(0.0, cfg.ball_runoff_side_m)
    back = HALF_LENGTH + max(0.0, cfg.ball_runoff_back_m)
    runoff = [(-side, -back), (side, -back), (side, back), (-side, back)]
    ground = apply_homography(inv, runoff)
    if not np.all(np.isfinite(ground)):
        return None

    lift = cfg.ball_up_margin_fraction * float(ground[:, 1].max() - ground[:, 1].min())
    lifted = ground.copy()
    lifted[:, 1] -= lift
    region = np.vstack([ground, lifted])
    hull = _convex_hull(region)
    return hull if len(hull) >= 3 else None


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Monotone-chain hull, counter-clockwise in image coordinates."""
    pts = np.unique(np.asarray(points, dtype=np.float64).round(6), axis=0)
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]
    if len(pts) < 3:
        return pts

    def build(seq):
        chain: List[np.ndarray] = []
        for p in seq:
            while len(chain) >= 2:
                a, b = chain[-2], chain[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) > 0:
                    break
                chain.pop()
            chain.append(p)
        return chain

    lower = build(pts)
    upper = build(pts[::-1])
    return np.array(lower[:-1] + upper[:-1], dtype=np.float64)


def ball_in_region(xy: Tuple[float, float], region: Optional[np.ndarray]) -> bool:
    """Is the point inside (or on) the convex ball region?"""
    if region is None or len(region) < 3:
        return False
    x, y = float(xy[0]), float(xy[1])
    n = len(region)
    sign = 0
    for i in range(n):
        ax, ay = region[i]
        bx, by = region[(i + 1) % n]
        cross = (bx - ax) * (y - ay) - (by - ay) * (x - ax)
        if abs(cross) <= 1e-9:
            continue
        current = 1 if cross > 0 else -1
        if sign == 0:
            sign = current
        elif current != sign:
            return False
    return True


def validated_detector(
    detect: Callable[[np.ndarray], object],
    cfg: Optional[CourtGateConfig] = None,
) -> Callable[[np.ndarray], object]:
    """Wrap a per-frame court detector so implausible calibrations read as None."""
    cfg = cfg or CourtGateConfig.tuned()

    def wrapped(frame: np.ndarray):
        calib = detect(frame)
        if calib is None:
            return None
        h, w = frame.shape[:2]
        return calib if validate_calibration(calib, w, h, cfg) else None

    return wrapped


class CourtPresence:
    """Streaming court-presence state machine.

    Feed it every court detection attempt via `observe`, and every frame that is
    not a court-sampling frame via `tick` (so a stale court still expires between
    samples). `ball_enabled` is the gate the ball model runs behind.
    """

    def __init__(self, cfg: Optional[CourtGateConfig] = None):
        self.cfg = cfg or CourtGateConfig.tuned()
        self._streak = 0
        self._locked = False
        self._last_t: Optional[float] = None
        self._calibration = None
        self._segments: Optional[List[Segment]] = None
        self._region: Optional[np.ndarray] = None

    def observe(self, t_seconds: float, calib, width: int, height: int) -> None:
        if validate_calibration(calib, width, height, self.cfg):
            self._streak += 1
            if self._streak >= self.cfg.min_consecutive_detections:
                segments = project_court_segments(calib.homography, width, height)
                region = ball_region(calib, self.cfg)
                if segments is not None and region is not None:
                    self._locked = True
                    self._calibration = calib
                    self._segments = segments
                    self._region = region
                    self._last_t = t_seconds
        else:
            self._streak = 0
        self.tick(t_seconds)

    def tick(self, t_seconds: float) -> None:
        if not self._locked:
            return
        if self._last_t is None or t_seconds - self._last_t > self.cfg.max_court_age_seconds:
            self.reset()

    def reset(self) -> None:
        self._streak = 0
        self._locked = False
        self._last_t = None
        self._calibration = None
        self._segments = None
        self._region = None

    @property
    def locked(self) -> bool:
        return self._locked and self._segments is not None

    def ball_enabled(self) -> bool:
        return True if not self.cfg.require_court_for_ball else self.locked

    def accepts_ball(self, xy: Tuple[float, float]) -> bool:
        if not self.cfg.require_court_for_ball:
            return True
        if not self.locked or self._region is None:
            return False
        return ball_in_region(xy, self._region)

    def segments(self) -> Optional[List[Segment]]:
        return self._segments if self.locked else None

    def calibration(self):
        return self._calibration if self.locked else None


def confirmed_timeline(timeline, stride: int, min_consecutive: int):
    """Drop isolated detections, keeping only runs of consecutive samples.

    The batch equivalent of `CourtPresence`'s hysteresis. One fluke calibration in
    a span of crowd footage would otherwise open a `max_court_age_seconds`-wide
    window in which the ball tracker runs unsupervised; requiring the same
    consecutive-sample streak `CourtPresence` requires closes it.
    """
    if timeline is None or timeline.is_empty() or min_consecutive <= 1:
        return timeline

    from .infer.court_timeline import CourtTimeline

    gap = max(1, stride)
    indices = timeline.frame_indices
    keep_indices: List[int] = []
    keep_calibs: List[object] = []
    run_start = 0
    for pos in range(len(indices) + 1):
        contiguous = pos < len(indices) and (pos == run_start or indices[pos] - indices[pos - 1] <= gap)
        if contiguous:
            continue
        if pos - run_start >= min_consecutive:
            keep_indices.extend(indices[run_start:pos])
            keep_calibs.extend(timeline.calibrations[run_start:pos])
        run_start = pos
    return CourtTimeline(frame_indices=keep_indices, calibrations=keep_calibs)


def batch_max_age_frames(cfg: CourtGateConfig, fps: float, stride: int) -> int:
    """How long a sampled calibration stays authoritative in batch use.

    A streaming caller cannot see the future, so it trusts a court for
    `max_court_age_seconds`. Batch use re-checks the court every `stride` frames,
    so a missing sample is not missing information: it means we looked there and
    found no court. Carrying a calibration past the next scheduled check would
    extrapolate a court over a span already known to be courtless. One stride is
    exactly "until the next check"; we take the tighter of that and the
    configured age.

    Used with `CourtTimeline.active`, which applies it forward-only.
    """
    age = int(round(cfg.max_court_age_seconds * (fps or 30.0)))
    return max(1, min(age, max(1, stride)))


class TimelineBallGate:
    """Court gate for batch use, backed by a sampled court timeline.

    A batch caller knows the whole timeline up front, so "is there a court at
    frame N" is a bounded-age lookup rather than a streaming state machine. Same
    three checks, same config, same answers, provided the timeline it is handed
    has already been through `validated_detector` and `confirmed_timeline`, which
    is where the plausibility and hysteresis checks live for batch use.
    """

    def __init__(
        self,
        timeline,
        fps: float,
        cfg: Optional[CourtGateConfig] = None,
        stride: int = 1,
    ):
        self.cfg = cfg or CourtGateConfig.tuned()
        self.timeline = timeline
        self.max_age_frames = batch_max_age_frames(self.cfg, fps, stride)
        self._region_cache: dict = {}

    def calibration(self, frame_index: int):
        if self.timeline is None or self.timeline.is_empty():
            return None
        return self.timeline.active(frame_index, self.max_age_frames)

    def allows_frame(self, frame_index: int) -> bool:
        if not self.cfg.require_court_for_ball:
            return True
        return self.calibration(frame_index) is not None

    def accepts(self, frame_index: int, x: float, y: float) -> bool:
        if not self.cfg.require_court_for_ball:
            return True
        calib = self.calibration(frame_index)
        if calib is None:
            return False
        return ball_in_region((x, y), self._region_for(calib))

    def _region_for(self, calib) -> Optional[np.ndarray]:
        key = id(calib)
        if key not in self._region_cache:
            self._region_cache[key] = ball_region(calib, self.cfg)
        return self._region_cache[key]
