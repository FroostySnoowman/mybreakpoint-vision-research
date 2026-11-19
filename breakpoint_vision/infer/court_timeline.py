"""Track the court through a moving-camera video.

A handheld camera pans, so the court homography is a function of time, not a
constant. We sample it: run the court keypoint model every `stride` frames and
store each successful calibration against its frame index. Image points are then
mapped through the calibration in force at their frame, via `active()`.

Sampling (rather than every frame) is the cost/accuracy knob: the court moves
slowly relative to the ball, so re-detecting a few times per second is plenty and
keeps compute low. `active()` is a small binary search over the samples.

This module is import-light (numpy only) so the timeline data structure and its
lookup are unit-testable without torch; the model call is injected as `detect`.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from .court_infer import CourtCalibration


@dataclass
class CourtTimeline:
    frame_indices: List[int]
    calibrations: List[CourtCalibration]

    def is_empty(self) -> bool:
        return not self.calibrations

    def active(self, frame_index: int, max_age_frames: int) -> Optional[CourtCalibration]:
        """The court in view at `frame_index`, or None if we cannot claim one.

        Forward-only. A sample is a check: it says what was true at its own frame,
        and stays authoritative until the next scheduled check contradicts it.
        Reaching backwards instead, as a symmetric nearest-in-time window does,
        hands a court to frames that an earlier check already found courtless.

        `max_age_frames` is how long that authority lasts; see
        `court_gate.batch_max_age_frames`.
        """
        if not self.calibrations:
            return None
        pos = bisect.bisect_right(self.frame_indices, frame_index) - 1
        if pos < 0:
            return None
        if frame_index - self.frame_indices[pos] >= max_age_frames:
            return None
        return self.calibrations[pos]

    def coverage(self) -> float:
        """Fraction of sampled frames that produced a valid calibration."""
        total = getattr(self, "_sampled", None)
        if not total:
            return 1.0 if self.calibrations else 0.0
        return len(self.calibrations) / total

    def mean_position_sigma(self) -> Optional[float]:
        if not self.calibrations:
            return None
        return float(np.mean([c.position_sigma_m for c in self.calibrations]))


def build_court_timeline(
    frame_source: Callable[[], "Iterable[Tuple[int, np.ndarray]]"],
    detect: Callable[[np.ndarray], Optional[CourtCalibration]],
    stride: int,
    max_frames: Optional[int] = None,
) -> CourtTimeline:
    stride = max(1, stride)
    frame_indices: List[int] = []
    calibrations: List[CourtCalibration] = []
    sampled = 0
    for idx, frame in frame_source():
        if max_frames is not None and idx >= max_frames:
            break
        if idx % stride != 0:
            continue
        sampled += 1
        calib = detect(frame)
        if calib is not None:
            frame_indices.append(idx)
            calibrations.append(calib)
    timeline = CourtTimeline(frame_indices=frame_indices, calibrations=calibrations)
    timeline._sampled = sampled  # type: ignore[attr-defined]
    return timeline
