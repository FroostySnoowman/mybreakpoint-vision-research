"""The moving-camera core: a court sampled over time, looked up per event.

These tests use fake calibrations and a fake detector so they run without torch
or a trained model. They lock in two behaviors: (1) we sample the court across
the video, not once; (2) a frame is mapped through the calibration in force at
it.
"""

import numpy as np

from breakpoint_vision.infer.court_infer import CourtCalibration
from breakpoint_vision.infer.court_timeline import CourtTimeline, build_court_timeline


def _calib(tag: float) -> CourtCalibration:
    return CourtCalibration(
        homography=np.eye(3) * tag,
        reprojection_error_m=0.02,
        used_keypoints=8,
        position_sigma_m=0.03,
    )


def test_active_uses_the_sample_in_force_at_the_frame():
    tl = CourtTimeline(frame_indices=[0, 30, 60], calibrations=[_calib(1), _calib(2), _calib(3)])
    assert tl.active(0, 30).homography[0, 0] == 1
    assert tl.active(12, 30).homography[0, 0] == 1
    assert tl.active(29, 30).homography[0, 0] == 1
    assert tl.active(30, 30).homography[0, 0] == 2
    assert tl.active(59, 30).homography[0, 0] == 2
    assert tl.active(60, 30).homography[0, 0] == 3
    assert tl.active(89, 30).homography[0, 0] == 3
    assert tl.active(999, 30) is None


def test_active_is_forward_only():
    """A sample is a check: authoritative from its own frame, never backwards.

    Reaching backwards would hand a court to frames an earlier check already found
    courtless.
    """
    tl = CourtTimeline(frame_indices=[30, 60], calibrations=[_calib(1), _calib(2)])
    assert tl.active(0, 30) is None
    assert tl.active(29, 30) is None
    assert tl.active(30, 30).homography[0, 0] == 1


def test_empty_timeline():
    tl = CourtTimeline(frame_indices=[], calibrations=[])
    assert tl.is_empty()
    assert tl.active(5, 30) is None
    assert tl.mean_position_sigma() is None


def test_build_samples_on_stride_and_skips_failed_detections():
    frames = [(i, np.zeros((4, 4, 3), dtype=np.uint8)) for i in range(100)]

    def source():
        return iter(frames)

    def detect(_frame):
        detect.calls += 1
        return None if detect.calls == 2 else _calib(float(detect.calls))

    detect.calls = 0

    tl = build_court_timeline(source, detect, stride=25)
    assert detect.calls == 4                       # frames 0,25,50,75 sampled
    assert tl.frame_indices == [0, 50, 75]         # frame 25 (2nd call) failed, dropped
    assert abs(tl.coverage() - 3 / 4) < 1e-9       # 3 of 4 samples succeeded


def test_stride_one_samples_every_frame():
    frames = [(i, np.zeros((2, 2, 3), dtype=np.uint8)) for i in range(5)]
    tl = build_court_timeline(lambda: iter(frames), lambda f: _calib(1.0), stride=1)
    assert tl.frame_indices == [0, 1, 2, 3, 4]
    assert tl.coverage() == 1.0
