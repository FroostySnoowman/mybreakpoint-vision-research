"""The court gate: ball tracking never runs without a believed court.

Fake calibrations only, so these run without torch or trained weights. They lock
in three behaviors: (1) a geometrically absurd homography is not a court, even
when the solver "succeeded"; (2) a court is acquired with hysteresis and expires
with age instead of being reused forever; (3) with a court held, detections
outside the projected court are rejected. Gate limits are illustrative values
chosen for these fixtures.
"""

import numpy as np

from breakpoint_vision.config import CourtGateConfig
from breakpoint_vision.court import HALF_DOUBLES, HALF_LENGTH
from breakpoint_vision.court_gate import (
    CourtPresence,
    TimelineBallGate,
    ball_in_region,
    ball_region,
    confirmed_timeline,
    court_pixel_quad,
    validate_calibration,
    validated_detector,
)
from breakpoint_vision.geometry import estimate_homography
from breakpoint_vision.infer.court_infer import CourtCalibration
from breakpoint_vision.infer.court_timeline import CourtTimeline

WIDTH = 1280
HEIGHT = 720

_IMAGE_CORNERS = [(240.0, 620.0), (1040.0, 620.0), (860.0, 200.0), (420.0, 200.0)]
_DISTANT_CORNERS = [(420.0, 440.0), (860.0, 440.0), (760.0, 300.0), (520.0, 300.0)]
_WORLD_CORNERS = [
    (-HALF_DOUBLES, -HALF_LENGTH),
    (HALF_DOUBLES, -HALF_LENGTH),
    (HALF_DOUBLES, HALF_LENGTH),
    (-HALF_DOUBLES, HALF_LENGTH),
]

ILLUSTRATIVE = dict(
    min_keypoints=5,
    max_reprojection_error_m=0.5,
    min_court_frame_fraction=0.05,
    max_court_frame_fraction=20.0,
    min_consecutive_detections=3,
    max_court_age_seconds=1.5,
    ball_runoff_side_m=3.0,
    ball_runoff_back_m=6.0,
    ball_up_margin_fraction=0.5,
)


def _cfg(**overrides) -> CourtGateConfig:
    return CourtGateConfig(**{**ILLUSTRATIVE, **overrides})


def _plausible_calib(**overrides) -> CourtCalibration:
    fields = dict(
        homography=estimate_homography(_IMAGE_CORNERS, _WORLD_CORNERS),
        reprojection_error_m=0.02,
        used_keypoints=8,
        position_sigma_m=0.03,
    )
    fields.update(overrides)
    return CourtCalibration(**fields)


def test_plausible_calibration_is_accepted():
    assert validate_calibration(_plausible_calib(), WIDTH, HEIGHT, _cfg()) is True


def test_none_and_degenerate_frames_are_rejected():
    assert validate_calibration(None, WIDTH, HEIGHT, _cfg()) is False
    assert validate_calibration(_plausible_calib(), 0, HEIGHT, _cfg()) is False


def test_too_few_keypoints_is_rejected():
    assert validate_calibration(_plausible_calib(used_keypoints=4), WIDTH, HEIGHT, _cfg()) is False


def test_large_or_nonfinite_reprojection_error_is_rejected():
    assert validate_calibration(_plausible_calib(reprojection_error_m=5.0), WIDTH, HEIGHT, _cfg()) is False
    assert validate_calibration(_plausible_calib(reprojection_error_m=float("nan")), WIDTH, HEIGHT, _cfg()) is False


def test_self_intersecting_court_is_rejected():
    """A homography from junk keypoint peaks solves fine but folds the court."""
    twisted = estimate_homography(
        [_IMAGE_CORNERS[0], _IMAGE_CORNERS[2], _IMAGE_CORNERS[1], _IMAGE_CORNERS[3]],
        _WORLD_CORNERS,
    )
    assert validate_calibration(_plausible_calib(homography=twisted), WIDTH, HEIGHT, _cfg()) is False


def test_microscopic_court_is_rejected():
    tiny = estimate_homography(
        [(10.0, 10.0), (24.0, 10.0), (22.0, 2.0), (12.0, 2.0)],
        _WORLD_CORNERS,
    )
    assert validate_calibration(_plausible_calib(homography=tiny), WIDTH, HEIGHT, _cfg()) is False


def test_singular_homography_has_no_quad():
    assert court_pixel_quad(np.zeros((3, 3))) is None


def test_validated_detector_swallows_implausible_calibrations():
    frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    good = validated_detector(lambda _f: _plausible_calib(), _cfg())
    bad = validated_detector(lambda _f: _plausible_calib(used_keypoints=4), _cfg())
    missing = validated_detector(lambda _f: None, _cfg())
    assert good(frame) is not None
    assert bad(frame) is None
    assert missing(frame) is None


def test_presence_requires_consecutive_detections_before_locking():
    cfg = _cfg(min_consecutive_detections=2)
    court = CourtPresence(cfg)
    assert court.ball_enabled() is False

    court.observe(0.0, _plausible_calib(), WIDTH, HEIGHT)
    assert court.locked is False
    assert court.ball_enabled() is False

    court.observe(0.5, _plausible_calib(), WIDTH, HEIGHT)
    assert court.locked is True
    assert court.ball_enabled() is True
    assert court.segments() is not None
    assert court.calibration() is not None


def test_a_single_missed_detection_does_not_drop_the_court():
    court = CourtPresence(_cfg(min_consecutive_detections=2, max_court_age_seconds=1.5))
    court.observe(0.0, _plausible_calib(), WIDTH, HEIGHT)
    court.observe(0.5, _plausible_calib(), WIDTH, HEIGHT)
    court.observe(1.0, None, WIDTH, HEIGHT)
    assert court.locked is True


def test_stale_court_expires_instead_of_being_reused_forever():
    court = CourtPresence(_cfg(min_consecutive_detections=2, max_court_age_seconds=1.5))
    court.observe(0.0, _plausible_calib(), WIDTH, HEIGHT)
    court.observe(0.5, _plausible_calib(), WIDTH, HEIGHT)
    assert court.locked is True

    court.tick(1.9)
    assert court.locked is True

    court.tick(2.1)
    assert court.locked is False
    assert court.segments() is None
    assert court.calibration() is None
    assert court.ball_enabled() is False


def test_relock_requires_the_full_streak_again():
    cfg = _cfg(min_consecutive_detections=2, max_court_age_seconds=1.0)
    court = CourtPresence(cfg)
    court.observe(0.0, _plausible_calib(), WIDTH, HEIGHT)
    court.observe(0.2, _plausible_calib(), WIDTH, HEIGHT)
    court.tick(5.0)
    assert court.locked is False

    court.observe(5.1, _plausible_calib(), WIDTH, HEIGHT)
    assert court.locked is False
    court.observe(5.2, _plausible_calib(), WIDTH, HEIGHT)
    assert court.locked is True


def test_ball_outside_the_court_region_is_rejected():
    """A court filling the frame has run-off past every frame edge, so anything
    visible is in play; only points well outside the frame are rejected."""
    court = CourtPresence(_cfg(min_consecutive_detections=1))
    court.observe(0.0, _plausible_calib(), WIDTH, HEIGHT)
    assert court.locked is True

    assert court.accepts_ball((640.0, 400.0)) is True
    assert court.accepts_ball((640.0, 60.0)) is True
    assert court.accepts_ball((20.0, 400.0)) is True
    assert court.accepts_ball((1270.0, 400.0)) is True
    assert court.accepts_ball((-400.0, 400.0)) is False
    assert court.accepts_ball((640.0, 1200.0)) is False


def test_ball_far_above_or_below_a_distant_court_is_rejected():
    """With the court occupying part of the frame, the stands are out of bounds."""
    calib = _plausible_calib(homography=estimate_homography(_DISTANT_CORNERS, _WORLD_CORNERS))
    court = CourtPresence(_cfg(min_consecutive_detections=1))
    court.observe(0.0, calib, WIDTH, HEIGHT)
    assert court.locked is True

    assert court.accepts_ball((640.0, 380.0)) is True
    assert court.accepts_ball((640.0, 200.0)) is True
    assert court.accepts_ball((640.0, 40.0)) is False
    assert court.accepts_ball((640.0, 640.0)) is False


def test_unlocked_presence_accepts_no_ball():
    court = CourtPresence(_cfg())
    assert court.accepts_ball((640.0, 400.0)) is False


def test_requirement_can_be_disabled_for_diagnostics():
    court = CourtPresence(_cfg(require_court_for_ball=False))
    assert court.ball_enabled() is True
    assert court.accepts_ball((0.0, 0.0)) is True


def test_ball_region_extends_further_up_than_down():
    """A ball above the court plane projects upward, so headroom is asymmetric."""
    calib = _plausible_calib(homography=estimate_homography(_DISTANT_CORNERS, _WORLD_CORNERS))
    quad = court_pixel_quad(calib.homography)
    region = ball_region(calib, _cfg())
    top = float(quad[:, 1].min())
    bottom = float(quad[:, 1].max())
    assert top - float(region[:, 1].min()) > float(region[:, 1].max()) - bottom


def test_ball_region_covers_the_runoff_beyond_the_lines():
    """A ball hit wide lands off the court but still inside the venue's run-off."""
    calib = _plausible_calib(homography=estimate_homography(_DISTANT_CORNERS, _WORLD_CORNERS))
    quad = court_pixel_quad(calib.homography)
    region = ball_region(calib, _cfg())
    assert float(region[:, 0].min()) < float(quad[:, 0].min())
    assert float(region[:, 0].max()) > float(quad[:, 0].max())


def test_ball_region_follows_perspective_not_a_pixel_box():
    """Near-camera run-off is wide; far-baseline run-off is a sliver.

    A fixed pixel margin around the court's bounding box would reject legitimate
    balls hit wide down the near sideline, because the same meters of apron are
    far wider in pixels near the camera than at the far end.
    """
    calib = _plausible_calib()
    quad = court_pixel_quad(calib.homography)
    region = ball_region(calib, _cfg())

    near_y = float(quad[:, 1].max())
    far_y = float(quad[:, 1].min())
    near_left = min(float(x) for (x, y) in region if y > (near_y + far_y) / 2)
    far_left = min(float(x) for (x, y) in region if y <= (near_y + far_y) / 2)
    assert near_left < far_left

    assert ball_in_region((float(quad[:, 0].min()) - 60.0, near_y - 5.0), region) is True


def test_ball_region_is_none_for_a_singular_homography():
    assert ball_region(_plausible_calib(homography=np.zeros((3, 3))), _cfg()) is None
    assert ball_in_region((0.0, 0.0), None) is False


def test_confirmed_timeline_drops_isolated_detections():
    """One fluke court in a crowd shot must not open a ball-tracking window."""
    calib = _plausible_calib()
    sampled = CourtTimeline(
        frame_indices=[0, 15, 30, 300, 600, 615, 630],
        calibrations=[calib] * 7,
    )
    confirmed = confirmed_timeline(sampled, stride=15, min_consecutive=2)
    assert confirmed.frame_indices == [0, 15, 30, 600, 615, 630]


def test_confirmed_timeline_is_a_no_op_without_hysteresis():
    calib = _plausible_calib()
    sampled = CourtTimeline(frame_indices=[0, 300], calibrations=[calib, calib])
    assert confirmed_timeline(sampled, 15, 1) is sampled
    assert confirmed_timeline(None, 15, 2) is None


def test_confirmed_timeline_can_reject_everything():
    calib = _plausible_calib()
    sampled = CourtTimeline(frame_indices=[0, 300, 900], calibrations=[calib] * 3)
    assert confirmed_timeline(sampled, 15, 2).is_empty()


def test_timeline_gate_blocks_frames_beyond_the_court_age():
    calib = _plausible_calib(homography=estimate_homography(_DISTANT_CORNERS, _WORLD_CORNERS))
    timeline = CourtTimeline(frame_indices=[0, 15, 30], calibrations=[calib] * 3)
    gate = TimelineBallGate(timeline, fps=30.0, cfg=_cfg(), stride=15)

    assert gate.allows_frame(7) is True
    assert gate.allows_frame(30) is True
    assert gate.allows_frame(44) is True
    assert gate.allows_frame(45) is False
    assert gate.allows_frame(600) is False
    assert gate.accepts(600, 640.0, 400.0) is False
    assert gate.accepts(7, 640.0, 400.0) is True
    assert gate.accepts(7, 20.0, 400.0) is False


def test_timeline_gate_does_not_extrapolate_a_court_into_courtless_spans():
    """A missing batch sample means we looked and found nothing, not "unknown".

    Frames 0-149 have no court; the rally starts at 150. Reusing frame 150's
    calibration for frame 149 would let the ball tracker run over the footage
    that precedes the rally.
    """
    calib = _plausible_calib()
    timeline = CourtTimeline(frame_indices=[150, 165, 180], calibrations=[calib] * 3)
    gate = TimelineBallGate(timeline, fps=30.0, cfg=_cfg(), stride=15)

    assert gate.allows_frame(100) is False
    assert gate.allows_frame(140) is False
    assert gate.allows_frame(149) is False
    assert gate.allows_frame(150) is True
    assert gate.allows_frame(194) is True
    assert gate.allows_frame(195) is False


def test_timeline_gate_blocks_everything_when_no_court_was_found():
    gate = TimelineBallGate(CourtTimeline(frame_indices=[], calibrations=[]), 30.0, _cfg())
    assert gate.allows_frame(0) is False
    assert gate.accepts(0, 640.0, 400.0) is False


def test_timeline_gate_passes_everything_when_gating_is_disabled():
    gate = TimelineBallGate(
        CourtTimeline(frame_indices=[], calibrations=[]),
        30.0,
        _cfg(require_court_for_ball=False),
    )
    assert gate.allows_frame(999) is True
    assert gate.accepts(999, -50.0, -50.0) is True
