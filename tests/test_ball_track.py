"""Reading a heatmap as a trajectory rather than a per-frame argmax.

These pin the tracker's behaviors: a secondary peak can win if the track vouches
for it, but only a confirmed track may vouch, the gate is a bounded distance, and
a track that stops being fed expires instead of coasting forever. Thresholds are
illustrative values chosen for these fixtures.
"""

import numpy as np

from breakpoint_vision.ball_track import (
    BallTracker,
    BallTrackerConfig,
    Candidate,
    heatmap_candidates,
)

WIDTH = 80
HEIGHT = 48

ILLUSTRATIVE = dict(
    min_candidate_confidence=0.03,
    seed_confidence=0.6,
    keep_confidence=0.1,
    gate_fraction=0.03,
    gate_growth_fraction=0.015,
    coast_frames=3,
    confirm_detections=3,
)


def _cfg(**overrides) -> BallTrackerConfig:
    return BallTrackerConfig(**{**ILLUSTRATIVE, **overrides})


def _heatmap(*blobs) -> np.ndarray:
    hm = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    for (cx, cy, amp) in blobs:
        hm += amp * np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * 1.5**2)))
    return np.clip(hm, 0.0, 1.0)


def _seed(tracker: BallTracker, x: float, y: float, steps: int = 3, dx: float = 2.0):
    """Walk a strong, straight track until the tracker trusts its prediction."""
    for i in range(steps):
        tracker.update([Candidate(x=x + dx * i, y=y, confidence=0.9)])
    assert tracker.confirmed


def test_a_secondary_peak_survives_alongside_a_taller_one():
    """A plain argmax throws the ball away whenever a distractor scores higher."""
    cands = heatmap_candidates(_heatmap((20, 20, 0.9), (34, 20, 0.4)), _cfg())
    assert len(cands) >= 2
    xs = sorted(round(c.x) for c in cands[:2])
    assert xs == [20, 34]
    assert cands[0].confidence > cands[1].confidence


def test_candidates_are_ranked_strongest_first_and_capped():
    cfg = _cfg(max_candidates=2)
    cands = heatmap_candidates(_heatmap((10, 10, 0.9), (40, 10, 0.7), (60, 30, 0.5)), cfg)
    assert len(cands) == 2
    assert cands[0].confidence >= cands[1].confidence


def test_subpixel_refinement_beats_the_integer_peak():
    """A blob centered between pixels should report a fractional position."""
    cands = heatmap_candidates(_heatmap((20.5, 15.5, 0.9)), _cfg())
    assert cands
    assert abs(cands[0].x - 20.5) < 0.3
    assert abs(cands[0].y - 15.5) < 0.3
    assert cands[0].x != round(cands[0].x)


def test_a_saturated_plateau_does_not_spend_every_candidate_slot_on_itself():
    """A confident blob saturates to a flat 1.0 region, and `>=` calls every pixel
    of it a maximum. Left alone that fills all four slots with one ball and hides
    the genuine second peak the tracker needs."""
    hm = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    hm[16:26, 20:33] = 1.0
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    hm += 0.3 * np.exp(-(((xs - 60.0) ** 2 + (ys - 30.0) ** 2) / (2 * 2.0**2)))
    hm = np.clip(hm, 0.0, 1.0)

    cands = heatmap_candidates(hm, _cfg())
    assert len(cands) == 2
    assert any(abs(c.x - 60.0) < 1.0 and abs(c.y - 30.0) < 1.0 for c in cands)


def test_tied_peaks_are_ordered_by_position_not_scan_order():
    """Ties break on (y, x) rather than on however the array happened to be walked,
    so every implementation produces the same candidate list for a frame."""
    hm = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for (x, y) in [(15, 30), (55, 12), (35, 41)]:
        hm[y, x] = 0.77
    cands = heatmap_candidates(hm, _cfg())
    assert [(round(c.x), round(c.y)) for c in cands] == [(55, 12), (15, 30), (35, 41)]


def test_a_peak_at_the_frame_edge_is_not_pulled_inward():
    """The sub-pixel window clamps out-of-bounds positions instead of dropping them;
    dropping them leaves an asymmetric window that biases the centroid inward."""
    hm = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    ys, xs = np.mgrid[0:HEIGHT, 0:WIDTH]
    hm += np.exp(-(((xs - 0.0) ** 2 + (ys - 0.0) ** 2) / (2 * 1.8**2)))
    cands = heatmap_candidates(hm, _cfg())
    assert cands
    assert np.hypot(cands[0].x, cands[0].y) < 0.7


def test_noise_below_the_floor_produces_no_candidates():
    hm = np.full((HEIGHT, WIDTH), 0.005, dtype=np.float32)
    assert heatmap_candidates(hm, _cfg()) == []


def test_a_confirmed_track_admits_a_weak_peak_that_continues_it():
    """Correct but faint detections that continue the track are not suppressed."""
    tracker = BallTracker(WIDTH, _cfg())
    _seed(tracker, 10.0, 20.0)
    picked = tracker.update([Candidate(x=16.0, y=20.0, confidence=0.12)])
    assert picked is not None
    assert picked == (16.0, 20.0)
    assert tracker.confidence == 0.12


def test_an_unconfirmed_track_may_not_admit_a_weak_peak():
    """Otherwise one fluke opens a low-threshold window and invents a ball."""
    tracker = BallTracker(WIDTH, _cfg())
    assert tracker.update([Candidate(x=10.0, y=20.0, confidence=0.9)]) is not None
    assert not tracker.confirmed
    assert tracker.update([Candidate(x=12.0, y=20.0, confidence=0.12)]) is None


def test_a_weak_peak_far_from_the_prediction_is_rejected():
    tracker = BallTracker(WIDTH, _cfg())
    _seed(tracker, 10.0, 20.0)
    assert tracker.update([Candidate(x=70.0, y=40.0, confidence=0.12)]) is None


def test_a_strong_peak_that_teleports_still_reseeds_the_track():
    """A camera cut is a real event; refusing to ever move again would be worse."""
    tracker = BallTracker(WIDTH, _cfg())
    _seed(tracker, 10.0, 20.0)
    picked = tracker.update([Candidate(x=70.0, y=40.0, confidence=0.95)])
    assert picked == (70.0, 40.0)


def test_the_gate_widens_while_coasting():
    """Uncertainty grows while blind, so the same offset is out then in.

    Here the gate is 0.03*80 = 2.4 px, growing 0.015*80 = 1.2 px per coasted frame,
    so 3.0 px away is outside the fresh gate and inside it after one miss.
    """
    cfg = _cfg()
    offset = 3.0
    assert cfg.gate_fraction * WIDTH < offset < (cfg.gate_fraction + cfg.gate_growth_fraction) * WIDTH

    tight = BallTracker(WIDTH, cfg)
    _seed(tight, 10.0, 20.0, steps=3, dx=0.0)
    assert tight.update([Candidate(x=10.0 + offset, y=20.0, confidence=0.12)]) is None

    coasted = BallTracker(WIDTH, cfg)
    _seed(coasted, 10.0, 20.0, steps=3, dx=0.0)
    assert coasted.update([]) is None
    assert coasted.update([Candidate(x=10.0 + offset, y=20.0, confidence=0.12)]) is not None


def test_a_track_expires_after_coast_frames_and_stops_predicting():
    cfg = _cfg(coast_frames=2)
    tracker = BallTracker(WIDTH, cfg)
    _seed(tracker, 10.0, 20.0)
    for _ in range(cfg.coast_frames + 1):
        assert tracker.update([]) is None
    assert not tracker.confirmed
    assert tracker.update([Candidate(x=16.0, y=20.0, confidence=0.12)]) is None


def test_reset_forgets_the_track_entirely():
    """Called on a court-gate drop: a prediction across a cut is worse than none."""
    tracker = BallTracker(WIDTH, _cfg())
    _seed(tracker, 10.0, 20.0)
    tracker.reset()
    assert not tracker.confirmed
    assert tracker.confidence == 0.0
    assert tracker.update([Candidate(x=16.0, y=20.0, confidence=0.12)]) is None


def test_reported_confidence_is_the_accepted_peak_not_the_strongest():
    """A faint detection the prediction vouched for must not be overstated."""
    tracker = BallTracker(WIDTH, _cfg())
    _seed(tracker, 10.0, 20.0)
    picked = tracker.update(
        [Candidate(x=70.0, y=40.0, confidence=0.80), Candidate(x=16.0, y=20.0, confidence=0.11)]
    )
    assert picked == (16.0, 20.0)
    assert tracker.confidence == 0.11


def test_velocity_survives_a_coasted_frame():
    """Position is divided by the elapsed steps, not assumed to be one frame."""
    tracker = BallTracker(WIDTH, _cfg())
    _seed(tracker, 10.0, 20.0, steps=3, dx=2.0)
    assert tracker.update([]) is None
    picked = tracker.update([Candidate(x=18.0, y=20.0, confidence=0.9)])
    assert picked == (18.0, 20.0)
    assert tracker.update([Candidate(x=20.0, y=20.0, confidence=0.12)]) is not None


def test_a_non_2d_heatmap_is_rejected():
    try:
        heatmap_candidates(np.zeros((1, 1, HEIGHT, WIDTH), dtype=np.float32), _cfg())
    except ValueError:
        return
    raise AssertionError("a 4-D tensor should not be read as a heatmap")
