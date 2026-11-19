"""Ball tracking evaluation helpers: counting, frame selection, paired tests.

Tracker thresholds are illustrative values for these fixtures.
"""

import random

import numpy as np

from breakpoint_vision.ball_track import BallTrackerConfig
from evaluation.ball_tracking.frames import context_indices, sequence_indices, variant_indices
from evaluation.ball_tracking.metrics import (
    count_frame, exact_mcnemar_p, is_hit, new_counts, paired_counts, rates, summarize_counts, track_last, wilson_interval,
)
from evaluation.ball_tracking.smoother import SmootherParams, pool, smooth

ILLUSTRATIVE = BallTrackerConfig(
    min_candidate_confidence=0.03,
    seed_confidence=0.6,
    keep_confidence=0.1,
    gate_fraction=0.03,
    gate_growth_fraction=0.015,
    coast_frames=3,
    confirm_detections=3,
)


def visible(x, y):
    return {'visible': True, 'x': x, 'y': y}


def test_hits_misses_and_no_output_are_counted_on_visible_frames():
    g = new_counts()
    count_frame(g, visible(100, 100), (102, 100))
    count_frame(g, visible(100, 100), (105, 100))
    count_frame(g, visible(100, 100), (100, 108))
    count_frame(g, visible(100, 100), (110, 100))
    count_frame(g, visible(100, 100), None)
    assert (g['vis'], g['hit3'], g['hit8'], g['none']) == (5, 1, 3, 1)
    r = rates(g)
    assert r['track@8px'] == 0.6 and r['track@3px'] == 0.2
    assert r['no_output'] == 0.2 and r['wrong'] == 0.2


def test_any_output_on_a_negative_frame_is_a_false_alarm_and_uncertain_frames_are_excluded():
    g = new_counts()
    count_frame(g, {'visible': False, 'x': None, 'y': None}, (10, 10))
    count_frame(g, {'visible': 'not_in_play', 'x': None, 'y': None}, None)
    count_frame(g, {'visible': 'uncertain', 'x': None, 'y': None}, (10, 10))
    count_frame(g, {'visible': 'uncertain', 'x': None, 'y': None}, None)
    assert (g['neg'], g['fa'], g['uncertain'], g['vis']) == (2, 1, 2, 0)
    assert rates(g)['false_alarm'] == '1/2'


def test_summary_pools_videos_into_groups_and_totals():
    a, b = new_counts(), new_counts()
    count_frame(a, visible(0, 0), (1, 1))
    count_frame(b, visible(0, 0), None)
    count_frame(b, {'visible': False}, (5, 5))
    s = summarize_counts({'v1': a, 'v2': b}, {'v1': 'day', 'v2': 'night'})
    assert s['total']['vis'] == 2 and s['total']['hit8'] == 1 and s['total']['fa'] == 1
    assert set(s['by_group']) == {'day', 'night'}
    assert s['per_video']['v2']['false_alarm'] == '1/1'


def test_is_hit_uses_the_8px_tolerance():
    assert is_hit(visible(0, 0), (8, 0))
    assert not is_hit(visible(0, 0), (8.01, 0))
    assert not is_hit(visible(0, 0), None)


def test_sequence_ends_at_the_nearest_frame_when_timestamps_are_rounded():
    times = [round(k / 30, 3) for k in range(90)]
    t = 31 / 30
    kept = sequence_indices(times, t, lead=0.5)
    assert kept[-1] == 31
    assert all(times[i] >= t - 0.5 for i in kept)
    assert kept == list(range(kept[0], 32))


def test_sequence_thins_60fps_to_30fps():
    times = [k / 60 for k in range(240)]
    kept = sequence_indices(times, 2.0, lead=1.0)
    assert kept[-1] == 120
    assert np.allclose(np.diff([times[i] for i in kept]), 1 / 30)


def test_variant_frames_anchor_on_the_nearest_frame_and_thin_backward():
    times = [round(k / 60, 3) for k in range(0, 61)]
    t1 = 0.5085
    kept = variant_indices(times, t1)
    assert times[kept[-1]] == 0.5
    assert np.all(np.diff([times[i] for i in kept]) >= 1 / 30 - 0.005)


def test_context_window_reports_the_frame_nearest_the_label():
    times = [round(k / 30, 3) for k in range(150)]
    kept, idx = context_indices(times, 2.0005, context=1.5)
    assert times[kept[idx]] == 2.0
    assert times[kept[0]] >= 0.5 and times[kept[-1]] <= 3.5005


def test_track_last_follows_a_moving_ball():
    frames = [[(100 + 5 * i, 50 + 2 * i, 0.9), (400, 200, 0.5)] for i in range(10)]
    assert track_last(frames, ILLUSTRATIVE) == (145.0, 68.0)
    assert track_last([[]] * 5, ILLUSTRATIVE) is None


class _TupleTracker:
    """Reference tracker on (x, y, confidence) tuples, written independently of BallTracker."""

    def __init__(self, width, cfg):
        self.width, self.cfg = width, cfg
        self.pos, self.vel, self.missed, self.count = None, (0.0, 0.0), 0, 0

    def update(self, cands):
        c = self.cfg
        if not cands:
            return self.miss()
        if self.pos is not None and self.count >= c.confirm_detections:
            steps = self.missed + 1
            px, py = self.pos[0] + self.vel[0] * steps, self.pos[1] + self.vel[1] * steps
            budget = (c.gate_fraction + c.gate_growth_fraction * self.missed) * self.width
            best = min((k for k in cands if k[2] >= c.keep_confidence and np.hypot(k[0] - px, k[1] - py) <= budget), key=lambda k: np.hypot(k[0] - px, k[1] - py), default=None)
            if best is not None:
                return self.accept(best, best[2] >= c.seed_confidence)
        if cands[0][2] >= c.seed_confidence:
            return self.accept(cands[0], True)
        return self.miss()

    def accept(self, k, strong):
        if self.pos is not None:
            steps = self.missed + 1
            self.vel = ((k[0] - self.pos[0]) / steps, (k[1] - self.pos[1]) / steps)
        self.pos = (k[0], k[1])
        self.missed = 0
        self.count = self.count + 1 if strong or self.count >= self.cfg.confirm_detections else 0
        return self.pos

    def miss(self):
        self.missed += 1
        if self.missed > self.cfg.coast_frames:
            self.pos, self.vel, self.missed, self.count = None, (0.0, 0.0), 0, 0
        return None


def test_ball_tracker_matches_the_tuple_reference_on_random_candidate_lists():
    rng = random.Random(0)
    for _ in range(200):
        x, y = rng.uniform(0, 512), rng.uniform(0, 288)
        frames = []
        for _ in range(30):
            x, y = x + rng.uniform(-12, 12), y + rng.uniform(-8, 8)
            cands = []
            if rng.random() < 0.8:
                cands.append((x + rng.uniform(-2, 2), y + rng.uniform(-2, 2), rng.uniform(0.05, 1.0)))
            for _ in range(rng.randint(0, 3)):
                cands.append((rng.uniform(0, 512), rng.uniform(0, 288), rng.uniform(0.03, 1.0)))
            cands.sort(key=lambda c: -c[2])
            frames.append(cands)
        ref = _TupleTracker(512, ILLUSTRATIVE)
        for i in range(1, len(frames) + 1):
            expected = None
            for c in frames[i - 1:i]:
                expected = ref.update(c)
            assert track_last(frames[:i], ILLUSTRATIVE) == expected


def test_exact_mcnemar_matches_known_values():
    assert exact_mcnemar_p(4, 0) == 0.125
    assert abs(exact_mcnemar_p(4, 2) - 0.6875) < 1e-12
    assert exact_mcnemar_p(3, 3) == 1.0
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(21, 13) == exact_mcnemar_p(13, 21)


def test_paired_counts_only_use_frames_present_in_both_runs():
    a = {1: True, 2: True, 3: False, 4: True}
    b = {1: True, 2: False, 3: True}
    assert paired_counts(a, b) == (1, 1)


def test_wilson_interval_brackets_the_proportion():
    lo, hi = wilson_interval(45, 50)
    assert lo < 0.9 < hi
    assert 0 < lo < hi < 1


def test_pool_merges_nearby_candidates_with_a_noisy_or():
    merged = pool([(10.0, 10.0, 0.5)], [(12.0, 10.0, 0.6), (100.0, 100.0, 0.3)])
    assert len(merged) == 2
    x, y, c = merged[0]
    assert (x, y) == (12.0, 10.0) and abs(c - 0.8) < 1e-9
    assert merged[1] == (100.0, 100.0, 0.3)


def test_smoother_prefers_a_smooth_track_over_a_jumping_distractor():
    cands = []
    for i in range(12):
        track = (50.0 + 6 * i, 40.0 + 3 * i, 0.6)
        distractor = (300.0 if i % 2 else 20.0, 200.0, 0.7)
        cands.append([distractor, track])
    path = smooth(cands, SmootherParams())
    assert all(p is not None and abs(p[0] - (50.0 + 6 * i)) < 1e-9 for i, p in enumerate(path))
