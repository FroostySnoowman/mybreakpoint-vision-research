import numpy as np

from breakpoint_vision.points.decode import DecodeConfig, decode, evaluate, match_segments

ILLUSTRATIVE = dict(
    on=0.55,
    off=0.35,
    smooth_steps=1,
    min_seconds=0.8,
    merge_gap_seconds=0.8,
    scale_quantile=0.85,
    scale_floor=1.0,
)


def _cfg(**overrides) -> DecodeConfig:
    return DecodeConfig(**{**ILLUSTRATIVE, **overrides})


def test_decode_finds_segments_with_hysteresis():
    p = np.zeros(100, np.float32)
    p[10:30] = 0.9
    p[30:33] = 0.5
    p[50:70] = 0.8
    segs = decode(p, _cfg(merge_gap_seconds=0.0))
    assert [(s['start'], s['end']) for s in segs] == [(1.0, 3.3), (5.0, 7.0)]


def test_decode_merges_short_gaps_and_drops_blips():
    p = np.zeros(100, np.float32)
    p[10:20] = 0.9
    p[25:40] = 0.9
    p[60:64] = 0.9
    segs = decode(p, _cfg(merge_gap_seconds=1.0, min_seconds=1.0))
    assert [(s['start'], s['end']) for s in segs] == [(1.0, 4.0)]


def test_decode_handles_play_at_the_edges():
    p = np.ones(20, np.float32)
    assert [(s['start'], s['end']) for s in decode(p, _cfg())] == [(0.0, 2.0)]
    assert decode(np.zeros(20, np.float32), _cfg()) == []


def test_match_is_one_to_one():
    pred = [{'start': 0, 'end': 10}, {'start': 1, 'end': 9}]
    gold = [{'start': 0, 'end': 10}]
    assert len(match_segments(pred, gold)) == 1


def test_evaluate_reports_time_and_boundary_metrics():
    gold = [{'start': 10, 'end': 20}, {'start': 40, 'end': 50}]
    pred = [{'start': 9, 'end': 21}, {'start': 70, 'end': 75}]
    m = evaluate(pred, gold, (0, 100))
    assert m['recall'] == 0.5 and m['precision'] == 0.5
    assert abs(m['play_time_kept'] - 0.5) < 0.01
    assert abs(m['start_err_median'] - 1.0) < 1e-9 and abs(m['end_err_median'] - 1.0) < 1e-9
    assert abs(m['dead_time_removed'] - 73 / 80) < 0.01


def test_rescale_lifts_an_under_confident_video_over_the_thresholds():
    p = np.full(100, 0.02, np.float32)
    p[20:50] = 0.4
    fixed = _cfg(min_seconds=0.5)
    assert decode(p, fixed) == []
    scaled = decode(p, _cfg(min_seconds=0.5, scale_quantile=0.75, scale_floor=0.1))
    assert [(s['start'], s['end']) for s in scaled] == [(2.0, 5.0)]
    assert scaled[0]['confidence'] == 1.0


def test_rescale_floor_keeps_noise_in_a_video_without_play_below_the_thresholds():
    p = np.random.default_rng(0).uniform(0.0, 0.08, 200).astype(np.float32)
    assert decode(p, _cfg(scale_quantile=0.75, scale_floor=0.2)) == []


def test_a_floor_of_one_leaves_probabilities_unchanged():
    p = np.random.default_rng(1).uniform(0.0, 1.0, 300).astype(np.float32)
    assert decode(p, _cfg(scale_quantile=0.5, scale_floor=1.0)) == decode(p, _cfg())
