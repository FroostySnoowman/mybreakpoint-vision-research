import numpy as np

from breakpoint_vision.points.decode import DecodeConfig
from breakpoint_vision.points.scoring import Playback, play_segments, playback_plan, product_metrics, score_videos, summarize


def seg(a, b, kind='point'):
    return {'start': a, 'end': b, 'kind': kind}


def test_playback_plan_pads_and_merges_padded_segments():
    plan = playback_plan([seg(10.0, 14.0), seg(17.0, 20.0), seg(40.0, 45.0)], Playback(1.0, 1.5, 2.5))
    assert plan == [[9.0, 21.5], [39.0, 46.5]]


def test_playback_plan_keeps_ranges_apart_at_the_merge_gap_and_clamps_at_zero():
    assert playback_plan([seg(0.5, 2.0), seg(6.0, 7.0)], Playback(1.0, 1.0, 2.0)) == [[0.0, 3.0], [5.0, 8.0]]


def test_a_point_is_kept_whole_only_when_almost_all_of_it_plays():
    gold = [seg(10.0, 20.0), seg(40.0, 50.0)]
    pred = [seg(11.0, 19.0), seg(40.0, 44.0)]
    m = product_metrics(pred, gold, (0.0, 60.0), Playback(1.0, 1.0, 0.0))
    assert m['n_points'] == 2
    assert m['points_kept_whole'] == 0.5
    assert abs(m['play_time_kept'] - 0.75) < 1e-6
    assert abs(m['dead_time_removed'] - 0.975) < 1e-6


def test_points_outside_the_window_are_not_counted():
    m = product_metrics([seg(5.0, 8.0)], [seg(5.0, 8.0), seg(100.0, 104.0)], (0.0, 60.0))
    assert m['n_points'] == 1 and m['points_kept_whole'] == 1.0


def test_play_segments_drop_non_play_labels():
    label = {'segments': [seg(0, 1), seg(2, 3, 'fault'), seg(4, 5, 'let'), seg(6, 7, 'warmup')]}
    assert [s['kind'] for s in play_segments(label)] == ['point', 'fault', 'let']


def test_score_and_summarize_pool_points_across_videos():
    p = np.zeros(600, np.float32)
    p[100:200] = 0.9
    probs = {'a': p, 'b': np.zeros(600, np.float32)}
    golds = {'a': ([seg(10.0, 20.0)], (0.0, 60.0)), 'b': ([seg(10.0, 20.0), seg(30.0, 40.0), seg(45.0, 50.0)], (0.0, 60.0))}
    cfg = DecodeConfig(on=0.55, off=0.35, smooth_steps=1, min_seconds=0.8, merge_gap_seconds=0.8, scale_quantile=0.85, scale_floor=1.0)
    rows = score_videos(probs, golds, cfg)
    assert rows['a']['points_kept_whole'] == 1.0 and rows['b']['points_kept_whole'] == 0.0
    s = summarize(rows)
    assert s['points_kept_whole'] == 0.5
    assert s['points_kept_whole_pooled'] == 0.25
