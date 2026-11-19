"""Point segmentation evaluation: nested decode selection and the bootstrap."""

import numpy as np

from breakpoint_vision.points.decode import DecodeConfig
from breakpoint_vision.points.scoring import Playback
from evaluation.point_segmentation.nested_decode import nested, select
from evaluation.point_segmentation.score_oof import bootstrap_over_videos, edited_like, time_weighted_dead_removed

VIDS = ['v0', 'v1', 'v2', 'v3']
FOLD_OF = {v: i for i, v in enumerate(VIDS)}
FREE = {v: np.zeros(600, np.float32) for v in VIDS}


def row(kept, dead=0.5, n=10):
    return {'points_kept_whole': kept, 'play_time_kept': 0.9, 'dead_time_removed': dead, 'recall': 0.8, 'precision': 0.7,
            'start_err_median': 0.4, 'end_err_median': 1.0, 'n_points': n}


def setting(floor, kept, fake=1, dead=0.5):
    cfg = DecodeConfig(on=0.55, off=0.35, smooth_steps=3, min_seconds=0.8, merge_gap_seconds=0.7, scale_quantile=0.85, scale_floor=floor)
    kept = kept if isinstance(kept, dict) else {v: kept for v in VIDS}
    return cfg, Playback(1.0, 2.0, 2.5), {v: row(kept[v], dead) for v in VIDS}, {v: fake for v in VIDS}


def toy_grid():
    return [
        setting(1.0, 0.8),
        setting(0.2, {'v0': 1.0, 'v1': 1.0, 'v2': 1.0, 'v3': 0.0}),
        setting(0.3, 0.9),
        setting(0.4, 1.0, fake=10),
    ]


def test_selection_uses_only_the_given_videos():
    grid = toy_grid()
    assert select(grid, ['v0', 'v1', 'v2'], FREE)[0].scale_floor == 0.2
    assert select(grid, VIDS, FREE)[0].scale_floor == 0.3


def test_selection_rejects_settings_over_the_play_free_cap():
    picked = [select(toy_grid(), vs, FREE)[0].scale_floor for vs in (VIDS, VIDS[:3], VIDS[1:])]
    assert 0.4 not in picked


def test_ties_go_to_the_highest_scale_floor():
    grid = [setting(1.0, 0.8), setting(0.2, 0.9), setting(0.25, 0.9)]
    assert select(grid, VIDS, FREE)[0].scale_floor == 0.25


def test_each_held_fold_is_scored_with_the_setting_chosen_on_the_others():
    rows, cfgs = nested(toy_grid(), FOLD_OF, FREE)
    assert {k: c['scale_floor'] for k, c in cfgs.items()} == {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.2}
    assert cfgs[3]['lead'] == 1.0 and cfgs[3]['tail'] == 2.0
    assert [rows[v]['points_kept_whole'] for v in VIDS] == [0.9, 0.9, 0.9, 0.0]


def test_bootstrap_is_deterministic_for_a_seed():
    rng = np.random.default_rng(1)
    rows = {f'v{i}': row(float(rng.uniform(0.7, 1.0)), float(rng.uniform(0.2, 0.8)), int(rng.integers(5, 40))) for i in range(12)}
    a = bootstrap_over_videos(rows, samples=300, seed=0)
    b = bootstrap_over_videos(rows, samples=300, seed=0)
    c = bootstrap_over_videos(rows, samples=300, seed=1)
    assert a == b
    assert a != c
    lo, hi = a['points_kept_whole_pooled']
    assert lo <= hi


def test_time_weighted_dead_time_weights_videos_by_their_dead_time():
    golds = {'a': ([{'start': 0.0, 'end': 10.0}], (0.0, 20.0)), 'b': ([{'start': 0.0, 'end': 10.0}], (0.0, 40.0))}
    rows = {'a': row(1.0, dead=1.0), 'b': row(1.0, dead=0.0)}
    assert abs(time_weighted_dead_removed(rows, golds) - 10 / 40) < 1e-9


def test_edited_like_videos_remove_almost_no_dead_time_and_have_points():
    rows = {'a': row(1.0, dead=0.01), 'b': row(1.0, dead=0.3), 'c': row(None, dead=0.0, n=0)}
    assert edited_like(rows) == ['a']
