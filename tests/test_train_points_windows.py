import numpy as np

from breakpoint_vision.train.train_points import Windows


def _video(T, gold=None):
    weight = np.full(T, 0.3, np.float32)
    if gold is not None:
        weight[gold[0]:gold[1]] = 1.0
    return {'weight': weight, 'label': np.zeros(T, np.float32), 'feats': np.zeros((T, 8), np.float32), 'logmel': np.zeros((T * 10, 64), np.float32)}


def test_videos_without_train_gold_are_allowed_and_skipped_for_gold_spans():
    w = Windows([_video(1000), _video(1200, (300, 900)), _video(800)], steps=100, n=4, seed=0, augment=False)
    assert w.gold_spans == [(1, 300, 899)]


def test_no_train_gold_anywhere_gives_no_gold_spans():
    w = Windows([_video(500), _video(700)], steps=100, n=4, seed=0, augment=False)
    assert w.gold_spans == []


def _labeled(T, segments):
    from breakpoint_vision.train.train_points import target_boundaries
    v = _video(T, (0, T))
    for a, b in segments:
        v['label'][a:b] = 1.0
    v['boundaries'] = target_boundaries(v['label'])
    v['logmel'] = np.zeros((64, T * 10), np.float32)
    return v


def _start_of(sample, T):
    return int(round(float(sample[1].numpy()[0, 2]) * T))


def test_boundary_targets_come_from_the_whole_video_not_the_window_edges():
    T = 1000
    v = _labeled(T, [(0, T)])
    v['feats'][:, 2] = np.arange(T) / T
    w = Windows([v], steps=100, n=60, seed=3, augment=False, gold_fraction=0.0)
    inside = 0
    for i in range(60):
        sample = w[i]
        if _start_of(sample, T) >= 20:
            inside += 1
            assert float(sample[5].numpy()[:, 0].max()) < 1e-3
            assert float(sample[5].numpy()[:-20, 1].max()) < 1e-3
    assert inside > 30


def test_padded_steps_use_the_neutral_ball_position():
    v = _labeled(50, [(10, 20)])
    w = Windows([v], steps=80, n=1, seed=0, augment=False, gold_fraction=0.0)
    f = w[0][1].numpy()
    assert np.allclose(f[50:, 3:5], 0.5)
    assert np.allclose(f[50:, [0, 1, 2, 5, 6, 7]], 0.0)


def test_every_window_start_including_the_last_is_reachable():
    T = 105
    v = _labeled(T, [(10, 20)])
    v['feats'][:, 2] = np.arange(T) / T
    w = Windows([v], steps=100, n=400, seed=1, augment=False, gold_fraction=0.0)
    assert {_start_of(w[i], T) for i in range(400)} == set(range(6))


def test_motion_augmentation_scales_every_motion_band():
    from breakpoint_vision.train.train_points import degrade_feats
    f = np.ones((200, 8), np.float32)
    scaled = []
    for seed in range(200):
        out = degrade_feats(f, np.random.default_rng(seed))
        scaled.append(np.allclose(out[:, 5], out[:, 6]) and np.allclose(out[:, 6], out[:, 7]))
    assert all(scaled)


def test_cut_mask_from_thumbnails_of_a_different_length_is_fitted_to_the_features(tmp_path):
    from breakpoint_vision.train.train_points import load_video
    T = 120
    thumbs = np.zeros((T + 1, 144, 256), np.uint8)
    thumbs[60:] = 200
    feats = np.zeros((T, 8), np.float32)
    np.savez(tmp_path / 'vid.npz', feats=feats, audio=np.zeros(T * 1600, np.float16), silver=np.zeros(T, np.float32))
    (tmp_path / 'thumbs').mkdir()
    np.save(tmp_path / 'thumbs' / 'vid.npy', thumbs)
    v = load_video(str(tmp_path / 'vid.npz'), thumbs_dir=str(tmp_path / 'thumbs'))
    assert len(v['weight']) == T
    assert v['weight'][60] == 0.0 and v['weight'][0] > 0


def test_short_reviewed_segments_get_extra_weight_and_long_ones_do_not():
    from breakpoint_vision.train.train_points import boost_short_segments
    label = np.zeros(200, np.float32)
    label[20:35] = 1.0
    label[100:180] = 1.0
    weight = np.ones(200, np.float32)
    weight[150:] = 0.3
    out = boost_short_segments(label, weight, max_steps=40, boost=3.0, margin=5)
    assert np.allclose(out[15:40], 3.0)
    assert np.allclose(out[100:150], 1.0)
    assert np.allclose(out[150:], 0.3)
    assert out[14] == 1.0 and out[40] == 1.0


def test_short_segments_with_silver_weight_are_not_boosted():
    from breakpoint_vision.train.train_points import boost_short_segments
    label = np.zeros(100, np.float32)
    label[10:20] = 1.0
    weight = np.full(100, 0.3, np.float32)
    assert np.allclose(boost_short_segments(label, weight, 40, 3.0), weight)
