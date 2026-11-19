import numpy as np

from breakpoint_vision.points.features import N_FEATURES, STEP_SECONDS, band_motion, motion_score, step_features, step_labels


def test_shape_and_defaults_for_an_empty_track():
    f = step_features([], [], [], [], 1280, 720, duration=1.0)
    assert f.shape == (10, N_FEATURES)
    assert np.all(f[:, 0] == 0) and np.all(f[:, 3] == 0.5) and np.all(f[:, 4] == 0.5)


def test_ball_confidence_seen_fraction_and_position():
    t = [0.00, 0.033, 0.066]
    f = step_features(t, [640, 640, 640], [360, 360, 360], [0.9, 0.2, 0.8], 1280, 720, duration=0.1)
    assert f[0, 0] == np.float32(0.9)
    assert abs(f[0, 1] - 2 / 3) < 1e-6
    assert abs(f[0, 3] - 0.5) < 1e-6 and abs(f[0, 4] - 0.5) < 1e-6


def test_speed_uses_consecutive_seen_samples_only():
    t = [0.0, 0.033, 0.066, 0.099]
    x = [0, 128, 256, 1000]
    conf = [0.9, 0.9, 0.1, 0.9]
    f = step_features(t, x, [0, 0, 0, 0], conf, 1280, 720, duration=0.1)
    assert abs(f[0, 2] - min(128 / 1280 / 0.033, 3.0)) < 1e-4


def test_speed_spans_step_boundaries_and_is_clipped():
    f = step_features([0.095, 0.105], [0, 1280], [0, 0], [0.9, 0.9], 1280, 720, duration=0.2)
    assert f[1, 2] == np.float32(3.0)


def test_large_gaps_do_not_produce_speed():
    f = step_features([0.0, 0.5], [0, 100], [0, 0], [0.9, 0.9], 1280, 720, duration=1.0)
    assert np.all(f[:, 2] == 0)


def test_band_motion_is_averaged_per_step_and_clipped():
    motion = np.array([[0.2, 0.0, 0.1], [0.4, 0.2, 0.1], [5.0, 0.3, 0.0]], np.float32)
    f = step_features([0.0, 0.05, 0.12], [0, 0, 0], [0, 0, 0], [0, 0, 0], 1280, 720, duration=0.2, motion=motion)
    assert np.allclose(f[0, 5:8], [0.3, 0.1, 0.1]) and np.allclose(f[1, 5:8], [1.0, 0.3, 0.0])


def test_band_motion_localizes_movement():
    a = np.full((36, 64), 0.4, np.float32)
    b = a.copy()
    b[2:8, 10:20] = 0.9
    m = band_motion(a, b)
    assert m[0] > 0 and m[1] == 0 and m[2] == 0


def test_motion_score_is_exposure_invariant():
    rng = np.random.default_rng(0)
    a = rng.uniform(0.2, 0.6, (36, 64))
    b = a.copy()
    b[10:20, 10:20] += 0.2
    bright = motion_score(a, b)
    dark = motion_score(a * 0.1, b * 0.1)
    assert bright > 0 and abs(bright - dark) / bright < 0.45


def test_labels_mark_step_centers_inside_play_segments():
    y = step_labels([{'start': 0.1, 'end': 0.3, 'kind': 'point'}, {'start': 0.5, 'end': 0.7, 'kind': 'warmup'}], duration=1.0)
    assert y.tolist() == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert STEP_SECONDS == 0.1
