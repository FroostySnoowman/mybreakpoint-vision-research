"""Training-time augmentation moves the label with the image.

An augmentation that transforms pixels but not the label is worse than no
augmentation: it actively teaches the network the wrong position. These tests pin
three ways that can go wrong: geometry drifting from the label, a mirrored court
keeping its left/right keypoint names, and a frame stack being jittered per frame
so that flicker becomes a "ball" cue.
"""

import numpy as np

from breakpoint_vision.court import CourtKeypoints
from breakpoint_vision.data.augment import HORIZONTAL_FLIP_KEYPOINTS, AugmentConfig, Augmenter

H, W = 96, 160


def _geometry_only() -> AugmentConfig:
    """Geometry with no photometric noise, so pixels can be compared exactly."""
    return AugmentConfig(
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue_degrees=0.0,
        noise_std=0.0,
        blur_prob=0.0,
        jpeg_prob=0.0,
    )


def _dot_frame(x: int, y: int) -> np.ndarray:
    frame = np.zeros((H, W, 3), dtype=np.uint8)
    frame[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = 255
    return frame


def _brightest(frame: np.ndarray):
    gray = frame.astype(np.float32).sum(axis=2)
    if gray.max() <= 0:
        return None
    y, x = np.unravel_index(int(gray.argmax()), gray.shape)
    return float(x), float(y)


def test_the_label_follows_the_pixels_it_labels():
    """The whole point: wherever the dot lands, the point must land too."""
    cfg = _geometry_only()
    checked = 0
    for seed in range(40):
        aug = Augmenter(cfg, seed=seed)
        x, y = 40 + seed % 60, 30 + seed % 40
        frames, points = aug(_dot_frame(x, y)[None], [(float(x), float(y))])
        if points[0] is None:
            continue
        found = _brightest(frames[0])
        assert found is not None
        assert abs(found[0] - points[0][0]) <= 3.0, (seed, found, points[0])
        assert abs(found[1] - points[0][1]) <= 3.0, (seed, found, points[0])
        checked += 1
    assert checked >= 30, f"too few usable cases ({checked}) to call this a test"


def test_a_point_pushed_out_of_frame_becomes_not_visible():
    """Clamping it to the edge would train the model to see a ball on the border."""
    cfg = AugmentConfig(
        horizontal_flip_prob=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue_degrees=0.0,
        noise_std=0.0,
        blur_prob=0.0,
        jpeg_prob=0.0,
        scale_jitter=0.0,
        rotation_degrees=0.0,
        shift_fraction=0.9,
    )
    dropped = 0
    for seed in range(40):
        _, points = Augmenter(cfg, seed=seed)(np.zeros((1, H, W, 3), np.uint8), [(2.0, 2.0)])
        if points[0] is None:
            dropped += 1
        else:
            assert 0 <= points[0][0] < W and 0 <= points[0][1] < H
    assert dropped > 0, "a large shift should sometimes push the point out of frame"


def test_every_frame_in_a_stack_gets_the_same_global_transform():
    """Per-frame global jitter would make flicker a 'ball' cue.

    Sensor noise is excluded because it is deliberately per-frame (see the module
    docstring), so this checks the transforms that must not vary.
    """
    frame = np.tile(np.arange(W, dtype=np.uint8)[None, :, None], (H, 1, 3))
    stack = np.stack([frame, frame, frame], axis=0)
    frames, _ = Augmenter(AugmentConfig(noise_std=0.0), seed=5)(stack, [None])
    assert np.array_equal(frames[0], frames[1])
    assert np.array_equal(frames[1], frames[2])


def test_sensor_noise_is_independent_per_frame():
    """Identical noise on every frame would be a static pattern, not sensor noise."""
    stack = np.full((3, H, W, 3), 120, dtype=np.uint8)
    cfg = AugmentConfig(
        horizontal_flip_prob=0.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue_degrees=0.0,
        scale_jitter=0.0,
        shift_fraction=0.0,
        rotation_degrees=0.0,
        blur_prob=0.0,
        jpeg_prob=0.0,
        noise_std=0.05,
    )
    frames, _ = Augmenter(cfg, seed=1)(stack, [None])
    assert not np.array_equal(frames[0], frames[1])


def test_flip_map_is_a_total_involution_over_the_court_keypoints():
    names = CourtKeypoints().names
    assert set(HORIZONTAL_FLIP_KEYPOINTS) == set(names)
    for name in names:
        assert HORIZONTAL_FLIP_KEYPOINTS[HORIZONTAL_FLIP_KEYPOINTS[name]] == name
    assert HORIZONTAL_FLIP_KEYPOINTS["center_svc_near"] == "center_svc_near"
    assert HORIZONTAL_FLIP_KEYPOINTS["center_svc_far"] == "center_svc_far"


def test_mirroring_a_court_swaps_left_and_right_keypoints():
    """Without the swap the homography's point correspondences are wrong."""
    cfg = AugmentConfig(
        horizontal_flip_prob=1.0,
        brightness=0.0,
        contrast=0.0,
        saturation=0.0,
        hue_degrees=0.0,
        noise_std=0.0,
        blur_prob=0.0,
        jpeg_prob=0.0,
        scale_jitter=0.0,
        rotation_degrees=0.0,
        shift_fraction=0.0,
    )
    names = CourtKeypoints().names
    index_of = {name: i for i, name in enumerate(names)}
    flip_index_map = [index_of[HORIZONTAL_FLIP_KEYPOINTS[name]] for name in names]

    points = [(10.0 + 5 * i, 20.0 + i) for i in range(len(names))]
    _, moved = Augmenter(cfg, seed=0)(np.zeros((1, H, W, 3), np.uint8), points, flip_index_map)

    for i, name in enumerate(names):
        partner = index_of[HORIZONTAL_FLIP_KEYPOINTS[name]]
        expected_x = W - 1.0 - points[partner][0]
        assert moved[i] is not None
        assert abs(moved[i][0] - expected_x) < 1e-6, name
        assert abs(moved[i][1] - points[partner][1]) < 1e-6, name


def test_disabled_augmentation_is_a_pass_through():
    """Validation must see untransformed data or its loss is not comparable."""
    stack = np.full((2, H, W, 3), 123, dtype=np.uint8)
    frames, points = Augmenter(AugmentConfig(enabled=False), seed=0)(stack, [(1.0, 2.0)])
    assert np.array_equal(frames, stack)
    assert points == [(1.0, 2.0)]


def test_augmentation_is_derived_from_the_sample_index_not_the_draw_order():
    """A DataLoader forks the dataset per worker, so every worker inherits the same
    generator state and draws the same transforms, so an epoch shows the network
    far fewer distinct views than intended. Deriving the draw from (seed, index, epoch)
    makes it reproducible, worker-count independent, and still varied per epoch."""
    cfg = _geometry_only()

    def draw(seed: int, index: int, epoch: int):
        aug = Augmenter(cfg, seed=seed)
        aug.seed_for_item(index, epoch)
        _, moved = aug(_dot_frame(80, 48)[None], [(80.0, 48.0)])
        return moved[0]

    assert draw(0, 5, 0) == draw(0, 5, 0)
    assert draw(0, 5, 0) != draw(0, 6, 0)
    assert draw(0, 5, 0) != draw(0, 5, 1)


def test_the_same_index_yields_distinct_views_across_an_epoch_sweep():
    cfg = _geometry_only()
    views = set()
    for epoch in range(8):
        aug = Augmenter(cfg, seed=7)
        aug.seed_for_item(3, epoch)
        _, moved = aug(_dot_frame(80, 48)[None], [(80.0, 48.0)])
        views.add(moved[0])
    assert len(views) >= 6
