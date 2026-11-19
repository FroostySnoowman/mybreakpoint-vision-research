import numpy as np

from breakpoint_vision.points.resize import area_spans, portable_area_resize, portable_area_resizes, portable_resize


def test_resize_keeps_flat_images_flat_and_shapes_right():
    img = np.full((360, 640, 3), 77, np.uint8)
    out = portable_resize(img, 512, 288)
    assert out.shape == (288, 512, 3) and np.all(out == 77)
    assert portable_resize(img[:, :, 0], 64, 36).shape == (36, 64)


def test_resize_matches_opencv_portable_kernel_on_a_gradient():
    x = np.tile(np.arange(256, dtype=np.uint8), (4, 1))
    out = portable_resize(x, 128, 2)
    assert out.shape == (2, 128)
    assert np.all(np.diff(out[0].astype(int)) >= 0)
    assert abs(int(out[0, 0]) - 0) <= 1 and abs(int(out[0, -1]) - 255) <= 1


SOURCES = [(1920, 1080), (1280, 720), (854, 480), (640, 360)]
TARGETS = [(512, 288), (64, 36)]


def test_area_resize_agrees_with_opencv_inter_area_within_one_level():
    import cv2

    rng = np.random.default_rng(11)
    for width, height in SOURCES:
        img = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
        for out_w, out_h in TARGETS:
            ours = portable_area_resize(img, out_w, out_h).astype(int)
            theirs = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA).astype(int)
            assert ours.shape == (out_h, out_w, 3)
            assert np.abs(ours - theirs).max() <= 1, (width, height, out_w, out_h)


def test_area_resize_is_the_exact_area_weighted_mean():
    img = np.array([[0, 10, 20, 30, 40]], np.uint8)
    out = portable_area_resize(img, 2, 1)
    assert out.tolist() == [[round((0 * 2 + 10 * 2 + 20 * 1) / 5), round((20 * 1 + 30 * 2 + 40 * 2) / 5)]]


def test_area_resize_rounds_half_up():
    img = np.array([[0, 1]], np.uint8)
    assert portable_area_resize(img, 1, 1).tolist() == [[1]]
    img = np.array([[0, 0, 0, 1]], np.uint8)
    assert portable_area_resize(img, 1, 1).tolist() == [[0]]


def test_area_spans_weights_sum_to_the_source_size():
    for source, destination in [(1920, 512), (854, 512), (360, 36), (480, 288), (5, 2), (512, 512)]:
        first, last, first_weight, last_weight = area_spans(source, destination)
        middle = np.maximum(last - first - 1, 0) * destination
        assert np.all(first_weight + last_weight + middle == source)
        assert first[0] == 0 and last[-1] == source - 1


def test_area_resizes_share_work_and_match_single_resizes():
    rng = np.random.default_rng(5)
    img = rng.integers(0, 256, (360, 640, 3), dtype=np.uint8)
    ball, thumb = portable_area_resizes(img, TARGETS)
    assert np.array_equal(ball, portable_area_resize(img, 512, 288))
    assert np.array_equal(thumb, portable_area_resize(img, 64, 36))


def test_area_resize_keeps_flat_images_flat_and_handles_gray():
    img = np.full((1080, 1920), 200, np.uint8)
    assert np.all(portable_area_resize(img, 512, 288) == 200)
    assert portable_area_resize(img, 64, 36).shape == (36, 64)
