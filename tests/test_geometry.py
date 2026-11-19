import numpy as np

from breakpoint_vision.geometry import (
    apply_homography,
    estimate_homography,
    estimate_homography_robust,
    reprojection_error,
)


def test_identity_homography_roundtrip():
    src = [(0, 0), (1, 0), (1, 1), (0, 1)]
    dst = [(0, 0), (1, 0), (1, 1), (0, 1)]
    h = estimate_homography(src, dst)
    mapped = apply_homography(h, src)
    assert np.allclose(mapped, np.array(dst), atol=1e-6)


def test_known_perspective_recovered():
    # A synthetic perspective: map a unit square to a trapezoid, then check the
    # solved homography reproduces the mapping with near-zero error.
    src = [(100, 100), (500, 120), (480, 400), (120, 380)]
    dst = [(-4.115, -11.885), (4.115, -11.885), (4.115, 11.885), (-4.115, 11.885)]
    h = estimate_homography(src, dst)
    err = reprojection_error(h, src, dst)
    assert err < 1e-6


def _synthetic_court_view():
    """14 exact image<->world correspondences from a plausible camera."""
    from breakpoint_vision.court import CourtKeypoints

    world = np.asarray(CourtKeypoints().ordered_world_array(), dtype=np.float64)
    h_true = np.array([[42.0, 6.0, 640.0], [1.5, -33.0, 500.0], [0.001, -0.0032, 1.0]])
    homo = np.hstack([world, np.ones((world.shape[0], 1))])
    img = (h_true @ homo.T).T
    return (img[:, :2] / img[:, 2:3]).tolist(), world.tolist()


def test_a_single_bad_keypoint_poisons_the_least_squares_fit():
    """The reason a robust fit is needed: least squares spreads one keypoint's error
    over every other correspondence, and the reported error does not reveal it."""
    src, dst = _synthetic_court_view()
    src[3] = [src[3][0] + 60.0, src[3][1] - 40.0]

    h = estimate_homography(src, dst)
    clean = [i for i in range(len(src)) if i != 3]
    poisoned = reprojection_error(h, [src[i] for i in clean], [dst[i] for i in clean])
    assert poisoned > 0.05


def test_robust_homography_discards_outliers_and_recovers_the_true_fit():
    src, dst = _synthetic_court_view()
    for i, (dx, dy) in [(2, (-70.0, 55.0)), (7, (40.0, 90.0)), (11, (-30.0, -60.0))]:
        src[i] = [src[i][0] + dx, src[i][1] + dy]

    h, inliers = estimate_homography_robust(src, dst)
    assert set(inliers).isdisjoint({2, 7, 11})
    assert len(inliers) >= 10
    err = reprojection_error(h, [src[i] for i in inliers], [dst[i] for i in inliers])
    assert err < 1e-6


def test_robust_homography_keeps_every_point_when_all_agree():
    src, dst = _synthetic_court_view()
    h, inliers = estimate_homography_robust(src, dst)
    assert inliers == list(range(len(src)))
    assert reprojection_error(h, src, dst) < 1e-6


def test_robust_homography_falls_back_rather_than_failing_on_four_points():
    src, dst = _synthetic_court_view()
    h, inliers = estimate_homography_robust(src[:4], dst[:4])
    assert inliers == [0, 1, 2, 3]
    assert np.all(np.isfinite(h))


def test_homography_needs_four_points():
    try:
        estimate_homography([(0, 0), (1, 0)], [(0, 0), (1, 0)])
        assert False, "expected ValueError"
    except ValueError:
        pass
