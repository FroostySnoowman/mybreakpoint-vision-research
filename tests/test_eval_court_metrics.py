"""Court plane error: zero for the labeled court, failures counted as misses."""

import cv2
import numpy as np

from evaluation.court_detection.metrics import (
    FAILURE_CM, NAMES, WORLD, TemporalMedian, label_homography, plane_error_cm, plane_summary, points_only_fit, project,
)

CORNERS_WORLD = np.float32([[-5.485, -11.885], [5.485, -11.885], [5.485, 11.885], [-5.485, 11.885]])
CORNERS_IMAGE = np.float32([[60, 270], [452, 270], [330, 40], [182, 40]])


def synthetic_row():
    Hm = cv2.getPerspectiveTransform(CORNERS_WORLD, CORNERS_IMAGE).astype(np.float64)
    img, _ = project(Hm, WORLD)
    return Hm, {'keypoints': {n: [float(x), float(y)] for n, (x, y) in zip(NAMES, img)}}


def test_label_homography_reproduces_the_labeled_court():
    Hm, row = synthetic_row()
    H = label_homography(row)
    assert np.allclose(H / H[2, 2], Hm / Hm[2, 2], atol=1e-6)


def test_predicting_the_labeled_homography_gives_zero_error():
    Hm, row = synthetic_row()
    assert plane_error_cm(label_homography(row), row) < 1e-9
    assert plane_error_cm(Hm, row) < 1e-3


def test_an_offset_prediction_has_a_positive_error():
    Hm, row = synthetic_row()
    shifted = np.array([[1, 0, 4], [0, 1, 0], [0, 0, 1]], np.float64) @ Hm
    assert plane_error_cm(shifted, row) > 1.0


def test_missing_or_singular_predictions_are_failures():
    _, row = synthetic_row()
    assert plane_error_cm(None, row) == FAILURE_CM
    assert plane_error_cm(np.zeros((3, 3)), row) == FAILURE_CM


def test_points_only_fit_needs_four_confident_keypoints():
    Hm, row = synthetic_row()
    img, _ = project(Hm, WORLD)
    pk = np.c_[img, np.full(14, 0.9)].astype(np.float32)
    assert plane_error_cm(points_only_fit(pk), row) < 0.5
    pk[3:, 2] = 0.1
    assert points_only_fit(pk) is None


def test_summary_counts_failures_as_misses():
    s = plane_summary([0.0, 5.0, 20.0, FAILURE_CM])
    assert s['frames'] == 4 and s['failures'] == 1
    assert s['within_15cm'] == 0.5
    assert s['p50_cm'] == 12.5


def test_temporal_median_waits_for_enough_frames_and_ignores_an_outlier():
    agg = TemporalMedian(window=12, min_frames=3)
    base = np.zeros((14, 3), np.float32)
    base[:, 0], base[:, 1], base[:, 2] = np.arange(14), 10.0, 0.9
    outlier = base.copy()
    outlier[0, :2] = (300.0, 200.0)
    assert agg.add(base) is None
    assert agg.add(outlier) is None
    out = agg.add(base)
    assert out[0, 0] == 0.0 and out[0, 2] == 1.0
    low = base.copy()
    low[5, 2] = 0.1
    agg2 = TemporalMedian(window=12, min_frames=3)
    for _ in range(3):
        last = agg2.add(low)
    assert last[5, 2] == 0.0 and last[4, 2] == 1.0
