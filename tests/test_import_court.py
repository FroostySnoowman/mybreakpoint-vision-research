"""The court-import mapping must be geometrically correct.

The one thing that can silently corrupt court training is a wrong mapping from
the public TennisCourtDetector keypoint order to our canonical CourtKeypoints
order. We guard it the only way that actually proves correctness: project the
known world court through a camera homography to synthesize image keypoints in
*their* index order, import them, reorder into *our* names, and require the
world<->image homography to solve back with ~zero error. A bad permutation
cannot pass this.
"""

import json

from breakpoint_vision.court import (
    HALF_DOUBLES,
    HALF_LENGTH,
    CourtKeypoints,
)
from breakpoint_vision.data.import_court import TCD_ORDER_TO_NAME, import_court_dataset
from breakpoint_vision.data.court_labels import read_court_labels
from breakpoint_vision.geometry import estimate_homography, reprojection_error


def _project(h, wp):
    import numpy as np

    v = h @ np.array([wp[0], wp[1], 1.0])
    return [float(v[0] / v[2]), float(v[1] / v[2])]


def test_import_court_mapping_is_geometrically_consistent(tmp_path):
    kp = CourtKeypoints()
    world = kp.world_points()

    world_corners = [
        (-HALF_DOUBLES, -HALF_LENGTH),
        (HALF_DOUBLES, -HALF_LENGTH),
        (HALF_DOUBLES, HALF_LENGTH),
        (-HALF_DOUBLES, HALF_LENGTH),
    ]
    image_corners = [(360, 620), (920, 620), (1140, 240), (140, 240)]
    h_world_to_image = estimate_homography(world_corners, image_corners)

    kps = [_project(h_world_to_image, world[TCD_ORDER_TO_NAME[i]]) for i in range(14)]

    src = tmp_path / "tcd"
    (src / "images").mkdir(parents=True)
    (src / "images" / "img0.png").write_bytes(b"\x89PNG\r\n")  # importer only needs the file to exist
    (src / "data_train.json").write_text(json.dumps([{"id": "img0", "kps": kps}]))
    (src / "data_val.json").write_text(json.dumps([]))

    out = tmp_path / "out"
    report = import_court_dataset(str(src), str(out))
    assert report.images == 1

    sample = read_court_labels(report.labels_json)[0]
    src_pts = [sample.keypoints[name] for name in kp.names]
    dst_pts = kp.ordered_world_array()
    h = estimate_homography(src_pts, dst_pts)
    assert reprojection_error(h, src_pts, dst_pts) < 0.01


def test_tcd_order_covers_all_our_keypoints():
    assert set(TCD_ORDER_TO_NAME) == set(CourtKeypoints().names)
    assert len(TCD_ORDER_TO_NAME) == 14
