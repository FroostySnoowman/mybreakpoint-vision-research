"""Court keypoint label parsing + target rendering (no torch model needed).

Locks in the label format and the keypoint->heatmap mapping that the court
trainer depends on. The synthetic-court smoke path (train_court --smoke) covers
the training loop itself.
"""

import json

from breakpoint_vision.court import CourtKeypoints
from breakpoint_vision.data.court_labels import read_court_labels


def test_read_court_labels_parses_named_keypoints(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(
        json.dumps(
            {
                "images": [
                    {
                        "file": "g1/0.jpg",
                        "width": 1280,
                        "height": 720,
                        "keypoints": {
                            "bl_doubles_near_left": [100, 650],
                            "sl_far_right": [780, 120],
                        },
                    }
                ]
            }
        )
    )
    samples = read_court_labels(str(p))
    assert len(samples) == 1
    s = samples[0]
    assert s.file == "g1/0.jpg"
    assert s.width == 1280 and s.height == 720
    assert s.keypoints["bl_doubles_near_left"] == (100.0, 650.0)
    assert set(s.keypoints).issubset(set(CourtKeypoints().names))


def test_read_court_labels_omits_empty_keypoints(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(
        json.dumps({"images": [{"file": "a.jpg", "keypoints": {"sl_near_left": [1, 2], "sl_near_right": []}}]})
    )
    s = read_court_labels(str(p))[0]
    assert "sl_near_left" in s.keypoints
    assert "sl_near_right" not in s.keypoints
