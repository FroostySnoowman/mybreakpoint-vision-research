"""Import the public TennisCourtDetector dataset into this project's format.

Source: https://github.com/yastrebksv/TennisCourtDetector (dataset on Google
Drive, linked from the README). Unzipped layout:

    <root>/images/<id>.png              # 1280x720
    <root>/data_train.json              # [{"id": …, "kps": [[x,y] x 14]}, …]
    <root>/data_val.json

Their 14 keypoints are ordered by court line (see their court_reference.py):
baseline_top, baseline_bottom, left/right singles sidelines, top/bottom service
lines, then the two service Ts, each line listing its "top" (far, in broadcast
view) point first. We re-map that fixed order onto our named keypoints
(court.CourtKeypoints.names). The world court is symmetric about the net, so the
top==far convention only fixes orientation; it cannot corrupt the homography.

Output matches data/court_labels.py: <out>/court_frames/ (symlinked images) and
<out>/court_labels.json.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import List

# Their index -> our keypoint name (see module docstring).
TCD_ORDER_TO_NAME = (
    "bl_doubles_far_left",    # 0  baseline_top left  (top-left doubles corner)
    "bl_doubles_far_right",   # 1  baseline_top right
    "bl_doubles_near_left",   # 2  baseline_bottom left
    "bl_doubles_near_right",  # 3  baseline_bottom right
    "sl_far_left",            # 4  left singles sideline, top end
    "sl_near_left",           # 5  left singles sideline, bottom end
    "sl_far_right",           # 6  right singles sideline, top end
    "sl_near_right",          # 7  right singles sideline, bottom end
    "svc_far_left",           # 8  top service line x left singles
    "svc_far_right",          # 9  top service line x right singles
    "svc_near_left",          # 10 bottom service line x left singles
    "svc_near_right",         # 11 bottom service line x right singles
    "center_svc_far",         # 12 top service T
    "center_svc_near",        # 13 bottom service T
)


@dataclass
class CourtImportReport:
    images: int
    skipped: int
    labels_json: str
    frames_root: str


def _find_split_files(src_dir: str) -> List[str]:
    found = []
    for name in ("data_train.json", "data_val.json"):
        p = os.path.join(src_dir, name)
        if os.path.isfile(p):
            found.append(p)
    return found


def import_court_dataset(src_dir: str, out_dir: str, copy: bool = False) -> CourtImportReport:
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"dataset folder not found: {src_dir}")
    images_dir = os.path.join(src_dir, "images")
    split_files = _find_split_files(src_dir)
    if not split_files or not os.path.isdir(images_dir):
        raise FileNotFoundError(
            f"expected {src_dir} to contain images/ plus data_train.json / data_val.json "
            "(the unzipped TennisCourtDetector dataset)."
        )

    frames_root = os.path.join(out_dir, "court_frames")
    labels_out = os.path.join(out_dir, "court_labels.json")
    os.makedirs(frames_root, exist_ok=True)

    entries = []
    skipped = 0
    for split in split_files:
        with open(split) as fh:
            records = json.load(fh)
        for rec in records:
            img_name = f"{rec['id']}.png"
            src_img = os.path.join(images_dir, img_name)
            kps = rec.get("kps") or []
            if not os.path.isfile(src_img) or len(kps) != len(TCD_ORDER_TO_NAME):
                skipped += 1
                continue
            keypoints = {}
            for i, name in enumerate(TCD_ORDER_TO_NAME):
                x, y = float(kps[i][0]), float(kps[i][1])
                keypoints[name] = [x, y]
            dst_img = os.path.join(frames_root, img_name)
            if os.path.lexists(dst_img):
                os.remove(dst_img)
            if copy:
                import shutil

                shutil.copy2(src_img, dst_img)
            else:
                os.symlink(os.path.abspath(src_img), dst_img)
            entries.append({"file": img_name, "width": 1280, "height": 720, "keypoints": keypoints})

    if not entries:
        raise RuntimeError("found the dataset JSONs but could not match any records to images.")

    with open(labels_out, "w") as fh:
        json.dump({"images": entries}, fh)

    return CourtImportReport(
        images=len(entries),
        skipped=skipped,
        labels_json=labels_out,
        frames_root=frames_root,
    )
