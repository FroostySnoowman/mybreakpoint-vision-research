"""Annotation format and import/export.

The ball-tracking label format is intentionally the same CSV shape used by the
public TrackNet tennis dataset, so that dataset (and the common labeling tools
that emit it) drops in without conversion:

    clip,frame,visible,x,y
    game1_clip1,0,1,412,233
    game1_clip1,1,1,418,231
    game1_clip1,2,0,-1,-1
    …

- `clip` groups frames that are temporally contiguous (the model stacks frames
  only within a clip, never across a boundary).
- `visible` is 1/0; when 0, x and y are ignored (store -1,-1 by convention).
- Coordinates are in the ORIGINAL frame pixels; the dataset rescales to the
  model input size and adjusts labels accordingly.

Frames are expected as image files named `<clip>/<frame>.jpg` (or .png) under a
`frames_root`, or the dataset can decode them on the fly from source videos via
`extract_frames`.
"""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class BallLabel:
    clip: str
    frame: int
    visible: bool
    x: float
    y: float


def read_ball_labels(csv_path: str) -> List[BallLabel]:
    labels: List[BallLabel] = []
    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        required = {"clip", "frame", "visible", "x", "y"}
        if not required.issubset({c.strip() for c in (reader.fieldnames or [])}):
            raise ValueError(f"label CSV must have columns {sorted(required)}, got {reader.fieldnames}")
        for row in reader:
            visible = str(row["visible"]).strip() in ("1", "true", "True")
            labels.append(
                BallLabel(
                    clip=row["clip"].strip(),
                    frame=int(row["frame"]),
                    visible=visible,
                    x=float(row["x"]),
                    y=float(row["y"]),
                )
            )
    return labels


def write_ball_labels(csv_path: str, labels: List[BallLabel]) -> None:
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["clip", "frame", "visible", "x", "y"])
        for lb in labels:
            writer.writerow([lb.clip, lb.frame, 1 if lb.visible else 0, lb.x if lb.visible else -1, lb.y if lb.visible else -1])


def group_by_clip(labels: List[BallLabel]) -> Dict[str, List[BallLabel]]:
    groups: Dict[str, List[BallLabel]] = {}
    for lb in labels:
        groups.setdefault(lb.clip, []).append(lb)
    for clip in groups:
        groups[clip].sort(key=lambda l: l.frame)
    return groups


def extract_frames(video_path: str, out_dir: str, clip_name: Optional[str] = None, every: int = 1) -> int:
    """Decode a video to numbered JPGs under out_dir/clip_name/. Returns count.

    Kept here (not in the dataset) so annotation prep is a separate, explicit
    step. Requires opencv.
    """
    import cv2

    clip_name = clip_name or os.path.splitext(os.path.basename(video_path))[0]
    dst = os.path.join(out_dir, clip_name)
    os.makedirs(dst, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    idx = 0
    saved = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            cv2.imwrite(os.path.join(dst, f"{idx}.jpg"), frame)
            saved += 1
        idx += 1
    cap.release()
    return saved
