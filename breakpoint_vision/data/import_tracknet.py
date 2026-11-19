"""Import the public TrackNet tennis dataset into this project's format.

The public dataset (e.g. the Kaggle "tracknet-tennis" upload, and the original
TrackNetV2 release) is laid out as:

    <root>/game1/Clip1/0000.jpg, 0001.jpg, …
    <root>/game1/Clip1/Label.csv     # columns: file name, visibility, x-coordinate, y-coordinate[, status]
    <root>/game1/Clip2/…
    <root>/game2/…

Our trainer wants a single combined CSV (clip,frame,visible,x,y) and frames
reachable at <frames_root>/<clip>/<int>.jpg (see data/annotate.py). This module
converts one into the other with no manual editing:

  - each Clip folder becomes one clip, named "<game>_<clip>" (path-derived)
  - the per-clip Label.csv rows are merged into one labels.csv in our format
  - each image is linked (default) or copied to <frames_root>/<clip>/<frameidx>.jpg

Column names vary between uploads, so we resolve them flexibly.
"""

from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .annotate import BallLabel, write_ball_labels

_IMG_EXTS = (".jpg", ".jpeg", ".png")

_VIS_KEYS = ("visibility", "visible", "vis")
_X_KEYS = ("x-coordinate", "x", "x-coord", "xcoord", "x coordinate")
_Y_KEYS = ("y-coordinate", "y", "y-coord", "ycoord", "y coordinate")
_NAME_KEYS = ("file name", "filename", "file", "name", "frame")


@dataclass
class ImportReport:
    clips: int
    frames: int
    visible_frames: int
    labels_csv: str
    frames_root: str
    skipped: List[str]


def _norm(s: str) -> str:
    return s.strip().lower().replace("_", " ").replace("-", "-")


def _pick(fieldnames: List[str], candidates: Tuple[str, ...]) -> Optional[str]:
    norm_map = {_norm(f): f for f in fieldnames}
    for cand in candidates:
        if cand in norm_map:
            return norm_map[cand]
    for key, original in norm_map.items():
        if any(cand in key for cand in candidates):
            return original
    return None


def _frame_index_from_name(name: str, fallback: int) -> int:
    stem = os.path.splitext(os.path.basename(name.strip()))[0]
    digits = re.findall(r"\d+", stem)
    if digits:
        return int(digits[-1])
    return fallback


def _find_label_csvs(root: str) -> List[str]:
    found: List[str] = []
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".csv") and "label" in f.lower():
                found.append(os.path.join(dirpath, f))
    return sorted(found)


def _clip_name(root: str, label_csv: str) -> str:
    rel = os.path.relpath(os.path.dirname(label_csv), root)
    if rel == ".":
        rel = os.path.splitext(os.path.basename(label_csv))[0]
    parts = [p for p in re.split(r"[\\/]+", rel) if p and p != "."]
    name = "_".join(parts) if parts else "clip"
    return re.sub(r"[^A-Za-z0-9_]+", "_", name)


def _locate_image(clip_dir: str, name: str, frame_idx: int) -> Optional[str]:
    direct = os.path.join(clip_dir, name.strip())
    if os.path.isfile(direct):
        return direct
    stem = os.path.splitext(os.path.basename(name.strip()))[0]
    for ext in _IMG_EXTS:
        cand = os.path.join(clip_dir, stem + ext)
        if os.path.isfile(cand):
            return cand
    for width in (4, 5, 6, 3):
        for ext in _IMG_EXTS:
            cand = os.path.join(clip_dir, f"{frame_idx:0{width}d}{ext}")
            if os.path.isfile(cand):
                return cand
    return None


def _place(src_img: str, dst_img: str, copy: bool) -> None:
    os.makedirs(os.path.dirname(dst_img), exist_ok=True)
    if os.path.lexists(dst_img):
        os.remove(dst_img)
    if copy:
        import shutil

        shutil.copy2(src_img, dst_img)
    else:
        os.symlink(os.path.abspath(src_img), dst_img)


def import_tracknet(src_dir: str, out_dir: str, copy: bool = False) -> ImportReport:
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"dataset folder not found: {src_dir}")

    label_csvs = _find_label_csvs(src_dir)
    if not label_csvs:
        raise FileNotFoundError(
            f"no Label.csv files found under {src_dir}. Point src_dir at the unzipped "
            "TrackNet dataset folder (the one containing game1/, game2/, …)."
        )

    frames_root = os.path.join(out_dir, "frames")
    labels_out = os.path.join(out_dir, "labels.csv")
    os.makedirs(frames_root, exist_ok=True)

    all_labels: List[BallLabel] = []
    skipped: List[str] = []
    clips = 0
    visible = 0

    for label_csv in label_csvs:
        clip_dir = os.path.dirname(label_csv)
        clip = _clip_name(src_dir, label_csv)
        with open(label_csv, newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames or []
            x_col = _pick(fields, _X_KEYS)
            y_col = _pick(fields, _Y_KEYS)
            vis_col = _pick(fields, _VIS_KEYS)
            name_col = _pick(fields, _NAME_KEYS)
            if x_col is None or y_col is None:
                skipped.append(f"{label_csv} (no x/y columns in {fields})")
                continue

            clip_had_frame = False
            for i, row in enumerate(reader):
                name = row.get(name_col, "") if name_col else ""
                frame_idx = _frame_index_from_name(name, i)
                src_img = _locate_image(clip_dir, name, frame_idx)
                if src_img is None:
                    continue
                try:
                    vis_raw = str(row.get(vis_col, "1")).strip() if vis_col else "1"
                    is_visible = vis_raw not in ("0", "", "false", "False")
                    x = float(row[x_col]) if row.get(x_col) not in (None, "") else -1.0
                    y = float(row[y_col]) if row.get(y_col) not in (None, "") else -1.0
                except (ValueError, KeyError):
                    continue
                if x < 0 or y < 0:
                    is_visible = False
                ext = os.path.splitext(src_img)[1].lower()
                dst_img = os.path.join(frames_root, clip, f"{frame_idx}{ext}")
                _place(src_img, dst_img, copy)
                all_labels.append(BallLabel(clip=clip, frame=frame_idx, visible=is_visible, x=x, y=y))
                clip_had_frame = True
                if is_visible:
                    visible += 1
            if clip_had_frame:
                clips += 1

    if not all_labels:
        raise RuntimeError(
            "found Label.csv files but could not match any rows to images. "
            "Check that image files sit next to their Label.csv."
        )

    write_ball_labels(labels_out, all_labels)
    return ImportReport(
        clips=clips,
        frames=len(all_labels),
        visible_frames=visible,
        labels_csv=labels_out,
        frames_root=frames_root,
        skipped=skipped,
    )
