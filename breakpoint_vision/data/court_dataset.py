"""Court keypoint dataset.

Yields (image_tensor, target_heatmaps) where image is (3, H, W) normalized to
[0,1] and target is (K[+1], H, W): one Gaussian channel per court keypoint, in
the canonical order of `court.CourtKeypoints.names`. If the model config has a
center channel, an extra channel is rendered at the mean of the visible
keypoints (an easy-to-learn anchor that stabilizes training; it is not used for
the homography).

Label format (one JSON file, the common keypoint-annotation shape):

    {
      "images": [
        {
          "file": "game1/clip1/0000.jpg",   # relative to frames_root
          "width": 1280, "height": 720,      # optional; else read from image
          "keypoints": {                      # any subset of the 14 names
            "bl_doubles_near_left": [x, y],
            "sl_near_right": [x, y],
            …
          }
        },
        …
      ]
    }

Missing/occluded keypoints are simply omitted (their target channel is all
zeros, i.e. "not visible here"). Coordinates are in the ORIGINAL image pixels;
they are rescaled to the model input size with the image.

`groups()` returns the source video each frame was sampled from (the filename
stem before the frame number), because frames of the same court from the same
camera are not independent samples: validation must hold out whole videos, not
random frames (see `split.py`).
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import CourtModelConfig
from ..court import CourtKeypoints
from .augment import HORIZONTAL_FLIP_KEYPOINTS, AugmentConfig, Augmenter
from .court_labels import CourtSample, read_court_labels
from .heatmap import render_multi

__all__ = ["CourtDataset", "CourtSample", "read_court_labels"]


def _cv2():
    import cv2

    return cv2


class CourtDataset(Dataset):
    def __init__(
        self,
        cfg: CourtModelConfig,
        frames_root: str,
        labels_json: str,
        augment: Optional[AugmentConfig] = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.frames_root = frames_root
        self.samples = [s for s in read_court_labels(labels_json) if s.keypoints]
        self.names = CourtKeypoints().names
        self.augmenter = Augmenter(augment, seed) if augment is not None and augment.enabled else None
        self.epoch = 0
        index_of = {name: i for i, name in enumerate(self.names)}
        self.flip_index_map = [index_of[HORIZONTAL_FLIP_KEYPOINTS[name]] for name in self.names]

    def __len__(self) -> int:
        return len(self.samples)

    def set_epoch(self, epoch: int) -> None:
        """Vary augmentation between epochs while keeping it index-derived."""
        self.epoch = int(epoch)

    def groups(self) -> Sequence[str]:
        """The source video of each frame: `<video>_<frame>.png` -> `<video>`."""
        return [os.path.basename(s.file).rsplit("_", 1)[0] for s in self.samples]

    def __getitem__(self, idx: int):
        cv2 = _cv2()
        sample = self.samples[idx]
        path = os.path.join(self.frames_root, sample.file)
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"court frame not found: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        src_h, src_w = img.shape[:2]
        if sample.width:
            src_w = sample.width
        if sample.height:
            src_h = sample.height

        img_resized = cv2.resize(img, (self.cfg.input_width, self.cfg.input_height))

        sx = self.cfg.input_width / src_w
        sy = self.cfg.input_height / src_h

        centers: List[Optional[Tuple[float, float]]] = []
        for name in self.names:
            if name in sample.keypoints:
                x, y = sample.keypoints[name]
                centers.append((x * sx, y * sy))
            else:
                centers.append(None)

        if self.augmenter is not None:
            self.augmenter.seed_for_item(idx, self.epoch)
            frames, centers = self.augmenter(img_resized[None], centers, self.flip_index_map)
            img_resized = frames[0]

        tensor = torch.from_numpy(np.ascontiguousarray(img_resized)).float().div(255.0).permute(2, 0, 1)

        if self.cfg.center_channel:
            visible = [c for c in centers if c is not None]
            if visible:
                mx = float(np.mean([c[0] for c in visible]))
                my = float(np.mean([c[1] for c in visible]))
                centers.append((mx, my))
            else:
                centers.append(None)

        target = render_multi(self.cfg.input_height, self.cfg.input_width, centers, self.cfg.gaussian_sigma)
        return tensor, torch.from_numpy(target)
