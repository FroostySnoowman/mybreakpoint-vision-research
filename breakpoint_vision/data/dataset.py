"""Ball-heatmap dataset.

Yields (frame_stack, target_heatmap) where frame_stack is (T*3, H, W): T
consecutive frames concatenated on the channel axis, and target_heatmap is
(1, H, W) with a Gaussian blob at the ball position in the *last* frame of the
stack.

Two sources:
  - SyntheticBallDataset: generates clips in memory (no files). Used by the
    smoke run when no data is available.
  - DiskBallDataset: reads frames + a TrackNet-format CSV (see annotate.py).

`DiskBallDataset` exposes `groups()`: the clip each window came from, because
consecutive windows share all but one frame, so validation must be split by clip
rather than by window (see `split.py`).

Augmentation applies to the training split only; the validation split must stay
untransformed or its loss is not comparable between epochs.
"""

from __future__ import annotations

import os
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import BallModelConfig
from .annotate import BallLabel, group_by_clip, read_ball_labels
from .augment import AugmentConfig, Augmenter
from .heatmap import render_gaussian
from .synthetic import make_clip

Window = Tuple[str, List[int]]


def _scale_point(x: float, y: float, src_wh: Tuple[int, int], dst_wh: Tuple[int, int]) -> Tuple[float, float]:
    sw, sh = src_wh
    dw, dh = dst_wh
    return x * dw / sw, y * dh / sh


def _stack_to_tensor(frames: np.ndarray) -> torch.Tensor:
    """(T, H, W, 3) uint8 -> (T*3, H, W) float32 in [0,1]."""
    t = torch.from_numpy(frames).float() / 255.0
    t = t.permute(0, 3, 1, 2)
    return t.reshape(-1, t.shape[2], t.shape[3])


class SyntheticBallDataset(Dataset):
    def __init__(self, cfg: BallModelConfig, num_clips: int = 40, clip_len: int = 8, seed: int = 0):
        self.cfg = cfg
        self.samples = []
        for i in range(num_clips):
            clip = make_clip(
                length=clip_len,
                height=cfg.input_height,
                width=cfg.input_width,
                seed=seed + i,
            )
            for end in range(cfg.input_frames - 1, clip_len):
                start = end - cfg.input_frames + 1
                self.samples.append((clip, start, end))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        clip, start, end = self.samples[idx]
        frames = clip.frames[start : end + 1]
        stack = _stack_to_tensor(frames)
        pos = clip.positions[end]
        hm = render_gaussian(self.cfg.input_height, self.cfg.input_width, pos, self.cfg.gaussian_sigma)
        return stack, torch.from_numpy(hm).unsqueeze(0)


def _cv2():
    import cv2

    return cv2


class DiskBallDataset(Dataset):
    def __init__(
        self,
        cfg: BallModelConfig,
        frames_root: str,
        labels_csv: str,
        augment: Optional[AugmentConfig] = None,
        seed: int = 0,
    ):
        self.cfg = cfg
        self.frames_root = frames_root
        labels = read_ball_labels(labels_csv)
        self.by_clip = group_by_clip(labels)
        self.windows: List[Tuple[str, List[BallLabel]]] = []
        for clip, items in self.by_clip.items():
            for end in range(cfg.input_frames - 1, len(items)):
                window = items[end - cfg.input_frames + 1 : end + 1]
                if _contiguous(window):
                    self.windows.append((clip, window))
        self.augmenter = Augmenter(augment, seed) if augment is not None and augment.enabled else None
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.windows)

    def set_epoch(self, epoch: int) -> None:
        """Vary augmentation between epochs while keeping it index-derived."""
        self.epoch = int(epoch)

    def groups(self) -> Sequence[str]:
        """The clip each window belongs to, for a group-aware split."""
        return [clip for clip, _ in self.windows]

    def _load_frame(self, clip: str, frame_idx: int) -> np.ndarray:
        cv2 = _cv2()
        for ext in (".jpg", ".png", ".jpeg"):
            path = os.path.join(self.frames_root, clip, f"{frame_idx}{ext}")
            if os.path.exists(path):
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                return img
        raise FileNotFoundError(f"frame not found: {clip}/{frame_idx}")

    def __getitem__(self, idx: int):
        cv2 = _cv2()
        clip, window = self.windows[idx]
        imgs = []
        src_wh = None
        for lb in window:
            img = self._load_frame(clip, lb.frame)
            if src_wh is None:
                src_wh = (img.shape[1], img.shape[0])
            img = cv2.resize(img, (self.cfg.input_width, self.cfg.input_height))
            imgs.append(img)
        frames = np.stack(imgs, axis=0)

        last = window[-1]
        if last.visible and src_wh is not None:
            x, y = _scale_point(last.x, last.y, src_wh, (self.cfg.input_width, self.cfg.input_height))
            pos = (x, y)
        else:
            pos = None

        if self.augmenter is not None:
            self.augmenter.seed_for_item(idx, self.epoch)
            frames, moved = self.augmenter(frames, [pos])
            pos = moved[0]

        stack = _stack_to_tensor(frames)
        hm = render_gaussian(self.cfg.input_height, self.cfg.input_width, pos, self.cfg.gaussian_sigma)
        return stack, torch.from_numpy(hm).unsqueeze(0)


def _contiguous(window: List[BallLabel]) -> bool:
    for a, b in zip(window, window[1:]):
        if b.frame - a.frame != 1:
            return False
    return True
