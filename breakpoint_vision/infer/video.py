"""Video reading utilities shared by the inference modules.

Isolated so higher-level inference code stays testable with fake frame sources.
Requires opencv only when actually decoding a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class VideoMeta:
    fps: float
    width: int
    height: int
    frame_count: int


class VideoReader:
    def __init__(self, path: str):
        import cv2

        self._cv2 = cv2
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"could not open video: {path}")
        self.meta = VideoMeta(
            fps=float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            width=int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            frame_count=int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )

    def frames(self, max_frames: Optional[int] = None) -> Iterator[Tuple[int, np.ndarray]]:
        idx = 0
        while True:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield idx, self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
            idx += 1
            if max_frames is not None and idx >= max_frames:
                break

    def close(self) -> None:
        self.cap.release()


def read_first_frame(path: str) -> np.ndarray:
    reader = VideoReader(path)
    try:
        for _, frame in reader.frames(max_frames=1):
            return frame
    finally:
        reader.close()
    raise RuntimeError(f"no frames in video: {path}")
