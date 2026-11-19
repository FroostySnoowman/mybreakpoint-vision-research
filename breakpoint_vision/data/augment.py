"""Training-time augmentation for the ball and court datasets.

Both datasets are small relative to the variation real footage shows: few ball
clips, and court frames with only one camera angle per video. Without
augmentation the models memorize those specific courts, lightings and camera
placements.

Every global transform here (geometry, brightness, contrast, hue, saturation,
blur, JPEG quality) is sampled once and applied identically to all frames of a
window, and the label is transformed with it. A stack must stay temporally
consistent: the ball model's whole input is what changed between frames, so
jittering brightness per frame would teach it that global flicker means "ball".

Gaussian noise is the deliberate exception: it is drawn per pixel per frame,
because that is what a real sensor does and it is precisely the nuisance signal
the model must learn to ignore. It carries no frame-to-frame structure, so it
cannot be mistaken for motion.

Deliberately excluded:
  - Vertical flips and large rotations. A tennis court seen from courtside always
    has the far baseline above the near one; teaching the network otherwise
    spends capacity on frames that cannot occur.
  - Anything that moves the ball independently of the frames, which would make
    the label wrong rather than augmented.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class AugmentConfig:
    """Augmentation strengths. All probabilities; set `enabled=False` to disable.

    Defaults are deliberately mild: the geometry of a tennis court is the signal,
    so scale/shift stay small enough that the court cannot leave the frame, and
    color jitter covers venue lighting rather than inventing new sports.
    """

    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.15
    hue_degrees: float = 6.0
    scale_jitter: float = 0.10
    shift_fraction: float = 0.05
    rotation_degrees: float = 4.0
    noise_std: float = 0.012
    blur_prob: float = 0.15
    jpeg_prob: float = 0.15
    jpeg_quality_min: int = 45


def _cv2():
    import cv2

    return cv2


class Augmenter:
    """Applies one sampled transform to a whole frame stack plus its points.

    Points are (x, y) in the *resized model input* frame; None means "not
    visible", and a point pushed outside the frame by the transform becomes None
    so the target heatmap correctly says "nothing here" rather than clamping the
    ball to an edge.
    """

    def __init__(self, cfg: Optional[AugmentConfig] = None, seed: int = 0):
        self.cfg = cfg or AugmentConfig()
        self.seed = int(seed)
        self.rng = np.random.default_rng(seed)

    def seed_for_item(self, index: int, epoch: int = 0) -> None:
        """Reseed from (dataset seed, sample index, epoch) before augmenting it.

        A DataLoader with `num_workers > 0` forks the dataset, and therefore this
        object, once per worker, so every worker inherits the same generator state
        and draws the same transform sequence. That cuts the effective
        augmentation variety by the worker count and correlates it with batch
        position.

        Deriving the state from the sample index instead makes a sample's transform
        a pure function of (seed, index, epoch): reproducible, worker-count
        independent, and still different every epoch.
        """
        self.rng = np.random.default_rng((self.seed, index, epoch))

    def __call__(
        self,
        frames: np.ndarray,
        points: List[Optional[Tuple[float, float]]],
        flip_index_map: Optional[Sequence[int]] = None,
    ) -> Tuple[np.ndarray, List[Optional[Tuple[float, float]]]]:
        """frames: (T, H, W, 3) uint8. Returns the same shape, plus moved points.

        `flip_index_map[i]` names the slot whose point becomes slot `i` after a
        mirror. Court keypoints need it: mirroring the image without swapping
        `sl_near_left` and `sl_near_right` makes the homography's point
        correspondences wrong, which is worse than not flipping at all. The ball
        has a single unnamed point and needs no map.
        """
        if not self.cfg.enabled:
            return frames, points

        cv2 = _cv2()
        t, h, w = frames.shape[:3]
        out = frames
        pts = list(points)

        if self.rng.random() < self.cfg.horizontal_flip_prob:
            out = out[:, :, ::-1]
            pts = [None if p is None else (w - 1.0 - p[0], p[1]) for p in pts]
            if flip_index_map is not None:
                pts = [pts[flip_index_map[i]] for i in range(len(pts))]

        matrix = self._affine(w, h)
        if matrix is not None:
            out = np.stack(
                [
                    cv2.warpAffine(
                        np.ascontiguousarray(f),
                        matrix,
                        (w, h),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REPLICATE,
                    )
                    for f in out
                ],
                axis=0,
            )
            pts = [None if p is None else _apply_affine(matrix, p) for p in pts]
            pts = [None if p is None or not _inside(p, w, h) else p for p in pts]

        out = self._photometric(out)
        return np.ascontiguousarray(out), pts

    def _affine(self, w: int, h: int) -> Optional[np.ndarray]:
        cfg = self.cfg
        if cfg.scale_jitter <= 0 and cfg.shift_fraction <= 0 and cfg.rotation_degrees <= 0:
            return None
        cv2 = _cv2()
        scale = 1.0 + float(self.rng.uniform(-cfg.scale_jitter, cfg.scale_jitter))
        angle = float(self.rng.uniform(-cfg.rotation_degrees, cfg.rotation_degrees))
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        matrix[0, 2] += float(self.rng.uniform(-cfg.shift_fraction, cfg.shift_fraction)) * w
        matrix[1, 2] += float(self.rng.uniform(-cfg.shift_fraction, cfg.shift_fraction)) * h
        return matrix

    def _photometric(self, frames: np.ndarray) -> np.ndarray:
        """Color/quality jitter, sampled once and applied to every frame."""
        cv2 = _cv2()
        cfg = self.cfg
        f = frames.astype(np.float32) / 255.0

        if cfg.brightness > 0:
            f = f * (1.0 + float(self.rng.uniform(-cfg.brightness, cfg.brightness)))
        if cfg.contrast > 0:
            factor = 1.0 + float(self.rng.uniform(-cfg.contrast, cfg.contrast))
            f = (f - f.mean()) * factor + f.mean()
        f = np.clip(f, 0.0, 1.0)

        if cfg.saturation > 0 or cfg.hue_degrees > 0:
            sat = 1.0 + float(self.rng.uniform(-cfg.saturation, cfg.saturation))
            hue = float(self.rng.uniform(-cfg.hue_degrees, cfg.hue_degrees))
            shifted = []
            for frame in f:
                hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
                hsv[..., 0] = (hsv[..., 0] + hue) % 360.0
                hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0.0, 1.0)
                shifted.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB))
            f = np.stack(shifted, axis=0)

        if cfg.blur_prob > 0 and self.rng.random() < cfg.blur_prob:
            sigma = float(self.rng.uniform(0.4, 1.1))
            f = np.stack([cv2.GaussianBlur(frame, (0, 0), sigma) for frame in f], axis=0)

        if cfg.noise_std > 0:
            f = f + self.rng.normal(0.0, cfg.noise_std, size=f.shape).astype(np.float32)

        f = np.clip(f, 0.0, 1.0)
        out = (f * 255.0).astype(np.uint8)

        if cfg.jpeg_prob > 0 and self.rng.random() < cfg.jpeg_prob:
            quality = int(self.rng.integers(cfg.jpeg_quality_min, 96))
            encoded = []
            for frame in out:
                ok, buf = cv2.imencode(".jpg", frame[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
                encoded.append(cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1] if ok else frame)
            out = np.stack(encoded, axis=0)

        return out


def _apply_affine(matrix: np.ndarray, point: Tuple[float, float]) -> Tuple[float, float]:
    x, y = point
    return (
        float(matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]),
        float(matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]),
    )


def _inside(point: Tuple[float, float], width: int, height: int) -> bool:
    return 0.0 <= point[0] < width and 0.0 <= point[1] < height


HORIZONTAL_FLIP_KEYPOINTS = {
    "bl_doubles_near_left": "bl_doubles_near_right",
    "bl_doubles_near_right": "bl_doubles_near_left",
    "bl_doubles_far_left": "bl_doubles_far_right",
    "bl_doubles_far_right": "bl_doubles_far_left",
    "sl_near_left": "sl_near_right",
    "sl_near_right": "sl_near_left",
    "sl_far_left": "sl_far_right",
    "sl_far_right": "sl_far_left",
    "svc_near_left": "svc_near_right",
    "svc_near_right": "svc_near_left",
    "svc_far_left": "svc_far_right",
    "svc_far_right": "svc_far_left",
    "center_svc_near": "center_svc_near",
    "center_svc_far": "center_svc_far",
}
"""Left/right keypoint pairs, swapped when a court frame is mirrored.

Mirroring an image without swapping these labels teaches the model that
`sl_near_left` is sometimes on the right, which makes the homography solver's
point correspondences wrong and is far worse than no flip at all.
"""
