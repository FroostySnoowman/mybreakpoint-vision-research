"""Physically motivated lighting degradations for robustness training.

The color jitter in `augment.py` covers small venue differences. Phones filming
tennis see far more than that: night sessions under floodlights where the far
baseline is a few photons per pixel, indoor halls with flicker and color casts,
dusk with a sky brighter than the court, and midday sun with hard tree shadows.
A model that has only seen well-exposed broadcast frames fails silently in all
of them.

Degradations are applied in linear light (the sRGB curve is undone first),
because exposure and sensor noise are linear phenomena: halving the light halves
the linear signal, and shot noise grows with the square root of the photon count.
Everything global (exposure, white balance, shadows, vignetting, the
noise-reduction blur strength, JPEG quality) is sampled once per window and
applied identically to every frame, so the only frame-to-frame change the model
sees is real motion plus independent sensor noise, exactly as on a phone.

Phones auto-expose, so a dark scene comes out somewhat below normal brightness
with more noise, not near-black. Sensor noise and the phone's denoise blur happen
at capture resolution, and downscaling to the model input averages most of that
noise away, so the photon count is scaled by the area ratio between capture and
model resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from ..params import tuned


@dataclass
class LightingConfig:
    """Probabilities and ranges for each degradation.

    `night_prob` selects a low-light sample (exposure drawn from `night_exposure`,
    then camera gain back to `residual_exposure` of normal brightness);
    `harsh_prob` selects an over-exposed sample (`harsh_exposure`, with clipping).
    `capture_area_ratio` is capture pixels per model-input pixel.
    The remaining samples keep exposure near 1 and still receive the milder
    effects, so clean frames stay well represented. The exposure, noise and
    blur ranges are calibrated values: build with `LightingConfig.tuned()` or
    pass them explicitly.
    """

    night_exposure: tuple
    harsh_exposure: tuple
    read_noise: tuple
    dark_read_noise: tuple
    photons_at_white: tuple
    residual_exposure: tuple
    capture_area_ratio: tuple
    denoise_sigma: tuple
    night_prob: float = 0.45
    harsh_prob: float = 0.15
    cast_prob: float = 0.5
    shadow_prob: float = 0.25
    vignette_prob: float = 0.4
    denoise_prob: float = 0.6
    jpeg_prob: float = 0.6
    jpeg_quality: tuple = (35, 90)

    @classmethod
    def tuned(cls, **overrides) -> "LightingConfig":
        section = {k: tuple(v) for k, v in tuned("lighting").items()}
        return cls(**{**section, **overrides})


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92, 1.055 * np.power(x, 1 / 2.4) - 0.055)


@dataclass
class LightingSample:
    """One sampled set of global parameters, shared by every frame of a window."""

    exposure: float
    gain: np.ndarray
    read_noise: float
    photons: float
    shadow: Optional[np.ndarray]
    vignette: float
    denoise_sigma: float
    jpeg_quality: Optional[int]
    auto_gain_cap: float = 12.0


class LightingDegrader:
    def __init__(self, cfg: Optional[LightingConfig] = None, seed: int = 0):
        self.cfg = cfg or LightingConfig.tuned()
        self.rng = np.random.default_rng(seed)

    def sample(self, height: int, width: int) -> LightingSample:
        c, r = self.cfg, self.rng
        mode = r.random()
        if mode < c.night_prob:
            exposure = float(np.exp(r.uniform(np.log(c.night_exposure[0]), np.log(c.night_exposure[1]))))
        elif mode < c.night_prob + c.harsh_prob:
            exposure = float(r.uniform(*c.harsh_exposure))
        else:
            exposure = float(r.uniform(0.75, 1.25))
        gain = np.ones(3, np.float32)
        if r.random() < c.cast_prob:
            warmth = r.uniform(-0.25, 0.45)
            gain = np.array([1 + warmth * 0.6, 1.0, 1 - warmth * 0.7], np.float32)
            gain *= r.uniform(0.95, 1.05, size=3).astype(np.float32)
        shadow = self._shadow(height, width) if r.random() < c.shadow_prob else None
        vignette = float(r.uniform(0.15, 0.55)) if r.random() < c.vignette_prob else 0.0
        dark = exposure < 0.5
        denoise = float(r.uniform(*c.denoise_sigma)) if dark and r.random() < c.denoise_prob else 0.0
        area = float(r.uniform(*c.capture_area_ratio))
        quality = int(r.integers(c.jpeg_quality[0], c.jpeg_quality[1] + 1)) if r.random() < c.jpeg_prob else None
        return LightingSample(
            exposure=exposure,
            gain=gain,
            read_noise=float(r.uniform(*(c.dark_read_noise if dark else c.read_noise))) / float(np.sqrt(area)),
            photons=float(r.uniform(*c.photons_at_white)) * area,
            shadow=shadow,
            vignette=vignette,
            denoise_sigma=denoise,
            jpeg_quality=quality,
            auto_gain_cap=float(r.uniform(*c.residual_exposure)) / min(exposure, 1.0),
        )

    def _shadow(self, height: int, width: int) -> np.ndarray:
        import cv2

        r = self.rng
        mask = np.ones((height, width), np.float32)
        for _ in range(int(r.integers(1, 4))):
            n = int(r.integers(4, 9))
            cx, cy = r.uniform(0, width), r.uniform(height * 0.3, height)
            radius = r.uniform(0.1, 0.5) * width
            angles = np.sort(r.uniform(0, 2 * np.pi, n))
            radii = radius * r.uniform(0.5, 1.0, n)
            pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], 1).astype(np.int32)
            cv2.fillPoly(mask, [pts], float(r.uniform(0.25, 0.6)))
        k = int(max(3, width // 60)) | 1
        return cv2.GaussianBlur(mask, (k, k), 0)

    def apply_frame(self, frame: np.ndarray, s: LightingSample) -> np.ndarray:
        """Degrade one uint8 RGB frame with the shared sample `s`."""
        import cv2

        h, w = frame.shape[:2]
        lin = srgb_to_linear(frame.astype(np.float32) / 255.0)
        lin = lin * s.gain[None, None, :]
        if s.shadow is not None:
            lin = lin * s.shadow[:, :, None]
        if s.vignette > 0:
            yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
            d = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
            lin = lin * (1 - s.vignette * np.clip(d, 0, 1.5) / 1.5)[:, :, None]
        lin = lin * s.exposure
        photons = np.clip(lin, 0, None) * s.photons
        noisy = self.rng.poisson(photons).astype(np.float32) / s.photons
        noisy += self.rng.normal(0, s.read_noise, size=noisy.shape).astype(np.float32)
        out_lin = noisy * min(1.0 / s.exposure, s.auto_gain_cap) if s.exposure < 1 else noisy
        out = linear_to_srgb(out_lin)
        img = (out * 255.0 + 0.5).astype(np.uint8)
        if s.denoise_sigma > 0:
            img = cv2.GaussianBlur(img, (0, 0), s.denoise_sigma)
        if s.jpeg_quality is not None:
            ok, enc = cv2.imencode('.jpg', img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, s.jpeg_quality])
            if ok:
                img = cv2.imdecode(enc, cv2.IMREAD_COLOR)[:, :, ::-1]
        return np.ascontiguousarray(img)

    def apply_window(self, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Degrade a temporally consistent window of uint8 RGB frames."""
        h, w = frames[0].shape[:2]
        s = self.sample(h, w)
        return [self.apply_frame(f, s) for f in frames]
