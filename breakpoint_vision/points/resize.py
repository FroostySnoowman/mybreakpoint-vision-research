"""Deterministic image resizes defined in integer arithmetic.

`portable_area_resize` (area averaging, what the models are fed) and
`portable_resize` (bilinear) produce identical output on every platform, so any
reimplementation can match them bit for bit.

OpenCV's `resize` picks different kernels on different machines (arm64 wheels
route some ratios through KleidiCV, which rounds differently), so it can disagree
with itself across computers. `portable_resize` is OpenCV's portable fixed-point
INTER_LINEAR for uint8 images.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=32)
def _axis(source: int, destination: int):
    scale = 1.0 / (destination / source)
    first = np.zeros(destination, np.int64)
    w0 = np.zeros(destination, np.int64)
    w1 = np.zeros(destination, np.int64)
    for d in range(destination):
        fx = np.float32((d + 0.5) * scale - 0.5)
        sx = int(np.floor(fx))
        fx = np.float32(fx - np.float32(sx))
        if sx < 0:
            fx, sx = np.float32(0), 0
        if sx >= source - 1:
            fx, sx = np.float32(0), source - 1
        first[d] = sx
        w0[d] = int(np.rint((np.float32(1) - fx) * np.float32(2048)))
        w1[d] = int(np.rint(fx * np.float32(2048)))
    second = np.minimum(first + 1, source - 1)
    return first, second, w0, w1


def portable_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Resize a uint8 image (H, W) or (H, W, C) to (height, width)."""
    squeeze = img.ndim == 2
    src = img[:, :, None] if squeeze else img
    h, w = src.shape[:2]
    x0, x1, a0, a1 = _axis(w, width)
    y0, y1, b0, b1 = _axis(h, height)
    data = src.astype(np.int64)
    rows = data[:, x0] * a0[None, :, None] + data[:, x1] * a1[None, :, None]
    top = rows[y0] >> 4
    bottom = rows[y1] >> 4
    value = ((top * b0[:, None, None]) >> 16) + ((bottom * b1[:, None, None]) >> 16) + 2
    out = np.clip(value >> 2, 0, 255).astype(np.uint8)
    return out[:, :, 0] if squeeze else out


@lru_cache(maxsize=64)
def area_spans(source: int, destination: int):
    """Per output index: (first, last, first_weight, last_weight) over the source.

    Output pixel d covers the source interval [d * source, (d + 1) * source) in
    units of 1 / destination source pixels; source pixel i covers
    [i * destination, (i + 1) * destination). A source pixel's weight is the
    length of the overlap, an integer, so the weights of one output sum to
    exactly `source`. Pixels strictly between first and last weigh
    `destination`.
    """
    if source <= 0 or destination <= 0:
        raise ValueError(f'invalid area resize {source} -> {destination}')
    d = np.arange(destination, dtype=np.int64)
    start = d * source
    end = start + source
    first = start // destination
    last = (end - 1) // destination
    first_weight = np.minimum(end, (first + 1) * destination) - start
    last_weight = np.where(last > first, end - last * destination, 0)
    return first, last, first_weight, last_weight


def _area_from_cumulative(cumulative: np.ndarray, moved: np.ndarray, source: int, destination: int) -> np.ndarray:
    first, last, first_weight, last_weight = area_spans(source, destination)
    shape = (-1,) + (1,) * (moved.ndim - 1)
    middle = (cumulative[np.maximum(last, first + 1)] - cumulative[first + 1]).astype(np.int64) * destination
    edges = moved[first].astype(np.int64) * first_weight.reshape(shape) + moved[last].astype(np.int64) * last_weight.reshape(shape)
    return middle + edges


def _cumulative(moved: np.ndarray, dtype) -> np.ndarray:
    out = np.empty((moved.shape[0] + 1,) + moved.shape[1:], dtype)
    out[0] = 0
    np.cumsum(moved, axis=0, dtype=dtype, out=out[1:])
    return out


def portable_area_resizes(img: np.ndarray, sizes) -> list:
    """`portable_area_resize` to several (width, height) sizes, sharing the column sums."""
    squeeze = img.ndim == 2
    src = img[:, :, None] if squeeze else img
    h, w = src.shape[:2]
    across = np.moveaxis(src, 1, 0)
    cumulative = _cumulative(across, np.int32 if w * 255 < 2 ** 31 else np.int64)
    outputs = []
    for width, height in sizes:
        rows = np.moveaxis(_area_from_cumulative(cumulative, across, w, width), 0, 1)
        total = _area_from_cumulative(_cumulative(rows, np.int64), rows, h, height)
        denominator = w * h
        out = ((total + denominator // 2) // denominator).astype(np.uint8)
        outputs.append(out[:, :, 0] if squeeze else out)
    return outputs


def portable_area_resize(img: np.ndarray, width: int, height: int) -> np.ndarray:
    """Area-averaging resize of a uint8 image (H, W) or (H, W, C) to (height, width).

    Each output pixel is the area-weighted mean of the source pixels it covers,
    with fractional edge weights for non-integer ratios (OpenCV INTER_AREA
    semantics for downscaling). Exact integer arithmetic: with the per-axis
    integer weights of `area_spans`, the weighted sum S over the covered
    pixels is divided by D = source_width * source_height and rounded half up,
    out = (S + D // 2) // D. Columns are summed first, then rows (64-bit); the
    order does not matter because every step is exact.
    """
    return portable_area_resizes(img, [(width, height)])[0]
