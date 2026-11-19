"""Which decoded video frames feed the ball model for one labeled frame.

Labels carry a timestamp. Source videos run at 30 fps or faster, and the
pipeline sees at most 30 fps, so frames closer than about one 30 fps step to the
previously kept frame are skipped. The selection functions here work on frame
timestamps only; `decode_window` supplies them from a video file.

The labeled frame is the decoded frame nearest the label timestamp, not the
first frame at or after it: timestamps are rounded to the millisecond, so "at or
after" can land one frame late.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

STEP_30FPS = 1 / 30
SKIP_BELOW = 1 / 30 - 1 / 240
VARIANT_SKIP_BELOW = 1 / 30 - 0.005


def context_indices(times: Sequence[float], t: float, context: float) -> Tuple[List[int], Optional[int]]:
    """Frames within `context` seconds either side of `t`, and the one nearest `t`.

    Returns the kept indices into `times` and the position, within the kept list,
    of the frame nearest `t` (None if nothing was kept).
    """
    kept, last = [], None
    for i, ft in enumerate(times):
        if ft < t - context:
            continue
        if ft > t + context:
            break
        if last is not None and ft - last < SKIP_BELOW:
            continue
        kept.append(i)
        last = ft
    if not kept:
        return kept, None
    return kept, int(np.argmin(np.abs(np.array([times[i] for i in kept]) - t)))


def sequence_indices(times: Sequence[float], t: float, lead: float) -> List[int]:
    """Consecutive frames over the `lead` seconds before `t`, ending at the frame nearest `t`."""
    kept, last = [], None
    for i, ft in enumerate(times):
        if ft < t - lead:
            continue
        if last is not None and ft - last < SKIP_BELOW:
            if ft > t + STEP_30FPS:
                break
            continue
        kept.append(i)
        last = ft
        if ft >= t:
            break
    if not kept:
        return kept
    ts = np.array([times[i] for i in kept])
    return kept[:int(np.argmin(np.abs(ts - t))) + 1]


def variant_indices(times: Sequence[float], t1: float) -> List[int]:
    """Frames anchored at the one nearest `t1`, thinned backward to 30 fps.

    `times` are the decoded timestamps from the window start up to and including
    the first frame at or after `t1`.
    """
    if not times:
        return []
    anchor = int(np.argmin([abs(t - t1) for t in times]))
    keep, last = [anchor], times[anchor]
    for i in range(anchor - 1, -1, -1):
        if last - times[i] < VARIANT_SKIP_BELOW:
            continue
        last = times[i]
        keep.append(i)
    return list(reversed(keep))


def decode_window(path: str, seek_to: float, stop: Callable[[float], bool], interpolation: int, start: Optional[float] = None, size: Tuple[int, int] = (512, 288)) -> List[Tuple[float, np.ndarray]]:
    """(timestamp, resized RGB frame) pairs from `seek_to`, up to and including the first frame where `stop(t)`.

    Frames without a timestamp are dropped, and so are frames before `start` when
    it is given. Each frame is resized as it is decoded; resizing is per frame,
    so resizing frames the selection later drops changes nothing.
    """
    import av
    import cv2

    out = []
    with av.open(path) as c:
        s = c.streams.video[0]
        c.seek(int(max(0, seek_to) / s.time_base), stream=s)
        for f in c.decode(s):
            if f.time is None:
                continue
            ft = float(f.time)
            if start is not None and ft < start:
                continue
            out.append((ft, cv2.resize(f.to_ndarray(format='rgb24'), size, interpolation=interpolation)))
            if stop(ft):
                break
    return out


def interpolation(name: str) -> int:
    import cv2

    return {'area': cv2.INTER_AREA, 'linear': cv2.INTER_LINEAR}[name]
