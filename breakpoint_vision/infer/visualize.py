"""Render an annotated video: ball trail plus detected court lines.

Re-runs court detection and ball tracking and draws the results onto each frame,
writing a new mp4, so model quality can be inspected by eye.

What you see:
  - yellow dot + fading trail: the tracked ball
  - green court lines: the court in force at that frame (re-detected every
    `court_stride` frames, so it follows a moving camera)

Frames with no court drawn also have no ball drawn: the tracker is gated on the
court (see `court_gate`).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import PipelineConfig
from ..court_gate import batch_max_age_frames, court_world_segments
from ..geometry import apply_homography

TRAIL_LEN = 12


def _invert(h: np.ndarray) -> Optional[np.ndarray]:
    try:
        return np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return None


def render_video(
    video_path: str,
    out_path: str,
    cfg: PipelineConfig,
    device: Optional[str] = None,
) -> Dict[str, object]:
    import cv2

    from ..torch_utils import select_device
    from .video import VideoReader

    dev = select_device(device)

    probe = VideoReader(video_path)
    probe_meta = probe.meta
    probe.close()

    court_timeline = None
    if os.path.exists(cfg.court_weights):
        from ..court_gate import confirmed_timeline, validated_detector
        from .court_infer import CourtTracker
        from .court_timeline import build_court_timeline

        tracker = CourtTracker(cfg.court_weights, dev)

        def _frames():
            r = VideoReader(video_path)
            try:
                yield from r.frames(max_frames=cfg.video.max_frames)
            finally:
                r.close()

        court_timeline = confirmed_timeline(
            build_court_timeline(
                _frames,
                validated_detector(tracker.calibrate, cfg.court_gate),
                cfg.court_stride,
                cfg.video.max_frames,
            ),
            cfg.court_stride,
            cfg.court_gate.min_consecutive_detections,
        )

    max_court_age_frames = batch_max_age_frames(cfg.court_gate, probe_meta.fps, cfg.court_stride)

    ball_by_frame: Dict[int, Tuple[float, float, bool]] = {}
    court_available = court_timeline is not None and not court_timeline.is_empty()
    if os.path.exists(cfg.ball_weights) and (court_available or not cfg.court_gate.require_court_for_ball):
        from ..court_gate import TimelineBallGate
        from .ball_infer import track_ball

        gate = (
            TimelineBallGate(court_timeline, probe_meta.fps, cfg.court_gate, cfg.court_stride)
            if cfg.court_gate.require_court_for_ball
            else None
        )
        track = track_ball(video_path, cfg.ball_weights, dev, max_frames=cfg.video.max_frames, gate=gate)
        for (f, x, y, v) in track.trajectory:
            ball_by_frame[f] = (x, y, v)

    world_segs = court_world_segments()

    reader = VideoReader(video_path)
    meta = reader.meta
    writer = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        meta.fps or 30.0,
        (meta.width, meta.height),
    )
    trail: List[Tuple[int, int]] = []
    ball_frames = 0
    court_frames = 0

    try:
        for idx, frame_rgb in reader.frames(max_frames=cfg.video.max_frames):
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            calib = (
                court_timeline.active(idx, max_court_age_frames)
                if court_timeline and not court_timeline.is_empty()
                else None
            )
            if calib is not None:
                inv = _invert(calib.homography)
                if inv is not None:
                    court_frames += 1
                    for (a, b) in world_segs:
                        pa = apply_homography(inv, [a])[0]
                        pb = apply_homography(inv, [b])[0]
                        cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), (0, 220, 0), 2)

            sample = ball_by_frame.get(idx)
            if sample and sample[2]:
                bx, by = int(sample[0]), int(sample[1])
                trail.append((bx, by))
                if len(trail) > TRAIL_LEN:
                    trail.pop(0)
                ball_frames += 1
            for i in range(1, len(trail)):
                alpha = i / len(trail)
                cv2.line(frame, trail[i - 1], trail[i], (0, int(200 * alpha), int(255 * alpha)), 2)
            if sample and sample[2]:
                cv2.circle(frame, (int(sample[0]), int(sample[1])), 6, (0, 255, 255), -1)
                cv2.circle(frame, (int(sample[0]), int(sample[1])), 6, (0, 0, 0), 1)

            writer.write(frame)
    finally:
        reader.close()
        writer.release()

    return {
        "out_path": out_path,
        "frames": meta.frame_count,
        "ball_frames_drawn": ball_frames,
        "court_frames_drawn": court_frames,
    }
