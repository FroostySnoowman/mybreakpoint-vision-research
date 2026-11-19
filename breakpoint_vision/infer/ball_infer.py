"""Run the trained ball tracker over a video, producing a trajectory.

Output is a (frame, x, y, visible) trajectory in original video pixels.
Coordinates are rescaled from model input size back to source resolution.

Tracking is gated on the court: pass a `gate` (see `court_gate.TimelineBallGate`)
and frames with no believed court in view are skipped entirely (the model is not
even run), while detections that land far outside the projected court are
discarded. Without a gate the tracker runs ungated, which is meant for
diagnostics.

Per-frame heatmaps are read by `ball_track.BallTracker` rather than by taking the
argmax over a fixed threshold, which recovers frames where the ball is a
secondary peak or a faint but continuous one. See `ball_track`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..ball_track import BallTracker, BallTrackerConfig, heatmap_candidates
from ..config import BallModelConfig
from .video import VideoReader


@dataclass
class BallTrack:
    trajectory: List[Tuple[int, float, float, bool]]
    fps: float
    width: int
    height: int
    gated_frames: int = 0
    rejected_detections: int = 0


def load_ball_model(weights_path: str, device: str):
    import torch

    from ..models.tracknet import TrackNet

    ckpt = torch.load(weights_path, map_location=device)
    cfg_dict = ckpt.get("model_cfg", {})
    model_cfg = BallModelConfig(**{k: v for k, v in cfg_dict.items() if k in BallModelConfig().__dict__})
    model = TrackNet(input_frames=model_cfg.input_frames, base=model_cfg.base_channels).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_cfg


def track_ball(
    video_path: str,
    weights_path: str,
    device: str,
    max_frames: Optional[int] = None,
    gate=None,
    tracker_cfg: Optional[BallTrackerConfig] = None,
) -> BallTrack:
    import cv2
    import torch

    model, model_cfg = load_ball_model(weights_path, device)
    reader = VideoReader(video_path)
    meta = reader.meta

    tracker_cfg = tracker_cfg or BallTrackerConfig.tuned()
    tracker = BallTracker(model_cfg.input_width, tracker_cfg)
    buf: List[np.ndarray] = []
    trajectory: List[Tuple[int, float, float, bool]] = []
    gated_frames = 0
    rejected_detections = 0
    sx = meta.width / model_cfg.input_width
    sy = meta.height / model_cfg.input_height

    try:
        for idx, frame in reader.frames(max_frames=max_frames):
            if gate is not None and not gate.allows_frame(idx):
                buf.clear()
                tracker.reset()
                gated_frames += 1
                trajectory.append((idx, 0.0, 0.0, False))
                continue

            small = cv2.resize(frame, (model_cfg.input_width, model_cfg.input_height))
            buf.append(small)
            if len(buf) < model_cfg.input_frames:
                trajectory.append((idx, 0.0, 0.0, False))
                continue
            if len(buf) > model_cfg.input_frames:
                buf.pop(0)

            stack = np.stack(buf, axis=0).astype(np.float32) / 255.0
            tensor = torch.from_numpy(stack).permute(0, 3, 1, 2).reshape(1, -1, model_cfg.input_height, model_cfg.input_width)
            tensor = tensor.to(device)
            with torch.no_grad():
                prob = model.predict_heatmap(tensor)
            heatmap = prob[0, 0].detach().cpu().numpy()

            picked = tracker.update(heatmap_candidates(heatmap, tracker_cfg))
            if picked is None:
                trajectory.append((idx, 0.0, 0.0, False))
                continue

            x_px = picked[0] * sx
            y_px = picked[1] * sy
            visible = True
            if gate is not None and not gate.accepts(idx, x_px, y_px):
                visible = False
                rejected_detections += 1
                tracker.reset()
            trajectory.append((idx, x_px, y_px, visible))
    finally:
        reader.close()

    return BallTrack(
        trajectory=trajectory,
        fps=meta.fps,
        width=meta.width,
        height=meta.height,
        gated_frames=gated_frames,
        rejected_detections=rejected_detections,
    )
