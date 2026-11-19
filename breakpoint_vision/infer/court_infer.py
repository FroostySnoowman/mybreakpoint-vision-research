"""Court inference: predict keypoints on a frame and solve the court homography.

The camera moves (handheld / panning to follow play), so the court's position in
the image changes over time. We therefore re-detect the court repeatedly through
the video (see CourtTracker) rather than calibrating once. Each detection runs
the keypoint model on one frame, keeps only confident keypoints, requires at
least 4 to solve, and reports its reprojection error along with a derived
court-plane position uncertainty.

CourtTracker loads the model once and exposes `calibrate(frame)` so the per-frame
cost is a single forward pass, not a checkpoint reload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from ..config import CourtModelConfig
from ..court import CourtKeypoints
from ..geometry import estimate_homography_robust, reprojection_error


@dataclass
class CourtCalibration:
    homography: np.ndarray
    reprojection_error_m: float
    used_keypoints: int
    position_sigma_m: float


def load_court_model(weights_path: str, device: str):
    import torch

    from ..models.court_net import CourtNet

    ckpt = torch.load(weights_path, map_location=device)
    cfg_dict = ckpt.get("model_cfg", {})
    model_cfg = CourtModelConfig(**{k: v for k, v in cfg_dict.items() if k in CourtModelConfig().__dict__})
    model = CourtNet(
        num_keypoints=model_cfg.num_keypoints,
        base=model_cfg.base_channels,
        output_channels=model_cfg.output_channels,
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, model_cfg


class CourtTracker:
    """Loads the court keypoint model once; re-detects the court per frame.

    Because the camera moves, callers invoke `calibrate(frame)` repeatedly (once
    every `court_stride` frames). Only the forward pass runs each time; the model
    stays resident.
    """

    def __init__(self, weights_path: str, device: str, conf_threshold: float = 0.3):
        self.device = device
        self.conf_threshold = conf_threshold
        self.model, self.model_cfg = load_court_model(weights_path, device)

    def calibrate(self, frame_rgb: np.ndarray) -> Optional[CourtCalibration]:
        import cv2
        import torch

        from ..models.court_net import keypoint_peaks

        src_h, src_w = frame_rgb.shape[:2]
        small = cv2.resize(frame_rgb, (self.model_cfg.input_width, self.model_cfg.input_height))
        tensor = torch.from_numpy(small).float().div(255.0).permute(2, 0, 1).unsqueeze(0).to(self.device)

        with torch.no_grad():
            heatmaps = self.model.predict_heatmaps(tensor)
        coords, conf = keypoint_peaks(heatmaps[:, : self.model_cfg.num_keypoints], threshold=self.conf_threshold)
        coords = coords[0].cpu().numpy()
        conf = conf[0].cpu().numpy()

        kp = CourtKeypoints()
        world = kp.ordered_world_array()

        sx = src_w / self.model_cfg.input_width
        sy = src_h / self.model_cfg.input_height

        src_pts: List[Tuple[float, float]] = []
        dst_pts: List[Tuple[float, float]] = []
        for i, _name in enumerate(kp.names):
            if i >= coords.shape[0]:
                break
            if conf[i] >= self.conf_threshold:
                src_pts.append((float(coords[i, 0] * sx), float(coords[i, 1] * sy)))
                dst_pts.append(world[i])

        return solve_calibration(src_pts, dst_pts)


def calibrate_from_frame(
    frame_rgb: np.ndarray,
    weights_path: str,
    device: str,
    conf_threshold: float = 0.3,
) -> Optional[CourtCalibration]:
    return CourtTracker(weights_path, device, conf_threshold).calibrate(frame_rgb)


def solve_calibration(src_pts, dst_pts) -> Optional[CourtCalibration]:
    """Fit the court homography, rejecting keypoints the rest of the court contradicts.

    The reported error and `used_keypoints` describe the *inlier* fit, so a frame
    where one keypoint was discarded reports the accuracy of the calibration
    actually in use rather than an average that includes the point it threw away.
    """
    if len(src_pts) < 4:
        return None
    h, inliers = estimate_homography_robust(src_pts, dst_pts)
    kept_src = [src_pts[i] for i in inliers]
    kept_dst = [dst_pts[i] for i in inliers]
    err = reprojection_error(h, kept_src, kept_dst)
    return CourtCalibration(
        homography=h,
        reprojection_error_m=err,
        used_keypoints=len(inliers),
        position_sigma_m=_sigma_from_error(err),
    )


def _sigma_from_error(reprojection_error_m: float) -> float:
    """Turn calibration reprojection error into a court-plane position sigma.

    Position error for a point mapped through the homography is dominated by (a)
    how well the plane is calibrated and (b) the detection pixel error propagated
    through the homography, which grows with perspective foreshortening. We use
    the reprojection error as a floor and inflate it, clamped to a sane range,
    rather than pretend the raw reprojection error is the true position error.
    """
    floor = 0.02
    return float(min(0.12, max(floor, reprojection_error_m * 1.5)))
