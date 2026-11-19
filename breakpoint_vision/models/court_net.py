"""Court keypoint model.

Predicts one heatmap per court keypoint (14 by default; see
`court.CourtKeypoints`). The peaks give image-space locations for the court
corners and line intersections, which are matched to their known world
coordinates to solve the court homography.

Same compact encoder-decoder family as the ball tracker, but with a multi-channel
head (one channel per keypoint) and a single RGB frame as input, since the court
is static and needs no motion cue.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .tracknet import _conv_block, init_heatmap_bias


def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    h, w = int(ref.shape[-2]), int(ref.shape[-1])
    if int(x.shape[-2]) * 2 == h and int(x.shape[-1]) * 2 == w:
        return F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
    return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)


class CourtNet(nn.Module):
    def __init__(self, num_keypoints: int = 14, base: int = 32, output_channels: int = None):
        super().__init__()
        if output_channels is None:
            output_channels = num_keypoints
        self.enc1 = _conv_block(3, base, 2)
        self.enc2 = _conv_block(base, base * 2, 2)
        self.enc3 = _conv_block(base * 2, base * 4, 3)
        self.enc4 = _conv_block(base * 4, base * 8, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.up3 = _conv_block(base * 8 + base * 4, base * 4, 3)
        self.up2 = _conv_block(base * 4 + base * 2, base * 2, 2)
        self.up1 = _conv_block(base * 2 + base, base, 2)
        self.head = nn.Conv2d(base, output_channels, kernel_size=1)
        init_heatmap_bias(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.up3(torch.cat([_match(e4, e3), e3], dim=1))
        d2 = self.up2(torch.cat([_match(d3, e2), e2], dim=1))
        d1 = self.up1(torch.cat([_match(d2, e1), e1], dim=1))
        return self.head(d1)

    def predict_heatmaps(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def keypoint_peaks(
    heatmaps: torch.Tensor,
    threshold: float = 0.3,
    subpixel_radius: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """(B, K, 2) xy coords and (B, K) confidences for K keypoint heatmaps.

    Coordinates are refined to sub-pixel by the intensity-weighted centroid of the
    patch around each integer peak, because the integer grid is a real error
    source rather than a rounding detail: at 512 wide against a 1280-wide source
    one heatmap pixel is 2.5 source pixels, and that quantization propagates
    through the homography onto the court plane. `subpixel_radius=0` restores the
    plain argmax.

    `threshold` is unused here and kept for call-site compatibility; filtering by
    confidence is the caller's decision, since a keypoint dropped from the
    homography is not the same as one that does not exist.
    """
    b, k, h, w = heatmaps.shape
    flat = heatmaps.reshape(b, k, -1)
    conf, idx = flat.max(dim=2)
    ys = (idx // w).float()
    xs = (idx % w).float()

    r = int(subpixel_radius)
    if r > 0:
        offs = torch.arange(-r, r + 1, device=heatmaps.device)
        dy, dx = torch.meshgrid(offs, offs, indexing="ij")
        py = ((idx // w).unsqueeze(-1).unsqueeze(-1) + dy).clamp(0, h - 1)
        px = ((idx % w).unsqueeze(-1).unsqueeze(-1) + dx).clamp(0, w - 1)
        patch = flat.gather(2, (py * w + px).reshape(b, k, -1)).reshape(b, k, 2 * r + 1, 2 * r + 1)
        total = patch.sum(dim=(2, 3))
        ok = total > 1e-9
        safe = total.clamp(min=1e-9)
        cx = (patch * px.float()).sum(dim=(2, 3)) / safe
        cy = (patch * py.float()).sum(dim=(2, 3)) / safe
        xs = torch.where(ok, cx, xs)
        ys = torch.where(ok, cy, ys)

    coords = torch.stack([xs, ys], dim=2)
    return coords, conf
