"""TrackNet-style ball tracker, trained from scratch.

A tennis ball is a tiny, fast, motion-blurred object that generic pretrained
detectors handle poorly, and the network is small enough to train from random
initialization on a single GPU.

Architecture: a compact encoder-decoder (VGG-ish encoder, upsampling decoder)
that consumes a stack of consecutive frames (default 3) and outputs a single
heatmap whose peak marks the ball in the *last* input frame. Feeding multiple
frames lets the network use motion, which is essential when the ball blurs into
a streak. Output is a per-pixel probability heatmap trained against a Gaussian
blob centered on the labeled ball position.

This follows the published TrackNet / TrackNetV2 idea but is an independent
implementation.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

HEATMAP_BIAS_PRIOR = 0.01


def init_heatmap_bias(head: nn.Conv2d, prior: float = HEATMAP_BIAS_PRIOR) -> None:
    """Bias the output layer to predict `prior` before it has learned anything.

    Required for `gaussian_focal_loss` to train at all. With a default zero bias
    the head starts at p=0.5 on every pixel, and because that loss sums over
    pixels and divides only by the peak count, the initial gradient norm is
    enormous; gradient clipping then scales every step down to almost nothing.
    With the prior, clipping only catches outliers.
    """
    if head.bias is None:
        return
    nn.init.constant_(head.bias, -math.log((1.0 - prior) / prior))


def _conv_block(in_ch: int, out_ch: int, n: int = 2) -> nn.Sequential:
    layers = []
    c = in_ch
    for _ in range(n):
        layers += [
            nn.Conv2d(c, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        c = out_ch
    return nn.Sequential(*layers)


class TrackNet(nn.Module):
    def __init__(self, input_frames: int = 3, in_channels_per_frame: int = 3, base: int = 32, out_frames: int = 1):
        super().__init__()
        in_ch = input_frames * in_channels_per_frame

        self.enc1 = _conv_block(in_ch, base, 2)
        self.enc2 = _conv_block(base, base * 2, 2)
        self.enc3 = _conv_block(base * 2, base * 4, 3)
        self.enc4 = _conv_block(base * 4, base * 8, 3)

        self.pool = nn.MaxPool2d(2, 2)

        self.up3 = _conv_block(base * 8 + base * 4, base * 4, 3)
        self.up2 = _conv_block(base * 4 + base * 2, base * 2, 2)
        self.up1 = _conv_block(base * 2 + base, base, 2)

        self.head = nn.Conv2d(base, out_frames, kernel_size=1)
        init_heatmap_bias(self.head)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d3 = F.interpolate(e4, scale_factor=2, mode="bilinear", align_corners=False)
        d3 = self.up3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, scale_factor=2, mode="bilinear", align_corners=False)
        d2 = self.up2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, scale_factor=2, mode="bilinear", align_corners=False)
        d1 = self.up1(torch.cat([d1, e1], dim=1))

        return self.head(d1)

    def predict_heatmap(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def heatmap_peak(heatmap: torch.Tensor, threshold: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (coords, confidence) for a batch of single-channel heatmaps.

    coords is (B, 2) as (x, y) in heatmap pixels; confidence is (B,) peak value.
    A confidence below `threshold` means "ball not visible this frame" and the
    caller should treat the coordinate as invalid.
    """
    if heatmap.dim() == 4:
        heatmap = heatmap[:, 0]
    b, h, w = heatmap.shape
    flat = heatmap.view(b, -1)
    conf, idx = flat.max(dim=1)
    ys = (idx // w).float()
    xs = (idx % w).float()
    coords = torch.stack([xs, ys], dim=1)
    return coords, conf


def focal_bce_loss(pred_logits: torch.Tensor, target: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """Detection-style focal BCE (RetinaNet form), kept for comparison.

    `alpha` here follows the RetinaNet convention, where the positive class is the
    rare one and `alpha` weights it. On a Gaussian heatmap the ball covers a tiny
    fraction of pixels, so `alpha=0.25` weights the background more than the ball.
    Both trainers use `gaussian_focal_loss` instead.
    """
    prob = torch.sigmoid(pred_logits)
    ce = F.binary_cross_entropy_with_logits(pred_logits, target, reduction="none")
    p_t = prob * target + (1 - prob) * (1 - target)
    alpha_t = alpha * target + (1 - alpha) * (1 - target)
    loss = alpha_t * (1 - p_t) ** gamma * ce
    return loss.mean()


def gaussian_focal_loss(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 2.0,
    beta: float = 4.0,
    peak_rtol: float = 1e-2,
    peak_floor: float = 0.5,
) -> torch.Tensor:
    """The CornerNet/CenterNet penalty-reduced focal loss, normalized by peak count.

    Suited to a regressed Gaussian target, as opposed to a binary mask.

    Positives are the peaks, not every pixel above zero. The Gaussian's shoulder
    is neither a positive nor a plain negative: a prediction one pixel off the
    center is nearly right, so penalizing it as a full false positive fights the
    sub-pixel refinement the tracker depends on. `(1 - target) ** beta` scales the
    negative penalty down near the peak and to zero at it, so the shoulder guides
    rather than punishes.

    Normalizing by the number of peaks, not by pixel count, keeps the gradient
    from vanishing: under `.mean()` the few ball pixels are averaged over the
    whole frame. Dividing by the peak count makes one frame's ball worth the same
    regardless of resolution, so the loss stays comparable when the input size
    changes.

    A frame with no visible ball (an all-zero target) is a legitimate sample and
    contributes only the negative term, so `num_pos` is floored at 1 rather than
    letting an empty frame divide by zero.

    The peak mask is derived from each channel's own maximum, not from a fixed
    threshold like `target >= 1 - 1e-4`. Real targets never reach exactly 1.0: the
    Gaussian is centered on a sub-pixel position, so the brightest pixel falls
    short of it. A fixed threshold can therefore select zero positives, leaving
    only the negative term, whose optimum is "predict background everywhere".

    Per channel, not per sample, because the court target stacks 15 keypoint
    heatmaps and each has its own independent sub-pixel rounding. Taking one
    maximum across the whole sample sets the bar at the luckiest keypoint and
    starves every channel below it of positive gradient. The ball's
    single-channel target is the degenerate case of the same rule.

    `peak_rtol` is relative to that maximum so it adapts to whatever
    `gaussian_sigma` produces, while `peak_floor` keeps a genuinely empty (all-zero)
    channel, an occluded keypoint or an invisible ball, from promoting its own
    background to a peak.
    """
    if target.dim() != 4:
        raise ValueError(
            f"gaussian_focal_loss expects a (B, C, H, W) target, got {tuple(target.shape)}. "
            "The peak mask is per-channel, so a squeezed (B, H, W) tensor would be read as "
            "B x H single-row channels and silently mask the wrong pixels."
        )

    prob = torch.sigmoid(pred_logits).clamp(1e-6, 1.0 - 1e-6)
    channel_max = target.flatten(2).max(dim=2).values[:, :, None, None]
    peaks = target.ge((channel_max * (1.0 - peak_rtol)).clamp(min=peak_floor)).float()

    pos_loss = -torch.log(prob) * (1.0 - prob) ** alpha * peaks
    neg_loss = -torch.log(1.0 - prob) * prob**alpha * (1.0 - target) ** beta * (1.0 - peaks)

    num_pos = peaks.sum().clamp(min=1.0)
    return (pos_loss.sum() + neg_loss.sum()) / num_pos
