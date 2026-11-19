"""Shared training machinery: LR schedule, weight averaging, gradient clipping.

LR schedule: cosine decay to a small floor lets the last epochs refine sub-pixel
peak placement, which is the precision the homography depends on. A short linear
warmup avoids taking the large first step of a freshly initialized heatmap head
at full LR.

Weight averaging: the final iterate sits wherever the last minibatch pushed it.
An exponential moving average of the weights is a cheap approximation to
averaging over the tail of training. The EMA weights are what get validated and
checkpointed.

Gradient clipping: heatmap losses are dominated by a handful of peak pixels, so a
batch where the model is badly wrong about the ball produces an outsized
gradient. Clipping the global norm bounds the damage a single bad batch can do.

Checkpoint selection: loss and localization accuracy are correlated but not
identical, so `better_checkpoint` selects on accuracy and uses loss only to break
ties.
"""

from __future__ import annotations

import copy
import math
from typing import Optional

import torch


def lr_scale(step: int, total_steps: int, warmup_steps: int, min_lr_fraction: float) -> float:
    """Multiplier on the base LR: linear warmup, then cosine decay to a floor.

    Returns 1.0 for a degenerate schedule (no steps) so a smoke run with a single
    batch still trains at the configured LR rather than at zero.
    """
    if total_steps <= 0:
        return 1.0
    warmup = max(0, int(warmup_steps))
    if warmup > 0 and step < warmup:
        return float(step + 1) / float(warmup)

    progress = (step - warmup) / max(1, total_steps - warmup)
    progress = min(1.0, max(0.0, progress))
    floor = min(1.0, max(0.0, float(min_lr_fraction)))
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def apply_lr(optimizer: torch.optim.Optimizer, base_lr: float, scale: float) -> float:
    lr = base_lr * scale
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


class ModelEma:
    """Exponential moving average of model weights, evaluated and checkpointed in
    place of the raw weights.

    `decay` is warmed up over the first steps so the average is not anchored to
    the random initialization early in training. Buffers (e.g. BatchNorm running
    stats) are copied rather than averaged, since they are already running
    averages.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.updates = 0

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.updates += 1
        d = min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))
        ema_params = dict(self.module.named_parameters())
        for name, param in model.named_parameters():
            target = ema_params.get(name)
            if target is not None:
                target.mul_(d).add_(param.detach(), alpha=1.0 - d)
        ema_buffers = dict(self.module.named_buffers())
        for name, buf in model.named_buffers():
            target = ema_buffers.get(name)
            if target is not None:
                target.copy_(buf)

    def state_dict(self):
        return self.module.state_dict()


def clip_gradients(model: torch.nn.Module, max_norm: float) -> float:
    """Clip the global gradient norm; returns the pre-clip norm (0.0 when disabled)."""
    if max_norm is None or max_norm <= 0:
        return 0.0
    return float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm))


def better_checkpoint(
    acc: float,
    loss: float,
    best_acc: float,
    best_loss: float,
    tolerance: float = 1e-4,
) -> bool:
    """Select on localization accuracy, breaking near-ties on validation loss.

    Accuracy is a coarse, quantized metric (a fixed pixel tolerance), so many
    epochs score identically; the tie-break keeps
    the run improving through those plateaus instead of freezing on the first epoch
    that reached the best accuracy.
    """
    if acc > best_acc + tolerance:
        return True
    if acc < best_acc - tolerance:
        return False
    return loss < best_loss


def evaluation_module(model: torch.nn.Module, ema: Optional[ModelEma]) -> torch.nn.Module:
    """The weights to validate and checkpoint: EMA when enabled, otherwise the model."""
    return ema.module if ema is not None else model
