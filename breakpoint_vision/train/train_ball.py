"""Train the TrackNet-style ball tracker.

Device-aware (MPS / CUDA / CPU) and reports a detection accuracy (fraction of
visible-ball frames whose predicted peak lands within a pixel tolerance of the
label). The `smoke` path trains a handful of steps on synthetic data to prove the
whole loop is wired, requiring no dataset.

Validation is split by clip, not by window. Windows slide one frame at a time,
so a random split puts a near-identical window on the other side and the
reported accuracy measures memorization. Augmentation is applied to the training
split only (a shared dataset object would otherwise transform validation too and
make its loss incomparable between epochs), and `set_epoch` varies it per epoch
(see `data/augment.py`).

The loss is `gaussian_focal_loss`, not the detection focal loss: on a Gaussian
heatmap the latter's `alpha` weights the background more than the ball, which
starves the peak of gradient. Optimization is AdamW with cosine decay after a
warmup, gradient clipping, and an EMA of the weights; the EMA is what gets
validated and checkpointed, and selection is on detection accuracy with val loss
as the tie-break. See `train/optim.py`.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, random_split

from ..config import AugmentConfig, BallModelConfig, BallTrainConfig, OptimConfig
from ..data.dataset import SyntheticBallDataset
from ..data.split import group_split
from ..models.tracknet import TrackNet, gaussian_focal_loss, heatmap_peak
from ..torch_utils import device_report, make_workers, select_device, set_seed
from .optim import (
    ModelEma,
    apply_lr,
    better_checkpoint,
    clip_gradients,
    evaluation_module,
    lr_scale,
)


@dataclass
class TrainReport:
    epochs_run: int
    best_val_loss: float
    final_detection_acc: float
    checkpoint_path: str
    device: str
    train_samples: int = 0
    val_samples: int = 0
    val_groups: int = 0


def _detection_accuracy(pred_logits: torch.Tensor, target: torch.Tensor, tol_px: int) -> Tuple[int, int]:
    """Count (correct, total) over frames where the target has a visible ball."""
    with torch.no_grad():
        prob = torch.sigmoid(pred_logits)
        pred_xy, _ = heatmap_peak(prob)
        tgt_xy, tgt_conf = heatmap_peak(target)
        visible = tgt_conf > 0.5
        if visible.sum() == 0:
            return 0, 0
        dist = torch.sqrt(((pred_xy - tgt_xy) ** 2).sum(dim=1))
        correct = ((dist <= tol_px) & visible).sum().item()
        return int(correct), int(visible.sum().item())


def build_dataset(
    model_cfg: BallModelConfig,
    data_root: Optional[str],
    labels_csv: Optional[str],
    smoke: bool,
    augment: Optional[AugmentConfig] = None,
    seed: int = 0,
):
    if smoke or not data_root or not labels_csv:
        num = 12 if smoke else 40
        clip_len = 6 if smoke else 8
        return SyntheticBallDataset(model_cfg, num_clips=num, clip_len=clip_len)
    from ..data.dataset import DiskBallDataset

    return DiskBallDataset(model_cfg, data_root, labels_csv, augment=augment, seed=seed)


def _split_datasets(train_source, val_source, train_cfg: BallTrainConfig):
    """(train_subset, val_subset, n_val_groups), split by clip when possible.

    `train_source` and `val_source` are two views of the same underlying data with
    identical sample ordering, but only the training one augments. Synthetic data
    has no clip grouping, so it falls back to a random split.
    """
    groups = train_source.groups() if hasattr(train_source, "groups") else None
    if groups is None:
        val_len = max(1, int(len(train_source) * train_cfg.val_fraction))
        generator = torch.Generator().manual_seed(train_cfg.seed)
        train_ds, val_ds = random_split(
            train_source, [len(train_source) - val_len, val_len], generator=generator
        )
        return train_ds, val_ds, 0

    train_idx, val_idx = group_split(groups, train_cfg.val_fraction, train_cfg.seed)
    n_val_groups = len({groups[i] for i in val_idx})
    return Subset(train_source, train_idx), Subset(val_source, val_idx), n_val_groups


def train_ball(
    model_cfg: BallModelConfig,
    train_cfg: BallTrainConfig,
    data_root: Optional[str] = None,
    labels_csv: Optional[str] = None,
    device: Optional[str] = None,
    smoke: bool = False,
) -> TrainReport:
    set_seed(train_cfg.seed)
    dev = select_device(device)
    print(f"[train_ball] device: {dev} ({device_report(dev)})")

    if smoke:
        train_cfg = _smoke_overrides(train_cfg)
        model_cfg = _smoke_model(model_cfg)

    augment = train_cfg.augment if not smoke else None
    train_source = build_dataset(model_cfg, data_root, labels_csv, smoke, augment, train_cfg.seed)
    if len(train_source) < 2:
        raise RuntimeError("dataset too small to train")
    val_source = build_dataset(model_cfg, data_root, labels_csv, smoke, None, train_cfg.seed)

    train_ds, val_ds, n_val_groups = _split_datasets(train_source, val_source, train_cfg)
    print(
        f"[train_ball] train {len(train_ds)} / val {len(val_ds)} samples"
        + (f", validation held out by clip ({n_val_groups} clips)" if n_val_groups else "")
        + (f", augmentation on" if augment and augment.enabled else ", augmentation off")
    )

    workers = 0 if smoke else make_workers(train_cfg.num_workers)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=workers)

    model = TrackNet(input_frames=model_cfg.input_frames, base=model_cfg.base_channels).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    opt_cfg = train_cfg.optim
    total_steps = train_cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * opt_cfg.warmup_fraction)
    ema = ModelEma(model, opt_cfg.ema_decay) if opt_cfg.use_ema else None
    print(
        f"[train_ball] {train_cfg.epochs} epochs, {total_steps} steps, "
        f"warmup {warmup_steps}, cosine to {opt_cfg.min_lr_fraction:g}x, "
        f"clip {opt_cfg.grad_clip_norm:g}, ema {'on' if ema else 'off'}"
    )

    os.makedirs(train_cfg.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(train_cfg.checkpoint_dir, "best.pt")

    best_val = float("inf")
    best_acc = -1.0
    last_acc = 0.0
    global_step = 0
    for epoch in range(train_cfg.epochs):
        if hasattr(train_source, "set_epoch"):
            train_source.set_epoch(epoch)
        model.train()
        running = 0.0
        t0 = time.time()
        lr = train_cfg.lr
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(dev), y.to(dev)
            lr = apply_lr(
                optimizer,
                train_cfg.lr,
                lr_scale(global_step, total_steps, warmup_steps, opt_cfg.min_lr_fraction),
            )
            optimizer.zero_grad()
            logits = model(x)
            loss = gaussian_focal_loss(logits, y)
            loss.backward()
            clip_gradients(model, opt_cfg.grad_clip_norm)
            optimizer.step()
            if ema is not None:
                ema.update(model)
            running += loss.item()
            global_step += 1
            if step % train_cfg.log_every == 0:
                print(
                    f"  epoch {epoch} step {step}/{len(train_loader)} "
                    f"loss {loss.item():.5f} lr {lr:.2e}"
                )

        evaluated = evaluation_module(model, ema)
        val_loss, acc = _evaluate(evaluated, val_loader, dev, train_cfg.detection_tolerance_px)
        last_acc = acc
        dt = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss {running / max(1, len(train_loader)):.5f} "
            f"val_loss {val_loss:.5f} det_acc {acc:.3f} lr {lr:.2e} ({dt:.1f}s)"
        )

        if better_checkpoint(acc, val_loss, best_acc, best_val):
            best_acc = max(best_acc, acc)
            best_val = val_loss
            torch.save(
                {
                    "model_state": evaluated.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "val_loss": val_loss,
                    "detection_acc": acc,
                    "detection_tolerance_px": train_cfg.detection_tolerance_px,
                    "ema": ema is not None,
                    "epoch": epoch,
                },
                ckpt_path,
            )
            print(f"  saved checkpoint -> {ckpt_path} (det_acc {acc:.3f}, val_loss {val_loss:.5f})")

    return TrainReport(
        epochs_run=train_cfg.epochs,
        best_val_loss=best_val,
        final_detection_acc=last_acc,
        checkpoint_path=ckpt_path,
        device=dev,
        train_samples=len(train_ds),
        val_samples=len(val_ds),
        val_groups=n_val_groups,
    )


def _evaluate(model, loader, dev, tol_px):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            total_loss += gaussian_focal_loss(logits, y).item()
            c, t = _detection_accuracy(logits, y, tol_px)
            correct += c
            total += t
    acc = correct / total if total else 0.0
    return total_loss / max(1, len(loader)), acc


def _smoke_overrides(cfg: BallTrainConfig) -> BallTrainConfig:
    return BallTrainConfig(
        epochs=2,
        batch_size=2,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_workers=0,
        val_fraction=0.2,
        checkpoint_dir="checkpoints/ball_smoke",
        log_every=5,
        detection_tolerance_px=12,
        seed=cfg.seed,
        optim=OptimConfig(warmup_fraction=0.0, use_ema=False),
    )


def _smoke_model(cfg: BallModelConfig) -> BallModelConfig:
    return BallModelConfig(
        input_frames=cfg.input_frames,
        input_height=96,
        input_width=160,
        heatmap_scale=1,
        base_channels=8,
        gaussian_sigma=cfg.gaussian_sigma,
    )
