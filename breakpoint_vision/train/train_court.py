"""Train the court keypoint model (CourtNet).

Mirrors train_ball, including its optimization setup, `gaussian_focal_loss`,
AdamW with warmup + cosine decay, gradient clipping, an EMA of the weights that is
what gets validated and checkpointed, and checkpoint selection on keypoint accuracy
rather than val loss (see `train/optim.py`). Reports keypoint detection accuracy:
the fraction of labeled keypoints whose predicted peak lands within a pixel
tolerance. The `smoke` path trains a few steps on synthetic courts to prove the
loop is wired, needing no dataset.

The court model detects all keypoints per frame, which is what lets inference
re-detect the court as the camera moves (see infer/court_timeline.py). Keypoint
precision matters more here than the accuracy number suggests: every keypoint
feeds the homography, so a systematic sub-pixel bias becomes court-plane error.

Validation is split by source video: the dataset holds many frames of each court
from one fixed camera, so a random frame split leaves the same court on both
sides and the reported keypoint accuracy measures memorization rather than
generalization to a new venue.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from ..config import AugmentConfig, CourtModelConfig, CourtTrainConfig, OptimConfig
from ..data.heatmap import render_multi
from ..data.split import group_split
from ..models.court_net import CourtNet, keypoint_peaks
from ..models.tracknet import gaussian_focal_loss
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


class SyntheticCourtDataset(Dataset):
    """Random-but-consistent keypoints -> image + target, for the smoke path.

    The 'image' is a low-effort rendering (blobs at the keypoints): enough to
    prove the training loop runs and loss falls; it is NOT a court simulator.
    """

    def __init__(self, cfg: CourtModelConfig, num: int = 12, seed: int = 0):
        self.cfg = cfg
        self.num = num
        self.k = cfg.num_keypoints
        rng = np.random.default_rng(seed)
        self.centers = []
        for _ in range(num):
            pts = rng.uniform(
                low=[0.15 * cfg.input_width, 0.15 * cfg.input_height],
                high=[0.85 * cfg.input_width, 0.85 * cfg.input_height],
                size=(self.k, 2),
            )
            self.centers.append(pts)

    def __len__(self) -> int:
        return self.num

    def __getitem__(self, idx: int):
        pts = self.centers[idx]
        centers = [tuple(p) for p in pts]
        if self.cfg.center_channel:
            centers = centers + [(float(pts[:, 0].mean()), float(pts[:, 1].mean()))]
        target = render_multi(self.cfg.input_height, self.cfg.input_width, centers, self.cfg.gaussian_sigma)
        image = np.clip(target[: self.k].sum(axis=0), 0, 1)
        img3 = np.stack([image, image, image], axis=0).astype(np.float32)
        return torch.from_numpy(img3), torch.from_numpy(target)


def build_dataset(
    model_cfg: CourtModelConfig,
    data_root: Optional[str],
    labels_json: Optional[str],
    smoke: bool,
    augment: Optional[AugmentConfig] = None,
    seed: int = 0,
):
    if smoke or not data_root or not labels_json:
        return SyntheticCourtDataset(model_cfg, num=12 if smoke else 40)
    from ..data.court_dataset import CourtDataset

    return CourtDataset(model_cfg, data_root, labels_json, augment=augment, seed=seed)


def _split_datasets(train_source, val_source, train_cfg: CourtTrainConfig):
    """(train_subset, val_subset, n_val_groups), holding out whole source videos."""
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


def _detection_accuracy(pred: torch.Tensor, target: torch.Tensor, tol_px: int, num_keypoints: int) -> Tuple[int, int]:
    with torch.no_grad():
        prob = torch.sigmoid(pred[:, :num_keypoints])
        tgt = target[:, :num_keypoints]
        pred_xy, _ = keypoint_peaks(prob)
        tgt_xy, tgt_conf = keypoint_peaks(tgt)
        labeled = tgt_conf > 0.5
        if labeled.sum() == 0:
            return 0, 0
        dist = torch.sqrt(((pred_xy - tgt_xy) ** 2).sum(dim=2))
        correct = ((dist <= tol_px) & labeled).sum().item()
        return int(correct), int(labeled.sum().item())


def train_court(
    model_cfg: CourtModelConfig,
    train_cfg: CourtTrainConfig,
    data_root: Optional[str] = None,
    labels_json: Optional[str] = None,
    device: Optional[str] = None,
    smoke: bool = False,
) -> TrainReport:
    set_seed(train_cfg.seed)
    dev = select_device(device)
    print(f"[train_court] device: {dev} ({device_report(dev)})")

    if smoke:
        train_cfg = _smoke_overrides(train_cfg)
        model_cfg = _smoke_model(model_cfg)

    augment = train_cfg.augment if not smoke else None
    train_source = build_dataset(model_cfg, data_root, labels_json, smoke, augment, train_cfg.seed)
    if len(train_source) < 2:
        raise RuntimeError("dataset too small to train")
    val_source = build_dataset(model_cfg, data_root, labels_json, smoke, None, train_cfg.seed)

    train_ds, val_ds, n_val_groups = _split_datasets(train_source, val_source, train_cfg)
    print(
        f"[train_court] train {len(train_ds)} / val {len(val_ds)} samples"
        + (f", validation held out by source video ({n_val_groups} videos)" if n_val_groups else "")
        + (", augmentation on" if augment and augment.enabled else ", augmentation off")
    )

    workers = 0 if smoke else make_workers(train_cfg.num_workers)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_ds, batch_size=train_cfg.batch_size, shuffle=False, num_workers=workers)

    model = CourtNet(
        num_keypoints=model_cfg.num_keypoints,
        base=model_cfg.base_channels,
        output_channels=model_cfg.output_channels,
    ).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    opt_cfg = train_cfg.optim
    total_steps = train_cfg.epochs * max(1, len(train_loader))
    warmup_steps = int(total_steps * opt_cfg.warmup_fraction)
    ema = ModelEma(model, opt_cfg.ema_decay) if opt_cfg.use_ema else None
    print(
        f"[train_court] {train_cfg.epochs} epochs, {total_steps} steps, "
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
        val_loss, acc = _evaluate(
            evaluated, val_loader, dev, train_cfg.detection_tolerance_px, model_cfg.num_keypoints
        )
        last_acc = acc
        dt = time.time() - t0
        print(
            f"[epoch {epoch}] train_loss {running / max(1, len(train_loader)):.5f} "
            f"val_loss {val_loss:.5f} kp_acc {acc:.3f} lr {lr:.2e} ({dt:.1f}s)"
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
            print(f"  saved checkpoint -> {ckpt_path} (kp_acc {acc:.3f}, val_loss {val_loss:.5f})")

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


def _evaluate(model, loader, dev, tol_px, num_keypoints):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(dev), y.to(dev)
            logits = model(x)
            total_loss += gaussian_focal_loss(logits, y).item()
            c, t = _detection_accuracy(logits, y, tol_px, num_keypoints)
            correct += c
            total += t
    acc = correct / total if total else 0.0
    return total_loss / max(1, len(loader)), acc


def _smoke_overrides(cfg: CourtTrainConfig) -> CourtTrainConfig:
    return CourtTrainConfig(
        epochs=2,
        batch_size=2,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        num_workers=0,
        val_fraction=0.2,
        checkpoint_dir="checkpoints/court_smoke",
        log_every=5,
        detection_tolerance_px=8,
        seed=cfg.seed,
        optim=OptimConfig(warmup_fraction=0.0, use_ema=False),
    )


def _smoke_model(cfg: CourtModelConfig) -> CourtModelConfig:
    return CourtModelConfig(
        input_height=96,
        input_width=160,
        num_keypoints=cfg.num_keypoints,
        base_channels=8,
        gaussian_sigma=cfg.gaussian_sigma,
        center_channel=cfg.center_channel,
    )
