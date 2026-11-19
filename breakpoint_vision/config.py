"""Configuration dataclasses and a small YAML loader.

PyYAML is only imported when a file is actually loaded, so importing this module
(and the pure-logic modules that reference the dataclasses) never requires torch
or yaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, Optional

from .data.augment import AugmentConfig
from .params import tuned


@dataclass
class BallModelConfig:
    input_frames: int = 3
    input_height: int = 288
    input_width: int = 512
    heatmap_scale: int = 1
    base_channels: int = 32
    gaussian_sigma: float = 3.0


@dataclass
class OptimConfig:
    """Schedule and stabilization knobs shared by both trainers.

    See `train/optim.py` for why each exists. `warmup_fraction` is a fraction of
    total training steps rather than a step count so it stays correct when the
    dataset or epoch count changes; `min_lr_fraction` is the cosine floor.
    `grad_clip_norm` sits above the typical gradient norm so that clipping only
    catches outlier batches.
    """

    warmup_fraction: float = 0.03
    min_lr_fraction: float = 0.01
    grad_clip_norm: float = 50.0
    ema_decay: float = 0.999
    use_ema: bool = True


@dataclass
class BallTrainConfig:
    epochs: int = 60
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4
    val_fraction: float = 0.1
    checkpoint_dir: str = "checkpoints/ball"
    log_every: int = 20
    detection_tolerance_px: int = 6
    seed: int = 1337
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class CourtTrainConfig:
    epochs: int = 80
    batch_size: int = 8
    lr: float = 1e-3
    weight_decay: float = 1e-5
    num_workers: int = 4
    val_fraction: float = 0.1
    checkpoint_dir: str = "checkpoints/court"
    log_every: int = 20
    detection_tolerance_px: int = 8
    seed: int = 1337
    augment: AugmentConfig = field(default_factory=AugmentConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)


@dataclass
class CourtModelConfig:
    input_height: int = 288
    input_width: int = 512
    num_keypoints: int = 14
    base_channels: int = 32
    gaussian_sigma: float = 3.0
    center_channel: bool = True

    @property
    def output_channels(self) -> int:
        return self.num_keypoints + (1 if self.center_channel else 0)


@dataclass
class VideoConfig:
    max_frames: Optional[int] = None


@dataclass
class CourtGateConfig:
    """How the ball tracker is gated on a believed court detection.

    See `court_gate` for the reasoning. With `require_court_for_ball` set, no
    court in view means no ball output at all. The numeric limits are tuned
    operating values: build with `CourtGateConfig.tuned()` or pass them explicitly.
    """

    min_keypoints: int
    max_reprojection_error_m: float
    min_court_frame_fraction: float
    max_court_frame_fraction: float
    min_consecutive_detections: int
    max_court_age_seconds: float
    ball_runoff_side_m: float
    ball_runoff_back_m: float
    ball_up_margin_fraction: float
    require_court_for_ball: bool = True

    @classmethod
    def tuned(cls, **overrides) -> "CourtGateConfig":
        return cls(**{**tuned("court_gate"), **overrides})


@dataclass
class PipelineConfig:
    """Inference-time settings for rendering a tracked video.

    Has no ball/court model shape fields: input resolution, frame count and
    channel width are read back from the checkpoint's `model_cfg` by
    `load_ball_model`/`load_court_model`.
    """

    video: VideoConfig = field(default_factory=VideoConfig)
    court_gate: CourtGateConfig = field(default_factory=CourtGateConfig.tuned)
    ball_weights: str = "checkpoints/ball/best.pt"
    court_weights: str = "checkpoints/court/best.pt"
    court_stride: int = 15

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _build(dc_type, data: Dict[str, Any]):
    if data is None:
        return dc_type()
    field_names = {f.name for f in fields(dc_type)}
    kwargs = {k: v for k, v in data.items() if k in field_names}
    return dc_type(**kwargs)


def _build_train(dc_type, data: Optional[Dict[str, Any]], root: Optional[Dict[str, Any]] = None):
    """Build a training config, resolving the nested `augment:` and `optim:` blocks.

    `_build` only handles flat scalars, so an `augment:` mapping would otherwise
    arrive as a raw dict and silently replace the dataclass. `augment:` is accepted
    both inside the training section and at the document root, where the configs
    in `configs/` put it. The `optim:` block is resolved the same way.
    """
    cfg = _build(dc_type, data)
    for source in (root, data):
        if source and isinstance(source.get("augment"), dict):
            cfg.augment = _build(AugmentConfig, source["augment"])
        if source and isinstance(source.get("optim"), dict):
            cfg.optim = _build(OptimConfig, source["optim"])
    return cfg


def load_ball_train_config(path: Optional[str]) -> BallTrainConfig:
    if not path:
        return BallTrainConfig()
    import yaml

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build_train(BallTrainConfig, raw.get("ball_train", raw), raw)


def load_court_train_config(path: Optional[str]) -> CourtTrainConfig:
    if not path:
        return CourtTrainConfig()
    import yaml

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build_train(CourtTrainConfig, raw.get("court_train", raw), raw)


def load_ball_model_config(path: Optional[str]) -> BallModelConfig:
    if not path:
        return BallModelConfig()
    import yaml

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build(BallModelConfig, raw.get("ball_model", raw))


def load_court_model_config(path: Optional[str]) -> CourtModelConfig:
    if not path:
        return CourtModelConfig()
    import yaml

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    return _build(CourtModelConfig, raw.get("court", raw))
