"""Device selection and seeding helpers.

Torch is imported inside each function so the pure-logic modules never import it.
"""

from __future__ import annotations

import os
from typing import Optional


def select_device(preferred: Optional[str] = None) -> str:
    """Pick a training/inference device.

    Priority when unspecified: Apple MPS -> CUDA -> CPU.
    """
    import torch

    if preferred and preferred != "auto":
        return preferred

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_report(device: str) -> str:
    import torch

    if device == "mps":
        return "Apple Silicon GPU (MPS)"
    if device == "cuda":
        try:
            return f"CUDA: {torch.cuda.get_device_name(0)}"
        except Exception:
            return "CUDA"
    return "CPU"


def set_seed(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_workers(requested: int) -> int:
    cpu = os.cpu_count() or 2
    return max(0, min(requested, cpu - 1))
