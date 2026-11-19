"""Loader for tuned operating values that are not part of the public release.

Tuned thresholds, gates and calibration constants live in `configs/tuned.yaml`,
which is withheld. Configs that depend on them expose a `tuned()` constructor
that reads its section from that file.
"""

from __future__ import annotations

import os
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TUNED_PATH = os.path.join(REPO_ROOT, "configs", "tuned.yaml")


def tuned(section: str) -> Dict[str, Any]:
    """Return one section of `configs/tuned.yaml` as a dict."""
    if not os.path.exists(TUNED_PATH):
        raise FileNotFoundError(
            "configs/tuned.yaml is not part of the public release; see docs/reproducibility.md."
        )
    import yaml

    with open(TUNED_PATH, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    if section not in raw:
        raise KeyError(f"configs/tuned.yaml has no section {section!r}")
    return dict(raw[section])
