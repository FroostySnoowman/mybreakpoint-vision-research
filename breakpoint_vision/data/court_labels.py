"""Court keypoint label format + parsing (torch-free, so it unit-tests cheaply).

The dataset (data/court_dataset.py) builds on this to render training targets;
this module only reads/represents labels. See court_dataset.py for the full
format description.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class CourtSample:
    file: str
    keypoints: Dict[str, Tuple[float, float]]
    width: Optional[int]
    height: Optional[int]


def read_court_labels(json_path: str) -> List[CourtSample]:
    with open(json_path) as fh:
        raw = json.load(fh)
    images = raw.get("images", raw if isinstance(raw, list) else [])
    samples: List[CourtSample] = []
    for entry in images:
        kps = {name: (float(xy[0]), float(xy[1])) for name, xy in entry.get("keypoints", {}).items() if xy}
        samples.append(
            CourtSample(
                file=entry["file"],
                keypoints=kps,
                width=entry.get("width"),
                height=entry.get("height"),
            )
        )
    return samples
