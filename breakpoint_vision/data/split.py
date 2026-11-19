"""Group-aware train/validation splitting.

Both datasets here are built from video, and video frames are not independent
samples. `DiskBallDataset` emits one window per frame, sliding by one, so
consecutive windows share `input_frames - 1` of their frames; the court dataset
holds many frames from each source video, all showing the same court from the
same camera. A random split over samples therefore puts near-duplicates on both
sides, and validation measures memorization of frames the model effectively
trained on.

Splitting by group (the clip for ball windows, the source video for court
frames) makes validation measure generalization to unseen footage.

`group_split` is deterministic given the seed: groups are hashed rather than
shuffled by index, so adding data to one group never reshuffles the others.
"""

from __future__ import annotations

import hashlib
from typing import Dict, List, Sequence, Tuple


def _group_rank(group: str, seed: int) -> float:
    """A stable pseudo-random value in [0, 1) for a group name."""
    digest = hashlib.sha256(f"{seed}:{group}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def split_groups(groups: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    """Partition distinct group names into (train, val) by hashed rank.

    Takes whole groups, so the smallest possible validation set is one group. We
    never return an empty side while there are at least two groups to split.
    """
    unique = sorted(set(groups))
    if len(unique) < 2:
        return unique, []

    ranked = sorted(unique, key=lambda g: _group_rank(g, seed))
    n_val = int(round(len(ranked) * max(0.0, min(1.0, val_fraction))))
    n_val = max(1, min(len(ranked) - 1, n_val))
    return ranked[n_val:], ranked[:n_val]


def group_split(sample_groups: Sequence[str], val_fraction: float, seed: int) -> Tuple[List[int], List[int]]:
    """Sample indices split so that no group appears on both sides.

    `sample_groups[i]` is the group of sample `i`. Returns (train_indices,
    val_indices) for use with `torch.utils.data.Subset`.
    """
    _, val_groups = split_groups(sample_groups, val_fraction, seed)
    val_set = set(val_groups)
    train_idx: List[int] = []
    val_idx: List[int] = []
    for i, g in enumerate(sample_groups):
        (val_idx if g in val_set else train_idx).append(i)
    return train_idx, val_idx


def group_sizes(sample_groups: Sequence[str]) -> Dict[str, int]:
    """Sample count per group, for reporting a split."""
    sizes: Dict[str, int] = {}
    for g in sample_groups:
        sizes[g] = sizes.get(g, 0) + 1
    return sizes
