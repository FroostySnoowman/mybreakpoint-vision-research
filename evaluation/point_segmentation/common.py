"""Loading out-of-fold predictions, labels and decode settings.

A cross-validation root holds fold<K>/oof.npz, one in-play probability series
per held-out video (one value per feature step). Labels are
<data>/gold_train/<video>.json, else <data>/gold/<video>.json; only point, fault
and let segments count as play, and the labeled window defaults to seconds
300-900. The decode search grid, objective and tie rule are the ones in
scripts/choose_decode.py, imported from there.
"""

from __future__ import annotations

import glob
import importlib.util
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from breakpoint_vision.points.decode import DecodeConfig
from breakpoint_vision.points.scoring import Playback, play_segments

from evaluation.common import REPO_ROOT

DEFAULT_WINDOW = (300.0, 900.0)
LEAD_IN_SECONDS = 1.0
TAIL_SECONDS = 2.0
PLAYBACK_MERGE_SECONDS = 2.5


def choose_decode():
    spec = importlib.util.spec_from_file_location('choose_decode', REPO_ROOT / 'scripts' / 'choose_decode.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_oof(root) -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Out-of-fold probabilities per video, and the fold that held each video out."""
    probs, fold_of = {}, {}
    for path in sorted(glob.glob(f'{root}/fold*/oof.npz')):
        k = int(Path(path).parent.name[4:])
        d = np.load(path)
        for v in d.files:
            probs[v] = d[v][:, 0] if d[v].ndim == 2 else d[v]
            fold_of[v] = k
    return probs, fold_of


def label_path(data, vid) -> Optional[Path]:
    for d in ('gold_train', 'gold'):
        p = Path(data) / d / f'{vid}.json'
        if p.exists():
            return p
    return None


def load_gold(path) -> tuple:
    with open(path) as fh:
        label = json.load(fh)
    return play_segments(label), tuple(label.get('window', DEFAULT_WINDOW))


def load_golds(data, vids) -> Dict[str, tuple]:
    out = {}
    for v in vids:
        p = label_path(data, v)
        if p is not None:
            out[v] = load_gold(p)
    return out


def add_decode_args(ap) -> None:
    ap.add_argument('--decode', type=Path, default=None, help='decode JSON (as written by scripts/choose_decode.py); default: DecodeConfig.tuned()')
    ap.add_argument('--lead', type=float, default=LEAD_IN_SECONDS, help='playback lead-in seconds when --decode is not given')
    ap.add_argument('--tail', type=float, default=TAIL_SECONDS, help='playback tail seconds when --decode is not given')
    ap.add_argument('--merge', type=float, default=PLAYBACK_MERGE_SECONDS, help='playback merge gap seconds when --decode is not given')


def decode_settings(args) -> Tuple[DecodeConfig, Playback]:
    if args.decode is None:
        return DecodeConfig.tuned(), Playback(args.lead, args.tail, args.merge)
    with open(args.decode) as fh:
        d = json.load(fh)
    cfg = DecodeConfig(**{k: v for k, v in d.items() if k in DecodeConfig.__dataclass_fields__})
    return cfg, Playback(d['lead_in_seconds'], d['tail_seconds'], d.get('merge_gap_seconds_playback', PLAYBACK_MERGE_SECONDS))
