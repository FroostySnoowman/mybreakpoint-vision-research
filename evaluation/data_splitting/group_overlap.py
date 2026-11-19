"""Group overlap between a training and a validation set.

A validation sample is "seen" when its group (the clip for ball windows, the
source video for court frames) also has samples in training. The share of seen
validation samples measures how much a validation score can reflect memorized
footage rather than generalization.

Modes:
  tcd                 the imported TennisCourtDetector splits,
                      <data>/court_tcd/train and <data>/court_tcd/val, grouped by
                      source video; also reported for the 200 validation frames
                      the broadcast court evaluation draws
  labels TRAIN VAL    two labels.json files ({"images": […]}) grouped by --key
  lists TRAIN VAL     two text files with one group name per sample per line

Output: sample and group counts on each side, how many validation samples and
groups are seen, and their shares. Group names are not written.
"""

import json
import sys
from pathlib import Path
from typing import Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.common import base_parser, write_json  # noqa: E402
from evaluation.court_detection.evaluate_court import broadcast_rows  # noqa: E402


def group_overlap(train_groups: Iterable[str], val_groups: Iterable[str]) -> dict:
    train, val = list(train_groups), list(val_groups)
    train_set, val_set = set(train), set(val)
    seen_samples = sum(g in train_set for g in val)
    seen_groups = len(val_set & train_set)
    return {
        'train_samples': len(train),
        'train_groups': len(train_set),
        'val_samples': len(val),
        'val_groups': len(val_set),
        'val_samples_seen_in_train': seen_samples,
        'val_groups_seen_in_train': seen_groups,
        'share_val_samples_seen': round(seen_samples / len(val), 4) if val else None,
        'share_val_groups_seen': round(seen_groups / len(val_set), 4) if val_set else None,
    }


def label_groups(path, key='video'):
    with open(path) as fh:
        return [r[key] for r in json.load(fh)['images']]


def list_groups(path):
    with open(path) as fh:
        return [line.strip() for line in fh if line.strip()]


def main():
    ap = base_parser(__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='mode', required=True)
    sub.add_parser('tcd')
    lab = sub.add_parser('labels')
    lab.add_argument('train', type=Path)
    lab.add_argument('val', type=Path)
    lab.add_argument('--key', default='video')
    lst = sub.add_parser('lists')
    lst.add_argument('train', type=Path)
    lst.add_argument('val', type=Path)
    a = ap.parse_args()
    if a.mode == 'tcd':
        train, val = label_groups(a.data / 'court_tcd' / 'train' / 'labels.json'), label_groups(a.data / 'court_tcd' / 'val' / 'labels.json')
        extra = {'broadcast_eval_frames': group_overlap(train, [r['video'] for r in broadcast_rows(a.data)])}
    elif a.mode == 'labels':
        train, val, extra = label_groups(a.train, a.key), label_groups(a.val, a.key), {}
    else:
        train, val, extra = list_groups(a.train), list_groups(a.val), {}
    result = {'mode': a.mode, **group_overlap(train, val), **extra}
    print(json.dumps(result), flush=True)
    write_json(result, a.out)


if __name__ == '__main__':
    main()
