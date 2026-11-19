"""Nested decode selection: choose the decode on three folds, score the fourth.

Usage: nested_decode.py ROOT --out …

The full-data selection in scripts/choose_decode.py picks decode and playback
settings on the same out-of-fold predictions it reports. Here, for each held-out
fold, the same grid, objective, tie rule and play-free guard choose a setting
using only the other folds' videos; the held-out fold's videos are then scored
with that setting, and the pooled result over all held-out folds is reported.
The play-free guard counts spurious segments per 10 minutes on the selecting
videos only. This reuses existing out-of-fold predictions: selecting-fold
models were trained on the scored fold. It is decode-selection sensitivity
analysis, not fully nested model cross-validation.
"""

import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from breakpoint_vision.points.decode import DecodeConfig, decode  # noqa: E402
from breakpoint_vision.points.features import STEP_SECONDS  # noqa: E402
from breakpoint_vision.points.scoring import Playback, score_videos, summarize  # noqa: E402

from evaluation.common import base_parser, load_id_map, write_json  # noqa: E402
from evaluation.point_segmentation.common import choose_decode, load_golds, load_oof  # noqa: E402

cd = choose_decode()


def grid():
    settings = list(itertools.product(cd.QUANTILES, cd.FLOORS, cd.THRESHOLDS, cd.MIN_SECONDS, cd.MERGE_SECONDS, cd.PADDING))
    settings += list(itertools.product((0.9,), (1.0,), cd.THRESHOLDS, cd.MIN_SECONDS, cd.MERGE_SECONDS, cd.PADDING))
    return settings


def setting_rows(args):
    """Per-video rows and per-video spurious segment counts for one setting."""
    probs, golds, free, setting = args
    q, floor, (on, off), mins, merge, (lead, tail) = setting
    cfg = DecodeConfig(on=on, off=off, smooth_steps=1, min_seconds=mins, merge_gap_seconds=merge, scale_quantile=q, scale_floor=floor)
    pb = Playback(lead, tail, cd.MERGE_GAP)
    rows = score_videos(probs, golds, cfg, pb)
    fake = {v: len(decode(p, cfg)) for v, p in free.items()}
    return cfg, pb, rows, fake


def select(results, vids, free):
    """The setting choose_decode.py would pick if only `vids` were available."""
    minutes = sum(len(free[v]) for v in vids if v in free) * STEP_SECONDS / 60

    def summ(r):
        cfg, pb, rows, fake = r
        s = summarize({v: rows[v] for v in vids})
        s['fake_per_10min'] = sum(fake[v] for v in vids if v in fake) / minutes * 10
        return s
    scored = [(r, summ(r)) for r in results]

    def obj(s):
        return s['points_kept_whole_pooled'] + cd.DEAD_WEIGHT * s['dead_time_removed']

    def best(cands):
        top = max(obj(s) for _, s in cands)
        return max([c for c in cands if obj(c[1]) >= top - cd.TIE], key=lambda c: (c[0][0].scale_floor, obj(c[1])))
    fixed = best([c for c in scored if c[0][0].scale_floor >= 1.0])
    cap = fixed[1]['fake_per_10min'] * cd.FAKE_ALLOWANCE
    return best([c for c in scored if c[0][0].scale_floor < 1.0 and c[1]['fake_per_10min'] <= cap])[0]


def nested(results, fold_of, free):
    """Held-out rows chosen fold by fold, and the setting chosen for each fold."""
    vids = sorted(fold_of)
    rows, cfgs = {}, {}
    for k in sorted(set(fold_of.values())):
        train_v = [v for v in vids if fold_of[v] != k]
        test_v = [v for v in vids if fold_of[v] == k]
        c, p, rr, _ = select(results, train_v, free)
        cfgs[k] = {**c.__dict__, 'lead': p.lead_in_seconds, 'tail': p.tail_seconds}
        rows.update({v: rr[v] for v in test_v})
    return rows, cfgs


def main():
    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('root', type=Path, help='cross-validation root with fold<K>/oof.npz')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    probs, fold_of = load_oof(a.root)
    golds = load_golds(a.data, probs)
    probs = {v: p for v, p in probs.items() if v in golds}
    fold_of = {v: fold_of[v] for v in probs}
    free = cd.play_free(probs, golds)
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(setting_rows, [(probs, golds, free, s) for s in grid()], chunksize=8))
    rows, cfgs = nested(results, fold_of, free)
    per_fold = {k: summarize({v: rows[v] for v in rows if fold_of[v] == k}) for k in sorted(cfgs)}
    result = {'root': a.root.name, 'videos': len(rows), 'cfg_per_held_fold': cfgs, 'summary_per_held_fold': per_fold, 'summary': summarize(rows)}
    print('nested', result['summary'], flush=True)
    write_json(result, a.out, load_id_map(a.id_map))


if __name__ == '__main__':
    main()
