"""Score out-of-fold point segmenter predictions at one decode setting.

Usage: score_oof.py ROOT [ROOT …] --out … [--decode decode.json]

Every video is scored with probabilities from a fold model that never saw it,
decoded with one setting (--decode, else DecodeConfig.tuned() with the playback
padding given on the command line) and scored with
breakpoint_vision.points.scoring. For each root:

- summary: per-video means, and points kept whole pooled over all points;
- dead time removed weighted by time (pooled over every video's dead time on
  the 0.05 s grid) next to the per-video mean;
- per fold and per lighting group (from <data>/footage/manifest.tsv);
- a bootstrap over videos (resampled with replacement, seeded) for points kept
  whole (pooled) and dead time removed (mean);
- the same decode without the per-video rescale (scale_floor 1);
- videos that look edited to near-continuous play (dead time removed below
  --edited-below at this decode, with at least one point), and the summary
  without them;
- per-video rows.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from breakpoint_vision.points.decode import DecodeConfig  # noqa: E402
from breakpoint_vision.points.scoring import GRID_SECONDS, score_videos, summarize  # noqa: E402

from evaluation.common import base_parser, load_id_map, manifest_groups, write_json  # noqa: E402
from evaluation.point_segmentation.common import add_decode_args, decode_settings, load_golds, load_oof  # noqa: E402

BOOTSTRAP_SAMPLES = 2000
EDITED_BELOW = 0.05


def time_weighted_dead_removed(rows, golds):
    """Dead time removed pooled over all videos, weighting each by its dead time."""
    num = den = 0.0
    for v, r in rows.items():
        segs, (t0, t1) = golds[v]
        grid = np.arange(t0, t1, GRID_SECONDS) + GRID_SECONDS / 2
        play = np.zeros_like(grid, bool)
        for g in segs:
            play |= (grid >= g['start']) & (grid < g['end'])
        n_dead = (~play).sum()
        num += r['dead_time_removed'] * n_dead
        den += n_dead
    return num / den


def bootstrap_over_videos(rows, samples=BOOTSTRAP_SAMPLES, seed=0):
    """95% percentile intervals with videos resampled with replacement."""
    rng = np.random.default_rng(seed)
    vids = sorted(rows)
    boots_k, boots_d = [], []
    for _ in range(samples):
        pick = rng.choice(vids, len(vids), replace=True)
        n = sum(rows[v]['n_points'] for v in pick)
        boots_k.append(sum((rows[v]['points_kept_whole'] or 0) * rows[v]['n_points'] for v in pick) / max(n, 1))
        boots_d.append(np.mean([rows[v]['dead_time_removed'] for v in pick]))
    ci = lambda xs: [round(float(np.percentile(xs, 2.5)), 4), round(float(np.percentile(xs, 97.5)), 4)]  # noqa: E731
    return {'samples': samples, 'seed': seed, 'points_kept_whole_pooled': ci(boots_k), 'dead_time_removed_mean': ci(boots_d)}


def per_fold(rows, fold_of, light):
    out = {}
    for k in sorted(set(fold_of[v] for v in rows)):
        vs = [v for v in rows if fold_of[v] == k]
        s = summarize({v: rows[v] for v in vs})
        out[k] = {
            'videos': vs,
            'lighting': {g: sum(light.get(v) == g for v in vs) for g in sorted({light.get(v, '?') for v in vs})},
            'n_points': sum(rows[v]['n_points'] for v in vs),
            'points_kept_whole_pooled': s['points_kept_whole_pooled'],
            'dead_time_removed_mean': s['dead_time_removed'],
        }
    return out


def per_group(rows, light):
    groups = {}
    for v, r in rows.items():
        groups.setdefault(light.get(v, '?'), {})[v] = r
    return {g: {'videos': len(rs), 'n_points': sum(r['n_points'] for r in rs.values()), **summarize(rs)} for g, rs in sorted(groups.items())}


def edited_like(rows, below=EDITED_BELOW):
    return sorted(v for v, r in rows.items() if r['dead_time_removed'] is not None and r['dead_time_removed'] < below and r['n_points'] > 0)


def score_root(root, data, cfg, playback, light, samples, seed, edited_below):
    probs, fold_of = load_oof(root)
    golds = load_golds(data, probs)
    probs = {v: p for v, p in probs.items() if v in golds}
    rows = score_videos(probs, golds, cfg, playback)
    edited = edited_like(rows, edited_below)
    rest = {v: r for v, r in rows.items() if v not in edited}
    no_rescale = DecodeConfig(**{**cfg.__dict__, 'scale_floor': 1.0})
    return {
        'n_videos': len(rows),
        'n_points_scored': sum(r['n_points'] for r in rows.values()),
        'points_kept_whole_count': int(round(sum((r['points_kept_whole'] or 0) * r['n_points'] for r in rows.values()))),
        'videos_with_zero_points': sorted(v for v, r in rows.items() if r['n_points'] == 0),
        'summary': summarize(rows),
        'dead_time_removed_time_weighted': round(time_weighted_dead_removed(rows, golds), 4),
        'per_fold': per_fold(rows, fold_of, light),
        'per_lighting': per_group(rows, light),
        'bootstrap_over_videos_95ci': bootstrap_over_videos(rows, samples, seed),
        'no_per_video_rescale': summarize(score_videos(probs, golds, no_rescale, playback)),
        'edited_like_videos': {'rule': f'dead time removed below {edited_below} with at least one point', 'videos': edited, 'points': sum(rows[v]['n_points'] for v in edited)},
        'without_edited_like': {**summarize(rest), 'points': sum(r['n_points'] for r in rest.values()), 'videos': len(rest)},
        'per_video': {v: {'fold': fold_of[v], 'light': light.get(v, '?'), **{k: (round(x, 4) if isinstance(x, float) else x) for k, x in r.items()}} for v, r in sorted(rows.items(), key=lambda kv: (light.get(kv[0], '?'), kv[0]))},
    }


def main():
    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('roots', nargs='+', type=Path, help='cross-validation roots with fold<K>/oof.npz')
    add_decode_args(ap)
    ap.add_argument('--bootstrap', type=int, default=BOOTSTRAP_SAMPLES)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--edited-below', type=float, default=EDITED_BELOW)
    a = ap.parse_args()
    cfg, playback = decode_settings(a)
    light = manifest_groups(a.data)
    result = {'decode': cfg.__dict__, 'playback': playback.__dict__, 'runs': {}}
    for root in a.roots:
        run = score_root(root, a.data, cfg, playback, light, a.bootstrap, a.seed, a.edited_below)
        result['runs'][root.name] = run
        s = run['summary']
        print(root.name, run['n_videos'], run['n_points_scored'], 'kept_whole', s['points_kept_whole_pooled'], 'dead_removed', s['dead_time_removed'], 'time_weighted', run['dead_time_removed_time_weighted'], flush=True)
    write_json(result, a.out, load_id_map(a.id_map))


if __name__ == '__main__':
    main()
