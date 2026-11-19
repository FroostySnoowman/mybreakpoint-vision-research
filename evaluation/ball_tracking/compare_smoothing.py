"""Online tracker versus pooled candidates and the offline Viterbi smoother.

Scores, on cached candidates (see score_cached.py):

- the online tracker on the five-frame model's candidates (the baseline),
- the online tracker on both detectors' candidates pooled per frame,
- the offline smoother (smoother.py) at its default parameters on pooled,
  five-frame and three-frame candidates, using every cached frame (1.5 s either
  side of the label),
- with --grid, the smoother on pooled candidates over a grid of none_cost,
  acc_sigma and acc_cap. The grid is scored on the same labeled frames, so its
  best row is an optimistic estimate.

Output: totals, per-group and per-video counts and rates for each method.
"""

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ball_tracking.metrics import load_cached_rows, summarize_counts, tracker_config  # noqa: E402
from evaluation.ball_tracking.score_cached import online, score, source_candidates  # noqa: E402
from evaluation.ball_tracking.smoother import SmootherParams, smooth  # noqa: E402
from evaluation.common import base_parser, load_groups, load_id_map, write_json  # noqa: E402

NONE_COSTS = (0.8, 1.2, 2.0, 3.0)
ACC_SIGMAS = (4.0, 8.0)
ACC_CAPS = (4.0, 6.0, 10.0)


def smoothed(source, params):
    def run(row):
        cands = [[tuple(x) for x in c] for c in source_candidates(row, source)]
        return smooth(cands, params)[row['idx']]
    return run


def main():
    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('--groups', type=Path, default=None, help='video to group (JSON or TSV); default: manifest lighting')
    ap.add_argument('--tracker-config', type=Path, default=None, help='JSON with explicit BallTrackerConfig values (default: tuned)')
    ap.add_argument('--grid', action='store_true', help='also score the smoother parameter grid on pooled candidates')
    a = ap.parse_args()
    cfg = tracker_config(a.tracker_config)
    groups = load_groups(a.data, a.groups)
    rows = load_cached_rows(a.data)
    default = SmootherParams()
    methods = [
        ('online app', online('app', cfg)),
        ('online pool', online('pool', cfg)),
        ('smooth pool default', smoothed('pool', default)),
        ('smooth app default', smoothed('app', default)),
        ('smooth teacher default', smoothed('teacher', default)),
    ]
    if a.grid:
        for none_cost, acc_sigma, acc_cap in itertools.product(NONE_COSTS, ACC_SIGMAS, ACC_CAPS):
            params = SmootherParams(none_cost=none_cost, acc_sigma=acc_sigma, acc_cap=acc_cap)
            methods.append((f'smooth pool none {none_cost} sig {acc_sigma} cap {acc_cap}', smoothed('pool', params)))
    result = {'labeled_frames': len(rows), 'smoother_default': default.__dict__, 'methods': {}}
    for name, method in methods:
        result['methods'][name] = summarize_counts(score(rows, method), groups)
        t = result['methods'][name]['total']
        print(f"{name:44s} track@8px {t['track@8px']:.3f} track@3px {t['track@3px']:.3f} false_alarm {t['false_alarm']}", flush=True)
    write_json(result, a.out, load_id_map(a.id_map))


if __name__ == '__main__':
    main()
