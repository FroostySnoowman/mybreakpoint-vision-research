"""System-level ball tracking score from cached candidates.

For every cached labeled frame, a fresh tracker runs over the candidates of the
frames up to and including the one nearest the label (online: no future frames),
and its output there is scored (see metrics.py). Sources:

- `app`: candidates of the five-frame heatmap model.
- `teacher`: candidates of the three-frame model.
- `pool`: both pooled per frame (smoother.pool).

Inputs: <data>/ball_gt/*.json and <data>/ball_gt_cache/*.json
(build_candidate_cache.py). Groups come from <data>/footage/manifest.tsv unless
--groups is given. Output: totals, per-group and per-video counts and rates for
each source.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ball_tracking.metrics import (  # noqa: E402
    count_frame, load_cached_rows, new_counts, summarize_counts, track_last, tracker_config,
)
from evaluation.ball_tracking.smoother import pool  # noqa: E402
from evaluation.common import base_parser, load_groups, load_id_map, write_json  # noqa: E402

SOURCES = ('app', 'teacher', 'pool')


def source_candidates(row, source):
    if source == 'pool':
        return [pool(a, b) for a, b in zip(row['app'], row['teacher'])]
    return row[source]


def online(source, cfg):
    def run(row):
        return track_last(source_candidates(row, source)[:row['idx'] + 1], cfg)
    return run


def score(rows, method):
    per_video = {}
    for vid, fr, row in rows:
        count_frame(per_video.setdefault(vid, new_counts()), fr, method(row))
    return per_video


def main():
    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('--groups', type=Path, default=None, help='video to group (JSON or TSV); default: manifest lighting')
    ap.add_argument('--tracker-config', type=Path, default=None, help='JSON with explicit BallTrackerConfig values (default: tuned)')
    ap.add_argument('--sources', nargs='+', default=list(SOURCES), choices=SOURCES)
    a = ap.parse_args()
    cfg = tracker_config(a.tracker_config)
    groups = load_groups(a.data, a.groups)
    rows = load_cached_rows(a.data)
    result = {'labeled_frames': len(rows), 'videos': len({r[0] for r in rows}), 'sources': {}}
    for source in a.sources:
        result['sources'][source] = summarize_counts(score(rows, online(source, cfg)), groups)
        t = result['sources'][source]['total']
        print(f"{source:8s} track@8px {t['hit8']}/{t['vis']} ({t['track@8px']:.4f}) track@3px {t['track@3px']:.4f} false_alarm {t['false_alarm']} uncertain_excluded {t['uncertain']}", flush=True)
    write_json(result, a.out, load_id_map(a.id_map))


if __name__ == '__main__':
    main()
