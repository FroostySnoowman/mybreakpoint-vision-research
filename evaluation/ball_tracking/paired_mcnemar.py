"""Paired comparison of two ball tracking runs on the same labeled frames.

Each visible labeled frame is a hit (tracker output within 8 px) or not under
each run. Reports both hit counts with 95% Wilson intervals, the discordant
pairs (frames only one run hits) and the two-sided exact McNemar p-value.

Two input modes:

- `cached`: two online sources scored on cached candidates (score_cached.py),
  for example `--a app --b teacher`. Also reports hits by label source.
- `dumps`: two per-frame JSONL dumps written by score_redecode.py or
  score_variants.py with --dump. Frames are paired by (video, k).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ball_tracking.metrics import (  # noqa: E402
    exact_mcnemar_p, is_hit, load_cached_rows, paired_counts, tracker_config, wilson_interval,
)
from evaluation.ball_tracking.score_cached import SOURCES, online  # noqa: E402
from evaluation.common import base_parser, load_id_map, write_json  # noqa: E402


def cached_hits(rows, method):
    return {(vid, fr['k']): is_hit(fr, method(row)) for vid, fr, row in rows if fr['visible'] is True}


def dump_hits(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line.startswith('{'):
                continue
            r = json.loads(line)
            if r['visible'] is True:
                out[(r['video'], r['k'])] = is_hit(r, r['pos'])
    return out


def compare(a, b):
    common = [k for k in a if k in b]
    ka, kb = sum(a[k] for k in common), sum(b[k] for k in common)
    n = len(common)
    only_a, only_b = paired_counts(a, b)
    return {
        'visible': n,
        'a_hits': int(ka), 'a_wilson95': [round(x, 4) for x in wilson_interval(ka, n)] if n else None,
        'b_hits': int(kb), 'b_wilson95': [round(x, 4) for x in wilson_interval(kb, n)] if n else None,
        'only_a_hit': only_a, 'only_b_hit': only_b,
        'exact_mcnemar_p': exact_mcnemar_p(only_a, only_b),
    }


def main():
    ap = base_parser(__doc__.split('\n')[0])
    sub = ap.add_subparsers(dest='mode', required=True)
    c = sub.add_parser('cached')
    c.add_argument('--a', default='app', choices=SOURCES)
    c.add_argument('--b', default='teacher', choices=SOURCES)
    c.add_argument('--tracker-config', type=Path, default=None, help='JSON with explicit BallTrackerConfig values (default: tuned)')
    d = sub.add_parser('dumps')
    d.add_argument('a_dump', type=Path)
    d.add_argument('b_dump', type=Path)
    a = ap.parse_args()
    if a.mode == 'cached':
        cfg = tracker_config(a.tracker_config)
        rows = load_cached_rows(a.data)
        ha, hb = cached_hits(rows, online(a.a, cfg)), cached_hits(rows, online(a.b, cfg))
        result = {'a': a.a, 'b': a.b, **compare(ha, hb), 'by_label_source': {}}
        sources = {(vid, fr['k']): fr.get('source') for vid, fr, _ in rows if fr['visible'] is True}
        for s in sorted({str(v) for v in sources.values()}):
            keys = [k for k, v in sources.items() if str(v) == s]
            result['by_label_source'][s] = {'n': len(keys), 'a_hits': int(np.sum([ha[k] for k in keys])), 'b_hits': int(np.sum([hb[k] for k in keys]))}
    else:
        result = {'a': a.a_dump.name, 'b': a.b_dump.name, **compare(dump_hits(a.a_dump), dump_hits(a.b_dump))}
    print(json.dumps(result), flush=True)
    write_json(result, a.out, load_id_map(a.id_map))


if __name__ == '__main__':
    main()
