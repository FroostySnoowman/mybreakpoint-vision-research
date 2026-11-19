"""Choose decode and playback settings from out-of-fold predictions.

Usage: choose_decode.py --out decode.json ROOT [ROOT …]   (several roots are averaged)

Every video is scored with probabilities from a model that never saw it. The
objective is points kept whole (pooled) plus DEAD_WEIGHT x dead time removed:
cutting into a rally is worse than leaving a few seconds of dead time. Among
settings within TIE of the best, the highest scale floor wins (safest on footage
with no play). A guard rejects settings that produce more than FAKE_ALLOWANCE x
the fixed decoder's spurious segments on play-free footage (each video's dead
time with 3 s around every point removed).
"""
import argparse, csv, glob, itertools, json, os, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breakpoint_vision.points.decode import DecodeConfig, decode  # noqa: E402
from breakpoint_vision.points.features import STEP_SECONDS  # noqa: E402
from breakpoint_vision.points.scoring import Playback, play_segments, score_videos, summarize  # noqa: E402

DEAD_WEIGHT = 0.5
TIE = 0.003
FAKE_ALLOWANCE = 2.0
MERGE_GAP = 2.5
QUANTILES = (0.7, 0.75, 0.8, 0.85, 0.9)
FLOORS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4)
THRESHOLDS = ((0.4, 0.2), (0.5, 0.3), (0.6, 0.3))
MIN_SECONDS = (0.6, 1.0)
MERGE_SECONDS = (0.5, 1.0)
PADDING = ((1.0, 2.0), (1.5, 2.0))


def gold(vid):
    path = f'data/gold_train/{vid}.json' if os.path.exists(f'data/gold_train/{vid}.json') else f'data/gold/{vid}.json'
    label = json.load(open(path))
    return play_segments(label), tuple(label.get('window', (300.0, 900.0)))


def load(roots):
    per_root = []
    for root in roots:
        probs = {}
        for path in sorted(glob.glob(f'{root}/fold*/oof.npz')):
            d = np.load(path)
            probs.update({v: d[v] for v in d.files})
        per_root.append(probs)
    vids = sorted(per_root[0])
    return {v: np.mean([p[v] for p in per_root], 0).astype(np.float32) for v in vids}, {v: gold(v) for v in vids}


def play_free(probs, golds):
    out = {}
    for v, p in probs.items():
        segs, (t0, t1) = golds[v]
        keep = np.zeros(len(p), bool)
        keep[int(t0 / STEP_SECONDS):int(t1 / STEP_SECONDS)] = True
        for s in segs:
            keep[max(0, int((s['start'] - 3) / STEP_SECONDS)):int((s['end'] + 3) / STEP_SECONDS)] = False
        if keep.sum() > 600:
            out[v] = p[keep]
    return out


def objective(s):
    return s['points_kept_whole_pooled'] + DEAD_WEIGHT * s['dead_time_removed']


def evaluate(args):
    probs, golds, free, setting = args
    q, floor, (on, off), mins, merge, (lead, tail) = setting
    cfg = DecodeConfig(on=on, off=off, smooth_steps=1, min_seconds=mins, merge_gap_seconds=merge, scale_quantile=q, scale_floor=floor)
    rows = score_videos(probs, golds, cfg, Playback(lead, tail, MERGE_GAP))
    minutes = sum(len(p) for p in free.values()) * STEP_SECONDS / 60
    fake = sum(len(decode(p, cfg)) for p in free.values()) / minutes * 10
    return cfg, Playback(lead, tail, MERGE_GAP), {**summarize(rows), 'fake_per_10min': round(fake, 2)}, rows


def best(results):
    top = max(objective(r[2]) for r in results)
    return max([r for r in results if objective(r[2]) >= top - TIE], key=lambda r: (r[0].scale_floor, objective(r[2])))


def report(name, result, light):
    cfg, playback, s, rows = result
    print(name, round(objective(s), 4), 'decode', json.dumps(cfg.__dict__), 'playback', json.dumps(playback.__dict__), json.dumps(s))
    print('   worst', sorted((round(r['points_kept_whole'] or 0, 2), v) for v, r in rows.items() if r['n_points'])[:8])
    groups = {}
    for v, r in rows.items():
        groups.setdefault(light.get(v, '?'), {})[v] = r
    for g, rs in sorted(groups.items()):
        sg = summarize(rs)
        print(f"   {g:10s} kept_whole {sg['points_kept_whole_pooled']:.3f} dead_removed {sg['dead_time_removed']:.3f} recall {sg['recall']:.3f} precision {sg['precision']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('roots', nargs='+')
    a = ap.parse_args()
    probs, golds = load(a.roots)
    free = play_free(probs, golds)
    settings = list(itertools.product(QUANTILES, FLOORS, THRESHOLDS, MIN_SECONDS, MERGE_SECONDS, PADDING))
    settings += list(itertools.product((0.9,), (1.0,), THRESHOLDS, MIN_SECONDS, MERGE_SECONDS, PADDING))
    with ProcessPoolExecutor(a.workers) as ex:
        results = list(ex.map(evaluate, [(probs, golds, free, s) for s in settings], chunksize=8))
    light = {row[0]: row[1] for row in csv.reader(open('data/footage/manifest.tsv'), delimiter='\t') if len(row) >= 2}
    fixed = best([r for r in results if r[0].scale_floor >= 1.0])
    cap = fixed[2]['fake_per_10min'] * FAKE_ALLOWANCE
    chosen = best([r for r in results if r[0].scale_floor < 1.0 and r[2]['fake_per_10min'] <= cap])
    print('videos', len(probs), 'play-free minutes', round(sum(len(p) for p in free.values()) * STEP_SECONDS / 60, 1), 'fake cap', round(cap, 2))
    report('fixed', fixed, light)
    report('chosen', chosen, light)
    cfg, playback, s, _ = chosen
    json.dump({**cfg.__dict__, 'lead_in_seconds': playback.lead_in_seconds, 'tail_seconds': playback.tail_seconds, 'merge_gap_seconds_playback': playback.merge_gap_seconds, 'summary': s, 'roots': a.roots}, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
