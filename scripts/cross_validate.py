"""Cross-validated point segmenter training, and the final all-data model.

Usage:
  cross_validate.py fold K --root data/runs/points_cv [--features DIR] [--folds 4] [--steps 1500] [--short-boost 3]
  cross_validate.py final --out data/runs/points_final [--features DIR] [--steps 1500] [--short-boost 3]

Labeled videos have reviewed labels in data/gold_train or data/gold and a feature
file <features>/<vid>.npz. Folds are stratified by the lighting column of
data/footage/manifest.tsv. A fold trains on every feature file except its held
videos, then writes their out-of-fold probabilities to <root>/fold<K>/oof.npz.
"""
import argparse, csv, glob, json, os, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import breakpoint_vision.train.train_points as tp  # noqa: E402
from breakpoint_vision.points.model import load_segmenter  # noqa: E402

SILVER_WEIGHT = 0.15


def lighting():
    return {row[0]: row[1] for row in csv.reader(open('data/footage/manifest.tsv'), delimiter='\t') if len(row) >= 2}


def labeled(features):
    vids = {Path(p).stem for p in glob.glob('data/gold_train/*.json')} | {Path(p).stem for p in glob.glob('data/gold/*.json')}
    return sorted(v for v in vids if os.path.exists(f'{features}/{v}.npz'))


def folds(features, n):
    light = lighting()
    groups = {}
    for v in labeled(features):
        groups.setdefault(light.get(v, 'other'), []).append(v)
    out, i = [[] for _ in range(n)], 0
    for g in sorted(groups):
        for v in sorted(groups[g]):
            out[i % n].append(v)
            i += 1
    return out


def use_reviewed_labels(short_boost):
    original = tp.load_video
    tp.load_video = lambda path, thumbs_dir='data/thumbs', silver_weight=SILVER_WEIGHT: original(path, thumbs_dir, silver_weight, label_names=('train_gold', 'gold'), short_boost=short_boost)
    return original


def train(features, out, exclude, steps, seed=0, device='mps'):
    tp.train(features, out, exclude, (), steps=steps, batch=16, window_steps=600, lr=2e-3, device=device, seed=seed, visual_dim=0, eval_every=10 ** 9)


def run_fold(a):
    held = folds(a.features, a.folds)[a.k]
    out = f'{a.root}/fold{a.k}'
    original = use_reviewed_labels(a.short_boost)
    train(a.features, out, held, a.steps, seed=a.seed, device=a.device)
    model = load_segmenter(f'{out}/points_last.pt', 'cpu').to(a.device)
    np.savez(f'{out}/oof.npz', **{v: tp.predict(model, original(f'{a.features}/{v}.npz'), a.device) for v in held})
    json.dump({'held': held, 'features': a.features, 'steps': a.steps, 'short_boost': a.short_boost, 'seed': a.seed, 'device': a.device}, open(f'{out}/fold.json', 'w'))
    print('fold', a.k, 'held', held, flush=True)


def run_final(a):
    use_reviewed_labels(a.short_boost)
    train(a.features, a.out, (), a.steps, seed=a.seed, device=a.device)
    print('final model', f'{a.out}/points_last.pt', flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    f = sub.add_parser('fold')
    f.add_argument('k', type=int)
    f.add_argument('--root', required=True)
    f.add_argument('--folds', type=int, default=4)
    fin = sub.add_parser('final')
    fin.add_argument('--out', required=True)
    for p in (f, fin):
        p.add_argument('--features', default='data/features')
        p.add_argument('--steps', type=int, default=1500)
        p.add_argument('--short-boost', type=float, default=3.0)
        p.add_argument('--seed', type=int, default=0)
        p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='mps')
    a = ap.parse_args()
    run_fold(a) if a.cmd == 'fold' else run_final(a)


if __name__ == '__main__':
    main()
