"""Segmenter training features from ball tracks, motion and audio.

Each <tracks>/<NAME>.json holds per-frame ball samples (t, x, y, confidence),
per-frame band motion, and the frame size and rate.

Usage:
  features.py --tracks data/ball_tracks --out data/features VID [VID …]
  features.py --tracks data/ball_tracks_extra --out data/features_extra --audio data/footage/extra --no-labels NAME […]

Audio comes from <audio>/<NAME>.mp4 when --audio is given, otherwise from
data/footage/raw/<NAME>.mp4. Labels come from data/teacher/silver, data/gold and
data/gold_train unless --no-labels.
"""
import argparse, json, os, sys
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from breakpoint_vision.points.features import MOTION_BANDS, step_features, step_labels  # noqa: E402
from breakpoint_vision.points.model import SAMPLE_RATE, SAMPLES_PER_STEP  # noqa: E402

LABELS = (('silver', 'data/teacher/silver'), ('gold', 'data/gold'), ('train_gold', 'data/gold_train'))


def decode_audio(path):
    chunks = []
    with av.open(path) as c:
        if not c.streams.audio:
            return np.zeros(0, np.float32)
        rs = av.AudioResampler(format='flt', layout='mono', rate=SAMPLE_RATE)
        for f in c.decode(c.streams.audio[0]):
            for g in rs.resample(f):
                chunks.append(g.to_ndarray()[0])
        for g in rs.resample(None):
            chunks.append(g.to_ndarray()[0])
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, np.float32)


def labels_for(vid, duration):
    out = {}
    for key, ldir in LABELS:
        p = f'{ldir}/{vid}.json'
        if not os.path.exists(p):
            continue
        lab = json.load(open(p))
        out[key] = step_labels(lab['segments'], duration)
        if 'window' in lab:
            out[key + '_window'] = np.array(lab['window'], np.float32)
    return out


def build(name, a):
    out = f'{a.out}/{name}.npz'
    if os.path.exists(out):
        return
    d = json.load(open(f'{a.tracks}/{name}.json'))
    s = np.array(d['samples'], np.float64)
    mm = np.array(d['motion'], np.float64)
    duration = float(max(mm[-1, 0], s[-1, 0]) + 1.0 / max(d['fps'], 1))
    motion = np.stack([np.interp(s[:, 0], mm[:, 0], mm[:, 1 + b]) for b in range(len(MOTION_BANDS))], 1)
    feats = step_features(s[:, 0], s[:, 1], s[:, 2], s[:, 3], d['width'], d['height'], duration, motion=motion)
    audio = decode_audio(f"{a.audio or 'data/footage/raw'}/{name}.mp4")
    need = len(feats) * SAMPLES_PER_STEP
    audio = np.pad(audio[:need], (0, max(0, need - len(audio))))
    labels = {} if a.no_labels else labels_for(name, duration)
    os.makedirs(a.out, exist_ok=True)
    np.savez_compressed(out + '.tmp.npz', feats=feats, audio=audio.astype(np.float16), duration=duration, **labels)
    os.replace(out + '.tmp.npz', out)
    print(name, len(feats), sorted(labels), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tracks', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--audio', default='')
    ap.add_argument('--no-labels', action='store_true')
    ap.add_argument('names', nargs='+')
    a = ap.parse_args()
    for name in a.names:
        build(name, a)


if __name__ == '__main__':
    main()
