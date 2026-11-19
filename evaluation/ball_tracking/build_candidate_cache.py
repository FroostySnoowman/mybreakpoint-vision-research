"""Cache per-frame ball candidates around every labeled ball frame.

For each labeled frame in <data>/ball_gt/<video>.json, decodes the 1.5 s either
side of it from <data>/footage/raw/<video>.mp4 (at most 30 fps, area resize to
512x288), runs both detectors over that window and stores their candidates.
Trackers and smoothers can then be scored without decoding video again.

Writes <data>/ball_gt_cache/<video>.json (or --cache-dir), skipping videos
already cached:
    {"video", "frames": [{"k", "idx", "times", "app", "teacher"}]}
where `idx` is the position of the frame nearest the label in `times`.

The detectors are ONNX exports (see models.py): --app is a five-frame heatmap
model, --teacher a three-frame candidate model.
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ball_tracking.frames import context_indices, decode_window, interpolation  # noqa: E402
from evaluation.ball_tracking.metrics import load_labels, tracker_config  # noqa: E402
from evaluation.ball_tracking.models import heatmap_model, triplet_model  # noqa: E402
from evaluation.common import DEFAULT_DATA  # noqa: E402

CONTEXT = 1.5
MIN_FRAMES = 10


def frames_around(path, t, interp):
    decoded = decode_window(path, t - CONTEXT - 1.0, lambda ft: ft > t + CONTEXT, interp, start=t - CONTEXT)
    times = [ft for ft, _ in decoded]
    kept, idx = context_indices(times, t, CONTEXT)
    return [decoded[i][1] for i in kept], [times[i] for i in kept], idx


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', type=Path, default=DEFAULT_DATA)
    ap.add_argument('--cache-dir', type=Path, default=None)
    ap.add_argument('--app', required=True, help='five-frame heatmap ONNX model')
    ap.add_argument('--teacher', required=True, help='three-frame candidate ONNX model')
    ap.add_argument('--tracker-config', type=Path, default=None, help='JSON with explicit BallTrackerConfig values (default: tuned)')
    a = ap.parse_args()
    out_dir = a.cache_dir or a.data / 'ball_gt_cache'
    os.makedirs(out_dir, exist_ok=True)
    min_conf = tracker_config(a.tracker_config).min_candidate_confidence
    app, teacher = heatmap_model(a.app, min_conf), triplet_model(a.teacher, min_conf)
    interp = interpolation('area')
    for vid, gt in load_labels(a.data).items():
        target = out_dir / f'{vid}.json'
        if target.exists():
            continue
        rows = []
        for fr in gt['frames']:
            frames, times, idx = frames_around(str(a.data / 'footage' / 'raw' / f'{vid}.mp4'), fr['t'], interp)
            if len(frames) < MIN_FRAMES:
                continue
            rows.append({'k': fr['k'], 'idx': idx, 'times': times, 'app': app(frames), 'teacher': teacher(frames)})
        with open(target, 'w') as fh:
            json.dump({'video': vid, 'frames': rows}, fh)
        print(vid, len(rows), flush=True)


if __name__ == '__main__':
    main()
