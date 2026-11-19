"""Score point segmentation on one labeled video with the cross-validation metrics.

Usage:
  score_video.py --label LABEL.json --probs PROBS.npz [--key in_play] --out …
  score_video.py --label LABEL.json --segments SEGMENTS.json --out …

LABEL.json is a label in the gold format ({"video", "window", "segments"}).
With --probs, the in-play probability series (one value per feature step) is
decoded with --decode or DecodeConfig.tuned(). With --segments, already decoded
segments are scored as given: a JSON list of {"start", "end"}, or an object
holding that list under "raw_segments". Either way the segments are padded with
the playback settings and scored exactly as in score_oof.py.

Also reports the seconds of video the padded playback keeps (ranges clipped to
--duration, default the label window's end) and the predicted segments that
overlap no labeled play, with their total length.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from breakpoint_vision.points.decode import decode, evaluate  # noqa: E402
from breakpoint_vision.points.scoring import Playback, playback_plan, product_metrics  # noqa: E402

from evaluation.common import base_parser, load_id_map, write_json  # noqa: E402
from evaluation.point_segmentation.common import add_decode_args, decode_settings, load_gold  # noqa: E402


def load_probs(path, key=None):
    d = np.load(path)
    if isinstance(d, np.ndarray):
        p = d
    else:
        p = d[key] if key else d[d.files[0]]
    return p[:, 0] if p.ndim == 2 else p


def load_segments(path):
    with open(path) as fh:
        d = json.load(fh)
    return d['raw_segments'] if isinstance(d, dict) else d


def score(pred, gold, window, playback):
    return {**evaluate(pred, gold, window), **product_metrics(pred, gold, window, playback)}


def kept_seconds(plan, duration):
    return sum(min(b, duration) - a for a, b in plan)


def segments_without_play(pred, gold):
    return [s for s in pred if not any(min(s['end'], g['end']) > max(s['start'], g['start']) for g in gold)]


def main():
    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('--label', type=Path, required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--probs', type=Path, help='.npz or .npy in-play probabilities')
    src.add_argument('--segments', type=Path, help='JSON with decoded segments')
    ap.add_argument('--key', default=None, help='array name in --probs (default: the first)')
    ap.add_argument('--duration', type=float, default=None, help='video length in seconds (default: the label window end)')
    add_decode_args(ap)
    a = ap.parse_args()
    gold, window = load_gold(a.label)
    with open(a.label) as fh:
        video = json.load(fh).get('video')
    if a.probs is not None:
        cfg, playback = decode_settings(a)
        pred = decode(load_probs(a.probs, a.key), cfg)
        decode_used = cfg.__dict__
    else:
        playback = decode_settings(a)[1] if a.decode is not None else Playback(a.lead, a.tail, a.merge)
        pred = load_segments(a.segments)
        decode_used = None
    metrics = score(pred, gold, window, playback)
    plan = playback_plan(pred, playback)
    false_segments = segments_without_play(pred, gold)
    result = {
        'video': video,
        'window': list(window),
        'decode': decode_used,
        'playback': playback.__dict__,
        'segments': pred,
        'playback_ranges': plan,
        'playback_kept_seconds': round(kept_seconds(plan, a.duration if a.duration is not None else window[1]), 3),
        'segments_without_play': len(false_segments),
        'segments_without_play_seconds': round(sum(x['end'] - x['start'] for x in false_segments), 2),
        **{k: (round(x, 4) if isinstance(x, float) else x) for k, x in metrics.items()},
    }
    print(json.dumps({k: v for k, v in result.items() if k not in ('segments', 'playback_ranges')}), flush=True)
    write_json(result, a.out, load_id_map(a.id_map))


if __name__ == '__main__':
    main()
