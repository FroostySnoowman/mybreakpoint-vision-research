"""System-level ball tracking score, decoding the video for every labeled frame.

For each labeled frame in <data>/ball_gt/<video>.json, decodes the 1.5 s before
it from <data>/footage/raw/<video>.mp4 as consecutive frames at most 30 fps
apart, ending at the decoded frame nearest the label timestamp. The model and a
fresh tracker run over those frames, and the tracker's output at the last frame
is scored (see metrics.py). Labels with fewer than 10 frames are skipped.

Models are given as `[name=]kind:path` with kind `heatmap`, `triplet` or
`candidate` (see models.py). With --dump, one JSON line per labeled frame is
written for paired_mcnemar.py. Output: totals, per-group and per-video counts and
rates for each model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ball_tracking.frames import STEP_30FPS, decode_window, interpolation, sequence_indices  # noqa: E402
from evaluation.ball_tracking.metrics import (  # noqa: E402
    count_frame, load_labels, new_counts, summarize_counts, track_last, tracker_config,
)
from evaluation.ball_tracking.models import MODEL_KINDS, parse_model_spec  # noqa: E402
from evaluation.common import anonymize, base_parser, load_groups, load_id_map, write_json  # noqa: E402

LEAD = 1.5
MIN_FRAMES = 10


def sequence(path, t, interp):
    decoded = decode_window(path, t - LEAD - 1.0, lambda ft: ft > t + STEP_30FPS, interp, start=t - LEAD)
    return [decoded[i][1] for i in sequence_indices([ft for ft, _ in decoded], t, LEAD)]


def main():
    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('models', nargs='+', help='[name=]kind:path, kind in heatmap, triplet, candidate')
    ap.add_argument('--resize', choices=('area', 'linear'), default='area')
    ap.add_argument('--groups', type=Path, default=None, help='video to group (JSON or TSV); default: manifest lighting')
    ap.add_argument('--tracker-config', type=Path, default=None, help='JSON with explicit BallTrackerConfig values (default: tuned)')
    ap.add_argument('--dump', type=Path, default=None, help='JSONL with one line per labeled frame and model')
    a = ap.parse_args()
    cfg = tracker_config(a.tracker_config)
    id_map = load_id_map(a.id_map)
    groups = load_groups(a.data, a.groups)
    interp = interpolation(a.resize)
    specs = [parse_model_spec(s) for s in a.models]
    runs = {name: MODEL_KINDS[kind](path, cfg.min_candidate_confidence) for name, kind, path in specs}
    per_model = {name: {} for name in runs}
    dump = open(a.dump, 'w') if a.dump else None
    for vid, gt in load_labels(a.data).items():
        for fr in gt['frames']:
            frames = sequence(str(a.data / 'footage' / 'raw' / f'{vid}.mp4'), fr['t'], interp)
            if len(frames) < MIN_FRAMES:
                continue
            for name, run in runs.items():
                all_cands = run(frames)
                pos = track_last(all_cands, cfg)
                count_frame(per_model[name].setdefault(vid, new_counts()), fr, pos)
                if dump:
                    rec = {'model': name, 'video': vid, 'k': fr['k'], 't': fr['t'], 'visible': fr['visible'], 'x': fr.get('x'), 'y': fr.get('y'), 'n_frames': len(frames), 'pos': pos, 'last_cands': all_cands[-1]}
                    dump.write(json.dumps(anonymize(rec, id_map)) + '\n')
    if dump:
        dump.close()
    result = {'resize': a.resize, 'models': {name: summarize_counts(pv, groups) for name, pv in per_model.items()}}
    for name, s in result['models'].items():
        t = s['total']
        print(f"{name:16s} track@8px {t['hit8']}/{t['vis']} ({t['track@8px']:.4f}) track@3px {t['track@3px']:.4f} no_output {t['no_output']:.3f} false_alarm {t['false_alarm']}", flush=True)
    write_json(result, a.out, id_map)


if __name__ == '__main__':
    main()
