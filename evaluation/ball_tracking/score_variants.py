"""Ball and court scores of the on-device models on another encoding of a test video.

Usage: score_variants.py VIDEO VARIANT_MP4 OFFSET [--label NAME] --ball-onnx … --court-onnx … --out …

OFFSET is the number of seconds to add to a variant timestamp to get the
labeled (reference) timestamp.

Ball: for each labeled frame of VIDEO, frames of the variant over the 1.5 s
before it are anchored at the decoded frame nearest the label and thinned
backward to 30 fps, resized to 512x288 (bilinear by default), and run through
the five-frame candidate model and a fresh tracker; the tracker's output at the
anchor frame is scored (see metrics.py).

Court: for each <data>/court_eval frame of VIDEO in time order, the first
variant frame at or after its timestamp goes through the court model; keypoints
are aggregated with the 12-frame running median and the court is fitted and
scored with the plane error (see ../court_detection/metrics.py).

Output: one JSON object with the ball and court scores. With --dump, one JSON
line per labeled ball frame is appended for paired_mcnemar.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.ball_tracking.frames import decode_window, interpolation, variant_indices  # noqa: E402
from evaluation.ball_tracking.metrics import count_frame, new_counts, track_last, tracker_config  # noqa: E402
from evaluation.ball_tracking.models import candidate_model, session  # noqa: E402
from evaluation.court_detection.metrics import (  # noqa: E402
    AGGREGATED_THRESHOLD, PASS_CM, TemporalMedian, load_rows, plane_error_cm, points_only_fit,
)
from evaluation.common import anonymize, base_parser, load_id_map, write_json  # noqa: E402

LEAD = 1.5
MIN_FRAMES = 10


def kept_frames(path, t0, t1, interp):
    decoded = decode_window(path, t0 - 1.0, lambda ft: ft >= t1, interp, start=t0)
    return [decoded[i] for i in variant_indices([ft for ft, _ in decoded], t1)]


def ball_scores(data, vid, path, offset, label, onnx, cfg, interp, dump, id_map):
    run = candidate_model(onnx, cfg.min_candidate_confidence)
    gt = json.load(open(f'{data}/ball_gt/{vid}.json'))
    g = new_counts()
    for fr in gt['frames']:
        tv = fr['t'] - offset
        frames = kept_frames(path, tv - LEAD, tv, interp)
        if len(frames) < MIN_FRAMES:
            continue
        pos = track_last(run([f for _, f in frames]), cfg)
        if dump:
            rec = {'video': vid, 'variant': label, 'offset': offset, 'k': fr['k'], 't': fr['t'], 'frame_time': frames[-1][0], 'visible': fr['visible'], 'x': fr.get('x'), 'y': fr.get('y'), 'n_frames': len(frames), 'pos': pos}
            dump.write(json.dumps(anonymize(rec, id_map)) + '\n')
        count_frame(g, fr, pos)
    v = max(g['vis'], 1)
    return {'ball_track@8px': round(g['hit8'] / v, 3), 'ball_track@3px': round(g['hit3'] / v, 3), 'ball_no_output': round(g['none'] / v, 3), 'ball_false_alarm': f"{g['fa']}/{g['neg']}", 'ball_visible': g['vis']}


def court_scores(data, vid, path, offset, onnx, interp):
    import av
    import cv2

    rows = sorted([r for r in load_rows([f'{data}/court_eval']) if r['video'] == vid], key=lambda r: r['t'])
    if not rows:
        return {}
    sess = session(onnx)
    agg_state, errs = TemporalMedian(), []
    with av.open(path) as c:
        s = c.streams.video[0]
        for r in rows:
            tv = r['t'] - offset
            c.seek(int(max(0, tv - 1.0) / s.time_base), stream=s)
            img = None
            for f in c.decode(s):
                if f.time is not None and f.time >= tv:
                    img = cv2.resize(f.to_ndarray(format='rgb24'), (512, 288), interpolation=interp)
                    break
            if img is None:
                continue
            pk = sess.run(None, {'image': (img.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]})[0][0]
            agg = agg_state.add(pk[:14])
            if agg is None:
                continue
            errs.append(plane_error_cm(points_only_fit(agg, AGGREGATED_THRESHOLD), r))
    e = np.array(errs)
    return {'court_within_15cm': round(float((e <= PASS_CM).mean()), 3) if len(e) else None, 'court_p50_cm': round(float(np.median(e)), 1) if len(e) else None}


def main():
    import av

    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('video', help='labeled video id')
    ap.add_argument('variant', help='path to the variant encoding')
    ap.add_argument('offset', type=float, help='seconds added to a variant timestamp to get the labeled timestamp')
    ap.add_argument('--label', default=None, help='variant name (default: file stem)')
    ap.add_argument('--ball-onnx', required=True, help='five-frame candidate model (input "frames")')
    ap.add_argument('--court-onnx', required=True, help='court keypoint model (input "image")')
    ap.add_argument('--resize', choices=('area', 'linear'), default='linear')
    ap.add_argument('--tracker-config', type=Path, default=None, help='JSON with explicit BallTrackerConfig values (default: tuned)')
    ap.add_argument('--dump', type=Path, default=None, help='JSONL to append one line per labeled ball frame')
    a = ap.parse_args()
    cfg = tracker_config(a.tracker_config)
    id_map = load_id_map(a.id_map)
    interp = interpolation(a.resize)
    label = a.label or Path(a.variant).stem
    with av.open(a.variant) as c:
        s = c.streams.video[0]
        fmt = f'{s.width}x{s.height}@{float(s.average_rate):.2f}'
    dump = open(a.dump, 'a') if a.dump else None
    try:
        ball = ball_scores(a.data, a.video, a.variant, a.offset, label, a.ball_onnx, cfg, interp, dump, id_map)
    finally:
        if dump:
            dump.close()
    result = {'video': a.video, 'variant': label, 'format': fmt, 'resize': a.resize, **ball, **court_scores(a.data, a.video, a.variant, a.offset, a.court_onnx, interp)}
    print(json.dumps(anonymize(result, id_map)), flush=True)
    write_json(result, a.out, id_map)


if __name__ == '__main__':
    main()
