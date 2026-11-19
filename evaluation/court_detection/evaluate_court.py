"""Court keypoint model accuracy: plane error per frame, single-frame and temporal.

Runs a CourtNet checkpoint on labeled 512x288 frames and scores the fitted court
with the plane error in metrics.py. Two label sets:

- `amateur`: <data>/court_eval (frames from test videos, each with a timestamp
  `t`). Reports single-frame and temporal-median results, overall and per video.
  With --hires-offsets (JSON: video id to seconds to subtract from `t`), each
  frame is instead decoded from <data>/footage/hires/<video>.mp4 and resized;
  labels without a high-resolution source are skipped.
- `broadcast`: 200 frames drawn with seed 0 from <data>/court_tcd/val, the
  imported TennisCourtDetector validation split. Single-frame only.

Single-frame results also include keypoint recall within 3 px (in-frame labeled
keypoints), the false positive rate on keypoints outside the frame, and the
share of frames whose fitted court reprojects every in-frame keypoint within
4 px.

Output: overall and per-video summaries.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evaluation.court_detection.metrics import (  # noqa: E402
    CONF_THRESHOLD, W, H, AGGREGATED_THRESHOLD, PASS_CM, TemporalMedian, calib_error, full_projection, in_frame,
    load_rows, peaks, plane_error_cm, plane_summary, points_array, points_only_fit,
)
from evaluation.common import base_parser, load_id_map, write_json  # noqa: E402

BROADCAST_FRAMES = 200
KEYPOINT_PX = 3.0
CALIB_PX = 4.0


def load_model(path, device):
    import torch

    from breakpoint_vision.models.court_net import CourtNet

    sd = torch.load(path, map_location='cpu')['model_state_dict']
    net = CourtNet(num_keypoints=14, base=32, output_channels=sd['head.weight'].shape[0])
    net.load_state_dict(sd)
    return net.to(device).eval()


def broadcast_rows(data, n=BROADCAST_FRAMES):
    rows = load_rows([f'{data}/court_tcd/val'])
    return [rows[i] for i in np.random.default_rng(0).choice(len(rows), min(n, len(rows)), replace=False)]


def labeled_image(row):
    import cv2

    return cv2.cvtColor(cv2.imread(row['path']), cv2.COLOR_BGR2RGB)


def hires_image(data, offsets, interp, row):
    import av
    import cv2

    vid = row['video']
    path = Path(data) / 'footage' / 'hires' / f'{vid}.mp4'
    if vid not in offsets or not path.exists():
        return None
    tv = row['t'] - offsets[vid]
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0, tv - 1.0) / s.time_base), stream=s)
        for f in c.decode(s):
            if f.time is not None and f.time >= tv:
                return cv2.resize(f.to_ndarray(format='rgb24'), (W, H), interpolation=interp)
    return None


def evaluate(net, rows, image_of, device, temporal):
    import torch

    by_video = {}
    for r in rows:
        by_video.setdefault(r['video'], []).append(r)
    single, temp = [], []
    kp = {'hit': 0, 'total': 0, 'fp': 0, 'absent': 0}
    calib = {'frames': 0, 'fitted': 0, 'close': 0, 'max_err': []}
    per_video = {}
    with torch.no_grad():
        for vid, rs in by_video.items():
            if temporal:
                rs = sorted(rs, key=lambda r: r.get('t', 0))
            agg_state = TemporalMedian()
            v = {'single': [], 'temporal': [], 'calib_close': 0}
            for r in rs:
                img = image_of(r)
                if img is None:
                    continue
                x = torch.from_numpy(img).float().div(255.0).permute(2, 0, 1)[None].to(device)
                hm = torch.sigmoid(net(x))[0].float().cpu().numpy()
                pk = peaks(hm[:15])
                e = plane_error_cm(points_only_fit(pk, CONF_THRESHOLD), r)
                single.append(e)
                v['single'].append(e)
                truth = in_frame(points_array(r))
                for i in range(14):
                    if np.isnan(truth[i, 0]):
                        kp['absent'] += 1
                        kp['fp'] += int(pk[i, 2] >= CONF_THRESHOLD)
                    else:
                        kp['total'] += 1
                        kp['hit'] += int(pk[i, 2] >= CONF_THRESHOLD and np.hypot(*(pk[i, :2] - truth[i])) <= KEYPOINT_PX)
                err, _ = calib_error(pk, full_projection(r), CONF_THRESHOLD)
                calib['frames'] += 1
                if err is not None:
                    calib['fitted'] += 1
                    calib['max_err'].append(err)
                    if err <= CALIB_PX:
                        calib['close'] += 1
                        v['calib_close'] += 1
                if temporal:
                    agg = agg_state.add(pk)
                    if agg is not None:
                        te = plane_error_cm(points_only_fit(agg, AGGREGATED_THRESHOLD), r)
                        temp.append(te)
                        v['temporal'].append(te)
            per_video[vid] = {
                'frames': len(v['single']),
                'single_within_15cm': round(float(np.mean(np.array(v['single']) <= PASS_CM)), 2) if v['single'] else None,
                'calib_within_4px': round(v['calib_close'] / max(1, len(v['single'])), 2),
            }
            if temporal:
                per_video[vid]['temporal_within_15cm'] = round(float(np.mean(np.array(v['temporal']) <= PASS_CM)) if v['temporal'] else 0.0, 2)
    out = {
        'single_frame': plane_summary(single),
        'kp_recall@3px': round(kp['hit'] / max(1, kp['total']), 3),
        'kp_false_pos': round(kp['fp'] / max(1, kp['absent']), 3),
        'calib_rate': round(calib['fitted'] / max(1, calib['frames']), 3),
        'calib_within_4px': round(calib['close'] / max(1, calib['frames']), 3),
        'median_max_err_px': round(float(np.median(calib['max_err'])), 2) if calib['max_err'] else None,
        'per_video': per_video,
    }
    if temporal:
        out['temporal_median'] = plane_summary(temp)
    return out


def main():
    ap = base_parser(__doc__.split('\n')[0])
    ap.add_argument('--checkpoint', required=True, help='CourtNet checkpoint with model_state_dict')
    ap.add_argument('--set', choices=('amateur', 'broadcast'), default='amateur')
    ap.add_argument('--hires-offsets', type=Path, default=None, help='JSON: video id to seconds subtracted from t (amateur only)')
    ap.add_argument('--resize', choices=('linear', 'area'), default='linear', help='resize for --hires-offsets frames')
    ap.add_argument('--device', default='cpu')
    a = ap.parse_args()
    net = load_model(a.checkpoint, a.device)
    if a.set == 'broadcast':
        rows, image_of, temporal = broadcast_rows(a.data), labeled_image, False
    else:
        rows, temporal = load_rows([f'{a.data}/court_eval']), True
        image_of = labeled_image
        if a.hires_offsets:
            import cv2

            offsets = json.load(open(a.hires_offsets))
            interp = cv2.INTER_AREA if a.resize == 'area' else cv2.INTER_LINEAR
            image_of = lambda r: hires_image(a.data, offsets, interp, r)  # noqa: E731
    result = {'set': a.set, 'frames_labeled': len(rows), 'source': 'hires' if a.hires_offsets else 'labeled frames', **evaluate(net, rows, image_of, a.device, temporal)}
    print(json.dumps({k: v for k, v in result.items() if k != 'per_video'}), flush=True)
    write_json(result, a.out, load_id_map(a.id_map))


if __name__ == '__main__':
    main()
