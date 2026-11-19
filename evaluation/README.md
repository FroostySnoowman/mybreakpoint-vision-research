# Evaluation

The scripts that computed the numbers in [docs/results.md](../docs/results.md). The metric definitions and decision rules (tolerances, thresholds, tie rules, padding, the 95% kept-whole rule) are the ones the reported numbers were computed with.

## Running

From the repository root, after `pip install -e .`:

```bash
python evaluation/ball_tracking/score_cached.py --data data --out results/ball_cached.json --id-map ids.json
```

Every script takes:

- `--data`: the data root, default `data/` inside the repository root, laid out as in the original project (below).
- `--out`: the JSON file to write.
- `--id-map`: optional JSON object from real video id to anonymous label (`{"<id>": "V01", …}`). Every per-video key and video name in the output is replaced, so published tables can use V01, V02, ….

Tuned values are never written into these scripts. The tracker reads `BallTrackerConfig.tuned()` and decoding reads `DecodeConfig.tuned()`, both from the withheld `configs/tuned.yaml`. To run without it, pass explicit values: `--tracker-config tracker.json` (the `BallTrackerConfig` fields) or `--decode decode.json` (as written by `scripts/choose_decode.py`). Playback padding defaults to the public 1.0 s lead-in, 2.0 s tail and 2.5 s merge gap.

The scripts that run models (`build_candidate_cache.py`, `score_redecode.py`, `score_variants.py`) also need `onnxruntime`, and `evaluate_court.py` needs a CourtNet checkpoint. Everything else runs on saved predictions and labels.

## Which script produces which number

| Result in docs/results.md | Script | Inputs |
|---|---|---|
| Ball tracking: 218 of 248 visible frames within 8 px (87.9%), within 3 px, false alarms on no-ball frames (20 of 88), per lighting group and per video | `ball_tracking/score_cached.py` (source `app`) | `ball_gt/`, `ball_gt_cache/` |
| Ball tracking: three-frame model and pooled candidates through the same tracker | `ball_tracking/score_cached.py` (sources `teacher`, `pool`) | as above |
| Candidate cache used by the rows above | `ball_tracking/build_candidate_cache.py` | `ball_gt/`, `footage/raw/<video>.mp4`, two ONNX models |
| Ball tracking with every labeled frame decoded from video | `ball_tracking/score_redecode.py` | `ball_gt/`, `footage/raw/`, ONNX models |
| Ball and court results on other encodings (resolution, bitrate, frame rate, darkening, resize method) | `ball_tracking/score_variants.py`, once per video and variant | `ball_gt/`, `court_eval/`, the variant file and its time offset, on-device ONNX models |
| Five-frame model versus three-frame model, paired (exact McNemar, Wilson intervals); paired comparisons between encodings | `ball_tracking/paired_mcnemar.py` (`cached`, or `dumps` from `--dump` files) | as above |
| Pooled candidates and the offline Viterbi smoother versus the online tracker, including the smoother grid | `ball_tracking/compare_smoothing.py --grid` | `ball_gt/`, `ball_gt_cache/` |
| Court: share of frames within 15 cm and median plane error, single frame, on amateur footage and on the broadcast validation frames | `court_detection/evaluate_court.py --set amateur` / `--set broadcast` | `court_eval/` or `court_tcd/val/`, checkpoint |
| Court: 12-frame temporal median, per video; high-resolution source frames | `court_detection/evaluate_court.py --set amateur [--hires-offsets offsets.json]` | `court_eval/`, `footage/hires/`, checkpoint |
| Points: 94.2% kept whole (pooled over 1,351 points), 50.4% dead time removed (per-video mean), time-weighted dead time, per fold, per lighting group, bootstrap intervals over videos, no per-video rescale, without the edited videos, extra seeds | `point_segmentation/score_oof.py ROOT [ROOT …] --decode ROOT/decode.json` | `runs/<cv root>/fold<K>/oof.npz`, `gold/`, `gold_train/`, `footage/manifest.tsv` |
| Points: nested decode selection (choose on three folds, score the fourth): 91.9% kept whole, 52.4% dead time removed | `point_segmentation/nested_decode.py ROOT` | as above |
| Points on the author's match: serves kept whole (42 of 42), dead time removed, seconds kept by the padded playback, predicted segments with no labeled play | `point_segmentation/score_video.py --label … --probs … --duration …` (or `--segments …`) | a label in the gold format and the model's in-play probabilities |
| Data splits: TennisCourtDetector validation videos also in its training split (434 of 442), and for the 200 broadcast evaluation frames | `data_splitting/group_overlap.py tcd` | `court_tcd/train/`, `court_tcd/val/` |
| Data splits: share of validation samples whose clip or source video is also in training, for any split | `data_splitting/group_overlap.py labels …` or `lists …` | two label files or group lists |

The decode setting itself is chosen by `scripts/choose_decode.py`; the scripts here import its grid, objective and tie rule.

The ball scores omit the court gate. The paired p = 0.23 compares cached predictions (218 versus 210 hits), not the separately decoded teacher (205 hits). `nested_decode.py` reselects decoding on existing predictions and does not implement nested model retraining. See the interpretation limits in [results.md](../docs/results.md).

## Data layout

```
data/
  ball_gt/<video>.json          {"video", "frames": [{"k", "t", "visible", "x", "y", "source"}]}
  ball_gt_cache/<video>.json    written by build_candidate_cache.py
  court_eval/labels.json        {"images": [{"file", "video", "t", "keypoints"}]} plus the frames
  court_tcd/{train,val}/        imported TennisCourtDetector splits, same format
  gold/<video>.json             {"video", "window", "segments": [{"start", "end", "kind"}]}
  gold_train/<video>.json       same format
  runs/<cv root>/fold<K>/oof.npz
  footage/manifest.tsv          video id, lighting group, …
  footage/raw/<video>.mp4, footage/hires/<video>.mp4
```

In the candidate cache, `app` holds the evaluated five-frame model's candidates and `teacher` the three-frame teacher's; the ball scripts use the same names for their sources. Ball positions are in 512x288 model pixels. `visible` is `true`, `false`, `"not_in_play"` or `"uncertain"`; uncertain frames are excluded from every ball metric.

## Notes on the published versions

- **Label timing fix.** The first versions of the full re-decode and format-variant scripts matched each label to the first decoded frame at or after its timestamp. Because timestamps are rounded to the millisecond, that was sometimes the next or previous frame. The published `score_redecode.py` and `score_variants.py` are the fixed versions: each label is matched to the nearest decoded frame. The cached-candidate scores were never affected.
- **Tracker.** The original scripts used a separate Python port of the tracker working on `(x, y, confidence)` tuples, with the same values `BallTrackerConfig.tuned()` returns. It makes the same decisions as `breakpoint_vision.ball_track.BallTracker` on candidate lists, so the published scripts use `BallTracker`; `tests/test_eval_ball_metrics.py` checks the two agree. The candidate confidence floor is the tracker's `min_candidate_confidence`.
- **Three-frame model.** The original scripts exported the three-frame network to ONNX together with a candidate-extraction head before scoring. That export code is not part of this release; the scripts take the exported ONNX file.
- **Lighting groups.** The ball scripts originally grouped videos with a fixed table that put one indoor video in the day group. Groups now come from the manifest; pass `--groups` with a JSON or TSV map to reproduce the original grouping. Totals do not change.
- **Court plane error.** Two original scripts had slightly different copies of the metric; one also required projected grid points to lie in front of the camera and returned a failure when no grid point was in frame. On every labeled frame of both court sets every in-frame grid point is in front of the camera and at least one is in frame, so the copies agree; the published version has both guards.
- **Court lighting variants.** The single-frame court script could also score synthetically darkened copies of each frame. That depends on withheld lighting constants and is not part of the reported results, so it is left out.
- **Single match.** The match's segments came from the same decoder and playback padding, but no standalone script scored them; `score_video.py` was written for this release and scores one labeled video with the same functions the cross-validation uses.
- **Time-weighted dead time.** Two original analyses computed it with slightly different code and got the same value; `score_oof.py` keeps the one that reuses the per-video rows.
