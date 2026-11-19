# Claims and how each was checked

Every quantitative statement in this repository, where it comes from and whether it reproduces. "Original output" files are from the private project folder and are not published, because they contain video IDs. "Recomputed" means the number was rebuilt for this write-up from saved predictions and labels with the script listed.

Status key:

- **Verified**: the original saved output and a fresh recomputation agree.
- **Recomputed**: no saved original exists or the original was wrong, and the number comes from a fresh computation on saved predictions.
- **Documented only**: recorded in project notes or a commit message, with no saved output. Reported with that label, never as a result.
- **Dropped**: claimed in project notes, unsupported, and left out.

## Ball tracking

| Claim | Status | Original output | Script | Data |
|---|---|---|---|---|
| 87.9% of visible frames within 8 px (218/248), 82.3% within 3 px, 20/88 false alarms | Verified | Candidate cache and stdout of the tuning script (stdout not saved); full re-decode rerun matches exactly | `evaluation/ball_tracking/` | 350 frames, 7 videos, 50 each; 248 visible, 88 no ball, 14 uncertain |
| "88% on 350 frames" (project notes) | Corrected | The only saved log said 0.88 for day footage and 0.827 overall because of a frame-timing bug | same | same |
| Teacher 84.7% (cached) and 82.7% (end-to-end) | Recomputed | | `evaluation/ball_tracking/` | same |
| Shipped vs cached teacher: frame-level McNemar p = 0.23; no equivalence claim | Recomputed | | `evaluation/ball_tracking/` | same |
| Pooling/Viterbi: +3 to +4 points at 2.4x to 2.9x false alarms | Recomputed | | `evaluation/ball_tracking/` | same; tuned on the test frames |
| Distilled student lost on the labels | Dropped | Only teacher-relative logs exist | | |
| Heavy compression hurts; other format changes inconclusive | Recomputed after the timing fix | Original variant runs (timing bug) | `evaluation/ball_tracking/` | 6 videos, 29-42 visible frames each, one run |
| 60 fps is fine | Dropped | Never isolated: the harness decimates to 30 fps | | |
| Lighting simulation calibrated against real night footage | Documented only | Calibration images exist, no saved measurement | | |

## Court detection

| Claim | Status | Original output | Script | Data |
|---|---|---|---|---|
| First model 97.5% within 15 cm on broadcast, median 4.2 cm | Verified | Fine-tuning log, "baseline broadcast" | `evaluation/court_detection/` | 200 TennisCourtDetector val frames (source videos overlap its train split) |
| First model 25.1% on amateur footage | Verified | same log, "baseline clean" | same | 359 frames, 10 videos |
| Current model 53.5% single frame, 61.9% with a 12-frame median (12.2 cm median) | Verified | Comparison table | same | same 359 frames; also used for checkpoint selection |
| Sharp 1080p through net mesh: 88% to 32% | Verified | Comparison table, high-res rows | same | 1 video |
| Author's match: court locked on 96.0% of samples | Recomputed | | demo analysis (private) | 2,395 samples at 2 per second |

## Point segmentation

| Claim | Status | Original output | Script | Data |
|---|---|---|---|---|
| 94.2% kept whole (pooled), 50.4% dead time removed (mean of videos) | Verified | Decode summary of the cross-validation run | `scripts/choose_decode.py`, `evaluation/point_segmentation/` | 42 videos, 1,351 labeled serve segments, 4 folds |
| Dead time removed, time-weighted: 58.7% | Recomputed | | `evaluation/point_segmentation/` | same |
| Held-fold decode re-selection: 91.9% / 52.4%; not fully nested model CV | Recomputed | | same | same |
| Without 5 edited videos: 92.2% / 57.2% (978 points) | Recomputed | | same | 37 videos |
| Original-selection bootstrap 95% intervals [91.1%, 96.4%] and [42%, 58%] | Recomputed | | same | resampled by video |
| Per-fold results | Verified | Per-fold out-of-fold predictions | same | 11/11/10/10 videos |
| No short-segment boost: 92.9% | Verified | Decode summary of that run | same | 42 videos |
| Averaging two models: 92.5% / 53.7% | Verified | Ensemble decode summary | same | 42 videos |
| No per-video rescale: 90.7% / 55.0% | Recomputed | | same | 42 videos |
| Bilinear resize: -1.0 point | Verified, partial | Fold 0 run | same | 11 videos |
| Compressed training copies: 86.3% vs 94.7% | Verified, partial | Folds 0-1 of an abandoned run | same | 22 videos |
| "Compressed training hurt everywhere, even at low bitrate" | Documented only | Partial run only; no low-bitrate result saved | | |
| Seed variation (seeds 0, 1, 2) | Recomputed | Two new seeds trained for this write-up | `scripts/cross_validate.py` | 42 videos x 3 seeds |
| Author's match: 42/42 serves kept whole, 54.6% dead time removed | Recomputed | | `evaluation/point_segmentation/` | 1 video, 42 labeled serves, labeled from frames without model output |

## Splits and leakage

| Claim | Status | Source |
|---|---|---|
| Random split put 99.1% of validation ball windows one frame from a training window; 99.8% of validation court frames shared a source video with training | Documented only | Commit that introduced the group split (2026-08-03) and project notes; raw logs not kept |
| Held-out ball detection accuracy 0.9067 before, 0.812 after | Documented only | same; tolerances differ, not comparable |
| 434 of 442 TennisCourtDetector validation videos also appear in its training split | Recomputed | `evaluation/data_splitting/` |
