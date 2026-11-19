# Reproducibility

This repository is published for reading. It contains the real model, training, tracking, geometry, splitting and evaluation code, but not the data, weights or tuned settings, so it does not reproduce the reported numbers on its own.

## What you can run

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -m pytest
```

The tests run on synthetic inputs. They cover the split logic, heatmap targets, augmentation, the ball tracker, homography estimation and robust fitting, the court gate, the point features, decoding and scoring, the segmenter's shapes and the training window sampler. They use illustrative parameter values, not the tuned ones.

Tested with Python 3.12 on macOS (Apple M1 Max, PyTorch MPS). Exact package versions are pinned in `requirements.txt`.

The evaluation scripts in `evaluation/` are the ones that produced the reported numbers. With the original data in place they recompute them. Without it they show exactly how each metric is defined and computed.

## What is withheld

| Item | Why |
|---|---|
| Footage and frames | Public YouTube videos I don't hold redistribution rights to, with identifiable players |
| Labels (ball positions, court homographies, point segments) | Tied to that footage, and together with it they would make a ready-made training and test set |
| Video IDs | Replaced with anonymous labels `V01`, `V02`, … in every table |
| Trained weights | The models ship in a commercial product |
| Tuned operating values | Same reason: tracker thresholds and gates, decode thresholds and durations, court gate limits, lighting calibration constants |
| On-device export and app integration | Not research code |

The tuned values are read from `configs/tuned.yaml`, which is not included. Each affected config has a `tuned()` constructor, for example `BallTrackerConfig.tuned()` or `DecodeConfig.tuned()`, which raises a clear error when the file is missing. You can pass your own values to the dataclass constructors instead.

Withheld names, by section:

- `ball_tracker`: `min_candidate_confidence`, `seed_confidence`, `keep_confidence`, `gate_fraction`, `gate_growth_fraction`, `coast_frames`, `confirm_detections`
- `decode`: `on`, `off`, `smooth_steps`, `min_seconds`, `merge_gap_seconds`, `scale_quantile`, `scale_floor`
- `court_gate`: `min_keypoints`, `max_reprojection_error_m`, `min_court_frame_fraction`, `max_court_frame_fraction`, `min_consecutive_detections`, `max_court_age_seconds`, `ball_runoff_side_m`, `ball_runoff_back_m`, `ball_up_margin_fraction`
- `lighting`: `night_exposure`, `harsh_exposure`, `read_noise`, `dark_read_noise`, `photons_at_white`, `residual_exposure`, `capture_area_ratio`, `denoise_sigma`

Everything that defines a metric stays public: the playback padding and the 95% kept-whole rule, the 8 px and 3 px tolerances, the 15 cm court threshold and grid, the split rules and the fold assignment.

## Running training and evaluation with your own data

The code expects the same layout as the original project: a `data/` folder inside the repository root. Clips are imported with `breakpoint_vision/data/import_tracknet.py` (TrackNet CSV format) and `breakpoint_vision/data/import_court.py` (TennisCourtDetector JSON).

```python
from breakpoint_vision.config import (
    load_ball_model_config, load_ball_train_config,
    load_court_model_config, load_court_train_config,
)
from breakpoint_vision.train.train_ball import train_ball
from breakpoint_vision.train.train_court import train_court

train_ball(load_ball_model_config("configs/ball_train.yaml"), load_ball_train_config("configs/ball_train.yaml"), data_root="data/ball")
train_court(load_court_model_config("configs/court_train.yaml"), load_court_train_config("configs/court_train.yaml"), data_root="data/court")
```

Point segmenter cross-validation, from features to decode selection:

```bash
python scripts/features.py --tracks data/ball_tracks --out data/features VIDEO_ID [MORE_VIDEO_IDS]
for k in 0 1 2 3; do
  python scripts/cross_validate.py fold $k --root data/runs/points_cv --features data/features --steps 1500 --short-boost 3
done
python scripts/choose_decode.py --out data/runs/points_cv/decode.json data/runs/points_cv
```

Labels go in `data/gold/<video>.json` and `data/gold_train/<video>.json` (`{"video", "window", "segments": [{"start", "end", "kind"}]}`), and the lighting groups for stratification in `data/footage/manifest.tsv`.

## Reproducibility of each reported result

[claims.md](claims.md) lists every number with its source file, the script that computes it and whether it can be recomputed today. In short:

- Ball tracking, court detection and point segmentation results all recompute from saved predictions and labels with the scripts in `evaluation/`.
- The point-segmenter cross-validation was also retrained from scratch with two more seeds.
- The training runs behind the evaluated ball weights and the first court model can't be repeated: their data and logs were not kept.
- Results that exist only as notes, with no saved output, are not reported.

## Seeds

Every random choice takes a seed:

- Ball and court training: `seed: 1337` in the configs.
- Point segmenter: seed 0 for the original runs, seeds 1 and 2 for the repeats.
- Label frame sampling: seed 0.
- The group split: seeded hash.

Training on Apple MPS is not bit-for-bit deterministic, so a rerun with the same seed gives close but not identical numbers.

## Build and check the publication

The PDF sources are in `report/`. Building the documents does not require the private models or datasets:

```bash
python3.12 -m venv .venv-report
source .venv-report/bin/activate
pip install -r requirements-report.txt
python scripts/build_report.py
python scripts/check_publication.py
```

The compiler is pinned to the version used for this revision. It builds the full report and a concise research summary. The consistency check reconciles public counts, fold mappings, rounded headline values, local Markdown links, and bibliography references. It does not rerun inference or independently verify private evidence.

To train the point folds with explicit seeds and hardware using your own prepared data:

```bash
for seed in 0 1 2; do
  for fold in 0 1 2 3; do
    python scripts/cross_validate.py fold "$fold" --root "data/runs/points_seed${seed}" --features data/features --steps 1500 --short-boost 3 --seed "$seed" --device cpu
  done
done
```

Use `--device mps` or `--device cuda` when available. The historical runs used MPS; seed control does not guarantee bit-for-bit determinism across backends. These flags replace a private seed wrapper and retain the original default seed and device. Fold metadata records both.

The report's historical term "nested decode" denotes re-selection on existing predictions. It does not provide the isolation of fully nested model training. The public evaluation scripts are inspectable, but private labels, predictions, weights, and operating values remain required to recompute the experimental metrics.
