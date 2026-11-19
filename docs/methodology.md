# Methodology

This document describes the three systems and how each was evaluated. Numbers are in [results.md](results.md); where each one comes from is in [claims.md](claims.md).

The constraint behind most design choices: everything runs on the phone, offline, on amateur footage filmed from a fence or tripod, in any light. That rules out large models and cloud inference, and it means broadcast footage is not representative of the input.

## 1. Ball tracking

### Model

`breakpoint_vision/models/tracknet.py` is a TrackNet-style encoder-decoder (after Huang et al., 2019, and TrackNetV2). It is an independent implementation, not a port of the original code.

- Input: 5 consecutive RGB frames resized to 512x288 and stacked on the channel axis (15 channels), scaled to 0..1. Frames are resized with area interpolation, the same way the phone resizes them.
- Encoder: four VGG-style stages (2, 2, 3 and 3 convolutions, base width 16, batch norm, ReLU, max pooling).
- Decoder: bilinear upsampling with skip connections back to full resolution.
- Output: one heatmap for the last input frame. The output bias is initialized to a 1% prior so the focal loss starts from a stable gradient.
- Loss: a Gaussian-target focal loss (`gaussian_focal_loss`), with a plain focal BCE also available.

### From heatmap to position

`breakpoint_vision/ball_track.py` turns heatmaps into a trajectory instead of reading the argmax of each frame:

1. Take the top few local maxima per frame, collapse plateaus and refine each to sub-pixel by a local centroid.
2. Run a constant-velocity tracker over the candidates. A track starts only after consecutive confident detections. Once confirmed, a weaker candidate is accepted if it lies inside a gate around the predicted position. The gate widens for every frame the track coasts without a detection, and the track ends after a few empty frames.

Distances are fractions of the input width, so the same settings work at any model resolution. The tuned confidence levels, gate sizes and frame counts are withheld (see [reproducibility.md](reproducibility.md)).

### Evaluation

- Labels: 350 frames, 50 per video, from 7 test videos covering day, dusk and night play. Frames are sampled uniformly over in-play time with a fixed seed, independent of any model. Each frame is marked `visible`, `false` (no ball in view), `not_in_play` or `uncertain`, with a position for visible balls.
- Detector-plus-tracker score (without court gating): for each labeled frame, the model and tracker run causally over the preceding 1.5 s at 30 fps, and the tracker's output at the labeled frame is scored. This evaluates candidate extraction and temporal tracking. The scoring harness does not apply the court-presence gate, so it does not measure the complete live overlay.
- Metric: share of visible frames where the output is within 8 px (and 3 px) of the label, measured in 512x288 model pixels. 8 px at that scale is about 10 to 13 px in the 480p and 360p source footage. No output on a visible frame counts as a miss. Any output on a `false` or `not_in_play` frame counts as a false alarm and is reported separately. `uncertain` frames are excluded.
- Code: `evaluation/ball_tracking/`.

The locally acquired evaluation files postdate training. That fact alone does not prove source-video disjointness: the full training manifest for these weights was not retained, and upstream datasets could contain the same source footage. The tracker settings, however, were checked on these same frames (see [limitations.md](limitations.md)).

## 2. Court detection and calibration

### Model

`breakpoint_vision/models/court_net.py` is a U-Net style keypoint network (base width 32, four encoder stages, three decoder stages). It predicts one heatmap per court keypoint at 512x288; peaks are refined to sub-pixel. The 14-keypoint layout follows the public TennisCourtDetector dataset; the network is an independent implementation.

### Court coordinate system

`breakpoint_vision/court.py` defines the court in meters with the origin at the center of the net: x across the court, y along it. It uses the ITF dimensions: 23.77 m long, 10.97 m doubles width, 8.23 m singles width, service line 6.40 m from the net. Each keypoint (corners, singles sideline ends, service line ends and center service line ends) has a fixed position in that frame.

### Homography

`breakpoint_vision/geometry.py` maps image points to court coordinates:

1. A normalized direct linear transform (DLT) solves the homography from image keypoints to their court positions.
2. A robust fit first trims the single worst point repeatedly while any error exceeds a threshold on the court plane. It then tries every 4-point subset, keeps the largest consensus set within 0.30 m and refits on it. At least 6 inliers are required, otherwise no homography is returned.
3. Errors are measured on the court plane in meters, not in pixels, so a far-side keypoint error counts as much as a near-side one in real distance.

`breakpoint_vision/court_gate.py` decides whether a court is in view before ball positions are trusted. A detection must have enough keypoints, low reprojection error, a convex court quad of plausible size, and persistence over consecutive detections; it expires after a short time. The exact limits are withheld.

### Evaluation

- Metric: for each frame, project a 12x24 grid of court points through the labeled homography, map them back through the predicted one, and take the median distance on the court plane. A frame passes if that median is within 15 cm. Frames with no homography count as failures.
- Two sets: 200 broadcast frames from the TennisCourtDetector validation split, and 359 frames from 10 amateur videos with one labeled homography per fixed-camera video (projected into frames that pass a painted-line check).
- Results are reported per frame and with the 12-frame running median the app applies over time.
- Code: `evaluation/court_detection/`.

## 3. Automatic point segmentation

The goal is to cut a recorded match down to the points: keep every rally whole and remove the walking, ball collection and changeovers in between.

### Features

At 10 steps per second (`STEP_SECONDS = 0.1`), `breakpoint_vision/points/features.py` computes 8 visual features per step:

- `ball_max`, `ball_seen`, `ball_speed`, `ball_x`, `ball_y`: from the ball model above, run exactly as on the phone (area resize, same weights).
- `motion_top`, `motion_mid`, `motion_bottom`: brightness-normalized frame differences in three horizontal bands of a 64x36 thumbnail (far court, net, near court). A far-side serve shows up in motion even when the ball is too small to track, and normalizing by brightness keeps night and day comparable.

Audio enters the network as raw 16 kHz PCM. The log-mel front end (64 mel bands, 25 ms window, 10 ms hop) is a fixed convolution inside the model, so the phone and the training code compute the same spectrogram.

### Model

`breakpoint_vision/points/model.py`, `PointSegmenter`:

- an audio encoder (1D convolutions down to one vector per step, 48 dims)
- a feature MLP (32 dims)
- a local convolution to 64 channels
- ten dilated residual temporal blocks (dilations 1, 2, 4, 8, 16, 32, 64, 1, 4, 16), a temporal convolutional network (TCN) that sees about 15 s on each side of a step

The output has three channels per step: in play, point start and point end. The start and end heads sharpen boundaries during training.

### Training

`breakpoint_vision/train/train_points.py`:

- Loss: per-step BCE on in-play plus a boundary BCE (weight 0.5).
- Sampling: random 60 s windows, 1500 steps, batch 16, learning rate 2e-3.
- Short segments: reviewed segments of 4 s or less (serve faults, unreturned serves), plus a few steps around them, get 3x weight (`--short-boost 3`), because per-step losses otherwise barely register them next to long rallies.
- Weak labels: videos with only automatic silver labels are weighted 0.15.

### Decoding

`breakpoint_vision/points/decode.py` turns probabilities into segments with hysteresis thresholds, minimum segment length and merging of short gaps. The thresholds are rescaled per video from that video's probability distribution, which adapts the operating point on under-confident videos. It can increase false segments; selection includes a play-free guard. Playback adds 1.0 s before and 2.0 s after each segment. The decode values are chosen by `scripts/choose_decode.py` from out-of-fold predictions; the chosen values are withheld.

### Metrics

From `breakpoint_vision/points/scoring.py`:

- **Points kept whole**: a labeled point counts if at least 95% of it (on a 0.05 s grid) falls inside the padded playback. Pass or fail per point, pooled over all points.
- **Dead time removed**: the share of non-point time that is not played, with padding counting against it and warm-ups counting as dead time. Reported both as the mean of per-video fractions (the original figure) and weighted by time.

Only seconds 300 to 900 of each 20-minute clip are labeled and scored.

### Cross-validation

`scripts/cross_validate.py`: 4 folds over 42 labeled videos, stratified by lighting (day, dusk, indoor, dim indoor, night) by dealing videos round-robin within each lighting group. Each fold trains on every other video and predicts its held-out videos, so every video is scored by a model that never saw it.

The additional held-fold decode analysis reuses these predictions. It does not retrain inner models excluding the outer fold, so its historical name, "nested decode selection," must not be read as fully nested cross-validation.

## 4. Splitting and leakage

See [dataset_and_splits.md](dataset_and_splits.md) for the full account. In short:

- The ball and court training code originally split individual samples at random. Ball samples are overlapping 5-frame windows and court samples are frames from a small number of source videos, so almost every validation sample had a near-duplicate in training.
- The fix (`breakpoint_vision/data/split.py`) assigns whole groups to validation by a hashed rank: clips for the ball model, source videos for the court model. The point segmenter is grouped by video.
- Some leakage remains and is listed in [limitations.md](limitations.md).
