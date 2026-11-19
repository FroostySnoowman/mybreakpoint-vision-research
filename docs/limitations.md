# Limitations

This is independent engineering research done while building a product, not supervised or peer-reviewed research. The evaluation sets are small, annotation uses a single review process, and the footage can't be shared. These are the weaknesses I know about.

## Evaluation design

- **Small test sets.** Ball: 248 visible frames from 7 videos. Court: 359 frames from 10 amateur videos. Points: 1,351 points from 42 videos. Frames within a video are strongly correlated, so the effective sample size is closer to the number of videos than the number of frames. Confidence intervals in [results.md](results.md) are bootstrapped over videos for this reason.
- **Settings tuned on the evaluation data.**
  - Ball: the tracker's defaults were set before the test labels existed. The pooling and smoothing variants were tuned on the same 350 frames they are scored on, so their gains are optimistic.
  - Court: the amateur set was used to choose checkpoints and the averaged model, so it is a validation set, not a test set.
  - Points: decode settings, playback padding, the short-segment weight, the resize method and the model-averaging test were all chosen on the same out-of-fold predictions the headline is computed from. A nested version (choose on 3 folds, score the 4th) is reported next to the original.
- **Decode selection is only partially isolated.** The held-fold re-selection uses existing out-of-fold predictions. Selecting-fold models were trained on the scored fold, and upstream design decisions are not reselected in an inner loop. This is not fully nested cross-validation.
- **No sealed test set** for the point segmenter: the 14 videos first held out as a test set were folded into cross-validation.
- **Leakage that remains.**
  - Videos from the same channel or event can sit in different point folds.
  - The ball split groups by clip, not by match.
  - The broadcast court set shares 434 of 442 source videos with its dataset's training split.
- **Few seeds.** The original point-segmenter runs all used one seed. Repeat runs with two more seeds are reported in [results.md](results.md); there are still only three.
- **No significance tests in the original work.** The paired test and bootstrap intervals in [results.md](results.md) were computed afterward from saved predictions.

## Labels

- **One review pipeline.** Benchmark labels use automatically generated drafts, model candidates, and author review. There is no second independent human annotator or inter-annotator agreement measurement. The demonstration match uses a separate frame-only labeling pass; independence from model overlays does not establish independent human annotation.
- **Anchoring.** The ball review sheets showed the evaluated model's own top candidates. 7 accepted positions are that model's candidates and 116 are the teacher's. This can bias the labels toward positions the models find, and favor them.
- **Coverage.** Point candidates came from the teacher's ball tracks, so points it missed entirely may be under-represented in the labels.
- **Ball positions are in 512x288 model pixels**, so 8 px corresponds to different real distances at different source resolutions and court distances.

## Metrics

- "Points kept whole" counts a point as kept if 95% of it is inside the padded playback. Generous padding raises it and lowers dead-time removal, so the two numbers have to be read together.
- Five videos are edited, near-continuous play. All their points are kept and almost none of their dead time is removed. One of them scores 100% kept whole although the model found only 3% of its play, because the edited video has almost no dead time to remove. Results without these videos are reported separately.
- Only seconds 300-900 of each clip are labeled, so warm-ups at the start of recordings are under-represented.
- The court metric uses OpenCV RANSAC to fit the predicted homography, which approximates but does not exactly reproduce the trimmed-subset fit used at run time.

## Reproducibility

- The training run, data and logs behind the evaluated ball weights (shipped in August 2026) and the first court model were not kept. Their architecture and training code are here, and their evaluation reproduces from saved outputs, but the training itself can't be rerun.
- Some results in the project notes have no saved output and are not reported as results here (see [claims.md](claims.md)): 60 fps robustness, the claim that training on compressed copies hurt everywhere, the distilled student's score on the labels, fp32 versus fp16 numbers and the lighting-simulation calibration measurements.
- Weights, labels, footage and tuned settings are withheld, so the published numbers can be checked against the published code and tables but not rerun from scratch (see [reproducibility.md](reproducibility.md)).

## Scope

- The ball headline excludes the court gate and must not be described as whole-app accuracy.
- No reproducible latency, memory, battery, sustained-throughput, or device-matrix benchmark is included. Native Core ML and ONNX integration exists; that does not establish performance on every phone.
- The original training manifests are incomplete. Download dates alone cannot establish that benchmark source videos were absent from all training data.
- Frame-level Wilson intervals and McNemar tests ignore within-video correlation. Failure to reject a difference does not establish equivalence.
- The format study uses a different model artifact from the headline ball result.
- Court-fit residuals and lock coverage are not independent measurements of line-call accuracy. Bounce detection, automatic score inference, and player statistics have no empirical benchmark in this report.


- No controlled camera-angle experiment. Angle variety comes from the footage itself.
- Night robustness for the court model was only tested with synthetic darkening.
- The format experiments use 6 videos and one run each. Only heavy compression gives a consistent effect.
