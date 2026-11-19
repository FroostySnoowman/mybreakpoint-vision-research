# Data and splits

## Footage

All footage used for evaluation, and most of the footage used for training, comes from public YouTube videos of real matches: club, college, high-school and junior events, plus some professional and broadcast footage. It covers day, dusk, indoor, dim indoor and floodlit night play, filmed mostly from a fixed camera behind or beside the court, which is how people film with a phone.

Clips were downloaded as 20-minute segments at 480p or lower, the resolution range the phone analyzes. Higher-resolution copies of some test videos were downloaded separately for the format experiments.

None of this downloaded benchmark footage is published here. The demonstration images and clips are separately described as the author's own recording; the author reports participant permission. I don't hold redistribution rights, many players are identifiable, and some are minors. Video IDs are replaced with stable anonymous labels (`V01`, `V02`, …) in every published table. No footage or data from MyBreakPoint users was used anywhere.

Two public datasets were also used:

- **TennisCourtDetector** (github.com/yastrebksv/TennisCourtDetector): broadcast frames with 14 court keypoints. Used to train the first court model and as the broadcast evaluation set.
- **TrackNet tennis data** (Huang et al., 2019): broadcast frames with ball labels. The project notes record it as the training data for the first ball models. The training run behind the evaluated ball weights could not be recovered (see [limitations.md](limitations.md)).

The WASB-SBDT tennis model (Tarashima et al., 2023; MIT license) was used as a teacher: to propose label candidates, as a comparison, and for an unsuccessful distillation experiment. Its code is not included here.

## Labels

Benchmark annotation used automatically generated drafts, model candidate positions, visual review sheets, and final review by Jacob Beal. These are author-reviewed labels; no independent human annotation-agreement study was performed. The separately recorded demonstration match was labeled from frames without model-output overlays.

| Set | Size | How candidates were made | Review |
|---|---|---|---|
| Ball positions | 350 frames, 7 videos | Sampled uniformly over in-play time (fixed seed). Review sheets showed the frame, a motion trail, zoomed crops and the top 4 candidates from the teacher and from the evaluated model | Each frame marked visible / no ball / not in play / uncertain, with a position for visible balls. 125 of the 248 visible positions were placed without a matching candidate; 123 accept a candidate within about 1 px (116 from the teacher, 7 from the evaluated model) |
| Court homographies | 44 files (29 fixed-camera videos, 10 split into 32 camera segments, 5 skipped) | Gridded median frames and zoomed crops | Keypoints placed to 0.5 px, checked with a fitted-court overlay |
| Point segments | 42 videos (14 + 28), seconds 300-900 of each clip | Contact sheets of candidate segments proposed from the teacher's ball tracks | Each candidate accepted or rejected with a written reason; boundaries at 0.5 s |

A point runs from the serve to the dead ball. Each fault is its own segment. Warm-ups count as dead time.

The ball review sheets showed the evaluated model's own candidates, which can bias a reviewer toward accepting them. This is discussed in [limitations.md](limitations.md).

## Splits

### The leakage problem

The first ball and court training code split individual samples at random into training and validation (`torch.utils.data.random_split`). That was wrong for both datasets:

- **Ball**: training samples are 5-frame windows that slide one frame at a time, so neighboring windows share 4 of their 5 frames. With a random split, 99.1% of validation windows had a training window one frame away, and no clip was validation-only.
- **Court**: each court contributes about 18 frames from one fixed camera. With a random split, 99.8% of validation frames came from a source video that was also in training.

The validation accuracies reported at the time (detection accuracy 0.9067 for the ball model, keypoint accuracy 0.9877 for the court model) were therefore mostly memorization.

These overlap figures and the old accuracies are recorded in the project notes and in the commit that fixed the split (2026-08-03). The raw logs behind them were not kept, so I report them as documented, not reproduced.

### The fix

`breakpoint_vision/data/split.py` holds out whole groups. Each group gets a rank from `sha256(f"{seed}:{group}")`, and the lowest-ranked groups, a fixed fraction of all groups, go to validation. Adding samples within an existing group preserves its assignment. Adding new groups can change the rank cutoff and move existing groups across the split; freeze the group manifest for comparisons. `tests/test_split.py` checks that no group appears on both sides.

| Model | Grouping unit |
|---|---|
| Ball | clip (`<game>_<clip>`) |
| Court | source video |
| Point segmenter | video (cross-validation folds) |

After the fix, held-out detection accuracy for the ball model was 0.812. It is not directly comparable to the old 0.9067, because the tolerance also changed from 6/8 px to 4 px at the same time.

### Point segmenter folds

`scripts/cross_validate.py` sorts the 42 labeled videos by lighting group (from a hand-tagged manifest) and deals them round-robin into 4 folds, so each fold has a similar mix of day, dusk, indoor, dim indoor and night footage. Fold sizes are 11, 11, 10 and 10 videos (386, 278, 340 and 347 points). The anonymized fold assignment is in `results/tables/point_folds.csv`.

### What the split does not cover

- Several videos come from the same channel or event (for example different courts at one tournament) and can land in different folds. There is no channel or event metadata to group by, so this was not controlled.
- The ball split groups by clip, not by match.
- The 14 videos first set aside as a test set were later folded into cross-validation, so there is no sealed test set for the point segmenter.
- The broadcast court evaluation uses the TennisCourtDetector validation split. I measured that 434 of its 442 validation videos also appear in its training split, so that set is not held out by video.
