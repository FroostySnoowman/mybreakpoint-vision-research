# Result tables

The CSV files are the numerical source for the publication summary. They contain rounded published aggregates, not the private labels or predictions needed to rerun inference.

| File | Unit and interpretation |
|---|---|
| `tables/ball_per_video.csv` | Seven videos; integer frame counts. Visible, no-ball, and uncertain counts partition the 350 sampled frames. Within-3-pixel hits are a subset of within-8-pixel hits. The latter, no-output, and wrong-output counts partition visible frames. Court gating is absent. |
| `tables/ball_formats.csv` | Per-video, per-resize, per-encoding results for the separate format-study artifact. Each original-video row reuses its label sample. Re-encodings are not independent samples. |
| `tables/ball_formats_pooled.csv` | Counts pooled over the six videos with all six encodings. The seventh original-only video is excluded. |
| `tables/model_artifacts.csv` | SHA-256 and byte size of the two locally retained ONNX artifacts. Hashes establish file identity, not provenance or equivalence. Weights are withheld. |
| `tables/court.csv` | Fraction of frames passing a 15 cm median plane-error threshold. Amateur-trained rows are checkpoint-selection validation results. Blank error cells mean unreported, not zero. |
| `tables/point_folds.csv` | Anonymous video, lighting group, and fold assignment. Videos are grouped, but related matches/events are not independently grouped. |
| `tables/points_per_video.csv` | Original seed-0 selection on all out-of-fold predictions. `kept_whole` and `dead_time_removed` are fractions, rounded to four decimals. Empty retention means no labeled segments. |
| `tables/points_seeds.csv` | Three training seeds. `nested` columns mean held-fold decode re-selection on existing predictions, not fully nested retraining. Dead-time means weight videos equally unless explicitly time-weighted. |
| `tables/match_demo.csv` | One author-recorded demonstration match. Kept-whole and court-lock values are fractions. Lock is not measured court-position accuracy. |

Anonymous identifiers are consistent across tables. Their private mapping is withheld. The reported point unit includes faults and lets, not only completed tennis points. A segment is kept whole if at least 95% is inside padded playback, measured on a 0.05 s grid.

Run `python scripts/check_publication.py` to verify public count relationships, rounded headline values, fold assignments, and local Markdown links. This checks artifact consistency; it does not independently validate the underlying labels, experiments, or author declarations.
