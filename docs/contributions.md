# Author contributions

**Jacob Beal is the sole author and developer of this project.** This is independent research and development conducted while building MyBreakPoint at Break Point Technologies, LLC.

## Author contributions

Beal conceived and directed the project, including research, implementation, annotation review, training, evaluation, and product integration. The released code and retained experiment records document the following work:

| Contribution | Evidence in this repository |
|---|---|
| Model implementation and training | `breakpoint_vision/models/`, `breakpoint_vision/train/`, `breakpoint_vision/points/model.py` |
| Temporal tracking and geometric calibration | `breakpoint_vision/ball_track.py`, `geometry.py`, `court_gate.py` |
| Data preparation and split design | `breakpoint_vision/data/`, `scripts/cross_validate.py` |
| Experiment design, scoring, and error analysis | `evaluation/`, `results/tables/`, `docs/claims.md` |
| System integration and research communication | The technical report, public research code, and native product integrations described in `system_scope.md` |

## Methods, data, and tools

- TrackNet and TrackNetV2 informed multi-frame heatmap tracking; temporal convolutional networks informed point segmentation. The implementations and system adaptation are part of this project. References appear in the report.
- The TrackNet tennis data and TennisCourtDetector data are external datasets. The evaluated ball model's original training manifest and logs were not retained, limiting reconstruction of its exact training provenance.
- The pretrained WASB-SBDT model supplied candidate labels, a comparison baseline, and a distillation teacher. Its published code license does not by itself establish redistribution rights for all data or weights.
- PyTorch, NumPy, OpenCV, PyAV, and other dependencies are acknowledged through dependency files and method descriptions.
