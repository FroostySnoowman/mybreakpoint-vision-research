# System scope and evidence

The report focuses on three evaluated tasks. The larger MyBreakPoint project contains additional production integrations and research prototypes; their existence is not evidence of measured accuracy.

| Area | Implementation inspected | What the report establishes |
|---|---|---|
| Ball localization | Multi-frame heatmaps, local-maximum refinement, constant-velocity candidate tracking | 218/248 visible-ball frames within 8 model pixels, without court gating; 20/88 false alarms |
| Court calibration | Keypoint heatmaps, normalized DLT, robust fitting, temporal aggregation, presence gate | Domain-shift study and validation results on 359 amateur frames; no line-call accuracy claim |
| Point segmentation | Ball/motion features, audio front end, temporal network, playback decoder | Four-fold video CV and decode-selection sensitivity over 42 videos, three training seeds |
| Native mobile inference | Product source includes Core ML model loading on iOS and ONNX Runtime model loading on Android | Integration exists; latency, energy, sustained FPS, and cross-runtime numerical equivalence are not benchmarked here |
| Live overlay | Product source combines court presence, homography, ball tracking, and an overlay | Demonstration media; headline ball accuracy is not a gated whole-app score |
| Bounce detection | Product Python module detects image-space vertical reversals and trajectory prominence | Prototype heuristic, without an event-level labeled benchmark in this report |
| In/out classification | Product geometry maps proposed bounces to the court and applies a configurable uncertainty band | Assistive prototype, without empirical calibration of confidence or officiating accuracy |
| Scoring and statistics | Product includes scoring state machines and trajectory-derived summaries; the Python analysis pipeline returns an empty score object | No claim of validated automatic score reconstruction from video |
| Player tracking | Product offers an optional pretrained YOLO backend | External detector integration, outside the evaluated research scope |
| Service and API | Product includes a FastAPI analysis service and a versioned job/result contract | Deployment infrastructure, not a new learning algorithm or a result in this paper |

The public repository is an extracted research artifact, not the entire product. Its parameter constructors intentionally require private tuned values for original operating points. Synthetic tests use illustrative values. The private work folder retains raw audit records, predictions, intermediate figures, and the video-ID mapping; those are not part of the public release.

## Study boundaries

The ball format experiment uses a different ONNX artifact from the headline detector-plus-tracker evaluation. A matching aggregate count does not establish artifact equivalence. Ball localization errors are measured in 512x288 image coordinates. Court errors are measured on a plane. Neither is a measurement of 3D ball position.

The point segmenter uses future temporal context and a per-video probability rescale, appropriate for recorded-match processing. It should not be described as a strictly causal live point detector.

The technical report is an empirical systems study. Novelty is in the application, integration, evaluation findings, and engineering decisions documented here; no priority claim or exhaustive comparison with all contemporary methods is made.
