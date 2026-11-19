#set document(title: "MyBreakPoint Vision: Research Summary", author: "Jacob Beal")
#set page(paper: "us-letter", margin: (x: 0.8in, y: 0.7in))
#set text(font: "New Computer Modern", size: 10pt, lang: "en", region: "us")
#set par(leading: 0.5em, spacing: 0.8em)
#set heading(numbering: none)
#show heading: it => block(above: 1em, below: 0.5em, text(size: 11pt, weight: "bold", it))
#set table(stroke: none, inset: 5pt)

#text(size: 18pt, weight: "bold")[Learning from Amateur Tennis Video]
#v(0.3em)
#text(size: 12pt)[Jacob Beal]
#linebreak()
Independent research and development · Break Point Technologies, LLC · November 2026
#linebreak()
#text(size: 9pt)[Research summary accompanying a sole-author technical report; not peer reviewed.]
#v(0.5em)
#line(length: 100%, stroke: 0.6pt)

= Research question
How can a single-camera system track a tiny, fast-moving ball, calibrate an off-center court view, and extract complete points from amateur tennis recordings? MyBreakPoint provides a concrete setting for studying generalization across lighting, camera placement, and compression while keeping inference on the device.

= Contribution
I designed and developed a system combining a five-frame heatmap detector and candidate-level tracker, a court-keypoint network with robust homography fitting, and a temporal point segmenter using ball detections, banded frame motion, and audio. The contribution is the integration and empirical study of these components under real product constraints, building on established model families.

I investigated overlapping-window validation leakage, replaced sample-level splits with clip or source-video groups, and examined how deployment preprocessing and evaluation timestamps affect measured performance. The resulting report preserves negative results and separates model-selection effects from headline scores.

= Evidence
#table(
  columns: (1.1fr, 2.5fr),
  table.hline(stroke: 0.5pt),
  [*Task*], [*Observed result and scope*],
  table.hline(stroke: 0.3pt),
  [Ball tracking], [218/248 visible-ball frames within 8 pixels at 512×288 resolution (87.9%); 20/88 false alarms. Seven videos, without court gating.],
  [Court calibration], [25.1% to 61.9% of 359 amateur frames within 15 cm after domain adaptation and aggregation. Ten videos; the adapted model was selected on this validation set.],
  [Point segmentation], [91.9% segment retention and 52.4% mean dead-time removal under held-fold decode re-selection. Four folds, 42 videos, 1,351 serve segments; retention 91.9% to 92.4% across three seeds.],
  table.hline(stroke: 0.5pt),
)

Retention means at least 95% of a labeled serve segment lies inside padded playback. The decode analysis reuses existing out-of-fold predictions and is not fully nested model cross-validation. A separate 20-minute match retained 42/42 serve segments while removing 54.6% of dead time; this is a demonstration, not a population estimate.

= Research lessons and next questions
Broadcast performance did not predict performance on amateur court views. Heavy compression reduced ball localization by 20.7 percentage points on six paired videos. Timestamp-rounding bugs altered apparent format effects. These findings motivate match-level held-out evaluation, independent labels, controlled modality ablations, and measured device throughput and energy use.

The broader research direction is reliable learning under distribution shift and imperfect supervision. This work provides experience in multimodal modeling, geometric constraints, and evaluation design. Application to healthcare or other high-stakes domains is a future research interest, not an outcome demonstrated here.

= Authorship and availability
I am the sole author and developer. Methods and data sources are documented in the full report. Public code, tests, methods, and aggregate tables are provided; footage, labels, weights, and tuned operating settings are withheld, so full experimental reproduction is unavailable.

#text(size: 9pt)[Full report and code: #link("https://github.com/FroostySnoowman/mybreakpoint-vision-research")[github.com/FroostySnoowman/mybreakpoint-vision-research]. See the full report for metrics, provenance, and limitations.]
