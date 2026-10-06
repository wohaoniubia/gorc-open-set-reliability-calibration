# Final actual-prompt fixed-subset stress test

Status: completed and independently verified on 2026-10-01 (Asia/Singapore).

## Design and source

- Grounding DINO tiny (`IDEA-Research/grounding-dino-tiny`) uses three actual detector-output caches: canonical sentence periods, comma separators, and `a photo of` phrases. Actual template strings and inference sources are bound to the original `prompt_sensitivity_integrity_report.json`; they are copied in `predeclared_prompt_manifest.json` and `prompt_stress_summary.csv`.
- The historical fixed subset contains 100 calibration and 300 test images. All 400 image-ledger entries are verified, including empty outputs. Only these 400 source annotation files contribute GT; no 5000-image denominator is used. Test GT is 1054 known and 1114 unknown objects in every variant.
- Original raw predictions are reprocessed with the original step8j class NMS: IoU 0.5 or containment 0.9. Geometry is recomputed from post-NMS boxes and actual image dimensions. No historical risk-model scores are reused.
- SCG is refitted per template with the locked `class_interactions`, C=0.1 recipe, calibration-only scaler/encoder/interactions, balanced logistic regression, unknown/BG weights 2/1.25, and full class prior. Raw/RF/AP-C selections use the same 16 streams and same calibration-only threshold rule; AP-C uses zero calibration AP/AP50 tolerance.
- Observed calibration classes lead to actual SCG dimensions 219, 131, and 76; these reflect the template candidate pools. The family/C and feature construction are identical.
- Metrics use corrected matching on the complete 300-image test population. AP is custom project 101-point per-class AP over all post-NMS candidates before final acceptance threshold, not official COCOeval.

## Test results

| Actual prompt template | Raw/post-NMS candidates | Raw B | RF B | RF minus Raw B | Raw AP | RF AP | Raw to RF UFA | Raw to RF BG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| canonical_periods | 3914/2956 | 0.747734139 | 0.748325316 | +0.000591177 | 0.471158979 | 0.476763018 | 0 to 2 | 262 to 331 |
| comma_separated | 610/553 | 0.015534415 | 0.000000000 | -0.015534415 | 0.009033673 | 0.005465019 | 119 to 2 | 67 to 3 |
| phrase_photo | 99/96 | 0.116718413 | 0.027777778 | -0.088940635 | 0.079994923 | 0.076889702 | 8 to 0 | 26 to 6 |

All three AP-C selections coincide with RF here. They are aliases of the same policy, not three independent confirmations. Calibration AP eligibility does not guarantee held-out AP preservation; comma and phrase templates lose test AP and B despite zero-tolerance calibration gates.

## Interpretation boundary

This is supplementary 100cal/300test Grounding DINO prompt/candidate sensitivity evidence, not full-protocol or YOLO-World prompt robustness. It measures the original inference and text-to-known-class mapping pipeline end to end. The commas and phrase variants severely reduce useful candidates; calibration cannot recover detections absent from those streams. RF B becomes 0 for comma separators (held-out TP=0 at the selected policy), and the phrase variant keeps only 10 held-out known TPs (recall approximately 0.00949). The canonical RF B gain is small and accompanied by increased UFA and BG. Do not describe prompt robustness or uniform risk reduction.

## Evidence and reproducibility

- `prompt_stress_summary.csv`: plotting-ready Raw/RF/AP-C test results and RF minus Raw deltas, actual strings, candidate counts, dimensions, and population denominators.
- `all_prompt_metrics.csv`: calibration and test metrics for all variants/methods.
- `population_split.csv` and `ground_truth.csv`: the exact fixed population and its GT.
- Variant directories contain post-NMS candidates/geometry, all400 ledgers and candidate counts, new model/config/training labels, scores/alignment, prior, calibration policy grid/capacity, selected policies, per-class AP and integrity hashes.
- `FINAL_PROMPT_STRESS_VALIDATION.json`: runner completion and source binding.
- `INDEPENDENT_PROMPT_STRESS_AUDIT.json`: reconstruction from original raw candidates through NMS/geometry, model score round trips, calibration-only labels/prior, full300 test metrics/AP and summary deltas. Subset GT IDs/labels/boxes were independently matched exactly to the original COCO `instances_val2017.json`.

Run with `[original local path omitted] and `code/run_final_figure_diagnostics.py --only prompt_stress`. No original raw caches, geometry-score caches, ledgers, or source annotations were modified.
