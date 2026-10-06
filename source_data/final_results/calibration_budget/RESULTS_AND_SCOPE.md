# Final calibration-budget diagnostic

Status: completed and independently verified on 2026-10-01 (Asia/Singapore).

## Design and source

- Original primary COCO YOLO-World candidate cache from `load_dataset("coco_yolo")`; the fresh E3 `original20` cache is not used.
- Original 1000 calibration and 4000 test images, including empty-output images. The 4000-image test population stays fixed across all runs.
- Budgets: 25, 50, 100, 250, 500, 1000 images. Seeds 711, 712, and 713 separately permute the original calibration IDs; prefixes form nested subsets. Every ID and permutation was saved in `predeclared_subset_manifest.json` before any fit.
- `class_interactions`, SCG, C=0.1 are already fixed by `audit/final_method_lock.json`. Each run refits scaler, predicted-class one-hot, standardized numeric-by-class interactions, balanced logistic regression with unknown/BG sample weights 2/1.25, and the full class prior on that budget only.
- The same 16-stream policy family and threshold construction are used. Raw, RF, and AP-C policies are selected on the same budget subset. AP-C uses zero calibration AP/AP50 tolerance.
- All outputs use corrected matching and complete-population denominators. AP is custom project 101-point AP before the final acceptance threshold, not official COCOeval.

## RF test results

| Calibration images | Runs | RF B mean | Sample SD | RF AP mean |
|---:|---:|---:|---:|---:|
| 25 | 3 | 0.762054188 | 0.003751932 | 0.473777493 |
| 50 | 3 | 0.762318411 | 0.016628039 | 0.479314309 |
| 100 | 3 | 0.773404307 | 0.001497081 | 0.475598491 |
| 250 | 3 | 0.776461472 | 0.001734624 | 0.478472518 |
| 500 | 3 | 0.777333083 | 0.000535893 | 0.480614615 |
| 1000 | 1 | 0.778918359 | 0.000000000 | 0.485755217 |

The points below 1000 are means of three calibration subsets; error bars are sample SD (ddof=1), not confidence intervals. The 1000-image point is one shared full-calibration run. Its SD is recorded as 0 solely because no replicate spread was estimated; it is not evidence of zero uncertainty.

The 1000-image run matches final E0 Raw/RF/AP-C for both calibration and test: selected score stream, threshold, B, precision, recall, UFA, BG, AP/AP50 and denominators. See `FULL_BUDGET_E0_EQUIVALENCE.json`.

## Interpretation boundary

This measures reduced fit-and-policy-selection budgets conditional on a method family and C already chosen using the full calibration program. It does not demonstrate that 25 images suffice to select family/C or repeat the earlier E9 hyperparameter search. Three subset seeds give sensitivity evidence, not independent external datasets. Do not claim monotonic improvement for each individual seed, statistical significance from these SD bars, or low-budget equivalence to the full experiment.

## Evidence and reproducibility

- `all_run_metrics.csv`: all 16 runs, each with Raw/RF/AP-C and calibration/test metrics.
- `budget_summary.csv`: all mean/SD summaries; `RF_test_budget_summary.csv`: plotting-ready RF rows.
- Per-run directories contain the locked configuration, fit IDs, fitted model/config/training labels, full calibration policy grid/capacity, selected policies saved before test evaluation, class prior, new scores and score alignment, per-class AP, metrics, and integrity hashes.
- `FINAL_CALIBRATION_BUDGET_VALIDATION.json`: runner completion and source binding.
- `INDEPENDENT_CALIBRATION_BUDGET_AUDIT.json`: saved-model score round trips, calibration-only labels/prior, policy reselection from saved grid, recomputed full 4000-image test metrics/AP, nested subset and mean/SD checks.

Run with `[original local path omitted] and `code/run_final_figure_diagnostics.py --only calibration_budget`. No original caches or source annotations were modified.
