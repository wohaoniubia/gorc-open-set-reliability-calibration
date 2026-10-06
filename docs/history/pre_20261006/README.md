# GORC Open-Set Reliability Calibration

This repository contains source data, final figure files, and analysis scripts for Geometry-aware Open-Set Reliability Calibration (GORC). It is organized for result checking and reuse. Image datasets, detector weights, and large detector-output caches are not included.

## Folder Layout

- `data/figure_source/`: source files for data-based figures.
- `data/table_source/`: source files for main Tables 1-6.
- `data/supplementary_source/`: source files for additional tables and diagnostics.
- `data/cache_metadata/`: notes on omitted datasets, model weights, and large candidate caches.
- `figures/`: final figure files, provided as `Fig1.jpg` through `Fig9.jpg`.
- `code/analysis/`: scripts for protocol preparation, detector evaluation, GORC calibration, bootstrap diagnostics, and table construction.
- `code/audit/`: secondary consistency checks and evidence-building scripts.
- `scripts/`: lightweight repository checks.

## Figure Source Mapping

| Figure | Source data |
|---|---|
| Fig. 1 | final diagram file in `figures/Fig1.jpg` |
| Fig. 2 | final diagram file in `figures/Fig2.jpg` |
| Fig. 3 | `data/figure_source/fig03_coco_operating_landscape.csv` |
| Fig. 4 | `data/supplementary_source/s_policy_grid.csv`; `data/figure_source/fig04_reliability_score_bins.csv` |
| Fig. 5 | `data/figure_source/fig05_preference_scores.csv` |
| Fig. 6 | `data/figure_source/fig06_geometry_group_ablation.csv` |
| Fig. 7 | `data/figure_source/fig07_bootstrap_ci.csv`; `data/supplementary_source/s_repeated_split_exact_ap.csv`; `data/figure_source/fig07_calibration_budget.csv`; `data/figure_source/fig07_repeated_split_false_accept_changes.csv` |
| Fig. 8 | `data/figure_source/fig08_prompt_stress.csv` |
| Fig. 9 | `data/figure_source/fig09_qualitative_cases_manifest.csv` |

## Table Source Mapping

| Table | Source data |
|---|---|
| Table 1 | `data/table_source/table01_validation_protocols.csv` |
| Table 2 | `data/table_source/table02_main_coco_operating_points.csv` |
| Table 3 | `data/table_source/table03_coco_baseline_ablation_audit.csv` |
| Table 4 | `data/table_source/table04_computational_footprint.csv` |
| Table 5 | `data/table_source/table05_selected_policy_apc_card.csv` |
| Table 6 | `data/table_source/table06_lvis_grounding_dino_diagnostics.csv` |

## Additional Source Data

- `s_policy_grid.csv`: policy-grid source for threshold and operating-mode selection.
- `s_exact_ap_bootstrap.csv`: fixed-policy exact AP bootstrap source.
- `s_repeated_split_exact_ap.csv`: repeated calibration/test split diagnostics. Split 1-5 correspond to seeds 101, 202, 303, 404, and 505.
- `s_calibration_budget.csv`: calibration-budget sensitivity source.
- `s_known_label_only_diagnostic.csv`: known-label-only threshold diagnostic source.
- `s_preference_table.csv`: preference-sensitive accepted-output risk source.
- `s_prompt_stress.csv`: Grounding DINO prompt-template stress source.
- `s_detector_config.csv`: detector configuration and reproducibility-boundary fields.

## Data Check

Install Python with `pandas`, then run:

```bash
python scripts/validate_released_data.py
```

The script checks required source files, verifies selected cross-table values, confirms the final figure set, and scans the release tree for local-path markers and omitted working-package folders.

## Reproducing Analyses

The scripts in `code/analysis/` are intended to be run from a local clone after dataset and model resources have been configured. See `code/analysis/CONFIGURE_PATHS.md` for the expected local resources. Large image datasets, detector checkpoints, and scored-candidate caches must be obtained or regenerated separately.

## License

No reuse license is included at this stage. External datasets and model checkpoints remain under the terms of their original providers.
