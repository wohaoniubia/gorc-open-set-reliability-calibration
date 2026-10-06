# Manuscript result index

Numbers refer to the 2026-10-06 manuscript and [Online Resource 1](Online_Resource_1.pdf). Main Tables 1–7 replace the initial release's Tables 1–6. Exact captions and display values are in [main CSV tables](../source_data/manuscript_tables/) and [supplementary CSV tables](../source_data/supplementary_tables/).

## Figures

Final images come directly from the manuscript; see the [gallery](../figures/README.md). The numerical inputs below are relative to `source_data/figure_data`.

| Figure | Inputs |
|---|---|
| 1 Workflow | `fig01_workflow_spec.json`; manuscript artwork |
| 2 Taxonomy/geometry | `fig02_taxonomy_contract.json`; manuscript artwork |
| 3 Operating points | `fig03_coco_operating_landscape.csv` |
| 4 Coverage/bins | `fig04_policy_grid_risk_coverage.csv`, `fig04_reliability_score_bins.csv`, Fig. 3 source |
| 5 Preferences | `fig05_preference_scores.csv`, `fig05_preference_winners.csv` |
| 6 Geometry groups | `fig06_geometry_group_ablation.csv` |
| 7 Uncertainty/calibration | `fig07_bootstrap_ci.csv`, `fig07_repeated_split_exact_ap.csv`, `fig07_calibration_budget.csv`, `fig07_repeated_split_false_accept_changes.csv` |
| 8 Prompts | `fig08_prompt_stress.csv` |
| 9 Examples | `fig09_qualitative_cases_manifest.csv`, `fig09_object_level_verification.json`, `fig09_crop_manifest.json` |

`scripts/extract_manuscript_assets.py` extracts figures and display tables from supplied DOCX files. It applies only crop metadata already stored in Word and does not redraw plots or rescale values. Run it with `--out` pointing to a separate output directory; the manuscript itself is not distributed here.

## Main tables

| Table | Subject | Numerical/configuration source |
|---|---|---|
| 1 | Variants | `config/final_method_lock.json`; display CSV |
| 2 | Protocol populations | `config/datasets.json`, `data/<stream>/scene_split.csv` |
| 3 | Primary COCO policies | `artifacts/coco_yolo/main_metrics.csv` |
| 4 | Baselines/features | `source_data/external_baselines`, `source_data/final_results/baselines_matched_score_C01`, main metrics |
| 5 | Runtime | `source_data/final_results/E8`; `code/benchmark_cli.py` |
| 6 | Policies | `artifacts/<stream>/selected_policies.csv` |
| 7 | LVIS/Grounding DINO | `artifacts/lvis_yolo/main_metrics.csv`, `artifacts/coco_gdino/main_metrics.csv` |

## Supplementary tables

| Tables | Subject | Evidence location |
|---|---|---|
| S1–S2 | Grids/ranker selection | `config`, `source_data/calibration_family_C_selection`, `source_data/final_results/E0` |
| S3–S4 | Inference/vocabularies | `config/detector_vocabulary_lock.json`, `config/datasets.json`, `data/<stream>` |
| S5–S9 | Uncertainty/geometry | `source_data/final_results/E1`, `source_data/figure_data` |
| S10 | Per-class comparison | `source_data/final_results/reassessment_20261001` |
| S11–S12 | Annotation/mechanism | `source_data/final_results/E2` |
| S13 | Five vocabularies | `source_data/final_results/E3`, corresponding `artifacts` streams |
| S14 | Calibration budgets | `source_data/final_results/calibration_budget` |
| S15 | Nested selection | `source_data/final_results/E4`, `source_data/final_results/reassessment_20261001` |
| S16–S17 | Preferences | `source_data/final_results/E5`, Fig. 5 numerical data |
| S18–S19 | Matched/official AP | `source_data/final_results/E6`, `source_data/final_results/official_COCO` |
| S20–S21 | Matching/LVIS/inspection | `source_data/final_results/E7`, `source_data/final_results/reassessment_20261001`, `source_data/manual_review_20261006` |
| Crowd diagnostic | Crowd boxes/names | `source_data/round5_20261006`; `code/round5_crowd_review.py` |
| S22 | Prompt formats | `source_data/final_results/prompt_stress` |
| S23 | Processing time | `source_data/final_results/E8`; `code/benchmark_cli.py` |

The reproduction guide distinguishes executable commands from archived records. E0–E8 are file-navigation labels, not paper table numbers.
