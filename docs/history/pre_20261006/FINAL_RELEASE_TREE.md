# Final Release Tree

```text
gorc-open-set-reliability-calibration/
|-- README.md
|-- FINAL_RELEASE_TREE.md
|-- code/
|   |-- analysis/
|   |   |-- CONFIGURE_PATHS.md
|   |   |-- analyze_calibration_bins.py
|   |   |-- analyze_geometry_group_ablation.py
|   |   |-- analyze_iou75_diagnostics.py
|   |   |-- analyze_prompt_sensitivity.py
|   |   |-- analyze_selection_stability.py
|   |   |-- bootstrap_coco_operating_points.py
|   |   |-- bootstrap_exact_ap.py
|   |   |-- bootstrap_lvis_ap_constrained_reliability.py
|   |   |-- bootstrap_lvis_operating_points.py
|   |   |-- build_baseline_audit_tables.py
|   |   |-- build_coco_gorc_policy_grid.py
|   |   |-- build_lvis_gorc_policy_grid.py
|   |   |-- calibrate_groundingdino_coco_reliability.py
|   |   |-- calibrate_groundingdino_geometry_reliability.py
|   |   |-- calibrate_lvis_ap_constrained_reliability.py
|   |   |-- compare_lvis_external_baselines.py
|   |   |-- compute_reliability_diagnostics.py
|   |   |-- evaluate_groundingdino_lvis_baseline.py
|   |   |-- evaluate_strong_coco_baselines.py
|   |   |-- evaluate_yoloworld_coco_baseline.py
|   |   |-- evaluate_yoloworld_lvis_baseline.py
|   |   |-- extract_qualitative_error_cases.py
|   |   |-- prepare_coco_val_openset_protocol.py
|   |   |-- prepare_lvis_clear_mini_protocol.py
|   |   |-- run_groundingdino_feasibility_check.py
|   |   |-- run_known_label_and_budget_diagnostics.py
|   |   |-- summarize_coco_cross_detector_results.py
|   |   |-- summarize_coco_operating_points.py
|   |   `-- summarize_cross_detector_comparison.py
|   `-- audit/
|       |-- audit_ap_constrained_claims.py
|       |-- audit_metric_accounting.py
|       |-- audit_protocol_reproducibility.py
|       |-- build_detector_diagnostic_evidence_package.py
|       `-- build_lvis_evidence_package.py
|-- data/
|   |-- cache_metadata/
|   |   `-- omitted_resources.md
|   |-- figure_source/
|   |   |-- fig03_coco_operating_landscape.csv
|   |   |-- fig04_reliability_score_bins.csv
|   |   |-- fig05_preference_scores.csv
|   |   |-- fig06_geometry_group_ablation.csv
|   |   |-- fig07_bootstrap_ci.csv
|   |   |-- fig07_calibration_budget.csv
|   |   |-- fig07_repeated_split_false_accept_changes.csv
|   |   |-- fig08_prompt_stress.csv
|   |   `-- fig09_qualitative_cases_manifest.csv
|   |-- supplementary_source/
|   |   |-- README.md
|   |   |-- s_bootstrap_ci.csv
|   |   |-- s_calibration_budget.csv
|   |   |-- s_detector_config.csv
|   |   |-- s_exact_ap_bootstrap.csv
|   |   |-- s_gorc_operating_mode_definitions.csv
|   |   |-- s_known_label_only_diagnostic.csv
|   |   |-- s_policy_grid.csv
|   |   |-- s_preference_table.csv
|   |   |-- s_prompt_stress.csv
|   |   `-- s_repeated_split_exact_ap.csv
|   `-- table_source/
|       |-- table01_validation_protocols.csv
|       |-- table02_main_coco_operating_points.csv
|       |-- table03_coco_baseline_ablation_audit.csv
|       |-- table04_computational_footprint.csv
|       |-- table05_selected_policy_apc_card.csv
|       `-- table06_lvis_grounding_dino_diagnostics.csv
|-- figures/
|   |-- Fig1.jpg
|   |-- Fig2.jpg
|   |-- Fig3.jpg
|   |-- Fig4.jpg
|   |-- Fig5.jpg
|   |-- Fig6.jpg
|   |-- Fig7.jpg
|   |-- Fig8.jpg
|   `-- Fig9.jpg
`-- scripts/
    `-- validate_released_data.py
```
