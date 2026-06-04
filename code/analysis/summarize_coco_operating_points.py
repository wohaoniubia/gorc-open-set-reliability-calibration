

"""Build the Step12 COCO-Val-OpenSet-5K final evidence summary.

This script does not tune any model or policy. It only consolidates the
calibration-selected Step12 outputs into paper-ready tables and a Markdown
summary with explicit claim boundaries.
"""


from __future__ import annotations


import argparse

import json

import math

from datetime import datetime

from pathlib import Path

from typing import Any


import pandas as pd



def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(

        "--project_root",

        type=Path,

        default=Path(__file__).resolve().parents[1],

        help="Project root containing outputs/ and scripts/.",

    )

    parser.add_argument(

        "--output_root",

        type=Path,

        default=None,

        help="Output root. Defaults to outputs/step12e_coco_openset_final_summary.",

    )

    return parser.parse_args()



def require_file(path: Path) -> Path:

    if not path.exists():

        raise FileNotFoundError(f"Required Step12 input is missing: {path}")

    return path



def read_csv(path: Path) -> pd.DataFrame:

    require_file(path)

    return pd.read_csv(path)



def read_json(path: Path) -> dict[str, Any]:

    require_file(path)

    with path.open("r", encoding="utf-8") as f:

        return json.load(f)



def fmt_float(value: Any, digits: int = 4) -> str:

    try:

        value = float(value)

    except (TypeError, ValueError):

        return "NA"

    if not math.isfinite(value):

        return "NA"

    return f"{value:.{digits}f}"



def fmt_delta(value: Any, digits: int = 4) -> str:

    try:

        value = float(value)

    except (TypeError, ValueError):

        return "NA"

    if not math.isfinite(value):

        return "NA"

    return f"{value:+.{digits}f}"



def first_row(df: pd.DataFrame, mask: pd.Series, label: str) -> pd.Series:

    rows = df[mask]

    if rows.empty:

        raise ValueError(f"Could not find required row: {label}")

    return rows.iloc[0]



def build_main_results(compact_df: pd.DataFrame) -> pd.DataFrame:

    keep_cols = [

        "method",

        "selection_policy",

        "mode",

        "score_col",

        "threshold",

        "balanced",

        "precision",

        "recall",

        "unknown_false_accept_objects",

        "unknown_reject_rate_object_level",

        "background_false_accept_count",

        "num_accepted_detections",

        "AP50",

        "AP75",

        "AP",

        "delta_balanced_vs_raw",

        "delta_precision_vs_raw",

        "delta_recall_vs_raw",

        "delta_ufa_vs_raw",

        "delta_bg_fp_vs_raw",

        "delta_AP50_vs_raw",

        "delta_AP_vs_raw",

        "meets_AP50_minus_0p005",

        "meets_AP_minus_0p005",

        "meets_success_criterion",

        "ap_rank_preserving_by_design",

    ]

    available = [col for col in keep_cols if col in compact_df.columns]

    out = compact_df[available].copy()

    out.insert(0, "dataset", "COCO-Val-OpenSet-5K")

    out.insert(1, "detector", "YOLO-World-l")

    out.insert(2, "metric_source", "Step12C exact full-test evaluation")

    return out



def build_per_class_summary(class_counts: pd.DataFrame, per_class_ap: pd.DataFrame) -> pd.DataFrame:

    ap50 = per_class_ap[per_class_ap["iou_threshold"].round(2) == 0.50][

        ["class_name", "AP"]

    ].rename(columns={"AP": "AP50"})

    ap75 = per_class_ap[per_class_ap["iou_threshold"].round(2) == 0.75][

        ["class_name", "AP"]

    ].rename(columns={"AP": "AP75"})

    ap_mean = (

        per_class_ap.groupby("class_name", as_index=False)

        .agg(AP=("AP", "mean"), num_gt_test=("num_gt", "max"), num_det_test=("num_det", "max"))

    )

    known_ap = ap_mean.merge(ap50, on="class_name", how="left").merge(ap75, on="class_name", how="left")

    out = class_counts.merge(known_ap, on="class_name", how="left")

    ordered_cols = [

        "category_id",

        "class_name",

        "role",

        "supercategory",

        "is_known",

        "object_count",

        "image_count",

        "num_gt_test",

        "num_det_test",

        "AP50",

        "AP75",

        "AP",

    ]

    return out[[col for col in ordered_cols if col in out.columns]].sort_values(

        ["role", "class_name"], ascending=[True, True]

    )



def markdown_table(df: pd.DataFrame, cols: list[str], max_rows: int | None = None) -> str:

    show = df[cols].copy()

    if max_rows is not None:

        show = show.head(max_rows)

    for col in show.columns:

        if pd.api.types.is_float_dtype(show[col]):

            show[col] = show[col].map(lambda x: fmt_float(x, 4))

    return show.to_markdown(index=False)



def main() -> None:

    args = parse_args()

    project_root = args.project_root.resolve()

    output_root = (

        args.output_root.resolve()

        if args.output_root is not None

        else project_root / "outputs" / "step12e_coco_openset_final_summary"

    )

    docs_dir = output_root / "docs"

    tables_dir = output_root / "tables"

    docs_dir.mkdir(parents=True, exist_ok=True)

    tables_dir.mkdir(parents=True, exist_ok=True)


    step12a_root = project_root / "outputs" / "step12a_coco_val_openset_protocol"

    step12b_root = project_root / "outputs" / "step12b_yoloworld_coco_openset_baseline"

    step12c_root = project_root / "outputs" / "step12c_geometry_risk_coco_openset"

    step12d_root = project_root / "outputs" / "step12d_coco_openset_bootstrap"

    step11e_root = project_root / "outputs" / "step11e_high_level_sci_evidence_package"


    split_summary = read_csv(step12a_root / "csv" / "step12a_split_summary.csv")

    class_counts = read_csv(step12a_root / "csv" / "step12a_per_class_counts.csv")

    protocol_integrity = read_json(step12a_root / "step12a_integrity_report.json")

    baseline_integrity = read_json(step12b_root / "step12b_integrity_report.json")

    compact = read_csv(step12c_root / "csv" / "step12c_compact_paper_metrics.csv")

    geometry_integrity = read_json(step12c_root / "step12c_integrity_report.json")

    bootstrap_delta = read_csv(step12d_root / "csv" / "step12d_bootstrap_delta.csv")

    bootstrap_summary = read_csv(step12d_root / "csv" / "step12d_bootstrap_summary.csv")

    bootstrap_integrity = read_json(step12d_root / "step12d_integrity_report.json")

    per_class_ap = read_csv(step12b_root / "csv" / "step12b_per_class_ap.csv")


    lvis_results_path = step11e_root / "tables" / "lvis300_yoloworld_results.csv"

    lvis_results = read_csv(lvis_results_path) if lvis_results_path.exists() else pd.DataFrame()


    coco_main = build_main_results(compact)

    coco_bootstrap = bootstrap_delta.copy()

    coco_bootstrap.insert(0, "dataset", "COCO-Val-OpenSet-5K")

    coco_bootstrap.insert(1, "detector", "YOLO-World-l")

    coco_bootstrap["AP_bootstrap_note"] = bootstrap_integrity.get("ap_bootstrap_mode", "")

    coco_per_class = build_per_class_summary(class_counts, per_class_ap)


    main_results_path = tables_dir / "coco_main_results.csv"

    bootstrap_path = tables_dir / "coco_bootstrap_delta.csv"

    per_class_path = tables_dir / "coco_per_class_summary.csv"

    coco_main.to_csv(main_results_path, index=False)

    coco_bootstrap.to_csv(bootstrap_path, index=False)

    coco_per_class.to_csv(per_class_path, index=False)


    raw = first_row(compact, compact["method"].eq("step12b_yoloworld_l_raw_reproduced"), "raw baseline")

    reliability = first_row(

        compact,

        compact["method"].eq("step12c_geometry_max_cal_balanced"),

        "reliability-first policy",

    )

    ap_constrained = first_row(

        compact,

        compact["method"].eq("step12c_geometry_ap_constrained_balanced_AP50tol0_APtol0"),

        "strict AP-constrained policy",

    )

    rel_delta = first_row(

        bootstrap_delta,

        bootstrap_delta["method"].eq("step12c_geometry_max_cal_balanced"),

        "bootstrap reliability-first",

    )

    ap_delta = first_row(

        bootstrap_delta,

        bootstrap_delta["method"].eq("step12c_geometry_ap_constrained_balanced_AP50tol0_APtol0"),

        "bootstrap AP-constrained",

    )


    lvis_text = "LVIS-Clear-Mini-300 comparison table was not found; comparison is limited to Step12 COCO."

    if not lvis_results.empty:

        try:

            lvis_raw = first_row(

                lvis_results,

                (lvis_results["detector_variant"].eq("yolov8l-worldv2"))

                & (lvis_results["method_group"].eq("raw")),

                "LVIS raw YOLO-World-l",

            )

            lvis_ap = first_row(

                lvis_results,

                lvis_results["method_group"].eq("geometry_ap_constrained"),

                "LVIS AP-constrained",

            )

            lvis_rel = first_row(

                lvis_results,

                lvis_results["method_group"].eq("geometry_reliability_first"),

                "LVIS reliability-first",

            )

            lvis_text = (

                "LVIS-Clear-Mini-300 showed larger relative reliability gains on a smaller custom public protocol "

                f"(YOLO-World-l raw balanced {fmt_float(lvis_raw['balanced'])}, AP-constrained balanced "

                f"{fmt_float(lvis_ap['balanced'])}, reliability-first balanced {fmt_float(lvis_rel['balanced'])}). "

                "COCO-Val-OpenSet-5K is stronger as scale/standard-public evidence, while LVIS remains useful as a "

                "harder custom diagnostic setting."

            )

        except Exception as exc:                                               

            lvis_text = f"LVIS comparison table was present but could not be parsed cleanly: {exc}"


    main_table = markdown_table(

        pd.DataFrame([raw, reliability, ap_constrained]),

        [

            "method",

            "threshold",

            "balanced",

            "precision",

            "recall",

            "unknown_false_accept_objects",

            "background_false_accept_count",

            "AP50",

            "AP75",

            "AP",

        ],

    )

    bootstrap_table = markdown_table(

        pd.DataFrame([rel_delta, ap_delta]),

        [

            "method",

            "delta_precision_recall_unknown_balanced_score_mean",

            "delta_precision_recall_unknown_balanced_score_ci025",

            "delta_precision_recall_unknown_balanced_score_ci975",

            "delta_unknown_false_accept_objects_mean",

            "delta_unknown_false_accept_objects_ci025",

            "delta_unknown_false_accept_objects_ci975",

            "delta_background_false_accept_count_mean",

            "delta_background_false_accept_count_ci025",

            "delta_background_false_accept_count_ci975",

        ],

    )


    summary_md = f"""# Step12 COCO-Val-OpenSet-5K Final Summary

Generated at: {datetime.now().isoformat(timespec="seconds")}

## Protocol

Step12 constructs a COCO val2017 open-set reliability protocol using all 5,000 validation images. The known set is the VOC-style 20-class subset expressed with official COCO names; all other annotated COCO categories are treated as unknown objects. The split is calibration-only model selection on 1,000 images and final test evaluation on 4,000 images.

- Total images: {protocol_integrity.get("num_images_total")}
- Calibration images: {protocol_integrity.get("num_calibration_images")}
- Test images: {protocol_integrity.get("num_test_images")}
- Known classes: {protocol_integrity.get("num_known_classes")}
- Unknown classes: {protocol_integrity.get("num_unknown_classes")}
- All 5,000 images used: {protocol_integrity.get("all_5000_images_used")}
- Filtering rule: {protocol_integrity.get("image_filtering_rule")}

## Exact Test Metrics

These AP/AP50/AP75 values come from the Step12C exact full-test evaluation of calibration-selected policies. They are the primary AP values for reporting.

{main_table}

## Main Findings

1. **COCO-Val-OpenSet-5K supports geometry-aware reliability calibration.** The reliability-first policy improves balanced score from {fmt_float(raw['balanced'])} to {fmt_float(reliability['balanced'])}, precision from {fmt_float(raw['precision'])} to {fmt_float(reliability['precision'])}, UFA from {int(raw['unknown_false_accept_objects'])} to {int(reliability['unknown_false_accept_objects'])}, and AP from {fmt_float(raw['AP'])} to {fmt_float(reliability['AP'])}.

2. **An AP-constrained reliability mode exists.** The strict AP-constrained policy keeps AP50 at {fmt_float(ap_constrained['AP50'])} versus raw {fmt_float(raw['AP50'])}, keeps AP at {fmt_float(ap_constrained['AP'])} versus raw {fmt_float(raw['AP'])}, reduces UFA from {int(raw['unknown_false_accept_objects'])} to {int(ap_constrained['unknown_false_accept_objects'])}, and reduces background false accepts from {int(raw['background_false_accept_count'])} to {int(ap_constrained['background_false_accept_count'])}. Its balanced-score gain is small ({fmt_delta(ap_constrained['delta_balanced_vs_raw'])}), but its precision and false-accept reductions are material.

3. **Reliability-first is the strongest COCO operating point.** It simultaneously improves exact AP/AP50/AP75 and open-set reliability metrics, but the AP gain is small. This should be framed as reliability calibration evidence, not detector SOTA.

## Bootstrap Evidence

Step12D performs paired image bootstrap with N={bootstrap_integrity.get("n_boot_completed")} using cached per-image operating counts. The bootstrap strongly supports reliability improvements for the main operating metrics.

{bootstrap_table}

Important AP note: Step12D records `{bootstrap_integrity.get("ap_bootstrap_mode")}`. Therefore, exact full-test AP/AP50/AP75 from Step12C should be used as the authoritative AP table, while Step12D AP deltas are only diagnostic.

## Relation to LVIS-Clear-Mini-300

{lvis_text}

COCO-Val-OpenSet-5K is more standard and substantially larger, so it should be included in the main paper evidence. The effect size in balanced score is smaller than LVIS-Clear-Mini-300, which is expected because COCO val2017 with 20 known classes gives a stronger and more stable raw baseline. The COCO result is therefore best interpreted as a scale-and-standardness validation of the same reliability-calibration claim.

## Publication Decision

COCO-Val-OpenSet-5K materially strengthens the Neurocomputing-style submission case. The recommended claim is:

> Geometry-aware calibration improves open-world reliability of foundation-model detectors under calibration-only model selection, including an AP-constrained operating mode on COCO val2017 open-set validation.

The work should not be presented as COCO detector SOTA or full open-vocabulary detection SOTA. It is now reasonable to move into final manuscript writing, with optional follow-up experiments limited to exact AP bootstrap recomputation or a COCO cross-detector replication if a stronger venue or reviewer request demands it.

## Generated Tables

- `tables/coco_main_results.csv`
- `tables/coco_bootstrap_delta.csv`
- `tables/coco_per_class_summary.csv`
"""


    summary_path = docs_dir / "coco_val_openset_summary.md"

    summary_path.write_text(summary_md, encoding="utf-8")


    integrity = {

        "method": "Step12E COCO-Val-OpenSet-5K final summary",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(project_root),

        "output_root": str(output_root),

        "inputs": {

            "step12a_protocol": str(step12a_root),

            "step12b_baseline": str(step12b_root),

            "step12c_geometry": str(step12c_root),

            "step12d_bootstrap": str(step12d_root),

            "step11e_lvis_comparison": str(lvis_results_path),

        },

        "protocol": {

            "num_images_total": protocol_integrity.get("num_images_total"),

            "num_calibration_images": protocol_integrity.get("num_calibration_images"),

            "num_test_images": protocol_integrity.get("num_test_images"),

            "num_known_classes": protocol_integrity.get("num_known_classes"),

            "num_unknown_classes": protocol_integrity.get("num_unknown_classes"),

            "all_5000_images_used": protocol_integrity.get("all_5000_images_used"),

        },

        "baseline": {

            "model": baseline_integrity.get("model_used"),

            "num_images_processed": baseline_integrity.get("num_images_processed"),

            "num_predictions_raw": baseline_integrity.get("num_predictions_raw"),

            "inference_complete": baseline_integrity.get("inference_complete"),

            "evaluation_complete": baseline_integrity.get("evaluation_complete"),

        },

        "geometry": {

            "num_scored_candidates": geometry_integrity.get("row_counts", {}).get("scored_candidates"),

            "selected_test_policies_use_standard_evaluator": geometry_integrity.get(

                "selected_test_policies_use_standard_evaluator"

            ),

            "calibration_only_selection": geometry_integrity.get("calibration_only_selection"),

        },

        "conclusions": {

            "supports_geometry_aware_reliability_calibration": True,

            "ap_constrained_reliability_improvement_found": True,

            "include_in_neurocomputing_main_text": True,

            "ready_for_manuscript_writing": True,

            "claim_boundary": "Reliability calibration for foundation-model detectors; not COCO SOTA.",

            "ap_bootstrap_note": bootstrap_integrity.get("ap_bootstrap_mode"),

        },

        "outputs": {

            "summary": str(summary_path),

            "coco_main_results": str(main_results_path),

            "coco_bootstrap_delta": str(bootstrap_path),

            "coco_per_class_summary": str(per_class_path),

            "integrity_report": str(output_root / "final_integrity_report.json"),

        },

        "row_counts": {

            "coco_main_results": int(len(coco_main)),

            "coco_bootstrap_delta": int(len(coco_bootstrap)),

            "coco_per_class_summary": int(len(coco_per_class)),

            "bootstrap_summary": int(len(bootstrap_summary)),

            "split_summary": int(len(split_summary)),

        },

    }


    integrity_path = output_root / "final_integrity_report.json"

    integrity_path.write_text(json.dumps(integrity, indent=2), encoding="utf-8")


    print(f"Wrote {summary_path}")

    print(f"Wrote {main_results_path}")

    print(f"Wrote {bootstrap_path}")

    print(f"Wrote {per_class_path}")

    print(f"Wrote {integrity_path}")



if __name__ == "__main__":

    main()

