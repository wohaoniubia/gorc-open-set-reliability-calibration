


"""
Step9d: Same-protocol external baseline comparison for LVIS-Clear-Mini-300.

The external baseline used here is a YOLO-World model-size variant comparison
under the exact Step9 protocol and evaluation code.
"""


from __future__ import annotations


import argparse

import json

from pathlib import Path

from typing import List


import pandas as pd



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step9d_external_baselines_lvis300"

DEFAULT_BASELINE_DIRS = [

    DEFAULT_PROJECT_ROOT / "outputs" / "step9d_external_baselines_lvis300" / "yolov8s_worldv2",

    DEFAULT_PROJECT_ROOT / "outputs" / "step9b_yoloworld_lvis300_raw_baseline",

]

DEFAULT_GEOMETRY = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300" / "csv" / "step9c_compact_paper_metrics.csv"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return pd.read_csv(path, encoding="utf-8-sig")



def method_label_from_dir(path: Path) -> str:

    report_path = path / "step9b_integrity_report.json"

    if report_path.exists():

        with open(report_path, "r", encoding="utf-8") as f:

            report = json.load(f)

        model = Path(str(report.get("model", path.name))).stem

        prompt_mode = report.get("prompt_mode", "synonyms")

        return f"YOLO-World {model} {prompt_mode}"

    return path.name



def compact_row_to_summary(row: pd.Series, label: str, family: str) -> dict:

    return {

        "method": label,

        "family": family,

        "balanced": float(row["precision_recall_unknown_balanced_score"]),

        "precision": float(row["known_precision"]),

        "recall": float(row["known_recall"]),

        "unknown_false_accepts": int(row["unknown_false_accept_objects"]),

        "unknown_reject": float(row["unknown_reject_rate_object_level"]),

        "background_false_accepts": int(row.get("background_false_accept_count", 0)),

        "AP50": float(row.get("AP50", row.get("unthresholded_AP50", float("nan")))),

        "AP75": float(row.get("AP75", row.get("unthresholded_AP75", float("nan")))),

        "AP": float(row.get("AP", row.get("unthresholded_AP", float("nan")))),

        "threshold": float(row.get("conf_thr", row.get("threshold", float("nan")))),

        "selection_policy": str(row.get("selection_policy", "calibration_selected_raw_threshold")),

    }



def write_markdown(path: Path, summary: pd.DataFrame) -> None:

    lines = [

        "# Step9d External Baseline Comparison",

        "",

        "Same-protocol comparison on LVIS-Clear-Mini-300.",

        "",

        summary.to_markdown(index=False),

        "",

        "Notes:",

        "",

        "- YOLO-World-s and YOLO-World-l are evaluated with the same prompts, splits, threshold-selection rule, and AP/open-set metrics.",

        "- The proposed geometry-risk row is included for context, but it is not a different detector backbone.",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--baseline_dirs", type=str, default=";".join(str(p) for p in DEFAULT_BASELINE_DIRS))

    parser.add_argument("--geometry_compact_csv", type=Path, default=DEFAULT_GEOMETRY)

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    rows: List[dict] = []

    inputs = []


    for text in str(args.baseline_dirs).split(";"):

        if not text.strip():

            continue

        d = Path(text.strip())

        compact_path = d / "csv" / "step9b_compact_paper_metrics.csv"

        compact = read_csv(compact_path)

        row = compact[compact["iou_threshold"].astype(float).round(4).eq(0.50)].iloc[0]

        label = method_label_from_dir(d)

        rows.append(compact_row_to_summary(row, label, "external_yoloworld_variant"))

        inputs.append(str(compact_path))


    if args.geometry_compact_csv.exists():

        geom = read_csv(args.geometry_compact_csv)

        for _, r in geom[geom["iou_threshold"].astype(float).round(4).eq(0.50)].iterrows():

            if str(r["method"]) == "step9b_raw_global_reproduced":

                continue

            label = f"Geometry-risk {r['selection_policy']}"

            rows.append(compact_row_to_summary(r, label, "proposed_calibration"))

        inputs.append(str(args.geometry_compact_csv))


    summary = pd.DataFrame(rows)

    summary = summary.sort_values(["family", "balanced", "AP50"], ascending=[True, False, False]).reset_index(drop=True)

    summary.to_csv(csv_dir / "step9d_external_baseline_summary.csv", index=False, encoding="utf-8-sig")


    ap_cols = ["method", "family", "AP50", "AP75", "AP", "balanced", "precision", "recall"]

    ap_comp = summary[ap_cols].sort_values(["AP50", "AP"], ascending=[False, False])

    ap_comp.to_csv(csv_dir / "step9d_ap_comparison.csv", index=False, encoding="utf-8-sig")


    open_cols = [

        "method",

        "family",

        "balanced",

        "precision",

        "recall",

        "unknown_false_accepts",

        "unknown_reject",

        "background_false_accepts",

        "AP50",

        "AP",

    ]

    open_comp = summary[open_cols].sort_values(["balanced", "unknown_false_accepts", "background_false_accepts"], ascending=[False, True, True])

    open_comp.to_csv(csv_dir / "step9d_open_set_comparison.csv", index=False, encoding="utf-8-sig")

    write_markdown(args.output_root / "step9d_summary.md", summary)


    report = {

        "method": "Step9d Same-Protocol External Baseline Comparison",

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "external_baseline_type": "YOLO-World model-size variants",

        "grounding_dino_status": "not run; YOLO-World variants chosen as lower-risk same-protocol external baseline",

        "inputs": inputs,

        "outputs": {

            "external_baseline_summary": str(csv_dir / "step9d_external_baseline_summary.csv"),

            "ap_comparison": str(csv_dir / "step9d_ap_comparison.csv"),

            "open_set_comparison": str(csv_dir / "step9d_open_set_comparison.csv"),

            "markdown_summary": str(args.output_root / "step9d_summary.md"),

        },

    }

    with open(args.output_root / "step9d_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)


    print("\n========== Step9d completed ==========")

    print(summary.to_string(index=False))



if __name__ == "__main__":

    main()

