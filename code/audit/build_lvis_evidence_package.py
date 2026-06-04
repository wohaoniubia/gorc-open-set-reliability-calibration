


"""
Step9f: Build the final paper-ready evidence package.

This script intentionally performs no model selection and no new inference. It
collects verified outputs from Step7w and Step9A-E, computes a per-class error
analysis on the held-out LVIS-Clear-Mini-300 test split, and writes a compact
paper evidence bundle with CSV tables, markdown summaries, and an integrity
report.
"""


from __future__ import annotations


import argparse

import json

import math

import shutil

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Sequence, Tuple


import pandas as pd


from step8j_yoloworld_lvis_openvoc_baseline import (

    bbox_iou,

    clean_columns,

    load_annotations,

    load_known_classes,

    load_split,

    normalize_image_id,

    safe_class_name,

)



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step9f_final_paper_ready_evidence_package"


DEFAULT_STEP7W_MAIN = DEFAULT_PROJECT_ROOT / "outputs" / "step7w_final_pgv_evidence_package" / "csv" / "step7w_main_results.csv"

DEFAULT_STEP9A_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_STEP9B_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9b_yoloworld_lvis300_raw_baseline"

DEFAULT_STEP9C_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300"

DEFAULT_STEP9D_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9d_external_baselines_lvis300"

DEFAULT_STEP9E_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9e_bootstrap_lvis300"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def copy_table(src: Path, dst: Path) -> pd.DataFrame:

    df = read_csv(src)

    df.to_csv(dst, index=False, encoding="utf-8-sig")

    return df



def finite_float(value, default: float = 0.0) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def metric_col(row: pd.Series, *names: str, default=0.0):

    for name in names:

        if name in row.index and pd.notna(row[name]):

            return row[name]

    return default



def compact_lvis_metrics(step9c_compact: pd.DataFrame) -> pd.DataFrame:

    rows: List[dict] = []

    for _, r in step9c_compact.iterrows():

        rows.append(

            {

                "method": r.get("method", ""),

                "selection_policy": r.get("selection_policy", ""),

                "split": r.get("split", "test"),

                "iou_threshold": metric_col(r, "iou_threshold", default=0.50),

                "threshold": metric_col(r, "threshold", default=""),

                "alpha": metric_col(r, "alpha", default=""),

                "balanced": metric_col(r, "precision_recall_unknown_balanced_score"),

                "precision": metric_col(r, "known_precision"),

                "recall": metric_col(r, "known_recall"),

                "unknown_false_accepts": metric_col(r, "unknown_false_accept_objects"),

                "unknown_reject": metric_col(r, "unknown_reject_rate_object_level"),

                "background_false_accepts": metric_col(r, "background_false_accept_count"),

                "AP50": metric_col(r, "unthresholded_AP50"),

                "AP75": metric_col(r, "unthresholded_AP75"),

                "AP": metric_col(r, "unthresholded_AP"),

                "calibration_balanced": metric_col(r, "cal_balanced", default=""),

                "calibration_precision": metric_col(r, "cal_precision", default=""),

                "calibration_recall": metric_col(r, "cal_recall", default=""),

                "calibration_AP": metric_col(r, "cal_AP", default=""),

            }

        )

    return pd.DataFrame(rows)



def compact_ablation(step9c_alpha: pd.DataFrame) -> pd.DataFrame:

    keep = [

        "alpha",

        "score_col",

        "selected_threshold",

        "cal_balanced",

        "cal_precision",

        "cal_recall",

        "cal_unknown_reject",

        "cal_ufa",

        "cal_AP50",

        "cal_AP75",

        "cal_AP",

        "test_balanced",

        "test_precision",

        "test_recall",

        "test_unknown_reject",

        "test_ufa",

        "test_AP50",

        "test_AP75",

        "test_AP",

    ]

    out = step9c_alpha[[c for c in keep if c in step9c_alpha.columns]].copy()

    out.insert(0, "ablation_family", "geometry_alpha_sweep")

    out.insert(1, "selection_rule", "calibration_split_max_balanced_per_alpha")

    return out



def select_detection_view(scored: pd.DataFrame, score_col: str, threshold: float, split: str = "test") -> pd.DataFrame:

    det = scored[scored["split"].astype(str).str.lower() == split.lower()].copy()

    det["_score"] = pd.to_numeric(det[score_col], errors="coerce")

    det = det[det["_score"] >= float(threshold)].copy()

    det["score"] = det["_score"].astype(float)

    det["image_id"] = det["image_id"].map(normalize_image_id)

    det["pred_label"] = det["pred_label"].map(safe_class_name)

    return det.sort_values("score", ascending=False).reset_index(drop=True)



def match_per_class(dets: pd.DataFrame, gt: pd.DataFrame, known_classes: Sequence[str], iou_thr: float) -> pd.DataFrame:

    gt = gt.copy()

    gt["image_id"] = gt["image_id"].map(normalize_image_id)

    gt["gt_label"] = gt["gt_label"].map(safe_class_name)

    gt_test = gt[gt["split"].astype(str).str.lower() == "test"].copy()

    known_gt = gt_test[gt_test["gt_is_known"]].copy()

    unknown_gt = gt_test[~gt_test["gt_is_known"]].copy()

    gt_by_image = {img: g.copy() for img, g in gt_test.groupby("image_id", sort=False)}


    known_rows: Dict[str, dict] = {}

    for cls in known_classes:

        known_rows[cls] = {

            "analysis_class": cls,

            "class_role": "known",

            "gt_count": int((known_gt["gt_label"] == cls).sum()),

            "predicted_accept_count": int((dets["pred_label"] == cls).sum()) if len(dets) else 0,

            "tp": 0,

            "fp": 0,

            "fn": 0,

            "unknown_false_accept_objects": 0,

            "background_false_accept_count": 0,

            "wrong_known_or_duplicate_count": 0,

            "precision": 0.0,

            "recall": 0.0,

        }

    unknown_accept_sets: Dict[str, set] = {cls: set() for cls in known_classes}

    matched_known = set()


    for _, pr in dets.iterrows():

        pred_label = safe_class_name(pr.pred_label)

        if pred_label not in known_rows:

            continue

        image_id = normalize_image_id(pr.image_id)

        pbox = (finite_float(pr.x1), finite_float(pr.y1), finite_float(pr.x2), finite_float(pr.y2))

        gimg = gt_by_image.get(image_id, gt_test.iloc[0:0])


        best_known_iou, best_known_key, best_known_label = 0.0, None, ""

        for _, gr in gimg[gimg["gt_is_known"]].iterrows():

            key = (normalize_image_id(gr.image_id), gr.gt_id)

            if key in matched_known:

                continue

            iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

            if iou > best_known_iou:

                best_known_iou = float(iou)

                best_known_key = key

                best_known_label = safe_class_name(gr.gt_label)


        if best_known_key is not None and best_known_iou >= float(iou_thr) and pred_label == best_known_label:

            known_rows[pred_label]["tp"] += 1

            matched_known.add(best_known_key)

            continue


        known_rows[pred_label]["fp"] += 1


        best_unknown_iou, best_unknown_key = 0.0, None

        for _, gr in gimg[~gimg["gt_is_known"]].iterrows():

            iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

            if iou > best_unknown_iou:

                best_unknown_iou = float(iou)

                best_unknown_key = (normalize_image_id(gr.image_id), gr.gt_id)


        if best_unknown_key is not None and best_unknown_iou >= float(iou_thr):

            unknown_accept_sets[pred_label].add(best_unknown_key)

        elif best_known_iou >= float(iou_thr):

            known_rows[pred_label]["wrong_known_or_duplicate_count"] += 1

        else:

            known_rows[pred_label]["background_false_accept_count"] += 1


    rows: List[dict] = []

    for cls in known_classes:

        row = known_rows[cls]

        row["unknown_false_accept_objects"] = len(unknown_accept_sets[cls])

        row["fn"] = max(0, int(row["gt_count"]) - int(row["tp"]))

        den = int(row["tp"]) + int(row["fp"])

        row["precision"] = float(row["tp"] / den) if den else 0.0

        row["recall"] = float(row["tp"] / row["gt_count"]) if int(row["gt_count"]) else 0.0

        rows.append(row.copy())


    unknown_rows: List[dict] = []

    for cls, sub in unknown_gt.groupby("gt_label", sort=True):

        accepted = set()

        for _, gr in sub.iterrows():

            key = (normalize_image_id(gr.image_id), gr.gt_id)

            gbox = (gr.x1, gr.y1, gr.x2, gr.y2)

            candidates = dets[dets["image_id"].map(normalize_image_id) == normalize_image_id(gr.image_id)]

            for _, pr in candidates.iterrows():

                if bbox_iou((pr.x1, pr.y1, pr.x2, pr.y2), gbox) >= float(iou_thr):

                    accepted.add(key)

                    break

        gt_count = int(len(sub))

        ufa = int(len(accepted))

        unknown_rows.append(

            {

                "analysis_class": cls,

                "class_role": "unknown",

                "gt_count": gt_count,

                "predicted_accept_count": "",

                "tp": "",

                "fp": "",

                "fn": "",

                "unknown_false_accept_objects": ufa,

                "background_false_accept_count": "",

                "wrong_known_or_duplicate_count": "",

                "precision": "",

                "recall": "",

                "unknown_reject": float(1.0 - ufa / gt_count) if gt_count else 1.0,

            }

        )


    out = pd.DataFrame(rows + unknown_rows)

    if "unknown_reject" not in out.columns:

        out["unknown_reject"] = ""

    return out



def build_per_class_error_table(

    scored: pd.DataFrame,

    compact: pd.DataFrame,

    gt: pd.DataFrame,

    known_classes: Sequence[str],

    iou_thr: float,

) -> pd.DataFrame:

    method_rows = []

    for _, r in compact.iterrows():

        method = str(r.get("method", ""))

        if method not in {

            "step9b_raw_global_reproduced",

            "step9c_geometry_max_cal_balanced",

            "step9c_geometry_max_cal_ap",

            "step9c_geometry_precision_tiebreak_eps0p005",

        }:

            continue

        score_col = str(r.get("score_col", ""))

        threshold = finite_float(r.get("threshold", 0.0))

        if score_col not in scored.columns:

            continue

        dets = select_detection_view(scored, score_col, threshold, split="test")

        pc = match_per_class(dets, gt, known_classes, iou_thr=iou_thr)

        pc.insert(0, "method", method)

        pc.insert(1, "selection_policy", r.get("selection_policy", ""))

        pc.insert(2, "score_col", score_col)

        pc.insert(3, "threshold", threshold)

        pc.insert(4, "iou_threshold", float(iou_thr))

        method_rows.append(pc)

    if not method_rows:

        raise ValueError("No method rows available for per-class analysis.")

    return pd.concat(method_rows, ignore_index=True)



def md_table(df: pd.DataFrame, max_rows: int = 12) -> str:

    if df.empty:

        return "_No rows._"

    small = df.head(max_rows).copy()

    return small.to_markdown(index=False)



def write_final_method_summary(path: Path, report: dict, main85: pd.DataFrame, lvis: pd.DataFrame, boot_delta: pd.DataFrame) -> None:

    primary = lvis[lvis["method"].eq("step9c_geometry_max_cal_balanced")]

    raw = lvis[lvis["method"].eq("step9b_raw_global_reproduced")]

    if not primary.empty and not raw.empty:

        p = primary.iloc[0]

        r = raw.iloc[0]

        delta_bal = finite_float(p["balanced"]) - finite_float(r["balanced"])

        delta_ufa = finite_float(p["unknown_false_accepts"]) - finite_float(r["unknown_false_accepts"])

        delta_bg = finite_float(p["background_false_accepts"]) - finite_float(r["background_false_accepts"])

        delta_ap = finite_float(p["AP"]) - finite_float(r["AP"])

    else:

        delta_bal = delta_ufa = delta_bg = delta_ap = 0.0


    lines = [

        "# Final Method Summary",

        "",

        "## Recommended paper framing",

        "",

        "Geometry-Aware Open-Set Reliability Calibration for Foundation-Model Object Detection under Few-shot Open-world Settings.",

        "",

        "The public-validation branch should be presented as a calibration-only reliability method for a foundation detector, not as a full LVIS detector SOTA claim and not as a strict support-conditioned few-shot learner.",

        "",

        "## Core evidence",

        "",

        "- Practical 85-scene branch: PGV-Veto is the primary strict few-shot open-world route and consistently reduces hard unknown false accepts across 1/2/4-shot settings.",

        "- Public LVIS-Clear-Mini-300 branch: geometry-risk calibration improves the held-out IoU0.50 open-set operating point over raw YOLO-World-l.",

        f"- Main held-out LVIS operating delta: balanced `{delta_bal:.4f}`, UFA `{delta_ufa:.0f}`, background false accepts `{delta_bg:.0f}`, AP `{delta_ap:.4f}`.",

        "- Bootstrap confirms the open-set reliability gain, while AP does not improve for the strict balanced policy.",

        "",

        "## Claim boundaries",

        "",

        "- Use AP/AP50/AP75 and IoU0.50 open-set metrics as main public metrics.",

        "- Treat IoU0.30 as diagnostic only.",

        "- State that LVIS-Clear-Mini-300 is a custom public validation protocol derived from local LVIS/COCO assets.",

        "- The strongest external comparison in this package is same-protocol YOLO-World model-size variants; Grounding DINO was not included in this run.",

        "",

        "## 85-scene main table",

        "",

        md_table(main85, 10),

        "",

        "## LVIS-Clear-Mini-300 main table",

        "",

        md_table(lvis, 10),

        "",

        "## Bootstrap delta table",

        "",

        md_table(boot_delta, 10),

        "",

        "## Reproducibility",

        "",

        f"- Final package root: `{report['output_root']}`",

        f"- Generated at: `{report['generated_at']}`",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_public_validation_summary(

    path: Path,

    label_summary: pd.DataFrame,

    split_summary: pd.DataFrame,

    lvis: pd.DataFrame,

    external: pd.DataFrame,

    boot_delta: pd.DataFrame,

    per_class: pd.DataFrame,

) -> None:

    lines = [

        "# Public Validation Summary",

        "",

        "## Protocol",

        "",

        "LVIS-Clear-Mini-300 uses 300 public images with four fixed known classes: bottle, cup, chair, and bowl. Unknown classes are retained from the Step8 public branch and balanced as far as local annotations allow.",

        "",

        "## Split summary",

        "",

        md_table(split_summary, 10),

        "",

        "## Class counts",

        "",

        md_table(label_summary, 20),

        "",

        "## Held-out test metrics",

        "",

        md_table(lvis, 10),

        "",

        "## Same-protocol external baseline comparison",

        "",

        md_table(external, 10),

        "",

        "## Paired image bootstrap",

        "",

        md_table(boot_delta, 10),

        "",

        "## Per-class error snapshot",

        "",

        md_table(per_class, 18),

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--iou_threshold", type=float, default=0.50)

    args = parser.parse_args()


    docs_dir = args.output_root / "docs"

    tables_dir = args.output_root / "tables"

    ensure_dir(docs_dir)

    ensure_dir(tables_dir)


    step9a_root = args.project_root / "outputs" / "step9a_lvis_clear_mini_300_protocol"

    step9c_root = args.project_root / "outputs" / "step9c_geometry_risk_lvis300"

    step9d_root = args.project_root / "outputs" / "step9d_external_baselines_lvis300"

    step9e_root = args.project_root / "outputs" / "step9e_bootstrap_lvis300"


    paths = {

        "step7w_main_results": DEFAULT_STEP7W_MAIN,

        "step9a_label_summary": step9a_root / "csv" / "step9a_label_summary.csv",

        "step9a_split_summary": step9a_root / "csv" / "step9a_split_summary.csv",

        "step9a_scene_split": step9a_root / "csv" / "step9a_scene_split.csv",

        "step9a_class_map": step9a_root / "csv" / "step9a_class_map.csv",

        "step9a_annotation_dir": step9a_root / "annotations",

        "step9c_compact": step9c_root / "csv" / "step9c_compact_paper_metrics.csv",

        "step9c_alpha_sweep": step9c_root / "csv" / "step9c_alpha_sweep_summary.csv",

        "step9c_scored": step9c_root / "csv" / "step9c_geometry_scored_candidates.csv",

        "step9d_external": step9d_root / "csv" / "step9d_external_baseline_summary.csv",

        "step9e_bootstrap_delta": step9e_root / "csv" / "step9e_bootstrap_delta.csv",

    }


    missing = [str(p) for p in paths.values() if isinstance(p, Path) and not p.exists()]

    if missing:

        raise FileNotFoundError("Missing required input<SET_LOCAL_PATH>" + "\n".join(missing))


    main85 = copy_table(paths["step7w_main_results"], tables_dir / "main_results_85scene.csv")

    label_summary = read_csv(paths["step9a_label_summary"])

    split_summary = read_csv(paths["step9a_split_summary"])


    step9c_compact = read_csv(paths["step9c_compact"])

    lvis = compact_lvis_metrics(step9c_compact)

    lvis.to_csv(tables_dir / "main_results_lvis_larger.csv", index=False, encoding="utf-8-sig")


    ablation = compact_ablation(read_csv(paths["step9c_alpha_sweep"]))

    ablation.to_csv(tables_dir / "ablation_public.csv", index=False, encoding="utf-8-sig")


    external = copy_table(paths["step9d_external"], tables_dir / "external_baseline_comparison.csv")

    boot_delta = copy_table(paths["step9e_bootstrap_delta"], tables_dir / "bootstrap_delta.csv")


    split_df = load_split(paths["step9a_scene_split"], args.project_root, split_filter="all")

    known_classes = load_known_classes(paths["step9a_class_map"])

    gt = load_annotations(paths["step9a_annotation_dir"], split_df, known_classes)

    scored = read_csv(paths["step9c_scored"])

    per_class = build_per_class_error_table(scored, step9c_compact, gt, known_classes, args.iou_threshold)

    per_class.to_csv(tables_dir / "per_class_error_analysis.csv", index=False, encoding="utf-8-sig")


    report = {

        "method": "Step9f Final Paper-Ready Evidence Package",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "inputs": {k: str(v) for k, v in paths.items()},

        "outputs": {

            "docs/final_method_summary.md": str(docs_dir / "final_method_summary.md"),

            "docs/public_validation_summary.md": str(docs_dir / "public_validation_summary.md"),

            "tables/main_results_85scene.csv": str(tables_dir / "main_results_85scene.csv"),

            "tables/main_results_lvis_larger.csv": str(tables_dir / "main_results_lvis_larger.csv"),

            "tables/ablation_public.csv": str(tables_dir / "ablation_public.csv"),

            "tables/external_baseline_comparison.csv": str(tables_dir / "external_baseline_comparison.csv"),

            "tables/bootstrap_delta.csv": str(tables_dir / "bootstrap_delta.csv"),

            "tables/per_class_error_analysis.csv": str(tables_dir / "per_class_error_analysis.csv"),

            "final_integrity_report.json": str(args.output_root / "final_integrity_report.json"),

        },

        "row_counts": {

            "main_results_85scene": int(len(main85)),

            "main_results_lvis_larger": int(len(lvis)),

            "ablation_public": int(len(ablation)),

            "external_baseline_comparison": int(len(external)),

            "bootstrap_delta": int(len(boot_delta)),

            "per_class_error_analysis": int(len(per_class)),

        },

        "research_discipline": {

            "calibration_only_selection": True,

            "test_split_used_for_final_evaluation_only": True,

            "main_public_iou_threshold": float(args.iou_threshold),

            "iou030_main_metric": False,

            "full_lvis_sota_claim": False,

        },

    }


    write_final_method_summary(docs_dir / "final_method_summary.md", report, main85, lvis, boot_delta)

    write_public_validation_summary(

        docs_dir / "public_validation_summary.md",

        label_summary,

        split_summary,

        lvis,

        external,

        boot_delta,

        per_class,

    )



    log_src = args.project_root / "outputs" / "step9_research_log_2026-05-01.md"

    if log_src.exists():

        shutil.copyfile(log_src, args.output_root / "step9_research_log_final.md")

        report["outputs"]["step9_research_log_final.md"] = str(args.output_root / "step9_research_log_final.md")


    (args.output_root / "final_integrity_report.json").write_text(

        json.dumps(report, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )


    summary_lines = [

        "# Step9F Final Evidence Package Summary",

        "",

        f"- Output root: `{args.output_root}`",

        f"- Generated at: `{report['generated_at']}`",

        f"- Tables written: `{len(report['outputs']) - 3}` core table/docs entries plus integrity/log files",

        "",

        "## Row counts",

        "",

        pd.DataFrame([report["row_counts"]]).to_markdown(index=False),

        "",

        "## Required outputs",

        "",

    ]

    for rel, out_path in report["outputs"].items():

        summary_lines.append(f"- `{rel}` -> `{out_path}`")

    (args.output_root / "step9f_summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


    print("Step9F final package completed.")

    print(json.dumps(report["row_counts"], ensure_ascii=False, indent=2))



if __name__ == "__main__":

    main()

