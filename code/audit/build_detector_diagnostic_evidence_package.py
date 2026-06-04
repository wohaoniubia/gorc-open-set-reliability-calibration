


"""
Step11E: Build final high-level SCI evidence package.

The package consolidates the 85-scene practical branch, LVIS-Clear-Mini-300
YOLO/Grounding-DINO public validation, statistical support, diagnostics, and
claim-boundary documents.
"""


from __future__ import annotations


import argparse

import json

import shutil

from datetime import datetime

from pathlib import Path

from typing import Dict, List


import pandas as pd



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step11e_high_level_sci_evidence_package"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    return pd.read_csv(path, encoding="utf-8-sig")



def copy_if_exists(src: Path, dst: Path, copied: List[dict]) -> bool:

    if not src.exists():

        copied.append({"src": str(src), "dst": str(dst), "status": "missing"})

        return False

    ensure_dir(dst.parent)

    shutil.copy2(src, dst)

    copied.append({"src": str(src), "dst": str(dst), "status": "copied"})

    return True



def fmt(v, digits: int = 4) -> str:

    try:

        return f"{float(v):.{digits}f}"

    except Exception:

        return "NA"



def first_row(df: pd.DataFrame, predicate) -> dict:

    if df.empty:

        return {}

    sub = df[df.apply(predicate, axis=1)].copy()

    if sub.empty:

        return {}

    return sub.iloc[0].to_dict()



def write_readiness_doc(path: Path, comparison: pd.DataFrame, claim: pd.DataFrame, bootstrap_delta: pd.DataFrame) -> None:

    yolo_raw = first_row(comparison, lambda r: r.get("method_name") == "YOLO-World-l raw")

    yolo_ap = first_row(comparison, lambda r: r.get("method_name") == "YOLO-World-l geometry AP-constrained")

    yolo_rel = first_row(comparison, lambda r: r.get("method_name") == "YOLO-World-l geometry reliability-first")

    gd_raw = first_row(comparison, lambda r: r.get("method_name") == "Grounding DINO tiny raw")

    gd_rel = first_row(comparison, lambda r: r.get("method_name") == "Grounding DINO geometry reliability-first")

    gd_ap = first_row(comparison, lambda r: r.get("method_name") == "Grounding DINO geometry AP-constrained")

    boot = bootstrap_delta.iloc[0].to_dict() if len(bootstrap_delta) else {}


    lines = [

        "# Final SCI Readiness Assessment",

        "",

        "## 1. Is the work ready for manuscript drafting?",

        "",

        "Yes. The project now has a coherent paper-safe story: a practical 85-scene few-shot open-world branch plus a public LVIS-Clear-Mini-300 validation branch centered on geometry-aware open-set reliability calibration.",

        "",

        "## 2. Is it ready for high-level SCI submission?",

        "",

        "Conditionally yes for a carefully framed CAS Zone-2-oriented submission, provided the manuscript avoids SOTA overclaims. The strongest evidence supports reliability calibration, false-accept suppression, and AP/reliability trade-off analysis rather than full LVIS detector AP superiority.",

        "",

        "## 3. What claim should be made?",

        "",

        "Recommended claim: geometry-aware reliability calibration improves open-set reliability for foundation-model object detection under few-shot/open-world settings, with AP-constrained reliability demonstrated strongly on YOLO-World-l and cross-detector transfer evidence on Grounding DINO with quantified trade-offs.",

        "",

        "Key YOLO-World-l evidence:",

        "",

        f"- Raw: Balanced {fmt(yolo_raw.get('balanced'))}, UFA {fmt(yolo_raw.get('unknown_false_accepts'), 0)}, BG FP {fmt(yolo_raw.get('background_false_accepts'), 0)}, AP50 {fmt(yolo_raw.get('AP50'))}, AP {fmt(yolo_raw.get('AP'))}.",

        f"- Reliability-first: Balanced {fmt(yolo_rel.get('balanced'))}, UFA {fmt(yolo_rel.get('unknown_false_accepts'), 0)}, BG FP {fmt(yolo_rel.get('background_false_accepts'), 0)}, AP50 {fmt(yolo_rel.get('AP50'))}, AP {fmt(yolo_rel.get('AP'))}.",

        f"- AP-constrained: Balanced {fmt(yolo_ap.get('balanced'))}, UFA {fmt(yolo_ap.get('unknown_false_accepts'), 0)}, BG FP {fmt(yolo_ap.get('background_false_accepts'), 0)}, AP50 {fmt(yolo_ap.get('AP50'))}, AP {fmt(yolo_ap.get('AP'))}.",

        "",

        "Key Grounding DINO evidence:",

        "",

        f"- Raw: Balanced {fmt(gd_raw.get('balanced'))}, UFA {fmt(gd_raw.get('unknown_false_accepts'), 0)}, BG FP {fmt(gd_raw.get('background_false_accepts'), 0)}, AP50 {fmt(gd_raw.get('AP50'))}, AP {fmt(gd_raw.get('AP'))}.",

        f"- Reliability-first: Balanced {fmt(gd_rel.get('balanced'))}, UFA {fmt(gd_rel.get('unknown_false_accepts'), 0)}, BG FP {fmt(gd_rel.get('background_false_accepts'), 0)}, AP50 {fmt(gd_rel.get('AP50'))}, AP {fmt(gd_rel.get('AP'))}.",

        f"- AP-constrained: Balanced {fmt(gd_ap.get('balanced'))}, UFA {fmt(gd_ap.get('unknown_false_accepts'), 0)}, BG FP {fmt(gd_ap.get('background_false_accepts'), 0)}, AP50 {fmt(gd_ap.get('AP50'))}, AP {fmt(gd_ap.get('AP'))}.",

        "",

        "Bootstrap support for the strict YOLO AP-constrained policy:",

        "",

        f"- Delta balanced mean {fmt(boot.get('delta_precision_recall_unknown_balanced_score_mean'))}, 95% CI [{fmt(boot.get('delta_precision_recall_unknown_balanced_score_ci025'))}, {fmt(boot.get('delta_precision_recall_unknown_balanced_score_ci975'))}].",

        f"- Delta UFA mean {fmt(boot.get('delta_unknown_false_accept_objects_mean'))}, 95% CI [{fmt(boot.get('delta_unknown_false_accept_objects_ci025'))}, {fmt(boot.get('delta_unknown_false_accept_objects_ci975'))}].",

        f"- Delta BG FP mean {fmt(boot.get('delta_background_false_accept_count_mean'))}, 95% CI [{fmt(boot.get('delta_background_false_accept_count_ci025'))}, {fmt(boot.get('delta_background_false_accept_count_ci975'))}].",

        f"- Delta AP50 mean {fmt(boot.get('delta_AP50_mean'))}, 95% CI [{fmt(boot.get('delta_AP50_ci025'))}, {fmt(boot.get('delta_AP50_ci975'))}].",

        f"- Delta AP mean {fmt(boot.get('delta_AP_mean'))}, 95% CI [{fmt(boot.get('delta_AP_ci025'))}, {fmt(boot.get('delta_AP_ci975'))}].",

        "",

        "## 4. What exact gap remains?",

        "",

        "The remaining scientific gap is scope, not basic engineering: the public protocol is custom LVIS-Clear-Mini-300 rather than full LVIS, and Grounding DINO does not show strict AP-constrained balanced improvement. For a stronger venue or more ambitious claim, add a larger public validation or another detector/dataset.",

        "",

        "## 5. Does current evidence support detector-agnostic reliability calibration?",

        "",

        "Partially. Both YOLO-World-l and Grounding DINO show reduced false accepts/background errors under geometry-aware calibration. However, detector-agnostic AP-constrained balanced improvement is not established.",

        "",

        "## 6. Does current evidence support only YOLO-World-specific calibration?",

        "",

        "No. Grounding DINO transfer evidence is real for reliability/false-accept behavior, but the strongest AP-constrained claim should be stated as YOLO-World-l validated.",

        "",

        "## Final Decision",

        "",

        "Proceed to manuscript drafting. The paper should target a reliability-calibration contribution, not a full LVIS SOTA detector claim.",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_claim_boundaries(path: Path, claim: pd.DataFrame) -> None:

    lines = [

        "# Final Claim Boundaries",

        "",

        "## Claims Supported",

        "",

        "- Geometry-aware calibration improves YOLO-World-l open-set reliability on LVIS-Clear-Mini-300.",

        "- A strict YOLO-World-l AP-constrained operating mode reduces UFA and background false accepts with near-neutral AP/AP50.",

        "- Grounding DINO full same-protocol validation supports transfer of reliability/false-accept suppression behavior.",

        "- The AP/reliability trade-off is measurable and should be treated as part of the method design.",

        "",

        "## Claims Not Supported",

        "",

        "- Do not claim full LVIS SOTA.",

        "- Do not claim universal detector-agnostic AP-constrained superiority.",

        "- Do not claim Grounding DINO strict AP-constrained balanced improvement.",

        "- Do not present IoU0.30 as the public validation main metric.",

        "",

        "## Claim Support Matrix",

        "",

        claim.to_markdown(index=False) if len(claim) else "Claim matrix missing.",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_method_framing(path: Path) -> None:

    lines = [

        "# Final Method Framing",

        "",

        "Recommended title:",

        "",

        "Geometry-Aware Open-Set Reliability Calibration for Foundation-Model Object Detection under Few-Shot Open-World Settings",

        "",

        "Core contribution:",

        "",

        "1. A calibration-only geometry-aware reliability model for foundation-model object detectors.",

        "2. An AP-constrained reliability operating mode that reduces unknown/background false accepts while keeping AP/AP50 near-neutral on YOLO-World-l.",

        "3. A reliability-first operating mode for stronger false-accept suppression when the application prioritizes open-set risk control.",

        "4. Same-protocol public validation on LVIS-Clear-Mini-300 with YOLO-World and Grounding DINO.",

        "5. Reliability diagnostics, risk-coverage curves, and paired bootstrap uncertainty estimates.",

        "",

        "Method wording:",

        "",

        "The method should be described as post-hoc calibration using detector scores, class identity, and bounding-box geometry. The model-selection split is calibration-only; the held-out test split is used only for final evaluation.",

        "",

        "Main metrics:",

        "",

        "- AP/AP50/AP75",

        "- IoU0.50 open-set balanced",

        "- precision/recall",

        "- unknown false accepts",

        "- background false accepts",

        "",

        "IoU0.30 should remain diagnostic and should not be used as a public main-result metric.",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    args = parser.parse_args()


    docs_dir = args.output_root / "docs"

    tables_dir = args.output_root / "tables"

    figures_dir = args.output_root / "figures"

    ensure_dir(docs_dir)

    ensure_dir(tables_dir)

    ensure_dir(figures_dir)


    copied: List[dict] = []

    step9f = args.project_root / "outputs" / "step9f_final_paper_ready_evidence_package"

    step11c = args.project_root / "outputs" / "step11c_cross_detector_comparison"

    step11d = args.project_root / "outputs" / "step11d_reliability_diagnostics"

    step10b = args.project_root / "outputs" / "step10b_ap_constrained_bootstrap"

    step9e = args.project_root / "outputs" / "step9e_bootstrap_lvis300"


    copy_if_exists(step9f / "tables" / "main_results_85scene.csv", tables_dir / "main_85scene_results.csv", copied)


    comparison = read_csv(step11c / "csv" / "step11c_main_comparison.csv")

    yolo = comparison[comparison["detector"].astype(str).eq("YOLO-World")].copy() if len(comparison) else pd.DataFrame()

    gd = comparison[comparison["detector"].astype(str).eq("Grounding DINO")].copy() if len(comparison) else pd.DataFrame()

    yolo.to_csv(tables_dir / "lvis300_yoloworld_results.csv", index=False, encoding="utf-8-sig")

    gd.to_csv(tables_dir / "groundingdino_results.csv", index=False, encoding="utf-8-sig")

    comparison.to_csv(tables_dir / "cross_detector_comparison.csv", index=False, encoding="utf-8-sig")


    claim = read_csv(step11c / "csv" / "step11c_claim_support_matrix.csv")

    copy_if_exists(step11c / "csv" / "step11c_ap_reliability_tradeoff.csv", tables_dir / "ap_reliability_tradeoff.csv", copied)


    boot9 = read_csv(step9e / "csv" / "step9e_bootstrap_delta.csv")

    if len(boot9):

        boot9.insert(0, "source", "step9e_reliability_first")

    boot10 = read_csv(step10b / "csv" / "step10b_bootstrap_delta.csv")

    if len(boot10):

        boot10.insert(0, "source", "step10b_ap_constrained")

    bootstrap = pd.concat([boot9, boot10], ignore_index=True, sort=False) if len(boot9) or len(boot10) else pd.DataFrame()

    bootstrap.to_csv(tables_dir / "bootstrap_and_statistical_support.csv", index=False, encoding="utf-8-sig")


    ece = read_csv(step11d / "csv" / "step11d_ece_summary.csv")

    risk_key = read_csv(step11d / "csv" / "step11d_risk_coverage_key_points.csv")

    ece.to_csv(tables_dir / "reliability_diagnostics.csv", index=False, encoding="utf-8-sig")

    risk_key.to_csv(tables_dir / "risk_coverage_key_points.csv", index=False, encoding="utf-8-sig")



    for src_name, dst_name in [

        ("main_results_lvis_larger.csv", "step9_lvis300_main_results.csv"),

        ("ablation_public.csv", "ablation_public.csv"),

        ("external_baseline_comparison.csv", "external_baseline_comparison_step9.csv"),

        ("per_class_error_analysis.csv", "per_class_error_analysis.csv"),

    ]:

        copy_if_exists(step9f / "tables" / src_name, tables_dir / dst_name, copied)


    for fig in (step11d / "figures").glob("*.png"):

        copy_if_exists(fig, figures_dir / fig.name, copied)

    step10_fig_dir = args.project_root / "outputs" / "step10a_ap_constrained_reliability_calibration" / "figures"

    if step10_fig_dir.exists():

        for fig in step10_fig_dir.glob("*.png"):

            copy_if_exists(fig, figures_dir / fig.name, copied)


    copy_if_exists(step9f / "docs" / "final_method_summary.md", docs_dir / "step9_final_method_summary.md", copied)

    copy_if_exists(step9f / "docs" / "public_validation_summary.md", docs_dir / "step9_public_validation_summary.md", copied)

    copy_if_exists(args.project_root / "outputs" / "step11_research_log_2026-05-01.md", docs_dir / "step11_research_log_2026-05-01.md", copied)


    write_readiness_doc(docs_dir / "final_sci_readiness_assessment.md", comparison, claim, boot10)

    write_claim_boundaries(docs_dir / "final_claim_boundaries.md", claim)

    write_method_framing(docs_dir / "final_method_framing.md")


    report = {

        "method": "Step11E high-level SCI evidence package",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "required_outputs": {

            "docs/final_sci_readiness_assessment.md": str(docs_dir / "final_sci_readiness_assessment.md"),

            "docs/final_claim_boundaries.md": str(docs_dir / "final_claim_boundaries.md"),

            "docs/final_method_framing.md": str(docs_dir / "final_method_framing.md"),

            "tables/main_85scene_results.csv": str(tables_dir / "main_85scene_results.csv"),

            "tables/lvis300_yoloworld_results.csv": str(tables_dir / "lvis300_yoloworld_results.csv"),

            "tables/groundingdino_results.csv": str(tables_dir / "groundingdino_results.csv"),

            "tables/cross_detector_comparison.csv": str(tables_dir / "cross_detector_comparison.csv"),

            "tables/bootstrap_and_statistical_support.csv": str(tables_dir / "bootstrap_and_statistical_support.csv"),

            "tables/reliability_diagnostics.csv": str(tables_dir / "reliability_diagnostics.csv"),

            "figures": str(figures_dir),

        },

        "row_counts": {

            "comparison": int(len(comparison)),

            "yoloworld_results": int(len(yolo)),

            "groundingdino_results": int(len(gd)),

            "bootstrap_rows": int(len(bootstrap)),

            "reliability_diagnostics": int(len(ece)),

        },

        "copied_files": copied,

        "claim_boundary": "Detector-agnostic reliability/false-accept evidence; strongest AP-constrained balanced improvement is YOLO-World-l validated.",

        "ready_for_manuscript_drafting": True,

        "ready_for_high_level_sci_submission": "conditionally_yes_with_claim_boundaries",

    }

    with open(args.output_root / "final_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)


    print("\n========== Step11E completed ==========")

    print(f"Package: {args.output_root}")

    print(f"Docs: {docs_dir}")

    print(f"Tables: {tables_dir}")

    print(f"Figures: {figures_dir}")



if __name__ == "__main__":

    main()

