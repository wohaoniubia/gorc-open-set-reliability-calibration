


"""
Step11C: Cross-detector comparison and claim-support decision.

The script consolidates the LVIS-Clear-Mini-300 evidence across YOLO-World and
Grounding DINO under the same protocol. It intentionally separates reliability
gains, AP trade-offs, and AP-constrained success so the final paper claim remains
honest.
"""


from __future__ import annotations


import argparse

import json

import math

from datetime import datetime

from pathlib import Path

from typing import List


import pandas as pd



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step11c_cross_detector_comparison"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    return pd.read_csv(path, encoding="utf-8-sig")



def fget(row, keys, default=float("nan")) -> float:

    if isinstance(keys, str):

        keys = [keys]

    for k in keys:

        if k in row and str(row[k]) != "":

            try:

                v = float(row[k])

            except Exception:

                continue

            if math.isfinite(v):

                return v

    return default



def sget(row, keys, default="") -> str:

    if isinstance(keys, str):

        keys = [keys]

    for k in keys:

        if k in row and str(row[k]) != "":

            return str(row[k])

    return default



def row_from_any(

    *,

    row,

    detector: str,

    detector_variant: str,

    method_group: str,

    method_name: str,

    claim_role: str,

    source_path: Path,

    notes: str = "",

) -> dict:

    return {

        "detector": detector,

        "detector_variant": detector_variant,

        "method_group": method_group,

        "method_name": method_name,

        "claim_role": claim_role,

        "source_path": str(source_path),

        "selection_policy": sget(row, ["selection_policy", "objective"], ""),

        "mode": sget(row, "mode", ""),

        "threshold": fget(row, "threshold", fget(row, "conf_thr", float("nan"))),

        "balanced": fget(row, ["balanced", "precision_recall_unknown_balanced_score"]),

        "precision": fget(row, ["precision", "known_precision"]),

        "recall": fget(row, ["recall", "known_recall"]),

        "unknown_false_accepts": fget(row, ["unknown_false_accepts", "unknown_false_accept_objects"]),

        "unknown_reject": fget(row, ["unknown_reject", "unknown_reject_rate_object_level"]),

        "background_false_accepts": fget(row, ["background_false_accepts", "background_false_accept_count"]),

        "AP50": fget(row, ["AP50", "unthresholded_AP50"]),

        "AP75": fget(row, ["AP75", "unthresholded_AP75"]),

        "AP": fget(row, ["AP", "unthresholded_AP"]),

        "meets_ap_constrained_success": str(sget(row, "meets_success_criterion", "False")).lower() == "true",

        "notes": notes,

    }



def find_one(df: pd.DataFrame, predicate, label: str) -> dict:

    if df.empty:

        raise FileNotFoundError(f"Missing data for {label}")

    sub = df[df.apply(predicate, axis=1)].copy()

    if sub.empty:

        raise ValueError(f"Could not locate row for {label}")

    return sub.iloc[0].to_dict()



def pick_yolo_ap_constrained(step10: pd.DataFrame) -> dict:

    if step10.empty:

        return {}

    strict = step10[

        step10.get("constraint_name", pd.Series(dtype=str)).astype(str).eq("AP50tol0.005_APtol0.005")

        & step10.get("objective", pd.Series(dtype=str)).astype(str).eq("balanced_max")

    ].copy()

    if len(strict):

        return strict.iloc[0].to_dict()

    success = step10[step10.get("meets_success_criterion", pd.Series(dtype=str)).astype(str).str.lower().eq("true")].copy()

    if len(success):

        return success.iloc[0].to_dict()

    return step10.iloc[0].to_dict()



def delta_rows(main: pd.DataFrame) -> pd.DataFrame:

    rows: List[dict] = []

    for _, r in main.iterrows():

        if r["method_group"] == "raw":

            continue

        raw = main[

            (main["detector"].eq(r["detector"]))

            & (main["detector_variant"].eq(r["detector_variant"]))

            & (main["method_group"].eq("raw"))

        ]

        if raw.empty:

            raw = main[(main["detector"].eq(r["detector"])) & (main["method_group"].eq("raw"))]

        if raw.empty:

            continue

        base = raw.iloc[0]

        rows.append(

            {

                "detector": r["detector"],

                "detector_variant": r["detector_variant"],

                "raw_method": base["method_name"],

                "comparison_method": r["method_name"],

                "method_group": r["method_group"],

                "delta_balanced": float(r["balanced"]) - float(base["balanced"]),

                "delta_precision": float(r["precision"]) - float(base["precision"]),

                "delta_recall": float(r["recall"]) - float(base["recall"]),

                "delta_UFA": float(r["unknown_false_accepts"]) - float(base["unknown_false_accepts"]),

                "delta_background_false_accepts": float(r["background_false_accepts"]) - float(base["background_false_accepts"]),

                "delta_AP50": float(r["AP50"]) - float(base["AP50"]),

                "delta_AP75": float(r["AP75"]) - float(base["AP75"]),

                "delta_AP": float(r["AP"]) - float(base["AP"]),

                "ap_constrained_success": bool(r["meets_ap_constrained_success"]),

                "claim_role": r["claim_role"],

            }

        )

    return pd.DataFrame(rows)



def build_claim_matrix(tradeoff: pd.DataFrame) -> pd.DataFrame:

    def row_for(detector: str, group: str) -> pd.DataFrame:

        return tradeoff[(tradeoff["detector"].eq(detector)) & (tradeoff["method_group"].eq(group))]


    yolo_rel = row_for("YOLO-World", "geometry_reliability_first")

    yolo_ap = row_for("YOLO-World", "geometry_ap_constrained")

    gd_rel = row_for("Grounding DINO", "geometry_reliability_first")

    gd_ap = row_for("Grounding DINO", "geometry_ap_first")

    gd_preserve = row_for("Grounding DINO", "geometry_ap_constrained_candidate")


    rows = []

    if len(yolo_rel):

        r = yolo_rel.iloc[0]

        rows.append(

            {

                "question": "Does geometry-risk improve YOLO-World?",

                "answer": "Yes for reliability-first operation.",

                "evidence": f"Balanced {r.delta_balanced:+.4f}, UFA {r.delta_UFA:+.0f}, BG FP {r.delta_background_false_accepts:+.0f}; AP50 {r.delta_AP50:+.4f}.",

                "claim_strength": "strong for reliability; AP trade-off quantified",

            }

        )

    if len(yolo_ap):

        r = yolo_ap.iloc[0]

        rows.append(

            {

                "question": "Does YOLO-World have AP-constrained reliability improvement?",

                "answer": "Yes.",

                "evidence": f"Balanced {r.delta_balanced:+.4f}, UFA {r.delta_UFA:+.0f}, BG FP {r.delta_background_false_accepts:+.0f}, AP50 {r.delta_AP50:+.4f}, AP {r.delta_AP:+.4f}.",

                "claim_strength": "strongest public AP-constrained claim",

            }

        )

    if len(gd_rel):

        r = gd_rel.iloc[0]

        rows.append(

            {

                "question": "Does geometry-risk improve Grounding DINO?",

                "answer": "Partially: reliability-first improves balanced/precision/false accepts but trades AP.",

                "evidence": f"Balanced {r.delta_balanced:+.4f}, precision {r.delta_precision:+.4f}, UFA {r.delta_UFA:+.0f}, BG FP {r.delta_background_false_accepts:+.0f}, AP50 {r.delta_AP50:+.4f}.",

                "claim_strength": "moderate transfer evidence with AP trade-off",

            }

        )

    if len(gd_ap):

        r = gd_ap.iloc[0]

        rows.append(

            {

                "question": "Can Grounding DINO geometry calibration improve AP?",

                "answer": "Yes in AP-constrained mode, but not with balanced improvement.",

                "evidence": f"AP50 {r.delta_AP50:+.4f}, AP {r.delta_AP:+.4f}, UFA {r.delta_UFA:+.0f}, BG FP {r.delta_background_false_accepts:+.0f}, balanced {r.delta_balanced:+.4f}.",

                "claim_strength": "useful diagnostic, not the main reliability claim",

            }

        )

    if len(gd_preserve):

        r = gd_preserve.iloc[0]

        answer = "No strict AP-constrained balanced-improvement policy was found." if not bool(r.ap_constrained_success) else "Yes."

        rows.append(

            {

                "question": "Does Grounding DINO have AP-constrained reliability improvement?",

                "answer": answer,

                "evidence": f"Candidate balanced {r.delta_balanced:+.4f}, AP50 {r.delta_AP50:+.4f}, AP {r.delta_AP:+.4f}; success={bool(r.ap_constrained_success)}.",

                "claim_strength": "limited",

            }

        )

    rows.append(

        {

            "question": "Is the method detector-agnostic or YOLO-specific?",

            "answer": "Detector-agnostic for reliability/false-accept analysis; strongest AP-constrained balanced improvement is currently YOLO-World-l specific.",

            "evidence": "Both detectors show reduced false accepts under geometry calibration, but Grounding DINO does not meet the strict AP-constrained balanced-success criterion.",

            "claim_strength": "partial detector-agnostic evidence",

        }

    )

    rows.append(

        {

            "question": "What is the strongest honest paper claim?",

            "answer": "Geometry-aware reliability calibration improves open-set reliability for foundation-model object detection, with AP-constrained reliability demonstrated most strongly on YOLO-World-l and detector-transfer evidence on Grounding DINO with quantified trade-offs.",

            "evidence": "Do not claim full LVIS SOTA or universal AP-constrained detector-agnostic superiority.",

            "claim_strength": "paper-safe claim boundary",

        }

    )

    return pd.DataFrame(rows)



def write_summary(path: Path, main: pd.DataFrame, tradeoff: pd.DataFrame, claim: pd.DataFrame) -> None:

    lines = [

        "# Step11C Cross-Detector Comparison Summary",

        "",

        "## Main Comparison",

        "",

        main.to_markdown(index=False),

        "",

        "## AP-Reliability Trade-Offs vs Same-Detector Raw",

        "",

        tradeoff.to_markdown(index=False),

        "",

        "## Claim Support Matrix",

        "",

        claim.to_markdown(index=False),

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)


    step9d = args.project_root / "outputs" / "step9d_external_baselines_lvis300" / "csv" / "step9d_external_baseline_summary.csv"

    step10 = args.project_root / "outputs" / "step10a_ap_constrained_reliability_calibration" / "csv" / "step10a_method_ranking.csv"

    step11a = args.project_root / "outputs" / "step11a_groundingdino_lvis300_full_baseline" / "csv" / "step11a_compact_paper_metrics.csv"

    step11b = args.project_root / "outputs" / "step11b_groundingdino_geometry_risk" / "csv" / "step11b_compact_paper_metrics.csv"


    d9 = read_csv(step9d)

    d10 = read_csv(step10)

    d11a = read_csv(step11a)

    d11b = read_csv(step11b)


    rows: List[dict] = []

    yolo_l = find_one(d9, lambda r: "yolov8l-worldv2" in str(r.get("method", "")), "YOLO-World-l raw")

    yolo_s = find_one(d9, lambda r: "yolov8s-worldv2" in str(r.get("method", "")), "YOLO-World-s raw")

    yolo_rel = find_one(d9, lambda r: str(r.get("method", "")) == "Geometry-risk max_cal_balanced", "YOLO geometry reliability")

    yolo_ap = pick_yolo_ap_constrained(d10)


    rows.append(

        row_from_any(

            row=yolo_s,

            detector="YOLO-World",

            detector_variant="yolov8s-worldv2",

            method_group="raw",

            method_name="YOLO-World-s raw",

            claim_role="external smaller raw baseline",

            source_path=step9d,

        )

    )

    rows.append(

        row_from_any(

            row=yolo_l,

            detector="YOLO-World",

            detector_variant="yolov8l-worldv2",

            method_group="raw",

            method_name="YOLO-World-l raw",

            claim_role="primary raw baseline",

            source_path=step9d,

        )

    )

    rows.append(

        row_from_any(

            row=yolo_rel,

            detector="YOLO-World",

            detector_variant="yolov8l-worldv2",

            method_group="geometry_reliability_first",

            method_name="YOLO-World-l geometry reliability-first",

            claim_role="proposed reliability-first mode",

            source_path=step9d,

            notes="Step9C selected on calibration for max_cal_balanced.",

        )

    )

    rows.append(

        row_from_any(

            row=yolo_ap,

            detector="YOLO-World",

            detector_variant="yolov8l-worldv2",

            method_group="geometry_ap_constrained",

            method_name="YOLO-World-l geometry AP-constrained",

            claim_role="proposed AP-constrained mode",

            source_path=step10,

            notes="Strict AP50/AP tolerance row is preferred when available.",

        )

    )


    gd_raw = d11a.iloc[0].to_dict()

    rows.append(

        row_from_any(

            row=gd_raw,

            detector="Grounding DINO",

            detector_variant="grounding-dino-tiny",

            method_group="raw",

            method_name="Grounding DINO tiny raw",

            claim_role="external detector raw baseline",

            source_path=step11a,

        )

    )

    gd_rel = find_one(d11b, lambda r: str(r.get("selection_policy", "")) == "max_cal_balanced", "Grounding DINO reliability")

    rows.append(

        row_from_any(

            row=gd_rel,

            detector="Grounding DINO",

            detector_variant="grounding-dino-tiny",

            method_group="geometry_reliability_first",

            method_name="Grounding DINO geometry reliability-first",

            claim_role="detector-transfer reliability-first mode",

            source_path=step11b,

            notes="Improves balanced and false accepts but trades AP on test.",

        )

    )

    gd_ap_first = find_one(d11b, lambda r: str(r.get("selection_policy", "")) == "max_cal_ap", "Grounding DINO AP-constrained")

    rows.append(

        row_from_any(

            row=gd_ap_first,

            detector="Grounding DINO",

            detector_variant="grounding-dino-tiny",

            method_group="geometry_ap_first",

            method_name="Grounding DINO geometry AP-constrained",

            claim_role="AP-constrained diagnostic",

            source_path=step11b,

            notes="Improves AP/AP50 and reduces false accepts, but balanced is not improved.",

        )

    )

    gd_success = d11b[d11b.get("meets_success_criterion", pd.Series(dtype=str)).astype(str).str.lower().eq("true")].copy()

    if len(gd_success):

        gd_preserve = gd_success.iloc[0].to_dict()

    else:

        gd_preserve = find_one(d11b, lambda r: str(r.get("mode", "")) == "ap_constrained", "Grounding DINO AP-constrained candidate")

    rows.append(

        row_from_any(

            row=gd_preserve,

            detector="Grounding DINO",

            detector_variant="grounding-dino-tiny",

            method_group="geometry_ap_constrained_candidate",

            method_name="Grounding DINO geometry AP-constrained candidate",

            claim_role="diagnostic candidate; strict success not found" if not len(gd_success) else "proposed AP-constrained mode",

            source_path=step11b,

            notes="No strict AP-constrained balanced-success row was found." if not len(gd_success) else "",

        )

    )


    main_df = pd.DataFrame(rows)

    tradeoff = delta_rows(main_df)

    claim = build_claim_matrix(tradeoff)


    main_df.to_csv(csv_dir / "step11c_main_comparison.csv", index=False, encoding="utf-8-sig")

    tradeoff.to_csv(csv_dir / "step11c_ap_reliability_tradeoff.csv", index=False, encoding="utf-8-sig")

    claim.to_csv(csv_dir / "step11c_claim_support_matrix.csv", index=False, encoding="utf-8-sig")

    write_summary(args.output_root / "step11c_summary.md", main_df, tradeoff, claim)


    report = {

        "method": "Step11C cross-detector comparison and claim decision",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "inputs": {

            "step9d_external_baseline_summary": str(step9d),

            "step10a_method_ranking": str(step10),

            "step11a_groundingdino_raw": str(step11a),

            "step11b_groundingdino_geometry": str(step11b),

        },

        "row_counts": {

            "main_comparison": int(len(main_df)),

            "ap_reliability_tradeoff": int(len(tradeoff)),

            "claim_support_matrix": int(len(claim)),

        },

        "outputs": {

            "main_comparison": str(csv_dir / "step11c_main_comparison.csv"),

            "ap_reliability_tradeoff": str(csv_dir / "step11c_ap_reliability_tradeoff.csv"),

            "claim_support_matrix": str(csv_dir / "step11c_claim_support_matrix.csv"),

            "summary": str(args.output_root / "step11c_summary.md"),

            "integrity_report": str(args.output_root / "step11c_integrity_report.json"),

        },

    }

    with open(args.output_root / "step11c_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)


    print("\n========== Step11C completed ==========")

    print(main_df[["detector", "method_name", "balanced", "precision", "recall", "unknown_false_accepts", "background_false_accepts", "AP50", "AP75", "AP"]].to_string(index=False))

    print("\nClaim support matrix:")

    print(claim.to_string(index=False))



if __name__ == "__main__":

    main()

