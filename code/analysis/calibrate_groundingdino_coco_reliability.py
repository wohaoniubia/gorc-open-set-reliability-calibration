from __future__ import annotations


import json

import math

import sys

from datetime import datetime

from pathlib import Path

from typing import Iterable


import numpy as np

import pandas as pd



ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT / "GORC_paper_release_bundle"

OUT = ROOT / "outputs" / "step11b_groundingdino_coco_geometry_risk"

CSV_DIR = OUT / "csv"

ROUND2_OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "06_cross_detector"


CORE_CODE = BUNDLE / "code" / "core_evaluators"

LVIS_CODE = BUNDLE / "code" / "lvis_step9_step10"

for p in [str(CORE_CODE), str(LVIS_CODE)]:

    if p not in sys.path:

        sys.path.insert(0, p)


from step8j_yoloworld_lvis_openvoc_baseline import (

    clean_columns,

    compute_ap_summary,

    evaluate_detections,

    load_annotations,

    load_known_classes,

    load_split,

    safe_class_name,

)



SPLIT_CSV = ROOT / "outputs" / "step12a_coco_val_openset_protocol" / "csv" / "step12a_scene_split.csv"

ANNOTATION_DIR = ROOT / "outputs" / "step12a_coco_val_openset_protocol" / "annotations"

CLASS_MAP_CSV = ROOT / "outputs" / "step12a_coco_val_openset_protocol" / "csv" / "step12a_class_map.csv"

STEP11A_SELECTED = ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "csv" / "step11a_selected_calibration_config.csv"

STEP11A_COMPACT = ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "csv" / "step11a_compact_paper_metrics.csv"

SCORED_CSV = CSV_DIR / "step11b_geometry_scored_candidates.csv"


SCORE_SPECS = [

    ("raw_score", "raw", "raw Grounding DINO score"),

    ("geometry_risk_score", "geometry_probability", "calibration-only geometry reliability probability"),

    ("geometry_alpha_0p200", "raw_geometry_alpha_blend", "(1-alpha)*raw + alpha*geometry; alpha=0.2"),

    ("geometry_alpha_0p400", "raw_geometry_alpha_blend", "(1-alpha)*raw + alpha*geometry; alpha=0.4"),

    ("geometry_alpha_0p500", "raw_geometry_alpha_blend", "(1-alpha)*raw + alpha*geometry; alpha=0.5"),

    ("geometry_alpha_0p600", "raw_geometry_alpha_blend", "(1-alpha)*raw + alpha*geometry; alpha=0.6"),

    ("geometry_alpha_0p800", "raw_geometry_alpha_blend", "(1-alpha)*raw + alpha*geometry; alpha=0.8"),

    ("geometry_alpha_1p000", "raw_geometry_alpha_blend", "(1-alpha)*raw + alpha*geometry; alpha=1.0"),

    ("perclass_geometry_gamma_0p500", "per_class_geometry_scaled_raw", "raw * (class_reliability*geometry)^gamma; gamma=0.5"),

    ("perclass_geometry_gamma_1p000", "per_class_geometry_scaled_raw", "raw * (class_reliability*geometry)^gamma; gamma=1.0"),

]


BASE_THRESHOLDS = [

    0.01,

    0.02,

    0.03,

    0.05,

    0.08,

    0.10,

    0.15,

    0.20,

    0.25,

    0.30,

    0.35,

    0.40,

    0.45,

    0.50,

    0.55,

    0.60,

    0.65,

    0.70,

    0.75,

    0.80,

    0.85,

    0.90,

    0.95,

]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def finite(value, default: float = float("nan")) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def harmonic3(p: float, r: float, u: float) -> float:

    vals = [float(p), float(r), float(u)]

    if any(v <= 0 for v in vals):

        return 0.0

    return float(3.0 / sum(1.0 / v for v in vals))



def thresholds(values: pd.Series) -> list[float]:

    vals = set(BASE_THRESHOLDS)

    clean = pd.to_numeric(values, errors="coerce").dropna()

    if len(clean):

        for q in np.linspace(0.05, 0.95, 19):

            vals.add(float(clean.quantile(float(q))))

        vals.add(float(clean.max()))

    return sorted(v for v in vals if math.isfinite(v) and v >= 0.0)



def fast_proxy_grid(cal: pd.DataFrame, score_col: str, known_gt_count: int, unknown_gt_count: int) -> pd.DataFrame:

    score = pd.to_numeric(cal[score_col], errors="coerce").fillna(-np.inf).to_numpy()

    tp = pd.to_numeric(cal["risk_label_tp"], errors="coerce").fillna(0).to_numpy().astype(int)

    err = cal["risk_error_type"].astype(str).to_numpy()

    rows: list[dict] = []

    for thr in thresholds(cal[score_col]):

        mask = score >= float(thr)

        accepted = int(mask.sum())

        known_tp = int(tp[mask].sum())

        known_fp = int(accepted - known_tp)

        ufa_det = int((err[mask] == "unknown_false_accept").sum())

        bg_det = int((err[mask] == "background_false_accept").sum())

        p = known_tp / accepted if accepted else 0.0

        r = known_tp / known_gt_count if known_gt_count else 0.0

        urr_proxy = 1.0 - min(float(ufa_det), float(unknown_gt_count)) / float(unknown_gt_count) if unknown_gt_count else 1.0

        rows.append(

            {

                "score_col": score_col,

                "threshold": float(thr),

                "proxy_accepted": accepted,

                "proxy_tp": known_tp,

                "proxy_known_fp": known_fp,

                "proxy_unknown_false_accept_detections": ufa_det,

                "proxy_background_false_accept_detections": bg_det,

                "proxy_precision": p,

                "proxy_recall": r,

                "proxy_unknown_reject": urr_proxy,

                "proxy_balanced": harmonic3(p, r, urr_proxy),

            }

        )

    return pd.DataFrame(rows)



def exact_operating(scored_split: pd.DataFrame, gt_split: pd.DataFrame, score_col: str, threshold: float) -> dict:

    dets = scored_split[pd.to_numeric(scored_split[score_col], errors="coerce") >= float(threshold)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    metrics, _ = evaluate_detections(dets, gt_split, iou_thr=0.50)

    return metrics



def ap_for(scored: pd.DataFrame, gt_split: pd.DataFrame, known_classes: list[str], split: str, score_col: str) -> dict:

    dets = scored[scored["split"].astype(str).str.lower().eq(split)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    ap, _ = compute_ap_summary(dets, gt_split, known_classes)

    return ap.iloc[0].to_dict()



def metric_row(metrics: dict, prefix: str = "") -> dict:

    return {

        f"{prefix}num_accepted_detections": int(metrics.get("num_accepted_detections", 0)),

        f"{prefix}tp_known": int(metrics.get("tp_known", 0)),

        f"{prefix}fp_known": int(metrics.get("fp_known", 0)),

        f"{prefix}fn_known": int(metrics.get("fn_known", 0)),

        f"{prefix}known_precision": finite(metrics.get("known_precision", 0.0), 0.0),

        f"{prefix}known_recall": finite(metrics.get("known_recall", 0.0), 0.0),

        f"{prefix}unknown_false_accept_objects": int(metrics.get("unknown_false_accept_objects", 0)),

        f"{prefix}unknown_reject_rate_object_level": finite(metrics.get("unknown_reject_rate_object_level", 0.0), 0.0),

        f"{prefix}precision_recall_unknown_balanced_score": finite(metrics.get("precision_recall_unknown_balanced_score", 0.0), 0.0),

        f"{prefix}background_false_accept_count": int(metrics.get("background_false_accept_count", 0)),

    }



def choose_exact_candidates(

    scored: pd.DataFrame,

    gt_cal: pd.DataFrame,

    known_classes: list[str],

    raw_cal_ap: dict,

) -> tuple[pd.DataFrame, pd.DataFrame]:

    cal = scored[scored["split"].astype(str).str.lower().eq("calibration")].copy()

    known_gt_count = int(gt_cal["gt_is_known"].sum())

    unknown_gt_count = int((~gt_cal["gt_is_known"]).sum())

    proxy_frames = []

    exact_rows: list[dict] = []

    for score_col, family, desc in SCORE_SPECS:

        if score_col not in cal.columns:

            continue

        proxy = fast_proxy_grid(cal, score_col, known_gt_count, unknown_gt_count)

        proxy["score_family"] = family

        proxy["description"] = desc

        proxy_frames.append(proxy)

        top = proxy.sort_values(

            ["proxy_balanced", "proxy_recall", "proxy_unknown_reject", "proxy_precision"],

            ascending=[False, False, False, False],

        ).head(6)

        for _, r in top.iterrows():

            exact = exact_operating(cal, gt_cal, score_col, float(r["threshold"]))

            exact_rows.append(

                {

                    "score_col": score_col,

                    "score_family": family,

                    "description": desc,

                    "threshold": float(r["threshold"]),

                    **r.to_dict(),

                    **metric_row(exact, prefix="cal_"),

                }

            )

    exact_grid = pd.DataFrame(exact_rows).drop_duplicates(subset=["score_col", "threshold"]).reset_index(drop=True)

    cal_ap_cache = {}

    for score_col in exact_grid["score_col"].drop_duplicates().tolist():

        cal_ap_cache[score_col] = ap_for(scored, gt_cal, known_classes, "calibration", score_col)

    for metric in ["AP50", "AP75", "AP"]:

        exact_grid[f"cal_{metric}"] = exact_grid["score_col"].map(lambda c: finite(cal_ap_cache.get(c, {}).get(metric, float("nan"))))


    raw_best = exact_grid[exact_grid["score_col"].eq("raw_score")].sort_values(

        ["cal_precision_recall_unknown_balanced_score", "cal_known_recall", "cal_unknown_reject_rate_object_level", "cal_known_precision"],

        ascending=[False, False, False, False],

    ).iloc[0]

    raw_bal = finite(raw_best["cal_precision_recall_unknown_balanced_score"], 0.0)

    raw_ap50 = finite(raw_cal_ap.get("AP50", float("nan")))

    raw_ap = finite(raw_cal_ap.get("AP", float("nan")))


    selected: list[pd.Series] = []

    by_bal = exact_grid.sort_values(

        ["cal_precision_recall_unknown_balanced_score", "cal_known_recall", "cal_unknown_reject_rate_object_level", "cal_known_precision", "cal_AP50", "cal_AP"],

        ascending=[False, False, False, False, False, False],

    ).iloc[0].copy()

    by_bal["selection_policy"] = "max_cal_balanced_constrained_grid"

    by_bal["mode"] = "reliability_first"

    by_bal["status"] = "selected_on_calibration"

    selected.append(by_bal)


    by_ap = exact_grid.sort_values(

        ["cal_AP", "cal_AP50", "cal_precision_recall_unknown_balanced_score", "cal_known_precision"],

        ascending=[False, False, False, False],

    ).iloc[0].copy()

    by_ap["selection_policy"] = "max_cal_ap_constrained_grid"

    by_ap["mode"] = "ap_first"

    by_ap["status"] = "selected_on_calibration"

    selected.append(by_ap)


    preserving = exact_grid[

        (exact_grid["cal_AP50"] >= raw_ap50 - 0.005)

        & (exact_grid["cal_AP"] >= raw_ap - 0.005)

        & (exact_grid["cal_precision_recall_unknown_balanced_score"] > raw_bal)

    ].copy()

    if len(preserving):

        by_preserve = preserving.sort_values(

            ["cal_precision_recall_unknown_balanced_score", "cal_AP50", "cal_AP", "cal_unknown_reject_rate_object_level"],

            ascending=[False, False, False, False],

        ).iloc[0].copy()

        by_preserve["status"] = "selected_on_calibration"

    else:

        by_preserve = exact_grid.assign(

            ap_gap=(raw_ap - exact_grid["cal_AP"]).abs(),

            ap50_gap=(raw_ap50 - exact_grid["cal_AP50"]).abs(),

            balanced_gap=(exact_grid["cal_precision_recall_unknown_balanced_score"] - raw_bal),

        ).sort_values(

            ["ap_gap", "ap50_gap", "balanced_gap", "cal_precision_recall_unknown_balanced_score"],

            ascending=[True, True, False, False],

        ).iloc[0].copy()

        by_preserve["status"] = "no_strict_ap_constrained_balanced_success_in_constrained_grid"

    by_preserve["selection_policy"] = "ap_constrained_candidate_constrained_grid"

    by_preserve["mode"] = "ap_constrained"

    selected.append(by_preserve)


    selected_df = pd.DataFrame(selected).drop_duplicates(subset=["selection_policy", "score_col", "threshold"])

    proxy_grid = pd.concat(proxy_frames, ignore_index=True) if proxy_frames else pd.DataFrame()

    return exact_grid, selected_df



def evaluate_selected(

    scored: pd.DataFrame,

    gt_test: pd.DataFrame,

    known_classes: list[str],

    selected: pd.DataFrame,

    raw_threshold: float,

    raw_ap: dict,

) -> pd.DataFrame:

    test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    raw_metrics = exact_operating(test, gt_test, "raw_score", raw_threshold)

    raw_row = {

        "method": "step11a_groundingdino_tiny_raw",

        "selection_policy": "step11a_selected_threshold",

        "mode": "raw",

        "score_col": "raw_score",

        "score_family": "raw",

        "description": "Step11A raw Grounding DINO threshold reproduced on Step11B candidate table",

        "threshold": float(raw_threshold),

        **metric_row(raw_metrics),

        "AP50": finite(raw_ap.get("AP50", float("nan"))),

        "AP75": finite(raw_ap.get("AP75", float("nan"))),

        "AP": finite(raw_ap.get("AP", float("nan"))),

    }


    ap_cache: dict[str, dict] = {"raw_score": raw_ap}

    rows = [raw_row]

    for _, r in selected.iterrows():

        if str(r.get("status", "")).startswith("no_strict"):


            pass

        score_col = str(r["score_col"])

        thr = float(r["threshold"])

        metrics = exact_operating(test, gt_test, score_col, thr)

        if score_col not in ap_cache:

            ap_cache[score_col] = ap_for(scored, gt_test, known_classes, "test", score_col)

        rows.append(

            {

                "method": "step11b_groundingdino_coco_" + str(r["selection_policy"]),

                "selection_policy": str(r["selection_policy"]),

                "mode": str(r["mode"]),

                "score_col": score_col,

                "score_family": str(r.get("score_family", "")),

                "description": str(r.get("description", "")),

                "threshold": thr,

                "selection_status": str(r.get("status", "")),

                **metric_row(metrics),

                "AP50": finite(ap_cache[score_col].get("AP50", float("nan"))),

                "AP75": finite(ap_cache[score_col].get("AP75", float("nan"))),

                "AP": finite(ap_cache[score_col].get("AP", float("nan"))),

            }

        )

    out = pd.DataFrame(rows)

    raw_b = finite(out.iloc[0]["precision_recall_unknown_balanced_score"], 0.0)

    raw_p = finite(out.iloc[0]["known_precision"], 0.0)

    raw_r = finite(out.iloc[0]["known_recall"], 0.0)

    raw_ufa = int(out.iloc[0]["unknown_false_accept_objects"])

    raw_bg = int(out.iloc[0]["background_false_accept_count"])

    raw_ap50 = finite(out.iloc[0]["AP50"], float("nan"))

    raw_apv = finite(out.iloc[0]["AP"], float("nan"))

    out["delta_balanced_vs_raw"] = out["precision_recall_unknown_balanced_score"].map(lambda x: finite(x, 0.0) - raw_b)

    out["delta_precision_vs_raw"] = out["known_precision"].map(lambda x: finite(x, 0.0) - raw_p)

    out["delta_recall_vs_raw"] = out["known_recall"].map(lambda x: finite(x, 0.0) - raw_r)

    out["delta_ufa_vs_raw"] = out["unknown_false_accept_objects"].map(lambda x: int(x) - raw_ufa)

    out["delta_bg_fp_vs_raw"] = out["background_false_accept_count"].map(lambda x: int(x) - raw_bg)

    out["delta_AP50_vs_raw"] = out["AP50"].map(lambda x: finite(x, float("nan")) - raw_ap50)

    out["delta_AP_vs_raw"] = out["AP"].map(lambda x: finite(x, float("nan")) - raw_apv)

    out["meets_AP50_minus_0p005"] = out["AP50"].map(lambda x: finite(x, -1.0) >= raw_ap50 - 0.005)

    out["meets_AP_minus_0p005"] = out["AP"].map(lambda x: finite(x, -1.0) >= raw_apv - 0.005)

    out["meets_success_criterion"] = (

        (out["delta_balanced_vs_raw"] > 0)

        & (out["meets_AP50_minus_0p005"])

        & (out["meets_AP_minus_0p005"])

    )

    return out



def write_markdown(path: Path, title: str, df: pd.DataFrame, note: str = "") -> None:

    lines = [f"# {title}", ""]

    if note:

        lines.extend([note, ""])

    lines.append(df.to_markdown(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    ensure_dir(CSV_DIR)

    ensure_dir(ROUND2_OUT)

    if not SCORED_CSV.exists():

        raise FileNotFoundError(SCORED_CSV)

    known_classes = load_known_classes(CLASS_MAP_CSV)

    split_df = load_split(SPLIT_CSV, ROOT, split_filter="all")

    gt = load_annotations(ANNOTATION_DIR, split_df, known_classes)

    gt_cal = gt[gt["split"].astype(str).str.lower().eq("calibration")].copy()

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    scored = clean_columns(pd.read_csv(SCORED_CSV, encoding="utf-8-sig"))

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    for c in ["x1", "y1", "x2", "y2"] + [s[0] for s in SCORE_SPECS if s[0] in scored.columns]:

        scored[c] = pd.to_numeric(scored[c], errors="coerce")


    raw_thr = 0.4

    if STEP11A_SELECTED.exists():

        raw_thr = finite(pd.read_csv(STEP11A_SELECTED).iloc[0].get("conf_thr", raw_thr), raw_thr)

    raw_test_ap = {}

    if STEP11A_COMPACT.exists():

        raw_first = pd.read_csv(STEP11A_COMPACT).iloc[0].to_dict()

        raw_test_ap = {"AP50": raw_first.get("AP50"), "AP75": raw_first.get("AP75"), "AP": raw_first.get("AP")}

    if not raw_test_ap:

        raw_test_ap = ap_for(scored, gt_test, known_classes, "test", "raw_score")

    raw_cal_ap = ap_for(scored, gt_cal, known_classes, "calibration", "raw_score")


    exact_grid, selected = choose_exact_candidates(scored, gt_cal, known_classes, raw_cal_ap)

    compact = evaluate_selected(scored, gt_test, known_classes, selected, raw_thr, raw_test_ap)

    ranking = compact.sort_values(

        [

            "meets_success_criterion",

            "precision_recall_unknown_balanced_score",

            "known_precision",

            "known_recall",

            "AP50",

            "AP",

            "unknown_false_accept_objects",

            "background_false_accept_count",

        ],

        ascending=[False, False, False, False, False, False, True, True],

    ).reset_index(drop=True)

    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))


    alpha_summary = exact_grid.sort_values(

        ["score_col", "cal_precision_recall_unknown_balanced_score", "cal_unknown_reject_rate_object_level", "cal_known_precision", "cal_known_recall"],

        ascending=[True, False, False, False, False],

    ).groupby("score_col", as_index=False).head(1).reset_index(drop=True)


    exact_grid.to_csv(CSV_DIR / "step11b_policy_grid.csv", index=False, encoding="utf-8-sig")

    alpha_summary.to_csv(CSV_DIR / "step11b_alpha_sweep_summary.csv", index=False, encoding="utf-8-sig")

    selected.to_csv(CSV_DIR / "step11b_selected_policies.csv", index=False, encoding="utf-8-sig")

    compact.to_csv(CSV_DIR / "step11b_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")

    ranking.to_csv(CSV_DIR / "step11b_method_ranking.csv", index=False, encoding="utf-8-sig")


    report = {

        "method": "Round2 constrained finish for Grounding DINO COCO Step11B",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "raw_pred_source": str(ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "csv" / "step11a_raw_predictions.csv"),

        "scored_candidate_source": str(SCORED_CSV),

        "selection_scope": "calibration split only",

        "test_scope": "final selected-policy evaluation only",

        "note": "This script reuses the scored-candidate table, evaluates a constrained calibration grid with the same evaluator for exact selected rows, and writes the selected-row summary files.",

        "row_counts": {

            "scored_candidates": int(len(scored)),

            "calibration_candidates": int((scored["split"].astype(str).str.lower() == "calibration").sum()),

            "test_candidates": int((scored["split"].astype(str).str.lower() == "test").sum()),

            "policy_grid_exact_rows": int(len(exact_grid)),

            "selected_policies": int(len(selected)),

            "compact_paper_metrics": int(len(compact)),

        },

        "outputs": {

            "policy_grid": str(CSV_DIR / "step11b_policy_grid.csv"),

            "alpha_sweep_summary": str(CSV_DIR / "step11b_alpha_sweep_summary.csv"),

            "selected_policies": str(CSV_DIR / "step11b_selected_policies.csv"),

            "compact_paper_metrics": str(CSV_DIR / "step11b_compact_paper_metrics.csv"),

            "method_ranking": str(CSV_DIR / "step11b_method_ranking.csv"),

            "summary": str(OUT / "step11b_summary.md"),

            "integrity_report": str(OUT / "step11b_integrity_report.json"),

        },

    }

    (OUT / "step11b_integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


    lines = [

        "# Step11B Grounding DINO COCO Geometry-Risk Calibration Summary",

        "",

        report["note"],

        "",

        "## Selected Policies",

        "",

        selected.to_markdown(index=False),

        "",

        "## Held-Out Test Compact Metrics",

        "",

        compact.to_markdown(index=False),

    ]

    (OUT / "step11b_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


    cols = [

        "rank",

        "method",

        "mode",

        "score_col",

        "threshold",

        "precision_recall_unknown_balanced_score",

        "known_precision",

        "known_recall",

        "unknown_false_accept_objects",

        "background_false_accept_count",

        "AP50",

        "AP75",

        "AP",

        "delta_balanced_vs_raw",

        "delta_ufa_vs_raw",

        "delta_bg_fp_vs_raw",

        "meets_success_criterion",

    ]

    print(ranking[[c for c in cols if c in ranking.columns]].to_string(index=False))



if __name__ == "__main__":

    main()

