


"""
Step10a: AP-constrained open-set reliability calibration.

The script searches transparent score families under calibration-only selection:

1. raw score and monotonic raw transforms;
2. raw/geometry alpha blends;
3. class-wise reliability-scaled raw scores, which preserve within-class ranking
   and therefore preserve per-class AP while changing the operating threshold
   behavior;
4. light geometry gates around raw score.

The policy grid is evaluated on the calibration split only. Test metrics are
computed only for fixed references and selected policies.
"""


from __future__ import annotations


import argparse

import json

import math

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Sequence, Tuple


import numpy as np

import pandas as pd

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import (

    clean_columns,

    compute_ap_summary,

    evaluate_detections,

    load_annotations,

    load_known_classes,

    load_split,

    normalize_image_id,

    safe_class_name,

)



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_SCORED_CSV = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300" / "csv" / "step9c_geometry_scored_candidates.csv"

DEFAULT_RAW_PRED_CSV = DEFAULT_PROJECT_ROOT / "outputs" / "step9b_yoloworld_lvis300_raw_baseline" / "csv" / "step9b_yoloworld_raw_predictions.csv"

DEFAULT_STEP9C_COMPACT = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300" / "csv" / "step9c_compact_paper_metrics.csv"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step10a_ap_constrained_reliability_calibration"


AP_DROP_TOLERANCES = [0.00, 0.005, 0.01, 0.02]

AP_IOUS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def finite_float(x, default: float = 0.0) -> float:

    try:

        v = float(x)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def safe_score_name(prefix: str, value: float) -> str:

    return f"{prefix}_{value:.3f}".replace(".", "p").replace("-", "m")



def harmonic3(a: float, b: float, c: float) -> float:

    vals = [float(a), float(b), float(c)]

    if any(v <= 0 for v in vals):

        return 0.0

    return float(3.0 / sum(1.0 / v for v in vals))



def parse_float_list(text: str) -> List[float]:

    return [float(x.strip()) for x in str(text).split(",") if x.strip()]



def normalize_scores(s: pd.Series) -> pd.Series:

    x = pd.to_numeric(s, errors="coerce").fillna(0.0).clip(lower=0.0, upper=1.0)

    return x.astype(float)



def class_reliability_weights(scored: pd.DataFrame) -> Dict[str, Dict[str, float]]:

    cal = scored[scored["split"].astype(str).str.lower().eq("calibration")].copy()

    labels = sorted(cal["pred_label"].map(safe_class_name).dropna().unique().tolist())

    raw_stats = {}

    for label in labels:

        sub = cal[cal["pred_label"].map(safe_class_name).eq(label)].copy()

        n = max(0, int(len(sub)))

        tp = int(pd.to_numeric(sub.get("risk_label_tp", 0), errors="coerce").fillna(0).sum())

        bg = int(sub["risk_error_type"].astype(str).eq("background_false_accept").sum()) if "risk_error_type" in sub else 0

        ufa = int(sub["risk_error_type"].astype(str).eq("unknown_false_accept").sum()) if "risk_error_type" in sub else 0

        precision = (tp + 1.0) / (n + 2.0) if n else 0.5

        bg_safety = (n - bg + 1.0) / (n + 2.0) if n else 0.5

        unknown_safety = (n - ufa + 1.0) / (n + 2.0) if n else 0.5

        combined = (precision * bg_safety * unknown_safety) ** (1.0 / 3.0)

        raw_stats[label] = {

            "precision": precision,

            "bg_safety": bg_safety,

            "unknown_safety": unknown_safety,

            "combined": combined,

            "n": n,

            "tp": tp,

            "bg": bg,

            "ufa": ufa,

        }

    out: Dict[str, Dict[str, float]] = {}

    for family in ["precision", "bg_safety", "unknown_safety", "combined"]:

        vals = [raw_stats[l][family] for l in labels]

        vmax = max(vals) if vals else 1.0

        out[family] = {}

        for label in labels:

            rel = raw_stats[label][family] / max(vmax, 1e-9)

            out[family][label] = float(np.clip(rel, 0.20, 1.0))

    out["_stats"] = raw_stats

    return out



def build_score_columns(scored: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict], dict]:

    df = scored.copy()

    df["image_id"] = df["image_id"].map(normalize_image_id)

    df["pred_label"] = df["pred_label"].map(safe_class_name)

    df["raw_score"] = normalize_scores(df["raw_score"] if "raw_score" in df.columns else df["score"])

    if "geometry_risk_score" in df.columns:

        df["geometry_score"] = normalize_scores(df["geometry_risk_score"])

    elif "geometry_alpha_1p000" in df.columns:

        df["geometry_score"] = normalize_scores(df["geometry_alpha_1p000"])

    else:

        df["geometry_score"] = df["raw_score"]


    specs: List[dict] = []


    def add_spec(col: str, family: str, description: str, ap_rank_preserving: bool, **kw) -> None:

        specs.append(

            {

                "score_col": col,

                "score_family": family,

                "description": description,

                "ap_rank_preserving_by_design": bool(ap_rank_preserving),

                **kw,

            }

        )


    add_spec("raw_score", "raw", "raw YOLO-World score", True)


    for power in [0.50, 1.50, 2.00]:

        col = safe_score_name("raw_power", power)

        df[col] = np.power(df["raw_score"].clip(lower=0.0, upper=1.0), power)

        add_spec(col, "raw_monotonic", f"global monotonic raw_score^{power:g}", True, power=power)


    for alpha in [0.00, 0.025, 0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]:

        col = safe_score_name("blend_alpha", alpha)

        if alpha == 0.0:

            df[col] = df["raw_score"]

        elif alpha == 1.0:

            df[col] = df["geometry_score"]

        else:

            df[col] = (1.0 - alpha) * df["raw_score"] + alpha * df["geometry_score"]

        add_spec(col, "raw_geometry_alpha_blend", f"(1-alpha)*raw + alpha*geometry; alpha={alpha:g}", alpha == 0.0, alpha=alpha)


    weights = class_reliability_weights(df)

    gammas = [0.25, 0.50, 0.75, 1.00]

    for family in ["precision", "bg_safety", "unknown_safety", "combined"]:

        wmap = weights[family]

        base_w = df["pred_label"].map(lambda x: float(wmap.get(safe_class_name(x), 1.0)))

        for gamma in gammas:

            col = safe_score_name(f"class_{family}_gamma", gamma)

            df[col] = df["raw_score"] * np.power(base_w.astype(float), gamma)

            add_spec(

                col,

                f"class_{family}_scaled_raw",

                f"class-wise {family} reliability scaling; gamma={gamma:g}",

                True,

                gamma=gamma,

                weight_family=family,

            )


    for beta in [0.05, 0.10, 0.20, 0.30]:

        col = safe_score_name("raw_light_geom_gate", beta)

        gate = (1.0 - beta) + beta * df["geometry_score"]

        df[col] = df["raw_score"] * gate.clip(lower=0.0, upper=1.0)

        add_spec(col, "raw_light_geometry_gate", f"raw * ((1-beta)+beta*geometry); beta={beta:g}", False, beta=beta)


    return df, specs, weights



def make_thresholds(values: pd.Series) -> List[float]:

    clean = pd.to_numeric(values, errors="coerce").dropna()

    fixed = {

        0.05,

        0.10,

        0.20,

        0.30,

        0.40,

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

    }

    vals = set(float(v) for v in fixed)

    if len(clean):

        for q in [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]:

            vals.add(float(clean.quantile(q)))

        vals.add(float(clean.max()))

    return sorted(v for v in vals if math.isfinite(v) and v >= 0.0)



def ap_for(scored: pd.DataFrame, gt_split: pd.DataFrame, known_classes: Sequence[str], score_col: str) -> dict:

    dets = scored.copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    ap, _ = compute_ap_summary(dets, gt_split, known_classes)

    row = ap.iloc[0].to_dict()

    return {k: finite_float(v, float("nan")) for k, v in row.items()}



def eval_operating(scored_split: pd.DataFrame, gt_split: pd.DataFrame, score_col: str, threshold: float, iou_thr: float) -> dict:

    dets = scored_split[pd.to_numeric(scored_split[score_col], errors="coerce") >= float(threshold)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    metrics, _ = evaluate_detections(dets, gt_split, iou_thr=float(iou_thr))

    return metrics



def process_policy_grid(

    *,

    scored: pd.DataFrame,

    specs: List[dict],

    gt_cal: pd.DataFrame,

    known_classes: Sequence[str],

    grid_csv: Path,

    max_score_cols_per_run: int,

    iou_thr: float,

) -> pd.DataFrame:

    existing = read_csv(grid_csv) if grid_csv.exists() else pd.DataFrame()

    done = set(existing["score_col"].astype(str).tolist()) if len(existing) and "score_col" in existing.columns else set()

    todo = [s for s in specs if s["score_col"] not in done]

    if max_score_cols_per_run and max_score_cols_per_run > 0:

        todo = todo[: int(max_score_cols_per_run)]


    cal = scored[scored["split"].astype(str).str.lower().eq("calibration")].copy()

    new_rows: List[dict] = []

    for spec in tqdm(todo, desc="Step10a calibration policy grid"):

        score_col = spec["score_col"]

        ap = ap_for(cal, gt_cal, known_classes, score_col)

        thresholds = make_thresholds(cal[score_col])

        for thr in thresholds:

            m = eval_operating(cal, gt_cal, score_col, thr, iou_thr)

            row = {

                **spec,

                "split_used_for_selection": "calibration",

                "threshold": float(thr),

                "iou_threshold": float(iou_thr),

                "cal_AP50": ap.get("AP50", float("nan")),

                "cal_AP75": ap.get("AP75", float("nan")),

                "cal_AP": ap.get("AP", float("nan")),

                "cal_balanced": m["precision_recall_unknown_balanced_score"],

                "cal_precision": m["known_precision"],

                "cal_recall": m["known_recall"],

                "cal_unknown_reject": m["unknown_reject_rate_object_level"],

                "cal_ufa": m["unknown_false_accept_objects"],

                "cal_background_false_accepts": m["background_false_accept_count"],

                "cal_num_accepted": m["num_accepted_detections"],

            }

            new_rows.append(row)

        if new_rows:

            merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if len(existing) else pd.DataFrame(new_rows)

            merged = merged.drop_duplicates(subset=["score_col", "threshold"], keep="last")

            merged.to_csv(grid_csv, index=False, encoding="utf-8-sig")

            existing = merged

            new_rows = []

    return existing



def select_policies(grid: pd.DataFrame, raw_cal_ap50: float, raw_cal_ap: float) -> pd.DataFrame:

    rows = []

    objectives = {

        "balanced_max": (

            ["cal_balanced", "cal_unknown_reject", "cal_precision", "cal_recall", "cal_AP50", "cal_AP", "cal_num_accepted"],

            [False, False, False, False, False, False, True],

        ),

        "ufa_min": (

            ["cal_ufa", "cal_balanced", "cal_precision", "cal_recall", "cal_background_false_accepts"],

            [True, False, False, False, True],

        ),

        "background_fp_min": (

            ["cal_background_false_accepts", "cal_balanced", "cal_precision", "cal_recall", "cal_ufa"],

            [True, False, False, False, True],

        ),

        "precision_max": (

            ["cal_precision", "cal_balanced", "cal_unknown_reject", "cal_recall", "cal_ufa"],

            [False, False, False, False, True],

        ),

    }


    for ap50_tol in AP_DROP_TOLERANCES:

        for ap_tol in AP_DROP_TOLERANCES:

            eligible = grid[

                (pd.to_numeric(grid["cal_AP50"], errors="coerce") >= raw_cal_ap50 - ap50_tol - 1e-12)

                & (pd.to_numeric(grid["cal_AP"], errors="coerce") >= raw_cal_ap - ap_tol - 1e-12)

            ].copy()

            if eligible.empty:

                rows.append(

                    {

                        "constraint_name": f"AP50tol{ap50_tol:g}_APtol{ap_tol:g}",

                        "AP50_drop_tolerance": ap50_tol,

                        "AP_drop_tolerance": ap_tol,

                        "objective": "none",

                        "status": "no_calibration_candidate",

                    }

                )

                continue

            for obj, (cols, asc) in objectives.items():

                ranked = eligible.sort_values(cols, ascending=asc).reset_index(drop=True)

                best = ranked.iloc[0].to_dict()

                best.update(

                    {

                        "constraint_name": f"AP50tol{ap50_tol:g}_APtol{ap_tol:g}",

                        "AP50_drop_tolerance": ap50_tol,

                        "AP_drop_tolerance": ap_tol,

                        "objective": obj,

                        "status": "selected_on_calibration",

                    }

                )

                rows.append(best)


    selected = pd.DataFrame(rows)

    keep_cols = [

        "constraint_name",

        "AP50_drop_tolerance",

        "AP_drop_tolerance",

        "objective",

        "status",

        "score_col",

        "score_family",

        "description",

        "ap_rank_preserving_by_design",

        "threshold",

        "iou_threshold",

        "cal_AP50",

        "cal_AP75",

        "cal_AP",

        "cal_balanced",

        "cal_precision",

        "cal_recall",

        "cal_unknown_reject",

        "cal_ufa",

        "cal_background_false_accepts",

        "cal_num_accepted",

    ]

    existing_cols = [c for c in keep_cols if c in selected.columns]

    selected = selected[existing_cols].drop_duplicates(

        subset=["score_col", "threshold", "objective", "AP50_drop_tolerance", "AP_drop_tolerance"],

        keep="first",

    )

    return selected



def reference_specs(step9c_compact: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for _, r in step9c_compact.iterrows():

        method = str(r.get("method", ""))

        if method in {

            "step9b_raw_global_reproduced",

            "step9c_geometry_max_cal_balanced",

            "step9c_geometry_max_cal_ap",

            "step9c_geometry_precision_tiebreak_eps0p005",

        }:

            rows.append(

                {

                    "method": method,

                    "selection_policy": str(r.get("selection_policy", "")),

                    "score_col": str(r.get("score_col", "")),

                    "threshold": finite_float(r.get("threshold", 0.0)),

                    "source": "step9_reference",

                }

            )

    return pd.DataFrame(rows)



def evaluate_selected_on_test(

    *,

    scored: pd.DataFrame,

    gt_test: pd.DataFrame,

    known_classes: Sequence[str],

    selected: pd.DataFrame,

    references: pd.DataFrame,

    iou_thr: float,

    raw_test_ap: dict,

    raw_test_operating: dict,

) -> pd.DataFrame:

    test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    rows = []


    candidates: List[dict] = []

    for _, r in references.iterrows():

        if str(r.get("score_col", "")) in scored.columns:

            candidates.append(

                {

                    "method": str(r["method"]),

                    "selection_policy": str(r["selection_policy"]),

                    "score_col": str(r["score_col"]),

                    "threshold": finite_float(r["threshold"]),

                    "source": "fixed_step9_reference",

                    "constraint_name": "",

                    "objective": "",

                    "AP50_drop_tolerance": "",

                    "AP_drop_tolerance": "",

                }

            )


    chosen = selected[selected["status"].astype(str).eq("selected_on_calibration")].copy()

    chosen = chosen.drop_duplicates(subset=["score_col", "threshold", "objective", "AP50_drop_tolerance", "AP_drop_tolerance"])

    for idx, r in chosen.iterrows():

        method = f"step10a_{r['objective']}_{str(r['constraint_name']).replace('.', 'p')}_{str(r['score_col'])}"

        candidates.append(

            {

                "method": method,

                "selection_policy": "ap_constrained_calibration_selected",

                "score_col": str(r["score_col"]),

                "threshold": finite_float(r["threshold"]),

                "source": "step10a_selected_policy",

                "constraint_name": str(r.get("constraint_name", "")),

                "objective": str(r.get("objective", "")),

                "AP50_drop_tolerance": r.get("AP50_drop_tolerance", ""),

                "AP_drop_tolerance": r.get("AP_drop_tolerance", ""),

                "cal_AP50": r.get("cal_AP50", ""),

                "cal_AP": r.get("cal_AP", ""),

                "cal_balanced": r.get("cal_balanced", ""),

                "cal_ufa": r.get("cal_ufa", ""),

                "cal_background_false_accepts": r.get("cal_background_false_accepts", ""),

            }

        )


    seen = set()

    for spec in tqdm(candidates, desc="Step10a selected test evaluation"):

        key = (spec["method"], spec["score_col"], float(spec["threshold"]))

        if key in seen:

            continue

        seen.add(key)

        if spec["score_col"] not in scored.columns:

            continue

        op = eval_operating(test, gt_test, spec["score_col"], float(spec["threshold"]), iou_thr)

        ap = ap_for(test, gt_test, known_classes, spec["score_col"])

        row = {

            **spec,

            "split": "test",

            "iou_threshold": float(iou_thr),

            "num_known_gt": op["num_known_gt"],

            "num_unknown_gt": op["num_unknown_gt"],

            "num_accepted_detections": op["num_accepted_detections"],

            "tp_known": op["tp_known"],

            "fp_known": op["fp_known"],

            "fn_known": op["fn_known"],

            "precision": op["known_precision"],

            "recall": op["known_recall"],

            "unknown_false_accepts": op["unknown_false_accept_objects"],

            "unknown_reject": op["unknown_reject_rate_object_level"],

            "balanced": op["precision_recall_unknown_balanced_score"],

            "background_false_accepts": op["background_false_accept_count"],

            "AP50": ap.get("AP50", float("nan")),

            "AP75": ap.get("AP75", float("nan")),

            "AP": ap.get("AP", float("nan")),

        }

        row["delta_balanced_vs_raw"] = row["balanced"] - raw_test_operating["precision_recall_unknown_balanced_score"]

        row["delta_ufa_vs_raw"] = row["unknown_false_accepts"] - raw_test_operating["unknown_false_accept_objects"]

        row["delta_bg_fp_vs_raw"] = row["background_false_accepts"] - raw_test_operating["background_false_accept_count"]

        row["delta_AP50_vs_raw"] = row["AP50"] - raw_test_ap["AP50"]

        row["delta_AP_vs_raw"] = row["AP"] - raw_test_ap["AP"]

        row["meets_AP50_minus_0p005"] = bool(row["AP50"] >= raw_test_ap["AP50"] - 0.005 - 1e-12)

        row["meets_AP_minus_0p005"] = bool(row["AP"] >= raw_test_ap["AP"] - 0.005 - 1e-12)

        row["meets_success_criterion"] = bool(

            row["meets_AP50_minus_0p005"]

            and row["meets_AP_minus_0p005"]

            and row["balanced"] > raw_test_operating["precision_recall_unknown_balanced_score"]

            and (

                row["unknown_false_accepts"] < raw_test_operating["unknown_false_accept_objects"]

                or row["background_false_accepts"] < raw_test_operating["background_false_accept_count"]

            )

        )

        rows.append(row)

    return pd.DataFrame(rows)



def build_method_ranking(compact: pd.DataFrame) -> pd.DataFrame:

    ranked = compact.copy()

    ranked["ap_constrained_success_rank"] = ranked["meets_success_criterion"].map(lambda x: 1 if bool(x) else 0)

    ranked = ranked.sort_values(

        [

            "ap_constrained_success_rank",

            "balanced",

            "AP50",

            "AP",

            "unknown_false_accepts",

            "background_false_accepts",

        ],

        ascending=[False, False, False, False, True, True],

    ).reset_index(drop=True)

    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))

    return ranked



def pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:

    sub = df.dropna(subset=[x_col, y_col]).copy()

    rows = []

    for idx, r in sub.iterrows():

        dominated = sub[

            (pd.to_numeric(sub[x_col], errors="coerce") >= finite_float(r[x_col]) - 1e-12)

            & (pd.to_numeric(sub[y_col], errors="coerce") >= finite_float(r[y_col]) - 1e-12)

            & (

                (pd.to_numeric(sub[x_col], errors="coerce") > finite_float(r[x_col]) + 1e-12)

                | (pd.to_numeric(sub[y_col], errors="coerce") > finite_float(r[y_col]) + 1e-12)

            )

        ]

        if dominated.empty:

            rows.append(r.to_dict())

    return pd.DataFrame(rows).sort_values([x_col, y_col], ascending=[True, True]).reset_index(drop=True)



def write_pareto_outputs(

    *,

    grid: pd.DataFrame,

    compact: pd.DataFrame,

    out_csv: Path,

    fig_dir: Path,

) -> pd.DataFrame:

    cal = grid.copy()

    cal["source_split"] = "calibration_grid"

    cal["AP50"] = pd.to_numeric(cal["cal_AP50"], errors="coerce")

    cal["AP"] = pd.to_numeric(cal["cal_AP"], errors="coerce")

    cal["balanced"] = pd.to_numeric(cal["cal_balanced"], errors="coerce")

    cal["unknown_false_accepts"] = pd.to_numeric(cal["cal_ufa"], errors="coerce")

    cal["background_false_accepts"] = pd.to_numeric(cal["cal_background_false_accepts"], errors="coerce")


    p50 = pareto_frontier(cal, "AP50", "balanced")

    p50["pareto_axis"] = "AP50_vs_balanced"

    pap = pareto_frontier(cal, "AP", "balanced")

    pap["pareto_axis"] = "AP_vs_balanced"


    test_points = compact.copy()

    test_points["source_split"] = "test_selected_or_reference"

    test_points["pareto_axis"] = "selected_test_overlay"

    cols = sorted(set(p50.columns).union(pap.columns).union(test_points.columns))

    frontier = pd.concat(

        [p50.reindex(columns=cols), pap.reindex(columns=cols), test_points.reindex(columns=cols)],

        ignore_index=True,

    )

    frontier.to_csv(out_csv, index=False, encoding="utf-8-sig")


    try:

        import matplotlib.pyplot as plt


        for x_col, fig_name, label in [

            ("AP50", "step10a_pareto_ap50_balanced.png", "AP50"),

            ("AP", "step10a_pareto_ap_balanced.png", "AP"),

        ]:

            plt.figure(figsize=(9, 6), dpi=150)

            plt.scatter(cal[x_col], cal["balanced"], s=18, alpha=0.20, label="Calibration policy grid")

            front = pareto_frontier(cal, x_col, "balanced")

            plt.plot(front[x_col], front["balanced"], color="black", linewidth=1.5, label="Calibration Pareto frontier")

            selected = compact.copy()

            sizes = 160 / (1.0 + pd.to_numeric(selected["unknown_false_accepts"], errors="coerce").fillna(0.0))

            sizes = np.clip(sizes * 15.0, 35, 260)

            plt.scatter(selected[x_col], selected["balanced"], s=sizes, marker="D", label="Fixed test points")

            for _, r in selected.iterrows():

                name = str(r["method"]).replace("step10a_", "").replace("step9c_geometry_", "geom_").replace("step9b_raw_global_reproduced", "raw")

                text = f"{name}\nUFA={int(r['unknown_false_accepts'])}, BG={int(r['background_false_accepts'])}"

                plt.annotate(text, (r[x_col], r["balanced"]), fontsize=7, xytext=(4, 4), textcoords="offset points")

            plt.xlabel(label)

            plt.ylabel("IoU0.50 open-set balanced")

            plt.title(f"AP-constrained reliability frontier: {label} vs balanced")

            plt.grid(alpha=0.25)

            plt.legend(fontsize=8)

            plt.tight_layout()

            plt.savefig(fig_dir / fig_name)

            plt.close()

    except Exception as exc:

        (fig_dir / "step10a_pareto_plot_error.txt").write_text(repr(exc), encoding="utf-8")


    return frontier



def write_markdown(path: Path, compact: pd.DataFrame, selected: pd.DataFrame, report: dict) -> None:

    success = compact[compact["meets_success_criterion"].astype(bool)].copy() if "meets_success_criterion" in compact else pd.DataFrame()

    lines = [

        "# LVIS AP-constrained Reliability Calibration Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Policy grid rows: `{report['row_counts']['policy_grid']}`",

        f"- Selected policy rows: `{report['row_counts']['selected_policies']}`",

        f"- Compact test rows: `{report['row_counts']['compact_paper_metrics']}`",

        f"- AP-constrained success found: `{bool(report['success']['ap_constrained_policy_found'])}`",

        "",

        "## Successful AP-constrained Test Policies",

        "",

    ]

    if len(success):

        show_cols = [

            "method",

            "objective",

            "constraint_name",

            "score_col",

            "threshold",

            "balanced",

            "precision",

            "recall",

            "unknown_false_accepts",

            "background_false_accepts",

            "AP50",

            "AP",

            "delta_balanced_vs_raw",

            "delta_AP50_vs_raw",

            "delta_AP_vs_raw",

        ]

        lines.append(success[[c for c in show_cols if c in success.columns]].to_markdown(index=False))

    else:

        lines.append("No AP-constrained policy found under the requested success criterion.")

    lines.extend(

        [

            "",

            "## Compact Test Metrics",

            "",

            compact.to_markdown(index=False),

            "",

            "## Selected Calibration Policies",

            "",

            selected.head(40).to_markdown(index=False),

            "",

            "## Research Note",

            "",

            "The policy grid is selected exclusively on calibration metrics. Held-out test metrics are computed only for fixed selected policies and Step9 references.",

        ]

    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--scored_csv", type=Path, default=DEFAULT_SCORED_CSV)

    parser.add_argument("--raw_predictions_csv", type=Path, default=DEFAULT_RAW_PRED_CSV)

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_PROTOCOL_ROOT / "annotations")

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv")

    parser.add_argument("--step9c_compact_csv", type=Path, default=DEFAULT_STEP9C_COMPACT)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--iou_threshold", type=float, default=0.50)

    parser.add_argument("--max_score_cols_per_run", type=int, default=0)

    parser.add_argument("--select_only", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    fig_dir = args.output_root / "figures"

    ensure_dir(csv_dir)

    ensure_dir(fig_dir)


    print("\n========== LVIS AP-constrained Reliability Calibration ==========")

    print(f"output_root={args.output_root}")


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root, split_filter="all")

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    gt_cal = gt[gt["split"].astype(str).str.lower().eq("calibration")].copy()

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    raw_pred_rows = int(len(read_csv(args.raw_predictions_csv))) if args.raw_predictions_csv.exists() else 0


    scored0 = read_csv(args.scored_csv)

    scored, specs, weights = build_score_columns(scored0)


    score_spec_df = pd.DataFrame(specs)

    score_spec_df.to_csv(csv_dir / "step10a_score_specs.csv", index=False, encoding="utf-8-sig")

    class_weight_rows = []

    for family, wmap in weights.items():

        if family == "_stats":

            continue

        for label, weight in wmap.items():

            stat = weights["_stats"].get(label, {})

            class_weight_rows.append({"weight_family": family, "pred_label": label, "weight": weight, **stat})

    pd.DataFrame(class_weight_rows).to_csv(csv_dir / "step10a_class_reliability_weights.csv", index=False, encoding="utf-8-sig")


    grid_csv = csv_dir / "step10a_policy_grid.csv"

    if not args.select_only:

        grid = process_policy_grid(

            scored=scored,

            specs=specs,

            gt_cal=gt_cal,

            known_classes=known_classes,

            grid_csv=grid_csv,

            max_score_cols_per_run=args.max_score_cols_per_run,

            iou_thr=args.iou_threshold,

        )

    else:

        grid = read_csv(grid_csv)


    done_cols = set(grid["score_col"].astype(str).tolist()) if len(grid) else set()

    expected_cols = set(s["score_col"] for s in specs)

    grid_complete = expected_cols.issubset(done_cols)


    compact = pd.DataFrame()

    selected = pd.DataFrame()

    ranking = pd.DataFrame()

    frontier = pd.DataFrame()

    success_found = False

    if grid_complete:

        cal = scored[scored["split"].astype(str).str.lower().eq("calibration")].copy()

        test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

        raw_cal_ap = ap_for(cal, gt_cal, known_classes, "raw_score")

        raw_test_ap = ap_for(test, gt_test, known_classes, "raw_score")

        raw_ref_threshold = 0.65

        step9c_compact = read_csv(args.step9c_compact_csv)

        refs = reference_specs(step9c_compact)

        raw_ref = refs[refs["method"].eq("step9b_raw_global_reproduced")]

        if len(raw_ref):

            raw_ref_threshold = finite_float(raw_ref.iloc[0]["threshold"], 0.65)

        raw_test_operating = eval_operating(test, gt_test, "raw_score", raw_ref_threshold, args.iou_threshold)


        selected = select_policies(grid, raw_cal_ap50=raw_cal_ap["AP50"], raw_cal_ap=raw_cal_ap["AP"])

        selected.to_csv(csv_dir / "step10a_selected_policies.csv", index=False, encoding="utf-8-sig")


        compact = evaluate_selected_on_test(

            scored=scored,

            gt_test=gt_test,

            known_classes=known_classes,

            selected=selected,

            references=refs,

            iou_thr=args.iou_threshold,

            raw_test_ap=raw_test_ap,

            raw_test_operating=raw_test_operating,

        )

        compact.to_csv(csv_dir / "step10a_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")

        ranking = build_method_ranking(compact)

        ranking.to_csv(csv_dir / "step10a_method_ranking.csv", index=False, encoding="utf-8-sig")

        frontier = write_pareto_outputs(

            grid=grid,

            compact=compact,

            out_csv=csv_dir / "step10a_pareto_frontier.csv",

            fig_dir=fig_dir,

        )

        success_found = bool(compact["meets_success_criterion"].astype(bool).any()) if len(compact) else False

    else:

        selected.to_csv(csv_dir / "step10a_selected_policies.csv", index=False, encoding="utf-8-sig")

        compact.to_csv(csv_dir / "step10a_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")

        ranking.to_csv(csv_dir / "step10a_method_ranking.csv", index=False, encoding="utf-8-sig")

        frontier.to_csv(csv_dir / "step10a_pareto_frontier.csv", index=False, encoding="utf-8-sig")


    report = {

        "method": "LVIS AP-constrained Open-Set Reliability Calibration",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "inputs": {

            "scored_csv": str(args.scored_csv),

            "raw_predictions_csv": str(args.raw_predictions_csv),

            "annotation_dir": str(args.annotation_dir),

            "split_csv": str(args.split_csv),

            "class_map_csv": str(args.class_map_csv),

            "step9c_compact_csv": str(args.step9c_compact_csv),

        },

        "raw_prediction_rows": raw_pred_rows,

        "score_columns_expected": len(expected_cols),

        "score_columns_completed": len(done_cols),

        "grid_complete": bool(grid_complete),

        "selection_protocol": {

            "policy_grid_split": "calibration",

            "test_evaluation": "fixed references and calibration-selected policies only",

            "AP50_drop_tolerances": AP_DROP_TOLERANCES,

            "AP_drop_tolerances": AP_DROP_TOLERANCES,

            "main_iou_threshold": float(args.iou_threshold),

            "uses_test_for_policy_selection": False,

        },

        "row_counts": {

            "policy_grid": int(len(grid)),

            "selected_policies": int(len(selected)),

            "compact_paper_metrics": int(len(compact)),

            "method_ranking": int(len(ranking)),

            "pareto_frontier": int(len(frontier)),

        },

        "success": {

            "ap_constrained_policy_found": bool(success_found),

            "success_criterion": "AP50 >= raw AP50 - 0.005; AP >= raw AP - 0.005; balanced > raw; UFA or background FP reduced",

        },

        "outputs": {

            "policy_grid": str(csv_dir / "step10a_policy_grid.csv"),

            "selected_policies": str(csv_dir / "step10a_selected_policies.csv"),

            "compact_paper_metrics": str(csv_dir / "step10a_compact_paper_metrics.csv"),

            "method_ranking": str(csv_dir / "step10a_method_ranking.csv"),

            "pareto_frontier": str(csv_dir / "step10a_pareto_frontier.csv"),

            "pareto_ap50_figure": str(fig_dir / "step10a_pareto_ap50_balanced.png"),

            "pareto_ap_figure": str(fig_dir / "step10a_pareto_ap_balanced.png"),

            "summary": str(args.output_root / "step10a_summary.md"),

            "integrity_report": str(args.output_root / "step10a_integrity_report.json"),

        },

    }

    write_markdown(args.output_root / "step10a_summary.md", compact, selected, report)

    (args.output_root / "step10a_integrity_report.json").write_text(

        json.dumps(report, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )


    print("\n========== Step10A status ==========")

    print(json.dumps(report["row_counts"], ensure_ascii=False, indent=2))

    print(f"grid_complete={grid_complete}; ap_constrained_policy_found={success_found}")



if __name__ == "__main__":

    main()

