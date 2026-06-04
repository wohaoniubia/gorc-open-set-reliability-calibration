


"""
Step12C: Geometry-aware reliability calibration on COCO-Val-OpenSet-5K.

This script transfers the Step9/10/11 geometry-aware reliability family to the
larger COCO-Val-OpenSet-5K protocol. Policy selection uses calibration split
only. Held-out test metrics are computed only after policies are fixed.

For the large COCO calibration split, the policy grid uses a cached candidate
matching table to avoid repeatedly scanning all ground-truth boxes for every
score/threshold. Final selected test policies are evaluated with the standard
project evaluator.
"""


from __future__ import annotations


import argparse

import json

import math

import sys

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Sequence, Tuple


import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import average_precision_score, roc_auc_score

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import (

    bbox_iou,

    clean_columns,

    compute_ap_summary,

    evaluate_detections,

    load_annotations,

    load_known_classes,

    load_split,

    nms_detections,

    normalize_image_id,

    safe_class_name,

)



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step12a_coco_val_openset_protocol"

DEFAULT_RAW_PRED_CSV = (

    DEFAULT_PROJECT_ROOT

    / "outputs"

    / "step12b_yoloworld_coco_openset_baseline"

    / "csv"

    / "step12b_yoloworld_raw_predictions.csv"

)

DEFAULT_STEP12B_COMPACT = (

    DEFAULT_PROJECT_ROOT

    / "outputs"

    / "step12b_yoloworld_coco_openset_baseline"

    / "csv"

    / "step12b_compact_paper_metrics.csv"

)

DEFAULT_STEP12B_SELECTED = (

    DEFAULT_PROJECT_ROOT

    / "outputs"

    / "step12b_yoloworld_coco_openset_baseline"

    / "csv"

    / "step12b_selected_calibration_config.csv"

)

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step12c_geometry_risk_coco_openset"


AP_DROP_TOLERANCES = [0.00, 0.005, 0.01, 0.02]

GEOMETRY_NUMERIC = [

    "raw_score",

    "score_logit",

    "box_area_norm",

    "box_aspect_log",

    "box_width_norm",

    "box_height_norm",

    "center_x_norm",

    "center_y_norm",

    "edge_min_dist_norm",

    "edge_contact_count",

]

GEOMETRY_CATEGORICAL = ["pred_label"]

BASE_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def finite_float(value, default: float = 0.0) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def safe_score_name(prefix: str, value: float) -> str:

    return f"{prefix}_{value:.3f}".replace(".", "p").replace("-", "m")



def parse_float_list(text: str) -> List[float]:

    return [float(x.strip()) for x in str(text).split(",") if x.strip()]



def harmonic3(a: float, b: float, c: float) -> float:

    vals = [float(a), float(b), float(c)]

    if any(v <= 0 for v in vals):

        return 0.0

    return float(3.0 / sum(1.0 / v for v in vals))



def build_model(random_state: int) -> Pipeline:

    pre = ColumnTransformer(

        transformers=[

            ("num", StandardScaler(), GEOMETRY_NUMERIC),

            ("cat", OneHotEncoder(handle_unknown="ignore"), GEOMETRY_CATEGORICAL),

        ],

        remainder="drop",

    )

    clf = LogisticRegression(

        C=0.50,

        class_weight="balanced",

        max_iter=1000,

        solver="liblinear",

        random_state=int(random_state),

    )

    return Pipeline([("pre", pre), ("clf", clf)])



def make_thresholds(values: pd.Series) -> List[float]:

    vals = set(float(x) for x in BASE_THRESHOLDS)

    clean = pd.to_numeric(values, errors="coerce").dropna()

    if len(clean):

        for q in [0.25, 0.50, 0.70, 0.80, 0.90, 0.95]:

            vals.add(float(clean.quantile(q)))

        vals.add(float(clean.max()))

    return sorted(v for v in vals if math.isfinite(v) and v >= 0.0)



def feature_candidates(

    raw_pred: pd.DataFrame,

    split_df: pd.DataFrame,

    nms_iou: float,

    nms_contain: float,

    nms_mode: str,

    edge_margin_frac: float,

) -> pd.DataFrame:

    raw = clean_columns(raw_pred).copy()

    for c in ["score", "x1", "y1", "x2", "y2"]:

        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)

    raw["image_id"] = raw["image_id"].map(normalize_image_id)

    raw["pred_label"] = raw["pred_label"].map(safe_class_name)

    raw = raw[raw["pred_label"].astype(str).ne("")].copy()

    raw = raw.sort_values("score", ascending=False).reset_index(drop=True)

    cand = nms_detections(raw, iou_thr=float(nms_iou), contain_thr=float(nms_contain), mode=nms_mode).copy()


    size_df = split_df.copy()

    size_df["image_id"] = size_df["image_id"].map(normalize_image_id)

    size_map = {

        r["image_id"]: (float(r.get("width", 0) or 0), float(r.get("height", 0) or 0))

        for _, r in size_df.iterrows()

    }


    rows: List[dict] = []

    for idx, r in tqdm(cand.reset_index(drop=True).iterrows(), total=len(cand), desc="Step12C geometry features"):

        image_id = normalize_image_id(r["image_id"])

        width, height = size_map.get(image_id, (0.0, 0.0))

        x1, y1, x2, y2 = float(r.x1), float(r.y1), float(r.x2), float(r.y2)

        box_w = max(0.0, x2 - x1)

        box_h = max(0.0, y2 - y1)

        area = box_w * box_h

        img_area = width * height if width > 0 and height > 0 else 0.0

        eps_box = 1e-6

        log_aspect = math.log(max(box_w, eps_box) / max(box_h, eps_box))

        score = min(max(float(r.score), 1e-6), 1.0 - 1e-6)

        margin = edge_margin_frac * max(width, height, 1.0)

        right_gap = max(0.0, width - x2) if width > 0 else 0.0

        bottom_gap = max(0.0, height - y2) if height > 0 else 0.0

        left_gap = max(0.0, x1)

        top_gap = max(0.0, y1)

        edge_gaps = [left_gap, top_gap, right_gap, bottom_gap]

        contact_count = int(sum(g <= margin for g in edge_gaps))

        row = r.to_dict()

        row.update(

            {

                "det_id": row.get("det_id", f"{image_id}_step12c_{idx:06d}"),

                "candidate_score": float(r.score),

                "raw_score": float(r.score),

                "score_logit": float(math.log(score / (1.0 - score))),

                "box_area_norm": float(area / img_area) if img_area > 0 else 0.0,

                "box_aspect_log": float(log_aspect),

                "box_width_norm": float(box_w / width) if width > 0 else 0.0,

                "box_height_norm": float(box_h / height) if height > 0 else 0.0,

                "center_x_norm": float((x1 + x2) * 0.5 / width) if width > 0 else 0.5,

                "center_y_norm": float((y1 + y2) * 0.5 / height) if height > 0 else 0.5,

                "edge_min_dist_norm": float(min(edge_gaps) / max(min(width, height), 1.0)) if width > 0 and height > 0 else 0.0,

                "edge_contact_count": contact_count,

                "touches_edge": bool(contact_count > 0),

            }

        )

        rows.append(row)

    return pd.DataFrame(rows)



def attach_best_matches(candidates: pd.DataFrame, gt: pd.DataFrame, label_iou: float) -> pd.DataFrame:

    out = candidates.copy().sort_values("raw_score", ascending=False).reset_index(drop=True)

    gt_by_image = {img: g.copy() for img, g in gt.groupby("image_id", sort=False)}

    matched_known = set()

    labels: List[int] = []

    error_types: List[str] = []

    best_known_ious: List[float] = []

    best_unknown_ious: List[float] = []

    best_known_labels: List[str] = []

    best_unknown_labels: List[str] = []

    best_known_keys: List[str] = []

    best_unknown_keys: List[str] = []

    for _, pr in tqdm(out.iterrows(), total=len(out), desc="Step12C candidate matching"):

        image_id = normalize_image_id(pr.image_id)

        pred_label = safe_class_name(pr.pred_label)

        pbox = (float(pr.x1), float(pr.y1), float(pr.x2), float(pr.y2))

        gimg = gt_by_image.get(image_id, gt.iloc[0:0])

        best_known_iou, best_known_key, best_known_label = 0.0, "", ""

        for _, gr in gimg[gimg["gt_is_known"]].iterrows():

            iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

            if iou > best_known_iou:

                best_known_iou = float(iou)

                best_known_key = f"{normalize_image_id(gr.image_id)}::{gr.gt_id}"

                best_known_label = safe_class_name(gr.gt_label)

        best_unknown_iou, best_unknown_key, best_unknown_label = 0.0, "", ""

        for _, gr in gimg[~gimg["gt_is_known"]].iterrows():

            iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

            if iou > best_unknown_iou:

                best_unknown_iou = float(iou)

                best_unknown_key = f"{normalize_image_id(gr.image_id)}::{gr.gt_id}"

                best_unknown_label = safe_class_name(gr.gt_label)


        raw_tp_key = best_known_key if best_known_iou >= float(label_iou) and pred_label == best_known_label else ""

        if raw_tp_key and raw_tp_key not in matched_known:

            labels.append(1)

            error_types.append("known_tp")

            matched_known.add(raw_tp_key)

        elif best_unknown_iou >= float(label_iou):

            labels.append(0)

            error_types.append("unknown_false_accept")

        elif best_known_iou >= float(label_iou):

            labels.append(0)

            error_types.append("wrong_known_class_or_duplicate")

        else:

            labels.append(0)

            error_types.append("background_false_accept")

        best_known_ious.append(float(best_known_iou))

        best_unknown_ious.append(float(best_unknown_iou))

        best_known_labels.append(best_known_label)

        best_unknown_labels.append(best_unknown_label)

        best_known_keys.append(best_known_key)

        best_unknown_keys.append(best_unknown_key)

    out["risk_label_tp"] = labels

    out["risk_error_type"] = error_types

    out["best_known_iou"] = best_known_ious

    out["best_unknown_iou"] = best_unknown_ious

    out["best_known_label"] = best_known_labels

    out["best_unknown_label"] = best_unknown_labels

    out["best_known_key"] = best_known_keys

    out["best_unknown_key"] = best_unknown_keys

    return out



def class_reliability_weights(scored: pd.DataFrame) -> Dict[str, float]:

    cal = scored[scored["split"].astype(str).str.lower().eq("calibration")].copy()

    labels = sorted(cal["pred_label"].map(safe_class_name).dropna().unique().tolist())

    raw_weights: Dict[str, float] = {}

    for label in labels:

        sub = cal[cal["pred_label"].map(safe_class_name).eq(label)]

        n = int(len(sub))

        tp = int(pd.to_numeric(sub["risk_label_tp"], errors="coerce").fillna(0).sum())

        bg = int(sub["risk_error_type"].astype(str).eq("background_false_accept").sum())

        ufa = int(sub["risk_error_type"].astype(str).eq("unknown_false_accept").sum())

        precision = (tp + 1.0) / (n + 2.0) if n else 0.5

        bg_safety = (n - bg + 1.0) / (n + 2.0) if n else 0.5

        unknown_safety = (n - ufa + 1.0) / (n + 2.0) if n else 0.5

        raw_weights[label] = float((precision * bg_safety * unknown_safety) ** (1.0 / 3.0))

    vmax = max(raw_weights.values()) if raw_weights else 1.0

    return {label: float(np.clip(v / max(vmax, 1e-9), 0.20, 1.0)) for label, v in raw_weights.items()}



def add_score_columns(scored: pd.DataFrame, alphas: Sequence[float], gammas: Sequence[float], betas: Sequence[float]) -> Tuple[pd.DataFrame, List[dict]]:

    df = scored.copy()

    df["raw_score"] = pd.to_numeric(df["raw_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    df["geometry_risk_score"] = pd.to_numeric(df["geometry_risk_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    specs: List[dict] = [

        {

            "score_col": "raw_score",

            "score_family": "raw",

            "description": "raw YOLO-World-l score",

            "mode_family": "raw",

            "ap_rank_preserving_by_design": True,

        },

        {

            "score_col": "geometry_risk_score",

            "score_family": "geometry_probability",

            "description": "calibration-only geometry reliability probability",

            "mode_family": "geometry",

            "ap_rank_preserving_by_design": False,

        },

    ]

    for alpha in alphas:

        col = safe_score_name("geometry_alpha", float(alpha))

        df[col] = float(alpha) * df["geometry_risk_score"] + (1.0 - float(alpha)) * df["raw_score"]

        specs.append(

            {

                "score_col": col,

                "score_family": "raw_geometry_alpha_blend",

                "description": f"(1-alpha)*raw + alpha*geometry; alpha={float(alpha):g}",

                "mode_family": "alpha_blend",

                "alpha": float(alpha),

                "ap_rank_preserving_by_design": bool(float(alpha) == 0.0),

            }

        )

    weights = class_reliability_weights(df)

    base_w = df["pred_label"].map(lambda x: weights.get(safe_class_name(x), 1.0)).astype(float)

    for gamma in gammas:

        col = safe_score_name("class_reliability_gamma", float(gamma))

        df[col] = df["raw_score"] * np.power(base_w, float(gamma))

        specs.append(

            {

                "score_col": col,

                "score_family": "class_reliability_scaled_raw",

                "description": f"class-wise reliability scaling; gamma={float(gamma):g}",

                "mode_family": "class_scaled_raw",

                "gamma": float(gamma),

                "ap_rank_preserving_by_design": True,

            }

        )

    for beta in betas:

        col = safe_score_name("raw_light_geom_gate", float(beta))

        gate = (1.0 - float(beta)) + float(beta) * df["geometry_risk_score"]

        df[col] = df["raw_score"] * gate.clip(lower=0.0, upper=1.0)

        specs.append(

            {

                "score_col": col,

                "score_family": "raw_light_geometry_gate",

                "description": f"raw * ((1-beta)+beta*geometry); beta={float(beta):g}",

                "mode_family": "light_geometry_gate",

                "beta": float(beta),

                "ap_rank_preserving_by_design": False,

            }

        )


    unique_specs: List[dict] = []

    seen = set()

    for spec in specs:

        col = str(spec["score_col"])

        if col in seen:

            continue

        seen.add(col)

        unique_specs.append(spec)

    return df, unique_specs



def fast_operating_metrics(scored_split: pd.DataFrame, gt_split: pd.DataFrame, score_col: str, threshold: float, iou_thr: float) -> dict:

    dets = scored_split[pd.to_numeric(scored_split[score_col], errors="coerce") >= float(threshold)].copy()

    num_known_gt = int(gt_split["gt_is_known"].sum())

    num_unknown_gt = int((~gt_split["gt_is_known"]).sum())

    if dets.empty:

        return {

            "num_known_gt": num_known_gt,

            "num_unknown_gt": num_unknown_gt,

            "num_accepted_detections": 0,

            "tp_known": 0,

            "fp_known": 0,

            "fn_known": num_known_gt,

            "known_precision": 0.0,

            "known_recall": 0.0,

            "unknown_false_accept_objects": 0,

            "unknown_reject_rate_object_level": 1.0,

            "precision_recall_unknown_balanced_score": 0.0,

            "background_false_accept_count": 0,

        }

    dets = dets.assign(_score=pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0))

    dets = dets.sort_values("_score", ascending=False).reset_index(drop=True)

    matched_known = set()

    accepted_unknown = set()

    tp = 0

    fp = 0

    bg_fp = 0

    for _, pr in dets.iterrows():

        pred_label = safe_class_name(pr.pred_label)

        known_key = str(pr.get("best_known_key", ""))

        unknown_key = str(pr.get("best_unknown_key", ""))

        best_known_iou = finite_float(pr.get("best_known_iou", 0.0), 0.0)

        best_unknown_iou = finite_float(pr.get("best_unknown_iou", 0.0), 0.0)

        best_known_label = safe_class_name(pr.get("best_known_label", ""))

        if known_key and known_key not in matched_known and best_known_iou >= float(iou_thr) and pred_label == best_known_label:

            tp += 1

            matched_known.add(known_key)

            continue

        fp += 1

        if unknown_key and best_unknown_iou >= float(iou_thr):

            accepted_unknown.add(unknown_key)

        elif best_known_iou < float(iou_thr):

            bg_fp += 1

    fn = max(0, num_known_gt - tp)

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / num_known_gt if num_known_gt else 0.0

    unknown_false = len(accepted_unknown)

    unknown_reject = 1.0 - unknown_false / num_unknown_gt if num_unknown_gt else 1.0

    return {

        "num_known_gt": num_known_gt,

        "num_unknown_gt": num_unknown_gt,

        "num_accepted_detections": int(len(dets)),

        "tp_known": int(tp),

        "fp_known": int(fp),

        "fn_known": int(fn),

        "known_precision": float(precision),

        "known_recall": float(recall),

        "unknown_false_accept_objects": int(unknown_false),

        "unknown_reject_rate_object_level": float(unknown_reject),

        "precision_recall_unknown_balanced_score": harmonic3(precision, recall, unknown_reject),

        "background_false_accept_count": int(bg_fp),

    }



def candidate_ap_proxy(scored_cal: pd.DataFrame, score_col: str) -> float:

    y = pd.to_numeric(scored_cal["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    s = pd.to_numeric(scored_cal[score_col], errors="coerce").fillna(0.0)

    if y.nunique() < 2:

        return float("nan")

    return float(average_precision_score(y.to_numpy(), s.to_numpy()))



def exact_ap_for_score(

    scored: pd.DataFrame,

    gt: pd.DataFrame,

    known_classes: Sequence[str],

    split: str,

    score_col: str,

) -> dict:

    dets = scored[scored["split"].astype(str).str.lower().eq(split.lower())].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    gt_split = gt[gt["split"].astype(str).str.lower().eq(split.lower())].copy()

    ap, _ = compute_ap_summary(dets, gt_split, known_classes)

    return ap.iloc[0].to_dict()



def build_policy_grid(

    scored: pd.DataFrame,

    specs: List[dict],

    gt_cal: pd.DataFrame,

    raw_cal_ap: dict,

    iou_thr: float,

) -> pd.DataFrame:

    cal = scored[scored["split"].astype(str).str.lower().eq("calibration")].copy()

    rows: List[dict] = []

    for spec in tqdm(specs, desc="Step12C calibration policy grid"):

        score_col = str(spec["score_col"])

        ap_rank_preserving = bool(spec.get("ap_rank_preserving_by_design", False))

        ap_proxy = candidate_ap_proxy(cal, score_col)

        if ap_rank_preserving:

            cal_ap50 = finite_float(raw_cal_ap.get("AP50", float("nan")), float("nan"))

            cal_ap75 = finite_float(raw_cal_ap.get("AP75", float("nan")), float("nan"))

            cal_ap = finite_float(raw_cal_ap.get("AP", float("nan")), float("nan"))

        else:

            cal_ap50 = float("nan")

            cal_ap75 = float("nan")

            cal_ap = float("nan")

        for thr in make_thresholds(cal[score_col]):

            m = fast_operating_metrics(cal, gt_cal, score_col, thr, iou_thr)

            rows.append(

                {

                    **spec,

                    "split_used_for_selection": "calibration",

                    "threshold": float(thr),

                    "iou_threshold": float(iou_thr),

                    "cal_AP50": cal_ap50,

                    "cal_AP75": cal_ap75,

                    "cal_AP": cal_ap,

                    "cal_candidate_AP_proxy": ap_proxy,

                    "cal_balanced": finite_float(m["precision_recall_unknown_balanced_score"], float("nan")),

                    "cal_precision": finite_float(m["known_precision"], float("nan")),

                    "cal_recall": finite_float(m["known_recall"], float("nan")),

                    "cal_unknown_reject": finite_float(m["unknown_reject_rate_object_level"], float("nan")),

                    "cal_ufa": int(m["unknown_false_accept_objects"]),

                    "cal_background_false_accepts": int(m["background_false_accept_count"]),

                    "cal_num_accepted": int(m["num_accepted_detections"]),

                }

            )

    return pd.DataFrame(rows)



def select_policies(grid: pd.DataFrame, raw_cal_ap50: float, raw_cal_ap: float, eps: float) -> pd.DataFrame:

    rows: List[dict] = []

    by_bal = grid.sort_values(

        ["cal_balanced", "cal_unknown_reject", "cal_precision", "cal_recall", "cal_num_accepted"],

        ascending=[False, False, False, False, True],

    ).iloc[0].to_dict()

    by_bal.update({"selection_policy": "max_cal_balanced", "mode": "reliability_first", "status": "selected_on_calibration"})

    rows.append(by_bal)


    finite_ap = grid[pd.to_numeric(grid["cal_AP"], errors="coerce").notna()].copy()

    if not finite_ap.empty:

        by_ap = finite_ap.sort_values(

            ["cal_AP", "cal_AP50", "cal_balanced", "cal_precision", "cal_unknown_reject"],

            ascending=[False, False, False, False, False],

        ).iloc[0].to_dict()

    else:

        by_ap = grid.sort_values(["cal_candidate_AP_proxy", "cal_balanced"], ascending=[False, False]).iloc[0].to_dict()

    by_ap.update({"selection_policy": "max_cal_ap", "mode": "ap_first", "status": "selected_on_calibration"})

    rows.append(by_ap)


    max_bal = float(pd.to_numeric(grid["cal_balanced"], errors="coerce").max())

    near = grid[pd.to_numeric(grid["cal_balanced"], errors="coerce") >= max_bal - float(eps)].copy()

    by_prec = near.sort_values(

        ["cal_precision", "cal_unknown_reject", "cal_balanced", "cal_recall", "cal_num_accepted"],

        ascending=[False, False, False, False, True],

    ).iloc[0].to_dict()

    by_prec.update(

        {

            "selection_policy": f"precision_tiebreak_eps{str(eps).replace('.', 'p')}",

            "mode": "reliability_first_precision_tiebreak",

            "status": "selected_on_calibration",

        }

    )

    rows.append(by_prec)


    for ap50_tol in AP_DROP_TOLERANCES:

        for ap_tol in AP_DROP_TOLERANCES:

            eligible = grid[

                (pd.to_numeric(grid["cal_AP50"], errors="coerce") >= raw_cal_ap50 - ap50_tol - 1e-12)

                & (pd.to_numeric(grid["cal_AP"], errors="coerce") >= raw_cal_ap - ap_tol - 1e-12)

            ].copy()

            policy_name = f"ap_constrained_balanced_AP50tol{ap50_tol:g}_APtol{ap_tol:g}".replace(".", "p")

            if eligible.empty:

                rows.append(

                    {

                        "selection_policy": policy_name,

                        "mode": "ap_constrained",

                        "status": "no_calibration_candidate",

                        "AP50_drop_tolerance": ap50_tol,

                        "AP_drop_tolerance": ap_tol,

                    }

                )

                continue

            best = eligible.sort_values(

                ["cal_balanced", "cal_unknown_reject", "cal_precision", "cal_recall", "cal_background_false_accepts", "cal_num_accepted"],

                ascending=[False, False, False, False, True, True],

            ).iloc[0].to_dict()

            best.update(

                {

                    "selection_policy": policy_name,

                    "mode": "ap_constrained" if max(ap50_tol, ap_tol) <= 0.005 else "ap_tolerant_reliability",

                    "status": "selected_on_calibration",

                    "AP50_drop_tolerance": ap50_tol,

                    "AP_drop_tolerance": ap_tol,

                }

            )

            rows.append(best)

    selected = pd.DataFrame(rows)

    return selected.drop_duplicates(subset=["selection_policy", "score_col", "threshold"], keep="first")



def exact_operating(scored_split: pd.DataFrame, gt_split: pd.DataFrame, score_col: str, threshold: float, iou_thr: float) -> dict:

    dets = scored_split[pd.to_numeric(scored_split[score_col], errors="coerce") >= float(threshold)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    metrics, _ = evaluate_detections(dets, gt_split, iou_thr=float(iou_thr))

    return metrics



def evaluate_selected_on_test(

    scored: pd.DataFrame,

    selected: pd.DataFrame,

    gt_test: pd.DataFrame,

    known_classes: Sequence[str],

    raw_metrics: dict,

    raw_ap: dict,

    iou_thr: float,

) -> pd.DataFrame:

    test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    ap_cache: Dict[str, dict] = {"raw_score": raw_ap}

    rows: List[dict] = []

    for _, r in tqdm(selected.iterrows(), total=len(selected), desc="Step12C selected test evaluation"):

        if str(r.get("status", "")) != "selected_on_calibration":

            continue

        score_col = str(r.get("score_col", ""))

        if not score_col or score_col not in test.columns:

            continue

        threshold = finite_float(r.get("threshold", 0.0), 0.0)

        op = exact_operating(test, gt_test, score_col, threshold, iou_thr)

        if bool(r.get("ap_rank_preserving_by_design", False)):

            ap = raw_ap

        else:

            if score_col not in ap_cache:

                ap_cache[score_col] = exact_ap_for_score(scored, gt_test, known_classes, "test", score_col)

            ap = ap_cache[score_col]

        row = {

            "method": "step12c_geometry_" + str(r["selection_policy"]),

            "selection_policy": str(r["selection_policy"]),

            "mode": str(r.get("mode", "")),

            "score_col": score_col,

            "score_family": str(r.get("score_family", "")),

            "description": str(r.get("description", "")),

            "threshold": threshold,

            "split": "test",

            "iou_threshold": float(iou_thr),

            "num_known_gt": op["num_known_gt"],

            "num_unknown_gt": op["num_unknown_gt"],

            "num_accepted_detections": op["num_accepted_detections"],

            "tp_known": op["tp_known"],

            "fp_known": op["fp_known"],

            "fn_known": op["fn_known"],

            "known_precision": op["known_precision"],

            "known_recall": op["known_recall"],

            "precision": op["known_precision"],

            "recall": op["known_recall"],

            "unknown_false_accept_objects": op["unknown_false_accept_objects"],

            "unknown_false_accepts": op["unknown_false_accept_objects"],

            "unknown_reject_rate_object_level": op["unknown_reject_rate_object_level"],

            "precision_recall_unknown_balanced_score": op["precision_recall_unknown_balanced_score"],

            "balanced": op["precision_recall_unknown_balanced_score"],

            "background_false_accept_count": op["background_false_accept_count"],

            "background_false_accepts": op["background_false_accept_count"],

            "AP50": finite_float(ap.get("AP50", float("nan")), float("nan")),

            "AP75": finite_float(ap.get("AP75", float("nan")), float("nan")),

            "AP": finite_float(ap.get("AP", float("nan")), float("nan")),

            "cal_balanced": r.get("cal_balanced", float("nan")),

            "cal_precision": r.get("cal_precision", float("nan")),

            "cal_recall": r.get("cal_recall", float("nan")),

            "cal_AP50": r.get("cal_AP50", float("nan")),

            "cal_AP": r.get("cal_AP", float("nan")),

            "cal_candidate_AP_proxy": r.get("cal_candidate_AP_proxy", float("nan")),

            "AP50_drop_tolerance": r.get("AP50_drop_tolerance", ""),

            "AP_drop_tolerance": r.get("AP_drop_tolerance", ""),

            "ap_rank_preserving_by_design": bool(r.get("ap_rank_preserving_by_design", False)),

        }

        row["delta_balanced_vs_raw"] = row["balanced"] - raw_metrics["precision_recall_unknown_balanced_score"]

        row["delta_precision_vs_raw"] = row["known_precision"] - raw_metrics["known_precision"]

        row["delta_recall_vs_raw"] = row["known_recall"] - raw_metrics["known_recall"]

        row["delta_ufa_vs_raw"] = row["unknown_false_accept_objects"] - raw_metrics["unknown_false_accept_objects"]

        row["delta_bg_fp_vs_raw"] = row["background_false_accept_count"] - raw_metrics["background_false_accept_count"]

        row["delta_AP50_vs_raw"] = row["AP50"] - raw_ap["AP50"]

        row["delta_AP_vs_raw"] = row["AP"] - raw_ap["AP"]

        row["meets_AP50_minus_0p005"] = bool(row["AP50"] >= raw_ap["AP50"] - 0.005 - 1e-12)

        row["meets_AP_minus_0p005"] = bool(row["AP"] >= raw_ap["AP"] - 0.005 - 1e-12)

        row["meets_success_criterion"] = bool(

            row["meets_AP50_minus_0p005"]

            and row["meets_AP_minus_0p005"]

            and (

                row["balanced"] >= raw_metrics["precision_recall_unknown_balanced_score"]

                or row["known_precision"] > raw_metrics["known_precision"]

            )

            and (

                row["unknown_false_accept_objects"] < raw_metrics["unknown_false_accept_objects"]

                or row["background_false_accept_count"] < raw_metrics["background_false_accept_count"]

            )

        )

        rows.append(row)

    return pd.DataFrame(rows)



def write_markdown(path: Path, compact: pd.DataFrame, selected: pd.DataFrame, report: dict) -> None:

    lines = [

        "# Step12C Geometry-Risk COCO-Val-OpenSet-5K Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Scored candidates: `{report['row_counts']['scored_candidates']}`",

        f"- Calibration candidates: `{report['diagnostics']['calibration_num_candidates']}`",

        f"- Calibration positives: `{report['diagnostics']['calibration_positive']}`",

        f"- Calibration negatives: `{report['diagnostics']['calibration_negative']}`",

        f"- Geometry risk AP on calibration labels: `{report['diagnostics']['risk_average_precision']:.4f}`",

        f"- AP-constrained success found on test: `{report['success']['ap_constrained_policy_found']}`",

        "",

        "## Selected Policies",

        "",

        selected.to_markdown(index=False),

        "",

        "## Held-Out Test Compact Metrics",

        "",

        compact.to_markdown(index=False),

        "",

        "## Selection Note",

        "",

        "The large calibration grid uses cached candidate-to-ground-truth matches for fast calibration-only policy search. Final selected test policies use the standard project evaluator.",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--raw_pred_csv", type=Path, default=DEFAULT_RAW_PRED_CSV)

    parser.add_argument("--step12b_compact_csv", type=Path, default=DEFAULT_STEP12B_COMPACT)

    parser.add_argument("--step12b_selected_config", type=Path, default=DEFAULT_STEP12B_SELECTED)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step12a_scene_split.csv")

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_PROTOCOL_ROOT / "annotations")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step12a_class_map.csv")

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--alphas", type=str, default="0.05,0.10,0.20,0.40,0.60,0.80,1.0")

    parser.add_argument("--gammas", type=str, default="0.25,0.50,0.75,1.0")

    parser.add_argument("--betas", type=str, default="0.05,0.10,0.20")

    parser.add_argument("--selection_iou", type=float, default=0.50)

    parser.add_argument("--label_iou", type=float, default=0.50)

    parser.add_argument("--nms_iou", type=float, default=0.50)

    parser.add_argument("--nms_contain", type=float, default=0.90)

    parser.add_argument("--nms_mode", type=str, choices=["class", "global"], default="class")

    parser.add_argument("--edge_margin_frac", type=float, default=0.015)

    parser.add_argument("--precision_tiebreak_tolerance", type=float, default=0.005)

    parser.add_argument("--random_state", type=int, default=12)

    parser.add_argument("--force_rebuild", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    scored_csv = csv_dir / "step12c_geometry_scored_candidates.csv"

    print("\n========== Step12C Geometry-Risk COCO-Val-OpenSet-5K ==========")

    print(f"raw_pred_csv={args.raw_pred_csv}")


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root, split_filter="all")

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    gt_cal = gt[gt["split"].astype(str).str.lower().eq("calibration")].copy()

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    raw_baseline = read_csv(args.step12b_compact_csv).iloc[0].to_dict()

    raw_metrics = {

        "precision_recall_unknown_balanced_score": finite_float(raw_baseline["precision_recall_unknown_balanced_score"], 0.0),

        "known_precision": finite_float(raw_baseline["known_precision"], 0.0),

        "known_recall": finite_float(raw_baseline["known_recall"], 0.0),

        "unknown_false_accept_objects": int(finite_float(raw_baseline["unknown_false_accept_objects"], 0.0)),

        "background_false_accept_count": int(finite_float(raw_baseline["background_false_accept_count"], 0.0)),

    }

    raw_ap_test = {

        "AP50": finite_float(raw_baseline["AP50"], float("nan")),

        "AP75": finite_float(raw_baseline["AP75"], float("nan")),

        "AP": finite_float(raw_baseline["AP"], float("nan")),

    }

    raw_thr = finite_float(read_csv(args.step12b_selected_config).iloc[0].get("conf_thr", 0.30), 0.30)


    if scored_csv.exists() and not args.force_rebuild:

        scored = read_csv(scored_csv)

        diagnostics = json.loads((args.output_root / "step12c_integrity_report.json").read_text(encoding="utf-8")).get("diagnostics", {})

    else:

        raw_pred = read_csv(args.raw_pred_csv)

        candidates = feature_candidates(raw_pred, split_df, args.nms_iou, args.nms_contain, args.nms_mode, args.edge_margin_frac)

        labeled = attach_best_matches(candidates, gt, args.label_iou)

        labeled.to_csv(csv_dir / "step12c_candidate_features_labeled.csv", index=False, encoding="utf-8-sig")


        train = labeled[labeled["split"].astype(str).str.lower().eq("calibration")].copy()

        y = pd.to_numeric(train["risk_label_tp"], errors="coerce").fillna(0).astype(int)

        if y.nunique() < 2:

            raise ValueError("Calibration labels contain one class only; cannot fit geometry reliability model.")

        model = build_model(args.random_state)

        sample_weight = np.ones(len(train), dtype=float)

        sample_weight[train["risk_error_type"].eq("unknown_false_accept").to_numpy()] = 2.0

        sample_weight[train["risk_error_type"].eq("background_false_accept").to_numpy()] = 1.25

        model.fit(train, y, clf__sample_weight=sample_weight)


        scored = labeled.copy()

        scored["geometry_risk_score"] = model.predict_proba(scored)[:, 1].astype(float)

        train_probs = scored[scored["split"].astype(str).str.lower().eq("calibration")]["geometry_risk_score"].to_numpy()

        diagnostics = {

            "calibration_num_candidates": int(len(train)),

            "calibration_positive": int(y.sum()),

            "calibration_negative": int((1 - y).sum()),

            "calibration_unknown_negative": int(train["risk_error_type"].eq("unknown_false_accept").sum()),

            "calibration_background_negative": int(train["risk_error_type"].eq("background_false_accept").sum()),

            "risk_average_precision": float(average_precision_score(y.to_numpy(), train_probs)),

        }

        try:

            diagnostics["risk_roc_auc"] = float(roc_auc_score(y.to_numpy(), train_probs))

        except Exception:

            diagnostics["risk_roc_auc"] = float("nan")


        scored.to_csv(scored_csv, index=False, encoding="utf-8-sig")


    alphas = parse_float_list(args.alphas)

    gammas = parse_float_list(args.gammas)

    betas = parse_float_list(args.betas)

    scored, specs = add_score_columns(scored, alphas, gammas, betas)

    scored.to_csv(scored_csv, index=False, encoding="utf-8-sig")


    print("Computing exact raw calibration AP once for rank-preserving AP constraints...")

    raw_cal_ap = exact_ap_for_score(scored, gt_cal, known_classes, "calibration", "raw_score")

    grid = build_policy_grid(scored, specs, gt_cal, raw_cal_ap, args.selection_iou)

    raw_cal_ap50 = finite_float(raw_cal_ap.get("AP50", float("nan")), float("nan"))

    raw_cal_ap = finite_float(raw_cal_ap.get("AP", float("nan")), float("nan"))

    selected = select_policies(grid, raw_cal_ap50, raw_cal_ap, args.precision_tiebreak_tolerance)


    scored_test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    raw_row = {

        "method": "step12b_yoloworld_l_raw_reproduced",

        "selection_policy": "step12b_selected_threshold",

        "mode": "raw",

        "score_col": "raw_score",

        "score_family": "raw",

        "description": "Step12B raw YOLO-World-l baseline threshold reproduced",

        "threshold": raw_thr,

        "split": "test",

        "iou_threshold": float(args.selection_iou),

        "num_known_gt": int(raw_baseline["num_known_gt"]),

        "num_unknown_gt": int(raw_baseline["num_unknown_gt"]),

        "num_accepted_detections": int(raw_baseline["num_accepted_detections"]),

        "tp_known": int(raw_baseline["tp_known"]),

        "fp_known": int(raw_baseline["fp_known"]),

        "fn_known": int(raw_baseline["fn_known"]),

        "known_precision": raw_metrics["known_precision"],

        "known_recall": raw_metrics["known_recall"],

        "precision": raw_metrics["known_precision"],

        "recall": raw_metrics["known_recall"],

        "unknown_false_accept_objects": raw_metrics["unknown_false_accept_objects"],

        "unknown_false_accepts": raw_metrics["unknown_false_accept_objects"],

        "unknown_reject_rate_object_level": finite_float(raw_baseline["unknown_reject_rate_object_level"], float("nan")),

        "precision_recall_unknown_balanced_score": raw_metrics["precision_recall_unknown_balanced_score"],

        "balanced": raw_metrics["precision_recall_unknown_balanced_score"],

        "background_false_accept_count": raw_metrics["background_false_accept_count"],

        "background_false_accepts": raw_metrics["background_false_accept_count"],

        "AP50": raw_ap_test["AP50"],

        "AP75": raw_ap_test["AP75"],

        "AP": raw_ap_test["AP"],

        "delta_balanced_vs_raw": 0.0,

        "delta_precision_vs_raw": 0.0,

        "delta_recall_vs_raw": 0.0,

        "delta_ufa_vs_raw": 0,

        "delta_bg_fp_vs_raw": 0,

        "delta_AP50_vs_raw": 0.0,

        "delta_AP_vs_raw": 0.0,

        "meets_AP50_minus_0p005": True,

        "meets_AP_minus_0p005": True,

        "meets_success_criterion": False,

    }

    test_df = evaluate_selected_on_test(scored, selected, gt_test, known_classes, raw_metrics, raw_ap_test, args.selection_iou)

    compact = pd.concat([pd.DataFrame([raw_row]), test_df], ignore_index=True)

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


    alpha_summary = grid.sort_values(

        ["score_col", "cal_balanced", "cal_unknown_reject", "cal_precision", "cal_recall"],

        ascending=[True, False, False, False, False],

    ).groupby("score_col", as_index=False).head(1).reset_index(drop=True)


    grid.to_csv(csv_dir / "step12c_policy_grid.csv", index=False, encoding="utf-8-sig")

    alpha_summary.to_csv(csv_dir / "step12c_alpha_sweep_summary.csv", index=False, encoding="utf-8-sig")

    selected.to_csv(csv_dir / "step12c_selected_policies.csv", index=False, encoding="utf-8-sig")

    compact.to_csv(csv_dir / "step12c_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")

    ranking.to_csv(csv_dir / "step12c_method_ranking.csv", index=False, encoding="utf-8-sig")


    success_rows = compact[compact["meets_success_criterion"].astype(bool)].copy()

    report = {

        "method": "Step12C geometry-aware open-set reliability calibration on COCO-Val-OpenSet-5K",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "raw_pred_csv": str(args.raw_pred_csv),

        "split_csv": str(args.split_csv),

        "annotation_dir": str(args.annotation_dir),

        "class_map_csv": str(args.class_map_csv),

        "output_root": str(args.output_root),

        "known_classes": known_classes,

        "geometry_numeric": GEOMETRY_NUMERIC,

        "geometry_categorical": GEOMETRY_CATEGORICAL,

        "alphas": alphas,

        "gammas": gammas,

        "betas": betas,

        "selection_iou": float(args.selection_iou),

        "label_iou": float(args.label_iou),

        "nms_iou": float(args.nms_iou),

        "nms_contain": float(args.nms_contain),

        "nms_mode": args.nms_mode,

        "raw_threshold": float(raw_thr),

        "calibration_only_selection": True,

        "test_split_final_evaluation_only": True,

        "calibration_policy_grid_uses_cached_candidate_matches": True,

        "selected_test_policies_use_standard_evaluator": True,

        "diagnostics": diagnostics,

        "row_counts": {

            "scored_candidates": int(len(scored)),

            "policy_grid": int(len(grid)),

            "selected_policies": int(len(selected)),

            "compact_paper_metrics": int(len(compact)),

        },

        "success": {

            "ap_constrained_policy_found": bool(len(success_rows) > 0),

            "num_success_rows": int(len(success_rows)),

            "best_success_method": str(success_rows.iloc[0]["method"]) if len(success_rows) else "",

        },

        "outputs": {

            "geometry_scored_candidates": str(csv_dir / "step12c_geometry_scored_candidates.csv"),

            "alpha_sweep_summary": str(csv_dir / "step12c_alpha_sweep_summary.csv"),

            "selected_policies": str(csv_dir / "step12c_selected_policies.csv"),

            "compact_paper_metrics": str(csv_dir / "step12c_compact_paper_metrics.csv"),

            "method_ranking": str(csv_dir / "step12c_method_ranking.csv"),

            "summary": str(args.output_root / "step12c_summary.md"),

            "integrity_report": str(args.output_root / "step12c_integrity_report.json"),

        },

    }

    with open(args.output_root / "step12c_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)

    write_markdown(args.output_root / "step12c_summary.md", compact, selected, report)


    print("\n========== Step12C completed ==========")

    show_cols = [

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

    print(ranking[[c for c in show_cols if c in ranking.columns]].to_string(index=False))



if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("Interrupted by user.", file=sys.stderr)

        raise

