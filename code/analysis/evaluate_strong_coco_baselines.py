from __future__ import annotations


import argparse

import json

import math

import re

import sys

from pathlib import Path

from typing import Callable


import numpy as np

import pandas as pd

from scipy.optimize import minimize_scalar

from scipy.special import expit

from sklearn.compose import ColumnTransformer

from sklearn.isotonic import IsotonicRegression

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler



ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT / "GORC_paper_release_bundle"

OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "03_baselines"

TABLE_OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "09_tables_for_paper"


CORE_CODE = BUNDLE / "code" / "core_evaluators"

COCO_CODE = BUNDLE / "code" / "coco_step12"

LVIS_CODE = BUNDLE / "code" / "lvis_step9_step10"

for p in [str(CORE_CODE), str(COCO_CODE), str(LVIS_CODE)]:

    if p not in sys.path:

        sys.path.insert(0, p)


from step8j_yoloworld_lvis_openvoc_baseline import (

    clean_columns,

    compute_ap_summary,

    evaluate_detections,

    load_annotations,

    load_known_classes,

    load_split,

)



BASE_THRESHOLDS = [

    0.001,

    0.002,

    0.003,

    0.005,

    0.008,

    0.010,

    0.015,

    0.020,

    0.030,

    0.050,

    0.080,

    0.100,

    0.150,

    0.200,

    0.250,

    0.300,

    0.400,

    0.500,

    0.650,

    0.750,

    0.850,

    0.950,

]



def read_csv(path: Path) -> pd.DataFrame:

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def finite(value, default: float = float("nan")) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def safe_name(value) -> str:

    s = str(value).strip().lower()

    s = re.sub(r"\s+", "_", s)

    s = re.sub(r"[^a-z0-9_]+", "_", s)

    s = re.sub(r"_+", "_", s).strip("_")

    return s



def harmonic3(p: float, r: float, u: float) -> float:

    vals = [float(p), float(r), float(u)]

    if any(v <= 0 for v in vals):

        return 0.0

    return float(3.0 / sum(1.0 / v for v in vals))



def make_thresholds(values: pd.Series, extra: list[float] | None = None) -> list[float]:

    vals = set(BASE_THRESHOLDS)

    if extra:

        vals.update(float(x) for x in extra if math.isfinite(float(x)))

    clean = pd.to_numeric(values, errors="coerce").dropna()

    if len(clean):

        for q in np.linspace(0.02, 0.98, 25):

            vals.add(float(clean.quantile(q)))

        vals.add(float(clean.min()))

        vals.add(float(clean.max()))

    return sorted(v for v in vals if math.isfinite(v))



def make_class_thresholds(values: pd.Series, init_threshold: float) -> list[float]:

    vals = {float(init_threshold), 0.001, 0.01, 0.05, 0.10, 0.20, 0.30, 0.50, 0.65, 0.80, 0.95}

    clean = pd.to_numeric(values, errors="coerce").dropna()

    if len(clean):

        for q in [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]:

            vals.add(float(clean.quantile(q)))

    return sorted(v for v in vals if math.isfinite(v))



def load_protocol(protocol: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Path]:

    if protocol == "coco":

        proto = BUNDLE / "data_outputs" / "coco_main" / "step12a_protocol"

        split = load_split(proto / "csv" / "step12a_scene_split.csv", ROOT, split_filter="all")

        known = load_known_classes(proto / "csv" / "step12a_class_map.csv")

        gt = load_annotations(proto / "annotations", split, known)

        scored_path = BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_geometry_scored_candidates.csv"

        return split, gt, known, scored_path

    if protocol == "lvis":

        proto = BUNDLE / "data_outputs" / "lvis_main" / "step9a_protocol"

        split = load_split(proto / "csv" / "step9a_scene_split.csv", ROOT, split_filter="all")

        known = load_known_classes(proto / "csv" / "step9a_class_map.csv")

        gt = load_annotations(proto / "annotations", split, known)

        scored_path = BUNDLE / "data_outputs" / "lvis_main" / "step9c_gorc" / "csv" / "step9c_geometry_scored_candidates.csv"

        return split, gt, known, scored_path

    raise ValueError(protocol)



def prepare_scored(protocol: str) -> tuple[pd.DataFrame, pd.DataFrame, list[str], Path]:

    _, gt, known, scored_path = load_protocol(protocol)

    scored = read_csv(scored_path)

    scored["split"] = scored["split"].astype(str).str.lower()

    scored["pred_label"] = scored["pred_label"].map(safe_name)

    for c in ["x1", "y1", "x2", "y2", "raw_score", "score", "candidate_score", "score_logit"]:

        if c in scored.columns:

            scored[c] = pd.to_numeric(scored[c], errors="coerce")

    if "raw_score" not in scored.columns and "score" in scored.columns:

        scored["raw_score"] = scored["score"]

    if "score_logit" not in scored.columns:

        s = pd.to_numeric(scored["raw_score"], errors="coerce").clip(1e-6, 1 - 1e-6)

        scored["score_logit"] = np.log(s / (1 - s))

    return scored, gt, known, scored_path



def eval_dets(scored: pd.DataFrame, gt: pd.DataFrame, split: str, score_col: str, threshold: float, iou: float = 0.50) -> dict:

    sub = scored[scored["split"].eq(split)].copy()

    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce").fillna(-math.inf)

    dets = sub[sub[score_col] >= float(threshold)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    gt_split = gt[gt["split"].astype(str).str.lower().eq(split)].copy()

    metrics, _ = evaluate_detections(dets, gt_split, iou_thr=float(iou))

    return metrics



def eval_mask(scored: pd.DataFrame, gt: pd.DataFrame, split: str, score_col: str, mask: pd.Series, iou: float = 0.50) -> dict:

    sub = scored[scored["split"].eq(split)].copy()

    local_mask = mask.loc[sub.index] if hasattr(mask, "loc") else mask

    dets = sub[local_mask.to_numpy()].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    gt_split = gt[gt["split"].astype(str).str.lower().eq(split)].copy()

    metrics, _ = evaluate_detections(dets, gt_split, iou_thr=float(iou))

    return metrics



def exact_ap(scored: pd.DataFrame, gt: pd.DataFrame, known: list[str], split: str, score_col: str) -> dict:

    dets = scored[scored["split"].eq(split)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    gt_split = gt[gt["split"].astype(str).str.lower().eq(split)].copy()

    ap, _ = compute_ap_summary(dets, gt_split, known)

    return ap.iloc[0].to_dict()



def metrics_to_row(m: dict) -> dict:

    return {

        "B": finite(m.get("precision_recall_unknown_balanced_score")),

        "P": finite(m.get("known_precision")),

        "R": finite(m.get("known_recall")),

        "URR": finite(m.get("unknown_reject_rate_object_level")),

        "UFA": int(finite(m.get("unknown_false_accept_objects"), 0)),

        "BG_FP": int(finite(m.get("background_false_accept_count"), 0)),

        "accepted_detection_count": int(finite(m.get("num_accepted_detections"), 0)),

        "known_TP_count": int(finite(m.get("tp_known"), 0)),

        "known_FP_count": int(finite(m.get("fp_known"), 0)),

    }



def rank_metric(m: dict) -> tuple:

    return (

        finite(m.get("precision_recall_unknown_balanced_score"), -1),

        finite(m.get("unknown_reject_rate_object_level"), -1),

        finite(m.get("known_precision"), -1),

        finite(m.get("known_recall"), -1),

        -finite(m.get("num_accepted_detections"), 1e18),

    )



def select_global_threshold(scored: pd.DataFrame, gt: pd.DataFrame, score_col: str, split: str = "calibration") -> tuple[float, dict, pd.DataFrame]:

    rows = []

    for thr in make_thresholds(scored.loc[scored["split"].eq(split), score_col]):

        m = eval_dets(scored, gt, split, score_col, thr)

        rows.append({"threshold": thr, **metrics_to_row(m), "_metrics": m})

    df = pd.DataFrame(rows)

    best_i = max(range(len(rows)), key=lambda i: rank_metric(rows[i]["_metrics"]))

    best = rows[best_i]

    return float(best["threshold"]), best["_metrics"], df.drop(columns=["_metrics"])



def fit_logistic_score(scored: pd.DataFrame, feature_cols: list[str], include_class: bool, output_col: str, random_state: int = 2026) -> Pipeline:

    cal = scored[scored["split"].eq("calibration")].copy()

    y = pd.to_numeric(cal["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    num_cols = [c for c in feature_cols if c in scored.columns]

    transformers = []

    if num_cols:

        transformers.append(("num", StandardScaler(), num_cols))

    if include_class:

        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), ["pred_label"]))

    pipe = Pipeline(

        [

            ("pre", ColumnTransformer(transformers=transformers, remainder="drop")),

            ("clf", LogisticRegression(C=0.50, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=random_state)),

        ]

    )

    pipe.fit(cal, y)

    scored[output_col] = pipe.predict_proba(scored)[:, 1]

    return pipe



def fit_temperature_score(scored: pd.DataFrame, output_col: str) -> float:

    cal = scored[scored["split"].eq("calibration")].copy()

    y = pd.to_numeric(cal["risk_label_tp"], errors="coerce").fillna(0).to_numpy(dtype=float)

    z = pd.to_numeric(cal["score_logit"], errors="coerce").fillna(0.0).to_numpy(dtype=float)


    def nll(log_t: float) -> float:

        t = math.exp(float(log_t))

        p = np.clip(expit(z / t), 1e-6, 1 - 1e-6)

        return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


    res = minimize_scalar(nll, bounds=(math.log(0.05), math.log(20.0)), method="bounded")

    temp = math.exp(float(res.x))

    scored[output_col] = expit(pd.to_numeric(scored["score_logit"], errors="coerce").fillna(0.0).to_numpy(dtype=float) / temp)

    return temp



def fit_isotonic_score(scored: pd.DataFrame, output_col: str) -> None:

    cal = scored[scored["split"].eq("calibration")].copy()

    x = pd.to_numeric(cal["raw_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    y = pd.to_numeric(cal["risk_label_tp"], errors="coerce").fillna(0).to_numpy(dtype=float)

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    iso.fit(x, y)

    scored[output_col] = iso.predict(pd.to_numeric(scored["raw_score"], errors="coerce").fillna(0.0).to_numpy(dtype=float))



def fit_histogram_score(scored: pd.DataFrame, output_col: str, bins: int = 12) -> list[dict]:

    cal = scored[scored["split"].eq("calibration")].copy()

    x = pd.to_numeric(cal["raw_score"], errors="coerce").fillna(0.0)

    y = pd.to_numeric(cal["risk_label_tp"], errors="coerce").fillna(0.0)

    edges = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))

    if len(edges) < 3:

        edges = np.linspace(float(x.min()), float(x.max()), min(bins, max(2, len(x))) + 1)

    edges[0] = -math.inf

    edges[-1] = math.inf

    ids = np.digitize(x, edges[1:-1], right=True)

    means = {}

    specs = []

    global_mean = float(y.mean()) if len(y) else 0.0

    for b in range(len(edges) - 1):

        mask = ids == b

        val = float(y[mask].mean()) if mask.any() else global_mean

        means[b] = val

        specs.append({"bin": b, "left": edges[b], "right": edges[b + 1], "empirical_tp_rate": val, "n": int(mask.sum())})

    all_x = pd.to_numeric(scored["raw_score"], errors="coerce").fillna(0.0)

    all_ids = np.digitize(all_x, edges[1:-1], right=True)

    scored[output_col] = [means.get(int(i), global_mean) for i in all_ids]

    return specs



def fit_class_scaling(scored: pd.DataFrame, gamma: float, output_col: str) -> dict[str, float]:

    cal = scored[scored["split"].eq("calibration")].copy()

    weights = {}

    for label, sub in cal.groupby("pred_label", sort=False):

        n = len(sub)

        tp = float(pd.to_numeric(sub["risk_label_tp"], errors="coerce").fillna(0).sum())


        weights[str(label)] = float((tp + 1.0) / (n + 2.0)) if n else 0.5

    vmax = max(weights.values()) if weights else 1.0

    norm = {k: max(1e-6, v / max(vmax, 1e-9)) for k, v in weights.items()}

    scored[output_col] = pd.to_numeric(scored["raw_score"], errors="coerce").fillna(0.0) * scored["pred_label"].map(lambda x: norm.get(str(x), 0.5) ** float(gamma))

    return norm



def select_per_class_thresholds(scored: pd.DataFrame, gt: pd.DataFrame, score_col: str, init_threshold: float, max_iter: int = 2) -> tuple[dict[str, float], dict]:

    cal = scored[scored["split"].eq("calibration")].copy()

    labels = sorted(cal["pred_label"].dropna().astype(str).unique().tolist())

    thresholds = {label: float(init_threshold) for label in labels}

    choices = {label: make_class_thresholds(cal.loc[cal["pred_label"].eq(label), score_col], init_threshold) for label in labels}


    def eval_current() -> dict:

        split_mask = scored["split"].eq("calibration")

        scores = pd.to_numeric(scored[score_col], errors="coerce").fillna(-math.inf)

        thrs = scored["pred_label"].map(lambda x: thresholds.get(str(x), init_threshold)).astype(float)

        mask = split_mask & (scores >= thrs)

        return eval_mask(scored, gt, "calibration", score_col, mask)


    best_m = eval_current()

    for _ in range(max_iter):

        improved = False

        for label in labels:

            best_local = (rank_metric(best_m), thresholds[label], best_m)

            old = thresholds[label]

            for thr in choices[label]:

                thresholds[label] = float(thr)

                m = eval_current()

                key = rank_metric(m)

                if key > best_local[0]:

                    best_local = (key, float(thr), m)

            thresholds[label] = best_local[1]

            if thresholds[label] != old:

                improved = True

            best_m = best_local[2]

        if not improved:

            break

    return thresholds, best_m



def eval_per_class_thresholds(scored: pd.DataFrame, gt: pd.DataFrame, split: str, score_col: str, thresholds: dict[str, float], fallback: float) -> dict:

    sub = scored[scored["split"].eq(split)]

    scores = pd.to_numeric(sub[score_col], errors="coerce").fillna(-math.inf)

    thrs = sub["pred_label"].map(lambda x: thresholds.get(str(x), fallback)).astype(float)

    mask = pd.Series(False, index=scored.index)

    mask.loc[sub.index] = scores.to_numpy() >= thrs.to_numpy()

    return eval_mask(scored, gt, split, score_col, mask)



def completed_keys(path: Path) -> set[tuple[str, str, str]]:

    if not path.exists():

        return set()

    df = read_csv(path)

    if df.empty:

        return set()

    return set(zip(df["protocol"].astype(str), df["method"].astype(str), df["selection_mode"].astype(str)))



def append_row(path: Path, row: dict) -> None:

    ensure_dir(path.parent)

    exists = path.exists()

    with path.open("a", newline="", encoding="utf-8-sig") as f:

        writer = csv_dict_writer(f, row.keys(), write_header=not exists)

        writer.writerow(row)



def csv_dict_writer(handle, fieldnames, write_header: bool):

    import csv


    writer = csv.DictWriter(handle, fieldnames=list(fieldnames))

    if write_header:

        writer.writeheader()

    return writer



def run_protocol(protocol: str, resume: bool = True) -> pd.DataFrame:

    ensure_dir(OUT)

    result_path = OUT / f"{protocol}_strong_baselines.csv"

    done = completed_keys(result_path) if resume else set()

    if result_path.exists() and not resume:

        result_path.unlink()


    scored, gt, known, candidate_path = prepare_scored(protocol)

    print(f"[{protocol}] loaded scored={len(scored)} gt={len(gt)} known_classes={len(known)}", flush=True)

    cal_gt = gt[gt["split"].astype(str).str.lower().eq("calibration")].copy()

    test_gt = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    _ = cal_gt, test_gt

    protocol_name = "COCO-Val-OpenSet-5K" if protocol == "coco" else "LVIS-Clear-Mini-300"

    detector = "YOLO-World-l"


    numeric_geometry = [

        c

        for c in [

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

        if c in scored.columns

    ]


    method_specs: list[dict] = []

    ap_equiv_to_raw: set[str] = set()


    method_specs.append({"group": "Raw", "method": "Raw global threshold", "score_col": "raw_score", "hyperparameters": ""})


    platt_pipe = fit_logistic_score(scored, ["score_logit"], False, "round2_platt_score")

    platt_coef = float(platt_pipe.named_steps["clf"].coef_[0][0])

    if platt_coef >= 0:

        ap_equiv_to_raw.add("round2_platt_score")

    method_specs.append({"group": "Calibration", "method": "Score-only Platt/logistic", "score_col": "round2_platt_score", "hyperparameters": "C=0.50,class_weight=balanced"})


    temp = fit_temperature_score(scored, "round2_temperature_score")

    ap_equiv_to_raw.add("round2_temperature_score")

    method_specs.append({"group": "Calibration", "method": "Temperature-scaled score logit", "score_col": "round2_temperature_score", "hyperparameters": f"T={temp:.6g}"})


    fit_isotonic_score(scored, "round2_isotonic_score")

    method_specs.append({"group": "Calibration", "method": "Isotonic score calibration", "score_col": "round2_isotonic_score", "hyperparameters": "out_of_bounds=clip"})


    hist_specs = fit_histogram_score(scored, "round2_histogram_score", bins=12)

    (OUT / f"{protocol}_histogram_bins.json").write_text(json.dumps(hist_specs, indent=2), encoding="utf-8")

    method_specs.append({"group": "Calibration", "method": "Histogram/quantile binning", "score_col": "round2_histogram_score", "hyperparameters": "quantile_bins=12"})


    fit_class_scaling(scored, 0.50, "round2_class_precision_gamma_0p5")

    method_specs.append({"group": "Class", "method": "Class-wise score scaling", "score_col": "round2_class_precision_gamma_0p5", "hyperparameters": "Laplace class precision,gamma=0.5"})


    fit_logistic_score(scored, ["score_logit"], True, "round2_score_class_logistic")

    method_specs.append({"group": "Feature logistic", "method": "Score + class logistic", "score_col": "round2_score_class_logistic", "hyperparameters": "score_logit+pred_label"})


    fit_logistic_score(scored, numeric_geometry, False, "round2_score_geometry_logistic")

    method_specs.append({"group": "Feature logistic", "method": "Score + geometry logistic", "score_col": "round2_score_geometry_logistic", "hyperparameters": ",".join(numeric_geometry)})


    fit_logistic_score(scored, numeric_geometry, True, "round2_score_class_geometry_logistic")

    method_specs.append({"group": "Feature logistic", "method": "Score + class + geometry logistic", "score_col": "round2_score_class_geometry_logistic", "hyperparameters": ",".join(numeric_geometry + ["pred_label"])})


    ap_cache: dict[tuple[str, str], dict] = {}


    if result_path.exists():

        existing = read_csv(result_path)

        raw_existing = existing[

            existing["method"].astype(str).eq("Raw global threshold")

            & existing["selection_mode"].astype(str).eq("RF global threshold")

        ]

        if not raw_existing.empty:

            raw_row = raw_existing.iloc[0]

            ap_cache[("calibration", "raw_score")] = {

                "AP50": finite(raw_row.get("cal_AP50")),

                "AP75": finite(raw_row.get("cal_AP75")),

                "AP": finite(raw_row.get("cal_AP")),

            }

            ap_cache[("test", "raw_score")] = {

                "AP50": finite(raw_row.get("AP50")),

                "AP75": finite(raw_row.get("AP75")),

                "AP": finite(raw_row.get("AP")),

            }

            print(f"[{protocol}] seeded raw AP cache from existing CSV checkpoint", flush=True)


    def ap_for(split: str, score_col: str) -> dict:

        if score_col in ap_equiv_to_raw:

            score_col = "raw_score"

        key = (split, score_col)

        if key not in ap_cache:

            ap_cache[key] = exact_ap(scored, gt, known, split, score_col)

        return ap_cache[key]


    print(f"[{protocol}] computing raw calibration AP cache", flush=True)

    raw_cal_ap = ap_for("calibration", "raw_score")


    def emit(method_spec: dict, selection_mode: str, threshold, cal_m: dict, test_m: dict, score_col: str, hyper: str, per_class_thresholds: dict | None = None) -> None:

        key = (protocol_name, method_spec["method"], selection_mode)

        if key in done:

            return

        print(f"[{protocol}] computing AP for {method_spec['method']} / {selection_mode}", flush=True)

        cal_ap = ap_for("calibration", score_col)

        test_ap = ap_for("test", score_col)

        row = {

            "protocol": protocol_name,

            "detector": detector,

            "group": method_spec["group"],

            "method": method_spec["method"],

            "selection_mode": selection_mode,

            "score_col": score_col,

            "threshold": json.dumps(threshold, ensure_ascii=False) if isinstance(threshold, dict) else threshold,

            "B": metrics_to_row(test_m)["B"],

            "P": metrics_to_row(test_m)["P"],

            "R": metrics_to_row(test_m)["R"],

            "URR": metrics_to_row(test_m)["URR"],

            "UFA": metrics_to_row(test_m)["UFA"],

            "BG_FP": metrics_to_row(test_m)["BG_FP"],

            "AP50": finite(test_ap.get("AP50")),

            "AP75": finite(test_ap.get("AP75")),

            "AP": finite(test_ap.get("AP")),

            "cal_B": metrics_to_row(cal_m)["B"],

            "cal_P": metrics_to_row(cal_m)["P"],

            "cal_R": metrics_to_row(cal_m)["R"],

            "cal_URR": metrics_to_row(cal_m)["URR"],

            "cal_UFA": metrics_to_row(cal_m)["UFA"],

            "cal_BG_FP": metrics_to_row(cal_m)["BG_FP"],

            "cal_AP50": finite(cal_ap.get("AP50")),

            "cal_AP75": finite(cal_ap.get("AP75")),

            "cal_AP": finite(cal_ap.get("AP")),

            "raw_cal_AP50": finite(raw_cal_ap.get("AP50")),

            "raw_cal_AP": finite(raw_cal_ap.get("AP")),

            "meets_cal_AP50_raw": finite(cal_ap.get("AP50")) >= finite(raw_cal_ap.get("AP50")) - 1e-12,

            "meets_cal_AP_raw": finite(cal_ap.get("AP")) >= finite(raw_cal_ap.get("AP")) - 1e-12,

            "accepted_detection_count": metrics_to_row(test_m)["accepted_detection_count"],

            "known_TP_count": metrics_to_row(test_m)["known_TP_count"],

            "known_FP_count": metrics_to_row(test_m)["known_FP_count"],

            "hyperparameters": hyper,

            "per_class_thresholds_json": json.dumps(per_class_thresholds, ensure_ascii=False, sort_keys=True) if per_class_thresholds else "",

            "source_candidate_file": str(candidate_path),

            "notes": "All fitting and threshold/policy selection used calibration split only.",

        }

        append_row(result_path, row)

        done.add(key)

        print(f"[{protocol}] {method_spec['method']} / {selection_mode}: B={row['B']:.4f}, UFA={row['UFA']}, BG={row['BG_FP']}", flush=True)



    for spec in method_specs:

        score_col = spec["score_col"]

        key = (protocol_name, spec["method"], "RF global threshold")

        if key in done:

            print(f"[{protocol}] skipping completed {spec['method']} / RF global threshold", flush=True)

            continue

        print(f"[{protocol}] selecting RF global threshold for {spec['method']}", flush=True)

        thr, cal_m, _ = select_global_threshold(scored, gt, score_col, "calibration")

        test_m = eval_dets(scored, gt, "test", score_col, thr)

        emit(spec, "RF global threshold", thr, cal_m, test_m, score_col, spec["hyperparameters"])



    raw_spec = {"group": "Threshold", "method": "Raw per-class threshold", "score_col": "raw_score", "hyperparameters": "coordinate_ascent=3"}

    per_class_modes = ["RF per-class threshold", "AP-constrained per-class threshold"]

    per_class_keys = [(protocol_name, raw_spec["method"], mode) for mode in per_class_modes]

    if all(key in done for key in per_class_keys):

        print(f"[{protocol}] skipping completed raw per-class thresholds", flush=True)

    else:

        print(f"[{protocol}] selecting raw per-class thresholds", flush=True)

        raw_thr, _, _ = select_global_threshold(scored, gt, "raw_score", "calibration")

        thresholds, cal_m = select_per_class_thresholds(scored, gt, "raw_score", raw_thr, max_iter=1 if protocol == "coco" else 2)

        test_m = eval_per_class_thresholds(scored, gt, "test", "raw_score", thresholds, raw_thr)

        emit(raw_spec, "RF per-class threshold", thresholds, cal_m, test_m, "raw_score", raw_spec["hyperparameters"], thresholds)

        emit(raw_spec, "AP-constrained per-class threshold", thresholds, cal_m, test_m, "raw_score", raw_spec["hyperparameters"] + ";AP rank-preserving raw score", thresholds)



    copy_existing_full_gorc(protocol, result_path, done)


    return read_csv(result_path)



def copy_existing_full_gorc(protocol: str, result_path: Path, done: set[tuple[str, str, str]]) -> None:

    protocol_name = "COCO-Val-OpenSet-5K" if protocol == "coco" else "LVIS-Clear-Mini-300"

    detector = "YOLO-World-l"

    if protocol == "coco":

        path = BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_compact_paper_metrics.csv"

        df = read_csv(path)

        rows = [

            ("GORC-RF", "Existing selected RF", df[df["method"].eq("step12c_geometry_max_cal_balanced")].iloc[0]),

            ("GORC-AP-C", "Existing selected AP-C", df[df["method"].eq("step12c_geometry_ap_constrained_balanced_AP50tol0_APtol0")].iloc[0]),

        ]

    else:

        path = BUNDLE / "data_outputs" / "lvis_main" / "step10a_ap_constrained" / "csv" / "step10a_compact_paper_metrics.csv"

        df = read_csv(path)

        rf = df[df["method"].eq("step9c_geometry_max_cal_balanced")].iloc[0]

        app = df[df["method"].eq("step10a_balanced_max_AP50tol0p005_APtol0p005_blend_alpha_0p200")].iloc[0]

        rows = [("GORC-RF", "Existing selected RF", rf), ("GORC-AP-C", "Existing selected AP-C", app)]

    for method, selection_mode, r in rows:

        key = (protocol_name, method, selection_mode)

        if key in done:

            continue

        row = {

            "protocol": protocol_name,

            "detector": detector,

            "group": "GORC",

            "method": method,

            "selection_mode": selection_mode,

            "score_col": r.get("score_col", ""),

            "threshold": r.get("threshold", ""),

            "B": finite(r.get("balanced", r.get("precision_recall_unknown_balanced_score"))),

            "P": finite(r.get("precision", r.get("known_precision"))),

            "R": finite(r.get("recall", r.get("known_recall"))),

            "URR": finite(r.get("unknown_reject", r.get("unknown_reject_rate_object_level"))),

            "UFA": int(finite(r.get("unknown_false_accepts", r.get("unknown_false_accept_objects")), 0)),

            "BG_FP": int(finite(r.get("background_false_accepts", r.get("background_false_accept_count")), 0)),

            "AP50": finite(r.get("AP50", r.get("unthresholded_AP50"))),

            "AP75": finite(r.get("AP75", r.get("unthresholded_AP75"))),

            "AP": finite(r.get("AP", r.get("unthresholded_AP"))),

            "cal_B": finite(r.get("cal_balanced")),

            "cal_P": finite(r.get("cal_precision")),

            "cal_R": finite(r.get("cal_recall")),

            "cal_URR": finite(r.get("cal_unknown_reject")),

            "cal_UFA": int(finite(r.get("cal_ufa"), 0)) if "cal_ufa" in r.index else "",

            "cal_BG_FP": int(finite(r.get("cal_background_false_accepts"), 0)) if "cal_background_false_accepts" in r.index else "",

            "cal_AP50": finite(r.get("cal_AP50")),

            "cal_AP75": finite(r.get("cal_AP75")),

            "cal_AP": finite(r.get("cal_AP")),

            "raw_cal_AP50": "",

            "raw_cal_AP": "",

            "meets_cal_AP50_raw": "",

            "meets_cal_AP_raw": "",

            "accepted_detection_count": int(finite(r.get("num_accepted_detections"), 0)),

            "known_TP_count": int(finite(r.get("tp_known"), 0)),

            "known_FP_count": int(finite(r.get("fp_known"), 0)),

            "hyperparameters": "Copied from existing compact evaluator output.",

            "per_class_thresholds_json": "",

            "source_candidate_file": str(path),

            "notes": "Existing GORC selected-policy row, included for baseline comparison.",

        }

        append_row(result_path, row)

        done.add(key)

        print(f"[{protocol}] copied {method}: B={row['B']:.4f}, UFA={row['UFA']}, BG={row['BG_FP']}", flush=True)



def write_tables() -> None:

    ensure_dir(TABLE_OUT)

    for protocol in ["coco", "lvis"]:

        path = OUT / f"{protocol}_strong_baselines.csv"

        if not path.exists():

            continue

        df = read_csv(path)

        md_path = OUT / f"{protocol}_strong_baselines.md"

        try:

            md = df.to_markdown(index=False)

        except Exception:

            md = df.to_csv(index=False)

        md_path.write_text(f"# {protocol.upper()} Strong Baselines\n\n{md}\n", encoding="utf-8")

        if protocol == "coco":

            main_cols = ["group", "method", "selection_mode", "B", "P", "R", "URR", "UFA", "BG_FP", "AP50", "AP75", "AP", "cal_B"]

            main = df[main_cols].copy()

            main.to_csv(TABLE_OUT / "table_coco_strong_baselines_main.csv", index=False, encoding="utf-8-sig")

            df.to_csv(TABLE_OUT / "table_coco_strong_baselines_full.csv", index=False, encoding="utf-8-sig")

            try:

                (TABLE_OUT / "table_coco_strong_baselines_main.md").write_text("# COCO Strong Baselines Main Table\n\n" + main.to_markdown(index=False) + "\n", encoding="utf-8")

                (TABLE_OUT / "table_coco_strong_baselines_full.md").write_text("# COCO Strong Baselines Full Table\n\n" + df.to_markdown(index=False) + "\n", encoding="utf-8")

            except Exception:

                (TABLE_OUT / "table_coco_strong_baselines_main.md").write_text(main.to_csv(index=False), encoding="utf-8")

                (TABLE_OUT / "table_coco_strong_baselines_full.md").write_text(df.to_csv(index=False), encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--protocol", choices=["coco", "lvis", "all"], default="all")

    parser.add_argument("--no-resume", action="store_true")

    args = parser.parse_args()

    protocols = ["coco", "lvis"] if args.protocol == "all" else [args.protocol]

    for protocol in protocols:

        run_protocol(protocol, resume=not args.no_resume)

    write_tables()



if __name__ == "__main__":

    main()

