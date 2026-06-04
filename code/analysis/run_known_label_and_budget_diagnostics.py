


"""Known-label, calibration-budget, feature-ablation, and footprint diagnostics.

This script intentionally uses cached detector outputs and existing evaluator
functions. It does not rerun detector inference, retrain a detector, or change
the existing main result files.
"""


from __future__ import annotations


import argparse

import inspect

import json

import math

import re

import sys

import time

from datetime import datetime

from pathlib import Path

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tqdm import tqdm



SCRIPT_DIR = Path(__file__).resolve().parent

if str(SCRIPT_DIR) not in sys.path:

    sys.path.insert(0, str(SCRIPT_DIR))


from step8j_yoloworld_lvis_openvoc_baseline import (

    clean_columns,

    compute_ap_summary,

    evaluate_detections,

    load_annotations,

    load_known_classes,

    load_split,

    safe_class_name,

)

import step12c_geometry_risk_coco_openset as step12c              

import step9c_geometry_risk_lvis300 as step9c              



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

CAL_FEATURE_ROWS = [

    {

        "row_name": "Raw score threshold",

        "kind": "existing_score",

        "score_col": "raw_score",

        "feature_set": "raw_score",

        "selection_mode": "existing_raw_threshold",

    },

    {

        "row_name": "Score-only logistic calibration",

        "kind": "fit_logistic",

        "score_col": "ablation_score_only_logistic",

        "feature_set": "score_only",

        "numeric": ["raw_score", "score_logit"],

        "categorical": [],

        "selection_mode": "max_cal_balanced",

    },

    {

        "row_name": "Score + class identity",

        "kind": "fit_logistic",

        "score_col": "ablation_score_class_logistic",

        "feature_set": "score_class",

        "numeric": ["raw_score", "score_logit"],

        "categorical": ["pred_label"],

        "selection_mode": "max_cal_balanced",

    },

    {

        "row_name": "Score + geometry",

        "kind": "fit_logistic",

        "score_col": "ablation_score_geometry_logistic",

        "feature_set": "score_geometry",

        "numeric": "score_plus_geometry",

        "categorical": [],

        "selection_mode": "max_cal_balanced",

    },

    {

        "row_name": "Score + class identity + geometry",

        "kind": "fit_logistic",

        "score_col": "ablation_score_class_geometry_logistic",

        "feature_set": "score_class_geometry",

        "numeric": "score_plus_geometry",

        "categorical": ["pred_label"],

        "selection_mode": "max_cal_balanced",

    },

]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def read_json(path: Path) -> dict:

    if not path.exists():

        raise FileNotFoundError(path)

    with path.open("r", encoding="utf-8") as f:

        return json.load(f)



def finite(value: Any, default: float = float("nan")) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def fmt(value: Any, digits: int = 4) -> str:

    v = finite(value, float("nan"))

    if not math.isfinite(v):

        return "NA"

    return f"{v:.{digits}f}"



def normalize_split(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    out["split"] = out["split"].astype(str).str.lower()

    return out



def source_location(func: Any) -> str:

    try:

        src = inspect.getsourcefile(func) or "NOT FOUND"

        line = inspect.getsourcelines(func)[1]

        return f"{Path(src).name}:{line}::{func.__name__}"

    except Exception:

        return "NOT FOUND"



def dataset_configs(project_root: Path) -> Dict[str, dict]:

    out = project_root / "outputs"

    return {

        "coco": {

            "dataset": "COCO-Val-OpenSet-5K",

            "detector": "YOLO-World-l",

            "protocol_root": out / "step12a_coco_val_openset_protocol",

            "split_csv": out / "step12a_coco_val_openset_protocol" / "csv" / "step12a_scene_split.csv",

            "ann_dir": out / "step12a_coco_val_openset_protocol" / "annotations",

            "class_map_csv": out / "step12a_coco_val_openset_protocol" / "csv" / "step12a_class_map.csv",

            "raw_pred_csv": out / "step12b_yoloworld_coco_openset_baseline" / "csv" / "step12b_yoloworld_raw_predictions.csv",

            "raw_selected_csv": out / "step12b_yoloworld_coco_openset_baseline" / "csv" / "step12b_selected_calibration_config.csv",

            "raw_processed_csv": out / "step12b_yoloworld_coco_openset_baseline" / "csv" / "step12b_processed_images.csv",

            "raw_compact_csv": out / "step12b_yoloworld_coco_openset_baseline" / "csv" / "step12b_compact_paper_metrics.csv",

            "scored_csv": out / "step12c_geometry_risk_coco_openset" / "csv" / "step12c_geometry_scored_candidates.csv",

            "features_csv": out / "step12c_geometry_risk_coco_openset" / "csv" / "step12c_candidate_features_labeled.csv",

            "policy_grid_csv": out / "step12c_geometry_risk_coco_openset" / "csv" / "step12c_policy_grid.csv",

            "selected_policies_csv": out / "step12c_geometry_risk_coco_openset" / "csv" / "step12c_selected_policies.csv",

            "compact_csv": out / "step12c_geometry_risk_coco_openset" / "csv" / "step12c_compact_paper_metrics.csv",

            "integrity_json": out / "step12c_geometry_risk_coco_openset" / "step12c_integrity_report.json",

            "ablation_csv": project_root / "outputs" / "gorc_revision_ablation" / "coco_yoloworld_l_feature_ablation.csv",

            "ablation_md": project_root / "outputs" / "gorc_revision_ablation" / "table_feature_ablation_coco.md",

            "fast_calibration": True,

            "raw_threshold_col": "conf_thr",

        },

        "lvis": {

            "dataset": "LVIS-Clear-Mini-300",

            "detector": "YOLO-World-l",

            "protocol_root": out / "step9a_lvis_clear_mini_300_protocol",

            "split_csv": out / "step9a_lvis_clear_mini_300_protocol" / "csv" / "step9a_scene_split.csv",

            "ann_dir": out / "step9a_lvis_clear_mini_300_protocol" / "annotations",

            "class_map_csv": out / "step9a_lvis_clear_mini_300_protocol" / "csv" / "step9a_class_map.csv",

            "raw_pred_csv": out / "step9b_yoloworld_lvis300_raw_baseline" / "csv" / "step9b_yoloworld_raw_predictions.csv",

            "raw_selected_csv": out / "step9b_yoloworld_lvis300_raw_baseline" / "csv" / "step9b_selected_calibration_config.csv",

            "raw_processed_csv": out / "step9b_yoloworld_lvis300_raw_baseline" / "csv" / "step9b_processed_images.csv",

            "raw_compact_csv": out / "step9b_yoloworld_lvis300_raw_baseline" / "csv" / "step9b_compact_paper_metrics.csv",

            "scored_csv": out / "step9c_geometry_risk_lvis300" / "csv" / "step9c_geometry_scored_candidates.csv",

            "features_csv": out / "step9c_geometry_risk_lvis300" / "csv" / "step9c_candidate_features_labeled.csv",

            "policy_grid_csv": out / "step9c_geometry_risk_lvis300" / "csv" / "step9c_threshold_search.csv",

            "selected_policies_csv": out / "step9c_geometry_risk_lvis300" / "csv" / "step9c_selected_policies.csv",

            "compact_csv": out / "step9c_geometry_risk_lvis300" / "csv" / "step9c_compact_paper_metrics.csv",

            "integrity_json": out / "step9c_geometry_risk_lvis300" / "step9c_integrity_report.json",

            "ablation_csv": project_root / "outputs" / "gorc_revision_ablation" / "lvis_yoloworld_l_feature_ablation.csv",

            "ablation_md": project_root / "outputs" / "gorc_revision_ablation" / "table_feature_ablation_lvis.md",

            "fast_calibration": False,

            "raw_threshold_col": "conf_thr",

        },

    }



def missing_required(configs: Dict[str, dict]) -> List[Path]:

    required_keys = [

        "split_csv",

        "ann_dir",

        "class_map_csv",

        "raw_pred_csv",

        "raw_selected_csv",

        "scored_csv",

        "features_csv",

        "policy_grid_csv",

        "selected_policies_csv",

        "compact_csv",

    ]

    missing: List[Path] = []

    for cfg in configs.values():

        for key in required_keys:

            p = Path(cfg[key])

            if not p.exists():

                missing.append(p)

    return missing



def write_missing(path: Path, missing: Sequence[Path]) -> None:

    lines = ["# Missing Inputs", "", "The requested no-fabrication rule stopped execution because these required cached files were not found:", ""]

    for p in missing:

        lines.append(f"- `{p}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_file_map(project_root: Path, configs: Dict[str, dict], audit_dir: Path) -> None:

    rows: List[dict] = []

    scripts = {

        "COCO protocol": project_root / "scripts" / "step12a_prepare_coco_val_openset_protocol.py",

        "COCO YOLO-World-l baseline": project_root / "scripts" / "step12b_yoloworld_coco_openset_baseline.py",

        "COCO GORC geometry calibration": project_root / "scripts" / "step12c_geometry_risk_coco_openset.py",

        "LVIS protocol": project_root / "scripts" / "step9a_prepare_lvis_clear_mini_300_protocol.py",

        "LVIS YOLO-World-l baseline": project_root / "scripts" / "step9b_yoloworld_lvis300_raw_baseline.py",

        "LVIS GORC geometry calibration": project_root / "scripts" / "step9c_geometry_risk_lvis300.py",

        "Shared YOLO/Open-set evaluator": project_root / "scripts" / "step8j_yoloworld_lvis_openvoc_baseline.py",

        "GORC revision runner": project_root / "scripts" / "gorc_revision_minimal_experiments.py",

    }

    eval_functions = [

        source_location(evaluate_detections),

        source_location(compute_ap_summary),

        source_location(step12c.exact_ap_for_score),

        source_location(step12c.fast_operating_metrics),

        source_location(step12c.exact_operating),

        source_location(step12c.build_policy_grid),

        source_location(step12c.select_policies),

    ]

    for label, p in scripts.items():

        rows.append({"category": "script", "name": label, "path": str(p), "status": "FOUND" if p.exists() else "MISSING"})

    for cfg in configs.values():

        for key in [

            "raw_pred_csv",

            "features_csv",

            "scored_csv",

            "policy_grid_csv",

            "selected_policies_csv",

            "compact_csv",

            "raw_selected_csv",

            "raw_processed_csv",

            "split_csv",

            "class_map_csv",

        ]:

            p = Path(cfg[key])

            rows.append(

                {

                    "category": f"{cfg['dataset']} {key}",

                    "name": key,

                    "path": str(p),

                    "status": "FOUND" if p.exists() else "MISSING",

                }

            )

    for f in eval_functions:

        rows.append({"category": "evaluation_function", "name": f.split("::")[-1], "path": f, "status": "FOUND" if f != "NOT FOUND" else "NOT FOUND"})

    df = pd.DataFrame(rows)

    csv_path = audit_dir / "file_map.csv"

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    lines = ["# GORC Revision File Map", "", f"Generated at: {datetime.now().isoformat(timespec='seconds')}", ""]

    lines.append(df.to_markdown(index=False))

    (audit_dir / "file_map.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def geometry_numeric_columns(scored: pd.DataFrame) -> List[str]:

    preferred = [

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

    return [c for c in preferred if c in scored.columns]



def build_ablation_model(numeric_cols: Sequence[str], categorical_cols: Sequence[str], random_state: int) -> Pipeline:

    transformers = []

    if numeric_cols:

        transformers.append(("num", StandardScaler(), list(numeric_cols)))

    if categorical_cols:

        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), list(categorical_cols)))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(

        C=0.50,

        class_weight="balanced",

        max_iter=1000,

        solver="liblinear",

        random_state=int(random_state),

    )

    return Pipeline([("pre", pre), ("clf", clf)])



def get_feature_dimension(model: Pipeline) -> int:

    pre = model.named_steps["pre"]

    try:

        return int(len(pre.get_feature_names_out()))

    except Exception:

        clf = model.named_steps["clf"]

        return int(clf.coef_.shape[1])



def sample_weights(train: pd.DataFrame) -> np.ndarray:

    w = np.ones(len(train), dtype=float)

    err = train["risk_error_type"].astype(str)

    w[err.eq("unknown_false_accept").to_numpy()] = 2.0

    w[err.eq("background_false_accept").to_numpy()] = 1.25

    return w



def exact_operating_for_score(scored_split: pd.DataFrame, gt_split: pd.DataFrame, score_col: str, threshold: float, iou: float) -> dict:

    dets = scored_split[pd.to_numeric(scored_split[score_col], errors="coerce").fillna(-1.0) >= float(threshold)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    metrics, _ = evaluate_detections(dets, gt_split, iou_thr=float(iou))

    return metrics



def exact_ap_for_score(scored: pd.DataFrame, gt: pd.DataFrame, known_classes: Sequence[str], split: str, score_col: str) -> dict:

    dets = scored[scored["split"].astype(str).str.lower().eq(split.lower())].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    gt_split = gt[gt["split"].astype(str).str.lower().eq(split.lower())].copy()

    ap, _ = compute_ap_summary(dets, gt_split, known_classes)

    return ap.iloc[0].to_dict()



def calibration_metrics(scored_cal: pd.DataFrame, gt_cal: pd.DataFrame, score_col: str, threshold: float, use_fast: bool) -> dict:

    if use_fast and {"best_known_key", "best_unknown_key"}.issubset(set(scored_cal.columns)):

        return step12c.fast_operating_metrics(scored_cal, gt_cal, score_col, threshold, 0.50)

    return exact_operating_for_score(scored_cal, gt_cal, score_col, threshold, 0.50)



def select_threshold_for_score(scored_cal: pd.DataFrame, gt_cal: pd.DataFrame, score_col: str, use_fast: bool) -> Tuple[float, dict, pd.DataFrame]:

    thresholds = step12c.make_thresholds(scored_cal[score_col])

    rows: List[dict] = []

    for thr in tqdm(thresholds, desc=f"select threshold {score_col}", leave=False):

        m = calibration_metrics(scored_cal, gt_cal, score_col, float(thr), use_fast)

        rows.append(

            {

                "threshold": float(thr),

                "cal_balanced": finite(m.get("precision_recall_unknown_balanced_score")),

                "cal_precision": finite(m.get("known_precision")),

                "cal_recall": finite(m.get("known_recall")),

                "cal_unknown_reject": finite(m.get("unknown_reject_rate_object_level")),

                "cal_ufa": int(m.get("unknown_false_accept_objects", 0)),

                "cal_background_false_accepts": int(m.get("background_false_accept_count", 0)),

                "cal_num_accepted": int(m.get("num_accepted_detections", 0)),

            }

        )

    grid = pd.DataFrame(rows)

    ranked = grid.sort_values(

        ["cal_balanced", "cal_unknown_reject", "cal_precision", "cal_recall", "cal_num_accepted"],

        ascending=[False, False, False, False, True],

    )

    best = ranked.iloc[0].to_dict()

    return float(best["threshold"]), best, grid



def metric_row(

    *,

    dataset: str,

    detector: str,

    row_name: str,

    feature_set: str,

    selection_mode: str,

    score_col: str,

    threshold: float,

    cal_metrics: dict,

    cal_ap: dict,

    test_metrics: dict,

    test_ap: dict,

    raw_ref: dict | None,

    feature_dimension: int | None,

    logistic_parameter_count: int | None,

    source: str,

) -> dict:

    raw_ref = raw_ref or {}

    balanced = finite(test_metrics.get("precision_recall_unknown_balanced_score"))

    precision = finite(test_metrics.get("known_precision"))

    recall = finite(test_metrics.get("known_recall"))

    ufa = int(test_metrics.get("unknown_false_accept_objects", 0))

    bg = int(test_metrics.get("background_false_accept_count", 0))

    return {

        "dataset": dataset,

        "detector": detector,

        "row": row_name,

        "feature_set": feature_set,

        "selection_mode": selection_mode,

        "score_col": score_col,

        "threshold": float(threshold),

        "B": balanced,

        "precision": precision,

        "recall": recall,

        "UFA": ufa,

        "BG_FP": bg,

        "AP50": finite(test_ap.get("AP50")),

        "AP75": finite(test_ap.get("AP75")),

        "AP": finite(test_ap.get("AP")),

        "delta_UFA_vs_raw": ufa - int(raw_ref.get("UFA", ufa)),

        "delta_BG_FP_vs_raw": bg - int(raw_ref.get("BG_FP", bg)),

        "delta_B_vs_raw": balanced - finite(raw_ref.get("B", balanced)),

        "delta_AP50_vs_raw": finite(test_ap.get("AP50")) - finite(raw_ref.get("AP50", test_ap.get("AP50"))),

        "delta_AP_vs_raw": finite(test_ap.get("AP")) - finite(raw_ref.get("AP", test_ap.get("AP"))),

        "cal_B": finite(cal_metrics.get("precision_recall_unknown_balanced_score", cal_metrics.get("cal_balanced"))),

        "cal_precision": finite(cal_metrics.get("known_precision", cal_metrics.get("cal_precision"))),

        "cal_recall": finite(cal_metrics.get("known_recall", cal_metrics.get("cal_recall"))),

        "cal_UFA": int(cal_metrics.get("unknown_false_accept_objects", cal_metrics.get("cal_ufa", 0))),

        "cal_BG_FP": int(cal_metrics.get("background_false_accept_count", cal_metrics.get("cal_background_false_accepts", 0))),

        "cal_AP50": finite(cal_ap.get("AP50")),

        "cal_AP75": finite(cal_ap.get("AP75")),

        "cal_AP": finite(cal_ap.get("AP")),

        "heldout_split": "test",

        "calibration_split": "calibration",

        "feature_dimension": feature_dimension,

        "logistic_parameter_count": logistic_parameter_count,

        "source": source,

    }



def compact_row_to_metric_row(

    cfg: dict,

    compact_row: pd.Series,

    selected_row: pd.Series | None,

    raw_ref: dict,

    label: str,

    feature_set: str,

    source: str,

) -> dict:

    def row_first(row: pd.Series, names: Sequence[str], default: Any = float("nan")) -> Any:

        for name in names:

            if name in row.index:

                val = row.get(name)

                if pd.notna(val):

                    return val

        return default


    if selected_row is not None:

        cal_metrics = {

            "cal_balanced": selected_row.get("cal_balanced", compact_row.get("cal_balanced", float("nan"))),

            "cal_precision": selected_row.get("cal_precision", compact_row.get("cal_precision", float("nan"))),

            "cal_recall": selected_row.get("cal_recall", compact_row.get("cal_recall", float("nan"))),

            "cal_ufa": selected_row.get("cal_ufa", compact_row.get("cal_ufa", 0)),

            "cal_background_false_accepts": selected_row.get(

                "cal_background_false_accepts", selected_row.get("cal_bg_fp", compact_row.get("cal_background_false_accepts", 0))

            ),

        }

        cal_ap = {

            "AP50": selected_row.get("cal_AP50", compact_row.get("cal_AP50", float("nan"))),

            "AP75": selected_row.get("cal_AP75", compact_row.get("cal_AP75", float("nan"))),

            "AP": selected_row.get("cal_AP", compact_row.get("cal_AP", float("nan"))),

        }

    else:

        cal_metrics = {

            "cal_balanced": compact_row.get("cal_balanced", float("nan")),

            "cal_precision": compact_row.get("cal_precision", float("nan")),

            "cal_recall": compact_row.get("cal_recall", float("nan")),

            "cal_ufa": compact_row.get("cal_ufa", 0),

            "cal_background_false_accepts": compact_row.get("cal_background_false_accepts", 0),

        }

        cal_ap = {"AP50": compact_row.get("cal_AP50", float("nan")), "AP75": compact_row.get("cal_AP75", float("nan")), "AP": compact_row.get("cal_AP", float("nan"))}

    test_metrics = {

        "precision_recall_unknown_balanced_score": row_first(

            compact_row, ["balanced", "test_balanced", "precision_recall_unknown_balanced_score"]

        ),

        "known_precision": row_first(compact_row, ["precision", "test_precision", "known_precision"]),

        "known_recall": row_first(compact_row, ["recall", "test_recall", "known_recall"]),

        "unknown_false_accept_objects": row_first(compact_row, ["unknown_false_accept_objects", "test_ufa"], 0),

        "background_false_accept_count": row_first(compact_row, ["background_false_accept_count", "background_false_accepts"], 0),

    }

    test_ap = {

        "AP50": row_first(compact_row, ["AP50", "test_AP50", "unthresholded_AP50"]),

        "AP75": row_first(compact_row, ["AP75", "test_AP75", "unthresholded_AP75"]),

        "AP": row_first(compact_row, ["AP", "test_AP", "unthresholded_AP"]),

    }

    return metric_row(

        dataset=cfg["dataset"],

        detector=cfg["detector"],

        row_name=label,

        feature_set=feature_set,

        selection_mode=str(compact_row.get("selection_policy", selected_row.get("selection_policy") if selected_row is not None else "")),

        score_col=str(compact_row.get("score_col", selected_row.get("score_col") if selected_row is not None else "")),

        threshold=finite(compact_row.get("threshold", compact_row.get("selected_threshold", selected_row.get("threshold") if selected_row is not None else 0.0))),

        cal_metrics=cal_metrics,

        cal_ap=cal_ap,

        test_metrics=test_metrics,

        test_ap=test_ap,

        raw_ref=raw_ref,

        feature_dimension=None,

        logistic_parameter_count=None,

        source=source,

    )



def load_raw_threshold(cfg: dict) -> float:

    selected = read_csv(cfg["raw_selected_csv"])

    if "conf_thr" in selected.columns:

        return float(selected.iloc[0]["conf_thr"])

    if "threshold" in selected.columns:

        return float(selected.iloc[0]["threshold"])

    raise ValueError(f"Cannot find raw threshold in {cfg['raw_selected_csv']}")



def run_feature_ablation_for_dataset(cfg: dict, random_state: int) -> pd.DataFrame:

    scored = normalize_split(read_csv(cfg["scored_csv"]))

    split_df = load_split(Path(cfg["split_csv"]), DEFAULT_PROJECT_ROOT, "all")

    known_classes = load_known_classes(Path(cfg["class_map_csv"]))

    gt = normalize_split(load_annotations(Path(cfg["ann_dir"]), split_df, known_classes))

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    for col in ["raw_score", "score", "score_logit"]:

        if col in scored.columns:

            scored[col] = pd.to_numeric(scored[col], errors="coerce")

    if "raw_score" not in scored.columns and "score" in scored.columns:

        scored["raw_score"] = scored["score"]

    if "score_logit" not in scored.columns:

        s = pd.to_numeric(scored["raw_score"], errors="coerce").fillna(0.0).clip(1e-6, 1 - 1e-6)

        scored["score_logit"] = np.log(s / (1.0 - s))


    cal = scored[scored["split"].eq("calibration")].copy()

    test = scored[scored["split"].eq("test")].copy()

    gt_cal = gt[gt["split"].eq("calibration")].copy()

    gt_test = gt[gt["split"].eq("test")].copy()

    rows: List[dict] = []


    raw_thr = load_raw_threshold(cfg)

    raw_cal_m = calibration_metrics(cal, gt_cal, "raw_score", raw_thr, bool(cfg["fast_calibration"]))

    raw_test_m = exact_operating_for_score(test, gt_test, "raw_score", raw_thr, 0.50)

    raw_cal_ap = exact_ap_for_score(scored, gt, known_classes, "calibration", "raw_score")

    raw_test_ap = exact_ap_for_score(scored, gt, known_classes, "test", "raw_score")

    raw_ref = {

        "B": raw_test_m["precision_recall_unknown_balanced_score"],

        "UFA": raw_test_m["unknown_false_accept_objects"],

        "BG_FP": raw_test_m["background_false_accept_count"],

        "AP50": raw_test_ap.get("AP50"),

        "AP": raw_test_ap.get("AP"),

    }

    rows.append(

        metric_row(

            dataset=cfg["dataset"],

            detector=cfg["detector"],

            row_name="Raw score threshold",

            feature_set="raw_score",

            selection_mode="existing_raw_threshold_from_calibration",

            score_col="raw_score",

            threshold=raw_thr,

            cal_metrics=raw_cal_m,

            cal_ap=raw_cal_ap,

            test_metrics=raw_test_m,

            test_ap=raw_test_ap,

            raw_ref=raw_ref,

            feature_dimension=1,

            logistic_parameter_count=0,

            source=str(cfg["raw_selected_csv"]),

        )

    )


    train = cal.copy()

    y = pd.to_numeric(train["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    geom_cols = geometry_numeric_columns(scored)

    if "raw_score" not in geom_cols:

        geom_cols.insert(0, "raw_score")

    if "score_logit" not in geom_cols:

        geom_cols.insert(1, "score_logit")


    for spec in CAL_FEATURE_ROWS[1:]:

        numeric = geom_cols if spec["numeric"] == "score_plus_geometry" else list(spec["numeric"])

        numeric = [c for c in numeric if c in scored.columns]

        categorical = [c for c in spec["categorical"] if c in scored.columns]

        model = build_ablation_model(numeric, categorical, random_state)

        model.fit(train, y, clf__sample_weight=sample_weights(train))

        score_col = str(spec["score_col"])

        scored[score_col] = model.predict_proba(scored)[:, 1].astype(float)

        cal = scored[scored["split"].eq("calibration")].copy()

        test = scored[scored["split"].eq("test")].copy()

        selected_thr, cal_best, _ = select_threshold_for_score(cal, gt_cal, score_col, bool(cfg["fast_calibration"]))

        test_m = exact_operating_for_score(test, gt_test, score_col, selected_thr, 0.50)

        cal_ap = exact_ap_for_score(scored, gt, known_classes, "calibration", score_col)

        test_ap = exact_ap_for_score(scored, gt, known_classes, "test", score_col)

        dim = get_feature_dimension(model)

        rows.append(

            metric_row(

                dataset=cfg["dataset"],

                detector=cfg["detector"],

                row_name=str(spec["row_name"]),

                feature_set=str(spec["feature_set"]),

                selection_mode=str(spec["selection_mode"]),

                score_col=score_col,

                threshold=selected_thr,

                cal_metrics=cal_best,

                cal_ap=cal_ap,

                test_metrics=test_m,

                test_ap=test_ap,

                raw_ref=raw_ref,

                feature_dimension=dim,

                logistic_parameter_count=dim + 1,

                source="scripts/gorc_revision_minimal_experiments.py::run_feature_ablation_for_dataset",

            )

        )


    compact = read_csv(cfg["compact_csv"])

    selected = read_csv(cfg["selected_policies_csv"])


    if cfg["dataset"].startswith("COCO"):

        full_mask = compact["method"].astype(str).eq("step12c_geometry_max_cal_balanced")

        class_mask = compact["method"].astype(str).eq("step12c_geometry_max_cal_ap")

        selected_key = "selection_policy"

    else:

        full_mask = compact["selection_policy"].astype(str).eq("max_cal_balanced")

        class_mask = compact["selection_policy"].astype(str).eq("max_cal_ap")

        selected_key = "selection_policy"

    if full_mask.any():

        full = compact[full_mask].iloc[0]

        sel = None

        if selected_key in selected.columns:

            srows = selected[selected[selected_key].astype(str).eq(str(full.get("selection_policy", "max_cal_balanced")))]

            sel = srows.iloc[0] if not srows.empty else None

        rows.append(compact_row_to_metric_row(cfg, full, sel, raw_ref, "GORC selected policy", "full_gorc", str(cfg["compact_csv"])))

    if class_mask.any():

        cls = compact[class_mask].iloc[0]

        sel = None

        if selected_key in selected.columns:

            srows = selected[selected[selected_key].astype(str).eq(str(cls.get("selection_policy", "max_cal_ap")))]

            sel = srows.iloc[0] if not srows.empty else None

        rows.append(compact_row_to_metric_row(cfg, cls, sel, raw_ref, "Class-level scaling only", "class_level_scaling_only", str(cfg["compact_csv"])))


    out = pd.DataFrame(rows)

    return out



def write_ablation_outputs(df: pd.DataFrame, csv_path: Path, md_path: Path) -> None:

    ensure_dir(csv_path.parent)

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    table_cols = [

        "row",

        "selection_mode",

        "threshold",

        "B",

        "precision",

        "recall",

        "UFA",

        "BG_FP",

        "AP50",

        "AP75",

        "AP",

        "delta_UFA_vs_raw",

        "delta_BG_FP_vs_raw",

        "cal_B",

        "cal_AP50",

        "cal_AP",

    ]

    table = df[table_cols].copy()

    for c in ["threshold", "B", "precision", "recall", "AP50", "AP75", "AP", "cal_B", "cal_AP50", "cal_AP"]:

        table[c] = table[c].map(lambda x: fmt(x, 4))

    lines = [

        f"# Feature Ablation: {df.iloc[0]['dataset']} / {df.iloc[0]['detector']}",

        "",

        "All fitting and threshold/policy selection used the calibration split only. Held-out metrics are from the test split.",

        "",

        table.to_markdown(index=False),

    ]

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_policy_grid(project_root: Path, configs: Dict[str, dict], audit_dir: Path) -> None:

    rows: List[dict] = []

    for key, cfg in configs.items():

        integrity = read_json(cfg["integrity_json"]) if Path(cfg["integrity_json"]).exists() else {}

        policy_grid = read_csv(cfg["policy_grid_csv"]) if Path(cfg["policy_grid_csv"]).exists() else pd.DataFrame()

        selected = read_csv(cfg["selected_policies_csv"]) if Path(cfg["selected_policies_csv"]).exists() else pd.DataFrame()

        if key == "coco":

            rows.extend(

                [

                    {

                        "protocol": cfg["dataset"],

                        "item": "feature standardization",

                        "value": "StandardScaler on numeric features; OneHotEncoder(handle_unknown='ignore') on pred_label",

                        "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::build_model",

                    },

                    {

                        "protocol": cfg["dataset"],

                        "item": "logistic regularization lambda",

                        "value": "LogisticRegression(C=0.50); inverse-L2 C implies lambda=2.0 under lambda=1/C convention",

                        "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::build_model",

                    },

                    {"protocol": cfg["dataset"], "item": "alpha search values", "value": str(integrity.get("alphas", "NOT FOUND")), "defined_in": "step12c_integrity_report.json / parser --alphas"},

                    {"protocol": cfg["dataset"], "item": "gamma search values", "value": str(integrity.get("gammas", "NOT FOUND")), "defined_in": "step12c_integrity_report.json / parser --gammas"},

                    {

                        "protocol": cfg["dataset"],

                        "item": "threshold candidates",

                        "value": f"BASE_THRESHOLDS={step12c.BASE_THRESHOLDS}; plus make_thresholds score quantiles/max; actual unique grid thresholds={policy_grid['threshold'].nunique() if 'threshold' in policy_grid.columns else 'NOT FOUND'}",

                        "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::BASE_THRESHOLDS, make_thresholds",

                    },

                    {"protocol": cfg["dataset"], "item": "AP-constrained epsilon AP", "value": str(step12c.AP_DROP_TOLERANCES), "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::AP_DROP_TOLERANCES"},

                    {"protocol": cfg["dataset"], "item": "AP-constrained epsilon AP50", "value": str(step12c.AP_DROP_TOLERANCES), "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::AP_DROP_TOLERANCES"},

                    {"protocol": cfg["dataset"], "item": "AP75 used in selection", "value": "No; AP75 is reported but not used by select_policies sort/filter keys", "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::select_policies"},

                    {"protocol": cfg["dataset"], "item": "RF objective", "value": "maximize cal_balanced; tie: cal_unknown_reject, cal_precision, cal_recall, lower cal_num_accepted", "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::select_policies"},

                    {"protocol": cfg["dataset"], "item": "AP-C objective", "value": "filter cal_AP50 >= raw_cal_AP50 - eps50 and cal_AP >= raw_cal_AP - epsAP; maximize cal_balanced; tie: cal_unknown_reject, cal_precision, cal_recall, lower cal_background_false_accepts, lower cal_num_accepted", "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::select_policies"},

                    {"protocol": cfg["dataset"], "item": "tie-breaking rules", "value": "max_cal_ap tie: cal_AP, cal_AP50, cal_balanced, cal_precision, cal_unknown_reject; precision_tiebreak chooses max precision within eps of max cal_balanced", "defined_in": "scripts/step12c_geometry_risk_coco_openset.py::select_policies"},

                ]

            )

        else:

            rows.extend(

                [

                    {"protocol": cfg["dataset"], "item": "feature standardization", "value": "StandardScaler on numeric features; OneHotEncoder(handle_unknown='ignore') on pred_label", "defined_in": "scripts/step9c_geometry_risk_lvis300.py::build_model"},

                    {"protocol": cfg["dataset"], "item": "logistic regularization lambda", "value": "LogisticRegression(C=0.50); inverse-L2 C implies lambda=2.0 under lambda=1/C convention", "defined_in": "scripts/step9c_geometry_risk_lvis300.py::build_model"},

                    {"protocol": cfg["dataset"], "item": "alpha search values", "value": str(integrity.get("alphas", "NOT FOUND")), "defined_in": "step9c_integrity_report.json / parser --alphas"},

                    {"protocol": cfg["dataset"], "item": "gamma search values", "value": "NOT FOUND", "defined_in": "scripts/step9c_geometry_risk_lvis300.py"},

                    {"protocol": cfg["dataset"], "item": "threshold candidates", "value": f"base thresholds from integrity={integrity.get('thresholds', 'NOT FOUND')}; plus make_thresholds quantiles; actual rows={len(policy_grid)}", "defined_in": "scripts/step9c_geometry_risk_lvis300.py::make_thresholds"},

                    {"protocol": cfg["dataset"], "item": "AP-constrained epsilon AP", "value": "NOT FOUND in Step9C", "defined_in": "scripts/step9c_geometry_risk_lvis300.py"},

                    {"protocol": cfg["dataset"], "item": "AP-constrained epsilon AP50", "value": "NOT FOUND in Step9C", "defined_in": "scripts/step9c_geometry_risk_lvis300.py"},

                    {"protocol": cfg["dataset"], "item": "AP75 used in selection", "value": "No; AP75 reported but selected_policy_rows sorts by cal_AP/cal_AP50 and reliability fields", "defined_in": "scripts/step9c_geometry_risk_lvis300.py::selected_policy_rows"},

                    {"protocol": cfg["dataset"], "item": "RF objective", "value": "maximize cal_balanced; tie: cal_unknown_reject, cal_recall, cal_precision", "defined_in": "scripts/step9c_geometry_risk_lvis300.py::selected_policy_rows"},

                    {"protocol": cfg["dataset"], "item": "AP-C objective", "value": "NOT FOUND in Step9C; Step10A introduced strict AP-constrained policy", "defined_in": "scripts/step9c_geometry_risk_lvis300.py"},

                    {"protocol": cfg["dataset"], "item": "tie-breaking rules", "value": "precision_tiebreak within eps of max cal_balanced; max_cal_ap sorts cal_AP, cal_AP50, cal_balanced, cal_precision", "defined_in": "scripts/step9c_geometry_risk_lvis300.py::selected_policy_rows"},

                ]

            )

        if not selected.empty:

            rows.append({"protocol": cfg["dataset"], "item": "selected policy rows", "value": ", ".join(map(str, selected.get("selection_policy", pd.Series(dtype=str)).dropna().unique()[:10])), "defined_in": str(cfg["selected_policies_csv"])})

    df = pd.DataFrame(rows)

    df.to_csv(audit_dir / "policy_grid_extracted.csv", index=False, encoding="utf-8-sig")

    lines = ["# Policy Grid and Hyperparameter Audit", "", "Values are read from scripts and existing integrity/grid outputs; missing values are marked NOT FOUND.", "", df.to_markdown(index=False)]

    (audit_dir / "table_policy_grid.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def measure_footprint(configs: Dict[str, dict], audit_dir: Path, random_state: int) -> pd.DataFrame:

    rows: List[dict] = []

    for cfg in configs.values():

        scored = normalize_split(read_csv(cfg["scored_csv"]))

        processed = read_csv(cfg["raw_processed_csv"]) if Path(cfg["raw_processed_csv"]).exists() else pd.DataFrame()

        train = scored[scored["split"].eq("calibration")].copy()

        numeric = geometry_numeric_columns(scored)

        categorical = ["pred_label"] if "pred_label" in scored.columns else []

        model = build_ablation_model(numeric, categorical, random_state)

        y = pd.to_numeric(train["risk_label_tp"], errors="coerce").fillna(0).astype(int)

        model.fit(train, y, clf__sample_weight=sample_weights(train))

        dim = get_feature_dimension(model)

        start = time.perf_counter()

        _ = model.predict_proba(scored)[:, 1]

        scoring_seconds = time.perf_counter() - start

        start = time.perf_counter()

        _ = scored[pd.to_numeric(scored.get("geometry_alpha_0p800", scored["raw_score"]), errors="coerce").fillna(0.0) >= 0.7]

        filter_seconds = time.perf_counter() - start

        split_df = load_split(Path(cfg["split_csv"]), DEFAULT_PROJECT_ROOT, "all")

        all_image_ids = split_df["image_id"].astype(str).tolist()

        per_image = scored.groupby("image_id", sort=False).size().reindex(all_image_ids, fill_value=0)

        detector_seconds = float(processed["seconds"].sum()) if "seconds" in processed.columns else float("nan")

        detector_time_available = bool(math.isfinite(detector_seconds) and detector_seconds > 0)

        post_seconds = scoring_seconds + filter_seconds

        rows.append(

            {

                "dataset": cfg["dataset"],

                "detector": cfg["detector"],

                "num_images": int(len(all_image_ids)),

                "num_cached_candidates": int(len(scored)),

                "average_candidate_detections_per_image": float(per_image.mean()),

                "median_candidate_detections_per_image": float(per_image.median()),

                "feature_dimension_full_gorc": int(dim),

                "logistic_model_parameter_count": int(dim + 1),

                "gorc_scoring_time_seconds": float(scoring_seconds),

                "threshold_filter_time_seconds": float(filter_seconds),

                "post_processing_time_seconds": float(post_seconds),

                "post_processing_time_per_image_ms": float(post_seconds / max(per_image.shape[0], 1) * 1000.0),

                "detector_inference_time_seconds_available": detector_time_available,

                "detector_inference_time_seconds": detector_seconds if detector_time_available else float("nan"),

                "relative_postprocessing_over_detector": float(post_seconds / detector_seconds) if detector_time_available else float("nan"),

                "source_cached_candidates": str(cfg["scored_csv"]),

                "source_detector_timing": str(cfg["raw_processed_csv"]),

            }

        )

    df = pd.DataFrame(rows)

    df.to_csv(audit_dir / "computational_footprint.csv", index=False, encoding="utf-8-sig")

    table = df.copy()

    for c in [

        "average_candidate_detections_per_image",

        "median_candidate_detections_per_image",

        "gorc_scoring_time_seconds",

        "threshold_filter_time_seconds",

        "post_processing_time_seconds",

        "post_processing_time_per_image_ms",

        "detector_inference_time_seconds",

        "relative_postprocessing_over_detector",

    ]:

        table[c] = table[c].map(lambda x: fmt(x, 6))

    lines = [

        "# Computational Footprint",

        "",

        "Detector timings are reported only when cached `processed_images.csv` contains per-image seconds. No detector time is estimated.",

        "",

        table.to_markdown(index=False),

    ]

    (audit_dir / "table_computational_footprint.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return df



def write_ablation_summary(ablation_dir: Path, outputs: List[Path]) -> None:

    lines = ["# GORC Revision Ablation Summary", "", f"Generated at: {datetime.now().isoformat(timespec='seconds')}", ""]

    for p in outputs:

        lines.append(f"- `{p}`")

    (ablation_dir / "gorc_revision_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_audit_summary(audit_dir: Path, outputs: List[Path]) -> None:

    lines = ["# GORC Revision Audit Summary", "", f"Generated at: {datetime.now().isoformat(timespec='seconds')}", ""]

    for p in outputs:

        lines.append(f"- `{p}`")

    (audit_dir / "gorc_revision_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--random_state", type=int, default=52)

    parser.add_argument("--skip_ablation_if_exists", action="store_true")

    args = parser.parse_args()


    project_root = args.project_root.resolve()

    audit_dir = project_root / "outputs" / "gorc_revision_audit"

    ablation_dir = project_root / "outputs" / "gorc_revision_ablation"

    ensure_dir(audit_dir)

    ensure_dir(ablation_dir)

    configs = dataset_configs(project_root)


    missing = missing_required(configs)

    if missing:

        write_missing(audit_dir / "MISSING_INPUTS.md", missing)

        write_missing(ablation_dir / "MISSING_INPUTS.md", missing)

        print(f"Missing {len(missing)} required inputs; wrote MISSING_INPUTS.md")

        return


    write_file_map(project_root, configs, audit_dir)

    write_policy_grid(project_root, configs, audit_dir)


    ablation_outputs: List[Path] = []

    for cfg in configs.values():

        if args.skip_ablation_if_exists and Path(cfg["ablation_csv"]).exists() and Path(cfg["ablation_md"]).exists():

            print(f"Skipping existing ablation: {cfg['ablation_csv']}")

        else:

            print(f"Running ablation: {cfg['dataset']}")

            df = run_feature_ablation_for_dataset(cfg, args.random_state)

            write_ablation_outputs(df, Path(cfg["ablation_csv"]), Path(cfg["ablation_md"]))

        ablation_outputs.extend([Path(cfg["ablation_csv"]), Path(cfg["ablation_md"])])


    footprint_df = measure_footprint(configs, audit_dir, args.random_state)

    audit_outputs = [

        audit_dir / "file_map.md",

        audit_dir / "file_map.csv",

        audit_dir / "table_policy_grid.md",

        audit_dir / "policy_grid_extracted.csv",

        audit_dir / "table_computational_footprint.md",

        audit_dir / "computational_footprint.csv",

        audit_dir / "remove_85scene_from_main_text.md",

    ]

    write_ablation_summary(ablation_dir, ablation_outputs)

    write_audit_summary(audit_dir, audit_outputs)


    ablation_integrity = {

        "method": "GORC revision feature ablation",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(project_root),

        "rules": {

            "uses_cached_detector_outputs": True,

            "reruns_detector": False,

            "calibration_only_model_fitting": True,

            "test_split_final_evaluation_only": True,

            "does_not_modify_existing_main_results": True,

        },

        "outputs": [str(p) for p in ablation_outputs] + [str(ablation_dir / "gorc_revision_ablation_summary.md")],

    }

    (ablation_dir / "integrity_report.json").write_text(json.dumps(ablation_integrity, indent=2), encoding="utf-8")


    audit_integrity = {

        "method": "GORC revision audit and footprint",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(project_root),

        "rules": {

            "uses_cached_detector_outputs": True,

            "no_fabricated_results": True,

            "missing_inputs_file_written_if_blocked": True,

        },

        "footprint_rows": int(len(footprint_df)),

        "outputs": [str(p) for p in audit_outputs] + [str(audit_dir / "gorc_revision_audit_summary.md")],

    }

    (audit_dir / "integrity_report.json").write_text(json.dumps(audit_integrity, indent=2), encoding="utf-8")

    print("GORC revision audit and ablation completed.")



if __name__ == "__main__":

    main()
