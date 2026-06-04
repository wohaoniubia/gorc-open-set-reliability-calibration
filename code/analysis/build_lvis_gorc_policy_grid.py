


"""
Step9c: Geometry-aware open-set reliability calibration on LVIS-Clear-Mini-300.

This is the Step8s method definition migrated to the larger protocol:
raw detector score + class-conditioned geometry -> calibration-only reliability
model -> calibration-only alpha/threshold selection -> held-out test evaluation.
"""


from __future__ import annotations


import argparse

import json

import math

import sys

from pathlib import Path

from typing import List, Sequence, Tuple


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

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_SPLIT_CSV = DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv"

DEFAULT_ANN_DIR = DEFAULT_PROTOCOL_ROOT / "annotations"

DEFAULT_CLASS_MAP_CSV = DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv"

DEFAULT_RAW_PRED_CSV = DEFAULT_PROJECT_ROOT / "outputs" / "step9b_yoloworld_lvis300_raw_baseline" / "csv" / "step9b_yoloworld_raw_predictions.csv"

DEFAULT_STEP9B_SELECTED = DEFAULT_PROJECT_ROOT / "outputs" / "step9b_yoloworld_lvis300_raw_baseline" / "csv" / "step9b_selected_calibration_config.csv"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300"


GEOMETRY_NUMERIC = [

    "candidate_score",

    "score_logit",

    "box_area_norm",

    "box_aspect_log",

    "center_x_norm",

    "center_y_norm",

]

GEOMETRY_CATEGORICAL = ["pred_label"]

EVAL_IOUS = [0.50, 0.75]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def parse_float_list(text: str) -> List[float]:

    return [float(x.strip()) for x in str(text).split(",") if x.strip()]



def make_thresholds(values: pd.Series, base_thresholds: Sequence[float]) -> List[float]:

    vals = set(float(x) for x in base_thresholds)

    clean = pd.to_numeric(values, errors="coerce").dropna()

    if len(clean):

        for q in np.linspace(0.05, 0.95, 19):

            vals.add(float(clean.quantile(q)))

    return sorted(v for v in vals if math.isfinite(v))



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



def feature_candidates(raw_pred: pd.DataFrame, split_df: pd.DataFrame, nms_iou: float, nms_contain: float, nms_mode: str) -> pd.DataFrame:

    raw = clean_columns(raw_pred).copy()

    for c in ["score", "x1", "y1", "x2", "y2"]:

        raw[c] = pd.to_numeric(raw[c], errors="coerce")

    raw = raw.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)

    raw["image_id"] = raw["image_id"].map(normalize_image_id)

    raw["pred_label"] = raw["pred_label"].map(safe_class_name)

    raw = raw.sort_values("score", ascending=False).reset_index(drop=True)

    cand = nms_detections(raw, iou_thr=float(nms_iou), contain_thr=float(nms_contain), mode=nms_mode).copy()


    size_df = split_df.copy()

    size_df["image_id"] = size_df["image_id"].map(normalize_image_id)

    size_map = {

        r["image_id"]: (float(r.get("width", 0) or 0), float(r.get("height", 0) or 0))

        for _, r in size_df.iterrows()

    }


    rows = []

    for idx, r in cand.reset_index(drop=True).iterrows():

        image_id = normalize_image_id(r["image_id"])

        width, height = size_map.get(image_id, (0.0, 0.0))

        box_w = max(0.0, float(r.x2) - float(r.x1))

        box_h = max(0.0, float(r.y2) - float(r.y1))

        area = box_w * box_h

        img_area = width * height if width > 0 and height > 0 else 0.0

        eps_box = 1e-6

        log_aspect = math.log(max(box_w, eps_box) / max(box_h, eps_box))

        score = min(max(float(r.score), 1e-6), 1.0 - 1e-6)

        row = r.to_dict()

        row.update(

            {

                "det_id": row.get("det_id", f"{image_id}_step9c_{idx:06d}"),

                "candidate_score": float(r.score),

                "raw_score": float(r.score),

                "score_logit": float(math.log(score / (1.0 - score))),

                "box_area_norm": float(area / img_area) if img_area > 0 else 0.0,

                "box_aspect_log": float(log_aspect),

                "center_x_norm": float((float(r.x1) + float(r.x2)) * 0.5 / width) if width > 0 else 0.5,

                "center_y_norm": float((float(r.y1) + float(r.y2)) * 0.5 / height) if height > 0 else 0.5,

            }

        )

        rows.append(row)

    return pd.DataFrame(rows)



def label_candidates(candidates: pd.DataFrame, gt: pd.DataFrame, label_iou: float) -> pd.DataFrame:

    out = candidates.copy().sort_values("candidate_score", ascending=False).reset_index(drop=True)

    gt_by_image = {img: g.copy() for img, g in gt.groupby("image_id", sort=False)}

    matched_known = set()

    labels: List[int] = []

    error_types: List[str] = []

    best_known_ious: List[float] = []

    best_unknown_ious: List[float] = []

    best_known_labels: List[str] = []

    best_unknown_labels: List[str] = []

    for _, pr in out.iterrows():

        image_id = normalize_image_id(pr.image_id)

        pred_label = safe_class_name(pr.pred_label)

        pbox = (float(pr.x1), float(pr.y1), float(pr.x2), float(pr.y2))

        gimg = gt_by_image.get(image_id, gt.iloc[0:0])

        best_known_iou, best_known_key, best_known_label = 0.0, None, ""

        for _, gr in gimg[gimg["gt_is_known"]].iterrows():

            key = (gr.image_id, gr.gt_id)

            if key in matched_known:

                continue

            iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

            if iou > best_known_iou:

                best_known_iou = float(iou)

                best_known_key = key

                best_known_label = safe_class_name(gr.gt_label)

        best_unknown_iou, best_unknown_label = 0.0, ""

        for _, gr in gimg[~gimg["gt_is_known"]].iterrows():

            iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

            if iou > best_unknown_iou:

                best_unknown_iou = float(iou)

                best_unknown_label = safe_class_name(gr.gt_label)

        if best_known_key is not None and best_known_iou >= float(label_iou) and pred_label == best_known_label:

            labels.append(1)

            error_types.append("known_tp")

            matched_known.add(best_known_key)

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

    out["risk_label_tp"] = labels

    out["risk_error_type"] = error_types

    out["best_known_iou"] = best_known_ious

    out["best_unknown_iou"] = best_unknown_ious

    out["best_known_label"] = best_known_labels

    out["best_unknown_label"] = best_unknown_labels

    return out



def evaluate_threshold(scored: pd.DataFrame, gt_split: pd.DataFrame, score_col: str, threshold: float, iou_threshold: float) -> dict:

    dets = scored[pd.to_numeric(scored[score_col], errors="coerce") >= float(threshold)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    metrics, _ = evaluate_detections(dets, gt_split, iou_thr=float(iou_threshold))

    return metrics



def ap_for_split(scored: pd.DataFrame, gt: pd.DataFrame, known_classes: Sequence[str], split: str, score_col: str) -> dict:

    dets = scored[scored["split"].astype(str).str.lower() == split.lower()].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    gt_split = gt[gt["split"].astype(str).str.lower() == split.lower()].copy()

    ap, _ = compute_ap_summary(dets, gt_split, known_classes)

    return ap.iloc[0].to_dict()



def select_for_alpha(scored_cal: pd.DataFrame, gt_cal: pd.DataFrame, score_col: str, thresholds: Sequence[float], selection_iou: float) -> Tuple[float, dict, pd.DataFrame]:

    rows: List[dict] = []

    for thr in thresholds:

        m = evaluate_threshold(scored_cal, gt_cal, score_col, thr, selection_iou)

        m.update({"threshold": float(thr), "iou_threshold": float(selection_iou)})

        rows.append(m)

    df = pd.DataFrame(rows)

    ranked = df.sort_values(

        [

            "precision_recall_unknown_balanced_score",

            "unknown_reject_rate_object_level",

            "known_recall",

            "known_precision",

            "num_accepted_detections",

        ],

        ascending=[False, False, False, False, True],

    ).reset_index(drop=True)

    return float(ranked.iloc[0]["threshold"]), ranked.iloc[0].to_dict(), df



def selected_policy_rows(summary: pd.DataFrame, tolerance: float) -> pd.DataFrame:

    rows = []

    by_bal = summary.sort_values(

        ["cal_balanced", "cal_unknown_reject", "cal_recall", "cal_precision"],

        ascending=[False, False, False, False],

    ).iloc[0].to_dict()

    by_bal["selection_policy"] = "max_cal_balanced"

    rows.append(by_bal)


    max_bal = float(summary["cal_balanced"].max())

    near = summary[summary["cal_balanced"] >= max_bal - float(tolerance)].copy()

    by_precision = near.sort_values(

        ["cal_precision", "cal_unknown_reject", "cal_balanced", "cal_recall", "cal_num_accepted"],

        ascending=[False, False, False, False, True],

    ).iloc[0].to_dict()

    by_precision["selection_policy"] = f"precision_tiebreak_eps{str(tolerance).replace('.', 'p')}"

    rows.append(by_precision)


    by_ap = summary.sort_values(

        ["cal_AP", "cal_AP50", "cal_balanced", "cal_precision"],

        ascending=[False, False, False, False],

    ).iloc[0].to_dict()

    by_ap["selection_policy"] = "max_cal_ap"

    rows.append(by_ap)

    return pd.DataFrame(rows).drop_duplicates(subset=["selection_policy", "alpha", "selected_threshold"])



def write_markdown(path: Path, compact: pd.DataFrame, selected: pd.DataFrame, diagnostics: dict) -> None:

    lines = [

        "# Step9c Geometry-Risk LVIS-300 Summary",

        "",

        f"- Calibration candidates: `{diagnostics['calibration_num_candidates']}`",

        f"- Calibration positives: `{diagnostics['calibration_positive']}`",

        f"- Calibration negatives: `{diagnostics['calibration_negative']}`",

        f"- Geometry risk AP on calibration labels: `{diagnostics['risk_average_precision']:.4f}`",

        "",

        "## Selected Policies",

        "",

        selected.to_markdown(index=False),

        "",

        "## Held-Out Test Compact Metrics",

        "",

        compact.to_markdown(index=False),

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--raw_pred_csv", type=Path, default=DEFAULT_RAW_PRED_CSV)

    parser.add_argument("--step9b_selected_config", type=Path, default=DEFAULT_STEP9B_SELECTED)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_SPLIT_CSV)

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_ANN_DIR)

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_CLASS_MAP_CSV)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--alphas", type=str, default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")

    parser.add_argument("--thresholds", type=str, default="0.01,0.02,0.03,0.05,0.08,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.60,0.70,0.80,0.90")

    parser.add_argument("--selection_iou", type=float, default=0.50)

    parser.add_argument("--label_iou", type=float, default=0.50)

    parser.add_argument("--nms_iou", type=float, default=0.50)

    parser.add_argument("--nms_contain", type=float, default=0.90)

    parser.add_argument("--nms_mode", type=str, choices=["class", "global"], default="class")

    parser.add_argument("--precision_tiebreak_tolerance", type=float, default=0.005)

    parser.add_argument("--random_state", type=int, default=7)

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    print("\n========== Step9c Geometry-Risk LVIS-300 ==========")

    print(f"raw_pred_csv={args.raw_pred_csv}")


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root)

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    gt_cal = gt[gt["split"].astype(str).str.lower() == "calibration"].copy()

    gt_test = gt[gt["split"].astype(str).str.lower() == "test"].copy()

    raw = pd.read_csv(args.raw_pred_csv, encoding="utf-8-sig")

    candidates = feature_candidates(raw, split_df, args.nms_iou, args.nms_contain, args.nms_mode)

    labeled = label_candidates(candidates, gt, args.label_iou)

    labeled.to_csv(csv_dir / "step9c_candidate_features_labeled.csv", index=False, encoding="utf-8-sig")


    train = labeled[labeled["split"].astype(str).str.lower().eq("calibration")].copy()

    y = pd.to_numeric(train["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:

        raise ValueError("Calibration labels contain one class only.")

    model = build_model(args.random_state)

    sample_weight = np.ones(len(train), dtype=float)

    sample_weight[train["risk_error_type"].eq("unknown_false_accept").to_numpy()] = 2.0

    model.fit(train, y, clf__sample_weight=sample_weight)


    scored = labeled.copy()

    scored["raw_score"] = pd.to_numeric(scored["candidate_score"], errors="coerce").fillna(0.0)

    scored["geometry_risk_score"] = model.predict_proba(scored)[:, 1].astype(float)


    train_probs = scored[scored["split"].astype(str).str.lower().eq("calibration")]["geometry_risk_score"].to_numpy()

    diagnostics = {

        "calibration_num_candidates": int(len(train)),

        "calibration_positive": int(y.sum()),

        "calibration_negative": int((1 - y).sum()),

        "calibration_unknown_negative": int(train["risk_error_type"].eq("unknown_false_accept").sum()),

        "risk_average_precision": float(average_precision_score(y.to_numpy(), train_probs)),

    }

    try:

        diagnostics["risk_roc_auc"] = float(roc_auc_score(y.to_numpy(), train_probs))

    except Exception:

        diagnostics["risk_roc_auc"] = float("nan")


    alphas = parse_float_list(args.alphas)

    base_thresholds = parse_float_list(args.thresholds)

    summary_rows: List[dict] = []

    threshold_rows: List[pd.DataFrame] = []

    for alpha in tqdm(alphas, desc="Step9c alpha sweep"):

        score_col = f"geometry_alpha_{alpha:.3f}".replace(".", "p")

        scored[score_col] = float(alpha) * scored["geometry_risk_score"] + (1.0 - float(alpha)) * scored["raw_score"]

        scored_cal = scored[scored["split"].astype(str).str.lower().eq("calibration")].copy()

        thresholds = make_thresholds(scored_cal[score_col], base_thresholds)

        selected_thr, cal_best, thr_df = select_for_alpha(scored_cal, gt_cal, score_col, thresholds, args.selection_iou)

        thr_df.insert(0, "alpha", float(alpha))

        thr_df.insert(1, "score_col", score_col)

        threshold_rows.append(thr_df)

        test_iou50 = evaluate_threshold(scored[scored["split"].astype(str).str.lower().eq("test")].copy(), gt_test, score_col, selected_thr, args.selection_iou)

        cal_ap = ap_for_split(scored, gt, known_classes, "calibration", score_col)

        test_ap = ap_for_split(scored, gt, known_classes, "test", score_col)

        summary_rows.append(

            {

                "alpha": float(alpha),

                "score_col": score_col,

                "selected_threshold": float(selected_thr),

                "cal_balanced": float(cal_best["precision_recall_unknown_balanced_score"]),

                "cal_precision": float(cal_best["known_precision"]),

                "cal_recall": float(cal_best["known_recall"]),

                "cal_unknown_reject": float(cal_best["unknown_reject_rate_object_level"]),

                "cal_ufa": int(cal_best["unknown_false_accept_objects"]),

                "cal_num_accepted": int(cal_best["num_accepted_detections"]),

                "cal_AP50": float(cal_ap["AP50"]),

                "cal_AP75": float(cal_ap["AP75"]),

                "cal_AP": float(cal_ap["AP"]),

                "test_balanced": float(test_iou50["precision_recall_unknown_balanced_score"]),

                "test_precision": float(test_iou50["known_precision"]),

                "test_recall": float(test_iou50["known_recall"]),

                "test_unknown_reject": float(test_iou50["unknown_reject_rate_object_level"]),

                "test_ufa": int(test_iou50["unknown_false_accept_objects"]),

                "test_num_accepted": int(test_iou50["num_accepted_detections"]),

                "test_AP50": float(test_ap["AP50"]),

                "test_AP75": float(test_ap["AP75"]),

                "test_AP": float(test_ap["AP"]),

            }

        )


    summary = pd.DataFrame(summary_rows)

    selected = selected_policy_rows(summary, args.precision_tiebreak_tolerance)

    threshold_df = pd.concat(threshold_rows, ignore_index=True)


    test_rows: List[dict] = []

    for _, r in selected.iterrows():

        score_col = str(r["score_col"])

        threshold = float(r["selected_threshold"])

        scored_test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

        for iou_thr in EVAL_IOUS:

            m = evaluate_threshold(scored_test, gt_test, score_col, threshold, iou_thr)

            m.update(

                {

                    "method": f"step9c_geometry_{r['selection_policy']}",

                    "selection_policy": str(r["selection_policy"]),

                    "split": "test",

                    "alpha": float(r["alpha"]),

                    "score_col": score_col,

                    "threshold": threshold,

                    "iou_threshold": float(iou_thr),

                    "unthresholded_AP50": float(r["test_AP50"]),

                    "unthresholded_AP75": float(r["test_AP75"]),

                    "unthresholded_AP": float(r["test_AP"]),

                    "cal_balanced": float(r["cal_balanced"]),

                    "cal_precision": float(r["cal_precision"]),

                    "cal_recall": float(r["cal_recall"]),

                    "cal_AP": float(r["cal_AP"]),

                }

            )

            test_rows.append(m)


    if args.step9b_selected_config.exists():

        raw_thr = float(pd.read_csv(args.step9b_selected_config, encoding="utf-8-sig").iloc[0]["conf_thr"])

    else:

        raw_thr = float(summary[summary["alpha"].eq(0.0)].iloc[0]["selected_threshold"])

    raw_ap_test = ap_for_split(scored, gt, known_classes, "test", "raw_score")

    for iou_thr in EVAL_IOUS:

        m = evaluate_threshold(scored[scored["split"].astype(str).str.lower().eq("test")].copy(), gt_test, "raw_score", raw_thr, iou_thr)

        m.update(

            {

                "method": "step9b_raw_global_reproduced",

                "selection_policy": "raw_step9b_selected_threshold",

                "split": "test",

                "alpha": 0.0,

                "score_col": "raw_score",

                "threshold": raw_thr,

                "iou_threshold": float(iou_thr),

                "unthresholded_AP50": float(raw_ap_test["AP50"]),

                "unthresholded_AP75": float(raw_ap_test["AP75"]),

                "unthresholded_AP": float(raw_ap_test["AP"]),

                "cal_balanced": float("nan"),

                "cal_precision": float("nan"),

                "cal_recall": float("nan"),

                "cal_AP": float("nan"),

            }

        )

        test_rows.append(m)


    test_df = pd.DataFrame(test_rows)

    compact = test_df[np.isclose(pd.to_numeric(test_df["iou_threshold"], errors="coerce"), 0.50)].copy()

    ranking = compact.sort_values(

        [

            "precision_recall_unknown_balanced_score",

            "known_precision",

            "known_recall",

            "unthresholded_AP50",

        ],

        ascending=[False, False, False, False],

    ).reset_index(drop=True)


    scored.to_csv(csv_dir / "step9c_geometry_scored_candidates.csv", index=False, encoding="utf-8-sig")

    summary.to_csv(csv_dir / "step9c_alpha_sweep_summary.csv", index=False, encoding="utf-8-sig")

    threshold_df.to_csv(csv_dir / "step9c_threshold_search.csv", index=False, encoding="utf-8-sig")

    selected.to_csv(csv_dir / "step9c_selected_policies.csv", index=False, encoding="utf-8-sig")

    test_df.to_csv(csv_dir / "step9c_selected_policy_test_results.csv", index=False, encoding="utf-8-sig")

    compact.to_csv(csv_dir / "step9c_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")

    ranking.to_csv(csv_dir / "step9c_method_ranking.csv", index=False, encoding="utf-8-sig")

    raw_repro = compact[compact["method"].eq("step9b_raw_global_reproduced")].copy()

    raw_repro.to_csv(csv_dir / "step9c_raw_baseline_reproduced.csv", index=False, encoding="utf-8-sig")


    report = {

        "method": "Step9c Geometry-Aware Open-Set Reliability Calibration on LVIS-Clear-Mini-300",

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

        "thresholds": base_thresholds,

        "selection_iou": float(args.selection_iou),

        "label_iou": float(args.label_iou),

        "precision_tiebreak_tolerance": float(args.precision_tiebreak_tolerance),

        "diagnostics": diagnostics,

        "outputs": {

            "candidate_features_labeled": str(csv_dir / "step9c_candidate_features_labeled.csv"),

            "scored_candidates": str(csv_dir / "step9c_geometry_scored_candidates.csv"),

            "alpha_sweep_summary": str(csv_dir / "step9c_alpha_sweep_summary.csv"),

            "selected_policies": str(csv_dir / "step9c_selected_policies.csv"),

            "compact_paper_metrics": str(csv_dir / "step9c_compact_paper_metrics.csv"),

            "method_ranking": str(csv_dir / "step9c_method_ranking.csv"),

            "markdown_summary": str(args.output_root / "step9c_summary.md"),

        },

    }

    with open(args.output_root / "step9c_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)

    write_markdown(args.output_root / "step9c_summary.md", compact, selected, diagnostics)


    print("\n========== Step9c completed ==========")

    cols = [

        "method",

        "selection_policy",

        "alpha",

        "threshold",

        "precision_recall_unknown_balanced_score",

        "known_precision",

        "known_recall",

        "unknown_false_accept_objects",

        "background_false_accept_count",

        "unthresholded_AP50",

        "unthresholded_AP75",

        "unthresholded_AP",

    ]

    print(ranking[cols].to_string(index=False))



if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("Interrupted by user.", file=sys.stderr)

        raise

