from __future__ import annotations


import json

import math

import sys

from pathlib import Path


import pandas as pd



ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "outputs" / "gorc_major_revision_round2"

TABLE_OUT = OUT / "09_tables_for_paper"


SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:

    sys.path.insert(0, str(SCRIPTS))


from gorc_round2_strong_baselines import (

    eval_dets,

    eval_mask,

    finite,

    fit_logistic_score,

    metrics_to_row,

    prepare_scored,

    read_csv,

)



IOU = 0.75

BASELINE_CSV = OUT / "03_baselines" / "coco_strong_baselines.csv"



KEY_ROWS = [

    ("Raw global threshold", "RF global threshold"),

    ("Raw per-class threshold", "RF per-class threshold"),

    ("Score + class logistic", "RF global threshold"),

    ("Score + class + geometry logistic", "RF global threshold"),

    ("GORC-RF", "Existing selected RF"),

    ("GORC-AP-C", "Existing selected AP-C"),

]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def parse_threshold(value):

    text = str(value).strip()

    if text.startswith("{"):

        return {str(k): float(v) for k, v in json.loads(text).items()}

    return float(text)



def fit_needed_scores(scored: pd.DataFrame) -> None:

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

    if "round2_score_class_logistic" not in scored.columns:

        fit_logistic_score(scored, ["score_logit"], True, "round2_score_class_logistic")

    if "round2_score_class_geometry_logistic" not in scored.columns:

        fit_logistic_score(scored, numeric_geometry, True, "round2_score_class_geometry_logistic")



def row_for_method(baselines: pd.DataFrame, method: str, selection_mode: str) -> pd.Series:

    sub = baselines[

        baselines["method"].astype(str).eq(method)

        & baselines["selection_mode"].astype(str).eq(selection_mode)

    ].copy()

    if sub.empty:

        raise ValueError(f"Missing baseline row: {method} / {selection_mode}")

    return sub.iloc[0]



def evaluate_row(scored: pd.DataFrame, gt: pd.DataFrame, row: pd.Series) -> dict:

    score_col = str(row["score_col"])

    threshold = parse_threshold(row["threshold"])

    if isinstance(threshold, dict):

        fallback = min(threshold.values()) if threshold else 0.0

        sub = scored[scored["split"].eq("test")]

        scores = pd.to_numeric(sub[score_col], errors="coerce").fillna(-math.inf)

        thrs = sub["pred_label"].map(lambda x: threshold.get(str(x), fallback)).astype(float)

        mask = pd.Series(False, index=scored.index)

        mask.loc[sub.index] = scores.to_numpy() >= thrs.to_numpy()

        metrics = eval_mask(scored, gt, "test", score_col, mask, iou=IOU)

        threshold_display = json.dumps(threshold, ensure_ascii=False, sort_keys=True)

    else:

        metrics = eval_dets(scored, gt, "test", score_col, float(threshold), iou=IOU)

        threshold_display = threshold

    m = metrics_to_row(metrics)

    return {

        "protocol": "COCO-Val-OpenSet-5K",

        "detector": "YOLO-World-l",

        "method": str(row["method"]),

        "selection_mode": str(row["selection_mode"]),

        "score_col": score_col,

        "threshold": threshold_display,

        "selection_iou": 0.50,

        "evaluation_iou": IOU,

        "B_iou75": m["B"],

        "P_iou75": m["P"],

        "R_iou75": m["R"],

        "URR_iou75": m["URR"],

        "UFA_object_level_iou75": m["UFA"],

        "BGFP_detection_level_iou75": m["BG_FP"],

        "accepted_detection_count_iou75": m["accepted_detection_count"],

        "known_TP_count_iou75": m["known_TP_count"],

        "known_FP_count_iou75": m["known_FP_count"],

        "source_B_iou50": finite(row.get("B")),

        "source_P_iou50": finite(row.get("P")),

        "source_R_iou50": finite(row.get("R")),

        "source_URR_iou50": finite(row.get("URR")),

        "source_UFA_iou50": int(finite(row.get("UFA"), 0)),

        "source_BGFP_iou50": int(finite(row.get("BG_FP"), 0)),

        "unthresholded_AP50_reported": finite(row.get("AP50")),

        "unthresholded_AP75_reported": finite(row.get("AP75")),

        "unthresholded_AP_reported": finite(row.get("AP")),

        "source_baseline_file": str(BASELINE_CSV),

        "source_candidate_file": str(

            ROOT

            / "GORC_paper_release_bundle"

            / "data_outputs"

            / "coco_main"

            / "step12c_gorc"

            / "csv"

            / "step12c_geometry_scored_candidates.csv"

        ),

        "notes": "Thresholds and score families were selected at IoU=0.50 on calibration data; this row only re-evaluates held-out operating metrics with evaluator IoU=0.75.",

    }



def write_md(df: pd.DataFrame, path: Path) -> None:

    display_cols = [

        "method",

        "selection_mode",

        "score_col",

        "B_iou75",

        "P_iou75",

        "R_iou75",

        "URR_iou75",

        "UFA_object_level_iou75",

        "BGFP_detection_level_iou75",

        "unthresholded_AP75_reported",

        "source_B_iou50",

    ]

    lines = [

        "# Supplementary IoU0.75 Reliability Diagnostic",

        "",

        "- COCO-Val-OpenSet-5K / YOLO-World-l held-out test split.",

        "- Policies and thresholds are reused from calibration-selected IoU0.50 experiments; no IoU0.75 policy selection is performed.",

        "- AP75 is the standard unthresholded AP75 value reported by the existing AP evaluator; the reliability columns are operating-point metrics recomputed with `iou_thr=0.75`.",

        "",

        df[display_cols].to_markdown(index=False),

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    ensure_dir(TABLE_OUT)

    scored, gt, _known, _candidate_path = prepare_scored("coco")

    fit_needed_scores(scored)

    baselines = read_csv(BASELINE_CSV)

    rows = []

    for method, selection_mode in KEY_ROWS:

        rows.append(evaluate_row(scored, gt, row_for_method(baselines, method, selection_mode)))

    out = pd.DataFrame(rows)

    csv_path = TABLE_OUT / "table_s_iou075_reliability.csv"

    md_path = TABLE_OUT / "table_s_iou075_reliability.md"

    out.to_csv(csv_path, index=False, encoding="utf-8-sig")

    write_md(out, md_path)

    manifest = {

        "table_s_iou075_reliability_csv": str(csv_path),

        "table_s_iou075_reliability_md": str(md_path),

        "num_rows": int(len(out)),

        "evaluation_iou": IOU,

    }

    (TABLE_OUT / "table_s_iou075_reliability_manifest.json").write_text(

        json.dumps(manifest, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))



if __name__ == "__main__":

    main()

