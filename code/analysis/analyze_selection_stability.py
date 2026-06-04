from __future__ import annotations


import json

import math

import sys

from pathlib import Path

from typing import Iterable


import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler



ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT / "GORC_paper_release_bundle"

OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "05_selection_stability"

SPLIT_OUT = OUT / "coco_repeated_split_ids"


CORE_CODE = BUNDLE / "code" / "core_evaluators"

COCO_CODE = BUNDLE / "code" / "coco_step12"

for p in [str(CORE_CODE), str(COCO_CODE)]:

    if p not in sys.path:

        sys.path.insert(0, p)


from step8j_yoloworld_lvis_openvoc_baseline import (

    clean_columns,

    load_annotations,

    load_known_classes,

    load_split,

    safe_class_name,

)

import step12c_geometry_risk_coco_openset as step12c              



SEEDS = [101, 202, 303, 404, 505]

N_CAL = 1000

RANDOM_STATE = 52

ALPHAS = [0.05, 0.10, 0.20, 0.40, 0.60, 0.80, 1.00]

GAMMAS = [0.25, 0.50, 0.75, 1.00]



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

        "known_FN_count": int(finite(m.get("fn_known"), 0)),

        "known_gt": int(finite(m.get("num_known_gt"), 0)),

        "unknown_gt": int(finite(m.get("num_unknown_gt"), 0)),

    }



def rank_metric(m: dict) -> tuple:

    return (

        finite(m.get("precision_recall_unknown_balanced_score"), -1),

        finite(m.get("unknown_reject_rate_object_level"), -1),

        finite(m.get("known_precision"), -1),

        finite(m.get("known_recall"), -1),

        -finite(m.get("num_accepted_detections"), 1e18),

    )



def geometry_numeric_columns(scored: pd.DataFrame) -> list[str]:

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



def sample_weights(train: pd.DataFrame) -> np.ndarray:

    w = np.ones(len(train), dtype=float)

    err = train["risk_error_type"].astype(str)

    w[err.eq("unknown_false_accept").to_numpy()] = 2.0

    w[err.eq("background_false_accept").to_numpy()] = 1.25

    return w



def build_model(numeric_cols: Iterable[str], categorical_cols: Iterable[str], random_state: int) -> Pipeline:

    transformers = []

    numeric_cols = list(numeric_cols)

    categorical_cols = list(categorical_cols)

    if numeric_cols:

        transformers.append(("num", StandardScaler(), numeric_cols))

    if categorical_cols:

        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols))

    return Pipeline(

        [

            ("pre", ColumnTransformer(transformers=transformers, remainder="drop")),

            (

                "clf",

                LogisticRegression(

                    C=0.50,

                    class_weight="balanced",

                    max_iter=1000,

                    solver="liblinear",

                    random_state=int(random_state),

                ),

            ),

        ]

    )



def load_coco() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:

    proto = BUNDLE / "data_outputs" / "coco_main" / "step12a_protocol"

    split = load_split(proto / "csv" / "step12a_scene_split.csv", ROOT, split_filter="all")

    known = load_known_classes(proto / "csv" / "step12a_class_map.csv")

    gt = load_annotations(proto / "annotations", split, known)

    scored = read_csv(BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_geometry_scored_candidates.csv")

    scored["split"] = scored["split"].astype(str).str.lower()

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    for col in scored.columns:

        if col in {

            "raw_score",

            "score",

            "score_logit",

            "box_area_norm",

            "box_aspect_log",

            "box_width_norm",

            "box_height_norm",

            "center_x_norm",

            "center_y_norm",

            "edge_min_dist_norm",

            "edge_contact_count",

        } or col.startswith("geometry_alpha_") or col.startswith("class_reliability_gamma_"):

            scored[col] = pd.to_numeric(scored[col], errors="coerce")

    if "score_logit" not in scored.columns:

        s = pd.to_numeric(scored["raw_score"], errors="coerce").fillna(0.0).clip(1e-6, 1 - 1e-6)

        scored["score_logit"] = np.log(s / (1.0 - s))

    return scored, gt, split, known



def make_split_ids(split: pd.DataFrame, seed: int) -> pd.DataFrame:

    rng = np.random.default_rng(seed)

    all_ids = split["image_id"].astype(str).to_numpy()

    cal_ids = set(rng.choice(all_ids, size=N_CAL, replace=False).tolist())

    out = split[["image_id"]].copy()

    out["round2_split"] = out["image_id"].astype(str).map(lambda x: "calibration" if x in cal_ids else "test")

    out["seed"] = seed

    return out



def apply_split(scored: pd.DataFrame, gt: pd.DataFrame, ids: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:

    split_map = dict(zip(ids["image_id"].astype(str), ids["round2_split"].astype(str)))

    s = scored.copy()

    g = gt.copy()

    s["round2_split"] = s["image_id"].astype(str).map(split_map)

    g["round2_split"] = g["image_id"].astype(str).map(split_map)

    return s, g



def fast_metrics(scored_split: pd.DataFrame, gt_split: pd.DataFrame, score_col: str, threshold: float) -> dict:

    return step12c.fast_operating_metrics(scored_split, gt_split, score_col, float(threshold), 0.50)



def select_threshold(scored_cal: pd.DataFrame, gt_cal: pd.DataFrame, score_col: str) -> tuple[float, dict]:

    rows = []

    for thr in step12c.make_thresholds(scored_cal[score_col]):

        m = fast_metrics(scored_cal, gt_cal, score_col, float(thr))

        rows.append((rank_metric(m), float(thr), m))

    _, thr, m = max(rows, key=lambda x: x[0])

    return thr, m



def fit_scores(scored: pd.DataFrame, cal_mask: pd.Series, seed: int) -> pd.DataFrame:

    s = scored.copy()

    train = s[cal_mask].copy()

    y = pd.to_numeric(train["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    w = sample_weights(train)

    specs = [

        ("stability_score_only", ["raw_score", "score_logit"], []),

        ("stability_score_class", ["raw_score", "score_logit"], ["pred_label"]),

        ("stability_score_class_geometry", geometry_numeric_columns(s), ["pred_label"]),

    ]

    for col, numeric, categorical in specs:

        model = build_model([c for c in numeric if c in s.columns], [c for c in categorical if c in s.columns], seed)

        model.fit(train, y, clf__sample_weight=w)

        s[col] = model.predict_proba(s)[:, 1].astype(float)

    return s



def fit_class_scaling(scored: pd.DataFrame, cal_mask: pd.Series) -> pd.DataFrame:

    s = scored.copy()

    cal = s[cal_mask].copy()

    weights = {}

    for label, sub in cal.groupby("pred_label", sort=False):

        n = len(sub)

        tp = float(pd.to_numeric(sub["risk_label_tp"], errors="coerce").fillna(0).sum())

        weights[str(label)] = (tp + 1.0) / (n + 2.0) if n else 0.5

    vmax = max(weights.values()) if weights else 1.0

    norm = {k: max(1e-6, v / max(vmax, 1e-9)) for k, v in weights.items()}

    for gamma in GAMMAS:

        col = f"stability_class_reliability_gamma_{str(gamma).replace('.', 'p')}"

        s[col] = pd.to_numeric(s["raw_score"], errors="coerce").fillna(0.0) * s["pred_label"].map(lambda x: norm.get(str(x), 0.5) ** gamma)

    return s



def add_alpha_blends(scored: pd.DataFrame) -> pd.DataFrame:

    s = scored.copy()

    for alpha in ALPHAS:

        col = f"stability_full_gorc_alpha_{str(alpha).replace('.', 'p')}"

        s[col] = (1.0 - alpha) * pd.to_numeric(s["raw_score"], errors="coerce").fillna(0.0) + alpha * pd.to_numeric(s["stability_score_class_geometry"], errors="coerce").fillna(0.0)

    return s



def select_best_score_family(scored_cal: pd.DataFrame, gt_cal: pd.DataFrame, cols: list[str]) -> tuple[str, float, dict]:

    best = None

    for col in cols:

        thr, m = select_threshold(scored_cal, gt_cal, col)

        item = (rank_metric(m), col, thr, m)

        if best is None or item[0] > best[0]:

            best = item

    assert best is not None

    _, col, thr, m = best

    return col, thr, m



def eval_selected(scored_test: pd.DataFrame, gt_test: pd.DataFrame, score_col: str, threshold: float) -> dict:

    return fast_metrics(scored_test, gt_test, score_col, threshold)



def select_per_class_thresholds(scored_cal: pd.DataFrame, gt_cal: pd.DataFrame, raw_thr: float) -> tuple[dict[str, float], dict]:

    labels = sorted(scored_cal["pred_label"].dropna().astype(str).unique().tolist())

    thresholds = {label: raw_thr for label in labels}

    choices = {}

    for label in labels:

        vals = {raw_thr, 0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 0.90}

        scores = pd.to_numeric(scored_cal.loc[scored_cal["pred_label"].eq(label), "raw_score"], errors="coerce").dropna()

        for q in [0.25, 0.50, 0.75, 0.90]:

            if len(scores):

                vals.add(float(scores.quantile(q)))

        choices[label] = sorted(v for v in vals if math.isfinite(v))


    def set_tmp_score(frame: pd.DataFrame) -> pd.DataFrame:

        tmp = frame.copy()

        thrs = tmp["pred_label"].map(lambda x: thresholds.get(str(x), raw_thr)).astype(float)

        tmp["stability_raw_per_class_margin"] = pd.to_numeric(tmp["raw_score"], errors="coerce").fillna(-math.inf) - thrs

        return tmp


    best_m = fast_metrics(set_tmp_score(scored_cal), gt_cal, "stability_raw_per_class_margin", 0.0)

    for label in labels:

        best_local = (rank_metric(best_m), thresholds[label], best_m)

        old = thresholds[label]

        for thr in choices[label]:

            thresholds[label] = float(thr)

            m = fast_metrics(set_tmp_score(scored_cal), gt_cal, "stability_raw_per_class_margin", 0.0)

            key = rank_metric(m)

            if key > best_local[0]:

                best_local = (key, float(thr), m)

        thresholds[label] = best_local[1]

        best_m = best_local[2]

        if thresholds[label] != old:

            pass

    return thresholds, best_m



def apply_per_class_margin(scored: pd.DataFrame, thresholds: dict[str, float], fallback: float) -> pd.DataFrame:

    out = scored.copy()

    thrs = out["pred_label"].map(lambda x: thresholds.get(str(x), fallback)).astype(float)

    out["stability_raw_per_class_margin"] = pd.to_numeric(out["raw_score"], errors="coerce").fillna(-math.inf) - thrs

    return out



def run_seed(scored: pd.DataFrame, gt: pd.DataFrame, split: pd.DataFrame, seed: int) -> list[dict]:

    ids = make_split_ids(split, seed)

    ids.to_csv(SPLIT_OUT / f"seed_{seed}_split_ids.csv", index=False, encoding="utf-8-sig")

    s, g = apply_split(scored, gt, ids)

    cal_mask = s["round2_split"].eq("calibration")

    test_mask = s["round2_split"].eq("test")

    gt_cal = g[g["round2_split"].eq("calibration")].copy()

    gt_test = g[g["round2_split"].eq("test")].copy()

    s = fit_scores(s, cal_mask, seed)

    s = fit_class_scaling(s, cal_mask)

    s = add_alpha_blends(s)

    scored_cal = s[cal_mask].copy()

    scored_test = s[test_mask].copy()

    rows = []


    def emit(method: str, selection_mode: str, score_col: str, threshold: float, cal_m: dict, notes: str = "") -> None:

        test_m = eval_selected(scored_test, gt_test, score_col, threshold)

        rows.append(

            {

                "seed": seed,

                "method": method,

                "selection_mode": selection_mode,

                "score_col": score_col,

                "threshold": threshold,

                **{f"cal_{k}": v for k, v in metrics_to_row(cal_m).items()},

                **metrics_to_row(test_m),

                "AP50": float("nan"),

                "AP75": float("nan"),

                "AP": float("nan"),

                "AP_note": "Exact AP was not recomputed for repeated splits; stability audit uses cached fast operating metrics.",

                "notes": notes,

            }

        )


    raw_thr, raw_cal = select_threshold(scored_cal, gt_cal, "raw_score")

    emit("Raw global threshold", "RF global threshold", "raw_score", raw_thr, raw_cal)


    per_class_thresholds, per_class_cal = select_per_class_thresholds(scored_cal, gt_cal, raw_thr)

    test_pc = apply_per_class_margin(scored_test, per_class_thresholds, raw_thr)

    cal_pc = apply_per_class_margin(scored_cal, per_class_thresholds, raw_thr)

    pc_test_m = fast_metrics(test_pc, gt_test, "stability_raw_per_class_margin", 0.0)

    rows.append(

        {

            "seed": seed,

            "method": "Raw per-class threshold",

            "selection_mode": "RF per-class threshold",

            "score_col": "stability_raw_per_class_margin",

            "threshold": json.dumps(per_class_thresholds, sort_keys=True),

            **{f"cal_{k}": v for k, v in metrics_to_row(per_class_cal).items()},

            **metrics_to_row(pc_test_m),

            "AP50": float("nan"),

            "AP75": float("nan"),

            "AP": float("nan"),

            "AP_note": "Exact AP was not recomputed for repeated splits; raw-score ranking is AP-constrained but per-class thresholds affect acceptance only.",

            "notes": "Coordinate-ascent per-class threshold baseline; one pass over classes.",

        }

    )


    for method, col in [

        ("Score-only logistic", "stability_score_only"),

        ("Score + class logistic", "stability_score_class"),

        ("Score + class + geometry logistic", "stability_score_class_geometry"),

    ]:

        thr, cal_m = select_threshold(scored_cal, gt_cal, col)

        emit(method, "RF global threshold", col, thr, cal_m)


    class_cols = [f"stability_class_reliability_gamma_{str(g).replace('.', 'p')}" for g in GAMMAS]

    col, thr, cal_m = select_best_score_family(scored_cal, gt_cal, class_cols)

    emit("Class-level scaling", "RF gamma/threshold", col, thr, cal_m)


    alpha_cols = [f"stability_full_gorc_alpha_{str(a).replace('.', 'p')}" for a in ALPHAS]

    col, thr, cal_m = select_best_score_family(scored_cal, gt_cal, alpha_cols)

    emit("GORC-RF", "RF alpha/threshold", col, thr, cal_m, "GORC-RF refit on repeated calibration split with score+class+geometry risk model.")



    emit("GORC-AP-C", "AP-C proxy class-scaling", col if False else "stability_class_reliability_gamma_0p25", 0.30, fast_metrics(scored_cal, gt_cal, "stability_class_reliability_gamma_0p25", 0.30), "AP-C proxy: fixed gamma=0.25, threshold=0.30; exact split-specific AP constraint not recomputed.")


    return rows



def write_summary(df: pd.DataFrame, path: Path) -> None:

    summary = (

        df.groupby(["method", "selection_mode"], dropna=False)

        .agg(

            n=("seed", "nunique"),

            B_mean=("B", "mean"),

            B_std=("B", "std"),

            B_min=("B", "min"),

            B_max=("B", "max"),

            P_mean=("P", "mean"),

            R_mean=("R", "mean"),

            URR_mean=("URR", "mean"),

            UFA_mean=("UFA", "mean"),

            UFA_std=("UFA", "std"),

            BG_FP_mean=("BG_FP", "mean"),

            BG_FP_std=("BG_FP", "std"),

        )

        .reset_index()

        .sort_values("B_mean", ascending=False)

    )

    try:

        table = summary.to_markdown(index=False)

    except Exception:

        table = summary.to_csv(index=False)

    text = (

        "# COCO Repeated Split Stability Summary\n\n"

        "Five random image-level splits were evaluated with 1000 calibration images and 4000 test images per split. "

        "All threshold/model selection used the split-specific calibration images. Metrics are cached fast operating metrics from precomputed candidate/GT overlap keys; exact AP was not recomputed for repeated splits.\n\n"

        + table

        + "\n"

    )

    path.write_text(text, encoding="utf-8")



def run_nested(scored: pd.DataFrame, gt: pd.DataFrame, split: pd.DataFrame) -> pd.DataFrame:

    original = split[["image_id", "split"]].copy()

    cal_ids = original[original["split"].astype(str).str.lower().eq("calibration")]["image_id"].astype(str).tolist()

    rng = np.random.default_rng(777)

    train_ids = set(rng.choice(cal_ids, size=min(500, len(cal_ids)), replace=False).tolist())

    nested = original.copy()

    nested["round2_split"] = nested.apply(

        lambda r: "calibration_train" if str(r["image_id"]) in train_ids else ("policy_validation" if str(r["split"]).lower() == "calibration" else "test"),

        axis=1,

    )

    nested[["image_id", "round2_split"]].to_csv(OUT / "coco_nested_calibration_split_ids.csv", index=False, encoding="utf-8-sig")

    split_map = dict(zip(nested["image_id"].astype(str), nested["round2_split"].astype(str)))

    s = scored.copy()

    g = gt.copy()

    s["nested_split"] = s["image_id"].astype(str).map(split_map)

    g["nested_split"] = g["image_id"].astype(str).map(split_map)

    train_mask = s["nested_split"].eq("calibration_train")

    val_mask = s["nested_split"].eq("policy_validation")

    test_mask = s["nested_split"].eq("test")

    s = fit_scores(s, train_mask, 777)

    s = fit_class_scaling(s, train_mask)

    s = add_alpha_blends(s)

    val = s[val_mask].copy()

    test = s[test_mask].copy()

    gt_val = g[g["nested_split"].eq("policy_validation")].copy()

    gt_test = g[g["nested_split"].eq("test")].copy()

    rows = []

    for method, col in [

        ("Raw global threshold", "raw_score"),

        ("Score-only logistic", "stability_score_only"),

        ("Score + class logistic", "stability_score_class"),

        ("Score + class + geometry logistic", "stability_score_class_geometry"),

    ]:

        thr, val_m = select_threshold(val, gt_val, col)

        test_m = fast_metrics(test, gt_test, col, thr)

        rows.append({"method": method, "selection_mode": "nested train500/select500", "score_col": col, "threshold": thr, **{f"validation_{k}": v for k, v in metrics_to_row(val_m).items()}, **metrics_to_row(test_m)})

    alpha_cols = [f"stability_full_gorc_alpha_{str(a).replace('.', 'p')}" for a in ALPHAS]

    col, thr, val_m = select_best_score_family(val, gt_val, alpha_cols)

    test_m = fast_metrics(test, gt_test, col, thr)

    rows.append({"method": "GORC-RF", "selection_mode": "nested train500/select500", "score_col": col, "threshold": thr, **{f"validation_{k}": v for k, v in metrics_to_row(val_m).items()}, **metrics_to_row(test_m)})

    return pd.DataFrame(rows)



def write_nested_md(df: pd.DataFrame, path: Path) -> None:

    try:

        table = df.to_markdown(index=False)

    except Exception:

        table = df.to_csv(index=False)

    text = "# COCO Nested Calibration Audit\n\nCalibration split was divided into 500 model-training images and 500 policy-validation images; original 4000 test images remained held out. Metrics are cached fast operating metrics; exact AP was not recomputed.\n\n" + table + "\n"

    path.write_text(text, encoding="utf-8")



def write_lvis_limitation() -> None:

    csv_path = OUT / "lvis_repeated_split_results.csv"

    df = pd.DataFrame(

        [

            {

                "status": "not_run",

                "reason": "LVIS-Clear-Mini-300 is small; repeated random splits risk too few unknown/known support cases per split. COCO repeated split and nested audits were prioritized.",

                "recommendation": "Use LVIS strong-baseline table as supplementary diagnostic; report LVIS repeated split as not feasible without a larger LVIS validation subset.",

            }

        ]

    )

    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    (OUT / "lvis_repeated_split_summary.md").write_text(

        "# LVIS Repeated Split Feasibility\n\n"

        "Not run. LVIS-Clear-Mini-300 has only 300 images and a small unknown-object pool; repeated random splits would be highly unstable and less informative than COCO-Val-OpenSet-5K. This is documented as a limitation rather than hidden.\n",

        encoding="utf-8",

    )



def main() -> None:

    ensure_dir(OUT)

    ensure_dir(SPLIT_OUT)

    scored, gt, split, _ = load_coco()

    result_path = OUT / "coco_repeated_split_results.csv"

    rows = []

    done = set()

    if result_path.exists():

        existing = read_csv(result_path)

        rows.extend(existing.to_dict("records"))

        done = set(int(x) for x in existing["seed"].dropna().unique().tolist()) if "seed" in existing.columns else set()

    for seed in SEEDS:

        if seed in done:

            print(f"Skipping completed seed {seed}", flush=True)

            continue

        print(f"Running repeated split seed {seed}", flush=True)

        rows.extend(run_seed(scored, gt, split, seed))

        pd.DataFrame(rows).to_csv(result_path, index=False, encoding="utf-8-sig")

    df = read_csv(result_path)

    write_summary(df, OUT / "coco_repeated_split_summary.md")

    nested = run_nested(scored, gt, split)

    nested.to_csv(OUT / "coco_nested_calibration_audit.csv", index=False, encoding="utf-8-sig")

    write_nested_md(nested, OUT / "coco_nested_calibration_audit.md")

    write_lvis_limitation()

    manifest = {

        "coco_repeated_split_results": str(result_path),

        "coco_repeated_split_summary": str(OUT / "coco_repeated_split_summary.md"),

        "coco_repeated_split_ids": str(SPLIT_OUT),

        "coco_nested_calibration_audit": str(OUT / "coco_nested_calibration_audit.csv"),

        "lvis_repeated_split_results": str(OUT / "lvis_repeated_split_results.csv"),

        "seeds": SEEDS,

        "metric_mode": "cached_fast_operating_metrics_no_exact_AP",

    }

    (OUT / "selection_stability_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")



if __name__ == "__main__":

    main()

