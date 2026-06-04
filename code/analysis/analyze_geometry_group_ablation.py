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

OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "04_geometry_contribution"


CORE_CODE = BUNDLE / "code" / "core_evaluators"

COCO_CODE = BUNDLE / "code" / "coco_step12"

for p in [str(CORE_CODE), str(COCO_CODE)]:

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



RANDOM_STATE = 52

BOOTSTRAP_SEED = 2026

BOOTSTRAP_N = 500

IOU = 0.50



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



def harmonic3(p: float, r: float, u: float) -> float:

    vals = [float(p), float(r), float(u)]

    if any(v <= 0 for v in vals):

        return 0.0

    return float(3.0 / sum(1.0 / v for v in vals))



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



def build_model(numeric_cols: Iterable[str], categorical_cols: Iterable[str], random_state: int = RANDOM_STATE) -> Pipeline:

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



def load_coco() -> tuple[pd.DataFrame, pd.DataFrame, list[str], pd.DataFrame]:

    proto = BUNDLE / "data_outputs" / "coco_main" / "step12a_protocol"

    split = load_split(proto / "csv" / "step12a_scene_split.csv", ROOT, split_filter="all")

    known = load_known_classes(proto / "csv" / "step12a_class_map.csv")

    gt = load_annotations(proto / "annotations", split, known)

    scored = read_csv(BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_geometry_scored_candidates.csv")

    scored["split"] = scored["split"].astype(str).str.lower()

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    for col in [

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

        "geometry_alpha_0p800",

        "class_reliability_gamma_0p250",

    ]:

        if col in scored.columns:

            scored[col] = pd.to_numeric(scored[col], errors="coerce")

    if "raw_score" not in scored.columns and "score" in scored.columns:

        scored["raw_score"] = scored["score"]

    if "score_logit" not in scored.columns:

        s = pd.to_numeric(scored["raw_score"], errors="coerce").fillna(0.0).clip(1e-6, 1 - 1e-6)

        scored["score_logit"] = np.log(s / (1.0 - s))

    return scored, gt, known, split



def fit_ablation_scores(scored: pd.DataFrame) -> dict[str, Pipeline]:

    cal = scored[scored["split"].eq("calibration")].copy()

    y = pd.to_numeric(cal["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    weights = sample_weights(cal)

    models: dict[str, Pipeline] = {}


    score_class = build_model(["raw_score", "score_logit"], ["pred_label"])

    score_class.fit(cal, y, clf__sample_weight=weights)

    scored["ablation_score_class_logistic"] = score_class.predict_proba(scored)[:, 1].astype(float)

    models["score_class"] = score_class


    geom_cols = geometry_numeric_columns(scored)

    score_class_geometry = build_model(geom_cols, ["pred_label"])

    score_class_geometry.fit(cal, y, clf__sample_weight=weights)

    scored["ablation_score_class_geometry_logistic"] = score_class_geometry.predict_proba(scored)[:, 1].astype(float)

    models["score_class_geometry"] = score_class_geometry

    return models



def policy_specs_from_ablation() -> dict[str, dict]:

    ab = read_csv(BUNDLE / "data_outputs" / "ablation" / "coco_yoloworld_l_feature_ablation.csv")

    rows = {str(r["row"]): r for _, r in ab.iterrows()}

    def ap_fields(row) -> dict:

        return {

            "AP50": finite(row.get("AP50")),

            "AP75": finite(row.get("AP75")),

            "AP": finite(row.get("AP")),

            "AP_source": "data_outputs/ablation/coco_yoloworld_l_feature_ablation.csv",

        }

    return {

        "raw": {

            "label": "Raw score threshold",

            "score_col": "raw_score",

            "threshold": finite(rows["Raw score threshold"]["threshold"]),

            "source": "ablation_csv",

            **ap_fields(rows["Raw score threshold"]),

        },

        "score_class": {

            "label": "Score + class identity",

            "score_col": "ablation_score_class_logistic",

            "threshold": finite(rows["Score + class identity"]["threshold"]),

            "source": "ablation_csv",

            **ap_fields(rows["Score + class identity"]),

        },

        "score_class_geometry": {

            "label": "Score + class identity + geometry",

            "score_col": "ablation_score_class_geometry_logistic",

            "threshold": finite(rows["Score + class identity + geometry"]["threshold"]),

            "source": "ablation_csv",

            **ap_fields(rows["Score + class identity + geometry"]),

        },

        "full_gorc": {

            "label": "GORC selected policy",

            "score_col": "geometry_alpha_0p800",

            "threshold": finite(rows["GORC selected policy"]["threshold"]),

            "source": "step12c_selected_policy",

            **ap_fields(rows["GORC selected policy"]),

        },

    }



def selected_dets(scored: pd.DataFrame, score_col: str, threshold: float, split: str = "test") -> pd.DataFrame:

    sub = scored[scored["split"].eq(split)].copy()

    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce").fillna(-math.inf)

    dets = sub[sub[score_col] >= float(threshold)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    return dets



def unthresholded_score_dets(scored: pd.DataFrame, score_col: str, split: str = "test") -> pd.DataFrame:

    dets = scored[scored["split"].eq(split)].copy()

    dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

    return dets



def summarize_metrics(metrics: dict, error_rows: list[dict], ap: dict | None = None) -> dict:

    ap = ap or {}

    p = finite(metrics.get("known_precision"), 0.0)

    r = finite(metrics.get("known_recall"), 0.0)

    urr = finite(metrics.get("unknown_reject_rate_object_level"), 0.0)

    return {

        "B": harmonic3(p, r, urr),

        "P": p,

        "R": r,

        "URR": urr,

        "UFA_object": int(finite(metrics.get("unknown_false_accept_objects"), 0)),

        "UFA_detection": int(sum(1 for e in error_rows if e.get("error_type") == "unknown_false_accept")),

        "BGFP": int(finite(metrics.get("background_false_accept_count"), 0)),

        "accepted_detection_count": int(finite(metrics.get("num_accepted_detections"), 0)),

        "known_TP": int(finite(metrics.get("tp_known"), 0)),

        "known_FP": int(finite(metrics.get("fp_known"), 0)),

        "known_FN": int(finite(metrics.get("fn_known"), 0)),

        "known_gt": int(finite(metrics.get("num_known_gt"), 0)),

        "unknown_gt": int(finite(metrics.get("num_unknown_gt"), 0)),

        "AP50": finite(ap.get("AP50")),

        "AP75": finite(ap.get("AP75")),

        "AP": finite(ap.get("AP")),

    }



def aggregate_counts(counts: pd.DataFrame) -> dict:

    known_gt = int(counts["known_gt"].sum())

    unknown_gt = int(counts["unknown_gt"].sum())

    tp = int(counts["known_TP"].sum())

    fp = int(counts["known_FP"].sum())

    fn = int(counts["known_FN"].sum())

    ufa_obj = int(counts["UFA_object"].sum())

    ufa_det = int(counts["UFA_detection"].sum())

    bgfp = int(counts["BGFP"].sum())

    accepted = int(counts["accepted_detection_count"].sum())

    p = tp / (tp + fp) if (tp + fp) else 0.0

    r = tp / known_gt if known_gt else 0.0

    urr = 1.0 - (ufa_obj / unknown_gt) if unknown_gt else 1.0

    return {

        "B": harmonic3(p, r, urr),

        "P": p,

        "R": r,

        "URR": urr,

        "UFA_object": ufa_obj,

        "UFA_detection": ufa_det,

        "BGFP": bgfp,

        "accepted_detection_count": accepted,

        "known_TP": tp,

        "known_FP": fp,

        "known_FN": fn,

        "known_gt": known_gt,

        "unknown_gt": unknown_gt,

    }



def compute_global_metrics(scored: pd.DataFrame, gt: pd.DataFrame, known: list[str], specs: dict[str, dict]) -> pd.DataFrame:

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    global_rows = []

    for key, spec in specs.items():

        print(f"Evaluating global policy {key}", flush=True)

        dets = selected_dets(scored, spec["score_col"], spec["threshold"], "test")

        metrics, errors = evaluate_detections(dets, gt_test, iou_thr=IOU)

        ap = {"AP50": spec["AP50"], "AP75": spec["AP75"], "AP": spec["AP"]}

        row = {

            "policy_key": key,

            "policy_label": spec["label"],

            "score_col": spec["score_col"],

            "threshold": spec["threshold"],

            "AP_mode": "unthresholded_policy_score",

            "AP_source": spec["AP_source"],

            **summarize_metrics(metrics, errors, ap),

        }

        global_rows.append(row)

    return pd.DataFrame(global_rows)



def compute_policy_outputs(

    scored: pd.DataFrame,

    gt: pd.DataFrame,

    known: list[str],

    specs: dict[str, dict],

    test_image_ids: list[str],

) -> tuple[pd.DataFrame, pd.DataFrame]:

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    global_rows = []

    per_image_rows = []

    image_ids = sorted(str(x) for x in test_image_ids)

    for key, spec in specs.items():

        print(f"Evaluating policy {key}", flush=True)

        dets = selected_dets(scored, spec["score_col"], spec["threshold"], "test")

        metrics, errors = evaluate_detections(dets, gt_test, iou_thr=IOU)

        ap = {"AP50": spec["AP50"], "AP75": spec["AP75"], "AP": spec["AP"]}

        row = {

            "policy_key": key,

            "policy_label": spec["label"],

            "score_col": spec["score_col"],

            "threshold": spec["threshold"],

            "AP_mode": "unthresholded_policy_score",

            "AP_source": spec["AP_source"],

            **summarize_metrics(metrics, errors, ap),

        }

        global_rows.append(row)

        det_groups = {str(img): g.copy() for img, g in dets.groupby("image_id", sort=False)}

        gt_groups = {str(img): g.copy() for img, g in gt_test.groupby("image_id", sort=False)}

        for img in image_ids:

            m_img, e_img = evaluate_detections(det_groups.get(img, dets.iloc[0:0]), gt_groups.get(img, gt_test.iloc[0:0]), iou_thr=IOU)

            per_image_rows.append({"policy_key": key, "image_id": img, "image_universe": "split_test_all", **summarize_metrics(m_img, e_img)})

    return pd.DataFrame(global_rows), pd.DataFrame(per_image_rows)



def bootstrap_deltas(per_image: pd.DataFrame, global_metrics: pd.DataFrame, specs: dict[str, dict]) -> pd.DataFrame:

    metrics = ["B", "P", "R", "URR", "UFA_object", "UFA_detection", "BGFP"]

    point_metric_map = {r["policy_key"]: r for _, r in global_metrics.iterrows()}

    image_ids = sorted(per_image["image_id"].astype(str).unique().tolist())

    n = len(image_ids)

    per_policy = {k: per_image[per_image["policy_key"].eq(k)].set_index("image_id").loc[image_ids].reset_index() for k in specs}

    comparisons = [

        ("score_class_geometry_minus_score_class", "score_class", "score_class_geometry"),

        ("full_gorc_minus_score_class", "score_class", "full_gorc"),

        ("full_gorc_minus_score_class_geometry", "score_class_geometry", "full_gorc"),

    ]

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    rows = []

    for comparison, base, compare in comparisons:

        boot = {m: [] for m in metrics}

        for _ in range(BOOTSTRAP_N):

            idx = rng.integers(0, n, size=n)

            base_counts = aggregate_counts(per_policy[base].iloc[idx])

            compare_counts = aggregate_counts(per_policy[compare].iloc[idx])

            for m in metrics:

                boot[m].append(compare_counts[m] - base_counts[m])

        base_point = point_metric_map[base]

        compare_point = point_metric_map[compare]

        row = {

            "comparison": comparison,

            "base_policy": base,

            "compare_policy": compare,

            "base_label": specs[base]["label"],

            "compare_label": specs[compare]["label"],

            "n_bootstrap": BOOTSTRAP_N,

            "bootstrap_unit": "test image",

            "bootstrap_seed": BOOTSTRAP_SEED,

        }

        for m in metrics:

            delta = finite(compare_point[m]) - finite(base_point[m])

            vals = np.asarray(boot[m], dtype=float)

            row[f"delta_{m}"] = delta

            row[f"delta_{m}_ci025"] = float(np.quantile(vals, 0.025))

            row[f"delta_{m}_ci975"] = float(np.quantile(vals, 0.975))

        for m in ["AP", "AP50", "AP75"]:

            row[f"delta_{m}"] = finite(compare_point[m]) - finite(base_point[m])

            row[f"delta_{m}_ci025"] = float("nan")

            row[f"delta_{m}_ci975"] = float("nan")

        row["AP_bootstrap_note"] = "AP is a global rank metric; point deltas use exact AP, bootstrap CI not computed here."

        rows.append(row)

    return pd.DataFrame(rows)



def per_class_contribution(scored: pd.DataFrame, gt: pd.DataFrame, specs: dict[str, dict], known: list[str]) -> pd.DataFrame:

    rows = []

    cal_tp = scored[scored["split"].eq("calibration") & pd.to_numeric(scored["risk_label_tp"], errors="coerce").fillna(0).astype(int).eq(1)]

    test_tp = scored[scored["split"].eq("test") & pd.to_numeric(scored["risk_label_tp"], errors="coerce").fillna(0).astype(int).eq(1)]

    selected_by_policy = {}

    for key, spec in specs.items():

        dets = selected_dets(scored, spec["score_col"], spec["threshold"], "test")

        selected_by_policy[key] = dets

    for cls in known:

        row = {

            "class_name": cls,

            "num_calibration_TP": int(cal_tp["pred_label"].eq(cls).sum()),

            "num_test_TP": int(test_tp["pred_label"].eq(cls).sum()),

        }

        for key in ["raw", "score_class", "score_class_geometry", "full_gorc"]:

            dets = selected_by_policy[key]

            sub = dets[dets["pred_label"].eq(cls)].copy()

            unknown = sub[sub["risk_error_type"].astype(str).eq("unknown_false_accept")].copy()

            if "best_unknown_key" in unknown.columns:

                ufa = int(unknown["best_unknown_key"].dropna().astype(str).nunique())

            else:

                ufa = int(len(unknown))

            bg = int(sub["risk_error_type"].astype(str).eq("background_false_accept").sum())

            row[f"{key}_UFA"] = ufa

            row[f"{key}_BGFP"] = bg

        row["delta_UFA_class_to_class_geometry"] = row["score_class_geometry_UFA"] - row["score_class_UFA"]

        row["delta_BGFP_class_to_class_geometry"] = row["score_class_geometry_BGFP"] - row["score_class_BGFP"]

        row["per_class_count_note"] = "UFA/BGFP are candidate-level per-predicted-class diagnostics using cached risk labels."

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["delta_UFA_class_to_class_geometry", "delta_BGFP_class_to_class_geometry", "class_name"])



def permutation_diagnostic(scored: pd.DataFrame, gt: pd.DataFrame, known: list[str], model: Pipeline, threshold: float) -> pd.DataFrame:

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    test = scored[scored["split"].eq("test")].copy()

    geom_cols = [c for c in geometry_numeric_columns(test) if c not in {"raw_score", "score_logit"}]

    score_cols = [c for c in ["raw_score", "score_logit"] if c in test.columns]

    original_score = model.predict_proba(test)[:, 1].astype(float)

    accepted_k = int((original_score >= threshold).sum())


    def eval_variant(label: str, repeat: int, frame: pd.DataFrame) -> dict:

        scores = model.predict_proba(frame)[:, 1].astype(float)

        dets = frame.copy()

        dets["perm_score"] = scores

        dets = dets[dets["perm_score"] >= threshold].copy()

        dets["score"] = dets["perm_score"]

        metrics, errors = evaluate_detections(dets, gt_test, iou_thr=IOU)

        corr = pd.Series(original_score).corr(pd.Series(scores), method="spearman")

        top_orig = set(np.argsort(-original_score)[:accepted_k].tolist())

        top_perm = set(np.argsort(-scores)[:accepted_k].tolist())

        top_overlap = len(top_orig & top_perm) / max(accepted_k, 1)

        return {

            "permutation": label,

            "repeat": repeat,

            "threshold": threshold,

            "spearman_score_rank_vs_original": finite(corr),

            "top_accept_count_overlap_at_original_k": top_overlap,

            **summarize_metrics(metrics, errors),

        }


    rows = [eval_variant("none", 0, test)]

    rng = np.random.default_rng(BOOTSTRAP_SEED)

    groups = {

        "permute_score_features": score_cols,

        "permute_class_indicator": ["pred_label"],

        "permute_geometry_group": geom_cols,

        "permute_class_and_geometry": ["pred_label", *geom_cols],

    }

    for label, cols in groups.items():

        for repeat in range(20):

            frame = test.copy()

            for col in cols:

                frame[col] = rng.permutation(frame[col].to_numpy())

            rows.append(eval_variant(label, repeat, frame))

    return pd.DataFrame(rows)



def write_md(path: Path, title: str, df: pd.DataFrame, note: str = "") -> None:

    try:

        table = df.to_markdown(index=False)

    except Exception:

        table = df.to_csv(index=False)

    text = f"# {title}\n\n"

    if note:

        text += note.rstrip() + "\n\n"

    text += table + "\n"

    path.write_text(text, encoding="utf-8")



def main() -> None:

    ensure_dir(OUT)

    scored, gt, known, split = load_coco()

    models = fit_ablation_scores(scored)

    specs = policy_specs_from_ablation()

    test_image_ids = split[split["split"].astype(str).str.lower().eq("test")]["image_id"].astype(str).tolist()


    global_path = OUT / "geometry_policy_global_metrics_coco.csv"

    per_image_path = OUT / "geometry_policy_per_image_counts_coco.csv"

    reuse_per_image = False

    if per_image_path.exists():

        per_image = read_csv(per_image_path)

        counts = per_image.groupby("policy_key")["image_id"].nunique().to_dict() if not per_image.empty else {}

        expected = len(set(test_image_ids))

        reuse_per_image = bool(counts) and all(int(v) == expected for v in counts.values()) and per_image.get("image_universe", pd.Series(dtype=str)).astype(str).eq("split_test_all").all()

    if reuse_per_image:

        per_image = read_csv(per_image_path)

    else:

        global_metrics, per_image = compute_policy_outputs(scored, gt, known, specs, test_image_ids)

        per_image.to_csv(per_image_path, index=False, encoding="utf-8-sig")

    global_metrics = compute_global_metrics(scored, gt, known, specs)

    global_metrics.to_csv(global_path, index=False, encoding="utf-8-sig")


    effect = bootstrap_deltas(per_image, global_metrics, specs)

    effect_path = OUT / "geometry_incremental_effect_coco.csv"

    effect.to_csv(effect_path, index=False, encoding="utf-8-sig")

    write_md(

        OUT / "geometry_incremental_effect_coco.md",

        "Geometry Incremental Effect on COCO",

        effect,

        "Paired bootstrap CIs use test images as the resampling unit for count-derived metrics. AP deltas are exact unthresholded policy-score point estimates only.",

    )


    per_class = per_class_contribution(scored, gt, specs, known)

    per_class_path = OUT / "per_class_geometry_contribution_coco.csv"

    per_class.to_csv(per_class_path, index=False, encoding="utf-8-sig")

    write_md(

        OUT / "per_class_geometry_contribution_coco.md",

        "Per-Class Geometry Contribution on COCO",

        per_class,

        "Per-class UFA/BGFP are diagnostics based on cached candidate risk labels and predicted known class.",

    )


    perm_path = OUT / "group_permutation_diagnostic_coco.csv"

    if perm_path.exists():

        perm = read_csv(perm_path)

    else:

        perm = permutation_diagnostic(scored, gt, known, models["score_class_geometry"], specs["score_class_geometry"]["threshold"])

        perm.to_csv(perm_path, index=False, encoding="utf-8-sig")

    summary = (

        perm.groupby("permutation", dropna=False)

        .agg(

            repeats=("repeat", "count"),

            B_mean=("B", "mean"),

            B_std=("B", "std"),

            UFA_object_mean=("UFA_object", "mean"),

            BGFP_mean=("BGFP", "mean"),

            rank_spearman_mean=("spearman_score_rank_vs_original", "mean"),

            top_overlap_mean=("top_accept_count_overlap_at_original_k", "mean"),

        )

        .reset_index()

    )

    write_md(

        OUT / "group_permutation_diagnostic_coco.md",

        "Group Permutation Diagnostic on COCO",

        summary,

        "Rows summarize 20 held-out permutations per feature group plus the unpermuted reference row.",

    )


    manifest = {

        "global_metrics": str(global_path),

        "per_image_counts": str(per_image_path),

        "incremental_effect": str(effect_path),

        "per_class": str(per_class_path),

        "permutation": str(perm_path),

        "bootstrap_n": BOOTSTRAP_N,

        "bootstrap_seed": BOOTSTRAP_SEED,

    }

    (OUT / "geometry_contribution_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")



if __name__ == "__main__":

    main()

