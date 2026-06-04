from __future__ import annotations


import json

import math

from pathlib import Path


import numpy as np

import pandas as pd

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler



ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT / "GORC_paper_release_bundle"

OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "08_calibration_diagnostics"

FIG_OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "10_figures_for_paper"

SCORED = BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_geometry_scored_candidates.csv"


RANDOM_STATE = 52

N_BINS = 10

EPS = 1e-6


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



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def finite(value, default: float = float("nan")) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def build_model(numeric: list[str], categorical: list[str]) -> Pipeline:

    transformers = []

    if numeric:

        transformers.append(("num", StandardScaler(), numeric))

    if categorical:

        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), categorical))

    pre = ColumnTransformer(transformers=transformers, remainder="drop")

    clf = LogisticRegression(C=0.50, class_weight="balanced", max_iter=1000, solver="liblinear", random_state=RANDOM_STATE)

    return Pipeline([("pre", pre), ("clf", clf)])



def fit_scores(df: pd.DataFrame) -> pd.DataFrame:

    out = df.copy()

    train = out[out["split"].astype(str).str.lower().eq("calibration")].copy()

    y = pd.to_numeric(train["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    specs = [

        ("score_only_logistic", ["raw_score", "score_logit"], []),

        ("score_class_logistic", ["raw_score", "score_logit"], ["pred_label"]),

        ("score_class_geometry_logistic", GEOMETRY_NUMERIC, ["pred_label"]),

    ]

    for col, numeric, categorical in specs:

        model = build_model([c for c in numeric if c in out.columns], [c for c in categorical if c in out.columns])

        model.fit(train, y)

        out[col] = model.predict_proba(out)[:, 1]

    return out



def calibration_metrics(scores: np.ndarray, labels: np.ndarray, n_bins: int = N_BINS) -> dict:

    scores = np.asarray(scores, dtype=float)

    labels = np.asarray(labels, dtype=float)

    scores = np.clip(scores, 0.0, 1.0)

    bins = np.linspace(0.0, 1.0, n_bins + 1)

    ece = 0.0

    nonempty = 0

    for i in range(n_bins):

        if i == n_bins - 1:

            mask = (scores >= bins[i]) & (scores <= bins[i + 1])

        else:

            mask = (scores >= bins[i]) & (scores < bins[i + 1])

        if not mask.any():

            continue

        nonempty += 1

        ece += float(mask.mean()) * abs(float(scores[mask].mean()) - float(labels[mask].mean()))

    order = np.argsort(scores)

    chunks = np.array_split(order, n_bins)

    adaptive = 0.0

    adaptive_nonempty = 0

    for idx in chunks:

        if len(idx) == 0:

            continue

        adaptive_nonempty += 1

        adaptive += (len(idx) / len(scores)) * abs(float(scores[idx].mean()) - float(labels[idx].mean()))

    clipped = np.clip(scores, EPS, 1.0 - EPS)

    brier = float(np.mean((scores - labels) ** 2))

    nll = float(-np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped)))

    return {

        "num_candidates": int(len(scores)),

        "num_positive": int(labels.sum()),

        "empirical_tp_rate": float(labels.mean()) if len(labels) else float("nan"),

        "mean_score": float(scores.mean()) if len(scores) else float("nan"),

        "ECE_10bin": ece,

        "adaptive_ECE_10bin": adaptive,

        "Brier": brier,

        "NLL": nll,

        "nonempty_bins": int(nonempty),

        "adaptive_bins": int(adaptive_nonempty),

    }



def fixed_bins(df: pd.DataFrame, score_col: str, score_name: str) -> pd.DataFrame:

    scores = pd.to_numeric(df[score_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    labels = pd.to_numeric(df["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    err = df["risk_error_type"].astype(str)

    rows = []

    edges = np.linspace(0.0, 1.0, N_BINS + 1)

    for i in range(N_BINS):

        left = float(edges[i])

        right = float(edges[i + 1])

        if i == N_BINS - 1:

            mask = (scores >= left) & (scores <= right)

        else:

            mask = (scores >= left) & (scores < right)

        sub_scores = scores[mask]

        sub_labels = labels[mask]

        sub_err = err[mask]

        n = int(mask.sum())

        rel = float(sub_labels.mean()) if n else float("nan")

        rows.append(

            {

                "protocol": "COCO-Val-OpenSet-5K",

                "detector": "YOLO-World-l",

                "score_name": score_name,

                "bin_left": left,

                "bin_right": right,

                "num_candidates": n,

                "mean_score": float(sub_scores.mean()) if n else float("nan"),

                "empirical_reliability": rel,

                "empirical_error_rate": float(1.0 - rel) if n else float("nan"),

                "known_TP": int(sub_labels.sum()) if n else 0,

                "UFA_detection": int(sub_err.eq("unknown_false_accept").sum()) if n else 0,

                "BGFP": int(sub_err.eq("background_false_accept").sum()) if n else 0,

            }

        )

    return pd.DataFrame(rows)



def write_md(metrics: pd.DataFrame) -> None:

    cols = [

        "score_name",

        "score_col",

        "num_candidates",

        "num_positive",

        "empirical_tp_rate",

        "mean_score",

        "ECE_10bin",

        "adaptive_ECE_10bin",

        "Brier",

        "NLL",

        "interpretation",

    ]

    lines = [

        "# COCO Candidate-Level Calibration Metrics",

        "",

        "- Computed on the held-out COCO-Val-OpenSet-5K test split using candidate-level `risk_label_tp` labels.",

        "- These diagnostics evaluate whether scores behave like probabilities for known-TP correctness. They do not replace the operating-point B/UFA/BGFP metrics.",

        "- No separate D-ECE implementation was found in the release bundle; the D-ECE column is marked as not run.",

        "",

        metrics[cols].to_markdown(index=False),

    ]

    (OUT / "calibration_metrics_coco.md").write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    ensure_dir(OUT)

    ensure_dir(FIG_OUT)

    df = pd.read_csv(SCORED, encoding="utf-8-sig")

    for c in GEOMETRY_NUMERIC + ["geometry_risk_score", "geometry_alpha_0p800", "raw_score"]:

        if c in df.columns:

            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["risk_label_tp"] = pd.to_numeric(df["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    df["pred_label"] = df["pred_label"].astype(str)

    df = fit_scores(df)

    test = df[df["split"].astype(str).str.lower().eq("test")].copy()


    score_specs = [

        ("raw confidence", "raw_score", "raw detector confidence; useful for ranking but not trained as a calibrated probability"),

        ("score-only logistic", "score_only_logistic", "post-hoc score calibration on calibration split"),

        ("score + class logistic", "score_class_logistic", "post-hoc score and predicted-class calibration on calibration split"),

        ("score + class + geometry logistic", "score_class_geometry_logistic", "post-hoc score, class, and geometry calibration on calibration split"),

        ("GORC reliability score", "geometry_risk_score", "COCO GORC reliability probability trained on the calibration split"),

        ("GORC-RF selected score", "geometry_alpha_0p800", "selected reliability-first blend used for RF operating point; not a pure probability model"),

    ]


    rows = []

    bin_frames = []

    labels = test["risk_label_tp"].to_numpy(dtype=float)

    for name, col, interp in score_specs:

        m = calibration_metrics(test[col].to_numpy(dtype=float), labels)

        rows.append(

            {

                "protocol": "COCO-Val-OpenSet-5K",

                "detector": "YOLO-World-l",

                "score_name": name,

                "score_col": col,

                **m,

                "D_ECE_status": "not_run_no_existing_detection_ece_implementation_found",

                "source_candidate_file": str(SCORED),

                "interpretation": interp,

            }

        )

        bin_frames.append(fixed_bins(test, col, name))


    metrics = pd.DataFrame(rows)

    metrics.to_csv(OUT / "calibration_metrics_coco.csv", index=False, encoding="utf-8-sig")

    write_md(metrics)

    bins = pd.concat(bin_frames, ignore_index=True)

    bins.to_csv(FIG_OUT / "figure4_reliability_bins.csv", index=False, encoding="utf-8-sig")

    manifest = {

        "calibration_metrics_coco": str(OUT / "calibration_metrics_coco.csv"),

        "calibration_metrics_coco_md": str(OUT / "calibration_metrics_coco.md"),

        "figure4_reliability_bins": str(FIG_OUT / "figure4_reliability_bins.csv"),

        "num_test_candidates": int(len(test)),

        "num_scores": int(len(score_specs)),

    }

    (OUT / "calibration_diagnostics_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))



if __name__ == "__main__":

    main()

