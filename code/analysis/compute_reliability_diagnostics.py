


"""
Step11D: Reliability calibration diagnostics.

Outputs reliability diagrams, ECE summaries, and risk-coverage curves for
YOLO-World-l and Grounding DINO under the LVIS-Clear-Mini-300 protocol.
"""


from __future__ import annotations


import argparse

import json

import math

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Sequence


import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import evaluate_detections, load_annotations, load_known_classes, load_split



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step11d_reliability_diagnostics"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    return pd.read_csv(path, encoding="utf-8-sig")



def finite_float(value, default: float = 0.0) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def ece_bins(df: pd.DataFrame, score_col: str, label_col: str, detector: str, score_name: str, split: str, n_bins: int) -> tuple[pd.DataFrame, dict]:

    sub = df[df["split"].astype(str).str.lower().eq(split)].copy()

    sub[score_col] = pd.to_numeric(sub[score_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    sub[label_col] = pd.to_numeric(sub[label_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)

    rows: List[dict] = []

    total = max(1, int(len(sub)))

    ece = 0.0

    for i in range(n_bins):

        lo, hi = float(edges[i]), float(edges[i + 1])

        if i == n_bins - 1:

            b = sub[(sub[score_col] >= lo) & (sub[score_col] <= hi)].copy()

        else:

            b = sub[(sub[score_col] >= lo) & (sub[score_col] < hi)].copy()

        count = int(len(b))

        mean_conf = float(b[score_col].mean()) if count else float((lo + hi) * 0.5)

        empirical_tp = float(b[label_col].mean()) if count else float("nan")

        gap = abs(mean_conf - empirical_tp) if count else 0.0

        ece += (count / total) * gap

        rows.append(

            {

                "detector": detector,

                "score_name": score_name,

                "score_col": score_col,

                "split": split,

                "bin_index": i,

                "bin_low": lo,

                "bin_high": hi,

                "count": count,

                "mean_confidence": mean_conf,

                "empirical_tp_rate": empirical_tp,

                "abs_gap": gap,

            }

        )

    summary = {

        "detector": detector,

        "score_name": score_name,

        "score_col": score_col,

        "split": split,

        "num_candidates": int(len(sub)),

        "n_bins": int(n_bins),

        "ECE": float(ece),

        "mean_score": float(sub[score_col].mean()) if len(sub) else float("nan"),

        "empirical_tp_rate": float(sub[label_col].mean()) if len(sub) else float("nan"),

    }

    return pd.DataFrame(rows), summary



def make_thresholds(scores: pd.Series) -> List[float]:

    s = pd.to_numeric(scores, errors="coerce").dropna()

    vals = {0.0}

    if len(s):

        for q in np.linspace(0.0, 0.98, 26):

            vals.add(float(s.quantile(q)))

        vals.add(float(s.max()))

    return sorted(v for v in vals if math.isfinite(v))



def risk_coverage_rows(

    df: pd.DataFrame,

    gt_test: pd.DataFrame,

    known_classes: Sequence[str],

    detector: str,

    score_name: str,

    score_col: str,

    split: str,

) -> pd.DataFrame:

    sub = df[df["split"].astype(str).str.lower().eq(split)].copy()

    rows: List[dict] = []

    for thr in tqdm(make_thresholds(sub[score_col]), desc=f"Risk coverage {detector} {score_name}"):

        dets = sub[pd.to_numeric(sub[score_col], errors="coerce") >= float(thr)].copy()

        dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

        op, _ = evaluate_detections(dets, gt_test, iou_thr=0.50)

        accepted = int(op["num_accepted_detections"])

        fp = int(op["fp_known"])

        rows.append(

            {

                "detector": detector,

                "score_name": score_name,

                "score_col": score_col,

                "split": split,

                "threshold": float(thr),

                "accepted_detections": accepted,

                "coverage_fraction": float(accepted / max(1, len(sub))),

                "tp_known": int(op["tp_known"]),

                "fp_known": fp,

                "error_rate_among_accepted": float(fp / accepted) if accepted else 0.0,

                "known_precision": float(op["known_precision"]),

                "known_recall": float(op["known_recall"]),

                "balanced": float(op["precision_recall_unknown_balanced_score"]),

                "unknown_false_accepts": int(op["unknown_false_accept_objects"]),

                "background_false_accepts": int(op["background_false_accept_count"]),

            }

        )

    return pd.DataFrame(rows)



def plot_reliability(bin_df: pd.DataFrame, detector: str, out_path: Path) -> None:

    sub = bin_df[bin_df["detector"].eq(detector)].copy()

    plt.figure(figsize=(7.5, 6), dpi=150)

    plt.plot([0, 1], [0, 1], "--", color="black", linewidth=1, label="Ideal")

    for score_name, g in sub.groupby("score_name", sort=False):

        g = g[g["count"] > 0].copy()

        plt.plot(g["mean_confidence"], g["empirical_tp_rate"], marker="o", linewidth=1.5, label=score_name)

    plt.xlabel("Mean score")

    plt.ylabel("Empirical TP rate")

    plt.title(f"Reliability diagram: {detector}")

    plt.xlim(0, 1)

    plt.ylim(0, 1)

    plt.grid(alpha=0.25)

    plt.legend(fontsize=8)

    plt.tight_layout()

    plt.savefig(out_path)

    plt.close()



def plot_risk_coverage(curve_df: pd.DataFrame, detector: str, out_path: Path) -> None:

    sub = curve_df[curve_df["detector"].eq(detector)].copy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), dpi=150)

    for score_name, g in sub.groupby("score_name", sort=False):

        g = g.sort_values("accepted_detections")

        axes[0].plot(g["accepted_detections"], g["error_rate_among_accepted"], linewidth=1.5, label=score_name)

        axes[1].plot(g["accepted_detections"], g["unknown_false_accepts"], linewidth=1.5, label=score_name)

        axes[2].plot(g["accepted_detections"], g["background_false_accepts"], linewidth=1.5, label=score_name)

    axes[0].set_ylabel("Error rate")

    axes[1].set_ylabel("UFA")

    axes[2].set_ylabel("Background FP")

    for ax in axes:

        ax.set_xlabel("Accepted detections")

        ax.grid(alpha=0.25)

    axes[0].legend(fontsize=7)

    fig.suptitle(f"Risk-coverage diagnostics: {detector}")

    fig.tight_layout()

    fig.savefig(out_path)

    plt.close(fig)



def prepare_yoloworld(project_root: Path) -> tuple[pd.DataFrame, List[dict]]:

    path = project_root / "outputs" / "step9c_geometry_risk_lvis300" / "csv" / "step9c_geometry_scored_candidates.csv"

    df = read_csv(path)

    if df.empty:

        return df, []

    df["yolo_geometry_reliability"] = pd.to_numeric(df["geometry_risk_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)

    df["yolo_ap_constrained_blend"] = (

        0.8 * pd.to_numeric(df["raw_score"], errors="coerce").fillna(0.0)

        + 0.2 * pd.to_numeric(df["geometry_risk_score"], errors="coerce").fillna(0.0)

    ).clip(0.0, 1.0)

    scores = [

        {"score_col": "raw_score", "score_name": "raw confidence"},

        {"score_col": "yolo_geometry_reliability", "score_name": "geometry reliability"},

        {"score_col": "yolo_ap_constrained_blend", "score_name": "AP-constrained blend"},

    ]

    return df, scores



def prepare_groundingdino(project_root: Path) -> tuple[pd.DataFrame, List[dict]]:

    path = project_root / "outputs" / "step11b_groundingdino_geometry_risk" / "csv" / "step11b_geometry_scored_candidates.csv"

    df = read_csv(path)

    if df.empty:

        return df, []

    if "geometry_alpha_0p500" not in df.columns:

        df["geometry_alpha_0p500"] = (

            0.5 * pd.to_numeric(df["raw_score"], errors="coerce").fillna(0.0)

            + 0.5 * pd.to_numeric(df["geometry_risk_score"], errors="coerce").fillna(0.0)

        ).clip(0.0, 1.0)

    scores = [

        {"score_col": "raw_score", "score_name": "raw confidence"},

        {"score_col": "geometry_risk_score", "score_name": "geometry reliability"},

        {"score_col": "geometry_alpha_0p500", "score_name": "AP-constrained blend"},

    ]

    return df, scores



def write_summary(path: Path, ece_summary: pd.DataFrame, curve_summary: pd.DataFrame, figures: Dict[str, str]) -> None:

    lines = [

        "# Step11D Reliability Diagnostics Summary",

        "",

        "## ECE Summary",

        "",

        ece_summary.to_markdown(index=False),

        "",

        "## Risk-Coverage Key Points",

        "",

        curve_summary.to_markdown(index=False),

        "",

        "## Figures",

        "",

    ]

    for name, fp in figures.items():

        lines.append(f"- {name}: `{fp}`")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv")

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_PROTOCOL_ROOT / "annotations")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv")

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--split", type=str, default="test")

    parser.add_argument("--n_bins", type=int, default=10)

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    fig_dir = args.output_root / "figures"

    ensure_dir(csv_dir)

    ensure_dir(fig_dir)


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root)

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    gt_test = gt[gt["split"].astype(str).str.lower().eq(args.split.lower())].copy()


    detectors = []

    yolo_df, yolo_scores = prepare_yoloworld(args.project_root)

    if len(yolo_df):

        detectors.append(("YOLO-World-l", yolo_df, yolo_scores))

    gd_df, gd_scores = prepare_groundingdino(args.project_root)

    if len(gd_df):

        detectors.append(("Grounding DINO tiny", gd_df, gd_scores))


    bin_rows: List[pd.DataFrame] = []

    ece_rows: List[dict] = []

    curve_rows: List[pd.DataFrame] = []

    for detector, df, score_specs in detectors:

        for spec in score_specs:

            bins, summary = ece_bins(df, spec["score_col"], "risk_label_tp", detector, spec["score_name"], args.split.lower(), args.n_bins)

            bin_rows.append(bins)

            ece_rows.append(summary)

            curve_rows.append(risk_coverage_rows(df, gt_test, known_classes, detector, spec["score_name"], spec["score_col"], args.split.lower()))


    bin_df = pd.concat(bin_rows, ignore_index=True) if bin_rows else pd.DataFrame()

    ece_summary = pd.DataFrame(ece_rows)

    curve_df = pd.concat(curve_rows, ignore_index=True) if curve_rows else pd.DataFrame()

    key_curve = (

        curve_df.sort_values(["detector", "score_name", "balanced"], ascending=[True, True, False])

        .groupby(["detector", "score_name"], as_index=False)

        .head(1)

        .reset_index(drop=True)

        if len(curve_df)

        else pd.DataFrame()

    )


    bin_df.to_csv(csv_dir / "step11d_reliability_bins.csv", index=False, encoding="utf-8-sig")

    ece_summary.to_csv(csv_dir / "step11d_ece_summary.csv", index=False, encoding="utf-8-sig")

    curve_df.to_csv(csv_dir / "step11d_risk_coverage_summary.csv", index=False, encoding="utf-8-sig")

    key_curve.to_csv(csv_dir / "step11d_risk_coverage_key_points.csv", index=False, encoding="utf-8-sig")


    figures: Dict[str, str] = {}

    if len(bin_df[bin_df["detector"].eq("YOLO-World-l")]):

        fp = fig_dir / "reliability_yoloworld.png"

        plot_reliability(bin_df, "YOLO-World-l", fp)

        figures["reliability_yoloworld"] = str(fp)

    if len(curve_df[curve_df["detector"].eq("YOLO-World-l")]):

        fp = fig_dir / "risk_coverage_yoloworld.png"

        plot_risk_coverage(curve_df, "YOLO-World-l", fp)

        figures["risk_coverage_yoloworld"] = str(fp)

    if len(bin_df[bin_df["detector"].eq("Grounding DINO tiny")]):

        fp = fig_dir / "reliability_groundingdino.png"

        plot_reliability(bin_df, "Grounding DINO tiny", fp)

        figures["reliability_groundingdino"] = str(fp)

    if len(curve_df[curve_df["detector"].eq("Grounding DINO tiny")]):

        fp = fig_dir / "risk_coverage_groundingdino.png"

        plot_risk_coverage(curve_df, "Grounding DINO tiny", fp)

        figures["risk_coverage_groundingdino"] = str(fp)


    write_summary(args.output_root / "step11d_summary.md", ece_summary, key_curve, figures)

    report = {

        "method": "Step11D reliability diagnostics",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "split": args.split.lower(),

        "n_bins": int(args.n_bins),

        "detectors": [d[0] for d in detectors],

        "row_counts": {

            "ece_summary": int(len(ece_summary)),

            "reliability_bins": int(len(bin_df)),

            "risk_coverage_summary": int(len(curve_df)),

        },

        "outputs": {

            "ece_summary": str(csv_dir / "step11d_ece_summary.csv"),

            "risk_coverage_summary": str(csv_dir / "step11d_risk_coverage_summary.csv"),

            "summary": str(args.output_root / "step11d_summary.md"),

            "integrity_report": str(args.output_root / "step11d_integrity_report.json"),

            "figures": figures,

        },

    }

    with open(args.output_root / "step11d_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)


    print("\n========== Step11D completed ==========")

    print(ece_summary.to_string(index=False))

    print("Figures:")

    for name, fp in figures.items():

        print(f"- {name}: {fp}")



if __name__ == "__main__":

    main()

