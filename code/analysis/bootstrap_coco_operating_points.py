


"""
Step12D: Paired image bootstrap for COCO-Val-OpenSet-5K.

The bootstrap compares raw YOLO-World-l against selected Step12C geometry
policies. Operating metrics are aggregated from cached per-image counts. AP
metrics use a cached per-image AP approximation, because exact COCO-style AP
is not additively decomposable by image and would be too slow for repeated
bootstrap resampling at this scale.
"""


from __future__ import annotations


import argparse

import json

import math

import sys

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Sequence


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

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step12a_coco_val_openset_protocol"

DEFAULT_STEP12C_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step12c_geometry_risk_coco_openset"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step12d_coco_openset_bootstrap"

AP_COLS = ["AP50", "AP75", "AP"]

COUNT_COLS = [

    "num_known_gt",

    "num_unknown_gt",

    "num_accepted_detections",

    "tp_known",

    "fp_known",

    "fn_known",

    "unknown_false_accept_objects",

    "background_false_accept_count",

]

METRIC_COLS = [

    "precision_recall_unknown_balanced_score",

    "known_precision",

    "known_recall",

    "unknown_false_accept_objects",

    "background_false_accept_count",

    "AP50",

    "AP75",

    "AP",

]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def harmonic3(a: float, b: float, c: float) -> float:

    vals = [float(a), float(b), float(c)]

    if any(v <= 0 for v in vals):

        return 0.0

    return float(3.0 / sum(1.0 / v for v in vals))



def choose_methods(compact: pd.DataFrame) -> List[dict]:

    rows: List[dict] = []

    raw = compact[compact["method"].astype(str).eq("step12b_yoloworld_l_raw_reproduced")].copy()

    if raw.empty:

        raise ValueError("Step12C compact metrics missing raw baseline row.")

    r = raw.iloc[0]

    rows.append(

        {

            "method": str(r["method"]),

            "selection_policy": str(r.get("selection_policy", "step12b_selected_threshold")),

            "score_col": str(r["score_col"]),

            "threshold": float(r["threshold"]),

            "bootstrap_role": "baseline",

        }

    )


    rel = compact[compact["selection_policy"].astype(str).eq("max_cal_balanced")].copy()

    if not rel.empty:

        r = rel.iloc[0]

        rows.append(

            {

                "method": str(r["method"]),

                "selection_policy": str(r["selection_policy"]),

                "score_col": str(r["score_col"]),

                "threshold": float(r["threshold"]),

                "bootstrap_role": "reliability_first",

            }

        )


    strict = compact[

        compact["selection_policy"].astype(str).eq("ap_constrained_balanced_AP50tol0_APtol0")

    ].copy()

    if strict.empty:

        strict = compact[

            compact["mode"].astype(str).eq("ap_constrained")

            & compact["meets_success_criterion"].astype(str).str.lower().eq("true")

        ].copy()

    if not strict.empty:

        strict = strict.sort_values(

            ["precision_recall_unknown_balanced_score", "known_precision", "unknown_false_accept_objects", "background_false_accept_count"],

            ascending=[False, False, True, True],

        ).iloc[0]

        rows.append(

            {

                "method": str(strict["method"]),

                "selection_policy": str(strict["selection_policy"]),

                "score_col": str(strict["score_col"]),

                "threshold": float(strict["threshold"]),

                "bootstrap_role": "ap_constrained",

            }

        )


    unique: List[dict] = []

    seen = set()

    for row in rows:

        key = (row["method"], row["score_col"], row["threshold"])

        if key in seen:

            continue

        seen.add(key)

        unique.append(row)

    if len(unique) < 2:

        raise ValueError("Need at least raw and one geometry policy for bootstrap.")

    return unique



def image_ap_approx(dets: pd.DataFrame, gt_img: pd.DataFrame, known_classes: Sequence[str]) -> dict:

    if gt_img[gt_img["gt_is_known"]].empty:

        return {"AP50": float("nan"), "AP75": float("nan"), "AP": float("nan")}

    if dets.empty:

        return {"AP50": 0.0, "AP75": 0.0, "AP": 0.0}

    ap, _ = compute_ap_summary(dets, gt_img, known_classes)

    row = ap.iloc[0].to_dict()

    return {

        "AP50": float(row.get("AP50", float("nan"))),

        "AP75": float(row.get("AP75", float("nan"))),

        "AP": float(row.get("AP", float("nan"))),

    }



def precompute_per_image(

    scored: pd.DataFrame,

    gt_test: pd.DataFrame,

    image_ids: Sequence[str],

    known_classes: Sequence[str],

    methods: List[dict],

    iou_thr: float,

) -> pd.DataFrame:

    rows: List[dict] = []

    gt_by_image = {img: g.copy() for img, g in gt_test.groupby("image_id", sort=False)}

    det_by_image = {img: g.copy() for img, g in scored.groupby("image_id", sort=False)}

    for spec in tqdm(methods, desc="Step12D precompute per-image metrics"):

        score_col = str(spec["score_col"])

        threshold = float(spec["threshold"])

        for image_id in image_ids:

            gt_img = gt_by_image.get(image_id, gt_test.iloc[0:0]).copy()

            dets = det_by_image.get(image_id, scored.iloc[0:0]).copy()

            dets = dets[pd.to_numeric(dets[score_col], errors="coerce") >= threshold].copy()

            dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

            metrics, _ = evaluate_detections(dets, gt_img, iou_thr=float(iou_thr))

            ap = image_ap_approx(dets, gt_img, known_classes)

            rows.append({"method": spec["method"], "image_id": image_id, **metrics, **ap})

    return pd.DataFrame(rows)



def aggregate_weighted(per_image: pd.DataFrame, method: str, weights: np.ndarray, image_ids: Sequence[str]) -> dict:

    g = per_image[per_image["method"].eq(method)].copy()

    g = g.set_index("image_id").reindex(image_ids)

    w = weights.astype(float)

    acc = {}

    for c in COUNT_COLS:

        acc[c] = float(np.nansum(pd.to_numeric(g[c], errors="coerce").fillna(0.0).to_numpy(dtype=float) * w))

    precision = acc["tp_known"] / (acc["tp_known"] + acc["fp_known"]) if (acc["tp_known"] + acc["fp_known"]) else 0.0

    recall = acc["tp_known"] / acc["num_known_gt"] if acc["num_known_gt"] else 0.0

    unknown_reject = 1.0 - (acc["unknown_false_accept_objects"] / acc["num_unknown_gt"]) if acc["num_unknown_gt"] else 1.0

    out = {

        **acc,

        "known_precision": float(precision),

        "known_recall": float(recall),

        "unknown_reject_rate_object_level": float(unknown_reject),

        "precision_recall_unknown_balanced_score": harmonic3(precision, recall, unknown_reject),

    }

    for c in AP_COLS:

        vals = pd.to_numeric(g[c], errors="coerce").to_numpy(dtype=float)

        mask = np.isfinite(vals)

        denom = float(np.sum(w[mask]))

        out[c] = float(np.sum(vals[mask] * w[mask]) / denom) if denom > 0 else float("nan")

    return out



def summarize_samples(samples: pd.DataFrame) -> pd.DataFrame:

    rows = []

    for method, g in samples.groupby("method", sort=False):

        row = {"method": method, "n_boot": int(g["boot_idx"].nunique())}

        for m in METRIC_COLS:

            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)

            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")

            row[f"{m}_ci025"] = float(np.quantile(vals, 0.025)) if len(vals) else float("nan")

            row[f"{m}_ci975"] = float(np.quantile(vals, 0.975)) if len(vals) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)



def summarize_deltas(samples: pd.DataFrame, baseline: str) -> tuple[pd.DataFrame, pd.DataFrame]:

    rows = []

    for boot_idx, g in samples.groupby("boot_idx", sort=False):

        base = g[g["method"].eq(baseline)]

        if base.empty:

            continue

        base = base.iloc[0]

        for _, r in g.iterrows():

            if r["method"] == baseline:

                continue

            out = {"boot_idx": int(boot_idx), "method": r["method"], "baseline": baseline}

            for m in METRIC_COLS:

                out[f"delta_{m}"] = float(r[m]) - float(base[m])

            rows.append(out)

    delta = pd.DataFrame(rows)

    summary_rows = []

    for method, g in delta.groupby("method", sort=False):

        row = {"method": method, "baseline": baseline, "n_boot": int(g["boot_idx"].nunique())}

        for c in [c for c in delta.columns if c.startswith("delta_")]:

            vals = pd.to_numeric(g[c], errors="coerce").dropna().to_numpy(dtype=float)

            row[f"{c}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")

            row[f"{c}_ci025"] = float(np.quantile(vals, 0.025)) if len(vals) else float("nan")

            row[f"{c}_ci975"] = float(np.quantile(vals, 0.975)) if len(vals) else float("nan")

            if c in {"delta_unknown_false_accept_objects", "delta_background_false_accept_count"}:

                row[f"{c}_p_improved"] = float(np.mean(vals < 0)) if len(vals) else float("nan")

            else:

                row[f"{c}_p_improved"] = float(np.mean(vals > 0)) if len(vals) else float("nan")

        summary_rows.append(row)

    return pd.DataFrame(summary_rows), delta



def write_markdown(path: Path, summary: pd.DataFrame, delta: pd.DataFrame, methods: List[dict], report: dict) -> None:

    lines = [

        "# Step12D COCO-Val-OpenSet-5K Bootstrap Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Completed bootstrap samples: `{report['n_boot_completed']} / {report['n_boot_requested']}`",

        f"- AP bootstrap mode: `{report['ap_bootstrap_mode']}`",

        "",

        "## Bootstrap Method Specs",

        "",

        pd.DataFrame(methods).to_markdown(index=False),

        "",

        "## Bootstrap Summary",

        "",

        summary.to_markdown(index=False) if len(summary) else "_No summary rows._",

        "",

        "## Delta vs Raw Baseline",

        "",

        delta.to_markdown(index=False) if len(delta) else "_No delta rows._",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--scored_csv", type=Path, default=DEFAULT_STEP12C_ROOT / "csv" / "step12c_geometry_scored_candidates.csv")

    parser.add_argument("--compact_csv", type=Path, default=DEFAULT_STEP12C_ROOT / "csv" / "step12c_compact_paper_metrics.csv")

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step12a_scene_split.csv")

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_PROTOCOL_ROOT / "annotations")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step12a_class_map.csv")

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--n_boot", type=int, default=1000)

    parser.add_argument("--max_boot_per_run", type=int, default=0)

    parser.add_argument("--seed", type=int, default=52)

    parser.add_argument("--force_recompute_precompute", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    print("\n========== Step12D COCO-Val-OpenSet-5K Bootstrap ==========")


    compact = read_csv(args.compact_csv)

    methods = choose_methods(compact)

    baseline = methods[0]["method"]

    pd.DataFrame(methods).to_csv(csv_dir / "step12d_bootstrap_method_specs.csv", index=False, encoding="utf-8-sig")


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root, split_filter="all")

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    image_ids = sorted(split_df[split_df["split"].astype(str).str.lower().eq("test")]["image_id"].map(normalize_image_id).tolist())

    scored = read_csv(args.scored_csv)

    scored["image_id"] = scored["image_id"].map(normalize_image_id)

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    scored = scored[scored["split"].astype(str).str.lower().eq("test")].copy()


    per_image_path = csv_dir / "step12d_per_image_cached_metrics.csv"

    if args.force_recompute_precompute or not per_image_path.exists():

        per_image = precompute_per_image(scored, gt_test, image_ids, known_classes, methods, iou_thr=0.50)

        per_image.to_csv(per_image_path, index=False, encoding="utf-8-sig")

    else:

        per_image = read_csv(per_image_path)


    samples_path = csv_dir / "step12d_bootstrap_samples.csv"

    existing = read_csv(samples_path) if samples_path.exists() else pd.DataFrame()

    done = set(existing["boot_idx"].astype(int).tolist()) if len(existing) else set()

    remaining = [i for i in range(int(args.n_boot)) if i not in done]

    if args.max_boot_per_run and args.max_boot_per_run > 0:

        remaining = remaining[: int(args.max_boot_per_run)]


    rng_master = np.random.default_rng(int(args.seed))

    seeds = rng_master.integers(0, 2**31 - 1, size=int(args.n_boot), dtype=np.int64)

    image_ids_arr = np.asarray(image_ids, dtype=object)

    new_rows = []

    for boot_idx in tqdm(remaining, desc="Step12D bootstrap samples"):

        rng = np.random.default_rng(int(seeds[boot_idx]))

        sampled_idx = rng.integers(0, len(image_ids_arr), size=len(image_ids_arr))

        weights = np.bincount(sampled_idx, minlength=len(image_ids_arr)).astype(float)

        for spec in methods:

            method = str(spec["method"])

            vals = aggregate_weighted(per_image, method, weights, image_ids)

            new_rows.append({"boot_idx": int(boot_idx), "method": method, **vals})

        if len(new_rows) >= 300:

            merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if len(existing) else pd.DataFrame(new_rows)

            merged = merged.drop_duplicates(subset=["boot_idx", "method"], keep="last").sort_values(["boot_idx", "method"])

            merged.to_csv(samples_path, index=False, encoding="utf-8-sig")

            existing = merged

            new_rows = []

    if new_rows:

        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if len(existing) else pd.DataFrame(new_rows)

        merged = merged.drop_duplicates(subset=["boot_idx", "method"], keep="last").sort_values(["boot_idx", "method"])

        merged.to_csv(samples_path, index=False, encoding="utf-8-sig")

        existing = merged


    summary = summarize_samples(existing)

    delta_summary, delta_samples = summarize_deltas(existing, baseline=baseline)

    summary.to_csv(csv_dir / "step12d_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    delta_summary.to_csv(csv_dir / "step12d_bootstrap_delta.csv", index=False, encoding="utf-8-sig")

    delta_samples.to_csv(csv_dir / "step12d_bootstrap_delta_samples.csv", index=False, encoding="utf-8-sig")


    report = {

        "method": "Step12D COCO-Val-OpenSet-5K paired image bootstrap",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "inputs": {

            "scored_csv": str(args.scored_csv),

            "compact_csv": str(args.compact_csv),

            "split_csv": str(args.split_csv),

            "annotation_dir": str(args.annotation_dir),

            "class_map_csv": str(args.class_map_csv),

        },

        "n_boot_requested": int(args.n_boot),

        "n_boot_completed": int(existing["boot_idx"].nunique()) if len(existing) else 0,

        "seed": int(args.seed),

        "baseline_method": baseline,

        "methods": methods,

        "paired_image_bootstrap": True,

        "ap_bootstrap_mode": "cached per-image AP approximation; exact full-test AP is reported in Step12C",

        "uses_cached_per_image_counts": True,

        "outputs": {

            "bootstrap_summary": str(csv_dir / "step12d_bootstrap_summary.csv"),

            "bootstrap_delta": str(csv_dir / "step12d_bootstrap_delta.csv"),

            "bootstrap_samples": str(csv_dir / "step12d_bootstrap_samples.csv"),

            "per_image_cached_metrics": str(per_image_path),

            "summary": str(args.output_root / "step12d_summary.md"),

            "integrity_report": str(args.output_root / "step12d_integrity_report.json"),

        },

    }

    with open(args.output_root / "step12d_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)

    write_markdown(args.output_root / "step12d_summary.md", summary, delta_summary, methods, report)


    print("\n========== Step12D completed ==========")

    print(delta_summary.to_string(index=False))



if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("Interrupted by user.", file=sys.stderr)

        raise

