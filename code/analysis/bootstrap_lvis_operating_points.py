


"""
Step9e: Paired image bootstrap for LVIS-Clear-Mini-300.

This implementation is resumable and faster than recomputing object matching
from scratch for every bootstrap sample. Detection outcomes for AP are
precomputed per image/class/IoU, then each bootstrap sample applies image
multiplicity weights.
"""


from __future__ import annotations


import argparse

import json

import math

from pathlib import Path

from typing import Dict, List, Sequence


import numpy as np

import pandas as pd

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import (

    bbox_iou,

    clean_columns,

    evaluate_detections,

    load_annotations,

    load_known_classes,

    load_split,

    normalize_image_id,

    safe_class_name,

)



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_SPLIT_CSV = DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv"

DEFAULT_ANN_DIR = DEFAULT_PROTOCOL_ROOT / "annotations"

DEFAULT_CLASS_MAP_CSV = DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv"

DEFAULT_SCORED_CSV = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300" / "csv" / "step9c_geometry_scored_candidates.csv"

DEFAULT_COMPACT_CSV = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300" / "csv" / "step9c_compact_paper_metrics.csv"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step9e_bootstrap_lvis300"

AP_IOUS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def ap101_from_weighted(tp: np.ndarray, fp: np.ndarray, npos: float) -> float:

    if npos <= 0:

        return float("nan")

    if len(tp) == 0:

        return 0.0

    tp_cum = np.cumsum(tp.astype(float))

    fp_cum = np.cumsum(fp.astype(float))

    rec = tp_cum / max(float(npos), 1e-12)

    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)

    ap = 0.0

    for t in np.linspace(0.0, 1.0, 101):

        ap += float(np.max(prec[rec >= t])) if np.any(rec >= t) else 0.0

    return ap / 101.0



def method_specs(compact: pd.DataFrame) -> List[dict]:

    sub = compact[compact["iou_threshold"].astype(float).round(4).eq(0.50)].copy()

    keep = []

    for _, r in sub.iterrows():

        method = str(r["method"])

        if method == "step9b_raw_global_reproduced" or method.startswith("step9c_geometry_"):

            keep.append(

                {

                    "method": method,

                    "selection_policy": str(r.get("selection_policy", "")),

                    "score_col": str(r["score_col"]),

                    "threshold": float(r["threshold"]),

                }

            )

    if not any(m["method"] == "step9b_raw_global_reproduced" for m in keep):

        raise ValueError("Missing raw baseline row in compact metrics.")

    return keep



def precompute_operating_counts(scored: pd.DataFrame, gt_test: pd.DataFrame, image_ids: Sequence[str], methods: List[dict], iou_thr: float) -> pd.DataFrame:

    rows = []

    for spec in tqdm(methods, desc="Precompute operating counts"):

        score_col = spec["score_col"]

        threshold = float(spec["threshold"])

        for image_id in image_ids:

            gt_img = gt_test[gt_test["image_id"].eq(image_id)].copy()

            dets = scored[scored["image_id"].eq(image_id)].copy()

            dets = dets[pd.to_numeric(dets[score_col], errors="coerce") >= threshold].copy()

            dets["score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(0.0)

            metrics, _ = evaluate_detections(dets, gt_img, iou_thr=float(iou_thr))

            metrics.update({"method": spec["method"], "image_id": image_id})

            rows.append(metrics)

    return pd.DataFrame(rows)



def precompute_ap_outcomes(scored: pd.DataFrame, gt_test: pd.DataFrame, known_classes: Sequence[str], methods: List[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:

    gt_known = gt_test[gt_test["gt_is_known"]].copy()

    gt_count_rows = []

    for cls in known_classes:

        gcls = gt_known[gt_known["gt_label"].eq(cls)].copy()

        for image_id, g in gcls.groupby("image_id", sort=False):

            gt_count_rows.append({"class_name": cls, "image_id": image_id, "num_gt": int(len(g))})

    gt_counts = pd.DataFrame(gt_count_rows)


    det_rows = []

    for spec in tqdm(methods, desc="Precompute AP detection outcomes"):

        method = spec["method"]

        score_col = spec["score_col"]

        dets_all = scored.copy()

        dets_all["score"] = pd.to_numeric(dets_all[score_col], errors="coerce").fillna(0.0)

        for cls in known_classes:

            gt_cls = gt_known[gt_known["gt_label"].eq(cls)].copy()

            det_cls = dets_all[dets_all["pred_label"].eq(cls)].copy()

            det_cls = det_cls.sort_values("score", ascending=False).reset_index(drop=True)

            for iou_thr in AP_IOUS:

                matched = set()

                gt_by_image = {img: g.copy() for img, g in gt_cls.groupby("image_id", sort=False)}

                for det_idx, pr in det_cls.iterrows():

                    image_id = normalize_image_id(pr.image_id)

                    gimg = gt_by_image.get(image_id, gt_cls.iloc[0:0])

                    best_iou, best_key = 0.0, None

                    pbox = (float(pr.x1), float(pr.y1), float(pr.x2), float(pr.y2))

                    for _, gr in gimg.iterrows():

                        key = (gr.image_id, gr.gt_id)

                        if key in matched:

                            continue

                        iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

                        if iou > best_iou:

                            best_iou, best_key = float(iou), key

                    tp = 1.0 if best_key is not None and best_iou >= float(iou_thr) else 0.0

                    if tp:

                        matched.add(best_key)

                    det_rows.append(

                        {

                            "method": method,

                            "class_name": cls,

                            "iou_threshold": float(iou_thr),

                            "image_id": image_id,

                            "score": float(pr.score),

                            "tp": float(tp),

                            "fp": float(1.0 - tp),

                        }

                    )

    return pd.DataFrame(det_rows), gt_counts



def weighted_ap_summary(det_outcomes: pd.DataFrame, gt_counts: pd.DataFrame, known_classes: Sequence[str], method: str, weights: Dict[str, int]) -> dict:

    summary = {}

    method_rows = det_outcomes[det_outcomes["method"].eq(method)].copy()

    for iou_thr in AP_IOUS:

        per_class_ap = []

        for cls in known_classes:

            gt_cls = gt_counts[gt_counts["class_name"].eq(cls)].copy()

            npos = 0.0

            for _, r in gt_cls.iterrows():

                npos += float(r["num_gt"]) * float(weights.get(str(r["image_id"]), 0))

            det_cls = method_rows[

                method_rows["class_name"].eq(cls)

                & method_rows["iou_threshold"].astype(float).round(4).eq(float(iou_thr))

            ].copy()

            if len(det_cls):

                det_cls["w"] = det_cls["image_id"].map(lambda x: int(weights.get(str(x), 0)))

                det_cls = det_cls[det_cls["w"] > 0].sort_values("score", ascending=False)

                tp = det_cls["tp"].to_numpy(dtype=float) * det_cls["w"].to_numpy(dtype=float)

                fp = det_cls["fp"].to_numpy(dtype=float) * det_cls["w"].to_numpy(dtype=float)

            else:

                tp = np.asarray([], dtype=float)

                fp = np.asarray([], dtype=float)

            per_class_ap.append(ap101_from_weighted(tp, fp, npos))

        vals = [v for v in per_class_ap if not math.isnan(v)]

        summary[f"AP{int(iou_thr * 100)}"] = float(np.mean(vals)) if vals else float("nan")

    vals_all = [summary[f"AP{int(i * 100)}"] for i in AP_IOUS if not math.isnan(summary[f"AP{int(i * 100)}"])]

    summary["AP"] = float(np.mean(vals_all)) if vals_all else float("nan")

    return summary



def operating_from_weighted_counts(per_image: pd.DataFrame, method: str, weights: Dict[str, int]) -> dict:

    g = per_image[per_image["method"].eq(method)].copy()

    acc = {}

    count_cols = [

        "num_known_gt",

        "num_unknown_gt",

        "num_accepted_detections",

        "tp_known",

        "fp_known",

        "fn_known",

        "unknown_false_accept_objects",

        "background_false_accept_count",

    ]

    for c in count_cols:

        acc[c] = float(sum(float(r[c]) * float(weights.get(str(r["image_id"]), 0)) for _, r in g.iterrows()))

    precision = acc["tp_known"] / (acc["tp_known"] + acc["fp_known"]) if (acc["tp_known"] + acc["fp_known"]) else 0.0

    recall = acc["tp_known"] / acc["num_known_gt"] if acc["num_known_gt"] else 0.0

    unknown_reject = 1.0 - (acc["unknown_false_accept_objects"] / acc["num_unknown_gt"]) if acc["num_unknown_gt"] else 1.0

    h2 = 0.0 if recall + unknown_reject <= 0 else 2 * recall * unknown_reject / (recall + unknown_reject)

    vals = [precision, recall, unknown_reject]

    h3 = 0.0 if any(v <= 0 for v in vals) else 3.0 / sum(1.0 / v for v in vals)

    return {

        **acc,

        "known_precision": float(precision),

        "known_recall": float(recall),

        "unknown_reject_rate_object_level": float(unknown_reject),

        "open_set_hscore_object_level": float(h2),

        "precision_recall_unknown_balanced_score": float(h3),

    }



def summarize_samples(samples: pd.DataFrame) -> pd.DataFrame:

    metrics = [

        "precision_recall_unknown_balanced_score",

        "known_precision",

        "known_recall",

        "unknown_false_accept_objects",

        "background_false_accept_count",

        "AP50",

        "AP75",

        "AP",

    ]

    rows = []

    for method, g in samples.groupby("method", sort=False):

        row = {"method": method, "n_boot": int(g["boot_idx"].nunique())}

        for m in metrics:

            vals = pd.to_numeric(g[m], errors="coerce").dropna().to_numpy(dtype=float)

            row[f"{m}_mean"] = float(np.mean(vals)) if len(vals) else float("nan")

            row[f"{m}_ci025"] = float(np.quantile(vals, 0.025)) if len(vals) else float("nan")

            row[f"{m}_ci975"] = float(np.quantile(vals, 0.975)) if len(vals) else float("nan")

        rows.append(row)

    return pd.DataFrame(rows)



def summarize_deltas(samples: pd.DataFrame, baseline: str) -> pd.DataFrame:

    metrics = [

        "precision_recall_unknown_balanced_score",

        "known_precision",

        "known_recall",

        "unknown_false_accept_objects",

        "background_false_accept_count",

        "AP50",

        "AP75",

        "AP",

    ]

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

            for m in metrics:

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



def write_markdown(path: Path, summary: pd.DataFrame, delta: pd.DataFrame) -> None:

    lines = [

        "# Step9e Paired Image Bootstrap Summary",

        "",

        "## Bootstrap Summary",

        "",

        summary.to_markdown(index=False),

        "",

        "## Delta vs Raw Baseline",

        "",

        delta.to_markdown(index=False),

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--scored_csv", type=Path, default=DEFAULT_SCORED_CSV)

    parser.add_argument("--compact_csv", type=Path, default=DEFAULT_COMPACT_CSV)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_SPLIT_CSV)

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_ANN_DIR)

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_CLASS_MAP_CSV)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--n_boot", type=int, default=1000)

    parser.add_argument("--max_boot_per_run", type=int, default=0)

    parser.add_argument("--seed", type=int, default=31)

    parser.add_argument("--baseline_method", type=str, default="step9b_raw_global_reproduced")

    parser.add_argument("--force_recompute_precompute", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    print("\n========== Step9e Weighted Paired Image Bootstrap ==========")


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root)

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    image_ids = sorted(split_df[split_df["split"].astype(str).str.lower().eq("test")]["image_id"].map(normalize_image_id).tolist())

    scored = read_csv(args.scored_csv)

    scored["image_id"] = scored["image_id"].map(normalize_image_id)

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    scored = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    compact = read_csv(args.compact_csv)

    methods = method_specs(compact)


    per_image_path = csv_dir / "step9e_per_image_operating_counts.csv"

    ap_outcomes_path = csv_dir / "step9e_precomputed_ap_detection_outcomes.csv"

    gt_counts_path = csv_dir / "step9e_ap_gt_counts.csv"

    if args.force_recompute_precompute or not (per_image_path.exists() and ap_outcomes_path.exists() and gt_counts_path.exists()):

        per_image = precompute_operating_counts(scored, gt_test, image_ids, methods, iou_thr=0.50)

        det_outcomes, gt_counts = precompute_ap_outcomes(scored, gt_test, known_classes, methods)

        per_image.to_csv(per_image_path, index=False, encoding="utf-8-sig")

        det_outcomes.to_csv(ap_outcomes_path, index=False, encoding="utf-8-sig")

        gt_counts.to_csv(gt_counts_path, index=False, encoding="utf-8-sig")

    else:

        per_image = read_csv(per_image_path)

        det_outcomes = read_csv(ap_outcomes_path)

        gt_counts = read_csv(gt_counts_path)


    samples_path = csv_dir / "step9e_bootstrap_samples.csv"

    existing = read_csv(samples_path) if samples_path.exists() else pd.DataFrame()

    done = set(existing["boot_idx"].astype(int).tolist()) if len(existing) else set()

    remaining = [i for i in range(int(args.n_boot)) if i not in done]

    if args.max_boot_per_run and args.max_boot_per_run > 0:

        remaining = remaining[: int(args.max_boot_per_run)]


    new_rows = []

    rng_master = np.random.default_rng(int(args.seed))

    seeds = rng_master.integers(0, 2**31 - 1, size=int(args.n_boot), dtype=np.int64)

    for boot_idx in tqdm(remaining, desc="Step9e bootstrap samples"):

        rng = np.random.default_rng(int(seeds[boot_idx]))

        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)

        weights = {img: 0 for img in image_ids}

        for img in sampled:

            weights[str(img)] = weights.get(str(img), 0) + 1

        for spec in methods:

            method = spec["method"]

            op = operating_from_weighted_counts(per_image, method, weights)

            ap = weighted_ap_summary(det_outcomes, gt_counts, known_classes, method, weights)

            new_rows.append({"boot_idx": int(boot_idx), "method": method, **op, **ap})

        if len(new_rows) >= 100:

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

    delta_summary, delta_samples = summarize_deltas(existing, baseline=args.baseline_method)

    summary.to_csv(csv_dir / "step9e_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    delta_summary.to_csv(csv_dir / "step9e_bootstrap_delta.csv", index=False, encoding="utf-8-sig")

    delta_samples.to_csv(csv_dir / "step9e_bootstrap_delta_samples.csv", index=False, encoding="utf-8-sig")

    write_markdown(args.output_root / "step9e_summary.md", summary, delta_summary)


    report = {

        "method": "Step9e Weighted Paired Image Bootstrap for LVIS-Clear-Mini-300",

        "project_root": str(args.project_root),

        "scored_csv": str(args.scored_csv),

        "compact_csv": str(args.compact_csv),

        "split_csv": str(args.split_csv),

        "annotation_dir": str(args.annotation_dir),

        "class_map_csv": str(args.class_map_csv),

        "output_root": str(args.output_root),

        "n_boot_requested": int(args.n_boot),

        "n_boot_completed": int(existing["boot_idx"].nunique()) if len(existing) else 0,

        "seed": int(args.seed),

        "baseline_method": args.baseline_method,

        "methods": methods,

        "outputs": {

            "bootstrap_summary": str(csv_dir / "step9e_bootstrap_summary.csv"),

            "bootstrap_delta": str(csv_dir / "step9e_bootstrap_delta.csv"),

            "bootstrap_samples": str(samples_path),

            "bootstrap_delta_samples": str(csv_dir / "step9e_bootstrap_delta_samples.csv"),

            "per_image_operating_counts": str(per_image_path),

            "ap_detection_outcomes": str(ap_outcomes_path),

            "ap_gt_counts": str(gt_counts_path),

            "markdown_summary": str(args.output_root / "step9e_summary.md"),

        },

    }

    with open(args.output_root / "step9e_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)


    print("\n========== Step9e completed ==========")

    print(f"Completed bootstrap samples: {report['n_boot_completed']}/{args.n_boot}")

    print(delta_summary.to_string(index=False))



if __name__ == "__main__":

    main()

