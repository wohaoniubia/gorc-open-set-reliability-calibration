from __future__ import annotations


import argparse

import math

import sys

from pathlib import Path


import numpy as np

import pandas as pd



ROOT = Path(__file__).resolve().parents[2]

SCRIPTS = ROOT / "scripts"

if str(SCRIPTS) not in sys.path:

    sys.path.insert(0, str(SCRIPTS))


from step8j_yoloworld_lvis_openvoc_baseline import (

    bbox_iou,

    clean_columns,

    load_annotations,

    load_known_classes,

    load_split,

    safe_class_name,

)



BUNDLE = ROOT / "GORC_paper_release_bundle"

ROUND3 = ROOT / "outputs" / "gorc_major_revision_round3"

OUT = ROUND3 / "02_ap_bootstrap"


PROTO = BUNDLE / "data_outputs" / "coco_main" / "step12a_protocol"

SCORED = BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_geometry_scored_candidates.csv"

COMPACT = BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_compact_paper_metrics.csv"


IOU_THRESHOLDS = [round(x, 2) for x in np.arange(0.50, 0.96, 0.05)]

RECALL_POINTS = np.linspace(0.0, 1.0, 101)

POLICIES = [

    ("raw", "step12b_yoloworld_l_raw_reproduced", "raw_score"),

    ("RF", "step12c_geometry_max_cal_balanced", "geometry_alpha_0p800"),

    ("AP-C", "step12c_geometry_ap_constrained_balanced_AP50tol0_APtol0", "class_reliability_gamma_0p250"),

]



def finite(value, default: float = float("nan")) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def gt_key(row: pd.Series) -> tuple[str, object]:

    return (str(row.image_id), row.gt_id)



def candidate_gt_lists(dets: pd.DataFrame, gt_cls_by_image: dict[str, pd.DataFrame]) -> list[list[tuple[tuple[str, object], float]]]:

    out: list[list[tuple[tuple[str, object], float]]] = []

    for _, d in dets.iterrows():

        gimg = gt_cls_by_image.get(str(d.image_id))

        pairs: list[tuple[tuple[str, object], float]] = []

        if gimg is not None and not gimg.empty:

            dbox = (float(d.x1), float(d.y1), float(d.x2), float(d.y2))

            for _, gr in gimg.iterrows():

                iou = bbox_iou(dbox, (gr.x1, gr.y1, gr.x2, gr.y2))

                if iou > 0:

                    pairs.append((gt_key(gr), float(iou)))

        pairs.sort(key=lambda x: x[1], reverse=True)

        out.append(pairs)

    return out



def tp_flags_for_threshold(

    candidate_lists: list[list[tuple[tuple[str, object], float]]],

    iou_thr: float,

) -> np.ndarray:

    matched: set[tuple[str, object]] = set()

    tp = np.zeros(len(candidate_lists), dtype=np.float32)

    for i, pairs in enumerate(candidate_lists):

        for key, iou in pairs:

            if iou < iou_thr:

                break

            if key not in matched:

                matched.add(key)

                tp[i] = 1.0

                break

    return tp



def prepare_policy_class_payloads(scored: pd.DataFrame, gt_test: pd.DataFrame, known_classes: list[str]) -> dict:

    image_ids = sorted(gt_test["image_id"].astype(str).unique().tolist())

    image_to_idx = {img: i for i, img in enumerate(image_ids)}

    gt_known = gt_test[gt_test["gt_is_known"]].copy()

    gt_counts = {

        cls: np.bincount(

            gt_known[gt_known["gt_label"].map(safe_class_name).eq(cls)]["image_id"].astype(str).map(image_to_idx).to_numpy(),

            minlength=len(image_ids),

        ).astype(np.float32)

        for cls in known_classes

    }

    payloads: dict = {"image_ids": image_ids, "gt_counts": gt_counts, "policies": {}}

    gt_by_class_image = {

        cls: {img: g.copy() for img, g in gt_known[gt_known["gt_label"].map(safe_class_name).eq(cls)].groupby("image_id", sort=False)}

        for cls in known_classes

    }

    test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    test["pred_label"] = test["pred_label"].map(safe_class_name)

    for policy_label, _, score_col in POLICIES:

        policy_payload = {}

        for cls in known_classes:

            dets = test[test["pred_label"].eq(cls)].copy()

            dets["_score"] = pd.to_numeric(dets[score_col], errors="coerce").fillna(-math.inf)

            dets = dets.sort_values("_score", ascending=False).reset_index(drop=True)

            det_image_idx = dets["image_id"].astype(str).map(image_to_idx).fillna(-1).astype(int).to_numpy()

            cand_lists = candidate_gt_lists(dets, gt_by_class_image[cls])

            tp_by_iou = {iou: tp_flags_for_threshold(cand_lists, iou) for iou in IOU_THRESHOLDS}

            policy_payload[cls] = {

                "image_idx": det_image_idx,

                "tp_by_iou": tp_by_iou,

            }

        payloads["policies"][policy_label] = policy_payload

    return payloads



def ap_from_flags_for_batch(tp_flags: np.ndarray, image_idx: np.ndarray, image_weights: np.ndarray, npos: np.ndarray) -> np.ndarray:

    if len(tp_flags) == 0:

        return np.zeros(image_weights.shape[0], dtype=np.float64)

    weights = image_weights[:, image_idx].astype(np.float64)

    tp_w = weights * tp_flags.reshape(1, -1)

    fp_w = weights * (1.0 - tp_flags.reshape(1, -1))

    tp_cum = np.cumsum(tp_w, axis=1)

    fp_cum = np.cumsum(fp_w, axis=1)

    denom = np.maximum(tp_cum + fp_cum, 1e-12)

    precision = tp_cum / denom

    recall = tp_cum / np.maximum(npos.reshape(-1, 1), 1e-12)

    ap = np.zeros(image_weights.shape[0], dtype=np.float64)

    valid = npos > 0

    for t in RECALL_POINTS:

        mask = recall >= t

        vals = np.where(mask, precision, 0.0).max(axis=1)

        ap += vals

    ap = ap / len(RECALL_POINTS)

    ap[~valid] = np.nan

    return ap



def evaluate_batch(payloads: dict, sample_indices: list[int], seed: int) -> list[dict]:

    num_images = len(payloads["image_ids"])

    weights = np.zeros((len(sample_indices), num_images), dtype=np.float32)

    for row_i, sample_idx in enumerate(sample_indices):

        rng = np.random.default_rng(int(seed) + int(sample_idx))

        draw = rng.integers(0, num_images, size=num_images)

        weights[row_i] = np.bincount(draw, minlength=num_images).astype(np.float32)


    rows = [{"sample_idx": int(s)} for s in sample_indices]

    for policy_label, class_payloads in payloads["policies"].items():

        ap_by_iou: dict[float, list[np.ndarray]] = {iou: [] for iou in IOU_THRESHOLDS}

        for cls, cls_payload in class_payloads.items():

            npos = weights @ payloads["gt_counts"][cls]

            image_idx = cls_payload["image_idx"]

            for iou in IOU_THRESHOLDS:

                ap_vals = ap_from_flags_for_batch(cls_payload["tp_by_iou"][iou], image_idx, weights, npos)

                ap_by_iou[iou].append(ap_vals)

        summary = {}

        for iou in IOU_THRESHOLDS:

            arr = np.vstack(ap_by_iou[iou])

            summary[f"AP{int(iou * 100)}"] = np.nanmean(arr, axis=0)

        summary["AP"] = np.nanmean(np.vstack([summary[f"AP{int(iou * 100)}"] for iou in IOU_THRESHOLDS]), axis=0)

        for row_i, row in enumerate(rows):

            row[f"{policy_label}_AP50"] = float(summary["AP50"][row_i])

            row[f"{policy_label}_AP75"] = float(summary["AP75"][row_i])

            row[f"{policy_label}_AP"] = float(summary["AP"][row_i])


    for row in rows:

        for policy_label in ["RF", "AP-C"]:

            for metric in ["AP50", "AP75", "AP"]:

                row[f"delta_{policy_label}_minus_raw_{metric}"] = row[f"{policy_label}_{metric}"] - row[f"raw_{metric}"]

    return rows



def evaluate_full_ones(payloads: dict) -> dict:

    num_images = len(payloads["image_ids"])

    weights = np.ones((1, num_images), dtype=np.float32)

    rows = [{"sample_idx": -1}]

    for policy_label, class_payloads in payloads["policies"].items():

        ap_by_iou: dict[float, list[np.ndarray]] = {iou: [] for iou in IOU_THRESHOLDS}

        for cls, cls_payload in class_payloads.items():

            npos = weights @ payloads["gt_counts"][cls]

            image_idx = cls_payload["image_idx"]

            for iou in IOU_THRESHOLDS:

                ap_by_iou[iou].append(ap_from_flags_for_batch(cls_payload["tp_by_iou"][iou], image_idx, weights, npos))

        summary = {}

        for iou in IOU_THRESHOLDS:

            summary[f"AP{int(iou * 100)}"] = float(np.nanmean(np.vstack(ap_by_iou[iou]), axis=0)[0])

        summary["AP"] = float(np.nanmean([summary[f"AP{int(iou * 100)}"] for iou in IOU_THRESHOLDS]))

        rows[0][f"{policy_label}_AP50"] = summary["AP50"]

        rows[0][f"{policy_label}_AP75"] = summary["AP75"]

        rows[0][f"{policy_label}_AP"] = summary["AP"]

    return rows[0]



def write_summary(df: pd.DataFrame, full_check: pd.DataFrame, n: int, seed: int) -> None:

    rows = []

    for policy_label in ["RF", "AP-C"]:

        for metric in ["AP50", "AP75", "AP"]:

            col = f"delta_{policy_label}_minus_raw_{metric}"

            vals = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy()

            rows.append(

                {

                    "comparison": f"{policy_label} - raw",

                    "metric": metric,

                    "N": int(len(vals)),

                    "mean_delta": float(np.mean(vals)),

                    "ci025": float(np.quantile(vals, 0.025)),

                    "ci975": float(np.quantile(vals, 0.975)),

                }

            )

    summary = pd.DataFrame(rows)

    summary.to_csv(OUT / "exact_ap_bootstrap_summary_ci.csv", index=False, encoding="utf-8-sig")

    md = [

        "# Exact COCO AP Bootstrap Summary",

        "",

        f"Status: {'PASS' if len(df) >= n else 'PARTIAL'}",

        "",

        f"- Bootstrap samples requested: `{n}`",

        f"- Bootstrap samples completed: `{len(df)}`",

        f"- Seed base: `{seed}`",

        "- Resampling unit: held-out test image ID, sampled with replacement.",

        "- Selected policies are fixed; no policy is reselected inside bootstrap samples.",

        "- Duplicate-image-safe implementation: image multiplicities are used as weights for both detections and ground truth, preserving one-to-one matching capacity within each image.",

        "",

        "## Full-Sample AP Check",

        "",

        full_check.to_markdown(index=False),

        "",

        "## Delta Confidence Intervals",

        "",

        summary.to_markdown(index=False),

        "",

    ]

    (OUT / "exact_ap_bootstrap_summary.md").write_text("\n".join(md), encoding="utf-8")



def main() -> int:

    parser = argparse.ArgumentParser()

    parser.add_argument("--n", type=int, default=500)

    parser.add_argument("--batch", type=int, default=25)

    parser.add_argument("--seed", type=int, default=20260512)

    args = parser.parse_args()


    OUT.mkdir(parents=True, exist_ok=True)

    out_csv = OUT / "exact_ap_bootstrap_coco.csv"

    split = load_split(PROTO / "csv" / "step12a_scene_split.csv", ROOT, split_filter="all")

    known = [safe_class_name(x) for x in load_known_classes(PROTO / "csv" / "step12a_class_map.csv")]

    gt = load_annotations(PROTO / "annotations", split, known)

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    gt_test["gt_label"] = gt_test["gt_label"].map(safe_class_name)

    scored = clean_columns(pd.read_csv(SCORED, encoding="utf-8-sig"))

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    for _, _, score_col in POLICIES:

        scored[score_col] = pd.to_numeric(scored[score_col], errors="coerce").fillna(-math.inf)


    payloads = prepare_policy_class_payloads(scored, gt_test, known)

    full = evaluate_full_ones(payloads)

    compact = clean_columns(pd.read_csv(COMPACT, encoding="utf-8-sig"))

    full_rows = []

    for policy_label, method, _ in POLICIES:

        cr = compact[compact["method"].astype(str).eq(method)].iloc[0]

        for metric in ["AP50", "AP75", "AP"]:

            full_rows.append(

                {

                    "policy": policy_label,

                    "metric": metric,

                    "weighted_full": full[f"{policy_label}_{metric}"],

                    "compact": finite(cr.get(metric)),

                    "abs_delta": abs(full[f"{policy_label}_{metric}"] - finite(cr.get(metric))),

                }

            )

    full_check = pd.DataFrame(full_rows)

    if not (pd.to_numeric(full_check["abs_delta"], errors="coerce") < 1e-8).all():

        blocker = OUT / "EXACT_AP_BOOTSTRAP_BLOCKER.md"

        blocker.write_text(

            "# Exact AP Bootstrap Blocker\n\n"

            "Full-sample weighted AP did not match compact AP exactly, so bootstrap intervals were not generated.\n\n"

            + full_check.to_markdown(index=False)

            + "\n",

            encoding="utf-8",

        )

        return 2


    completed = set()

    if out_csv.exists():

        old = pd.read_csv(out_csv, encoding="utf-8-sig")

        if "sample_idx" in old.columns:

            completed = set(pd.to_numeric(old["sample_idx"], errors="coerce").dropna().astype(int).tolist())

    for start in range(0, int(args.n), int(args.batch)):

        sample_indices = [i for i in range(start, min(start + int(args.batch), int(args.n))) if i not in completed]

        if not sample_indices:

            continue

        batch_rows = evaluate_batch(payloads, sample_indices, int(args.seed))

        pd.DataFrame(batch_rows).to_csv(

            out_csv,

            mode="a",

            header=not out_csv.exists(),

            index=False,

            encoding="utf-8-sig",

        )

        print(f"completed bootstrap samples through {max(sample_indices)}", flush=True)


    df = clean_columns(pd.read_csv(out_csv, encoding="utf-8-sig")).sort_values("sample_idx").reset_index(drop=True)

    write_summary(df, full_check, int(args.n), int(args.seed))

    return 0



if __name__ == "__main__":

    raise SystemExit(main())

