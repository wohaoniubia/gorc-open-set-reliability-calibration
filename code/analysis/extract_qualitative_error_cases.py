from __future__ import annotations


import math

import sys

from pathlib import Path


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

    normalize_image_id,

    safe_class_name,

)



BUNDLE = ROOT / "GORC_paper_release_bundle"

ROUND3 = ROOT / "outputs" / "gorc_major_revision_round3"

OUT = ROUND3 / "01_error_rows"


PROTO = BUNDLE / "data_outputs" / "coco_main" / "step12a_protocol"

SCORED = BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_geometry_scored_candidates.csv"

COMPACT = BUNDLE / "data_outputs" / "coco_main" / "step12c_gorc" / "csv" / "step12c_compact_paper_metrics.csv"


POLICIES = [

    {

        "policy_name": "COCO raw",

        "method": "step12b_yoloworld_l_raw_reproduced",

        "score_col": "raw_score",

        "threshold": 0.30,

    },

    {

        "policy_name": "COCO GORC RF",

        "method": "step12c_geometry_max_cal_balanced",

        "score_col": "geometry_alpha_0p800",

        "threshold": 0.70,

    },

    {

        "policy_name": "COCO GORC AP-C",

        "method": "step12c_geometry_ap_constrained_balanced_AP50tol0_APtol0",

        "score_col": "class_reliability_gamma_0p250",

        "threshold": 0.30,

    },

]



def finite(value, default: float = 0.0) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def gt_key(row: pd.Series) -> str:

    return f"{normalize_image_id(row.image_id)}::{row.gt_id}"



def row_key_tuple(row: pd.Series) -> tuple[str, object]:

    return (normalize_image_id(row.image_id), row.gt_id)



def evaluate_rows_for_policy(

    scored: pd.DataFrame,

    gt_test: pd.DataFrame,

    *,

    policy_name: str,

    score_col: str,

    threshold: float,

    iou_thr: float = 0.50,

) -> tuple[pd.DataFrame, dict]:

    sub = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    sub["_policy_score"] = pd.to_numeric(sub[score_col], errors="coerce").fillna(-math.inf)

    dets = sub[sub["_policy_score"] >= float(threshold)].copy()

    dets = dets.sort_values("_policy_score", ascending=False).reset_index(drop=True)


    gt_by_image = {img: g.copy() for img, g in gt_test.groupby("image_id", sort=False)}

    matched_known: set[tuple[str, object]] = set()

    accepted_unknown: set[tuple[str, object]] = set()

    rows: list[dict] = []

    tp = 0

    fp = 0

    bgfp = 0


    for det_index, pr in dets.iterrows():

        image_id = normalize_image_id(pr.image_id)

        gimg = gt_by_image.get(image_id, gt_test.iloc[0:0])

        pbox = (finite(pr.x1), finite(pr.y1), finite(pr.x2), finite(pr.y2))

        pred_label = safe_class_name(pr.pred_label)







        best_known_iou = 0.0

        best_known_key: tuple[str, object] | None = None

        best_known_id = ""

        best_known_label = ""

        max_any_known_iou = 0.0

        max_any_known_id = ""

        max_any_known_label = ""

        for _, gr in gimg[gimg["gt_is_known"]].iterrows():

            key = row_key_tuple(gr)

            iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

            if iou > max_any_known_iou:

                max_any_known_iou = float(iou)

                max_any_known_id = gt_key(gr)

                max_any_known_label = safe_class_name(gr.gt_label)

            if key in matched_known:

                continue

            if iou > best_known_iou:

                best_known_iou = float(iou)

                best_known_key = key

                best_known_id = gt_key(gr)

                best_known_label = safe_class_name(gr.gt_label)


        known_tp = bool(best_known_key is not None and best_known_iou >= iou_thr and pred_label == best_known_label)

        matched_known_id = ""

        matched_unknown_id = ""

        best_unknown_iou = 0.0

        best_unknown_label = ""

        final_category = ""

        contributes_ufa = False

        contributes_bgfp = False


        if known_tp:

            tp += 1

            matched_known.add(best_known_key)                          

            matched_known_id = best_known_id

            final_category = "known_tp"

        else:

            fp += 1

            best_unknown_key: tuple[str, object] | None = None

            for _, gr in gimg[~gimg["gt_is_known"]].iterrows():

                iou = bbox_iou(pbox, (gr.x1, gr.y1, gr.x2, gr.y2))

                if iou > best_unknown_iou:

                    best_unknown_iou = float(iou)

                    best_unknown_key = row_key_tuple(gr)

                    matched_unknown_id = gt_key(gr)

                    best_unknown_label = safe_class_name(gr.gt_label)


            if best_unknown_key is not None and best_unknown_iou >= iou_thr:

                accepted_unknown.add(best_unknown_key)

                contributes_ufa = True

                final_category = "unknown_false_accept"

            elif best_known_iou >= iou_thr:

                matched_known_id = best_known_id

                final_category = "wrong_known_class_or_duplicate"

            else:

                bgfp += 1

                contributes_bgfp = True

                final_category = "background_false_accept"


        rows.append(

            {

                "protocol": "COCO-Val-OpenSet-5K",

                "detector": "YOLO-World-l",

                "policy_name": policy_name,

                "image_id": image_id,

                "detection_id": str(pr.get("det_id", f"{image_id}_{policy_name}_{det_index:06d}")),

                "pred_class": pred_label,

                "score": finite(pr["_policy_score"]),

                "accepted": True,

                "known_tp": known_tp,

                "matched_known_id": matched_known_id,

                "matched_unknown_id": matched_unknown_id if contributes_ufa else "",

                "max_known_iou": float(best_known_iou),

                "max_unknown_iou": float(best_unknown_iou),

                "contributes_to_UFA_object": contributes_ufa,

                "contributes_to_BGFP": contributes_bgfp,

                "final_error_category": final_category,

                "threshold": float(threshold),

                "score_col": score_col,

                "rank_within_policy": int(det_index + 1),

                "x1": finite(pr.x1),

                "y1": finite(pr.y1),

                "x2": finite(pr.x2),

                "y2": finite(pr.y2),

                "evaluator_iou_threshold": float(iou_thr),

                "max_any_known_iou": float(max_any_known_iou),

                "max_any_known_id": max_any_known_id,

                "max_any_known_label": max_any_known_label,

                "evaluator_known_overlap_note": "max_known_iou is over currently unmatched known GT, matching compact evaluator one-to-one state",

                "best_known_label": best_known_label,

                "best_unknown_label": best_unknown_label,

            }

        )


    num_known = int(gt_test["gt_is_known"].sum())

    num_unknown = int((~gt_test["gt_is_known"]).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0

    recall = tp / num_known if num_known else 0.0

    unknown_false = len(accepted_unknown)

    urr = 1.0 - unknown_false / num_unknown if num_unknown else 1.0

    b = 0.0 if min(precision, recall, urr) <= 0 else 3.0 / (1.0 / precision + 1.0 / recall + 1.0 / urr)

    metrics = {

        "policy_name": policy_name,

        "score_col": score_col,

        "threshold": float(threshold),

        "accepted": int(len(dets)),

        "tp_known": int(tp),

        "fp_known": int(fp),

        "num_known_gt": num_known,

        "num_unknown_gt": num_unknown,

        "known_precision": float(precision),

        "known_recall": float(recall),

        "unknown_false_accept_objects": int(unknown_false),

        "unknown_reject_rate_object_level": float(urr),

        "balanced": float(b),

        "background_false_accept_count": int(bgfp),

    }

    return pd.DataFrame(rows), metrics



def main() -> int:

    OUT.mkdir(parents=True, exist_ok=True)

    split = load_split(PROTO / "csv" / "step12a_scene_split.csv", ROOT, split_filter="all")

    known = load_known_classes(PROTO / "csv" / "step12a_class_map.csv")

    gt = load_annotations(PROTO / "annotations", split, known)

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    scored = clean_columns(pd.read_csv(SCORED, encoding="utf-8-sig"))

    compact = clean_columns(pd.read_csv(COMPACT, encoding="utf-8-sig"))


    all_rows: list[pd.DataFrame] = []

    recon_rows: list[dict] = []

    for spec in POLICIES:

        rows, metrics = evaluate_rows_for_policy(

            scored,

            gt_test,

            policy_name=spec["policy_name"],

            score_col=spec["score_col"],

            threshold=float(spec["threshold"]),

        )

        all_rows.append(rows)

        compact_row = compact[compact["method"].astype(str).eq(spec["method"])]

        if compact_row.empty:

            raise ValueError(f"Compact method not found: {spec['method']}")

        cr = compact_row.iloc[0]

        csv_bgfp = int(rows["contributes_to_BGFP"].sum())

        csv_ufa = int(rows.loc[rows["contributes_to_UFA_object"], "matched_unknown_id"].nunique())

        compact_bgfp = int(round(finite(cr.get("background_false_accept_count"))))

        compact_ufa = int(round(finite(cr.get("unknown_false_accept_objects"))))

        recon_rows.append(

            {

                "policy_name": spec["policy_name"],

                "method": spec["method"],

                "score_col": spec["score_col"],

                "threshold": float(spec["threshold"]),

                "csv_rows": int(len(rows)),

                "csv_BGFP_sum": csv_bgfp,

                "compact_BGFP": compact_bgfp,

                "BGFP_reconciles": csv_bgfp == compact_bgfp,

                "csv_UFA_nunique": csv_ufa,

                "compact_UFA": compact_ufa,

                "UFA_reconciles": csv_ufa == compact_ufa,

                "accepted": metrics["accepted"],

                "compact_accepted": int(round(finite(cr.get("num_accepted_detections")))),

                "accepted_reconciles": metrics["accepted"] == int(round(finite(cr.get("num_accepted_detections")))),

                "B_recomputed": metrics["balanced"],

                "compact_B": finite(cr.get("balanced")),

                "B_abs_delta": abs(metrics["balanced"] - finite(cr.get("balanced"))),

                "known_precision_recomputed": metrics["known_precision"],

                "known_recall_recomputed": metrics["known_recall"],

                "notes": "Rows are emitted by replaying the compact evaluator one-to-one matching order; no candidate-table subcategory metadata is used.",

            }

        )


    out_rows = pd.concat(all_rows, ignore_index=True)

    out_rows.to_csv(OUT / "coco_per_accepted_detection_error_rows.csv", index=False, encoding="utf-8-sig")

    recon = pd.DataFrame(recon_rows)

    pass_all = bool(recon["BGFP_reconciles"].all() and recon["UFA_reconciles"].all() and recon["accepted_reconciles"].all() and (recon["B_abs_delta"] < 1e-9).all())

    recon.to_csv(OUT / "coco_error_rows_reconciliation.csv", index=False, encoding="utf-8-sig")


    md = [

        "# COCO Evaluator-Level Accepted-Detection Error Rows Reconciliation",

        "",

        f"Status: {'PASS' if pass_all else 'FAIL'}",

        "",

        "This round3 audit does not use candidate-table `risk_error_type` metadata to infer public taxonomy. It replays the compact evaluator's one-to-one matching order for the final accepted detections under each selected policy.",

        "",

        "Important convention: `max_known_iou` in the CSV is the evaluator-level maximum over currently unmatched known ground-truth objects at the point where a detection is evaluated. This is the value used by the compact evaluator after earlier known TPs have consumed matched known objects. For auditability, the CSV also includes `max_any_known_iou`, which is not used for compact-count reconciliation.",

        "",

        "## Reconciliation Table",

        "",

        recon.to_markdown(index=False),

        "",

        "## Public-Use Decision",

        "",

    ]

    if pass_all:

        md.extend(

            [

                "The evaluator-level rows reconcile exactly with compact evaluator UFA and BG FP counts for the audited COCO raw, RF, and AP-C policies.",

                "",

                "Use decision: the row-level CSV can support an optional supplementary diagnostic table. It should not be promoted to the main manuscript.",

            ]

        )

    else:

        md.extend(

            [

                "The evaluator-level rows do not reconcile with compact evaluator counts. Do not publish error decomposition. Keep the previous blocker active and update it with this failed reconciliation.",

            ]

        )

    (OUT / "coco_error_rows_reconciliation.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    return 0 if pass_all else 2



if __name__ == "__main__":

    raise SystemExit(main())

