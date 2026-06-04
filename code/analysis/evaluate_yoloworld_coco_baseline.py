


"""
Step12B: YOLO-World raw baseline on COCO-Val-OpenSet-5K.

The script is chunked/resumable by image and uses calibration split only for
threshold selection. It detects only the known classes defined by Step12A.
"""


from __future__ import annotations


import argparse

import json

from datetime import datetime

from pathlib import Path


import pandas as pd


from step8j_yoloworld_lvis_openvoc_baseline import (

    DEFAULT_CONF_SWEEP,

    clean_columns,

    compute_ap_summary,

    evaluate_threshold_sweep,

    load_annotations,

    load_known_classes,

    load_split,

    nms_detections,

    normalize_image_id,

    safe_class_name,

)

from step9b_yoloworld_lvis300_raw_baseline import (

    PRED_COLUMNS,

    PROCESSED_COLUMNS,

    read_or_empty,

    resolve_model,

    run_resumable_yoloworld,

)



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step12a_coco_val_openset_protocol"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step12b_yoloworld_coco_openset_baseline"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def write_summary(path: Path, report: dict) -> None:

    lines = [

        "# Step12B YOLO-World COCO-Val-OpenSet-5K Baseline Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Model requested: `{report['model_requested']}`",

        f"- Model used: `{report['model_used']}`",

        f"- Prompt mode: `{report['prompt_mode']}`",

        f"- Intended images: `{report['num_images_intended']}`",

        f"- Processed images: `{report['num_images_processed']}`",

        f"- Inference complete: `{report['inference_complete']}`",

        f"- Evaluation complete: `{report['evaluation_complete']}`",

        f"- Known classes: `{len(report['known_classes'])}`",

    ]

    if report.get("selected_conf_thr") is not None:

        lines.append(f"- Selected calibration threshold: `{report['selected_conf_thr']}`")

    if report.get("compact_metrics"):

        lines.extend(["", "## Compact Paper Metrics", ""])

        lines.append(pd.DataFrame(report["compact_metrics"]).to_markdown(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def evaluate_complete(

    *,

    pred: pd.DataFrame,

    gt: pd.DataFrame,

    known_classes: list[str],

    csv_dir: Path,

    method_name: str,

    conf_sweep: list[float],

    nms_iou: float,

    nms_contain: float,

    nms_mode: str,

    selection_iou: float,

) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:

    for c in ["score", "x1", "y1", "x2", "y2"]:

        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    pred = pred.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)

    pred["image_id"] = pred["image_id"].map(normalize_image_id)

    pred["pred_label"] = pred["pred_label"].map(safe_class_name)


    test_pred = pred[pred["split"].astype(str).str.lower().eq("test")].copy()

    test_pred_nms = nms_detections(test_pred, iou_thr=nms_iou, contain_thr=nms_contain, mode=nms_mode)

    test_gt = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    ap_summary, per_class_ap = compute_ap_summary(test_pred_nms, test_gt, known_classes)

    ap_summary.insert(0, "method", method_name)

    ap_summary.to_csv(csv_dir / "step12b_ap_summary.csv", index=False, encoding="utf-8-sig")

    per_class_ap.to_csv(csv_dir / "step12b_per_class_ap.csv", index=False, encoding="utf-8-sig")


    cal_eval, _ = evaluate_threshold_sweep(pred, gt, "calibration", conf_sweep, nms_iou, nms_contain, nms_mode, [0.50, 0.75])

    best = cal_eval[cal_eval["iou_threshold"].astype(float).round(4).eq(float(selection_iou))].copy()

    if best.empty:

        raise ValueError(f"No calibration rows at IoU={selection_iou}")

    best = best.sort_values(

        [

            "precision_recall_unknown_balanced_score",

            "known_precision",

            "known_recall",

            "unknown_reject_rate_object_level",

        ],

        ascending=[False, False, False, False],

    ).iloc[0].to_dict()

    test_eval, test_errors = evaluate_threshold_sweep(

        pred, gt, "test", [float(best["conf_thr"])], nms_iou, nms_contain, nms_mode, [0.50, 0.75]

    )

    test_eval.insert(0, "method", method_name)

    compact = test_eval[test_eval["iou_threshold"].astype(float).round(4).eq(0.50)].copy()

    for c in ap_summary.columns:

        compact[c] = ap_summary.iloc[0][c]


    cal_eval.to_csv(csv_dir / "step12b_calibration_sweep.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([best]).to_csv(csv_dir / "step12b_selected_calibration_config.csv", index=False, encoding="utf-8-sig")

    test_eval.to_csv(csv_dir / "step12b_primary_test_results.csv", index=False, encoding="utf-8-sig")

    test_errors.to_csv(csv_dir / "step12b_test_error_cases.csv", index=False, encoding="utf-8-sig")

    compact.to_csv(csv_dir / "step12b_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")

    return compact, best, ap_summary, test_pred_nms



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step12a_scene_split.csv")

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_PROTOCOL_ROOT / "annotations")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step12a_class_map.csv")

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--model", type=str, default="yolov8l-worldv2.pt")

    parser.add_argument("--fallback_model", type=str, default="yolov8m-worldv2.pt")

    parser.add_argument("--prompt_mode", type=str, choices=["canonical", "synonyms"], default="canonical")

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--imgsz", type=int, default=960)

    parser.add_argument("--min_conf", type=float, default=0.001)

    parser.add_argument("--yolo_iou", type=float, default=0.70)

    parser.add_argument("--nms_iou", type=float, default=0.50)

    parser.add_argument("--nms_contain", type=float, default=0.90)

    parser.add_argument("--nms_mode", type=str, choices=["class", "global"], default="class")

    parser.add_argument("--selection_iou", type=float, default=0.50)

    parser.add_argument("--conf_sweep", type=str, default=",".join(str(x) for x in DEFAULT_CONF_SWEEP + [0.75, 0.85, 0.95]))

    parser.add_argument("--max_images", type=int, default=0)

    parser.add_argument("--max_images_per_run", type=int, default=0)

    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--allow_partial_eval", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    pred_csv = csv_dir / "step12b_yoloworld_raw_predictions.csv"

    processed_csv = csv_dir / "step12b_processed_images.csv"


    model_used = resolve_model(args.model, args.project_root)

    if not Path(model_used).exists() and Path(args.model).suffix == ".pt":

        fallback = resolve_model(args.fallback_model, args.project_root)

        if Path(fallback).exists():

            model_used = fallback


    print("\n========== Step12B YOLO-World COCO-Val-OpenSet-5K Raw Baseline ==========")

    print(f"split_csv={args.split_csv}")

    print(f"model_used={model_used}")


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root, split_filter="all")

    if args.max_images and args.max_images > 0:

        split_df = split_df.head(int(args.max_images)).copy()

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    pred, processed = run_resumable_yoloworld(

        split_df=split_df,

        known_classes=known_classes,

        pred_csv=pred_csv,

        processed_csv=processed_csv,

        project_root=args.project_root,

        model_name=model_used,

        prompt_mode=args.prompt_mode,

        device=args.device,

        imgsz=int(args.imgsz),

        min_conf=float(args.min_conf),

        yolo_iou=float(args.yolo_iou),

        max_images_per_run=int(args.max_images_per_run),

        eval_only=bool(args.eval_only),

    )


    processed_good = processed[processed["status"].astype(str).isin(["ok", "empty"])].copy()

    intended_ids = set(split_df["image_id"].map(normalize_image_id).tolist())

    processed_ids = set(processed_good["image_id"].map(normalize_image_id).tolist())

    inference_complete = intended_ids.issubset(processed_ids)

    report = {

        "method": "Step12B YOLO-World raw baseline on COCO-Val-OpenSet-5K",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "split_csv": str(args.split_csv),

        "annotation_dir": str(args.annotation_dir),

        "class_map_csv": str(args.class_map_csv),

        "output_root": str(args.output_root),

        "model_requested": args.model,

        "fallback_model": args.fallback_model,

        "model_used": model_used,

        "prompt_mode": args.prompt_mode,

        "known_classes": known_classes,

        "num_images_intended": int(len(split_df)),

        "num_images_processed": int(len(processed_ids)),

        "num_gt": int(len(gt)),

        "num_predictions_raw": int(len(pred)),

        "inference_complete": bool(inference_complete),

        "evaluation_complete": False,

        "selected_conf_thr": None,

        "calibration_only_threshold_selection": True,

        "test_split_final_evaluation_only": True,

        "outputs": {

            "raw_predictions": str(pred_csv),

            "processed_images": str(processed_csv),

        },

    }

    if not inference_complete and not args.allow_partial_eval:

        with open(args.output_root / "step12b_integrity_report.json", "w", encoding="utf-8") as f:

            json.dump(report, f, ensure_ascii=False, indent=2)

        write_summary(args.output_root / "step12b_summary.md", report)

        print(f"Chunk completed: {len(processed_ids)}/{len(split_df)} images processed. Run again to resume.")

        return


    if pred.empty:

        raise ValueError("YOLO-World produced no predictions.")

    conf_sweep = [float(x.strip()) for x in str(args.conf_sweep).split(",") if x.strip()]

    method_name = f"step12b_yoloworld_{Path(model_used).stem}_{args.prompt_mode}"

    compact, best, ap_summary, test_pred_nms = evaluate_complete(

        pred=pred.copy(),

        gt=gt,

        known_classes=known_classes,

        csv_dir=csv_dir,

        method_name=method_name,

        conf_sweep=conf_sweep,

        nms_iou=float(args.nms_iou),

        nms_contain=float(args.nms_contain),

        nms_mode=args.nms_mode,

        selection_iou=float(args.selection_iou),

    )

    clean_columns(pred).to_csv(pred_csv, index=False, encoding="utf-8-sig")

    report.update(

        {

            "num_predictions_raw": int(len(pred)),

            "num_predictions_test_nms": int(len(test_pred_nms)),

            "selected_conf_thr": float(best["conf_thr"]),

            "selection_iou": float(args.selection_iou),

            "nms_iou": float(args.nms_iou),

            "nms_contain": float(args.nms_contain),

            "nms_mode": args.nms_mode,

            "evaluation_complete": True,

            "compact_metrics": compact.to_dict("records"),

            "outputs": {

                **report["outputs"],

                "selected_config": str(csv_dir / "step12b_selected_calibration_config.csv"),

                "ap_summary": str(csv_dir / "step12b_ap_summary.csv"),

                "compact_paper_metrics": str(csv_dir / "step12b_compact_paper_metrics.csv"),

                "primary_test_results": str(csv_dir / "step12b_primary_test_results.csv"),

                "per_class_ap": str(csv_dir / "step12b_per_class_ap.csv"),

                "test_errors": str(csv_dir / "step12b_test_error_cases.csv"),

                "summary": str(args.output_root / "step12b_summary.md"),

                "integrity_report": str(args.output_root / "step12b_integrity_report.json"),

            },

        }

    )

    with open(args.output_root / "step12b_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)

    write_summary(args.output_root / "step12b_summary.md", report)


    print("\n========== Step12B completed ==========")

    print(compact.to_string(index=False))



if __name__ == "__main__":

    main()

