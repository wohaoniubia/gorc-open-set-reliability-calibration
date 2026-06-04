


"""
Step9b: Resumable YOLO-World raw baseline on the larger LVIS-Clear protocol.

This script is intentionally chunkable. It records processed images separately
from detections, so interrupted or time-boxed inference can resume safely.
"""


from __future__ import annotations


import argparse

import json

import time

from pathlib import Path

from typing import List


import pandas as pd

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import (

    DEFAULT_CONF_SWEEP,

    build_prompts,

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



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_SPLIT_CSV = DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv"

DEFAULT_ANN_DIR = DEFAULT_PROTOCOL_ROOT / "annotations"

DEFAULT_CLASS_MAP_CSV = DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step9b_yoloworld_lvis300_raw_baseline"


PRED_COLUMNS = ["image_id", "split", "image_path", "det_id", "prompt", "pred_label", "score", "x1", "y1", "x2", "y2"]

PROCESSED_COLUMNS = ["image_id", "split", "image_path", "status", "num_detections", "error", "seconds"]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def resolve_model(model_name: str, project_root: Path) -> str:

    p = Path(model_name)

    if p.is_absolute() or p.exists():

        return str(p)

    local = project_root / model_name

    return str(local) if local.exists() else model_name



def read_or_empty(path: Path, columns: List[str]) -> pd.DataFrame:

    if path.exists() and path.stat().st_size > 0:

        return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))

    return pd.DataFrame(columns=columns)



def write_progress_summary(path: Path, report: dict) -> None:

    lines = [

        "# Step9b YOLO-World Baseline Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Model: `{report['model']}`",

        f"- Prompt mode: `{report['prompt_mode']}`",

        f"- Intended images: `{report['num_images_intended']}`",

        f"- Processed images: `{report['num_images_processed']}`",

        f"- Completed inference: `{report['inference_complete']}`",

        f"- Evaluation complete: `{report['evaluation_complete']}`",

    ]

    if report.get("selected_conf_thr") is not None:

        lines.append(f"- Selected calibration threshold: `{report['selected_conf_thr']}`")

    if report.get("compact_metrics"):

        lines.extend(["", "## Compact IoU0.50 Metrics", ""])

        df = pd.DataFrame(report["compact_metrics"])

        lines.append(df.to_markdown(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def run_resumable_yoloworld(

    *,

    split_df: pd.DataFrame,

    known_classes: List[str],

    pred_csv: Path,

    processed_csv: Path,

    project_root: Path,

    model_name: str,

    prompt_mode: str,

    device: str,

    imgsz: int,

    min_conf: float,

    yolo_iou: float,

    max_images_per_run: int,

    eval_only: bool,

) -> tuple[pd.DataFrame, pd.DataFrame]:

    pred = read_or_empty(pred_csv, PRED_COLUMNS)

    processed = read_or_empty(processed_csv, PROCESSED_COLUMNS)

    processed_ids = set(processed[processed["status"].astype(str).eq("ok")]["image_id"].map(normalize_image_id).tolist())

    processed_ids.update(processed[processed["status"].astype(str).eq("empty")]["image_id"].map(normalize_image_id).tolist())


    if eval_only:

        return pred, processed


    work = split_df.copy().reset_index(drop=True)

    todo = [r for _, r in work.iterrows() if normalize_image_id(r["image_id"]) not in processed_ids]

    if max_images_per_run and max_images_per_run > 0:

        todo = todo[: int(max_images_per_run)]

    if not todo:

        return pred, processed


    try:

        from ultralytics import YOLO

    except Exception as exc:

        raise ImportError("Missing ultralytics. Install with: pip install ultralytics") from exc


    prompts, prompt_to_class = build_prompts(known_classes, prompt_mode)

    print(f"YOLO-World model: {model_name}")

    print(f"Prompt mode: {prompt_mode}; prompts={prompts}")

    model = YOLO(model_name)

    model.set_classes(prompts)


    pred_rows = pred.to_dict("records") if len(pred) else []

    processed_rows = processed.to_dict("records") if len(processed) else []

    for r in tqdm(todo, total=len(todo), desc="Step9b YOLO-World chunk"):

        start = time.time()

        image_id = normalize_image_id(r["image_id"])

        image_path = Path(str(r["image_path_resolved"]))

        split = str(r.get("split", ""))

        rows_before = len(pred_rows)

        status = "ok"

        error = ""

        if not image_path.exists():

            status = "missing"

            error = f"missing image: {image_path}"

        else:

            try:

                results = model.predict(

                    source=str(image_path),

                    imgsz=int(imgsz),

                    conf=float(min_conf),

                    iou=float(yolo_iou),

                    device=None if device.lower() == "auto" else device,

                    verbose=False,

                )

                if results and results[0].boxes is not None and len(results[0].boxes) > 0:

                    res = results[0]

                    xyxy = res.boxes.xyxy.detach().cpu().numpy()

                    confs = res.boxes.conf.detach().cpu().numpy()

                    clss = res.boxes.cls.detach().cpu().numpy().astype(int)

                    for j, (box, conf, ci) in enumerate(zip(xyxy, confs, clss)):

                        if ci < 0 or ci >= len(prompts):

                            continue

                        prompt = prompts[int(ci)]

                        canonical = prompt_to_class.get(prompt)

                        if canonical is None:

                            continue

                        x1, y1, x2, y2 = [float(v) for v in box.tolist()]

                        if x2 <= x1 or y2 <= y1:

                            continue

                        pred_rows.append(

                            {

                                "image_id": image_id,

                                "split": split,

                                "image_path": str(image_path),

                                "det_id": f"{image_id}_yw_{len(pred_rows):06d}",

                                "prompt": prompt,

                                "pred_label": canonical,

                                "score": float(conf),

                                "x1": x1,

                                "y1": y1,

                                "x2": x2,

                                "y2": y2,

                            }

                        )

                if len(pred_rows) == rows_before:

                    status = "empty"

            except Exception as exc:                  

                status = "error"

                error = repr(exc)

                print(f"Warning: YOLO failed for {image_id}: {error}")

        processed_rows.append(

            {

                "image_id": image_id,

                "split": split,

                "image_path": str(image_path),

                "status": status,

                "num_detections": int(len(pred_rows) - rows_before),

                "error": error,

                "seconds": float(time.time() - start),

            }

        )

        pd.DataFrame(pred_rows, columns=PRED_COLUMNS).to_csv(pred_csv, index=False, encoding="utf-8-sig")

        pd.DataFrame(processed_rows, columns=PROCESSED_COLUMNS).to_csv(processed_csv, index=False, encoding="utf-8-sig")


    return pd.DataFrame(pred_rows, columns=PRED_COLUMNS), pd.DataFrame(processed_rows, columns=PROCESSED_COLUMNS)



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_SPLIT_CSV)

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_ANN_DIR)

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_CLASS_MAP_CSV)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--model", type=str, default="yolov8l-worldv2.pt")

    parser.add_argument("--prompt_mode", type=str, choices=["canonical", "synonyms"], default="synonyms")

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--imgsz", type=int, default=960)

    parser.add_argument("--min_conf", type=float, default=0.001)

    parser.add_argument("--yolo_iou", type=float, default=0.70)

    parser.add_argument("--nms_iou", type=float, default=0.50)

    parser.add_argument("--nms_contain", type=float, default=0.90)

    parser.add_argument("--nms_mode", type=str, choices=["class", "global"], default="class")

    parser.add_argument("--selection_iou", type=float, default=0.50)

    parser.add_argument("--conf_sweep", type=str, default=",".join(str(x) for x in DEFAULT_CONF_SWEEP))

    parser.add_argument("--max_images", type=int, default=0, help="Debug limit over split rows. 0 means all.")

    parser.add_argument("--max_images_per_run", type=int, default=0, help="Chunk size for resumable inference. 0 means process all remaining.")

    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--allow_partial_eval", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    pred_csv = csv_dir / "step9b_yoloworld_raw_predictions.csv"

    processed_csv = csv_dir / "step9b_processed_images.csv"

    model_name = resolve_model(args.model, args.project_root)


    print("\n========== Step9b YOLO-World LVIS-300 Raw Baseline ==========")

    print(f"split_csv={args.split_csv}")

    print(f"model={model_name}")


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

        model_name=model_name,

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

        "method": "Step9b YOLO-World raw baseline on LVIS-Clear-Mini-300",

        "project_root": str(args.project_root),

        "split_csv": str(args.split_csv),

        "annotation_dir": str(args.annotation_dir),

        "class_map_csv": str(args.class_map_csv),

        "output_root": str(args.output_root),

        "model": model_name,

        "prompt_mode": args.prompt_mode,

        "known_classes": known_classes,

        "num_images_intended": int(len(split_df)),

        "num_images_processed": int(len(processed_ids)),

        "num_gt": int(len(gt)),

        "num_predictions_raw": int(len(pred)),

        "inference_complete": bool(inference_complete),

        "evaluation_complete": False,

        "selected_conf_thr": None,

        "outputs": {

            "raw_predictions": str(pred_csv),

            "processed_images": str(processed_csv),

        },

    }


    if not inference_complete and not args.allow_partial_eval:

        with open(args.output_root / "step9b_integrity_report.json", "w", encoding="utf-8") as f:

            json.dump(report, f, ensure_ascii=False, indent=2)

        write_progress_summary(args.output_root / "step9b_summary.md", report)

        print(f"Chunk completed: {len(processed_ids)}/{len(split_df)} images processed. Run again with --max_images_per_run to resume.")

        return


    if pred.empty:

        raise ValueError("YOLO-World produced no predictions.")

    for c in ["score", "x1", "y1", "x2", "y2"]:

        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    pred = pred.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)

    pred["image_id"] = pred["image_id"].map(normalize_image_id)

    pred["pred_label"] = pred["pred_label"].map(safe_class_name)

    pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")


    test_pred = pred[pred["split"].astype(str).str.lower() == "test"].copy()

    test_pred_nms = nms_detections(test_pred, iou_thr=args.nms_iou, contain_thr=args.nms_contain, mode=args.nms_mode)

    test_gt = gt[gt["split"].astype(str).str.lower() == "test"].copy()

    ap_summary, per_class_ap = compute_ap_summary(test_pred_nms, test_gt, known_classes)

    method_name = f"step9b_yoloworld_{Path(model_name).stem}_{args.prompt_mode}"

    ap_summary.insert(0, "method", method_name)

    ap_summary.to_csv(csv_dir / "step9b_ap_summary.csv", index=False, encoding="utf-8-sig")

    per_class_ap.to_csv(csv_dir / "step9b_per_class_ap.csv", index=False, encoding="utf-8-sig")


    conf_list = [float(x.strip()) for x in str(args.conf_sweep).split(",") if x.strip()]

    cal_eval, _ = evaluate_threshold_sweep(pred, gt, "calibration", conf_list, args.nms_iou, args.nms_contain, args.nms_mode, [0.50, 0.75])

    best = cal_eval[cal_eval["iou_threshold"].astype(float).round(4).eq(float(args.selection_iou))].copy()

    if best.empty:

        raise ValueError(f"No calibration rows at IoU={args.selection_iou}")

    best = best.sort_values(

        ["precision_recall_unknown_balanced_score", "known_recall", "unknown_reject_rate_object_level", "known_precision"],

        ascending=[False, False, False, False],

    ).iloc[0].to_dict()

    test_eval, test_errors = evaluate_threshold_sweep(pred, gt, "test", [float(best["conf_thr"])], args.nms_iou, args.nms_contain, args.nms_mode, [0.50, 0.75])

    cal_eval.to_csv(csv_dir / "step9b_calibration_sweep.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([best]).to_csv(csv_dir / "step9b_selected_calibration_config.csv", index=False, encoding="utf-8-sig")

    test_eval.insert(0, "method", method_name)

    test_eval.to_csv(csv_dir / "step9b_primary_test_results.csv", index=False, encoding="utf-8-sig")

    test_errors.to_csv(csv_dir / "step9b_test_error_cases.csv", index=False, encoding="utf-8-sig")


    compact = test_eval[test_eval["iou_threshold"].astype(float).round(4).eq(0.50)].copy()

    for c in ap_summary.columns:

        compact[c] = ap_summary.iloc[0][c]

    compact.to_csv(csv_dir / "step9b_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")


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

                "ap_summary": str(csv_dir / "step9b_ap_summary.csv"),

                "per_class_ap": str(csv_dir / "step9b_per_class_ap.csv"),

                "selected_config": str(csv_dir / "step9b_selected_calibration_config.csv"),

                "primary_test_results": str(csv_dir / "step9b_primary_test_results.csv"),

                "compact": str(csv_dir / "step9b_compact_paper_metrics.csv"),

                "test_errors": str(csv_dir / "step9b_test_error_cases.csv"),

                "markdown_summary": str(args.output_root / "step9b_summary.md"),

            },

        }

    )

    with open(args.output_root / "step9b_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)

    write_progress_summary(args.output_root / "step9b_summary.md", report)


    print("\n========== Step9b completed ==========")

    print("AP summary:")

    print(ap_summary.to_string(index=False))

    print("\nSelected calibration config:")

    print(pd.DataFrame([best]).to_string(index=False))

    print("\nCompact metrics:")

    print(compact.to_string(index=False))



if __name__ == "__main__":

    main()

