


"""
Step11A: Full same-protocol Grounding DINO baseline on LVIS-Clear-Mini-300.

The script is resumable by image. It writes predictions and a processed-image
log after every image, then performs calibration-only threshold selection and a
single held-out test evaluation once the intended image set is complete.
"""


from __future__ import annotations


import argparse

import json

import time

from datetime import datetime

from pathlib import Path

from typing import Dict, List, Sequence


import pandas as pd

from PIL import Image

from tqdm import tqdm


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



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step11a_groundingdino_lvis300_full_baseline"


PRED_COLUMNS = ["image_id", "split", "image_path", "det_id", "raw_label", "pred_label", "score", "x1", "y1", "x2", "y2"]

PROCESSED_COLUMNS = ["image_id", "split", "image_path", "status", "num_detections", "error", "seconds"]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_or_empty(path: Path, columns: Sequence[str]) -> pd.DataFrame:

    if path.exists() and path.stat().st_size > 0:

        return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))

    return pd.DataFrame(columns=list(columns))



def canonical_label(raw_label: str, known_classes: Sequence[str]) -> str:

    text = str(raw_label).strip().lower().replace("_", " ")

    known = [safe_class_name(c) for c in known_classes]

    text_safe = safe_class_name(text)

    if text_safe in known:

        return text_safe

    for cls in known:

        if cls.replace("_", " ") in text:

            return cls

    return ""



def build_prompt(known_classes: Sequence[str]) -> str:

    return ". ".join(c.replace("_", " ") for c in known_classes) + "."



def post_process_grounding_dino(processor, outputs, inputs: Dict, target_sizes, box_threshold: float, text_threshold: float):

    try:

        return processor.post_process_grounded_object_detection(

            outputs,

            inputs.get("input_ids"),

            box_threshold=float(box_threshold),

            text_threshold=float(text_threshold),

            target_sizes=target_sizes,

        )[0]

    except TypeError:

        try:

            return processor.post_process_grounded_object_detection(

                outputs,

                inputs.get("input_ids"),

                threshold=float(box_threshold),

                text_threshold=float(text_threshold),

                target_sizes=target_sizes,

            )[0]

        except TypeError:

            return processor.post_process_grounded_object_detection(

                outputs,

                threshold=float(box_threshold),

                text_threshold=float(text_threshold),

                target_sizes=target_sizes,

            )[0]



def run_resumable_groundingdino(

    *,

    split_df: pd.DataFrame,

    known_classes: Sequence[str],

    pred_csv: Path,

    processed_csv: Path,

    model_id: str,

    device: str,

    box_threshold: float,

    text_threshold: float,

    max_images_per_run: int,

    eval_only: bool,

) -> tuple[pd.DataFrame, pd.DataFrame, dict]:

    pred = read_or_empty(pred_csv, PRED_COLUMNS)

    processed = read_or_empty(processed_csv, PROCESSED_COLUMNS)

    processed_ids = set(

        processed[processed["status"].astype(str).isin(["ok", "empty"])]["image_id"].map(normalize_image_id).tolist()

    )

    if eval_only:

        return pred, processed, {"model_loaded": False, "prompt": build_prompt(known_classes)}


    todo = [r for _, r in split_df.iterrows() if normalize_image_id(r["image_id"]) not in processed_ids]

    if max_images_per_run and max_images_per_run > 0:

        todo = todo[: int(max_images_per_run)]

    if not todo:

        return pred, processed, {"model_loaded": False, "prompt": build_prompt(known_classes)}


    import torch

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


    actual_device = device

    if actual_device.lower() == "cuda" and not torch.cuda.is_available():

        actual_device = "cpu"

    prompt = build_prompt(known_classes)

    print(f"Grounding DINO model: {model_id}")

    print(f"Device: {actual_device}")

    print(f"Prompt: {prompt}")

    processor = AutoProcessor.from_pretrained(model_id)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(actual_device)

    model.eval()


    pred_rows = pred.to_dict("records") if len(pred) else []

    processed_rows = processed.to_dict("records") if len(processed) else []

    with torch.no_grad():

        for r in tqdm(todo, total=len(todo), desc="Step11A GroundingDINO chunk"):

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

                    image = Image.open(image_path).convert("RGB")

                    inputs = processor(images=image, text=prompt, return_tensors="pt")

                    inputs = {k: v.to(actual_device) if hasattr(v, "to") else v for k, v in inputs.items()}

                    outputs = model(**inputs)

                    result = post_process_grounding_dino(

                        processor,

                        outputs,

                        inputs,

                        [(image.height, image.width)],

                        box_threshold=float(box_threshold),

                        text_threshold=float(text_threshold),

                    )

                    boxes = result.get("boxes", [])

                    scores = result.get("scores", [])

                    labels = result.get("text_labels", result.get("labels", []))

                    for j, box in enumerate(boxes):

                        score = float(scores[j].detach().cpu().item()) if hasattr(scores[j], "detach") else float(scores[j])

                        vals = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)

                        raw_label = str(labels[j]) if j < len(labels) else ""

                        pred_label = canonical_label(raw_label, known_classes)

                        if not pred_label:

                            continue

                        x1, y1, x2, y2 = [float(v) for v in vals[:4]]

                        if x2 <= x1 or y2 <= y1:

                            continue

                        pred_rows.append(

                            {

                                "image_id": image_id,

                                "split": split,

                                "image_path": str(image_path),

                                "det_id": f"{image_id}_gdino_{len(pred_rows):06d}",

                                "raw_label": raw_label,

                                "pred_label": pred_label,

                                "score": score,

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

                    print(f"Warning: Grounding DINO failed for {image_id}: {error}")

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


    return pd.DataFrame(pred_rows, columns=PRED_COLUMNS), pd.DataFrame(processed_rows, columns=PROCESSED_COLUMNS), {

        "model_loaded": True,

        "prompt": prompt,

        "device": actual_device,

    }



def write_summary(path: Path, report: dict) -> None:

    lines = [

        "# Step11A Grounding DINO LVIS-Clear-Mini-300 Baseline Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Model: `{report['model_id']}`",

        f"- Prompt: `{report['prompt']}`",

        f"- Intended images: `{report['num_images_intended']}`",

        f"- Processed images: `{report['num_images_processed']}`",

        f"- Inference complete: `{report['inference_complete']}`",

        f"- Evaluation complete: `{report['evaluation_complete']}`",

    ]

    if report.get("selected_conf_thr") is not None:

        lines.append(f"- Selected calibration threshold: `{report['selected_conf_thr']}`")

    if report.get("compact_metrics"):

        lines.extend(["", "## Compact Paper Metrics", ""])

        lines.append(pd.DataFrame(report["compact_metrics"]).to_markdown(index=False))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv")

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_PROTOCOL_ROOT / "annotations")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv")

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny")

    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--box_threshold", type=float, default=0.05)

    parser.add_argument("--text_threshold", type=float, default=0.20)

    parser.add_argument("--nms_iou", type=float, default=0.50)

    parser.add_argument("--nms_contain", type=float, default=0.90)

    parser.add_argument("--nms_mode", type=str, choices=["class", "global"], default="class")

    parser.add_argument("--selection_iou", type=float, default=0.50)

    parser.add_argument("--conf_sweep", type=str, default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90")

    parser.add_argument("--max_images", type=int, default=0)

    parser.add_argument("--max_images_per_run", type=int, default=0)

    parser.add_argument("--eval_only", action="store_true")

    parser.add_argument("--allow_partial_eval", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    pred_csv = csv_dir / "step11a_raw_predictions.csv"

    processed_csv = csv_dir / "step11a_processed_images.csv"


    print("\n========== Step11A Grounding DINO LVIS-300 Baseline ==========")

    print(f"output_root={args.output_root}")

    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root, split_filter="all")

    if args.max_images and args.max_images > 0:

        split_df = split_df.head(int(args.max_images)).copy()

    gt = load_annotations(args.annotation_dir, split_df, known_classes)


    pred, processed, runtime = run_resumable_groundingdino(

        split_df=split_df,

        known_classes=known_classes,

        pred_csv=pred_csv,

        processed_csv=processed_csv,

        model_id=args.model_id,

        device=args.device,

        box_threshold=float(args.box_threshold),

        text_threshold=float(args.text_threshold),

        max_images_per_run=int(args.max_images_per_run),

        eval_only=bool(args.eval_only),

    )


    intended_ids = set(split_df["image_id"].map(normalize_image_id).tolist())

    good = processed[processed["status"].astype(str).isin(["ok", "empty"])].copy()

    processed_ids = set(good["image_id"].map(normalize_image_id).tolist())

    inference_complete = intended_ids.issubset(processed_ids)


    report = {

        "method": "Step11A full same-protocol Grounding DINO baseline",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "split_csv": str(args.split_csv),

        "annotation_dir": str(args.annotation_dir),

        "class_map_csv": str(args.class_map_csv),

        "output_root": str(args.output_root),

        "model_id": args.model_id,

        "device_requested": args.device,

        "device_used": runtime.get("device", args.device),

        "prompt": runtime.get("prompt", build_prompt(known_classes)),

        "known_classes": known_classes,

        "box_threshold": float(args.box_threshold),

        "text_threshold": float(args.text_threshold),

        "num_images_intended": int(len(split_df)),

        "num_images_processed": int(len(processed_ids)),

        "num_predictions_raw": int(len(pred)),

        "num_gt": int(len(gt)),

        "inference_complete": bool(inference_complete),

        "evaluation_complete": False,

        "selected_conf_thr": None,

        "outputs": {

            "raw_predictions": str(pred_csv),

            "processed_images": str(processed_csv),

            "summary": str(args.output_root / "step11a_summary.md"),

            "integrity_report": str(args.output_root / "step11a_integrity_report.json"),

        },

        "protocol": {

            "calibration_only_threshold_selection": True,

            "test_split_final_evaluation_only": True,

            "main_iou_threshold": float(args.selection_iou),

        },

    }


    if not inference_complete and not args.allow_partial_eval:

        (args.output_root / "step11a_integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        write_summary(args.output_root / "step11a_summary.md", report)

        print(f"Chunk completed: {len(processed_ids)}/{len(split_df)} images processed.")

        return


    if pred.empty:

        raise ValueError("Grounding DINO produced no usable predictions.")

    for c in ["score", "x1", "y1", "x2", "y2"]:

        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    pred = pred.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)

    pred["image_id"] = pred["image_id"].map(normalize_image_id)

    pred["pred_label"] = pred["pred_label"].map(safe_class_name)

    pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")


    method_name = "step11a_groundingdino_tiny_raw"

    test_pred = pred[pred["split"].astype(str).str.lower().eq("test")].copy()

    test_gt = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    test_pred_nms = nms_detections(test_pred, iou_thr=float(args.nms_iou), contain_thr=float(args.nms_contain), mode=args.nms_mode)

    ap_summary, per_class_ap = compute_ap_summary(test_pred_nms, test_gt, known_classes)

    ap_summary.insert(0, "method", method_name)

    ap_summary.to_csv(csv_dir / "step11a_ap_summary.csv", index=False, encoding="utf-8-sig")

    per_class_ap.to_csv(csv_dir / "step11a_per_class_ap.csv", index=False, encoding="utf-8-sig")


    conf_list = [float(x.strip()) for x in str(args.conf_sweep).split(",") if x.strip()]

    cal_eval, _ = evaluate_threshold_sweep(pred, gt, "calibration", conf_list, args.nms_iou, args.nms_contain, args.nms_mode, [0.50, 0.75])

    best_df = cal_eval[cal_eval["iou_threshold"].astype(float).round(4).eq(float(args.selection_iou))].copy()

    if best_df.empty:

        raise ValueError(f"No calibration rows at IoU={args.selection_iou}")

    best = best_df.sort_values(

        ["precision_recall_unknown_balanced_score", "known_recall", "unknown_reject_rate_object_level", "known_precision"],

        ascending=[False, False, False, False],

    ).iloc[0].to_dict()

    test_eval, test_errors = evaluate_threshold_sweep(pred, gt, "test", [float(best["conf_thr"])], args.nms_iou, args.nms_contain, args.nms_mode, [0.50, 0.75])

    cal_eval.to_csv(csv_dir / "step11a_calibration_sweep.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame([best]).to_csv(csv_dir / "step11a_selected_calibration_config.csv", index=False, encoding="utf-8-sig")

    test_eval.insert(0, "method", method_name)

    test_eval.to_csv(csv_dir / "step11a_primary_test_results.csv", index=False, encoding="utf-8-sig")

    test_errors.to_csv(csv_dir / "step11a_test_error_cases.csv", index=False, encoding="utf-8-sig")


    compact = test_eval[test_eval["iou_threshold"].astype(float).round(4).eq(0.50)].copy()

    for c in ap_summary.columns:

        compact[c] = ap_summary.iloc[0][c]

    compact.to_csv(csv_dir / "step11a_compact_paper_metrics.csv", index=False, encoding="utf-8-sig")


    report.update(

        {

            "nms_iou": float(args.nms_iou),

            "nms_contain": float(args.nms_contain),

            "nms_mode": args.nms_mode,

            "selected_conf_thr": float(best["conf_thr"]),

            "selection_iou": float(args.selection_iou),

            "num_predictions_raw": int(len(pred)),

            "num_predictions_test_nms": int(len(test_pred_nms)),

            "evaluation_complete": True,

            "compact_metrics": compact.to_dict("records"),

            "outputs": {

                **report["outputs"],

                "selected_config": str(csv_dir / "step11a_selected_calibration_config.csv"),

                "compact_paper_metrics": str(csv_dir / "step11a_compact_paper_metrics.csv"),

                "primary_test_results": str(csv_dir / "step11a_primary_test_results.csv"),

                "ap_summary": str(csv_dir / "step11a_ap_summary.csv"),

                "per_class_ap": str(csv_dir / "step11a_per_class_ap.csv"),

                "calibration_sweep": str(csv_dir / "step11a_calibration_sweep.csv"),

                "test_errors": str(csv_dir / "step11a_test_error_cases.csv"),

            },

        }

    )

    (args.output_root / "step11a_integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    write_summary(args.output_root / "step11a_summary.md", report)

    print("\n========== Step11A completed ==========")

    print(compact.to_string(index=False))



if __name__ == "__main__":

    main()

