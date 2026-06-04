


"""Round3 fixed-subset Grounding DINO prompt-sensitivity audit.

The detector-inference part is resumable by image and writes predictions plus a
processed-image log after every image. The canonical period prompt reuses the
existing full COCO Grounding DINO cache and filters it to the same fixed subset;
the comma and phrase prompts are run only for that subset.
"""


from __future__ import annotations


import argparse

import json

import math

import os

import sys

import time

from datetime import datetime

from pathlib import Path

from typing import Sequence


import numpy as np

import pandas as pd

from PIL import Image

from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "outputs" / "gorc_major_revision_round3" / "03_prompt_sensitivity"


CORE = ROOT / "GORC_paper_release_bundle" / "code" / "core_evaluators"

LVIS_CODE = ROOT / "GORC_paper_release_bundle" / "code" / "lvis_step9_step10"

SCRIPTS = ROOT / "scripts"

for p in [str(CORE), str(LVIS_CODE), str(SCRIPTS)]:

    if p not in sys.path:

        sys.path.insert(0, p)


from step8j_yoloworld_lvis_openvoc_baseline import (

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

import step11b_groundingdino_geometry_risk_calibration as gdino_cal              



SPLIT_CSV = ROOT / "outputs" / "step12a_coco_val_openset_protocol" / "csv" / "step12a_scene_split.csv"

ANNOTATION_DIR = ROOT / "outputs" / "step12a_coco_val_openset_protocol" / "annotations"

CLASS_MAP_CSV = ROOT / "outputs" / "step12a_coco_val_openset_protocol" / "csv" / "step12a_class_map.csv"

CANONICAL_CACHE = (

    ROOT

    / "outputs"

    / "step11a_groundingdino_coco_openset_baseline"

    / "csv"

    / "step11a_raw_predictions.csv"

)


PRED_COLUMNS = [

    "image_id",

    "split",

    "image_path",

    "det_id",

    "raw_label",

    "pred_label",

    "score",

    "x1",

    "y1",

    "x2",

    "y2",

]

PROCESSED_COLUMNS = ["image_id", "split", "image_path", "status", "num_detections", "error", "seconds"]

CONF_SWEEP = [

    0.05,

    0.10,

    0.15,

    0.20,

    0.25,

    0.30,

    0.35,

    0.40,

    0.45,

    0.50,

    0.55,

    0.60,

    0.65,

    0.70,

    0.75,

    0.80,

    0.85,

    0.90,

]

PROMPT_VARIANTS = ["canonical_periods", "comma_separated", "phrase_photo"]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_or_empty(path: Path, columns: Sequence[str]) -> pd.DataFrame:

    if path.exists() and path.stat().st_size > 0:

        return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))

    return pd.DataFrame(columns=list(columns))



def prompt_for(known_classes: Sequence[str], variant: str) -> str:

    names = [c.replace("_", " ") for c in known_classes]

    if variant == "canonical_periods":

        return ". ".join(names) + "."

    if variant == "comma_separated":

        return ", ".join(names)

    if variant == "phrase_photo":

        return ". ".join(f"a photo of a {name}" for name in names) + "."

    raise ValueError(f"unknown prompt variant: {variant}")



def canonical_label(raw_label: str, known_classes: Sequence[str]) -> str:

    text = str(raw_label).strip().lower().replace("_", " ")

    text_safe = safe_class_name(text)

    known_safe = [safe_class_name(c) for c in known_classes]

    if text_safe in known_safe:

        return text_safe

    for cls in sorted(known_safe, key=len, reverse=True):

        if cls.replace("_", " ") in text:

            return cls

    return ""



def post_process_grounding_dino(processor, outputs, inputs: dict, target_sizes, box_threshold: float, text_threshold: float):

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



def make_subset(split_csv: Path, out_csv: Path, cal_n: int, test_n: int) -> pd.DataFrame:

    split = clean_columns(pd.read_csv(split_csv, encoding="utf-8-sig"))

    split["image_id"] = split["image_id"].map(normalize_image_id)

    split["split"] = split["split"].astype(str).str.lower()

    cal = split[split["split"].eq("calibration")].sort_values("image_id").head(int(cal_n)).copy()

    test = split[split["split"].eq("test")].sort_values("image_id").head(int(test_n)).copy()

    subset = pd.concat([cal, test], ignore_index=True)

    subset.to_csv(out_csv, index=False, encoding="utf-8-sig")

    return subset



def materialize_canonical_cache(subset_ids: set[str], out_pred_csv: Path, out_processed_csv: Path) -> pd.DataFrame:

    pred = clean_columns(pd.read_csv(CANONICAL_CACHE, encoding="utf-8-sig"))

    pred["image_id"] = pred["image_id"].map(normalize_image_id)

    pred = pred[pred["image_id"].isin(subset_ids)].copy()

    pred = pred[[c for c in PRED_COLUMNS if c in pred.columns]].copy()

    for c in PRED_COLUMNS:

        if c not in pred.columns:

            pred[c] = ""

    pred = pred[PRED_COLUMNS].reset_index(drop=True)

    pred.to_csv(out_pred_csv, index=False, encoding="utf-8-sig")

    processed = (

        pred.groupby(["image_id", "split", "image_path"], dropna=False)

        .size()

        .reset_index(name="num_detections")

    )

    observed = set(processed["image_id"].map(normalize_image_id).tolist())

    empty_rows = []

    for image_id in sorted(subset_ids - observed):

        empty_rows.append(

            {

                "image_id": image_id,

                "split": "",

                "image_path": "",

                "status": "empty_cached",

                "num_detections": 0,

                "error": "",

                "seconds": 0.0,

            }

        )

    if len(processed):

        processed["status"] = "ok_cached"

        processed["error"] = ""

        processed["seconds"] = 0.0

        processed = processed[PROCESSED_COLUMNS]

    processed = pd.concat([processed, pd.DataFrame(empty_rows)], ignore_index=True)

    processed.to_csv(out_processed_csv, index=False, encoding="utf-8-sig")

    return pred



def run_variant_inference(

    *,

    variant: str,

    split_df: pd.DataFrame,

    known_classes: Sequence[str],

    pred_csv: Path,

    processed_csv: Path,

    model_id: str,

    device: str,

    box_threshold: float,

    text_threshold: float,

    max_images_per_run: int,

) -> tuple[pd.DataFrame, dict]:

    pred = read_or_empty(pred_csv, PRED_COLUMNS)

    processed = read_or_empty(processed_csv, PROCESSED_COLUMNS)

    processed_ids = set(

        processed[processed["status"].astype(str).isin(["ok", "empty"])]["image_id"].map(normalize_image_id).tolist()

    )

    todo = [r for _, r in split_df.iterrows() if normalize_image_id(r["image_id"]) not in processed_ids]

    if max_images_per_run and max_images_per_run > 0:

        todo = todo[: int(max_images_per_run)]


    runtime = {"model_loaded": False, "device_used": device, "prompt": prompt_for(known_classes, variant)}

    if not todo:

        return pred, runtime


    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

    import torch

    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor


    actual_device = device

    if actual_device.lower().startswith("cuda") and not torch.cuda.is_available():

        actual_device = "cpu"

    prompt = prompt_for(known_classes, variant)

    hardware = {

        "torch_version": str(torch.__version__),

        "cuda_available": bool(torch.cuda.is_available()),

        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),

        "device_requested": device,

        "device_used": actual_device,

        "gpu0_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",

    }

    runtime.update({"model_loaded": True, "device_used": actual_device, "prompt": prompt, "hardware": hardware})

    print(f"Grounding DINO prompt variant: {variant}")

    print(f"Model: {model_id}")

    print(f"Device: {actual_device}; GPU0={hardware.get('gpu0_name', '')}")

    print(f"Prompt: {prompt}")


    processor = AutoProcessor.from_pretrained(model_id)

    model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(actual_device)

    model.eval()


    pred_rows = pred.to_dict("records") if len(pred) else []

    processed_rows = processed.to_dict("records") if len(processed) else []

    with torch.no_grad():

        for r in tqdm(todo, total=len(todo), desc=f"Prompt sensitivity {variant}"):

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

                        float(box_threshold),

                        float(text_threshold),

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

                                "det_id": f"{image_id}_{variant}_gdino_{len(pred_rows):06d}",

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


    return pd.DataFrame(pred_rows, columns=PRED_COLUMNS), runtime



def selected_raw_threshold(pred: pd.DataFrame, gt: pd.DataFrame, csv_dir: Path) -> float:

    cal_eval, _ = evaluate_threshold_sweep(pred, gt, "calibration", CONF_SWEEP, 0.50, 0.90, "class", [0.50, 0.75])

    cal_eval.to_csv(csv_dir / "raw_calibration_sweep.csv", index=False, encoding="utf-8-sig")

    best_df = cal_eval[cal_eval["iou_threshold"].astype(float).round(4).eq(0.50)].copy()

    best = best_df.sort_values(

        ["precision_recall_unknown_balanced_score", "known_recall", "unknown_reject_rate_object_level", "known_precision"],

        ascending=[False, False, False, False],

    ).iloc[0].to_dict()

    pd.DataFrame([best]).to_csv(csv_dir / "raw_selected_calibration_config.csv", index=False, encoding="utf-8-sig")

    return float(best["conf_thr"])



def evaluate_variant(

    *,

    variant: str,

    pred_csv: Path,

    split_csv: Path,

    class_map_csv: Path,

    annotation_dir: Path,

    output_root: Path,

) -> dict:

    csv_dir = output_root / "csv"

    ensure_dir(csv_dir)

    known_classes = load_known_classes(class_map_csv)

    split_df = load_split(split_csv, ROOT, split_filter="all")

    gt = load_annotations(annotation_dir, split_df, known_classes)

    pred = clean_columns(pd.read_csv(pred_csv, encoding="utf-8-sig"))

    pred["image_id"] = pred["image_id"].map(normalize_image_id)

    subset_ids = set(split_df["image_id"].map(normalize_image_id).tolist())

    pred = pred[pred["image_id"].isin(subset_ids)].copy()

    for c in ["score", "x1", "y1", "x2", "y2"]:

        pred[c] = pd.to_numeric(pred[c], errors="coerce")

    pred = pred.dropna(subset=["score", "x1", "y1", "x2", "y2"]).reset_index(drop=True)

    pred["pred_label"] = pred["pred_label"].map(safe_class_name)

    pred.to_csv(pred_csv, index=False, encoding="utf-8-sig")


    raw_thr = selected_raw_threshold(pred, gt, csv_dir)


    candidates = gdino_cal.feature_candidates(pred, split_df, 0.50, 0.90, "class", 0.015)

    labeled = gdino_cal.label_candidates(candidates, gt, 0.50)

    labeled.to_csv(csv_dir / "candidate_features_labeled.csv", index=False, encoding="utf-8-sig")

    train = labeled[labeled["split"].astype(str).str.lower().eq("calibration")].copy()

    y = pd.to_numeric(train["risk_label_tp"], errors="coerce").fillna(0).astype(int)

    if y.nunique() < 2:

        raise ValueError(f"{variant}: calibration labels contain one class only; cannot fit reliability model.")

    model = gdino_cal.build_model(11)

    sample_weight = np.ones(len(train), dtype=float)

    sample_weight[train["risk_error_type"].eq("unknown_false_accept").to_numpy()] = 2.0

    sample_weight[train["risk_error_type"].eq("background_false_accept").to_numpy()] = 1.25

    model.fit(train, y, clf__sample_weight=sample_weight)


    scored = labeled.copy()

    scored["geometry_risk_score"] = model.predict_proba(scored)[:, 1].astype(float)

    scored, specs = gdino_cal.add_score_columns(

        scored,

        [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.80, 1.0],

        [0.25, 0.50, 0.75, 1.0],

    )

    scored.to_csv(csv_dir / "geometry_scored_candidates.csv", index=False, encoding="utf-8-sig")


    gt_cal = gt[gt["split"].astype(str).str.lower().eq("calibration")].copy()

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    grid = gdino_cal.build_policy_grid(scored, specs, gt_cal, known_classes, 0.50)

    raw_cal = grid[grid["score_col"].astype(str).eq("raw_score")].iloc[0]

    selected = gdino_cal.select_policies(

        grid,

        gdino_cal.finite_float(raw_cal["cal_AP50"], float("nan")),

        gdino_cal.finite_float(raw_cal["cal_AP"], float("nan")),

        0.005,

    )


    scored_test = scored[scored["split"].astype(str).str.lower().eq("test")].copy()

    raw_metrics = gdino_cal.eval_operating(scored_test, gt_test, "raw_score", raw_thr, 0.50)

    raw_ap = gdino_cal.ap_for_split(scored, gt_test, known_classes, "test", "raw_score")

    test_df = gdino_cal.evaluate_selected_on_test(scored, selected, gt_test, known_classes, raw_metrics, raw_ap, 0.50)

    raw_row = {

        "method": "prompt_sensitivity_raw",

        "selection_policy": "raw_max_cal_balanced_threshold",

        "mode": "raw",

        "score_col": "raw_score",

        "score_family": "raw",

        "description": "Raw Grounding DINO score with subset calibration threshold",

        "threshold": raw_thr,

        "split": "test",

        "iou_threshold": 0.50,

        "num_known_gt": raw_metrics["num_known_gt"],

        "num_unknown_gt": raw_metrics["num_unknown_gt"],

        "num_accepted_detections": raw_metrics["num_accepted_detections"],

        "tp_known": raw_metrics["tp_known"],

        "fp_known": raw_metrics["fp_known"],

        "fn_known": raw_metrics["fn_known"],

        "known_precision": raw_metrics["known_precision"],

        "known_recall": raw_metrics["known_recall"],

        "precision": raw_metrics["known_precision"],

        "recall": raw_metrics["known_recall"],

        "unknown_false_accept_objects": raw_metrics["unknown_false_accept_objects"],

        "unknown_false_accepts": raw_metrics["unknown_false_accept_objects"],

        "unknown_reject_rate_object_level": raw_metrics["unknown_reject_rate_object_level"],

        "precision_recall_unknown_balanced_score": raw_metrics["precision_recall_unknown_balanced_score"],

        "balanced": raw_metrics["precision_recall_unknown_balanced_score"],

        "background_false_accept_count": raw_metrics["background_false_accept_count"],

        "background_false_accepts": raw_metrics["background_false_accept_count"],

        "AP50": gdino_cal.finite_float(raw_ap.get("AP50", float("nan")), float("nan")),

        "AP75": gdino_cal.finite_float(raw_ap.get("AP75", float("nan")), float("nan")),

        "AP": gdino_cal.finite_float(raw_ap.get("AP", float("nan")), float("nan")),

        "delta_balanced_vs_raw": 0.0,

        "delta_precision_vs_raw": 0.0,

        "delta_recall_vs_raw": 0.0,

        "delta_ufa_vs_raw": 0,

        "delta_bg_fp_vs_raw": 0,

        "delta_AP50_vs_raw": 0.0,

        "delta_AP_vs_raw": 0.0,

        "meets_AP50_minus_0p005": True,

        "meets_AP_minus_0p005": True,

        "meets_success_criterion": False,

    }

    compact = pd.concat([pd.DataFrame([raw_row]), test_df], ignore_index=True)

    grid.to_csv(csv_dir / "policy_grid.csv", index=False, encoding="utf-8-sig")

    selected.to_csv(csv_dir / "selected_policies.csv", index=False, encoding="utf-8-sig")

    compact.to_csv(csv_dir / "compact_metrics.csv", index=False, encoding="utf-8-sig")


    def pick(mode: str) -> pd.Series:

        sub = compact[compact["mode"].astype(str).eq(mode)].copy()

        if sub.empty:

            raise ValueError(f"{variant}: missing mode {mode}")

        return sub.iloc[0]


    raw = pick("raw")

    rf = pick("reliability_first")

    ap_first = pick("ap_first")

    return {

        "detector": "Grounding DINO tiny",

        "prompt_variant": variant,

        "prompt": prompt_for(known_classes, variant),

        "subset_calibration_images": int((split_df["split"].astype(str) == "calibration").sum()),

        "subset_test_images": int((split_df["split"].astype(str) == "test").sum()),

        "candidate_count": int(len(pred)),

        "scored_candidate_count": int(len(scored)),

        "selected_threshold": float(raw_thr),

        "raw_selected_threshold": float(raw_thr),

        "RF_selected_threshold": float(rf["threshold"]),

        "AP_first_or_AP_P_selected_threshold": float(ap_first["threshold"]),

        "raw_B": float(raw["precision_recall_unknown_balanced_score"]),

        "raw_UFA": int(raw["unknown_false_accept_objects"]),

        "raw_BGFP": int(raw["background_false_accept_count"]),

        "raw_AP": float(raw["AP"]),

        "RF_B": float(rf["precision_recall_unknown_balanced_score"]),

        "RF_UFA": int(rf["unknown_false_accept_objects"]),

        "RF_BGFP": int(rf["background_false_accept_count"]),

        "RF_AP": float(rf["AP"]),

        "AP_first_or_AP_P_B": float(ap_first["precision_recall_unknown_balanced_score"]),

        "AP_first_or_AP_P_UFA": int(ap_first["unknown_false_accept_objects"]),

        "AP_first_or_AP_P_BGFP": int(ap_first["background_false_accept_count"]),

        "AP_first_or_AP_P_AP": float(ap_first["AP"]),

        "raw_AP50": float(raw["AP50"]),

        "raw_AP75": float(raw["AP75"]),

        "RF_AP50": float(rf["AP50"]),

        "RF_AP75": float(rf["AP75"]),

        "AP_first_or_AP_P_AP50": float(ap_first["AP50"]),

        "AP_first_or_AP_P_AP75": float(ap_first["AP75"]),

        "RF_score_col": str(rf["score_col"]),

        "AP_first_or_AP_P_score_col": str(ap_first["score_col"]),

        "variant_output_root": str(output_root),

    }



def write_summary(path: Path, rows: list[dict], runtimes: dict, report: dict) -> None:

    df = pd.DataFrame(rows)

    display_cols = [

        "prompt_variant",

        "candidate_count",

        "selected_threshold",

        "raw_B",

        "raw_UFA",

        "raw_BGFP",

        "raw_AP",

        "RF_B",

        "RF_UFA",

        "RF_BGFP",

        "RF_AP",

        "AP_first_or_AP_P_B",

        "AP_first_or_AP_P_UFA",

        "AP_first_or_AP_P_BGFP",

        "AP_first_or_AP_P_AP",

    ]

    lines = [

        "# Prompt Sensitivity Summary",

        "",

        "Status: PASS (fixed-subset diagnostic)",

        "",

        f"- Generated at: `{report['generated_at']}`",

        f"- Detector: `Grounding DINO tiny`",

        f"- Checkpoint: `{report['model_id']}`",

        f"- COCO subset: `{report['subset_calibration_images']}` calibration images and `{report['subset_test_images']}` test images.",

        "- Scope: fixed subset only; these rows are supplementary/limitation evidence and should not be promoted to the main manuscript.",

        "- Raw thresholds and GORC policies are selected on the subset calibration split only.",

        "- YOLO-World-l template sensitivity was not rerun because the existing Ultralytics YOLO-World path uses `set_classes` category prompts rather than a single natural-language prompt string comparable to Grounding DINO sentence templates; this remains a limitation.",

        "",

        "## Prompt Sets",

        "",

    ]

    for row in rows:

        rt = runtimes.get(row["prompt_variant"], {})

        lines.append(f"- `{row['prompt_variant']}`: `{row['prompt']}`")

        if rt.get("source") == "cached_full_run":

            lines.append(f"  - Source: cached full COCO run `{CANONICAL_CACHE}`.")

        else:

            hw = rt.get("hardware", {})

            lines.append(f"  - Hardware: requested `{hw.get('device_requested', '')}`, used `{hw.get('device_used', '')}`, GPU0 `{hw.get('gpu0_name', '')}`, CUDA_VISIBLE_DEVICES `{hw.get('cuda_visible_devices', '')}`.")

    lines.extend(["", "## Metrics", "", df[display_cols].to_markdown(index=False), ""])

    lines.extend(

        [

            "## Interpretation Boundary",

            "",

            "The subset audit checks whether the Grounding DINO candidate pool and downstream calibration are visibly sensitive to prompt punctuation/template changes. Because it uses a fixed subset rather than the full 5000-image protocol and does not include YOLO-World-l text-template reruns, it should be cited only as a supplementary prompt/candidate sensitivity diagnostic or limitation.",

        ]

    )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--output_root", type=Path, default=OUT)

    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny")

    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--box_threshold", type=float, default=0.05)

    parser.add_argument("--text_threshold", type=float, default=0.20)

    parser.add_argument("--cal_images", type=int, default=100)

    parser.add_argument("--test_images", type=int, default=300)

    parser.add_argument("--max_images_per_run", type=int, default=0)

    parser.add_argument("--eval_only", action="store_true")

    args = parser.parse_args()


    ensure_dir(args.output_root)

    subset_csv = args.output_root / "fixed_subset_split.csv"

    subset = make_subset(SPLIT_CSV, subset_csv, args.cal_images, args.test_images)

    split_df = load_split(subset_csv, ROOT, split_filter="all")

    known_classes = load_known_classes(CLASS_MAP_CSV)

    subset_ids = set(split_df["image_id"].map(normalize_image_id).tolist())


    runtimes: dict = {}

    for variant in PROMPT_VARIANTS:

        variant_root = args.output_root / "variants" / variant

        csv_dir = variant_root / "csv"

        ensure_dir(csv_dir)

        pred_csv = csv_dir / "raw_predictions.csv"

        processed_csv = csv_dir / "processed_images.csv"

        if variant == "canonical_periods":

            if not pred_csv.exists():

                materialize_canonical_cache(subset_ids, pred_csv, processed_csv)

            runtimes[variant] = {

                "source": "cached_full_run",

                "prompt": prompt_for(known_classes, variant),

                "model_id": args.model_id,

            }

        elif not args.eval_only:

            pred, runtime = run_variant_inference(

                variant=variant,

                split_df=split_df,

                known_classes=known_classes,

                pred_csv=pred_csv,

                processed_csv=processed_csv,

                model_id=args.model_id,

                device=args.device,

                box_threshold=args.box_threshold,

                text_threshold=args.text_threshold,

                max_images_per_run=args.max_images_per_run,

            )

            runtime["source"] = "detector_rerun_subset"

            runtimes[variant] = runtime

        else:

            runtimes[variant] = {"source": "existing_subset_cache", "prompt": prompt_for(known_classes, variant)}


    rows: list[dict] = []

    for variant in PROMPT_VARIANTS:

        variant_root = args.output_root / "variants" / variant

        pred_csv = variant_root / "csv" / "raw_predictions.csv"

        processed_csv = variant_root / "csv" / "processed_images.csv"

        processed = read_or_empty(processed_csv, PROCESSED_COLUMNS)

        ok = processed[processed["status"].astype(str).isin(["ok", "empty", "ok_cached", "empty_cached"])]

        if len(set(ok["image_id"].map(normalize_image_id).tolist())) < len(subset_ids):

            raise RuntimeError(f"{variant}: inference incomplete; rerun without --eval_only to resume.")

        rows.append(

            evaluate_variant(

                variant=variant,

                pred_csv=pred_csv,

                split_csv=subset_csv,

                class_map_csv=CLASS_MAP_CSV,

                annotation_dir=ANNOTATION_DIR,

                output_root=variant_root,

            )

        )


    out_csv = args.output_root / "prompt_sensitivity_coco_groundingdino.csv"

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    report = {

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "model_id": args.model_id,

        "box_threshold": float(args.box_threshold),

        "text_threshold": float(args.text_threshold),

        "subset_calibration_images": int((subset["split"].astype(str).str.lower() == "calibration").sum()),

        "subset_test_images": int((subset["split"].astype(str).str.lower() == "test").sum()),

        "prompt_variants": PROMPT_VARIANTS,

        "runtimes": runtimes,

        "outputs": {

            "csv": str(out_csv),

            "summary": str(args.output_root / "prompt_sensitivity_summary.md"),

            "subset_split": str(subset_csv),

        },

    }

    (args.output_root / "prompt_sensitivity_integrity_report.json").write_text(

        json.dumps(report, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    write_summary(args.output_root / "prompt_sensitivity_summary.md", rows, runtimes, report)

    print(pd.DataFrame(rows).to_string(index=False))



if __name__ == "__main__":

    main()

