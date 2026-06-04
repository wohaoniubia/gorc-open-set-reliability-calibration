


"""
Step10c: Grounding DINO feasibility smoke test.

This script uses the HuggingFace Grounding DINO interface when available. It is
deliberately limited to a small number of LVIS-Clear-Mini-300 test images and
does not run full evaluation or full inference.
"""


from __future__ import annotations


import argparse

import importlib.util

import json

import time

from datetime import datetime

from pathlib import Path

from typing import List


import pandas as pd

from PIL import Image

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import clean_columns, load_known_classes, load_split, normalize_image_id



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step10c_groundingdino_feasibility"



PRED_COLS = ["image_id", "image_path", "det_id", "raw_label", "score", "x1", "y1", "x2", "y2"]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def write_log(path: Path, lines: List[str]) -> None:

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv")

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--model_id", type=str, default="IDEA-Research/grounding-dino-tiny")

    parser.add_argument("--max_images", type=int, default=20)

    parser.add_argument("--box_threshold", type=float, default=0.25)

    parser.add_argument("--text_threshold", type=float, default=0.25)

    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    log_path = args.output_root / "step10c_install_or_runtime_log.md"

    pred_path = csv_dir / "step10c_smoke_predictions.csv"

    lines: List[str] = [

        "# Step10C Grounding DINO Feasibility Log",

        "",

        f"- Generated at: `{datetime.now().isoformat(timespec='seconds')}`",

        f"- Model id: `{args.model_id}`",

        f"- Max smoke images: `{args.max_images}`",

        "",

        "## Environment Check",

        "",

    ]


    installed = {m: importlib.util.find_spec(m) is not None for m in ["torch", "transformers", "PIL"]}

    for m, ok in installed.items():

        lines.append(f"- `{m}` available: `{ok}`")

    if not all(installed.values()):

        lines.append("")

        lines.append("Grounding DINO smoke test skipped because required packages are missing.")

        pd.DataFrame(columns=PRED_COLS).to_csv(pred_path, index=False, encoding="utf-8-sig")

        report = {

            "method": "Step10C Grounding DINO feasibility",

            "status": "skipped_missing_dependency",

            "installed": installed,

            "outputs": {

                "runtime_log": str(log_path),

                "smoke_predictions": str(pred_path),

                "integrity_report": str(args.output_root / "step10c_integrity_report.json"),

            },

        }

        write_log(log_path, lines)

        (args.output_root / "step10c_integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return


    try:

        import torch

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    except Exception as exc:

        lines.extend(["", "## Import Failure", "", f"`{repr(exc)}`"])

        pd.DataFrame(columns=PRED_COLS).to_csv(pred_path, index=False, encoding="utf-8-sig")

        report = {

            "method": "Step10C Grounding DINO feasibility",

            "status": "failed_import",

            "error": repr(exc),

            "outputs": {

                "runtime_log": str(log_path),

                "smoke_predictions": str(pred_path),

                "integrity_report": str(args.output_root / "step10c_integrity_report.json"),

            },

        }

        write_log(log_path, lines)

        (args.output_root / "step10c_integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        return


    device = args.device

    if device.lower() == "cuda" and not torch.cuda.is_available():

        device = "cpu"

    known_classes = load_known_classes(args.class_map_csv)

    prompt = ". ".join(c.replace("_", " ") for c in known_classes) + "."

    split = load_split(args.split_csv, args.project_root, split_filter="test")

    split = split.head(int(args.max_images)).copy()

    lines.extend(

        [

            "",

            "## Runtime Setup",

            "",

            f"- Device: `{device}`",

            f"- Prompt: `{prompt}`",

            f"- Smoke image count: `{len(split)}`",

        ]

    )


    rows: List[dict] = []

    status = "unknown"

    error = ""

    start_all = time.time()

    try:

        processor = AutoProcessor.from_pretrained(args.model_id)

        model = AutoModelForZeroShotObjectDetection.from_pretrained(args.model_id).to(device)

        model.eval()

        lines.append("- Model load: `ok`")

        with torch.no_grad():

            for _, r in tqdm(split.iterrows(), total=len(split), desc="Step10C GroundingDINO smoke"):

                image_id = normalize_image_id(r["image_id"])

                image_path = Path(str(r["image_path_resolved"]))

                if not image_path.exists():

                    continue

                image = Image.open(image_path).convert("RGB")

                inputs = processor(images=image, text=prompt, return_tensors="pt")

                inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

                outputs = model(**inputs)

                target_sizes = [(image.height, image.width)]

                try:

                    result = processor.post_process_grounded_object_detection(

                        outputs,

                        inputs.get("input_ids"),

                        box_threshold=float(args.box_threshold),

                        text_threshold=float(args.text_threshold),

                        target_sizes=target_sizes,

                    )[0]

                except TypeError:

                    try:

                        result = processor.post_process_grounded_object_detection(

                            outputs,

                            inputs.get("input_ids"),

                            threshold=float(args.box_threshold),

                            text_threshold=float(args.text_threshold),

                            target_sizes=target_sizes,

                        )[0]

                    except TypeError:

                        result = processor.post_process_grounded_object_detection(

                            outputs,

                            threshold=float(args.box_threshold),

                            text_threshold=float(args.text_threshold),

                            target_sizes=target_sizes,

                        )[0]

                boxes = result.get("boxes", [])

                scores = result.get("scores", [])

                labels = result.get("text_labels", result.get("labels", []))

                for j, box in enumerate(boxes):

                    score = float(scores[j].detach().cpu().item()) if hasattr(scores[j], "detach") else float(scores[j])

                    vals = box.detach().cpu().tolist() if hasattr(box, "detach") else list(box)

                    raw_label = str(labels[j]) if j < len(labels) else ""

                    rows.append(

                        {

                            "image_id": image_id,

                            "image_path": str(image_path),

                            "det_id": f"{image_id}_gdino_{j:04d}",

                            "raw_label": raw_label,

                            "score": score,

                            "x1": float(vals[0]),

                            "y1": float(vals[1]),

                            "x2": float(vals[2]),

                            "y2": float(vals[3]),

                        }

                    )

        status = "smoke_success"

    except Exception as exc:

        status = "failed_runtime"

        error = repr(exc)

        lines.extend(["", "## Runtime Failure", "", f"`{error}`"])


    pred_df = pd.DataFrame(rows, columns=PRED_COLS)

    pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

    elapsed = time.time() - start_all

    lines.extend(

        [

            "",

            "## Smoke Result",

            "",

            f"- Status: `{status}`",

            f"- Predictions: `{len(pred_df)}`",

            f"- Elapsed seconds: `{elapsed:.2f}`",

        ]

    )

    if status == "smoke_success":

        lines.append("- Full-run decision: feasible in principle; full LVIS-Clear-Mini-300 run should be explicitly launched separately.")


    report = {

        "method": "Step10C Grounding DINO feasibility",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "status": status,

        "error": error,

        "model_id": args.model_id,

        "device": device,

        "max_images": int(args.max_images),

        "num_predictions": int(len(pred_df)),

        "elapsed_seconds": float(elapsed),

        "outputs": {

            "runtime_log": str(log_path),

            "smoke_predictions": str(pred_path),

            "integrity_report": str(args.output_root / "step10c_integrity_report.json"),

        },

    }

    write_log(log_path, lines)

    (args.output_root / "step10c_integrity_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))



if __name__ == "__main__":

    main()

