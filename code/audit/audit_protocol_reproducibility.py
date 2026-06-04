from __future__ import annotations


import importlib.metadata as md

import json

from pathlib import Path


import pandas as pd



ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT / "GORC_paper_release_bundle"

OUT = ROOT / "outputs" / "gorc_major_revision_round2" / "07_protocol_reproducibility"


COCO_PROTOCOL = ROOT / "outputs" / "step12a_coco_val_openset_protocol"

LVIS_PROTOCOL = ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_json(path: Path) -> dict:

    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}



def read_csv(path: Path) -> pd.DataFrame:

    return pd.read_csv(path, encoding="utf-8-sig")



def pkg_version(name: str) -> str:

    try:

        return md.version(name)

    except Exception:

        return "unavailable"



def class_records(class_map: pd.DataFrame, role: str) -> list[dict]:

    rows = []

    sub = class_map[class_map["role"].astype(str).eq(role)].copy()

    for _, r in sub.iterrows():

        rows.append(

            {

                "raw_name": str(r.get("raw_name", r.get("safe_name", r.get("class_name", "")))),

                "class_name": str(r.get("class_name", r.get("safe_name", r.get("raw_name", "")))),

                "category_id": int(r.get("category_id", r.get("lvis_category_id", r.get("local_class_id", -1)))),

                "role": role,

            }

        )

    return rows



def write_json(path: Path, obj: object) -> None:

    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")



def write_ids(path: Path, ids: list[str]) -> None:

    path.write_text("\n".join(ids) + "\n", encoding="utf-8")



def protocol_card(protocol: str, report: dict, split_summary: pd.DataFrame, class_map: pd.DataFrame, split_csv: Path, card_path: Path) -> None:

    if protocol == "coco":

        title = "COCO-Val-OpenSet-5K Protocol Card"

        dataset_version = "COCO 2017 validation (`instances_val2017.json`, 5000 images)"

        image_split = "`step12a_scene_split.csv`; 1000 calibration / 4000 test, script default seed 1201"

        ann_rules = "All COCO val2017 images are used. VOC-style COCO categories are known; all other annotated COCO categories are unknown. Invalid non-positive-area boxes are skipped when protocol annotations are written."

        ignored = "None among annotated COCO categories; categories are partitioned into known or unknown."

        crowd = "`iscrowd` is retained in per-image JSON; evaluator uses the written protocol objects as detection targets without a separate crowd-ignore area."

        area = "No COCO small/medium/large area filter is applied beyond dropping non-positive boxes."

        license_note = "Raw COCO images/annotations are not redistributed by the revision outputs; users must obtain COCO 2017 under its original terms."

        known_path = "coco_known_classes.json"

        unknown_path = "coco_unknown_classes.json"

        cal_path = "coco_calibration_image_ids.txt"

        test_path = "coco_test_image_ids.txt"

    else:

        title = "LVIS-Clear-Mini-300 Protocol Card"

        dataset_version = "LVIS v1 validation annotations with locally resolved COCO images copied into `data/public/lvis_mini_clear_300`"

        image_split = "`step9a_scene_split.csv`; 105 calibration / 195 test, calibration ratio 0.35, script default seed 42"

        ann_rules = "Known classes are bottle, cup, chair, bowl. Unknown classes are apple, banana, orange_fruit, plate, fork, vase, glass_drink_container, knife. Query boxes require minimum box size 48 px and area 2304 px in Step9A."

        ignored = "LVIS categories outside the selected known/unknown set are not part of the written protocol targets."

        crowd = "The Step9A written JSON target objects are used directly; separate LVIS crowd/negative-category handling is not exposed in the detector-output evaluator."

        area = "Step9A query construction applies `min_query_box=48`, `min_query_area=2304`, `max_selected_objects_per_image=8`, and `min_mean_box_area_frac=0.008`."

        license_note = "Raw LVIS/COCO images are not redistributed by the revision outputs; users must obtain LVIS/COCO under their original terms."

        known_path = "lvis_known_classes.json"

        unknown_path = "lvis_unknown_classes.json"

        cal_path = "lvis_calibration_image_ids.txt"

        test_path = "lvis_test_image_ids.txt"


    known = class_records(class_map, "known")

    unknown = class_records(class_map, "unknown")

    split_lines = split_summary.to_markdown(index=False)

    lines = [

        f"# {title}",

        "",

        f"- Dataset version: {dataset_version}",

        f"- Image split file: `{split_csv}`",

        f"- Split construction: {image_split}",

        f"- Calibration image IDs: `{OUT / cal_path}`",

        f"- Test image IDs: `{OUT / test_path}`",

        f"- Known class file: `{OUT / known_path}`",

        f"- Unknown class file: `{OUT / unknown_path}`",

        f"- Known classes: {', '.join(r['class_name'] for r in known)}",

        f"- Unknown classes: {', '.join(r['class_name'] for r in unknown)}",

        f"- Ignored classes: {ignored}",

        f"- Annotation filtering rules: {ann_rules}",

        f"- Crowd/iscrowd handling: {crowd}",

        f"- Area range handling: {area}",

        "- Matching priority: detections are sorted by score and matched one-to-one to known-class targets first; accepted detections overlapping unknown targets above the IoU threshold count as object-level unknown false accepts; detections overlapping neither known nor unknown targets above threshold count as BG FP.",

        "- Main IoU threshold: 0.50 for reliability metrics.",

        "- AP evaluator settings: AP is computed on known-class detections over IoU thresholds 0.50:0.05:0.95; AP50/AP75 are reported separately.",

        "- Max detections: detector post-processing retains all candidates passing inference prefilter and class-wise NMS; no explicit max-det cap is recorded in the protocol builder.",

        f"- License/raw-data note: {license_note}",

        "",

        "## Split Summary",

        "",

        split_lines,

    ]

    card_path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def write_protocol_cards() -> None:

    coco_report = read_json(COCO_PROTOCOL / "step12a_integrity_report.json")

    lvis_report = read_json(LVIS_PROTOCOL / "step9a_integrity_report.json")

    coco_class = read_csv(COCO_PROTOCOL / "csv" / "step12a_class_map.csv")

    lvis_class = read_csv(LVIS_PROTOCOL / "csv" / "step9a_class_map.csv")

    coco_split = read_csv(COCO_PROTOCOL / "csv" / "step12a_scene_split.csv")

    lvis_split = read_csv(LVIS_PROTOCOL / "csv" / "step9a_scene_split.csv")


    write_json(OUT / "coco_known_classes.json", class_records(coco_class, "known"))

    write_json(OUT / "coco_unknown_classes.json", class_records(coco_class, "unknown"))

    write_ids(OUT / "coco_calibration_image_ids.txt", coco_split[coco_split["split"].eq("calibration")]["image_id"].astype(str).tolist())

    write_ids(OUT / "coco_test_image_ids.txt", coco_split[coco_split["split"].eq("test")]["image_id"].astype(str).tolist())


    write_json(OUT / "lvis_known_classes.json", class_records(lvis_class, "known"))

    write_json(OUT / "lvis_unknown_classes.json", class_records(lvis_class, "unknown"))

    write_ids(OUT / "lvis_calibration_image_ids.txt", lvis_split[lvis_split["split"].eq("calibration")]["image_id"].astype(str).tolist())

    write_ids(OUT / "lvis_test_image_ids.txt", lvis_split[lvis_split["split"].eq("test")]["image_id"].astype(str).tolist())


    protocol_card(

        "coco",

        coco_report,

        read_csv(COCO_PROTOCOL / "csv" / "step12a_split_summary.csv"),

        coco_class,

        COCO_PROTOCOL / "csv" / "step12a_scene_split.csv",

        OUT / "coco_val_openset_5k_protocol_card.md",

    )

    protocol_card(

        "lvis",

        lvis_report,

        read_csv(LVIS_PROTOCOL / "csv" / "step9a_split_summary.csv"),

        lvis_class,

        LVIS_PROTOCOL / "csv" / "step9a_scene_split.csv",

        OUT / "lvis_clear_mini_300_protocol_card.md",

    )



def detector_row(

    protocol: str,

    detector: str,

    checkpoint: str,

    package: str,

    input_resolution: str,

    prompt_source: str,

    prompt_template: str,

    class_mode: str,

    score_prefilter: str,

    nms: str,

    max_detections: str,

    tta: str,

    hardware: str,

    software_versions: str,

    cached_output_path: str,

    runtime_log_path: str,

    notes: str,

) -> dict:

    return {

        "protocol": protocol,

        "detector": detector,

        "checkpoint": checkpoint,

        "source_repository_or_package": package,

        "input_resolution": input_resolution,

        "prompt_list_source": prompt_source,

        "prompt_template": prompt_template,

        "class_query_mode": class_mode,

        "score_prefilter": score_prefilter,

        "NMS_threshold": nms,

        "max_detections": max_detections,

        "TTA": tta,

        "hardware": hardware,

        "software_versions": software_versions,

        "cached_output_path": cached_output_path,

        "runtime_log_path": runtime_log_path,

        "notes": notes,

    }



def write_detector_config() -> pd.DataFrame:

    sw_yolo = f"ultralytics={pkg_version('ultralytics')}; torch={pkg_version('torch')}; numpy={pkg_version('numpy')}; pandas={pkg_version('pandas')}"

    sw_gd = f"transformers={pkg_version('transformers')}; torch={pkg_version('torch')}; Pillow={pkg_version('Pillow')}; numpy={pkg_version('numpy')}; pandas={pkg_version('pandas')}"

    coco_yolo = read_json(BUNDLE / "data_outputs" / "coco_main" / "step12b_raw_baseline" / "step12b_integrity_report.json")

    lvis_yolo = read_json(BUNDLE / "data_outputs" / "lvis_main" / "step9b_raw_baseline" / "step9b_integrity_report.json")

    yws = read_json(BUNDLE / "data_outputs" / "integrity_reports" / "yolov8s_worldv2__step9b_integrity_report.json")

    gd_lvis = read_json(BUNDLE / "data_outputs" / "grounding_dino" / "step11a_raw" / "step11a_integrity_report.json")

    gd_coco = read_json(ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "step11a_integrity_report.json")


    rows = [

        detector_row(

            "COCO-Val-OpenSet-5K",

            "YOLO-World",

            coco_yolo.get("model_used", ""),

            "ultralytics YOLO",

            "imgsz=960",

            str(COCO_PROTOCOL / "csv" / "step12a_class_map.csv"),

            "canonical category names",

            "model.set_classes(prompts)",

            f"model predict conf=0.001; calibration-selected operating conf={coco_yolo.get('selected_conf_thr', '')}",

            f"YOLO predict iou=0.70; post NMS iou={coco_yolo.get('nms_iou', '')}, contain={coco_yolo.get('nms_contain', '')}, mode={coco_yolo.get('nms_mode', '')}",

            "not explicitly set in script; inherited package default",

            "not used",

            "cuda requested; exact GPU model not recorded",

            sw_yolo,

            coco_yolo.get("outputs", {}).get("raw_predictions", ""),

            coco_yolo.get("outputs", {}).get("processed_images", ""),

            "Primary COCO detector-output cache.",

        ),

        detector_row(

            "LVIS-Clear-Mini-300",

            "YOLO-World",

            lvis_yolo.get("model", ""),

            "ultralytics YOLO",

            "imgsz=960",

            str(LVIS_PROTOCOL / "csv" / "step9a_class_map.csv"),

            "synonym prompt strings",

            "model.set_classes(prompts)",

            f"model predict conf=0.001; calibration-selected operating conf={lvis_yolo.get('selected_conf_thr', '')}",

            f"YOLO predict iou=0.70; post NMS iou={lvis_yolo.get('nms_iou', '')}, contain={lvis_yolo.get('nms_contain', '')}, mode={lvis_yolo.get('nms_mode', '')}",

            "not explicitly set in script; inherited package default",

            "not used",

            "cuda requested; exact GPU model not recorded",

            sw_yolo,

            lvis_yolo.get("outputs", {}).get("raw_predictions", ""),

            lvis_yolo.get("outputs", {}).get("processed_images", ""),

            "Primary LVIS detector-output cache.",

        ),

        detector_row(

            "LVIS-Clear-Mini-300",

            "YOLO-World",

            yws.get("model", "yolov8s-worldv2.pt"),

            "ultralytics YOLO",

            "imgsz=960",

            str(LVIS_PROTOCOL / "csv" / "step9a_class_map.csv"),

            "synonym prompt strings",

            "model.set_classes(prompts)",

            f"model predict conf=0.001; calibration-selected operating conf={yws.get('selected_conf_thr', '')}",

            f"YOLO predict iou=0.70; post NMS iou={yws.get('nms_iou', '')}, contain={yws.get('nms_contain', '')}, mode={yws.get('nms_mode', '')}",

            "not explicitly set in script; inherited package default",

            "not used",

            "cuda requested; exact GPU model not recorded",

            sw_yolo,

            yws.get("outputs", {}).get("raw_predictions", ""),

            yws.get("outputs", {}).get("processed_images", ""),

            "LVIS model-size diagnostic cache.",

        ),

        detector_row(

            "LVIS-Clear-Mini-300",

            "Grounding DINO",

            gd_lvis.get("model_id", ""),

            "HuggingFace transformers AutoModelForZeroShotObjectDetection",

            "processor default; exact resize/crop not recorded",

            str(LVIS_PROTOCOL / "csv" / "step9a_class_map.csv"),

            "single concatenated class prompt ending with periods",

            "processor text prompt with known class names only",

            f"box_threshold={gd_lvis.get('box_threshold', '')}; text_threshold={gd_lvis.get('text_threshold', '')}; calibration-selected operating conf={gd_lvis.get('selected_conf_thr', '')}",

            f"post NMS iou={gd_lvis.get('nms_iou', '')}, contain={gd_lvis.get('nms_contain', '')}, mode={gd_lvis.get('nms_mode', '')}",

            "not explicitly set; all post-processed boxes retained before NMS",

            "not used",

            f"{gd_lvis.get('device_used', gd_lvis.get('device_requested', 'cuda'))}; exact GPU model not recorded",

            sw_gd,

            gd_lvis.get("outputs", {}).get("raw_predictions", ""),

            gd_lvis.get("outputs", {}).get("processed_images", ""),

            "LVIS cross-detector diagnostic cache.",

        ),

        detector_row(

            "COCO-Val-OpenSet-5K",

            "Grounding DINO",

            gd_coco.get("model_id", ""),

            "HuggingFace transformers AutoModelForZeroShotObjectDetection",

            "processor default; exact resize/crop not recorded",

            str(COCO_PROTOCOL / "csv" / "step12a_class_map.csv"),

            "single concatenated class prompt ending with periods",

            "processor text prompt with known class names only",

            f"box_threshold={gd_coco.get('box_threshold', '')}; text_threshold={gd_coco.get('text_threshold', '')}; calibration-selected operating conf={gd_coco.get('selected_conf_thr', '')}",

            f"post NMS iou={gd_coco.get('nms_iou', '')}, contain={gd_coco.get('nms_contain', '')}, mode={gd_coco.get('nms_mode', '')}",

            "not explicitly set; all post-processed boxes retained before NMS",

            "not used",

            f"{gd_coco.get('device_used', gd_coco.get('device_requested', 'cuda'))}; exact GPU model not recorded",

            sw_gd,

            gd_coco.get("outputs", {}).get("raw_predictions", ""),

            gd_coco.get("outputs", {}).get("processed_images", ""),

            "Round2 full COCO Grounding DINO detector-output cache.",

        ),

    ]

    df = pd.DataFrame(rows).fillna("")

    df.to_csv(OUT / "detector_configuration_table.csv", index=False, encoding="utf-8-sig")

    (OUT / "detector_configuration_table.md").write_text(

        "# Detector Configuration Table\n\n" + df.to_markdown(index=False) + "\n",

        encoding="utf-8",

    )


    missing = [

        "# Missing Detector Configuration Notes",

        "",

        "- YOLO-World scripts do not set an explicit `max_det`; the value is inherited from the installed Ultralytics package default.",

        "- Grounding DINO scripts use the HuggingFace processor defaults; exact resize/crop internals are not logged separately.",

        "- Exact GPU model and driver details are not recorded in the cached integrity reports; only CUDA/CPU device selection is logged.",

        "- These gaps should be disclosed as reproducibility limitations rather than filled with inferred values.",

    ]

    (OUT / "MISSING_DETECTOR_CONFIG.md").write_text("\n".join(missing) + "\n", encoding="utf-8")

    return df



def write_prefilter_sensitivity() -> pd.DataFrame:

    coco_yolo = read_json(BUNDLE / "data_outputs" / "coco_main" / "step12b_raw_baseline" / "step12b_integrity_report.json")

    lvis_yolo = read_json(BUNDLE / "data_outputs" / "lvis_main" / "step9b_raw_baseline" / "step9b_integrity_report.json")

    yws = read_json(BUNDLE / "data_outputs" / "integrity_reports" / "yolov8s_worldv2__step9b_integrity_report.json")

    gd_lvis = read_json(BUNDLE / "data_outputs" / "grounding_dino" / "step11a_raw" / "step11a_integrity_report.json")

    gd_coco = read_json(ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "step11a_integrity_report.json")


    rows = []

    for protocol, detector, variant, rep, prefilter, internal_nms in [

        ("COCO-Val-OpenSet-5K", "YOLO-World", "yolov8l-worldv2", coco_yolo, "min_conf=0.001", "YOLO iou=0.70"),

        ("LVIS-Clear-Mini-300", "YOLO-World", "yolov8l-worldv2", lvis_yolo, "min_conf=0.001", "YOLO iou=0.70"),

        ("LVIS-Clear-Mini-300", "YOLO-World", "yolov8s-worldv2", yws, "min_conf=0.001", "YOLO iou=0.70"),

        ("LVIS-Clear-Mini-300", "Grounding DINO", "grounding-dino-tiny", gd_lvis, f"box_threshold={gd_lvis.get('box_threshold', '')}; text_threshold={gd_lvis.get('text_threshold', '')}", "HF post-process thresholds"),

        ("COCO-Val-OpenSet-5K", "Grounding DINO", "grounding-dino-tiny", gd_coco, f"box_threshold={gd_coco.get('box_threshold', '')}; text_threshold={gd_coco.get('text_threshold', '')}", "HF post-process thresholds"),

    ]:

        rows.append(

            {

                "protocol": protocol,

                "detector": detector,

                "detector_variant": variant,

                "candidate_prefilter": prefilter,

                "model_internal_nms_or_postprocess": internal_nms,

                "post_nms": f"iou={rep.get('nms_iou', '')}; contain={rep.get('nms_contain', '')}; mode={rep.get('nms_mode', '')}",

                "selected_operating_threshold": rep.get("selected_conf_thr", ""),

                "num_images": rep.get("num_images_processed", rep.get("num_images_intended", "")),

                "raw_predictions": rep.get("num_predictions_raw", ""),

                "test_nms_candidates": rep.get("num_predictions_test_nms", ""),

                "lower_prefilter_tested": False,

                "sensitivity_rerun_status": "not_rerun_current_pool_documented",

                "source_path": rep.get("outputs", {}).get("raw_predictions", ""),

                "notes": "Current candidate pool documented; lower-prefilter rerun was not performed in round2.",

            }

        )

    df = pd.DataFrame(rows).fillna("")

    df.to_csv(OUT / "candidate_prefilter_sensitivity.csv", index=False, encoding="utf-8-sig")

    lines = [

        "# Candidate Prefilter Sensitivity",

        "",

        "Lower-prefilter detector reruns were not performed in round2. For YOLO-World, the cached detector pass already used a low model-confidence prefilter (`min_conf=0.001`) before class-wise post-NMS and calibration threshold selection. For Grounding DINO, the available passes used explicit box/text thresholds before post-NMS. The table below documents the current candidate pools and should be used as the manuscript limitation.",

        "",

        df.to_markdown(index=False),

    ]

    (OUT / "candidate_prefilter_sensitivity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return df



def main() -> None:

    ensure_dir(OUT)

    write_protocol_cards()

    detector = write_detector_config()

    prefilter = write_prefilter_sensitivity()

    manifest = {

        "generated_outputs": [

            "coco_val_openset_5k_protocol_card.md",

            "lvis_clear_mini_300_protocol_card.md",

            "coco_known_classes.json",

            "coco_unknown_classes.json",

            "coco_calibration_image_ids.txt",

            "coco_test_image_ids.txt",

            "lvis_known_classes.json",

            "lvis_unknown_classes.json",

            "lvis_calibration_image_ids.txt",

            "lvis_test_image_ids.txt",

            "detector_configuration_table.csv",

            "detector_configuration_table.md",

            "MISSING_DETECTOR_CONFIG.md",

            "candidate_prefilter_sensitivity.csv",

            "candidate_prefilter_sensitivity.md",

        ],

        "row_counts": {

            "detector_configuration_table": int(len(detector)),

            "candidate_prefilter_sensitivity": int(len(prefilter)),

        },

    }

    (OUT / "protocol_reproducibility_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))



if __name__ == "__main__":

    main()

