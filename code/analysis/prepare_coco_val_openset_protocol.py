


"""
Step12A: Build COCO-Val-OpenSet-5K protocol.

The protocol uses all COCO val2017 images, treats VOC-style COCO category names
as known classes, and treats all other annotated COCO categories as unknown.
"""


from __future__ import annotations


import argparse

import json

import random

from collections import Counter, defaultdict

from datetime import datetime

from pathlib import Path

from typing import Dict, List


import pandas as pd

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import safe_class_name



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COCO_ANN = DEFAULT_PROJECT_ROOT / "data" / "public" / "coco2017" / "annotations" / "instances_val2017.json"

DEFAULT_COCO_IMG_DIR = DEFAULT_PROJECT_ROOT / "data" / "public" / "coco2017" / "val2017"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step12a_coco_val_openset_protocol"


VOC_STYLE_KNOWN = [

    "person",

    "bicycle",

    "car",

    "motorcycle",

    "airplane",

    "bus",

    "train",

    "truck",

    "boat",

    "traffic light",

    "fire hydrant",

    "stop sign",

    "parking meter",

    "bench",

    "bird",

    "cat",

    "dog",

    "horse",

    "sheep",

    "cow",

]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def xywh_to_xyxy(box: List[float]) -> List[float]:

    x, y, w, h = [float(v) for v in box[:4]]

    return [x, y, x + w, y + h]



def load_coco(path: Path) -> dict:

    if not path.exists():

        raise FileNotFoundError(f"COCO annotation file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:

        return json.load(f)



def image_id_str(coco_id: int) -> str:

    return f"coco_{int(coco_id):012d}"



def build_split(images: List[dict], calibration_images: int, seed: int) -> Dict[int, str]:

    ids = [int(im["id"]) for im in images]

    rng = random.Random(int(seed))

    rng.shuffle(ids)

    cal = set(ids[: int(calibration_images)])

    return {i: ("calibration" if i in cal else "test") for i in ids}



def write_markdown(path: Path, report: dict, label_summary: pd.DataFrame, split_summary: pd.DataFrame) -> None:

    lines = [

        "# Step12A COCO-Val-OpenSet-5K Protocol Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- COCO annotation: `{report['coco_annotation_json']}`",

        f"- COCO image dir: `{report['coco_image_dir']}`",

        f"- Total images used: `{report['num_images_total']}`",

        f"- Calibration images: `{report['num_calibration_images']}`",

        f"- Test images: `{report['num_test_images']}`",

        f"- Known classes: `{report['num_known_classes']}`",

        f"- Unknown classes: `{report['num_unknown_classes']}`",

        f"- All 5000 images used: `{report['all_5000_images_used']}`",

        f"- Image filtering rule: `{report['image_filtering_rule']}`",

        "",

        "## Known Classes",

        "",

        ", ".join(report["known_classes"]),

        "",

        "## Split Summary",

        "",

        split_summary.to_markdown(index=False),

        "",

        "## Label Summary",

        "",

        label_summary.to_markdown(index=False),

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--coco_annotation_json", type=Path, default=DEFAULT_COCO_ANN)

    parser.add_argument("--coco_image_dir", type=Path, default=DEFAULT_COCO_IMG_DIR)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--calibration_images", type=int, default=1000)

    parser.add_argument("--seed", type=int, default=1201)

    args = parser.parse_args()


    if not args.coco_annotation_json.exists():

        raise FileNotFoundError(f"Required COCO annotation file is missing: {args.coco_annotation_json}")

    if not args.coco_image_dir.exists():

        raise FileNotFoundError(f"Required COCO image directory is missing: {args.coco_image_dir}")


    csv_dir = args.output_root / "csv"

    ann_dir = args.output_root / "annotations"

    ensure_dir(csv_dir)

    ensure_dir(ann_dir)


    print("\n========== Step12A COCO-Val-OpenSet-5K protocol ==========")

    print(f"annotation={args.coco_annotation_json}")

    print(f"image_dir={args.coco_image_dir}")


    coco = load_coco(args.coco_annotation_json)

    categories = sorted(coco.get("categories", []), key=lambda c: int(c["id"]))

    id_to_cat = {int(c["id"]): c for c in categories}

    coco_names = {str(c["name"]) for c in categories}

    missing = [c for c in VOC_STYLE_KNOWN if c not in coco_names]

    if missing:

        raise ValueError(f"Requested VOC-style known classes missing from COCO categories: {missing}")

    known_classes = [c for c in VOC_STYLE_KNOWN if c in coco_names]

    known_set = set(known_classes)


    annotations_by_image: Dict[int, List[dict]] = defaultdict(list)

    for ann in coco.get("annotations", []):

        annotations_by_image[int(ann["image_id"])].append(ann)


    images = sorted(coco.get("images", []), key=lambda im: int(im["id"]))

    split_map = build_split(images, args.calibration_images, args.seed)


    split_rows: List[dict] = []

    class_counts = Counter()

    class_image_counts = defaultdict(set)

    known_object_counts = Counter()

    unknown_object_counts = Counter()

    known_images = defaultdict(int)

    unknown_images = defaultdict(int)

    missing_images: List[str] = []


    for im in tqdm(images, desc="Step12A write per-image annotations"):

        coco_id = int(im["id"])

        image_id = image_id_str(coco_id)

        file_name = str(im["file_name"])

        image_path = args.coco_image_dir / file_name

        if not image_path.exists():

            missing_images.append(str(image_path))

        split = split_map[coco_id]

        objects = []

        num_known = 0

        num_unknown = 0

        for k, ann in enumerate(annotations_by_image.get(coco_id, [])):

            cat = id_to_cat.get(int(ann["category_id"]))

            if not cat:

                continue

            label = str(cat["name"])

            x1, y1, x2, y2 = xywh_to_xyxy(ann["bbox"])

            if x2 <= x1 or y2 <= y1:

                continue

            is_known = label in known_set

            super_type = "known" if is_known else "unknown"

            obj = {

                "object_id": int(ann.get("id", k + 1)),

                "id": int(ann.get("id", k + 1)),

                "label": safe_class_name(label),

                "raw_label": label,

                "category_id": int(ann["category_id"]),

                "super_type": super_type,

                "bbox": [float(x1), float(y1), float(x2), float(y2)],

                "bbox_format": "xyxy",

                "area": float(ann.get("area", max(0.0, x2 - x1) * max(0.0, y2 - y1))),

                "iscrowd": int(ann.get("iscrowd", 0)),

                "difficult": False,

                "note": "COCO val2017 annotation",

            }

            objects.append(obj)

            class_counts[label] += 1

            class_image_counts[label].add(coco_id)

            if is_known:

                num_known += 1

                known_object_counts[label] += 1

            else:

                num_unknown += 1

                unknown_object_counts[label] += 1

        if num_known:

            known_images[split] += 1

        if num_unknown:

            unknown_images[split] += 1

        out_ann = {

            "image_id": image_id,

            "coco_image_id": coco_id,

            "file_name": file_name,

            "image_path": str(image_path),

            "width": int(im.get("width", 0)),

            "height": int(im.get("height", 0)),

            "scene_group": "coco_val2017_openset_5k",

            "split": split,

            "objects": objects,

        }

        with open(ann_dir / f"{image_id}.json", "w", encoding="utf-8") as f:

            json.dump(out_ann, f, ensure_ascii=False, indent=2)

        split_rows.append(

            {

                "image_id": image_id,

                "coco_image_id": coco_id,

                "file_name": file_name,

                "image_path": str(image_path),

                "split": split,

                "scene_group": "coco_val2017_openset_5k",

                "width": int(im.get("width", 0)),

                "height": int(im.get("height", 0)),

                "num_known_objects": int(num_known),

                "num_unknown_objects": int(num_unknown),

                "num_selected_objects": int(num_known + num_unknown),

                "has_known_object": bool(num_known > 0),

                "has_unknown_object": bool(num_unknown > 0),

            }

        )


    class_rows = []

    for c in categories:

        name = str(c["name"])

        class_rows.append(

            {

                "category_id": int(c["id"]),

                "raw_name": name,

                "class_name": safe_class_name(name),

                "role": "known" if name in known_set else "unknown",

                "supercategory": str(c.get("supercategory", "")),

                "is_known": bool(name in known_set),

                "object_count": int(class_counts[name]),

                "image_count": int(len(class_image_counts[name])),

            }

        )

    class_map = pd.DataFrame(class_rows)

    split_df = pd.DataFrame(split_rows)


    label_summary = (

        class_map.assign(role=class_map["is_known"].map(lambda x: "known" if x else "unknown"))

        .groupby("role", as_index=False)

        .agg(num_classes=("class_name", "count"), object_count=("object_count", "sum"), image_count=("image_count", "sum"))

    )

    class_map.to_csv(csv_dir / "step12a_class_map.csv", index=False, encoding="utf-8-sig")

    split_df.to_csv(csv_dir / "step12a_scene_split.csv", index=False, encoding="utf-8-sig")

    label_summary.to_csv(csv_dir / "step12a_label_summary.csv", index=False, encoding="utf-8-sig")

    class_map.to_csv(csv_dir / "step12a_per_class_counts.csv", index=False, encoding="utf-8-sig")


    split_summary = (

        split_df.groupby("split", as_index=False)

        .agg(

            num_images=("image_id", "count"),

            images_with_known=("has_known_object", "sum"),

            images_with_unknown=("has_unknown_object", "sum"),

            known_objects=("num_known_objects", "sum"),

            unknown_objects=("num_unknown_objects", "sum"),

            selected_objects=("num_selected_objects", "sum"),

        )

        .sort_values("split")

    )

    split_summary.to_csv(csv_dir / "step12a_split_summary.csv", index=False, encoding="utf-8-sig")


    known_counts = {k: int(v) for k, v in sorted(known_object_counts.items())}

    unknown_counts = {k: int(v) for k, v in sorted(unknown_object_counts.items())}

    report = {

        "method": "Step12A COCO-Val-OpenSet-5K protocol construction",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "coco_annotation_json": str(args.coco_annotation_json),

        "coco_image_dir": str(args.coco_image_dir),

        "output_root": str(args.output_root),

        "num_images_total": int(len(images)),

        "num_calibration_images": int((split_df["split"] == "calibration").sum()),

        "num_test_images": int((split_df["split"] == "test").sum()),

        "num_known_classes": int(len(known_classes)),

        "num_unknown_classes": int(len(categories) - len(known_classes)),

        "known_classes": known_classes,

        "unknown_classes": [str(c["name"]) for c in categories if str(c["name"]) not in known_set],

        "known_object_counts": known_counts,

        "unknown_object_counts": unknown_counts,

        "num_known_objects_total": int(sum(known_counts.values())),

        "num_unknown_objects_total": int(sum(unknown_counts.values())),

        "image_filtering_rule": "No image filtering; all COCO val2017 images are retained, including images without selected known/unknown objects for background-FP analysis.",

        "all_5000_images_used": bool(len(images) == 5000 and not missing_images),

        "missing_image_count": int(len(missing_images)),

        "missing_images_first20": missing_images[:20],

        "split_seed": int(args.seed),

        "outputs": {

            "scene_split": str(csv_dir / "step12a_scene_split.csv"),

            "class_map": str(csv_dir / "step12a_class_map.csv"),

            "label_summary": str(csv_dir / "step12a_label_summary.csv"),

            "split_summary": str(csv_dir / "step12a_split_summary.csv"),

            "annotations": str(ann_dir),

            "summary": str(args.output_root / "step12a_summary.md"),

            "integrity_report": str(args.output_root / "step12a_integrity_report.json"),

        },

    }

    with open(args.output_root / "step12a_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)

    write_markdown(args.output_root / "step12a_summary.md", report, label_summary, split_summary)


    print("\n========== Step12A completed ==========")

    print(split_summary.to_string(index=False))

    print(label_summary.to_string(index=False))



if __name__ == "__main__":

    main()

