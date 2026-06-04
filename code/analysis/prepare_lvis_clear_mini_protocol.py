


"""
Step9a: Build a larger LVIS-Clear-Mini public validation protocol.

The protocol keeps the Step8 known classes and unknown pressure, expands the
query split toward 300 local COCO/LVIS images, and writes Step9-specific output
names. It uses only local LVIS/COCO data.
"""


from __future__ import annotations


import argparse

import json

import shutil

from dataclasses import asdict

from pathlib import Path

from typing import List


import pandas as pd


from step8a_v5_prepare_lvis_clear_mini_protocol_fixed import (

    CategoryInfo,

    build_category_maps,

    build_query_tables,

    build_support_samples,

    collect_manual_supports,

    ensure_dir,

    load_json,

    parse_list,

    read_manual_manifest,

    resolve_class,

    safe_name,

    write_csv,

)



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OUTPUT_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_PUBLIC_ROOT = DEFAULT_PROJECT_ROOT / "data" / "public" / "lvis_mini_clear_300"

DEFAULT_KNOWN = ["bottle", "cup", "chair", "bowl"]

DEFAULT_UNKNOWN = [

    "apple",

    "banana",

    "orange_(fruit)",

    "plate",

    "fork",

    "vase",

    "glass_(drink_container)",

    "knife",

]



def safe_reset(path: Path, allowed_parent: Path) -> None:

    path = path.resolve()

    allowed_parent = allowed_parent.resolve()

    if not str(path).startswith(str(allowed_parent)):

        raise ValueError(f"Refusing to clear path outside {allowed_parent}: {path}")

    if path.exists():

        shutil.rmtree(path)



def copy_with_step9_name(src: Path, dst: Path) -> None:

    ensure_dir(dst.parent)

    if not src.exists():

        raise FileNotFoundError(src)

    shutil.copy2(src, dst)



def build_once(

    *,

    project_root: Path,

    lvis: dict,

    cats_by_safe: dict,

    known_names: List[str],

    unknown_names: List[str],

    shots: List[int],

    manual_support_dir: Path,

    manual_support_manifest: Path | None,

    coco_root: Path,

    image_dirs: List[Path],

    output_root: Path,

    public_root: Path,

    max_query_images: int,

    calibration_ratio: float,

    seed: int,

    min_query_box: float,

    min_query_area: float,

    max_selected_objects_per_image: int,

    min_mean_box_area_frac: float,

    image_mode: str,

) -> dict:

    csv_dir = output_root / "csv"

    annotations_dir = output_root / "annotations"

    public_images_dir = public_root / "images"

    ensure_dir(csv_dir)

    ensure_dir(annotations_dir)

    ensure_dir(public_images_dir)


    known_infos: List[CategoryInfo] = []

    unknown_infos: List[CategoryInfo] = []

    local_id = 0

    for name in known_names:

        c = resolve_class(name, cats_by_safe)

        known_infos.append(CategoryInfo(c.get("name", name), safe_name(c.get("name", name)), int(c["id"]), "known", local_id))

        local_id += 1

    for name in unknown_names:

        c = resolve_class(name, cats_by_safe)

        unknown_infos.append(CategoryInfo(c.get("name", name), safe_name(c.get("name", name)), int(c["id"]), "unknown", local_id))

        local_id += 1

    selected_infos = known_infos + unknown_infos


    class_map_df = pd.DataFrame([asdict(c) for c in selected_infos])

    write_csv(class_map_df, csv_dir / "step9a_class_map.csv")


    manifest = read_manual_manifest(manual_support_manifest, project_root) if manual_support_manifest else None

    manual_df = collect_manual_supports(manual_support_dir, known_infos, project_root, manifest)

    support_df, support_diag_df = build_support_samples(manual_df, shots)

    write_csv(support_df, csv_dir / "step9a_support_samples.csv")

    write_csv(support_diag_df, csv_dir / "step9a_support_diagnostics.csv")


    exclude_source_ids = set()

    if "source_image_id" in support_diag_df.columns:

        for v in support_diag_df["source_image_id"].fillna("").astype(str):

            v = v.strip()

            if v:

                try:

                    exclude_source_ids.add(str(int(float(v))))

                except Exception:

                    exclude_source_ids.add(v)


    scene_df, label_df, split_df, extra = build_query_tables(

        lvis=lvis,

        cats_by_id={},

        selected_infos=selected_infos,

        coco_root=coco_root,

        image_dirs=image_dirs,

        public_images_dir=public_images_dir,

        annotations_dir=annotations_dir,

        max_query_images=max_query_images,

        calibration_ratio=calibration_ratio,

        seed=seed,

        min_query_box=min_query_box,

        min_query_area=min_query_area,

        exclude_source_ids=exclude_source_ids,

        image_mode=image_mode,

        max_selected_objects_per_image=max_selected_objects_per_image if max_selected_objects_per_image > 0 else 999999,

        min_mean_box_area_frac=min_mean_box_area_frac,

    )


    write_csv(scene_df, csv_dir / "step9a_scene_split.csv")

    write_csv(label_df, csv_dir / "step9a_label_summary.csv")

    write_csv(split_df, csv_dir / "step9a_split_summary.csv")


    return {

        "known_infos": known_infos,

        "unknown_infos": unknown_infos,

        "manual_df": manual_df,

        "support_df": support_df,

        "support_diag_df": support_diag_df,

        "scene_df": scene_df,

        "label_df": label_df,

        "split_df": split_df,

        "extra": extra,

        "exclude_source_ids": exclude_source_ids,

    }



def write_markdown_summary(path: Path, report: dict, label_df: pd.DataFrame, split_df: pd.DataFrame) -> None:

    lines = [

        "# Step9a LVIS-Clear-Mini-300 Protocol Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Public root: `{report['public_root']}`",

        f"- Requested images: `{report['requested_query_images']}`",

        f"- Final target used: `{report['final_query_target']}`",

        f"- Selected images: `{report['num_query_images']}`",

        f"- Fallback used: `{report['fallback_used']}`",

        "",

        "## Split Summary",

        "",

        split_df.to_markdown(index=False),

        "",

        "## Label Summary",

        "",

        label_df.to_markdown(index=False),

        "",

        "## Notes",

        "",

        "- Known classes are fixed to bottle, cup, chair, bowl.",

        "- Unknown classes follow the Step8 balanced-unknown protocol.",

        "- Images are selected from local LVIS/COCO files only.",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--lvis_json", type=Path, default=None)

    parser.add_argument("--coco_root", type=Path, default=None)

    parser.add_argument("--image_dirs", type=str, default="")

    parser.add_argument("--manual_support_dir", type=Path, default=None)

    parser.add_argument("--manual_support_manifest", type=Path, default=None)

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)

    parser.add_argument("--public_root", type=Path, default=DEFAULT_PUBLIC_ROOT)

    parser.add_argument("--known_classes", type=str, default=",".join(DEFAULT_KNOWN))

    parser.add_argument("--unknown_classes", type=str, default=",".join(DEFAULT_UNKNOWN))

    parser.add_argument("--shots", type=str, default="1,2,4")

    parser.add_argument("--target_query_images", type=int, default=300)

    parser.add_argument("--fallback_query_images", type=int, default=200)

    parser.add_argument("--calibration_ratio", type=float, default=0.35)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--min_query_box", type=float, default=48)

    parser.add_argument("--min_query_area", type=float, default=2304)

    parser.add_argument("--max_selected_objects_per_image", type=int, default=8)

    parser.add_argument("--min_mean_box_area_frac", type=float, default=0.008)

    parser.add_argument("--image_mode", type=str, choices=["copy", "link"], default="copy")

    parser.add_argument("--reset", action="store_true", help="Clear the Step9a output/public directories before building.")

    args = parser.parse_args()


    project_root = args.project_root

    lvis_json = args.lvis_json or (project_root / "data" / "public" / "lvis" / "annotations" / "lvis_v1_val.json")

    coco_root = args.coco_root or (project_root / "data" / "public" / "lvis" / "coco")

    image_dirs = [Path(x) for x in args.image_dirs.split(";") if x.strip()]

    if not image_dirs:

        image_dirs = [coco_root / "val2017", coco_root / "train2017"]

    manual_support_dir = args.manual_support_dir or (project_root / "data" / "public" / "lvis_manual_support")

    output_root = args.output_root

    public_root = args.public_root


    if args.reset:

        safe_reset(output_root, project_root / "outputs")

        safe_reset(public_root, project_root / "data" / "public")

    ensure_dir(output_root)

    ensure_dir(public_root)


    lvis = load_json(lvis_json)

    _, cats_by_safe = build_category_maps(lvis)

    known_names = parse_list(args.known_classes, DEFAULT_KNOWN)

    unknown_names = parse_list(args.unknown_classes, DEFAULT_UNKNOWN)

    shots = [int(x.strip()) for x in args.shots.split(",") if x.strip()]


    result = build_once(

        project_root=project_root,

        lvis=lvis,

        cats_by_safe=cats_by_safe,

        known_names=known_names,

        unknown_names=unknown_names,

        shots=shots,

        manual_support_dir=manual_support_dir,

        manual_support_manifest=args.manual_support_manifest,

        coco_root=coco_root,

        image_dirs=image_dirs,

        output_root=output_root,

        public_root=public_root,

        max_query_images=int(args.target_query_images),

        calibration_ratio=float(args.calibration_ratio),

        seed=int(args.seed),

        min_query_box=float(args.min_query_box),

        min_query_area=float(args.min_query_area),

        max_selected_objects_per_image=int(args.max_selected_objects_per_image),

        min_mean_box_area_frac=float(args.min_mean_box_area_frac),

        image_mode=args.image_mode,

    )


    fallback_used = False

    final_target = int(args.target_query_images)

    selected_count = int(len(result["scene_df"]))

    if selected_count < int(args.target_query_images) and selected_count >= int(args.fallback_query_images):

        fallback_used = True

        final_target = int(args.fallback_query_images)

        safe_reset(output_root / "annotations", output_root)

        safe_reset(output_root / "csv", output_root)

        safe_reset(public_root / "images", public_root)

        result = build_once(

            project_root=project_root,

            lvis=lvis,

            cats_by_safe=cats_by_safe,

            known_names=known_names,

            unknown_names=unknown_names,

            shots=shots,

            manual_support_dir=manual_support_dir,

            manual_support_manifest=args.manual_support_manifest,

            coco_root=coco_root,

            image_dirs=image_dirs,

            output_root=output_root,

            public_root=public_root,

            max_query_images=final_target,

            calibration_ratio=float(args.calibration_ratio),

            seed=int(args.seed),

            min_query_box=float(args.min_query_box),

            min_query_area=float(args.min_query_area),

            max_selected_objects_per_image=int(args.max_selected_objects_per_image),

            min_mean_box_area_frac=float(args.min_mean_box_area_frac),

            image_mode=args.image_mode,

        )

        selected_count = int(len(result["scene_df"]))


    csv_dir = output_root / "csv"

    report = {

        "method": "Step9a LVIS-Clear-Mini larger public validation protocol",

        "project_root": str(project_root),

        "lvis_json": str(lvis_json),

        "coco_root": str(coco_root),

        "image_dirs": [str(p) for p in image_dirs],

        "output_root": str(output_root),

        "public_root": str(public_root),

        "known_classes": [asdict(x) for x in result["known_infos"]],

        "unknown_classes": [asdict(x) for x in result["unknown_infos"]],

        "shots": shots,

        "manual_support_dir": str(manual_support_dir),

        "num_manual_support_images": int(len(result["manual_df"])),

        "requested_query_images": int(args.target_query_images),

        "fallback_query_images": int(args.fallback_query_images),

        "fallback_used": bool(fallback_used),

        "final_query_target": int(final_target),

        "num_query_images": selected_count,

        "calibration_ratio": float(args.calibration_ratio),

        "min_query_box": float(args.min_query_box),

        "min_query_area": float(args.min_query_area),

        "max_selected_objects_per_image": int(args.max_selected_objects_per_image),

        "min_mean_box_area_frac": float(args.min_mean_box_area_frac),

        "image_mode": args.image_mode,

        "query_extra": result["extra"],

        "outputs": {

            "annotations": str(output_root / "annotations"),

            "scene_split": str(csv_dir / "step9a_scene_split.csv"),

            "support_samples": str(csv_dir / "step9a_support_samples.csv"),

            "support_diagnostics": str(csv_dir / "step9a_support_diagnostics.csv"),

            "class_map": str(csv_dir / "step9a_class_map.csv"),

            "label_summary": str(csv_dir / "step9a_label_summary.csv"),

            "split_summary": str(csv_dir / "step9a_split_summary.csv"),

            "markdown_summary": str(output_root / "step9a_summary.md"),

        },

    }

    with open(output_root / "step9a_integrity_report.json", "w", encoding="utf-8") as f:

        json.dump(report, f, ensure_ascii=False, indent=2)


    write_markdown_summary(output_root / "step9a_summary.md", report, result["label_df"], result["split_df"])


    print("\n========== Step9a completed ==========")

    print(f"Selected images: {selected_count}")

    print(f"Fallback used: {fallback_used}")

    print(f"Scene split: {csv_dir / 'step9a_scene_split.csv'}")

    print(f"Integrity: {output_root / 'step9a_integrity_report.json'}")

    print("\nSplit summary:")

    print(result["split_df"].to_string(index=False))

    print("\nLabel summary:")

    print(result["label_df"].to_string(index=False))



if __name__ == "__main__":

    main()

