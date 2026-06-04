from __future__ import annotations


import json

from pathlib import Path


import pandas as pd



ROOT = Path(__file__).resolve().parents[2]

BUNDLE = ROOT / "GORC_paper_release_bundle"

ROUND2 = ROOT / "outputs" / "gorc_major_revision_round2"

OUT = ROUND2 / "06_cross_detector"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_json(path: Path) -> dict:

    if not path.exists():

        return {}

    return json.loads(path.read_text(encoding="utf-8"))



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        return pd.DataFrame()

    return pd.read_csv(path, encoding="utf-8-sig")



def write_md(path: Path, title: str, df: pd.DataFrame, notes: list[str] | None = None) -> None:

    lines = [f"# {title}", ""]

    for note in notes or []:

        lines.append(f"- {note}")

    if notes:

        lines.append("")

    if len(df):

        lines.append(df.fillna("").to_markdown(index=False))

    else:

        lines.append("_No rows._")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def compact_to_row(row: pd.Series, *, detector: str, variant: str, method_label: str, mode: str, source_path: Path, notes: str) -> dict:

    return {

        "protocol": "COCO-Val-OpenSet-5K",

        "detector": detector,

        "detector_variant": variant,

        "method": method_label,

        "mode": mode,

        "threshold": row.get("threshold", row.get("conf_thr", "")),

        "balanced": row.get("precision_recall_unknown_balanced_score", row.get("balanced", "")),

        "precision": row.get("known_precision", row.get("precision", "")),

        "recall": row.get("known_recall", row.get("recall", "")),

        "unknown_false_accepts": row.get("unknown_false_accept_objects", row.get("unknown_false_accepts", "")),

        "unknown_reject": row.get("unknown_reject_rate_object_level", row.get("unknown_reject", "")),

        "background_false_accepts": row.get("background_false_accept_count", row.get("background_false_accepts", "")),

        "AP50": row.get("AP50", ""),

        "AP75": row.get("AP75", ""),

        "AP": row.get("AP", ""),

        "delta_balanced_vs_raw": row.get("delta_balanced_vs_raw", ""),

        "delta_ufa_vs_raw": row.get("delta_ufa_vs_raw", ""),

        "delta_bg_fp_vs_raw": row.get("delta_bg_fp_vs_raw", ""),

        "meets_ap_constrained_success": row.get("meets_success_criterion", ""),

        "source_path": str(source_path),

        "notes": notes,

    }



def build_coco_groundingdino() -> pd.DataFrame:

    raw_path = ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "csv" / "step11a_compact_paper_metrics.csv"

    gorc_path = ROOT / "outputs" / "step11b_groundingdino_coco_geometry_risk" / "csv" / "step11b_compact_paper_metrics.csv"

    raw = read_csv(raw_path)

    gorc = read_csv(gorc_path)

    rows: list[dict] = []

    if len(raw):

        rows.append(

            compact_to_row(

                raw.iloc[0],

                detector="Grounding DINO",

                variant="grounding-dino-tiny",

                method_label="Grounding DINO tiny raw",

                mode="raw",

                source_path=raw_path,

                notes="Full COCO run launched in round2 with calibration-selected raw threshold.",

            )

        )

    if len(gorc):

        method_map = {

            "reliability_first": "Grounding DINO COCO GORC RF",

            "ap_first": "Grounding DINO COCO GORC AP-constrained",

            "ap_constrained": "Grounding DINO COCO GORC AP-constrained candidate",

        }

        for _, r in gorc.iterrows():

            mode = str(r.get("mode", ""))

            if mode == "raw":

                continue

            rows.append(

                compact_to_row(

                    r,

                    detector="Grounding DINO",

                    variant="grounding-dino-tiny",

                    method_label=method_map.get(mode, f"Grounding DINO COCO {mode}"),

                    mode=mode,

                    source_path=gorc_path,

                    notes=(

                        "Step11B constrained-grid finish: selection on calibration only; "

                        "final rows evaluated with the original evaluator on held-out test."

                    ),

                )

            )

    df = pd.DataFrame(rows)

    df.to_csv(OUT / "coco_groundingdino_main_results.csv", index=False, encoding="utf-8-sig")

    write_md(

        OUT / "coco_groundingdino_main_results.md",

        "COCO Grounding DINO Main Results",

        df,

        [

            "Grounding DINO COCO was not cached initially; round2 ran the resumable Step11A detector pass on all 5000 images.",

            "The original full Step11B grid was interrupted after scored candidates were written; the finish script uses a constrained calibration grid and exact held-out evaluation for selected rows.",

            "These rows should be described as additional detector-family evidence, not as a universal detector-agnostic guarantee.",

        ],

    )

    return df



def build_model_size() -> pd.DataFrame:

    src = BUNDLE / "data_outputs" / "grounding_dino" / "step11c_cross_detector" / "csv" / "step11c_main_comparison.csv"

    df = read_csv(src)

    rows: list[dict] = []

    if len(df):

        yolo = df[df["detector"].astype(str).eq("YOLO-World")].copy()

        for _, r in yolo.iterrows():

            rows.append(

                {

                    "protocol": "LVIS-Clear-Mini-300",

                    "detector": r.get("detector", ""),

                    "detector_variant": r.get("detector_variant", ""),

                    "method_group": r.get("method_group", ""),

                    "method_name": r.get("method_name", ""),

                    "selection_policy": r.get("selection_policy", ""),

                    "threshold": r.get("threshold", ""),

                    "balanced": r.get("balanced", ""),

                    "precision": r.get("precision", ""),

                    "recall": r.get("recall", ""),

                    "unknown_false_accepts": r.get("unknown_false_accepts", ""),

                    "unknown_reject": r.get("unknown_reject", ""),

                    "background_false_accepts": r.get("background_false_accepts", ""),

                    "AP50": r.get("AP50", ""),

                    "AP75": r.get("AP75", ""),

                    "AP": r.get("AP", ""),

                    "source_path": r.get("source_path", str(src)),

                    "cache_scope": "LVIS cached diagnostic; no COCO YOLO-World-s/m cache found in release bundle",

                    "notes": r.get("notes", ""),

                }

            )

    out = pd.DataFrame(rows)

    out = out.fillna("")

    out.to_csv(OUT / "yoloworld_model_size_diagnostic.csv", index=False, encoding="utf-8-sig")

    write_md(

        OUT / "yoloworld_model_size_diagnostic.md",

        "YOLO-World Model-Size Diagnostic",

        out,

        [

            "Cached model-size evidence is LVIS-only: YOLO-World-s and YOLO-World-l are available on LVIS-Clear-Mini-300.",

            "No COCO YOLO-World-s/m cache was found; the COCO primary detector remains YOLO-World-l.",

        ],

    )

    return out



def unique_prompts(path: Path, limit: int = 80) -> tuple[int, str]:

    if not path.exists():

        return 0, ""

    vals = pd.read_csv(path, usecols=["prompt"], encoding="utf-8-sig")["prompt"].dropna().astype(str).drop_duplicates().tolist()

    return len(vals), "; ".join(vals[:limit])



def build_prompt_audit() -> pd.DataFrame:

    rows: list[dict] = []


    coco_int = read_json(BUNDLE / "data_outputs" / "coco_main" / "step12b_raw_baseline" / "step12b_integrity_report.json")

    coco_prompt_path = BUNDLE / "data_outputs" / "coco_main" / "step12b_raw_baseline" / "csv" / "step12b_yoloworld_raw_predictions.csv"

    n, prompt_list = unique_prompts(coco_prompt_path)

    rows.append(

        {

            "protocol": "COCO-Val-OpenSet-5K",

            "detector": "YOLO-World",

            "detector_variant": coco_int.get("model_requested", "yolov8l-worldv2.pt"),

            "prompt_mode": coco_int.get("prompt_mode", "canonical"),

            "prompt_template": "category name only",

            "num_prompt_strings_observed": n,

            "prompt_list_or_summary": prompt_list,

            "full_sensitivity_run": False,

            "status": "configuration_recorded_no_variant_eval",

            "source_path": str(coco_prompt_path),

            "notes": "No alternative full COCO prompt-variant cache was found.",

        }

    )


    lvis_int = read_json(BUNDLE / "data_outputs" / "lvis_main" / "step9b_raw_baseline" / "step9b_integrity_report.json")

    lvis_prompt_path = BUNDLE / "data_outputs" / "lvis_main" / "step9b_raw_baseline" / "csv" / "step9b_yoloworld_raw_predictions.csv"

    n, prompt_list = unique_prompts(lvis_prompt_path)

    rows.append(

        {

            "protocol": "LVIS-Clear-Mini-300",

            "detector": "YOLO-World",

            "detector_variant": lvis_int.get("model", "yolov8l-worldv2.pt"),

            "prompt_mode": lvis_int.get("prompt_mode", "synonyms"),

            "prompt_template": "synonym strings from cached detector run",

            "num_prompt_strings_observed": n,

            "prompt_list_or_summary": prompt_list,

            "full_sensitivity_run": False,

            "status": "configuration_recorded_no_variant_eval",

            "source_path": str(lvis_prompt_path),

            "notes": "Synonym prompt mode is recorded; no paired official-name/template prompt experiment was cached.",

        }

    )


    yws_int = read_json(BUNDLE / "data_outputs" / "integrity_reports" / "yolov8s_worldv2__step9b_integrity_report.json")

    yws_prompt_path = BUNDLE / "data_outputs" / "lvis_main" / "step9d_external" / "yolov8s_worldv2" / "csv" / "step9b_yoloworld_raw_predictions.csv"

    n, prompt_list = unique_prompts(yws_prompt_path)

    rows.append(

        {

            "protocol": "LVIS-Clear-Mini-300",

            "detector": "YOLO-World",

            "detector_variant": yws_int.get("model", "yolov8s-worldv2.pt"),

            "prompt_mode": yws_int.get("prompt_mode", "synonyms"),

            "prompt_template": "synonym strings from cached detector run",

            "num_prompt_strings_observed": n,

            "prompt_list_or_summary": prompt_list,

            "full_sensitivity_run": False,

            "status": "configuration_recorded_no_variant_eval",

            "source_path": str(yws_prompt_path),

            "notes": "Smaller YOLO-World variant uses the same LVIS synonym prompt design.",

        }

    )


    gd_lvis = read_json(BUNDLE / "data_outputs" / "grounding_dino" / "step11a_raw" / "step11a_integrity_report.json")

    rows.append(

        {

            "protocol": "LVIS-Clear-Mini-300",

            "detector": "Grounding DINO",

            "detector_variant": gd_lvis.get("model_id", "IDEA-Research/grounding-dino-tiny"),

            "prompt_mode": "single concatenated class prompt",

            "prompt_template": "'. '.join(class names) + '.'",

            "num_prompt_strings_observed": len(gd_lvis.get("known_classes", [])),

            "prompt_list_or_summary": gd_lvis.get("prompt", ""),

            "full_sensitivity_run": False,

            "status": "configuration_recorded_no_variant_eval",

            "source_path": str(BUNDLE / "data_outputs" / "grounding_dino" / "step11a_raw" / "step11a_integrity_report.json"),

            "notes": "No paired prompt variant was run for LVIS Grounding DINO.",

        }

    )


    gd_coco = read_json(ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "step11a_integrity_report.json")

    rows.append(

        {

            "protocol": "COCO-Val-OpenSet-5K",

            "detector": "Grounding DINO",

            "detector_variant": gd_coco.get("model_id", "IDEA-Research/grounding-dino-tiny"),

            "prompt_mode": "single concatenated class prompt",

            "prompt_template": "'. '.join(class names) + '.'",

            "num_prompt_strings_observed": len(gd_coco.get("known_classes", [])),

            "prompt_list_or_summary": gd_coco.get("prompt", ""),

            "full_sensitivity_run": False,

            "status": "configuration_recorded_no_variant_eval",

            "source_path": str(ROOT / "outputs" / "step11a_groundingdino_coco_openset_baseline" / "step11a_integrity_report.json"),

            "notes": "Round2 ran full COCO Grounding DINO with the concatenated class prompt; alternate prompt sensitivity remains untested.",

        }

    )


    out = pd.DataFrame(rows)

    out.to_csv(OUT / "prompt_sensitivity_audit.csv", index=False, encoding="utf-8-sig")

    write_md(

        OUT / "prompt_sensitivity_audit.md",

        "Prompt Sensitivity Audit",

        out,

        [

            "Exact prompt configurations are recorded for the public detector-output runs.",

            "No full paired prompt-variant experiment was available or launched; manuscript claims should state that prompt sensitivity remains a limitation.",

        ],

    )

    return out



def main() -> None:

    ensure_dir(OUT)

    coco = build_coco_groundingdino()

    model_size = build_model_size()

    prompts = build_prompt_audit()

    manifest = {

        "generated_outputs": {

            "coco_groundingdino_main_results": str(OUT / "coco_groundingdino_main_results.csv"),

            "yoloworld_model_size_diagnostic": str(OUT / "yoloworld_model_size_diagnostic.csv"),

            "prompt_sensitivity_audit": str(OUT / "prompt_sensitivity_audit.csv"),

        },

        "row_counts": {

            "coco_groundingdino_main_results": int(len(coco)),

            "yoloworld_model_size_diagnostic": int(len(model_size)),

            "prompt_sensitivity_audit": int(len(prompts)),

        },

    }

    (OUT / "cross_detector_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))



if __name__ == "__main__":

    main()

