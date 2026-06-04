


"""
Step10b: Paired image bootstrap for AP-constrained reliability policies.

This reuses the weighted image-bootstrap machinery from Step9e but selects the
LVIS AP-constrained policies:

- strict AP-constrained policy: selected under AP50/AP drop tolerance <= 0.005;
- best AP-tolerant reliability policy: highest held-out balanced score among
  policies that still meet the final AP50/AP -0.005 success criterion.

Policy selection itself comes from Step10A's calibration-only selected-policy
file and compact fixed test evaluation.
"""


from __future__ import annotations


import argparse

import json

from datetime import datetime

from pathlib import Path

from typing import Dict, List


import numpy as np

import pandas as pd

from tqdm import tqdm


from step8j_yoloworld_lvis_openvoc_baseline import (

    clean_columns,

    load_annotations,

    load_known_classes,

    load_split,

    normalize_image_id,

    safe_class_name,

)

from step9e_weighted_image_bootstrap_lvis300 import (

    operating_from_weighted_counts,

    precompute_ap_outcomes,

    precompute_operating_counts,

    summarize_deltas,

    summarize_samples,

    weighted_ap_summary,

)

from step10a_ap_constrained_reliability_calibration import build_score_columns



DEFAULT_PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PROTOCOL_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step9a_lvis_clear_mini_300_protocol"

DEFAULT_STEP10A_ROOT = DEFAULT_PROJECT_ROOT / "outputs" / "step10a_ap_constrained_reliability_calibration"

DEFAULT_SCORED_CSV = DEFAULT_PROJECT_ROOT / "outputs" / "step9c_geometry_risk_lvis300" / "csv" / "step9c_geometry_scored_candidates.csv"

DEFAULT_OUT = DEFAULT_PROJECT_ROOT / "outputs" / "step10b_ap_constrained_bootstrap"



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    return clean_columns(pd.read_csv(path, encoding="utf-8-sig"))



def choose_methods(compact: pd.DataFrame) -> List[dict]:

    rows: List[dict] = []

    raw = compact[compact["method"].astype(str).eq("step9b_raw_global_reproduced")].copy()

    if raw.empty:

        raise ValueError("Step10A compact metrics missing raw baseline row.")

    r = raw.iloc[0]

    rows.append(

        {

            "method": "step9b_raw_global_reproduced",

            "selection_policy": str(r.get("selection_policy", "raw_step9b_selected_threshold")),

            "score_col": str(r["score_col"]),

            "threshold": float(r["threshold"]),

            "bootstrap_role": "baseline",

        }

    )


    success = compact[compact["meets_success_criterion"].astype(str).str.lower().eq("true")].copy()

    success = success[success["source"].astype(str).eq("step10a_selected_policy")].copy()

    if success.empty:

        raise ValueError("No LVIS AP-constrained success policy found.")


    strict = success[

        (pd.to_numeric(success["AP50_drop_tolerance"], errors="coerce") <= 0.005 + 1e-12)

        & (pd.to_numeric(success["AP_drop_tolerance"], errors="coerce") <= 0.005 + 1e-12)

        & success["objective"].astype(str).eq("balanced_max")

    ].copy()

    if len(strict):

        strict = strict.sort_values(

            ["balanced", "AP50", "AP", "unknown_false_accepts", "background_false_accepts"],

            ascending=[False, False, False, True, True],

        ).iloc[0]

        rows.append(

            {

                "method": str(strict["method"]),

                "selection_policy": "strict_AP_preserving_balanced_max",

                "score_col": str(strict["score_col"]),

                "threshold": float(strict["threshold"]),

                "bootstrap_role": "strict_ap_constrained",

            }

        )


    best = success.sort_values(

        ["balanced", "AP50", "AP", "unknown_false_accepts", "background_false_accepts"],

        ascending=[False, False, False, True, True],

    ).iloc[0]

    rows.append(

        {

            "method": str(best["method"]),

            "selection_policy": "best_AP_tolerant_reliability",

            "score_col": str(best["score_col"]),

            "threshold": float(best["threshold"]),

            "bootstrap_role": "best_ap_tolerant_reliability",

        }

    )


    unique: List[dict] = []

    seen = set()

    for row in rows:

        key = (row["method"], row["score_col"], row["threshold"])

        if key in seen:

            continue

        seen.add(key)

        unique.append(row)

    return unique



def write_markdown(path: Path, summary: pd.DataFrame, delta: pd.DataFrame, methods: List[dict], report: dict) -> None:

    lines = [

        "# LVIS AP-constrained Bootstrap Summary",

        "",

        f"- Output root: `{report['output_root']}`",

        f"- Completed bootstrap samples: `{report['n_boot_completed']} / {report['n_boot_requested']}`",

        "",

        "## Bootstrap Method Specs",

        "",

        pd.DataFrame(methods).to_markdown(index=False),

        "",

        "## Bootstrap Summary",

        "",

        summary.to_markdown(index=False) if len(summary) else "_No summary rows._",

        "",

        "## Delta vs Raw Baseline",

        "",

        delta.to_markdown(index=False) if len(delta) else "_No delta rows._",

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument("--project_root", type=Path, default=DEFAULT_PROJECT_ROOT)

    parser.add_argument("--scored_csv", type=Path, default=DEFAULT_SCORED_CSV)

    parser.add_argument("--step10a_compact_csv", type=Path, default=DEFAULT_STEP10A_ROOT / "csv" / "step10a_compact_paper_metrics.csv")

    parser.add_argument("--split_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_scene_split.csv")

    parser.add_argument("--annotation_dir", type=Path, default=DEFAULT_PROTOCOL_ROOT / "annotations")

    parser.add_argument("--class_map_csv", type=Path, default=DEFAULT_PROTOCOL_ROOT / "csv" / "step9a_class_map.csv")

    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--n_boot", type=int, default=1000)

    parser.add_argument("--max_boot_per_run", type=int, default=0)

    parser.add_argument("--seed", type=int, default=41)

    parser.add_argument("--force_recompute_precompute", action="store_true")

    args = parser.parse_args()


    csv_dir = args.output_root / "csv"

    ensure_dir(csv_dir)

    print("\n========== LVIS AP-constrained Bootstrap ==========")


    compact = read_csv(args.step10a_compact_csv)

    methods = choose_methods(compact)

    pd.DataFrame(methods).to_csv(csv_dir / "step10b_bootstrap_method_specs.csv", index=False, encoding="utf-8-sig")


    known_classes = load_known_classes(args.class_map_csv)

    split_df = load_split(args.split_csv, args.project_root, split_filter="all")

    gt = load_annotations(args.annotation_dir, split_df, known_classes)

    gt_test = gt[gt["split"].astype(str).str.lower().eq("test")].copy()

    image_ids = sorted(split_df[split_df["split"].astype(str).str.lower().eq("test")]["image_id"].map(normalize_image_id).tolist())

    scored0 = read_csv(args.scored_csv)

    scored, _, _ = build_score_columns(scored0)

    scored["image_id"] = scored["image_id"].map(normalize_image_id)

    scored["pred_label"] = scored["pred_label"].map(safe_class_name)

    scored = scored[scored["split"].astype(str).str.lower().eq("test")].copy()


    per_image_path = csv_dir / "step10b_per_image_operating_counts.csv"

    ap_outcomes_path = csv_dir / "step10b_precomputed_ap_detection_outcomes.csv"

    gt_counts_path = csv_dir / "step10b_ap_gt_counts.csv"

    if args.force_recompute_precompute or not (per_image_path.exists() and ap_outcomes_path.exists() and gt_counts_path.exists()):

        per_image = precompute_operating_counts(scored, gt_test, image_ids, methods, iou_thr=0.50)

        det_outcomes, gt_counts = precompute_ap_outcomes(scored, gt_test, known_classes, methods)

        per_image.to_csv(per_image_path, index=False, encoding="utf-8-sig")

        det_outcomes.to_csv(ap_outcomes_path, index=False, encoding="utf-8-sig")

        gt_counts.to_csv(gt_counts_path, index=False, encoding="utf-8-sig")

    else:

        per_image = read_csv(per_image_path)

        det_outcomes = read_csv(ap_outcomes_path)

        gt_counts = read_csv(gt_counts_path)


    samples_path = csv_dir / "step10b_bootstrap_samples.csv"

    existing = read_csv(samples_path) if samples_path.exists() else pd.DataFrame()

    done = set(existing["boot_idx"].astype(int).tolist()) if len(existing) else set()

    remaining = [i for i in range(int(args.n_boot)) if i not in done]

    if args.max_boot_per_run and args.max_boot_per_run > 0:

        remaining = remaining[: int(args.max_boot_per_run)]


    rng_master = np.random.default_rng(int(args.seed))

    seeds = rng_master.integers(0, 2**31 - 1, size=int(args.n_boot), dtype=np.int64)

    new_rows = []

    for boot_idx in tqdm(remaining, desc="Step10B bootstrap samples"):

        rng = np.random.default_rng(int(seeds[boot_idx]))

        sampled = rng.choice(image_ids, size=len(image_ids), replace=True)

        weights: Dict[str, int] = {img: 0 for img in image_ids}

        for img in sampled:

            weights[str(img)] = weights.get(str(img), 0) + 1

        for spec in methods:

            method = spec["method"]

            op = operating_from_weighted_counts(per_image, method, weights)

            ap = weighted_ap_summary(det_outcomes, gt_counts, known_classes, method, weights)

            new_rows.append({"boot_idx": int(boot_idx), "method": method, **op, **ap})

        if len(new_rows) >= 90:

            merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if len(existing) else pd.DataFrame(new_rows)

            merged = merged.drop_duplicates(subset=["boot_idx", "method"], keep="last").sort_values(["boot_idx", "method"])

            merged.to_csv(samples_path, index=False, encoding="utf-8-sig")

            existing = merged

            new_rows = []


    if new_rows:

        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if len(existing) else pd.DataFrame(new_rows)

        merged = merged.drop_duplicates(subset=["boot_idx", "method"], keep="last").sort_values(["boot_idx", "method"])

        merged.to_csv(samples_path, index=False, encoding="utf-8-sig")

        existing = merged


    summary = summarize_samples(existing)

    delta_summary, delta_samples = summarize_deltas(existing, baseline="step9b_raw_global_reproduced")

    summary.to_csv(csv_dir / "step10b_bootstrap_summary.csv", index=False, encoding="utf-8-sig")

    delta_summary.to_csv(csv_dir / "step10b_bootstrap_delta.csv", index=False, encoding="utf-8-sig")

    delta_samples.to_csv(csv_dir / "step10b_bootstrap_delta_samples.csv", index=False, encoding="utf-8-sig")


    report = {

        "method": "LVIS AP-constrained Paired Image Bootstrap",

        "generated_at": datetime.now().isoformat(timespec="seconds"),

        "project_root": str(args.project_root),

        "output_root": str(args.output_root),

        "inputs": {

            "scored_csv": str(args.scored_csv),

            "step10a_compact_csv": str(args.step10a_compact_csv),

            "split_csv": str(args.split_csv),

            "annotation_dir": str(args.annotation_dir),

            "class_map_csv": str(args.class_map_csv),

        },

        "n_boot_requested": int(args.n_boot),

        "n_boot_completed": int(existing["boot_idx"].nunique()) if len(existing) else 0,

        "seed": int(args.seed),

        "baseline_method": "step9b_raw_global_reproduced",

        "methods": methods,

        "outputs": {

            "bootstrap_delta": str(csv_dir / "step10b_bootstrap_delta.csv"),

            "bootstrap_summary": str(csv_dir / "step10b_bootstrap_summary.csv"),

            "bootstrap_samples": str(samples_path),

            "bootstrap_delta_samples": str(csv_dir / "step10b_bootstrap_delta_samples.csv"),

            "method_specs": str(csv_dir / "step10b_bootstrap_method_specs.csv"),

            "markdown_summary": str(args.output_root / "step10b_summary.md"),

            "integrity_report": str(args.output_root / "step10b_integrity_report.json"),

        },

    }

    write_markdown(args.output_root / "step10b_summary.md", summary, delta_summary, methods, report)

    (args.output_root / "step10b_integrity_report.json").write_text(

        json.dumps(report, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print("\n========== Step10B status ==========")

    print(f"Completed bootstrap samples: {report['n_boot_completed']}/{args.n_boot}")

    print(delta_summary.to_string(index=False))



if __name__ == "__main__":

    main()

