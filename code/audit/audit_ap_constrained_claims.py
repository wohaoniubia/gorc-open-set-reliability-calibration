from __future__ import annotations


import json

import math

from pathlib import Path


import pandas as pd



ROOT = Path(__file__).resolve().parents[2]

OUT = ROOT / "outputs" / "gorc_major_revision_round2"

AUDIT_OUT = OUT / "01_metric_audit"

BASELINES = OUT / "03_baselines" / "coco_strong_baselines.csv"

STEP12D_DELTA = ROOT / "outputs" / "step12d_coco_openset_bootstrap" / "csv" / "step12d_bootstrap_delta.csv"

STEP12D_REPORT = ROOT / "outputs" / "step12d_coco_openset_bootstrap" / "step12d_integrity_report.json"



METHOD_MAP = {

    "GORC-RF": "step12c_geometry_max_cal_balanced",

    "GORC-AP-C": "step12c_geometry_ap_constrained_balanced_AP50tol0_APtol0",

}



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def read_csv(path: Path) -> pd.DataFrame:

    if not path.exists():

        raise FileNotFoundError(path)

    df = pd.read_csv(path, encoding="utf-8-sig")

    df.columns = [str(c).strip().lstrip("\ufeff") for c in df.columns]

    return df



def finite(value, default: float = float("nan")) -> float:

    try:

        v = float(value)

    except Exception:

        return default

    return v if math.isfinite(v) else default



def baseline_row(df: pd.DataFrame, method: str) -> pd.Series:

    sub = df[df["method"].astype(str).eq(method)].copy()

    if sub.empty:

        raise ValueError(f"Missing baseline row: {method}")

    return sub.iloc[0]



def exact_point_rows() -> pd.DataFrame:

    baselines = read_csv(BASELINES)

    raw = baseline_row(baselines, "Raw global threshold")

    step12d = read_csv(STEP12D_DELTA) if STEP12D_DELTA.exists() else pd.DataFrame()

    rows = []

    for policy_name, step12d_method in METHOD_MAP.items():

        pol = baseline_row(baselines, policy_name)

        approx = step12d[step12d["method"].astype(str).eq(step12d_method)].copy()

        approx_row = approx.iloc[0] if len(approx) else pd.Series(dtype=object)

        row = {

            "protocol": "COCO-Val-OpenSet-5K",

            "detector": "YOLO-World-l",

            "comparison": f"{policy_name} vs Raw global threshold",

            "policy_method": policy_name,

            "raw_method": "Raw global threshold",

            "score_col": str(pol.get("score_col", "")),

            "threshold": str(pol.get("threshold", "")),

            "exact_full_test_raw_AP50": finite(raw.get("AP50")),

            "exact_full_test_policy_AP50": finite(pol.get("AP50")),

            "exact_full_test_delta_AP50": finite(pol.get("AP50")) - finite(raw.get("AP50")),

            "exact_full_test_raw_AP75": finite(raw.get("AP75")),

            "exact_full_test_policy_AP75": finite(pol.get("AP75")),

            "exact_full_test_delta_AP75": finite(pol.get("AP75")) - finite(raw.get("AP75")),

            "exact_full_test_raw_AP": finite(raw.get("AP")),

            "exact_full_test_policy_AP": finite(pol.get("AP")),

            "exact_full_test_delta_AP": finite(pol.get("AP")) - finite(raw.get("AP")),

            "exact_full_test_raw_B": finite(raw.get("B")),

            "exact_full_test_policy_B": finite(pol.get("B")),

            "exact_full_test_delta_B": finite(pol.get("B")) - finite(raw.get("B")),

            "exact_full_test_raw_UFA": int(finite(raw.get("UFA"), 0)),

            "exact_full_test_policy_UFA": int(finite(pol.get("UFA"), 0)),

            "exact_full_test_delta_UFA": int(finite(pol.get("UFA"), 0)) - int(finite(raw.get("UFA"), 0)),

            "exact_full_test_raw_BGFP": int(finite(raw.get("BG_FP"), 0)),

            "exact_full_test_policy_BGFP": int(finite(pol.get("BG_FP"), 0)),

            "exact_full_test_delta_BGFP": int(finite(pol.get("BG_FP"), 0)) - int(finite(raw.get("BG_FP"), 0)),

            "exact_AP_bootstrap_status": "not_run_exact_ap_is_nonadditive_and_large_scale",

            "exact_AP_bootstrap_reason": "Exact AP bootstrap would require repeated class-wise ranked matching across 4000 held-out images, 20 classes, and 10 IoU thresholds; the existing COCO bootstrap uses a documented per-image AP approximation instead.",

            "approx_bootstrap_source": str(STEP12D_DELTA) if STEP12D_DELTA.exists() else "",

            "approx_bootstrap_n": int(finite(approx_row.get("n_boot", 0), 0)),

            "approx_delta_AP50_mean": finite(approx_row.get("delta_AP50_mean")),

            "approx_delta_AP50_ci025": finite(approx_row.get("delta_AP50_ci025")),

            "approx_delta_AP50_ci975": finite(approx_row.get("delta_AP50_ci975")),

            "approx_delta_AP75_mean": finite(approx_row.get("delta_AP75_mean")),

            "approx_delta_AP75_ci025": finite(approx_row.get("delta_AP75_ci025")),

            "approx_delta_AP75_ci975": finite(approx_row.get("delta_AP75_ci975")),

            "approx_delta_AP_mean": finite(approx_row.get("delta_AP_mean")),

            "approx_delta_AP_ci025": finite(approx_row.get("delta_AP_ci025")),

            "approx_delta_AP_ci975": finite(approx_row.get("delta_AP_ci975")),

            "approx_bootstrap_use": "diagnostic_only_not_authoritative_for_exact_AP_claim",

            "exact_full_test_source": str(BASELINES),

        }

        rows.append(row)

    return pd.DataFrame(rows)



def write_ap_audit_md(path: Path, delta: pd.DataFrame) -> None:

    step12d_report = json.loads(STEP12D_REPORT.read_text(encoding="utf-8")) if STEP12D_REPORT.exists() else {}

    source_lines = [

        "# AP Computation Audit",

        "",

        "## Finding",

        "",

        "COCO AP is computed from all fixed candidate detections after the relevant score transformation, ranked within each known class. The operating acceptance threshold used for B/UFA/BGFP is not applied before AP computation.",

        "",

        "This corresponds to plan option 1 with an important implementation qualifier: the detections are the fixed post-prefilter/post-NMS candidate pool, not an unlimited raw detector stream.",

        "",

        "## Evidence",

        "",

        "- `step8j_yoloworld_lvis_openvoc_baseline.py:446` defines per-class AP by sorting detections by `score` and performing one-to-one matching at a requested IoU threshold.",

        "- `step8j_yoloworld_lvis_openvoc_baseline.py:484` computes AP50, AP75, and AP averaged over IoU thresholds 0.50:0.95 for known classes.",

        "- `step12c_geometry_risk_coco_openset.py:443` creates AP inputs by copying all detections from the requested split and replacing `score` with the policy score column.",

        "- `step12c_geometry_risk_coco_openset.py:580-604` evaluates selected operating points with a threshold for B/UFA/BGFP, while AP is obtained from `exact_ap_for_score` or reused from raw AP for rank-preserving score transforms.",

        "- `step12c_geometry_risk_coco_openset.py:502-538` selects AP-constrained policies using calibration AP and AP50 constraints; AP75 is not used as a selection constraint.",

        "",

        "## Exact Point Estimates",

        "",

        delta[

            [

                "comparison",

                "exact_full_test_delta_AP50",

                "exact_full_test_delta_AP75",

                "exact_full_test_delta_AP",

                "exact_full_test_delta_B",

                "exact_full_test_delta_UFA",

                "exact_full_test_delta_BGFP",

            ]

        ].to_markdown(index=False),

        "",

        "## Bootstrap Status",

        "",

        f"- Existing COCO bootstrap mode: `{step12d_report.get('ap_bootstrap_mode', 'not available')}`.",

        "- Exact AP bootstrap was not rerun because AP is non-additive over images and exact repeated ranked matching at COCO scale would be expensive.",

        "- The CSV `ap_delta_bootstrap.csv` therefore records exact full-test AP point deltas and Step12D's documented approximate AP bootstrap diagnostics separately.",

        "",

        "## Manuscript Wording",

        "",

        "Use: AP-constrained policies are selected using calibration AP and AP50 constraints. AP75 is not a selection constraint; it is reported on held-out data.",

        "",

        "Avoid: reports AP after selection/AP50/AP75 preservation.",

    ]

    path.write_text("\n".join(source_lines) + "\n", encoding="utf-8")



def write_delta_md(path: Path, delta: pd.DataFrame) -> None:

    display_cols = [

        "comparison",

        "exact_full_test_delta_AP50",

        "exact_full_test_delta_AP75",

        "exact_full_test_delta_AP",

        "approx_delta_AP50_mean",

        "approx_delta_AP50_ci025",

        "approx_delta_AP50_ci975",

        "approx_delta_AP75_mean",

        "approx_delta_AP75_ci025",

        "approx_delta_AP75_ci975",

        "approx_delta_AP_mean",

        "approx_delta_AP_ci025",

        "approx_delta_AP_ci975",

        "approx_bootstrap_use",

    ]

    lines = [

        "# AP Delta Bootstrap Audit",

        "",

        "- Exact full-test AP values are authoritative point estimates.",

        "- Approximate bootstrap columns come from Step12D's cached per-image AP approximation and are diagnostic only.",

        "- Exact AP bootstrap was not run for this round.",

        "",

        delta[display_cols].to_markdown(index=False),

    ]

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")



def main() -> None:

    ensure_dir(AUDIT_OUT)

    delta = exact_point_rows()

    csv_path = AUDIT_OUT / "ap_delta_bootstrap.csv"

    delta.to_csv(csv_path, index=False, encoding="utf-8-sig")

    write_delta_md(AUDIT_OUT / "ap_delta_bootstrap.md", delta)

    write_ap_audit_md(AUDIT_OUT / "ap_computation_audit.md", delta)

    manifest = {

        "ap_computation_audit": str(AUDIT_OUT / "ap_computation_audit.md"),

        "ap_delta_bootstrap_csv": str(csv_path),

        "ap_delta_bootstrap_md": str(AUDIT_OUT / "ap_delta_bootstrap.md"),

        "exact_ap_bootstrap_run": False,

        "num_comparisons": int(len(delta)),

    }

    (AUDIT_OUT / "ap_claim_audit_manifest.json").write_text(

        json.dumps(manifest, ensure_ascii=False, indent=2),

        encoding="utf-8",

    )

    print(json.dumps(manifest, ensure_ascii=False, indent=2))



if __name__ == "__main__":

    main()

