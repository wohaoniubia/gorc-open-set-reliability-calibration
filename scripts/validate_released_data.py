from __future__ import annotations

from pathlib import Path

import pandas as pd


RELEASE_ROOT = Path(__file__).resolve().parents[1]
FIGURE_SOURCE = RELEASE_ROOT / "data" / "figure_source"
TABLE_SOURCE = RELEASE_ROOT / "data" / "table_source"
SUPPLEMENTARY_SOURCE = RELEASE_ROOT / "data" / "supplementary_source"
FIGURES = RELEASE_ROOT / "figures"

REQUIRED_FILES = [
    "README.md",
    "FINAL_RELEASE_TREE.md",
    "data/cache_metadata/omitted_resources.md",
    "data/figure_source/fig03_coco_operating_landscape.csv",
    "data/figure_source/fig04_reliability_score_bins.csv",
    "data/figure_source/fig05_preference_scores.csv",
    "data/figure_source/fig06_geometry_group_ablation.csv",
    "data/figure_source/fig07_bootstrap_ci.csv",
    "data/figure_source/fig07_calibration_budget.csv",
    "data/figure_source/fig07_repeated_split_false_accept_changes.csv",
    "data/figure_source/fig08_prompt_stress.csv",
    "data/figure_source/fig09_qualitative_cases_manifest.csv",
    "data/supplementary_source/s_policy_grid.csv",
    "data/supplementary_source/s_repeated_split_exact_ap.csv",
    "data/table_source/table01_validation_protocols.csv",
    "data/table_source/table02_main_coco_operating_points.csv",
    "data/table_source/table03_coco_baseline_ablation_audit.csv",
    "data/table_source/table04_computational_footprint.csv",
    "data/table_source/table05_selected_policy_apc_card.csv",
    "data/table_source/table06_lvis_grounding_dino_diagnostics.csv",
]

SLASH = "/"
BACKSLASH = chr(92)

REMOVED_PATHS = [
    "metadata",
    SLASH.join(["figures", "generated"]),
    SLASH.join(["code", "figure_generation"]),
    "WORK" + "_SUMMARY_20260531.md",
]

TEXT_MARKERS = [
    "<" + "SET_LOCAL_PROJECT_ROOT" + ">",
    "E:" + BACKSLASH,
    "C:" + BACKSLASH,
    "Users" + BACKSLASH,
    "PHD" + "Work",
    "amir" + "_start",
    "Path requires " + "scrubbing",
    "tool " + "timeout",
    SLASH.join(["figures", "generated"]),
    SLASH.join(["code", "figure_generation"]),
    "WORK" + "_SUMMARY",
]

DISALLOWED_SUFFIXES = {
    "." + "doc",
    "." + "docx",
    ".pt",
    ".pth",
    ".onnx",
    ".ckpt",
    ".pkl",
    ".pickle",
    ".npy",
    ".npz",
    ".h5",
    ".hdf5",
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".gz",
}


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def require_columns(df: pd.DataFrame, columns: list[str], label: str) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def active_text_files() -> list[Path]:
    suffixes = {".csv", ".md", ".py", ".txt"}
    return [
        path
        for path in RELEASE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in suffixes
    ]


def check_required_files() -> None:
    missing = [rel for rel in REQUIRED_FILES if not (RELEASE_ROOT / rel).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required release files: {missing}")
    for rel in REMOVED_PATHS:
        if (RELEASE_ROOT / rel).exists():
            raise ValueError(f"Removed working-package path is still present: {rel}")


def check_figures() -> None:
    expected = {f"Fig{i}.jpg" for i in range(1, 10)}
    actual = {path.name for path in FIGURES.iterdir() if path.is_file()}
    if actual != expected:
        raise ValueError(f"Unexpected figure set: {sorted(actual)}")
    extra_rasters = [
        path.name
        for path in FIGURES.iterdir()
        if path.is_file() and path.suffix.lower() in {".png", ".svg", ".pdf", ".tif", ".tiff"}
    ]
    if extra_rasters:
        raise ValueError(f"Unexpected non-JPG figure files: {extra_rasters}")


def check_table_consistency() -> None:
    table2 = read_csv(TABLE_SOURCE / "table02_main_coco_operating_points.csv")
    fig3 = read_csv(FIGURE_SOURCE / "fig03_coco_operating_landscape.csv")
    for method in ["Raw global threshold", "GORC-SCG", "GORC-RF", "GORC-AP-C"]:
        trow = table2[table2["method"].eq(method)].iloc[0]
        frow = fig3[fig3["method"].eq(method)].iloc[0]
        for table_col, fig_col in [
            ("B", "B"),
            ("R", "R"),
            ("coverage", "coverage"),
            ("UFA", "UFA"),
            ("BGFP", "BG_FP"),
            ("AP", "AP"),
        ]:
            if abs(float(trow[table_col]) - float(frow[fig_col])) > 1e-9:
                raise ValueError(f"Fig. 3/Table 2 mismatch: {method} {table_col}")

    fig6 = read_csv(FIGURE_SOURCE / "fig06_geometry_group_ablation.csv")
    scg = fig6[fig6["variant"].eq("GORC-SCG")].iloc[0]
    rounded = {
        "B": round(float(scg["B"]), 4),
        "UFA": int(scg["UFA"]),
        "BG_FP": int(scg["BG_FP"]),
        "AP": round(float(scg["AP"]), 4),
    }
    expected = {"B": 0.7748, "UFA": 114, "BG_FP": 2333, "AP": 0.4848}
    if rounded != expected:
        raise ValueError(f"Fig. 6 GORC-SCG row mismatch: {rounded} != {expected}")

    fig7 = read_csv(FIGURE_SOURCE / "fig07_repeated_split_false_accept_changes.csv")
    require_columns(
        fig7,
        ["seed", "delta_UFA_GORC_RF_minus_raw", "delta_BG_FP_GORC_RF_minus_raw"],
        "Fig. 7 false-accept source",
    )

    prompt = read_csv(SUPPLEMENTARY_SOURCE / "s_prompt_stress.csv")
    require_columns(
        prompt,
        [
            "detector",
            "prompt_variant",
            "candidate_count",
            "raw_B",
            "RF_B",
            "raw_UFA",
            "RF_UFA",
            "raw_BGFP",
            "RF_BGFP",
        ],
        "Prompt-stress source",
    )


def check_text_markers() -> None:
    hits: list[str] = []
    for path in active_text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for marker in TEXT_MARKERS:
            if marker in text:
                hits.append(f"{path.relative_to(RELEASE_ROOT)}: {marker}")
    if hits:
        raise ValueError("Local or working-package markers found:\n" + "\n".join(hits[:50]))


def check_disallowed_files() -> None:
    hits = [
        path.relative_to(RELEASE_ROOT).as_posix()
        for path in RELEASE_ROOT.rglob("*")
        if path.is_file() and path.suffix.lower() in DISALLOWED_SUFFIXES
    ]
    if hits:
        raise ValueError(f"Disallowed file types found: {hits}")


def main() -> None:
    check_required_files()
    check_figures()
    check_table_consistency()
    check_text_markers()
    check_disallowed_files()
    print("Release data validation passed.")


if __name__ == "__main__":
    main()
