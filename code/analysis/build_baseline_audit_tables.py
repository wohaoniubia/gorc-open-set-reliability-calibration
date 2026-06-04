


"""Generate round3 compact main Table 3 and full supplementary COCO baselines."""


from __future__ import annotations


from pathlib import Path


import numpy as np

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SRC = ROOT / "outputs" / "gorc_major_revision_round2" / "09_tables_for_paper" / "table_coco_strong_baselines_full.csv"

OUT = ROOT / "outputs" / "gorc_major_revision_round3" / "04_tables"



MAIN_ORDER = [

    ("Raw global threshold", None),

    ("Raw per-class threshold", "merged_per_class"),

    ("Score-only Platt/logistic", None),

    ("Temperature-scaled score logit", None),

    ("Isotonic score calibration", None),

    ("Class-wise score scaling", None),

    ("Score + class logistic", None),

    ("Score + geometry logistic", None),

    ("Score + class + geometry logistic", None),

    ("GORC-RF", None),

    ("GORC-AP-C", None),

]



def ensure_dir(path: Path) -> None:

    path.mkdir(parents=True, exist_ok=True)



def finite_float(value):

    try:

        v = float(value)

    except Exception:

        return np.nan

    return v if np.isfinite(v) else np.nan



def fmt_float(value: float, digits: int = 4) -> str:

    v = finite_float(value)

    if np.isnan(v):

        return ""

    return f"{v:.{digits}f}"



def fmt_threshold(value) -> str:

    text = str(value)

    if text.startswith("{"):

        return "per-class"

    v = finite_float(value)

    return "" if np.isnan(v) else f"{v:g}"



def row_to_display(row: pd.Series, method: str | None = None, selection: str | None = None, include_ap75: bool = False, include_cal: bool = False) -> dict:

    out = {

        "Method": method or str(row["method"]),

        "Selection": selection or str(row["selection_mode"]),

        "Thr.": fmt_threshold(row.get("threshold", "")),

        "B": fmt_float(row.get("B")),

        "P": fmt_float(row.get("P")),

        "R": fmt_float(row.get("R")),

        "URR": fmt_float(row.get("URR")),

        "UFA": int(float(row.get("UFA", 0))) if str(row.get("UFA", "")).strip() else "",

        "BG FP": int(float(row.get("BG_FP", 0))) if str(row.get("BG_FP", "")).strip() else "",

        "AP50": fmt_float(row.get("AP50")),

        "AP": fmt_float(row.get("AP")),

    }

    if include_ap75:

        out["AP75"] = fmt_float(row.get("AP75"))

    if include_cal:

        out["cal B"] = fmt_float(row.get("cal_B"))

        out["cal AP"] = fmt_float(row.get("cal_AP"))

    return out



def same_metrics(a: pd.Series, b: pd.Series) -> bool:

    cols = ["B", "P", "R", "URR", "UFA", "BG_FP", "AP50", "AP75", "AP", "cal_B", "cal_AP"]

    for c in cols:

        av = finite_float(a.get(c))

        bv = finite_float(b.get(c))

        if np.isnan(av) and np.isnan(bv):

            continue

        if abs(av - bv) > 1e-12:

            return False

    return True



def main() -> None:

    ensure_dir(OUT)

    df = pd.read_csv(SRC, encoding="utf-8-sig")

    df["method"] = df["method"].astype(str)

    df["selection_mode"] = df["selection_mode"].astype(str)


    main_rows: list[dict] = []

    per_class_note = ""

    for method, special in MAIN_ORDER:

        if special == "merged_per_class":

            sub = df[df["method"].eq(method)].copy()

            if len(sub) < 1:

                raise ValueError("Missing Raw per-class threshold rows")

            if len(sub) >= 2 and same_metrics(sub.iloc[0], sub.iloc[1]):

                main_rows.append(

                    row_to_display(

                        sub.iloc[0],

                        method="Raw per-class threshold",

                        selection="RF/AP-C per-class threshold",

                    )

                )

                per_class_note = (

                    "The RF and AP-C per-class-threshold selections are identical in this run; "

                    "the AP-C constraint did not change the selected per-class-threshold row."

                )

            else:

                for _, r in sub.iterrows():

                    main_rows.append(row_to_display(r))

            continue

        sub = df[df["method"].eq(method)]

        if sub.empty:

            raise ValueError(f"Missing method: {method}")

        main_rows.append(row_to_display(sub.iloc[0]))


    main_table = pd.DataFrame(main_rows)

    main_path = OUT / "table3_coco_baselines_main_compact.md"

    lines = [

        "# Table 3. COCO baseline comparison (main compact)",

        "",

        main_table.to_markdown(index=False),

        "",

        "Note. All model fitting and operating-point selection use the calibration split only. AP75 is reported in the supplementary full table, not used as a selection constraint.",

    ]

    if per_class_note:

        lines.append(f"Note. {per_class_note}")

    main_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    main_table.to_csv(OUT / "table3_coco_baselines_main_compact.csv", index=False, encoding="utf-8-sig")


    full_rows = [

        row_to_display(row, include_ap75=True, include_cal=True)

        | {

            "Group": str(row.get("group", "")),

            "Score": str(row.get("score_col", "")),

            "Hyperparameters": str(row.get("hyperparameters", "")) if str(row.get("hyperparameters", "")).lower() != "nan" else "",

        }

        for _, row in df.iterrows()

    ]


    full = pd.DataFrame(full_rows)

    first = ["Group", "Method", "Selection", "Score", "Thr.", "B", "P", "R", "URR", "UFA", "BG FP", "AP50", "AP75", "AP", "cal B", "cal AP", "Hyperparameters"]

    full = full[[c for c in first if c in full.columns]]

    supp_path = OUT / "tableS_coco_baselines_full.md"

    supp_lines = [

        "# Supplementary Table S. Full COCO baseline comparison",

        "",

        full.to_markdown(index=False),

        "",

        "Note. This full supplementary table retains histogram/quantile binning, AP75, calibration-split B, and calibration-split AP. AP75 is reported diagnostically and was not a policy-selection constraint.",

    ]

    supp_path.write_text("\n".join(supp_lines) + "\n", encoding="utf-8")

    full.to_csv(OUT / "tableS_coco_baselines_full.csv", index=False, encoding="utf-8-sig")


    manifest = {

        "source": str(SRC),

        "main": str(main_path),

        "supplementary": str(supp_path),

        "main_rows": len(main_table),

        "supplementary_rows": len(full),

        "per_class_note": per_class_note,

        "histogram_in_supplementary": bool(df["method"].str.contains("Histogram", case=False, regex=False).any()),

    }

    pd.Series(manifest).to_json(OUT / "table_generation_manifest.json", force_ascii=False, indent=2)

    print(f"Wrote {main_path}")

    print(f"Wrote {supp_path}")

    print(pd.Series(manifest).to_string())



if __name__ == "__main__":

    main()

