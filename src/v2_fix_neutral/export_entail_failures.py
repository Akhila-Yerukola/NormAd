"""
Export neutral rows where entail_pred_temp0.0 == 0.0 for manual audit.

These are neutrals whose Rule-of-Thumb was judged to entail the pre-swap
story-side Background (C1 leakage).
"""

import argparse
import os

import pandas as pd

AUDIT_COLUMNS = [
    "ID",
    "Country",
    "Other Country",
    "Background",
    "Other Background",
    "Axis",
    "Subaxis",
    "Value",
    "Rule-of-Thumb",
    "Story",
    "Explanation",
    "Gold Label",
    "entail_pred_temp0.0",
    "model",
]


def export_entail_failures(df: pd.DataFrame) -> pd.DataFrame:
    neutral_mask = df["Gold Label"].astype(str).str.strip().str.lower() == "neutral"
    entail_mask = df["entail_pred_temp0.0"].astype(str).str.strip() == "0.0"
    subset = df.loc[neutral_mask & entail_mask].copy()

    columns = [c for c in AUDIT_COLUMNS if c in subset.columns]
    return subset[columns]


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_input = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_neutral_swapped.csv",
    )
    default_output = os.path.join(
        repo_root,
        "output/v2_fix_neutral/neutral_entail_failures_audit.csv",
    )

    parser = argparse.ArgumentParser(
        description="Export neutral rows with entail_pred_temp0.0 == 0.0 for manual audit."
    )
    parser.add_argument("--input_path", type=str, default=default_input)
    parser.add_argument("--output_path", type=str, default=default_output)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    audit_df = export_entail_failures(df)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    audit_df.to_csv(args.output_path, index=False)

    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Exported {len(audit_df)} neutral rows with entail_pred_temp0.0 == 0.0.")


if __name__ == "__main__":
    main()
