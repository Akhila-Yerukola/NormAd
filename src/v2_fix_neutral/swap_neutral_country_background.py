"""
Swap Country/Background with Other Country/Other Background for neutral rows.

For neutral examples, published context (Country, Background, Value, Rule-of-Thumb)
should describe the ROT culture (C2), while the story stays from the story
culture (C1). The swapped story-side fields are preserved in Other Country /
Other Background for auditing.
"""

import argparse
import os

import pandas as pd


def swap_neutral_country_background(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    neutral_mask = df["Gold Label"].astype(str).str.strip().str.lower() == "neutral"

    missing_other = neutral_mask & (
        df["Other Country"].isna()
        | df["Other Background"].isna()
        | (df["Other Country"].astype(str).str.strip() == "")
        | (df["Other Background"].astype(str).str.strip() == "")
    )
    if missing_other.any():
        missing_ids = df.loc[missing_other, "ID"].tolist()
        raise ValueError(
            f"{missing_other.sum()} neutral row(s) missing Other Country/Other Background: {missing_ids[:10]}"
        )

    country = df.loc[neutral_mask, "Country"].copy()
    background = df.loc[neutral_mask, "Background"].copy()

    df.loc[neutral_mask, "Country"] = df.loc[neutral_mask, "Other Country"]
    df.loc[neutral_mask, "Background"] = df.loc[neutral_mask, "Other Background"]
    df.loc[neutral_mask, "Other Country"] = country.values
    df.loc[neutral_mask, "Other Background"] = background.values

    return df


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_input = os.path.join(
        repo_root,
        "data_and_heval/datasets.zip.enc.extracted/normad_etiquette_final_data.csv",
    )
    default_output = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_neutral_swapped.csv",
    )

    parser = argparse.ArgumentParser(
        description="Swap Country/Background with Other Country/Other Background for neutral rows."
    )
    parser.add_argument("--input_path", type=str, default=default_input)
    parser.add_argument("--output_path", type=str, default=default_output)
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)
    neutral_count = (df["Gold Label"].astype(str).str.strip().str.lower() == "neutral").sum()

    df_out = swap_neutral_country_background(df)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_out.to_csv(args.output_path, index=False)

    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Swapped Country/Background for {neutral_count} neutral rows.")


if __name__ == "__main__":
    main()
