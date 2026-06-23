"""
Regenerate explanations for neutral rows.

The original few-shot explanations argue yes/no under the story culture instead of
justifying why the label is neutral. This script rewrites them to explain why the
given Rule-of-Thumb and Value cannot determine whether the story is socially
acceptable.

Post-swap schema:
  - Country / Background        -> context culture (ROT/Value culture)
  - Other Country / Other Background -> story culture
"""

import argparse
import os
import random
import re
import time

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = "gpt-4o"

TASK_PROMPT = """Task: Write an explanation for why the gold label for this example is Neutral (neither Yes nor No).

The story reflects norms from Country 2. The Rule-of-Thumb and Value reflect norms from Country 1. Because these come from different cultures, the Rule-of-Thumb and Value cannot be used to judge whether the story action is socially acceptable.

Country 1 (context culture for the Rule-of-Thumb and Value): {country}

Rule-of-Thumb for Country 1: {rot}

Value for Country 1: {value}

Cultural background for Country 1:
{background}

Story: {story}

Country 2 (story culture):
{other_country}

Cultural background for Country 2:
{other_background}

Write 1-3 sentences that:

1. Unless otherwise, explain why the answer is neutral -- because the provided Rule-of-Thumb, Value, and background of {country} are irrelevant to judging the story.
2. Don't mention {other_country} or {other_background}.
3. If the answer is obviously not neutral (i.e. story is relevant to the rule of thumb, value, or background of {country}), just say the word "WRONG".

Format your response exactly as:
Explanation: <text>
"""

FILTER_CHOICES = ("entail_good", "entail_bad", "all")


def load_api_key(key_path: str) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    with open(key_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def is_neutral(row: pd.Series) -> bool:
    return str(row.get("Gold Label", "")).strip().lower() == "neutral"


def parse_explanation(response: str) -> str:
    match = re.search(r"^Explanation:\s*(.+)$", response, flags=re.MULTILINE | re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def should_regenerate_row(row: pd.Series, row_filter: str) -> bool:
    if not is_neutral(row):
        return False
    entail = str(row.get("entail_pred_temp0.0", "")).strip()
    if row_filter == "all":
        return True
    if row_filter == "entail_good":
        return entail == "1.0"
    if row_filter == "entail_bad":
        return entail == "0.0"
    raise ValueError(f"Unknown filter: {row_filter}")


def build_prompt(row: pd.Series) -> str:
    return TASK_PROMPT.format(
        story=row["Story"],
        rot=row["Rule-of-Thumb"],
        value=row["Value"],
        country=row["Country"],
        background=row["Background"],
        other_country=row["Other Country"],
        other_background=row["Other Background"],
    )


def get_regen_indices(df: pd.DataFrame, row_filter: str, sample_size: int = 0) -> list:
    regen_mask = df.apply(lambda row: should_regenerate_row(row, row_filter), axis=1)
    regen_indices = df.index[regen_mask].tolist()
    if sample_size > 0:
        n = min(sample_size, len(regen_indices))
        regen_indices = random.sample(regen_indices, n)
    return regen_indices


def regenerate_explanations(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    temperature: float,
    regen_indices: list,
    sleep_every: int = 100,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()

    log_rows = []
    for n, idx in enumerate(tqdm(regen_indices, desc="Regenerating explanations")):
        if sleep_every and n > 0 and n % sleep_every == 0:
            time.sleep(1)

        row = df.loc[idx]
        prompt = build_prompt(row)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=temperature
        )
        content = response.choices[0].message.content or ""
        new_explanation = parse_explanation(content)

        log_rows.append(
            {
                "ID": row["ID"],
                "old_Explanation": row["Explanation"],
                "new_Explanation": new_explanation,
            }
        )

        df.loc[idx, "Explanation"] = new_explanation
        if "model" in df.columns:
            df.loc[idx, "model"] = model

    return df, pd.DataFrame(log_rows)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_input = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_neutral_swapped.csv",
    )
    default_output = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_explanations_regenerated.csv",
    )
    default_log = os.path.join(
        repo_root,
        "output/v2_fix_neutral/explanation_regeneration_log.csv",
    )
    default_key = os.path.join(repo_root, "key/key.txt")

    parser = argparse.ArgumentParser(
        description="Regenerate explanations for neutral rows."
    )
    parser.add_argument("--input_path", type=str, default=default_input)
    parser.add_argument("--output_path", type=str, default=default_output)
    parser.add_argument("--log_path", type=str, default=default_log)
    parser.add_argument("--api_key_path", type=str, default=default_key)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--save_only_sample",
        action="store_true",
        help="If set, write only the sampled/edited rows to --output_path (instead of the full dataset).",
    )
    parser.add_argument(
        "--sample_size",
        type=int,
        default=25,
        help="Sample this many rows after filtering (0 = all matching rows).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    parser.add_argument(
        "--filter",
        choices=FILTER_CHOICES,
        default="entail_good",
        help="Which neutral rows to regenerate (default: entail_good = entail_pred 1.0).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    df = pd.read_csv(args.input_path).fillna("")
    eligible = df.apply(lambda row: should_regenerate_row(row, args.filter), axis=1).sum()
    regen_indices = get_regen_indices(df, args.filter, sample_size=args.sample_size)
    if not regen_indices:
        print(f"No rows to regenerate for filter={args.filter!r}.")
        return

    client = OpenAI(api_key=load_api_key(args.api_key_path))
    df_out, log_df = regenerate_explanations(
        df,
        client=client,
        model=args.model,
        temperature=args.temperature,
        regen_indices=regen_indices,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.save_only_sample:
        df_out.loc[regen_indices].to_csv(args.output_path, index=False)
    else:
        df_out.to_csv(args.output_path, index=False)
    log_df.to_csv(args.log_path, index=False)

    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Log:    {args.log_path}")
    print(f"Model:  {args.model}")
    print(f"Filter: {args.filter}")
    print(f"Eligible rows: {eligible}")
    print(f"Regenerated explanations for {len(log_df)} rows.")


if __name__ == "__main__":
    main()
