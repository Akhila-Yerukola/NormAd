"""
Regenerate Value and Rule-of-Thumb for leaky neutral rows (entail_pred_temp0.0 == 0.0).

Adapted from src/story_collection/run_model_validation_stage2_fix_rot.py for the
post-swap schema:
  - Country / Background        -> context culture (C2; ROT/Value should live here)
  - Other Country / Other Background -> story culture (C1; story should live here)

Uses gpt-4o by default.
"""

import argparse
import os
import random
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = "gpt-4o"
DEFAULT_WORKERS = 16

TASK_PROMPT = """Task: You will be given a story, a rule-of-thumb, and an abstracted value.

The story reflects norms from Country 2. Regenerate the Rule-of-Thumb and Value so they are clearly grounded in Country 1's cultural background and cannot be used to judge whether the story action is socially acceptable (the answer should be Neither, not Yes or No).

Story: {story}

Current Rule-of-thumb: {rot}

Current Value: {value}

Country 1 (context culture — anchor the new Rule-of-Thumb and Value here):
{country}

Cultural background for Country 1:
{background}

Country 2 (story culture — do NOT anchor the new Rule-of-Thumb or Value here):
{other_country}

Cultural background for Country 2:
{other_background}

Generate a new Rule-of-Thumb and Value that:
1. Are entailed by the Country 1 cultural background.
2. Are irrelevant to the Country 2 cultural background and the story.
3. Do not let someone answer Yes or No about the story's social acceptability using only the new Rule-of-Thumb and Value.
4. Do not include any country names.

Format your response exactly as:
Rule-of-Thumb: <text>
Value: <text>
"""


def load_api_key(key_path: str) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    with open(key_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def parse_rot_value(response: str) -> Tuple[str, str]:
    rot_match = re.search(r"^Rule-of-Thumb:\s*(.+)$", response, flags=re.MULTILINE | re.IGNORECASE)
    value_match = re.search(r"^Value:\s*(.+)$", response, flags=re.MULTILINE | re.IGNORECASE)
    if not rot_match or not value_match:
        raise ValueError(f"Could not parse model response:\n{response}")
    return rot_match.group(1).strip(), value_match.group(1).strip()


def should_fix_row(row: pd.Series) -> bool:
    if str(row.get("Gold Label", "")).strip().lower() != "neutral":
        return False
    return str(row.get("entail_pred_temp0.0", "")).strip() == "0.0"


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


def _process_one(
    idx: int,
    row: pd.Series,
    client: OpenAI,
    model: str,
    temperature: float,
) -> Tuple[int, Dict]:
    prompt = build_prompt(row)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=temperature,
    )
    content = response.choices[0].message.content or ""
    new_rot, new_value = parse_rot_value(content)
    return idx, {
        "ID": row["ID"],
        "old_Rule-of-Thumb": row["Rule-of-Thumb"],
        "new_Rule-of-Thumb": new_rot,
        "old_Value": row["Value"],
        "new_Value": new_value,
    }


def regenerate_rot_value(
    df: pd.DataFrame,
    client: OpenAI,
    model: str,
    temperature: float,
    sample_size: int = 0,
    workers: int = DEFAULT_WORKERS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    fix_mask = df.apply(should_fix_row, axis=1)
    fix_indices = df.index[fix_mask].tolist()
    if sample_size > 0:
        fix_indices = random.sample(fix_indices, min(sample_size, len(fix_indices)))

    log_rows: List[Dict] = []
    workers = max(1, workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_process_one, idx, df.loc[idx], client, model, temperature): idx
            for idx in fix_indices
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Regenerating ROT/Value"):
            idx, log_row = fut.result()
            df.loc[idx, "Rule-of-Thumb"] = log_row["new_Rule-of-Thumb"]
            df.loc[idx, "Value"] = log_row["new_Value"]
            if "model" in df.columns:
                df.loc[idx, "model"] = model
            log_rows.append(log_row)

    return df, pd.DataFrame(log_rows)


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_input = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_neutral_swapped.csv",
    )
    default_output = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_rot_regenerated.csv",
    )
    default_log = os.path.join(
        repo_root,
        "output/v2_fix_neutral/rot_value_regeneration_log.csv",
    )
    default_key = os.path.join(repo_root, "key/key.txt")

    parser = argparse.ArgumentParser(
        description="Regenerate Value/Rule-of-Thumb for neutral rows with entail_pred_temp0.0 == 0.0."
    )
    parser.add_argument("--input_path", type=str, default=default_input)
    parser.add_argument("--output_path", type=str, default=default_output)
    parser.add_argument("--log_path", type=str, default=default_log)
    parser.add_argument("--api_key_path", type=str, default=default_key)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--sample_size", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--seed", type=int, default=None, help="Random seed for sampling.")
    parser.add_argument(
        "--save_only_sample",
        action="store_true",
        help="If set, write only the sampled/edited rows to --output_path (instead of the full dataset).",
    )
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    df = pd.read_csv(args.input_path).fillna("")

    n_fix = df.apply(should_fix_row, axis=1).sum()
    if n_fix == 0:
        print("No rows to fix (neutral with entail_pred_temp0.0 == 0.0).")
        return

    client = OpenAI(api_key=load_api_key(args.api_key_path))
    df_out, log_df = regenerate_rot_value(
        df,
        client=client,
        model=args.model,
        temperature=args.temperature,
        sample_size=args.sample_size,
        workers=args.workers,
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.save_only_sample:
        df_out[df_out["ID"].isin(log_df["ID"])].to_csv(args.output_path, index=False)
    else:
        df_out.to_csv(args.output_path, index=False)
    log_df.to_csv(args.log_path, index=False)

    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Log:    {args.log_path}")
    print(f"Model:  {args.model}")
    print(f"Workers: {args.workers}")
    print(f"Regenerated ROT/Value for {len(log_df)} rows.")


if __name__ == "__main__":
    main()
