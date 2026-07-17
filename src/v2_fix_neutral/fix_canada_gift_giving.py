"""
Fix neutral rows where Country == 'canada' and Subaxis == 'gift_giving'.

Overwrites Background with canonical Canada Gift Giving text and regenerates
Rule-of-Thumb and Value with gpt-4o.

Post-swap schema:
  - Country / Background             -> Canada (ROT/Value culture)
  - Other Country / Other Background -> story culture
"""

import argparse
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

DEFAULT_MODEL = "gpt-4o"
DEFAULT_WORKERS = 16

CANADA_GIFT_GIVING_BACKGROUND = """### Gift Giving
- Gifts are usually only given on special occasions and are almost always accompanied with a card.
- People tend to open gifts in front of the giver, either upon receive them or later along with other presents.
- For occasions that require a gift (e.g. birthday, wedding, baby shower), a modest value of about 5 is acceptable unless you know the recipient very well.
- It is distasteful to give cash or money as a present, however gift cards are okay if the shop they are for holds a specific significance to the recipient.
- Gifts that are given as a personal gesture outside of special occasions are often grander or more heartfelt. For example, to reflect deep gratitude for a favour someone has done for you, you may give them sports tickets or take them to an expensive restaurant.
- Token gifts may be given when visiting a house (e.g. wine, chocolate).
- In Quebec, flowers are commonly sent to the host before holding dinner parties. Expensive wine is a good gift for this occasion as well.
"""

TASK_PROMPT = """Task: Generate a Rule-of-Thumb and Value for a neutral-label example.

The Rule-of-Thumb and Value should be grounded in Country 1's cultural background.
The story is based on Country 2's cultural background and should not be judged using Country 1's norms.

Story (based on Country 2):
{story}

Country 1:
{country}

Cultural background for Country 1:
{background}

Country 2:
{other_country}

Cultural background for Country 2:
{other_background}

Write a Rule-of-Thumb and Value that reflect Country 1's gift-giving norms only.
The story follows Country 2's norms; do not use Country 2's norms in the Rule-of-Thumb or Value.
Do not include any country names.

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


def _process_one(
    idx: int,
    row: pd.Series,
    client: OpenAI,
    model: str,
    temperature: float,
) -> Tuple[int, Dict]:
    prompt = TASK_PROMPT.format(
        story=row["Story"],
        country=row["Country"],
        background=row["Background"],
        other_country=row["Other Country"],
        other_background=row["Other Background"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    new_rot, new_value = parse_rot_value(content)
    return idx, {
        "ID": row["ID"],
        "Other Country": row["Other Country"],
        "old_Rule-of-Thumb": row["Rule-of-Thumb"],
        "new_Rule-of-Thumb": new_rot,
        "old_Value": row["Value"],
        "new_Value": new_value,
    }


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_input = os.path.join(
        repo_root, "output/v2_fix_neutral/normad_etiquette_final_data_neutral_swapped.csv"
    )
    default_output = default_input
    default_log = os.path.join(
        repo_root, "output/v2_fix_neutral/canada_gift_giving_fix_log.csv"
    )
    default_key = os.path.join(repo_root, "key/key.txt")

    parser = argparse.ArgumentParser(description="Fix Canada gift_giving neutral rows.")
    parser.add_argument("--input_path", type=str, default=default_input)
    parser.add_argument("--output_path", type=str, default=default_output)
    parser.add_argument("--log_path", type=str, default=default_log)
    parser.add_argument("--api_key_path", type=str, default=default_key)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--ids",
        type=str,
        default="",
        help="Comma-separated IDs to fix (default: all matching rows).",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_path).fillna("")

    target_mask = (
        (df["Gold Label"].astype(str).str.strip().str.lower() == "neutral")
        & (df["Country"] == "canada")
        & (df["Subaxis"] == "gift_giving")
    )
    if args.ids.strip():
        id_set = {x.strip() for x in args.ids.split(",") if x.strip()}
        target_mask &= df["ID"].astype(str).isin(id_set)
    target_indices = df.index[target_mask].tolist()

    if not target_indices:
        print("No matching rows found (neutral, Country=canada, Subaxis=gift_giving).")
        return

    df.loc[target_mask, "Background"] = CANADA_GIFT_GIVING_BACKGROUND

    client = OpenAI(api_key=load_api_key(args.api_key_path))
    log_rows: List[Dict] = []
    workers = max(1, args.workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_one, idx, df.loc[idx], client, args.model, args.temperature
            ): idx
            for idx in target_indices
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Fixing Canada gift_giving"):
            idx, log_row = fut.result()
            df.loc[idx, "Rule-of-Thumb"] = log_row["new_Rule-of-Thumb"]
            df.loc[idx, "Value"] = log_row["new_Value"]
            if "model" in df.columns:
                df.loc[idx, "model"] = args.model
            log_rows.append(log_row)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df.to_csv(args.output_path, index=False)
    pd.DataFrame(log_rows).to_csv(args.log_path, index=False)

    print(f"Input:  {args.input_path}")
    print(f"Output: {args.output_path}")
    print(f"Log:    {args.log_path}")
    print(f"Workers: {workers}")
    print(f"Rows fixed: {len(target_indices)}")


if __name__ == "__main__":
    main()
