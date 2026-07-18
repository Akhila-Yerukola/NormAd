"""
Validate neutral stories against Background and reject (then regenerate) if related.

Background / Country / Value / Rule-of-Thumb are never modified.

Post-swap schema:
  - Country / Background             -> context culture (must NOT relate to story)
  - Other Country / Other Background -> story culture (story must be grounded here)

Flow:
  1. Ask: given ONLY Background, is the answer Yes/No/Neutral?
     Reject if Yes or No (Background alone determines the label).
  2. For rejected rows, regenerate Story (+ Explanation) from Other Background
  3. Re-validate; keep regenerating until Neutral or max retries
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

RELEVANCE_QUERY = (
    "Given ONLY this cultural background, answer the story question with exactly one word: "
    "Yes, No, or Neutral.\n"
    "- Yes / No: ONLY if a background bullet specifically covers the story's decisive action "
    "(the behavior being judged), tightly enough that a reader could confidently answer Yes or No.\n"
    "- Neutral: if the background is merely the same topic, only covers a secondary detail, "
    "is vague (e.g. 'show respect'), or leaves the decisive action unaddressed.\n"
    "- For multi-part stories, answer Yes/No only if the background determines the overall "
    "acceptability of what was done — not because one incidental detail matches.\n"
    "When unsure, answer Neutral. Prefer Neutral over Yes/No."
)

RELEVANCE_FEWSHOT = [
    {
        "role": "user",
        "content": f"""Cultural background:
### Gift Giving
- Gifts are usually only given on special occasions and are almost always accompanied with a card.
- Token gifts may be given when visiting a house (e.g. wine, chocolate).
- People tend to open gifts in front of the giver.

Story: During a dinner hosted by her friend, Lisa decided to present a carefully chosen and beautifully wrapped gift to her friend as soon as she entered the home. Is what Lisa did socially acceptable?

{RELEVANCE_QUERY}""",
    },
    {"role": "assistant", "content": "Neutral"},
    {
        "role": "user",
        "content": f"""Cultural background:
### Gift Giving
- It is common to wrap gifts in decorative wrapping. Avoid black or purple.
- Avoid giving knives or scissors as gifts.
- If giving flowers, chrysanthemums symbolise death.

Story: Emily visited her friend's house for dinner and brought a thoughtfully chosen book she knew her friend wanted. Is what Emily did socially acceptable?

{RELEVANCE_QUERY}""",
    },
    {"role": "assistant", "content": "Neutral"},
    {
        "role": "user",
        "content": f"""Cultural background:
### Visiting
- It is customary to bring a bottle of alcohol or a cake as a gesture of thanks.
- Take your shoes off when you enter somebody's house.
- Guests usually stay for as long as they wish.

Story: Alex decided to surprise his friend by showing up at their house without prior notice, bringing along a homemade cake as a gift. Is what Alex did socially acceptable?

{RELEVANCE_QUERY}""",
    },
    {"role": "assistant", "content": "Neutral"},
    {
        "role": "user",
        "content": f"""Cultural background:
### Basic Etiquette
- It is expected that the elderly and those of a higher social status are treated with respect.
- People dress modestly in public.
- Many people avoid blasphemy and cursing.

Story: At a family dinner, Emily continued speaking at her usual volume and directly disagreed with an opinion shared by her friend's father. Is what Emily did socially acceptable?

{RELEVANCE_QUERY}""",
    },
    {"role": "assistant", "content": "Neutral"},
    {
        "role": "user",
        "content": f"""Cultural background:
### Visiting
- Always schedule visits in advance.
- Arrive punctually to respect the host's time.

Story: Laura decided to surprise her friend by showing up at her house one afternoon without calling ahead. Is what Laura did socially acceptable?

{RELEVANCE_QUERY}""",
    },
    {"role": "assistant", "content": "No"},
    {
        "role": "user",
        "content": f"""Cultural background:
### Basic Etiquette
- Strict punctuality is not expected in casual settings, but considerable lateness is also unacceptable.
- Avoid passing objects with the left hand.

Story: Alex was invited to a dinner party at 7 PM but arrived at 8 PM without telling the host. Is what Alex did socially acceptable?

{RELEVANCE_QUERY}""",
    },
    {"role": "assistant", "content": "No"},
    {
        "role": "user",
        "content": f"""Cultural background:
### Gift Giving
- Gifts are generally not opened when received. Often one will thank the giver and set the gift aside.

Story: Upon receiving a birthday gift, her friend immediately opened it, showing excitement and gratitude. Is what her friend did socially acceptable?

{RELEVANCE_QUERY}""",
    },
    {"role": "assistant", "content": "No"},
]


def parse_judgment(response: str) -> str:
    """Return 'yes', 'no', or 'neutral'."""
    text = (response or "").strip().lower()
    # take first token/line
    first = text.replace(".", " ").split()[0] if text else ""
    if first in ("yes", "no", "neutral"):
        return first
    if "neutral" in text or "neither" in text or "irrelevant" in text:
        return "neutral"
    if text.startswith("yes"):
        return "yes"
    if text.startswith("no"):
        return "no"
    return "neutral"


def story_relevant_to_background(
    client: OpenAI,
    model: str,
    background: str,
    story: str,
    temperature: float = 0.0,
) -> bool:
    """True iff Background alone determines Yes/No (story should be rejected)."""
    messages = list(RELEVANCE_FEWSHOT) + [
        {
            "role": "user",
            "content": (
                f"Cultural background:\n{background.strip()}\n\n"
                f"Story: {story.strip()}\n\n"
                f"{RELEVANCE_QUERY}"
            ),
        }
    ]
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=5,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    judgment = parse_judgment(content)
    return judgment in ("yes", "no")

STORY_PROMPT = """Task: Write a short story for a Neutral gold-label example.

The story must be grounded in Country 2's cultural background.
Someone reading only Country 1's cultural background must NOT be able to decide Yes or No about the story's social acceptability.

Country 1 (context — do NOT write a story about these norms):
{country}

Cultural background for Country 1:
{background}

Country 2 (story culture — ground the story here):
{other_country}

Cultural background for Country 2:
{other_background}

Write:
1) A <=5 sentence story describing someone taking an action, ending with "Is what <Name> did socially acceptable?"
2) Do not mention any country, nationality, or geographic location.
3) Do not mention cultural festivals, religions, or other culture identifiers by name.
4) Base the action on Country 2's background, not Country 1's.
5) The action must NOT be something Country 1's background specifically addresses.

Format your response exactly as:
Story: <text>
Explanation: <1-2 sentences explaining why Country 1's background cannot judge this story, without naming Country 2>
"""


def load_api_key(key_path: str) -> str:
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    with open(key_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def is_neutral(row: pd.Series) -> bool:
    return str(row.get("Gold Label", "")).strip().lower() == "neutral"


def parse_story_explanation(response: str) -> Tuple[str, str]:
    story_match = re.search(
        r"^Story:\s*(.+?)(?=^Explanation:|\Z)",
        response,
        flags=re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    expl_match = re.search(
        r"^Explanation:\s*(.+)$",
        response,
        flags=re.MULTILINE | re.IGNORECASE | re.DOTALL,
    )
    if not story_match:
        raise ValueError(f"Could not parse Story from:\n{response}")
    story = story_match.group(1).strip()
    explanation = expl_match.group(1).strip() if expl_match else ""
    return story, explanation


def regenerate_story(
    client: OpenAI,
    model: str,
    row: pd.Series,
    temperature: float,
) -> Tuple[str, str]:
    prompt = STORY_PROMPT.format(
        country=row["Country"],
        background=row["Background"],
        other_country=row["Other Country"],
        other_background=row["Other Background"],
    )
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=500,
        temperature=temperature,
    )
    content = resp.choices[0].message.content or ""
    return parse_story_explanation(content)


def get_target_indices(
    df: pd.DataFrame,
    sample_size: int,
    seed: Optional[int],
    ids: Optional[List[int]] = None,
) -> List[int]:
    indices = df.index[df.apply(is_neutral, axis=1)].tolist()
    if ids is not None:
        id_set = set(ids)
        indices = [i for i in indices if int(df.loc[i, "ID"]) in id_set]
    if sample_size > 0:
        if seed is not None:
            random.seed(seed)
        indices = random.sample(indices, min(sample_size, len(indices)))
    return indices


def _process_one(
    idx: int,
    row: pd.Series,
    client: OpenAI,
    model: str,
    temperature: float,
    regen_temperature: float,
    max_retries: int,
    validate_only: bool,
) -> Tuple[int, Optional[Dict], Optional[Dict]]:
    """Returns (idx, reject_row_or_None, regen_row_or_None)."""
    relevant = story_relevant_to_background(
        client,
        model=model,
        background=row["Background"],
        story=row["Story"],
        temperature=temperature,
    )
    if not relevant:
        return idx, None, None

    reject_row = {
        "ID": row["ID"],
        "Country": row["Country"],
        "Other Country": row["Other Country"],
        "Subaxis": row["Subaxis"],
        "Story": row["Story"],
        "Rule-of-Thumb": row["Rule-of-Thumb"],
        "Background_head": str(row["Background"])[:200],
        "relevant_to_background": True,
    }

    if validate_only:
        return idx, reject_row, None

    old_story = row["Story"]
    old_expl = row["Explanation"]
    accepted = False
    new_story = old_story
    new_expl = old_expl
    attempts = 0

    for attempt in range(max_retries):
        attempts = attempt + 1
        new_story, new_expl = regenerate_story(
            client,
            model=model,
            row=row,
            temperature=regen_temperature,
        )
        still_relevant = story_relevant_to_background(
            client,
            model=model,
            background=row["Background"],
            story=new_story,
            temperature=temperature,
        )
        if not still_relevant:
            accepted = True
            break

    regen_row = {
        "ID": row["ID"],
        "accepted": accepted,
        "attempts": attempts,
        "old_Story": old_story,
        "new_Story": new_story,
        "old_Explanation": old_expl,
        "new_Explanation": new_expl,
    }
    return idx, reject_row, regen_row


def main() -> None:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    default_input = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_neutral_swapped.csv",
    )
    default_output = os.path.join(
        repo_root,
        "output/v2_fix_neutral/normad_etiquette_final_data_story_bg_validated.csv",
    )
    default_reject_log = os.path.join(
        repo_root,
        "output/v2_fix_neutral/story_background_rejects.csv",
    )
    default_regen_log = os.path.join(
        repo_root,
        "output/v2_fix_neutral/story_background_regen_log.csv",
    )
    default_key = os.path.join(repo_root, "key/key.txt")

    parser = argparse.ArgumentParser(
        description="Reject/regenerate neutral stories that are relevant to Background."
    )
    parser.add_argument("--input_path", type=str, default=default_input)
    parser.add_argument("--output_path", type=str, default=default_output)
    parser.add_argument("--reject_log_path", type=str, default=default_reject_log)
    parser.add_argument("--regen_log_path", type=str, default=default_regen_log)
    parser.add_argument("--api_key_path", type=str, default=default_key)
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--regen_temperature", type=float, default=0.7)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument(
        "--sample_size",
        type=int,
        default=0,
        help="Sample this many neutral rows (0 = all neutrals).",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--ids",
        type=str,
        default=None,
        help="Comma-separated IDs to process (optional).",
    )
    parser.add_argument(
        "--ids_file",
        type=str,
        default=None,
        help="CSV with an ID column; only those IDs are processed.",
    )
    parser.add_argument(
        "--validate_only",
        action="store_true",
        help="Only flag rejected stories; do not regenerate.",
    )
    parser.add_argument(
        "--save_only_rejected",
        action="store_true",
        help="Write only rejected/regenerated rows to --output_path.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_path).fillna("")
    id_filter: Optional[List[int]] = None
    if args.ids_file:
        id_filter = pd.read_csv(args.ids_file)["ID"].astype(int).tolist()
    elif args.ids:
        id_filter = [int(x.strip()) for x in args.ids.split(",") if x.strip()]
    target_indices = get_target_indices(df, args.sample_size, args.seed, id_filter)
    if not target_indices:
        print("No neutral rows to validate.")
        return

    client = OpenAI(api_key=load_api_key(args.api_key_path))
    reject_rows: List[Dict] = []
    regen_rows: List[Dict] = []
    touched_indices: List[int] = []
    workers = max(1, args.workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                _process_one,
                idx,
                df.loc[idx],
                client,
                args.model,
                args.temperature,
                args.regen_temperature,
                args.max_retries,
                args.validate_only,
            ): idx
            for idx in target_indices
        }
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Validating stories vs Background",
        ):
            idx, reject_row, regen_row = fut.result()
            if reject_row is None:
                continue
            reject_rows.append(reject_row)
            touched_indices.append(idx)
            if regen_row is not None:
                df.loc[idx, "Story"] = regen_row["new_Story"]
                if regen_row["new_Explanation"]:
                    df.loc[idx, "Explanation"] = regen_row["new_Explanation"]
                if "model" in df.columns:
                    df.loc[idx, "model"] = args.model
                regen_rows.append(regen_row)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    if args.save_only_rejected:
        df.loc[touched_indices].to_csv(args.output_path, index=False)
    else:
        df.to_csv(args.output_path, index=False)

    pd.DataFrame(reject_rows).to_csv(args.reject_log_path, index=False)
    if regen_rows:
        pd.DataFrame(regen_rows).to_csv(args.regen_log_path, index=False)

    print(f"Input:     {args.input_path}")
    print(f"Output:    {args.output_path}")
    print(f"Rejects:   {args.reject_log_path}")
    if regen_rows:
        print(f"Regen log: {args.regen_log_path}")
    print(f"Workers:   {workers}")
    print(f"Neutrals checked: {len(target_indices)}")
    print(f"Rejected (relevant to Background): {len(reject_rows)}")
    if not args.validate_only:
        accepted = sum(1 for r in regen_rows if r["accepted"])
        print(f"Regenerated & accepted: {accepted}/{len(regen_rows)}")


if __name__ == "__main__":
    main()
