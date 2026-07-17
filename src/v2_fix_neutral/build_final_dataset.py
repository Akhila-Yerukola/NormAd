"""
Build the single v2-fixed NormAd etiquette dataset.

Pipeline (in order):
  1. Swap Country/Background for all neutrals
  2. Regenerate ROT/Value for entail_pred==0.0 (~158)
  3. Fix Canada gift_giving neutrals (~10)
  4. Validate stories against Background; regenerate rejected stories
  5. Regenerate explanations for all neutrals (~815)

Final artifact:
  output/v2_fix_neutral/normad_etiquette_final_data_neutral_fixed.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def run_step(cmd: list[str], dry_run: bool) -> None:
    print("\n>>>", " ".join(cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, check=True)


def main() -> None:
    root = repo_root()
    out_dir = os.path.join(root, "output/v2_fix_neutral")
    src_dir = os.path.dirname(__file__)
    py = sys.executable

    original = os.path.join(
        root,
        "data_and_heval/datasets.zip.enc.extracted/normad_etiquette_final_data.csv",
    )
    swapped = os.path.join(out_dir, "01_neutral_swapped.csv")
    after_rot = os.path.join(out_dir, "02_rot_regenerated.csv")
    after_canada = os.path.join(out_dir, "03_canada_gift_giving_fixed.csv")
    after_story = os.path.join(out_dir, "04_story_bg_validated.csv")
    final = os.path.join(out_dir, "normad_etiquette_final_data_neutral_fixed.csv")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample_size", type=int, default=0, help="0 = all matching rows")
    parser.add_argument("--workers", type=int, default=16, help="Parallel API worker threads")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_swap", action="store_true")
    parser.add_argument("--skip_rot", action="store_true")
    parser.add_argument("--skip_canada", action="store_true")
    parser.add_argument("--skip_story_validate", action="store_true")
    parser.add_argument("--skip_explanations", action="store_true")
    parser.add_argument(
        "--start_from",
        choices=["swap", "rot", "canada", "story", "explanations"],
        default="swap",
        help="Resume from this stage (uses previous stage output as input).",
    )
    args = parser.parse_args()

    os.makedirs(out_dir, exist_ok=True)
    stages = ["swap", "rot", "canada", "story", "explanations"]
    start_idx = stages.index(args.start_from)

    def should_run(name: str, skip: bool) -> bool:
        return (not skip) and stages.index(name) >= start_idx

    # If resuming, prefer existing intermediate if present
    if args.start_from != "swap":
        prev = {
            "rot": swapped,
            "canada": after_rot,
            "story": after_canada,
            "explanations": after_story,
        }[args.start_from]
        if not os.path.exists(prev) and not args.dry_run:
            raise FileNotFoundError(
                f"Cannot start from {args.start_from}: missing {prev}"
            )

    print("=== v2_fix_neutral full pipeline ===")
    print(f"Final output: {final}")
    print(f"sample_size={args.sample_size} workers={args.workers} model={args.model}")

    workers = str(args.workers)
    sample = str(args.sample_size)

    if should_run("swap", args.skip_swap):
        run_step(
            [
                py,
                os.path.join(src_dir, "swap_neutral_country_background.py"),
                "--input_path",
                original,
                "--output_path",
                swapped,
            ],
            args.dry_run,
        )

    if should_run("rot", args.skip_rot):
        run_step(
            [
                py,
                os.path.join(src_dir, "regenerate_rot_value.py"),
                "--input_path",
                swapped,
                "--output_path",
                after_rot,
                "--log_path",
                os.path.join(out_dir, "rot_value_regeneration_log.csv"),
                "--model",
                args.model,
                "--sample_size",
                sample,
                "--workers",
                workers,
            ],
            args.dry_run,
        )

    if should_run("canada", args.skip_canada):
        run_step(
            [
                py,
                os.path.join(src_dir, "fix_canada_gift_giving.py"),
                "--input_path",
                after_rot,
                "--output_path",
                after_canada,
                "--log_path",
                os.path.join(out_dir, "canada_gift_giving_fix_log.csv"),
                "--model",
                args.model,
                "--workers",
                workers,
            ],
            args.dry_run,
        )

    if should_run("story", args.skip_story_validate):
        run_step(
            [
                py,
                os.path.join(src_dir, "validate_story_vs_background.py"),
                "--input_path",
                after_canada,
                "--output_path",
                after_story,
                "--reject_log_path",
                os.path.join(out_dir, "story_background_rejects.csv"),
                "--regen_log_path",
                os.path.join(out_dir, "story_background_regen_log.csv"),
                "--model",
                args.model,
                "--sample_size",
                sample,
                "--workers",
                workers,
            ],
            args.dry_run,
        )

    if should_run("explanations", args.skip_explanations):
        run_step(
            [
                py,
                os.path.join(src_dir, "regenerate_explanations.py"),
                "--input_path",
                after_story,
                "--output_path",
                final,
                "--log_path",
                os.path.join(out_dir, "explanation_regeneration_log.csv"),
                "--model",
                args.model,
                "--filter",
                "all",
                "--sample_size",
                sample,
                "--workers",
                workers,
            ],
            args.dry_run,
        )

    print("\n=== Pipeline complete ===")
    print(f"Single final file: {final}")


if __name__ == "__main__":
    main()
