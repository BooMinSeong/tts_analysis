#!/usr/bin/env python
"""Unify per-seed experiment data into a single dataset per temperature.

Merges all seed subsets for each temperature into one dataset with concatenated
predictions, scores, and completions per problem. This produces a unified dataset
with more samples per problem, suitable for simulation experiments.

Usage:
    # Save locally for testing
    uv run python scripts/unify_dataset.py \
        --hub-path ENSEONG/preprocessed-default-MATH-500-Qwen2.5-3B-Instruct-bon \
        --output-dir /tmp/unified-test --verbose

    # Push to Hub
    uv run python scripts/unify_dataset.py \
        --hub-path ENSEONG/preprocessed-default-MATH-500-Qwen2.5-3B-Instruct-bon \
        --output-hub-path ENSEONG/unified-default-MATH-500-Qwen2.5-3B-Instruct-bon \
        --push-to-hub --verbose
"""

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import discover_experiment


# Fields to keep in the unified dataset
KEEP_FIELDS = {
    "unique_id", "problem", "answer", "solution", "level", "subject",
    "preds", "scores", "completions",
}


def unify_temperature_subsets(
    hub_path: str,
    temperature: float,
    subsets: list,
    shuffle_seed: int = 42,
    verbose: bool = True,
) -> Dataset:
    """Merge all seed subsets for a single temperature into one dataset.

    For each problem (matched by unique_id), concatenates preds, scores,
    and completions across all seeds, then shuffles with a fixed seed.

    Args:
        hub_path: HuggingFace Hub dataset path
        temperature: Temperature value (for logging)
        subsets: List of SubsetInfo for this temperature
        shuffle_seed: Random seed for reproducible shuffling
        verbose: Print progress

    Returns:
        Unified Dataset with concatenated predictions per problem
    """
    if verbose:
        print(f"\n  Temperature {temperature}: merging {len(subsets)} seed subsets")

    # Load all seed subsets and index by unique_id
    # {unique_id: {field: [values_from_seed1, values_from_seed2, ...]}}
    problem_data = defaultdict(lambda: {"preds": [], "scores": [], "completions": []})
    problem_meta = {}  # unique_id -> metadata fields

    for subset in subsets:
        if verbose:
            print(f"    Loading seed {subset.seed}: {subset.raw_name[:60]}...")

        ds = load_dataset(hub_path, subset.raw_name)["train"]

        for row in ds:
            uid = row["unique_id"]

            # Store metadata from first encounter
            if uid not in problem_meta:
                problem_meta[uid] = {
                    k: row[k] for k in ("unique_id", "problem", "answer", "solution", "level", "subject")
                    if k in row
                }

            # Concatenate list fields
            for field in ("preds", "scores", "completions"):
                if field in row and row[field] is not None:
                    problem_data[uid][field].extend(row[field])

    if verbose:
        print(f"    Found {len(problem_meta)} unique problems")

    # Shuffle with aligned permutation and build output rows
    rng = np.random.default_rng(shuffle_seed)
    rows = []

    for uid in sorted(problem_meta.keys()):
        meta = problem_meta[uid]
        data = problem_data[uid]

        n_samples = len(data["preds"])
        perm = rng.permutation(n_samples)

        row = dict(meta)
        row["preds"] = [data["preds"][i] for i in perm]
        row["scores"] = [data["scores"][i] for i in perm]
        row["completions"] = [data["completions"][i] for i in perm]
        rows.append(row)

    # Validate
    for row in rows:
        n_preds = len(row["preds"])
        n_scores = len(row["scores"])
        n_completions = len(row["completions"])
        assert n_preds == n_scores == n_completions, (
            f"Length mismatch for {row['unique_id']}: "
            f"preds={n_preds}, scores={n_scores}, completions={n_completions}"
        )

    if verbose:
        sample_row = rows[0]
        print(f"    Samples per problem: {len(sample_row['preds'])}")
        print(f"    Validation passed: all lengths consistent")

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Unify per-seed experiment data into one dataset per temperature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hub-path", required=True, help="Source hub dataset path")
    parser.add_argument("--output-hub-path", help="Target hub path for push")
    parser.add_argument("--output-dir", help="Local output directory")
    parser.add_argument("--push-to-hub", action="store_true", help="Push to Hub")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed for shuffling")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    if not args.push_to_hub and not args.output_dir:
        print("Error: Must specify either --push-to-hub or --output-dir")
        sys.exit(1)

    if verbose:
        print(f"Discovering experiment: {args.hub_path}")

    config = discover_experiment(args.hub_path)

    if verbose:
        print(f"  Found {len(config.subsets)} subsets")
        print(f"  Seeds: {config.seeds}")
        print(f"  Temperatures: {config.temperatures}")

    # Group subsets by temperature
    temp_groups = config.group_by_temperature()

    if verbose:
        print(f"\n  Temperature groups: {list(temp_groups.keys())}")

    # Process each temperature
    for temp, subsets in sorted(temp_groups.items()):
        unified_ds = unify_temperature_subsets(
            hub_path=args.hub_path,
            temperature=temp,
            subsets=subsets,
            shuffle_seed=args.shuffle_seed,
            verbose=verbose,
        )

        subset_name = f"T-{temp}"

        if args.push_to_hub:
            output_hub_path = args.output_hub_path
            if output_hub_path is None:
                # Generate default: replace 'preprocessed-' with 'unified-'
                parts = args.hub_path.split("/")
                repo = parts[-1]
                if repo.startswith("preprocessed-"):
                    repo = "unified-" + repo[len("preprocessed-"):]
                else:
                    repo = "unified-" + repo
                output_hub_path = f"{parts[0]}/{repo}"

            if verbose:
                print(f"\n  Pushing {subset_name} to {output_hub_path}...")

            unified_ds.push_to_hub(
                output_hub_path,
                config_name=subset_name,
                private=False,
            )

            if verbose:
                print(f"  Done: {len(unified_ds)} problems pushed")
        else:
            save_path = os.path.join(args.output_dir, subset_name)
            os.makedirs(save_path, exist_ok=True)

            if verbose:
                print(f"\n  Saving {subset_name} to {save_path}...")

            unified_ds.save_to_disk(save_path)

            if verbose:
                print(f"  Done: {len(unified_ds)} problems saved")

    if verbose:
        print(f"\nUnification complete!")


if __name__ == "__main__":
    main()
