#!/usr/bin/env python
"""Simulate early difficulty estimation for adaptive temperature allocation.

Uses a small number of low-temperature samples to estimate problem difficulty,
then adaptively allocates remaining samples from either low or high temperature.

Hypothesis: if low-temperature probe predictions show strong agreement (dominant),
the problem is easy → use all low-temp samples. Otherwise, the problem is hard →
mix in high-temp samples for diversity.

Usage:
    uv run python scripts/simulate_early_estimation.py \
        --hub-path ENSEONG/unified-default-MATH-500-Qwen2.5-3B-Instruct-bon \
        --low-temp 0.1 --high-temp 0.8 \
        --probe-n 8 --total-n 64 --dominance-threshold 4 \
        --output-dir analysis_output-early-estimation --verbose
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_unified_subsets(hub_path: str, low_temp: float, high_temp: float, verbose: bool = True):
    """Load low and high temperature subsets from unified dataset.

    Args:
        hub_path: Hub path to unified dataset
        low_temp: Low temperature value
        high_temp: High temperature value
        verbose: Print progress

    Returns:
        Tuple of (low_temp_dataset, high_temp_dataset) indexed by unique_id
    """
    low_name = f"T-{low_temp}"
    high_name = f"T-{high_temp}"

    if verbose:
        print(f"Loading {low_name}...")
    ds_low = load_dataset(hub_path, low_name)["train"]

    if verbose:
        print(f"Loading {high_name}...")
    ds_high = load_dataset(hub_path, high_name)["train"]

    # Index by unique_id
    low_by_id = {row["unique_id"]: row for row in ds_low}
    high_by_id = {row["unique_id"]: row for row in ds_high}

    if verbose:
        print(f"  Low-temp problems: {len(low_by_id)}")
        print(f"  High-temp problems: {len(high_by_id)}")

        common = set(low_by_id.keys()) & set(high_by_id.keys())
        print(f"  Common problems: {len(common)}")

    return low_by_id, high_by_id


def simulate_adaptive_selection(
    low_by_id: dict,
    high_by_id: dict,
    probe_n: int = 8,
    total_n: int = 64,
    dominance_threshold: int = 4,
    seed: int = 42,
    verbose: bool = True,
) -> list[dict]:
    """Run adaptive selection simulation for all problems.

    For each problem:
    1. Take probe_n predictions from low-temp as probe
    2. Check if the most common prediction appears >= dominance_threshold times
    3. If dominant: use all low-temp predictions (easy problem)
    4. If not dominant: use probe + remaining from high-temp (hard problem)

    Args:
        low_by_id: Low-temp data indexed by unique_id
        high_by_id: High-temp data indexed by unique_id
        probe_n: Number of probe samples from low-temp
        total_n: Total budget of predictions to use
        dominance_threshold: Min count for dominant prediction
        seed: Random seed for reproducible sampling
        verbose: Print progress

    Returns:
        List of result dicts per problem
    """
    common_ids = sorted(set(low_by_id.keys()) & set(high_by_id.keys()))
    remaining_n = total_n - probe_n

    # Initialize RNG for reproducible sampling
    rng = np.random.default_rng(seed)

    if verbose:
        print(f"\nSimulation parameters:")
        print(f"  Probe samples (low-temp): {probe_n}")
        print(f"  Total budget: {total_n}")
        print(f"  Dominance threshold: {dominance_threshold}")
        print(f"  Random seed: {seed}")
        print(f"  Remaining after probe: {remaining_n}")

    results = []

    for uid in common_ids:
        row_low = low_by_id[uid]
        row_high = high_by_id[uid]

        preds_low = row_low["preds"]
        preds_high = row_high["preds"]
        scores_low = row_low.get("scores", [])
        scores_high = row_high.get("scores", [])
        completions_low = row_low.get("completions", [])
        completions_high = row_high.get("completions", [])

        n_low = len(preds_low)
        n_high = len(preds_high)

        # Probe: randomly sample probe_n from low-temp (instead of [:probe_n])
        probe_indices = rng.choice(n_low, size=min(probe_n, n_low), replace=False)
        probe_preds = [preds_low[i] for i in probe_indices]
        probe_scores = [scores_low[i] for i in probe_indices] if scores_low else []
        probe_completions = [completions_low[i] for i in probe_indices] if completions_low else []

        # Agreement check
        counter = Counter(probe_preds)
        most_common_count = counter.most_common(1)[0][1]
        dominant = most_common_count >= dominance_threshold

        # Adaptive selection with random sampling
        if dominant:
            # Randomly sample total_n from low-temp (instead of [:total_n])
            all_indices = rng.choice(n_low, size=min(total_n, n_low), replace=False)
            selected_preds = [preds_low[i] for i in all_indices]
            selected_scores = [scores_low[i] for i in all_indices] if scores_low else []
            selected_completions = [completions_low[i] for i in all_indices] if completions_low else []
        else:
            # Keep probe + randomly sample remaining from high-temp (instead of [:remaining_n])
            remaining_indices = rng.choice(n_high, size=min(remaining_n, n_high), replace=False)
            remaining_preds = [preds_high[i] for i in remaining_indices]
            remaining_scores = [scores_high[i] for i in remaining_indices] if scores_high else []
            remaining_completions = [completions_high[i] for i in remaining_indices] if completions_high else []

            selected_preds = probe_preds + remaining_preds
            selected_scores = probe_scores + remaining_scores
            selected_completions = probe_completions + remaining_completions

        # Build result row
        result = {
            "unique_id": uid,
            "problem": row_low["problem"],
            "answer": row_low["answer"],
            "solution": row_low.get("solution", ""),
            "level": row_low.get("level", ""),
            "subject": row_low.get("subject", ""),
            "dominant": dominant,
            "preds": selected_preds,
            "scores": selected_scores,
            "completions": selected_completions,
            "probe_agreement": most_common_count,
        }
        results.append(result)

    return results


def print_statistics(results: list[dict], verbose: bool = True):
    """Print classification statistics.

    Args:
        results: List of result dicts from simulate_adaptive_selection
        verbose: Print detailed output
    """
    total = len(results)
    n_dominant = sum(1 for r in results if r["dominant"])
    n_non_dominant = total - n_dominant

    print(f"\n{'='*60}")
    print(f"Classification Statistics")
    print(f"{'='*60}")
    print(f"  Total problems: {total}")
    print(f"  Dominant (easy): {n_dominant} ({100*n_dominant/total:.1f}%)")
    print(f"  Non-dominant (hard): {n_non_dominant} ({100*n_non_dominant/total:.1f}%)")

    # By difficulty level
    level_stats = {}
    for r in results:
        level = r.get("level", "unknown")
        if level not in level_stats:
            level_stats[level] = {"total": 0, "dominant": 0}
        level_stats[level]["total"] += 1
        if r["dominant"]:
            level_stats[level]["dominant"] += 1

    if any(r.get("level") for r in results):
        print(f"\n  By difficulty level:")
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            pct = 100 * stats["dominant"] / stats["total"] if stats["total"] > 0 else 0
            print(f"    {level}: {stats['dominant']}/{stats['total']} dominant ({pct:.1f}%)")

    # Agreement distribution
    if verbose:
        agreement_counts = Counter(r["probe_agreement"] for r in results)
        probe_size = len(results[0]["preds"]) if results[0].get("dominant") else results[0]["probe_agreement"]
        print(f"\n  Probe agreement distribution:")
        for k in sorted(agreement_counts.keys()):
            print(f"    {k} agreements: {agreement_counts[k]} problems")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate early difficulty estimation for adaptive temperature",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--hub-path", required=True,
                        help="Hub path to unified dataset")
    parser.add_argument("--low-temp", type=float, default=0.1,
                        help="Low temperature value (default: 0.1)")
    parser.add_argument("--high-temp", type=float, default=0.8,
                        help="High temperature value (default: 0.8)")
    parser.add_argument("--probe-n", type=int, default=8,
                        help="Number of probe samples (default: 8)")
    parser.add_argument("--total-n", type=int, default=64,
                        help="Total prediction budget (default: 64)")
    parser.add_argument("--dominance-threshold", type=int, default=4,
                        help="Min count for dominant prediction (default: 4)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling (default: 42)")
    parser.add_argument("--output-dir", type=str,
                        help="Directory to save results")
    parser.add_argument("--output-hub-path", type=str,
                        help="Hub path to push result dataset")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push result dataset to Hub")
    parser.add_argument("--verbose", "-v", action="store_true", default=True)
    parser.add_argument("--quiet", "-q", action="store_true")
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    if verbose:
        print(f"Early Difficulty Estimation Simulation")
        print(f"{'='*60}")
        print(f"  Hub path: {args.hub_path}")
        print(f"  Low temp: {args.low_temp}, High temp: {args.high_temp}")
        print(f"  Probe: {args.probe_n}, Total: {args.total_n}")
        print(f"  Dominance threshold: {args.dominance_threshold}")
        print(f"  Random seed: {args.seed}")

    # Load data
    low_by_id, high_by_id = load_unified_subsets(
        args.hub_path, args.low_temp, args.high_temp, verbose=verbose
    )

    # Run simulation
    results = simulate_adaptive_selection(
        low_by_id, high_by_id,
        probe_n=args.probe_n,
        total_n=args.total_n,
        dominance_threshold=args.dominance_threshold,
        seed=args.seed,
        verbose=verbose,
    )

    # Print statistics
    print_statistics(results, verbose=verbose)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

        # Save as HF dataset
        ds = Dataset.from_list(results)
        ds_path = os.path.join(args.output_dir, "adaptive_selection")
        ds.save_to_disk(ds_path)
        if verbose:
            print(f"\n  Dataset saved to {ds_path}")

        # Save summary as JSON
        subset_name = f"temp-low-{args.low_temp}-high-{args.high_temp}-probe-{args.probe_n}_total-{args.total_n}_thresh-{args.dominance_threshold}_seed-{args.seed}"
        summary = {
            "hub_path": args.hub_path,
            "subset_name": subset_name,
            "low_temp": args.low_temp,
            "high_temp": args.high_temp,
            "probe_n": args.probe_n,
            "total_n": args.total_n,
            "dominance_threshold": args.dominance_threshold,
            "seed": args.seed,
            "total_problems": len(results),
            "n_dominant": sum(1 for r in results if r["dominant"]),
            "n_non_dominant": sum(1 for r in results if not r["dominant"]),
        }
        summary_path = os.path.join(args.output_dir, "simulation_summary.json")
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        if verbose:
            print(f"  Summary saved to {summary_path}")

    if args.push_to_hub and args.output_hub_path:
        ds = Dataset.from_list(results)
        # Include temperature config and seed in subset name for reproducibility
        # Format: temp-low-{low}-high-{high}-probe-{p}_total-{t}_thresh-{th}_seed-{s}
        config_name = f"temp-low-{args.low_temp}-high-{args.high_temp}-probe-{args.probe_n}_total-{args.total_n}_thresh-{args.dominance_threshold}_seed-{args.seed}"
        if verbose:
            print(f"\n  Pushing to {args.output_hub_path} (config: {config_name})...")
        ds.push_to_hub(args.output_hub_path, config_name=config_name, private=False)
        if verbose:
            print(f"  Done!")

    if verbose:
        print(f"\nSimulation complete.")


if __name__ == "__main__":
    main()
