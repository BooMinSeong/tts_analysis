#!/usr/bin/env python3
"""
Simple analysis script for the early difficulty estimation dataset.

This demonstrates how to analyze the preprocessed-score-early dataset
using the existing analysis infrastructure.

Usage:
    uv run python scripts/analyze_early_dataset.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import discover_experiment, load_experiment_data_by_temperature
from analysis.metrics import analyze_single_dataset, aggregate_across_seeds
import numpy as np


def main():
    # Hub path for the early difficulty estimation dataset
    hub_path = "ENSEONG/preprocessed-score-early-MATH-500-Qwen2.5-3B-Instruct-bon"

    print("=" * 70)
    print("Early Difficulty Estimation Dataset Analysis")
    print("=" * 70)
    print()

    # Discover experiment configuration
    print(f"Discovering: {hub_path}")
    config = discover_experiment(hub_path)

    print(f"  Model: {config.model}")
    print(f"  Approach: {config.approach}")
    print(f"  Strategy: {config.strategy}")
    print(f"  Seeds: {config.seeds}")
    print(f"  Temperatures: {config.temperatures}")
    print()

    # Load datasets
    print("Loading datasets...")
    datasets_by_temp = load_experiment_data_by_temperature(
        config,
        verbose=True,
    )
    print()

    # Analyze each temperature configuration
    for temp, seed_datasets in datasets_by_temp.items():
        temp_str = f"temps_({temp[0]}, {temp[1]})" if isinstance(temp, tuple) else f"T={temp}"
        print(f"Analyzing {temp_str}:")
        print("-" * 70)

        # Analyze each seed
        all_results = {}
        for seed, dataset in seed_datasets.items():
            print(f"  Seed {seed}...")
            results = analyze_single_dataset(
                dataset["train"],
                config.approach,
                seed,
                verbose=False,
            )
            all_results[seed] = results

        # Handle single seed vs multiple seeds
        if len(all_results) == 1:
            # Single seed - just display results directly
            seed = list(all_results.keys())[0]
            results = all_results[seed]

            print("\n  Results (single seed, no aggregation):")
            print("  " + "-" * 50)
            print(f"  {'Method':<20} {'n':<10} {'Accuracy':<15}")
            print("  " + "-" * 50)

            for method, n_results in sorted(results.items()):
                for n, accuracy in sorted(n_results.items()):
                    print(f"  {method:<20} {n:<10} {accuracy:>6.4f}")
        else:
            # Multiple seeds - aggregate
            print("\n  Aggregating results across seeds...")
            aggregated = aggregate_across_seeds(all_results)

            print("\n  Results:")
            print("  " + "-" * 66)
            print(f"  {'Method':<20} {'n':<10} {'Accuracy':<15} {'Std':<10}")
            print("  " + "-" * 66)

            for method, n_results in sorted(aggregated.items()):
                for n, (mean, std) in sorted(n_results.items()):
                    print(f"  {method:<20} {n:<10} {mean:>6.4f}         {std:>6.4f}")

        print()

    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
