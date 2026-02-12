#!/usr/bin/env python
"""Score early difficulty estimation datasets.

Computes pred_*@n and pass@k fields from raw completions,
preparing datasets for preprocessing.

Usage:
    # Score from Hub, push results back to Hub
    uv run python scripts/score_early_estimation.py \
        --hub-path ENSEONG/early-MATH-500-Qwen2.5-3B-Instruct-bon \
        --output-hub-path ENSEONG/score-early-MATH-500-Qwen2.5-3B-Instruct-bon \
        --push-to-hub \
        --n-max 64

    # Score from local disk, save locally
    uv run python scripts/score_early_estimation.py \
        --hub-path ENSEONG/early-MATH-500-Qwen2.5-3B-Instruct-bon \
        --output-dir /tmp/scored-early \
        --n-max 16

    # Score specific seeds only
    uv run python scripts/score_early_estimation.py \
        --hub-path ENSEONG/early-MATH-500-Qwen2.5-3B-Instruct-bon \
        --output-dir /tmp/scored-early \
        --seeds 0 42 64 \
        --n-max 64
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from datasets import DatasetDict, load_dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis.discovery import discover_experiment
from analysis.scoring import ScoringConfig, score_dataset, score_pass_at_k

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Score early difficulty estimation datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--hub-path", required=True,
        help="Hub path to early estimation dataset (or local disk path)",
    )
    parser.add_argument(
        "--output-hub-path", type=str, default=None,
        help="Hub path to push scored dataset (required with --push-to-hub)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Local directory to save scored dataset",
    )
    parser.add_argument(
        "--push-to-hub", action="store_true",
        help="Push scored dataset to Hub",
    )
    parser.add_argument(
        "--n-max", type=int, default=64,
        help="Maximum N value for scoring (default: 64)",
    )
    parser.add_argument(
        "--num-proc", type=int, default=4,
        help="Number of processes for parallel processing (default: 4)",
    )
    parser.add_argument(
        "--agg-strategy",
        choices=["min", "prod", "last"],
        default="last",
        help="Score aggregation strategy (default: last)",
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=None,
        help="Specific seeds to process (default: all discovered seeds)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
    )
    args = parser.parse_args()

    verbose = args.verbose and not args.quiet

    if args.push_to_hub and not args.output_hub_path:
        parser.error("--output-hub-path is required when using --push-to-hub")

    if not args.push_to_hub and not args.output_dir:
        parser.error("Either --output-dir or --push-to-hub (with --output-hub-path) is required")

    config = ScoringConfig(n=args.n_max, num_proc=args.num_proc)

    if verbose:
        print(f"Score Early Estimation Datasets")
        print(f"{'='*60}")
        print(f"  Hub path: {args.hub_path}")
        print(f"  N max: {config.n}")
        print(f"  Num proc: {config.num_proc}")
        print(f"  Agg strategy: {args.agg_strategy}")

    # Discover subsets from Hub
    is_local = os.path.isdir(args.hub_path)

    if is_local:
        # Local disk path - load directly (single dataset, no subsets)
        if verbose:
            print(f"\n  Loading from local path: {args.hub_path}")
        dataset = load_from_disk(args.hub_path)

        if verbose:
            print(f"  Loaded {len(dataset)} examples")
            print(f"\n  Scoring...")

        dataset = score_dataset(dataset, config, agg_strategy=args.agg_strategy, verbose=verbose)
        dataset = score_pass_at_k(dataset, config, verbose=verbose)

        # Remove temporary fields
        cols_to_remove = [c for c in ["preds", "scores", "completions", "agg_scores"] if c in dataset.column_names]
        if cols_to_remove:
            dataset = dataset.remove_columns(cols_to_remove)

        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            dataset.save_to_disk(args.output_dir)
            if verbose:
                print(f"\n  Saved to {args.output_dir}")

        if args.push_to_hub:
            if verbose:
                print(f"\n  Pushing to {args.output_hub_path}...")
            dataset.push_to_hub(args.output_hub_path, private=False)
            if verbose:
                print(f"  Done!")
    else:
        # Hub path - discover subsets via discovery system
        try:
            experiment = discover_experiment(args.hub_path)
            subset_names = [s.raw_name for s in experiment.subsets]
            if verbose:
                print(f"  Discovered {len(subset_names)} subsets")
                print(f"  Seeds: {experiment.seeds}")
        except Exception as e:
            logger.error(f"Failed to discover experiment: {e}")
            logger.info("Falling back to loading 'default' config")
            subset_names = ["default"]

        # Filter by seeds if specified
        if args.seeds is not None:
            seed_strs = [str(s) for s in args.seeds]
            filtered = []
            for name in subset_names:
                for seed_str in seed_strs:
                    if f"seed-{seed_str}" in name or f"seed_{seed_str}" in name:
                        filtered.append(name)
                        break
            if verbose:
                print(f"  Filtered to {len(filtered)} subsets for seeds {args.seeds}")
            subset_names = filtered

        if not subset_names:
            logger.error("No subsets found to process")
            sys.exit(1)

        # Process each subset
        for i, subset_name in enumerate(subset_names):
            if verbose:
                print(f"\n{'='*60}")
                print(f"  [{i+1}/{len(subset_names)}] Processing subset: {subset_name}")

            try:
                ds = load_dataset(args.hub_path, subset_name)
                # Handle DatasetDict (take "train" split)
                if isinstance(ds, DatasetDict):
                    ds = ds["train"]

                if verbose:
                    print(f"  Loaded {len(ds)} examples")
                    print(f"  Scoring...")

                ds = score_dataset(ds, config, agg_strategy=args.agg_strategy, verbose=verbose)
                ds = score_pass_at_k(ds, config, verbose=verbose)

                # Remove temporary fields
                cols_to_remove = [c for c in ["preds", "scores", "completions", "agg_scores"] if c in ds.column_names]
                if cols_to_remove:
                    ds = ds.remove_columns(cols_to_remove)

                if verbose:
                    # Show sample of computed fields
                    pred_fields = [c for c in ds.column_names if c.startswith("pred_") or c.startswith("pass@")]
                    print(f"  Computed fields: {pred_fields[:10]}...")

                # Save output
                if args.output_dir:
                    subset_dir = os.path.join(args.output_dir, subset_name)
                    os.makedirs(subset_dir, exist_ok=True)
                    ds.save_to_disk(subset_dir)
                    if verbose:
                        print(f"  Saved to {subset_dir}")

                if args.push_to_hub:
                    if verbose:
                        print(f"  Pushing to {args.output_hub_path} (config: {subset_name})...")
                    ds.push_to_hub(
                        repo_id=args.output_hub_path,
                        config_name=subset_name,
                        private=False,
                    )
                    if verbose:
                        print(f"  Pushed successfully!")

            except Exception as e:
                logger.error(f"Error processing subset {subset_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

    if verbose:
        print(f"\nScoring complete.")


if __name__ == "__main__":
    main()
