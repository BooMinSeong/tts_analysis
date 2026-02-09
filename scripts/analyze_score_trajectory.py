#!/usr/bin/env python
"""Step-by-step PRM score trajectory analysis script.

Analyzes how PRM (Process Reward Model) scores evolve across reasoning steps
for correct vs incorrect completions within individual problems.

The key output is per-problem trajectory plots showing all individual completion
paths (not averaged), colored by correctness. This preserves each problem's
unique characteristics instead of burying them in aggregate statistics.

Difficulty is measured by per-problem accuracy (fraction of correct completions)
at the selected temperature, then stratified by absolute accuracy thresholds:
  Level 1: 80-100% (easiest)
  Level 2: 60-80%
  Level 3: 40-60%
  Level 4: 20-40%
  Level 5: 0-20% (hardest)

Usage:
    # Per-problem trajectory analysis (output auto-generated)
    uv run python scripts/analyze_score_trajectory.py \
        --category math500_Qwen2.5-3B \
        --approach bon \
        --temperature 0.8 \
        --verbose

    # More samples per level
    uv run python scripts/analyze_score_trajectory.py \
        --category math500_Qwen2.5-3B \
        --approach bon \
        --temperature 0.8 \
        --samples-per-level 6

    # From hub path (requires --output-dir)
    uv run python scripts/analyze_score_trajectory.py \
        --hub-path ENSEONG/preprocessed-default-MATH-500-Qwen2.5-3B-Instruct-bon \
        --temperature 0.8 \
        --output-dir outputs/traj-custom

Output (in outputs/traj-{category}-{approach}-{temperature}/):
    - per_problem/level_{N}.png: Grid of sampled problems per difficulty level,
      each subplot showing all individual completion trajectories
    - score_trajectory_report.md: Per-problem statistics table
    - score_trajectory_overall.png: Aggregate comparison (supplementary)
"""

import argparse
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs import load_hub_registry, HubRegistry
from analysis import (
    discover_experiment,
    ExperimentConfig,
    load_experiment_data_by_temperature,
)
from analysis.difficulty import (
    compute_problem_baselines_from_completions,
    stratify_by_absolute_difficulty,
)
from analysis.difficulty_temperature import DEFAULT_DIFFICULTY_THRESHOLDS


NUM_BINS = 50  # For aggregate interpolation


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze PRM score trajectories across reasoning steps",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--hub-path",
        type=str,
        help="Hub dataset path",
    )
    input_group.add_argument(
        "--category",
        type=str,
        help="Category from registry (requires --approach)",
    )

    parser.add_argument(
        "--registry",
        type=str,
        default="configs/registry.yaml",
        help="Path to registry YAML file",
    )
    parser.add_argument(
        "--approach",
        type=str,
        choices=["bon", "beam_search", "dvts"],
        help="Approach to use when using --category",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: outputs/traj-{category}-{approach}-{temperature})",
    )
    parser.add_argument(
        "--samples-per-level",
        type=int,
        default=4,
        help="Number of problems to sample per difficulty level (default: 4)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        required=True,
        help="Temperature to analyze",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Use only this seed (default: use first available)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def get_hub_path_from_category(
    registry: HubRegistry,
    category: str,
    approach: str,
    verbose: bool = True,
) -> Optional[str]:
    """Get hub path from category and approach."""
    paths = registry.get_category(category)
    if not paths:
        if verbose:
            print(f"Error: Category '{category}' not found in registry")
        return None

    for path in paths:
        if approach.lower() in path.lower():
            return path

    if verbose:
        print(f"Error: No path found for category '{category}' with approach '{approach}'")
        print(f"Available paths: {paths}")
    return None


def extract_per_problem_data(dataset) -> dict[str, dict]:
    """Extract per-problem trajectory data from a single dataset.

    Returns:
        {unique_id: {
            'correct': [[scores...], ...],
            'incorrect': [[scores...], ...],
            'problem_text': str,
            'answer': str,
        }}
    """
    if "train" in dataset:
        dataset = dataset["train"]

    problems = {}
    for problem in dataset:
        unique_id = problem.get("unique_id", problem.get("problem", ""))
        if not unique_id:
            continue

        scores = problem.get("scores", [])
        is_correct_preds = problem.get("is_correct_preds", [])
        if not scores or not is_correct_preds:
            continue

        correct = []
        incorrect = []
        for score_list, is_correct in zip(scores, is_correct_preds):
            if not isinstance(score_list, (list, tuple)) or len(score_list) == 0:
                continue
            if is_correct:
                correct.append(list(score_list))
            else:
                incorrect.append(list(score_list))

        problems[unique_id] = {
            "correct": correct,
            "incorrect": incorrect,
            "problem_text": problem.get("problem", ""),
            "answer": str(problem.get("answer", "")),
        }

    return problems


def plot_single_problem(
    ax: plt.Axes,
    correct_trajs: list[list[float]],
    incorrect_trajs: list[list[float]],
    title: str = "",
):
    """Plot all individual trajectories for a single problem.

    Each line = one completion's step-by-step PRM scores.
    Blue = correct, Red = incorrect. Raw step indices on x-axis.
    """
    for traj in incorrect_trajs:
        ax.plot(
            range(len(traj)), traj,
            color="tab:red", alpha=0.15, linewidth=0.7,
        )

    for traj in correct_trajs:
        ax.plot(
            range(len(traj)), traj,
            color="tab:blue", alpha=0.25, linewidth=0.7,
        )

    if correct_trajs:
        ax.plot([], [], color="tab:blue", linewidth=1.5, label=f"Correct ({len(correct_trajs)})")
    if incorrect_trajs:
        ax.plot([], [], color="tab:red", linewidth=1.5, label=f"Incorrect ({len(incorrect_trajs)})")

    ax.set_xlabel("Step")
    ax.set_ylabel("PRM Score")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)


def sample_problems_for_level(
    level_problem_ids: list[str],
    per_problem_data: dict[str, dict],
    n_samples: int,
    rng: np.random.Generator,
) -> list[str]:
    """Sample problems, prioritizing those with both correct and incorrect completions."""
    mixed = []
    one_sided = []
    for pid in level_problem_ids:
        data = per_problem_data.get(pid)
        if data is None:
            continue
        if data["correct"] and data["incorrect"]:
            mixed.append(pid)
        elif data["correct"] or data["incorrect"]:
            one_sided.append(pid)

    selected = []
    if len(mixed) >= n_samples:
        indices = rng.choice(len(mixed), size=n_samples, replace=False)
        selected = [mixed[i] for i in indices]
    else:
        selected = list(mixed)
        remaining = n_samples - len(selected)
        if one_sided and remaining > 0:
            take = min(remaining, len(one_sided))
            indices = rng.choice(len(one_sided), size=take, replace=False)
            selected.extend(one_sided[i] for i in indices)

    return selected


def generate_per_problem_plots(
    per_problem_data: dict[str, dict],
    difficulty_levels: dict,
    baselines: dict,
    output_dir: str,
    samples_per_level: int = 4,
    verbose: bool = True,
):
    """Generate per-problem trajectory grids, one image per difficulty level."""
    per_problem_dir = os.path.join(output_dir, "per_problem")
    os.makedirs(per_problem_dir, exist_ok=True)

    rng = np.random.default_rng(42)

    for level in sorted(difficulty_levels.keys()):
        level_info = difficulty_levels[level]
        sampled = sample_problems_for_level(
            level_info.problem_ids,
            per_problem_data,
            samples_per_level,
            rng,
        )

        if not sampled:
            if verbose:
                print(f"  Level {level}: no problems with trajectory data, skipping")
            continue

        n_cols = min(len(sampled), 2)
        n_rows = (len(sampled) + n_cols - 1) // n_cols

        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(7 * n_cols, 5 * n_rows),
            squeeze=False,
        )

        for idx, pid in enumerate(sampled):
            row, col = divmod(idx, n_cols)
            ax = axes[row][col]
            data = per_problem_data[pid]

            baseline = baselines.get(pid)
            acc_str = f"{baseline.mean_accuracy:.2f}" if baseline else "?"
            n_correct = len(data["correct"])
            n_incorrect = len(data["incorrect"])

            ptext = data["problem_text"].replace("$", "").replace("\\", "")
            if len(ptext) > 80:
                ptext = ptext[:77] + "..."

            title = (
                f"acc={acc_str}  |  {n_correct}C / {n_incorrect}I\n"
                f"{ptext}"
            )

            plot_single_problem(ax, data["correct"], data["incorrect"], title=title)

        for idx in range(len(sampled), n_rows * n_cols):
            row, col = divmod(idx, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            f"Level {level}  (acc: [{level_info.min_accuracy:.0%}, {level_info.max_accuracy:.0%}], "
            f"{level_info.problem_count} problems)",
            fontsize=13,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        path = os.path.join(per_problem_dir, f"level_{level}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if verbose:
            print(f"  Saved: {path} ({len(sampled)} problems)")


def generate_aggregate_plot(
    per_problem_data: dict[str, dict],
    output_dir: str,
    verbose: bool = True,
):
    """Generate aggregate correct vs incorrect trajectory plot (supplementary)."""
    all_correct = []
    all_incorrect = []
    for data in per_problem_data.values():
        all_correct.extend(data["correct"])
        all_incorrect.extend(data["incorrect"])

    target_x = np.linspace(0.0, 1.0, NUM_BINS)

    def interpolate_all(trajs):
        if not trajs:
            return None
        result = []
        for traj in trajs:
            if len(traj) == 1:
                result.append(np.full(NUM_BINS, traj[0]))
            else:
                src_x = np.linspace(0.0, 1.0, len(traj))
                result.append(np.interp(target_x, src_x, traj))
        return np.array(result)

    c_interp = interpolate_all(all_correct)
    i_interp = interpolate_all(all_incorrect)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0.0, 1.0, NUM_BINS)

    if c_interp is not None:
        mean, std = c_interp.mean(axis=0), c_interp.std(axis=0)
        ax.plot(x, mean, color="tab:blue", linewidth=1.5, label=f"Correct (n={len(c_interp)})")
        ax.fill_between(x, mean - std, mean + std, color="tab:blue", alpha=0.15)

    if i_interp is not None:
        mean, std = i_interp.mean(axis=0), i_interp.std(axis=0)
        ax.plot(x, mean, color="tab:red", linewidth=1.5, label=f"Incorrect (n={len(i_interp)})")
        ax.fill_between(x, mean - std, mean + std, color="tab:red", alpha=0.15)

    ax.set_xlabel("Normalized Step Position")
    ax.set_ylabel("PRM Score")
    ax.set_title("PRM Score Trajectory: Correct vs Incorrect (Aggregate)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = os.path.join(output_dir, "score_trajectory_overall.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if verbose:
        print(f"  Saved: {path}")


def generate_report(
    per_problem_data: dict[str, dict],
    difficulty_levels: dict,
    baselines: dict,
    output_dir: str,
    hub_path: str,
    temp_used: Any,
    seed_used: Any,
    verbose: bool = True,
):
    """Generate markdown report with per-problem statistics."""
    lines = []
    lines.append("# PRM Score Trajectory Analysis Report\n")
    lines.append(f"**Hub path:** `{hub_path}`\n")
    lines.append(f"**Temperature:** {temp_used}  |  **Seed:** {seed_used}\n")

    # Overall stats
    total_c = sum(len(d["correct"]) for d in per_problem_data.values())
    total_i = sum(len(d["incorrect"]) for d in per_problem_data.values())
    all_steps_c = [len(t) for d in per_problem_data.values() for t in d["correct"]]
    all_steps_i = [len(t) for d in per_problem_data.values() for t in d["incorrect"]]

    lines.append("## Overall\n")
    lines.append(f"- **Problems:** {len(per_problem_data)}")
    lines.append(f"- **Correct completions:** {total_c}")
    lines.append(f"- **Incorrect completions:** {total_i}")
    if all_steps_c:
        lines.append(f"- **Correct step count:** mean={np.mean(all_steps_c):.1f}, median={np.median(all_steps_c):.0f}")
    if all_steps_i:
        lines.append(f"- **Incorrect step count:** mean={np.mean(all_steps_i):.1f}, median={np.median(all_steps_i):.0f}")
    lines.append("")

    # Difficulty thresholds used
    lines.append("## Difficulty Thresholds\n")
    lines.append("| Level | Accuracy Range | Problem Count |")
    lines.append("|-------|----------------|---------------|")
    for level in sorted(difficulty_levels.keys()):
        info = difficulty_levels[level]
        lines.append(f"| {level} | {info.min_accuracy:.0%} - {info.max_accuracy:.0%} | {info.problem_count} |")
    lines.append("")

    # Per-level summary
    if difficulty_levels:
        lines.append("## By Difficulty Level\n")

        for level in sorted(difficulty_levels.keys()):
            info = difficulty_levels[level]
            lines.append(f"### Level {level} ({info.min_accuracy:.0%} - {info.max_accuracy:.0%}, {info.problem_count} problems)\n")
            lines.append("| Problem (truncated) | Accuracy | Correct | Incorrect | Avg Steps (C) | Avg Steps (I) | Final Score (C) | Final Score (I) |")
            lines.append("|---------------------|----------|---------|-----------|---------------|---------------|-----------------|-----------------|")

            for pid in info.problem_ids:
                data = per_problem_data.get(pid)
                if data is None:
                    continue

                baseline = baselines.get(pid)
                acc = f"{baseline.mean_accuracy:.3f}" if baseline else "?"
                n_c = len(data["correct"])
                n_i = len(data["incorrect"])

                avg_steps_c = f"{np.mean([len(t) for t in data['correct']]):.0f}" if data["correct"] else "-"
                avg_steps_i = f"{np.mean([len(t) for t in data['incorrect']]):.0f}" if data["incorrect"] else "-"

                final_c = f"{np.mean([t[-1] for t in data['correct']]):.3f}" if data["correct"] else "-"
                final_i = f"{np.mean([t[-1] for t in data['incorrect']]):.3f}" if data["incorrect"] else "-"

                ptext = data["problem_text"][:50].replace("|", "\\|")
                lines.append(f"| {ptext} | {acc} | {n_c} | {n_i} | {avg_steps_c} | {avg_steps_i} | {final_c} | {final_i} |")

            lines.append("")

    report_path = os.path.join(output_dir, "score_trajectory_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    if verbose:
        print(f"  Saved: {report_path}")


def main():
    args = parse_args()

    # Determine hub path and category/approach for output path
    hub_path = None
    category = None
    approach = None

    if args.hub_path:
        hub_path = args.hub_path
    elif args.category:
        if not args.approach:
            print("Error: --approach is required when using --category")
            sys.exit(1)

        category = args.category
        approach = args.approach
        registry = load_hub_registry(args.registry)
        hub_path = get_hub_path_from_category(
            registry, category, approach, verbose=args.verbose
        )
        if not hub_path:
            sys.exit(1)

    if not hub_path:
        print("Error: No hub path specified")
        sys.exit(1)

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif category and approach:
        output_dir = f"outputs/traj-{category}-{approach}-{args.temperature}"
    else:
        print("Error: --output-dir is required when using --hub-path")
        sys.exit(1)

    print("=" * 70)
    print("PRM Score Trajectory Analysis (Per-Problem)")
    print("=" * 70)
    print(f"\nHub path: {hub_path}")
    print(f"Output:   {output_dir}")

    # Clean output directory
    if os.path.exists(output_dir):
        if args.verbose:
            print(f"\nCleaning output directory: {output_dir}")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Discover experiment
    if args.verbose:
        print("\nDiscovering experiment configuration...")

    try:
        config = discover_experiment(hub_path)
    except Exception as e:
        print(f"Error discovering experiment: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.verbose:
        print(f"  Model: {config.model}")
        print(f"  Approach: {config.approach}")
        print(f"  Strategy: {config.strategy}")
        print(f"  Seeds: {config.seeds}")
        print(f"  Temperatures: {config.temperatures}")

    # Validate temperature
    single_temps = [t for t in config.temperatures if isinstance(t, (int, float))]
    if args.temperature not in single_temps:
        print(f"Error: Temperature {args.temperature} not available. Available: {single_temps}")
        sys.exit(1)

    selected_temp = args.temperature

    # Select seed
    if args.seed is not None:
        if args.seed not in config.seeds:
            print(f"Error: Seed {args.seed} not available. Available: {config.seeds}")
            sys.exit(1)
        selected_seed = args.seed
    else:
        selected_seed = config.seeds[0]

    print(f"\nTemperature: {selected_temp}")
    print(f"Seed (for plots): {selected_seed}")

    # Load only the selected temperature (all seeds for difficulty baseline)
    if args.verbose:
        print(f"\nLoading datasets for T={selected_temp}...")

    try:
        datasets_by_temp = load_experiment_data_by_temperature(
            config,
            temperatures=[selected_temp],
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error loading datasets: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if selected_temp not in datasets_by_temp or not datasets_by_temp[selected_temp]:
        print(f"Error: No datasets loaded for T={selected_temp}")
        sys.exit(1)

    seed_datasets = datasets_by_temp[selected_temp]

    # Compute difficulty baselines from this temperature's data (all seeds)
    if args.verbose:
        print(f"\nComputing difficulty baselines from T={selected_temp} ({len(seed_datasets)} seeds)...")

    # Format as {approach: {seed: Dataset}} for baseline computation
    baseline_input = {f"T{selected_temp}": seed_datasets}

    try:
        baselines = compute_problem_baselines_from_completions(
            baseline_input,
            aggregate_across_approaches=True,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error computing baselines: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Stratify by absolute difficulty thresholds
    try:
        difficulty_levels = stratify_by_absolute_difficulty(
            baselines,
            thresholds=DEFAULT_DIFFICULTY_THRESHOLDS,
            verbose=args.verbose,
        )
    except Exception as e:
        print(f"Error stratifying: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Extract per-problem data from selected seed for plotting
    if selected_seed not in seed_datasets:
        print(f"Error: No dataset for seed={selected_seed} at T={selected_temp}")
        print(f"Available seeds: {list(seed_datasets.keys())}")
        sys.exit(1)

    target_dataset = seed_datasets[selected_seed]
    per_problem_data = extract_per_problem_data(target_dataset)

    n_with_scores = sum(1 for d in per_problem_data.values() if d["correct"] or d["incorrect"])
    print(f"\nExtracted trajectories for {n_with_scores} problems (T={selected_temp}, seed={selected_seed})")

    if n_with_scores == 0:
        print("Error: No problems with score trajectories found.")
        sys.exit(1)

    # Generate per-problem plots
    print("\nGenerating per-problem trajectory plots...")
    generate_per_problem_plots(
        per_problem_data,
        difficulty_levels,
        baselines,
        output_dir,
        samples_per_level=args.samples_per_level,
        verbose=args.verbose,
    )

    # Generate aggregate plot (supplementary)
    print("\nGenerating aggregate plot...")
    generate_aggregate_plot(per_problem_data, output_dir, verbose=args.verbose)

    # Generate report
    print("\nGenerating report...")
    generate_report(
        per_problem_data,
        difficulty_levels,
        baselines,
        output_dir,
        hub_path=hub_path,
        temp_used=selected_temp,
        seed_used=selected_seed,
        verbose=args.verbose,
    )

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print(f"Output directory: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
