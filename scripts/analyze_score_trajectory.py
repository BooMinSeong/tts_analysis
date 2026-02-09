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
    # Single-temperature mode (existing behavior)
    uv run python scripts/analyze_score_trajectory.py \
        --category math500_Qwen2.5-3B \
        --approach bon \
        --temperature 0.8 \
        --verbose

    # Multi-temperature mode (NEW - analyzes all temperatures)
    uv run python scripts/analyze_score_trajectory.py \
        --category math500_Qwen2.5-3B \
        --approach bon \
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

Output (single-temp in outputs/traj-{category}-{approach}-{temperature}/):
    - per_problem/level_{N}.png: Grid of sampled problems per difficulty level,
      each subplot showing all individual completion trajectories
    - score_trajectory_report.md: Per-problem statistics table
    - score_trajectory_overall.png: Aggregate comparison (supplementary)

Output (multi-temp in outputs/traj-{category}-{approach}-multi/):
    - metadata.json: Temperatures analyzed, reference temp, seeds
    - temperature_comparison_report.md: Cross-temp summary
    - difficulty_baselines.json: From reference temp (if --save-baselines)
    - T{temp}/ subdirectories with per-temp analysis
    - comparison/level_*.png: Cross-temperature trajectory comparisons
"""

import argparse
import json
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
from analysis.difficulty_temperature import (
    DEFAULT_DIFFICULTY_THRESHOLDS,
    filter_dataset_by_problems,
)


NUM_BINS = 50  # For aggregate interpolation
REFERENCE_TEMPERATURE = 0.1  # Default reference temperature for difficulty baselines


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
        default=None,
        help="Temperature to analyze (single-temp mode). Omit for multi-temp mode.",
    )
    parser.add_argument(
        "--reference-temp",
        type=float,
        default=REFERENCE_TEMPERATURE,
        help=f"Reference temperature for difficulty baselines (default: {REFERENCE_TEMPERATURE})",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Use only this seed (default: use first available)",
    )
    parser.add_argument(
        "--save-baselines",
        action="store_true",
        help="Save difficulty baselines to JSON file",
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


def filter_valid_temperatures(
    datasets_by_temp: dict[float, dict[int, Any]],
    reference_temp: float,
    verbose: bool = True,
) -> list[float]:
    """Filter to temperatures with complete seed coverage.

    Args:
        datasets_by_temp: {temp: {seed: Dataset}}
        reference_temp: Reference temperature (baseline for seed requirements)
        verbose: Print filtering messages

    Returns:
        List of valid temperatures (those with complete seed coverage)
    """
    if reference_temp not in datasets_by_temp:
        raise ValueError(f"Reference temperature {reference_temp} not found in data")

    reference_seeds = set(datasets_by_temp[reference_temp].keys())
    valid_temps = []

    for temp in sorted(datasets_by_temp.keys()):
        temp_seeds = set(datasets_by_temp[temp].keys())
        if temp_seeds >= reference_seeds:  # Has all reference seeds
            valid_temps.append(temp)
        elif verbose:
            missing = reference_seeds - temp_seeds
            print(f"  Skipping T={temp}: missing seeds {sorted(missing)}")

    return valid_temps


def get_consistent_problem_set(
    datasets_by_temp: dict[float, dict[int, Any]],
    difficulty_levels: dict,
    verbose: bool = True,
) -> dict[int, list[str]]:
    """Get consistent problem sets per difficulty level across all temperatures.

    For each difficulty level, computes the intersection of problems available
    across all temperatures. This ensures fair comparison.

    Args:
        datasets_by_temp: {temp: {seed: Dataset}}
        difficulty_levels: {level: DifficultyLevel}
        verbose: Print consistency info

    Returns:
        {level: [problem_ids]} - consistent problem sets per level
    """
    consistent_sets = {}

    for level, level_info in difficulty_levels.items():
        # For this level, collect problem sets from each temperature
        problem_sets = []

        for temp in sorted(datasets_by_temp.keys()):
            # Get all problems from all seeds at this temperature
            temp_problems = set()
            for seed, dataset in datasets_by_temp[temp].items():
                if "train" in dataset:
                    dataset = dataset["train"]
                for row in dataset:
                    unique_id = row.get("unique_id", row.get("problem", ""))
                    if unique_id in level_info.problem_ids:
                        temp_problems.add(unique_id)
            problem_sets.append(temp_problems)

        # Take intersection across all temperatures
        if problem_sets:
            consistent = set.intersection(*problem_sets)
            consistent_sets[level] = sorted(consistent)

            if verbose:
                original_count = len(level_info.problem_ids)
                consistent_count = len(consistent)
                if consistent_count < original_count:
                    print(f"  Level {level}: {consistent_count}/{original_count} problems (intersection)")
        else:
            consistent_sets[level] = []

    return consistent_sets


def extract_per_problem_data(dataset, problem_filter: Optional[set[str]] = None) -> dict[str, dict]:
    """Extract per-problem trajectory data from a single dataset.

    Args:
        dataset: Dataset to extract from
        problem_filter: Optional set of problem IDs to include (for filtering)

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

        # Apply filter if provided
        if problem_filter is not None and unique_id not in problem_filter:
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


def generate_temperature_comparison_plots(
    per_problem_data_by_temp: dict[float, dict[str, dict]],
    difficulty_levels: dict,
    output_dir: str,
    verbose: bool = True,
):
    """Generate cross-temperature comparison plots, one per difficulty level.

    Args:
        per_problem_data_by_temp: {temp: {problem_id: trajectory_data}}
        difficulty_levels: {level: DifficultyLevel}
        output_dir: Output directory
        verbose: Print progress
    """
    comparison_dir = os.path.join(output_dir, "comparison")
    os.makedirs(comparison_dir, exist_ok=True)

    temperatures = sorted(per_problem_data_by_temp.keys())
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(temperatures)))

    for level in sorted(difficulty_levels.keys()):
        level_info = difficulty_levels[level]

        fig, (ax_correct, ax_incorrect) = plt.subplots(1, 2, figsize=(14, 6))

        for temp_idx, temp in enumerate(temperatures):
            per_problem_data = per_problem_data_by_temp[temp]

            # Collect trajectories for this level
            correct_trajs = []
            incorrect_trajs = []

            for pid in level_info.problem_ids:
                data = per_problem_data.get(pid)
                if data is None:
                    continue
                correct_trajs.extend(data["correct"])
                incorrect_trajs.extend(data["incorrect"])

            # Interpolate to common grid
            target_x = np.linspace(0.0, 1.0, NUM_BINS)

            def interpolate_trajs(trajs):
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

            # Plot correct trajectories
            if correct_trajs:
                c_interp = interpolate_trajs(correct_trajs)
                mean, std = c_interp.mean(axis=0), c_interp.std(axis=0)
                ax_correct.plot(
                    target_x, mean,
                    color=colors[temp_idx],
                    linewidth=2,
                    label=f"T={temp} (n={len(c_interp)})",
                )
                ax_correct.fill_between(
                    target_x, mean - std, mean + std,
                    color=colors[temp_idx],
                    alpha=0.15,
                )

            # Plot incorrect trajectories
            if incorrect_trajs:
                i_interp = interpolate_trajs(incorrect_trajs)
                mean, std = i_interp.mean(axis=0), i_interp.std(axis=0)
                ax_incorrect.plot(
                    target_x, mean,
                    color=colors[temp_idx],
                    linewidth=2,
                    label=f"T={temp} (n={len(i_interp)})",
                )
                ax_incorrect.fill_between(
                    target_x, mean - std, mean + std,
                    color=colors[temp_idx],
                    alpha=0.15,
                )

        # Format correct panel
        ax_correct.set_xlabel("Normalized Step Position")
        ax_correct.set_ylabel("PRM Score")
        ax_correct.set_title("Correct Completions")
        ax_correct.legend(fontsize=9)
        ax_correct.grid(True, alpha=0.3)
        ax_correct.set_ylim(-0.05, 1.05)

        # Format incorrect panel
        ax_incorrect.set_xlabel("Normalized Step Position")
        ax_incorrect.set_ylabel("PRM Score")
        ax_incorrect.set_title("Incorrect Completions")
        ax_incorrect.legend(fontsize=9)
        ax_incorrect.grid(True, alpha=0.3)
        ax_incorrect.set_ylim(-0.05, 1.05)

        fig.suptitle(
            f"Level {level} - Temperature Comparison "
            f"(acc: [{level_info.min_accuracy:.0%}, {level_info.max_accuracy:.0%}], "
            f"{level_info.problem_count} problems)",
            fontsize=14,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        path = os.path.join(comparison_dir, f"level_{level}_temp_comparison.png")
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

    # Filter to single temperatures only (not HNC)
    single_temps = [t for t in config.temperatures if isinstance(t, (int, float))]

    # Mode detection: single-temp vs multi-temp
    if args.temperature is not None:
        # SINGLE-TEMPERATURE MODE (existing behavior)
        if args.temperature not in single_temps:
            print(f"Error: Temperature {args.temperature} not available. Available: {single_temps}")
            sys.exit(1)

        selected_temp = args.temperature

        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        elif category and approach:
            output_dir = f"outputs/traj-{category}-{approach}-{args.temperature}"
        else:
            print("Error: --output-dir is required when using --hub-path")
            sys.exit(1)

        print("=" * 70)
        print("PRM Score Trajectory Analysis (Single-Temperature)")
        print("=" * 70)
        print(f"\nHub path: {hub_path}")
        print(f"Output:   {output_dir}")

        # Clean output directory
        if os.path.exists(output_dir):
            if args.verbose:
                print(f"\nCleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

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

    else:
        # MULTI-TEMPERATURE MODE (new behavior)
        # Determine output directory
        if args.output_dir:
            output_dir = args.output_dir
        elif category and approach:
            output_dir = f"outputs/traj-{category}-{approach}-multi"
        else:
            print("Error: --output-dir is required when using --hub-path")
            sys.exit(1)

        print("=" * 70)
        print("PRM Score Trajectory Analysis (Multi-Temperature)")
        print("=" * 70)
        print(f"\nHub path: {hub_path}")
        print(f"Output:   {output_dir}")

        # Clean output directory
        if os.path.exists(output_dir):
            if args.verbose:
                print(f"\nCleaning output directory: {output_dir}")
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)

        # Validate reference temperature
        reference_temp = args.reference_temp
        if reference_temp not in single_temps:
            print(f"Error: Reference temperature {reference_temp} not found in data")
            print(f"Available temperatures: {single_temps}")
            sys.exit(1)

        print(f"\nReference temperature: {reference_temp}")

        # Load ALL temperatures
        if args.verbose:
            print(f"\nLoading datasets for all temperatures...")

        try:
            datasets_by_temp = load_experiment_data_by_temperature(
                config,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"Error loading datasets: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        # Filter to only single temperatures (not HNC)
        datasets_by_temp = {t: ds for t, ds in datasets_by_temp.items() if isinstance(t, (int, float))}

        # Filter to valid temperatures (complete seed coverage)
        print(f"\nFiltering to temperatures with complete seed coverage...")
        try:
            valid_temps = filter_valid_temperatures(
                datasets_by_temp,
                reference_temp,
                verbose=args.verbose,
            )
        except Exception as e:
            print(f"Error filtering temperatures: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

        if not valid_temps:
            print("Error: No temperatures with complete seed coverage")
            sys.exit(1)

        print(f"Valid temperatures: {valid_temps}")

        # Filter datasets to only valid temperatures
        datasets_by_temp = {t: ds for t, ds in datasets_by_temp.items() if t in valid_temps}

        # Compute difficulty baselines from reference temperature
        print(f"\nComputing difficulty baselines from T={reference_temp}...")
        reference_datasets = datasets_by_temp[reference_temp]
        baseline_input = {f"T{reference_temp}": reference_datasets}

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

        # Get consistent problem sets across temperatures
        print(f"\nComputing consistent problem sets across temperatures...")
        consistent_problem_sets = get_consistent_problem_set(
            datasets_by_temp,
            difficulty_levels,
            verbose=args.verbose,
        )

        # Update difficulty levels with consistent problem sets
        for level in difficulty_levels:
            difficulty_levels[level].problem_ids = consistent_problem_sets[level]
            difficulty_levels[level].problem_count = len(consistent_problem_sets[level])

        # Select seed for trajectory extraction
        if args.seed is not None:
            if args.seed not in config.seeds:
                print(f"Error: Seed {args.seed} not available. Available: {config.seeds}")
                sys.exit(1)
            selected_seed = args.seed
        else:
            selected_seed = config.seeds[0]

        print(f"Using seed {selected_seed} for trajectory extraction")

        # Extract per-problem data for each temperature
        print(f"\nExtracting trajectory data for each temperature...")
        per_problem_data_by_temp = {}

        for temp in tqdm(valid_temps, desc="Processing temperatures", disable=not args.verbose):
            seed_datasets = datasets_by_temp[temp]
            if selected_seed not in seed_datasets:
                print(f"  Warning: Seed {selected_seed} not found for T={temp}, skipping")
                continue

            # Get consistent problems for filtering
            all_consistent_problems = set()
            for level_pids in consistent_problem_sets.values():
                all_consistent_problems.update(level_pids)

            target_dataset = seed_datasets[selected_seed]
            per_problem_data = extract_per_problem_data(
                target_dataset,
                problem_filter=all_consistent_problems,
            )
            per_problem_data_by_temp[temp] = per_problem_data

            n_with_scores = sum(1 for d in per_problem_data.values() if d["correct"] or d["incorrect"])
            if args.verbose:
                print(f"  T={temp}: {n_with_scores} problems with trajectories")

        # Generate per-temperature plots
        print(f"\nGenerating per-temperature plots...")
        for temp in tqdm(valid_temps, desc="Generating plots", disable=not args.verbose):
            if temp not in per_problem_data_by_temp:
                continue

            temp_output_dir = os.path.join(output_dir, f"T{temp}")
            os.makedirs(temp_output_dir, exist_ok=True)

            per_problem_data = per_problem_data_by_temp[temp]

            # Generate per-problem plots
            generate_per_problem_plots(
                per_problem_data,
                difficulty_levels,
                baselines,
                temp_output_dir,
                samples_per_level=args.samples_per_level,
                verbose=False,  # Suppress per-file messages in multi-temp mode
            )

            # Generate aggregate plot
            generate_aggregate_plot(per_problem_data, temp_output_dir, verbose=False)

            # Generate report
            generate_report(
                per_problem_data,
                difficulty_levels,
                baselines,
                temp_output_dir,
                hub_path=hub_path,
                temp_used=temp,
                seed_used=selected_seed,
                verbose=False,
            )

        # Generate cross-temperature comparison plots
        print(f"\nGenerating cross-temperature comparison plots...")
        generate_temperature_comparison_plots(
            per_problem_data_by_temp,
            difficulty_levels,
            output_dir,
            verbose=args.verbose,
        )

        # Save metadata
        metadata = {
            "hub_path": hub_path,
            "reference_temperature": reference_temp,
            "valid_temperatures": valid_temps,
            "seeds": config.seeds,
            "seed_used_for_plots": selected_seed,
            "difficulty_levels": {
                level: {
                    "min_accuracy": info.min_accuracy,
                    "max_accuracy": info.max_accuracy,
                    "problem_count": info.problem_count,
                }
                for level, info in difficulty_levels.items()
            },
        }
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        if args.verbose:
            print(f"  Saved: {metadata_path}")

        # Optionally save baselines
        if args.save_baselines:
            baselines_dict = {
                pid: {
                    "unique_id": bl.unique_id,
                    "mean_accuracy": bl.mean_accuracy,
                    "num_evaluations": bl.num_evaluations,
                    "problem_text": bl.problem_text[:100],  # Truncate for readability
                    "answer": bl.answer,
                }
                for pid, bl in baselines.items()
            }
            baselines_path = os.path.join(output_dir, "difficulty_baselines.json")
            with open(baselines_path, "w") as f:
                json.dump(baselines_dict, f, indent=2)

            if args.verbose:
                print(f"  Saved: {baselines_path}")

        # Generate temperature comparison report
        report_lines = []
        report_lines.append("# Multi-Temperature PRM Score Trajectory Analysis\n")
        report_lines.append(f"**Hub path:** `{hub_path}`\n")
        report_lines.append(f"**Reference temperature:** {reference_temp}\n")
        report_lines.append(f"**Valid temperatures:** {valid_temps}\n")
        report_lines.append(f"**Seed (for plots):** {selected_seed}\n")
        report_lines.append("\n## Temperature Coverage\n")
        report_lines.append("| Temperature | Seeds Available | Status |")
        report_lines.append("|-------------|-----------------|--------|")
        for temp in sorted(single_temps):
            if temp in valid_temps:
                seeds = sorted(datasets_by_temp[temp].keys())
                report_lines.append(f"| {temp} | {seeds} | ✓ Analyzed |")
            else:
                status = "✗ Incomplete seed coverage"
                report_lines.append(f"| {temp} | - | {status} |")
        report_lines.append("\n## Difficulty Levels\n")
        report_lines.append("| Level | Accuracy Range | Problem Count (Consistent) |")
        report_lines.append("|-------|----------------|----------------------------|")
        for level in sorted(difficulty_levels.keys()):
            info = difficulty_levels[level]
            report_lines.append(f"| {level} | {info.min_accuracy:.0%} - {info.max_accuracy:.0%} | {info.problem_count} |")
        report_lines.append("\n## Output Structure\n")
        report_lines.append("- `T{temp}/`: Per-temperature analysis (plots, reports)")
        report_lines.append("- `comparison/`: Cross-temperature comparison plots")
        report_lines.append("- `metadata.json`: Analysis metadata")
        if args.save_baselines:
            report_lines.append("- `difficulty_baselines.json`: Difficulty baseline data")

        report_path = os.path.join(output_dir, "temperature_comparison_report.md")
        with open(report_path, "w") as f:
            f.write("\n".join(report_lines))

        if args.verbose:
            print(f"  Saved: {report_path}")

        print("\n" + "=" * 70)
        print("Multi-temperature analysis complete!")
        print(f"Output directory: {output_dir}")
        print(f"Temperatures analyzed: {valid_temps}")
        print("=" * 70)


if __name__ == "__main__":
    main()
