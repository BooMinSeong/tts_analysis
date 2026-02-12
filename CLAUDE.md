# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> **For user-facing documentation, see [README.md](README.md)**

## Project Overview

This is a **test-time scaling (TTS) experiment analysis** tool for mathematical reasoning tasks. The project analyzes LLM outputs from various search strategies (Best-of-N, Beam Search, DVTS) across different temperatures and model sizes.

**Core Principle**: Hub data is the Single Source of Truth. All experiment metadata (seeds, temperatures, model configs) is automatically discovered from HuggingFace Hub dataset subset names, not manually specified in config files.

The project uses **uv** for Python package management. Always use `uv run python` instead of `python` to run scripts.

## Command Reference

See [README.md](README.md) for all commands and usage examples.

## Key Concepts

**Approaches**: Search strategies used during inference
- `bon`: Best-of-N sampling
- `beam_search`: Beam search decoding
- `dvts`: Diverse verification tree search

**Strategies**: Temperature allocation strategies
- `default`: Single fixed temperature per experiment
- `hnc`: Heterogeneous N-sample Composition (multiple temps with ratios)
- `early`: Early difficulty estimation (temperature range)

**Aggregation Methods**: How to select final answer from N samples
- `naive`: Select based on raw model scores
- `weighted`: Weight by inverse difficulty
- `maj`: Majority vote
- `completions`: Use first correct completion (for baselines)

**Difficulty Levels**: Problems stratified by baseline solve rate
- `easy`: ≥80% solve rate
- `medium`: 40-80% solve rate
- `hard`: <40% solve rate

## Auto-Discovery System

The project's key architectural innovation is automatic metadata discovery from Hub dataset subset names:

1. **Parser (`analysis/parser.py`)**: Parses subset names to extract temperature, seed, strategy
   - Default strategy: `{source}_{dataset}--T-{temp}--...--seed-{seed}--...`
   - HNC strategy: `{source}_{dataset}--temps_{t1}_{t2}...--...--seed-{seed}--...`
   - Early strategy: `temp-low-{low}-high-{high}`

2. **Discovery (`analysis/discovery.py`)**: Auto-discovers experiment configs from Hub
   - Returns `ExperimentConfig` with all seeds, temperatures, subsets discovered
   - No manual config needed beyond Hub path

3. **Registry (`configs/registry.yaml`)**: Only stores Hub paths grouped by category
   - Seeds, temperatures auto-discovered at runtime
   - Minimal, declarative configuration

## Development Patterns

### Never Hardcode Metadata

❌ **Don't do this**:
```python
seeds = [128, 192, 256]  # Hardcoded
temperatures = [0.4, 0.8, 1.2, 1.6]  # Hardcoded
```

✅ **Do this instead**:
```python
config = discover_experiment(hub_path)
seeds = config.seeds  # Auto-discovered
temperatures = config.temperatures  # Auto-discovered
```

### Loading Data Pattern

```python
from analysis import discover_experiment, load_experiment_data_by_temperature

# Auto-discover config
config = discover_experiment(hub_path)

# Load by temperature (for temperature analysis)
datasets_by_temp = load_experiment_data_by_temperature(config)
for temp, seed_datasets in datasets_by_temp.items():
    for seed, dataset in seed_datasets.items():
        # analyze this (temp, seed) combination
        metrics = analyze_single_dataset(dataset, hub_path, seed)
```

### Random Sampling Pattern

```python
import numpy as np

# Initialize RNG for reproducible sampling
rng = np.random.default_rng(seed)

# Sample without replacement
n = len(items)
k = min(sample_size, n)  # Avoid sampling more than available
indices = rng.choice(n, size=k, replace=False)
sampled = [items[i] for i in indices]
```

### Progress Tracking Pattern

```python
from tqdm import tqdm

# For user-facing progress
iterator = tqdm(items, desc="Description", disable=not verbose)

# For nested progress (auto-cleans up)
nested = tqdm(items, desc="  Detail", leave=False, disable=not verbose)
```

### Error Handling Pattern

```python
try:
    # risky operation
except Exception as e:
    print(f"ERROR in {context}: {e}")
    if verbose:
        import traceback
        traceback.print_exc()
    raise  # or continue, depending on severity
```

### Answer Evaluation

All answer evaluation uses `math_verify` library for mathematical equivalence:
```python
from analysis import evaluate_answer, evaluate_result

# For raw completions
is_correct = evaluate_answer(completion_text, gold_answer)

# For preprocessed predictions
is_correct = evaluate_result(data_row, key="pred_naive@1")
```

## Common Pitfalls

1. **Don't hardcode metadata** - Always discover from Hub using `discover_experiment()`
2. **Don't forget preprocessing** - Raw datasets won't have `is_correct_*` fields
3. **Check for verbose flag** - Respect user's output preference with `disable=not verbose`
4. **Include error context** - Always show (hub, temp, seed) on failure
5. **Use uv run** - Don't use bare `python` command

## Data Flow

1. **Load**: `load_experiment_data_by_temperature()` → `Dict[temp, Dict[seed, Dataset]]`
2. **Evaluate**: `analyze_single_dataset()` → `Dict[method, Dict[n, accuracy]]`
3. **Aggregate**: `aggregate_across_seeds()` → Mean ± std across seeds
4. **Visualize**: `plot_scaling_curve()`, `plot_comparison()` → Figures + reports

## Registry Structure

`configs/registry.yaml` organizes experiments by category:

**Format**: `{dataset}_{model}[_{strategy}]`

**Examples**:
- `math500_Qwen2.5-3B`: MATH-500, 3B model, default strategy
- `math500_Qwen2.5-3B_hnc`: MATH-500, 3B model, HNC strategy
- `math500-early_Qwen2.5-3B`: MATH-500, 3B model, early strategy

Contains only Hub paths - metadata auto-discovered at runtime.

## Output Conventions

Analysis scripts generate outputs in designated directories:
- `analysis_output/`: Default output for comparative analyses
- `analysis_output-{DATASET}-{MODEL}-{APPROACH}-{TYPE}/`: Specific analyses

Generated artifacts:
- `*.png`: Matplotlib figures (scaling curves, comparisons, heatmaps)
- `*_report.md`: Markdown summary tables
- Console output: Human-readable results with statistical significance

## Notes

- This project is for **analysis only**, not for running inference experiments
- All datasets are hosted on HuggingFace Hub under the `ENSEONG` organization
- The `legacy/` directory contains old scripts for reference; don't use them for new work
- Preprocessed datasets have the `preprocessed-` prefix in their Hub names
