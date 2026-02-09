# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **test-time scaling (TTS) experiment analysis** tool for mathematical reasoning tasks. The project analyzes LLM outputs from various search strategies (Best-of-N, Beam Search, DVTS) across different temperatures and model sizes.

**Core Principle**: Hub data is the Single Source of Truth. All experiment metadata (seeds, temperatures, model configs) is automatically discovered from HuggingFace Hub dataset subset names, not manually specified in config files.

The project uses **uv** for Python package management. Always use `uv run python` instead of `python` to run scripts.

## Command Reference

### Running Analyses

```bash
# List all available experiments
uv run python scripts/analyze_results.py --list

# Analyze single experiment (auto-discovers all metadata from Hub)
uv run python scripts/analyze_results.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon

# Analyze by category (defined in configs/registry.yaml)
uv run python scripts/analyze_results.py --category math500_Qwen2.5-1.5B

# Compare HNC vs Default strategies
uv run python scripts/analyze_results.py \
    --category math500_Qwen2.5-1.5B_hnc,math500_Qwen2.5-1.5B \
    --analysis-type hnc_comparison

# Temperature comparison analysis
uv run python scripts/analyze_results.py \
    --category aime25_Qwen2.5-1.5B \
    --analysis-type temperature_comparison

# Difficulty-temperature analysis
uv run python scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-1.5B \
    --approach bon \
    --output-dir analysis_output

# Run all difficulty-temperature analyses
bash analyze_difficulty_temperature_all.sh
```

### Dataset Preprocessing

Preprocessing is required for raw datasets before analysis. It extracts predictions into standardized fields.

```bash
# Preprocess all experiments (2-4 hours)
bash preprocess_all.sh

# Preprocess by category
uv run python scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B \
    --push-to-hub \
    --validate

# Preprocess single experiment
uv run python scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --push-to-hub

# Local testing (no Hub push)
uv run python scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --output-dir /tmp/preprocessed-test
```

After preprocessing, update `configs/registry.yaml` to use the preprocessed Hub paths (prefix: `preprocessed-`).

## Architecture

### Auto-Discovery System

The project's key architectural innovation is automatic metadata discovery from Hub dataset subset names:

1. **Parser (`analysis/parser.py`)**: Parses subset names to extract temperature, seed, strategy
   - Default strategy: `{source}_{dataset}--T-{temp}--...--seed-{seed}--...`
   - HNC strategy: `{source}_{dataset}--temps_{t1}_{t2}...--...--seed-{seed}--...`

2. **Discovery (`analysis/discovery.py`)**: Auto-discovers experiment configs from Hub
   - Returns `ExperimentConfig` with all seeds, temperatures, subsets discovered
   - No manual config needed beyond Hub path

3. **Registry (`configs/registry.yaml`)**: Only stores Hub paths grouped by category
   - Seeds, temperatures auto-discovered at runtime
   - Minimal, declarative configuration

### Module Structure

```
├── analysis/               # Core analysis modules
│   ├── parser.py          # Subset name parsing
│   ├── discovery.py       # Auto-discovery from Hub
│   ├── datasets.py        # Dataset loading (by temp, seed)
│   ├── core.py            # Answer evaluation (uses math_verify)
│   ├── metrics.py         # Pass@k, accuracy calculations
│   ├── difficulty.py      # Problem difficulty stratification
│   ├── difficulty_temperature.py  # Difficulty × temperature analysis
│   ├── preprocessing.py   # Dataset preprocessing
│   ├── visualization.py   # Plotting utilities
│   └── comparative_analysis.py  # Cross-experiment comparison
├── configs/
│   ├── registry.yaml      # Hub paths by category
│   └── schemas.py         # Config dataclasses
├── scripts/               # Executable analysis scripts
│   ├── analyze_results.py  # Main analysis CLI
│   ├── analyze_difficulty_temperature.py
│   ├── preprocess_dataset.py
│   └── compare_baselines.py
└── legacy/                # Old scripts (reference only, don't use)
```

### Key Concepts

**Approaches**: Search strategies used during inference
- `bon`: Best-of-N sampling
- `beam_search`: Beam search decoding
- `dvts`: Diverse verification tree search

**Strategies**: Temperature allocation strategies
- `default`: Single fixed temperature per experiment
- `hnc`: Heterogeneous N-sample Composition (multiple temps with ratios)

**Aggregation Methods**: How to select final answer from N samples
- `naive`: Select based on raw model scores
- `weighted`: Weight by inverse difficulty
- `maj`: Majority vote
- `completions`: Use first correct completion (for baselines)

**Difficulty Levels**: Problems stratified by baseline solve rate
- `easy`: ≥80% solve rate
- `medium`: 40-80% solve rate
- `hard`: <40% solve rate

### Data Flow

1. **Load**: `load_experiment_data_by_temperature()` → Dict[temp, Dict[seed, Dataset]]
2. **Evaluate**: `analyze_single_dataset()` → Dict[method, Dict[n, accuracy]]
3. **Aggregate**: `aggregate_across_seeds()` → Mean ± std across seeds
4. **Visualize**: `plot_scaling_curve()`, `plot_comparison()` → Figures + reports

## Development Patterns

### Adding New Analysis

1. Load data using auto-discovery:
```python
from analysis import discover_experiment, load_experiment_data_by_temperature

config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
datasets = load_experiment_data_by_temperature(config)  # {temp: {seed: Dataset}}
```

2. Use existing metrics functions from `analysis/metrics.py`
3. Create visualizations using `analysis/visualization.py` utilities

### Important: Never Hardcode Metadata

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

### Working with Temperature Data

For temperature comparison analyses, always use `load_experiment_data_by_temperature()`:
```python
datasets_by_temp = load_experiment_data_by_temperature(config)
for temp, seed_datasets in datasets_by_temp.items():
    for seed, dataset in seed_datasets.items():
        # Analyze this (temp, seed) combination
        metrics = analyze_single_dataset(dataset, config.approach, seed)
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

## Registry Categories

The `configs/registry.yaml` organizes experiments by:
- Dataset: `math500` (MATH-500), `aime25` (AIME 2025)
- Model: `Qwen2.5-1.5B`, `Qwen2.5-3B`
- Strategy: `_hnc` suffix for heterogeneous temperature, no suffix for default

Examples:
- `math500_Qwen2.5-1.5B_hnc`: MATH-500, 1.5B model, HNC strategy
- `math500_Qwen2.5-3B`: MATH-500, 3B model, default strategy
- `aime25_Qwen2.5-1.5B`: AIME25, 1.5B model, default strategy

## Output Conventions

Analysis scripts generate outputs in designated directories:
- `analysis_output/`: Default output for comparative analyses
- `exp/analysis_output-{DATASET}-{MODEL}-{APPROACH}-{TYPE}/`: Specific analyses

Generated artifacts:
- `*.png`: Matplotlib figures (scaling curves, comparisons, heatmaps)
- `*_report.md`: Markdown summary tables
- Console output: Human-readable results with statistical significance

## Notes

- This project is for **analysis only**, not for running inference experiments
- All datasets are hosted on HuggingFace Hub under the `ENSEONG` organization
- The `legacy/` directory contains old scripts for reference; don't use them for new work
- Preprocessed datasets have the `preprocessed-` prefix in their Hub names
