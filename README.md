# TTS Analysis Tool

A test-time scaling (TTS) experiment analysis tool for mathematical reasoning tasks. Analyzes LLM outputs from various search strategies (Best-of-N, Beam Search, DVTS) across different temperatures and model sizes.

## Quick Start

### Installation

```bash
# Install dependencies
uv sync
```

### Essential Commands

```bash
# 1. List all available experiments
uv run python scripts/analyze_results.py --list

# 2. Analyze a specific experiment (auto-discovers metadata)
uv run python scripts/analyze_results.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon

# 3. Analyze by category
uv run python scripts/analyze_results.py --category math500_Qwen2.5-1.5B

# 4. Temperature comparison analysis
uv run python scripts/analyze_results.py \
    --category math500_Qwen2.5-1.5B \
    --analysis-type temperature_comparison
```

**Expected outputs:**
- `{model}-{approach}-scaling.png`: Performance scaling curves
- `analysis_report.md`: Detailed statistics (mean ± std)
- Console output: Key metrics summary

### Using in Python

```python
from analysis import discover_experiment, load_experiment_data_by_temperature, analyze_single_dataset

# Auto-discover from Hub (seeds, temperatures, etc.)
config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
print(f"Discovered seeds: {config.seeds}")
print(f"Discovered temperatures: {config.temperatures}")

# Load data by temperature
datasets_by_temp = load_experiment_data_by_temperature(config)

# Analyze first temperature's first seed
temp = config.temperatures[0]
seed = config.seeds[0]
dataset = datasets_by_temp[temp][seed]

metrics = analyze_single_dataset(dataset, config.approach, seed)
print(f"Naive@4 accuracy: {metrics['naive'][4]:.2%}")
```

## Core Concepts

### Auto-Discovery System

**Hub data is the Single Source of Truth.** All experiment metadata (seeds, temperatures, model configs) is automatically discovered from HuggingFace Hub dataset subset names, not manually specified in config files.

```python
# ❌ OLD: Manual specification
registry.yaml:
  seeds: [128, 192, 256]
  temperatures: [0.4, 0.8, 1.2, 1.6]

# ✅ NEW: Auto-discovery
config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
print(config.seeds)        # [0, 42, 64, 128, 192, 256] - auto-discovered!
print(config.temperatures) # [(0.4, 0.8, 1.2, 1.6), ...] - auto-discovered!
```

### Approaches

Search strategies used during inference:
- **bon**: Best-of-N sampling
- **beam_search**: Beam search decoding
- **dvts**: Diverse verification tree search

### Strategies

Temperature allocation strategies:
- **default**: Single fixed temperature per experiment
- **hnc**: Heterogeneous N-sample Composition (multiple temps with ratios)
- **early**: Early difficulty estimation (temperature range)

### Aggregation Methods

How to select final answer from N samples:
- **naive**: Select based on raw model scores
- **weighted**: Weight by inverse difficulty
- **maj**: Majority vote
- **completions**: Use first correct completion (for baselines)

### Preprocessing Requirement

**Raw datasets must be preprocessed before analysis.** Preprocessing extracts predictions into standardized `pred_*` and `is_correct_*` fields. Preprocessed datasets have the `preprocessed-` prefix in their Hub names.

## Commands

### Analysis Commands

#### List Experiments

```bash
uv run python scripts/analyze_results.py --list
```

Output shows experiments grouped by category from `configs/registry.yaml`.

#### Analyze Single Experiment

```bash
uv run python scripts/analyze_results.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon
```

Automatically discovers all metadata (seeds, temperatures) from Hub.

#### Analyze by Category

```bash
# Single category
uv run python scripts/analyze_results.py --category math500_Qwen2.5-1.5B

# Multiple categories (comparison)
uv run python scripts/analyze_results.py \
    --category math500_Qwen2.5-1.5B_hnc,math500_Qwen2.5-1.5B \
    --analysis-type hnc_comparison
```

#### Analysis Types

| Type | Description |
|------|-------------|
| `summary` | Basic accuracy summary (default) |
| `hnc_comparison` | HNC vs Default strategy comparison |
| `temperature_comparison` | Compare across temperatures (T=0.4 vs T=0.8 etc.) |
| `model_comparison` | Compare model sizes (1.5B vs 3B etc.) |

#### Temperature Comparison

```bash
uv run python scripts/analyze_results.py \
    --category aime25_Qwen2.5-1.5B \
    --analysis-type temperature_comparison
```

Outputs:
- `{model}-{approach}-temp_scaling.png`: Temperature-wise scaling curves
- `{model}-{approach}-{method}-temp_comparison.png`: Method-specific temperature comparison
- `temperature_comparison_report.md`: Detailed markdown report

#### Difficulty-Temperature Analysis

```bash
uv run python scripts/analyze_difficulty_temperature.py \
    --category math500_Qwen2.5-1.5B \
    --approach bon \
    --output-dir analysis_output

# Run all difficulty-temperature analyses
bash analyze_difficulty_temperature_all.sh
```

#### Common Options

```bash
# Disable plots (report only)
--no-plots

# Verbose output
-v, --verbose

# Custom output directory
--output-dir ./my_output

# Analyze specific temperature only
--temperature 0.8
```

### Preprocessing Commands

#### Preprocess All Experiments

```bash
# Preprocess everything (2-4 hours)
bash preprocess_all.sh
```

#### Preprocess by Category

```bash
uv run python scripts/preprocess_dataset.py \
    --category math500_Qwen2.5-1.5B \
    --push-to-hub \
    --validate
```

#### Preprocess Single Experiment

```bash
# Push to Hub
uv run python scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --push-to-hub

# Local testing (no Hub push)
uv run python scripts/preprocess_dataset.py \
    --hub-path ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon \
    --output-dir /tmp/preprocessed-test
```

#### Preprocessing Options

- `--push-to-hub`: Upload preprocessed dataset to HuggingFace Hub
- `--validate`: Validate preprocessing correctness (recommended)
- `--force`: Force reprocessing even if already preprocessed
- `--subsets`: Process specific subsets only

#### After Preprocessing: Update Registry

Update `configs/registry.yaml` to use preprocessed paths:

```yaml
# Before
math500_Qwen2.5-1.5B:
  - ENSEONG/default-MATH-500-Qwen2.5-1.5B-Instruct-bon

# After
math500_Qwen2.5-1.5B:
  - ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-bon
```

Or use sed to update all paths:
```bash
sed -i 's|ENSEONG/|ENSEONG/preprocessed-|g' configs/registry.yaml
```

#### Verify Preprocessing

```bash
# Check preprocessing status
uv run python -c "
from analysis.preprocessing import get_preprocessing_stats
from datasets import load_dataset

dataset = load_dataset('ENSEONG/preprocessed-default-MATH-500-Qwen2.5-1.5B-Instruct-bon',
                       'HuggingFaceH4_MATH-500--T-0.4--top_p-1.0--n-64--seed-42--agg_strategy-last')
stats = get_preprocessing_stats(dataset['train'])
print(stats)
"

# List all preprocessed datasets in your org
huggingface-cli repo list --organization ENSEONG | grep preprocessed
```

## Architecture

### Directory Structure

```
tts_analysis/
├── configs/
│   ├── registry.yaml     # Hub paths only (minimal config)
│   └── schemas.py        # Configuration dataclasses
├── analysis/
│   ├── parser.py         # Subset name parsing
│   ├── discovery.py      # Auto-discovery from Hub
│   ├── datasets.py       # Dataset loading (by temp, seed)
│   ├── core.py           # Answer evaluation (math_verify)
│   ├── metrics.py        # Pass@k, accuracy calculations
│   ├── difficulty.py     # Problem difficulty stratification
│   ├── difficulty_temperature.py  # Difficulty × temperature analysis
│   ├── preprocessing.py  # Dataset preprocessing
│   ├── visualization.py  # Plotting utilities
│   └── comparative_analysis.py  # Cross-experiment comparison
├── scripts/
│   ├── analyze_results.py  # Main analysis CLI
│   ├── analyze_difficulty_temperature.py
│   ├── preprocess_dataset.py
│   └── compare_baselines.py
└── legacy/               # Old scripts (reference only, don't use)
```

### Module Responsibilities

- **parser.py**: Parses subset names to extract temperature, seed, strategy
- **discovery.py**: Auto-discovers experiment configs from Hub (returns `ExperimentConfig`)
- **datasets.py**: Loads datasets by temperature and seed
- **core.py**: Evaluates answers using `math_verify` for mathematical equivalence
- **metrics.py**: Computes pass@k and accuracy metrics
- **difficulty.py**: Stratifies problems by baseline solve rate (easy/medium/hard)
- **visualization.py**: Creates plots (scaling curves, comparisons, heatmaps)
- **preprocessing.py**: Extracts predictions into standardized fields

### Data Flow

1. **Load**: `load_experiment_data_by_temperature()` → `Dict[temp, Dict[seed, Dataset]]`
2. **Evaluate**: `analyze_single_dataset()` → `Dict[method, Dict[n, accuracy]]`
3. **Aggregate**: `aggregate_across_seeds()` → Mean ± std across seeds
4. **Visualize**: `plot_scaling_curve()`, `plot_comparison()` → Figures + reports

## Registry Structure

`configs/registry.yaml` organizes experiments by category:

**Format**: `{dataset}_{model}[_{strategy}]`

**Examples**:
- `math500_Qwen2.5-1.5B`: MATH-500, 1.5B model, default strategy
- `math500_Qwen2.5-1.5B_hnc`: MATH-500, 1.5B model, HNC strategy
- `aime25_Qwen2.5-3B`: AIME25, 3B model, default strategy

The registry contains **only Hub paths**. All metadata (seeds, temperatures) is auto-discovered at runtime.

### Adding New Experiments

Simply add the Hub path to `registry.yaml`:

```yaml
hub_paths:
  my_experiments:
    - MY_ORG/my-experiment-bon
    - MY_ORG/my-experiment-beam_search
```

Seeds, temperatures, and other metadata are **automatically discovered** from subset names.

## Troubleshooting

### Error: "Dataset is not preprocessed"

The dataset hasn't been preprocessed yet. Run:

```bash
uv run python scripts/preprocess_dataset.py \
    --hub-path <YOUR_DATASET> \
    --push-to-hub
```

### Error: "No module named 'datasets'"

Install dependencies:

```bash
uv sync
```

### Hub Authentication Required

```bash
huggingface-cli login
```

### Script Appears Frozen

Use `--verbose` flag to see progress:

```bash
uv run python scripts/analyze_results.py --category math500_Qwen2.5-3B --verbose
```

### Out of Memory During Preprocessing

Process smaller batches (single experiments instead of categories):

```bash
uv run python scripts/preprocess_dataset.py \
    --hub-path ENSEONG/single-experiment \
    --push-to-hub
```

## Development

For AI assistant context and development patterns, see [CLAUDE.md](CLAUDE.md).

### Contributing

This project uses:
- **uv** for package management (always use `uv run python`)
- **Git** for version control
- **HuggingFace Hub** for dataset storage

When contributing:
1. Never hardcode metadata (always use auto-discovery)
2. Follow existing patterns in `analysis/` modules
3. Use `--help` to document new CLI options
4. Test with both verbose and silent modes

## Notes

- This project is for **analysis only**, not for running inference experiments
- All datasets are hosted on HuggingFace Hub under the `ENSEONG` organization
- The `legacy/` directory contains old scripts for reference; don't use them for new work
- Preprocessed datasets have the `preprocessed-` prefix in their Hub names
