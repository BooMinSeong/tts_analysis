# Early Difficulty Estimation Dataset - Analysis Guide

## Overview

This guide explains how to analyze the `preprocessed-score-early-MATH-500-Qwen2.5-3B-Instruct-bon` dataset using the existing analysis infrastructure.

## Dataset Information

- **Hub Path**: `ENSEONG/preprocessed-score-early-MATH-500-Qwen2.5-3B-Instruct-bon`
- **Subset Name**: `temp-low-0.1-high-0.8`
- **Strategy**: `early` (temperature range: 0.1 to 0.8)
- **Problems**: 500 (MATH-500 dataset)
- **Completions**: 64 per problem
- **Seeds**: 1 (seed=0)
- **Model**: Qwen2.5-3B-Instruct
- **Approach**: Best-of-N (bon)

## Supported Analyses

### 1. General Summary Analysis ✅

**Command**:
```bash
uv run python scripts/analyze_results.py \
    --category math500-early_Qwen2.5-3B \
    --analysis-type summary \
    --output-dir analysis_output-early-summary \
    --verbose
```

**Outputs**:
- `analysis_output-early-summary/summary_report.md`: Markdown report with accuracy metrics
- Console output with scaling curves and pass@k statistics

**Results**:
```
Model: Qwen2.5-3B-Instruct
Approach: bon, Strategy: early
Accuracy: 0.7760 ± 0.0000 (single seed, no std)
```

### 2. Custom Analysis Script ✅

A demonstration script is provided for detailed analysis:

**Command**:
```bash
uv run python scripts/analyze_early_dataset.py
```

**Outputs**:
```
Method               n          Accuracy
--------------------------------------------------
maj                  1          0.6420
maj                  2          0.6420
maj                  4          0.6660
maj                  8          0.6840
maj                  16         0.7060
maj                  32         0.7080
maj                  64         0.7220
naive                1          0.6420
naive                2          0.6680
naive                4          0.7100
naive                8          0.7460
naive                16         0.7720
naive                32         0.7680
naive                64         0.7760
weighted             1          0.6420
weighted             2          0.6680
weighted             4          0.7080
weighted             8          0.7260
weighted             16         0.7460
weighted             32         0.7460
weighted             64         0.7520
```

**Key Insights**:
- `naive` method performs best at n=64 (77.6% accuracy)
- `weighted` method shows similar performance at lower n
- `maj` (majority vote) is more conservative but consistent

### 3. Difficulty-Temperature Analysis ❌

**Not applicable** - The difficulty-temperature comparison script requires multiple temperature configurations. The early dataset has only one configuration (the low-high temperature range).

**Error**:
```
Error: No single-temperature experiments found (HNC not supported)
```

**Reason**: The dataset uses `strategy="early"` with a temperature tuple `(0.1, 0.8)`, which represents a range, not individual temperatures to compare.

## Technical Implementation

### Parser Extension

The parser was extended to support the early subset naming convention:

**Format**: `temp-low-{low_temp}-high-{high_temp}`
**Example**: `temp-low-0.1-high-0.8`

```python
# In analysis/parser.py
if name.startswith("temp-low-"):
    match = re.match(r"temp-low-([\d.]+)-high-([\d.]+)", name)
    if match:
        low_temp = float(match.group(1))
        high_temp = float(match.group(2))
        info.strategy = "early"
        info.temperatures = [low_temp, high_temp]
        info.seed = 0
        return info
```

### Discovery System Updates

All discovery methods were updated to handle the `early` strategy:

- `ExperimentConfig.temperatures`: Returns `[(0.1, 0.8)]` for early datasets
- `ExperimentConfig.group_by_temperature()`: Groups by temperature tuple
- `ExperimentConfig.filter_by_temperature()`: Filters by temperature tuple
- `ExperimentConfig.get_subset_name()`: Matches temperature tuples correctly

### Auto-Discovery Verification

```python
from analysis import discover_experiment

config = discover_experiment('ENSEONG/preprocessed-score-early-MATH-500-Qwen2.5-3B-Instruct-bon')

print(f"Strategy: {config.strategy}")           # early
print(f"Temperatures: {config.temperatures}")   # [(0.1, 0.8)]
print(f"Seeds: {config.seeds}")                 # [0]
print(f"Model: {config.model}")                 # Qwen2.5-3B-Instruct
print(f"Approach: {config.approach}")           # bon
```

## Comparison with Other Strategies

### Default Strategy
- **Format**: `{source}_{dataset}--T-{temp}--...--seed-{seed}--...`
- **Example**: `HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--seed-42--...`
- **Temperature**: Single float value
- **Multiple configs**: Different temperatures (0.4, 0.8, 1.2, 1.6)
- **Temperature comparison**: ✅ Supported

### HNC Strategy
- **Format**: `{source}_{dataset}--temps_{t1}_{t2}...--...--seed-{seed}--...`
- **Example**: `HuggingFaceH4_MATH-500--temps_0.4_0.8_1.2_1.6__r_0.25_0.25_0.25_0.25--...`
- **Temperature**: Tuple of floats with ratios
- **Multiple configs**: Single config with mixed temperatures
- **Temperature comparison**: ❌ Not supported

### Early Strategy
- **Format**: `temp-low-{low}-high-{high}`
- **Example**: `temp-low-0.1-high-0.8`
- **Temperature**: Tuple of two floats (low, high)
- **Multiple configs**: Single config representing a range
- **Temperature comparison**: ❌ Not supported

## Files Modified

### Core Analysis Modules
1. `analysis/parser.py`:
   - Added early format detection
   - Updated `parse_subset_name()` to parse early format
   - Updated `infer_strategy_from_hub_path()` to detect "early"

2. `analysis/discovery.py`:
   - Updated `temperatures` property
   - Updated `group_by_temperature()`
   - Updated `filter_by_temperature()`
   - Updated `get_subset_name()`
   - Updated `get_all_subset_names_for_seed()`
   - Updated `iter_all_configs()`

### Configuration
3. `configs/registry.yaml`:
   - Added `math500-early_Qwen2.5-3B` category

### New Scripts
4. `scripts/analyze_early_dataset.py`:
   - Demonstration script for early dataset analysis

## Best Practices

1. **Always use auto-discovery**: Never hardcode temperatures or seeds
   ```python
   config = discover_experiment(hub_path)
   temps = config.temperatures  # Auto-discovered
   ```

2. **Use existing scripts**: The general analysis script works perfectly
   ```bash
   uv run python scripts/analyze_results.py --category math500-early_Qwen2.5-3B
   ```

3. **Understand single-seed limitations**: With only one seed, no standard deviation
   - Results show `± 0.0000` for std
   - No statistical significance testing possible

4. **Use appropriate analysis type**: Don't force temperature comparison when not applicable
   - Early dataset: Use `summary` analysis type
   - Default strategy: Use `temperature_comparison` analysis type

## Future Enhancements

If per-temperature analysis is needed for early datasets, consider:

1. **Virtual subset splitting**: Split the 64 completions by actual generation temperature
2. **Separate low/high analysis**: Analyze low-temp and high-temp completions separately
3. **Discovery extension**: Extend discovery system to support virtual subsets

This would require modifications to the preprocessing and discovery logic.
