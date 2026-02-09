# Experiment Analysis Module (Auto-Discovery)

실험 결과 분석을 위한 모듈입니다. **Hub 데이터가 Single Source of Truth**입니다.
Seeds, temperatures 등 모든 메타데이터는 Hub에서 자동으로 발견됩니다.

## 간단한 사용 예제

### 5분 안에 시작하기

```bash
# 1. 사용 가능한 실험 목록 확인
uv run python scripts/analyze_results.py --list

# 2. 특정 실험 분석 (자동으로 seeds, temperatures 발견)
uv run python scripts/analyze_results.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon

# 3. 카테고리 전체 분석
uv run python scripts/analyze_results.py --category math500_Qwen2.5-1.5B_hnc

# 4. 온도별 비교 분석
uv run python scripts/analyze_results.py \
    --category math500_Qwen2.5-1.5B \
    --analysis-type temperature_comparison
```

**결과물**:
- `{model}-{approach}-scaling.png`: 성능 스케일링 커브
- `analysis_report.md`: 상세 통계 리포트 (평균 ± 표준편차)
- 콘솔 출력: 주요 메트릭 요약

### Python에서 사용하기

```python
from analysis import discover_experiment, load_experiment_data_by_temperature, analyze_single_dataset

# Hub에서 자동 발견 (seeds, temperatures 등)
config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
print(f"발견된 seeds: {config.seeds}")
print(f"발견된 temperatures: {config.temperatures}")

# 온도별로 데이터 로드
datasets_by_temp = load_experiment_data_by_temperature(config)

# 첫 번째 온도의 첫 번째 seed 분석
temp = config.temperatures[0]
seed = config.seeds[0]
dataset = datasets_by_temp[temp][seed]

# 분석 실행
metrics = analyze_single_dataset(dataset, config.approach, seed)
print(f"Naive@4 accuracy: {metrics['naive'][4]:.2%}")
print(f"Majority@16 accuracy: {metrics['maj'][16]:.2%}")
```

## 핵심 원칙

기존 방식에서는 registry.yaml에 seeds, temperatures를 수동으로 지정했지만,
이제는 Hub dataset의 subset 이름을 파싱하여 자동으로 발견합니다.

```python
# 기존 (수동 지정)
registry.yaml:
  seeds: [128, 192, 256]
  temperatures: [0.4, 0.8, 1.2, 1.6]

# 새 방식 (자동 발견)
config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
print(config.seeds)        # [0, 42, 64, 128, 192, 256] - 자동 발견!
print(config.temperatures) # [(0.4, 0.8, 1.2, 1.6), ...] - 자동 발견!
```

## 디렉토리 구조

```
tts_analysis/
├── configs/
│   ├── registry.yaml     # Hub 경로만 저장 (최소화)
│   └── schemas.py        # HubRegistry, 설정 클래스
├── analysis/
│   ├── parser.py         # 명명 규칙 파서
│   ├── discovery.py      # 자동 발견 로직
│   ├── datasets.py       # 데이터셋 로딩 (온도별 지원)
│   ├── core.py           # 평가 함수
│   ├── metrics.py        # 메트릭 계산
│   ├── difficulty.py     # 난이도 분석
│   └── visualization.py  # 시각화
├── scripts/
│   └── analyze_results.py
└── legacy/               # 기존 스크립트 (참조용)
```

## 빠른 시작

### 1. 자동 발견

```python
from analysis import discover_experiment, summarize_experiment

# 단일 실험 발견
config = discover_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
print(f"Model: {config.model}")           # Qwen2.5-1.5B-Instruct
print(f"Approach: {config.approach}")     # bon
print(f"Strategy: {config.strategy}")     # hnc
print(f"Seeds: {config.seeds}")           # [0, 42, 64, 128, 192, 256]
print(f"Temps: {config.temperatures}")    # [(0.4, 0.8, 1.2, 1.6), ...]

# 요약 정보
summary = summarize_experiment("ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon")
print(summary)
```

### 2. 데이터셋 로드

```python
from analysis import discover_experiment, load_experiment_data, load_experiment_data_by_temperature

config = discover_experiment("ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon")

# 방법 1: 기본 로드 (첫 번째 온도만)
datasets = load_experiment_data(config)  # {seed: Dataset}

# 방법 2: 온도별 로드 (권장 - 온도 비교 분석용)
datasets_by_temp = load_experiment_data_by_temperature(config)
# {temperature: {seed: Dataset}}
# 예: datasets_by_temp[0.4][42] = T=0.4, seed=42의 Dataset

for temp, seed_datasets in datasets_by_temp.items():
    print(f"Temperature {temp}:")
    for seed, dataset in seed_datasets.items():
        print(f"  Seed {seed}: {len(dataset['train'])} samples")
```

### 3. 분석 실행

```python
from analysis import analyze_single_dataset, analyze_pass_at_k

# 단일 데이터셋 분석
metrics = analyze_single_dataset(dataset, "hnc-bon", seed=128)
# {'naive': {1: 0.45, 2: 0.52, ...}, 'weighted': {...}, 'maj': {...}}

# Pass@k 분석
pass_at_k = analyze_pass_at_k(dataset, "hnc-bon", seed=128)
# {1: 0.75, 2: 0.85, 4: 0.92, ...}
```

## CLI 사용법

### 실험 목록 확인

```bash
uv run python scripts/analyze_results.py --list
```

출력:
```
[math500_hnc]
  - ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon
  - ENSEONG/hnc-Qwen2.5-1.5B-Instruct-beam_search
  - ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts

[math500_default]
  - ENSEONG/default-Qwen2.5-1.5B-Instruct-bon
  ...
```

### 단일 실험 분석

```bash
uv run python scripts/analyze_results.py ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon
```

### 카테고리 분석

```bash
# MATH-500 HNC 실험들 분석
uv run python scripts/analyze_results.py --category math500_hnc

# HNC vs Default 비교
uv run python scripts/analyze_results.py \
    --category math500_hnc,math500_default \
    --analysis-type hnc_comparison
```

### 분석 타입

| 타입 | 설명 |
|------|------|
| `summary` | 기본 정확도 요약 (기본값) - 온도별 결과 포함 |
| `hnc_comparison` | HNC vs Default 비교 |
| `temperature_comparison` | 온도별 비교 (T=0.4 vs T=0.8 등) |
| `model_comparison` | 모델 크기 비교 (1.5B vs 3B 등) |

### 온도 비교 분석 (Temperature Comparison)

여러 온도에서의 실험 결과를 비교합니다:

```bash
# AIME25 데이터셋의 온도별 비교
uv run python scripts/analyze_results.py \
    --category aime25_1.5B \
    --analysis-type temperature_comparison

# 특정 실험의 온도 비교
uv run python scripts/analyze_results.py \
    ENSEONG/default-aime25-Qwen2.5-1.5B-Instruct-bon \
    --analysis-type temperature_comparison
```

출력물:
- `{model}-{approach}-temp_scaling.png`: 온도별 스케일링 커브
- `{model}-{approach}-{method}-temp_comparison.png`: 메서드별 온도 비교
- `{model}-{approach}-pass_at_k_by_temp.png`: Pass@k 온도별 비교
- `temperature_comparison_report.md`: 상세 마크다운 리포트

리포트 예시:
```markdown
## 1.5B - BON

### NAIVE

| n | T0.4 | T0.8 |
|---|------|------|
| 1 | 0.4500±0.012 | 0.4200±0.015 |
| 2 | 0.5200±0.010 | 0.4900±0.013 |
...
```

### 옵션

```bash
# 플롯 없이 리포트만
uv run python scripts/analyze_results.py --category math500_hnc --no-plots

# 상세 출력
uv run python scripts/analyze_results.py --category math500_hnc -v

# 출력 디렉토리 지정
uv run python scripts/analyze_results.py --category math500_hnc --output-dir ./my_output

# 특정 온도만 분석
uv run python scripts/analyze_results.py --category math500_default --temperature 0.8
```

## Registry 구조

registry.yaml은 이제 Hub 경로만 저장합니다:

```yaml
hub_paths:
  math500_hnc:
    - ENSEONG/hnc-Qwen2.5-1.5B-Instruct-bon
    - ENSEONG/hnc-Qwen2.5-1.5B-Instruct-beam_search
    - ENSEONG/hnc-Qwen2.5-1.5B-Instruct-dvts

  math500_default:
    - ENSEONG/default-Qwen2.5-1.5B-Instruct-bon
    - ...
```

## 명명 규칙

Subset 이름에서 자동으로 메타데이터를 추출합니다:

**Default 전략** (단일 온도):
```
HuggingFaceH4_MATH-500--T-0.8--top_p-1.0--n-64--seed-42--agg_strategy-last
                       ^^^^^                    ^^^^^^^
                       temp                     seed
```

**HNC 전략** (다중 온도):
```
HuggingFaceH4_MATH-500--temps_0.4_0.8_1.2_1.6__r_0.25_0.25_0.25_0.25--...--seed-128--...
                        ^^^^^^^^^^^^^^^^^^^^^^^                           ^^^^^^^^
                        temperatures + ratios                             seed
```

## 모듈 API

### 자동 발견

```python
from analysis import (
    discover_experiment,      # Hub에서 설정 자동 발견
    ExperimentConfig,         # 발견된 설정 클래스
    create_registry_from_hub_paths,  # 여러 경로 한번에 발견
)
```

### 파서

```python
from analysis import (
    parse_subset_name,        # Subset 이름 파싱
    SubsetInfo,               # 파싱 결과 클래스
    infer_approach_from_hub_path,
    infer_model_from_hub_path,
)
```

### 데이터셋

```python
from analysis import (
    load_experiment_data,              # 기본 로드 (첫 번째 온도)
    load_experiment_data_by_temperature,  # 온도별 로드 (권장)
    load_all_experiment_data,          # 모든 (온도, seed) 조합 로드
    load_from_hub_path,                # Hub 경로에서 직접 로드
    summarize_experiment,              # 실험 요약 정보
)
```

### 분석

```python
from analysis import (
    analyze_single_dataset,   # 단일 데이터셋 분석
    analyze_pass_at_k,        # Pass@k 분석
    evaluate_answer,          # 답변 평가
)
```

## 새 실험 추가

registry.yaml에 Hub 경로만 추가하면 됩니다:

```yaml
hub_paths:
  my_experiments:
    - MY_ORG/my-experiment-bon
    - MY_ORG/my-experiment-beam_search
```

Seeds, temperatures 등은 **자동으로 발견**됩니다!
