#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Scoring module for computing prediction fields from raw completions.

Ported from hotncold repository (sal/utils/math.py and sal/utils/score.py).
Computes pred_naive@n, pred_weighted@n, pred_maj@n, and pass@k fields
from raw completions and scores.

Usage:
    from analysis.scoring import ScoringConfig, score_dataset, score_pass_at_k

    config = ScoringConfig(n=64, num_proc=4)
    dataset = score_dataset(dataset, config)
    dataset = score_pass_at_k(dataset, config)
"""

import logging
import math
import signal
from collections import defaultdict
from dataclasses import dataclass
from multiprocessing import Manager
from typing import Any, Dict, List, Literal

import numpy as np
from datasets import Dataset
from latex2sympy2_extended import latex2sympy
from math_verify import parse, verify
from sympy import latex, simplify
from tqdm import tqdm

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ScoringConfig:
    """Configuration for scoring pipeline.

    Attributes:
        n: Maximum N value for scoring (generates [1, 2, 4, ..., n])
        num_proc: Number of processes for parallel processing
    """
    n: int = 64
    num_proc: int = 4


# =============================================================================
# Canonical form computation (from hotncold/src/sal/utils/math.py)
# =============================================================================

class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException


_manager = None
_shared_cache = None


def _get_shared_cache():
    """Lazily initialize the multiprocessing Manager and shared cache."""
    global _manager, _shared_cache
    if _shared_cache is None:
        _manager = Manager()
        _shared_cache = _manager.dict()
    return _shared_cache


def memoized_canonical_form(expression: str, timeout_seconds: int = 3) -> str:
    """Compute a canonical form for a mathematical expression using sympy.

    Uses a shared cache across processes for memoization.
    Ensures symmetric comparisons (A == B iff B == A) and performance through caching.

    Args:
        expression: A LaTeX-formatted mathematical expression.
        timeout_seconds: Timeout duration in seconds.

    Returns:
        The canonical form of the expression or the original expression as fallback.
    """
    shared_cache = _get_shared_cache()
    if expression in shared_cache:
        return shared_cache[expression]

    try:
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)

        parsed_expr = latex2sympy(expression)
        simplified_expr = simplify(parsed_expr)

        signal.alarm(0)

        canonical_form = latex(simplified_expr)
        shared_cache[expression] = canonical_form
        return canonical_form
    except TimeoutException:
        return expression
    except Exception:
        return expression
    finally:
        signal.alarm(0)


# =============================================================================
# Answer parsing and prediction computation
# =============================================================================

def safe_parse_answer(text: str) -> str:
    """Safely parse answer from completion text.

    Splits by '\\n\\n' and parses only the last chunk (final step).

    Args:
        text: Completion text to parse

    Returns:
        Extracted answer string, or empty string if parsing fails
    """
    try:
        chunks = text.split("\n\n")
        last_chunk = chunks[-1].strip()

        if not last_chunk:
            return ""

        result = parse(last_chunk)

        if isinstance(result, list) and len(result) >= 2:
            return result[1]
        elif isinstance(result, list) and len(result) == 1:
            return str(result[0])
        else:
            return ""
    except Exception as e:
        logger.debug(f"Failed to parse answer from text: {text[:100]}... Error: {e}")
        return ""


def subsample_completions(x: Dict[str, List[Any]], n: int) -> Dict[str, List[Any]]:
    """Subsample first n completions and their scores.

    Takes the first n samples, as completions are ordered in groups.
    Groups must not be broken up for valid comparison at smaller n.
    """
    completions = x["completions"]
    agg_scores = x["agg_scores"]
    if len(completions) != len(agg_scores):
        raise ValueError(
            f"The number of completions and agg_scores should be the same. "
            f"Got {len(completions)} completions and {len(agg_scores)} agg_scores."
        )

    return {
        f"completions@{n}": completions[:n],
        f"agg_scores@{n}": agg_scores[:n],
    }


def extract_completion_answers(
    x: Dict[str, List[Any]], n: int | None = None
) -> Dict[str, List[str]]:
    """Extract parsed answers from completions."""
    if n is None:
        return {"preds": [safe_parse_answer(p) for p in x["completions"]]}
    else:
        return {f"preds@{n}": [safe_parse_answer(p) for p in x[f"completions@{n}"]]}


def compute_naive_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, str]:
    """Compute naive prediction: highest-scoring answer."""
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    preds = [
        (p, s) for p, s in sorted(zip(preds, scores), key=lambda x: x[1], reverse=True)
    ]
    return {f"pred_naive@{n}": "\\boxed{" + preds[0][0] + "}"}


def compute_weighted_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, str]:
    """Compute weighted prediction: answer with largest summed score across equivalence group."""
    preds = x[f"preds@{n}"]
    scores = x[f"agg_scores@{n}"]
    return {
        f"pred_weighted@{n}": "\\boxed{"
        + find_answer_with_largest_sum(preds, scores)
        + "}"
    }


def compute_maj_pred(x: Dict[str, List[Any]], n: int) -> Dict[str, str]:
    """Compute majority prediction: most frequent answer across equivalence group."""
    preds = x[f"preds@{n}"]
    return {f"pred_maj@{n}": "\\boxed{" + find_majority_answer(preds) + "}"}


# =============================================================================
# Answer grouping by mathematical equivalence
# =============================================================================

def find_answer_with_largest_sum(answers: List[str], scores: List[float]) -> str:
    """Group answers by canonical form and find group with largest summed score.

    Args:
        answers: List of answer strings to be grouped.
        scores: List of scores corresponding to each answer.

    Returns:
        The answer string representing the group with the largest sum of scores.
    """
    if len(answers) == 0 or len(scores) == 0:
        raise ValueError("answers and scores cannot be empty")

    canonical_groups = defaultdict(float)
    canonical_to_original = {}

    for answer, score in zip(answers, scores):
        canonical_form = memoized_canonical_form(answer)
        canonical_groups[canonical_form] += score
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    max_canonical = max(canonical_groups, key=canonical_groups.get)
    return canonical_to_original[max_canonical]


def find_majority_answer(answers: List[str]) -> str:
    """Group answers by canonical form and find group with most elements.

    In case of a tie, returns the first occurring group with the largest size.

    Args:
        answers: List of answer strings to be grouped.

    Returns:
        The answer string representing the most frequent group.
    """
    if len(answers) == 0:
        raise ValueError("answers cannot be empty")

    canonical_groups = defaultdict(int)
    canonical_to_original = {}

    for answer in answers:
        canonical_form = memoized_canonical_form(answer)
        canonical_groups[canonical_form] += 1
        if canonical_form not in canonical_to_original:
            canonical_to_original[canonical_form] = answer

    max_count = max(canonical_groups.values())
    for canonical_form, count in canonical_groups.items():
        if count == max_count:
            return canonical_to_original[canonical_form]


# =============================================================================
# Pass@k computation
# =============================================================================

def pass_at_k(n: int, c: int, k: int) -> float:
    """Numerically stable unbiased estimate of pass@k.

    From OpenAI's Codex paper: https://arxiv.org/abs/2107.03374

    Args:
        n: Total number of samples
        c: Number of correct samples
        k: k in pass@k

    Returns:
        Unbiased estimate of pass@k
    """
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def compute_pass_at_k(x, k):
    """Compute pass@k for predictions using canonical forms.

    Args:
        x: Dictionary containing "preds" (list of predictions) and "answer" (correct answer).
        k: The cutoff for pass@k.

    Returns:
        Dictionary containing pass@k result.
    """
    n = len(x["preds"])
    if n == 0:
        raise ValueError("No predictions found")
    if x["answer"] == "":
        raise ValueError("Answer is empty")

    gold_answer = parse("\\boxed{" + x["answer"] + "}")
    c = sum(verify(gold_answer, parse("\\boxed{" + pred + "}")) for pred in x["preds"])

    return {f"pass@{k}": pass_at_k(n, c, k)}


# =============================================================================
# Difficulty level computation
# =============================================================================

def compute_level(
    x, metric: Literal["mean_score", "pass@1"], name: str, quintiles: List[float]
) -> Dict[str, int]:
    """Compute difficulty level (1-5) based on metric and quintiles.

    Easier problems have higher metric values, so levels are reversed
    (1 is easiest, 5 is hardest).
    """
    if x[metric] < quintiles[0]:
        return {f"level_{name}": 5}
    elif x[metric] < quintiles[1]:
        return {f"level_{name}": 4}
    elif x[metric] < quintiles[2]:
        return {f"level_{name}": 3}
    elif x[metric] < quintiles[3]:
        return {f"level_{name}": 2}
    else:
        return {f"level_{name}": 1}


# =============================================================================
# Score aggregation
# =============================================================================

def aggregate_scores(
    scores: list[float], agg_strategy: Literal["min", "prod", "last"]
) -> float:
    """Aggregate per-step scores into a single score.

    Args:
        scores: List of per-step scores for a completion.
        agg_strategy: Aggregation strategy ("min", "prod", "last").

    Returns:
        Aggregated score.
    """
    if agg_strategy == "min":
        return min(scores)
    elif agg_strategy == "prod":
        return math.prod(scores)
    elif agg_strategy == "last":
        return scores[-1]
    else:
        raise ValueError(f"Invalid aggregation strategy: {agg_strategy}")


# =============================================================================
# Dataset scoring pipeline
# =============================================================================

def score_dataset(
    dataset: Dataset,
    config: ScoringConfig,
    agg_strategy: Literal["min", "prod", "last"] = "last",
    verbose: bool = True,
) -> Dataset:
    """Score a dataset by computing pred_naive@n, pred_weighted@n, pred_maj@n.

    Args:
        dataset: Dataset with "completions" and "scores" fields.
        config: Scoring configuration (n, num_proc).
        agg_strategy: How to aggregate per-step scores.
        verbose: Show progress.

    Returns:
        Dataset with pred_*@n fields added and temporary fields removed.
    """
    # Aggregate per-step scores into single scores per completion
    dataset = dataset.map(
        lambda x: {"agg_scores": [aggregate_scores(s, agg_strategy) for s in x["scores"]]}
    )

    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for n in tqdm(subsets, desc="Computing predictions", disable=not verbose):
        dataset = dataset.map(
            subsample_completions,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Subsample {n}",
        )
        dataset = dataset.map(
            extract_completion_answers,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Extract answers {n}",
        )
        dataset = dataset.map(
            compute_weighted_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute weighted pred {n}",
        )
        dataset = dataset.map(
            compute_maj_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute majority pred {n}",
        )
        dataset = dataset.map(
            compute_naive_pred,
            fn_kwargs={"n": n},
            num_proc=config.num_proc,
            desc=f"Compute naive pred {n}",
        )
        # Remove temporary columns to keep dataset lean
        dataset = dataset.remove_columns(
            [f"completions@{n}", f"agg_scores@{n}", f"preds@{n}"]
        )
    return dataset


def score_pass_at_k(
    dataset: Dataset,
    config: ScoringConfig,
    verbose: bool = True,
) -> Dataset:
    """Compute pass@k metrics for a dataset.

    Args:
        dataset: Dataset with "completions" and "answer" fields.
        config: Scoring configuration (n, num_proc).
        verbose: Show progress.

    Returns:
        Dataset with pass@k fields added.
    """
    dataset = dataset.map(
        extract_completion_answers,
        fn_kwargs={"n": None},
        num_proc=config.num_proc,
        desc="Extract answers for Pass@k",
    )

    subsets = [2**i for i in range(config.n) if 2**i <= config.n]
    for k in tqdm(subsets, desc="Computing pass@k", disable=not verbose):
        dataset = dataset.map(
            compute_pass_at_k,
            fn_kwargs={"k": k},
            num_proc=config.num_proc,
            desc=f"Compute Pass@{k}",
        )
    return dataset
