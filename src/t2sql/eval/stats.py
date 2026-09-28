"""Paired statistics for the ablation matrix (design §11).

Bootstrap CIs (1000 replicates, fixed seed) and exact two-sided McNemar on
discordant pairs. Pure functions — no gold, no I/O.
"""
from __future__ import annotations

import random
from math import comb


def _quantile(sorted_vals: list[float], q: float) -> float:
    idx = min(len(sorted_vals) - 1, max(0, round(q * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def bootstrap_ci(values: list[float], reps: int = 1000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float]:
    """Percentile CI for the mean of per-item scores (e.g. 0/1 correctness)."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n
                   for _ in range(reps))
    return (_quantile(means, alpha / 2), _quantile(means, 1 - alpha / 2))


def paired_delta_ci(a: list[float], b: list[float], reps: int = 1000,
                    seed: int = 0, alpha: float = 0.05
                    ) -> tuple[float, float, float]:
    """CI for mean(b) - mean(a) using the SAME resampled indices (paired)."""
    if not a or len(a) != len(b):
        raise ValueError("paired inputs required")
    rng = random.Random(seed)
    n = len(a)
    deltas = sorted(
        sum(b[rng.randrange(n)] - a[rng.randrange(n)] for _ in range(n)) / n
        for _ in range(reps))
    point = sum(b) / n - sum(a) / n
    return (point, _quantile(deltas, alpha / 2), _quantile(deltas, 1 - alpha / 2))


def mcnemar_p(a: list[bool], b: list[bool]) -> float:
    """Exact two-sided McNemar (binomial on discordant pairs)."""
    if len(a) != len(b):
        raise ValueError("paired inputs required")
    a_only = sum(1 for x, y in zip(a, b) if x and not y)
    b_only = sum(1 for x, y in zip(a, b) if y and not x)
    n = a_only + b_only
    if n == 0:
        return 1.0
    k = min(a_only, b_only)
    p = sum(comb(n, j) for j in range(k + 1)) / 2 ** n * 2
    return min(1.0, p)