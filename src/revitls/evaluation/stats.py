"""Small, dependency-free statistics helpers used by the evaluation scripts."""

import math
import random


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def exact_mcnemar_p(a: list[int], b: list[int]) -> float:
    """Two-sided exact McNemar p-value over paired binary outcomes."""
    if len(a) != len(b) or not a:
        return float("nan")
    only_a = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    only_b = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2.0 * tail)


def holm_adjust(pvals: list[float]) -> list[float]:
    """Holm-Bonferroni step-down adjusted p-values (NaNs pass through)."""
    idx = [i for i, p in enumerate(pvals) if not math.isnan(p)]
    order = sorted(idx, key=lambda i: pvals[i])
    m = len(order)
    adj = list(pvals)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_bootstrap_diff(
    a: list[int], b: list[int], n_boot: int = 10000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean difference (a - b) and its 95% percentile bootstrap interval over cases."""
    n = len(a)
    if n == 0 or n != len(b):
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b)]
    means = []
    for _ in range(n_boot):
        means.append(sum(diffs[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return sum(diffs) / n, means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]
