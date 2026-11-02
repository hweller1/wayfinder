"""Paired task bootstrap (percentile, resampling tasks) and a repo-cluster sensitivity version."""

import random
from collections import defaultdict
from collections.abc import Mapping, Sequence

SEED = 20261025
N_BOOT = 10_000


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs)


def _pct(sorted_vals: Sequence[float], q: float) -> float:
    i = q * (len(sorted_vals) - 1)
    lo = int(i)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (i - lo)


def bootstrap_ci(values: Sequence[float], n: int = N_BOOT, seed: int = SEED) -> tuple[float, float, float]:
    """`(mean, lo, hi)`, 95% percentile interval, resampling the items (tasks)."""
    rng = random.Random(seed)
    k = len(values)
    means = sorted(_mean([values[rng.randrange(k)] for _ in range(k)]) for _ in range(n))
    return _mean(values), _pct(means, 0.025), _pct(means, 0.975)


def cluster_bootstrap_ci(
    values_by_cluster: Mapping[str, Sequence[float]], n: int = N_BOOT, seed: int = SEED
) -> tuple[float, float, float]:
    """Resample clusters (repos), then every task in a drawn cluster; mean over all drawn tasks."""
    rng = random.Random(seed)
    keys = sorted(values_by_cluster)
    allv = [v for k in keys for v in values_by_cluster[k]]
    means: list[float] = []
    for _ in range(n):
        drawn = [values_by_cluster[keys[rng.randrange(len(keys))]] for _ in keys]
        flat = [v for c in drawn for v in c]
        means.append(_mean(flat))
    means.sort()
    return _mean(allv), _pct(means, 0.025), _pct(means, 0.975)


def paired_delta(a: Mapping[str, float], b: Mapping[str, float]) -> dict[str, float]:
    """Per-task `a - b` over the tasks present in both."""
    return {t: a[t] - b[t] for t in sorted(set(a) & set(b))}


def by_cluster(deltas: Mapping[str, float], cluster_of: Mapping[str, str]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = defaultdict(list)
    for t, d in deltas.items():
        out[cluster_of[t]].append(d)
    return dict(out)


def median_p90(xs: Sequence[float]) -> tuple[float, float]:
    s = sorted(xs)
    return _pct(s, 0.5), _pct(s, 0.9)
