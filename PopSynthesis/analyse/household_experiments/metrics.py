"""Small, independently testable metrics used by the household analysis."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


def probabilities(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if np.any(array < 0) or not np.all(np.isfinite(array)):
        raise ValueError("Distribution weights must be finite and nonnegative")
    total = float(array.sum())
    return array / total if total else np.zeros_like(array)


def gini(values: Iterable[float]) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array) & (array >= 0)]
    if not len(array) or array.sum() == 0:
        return 0.0
    ordered = np.sort(array)
    n = len(ordered)
    return float((2 * np.dot(np.arange(1, n + 1), ordered) / ordered.sum() - n - 1) / n)


def distribution_metrics(
    reference: Iterable[float],
    synthetic: Iterable[float],
    *,
    smoothing: float = 1e-12,
) -> dict[str, float]:
    """Compare aligned nonnegative vectors.

    ``js_distance`` is SciPy's metric and ``js_divergence`` is its square.
    SRMSE follows the repository convention: RMSE divided by the mean
    reference probability over the complete supplied universe.
    """

    r = probabilities(reference)
    s = probabilities(synthetic)
    if r.shape != s.shape:
        raise ValueError("Distributions must have the same shape")
    if not len(r):
        raise ValueError("Distributions must not be empty")
    js_distance = float(jensenshannon(r, s, base=2)) if (r.sum() and s.sum()) else math.nan
    delta = r - s
    support_r, support_s = r > 0, s > 0
    union = support_r | support_s
    intersection = support_r & support_s
    support_jaccard = float(intersection.sum() / union.sum()) if union.any() else 1.0
    weighted_union = np.maximum(r, s).sum()
    weighted_jaccard = float(np.minimum(r, s).sum() / weighted_union) if weighted_union else 1.0
    nonzero = union
    pearson = math.nan
    if nonzero.sum() >= 2 and np.std(r[nonzero]) > 0 and np.std(s[nonzero]) > 0:
        pearson = float(np.corrcoef(r[nonzero], s[nonzero])[0, 1])
    rp = (r + smoothing) / (r + smoothing).sum()
    sp = (s + smoothing) / (s + smoothing).sum()
    symmetric_kl = float(
        0.5 * (np.sum(rp * np.log(rp / sp)) + np.sum(sp * np.log(sp / rp)))
    )
    rmse = float(np.sqrt(np.mean(delta**2)))
    return {
        "js_distance": js_distance,
        "js_divergence": js_distance**2,
        "total_variation": float(0.5 * np.abs(delta).sum()),
        "hellinger": float(np.sqrt(0.5 * np.square(np.sqrt(r) - np.sqrt(s)).sum())),
        "srmse": float(rmse / r.mean()) if r.mean() else math.nan,
        "l1_probability_error": float(np.abs(delta).sum()),
        "l2_probability_error": float(np.sqrt(np.square(delta).sum())),
        "shared_probability_mass": float(np.minimum(r, s).sum()),
        "support_jaccard": support_jaccard,
        "weighted_jaccard": weighted_jaccard,
        "nonzero_probability_pearson": pearson,
        "symmetric_kl": symmetric_kl,
        "smoothing_constant": smoothing,
    }


def entropy_metrics(counts: Iterable[float], universe_size: int) -> dict[str, float]:
    values = np.asarray(counts, dtype=float)
    positive = values[values > 0]
    p = probabilities(positive)
    entropy = float(-(p * np.log(p)).sum()) if len(p) else 0.0
    simpson_concentration = float(np.square(p).sum()) if len(p) else 0.0
    sorted_desc = np.sort(positive)[::-1]
    total = positive.sum()
    top_one_n = max(1, math.ceil(0.01 * len(positive))) if len(positive) else 0
    return {
        "occupied_cells": int(len(positive)),
        "universe_size": int(universe_size),
        "occupancy_fraction": float(len(positive) / universe_size) if universe_size else math.nan,
        "shannon_entropy": entropy,
        "normalized_entropy": float(entropy / math.log(universe_size)) if universe_size > 1 else 0.0,
        "effective_shannon_count": float(math.exp(entropy)),
        "simpson_diversity": float(1 - simpson_concentration),
        "effective_simpson_count": float(1 / simpson_concentration) if simpson_concentration else 0.0,
        "top_1_percent_mass": float(sorted_desc[:top_one_n].sum() / total) if total else 0.0,
        "top_10_mass": float(sorted_desc[:10].sum() / total) if total else 0.0,
        "top_100_mass": float(sorted_desc[:100].sum() / total) if total else 0.0,
        "count_gini": gini(positive),
        "count_q25": float(np.quantile(positive, 0.25)) if len(positive) else 0.0,
        "count_median": float(np.quantile(positive, 0.5)) if len(positive) else 0.0,
        "count_q75": float(np.quantile(positive, 0.75)) if len(positive) else 0.0,
    }


def cramers_v(table: np.ndarray) -> float:
    table = np.asarray(table, dtype=float)
    n = table.sum()
    if n <= 0 or min(table.shape) <= 1:
        return 0.0
    expected = np.outer(table.sum(axis=1), table.sum(axis=0)) / n
    valid = expected > 0
    chi2 = np.square(table - expected)[valid].dot(1 / expected[valid])
    return float(np.sqrt((chi2 / n) / min(table.shape[0] - 1, table.shape[1] - 1)))


def mutual_information(table: np.ndarray) -> tuple[float, float]:
    values = np.asarray(table, dtype=float)
    total = values.sum()
    if total <= 0:
        return 0.0, 0.0
    joint = values / total
    px, py = joint.sum(axis=1), joint.sum(axis=0)
    expected = np.outer(px, py)
    valid = joint > 0
    mi = float(np.sum(joint[valid] * np.log(joint[valid] / expected[valid])))
    hx = float(-np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = float(-np.sum(py[py > 0] * np.log(py[py > 0])))
    nmi = float(mi / math.sqrt(hx * hy)) if hx and hy else 0.0
    return mi, nmi


def summary_statistics(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: math.nan for key in ("mean", "std", "cv", "median", "iqr", "min", "max", "ci95_low", "ci95_high")}
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else 0.0
    se = std / math.sqrt(len(array))
    return {
        "mean": mean,
        "std": std,
        "cv": float(std / abs(mean)) if mean else math.nan,
        "median": float(np.median(array)),
        "iqr": float(np.quantile(array, 0.75) - np.quantile(array, 0.25)),
        "min": float(array.min()),
        "max": float(array.max()),
        "ci95_low": mean - 1.96 * se,
        "ci95_high": mean + 1.96 * se,
    }


def condense(frame: pd.DataFrame, keys: list[str], weight: str) -> pd.Series:
    """Return a weighted joint Series without expanding weighted records."""

    return frame.groupby(keys, observed=True, dropna=False)[weight].sum().astype(float)
