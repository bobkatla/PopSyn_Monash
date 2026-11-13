from __future__ import annotations

import numpy as np
import pandas as pd

from PopSynthesis.analyse.household_experiments.metrics import (
    condense,
    distribution_metrics,
    entropy_metrics,
)
from PopSynthesis.analyse.household_experiments.pipeline import (
    HouseholdExperimentAnalysis,
    _full_index,
    _holm,
)


def test_weighted_condensation_never_expands_records() -> None:
    frame = pd.DataFrame(
        {"a": ["x", "x", "y"], "b": ["m", "m", "n"], "weight": [1.5, 2.5, 10.0]}
    )
    result = condense(frame, ["a", "b"], "weight")
    assert len(result) == 2
    assert result.loc[("x", "m")] == 4.0
    assert result.sum() == 14.0


def test_js_distance_and_divergence_are_labelled_separately() -> None:
    result = distribution_metrics([1, 0], [0, 1])
    assert result["js_distance"] == 1.0
    assert result["js_divergence"] == result["js_distance"] ** 2
    assert result["total_variation"] == 1.0


def test_fixed_cartesian_universe_includes_zero_cells() -> None:
    index = _full_index({"a": ["x", "y"], "b": ["m", "n", "o"]}, ["a", "b"])
    assert len(index) == 6
    sparse = pd.Series([4], index=pd.MultiIndex.from_tuples([("x", "m")], names=["a", "b"]))
    assert sparse.reindex(index, fill_value=0).tolist() == [4, 0, 0, 0, 0, 0]


def test_provenance_tiers_are_mutually_exclusive_and_exhaustive() -> None:
    analyzer = HouseholdExperimentAnalysis.__new__(HouseholdExperimentAnalysis)
    analyzer.seed_joint = pd.Series(
        [1], index=pd.MultiIndex.from_tuples([("seed",)], names=["a"])
    )
    analyzer.bn_joint = pd.Series(
        [1, 1], index=pd.MultiIndex.from_tuples([("seed",), ("bn",)], names=["a"])
    )
    generated = pd.MultiIndex.from_tuples([("seed",), ("bn",), ("new",)], names=["a"])
    assert analyzer._support_tier(generated).tolist() == [
        "seed_supported", "bn_only_supported", "unconfirmed"
    ]


def test_entropy_uses_configured_universe() -> None:
    result = entropy_metrics([5, 5], universe_size=4)
    assert result["occupied_cells"] == 2
    assert result["occupancy_fraction"] == 0.5
    assert np.isclose(result["effective_shannon_count"], 2)


def test_holm_adjustment_is_monotone_in_sorted_pvalues() -> None:
    adjusted = _holm([0.04, 0.001, 0.02])
    ordered = [adjusted[i] for i in np.argsort([0.04, 0.001, 0.02])]
    assert ordered == sorted(ordered)
    assert all(0 <= value <= 1 for value in adjusted)
