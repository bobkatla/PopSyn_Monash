from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pgmpy.factors.discrete import TabularCPD

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetworkModel
except ImportError:  # pragma: no cover
    from pgmpy.models import BayesianNetwork as BayesianNetworkModel

from PopSynthesis.Methods.BN.utils.bn_pool import (
    learn_bn,
    learn_census_prior_bn,
    learn_seed_only_bn,
    load_bn_artifact,
    sample_condensed_pool,
    sample_exact_marginal,
    sample_zonal_exact_marginal,
    save_bn_artifact,
)


ATTRIBUTES = ["hhsize", "hhinc"]


def _seed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hhsize": ["1", "1", "8+", "8+", "8+", "1"],
            "hhinc": ["Low", "High", "Low", "High", "High", "Low"],
            "_weight": [10.0, 8.0, 12.0, 9.0, 11.0, 10.0],
        }
    )


def _census(include_other: bool = False) -> pd.DataFrame:
    values = {
        ("zone_id", "Unnamed: 0_level_1"): ["z1", "z2"],
        ("hhsize", "1"): [30, 20],
        ("hhsize", "8+"): [20, 30],
        ("hhinc", "Low"): [20, 20],
        ("hhinc", "High"): [25, 25],
        ("hhinc", "Negative income"): [5, 5],
    }
    if include_other:
        values[("hhinc", "Other unseen")] = [1, 1]
        values[("hhinc", "Low")] = [19, 19]
    frame = pd.DataFrame(values)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def _learn(include_negative: bool = True):
    return learn_census_prior_bn(
        seed=_seed(),
        census=_census(),
        attributes=ATTRIBUTES,
        extra_states={"hhinc": ["Negative income"]} if include_negative else None,
        ignored_census_states=(
            None if include_negative else {"hhinc": ["Negative income"]}
        ),
    )


def test_unseen_state_and_census_prior_shapes() -> None:
    result = _learn()

    assert "Negative income" in result.state_names["hhinc"]
    assert result.prior_strength == 100
    for node in result.model.nodes():
        cpd = result.model.get_cpds(node)
        assert result.pseudo_counts[node].shape == cpd.get_values().shape
        assert result.pseudo_counts[node].sum() == pytest.approx(100)

        parents = sorted(result.model.get_parents(node))
        if not parents:
            expected = np.array(
                list(result.census_probabilities[node].values())
            ) * 100
            np.testing.assert_allclose(result.pseudo_counts[node][:, 0], expected)

    negative_index = result.state_names["hhinc"].index("Negative income")
    assert (result.model.get_cpds("hhinc").get_values()[negative_index] > 0).all()


def test_optional_state_does_not_change_structure() -> None:
    included = _learn(include_negative=True)
    excluded = _learn(include_negative=False)

    assert set(included.model.edges()) == set(excluded.model.edges())
    assert "Negative income" not in excluded.state_names["hhinc"]


def test_seed_mode_uses_explicit_unseen_probability_without_census() -> None:
    result = learn_seed_only_bn(
        seed=_seed(),
        attributes=ATTRIBUTES,
        extra_states={"hhinc": ["Negative income"]},
        unseen_state_probabilities={"hhinc": {"Negative income": 0.025}},
    )

    assert result.mode == "seed"
    assert result.census_probabilities == {}
    assert result.pseudo_counts == {}
    assert result.prior_strength == 0
    assert result.diagnostics["fitted_marginals"]["hhinc"][
        "Negative income"
    ] == pytest.approx(0.025)


def test_learning_mode_dispatch_and_unseen_probability_validation() -> None:
    seed_result = learn_bn(
        seed=_seed(),
        attributes=ATTRIBUTES,
        mode="seed",
    )
    assert seed_result.mode == "seed"

    with pytest.raises(ValueError, match="must exactly match"):
        learn_seed_only_bn(
            seed=_seed(),
            attributes=ATTRIBUTES,
            extra_states={"hhinc": ["Negative income"]},
        )
    with pytest.raises(ValueError, match="mode must be"):
        learn_bn(seed=_seed(), attributes=ATTRIBUTES, mode="unknown")


def test_weighted_seed_and_soft_prior_posterior() -> None:
    seed = pd.DataFrame({"income": ["Known", "Known"], "_weight": [3.0, 1.0]})
    census = pd.DataFrame(
        {
            ("income", "Known"): [90],
            ("income", "Negative income"): [10],
        }
    )
    census.columns = pd.MultiIndex.from_tuples(census.columns)

    result = learn_census_prior_bn(
        seed,
        census,
        attributes=["income"],
        extra_states={"income": ["Negative income"]},
    )
    cpd = result.model.get_cpds("income")
    negative_index = cpd.state_names["income"].index("Negative income")
    assert cpd.get_values()[negative_index, 0] == pytest.approx(10 / 104)


def test_chunked_pool_is_reproducible_and_condensed() -> None:
    model = _learn().model
    first = sample_condensed_pool(
        model,
        size=2_000,
        chunk_size=250,
        seed=7,
        attributes=ATTRIBUTES,
        required_states={"hhinc": ["Negative income"]},
    )
    second = sample_condensed_pool(
        model,
        size=2_000,
        chunk_size=250,
        seed=7,
        attributes=ATTRIBUTES,
        required_states={"hhinc": ["Negative income"]},
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["count"].sum() == 2_000
    assert not first.duplicated(ATTRIBUTES).any()


def test_xmlbif_round_trip(tmp_path) -> None:
    result = _learn()
    model_path = tmp_path / "household_bn.xml"
    _, metadata_path = save_bn_artifact(
        result.model,
        {"prior_strength": result.prior_strength},
        model_path,
    )
    loaded = load_bn_artifact(model_path)

    assert metadata_path.exists()
    assert set(loaded.edges()) == set(result.model.edges())
    for node in result.model.nodes():
        original = result.model.get_cpds(node)
        restored = loaded.get_cpds(node)
        assert original.state_names == restored.state_names
        np.testing.assert_allclose(original.get_values(), restored.get_values())


def _sampling_model():
    model = BayesianNetworkModel([("A", "B")])
    model.add_cpds(
        TabularCPD("A", 2, [[0.6], [0.4]], state_names={"A": ["a0", "a1"]}),
        TabularCPD(
            "B",
            2,
            [[0.8, 0.2], [0.2, 0.8]],
            evidence=["A"],
            evidence_card=[2],
            state_names={"A": ["a0", "a1"], "B": ["b0", "b1"]},
        ),
    )
    return model


def test_exact_marginal_root_and_rejection_sampling() -> None:
    model = _sampling_model()
    root_sample = sample_exact_marginal(
        model, "A", {"a0": 40, "a1": 10}, method="conditional", seed=3
    )
    rejection_sample = sample_exact_marginal(
        model, "B", {"b0": 25, "b1": 25}, method="rejection", seed=3
    )

    assert root_sample["A"].value_counts().to_dict() == {"a0": 40, "a1": 10}
    assert rejection_sample["B"].value_counts().to_dict() == {"b0": 25, "b1": 25}
    with pytest.raises(ValueError, match="has parents"):
        sample_exact_marginal(model, "B", {"b0": 1}, method="root")


def test_zonal_exact_marginal_matches_every_zone() -> None:
    targets = pd.DataFrame(
        {"b0": [3, 1], "b1": [2, 4]},
        index=pd.Index(["z1", "z2"], name="zone_id"),
    )
    population = sample_zonal_exact_marginal(
        _sampling_model(),
        attribute="B",
        zone_targets=targets,
        method="rejection",
        seed=4,
    )
    actual = (
        population.groupby(["zone_id", "B"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=targets.index, columns=targets.columns, fill_value=0)
    )
    pd.testing.assert_frame_equal(actual, targets)


def test_unsupported_positive_census_state_fails() -> None:
    with pytest.raises(ValueError, match="Other unseen"):
        learn_census_prior_bn(
            _seed(),
            _census(include_other=True),
            attributes=ATTRIBUTES,
            extra_states={"hhinc": ["Negative income"]},
        )


def test_required_pool_state_and_invalid_target_fail() -> None:
    model = _learn(include_negative=False).model
    with pytest.raises(RuntimeError, match="missing required"):
        sample_condensed_pool(
            model,
            size=100,
            required_states={"hhinc": ["Negative income"]},
            attributes=ATTRIBUTES,
        )
    with pytest.raises(ValueError, match="does not support"):
        sample_exact_marginal(model, "hhsize", {"9": 1})
