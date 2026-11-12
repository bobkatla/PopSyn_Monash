from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from PopSynthesis.Methods.GAN.wgan import (
    HOUSEHOLD_ATTRIBUTES,
    WGANConfig,
    load_wgan_artifact,
    prepare_household_training_data,
    sample_exact_marginal,
    sample_zonal_exact_marginal,
    save_wgan_artifact,
    train_household_wgan,
)


def _seed() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "dwelltype": ["House", "Flat", "House", "Flat", "House", "Flat"],
            "owndwell": ["Own", "Rent", "Rent", "Own", "Own", "Rent"],
            "hhinc": ["Nil income", "Low", "Nil income", "High", "Low", "Nil income"],
            "totalvehs": ["1", "0", "2", "1", "2", "0"],
            "hhsize": ["1", "1", "2", "2", "2", "1"],
            "_weight": [20.0, 30.0, 25.0, 10.0, 15.0, 20.0],
        }
    )


def _census() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            ("zone_id", "id"): ["z1", "z2"],
            ("hhsize", "1"): [3, 2],
            ("hhsize", "2"): [2, 3],
            ("hhinc", "Low"): [40, 40],
            ("hhinc", "High"): [5, 5],
            ("hhinc", "Nil income"): [4, 4],
            ("hhinc", "Negative income"): [1, 1],
        }
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def _trained():
    prepared = prepare_household_training_data(_seed(), _census())
    config = WGANConfig(
        latent_dim=8,
        hidden_dim=16,
        batch_size=8,
        epochs=1,
        n_critic=1,
        temperature_start=0.8,
        temperature_end=0.8,
        seed=7,
    )
    return prepared, train_household_wgan(prepared, config=config, device="cpu")


def test_donor_augmentation_preserves_strata_and_probability() -> None:
    seed = _seed()
    prepared = prepare_household_training_data(seed, _census())
    expected_probability = 2 / 100

    assert prepared.negative_income_probability == pytest.approx(expected_probability)
    assert "Negative income" in prepared.state_names["hhinc"]
    for hhsize, original_group in seed.groupby("hhsize"):
        augmented_group = prepared.frame[prepared.frame["hhsize"].eq(hhsize)]
        assert augmented_group["_weight"].sum() == pytest.approx(original_group["_weight"].sum())
        negative = augmented_group.loc[augmented_group["hhinc"].eq("Negative income"), "_weight"].sum()
        assert negative / augmented_group["_weight"].sum() == pytest.approx(expected_probability)


def test_donor_augmentation_requires_nil_income_in_every_stratum() -> None:
    seed = _seed()
    seed.loc[seed["hhsize"].eq("2") & seed["hhinc"].eq("Nil income"), "hhinc"] = "Low"
    with pytest.raises(ValueError, match="hhsize='2'"):
        prepare_household_training_data(seed, _census())


def test_training_and_exact_sampling_are_valid_and_reproducible() -> None:
    prepared, result = _trained()
    first = sample_exact_marginal(
        result.model, "hhsize", {"1": 12, "2": 9}, seed=11, chunk_size=5
    )
    second = sample_exact_marginal(
        result.model, "hhsize", {"1": 12, "2": 9}, seed=11, chunk_size=5
    )

    pd.testing.assert_frame_equal(first, second)
    assert first["hhsize"].value_counts().to_dict() == {"1": 12, "2": 9}
    assert list(first.columns) == list(HOUSEHOLD_ATTRIBUTES)
    for attribute in HOUSEHOLD_ATTRIBUTES:
        assert set(first[attribute]) <= set(prepared.state_names[attribute])


def test_zonal_sampling_matches_every_household_size_count() -> None:
    _, result = _trained()
    targets = pd.DataFrame(
        {"1": [3, 0], "2": [2, 4]},
        index=pd.Index(["z1", "z2"], name="zone_id"),
    )
    population = sample_zonal_exact_marginal(
        result.model, "hhsize", targets, seed=13, chunk_size=3
    )
    actual = (
        population.groupby(["zone_id", "hhsize"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=targets.index, columns=targets.columns, fill_value=0)
    )
    pd.testing.assert_frame_equal(actual, targets)


def test_artifact_round_trip_preserves_deterministic_sampling(tmp_path: Path) -> None:
    _, result = _trained()
    model_path = tmp_path / "household_wgan.pt"
    save_wgan_artifact(result.model, {"diagnostics": result.diagnostics}, model_path)
    loaded, metadata = load_wgan_artifact(model_path, device="cpu")
    conditions = pd.DataFrame(
        {"hhsize": ["1", "2", "1"], "hhinc": ["Low", "Negative income", "Nil income"]}
    )

    expected = result.model.sample_conditions(conditions, seed=19, chunk_size=2)
    actual = loaded.sample_conditions(conditions, seed=19, chunk_size=2)
    pd.testing.assert_frame_equal(actual, expected)
    assert actual.loc[1, "hhinc"] == "Negative income"
    assert metadata["format_version"] == 1


def test_corrupted_model_is_not_saved(tmp_path: Path) -> None:
    _, result = _trained()
    with torch.no_grad():
        next(result.model.generator.parameters()).fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="contains NaN"):
        save_wgan_artifact(result.model, {}, tmp_path / "corrupt.pt")
    assert not (tmp_path / "corrupt.pt").exists()


def test_notebook_has_expected_workflow() -> None:
    notebook_path = (
        Path(__file__).parents[2] / "Methods/GAN/00_create_wgan_population.ipynb"
    )
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "prepare_household_training_data" in source
    assert "sample_zonal_exact_marginal" in source
    assert "hh_wgan_exact_hhsize.csv" in source
    assert "HH_pool" not in source
