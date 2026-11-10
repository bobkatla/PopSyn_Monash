"""Verify direct-count outputs and compare their joints with the weighted seed."""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import counts_to_marginals


ATTRIBUTES = ["hhsize", "totalvehs", "dwelltype", "owndwell", "hhinc"]
EXPECTED_COLUMNS = [zone_field, *ATTRIBUTES, count_field]


def _load_census(path: Path) -> pd.DataFrame:
    census = pd.read_csv(path, header=[0, 1])
    zone_columns = census.columns[
        census.columns.get_level_values(0) == zone_field
    ]
    if len(zone_columns) != 1:
        raise ValueError(f"Expected exactly one {zone_field!r} column in {path}")
    census = census.set_index(zone_columns[0])
    census.index = census.index.astype(str)
    census.index.name = zone_field
    return census


def _validate_output(
    path: Path, expected: pd.DataFrame
) -> tuple[pl.DataFrame, pd.DataFrame]:
    population = pl.read_csv(
        path,
        schema_overrides={
            **{column: pl.String for column in EXPECTED_COLUMNS[:-1]},
            count_field: pl.Int64,
        },
    )
    if population.columns != EXPECTED_COLUMNS:
        raise AssertionError(
            f"Unexpected columns in {path}: {population.columns}"
        )
    if population[zone_field].n_unique() != len(expected):
        raise AssertionError(f"{path} does not contain all {len(expected)} zones")
    if not population[count_field].dtype.is_integer() or (
        population[count_field] <= 0
    ).any():
        raise AssertionError(f"{path} contains invalid population counts")
    if population.select(EXPECTED_COLUMNS[:-1]).is_duplicated().any():
        raise AssertionError(f"{path} contains duplicate zone/type keys")

    actual = counts_to_marginals(population, ATTRIBUTES).reindex(
        index=expected.index, columns=expected.columns, fill_value=0
    )
    pd.testing.assert_frame_equal(
        actual.astype(int), expected.astype(int), check_names=False
    )
    return population.sort(EXPECTED_COLUMNS[:-1]), actual.astype(int)


def _load_seed(path: Path, weight_column: str) -> pd.DataFrame:
    seed = pd.read_csv(
        path,
        usecols=[*ATTRIBUTES, weight_column],
        dtype={attribute: str for attribute in ATTRIBUTES},
    )
    weights = pd.to_numeric(seed[weight_column], errors="coerce")
    if weights.isna().any() or not np.isfinite(weights).all():
        raise ValueError(f"{path} contains invalid {weight_column!r} values")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError(f"{path} must contain nonnegative, nonzero seed weights")
    if seed[ATTRIBUTES].isna().any().any():
        raise ValueError(f"{path} contains missing controlled attribute values")
    seed[weight_column] = weights.astype(float)
    return seed


def _population_frame(population: pl.DataFrame) -> pd.DataFrame:
    frame = population.to_pandas()
    for attribute in ATTRIBUTES:
        frame[attribute] = frame[attribute].astype(str)
    return frame


def _distribution(
    frame: pd.DataFrame, attributes: list[str], value_column: str
) -> pd.Series:
    return frame.groupby(attributes, dropna=False, observed=True)[
        value_column
    ].sum()


def _compare_distribution(
    seed: pd.DataFrame,
    population: pd.DataFrame,
    attributes: list[str],
    weight_column: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    seed_joint = _distribution(seed, attributes, weight_column).astype(float)
    synthetic_joint = _distribution(population, attributes, count_field).astype(float)
    joint_index = seed_joint.index.union(synthetic_joint.index)
    seed_joint = seed_joint.reindex(joint_index, fill_value=0.0)
    synthetic_joint = synthetic_joint.reindex(joint_index, fill_value=0.0)
    seed_probability = seed_joint / seed_joint.sum()
    synthetic_probability = synthetic_joint / synthetic_joint.sum()
    absolute_difference = (synthetic_probability - seed_probability).abs()

    comparison = pd.DataFrame(
        {
            "seed_weight": seed_joint,
            "synthetic_count": synthetic_joint.astype("int64"),
            "seed_probability": seed_probability,
            "synthetic_probability": synthetic_probability,
            "absolute_probability_difference": absolute_difference,
        }
    ).reset_index()
    metrics: dict[str, float | int | str] = {
        "scope": "full" if len(attributes) == len(ATTRIBUTES) else "pairwise",
        "attributes": "+".join(attributes),
        "total_variation_from_seed": float(0.5 * absolute_difference.sum()),
        "synthetic_mass_outside_seed_support": float(
            synthetic_probability[seed_joint == 0].sum()
        ),
        "seed_mass_absent_from_synthetic": float(
            seed_probability[synthetic_joint == 0].sum()
        ),
        "seed_supported_types": int((seed_joint > 0).sum()),
        "synthetic_types": int((synthetic_joint > 0).sum()),
    }
    return comparison, metrics


def _marginal_report(
    expected: pd.DataFrame, actual: pd.DataFrame, rerun: int
) -> pd.DataFrame:
    reports: list[pd.DataFrame] = []
    for attribute, state in expected.columns:
        report = pd.DataFrame(
            {
                "rerun": rerun,
                zone_field: expected.index,
                "attribute": attribute,
                "state": state,
                "census_count": expected[(attribute, state)].to_numpy(),
                "synthetic_count": actual[(attribute, state)].to_numpy(),
            }
        )
        report["difference"] = (
            report["synthetic_count"] - report["census_count"]
        )
        reports.append(report)
    return pd.concat(reports, ignore_index=True)


def verify(
    output_root: Path,
    census_file: Path,
    seed_file: Path,
    weight_column: str,
    report_dir: Path,
) -> None:
    expected = _load_census(census_file)
    validated = [
        _validate_output(
            output_root / f"reruns_{rerun}" / "household_counts.csv",
            expected,
        )
        for rerun in range(3)
    ]
    outputs = [item[0] for item in validated]
    actual_marginals = [item[1] for item in validated]
    if not outputs[0].equals(outputs[1]) or not outputs[0].equals(outputs[2]):
        raise AssertionError("The three direct-count reruns are not identical")

    report_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            _marginal_report(expected, actual, rerun)
            for rerun, actual in enumerate(actual_marginals)
        ],
        ignore_index=True,
    ).to_csv(report_dir / "marginal_comparison.csv", index=False)

    seed = _load_seed(seed_file, weight_column)
    metric_rows: list[dict[str, float | int | str]] = []
    full_comparisons: list[pd.DataFrame] = []
    scopes = [ATTRIBUTES, *[list(pair) for pair in combinations(ATTRIBUTES, 2)]]
    for rerun, output in enumerate(outputs):
        population = _population_frame(output)
        for attributes in scopes:
            comparison, metrics = _compare_distribution(
                seed, population, attributes, weight_column
            )
            metrics["rerun"] = rerun
            metric_rows.append(metrics)
            if len(attributes) == len(ATTRIBUTES):
                comparison.insert(0, "rerun", rerun)
                full_comparisons.append(comparison)

    metrics_frame = pd.DataFrame(metric_rows)
    metrics_frame.to_csv(report_dir / "joint_distribution_metrics.csv", index=False)
    pd.concat(full_comparisons, ignore_index=True).to_csv(
        report_dir / "full_joint_distribution_comparison.csv", index=False
    )
    full_tv = metrics_frame.loc[
        metrics_frame["scope"] == "full", "total_variation_from_seed"
    ].tolist()
    print(
        "Verified 3 identical compressed outputs across "
        f"{len(expected)} zones and all controlled marginals."
    )
    print(
        f"Weighted-seed joint reports written to {report_dir}; "
        f"full-joint total variation: {full_tv}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "IO/output/direct_count_weighted_seed_full/weighted_seed_flow"
        ),
    )
    parser.add_argument(
        "--census-file",
        type=Path,
        default=Path("IO/processed_data/census_POA_hh_saa.csv"),
    )
    parser.add_argument(
        "--seed-file",
        type=Path,
        default=Path("IO/processed_data/ori_sample_hh.csv"),
    )
    parser.add_argument("--weight-column", default="_weight")
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Defaults to <output-root>/verification.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    verify(
        arguments.output_root,
        arguments.census_file,
        arguments.seed_file,
        arguments.weight_column,
        arguments.report_dir or arguments.output_root / "verification",
    )
