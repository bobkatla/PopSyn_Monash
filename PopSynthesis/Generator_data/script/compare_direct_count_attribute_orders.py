"""Compare all relaxed direct-count attribute-order experiment outputs."""

from __future__ import annotations

import argparse
import hashlib
from itertools import combinations, permutations
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import yaml


ATTRIBUTES = ["hhsize", "totalvehs", "dwelltype", "owndwell", "hhinc"]
ZONE_FIELD = "zone_id"
COUNT_FIELD = "count"
CANONICAL_COLUMNS = [ZONE_FIELD, *ATTRIBUTES, COUNT_FIELD]
SUMMARY_METRICS = [
    "census_rmse",
    "census_mae",
    "census_l1_error",
    "census_max_absolute_error",
    "total_stage1_slack",
    "full_joint_total_variation",
    "mean_pairwise_total_variation",
    "max_pairwise_total_variation",
    "synthetic_mass_outside_seed_support",
    "seed_mass_absent_from_synthetic",
    "synthetic_joint_types",
    "compressed_zone_type_rows",
    "runtime_seconds",
]


def _resolve_config_path(raw_path: str, runs_yml: Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else runs_yml.parent / path


def load_runs(runs_yml: Path) -> list[dict[str, object]]:
    runs = yaml.safe_load(runs_yml.read_text(encoding="utf-8"))
    if not isinstance(runs, list) or not runs:
        raise ValueError(f"Expected a non-empty run list in {runs_yml}")

    policies = {run.get("zero_cell_policy") for run in runs if isinstance(run, dict)}
    if policies == {"relax_empty_rows_columns"}:
        expected_orders = list(permutations(ATTRIBUTES))
        name_prefix = "relaxed"
    elif policies == {"unrestricted"}:
        expected_orders = list(permutations(ATTRIBUTES))
        name_prefix = "unrestricted"
    elif policies == {"seed_only"}:
        expected_orders = [
            order for order in permutations(ATTRIBUTES) if order[0] != "hhinc"
        ]
        name_prefix = "seed_only"
    else:
        raise ValueError(
            "Runs must use one supported policy: relax_empty_rows_columns, "
            "unrestricted, or seed_only"
        )
    if len(runs) != len(expected_orders):
        raise ValueError(
            f"Expected exactly {len(expected_orders)} {name_prefix} runs in {runs_yml}"
        )

    actual_orders: list[tuple[str, ...]] = []
    output_names: list[str] = []
    for index, run in enumerate(runs, start=1):
        if not isinstance(run, dict):
            raise ValueError(f"Run {index} is not a mapping")
        order = tuple(run.get("ordered_to_adjust_atts", []))
        actual_orders.append(order)
        output_name = str(run.get("output_name", ""))
        expected_name = f"{name_prefix}_order_{index:03d}_{'_'.join(order)}"
        if output_name != expected_name:
            raise ValueError(
                f"Run {index} has output_name {output_name!r}; expected {expected_name!r}"
            )
        output_names.append(output_name)
        for key in ("considered_atts", "last_adjustment_order"):
            if tuple(run.get(key, [])) != order:
                raise ValueError(f"Run {index} has inconsistent {key}")
        if run.get("reruns") != 1:
            raise ValueError(f"Run {index} must have reruns: 1")

    if actual_orders != expected_orders:
        raise ValueError("Runs are not the complete deterministic permutation sequence")
    if len(set(output_names)) != len(expected_orders):
        raise ValueError("Output names are not unique")
    return runs


def load_census(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    census = pd.read_csv(path, header=[0, 1])
    zone_columns = census.columns[census.columns.get_level_values(0) == ZONE_FIELD]
    if len(zone_columns) != 1:
        raise ValueError(f"Expected exactly one {ZONE_FIELD!r} column in {path}")
    census = census.set_index(zone_columns[0])
    census.index = census.index.astype(str)
    census.index.name = ZONE_FIELD
    if census.index.duplicated().any():
        raise ValueError(f"Census zones are not unique in {path}")
    if len(census) != 691:
        raise ValueError(f"Expected 691 Census zones, found {len(census)}")

    records: list[pd.DataFrame] = []
    for attribute in ATTRIBUTES:
        if attribute not in census.columns.get_level_values(0):
            raise ValueError(f"Census is missing attribute {attribute!r}")
        for state in census[attribute].columns:
            values = pd.to_numeric(census[(attribute, state)], errors="raise")
            records.append(
                pd.DataFrame(
                    {
                        ZONE_FIELD: census.index,
                        "attribute": attribute,
                        "state": str(state),
                        "census_count": values.to_numpy(dtype=float),
                    }
                )
            )
    return census, pd.concat(records, ignore_index=True)


def load_seed(path: Path, weight_column: str) -> pd.DataFrame:
    seed = pd.read_csv(
        path,
        usecols=[*ATTRIBUTES, weight_column],
        dtype={attribute: str for attribute in ATTRIBUTES},
    )
    if seed[ATTRIBUTES].isna().any().any():
        raise ValueError(f"Seed contains missing controlled attributes in {path}")
    weights = pd.to_numeric(seed[weight_column], errors="raise")
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"Seed weights must be finite and positive in {path}")
    seed[weight_column] = weights.astype(float)
    return seed


def load_canonical_output(path: Path, expected_zones: set[str]) -> pd.DataFrame:
    population = pd.read_csv(
        path,
        dtype={ZONE_FIELD: str, **{attribute: str for attribute in ATTRIBUTES}},
    )
    if set(population.columns) != set(CANONICAL_COLUMNS):
        raise ValueError(
            f"Unexpected output columns in {path}: {population.columns.tolist()}"
        )
    population = population[CANONICAL_COLUMNS]
    if population[[ZONE_FIELD, *ATTRIBUTES]].isna().any().any():
        raise ValueError(f"Output contains missing keys in {path}")
    actual_zones = set(population[ZONE_FIELD].unique())
    if actual_zones != expected_zones:
        missing = sorted(expected_zones - actual_zones)
        extra = sorted(actual_zones - expected_zones)
        raise ValueError(
            f"Output zone set differs from Census in {path}; "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    counts = pd.to_numeric(population[COUNT_FIELD], errors="raise")
    if (
        not np.isfinite(counts).all()
        or (counts <= 0).any()
        or not np.equal(counts, np.floor(counts)).all()
    ):
        raise ValueError(f"Output counts must be positive integers in {path}")
    population[COUNT_FIELD] = counts.astype("int64")
    key_columns = [ZONE_FIELD, *ATTRIBUTES]
    if population.duplicated(key_columns).any():
        raise ValueError(f"Output contains duplicate compressed keys in {path}")
    return population.sort_values(key_columns, kind="mergesort").reset_index(drop=True)


def census_error_metrics(
    population: pd.DataFrame, census_long: pd.DataFrame
) -> dict[str, float]:
    observed_parts = []
    for attribute in ATTRIBUTES:
        observed = (
            population.groupby([ZONE_FIELD, attribute], observed=True)[COUNT_FIELD]
            .sum()
            .rename("synthetic_count")
            .reset_index()
            .rename(columns={attribute: "state"})
        )
        observed.insert(1, "attribute", attribute)
        observed_parts.append(observed)
    observed_long = pd.concat(observed_parts, ignore_index=True)
    joined = census_long.merge(
        observed_long,
        on=[ZONE_FIELD, "attribute", "state"],
        how="outer",
    ).fillna({"census_count": 0.0, "synthetic_count": 0.0})
    differences = joined["synthetic_count"] - joined["census_count"]
    absolute = differences.abs()
    return {
        "census_rmse": float(np.sqrt(np.mean(np.square(differences)))),
        "census_mae": float(absolute.mean()),
        "census_l1_error": float(absolute.sum()),
        "census_max_absolute_error": float(absolute.max()),
    }


def _weighted_distribution(
    frame: pd.DataFrame, attributes: list[str], value_column: str
) -> pd.Series:
    totals = frame.groupby(attributes, observed=True, dropna=False)[value_column].sum()
    return totals.astype(float) / float(totals.sum())


def distribution_metrics(
    seed: pd.DataFrame, population: pd.DataFrame, weight_column: str
) -> dict[str, float | int]:
    seed_full = _weighted_distribution(seed, ATTRIBUTES, weight_column)
    synthetic_full = _weighted_distribution(population, ATTRIBUTES, COUNT_FIELD)
    full_index = seed_full.index.union(synthetic_full.index)
    seed_full = seed_full.reindex(full_index, fill_value=0.0)
    synthetic_full = synthetic_full.reindex(full_index, fill_value=0.0)
    full_tv = 0.5 * (seed_full - synthetic_full).abs().sum()

    pairwise_tvs = []
    for pair in combinations(ATTRIBUTES, 2):
        pair_attributes = list(pair)
        seed_pair = _weighted_distribution(seed, pair_attributes, weight_column)
        synthetic_pair = _weighted_distribution(
            population, pair_attributes, COUNT_FIELD
        )
        pair_index = seed_pair.index.union(synthetic_pair.index)
        pairwise_tvs.append(
            float(
                0.5
                * (
                    seed_pair.reindex(pair_index, fill_value=0.0)
                    - synthetic_pair.reindex(pair_index, fill_value=0.0)
                )
                .abs()
                .sum()
            )
        )

    return {
        "full_joint_total_variation": float(full_tv),
        "mean_pairwise_total_variation": float(np.mean(pairwise_tvs)),
        "max_pairwise_total_variation": float(np.max(pairwise_tvs)),
        "synthetic_mass_outside_seed_support": float(
            synthetic_full[seed_full == 0].sum()
        ),
        "seed_mass_absent_from_synthetic": float(
            seed_full[synthetic_full == 0].sum()
        ),
        "synthetic_joint_types": int((synthetic_full > 0).sum()),
    }


def total_stage1_slack(path: Path) -> float:
    diagnostics = pd.read_csv(path)
    if "stage1_objective" not in diagnostics.columns:
        raise ValueError(f"Missing stage1_objective in {path}")
    values = pd.to_numeric(diagnostics["stage1_objective"], errors="coerce")
    return float(values.fillna(0.0).sum())


def canonical_digest(population: pd.DataFrame) -> str:
    canonical_bytes = population.to_csv(index=False, lineterminator="\n").encode()
    return hashlib.sha256(canonical_bytes).hexdigest()


def load_runtimes(path: Path, output_names: Iterable[str]) -> dict[str, float]:
    runtimes = pd.read_csv(path)
    required = {"run_name", "rerun", "time_seconds"}
    if not required <= set(runtimes.columns):
        raise ValueError(f"Missing runtime columns in {path}")
    names = list(output_names)
    selected = runtimes[
        runtimes["run_name"].isin(names) & (runtimes["rerun"] == 0)
    ]
    counts = selected["run_name"].value_counts()
    invalid = [name for name in names if counts.get(name, 0) != 1]
    if invalid or len(selected) != len(names):
        raise ValueError(
            "run_times.csv must contain exactly one rerun-0 record for every order; "
            f"invalid={invalid[:5]}"
        )
    seconds = pd.to_numeric(selected["time_seconds"], errors="raise")
    if not np.isfinite(seconds).all() or (seconds < 0).any():
        raise ValueError(f"Invalid runtimes in {path}")
    return dict(zip(selected["run_name"], seconds.astype(float)))


def build_position_effects(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for attribute in ATTRIBUTES:
        position_column = f"{attribute}_position"
        for position in range(1, len(ATTRIBUTES) + 1):
            group = summary[summary[position_column] == position]
            if group.empty:
                continue
            row: dict[str, float | int | str] = {
                "attribute": attribute,
                "position": position,
                "number_of_orders": len(group),
                "duplicate_output_rate": float(
                    group["duplicates_another_order"].astype(bool).mean()
                ),
            }
            for metric in SUMMARY_METRICS:
                values = pd.to_numeric(group[metric], errors="raise")
                row[f"{metric}_mean"] = float(values.mean())
                row[f"{metric}_std"] = float(values.std(ddof=1))
                row[f"{metric}_min"] = float(values.min())
                row[f"{metric}_max"] = float(values.max())
            rows.append(row)
    return pd.DataFrame(rows)


def compare(
    runs_yml: Path, output_root: Path, report_dir: Path | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    runs_yml = runs_yml.resolve()
    runs = load_runs(runs_yml)
    first = runs[0]
    census_path = _resolve_config_path(str(first["hh_marg_file"]), runs_yml)
    seed_path = _resolve_config_path(str(first["hh_seed_data"]), runs_yml)
    weight_column = str(first["pool_count_column"])
    census, census_long = load_census(census_path)
    seed = load_seed(seed_path, weight_column)
    expected_zones = set(census.index)
    output_names = [str(run["output_name"]) for run in runs]
    runtimes = load_runtimes(output_root / "run_times.csv", output_names)

    rows: list[dict[str, object]] = []
    digest_representatives: dict[str, Path] = {}
    digest_members: dict[str, list[int]] = {}
    for permutation_index, run in enumerate(runs, start=1):
        order = list(run["ordered_to_adjust_atts"])
        output_name = str(run["output_name"])
        run_root = output_root / output_name / "reruns_0"
        output_path = run_root / str(run["hh_syn_name"])
        diagnostics_path = run_root / "meta" / "direct_count_adjustments.csv"
        if not output_path.is_file() or not diagnostics_path.is_file():
            raise FileNotFoundError(
                f"Missing output or diagnostics for {output_name}: {run_root}"
            )
        population = load_canonical_output(output_path, expected_zones)
        digest = canonical_digest(population)
        if digest in digest_representatives:
            representative_path = digest_representatives[digest]
            representative = load_canonical_output(
                representative_path, expected_zones
            )
            if not population.equals(representative):
                raise RuntimeError(
                    f"SHA-256 collision between {representative_path} and {output_path}"
                )
        else:
            digest_representatives[digest] = output_path
        digest_members.setdefault(digest, []).append(permutation_index)

        row: dict[str, object] = {
            "permutation_index": permutation_index,
            "output_name": output_name,
            "attribute_order": ">".join(order),
            **{f"attribute_{position}": attribute for position, attribute in enumerate(order, 1)},
            **{f"{attribute}_position": order.index(attribute) + 1 for attribute in ATTRIBUTES},
            **census_error_metrics(population, census_long),
            "total_stage1_slack": total_stage1_slack(diagnostics_path),
            **distribution_metrics(seed, population, weight_column),
            "compressed_zone_type_rows": len(population),
            "runtime_seconds": runtimes[output_name],
            "canonical_sha256": digest,
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    duplicate_groups = [
        members for members in digest_members.values() if len(members) > 1
    ]
    duplicate_groups.sort(key=lambda members: members[0])
    group_by_index: dict[int, str] = {}
    duplicate_rows: list[dict[str, object]] = []
    for group_number, members in enumerate(duplicate_groups, start=1):
        group_id = f"duplicate_group_{group_number:03d}"
        member_rows = summary[summary["permutation_index"].isin(members)]
        digest = str(member_rows.iloc[0]["canonical_sha256"])
        for member in members:
            group_by_index[member] = group_id
        duplicate_rows.append(
            {
                "duplicate_group_id": group_id,
                "canonical_sha256": digest,
                "group_size": len(members),
                "permutation_indices": "|".join(f"{member:03d}" for member in members),
                "output_names": "|".join(member_rows["output_name"]),
                "attribute_orders": "|".join(member_rows["attribute_order"]),
            }
        )
    summary["duplicate_group_id"] = summary["permutation_index"].map(group_by_index)
    summary["duplicates_another_order"] = summary["duplicate_group_id"].notna()
    duplicate_outputs = pd.DataFrame(
        duplicate_rows,
        columns=[
            "duplicate_group_id",
            "canonical_sha256",
            "group_size",
            "permutation_indices",
            "output_names",
            "attribute_orders",
        ],
    )
    position_effects = build_position_effects(summary)

    destination = report_dir or output_root / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    summary.to_csv(destination / "attribute_order_summary.csv", index=False)
    position_effects.to_csv(
        destination / "attribute_position_effects.csv", index=False
    )
    duplicate_outputs.to_csv(destination / "duplicate_outputs.csv", index=False)
    return summary, position_effects, duplicate_outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-yml", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Defaults to <output-root>/comparison.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    summary_frame, effects_frame, duplicate_frame = compare(
        arguments.runs_yml, arguments.output_root, arguments.report_dir
    )
    print(
        f"Compared {len(summary_frame)} orders; wrote {len(effects_frame)} "
        f"position effects and {len(duplicate_frame)} duplicate groups."
    )
