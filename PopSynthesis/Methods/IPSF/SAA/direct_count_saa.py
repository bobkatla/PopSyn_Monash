"""One-pass sampling-free progressive-count SAA engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from numbers import Real
from typing import Sequence

import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import population_size
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_ilp import (
    prepare_direct_adjustment,
    solve_direct_adjustment,
)


@dataclass(frozen=True)
class DirectCountPassResult:
    population: pl.DataFrame
    adjustments: list[dict[str, object]]


def _integer_count(value: object, label: str) -> int:
    if pd.isna(value) or isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{label} must be a nonnegative integer")
    converted = int(value)
    if converted < 0 or float(value) != converted:
        raise ValueError(f"{label} must be a nonnegative integer")
    return converted


def _targets_by_attribute(
    residual_marginals: pd.DataFrame, ordered_attributes: Sequence[str]
) -> tuple[str, dict[str, dict[str, int]]]:
    if not isinstance(residual_marginals.columns, pd.MultiIndex):
        raise ValueError("Marginal columns must be a two-level MultiIndex")
    if len(residual_marginals) != 1 or not residual_marginals.index.is_unique:
        raise ValueError("count_native_ilp requires exactly one Census zone")
    if residual_marginals.index.name != zone_field:
        raise ValueError(f"Marginal index must be named {zone_field!r}")
    if not ordered_attributes or len(set(ordered_attributes)) != len(
        ordered_attributes
    ):
        raise ValueError("The adjustment order must contain unique attributes")

    zone = str(residual_marginals.index[0])
    targets: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for attribute in ordered_attributes:
        columns = [
            column
            for column in residual_marginals.columns
            if column[0] == attribute
        ]
        if not columns:
            raise ValueError(f"Marginals do not contain attribute {attribute!r}")
        states: dict[str, int] = {}
        for column in columns:
            state = str(column[1])
            if state in states:
                raise ValueError(
                    f"Marginals contain duplicate state {state!r} for {attribute!r}"
                )
            states[state] = _integer_count(
                residual_marginals.iloc[0][column], f"{attribute}.{state}"
            )
        targets[attribute] = states
        totals[attribute] = sum(states.values())
    if len(set(totals.values())) != 1:
        raise ValueError(f"Marginal population totals are inconsistent: {totals}")
    return zone, targets


def _validate_pool(count_pool: pl.DataFrame, attributes: Sequence[str]) -> None:
    missing = (set(attributes) | {count_field}) - set(count_pool.columns)
    if missing:
        raise ValueError(f"Pool is missing columns: {sorted(missing)}")
    if count_pool.is_empty():
        raise ValueError("Pool cannot be empty")
    counts = count_pool[count_field]
    if not counts.dtype.is_integer() or (counts <= 0).any():
        raise ValueError("Pool counts must be positive integers")
    if count_pool.select(
        pl.any_horizontal([pl.col(attribute).is_null() for attribute in attributes]).any()
    ).item():
        raise ValueError("Pool attributes cannot contain nulls")


def run_direct_count_pass(
    residual_marginals: pd.DataFrame,
    count_pool: pl.DataFrame,
    ordered_attributes: Sequence[str],
    *,
    pass_index: int,
) -> DirectCountPassResult:
    """Synthesize one complete compressed population without sampling."""

    order = list(ordered_attributes)
    zone, targets = _targets_by_attribute(residual_marginals, order)
    _validate_pool(count_pool, order)
    first_attribute = order[0]
    first_rows = [
        {
            zone_field: zone,
            first_attribute: state,
            count_field: target,
        }
        for state, target in targets[first_attribute].items()
        if target > 0
    ]
    if not first_rows:
        raise ValueError("A direct-count pass requires a positive target population")
    current = pl.DataFrame(first_rows).with_columns(pl.col(count_field).cast(pl.Int64))
    target_population = population_size(current)
    adjustments: list[dict[str, object]] = [
        {
            "mode": "count_native_ilp",
            "zone": zone,
            "pass": pass_index,
            "adjustment_index": 0,
            "active_attribute": first_attribute,
            "pass_target_population": target_population,
            "prior_row_count": 0,
            "ilp_cell_count": 0,
            "stage1_status": "NotRun",
            "stage1_objective": None,
            "stage2_status": "NotRun",
            "stage2_objective": None,
            "stage2_objective_scale": None,
            "target_counts_json": json.dumps(
                targets[first_attribute], sort_keys=True
            ),
            "achieved_counts_json": json.dumps(
                targets[first_attribute], sort_keys=True
            ),
            "marginal_l1_error": 0,
            "row_totals_preserved": True,
            "unsupported_row_count": 0,
            "unsupported_column_count": sum(
                1
                for state, target in targets[first_attribute].items()
                if target > 0
                and count_pool.filter(pl.col(first_attribute) == state).is_empty()
            ),
            "ordinary_zero_cells_made_positive": 0,
            "solver_runtime_seconds": 0.0,
            "pulp_version": None,
            "solver_version": None,
            "solver_options_json": None,
        }
    ]

    for adjustment_index, attribute in enumerate(order[1:], start=1):
        prepared = prepare_direct_adjustment(
            current, count_pool, attribute, targets[attribute]
        )
        solved = solve_direct_adjustment(prepared)
        current = solved.counts
        diagnostic = dict(solved.diagnostic)
        diagnostic.update(
            {
                "pass": pass_index,
                "adjustment_index": adjustment_index,
                "pass_target_population": target_population,
            }
        )
        adjustments.append(diagnostic)
        if population_size(current) != target_population:
            raise AssertionError("Direct propagation changed the pass population")

    expected_columns = [zone_field, *order, count_field]
    current = current.select(expected_columns)
    if current.select(expected_columns[:-1]).is_duplicated().any():
        raise AssertionError("Direct-count pass returned duplicate record keys")
    return DirectCountPassResult(current, adjustments)
