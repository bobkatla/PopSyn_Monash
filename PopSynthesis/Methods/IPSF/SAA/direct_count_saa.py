"""One-pass sampling-free progressive-count SAA engine."""

from __future__ import annotations

from dataclasses import dataclass
import json
from math import isfinite
from numbers import Real
from typing import Literal, Sequence

import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import population_size
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_ilp import (
    DIRECT_REFERENCE_MODES,
    PoolLookup,
    ZERO_CELL_COLUMN_SCOPES,
    ZERO_CELL_POLICIES,
    build_pool_lookup,
    prepare_direct_adjustment,
    solve_direct_adjustment,
)
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_flow import (
    solve_direct_adjustment_flow,
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
    residual_marginals: pd.DataFrame,
    ordered_attributes: Sequence[str],
    *,
    enforce_consistent_totals: bool = True,
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
    if enforce_consistent_totals and len(set(totals.values())) != 1:
        raise ValueError(f"Marginal population totals are inconsistent: {totals}")
    return zone, targets


def _validate_pool(count_pool: pl.DataFrame, attributes: Sequence[str]) -> None:
    missing = (set(attributes) | {count_field}) - set(count_pool.columns)
    if missing:
        raise ValueError(f"Pool is missing columns: {sorted(missing)}")
    if count_pool.is_empty():
        raise ValueError("Pool cannot be empty")
    counts = count_pool[count_field]
    if not counts.dtype.is_numeric():
        raise ValueError("Pool counts must be numeric")
    invalid_counts = any(
        value is None or not isfinite(float(value)) or value <= 0
        for value in counts
    )
    if invalid_counts:
        raise ValueError("Pool counts must be finite and positive")
    if count_pool.select(
        pl.any_horizontal([pl.col(attribute).is_null() for attribute in attributes]).any()
    ).item():
        raise ValueError("Pool attributes cannot contain nulls")


def _validate_first_attribute_support(
    zone: str,
    attribute: str,
    targets: dict[str, int],
    count_pool: pl.DataFrame,
) -> None:
    support = {
        str(row[attribute]): row[count_field]
        for row in count_pool.group_by(attribute).agg(
            pl.col(count_field).sum()
        ).iter_rows(named=True)
    }
    for state, target in targets.items():
        pool_support = support.get(state, 0)
        if target > 0 and pool_support <= 0:
            raise ValueError(
                "Positive first-attribute Census target has no pool support: "
                f"zone={zone}, attribute={attribute!r}, category={state!r}, "
                f"target={target}, pool_support={pool_support}"
            )


def run_direct_count_pass(
    residual_marginals: pd.DataFrame,
    count_pool: pl.DataFrame | None,
    ordered_attributes: Sequence[str],
    *,
    pass_index: int,
    solver_mode: Literal["single_stage", "two_stage"] = "single_stage",
    adjustment_backend: Literal["cbc", "min_cost_flow"] = "cbc",
    zero_cell_policy: Literal[
        "unrestricted",
        "seed_only",
        "relax_empty_rows_columns",
        "relax_empty_rows_global_columns",
    ] = "unrestricted",
    direct_reference_mode: Literal["pool", "all_zero"] = "pool",
    conditional_cache: dict[
        tuple[tuple[str, ...], str], PoolLookup
    ] | None = None,
) -> DirectCountPassResult:
    """Synthesize one complete compressed population without sampling."""

    order = list(ordered_attributes)
    if adjustment_backend not in {"cbc", "min_cost_flow"}:
        raise ValueError(
            "adjustment_backend must be 'cbc' or 'min_cost_flow'"
        )
    if zero_cell_policy not in ZERO_CELL_POLICIES:
        raise ValueError(
            "zero_cell_policy must be 'unrestricted', 'seed_only', "
            "'relax_empty_rows_columns', or "
            "'relax_empty_rows_global_columns'"
        )
    if direct_reference_mode not in DIRECT_REFERENCE_MODES:
        raise ValueError("direct_reference_mode must be 'pool' or 'all_zero'")
    if direct_reference_mode == "all_zero" and zero_cell_policy != "unrestricted":
        raise ValueError(
            "direct_reference_mode='all_zero' requires zero_cell_policy='unrestricted'"
        )
    if solver_mode == "single_stage" and zero_cell_policy != "unrestricted":
        raise ValueError("Restricted zero-cell policies require two-stage solving")
    if conditional_cache is None:
        conditional_cache = {}
    zone, targets = _targets_by_attribute(
        residual_marginals,
        order,
        enforce_consistent_totals=solver_mode != "two_stage",
    )
    if direct_reference_mode == "pool":
        if count_pool is None:
            raise ValueError("A pool is required when direct_reference_mode='pool'")
        _validate_pool(count_pool, order)
    elif count_pool is not None:
        raise ValueError("A pool must not be supplied when direct_reference_mode='all_zero'")
    first_attribute = order[0]
    if (
        direct_reference_mode == "pool"
        and zero_cell_policy != "relax_empty_rows_columns"
    ):
        assert count_pool is not None
        _validate_first_attribute_support(
            zone, first_attribute, targets[first_attribute], count_pool
        )
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
            "adjustment_backend": adjustment_backend,
            "solver_mode": solver_mode,
            "zone": zone,
            "pass": pass_index,
            "adjustment_index": 0,
            "active_attribute": first_attribute,
            "direct_reference_mode": direct_reference_mode,
            "zero_cell_policy": zero_cell_policy,
            "zero_cell_column_scope": ZERO_CELL_COLUMN_SCOPES[zero_cell_policy],
            "pass_target_population": target_population,
            "prior_row_count": 0,
            "ilp_cell_count": 0,
            "permitted_cell_count": 0,
            "prohibited_cell_count": 0,
            "relaxed_cell_count": 0,
            "stage1_status": "NotRun",
            "stage1_objective": None,
            "stage1_solver_gap": None,
            "stage1_solve_seconds": 0.0,
            "stage1_variable_count": 0,
            "stage1_constraint_count": 0,
            "stage2_status": "NotRun",
            "stage2_objective": None,
            "stage2_objective_scale": None,
            "stage2_solver_gap": None,
            "stage2_slack_objective": None,
            "stage2_solve_seconds": 0.0,
            "stage2_variable_count": 0,
            "stage2_constraint_count": 0,
            "target_counts_json": json.dumps(
                targets[first_attribute], sort_keys=True
            ),
            "achieved_counts_json": json.dumps(
                targets[first_attribute], sort_keys=True
            ),
            "marginal_l1_error": 0,
            "row_totals_preserved": True,
            "unsupported_row_count": 0,
            "unsupported_column_count": (
                sum(target > 0 for target in targets[first_attribute].values())
                if direct_reference_mode == "all_zero"
                else sum(
                    1
                    for state, target in targets[first_attribute].items()
                    if target > 0
                    and count_pool is not None
                    and count_pool.filter(pl.col(first_attribute) == state).is_empty()
                )
            ),
            "ordinary_zero_cells_made_positive": 0,
            "seed_zero_cells_made_positive": 0,
            "solver_runtime_seconds": 0.0,
            "pulp_version": None,
            "solver_version": None,
            "solver_options_json": None,
        }
    ]

    for adjustment_index, attribute in enumerate(order[1:], start=1):
        prior_attributes = tuple(order[:adjustment_index])
        cache_key = (prior_attributes, attribute)
        if direct_reference_mode == "pool" and cache_key not in conditional_cache:
            assert count_pool is not None
            conditional_cache[cache_key] = build_pool_lookup(
                count_pool, prior_attributes, attribute
            )
        prepared = prepare_direct_adjustment(
            current,
            count_pool,
            attribute,
            targets[attribute],
            pool_lookup=(
                conditional_cache[cache_key]
                if direct_reference_mode == "pool"
                else None
            ),
            zero_cell_policy=zero_cell_policy,
            direct_reference_mode=direct_reference_mode,
        )
        if adjustment_backend == "min_cost_flow":
            solved = solve_direct_adjustment_flow(
                prepared, solver_mode=solver_mode
            )
        else:
            solved = solve_direct_adjustment(prepared, solver_mode=solver_mode)
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
