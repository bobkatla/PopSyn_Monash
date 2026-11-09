"""Dense lexicographic ILP for sampling-free progressive count adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import json
from math import lcm
import time
from typing import Mapping, Sequence

import polars as pl
import pulp
from pulp import LpMinimize, LpProblem, LpSolveStatus, lpSum

from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.SAA.operations.cbc_solver import (
    CBC_RANDOM_SEED,
    create_cbc_solver,
)
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import aggregate_counts


INTEGER_TOLERANCE = 1e-6
OBJECTIVE_TOLERANCE = 1e-7


@dataclass(frozen=True)
class DirectAdjustmentInput:
    zone: str
    attribute: str
    prior_attributes: tuple[str, ...]
    states: tuple[str, ...]
    row_keys: tuple[tuple[str, ...], ...]
    row_totals: tuple[int, ...]
    target_counts: tuple[int, ...]
    pool_counts: tuple[tuple[int, ...], ...]
    references: tuple[tuple[Fraction, ...], ...]
    unsupported_columns: tuple[str, ...]


@dataclass(frozen=True)
class DirectAdjustmentResult:
    counts: pl.DataFrame
    diagnostic: dict[str, object]


def prepare_direct_adjustment(
    partial_counts: pl.DataFrame,
    count_pool: pl.DataFrame,
    attribute: str,
    target_counts: Mapping[str, int],
) -> DirectAdjustmentInput:
    """Build the complete prior-row by current-state pool contingency."""

    required = {zone_field, count_field}
    if not required <= set(partial_counts.columns):
        raise ValueError(f"Partial counts must contain {sorted(required)}")
    zones = partial_counts[zone_field].cast(pl.String).unique().to_list()
    if len(zones) != 1:
        raise ValueError("Direct-count adjustment requires exactly one zone")

    prior_attributes = tuple(
        column
        for column in partial_counts.columns
        if column not in {zone_field, count_field}
    )
    if not prior_attributes:
        raise ValueError("At least one prior attribute is required")
    if attribute in prior_attributes:
        raise ValueError(f"Attribute {attribute!r} was already introduced")
    if partial_counts.select(
        [zone_field, *prior_attributes]
    ).is_duplicated().any():
        raise ValueError("Partial count rows must have unique keys")
    partial_values = partial_counts[count_field]
    if not partial_values.dtype.is_integer() or (partial_values <= 0).any():
        raise ValueError("Partial counts must be positive integers")
    if partial_counts.select(
        pl.any_horizontal(
            [pl.col(name).is_null() for name in [zone_field, *prior_attributes]]
        ).any()
    ).item():
        raise ValueError("Partial count keys cannot contain nulls")
    missing_pool = set(prior_attributes + (attribute, count_field)) - set(
        count_pool.columns
    )
    if missing_pool:
        raise ValueError(f"Pool is missing columns: {sorted(missing_pool)}")

    states = tuple(str(state) for state in target_counts)
    targets = tuple(int(target_counts[state]) for state in target_counts)
    sorted_partial = partial_counts.sort(list(prior_attributes))
    row_keys: list[tuple[str, ...]] = []
    row_totals: list[int] = []
    for row in sorted_partial.iter_rows(named=True):
        row_keys.append(tuple(str(row[name]) for name in prior_attributes))
        row_totals.append(int(row[count_field]))

    pool_lookup: dict[tuple[str, ...], int] = {}
    selected = count_pool.select(list(prior_attributes) + [attribute, count_field])
    for row in selected.iter_rows(named=True):
        key = tuple(str(row[name]) for name in prior_attributes) + (
            str(row[attribute]),
        )
        pool_lookup[key] = pool_lookup.get(key, 0) + int(row[count_field])

    pool_rows: list[tuple[int, ...]] = []
    reference_rows: list[tuple[Fraction, ...]] = []
    unsupported_rows: list[tuple[str, ...]] = []
    for key, row_total in zip(row_keys, row_totals):
        pool_row = tuple(pool_lookup.get(key + (state,), 0) for state in states)
        support = sum(pool_row)
        if row_total > 0 and support == 0:
            unsupported_rows.append(key)
            continue
        pool_rows.append(pool_row)
        reference_rows.append(
            tuple(Fraction(row_total * value, support) for value in pool_row)
        )

    if unsupported_rows:
        rendered = ", ".join(repr(row) for row in unsupported_rows[:5])
        suffix = "" if len(unsupported_rows) <= 5 else ", ..."
        raise ValueError(
            f"{len(unsupported_rows)} completely unsupported prior row(s) "
            f"while adjusting {attribute!r}: {rendered}{suffix}"
        )

    unsupported_columns = tuple(
        state
        for column_index, state in enumerate(states)
        if sum(row[column_index] for row in pool_rows) == 0
    )
    return DirectAdjustmentInput(
        zone=str(zones[0]),
        attribute=attribute,
        prior_attributes=prior_attributes,
        states=states,
        row_keys=tuple(row_keys),
        row_totals=tuple(row_totals),
        target_counts=targets,
        pool_counts=tuple(pool_rows),
        references=tuple(reference_rows),
        unsupported_columns=unsupported_columns,
    )


def _require_optimal(stats: object, stage: str, attribute: str) -> None:
    if stats.status != LpSolveStatus.Optimal:
        raise RuntimeError(
            f"{stage} for {attribute!r} did not prove optimality: "
            f"status={stats.status_str}, has_solution={stats.has_solution}"
        )


def _integer_value(variable: object, name: str) -> int:
    value = variable.value()
    if value is None:
        raise RuntimeError(f"Solver returned no value for {name}")
    rounded = round(float(value))
    if abs(float(value) - rounded) > INTEGER_TOLERANCE:
        raise RuntimeError(f"Solver returned non-integral value {value} for {name}")
    if rounded < 0:
        raise RuntimeError(f"Solver returned negative value {value} for {name}")
    return int(rounded)


def _base_variables(problem: LpProblem, data: DirectAdjustmentInput):
    return {
        (row_index, column_index): problem.add_variable(
            f"Y_{row_index}_{column_index}", lowBound=0, cat="Integer"
        )
        for row_index in range(len(data.row_keys))
        for column_index in range(len(data.states))
    }


def _add_row_constraints(problem, variables, data):
    for row_index, row_total in enumerate(data.row_totals):
        problem += (
            lpSum(
                variables[(row_index, column_index)]
                for column_index in range(len(data.states))
            )
            == row_total,
            f"row_{row_index}",
        )


def solve_direct_adjustment(data: DirectAdjustmentInput) -> DirectAdjustmentResult:
    """Solve the Census-first and pool-reference objectives lexicographically."""

    started = time.perf_counter()
    stage1 = LpProblem(f"DirectCountStage1_{data.attribute}", LpMinimize)
    stage1_y = _base_variables(stage1, data)
    slack_positive = {
        column_index: stage1.add_variable(
            f"L_pos_{column_index}", lowBound=0, cat="Continuous"
        )
        for column_index in range(len(data.states))
    }
    slack_negative = {
        column_index: stage1.add_variable(
            f"L_neg_{column_index}", lowBound=0, cat="Continuous"
        )
        for column_index in range(len(data.states))
    }
    _add_row_constraints(stage1, stage1_y, data)
    for column_index, target in enumerate(data.target_counts):
        stage1 += (
            lpSum(
                stage1_y[(row_index, column_index)]
                for row_index in range(len(data.row_keys))
            )
            == target
            + slack_positive[column_index]
            - slack_negative[column_index],
            f"column_{column_index}",
        )
    stage1 += lpSum(slack_positive.values()) + lpSum(slack_negative.values())
    stage1_stats = stage1.solve(create_cbc_solver(reproducible=True))
    _require_optimal(stage1_stats, "Stage 1", data.attribute)
    stage1_objective = float(stage1_stats.objective or 0.0)
    if abs(stage1_objective) > OBJECTIVE_TOLERANCE:
        raise RuntimeError(
            f"Stage 1 for {data.attribute!r} has nonzero marginal error "
            f"{stage1_objective}"
        )

    stage2 = LpProblem(f"DirectCountStage2_{data.attribute}", LpMinimize)
    stage2_y = _base_variables(stage2, data)
    scaled_deviations = {
        (row_index, column_index): stage2.add_variable(
            f"E_{row_index}_{column_index}", lowBound=0, cat="Continuous"
        )
        for row_index in range(len(data.row_keys))
        for column_index in range(len(data.states))
    }
    _add_row_constraints(stage2, stage2_y, data)
    for column_index, target in enumerate(data.target_counts):
        stage2 += (
            lpSum(
                stage2_y[(row_index, column_index)]
                for row_index in range(len(data.row_keys))
            )
            == target,
            f"column_{column_index}",
        )
    # CBC's proof/status detection can mistake a sub-ULP negative gap on a
    # fractional objective for termination at the gap limit.  Use the least
    # common denominator to express the *same* global L1 objective with integer
    # coefficients; unlike per-row scaling, this preserves relative row
    # weights exactly.
    objective_scale = lcm(
        *(reference.denominator for row in data.references for reference in row)
    )
    objective_terms = []
    for key, deviation in scaled_deviations.items():
        reference = data.references[key[0]][key[1]]
        stage2 += (
            deviation
            >= reference.denominator * stage2_y[key] - reference.numerator
        )
        stage2 += (
            deviation
            >= reference.numerator - reference.denominator * stage2_y[key]
        )
        objective_terms.append(
            (objective_scale // reference.denominator) * deviation
        )
    stage2 += lpSum(objective_terms)
    stage2_stats = stage2.solve(create_cbc_solver(reproducible=True))
    _require_optimal(stage2_stats, "Stage 2", data.attribute)

    values = {
        key: _integer_value(variable, variable.name)
        for key, variable in stage2_y.items()
    }
    exact_stage2_objective = sum(
        abs(Fraction(value) - data.references[key[0]][key[1]])
        for key, value in values.items()
    )
    rows: list[dict[str, object]] = []
    for row_index, row_key in enumerate(data.row_keys):
        for column_index, state in enumerate(data.states):
            value = values[(row_index, column_index)]
            if value == 0:
                continue
            row: dict[str, object] = {zone_field: data.zone}
            row.update(zip(data.prior_attributes, row_key))
            row[data.attribute] = state
            row[count_field] = value
            rows.append(row)
    result = aggregate_counts(
        pl.DataFrame(rows),
        [zone_field, *data.prior_attributes, data.attribute],
    )

    achieved = {
        state: sum(
            values[(row_index, column_index)]
            for row_index in range(len(data.row_keys))
        )
        for column_index, state in enumerate(data.states)
    }
    preserved_rows = all(
        sum(
            values[(row_index, column_index)]
            for column_index in range(len(data.states))
        )
        == expected
        for row_index, expected in enumerate(data.row_totals)
    )
    ordinary_zero_positive = sum(
        1
        for row_index, pool_row in enumerate(data.pool_counts)
        for column_index, pool_value in enumerate(pool_row)
        if pool_value == 0 and values[(row_index, column_index)] > 0
    )
    diagnostic = {
        "mode": "count_native_ilp",
        "zone": data.zone,
        "active_attribute": data.attribute,
        "prior_row_count": len(data.row_keys),
        "ilp_cell_count": len(data.row_keys) * len(data.states),
        "stage1_status": stage1_stats.status_str,
        "stage1_objective": stage1_objective,
        "stage2_status": stage2_stats.status_str,
        "stage2_objective": float(exact_stage2_objective),
        "stage2_objective_scale": objective_scale,
        "target_counts_json": json.dumps(
            dict(zip(data.states, data.target_counts)), sort_keys=True
        ),
        "achieved_counts_json": json.dumps(achieved, sort_keys=True),
        "marginal_l1_error": sum(
            abs(achieved[state] - target)
            for state, target in zip(data.states, data.target_counts)
        ),
        "row_totals_preserved": preserved_rows,
        "unsupported_row_count": 0,
        "unsupported_column_count": len(data.unsupported_columns),
        "ordinary_zero_cells_made_positive": ordinary_zero_positive,
        "solver_runtime_seconds": time.perf_counter() - started,
        "pulp_version": pulp.__version__,
        "solver_version": stage2_stats.solver_version,
        "solver_options_json": json.dumps(
            {
                "threads": 1,
                "randomSeed": CBC_RANDOM_SEED,
                "randomCbcSeed": CBC_RANDOM_SEED,
                "gapRel": 0.0,
                "gapAbs": 0.0,
            },
            sort_keys=True,
        ),
    }
    return DirectAdjustmentResult(result, diagnostic)
