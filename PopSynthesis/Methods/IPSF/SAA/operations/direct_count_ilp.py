"""Dense lexicographic ILP for sampling-free progressive count adjustment."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from fractions import Fraction
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time
from typing import Literal, Mapping, Sequence

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
PoolLookup = dict[tuple[str, ...], Fraction]


@dataclass(frozen=True)
class DirectAdjustmentInput:
    zone: str
    attribute: str
    prior_attributes: tuple[str, ...]
    states: tuple[str, ...]
    row_keys: tuple[tuple[str, ...], ...]
    row_totals: tuple[int, ...]
    target_counts: tuple[int, ...]
    pool_counts: tuple[tuple[Fraction, ...], ...]
    references: tuple[tuple[Fraction, ...], ...]
    unsupported_rows: tuple[tuple[str, ...], ...]
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
    *,
    pool_lookup: PoolLookup | None = None,
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

    if pool_lookup is None:
        pool_lookup = build_pool_lookup(count_pool, prior_attributes, attribute)

    pool_rows: list[tuple[Fraction, ...]] = []
    reference_rows: list[tuple[Fraction, ...]] = []
    unsupported_rows: list[tuple[str, ...]] = []
    for key, row_total in zip(row_keys, row_totals):
        pool_row = tuple(
            pool_lookup.get(key + (state,), Fraction(0)) for state in states
        )
        support = sum(pool_row)
        if support == 0:
            unsupported_rows.append(key)
        pool_rows.append(pool_row)
        reference_rows.append(
            tuple(Fraction(0) for _ in states)
            if support == 0
            else tuple(
                Fraction(row_total) * value / support for value in pool_row
            )
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
        unsupported_rows=tuple(unsupported_rows),
        unsupported_columns=unsupported_columns,
    )


def build_pool_lookup(
    count_pool: pl.DataFrame,
    prior_attributes: Sequence[str],
    attribute: str,
) -> PoolLookup:
    """Aggregate exact pool weights once for a progressive adjustment."""

    lookup: PoolLookup = {}
    selected = count_pool.select(list(prior_attributes) + [attribute, count_field])
    for row in selected.iter_rows(named=True):
        key = tuple(str(row[name]) for name in prior_attributes) + (
            str(row[attribute]),
        )
        weight = Fraction(Decimal(str(row[count_field])))
        lookup[key] = lookup.get(key, Fraction(0)) + weight
    return lookup


def _require_optimal(
    stats: object,
    stage: str,
    attribute: str,
    *,
    allow_zero_gap: bool = False,
) -> None:
    zero_gap_optimum = (
        allow_zero_gap
        and stats.status == LpSolveStatus.GapLimit
        and stats.has_solution
        and stats.solver_gap == 0
        and stats.solver_status == "Optimal solution found"
    )
    if stats.status != LpSolveStatus.Optimal and not zero_gap_optimum:
        raise RuntimeError(
            f"{stage} for {attribute!r} did not prove optimality: "
            f"status={stats.status_str}, has_solution={stats.has_solution}"
        )


def _solve_with_cbc_log(problem: LpProblem):
    """Solve while retaining CBC's reported status and gap in PuLP stats."""

    with TemporaryDirectory(prefix="popsyn-cbc-") as directory:
        log_path = Path(directory) / "cbc.log"
        return problem.solve(
            create_cbc_solver(reproducible=True, log_path=log_path)
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


def _base_variables(
    problem: LpProblem,
    data: DirectAdjustmentInput,
    column_indices: Sequence[int],
):
    return {
        (row_index, column_index): problem.add_variable(
            f"Y_{row_index}_{column_index}", lowBound=0, cat="Integer"
        )
        for row_index in range(len(data.row_keys))
        for column_index in column_indices
    }


def _add_row_constraints(problem, variables, data, column_indices):
    for row_index, row_total in enumerate(data.row_totals):
        problem += (
            lpSum(
                variables[(row_index, column_index)]
                for column_index in column_indices
            )
            == row_total,
            f"row_{row_index}",
        )


def solve_direct_adjustment(
    data: DirectAdjustmentInput,
    *,
    solver_mode: Literal["single_stage", "two_stage"] = "single_stage",
) -> DirectAdjustmentResult:
    """Solve the Census-first and pool-reference objectives lexicographically."""

    if solver_mode not in {"single_stage", "two_stage"}:
        raise ValueError("solver_mode must be 'single_stage' or 'two_stage'")
    started = time.perf_counter()
    active_columns = [
        index for index, target in enumerate(data.target_counts) if target > 0
    ]
    stage1_stats = None
    stage1_objective = 0.0
    if solver_mode == "two_stage":
        stage1 = LpProblem(f"DirectCountStage1_{data.attribute}", LpMinimize)
        stage1_y = _base_variables(stage1, data, active_columns)
        slack_positive = {
            column_index: stage1.add_variable(
                f"L_pos_{column_index}", lowBound=0, cat="Continuous"
            )
            for column_index in active_columns
        }
        slack_negative = {
            column_index: stage1.add_variable(
                f"L_neg_{column_index}", lowBound=0, cat="Continuous"
            )
            for column_index in active_columns
        }
        _add_row_constraints(stage1, stage1_y, data, active_columns)
        for column_index in active_columns:
            target = data.target_counts[column_index]
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
    stage2_y = _base_variables(stage2, data, active_columns)
    deviations = {
        (row_index, column_index): stage2.add_variable(
            f"D_{row_index}_{column_index}", lowBound=0, cat="Continuous"
        )
        for row_index in range(len(data.row_keys))
        for column_index in active_columns
    }
    _add_row_constraints(stage2, stage2_y, data, active_columns)
    for column_index in active_columns:
        target = data.target_counts[column_index]
        stage2 += (
            lpSum(
                stage2_y[(row_index, column_index)]
                for row_index in range(len(data.row_keys))
            )
            == target,
            f"column_{column_index}",
        )
    for key, deviation in deviations.items():
        reference = float(data.references[key[0]][key[1]])
        stage2 += deviation >= stage2_y[key] - reference
        stage2 += deviation >= reference - stage2_y[key]
    stage2 += lpSum(deviations.values())
    stage2_stats = _solve_with_cbc_log(stage2)
    _require_optimal(
        stage2_stats, "Stage 2", data.attribute, allow_zero_gap=True
    )

    values = {
        (row_index, column_index): 0
        for row_index in range(len(data.row_keys))
        for column_index in range(len(data.states))
    }
    values.update(
        {
            key: _integer_value(variable, variable.name)
            for key, variable in stage2_y.items()
        }
    )
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
        "adjustment_backend": "cbc",
        "solver_mode": solver_mode,
        "zone": data.zone,
        "active_attribute": data.attribute,
        "prior_row_count": len(data.row_keys),
        "ilp_cell_count": len(data.row_keys) * len(data.states),
        "active_column_count": len(active_columns),
        "eliminated_zero_target_columns": len(data.states) - len(active_columns),
        "stage1_status": (
            stage1_stats.status_str
            if stage1_stats is not None
            else "SkippedExactConstraints"
        ),
        "stage1_objective": stage1_objective,
        "stage2_status": stage2_stats.status_str,
        "stage2_objective": float(exact_stage2_objective),
        "stage2_objective_fraction": (
            f"{exact_stage2_objective.numerator}/{exact_stage2_objective.denominator}"
        ),
        "stage2_objective_scale": None,
        "stage2_solver_gap": stage2_stats.solver_gap,
        "target_counts_json": json.dumps(
            dict(zip(data.states, data.target_counts)), sort_keys=True
        ),
        "achieved_counts_json": json.dumps(achieved, sort_keys=True),
        "marginal_l1_error": sum(
            abs(achieved[state] - target)
            for state, target in zip(data.states, data.target_counts)
        ),
        "row_totals_preserved": preserved_rows,
        "unsupported_row_count": len(data.unsupported_rows),
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
