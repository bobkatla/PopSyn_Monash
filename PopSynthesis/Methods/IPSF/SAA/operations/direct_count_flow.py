"""Exact min-cost-flow backend for progressive direct-count adjustment."""

from __future__ import annotations

from fractions import Fraction
import json
import time
from typing import Literal

import networkx as nx
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import aggregate_counts
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_ilp import (
    DirectAdjustmentInput,
    DirectAdjustmentResult,
)


def _cell_pieces(reference: Fraction, capacity: int):
    """Return capacities and exact marginal costs for |y-reference|."""

    floor = reference.numerator // reference.denominator
    fraction = reference - floor
    used = min(floor, capacity)
    if used:
        yield used, Fraction(-1)
    if fraction and used < capacity:
        yield 1, Fraction(1) - 2 * fraction
        used += 1
    if used < capacity:
        yield capacity - used, Fraction(1)


def _build_result(
    data: DirectAdjustmentInput,
    values: dict[tuple[int, int], int],
    *,
    started: float,
    node_count: int,
    arc_count: int,
    solve_seconds: float,
    trivial: bool,
    solver_mode: Literal["single_stage", "two_stage"],
    stage1_objective: int = 0,
    stage1_solve_seconds: float = 0.0,
    stage1_node_count: int = 0,
    stage1_arc_count: int = 0,
) -> DirectAdjustmentResult:
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
    objective = sum(
        abs(Fraction(value) - data.references[key[0]][key[1]])
        for key, value in values.items()
    )
    ordinary_zero_positive = sum(
        1
        for row_index, pool_row in enumerate(data.pool_counts)
        for column_index, pool_value in enumerate(pool_row)
        if pool_value == 0 and values[(row_index, column_index)] > 0
    )
    permitted_cell_count = sum(
        permitted for row in data.permitted_cells for permitted in row
    )
    active_columns = sum(target > 0 for target in data.target_counts)
    marginal_l1_error = sum(
        abs(achieved[state] - target)
        for state, target in zip(data.states, data.target_counts)
    )
    if solver_mode == "two_stage" and marginal_l1_error != stage1_objective:
        raise RuntimeError(
            f"Flow Stage 2 for {data.attribute!r} has marginal error "
            f"{marginal_l1_error}, expected {stage1_objective}"
        )
    diagnostic = {
        "mode": "count_native_ilp",
        "adjustment_backend": "min_cost_flow",
        "solver_mode": (
            "two_stage" if solver_mode == "two_stage" else "exact_flow"
        ),
        "zone": data.zone,
        "active_attribute": data.attribute,
        "direct_reference_mode": data.direct_reference_mode,
        "zero_cell_policy": data.zero_cell_policy,
        "prior_row_count": len(data.row_keys),
        "ilp_cell_count": len(data.row_keys) * len(data.states),
        "permitted_cell_count": permitted_cell_count,
        "prohibited_cell_count": (
            len(data.row_keys) * len(data.states) - permitted_cell_count
        ),
        "relaxed_cell_count": data.relaxed_cell_count,
        "active_column_count": active_columns,
        "eliminated_zero_target_columns": (
            0
            if solver_mode == "two_stage"
            else len(data.states) - active_columns
        ),
        "stage1_status": (
            "Optimal" if solver_mode == "two_stage" else "SkippedExactConstraints"
        ),
        "stage1_objective": stage1_objective,
        "stage1_solver_gap": 0 if solver_mode == "two_stage" else None,
        "stage1_solve_seconds": stage1_solve_seconds,
        "stage1_variable_count": stage1_arc_count,
        "stage1_constraint_count": stage1_node_count,
        "stage2_status": "Optimal",
        "stage2_objective": float(objective),
        "stage2_objective_fraction": f"{objective.numerator}/{objective.denominator}",
        "stage2_objective_scale": None,
        "stage2_solver_gap": 0,
        "stage2_slack_objective": stage1_objective,
        "stage2_solve_seconds": solve_seconds,
        "stage2_variable_count": arc_count,
        "stage2_constraint_count": node_count,
        "target_counts_json": json.dumps(
            dict(zip(data.states, data.target_counts)), sort_keys=True
        ),
        "achieved_counts_json": json.dumps(achieved, sort_keys=True),
        "marginal_l1_error": marginal_l1_error,
        "row_totals_preserved": all(
            sum(values[(row_index, column_index)] for column_index in range(len(data.states)))
            == expected
            for row_index, expected in enumerate(data.row_totals)
        ),
        "unsupported_row_count": len(data.unsupported_rows),
        "unsupported_column_count": len(data.unsupported_columns),
        "ordinary_zero_cells_made_positive": ordinary_zero_positive,
        "seed_zero_cells_made_positive": ordinary_zero_positive,
        "solver_runtime_seconds": time.perf_counter() - started,
        "flow_solve_seconds": solve_seconds,
        "flow_node_count": node_count,
        "flow_arc_count": arc_count,
        "flow_trivial_solution": trivial,
        "pulp_version": None,
        "solver_version": nx.__version__,
        "solver_options_json": None,
    }
    return DirectAdjustmentResult(result, diagnostic)


def solve_direct_adjustment_flow(
    data: DirectAdjustmentInput,
    *,
    solver_mode: Literal["single_stage", "two_stage"] = "single_stage",
) -> DirectAdjustmentResult:
    """Solve the dense integer transportation problem exactly as min-cost flow."""

    if solver_mode not in {"single_stage", "two_stage"}:
        raise ValueError("solver_mode must be 'single_stage' or 'two_stage'")
    if solver_mode == "single_stage" and data.zero_cell_policy != "unrestricted":
        raise ValueError("Restricted zero-cell policies require two-stage solving")
    started = time.perf_counter()
    row_count = len(data.row_keys)
    column_count = len(data.states)
    active_columns = [
        index for index, target in enumerate(data.target_counts) if target > 0
    ]
    values = {
        (row_index, column_index): 0
        for row_index in range(row_count)
        for column_index in range(column_count)
    }

    if solver_mode == "single_stage" and len(active_columns) == 1:
        column_index = active_columns[0]
        for row_index, row_total in enumerate(data.row_totals):
            values[(row_index, column_index)] = row_total
        return _build_result(
            data,
            values,
            started=started,
            node_count=0,
            arc_count=0,
            solve_seconds=0.0,
            trivial=True,
            solver_mode=solver_mode,
        )
    if solver_mode == "single_stage" and row_count == 1:
        for column_index in active_columns:
            values[(0, column_index)] = data.target_counts[column_index]
        return _build_result(
            data,
            values,
            started=started,
            node_count=0,
            arc_count=0,
            solve_seconds=0.0,
            trivial=True,
            solver_mode=solver_mode,
        )

    model_columns = (
        list(range(column_count))
        if solver_mode == "two_stage"
        else active_columns
    )
    total_supply = sum(data.row_totals)
    target_total = sum(data.target_counts)
    stage1_objective = 0
    stage1_solve_seconds = 0.0
    stage1_node_count = 0
    stage1_arc_count = 0
    if solver_mode == "two_stage":
        stage1_graph = nx.MultiDiGraph()
        for row_index, supply in enumerate(data.row_totals):
            stage1_graph.add_node(("row", row_index), demand=-supply)
        for column_index in model_columns:
            stage1_graph.add_node(
                ("column", column_index),
                demand=data.target_counts[column_index],
            )
        for row_index, supply in enumerate(data.row_totals):
            for column_index in model_columns:
                if not data.permitted_cells[row_index][column_index]:
                    continue
                stage1_graph.add_edge(
                    ("row", row_index),
                    ("column", column_index),
                    capacity=supply,
                    weight=0,
                )
        stage1_graph.add_node(
            "slack", demand=total_supply - target_total
        )
        for column_index in model_columns:
            stage1_graph.add_edge(
                ("column", column_index),
                "slack",
                capacity=total_supply,
                weight=1,
            )
            if data.target_counts[column_index] > 0:
                stage1_graph.add_edge(
                    "slack",
                    ("column", column_index),
                    capacity=data.target_counts[column_index],
                    weight=1,
                )
        stage1_node_count = stage1_graph.number_of_nodes()
        stage1_arc_count = stage1_graph.number_of_edges()
        stage1_started = time.perf_counter()
        stage1_cost, _ = nx.network_simplex(stage1_graph)
        stage1_solve_seconds = time.perf_counter() - stage1_started
        stage1_objective = int(stage1_cost)
        total_difference = total_supply - target_total
        if (
            stage1_objective < abs(total_difference)
            or (stage1_objective - total_difference) % 2
        ):
            raise RuntimeError(
                f"Flow Stage 1 for {data.attribute!r} returned invalid slack "
                f"objective {stage1_objective}"
            )

    graph = nx.MultiDiGraph()
    for row_index, row_total in enumerate(data.row_totals):
        graph.add_node(("row", row_index), demand=-row_total)
    for column_index in model_columns:
        graph.add_node(
            ("column", column_index),
            demand=data.target_counts[column_index],
        )
    over_supply = 0
    under_supply = 0
    if solver_mode == "two_stage":
        total_difference = total_supply - target_total
        over_supply = (stage1_objective + total_difference) // 2
        under_supply = (stage1_objective - total_difference) // 2
        if over_supply:
            graph.add_node("over_supply", demand=over_supply)
            for column_index in model_columns:
                graph.add_edge(
                    ("column", column_index),
                    "over_supply",
                    capacity=over_supply,
                    weight=0,
                )
        if under_supply:
            graph.add_node("under_supply", demand=-under_supply)
            for column_index in model_columns:
                if data.target_counts[column_index] > 0:
                    graph.add_edge(
                        "under_supply",
                        ("column", column_index),
                        capacity=data.target_counts[column_index],
                        weight=0,
                    )

    edge_keys: dict[tuple[int, int], list[int]] = {}
    for row_index, row_total in enumerate(data.row_totals):
        for column_index in model_columns:
            if not data.permitted_cells[row_index][column_index]:
                continue
            column_capacity = data.target_counts[column_index]
            if solver_mode == "two_stage":
                column_capacity += over_supply
            capacity = min(row_total, column_capacity)
            keys = []
            for piece_capacity, cost in _cell_pieces(
                data.references[row_index][column_index], capacity
            ):
                keys.append(
                    graph.add_edge(
                        ("row", row_index),
                        ("column", column_index),
                        capacity=piece_capacity,
                        weight=cost,
                    )
                )
            edge_keys[(row_index, column_index)] = keys

    solve_started = time.perf_counter()
    _, flow = nx.network_simplex(graph)
    solve_seconds = time.perf_counter() - solve_started
    for key, keys in edge_keys.items():
        row_index, column_index = key
        values[key] = sum(
            flow[("row", row_index)][("column", column_index)][edge_key]
            for edge_key in keys
        )

    return _build_result(
        data,
        values,
        started=started,
        node_count=graph.number_of_nodes(),
        arc_count=graph.number_of_edges(),
        solve_seconds=solve_seconds,
        trivial=False,
        solver_mode=solver_mode,
        stage1_objective=stage1_objective,
        stage1_solve_seconds=stage1_solve_seconds,
        stage1_node_count=stage1_node_count,
        stage1_arc_count=stage1_arc_count,
    )
