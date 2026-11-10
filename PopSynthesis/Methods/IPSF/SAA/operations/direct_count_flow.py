"""Exact min-cost-flow backend for progressive direct-count adjustment."""

from __future__ import annotations

from fractions import Fraction
import json
import time

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
    active_columns = sum(target > 0 for target in data.target_counts)
    diagnostic = {
        "mode": "count_native_ilp",
        "adjustment_backend": "min_cost_flow",
        "solver_mode": "exact_flow",
        "zone": data.zone,
        "active_attribute": data.attribute,
        "prior_row_count": len(data.row_keys),
        "ilp_cell_count": len(data.row_keys) * len(data.states),
        "active_column_count": active_columns,
        "eliminated_zero_target_columns": len(data.states) - active_columns,
        "stage1_status": "SkippedExactConstraints",
        "stage1_objective": 0.0,
        "stage2_status": "Optimal",
        "stage2_objective": float(objective),
        "stage2_objective_fraction": f"{objective.numerator}/{objective.denominator}",
        "stage2_objective_scale": None,
        "stage2_solver_gap": 0,
        "target_counts_json": json.dumps(
            dict(zip(data.states, data.target_counts)), sort_keys=True
        ),
        "achieved_counts_json": json.dumps(achieved, sort_keys=True),
        "marginal_l1_error": sum(
            abs(achieved[state] - target)
            for state, target in zip(data.states, data.target_counts)
        ),
        "row_totals_preserved": all(
            sum(values[(row_index, column_index)] for column_index in range(len(data.states)))
            == expected
            for row_index, expected in enumerate(data.row_totals)
        ),
        "unsupported_row_count": len(data.unsupported_rows),
        "unsupported_column_count": len(data.unsupported_columns),
        "ordinary_zero_cells_made_positive": ordinary_zero_positive,
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
) -> DirectAdjustmentResult:
    """Solve the dense integer transportation problem exactly as min-cost flow."""

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

    if len(active_columns) == 1:
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
        )
    if row_count == 1:
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
        )

    graph = nx.MultiDiGraph()
    for row_index, row_total in enumerate(data.row_totals):
        graph.add_node(("row", row_index), demand=-row_total)
    for column_index in active_columns:
        graph.add_node(
            ("column", column_index),
            demand=data.target_counts[column_index],
        )

    edge_keys: dict[tuple[int, int], list[int]] = {}
    for row_index, row_total in enumerate(data.row_totals):
        for column_index in active_columns:
            capacity = min(row_total, data.target_counts[column_index])
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
    )
