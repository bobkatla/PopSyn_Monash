"""Compare CBC and exact min-cost flow on identical direct-adjustment inputs."""

from __future__ import annotations

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field, zone_field
from PopSynthesis.Methods.IPSF.run import process_data_from_files, _prepare_direct_pool
from PopSynthesis.Methods.IPSF.SAA.direct_count_saa import (
    _targets_by_attribute,
    run_direct_count_pass,
)
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_flow import (
    solve_direct_adjustment_flow,
)
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_ilp import (
    build_pool_lookup,
    prepare_direct_adjustment,
    solve_direct_adjustment,
)


ATTRIBUTES = ["hhsize", "totalvehs", "dwelltype", "owndwell", "hhinc"]
DEFAULT_ZONES = [3034, 3021, 3073, 3049, 3020, 3082, 3000, 3070]


def _initial_counts(zone: str, targets: dict[str, dict[str, int]]) -> pl.DataFrame:
    first = ATTRIBUTES[0]
    return pl.DataFrame(
        [
            {zone_field: zone, first: state, count_field: count}
            for state, count in targets[first].items()
            if count > 0
        ]
    ).with_columns(pl.col(count_field).cast(pl.Int64))


def benchmark_zone(
    zone_marginal: pd.DataFrame,
    pool: pl.DataFrame,
    flow_repeats: int,
    solver_mode: str = "single_stage",
    zero_cell_policy: str = "unrestricted",
) -> list[dict[str, object]]:
    zone, targets = _targets_by_attribute(zone_marginal, ATTRIBUTES)
    current = _initial_counts(zone, targets)
    cache = {}
    records: list[dict[str, object]] = []

    for adjustment_index, attribute in enumerate(ATTRIBUTES[1:], start=1):
        prior = tuple(ATTRIBUTES[:adjustment_index])
        cache_key = (prior, attribute)
        prepare_started = perf_counter()
        if cache_key not in cache:
            cache[cache_key] = build_pool_lookup(pool, prior, attribute)
        prepared = prepare_direct_adjustment(
            current,
            pool,
            attribute,
            targets[attribute],
            pool_lookup=cache[cache_key],
            zero_cell_policy=zero_cell_policy,
        )
        prepare_seconds = perf_counter() - prepare_started

        cbc_started = perf_counter()
        cbc = solve_direct_adjustment(prepared, solver_mode=solver_mode)
        cbc_seconds = perf_counter() - cbc_started

        flow_results = []
        flow_times = []
        for _ in range(flow_repeats):
            flow_started = perf_counter()
            flow_results.append(
                solve_direct_adjustment_flow(
                    prepared, solver_mode=solver_mode
                )
            )
            flow_times.append(perf_counter() - flow_started)
        flow = flow_results[0]
        if any(not flow.counts.equals(result.counts) for result in flow_results[1:]):
            raise AssertionError(f"Flow was not repeatable for {zone}.{attribute}")
        if (
            cbc.diagnostic["stage1_objective"]
            != flow.diagnostic["stage1_objective"]
        ):
            raise AssertionError(
                f"Stage 1 objective mismatch for {zone}.{attribute}"
            )
        if (
            cbc.diagnostic["stage2_objective_fraction"]
            != flow.diagnostic["stage2_objective_fraction"]
        ):
            raise AssertionError(
                f"Objective mismatch for {zone}.{attribute}: "
                f"CBC={cbc.diagnostic['stage2_objective_fraction']}, "
                f"flow={flow.diagnostic['stage2_objective_fraction']}"
            )

        common = {
            "zone": zone,
            "adjustment_index": adjustment_index,
            "attribute": attribute,
            "prepare_seconds": prepare_seconds,
            "prior_rows": len(prepared.row_keys),
            "states": len(prepared.states),
            "dense_cells": len(prepared.row_keys) * len(prepared.states),
            "active_columns": sum(value > 0 for value in prepared.target_counts),
            "unsupported_rows": len(prepared.unsupported_rows),
            "solver_mode": solver_mode,
            "zero_cell_policy": zero_cell_policy,
            "stage1_objective": cbc.diagnostic["stage1_objective"],
            "objective_fraction": cbc.diagnostic["stage2_objective_fraction"],
        }
        records.append({**common, "backend": "cbc", "solve_seconds": cbc_seconds})
        records.append(
            {
                **common,
                "backend": "min_cost_flow",
                "solve_seconds": sum(flow_times) / len(flow_times),
                "flow_min_seconds": min(flow_times),
                "flow_repeats": flow_repeats,
                "flow_nodes": flow.diagnostic["flow_node_count"],
                "flow_arcs": flow.diagnostic["flow_arc_count"],
            }
        )
        current = cbc.counts

    end_to_end = []
    end_to_end_times = []
    for _ in range(flow_repeats):
        started = perf_counter()
        end_to_end.append(
            run_direct_count_pass(
                zone_marginal,
                pool,
                ATTRIBUTES,
                pass_index=0,
                solver_mode=solver_mode,
                adjustment_backend="min_cost_flow",
                zero_cell_policy=zero_cell_policy,
                conditional_cache={},
            ).population
        )
        end_to_end_times.append(perf_counter() - started)
    if any(not end_to_end[0].equals(result) for result in end_to_end[1:]):
        raise AssertionError(f"End-to-end flow was not repeatable for zone {zone}")
    records.append(
        {
            "zone": zone,
            "adjustment_index": "all",
            "attribute": "all",
            "backend": "min_cost_flow_end_to_end",
            "solve_seconds": sum(end_to_end_times) / len(end_to_end_times),
            "flow_min_seconds": min(end_to_end_times),
            "flow_repeats": flow_repeats,
            "output_rows": end_to_end[0].height,
        }
    )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", nargs="+", type=int, default=DEFAULT_ZONES)
    parser.add_argument("--flow-repeats", type=int, default=3)
    parser.add_argument(
        "--solver-mode",
        choices=["single_stage", "two_stage"],
        default="single_stage",
    )
    parser.add_argument(
        "--zero-cell-policy",
        choices=[
            "unrestricted",
            "seed_only",
            "relax_empty_rows_columns",
        ],
        default="unrestricted",
    )
    parser.add_argument(
        "--marginals",
        type=Path,
        default=Path("IO/processed_data/census_POA_hh_saa.csv"),
    )
    parser.add_argument(
        "--pool",
        type=Path,
        default=Path("IO/processed_data/ori_sample_hh.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("IO/output/direct_backend_benchmark/results.csv"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.flow_repeats < 1:
        raise ValueError("flow-repeats must be positive")
    marginals, raw_pool = process_data_from_files(
        arguments.marginals, arguments.pool, "zone_id"
    )
    pool = _prepare_direct_pool(raw_pool, ATTRIBUTES, "_weight")
    benchmark_records = []
    for zone in arguments.zones:
        print(f"Benchmarking zone {zone}", flush=True)
        benchmark_records.extend(
            benchmark_zone(
                marginals.loc[[zone]],
                pool,
                arguments.flow_repeats,
                arguments.solver_mode,
                arguments.zero_cell_policy,
            )
        )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(benchmark_records).to_csv(arguments.output, index=False)
    print(f"Wrote {arguments.output}")
