"""Reusable weighted household IPF matching the production notebook workflow."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from ipfn import ipfn

from PopSynthesis.Generator_data.generate_combine_census.utils import TRS


@dataclass(frozen=True)
class WeightedIPFResult:
    counts: pd.DataFrame
    fractional: pd.DataFrame
    diagnostics: pd.DataFrame
    fold_log: list[dict[str, object]]
    input_rows: int
    joint_cells: int


_IPF_WORKER_CONTEXT: dict[str, object] | None = None


def _initialise_ipf_worker(context: dict[str, object]) -> None:
    global _IPF_WORKER_CONTEXT
    _IPF_WORKER_CONTEXT = context


def _zone_seed(seed: int, position: int) -> int:
    return int(np.random.SeedSequence([seed, position]).generate_state(1)[0])


def read_household_census(
    source: Path | str | pd.DataFrame,
    attributes: Sequence[str],
    zone_field: str,
) -> pd.DataFrame:
    """Read and validate the repository's two-row household census format."""

    frame = (
        source.copy()
        if isinstance(source, pd.DataFrame)
        else pd.read_csv(source, header=[0, 1])
    )
    if not isinstance(frame.columns, pd.MultiIndex) or frame.columns.nlevels != 2:
        raise ValueError("Census columns must be a two-level MultiIndex")
    zone_columns = [column for column in frame.columns if column[0] == zone_field]
    if len(zone_columns) != 1:
        raise ValueError(
            f"Expected exactly one census zone column {zone_field!r}; "
            f"found {len(zone_columns)}"
        )
    missing = set(attributes) - set(frame.columns.get_level_values(0))
    if missing:
        raise ValueError(f"Census is missing attributes: {sorted(missing)}")
    frame = frame.set_index(zone_columns[0])
    frame.index.name = zone_field
    if not frame.index.is_unique:
        raise ValueError("Census zone identifiers must be unique")
    frame = frame.loc[:, frame.columns.get_level_values(0).isin(attributes)]
    frame = frame.apply(pd.to_numeric, errors="raise")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Census counts must be finite and nonnegative")
    return frame


def prepare_weighted_cells(
    source: Path | str | pd.DataFrame,
    attributes: Sequence[str],
    weight_column: str,
) -> tuple[pd.DataFrame, int]:
    """Condense weighted records without expanding them into raw households."""

    columns = [*attributes, weight_column]
    frame = (
        source.loc[:, columns].copy()
        if isinstance(source, pd.DataFrame)
        else pd.read_csv(source, usecols=columns)
    )
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"Weighted pool is missing columns: {sorted(missing)}")
    if frame[columns].isna().any().any():
        raise ValueError("Weighted pool contains missing values")
    frame[list(attributes)] = frame[list(attributes)].astype(str)
    frame[weight_column] = pd.to_numeric(frame[weight_column], errors="raise")
    weights = frame[weight_column].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError(f"{weight_column} must contain finite positive weights")
    cells = (
        frame.groupby(list(attributes), as_index=False, dropna=False, sort=True)[
            weight_column
        ]
        .sum()
    )
    # ipfn updates the weight column in place with fractional values.
    cells[weight_column] = cells[weight_column].astype(float)
    return cells, len(frame)


def apply_state_folds(
    marginals: pd.DataFrame,
    cells: pd.DataFrame,
    attributes: Sequence[str],
    folds: Mapping[tuple[str, str], str] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Apply explicit unsupported-state folds and reject remaining gaps."""

    result = marginals.copy()
    log: list[dict[str, object]] = []
    for (attribute, source_state), destination_state in (folds or {}).items():
        if attribute not in attributes:
            raise ValueError(f"State fold refers to unknown attribute {attribute!r}")
        source = (attribute, str(source_state))
        destination = (attribute, str(destination_state))
        supported = set(cells[attribute].astype(str))
        if source in result.columns and str(source_state) not in supported:
            if destination not in result.columns or str(destination_state) not in supported:
                raise ValueError(
                    f"Cannot fold {source!r} into unsupported destination "
                    f"{destination!r}"
                )
            amount = float(result[source].sum())
            result[destination] = result[destination] + result[source]
            result = result.drop(columns=[source])
            log.append(
                {"attribute": attribute, "source": str(source_state),
                 "destination": str(destination_state), "count": amount}
            )

    unsupported: dict[str, list[str]] = {}
    for attribute in attributes:
        supported = set(cells[attribute].astype(str))
        states = [
            str(state)
            for name, state in result.columns
            if name == attribute
            and str(state) not in supported
            and float(result[(name, state)].sum()) > 0
        ]
        if states:
            unsupported[attribute] = states
    if unsupported:
        raise ValueError(
            f"Positive census states are absent from the weighted pool: {unsupported}"
        )
    return result, log


def _marginal_errors(
    frame: pd.DataFrame,
    attributes: Sequence[str],
    targets: Sequence[pd.Series],
    value_column: str,
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for attribute, target in zip(attributes, targets):
        achieved = frame.groupby(attribute, observed=True)[value_column].sum()
        errors[attribute] = float(achieved.sub(target, fill_value=0).abs().max())
    return errors


def _fit_zone(
    position: int,
    zone: object,
    target_values: dict[str, dict[str, float]],
    context: dict[str, object] | None = None,
) -> tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Fit and integerise one zone; safe to execute in a worker process."""

    active = _IPF_WORKER_CONTEXT if context is None else context
    if active is None:
        raise RuntimeError("IPF worker context was not initialized")
    started = time.perf_counter()
    attributes = list(active["attributes"])
    weight_column = str(active["weight_column"])
    zone_field = str(active["zone_field"])
    cells = active["cells"]
    targets = [pd.Series(target_values[attribute], dtype=float) for attribute in attributes]
    fitter = ipfn.ipfn(
        cells.copy(),
        targets,
        [[attribute] for attribute in attributes],
        weight_col=weight_column,
        convergence_rate=float(active["convergence_rate"]),
        max_iteration=int(active["max_iterations"]),
        verbose=2,
    )
    fitted, converged, convergence = fitter.iteration()
    zone_total = int(active["zone_totals"][zone])
    fractional_errors = _marginal_errors(
        fitted, attributes, targets, weight_column
    )
    seed = _zone_seed(int(active["seed"]), position)
    fitted["count"] = TRS(
        fitted[weight_column].to_numpy(dtype=float),
        zone_total,
        rng=np.random.default_rng(seed),
    )
    integer_errors = _marginal_errors(fitted, attributes, targets, "count")
    fitted.insert(0, zone_field, zone)
    fractional = fitted[[zone_field, *attributes, weight_column]].copy()
    counts = fitted.loc[
        fitted["count"] > 0, [zone_field, *attributes, "count"]
    ].copy()
    diagnostic = {
        zone_field: zone,
        "households": zone_total,
        "zone_seed": seed,
        "worker_pid": os.getpid(),
        "runtime_seconds": time.perf_counter() - started,
        "fractional_max_abs_error": max(fractional_errors.values()),
        "integer_max_abs_error": max(integer_errors.values()),
        "converged": bool(converged),
        "iterations": int(len(convergence)),
        "final_convergence_rate": (
            float(convergence["conv"].iloc[-1]) if not convergence.empty else None
        ),
        **{
            f"fractional_error_{name}": value
            for name, value in fractional_errors.items()
        },
        **{
            f"integer_error_{name}": value
            for name, value in integer_errors.items()
        },
    }
    return position, counts, fractional, diagnostic


def _fit_zone_worker(
    position: int,
    zone: object,
    target_values: dict[str, dict[str, float]],
) -> tuple[int, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    return _fit_zone(position, zone, target_values)


def fit_weighted_ipf(
    census: Path | str | pd.DataFrame,
    pool: Path | str | pd.DataFrame,
    attributes: Sequence[str],
    zone_field: str,
    weight_column: str,
    seed: int,
    state_folds: Mapping[tuple[str, str], str] | None = None,
    convergence_rate: float = 1e-5,
    max_iterations: int = 500,
    workers: int = 1,
) -> WeightedIPFResult:
    """Fit weighted joint cells per zone and integerise each zone with TRS."""

    attributes = list(dict.fromkeys(attributes))
    if not attributes:
        raise ValueError("At least one IPF attribute is required")
    marginals = read_household_census(census, attributes, zone_field)
    cells, input_rows = prepare_weighted_cells(pool, attributes, weight_column)
    marginals, fold_log = apply_state_folds(
        marginals, cells, attributes, state_folds
    )
    totals = pd.DataFrame(
        {attribute: marginals[attribute].sum(axis=1) for attribute in attributes}
    )
    consistent = totals.nunique(axis=1).eq(1)
    if not consistent.all():
        raise ValueError(
            "Census control totals disagree across attributes for zones: "
            f"{totals.index[~consistent].tolist()[:10]}"
        )
    if not np.equal(totals.to_numpy(), np.floor(totals.to_numpy())).all():
        raise ValueError("Census zone totals must be integers")

    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    context: dict[str, object] = {
        "attributes": list(attributes),
        "weight_column": weight_column,
        "zone_field": zone_field,
        "cells": cells,
        "zone_totals": totals[attributes[0]].astype("int64").to_dict(),
        "convergence_rate": convergence_rate,
        "max_iterations": max_iterations,
        "seed": seed,
    }
    jobs = [
        (
            position,
            zone,
            {
                attribute: {
                    str(state): float(value)
                    for state, value in marginals.loc[zone, attribute].items()
                }
                for attribute in attributes
            },
        )
        for position, zone in enumerate(marginals.index)
    ]
    if workers == 1:
        zone_results = [_fit_zone(*job, context=context) for job in jobs]
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialise_ipf_worker,
            initargs=(context,),
        ) as executor:
            zone_results = list(executor.map(_fit_zone_worker, *zip(*jobs)))
    zone_results.sort(key=lambda item: item[0])
    count_parts = [item[1] for item in zone_results]
    fractional_parts = [item[2] for item in zone_results]
    diagnostic_rows = [item[3] for item in zone_results]

    counts = pd.concat(count_parts, ignore_index=True)
    fractional = pd.concat(fractional_parts, ignore_index=True)
    diagnostics = pd.DataFrame(diagnostic_rows)
    expected = totals[attributes[0]].astype("int64").sort_index()
    actual = counts.groupby(zone_field, observed=True)["count"].sum().astype("int64").sort_index()
    if not actual.equals(expected):
        raise AssertionError("IPF integerised zone totals do not match the census")
    return WeightedIPFResult(
        counts=counts,
        fractional=fractional,
        diagnostics=diagnostics,
        fold_log=fold_log,
        input_rows=input_rows,
        joint_cells=len(cells),
    )

