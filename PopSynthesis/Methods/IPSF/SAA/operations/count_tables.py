"""Utilities for populations represented as unique records with integer counts."""

from __future__ import annotations

from itertools import combinations, product
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field, zone_field


def population_size(population: pl.DataFrame) -> int:
    if population.is_empty():
        return 0
    return int(population[count_field].sum())


def aggregate_counts(
    population: pl.DataFrame, columns: Optional[Sequence[str]] = None
) -> pl.DataFrame:
    """Return one positive-count row per unique key."""
    if count_field not in population.columns:
        raise ValueError(f"Count table must contain {count_field!r}")
    keys = list(columns) if columns is not None else [
        col for col in population.columns if col != count_field
    ]
    if not keys:
        raise ValueError("A count table requires at least one key column")
    result = (
        population.group_by(keys, maintain_order=True)
        .agg(pl.col(count_field).sum())
        .filter(pl.col(count_field) > 0)
        .with_columns(pl.col(count_field).cast(pl.Int64))
    )
    if result.select(
        pl.any_horizontal([pl.col(key).is_null() for key in keys]).any()
    ).item():
        raise ValueError("Count-table keys cannot contain nulls")
    return result


def concat_counts(
    populations: Iterable[pl.DataFrame], columns: Optional[Sequence[str]] = None
) -> pl.DataFrame:
    frames = [frame for frame in populations if frame is not None and not frame.is_empty()]
    if not frames:
        if columns is None:
            raise ValueError("columns are required when concatenating no populations")
        return pl.DataFrame(
            schema={**{col: pl.String for col in columns}, count_field: pl.Int64}
        )
    return aggregate_counts(pl.concat(frames, how="diagonal_relaxed"), columns)


def expand_counts(population: pl.DataFrame) -> pl.DataFrame:
    """Expand a count table at an explicit output compatibility boundary."""
    if count_field not in population.columns:
        raise ValueError(f"Count table must contain {count_field!r}")
    if population.is_empty():
        return population.drop(count_field)
    return population.with_columns(
        pl.exclude(count_field).repeat_by(pl.col(count_field))
    ).select(pl.exclude(count_field).arr.explode())


def weighted_sample_counts(pool: pl.DataFrame, n: int) -> pl.DataFrame:
    """Sample complete pool types with replacement and return sampled counts."""
    if n < 0:
        raise ValueError("Sample size cannot be negative")
    if n == 0:
        return pool.head(0).with_columns(pl.col(count_field).cast(pl.Int64))
    weights = pool[count_field].to_numpy().astype(float)
    if len(weights) == 0 or weights.sum() <= 0:
        raise ValueError("Cannot sample from an empty or zero-weight pool")
    draws = np.random.multinomial(n, weights / weights.sum())
    return pool.with_columns(pl.Series(count_field, draws, dtype=pl.Int64)).filter(
        pl.col(count_field) > 0
    )


def sample_without_replacement_counts(
    population: pl.DataFrame, n: int
) -> pl.DataFrame:
    """Draw a uniform subset of individuals using sequential hypergeometrics."""
    total = population_size(population)
    if not 0 <= n <= total:
        raise ValueError(f"Cannot retain {n} records from a population of {total}")
    if n == 0:
        return population.head(0)
    if n == total:
        return population.clone()

    counts = population[count_field].to_numpy().astype(np.int64)
    retained = np.zeros(len(counts), dtype=np.int64)
    remaining_total = total
    remaining_draws = n
    for idx, available in enumerate(counts[:-1]):
        if remaining_draws == 0:
            break
        draw = np.random.hypergeometric(
            ngood=int(available),
            nbad=int(remaining_total - available),
            nsample=int(remaining_draws),
        )
        retained[idx] = draw
        remaining_draws -= int(draw)
        remaining_total -= int(available)
    retained[-1] = remaining_draws
    return population.with_columns(
        pl.Series(count_field, retained, dtype=pl.Int64)
    ).filter(pl.col(count_field) > 0)


def counts_to_marginals(
    population: pl.DataFrame, attributes: Sequence[str]
) -> pd.DataFrame:
    """Create the current pandas MultiIndex marginal format from weighted rows."""
    if zone_field not in population.columns:
        raise ValueError(f"Count table must contain {zone_field!r}")
    frames: list[pd.DataFrame] = []
    source = population.to_pandas()
    if source.empty:
        return pd.DataFrame(
            index=pd.Index([], name=zone_field),
            columns=pd.MultiIndex(levels=[[], []], codes=[[], []]),
        )
    source[zone_field] = source[zone_field].astype(str)
    for attribute in attributes:
        table = source.pivot_table(
            index=zone_field,
            columns=attribute,
            values=count_field,
            aggfunc="sum",
            fill_value=0,
        ).astype(int)
        table.columns = pd.MultiIndex.from_tuples(
            [(attribute, str(state)) for state in table.columns]
        )
        frames.append(table)
    if not frames:
        return pd.DataFrame(index=pd.Index([], name=zone_field))
    return pd.concat(frames, axis=1)


def prune_oversupplied_counts(
    population: pl.DataFrame, diff_census: pd.DataFrame
) -> pl.DataFrame:
    """Count-table equivalent of adjust_kept_rec_match_census."""
    if population.is_empty():
        return population.clone()
    key_columns = [col for col in population.columns if col != count_field]
    count_kept = population.to_pandas().set_index(key_columns)[count_field]
    count_kept.index = count_kept.index.set_levels(
        [level.astype(str) for level in count_kept.index.levels]
    )
    count_kept = count_kept.sort_values(ascending=False)

    for zone, row in diff_census.iterrows():
        zone = str(zone)
        zone_mask = count_kept.index.get_level_values(zone_field) == zone
        sub_count_kept = count_kept.loc[zone_mask].copy()
        previous_indexes = sub_count_kept.index
        negative = row[row < 0].copy()
        by_attribute: dict[str, list[tuple[str, str]]] = {}
        for attribute, state in negative.index:
            by_attribute.setdefault(attribute, []).append((attribute, state))

        for omitted in range(len(by_attribute)):
            raw_combinations = combinations(
                by_attribute.values(), len(by_attribute) - omitted
            )
            for raw in raw_combinations:
                if negative.sum() == 0:
                    break
                for combination in product(*raw):
                    condition = np.ones(len(sub_count_kept), dtype=bool)
                    to_delete = np.inf
                    for attribute, state in combination:
                        condition &= (
                            sub_count_kept.index.get_level_values(attribute)
                            == str(state)
                        )
                        if attribute != zone_field:
                            to_delete = min(to_delete, -negative.loc[(attribute, state)])
                    matching = sub_count_kept.loc[condition].copy()
                    if matching.empty or to_delete == 0:
                        continue
                    to_delete = min(to_delete, matching.sum())
                    proportional = matching / matching.sum() * to_delete
                    matching -= np.floor(proportional)
                    remainder = int(to_delete - np.floor(proportional).sum())
                    matching.sort_values(ascending=False, inplace=True)
                    matching.iloc[:remainder] -= 1
                    if (matching < 0).any():
                        raise AssertionError("Residual pruning produced negative counts")
                    sub_count_kept.loc[matching.index] = matching
                    negative.loc[list(combination)] += to_delete
                    sub_count_kept = sub_count_kept[sub_count_kept > 0]

        zero_indexes = previous_indexes.difference(sub_count_kept.index)
        count_kept.loc[sub_count_kept.index] = sub_count_kept
        count_kept.loc[zero_indexes] = 0
        if negative.sum() != 0:
            raise AssertionError("Could not remove all oversupplied records")

    result = count_kept[count_kept > 0].rename(count_field).reset_index()
    return aggregate_counts(pl.from_pandas(result), key_columns)
