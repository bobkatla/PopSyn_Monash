"""Publication analysis for Section 4.4: SAA attribute-order sensitivity.

The module reads completed synthesis outputs only.  It never imports or invokes
the synthesis CLI.  All five policies are evaluated against a common weighted
VISTA joint reference so their joint-distribution metrics are comparable.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations, permutations
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import yaml

from PopSynthesis.analyse.household_experiments.metrics import distribution_metrics


ATTRIBUTES = ["hhsize", "dwelltype", "owndwell", "hhinc", "totalvehs"]
PERMUTATION_BASE = ["hhsize", "totalvehs", "dwelltype", "owndwell", "hhinc"]
ZONE_FIELD = "zone_id"
MAIN_ORDER = ("hhsize", "dwelltype", "owndwell", "hhinc", "totalvehs")
HEURISTIC_ORDER = ("dwelltype", "owndwell", "hhinc", "hhsize", "totalvehs")
KNOWN_VISTA = {
    "dwelltype": (0.5068, 0.0599, 0.5667),
    "owndwell": (0.3650, 0.0974, 0.4624),
    "hhinc": (0.2098, 0.1877, 0.3975),
    "hhsize": (0.2935, 0.0455, 0.3390),
    "totalvehs": (0.2846, 0.0496, 0.3342),
}

VARIANTS = {
    "unrestricted": {
        "label": "SAA Unrestricted", "expected": 120,
        "config": "IO/configs/direct_count_unrestricted_attribute_orders.yml",
        "output": "IO/output/direct_count_unrestricted_attribute_orders_120",
        "main_run_id": "saa_original_unrestricted", "score_source": "VISTA",
    },
    "relaxed": {
        "label": "SAA Relaxed", "expected": 120,
        "config": "IO/configs/direct_count_relaxed_attribute_orders.yml",
        "output": "IO/output/direct_count_relaxed_attribute_orders",
        "main_run_id": "saa_original_relax_empty_rows_columns", "score_source": "VISTA",
    },
    "seed_only": {
        "label": "SAA Seed-only", "expected": 96,
        "config": "IO/configs/direct_count_seed_only_attribute_orders.yml",
        "output": "IO/output/direct_count_seed_only_attribute_orders",
        "main_run_id": "saa_original_seed_only", "score_source": "VISTA",
    },
    "seed_free": {
        "label": "SAA Seed-free", "expected": 120,
        "config": "IO/configs/direct_count_seed_free_attribute_orders.yml",
        "output": "IO/output/direct_count_seed_free_attribute_orders",
        "main_run_id": "saa_seed_free_all_zero", "score_source": "VISTA",
    },
    "bn_pool": {
        "label": "SAA BN-pool", "expected": 120,
        "config": "IO/configs/direct_count_bn_pool_seed_only_attribute_orders.yml",
        "output": "IO/output/direct_count_bn_pool_seed_only_attribute_orders",
        "main_run_id": "saa_bn_pool_seed_only", "score_source": "BN pool",
    },
}
VARIANT_ORDER = list(VARIANTS)
COLORS = {
    "unrestricted": "#0072B2", "relaxed": "#E69F00", "seed_only": "#009E73",
    "seed_free": "#CC79A7", "bn_pool": "#D55E00",
}
ATTRIBUTE_LABELS = {
    "hhsize": "Household size", "dwelltype": "Dwelling type",
    "owndwell": "Dwelling ownership", "hhinc": "Household income",
    "totalvehs": "Household vehicles",
}


def repository_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "PopSynthesis").exists():
            return candidate
    raise FileNotFoundError("Could not locate the PopSyn_Monash repository root")


def output_path(root: Path) -> Path:
    return root / "IO/output/full_household_analysis/paper_section_4_4"


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 350, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 7.5,
        "ytick.labelsize": 8, "legend.fontsize": 7.5, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def _order_text(order: Iterable[str]) -> str:
    return " > ".join(order)


def _validation(records: list[dict[str, Any]], check: str, passed: bool, detail: str) -> None:
    records.append({"check": check, "passed": bool(passed), "detail": detail})


def discover_runs(root: Path, validations: list[dict[str, Any]]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    all_perms = set(permutations(PERMUTATION_BASE))
    for variant, spec in VARIANTS.items():
        config_path = root / spec["config"]
        runs = yaml.safe_load(config_path.read_text())
        output_root = root / spec["output"]
        runtime_path = output_root / "run_times.csv"
        if not runtime_path.exists():
            raise FileNotFoundError(runtime_path)
        runtimes = pd.read_csv(runtime_path)
        runtime_map = runtimes.set_index("run_name")["time_seconds"]
        orders: list[tuple[str, ...]] = []
        for index, run in enumerate(runs, 1):
            order = tuple(run["ordered_to_adjust_atts"])
            name = str(run["output_name"])
            orders.append(order)
            population = output_root / name / "reruns_0/household_counts.csv"
            diagnostics = output_root / name / "reruns_0/meta/direct_count_adjustments.csv"
            if name not in runtime_map.index:
                raise ValueError(f"Missing runtime for {name}")
            if not population.exists() or not diagnostics.exists():
                raise FileNotFoundError(f"Incomplete output for {name}")
            if set(order) != set(ATTRIBUTES) or len(order) != len(ATTRIBUTES):
                raise ValueError(f"Invalid order for {name}: {order}")
            if any(tuple(run[key]) != order for key in
                   ("considered_atts", "last_adjustment_order")):
                raise ValueError(f"Order-dependent configuration differs for {name}")
            records.append({
                "variant": variant, "variant_label": spec["label"],
                "permutation_index": index, "run_name": name,
                "order": order, "order_text": _order_text(order),
                **{f"attribute_{i}": value for i, value in enumerate(order, 1)},
                **{f"{attribute}_position": order.index(attribute) + 1 for attribute in ATTRIBUTES},
                "population_path": population, "diagnostics_path": diagnostics,
                "runtime_seconds": float(runtime_map.loc[name]),
            })
        unique = set(orders)
        expected = int(spec["expected"])
        _validation(validations, f"{variant}_permutation_count",
                    len(runs) == expected, f"observed={len(runs)}, expected={expected}")
        _validation(validations, f"{variant}_unique_orders",
                    len(unique) == expected, f"unique={len(unique)}")
        eligible = all_perms if variant != "seed_only" else {p for p in all_perms if p[0] != "hhinc"}
        _validation(validations, f"{variant}_eligible_orders", unique == eligible,
                    f"observed={len(unique)}, eligible={len(eligible)}")
    frame = pd.DataFrame(records)
    _validation(validations, "total_permutation_count", len(frame) == 576,
                f"observed={len(frame)}, expected=576")
    return frame


def load_census(path: Path) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    wide = pd.read_csv(path, header=[0, 1])
    zone_col = [c for c in wide.columns if c[0] == ZONE_FIELD]
    if len(zone_col) != 1:
        raise ValueError("Census must contain exactly one zone_id column")
    wide = wide.set_index(zone_col[0])
    wide.index = wide.index.astype(str)
    records = []
    for attribute in ATTRIBUTES:
        for state in wide[attribute].columns:
            records.append(pd.DataFrame({
                ZONE_FIELD: wide.index, "attribute": attribute, "state": str(state),
                "census_count": pd.to_numeric(wide[(attribute, state)]).to_numpy(float),
            }))
    return wide, pd.concat(records, ignore_index=True), set(wide.index)


def load_pool(path: Path, weight: str) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=[*ATTRIBUTES, weight],
                        dtype={a: str for a in ATTRIBUTES})
    frame[weight] = pd.to_numeric(frame[weight], errors="raise")
    if frame[ATTRIBUTES].isna().any().any() or (frame[weight] <= 0).any():
        raise ValueError(f"Invalid pool data in {path}")
    return frame


def conditional_rigidity(pool: pd.DataFrame, attribute: str, weight: str) -> float:
    supported = pool.loc[pool[weight] > 0]
    contexts = [a for a in ATTRIBUTES if a != attribute]
    state_counts = supported.groupby(contexts, observed=True)[attribute].nunique()
    return float((state_counts == 1).mean())


def marginal_pressure(pool: pd.DataFrame, census_wide: pd.DataFrame,
                      attribute: str, weight: str) -> float:
    pool_counts = pool.groupby(attribute, observed=True)[weight].sum().astype(float)
    census_counts = census_wide[attribute].sum().astype(float)
    states = pool_counts.index.union(census_counts.index.astype(str))
    p = pool_counts.reindex(states, fill_value=0) / pool_counts.sum()
    q = census_counts.set_axis(census_counts.index.astype(str)).reindex(states, fill_value=0)
    q = q / q.sum()
    return float(0.5 * (p - q).abs().sum())


def calculate_heuristic_scores(vista: pd.DataFrame, bn: pd.DataFrame,
                               census_wide: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source, pool, weight in (("VISTA", vista, "_weight"), ("BN pool", bn, "count")):
        for attribute in ATTRIBUTES:
            rigidity = conditional_rigidity(pool, attribute, weight)
            pressure = marginal_pressure(pool, census_wide, attribute, weight)
            rows.append({"score_source": source, "attribute": attribute,
                         "rigidity": rigidity, "marginal_pressure": pressure,
                         "heuristic_score": rigidity + pressure})
    return pd.DataFrame(rows)


def parse_slack_row(row: pd.Series) -> dict[str, float]:
    target = {str(k): float(v) for k, v in json.loads(row.target_counts_json).items()}
    achieved = {str(k): float(v) for k, v in json.loads(row.achieved_counts_json).items()}
    states = set(target) | set(achieved)
    delta = np.array([achieved.get(s, 0) - target.get(s, 0) for s in states], dtype=float)
    return {
        "absolute_slack": float(np.abs(delta).sum()),
        "positive_slack": float(delta[delta > 0].sum()),
        "negative_slack": float(-delta[delta < 0].sum()),
        "max_state_absolute_error": float(np.abs(delta).max(initial=0)),
    }


def census_metrics_from_diagnostics(diagnostics: pd.DataFrame) -> dict[str, float]:
    absolute: list[float] = []
    squared: list[float] = []
    zone_tv: list[float] = []
    zone_weight: list[float] = []
    for row in diagnostics.itertuples(index=False):
        target = {str(k): float(v) for k, v in json.loads(row.target_counts_json).items()}
        achieved = {str(k): float(v) for k, v in json.loads(row.achieved_counts_json).items()}
        states = set(target) | set(achieved)
        delta = np.array([achieved.get(s, 0) - target.get(s, 0) for s in states], dtype=float)
        target_total = sum(target.values())
        achieved_total = sum(achieved.values())
        absolute.extend(np.abs(delta))
        squared.extend(np.square(delta))
        if target_total > 0 and achieved_total > 0:
            p = np.array([target.get(s, 0) for s in states]) / target_total
            q = np.array([achieved.get(s, 0) for s in states]) / achieved_total
            zone_tv.append(float(0.5 * np.abs(p - q).sum()))
            zone_weight.append(float(target_total))
    values = np.asarray(absolute)
    return {
        "census_macro_tvd": float(np.mean(zone_tv)),
        "census_household_weighted_tvd": float(np.average(zone_tv, weights=zone_weight)),
        "census_l1_error": float(values.sum()),
        "census_mae": float(values.mean()),
        "census_rmse": float(np.sqrt(np.mean(squared))),
        "census_max_absolute_error": float(values.max(initial=0)),
    }


def _distribution(frame: pd.DataFrame, attributes: list[str], value: str) -> pd.Series:
    counts = frame.groupby(attributes, observed=True, dropna=False)[value].sum().astype(float)
    return counts / counts.sum()


def joint_metrics(population: pd.DataFrame, vista: pd.DataFrame) -> dict[str, float]:
    reference = _distribution(vista, ATTRIBUTES, "_weight")
    synthetic = _distribution(population, ATTRIBUTES, "count")
    universe = reference.index.union(synthetic.index)
    aligned_r = reference.reindex(universe, fill_value=0)
    aligned_s = synthetic.reindex(universe, fill_value=0)
    full = distribution_metrics(aligned_r, aligned_s)
    pairs = []
    for pair in combinations(ATTRIBUTES, 2):
        r = _distribution(vista, list(pair), "_weight")
        s = _distribution(population, list(pair), "count")
        index = r.index.union(s.index)
        pairs.append(distribution_metrics(r.reindex(index, fill_value=0),
                                          s.reindex(index, fill_value=0))["total_variation"])
    return {
        "five_way_tvd": float(full["total_variation"]),
        "five_way_js_distance": float(full["js_distance"]),
        "mean_pairwise_tvd": float(np.mean(pairs)),
        "max_pairwise_tvd": float(np.max(pairs)),
    }


def load_population(path: Path, expected_zones: set[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={ZONE_FIELD: str, **{a: str for a in ATTRIBUTES}})
    required = {ZONE_FIELD, *ATTRIBUTES, "count"}
    if set(frame.columns) != required:
        raise ValueError(f"Unexpected columns in {path}: {frame.columns.tolist()}")
    frame = frame[[ZONE_FIELD, *ATTRIBUTES, "count"]]
    counts = pd.to_numeric(frame["count"], errors="raise")
    if (counts <= 0).any() or not np.isfinite(counts).all() or not np.equal(counts, np.floor(counts)).all():
        raise ValueError(f"Counts must be positive integers in {path}")
    frame["count"] = counts.astype("int64")
    if set(frame[ZONE_FIELD]) != expected_zones:
        raise ValueError(f"Output zone set differs from Census in {path}")
    if frame.duplicated([ZONE_FIELD, *ATTRIBUTES]).any():
        raise ValueError(f"Duplicate compressed zone/type keys in {path}")
    return frame


def final_census_l1(population: pd.DataFrame, census_long: pd.DataFrame) -> float:
    pieces = []
    for attribute in ATTRIBUTES:
        part = (population.groupby([ZONE_FIELD, attribute], observed=True)["count"].sum()
                .rename("synthetic_count").reset_index().rename(columns={attribute: "state"}))
        part.insert(1, "attribute", attribute)
        pieces.append(part)
    observed = pd.concat(pieces, ignore_index=True)
    aligned = census_long.merge(observed, how="outer",
                                on=[ZONE_FIELD, "attribute", "state"]).fillna(0)
    return float((aligned.synthetic_count - aligned.census_count).abs().sum())


def analyse_permutations(runs: pd.DataFrame, vista: pd.DataFrame,
                         census_long: pd.DataFrame, expected_zones: set[str],
                         validations: list[dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, Any]] = []
    slack_permutation: list[dict[str, Any]] = []
    slack_attribute: list[dict[str, Any]] = []
    for number, run in enumerate(runs.itertuples(index=False), 1):
        diagnostics = pd.read_csv(run.diagnostics_path, dtype={"zone": str})
        if len(diagnostics) != 691 * 5 or diagnostics.zone.nunique() != 691:
            raise ValueError(f"Expected 3,455 diagnostic records for {run.run_name}")
        expected_positions = set(range(5))
        if set(diagnostics.adjustment_index) != expected_positions:
            raise ValueError(f"Unexpected adjustment positions for {run.run_name}")
        parsed = pd.DataFrame([parse_slack_row(row) for _, row in diagnostics.iterrows()])
        stage = pd.concat([diagnostics[["zone", "adjustment_index", "active_attribute",
                                        "stage1_objective", "marginal_l1_error"]].reset_index(drop=True),
                           parsed], axis=1)
        stage["position"] = stage.adjustment_index + 1
        objective = pd.to_numeric(stage.stage1_objective, errors="coerce").fillna(0)
        objective_sum = float(objective.sum())
        derived_sum = float(stage.absolute_slack.sum())
        marginal_sum = float(stage.marginal_l1_error.sum())
        first = stage[stage.position == 1]
        first_ok = bool((first.absolute_slack == 0).all())
        objective_ok = math.isclose(objective_sum, derived_sum, abs_tol=1e-8)
        marginal_ok = math.isclose(marginal_sum, derived_sum, abs_tol=1e-8)

        population = load_population(run.population_path, expected_zones)
        final_l1 = final_census_l1(population, census_long)
        final_ok = math.isclose(final_l1, derived_sum, abs_tol=1e-8)
        census = census_metrics_from_diagnostics(diagnostics)
        joint = joint_metrics(population, vista)
        metrics_rows.append({
            **{c: getattr(run, c) for c in runs.columns
               if c not in {"population_path", "diagnostics_path", "order"}},
            **census, **joint, "total_stage1_slack": objective_sum,
            "final_census_l1": final_l1,
            "synthetic_joint_types": int(population.groupby(ATTRIBUTES, observed=True).ngroups),
            "compressed_zone_type_rows": int(len(population)),
        })
        zonal = stage.groupby("zone", observed=True).absolute_slack.sum()
        slack_permutation.append({
            "variant": run.variant, "permutation_index": run.permutation_index,
            "run_name": run.run_name, "order_text": run.order_text,
            "total_slack": derived_sum, "positive_slack": float(stage.positive_slack.sum()),
            "negative_slack": float(stage.negative_slack.sum()),
            "nonzero_zone_percent": float((zonal > 0).mean() * 100),
            "mean_zonal_slack": float(zonal.mean()), "median_zonal_slack": float(zonal.median()),
            "p95_zonal_slack": float(zonal.quantile(.95)), "max_zonal_slack": float(zonal.max()),
        })
        for (attribute, position), part in stage.groupby(["active_attribute", "position"], observed=True):
            zonal_attribute = part.groupby("zone", observed=True).absolute_slack.sum()
            slack_attribute.append({
                "variant": run.variant, "permutation_index": run.permutation_index,
                "run_name": run.run_name, "order_text": run.order_text,
                "attribute": attribute, "position": int(position),
                "total_slack": float(part.absolute_slack.sum()),
                "positive_slack": float(part.positive_slack.sum()),
                "negative_slack": float(part.negative_slack.sum()),
                "nonzero_zone_percent": float((zonal_attribute > 0).mean() * 100),
                "mean_zonal_slack": float(zonal_attribute.mean()),
                "median_zonal_slack": float(zonal_attribute.median()),
                "p95_zonal_slack": float(zonal_attribute.quantile(.95)),
                "max_zonal_slack": float(zonal_attribute.max()),
                "max_state_absolute_error": float(part.max_state_absolute_error.max()),
            })
        _validation(validations, f"{run.run_name}_first_attribute_exact", first_ok,
                    f"first-stage absolute slack={first.absolute_slack.sum():.0f}")
        _validation(validations, f"{run.run_name}_stage1_matches_states", objective_ok and marginal_ok,
                    f"objective={objective_sum:.0f}, states={derived_sum:.0f}, marginal={marginal_sum:.0f}")
        _validation(validations, f"{run.run_name}_stage1_matches_final", final_ok,
                    f"stage1={derived_sum:.0f}, final Census L1={final_l1:.0f}")
        if number % 24 == 0 or number == len(runs):
            print(f"Analysed {number}/{len(runs)} permutations", flush=True)
    return pd.DataFrame(metrics_rows), pd.DataFrame(slack_permutation), pd.DataFrame(slack_attribute)


def rank_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    result = frame.copy()
    result[f"{metric}_rank"] = result.groupby("variant")[metric].rank(method="min", ascending=True).astype(int)
    sizes = result.groupby("variant")[metric].transform("size")
    ranks = result[f"{metric}_rank"]
    result[f"{metric}_percentile"] = np.where(sizes > 1, 100 * (sizes - ranks) / (sizes - 1), 100.0)
    return result


def build_rankings(metrics: pd.DataFrame) -> pd.DataFrame:
    result = metrics.copy()
    for metric in ("census_macro_tvd", "five_way_tvd", "five_way_js_distance",
                   "mean_pairwise_tvd", "total_stage1_slack"):
        ranked = rank_metric(result[["variant", metric]], metric)
        result[f"{metric}_rank"] = ranked[f"{metric}_rank"].to_numpy()
        result[f"{metric}_percentile"] = ranked[f"{metric}_percentile"].to_numpy()
    return result


def sensitivity_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (variant, label), group in metrics.groupby(["variant", "variant_label"], sort=False):
        for metric in ("census_macro_tvd", "total_stage1_slack", "five_way_tvd",
                       "five_way_js_distance", "mean_pairwise_tvd"):
            values = group[metric]
            rows.append({
                "variant": variant, "variant_label": label, "metric": metric,
                "n": len(values), "minimum": values.min(), "maximum": values.max(),
                "mean": values.mean(), "median": values.median(), "standard_deviation": values.std(ddof=1),
                "iqr": values.quantile(.75) - values.quantile(.25),
                "p05": values.quantile(.05), "p95": values.quantile(.95),
                "exact_count": int(np.isclose(values, 0, atol=1e-12).sum()),
                "exact_percent": float(np.isclose(values, 0, atol=1e-12).mean() * 100),
            })
    return pd.DataFrame(rows)


def _select_order(group: pd.DataFrame, role: str) -> pd.Series:
    if role == "household_size_first":
        selected = group[group.order_text == _order_text(MAIN_ORDER)]
    elif role == "unconstrained_heuristic":
        selected = group[group.order_text == _order_text(HEURISTIC_ORDER)]
    elif role in {"minimum_macro_tvd", "minimum_total_slack", "minimum_five_way_js"}:
        metric = {"minimum_macro_tvd": "census_macro_tvd",
                  "minimum_total_slack": "total_stage1_slack",
                  "minimum_five_way_js": "five_way_js_distance"}[role]
        candidates = group[np.isclose(group[metric], group[metric].min(), atol=1e-15, rtol=0)]
        # If every eligible order ties, use the production order as a clearly
        # labelled representative rather than implying a unique empirical optimum.
        production = candidates[candidates.order_text == _order_text(MAIN_ORDER)]
        if len(candidates) == len(group) and not production.empty:
            selected = production
        else:
            tie_breakers = [m for m in ("census_macro_tvd", "total_stage1_slack",
                                         "five_way_js_distance", "permutation_index") if m != metric]
            selected = candidates.sort_values(tie_breakers).head(1)
    elif role == "median_census":
        selected = group.assign(distance=(group.census_macro_tvd - group.census_macro_tvd.median()).abs()).sort_values(
            ["distance", "five_way_js_distance", "permutation_index"]).head(1)
    else:
        raise ValueError(role)
    if selected.empty:
        raise ValueError(f"Order role {role} is unavailable for {group.variant.iloc[0]}")
    return selected.iloc[0]


def selected_order_comparison(rankings: pd.DataFrame) -> pd.DataFrame:
    rows = []
    roles = ["household_size_first", "unconstrained_heuristic", "minimum_macro_tvd",
             "minimum_total_slack", "minimum_five_way_js", "median_census"]
    for variant, group in rankings.groupby("variant", sort=False):
        selected = {role: _select_order(group, role) for role in roles}
        order_roles: dict[str, list[str]] = {}
        for role, row in selected.items():
            order_roles.setdefault(row.order_text, []).append(role)
        for role, row in selected.items():
            criterion = {"minimum_macro_tvd": "census_macro_tvd",
                         "minimum_total_slack": "total_stage1_slack",
                         "minimum_five_way_js": "five_way_js_distance"}.get(role)
            tie_count = (int(np.isclose(group[criterion], group[criterion].min(), atol=1e-15, rtol=0).sum())
                         if criterion else 1)
            rows.append({
                "variant": variant, "variant_label": row.variant_label, "strategy": role,
                "overlapping_strategies": "; ".join(order_roles[row.order_text]),
                "optimum_tie_count": tie_count,
                "tie_note": (f"all {tie_count} eligible orders tie" if tie_count == len(group)
                             else (f"{tie_count}-way tie" if tie_count > 1 else "unique")),
                "rank_denominator": len(group),
                "permutation_index": row.permutation_index, "order_text": row.order_text,
                "census_macro_tvd": row.census_macro_tvd,
                "census_rank": row.census_macro_tvd_rank,
                "census_percentile": row.census_macro_tvd_percentile,
                "census_difference_from_best": row.census_macro_tvd - group.census_macro_tvd.min(),
                "census_difference_from_median": row.census_macro_tvd - group.census_macro_tvd.median(),
                "five_way_tvd": row.five_way_tvd, "five_way_js_distance": row.five_way_js_distance,
                "joint_rank": row.five_way_js_distance_rank,
                "joint_percentile": row.five_way_js_distance_percentile,
                "joint_difference_from_best": row.five_way_js_distance - group.five_way_js_distance.min(),
                "joint_difference_from_median": row.five_way_js_distance - group.five_way_js_distance.median(),
                "total_stage1_slack": row.total_stage1_slack,
                "slack_rank": row.total_stage1_slack_rank,
                "slack_difference_from_best": row.total_stage1_slack - group.total_stage1_slack.min(),
            })
    return pd.DataFrame(rows)


STRATEGY_LABELS = {
    "household_size_first": "Household-size-first",
    "unconstrained_heuristic": "Unconstrained heuristic",
    "minimum_macro_tvd": "Minimum macro TVD",
    "minimum_total_slack": "Minimum total slack",
    "minimum_five_way_js": "Minimum five-way JS",
}


def ordering_comparison_tables(selected: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    detailed = selected[selected.strategy.isin(STRATEGY_LABELS)].copy()
    detailed["strategy_label"] = detailed.strategy.map(STRATEGY_LABELS)
    detailed["macro_tvd_rank_display"] = (detailed.census_rank.astype(int).astype(str) + "/" +
                                           detailed.rank_denominator.astype(int).astype(str))
    detailed["joint_rank_display"] = (detailed.joint_rank.astype(int).astype(str) + "/" +
                                       detailed.rank_denominator.astype(int).astype(str))
    detailed = detailed[[
        "variant", "variant_label", "strategy", "strategy_label", "order_text",
        "overlapping_strategies", "optimum_tie_count", "tie_note", "rank_denominator",
        "census_macro_tvd", "total_stage1_slack", "five_way_js_distance",
        "census_rank", "macro_tvd_rank_display", "joint_rank", "joint_rank_display",
        "slack_rank", "census_difference_from_best", "joint_difference_from_best",
        "slack_difference_from_best",
    ]]
    rows = []
    for (variant, order), group in detailed.groupby(["variant", "order_text"], sort=False):
        representative = group.iloc[0]
        labels = [STRATEGY_LABELS[s] for s in STRATEGY_LABELS if s in set(group.strategy)]
        tie_notes = sorted({note for note in group.tie_note if note != "unique"})
        rows.append({
            "variant": variant, "variant_label": representative.variant_label,
            "ordering_strategy": "; ".join(labels),
            "overlap_or_tie_note": "; ".join(tie_notes),
            "macro_census_tvd": representative.census_macro_tvd,
            "total_stage1_slack": representative.total_stage1_slack,
            "five_way_js_distance": representative.five_way_js_distance,
            "macro_tvd_rank": representative.macro_tvd_rank_display,
            "joint_rank": representative.joint_rank_display,
        })
    main = pd.DataFrame(rows)
    main["variant"] = pd.Categorical(main.variant, VARIANT_ORDER, ordered=True)
    main = main.sort_values(["variant", "ordering_strategy"]).reset_index(drop=True)
    main["variant"] = main.variant.astype(str)
    return main, detailed


def aggregate_slack(slack_attribute: pd.DataFrame) -> pd.DataFrame:
    return (slack_attribute.groupby(["variant", "attribute", "position"], as_index=False)
            .agg(mean_total_slack=("total_slack", "mean"), median_total_slack=("total_slack", "median"),
                 p95_total_slack=("total_slack", lambda x: x.quantile(.95)),
                 max_total_slack=("total_slack", "max"), contributing_permutations=("run_name", "nunique"),
                 mean_nonzero_zone_percent=("nonzero_zone_percent", "mean"),
                 mean_positive_slack=("positive_slack", "mean"), mean_negative_slack=("negative_slack", "mean")))


def add_aggregate_validations(validations: pd.DataFrame, slack_permutation: pd.DataFrame,
                              slack_attribute: pd.DataFrame,
                              slack_position: pd.DataFrame) -> pd.DataFrame:
    names = {"slack_attribute_aggregation", "first_position_aggregate_zero",
             "seed_only_hhinc_first_not_evaluated"}
    validations = validations[~validations.check.isin(names)].copy()
    by_run = slack_attribute.groupby("run_name").total_slack.sum().sort_index()
    expected = slack_permutation.set_index("run_name").total_slack.sort_index()
    aggregation_ok = by_run.index.equals(expected.index) and np.allclose(by_run, expected)
    first = slack_position[slack_position.position == 1]
    first_ok = np.allclose(first.mean_total_slack, 0)
    excluded = slack_position[(slack_position.variant == "seed_only") &
                              (slack_position.attribute == "hhinc") &
                              (slack_position.position == 1)]
    additions = pd.DataFrame([
        {"check": "slack_attribute_aggregation", "passed": aggregation_ok,
         "detail": "per-attribute slack sums reproduce each permutation total"},
        {"check": "first_position_aggregate_zero", "passed": first_ok,
         "detail": "all evaluated first-position mean slack values are zero"},
        {"check": "seed_only_hhinc_first_not_evaluated", "passed": excluded.empty,
         "detail": "VISTA Seed-only hhinc position 1 is absent and plotted as N/A"},
    ])
    return pd.concat([validations, additions], ignore_index=True)


def rigidity_relationship(metrics: pd.DataFrame, slack_attribute: pd.DataFrame,
                          scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant in VARIANT_ORDER:
        source = VARIANTS[variant]["score_source"]
        variant_scores = scores[scores.score_source == source]
        variant_metrics = metrics[metrics.variant == variant]
        variant_slack = slack_attribute[slack_attribute.variant == variant]
        for score in variant_scores.itertuples(index=False):
            own = variant_slack[variant_slack.attribute == score.attribute]
            late = own[own.position.isin([4, 5])].total_slack
            positions = variant_metrics[f"{score.attribute}_position"]
            early_total = variant_metrics.loc[positions.isin([1, 2]), "total_stage1_slack"]
            late_total = variant_metrics.loc[positions.isin([4, 5]), "total_stage1_slack"]
            rows.append({
                "variant": variant, "variant_label": VARIANTS[variant]["label"],
                "score_source": source, "attribute": score.attribute,
                "rigidity": score.rigidity, "marginal_pressure": score.marginal_pressure,
                "heuristic_score": score.heuristic_score,
                "mean_late_attribute_slack": late.mean(), "late_attribute_observations": len(late),
                "mean_total_slack_early": early_total.mean(), "early_permutations": len(early_total),
                "mean_total_slack_late": late_total.mean(), "late_permutations": len(late_total),
                "early_placement_benefit": late_total.mean() - early_total.mean(),
            })
    result = pd.DataFrame(rows)
    def spearman(x: pd.Series, y: pd.Series) -> float:
        valid = pd.concat([x, y], axis=1).dropna()
        if len(valid) < 2 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
            return math.nan
        return float(valid.iloc[:, 0].corr(valid.iloc[:, 1], method="spearman"))
    for variant, index in result.groupby("variant").groups.items():
        part = result.loc[index]
        result.loc[index, "spearman_rigidity_late_slack"] = spearman(
            part.rigidity, part.mean_late_attribute_slack)
        result.loc[index, "spearman_pressure_late_slack"] = spearman(
            part.marginal_pressure, part.mean_late_attribute_slack)
        result.loc[index, "spearman_score_early_benefit"] = spearman(
            part.heuristic_score, part.early_placement_benefit)
    return result


def heuristic_strategy_comparison(rankings: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in rankings.groupby("variant", sort=False):
        source = VARIANTS[variant]["score_source"]
        score = scores[scores.score_source == source].set_index("attribute")
        orders = {
            "descending_H": tuple(score.heuristic_score.sort_values(ascending=False, kind="mergesort").index),
            "descending_R": tuple(score.rigidity.sort_values(ascending=False, kind="mergesort").index),
            "descending_D": tuple(score.marginal_pressure.sort_values(ascending=False, kind="mergesort").index),
        }
        orders["household_first_then_H"] = ("hhsize", *[a for a in orders["descending_H"] if a != "hhsize"])
        for strategy, role in (("best_macro_tvd", "minimum_macro_tvd"),
                               ("best_total_slack", "minimum_total_slack"),
                               ("best_joint", "minimum_five_way_js")):
            selected = _select_order(group, role)
            orders[strategy] = tuple(selected[f"attribute_{i}"] for i in range(1, 6))
        for strategy, order in orders.items():
            match = group[group.order_text == _order_text(order)]
            if match.empty:
                rows.append({"variant": variant, "variant_label": VARIANTS[variant]["label"],
                             "score_source": source, "strategy": strategy,
                             "order_text": _order_text(order), "eligible": False})
                continue
            row = match.iloc[0]
            rows.append({
                "variant": variant, "variant_label": row.variant_label, "score_source": source,
                "strategy": strategy, "order_text": row.order_text, "eligible": True,
                "permutation_index": row.permutation_index, "total_stage1_slack": row.total_stage1_slack,
                "slack_rank": row.total_stage1_slack_rank,
                "slack_difference_from_best": row.total_stage1_slack - group.total_stage1_slack.min(),
                "census_macro_tvd": row.census_macro_tvd, "census_rank": row.census_macro_tvd_rank,
                "census_difference_from_best": row.census_macro_tvd - group.census_macro_tvd.min(),
                "five_way_tvd": row.five_way_tvd, "five_way_js_distance": row.five_way_js_distance,
                "joint_rank": row.five_way_js_distance_rank,
                "joint_difference_from_best": row.five_way_js_distance - group.five_way_js_distance.min(),
            })
    return pd.DataFrame(rows)


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(output / f"{stem}.png", dpi=350, bbox_inches="tight")
    plt.close(fig)


def figure_a(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(16.2, 3.65), sharey=False)
    role_style = {
        "P": ("Household-size-first", "s", "black"),
        "H": ("Unconstrained heuristic", "^", "#7B3294"),
        "T": ("Minimum macro TVD", "P", "#56B4E9"),
        "J": ("Minimum five-way JS", "X", "#F0E442"),
        "S": ("Minimum total slack", "D", "#D55E00"),
    }
    for ax, variant in zip(axes, VARIANT_ORDER):
        data = metrics[metrics.variant == variant]
        exact = np.isclose(data.census_macro_tvd, 0, atol=1e-15).all()
        if exact:
            ax.boxplot(data.five_way_js_distance, positions=[0], widths=.32,
                       showfliers=False, whis=1.5, patch_artist=True,
                       boxprops={"facecolor": "#E6E6E6", "edgecolor": "#555555"},
                       medianprops={"color": "black"})
            ax.scatter(np.zeros(len(data)), data.five_way_js_distance, s=12,
                       alpha=.28, color=COLORS[variant], linewidths=0)
            ax.set_xlim(-.55, .55); ax.set_xticks([0], ["Exact alignment\n(all orders)"])
            ax.set_xlabel("Census marginal result")
        else:
            ax.scatter(data.census_macro_tvd, data.five_way_js_distance, s=15,
                       alpha=.48, color=COLORS[variant], linewidths=0)
            ax.set_xlabel("Macro Census TVD\n(lower is better)")
        chosen = {
            "P": data[data.order_text == _order_text(MAIN_ORDER)].iloc[0],
            "H": data[data.order_text == _order_text(HEURISTIC_ORDER)].iloc[0],
            "J": _select_order(data, "minimum_five_way_js"),
        }
        if not exact:
            chosen["T"] = _select_order(data, "minimum_macro_tvd")
            chosen["S"] = _select_order(data, "minimum_total_slack")
        by_order: dict[str, list[str]] = {}
        for code, row in chosen.items():
            by_order.setdefault(row.order_text, []).append(code)
            _, marker, color = role_style[code]
            x = 0 if exact else row.census_macro_tvd
            ax.scatter(x, row.five_way_js_distance, marker=marker, s=62,
                       facecolor=color, edgecolor="white", linewidth=.7, zorder=4)
        for order, codes in by_order.items():
            row = data[data.order_text == order].iloc[0]
            x = 0 if exact else row.census_macro_tvd
            ax.annotate("/".join(codes), (x, row.five_way_js_distance), xytext=(4, 5),
                        textcoords="offset points", fontsize=6.5, fontweight="bold")
        if exact:
            ax.text(.5, .98, f"All {len(data)} orders have TVD = 0\nand total slack = 0",
                    transform=ax.transAxes, ha="center", va="top", fontsize=6.8)
        ax.set_title(VARIANTS[variant]["label"] + f"\n(n={len(data)})")
        ax.set_ylabel("Five-way JS distance\nto weighted VISTA")
        ax.margins(y=.08)
    handles = [Line2D([0], [0], marker=marker, color="none", markerfacecolor=color,
                      markeredgecolor="black", markersize=7, label=label)
               for label, marker, color in role_style.values()]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False,
               bbox_to_anchor=(.5, 1.06))
    fig.text(.5, .005, "P/H/T/J/S labels identify coincident highlighted strategies; lower JS is better.",
             ha="center", fontsize=7.2)
    fig.tight_layout(rect=(0, .045, 1, .91))
    _save_figure(fig, output, "figure_A_order_sensitivity")


def figure_b(metrics: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.45))
    labels = [f"{VARIANTS[v]['label']}\n(n={len(metrics[metrics.variant == v])})" for v in VARIANT_ORDER]
    for ax, metric, title, ylabel in (
        (axes[0], "census_macro_tvd", "Census alignment", "Macro Census TVD"),
        (axes[1], "five_way_js_distance", "Joint fidelity", "Five-way JS distance"),
    ):
        groups = [metrics.loc[metrics.variant == v, metric].to_numpy() for v in VARIANT_ORDER]
        ax.boxplot(groups, widths=.55, patch_artist=True, showfliers=False,
                   boxprops={"facecolor": "#D9EAF2", "edgecolor": "#555555"},
                   medianprops={"color": "black"})
        for i, variant in enumerate(VARIANT_ORDER, 1):
            values = groups[i - 1]
            offsets = np.linspace(-.18, .18, len(values)) if not np.allclose(values, values[0]) else np.zeros(len(values))
            ax.scatter(i + offsets, values, s=7, alpha=.25, color=COLORS[variant], linewidths=0)
            main = metrics[(metrics.variant == variant) & (metrics.order_text == _order_text(MAIN_ORDER))].iloc[0]
            ax.scatter(i, main[metric], marker="s", s=48, color="black", edgecolor="white", zorder=5)
        ax.set_xticks(range(1, 6), labels, rotation=25, ha="right")
        ax.set_ylabel(ylabel + " (lower is better)")
        ax.set_title(title + " across attribute permutations")
        ax.legend([Line2D([0], [0], marker="s", color="none", markerfacecolor="black",
                          markersize=7)], ["Household-size-first"], frameon=False, loc="best")
    fig.text(.5, .01,
             "Boxes show the interquartile range and median; whiskers extend to 1.5 IQR. "
             "Points are distinct attribute permutations, not stochastic repetitions.",
             ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .06, 1, 1))
    _save_figure(fig, output, "figure_B_order_distributions")


def figure_c(slack_position: pd.DataFrame, output: Path) -> None:
    vmax = float(slack_position.mean_total_slack.max())
    fig, axes = plt.subplots(1, 5, figsize=(15.8, 3.75))
    image = None
    for ax, variant in zip(axes, VARIANT_ORDER):
        data = slack_position[slack_position.variant == variant]
        matrix = data.pivot(index="attribute", columns="position", values="mean_total_slack").reindex(ATTRIBUTES)
        counts = data.pivot(index="attribute", columns="position", values="contributing_permutations").reindex(ATTRIBUTES)
        image = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
        for r in range(5):
            for c in range(5):
                value = matrix.iloc[r, c]
                if pd.isna(value):
                    ax.text(c, r, "N/A", ha="center", va="center", fontsize=7,
                            color="black", fontweight="bold")
                else:
                    color = "white" if value > .55 * vmax else "black"
                    ax.text(c, r, f"{value:,.0f}", ha="center", va="center",
                            fontsize=7, color=color)
        ax.set_xticks(range(5), range(1, 6)); ax.set_xlabel("Adjustment position")
        ax.set_yticks(range(5), ATTRIBUTES if ax is axes[0] else [])
        ax.set_title(VARIANTS[variant]["label"])
    fig.tight_layout(rect=(0, .10, .95, .94))
    color_axis = fig.add_axes([.96, .20, .012, .60])
    fig.colorbar(image, cax=color_axis, label="Mean total absolute Stage 1 slack (counts)")
    fig.text(.5, .02,
             "Position 1 has zero slack by construction. N/A marks the unevaluated hhinc-first "
             "VISTA Seed-only case; contributing counts are reported in slack_by_position.csv.",
             ha="center", fontsize=7.5)
    _save_figure(fig, output, "figure_C_slack_by_attribute_position")


def figure_d(relationship: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.35))
    panels = [
        ("rigidity", "mean_late_attribute_slack", "Conditional rigidity, $R_a$", "Mean late-position slack (counts)", "$R_a$ and late slack"),
        ("marginal_pressure", "mean_late_attribute_slack", "Marginal pressure, $D_a$ (TVD)", "Mean late-position slack (counts)", "$D_a$ and late slack"),
        ("heuristic_score", "early_placement_benefit", "Combined score, $H_a=R_a+D_a$", "Early-placement benefit (counts)", "$H_a$ and early-placement benefit"),
    ]
    informative = ["relaxed", "seed_only", "bn_pool"]
    attribute_markers = {"hhsize": "o", "dwelltype": "s", "owndwell": "^",
                         "hhinc": "D", "totalvehs": "P"}
    for ax, (x, y, xlabel, ylabel, title) in zip(axes, panels):
        for variant in informative:
            data = relationship[relationship.variant == variant]
            for row in data.itertuples(index=False):
                ax.scatter(getattr(row, x), getattr(row, y), s=38,
                           marker=attribute_markers[row.attribute], color=COLORS[variant],
                           alpha=.82, edgecolor="white", linewidth=.45)
        ax.axhline(0, color="#888888", linewidth=.6, linestyle="--") if y == "early_placement_benefit" else None
        ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    policy_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=COLORS[v],
                             markersize=7, label=VARIANTS[v]["label"]) for v in informative]
    attribute_handles = [Line2D([0], [0], marker=attribute_markers[a], color="none",
                                markerfacecolor="#777777", markersize=6,
                                label=a) for a in ATTRIBUTES]
    policy_legend = fig.legend(policy_handles, [h.get_label() for h in policy_handles],
                               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.05))
    fig.add_artist(policy_legend)
    fig.legend(attribute_handles, [h.get_label() for h in attribute_handles],
               loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(.5, .975))
    fig.text(.5, .01,
             "Each point is one attribute within one SAA policy. Late slack is mean attribute-stage slack at positions 4–5; "
             "early-placement benefit is mean total slack at positions 4–5 minus positions 1–2. "
             "Exact-slack policies are omitted because their correlations are undefined.",
             ha="center", fontsize=7.2)
    fig.tight_layout(rect=(0, .10, 1, .84))
    _save_figure(fig, output, "figure_D_rigidity_pressure_vs_slack")


def validate_heuristics(scores: pd.DataFrame, validations: list[dict[str, Any]]) -> None:
    vista = scores[scores.score_source == "VISTA"].set_index("attribute")
    for attribute, (r, d, h) in KNOWN_VISTA.items():
        observed = vista.loc[attribute]
        passed = (abs(observed.rigidity - r) <= 5e-5 and
                  abs(observed.marginal_pressure - d) <= 5e-5 and
                  abs(observed.heuristic_score - h) <= 1e-4)
        _validation(validations, f"vista_heuristic_{attribute}", passed,
                    f"R={observed.rigidity:.6f}, D={observed.marginal_pressure:.6f}, H={observed.heuristic_score:.6f}")


def validate_main_results(root: Path, metrics: pd.DataFrame,
                          validations: list[dict[str, Any]]) -> None:
    analysis = root / "IO/output/full_household_analysis"
    method = pd.read_csv(analysis / "method_summary.csv").set_index("run_id")
    for variant, spec in VARIANTS.items():
        row = metrics[(metrics.variant == variant) &
                      (metrics.order_text == _order_text(MAIN_ORDER))].iloc[0]
        previous = method.loc[spec["main_run_id"]]
        differences = {
            "macro_tvd": abs(row.census_macro_tvd - previous.census_macro_tv),
            "five_way_tvd": abs(row.five_way_tvd - previous.joint_seed_tv),
            "five_way_js": abs(row.five_way_js_distance - previous.joint_seed_js),
        }
        passed = all(value <= 1e-10 for value in differences.values())
        _validation(validations, f"{variant}_main_order_matches_nine_method_analysis", passed,
                    ", ".join(f"{key}_difference={value:.3g}" for key, value in differences.items()))


def write_latex_tables(output: Path, sensitivity: pd.DataFrame,
                       selected: pd.DataFrame, strategies: pd.DataFrame) -> None:
    census = sensitivity[sensitivity.metric == "census_macro_tvd"].copy()
    census = census[["variant_label", "n", "minimum", "median", "maximum",
                     "standard_deviation", "exact_count", "exact_percent"]]
    census.to_latex(output / "ordering_sensitivity_table.tex", index=False,
                    float_format=lambda x: f"{x:.5f}", escape=True)
    comparison = selected[selected.strategy.isin(STRATEGY_LABELS)][
        ["variant_label", "strategy", "order_text", "census_macro_tvd", "total_stage1_slack",
         "census_rank", "five_way_js_distance", "joint_rank", "tie_note"]]
    comparison.to_latex(output / "ordering_comparison_table.tex", index=False,
                        float_format=lambda x: f"{x:.5f}", escape=True)
    main, detailed = ordering_comparison_tables(selected)
    main.to_csv(output / "ordering_comparison_main.csv", index=False)
    detailed.to_csv(output / "ordering_comparison_detailed.csv", index=False)
    latex = main.rename(columns={
        "variant_label": "SAA variant", "ordering_strategy": "Ordering strategy",
        "macro_census_tvd": "Macro TVD", "total_stage1_slack": "Stage 1 slack",
        "five_way_js_distance": "Five-way JS", "macro_tvd_rank": "TVD rank",
        "joint_rank": "JS rank",
    })[["SAA variant", "Ordering strategy", "Macro TVD", "Stage 1 slack",
        "Five-way JS", "TVD rank", "JS rank"]]
    latex.to_latex(output / "ordering_comparison_main.tex", index=False,
                   float_format=lambda x: f"{x:.5f}", escape=True,
                   caption=("Attribute-order strategy comparison. Lower values are better; "
                            "ranks are among 120 eligible orders, or 96 for VISTA Seed-only."),
                   label="tab:attribute_order_comparison")
    strategy_table = strategies[strategies.eligible.fillna(False)][
        ["variant_label", "strategy", "order_text", "total_stage1_slack",
         "census_macro_tvd", "census_rank", "five_way_js_distance", "joint_rank"]]
    strategy_table.to_latex(output / "heuristic_slack_table.tex", index=False,
                            float_format=lambda x: f"{x:.5f}", escape=True)


def write_findings(output: Path, metrics: pd.DataFrame, sensitivity: pd.DataFrame,
                   selected: pd.DataFrame, slack_position: pd.DataFrame,
                   relationship: pd.DataFrame, strategies: pd.DataFrame,
                   validations: pd.DataFrame) -> None:
    census = sensitivity[sensitivity.metric == "census_macro_tvd"].set_index("variant")
    slack = sensitivity[sensitivity.metric == "total_stage1_slack"].set_index("variant")
    joint = sensitivity[sensitivity.metric == "five_way_js_distance"].set_index("variant")
    main = selected[selected.strategy == "household_size_first"].set_index("variant")
    heuristic = selected[selected.strategy == "unconstrained_heuristic"].set_index("variant")
    most_sensitive = census.standard_deviation.idxmax()
    lines = ["# Section 4.4 — Attribute-Order Sensitivity and Slack Analysis", "",
             "All reported variation is variation across attribute-order configurations, not stochastic run-to-run variation.", ""]
    lines += ["## A. Coverage and validation", ""]
    for variant in VARIANT_ORDER:
        lines.append(f"- {VARIANTS[variant]['label']}: {int(census.loc[variant, 'n'])} eligible permutations.")
    lines += ["- Total coverage is 576 configurations. VISTA Seed-only contains 96 rather than 120 orders because its 24 `hhinc`-first permutations are excluded: Negative income has no VISTA-pool support.",
              f"- {int(validations.passed.sum())}/{len(validations)} validation checks pass. Stage 1 slack is nonnegative, every first attribute has zero slack, and summed Stage 1 slack equals final Census L1 for every permutation.",
              "- Slack values are summed absolute household counts over zones, controlled attributes, and states. Macro Census TVD instead equally averages normalised zone-attribute discrepancies over 691 POAs and five attributes; the two measures are not interchangeable.", ""]
    lines += ["## B. Ordering sensitivity", ""]
    for variant in VARIANT_ORDER:
        c, s, j = census.loc[variant], slack.loc[variant], joint.loc[variant]
        lines.append(
            f"- {VARIANTS[variant]['label']}: macro TVD min/median/max/SD = {c['minimum']:.6f}/{c['median']:.6f}/{c['maximum']:.6f}/{c['standard_deviation']:.6f}; "
            f"total slack = {s['minimum']:,.0f}/{s['median']:,.0f}/{s['maximum']:,.0f}/{s['standard_deviation']:,.1f}; "
            f"five-way JS = {j['minimum']:.6f}/{j['median']:.6f}/{j['maximum']:.6f}/{j['standard_deviation']:.6f}.")
        picks = selected[(selected.variant == variant) & selected.strategy.isin(
            ["minimum_macro_tvd", "minimum_total_slack", "minimum_five_way_js"])]
        for pick in picks.itertuples(index=False):
            lines.append(f"  - {STRATEGY_LABELS[pick.strategy]}: `{pick.order_text}`; TVD={pick.census_macro_tvd:.6f}, slack={pick.total_stage1_slack:,.0f}, JS={pick.five_way_js_distance:.6f} ({pick.tie_note}).")
    lines += [f"- By across-order macro-TVD standard deviation, {VARIANTS[most_sensitive]['label']} is most sensitive ({census.loc[most_sensitive, 'standard_deviation']:.6f}). Figure A shows that the TVD-, slack-, and JS-minimising orders need not coincide.", ""]
    lines += ["## C. Selected household-size-first ordering", ""]
    for variant in VARIANT_ORDER:
        r = main.loc[variant]
        n = int(census.loc[variant, "n"])
        lines.append(f"- {VARIANTS[variant]['label']}: macro TVD {r.census_macro_tvd:.6f} (rank {int(r.census_rank)}/{n}); total slack {r.total_stage1_slack:,.0f} (rank {int(r.slack_rank)}/{n}); five-way JS {r.five_way_js_distance:.6f} (rank {int(r.joint_rank)}/{n}).")
    lines += ["The production order is a policy decision: introducing `hhsize` first guarantees exact household-size marginals at initialization. It is not an empirically selected global optimum. Notably, it ranks 110/120 on macro TVD for SAA Relaxed despite ranking 26/120 on five-way JS.", ""]
    lines += ["## D. Heuristic effectiveness", ""]
    for variant in VARIANT_ORDER:
        p, h = main.loc[variant], heuristic.loc[variant]
        n = int(census.loc[variant, "n"])
        lines.append(
            f"- {VARIANTS[variant]['label']}: household-first TVD/slack/JS = {p.census_macro_tvd:.6f}/{p.total_stage1_slack:,.0f}/{p.five_way_js_distance:.6f} "
            f"(ranks {int(p.census_rank)}/{int(p.slack_rank)}/{int(p.joint_rank)} of {n}); unconstrained heuristic = "
            f"{h.census_macro_tvd:.6f}/{h.total_stage1_slack:,.0f}/{h.five_way_js_distance:.6f} "
            f"(ranks {int(h.census_rank)}/{int(h.slack_rank)}/{int(h.joint_rank)} of {n}).")
    lines += ["- `heuristic_strategy_comparison.csv` compares descending H, R-only, D-only, household-first-plus-H, and the three empirical minima. The results are mixed: no input-derived rule dominates every support policy or every objective.",
              "- BN-pool uses BN-specific rigidity and pressure scores; VISTA-derived rigidity is not silently transferred to that pool.", ""]
    lines += ["## E. Marginal slack", ""]
    for variant in VARIANT_ORDER:
        part = slack_position[slack_position.variant == variant]
        if np.isclose(part.mean_total_slack.max(), 0):
            lines.append(f"- {VARIANTS[variant]['label']}: all attributes at all positions have zero Stage 1 slack.")
            continue
        worst = part.sort_values("mean_total_slack", ascending=False).iloc[0]
        lines.append(f"- {VARIANTS[variant]['label']}: largest mean attribute-stage slack occurs for {ATTRIBUTE_LABELS[worst.attribute]} at position {int(worst.position)} ({worst.mean_total_slack:,.1f}; n={int(worst.contributing_permutations)}).")
    lines += ["- Figure C reports mean summed absolute count slack. Position 1 is zero by construction; zero first-position slack does not make the full ordering optimal. The VISTA Seed-only `hhinc`-first cell is N/A, not zero.",
              "- The restrictions are empirical pool-support restrictions and must not be interpreted as verified structural zeros.", ""]
    lines += ["## F. Rigidity and marginal pressure", ""]
    for variant in ["relaxed", "seed_only", "bn_pool"]:
        r = relationship[relationship.variant == variant].iloc[0]
        lines.append(f"- {VARIANTS[variant]['label']}: descriptive Spearman R–late-slack = {r.spearman_rigidity_late_slack:.3f}, D–late-slack = {r.spearman_pressure_late_slack:.3f}, and H–early-benefit = {r.spearman_score_early_benefit:.3f}.")
    lines += ["- Correlations are undefined for Unrestricted and Seed-free because slack is identically zero; they are not displayed in Figure D.",
              "- With only five attributes per policy, the mixed signs are descriptive and do not validate the heuristic statistically or causally. In particular, BN-pool has H–early-benefit Spearman correlation -0.300.", ""]
    lines += ["## G. Manuscript interpretation", "",
              "1. All 120 Unrestricted and Seed-free orders attain exact Census marginals, but their joint fidelity still varies (Figure A), so order remains relevant even when slack is zero.",
              f"2. VISTA Seed-only is most macro-TVD-sensitive: {census.loc['seed_only','minimum']:.6f}–{census.loc['seed_only','maximum']:.6f}, with SD {census.loc['seed_only','standard_deviation']:.6f} across 96 eligible orders (Figure B).",
              "3. Relaxed SAA separates the objectives clearly: minimum macro TVD has slack 5,318, whereas minimum total slack is 4,310 and occurs under another order (Figure A; Table `ordering_comparison_main.tex`).",
              "4. BN-pool shows the same distinction: the minimum-macro-TVD order has slack 2,428, while the minimum-slack order achieves 1,482.",
              "5. The production order trades Census accuracy for policy priority and relatively strong joint fidelity under Relaxed SAA: macro-TVD rank 110/120 versus JS rank 26/120.",
              "6. Figure C shows that late dwelling type and household income adjustments dominate slack under restricted support, but the responsible attribute differs by policy.",
              "7. The rigidity/pressure evidence is mixed rather than confirmatory; no heuristic strategy dominates every policy and objective (Figure D and `heuristic_strategy_comparison.csv`).", ""]
    lines += ["## H. Limitations", "",
              "- Ordering sensitivity is evaluated on one case study.",
              "- Within-dataset permutation sensitivity is not independent validation of the heuristic.",
              "- Total Stage 1 slack is an absolute count objective; macro Census TVD is an equally weighted normalised evaluation metric.",
              "- Fidelity to weighted VISTA is reference-joint fidelity, not evidence of population realism.",
              "- Empirical pool-support restrictions are not verified structural constraints.",
              "- Permutation variability is configuration sensitivity, not stochastic sampling variability.",
              "- The input-derived heuristic may perform differently under other datasets, pools, or support policies.", "",
              f"Validation status: {int(validations.passed.sum())}/{len(validations)} checks passed."]
    (output / "section_4_4_findings.md").write_text("\n".join(lines) + "\n")


def run_analysis(root: Path | None = None) -> dict[str, Any]:
    started = time.time()
    root = repository_root(root)
    output = output_path(root)
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    cache_files = {
        "metrics": "permutation_metrics.csv", "summary": "ordering_sensitivity_summary.csv",
        "selected": "selected_order_comparison.csv", "rankings": "ordering_rankings.csv",
        "scores": "heuristic_scores.csv", "slack_permutation": "slack_by_permutation.csv",
        "slack_attribute": "slack_by_attribute.csv", "slack_position": "slack_by_position.csv",
        "relationship": "slack_rigidity_relationship.csv",
        "strategies": "heuristic_strategy_comparison.csv", "validations": "validation_checks.csv",
    }
    if all((output / name).exists() for name in cache_files.values()):
        cached = {key: pd.read_csv(output / name) for key, name in cache_files.items()}
        if len(cached["metrics"]) == 576 and cached["validations"].passed.all():
            cached["summary"] = sensitivity_summary(cached["metrics"])
            cached["rankings"] = build_rankings(cached["metrics"])
            cached["selected"] = selected_order_comparison(cached["rankings"])
            cached["slack_position"] = aggregate_slack(cached["slack_attribute"])
            cached["relationship"] = rigidity_relationship(
                cached["metrics"], cached["slack_attribute"], cached["scores"])
            cached["strategies"] = heuristic_strategy_comparison(
                cached["rankings"], cached["scores"])
            cached["validations"] = add_aggregate_validations(
                cached["validations"], cached["slack_permutation"],
                cached["slack_attribute"], cached["slack_position"])
            cached["summary"].to_csv(output / cache_files["summary"], index=False)
            cached["rankings"].to_csv(output / cache_files["rankings"], index=False)
            cached["selected"].to_csv(output / cache_files["selected"], index=False)
            cached["slack_position"].to_csv(output / cache_files["slack_position"], index=False)
            cached["relationship"].to_csv(output / cache_files["relationship"], index=False)
            cached["strategies"].to_csv(output / cache_files["strategies"], index=False)
            cached["validations"].to_csv(output / cache_files["validations"], index=False)
            figure_a(cached["metrics"], output); figure_b(cached["metrics"], output)
            figure_c(cached["slack_position"], output); figure_d(cached["relationship"], output)
            write_latex_tables(output, cached["summary"], cached["selected"], cached["strategies"])
            write_findings(output, cached["metrics"], cached["summary"], cached["selected"],
                           cached["slack_position"], cached["relationship"], cached["strategies"],
                           cached["validations"])
            manifest = {
                "created_utc": pd.Timestamp.utcnow().isoformat(), "elapsed_seconds": time.time() - started,
                "permutations": 576, "zones": 691, "attributes": ATTRIBUTES,
                "joint_reference": "weighted ori_sample_hh.csv (_weight)",
                "census_metric": "mean zone-attribute TVD over 691 POAs x 5 attributes",
                "js_metric": "scipy.spatial.distance.jensenshannon(base=2)",
                "ranking": "competition rank (method=min), lower is better",
                "score_sources": {variant: spec["score_source"] for variant, spec in VARIANTS.items()},
                "input_configs": [spec["config"] for spec in VARIANTS.values()],
                "cache_reused": True,
            }
            (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            return {"output": output, "metrics": cached["metrics"], "summary": cached["summary"],
                    "selected": cached["selected"], "scores": cached["scores"],
                    "slack_position": cached["slack_position"], "relationship": cached["relationship"],
                    "strategies": cached["strategies"], "validations": cached["validations"],
                    "manifest": manifest}
    validations: list[dict[str, Any]] = []
    runs = discover_runs(root, validations)
    census_wide, census_long, zones = load_census(root / "IO/processed_data/census_POA_hh_saa.csv")
    _validation(validations, "census_zone_count", len(zones) == 691, f"observed={len(zones)}")
    vista = load_pool(root / "IO/processed_data/ori_sample_hh.csv", "_weight")
    bn = load_pool(root / "IO/processed_data/HH_pool_condensed.csv", "count")
    scores = calculate_heuristic_scores(vista, bn, census_wide)
    validate_heuristics(scores, validations)
    metrics, slack_permutation, slack_attribute = analyse_permutations(
        runs, vista, census_long, zones, validations)
    rankings = build_rankings(metrics)
    summary = sensitivity_summary(metrics)
    selected = selected_order_comparison(rankings)
    slack_position = aggregate_slack(slack_attribute)
    relationship = rigidity_relationship(metrics, slack_attribute, scores)
    strategies = heuristic_strategy_comparison(rankings, scores)
    validate_main_results(root, metrics, validations)
    _validation(validations, "joint_metric_bounds",
                metrics[["five_way_tvd", "five_way_js_distance", "mean_pairwise_tvd"]].apply(
                    lambda column: column.between(0, 1).all()).all(), "all TVD/JS values lie in [0,1]")
    _validation(validations, "all_slack_nonnegative",
                (slack_attribute[["total_slack", "positive_slack", "negative_slack"]] >= 0).all().all(),
                "all derived slack values are nonnegative")
    validation_frame = add_aggregate_validations(
        pd.DataFrame(validations), slack_permutation, slack_attribute, slack_position)

    metrics.to_csv(output / "permutation_metrics.csv", index=False)
    summary.to_csv(output / "ordering_sensitivity_summary.csv", index=False)
    selected.to_csv(output / "selected_order_comparison.csv", index=False)
    rankings.to_csv(output / "ordering_rankings.csv", index=False)
    scores.to_csv(output / "heuristic_scores.csv", index=False)
    slack_permutation.to_csv(output / "slack_by_permutation.csv", index=False)
    slack_attribute.to_csv(output / "slack_by_attribute.csv", index=False)
    slack_position.to_csv(output / "slack_by_position.csv", index=False)
    relationship.to_csv(output / "slack_rigidity_relationship.csv", index=False)
    strategies.to_csv(output / "heuristic_strategy_comparison.csv", index=False)
    validation_frame.to_csv(output / "validation_checks.csv", index=False)
    figure_a(metrics, output); figure_b(metrics, output)
    figure_c(slack_position, output); figure_d(relationship, output)
    write_latex_tables(output, summary, selected, strategies)
    write_findings(output, metrics, summary, selected, slack_position,
                   relationship, strategies, validation_frame)
    manifest = {
        "created_utc": pd.Timestamp.utcnow().isoformat(),
        "elapsed_seconds": time.time() - started,
        "permutations": len(metrics), "zones": len(zones),
        "attributes": ATTRIBUTES, "joint_reference": "weighted ori_sample_hh.csv (_weight)",
        "census_metric": "mean zone-attribute TVD over 691 POAs x 5 attributes",
        "js_metric": "scipy.spatial.distance.jensenshannon(base=2)",
        "ranking": "competition rank (method=min), lower is better",
        "score_sources": {variant: spec["score_source"] for variant, spec in VARIANTS.items()},
        "input_configs": [spec["config"] for spec in VARIANTS.values()],
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    if not validation_frame.passed.all():
        failed = validation_frame.loc[~validation_frame.passed, ["check", "detail"]]
        raise AssertionError("Section 4.4 validations failed:\n" + failed.to_string(index=False))
    return {"output": output, "metrics": metrics, "summary": summary,
            "selected": selected, "scores": scores, "slack_position": slack_position,
            "relationship": relationship, "strategies": strategies,
            "validations": validation_frame, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None)
    args = parser.parse_args()
    result = run_analysis(args.root)
    print(f"Section 4.4 outputs written to {result['output']}")


if __name__ == "__main__":
    main()



