"""Main place to run SAA for households synthesis"""


import json
import pandas as pd
from tempfile import NamedTemporaryFile
from PopSynthesis.Methods.IPSF.const import (
    data_dir,
    small_test_dir,
    processed_dir,
    zone_field,
    output_dir,
)
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import (
    adjust_kept_rec_match_census,
    get_diff_marg,
    convert_full_to_marg_count,
)
from PopSynthesis.Methods.IPSF.SAA.SAA import SAA
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import polars as pl
from typing import Tuple, List, Optional
import random
import numpy as np
from pathlib import Path
from typing import Literal
from PopSynthesis.Methods.IPSF.const import count_field
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import (
    concat_counts,
    counts_to_marginals,
    expand_counts,
    population_size,
    prune_oversupplied_counts,
    sample_without_replacement_counts,
)
from PopSynthesis.Methods.IPSF.SAA.operations.shared_vars import zero_cell_scope
from PopSynthesis.Methods.IPSF.SAA.direct_count_saa import run_direct_count_pass


def get_test_pp() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pp_marg = pd.read_csv(small_test_dir / "pp_small_marginals.csv", header=[0, 1])
    pp_marg = pp_marg.set_index(
        pp_marg.columns[pp_marg.columns.get_level_values(0) == zone_field][0]
    )
    pool = pd.read_csv(data_dir / "pp_sample_ipu.csv")
    pool = pool.drop(columns=["serialno", "sample_geog"])
    return pp_marg, pool


def get_test_hh() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(small_test_dir / "hh_marginals_small.csv", header=[0, 1])
    hh_marg = hh_marg.set_index(
        hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0]
    )
    pool = pd.read_csv(small_test_dir / "HH_pool_small_test.csv")
    return hh_marg, pool


def get_hh_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    hh_marg = pd.read_csv(data_dir / "hh_marginals_ipu.csv", header=[0, 1])
    hh_marg = hh_marg.drop(
        columns=hh_marg.columns[hh_marg.columns.get_level_values(0) == "sample_geog"][0]
    ).set_index(hh_marg.columns[hh_marg.columns.get_level_values(0) == zone_field][0])
    pool = pd.read_csv(processed_dir / "HH_pool.csv")
    return hh_marg, pool


def get_pp_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    pp_marg = pd.read_csv(data_dir / "person_marginals_ipu.csv", header=[0, 1])
    pp_marg = pp_marg.set_index(pp_marg.columns[pp_marg.columns.get_level_values(0) == zone_field][0])
    pool = pd.read_csv(processed_dir / "PP_pool.csv")
    return pp_marg, pool



def err_check_against_marg(
    syn_pop: pd.DataFrame, marg: pd.DataFrame, extra_rm_frac: float = 0, exclude_atts: List[str] = []
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # rm extra first
    assert extra_rm_frac <= 1 and extra_rm_frac >= 0
    remain_syn = syn_pop.copy()
    remain_syn = remain_syn.sample(frac=1-extra_rm_frac)
    print(f"removed first {len(syn_pop) - len(remain_syn)} agents")

    marg_from_created = convert_full_to_marg_count(remain_syn, [zone_field])
    converted_marg = marg
    converted_marg = converted_marg.drop(converted_marg.columns[converted_marg.columns.get_level_values(0).isin(exclude_atts)], axis=1)
    marg_from_created = marg_from_created.drop(marg_from_created.columns[marg_from_created.columns.get_level_values(0).isin(exclude_atts)], axis=1)

    diff_marg = get_diff_marg(converted_marg, marg_from_created)

    kept_syn = adjust_kept_rec_match_census(remain_syn, diff_marg)

    # checking
    kept_marg = convert_full_to_marg_count(kept_syn, [zone_field])
    kept_marg = kept_marg.drop(kept_marg.columns[kept_marg.columns.get_level_values(0).isin(exclude_atts)], axis=1)
    new_diff_marg = get_diff_marg(converted_marg, kept_marg)
    # check it is no neg indeed
    checking_not_neg = new_diff_marg < 0
    assert not checking_not_neg.any(axis=None)
    # now get the new marg
    new_diff_marg.index = new_diff_marg.index.astype(int)
    new_diff_marg.index.name = zone_field
    return kept_syn, new_diff_marg


def process_shuffle_order(
    orginal_order: List[str], shuffle_order: List[str], idx_atts_states: pd.MultiIndex, check_run_time: str, randomly_add_last: Optional[List[str]] = None
) -> List[str]:
    randomly_add_last = [] if randomly_add_last is None else randomly_add_last
    if check_run_time == "first":
        # First run, change the order from most states to least states
        flatten_idx = idx_atts_states.to_frame(index=False, name=["att", "state"])
        count_state = flatten_idx.groupby("att").count().sort_values("state", ascending=False)
        return count_state.index.tolist()
    elif check_run_time == "last":
        assert set(shuffle_order) <= set(orginal_order)
        not_in_shuffle = [x for x in orginal_order if x not in shuffle_order]
        random.shuffle(not_in_shuffle)
        return shuffle_order + not_in_shuffle
    else:
        # mid case, this will be a number
        n_run_time = int(check_run_time)
        # want each value got to be the first once at least
        first_att = orginal_order[n_run_time % len(orginal_order)]
        random_boolean = random.choices([True, False], weights=[1, 3])[0]
        
        random.shuffle(orginal_order)
        orginal_order.remove(first_att)
        orginal_order.insert(0, first_att)
        if random_boolean and len(randomly_add_last) > 0:
            for att in randomly_add_last:
                orginal_order.remove(att)
                orginal_order.append(att)

        return orginal_order


def saa_run(
    targeted_marg: pd.DataFrame,
    count_pool: pl.DataFrame,
    considered_atts: Optional[List[str]] = None,
    ordered_to_adjust_atts: Optional[List[str]] = None,
    last_adjustment_order: Optional[List[str]] = None,
    max_run_time: int = 30,
    extra_rm_frac: float = 0,
    output_each_step: bool = False,
    add_name_for_step_output: str = "",
    include_zero_cell_values: bool = False,
    randomly_add_last: Optional[List[str]] = None,
    meta_output_dir: Path = output_dir,
    stochastic_mode: Literal[
        "count_native", "legacy_exact", "count_native_ilp"
    ] = "count_native",
    output_format: Literal["rows", "counts"] = "rows",
    direct_solver_mode: Literal["single_stage", "two_stage"] = "single_stage",
    direct_adjustment_backend: Literal[
        "cbc", "min_cost_flow"
    ] = "cbc",
    direct_pool_cache: dict | None = None,
    direct_diagnostics_output: bool = False,
) -> Tuple[pl.DataFrame, List[int]]:
    """Run one of the complete-record, legacy, or direct-count SAA modes."""
    if stochastic_mode not in {
        "count_native",
        "legacy_exact",
        "count_native_ilp",
    }:
        raise ValueError(
            "stochastic_mode must be 'count_native', 'legacy_exact', or "
            "'count_native_ilp'"
        )
    if considered_atts is None or ordered_to_adjust_atts is None:
        raise ValueError("considered_atts and ordered_to_adjust_atts are required")
    if output_format not in {"rows", "counts"}:
        raise ValueError("output_format must be 'rows' or 'counts'")
    if output_format == "counts" and stochastic_mode != "count_native_ilp":
        raise ValueError("output_format='counts' is currently direct-mode only")
    if direct_solver_mode not in {"single_stage", "two_stage"}:
        raise ValueError(
            "direct_solver_mode must be 'single_stage' or 'two_stage'"
        )
    if direct_adjustment_backend not in {"cbc", "min_cost_flow"}:
        raise ValueError(
            "direct_adjustment_backend must be 'cbc' or 'min_cost_flow'"
        )
    ordered = list(ordered_to_adjust_atts)
    final_order = list(last_adjustment_order or [])
    random_last = list(randomly_add_last or [])
    if stochastic_mode == "count_native_ilp":
        return _saa_run_count_native_ilp(
            targeted_marg,
            count_pool,
            considered_atts,
            ordered,
            final_order,
            max_run_time,
            extra_rm_frac,
            output_each_step,
            include_zero_cell_values,
            random_last,
            meta_output_dir,
            output_format,
            direct_solver_mode,
            direct_adjustment_backend,
            direct_pool_cache,
            direct_diagnostics_output,
        )
    with zero_cell_scope():
        if stochastic_mode == "legacy_exact":
            return _saa_run_legacy(
                targeted_marg,
                count_pool,
                considered_atts,
                ordered,
                final_order,
                max_run_time,
                extra_rm_frac,
                output_each_step,
                include_zero_cell_values,
                random_last,
                meta_output_dir,
            )
        return _saa_run_count_native(
            targeted_marg,
            count_pool,
            considered_atts,
            ordered,
            final_order,
            max_run_time,
            extra_rm_frac,
            output_each_step,
            include_zero_cell_values,
            random_last,
            meta_output_dir,
        )


def _validate_run_inputs(targeted_marg, considered_atts, ordered_to_adjust_atts):
    assert set(ordered_to_adjust_atts) <= set(considered_atts)
    atts_in_marg = set(targeted_marg.columns.get_level_values(0)) - {zone_field}
    assert set(ordered_to_adjust_atts) <= atts_in_marg
    assert zone_field in targeted_marg.index.name
    return atts_in_marg


def _order_for_pass(
    ordered_atts, final_order, marg_columns, pass_index, max_passes, random_last
):
    marker = str(pass_index)
    if pass_index == 0:
        marker = "first"
    if pass_index == max_passes - 1:
        marker = "last"
    return process_shuffle_order(
        ordered_atts,
        final_order,
        marg_columns,
        check_run_time=marker,
        randomly_add_last=random_last,
    )


def _metric_rows(
    population_count,
    original_marg,
    pass_index,
    adjustment_index,
    adjusted_attribute,
    target_size,
    zone_errors,
):
    attributes = list(dict.fromkeys(original_marg.columns.get_level_values(0)))
    synthetic = counts_to_marginals(population_count, attributes)
    target = original_marg.copy()
    target.index = target.index.astype(str)
    synthetic = synthetic.reindex(index=target.index, columns=target.columns, fill_value=0)
    differences = target.astype(float) - synthetic.astype(float)
    rmses = np.sqrt((differences ** 2).mean(axis=0))
    mean_rmse = float(rmses.mean())
    ilp_error = float(sum(zone_errors.values())) if zone_errors else 0.0
    return [
        {
            "pass": pass_index,
            "adjustment_index": adjustment_index,
            "adjusted_attribute": adjusted_attribute,
            "target_n_syn": target_size,
            "evaluated_attribute": attribute,
            "evaluated_state": state,
            "rmse": float(value),
            "mean_rmse": mean_rmse,
            "ilp_error": ilp_error,
        }
        for (attribute, state), value in rmses.items()
    ]


def _write_diagnostics(meta_output_dir, metrics, snapshots):
    meta_output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).to_csv(
        meta_output_dir / "adjustment_metrics.csv", index=False
    )
    for pass_index, population in snapshots:
        expand_counts(population).write_csv(
            meta_output_dir / f"loopback_population_{pass_index}.csv"
        )


def _zone_as_string(population):
    return population.with_columns(pl.col(zone_field).cast(pl.String))


def _err_check_count_population(
    population, marg, extra_rm_frac, excluded_atts
):
    assert 0 <= extra_rm_frac <= 1
    retain_n = round(population_size(population) * (1 - extra_rm_frac))
    remaining = sample_without_replacement_counts(population, retain_n)
    adjusted_attributes = [
        attribute
        for attribute in dict.fromkeys(marg.columns.get_level_values(0))
        if attribute not in excluded_atts
    ]
    created_marg = counts_to_marginals(remaining, adjusted_attributes)
    converted_marg = marg.drop(
        marg.columns[marg.columns.get_level_values(0).isin(excluded_atts)], axis=1
    )
    diff_marg = get_diff_marg(converted_marg, created_marg)
    kept = prune_oversupplied_counts(remaining, diff_marg)
    kept_marg = counts_to_marginals(kept, adjusted_attributes)
    new_diff = get_diff_marg(converted_marg, kept_marg)
    assert not (new_diff < 0).any(axis=None)
    new_diff.index = new_diff.index.astype(int)
    new_diff.index.name = zone_field
    return kept, new_diff


def _saa_run_count_native(
    targeted_marg,
    count_pool,
    considered_atts,
    ordered_to_adjust_atts,
    last_adjustment_order,
    max_run_time,
    extra_rm_frac,
    output_each_step,
    include_zero_cell_values,
    randomly_add_last,
    meta_output_dir,
):
    pool_keys = [column for column in count_pool.columns if column != count_field]
    count_pool = count_pool.sort(pool_keys)
    atts_in_marg = _validate_run_inputs(
        targeted_marg, considered_atts, ordered_to_adjust_atts
    )
    excluded_atts = list(set(considered_atts) - set(ordered_to_adjust_atts))
    original_marg = targeted_marg.copy()
    residual_size = int(targeted_marg.sum().sum() / len(atts_in_marg))
    retained_populations = []
    target_sizes = []
    metrics = []
    snapshots = []
    current_order = list(ordered_to_adjust_atts)

    pass_index = 0
    while pass_index < max_run_time and residual_size > 0:
        pass_order = _order_for_pass(
            current_order,
            last_adjustment_order,
            targeted_marg.columns,
            pass_index,
            max_run_time,
            randomly_add_last,
        )
        current_order = pass_order
        target_sizes.append(residual_size)
        saa = SAA(targeted_marg, considered_atts, pass_order, count_pool)
        pass_population, steps = saa.run_count(include_zero_cell_values)
        assert population_size(pass_population) == residual_size

        if output_each_step:
            for adjustment_index, attribute, step_population, zone_errors in steps:
                cumulative = concat_counts(
                    retained_populations + [_zone_as_string(step_population)]
                )
                metrics.extend(
                    _metric_rows(
                        cumulative,
                        original_marg,
                        pass_index,
                        adjustment_index,
                        attribute,
                        population_size(step_population),
                        zone_errors,
                    )
                )
            snapshots.append(
                (
                    pass_index,
                    concat_counts(
                        retained_populations + [_zone_as_string(pass_population)]
                    ),
                )
            )

        pass_index += 1
        if pass_index == max_run_time:
            retained_populations.append(_zone_as_string(pass_population))
        else:
            kept, targeted_marg = _err_check_count_population(
                pass_population, targeted_marg, extra_rm_frac, excluded_atts
            )
            if population_size(kept) > 0:
                retained_populations.append(kept)
            residual_size = population_size(pass_population) - population_size(kept)

    final_count = concat_counts(retained_populations)
    if output_each_step:
        _write_diagnostics(meta_output_dir, metrics, snapshots)
    final = expand_counts(final_count.select(considered_atts + [zone_field, count_field]))
    return final, target_sizes


def _validate_direct_mode_inputs(
    targeted_marg,
    count_pool,
    considered_atts,
    ordered_to_adjust_atts,
    last_adjustment_order,
    max_run_time,
    extra_rm_frac,
    include_zero_cell_values,
    randomly_add_last,
):
    errors = []
    if not ordered_to_adjust_atts:
        errors.append("ordered_to_adjust_atts must not be empty")
    if considered_atts != ordered_to_adjust_atts:
        errors.append(
            "considered_atts must exactly equal ordered_to_adjust_atts"
        )
    if len(targeted_marg) != 1 or not targeted_marg.index.is_unique:
        errors.append("exactly one Census zone is required")
    if not isinstance(max_run_time, int) or isinstance(max_run_time, bool) or max_run_time < 1:
        errors.append("max_run_time must be a positive integer")
    if not 0 <= extra_rm_frac <= 1:
        errors.append("extra_rm_frac must be between 0 and 1")
    if include_zero_cell_values:
        errors.append("include_zero_cell_values=True is unsupported")
    if len(set(ordered_to_adjust_atts)) != len(ordered_to_adjust_atts):
        errors.append("ordered_to_adjust_atts must not contain duplicates")
    marginal_attributes = set(targeted_marg.columns.get_level_values(0)) - {
        zone_field
    }
    if marginal_attributes != set(ordered_to_adjust_atts):
        errors.append(
            "marginals must contain exactly the controlled attributes"
        )
    if not set(last_adjustment_order) <= set(ordered_to_adjust_atts):
        errors.append("last_adjustment_order must contain only controlled attributes")
    if not set(randomly_add_last) <= set(ordered_to_adjust_atts):
        errors.append("randomly_add_last must contain only controlled attributes")
    missing_pool = (
        set(considered_atts) | {count_field}
    ) - set(count_pool.columns)
    if missing_pool:
        errors.append(f"pool is missing columns {sorted(missing_pool)}")
    if errors:
        raise ValueError("Invalid count_native_ilp configuration: " + "; ".join(errors))


def _write_frame_atomically(frame: pd.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            frame.to_csv(temporary, index=False)
        temporary_path.replace(output_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_direct_diagnostics(meta_output_dir, adjustments, loopbacks):
    meta_output_dir.mkdir(parents=True, exist_ok=True)
    _write_frame_atomically(
        pd.DataFrame(adjustments),
        meta_output_dir / "direct_count_adjustments.csv",
    )
    _write_frame_atomically(
        pd.DataFrame(loopbacks),
        meta_output_dir / "direct_count_loopbacks.csv",
    )


def _saa_run_count_native_ilp(
    targeted_marg,
    count_pool,
    considered_atts,
    ordered_to_adjust_atts,
    last_adjustment_order,
    max_run_time,
    extra_rm_frac,
    output_each_step,
    include_zero_cell_values,
    randomly_add_last,
    meta_output_dir,
    output_format,
    direct_solver_mode,
    direct_adjustment_backend,
    direct_pool_cache,
    direct_diagnostics_output,
):
    """Run progressive direct-count passes with existing count loop-backs."""

    _validate_direct_mode_inputs(
        targeted_marg,
        count_pool,
        considered_atts,
        ordered_to_adjust_atts,
        last_adjustment_order,
        max_run_time,
        extra_rm_frac,
        include_zero_cell_values,
        randomly_add_last,
    )
    pool_keys = [column for column in count_pool.columns if column != count_field]
    count_pool = count_pool.with_columns(
        [pl.col(attribute).cast(pl.String) for attribute in considered_atts]
    ).sort(pool_keys)
    original_marg = targeted_marg.copy()
    current_marg = targeted_marg.copy()
    residual_size = int(
        current_marg[ordered_to_adjust_atts[0]].iloc[0].sum()
    )
    retained_populations = []
    target_sizes = []
    adjustments = []
    loopbacks = []
    current_order = list(ordered_to_adjust_atts)

    pass_index = 0
    while pass_index < max_run_time and residual_size > 0:
        pass_order = (
            list(ordered_to_adjust_atts)
            if max_run_time == 1
            else _order_for_pass(
                current_order,
                last_adjustment_order,
                current_marg.columns,
                pass_index,
                max_run_time,
                randomly_add_last,
            )
        )
        current_order = pass_order
        target_sizes.append(residual_size)
        pass_result = run_direct_count_pass(
            current_marg,
            count_pool,
            pass_order,
            pass_index=pass_index,
            solver_mode=direct_solver_mode,
            adjustment_backend=direct_adjustment_backend,
            conditional_cache=direct_pool_cache,
        )
        pass_population = _zone_as_string(pass_result.population)
        if population_size(pass_population) != residual_size:
            raise AssertionError(
                "Direct-count pass population does not match residual target"
            )
        adjustments.extend(pass_result.adjustments)

        pass_index += 1
        is_final_allowed_pass = pass_index == max_run_time
        requested_retention = residual_size
        kept_size = residual_size
        next_residual_size = 0
        if is_final_allowed_pass:
            retained_populations.append(pass_population)
        else:
            requested_retention = round(residual_size * (1 - extra_rm_frac))
            kept, next_marg = _err_check_count_population(
                pass_population,
                current_marg,
                extra_rm_frac,
                [],
            )
            kept_size = population_size(kept)
            if kept_size > 0:
                retained_populations.append(_zone_as_string(kept))
            next_residual_size = residual_size - kept_size
            current_marg = next_marg
            residual_size = next_residual_size

        loopbacks.append(
            {
                "mode": "count_native_ilp",
                "zone": str(original_marg.index[0]),
                "pass": pass_index - 1,
                "pass_order_json": json.dumps(pass_order),
                "pass_target_population": target_sizes[-1],
                "generated_population": population_size(pass_population),
                "requested_retention_population": requested_retention,
                "kept_after_pruning": kept_size,
                "residual_population": next_residual_size,
                "cumulative_retained_population": sum(
                    population_size(population)
                    for population in retained_populations
                ),
                "terminated_after_pass": is_final_allowed_pass
                or next_residual_size == 0,
            }
        )

    final_count = concat_counts(retained_populations)
    final_marg = counts_to_marginals(final_count, considered_atts)
    expected = original_marg.copy()
    expected.index = expected.index.astype(str)
    final_marg = final_marg.reindex(
        index=expected.index, columns=expected.columns, fill_value=0
    )
    exact_marginals = final_marg.astype(int).equals(expected.astype(int))
    if not exact_marginals and direct_solver_mode != "two_stage":
        raise AssertionError("Final direct-count population does not match marginals")
    if output_each_step or direct_diagnostics_output:
        _write_direct_diagnostics(meta_output_dir, adjustments, loopbacks)
    selected = final_count.select(
        [zone_field, *considered_atts, count_field]
    ).sort([zone_field, *considered_atts])
    final = selected if output_format == "counts" else expand_counts(
        selected.select(considered_atts + [zone_field, count_field])
    )
    return final, target_sizes


def _saa_run_legacy(
    targeted_marg,
    count_pool,
    considered_atts,
    ordered_to_adjust_atts,
    last_adjustment_order,
    max_run_time,
    extra_rm_frac,
    output_each_step,
    include_zero_cell_values,
    randomly_add_last,
    meta_output_dir,
):
    atts_in_marg = _validate_run_inputs(
        targeted_marg, considered_atts, ordered_to_adjust_atts
    )
    excluded_atts = list(set(considered_atts) - set(ordered_to_adjust_atts))
    original_marg = targeted_marg.copy()
    residual_size = int(targeted_marg.sum().sum() / len(atts_in_marg))
    retained_counts = []
    target_sizes = []
    metrics = []
    snapshots = []
    pass_index = 0
    current_order = list(ordered_to_adjust_atts)

    while pass_index < max_run_time and residual_size > 0:
        pass_order = _order_for_pass(
            current_order,
            last_adjustment_order,
            targeted_marg.columns,
            pass_index,
            max_run_time,
            randomly_add_last,
        )
        current_order = pass_order
        target_sizes.append(residual_size)
        saa = SAA(targeted_marg, considered_atts, pass_order, count_pool)
        pass_population, steps = saa.run_with_steps(include_zero_cell_values)
        assert len(pass_population) == residual_size

        if output_each_step:
            for adjustment_index, attribute, step_population, zone_errors in steps:
                step_count = _zone_as_string(condense_df(step_population))
                cumulative = concat_counts(retained_counts + [step_count])
                metrics.extend(
                    _metric_rows(
                        cumulative,
                        original_marg,
                        pass_index,
                        adjustment_index,
                        attribute,
                        len(step_population),
                        zone_errors,
                    )
                )
            snapshots.append(
                (
                    pass_index,
                    concat_counts(
                        retained_counts
                        + [_zone_as_string(condense_df(pass_population))]
                    ),
                )
            )

        pass_index += 1
        if pass_index == max_run_time:
            retained_counts.append(_zone_as_string(condense_df(pass_population)))
        else:
            kept, targeted_marg = err_check_against_marg(
                pass_population.to_pandas(),
                targeted_marg,
                extra_rm_frac,
                excluded_atts,
            )
            if len(kept) > 0:
                kept_pl = pl.from_pandas(kept).with_columns(
                    pl.col(zone_field).cast(pl.String)
                )
                retained_counts.append(condense_df(kept_pl))
            residual_size = len(pass_population) - len(kept)

    if output_each_step:
        _write_diagnostics(meta_output_dir, metrics, snapshots)
    final_count = concat_counts(retained_counts)
    final = expand_counts(final_count.select(considered_atts + [zone_field, count_field]))
    return final, target_sizes
