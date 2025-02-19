import polars as pl
import pandas as pd
import numpy as np
from PopSynthesis.Methods.IPSF.const import zone_field, count_field
from PopSynthesis.Methods.IPSF.utils.ILP_matrix_ad import (
    convert_to_required_ILP_format,
    convert_back_to_syn_count,
    update_count_tables,
)
from PopSynthesis.Methods.IPSF.utils.condensed import explode_df
from PopSynthesis.Generator_data.generate_combine_census.utils import TRS
from PopSynthesis.Methods.IPSF.SAA.operations.shared_vars import update_zero_cells, get_zero_cells_all
from typing import List, Dict, Tuple


def convert_to_ILP_inputs(
    syn_count: pl.DataFrame,
    att: str,
    adjusted_atts: List[str],
    pool_count: pl.DataFrame,
) -> pl.DataFrame:
    """Convert the syn and pool to the required format for ILP"""
    # group by and pivot the syn to have adjusted_atts as index and states in att as columns
    assert count_field in syn_count.columns
    converted_syn = convert_to_required_ILP_format(syn_count, att, adjusted_atts)
    converted_syn = converted_syn.fill_null(-1)

    # repeat the same for pool, maybe there can be faster as we want to check only whether it is feasible or not
    converted_pool = convert_to_required_ILP_format(pool_count, att, adjusted_atts)
    converted_pool = converted_pool.with_columns(
        pl.when(pl.col(pl.Int32).is_not_null())
        .then(pl.lit(0))
        .otherwise(pl.col(pl.Int32))
        .name.suffix("")
    )

    # combine them both to check
    converted_syn = converted_syn.to_pandas().set_index(adjusted_atts)
    converted_pool = converted_pool.to_pandas().set_index(adjusted_atts)

    missing_cols_in_syn = [x for x in converted_pool.columns if x not in converted_syn.columns]
    converted_syn[missing_cols_in_syn] = 0

    assert set(converted_syn.columns) == set(converted_pool.columns)
    filtered_pool_by_index = converted_pool.loc[converted_syn.index]
    result = converted_syn + filtered_pool_by_index
    result = result.replace(-1, 0)
    return pl.from_pandas(result.reset_index())


def process_pool_to_sample_count(
    pool: pl.DataFrame,
    considered_atts: List[str],
    not_considered_atts: List[str],
    other_atts_col: str = "oa",
    weights_col: str = "w",
) -> pl.DataFrame:
    """Process the pool to sample count"""
    assert count_field in pool.columns
    if len(not_considered_atts) == 0:
        return pool
    pool = pool.with_columns(pl.concat_list(not_considered_atts).alias(other_atts_col))
    gb_pool = pool.group_by(considered_atts).agg(
        pl.col(other_atts_col), pl.col(count_field).alias(weights_col)
    )
    return gb_pool


def sample_full_from_combined_df(
    combined_df: pl.DataFrame,
    count_field: str,
    other_atts_col: str,
    weights_col: str,
    other_atts: List[str],
) -> pl.DataFrame:
    """Sample the full records from the combined_df"""
    assert count_field in combined_df.columns
    assert other_atts_col in combined_df.columns
    assert weights_col in combined_df.columns
    expected_sum = combined_df[count_field].sum()

    def process_row_to_sample(r):
        result = np.random.choice(
            a=range(0, len(r[other_atts_col])),
            size=int(r[count_field]),
            p=[x / sum(r[weights_col]) for x in r[weights_col]],
            replace=True,
        )
        return [r[other_atts_col][x] for x in result]

    draw_col = "draw"
    updated_df = combined_df.with_columns(
        pl.struct(pl.col([other_atts_col, weights_col, count_field]))
        .map_elements(process_row_to_sample, return_dtype=pl.List(pl.List(str)))
        .alias(draw_col)
    )
    updated_df = updated_df.drop([other_atts_col, weights_col, count_field])
    updated_df = updated_df.explode(draw_col)
    final_result = updated_df.with_columns(
        pl.col(draw_col).list.to_struct(
            fields=lambda idx: other_atts[idx], n_field_strategy="max_width"
        )
    ).unnest(draw_col)
    assert len(final_result) == expected_sum
    return final_result


def handle_zero_cells(process_df: pl.DataFrame, missing_records: pl.DataFrame, processed_pool: pl.DataFrame, last_adjustment: bool, include_value: bool = False):
    # Handle zero cells directly or delete completely
    # NOTE: working in progress for now, not sure how to handle yet
    if include_value:
        print(process_df)
        print(missing_records)
        print(processed_pool)
        if not last_adjustment:
            # if this the last adjustment, just return process_df concat with missing_records
            zero_cells = get_zero_cells_all()
            common_cols = list(set(processed_pool.columns) & set(missing_records.columns))
            missing_records = missing_records.with_row_index("temp_row_id")
            # Perform cross join, maybe computing expensive
            df_merged = missing_records.join(processed_pool, how="cross")
            # Filter rows where common columns match (except "X" can match anything)
            for col in common_cols:
                if col in zero_cells.keys():
                    df_merged = df_merged.filter(
                        (df_merged[f"{col}_right"] == df_merged[col]) |
                        (df_merged[col].is_in(zero_cells[col])) |
                        (df_merged[f"{col}_right"].is_in(zero_cells[col]))
                    )
                else:
                    df_merged = df_merged.filter(
                        (df_merged[f"{col}_right"] == df_merged[col])
                    )
            # Drop duplicate column suffixes
            df_merged = df_merged.drop([f"{col}_right" for col in common_cols])
            df_merged = df_merged.sample(fraction=1,shuffle=True)
            missing_records = df_merged.group_by("temp_row_id").agg(pl.all().first())
        # combine to find results
        return pl.concat([process_df, missing_records.select(process_df.columns)])
    else:
        if len(process_df) == 0:
            return process_df
        # Calculate the sum of the count_field of the missing records
        missing_records_sum = missing_records[count_field].sum()
        # Get the current total count
        current_total = process_df[count_field].sum()
        # Compute the proportion of each row's count relative to the total count
        proportions = process_df[count_field] / current_total
        # Use TRS to adjust, not rounding
        # Distribute the new value according to the existing proportions
        distributed_values = (proportions * missing_records_sum)
        rounded_distributed_values = TRS(list(distributed_values), missing_records_sum)
        assert sum(rounded_distributed_values) == missing_records_sum
        
        # Update the count column
        process_df = process_df.with_columns((process_df[count_field] + rounded_distributed_values).alias(count_field))
        return process_df
    

def simple_zero_cells_handle(process_df: pl.DataFrame, missing_records: pl.DataFrame, processed_pool: pl.DataFrame, last_adjustment: bool, include_value: bool = False):
    if not include_value:
        return handle_zero_cells(process_df, missing_records, processed_pool, last_adjustment, include_value)
    else:
        print("FOR NOW, ONLY THE LAST ADJUSTMENT IS SUPPORTED TO GET ZERO_CELLS VALUE")
        if last_adjustment:
            return pl.concat([process_df, missing_records.select(process_df.columns)])
        else:
            return handle_zero_cells(process_df, missing_records, processed_pool, False, False)


def sample_count_syn_to_full(
    syn_count: pl.DataFrame, pool: pl.DataFrame
) -> pl.DataFrame:
    """Sample the syn_count to full records"""
    assert set(syn_count.columns) <= set(pool.columns)
    other_atts_col = "other_atts"
    weights_col = "weights"
    considered_atts = [x for x in syn_count.columns if x != count_field]
    not_considered_atts = [x for x in pool.columns if x not in considered_atts+[count_field]]
    
    expected_sum = int(syn_count[count_field].sum())
    processed_pool = process_pool_to_sample_count(
        pool, considered_atts, not_considered_atts, other_atts_col, weights_col
    )
    if count_field in processed_pool.columns:
        # this means we are at the last adjustment
        processed_pool = processed_pool.drop(count_field)
    process_df = syn_count.join(
        processed_pool, on=[x for x in syn_count.columns if x != count_field]
    )
    missing_records = syn_count.join(
        process_df, on=[x for x in syn_count.columns if x != count_field], how="anti"
    )
    assert len(process_df) + len(missing_records) == len(syn_count)
    last_adjustment = False
    if len(not_considered_atts) == 0:
        last_adjustment = True

    if len(missing_records) > 0:
        process_df = simple_zero_cells_handle(process_df, missing_records, processed_pool, last_adjustment, include_value=True)

        if len(process_df) == 0: #This means did not update the process_df
            # Random sample from pool
            temp_pool = pool.to_pandas()
            random_syn = temp_pool.sample(n=expected_sum, replace=True, weights=count_field)
            return pl.from_pandas(random_syn.drop(columns=[count_field]))
    
    assert process_df[count_field].sum() == expected_sum
    
    if last_adjustment:
        # this is for the last att adjustment
        return explode_df(process_df, weight_col=count_field)
    else:
        return sample_full_from_combined_df(
            combined_df=process_df,
            count_field=count_field,
            other_atts_col=other_atts_col,
            weights_col=weights_col,
            other_atts=not_considered_atts
        )


def ILP_zone_adjustment(
    att: str,
    curr_count_syn: pl.DataFrame,
    diff_zone_census: Dict[str, int],
    count_pool: pl.DataFrame,
    adjusted_atts: List[str],
) -> Tuple[pl.DataFrame, int]:
    """Solved using ILP to output the adjusted syn_pop"""
    assert len(curr_count_syn[zone_field].unique()) == 1
    zone = curr_count_syn[zone_field].unique()[0]
    # convert current syn to the ILP_required format, considered pool as well for finding feasible solution
    converted_syn = convert_to_ILP_inputs(
        curr_count_syn, att, adjusted_atts, count_pool
    )
    # Solve using ILP
    # Adding the id col here, as we need to reference the rows
    row_id = "row_id"
    converted_syn = converted_syn.with_row_index(row_id)
    store_prev_atts_with_id = converted_syn[adjusted_atts + [row_id]]
    considered_atts = [x for x in converted_syn.columns if x not in adjusted_atts]
    count_table = converted_syn[considered_atts]

    # handle zero cells
    zero_cells = set(diff_zone_census.keys()) - set(count_table.columns)
    if len(zero_cells) > 0:
        for zero_cell in zero_cells:
            update_zero_cells(att, zero_cell)
        count_table = count_table.with_columns(
            [pl.lit(0).alias(x) for x in zero_cells]
        )
    
    adjusted_results, err_score = update_count_tables(
        count_table, diff_zone_census, row_id, deviation_type="absolute"
    )
    updated_syn = store_prev_atts_with_id.join(adjusted_results, on=row_id).drop(row_id)
    assert len(updated_syn) == len(converted_syn)
    # Convert the result back to the original format
    updated_syn_count = convert_back_to_syn_count(updated_syn, att, adjusted_atts)
    assert updated_syn_count[count_field].sum() == curr_count_syn[count_field].sum()
    updated_syn_count = updated_syn_count.filter(pl.col(count_field) > 0)
    resulted_syn = sample_count_syn_to_full(updated_syn_count, count_pool)
    resulted_syn = resulted_syn.with_columns(pl.lit(zone).alias(zone_field))
    return resulted_syn, err_score
