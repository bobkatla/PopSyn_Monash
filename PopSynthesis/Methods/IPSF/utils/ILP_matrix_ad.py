import polars as pl
from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpMinimize, PULP_CBC_CMD
from typing import Tuple, Dict, List
from PopSynthesis.Methods.IPSF.const import count_field
from math import sqrt

# Define a small epsilon value
EPSILON = 1e-6  # Solve division by zeros
LARGE_M = 1e6  # Large M value for binary activation variables


def convert_to_required_ILP_format(
    syn_count: pl.DataFrame, att: str, adjusted_atts: List[str]
) -> pl.DataFrame:
    """Convert the disagg data to the required format for ILP"""
    # group by and pivot the syn to have adjusted_atts as index and states in att as columns
    assert count_field in syn_count.columns
    gb_syn = syn_count.group_by(pl.col(adjusted_atts + [att])).agg(
        pl.sum(count_field).alias(count_field)
    )
    ILP_formatted_syn = gb_syn.pivot(att, index=adjusted_atts, values=count_field)
    ILP_formatted_syn = ILP_formatted_syn.cast(
        {x: pl.Int32 for x in ILP_formatted_syn.columns if x not in adjusted_atts}
    )
    return ILP_formatted_syn


def convert_back_to_syn_count(
    ILP_formatted_syn: pl.DataFrame, att: str, adjusted_atts: List[str]
) -> pl.DataFrame:
    """Convert the ILP output to the original format"""
    # NOTE: this assumes all the other cols are the states
    states_of_att = [x for x in ILP_formatted_syn.columns if x not in adjusted_atts]
    unpivoted_df = ILP_formatted_syn.unpivot(
        states_of_att, index=adjusted_atts, variable_name=att, value_name=count_field
    ).filter(pl.col(count_field).is_not_null())
    return unpivoted_df


def _ILP_solving_adjustment(
    count_df: pl.DataFrame,
    states_diff: Dict[str, int],
    id_col=str,
    basic_filter: bool = True,
    deviation_type: str = "relative",
    spread_penalty: float = 0,
) -> Tuple[pl.DataFrame, Dict[str, int]]:
    # Clone count_table to prevent changes to the original
    ori_count_table = count_df.clone()
    count_table = count_df.clone()

    assert not count_table.select(pl.col(id_col)).is_duplicated().any()

    # Filter out not needed rows and columns
    zero_val_states = []
    if basic_filter:
        zero_val_states = [x for x in states_diff.keys() if states_diff[x] == 0]
        count_table = count_table.drop(zero_val_states)
        pos_states = [x for x in states_diff.keys() if states_diff[x] > 0]
        neg_states = [x for x in states_diff.keys() if states_diff[x] < 0]
        count_table = count_table.filter(
            ~pl.all_horizontal(pl.col(pos_states).is_null())
        )
        count_table = count_table.filter(
            ~pl.all_horizontal(pl.col(neg_states).is_null())
        )

    # Ensure each row has a unique identifier so we can reference rows without using an index
    rows = count_table[id_col].to_list()
    r_idx = {row: i for i, row in enumerate(rows)}  # NOTE: order must match
    columns = [col for col in count_table.columns if col != id_col]

    # Initialize the ILP problem
    problem = LpProblem("MatrixAdjustment", LpMinimize)

    # Initialize adjustment variables for each cell with a non-null value
    adjustments = {
        (i, j): LpVariable(
            f"A_{i}_{j}", lowBound=-count_table[r_idx[i], j], cat="Integer"
        )
        for i in rows
        for j in columns
        if count_table[r_idx[i], j] is not None
    }

    # Initialize slack variables for each column to allow deviations from target adjustments
    slack_pos = {
        j: LpVariable(f"slack_pos_{j}", lowBound=0, cat="Continuous") for j in columns
    }
    slack_neg = {
        j: LpVariable(f"slack_neg_{j}", lowBound=0, cat="Continuous") for j in columns
    }

    # Initialize deviation variables and activation indicators for each cell with a non-null value
    deviations = {
        (i, j): LpVariable(f"rel_dev_{i}_{j}", lowBound=0, cat="Continuous")
        for (i, j) in adjustments
    }
    # Binary activation variables to encourage spreading adjustments
    activations = {
        (i, j): LpVariable(f"act_{i}_{j}", cat="Binary") for (i, j) in adjustments
    }

    # Objective: Minimize the sum of deviations, slack, and activation penalties
    problem += lpSum(
        deviations[key] + slack_pos[j] + slack_neg[j] + spread_penalty * activations[key]
        for key in deviations
        for j in columns
    )

    # Row constraints: Ensure the sum of adjustments in each row is zero
    for i in rows:
        row_adjustments = [
            adjustments[(i, j)] for j in columns if (i, j) in adjustments
        ]
        problem += lpSum(row_adjustments) == 0

    # Column constraints with positive and negative slack for deviations
    for j, target_diff in states_diff.items():
        if j not in zero_val_states:
            col_adjustments = [
                adjustments[(i, j)] for i in rows if (i, j) in adjustments
            ]
            problem += (
                lpSum(col_adjustments) == target_diff + slack_pos[j] - slack_neg[j]
            )

    # Deviation and activation constraints
    for (i, j), var in adjustments.items():
        original_value = count_table[r_idx[i], j]
        # Non-negativity constraints
        problem += var >= -original_value  # Ensures X_ij + A_ij >= 0

        check_dev = None
        if deviation_type == "relative":
            # Handle zero values by assigning epsilon
            adjusted_value = original_value if original_value != 0 else EPSILON
            check_dev = deviations[(i, j)] * adjusted_value
        elif deviation_type == "absolute":
            check_dev = deviations[(i, j)]
            problem += check_dev >= original_value - (
                original_value + var
            )  # maintain closeness to original values
        else:
            raise ValueError(f"Unknown deviation type: {deviation_type}")

        # Deviation constraints
        problem += check_dev >= var  # dev >= adj
        problem += check_dev >= -var  # dev >= -adj

        # Link the activation variable to the deviation: activate if deviation > 0
        problem += deviations[(i, j)] <= activations[(i, j)] * LARGE_M  # Large M constraint

    # Solve the ILP
    problem.solve(PULP_CBC_CMD(msg=False))

    adjustment_remaining = {}
    # Check the solution status
    if LpStatus[problem.status] == "Optimal":
        # Apply adjustments to count_table based on the solution
        for (i, j), var in adjustments.items():
            ori_count_table = ori_count_table.with_columns(
                pl.when(pl.col(id_col) == i)
                .then(ori_count_table[j] + var.value())
                .otherwise(ori_count_table[j])
                .alias(j)
            )

        # Calculate the resulting actual column adjustments and print the achieved diff
        actual_diff = {
            j: sum(adjustments[(i, j)].value() for i in rows if (i, j) in adjustments)
            for j in columns
        }
        adjustment_remaining = {
            j: int(states_diff[j] - actual_diff[j]) for j in columns
        } | {j: 0 for j in zero_val_states}
    else:
        print("No feasible solution found.")

    return ori_count_table, adjustment_remaining


def update_count_tables(
    count_table: pl.DataFrame, states_diff: Dict[str, int], id_col: str, deviation_type: str = "relative", spread_penalty: float = 0
) -> Tuple[pl.DataFrame, int]:
    """Update count table with adjustments, ensuring row and column sums meet expected values."""
    assert sum(states_diff.values()) == 0

    expected_sum_row = count_table.select(
        pl.sum_horizontal(pl.exclude(id_col)).alias("row_sum")
    )
    expected_sum_col = count_table.select(
        [
            (pl.sum(col) + states_diff[col]).alias(f"{col}_sum")
            for col in count_table.columns
            if col != id_col
        ]
    )

    count_table, adjustment_remaining = _ILP_solving_adjustment(
        count_table, states_diff, id_col, deviation_type=deviation_type, spread_penalty=spread_penalty
    )
    # Resulting adjusted DataFrame
    result_sum_row = count_table.select(
        pl.sum_horizontal(pl.exclude(id_col)).alias("row_sum")
    )
    result_sum_col = count_table.select(
        [
            (pl.sum(col) + adjustment_remaining[col]).alias(f"{col}_sum")
            for col in count_table.columns
            if col != id_col
        ]
    )

    # Validate that the new column sums match expected adjustments after applying states_diff
    assert result_sum_row.equals(expected_sum_row)
    assert result_sum_col.equals(expected_sum_col)

    # Calculate remaining error as sum of squared remaining adjustments
    err_score = sum(value ** 2 for value in adjustment_remaining.values())

    return count_table, sqrt(err_score)
