"""Testing the updates given a formatted counts table"""


import pandas as pd
import polars as pl
from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpMinimize

test_df = pd.DataFrame(
    {
        "s1": [None, 1, 2, 2, 33, 3],
        "s2": [None, 32, 12, 2, 0, None],
        "s3": [10, 20, 30, 40, 50, 60],
        "s4": [0, 34, None, 34, 50, 60],
        "s5": [10, 2, None, 123, 3, 432],
    },
    index=["a", "b", "c", "d", "e", "f"],
)
test_diff = pd.Series([-3, -8, 0, 5, 6], index=["s1", "s2", "s3", "s4", "s5"])


def update_count_tables(count_table: pd.DataFrame, states_diff: pd.Series) -> pd.DataFrame:
    """Temp give the func here for debugging"""
    assert sum(states_diff) == 0
    expected_sum_row = count_table.sum(axis=1)
    expected_sum_col = count_table.sum(axis=0) + states_diff

    # Initialize the ILP problem
    problem = LpProblem("MatrixAdjustment", LpMinimize)  # Minimize adjustments to maintain distribution

    # Initialize adjustment variables and absolute value variables
    adjustments = {(i, j): LpVariable(f"A_{i}_{j}", cat="Integer") 
                for i in test_df.index for j in test_df.columns 
                if pd.notnull(test_df.loc[i, j])}

    # Auxiliary variables to represent the absolute value of adjustments
    abs_adjustments = {key: LpVariable(f"abs_A_{key[0]}_{key[1]}", lowBound=0, cat="Continuous")
                    for key in adjustments}

    # Objective: Minimize the sum of absolute adjustments
    problem += lpSum(abs_adjustments[key] for key in abs_adjustments)

    # Row constraints: Ensure the sum of adjustments in each row is zero
    for i in test_df.index:
        row_adjustments = [adjustments[(i, j)] for j in test_df.columns if (i, j) in adjustments]
        problem += lpSum(row_adjustments) == 0

    # Column constraints: Ensure the sum of adjustments in each column matches test_diff
    for j, diff in test_diff.items():
        col_adjustments = [adjustments[(i, j)] for i in test_df.index if (i, j) in adjustments]
        problem += lpSum(col_adjustments) == diff

    # Non-negativity constraints: Ensure each adjusted cell remains positive
    for (i, j), var in adjustments.items():
        original_value = test_df.loc[i, j]
        problem += var >= -original_value  # Ensures X_ij + A_ij >= 0

    # Constraints for absolute values of adjustments
    for key, adj_var in adjustments.items():
        problem += abs_adjustments[key] >= adj_var  # abs_adjustment >= adjustment
        problem += abs_adjustments[key] >= -adj_var  # abs_adjustment >= -adjustment

    # Solve the ILP
    problem.solve()

    print(count_table)
    # Check the solution status
    if LpStatus[problem.status] == "Optimal":
        # Apply adjustments to test_df based on the solution
        for (i, j), var in adjustments.items():
            count_table.loc[i, j] += var.value()
    else:
        print("No feasible solution found.")

    # Resulting adjusted DataFrame
    result_sum_row = count_table.sum(axis=1)
    result_sum_col = count_table.sum(axis=0)
    assert result_sum_row.equals(expected_sum_row)
    assert result_sum_col.equals(expected_sum_col)
    print(count_table)

    return None


def test_update_states():
    update_count_tables(test_df, test_diff)


test_update_states()