"""Testing the updates given a formatted counts table"""


import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpMinimize
from pathlib import Path

# simple pass case
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

# Case of no absoluate solution
noabs_test_df = pd.DataFrame(
    {
        "s1": [5, 3],
        "s2": [4, 1],
        "s3": [6, 2],
    },
    index=["a", "b"]
)
noabs_test_diff = pd.Series([-10, 10, 0], index=["s1", "s2", "s3"])

########### large data ###########
data_folder = Path(__file__).parent.parent.parent.resolve() / "test_data" / "IPL"
large_count_table = pd.read_csv(data_folder / "large_count_table.csv")
large_states_diff = pd.read_csv(data_folder / "large_states_diff.csv").iloc[0]


def _ILP_solving_adjustment(count_table: pd.DataFrame, states_diff: pd.Series) -> pd.DataFrame:
    count_table = count_table.copy()
    # Initialize the ILP problem
    problem = LpProblem("MatrixAdjustment", LpMinimize)

    # Initialize adjustment variables
    adjustments = {(i, j): LpVariable(f"A_{i}_{j}", lowBound=-count_table.loc[i, j], cat="Interger") 
                for i in count_table.index for j in count_table.columns 
                if pd.notnull(count_table.loc[i, j])}

    # Initialize slack variables for each column to allow deviations from target adjustments
    slack_pos = {j: LpVariable(f"slack_pos_{j}", lowBound=0, cat="Continuous") for j in states_diff.index}
    slack_neg = {j: LpVariable(f"slack_neg_{j}", lowBound=0, cat="Continuous") for j in states_diff.index}

    # Deviation variables to penalize large deviations from original values
    deviations = {key: LpVariable(f"dev_{key[0]}_{key[1]}", lowBound=0, cat="Continuous") 
                for key in adjustments}

    # Objective: Minimize the sum of deviations from original values and slack
    problem += lpSum(deviations[key] + slack_pos[j] + slack_neg[j] for key in deviations for j in states_diff.index)

    # Row constraints: Ensure the sum of adjustments in each row is zero
    for i in count_table.index:
        row_adjustments = [adjustments[(i, j)] for j in count_table.columns if (i, j) in adjustments]
        problem += lpSum(row_adjustments) == 0

    # Column constraints with positive and negative slack for deviations
    for j, target_diff in states_diff.items():
        col_adjustments = [adjustments[(i, j)] for i in count_table.index if (i, j) in adjustments]
        problem += lpSum(col_adjustments) == target_diff + slack_pos[j] - slack_neg[j]

    # Non-negativity constraints: Ensure each adjusted cell remains positive
    for (i, j), var in adjustments.items():
        original_value = count_table.loc[i, j]
        problem += var >= -original_value  # Ensures X_ij + A_ij >= 0

    # Deviation constraints: Ensure deviations represent the absolute change from the original value
    for (i, j), adj_var in adjustments.items():
        original_value = count_table.loc[i, j]
        problem += deviations[(i, j)] >= adj_var               # dev >= adj
        problem += deviations[(i, j)] >= -adj_var              # dev >= -adj
        problem += deviations[(i, j)] >= original_value - (original_value + adj_var)  # maintain closeness to original values

    # Solve the ILP
    problem.solve()

    adjustment_remaining = None
    # Check the solution status
    if LpStatus[problem.status] == "Optimal":
        # Apply adjustments to count_table based on the solution
        for (i, j), var in adjustments.items():
            count_table.loc[i, j] += var.value()
        
        # Calculate the resulting actual column adjustments and print the achieved diff
        actual_diff = {j: sum(adjustments[(i, j)].value() for i in count_table.index if (i, j) in adjustments)
                    for j in states_diff.index}
        adjustment_remaining = pd.Series({j: int(states_diff[j] - actual_diff[j]) for j in states_diff.index})
    else:
        print("No feasible solution found.")

    # Resulting adjusted DataFrame
    
    return count_table, adjustment_remaining


def update_count_tables(count_table: pd.DataFrame, states_diff: pd.Series) -> pd.DataFrame:
    """Temp give the func here for debugging"""
    assert sum(states_diff) == 0
    expected_sum_row = count_table.sum(axis=1)
    expected_sum_col = count_table.sum(axis=0) + states_diff

    count_table, adjustment_remaining = _ILP_solving_adjustment(count_table, states_diff)

    # Resulting adjusted DataFrame
    result_sum_row = count_table.sum(axis=1)
    result_sum_col = count_table.sum(axis=0) + adjustment_remaining
    assert result_sum_row.equals(expected_sum_row)
    assert result_sum_col.equals(expected_sum_col)

    err_remaining = (adjustment_remaining ** 2).sum()

    return count_table, err_remaining


def test_update_states():
    a, b = update_count_tables(test_df, test_diff)
    # a, b = update_count_tables(noabs_test_df, noabs_test_diff)
    # a, b = update_count_tables(large_count_table, large_states_diff)
    print(a)
    print(b)

test_update_states()