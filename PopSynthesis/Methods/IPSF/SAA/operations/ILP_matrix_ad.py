"""Testing the updates given a formatted counts table"""
import pandas as pd
from pulp import LpProblem, LpVariable, lpSum, LpStatus, LpMinimize


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

    # TODO: Add filtering to remove rows and columns that cannot be adjusted

    count_table, adjustment_remaining = _ILP_solving_adjustment(count_table, states_diff)

    # Resulting adjusted DataFrame
    result_sum_row = count_table.sum(axis=1)
    result_sum_col = count_table.sum(axis=0) + adjustment_remaining
    assert result_sum_row.equals(expected_sum_row)
    assert result_sum_col.equals(expected_sum_col)

    err_remaining = (adjustment_remaining ** 2).sum()

    return count_table, err_remaining

