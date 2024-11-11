"""Testing the updates given a formatted counts table"""


import pandas as pd
from PopSynthesis.Methods.IPSF.SAA.operations.ILP_matrix_ad import update_count_tables
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


def test_update_states():
    a, b = update_count_tables(test_df, test_diff)
    # a, b = update_count_tables(noabs_test_df, noabs_test_diff)
    # a, b = update_count_tables(large_count_table, large_states_diff)
    print(a)
    print(b)

test_update_states()