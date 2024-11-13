import polars as pl
from PopSynthesis.Methods.IPSF.utils.ILP_matrix_ad import update_count_tables
from pathlib import Path

# Simple pass case with polars, including row labels as a new column 'row_id'
test_df = pl.DataFrame(
    {
        "row_id": ["a", "b", "c", "d", "e", "f"],
        "s1": [None, 1, 2, 2, 33, 3],
        "s2": [None, 32, 12, 2, 0, None],
        "s3": [10, 20, 30, 40, 50, 60],
        "s4": [0, 34, None, 34, 50, 60],
        "s5": [10, 2, None, 123, 3, 432],
    }
)
test_diff = dict(zip(test_df.select(pl.exclude("row_id")).columns, [-3, -8, 0, 5, 6]))

# Case of no absolute solution with polars, including row labels as a new column 'row_id'
noabs_test_df = pl.DataFrame(
    {
        "row_id": ["a", "b"],
        "s1": [5, 3],
        "s2": [4, 1],
        "s3": [6, 2],
    }
)
noabs_test_diff = dict(zip(noabs_test_df.select(pl.exclude("row_id")).columns, [-10, 10, 0]))

########### large data ###########
data_folder = Path(__file__).parent.parent.parent.resolve() / "test_data" / "IPL"
large_count_table = pl.read_csv(data_folder / "large_count_table.csv").with_row_index(name="row_id")
large_states_diff = pl.read_csv(data_folder / "large_states_diff.csv").row(0, named=True)

# Add a 'row_id' column to maintain unique identifiers if needed
# large_count_table = large_count_table.with_row_count(name="row_id")

def test_update_states():
    # Test the update on each case by uncommenting as needed
    id_col = "row_id"
    a, b = update_count_tables(test_df, test_diff, id_col, deviation_type="relative")
    # a, b = update_count_tables(noabs_test_df, noabs_test_diff, id_col)
    # a, b = update_count_tables(large_count_table, large_states_diff, id_col)
    print(a)
    print(b)

test_update_states()
