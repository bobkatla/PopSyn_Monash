"""
As we will deal alot with data that have a lot of repetition in records (e.g. the pool and the population)

We need the tools to convert it to condensed form for faster processing
"""


# We will use a class so we can tell the diff from normal df (which is the normal form)
# Convert to the condensed format, and vice versa, sampling from it as well (using a condition), add and remove.
# However need to think about how to combine the pool and count idx, as we have the comparisons results
# Can just be an outside func, quite easy

import pandas as pd
from typing import List, Any, Tuple


class CondensedDF:
    id_col = "ids"
    count_col = "count"

    def __init__(self, original_records: pd.DataFrame) -> None:
        self.full_records = original_records.copy()
        self.all_cols = original_records.columns.tolist()

        assert self.id_col not in self.all_cols
        self.full_records.loc[:, self.id_col] = range(1, len(self.full_records) + 1) # give the id

        self.condense()

    def get_full_records(self) -> pd.DataFrame:
        return self.full_records.drop(columns=[self.id_col])
    
    def get_condensed(self) -> pd.DataFrame:
        return self.condensed_records.copy(deep=True)

    def condense(self) -> None:
        assert self.count_col not in self.all_cols
        condensed_records = self.full_records.groupby(self.all_cols)[self.id_col].apply(lambda x: list(x)).reset_index()
        condensed_records[self.count_col] = condensed_records[self.id_col].apply(lambda x: len(x))

        self.condensed_records = condensed_records

    def get_sub_records_by_ids(self, ids: List[int]):
        return self.full_records[self.full_records[self.id_col].isin(ids)].drop(columns=[self.id_col])

    def remove_identified_ids(self, ids: List[int]):
        self.full_records = self.full_records[~self.full_records[self.id_col].isin(ids)]
        self.condense()

    def add_new_records(self, new_records: pd.DataFrame):
        assert set(new_records.columns) == set(self.all_cols)
        max_id = self.full_records[self.id_col].max() if not self.full_records.empty else 1
        new_records[self.id_col] = range(max_id, len(new_records) + max_id)
        self.full_records = pd.concat([self.full_records, new_records])
        self.condense()

    def __str__(self) -> str:
        return str(self.condensed_records)


def filter_by_SAA_adjusted(src: CondensedDF, list_to_filter: List[Any], adjusted_atts: List[str]) -> Tuple[CondensedDF, pd.DataFrame]:
    assert set(adjusted_atts).issubset(set(src.all_cols))
    temp_ori = src.get_full_records().set_index(adjusted_atts)
    assert set(list_to_filter).issubset(set(temp_ori.index))
    remaining_list_comb = list(set(temp_ori.index) - set(list_to_filter))
    filtered_ori_records = temp_ori.loc[list_to_filter].reset_index()
    # Also return the remaining records that are not filtered
    unfiltered_df = temp_ori.loc[remaining_list_comb].reset_index()
    return CondensedDF(filtered_ori_records), unfiltered_df


def sample_from_condensed(src: CondensedDF, n:int) -> CondensedDF:
    condensed_records = src.get_condensed()
    sample_ids = sample_ids_use_ids_count(condensed_records, n, src.id_col, src.count_col)
    full_sample_records = src.get_full_records().loc[sample_ids]
    return CondensedDF(full_sample_records)


def sample_ids_use_ids_count(condensed_records: pd.DataFrame, n: int, id_col: str, count_col: str) -> List[int]:
    only_id_and_count = condensed_records[[id_col, id_col]]
    only_id_and_count = only_id_and_count.explode(id_col)
    sample_ids = only_id_and_count.sample(n=n, weights=count_col, replace=False)[id_col].tolist()
    return sample_ids