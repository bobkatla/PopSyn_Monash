"""
As we will deal alot with data that have a lot of repetition in records (e.g. the pool and the population)

We need the tools to convert it to condensed form for faster processing
"""


# We will use a class so we can tell the diff from normal df (which is the normal form)
# Convert to the condensed format, and vice versa, sampling from it as well (using a condition), add and remove.
# However need to think about how to combine the pool and count idx, as we have the comparisons results
# Can just be an outside func, quite easy

import pandas as pd
import numpy as np


class CondensedDF:
    id_col = "ids"
    count_col = "count"

    def __init__(self, original_records: pd.DataFrame) -> None:
        self.original_records = original_records
        self.all_cols = original_records.columns.tolist()

        assert self.id_col not in self.all_cols
        self.original_records[self.id_col] = range(1, len(self.original_records) + 1) # give the id

        self.condense()

    def get_original(self) -> pd.DataFrame:
        return self.original_records.drop(columns=[self.id_col])
    
    def get_condensed(self) -> pd.DataFrame:
        return self.condensed_records

    def condense(self) -> None:
        assert self.count_col not in self.all_cols
        condensed_records = self.original_records.groupby(self.all_cols)[self.id_col].apply(lambda x: list(x)).reset_index()
        condensed_records[self.count_col] = condensed_records[self.id_col].apply(lambda x: len(x))

        self.condensed_records = condensed_records

    def __str__(self) -> str:
        return str(self.condensed_records)


def sample_from_condensed(src: CondensedDF, n:int) -> CondensedDF:
    condensed_records = src.get_condensed()
    only_id_and_count = condensed_records[[src.id_col, src.count_col]]
    only_id_and_count = only_id_and_count.explode(src.id_col)
    sample_ids = only_id_and_count.sample(n=n, weights=src.count_col, replace=False)[src.id_col]
    full_sample_records = src.get_original().loc[sample_ids]
    return CondensedDF(full_sample_records)