"""Tools to condense the disaggregated data into a condensed format and vice versa."""

import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field
from typing import Union


def condense_df(df: Union[pl.DataFrame, pd.DataFrame], id_col: str) -> pl.DataFrame:
    """Convert the disaggregated data into a condensed format."""
    assert id_col in df.columns
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    return df.groupby(pl.exclude(id_col)).agg(pl.col(id_col), pl.len().alias(count_field))


def explode_df(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Convert the condensed data back to the disaggregated format."""
    assert id_col in df.columns
    assert count_field in df.columns
    df = df.drop(count_field)
    return df.explode(id_col)

