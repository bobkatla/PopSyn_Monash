"""Tools to condense the disaggregated data into a condensed format and vice versa."""

import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field
from typing import Union


def condense_df(df: Union[pl.DataFrame, pd.DataFrame], id_col: Union[None, str]= None) -> pl.DataFrame:
    """Convert the disaggregated data into a condensed format."""
    if isinstance(df, pd.DataFrame):
        df = pl.from_pandas(df)
    expr = pl.all()
    to_agg = [pl.len().alias(count_field)]
    if id_col is not None:
        assert id_col in df.columns
        expr = pl.exclude(id_col)
        to_agg.append(pl.col(id_col))
    return df.group_by(expr).agg(*to_agg)


def explode_df_by_id(df: pl.DataFrame, id_col: str) -> pl.DataFrame:
    """Convert the condensed data back to the disaggregated format."""
    assert id_col in df.columns
    assert count_field in df.columns
    df = df.drop(count_field)
    return df.explode(id_col)
