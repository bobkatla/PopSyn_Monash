import polars as pl

from PopSynthesis.Methods.IPSF.const import count_field

# Consider the BN from another place


def create_pool(pool_sz: int, convert_to_count=True) -> pl.DataFrame:
    # We will have the BN learning here
    # Forward sampling
    # Convert to count, using special field count
    NotImplemented
