import polars as pl


def convert_hh_totvehs(hh_df: pl.DataFrame, veh_limit=4):
    # Define the conditional operation
    def convert_veh(col, veh_limit):
        return pl.when(col < veh_limit).then(col.cast(pl.Utf8)).otherwise(pl.lit(f"{veh_limit}+"))

    hh_df = hh_df.with_columns(
        convert_veh(pl.col("totalvehs"), veh_limit).alias("totalvehs")
    )
    return hh_df


def convert_hh_inc(hh_df, check_states):
    # Note there can be null
    hhinc_col = pl.col("hhinc")
    
    # Base expression
    expr = pl.when(hhinc_col < 0).then(pl.lit("Negative income"))
    expr = expr.when(hhinc_col == 0).then(pl.lit("Nil income"))
    
    # Generate conditions and results for each state in check_states
    for state in check_states:
        state_clean = state.replace(",", "").replace("$", "").split(" ")[0]
        if "more" in state:
            val = int(state_clean)
            expr = expr.when(hhinc_col >= val).then(pl.lit(f"{val}+"))
        elif "-" in state:
            a, b = map(int, state_clean.split("-"))
            expr = expr.when((hhinc_col >= a) & (hhinc_col <= b)).then(pl.lit(f"{a}-{b}"))
        else:
            raise ValueError(f"Dunno I never seen this lol {state}")
    
    # Final otherwise to retain the original value if no conditions match
    expr = expr.otherwise(hhinc_col)
    
    # Apply the transformation
    hh_df = hh_df.with_columns(expr.alias("hhinc"))

    return hh_df
    

def convert_hh_dwell(hh_df: pl.DataFrame):  # Removing the occupied rent free
    col_owndwell = pl.col("owndwell")
    expr = pl.when(col_owndwell=="Occupied Rent-Free").then(pl.lit("Something Else")).otherwise(col_owndwell)
    hh_df = hh_df.with_columns(expr.alias("owndwell"))
    return hh_df


def convert_hh_size(hh_df):
    col_hhsz = pl.col("hhsize")
    max_hhsz = 8 # const, based on census
    expr = pl.when(col_hhsz >= max_hhsz).then(pl.lit(f"{max_hhsz}+")).otherwise(col_hhsz.cast(pl.Utf8))
    hh_df = hh_df.with_columns(expr.alias("hhsize"))
    return hh_df
