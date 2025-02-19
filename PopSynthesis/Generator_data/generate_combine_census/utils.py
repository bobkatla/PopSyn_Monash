import pandas as pd
import os
import glob
import numpy as np


data_loc = "../data/tablebuilder"


def process_from_census_data(geo_lev="POA"):
    # This is simple to get the census data clean (assuming all shape the same, need to be quick)
    all_files = glob.glob(os.path.join(data_loc, f"{geo_lev}*"))
    # remove header and footer from ABS
    total_hh_df = pd.read_csv(
        f"{data_loc}/total_hh_{geo_lev}.csv", skiprows=9, skipfooter=7, engine="python"
    )
    total_hh_df = total_hh_df.dropna(axis=1, how="all")
    total_hh_df.index = total_hh_df.index.map(lambda r: r.replace(", VIC", ""))
    total_hh_df = total_hh_df.add_prefix("Dwelling_")

    total_pp_df = pd.read_csv(
        f"{data_loc}/total_pp_{geo_lev}.csv", skiprows=9, skipfooter=7, engine="python"
    )
    total_pp_df = total_pp_df.dropna(axis=1, how="all")
    total_pp_df.index = total_pp_df.index.map(lambda r: r.replace(", VIC", ""))
    total_pp_df = total_pp_df.add_prefix("Person_")

    ls_df = [total_pp_df, total_hh_df]
    for f in all_files:
        df_metadata = pd.read_csv(f, nrows=3)
        type_count = df_metadata.iat[2, 0].split(": ")[1].split(" ")[0]
        df = pd.read_csv(f, skiprows=9, skipfooter=7, engine="python")
        df = df.dropna(axis=1, how="all")
        df = df.dropna(axis=0, how="all")
        df = df[1:-1]  # last row is total, first row is just geo, so drop
        if "Total" in df.columns:
            df = df.drop(columns=["Total"])
        first_row = df.columns[0]
        df[first_row] = df[first_row].apply(lambda r: r.replace(", VIC", ""))
        df = df.set_index(first_row)
        df = df.add_prefix(f"{type_count}_{df.index.name}__")
        df.index.name = geo_lev
        ls_df.append(df)
    final_df = pd.concat(ls_df, axis=1)
    final_df = final_df.dropna(axis=0, thresh=10)
    return final_df


def TRS(initial_weights: list[float], know_tot: int = None):
    # Desired total constraint
    desired_total = know_tot if know_tot else sum(initial_weights)

    # Step 1: Truncate the weights to integers
    truncated_weights = np.floor(initial_weights).astype(int)

    # Step 2: Calculate the discrepancy
    total_truncated = np.sum(truncated_weights)
    discrepancy = desired_total - total_truncated

    # Step 3: Replicate individuals to match the constraint
    if discrepancy > 0:
        # Calculate fractional parts
        fractional_parts = initial_weights - truncated_weights

        # Replicate individuals in proportion to their fractional parts
        replication_probs = (
            fractional_parts / np.sum(fractional_parts)
            if np.sum(fractional_parts) != 0
            else fractional_parts
        )
        num_replications = np.random.multinomial(int(discrepancy), replication_probs)
        truncated_weights += num_replications

    # Step 4: Sample individuals if there is an excess
    if discrepancy < 0:
        excess_indices = np.where(truncated_weights > 0)[0]
        excess_weights = truncated_weights[excess_indices]

        # Calculate sampling probabilities based on truncated weights
        sampling_probs = (
            excess_weights / np.sum(excess_weights)
            if np.sum(excess_weights) != 0
            else excess_weights
        )

        # Randomly sample individuals to reduce excess
        num_samples = np.random.multinomial(abs(discrepancy), sampling_probs)
        truncated_weights[excess_indices] -= num_samples

    # Step 5: Your final truncated and rounded integer weights
    return truncated_weights


def update_int_all(sub_df, total_seri):
    cols = sub_df.columns

    def f(r):
        vals = list(r)
        poa = r.name
        tot = total_seri[poa]
        return TRS(vals, tot)

    sub_df[cols] = sub_df.apply(f, axis=1, result_type="expand")
    return sub_df


def main():
    check = process_from_census_data()
    print(check)
    # check.to_csv("./checksing_first.csv", index=False)


if __name__ == "__main__":
    main()
