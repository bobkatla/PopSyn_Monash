import pandas as pd
import os
import glob


data_loc = "../data/tablebuilder"


def process_from_census_data(geo_lev='POA', normalise=True, return_tot=False):
    # This is simple to get the census data clean (assuming all shape the same, need to be quick)
    all_files =  glob.glob(os.path.join(data_loc , f"{geo_lev}*"))
    # remove header and footer from ABS
    total_df = pd.read_csv(f"{data_loc}/total_{geo_lev}.csv", skiprows=9, skipfooter=7, engine='python')
    total_df = total_df.dropna(axis=1, how='all')
    total_df.index = total_df.index.map(lambda r: r.replace(", VIC", ""))
    ls_df = [total_df]
    for f in all_files:
        df = pd.read_csv(f, skiprows=9, skipfooter=7, engine='python')
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, thresh=6)
        df = df[:-1]
        if "Total" in df.columns:
            df = df.drop(columns=["Total"])
        first_row = df.columns[0]
        df[first_row] = df[first_row].apply(lambda r: r.replace(", VIC", ""))
        df = df.set_index(first_row)
        df =df.add_prefix(f"{df.index.name}__")
        df.index.name = geo_lev
        ls_df.append(df)
    final_df = pd.concat(ls_df, axis=1)
    final_df = final_df.dropna(axis=0, thresh=10)

    # Normalisation
    tot_df = final_df[f"{geo_lev} (EN)"]
    if normalise:
        for col in final_df.columns:
            if col != f"{geo_lev} (EN)":
                final_df[col]= final_df[col].astype(float) / final_df[f"{geo_lev} (EN)"].astype(float)
    final_df = final_df.drop(columns=[f"{geo_lev} (EN)"])

    return (final_df, tot_df) if return_tot else final_df


def main():
    check = process_from_census_data(normalise=False)
    # check.to_csv("./checksing_first.csv", index=False)


if __name__ == "__main__":
    main()