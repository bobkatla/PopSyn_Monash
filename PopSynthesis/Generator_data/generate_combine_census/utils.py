import pandas as pd
import os
import glob


data_loc = "../data/tablebuilder"


def process_from_census_data(geo_lev='POA'):
    # This is simple to get the census data clean (assuming all shape the same, need to be quick)
    all_files =  glob.glob(os.path.join(data_loc , f"{geo_lev}*"))
    # remove header and footer from ABS
    total_hh_df = pd.read_csv(f"{data_loc}/total_hh_{geo_lev}.csv", skiprows=9, skipfooter=7, engine='python')
    total_hh_df = total_hh_df.dropna(axis=1, how='all')
    total_hh_df.index = total_hh_df.index.map(lambda r: r.replace(", VIC", ""))
    total_hh_df = total_hh_df.add_prefix(f"Dwelling__")

    total_pp_df = pd.read_csv(f"{data_loc}/total_pp_{geo_lev}.csv", skiprows=9, skipfooter=7, engine='python')
    total_pp_df = total_pp_df.dropna(axis=1, how='all')
    total_pp_df.index = total_pp_df.index.map(lambda r: r.replace(", VIC", ""))
    total_pp_df = total_pp_df.add_prefix(f"Person__")

    ls_df = [total_pp_df, total_hh_df]
    for f in all_files:
        df_metadata = pd.read_csv(f, nrows=3)
        type_count = df_metadata.iat[2, 0].split(": ")[1].split(" ")[0]
        df = pd.read_csv(f, skiprows=9, skipfooter=7, engine='python')
        df = df.dropna(axis=1, how='all')
        df = df.dropna(axis=0, thresh=6)
        df = df[:-1]
        if "Total" in df.columns:
            df = df.drop(columns=["Total"])
        first_row = df.columns[0]
        df[first_row] = df[first_row].apply(lambda r: r.replace(", VIC", ""))
        df = df.set_index(first_row)
        df =df.add_prefix(f"{type_count}_{df.index.name}__")
        df.index.name = geo_lev
        ls_df.append(df)
        break
    final_df = pd.concat(ls_df, axis=1)
    final_df = final_df.dropna(axis=0, thresh=10)
    return final_df


def main():
    check = process_from_census_data()
    print(check)
    # check.to_csv("./checksing_first.csv", index=False)


if __name__ == "__main__":
    main()