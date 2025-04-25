"""Main run func to run the CSP with a given hh df"""


import pandas as pd
from PopSynthesis.Methods.CSP.run.run import run_csp
from PopSynthesis.Methods.CSP.const import (
    DATA_FOLDER,
    OUTPUT_FOLDER,
    HHID,
    ZONE_ID,
    HH_SZ,
    HH_ATTS,
    RELATIONSHIP,
    PP_ATTS,
    hh_seed_file_path,
    pp_seed_file_path,
)


def inflate_based_on_total(df, target_col: str) -> pd.DataFrame:
    assert target_col in df.columns, "The dataframe must contain the target column"
    # Repeat rows based on column values
    df_repeated = df.loc[df.index.repeat(df[target_col])].reset_index(drop=True)

    # Drop the column
    df_repeated = df_repeated.drop(columns=target_col)
    return df_repeated


def load_configurations():
    """from const file, create a config dict"""
    config = {
        "hh_seed": pd.read_csv(hh_seed_file_path)[HH_ATTS + [HHID, HH_SZ]],
        "pp_seed": pd.read_csv(pp_seed_file_path)[PP_ATTS + [HHID, RELATIONSHIP]],
        "hhid": HHID,
        "zone_id": ZONE_ID,
        "hh_size": HH_SZ,
        "relationship": RELATIONSHIP,
        "output_folder": OUTPUT_FOLDER,
    }
    return config


def main():
    # hh_df = pd.read_csv(DATA_FOLDER / "new_IPF_results_wo_zerocell.csv")
    # hh_df = inflate_based_on_total(hh_df, "total")
    # # add hhid
    # hh_df[HHID] = hh_df.reset_index(drop=True).index + 1
    # hh_df.to_csv(OUTPUT_FOLDER / "syn_hh_ipf.csv", index=False)

    # hh_df = pd.read_csv(OUTPUT_FOLDER / "syn_hh_ipf.csv")
    # hh_df = pd.read_csv(DATA_FOLDER / "pure_BN_HH.csv")
    hh_df = pd.read_csv(DATA_FOLDER / "hh_pureBN_hhtype_filter.csv")
    configs = load_configurations()
    # Run CSP with the given hh_df and configs
    resulted_pp = run_csp(hh_df, configs, False, True, True, True) # We must not change the hh
    # resulted_pp.to_csv(OUTPUT_FOLDER / "csp_BN_from_IPF.csv", index=False)
    # resulted_pp.to_csv(OUTPUT_FOLDER / "csp_BN_from_BN.csv", index=False)
    resulted_pp.to_csv(OUTPUT_FOLDER / "csp_BN_direct_hhtype.csv", index=False)

    # resulted_pp = run_csp(hh_df, configs, True, True, False) # We must not change the hh
    # resulted_pp.to_csv(OUTPUT_FOLDER / "csp_results_eachz.csv", index=False)

    # resulted_pp = run_csp(hh_df, configs, False, True, False) # We must not change the hh
    # resulted_pp.to_csv(OUTPUT_FOLDER / "csp_results_allz.csv", index=False)

if __name__ == "__main__":
    main()