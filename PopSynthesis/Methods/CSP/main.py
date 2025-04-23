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
    hh_df = pd.read_csv(DATA_FOLDER / "new_IPF_results_wo_zerocell.csv")
    configs = load_configurations()
    # Run CSP with the given hh_df and configs
    resulted_pp = run_csp(hh_df, configs, True, True) # We must not change the hh
    resulted_pp.to_csv(OUTPUT_FOLDER / "csp_results_seed_eachz.csv", index=False)

if __name__ == "__main__":
    main()