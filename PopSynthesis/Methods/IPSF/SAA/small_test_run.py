"""Main place to run SAA for households synthesis"""

from PopSynthesis.Methods.IPSF.const import SAA_ODERED_ATTS_HH, CONSIDERED_ATTS_HH
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import (
    saa_run,
    get_test_hh,
)
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import time
import pandas as pd
from pathlib import Path


def run_main() -> None:
    hh_zero_cells = pd.read_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\IPSF\data\small_test\HH_pool_small_test_zerocell.csv")
    hh_zero_cells = hh_zero_cells.drop(columns=["serialno", "sample_geog"])
    hh_marg, hh_pool = get_test_hh()
    condensed_hh_pool = condense_df(hh_zero_cells)
    extra_rm_frac = 0
    for i in range(1):
        start_time = time.time()
        # saa run
        final_syn_hh, err_rm = saa_run(
            hh_marg,
            condensed_hh_pool,
            considered_atts=CONSIDERED_ATTS_HH,
            ordered_to_adjust_atts=SAA_ODERED_ATTS_HH,
            max_run_time=10,
            extra_rm_frac=extra_rm_frac,
            last_adjustment_order=["hhsize"],
            randomly_add_last=["hhinc"],
        )

        # record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
        minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
        print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
        print(f"Number of aimed to synthesis hh: {err_rm}")
        print(final_syn_hh)
        # output
        # output, hhsize should be perfect??
        output_dir = Path(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\IPSF\data\small_test_output_check")
        output_file_name = "test.csv"
        print(f"outputting to this file: {output_dir / output_file_name}")
        final_syn_hh.write_csv(output_dir / output_file_name)
        # final_syn_hh.write_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\IPSF\data\small_test_output_check" + "\\" + f"smallpenal_{extra_rm_frac}_{minutes*60 + seconds:.2f}_{i}.csv")
        # final_syn_hh.write_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\IPSF\data\small_test_output_check" + "\\" + f"oriseed_check.csv")


if __name__ == "__main__":
    run_main()
