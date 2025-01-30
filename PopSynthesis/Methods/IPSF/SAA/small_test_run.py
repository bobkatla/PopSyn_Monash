"""Main place to run SAA for households synthesis"""

from PopSynthesis.Methods.IPSF.const import SAA_ODERED_ATTS_HH, CONSIDERED_ATTS_HH
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import (
    saa_run,
    get_test_hh,
)
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import time


def run_main() -> None:
    for i in range(10):
        hh_marg, hh_pool = get_test_hh()
        condensed_hh_pool = condense_df(hh_pool)
        start_time = time.time()
        extra_rm_frac = 0
        # saa run
        final_syn_hh, err_rm = saa_run(
            hh_marg,
            condensed_hh_pool,
            considered_atts=CONSIDERED_ATTS_HH,
            ordered_to_adjust_atts=SAA_ODERED_ATTS_HH,
            max_run_time=5,
            extra_rm_frac=extra_rm_frac,
            shuffle_order=[],
        )

        # record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
        minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
        print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
        print(f"Number of aimed to synthesis hh: {err_rm}")
        # output
        final_syn_hh.write_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\IPSF\data\small_test_output_check\penalty" + "\\" + f"smallpenal_{extra_rm_frac}_{minutes*60 + seconds:.2f}_{i}.csv")


if __name__ == "__main__":
    run_main()
