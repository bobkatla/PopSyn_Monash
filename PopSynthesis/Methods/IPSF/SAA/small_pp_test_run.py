"""Main place to run SAA for households synthesis"""

from PopSynthesis.Methods.IPSF.const import SAA_ODERED_ATTS_PP, CONSIDERED_ATTS_PP
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import (
    saa_run,
    get_test_pp,
)
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import time
from pathlib import Path


def run_main() -> None:
    pp_marg, pp_pool = get_test_pp()
    condensed_hh_pool = condense_df(pp_pool.astype(str))
    extra_rm_frac = 0
    for i in range(1):
        start_time = time.time()
        # saa run
        final_syn_pp, err_rm = saa_run(
            pp_marg,
            condensed_hh_pool,
            considered_atts=CONSIDERED_ATTS_PP,
            ordered_to_adjust_atts=SAA_ODERED_ATTS_PP,
            max_run_time=5,
            extra_rm_frac=extra_rm_frac,
            last_adjustment_order=["age"],
        )

        # record time
        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
        minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
        print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
        print(f"Number of aimed to synthesis hh: {err_rm}")
        print(final_syn_pp)
        # output
        output_dir = Path(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\IPSF\data\small_test_output_check")
        output_file_name = "pp_test.csv"
        print(f"outputting to this file: {output_dir / output_file_name}")
        final_syn_pp.write_csv(output_dir / output_file_name)

if __name__ == "__main__":
    run_main()
