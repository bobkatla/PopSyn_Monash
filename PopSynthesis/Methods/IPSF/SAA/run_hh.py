"""Main place to run SAA for households synthesis"""

from PopSynthesis.Methods.IPSF.const import (
    output_dir,
    SAA_ODERED_ATTS_HH,
    CONSIDERED_ATTS_HH,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import (
    saa_run,
    get_hh_data,
)
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import time


def run_main() -> None:
    hh_marg, hh_pool = get_hh_data()
    condensed_hh_pool = condense_df(hh_pool)

    start_time = time.time()
    # saa run
    final_syn_hh, err_rm = saa_run(
        hh_marg,
        condensed_hh_pool,
        considered_atts=CONSIDERED_ATTS_HH,
        ordered_to_adjust_atts=SAA_ODERED_ATTS_HH,
        max_run_time=15,
        shuffle_order=True,
    )

    # record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
    print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
    print(f"Error hh rm are: {err_rm}")
    # output
    final_syn_hh.write_csv(output_dir / "SAA_HH_IPL_abs_no_order_no_penal.csv")


if __name__ == "__main__":
    run_main()
