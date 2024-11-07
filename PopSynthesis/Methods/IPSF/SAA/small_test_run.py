"""Main place to run SAA for households synthesis"""

from PopSynthesis.Methods.IPSF.const import SAA_ODERED_ATTS_HH, CONSIDERED_ATTS_HH
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run, get_test_hh
import time


def run_main() -> None:
    # Processing took 0h-1m-2.17s
    # Error hh rm are: [48928.0, 25119, 18762, 12693, 10850]
    hh_marg, hh_pool = get_test_hh()

    start_time = time.time()
    # saa run
    final_syn_hh, err_rm = saa_run(hh_marg, hh_pool, considered_atts=CONSIDERED_ATTS_HH, ordered_to_adjust_atts=SAA_ODERED_ATTS_HH, max_run_time=5)

    # record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
    print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
    print(f"Error hh rm are: {err_rm}")
    # output
    print(final_syn_hh)


if __name__ == "__main__":
    run_main()
