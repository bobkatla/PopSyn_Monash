"""Main place to run SAA for households synthesis"""

from PopSynthesis.Methods.IPSF.const import output_dir
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run, get_test_hh
import time


def run_main() -> None:
    # How long: 0h-0m-0.00
    hh_marg, hh_pool = get_test_hh()

    start_time = time.time()
    # saa run
    final_syn_hh, err_rm = saa_run(hh_marg, hh_pool, max_run_time=5)

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
