"""Main place to run SAA for households synthesis"""

from PopSynthesis.Methods.IPSF.const import (
    output_dir,
    SAA_ODERED_ATTS_PP,
    CONSIDERED_ATTS_PP,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import (
    saa_run,
    get_pp_data,
)
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
import time
import pandas as pd


def run_main() -> None:
    pp_marg, pp_pool = get_pp_data()
    seed_data = pd.read_csv(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\PopSynthesis\Methods\IPSF\data\raw\pp_sample_ipu.csv")
    seed_data = seed_data.drop(columns=["serialno", "sample_geog"])
    condensed_hh_pool = condense_df(seed_data.astype(str))

    start_time = time.time()
    # saa run
    final_syn_hh, err_rm = saa_run(
        pp_marg,
        condensed_hh_pool,
        considered_atts=CONSIDERED_ATTS_PP,
        ordered_to_adjust_atts=SAA_ODERED_ATTS_PP,
        max_run_time=15,
        extra_rm_frac=0.3,
        last_adjustment_order=SAA_ODERED_ATTS_PP,
        output_each_step=False,
        add_name_for_step_output="test",
        include_zero_cell_values=False,
        # randomly_add_last=["hhinc"],
    )

    # record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
    print(f"Processing took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
    print(f"Error pp rm are: {err_rm}")

    output_file_name = "SAA_PP_seed_test.csv"
    print(f"outputting to this file: {output_dir / output_file_name}")
    final_syn_hh.write_csv(output_dir / output_file_name)


if __name__ == "__main__":
    run_main()
