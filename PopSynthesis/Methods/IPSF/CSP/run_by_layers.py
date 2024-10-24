"""Run the CSP from a given syn HH (also SAA to adjust again)"""


from PopSynthesis.Methods.IPSF.const import output_dir
from PopSynthesis.Methods.IPSF.CSP.common_run_funcs import ipsf_full_loop, get_cross_checked_data
import time


# For SAA
order_adjustment = [
        "hhsize",
        "hhinc",
        "totalvehs",
        "dwelltype",
        "owndwell",
    ]  # these must exist in both marg and syn


def main():
    # TODO: is there anyway to use pp marg
    syn_hh, hh_pool, hh_marg, pools_ref = get_cross_checked_data()

    start_time = time.time()
    ##
    final_syn_hh, final_syn_pp, err_rm, cannot_assign_hh = ipsf_full_loop(order_adjustment=order_adjustment, syn_hh=syn_hh, hh_pool=hh_pool, hh_marg=hh_marg, pools_ref=pools_ref, max_run_time=5)
    ##

    # record time
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)  # 3600 seconds in an hour
    minutes, seconds = divmod(rem, 60)  # 60 seconds in a minute
    print(f"IPSF took {int(hours)}h-{int(minutes)}m-{seconds:.2f}s")
    print(f"Error hh rm are: {err_rm}")

    # output
    final_syn_hh.to_csv(output_dir / "IPSF_HH.csv", index=False)
    final_syn_pp.to_csv(output_dir / "IPSF_PP.csv", index=False)
    if cannot_assign_hh is not None:
        cannot_assign_hh.to_csv(output_dir / "IPSF_cannot_assign_hh.csv", index=False)
   

if __name__ == "__main__":
    main()
