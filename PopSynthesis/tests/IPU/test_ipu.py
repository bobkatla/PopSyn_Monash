# from synthpop.recipes.tests import test_starter
import synthpop.zone_synthesizer as zs

import time

from pathlib import Path
data_path = Path(__file__).parent.parent.resolve() / "test_data"
expected_folder = None


def test_ipu():
    name_f = lambda x: data_path / f"{x}.csv"
    hh_marg = name_f("hh_marg_minor_test")
    p_marg = name_f("person_marg_minor_test")
    hh_sample = name_f("hh_sample_minors")
    p_sample = name_f("pp_sample_minors")
    
    start_time = time.time()

    hh_marg, p_marg, hh_sample, p_sample, xwalk = zs.load_data(
        hh_marg, p_marg, hh_sample, p_sample
    )
    all_households, all_persons, all_stats = zs.synthesize_all_zones(
        hh_marg, p_marg, hh_sample, p_sample, xwalk
    )

    elapsed_time = time.time() - start_time

    print(all_households)
    print(all_persons)
    print(all_stats)

    print("--- %s seconds ---" % elapsed_time)
    print("--- %s minutes ---" % (elapsed_time / 60))
    print("--- %s hours ---" % (elapsed_time / 360))

test_ipu()
