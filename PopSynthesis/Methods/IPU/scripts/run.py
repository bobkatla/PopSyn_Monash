# from synthpop.recipes.tests import test_starter
import synthpop.zone_synthesizer as zs

import time

from pathlib import Path
ipu_dir_path = Path(__file__).parent.parent.resolve()
data_path = Path(ipu_dir_path / "data")
output_path = Path(ipu_dir_path / "output")
assert data_path.exists()
assert output_path.exists()


def test_run(hh_marg, p_marg, hh_sample, p_sample):
    hh_marg, p_marg, hh_sample, p_sample, xwalk = zs.load_data(
        hh_marg, p_marg, hh_sample, p_sample
    )

    all_households, all_persons, all_stats = zs.synthesize_all_zones(
        hh_marg, p_marg, hh_sample, p_sample, xwalk
    )

    print(all_households)
    print(all_persons)
    print(all_stats)
    
    all_households.to_csv(output_path / "syn_hh_ipu.csv", index=False)
    all_persons.to_csv(output_path / "syn_pp_ipu.csv", index=False)
    all_stats.to_csv(output_path / "stats_ipu.csv", index=False)


def main():
    name_f = lambda x: data_path / f"{x}.csv"
    hh_marg = name_f("hh_marginals_ipu")
    p_marg = name_f("person_marginals_ipu")
    hh_sample = name_f("hh_sample_ipu")
    p_sample = name_f("pp_sample_ipu")
    
    start_time = time.time()
    test_run(hh_marg, p_marg, hh_sample, p_sample)
    elapsed_time = time.time() - start_time

    print("--- %s seconds ---" % elapsed_time)
    print("--- %s minutes ---" % (elapsed_time / 60))
    print("--- %s hours ---" % (elapsed_time / 360))


if __name__ == "__main__":
    main()
