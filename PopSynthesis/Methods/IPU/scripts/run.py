# from synthpop.recipes.tests import test_starter
import synthpop.zone_synthesizer as zs
import pathlib

import os
import time


def test_run(hh_marg, p_marg, hh_sample, p_sample):
    hh_marg, p_marg, hh_sample, p_sample, xwalk = zs.load_data(
        hh_marg, p_marg, hh_sample, p_sample
    )
    # a = zs.synch_hhids(hh_marg, p_marg, xwalk)
    # sample_df = hh_sample[hh_sample.sample_geog == xwalk[0][1]]
    # category_df = cat.category_combinations(hh_marg.columns)
    # hhs, hh_jd = cat.joint_distribution(
    #         hh_sample[hh_sample.sample_geog == xwalk[0][1]],
    #         cat.category_combinations(hh_marg.columns))

    # category_names = list(category_df.index.names)
    # print(sample_df.groupby(category_names).size())
    # category_df["frequency"] = sample_df.groupby(category_names).size()
    # print(hhs)
    # print(hh_jd)

    all_households, all_persons, all_stats = zs.synthesize_all_zones(
        hh_marg, p_marg, hh_sample, p_sample, xwalk
    )

    print(all_households)
    print(all_persons)
    print(all_stats)
    all_households.to_csv("../output/syn_hh_ipu.csv", index=False)
    all_persons.to_csv("../output/syn_pp_ipu.csv", index=False)
    all_stats.to_csv("../output/stats_ipu.csv", index=False)


def main():
    name_f = lambda x: os.path.join(
        pathlib.Path(__file__).parent.parent.resolve(),
        "data",
        "testing_new",
        f"{x}.csv",
    )
    hh_marg = name_f("hh_marginals_ipu")
    p_marg = name_f("person_marginals_ipu")
    hh_sample = name_f("hh_sample_ipu")
    p_sample = name_f("pp_sample_ipu")

    # TESTING SAMPLE
    # hh_marg = name_f("hh_marginals")
    # p_marg = name_f("person_marginals")
    # hh_sample = name_f("household_sample")
    # p_sample = name_f("person_sample")
    start_time = time.time()
    test_run(hh_marg, p_marg, hh_sample, p_sample)
    elapsed_time = time.time() - start_time
    print("--- %s seconds ---" % elapsed_time)
    print("--- %s minutes ---" % (elapsed_time / 60))
    print("--- %s hours ---" % (elapsed_time / 360))


if __name__ == "__main__":
    main()
