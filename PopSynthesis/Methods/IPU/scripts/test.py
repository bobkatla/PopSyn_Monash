# from synthpop.recipes.tests import test_starter
import synthpop.zone_synthesizer as zs
import synthpop.ipu.ipu as ipu
from synthpop import categorizer as cat

import os
import pandas as pd


def test_run(hh_marg, p_marg, hh_sample, p_sample):
    hh_marg, p_marg, hh_sample, p_sample, xwalk = zs.load_data(hh_marg,
                                                               p_marg,
                                                               hh_sample,
                                                               p_sample)
    # a = zs.synch_hhids(hh_marg, p_marg, xwalk)
    sample_df = hh_sample[hh_sample.sample_geog == xwalk[0][1]]
    category_df = cat.category_combinations(hh_marg.columns)
    # hhs, hh_jd = cat.joint_distribution(
    #         hh_sample[hh_sample.sample_geog == xwalk[0][1]],
    #         cat.category_combinations(hh_marg.columns))
    
    category_names = list(category_df.index.names)
    print(sample_df.groupby(category_names).size())
    # category_df["frequency"] = sample_df.groupby(category_names).size()
    # print(hhs)
    # print(hh_jd)
    
    # all_households, all_persons, all_stats = zs.synthesize_all_zones(hh_marg,
    #                                                                  p_marg,
    #                                                                  hh_sample,
    #                                                                  p_sample,
    #                                                                  xwalk)
    
    # print(all_households)
    # print(all_persons)
    # print(all_stats)
    

def main():
    name_f = lambda x: os.path.join(os.path.dirname(__file__), f'../data/{x}.csv')
    hh_marg = name_f("hh_marginals")
    p_marg = name_f("person_marginals")
    hh_sample = name_f("household_sample")
    p_sample = name_f("person_sample")
    a = test_run(hh_marg, p_marg, hh_sample, p_sample)


if __name__ == "__main__":
    main()