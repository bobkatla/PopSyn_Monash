from random import sample
import synthpop.ipf.ipf as ipf
import pandas as pd
from checker import SRMSE

def IPF_sampling(constraints):
    # constraints.to_csv('./Joint_dist_result_IPF.csv')
    return constraints


def IPF_training(df, sample_rate):
    atts = df.columns
    ls_tups = []
    margi_val = []
    
    for att in atts:
        counts = df[att].value_counts()
        indexs = list(counts.index)
        for i, c in enumerate(counts):
            ls_tups.append((att, indexs[i]))
            margi_val.append(c)

    # Margi dist for IPF
    marginal_midx = pd.MultiIndex.from_tuples(ls_tups)
    marginals = pd.Series(margi_val, index=marginal_midx)

    # joint dist for IPF but only the bone
    j_cou = df.value_counts()
    j_idx = list(j_cou.index)
    # To solve zero cell by making a extremely small number
    j_vals = [0.000000000000000000000000000001]*len(j_idx)

    # Fill up the vals for joint from sample
    N = df.shape[0]
    one_percent = int(N/100)
    seed_df = df.sample(n = sample_rate * one_percent).copy()

    seed_cou = seed_df.value_counts()
    seed_idx = list(seed_cou.index)
    for idx in seed_idx:
        i = j_idx.index(idx)
        j_vals[i] = seed_cou[idx]

    joint_dist_midx = pd.MultiIndex.from_tuples(j_idx, names=atts)
    joint_dist = pd.Series(j_vals, index=joint_dist_midx)
    
    constraints, iterations = ipf.calculate_constraints(marginals, joint_dist, tolerance=1e-9)
    return IPF_sampling(constraints)
    # print(iterations)


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'CARLICENCE', 'SEX', 'PERSINC', 'DWELLTYPE', 'TOTALVEHS']
    
    # import data
    p_original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    # Only have record of the main person (the person that did the survey)
    p_self_df = p_original_df[p_original_df['RELATIONSHIP']=='Self']
    h_original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/H_VISTA12_16_SA1_V1.csv")

    orignal_df = pd.merge(p_self_df, h_original_df, on=['HHID'])
    df = orignal_df[ATTRIBUTES].dropna()

    make_like_paper = True
    if make_like_paper:
        df.loc[df['TOTALVEHS'] == 0, 'TOTALVEHS'] = 'NO'
        df.loc[df['TOTALVEHS'] != 'NO', 'TOTALVEHS'] = 'YES'

        df.loc[df['CARLICENCE'] == 'No Car Licence', 'CARLICENCE'] = 'NO'
        df.loc[df['CARLICENCE'] != 'NO', 'CARLICENCE'] = 'YES'

    result_sample = IPF_training(df, 10)
    # print(SRMSE(df, result_sample))
    print(result_sample)
