import pandas as pd
import numpy as np
from pgmpy.estimators import HillClimbSearch, K2Score, BDeuScore, BicScore, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.models import BayesianNetwork
import networkx as nx
import pylab as plt


location_of_processed_data = "../../data/data_processed"
location_data = "../../data"

def input_data():
    ATTRIBUTES = ['AGEGROUP', 'CARLICENCE', 'SEX', 'PERSINC', 'ANYWORK', 'HomeSA1', "CW_ADPERSWGT_SA3"]
    
    # import data
    p_original_df = pd.read_csv("../../data/source/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    # Only have record of the main person (the person that did the survey)
    # p_self_df = p_original_df[p_original_df['RELATIONSHIP']=='Self']
    # h_original_df = pd.read_csv("./source/VISTA_2012_16_v1_SA1_CSV/H_VISTA12_16_SA1_V1.csv")

    # orignal_df = pd.merge(p_self_df, h_original_df, on=['HHID'])
    df = p_original_df[ATTRIBUTES].dropna()
    df = df.rename(columns={'CW_ADPERSWGT_SA3': '_weight'})

    make_like_paper = False
    if make_like_paper:
        df.loc[df['TOTALVEHS'] == 0, 'TOTALVEHS'] = 'NO'
        df.loc[df['TOTALVEHS'] != 'NO', 'TOTALVEHS'] = 'YES'

        df.loc[df['CARLICENCE'] == 'No Car Licence', 'CARLICENCE'] = 'NO'
        df.loc[df['CARLICENCE'] != 'NO', 'CARLICENCE'] = 'YES'
    return df


def input_2():
    df_p_ori = pd.read_csv(location_of_processed_data + "/p_test_seed.csv")
    ATTRIBUTES = ["AGEGROUP", "SEX", "ANYWORK", "CARLICENCE", "PERSINC", "CW_ADPERSWGT_SA3"]
    df = df_p_ori[ATTRIBUTES].dropna()
    df = df.rename(columns={'CW_ADPERSWGT_SA3': '_weight'})
    return df

def input_3():
    df_h_ori = pd.read_csv(location_of_processed_data + "/h_test_seed.csv")
    ATTRIBUTES = [ 
    "HHSIZE", 
    "CARS", 
    "TOTALVEHS",
    "CW_ADHHWGT_SA3",
    "SA3"
    ]
    df = df_h_ori[ATTRIBUTES].dropna()
    df = df.rename(columns={'CW_ADHHWGT_SA3': '_weight'})
    # df_w = pd.read_csv("data/final_summary_hh_weights.csv")
    # df['_weight'] = df_w['SA4_balanced_weight']
    return df

if __name__ == "__main__":
    df_p = input_2()
    df_struc_learn = df_p.drop(columns="_weight")

    est = HillClimbSearch(df_struc_learn)
    #  dunno why but when apply weight into the score it makes the final DAG extremely complicated
    best_DAG = est.estimate(
        scoring_method=BicScore(df_struc_learn),
        # tabu_length=100,
        # epsilon=1e-10,
        # max_iter=1e10
        )
    model = BayesianNetwork(best_DAG)

    # model.fit()
    para_learn = BayesianEstimator(model=model, data=df_p)
    ls_CPDs = para_learn.get_parameters()
    model.add_cpds(*ls_CPDs)

    inference = BayesianModelSampling(model)
    # a = inference.forward_sample(size=100)
    # a.to_csv("bn_test_h_SA3_2.csv", index=False)
    nx.draw_circular(model ,with_labels=True)
    plt.show()