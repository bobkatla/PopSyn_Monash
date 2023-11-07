import pandas as pd
import os, glob

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State
import random
from numpy.random import multinomial


def main():
    # path = r'../data' # use your path
    # all_files = glob.glob(os.path.join(path , "connect*"))
    # for file in all_files:
    #     print(f"DOING {file}")
    #     df = pd.read_csv(file)
    #     # drop all the ids as they are not needed for in BN learning
    #     id_cols = [x for x in df.columns if "hhid" in x or "persid" in x]
    #     df = df.drop(columns=id_cols)
    #     print("Learn BN")
    #     model = learn_struct_BN_score(df, show_struct=True)
    #     model = learn_para_BN(model, df)
    #     print("Doing the sampling")

    #learning to get the HH only with main person
    df_seed = pd.read_csv("connect_hh_main.csv")
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)
    print("Learn BN")
    model = learn_struct_BN_score(df_seed, show_struct=False)
    model = learn_para_BN(model, df_seed)
    print("Doing the sampling")
    census_df = pd.read_csv("census_sa1.csv")
    inference = BayesianModelSampling(model)
    final_syn_pop = inference.forward_sample(size=2000000, show_progress=True)
    final_syn_pop.to_csv("SynPop_hh_check_foward_sa1.csv", index=False)

if __name__ == "__main__":
    main()