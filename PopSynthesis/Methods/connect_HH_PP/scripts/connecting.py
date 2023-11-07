import pandas as pd
import os, glob

from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State


def main():
    path = r'../data' # use your path
    all_files = glob.glob(os.path.join(path , "connect*"))
    for file in all_files:
        print(f"DOING {file}")
        df = pd.read_csv(file)
        # drop all the ids as they are not needed for in BN learning
        id_cols = [x for x in df.columns if "hhid" in x or "persid" in x]
        df = df.drop(columns=id_cols)
        print("Learn BN")
        model = learn_struct_BN_score(df, show_struct=True)
        model = learn_para_BN(model, df)
        # print("Doing the sampling")
        # inference = BayesianModelSampling(model)


if __name__ == "__main__":
    main()