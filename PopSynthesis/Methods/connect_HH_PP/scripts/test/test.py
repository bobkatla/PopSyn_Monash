import pandas as pd
import os
import torch
from pomegranate.bayesian_network import BayesianNetwork
from PopSynthesis.Methods.connect_HH_PP.paras_dir import data_dir, processed_data


def main():
    # Import HH and PP samples (VISTA)
    hh_df_raw = pd.read_csv(os.path.join(data_dir ,"H_VISTA_1220_SA1.csv"))
    pp_df_raw = pd.read_csv(os.path.join(data_dir, "P_VISTA_1220_SA1.csv"))
    hold_hh_df = hh_df_raw[["totalvehs", "hhsize", "hhinc", "yearslived"]]
    hold_hh_df = hold_hh_df.fillna(0).astype("int64")
    test_hh_tensor = torch.tensor(hold_hh_df.values)
    model = BayesianNetwork(algorithm='chow-liu')
    model.fit(test_hh_tensor)
    print(model.structure)

if __name__ == "__main__":
    main()