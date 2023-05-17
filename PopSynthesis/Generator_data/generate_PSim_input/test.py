import pandas as pd


df_h = pd.read_csv("./data/H_sample.csv")
df_p = pd.read_csv("./data/P_sample.csv")
print(df_p["persinc"].unique())