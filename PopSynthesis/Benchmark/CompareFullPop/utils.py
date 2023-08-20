"""
Tools to help realise the full pop and prepare to calculate the SRMSE
"""
import pandas as pd


def realise_full_pop_based_on_weight(df:pd.DataFrame, weight_col:str) -> pd.DataFrame:
    NotImplemented


def main():
    # later will make this dynamic by getting input from the cmd
    loc_data = "../data/"
    # import the seed data
    seed_df_hh = pd.read_csv(loc_data + "H_sample.csv")
    seed_df_pp = pd.read_csv(loc_data + "P_sample.csv")
    # check for the weights and intergerlise it and then realise the full population
    # sample from it based on the percentage
    # find a way to do it for both HH and PP



if __name__ == "__main__":
    main()

