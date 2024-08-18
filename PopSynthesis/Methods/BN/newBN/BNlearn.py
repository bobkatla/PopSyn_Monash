from PopSynthesis.Methods.BN.newBN import process_data
from PopSynthesis.Methods.BN.TBN.utils import get_state_names


def get_state_names(con_df):
    NotImplemented


def main():
    df_hh_census, df_hh_seed, df_con_hh = process_data.main()
    print(get_state_names(df_con_hh))
    # learn the para and get the final model


if __name__ == "__main__":
    main()
