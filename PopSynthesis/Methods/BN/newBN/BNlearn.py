from PopSynthesis.Methods.BN.newBN import process_data
from PopSynthesis.Methods.BN.TBN.utils import get_prior

def main():
    df_hh_census, df_hh_seed, df_con_hh = process_data.main()
    # learn the struct
    # learn the para and get the final model
    

if __name__ == "__main__":
    main()