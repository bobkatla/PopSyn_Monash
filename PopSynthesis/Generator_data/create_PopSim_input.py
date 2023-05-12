import pandas as pd


def get_seed_P(atts):
    NotImplemented


def get_seed_H(atts):
    NotImplemented


def get_census_sa(atts, sa_level):
    NotImplemented


def get_geo_cross():
    NotImplemented


def get_ls_needed_df(seed_atts_P, seed_atts_H, census_atts):

    seed_data_P = get_seed_P(seed_atts_P)
    seed_data_H = get_seed_H(seed_atts_H)
    census_data_sa1 = get_census_sa(census_atts, sa_level="SA1")
    census_data_sa2 = get_census_sa(census_atts, sa_level="SA2")
    census_data_sa3 = get_census_sa(census_atts, sa_level="SA3")
    census_data_sa4 = get_census_sa(census_atts, sa_level="SA4")
    geo_cross = get_geo_cross()

    return (
        (seed_data_P, "P_sample.csv"),
        (seed_data_H, "P_sample.csv"),
        (census_data_sa1, "P_sample.csv"),
        (census_data_sa2, "P_sample.csv"),
        (census_data_sa3, "P_sample.csv"),
        (census_data_sa4, "P_sample.csv"),
        (geo_cross, "P_sample.csv")
    )


def output_csv(ls_to_csv, out_loc="./"):
    NotImplemented


def main():
    seed_atts_P =[

    ] 
    seed_atts_H = [

    ]
    census_atts = [

    ]

    ls_to_csv = get_ls_needed_df(seed_atts_P, seed_atts_H, census_atts)

    output_csv(ls_to_csv, out_loc="./")

if __name__ == "__main__":
    main()