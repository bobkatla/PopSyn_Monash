import pandas as pd
from pathlib import Path

# ------------------
# Step 2: Load CSVs and Compute Conditional Probability Tables
# ------------------
def load_seed_data(data_folder):
    marginals = pd.read_csv(data_folder / "marginals.csv", header=[0, 1, 2], index_col=0)
    seed_households = pd.read_csv(data_folder / "seed_households.csv")
    seed_individuals = pd.read_csv(data_folder / "seed_individuals.csv")
    return marginals, seed_households, seed_individuals


def compute_conditional_prob_gender_given_hhtype(seed_indiv_df, seed_hh_df):
    merged = pd.merge(seed_indiv_df, seed_hh_df[["hid", "hh_type"]], on="hid")
    ctab = pd.crosstab(merged["hh_type"], merged["gender"], normalize="index")
    return ctab


def compute_conditional_prob_marital_given_age(seed_indiv_df):
    ctab = pd.crosstab(seed_indiv_df["age"], seed_indiv_df["marital"], normalize="index")
    return ctab


def compute_conditional_prob_employment_given_age(seed_indiv_df):
    ctab = pd.crosstab(seed_indiv_df["age"], seed_indiv_df["employment"], normalize="index")
    return ctab


def compute_conditional_prob_license_given_age(seed_indiv_df):
    ctab = pd.crosstab(seed_indiv_df["age"], seed_indiv_df["license"], normalize="index")
    return ctab


def main():
    curr_folder = Path(__file__).parent
    data_folder = curr_folder / "data"

    print("\n[STEP 2] Loading seed data...")
    marginals, seed_households, seed_individuals = load_seed_data(data_folder)

    print("\n[STEP 2] Computing conditional probabilities...")
    p_gender_given_hhtype = compute_conditional_prob_gender_given_hhtype(seed_individuals, seed_households)
    p_marital_given_age = compute_conditional_prob_marital_given_age(seed_individuals)
    p_employment_given_age = compute_conditional_prob_employment_given_age(seed_individuals)
    p_license_given_age = compute_conditional_prob_license_given_age(seed_individuals)

    print("\nP(gender | hh_type):\n", p_gender_given_hhtype)
    print("\nP(marital | age):\n", p_marital_given_age)
    print("\nP(employment | age):\n", p_employment_given_age)
    print("\nP(license | age):\n", p_license_given_age)

    # Optionally, save conditionals
    p_gender_given_hhtype.to_csv(data_folder / "cond_gender_given_hhtype.csv")
    p_marital_given_age.to_csv(data_folder / "cond_marital_given_age.csv")
    p_employment_given_age.to_csv(data_folder / "cond_employment_given_age.csv")
    p_license_given_age.to_csv(data_folder / "cond_license_given_age.csv")
    print("\n[STEP 2] Conditional tables saved to /data folder.")


if __name__ == "__main__":
    main()
