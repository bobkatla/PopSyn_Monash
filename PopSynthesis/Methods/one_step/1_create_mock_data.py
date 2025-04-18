import numpy as np
import pandas as pd
import random
from collections import Counter
from pathlib import Path

# ------------------
# Step 1: Full Population -> Marginals + Sample (Seed)
# ------------------
def generate_full_population(n_households=10000):
    household_data = []
    individual_data = []

    for h_id in range(n_households):
        hsize = random.choice([1, 2, 3, 4, 5])
        n_license = max(0, np.random.binomial(hsize, 0.6))
        n_cars = min(n_license, np.random.poisson(1.5))
        hh_type = random.choice(["S", "C", "S+C", "C+C", "NF"])

        household_data.append({
            "hid": h_id,
            "hsize": hsize,
            "hh_type": hh_type,
            "n_cars": n_cars,
            "n_license": n_license
        })

        for i in range(hsize):
            age = random.choices([10, 16, 20, 30, 50, 70], weights=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1])[0]
            gender = random.choice(["M", "F"])
            marital = "Single" if age < 25 else random.choice(["Single", "Married", "Divorced"])
            employment = "Education" if age < 18 else random.choice(["Employed", "Unemployed", "Retired"])
            has_license = int(age >= 18 and random.random() > 0.3)

            individual_data.append({
                "hid": h_id,
                "age": age,
                "gender": gender,
                "marital": marital,
                "employment": employment,
                "license": has_license
            })

    df_households = pd.DataFrame(household_data)
    df_individuals = pd.DataFrame(individual_data)
    return df_households, df_individuals


def compute_marginals_from_population(df_households, df_individuals):
    marginals_dict = {}

    def norm_counts(series):
        return series.value_counts(normalize=True).sort_index()

    marginals_dict.update({("household", "hsize", k): v for k, v in norm_counts(df_households["hsize"]).items()})
    marginals_dict.update({("household", "hh_type", k): v for k, v in norm_counts(df_households["hh_type"]).items()})
    marginals_dict.update({("household", "n_cars", k): v for k, v in norm_counts(df_households["n_cars"]).items()})
    marginals_dict.update({("individual", "age", k): v for k, v in norm_counts(df_individuals["age"]).items()})
    marginals_dict.update({("individual", "gender", k): v for k, v in norm_counts(df_individuals["gender"]).items()})
    marginals_dict.update({("individual", "marital", k): v for k, v in norm_counts(df_individuals["marital"]).items()})
    marginals_dict.update({("individual", "employment", k): v for k, v in norm_counts(df_individuals["employment"]).items()})
    marginals_dict.update({("individual", "license", k): v for k, v in norm_counts(df_individuals["license"]).items()})

    marginals = pd.DataFrame(marginals_dict, index=["zone_1"])
    marginals.columns = pd.MultiIndex.from_tuples(marginals.columns)
    return marginals


def draw_seed_sample(full_households, full_individuals, sample_frac=0.05):
    sampled_hids = full_households.sample(frac=sample_frac, random_state=42)["hid"]
    seed_households = full_households[full_households["hid"].isin(sampled_hids)].copy()
    seed_individuals = full_individuals[full_individuals["hid"].isin(sampled_hids)].copy()
    return seed_households.reset_index(drop=True), seed_individuals.reset_index(drop=True)


if __name__ == "__main__":
    print("\n[STEP 1] Generating full mock population...")
    full_households, full_individuals = generate_full_population(n_households=10000)

    print("\n[STEP 1] Computing marginals from full population...")
    marginals = compute_marginals_from_population(full_households, full_individuals)

    print("\n[STEP 1] Drawing seed sample from full population...")
    seed_households, seed_individuals = draw_seed_sample(full_households, full_individuals, sample_frac=0.05)

    print("\nSaving outputs to /data folder...")
    curr_folder_path = Path(__file__).parent
    data_folder = curr_folder_path / "data"
    data_folder.mkdir(exist_ok=True)

    marginals.to_csv(data_folder / "marginals.csv")
    seed_households.to_csv(data_folder / "seed_households.csv", index=False)
    seed_individuals.to_csv(data_folder / "seed_individuals.csv", index=False)

    print("\nFiles saved:")
    print("- data/marginals.csv")
    print("- data/seed_households.csv")
    print("- data/seed_individuals.csv")
