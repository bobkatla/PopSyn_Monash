import pandas as pd
import numpy as np
from pathlib import Path
import random
from collections import defaultdict, Counter

# ------------------
# Step 3: Sampling with Gibbs iterations + P(n_cars | n_license, hsize)
# ------------------
def load_conditionals(data_folder):
    cond_gender = pd.read_csv(data_folder / "cond_gender_given_hhtype.csv", index_col=0)
    cond_marital = pd.read_csv(data_folder / "cond_marital_given_age.csv", index_col=0)
    cond_employment = pd.read_csv(data_folder / "cond_employment_given_age.csv", index_col=0)
    cond_license = pd.read_csv(data_folder / "cond_license_given_age.csv", index_col=0)
    cond_ncars = pd.read_csv(data_folder / "cond_ncars_given_nlicense_hsize.csv", index_col=[0, 1])
    return cond_gender, cond_marital, cond_employment, cond_license, cond_ncars


def sample_from_distribution(probs):
    probs = probs.dropna()
    if probs.sum() == 0:
        probs = pd.Series([1 / len(probs)] * len(probs), index=probs.index)
    return probs.sample(weights=probs.values).index[0]


def sample_n_cars(n_license, hsize, cond_ncars):
    key = (n_license, hsize)
    if key in cond_ncars.index:
        return int(sample_from_distribution(cond_ncars.loc[key]))
    return min(n_license, np.random.poisson(1.5))  # fallback


def sample_gender(hh_type, cond_table):
    if hh_type in cond_table.index:
        return sample_from_distribution(cond_table.loc[hh_type])
    return np.random.choice(cond_table.columns)


def sample_marital(age, cond_table):
    if age in cond_table.index:
        return sample_from_distribution(cond_table.loc[age])
    return np.random.choice(cond_table.columns)


def sample_employment(age, cond_table):
    if age in cond_table.index:
        return sample_from_distribution(cond_table.loc[age])
    return np.random.choice(cond_table.columns)


def sample_license(age, cond_table):
    if age < 18:
        return 0
    if age in cond_table.index:
        return int(sample_from_distribution(cond_table.loc[age]))
    return int(np.random.rand() > 0.3)


def update_individual(indiv, hh_type, cond_gender, cond_marital, cond_employment, cond_license):
    age = indiv["age"]
    indiv["gender"] = sample_gender(hh_type, cond_gender)
    indiv["marital"] = sample_marital(age, cond_marital)
    indiv["employment"] = sample_employment(age, cond_employment)
    indiv["license"] = sample_license(age, cond_license)
    return indiv


def compute_hhtype_conditionals(seed_individuals, seed_households):
    merged = pd.merge(seed_individuals, seed_households[["hid", "hh_type"]], on="hid")
    grouped = merged.groupby("hid")
    pattern_counts = defaultdict(Counter)

    for hid, group in grouped:
        ages = tuple(sorted(group["age"], reverse=True))
        hh_type = group["hh_type"].iloc[0]
        pattern_counts[ages][hh_type] += 1

    pattern_probs = {ages: pd.Series(hh_counts).sort_index() / sum(hh_counts.values())
                     for ages, hh_counts in pattern_counts.items()}
    return pattern_probs


def compute_conditional_ncars(seed_households):
    ctab = pd.crosstab(index=[seed_households["n_license"], seed_households["hsize"]],
                       columns=seed_households["n_cars"], normalize="index")
    return ctab


def sample_hhtype_given_ages(ages, pattern_probs):
    key = tuple(sorted(ages, reverse=True))
    if key in pattern_probs:
        return sample_from_distribution(pattern_probs[key])
    return random.choice(["S", "C", "S+C", "C+C", "NF"])


def synthesize_household(hsize, pattern_probs, conds, n_iterations=5):
    cond_gender, cond_marital, cond_employment, cond_license, cond_ncars = conds
    ages = sorted(np.random.choice([10, 16, 20, 30, 50, 70], size=hsize, replace=True), reverse=True)
    individuals = [{"age": age} for age in ages]

    hh_type = sample_hhtype_given_ages(ages, pattern_probs)

    for _ in range(n_iterations):
        hh_type = sample_hhtype_given_ages([ind["age"] for ind in individuals], pattern_probs)
        for ind in individuals:
            update_individual(ind, hh_type, cond_gender, cond_marital, cond_employment, cond_license)

    n_license = sum(ind["license"] for ind in individuals)
    n_cars = sample_n_cars(n_license, hsize, cond_ncars)

    household = {
        "hsize": hsize,
        "hh_type": hh_type,
        "n_cars": n_cars,
        "n_license": n_license,
        "individuals": individuals
    }
    return household


def generate_synthetic_population(marginals, n_total_hh, pattern_probs, conds):
    hsize_dist = marginals.loc["zone_1"]["household"]["hsize"].to_dict()
    hsize_pool = [int(size) for size, p in hsize_dist.items() for _ in range(int(n_total_hh * p))]
    random.shuffle(hsize_pool)

    households = []
    for hid, hsize in enumerate(hsize_pool):
        hh = synthesize_household(hsize, pattern_probs, conds, n_iterations=5)
        hh["hid"] = hid
        households.append(hh)
    return households


def explode_population(households):
    hh_records = []
    ind_records = []
    for hh in households:
        hh_rec = {k: hh[k] for k in ["hid", "hsize", "hh_type", "n_cars", "n_license"]}
        hh_records.append(hh_rec)
        for i, ind in enumerate(hh["individuals"]):
            ind_rec = {"hid": hh["hid"], **ind}
            ind_records.append(ind_rec)
    return pd.DataFrame(hh_records), pd.DataFrame(ind_records)


def main():
    curr_folder = Path(__file__).parent
    data_folder = curr_folder / "data"

    print("\n[STEP 3] Loading data and conditionals...")
    conds = load_conditionals(data_folder)
    marginals = pd.read_csv(data_folder / "marginals.csv", header=[0, 1, 2], index_col=0)
    seed_households = pd.read_csv(data_folder / "seed_households.csv")
    seed_individuals = pd.read_csv(data_folder / "seed_individuals.csv")

    print("\n[STEP 3] Computing P(hh_type | sorted_ages)...")
    pattern_probs = compute_hhtype_conditionals(seed_individuals, seed_households)

    print("\n[STEP 3] Computing P(n_cars | n_license, hsize)...")
    cond_ncars = compute_conditional_ncars(seed_households)
    cond_ncars.to_csv(data_folder / "cond_ncars_given_nlicense_hsize.csv")

    print("\n[STEP 3] Generating synthetic population (n=10000 households) with Gibbs iterations...")
    synthetic_households = generate_synthetic_population(marginals, n_total_hh=10000, pattern_probs=pattern_probs, conds=conds)

    df_households, df_individuals = explode_population(synthetic_households)

    df_households.to_csv(data_folder / "synthetic_households.csv", index=False)
    df_individuals.to_csv(data_folder / "synthetic_individuals.csv", index=False)
    print("\nSynthetic population saved:")
    print("- data/synthetic_households.csv")
    print("- data/synthetic_individuals.csv")


if __name__ == "__main__":
    main()
