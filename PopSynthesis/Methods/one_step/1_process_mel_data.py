import pandas as pd
from pathlib import Path


def load_marginals(hh_path, person_path):
    hh_marginals = pd.read_csv(hh_path, header=[0, 1], index_col=0)
    person_marginals = pd.read_csv(person_path, header=[0, 1], index_col=0)
    return hh_marginals, person_marginals


def load_seed_data(hh_seed_path, pp_seed_path):
    hh_seed = pd.read_csv(hh_seed_path)
    pp_seed = pd.read_csv(pp_seed_path)
    return hh_seed, pp_seed


def classify_household_type(group):
    rels = group['relationship'].value_counts()
    n_people = len(group)
    if 'Others' in rels:
        return 'Group Household'
    if n_people == 1:
        return 'Single Person'
    if rels.get('Main', 0) == 1 and rels.get('Spouse', 0) == 1 and n_people == 2:
        return 'Couple Only'
    if rels.get('Main', 0) == 1 and rels.get('Spouse', 0) == 1 and rels.get('Child', 0) >= 1 and set(rels.index) == {'Main', 'Spouse', 'Child'}:
        return 'Couple with Children'
    if rels.get('Main', 0) == 1 and rels.get('Child', 0) >= 1 and 'Spouse' not in rels and set(rels.index) == {'Main', 'Child'}:
        return 'Single Parent Family'
    return 'Complicated Family'


def assign_household_type(pp_df):
    hh_types = pp_df.groupby("serialno").apply(classify_household_type).rename("hh_type").reset_index()
    return hh_types


def merge_with_households(hh_df, pp_df):
    hh_types = assign_household_type(pp_df)
    true_hhsize = pp_df.groupby("serialno").size().rename("true_hhsize").reset_index()
    hh_out = pd.merge(hh_df, hh_types, on="serialno")
    hh_out = pd.merge(hh_out, true_hhsize, on="serialno")
    return hh_out


def save_prepared_data(hh_out, pp_df, out_folder):
    out_folder.mkdir(parents=True, exist_ok=True)
    hh_out.to_csv(out_folder / "processed_households.csv", index=False)
    pp_df.to_csv(out_folder / "processed_individuals.csv", index=False)


if __name__ == "__main__":
    curr_dir = Path(__file__).parent
    data_dir = curr_dir / "data"
    output_dir = curr_dir / "output"

    hh_marginals, person_marginals = load_marginals(
        data_dir / "hh_marginals_ipu.csv", data_dir / "person_marginals_ipu.csv"
    )
    hh_seed, pp_seed = load_seed_data(
        data_dir / "hh_sample_ipu.csv", data_dir / "pp_sample_ipu.csv"
    )
    
    hh_processed = merge_with_households(hh_seed, pp_seed)
    save_prepared_data(hh_processed, pp_seed, output_dir)
    print("Prepared data saved to output folder.")
