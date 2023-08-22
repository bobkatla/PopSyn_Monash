"""
Tools to help realise the full pop and prepare to calculate the SRMSE
"""
import pandas as pd
import numpy as np


def realise_full_pop_based_on_weight(df:pd.DataFrame, weight_col:str) -> pd.DataFrame:
    final_data = None
    
    raw_non_weight = df.drop(columns=weight_col).to_numpy()
    weights = df[weight_col].to_numpy()
    cols = df.drop(columns=weight_col).columns

    final_raw = []
    for i in range(len(weights)):
        d = raw_non_weight[i]
        # NOTE: can do better intergisation
        w = int(float(weights[i]))
        final_raw.append(np.repeat([d], w, axis=0))
    
    processed_final = np.concatenate(final_raw, axis=0)
    final_data = pd.DataFrame(processed_final, columns=cols)
    return final_data


def sampling_from_full_pop(full_pop:pd.DataFrame, rate: float) -> pd.DataFrame:
    assert rate <= 1
    sample_pop = full_pop.sample(frac=rate)
    return sample_pop


def condense_pop(pop_df: pd.DataFrame, name_for_weight_col: str) -> pd.DataFrame:
    # This is used to condense the given dataframe and have the new weight col
    # Due to its nature, the weight col will be int, maybe unnatural
    counts = pop_df.value_counts()
    final_df = counts.index.to_frame(index=False)
    final_df[name_for_weight_col] = counts.values
    if 'hh_num' in final_df: final_df = final_df.drop(columns="hh_num")
    return final_df


def get_pp_based_on_id(hh_pop: pd.DataFrame, full_pp_pop: pd.DataFrame, shared_id_col: str) -> pd.DataFrame:
    # Return the corr PP pop based on the HH pop (based on the id)
    assert shared_id_col in hh_pop
    assert shared_id_col in full_pp_pop
    hh_ids_in_seed = hh_pop[shared_id_col]
    final_df = full_pp_pop[full_pp_pop[shared_id_col].isin(hh_ids_in_seed)]
    if 'hh_num' in final_df: final_df = final_df.drop(columns="hh_num")
    return final_df


def wrapper_get_all(seed_df_hh:pd.DataFrame, seed_df_pp:pd.DataFrame, sample_rate:float, name_weights_in_hh:str, new_name_weights_in_hh:str, shared_ids_name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    # check for the weights and intergerlise it and then realise the full populationj
    full_hh_seed = realise_full_pop_based_on_weight(seed_df_hh, name_weights_in_hh)
    # sample from it based on the percentage
    sample_of_pop = sampling_from_full_pop(full_pop=full_hh_seed, rate=sample_rate)
    # Condense the population to create a 'new seed data'
    fake_HH_seed_data = condense_pop(sample_of_pop, name_for_weight_col=new_name_weights_in_hh)
    # find a way to do it for both HH and PP
    fake_PP_seed_data = get_pp_based_on_id(fake_HH_seed_data, seed_df_pp, shared_ids_name)

    return fake_HH_seed_data, fake_PP_seed_data


def main():
    # NOTE: later will make this dynamic by getting input from the cmd
    loc_data = "../data/"
    # import the seed data
    seed_df_hh = pd.read_csv(loc_data + "H_sample.csv")
    seed_df_pp = pd.read_csv(loc_data + "P_sample.csv")
    
    fake_HH_seed_data, fake_PP_seed_data = wrapper_get_all(seed_df_hh, seed_df_pp, sample_rate=0.01, name_weights_in_hh="wdhhwgt_sa3", new_name_weights_in_hh='_weight', shared_ids_name='hhid')

    print(fake_HH_seed_data, fake_PP_seed_data)


if __name__ == "__main__":
    main()

