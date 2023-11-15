import pandas as pd
import pickle
import os
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, learn_para_BN
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

from PopSynthesis.Methods.connect_HH_PP.paras_dir import data_dir, processed_data, geo_lev


def reject_samp_veh(BN, df_marg, zone_lev):
    inference = BayesianModelSampling(BN)
    ls_total_veh = [
        ("Four or more motor vehicles", "4+"), 
        ("Three motor vehicles", "3"), 
        ("Two motor vehicles", "2"), 
        ("One motor vehicle", "1"),
        ("No motor vehicles", "0"),
        ("None info", None)]
    ls_all = []
    for zone in df_marg[zone_lev]:
        print(f"DOING {zone}")
        ls_re = []
        zone_info = df_marg[df_marg[zone_lev]==zone]
        assert len(zone_info) == 1
        for totveh_label in ls_total_veh:
            n_totvehs = int(zone_info[totveh_label[0]].iat[0])
            evidence = State('totalvehs', totveh_label[1]) if totveh_label[1] is not None else None
            # Weird case of multiple
            syn = None
            if evidence:
                syn = inference.rejection_sample(evidence=[evidence], size=n_totvehs, show_progress=True)
            else:
                syn = inference.forward_sample(size=n_totvehs, show_progress=True)
            ls_re.append(syn)
        if ls_re == []: continue
        final_for_zone = pd.concat(ls_re, axis=0)
        final_for_zone[zone_lev] = zone
        ls_all.append(final_for_zone)
    final_result = pd.concat(ls_all, axis=0)
    return final_result


def process_census_numvehs(geo_lev="POA"):
    df = pd.read_csv(os.path.join(data_dir, f"{geo_lev}_numvehs.csv"), skiprows=9, skipfooter=7, engine='python')
    df = df.dropna(axis=1, how='all')
    df = df.dropna(axis=0, thresh=6)
    df = df[:-1]
    df["None info"] = df["Not stated"] + df["Not applicable"]
    df = df.drop(columns=["Not stated", "Not applicable", "Total"])
    df = df.rename({"VEHRD Number of Motor Vehicles (ranges)" : geo_lev}, axis=1)
    df[geo_lev] = df.apply(lambda r: r[geo_lev].replace(", VIC", ""), axis=1)
    return df


def main():
    #learning to get the HH only with main person
    df_seed = pd.read_csv(os.path.join(processed_data, "connect_hh_main.csv"))
    # drop all the ids as they are not needed for in BN learning
    id_cols = [x for x in df_seed.columns if "hhid" in x or "persid" in x]
    df_seed = df_seed.drop(columns=id_cols)

    pp_state_names = None
    with open(os.path.join(processed_data, 'dict_pp_states.pickle'), 'rb') as handle:
        pp_state_names = pickle.load(handle)
    hh_state_names = None
    with open(os.path.join(processed_data, 'dict_hh_states.pickle'), 'rb') as handle:
        hh_state_names = pickle.load(handle)
    state_names = hh_state_names | pp_state_names

    print("Learn BN")
    model = learn_struct_BN_score(df_seed, show_struct=False, state_names=state_names)
    model = learn_para_BN(model, df_seed)
    print("Doing the sampling")

    census_df = process_census_numvehs(geo_lev=geo_lev)
    final_syn_pop = reject_samp_veh(BN=model, df_marg=census_df, zone_lev=geo_lev)
    final_syn_pop.to_csv(os.path.join(processed_data, f"SynPop_hh_main_{geo_lev}.csv"), index=False)


if __name__ == "__main__":
    main()