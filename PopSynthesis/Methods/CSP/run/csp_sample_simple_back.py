"""CSP core: sample by each relationship and ensure the household size matching"""

import pandas as pd
from typing import Dict
from PopSynthesis.Methods.CSP.run.rela_const import (
    HH_TAG,
    BACK_CONNECTIONS,
    RELA_BY_LEVELS
)
from PopSynthesis.Methods.CSP.const import HHID, PP_ATTS
from PopSynthesis.Methods.CSP.run.sample_utils import (
    TARGET_ID,
    SYN_COUNT_COL,
    determine_n_rela_for_each_hh,
    direct_sample_from_conditional,
    handle_resample_and_update_possible_df,
    merge_chosen_target_ids_with_known_cond
)


def csp_sample_by_hh(hh_df: pd.DataFrame, final_conditonals: Dict[str, pd.DataFrame], hhsz:str, relationship:str) -> pd.DataFrame:
    # process each conditionals to have target id
    hh_df = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    for key in final_conditonals.keys():
        final_conditonals[key] = final_conditonals[key].reset_index(drop=True) # ensure we have a clean index
        final_conditonals[key][TARGET_ID] = final_conditonals[key].index + 1
    # print("Processing hh_df... to get the n_rela for each hh")
    processed_hh_df = determine_n_rela_for_each_hh(hh_df.copy(), hhsz, final_conditonals[f"{HH_TAG}-counts"])
    assert processed_hh_df[HHID].nunique() == len(hh_df), "Processed hh df must have same hhid as original hh df"
    processed_hh_df[f"n_{HH_TAG}"] = 1 # just for compleness

    # TODO: implement the resampling directly at the current sampling
    # TODO: for case that cannot do sampling again, we do the whole process with the updated hh_df
    # Start the sampling process
    evidences_store = {HH_TAG: hh_df}
    possibles_cond = {}
    store_target_mapping = {}
    cannot_sample_cases = {}
    for relationships in RELA_BY_LEVELS:
        # Doing the relationship in this level
        for rela in relationships:
            # print(f"Sampling {rela}...")

            rela_results = []
            sampled_ids = []
            for prev_src in BACK_CONNECTIONS[rela]:
                # print(f"Sampling {rela} from {prev_src}...")
                if prev_src not in evidences_store:
                    # Somehow this area does not have the prev rela
                    continue
                conditional = final_conditonals[f"{prev_src}-{rela}"]
                evidences = evidences_store[prev_src].drop(columns=[TARGET_ID, "src_sample"], errors='ignore')

                # Process evidences to get the needed to sample only
                evidences["prev_src_count"] = evidences[HHID].map(processed_hh_df.set_index(HHID)[f"n_{prev_src}"])
                evidences[SYN_COUNT_COL] = evidences[HHID].map(processed_hh_df.set_index(HHID)[f"n_{rela}"])
                evidences = evidences[(evidences[SYN_COUNT_COL] > 0) & (evidences["prev_src_count"] > 0)].copy() # only care where we need to sample
                evidences = evidences.drop(columns=["prev_src_count"])
                evidences = evidences[~evidences[HHID].isin(sampled_ids)] # get the ones have not sampled yet
                if len(evidences) == 0:
                    continue

                final_sampled_df, possible_to_sample, impossible_to_sample, target_mapping = direct_sample_from_conditional(conditional.copy(), evidences, can_sample_all=False)

                # store for later use if needed
                hold_possible_to_sample = [possible_to_sample]

                fin_sampled_results = [final_sampled_df] if len(final_sampled_df) > 0 else []
                if len(impossible_to_sample) > 0:
                    prev_samples = evidences_store[prev_src].copy()
                    to_resample_prev = prev_samples[prev_samples[HHID].isin(impossible_to_sample[HHID])]
                    updated_prev_samples = [prev_samples[~prev_samples[HHID].isin(impossible_to_sample[HHID])]]

                    assert "src_sample" in to_resample_prev.columns, "src_sample must be in the prev df"
                    assert TARGET_ID in to_resample_prev.columns, "target_id must be in the prev df"
                    # split by src sample to handle
                    for src_sample in to_resample_prev["src_sample"].unique():
                        hold_prev_results = []
                        sub_prev_samples = to_resample_prev[to_resample_prev["src_sample"] == src_sample]
                        sub_prev_samples = sub_prev_samples.drop(columns=["src_sample"])
                        # start doing the resampling for prev
                        prev_conditionals = possibles_cond[f"{src_sample}-{prev_src}"].copy()
                        prev_target_mapping = store_target_mapping[f"{src_sample}-{prev_src}"].copy()
                        # likely a while loop here that can update
                        resulted_new_samples = []
                        ls_cannot = []
                        while True:
                            updated_conditional, cannot_sample = handle_resample_and_update_possible_df(sub_prev_samples, prev_conditionals)
                            if len(cannot_sample) > 0:
                                ls_cannot.append(cannot_sample)

                            new_prev_samples, new_curr_samples, new_impossibles = None, None, None # init 
                            if len(updated_conditional) > 0:
                                new_prev_samples = merge_chosen_target_ids_with_known_cond(updated_conditional, prev_target_mapping, agg_ids=False)
                                new_curr_samples, sub_possibles, new_impossibles, _ = direct_sample_from_conditional(conditional.copy(), new_prev_samples.drop(columns=[TARGET_ID]), can_sample_all=False)
                                if len(sub_possibles) > 0:
                                    hold_possible_to_sample.append(sub_possibles)
                            else:
                                break

                            if len(new_curr_samples) > 0:
                                resulted_new_samples.append(new_curr_samples)
                                kept_prev_samples = new_prev_samples[~new_prev_samples[HHID].isin(new_impossibles[HHID])]
                                hold_prev_results.append(kept_prev_samples)
                            if len(new_impossibles) == 0:
                                break
                            else:
                                sub_prev_samples = new_prev_samples[new_prev_samples[HHID].isin(new_impossibles[HHID])]
                                prev_conditionals = updated_conditional

                        if len(resulted_new_samples) > 0:
                            fin_sampled_results.append(pd.concat(resulted_new_samples, ignore_index=True))
                        final_cannot = pd.concat(ls_cannot, ignore_index=True) if len(ls_cannot) > 0 else pd.DataFrame()

                        if len(hold_prev_results) > 0:
                            concat_prev_samples = pd.concat(hold_prev_results, ignore_index=True)
                            concat_prev_samples["src_sample"] = src_sample
                            updated_prev_samples.append(concat_prev_samples)
                    # update prev results
                    assert len(updated_prev_samples) > 0, "Somehow we just have empty results"
                    evidences_store[prev_src] = pd.concat(updated_prev_samples, ignore_index=True)

                # cannot_sample_cases[f"{prev_src}-{rela}"] = cannot_sample

                if len(final_sampled_df) > 0:
                    final_syn_rela = pd.concat(fin_sampled_results, ignore_index=True)
                    final_syn_rela["src_sample"] = prev_src
                    rela_results.append(final_syn_rela)
                    sampled_ids += final_syn_rela[HHID].tolist() # updated to avoid duplicates

                possibles_cond[f"{prev_src}-{rela}"] = pd.concat(hold_possible_to_sample, ignore_index=True)
                store_target_mapping[f"{prev_src}-{rela}"] = target_mapping


            if len(rela_results) == 0:
                # print(f"No {rela} to sample")
                continue
            final_rela = pd.concat(rela_results, ignore_index=True)
            evidences_store[rela] = final_rela

    concat_pp_ls = []
    for rela, df in evidences_store.items():
        if rela == HH_TAG:
            continue
        df = df.drop(columns=[TARGET_ID, SYN_COUNT_COL, "src_sample"], errors='ignore')
        df[relationship] = rela
        df = df.rename(columns={f"{rela}_{att}": att for att in PP_ATTS})
        concat_pp_ls.append(df)
    # for rela in EXPECTED_RELATIONSHIPS:
    #     print(f"Checking {rela}...")
    #     print(processed_hh_df[f"n_{rela}"].sum(), len(evidences_store.get(rela, [])))
    return pd.concat(concat_pp_ls, ignore_index=True)
