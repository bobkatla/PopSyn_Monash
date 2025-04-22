"""CSP core: sample by each relationship and ensure the household size matching"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from PopSynthesis.Methods.CSP.run.rela_const import (
    HH_TAG,
    MAIN_PERSON,
    COUNT_COL,
    EXPECTED_RELATIONSHIPS,
    BACK_CONNECTIONS,
    RELA_BY_LEVELS
)
from PopSynthesis.Methods.CSP.const import HHID

TARGET_ID = "target_id"
SYN_COUNT_COL = "syn_count"
MAP_IDS_COL = "potential_target_ids"
MAP_COUNTS_COL = "asscociated_counts"
EVIDENCE_ID = "evidence_id"
global debug_mode
debug_mode = False

def split_and_process_by_target_id(df: pd.DataFrame, evidence_cols: List[str], target_id: str) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Split the df by target_id and return the evidence and sample df"""
    assert target_id in df.columns, f"{target_id} must be in df columns"
    evidences_cond = df[evidence_cols + [target_id, COUNT_COL]].copy()
    target_mapping = df[[x for x in df.columns if x not in (evidence_cols + [COUNT_COL])]].copy()
    evidences_to_ids = evidences_cond.groupby(evidence_cols)[target_id].apply(list)
    evidences_to_counts = evidences_cond.groupby(evidence_cols)[COUNT_COL].apply(list)
    return evidences_to_ids, evidences_to_counts, target_mapping


def init_potentials_to_sample(conditionals: pd.DataFrame, evidences: pd.DataFrame, can_sample_all: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process conditionals by having it evidence as index and process the remainings"""
    assert COUNT_COL in conditionals.columns, "Count column must be in conditionals"

    evidence_cols = [x for x in evidences.columns if x not in (HHID, SYN_COUNT_COL)]

    assert set(evidence_cols).issubset(set(conditionals.columns)), "Evidences must be subset of conditionals"

    evidences_combs_to_target_ids, evidences_combs_to_target_counts, target_mapping = split_and_process_by_target_id(conditionals, evidence_cols, TARGET_ID)
    assert len(evidences_combs_to_target_ids) == len(evidences_combs_to_target_counts), "Evidences to ids and counts must be same length"
    assert set(evidences_combs_to_target_ids.index) == set(evidences_combs_to_target_counts.index), "Evidences to ids and counts must be same things"
    
    # handle multiple hhid
    evidences[MAP_IDS_COL] = evidences.set_index(evidence_cols).index.map(evidences_combs_to_target_ids)
    evidences[MAP_COUNTS_COL] = evidences.set_index(evidence_cols).index.map(evidences_combs_to_target_counts)

    # split by na to check
    possible_to_sample_df = evidences[evidences[MAP_IDS_COL].notna()]
    impossible_to_sample_df = evidences[evidences[MAP_IDS_COL].isna()]
    
    to_sample_df = pd.DataFrame()
    if len(possible_to_sample_df) > 0:
        to_sample_df = possible_to_sample_df.groupby(HHID)[[MAP_IDS_COL, MAP_COUNTS_COL]].agg(list).reset_index()
        
        def process_combined_to_combine_ids_counts(row):
            ids = row[MAP_IDS_COL]
            counts = row[MAP_COUNTS_COL]
            results = {}
            for target_ids, counts in zip(ids, counts):
                for id, count in zip(target_ids, counts):
                    if id not in results:
                        results[id] = count
                    results[id] += count
            return list(results.keys()), list(results.values())
        to_sample_df[[MAP_IDS_COL, MAP_COUNTS_COL]] = to_sample_df.apply(process_combined_to_combine_ids_counts, axis=1, result_type="expand")
        # NOTE: use first NOT sum when do groupby HHID with SYN_COUNT_COL as they are repeated value, also can do mean(), same thing
        to_sample_df[SYN_COUNT_COL] = to_sample_df[HHID].map(evidences.groupby(HHID)[SYN_COUNT_COL].first())

        # TODO: reduce runtime by grouping cases of similar potential ids
        # to_sample_df = to_sample_df.groupby([MAP_IDS_COL, MAP_COUNTS_COL])[HHID].apply(list)

    cannot_sample_df = impossible_to_sample_df.groupby(HHID)[SYN_COUNT_COL].first().reset_index()
    cannot_sample_df = cannot_sample_df[~cannot_sample_df[HHID].isin(to_sample_df[HHID])]

    if can_sample_all:
        assert len(impossible_to_sample_df) == 0, "Impossible to sample df must be empty"
    assert len(to_sample_df) + len(cannot_sample_df) == len(evidences[HHID].unique()), "Ensure all hhids are sampled"

    return to_sample_df, cannot_sample_df, target_mapping


def sample_and_choose_target_ids(possible_to_sample: pd.DataFrame) -> pd.DataFrame:
    def sample_by_row(row):
        ids = row[MAP_IDS_COL]
        counts = row[MAP_COUNTS_COL]
        probs = [count / sum(counts) for count in counts]
        n = row[SYN_COUNT_COL]
        return np.random.choice(ids, size=n, p=probs, replace=True)
    
    # Apply to get the sampled ids
    to_sample_df = possible_to_sample.copy()
    to_sample_df[TARGET_ID] = to_sample_df.apply(sample_by_row, axis=1)

    return to_sample_df


def merge_chosen_target_ids_with_known_cond(to_sample_df: pd.DataFrame, target_mapping: pd.DataFrame, agg_ids: bool) -> pd.DataFrame:
    # merge with known conds to get the final sampled df
    if agg_ids:
        # the case of HHID are list
        to_sample_df['zip_cols'] = to_sample_df.apply(lambda row: list(zip(row[HHID], row[TARGET_ID])), axis=1)
        to_sample_df = to_sample_df.explode('zip_cols')
        to_sample_df[[HHID, TARGET_ID]] = pd.DataFrame(to_sample_df['zip_cols'].tolist(), index=to_sample_df.index)

    else:
        to_sample_df = to_sample_df.explode(TARGET_ID)

    to_sample_df = to_sample_df.drop(columns=[MAP_IDS_COL, MAP_COUNTS_COL, 'zip_cols'], errors='ignore')
    # merge with known conds to get the final sampled df
    final_sampled_df = to_sample_df.merge(target_mapping, on=TARGET_ID, how="inner")
    assert len(final_sampled_df) == len(to_sample_df), "Sampled df must be same length as original df"
    return final_sampled_df


def direct_sample_from_conditional(conditionals: pd.DataFrame, evidences: pd.DataFrame, can_sample_all: bool) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Process conditionals by having it evidence as index and process the remainings"""
    possible_to_sample, impossible_to_sample, target_mapping = init_potentials_to_sample(conditionals.copy(), evidences.copy(), can_sample_all)

    final_sampled = pd.DataFrame()
    if len(possible_to_sample) > 0:
        to_sample_df = sample_and_choose_target_ids(possible_to_sample)
        final_sampled = merge_chosen_target_ids_with_known_cond(to_sample_df, target_mapping, agg_ids=False)

    expected_n_sampled = evidences.groupby(HHID)[SYN_COUNT_COL].first().sum()
    assert len(final_sampled) + impossible_to_sample[SYN_COUNT_COL].sum() == expected_n_sampled, "Processed samples must be same as given evidences"
    return final_sampled, possible_to_sample, impossible_to_sample, target_mapping


def handle_resample_and_update_possible_df(resample_evidences: pd.DataFrame, possible_to_sample_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Resample based on the wrong results and remove the just sampled one as well"""
    assert HHID in resample_evidences.columns, f"{HHID} must be in resample evidences"
    assert TARGET_ID in resample_evidences.columns, f"{TARGET_ID} must be in resample evidences"
    assert possible_to_sample_df[HHID].nunique() == len(possible_to_sample_df), "HHID must be unique in possible to sample df"

    wrong_chosen_target_ids = resample_evidences.groupby(HHID)[TARGET_ID].apply(list)
    sub_possibles = possible_to_sample_df.set_index(HHID).loc[wrong_chosen_target_ids.index]
    
    # map to get rm ids
    sub_possibles["to_rm_ids"] = sub_possibles.index.map(wrong_chosen_target_ids) # map the target id to remove
    def process_get_new_sampling(row):
        # get the target id to remove
        to_rm_id = row["to_rm_ids"] # list
        # get the possible ids to sample from
        possible_ids = row[MAP_IDS_COL]
        associated_counts = row[MAP_COUNTS_COL]
        for rm_id in to_rm_id:
            rm_idx = possible_ids.index(rm_id) # get the index to remove
            # remove the id from the possible ids
            possible_ids = possible_ids[:rm_idx] + possible_ids[rm_idx+1:]
            # do the same for counts
            associated_counts = associated_counts[:rm_idx] + associated_counts[rm_idx+1:]
        assert len(possible_ids) == len(associated_counts), "Possible ids and counts must be same length after removing"
        if len(possible_ids) == 0:
            return None, None
        return possible_ids, associated_counts
    
    sub_possibles[[MAP_IDS_COL, MAP_COUNTS_COL]] = sub_possibles.apply(process_get_new_sampling, axis=1, result_type="expand")
    sub_possibles = sub_possibles.drop(columns=["to_rm_ids"])

    cannot_sample_anymore = sub_possibles[sub_possibles[MAP_IDS_COL].isna()]

    sub_possibles = sub_possibles[sub_possibles[MAP_IDS_COL].notna()]
    # now we can sample from the new possible df
    updated_chosen_target_ids = sample_and_choose_target_ids(sub_possibles)

    return updated_chosen_target_ids.reset_index(), cannot_sample_anymore


def determine_n_rela_for_each_hh(hh_df: pd.DataFrame, hhsz: str, n_rela_conditional: pd.DataFrame) -> Dict[str, int]:
    # we can just build a BN or a direct sample from the hh_df
    hh_df[SYN_COUNT_COL] = 1 # just to have a count for each hh
    final_sampled_df, possible_to_sample, impossible_to_sample, target_mapping = direct_sample_from_conditional(n_rela_conditional, hh_df, can_sample_all=True)
    assert len(impossible_to_sample) == 0, "Impossible to sample df must be empty"
    final_sampled_df = final_sampled_df.merge(hh_df.drop(columns=SYN_COUNT_COL), on=HHID, how="inner")
    assert len(final_sampled_df) == len(hh_df), "Final sampled df must be same length as original hh df"

    # check with hhsz
    n_rela_cols = [f"n_{rela}" for rela in EXPECTED_RELATIONSHIPS]
    final_results = []
    while True:
        final_sampled_df["synthesized_hhsz"] = final_sampled_df[n_rela_cols].sum(axis=1)
        def check_match_hhsz(row):
            expected_hhsz = row[f"{HH_TAG}_{hhsz}"]
            synthesized_hhsz = row["synthesized_hhsz"]
            if expected_hhsz == "8+":
                return synthesized_hhsz >= 8
            else:
                return synthesized_hhsz == int(expected_hhsz)
        final_sampled_df["correct_hhsz"] = final_sampled_df.apply(check_match_hhsz, axis=1)
        correct_results = final_sampled_df[final_sampled_df["correct_hhsz"]]
        final_results.append(correct_results)
        wrong_results = final_sampled_df[~final_sampled_df["correct_hhsz"]]
        n_wrong = len(wrong_results)
        if n_wrong > 0:
            updated_possible_samples, cannot_sample_cases = handle_resample_and_update_possible_df(wrong_results, possible_to_sample)
            assert len(cannot_sample_cases) == 0, "Cannot sample cases must be empty" # this is the current case for hhsz confirmation
            new_sampled = merge_chosen_target_ids_with_known_cond(updated_possible_samples, target_mapping, agg_ids=False)
            final_sampled_df = new_sampled
            possible_to_sample = updated_possible_samples
        else:
            break

    return pd.concat(final_results, ignore_index=True).drop(columns=["synthesized_hhsz", "correct_hhsz"])


def csp_sample_by_hh(hh_df: pd.DataFrame, final_conditonals: Dict[str, pd.DataFrame], hhsz:str, relationship:str) -> pd.DataFrame:
    # process each conditionals to have target id
    hh_df = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    for key in final_conditonals.keys():
        final_conditonals[key] = final_conditonals[key].reset_index(drop=True) # ensure we have a clean index
        final_conditonals[key][TARGET_ID] = final_conditonals[key].index + 1
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
            print(f"Sampling {rela}...")

            rela_results = []
            sampled_ids = []
            for prev_src in BACK_CONNECTIONS[rela]:
                print(f"Sampling {rela} from {prev_src}...")
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

                assert len(final_sampled_df) > 0, "Final sampled df must be not empty"
                final_syn_rela = pd.concat(fin_sampled_results, ignore_index=True)
                final_syn_rela["src_sample"] = prev_src
                rela_results.append(final_syn_rela)
                sampled_ids += final_syn_rela[HHID].tolist() # updated to avoid duplicates

                possibles_cond[f"{prev_src}-{rela}"] = pd.concat(hold_possible_to_sample, ignore_index=True)
                store_target_mapping[f"{prev_src}-{rela}"] = target_mapping


            if len(rela_results) == 0:
                print(f"No {rela} to sample")
                continue
            final_rela = pd.concat(rela_results, ignore_index=True)
            evidences_store[rela] = final_rela

    raise NotImplementedError("Not implemented yet")
    return