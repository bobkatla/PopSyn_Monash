"""To sample one way, i.e. no loop back, we use the n_rela count as well"""
import pandas as pd
from PopSynthesis.Methods.CSP.const import HHID, PP_ATTS
from PopSynthesis.Methods.CSP.run.rela_const import HH_TAG, RELA_BY_LEVELS, BACK_CONNECTIONS, EXPECTED_RELATIONSHIPS
from PopSynthesis.Methods.CSP.run.sample_utils import (
    TARGET_ID,
    SYN_COUNT_COL,
    determine_n_rela_for_each_hh,
    direct_sample_from_conditional,
    handle_resample_and_update_possible_df,
    merge_chosen_target_ids_with_known_cond
)
from typing import Dict, List

def sample_one_way(hh_df: pd.DataFrame, final_conditonals: Dict[str, pd.DataFrame], hhsz: str, relationship: str, possible_states:Dict[str, List[str]]=None) -> pd.DataFrame:
    """Sample one way from the hh df"""
    # the n rela will be determined as normal
    # process each conditionals to have target id
    hh_df = hh_df.rename(columns={col: f"{HH_TAG}_{col}" for col in hh_df.columns if col != HHID})
    for key in final_conditonals.keys():
        final_conditonals[key] = final_conditonals[key].reset_index(drop=True) # ensure we have a clean index
        final_conditonals[key][TARGET_ID] = final_conditonals[key].index + 1
    # print("Processing hh_df... to get the n_rela for each hh")
    processed_hh_df = determine_n_rela_for_each_hh(hh_df.copy(), hhsz, final_conditonals[f"{HH_TAG}-counts"])
    assert processed_hh_df[HHID].nunique() == len(hh_df), "Processed hh df must have same hhid as original hh df"
    
    rela_count_cols = [f"n_{rela}" for rela in EXPECTED_RELATIONSHIPS]
    expected_n_pp = processed_hh_df[rela_count_cols].sum().sum()
    
    evidences_store = {HH_TAG: processed_hh_df}
    for relationships in RELA_BY_LEVELS:
        # Doing the relationship in this level
        for rela in relationships:
            # print(f"Sampling {rela}...")
            for prev_rela in BACK_CONNECTIONS[rela]:
                # print(f"Sampling {rela} from {prev_rela}...")
                rela_results = []
                if prev_rela not in evidences_store:
                    # Somehow this area does not have the prev rela
                    continue
                conditional = final_conditonals[f"{prev_rela}-{rela}"]
                evidences = evidences_store[prev_rela].drop(columns=[TARGET_ID, SYN_COUNT_COL], errors='ignore')
                check = (evidences[f"n_{rela}"] > 0)
                evidences[SYN_COUNT_COL] = evidences[f"n_{rela}"]
                if f"n_{prev_rela}" in evidences.columns:
                    check = check & (evidences[f"n_{prev_rela}"] > 0)
                evidences = evidences[check].copy() # only care where we need to sample
                if len(evidences) == 0:
                    continue
                final_sampled_df, _, _, _ = direct_sample_from_conditional(conditional.copy(), evidences, can_sample_all=False)
                # map to the n_count
                sub_n_count = evidences[rela_count_cols + [HHID]]
                final_sampled_df = final_sampled_df.merge(sub_n_count, on=HHID, how="inner")
                assert len(final_sampled_df) == evidences[SYN_COUNT_COL].sum(), "Must be able to sample all, no err accepted"

                rela_results.append(final_sampled_df)
            if len(rela_results) == 0:
                continue
            evidences_store[rela] = pd.concat(rela_results, ignore_index=True)
            assert len(evidences_store[rela]) == processed_hh_df[f"n_{rela}"].sum(), f"Must be able to sample all for this {rela}"

    concat_pp_ls = []
    for rela, df in evidences_store.items():
        if rela == HH_TAG:
            continue
        df = df.drop(columns=[TARGET_ID, SYN_COUNT_COL]+rela_count_cols, errors='ignore')
        df[relationship] = rela
        df = df.rename(columns={f"{rela}_{att}": att for att in PP_ATTS})
        concat_pp_ls.append(df)
    final_pp = pd.concat(concat_pp_ls, ignore_index=True)
    assert len(final_pp) == expected_n_pp, f"Final syn pp must match from HH, but got {len(final_pp)} vs {expected_n_pp}"
    return final_pp