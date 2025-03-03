import pandas as pd


def convert_full_to_marg_count(
    full_pop: pd.DataFrame, filter_ls: list[str] = [], zone_field: str = "zone_id"
) -> pd.DataFrame:
    assert zone_field in full_pop.columns
    cols = [x for x in full_pop.columns if x not in filter_ls + [zone_field]]
    ls_temp_hold = []
    for att in cols:
        full_pop[att] = full_pop[att].astype(str)
        temp_hold = full_pop.groupby(zone_field)[att].value_counts().unstack().fillna(0)
        temp_hold.columns = [(temp_hold.columns.name, x) for x in temp_hold.columns]
        temp_hold = temp_hold.astype(int)
        ls_temp_hold.append(temp_hold)
    marg_new_raw = pd.concat(ls_temp_hold, axis=1)
    convert_marg_dict = {
        com_col: marg_new_raw[com_col] for com_col in marg_new_raw.columns
    }
    convert_marg_dict[(zone_field, None)] = marg_new_raw.index
    new_marg_hh = pd.DataFrame(convert_marg_dict)
    ls_drop_m = list(
        new_marg_hh.columns[new_marg_hh.columns.get_level_values(0).isin([zone_field])]
    )
    new_marg_hh = new_marg_hh.drop(columns=ls_drop_m)
    return new_marg_hh