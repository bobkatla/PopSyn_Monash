import pandas as pd


ATTRIBUTES = ['AGEGROUP', 'HHSIZE', 'CARLICENCE', 'SEX', 'PERSINC', 'DWELLTYPE', 'TOTALVEHS']


def flatten_data_self_generate(ls_atts, make_like_paper=False, to_csv=False, csv_name='flatten_seed_data.csv'):
    # import data
    p_original_df = pd.read_csv("./data/source/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    # Only have record of the main person (the person that did the survey)
    p_self_df = p_original_df[p_original_df['RELATIONSHIP']=='Self']
    h_original_df = pd.read_csv("./data/source/VISTA_2012_16_v1_SA1_CSV/H_VISTA12_16_SA1_V1.csv")

    orignal_df = pd.merge(p_self_df, h_original_df, on=['HHID'])
    df = orignal_df[ls_atts].dropna()

    if make_like_paper:
        df.loc[df['TOTALVEHS'] == 0, 'TOTALVEHS'] = 'NO'
        df.loc[df['TOTALVEHS'] != 'NO', 'TOTALVEHS'] = 'YES'

        df.loc[df['CARLICENCE'] == 'No Car Licence', 'CARLICENCE'] = 'NO'
        df.loc[df['CARLICENCE'] != 'NO', 'CARLICENCE'] = 'YES'
    
    if to_csv:
        df.to_csv(f'./data/data_processed_here/{csv_name}', index=False)
    
    return df


def generate_control_files_from_population(pop_df, to_csv=False, csv_name_control='flat_tot.csv', csv_name_marg='flat_marg.csv'):
    # NOTE: this is mainly use in case we have a whole population data, or testing using the seed data
    d_marg = {
        'total': len(pop_df)
    }
    d_con = {
        'seed_table': [],
        'att': [],
        'state': [],
        'tot_name': [],
        'expression': [],
    }
    for att in pop_df.columns:
        data_seri = pop_df[att]
        att_freq = data_seri.value_counts(sort=False).sort_index()
        for val in att_freq.index: 
            tot_name = f'{att}__{val}'
            d_marg[tot_name] = att_freq[val]
            
            d_con['seed_table'].append('flat_table')
            d_con['att'].append(att)
            d_con['state'].append(val)
            d_con['tot_name'].append(tot_name)
            d_con['expression'].append(f"flat_table.{att} == '{val}'")
    df_marg = pd.DataFrame(data=d_marg, index=[0])
    df_con = pd.DataFrame(data=d_con)

    if to_csv:
        df_marg.to_csv(f'./data/data_processed_here/{csv_name_marg}', index=False)
        df_con.to_csv(f'./data/data_processed_here/{csv_name_control}', index=False)

    return df_marg, df_con


if __name__ == "__main__":
    flatten_df = flatten_data_self_generate(ATTRIBUTES, to_csv=True)
    a = generate_control_files_from_population(flatten_df, to_csv=True)
