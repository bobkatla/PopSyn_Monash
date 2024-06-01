import pandas as pd
import numpy as np


ATTRIBUTES = ['AGEGROUP', 'HHSIZE', 'CARLICENCE', 'SEX', 'PERSINC', 'DWELLTYPE', 'TOTALVEHS', 'CW_ADHHWGT_SA3']


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


def generate_control_files_from_population(pop_df, to_csv=False, csv_name_control='flat_con.csv', csv_name_marg='flat_marg.csv'):
    # NOTE: this is mainly use in case we have a whole population data, or testing using the seed data
    d_marg = {
        'total': len(pop_df)
    }
    d_con = {
        'target': [],
        'geography': [],
        'seed_table': [],
        'importance': [],
        'att': [],
        'state': [],
        'control_field': [],
        'expression': [],
    }
    for att in pop_df.columns:
        data_seri = pop_df[att]
        att_freq = data_seri.value_counts(sort=False).sort_index()
        for val in att_freq.index: 
            tot_name = f'{att}__{val}'
            d_marg[tot_name] = att_freq[val]
            
            d_con['target'].append('NA')
            d_con['geography'].append('NA')
            d_con['seed_table'].append('flat_table')
            d_con['importance'].append(1000)
            d_con['att'].append(att)
            d_con['state'].append(val)
            d_con['control_field'].append(tot_name)
            d_con['expression'].append(f"flat_table.{att} == '{val}'")
    df_marg = pd.DataFrame(data=d_marg, index=[0])
    df_con = pd.DataFrame(data=d_con)

    if to_csv:
        df_marg.to_csv(f'./data/data_processed_here/{csv_name_marg}', index=False)
        df_con.to_csv(f'./data/data_processed_here/{csv_name_control}', index=False)

    return df_marg, df_con


if __name__ == "__main__":
    flatten_df = flatten_data_self_generate(ATTRIBUTES, to_csv=False)
    # Because I want to include the weights and transform it (now just integri to the nearest so not that correct)
    # the csv saving would be for later
    print(pd.unique(flatten_df['CARLICENCE']))

    final_data = None

    want_short = False

    if want_short:
        final_data = flatten_df.drop(columns='CW_ADHHWGT_SA3')
    else:
        raw_non_weight = flatten_df.drop(columns='CW_ADHHWGT_SA3').to_numpy()
        weights = flatten_df['CW_ADHHWGT_SA3'].to_numpy()
        cols = flatten_df.drop(columns='CW_ADHHWGT_SA3').columns

        final_raw = []
        for i in range(len(weights)):
            d = raw_non_weight[i]
            w = int(float(weights[i]))
            final_raw.append(np.repeat([d], w, axis=0))
        
        processed_final = np.concatenate(final_raw, axis=0)
        final_data = pd.DataFrame(processed_final, columns=cols)
    final_data.to_csv('./data/data_processed_here/flatten_seed_data.csv', index=False)
    a = generate_control_files_from_population(final_data, to_csv=True)
    