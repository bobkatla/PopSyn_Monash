import pandas as pd


data_location = "../../../Generator_data/data/data_processed_here/"


def convert_to_prob(d1_arr):
    new_arr = []
    sum_arr = sum(d1_arr)
    for val in d1_arr:
        new_arr.append(val / sum_arr)
    return new_arr


def get_state_names(con_df):
    # return the dict of att and their states (order is important)
    state_names = {}
    ls_atts = con_df['att'].unique()
    for att in ls_atts:
        df_att = con_df[con_df['att']==att]
        state_names[att] = list(df_att['state'])
    return state_names


def cal_count_states(con_df, tot_df):
    #  calculate the prior of each 
    state_names = get_state_names(con_df)
    final_count = {}
    for att in state_names:
        ls_states = state_names[att]
        ls_count = [] # note that this would match with the ls_states
        for state in ls_states:
            tot_name = con_df[(con_df['att']==att) & (con_df['state']==state)]['tot_name']
            ls_count.append(tot_df[tot_name.iloc[0]].iloc[0])
        ls_prob = convert_to_prob(ls_count)
        final_count[att] = {
            'states': ls_states,
            'count': ls_count,
            'prob': ls_prob
        }
    return final_count


def get_prior_counts(DAG, con_df, tot_df):
    NotImplemented


def main():
    con_df = pd.read_csv(data_location + "flat_con.csv")
    tot_df = pd.read_csv(data_location + "flat_marg.csv")
    seed_data = pd.read_csv(data_location + "flatten_seed_data.csv")
    a = cal_count_states(con_df, tot_df)
    print(a)


if __name__ == '__main__':
    main()