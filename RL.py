import pandas as pd


# thinking of creating an env object so it better to control, creating new env will be like initialize again

class Env:
    def __init__(self):
        NotImplemented
        

    def action_maker(current_state):
        # return the action (can be random) based on your curren state
        NotImplemented


    def state_transition(current_state, action):
        # base on the current state and action, return the next state
        NotImplemented


    def reward(new_state):
        # return the reward based on the new state
        NotImplemented


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'PERSINC', 'SEX', 'CARLICENCE']
    
    # import data
    original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    df = original_df[ATTRIBUTES].dropna()

    print(df)
