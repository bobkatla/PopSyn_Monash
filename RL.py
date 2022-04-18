import pandas as pd
import numpy as np


# thinking of creating an env object so it better to control, creating new env will be like initialize again

class Env:
    def __init__(self, df):
        self.states = df.columns
        width = len(self.states)
        max_length = 0
        self.actions = {}
        for att in self.states:
            possible_val = pd.unique(df[att])
            if len(possible_val) > max_length: max_length = len(possible_val) 
            self.actions[att] = possible_val
        self.env = np.zeros((max_length, width))

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
    # seed_df = df.sample(n = 10).copy()

    env_test = Env(df)
    print(env_test.actions)
    print(env_test.env)
    print(env_test.states)
