import pandas as pd
import numpy as np


# thinking of creating an env object so it better to control, creating new env will be like initialize again

class Env:
    def __init__(self, df, order):
        self.states = df.columns
        width = len(self.states)
        max_length = 0
        self.actions = {}
        for att in self.states:
            possible_val = pd.unique(df[att])
            if len(possible_val) > max_length: max_length = len(possible_val) 
            self.actions[att] = possible_val
        self.env = np.zeros((max_length, width))

    def choose_action(current_state):
        # return the action (can be random) based on your curren state
        NotImplemented


    def state_transition(current_state, action):
        # base on the current state and action, return the next state
        NotImplemented


    def reward(new_state):
        # return the reward based on the new state
        NotImplemented

    def step(action):
        # return the next state, reward, done (and maybe any extra)
        NotImplemented

    def update_Qtable(current_state, action, reward, next_state):
        NotImplemented


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'PERSINC', 'CARLICENCE', 'SEX']
    
    # import data
    original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    df = original_df[ATTRIBUTES].dropna()
    # seed_df = df.sample(n = 10).copy()

    # parameters
    alpha = 0.1
    gamma = 0.6
    epsilon = 0.9 #exploration heavy
    eps = 1000

    env_test = Env(df, [])
    print(env_test.actions)
    print(env_test.env)
    print(env_test.states)


'''

Random thought put here:

state trans prob as a square matrix between each att to see what would be a good next att for the sequence
then we have the action after making the first state 
this can be better by considering the heterogeinity with the sum-up dostribution and even households
the goal would be to create a record that exist, reward would be the number of existing record
Multi-dimensional actions
Hierarchical reinforcement learning (HRL) can decompose a task into several sub-tasks and solve each job with a sub-model which will be more potent than solving the entire task with one model
https://www.mdpi.com/2073-8994/13/8/1335/htm
https://ieeexplore.ieee.org/document/5967381
'''