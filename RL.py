import random
import pandas as pd
import numpy as np


# thinking of creating an env object so it better to control, creating new env will be like initialize again

class Env:
    def __init__(self, df, order):
        # parameters of RL
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.9 #exploration heavy
        self.eps = 1000

        # init 
        self.df = df
        self.order = order
        self.order.insert(0, 'start')
        self.states = df.columns
        self.policy = {'start': {'start': 0}}
        self.seq = {}
        for att in self.states:
            possible_val = pd.unique(df[att])
            self.policy[att] = {}
            for val in possible_val:
                self.policy[att][val] = 0
        

    def choose_action(self, next_state_i):
        next_state = self.order[next_state_i]
        # return the action (can be random) based on your curren state
        actions_space = self.policy[next_state]
        action = None
        if random.uniform(0, 1) < self.epsilon:
            action = np.random.choice(list(actions_space.keys()))
        else:
            action = max(actions_space, key=actions_space.get)
        return action


    def reward(self):
        check = None
        for att in self.seq:
            val = self.seq[att]
            if check is None:
                check = (self.df[att] == val)
            else:
                check &= (self.df[att] == val)
        # This assumes that there will always be False result
        matching_num = len(check) - check.value_counts()[False]
        return matching_num


    def step(self, current_state_i, action):
        # return the next state, reward, done (and maybe any extra)
        current_state = self.order[current_state_i]
        next_state_i = current_state_i + 1
        next_state = self.order[next_state_i]
        self.seq[next_state] = action
        re = self.reward()
        done = next_state_i == (len(self.order) - 1)
        return re, done
        

    def update_Qtable(self, current_state_i, current_value, reward):
        current_state = self.order[current_state_i]
        next_state = self.order[current_state_i + 1]
        predict = self.policy[current_state][current_value]
        target = reward + self.gamma * max(self.policy[next_state].values())
        self.policy[current_state][current_value] = self.policy[current_state][current_value] + self.alpha * (target - predict)


    def RL_trainning(self, eps, max_train):
        for j in range(eps):
            print(f"START TRAINNING EP {j}")
            final = False
            l = 0
            while not final:
                l += 1
                cur_i = 0
                done = False
                cur_val = 'start'
                while not done:
                    action = self.choose_action(cur_i + 1)
                    reward, done = self.step(cur_i, action)
                    # print(cur_val, action, reward)
                    self.update_Qtable(cur_i, cur_val, reward)
                    cur_i += 1
                    cur_val = action
                    if cur_i == len(self.order):
                        print(f"Got seq {self.seq}")
                        final = reward != 0
                if l >= max_train: 
                    print ("FINISH EARLY")
                    break
            print(f"FINISH TRAINING EP {j}")
        

if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'PERSINC', 'CARLICENCE', 'SEX']
    
    # import data
    original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    df = original_df[ATTRIBUTES].dropna()
    # seed_df = df.sample(n = 10).copy()

    env_test = Env(df, ATTRIBUTES.copy())
    
    env_test.RL_trainning(2, 1000)
    print(env_test.policy)


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