import random
import pandas as pd
import numpy as np
from multiprocessing import Process, Lock, Array
from BN import SRMSE


# thinking of creating an env object so it better to control, creating new env will be like initialize again

class Env:
    def __init__(self, df, order):
        # parameters of RL
        self.alpha = 0.1
        self.gamma = 0.6
        self.epsilon = 0.8 #exploration heavy

        # init 
        self.df = df
        self.order = order
        self.order.insert(0, 'start')
        self.states = df.columns
        self.policy = {}
        self.seq = {}
        for i, state in enumerate(self.order):
            self.policy[state] = {}
            if i == (len(self.order) - 1):
                self.policy[state]['finish'] = 0
                break
            possible_actions = pd.unique(df[self.order[i+1]])
            for action in possible_actions:
                self.policy[state][action] = 0
        

    def choose_action(self, current_state_i):
        current_state = self.order[current_state_i]
        # return the action (can be random) based on your curren state
        actions_space = self.policy[current_state]
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
        return len(self.seq) * matching_num


    def step(self, current_state_i, action):
        # return the next state, reward, done (and maybe any extra)
        next_state = self.order[current_state_i + 1]
        self.seq[next_state] = action
        re = self.reward()
        done = current_state_i == (len(self.order) - 2)
        return re, done
        

    def update_Qtable(self, current_state_i, action, reward):
        current_state = self.order[current_state_i]
        next_state = self.order[current_state_i + 1]
        # print(current_state, action)
        predict = self.policy[current_state][action]
        target = reward + self.gamma * max(self.policy[next_state].values())
        self.policy[current_state][action] += self.alpha * (target - predict)


    def RL_trainning(self, eps, max_train):
        for j in range(eps):
            # print(f"START TRAINNING EP {j}")
            final = False
            l = 0
            while not final:
                self.seq = {}
                l += 1
                cur_i = 0
                done = False
                while not done:
                    action = self.choose_action(cur_i)
                    reward, done = self.step(cur_i, action)
                    # print(cur_val, action, reward)
                    self.update_Qtable(cur_i, action, reward)
                    cur_i += 1
                    if cur_i == (len(self.order)-1):
                        # print(f"Got seq {self.seq}")
                        final = reward != 0
                        if final: print(f"I REACH THE GOAL at step {l}")
                if l >= max_train: 
                    print ("FINISH EARLY")
                    break
            # print(f"FINISH TRAINING EP {j}")


    def sampling(self, n):
        hold = {}
        for i, state in enumerate(self.order):
            if i == (len(self.order) - 1):
                break
            next_state = self.order[i+1]
            hold[next_state] = ([], [])
            for action in self.policy[state]:
                hold[next_state][0].append(action)
                hold[next_state][1].append(self.policy[state][action])
            ls_values = hold[next_state][1]
            sum_val = sum(ls_values)
            for i, val in enumerate(ls_values):
                ls_values[i] = val / sum_val

        # picking time
        ls_samples = []
        for _ in range(n):
            ls_pick = {}
            for att in hold:
                ls_pick[att] = [np.random.choice(a=hold[att][0], p=hold[att][1])]
            new_sample = pd.DataFrame(ls_pick)
            ls_samples.append(new_sample)
        result = pd.concat(ls_samples)
        return result


def calculate_SRMSE_given_rate(sample_rate, df, order):
    num_train = 8000
    max_step = 5000
    N = df.shape[0]
    seed_df = df.sample(n = (int(N/100)*sample_rate)).copy()
    e = Env(seed_df, order.copy())
    e.RL_trainning(num_train, max_step)
    predict_df = e.sampling(N*2)
    return SRMSE(df, predict_df)


def multithreading_func(l, i, df, order, results_arr):
    print(f"START {i}")
    err_cal = calculate_SRMSE_given_rate(i+1, df, order)
    l.acquire()
    try:
        results_arr[i] = err_cal
        print(f"DONE {i}")
    finally:
        l.release()


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'CARLICENCE', 'SEX']
    
    # import data
    original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    df = original_df[ATTRIBUTES].dropna()
    # print(df.shape)

    # try to do multi threading now
    min_num_percen = 0
    max_num_percen = 5
    results = Array('d', range(max_num_percen))
    lock = Lock()
    hold_p = []

    for num in range(min_num_percen, max_num_percen):
        p = Process(target=multithreading_func, args=(lock, num, df, ATTRIBUTES, results))
        p.start()
        hold_p.append(p)

    for p in hold_p:
        p.join()

    fin = results[:]
    print(fin)

    txt_file = open("RL_results.txt", "w")
    for ele in fin:
        txt_file.write(str(ele) + ", ")
    txt_file.close()

    print("DONEEEE")

    


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