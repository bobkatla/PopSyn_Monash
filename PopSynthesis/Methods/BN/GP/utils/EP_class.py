"""
This will be the main one to create base abstract for later object-oritented python
"""

from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
import pylab as plt
from pgmpy.models import BayesianNetwork
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, sample_BN
from PopSynthesis.Methods.BN.TBN.test_GA import mutation as base_mut, crossover as base_cross
from PopSynthesis.Methods.BN.utils.data_process import sample_from_full_pop


class EP_base(ABC):
    @abstractmethod
    def init_first_gen(self): pass
    @abstractmethod
    def mutation(self): pass
    @abstractmethod
    def cross_over(self): pass
    @abstractmethod
    def annihilation(self): pass
    @abstractmethod
    def eval_func(self): pass


class EP_BN_creator(EP_base):
    def __init__(self, sample:pd.DataFrame, marginal:pd.DataFrame, controls:pd.DataFrame) -> None:
        super().__init__()
        self.sample = sample
        self.marginal = marginal
        self.controls = controls

        self.N = 10000
        self.set_BN()
        self.init_first_gen() #should always be sorted

    def set_BN(self, default=True):
        # TODO: make it more advanced with customisation
        model = None
        if default:
            model = learn_struct_BN_score(self.sample)
            # this will learning using MLE
            model.fit(self.sample)
        else:
            NotImplemented
        self.BN_model = model

    def plot_BN(self):
        nx.draw_circular(self.BN_model ,with_labels=True)
        plt.show()
        
    def init_first_gen(self):
        first_gen = sample_BN(self.BN_model, self.N)
        score = self.eval_func(first_gen)
        self.ls_sols = [(first_gen, score, )] 

    def insert_into_sorted(self, new_indi):
        score_new = self.eval_func(new_indi)
        for i, sol in enumerate(self.ls_sols):
            if score_new >= sol[1]:
                self.ls_sols.insert(i, (new_indi, score_new, ))
                break  
            if i == len(self.ls_sols) - 1:
                # it has the lowest score
                self.ls_sols.append((new_indi, score_new, )) 

    def mutation(self):
        # pick the worst results to mutate
        indi_to_mutate = self.ls_sols[-1][0]
        print(indi_to_mutate)
        mutated_childs = base_mut(
            indi=indi_to_mutate, 
            BN_model=self.BN_model,
            con_df=self.controls,
            tot_df=self.marginal
            )
        for child in mutated_childs:
            self.insert_into_sorted(child)
    
    def cross_over(self):
        return
    
    def annihilation(self):
        return
    
    def eval_func(self, sol):
        return 0
    

from PopSynthesis.Methods.IPF.src.data_process import get_marg_val_from_full
from PopSynthesis.Benchmark.CompareFullPop.utils import sampling_from_full_pop, realise_full_pop_based_on_weight
from PopSynthesis.Benchmark.CompareFullPop.compare import SRMSE_based_on_counts
import numpy as np
import math
    

class EP_for_full_pop_creator(EP_base):
    def __init__(self, loc_data) -> None:
        self.seed_df_hh = pd.read_csv(loc_data + "H_sample.csv")
        self.seed_df_pp = pd.read_csv(loc_data + "P_sample.csv")

        # NOTE: for now, only HH data
        to_drop_cols = ["hh_num", "hhid", "SA1", "SA2", "SA3", "SA4"]
        self.seed_df_hh = self.seed_df_hh.drop(columns=to_drop_cols)
        self.full_df_hh = realise_full_pop_based_on_weight(self.seed_df_hh, weight_col="wdhhwgt_sa3")
        self.N = len(self.full_df_hh)

        self.marginals = get_marg_val_from_full(self.full_df_hh)


    def loop_check(self,  range_sample=np.linspace(0.01, 0.1, 10)):
        results = []
        for rate in range_sample:
            print(f"PROCESSING rate {rate}")
            seed_df = sampling_from_full_pop(self.full_df_hh, rate=rate)
            print("Doing the EP now")
            syn_pop = self.run_EP(seed_data=seed_df)
            print("Calculate SRMSE now")
            SRMSE = SRMSE_based_on_counts(self.full_df_hh.value_counts(), syn_pop.value_counts())
            results.append(SRMSE)
            print(f"Done rate {rate} with score of {SRMSE}")
        return results
    
    
    def run_EP(self, seed_data, num_pop=10, random_rate=0.2, num_gen=1000, err_converg=math.inf, crossover_time=3):
        self.set_BN()
        self.init_first_gen()
        
        return None


    def set_BN(self, seed_data, default=True):
        # TODO: make it more advanced with customisation
        model = None
        if default:
            model = learn_struct_BN_score(seed_data)
            # this will learning using MLE
            model.fit(seed_data)
        else:
            NotImplemented
        self.BN_model = model


    def init_first_gen(self):
        first_gen = sample_BN(self.BN_model, self.N)
        score = self.eval_func(first_gen)
        self.ls_sols = [(first_gen, score, )]


    def mutation(self): pass

    def cross_over(self): pass

    def annihilation(self): pass

    def eval_func(self, gen): 
        return 0


def test():
    data_loc = "../data/flatten/"
    # controls_loc = "../controls/"
    full_pop = pd.read_csv(data_loc + "full_population_2021.csv")
    sample = sample_from_full_pop(full_pop=full_pop, sample_rate=1)
    marginal = pd.read_csv(data_loc + "marginal_2021.csv")
    controls = pd.read_csv(data_loc + "controls_2021.csv")

    # min_rate, max_rate, tot = 0.0001, 0.001, 10
        # results = self.EP_run(loc_data=loc_data, range_sample)
        # data = np.asarray(results)
        # np.save(f'{output_loc}/result_EP_{min_rate}_{max_rate}.npy', data)
        # print(results)


if __name__ == "__main__":
    test()