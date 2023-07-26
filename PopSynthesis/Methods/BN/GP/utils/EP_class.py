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


def test():
    data_loc = "../data/"
    # controls_loc = "../controls/"
    sample = pd.read_csv(data_loc + "flatten_seed_data.csv")
    marginal = pd.read_csv(data_loc + "flat_marg.csv")
    controls = pd.read_csv(data_loc + "flat_con.csv")
    test_creator = EP_BN_creator(
        sample=sample,
        marginal=marginal,
        controls=controls
    )

    test_creator.mutation()
    print(test_creator.ls_sols)


if __name__ == "__main__":
    test()