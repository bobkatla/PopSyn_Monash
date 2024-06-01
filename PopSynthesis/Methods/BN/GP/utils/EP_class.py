"""
This will be the main one to create base abstract for later object-oritented python
"""

from abc import ABC, abstractmethod
import pandas as pd
import networkx as nx
import pylab as plt
from PopSynthesis.Methods.BN.utils.learn_BN import learn_struct_BN_score, sample_BN
from PopSynthesis.Methods.BN.TBN.test_GA import partition_df


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
        NotImplemented
    
    def cross_over(self):
        return
    
    def annihilation(self):
        return
    
    def eval_func(self, sol):
        return 0
    

from PopSynthesis.Methods.IPF.src.data_process import get_marg_val_from_full
from PopSynthesis.Benchmark.CompareFullPop.utils import sampling_from_full_pop, realise_full_pop_based_on_weight
from PopSynthesis.Benchmark.CompareFullPop.compare import SRMSE_based_on_counts
from PopSynthesis.Benchmark.CompareCensus.compare import compare_RMS_census
from pgmpy.factors.discrete import State
import numpy as np
import math
import random
    

class EP_for_full_pop_creator(EP_base):
    def __init__(self, loc_data) -> None:
        self.seed_df_hh = pd.read_csv(loc_data + "H_sample.csv")
        self.seed_df_pp = pd.read_csv(loc_data + "P_sample.csv")

        # NOTE: for now, only HH data
        to_drop_cols = ["hh_num", "hhid", "SA1", "SA2", "SA3", "SA4"]
        self.seed_df_hh = self.seed_df_hh.drop(columns=to_drop_cols)
        self.full_df_hh = realise_full_pop_based_on_weight(self.seed_df_hh, weight_col="wdhhwgt_sa3")
        self.atts = self.full_df_hh.columns
        self.N = len(self.full_df_hh)

        self.marginals = get_marg_val_from_full(self.full_df_hh)
        self.norm_probs_census = self.get_norm_vals_for_atts(self.marginals)


    def get_norm_vals_for_atts(self, marg_processed):
        ls_probs_atts = {}
        for att in self.atts:
            vals = marg_processed[att]
            norm_vals = vals / vals.sum()
            ls_probs_atts[att] = norm_vals
        return ls_probs_atts
    

    def cross_entropy(self, dist_target, dist_check):
        # They will be 2 same size pd.Serires for the same atts, 
        result = 0
        # NOTE: dist check will be the base, no 0 there
        for k in dist_check.index:
            val_target = dist_target[k] if k in dist_target.index else 0
            result += val_target * math.log(dist_check[k], 2) # in bits, change to e if want nats
        return -result


    def loop_check(self, range_sample=np.linspace(0.01, 0.1, 10)):
        results = []
        results_rmse = []
        for rate in range_sample:
            print(f"PROCESSING rate {rate}")
            seed_df = sampling_from_full_pop(self.full_df_hh, rate=rate)
            print("Doing the EP now")
            syn_pop = self.run_EP(seed_data=seed_df)
            print("Calculate SRMSE now")
            SRMSE = SRMSE_based_on_counts(self.full_df_hh.value_counts(), syn_pop.value_counts())
            results.append(SRMSE)
            results_rmse.append(compare_RMS_census(self.marginals, get_marg_val_from_full(syn_pop)))
            print(f"Done rate {rate} with score of {SRMSE}")
        return results, results_rmse
    

    def run_EP(self, seed_data, num_pop=8, random_rate=0.2, num_gen=20, crossover_time=2):
        self.set_BN(seed_data=seed_data)
        self.init_first_gen()

        assert random_rate < 1
        num_random = max(int(num_pop * random_rate), 1) if random_rate > 0 else 0

        check_RMSD=[] 
        check_SRMSE=[]

        counter = 0
        marg_count = self.full_df_hh.value_counts()
        while counter < num_gen:
            print(f"RUNNING FOR GEN {counter}")
            #TODO: defo can optimise the work on eval solution, will work on it later
            # pick the best solution
            best_sol = self.ls_sols[0]

            ######### TEST
            test_score = best_sol[1]
            print("Best at the moment", test_score)
            check_RMSD.append(test_score)
            srmse = SRMSE_based_on_counts(marg_count, best_sol[0].value_counts())
            print(f"SMRSE: {srmse}")
            check_SRMSE.append(srmse)
            ########## TEST

            # Mutate offspring, mutate all using the BN of the best, best one will get mutate more
            print(f"GA - gen {counter}: mutation")
            for i, sol in enumerate(self.ls_sols):
                self.mutation(
                    indi= sol[0],
                    partition_rate=0.2,
                    num_keep_atts=int(len(self.atts)/3 + 1), # A more robust way/ dynamic to declare this
                    num_child=(num_pop-i) # this is to make sure that the population size is correct
                )

            # Producing offsprings (reproduction/ crossover)
            print(f"GA - gen {counter}: crossover")
            for _ in range(crossover_time):
                best_pa_sol = [self.ls_sols[0], self.ls_sols[1]]
                self.cross_over(
                    pa1=best_pa_sol[0][0],
                    pa2=best_pa_sol[1][0],
                    partition_rate=0.4
                )
            # Select the "best" for next round (or replacement)
            # Having some random solutions from the worst to increase diversity
            print(f"GA - gen {counter}: selection")
            self.annihilation(num_random, num_pop)
            counter += 1
        
        # Pick the final solution, can create BN as well
        result = self.ls_sols[0][0]

        print(check_RMSD) 
        print(check_SRMSE)
        np.save('../output/GA_results_RMSD.npy_XL', np.array(check_RMSD))
        np.save('../output/GA_results_SRMSE.npy_XL', np.array(check_SRMSE))

        return result


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


    def insert_new_sol(self, new_indi):
        # This is used to input new offspring into the list, sorted
        score_new = self.eval_func(new_indi)
        for i, sol in enumerate(self.ls_sols):
            if score_new <= sol[1]:
                self.ls_sols.insert(i, (new_indi, score_new, ))
                break  
            if i == len(self.ls_sols) - 1:
                # it has the lowest score
                self.ls_sols.append((new_indi, score_new, )) 
                break


    def init_first_gen(self):
        first_gen = sample_BN(self.BN_model, self.N)
        score = self.eval_func(first_gen)
        self.ls_sols = [(first_gen, score, )]


    def best_fit_atts(self, syn_pop, num_att=1):
        final_re = {}
        marg_processed_syn_pop = get_marg_val_from_full(syn_pop)
        census_dist = self.norm_probs_census
        syn_dist = self.get_norm_vals_for_atts(marg_processed_syn_pop)
        for att in self.atts:
            # Have to make the syn_dist as target cause' it may have 0, if we swap there will be err as log(0) is undefined
            final_re[att] = self.cross_entropy(syn_dist[att], census_dist[att])
        sort_result = sorted(final_re.items(), key=lambda item: item[1])
        assert num_att <= len(sort_result)
        return [sort_result[i][0] for i in range(num_att)]


    def mutation(self, indi, partition_rate, num_keep_atts, num_child):
        # find the best fit atts
        ls_best_atts = self.best_fit_atts(indi, num_att=num_keep_atts)
        ls_atts = list(indi.columns)

        # create children from mutation
        for _ in range(num_child):
            # partition randomly based on the ratio
            mut_part, rest_part = partition_df(indi, frac=partition_rate)
            # BN inference for the rest of them atts in mutation, this has to be rejection sample
            # create new df based on mutation part
            final_list_df = [rest_part]
            count_combine_best_att = mut_part.groupby(ls_best_atts)[ls_atts[0]].count()
            for state_combine in count_combine_best_att.index:
                # create evidence
                evidence = [State(att, state) for att, state in zip(ls_best_atts,state_combine)]
                num_to_sample = count_combine_best_att[state_combine]
                new_rec = sample_BN(
                    model=self.BN_model, 
                    n=num_to_sample, # NOTE: can try further test of instead of having only 1, we can create more and select the best of mutation (maybe most different one?)
                    typeOf='rejection',
                    evidence=evidence, show_progress=False)
                final_list_df.append(new_rec)
            # combine again
            final_child = pd.concat(final_list_df, ignore_index=True)
            print("Inserting mutated child")
            self.insert_new_sol(final_child)


    def cross_over(self, pa1, pa2, partition_rate):
        # partition randomly based on the ratio for pa1
        swap_pa1, keep_pa1 = partition_df(pa1, frac=partition_rate)
        # partition randomly based on the ratio for pa2
        swap_pa2, keep_pa2 = partition_df(pa2, frac=partition_rate)
        # swap
        offspring1 = pd.concat([keep_pa1, swap_pa2], ignore_index=True)
        offspring2 = pd.concat([keep_pa2, swap_pa1], ignore_index=True)
        self.insert_new_sol(offspring1)
        self.insert_new_sol(offspring2)


    def annihilation(self, num_random, num_pop):
        num_best = num_pop - num_random
        worst_solutions = self.ls_sols[num_best:]
        random_solutions = random.sample(worst_solutions, k=num_random)
        solutions = self.ls_sols[:num_best]
        solutions.extend(random_solutions)
        self.ls_sols = solutions


    def eval_func(self, sol): 
        # NOTE: trying to minimise this
        marg_processed_syn_pop = get_marg_val_from_full(sol)
        return compare_RMS_census(self.marginals, marg_processed_syn_pop)


def test():
    data_loc = "../data/basics/"
    output_loc = "../output/"
    min_rate, max_rate, tot = 0.1, 1, 10
    
    EP_creator = EP_for_full_pop_creator(data_loc)
    results, results_rmse = EP_creator.loop_check(range_sample=np.linspace(min_rate, max_rate, tot))

    np.save(f'{output_loc}/result_EP_{min_rate}_{max_rate}.npy', np.asarray(results))
    print(results)
    np.save(f'{output_loc}/result_EP_{min_rate}_{max_rate}.npy', np.asarray(results_rmse))
    print(results_rmse)


if __name__ == "__main__":
    test()