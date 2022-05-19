from unittest import result
import pandas as pd
import matplotlib.pyplot as plt
import bnlearn as bn
from pgmpy.sampling import GibbsSampling, BayesianModelSampling
from checker import update_SRMSE
from multiprocessing import Process, Lock, Array


def sampling(model, n=1000, type='forward', init_state=None, show_progress=True):
    '''
    This will define different sampling methods using the BN_network (model)
    The network is pgmpy.model.BayesianNetwork
    '''
    sampling_df = None

    if type == 'gibbs':
        # Using GibbsSampling, noted here it initial/seed is random by default
        gibbs_chain = GibbsSampling(model)
        # For Gibbs, always show progress, cannot change this
        sampling_df = gibbs_chain.sample(size=n, start_state=init_state)

        # This is needed as the result of Gibb is set to only int
        sampling_df = sampling_df.astype('object')
        # This part is updating the cardinality of Gibb with their corresponding label
        for att in gibbs_chain.variables:
            pos_ref = model.get_cpds(att).state_names[att]
            for i in sampling_df.index:
                sampling_df.at[i, att] = pos_ref[sampling_df.at[i, att]]
    else:
        # Using other sampling methods
        infer_model = BayesianModelSampling(model)
        match type:
            case 'forward':
                sampling_df = infer_model.forward_sample(size=n, show_progress=show_progress)
            case 'rejection':
                sampling_df = infer_model.rejection_sample(size=n, show_progress=show_progress)
            case 'likelihood':
                sampling_df = infer_model.likelihood_weighted_sample(size=n, show_progress=show_progress)
            case _:
                print("Wrong sampling type name")
        
    return sampling_df


def make_black_list_for_root(ls_atts, root_att):
    # Return the list of edges for black list to set up root node
    return [(att, root_att) for att in ls_atts if att != root_att]

def contain_all_nodes(DAG):
    adjmat = DAG['adjmat']
    for att in adjmat:
        ls_target = list(adjmat.T[att])
        ls_source = list(adjmat[att])
        if True not in ls_target and True not in ls_source:
            return False
    return True


def BN_training(df, sample_rate, ite_check=50,sample=True, plotting=False, sampling_type='forward', struct_method='hc', para_method='bayes', black_ls = None, show_progress=True):
    N = df.shape[0]
    one_percent = int(N/100)
    # It is noted that with small samples, cannot ebtablish the edges
    seed_df = df.sample(n = sample_rate * one_percent).copy()

    cannot_create_DAG = True
    for _ in range(ite_check):
        # Learn the DAG in data using Bayesian structure learning:
        # DAG = bn.structure_learning.fit(seed_df, methodtype='cl', root_node='AGEGROUP', verbose=0)
        DAG = bn.structure_learning.fit(seed_df, methodtype=struct_method, scoretype='bic', verbose=0, black_list=black_ls, bw_list_method='edges')
        # Remove insignificant edges
        DAG = bn.independence_test(DAG, seed_df, alpha=0.05, prune=True, verbose=0)
        if contain_all_nodes(DAG):
            cannot_create_DAG = False
            break
    if cannot_create_DAG:
        print(f"After {ite_check} tries, cannot create the DAG of {len(df.columns)} nodes for the sample rate of {sample_rate}")
        return None
    if plotting: bn.plot(DAG)
    # Parameter learning on the user-defined DAG and input data using Bayes to estimate the CPTs
    model = bn.parameter_learning.fit(DAG, seed_df, methodtype=para_method, verbose=0)
    if sample:
        # Sampling
        sampling_df = sampling(model['model'], n = N*2, type = sampling_type, show_progress=show_progress)
        return sampling_df
    else: return None

def multi_thread_f(df, s_rate, re_arr, l, bl):
    print(f"START THREAD FOR SAMPLE RATE {s_rate}")
    # NOTE: this can be increased for more objective results
    check_time = 10
    re = 0
    for _ in range(check_time):
        # NOTE: change the para for BN here, putting rejection now cause' don't wanna see the progress bar
        sampling_df = BN_training(df=df, sample_rate=s_rate, sampling_type='rejection', black_ls=bl, show_progress=False)
        er_score = update_SRMSE(df, sampling_df) if sampling_df is not None else None
        if er_score is None: check_time -= 1
        else: re += er_score
    re = -1 if check_time <= 0 else re/check_time
    # Calculate the SRMSE
    l.acquire()
    try:
        # NOTE: this is depends on the range we put the array, it should be same size but accessing the index is diff
        re_arr[s_rate-1] = re
        print(f"DONE {s_rate}")
    finally:
        l.release()


def plot_SRMSE_bayes(original, root_node = None):
    bl = None
    if root_node:
        atts = original.columns
        bl = make_black_list_for_root(atts, root_node)
        
    # Maybe will not make this fixed like this
    X = range(1, 100)

    results = Array('d', X)
    lock = Lock()
    hold_p = []

    print("START THE PROCESS OF PLOTTING SRMSE")
    for i in X:
        p = Process(target=multi_thread_f, args=(original, i, results, lock, bl))
        p.start()
        hold_p.append(p)
    for p in hold_p: p.join()

    print("DONE ALL, PLOTTING NOW")
    Y = results[:]
    plt.plot(X, Y)
    plt.xlabel('Percentages of sampling rate')
    plt.ylabel('SRMSE')
    plt.savefig('./img_data/BN_SRMSE_reject_final.png')
    plt.show()


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'CARLICENCE', 'SEX', 'PERSINC', 'DWELLTYPE', 'TOTALVEHS']
    
    # import data
    p_original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    # Only have record of the main person (the person that did the survey)
    p_self_df = p_original_df[p_original_df['RELATIONSHIP']=='Self']
    h_original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/H_VISTA12_16_SA1_V1.csv")

    orignal_df = pd.merge(p_self_df, h_original_df, on=['HHID'])
    df = orignal_df[ATTRIBUTES].dropna()

    make_like_paper = True
    if make_like_paper:
        df.loc[df['TOTALVEHS'] == 0, 'TOTALVEHS'] = 'NO'
        df.loc[df['TOTALVEHS'] != 'NO', 'TOTALVEHS'] = 'YES'

        df.loc[df['CARLICENCE'] == 'No Car Licence', 'CARLICENCE'] = 'NO'
        df.loc[df['CARLICENCE'] != 'NO', 'CARLICENCE'] = 'YES'

    # sampling_df = BN_training(df, sample_rate=20, sample=True, plotting=True, sampling_type='gibbs')
    # print(update_SRMSE(df, sampling_df))
    plot_SRMSE_bayes(df, root_node='AGEGROUP')
    # b_ls = make_black_list_for_root(ATTRIBUTES, root_att='AGEGROUP')
    # BN_training(df, sample_rate=15, sample=False, plotting=False, sampling_type='gibbs', black_ls=b_ls, show_progress=False)
