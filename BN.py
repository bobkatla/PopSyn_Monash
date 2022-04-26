import pandas as pd
import matplotlib.pyplot as plt
import bnlearn as bn
from pgmpy.sampling import GibbsSampling, BayesianModelSampling
from checker import SRMSE


def sampling(model, n=1000, type='forward', init_state=None):
    '''
    This will define different sampling methods using the BN_network (model)
    The network is pgmpy.model.BayesianNetwork
    '''
    sampling_df = None

    if type == 'gibbs':
        # Using GibbsSampling, noted here it initial/seed is random by default
        gibbs_chain = GibbsSampling(model)
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
                sampling_df = infer_model.forward_sample(size=n)
            case 'rejection':
                sampling_df = infer_model.rejection_sample(size=n)
            case 'likelihood':
                sampling_df = infer_model.likelihood_weighted_sample(size=n)
            case _:
                print("Wrong type name")
        
    return sampling_df


def BN_training(df, sample_rate, sample=True):
    N = df.shape[0]
    one_percent = int(N/100)
    # It is noted that with small samples, cannot ebtablish the edges
    seed_df = df.sample(n = sample_rate * one_percent).copy()
    # Learn the DAG in data using Bayesian structure learning:
    DAG = bn.structure_learning.fit(seed_df, methodtype='hc', scoretype='bic', verbose=0)
    # Remove insignificant edges
    # DAG = bn.independence_test(DAG, seed_df, alpha=0.05, prune=True, verbose=0)
    bn.plot(DAG)
    # Parameter learning on the user-defined DAG and input data using Bayes to estimate the CPTs
    model = bn.parameter_learning.fit(DAG, seed_df, methodtype='bayes', verbose=0)
    if sample:
        # Sampling
        sampling_df = sampling(model['model'], n = N*2, type = 'gibbs')
        return sampling_df
    else: return None


def plot_SRMSE_bayes(orginal):
    N = orginal.shape[0]
    X = []
    Y = []

    for i in range(1, 100):
        X.append(i)

        sampling_df = BN_training(orginal, i)
        
        # Calculate the SRMSE
        Y.append(SRMSE(df, sampling_df))
        # print(X, Y)

    plt.plot(X, Y)
    plt.xlabel('Percentages of sampling rate')
    plt.ylabel('SRMSE')
    plt.savefig('./img_data/BN_SRMSE.png')
    # plt.show()


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'CARLICENCE', 'SEX', 'PERSINC', 'DWELLTYPE', 'TOTALVEHS']
    
    # import data
    p_original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    # Only have record of the main person (the person that did the survey)
    p_self_df = p_original_df[p_original_df['RELATIONSHIP']=='Self']
    h_original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/H_VISTA12_16_SA1_V1.csv")

    orignal_df = pd.merge(p_self_df, h_original_df, on=['HHID'])
    df = orignal_df[ATTRIBUTES].dropna()

    # TODO: change the grouping to match the paper

    sampling_df = BN_training(df, sample_rate=99)
    print(SRMSE(df, sampling_df))

    # plot_SRMSE_bayes(df)

    # TODO: for the missing att (they are not in the graph) they can be sampled from distribution - I think?
