from cmath import sqrt
from re import L
from tabnanny import verbose
import pandas as pd
import itertools
from math import sqrt
import matplotlib.pyplot as plt
from pyparsing import one_of
import bnlearn as bn
from pgmpy.sampling import GibbsSampling, BayesianModelSampling


def SRMSE(actual, pred):
    '''
    This calculate the SRMSE for 2 pandas.dataframe based on the list of their attributes
    The actual has to have the same or more columns than pred
    This will compare only the one that is in pred's colls
    '''

    total_att = 1
    full_list = {}
    attributes = pred.columns

    # Get the possible values for each att
    for att in attributes:
        possible_values = actual[att].unique()
        total_att *= len(possible_values)
        full_list[att] = possible_values

    # Generate all the possible combinations
    keys, values = zip(*full_list.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    #calculate
    hold = 0
    for instance in combinations:
        # the 2 would be None or not None at the same time
        check_actual = None
        check_pred = None
        for att in attributes:
            if check_actual is not None:
                check_actual &= (actual[att] == instance[att])
                check_pred &= (pred[att] == instance[att])
            else:
                check_actual = (actual[att] == instance[att])
                check_pred = (pred[att] == instance[att])
        # This assumes that there will always be False result
        freq_actual = 1 - ((check_actual.value_counts()[False]) / len(check_actual))
        freq_pred = 1 - ((check_pred.value_counts()[False]) / len(check_pred))
        hold += (freq_actual - freq_pred)**2
    
    result = sqrt(hold * total_att)

    return result


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


def plot_SRMSE_bayes(orginal):
    N = orginal.shape[0]
    X = []
    Y = []

    one_percent = int(N/100)
    i = 0
    for num in range(one_percent, N, one_percent):
        i += 1
        X.append(i)
        print(i)
        # It is noted that with small samples, cannot ebtablish the edges
        seed_df = df.sample(n = num).copy()
        # Learn the DAG in data using Bayesian structure learning:
        DAG = bn.structure_learning.fit(seed_df, methodtype='hc', scoretype='bic', verbose=0)
        # Remove insignificant edges
        DAG = bn.independence_test(DAG, seed_df, alpha=0.05, prune=True, verbose=0)
        # Parameter learning on the user-defined DAG and input data using Bayes to estimate the CPTs
        model = bn.parameter_learning.fit(DAG, seed_df, methodtype='bayes', verbose=0)
        # Sampling
        sampling_df = sampling(model['model'], n = N*2, type = 'gibbs')
        # Calculate the SRMSE
        Y.append(SRMSE(df, sampling_df))
        # print(X, Y)

    plt.plot(X, Y)
    plt.xlabel('Percentages of sampling rate')
    plt.ylabel('SRMSE')
    plt.savefig('./img_data/BN_SRMSE.png')
    # plt.show()


if __name__ == "__main__":
    ATTRIBUTES = ['AGEGROUP', 'PERSINC', 'SEX', 'CARLICENCE']
    
    # import data
    original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    df = original_df[ATTRIBUTES].dropna()

    plot_SRMSE_bayes(df)
    
    # TODO: for the missing att (they are not in the graph) they can be sampled from distribution - I think?
