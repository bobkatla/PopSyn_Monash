from cmath import sqrt
from tabnanny import verbose
import pandas as pd
import bnlearn as bn
from math import sqrt
from pgmpy.sampling import GibbsSampling
import itertools

ATTRIBUTES = ['AGEGROUP', 'PERSINC', 'SEX', 'CARLICENCE']

# This assume that both df have the same collumns/attributes
def SRMSE(actual, pred, attributes):
    total_att = 1
    full_list = {}

    # Get the possible values for each att
    for att in attributes:
        possible_values = actual[att].unique()
        total_att *= len(possible_values)
        full_list[att] = possible_values

    # Generate all the possible combinations
    keys, values = zip(*full_list.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    hold = 0
    for instance in combinations:
        for att in attributes:
            NotImplemented

if __name__ == "__main__":
    # import data
    original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
    df = original_df[ATTRIBUTES]
    # print(df.shape)

    # Learn the DAG in data using Bayesian structure learning:
    DAG = bn.structure_learning.fit(df, root_node='AGEGROUP', methodtype='ex', scoretype='bic', verbose=0)

    # Adjacency matrix
    # print(DAG['adjmat'])

    # Plot
    # G = bn.plot(DAG)

    # Parameter learning on the user-defined DAG and input data using Bayes to estimate the CPTs
    model = bn.parameter_learning.fit(DAG, df, methodtype='bayes', verbose=0)
    # bn.print_CPD(model)

    sampling_df = bn.sampling(model, n=46562, verbose=0)

    # print(sampling_df)
    # a = (df['SEX'] == "Male") & (df['CARLICENCE']=="No Car Licence")
    # print(a.value_counts()[True])

    # print(df['SEX'].unique())

    # print(SRMSE(df, sampling_df, ATTRIBUTES))