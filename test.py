from cmath import sqrt
import pandas as pd
import bnlearn as bn
from math import sqrt

# import data
original_df = pd.read_csv("./data/VISTA_2012_16_v1_SA1_CSV/P_VISTA12_16_SA1_V1.csv")
df = original_df[['AGEGROUP', 'PERSINC', 'SEX', 'CARLICENCE']]
print(df.shape)

# Learn the DAG in data using Bayesian structure learning:
DAG = bn.structure_learning.fit(df, root_node='AGEGROUP', methodtype='ex', scoretype='bic')

# Adjacency matrix
# print(DAG['adjmat'])

# Plot
# G = bn.plot(DAG)

# Parameter learning on the user-defined DAG and input data using Bayes to estimate the CPTs
model = bn.parameter_learning.fit(DAG, df, methodtype='bayes')
# bn.print_CPD(model)

sampling_df = bn.sampling(model, n=46562)

# print(sampling_df)