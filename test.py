import synthpop.ipf.ipf as ipf
import synthpop.ipu.ipu as ipu
import pandas as pd

if __name__ == '__main__':
    marginal_midx = pd.MultiIndex.from_tuples(
        [('cat_owner', 'yes'),
         ('cat_owner', 'no'),
         ('car_color', 'blue'),
         ('car_color', 'red'),
         ('car_color', 'green')])
    marginals = pd.Series([60, 40, 50, 30, 20], index=marginal_midx)
    joint_dist_midx = pd.MultiIndex.from_product(
        [('yes', 'no'), ('blue', 'red', 'green')],
        names=['cat_owner', 'car_color'])
    joint_dist = pd.Series([8, 4, 2, 5, 3, 2], index=joint_dist_midx)


    constraints, iterations = ipf.calculate_constraints(marginals, joint_dist)
    print(constraints)
    print(iterations)

# from synthpop.recipes.starter2 import Starter
# from synthpop.synthesizer import synthesize_all, enable_logging 
# import os

# def synthesize_county(county):
#     starter = Starter(os.environ["CENSUS"], "CO", county)
#     synthetic_population = synthesize_all(starter)
#     return synthetic_population
    
# synthesize_county('Gilpin County')