import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, MaximumLikelihoodEstimator
from pgmpy.sampling import BayesianModelSampling
from pgmpy.factors.discrete import State

from PopSynthesis.Benchmark.checker import total_RMSE_flat, update_SRMSE
from PopSynthesis.Methods.BN.newBN.process_data import get_hh_census, get_hh_seed
from PopSynthesis.Methods.BN.TBN.utils import learn_struct_BN_score


def sampling_data(model, census, ls_lga):
    inference = BayesianModelSampling(model)
    result = []
    for lga, tot in zip(census['LGA'], census['Total_Total']):
        print(lga, tot)
        if lga in ls_lga:
            final_syn = inference.rejection_sample(
                size=tot,
                evidence=[State('LGA', lga)]
                )
            result.append(final_syn)
    final_df = pd.concat(result)
    return final_df


def main():
    atts = [
        'homeLGA',
        'hhsize',
        'totalvehs',
        'dwelltype',
        'hhinc',
        'owndwell',
        'wdhhwgt_LGA'
    ]
    df_hh_seed = get_hh_seed(atts)
    df_hh_census = get_hh_census()

    model = learn_struct_BN_score(df_hh_seed, show_struct=True)
    para_learn = BayesianEstimator(
            model=model,
            data=df_hh_seed
        )
    ls_CPDs = para_learn.get_parameters(
        prior_type='BDeu'
    )
    model.add_cpds(*ls_CPDs)

    final_df = sampling_data(model, df_hh_census, list(df_hh_seed['LGA'].unique()))
    final_df.to_csv('BN_Melbourne_hh_2021.csv', index=False)


if __name__ == "__main__":
    main()