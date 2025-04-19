"""Main run func to run the CSP with a given hh df"""


import pandas as pd
from PopSynthesis.Methods.CSP.run.run import run_csp


def load_configurations():
    """from const file, create a config dict"""
    NotImplemented


def main():
    hh_df = pd.read_csv()
    configs = load_configurations()
    # Run CSP with the given hh_df and configs

    resulted_pp = run_csp(hh_df, configs) # We must not change the hh
    print(resulted_pp)