import pandas as pd


def main():
    # Convert PopSyn into optimisation problem, so we will need to know how to get the constraints of PP and HH
    # Optimise those goals, we can divide into 2 big goals, each goals will have smaller goals
    # Verify the resultsn of optimisation
    H_df = pd.read_csv("../data/H_sample.csv")
    P_df = pd.read_csv("../data/P_sample.csv")
    # Build BN with both


if __name__ == "__main__":
    main()