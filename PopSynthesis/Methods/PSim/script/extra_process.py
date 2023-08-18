import pandas as pd


def main():
    old_df = pd.read_csv("synthetic_2021_HH.csv")
    ls_sa1 = old_df[["SA1", "hhsize"]]
    counts = ls_sa1.value_counts()
    print(counts[20301103401])


if __name__ == "__main__":
    main()