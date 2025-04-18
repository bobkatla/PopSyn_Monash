import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.distance import jensenshannon

# ------------------
# Step 4: Evaluate Against Seed Distributions
# ------------------
def compute_normalized_distributions(df_households, df_individuals):
    distributions = {}

    def norm_counts(series):
        return series.value_counts(normalize=True).sort_index()

    distributions["household_hsize"] = norm_counts(df_households["hsize"])
    distributions["household_hh_type"] = norm_counts(df_households["hh_type"])
    distributions["household_n_cars"] = norm_counts(df_households["n_cars"])
    distributions["individual_age"] = norm_counts(df_individuals["age"])
    distributions["individual_gender"] = norm_counts(df_individuals["gender"])
    distributions["individual_marital"] = norm_counts(df_individuals["marital"])
    distributions["individual_employment"] = norm_counts(df_individuals["employment"])
    distributions["individual_license"] = norm_counts(df_individuals["license"])

    return distributions


def jsd(p, q):
    """Jensen-Shannon Divergence"""
    all_keys = sorted(set(p.index).union(set(q.index)))
    p_aligned = p.reindex(all_keys, fill_value=0)
    q_aligned = q.reindex(all_keys, fill_value=0)
    return jensenshannon(p_aligned, q_aligned) ** 2


def plot_distributions(seed_dist, synth_dist, title):
    all_keys = sorted(set(seed_dist.index).union(set(synth_dist.index)))
    p = seed_dist.reindex(all_keys, fill_value=0)
    q = synth_dist.reindex(all_keys, fill_value=0)

    x = np.arange(len(all_keys))
    plt.figure(figsize=(10, 4))
    plt.bar(x - 0.2, p.values, width=0.4, label="Seed")
    plt.bar(x + 0.2, q.values, width=0.4, label="Synthetic")
    plt.xticks(x, all_keys, rotation=45)
    plt.title(f"{title}\nJSD: {jsd(p, q):.4f}")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def main():
    curr_folder = Path(__file__).parent
    data_folder = curr_folder / "data"

    print("\n[STEP 4] Loading seed and synthetic data...")
    seed_households = pd.read_csv(data_folder / "seed_households.csv")
    seed_individuals = pd.read_csv(data_folder / "seed_individuals.csv")
    synth_households = pd.read_csv(data_folder / "synthetic_households.csv")
    synth_individuals = pd.read_csv(data_folder / "synthetic_individuals.csv")

    print("\n[STEP 4] Computing distributions...")
    seed_dist = compute_normalized_distributions(seed_households, seed_individuals)
    synth_dist = compute_normalized_distributions(synth_households, synth_individuals)

    for key in seed_dist:
        print(f"\nVisualizing distribution: {key}")
        plot_distributions(seed_dist[key], synth_dist[key], title=key.replace("_", ": ").title())


if __name__ == "__main__":
    main()