# plot_missing_combinations.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# --- Config ---
csv_path = r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\combined_fin_missing_percen.csv"
out_png = "missing_combinations_boxplot.png"

# Font size settings for better readability
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16
LEGEND_TITLE_FONTSIZE = 16
TICK_FONTSIZE = 14
method_map = {
    "saa_BN_pool":       "SAA_BN_pool",
    "saa_seed_addzero":  "SAA_seed_addzero",
    "saa_seed_misszero": "SAA_seed_misszero",
    "ipf_normal":        "IPF_normal",
    "ipf_bn":            "IPF_fromBN",
    "wgan":              "WGAN_hhsz",
    "bn":                "BN_hhsz",
}

# Paper order
ordered_cols = [
    "SAA_BN_pool",
    "SAA_seed_addzero",
    "SAA_seed_misszero",
    "IPF_normal",
    "IPF_fromBN",
    "WGAN_hhsz",
    "BN_hhsz",
]

# Consistent colors
color_code = {
    "SAA_BN_pool": "#1f77b4",
    "SAA_seed_addzero": "#ff7f0e",
    "SAA_seed_misszero": "#2ca02c",
    "IPF_normal": "#d62728",
    "IPF_fromBN": "#9467bd",
    "WGAN_hhsz": "#8c564b",
    "BN_hhsz": "#e377c2",
}

# --- Load ---
df = pd.read_csv(csv_path)

# Drop run index column if present (e.g., 'Unnamed: 0' or 'run')
for c in list(df.columns):
    if c.lower().startswith("unnamed") or c.lower() in {"run", "index"}:
        df = df.drop(columns=[c])

# Rename and order
df = df.rename(columns=method_map)
df = df[ordered_cols]

# --- Plot ---
plt.figure(figsize=(13, 8))  # Larger figure for better readability
box = df.boxplot(grid=False, patch_artist=True, return_type='dict')

# Color each box
for patch, col in zip(box['boxes'], ordered_cols):
    patch.set_facecolor(color_code[col])

# Overlay mean markers
means = df.mean()
positions = np.arange(1, len(ordered_cols) + 1)
plt.scatter(positions, means, color="black", marker="o", zorder=3, label="Mean", s=50)

plt.ylabel("Missing % of valid combinations from seed data", fontsize=AXIS_LABEL_FONTSIZE)
plt.title("Comparison of methods across 10 reruns", fontsize=TITLE_FONTSIZE, pad=20)
plt.xticks(rotation=30, fontsize=TICK_FONTSIZE)
plt.yticks(fontsize=TICK_FONTSIZE)
plt.legend(fontsize=LEGEND_FONTSIZE)

# Save and show
Path(out_png).parent.mkdir(parents=True, exist_ok=True)
plt.tight_layout()
plt.savefig(out_png, dpi=300, bbox_inches='tight')
plt.show()

# --- Print exact means ---
print("\nMean missing % across 10 reruns (lower = better):")
for col in ordered_cols:
    print(f"{col:16s}: {means[col]:.6f}")
