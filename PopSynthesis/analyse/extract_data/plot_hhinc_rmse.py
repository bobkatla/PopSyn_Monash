# hhinc_rmse_plots_simple.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ------------------ Config ------------------
in_csv = r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\combined_fin_rmse_records.csv"
out_full = "hhinc_rmse_full.png"
out_zoom = "hhinc_rmse_zoom.png"

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
ordered_methods = [
    "SAA_BN_pool",
    "SAA_seed_addzero",
    "SAA_seed_misszero",
    "IPF_normal",
    "IPF_fromBN",
    "WGAN_hhsz",
    "BN_hhsz",
]
color_code = {
    "SAA_BN_pool": "#1f77b4",
    "SAA_seed_addzero": "#ff7f0e",
    "SAA_seed_misszero": "#2ca02c",
    "IPF_normal": "#d62728",
    "IPF_fromBN": "#9467bd",
    "WGAN_hhsz": "#8c564b",
    "BN_hhsz": "#e377c2",
}

# ------------------ Load & filter ------------------
df = pd.read_csv(in_csv)

# Drop any index-like cols
for c in list(df.columns):
    if c.lower().startswith("unnamed") or c.lower() in {"index"}:
        df = df.drop(columns=c)

# Keep only hhinc
df = df[df["att"].str.contains("hhinc", case=False)]

# Clean quotes/whitespace
df["state"] = df["state"].astype(str).str.strip().str.strip("'").str.strip('"')

# Exclude Nil income
df = df[~df["state"].str.contains("nil", case=False)]

# ------------------ Sort categories ------------------
def parse_income_sort_key(s: str) -> float:
    s = s.lower().replace(",", "").strip()
    if "negative" in s: return -1
    if "zero" in s: return 0
    if "+" in s: return float(s.replace("+",""))
    if "-" in s:
        return float(s.split("-")[0])
    return float("inf")

df["sort_key"] = df["state"].apply(parse_income_sort_key)
df = df.sort_values(["sort_key","state"], kind="mergesort")

income_categories = df["state"].unique().tolist()
x = np.arange(len(income_categories))

def pretty_label(s: str) -> str:
    if s.lower().startswith("negative"): return "Negative income"
    if s.lower().startswith("zero"): return "Zero income"
    if "+" in s: return f"${s.replace('+','')}+"
    if "-" in s:
        lo,hi = s.split("-")
        return f"${int(lo):,}–{int(hi):,}"
    return s
income_labels = [pretty_label(cat) for cat in income_categories]

# ------------------ Map methods ------------------
df["method_label"] = df["method_run"].map(method_map)
run_cols = [c for c in df.columns if c.startswith("run_")]

# ------------------ Plot function ------------------
def plot_rmse(filename, ylim=None, title_suffix=""):
    plt.figure(figsize=(14, 8))  # Larger figure for better readability
    for method in ordered_methods:
        sub = df[df["method_label"] == method]
        if sub.empty: 
            continue
        runs = sub[run_cols]
        mean_vals = runs.mean(axis=1).values
        min_vals = runs.min(axis=1).values
        max_vals = runs.max(axis=1).values
        idx = np.arange(len(income_categories))
        plt.plot(idx, mean_vals, marker='o', color=color_code[method], label=method, markersize=6, linewidth=2)
        plt.fill_between(idx, min_vals, max_vals, alpha=0.18, color=color_code[method])
    plt.ylim(ylim)
    plt.xticks(x, income_labels, rotation=45, ha='right', fontsize=TICK_FONTSIZE)
    plt.yticks(fontsize=TICK_FONTSIZE)
    plt.ylabel("RMSE (Household Income)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.title("RMSE across Household Income categories" + title_suffix, fontsize=TITLE_FONTSIZE, pad=20)
    plt.legend(ncol=3, fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()

# ------------------ Make plots ------------------
plot_rmse(out_full, ylim=None, title_suffix="")
plot_rmse(out_zoom, ylim=(0,50), title_suffix=" (Zoomed 0-50)")
