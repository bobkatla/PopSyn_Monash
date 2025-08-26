import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Config
# -------------------------------
FILE_PATH = r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\combined_fin_neg_inc.csv"  # update path if needed

# Method display names
method_map = {
    "saa_BN_pool":      "SAA_BN_pool",
    "saa_seed_addzero": "SAA_seed_addzero",
    "ipf_bn":           "IPF_fromBN",
}

# Font size settings
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14

key_methods = ["saa_BN_pool", "saa_seed_addzero", "ipf_bn"]

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(FILE_PATH, header=[0, 1], index_col=0)
census = pd.to_numeric(df[("hhinc", "Negative income")], errors="coerce")

def get_method_mean(method_name: str) -> pd.Series:
    return df[method_name].apply(pd.to_numeric, errors="coerce").mean(axis=1)

# -------------------------------
# Plot: side-by-side hexbin
# -------------------------------
fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

hb = None
x = census.values
x_max = float(np.nanmax(x)) if np.isfinite(x).any() else 1.0
line_handle = None

for ax, method in zip(axes, key_methods):
    y = get_method_mean(method).values
    hb = ax.hexbin(x, y, gridsize=35, mincnt=1, cmap="viridis")

    # 45-degree reference line with label (legend handle saved only once)
    line, = ax.plot([0, x_max], [0, x_max], "k--", linewidth=1,
                    label="Census = Prediction")
    if line_handle is None:
        line_handle = line  # save handle for shared legend

    ax.set_title(method_map.get(method, method), fontsize=TITLE_FONTSIZE)
    ax.set_xlabel("Census count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel("Predicted count", fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', labelsize=TICK_FONTSIZE)
    ax.grid(True, alpha=0.3)

# External colorbar
cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
cbar = fig.colorbar(hb, cax=cax)
cbar.set_label("Zones per bin", fontsize=AXIS_LABEL_FONTSIZE)
cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

# Shared legend below plots
fig.legend(handles=[line_handle],
           labels=["Census = Prediction"],
           loc="lower center",
           bbox_to_anchor=(0.5, -0.02),
           fontsize=LEGEND_FONTSIZE,
           frameon=False)

plt.subplots_adjust(left=0.07, right=0.90, bottom=0.18, top=0.90)
plt.savefig("neg_inc_hexbin.png", dpi=300, bbox_inches='tight')
plt.show()
