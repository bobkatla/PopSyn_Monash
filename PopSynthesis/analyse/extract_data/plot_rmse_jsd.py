# plotting_eval_rmse_jsd.py
# Requirements: pandas, matplotlib, seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ---------- CONFIG ----------
RMSE_CSV = Path(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\combined_fin_rmse_records.csv")
JSD_CSV  = Path(r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\combined_fin_jsd_records.csv")

# Font size settings for better readability
TITLE_FONTSIZE = 20
AXIS_LABEL_FONTSIZE = 18
LEGEND_FONTSIZE = 16
LEGEND_TITLE_FONTSIZE = 16
TICK_FONTSIZE = 14

# Method naming/order (SAA → IPF → WGAN → BN)
method_order = [
    "SAA_BN_pool",
    "SAA_seed_addzero",
    "SAA_seed_misszero",
    "IPF_normal",
    "IPF_fromBN",
    "WGAN_hhsz",
    "BN_hhsz",
]
method_map = {
    "saa_BN_pool":       "SAA_BN_pool",
    "saa_seed_addzero":  "SAA_seed_addzero",
    "saa_seed_misszero": "SAA_seed_misszero",
    "ipf_normal":        "IPF_normal",
    "ipf_bn":            "IPF_fromBN",   # rename to match requested label
    "wgan":              "WGAN_hhsz",    # rename to match requested label
    "bn":                "BN_hhsz",
}

# Attribute order
att_order = ["hhsize", "totalvehs", "hhinc", "dwelltype", "owndwell"]

# ---------- HELPERS ----------
def _bar_with_minmax(ax, df, x_col, y_col, min_col, max_col,
                     att_order, method_order, title, y_label,
                     annotate=False, y_top=None, shift_frac=0.25, bold=True):
    """Grouped barplot with manual min-max whiskers and optional annotations."""
    # Draw bars
    sns.barplot(
        data=df, x=x_col, y=y_col, hue="method_run",
        ci=None, order=att_order, hue_order=method_order, ax=ax
    )

    # Compute geometry for whiskers + labels
    n_hues = len(method_order)
    total_width = 0.8
    single_width = total_width / n_hues

    # Make a stable ordering lookups
    att_positions = {att: i for i, att in enumerate(att_order)}
    method_positions = {m: i for i, m in enumerate(method_order)}

    for _, row in df.iterrows():
        att = row[x_col]
        meth = row["method_run"]
        x_base = att_positions[att]
        hue_index = method_positions[meth]
        offset = (hue_index - (n_hues - 1) / 2) * single_width
        x = x_base + offset

        # Min–max vertical line
        ax.plot([x, x], [row[min_col], row[max_col]], color="black", linewidth=1.5)

        if annotate:
            y_val = row[y_col]
            # Place a bit to the left to avoid overlap with the vertical min–max line
            x_text = x - single_width * shift_frac
            if (y_top is not None) and (y_val > y_top):
                y_text = y_top - 0.8  # keep inside the plot near the top
            else:
                y_text = y_val + 0.3
            ax.text(
                x_text, y_text, f"{y_val:.1f}",
                ha="center", va="bottom", rotation=90,
                fontsize=10, fontweight=("bold" if bold else "normal")  # Slightly larger font
            )

    ax.set_title(title, fontsize=TITLE_FONTSIZE, pad=20)
    ax.set_xlabel("Attribute", fontsize=AXIS_LABEL_FONTSIZE)
    ax.set_ylabel(y_label, fontsize=AXIS_LABEL_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    if y_top is not None:
        ax.set_ylim(0, y_top)
    ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left",
              fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    plt.tight_layout()


def summarize_rmse(rmse_csv: Path) -> pd.DataFrame:
    """Summarize RMSE: average across 'state' first, then across runs for each (method, attribute)."""
    df = pd.read_csv(rmse_csv)
    df.columns = [c.strip().replace("'", "") for c in df.columns]

    # Melt runs
    value_cols = [c for c in df.columns if c.startswith("run_")]
    long = df.melt(
        id_vars=["att", "state", "method_run"],
        value_vars=value_cols,
        var_name="run",
        value_name="rmse"
    )

    # Standardize method names
    long["method_run"] = long["method_run"].map(method_map)

    # 1) Average across states for each (method, att, run)
    by_state = (
        long.groupby(["method_run", "att", "run"], as_index=False)
            .agg(mean_rmse=("rmse", "mean"))
    )

    # 2) Then summarize across runs: mean + min + max (these are run-to-run stats)
    summ = (
        by_state.groupby(["method_run", "att"], as_index=False)
                .agg(mean_rmse=("mean_rmse", "mean"),
                     min_rmse=("mean_rmse", "min"),
                     max_rmse=("mean_rmse", "max"))
    )
    # Clean attributes and set orders
    summ["att"] = summ["att"].str.replace("'", "")
    summ["method_run"] = pd.Categorical(summ["method_run"], categories=method_order, ordered=True)
    summ = summ.sort_values(["att", "method_run"]).reset_index(drop=True)
    return summ


def summarize_jsd(jsd_csv: Path) -> pd.DataFrame:
    """Summarize JSD per (method, attribute): mean/min/max across runs."""
    df = pd.read_csv(jsd_csv)
    df.columns = [c.strip().replace("'", "") for c in df.columns]

    value_cols = [c for c in df.columns if c.startswith("run_")]
    long = df.melt(
        id_vars=["att", "method_run"],
        value_vars=value_cols,
        var_name="run",
        value_name="jsd"
    )
    long["method_run"] = long["method_run"].map(method_map)

    by_run = (
        long.groupby(["method_run", "att", "run"], as_index=False)
            .agg(mean_jsd=("jsd", "mean"))
    )
    summ = (
        by_run.groupby(["method_run", "att"], as_index=False)
              .agg(mean_jsd=("mean_jsd", "mean"),
                   min_jsd=("mean_jsd", "min"),
                   max_jsd=("mean_jsd", "max"))
    )
    summ["att"] = summ["att"].str.replace("'", "")
    summ["method_run"] = pd.Categorical(summ["method_run"], categories=method_order, ordered=True)
    summ = summ.sort_values(["att", "method_run"]).reset_index(drop=True)
    return summ


# ---------- RUN ----------
if __name__ == "__main__":
    sns.set_style("whitegrid")

    # --- RMSE summary + plot ---
    rmse_summary = summarize_rmse(RMSE_CSV)

    fig, ax = plt.subplots(figsize=(14, 8))  # Larger figure
    _bar_with_minmax(
        ax=ax,
        df=rmse_summary,
        x_col="att",
        y_col="mean_rmse",
        min_col="min_rmse",
        max_col="max_rmse",
        att_order=att_order,
        method_order=method_order,
        title="RMSE by Attribute",
        y_label="RMSE",
        annotate=True,         # show bold numbers
        y_top=15,              # 0-15 axis
        shift_frac=0.25,
        bold=True
    )
    plt.savefig("rmse_mean_minmax_0to15.png", dpi=300, bbox_inches='tight')
    plt.show()

    # --- JSD summary + plot (no numbers) ---
    jsd_summary = summarize_jsd(JSD_CSV)

    fig, ax = plt.subplots(figsize=(14, 8))  # Larger figure
    _bar_with_minmax(
        ax=ax,
        df=jsd_summary,
        x_col="att",
        y_col="mean_jsd",
        min_col="min_jsd",
        max_col="max_jsd",
        att_order=att_order,
        method_order=method_order,
        title="JSD by Attribute",
        y_label="JSD",
        annotate=False,        # <- no numbers for JSD
        y_top=None
    )
    plt.savefig("jsd_mean_minmax.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("Saved: rmse_mean_minmax_0to15.png, jsd_mean_minmax.png")
