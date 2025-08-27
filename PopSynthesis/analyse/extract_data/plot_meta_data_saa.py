import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
CSV_PATH = r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\big\fin_meta_results.csv"
SAA_METHODS = ["saa_BN_pool", "saa_seed_addzero", "saa_seed_misszero"]
ATTRIBUTES = ["hhinc", "hhsize", "dwelltype", "totalvehs", "owndwell"]
ZOOM_START = 6
ZOOM_END = 15

# Font size settings for better readability
TITLE_FONTSIZE = 24
AXIS_LABEL_FONTSIZE = 24
LEGEND_FONTSIZE = 26
LEGEND_TITLE_FONTSIZE = 18
TICK_FONTSIZE = 22
# -----------------------------
# Helpers
# -----------------------------
def pick_top_level(df_multi, top_name):
    """Return the Series for a given top-level MultiIndex column name."""
    matches = [
        c for c in df_multi.columns
        if isinstance(c, tuple) and str(c[0]).strip().lower() == top_name.lower()
    ]
    if not matches:
        raise KeyError(f"Could not find top-level column '{top_name}'.")
    return df_multi[matches[0]]

def parse_adjust_order(s):
    """Parse 'run_X_adjusted_Y' -> (run_idx, att_idx) as ints or (None, None)."""
    if not isinstance(s, str):
        return None, None
    m = re.search(r"run_(\d+)_adjusted_(\d+)", s)
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))

def normalize_attr_name(txt):
    """Map any adjusted_att string to one of our canonical attributes."""
    if txt is None:
        return None
    low = str(txt).lower()
    for a in ATTRIBUTES:
        if a in low:
            return a
    return txt  # fallback (unlikely)

def base_dataframe(csv_path):
    """Load CSV with two-row header and build a tidy frame for plotting."""
    df_multi = pd.read_csv(csv_path, header=[0, 1])

    df = pd.DataFrame({
        "mean": pd.to_numeric(pick_top_level(df_multi, "mean"), errors="coerce"),
        "adjusted_att": pick_top_level(df_multi, "adjusted_att"),
        "rerun": pd.to_numeric(pick_top_level(df_multi, "rerun"), errors="coerce"),
        "method_run": pick_top_level(df_multi, "method_run"),
        "adjust_order": pick_top_level(df_multi, "adjust_order"),
        "target_n_syn": pd.to_numeric(pick_top_level(df_multi, "target_n_syn"), errors="coerce"),
    })
    df = df.dropna(subset=["mean", "rerun", "method_run", "adjust_order"])

    run_att = df["adjust_order"].apply(parse_adjust_order)
    df["run_idx"] = [x[0] for x in run_att]
    df["att_idx"] = [x[1] for x in run_att]
    df = df.dropna(subset=["run_idx", "att_idx"])
    df["run_idx"] = df["run_idx"].astype(int)
    df["att_idx"] = df["att_idx"].astype(int)

    # loopback = 1..15, att_order = 1..5, step = 1..75
    df["loopback"] = df["run_idx"] + 1
    df["att_order"] = df["att_idx"] + 1
    df["step"] = df["run_idx"] * 5 + df["att_idx"] + 1

    df["method_run"] = df["method_run"].astype(str)
    df = df[df["method_run"].isin(SAA_METHODS)].copy()
    df["rerun"] = df["rerun"].astype("Int64")

    # Keep the original multi-index dataframe for per-attribute RMSE calculation
    return df_multi, df

def get_attribute_columns(df_multi, attribute):
    """Return the full list of (top, sub) columns for an attribute's states."""
    return [c for c in df_multi.columns if isinstance(c, tuple) and c[0] == attribute]

def compute_per_attribute_rmse_for_rows(df_multi, row_indexer):
    """
    For the selected rows (row_indexer), compute per-attribute RMSEs:
        RMSE_attr(row) = sqrt( mean_k ( value(row, state_k)^2 ) )
    Returns a DataFrame with columns = ATTRIBUTES, aligned to the filtered rows.
    """
    out = {}
    for att in ATTRIBUTES:
        cols = get_attribute_columns(df_multi, att)
        # if attribute missing (shouldn't be), fill with NaN
        if len(cols) == 0:
            out[att] = np.full(np.sum(row_indexer), np.nan)
            continue
        block = df_multi.loc[row_indexer, cols]
        # Convert to float
        block_vals = block.to_numpy(dtype=float)
        # RMSE over states for this row
        rmse_attr = np.sqrt(np.nanmean(np.square(block_vals), axis=1))
        out[att] = rmse_attr
    return pd.DataFrame(out, index=df_multi.index[row_indexer])

# -----------------------------
# Plotters already provided earlier (kept here)
# -----------------------------
def plot_rmse_per_loopback_mean(df):
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["mean"].mean().reset_index()
    plt.figure(figsize=(16, 8))  # Larger figure
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="mean")
        pivot = pivot.reindex(np.arange(1, 15 + 1))
        steps = pivot.index.to_numpy(int)
        mean_vals = pivot.mean(axis=1, skipna=True).to_numpy(float)
        min_vals = pivot.min(axis=1, skipna=True).to_numpy(float)
        max_vals = pivot.max(axis=1, skipna=True).to_numpy(float)
        plt.plot(steps, mean_vals, linewidth=3, label=f"{method}")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)
    plt.xlabel("Loopback", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("RMSE", fontsize=AXIS_LABEL_FONTSIZE)
    # plt.title("SAA RMSE per loopback", fontsize=TITLE_FONTSIZE, pad=20)
    plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("saa_rmse_per_loopback.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_target_per_loopback_mean(df):
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["target_n_syn"].mean().reset_index()
    plt.figure(figsize=(16, 8))  # Larger figure
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="target_n_syn")
        pivot = pivot.reindex(np.arange(1, 15 + 1))
        steps = pivot.index.to_numpy(int)
        mean_vals = pivot.mean(axis=1, skipna=True).to_numpy(float)
        min_vals = pivot.min(axis=1, skipna=True).to_numpy(float)
        max_vals = pivot.max(axis=1, skipna=True).to_numpy(float)
        plt.plot(steps, mean_vals, linewidth=3, label=f"{method}")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)
    plt.xlabel("Loopback", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Target number of synthetic agents", fontsize=AXIS_LABEL_FONTSIZE)
    # plt.title("Target synthetic population per loopback", fontsize=TITLE_FONTSIZE, pad=20)
    plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("saa_target_per_loopback.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_rmse_75_steps(df):
    plt.figure(figsize=(18, 8))  # Larger figure
    for method in SAA_METHODS:
        df_m = df[df["method_run"] == method]
        pivot = df_m.pivot_table(index="step", columns="rerun", values="mean")
        pivot = pivot.reindex(np.arange(1, 75 + 1))
        steps = pivot.index.to_numpy(int)
        mean_vals = pivot.mean(axis=1, skipna=True).to_numpy(float)
        min_vals = pivot.min(axis=1, skipna=True).to_numpy(float)
        max_vals = pivot.max(axis=1, skipna=True).to_numpy(float)
        plt.plot(steps, mean_vals, linewidth=3, label=f"{method}")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.18)
    for b in range(5, 75, 5):
        plt.axvline(b + 0.5, linestyle=":", linewidth=0.8)
    plt.xlabel("Adjustment step (1-75)", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("RMSE", fontsize=AXIS_LABEL_FONTSIZE)
    plt.title("SAA RMSE across 75 adjustment steps", fontsize=TITLE_FONTSIZE, pad=20)
    plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("saa_rmse_across_75_adjustment_steps.png", dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# NEW: First loopback, per-attribute RMSE with min–max bands, x = adjusted attribute names
# -----------------------------
def plot_first_loopback_per_attribute_rmse(df_multi, df):
    """
    For loopback == 1 only:
      - X-axis = attribute being adjusted at each step (actual 'adjusted_att' order for loopback 1)
      - For each method: 5 lines (one per attribute), each is mean across 10 reruns,
        shaded with min-max across reruns at each step.
    """
    # Identify loopback 1 rows
    mask_first = (df["loopback"] == 1)
    # Compute per-attribute RMSE for those rows using the multi-index numeric blocks
    per_attr_rmse = compute_per_attribute_rmse_for_rows(df_multi, mask_first.values)

    # Align with df subset (same row order)
    df_first = df.loc[mask_first, ["method_run", "rerun", "att_order", "adjusted_att"]].copy().reset_index(drop=True)
    # Normalize adjusted_att for labeling the x-axis
    df_first["adjusted_attr_base"] = df_first["adjusted_att"].map(normalize_attr_name)

    # Determine the actual order of adjusted attributes in loopback 1 (by att_order 1..5)
    order_map = (
        df_first[["att_order", "adjusted_attr_base"]]
        .dropna()
        .sort_values("att_order")
        .drop_duplicates(subset=["att_order"])
        .set_index("att_order")["adjusted_attr_base"]
        .to_dict()
    )
    # x-axis labels in step order (1..5)
    x_labels = [order_map.get(i, str(i)) for i in range(1, 6)]

    # Attach RMSE columns
    for att in ATTRIBUTES:
        df_first[att] = per_attr_rmse[att].values

    # For each method, build the figure
    for method in SAA_METHODS:
        sub = df_first[df_first["method_run"] == method].copy()

        # We'll compute, for each plotted attribute P in ATTRIBUTES:
        #    for each step s in 1..5:
        #        collect RMSE_P across reruns where att_order == s
        #        aggregate mean, min, max over reruns
        step_idx = np.array([1, 2, 3, 4, 5], dtype=int)

        plt.figure(figsize=(12, 8))  # Larger figure
        for plot_attr in ATTRIBUTES:
            means = []
            mins = []
            maxs = []
            for s in step_idx:
                vals = sub.loc[sub["att_order"] == s, plot_attr].astype(float)
                if len(vals) == 0:
                    means.append(np.nan)
                    mins.append(np.nan)
                    maxs.append(np.nan)
                else:
                    means.append(vals.mean())
                    mins.append(vals.min())
                    maxs.append(vals.max())

            x = np.arange(1, 6)
            means = np.array(means, dtype=float)
            mins = np.array(mins, dtype=float)
            maxs = np.array(maxs, dtype=float)

            # plot mean + min–max band for this attribute line
            plt.plot(x, means, marker="o", label=plot_attr, linewidth=2.5, markersize=7)
            plt.fill_between(x, mins, maxs, alpha=0.2)

        # plt.title(f"First loopback adjustment (method: {method})", fontsize=TITLE_FONTSIZE, pad=20)
        plt.xlabel("Adjusted attribute", fontsize=AXIS_LABEL_FONTSIZE)
        plt.ylabel("RMSE", fontsize=AXIS_LABEL_FONTSIZE)
        plt.xticks([1, 2, 3, 4, 5], x_labels, fontsize=TICK_FONTSIZE)
        plt.yticks(fontsize=TICK_FONTSIZE)
        plt.grid(True, axis="y")
        plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE, ncol=2)
        plt.tight_layout()
        plt.savefig(f"first_loopback_adjustment_{method}.png", dpi=300, bbox_inches='tight')
        plt.show()

# -----------------------------
# Zoomed versions kept from before
# -----------------------------
def _zoom_slice(pivot, start=6, end=15):
    z = pivot.loc[start:end]
    steps = z.index.to_numpy(int)
    mean_vals = z.mean(axis=1, skipna=True).to_numpy(float)
    min_vals = z.min(axis=1, skipna=True).to_numpy(float)
    max_vals = z.max(axis=1, skipna=True).to_numpy(float)
    return steps, mean_vals, min_vals, max_vals

def plot_rmse_per_loopback_mean_zoom(df, start=ZOOM_START, end=ZOOM_END):
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["mean"].mean().reset_index()
    plt.figure(figsize=(14, 8))  # Larger figure
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="mean")
        pivot = pivot.reindex(np.arange(1, 15 + 1))
        steps, mean_vals, min_vals, max_vals = _zoom_slice(pivot, start, end)
        plt.plot(steps, mean_vals, linewidth=3, label=f"{method}")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)
    plt.xlabel("Loopback", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("RMSE", fontsize=AXIS_LABEL_FONTSIZE)
    # plt.title("SAA RMSE per loopback (zoom)", fontsize=TITLE_FONTSIZE, pad=20)
    # plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("saa_rmse_per_loopback_zoom.png", dpi=300, bbox_inches='tight')
    plt.show()

def plot_target_per_loopback_mean_zoom(df, start=ZOOM_START, end=ZOOM_END):
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["target_n_syn"].mean().reset_index()
    plt.figure(figsize=(14, 8))  # Larger figure
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="target_n_syn")
        pivot = pivot.reindex(np.arange(1, 15 + 1))
        steps, mean_vals, min_vals, max_vals = _zoom_slice(pivot, start, end)
        plt.plot(steps, mean_vals, linewidth=3, label=f"{method}")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)
    plt.xlabel("Loopback", fontsize=AXIS_LABEL_FONTSIZE)
    plt.ylabel("Target number of synthetic agents", fontsize=AXIS_LABEL_FONTSIZE)
    # plt.title("Target synthetic population per loopback (zoom)", fontsize=TITLE_FONTSIZE, pad=20)
    # plt.legend(fontsize=LEGEND_FONTSIZE, title_fontsize=LEGEND_TITLE_FONTSIZE)
    plt.tick_params(axis='both', which='major', labelsize=TICK_FONTSIZE)
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("zoom_saa_target_per_loopback.png", dpi=300, bbox_inches='tight')
    plt.show()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    df_multi, df_all = base_dataframe(CSV_PATH)

    # Existing plots
    plot_rmse_per_loopback_mean(df_all)
    plot_target_per_loopback_mean(df_all)
    plot_rmse_75_steps(df_all)

    # Zoomed (6–15)
    plot_rmse_per_loopback_mean_zoom(df_all, start=6, end=15)
    plot_target_per_loopback_mean_zoom(df_all, start=6, end=15)

    # NEW: First loopback, per-attribute RMSE with min–max bands, x labels by actual adjusted attribute
    plot_first_loopback_per_attribute_rmse(df_multi, df_all)
