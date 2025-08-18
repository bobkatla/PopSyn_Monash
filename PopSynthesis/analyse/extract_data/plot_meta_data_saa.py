import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
CSV_PATH = r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\big\fin_meta_results.csv"
SAA_METHODS = ["saa_BN_pool", "saa_seed_addzero", "saa_seed_misszero"]
ZOOM_START = 6        # start loopback for zoom
ZOOM_END = 15         # end loopback for zoom (inclusive)

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

    df = df[df["method_run"].isin(SAA_METHODS)].copy()
    df["rerun"] = df["rerun"].astype("Int64")
    return df

# -----------------------------
# Plotters (full-range)
# -----------------------------
def plot_rmse_per_loopback_mean(df, title_suffix="(full range)"):
    """RMSE per loopback (avg of 5 attributes), mean across reruns + min–max band."""
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["mean"].mean().reset_index()

    plt.figure(figsize=(14, 7))
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="mean")
        pivot = pivot.reindex(np.arange(1, 15 + 1))

        steps = pivot.index.to_numpy(int)
        mean_vals = pivot.mean(axis=1, skipna=True).to_numpy(float)
        min_vals = pivot.min(axis=1, skipna=True).to_numpy(float)
        max_vals = pivot.max(axis=1, skipna=True).to_numpy(float)

        plt.plot(steps, mean_vals, linewidth=2, label=f"{method} (mean)")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)

    plt.xlabel("Loopback (1–15)")
    plt.ylabel("RMSE (mean of 5 attributes per loopback)")
    plt.title(f"SAA RMSE per loopback {title_suffix}")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

def plot_target_per_loopback_mean(df, title_suffix="(full range)"):
    """Target_n_syn per loopback (avg of 5 attributes), mean across reruns + min–max band."""
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["target_n_syn"].mean().reset_index()

    plt.figure(figsize=(14, 7))
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="target_n_syn")
        pivot = pivot.reindex(np.arange(1, 15 + 1))

        steps = pivot.index.to_numpy(int)
        mean_vals = pivot.mean(axis=1, skipna=True).to_numpy(float)
        min_vals = pivot.min(axis=1, skipna=True).to_numpy(float)
        max_vals = pivot.max(axis=1, skipna=True).to_numpy(float)

        plt.plot(steps, mean_vals, linewidth=2, label=f"{method} (mean)")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)

    plt.xlabel("Loopback (1–15)")
    plt.ylabel("Target number of synthetic agents (avg over 5 attributes)")
    plt.title(f"Target synthetic population per loopback {title_suffix}")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

def plot_rmse_75_steps(df):
    """RMSE across all 75 steps: mean across reruns + min–max band; separators each 5 steps."""
    plt.figure(figsize=(16, 7))
    for method in SAA_METHODS:
        df_m = df[df["method_run"] == method]
        pivot = df_m.pivot_table(index="step", columns="rerun", values="mean")
        pivot = pivot.reindex(np.arange(1, 75 + 1))

        steps = pivot.index.to_numpy(int)
        mean_vals = pivot.mean(axis=1, skipna=True).to_numpy(float)
        min_vals = pivot.min(axis=1, skipna=True).to_numpy(float)
        max_vals = pivot.max(axis=1, skipna=True).to_numpy(float)

        plt.plot(steps, mean_vals, linewidth=2, label=f"{method} (mean)")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.18)

    for b in range(5, 75, 5):
        plt.axvline(b + 0.5, linestyle=":", linewidth=0.8)

    plt.xlabel("Adjustment step (1–75)")
    plt.ylabel("RMSE")
    plt.title("SAA RMSE across 75 adjustment steps (mean + range)")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Plotters (zoomed)
# -----------------------------
def _zoom_slice(pivot, start=6, end=15):
    """Slice a pivot (index=loopback) to [start..end] inclusive, keeping dtype sane."""
    z = pivot.loc[start:end]
    steps = z.index.to_numpy(int)
    mean_vals = z.mean(axis=1, skipna=True).to_numpy(float)
    min_vals = z.min(axis=1, skipna=True).to_numpy(float)
    max_vals = z.max(axis=1, skipna=True).to_numpy(float)
    return steps, mean_vals, min_vals, max_vals

def plot_rmse_per_loopback_mean_zoom(df, start=ZOOM_START, end=ZOOM_END):
    """Zoomed version of RMSE per loopback (avg of 5 attributes) for loopbacks [start..end]."""
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["mean"].mean().reset_index()

    plt.figure(figsize=(12, 6))
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="mean")
        pivot = pivot.reindex(np.arange(1, 15 + 1))

        steps, mean_vals, min_vals, max_vals = _zoom_slice(pivot, start, end)
        plt.plot(steps, mean_vals, linewidth=2, label=f"{method} (mean)")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)

    plt.xlabel(f"Loopback ({start}–{end})")
    plt.ylabel("RMSE (mean of 5 attributes per loopback)")
    plt.title(f"SAA RMSE per loopback (zoom {start}–{end})")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

def plot_target_per_loopback_mean_zoom(df, start=ZOOM_START, end=ZOOM_END):
    """Zoomed version of target_n_syn per loopback (avg of 5 attributes) for loopbacks [start..end]."""
    df_loop = df.groupby(["method_run", "rerun", "loopback"])["target_n_syn"].mean().reset_index()

    plt.figure(figsize=(12, 6))
    for method in SAA_METHODS:
        df_m = df_loop[df_loop["method_run"] == method]
        pivot = df_m.pivot_table(index="loopback", columns="rerun", values="target_n_syn")
        pivot = pivot.reindex(np.arange(1, 15 + 1))

        steps, mean_vals, min_vals, max_vals = _zoom_slice(pivot, start, end)
        plt.plot(steps, mean_vals, linewidth=2, label=f"{method} (mean)")
        plt.fill_between(steps, min_vals, max_vals, alpha=0.2)

    plt.xlabel(f"Loopback ({start}–{end})")
    plt.ylabel("Target number of synthetic agents (avg over 5 attributes)")
    plt.title(f"Target synthetic population per loopback (zoom {start}–{end})")
    plt.legend()
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.show()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    df_all = base_dataframe(CSV_PATH)

    # Full-range plots
    plot_rmse_per_loopback_mean(df_all, title_suffix="(full range)")
    plot_target_per_loopback_mean(df_all, title_suffix="(full range)")
    plot_rmse_75_steps(df_all)

    # Zoomed plots (last 10 loopbacks: 6–15)
    plot_rmse_per_loopback_mean_zoom(df_all, start=6, end=15)
    plot_target_per_loopback_mean_zoom(df_all, start=6, end=15)
