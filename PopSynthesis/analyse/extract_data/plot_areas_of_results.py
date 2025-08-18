# ============================
# Three figures:
# 1) Scatter: one point per method (mean across 10 runs)
# 2) Area (zoom): SAA + IPF
# 3) Area (zoom): SAA + WGAN
# ============================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import alphashape
from shapely.geometry import Polygon, Point, MultiPoint
from shapely.affinity import scale as shapely_scale
from scipy.spatial import ConvexHull
from shapely.affinity import scale as shapely_scale

method_map = {
    "saa_BN_pool":       "SAA_BN_pool",
    "saa_seed_addzero":  "SAA_seed_addzero",
    "saa_seed_misszero": "SAA_seed_misszero",
    "ipf_normal":        "IPF_normal",
    "ipf_bn":            "IPF_fromBN",   # rename to match requested label
    "wgan":              "WGAN_hhsz",    # rename to match requested label
    "bn":                "BN_hhsz",
}

# --- Input files ---
RMSE_CSV = r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\combined_fin_rmse_records.csv"
JSD_CSV  = r"C:\Users\dlaa0001\Documents\PhD\PopSyn_Monash\IO\output\runs\combined_fin_jsd_records.csv"

# ---------- Appearance knobs ----------
USE_CONVEX_HULL = True
SCATTER_SIZE = 90
SCATTER_EDGE = 1.2

AREA_ALPHA   = 0.35
EDGE_WIDTH   = 1.4
JITTER_FAC   = 0.01     # jitter if degenerate: fraction of data range
BUFFER_FAC   = 0.010     # fallback tiny circle radius as fraction of data range
AREA_BUFFER_FAC = 0.2   # outward polygon scale (e.g., 0.02 = +2%)

PALETTE = [
    "#1f77b4","#ff7f0e","#2ca02c","#d62728","#674586",
    "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"
]

# ---------- Which methods belong to each family ----------
def is_saa(name: str) -> bool:
    return name.lower().startswith("saa")

def is_ipf(name: str) -> bool:
    return name.lower().startswith("ipf")

def is_wgan(name: str) -> bool:
    return name.lower() == "wgan" or "wgan" in name.lower()

def is_bn(name: str) -> bool:
    return name.lower() == "bn" or name.lower().startswith("bn")

# ---------- Data -> per-run XY and per-method means ----------
def compute_per_run_xy(rmse_csv: str, jsd_csv: str) -> pd.DataFrame:
    """
    Returns per-run (x,y) for each method_run:
      x = mean RMSE across attributes per run
      y = mean JSD  across attributes per run
    Columns: [method_run, run, x, y]
    """
    # RMSE: exclude 'total', avg across states per attribute, then across attributes
    df_r = pd.read_csv(rmse_csv)
    df_r = df_r[df_r["att"] != "'total'"].copy()
    r_long = df_r.melt(
        id_vars=["att","state","method_run"],
        value_vars=[f"run_{i}" for i in range(10)],
        var_name="run", value_name="rmse",
    )
    r_attr_run = r_long.groupby(["method_run","att","run"], as_index=False)["rmse"].mean()
    rmse_run   = r_attr_run.groupby(["method_run","run"], as_index=False)["rmse"].mean()
    rmse_run.rename(columns={"rmse":"x"}, inplace=True)

    # JSD: avg across attributes per run
    df_j = pd.read_csv(jsd_csv)
    j_long = df_j.melt(
        id_vars=["att","method_run"],
        value_vars=[f"run_{i}" for i in range(10)],
        var_name="run", value_name="jsd",
    )
    j_attr_run = j_long.groupby(["method_run","att","run"], as_index=False)["jsd"].mean()
    jsd_run    = j_attr_run.groupby(["method_run","run"], as_index=False)["jsd"].mean()
    jsd_run.rename(columns={"jsd":"y"}, inplace=True)

    pts = pd.merge(rmse_run, jsd_run, on=["method_run","run"])
    pts["x"] = pd.to_numeric(pts["x"], errors="raise")
    pts["y"] = pd.to_numeric(pts["y"], errors="raise")
    return pts

def compute_method_means(per_run_xy: pd.DataFrame) -> pd.DataFrame:
    """
    One point per method: mean over its 10 runs.
    Columns: [method_run, x_mean, y_mean]
    """
    m = per_run_xy.groupby("method_run", as_index=False).agg(
        x_mean=("x","mean"),
        y_mean=("y","mean"),
    )
    return m

# ---------- Geometry: concave hull -> convex hull -> buffer ----------
def ensure_non_degenerate(points: np.ndarray, xrng: float, yrng: float) -> np.ndarray:
    uniq = np.unique(points, axis=0)
    if uniq.shape[0] >= 3 and not (np.isclose(points[:,0].std(), 0.0) or np.isclose(points[:,1].std(), 0.0)):
        return points
    jx = JITTER_FAC * (xrng if xrng > 0 else 1.0)
    jy = JITTER_FAC * (yrng if yrng > 0 else 1.0)
    return points + np.random.normal(0.0, [jx, jy], size=points.shape)

def alpha_or_hull_polygon(points: np.ndarray, xrng: float, yrng: float) -> Polygon:
    pts = np.asarray(points, float)

    if USE_CONVEX_HULL:
        if pts.shape[0] >= 3:
            hull = ConvexHull(pts)
            coords = pts[hull.vertices]
            poly = Polygon(coords)
            poly = shapely_scale(poly, xfact=1.08, yfact=1.08, origin='center')  # +2% size
            if poly.area > 0:
                return poly
        cx, cy = pts.mean(axis=0)
        r = max(1e-6, 0.01 * (xrng + yrng) / 2.0)
        return Point(cx, cy).buffer(r, resolution=64)

    # (alphashape path if USE_CONVEX_HULL = False)
    mp = MultiPoint([tuple(p) for p in pts])
    alpha = alphashape.optimizealpha(mp)
    shape = alphashape.alphashape(mp, alpha)
    if shape.geom_type == "Polygon":
        poly = shape
    elif shape.geom_type == "MultiPolygon":
        poly = max(list(shape.geoms), key=lambda g: g.area)
    else:
        poly = None

    if (poly is None) or (poly.area <= 0):
        if pts.shape[0] >= 3:
            hull = ConvexHull(pts)
            coords = pts[hull.vertices]
            poly = Polygon(coords)
        else:
            cx, cy = pts.mean(axis=0)
            r = max(1e-6, 0.01 * (xrng + yrng) / 2.0)
            poly = Point(cx, cy).buffer(r, resolution=64)

    return poly

def outward_buffer(poly: Polygon, frac: float) -> Polygon:
    if frac == 0.0:
        return poly
    minx, miny, maxx, maxy = poly.bounds
    cx, cy = (minx+maxx)/2.0, (miny+maxy)/2.0
    return shapely_scale(poly, xfact=1.0+frac, yfact=1.0+frac, origin=(cx, cy))

# ---------- Plot helpers ----------
def color_map_for(methods):
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(methods))}

def scatter_means_plot(method_means: pd.DataFrame, title: str, colors: dict):
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    for _, row in method_means.iterrows():
        m = row["method_run"]
        ax.scatter(row["x_mean"], row["y_mean"],
                   s=SCATTER_SIZE, c=colors[m], edgecolors="black", linewidths=SCATTER_EDGE,
                   label=m)
    # legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), title="Method", loc="best", frameon=True)
    ax.set_xlabel("RMSE (avg across attributes)")
    ax.set_ylabel("JSD (avg across attributes)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

def area_zoom_plot(per_run_xy: pd.DataFrame, method_filter, title: str, colors: dict, clip_pct=(2,98)):
    """
    per_run_xy: DataFrame with columns [method_run, run, x, y]
    method_filter: function(str)->bool, to select methods to include
    clip_pct: percentiles to set axis limits (zoom)
    """
    df = per_run_xy[per_run_xy["method_run"].apply(method_filter)].copy()

    # Build polygons
    x_all = df["x"].values; y_all = df["y"].values
    xrng = float(x_all.max() - x_all.min()) if len(x_all) else 1.0
    yrng = float(y_all.max() - y_all.min()) if len(y_all) else 1.0

    polys = []
    for m, g in df.groupby("method_run"):
        pts = g[["x","y"]].to_numpy(float)
        pts = ensure_non_degenerate(pts, xrng, yrng)
        poly = alpha_or_hull_polygon(pts, xrng, yrng)
        if AREA_BUFFER_FAC > 0.0:
            poly = outward_buffer(poly, AREA_BUFFER_FAC)
        coords = np.asarray(poly.exterior.coords)
        polys.append((m, coords, float(poly.area)))

    # Draw big first
    polys.sort(key=lambda t: t[2], reverse=True)

    fig, ax = plt.subplots(figsize=(9.0, 7.0))
    for m, coords, _ in polys:
        col = colors[m]
        ax.fill(coords[:,0], coords[:,1],
                facecolor=col, edgecolor=col,
                alpha=AREA_ALPHA, linewidth=EDGE_WIDTH, label=m)

    # Axis zoom by percentiles (on included methods only)
    x = df["x"].values; y = df["y"].values
    xlo, xhi = np.percentile(x, clip_pct[0]), np.percentile(x, clip_pct[1])
    ylo, yhi = np.percentile(y, clip_pct[0]), np.percentile(y, clip_pct[1])
    xpad = 0.06 * (xhi - xlo if xhi > xlo else 1.0)
    ypad = 0.06 * (yhi - ylo if yhi > ylo else 1.0)
    ax.set_xlim(xlo - xpad, xhi + xpad)
    ax.set_ylim(ylo - ypad, yhi + ypad)

    # Legend without duplicates
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), title="Method", loc="best", frameon=True)

    ax.set_xlabel("RMSE (avg across attributes)")
    ax.set_ylabel("JSD (avg across attributes)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    # ---------- Compute once, plot three figures ----------
    per_run = compute_per_run_xy(RMSE_CSV, JSD_CSV)
    per_run["method_run"] = per_run["method_run"].map(method_map)
    method_means = compute_method_means(per_run)
    fixed_colors = {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(per_run["method_run"].unique()))}

    # 1) Scatter of method means (one point per method)
    scatter_means_plot(method_means, title="Average across 10 runs", colors=fixed_colors)

    # 2) Area zoom: SAA + IPF
    area_zoom_plot(
        per_run,
        method_filter=lambda m: is_saa(m) or is_ipf(m),
        title="SAA-based and IPF-based (from 10 per-run points)",
        clip_pct=(2, 98),
        colors=fixed_colors
    )

    # 3) Area zoom: BN + WGAN
    area_zoom_plot(
        per_run,
        method_filter=lambda m: is_bn(m) or is_wgan(m),
        title="BN and WGAN (from 10 per-run points)",
        clip_pct=(2, 98),
        colors=fixed_colors
    )
