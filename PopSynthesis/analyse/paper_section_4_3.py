"""Publication analysis for paper Section 4.3.

This module consumes existing analysis artifacts only.  It never runs synthesis.
For paper-level repeat summaries, each single observed SAA result is represented as
ten identical analytical repetitions.  The imputation is explicitly recorded in
the validation and exported tables.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RUN_LABELS = {
    "saa_original_unrestricted": "SAA Unrestricted",
    "saa_original_relax_empty_rows_columns": "SAA Relaxed",
    "saa_original_seed_only": "SAA Seed-only",
    "saa_seed_free_all_zero": "SAA Seed-free",
    "saa_bn_pool_seed_only": "SAA BN-pool",
    "ipf_original_weighted_seed": "IPF Original",
    "ipf_bn_pool": "IPF BN-pool",
    "bn_exact_hhsize": "BN",
    "wgan_exact_hhsize": "WGAN",
}
RUN_ORDER = list(RUN_LABELS)
SAA_RUNS = RUN_ORDER[:5]
ATTRIBUTES = ["hhsize", "dwelltype", "owndwell", "hhinc", "totalvehs"]
GROUP = {r: "SAA" for r in SAA_RUNS} | {
    "ipf_original_weighted_seed": "IPF",
    "ipf_bn_pool": "IPF",
    "bn_exact_hhsize": "Generative",
    "wgan_exact_hhsize": "Generative",
}
COLORS = {"SAA": "#3978A8", "IPF": "#E07A33", "Generative": "#4C9A68"}
MARKERS = {"SAA": "o", "IPF": "s", "Generative": "^"}
SHORT_LABELS = {
    "saa_original_unrestricted": "SAA\nUnrestricted",
    "saa_original_relax_empty_rows_columns": "SAA\nRelaxed",
    "saa_original_seed_only": "SAA\nSeed-only",
    "saa_seed_free_all_zero": "SAA\nSeed-free",
    "saa_bn_pool_seed_only": "SAA\nBN-pool",
    "ipf_original_weighted_seed": "IPF\nOriginal",
    "ipf_bn_pool": "IPF\nBN-pool",
    "bn_exact_hhsize": "BN",
    "wgan_exact_hhsize": "WGAN",
}


def repository_root(start: Path | None = None) -> Path:
    path = (start or Path.cwd()).resolve()
    for candidate in (path, *path.parents):
        if (candidate / "pyproject.toml").exists() and (candidate / "PopSynthesis").exists():
            return candidate
    raise FileNotFoundError("Could not locate the PopSyn_Monash repository root")


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 350, "font.size": 9,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 7.5,
        "ytick.labelsize": 8, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "figure.facecolor": "white",
        "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def paths(root: Path | None = None) -> dict[str, Path]:
    root = root or repository_root()
    analysis = root / "IO/output/full_household_analysis"
    output = analysis / "paper_section_4_3"
    figures = output / "figures"
    tables = output / "tables"
    supplementary = output / "supplementary"
    for directory in (output, figures, tables, supplementary):
        directory.mkdir(parents=True, exist_ok=True)
    return {"root": root, "analysis": analysis, "output": output,
            "figures": figures, "tables": tables, "supplementary": supplementary}


def load_inputs(p: dict[str, Path]) -> dict[str, Any]:
    a = p["analysis"]
    return {
        "joint": pd.read_csv(a / "joint_distribution_metrics.csv"),
        "conditional": pd.read_csv(a / "conditional_relationship_metrics.csv"),
        "support": pd.read_csv(a / "support_provenance_metrics.csv"),
        "census": pd.read_csv(a / "census_metric_summary.csv"),
        "method": pd.read_csv(a / "method_summary.csv"),
        "diversity": pd.read_csv(a / "diversity_metrics.csv"),
        "unconfirmed": pd.read_parquet(a / "unconfirmed_combinations.parquet"),
        "aligned": pd.read_parquet(a / "aligned_five_way_distributions.parquet"),
        "catalog": pd.read_csv(a / "run_catalog.csv"),
        "manifest": json.loads((a / "analysis_manifest.json").read_text()),
    }


def impute_saa_repeats(frame: pd.DataFrame, repeats: int = 10) -> pd.DataFrame:
    """Copy each observed SAA repeat to analytical repeats 1..10.

    ``repeat_source`` and ``repeat_assumption`` prevent the copies being mistaken
    for independently executed outputs.
    """
    pieces = []
    for run_id, part in frame.groupby("run_id", sort=False, dropna=False):
        part = part.copy()
        if run_id in SAA_RUNS and part["repeat"].nunique() == 1:
            source_repeat = int(part["repeat"].iloc[0])
            for repeat in range(1, repeats + 1):
                copy = part.copy()
                copy["repeat_source"] = source_repeat
                copy["repeat"] = repeat
                copy["repeat_assumption"] = "assumed_identical_SAA_copy"
                pieces.append(copy)
        else:
            part["repeat_source"] = part["repeat"]
            part["repeat_assumption"] = "observed"
            pieces.append(part)
    return pd.concat(pieces, ignore_index=True)


def validate_inputs(data: dict[str, Any], p: dict[str, Path]) -> pd.DataFrame:
    joint = data["joint"]
    observed = data["catalog"].groupby("run_id")["repeat"].nunique().reindex(RUN_ORDER)
    records = []
    def check(name: str, passed: bool, detail: str) -> None:
        records.append({"check": name, "passed": bool(passed), "detail": detail})

    check("all_nine_methods", set(RUN_ORDER) == set(data["method"]["run_id"]),
          f"Found {data['method']['run_id'].nunique()} methods")
    check("weighted_VISTA_reference", "weighted_seed" in set(joint["reference"]),
          "joint reference identifier is weighted_seed; manifest defines it as global joint reference")
    js = joint["js_distance"].dropna()
    check("JS_distance_bounds", js.between(0, 1).all(), f"range [{js.min():.6f}, {js.max():.6f}]")
    check("JS_distance_definition", True,
          "metrics.py calls scipy.spatial.distance.jensenshannon(..., base=2); distance is sqrt(divergence)")
    expected = {2: 10, 3: 10, 4: 5, 5: 1}
    got = (joint[joint.reference.eq("weighted_seed")]
           .groupby("interaction_order")["attributes"].nunique().to_dict())
    check("expected_joint_subsets", got == expected, f"expected={expected}; observed={got}")
    check("observed_repeat_catalog", (observed.loc[SAA_RUNS] == 1).all() and
          (observed.drop(SAA_RUNS) == 10).all(),
          "Observed SAA repeats=1 each; IPF/BN/WGAN repeats=10 each")
    check("paper_repeat_assumption", True,
          "SAA repeat 1 is copied analytically to repeats 1..10; reported SAA SD is therefore assumed 0")

    # Preserve the exact experimental fact alongside the requested paper assumption.
    repeat_audit = pd.DataFrame({"run_id": RUN_ORDER, "method": [RUN_LABELS[r] for r in RUN_ORDER],
                                 "observed_repeats": observed.values})
    repeat_audit["reported_repeats"] = 10
    repeat_audit["repeat_status"] = np.where(
        repeat_audit.run_id.isin(SAA_RUNS), "1 observed + 9 assumed identical", "10 observed")
    repeat_audit.to_csv(p["tables"] / "repeat_assumption_audit.csv", index=False)
    validation = pd.DataFrame(records)
    validation.to_csv(p["tables"] / "validation_checks.csv", index=False)
    assert validation.loc[validation.check.isin([
        "all_nine_methods", "weighted_VISTA_reference", "JS_distance_bounds",
        "expected_joint_subsets"]), "passed"].all()
    return validation


def joint_order_summary(data: dict[str, Any], p: dict[str, Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    joint = data["joint"]
    joint = joint[joint.reference.eq("weighted_seed")].copy()
    joint = impute_saa_repeats(joint)
    per_repeat = (joint.groupby(["run_id", "repeat", "repeat_assumption", "interaction_order"], as_index=False)
                  .agg(js_distance=("js_distance", "mean"),
                       number_of_subsets=("attributes", "nunique")))
    summary = (per_repeat.groupby(["run_id", "interaction_order"], as_index=False)
               .agg(mean_js_distance=("js_distance", "mean"),
                    sd_across_runs=("js_distance", "std"),
                    min_js_distance=("js_distance", "min"),
                    max_js_distance=("js_distance", "max"),
                    repeats_reported=("repeat", "nunique"),
                    number_of_subsets=("number_of_subsets", "first")))
    summary["method"] = summary.run_id.map(RUN_LABELS)
    summary["repeat_basis"] = np.where(summary.run_id.isin(SAA_RUNS),
        "1 observed result represented as 10 identical repeats", "10 observed repeats")
    summary = summary[["run_id", "method", "interaction_order", "mean_js_distance",
                       "sd_across_runs", "min_js_distance", "max_js_distance",
                       "number_of_subsets", "repeats_reported", "repeat_basis"]]
    per_repeat.to_csv(p["tables"] / "joint_JS_by_order_per_repeat.csv", index=False)
    summary.to_csv(p["tables"] / "joint_JS_by_order_summary.csv", index=False)
    return per_repeat, summary


def figure_joint_order(summary: pd.DataFrame, p: dict[str, Path]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.7), sharey=True)
    xmax = summary.mean_js_distance.max() + summary.sd_across_runs.fillna(0).max()
    for ax, order in zip(axes.flat, (2, 3, 4, 5)):
        part = summary[summary.interaction_order.eq(order)].set_index("run_id").reindex(RUN_ORDER)
        x = np.arange(len(RUN_ORDER))
        for i, run_id in enumerate(RUN_ORDER):
            y = part.loc[run_id, "mean_js_distance"]
            err = part.loc[run_id, "sd_across_runs"]
            ax.errorbar(i, y, yerr=0 if pd.isna(err) else err, fmt=MARKERS[GROUP[run_id]],
                        color=COLORS[GROUP[run_id]], markersize=5.2, capsize=2.2,
                        elinewidth=.9, markeredgecolor="white", markeredgewidth=.4)
        ax.set_title(f"{order}-way ({int(part.number_of_subsets.iloc[0])} subsets)")
        ax.set_xticks(x, [SHORT_LABELS[r] for r in RUN_ORDER])
        ax.grid(axis="y", color="#dddddd", linewidth=.6)
        ax.set_ylim(0, min(1.0, xmax * 1.08))
    axes[0, 0].set_ylabel("JS distance to weighted VISTA")
    axes[1, 0].set_ylabel("JS distance to weighted VISTA")
    handles = [plt.Line2D([], [], color=COLORS[g], marker=MARKERS[g], linestyle="none", label=g)
               for g in ("SAA", "IPF", "Generative")]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(.5, 1.01))
    fig.suptitle("Joint-distribution fidelity by interaction order", y=1.045, fontsize=12)
    fig.text(.5, -.01, "Points are repeat means; bars show ±1 SD. SAA SD=0 under the stated identical-repeat assumption.",
             ha="center", fontsize=8)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(p["figures"] / f"figure_A_joint_fidelity_by_order.{ext}", bbox_inches="tight")
    plt.close(fig)


def tradeoff_summary(data: dict[str, Any], per_repeat: pd.DataFrame,
                     p: dict[str, Path]) -> pd.DataFrame:
    census = data["census"]
    census = census[(census.evaluation == "canonical") &
                    (census.aggregation == "macro") & census.attribute.isin(ATTRIBUTES)].copy()
    census = impute_saa_repeats(census)
    cpr = (census.groupby(["run_id", "repeat"], as_index=False)
           .agg(census_macro_tvd=("total_variation", "mean")))
    j5 = per_repeat[per_repeat.interaction_order.eq(5)][["run_id", "repeat", "js_distance"]]
    combined = cpr.merge(j5, on=["run_id", "repeat"], validate="one_to_one")
    summary = (combined.groupby("run_id", as_index=False)
               .agg(macro_census_tvd_mean=("census_macro_tvd", "mean"),
                    macro_census_tvd_sd=("census_macro_tvd", "std"),
                    five_way_js_mean=("js_distance", "mean"),
                    five_way_js_sd=("js_distance", "std"),
                    repeats_reported=("repeat", "nunique")))
    summary["method"] = summary.run_id.map(RUN_LABELS)
    dominated = []
    for _, row in summary.iterrows():
        other = summary[summary.run_id.ne(row.run_id)]
        is_dominated = ((other.macro_census_tvd_mean <= row.macro_census_tvd_mean) &
                        (other.five_way_js_mean <= row.five_way_js_mean) &
                        ((other.macro_census_tvd_mean < row.macro_census_tvd_mean) |
                         (other.five_way_js_mean < row.five_way_js_mean))).any()
        dominated.append(is_dominated)
    summary["non_dominated_on_these_two_metrics"] = ~np.array(dominated)
    summary["repeat_basis"] = np.where(summary.run_id.isin(SAA_RUNS),
        "1 observed result represented as 10 identical repeats", "10 observed repeats")
    summary.loc[summary.run_id.isin(SAA_RUNS),
                ["macro_census_tvd_sd", "five_way_js_sd"]] = 0.0
    summary = summary.set_index("run_id").reindex(RUN_ORDER).reset_index()
    summary.to_csv(p["tables"] / "census_vs_five_way_joint_fidelity.csv", index=False)
    return summary


def figure_tradeoff(summary: pd.DataFrame, p: dict[str, Path]) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 5.1))
    offsets = {
        "saa_original_unrestricted": (6, -20), "saa_seed_free_all_zero": (6, 8),
        "saa_original_relax_empty_rows_columns": (8, 16), "saa_original_seed_only": (12, 5),
        "saa_bn_pool_seed_only": (6, 8), "ipf_original_weighted_seed": (6, -13),
        "ipf_bn_pool": (6, 7), "bn_exact_hhsize": (6, 7), "wgan_exact_hhsize": (6, -13),
    }
    for _, row in summary.iterrows():
        run_id = row.run_id; group = GROUP[run_id]
        ax.errorbar(row.macro_census_tvd_mean, row.five_way_js_mean,
                    xerr=row.macro_census_tvd_sd, yerr=row.five_way_js_sd,
                    fmt=MARKERS[group], color=COLORS[group], markersize=7, capsize=2,
                    markeredgecolor="white", markeredgewidth=.5)
        ax.annotate(RUN_LABELS[run_id], (row.macro_census_tvd_mean, row.five_way_js_mean),
                    xytext=offsets[run_id], textcoords="offset points", fontsize=7.5)
        if row.non_dominated_on_these_two_metrics:
            ax.scatter(row.macro_census_tvd_mean, row.five_way_js_mean, s=105,
                       facecolors="none", edgecolors="#222222", linewidths=.8)
    ax.set_xlim(left=-.006)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Macro Census TVD (lower is better)")
    ax.set_ylabel("Five-way JS distance to weighted VISTA (lower is better)")
    ax.set_title("Census alignment versus five-way joint fidelity")
    ax.grid(color="#dddddd", linewidth=.6)
    handles = [plt.Line2D([], [], color=COLORS[g], marker=MARKERS[g], linestyle="none", label=g)
               for g in ("SAA", "IPF", "Generative")]
    handles.append(plt.Line2D([], [], color="#222222", marker="o", markerfacecolor="none",
                              linestyle="none", label="Non-dominated on these metrics"))
    ax.legend(handles=handles, frameon=False, loc="best")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(p["figures"] / f"figure_B_census_vs_five_way_joint.{ext}", bbox_inches="tight")
    plt.close(fig)


def support_summary(data: dict[str, Any], p: dict[str, Path]) -> pd.DataFrame:
    support = data["support"]
    support = support[support.scope.eq("global")].copy()
    support = impute_saa_repeats(support)
    tier_names = {"seed_supported": "VISTA-supported", "bn_only_supported": "BN-only",
                  "unconfirmed": "Unconfirmed"}
    support["category"] = support.tier.map(tier_names)
    per_repeat = support.pivot_table(index=["run_id", "repeat"], columns="category",
                                     values="population_mass_percentage", aggfunc="first").reset_index()
    sums = per_repeat[list(tier_names.values())].sum(axis=1)
    assert np.allclose(sums, 100, atol=1e-6), f"Support shares do not sum to 100: {sums.describe()}"
    rows = []
    for run_id, part in per_repeat.groupby("run_id"):
        row = {"run_id": run_id, "method": RUN_LABELS[run_id], "repeats_reported": len(part)}
        for tier in tier_names.values():
            row[f"{tier}_mean_percent"] = part[tier].mean()
            row[f"{tier}_sd_percent"] = part[tier].std()
        row["repeat_basis"] = ("1 observed result represented as 10 identical repeats"
                               if run_id in SAA_RUNS else "10 observed repeats")
        rows.append(row)
    result = pd.DataFrame(rows).set_index("run_id").reindex(RUN_ORDER).reset_index()
    sd_columns = [column for column in result.columns if column.endswith("_sd_percent")]
    result.loc[result.run_id.isin(SAA_RUNS), sd_columns] = 0.0
    result.to_csv(p["tables"] / "support_provenance_summary.csv", index=False)

    lines = ["\\begin{tabular}{lrrrrrr}", "\\toprule",
             "Method & \\multicolumn{2}{c}{VISTA-supported (\\%)} & \\multicolumn{2}{c}{BN-only (\\%)} & \\multicolumn{2}{c}{Unconfirmed (\\%)} \\\\",
             " & Mean & SD & Mean & SD & Mean & SD \\\\", "\\midrule"]
    for _, r in result.iterrows():
        vals = []
        for tier in tier_names.values():
            vals.extend([r[f"{tier}_mean_percent"], r[f"{tier}_sd_percent"]])
        lines.append(f"{r['method']} & " + " & ".join(f"{v:.3f}" for v in vals) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "% SAA standard deviations are zero under the identical-repeat assumption."]
    (p["tables"] / "support_provenance_table.tex").write_text("\n".join(lines) + "\n")
    return result


def conditional_diagnostics(data: dict[str, Any], p: dict[str, Path]) -> pd.DataFrame:
    cond = data["conditional"]
    cond = cond[(cond.metric_scope == "conditional_vs_seed") & cond.run_id.isin(RUN_ORDER)].copy()
    cond = impute_saa_repeats(cond)
    per_run = (cond.groupby(["run_id", "conditioning_attribute", "outcome_attribute"], as_index=False)
               .agg(mean_conditional_js=("conditional_js_distance", "mean"),
                    mean_conditional_tv=("conditional_total_variation", "mean")))
    rows = []
    for pair, part in per_run.groupby(["conditioning_attribute", "outcome_attribute"]):
        indexed = part.set_index("run_id")
        saa = indexed.reindex(SAA_RUNS)
        all_tv = indexed.mean_conditional_tv.dropna()
        all_js = indexed.mean_conditional_js.dropna()
        rows.append({
            "conditioning_attribute": pair[0], "outcome_attribute": pair[1],
            "relationship": f"{pair[1]} | {pair[0]}",
            "methods_with_conditional_js": int(all_js.size),
            "methods_with_conditional_tv": int(all_tv.size),
            "mean_method_conditional_js": all_js.mean(),
            "max_method_conditional_js": all_js.max(),
            "method_spread_conditional_js": all_js.max() - all_js.min() if len(all_js) else np.nan,
            "mean_method_conditional_tv": all_tv.mean(),
            "max_method_conditional_tv": all_tv.max(),
            "method_spread_conditional_tv": all_tv.max() - all_tv.min() if len(all_tv) else np.nan,
            "SAA_policy_spread_conditional_tv": saa.mean_conditional_tv.max() - saa.mean_conditional_tv.min(),
        })
    ranking = pd.DataFrame(rows)
    # Transparent descriptive ranking: method separation plus SAA-policy separation.
    for column in ("method_spread_conditional_tv", "SAA_policy_spread_conditional_tv", "mean_method_conditional_tv"):
        maximum = ranking[column].max()
        ranking[f"scaled_{column}"] = ranking[column] / maximum if maximum else 0
    ranking["diagnostic_score"] = (0.45 * ranking.scaled_method_spread_conditional_tv +
                                   0.35 * ranking.scaled_SAA_policy_spread_conditional_tv +
                                   0.20 * ranking.scaled_mean_method_conditional_tv)
    ranking["priority_reason"] = "Ranks method separation (45%), SAA-policy separation (35%), and mean distortion (20%)"
    ranking = ranking.sort_values("diagnostic_score", ascending=False).reset_index(drop=True)
    ranking.insert(0, "rank", np.arange(1, len(ranking) + 1))
    ranking.to_csv(p["tables"] / "conditional_relationship_ranking.csv", index=False)

    selected = ranking.head(3)[["conditioning_attribute", "outcome_attribute", "relationship"]]
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.7), sharey=True)
    for ax, (_, rel) in zip(axes, selected.iterrows()):
        part = per_run[(per_run.conditioning_attribute == rel.conditioning_attribute) &
                       (per_run.outcome_attribute == rel.outcome_attribute)].set_index("run_id").reindex(RUN_ORDER)
        y = np.arange(len(RUN_ORDER))
        for i, run_id in enumerate(RUN_ORDER):
            ax.scatter(part.loc[run_id, "mean_conditional_tv"], i, color=COLORS[GROUP[run_id]],
                       marker=MARKERS[GROUP[run_id]], s=28)
        ax.set_title(rel.relationship)
        ax.set_xlabel("Conditional TVD")
        ax.grid(axis="x", color="#dddddd", linewidth=.6)
        ax.set_yticks(y, [RUN_LABELS[r] for r in RUN_ORDER])
        ax.invert_yaxis()
    axes[0].set_ylabel("Method")
    fig.suptitle("Supplement: highest-ranked conditional relationship distortions", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(p["supplementary"] / f"conditional_relationship_diagnostics.{ext}", bbox_inches="tight")
    plt.close(fig)
    return ranking


def diversity_summary(data: dict[str, Any], p: dict[str, Path]) -> pd.DataFrame:
    diversity = data["diversity"]
    global_div = impute_saa_repeats(diversity[diversity.scope.eq("global")].copy())
    base = (global_div.groupby("run_id", as_index=False)
            .agg(occupied_five_way_mean=("occupied_cells", "mean"),
                 occupied_five_way_sd=("occupied_cells", "std"),
                 effective_diversity_mean=("effective_shannon_count", "mean"),
                 effective_diversity_sd=("effective_shannon_count", "std")))
    support = impute_saa_repeats(data["support"][data["support"].scope.eq("global")].copy())
    unc = support[support.tier.eq("unconfirmed")]
    unc_s = (unc.groupby("run_id", as_index=False)
             .agg(unconfirmed_mass_mean_percent=("population_mass_percentage", "mean"),
                  unconfirmed_mass_sd_percent=("population_mass_percentage", "std"),
                  distinct_unconfirmed_mean=("unique_combinations", "mean"),
                  distinct_unconfirmed_sd=("unique_combinations", "std")))
    u = impute_saa_repeats(data["unconfirmed"].copy())
    if len(u):
        u["weighted_seed_hamming"] = u["count"] * u["nearest_seed_hamming"]
        u["weighted_bn_hamming"] = u["count"] * u["nearest_bn_pool_hamming"]
        h = (u.groupby("run_id", as_index=False)
             .agg(unconfirmed_households=("count", "sum"), seed_hamming_numerator=("weighted_seed_hamming", "sum"),
                  bn_hamming_numerator=("weighted_bn_hamming", "sum")))
        h["mean_nearest_VISTA_hamming_mass_weighted"] = h.seed_hamming_numerator / h.unconfirmed_households
        h["mean_nearest_BN_pool_hamming_mass_weighted"] = h.bn_hamming_numerator / h.unconfirmed_households
        h = h[["run_id", "mean_nearest_VISTA_hamming_mass_weighted", "mean_nearest_BN_pool_hamming_mass_weighted"]]
    else:
        h = pd.DataFrame({"run_id": RUN_ORDER})
    result = base.merge(unc_s, on="run_id", how="outer").merge(h, on="run_id", how="left")
    result["method"] = result.run_id.map(RUN_LABELS)
    sd_columns = [column for column in result.columns if column.endswith("_sd") or column.endswith("_sd_percent")]
    result.loc[result.run_id.isin(SAA_RUNS), sd_columns] = 0.0
    result["effective_diversity_definition"] = "exp(-sum_i p_i ln p_i), natural logarithm"
    result["interpretation_note"] = "Diversity and occupied support are descriptive; neither alone establishes realism"
    result = result.set_index("run_id").reindex(RUN_ORDER).reset_index()
    result.to_csv(p["tables"] / "diversity_and_support_expansion_summary.csv", index=False)
    return result


def cross_checks(data: dict[str, Any], joint_summary: pd.DataFrame,
                 tradeoff: pd.DataFrame, support: pd.DataFrame,
                 p: dict[str, Path]) -> pd.DataFrame:
    method = data["method"].set_index("run_id")
    five = joint_summary[joint_summary.interaction_order.eq(5)].set_index("run_id")
    trade = tradeoff.set_index("run_id")
    records = []
    for run_id in RUN_ORDER:
        records.append({"run_id": run_id, "check": "five_way_JS_vs_method_summary",
                        "absolute_difference": abs(five.loc[run_id, "mean_js_distance"] - method.loc[run_id, "joint_seed_js"]),
                        "tolerance": 1e-12})
        records.append({"run_id": run_id, "check": "macro_Census_TVD_vs_method_summary",
                        "absolute_difference": abs(trade.loc[run_id, "macro_census_tvd_mean"] - method.loc[run_id, "census_macro_tv"]),
                        "tolerance": 1e-12})
    result = pd.DataFrame(records)
    result["passed"] = result.absolute_difference <= result.tolerance
    result.to_csv(p["tables"] / "summary_cross_checks.csv", index=False)
    assert result.passed.all(), result[~result.passed].to_string(index=False)
    return result


def write_findings(joint: pd.DataFrame, trade: pd.DataFrame, support: pd.DataFrame,
                   conditional: pd.DataFrame, diversity: pd.DataFrame,
                   validation: pd.DataFrame, p: dict[str, Path]) -> Path:
    j5 = joint[joint.interaction_order.eq(5)].set_index("run_id")
    t = trade.set_index("run_id"); s = support.set_index("run_id"); d = diversity.set_index("run_id")
    best5 = j5.mean_js_distance.idxmin()
    order_rank = {}
    for order in (2, 3, 4, 5):
        q = joint[joint.interaction_order.eq(order)].sort_values("mean_js_distance")
        order_rank[order] = RUN_LABELS[q.iloc[0].run_id]
    top_cond = conditional.iloc[0]
    lines = [
        "# Section 4.3 findings — Joint-Distribution Fidelity and Empirical Support", "",
        "## A. Main numerical findings", "",
        f"- {RUN_LABELS[best5]} has the lowest five-way JS distance to weighted VISTA ({j5.loc[best5, 'mean_js_distance']:.4f}).",
        f"- The exact-Census methods separate sharply in joint fidelity: SAA Unrestricted has five-way JS={j5.loc['saa_original_unrestricted','mean_js_distance']:.4f}, while SAA Seed-free has JS={j5.loc['saa_seed_free_all_zero','mean_js_distance']:.4f}.",
        f"- WGAN has five-way JS={j5.loc['wgan_exact_hhsize','mean_js_distance']:.4f} but macro Census TVD={t.loc['wgan_exact_hhsize','macro_census_tvd_mean']:.4f}; the comparison illustrates that global joint fidelity and zonal marginal alignment measure different properties.", "",
        "## B. SAA configuration trade-offs", "",
        f"- Seed-only assigns {s.loc['saa_original_seed_only','VISTA-supported_mean_percent']:.3f}% of mass to VISTA-supported combinations and 0% to unconfirmed combinations, but does not exactly recover every Census margin.",
        f"- Relaxed assigns {s.loc['saa_original_relax_empty_rows_columns','Unconfirmed_mean_percent']:.3f}% to unconfirmed combinations, compared with {s.loc['saa_original_unrestricted','Unconfirmed_mean_percent']:.3f}% for Unrestricted and {s.loc['saa_seed_free_all_zero','Unconfirmed_mean_percent']:.3f}% for Seed-free.",
        f"- BN-pool assigns {s.loc['saa_bn_pool_seed_only','BN-only_mean_percent']:.3f}% to BN-only support and {s.loc['saa_bn_pool_seed_only','Unconfirmed_mean_percent']:.3f}% to unconfirmed support. The BN pool is a secondary generated-support reference, not ground truth.", "",
        "## C. Joint-order findings", "",
        "- The lowest mean JS method by order is " + "; ".join(f"{o}-way: {m}" for o,m in order_rank.items()) + ".",
        "- Values are means of the already-computed subset-level JS distances; distributions were not averaged before calculating JS.", "",
        "## D. Conditional relationships", "",
        f"- The diagnostic ranking places **{top_cond.relationship}** first (score {top_cond.diagnostic_score:.3f}) because it most strongly combines between-method separation, SAA-policy separation, and average conditional TVD.",
        "- The ranking is descriptive and intended for figure selection; it is not a statistical significance test.", "",
        "## Reproducibility note", "",
        "- IPF, BN, and WGAN each have ten observed repeats. Each SAA variant has one observed result. At the user's direction, the paper tables represent that result as ten identical analytical repeats, so SAA repeat SD is assumed to be zero. This is not evidence from ten independent SAA reruns.",
        "- JS distance is computed with base-2 logarithms and is the square root of JS divergence.",
        f"- Effective diversity is exp(Shannon entropy). For example, SAA Unrestricted has {d.loc['saa_original_unrestricted','occupied_five_way_mean']:.0f} occupied combinations but effective diversity {d.loc['saa_original_unrestricted','effective_diversity_mean']:.1f}; these quantities are not interchangeable.",
    ]
    target = p["output"] / "section_4_3_findings.md"
    target.write_text("\n".join(lines) + "\n")
    return target


def run_all(root: Path | None = None) -> dict[str, Any]:
    configure_style()
    p = paths(root)
    data = load_inputs(p)
    validation = validate_inputs(data, p)
    per_repeat, joint = joint_order_summary(data, p)
    figure_joint_order(joint, p)
    tradeoff = tradeoff_summary(data, per_repeat, p)
    figure_tradeoff(tradeoff, p)
    support = support_summary(data, p)
    conditional = conditional_diagnostics(data, p)
    diversity = diversity_summary(data, p)
    checks = cross_checks(data, joint, tradeoff, support, p)
    report = write_findings(joint, tradeoff, support, conditional, diversity, validation, p)
    return {"paths": p, "data": data, "validation": validation, "joint": joint,
            "tradeoff": tradeoff, "support": support, "conditional": conditional,
            "diversity": diversity, "checks": checks, "report": report}


if __name__ == "__main__":
    result = run_all()
    print(f"Section 4.3 outputs: {result['paths']['output']}")
