"""Comprehensive, count-native analysis of household synthesis experiments."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import json
import math
from pathlib import Path
import platform
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

from PopSynthesis.experiments.config import load_config

from .metrics import (
    condense,
    cramers_v,
    distribution_metrics,
    entropy_metrics,
    mutual_information,
    summary_statistics,
)


@dataclass(frozen=True)
class AnalysisOptions:
    smoothing: float = 1e-12
    rare_thresholds: tuple[int, ...] = (1, 5, 10)
    bootstrap_replicates: int = 2_000
    analysis_seed: int = 20_261_113
    write_figures: bool = True


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    if path.suffix == ".parquet":
        frame.to_parquet(temporary, index=False)
    else:
        frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _flatten_columns(columns: Iterable[Any]) -> list[str]:
    return ["__".join(str(x) for x in item if str(x) != "") if isinstance(item, tuple) else str(item) for item in columns]


def read_census(path: Path, zone_field: str) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Read the repository's two-row categorical marginal format."""

    raw = pd.read_csv(path, header=[0, 1], dtype=str)
    zone_column = raw.columns[0]
    zones = raw[zone_column].astype(str)
    records: list[pd.DataFrame] = []
    states: dict[str, list[str]] = {}
    for attribute, state in raw.columns[1:]:
        attribute, state = str(attribute), str(state)
        states.setdefault(attribute, []).append(state)
        records.append(
            pd.DataFrame(
                {
                    zone_field: zones,
                    "attribute": attribute,
                    "state": state,
                    "census_count": pd.to_numeric(raw[(attribute, state)]),
                }
            )
        )
    return pd.concat(records, ignore_index=True), states


def _full_index(states: Mapping[str, list[str]], attributes: list[str]) -> pd.MultiIndex:
    return pd.MultiIndex.from_product([states[a] for a in attributes], names=attributes)


def _aligned_joint(
    frame: pd.DataFrame,
    attributes: list[str],
    weight: str,
    states: Mapping[str, list[str]],
) -> pd.Series:
    return condense(frame, attributes, weight).reindex(_full_index(states, attributes), fill_value=0.0)


def _project_joint(joint: pd.Series, attributes: list[str]) -> pd.Series:
    """Project an already condensed full joint table onto selected dimensions."""

    return joint.groupby(level=attributes, observed=True).sum()


def _safe_relative(error: pd.Series, reference: pd.Series) -> pd.Series:
    return error / reference.clip(lower=1)


def _holm(pvalues: list[float]) -> list[float]:
    result = np.full(len(pvalues), np.nan)
    valid = [(i, p) for i, p in enumerate(pvalues) if np.isfinite(p)]
    valid.sort(key=lambda item: item[1])
    running = 0.0
    n = len(valid)
    for rank, (index, pvalue) in enumerate(valid):
        running = max(running, min(1.0, (n - rank) * pvalue))
        result[index] = running
    return result.tolist()


class HouseholdExperimentAnalysis:
    """Analyze a completed suite described by its authoritative YAML."""

    def __init__(
        self,
        config_path: Path | str,
        output_dir: Path | str,
        options: AnalysisOptions | None = None,
    ) -> None:
        self.config = load_config(config_path, check_files=True)
        self.output_dir = Path(output_dir).resolve()
        self.options = options or AnalysisOptions()
        self.attributes = list(self.config["attributes"])
        self.zone_field = str(self.config["zone_field"])
        self.experiment_root = Path(self.config["output"]["root"])
        self.census, self.states = read_census(
            Path(self.config["inputs"]["census"]["path"]), self.zone_field
        )
        self.zones = self.census[self.zone_field].drop_duplicates().tolist()
        self.zone_totals = (
            self.census[self.census["attribute"] == self.attributes[0]]
            .groupby(self.zone_field)["census_count"].sum()
        )
        self.seed = self._read_reference("original_seed")
        self.bn_pool = self._read_reference("bn_pool")
        self.seed_joint = condense(self.seed, self.attributes, "_reference_weight")
        self.bn_joint = condense(self.bn_pool, self.attributes, "_reference_weight")
        self.universe = _full_index(self.states, self.attributes)
        self._validate_reference_states()
        self.catalog = self._discover_runs()
        self.counts: dict[tuple[str, int], pd.DataFrame] = {}
        self._nearest_caches: dict[str, dict[tuple[str, ...], dict[str, Any]]] = {
            "seed": {}, "bn_pool": {}
        }

    def _read_reference(self, name: str) -> pd.DataFrame:
        spec = self.config["inputs"][name]
        frame = pd.read_csv(spec["path"], dtype={attribute: str for attribute in self.attributes})
        missing = set(self.attributes + [spec["weight_column"]]) - set(frame.columns)
        if missing:
            raise ValueError(f"{name} is missing columns: {sorted(missing)}")
        result = frame[self.attributes + [spec["weight_column"]]].copy()
        result["_reference_weight"] = pd.to_numeric(result.pop(spec["weight_column"]), errors="raise")
        if (result["_reference_weight"] < 0).any():
            raise ValueError(f"{name} contains negative weights")
        for attribute in self.attributes:
            result[attribute] = pd.Categorical(result[attribute], categories=self.states[attribute])
        return result

    def _validate_reference_states(self) -> None:
        for name, frame in (("seed", self.seed), ("BN pool", self.bn_pool)):
            unsupported = {
                attribute: sorted(set(frame[attribute]) - set(self.states[attribute]))
                for attribute in self.attributes
            }
            unsupported = {key: value for key, value in unsupported.items() if value}
            if unsupported:
                raise ValueError(f"{name} contains states outside the census universe: {unsupported}")

    def _discover_runs(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for run in self.config["runs"]:
            if not run.get("enabled", True):
                continue
            for repeat in range(1, int(run.get("repeats", 1)) + 1):
                repeat_dir = self.experiment_root / run["id"] / f"repeat_{repeat:02d}"
                manifest_path = repeat_dir / "manifest.json"
                count_path = repeat_dir / "household_counts.csv"
                if not manifest_path.is_file() or not count_path.is_file():
                    raise FileNotFoundError(f"Incomplete configured result: {repeat_dir}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") != "complete":
                    raise ValueError(f"Configured result is not complete: {repeat_dir}")
                diagnostics = manifest.get("diagnostics", {})
                rows.append(
                    {
                        "run_id": run["id"],
                        "method": run["method"],
                        "repeat": repeat,
                        "repeats_configured": int(run.get("repeats", 1)),
                        "lifecycle": run.get("lifecycle"),
                        "controlled_attributes": ",".join(
                            ["hhsize"] if run["method"] in {"bn", "wgan"} else self.attributes
                        ),
                        "sampling_seed": manifest.get("sampling_seed"),
                        "training_seed": diagnostics.get("training_seed"),
                        "workers": diagnostics.get("workers"),
                        "runtime_seconds": manifest.get("runtime_seconds"),
                        "count_path": str(count_path),
                        "manifest_path": str(manifest_path),
                        "output_sha256": manifest.get("artifact_hashes", {}).get("household_counts.csv")
                        or _file_hash(count_path),
                        "model_path": diagnostics.get("model_path"),
                        "model_sha256": diagnostics.get("model_sha256"),
                    }
                )
        return pd.DataFrame(rows)

    def _load_counts(self, row: Mapping[str, Any]) -> pd.DataFrame:
        key = (str(row["run_id"]), int(row["repeat"]))
        if key in self.counts:
            return self.counts[key]
        frame = pd.read_csv(
            row["count_path"],
            dtype={self.zone_field: str, **{attribute: str for attribute in self.attributes}},
        )
        required = set([self.zone_field, *self.attributes, "count"])
        if required - set(frame.columns):
            raise ValueError(f"{row['count_path']} lacks {sorted(required - set(frame.columns))}")
        result = frame[[self.zone_field, *self.attributes, "count"]].copy()
        result["count"] = pd.to_numeric(result["count"], errors="raise")
        if (result["count"] < 0).any() or not np.allclose(result["count"], np.round(result["count"])):
            raise ValueError(f"Invalid condensed counts in {row['count_path']}")
        result["count"] = result["count"].astype("int64")
        unsupported = {
            attribute: sorted(set(result[attribute]) - set(self.states[attribute]))
            for attribute in self.attributes
        }
        unsupported = {key: value for key, value in unsupported.items() if value}
        if unsupported:
            raise ValueError(
                f"Synthetic states outside the census universe in {row['count_path']}: {unsupported}"
            )
        for attribute in self.attributes:
            result[attribute] = pd.Categorical(result[attribute], categories=self.states[attribute])
        unknown_zones = set(result[self.zone_field]) - set(self.zones)
        if unknown_zones:
            raise ValueError(f"Unknown zones in {row['count_path']}: {sorted(unknown_zones)[:5]}")
        self.counts[key] = result
        return result

    def _operational_census(self) -> pd.DataFrame:
        operational = self.census.copy()
        is_negative = (operational["attribute"] == "hhinc") & (operational["state"] == "Negative income")
        negative = operational.loc[is_negative, [self.zone_field, "census_count"]].set_index(self.zone_field)["census_count"]
        is_nil = (operational["attribute"] == "hhinc") & (operational["state"] == "Nil income")
        operational.loc[is_nil, "census_count"] += operational.loc[is_nil, self.zone_field].map(negative).to_numpy()
        operational.loc[is_negative, "census_count"] = 0
        return operational

    def census_metrics(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        cells: list[pd.DataFrame] = []
        summaries: list[dict[str, Any]] = []
        zone_headline: list[dict[str, Any]] = []
        operational = self._operational_census()
        for catalog_row in self.catalog.to_dict("records"):
            counts = self._load_counts(catalog_row)
            evaluations = [("canonical", self.census)]
            if catalog_row["run_id"] == "ipf_original_weighted_seed":
                evaluations.append(("operational_folded", operational))
            for evaluation, census in evaluations:
                for attribute in self.attributes:
                    target = census[census["attribute"] == attribute].copy()
                    generated = (
                        counts.groupby([self.zone_field, attribute], observed=True)["count"]
                        .sum().rename("synthetic_count").reset_index().rename(columns={attribute: "state"})
                    )
                    aligned = target.merge(generated, how="left", on=[self.zone_field, "state"])
                    aligned["synthetic_count"] = aligned["synthetic_count"].fillna(0).astype(float)
                    aligned["signed_error"] = aligned["synthetic_count"] - aligned["census_count"]
                    aligned["absolute_error"] = aligned["signed_error"].abs()
                    aligned["squared_error"] = aligned["signed_error"] ** 2
                    aligned["relative_error"] = _safe_relative(aligned["signed_error"], aligned["census_count"])
                    aligned["census_proportion"] = aligned["census_count"] / aligned[self.zone_field].map(self.zone_totals)
                    aligned["synthetic_proportion"] = aligned["synthetic_count"] / aligned.groupby(self.zone_field)["synthetic_count"].transform("sum").clip(lower=1)
                    aligned["exact_cell"] = aligned["signed_error"] == 0
                    aligned["census_positive_omitted"] = (aligned["census_count"] > 0) & (aligned["synthetic_count"] == 0)
                    aligned["synthetic_mass_in_census_zero"] = np.where(aligned["census_count"] == 0, aligned["synthetic_count"], 0)
                    aligned.insert(0, "evaluation", evaluation)
                    aligned.insert(0, "controlled", attribute in str(catalog_row["controlled_attributes"]).split(","))
                    aligned["attribute"] = attribute
                    aligned.insert(0, "repeat", catalog_row["repeat"])
                    aligned.insert(0, "method", catalog_row["method"])
                    aligned.insert(0, "run_id", catalog_row["run_id"])
                    cells.append(aligned)
                    zone_values: list[dict[str, Any]] = []
                    for zone, group in aligned.groupby(self.zone_field, sort=False):
                        metrics = distribution_metrics(group["census_count"], group["synthetic_count"], smoothing=self.options.smoothing)
                        zone_values.append(
                            {
                                self.zone_field: zone,
                                "rmse_count": float(np.sqrt(group["squared_error"].mean())),
                                "mae_count": float(group["absolute_error"].mean()),
                                "rmse_proportion": float(np.sqrt(np.mean(np.square(group["synthetic_proportion"] - group["census_proportion"])))),
                                "exact_zone_attribute": bool(group["exact_cell"].all()),
                                "max_absolute_error": float(group["absolute_error"].max()),
                                "census_positive_omitted_cells": int(group["census_positive_omitted"].sum()),
                                "synthetic_mass_in_census_zero": float(group["synthetic_mass_in_census_zero"].sum()),
                                **metrics,
                            }
                        )
                    zone_frame = pd.DataFrame(zone_values)
                    weights = zone_frame[self.zone_field].map(self.zone_totals).astype(float)
                    for aggregation in ("macro", "household_weighted_micro"):
                        record: dict[str, Any] = {
                            "run_id": catalog_row["run_id"], "method": catalog_row["method"],
                            "repeat": catalog_row["repeat"], "evaluation": evaluation,
                            "attribute": attribute,
                            "controlled": attribute in str(catalog_row["controlled_attributes"]).split(","),
                            "aggregation": aggregation,
                            "exact_cell_rate": float(aligned["exact_cell"].mean()),
                            "state_bias_mean": float(aligned["signed_error"].mean()),
                            "absolute_error_p95": float(aligned["absolute_error"].quantile(0.95)),
                            "absolute_error_max": float(aligned["absolute_error"].max()),
                            "exact_zone_attribute_rate": float(zone_frame["exact_zone_attribute"].mean()),
                        }
                        for metric in ["rmse_count", "mae_count", "rmse_proportion", "total_variation", "js_distance", "js_divergence", "hellinger", "max_absolute_error", "census_positive_omitted_cells", "synthetic_mass_in_census_zero"]:
                            record[metric] = float(np.average(zone_frame[metric], weights=None if aggregation == "macro" else weights))
                        summaries.append(record)
                    for zone_record in zone_values:
                        zone_headline.append({
                            "run_id": catalog_row["run_id"], "method": catalog_row["method"],
                            "repeat": catalog_row["repeat"], "evaluation": evaluation,
                            "attribute": attribute, **zone_record,
                        })
                generated_totals = counts.groupby(self.zone_field)["count"].sum().reindex(self.zones, fill_value=0)
                total_error = generated_totals - self.zone_totals.reindex(self.zones)
                complete = pd.concat(cells[-len(self.attributes):]).groupby(self.zone_field)["exact_cell"].all()
                summaries.append({
                    "run_id": catalog_row["run_id"], "method": catalog_row["method"],
                    "repeat": catalog_row["repeat"], "evaluation": evaluation,
                    "attribute": "__all__", "controlled": False, "aggregation": "suite",
                    "household_total_exact_zone_rate": float((total_error == 0).mean()),
                    "household_total_max_absolute_error": float(total_error.abs().max()),
                    "complete_zone_exact_rate": float(complete.mean()),
                })
        return pd.concat(cells, ignore_index=True), pd.DataFrame(summaries), pd.DataFrame(zone_headline)

    def joint_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        references = {"weighted_seed": self.seed_joint, "bn_pool": self.bn_joint}
        for catalog_row in self.catalog.to_dict("records"):
            counts = self._load_counts(catalog_row)
            generated_full = condense(counts, self.attributes, "count")
            for order in range(2, len(self.attributes) + 1):
                for subset_tuple in combinations(self.attributes, order):
                    subset = list(subset_tuple)
                    universe = _full_index(self.states, subset)
                    generated = _project_joint(generated_full, subset).reindex(universe, fill_value=0)
                    for reference_name, full_joint in references.items():
                        reference = _project_joint(full_joint, subset).reindex(universe, fill_value=0)
                        rows.append({
                            "run_id": catalog_row["run_id"], "method": catalog_row["method"],
                            "repeat": catalog_row["repeat"], "reference": reference_name,
                            "interaction_order": order, "attributes": "|".join(subset),
                            "cells": len(universe),
                            **distribution_metrics(reference, generated, smoothing=self.options.smoothing),
                        })
        return pd.DataFrame(rows)

    def aligned_five_way_distributions(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        sources: list[tuple[str, str, int, pd.Series]] = [
            ("weighted_seed", "reference", 0, self.seed_joint),
            ("bn_pool", "reference", 0, self.bn_joint),
        ]
        for row in self.catalog.to_dict("records"):
            sources.append((
                row["run_id"], row["method"], int(row["repeat"]),
                condense(self._load_counts(row), self.attributes, "count"),
            ))
        for run_id, method, repeat, values in sources:
            aligned = values.reindex(self.universe, fill_value=0).astype(float)
            frame = aligned.index.to_frame(index=False)
            frame.insert(0, "repeat", repeat)
            frame.insert(0, "method", method)
            frame.insert(0, "run_id", run_id)
            frame["weight_or_count"] = aligned.to_numpy()
            frame["probability"] = aligned.to_numpy() / aligned.sum()
            frames.append(frame)
        return pd.concat(frames, ignore_index=True)

    def relationship_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        sources: list[tuple[dict[str, Any], str, pd.Series]] = []
        for name, joint in (("weighted_seed", self.seed_joint), ("bn_pool", self.bn_joint)):
            sources.append(({"run_id": name, "method": "reference", "repeat": 0}, name, joint))
        for row in self.catalog.to_dict("records"):
            sources.append((row, "synthetic", condense(self._load_counts(row), self.attributes, "count")))
        seed_pairs = {
            pair: _project_joint(self.seed_joint, list(pair)).reindex(_full_index(self.states, list(pair)), fill_value=0)
            for pair in combinations(self.attributes, 2)
        }
        pair_associations: dict[tuple[str, int, str, str], tuple[float, float]] = {}
        for metadata, source_type, full_joint in sources:
            for left, right in combinations(self.attributes, 2):
                index = _full_index(self.states, [left, right])
                vector = _project_joint(full_joint, [left, right]).reindex(index, fill_value=0)
                table = vector.to_numpy().reshape(len(self.states[left]), len(self.states[right]))
                cv = cramers_v(table)
                mi, nmi = mutual_information(table)
                pair_associations[(str(metadata["run_id"]), int(metadata["repeat"]), left, right)] = (cv, nmi)
                seed_cv, seed_nmi = pair_associations.get(("weighted_seed", 0, left, right), (math.nan, math.nan))
                rows.append({
                    "run_id": metadata["run_id"], "method": metadata["method"],
                    "repeat": metadata["repeat"], "source": source_type,
                    "metric_scope": "association", "conditioning_attribute": left,
                    "outcome_attribute": right, "cramers_v": cv,
                    "mutual_information": mi, "normalized_mutual_information": nmi,
                    "absolute_cramers_v_error_vs_seed": abs(cv - seed_cv),
                    "absolute_nmi_error_vs_seed": abs(nmi - seed_nmi),
                })
                # Both conditional directions are retained.
                for conditioning, outcome in ((left, right), (right, left)):
                    joint_index = _full_index(self.states, [conditioning, outcome])
                    values = _project_joint(full_joint, [conditioning, outcome]).reindex(joint_index, fill_value=0)
                    if source_type == "synthetic":
                        pair_key = (conditioning, outcome) if self.attributes.index(conditioning) < self.attributes.index(outcome) else (outcome, conditioning)
                        reference = seed_pairs[pair_key]
                        if list(reference.index.names) != [conditioning, outcome]:
                            reference = reference.reorder_levels([conditioning, outcome]).sort_index().reindex(joint_index, fill_value=0)
                        weighted_metrics = []
                        for state in self.states[conditioning]:
                            ref_slice = reference.xs(state, level=conditioning)
                            syn_slice = values.xs(state, level=conditioning)
                            dm = distribution_metrics(ref_slice, syn_slice, smoothing=self.options.smoothing)
                            weighted_metrics.append((float(ref_slice.sum()), dm))
                        total_weight = sum(item[0] for item in weighted_metrics)
                        rows.append({
                            "run_id": metadata["run_id"], "method": metadata["method"],
                            "repeat": metadata["repeat"], "source": source_type,
                            "metric_scope": "conditional_vs_seed",
                            "conditioning_attribute": conditioning, "outcome_attribute": outcome,
                            "conditional_js_distance": sum(w * m["js_distance"] for w, m in weighted_metrics) / total_weight,
                            "conditional_total_variation": sum(w * m["total_variation"] for w, m in weighted_metrics) / total_weight,
                        })
        result = pd.DataFrame(rows)
        # Per-run association-matrix errors (upper triangle vectors).
        extras = []
        seed_vector = np.array([pair_associations[("weighted_seed", 0, *pair)] for pair in combinations(self.attributes, 2)])
        for row in self.catalog.to_dict("records"):
            vector = np.array([pair_associations[(row["run_id"], row["repeat"], *pair)] for pair in combinations(self.attributes, 2)])
            delta = vector - seed_vector
            extras.append({
                "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                "source": "synthetic", "metric_scope": "association_matrix_vs_seed",
                "association_frobenius_error": float(np.linalg.norm(delta)),
                "association_mean_absolute_error": float(np.abs(delta).mean()),
            })
        return pd.concat([result, pd.DataFrame(extras)], ignore_index=True)

    def _support_tier(self, index: pd.MultiIndex) -> np.ndarray:
        seed_support, bn_support = set(self.seed_joint.index), set(self.bn_joint.index)
        return np.array([
            "seed_supported" if key in seed_support else "bn_only_supported" if key in bn_support else "unconfirmed"
            for key in index
        ])

    def _nearest_support(self, novel: pd.DataFrame, reference: pd.DataFrame, prefix: str) -> pd.DataFrame:
        if novel.empty:
            return pd.DataFrame(index=novel.index)
        keys = [tuple(str(row[a]) for a in self.attributes) for _, row in novel.iterrows()]
        cache = self._nearest_caches[prefix]
        missing_keys = list(dict.fromkeys(key for key in keys if key not in cache))
        if not missing_keys:
            return pd.DataFrame([cache[key] for key in keys], index=novel.index)
        state_codes = {a: {s: i for i, s in enumerate(self.states[a])} for a in self.attributes}
        missing = pd.DataFrame(missing_keys, columns=self.attributes)
        novel_codes = np.column_stack([missing[a].map(state_codes[a]).to_numpy() for a in self.attributes])
        reference_unique = reference[self.attributes].drop_duplicates()
        ref_codes = np.column_stack([reference_unique[a].map(state_codes[a]).to_numpy() for a in self.attributes])
        minimum = np.full(len(missing), len(self.attributes) + 1, dtype=np.int8)
        differences = np.zeros((len(missing), len(self.attributes)), dtype=bool)
        for start in range(0, len(missing), 512):
            block = novel_codes[start:start + 512]
            distance = (block[:, None, :] != ref_codes[None, :, :]).sum(axis=2)
            nearest_index = distance.argmin(axis=1)
            minimum[start:start + len(block)] = distance[np.arange(len(block)), nearest_index]
            differences[start:start + len(block)] = block != ref_codes[nearest_index]
        result = pd.DataFrame({f"nearest_{prefix}_hamming": minimum})
        for position, attribute in enumerate(self.attributes):
            result[f"nearest_{prefix}_differs_{attribute}"] = differences[:, position]
        for key, record in zip(missing_keys, result.to_dict("records")):
            cache[key] = record
        return pd.DataFrame([cache[key] for key in keys], index=novel.index)

    def support_metrics(self) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        summaries: list[dict[str, Any]] = []
        unconfirmed: list[pd.DataFrame] = []
        zones: list[dict[str, Any]] = []
        seed_support, bn_support = set(self.seed_joint.index), set(self.bn_joint.index)
        seed_total, bn_total = self.seed_joint.sum(), self.bn_joint.sum()
        seed_quantiles = pd.qcut(self.seed_joint.rank(method="first"), 4, labels=["Q1_rarest", "Q2", "Q3", "Q4_common"])
        for row in self.catalog.to_dict("records"):
            counts = self._load_counts(row)
            joint = condense(counts, self.attributes, "count")
            tier = pd.Series(self._support_tier(joint.index), index=joint.index, name="tier")
            total = joint.sum()
            for tier_name in ("seed_supported", "bn_only_supported", "unconfirmed"):
                mask = tier == tier_name
                summaries.append({
                    "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                    "scope": "global", "tier": tier_name,
                    "unique_combinations": int(mask.sum()),
                    "unique_combination_percentage": float(mask.mean() * 100),
                    "household_count": float(joint[mask].sum()),
                    "population_mass_percentage": float(joint[mask].sum() / total * 100),
                })
            generated_support = set(joint.index)
            common_seed = self.seed_joint.index.isin(generated_support)
            common_bn = self.bn_joint.index.isin(generated_support)
            summaries.append({
                "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                "scope": "coverage", "tier": "all",
                "seed_support_recall": float(common_seed.mean()),
                "weighted_seed_mass_coverage": float(self.seed_joint[common_seed].sum() / seed_total),
                "bn_pool_support_recall": float(common_bn.mean()),
                "bn_pool_weighted_coverage": float(self.bn_joint[common_bn].sum() / bn_total),
                "seed_precision_unique": float(sum(x in seed_support for x in generated_support) / len(generated_support)),
                "seed_precision_mass": float(joint[tier == "seed_supported"].sum() / total),
            })
            # Rare retention and amplification by weighted-seed quartile.
            generated_prob = joint / total
            seed_prob = self.seed_joint / seed_total
            for quantile in seed_quantiles.unique():
                qindex = seed_quantiles[seed_quantiles == quantile].index
                reproduced = qindex.isin(generated_support)
                ratios = generated_prob.reindex(qindex, fill_value=0) / seed_prob.reindex(qindex)
                summaries.append({
                    "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                    "scope": "seed_weight_quantile", "tier": str(quantile),
                    "seed_support_recall": float(reproduced.mean()),
                    "mean_amplification_ratio": float(ratios.mean()),
                    "median_amplification_ratio": float(ratios.median()),
                })
            # Zone provenance and composition.
            annotated = counts.copy()
            idx = pd.MultiIndex.from_frame(annotated[self.attributes])
            annotated["tier"] = self._support_tier(idx)
            zonal_totals = annotated.groupby(self.zone_field)["count"].sum()
            for (zone, tier_name), group in annotated.groupby([self.zone_field, "tier"], observed=True):
                zones.append({
                    "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                    self.zone_field: zone, "tier": tier_name,
                    "unique_combinations": len(group), "household_count": int(group["count"].sum()),
                    "population_mass_percentage": float(group["count"].sum() / zonal_totals.loc[zone] * 100),
                })
            for (tier_name, attribute, state), group in pd.concat(
                [
                    annotated[["tier", attribute, "count"]]
                    .rename(columns={attribute: "state"})
                    .assign(attribute=attribute)
                    for attribute in self.attributes
                ],
                ignore_index=True,
            ).groupby(["tier", "attribute", "state"], observed=True):
                tier_total = annotated.loc[annotated["tier"] == tier_name, "count"].sum()
                summaries.append({
                    "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                    "scope": "tier_state_composition", "tier": tier_name,
                    "attribute": attribute, "state": state,
                    "household_count": float(group["count"].sum()),
                    "population_mass_percentage": float(group["count"].sum() / tier_total * 100) if tier_total else 0.0,
                })
            top = annotated.groupby(["tier", *self.attributes], observed=True)["count"].sum().reset_index()
            top["rank_within_tier"] = top.groupby("tier")["count"].rank(method="first", ascending=False)
            for record in top[top["rank_within_tier"] <= 20].to_dict("records"):
                summaries.append({
                    "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                    "scope": "top_combination", "household_count": record.pop("count"), **record,
                })
            novel = annotated[annotated["tier"] == "unconfirmed"].groupby(self.attributes, observed=True)["count"].sum().reset_index()
            if not novel.empty:
                nearest_seed = self._nearest_support(novel, self.seed, "seed")
                nearest_bn = self._nearest_support(novel, self.bn_pool, "bn_pool")
                novel = pd.concat([novel, nearest_seed, nearest_bn], axis=1)
                novel_total = float(novel["count"].sum())
                for prefix in ("seed", "bn_pool"):
                    for distance, group in novel.groupby(f"nearest_{prefix}_hamming"):
                        summaries.append({
                            "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                            "scope": "unconfirmed_nearest_support", "tier": prefix,
                            "hamming_distance": int(distance), "household_count": float(group["count"].sum()),
                            "population_mass_percentage": float(group["count"].sum() / novel_total * 100),
                        })
                    for attribute in self.attributes:
                        mask = novel[f"nearest_{prefix}_differs_{attribute}"]
                        summaries.append({
                            "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                            "scope": "unconfirmed_nearest_attribute_difference", "tier": prefix,
                            "attribute": attribute,
                            "household_count": float(novel.loc[mask, "count"].sum()),
                            "population_mass_percentage": float(novel.loc[mask, "count"].sum() / novel_total * 100),
                        })
                novel.insert(0, "repeat", row["repeat"])
                novel.insert(0, "method", row["method"])
                novel.insert(0, "run_id", row["run_id"])
                unconfirmed.append(novel)
            # Novel pair/triple supports.
            for order in (2, 3):
                for subset in combinations(self.attributes, order):
                    generated_sub = condense(counts, list(subset), "count")
                    seed_sub = set(condense(self.seed, list(subset), "_reference_weight").index)
                    bn_sub = set(condense(self.bn_pool, list(subset), "_reference_weight").index)
                    new_seed = ~generated_sub.index.isin(seed_sub)
                    new_both = ~generated_sub.index.isin(seed_sub | bn_sub)
                    summaries.append({
                        "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                        "scope": f"novel_{order}_way", "tier": "|".join(subset),
                        "novel_vs_seed_unique": int(new_seed.sum()),
                        "novel_vs_seed_mass": float(generated_sub[new_seed].sum() / total),
                        "novel_vs_seed_and_bn_unique": int(new_both.sum()),
                        "novel_vs_seed_and_bn_mass": float(generated_sub[new_both].sum() / total),
                    })
        empty_novel = pd.DataFrame(columns=["run_id", "method", "repeat", *self.attributes, "count"])
        return pd.DataFrame(summaries), pd.concat(unconfirmed, ignore_index=True) if unconfirmed else empty_novel, pd.DataFrame(zones)

    def diversity_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        universe_size = int(np.prod([len(self.states[a]) for a in self.attributes]))
        seed_support = set(self.seed_joint.index)
        for row in self.catalog.to_dict("records"):
            counts = self._load_counts(row)
            for scope, zone, frame in [("global", None, counts), *[("zone", z, g) for z, g in counts.groupby(self.zone_field, sort=False)]]:
                joint = condense(frame, self.attributes, "count")
                record = {
                    "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                    "scope": scope, self.zone_field: zone, "population": float(joint.sum()),
                    **entropy_metrics(joint, universe_size),
                    "newly_occupied_seed_zero_cells": int(sum(key not in seed_support for key in joint.index)),
                    "structural_zero_preservation": float(1 - sum(key not in seed_support for key in joint.index) / (universe_size - len(seed_support))),
                    "occupied_cells_per_1000_households": float(len(joint) / joint.sum() * 1000),
                }
                for threshold in self.options.rare_thresholds:
                    record[f"count_le_{threshold}_cell_fraction"] = float((joint <= threshold).mean())
                    record[f"count_le_{threshold}_mass_fraction"] = float(joint[joint <= threshold].sum() / joint.sum())
                rows.append(record)
        return pd.DataFrame(rows)

    def repeat_metrics(self, headline: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for run_id, catalog in self.catalog.groupby("run_id", sort=False):
            method = catalog["method"].iloc[0]
            repeats = catalog["repeat"].tolist()
            if len(repeats) < 2:
                rows.append({"run_id": run_id, "method": method, "scope": "availability", "repeat_variability_available": False})
                continue
            vectors: dict[int, pd.Series] = {}
            for record in catalog.to_dict("records"):
                vectors[int(record["repeat"])] = condense(self._load_counts(record), self.attributes, "count").reindex(self.universe, fill_value=0)
            for first, second in combinations(repeats, 2):
                rows.append({
                    "run_id": run_id, "method": method, "scope": "pairwise_repeat",
                    "repeat_a": first, "repeat_b": second,
                    **distribution_metrics(vectors[first], vectors[second], smoothing=self.options.smoothing),
                })
            matrix = pd.DataFrame(vectors)
            mean = matrix.mean(axis=1)
            cell_stats = pd.DataFrame({
                "inclusion_probability": (matrix > 0).mean(axis=1),
                "count_mean": mean,
                "count_variance": matrix.var(axis=1, ddof=1),
                "count_cv": matrix.std(axis=1, ddof=1) / mean.replace(0, np.nan),
            })
            rows.append({
                "run_id": run_id, "method": method, "scope": "cell_stability",
                "mean_inclusion_probability": float(cell_stats["inclusion_probability"].mean()),
                "mean_positive_cell_cv": float(cell_stats.loc[cell_stats["count_mean"] > 0, "count_cv"].mean()),
                "exact_duplicate_hash_rate": float(1 - catalog["output_sha256"].nunique() / len(catalog)),
            })
            # Variation of every numeric headline metric.
            subset = headline[(headline["run_id"] == run_id) & (headline["evaluation"] == "canonical")]
            numeric = [column for column in subset.select_dtypes(include="number") if column != "repeat"]
            for keys, group in subset.groupby(["attribute", "aggregation"], dropna=False):
                for metric in numeric:
                    values = group[metric].dropna()
                    if len(values):
                        rows.append({
                            "run_id": run_id, "method": method, "scope": "headline_metric",
                            "attribute": keys[0], "aggregation": keys[1], "metric": metric,
                            **summary_statistics(values),
                        })
            seed_values = catalog.set_index("repeat")["sampling_seed"]
            tv = subset[subset["aggregation"] == "macro"].groupby("repeat")["total_variation"].mean()
            if len(tv.dropna()) >= 3:
                aligned = pd.concat([seed_values.rename("sampling_seed"), tv.rename("metric")], axis=1).dropna()
                rows.append({"run_id": run_id, "method": method, "scope": "seed_correlation_diagnostic", "sampling_seed_metric_correlation": float(aligned.corr().iloc[0, 1])})
        return pd.DataFrame(rows)

    def repeat_cell_metrics(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for run_id, catalog in self.catalog.groupby("run_id", sort=False):
            if len(catalog) < 2:
                continue
            vectors = {
                int(row["repeat"]): condense(self._load_counts(row), self.attributes, "count").reindex(self.universe, fill_value=0)
                for row in catalog.to_dict("records")
            }
            matrix = pd.DataFrame(vectors)
            mean = matrix.mean(axis=1)
            result = matrix.index.to_frame(index=False)
            result.insert(0, "method", catalog["method"].iloc[0])
            result.insert(0, "run_id", run_id)
            result["inclusion_probability"] = (matrix > 0).mean(axis=1).to_numpy()
            result["count_mean"] = mean.to_numpy()
            result["count_variance"] = matrix.var(axis=1, ddof=1).to_numpy()
            result["count_cv"] = (matrix.std(axis=1, ddof=1) / mean.replace(0, np.nan)).to_numpy()
            frames.append(result)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def repeat_zone_metrics(
        self, zone_metrics: pd.DataFrame, zone_provenance: pd.DataFrame
    ) -> pd.DataFrame:
        marginal = (
            zone_metrics[zone_metrics["evaluation"] == "canonical"]
            .groupby(["run_id", "method", self.zone_field, "repeat"])["total_variation"]
            .mean().reset_index()
        )
        marginal_summary = marginal.groupby(["run_id", "method", self.zone_field])["total_variation"].agg(
            marginal_tv_mean="mean", marginal_tv_std="std", marginal_tv_min="min", marginal_tv_max="max"
        ).reset_index()
        provenance = zone_provenance.groupby(
            ["run_id", "method", self.zone_field, "tier"]
        )["population_mass_percentage"].agg(
            provenance_mass_mean="mean", provenance_mass_std="std",
            provenance_mass_min="min", provenance_mass_max="max",
        ).reset_index()
        return provenance.merge(marginal_summary, how="left", on=["run_id", "method", self.zone_field])

    def runtime_metrics(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        shared_training: dict[str, float] = {}
        for run_id in self.catalog["run_id"].unique():
            path = self.experiment_root / run_id / "shared_model_manifest.json"
            if path.is_file():
                shared_training[run_id] = float(json.loads(path.read_text())["training_and_save_seconds"])
        for row in self.catalog.to_dict("records"):
            manifest = json.loads(Path(row["manifest_path"]).read_text())
            diagnostics = manifest.get("diagnostics", {})
            timings = diagnostics.get("timings", {})
            repeat_dir = Path(row["count_path"]).parent
            zone_path = diagnostics.get("zone_diagnostics")
            zone_seconds = pd.Series(dtype=float)
            load_seconds = 0.0
            if zone_path and (repeat_dir / zone_path).is_file():
                zone_data = pd.read_csv(repeat_dir / zone_path)
                for candidate in ("runtime_seconds", "sampling_seconds"):
                    if candidate in zone_data:
                        zone_seconds = zone_data[candidate].astype(float)
                        break
                if "worker_model_load_seconds" in zone_data:
                    load_seconds = float(zone_data[["worker_pid", "worker_model_load_seconds"]].drop_duplicates()["worker_model_load_seconds"].sum())
            total_households = int(self._load_counts(row)["count"].sum())
            runtime = float(manifest["runtime_seconds"])
            model_path = diagnostics.get("model_path")
            model_size = Path(model_path).stat().st_size if model_path and Path(model_path).is_file() else math.nan
            rows.append({
                "run_id": row["run_id"], "method": row["method"], "repeat": row["repeat"],
                "end_to_end_seconds": runtime,
                "shared_training_seconds": shared_training.get(row["run_id"], 0.0),
                "model_prepare_seconds": timings.get("model_prepare_seconds", 0.0),
                "zonal_seconds_recorded": sum(float(v) for k, v in timings.items() if "zonal" in k or "sampling" in k),
                "output_write_seconds": timings.get("output_write_seconds", 0.0),
                "artifact_hash_seconds": timings.get("artifact_hash_seconds", 0.0),
                "worker_model_loading_seconds": load_seconds,
                "zone_time_mean": float(zone_seconds.mean()) if len(zone_seconds) else math.nan,
                "zone_time_median": float(zone_seconds.median()) if len(zone_seconds) else math.nan,
                "zone_time_p95": float(zone_seconds.quantile(0.95)) if len(zone_seconds) else math.nan,
                "zone_time_max": float(zone_seconds.max()) if len(zone_seconds) else math.nan,
                "zone_time_imbalance_ratio": float(zone_seconds.max() / zone_seconds.mean()) if len(zone_seconds) and zone_seconds.mean() else math.nan,
                "households_per_second": total_households / runtime,
                "zones_per_second": len(self.zones) / runtime,
                "worker_efficiency_proxy": float(zone_seconds.sum() / (float(row["workers"]) * runtime)) if len(zone_seconds) and row["workers"] else math.nan,
                "model_artifact_bytes": model_size,
                "output_file_bytes": Path(row["count_path"]).stat().st_size,
                "condensed_rows": len(self._load_counts(row)),
                "cold_start": int(row["repeat"]) == 1,
            })
        result = pd.DataFrame(rows)
        for run_id, group in result.groupby("run_id"):
            result.loc[group.index, "runtime_cv_across_repeats"] = group["end_to_end_seconds"].std(ddof=1) / group["end_to_end_seconds"].mean() if len(group) > 1 else math.nan
            result.loc[group.index, "ten_sample_suite_seconds"] = group["end_to_end_seconds"].sum() + group["shared_training_seconds"].max()
            result.loc[group.index, "amortized_seconds_per_sample"] = result.loc[group.index, "ten_sample_suite_seconds"] / len(group)
            result.loc[group.index, "warm_sampling_mean_seconds"] = group.loc[~group["cold_start"], "end_to_end_seconds"].mean()
        return result

    def internal_diagnostics(self) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for row in self.catalog.to_dict("records"):
            repeat_dir = Path(row["count_path"]).parent
            manifest = json.loads(Path(row["manifest_path"]).read_text())
            diagnostic = manifest.get("diagnostics", {}).get("zone_diagnostics")
            diagnostic_files = [repeat_dir / diagnostic] if diagnostic else []
            diagnostic_files.extend((repeat_dir / "diagnostics").glob("*.csv") if (repeat_dir / "diagnostics").is_dir() else [])
            seen: set[Path] = set()
            for diagnostic_path in diagnostic_files:
                if not diagnostic_path.is_file() or diagnostic_path in seen:
                    continue
                seen.add(diagnostic_path)
                frame = pd.read_csv(diagnostic_path)
                frame.insert(0, "diagnostic_source", str(diagnostic_path.relative_to(repeat_dir)))
                frame.insert(0, "repeat", row["repeat"])
                frame.insert(0, "method", row["method"])
                frame.insert(0, "run_id", row["run_id"])
                frames.append(frame)
        for run_id in self.catalog["run_id"].unique():
            for path in (self.experiment_root / run_id / "model").glob("*.metadata.json"):
                metadata = json.loads(path.read_text())
                frames.append(pd.DataFrame([{
                    "run_id": run_id,
                    "method": self.catalog.loc[self.catalog["run_id"] == run_id, "method"].iloc[0],
                    "repeat": 0,
                    "diagnostic_source": str(path.relative_to(self.experiment_root / run_id)),
                    "model_metadata_json": json.dumps(metadata, sort_keys=True),
                }]))
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def statistical_comparisons(self, zone_metrics: pd.DataFrame) -> pd.DataFrame:
        rng = np.random.default_rng(self.options.analysis_seed)
        rows: list[dict[str, Any]] = []
        canonical = zone_metrics[zone_metrics["evaluation"] == "canonical"]
        scopes = [(attribute, canonical[canonical["attribute"] == attribute]) for attribute in self.attributes]
        scopes.append(("all_attributes", canonical))
        for scope, base in scopes:
            base = base.groupby(["run_id", "method", "repeat", self.zone_field], as_index=False)["total_variation"].mean()
            method_zone = base.groupby(["run_id", self.zone_field])["total_variation"].mean().unstack(0)
            for left, right in combinations(method_zone.columns, 2):
                paired = method_zone[[left, right]].dropna()
                delta = (paired[left] - paired[right]).to_numpy()
                zones = paired.index.to_list()
                left_matrix = base[base["run_id"] == left].pivot_table(
                    index="repeat", columns=self.zone_field, values="total_variation"
                ).reindex(columns=zones).to_numpy()
                right_matrix = base[base["run_id"] == right].pivot_table(
                    index="repeat", columns=self.zone_field, values="total_variation"
                ).reindex(columns=zones).to_numpy()
                boot = np.empty(self.options.bootstrap_replicates)
                for index in range(self.options.bootstrap_replicates):
                    zone_indices = rng.integers(0, len(zones), size=len(zones))
                    left_repeats = rng.integers(0, len(left_matrix), size=len(left_matrix))
                    right_repeats = rng.integers(0, len(right_matrix), size=len(right_matrix))
                    boot[index] = (
                        left_matrix[left_repeats][:, zone_indices].mean()
                        - right_matrix[right_repeats][:, zone_indices].mean()
                    )
                try:
                    pvalue = float(wilcoxon(delta).pvalue) if np.any(delta) else 1.0
                except ValueError:
                    pvalue = math.nan
                rows.append({
                    "metric": "zone_total_variation", "attribute_scope": scope,
                    "run_a": left, "run_b": right, "paired_zones": len(delta),
                    "mean_paired_difference_a_minus_b": float(delta.mean()),
                    "median_paired_difference": float(np.median(delta)),
                    "standardized_paired_effect": float(delta.mean() / delta.std(ddof=1)) if delta.std(ddof=1) else 0.0,
                    "bootstrap_ci95_low": float(np.quantile(boot, 0.025)),
                    "bootstrap_ci95_high": float(np.quantile(boot, 0.975)),
                    "wilcoxon_pvalue": pvalue,
                    "bootstrap_replicates": self.options.bootstrap_replicates,
                    "analysis_seed": self.options.analysis_seed,
                })
        adjusted = _holm([row["wilcoxon_pvalue"] for row in rows])
        for row, value in zip(rows, adjusted):
            row["holm_adjusted_pvalue"] = value
        return pd.DataFrame(rows)

    def _method_summary(self, census_summary: pd.DataFrame, joint: pd.DataFrame, support: pd.DataFrame, diversity: pd.DataFrame, runtime: pd.DataFrame) -> pd.DataFrame:
        marginal = census_summary[(census_summary["evaluation"] == "canonical") & (census_summary["aggregation"] == "macro")].groupby(["run_id", "method"]).agg(census_macro_tv=("total_variation", "mean"), census_macro_js=("js_distance", "mean")).reset_index()
        joint5 = joint[(joint["reference"] == "weighted_seed") & (joint["interaction_order"] == 5)].groupby(["run_id", "method"]).agg(joint_seed_js=("js_distance", "mean"), joint_seed_tv=("total_variation", "mean")).reset_index()
        provenance = support[(support["scope"] == "global") & (support["tier"] == "unconfirmed")].groupby(["run_id", "method"]).agg(unconfirmed_mass_percent=("population_mass_percentage", "mean")).reset_index()
        div = diversity[diversity["scope"] == "global"].groupby(["run_id", "method"]).agg(occupied_cells=("occupied_cells", "mean"), effective_combinations=("effective_shannon_count", "mean")).reset_index()
        perf = runtime.groupby(["run_id", "method"]).agg(runtime_seconds=("end_to_end_seconds", "mean"), households_per_second=("households_per_second", "mean")).reset_index()
        result = marginal.merge(joint5).merge(provenance).merge(div).merge(perf)
        # Pareto membership for four requested two-objective views.
        for name, first, second, minimize_first, minimize_second in [
            ("pareto_census_joint", "census_macro_tv", "joint_seed_js", True, True),
            ("pareto_census_unconfirmed", "census_macro_tv", "unconfirmed_mass_percent", True, True),
            ("pareto_joint_diversity", "joint_seed_js", "effective_combinations", True, False),
            ("pareto_accuracy_runtime", "census_macro_tv", "runtime_seconds", True, True),
        ]:
            keep = []
            for _, candidate in result.iterrows():
                a, b = result[first], result[second]
                ca, cb = candidate[first], candidate[second]
                no_worse = (a <= ca if minimize_first else a >= ca) & (b <= cb if minimize_second else b >= cb)
                strictly = (a < ca if minimize_first else a > ca) | (b < cb if minimize_second else b > cb)
                keep.append(not bool((no_worse & strictly).any()))
            result[name] = keep
        return result

    def tradeoff_correlations(self, method_summary: pd.DataFrame) -> pd.DataFrame:
        metrics = [
            "census_macro_tv", "joint_seed_js", "unconfirmed_mass_percent",
            "effective_combinations", "runtime_seconds", "households_per_second",
        ]
        rows = []
        for left, right in combinations(metrics, 2):
            rows.append({
                "metric_a": left, "metric_b": right,
                "pearson_correlation": method_summary[[left, right]].corr(method="pearson").iloc[0, 1],
                "spearman_correlation": method_summary[[left, right]].corr(method="spearman").iloc[0, 1],
                "methods": len(method_summary),
            })
        return pd.DataFrame(rows)

    def ranking_stability(self, census_summary: pd.DataFrame) -> pd.DataFrame:
        values = census_summary[
            (census_summary["evaluation"] == "canonical")
            & (census_summary["aggregation"] == "macro")
        ].groupby(["run_id", "repeat"])["total_variation"].mean().unstack(0)
        repeated = [column for column in values if values[column].notna().sum() > 1]
        ranks = values[repeated].rank(axis=1, method="average")
        rows = []
        for first, second in combinations(ranks.index, 2):
            pair = ranks.loc[[first, second]].T.dropna()
            rows.append({
                "repeat_a": first, "repeat_b": second, "methods_ranked": len(pair),
                "spearman_rank_correlation": pair.corr(method="spearman").iloc[0, 1]
                if len(pair) >= 2 else math.nan,
            })
        return pd.DataFrame(rows)

    def _orphan_audit(self) -> pd.DataFrame:
        path = self.experiment_root / "orphaned_results.json"
        if not path.is_file():
            return pd.DataFrame(columns=["path", "exists", "households", "rows", "sha256"])
        value = json.loads(path.read_text())
        items = value if isinstance(value, list) else value.get("orphaned_results", value.get("orphans", []))
        rows = []
        for item in items:
            paths = item.get("repeat_directories", []) if isinstance(item, dict) else [item]
            for relative in paths:
                candidate = self.experiment_root / relative
                count_path = candidate / "household_counts.csv" if candidate.is_dir() else candidate
                record = {
                    "run_id": item.get("run_id") if isinstance(item, dict) else Path(relative).parts[0],
                    "path": str(candidate), "exists": count_path.is_file(),
                }
                if count_path.is_file() and count_path.name == "household_counts.csv":
                    counts = pd.read_csv(count_path, usecols=["count"])
                    record.update(households=int(counts["count"].sum()), rows=len(counts), sha256=_file_hash(count_path))
                rows.append(record)
        return pd.DataFrame(rows)

    def _figures(self, method_summary: pd.DataFrame, census_summary: pd.DataFrame, joint: pd.DataFrame, support: pd.DataFrame, runtime: pd.DataFrame) -> list[str]:
        if not self.options.write_figures:
            return []
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            return []
        figures = self.output_dir / "figures"
        figures.mkdir(exist_ok=True)
        written: list[str] = []
        def save(name: str) -> None:
            path = figures / name
            plt.tight_layout(); plt.savefig(path, dpi=180, bbox_inches="tight"); plt.close()
            written.append(str(path.relative_to(self.output_dir)))
        heat = census_summary[(census_summary.evaluation == "canonical") & (census_summary.aggregation == "macro")].pivot_table(index="run_id", columns="attribute", values="total_variation", aggfunc="mean")
        plt.figure(figsize=(10, 6)); sns.heatmap(heat, annot=True, fmt=".3f", cmap="viridis_r"); plt.title("Census marginal total-variation distance")
        save("census_fidelity_heatmap.png")
        order = joint[joint.reference == "weighted_seed"].groupby(["run_id", "interaction_order"])["js_distance"].mean().reset_index()
        plt.figure(figsize=(10, 6)); sns.lineplot(data=order, x="interaction_order", y="js_distance", hue="run_id", marker="o"); plt.title("Joint fidelity by interaction order")
        save("joint_distance_by_order.png")
        prov = support[(support.scope == "global")].pivot_table(index="run_id", columns="tier", values="population_mass_percentage", aggfunc="mean")
        prov.plot(kind="bar", stacked=True, figsize=(11, 6)); plt.ylabel("Population mass (%)"); plt.title("Support provenance")
        save("support_provenance.png")
        plt.figure(figsize=(9, 6)); sns.scatterplot(data=method_summary, x="joint_seed_js", y="unconfirmed_mass_percent", size="runtime_seconds", hue="run_id", legend="brief"); plt.title("Joint fidelity, novelty, and runtime")
        save("accuracy_novelty_runtime.png")
        return written

    def run(self) -> dict[str, Any]:
        started = _utc_now()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        census_cells, census_summary, zone_metrics = self.census_metrics()
        aligned_joint = self.aligned_five_way_distributions()
        joint = self.joint_metrics()
        relationships = self.relationship_metrics()
        support, unconfirmed, zone_provenance = self.support_metrics()
        diversity = self.diversity_metrics()
        repeats = self.repeat_metrics(census_summary)
        repeat_cells = self.repeat_cell_metrics()
        repeat_zones = self.repeat_zone_metrics(zone_metrics, zone_provenance)
        runtime = self.runtime_metrics()
        diagnostics = self.internal_diagnostics()
        statistics = self.statistical_comparisons(zone_metrics)
        method_summary = self._method_summary(census_summary, joint, support, diversity, runtime)
        tradeoffs = self.tradeoff_correlations(method_summary)
        rankings = self.ranking_stability(census_summary)
        orphan = self._orphan_audit()
        outputs = {
            "run_catalog.csv": self.catalog,
            "census_cell_errors.parquet": census_cells,
            "census_metric_summary.csv": census_summary,
            "aligned_five_way_distributions.parquet": aligned_joint,
            "joint_distribution_metrics.csv": joint,
            "conditional_relationship_metrics.csv": relationships,
            "support_provenance_metrics.csv": support,
            "unconfirmed_combinations.parquet": unconfirmed,
            "zone_provenance_metrics.parquet": zone_provenance,
            "diversity_metrics.csv": diversity,
            "repeat_variability.csv": repeats,
            "repeat_cell_variability.parquet": repeat_cells,
            "repeat_zone_variability.csv": repeat_zones,
            "runtime_metrics.csv": runtime,
            "internal_diagnostics.parquet": diagnostics,
            "method_summary.csv": method_summary,
            "statistical_comparisons.csv": statistics,
            "tradeoff_correlations.csv": tradeoffs,
            "ranking_stability.csv": rankings,
            "orphan_audit.csv": orphan,
        }
        for name, frame in outputs.items():
            _write_frame(self.output_dir / name, frame)
        figures = self._figures(method_summary, census_summary, joint, support, runtime)
        manifest = {
            "analysis_version": 1, "started_at": started, "completed_at": _utc_now(),
            "config_path": str(self.config["config_path"]),
            "config_sha256": _file_hash(Path(self.config["config_path"])),
            "experiment_root": str(self.experiment_root), "output_dir": str(self.output_dir),
            "analysis_seed": self.options.analysis_seed,
            "bootstrap_replicates": self.options.bootstrap_replicates,
            "smoothing_constant": self.options.smoothing,
            "rare_thresholds": self.options.rare_thresholds,
            "configured_results": len(self.catalog), "zones": len(self.zones),
            "attributes": self.attributes,
            "state_cardinalities": {key: len(value) for key, value in self.states.items()},
            "cartesian_state_space": len(self.universe),
            "seed_occupied_combinations": len(self.seed_joint),
            "bn_pool_occupied_combinations": len(self.bn_joint),
            "shared_support": len(set(self.seed_joint.index) & set(self.bn_joint.index)),
            "bn_pool_only_support": len(set(self.bn_joint.index) - set(self.seed_joint.index)),
            "seed_only_support": len(set(self.seed_joint.index) - set(self.bn_joint.index)),
            "evidence_roles": {
                "census": "authoritative zonal one-way marginals",
                "weighted_seed": "global joint reference and observed support; not independent truth",
                "bn_pool": "secondary generated-support reference; not ground truth",
            },
            "outputs": {name: {"rows": len(frame), "sha256": _file_hash(self.output_dir / name)} for name, frame in outputs.items()},
            "figures": figures,
            "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__},
            "limitations": [
                "The seed has no zone field and is used only as a global joint reference.",
                "SAA variants have one repeat; repeat variability is unavailable for them.",
                "Peak memory, energy, and true CPU time were not recorded and are excluded.",
            ],
        }
        _write_json(self.output_dir / "analysis_manifest.json", manifest)
        return manifest

