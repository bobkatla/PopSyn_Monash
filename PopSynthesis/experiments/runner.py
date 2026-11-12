"""Execution engine for YAML-defined household experiment suites."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import random
import shutil
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from PopSynthesis.Methods.BN.utils.bn_pool import (
    learn_bn,
    load_bn_artifact,
    sample_exact_marginal as sample_bn_exact,
    save_bn_artifact,
)
from PopSynthesis.Methods.GAN.wgan import (
    HOUSEHOLD_ATTRIBUTES,
    WGANConfig,
    load_wgan_artifact,
    prepare_household_training_data,
    sample_exact_marginal as sample_wgan_exact,
    save_wgan_artifact,
    train_household_wgan,
)


_ZONAL_WORKER_MODEL: Any | None = None
_ZONAL_WORKER_METHOD: str | None = None
_ZONAL_WORKER_LOAD_SECONDS = 0.0


def _initialise_zonal_worker(method: str, model_path: str, device: str | None) -> None:
    global _ZONAL_WORKER_MODEL, _ZONAL_WORKER_METHOD, _ZONAL_WORKER_LOAD_SECONDS
    started = time.perf_counter()
    path = Path(model_path)
    _ZONAL_WORKER_METHOD = method
    if method == "bn":
        _ZONAL_WORKER_MODEL = load_bn_artifact(path)
    else:
        _ZONAL_WORKER_MODEL, _ = load_wgan_artifact(path, device=device)
    _ZONAL_WORKER_LOAD_SECONDS = time.perf_counter() - started


def _parallel_zone_seed(seed: int, position: int) -> int:
    return int(np.random.SeedSequence([seed, position]).generate_state(1)[0])


def _sample_zone_worker(
    position: int,
    zone: object,
    targets: dict[str, int],
    zone_field: str,
    sampling_seed: int,
    bn_method: str,
    chunk_size: int,
) -> tuple[int, pd.DataFrame, dict[str, object]]:
    if _ZONAL_WORKER_MODEL is None or _ZONAL_WORKER_METHOD is None:
        raise RuntimeError("Zonal sampling worker was not initialized")
    started = time.perf_counter()
    zone_seed = _parallel_zone_seed(sampling_seed, position)
    if _ZONAL_WORKER_METHOD == "bn":
        population = sample_bn_exact(
            _ZONAL_WORKER_MODEL,
            "hhsize",
            targets,
            method=bn_method,
            seed=zone_seed,
        )
    else:
        population = sample_wgan_exact(
            _ZONAL_WORKER_MODEL,
            "hhsize",
            targets,
            seed=zone_seed,
            chunk_size=chunk_size,
        )
    population.insert(0, zone_field, zone)
    diagnostic = {
        zone_field: zone,
        "households": int(sum(targets.values())),
        "zone_seed": zone_seed,
        "worker_pid": os.getpid(),
        "worker_model_load_seconds": _ZONAL_WORKER_LOAD_SECONDS,
        "sampling_seconds": time.perf_counter() - started,
    }
    return position, population, diagnostic


def sample_zones_parallel(
    method: str,
    model_path: Path,
    targets: pd.DataFrame,
    zone_field: str,
    sampling_seed: int,
    workers: int,
    bn_method: str = "auto",
    chunk_size: int = 100_000,
    device: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sample zones in isolated processes with deterministic per-zone seeds."""

    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    jobs = [
        (
            position,
            zone,
            {str(state): int(value) for state, value in row.items()},
            zone_field,
            sampling_seed,
            bn_method,
            chunk_size,
        )
        for position, (zone, row) in enumerate(targets.iterrows())
    ]
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_initialise_zonal_worker,
        initargs=(method, str(model_path), device),
    ) as executor:
        results = list(executor.map(_sample_zone_worker, *zip(*jobs)))
    results.sort(key=lambda item: item[0])
    return (
        pd.concat([item[1] for item in results], ignore_index=True),
        pd.DataFrame([item[2] for item in results]),
    )
from PopSynthesis.Methods.IPF.weighted import (
    apply_state_folds,
    fit_weighted_ipf,
    prepare_weighted_cells,
    read_household_census,
)
from PopSynthesis.Methods.IPSF.run import run_saa
from PopSynthesis.experiments.config import (
    ExperimentTask,
    derive_seed,
    expand_runs,
    fingerprint,
    jsonable,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifact_sha256(path: Path) -> str:
    """Hash a model together with its sidecar metadata when present."""

    digest = hashlib.sha256()
    for artifact in (path, path.with_suffix(".metadata.json")):
        if artifact.exists():
            digest.update(artifact.name.encode())
            digest.update(file_sha256(artifact).encode())
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def _versions() -> dict[str, str]:
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "ipfn", "pgmpy", "torch", "pyyaml"):
        try:
            result[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            result[package] = "unknown"
    return result


def _runtime_environment() -> dict[str, object]:
    import torch

    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
    }


def _result_artifact_hashes(output_dir: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file() or path.name == "manifest.json":
            continue
        relative = path.relative_to(output_dir)
        if "zone_artifacts" in relative.parts:
            continue
        hashes[str(relative)] = file_sha256(path)
    return hashes


def _zonal_hhsize_targets(census: pd.DataFrame, zone_field: str) -> pd.DataFrame:
    zone_columns = [column for column in census.columns if column[0] == zone_field]
    size_columns = [column for column in census.columns if column[0] == "hhsize"]
    if len(zone_columns) != 1 or not size_columns:
        raise ValueError("Census must contain one zone field and hhsize controls")
    targets = census.set_index(zone_columns[0])[size_columns].copy()
    targets.columns = [str(column[1]) for column in size_columns]
    targets.index.name = zone_field
    for column in targets:
        values = pd.to_numeric(targets[column], errors="raise")
        if values.isna().any() or not np.isfinite(values).all():
            raise ValueError("hhsize targets must be finite")
        if (values < 0).any() or not np.equal(values, np.floor(values)).all():
            raise ValueError("hhsize targets must be nonnegative integers")
        targets[column] = values.astype("int64")
    return targets


def _condense_population(
    population: pd.DataFrame, attributes: list[str], zone_field: str
) -> pd.DataFrame:
    missing = ({zone_field} | set(attributes)) - set(population.columns)
    if missing:
        raise ValueError(f"Generated population is missing columns: {sorted(missing)}")
    result = (
        population.groupby([zone_field, *attributes], dropna=False, sort=True)
        .size()
        .rename("count")
        .reset_index()
    )
    result["count"] = result["count"].astype("int64")
    return result


class ExperimentRunner:
    def __init__(self, config: Mapping[str, Any], overwrite: bool = False):
        self.config = dict(config)
        self.tasks = expand_runs(config)
        self.attributes = list(config["attributes"])
        self.zone_field = str(config["zone_field"])
        self.output_root = Path(config["output"]["root"])
        self.resume = config["output"].get("resume", "error")
        self.continue_on_error = bool(config["output"].get("continue_on_error", False))
        self.raw_rows = bool(config["output"].get("raw_rows", False))
        self.overwrite = bool(overwrite)
        self.input_hashes: dict[str, str] = {}
        self._models: dict[str, tuple[Any, Path, str]] = {}
        self._census_raw: pd.DataFrame | None = None

    def validate(self) -> None:
        """Perform all file, column, census, and reusable-model checks up front."""

        inputs = self.config["inputs"]
        for name, spec in inputs.items():
            path = Path(spec["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Input {name!r} does not exist: {path}")
            self.input_hashes[name] = file_sha256(path)

        census_path = Path(inputs["census"]["path"])
        census_marginals = read_household_census(
            census_path, self.attributes, self.zone_field
        )
        self._census_raw = pd.read_csv(census_path, header=[0, 1])
        _zonal_hhsize_targets(self._census_raw, self.zone_field)

        used_inputs = {
            run.get("input")
            for run in self.config["runs"]
            if run.get("enabled", True)
            and run["method"] in {"saa", "ipf"}
            and run.get("input") is not None
        }
        for name in used_inputs:
            spec = inputs[name]
            required = [*self.attributes, spec["weight_column"]]
            columns = pd.read_csv(spec["path"], nrows=0).columns
            missing = set(required) - set(columns)
            if missing:
                raise ValueError(f"Input {name!r} is missing columns: {sorted(missing)}")

        # IPF state support and weight validity are checked before any SAA run
        # can consume significant compute.
        for run in self.config["runs"]:
            if not run.get("enabled", True) or run["method"] != "ipf":
                continue
            spec = inputs[run["input"]]
            cells, _ = prepare_weighted_cells(
                Path(spec["path"]), self.attributes, spec["weight_column"]
            )
            folds = {
                (item["attribute"], str(item["source"])): str(item["destination"])
                for item in run.get("settings", {}).get("state_folds", [])
            }
            apply_state_folds(census_marginals, cells, self.attributes, folds)

        for run in self.config["runs"]:
            if not run.get("enabled", True) or run["method"] not in {"bn", "wgan"}:
                continue
            if run["lifecycle"] != "reuse":
                continue
            model_path = Path(run["model"]["model_path"])
            if run["method"] == "bn":
                model = load_bn_artifact(model_path)
                nodes = set(model.nodes())
                missing = set(self.attributes) - nodes
                model_states = {
                    attribute: set(
                        map(str, model.get_cpds(attribute).state_names[attribute])
                    )
                    for attribute in self.attributes
                    if attribute in nodes
                }
            else:
                model, _ = load_wgan_artifact(model_path, device=run["model"].get("device"))
                missing = set(self.attributes) - set(model.state_names)
                model_states = {
                    attribute: set(map(str, model.state_names[attribute]))
                    for attribute in self.attributes
                    if attribute in model.state_names
                }
            if missing:
                raise ValueError(f"Model for {run['id']!r} lacks attributes: {sorted(missing)}")
            unsupported = {
                attribute: sorted(
                    {
                        str(state)
                        for name, state in census_marginals.columns
                        if name == attribute
                        and float(census_marginals[(name, state)].sum()) > 0
                    }
                    - model_states[attribute]
                )
                for attribute in self.attributes
            }
            unsupported = {key: value for key, value in unsupported.items() if value}
            if unsupported:
                raise ValueError(
                    f"Model for {run['id']!r} does not support positive census "
                    f"states: {unsupported}"
                )

    def _task_fingerprint(self, task: ExperimentTask) -> str:
        return fingerprint(
            {
                "schema_version": self.config["schema_version"],
                "experiment": self.config["name"],
                "base_seed": self.config["base_seed"],
                "attributes": self.attributes,
                "zone_field": self.zone_field,
                "run": task.run,
                "repeat_index": task.repeat_index,
                "seed": task.seed,
                "input_hashes": self.input_hashes,
            }
        )

    def _existing_action(self, task: ExperimentTask) -> str:
        if task.output_dir.exists():
            return "overwrite" if self.overwrite else "skip"
        return "run"

    def _base_manifest(self, task: ExperimentTask) -> dict[str, Any]:
        return {
            "manifest_version": 1,
            "fingerprint": self._task_fingerprint(task),
            "experiment": self.config["name"],
            "run_id": task.run_id,
            "method": task.method,
            "repeat": task.repeat_number,
            "repeat_index": task.repeat_index,
            "sampling_seed": task.seed,
            "input_hashes": self.input_hashes,
            "resolved_run": task.run,
            "versions": _versions(),
            "runtime_environment": _runtime_environment(),
        }

    def _training_input(self, run: Mapping[str, Any]) -> tuple[pd.DataFrame, str, Path]:
        name = run.get("training_input", "original_seed")
        if name not in self.config["inputs"]:
            raise ValueError(f"Unknown training input {name!r} for run {run['id']!r}")
        spec = self.config["inputs"][name]
        path = Path(spec["path"])
        return pd.read_csv(path), spec["weight_column"], path

    def _train_bn(self, run: Mapping[str, Any], seed: int, path: Path) -> Any:
        training = run.get("training", {})
        frame, weight_column, seed_path = self._training_input(run)
        random.seed(seed)
        np.random.seed(seed)
        include_negative = bool(training.get("include_negative_income", True))
        extra = {"hhinc": ["Negative income"]} if include_negative else None
        mode = training.get("mode", "seed")
        unseen = None
        if mode == "seed" and include_negative:
            probability = training.get("unseen_negative_income_probability")
            if probability is None:
                raise ValueError(
                    "BN seed mode with Negative income requires "
                    "unseen_negative_income_probability"
                )
            unseen = {"hhinc": {"Negative income": float(probability)}}
        result = learn_bn(
            seed=frame,
            attributes=self.attributes,
            mode="census" if mode == "census_prior" else "seed",
            census=self._census_raw if mode == "census_prior" else None,
            extra_states=extra,
            unseen_state_probabilities=unseen,
            weight_column=weight_column,
            prior_strength=training.get("prior_strength"),
            ignored_census_states=(
                {"hhinc": ["Negative income"]}
                if mode == "census_prior" and not include_negative
                else None
            ),
            scoring_method=training.get("scoring_method", "bic-d"),
            n_jobs=int(training.get("n_jobs", 1)),
        )
        metadata_value = {
            "format_version": 1,
            "attributes": self.attributes,
            "training_seed": seed,
            "training_input": str(seed_path),
            "training_input_sha256": file_sha256(seed_path),
            "mode": mode,
            "include_negative_income": include_negative,
            "diagnostics": result.diagnostics,
        }
        save_bn_artifact(result.model, metadata_value, path)
        return result.model

    def _train_wgan(self, run: Mapping[str, Any], seed: int, path: Path) -> Any:
        training = run.get("training", {})
        frame, weight_column, seed_path = self._training_input(run)
        prepared = prepare_household_training_data(
            seed=frame,
            census=self._census_raw,
            attributes=self.attributes,
            weight_column=weight_column,
            negative_income_probability=training.get("negative_income_probability"),
        )
        config_fields = set(WGANConfig.__dataclass_fields__) - {"seed"}
        options = {key: training[key] for key in config_fields if key in training}
        model_config = WGANConfig(seed=seed, **options)
        result = train_household_wgan(
            prepared,
            config=model_config,
            device=training.get("device"),
            weight_column=weight_column,
        )
        metadata_value = {
            "attributes": list(HOUSEHOLD_ATTRIBUTES),
            "training_seed": seed,
            "training_input": str(seed_path),
            "training_input_sha256": file_sha256(seed_path),
            "diagnostics": result.diagnostics,
        }
        save_wgan_artifact(result.model, metadata_value, path)
        return result.model

    def _load_model(self, run: Mapping[str, Any], path: Path) -> Any:
        if run["method"] == "bn":
            return load_bn_artifact(path)
        model, _ = load_wgan_artifact(path, device=run.get("training", {}).get("device"))
        return model

    def _shared_model(self, run: Mapping[str, Any]) -> tuple[Any, Path, str]:
        run_id = run["id"]
        if run_id in self._models:
            return self._models[run_id]
        lifecycle = run["lifecycle"]
        if lifecycle == "reuse":
            path = Path(run["model"]["model_path"])
            model = self._load_model(run, path)
            result = (model, path, model_artifact_sha256(path))
            self._models[run_id] = result
            return result
        if lifecycle != "train_once_sample_many":
            raise ValueError("Shared model requested for train_per_repeat")

        path = Path(run["model"]["output_path"])
        manifest_path = self.output_root / run_id / "shared_model_manifest.json"
        training_seed = derive_seed(self.config["base_seed"], run_id, "train", 0)
        expected_fingerprint = fingerprint(
            {"run": run, "training_seed": training_seed, "input_hashes": self.input_hashes}
        )
        if manifest_path.exists():
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                existing.get("fingerprint") != expected_fingerprint
                or not path.is_file()
                or existing.get("model_sha256") != model_artifact_sha256(path)
            ):
                raise ValueError(f"Incompatible shared model output for run {run_id!r}")
            model = self._load_model(run, path)
        else:
            if path.exists():
                raise ValueError(f"Model output exists without a compatible manifest: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            training_started_at = _utc_now()
            training_started = time.perf_counter()
            model = (
                self._train_bn(run, training_seed, path)
                if run["method"] == "bn"
                else self._train_wgan(run, training_seed, path)
            )
            training_seconds = time.perf_counter() - training_started
            _write_json(
                manifest_path,
                {
                    "fingerprint": expected_fingerprint,
                    "run_id": run_id,
                    "lifecycle": lifecycle,
                    "training_seed": training_seed,
                    "model_path": path,
                    "model_sha256": model_artifact_sha256(path),
                    "training_and_save_seconds": training_seconds,
                    "training_started_at": training_started_at,
                    "training_completed_at": _utc_now(),
                },
            )
        result = (model, path, model_artifact_sha256(path))
        self._models[run_id] = result
        return result

    def _model_for_task(self, task: ExperimentTask) -> tuple[Any, Path, str, int | None]:
        run = task.run
        if run["lifecycle"] in {"reuse", "train_once_sample_many"}:
            model, path, digest = self._shared_model(run)
            training_seed = (
                derive_seed(self.config["base_seed"], run["id"], "train", 0)
                if run["lifecycle"] == "train_once_sample_many"
                else None
            )
            return model, path, digest, training_seed
        training_seed = derive_seed(
            self.config["base_seed"], run["id"], "train", task.repeat_index
        )
        configured = Path(run["model"]["output_path"])
        suffix = ".xml" if run["method"] == "bn" else ".pt"
        path = task.output_dir / "model" / f"{configured.stem}{suffix}"
        model = (
            self._train_bn(run, training_seed, path)
            if run["method"] == "bn"
            else self._train_wgan(run, training_seed, path)
        )
        return model, path, model_artifact_sha256(path), training_seed

    def _execute_saa(self, task: ExperimentTask) -> dict[str, Any]:
        run, settings = task.run, task.run.get("settings", {})
        direct_reference_mode = settings.get("direct_reference_mode", "pool")
        input_spec = (
            None
            if direct_reference_mode == "all_zero"
            else self.config["inputs"][run["input"]]
        )
        adjustment_order = list(
            settings.get("ordered_to_adjust_atts", self.attributes)
        )
        started = time.perf_counter()
        run_saa(
            marg_file=Path(self.config["inputs"]["census"]["path"]),
            pool_file=(None if input_spec is None else Path(input_spec["path"])),
            zone_field=self.zone_field,
            output_file=task.output_dir / "household_counts.csv",
            considered_atts=adjustment_order,
            ordered_to_adjust_atts=adjustment_order,
            last_adjustment_order=settings.get(
                "last_adjustment_order", adjustment_order
            ),
            max_run_time=int(settings.get("max_run_time", 1)),
            extra_rm_frac=float(settings.get("extra_rm_frac", 0)),
            randomly_add_last=settings.get("randomly_add_last", []),
            output_each_step=bool(settings.get("output_each_step", False)),
            add_name_for_step_output=run["id"],
            include_zero_cell_values=bool(settings.get("include_zero_cell_values", False)),
            meta_output_dir=task.output_dir / "diagnostics",
            stochastic_mode=settings.get("stochastic_mode", "count_native_ilp"),
            pool_count_column=(None if input_spec is None else input_spec["weight_column"]),
            output_format="counts",
            direct_solver_mode=settings.get("direct_solver_mode", "two_stage"),
            direct_adjustment_backend=settings.get("direct_adjustment_backend", "min_cost_flow"),
            zero_cell_policy=settings["zero_cell_policy"],
            direct_reference_mode=direct_reference_mode,
            zone_workers=int(settings.get("zone_workers", 1)),
        )
        return {
            "direct_reference_mode": direct_reference_mode,
            "pool_count_column": (
                None if input_spec is None else input_spec["weight_column"]
            ),
            "workers": int(settings.get("zone_workers", 1)),
            "timings": {"zonal_synthesis_and_write_seconds": time.perf_counter() - started},
            "zone_diagnostics": "diagnostics/zone_runtimes.csv",
        }

    def _execute_ipf(self, task: ExperimentTask) -> dict[str, Any]:
        run, settings = task.run, task.run.get("settings", {})
        input_spec = self.config["inputs"][run["input"]]
        folds = {
            (item["attribute"], str(item["source"])): str(item["destination"])
            for item in settings.get("state_folds", [])
        }
        fit_started = time.perf_counter()
        workers = int(settings.get("zone_workers", 1))
        result = fit_weighted_ipf(
            census=Path(self.config["inputs"]["census"]["path"]),
            pool=Path(input_spec["path"]),
            attributes=self.attributes,
            zone_field=self.zone_field,
            weight_column=input_spec["weight_column"],
            seed=task.seed,
            state_folds=folds,
            convergence_rate=float(settings.get("convergence_rate", 1e-5)),
            max_iterations=int(settings.get("max_iterations", 500)),
            workers=workers,
        )
        fit_seconds = time.perf_counter() - fit_started
        write_started = time.perf_counter()
        _write_csv(task.output_dir / "household_counts.csv", result.counts)
        _write_csv(task.output_dir / "diagnostics.csv", result.diagnostics)
        if settings.get("save_fractional_weights", True):
            _write_csv(task.output_dir / "household_fractional_weights.csv", result.fractional)
        _write_json(task.output_dir / "state_folds.json", result.fold_log)
        write_seconds = time.perf_counter() - write_started
        return {
            "workers": workers,
            "input_rows": result.input_rows,
            "joint_cells": result.joint_cells,
            "folds": result.fold_log,
            "fractional_max_abs_error": float(
                result.diagnostics["fractional_max_abs_error"].max()
            ),
            "integer_max_abs_error": float(
                result.diagnostics["integer_max_abs_error"].max()
            ),
            "timings": {
                "zonal_fit_and_integerisation_seconds": fit_seconds,
                "output_write_seconds": write_seconds,
            },
            "zone_diagnostics": "diagnostics.csv",
        }

    def _execute_model(self, task: ExperimentTask) -> dict[str, Any]:
        model_started = time.perf_counter()
        model, model_path, model_hash, training_seed = self._model_for_task(task)
        model_prepare_seconds = time.perf_counter() - model_started
        targets = _zonal_hhsize_targets(self._census_raw, self.zone_field)
        settings = task.run.get("sampling", {})
        del model  # Worker processes load the recorded artifact independently.
        workers = int(settings.get("zone_workers", 1))
        sampling_started = time.perf_counter()
        population, zone_diagnostics = sample_zones_parallel(
            method=task.method,
            model_path=model_path,
            targets=targets,
            zone_field=self.zone_field,
            sampling_seed=task.seed,
            workers=workers,
            bn_method=settings.get("method", "auto"),
            chunk_size=int(settings.get("chunk_size", 100_000)),
            device=settings.get("device", "cpu"),
        )
        sampling_seconds = time.perf_counter() - sampling_started
        validation_started = time.perf_counter()
        counts = _condense_population(population, self.attributes, self.zone_field)
        actual = (
            counts.groupby([self.zone_field, "hhsize"])["count"]
            .sum().unstack(fill_value=0)
            .reindex(index=targets.index, columns=targets.columns, fill_value=0)
            .astype("int64")
        )
        actual.index.name, actual.columns.name = targets.index.name, targets.columns.name
        if not actual.equals(targets):
            raise AssertionError("Generated zonal hhsize counts do not match the census")
        validation_seconds = time.perf_counter() - validation_started
        write_started = time.perf_counter()
        _write_csv(task.output_dir / "household_counts.csv", counts)
        _write_csv(task.output_dir / "zone_diagnostics.csv", zone_diagnostics)
        worker_diagnostics = zone_diagnostics[
            ["worker_pid", "worker_model_load_seconds"]
        ].drop_duplicates().sort_values("worker_pid")
        _write_csv(task.output_dir / "worker_diagnostics.csv", worker_diagnostics)
        if self.raw_rows or task.run.get("output", {}).get("raw_rows", False):
            _write_csv(task.output_dir / "households.csv", population)
        write_seconds = time.perf_counter() - write_started
        return {
            "lifecycle": task.run["lifecycle"],
            "training_seed": training_seed,
            "model_path": str(model_path),
            "model_sha256": model_hash,
            "households": int(counts["count"].sum()),
            "workers": workers,
            "sampling_device": settings.get("device", "cpu"),
            "zone_diagnostics": "zone_diagnostics.csv",
            "worker_diagnostics": "worker_diagnostics.csv",
            "timings": {
                "model_prepare_seconds": model_prepare_seconds,
                "parallel_zonal_sampling_seconds": sampling_seconds,
                "condense_and_validate_seconds": validation_seconds,
                "output_write_seconds": write_seconds,
            },
        }

    def _execute(self, task: ExperimentTask) -> dict[str, Any]:
        if task.method == "saa":
            return self._execute_saa(task)
        if task.method == "ipf":
            return self._execute_ipf(task)
        return self._execute_model(task)

    def _write_summary(self) -> None:
        rows: list[dict[str, Any]] = []
        for task in self.tasks:
            path = task.output_dir / "manifest.json"
            if not path.exists():
                rows.append(
                    {"run_id": task.run_id, "method": task.method,
                     "repeat": task.repeat_number, "status": "not_started",
                     "output_dir": str(task.output_dir)}
                )
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            diagnostics = manifest.get("diagnostics", {})
            timings = diagnostics.get("timings", {})
            rows.append(
                {
                    "run_id": task.run_id,
                    "method": task.method,
                    "repeat": task.repeat_number,
                    "status": manifest.get("status"),
                    "sampling_seed": task.seed,
                    "training_seed": diagnostics.get("training_seed"),
                    "runtime_seconds": manifest.get("runtime_seconds"),
                    "workers": diagnostics.get("workers"),
                    "model_sha256": diagnostics.get("model_sha256"),
                    **{f"timing_{key}": value for key, value in timings.items()},
                    "output_dir": str(task.output_dir),
                    "error": manifest.get("error"),
                }
            )
        frame = pd.DataFrame(rows)
        _write_csv(self.output_root / "summary.csv", frame)
        _write_json(self.output_root / "summary.json", rows)

    def _record_invocation(self, record: dict[str, Any]) -> None:
        path = self.output_root / "invocations.json"
        existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
        existing.append(record)
        _write_json(path, existing)

    def _write_orphaned_results(self) -> None:
        configured = {task.run_id for task in self.tasks}
        orphaned = []
        for run_dir in sorted(self.output_root.iterdir()):
            if not run_dir.is_dir() or run_dir.name in configured:
                continue
            repeats = sorted(
                str(path.relative_to(self.output_root))
                for path in run_dir.glob("repeat_*")
                if path.is_dir()
            )
            if repeats:
                orphaned.append({"run_id": run_dir.name, "repeat_directories": repeats})
        _write_json(self.output_root / "orphaned_results.json", orphaned)

    def _reset_output_root(self) -> int:
        """Remove the complete experiment output after explicit --overwrite."""

        root = self.output_root.resolve()
        config_path = Path(self.config["config_path"]).resolve()
        forbidden = {
            Path("/").resolve(),
            Path.home().resolve(),
            Path.cwd().resolve(),
            config_path.parent,
            config_path.parent.parent,
        }
        if root in forbidden or root.parent == root:
            raise ValueError(f"Refusing to overwrite unsafe output root: {root}")
        existing_results = sum(
            task.output_dir.exists() for task in self.tasks
        )
        if root.exists():
            shutil.rmtree(root)
        return existing_results

    def run(self) -> None:
        invocation_started_at = _utc_now()
        invocation_started = time.perf_counter()
        validation_started = time.perf_counter()
        self.validate()
        validation_seconds = time.perf_counter() - validation_started
        overwritten_before_run = (
            self._reset_output_root() if self.overwrite else 0
        )
        self.output_root.mkdir(parents=True, exist_ok=True)
        suite_manifest_path = self.output_root / "suite_manifest.json"
        suite_fingerprint = fingerprint(
            {
                "schema_version": self.config["schema_version"],
                "name": self.config["name"],
                "base_seed": self.config["base_seed"],
                "attributes": self.attributes,
                "zone_field": self.zone_field,
                "runs": self.config["runs"],
                "input_hashes": self.input_hashes,
            }
        )
        if suite_manifest_path.exists():
            existing = json.loads(suite_manifest_path.read_text(encoding="utf-8"))
            if existing.get("fingerprint") != suite_fingerprint:
                completed_results = []
                for result_manifest in self.output_root.glob(
                    "*/repeat_*/manifest.json"
                ):
                    result = json.loads(
                        result_manifest.read_text(encoding="utf-8")
                    )
                    if result.get("status") == "complete":
                        completed_results.append(str(result_manifest))
                _write_json(
                    suite_manifest_path,
                    {
                        "manifest_version": 1,
                        "fingerprint": suite_fingerprint,
                        "supersedes_fingerprint": existing.get("fingerprint"),
                        "experiment": self.config["name"],
                        "input_hashes": self.input_hashes,
                        "runtime_environment": _runtime_environment(),
                        "updated_at": _utc_now(),
                        "recovery_reason": (
                            "Configuration changed; existing repeat directories "
                            "are retained and governed by skip/overwrite policy"
                        ),
                        "completed_results_retained": len(completed_results),
                    },
                )
        else:
            _write_json(
                suite_manifest_path,
                {
                    "manifest_version": 1,
                    "fingerprint": suite_fingerprint,
                    "experiment": self.config["name"],
                    "input_hashes": self.input_hashes,
                    "runtime_environment": _runtime_environment(),
                    "created_at": _utc_now(),
                },
            )
        _write_json(self.output_root / "resolved_config.json", self.config)
        self._write_orphaned_results()
        print(f"Validated {len(self.tasks)} result sets for {self.config['name']!r}")
        executed = skipped = failed = 0
        overwritten = overwritten_before_run
        invocation_status = "complete"
        invocation_error: str | None = None
        try:
            for task in self.tasks:
                action = self._existing_action(task)
                if action == "skip":
                    print(
                        f"SKIP {task.run_id} repeat {task.repeat_number} "
                        f"seed={task.seed} output={task.output_dir}"
                    )
                    skipped += 1
                    continue
                if action == "overwrite":
                    expected_parent = self.output_root / task.run_id
                    if task.output_dir.parent != expected_parent:
                        raise ValueError(
                            f"Refusing to overwrite unexpected path: {task.output_dir}"
                        )
                    shutil.rmtree(task.output_dir)
                    overwritten += 1
                task.output_dir.mkdir(parents=True, exist_ok=True)
                print(
                    f"RUN  {task.run_id} repeat {task.repeat_number} "
                    f"seed={task.seed} output={task.output_dir}"
                )
                manifest = self._base_manifest(task)
                manifest.update({"status": "running", "started_at": _utc_now()})
                _write_json(task.output_dir / "manifest.json", manifest)
                started = time.perf_counter()
                try:
                    diagnostics = self._execute(task)
                    executed += 1
                    hash_started = time.perf_counter()
                    artifact_hashes = _result_artifact_hashes(task.output_dir)
                    artifact_hash_seconds = time.perf_counter() - hash_started
                    diagnostics.setdefault("timings", {})[
                        "artifact_hash_seconds"
                    ] = artifact_hash_seconds
                    manifest.update(
                        {
                            "status": "complete",
                            "completed_at": _utc_now(),
                            "runtime_seconds": time.perf_counter() - started,
                            "diagnostics": diagnostics,
                            "artifact_hashes": artifact_hashes,
                        }
                    )
                except Exception as error:
                    failed += 1
                    manifest.update(
                        {
                            "status": "failed",
                            "completed_at": _utc_now(),
                            "runtime_seconds": time.perf_counter() - started,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                    _write_json(task.output_dir / "manifest.json", manifest)
                    self._write_summary()
                    if not self.continue_on_error:
                        raise
                _write_json(task.output_dir / "manifest.json", manifest)
                self._write_summary()
        except Exception as error:
            invocation_status = "failed"
            invocation_error = f"{type(error).__name__}: {error}"
            raise
        finally:
            self._write_summary()
            if invocation_status == "complete" and failed:
                invocation_status = "complete_with_failures"
            self._record_invocation(
                {
                    "started_at": invocation_started_at,
                    "completed_at": _utc_now(),
                    "status": invocation_status,
                    "error": invocation_error,
                    "validation_seconds": validation_seconds,
                    "total_runtime_seconds": time.perf_counter() - invocation_started,
                    "configured_results": len(self.tasks),
                    "executed_results": executed,
                    "skipped_results": skipped,
                    "failed_results": failed,
                    "overwritten_results": overwritten,
                    "overwrite_enabled": self.overwrite,
                    "versions": _versions(),
                    "runtime_environment": _runtime_environment(),
                }
            )

