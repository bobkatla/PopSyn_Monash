"""Loading, validation, and deterministic expansion of experiment YAML."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Mapping

import yaml


METHODS = {"saa", "ipf", "bn", "wgan"}
LIFECYCLES = {"reuse", "train_once_sample_many", "train_per_repeat"}
RESUME_MODES = {"error", "skip_completed"}
ZERO_CELL_POLICIES = {
    "unrestricted",
    "seed_only",
    "relax_empty_rows_columns",
    "relax_empty_rows_global_columns",
}
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class ExperimentTask:
    run_id: str
    method: str
    repeat_index: int
    repeat_number: int
    seed: int
    output_dir: Path
    run: dict[str, Any]


def _require(mapping: Mapping[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_number(value: Any, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or number < 0 or (number == 0 and not allow_zero):
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ValueError(f"{name} must be a finite {qualifier} number")
    return number


def derive_seed(base_seed: int, run_id: str, phase: str, repeat_index: int = 0) -> int:
    """Derive a stable 32-bit seed without depending on YAML run order."""

    material = f"{base_seed}\0{run_id}\0{phase}\0{repeat_index}".encode()
    return int.from_bytes(hashlib.sha256(material).digest()[:4], "big")


def _resolve_path(config_dir: Path, value: Any, name: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise ValueError(f"{name} must be a path string")
    path = Path(value).expanduser()
    return (config_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _validate_run(
    run: dict[str, Any],
    inputs: Mapping[str, Any],
    attributes: list[str],
    index: int,
) -> None:
    context = f"runs[{index}]"
    run_id = _require(run, "id", context)
    if not isinstance(run_id, str) or not _ID.fullmatch(run_id):
        raise ValueError(f"{context}.id must be a filesystem-safe identifier")
    method = _require(run, "method", context)
    if method not in METHODS:
        raise ValueError(f"{context}.method must be one of {sorted(METHODS)}")
    _positive_int(run.get("repeats", 1), f"{context}.repeats")
    settings = run.get("settings", {})
    if not isinstance(settings, dict):
        raise ValueError(f"{context}.settings must be a mapping")
    input_name = run.get("input")
    direct_reference_mode = settings.get("direct_reference_mode", "pool")
    if method == "saa" and direct_reference_mode == "all_zero":
        if input_name is not None:
            raise ValueError(
                f"{context}.input must be null when direct_reference_mode='all_zero'"
            )
    elif method in {"saa", "ipf"}:
        if input_name not in inputs:
            raise ValueError(f"{context}.input refers to unknown input {input_name!r}")
        weight = inputs[input_name].get("weight_column")
        if not weight:
            raise ValueError(f"inputs.{input_name}.weight_column is required for {method}")
    if method == "saa":
        _positive_int(settings.get("zone_workers", 1), f"{context}.settings.zone_workers")
        adjustment_order = settings.get("ordered_to_adjust_atts", attributes)
        if (
            not isinstance(adjustment_order, list)
            or len(adjustment_order) != len(attributes)
            or set(adjustment_order) != set(attributes)
        ):
            raise ValueError(
                f"{context}.settings.ordered_to_adjust_atts must contain every "
                "configured attribute exactly once"
            )
        policy = _require(settings, "zero_cell_policy", f"{context}.settings")
        if policy not in ZERO_CELL_POLICIES:
            raise ValueError(
                f"{context}.settings.zero_cell_policy must be one of "
                f"{sorted(ZERO_CELL_POLICIES)}"
            )
        if direct_reference_mode not in {"pool", "all_zero"}:
            raise ValueError(
                f"{context}.settings.direct_reference_mode must be 'pool' or 'all_zero'"
            )
        if direct_reference_mode == "all_zero" and policy != "unrestricted":
            raise ValueError(
                f"{context}.settings.zero_cell_policy must be 'unrestricted' "
                "when direct_reference_mode='all_zero'"
            )
    if method == "ipf":
        _positive_int(settings.get("max_iterations", 500), f"{context}.settings.max_iterations")
        _positive_int(settings.get("zone_workers", 1), f"{context}.settings.zone_workers")
        _positive_number(
            settings.get("convergence_rate", 1e-5),
            f"{context}.settings.convergence_rate",
        )
        for fold_index, fold in enumerate(settings.get("state_folds", [])):
            if not isinstance(fold, dict) or not {"attribute", "source", "destination"} <= set(fold):
                raise ValueError(
                    f"{context}.settings.state_folds[{fold_index}] requires "
                    "attribute, source, and destination"
                )
    if method in {"bn", "wgan"}:
        lifecycle = _require(run, "lifecycle", context)
        if lifecycle not in LIFECYCLES:
            raise ValueError(f"{context}.lifecycle must be one of {sorted(LIFECYCLES)}")
        model = _require(run, "model", context)
        if not isinstance(model, dict):
            raise ValueError(f"{context}.model must be a mapping")
        required_path = "model_path" if lifecycle == "reuse" else "output_path"
        model_path = _require(model, required_path, f"{context}.model")
        expected_suffix = ".xml" if method == "bn" else ".pt"
        if Path(model_path).suffix.lower() != expected_suffix:
            raise ValueError(
                f"{context}.model.{required_path} must use {expected_suffix}"
            )
        training = run.get("training", {})
        if not isinstance(training, dict):
            raise ValueError(f"{context}.training must be a mapping")
        sampling = run.get("sampling", {})
        if not isinstance(sampling, dict):
            raise ValueError(f"{context}.sampling must be a mapping")
        _positive_int(
            sampling.get("zone_workers", 1),
            f"{context}.sampling.zone_workers",
        )
        if lifecycle != "reuse":
            training_input = run.get("training_input", "original_seed")
            if training_input not in inputs:
                raise ValueError(
                    f"{context}.training_input refers to unknown input "
                    f"{training_input!r}"
                )
            if not inputs[training_input].get("weight_column"):
                raise ValueError(
                    f"inputs.{training_input}.weight_column is required for training"
                )
        if method == "bn" and lifecycle != "reuse":
            mode = training.get("mode", "seed")
            if mode not in {"seed", "census_prior"}:
                raise ValueError(f"{context}.training.mode must be 'seed' or 'census_prior'")
            if training.get("include_negative_income", True) and mode == "seed":
                probability = training.get("unseen_negative_income_probability")
                if probability is None or not 0 < float(probability) < 1:
                    raise ValueError(
                        f"{context}.training.unseen_negative_income_probability "
                        "must be between zero and one"
                    )
            if training.get("prior_strength") is not None:
                _positive_number(
                    training["prior_strength"], f"{context}.training.prior_strength"
                )
            _positive_int(training.get("n_jobs", 1), f"{context}.training.n_jobs")
        if method == "wgan" and lifecycle != "reuse":
            for key in ("epochs", "batch_size", "hidden_dim", "latent_dim", "n_critic"):
                _positive_int(training.get(key), f"{context}.training.{key}")
            for key in ("learning_rate", "temperature_start", "temperature_end", "max_grad_norm"):
                _positive_number(training.get(key), f"{context}.training.{key}")
            _positive_number(
                training.get("lambda_gp"),
                f"{context}.training.lambda_gp",
                allow_zero=True,
            )
            probability = training.get("negative_income_probability")
            if probability is not None and not 0 < float(probability) < 1:
                raise ValueError(
                    f"{context}.training.negative_income_probability must be "
                    "between zero and one or null"
                )


def load_config(config_path: Path | str, check_files: bool = True) -> dict[str, Any]:
    """Load, fully validate, and resolve a version-1 experiment configuration."""

    path = Path(config_path).resolve()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Experiment YAML must contain a top-level mapping")
    if raw.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    _require(raw, "name", "configuration")
    base_seed = _require(raw, "base_seed", "configuration")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool) or base_seed < 0:
        raise ValueError("base_seed must be a nonnegative integer")
    attributes = _require(raw, "attributes", "configuration")
    if not isinstance(attributes, list) or not attributes or len(set(attributes)) != len(attributes):
        raise ValueError("attributes must be a nonempty list without duplicates")
    _require(raw, "zone_field", "configuration")
    inputs = _require(raw, "inputs", "configuration")
    if not isinstance(inputs, dict) or not inputs:
        raise ValueError("inputs must be a nonempty mapping")
    if "census" not in inputs:
        raise ValueError("inputs.census is required")
    config_dir = path.parent
    resolved = deepcopy(raw)
    resolved["config_path"] = path
    for name, spec in resolved["inputs"].items():
        if not isinstance(spec, dict):
            raise ValueError(f"inputs.{name} must be a mapping")
        spec["path"] = _resolve_path(config_dir, _require(spec, "path", f"inputs.{name}"), f"inputs.{name}.path")
    output = _require(resolved, "output", "configuration")
    if not isinstance(output, dict):
        raise ValueError("output must be a mapping")
    output["root"] = _resolve_path(config_dir, _require(output, "root", "output"), "output.root")
    resume = output.get("resume", "error")
    if resume not in RESUME_MODES:
        raise ValueError(f"output.resume must be one of {sorted(RESUME_MODES)}")
    if output.get("format", "counts") != "counts":
        raise ValueError("Only condensed output format 'counts' is supported")

    runs = _require(resolved, "runs", "configuration")
    if not isinstance(runs, list) or not runs:
        raise ValueError("runs must be a nonempty list")
    enabled_ids: list[str] = []
    for index, run in enumerate(runs):
        if not isinstance(run, dict):
            raise ValueError(f"runs[{index}] must be a mapping")
        _validate_run(run, resolved["inputs"], attributes, index)
        if run.get("enabled", True):
            enabled_ids.append(run["id"])
        if run["method"] in {"bn", "wgan"}:
            for key in ("model_path", "output_path"):
                if key in run["model"]:
                    run["model"][key] = _resolve_path(
                        config_dir, run["model"][key], f"runs[{index}].model.{key}"
                    )
    if len(enabled_ids) != len(set(enabled_ids)):
        raise ValueError("Enabled run IDs must be unique")
    if check_files:
        for name, spec in resolved["inputs"].items():
            if not spec["path"].is_file():
                raise FileNotFoundError(f"Input {name!r} does not exist: {spec['path']}")
        for run in runs:
            if (
                run.get("enabled", True)
                and run["method"] in {"bn", "wgan"}
                and run["lifecycle"] == "reuse"
                and not run["model"]["model_path"].is_file()
            ):
                raise FileNotFoundError(
                    f"Model for run {run['id']!r} does not exist: "
                    f"{run['model']['model_path']}"
                )
    return resolved


def expand_runs(config: Mapping[str, Any]) -> list[ExperimentTask]:
    tasks: list[ExperimentTask] = []
    root = Path(config["output"]["root"])
    for run in config["runs"]:
        if not run.get("enabled", True):
            continue
        for index in range(run.get("repeats", 1)):
            tasks.append(
                ExperimentTask(
                    run_id=run["id"],
                    method=run["method"],
                    repeat_index=index,
                    repeat_number=index + 1,
                    seed=derive_seed(config["base_seed"], run["id"], "sample", index),
                    output_dir=root / run["id"] / f"repeat_{index + 1:02d}",
                    run=run,
                )
            )
    return tasks


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def fingerprint(value: Any) -> str:
    encoded = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()

