from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from pgmpy.factors.discrete import TabularCPD

try:
    from pgmpy.models import DiscreteBayesianNetwork as BayesianNetworkModel
except ImportError:  # pragma: no cover
    from pgmpy.models import BayesianNetwork as BayesianNetworkModel

from PopSynthesis.Generator_data.generate_combine_census.utils import TRS
from PopSynthesis.Methods.BN.utils.bn_pool import save_bn_artifact
from PopSynthesis.Methods.GAN.wgan import (
    ConditionalHouseholdWGAN,
    WGANConfig,
    save_wgan_artifact,
)
from PopSynthesis.Methods.IPF.weighted import fit_weighted_ipf, prepare_weighted_cells
from PopSynthesis.Methods.IPSF.run import _prepare_zone_artifact_directory
from PopSynthesis.experiments.config import derive_seed, expand_runs, load_config
from PopSynthesis.experiments.run import build_parser
from PopSynthesis.experiments.runner import ExperimentRunner, sample_zones_parallel


REPO_ROOT = Path(__file__).parents[3]


def test_cli_overwrite_flag_is_explicit() -> None:
    parser = build_parser()
    normal = parser.parse_args(["--config", "experiment.yml"])
    overwrite = parser.parse_args(["--config", "experiment.yml", "--overwrite"])
    assert normal.overwrite is False
    assert overwrite.overwrite is True


def test_production_config_expands_45_results_and_uses_existing_bn_pool() -> None:
    config = load_config(REPO_ROOT / "IO/configs/full_household_experiments.yml")
    tasks = expand_runs(config)

    assert len(tasks) == 45
    assert sum(task.method == "saa" for task in tasks) == 5
    assert sum(task.method == "ipf" for task in tasks) == 20
    assert sum(task.method == "bn" for task in tasks) == 10
    assert sum(task.method == "wgan" for task in tasks) == 10
    assert config["inputs"]["bn_pool"]["path"] == (
        REPO_ROOT / "IO/processed_data/HH_pool_condensed.csv"
    )
    assert config["inputs"]["bn_pool"]["weight_column"] == "count"
    bn_runs = {run["id"]: run for run in config["runs"]}
    assert bn_runs["saa_bn_pool_seed_only"]["input"] == "bn_pool"
    assert bn_runs["ipf_bn_pool"]["input"] == "bn_pool"
    assert bn_runs["ipf_bn_pool"]["settings"]["state_folds"] == []
    assert bn_runs["bn_exact_hhsize"]["lifecycle"] == "train_once_sample_many"
    assert bn_runs["wgan_exact_hhsize"]["lifecycle"] == "train_once_sample_many"
    original_saa = [
        run for run in config["runs"]
        if run["method"] == "saa" and run["input"] == "original_seed"
    ]
    assert {run["settings"]["zero_cell_policy"] for run in original_saa} == {
        "unrestricted",
        "seed_only",
        "relax_empty_rows_columns",
    }
    seed_free = bn_runs["saa_seed_free_all_zero"]
    assert seed_free["input"] is None
    assert seed_free["settings"]["direct_reference_mode"] == "all_zero"
    assert seed_free["settings"]["zero_cell_policy"] == "unrestricted"
    assert "saa_original_relax_empty_rows_global_columns" not in bn_runs
    assert config["inputs"]["original_seed"]["weight_column"] == "_weight"
    for run in [*original_saa, seed_free, bn_runs["saa_bn_pool_seed_only"]]:
        assert run["settings"]["stochastic_mode"] == "count_native_ilp"
        assert run["settings"]["direct_solver_mode"] == "two_stage"
        assert run["settings"]["direct_adjustment_backend"] == "min_cost_flow"
        assert run["settings"]["zone_workers"] == 6
        assert run["settings"]["ordered_to_adjust_atts"] == [
            "hhsize", "dwelltype", "owndwell", "hhinc", "totalvehs"
        ]
        assert run["settings"]["last_adjustment_order"] == [
            "hhsize", "dwelltype", "owndwell", "hhinc", "totalvehs"
        ]
    for run_id in ("ipf_original_weighted_seed", "ipf_bn_pool"):
        assert bn_runs[run_id]["settings"]["zone_workers"] == 6
    for run_id in ("bn_exact_hhsize", "wgan_exact_hhsize"):
        assert bn_runs[run_id]["sampling"]["zone_workers"] == 6
        assert bn_runs[run_id]["sampling"]["device"] == "cpu"


def test_saa_runner_forwards_adjustment_order_as_considered_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = load_config(REPO_ROOT / "IO/configs/full_household_experiments.yml")
    runner = ExperimentRunner(config)
    task = next(task for task in runner.tasks if task.method == "saa")
    captured = {}

    def fake_run_saa(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr("PopSynthesis.experiments.runner.run_saa", fake_run_saa)
    runner._execute_saa(task)
    expected = ["hhsize", "dwelltype", "owndwell", "hhinc", "totalvehs"]
    assert captured["considered_atts"] == expected
    assert captured["ordered_to_adjust_atts"] == expected

    captured.clear()
    seed_free_task = next(
        task for task in runner.tasks if task.run_id == "saa_seed_free_all_zero"
    )
    runner._execute_saa(seed_free_task)
    assert captured["pool_file"] is None
    assert captured["pool_count_column"] is None
    assert captured["direct_reference_mode"] == "all_zero"


def test_empty_saa_checkpoint_manifest_can_adopt_corrected_settings(
    tmp_path: Path,
) -> None:
    marginal = tmp_path / "marginal.csv"
    pool = tmp_path / "pool.csv"
    marginal.write_text("marginal", encoding="utf-8")
    pool.write_text("pool", encoding="utf-8")
    artifacts = tmp_path / "zone_artifacts"
    _prepare_zone_artifact_directory(
        artifacts, marginal, pool, {"considered_atts": ["old"]}
    )
    _prepare_zone_artifact_directory(
        artifacts, marginal, pool, {"considered_atts": ["corrected"]}
    )
    manifest = json.loads((artifacts / "manifest.json").read_text())
    assert manifest["settings"]["considered_atts"] == ["corrected"]

    (artifacts / "zone_0001.csv").write_text("zone", encoding="utf-8")
    with pytest.raises(ValueError, match="different inputs or settings"):
        _prepare_zone_artifact_directory(
            artifacts, marginal, pool, {"considered_atts": ["unsafe"]}
        )


def _minimal_yaml(tmp_path: Path) -> Path:
    data = tmp_path / "data"
    data.mkdir()
    (data / "census.csv").write_text("zone_id,hhsize\nid,1\nz1,1\n", encoding="utf-8")
    (data / "pool.csv").write_text("hhsize,count\n1,1\n", encoding="utf-8")
    value = {
        "schema_version": 1,
        "name": "test",
        "base_seed": 9,
        "inputs": {
            "census": {"path": "data/census.csv"},
            "pool": {"path": "data/pool.csv", "weight_column": "count"},
        },
        "zone_field": "zone_id",
        "attributes": ["hhsize"],
        "output": {"root": "out", "format": "counts", "resume": "error"},
        "runs": [
            {
                "id": "saa",
                "method": "saa",
                "input": "pool",
                "repeats": 1,
                "settings": {"zero_cell_policy": "seed_only"},
            }
        ],
    }
    path = tmp_path / "experiment.yml"
    path.write_text(yaml.safe_dump(value), encoding="utf-8")
    return path


def test_yaml_paths_are_relative_to_yaml_and_method_errors_are_clear(tmp_path: Path) -> None:
    path = _minimal_yaml(tmp_path)
    config = load_config(path)
    assert config["inputs"]["pool"]["path"] == (tmp_path / "data/pool.csv")
    assert config["output"]["root"] == (tmp_path / "out")

    raw = yaml.safe_load(path.read_text())
    raw["runs"][0]["settings"].pop("zero_cell_policy")
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="zero_cell_policy"):
        load_config(path)


def test_seed_assignment_is_stable_distinct_and_order_independent() -> None:
    values = [derive_seed(42, "bn", "sample", index) for index in range(10)]
    assert values == [derive_seed(42, "bn", "sample", index) for index in range(10)]
    assert len(set(values)) == 10
    assert derive_seed(42, "bn", "train", 0) not in values
    assert derive_seed(42, "other", "sample", 0) != values[0]


def _census() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            ("zone_id", "id"): ["z1"],
            ("hhsize", "1"): [6],
            ("hhsize", "2"): [4],
            ("hhinc", "Nil income"): [7],
            ("hhinc", "Negative income"): [3],
        }
    )
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


def test_weighted_ipf_condenses_without_expansion_and_applies_original_fold() -> None:
    pool = pd.DataFrame(
        {
            "hhsize": ["1", "1", "2"],
            "hhinc": ["Nil income", "Nil income", "Nil income"],
            "_weight": [2.5, 3.5, 4.0],
        }
    )
    cells, rows = prepare_weighted_cells(pool, ["hhsize", "hhinc"], "_weight")
    assert rows == 3
    assert len(cells) == 2
    assert cells["_weight"].sum() == 10

    result = fit_weighted_ipf(
        _census(), pool, ["hhsize", "hhinc"], "zone_id", "_weight", seed=4,
        state_folds={("hhinc", "Negative income"): "Nil income"},
    )
    assert result.input_rows == 3
    assert result.joint_cells == 2
    assert result.fold_log[0]["source"] == "Negative income"
    assert set(result.counts["hhinc"]) == {"Nil income"}
    assert result.counts["count"].sum() == 10


def test_bn_pool_ipf_keeps_negative_income_and_rejects_unsupported_state() -> None:
    pool = pd.DataFrame(
        {
            "hhsize": ["1", "1", "2", "2"],
            "hhinc": ["Nil income", "Negative income"] * 2,
            "count": [5, 2, 3, 1],
        }
    )
    result = fit_weighted_ipf(
        _census(), pool, ["hhsize", "hhinc"], "zone_id", "count", seed=5,
        state_folds={},
    )
    assert result.fold_log == []
    assert "Negative income" in set(result.counts["hhinc"])

    unsupported = pool.loc[~pool["hhinc"].eq("Negative income")]
    with pytest.raises(ValueError, match="Negative income"):
        fit_weighted_ipf(
            _census(), unsupported, ["hhsize", "hhinc"], "zone_id", "count", seed=5,
        )


def test_parallel_ipf_matches_single_worker_and_records_zone_timings() -> None:
    census = pd.concat([_census(), _census()], ignore_index=True)
    census[("zone_id", "id")] = ["z1", "z2"]
    pool = pd.DataFrame(
        {
            "hhsize": ["1", "1", "2", "2"],
            "hhinc": ["Nil income", "Negative income"] * 2,
            "count": [5, 2, 3, 1],
        }
    )
    single = fit_weighted_ipf(
        census, pool, ["hhsize", "hhinc"], "zone_id", "count", seed=12,
        workers=1,
    )
    parallel = fit_weighted_ipf(
        census, pool, ["hhsize", "hhinc"], "zone_id", "count", seed=12,
        workers=2,
    )
    columns = ["zone_id", "hhsize", "hhinc", "count"]
    pd.testing.assert_frame_equal(
        single.counts.sort_values(columns).reset_index(drop=True),
        parallel.counts.sort_values(columns).reset_index(drop=True),
    )
    assert parallel.diagnostics["zone_seed"].nunique() == 2
    assert parallel.diagnostics["runtime_seconds"].gt(0).all()
    assert parallel.diagnostics["worker_pid"].notna().all()


def test_bn_and_wgan_parallel_zonal_sampling_records_worker_metadata(
    tmp_path: Path,
) -> None:
    attributes = ["hhsize", "totalvehs", "dwelltype", "owndwell", "hhinc"]
    bn = BayesianNetworkModel()
    bn.add_nodes_from(attributes)
    states = {
        "hhsize": ["1", "2"],
        "totalvehs": ["0"],
        "dwelltype": ["House"],
        "owndwell": ["Own"],
        "hhinc": ["Nil income"],
    }
    for attribute in attributes:
        probabilities = [[0.5], [0.5]] if attribute == "hhsize" else [[1.0]]
        bn.add_cpds(
            TabularCPD(
                attribute,
                len(states[attribute]),
                probabilities,
                state_names={attribute: states[attribute]},
            )
        )
    bn_path = tmp_path / "model.xml"
    save_bn_artifact(bn, {"attributes": attributes}, bn_path)

    wgan = ConditionalHouseholdWGAN(
        state_names=states,
        income_given_hhsize={
            "1": {"Nil income": 1.0},
            "2": {"Nil income": 1.0},
        },
        config=WGANConfig(
            latent_dim=4,
            hidden_dim=4,
            batch_size=2,
            epochs=1,
            n_critic=1,
            seed=1,
        ),
        device="cpu",
    )
    wgan_path = tmp_path / "model.pt"
    save_wgan_artifact(wgan, {"attributes": attributes}, wgan_path)

    targets = pd.DataFrame(
        {"1": [3, 2], "2": [2, 3]},
        index=pd.Index(["z1", "z2"], name="zone_id"),
    )
    for method, path in (("bn", bn_path), ("wgan", wgan_path)):
        population, diagnostics = sample_zones_parallel(
            method=method,
            model_path=path,
            targets=targets,
            zone_field="zone_id",
            sampling_seed=9,
            workers=2,
            device="cpu",
            chunk_size=3,
        )
        actual = (
            population.groupby(["zone_id", "hhsize"]).size().unstack(fill_value=0)
            .reindex(index=targets.index, columns=targets.columns, fill_value=0)
        )
        pd.testing.assert_frame_equal(actual, targets)
        assert diagnostics["zone_seed"].nunique() == 2
        assert diagnostics["sampling_seconds"].gt(0).all()
        assert diagnostics["worker_model_load_seconds"].ge(0).all()


def test_trs_accepts_local_rng_without_changing_legacy_api() -> None:
    weights = np.array([1.2, 2.3, 3.5])
    first = TRS(weights, 8, rng=np.random.default_rng(17))
    second = TRS(weights, 8, rng=np.random.default_rng(17))
    np.testing.assert_array_equal(first, second)
    assert first.sum() == 8
    assert TRS(weights, 8).sum() == 8


def _runner_config(
    tmp_path: Path,
    lifecycle: str = "train_once_sample_many",
    method: str = "bn",
) -> dict:
    model_path = tmp_path / ("model.xml" if method == "bn" else "model.pt")
    return {
        "schema_version": 1,
        "name": "runner-test",
        "base_seed": 3,
        "attributes": ["hhsize"],
        "zone_field": "zone_id",
        "inputs": {},
        "output": {
            "root": tmp_path / "results",
            "format": "counts",
            "resume": "skip_completed",
            "continue_on_error": False,
        },
        "runs": [
            {
                "id": method,
                "method": method,
                "repeats": 3,
                "lifecycle": lifecycle,
                "model": {"model_path": model_path, "output_path": model_path},
                "training": {},
                "sampling": {},
            }
        ],
        "config_path": tmp_path / "config.yml",
    }


@pytest.mark.parametrize(
    ("lifecycle", "expected_train"),
    [("reuse", 0), ("train_once_sample_many", 1), ("train_per_repeat", 3)],
)
@pytest.mark.parametrize("method", ["bn", "wgan"])
def test_model_lifecycle_training_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lifecycle: str,
    expected_train: int,
    method: str,
) -> None:
    config = _runner_config(tmp_path, lifecycle, method)
    runner = ExperimentRunner(config)
    calls = []
    model_path = Path(config["runs"][0]["model"]["model_path"])
    if lifecycle == "reuse":
        model_path.write_bytes(b"existing")
    monkeypatch.setattr(runner, "_load_model", lambda run, path: object())

    def train(run, seed, path):
        calls.append(seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(str(seed).encode())
        return object()

    monkeypatch.setattr(runner, "_train_bn" if method == "bn" else "_train_wgan", train)
    runner.input_hashes = {"seed": "digest"}
    for task in runner.tasks:
        runner._model_for_task(task)
    assert len(calls) == expected_train
    if lifecycle == "train_once_sample_many":
        assert len({task.seed for task in runner.tasks}) == 3


def test_manifests_summary_and_resume_skip(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _runner_config(tmp_path)
    config["runs"] = [
        {
            "id": "fake",
            "method": "saa",
            "input": "pool",
            "repeats": 2,
            "settings": {"zero_cell_policy": "seed_only"},
        }
    ]
    runner = ExperimentRunner(config)
    monkeypatch.setattr(runner, "validate", lambda: runner.input_hashes.update({"pool": "x"}))
    calls = []

    def execute(task):
        calls.append(task.repeat_number)
        pd.DataFrame({"zone_id": ["z"], "hhsize": ["1"], "count": [1]}).to_csv(
            task.output_dir / "household_counts.csv", index=False
        )
        return {"ok": True}

    monkeypatch.setattr(runner, "_execute", execute)
    runner.run()
    assert calls == [1, 2]
    assert (Path(config["output"]["root"]) / "summary.csv").is_file()
    assert (Path(config["output"]["root"]) / "invocations.json").is_file()
    manifest = json.loads((runner.tasks[0].output_dir / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert "household_counts.csv" in manifest["artifact_hashes"]
    assert manifest["diagnostics"]["timings"]["artifact_hash_seconds"] >= 0
    invocations = json.loads(
        (Path(config["output"]["root"]) / "invocations.json").read_text()
    )
    assert invocations[-1]["validation_seconds"] >= 0
    assert invocations[-1]["total_runtime_seconds"] >= 0
    assert "runtime_environment" in invocations[-1]

    resumed = ExperimentRunner(config)
    monkeypatch.setattr(resumed, "validate", lambda: resumed.input_hashes.update({"pool": "x"}))
    monkeypatch.setattr(
        resumed, "_execute", lambda task: pytest.fail("completed task was rerun")
    )
    resumed.run()

    changed = deepcopy(config)
    changed["runs"][0]["settings"]["zero_cell_policy"] = "unrestricted"
    changed_runner = ExperimentRunner(changed)
    monkeypatch.setattr(
        changed_runner,
        "validate",
        lambda: changed_runner.input_hashes.update({"pool": "x"}),
    )
    monkeypatch.setattr(
        changed_runner,
        "_execute",
        lambda task: pytest.fail("existing directory was not skipped"),
    )
    assert changed_runner._existing_action(changed_runner.tasks[0]) == "skip"
    changed_runner.run()
    migrated_suite = json.loads(
        (Path(config["output"]["root"]) / "suite_manifest.json").read_text()
    )
    assert migrated_suite["completed_results_retained"] == 2

    overwrite_calls = []
    overwriting = ExperimentRunner(config, overwrite=True)
    monkeypatch.setattr(
        overwriting, "validate", lambda: overwriting.input_hashes.update({"pool": "x"})
    )

    def overwrite_execute(task):
        overwrite_calls.append(task.repeat_number)
        pd.DataFrame(
            {"zone_id": ["z"], "hhsize": ["1"], "count": [2]}
        ).to_csv(task.output_dir / "household_counts.csv", index=False)
        return {"ok": True}

    monkeypatch.setattr(overwriting, "_execute", overwrite_execute)
    overwriting.run()
    assert overwrite_calls == [1, 2]
    invocations = json.loads(
        (Path(config["output"]["root"]) / "invocations.json").read_text()
    )
    assert invocations[-1]["overwrite_enabled"] is True
    assert invocations[-1]["overwritten_results"] == 2

