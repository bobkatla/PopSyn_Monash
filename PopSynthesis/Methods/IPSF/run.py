"""Main place to run the SAA method"""

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from decimal import Decimal
import hashlib
import json
from math import isfinite
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import List, Literal, Optional, Tuple

import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.const import (
    SAA_ODERED_ATTS_HH,
    CONSIDERED_ATTS_HH,
    count_field,
    zone_field as internal_zone_field,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df


_DIRECT_WORKER_POOL: pl.DataFrame | None = None
_DIRECT_WORKER_KWARGS: dict[str, object] | None = None
_DIRECT_WORKER_CACHE: dict = {}


def _initialize_direct_worker(
    count_pool: pl.DataFrame, run_kwargs: dict[str, object]
) -> None:
    global _DIRECT_WORKER_POOL, _DIRECT_WORKER_KWARGS, _DIRECT_WORKER_CACHE
    _DIRECT_WORKER_POOL = count_pool
    _DIRECT_WORKER_KWARGS = run_kwargs
    _DIRECT_WORKER_CACHE = {}


def _run_direct_zone_worker(
    zone_number: int,
    zone: object,
    zone_marginal: pd.DataFrame,
    artifact_path: Path,
) -> tuple[int, str]:
    if _DIRECT_WORKER_POOL is None or _DIRECT_WORKER_KWARGS is None:
        raise RuntimeError("Direct-count worker was not initialized")
    kwargs = dict(_DIRECT_WORKER_KWARGS)
    kwargs["direct_pool_cache"] = _DIRECT_WORKER_CACHE
    result, _ = saa_run(zone_marginal, _DIRECT_WORKER_POOL, **kwargs)
    _write_csv_atomically(result, artifact_path)
    return zone_number, str(zone)


def process_data_from_files(marg_file: Path, pool_file: Path, zone_field: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    marg = pd.read_csv(marg_file, header=[0, 1])
    sample_geog_columns = marg.columns[
        marg.columns.get_level_values(0) == "sample_geog"
    ]
    if len(sample_geog_columns) > 0:
        marg = marg.drop(columns=sample_geog_columns)

    zone_columns = marg.columns[marg.columns.get_level_values(0) == zone_field]
    if len(zone_columns) != 1:
        raise ValueError(
            f"Expected exactly one zone column named {zone_field!r}; "
            f"found {len(zone_columns)}."
        )
    marg = marg.set_index(zone_columns[0])
    marg.index.name = internal_zone_field
    pool = pd.read_csv(pool_file)
    pool = pool.drop(columns=["serialno", "sample_geog"], errors="ignore")
    return marg, pool


def _prepare_direct_pool(
    pool: pd.DataFrame,
    considered_atts: List[str],
    pool_count_column: Optional[str],
) -> pl.DataFrame:
    """Prepare only direct-mode attributes and preserve optional pool weights."""

    required = set(considered_atts)
    if pool_count_column is not None:
        required.add(pool_count_column)
    missing = required - set(pool.columns)
    if missing:
        raise ValueError(f"Pool is missing columns: {sorted(missing)}")

    selected = pool[considered_atts].copy()
    selected[considered_atts] = selected[considered_atts].astype(str)
    if pool_count_column is None:
        selected[count_field] = 1
    else:
        weights = pd.to_numeric(pool[pool_count_column], errors="raise")
        if weights.isna().any() or (~weights.map(isfinite)).any():
            raise ValueError("Pool weights must be finite")
        if (weights <= 0).any():
            raise ValueError("Pool weights must be positive")
        selected[count_field] = weights.map(lambda value: Decimal(str(value)))

    grouped = (
        selected.groupby(considered_atts, sort=True, as_index=False, dropna=False)[
            count_field
        ]
        .sum()
    )
    return pl.from_pandas(grouped).sort(considered_atts)


def _write_csv_atomically(frame: pl.DataFrame, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        frame.write_csv(temporary_path)
        temporary_path.replace(output_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _write_text_atomically(text: str, output_file: Path) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(text)
            temporary_path = Path(temporary.name)
        temporary_path.replace(output_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _prepare_zone_artifact_directory(
    artifact_dir: Path,
    marg_file: Path,
    pool_file: Path,
    settings: dict[str, object],
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = artifact_dir / "manifest.json"
    manifest = {
        "format_version": 1,
        "marginal_sha256": _file_sha256(marg_file),
        "pool_sha256": _file_sha256(pool_file),
        "settings": settings,
    }
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != manifest:
            raise ValueError(
                f"Zone checkpoints in {artifact_dir} were created with different "
                "inputs or settings; use a new output directory."
            )
        return
    if any(artifact_dir.glob("zone_*.csv")):
        raise ValueError(f"Zone checkpoints in {artifact_dir} have no manifest")
    _write_text_atomically(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", manifest_path
    )


def _validate_zone_artifact(
    artifact_path: Path,
    zone: object,
    expected_columns: list[str],
) -> None:
    artifact = pl.read_csv(
        artifact_path,
        schema_overrides={
            **{column: pl.String for column in expected_columns[:-1]},
            count_field: pl.Int64,
        },
    )
    if artifact.columns != expected_columns:
        raise ValueError(f"Invalid checkpoint columns in {artifact_path}")
    zones = artifact[internal_zone_field].cast(pl.String).unique().to_list()
    if zones != [str(zone)]:
        raise ValueError(f"Invalid checkpoint zone in {artifact_path}: {zones}")
    if not artifact[count_field].dtype.is_integer() or (
        artifact[count_field] <= 0
    ).any():
        raise ValueError(f"Invalid checkpoint counts in {artifact_path}")


def _combine_zone_artifacts_atomically(
    artifact_paths: list[Path], output_file: Path
) -> None:
    output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="wb",
            dir=output_file.parent,
            prefix=f".{output_file.name}.",
            suffix=".tmp",
            delete=False,
        ) as combined:
            temporary_path = Path(combined.name)
            for index, artifact_path in enumerate(artifact_paths):
                with artifact_path.open("rb") as artifact:
                    if index > 0:
                        artifact.readline()
                    shutil.copyfileobj(artifact, combined, length=1024 * 1024)
        temporary_path.replace(output_file)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def run_saa(
        marg_file: Path,
        pool_file: Path,
        zone_field: str,
        output_file: Path,
        considered_atts: Optional[List[str]] = None,
        ordered_to_adjust_atts: Optional[List[str]] = None,
        max_run_time: int = 15,
        extra_rm_frac: float = 0.3,
        last_adjustment_order: Optional[List[str]] = None,
        output_each_step: bool = False,
        add_name_for_step_output: str = "",
        include_zero_cell_values: bool = False,
        randomly_add_last: Optional[List[str]] = None,
        meta_output_dir: Path = None,
        stochastic_mode: Literal[
            "count_native", "legacy_exact", "count_native_ilp"
        ] = "count_native",
        pool_count_column: Optional[str] = None,
        output_format: Literal["rows", "counts"] = "rows",
        direct_solver_mode: Literal[
            "single_stage", "two_stage"
        ] = "single_stage",
        direct_adjustment_backend: Literal[
            "cbc", "min_cost_flow"
        ] = "cbc",
        zone_workers: int = 1,
        ) -> None:
    considered_atts = list(
        CONSIDERED_ATTS_HH if considered_atts is None else considered_atts
    )
    ordered_to_adjust_atts = list(
        SAA_ODERED_ATTS_HH
        if ordered_to_adjust_atts is None
        else ordered_to_adjust_atts
    )
    marg, pool = process_data_from_files(marg_file, pool_file, zone_field)
    if not isinstance(zone_workers, int) or isinstance(zone_workers, bool) or zone_workers < 1:
        raise ValueError("zone_workers must be a positive integer")
    if stochastic_mode == "count_native_ilp":
        condensed_pool = _prepare_direct_pool(
            pool, considered_atts, pool_count_column
        )
    else:
        if pool_count_column is not None:
            raise ValueError("pool_count_column is currently direct-mode only")
        condensed_pool = condense_df(pool.astype(str), maintain_order=True)

    run_kwargs = {
        "considered_atts": considered_atts,
        "ordered_to_adjust_atts": ordered_to_adjust_atts,
        "max_run_time": max_run_time,
        "extra_rm_frac": extra_rm_frac,
        "last_adjustment_order": last_adjustment_order or [],
        "output_each_step": output_each_step,
        "add_name_for_step_output": add_name_for_step_output,
        "include_zero_cell_values": include_zero_cell_values,
        "randomly_add_last": randomly_add_last or [],
        "meta_output_dir": (
            meta_output_dir
            if meta_output_dir is not None
            else output_file.parent / "meta"
        ),
        "stochastic_mode": stochastic_mode,
        "output_format": output_format,
        "direct_solver_mode": direct_solver_mode,
        "direct_adjustment_backend": direct_adjustment_backend,
    }

    if stochastic_mode == "count_native_ilp" and len(marg) > 1:
        if output_each_step:
            raise ValueError(
                "Multi-zone direct runs do not support output_each_step=True"
            )
        if output_format != "counts":
            raise ValueError(
                "Multi-zone checkpointing requires output_format='counts'"
            )
        artifact_dir = output_file.parent / "zone_artifacts"
        checkpoint_settings = {
            "considered_atts": considered_atts,
            "ordered_to_adjust_atts": ordered_to_adjust_atts,
            "last_adjustment_order": last_adjustment_order or [],
            "max_run_time": max_run_time,
            "extra_rm_frac": extra_rm_frac,
            "pool_count_column": pool_count_column,
            "output_format": output_format,
            "direct_solver_mode": direct_solver_mode,
            "direct_adjustment_backend": direct_adjustment_backend,
        }
        _prepare_zone_artifact_directory(
            artifact_dir, marg_file, pool_file, checkpoint_settings
        )
        expected_columns = [internal_zone_field, *considered_atts, count_field]
        artifact_paths = [
            artifact_dir / f"zone_{zone_number:04d}.csv"
            for zone_number in range(1, len(marg) + 1)
        ]
        total_zones = len(marg)
        missing_tasks = []
        completed = 0
        for zone_number, zone in enumerate(marg.index, start=1):
            artifact_path = artifact_paths[zone_number - 1]
            if artifact_path.exists():
                _validate_zone_artifact(artifact_path, zone, expected_columns)
                completed += 1
            else:
                missing_tasks.append(
                    (zone_number, zone, marg.loc[[zone]], artifact_path)
                )

        if zone_workers == 1:
            cache: dict = {}
            try:
                for zone_number, zone, zone_marginal, artifact_path in missing_tasks:
                    print(
                        f"\rProcessing zone {zone} [{zone_number}/{total_zones}]",
                        end="",
                        flush=True,
                    )
                    zone_kwargs = dict(run_kwargs)
                    zone_kwargs["direct_pool_cache"] = cache
                    zone_result, _ = saa_run(
                        zone_marginal, condensed_pool, **zone_kwargs
                    )
                    _write_csv_atomically(zone_result, artifact_path)
                    completed += 1
            finally:
                print()
        elif missing_tasks:
            task_iterator = iter(missing_tasks)
            with ProcessPoolExecutor(
                max_workers=zone_workers,
                initializer=_initialize_direct_worker,
                initargs=(condensed_pool, run_kwargs),
            ) as executor:
                active = {}
                for task in task_iterator:
                    future = executor.submit(_run_direct_zone_worker, *task)
                    active[future] = str(task[1])
                    if len(active) == zone_workers:
                        break
                try:
                    while active:
                        active_zones = ",".join(active.values())
                        print(
                            f"\rCompleted {completed}/{total_zones}; active zones: {active_zones}",
                            end="",
                            flush=True,
                        )
                        done, _ = wait(active, return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result()
                            active.pop(future)
                            completed += 1
                            try:
                                task = next(task_iterator)
                            except StopIteration:
                                continue
                            replacement = executor.submit(
                                _run_direct_zone_worker, *task
                            )
                            active[replacement] = str(task[1])
                finally:
                    print()
        _combine_zone_artifacts_atomically(artifact_paths, output_file)
        print(f"Combined {total_zones} zone checkpoints into {output_file}")
        return

    final_syn_hh, _ = saa_run(marg, condensed_pool, **run_kwargs)
    sort_columns = [internal_zone_field, *considered_atts]
    final_syn_hh = final_syn_hh.sort(sort_columns)
    _write_csv_atomically(final_syn_hh, output_file)

