"""Shared CBC construction for the SAA optimisation backends."""

from __future__ import annotations

from pathlib import Path

from pulp import COIN_CMD


CBC_RANDOM_SEED = 42


def create_cbc_solver(
    *,
    reproducible: bool,
    time_limit: float | None = None,
    log_path: Path | None = None,
) -> COIN_CMD:
    """Create the CBC command solver required by PuLP 4.

    Existing SAA modes retain CBC's default configuration.  The direct-count
    backend fixes both CLP and CBC seeds and uses one thread so repeated solves
    in the same software environment are as stable as CBC permits.
    """

    options: list[str] = []
    kwargs: dict[str, object] = {
        "msg": False,
        "timeLimit": time_limit,
        "logPath": str(log_path) if log_path is not None else None,
        "timeMode": "elapsed",
    }
    if reproducible:
        kwargs.update(
            threads=1,
            randomSeed=CBC_RANDOM_SEED,
            gapRel=0.0,
            gapAbs=0.0,
        )
        options.append(f"randomCbcSeed {CBC_RANDOM_SEED}")
    if options:
        kwargs["options"] = options

    solver = COIN_CMD(**kwargs)
    if not solver.available():
        raise RuntimeError(
            "CBC is unavailable. Install PuLP with CBC support using "
            "`pip install 'pulp[cbc]>=4.2,<5'`, or place `cbc` on PATH."
        )
    return solver
