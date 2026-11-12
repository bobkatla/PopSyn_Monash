"""Command line entry point for household experiment suites."""

from __future__ import annotations

import argparse
from pathlib import Path

from PopSynthesis.experiments.config import load_config
from PopSynthesis.experiments.runner import ExperimentRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML path")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and recreate the complete configured output root before "
            "running. Without this flag, any existing repeat directory is skipped."
        ),
    )
    return parser


def main() -> None:
    arguments = build_parser().parse_args()
    ExperimentRunner(
        load_config(arguments.config), overwrite=arguments.overwrite
    ).run()


if __name__ == "__main__":
    main()
