"""Command-line entry point for completed household experiment analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import AnalysisOptions, HouseholdExperimentAnalysis


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze completed household synthesis experiments without rerunning them."
    )
    parser.add_argument("--config", type=Path, required=True, help="Experiment YAML")
    parser.add_argument("--output", type=Path, required=True, help="Analysis output directory")
    parser.add_argument("--bootstrap-replicates", type=int, default=2_000)
    parser.add_argument("--analysis-seed", type=int, default=20_261_113)
    parser.add_argument("--smoothing", type=float, default=1e-12)
    parser.add_argument("--no-figures", action="store_true")
    arguments = parser.parse_args()
    if arguments.bootstrap_replicates <= 0:
        parser.error("--bootstrap-replicates must be positive")
    if arguments.smoothing <= 0:
        parser.error("--smoothing must be positive")
    analyzer = HouseholdExperimentAnalysis(
        arguments.config,
        arguments.output,
        AnalysisOptions(
            smoothing=arguments.smoothing,
            bootstrap_replicates=arguments.bootstrap_replicates,
            analysis_seed=arguments.analysis_seed,
            write_figures=not arguments.no_figures,
        ),
    )
    manifest = analyzer.run()
    print(
        f"Analyzed {manifest['configured_results']} results across "
        f"{manifest['zones']} zones; outputs: {arguments.output.resolve()}"
    )


if __name__ == "__main__":
    main()
