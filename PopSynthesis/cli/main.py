import click
import yaml
from pathlib import Path
from PopSynthesis.Methods.IPU.run import run_ipu


def process_absolute_path(path: Path, parent_path: Path) -> Path:
    if not path.is_absolute():
        path = parent_path / path
    return path

@click.group()
def cli():
    """A command-line interface for the PopSynthesis project."""
    pass

@click.option(
    "-runs", "--runs-yml",
    type=click.Path(exists=True, file_okay=True, dir_okay=False),
    required=True,
    help="The path to the runs YAML file."
)
@click.option(
    "-o", "--output-path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
    required=True,
    help="The path to the output directory."
)
@cli.command()
def synthesize(runs_yml, output_path):
    """Run the synthetic population generation."""
    with open(runs_yml, "r") as f:
        runs = yaml.safe_load(f)
    for run_info in runs:
        def get_file_general(key):
            return process_absolute_path(
                Path(run_info[key]),
                Path(runs_yml).parent
            )
        method = run_info["method"]  
        output_path_run = Path(output_path) / run_info["output_name"]
        output_path_run.mkdir(parents=True, exist_ok=True)

        # Run the method
        if method == "ipu":
            run_ipu(
                hh_marg_file=get_file_general("hh_marg_file"),
                p_marg_file=get_file_general("p_marg_file"),
                hh_sample_file=get_file_general("hh_sample_file"),
                p_sample_file=get_file_general("p_sample_file"),
                output_path=output_path_run,
                hh_syn_name=run_info["hh_syn_name"],
                pp_syn_name=run_info["pp_syn_name"],
                stats_name=run_info["stats_name"]
            )
        else:
            raise ValueError(f"Method {method} not supported.")
        click.echo(f"Finished running {method}, output saved to {output_path_run}")
    
