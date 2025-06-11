import click
import yaml
from pathlib import Path
from PopSynthesis.Methods.IPU.run import run_ipu
from PopSynthesis.Methods.IPSF.run import run_saa
import time
import pandas as pd

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
    timing_records = []
    for run_info in runs:
        def get_file_general(key):
            return process_absolute_path(
                Path(run_info[key]),
                Path(runs_yml).parent
            )
        method = run_info["method"]  
        for rerun in range(run_info["reruns"]):
            output_path_run = Path(output_path) / run_info["output_name"] / f"reruns_{rerun}"
            output_path_run.mkdir(parents=True, exist_ok=True)
            # Check runtime
            begin_time = time.time()
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
            elif method == "saa":
                marg = pool = None
                if run_info["level"] == "hh":
                    marg, pool = get_file_general("hh_marg_file"), get_file_general("hh_sample_file")
                elif run_info["level"] == "pp":
                    marg, pool = get_file_general("p_marg_file"), get_file_general("p_sample_file")
                else:
                    raise ValueError(f"Level {run_info['level']} not supported.")
                meta_output_dir = output_path_run / "meta"
                meta_output_dir.mkdir(parents=True, exist_ok=True)
                run_saa(
                    marg_file=marg,
                    pool_file=pool,
                    zone_field=run_info["zone_field"],
                    output_file=output_path_run / run_info["hh_syn_name"],
                    considered_atts=run_info["considered_atts"],
                    ordered_to_adjust_atts=run_info["ordered_to_adjust_atts"],
                    max_run_time=run_info["max_run_time"],
                    extra_rm_frac=run_info["extra_rm_frac"],
                    last_adjustment_order=run_info["last_adjustment_order"],
                    output_each_step=run_info["output_each_step"],
                    add_name_for_step_output=run_info["add_name_for_step_output"],
                    include_zero_cell_values=run_info["include_zero_cell_values"],
                    randomly_add_last=run_info.get("randomly_add_last", []),
                    meta_output_dir=meta_output_dir
                )
            else:
                raise ValueError(f"Method {method} not supported.")
            end_time = time.time()
            elapsed_time = end_time - begin_time
            timing_records.append(
                {
                    "method": method,
                    "run_name": run_info["output_name"],
                    "rerun": rerun,
                    "time_seconds": elapsed_time,
                }
            )
            click.echo(f"Finished running {method} for rerun {rerun}, output saved to {output_path_run}")
            click.echo(f"Time for run: {elapsed_time:.2f} seconds")

    if timing_records:
        df = pd.DataFrame(timing_records)
        csv_path = Path(output_path) / "run_times.csv"
        df.to_csv(csv_path, index=False)
        click.echo(f"All runs finished. Runtimes saved to {csv_path}")
    
