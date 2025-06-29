"""Use the .yml to extract the data combined for general processing"""
from pathlib import Path
import yaml
import pandas as pd
from PopSynthesis.Benchmark.CompareCensus.utils import convert_raw_census, convert_syn_pop_raw
from PopSynthesis.Benchmark.CompareCensus.compare import get_RMSE
from PopSynthesis.analyse.utils.process_yaml import handle_yaml_abs_path

def extract_general_from_resulted_syn(yaml_path: Path, output_path: Path, level: str = "hh") -> None:
    """Extract general data from the result .yml file."""
    with open(yaml_path, 'r') as file:
        configs = yaml.safe_load(file)
    for config_run in configs:
        # method = config_run["method"]
        if level == "hh":
            census_path_raw = config_run["hh_marg_file"]
            syn_pop_file = f"{config_run['hh_syn_name']}"
        elif level == "pp":
            census_path_raw = config_run["pp_marg_file"]
            syn_pop_file = f"{config_run['pp_syn_name']}"
        else:
            raise ValueError(f"Unknown level: {level}. Use 'hh' or 'pp'.")

        census_path = handle_yaml_abs_path(Path(census_path_raw), yaml_path)
        census_raw = pd.read_csv(census_path, header=[0, 1])
        census = convert_raw_census(census_raw)
        for run in range(config_run["reruns"]):
            result_path = output_path / config_run["output_name"] / f"reruns_{run}"
            syn_pop_path = result_path / syn_pop_file
            syn_pop = pd.read_csv(syn_pop_path)
            syn_pop = convert_syn_pop_raw(syn_pop, census)
            # Process the benchmarks for each population to check the results
            attr_rmse = get_RMSE(census, syn_pop, return_type="attribute")
            print(attr_rmse)


def extract_saa_runs_meta(meta_path: Path):
    NotImplemented

if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    IO_path = current_file_path / "../../../../IO"
    yaml_path = IO_path / "configs/test.yml"
    corresponding_output_path = IO_path / "output/runs"
    extract_general_from_resulted_syn(yaml_path, corresponding_output_path)