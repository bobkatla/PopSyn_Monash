"""Use the .yml to extract the data combined for general processing"""
from pathlib import Path
import yaml
import pandas as pd
# from PopSynthesis.analyse.utils.process_yaml import handle_yaml_abs_path

def extract_general_from_resulted_syn(yaml_path: Path, output_path: Path, level: str = "hh") -> None:
    """Extract general data from the result .yml file."""
    with open(yaml_path, 'r') as file:
        configs = yaml.safe_load(file)
    for config_run in configs:
        # method = config_run["method"]
        for run in range(config_run["reruns"]):
            result_path = output_path / config_run["output_name"] / f"reruns_{run}"
            if level == "hh":
                syn_pop_path = result_path / f"{config_run['hh_syn_name']}"
            elif level == "pp":
                syn_pop_path = result_path / f"{config_run['pp_syn_name']}"
            else:
                raise ValueError(f"Unknown level: {level}. Use 'hh' or 'pp'.")
            syn_pop = pd.read_csv(syn_pop_path)
            
            print(syn_pop)


def extract_saa_runs():
    NotImplemented

if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    IO_path = current_file_path / "../../../../IO"
    yaml_path = IO_path / "configs/test.yml"
    corresponding_output_path = IO_path / "output/runs"
    extract_general_from_resulted_syn(yaml_path, corresponding_output_path)