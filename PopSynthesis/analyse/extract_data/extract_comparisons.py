"""Use the .yml to extract the data combined for general processing"""
from pathlib import Path
import yaml
import pandas as pd
from glob import glob
from PopSynthesis.Benchmark.CompareCensus.utils import convert_raw_census, convert_syn_pop_raw
from PopSynthesis.Benchmark.CompareCensus.compare import get_RMSE
from PopSynthesis.analyse.utils.process_yaml import handle_yaml_abs_path
from typing import List

# NOTE: only RMSE for census comparison, not JSD yet

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

        store_diff_runs = []
        for run in range(config_run["reruns"]):
            # Results for each rerun (completely separate)
            result_path = output_path / config_run["output_name"] / f"reruns_{run}"
            syn_pop_path = result_path / syn_pop_file
            syn_pop = pd.read_csv(syn_pop_path)
            syn_pop = convert_syn_pop_raw(syn_pop, census)
            # Process the benchmarks for each population to check the results
            attr_rmse = get_RMSE(census, syn_pop, return_type="attribute")
            attr_rmse.name = f"run_{run}"
            store_diff_runs.append(attr_rmse)

            # consider special case for saa
            if config_run["method"] == "saa":
                saa_meta_path = result_path / "meta"
                meta_results = extract_saa_runs_meta(saa_meta_path, config_run["max_run_time"], config_run["ordered_to_adjust_atts"], census)
                print(meta_results)
        fin_rmse_records = pd.concat(store_diff_runs, axis=1).T


def extract_saa_runs_meta(meta_path: Path, n_adjust: int, adjusted_atts: List[str], census: pd.DataFrame) -> pd.DataFrame:
    # So what we need to do here is actually combine the kept and adjusted to have the population at each step
    rmse_results = []
    chosen_syn = []
    for i in range(n_adjust):
        for j in range(len(adjusted_atts)):
            # find the file
            glob_pattern = f"step_adjusted_{j}_*_{i}.csv"
            possible_files = glob(str(meta_path / glob_pattern))
            if len(possible_files) == 0:
                print(f"No adjusted files found for pattern: {glob_pattern}")
                continue
            assert len(possible_files) == 1, f"Expected one file matching only {glob_pattern}, found {len(possible_files)}"
            adjusted_file = possible_files[0]
            adjusted_att = Path(adjusted_file).stem.replace(f"step_adjusted_{j}_", "").replace(f"_{i}", "")
            adjusted_syn = pd.read_csv(adjusted_file)
                
            curr_syn = adjusted_syn
            if len(chosen_syn) > 0:
                curr_syn = pd.concat(chosen_syn + [adjusted_syn], axis=0)
            curr_syn = convert_syn_pop_raw(curr_syn, census)
            # Process the benchmarks for each population to check the results
            attr_rmse = get_RMSE(census, curr_syn, return_type="attribute")
            # replace the index of attr_rmse which is a pd.Series with the attribute name
            attr_rmse.name = f"run_{i}_adjusted_{j}"
            attr_rmse.loc["mean"] = attr_rmse.mean()
            attr_rmse.loc["adjusted_att"] = adjusted_att
            rmse_results.append(attr_rmse)

        result_kept_syn = pd.read_csv(meta_path / f"kept_syn_run_{i}.csv")
        chosen_syn.append(result_kept_syn)
    return pd.concat(rmse_results, axis=1).T


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    IO_path = current_file_path / "../../../../IO"
    yaml_path = IO_path / "configs/test.yml"
    corresponding_output_path = IO_path / "output/runs"
    extract_general_from_resulted_syn(yaml_path, corresponding_output_path)