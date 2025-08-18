"""Use the .yml to extract the data combined for general processing"""
from pathlib import Path
import yaml
import pandas as pd
import polars as pl
from glob import glob
from PopSynthesis.Benchmark.CompareCensus.utils import convert_raw_census, convert_syn_pop_raw
from PopSynthesis.Benchmark.CompareCensus.compare import get_RMSE
from PopSynthesis.analyse.utils.process_yaml import handle_yaml_abs_path
from typing import List

# NOTE: only RMSE for census comparison, not JSD yet

def extract_general_from_resulted_syn(yaml_path: Path, output_path: Path, level: str = "hh") -> tuple[pl.DataFrame, pl.DataFrame]:
    """Extract general data from the result .yml file."""
    with open(yaml_path, 'r') as file:
        configs = yaml.safe_load(file)
    
    results_main = []
    for config_run in configs:
        print(f"Processing run: {config_run['output_name']}")
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
        # Read census data with pandas (for MultiIndex header) and convert
        census_raw_pd = pd.read_csv(census_path, header=[0, 1])
        census_pd = convert_raw_census(census_raw_pd)

        store_diff_runs = []
        extra_results = []
        
        for run in range(config_run["reruns"]):
            # Results for each rerun (completely separate)
            result_path = output_path / config_run["output_name"] / f"reruns_{run}"
            syn_pop_path = result_path / syn_pop_file
            
            # Read synthetic population with Polars
            syn_pop_pl = pl.read_csv(syn_pop_path, infer_schema_length=0)
            # Convert to pandas for external function processing
            syn_pop_pd = syn_pop_pl.to_pandas()
            syn_pop_converted_pd = convert_syn_pop_raw(syn_pop_pd, census_pd)
            syn_pop_converted_pd = syn_pop_converted_pd[census_pd.columns]  # Ensure columns match census
            syn_pop_converted_pd = syn_pop_converted_pd.reindex(census_pd.index)  # Ensure index matches census
            assert syn_pop_converted_pd.columns.equals(census_pd.columns), \
                "Columns of synthetic population do not match census."
            assert syn_pop_converted_pd.index.equals(census_pd.index), \
                "Index of synthetic population does not match census."
            
            # Process the benchmarks for each population to check the results
            attr_rmse_pd = get_RMSE(census_pd.to_numpy(), syn_pop_converted_pd.to_numpy(), return_type="attribute")
            attr_rmse_pd = pd.Series(attr_rmse_pd, index=syn_pop_converted_pd.columns, name=f"run_{run}")
            
            # Convert RMSE results to Polars - handle MultiIndex properly
            attr_rmse_pl = pl.Series(name=f"run_{run}", values=attr_rmse_pd.values, dtype=pl.Float64)
            
            # Convert MultiIndex to strings for attribute names
            if isinstance(attr_rmse_pd.index, pd.MultiIndex):
                attr_names = [str(idx) for idx in attr_rmse_pd.index]
            else:
                attr_names = attr_rmse_pd.index.tolist()
            
            attr_rmse_pl = attr_rmse_pl.to_frame().with_columns(
                pl.Series(name="attribute", values=attr_names)
            )
            store_diff_runs.append(attr_rmse_pl)

            # consider special case for saa
            # if method == "saa":
            #     saa_meta_path = result_path / "meta"
            #     meta_results_pd = extract_saa_runs_meta(saa_meta_path, config_run["max_run_time"], 
            #                                           config_run["ordered_to_adjust_atts"], census_pd)
            #     # Convert meta results to Polars
            #     meta_results_pl = pl.from_pandas(meta_results_pd)
            #     extra_results.append(meta_results_pl)
        
        # Combine RMSE results
        fin_rmse_records = pl.DataFrame()
        if store_diff_runs:
            # Create a base DataFrame with attributes
            base_df = store_diff_runs[0].select("attribute")
            
            # Join all run results
            for i, run_df in enumerate(store_diff_runs):
                base_df = base_df.join(
                    run_df.select(["attribute", f"run_{i}"]),
                    on="attribute",
                    how="left"
                )
            
            fin_rmse_records = base_df
            # attribute now is like ('totalvehs', '0'), need separate them into 2 columns
            fin_rmse_records = fin_rmse_records.with_columns([
                fin_rmse_records["attribute"].str.extract(r'\(([^,]+),', 1).alias("att"),
                fin_rmse_records["attribute"].str.extract(r',\s*([^)]+)\)', 1).alias("state")
            ]).drop("attribute")
            fin_rmse_records = fin_rmse_records.with_columns(
                pl.lit(config_run["output_name"]).alias("method_run")
            )
        # Process meta results for SAA
        # fin_meta_results = pl.DataFrame()
        # if len(extra_results) > 0:
        #     # Extract mean columns from meta results
        #     meta_dfs = []
        #     for i, meta_df in enumerate(extra_results):
        #         # Find columns that contain "mean" in their name
        #         mean_cols = [col for col in meta_df.columns if "mean" in col.lower()]
        #         if mean_cols:
        #             mean_col = mean_cols[0]
        #             meta_series = meta_df.select(mean_col).to_series()
        #             meta_series = meta_series.alias(f"run_{i}")
        #             meta_dfs.append(meta_series.to_frame())
            
        #     if meta_dfs:
        #         # Combine all meta results horizontally
        #         fin_meta_results = meta_dfs[0]
        #         for meta_df in meta_dfs[1:]:
        #             fin_meta_results = fin_meta_results.hstack(meta_df)
        
        results_main.append(fin_rmse_records)
    return pl.concat(results_main)


def extract_saa_runs_meta(meta_path: Path, n_adjust: int, adjusted_atts: List[str], census_pd: pd.DataFrame) -> pd.DataFrame:
    # So what we need to do here is actually combine the kept and adjusted to have the population at each step
    rmse_results = []
    chosen_syn_dfs = []
    
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
            
            # Read with Polars and convert to pandas for external functions - treat all as strings
            adjusted_syn_pl = pl.read_csv(adjusted_file, infer_schema_length=0)
            adjusted_syn_pd = adjusted_syn_pl.to_pandas()
                
            curr_syn_pd = adjusted_syn_pd
            if len(chosen_syn_dfs) > 0:
                # Combine all chosen synthetic populations
                chosen_syn_combined = pl.concat(chosen_syn_dfs + [adjusted_syn_pl])
                curr_syn_pd = chosen_syn_combined.to_pandas()
            
            curr_syn_converted_pd = convert_syn_pop_raw(curr_syn_pd, census_pd)
            
            # Process the benchmarks for each population to check the results
            attr_rmse = get_RMSE(census_pd.to_numpy(), curr_syn_converted_pd.to_numpy(), return_type="attribute")
            attr_rmse = pd.Series(attr_rmse, index=curr_syn_converted_pd.columns, name=f"run_{i}_adjusted_{j}")
            # replace the index of attr_rmse which is a pd.Series with the attribute name
            attr_rmse.loc["mean"] = attr_rmse.mean()
            attr_rmse.loc["adjusted_att"] = adjusted_att
            rmse_results.append(attr_rmse)

        # Read kept synthetic population - treat all as strings
        result_kept_syn_pl = pl.read_csv(meta_path / f"kept_syn_run_{i}.csv", infer_schema_length=0)
        chosen_syn_dfs.append(result_kept_syn_pl)
    
    return pd.concat(rmse_results, axis=1).T


if __name__ == "__main__":
    current_file_path = Path(__file__).resolve()
    IO_path = current_file_path / "../../../../IO"
    # yaml_path = IO_path / "configs/test.yml"
    # corresponding_output_path = IO_path / "output/runs/small"
    # yaml_path = IO_path / "configs/runs.yml"
    # corresponding_output_path = IO_path / "output/runs/big"
    yaml_path = IO_path / "configs/extra_runs.yml"
    corresponding_output_path = IO_path / "output/runs/others_quick"
    a = extract_general_from_resulted_syn(yaml_path, corresponding_output_path)
    # print(a.mean(axis=1))
    a.write_csv(corresponding_output_path / "fin_rmse_records.csv")
    print(a)