# Population Synthesis

Population Synthesis is a modular framework for generating synthetic populations using various statistical and machine learning methods. It is designed for flexibility, reproducibility, and extensibility, supporting multiple synthesis algorithms and comprehensive benchmarking.

## Project Structure

- **PopSynthesis/**: Main package containing all core modules.
  - **cli/**: Command-line interface (CLI) for running synthesis tasks via `click`.
  - **DataProcessor/**: Classes and utilities for processing raw and intermediate data, including `DataProcessorGeneric` for unified data handling.
  - **Methods/**: Implementation of synthesis methods (e.g., IPU, IPSF/SAA, BN, CSP, GAN, RL, naive, IPF). Each method is in its own subfolder.
  - **Benchmark/**: Tools and scripts for comparing synthetic results to census or ground truth data.
  - **Generator_data/**: Scripts and notebooks for generating and preparing input data.
  - **Analyse/**: Jupyter notebooks for analysis and visualization of results.
  - **tests/**: Unit and integration tests for methods and utilities.

## Installation

Requirements are managed in `pyproject.toml`. Main dependencies include:
- Python >= 3.9
- pandas, polars, numpy, geopandas, scikit-learn, scipy, torch, click, pyyaml, pulp, pgmpy, ipfn, pyarrow

Install with:

```powershell
pip install .
```

Or, for development:

```powershell
pip install -e .
```

The project uses uv so it can be installed on another machine for CLI run only
```bash
uv pip install git+https://github.com/bobkatla/PopSyn_Monash.git
```
or to upgrade (for new developments) with
```bash
uv pip install --upgrade git+https://github.com/bobkatla/PopSyn_Monash.git
```

## Usage

### Command Line Interface (CLI)

The main entry point is the `psim` command (see `PopSynthesis/cli/main.py`). Example:

```powershell
psim synthesize --runs-yml IO/configs/runs.yml --output-path IO/output/
```

- `--runs-yml`: Path to a YAML file specifying synthesis runs and parameters.
- `--output-path`: Directory for output files.

### Programmatic Usage

You can also use the main classes and methods in your own scripts. For example, to run the IPU method:

```python
from PopSynthesis.Methods.IPU.run import run_ipu
run_ipu(hh_marg_file, p_marg_file, hh_sample_file, p_sample_file, output_path, hh_syn_name, pp_syn_name, stats_name)
```

Or to process data:

```python
from PopSynthesis.DataProcessor.DataProcessor import DataProcessorGeneric
processor = DataProcessorGeneric(raw_data_dir, processed_data_dir, output_dir)
processor.process_all_seed()
```

## Data and Configuration

- Input data and configuration files are in `IO/configs/` and `PopSynthesis/DataProcessor/data/`.
- Output and processed data are in `IO/output/` and `PopSynthesis/DataProcessor/output/`.

## Testing

Tests are in `PopSynthesis/tests/` and can be run with your preferred test runner (e.g., pytest):

```powershell
pytest PopSynthesis/tests/
```

## Notebooks

Analysis and visualization notebooks are in `PopSynthesis/Analyse/` and `PopSynthesis/DataProcessor/utils/`.

## Extending

- Add new synthesis methods in `PopSynthesis/Methods/`.
- Utilities and shared code go in `PopSynthesis/DataProcessor/utils/`.
- Benchmarking tools are in `PopSynthesis/Benchmark/`.
- Add tests in `PopSynthesis/tests/`.

---

For more details, see the code and docstrings in each module. Contributions and suggestions are welcome!
