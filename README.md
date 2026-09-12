# PopSynthesis

PopSynthesis is a research codebase for generating and evaluating spatial synthetic populations. The currently integrated command-line workflows are:

- **IPU** for linked household/person synthesis through the external [`synthpop`](https://github.com/bobkatla/synthpop) package.
- **SAA** (Sequential Attribute Adjustment), the main implementation in this repository, for synthesising household or person records from zonal one-way marginals and a record pool.

The repository also contains experimental or legacy implementations of IPF, Bayesian networks, CSP, GAN, RL, and naive sampling. These are useful research references, but they are not currently connected as working CLI methods.

For a code-oriented map of the repository and a detailed SAA execution trace, see [`context.md`](context.md).

## Current execution flow

The installed command is `psim`. A synthesis run starts at `PopSynthesis/cli/main.py`, reads a list of runs from YAML, resolves input paths relative to that YAML file, and creates this output layout:

```text
<output-path>/
├── run_times.csv
└── <output_name>/
    └── reruns_<n>/
        ├── <synthetic population CSV>
        └── meta/                    # optional SAA intermediate output
```

Run a configuration with:

```bash
mkdir -p IO/output
psim synthesize --runs-yml IO/configs/test.yml --output-path IO/output
```

`IO/configs/runs.yml` contains full SAA experiment configurations, while `IO/configs/test.yml` contains a smaller example. The referenced processed data are intentionally not tracked as ordinary Git files, so supply or generate those files before running the examples.

## Installation

Python 3.9 or newer is declared in `pyproject.toml`. The project uses a Git dependency, so Git must also be available.

With `uv`:

```bash
uv sync
uv run psim --help
```

Or with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
psim --help
```

The dependency set includes pandas, Polars, NumPy, PuLP/CBC, SciPy, scikit-learn, pgmpy, PyTorch, GeoPandas, and PyArrow. The `uv` configuration directs PyTorch to the CPU wheel index; a compatible wheel must exist for the host Python/platform combination.

## SAA inputs

SAA consumes two CSV files:

1. **Marginals**: a CSV with a two-row header. The first header level is the attribute and the second is its state. It must contain the configured zone column (normally `zone_id`) and, with the current loader, a `sample_geog` column that is dropped before synthesis. Counts should be non-negative integers, and every controlled attribute should imply the same population total in each zone.
2. **Pool**: one complete candidate record per row. It must contain every `considered_atts` column. `serialno` and `sample_geog` are dropped when present. Duplicate records are condensed and used as sampling weights.

The default household controls are:

```text
hhsize, hhinc, totalvehs, dwelltype, owndwell
```

The default person controls are:

```text
age, sex, persinc, anywork
```

Person output can additionally retain `nolicence` and `relationship` as uncontrolled attributes.

### SAA YAML example

```yaml
- method: saa
  level: hh
  hh_marg_file: ../processed_data/small/hh_marginals_small.csv
  hh_sample_file: ../processed_data/full/HH_pool.csv
  zone_field: zone_id

  considered_atts:
    - hhsize
    - hhinc
    - totalvehs
    - dwelltype
    - owndwell
  ordered_to_adjust_atts:
    - hhsize
    - totalvehs
    - hhinc
    - dwelltype
    - owndwell
  last_adjustment_order:
    - hhsize
    - totalvehs
    - hhinc
    - dwelltype
    - owndwell

  max_run_time: 10
  extra_rm_frac: 0.3
  randomly_add_last:
    - hhinc
  include_zero_cell_values: false
  output_each_step: true
  add_name_for_step_output: test

  output_name: saa_small
  hh_syn_name: hh_syn_saa.csv
  reruns: 2
```

Despite its name, `max_run_time` is a maximum **number of SAA passes**, not a wall-clock duration. SAA is stochastic and the current CLI does not expose a random seed.

## What SAA currently does

For each pass, SAA:

1. Initialises a population that exactly matches the first controlled attribute in every zone by weighted sampling from the pool.
2. Adjusts each later attribute zone-by-zone. A PuLP integer program redistributes counts while preserving combinations of attributes already adjusted and respecting combinations available in the pool.
3. Expands adjusted counts back into full records by weighted conditional sampling from the pool.
4. Unless it is the last pass, optionally removes a random fraction and then removes oversupplied records. The retained records are accumulated and the remaining marginal deficit becomes the target for the next pass.
5. Concatenates records retained across all passes and writes the requested CSV.

Current features include household and person-level runs, configurable control/output attributes, dynamic attribute ordering, a forced final order, optional zero-cell handling, multiple reruns, intermediate step CSVs, and runtime reporting. See `context.md` for exact semantics and known limitations.

## Programmatic SAA use

```python
from pathlib import Path
from PopSynthesis.Methods.IPSF.run import run_saa

run_saa(
    marg_file=Path("hh_marginals.csv"),
    pool_file=Path("HH_pool.csv"),
    zone_field="zone_id",
    output_file=Path("output/hh_syn_saa.csv"),
    considered_atts=["hhsize", "hhinc", "totalvehs", "dwelltype", "owndwell"],
    ordered_to_adjust_atts=["hhsize", "totalvehs", "hhinc", "dwelltype", "owndwell"],
    max_run_time=10,
    extra_rm_frac=0.3,
)
```

The output directory must already exist when calling `run_saa` directly.

## IPU

IPU delegates loading and synthesis to `synthpop.zone_synthesizer` and writes household, person, and statistics CSVs. Its YAML configuration requires household/person marginals and linked household/person sample files. See the commented examples in `IO/configs/runs.yml` and `PopSynthesis/Methods/IPU/run.py`.

## Repository map

```text
PopSynthesis/
├── cli/                  # psim command and YAML orchestration
├── Methods/
│   ├── IPSF/SAA/         # active SAA implementation
│   ├── IPSF/utils/       # ILP, condensation, and residual-marginal helpers
│   ├── IPU/              # active IPU wrapper
│   └── BN, CSP, GAN, ... # experimental and legacy methods
├── Benchmark/            # census/full-population comparison helpers
├── DataProcessor/        # seed-processing research code
├── Generator_data/       # data-generation scripts and notebooks
├── analyse/              # metric extraction, plots, and notebooks
└── tests/                # current tests and small fixtures
```

## Testing and current caveats

Run the tests after installing development dependencies and fetching Git LFS data:

```bash
pytest -q
```

Important current caveats:

- Several tracked CSV fixtures are stored through Git LFS. A checkout containing pointer text instead of CSV content cannot run those tests.
- The existing SAA tests execute functions at module import time and mostly print results; coverage and assertions are limited.
- IPF, BN, and WGAN are listed in the CLI dispatcher but are not implemented there. Their current branches also fail to raise the constructed `NotImplementedError`.
- Some standalone research scripts contain machine-specific Windows paths and are not portable entry points. Prefer `psim synthesize` or `PopSynthesis.Methods.IPSF.run.run_saa`.
- SAA zero-cell support is partial, and ILP slack means later attributes are best-effort when no exact pool-supported adjustment exists.

## Batch execution

`job_saa.slurm` is the current Slurm example. It activates a `psim` Conda environment and runs:

```bash
psim synthesize -runs runs.yml -o out/
```

Adjust the environment name, paths, resources, and configuration filename for the target cluster.
