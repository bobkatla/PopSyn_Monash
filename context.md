# PopSynthesis repository context

This is a fast code-reading guide for future agents. It describes the current `main` snapshot, with emphasis on the executed SAA path. Treat code as authoritative if this file and a later revision disagree.

## Executive summary

- This is a research repository for spatial synthetic population generation and evaluation.
- The package entry point is `psim = PopSynthesis.cli.main:cli` (`pyproject.toml`).
- `psim synthesize` currently has two real execution paths: IPU and SAA.
- SAA means **Sequential Attribute Adjustment**. It generates independent household-level or person-level records from zonal one-way marginals plus a complete-record pool.
- SAA is stochastic. It repeatedly builds a population for residual marginals, retains only records that do not oversupply any controlled cell, and synthesises the remaining deficit.
- Within one pass, the first attribute is matched directly. Each later attribute is adjusted per zone using a PuLP integer program that preserves distributions fixed earlier in that pass where the pool supports the required combinations.
- IPF, BN, WGAN, CSP, GAN, RL, and naive implementations exist as experiments/legacy code. The CLI's `ipf`, `bn`, and `wgan` branches do not execute them.
- Processed/raw datasets and outputs are ignored. Several committed test CSVs are Git LFS objects and may appear as pointer files when LFS content is unavailable.

## Main entry points

| Purpose | File / symbol | Notes |
|---|---|---|
| Package command | `PopSynthesis/cli/main.py::cli` | Click group; exposes `synthesize`. |
| YAML orchestration | `PopSynthesis/cli/main.py::synthesize` | Loops configurations and reruns; records wall time. |
| SAA file wrapper | `PopSynthesis/Methods/IPSF/run.py::run_saa` | Loads CSVs, condenses pool, calls `saa_run`, writes output. |
| Multi-pass SAA | `PopSynthesis/Methods/IPSF/SAA/operations/wrapper_saa_run.py::saa_run` | Attribute ordering, loopback/pruning, residual target. |
| One SAA pass | `PopSynthesis/Methods/IPSF/SAA/SAA.py::SAA.run` | Iterates controlled attributes. |
| Attribute adjustment | `PopSynthesis/Methods/IPSF/SAA/operations/general.py::adjust_atts_state_match_census` | Initial sample or per-zone ILP adjustment. |
| Zone ILP | `PopSynthesis/Methods/IPSF/SAA/operations/ILP_zone_ad.py::ILP_zone_adjustment` | Converts, solves, and samples counts back to records. |
| ILP model | `PopSynthesis/Methods/IPSF/utils/ILP_matrix_ad.py` | Integer adjustments, slack, row/column constraints. |
| Loopback pruning | `PopSynthesis/Methods/IPSF/utils/synthetic_checked_census.py::adjust_kept_rec_match_census` | Removes oversupplied record combinations. |
| IPU wrapper | `PopSynthesis/Methods/IPU/run.py::run_ipu` | Delegates to external `synthpop`. |
| SAA analysis | `PopSynthesis/analyse/extract_data/extract_comparisons.py` | RMSE/JSD/diversity and optional intermediate-step metrics. |

`PopSynthesis/Methods/IPSF/SAA/run_hh.py`, `run_pp.py`, and the two `small_*_run.py` files are old standalone research drivers. They contain hard-coded Windows paths and are not the portable interface.

## CLI execution flow

```bash
psim synthesize --runs-yml <config.yml> --output-path <existing-directory>
```

Flow in `PopSynthesis/cli/main.py`:

1. Parse the YAML as a list of run dictionaries.
2. Resolve each input path relative to the YAML's parent unless it is absolute.
3. For every `rerun`, create `<output>/<output_name>/reruns_<i>/`.
4. Dispatch on `method`:
   - `ipu`: call `run_ipu(...)`.
   - `saa`: choose household or person marginal/pool keys from `level`, create `meta/`, call `run_saa(...)`.
   - `ipf`, `bn`, `wgan`: currently construct `NotImplementedError` but do not `raise` it; no synthesis happens, yet timing/success output still follows.
5. Append timing data and write `<output>/run_times.csv`.

Output names for SAA are asymmetric: the CLI always passes `hh_syn_name`, including when `level: pp`. A person-level config therefore still needs `hh_syn_name` unless the dispatcher is fixed. `hh_seed_data` in the provided SAA configs is unused by synthesis; analysis scripts use it for JSD/diversity comparisons.

## SAA input contract

### Marginal CSV

- Read by pandas with `header=[0, 1]`.
- Two-level columns encode `(attribute, state)`.
- One top-level column must match configured `zone_field`. Production code assumes the internal name `zone_id` after indexing. Passing another name only helps locate the input column; the lower SAA stack imports `zone_field = "zone_id"`.
- A top-level `sample_geog` column is expected and dropped by `process_data_from_files`. Its absence currently causes an index error because the drop is not guarded.
- Each zone should be unique. Controlled counts should be non-negative integers.
- Every controlled attribute should total to the same number of agents. SAA derives requested population as `marg.sum().sum() / number_of_attributes`.
- State labels must compare cleanly with pool values. Pool values and CSV header state labels are strings.

### Pool CSV

- One candidate complete record per row.
- Must contain all `considered_atts` and controlled attributes.
- `serialno` and `sample_geog` are dropped if present.
- All fields are cast to string. `condense_df` groups identical rows and adds a `count` weight.
- Pool multiplicity becomes sampling probability. Sampling is with replacement, so counts are weights rather than capacities.

### Attribute sets

- `considered_atts`: fields retained in final output and available throughout synthesis.
- `ordered_to_adjust_atts`: fields constrained to census marginals within each pass. Must be a subset of `considered_atts` and exist in marginals.
- Other considered fields are sampled and retained but never directly fitted.
- Defaults in `PopSynthesis/Methods/IPSF/const.py`:
  - household considered/controlled: `hhsize`, `hhinc`, `totalvehs`, `dwelltype`, `owndwell`;
  - person controlled: `age`, `sex`, `persinc`, `anywork`;
  - person considered additionally includes `nolicence`, `relationship`.

## Detailed SAA algorithm

### 1. Load and segment

`PopSynthesis/Methods/IPSF/run.py` reads marginals, drops `sample_geog`, sets the zone index, reads/cleans the pool, casts it to strings, condenses duplicates, calls `saa_run`, and writes the result.

`SAA.__init__` calls `process_raw_ipu_marg`, producing one zone-by-state DataFrame per considered attribute that exists in the census.

### 2. Choose an attribute order

`process_shuffle_order` uses these rules:

- First pass: ignore configured order and sort attributes from most marginal states to fewest.
- Middle passes: force a different original attribute first by `pass_index % number_of_attributes`, shuffle the rest, and with 25% probability move each `randomly_add_last` attribute to the end.
- Final allowed pass: put `last_adjustment_order` first, then randomly shuffle controlled attributes omitted from it.

The function mutates the list supplied to it. There is no run seed, so order and samples are not reproducible from configuration alone.

The first-pass sort considers every top-level attribute present in the marginal table, not just `ordered_to_adjust_atts`. Extra marginal attributes can therefore be introduced into the active order and later cause a lookup failure if they are not also in `considered_atts`. Current inputs should contain only the intended controls (plus zone and `sample_geog`).

### 3. Initialise on the first attribute

For the first attribute, `init_syn_pop_saa` loops over every state and zone:

- filter pool rows to that attribute state;
- draw exactly the census cell count using condensed `count` as probability, with replacement;
- attach the zone ID and expand to one row per agent.

The first attribute is exact by construction when its states exist in the pool. If a census state is absent, the code records a zero cell and samples from the entire pool; sampled records do not necessarily receive the missing state, so exactness is then not guaranteed.

### 4. Adjust later attributes per zone

For each later attribute:

1. `calculate_states_diff` counts current rows by zone/state and calculates `census - synthetic`, adding missing zones/states as zero.
2. For each zone, current rows are condensed.
3. `convert_to_ILP_inputs` creates a matrix where rows are combinations of previously adjusted attributes, columns are current-attribute states, and cells are counts. Pool-unavailable combinations become null and receive no adjustment variable.
4. `update_count_tables` solves an integer program with CBC:
   - cell adjustments are integers and cannot make counts negative;
   - each row's adjustments sum to zero, preserving previously adjusted combinations/totals;
   - each column targets its census difference;
   - positive/negative continuous slack permits best-effort solutions;
   - absolute changes are minimised; `spread_penalty` exists but SAA leaves it at zero.
5. The matrix is unpivoted and expanded to complete records by weighted conditional sampling from the pool.

The ILP remaining-error score is collected per zone but not exposed or used. Later attributes can retain error when pool support is inadequate, though total zone population stays fixed.

`operations/naive_zone_ad.py` is an older random pairwise replacement implementation. The active path uses `ILP_zone_adjustment`.

### 5. Loopback: retain safe records, target the residual

After a full pass, unless it is the final allowed pass:

1. `err_check_against_marg` randomly keeps `1 - extra_rm_frac` of the output. This deliberate removal creates more residual demand and can increase diversity across passes.
2. Convert remaining records to zonal marginals.
3. Compute `target - current` for controlled attributes.
4. `adjust_kept_rec_match_census` removes record combinations contributing to negative cells (oversupply), trying larger combinations of oversupplied attributes first.
5. Assert that the retained subset has no negative marginal residuals.
6. Append retained records to `chosen_syn`.
7. Replace the target with the non-negative residual marginal and synthesise only that missing population next.

The loop ends when residual agent count reaches zero or `max_run_time` passes are reached. On the final allowed pass, the entire generated residual population is appended without pruning, even if ILP residual error remains. Final output concatenates retained pieces and selects `considered_atts + [zone_id]`.

`err_rm`, returned by `saa_run`, stores the **number of agents targeted at the start of each pass**, not a statistical error.

## SAA features exposed now

| Feature / parameter | Behavior |
|---|---|
| Household/person mode | CLI chooses `hh_*` or `p_*` input keys from `level`; core logic is generic categorical synthesis. |
| `considered_atts` | Final output fields and pool information retained. |
| `ordered_to_adjust_atts` | Census controls; literal order is not used on first pass. |
| `max_run_time` | Maximum pass count, not time. Default 15 in `run_saa`, 30 in `saa_run`. |
| `extra_rm_frac` | Random removal fraction before pruning on non-final passes; asserted in `[0, 1]`. |
| `last_adjustment_order` | Prefix order forced on final pass; omitted controls follow randomly. |
| `randomly_add_last` | On 25% of middle passes, listed attributes move to the end. |
| `include_zero_cell_values` | Experimental support for states/combinations absent from pool. |
| `output_each_step` | Writes every within-pass attribute result and each retained subset. |
| `add_name_for_step_output` | Added to step filename to distinguish experiments. |
| `meta_output_dir` | Intermediate destination; CLI uses each rerun's `meta/`. |
| `reruns` | CLI repeats a YAML run independently and records elapsed seconds. |
| Condensed pool | Duplicate records are grouped and empirical frequency is kept as weight. |
| Soft feasibility | ILP slack prevents hard infeasibility when pool support is insufficient. |

Intermediate output naming:

```text
meta/
├── step_adjusted_<attribute-index>_<attribute>_<extra-name>_<pass>.csv
└── kept_syn_run_<pass>.csv
```

`extract_saa_runs_meta` reconstructs cumulative intermediate populations and calculates per-attribute RMSE. It expects files for all configured passes and may fail after early convergence or when a kept file is absent.

## Zero-cell behavior

A zero cell here is a census-required state or adjusted combination missing from the pool.

- Missing state labels are stored in a module-global dictionary in `operations/shared_vars.py`.
- It is never reset between runs in one Python process, so reruns can influence later zero-cell matching.
- With `include_zero_cell_values=False`, missing mass is redistributed over pool-supported rows using proportional allocation plus TRS integer rounding. If nothing joins, a random weighted pool sample is used.
- With `include_zero_cell_values=True`, missing values are directly appended only on the last attribute adjustment. Earlier adjustments print that only last-adjustment support exists and fall back to redistribution/cross-match logic.
- Treat the feature as experimental, not a guarantee that all structural zeros are represented consistently.

## Outputs and evaluation

Final SAA CSV characteristics:

- one row per synthetic agent;
- only `considered_atts` plus `zone_id`;
- no generated household/person ID;
- no household-person linkage when levels are synthesised independently;
- zone values cast to strings when pieces are retained/combined.

`PopSynthesis/analyse/extract_data/extract_comparisons.py` calculates RMSE against census marginal cells, JSD against seed distributions, percentage of seed combinations missing from output, negative-income counts, and optional SAA step-level RMSE. Benchmark primitives live in `PopSynthesis/Benchmark/CompareCensus/` and `CompareFullPop/`.

## Important limitations / failure points

1. **Only SAA/IPU are live CLI implementations.** `ipf`, `bn`, and `wgan` silently do nothing because their `NotImplementedError` is not raised.
2. **Custom zone names are not end-to-end.** Core functions expect `zone_id`.
3. **`sample_geog` is effectively required in SAA marginals.** The wrapper indexes the matching column before dropping it.
4. **No reproducibility control.** NumPy and Python `random` are used without a seed.
5. **Mutable list defaults.** Multiple SAA functions use `[]`, and ordering mutates lists in place.
6. **Global zero-cell state leaks across calls/reruns.**
7. **ILP fitting is best-effort.** Slack is always available, zone error is discarded, and the last pass has no final no-oversupply check.
8. **`max_run_time=0` is unsafe.** No pieces are generated, so final concatenation fails.
9. **Totals must be consistent.** Counts are inferred using division and compared with integer row counts; inconsistent attribute totals fail or create wrong targets.
10. **Direct-call directories are not created.** CLI creates rerun/meta directories; `run_saa` does not create `output_file.parent`.
11. **Standalone runners are stale.** They contain local Windows paths and sometimes inconsistent pool variables.
12. **Data processor is incomplete/non-portable.** It processes in its constructor, census methods are placeholders, constants use Windows paths, and `const_process.py` imports absent `Methods.connect_HH_PP` code.
13. **Tests are weak/environment-sensitive.** Test modules invoke tests at import time, mostly print instead of assert, and fixtures require Git LFS content.
14. **macOS ARM `uv sync` may fail with the current lock.** The locked CPU PyTorch build lacks a matching macOS ARM wheel for locally selected Python 3.13.
15. **Extra marginal attributes can break first-pass ordering.** The initial state-count sort is not filtered to configured controls.

## Repository layout and maturity

- `PopSynthesis/cli/`: supported orchestration surface.
- `PopSynthesis/Methods/IPSF/SAA/`: active SAA class, wrapper, and operations.
- `PopSynthesis/Methods/IPSF/utils/`: PuLP adjustment, condensation, pool, and residual helpers.
- `PopSynthesis/Methods/IPU/`: thin active wrapper around external `synthpop`.
- `PopSynthesis/Methods/IPF/`: standalone implementation/notebooks, not CLI-wired.
- `PopSynthesis/Methods/BN/`: several generations of experimental BN code.
- `PopSynthesis/Methods/CSP/`: experimental household/person relationship construction.
- `PopSynthesis/Methods/GAN/`, `RL/`, `naive/`: research prototypes.
- `PopSynthesis/DataProcessor/`: VISTA-oriented seed processing; incomplete and machine-specific.
- `PopSynthesis/Generator_data/`: one-off census/input preparation scripts and notebooks.
- `PopSynthesis/Benchmark/`: RMSE/JSD and full-population comparisons.
- `PopSynthesis/analyse/`: experiment extraction and visualisation.
- `IO/configs/`: current experiments; relative paths are resolved from this directory.
- `job_saa.slurm`: current cluster example for CLI SAA.
- `job_ipsf.script`: stale reference to missing `PopSynthesis/Methods/IPSF/CSP/run_by_layers.py`.

## Development guidance

Follow this call chain when changing SAA:

```text
cli/main.py
  -> Methods/IPSF/run.py
  -> SAA/operations/wrapper_saa_run.py
  -> SAA/SAA.py
  -> SAA/operations/general.py
  -> SAA/operations/ILP_zone_ad.py
  -> IPSF/utils/ILP_matrix_ad.py
  -> IPSF/utils/synthetic_checked_census.py  (between passes)
```

High-value improvements: validate YAML/inputs; raise for unsupported methods; thread zone naming throughout; replace mutable defaults; add a seed; make zero-cell state local; expose ILP errors and validate final marginals; add deterministic assertion-based tests and non-LFS fixtures; tolerate early convergence in analysis; isolate stale research drivers.

## Verification status for this documentation update

- All tracked Python files passed `python -m compileall -q PopSynthesis` in the inspected checkout.
- Tests were not executable: system Python lacked dependencies; `uv` selected Python 3.13 on macOS ARM and could not install locked `torch==2.7.1+cpu`; tracked test CSVs were LFS pointer text and `git-lfs` was unavailable.
- No claim here is based on a successful full synthesis run; execution was derived by tracing current source and configs.
