import json
import random
import subprocess
import sys
import tempfile
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import yaml

from PopSynthesis.Methods.IPSF.SAA.direct_count_saa import run_direct_count_pass
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import (
    counts_to_marginals,
    population_size,
)
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_ilp import (
    prepare_direct_adjustment,
    solve_direct_adjustment,
)
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_flow import (
    solve_direct_adjustment_flow,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run
from PopSynthesis.Methods.IPSF.run import (
    _prepare_direct_pool,
    run_saa as run_saa_files,
)
from PopSynthesis.Methods.IPSF.const import count_field
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
from PopSynthesis.analyse.extract_data.extract_comparisons import (
    extract_saa_runs_meta,
)


class DirectCountFixture(unittest.TestCase):
    @staticmethod
    def marginal():
        return pd.DataFrame(
            [[3, 2, 2, 3]],
            index=pd.Index([101], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "m"), ("b", "n")]
            ),
        )

    @staticmethod
    def pool():
        return pl.DataFrame(
            {
                "a": ["x", "y"],
                "b": ["m", "n"],
                count_field: [3, 2],
            }
        )


class DirectCountILPTests(DirectCountFixture):
    def test_all_zero_reference_is_unrestricted_and_backend_equivalent(self):
        partial = pl.DataFrame(
            {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
        )
        prepared = prepare_direct_adjustment(
            partial,
            None,
            "b",
            {"m": 2, "n": 3},
            direct_reference_mode="all_zero",
        )

        zero_row = (Fraction(0), Fraction(0))
        self.assertEqual(prepared.pool_counts, (zero_row, zero_row))
        self.assertEqual(prepared.references, (zero_row, zero_row))
        self.assertEqual(
            prepared.permitted_cells, ((True, True), (True, True))
        )
        self.assertEqual(prepared.direct_reference_mode, "all_zero")

        cbc = solve_direct_adjustment(prepared, solver_mode="two_stage")
        flow = solve_direct_adjustment_flow(prepared, solver_mode="two_stage")
        for result in (cbc, flow):
            self.assertEqual(result.diagnostic["stage1_objective"], 0)
            self.assertEqual(result.diagnostic["marginal_l1_error"], 0)
            self.assertEqual(
                result.diagnostic["stage2_objective_fraction"], "5/1"
            )
            self.assertTrue(result.diagnostic["row_totals_preserved"])
            self.assertEqual(
                result.diagnostic["direct_reference_mode"], "all_zero"
            )

    def test_all_zero_direct_pass_requires_no_pool(self):
        result = run_direct_count_pass(
            self.marginal(),
            None,
            ["a", "b"],
            pass_index=0,
            solver_mode="two_stage",
            adjustment_backend="min_cost_flow",
            direct_reference_mode="all_zero",
        )

        actual = counts_to_marginals(result.population, ["a", "b"])
        expected = self.marginal().copy()
        expected.index = expected.index.astype(str)
        actual = actual.reindex(
            index=expected.index, columns=expected.columns, fill_value=0
        )
        pd.testing.assert_frame_equal(
            actual.astype(int), expected.astype(int), check_names=False
        )
        self.assertEqual(result.adjustments[1]["stage2_objective_fraction"], "5/1")

    def test_all_zero_reference_supports_existing_loopback(self):
        state = random.getstate()
        random.seed(7123)
        try:
            result, targets = saa_run(
                self.marginal(),
                None,
                considered_atts=["a", "b"],
                ordered_to_adjust_atts=["a", "b"],
                last_adjustment_order=["a", "b"],
                max_run_time=2,
                extra_rm_frac=0.4,
                stochastic_mode="count_native_ilp",
                output_format="counts",
                direct_solver_mode="two_stage",
                direct_adjustment_backend="min_cost_flow",
                direct_reference_mode="all_zero",
            )
        finally:
            random.setstate(state)

        self.assertGreaterEqual(len(targets), 1)
        actual = counts_to_marginals(result, ["a", "b"])
        expected = self.marginal().copy()
        expected.index = expected.index.astype(str)
        actual = actual.reindex(
            index=expected.index, columns=expected.columns, fill_value=0
        )
        pd.testing.assert_frame_equal(
            actual.astype(int), expected.astype(int), check_names=False
        )

    def test_all_zero_rejects_pool_and_restricted_support(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["x"], "count": [2]}
        )
        with self.assertRaisesRegex(ValueError, "must not be supplied"):
            prepare_direct_adjustment(
                partial,
                self.pool(),
                "b",
                {"m": 1, "n": 1},
                direct_reference_mode="all_zero",
            )
        with self.assertRaisesRegex(ValueError, "requires zero_cell_policy"):
            prepare_direct_adjustment(
                partial,
                None,
                "b",
                {"m": 1, "n": 1},
                zero_cell_policy="seed_only",
                direct_reference_mode="all_zero",
            )

    def test_hand_calculated_dense_adjustment(self):
        partial = pl.DataFrame(
            {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
        )
        prepared = prepare_direct_adjustment(
            partial, self.pool(), "b", {"m": 2, "n": 3}
        )
        self.assertEqual(prepared.pool_counts, ((3, 0), (0, 2)))
        self.assertEqual(
            prepared.references,
            ((3, 0), (0, 2)),
        )

        result = solve_direct_adjustment(prepared)
        expected = pl.DataFrame(
            {
                "zone_id": ["101", "101", "101"],
                "a": ["x", "x", "y"],
                "b": ["m", "n", "n"],
                "count": [2, 1, 2],
            }
        )
        self.assertTrue(
            result.counts.sort(["a", "b"]).equals(expected.sort(["a", "b"]))
        )
        self.assertEqual(result.diagnostic["stage1_objective"], 0)
        self.assertEqual(
            result.diagnostic["stage1_status"], "SkippedExactConstraints"
        )
        self.assertAlmostEqual(result.diagnostic["stage2_objective"], 2.0)
        self.assertEqual(
            result.diagnostic["ordinary_zero_cells_made_positive"], 1
        )

        two_stage = solve_direct_adjustment(prepared, solver_mode="two_stage")
        self.assertEqual(two_stage.diagnostic["stage1_status"], "Optimal")
        self.assertEqual(two_stage.diagnostic["stage1_objective"], 0)
        self.assertEqual(two_stage.diagnostic["stage2_slack_objective"], 0)
        self.assertEqual(two_stage.diagnostic["solver_mode"], "two_stage")
        self.assertTrue(two_stage.counts.equals(result.counts))
        for stage in ("stage1", "stage2"):
            self.assertGreater(two_stage.diagnostic[f"{stage}_solve_seconds"], 0)
            self.assertGreater(two_stage.diagnostic[f"{stage}_variable_count"], 0)
            self.assertGreater(two_stage.diagnostic[f"{stage}_constraint_count"], 0)

    def test_two_stage_retains_positive_optimal_slack(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["x"], "count": [5]}
        )
        pool = pl.DataFrame(
            {
                "a": ["x", "x"],
                "b": ["m", "n"],
                "count": [1, 4],
            }
        )
        prepared = prepare_direct_adjustment(
            partial, pool, "b", {"m": 2, "n": 2, "zero": 0}
        )

        result = solve_direct_adjustment(prepared, solver_mode="two_stage")
        flow = solve_direct_adjustment_flow(prepared, solver_mode="two_stage")

        self.assertEqual(result.diagnostic["stage1_status"], "Optimal")
        self.assertEqual(result.diagnostic["stage1_objective"], 1)
        self.assertEqual(result.diagnostic["stage2_slack_objective"], 1)
        self.assertEqual(result.diagnostic["marginal_l1_error"], 1)
        self.assertTrue(result.diagnostic["row_totals_preserved"])
        self.assertEqual(result.diagnostic["eliminated_zero_target_columns"], 0)
        self.assertEqual(result.counts["count"].sum(), 5)
        self.assertEqual(flow.diagnostic["stage1_status"], "Optimal")
        self.assertEqual(flow.diagnostic["stage1_objective"], 1)
        self.assertEqual(flow.diagnostic["stage2_slack_objective"], 1)
        self.assertEqual(flow.diagnostic["marginal_l1_error"], 1)
        self.assertEqual(
            flow.diagnostic["stage2_objective_fraction"],
            result.diagnostic["stage2_objective_fraction"],
        )

    def test_two_stage_direct_pass_allows_inconsistent_census_totals(self):
        marginal = pd.DataFrame(
            [[3, 2, 2, 2]],
            index=pd.Index([101], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "m"), ("b", "n")]
            ),
        )
        result = run_direct_count_pass(
            marginal,
            self.pool(),
            ["a", "b"],
            pass_index=0,
            solver_mode="two_stage",
            adjustment_backend="cbc",
        )

        self.assertEqual(population_size(result.population), 5)
        self.assertEqual(result.adjustments[1]["stage1_objective"], 1)
        self.assertEqual(result.adjustments[1]["marginal_l1_error"], 1)

        final, _ = saa_run(
            marginal,
            self.pool(),
            considered_atts=["a", "b"],
            ordered_to_adjust_atts=["a", "b"],
            last_adjustment_order=["a", "b"],
            max_run_time=1,
            extra_rm_frac=0,
            stochastic_mode="count_native_ilp",
            output_format="counts",
            direct_solver_mode="two_stage",
            direct_adjustment_backend="cbc",
        )
        self.assertEqual(population_size(final), 5)

    def test_two_stage_flow_matches_unique_cbc_solution(self):
        cbc = run_direct_count_pass(
            self.marginal(),
            self.pool(),
            ["a", "b"],
            pass_index=0,
            solver_mode="two_stage",
            adjustment_backend="cbc",
        )
        flow = run_direct_count_pass(
            self.marginal(),
            self.pool(),
            ["a", "b"],
            pass_index=0,
            solver_mode="two_stage",
            adjustment_backend="min_cost_flow",
        )
        self.assertTrue(cbc.population.equals(flow.population))
        self.assertEqual(flow.adjustments[1]["stage1_objective"], 0)

    def test_two_stage_flow_matches_cbc_for_deficit_and_zero_targets(self):
        partial = pl.DataFrame(
            {
                "zone_id": ["101", "101"],
                "a": ["x", "missing"],
                "count": [3, 1],
            }
        )
        pool = pl.DataFrame(
            {
                "a": ["x", "x"],
                "b": ["m", "n"],
                "count": [0.1, 0.2],
            }
        )
        prepared = prepare_direct_adjustment(
            partial, pool, "b", {"m": 2, "n": 3, "zero": 0}
        )

        cbc = solve_direct_adjustment(prepared, solver_mode="two_stage")
        flow = solve_direct_adjustment_flow(prepared, solver_mode="two_stage")

        self.assertEqual(cbc.diagnostic["stage1_objective"], 1)
        self.assertEqual(flow.diagnostic["stage1_objective"], 1)
        self.assertEqual(flow.diagnostic["marginal_l1_error"], 1)
        self.assertEqual(flow.diagnostic["unsupported_row_count"], 1)
        self.assertEqual(flow.diagnostic["eliminated_zero_target_columns"], 0)
        self.assertEqual(
            flow.diagnostic["stage2_objective_fraction"],
            cbc.diagnostic["stage2_objective_fraction"],
        )

    def test_completely_unsupported_row_uses_zero_references(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["missing"], "count": [2]}
        )
        prepared = prepare_direct_adjustment(
            partial, self.pool(), "b", {"m": 1, "n": 1}
        )
        self.assertEqual(prepared.references, ((Fraction(0), Fraction(0)),))
        self.assertEqual(prepared.unsupported_rows, (("missing",),))

        result = solve_direct_adjustment(prepared)
        self.assertEqual(result.diagnostic["unsupported_row_count"], 1)
        self.assertEqual(result.diagnostic["stage2_objective"], 2)
        self.assertEqual(result.counts["count"].sum(), 2)
        self.assertEqual(
            set(result.counts["b"].to_list()), {"m", "n"}
        )

    def test_fractional_references_are_not_rounded(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["x"], "count": [5]}
        )
        pool = pl.DataFrame(
            {"a": ["x", "x"], "b": ["m", "n"], "count": [0.1, 0.2]}
        )
        prepared = prepare_direct_adjustment(
            partial, pool, "b", {"m": 2, "n": 3}
        )
        self.assertEqual(
            prepared.references,
            ((Fraction(5, 3), Fraction(10, 3)),),
        )
        result = solve_direct_adjustment(prepared)
        self.assertEqual(result.diagnostic["stage2_status"], "Optimal")
        self.assertIsNone(result.diagnostic["stage2_objective_scale"])
        self.assertIn(result.diagnostic["stage2_solver_gap"], (None, 0))
        self.assertAlmostEqual(result.diagnostic["stage2_objective"], 2 / 3)

    def test_completely_unsupported_column_can_be_populated(self):
        partial = pl.DataFrame(
            {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
        )
        pool = pl.DataFrame(
            {"a": ["x", "y"], "b": ["m", "m"], "count": [1, 1]}
        )
        prepared = prepare_direct_adjustment(
            partial, pool, "b", {"m": 2, "n": 3}
        )
        self.assertEqual(prepared.unsupported_columns, ("n",))
        result = solve_direct_adjustment(prepared)
        achieved_n = result.counts.filter(pl.col("b") == "n")["count"].sum()
        self.assertEqual(achieved_n, 3)
        self.assertEqual(result.diagnostic["unsupported_column_count"], 1)

    def test_flow_matches_cbc_exact_objectives(self):
        cases = [
            (
                pl.DataFrame(
                    {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
                ),
                self.pool(),
                {"m": 2, "n": 3},
            ),
            (
                pl.DataFrame({"zone_id": ["101"], "a": ["x"], "count": [5]}),
                pl.DataFrame(
                    {"a": ["x", "x"], "b": ["m", "n"], "count": [0.1, 0.2]}
                ),
                {"m": 2, "n": 3},
            ),
            (
                pl.DataFrame(
                    {"zone_id": ["101"], "a": ["missing"], "count": [2]}
                ),
                self.pool(),
                {"m": 1, "n": 1},
            ),
            (
                pl.DataFrame(
                    {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
                ),
                self.pool(),
                {"m": 5, "n": 0},
            ),
        ]
        for partial, pool, targets in cases:
            with self.subTest(targets=targets, partial=partial.to_dicts()):
                prepared = prepare_direct_adjustment(partial, pool, "b", targets)
                cbc = solve_direct_adjustment(prepared)
                flow = solve_direct_adjustment_flow(prepared)
                self.assertEqual(
                    cbc.diagnostic["stage2_objective_fraction"],
                    flow.diagnostic["stage2_objective_fraction"],
                )
                self.assertEqual(flow.diagnostic["marginal_l1_error"], 0)
                self.assertTrue(flow.diagnostic["row_totals_preserved"])

    def test_pool_conditionals_are_cached_between_passes(self):
        marginal = pd.DataFrame(
            [[2, 2, 2, 2, 2, 2]],
            index=pd.Index([101], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [
                    ("a", "x"), ("a", "y"),
                    ("b", "m"), ("b", "n"),
                    ("c", "u"), ("c", "v"),
                ]
            ),
        )
        pool = pl.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["m", "n", "m", "n"],
                "c": ["u", "v", "v", "u"],
                "count": [1, 1, 1, 1],
            }
        )
        cache = {}
        first = run_direct_count_pass(
            marginal,
            pool,
            ["a", "b", "c"],
            pass_index=0,
            adjustment_backend="min_cost_flow",
            conditional_cache=cache,
        )
        cache_ids = {key: id(value) for key, value in cache.items()}
        second = run_direct_count_pass(
            marginal,
            pool,
            ["a", "b", "c"],
            pass_index=0,
            adjustment_backend="min_cost_flow",
            conditional_cache=cache,
        )
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache_ids, {key: id(value) for key, value in cache.items()})
        self.assertTrue(first.population.equals(second.population))

    def test_direct_pass_has_progressive_diagnostics(self):
        result = run_direct_count_pass(
            self.marginal(), self.pool(), ["a", "b"], pass_index=0
        )
        self.assertEqual(population_size(result.population), 5)
        self.assertEqual(result.population.columns, ["zone_id", "a", "b", "count"])
        self.assertEqual(len(result.adjustments), 2)
        self.assertEqual(result.adjustments[0]["stage1_status"], "NotRun")
        self.assertEqual(
            result.adjustments[1]["stage1_status"], "SkippedExactConstraints"
        )
        self.assertEqual(result.adjustments[1]["solver_mode"], "single_stage")

    def test_three_attribute_direct_propagation(self):
        marginal = pd.DataFrame(
            [[2, 2, 2, 2, 2, 2]],
            index=pd.Index([101], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [
                    ("a", "x"), ("a", "y"),
                    ("b", "m"), ("b", "n"),
                    ("c", "u"), ("c", "v"),
                ]
            ),
        )
        pool = pl.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["m", "n", "m", "n"],
                "c": ["u", "v", "v", "u"],
                "count": [1, 1, 1, 1],
            }
        )
        result = run_direct_count_pass(
            marginal, pool, ["a", "b", "c"], pass_index=0
        )
        self.assertEqual(result.population.columns, ["zone_id", "a", "b", "c", "count"])
        self.assertEqual(population_size(result.population), 4)
        self.assertEqual(len(result.adjustments), 3)


class ZeroCellPolicyTests(DirectCountFixture):
    def test_four_policy_experiment_config_has_one_rerun_each(self):
        config_path = (
            Path(__file__).resolve().parents[4]
            / "IO"
            / "configs"
            / "direct_count_zero_cell_policies.yml"
        )
        runs = yaml.safe_load(config_path.read_text(encoding="utf-8"))

        self.assertEqual(len(runs), 4)
        self.assertEqual(
            [run["zero_cell_policy"] for run in runs],
            [
                "unrestricted",
                "seed_only",
                "relax_empty_rows_columns",
                "relax_empty_rows_global_columns",
            ],
        )
        self.assertTrue(all(run["reruns"] == 1 for run in runs))
        self.assertEqual(len({run["output_name"] for run in runs}), 4)

    def test_unrestricted_default_matches_explicit_policy(self):
        partial = pl.DataFrame(
            {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
        )
        default = prepare_direct_adjustment(
            partial, self.pool(), "b", {"m": 2, "n": 3}
        )
        explicit = prepare_direct_adjustment(
            partial,
            self.pool(),
            "b",
            {"m": 2, "n": 3},
            zero_cell_policy="unrestricted",
        )
        self.assertEqual(default.permitted_cells, explicit.permitted_cells)
        default_result = solve_direct_adjustment_flow(
            default, solver_mode="two_stage"
        )
        explicit_result = solve_direct_adjustment_flow(
            explicit, solver_mode="two_stage"
        )
        self.assertTrue(default_result.counts.equals(explicit_result.counts))

    def test_seed_only_prohibits_zero_cells_and_accepts_fractional_support(self):
        partial = pl.DataFrame(
            {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
        )
        pool = pl.DataFrame(
            {"a": ["x", "y"], "b": ["m", "n"], "count": [0.1, 0.2]}
        )
        prepared = prepare_direct_adjustment(
            partial,
            pool,
            "b",
            {"m": 2, "n": 3},
            zero_cell_policy="seed_only",
        )
        self.assertEqual(prepared.permitted_cells, ((True, False), (False, True)))

        cbc = solve_direct_adjustment(prepared, solver_mode="two_stage")
        flow = solve_direct_adjustment_flow(prepared, solver_mode="two_stage")
        self.assertEqual(cbc.diagnostic["stage1_objective"], 2)
        self.assertEqual(flow.diagnostic["stage1_objective"], 2)
        self.assertEqual(flow.diagnostic["seed_zero_cells_made_positive"], 0)
        self.assertEqual(flow.diagnostic["permitted_cell_count"], 2)
        self.assertEqual(flow.diagnostic["prohibited_cell_count"], 2)
        self.assertEqual(
            cbc.diagnostic["stage2_objective_fraction"],
            flow.diagnostic["stage2_objective_fraction"],
        )
        produced = set(zip(flow.counts["a"], flow.counts["b"]))
        self.assertEqual(produced, {("x", "m"), ("y", "n")})

    def test_seed_only_rejects_unsupported_positive_row(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["missing"], "count": [2]}
        )
        with self.assertRaisesRegex(ValueError, "positive row without pool support"):
            prepare_direct_adjustment(
                partial,
                self.pool(),
                "b",
                {"m": 1, "n": 1},
                zero_cell_policy="seed_only",
            )

    def test_both_relax_policies_open_unsupported_row(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["missing"], "count": [2]}
        )
        for policy in (
            "relax_empty_rows_columns",
            "relax_empty_rows_global_columns",
        ):
            with self.subTest(policy=policy):
                prepared = prepare_direct_adjustment(
                    partial,
                    self.pool(),
                    "b",
                    {"m": 1, "n": 1},
                    zero_cell_policy=policy,
                )
                self.assertEqual(prepared.permitted_cells, ((True, True),))
                self.assertEqual(prepared.relaxed_cell_count, 2)
                result = solve_direct_adjustment_flow(
                    prepared, solver_mode="two_stage"
                )
                self.assertEqual(result.diagnostic["stage1_objective"], 0)
                self.assertEqual(
                    result.diagnostic["seed_zero_cells_made_positive"], 2
                )

    def test_both_relax_policies_open_globally_unsupported_column(self):
        partial = pl.DataFrame(
            {"zone_id": ["101", "101"], "a": ["x", "y"], "count": [3, 2]}
        )
        pool = pl.DataFrame(
            {"a": ["x", "y"], "b": ["m", "m"], "count": [1, 1]}
        )
        for policy in (
            "relax_empty_rows_columns",
            "relax_empty_rows_global_columns",
        ):
            with self.subTest(policy=policy):
                prepared = prepare_direct_adjustment(
                    partial,
                    pool,
                    "b",
                    {"m": 2, "n": 3, "zero": 0},
                    zero_cell_policy=policy,
                )
                self.assertEqual(
                    prepared.permitted_cells,
                    ((True, True, False), (True, True, False)),
                )
                self.assertEqual(prepared.relaxed_cell_count, 2)
                result = solve_direct_adjustment_flow(
                    prepared, solver_mode="two_stage"
                )
                self.assertEqual(result.diagnostic["stage1_objective"], 0)
                self.assertEqual(
                    result.counts.filter(pl.col("b") == "zero").height, 0
                )

    def test_relax_column_scope_active_versus_global_pool(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["x"], "count": [3]}
        )
        pool = pl.DataFrame(
            {
                "a": ["x", "y"],
                "b": ["m", "n"],
                "count": [1, 1],
            }
        )
        active = prepare_direct_adjustment(
            partial,
            pool,
            "b",
            {"m": 1, "n": 2},
            zero_cell_policy="relax_empty_rows_columns",
        )
        global_pool = prepare_direct_adjustment(
            partial,
            pool,
            "b",
            {"m": 1, "n": 2},
            zero_cell_policy="relax_empty_rows_global_columns",
        )

        self.assertEqual(active.permitted_cells, ((True, True),))
        self.assertEqual(active.unsupported_columns, ("n",))
        self.assertEqual(active.zero_cell_column_scope, "active_rows")
        self.assertEqual(global_pool.permitted_cells, ((True, False),))
        self.assertEqual(global_pool.unsupported_columns, ())
        self.assertEqual(global_pool.zero_cell_column_scope, "global_pool")

        active_cbc = solve_direct_adjustment(active, solver_mode="two_stage")
        active_flow = solve_direct_adjustment_flow(active, solver_mode="two_stage")
        global_cbc = solve_direct_adjustment(global_pool, solver_mode="two_stage")
        global_flow = solve_direct_adjustment_flow(
            global_pool, solver_mode="two_stage"
        )
        self.assertEqual(active_cbc.diagnostic["stage1_objective"], 0)
        self.assertEqual(active_flow.diagnostic["stage1_objective"], 0)
        self.assertEqual(global_cbc.diagnostic["stage1_objective"], 4)
        self.assertEqual(global_flow.diagnostic["stage1_objective"], 4)
        self.assertEqual(
            active_cbc.diagnostic["stage2_objective_fraction"],
            active_flow.diagnostic["stage2_objective_fraction"],
        )
        self.assertEqual(
            global_cbc.diagnostic["stage2_objective_fraction"],
            global_flow.diagnostic["stage2_objective_fraction"],
        )

    def test_missing_first_attribute_support_and_zero_target_exemption(self):
        marginal = pd.DataFrame(
            [[1, 0, 1, 0]],
            index=pd.Index([101], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "missing"), ("a", "x"), ("b", "m"), ("b", "n")]
            ),
        )
        with self.assertRaisesRegex(
            ValueError, "zone=101.*attribute='a'.*category='missing'.*target=1"
        ):
            run_direct_count_pass(
                marginal,
                self.pool(),
                ["a", "b"],
                pass_index=0,
                solver_mode="two_stage",
                adjustment_backend="min_cost_flow",
                zero_cell_policy="seed_only",
            )

        marginal.loc[101, ("a", "missing")] = 0
        marginal.loc[101, ("a", "x")] = 1
        result = run_direct_count_pass(
            marginal,
            self.pool(),
            ["a", "b"],
            pass_index=0,
            solver_mode="two_stage",
            adjustment_backend="min_cost_flow",
            zero_cell_policy="seed_only",
        )
        self.assertEqual(population_size(result.population), 1)

    def test_hhinc_first_permissive_policies_initialize_unsupported_state(self):
        marginal = pd.DataFrame(
            [[2, 1, 1, 2]],
            index=pd.Index([101], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [
                    ("hhinc", "Negative income"),
                    ("hhinc", "Nil income"),
                    ("hhsize", "1"),
                    ("hhsize", "2"),
                ]
            ),
        )
        pool = pl.DataFrame(
            {
                "hhinc": ["Nil income", "Nil income"],
                "hhsize": ["1", "2"],
                "count": [1, 1],
            }
        )

        for policy in ("relax_empty_rows_columns", "unrestricted"):
            with self.subTest(policy=policy):
                result = run_direct_count_pass(
                    marginal,
                    pool,
                    ["hhinc", "hhsize"],
                    pass_index=0,
                    solver_mode="two_stage",
                    adjustment_backend="min_cost_flow",
                    zero_cell_policy=policy,
                )
                negative_income = result.population.filter(
                    pl.col("hhinc") == "Negative income"
                )["count"].sum()
                self.assertEqual(negative_income, 2)
                self.assertEqual(
                    result.adjustments[0]["unsupported_column_count"], 1
                )

        for policy in (
            "seed_only",
            "relax_empty_rows_global_columns",
        ):
            with self.subTest(policy=policy), self.assertRaisesRegex(
                ValueError,
                "attribute='hhinc'.*category='Negative income'.*target=2",
            ):
                run_direct_count_pass(
                    marginal,
                    pool,
                    ["hhinc", "hhsize"],
                    pass_index=0,
                    solver_mode="two_stage",
                    adjustment_backend="min_cost_flow",
                    zero_cell_policy=policy,
                )

    def test_restricted_policy_requires_two_stage(self):
        with self.assertRaisesRegex(ValueError, "require.*two-stage"):
            run_direct_count_pass(
                self.marginal(),
                self.pool(),
                ["a", "b"],
                pass_index=0,
                zero_cell_policy="seed_only",
            )

    def test_seed_only_preserves_full_support_across_multiple_attributes(self):
        marginal = pd.DataFrame(
            [[2, 2, 2, 2, 2, 2]],
            index=pd.Index([101], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [
                    ("a", "x"), ("a", "y"),
                    ("b", "m"), ("b", "n"),
                    ("c", "u"), ("c", "v"),
                ]
            ),
        )
        pool = pl.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["m", "n", "m", "n"],
                "c": ["u", "v", "v", "u"],
                "count": [0.1, 0.2, 0.3, 0.4],
            }
        )
        result = run_direct_count_pass(
            marginal,
            pool,
            ["a", "b", "c"],
            pass_index=0,
            solver_mode="two_stage",
            adjustment_backend="min_cost_flow",
            zero_cell_policy="seed_only",
        )
        supported = set(zip(pool["a"], pool["b"], pool["c"]))
        generated = set(
            zip(
                result.population["a"],
                result.population["b"],
                result.population["c"],
            )
        )
        self.assertTrue(generated <= supported)
        self.assertTrue(result.population["count"].dtype.is_integer())
        self.assertTrue((result.population["count"] > 0).all())


class DirectCountLoopbackTests(DirectCountFixture):
    def setUp(self):
        random.seed(73)
        np.random.seed(73)

    def run_mode(self, **kwargs):
        return saa_run(
            self.marginal(),
            self.pool(),
            considered_atts=["a", "b"],
            ordered_to_adjust_atts=["a", "b"],
            last_adjustment_order=["a", "b"],
            stochastic_mode="count_native_ilp",
            **kwargs,
        )

    def test_zero_removal_terminates_after_exact_first_pass(self):
        result, targets = self.run_mode(max_run_time=4, extra_rm_frac=0)
        self.assertEqual(targets, [5])
        self.assertEqual(len(result), 5)

    def test_stochastic_loopbacks_conserve_final_marginals(self):
        result, targets = self.run_mode(max_run_time=3, extra_rm_frac=0.4)
        self.assertGreaterEqual(len(targets), 2)
        counts = condense_df(result)
        actual = counts_to_marginals(counts, ["a", "b"])
        expected = self.marginal()
        expected.index = expected.index.astype(str)
        actual = actual.reindex(
            index=expected.index, columns=expected.columns, fill_value=0
        )
        pd.testing.assert_frame_equal(
            actual.astype(int), expected.astype(int), check_names=False
        )

    def test_full_removal_regenerates_until_final_pass(self):
        result, targets = self.run_mode(max_run_time=3, extra_rm_frac=1)
        self.assertEqual(targets, [5, 5, 5])
        self.assertEqual(len(result), 5)

    def test_seed_only_support_is_reapplied_across_loopbacks(self):
        pool = pl.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["m", "n", "m", "n"],
                "count": [1, 1, 1, 1],
            }
        )
        result, targets = saa_run(
            self.marginal(),
            pool,
            considered_atts=["a", "b"],
            ordered_to_adjust_atts=["a", "b"],
            last_adjustment_order=["a", "b"],
            stochastic_mode="count_native_ilp",
            direct_solver_mode="two_stage",
            direct_adjustment_backend="min_cost_flow",
            zero_cell_policy="seed_only",
            max_run_time=3,
            extra_rm_frac=0.4,
        )
        self.assertGreaterEqual(len(targets), 2)
        supported = set(zip(pool["a"], pool["b"]))
        self.assertTrue(set(zip(result["a"], result["b"])))
        self.assertTrue(set(zip(result["a"], result["b"])) <= supported)

    def test_compact_direct_diagnostics_cover_passes(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            _, targets = self.run_mode(
                max_run_time=3,
                extra_rm_frac=0.4,
                output_each_step=True,
                meta_output_dir=Path(temp_dir),
            )
            adjustments = pd.read_csv(
                Path(temp_dir) / "direct_count_adjustments.csv"
            )
            loopbacks = pd.read_csv(
                Path(temp_dir) / "direct_count_loopbacks.csv"
            )
            self.assertEqual(len(loopbacks), len(targets))
            self.assertEqual(set(adjustments["mode"]), {"count_native_ilp"})
            self.assertFalse(list(Path(temp_dir).glob("loopback_population_*.csv")))
            for raw_order in loopbacks["pass_order_json"]:
                self.assertEqual(set(json.loads(raw_order)), {"a", "b"})
            extracted = extract_saa_runs_meta(
                Path(temp_dir),
                len(targets),
                ["a", "b"],
                self.marginal(),
                stochastic_mode="count_native_ilp",
            )
            self.assertEqual(len(extracted), len(adjustments))
            self.assertIn(("residual_population", ""), extracted.columns)

    def test_unadjusted_output_attributes_are_rejected(self):
        marginal = self.marginal()
        pool = self.pool().with_columns(pl.lit("extra").alias("c"))
        with self.assertRaisesRegex(ValueError, "must exactly equal"):
            saa_run(
                marginal,
                pool,
                considered_atts=["a", "b", "c"],
                ordered_to_adjust_atts=["a", "b"],
                max_run_time=1,
                stochastic_mode="count_native_ilp",
            )

    def test_direct_solver_is_repeatable_in_fresh_processes(self):
        script = """
import json
import pandas as pd
import polars as pl
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run
m = pd.DataFrame([[3,2,2,3]], index=pd.Index([101], name='zone_id'), columns=pd.MultiIndex.from_tuples([('a','x'),('a','y'),('b','m'),('b','n')]))
p = pl.DataFrame({'a':['x','y'], 'b':['m','n'], 'count':[3,2]})
r, _ = saa_run(m, p, considered_atts=['a','b'], ordered_to_adjust_atts=['a','b'], last_adjustment_order=['a','b'], max_run_time=1, stochastic_mode='count_native_ilp')
print(json.dumps(r.to_dicts(), sort_keys=True))
"""
        outputs = []
        for _ in range(2):
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=Path(__file__).parents[4],
                check=True,
                text=True,
                capture_output=True,
            )
            outputs.append(completed.stdout.strip())
        self.assertEqual(outputs[0], outputs[1])


class DirectCountFileRunTests(unittest.TestCase):
    def test_weighted_pool_aggregates_exact_decimal_weights(self):
        raw_pool = pd.DataFrame(
            {"a": ["x", "x"], "b": ["m", "m"], "_weight": [0.1, 0.2]}
        )
        prepared = _prepare_direct_pool(raw_pool, ["a", "b"], "_weight")
        self.assertEqual(prepared.height, 1)
        self.assertEqual(prepared["count"][0], Decimal("0.3"))

        for invalid in (0, -1, float("inf"), float("nan")):
            invalid_pool = raw_pool.copy()
            invalid_pool.loc[0, "_weight"] = invalid
            with self.assertRaisesRegex(ValueError, "finite|positive"):
                _prepare_direct_pool(invalid_pool, ["a", "b"], "_weight")

    def test_fractional_weighted_pool_batches_zones_and_writes_counts(self):
        marginal = pd.DataFrame(
            [[3, 2, 2, 3], [1, 3, 2, 2]],
            index=pd.Index([101, 102], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "m"), ("b", "negative")]
            ),
        )
        raw_pool = pd.DataFrame(
            {
                "a": ["x", "x", "y"],
                "b": ["m", "m", "m"],
                "_weight": [0.1, 0.2, 0.4],
                "ignored_id": [1, 2, 3],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marginal_path = directory / "marginal.csv"
            pool_path = directory / "pool.csv"
            output_paths = [
                directory / f"run_{index}" / "counts.csv" for index in range(2)
            ]
            marginal.reset_index().to_csv(marginal_path, index=False)
            raw_pool.to_csv(pool_path, index=False)

            for index, output_path in enumerate(output_paths):
                run_saa_files(
                    marg_file=marginal_path,
                    pool_file=pool_path,
                    zone_field="zone_id",
                    output_file=output_path,
                    considered_atts=["a", "b"],
                    ordered_to_adjust_atts=["a", "b"],
                    last_adjustment_order=["a", "b"],
                    max_run_time=1,
                    extra_rm_frac=0,
                    stochastic_mode="count_native_ilp",
                    pool_count_column="_weight",
                    output_format="counts",
                    direct_adjustment_backend="min_cost_flow",
                    zone_workers=index + 1,
                )

            first = pl.read_csv(output_paths[0])
            second = pl.read_csv(output_paths[1])
            self.assertTrue(first.equals(second))
            self.assertEqual(first.columns, ["zone_id", "a", "b", "count"])
            self.assertEqual(first["zone_id"].n_unique(), 2)
            self.assertTrue(first["count"].dtype.is_integer())
            self.assertTrue((first["count"] > 0).all())
            self.assertFalse(first.select(["zone_id", "a", "b"]).is_duplicated().any())

            actual = counts_to_marginals(
                first.with_columns(pl.col("zone_id").cast(pl.String)), ["a", "b"]
            )
            expected = marginal.copy()
            expected.index = expected.index.astype(str)
            actual = actual.reindex(
                index=expected.index, columns=expected.columns, fill_value=0
            )
            pd.testing.assert_frame_equal(
                actual.astype(int), expected.astype(int), check_names=False
            )

    def test_two_stage_parallel_diagnostics_are_combined_and_resumable(self):
        marginal = pd.DataFrame(
            [[3, 2, 2, 3], [1, 3, 2, 2]],
            index=pd.Index([101, 102], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "m"), ("b", "n")]
            ),
        )
        raw_pool = pd.DataFrame(
            {
                "a": ["x", "x", "y", "y"],
                "b": ["m", "n", "m", "n"],
                "_weight": [0.1, 0.2, 0.3, 0.4],
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marginal_path = directory / "marginal.csv"
            pool_path = directory / "pool.csv"
            output_path = directory / "run" / "counts.csv"
            marginal.reset_index().to_csv(marginal_path, index=False)
            raw_pool.to_csv(pool_path, index=False)

            def execute():
                run_saa_files(
                    marg_file=marginal_path,
                    pool_file=pool_path,
                    zone_field="zone_id",
                    output_file=output_path,
                    considered_atts=["a", "b"],
                    ordered_to_adjust_atts=["a", "b"],
                    last_adjustment_order=["a", "b"],
                    max_run_time=1,
                    extra_rm_frac=0,
                    stochastic_mode="count_native_ilp",
                    pool_count_column="_weight",
                    output_format="counts",
                    direct_solver_mode="two_stage",
                    direct_adjustment_backend="min_cost_flow",
                    zone_workers=2,
                )

            execute()
            combined = pd.read_csv(
                output_path.parent / "meta" / "direct_count_adjustments.csv",
                dtype={"zone": str},
            )
            self.assertEqual(set(combined["zone"]), {"101", "102"})
            self.assertEqual(len(combined), 4)
            solved = combined[combined["adjustment_index"] == 1]
            self.assertTrue((solved["stage1_status"] == "Optimal").all())
            self.assertTrue((solved["stage1_objective"] == 0).all())
            for column in [
                "stage1_solve_seconds",
                "stage2_solve_seconds",
                "stage1_variable_count",
                "stage2_variable_count",
                "stage1_constraint_count",
                "stage2_constraint_count",
            ]:
                self.assertIn(column, combined.columns)

            missing_sidecar = (
                output_path.parent
                / "zone_artifacts"
                / "diagnostics"
                / "zone_0001"
                / "direct_count_adjustments.csv"
            )
            missing_sidecar.unlink()
            execute()
            self.assertTrue(missing_sidecar.exists())

            manifest_path = output_path.parent / "zone_artifacts" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["settings"]["zero_cell_column_scope"],
                "not_applicable",
            )
            manifest["settings"].pop("zero_cell_column_scope")
            manifest_path.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "different inputs or settings"):
                execute()

    def test_all_zero_file_run_is_parallel_deterministic_and_has_null_pool_hash(self):
        marginal = pd.DataFrame(
            [[3, 2, 2, 3], [1, 3, 2, 2]],
            index=pd.Index([101, 102], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "m"), ("b", "n")]
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marginal_path = directory / "marginal.csv"
            marginal.reset_index().to_csv(marginal_path, index=False)
            outputs = [
                directory / "sequential" / "counts.csv",
                directory / "parallel" / "counts.csv",
            ]

            for workers, output_path in zip((1, 2), outputs):
                run_saa_files(
                    marg_file=marginal_path,
                    pool_file=None,
                    zone_field="zone_id",
                    output_file=output_path,
                    considered_atts=["a", "b"],
                    ordered_to_adjust_atts=["a", "b"],
                    last_adjustment_order=["a", "b"],
                    max_run_time=1,
                    extra_rm_frac=0,
                    stochastic_mode="count_native_ilp",
                    output_format="counts",
                    direct_solver_mode="two_stage",
                    direct_adjustment_backend="min_cost_flow",
                    direct_reference_mode="all_zero",
                    zone_workers=workers,
                )

            first = pl.read_csv(outputs[0])
            second = pl.read_csv(outputs[1])
            self.assertTrue(first.equals(second))
            for output_path in outputs:
                manifest = json.loads(
                    (output_path.parent / "zone_artifacts" / "manifest.json").read_text()
                )
                self.assertIsNone(manifest["pool_sha256"])
                self.assertEqual(
                    manifest["settings"]["direct_reference_mode"], "all_zero"
                )
                diagnostics = pd.read_csv(
                    output_path.parent / "meta" / "direct_count_adjustments.csv"
                )
                self.assertTrue(
                    (diagnostics["direct_reference_mode"] == "all_zero").all()
                )

    def test_failed_zone_does_not_publish_partial_output(self):
        marginal = pd.DataFrame(
            [[2, 0, 2, 0], [0, 2, 1, 0]],
            index=pd.Index([101, 102], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "m"), ("b", "n")]
            ),
        )
        raw_pool = pd.DataFrame({"a": ["x"], "b": ["m"], "_weight": [1.5]})
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            marginal_path = directory / "marginal.csv"
            pool_path = directory / "pool.csv"
            output_path = directory / "counts.csv"
            marginal.reset_index().to_csv(marginal_path, index=False)
            raw_pool.to_csv(pool_path, index=False)

            with self.assertRaisesRegex(ValueError, "inconsistent"):
                run_saa_files(
                    marg_file=marginal_path,
                    pool_file=pool_path,
                    zone_field="zone_id",
                    output_file=output_path,
                    considered_atts=["a", "b"],
                    ordered_to_adjust_atts=["a", "b"],
                    last_adjustment_order=["a", "b"],
                    max_run_time=1,
                    extra_rm_frac=0,
                    stochastic_mode="count_native_ilp",
                    pool_count_column="_weight",
                    output_format="counts",
                )
            self.assertFalse(output_path.exists())


if __name__ == "__main__":
    unittest.main()
