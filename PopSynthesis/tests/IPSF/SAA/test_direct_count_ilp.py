import json
import random
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.SAA.direct_count_saa import run_direct_count_pass
from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import (
    counts_to_marginals,
    population_size,
)
from PopSynthesis.Methods.IPSF.SAA.operations.direct_count_ilp import (
    prepare_direct_adjustment,
    solve_direct_adjustment,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run
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
        self.assertAlmostEqual(result.diagnostic["stage2_objective"], 2.0)
        self.assertEqual(
            result.diagnostic["ordinary_zero_cells_made_positive"], 1
        )

    def test_completely_unsupported_row_is_rejected(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["missing"], "count": [2]}
        )
        with self.assertRaisesRegex(ValueError, "completely unsupported"):
            prepare_direct_adjustment(
                partial, self.pool(), "b", {"m": 1, "n": 1}
            )

    def test_fractional_references_are_not_rounded(self):
        partial = pl.DataFrame(
            {"zone_id": ["101"], "a": ["x"], "count": [5]}
        )
        pool = pl.DataFrame(
            {"a": ["x", "x"], "b": ["m", "n"], "count": [1, 2]}
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
        self.assertEqual(result.diagnostic["stage2_objective_scale"], 3)
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

    def test_direct_pass_has_progressive_diagnostics(self):
        result = run_direct_count_pass(
            self.marginal(), self.pool(), ["a", "b"], pass_index=0
        )
        self.assertEqual(population_size(result.population), 5)
        self.assertEqual(result.population.columns, ["zone_id", "a", "b", "count"])
        self.assertEqual(len(result.adjustments), 2)
        self.assertEqual(result.adjustments[0]["stage1_status"], "NotRun")
        self.assertEqual(result.adjustments[1]["stage1_status"], "Optimal")

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


if __name__ == "__main__":
    unittest.main()
