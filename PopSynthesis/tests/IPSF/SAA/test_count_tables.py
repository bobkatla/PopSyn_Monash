import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import (
    aggregate_counts,
    counts_to_marginals,
    expand_counts,
    population_size,
    sample_without_replacement_counts,
    weighted_sample_counts,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import saa_run
from PopSynthesis.Methods.IPSF.SAA.operations.ILP_zone_ad import (
    sample_partial_to_complete_counts,
)
from PopSynthesis.Methods.IPSF.SAA.operations.shared_vars import (
    get_zero_cells_all,
    update_zero_cells,
    zero_cell_scope,
)
from PopSynthesis.Methods.IPSF.utils.condensed import condense_df
from PopSynthesis.Methods.IPSF.utils.synthetic_checked_census import (
    adjust_kept_rec_match_census,
)
from PopSynthesis.analyse.extract_data.extract_comparisons import (
    extract_saa_runs_meta,
)


class CountTableTests(unittest.TestCase):
    def setUp(self):
        random.seed(17)
        np.random.seed(17)

    def test_aggregate_expand_and_weighted_marginals(self):
        rows = pl.DataFrame(
            {
                "zone_id": ["1", "1", "1", "2"],
                "a": ["x", "x", "y", "x"],
                "count": [2, 3, 1, 4],
            }
        )
        counts = aggregate_counts(rows)
        self.assertEqual(population_size(counts), 10)
        expanded = expand_counts(counts)
        self.assertEqual(len(expanded), 10)
        expected = expanded.to_pandas().value_counts().sort_index()
        actual = (
            counts.to_pandas().set_index(["zone_id", "a"])["count"].sort_index()
        )
        pd.testing.assert_series_equal(actual, expected, check_names=False)
        marginal = counts_to_marginals(counts, ["a"])
        self.assertEqual(marginal.loc["1", ("a", "x")], 5)
        self.assertEqual(marginal.loc["2", ("a", "x")], 4)

    def test_sampling_counts_preserves_requested_sizes(self):
        pool = pl.DataFrame(
            {"a": ["x", "y", "z"], "count": [1, 2, 7]}
        )
        sampled = weighted_sample_counts(pool, 100)
        self.assertEqual(population_size(sampled), 100)
        retained = sample_without_replacement_counts(sampled, 41)
        self.assertEqual(population_size(retained), 41)
        joined = retained.join(sampled, on="a", suffix="_available")
        self.assertTrue(
            joined.select(pl.col("count") <= pl.col("count_available")).to_series().all()
        )

    def test_count_draws_have_expected_sampling_means(self):
        pool = pl.DataFrame({"a": ["x", "y"], "count": [3, 7]})
        sampled_x = []
        retained_x = []
        population = pl.DataFrame({"a": ["x", "y"], "count": [30, 70]})
        for _ in range(400):
            sampled = weighted_sample_counts(pool, 20)
            sampled_x.append(
                sampled.filter(pl.col("a") == "x")["count"].sum() or 0
            )
            retained = sample_without_replacement_counts(population, 20)
            retained_x.append(
                retained.filter(pl.col("a") == "x")["count"].sum() or 0
            )
        self.assertAlmostEqual(float(np.mean(sampled_x)), 6.0, delta=0.35)
        self.assertAlmostEqual(float(np.mean(retained_x)), 6.0, delta=0.35)

    def test_conditional_completion_and_zero_cell_paths(self):
        pool = pl.DataFrame(
            {
                "a": ["x", "x", "y"],
                "b": ["m", "n", "m"],
                "count": [3, 1, 2],
            }
        )
        partial = pl.DataFrame({"a": ["x"], "count": [25]})
        completed = sample_partial_to_complete_counts(partial, pool)
        self.assertEqual(population_size(completed), 25)
        self.assertEqual(set(completed["a"]), {"x"})

        unsupported = pl.DataFrame(
            {"a": ["x", "missing"], "count": [2, 5]}
        )
        redistributed = sample_partial_to_complete_counts(unsupported, pool)
        self.assertEqual(population_size(redistributed), 7)
        self.assertEqual(set(redistributed["a"]), {"x"})

        complete_zero = pl.DataFrame(
            {"a": ["missing"], "b": ["zero"], "count": [4]}
        )
        included = sample_partial_to_complete_counts(
            complete_zero, pool, include_value=True
        )
        self.assertEqual(population_size(included), 4)
        self.assertEqual(included.row(0)[:2], ("missing", "zero"))

    def test_zero_cell_scope_is_run_local(self):
        with zero_cell_scope():
            update_zero_cells("a", "missing")
            self.assertEqual(get_zero_cells_all(), {"a": {"missing"}})
        self.assertEqual(get_zero_cells_all(), {})

    def test_count_residual_pruning_matches_row_implementation(self):
        from PopSynthesis.Methods.IPSF.SAA.operations.count_tables import (
            prune_oversupplied_counts,
        )

        full = pd.DataFrame(
            [
                {"zone_id": "1", "a": a, "b": b}
                for a, b, count in [
                    ("x", "m", 4),
                    ("x", "n", 2),
                    ("y", "m", 3),
                    ("y", "n", 1),
                ]
                for _ in range(count)
            ]
        )
        diff = pd.DataFrame(
            [[-2, 2, -2, 2]],
            index=pd.Index(["1"], name="zone_id"),
            columns=pd.MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("b", "m"), ("b", "n")]
            ),
        )
        legacy = adjust_kept_rec_match_census(full.copy(), diff.copy())
        count_result = prune_oversupplied_counts(condense_df(full), diff.copy())
        expected = legacy.value_counts().sort_index()
        actual = (
            count_result.to_pandas()
            .set_index(["zone_id", "a", "b"])["count"]
            .sort_index()
        )
        pd.testing.assert_series_equal(actual, expected, check_names=False)


class CountNativeIntegrationTests(unittest.TestCase):
    @staticmethod
    def inputs():
        columns = pd.MultiIndex.from_tuples(
            [("a", "x"), ("a", "y"), ("b", "m"), ("b", "n")]
        )
        marginal = pd.DataFrame(
            [[6, 4, 5, 5], [4, 6, 5, 5]],
            index=pd.Index([1, 2], name="zone_id"),
            columns=columns,
        )
        pool = condense_df(
            pl.DataFrame(
                {
                    "a": ["x", "x", "y", "y", "x"],
                    "b": ["m", "n", "m", "n", "m"],
                }
            )
        )
        return marginal, pool

    def run_mode(self, mode, output_each_step=False, output_dir=None):
        random.seed(29)
        np.random.seed(29)
        marginal, pool = self.inputs()
        return saa_run(
            marginal,
            pool,
            considered_atts=["a", "b"],
            ordered_to_adjust_atts=["a", "b"],
            last_adjustment_order=["a", "b"],
            max_run_time=3,
            extra_rm_frac=0.3,
            output_each_step=output_each_step,
            meta_output_dir=output_dir,
            stochastic_mode=mode,
        )

    def test_count_native_is_repeatable_and_matches_population_size(self):
        first, first_targets = self.run_mode("count_native")
        second, second_targets = self.run_mode("count_native")
        self.assertEqual(first_targets, second_targets)
        self.assertTrue(first.equals(second))
        self.assertEqual(len(first), 20)
        self.assertEqual(first.schema["zone_id"], pl.String)

    def test_run_does_not_mutate_caller_attribute_order(self):
        marginal, pool = self.inputs()
        order = ["a", "b"]
        saa_run(
            marginal,
            pool,
            considered_atts=["a", "b"],
            ordered_to_adjust_atts=order,
            max_run_time=2,
            stochastic_mode="count_native",
        )
        self.assertEqual(order, ["a", "b"])

    def test_legacy_exact_repeats_for_the_same_ordered_count_pool(self):
        marginal, pool = self.inputs()
        outputs = []
        for _ in range(2):
            random.seed(31)
            np.random.seed(31)
            result, targets = saa_run(
                marginal.copy(),
                pool,
                considered_atts=["a", "b"],
                ordered_to_adjust_atts=["a", "b"],
                last_adjustment_order=["a", "b"],
                max_run_time=2,
                extra_rm_frac=0.3,
                stochastic_mode="legacy_exact",
            )
            outputs.append(
                (condense_df(result).sort(["zone_id", "a", "b"]), targets)
            )
        self.assertEqual(outputs[0][1], outputs[1][1])
        self.assertTrue(outputs[0][0].equals(outputs[1][0]))

    def test_both_backends_preserve_final_marginals_for_supported_pool(self):
        for mode in ("count_native", "legacy_exact"):
            random.seed(29)
            np.random.seed(29)
            target, pool = self.inputs()
            result, _ = saa_run(
                target.copy(),
                pool,
                considered_atts=["a", "b"],
                ordered_to_adjust_atts=["a", "b"],
                last_adjustment_order=["a", "b"],
                max_run_time=1,
                stochastic_mode=mode,
            )
            counts = condense_df(result)
            marginal = counts_to_marginals(counts, ["a", "b"])
            target.index = target.index.astype(str)
            marginal = marginal.reindex(
                index=target.index, columns=target.columns, fill_value=0
            )
            pd.testing.assert_frame_equal(
                marginal.astype(int), target.astype(int), check_names=False
            )

    def test_compact_metrics_and_cumulative_loopback_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result, targets = self.run_mode(
                "count_native", True, Path(temp_dir)
            )
            metrics = pd.read_csv(Path(temp_dir) / "adjustment_metrics.csv")
            self.assertEqual(
                set(metrics["pass"]), set(range(len(targets)))
            )
            self.assertEqual(
                set(metrics["adjusted_attribute"]), {"a", "b"}
            )
            for pass_index in range(len(targets)):
                snapshot = pl.read_csv(
                    Path(temp_dir) / f"loopback_population_{pass_index}.csv"
                )
                self.assertEqual(len(snapshot), len(result))
            self.assertFalse(list(Path(temp_dir).glob("step_adjusted_*.csv")))
            target, _ = self.inputs()
            extracted = extract_saa_runs_meta(
                Path(temp_dir), len(targets), ["a", "b"], target
            )
            self.assertEqual(len(extracted), len(targets) * 2)
            self.assertIn(("mean", ""), extracted.columns)


if __name__ == "__main__":
    unittest.main()
