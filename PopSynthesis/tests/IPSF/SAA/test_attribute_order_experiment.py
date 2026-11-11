from itertools import permutations
from pathlib import Path
import tempfile
import unittest

import pandas as pd
import yaml

from PopSynthesis.Generator_data.script.compare_direct_count_attribute_orders import (
    ATTRIBUTES,
    SUMMARY_METRICS,
    build_position_effects,
    canonical_digest,
    census_error_metrics,
    distribution_metrics,
    load_canonical_output,
    load_runs,
)
from PopSynthesis.Methods.IPSF.SAA.operations.wrapper_saa_run import (
    _validate_direct_mode_inputs,
)
from PopSynthesis.Methods.IPSF.run import _prepare_direct_pool


ROOT = Path(__file__).resolve().parents[4]
CONFIG_PATH = ROOT / "IO" / "configs" / "direct_count_relaxed_attribute_orders.yml"
UNRESTRICTED_CONFIG_PATH = (
    ROOT / "IO" / "configs" / "direct_count_unrestricted_attribute_orders.yml"
)
SEED_ONLY_CONFIG_PATH = (
    ROOT / "IO" / "configs" / "direct_count_seed_only_attribute_orders.yml"
)


class AttributeOrderConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))

    def test_complete_unique_permutation_configuration(self):
        expected_orders = list(permutations(ATTRIBUTES))
        actual_orders = [tuple(run["ordered_to_adjust_atts"]) for run in self.runs]
        self.assertEqual(len(self.runs), 120)
        self.assertEqual(actual_orders, expected_orders)
        self.assertEqual(len(set(actual_orders)), 120)

        fixed = {
            "hh_marg_file": "../processed_data/census_POA_hh_saa.csv",
            "hh_sample_file": "../processed_data/ori_sample_hh.csv",
            "hh_seed_data": "../processed_data/ori_sample_hh.csv",
            "pool_count_column": "_weight",
            "stochastic_mode": "count_native_ilp",
            "direct_solver_mode": "two_stage",
            "direct_adjustment_backend": "min_cost_flow",
            "zero_cell_policy": "relax_empty_rows_columns",
            "zone_workers": 4,
            "max_run_time": 1,
            "extra_rm_frac": 0,
            "output_format": "counts",
            "reruns": 1,
            "randomly_add_last": [],
        }
        names = []
        for index, (run, order) in enumerate(zip(self.runs, expected_orders), start=1):
            self.assertEqual(tuple(run["considered_atts"]), order)
            self.assertEqual(tuple(run["last_adjustment_order"]), order)
            self.assertEqual(len(order), len(set(order)))
            self.assertEqual(set(order), set(ATTRIBUTES))
            for key, value in fixed.items():
                self.assertEqual(run[key], value, f"run {index}: {key}")
            expected_name = f"relaxed_order_{index:03d}_{'_'.join(order)}"
            self.assertEqual(run["output_name"], expected_name)
            self.assertEqual(run["add_name_for_step_output"], expected_name)
            names.append(run["output_name"])
        self.assertEqual(len(set(names)), 120)
        self.assertEqual(load_runs(CONFIG_PATH), self.runs)

    def test_every_configuration_passes_direct_mode_validation(self):
        census_path = ROOT / "IO" / "processed_data" / "census_POA_hh_saa.csv"
        seed_path = ROOT / "IO" / "processed_data" / "ori_sample_hh.csv"
        census = pd.read_csv(census_path, header=[0, 1])
        zone_column = census.columns[census.columns.get_level_values(0) == "zone_id"][0]
        marginal = census.set_index(zone_column).iloc[[0]]
        seed = pd.read_csv(seed_path)
        pool = _prepare_direct_pool(seed, ATTRIBUTES, "_weight")

        for run in self.runs:
            with self.subTest(run=run["output_name"]):
                _validate_direct_mode_inputs(
                    marginal,
                    pool,
                    run["considered_atts"],
                    run["ordered_to_adjust_atts"],
                    run["last_adjustment_order"],
                    run["max_run_time"],
                    run["extra_rm_frac"],
                    run["include_zero_cell_values"],
                    run["randomly_add_last"],
                    "pool",
                )


class UnrestrictedAttributeOrderConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = yaml.safe_load(
            UNRESTRICTED_CONFIG_PATH.read_text(encoding="utf-8")
        )
        cls.census = pd.read_csv(
            ROOT / "IO" / "processed_data" / "census_POA_hh_saa.csv",
            header=[0, 1],
        )
        cls.seed = pd.read_csv(
            ROOT / "IO" / "processed_data" / "ori_sample_hh.csv"
        )

    def test_complete_unique_permutation_configuration(self):
        expected_orders = list(permutations(ATTRIBUTES))
        actual_orders = [tuple(run["ordered_to_adjust_atts"]) for run in self.runs]
        self.assertEqual(len(self.runs), 120)
        self.assertEqual(actual_orders, expected_orders)
        self.assertEqual(len(set(actual_orders)), 120)
        self.assertNotIn("Negative income", set(self.seed["hhinc"].astype(str)))

        fixed = {
            "hh_marg_file": "../processed_data/census_POA_hh_saa.csv",
            "hh_sample_file": "../processed_data/ori_sample_hh.csv",
            "hh_seed_data": "../processed_data/ori_sample_hh.csv",
            "pool_count_column": "_weight",
            "stochastic_mode": "count_native_ilp",
            "direct_solver_mode": "two_stage",
            "direct_adjustment_backend": "min_cost_flow",
            "zero_cell_policy": "unrestricted",
            "zone_workers": 4,
            "max_run_time": 1,
            "extra_rm_frac": 0,
            "output_format": "counts",
            "reruns": 1,
            "randomly_add_last": [],
        }
        names = []
        for index, (run, order) in enumerate(zip(self.runs, expected_orders), start=1):
            self.assertEqual(tuple(run["considered_atts"]), order)
            self.assertEqual(tuple(run["last_adjustment_order"]), order)
            self.assertEqual(set(order), set(ATTRIBUTES))
            for key, value in fixed.items():
                self.assertEqual(run[key], value, f"run {index}: {key}")
            expected_name = f"unrestricted_order_{index:03d}_{'_'.join(order)}"
            self.assertEqual(run["output_name"], expected_name)
            self.assertEqual(run["add_name_for_step_output"], expected_name)
            names.append(run["output_name"])
        self.assertEqual(len(set(names)), 120)
        self.assertEqual(load_runs(UNRESTRICTED_CONFIG_PATH), self.runs)

    def test_all_first_attributes_are_included_and_pass_direct_validation(self):
        first_attributes = {
            run["ordered_to_adjust_atts"][0] for run in self.runs
        }
        self.assertEqual(first_attributes, set(ATTRIBUTES))

        zone_column = self.census.columns[
            self.census.columns.get_level_values(0) == "zone_id"
        ][0]
        marginal = self.census.set_index(zone_column).iloc[[0]]
        pool = _prepare_direct_pool(self.seed, ATTRIBUTES, "_weight")
        for run in self.runs:
            with self.subTest(run=run["output_name"]):
                _validate_direct_mode_inputs(
                    marginal,
                    pool,
                    run["considered_atts"],
                    run["ordered_to_adjust_atts"],
                    run["last_adjustment_order"],
                    run["max_run_time"],
                    run["extra_rm_frac"],
                    run["include_zero_cell_values"],
                    run["randomly_add_last"],
                    "pool",
                )


class SeedOnlyAttributeOrderConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runs = yaml.safe_load(
            SEED_ONLY_CONFIG_PATH.read_text(encoding="utf-8")
        )
        cls.census = pd.read_csv(
            ROOT / "IO" / "processed_data" / "census_POA_hh_saa.csv",
            header=[0, 1],
        )
        cls.seed = pd.read_csv(
            ROOT / "IO" / "processed_data" / "ori_sample_hh.csv"
        )

    def test_all_and_only_pool_supported_first_attribute_orders(self):
        expected_orders = [
            order for order in permutations(ATTRIBUTES) if order[0] != "hhinc"
        ]
        actual_orders = [tuple(run["ordered_to_adjust_atts"]) for run in self.runs]
        self.assertEqual(len(self.runs), 96)
        self.assertEqual(actual_orders, expected_orders)
        self.assertEqual(len(set(actual_orders)), 96)

        fixed = {
            "hh_marg_file": "../processed_data/census_POA_hh_saa.csv",
            "hh_sample_file": "../processed_data/ori_sample_hh.csv",
            "hh_seed_data": "../processed_data/ori_sample_hh.csv",
            "pool_count_column": "_weight",
            "stochastic_mode": "count_native_ilp",
            "direct_solver_mode": "two_stage",
            "direct_adjustment_backend": "min_cost_flow",
            "zero_cell_policy": "seed_only",
            "zone_workers": 4,
            "max_run_time": 1,
            "extra_rm_frac": 0,
            "output_format": "counts",
            "reruns": 1,
            "randomly_add_last": [],
        }
        names = []
        for index, (run, order) in enumerate(zip(self.runs, expected_orders), start=1):
            self.assertNotEqual(order[0], "hhinc")
            self.assertEqual(tuple(run["considered_atts"]), order)
            self.assertEqual(tuple(run["last_adjustment_order"]), order)
            self.assertEqual(set(order), set(ATTRIBUTES))
            for key, value in fixed.items():
                self.assertEqual(run[key], value, f"run {index}: {key}")
            expected_name = f"seed_only_order_{index:03d}_{'_'.join(order)}"
            self.assertEqual(run["output_name"], expected_name)
            self.assertEqual(run["add_name_for_step_output"], expected_name)
            names.append(run["output_name"])
        self.assertEqual(len(set(names)), 96)
        self.assertEqual(load_runs(SEED_ONLY_CONFIG_PATH), self.runs)

    def test_first_attributes_have_seed_support_and_pass_direct_validation(self):
        first_attributes = {
            run["ordered_to_adjust_atts"][0] for run in self.runs
        }
        self.assertEqual(
            first_attributes,
            {"hhsize", "totalvehs", "dwelltype", "owndwell"},
        )
        for attribute in first_attributes:
            positive_census_states = {
                str(state)
                for state in self.census[attribute].columns
                if self.census[(attribute, state)].sum() > 0
            }
            seed_states = set(self.seed[attribute].astype(str))
            self.assertLessEqual(positive_census_states, seed_states)

        zone_column = self.census.columns[
            self.census.columns.get_level_values(0) == "zone_id"
        ][0]
        marginal = self.census.set_index(zone_column).iloc[[0]]
        pool = _prepare_direct_pool(self.seed, ATTRIBUTES, "_weight")
        for run in self.runs:
            with self.subTest(run=run["output_name"]):
                _validate_direct_mode_inputs(
                    marginal,
                    pool,
                    run["considered_atts"],
                    run["ordered_to_adjust_atts"],
                    run["last_adjustment_order"],
                    run["max_run_time"],
                    run["extra_rm_frac"],
                    run["include_zero_cell_values"],
                    run["randomly_add_last"],
                    "pool",
                )

    def test_position_effects_support_the_filtered_design(self):
        rows = []
        for index, run in enumerate(self.runs, start=1):
            order = run["ordered_to_adjust_atts"]
            row = {
                "duplicates_another_order": False,
                **{
                    f"{attribute}_position": order.index(attribute) + 1
                    for attribute in ATTRIBUTES
                },
                **{metric: float(index) for metric in SUMMARY_METRICS},
            }
            rows.append(row)
        effects = build_position_effects(pd.DataFrame(rows))
        self.assertEqual(len(effects), 24)
        self.assertTrue(
            effects[
                (effects["attribute"] == "hhinc")
                & (effects["position"] == 1)
            ].empty
        )
        self.assertEqual(set(effects["number_of_orders"]), {18, 24})


class AttributeOrderComparisonTests(unittest.TestCase):
    @staticmethod
    def population() -> pd.DataFrame:
        return pd.DataFrame(
            {
                "zone_id": ["1", "2"],
                "hhsize": ["1", "2"],
                "totalvehs": ["0", "1"],
                "dwelltype": ["Flat", "House"],
                "owndwell": ["Rent", "Own"],
                "hhinc": ["Low", "High"],
                "count": [1, 1],
            }
        )

    def test_canonicalization_detects_duplicate_outputs_across_column_orders(self):
        population = self.population()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            first_path = directory / "first.csv"
            second_path = directory / "second.csv"
            population.iloc[::-1].to_csv(first_path, index=False)
            population[
                ["hhinc", "count", "zone_id", "owndwell", "hhsize", "dwelltype", "totalvehs"]
            ].to_csv(second_path, index=False)
            first = load_canonical_output(first_path, {"1", "2"})
            second = load_canonical_output(second_path, {"1", "2"})
            self.assertTrue(first.equals(second))
            self.assertEqual(canonical_digest(first), canonical_digest(second))

    def test_metrics_and_position_aggregation(self):
        population = self.population()
        census_long = pd.DataFrame(
            {
                "zone_id": ["1", "2"],
                "attribute": ["hhsize", "hhsize"],
                "state": ["1", "2"],
                "census_count": [2, 1],
            }
        )
        errors = census_error_metrics(
            population[["zone_id", *ATTRIBUTES, "count"]], census_long
        )
        self.assertGreater(errors["census_l1_error"], 0)

        seed = population[ATTRIBUTES].copy()
        seed["_weight"] = [1.0, 3.0]
        distribution = distribution_metrics(seed, population, "_weight")
        self.assertAlmostEqual(distribution["full_joint_total_variation"], 0.25)
        self.assertEqual(distribution["synthetic_mass_outside_seed_support"], 0)
        self.assertEqual(distribution["seed_mass_absent_from_synthetic"], 0)

        rows = []
        for index, order in enumerate(permutations(ATTRIBUTES), start=1):
            row = {
                "permutation_index": index,
                "duplicates_another_order": index <= 2,
                **{f"{attribute}_position": order.index(attribute) + 1 for attribute in ATTRIBUTES},
            }
            row.update({metric: float(index) for metric in SUMMARY_METRICS})
            rows.append(row)
        effects = build_position_effects(pd.DataFrame(rows))
        self.assertEqual(len(effects), 25)
        self.assertTrue((effects["number_of_orders"] == 24).all())

    def test_output_validation_rejects_missing_zones_and_invalid_counts(self):
        population = self.population()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            missing_path = directory / "missing.csv"
            invalid_path = directory / "invalid.csv"
            population.iloc[[0]].to_csv(missing_path, index=False)
            invalid = population.copy()
            with self.assertRaisesRegex(ValueError, "zone set differs"):
                load_canonical_output(missing_path, {"1", "2"})
            invalid["count"] = invalid["count"].astype(float)
            invalid.loc[0, "count"] = 1.5
            invalid.to_csv(invalid_path, index=False)
            with self.assertRaisesRegex(ValueError, "positive integers"):
                load_canonical_output(invalid_path, {"1", "2"})


if __name__ == "__main__":
    unittest.main()
