from __future__ import annotations

import json

import numpy as np
import pandas as pd

from PopSynthesis.analyse.paper_section_4_4 import (
    ATTRIBUTES,
    MAIN_ORDER,
    _select_order,
    aggregate_slack,
    build_rankings,
    conditional_rigidity,
    marginal_pressure,
    parse_slack_row,
    rigidity_relationship,
)


def test_conditional_rigidity_counts_supported_contexts() -> None:
    rows = [
        {"hhsize": "1", "dwelltype": "A", "owndwell": "O", "hhinc": "I", "totalvehs": "0", "w": 1},
        {"hhsize": "2", "dwelltype": "A", "owndwell": "O", "hhinc": "I", "totalvehs": "0", "w": 1},
        {"hhsize": "1", "dwelltype": "B", "owndwell": "O", "hhinc": "I", "totalvehs": "0", "w": 1},
    ]
    # Two contexts formed by the other attributes: A supports two hhsize states,
    # while B supports one, hence rigidity=1/2.
    assert conditional_rigidity(pd.DataFrame(rows), "hhsize", "w") == 0.5


def test_marginal_pressure_is_total_variation() -> None:
    pool = pd.DataFrame({"hhsize": ["1", "2"], "w": [3, 1]})
    census = pd.DataFrame({"1": [1], "2": [3]})
    census.columns = pd.MultiIndex.from_product([["hhsize"], census.columns])
    assert np.isclose(marginal_pressure(pool, census, "hhsize", "w"), 0.5)


def test_parse_slack_row_separates_positive_and_negative() -> None:
    row = pd.Series({"target_counts_json": json.dumps({"a": 4, "b": 2}),
                     "achieved_counts_json": json.dumps({"a": 3, "b": 3})})
    result = parse_slack_row(row)
    assert result["absolute_slack"] == 2
    assert result["positive_slack"] == 1
    assert result["negative_slack"] == 1


def test_competition_ranking_and_percentile_ties() -> None:
    frame = pd.DataFrame({
        "variant": ["v"] * 3,
        "census_macro_tvd": [0.0, 0.0, 1.0],
        "five_way_tvd": [0.2, 0.1, 0.3],
        "five_way_js_distance": [0.2, 0.1, 0.3],
        "mean_pairwise_tvd": [0.2, 0.1, 0.3],
        "total_stage1_slack": [0.0, 0.0, 2.0],
    })
    ranked = build_rankings(frame)
    assert ranked.census_macro_tvd_rank.tolist() == [1, 1, 3]
    assert ranked.census_macro_tvd_percentile.tolist() == [100.0, 100.0, 0.0]


def test_all_order_optimum_uses_production_order_as_labelled_representative() -> None:
    orders = [MAIN_ORDER, tuple(reversed(MAIN_ORDER))]
    frame = pd.DataFrame({
        "order_text": [" > ".join(order) for order in orders],
        "census_macro_tvd": [0.0, 0.0], "total_stage1_slack": [0.0, 0.0],
        "five_way_js_distance": [0.4, 0.3], "permutation_index": [1, 2],
    })
    selected = _select_order(frame, "minimum_macro_tvd")
    assert selected.order_text == " > ".join(MAIN_ORDER)


def test_position_aggregation_retains_contributing_counts() -> None:
    frame = pd.DataFrame({
        "variant": ["v", "v"], "attribute": ["hhsize", "hhsize"],
        "position": [1, 1], "total_slack": [0, 2], "run_name": ["a", "b"],
        "nonzero_zone_percent": [0, 10], "positive_slack": [0, 1], "negative_slack": [0, 1],
    })
    result = aggregate_slack(frame).iloc[0]
    assert result.mean_total_slack == 1
    assert result.contributing_permutations == 2


def test_early_placement_benefit_is_late_minus_early_total_slack() -> None:
    metrics = []
    slack = []
    for position, total in enumerate([1, 2, 3, 9, 10], 1):
        record = {"variant": "unrestricted", "total_stage1_slack": total}
        for attribute in ATTRIBUTES:
            record[f"{attribute}_position"] = position
            slack.append({"variant": "unrestricted", "attribute": attribute,
                          "position": position, "total_slack": total})
        metrics.append(record)
    scores = pd.DataFrame({"score_source": ["VISTA"] * 5, "attribute": ATTRIBUTES,
                           "rigidity": np.arange(5), "marginal_pressure": np.arange(5),
                           "heuristic_score": np.arange(5)})
    result = rigidity_relationship(pd.DataFrame(metrics), pd.DataFrame(slack), scores)
    # late mean 9.5 minus early mean 1.5
    observed = result.loc[result.variant == "unrestricted", "early_placement_benefit"]
    assert np.allclose(observed, 8.0)
