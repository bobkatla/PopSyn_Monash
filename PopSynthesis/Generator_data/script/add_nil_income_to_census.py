"""Add the household-income residual as ``hhinc_Nil income``.

The input census is expected to use flattened ``attribute_state`` column
names, including ``tot_hh``. The source file is never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "IO" / "processed_data" / "census_POA.csv"
DEFAULT_OUTPUT = REPO_ROOT / "IO" / "processed_data" / "census_POA_with_nil_income.csv"
TOTAL_COLUMN = "tot_hh"
INCOME_PREFIX = "hhinc_"
NIL_INCOME_COLUMN = "hhinc_Nil income"


def add_nil_income(input_file: Path, output_file: Path) -> None:
    census = pd.read_csv(input_file)

    if TOTAL_COLUMN not in census.columns:
        raise ValueError(f"Missing required total column: {TOTAL_COLUMN!r}")
    if NIL_INCOME_COLUMN in census.columns:
        raise ValueError(
            f"{NIL_INCOME_COLUMN!r} already exists in {input_file}; "
            "refusing to calculate it twice."
        )

    income_columns = [
        column for column in census.columns if column.startswith(INCOME_PREFIX)
    ]
    if not income_columns:
        raise ValueError(f"No columns begin with {INCOME_PREFIX!r}")

    nil_income = census[TOTAL_COLUMN] - census[income_columns].sum(axis=1)
    if (nil_income < 0).any():
        invalid = census.loc[nil_income < 0, ["POA", TOTAL_COLUMN]].copy()
        invalid["hhinc_sum"] = census.loc[nil_income < 0, income_columns].sum(axis=1)
        raise ValueError(
            "Income marginals exceed tot_hh in one or more zones:\n"
            f"{invalid.head().to_string(index=False)}"
        )

    # Place the new flattened attribute/state column beside the other income cells.
    insert_at = census.columns.get_loc(TOTAL_COLUMN)
    census.insert(insert_at, NIL_INCOME_COLUMN, nil_income.astype(int))

    check_columns = [
        column for column in census.columns if column.startswith(INCOME_PREFIX)
    ]
    if not (census[check_columns].sum(axis=1) == census[TOTAL_COLUMN]).all():
        raise AssertionError("hhinc marginals do not equal tot_hh after adding Nil income")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    census.to_csv(output_file, index=False)

    print(f"Wrote {output_file}")
    print(f"Added {NIL_INCOME_COLUMN!r}; total residual: {int(nil_income.sum()):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add hhinc_Nil income as the residual to tot_hh."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_nil_income(args.input, args.output)
