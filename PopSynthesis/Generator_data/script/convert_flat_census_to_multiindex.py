"""Convert flattened ``attribute_state`` census columns to a two-row header."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = REPO_ROOT / "IO" / "processed_data" / "census_POA_with_nil_income.csv"

ATTRIBUTES = {
    "hh": ["hhsize", "dwelltype", "hhinc", "totalvehs", "owndwell"],
    "pp": ["age", "persinc", "sex", "anywork"],
}


def convert_flat_census(
    input_file: Path,
    output_file: Path,
    zone_column: str,
    level: str,
) -> None:
    census = pd.read_csv(input_file)
    attributes = ATTRIBUTES[level]

    if zone_column not in census.columns:
        raise ValueError(f"Missing zone column {zone_column!r} in {input_file}")

    columns = [("zone_id", "")]
    values = {("zone_id", ""): census[zone_column].astype(str)}

    for attribute in attributes:
        prefix = f"{attribute}_"
        state_columns = [column for column in census.columns if column.startswith(prefix)]
        if not state_columns:
            raise ValueError(f"No columns found for attribute {attribute!r}")

        for column in state_columns:
            state = column.removeprefix(prefix)
            multi_column = (attribute, state)
            columns.append(multi_column)
            values[multi_column] = census[column]

    converted = pd.DataFrame(values)
    converted.columns = pd.MultiIndex.from_tuples(columns)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    converted.to_csv(output_file, index=False)

    print(f"Wrote {output_file}")
    print(f"Rows: {len(converted):,}; columns: {len(converted.columns):,}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert flattened attribute_state census columns to two-level headers."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--zone-column", default="POA")
    parser.add_argument("--level", choices=ATTRIBUTES, default="hh")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert_flat_census(
        input_file=args.input,
        output_file=args.output,
        zone_column=args.zone_column,
        level=args.level,
    )
