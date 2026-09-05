#!/usr/bin/env python3
"""Minimal offline PyTDX market-data check.

Run after any PyTDX download / aggregation step. It only asserts
structural sanity on the normalized offline CSV; it is not a model
validation and it does not fetch anything.

Checks:

    timestamp monotonic
    no duplicate bar_start_time
    OHLC sane (high >= max(open, close), low <= min(open, close),
               all prices > 0)
    volume >= 0
    position >= 0
    no future event time relative to now

Usage:

    python scripts/check_tdx_data.py
    python scripts/check_tdx_data.py --path data/AG_5m.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DIR = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_5m"
)

REQUIRED_COLUMNS = (
    "bar_start_time",
    "bar_end_time",
    "availability_time",
    "open",
    "high",
    "low",
    "close",
    "trade",
    "position",
)


def check_frame(
    df: pd.DataFrame,
    *,
    label: str,
) -> list[str]:

    errors: list[str] = []

    if df.empty:
        return [
            f"[{label}] empty frame"
        ]

    missing = [
        c
        for c in REQUIRED_COLUMNS
        if c not in df.columns
    ]

    if missing:
        return [
            f"[{label}] missing "
            f"columns: {missing}"
        ]

    start = pd.to_datetime(
        df[
            "bar_start_time"
        ],
        errors="coerce",
    )

    end = pd.to_datetime(
        df[
            "bar_end_time"
        ],
        errors="coerce",
    )

    if start.isna().any():
        errors.append(
            f"[{label}] NaT in "
            "bar_start_time"
        )

    if not start.is_monotonic_increasing:
        errors.append(
            f"[{label}] "
            "bar_start_time not "
            "monotonic"
        )

    if start.duplicated().any():
        errors.append(
            f"[{label}] duplicate "
            "bar_start_time"
        )

    if (
        end - start
    ).dt.total_seconds().le(
        0
    ).any():
        errors.append(
            f"[{label}] "
            "bar_end_time <= "
            "bar_start_time"
        )

    prices = (
        "open",
        "high",
        "low",
        "close",
    )

    for col in prices:

        v = pd.to_numeric(
            df[col],
            errors="coerce",
        )

        if v.isna().any():
            errors.append(
                f"[{label}] NaN in "
                f"{col}"
            )

        if (v <= 0).any():
            errors.append(
                f"[{label}] "
                f"non-positive {col}"
            )

    hi = pd.to_numeric(
        df["high"],
        errors="coerce",
    )

    lo = pd.to_numeric(
        df["low"],
        errors="coerce",
    )

    oc_max = pd.to_numeric(
        df[
            [
                "open",
                "close",
            ]
        ].max(
            axis=1
        ),
        errors="coerce",
    )

    oc_min = pd.to_numeric(
        df[
            [
                "open",
                "close",
            ]
        ].min(
            axis=1
        ),
        errors="coerce",
    )

    if not (
        hi >= oc_max
    ).all():
        errors.append(
            f"[{label}] high < "
            "max(open, close)"
        )

    if not (
        lo <= oc_min
    ).all():
        errors.append(
            f"[{label}] low > "
            "min(open, close)"
        )

    if not (
        hi >= lo
    ).all():
        errors.append(
            f"[{label}] high < low"
        )

    trade = pd.to_numeric(
        df[
            "trade"
        ],
        errors="coerce",
    )

    position = (
        pd.to_numeric(
            df[
                "position"
            ],
            errors=(
                "coerce"
            ),
        )
    )

    if trade.isna().any():
        errors.append(
            f"[{label}] NaN in "
            "trade"
        )

    if trade.lt(0).any():
        errors.append(
            f"[{label}] negative "
            "trade"
        )

    if position.lt(0).any():
        errors.append(
            f"[{label}] negative "
            "position"
        )

    now = pd.Timestamp.now()

    future = (
        end
        > now
        + pd.Timedelta(
            minutes=5
        )
    )

    if future.any():
        errors.append(
            f"[{label}] "
            f"{int(future.sum())} "
            "rows with future "
            "bar_end_time"
        )

    return errors


def main() -> None:

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--path",
        type=str,
        default=None,
        help=(
            "single CSV file to check"
        ),
    )

    parser.add_argument(
        "--dir",
        type=str,
        default=str(
            DEFAULT_DIR
        ),
        help=(
            "directory of "
            "normalized CSV files"
        ),
    )

    args = (
        parser.parse_args()
    )

    if args.path:

        targets = [
            Path(
                args.path
            )
        ]

    else:

        directory = Path(
            args.dir
        )

        if (
            not directory.is_dir()
        ):
            print(
                "TDX DATA CHECK: FAIL"
            )

            print(
                f"  directory not "
                f"found: "
                f"{directory}"
            )

            sys.exit(1)

        targets = sorted(
            directory.glob(
                "*.csv"
            )
        )

        if not targets:

            print(
                "TDX DATA CHECK: FAIL"
            )

            print(
                f"  no CSV in "
                f"{directory}"
            )

            sys.exit(1)

    all_errors: dict[
        str, list[str]
    ] = {}

    for path in targets:

        df = pd.read_csv(
            path
        )

        errors = check_frame(
            df,
            label=path.stem,
        )

        if errors:
            all_errors[
                path.name
            ] = errors

    if all_errors:

        print(
            "TDX DATA CHECK: FAIL"
        )

        for name, errs in (
            all_errors.items()
        ):

            print(
                f"\n  {name}"
            )

            for e in errs:

                print(
                    f"    - {e}"
                )

        sys.exit(1)

    print(
        "TDX DATA CHECK: PASS"
    )

    print(
        f"  files checked: "
        f"{len(targets)}"
    )

    for path in targets:

        df = pd.read_csv(
            path
        )

        print(
            f"    "
            f"{path.name}: "
            f"{len(df)} rows"
        )


if __name__ == "__main__":
    main()
