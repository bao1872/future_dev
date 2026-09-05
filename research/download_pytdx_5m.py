#!/usr/bin/env python3
"""Download the PyTDX 5m L8 dataset for the quantile rebaseline.

Experiment 1 dataset:

    8 instruments
    5m L8 source bars
    one fixed common calendar window

    2025-01-01 -> last complete bar

The common calendar start is the point of this step. The previous
dataset took a fixed BAR COUNT per instrument, which produced
different calendar coverage per instrument and made cross-instrument
training pools incoherent. Here every instrument is pulled back to
the same calendar date and then cut to the same window.

Only 5m is downloaded. 15m is aggregated locally in the next step.

Outputs:

    research/exports/pytdx_5m/<INSTRUMENT>_5m.csv
    research/exports/pytdx_5m/download_manifest.json
    research/exports/pytdx_5m/download_validation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(
    __file__
).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from market_data.pytdx_source import (  # noqa: E402
    INSTRUMENTS,
    connect,
    download_5m_l8,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_5m"
)

# Fixed common calendar window.
CALENDAR_START = (
    "2025-01-01"
)

# Hard cap so a broken pagination cannot run away.
MAX_PAGES = 260

KEEP_COLUMNS = [
    "bar_start_time",
    "bar_end_time",
    "availability_time",
    "trading_day",
    "tdx_datetime_raw",
    "open",
    "high",
    "low",
    "close",
    "trade",
    "position",
]


def main() -> None:

    if OUT.exists() and any(
        OUT.iterdir()
    ):
        raise RuntimeError(
            f"{OUT} exists and is "
            "non-empty. Delete only for an "
            "intentional pre-commit rerun."
        )

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    cutoff = pd.Timestamp(
        CALENDAR_START
    )

    api = connect()

    manifest = {}
    validation = {}

    try:

        for (
            instrument
        ) in INSTRUMENTS:

            print(
                "=" * 64
            )

            print(
                f"{instrument} "
                f"{INSTRUMENTS[instrument]['code']}"
            )

            print(
                "=" * 64
            )

            df = download_5m_l8(
                instrument,
                api=api,
                max_pages=MAX_PAGES,
                not_before=cutoff,
            )

            if df.empty:
                raise RuntimeError(
                    f"{instrument}: no bars"
                )

            raw_rows = len(df)

            df = df[
                df[
                    "bar_start_time"
                ]
                >= cutoff
            ].copy()

            df = df[
                KEEP_COLUMNS
            ]

            path = (
                OUT
                / f"{instrument}_5m.csv"
            )

            df.to_csv(
                path,
                index=False,
            )

            errors = validate(
                df,
                instrument,
            )

            manifest[
                instrument
            ] = {
                "code": (
                    INSTRUMENTS[
                        instrument
                    ][
                        "code"
                    ]
                ),
                "market": (
                    INSTRUMENTS[
                        instrument
                    ][
                        "market"
                    ]
                ),
                "path": (
                    f"research/exports/"
                    f"pytdx_5m/"
                    f"{instrument}_5m.csv"
                ),
            }

            validation[
                instrument
            ] = {
                "status": (
                    "FAIL"
                    if errors
                    else "PASS"
                ),
                "rows_fetched": int(
                    raw_rows
                ),
                "rows_in_window": int(
                    len(df)
                ),
                "first_bar_start": str(
                    df[
                        "bar_start_time"
                    ].min()
                ),
                "last_bar_start": str(
                    df[
                        "bar_start_time"
                    ].max()
                ),
                "last_bar_end": str(
                    df[
                        "bar_end_time"
                    ].max()
                ),
                "errors": errors,
            }

            print(
                f"  rows in window: "
                f"{len(df)}"
            )

            print(
                f"  {df['bar_start_time'].min()}"
                f"  ->  "
                f"{df['bar_start_time'].max()}"
            )

            print(
                f"  status: "
                f"{validation[instrument]['status']}"
            )

            for e in errors:
                print(
                    f"    ! {e}"
                )

    finally:

        api.close()

    (
        OUT
        / "download_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    failed = [
        k
        for k, v in (
            validation.items()
        )
        if v["status"] != "PASS"
    ]

    starts = {
        k: v[
            "first_bar_start"
        ]
        for k, v in (
            validation.items()
        )
    }

    ends = {
        k: v[
            "last_bar_end"
        ]
        for k, v in (
            validation.items()
        )
    }

    summary = {
        "status": (
            "FAIL"
            if failed
            else "PASS"
        ),
        "calendar_start": (
            CALENDAR_START
        ),
        "instrument_count": len(
            INSTRUMENTS
        ),
        "failed_instruments": failed,
        "first_bar_start_by_instrument": (
            starts
        ),
        "last_bar_end_by_instrument": ends,
        "by_instrument": validation,
    }

    (
        OUT
        / "download_validation.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n"
        + "=" * 64
    )

    print(
        "COVERAGE"
    )

    print(
        "=" * 64
    )

    for k in INSTRUMENTS:
        v = validation[k]

        print(
            f"  {k:3s} "
            f"{v['rows_in_window']:7d}  "
            f"{v['first_bar_start']}  ->  "
            f"{v['last_bar_end']}"
        )

    if failed:
        raise RuntimeError(
            f"validation failed: "
            f"{failed}"
        )

    print(
        "\nPYTDX_5M_DOWNLOAD_PASS"
    )


def validate(
    df: pd.DataFrame,
    instrument: str,
) -> list[str]:

    errors: list[str] = []

    if df.empty:
        return [
            f"[{instrument}] empty"
        ]

    start = pd.to_datetime(
        df[
            "bar_start_time"
        ]
    )

    end = pd.to_datetime(
        df[
            "bar_end_time"
        ]
    )

    if not start.is_monotonic_increasing:
        errors.append(
            "bar_start_time not "
            "monotonic"
        )

    if start.duplicated().any():
        errors.append(
            "duplicate "
            "bar_start_time"
        )

    if (
        end - start
    ).dt.total_seconds().ne(
        300
    ).any():
        errors.append(
            "bar length is not "
            "exactly 5 minutes"
        )

    for col in (
        "open",
        "high",
        "low",
        "close",
    ):

        v = pd.to_numeric(
            df[col],
            errors=(
                "coerce"
            ),
        )

        if v.isna().any():
            errors.append(
                f"NaN in {col}"
            )

        if v.le(0).any():
            errors.append(
                f"non-positive "
                f"{col}"
            )

    hi = pd.to_numeric(
        df["high"],
        errors=(
            "coerce"
        ),
    )

    lo = pd.to_numeric(
        df["low"],
        errors=(
            "coerce"
        ),
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
        errors=(
            "coerce"
        ),
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
        errors=(
            "coerce"
        ),
    )

    if not (
        hi >= oc_max
    ).all():
        errors.append(
            "high < "
            "max(open, close)"
        )

    if not (
        lo <= oc_min
    ).all():
        errors.append(
            "low > "
            "min(open, close)"
        )

    trade = pd.to_numeric(
        df[
            "trade"
        ],
        errors=(
            "coerce"
        ),
    )

    position = pd.to_numeric(
        df[
            "position"
        ],
        errors=(
            "coerce"
        ),
    )

    if trade.lt(0).any():
        errors.append(
            "negative trade"
        )

    if position.lt(0).any():
        errors.append(
            "negative "
            "position"
        )

    now = pd.Timestamp.now()

    if (
        end
        > now
        + pd.Timedelta(
            minutes=5
        )
    ).any():
        errors.append(
            "future bar_end_time"
        )

    return errors


if __name__ == "__main__":
    main()
