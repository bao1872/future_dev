#!/usr/bin/env python3
"""Download the 16-instrument PyTDX 5m L8 dataset for V3R.

Separate from the 8-instrument `pytdx_5m` dataset so the two
research datasets keep distinct provenance.

Rules, identical to the 8-instrument download:

    fixed common calendar window
    only 5m is downloaded
    the incomplete trailing trading day is dropped

Outputs:

    research/exports/v3r_5m/<ROOT>_5m.csv
    research/exports/v3r_5m/download_validation.json
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
    drop_incomplete_tail,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "v3r_5m"
)

CALENDAR_START = (
    "2025-01-01"
)

# The experiment endpoint belongs to the experiment definition, not
# to the data source. It is enforced explicitly here.
#
# This was previously only implicit: every instrument happened to end
# at 2026-09-04 15:00 because the old drop_incomplete_tail() deleted
# the whole trading day containing any future bar. That looked like a
# designed common window but was a side effect, and it silently moved
# whenever the download clock or the server state changed.
CALENDAR_END = (
    "2026-09-04 15:00:00"
)

MAX_PAGES = 300

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
            "not monotonic"
        )

    if start.duplicated().any():
        errors.append(
            "duplicate bar_start_time"
        )

    if (
        end - start
    ).dt.total_seconds().ne(
        300
    ).any():
        errors.append(
            "bar length != 5m"
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
                f"NaN {col}"
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
            "high < max(o,c)"
        )

    if not (
        lo <= oc_min
    ).all():
        errors.append(
            "low > min(o,c)"
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

    end_cutoff = pd.Timestamp(
        CALENDAR_END
    )

    now = pd.Timestamp.now()

    api = connect()

    validation = {}

    try:

        for (
            instrument
        ) in INSTRUMENTS:

            print(
                f"{instrument} "
                f"{INSTRUMENTS[instrument]['code']}"
                f" ...",
                flush=True,
            )

            df = download_5m_l8(
                instrument,
                api=api,
                max_pages=(
                    MAX_PAGES
                ),
                not_before=(
                    cutoff
                ),
            )

            if df.empty:
                raise RuntimeError(
                    f"{instrument}: "
                    "no bars"
                )

            fetched = int(
                len(
                    df
                )
            )

            df = drop_incomplete_tail(
                df,
                now=now,
            )

            after_tail = int(
                len(
                    df
                )
            )

            df = df[
                df[
                    "bar_start_time"
                ]
                >= cutoff
            ].copy()

            # Experiment endpoint. Without this the window is
            # decided by how much the vendor happens to serve.
            after_start = int(
                len(
                    df
                )
            )

            df = df[
                df[
                    "availability_time"
                ]
                <= end_cutoff
            ].copy()

            df = df[
                KEEP_COLUMNS
            ]

            df.to_csv(
                OUT
                / f"{instrument}_5m.csv",
                index=False,
            )

            errors = validate(
                df,
                instrument,
            )

            validation[
                instrument
            ] = {
                "status": (
                    "FAIL"
                    if errors
                    else "PASS"
                ),
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
                "rows_fetched": (
                    fetched
                ),
                "rows_after_tail_drop": (
                    after_tail
                ),
                "rows_after_start_cut": (
                    after_start
                ),
                "rows_in_window": int(
                    len(
                        df
                    )
                ),
                "rows_dropped_by_calendar_end": (
                    after_start
                    - int(
                        len(
                            df
                        )
                    )
                ),
                "first_bar_start": str(
                    df[
                        "bar_start_time"
                    ].min()
                ),
                "last_bar_end": str(
                    df[
                        "bar_end_time"
                    ].max()
                ),
                "errors": errors,
            }

            print(
                f"    rows="
                f"{len(df)} "
                f"{df['bar_start_time'].min()}"
                f" -> "
                f"{df['bar_end_time'].max()} "
                f"{validation[instrument]['status']}"
            )

    finally:

        api.close()

    failed = [
        k
        for k, v in (
            validation.items()
        )
        if v["status"] != "PASS"
    ]

    summary = {
        "status": (
            "FAIL"
            if failed
            else "PASS"
        ),
        "calendar_start": (
            CALENDAR_START
        ),
        "calendar_end": (
            CALENDAR_END
        ),
        "instrument_count": len(
            INSTRUMENTS
        ),
        "failed_instruments": failed,
        "first_bar_start_by_instrument": {
            k: v[
                "first_bar_start"
            ]
            for k, v in (
                validation.items()
            )
        },
        "last_bar_end_by_instrument": {
            k: v[
                "last_bar_end"
            ]
            for k, v in (
                validation.items()
            )
        },
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
        + "=" * 60
    )

    print(
        "V3R 5M COVERAGE"
    )

    print(
        "=" * 60
    )

    for (
        k
    ) in INSTRUMENTS:

        v = validation[
            k
        ]

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
        "\nV3R_DOWNLOAD_PASS"
    )


if __name__ == "__main__":
    main()
