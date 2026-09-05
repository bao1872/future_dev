#!/usr/bin/env python3
"""Download historical continuous-contract underlying maps for roll audit.

TqSdk exposes `query_his_cont_quotes`, which returns the actual underlying
contract behind each main-continuous symbol per trading date. This removes
the need to infer roll dates from volume / open interest.

The downloaded map is used only to build a conservative exclusion window
around main-contract changes, so the H4 quantile results can be re-checked
with roll-contaminated observations removed from BOTH train and test.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

from tqsdk import TqApi, TqAuth


ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from research.download_robustness_futures import (  # noqa: E402
    INSTRUMENTS,
    load_dotenv,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v3_roll_audit"
)

HISTORY_DAYS = 320


def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    load_dotenv(
        ROOT / ".env"
    )

    user = os.environ.get(
        "TQ_USER",
        "",
    ).strip()

    password = os.environ.get(
        "TQ_PASSWORD",
        "",
    ).strip()

    if not user or not password:
        raise RuntimeError(
            "Missing TQ_USER/TQ_PASSWORD"
        )

    api = None

    try:

        api = TqApi(
            auth=TqAuth(
                user,
                password,
            )
        )

        symbols = list(
            INSTRUMENTS.values()
        )

        history = (
            api.query_his_cont_quotes(
                symbol=symbols,
                n=HISTORY_DAYS,
            )
        )

    finally:

        if api is not None:
            api.close()

    if history.empty:
        raise RuntimeError(
            "Empty continuous-contract history"
        )

    if "date" not in history.columns:
        raise RuntimeError(
            f"Missing date column: "
            f"{history.columns.tolist()}"
        )

    history = history.copy()

    history[
        "date"
    ] = pd.to_datetime(
        history["date"]
    ).dt.normalize()

    # Wide -> long, fully vectorized.
    long = history.melt(
        id_vars=["date"],
        var_name="symbol",
        value_name="underlying_symbol",
    )

    symbol_to_code = {
        symbol: code
        for code, symbol
        in INSTRUMENTS.items()
    }

    long[
        "instrument"
    ] = long[
        "symbol"
    ].map(
        symbol_to_code
    )

    long[
        "underlying_symbol"
    ] = (
        long[
            "underlying_symbol"
        ]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    long = long[
        long["instrument"].notna()
        &
        (
            long[
                "underlying_symbol"
            ]
            != ""
        )
    ].copy()

    long = long.sort_values(
        [
            "instrument",
            "date",
        ]
    ).reset_index(
        drop=True
    )

    long[
        "previous_underlying"
    ] = (
        long
        .groupby(
            "instrument",
            observed=True,
        )[
            "underlying_symbol"
        ]
        .shift(1)
    )

    long[
        "is_roll"
    ] = (
        long[
            "previous_underlying"
        ].notna()
        &
        (
            long[
                "underlying_symbol"
            ]
            !=
            long[
                "previous_underlying"
            ]
        )
    )

    events = long[
        long[
            "is_roll"
        ]
    ].copy()

    counts = (
        events
        .groupby(
            "instrument",
            observed=True,
        )
        .size()
        .reindex(
            INSTRUMENTS.keys(),
            fill_value=0,
        )
    )

    missing = sorted(
        set(
            INSTRUMENTS
        )
        -
        set(
            long[
                "instrument"
            ].unique()
        )
    )

    if missing:
        raise RuntimeError(
            f"Missing instruments "
            f"in roll map: {missing}"
        )

    long.to_csv(
        OUT
        / "continuous_underlying_history.csv",
        index=False,
    )

    events.to_csv(
        OUT
        / "roll_events.csv",
        index=False,
    )

    summary = pd.DataFrame(
        {
            "instrument": (
                list(
                    INSTRUMENTS.keys()
                )
            ),
            "roll_event_count": (
                counts.to_numpy()
            ),
        }
    )

    summary.to_csv(
        OUT
        / "roll_summary.csv",
        index=False,
    )

    validation = {
        "status": "PASS",
        "history_days": HISTORY_DAYS,
        "instrument_count": (
            len(INSTRUMENTS)
        ),
        "history_rows": int(
            len(long)
        ),
        "roll_event_rows": int(
            len(events)
        ),
    }

    (
        OUT
        / "roll_validation.json"
    ).write_text(
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
        )
    )

    print(
        "ROLL_MAP_DOWNLOAD_PASS"
    )


if __name__ == "__main__":
    main()
