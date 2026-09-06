#!/usr/bin/env python3
"""Direction V3R instrument preflight (16 instruments).

Before downloading any formal history, every candidate L8 series
must pass a capability preflight. Market ids and L8 codes are NOT
trusted from strings; each code is probed across candidate markets
and only an actual successful bar fetch counts.

Per instrument, on 700 most recent 5m bars:

    rows > 0
    trade present and non-null
    position present and non-null
    latest timestamp sane AFTER event-time reconstruction
    event-time reconstruction PASS

The sanity check is deliberately applied to the RECONSTRUCTED event
time, not to the raw label. The raw TDX datetime carries trading-day
semantics, so a raw label such as Monday 2026-09-07 23:55 is really
Friday 2026-09-04 23:55 in calendar time. Judging sanity on the raw
label would reject a perfectly valid series.

Reconstruction PASS means, after dropping the earliest trading day
which has no observable predecessor:

    event_datetime has no NaN
    no row is more than 5 minutes in the future
    sorted event_datetime is monotonic

Outputs:

    research/exports/v3r_preflight/instrument_capability.json
    research/exports/v3r_preflight/instrument_capability.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
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
    FREQ_5M,
    PAGE_SIZE,
    TDX_HOST,
    TDX_PORT,
    TDX_TIMEOUT,
    drop_incomplete_tail,
    normalize_tdx_bars,
)

from pytdx.exhq import (  # noqa: E402
    TdxExHq_API,
)


OUT = (
    ROOT
    / "research"
    / "exports"
    / "v3r_preflight"
)

# Candidate exchange markets, resolved empirically.
CANDIDATE_MARKETS = (
    28,
    29,
    30,
    60,
    66,
    47,
)

# root -> (market, code) as expected, but verified below.
EXPECTED = {
    "AG": (
        30,
        "AGL8",
    ),
    "AU": (
        30,
        "AUL8",
    ),
    "CU": (
        30,
        "CUL8",
    ),
    "AL": (
        30,
        "ALL8",
    ),
    "SN": (
        30,
        "SNL8",
    ),
    "NI": (
        30,
        "NIL8",
    ),
    "RB": (
        30,
        "RBL8",
    ),
    "I": (
        29,
        "IL8",
    ),
    "M": (
        29,
        "ML8",
    ),
    "P": (
        29,
        "PL8",
    ),
    "SC": (
        30,
        "SCL8",
    ),
    "RU": (
        30,
        "RUL8",
    ),
    "CF": (
        28,
        "CFL8",
    ),
    "MA": (
        28,
        "MAL8",
    ),
    "TA": (
        28,
        "TAL8",
    ),
    "LC": (
        66,
        "LCL8",
    ),
}

PREFLIGHT_PAGES = 1


def probe(
    api,
    market: int,
    code: str,
):

    try:

        raw = api.get_instrument_bars(
            FREQ_5M,
            int(
                market
            ),
            str(
                code
            ),
            0,
            PAGE_SIZE,
        )

        return api.to_df(
            raw
        )

    except Exception:

        return None


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

    now = pd.Timestamp.now()

    api = TdxExHq_API(
        raise_exception=True,
        auto_retry=True,
    )

    records = []

    with api.connect(
        TDX_HOST,
        TDX_PORT,
        time_out=TDX_TIMEOUT,
    ):

        for (
            root,
            spec,
        ) in EXPECTED.items():

            (
                want_market,
                code,
            ) = spec

            print(
                "=" * 60
            )

            print(
                f"{root}  {code}"
            )

            # Resolve the market empirically.
            resolved = []
            frames = {}

            for market in (
                CANDIDATE_MARKETS
            ):

                df = probe(
                    api,
                    market,
                    code,
                )

                if (
                    df is not None
                    and len(
                        df
                    )
                ):

                    resolved.append(
                        market
                    )

                    frames[
                        market
                    ] = df

            rec: dict = {
                "root": root,
                "code": code,
                "expected_market": int(
                    want_market
                ),
                "resolved_markets": [
                    int(
                        m
                    )
                    for m in (
                        resolved
                    )
                ],
            }

            if not resolved:

                rec[
                    "status"
                ] = (
                    "FAIL"
                )

                rec[
                    "errors"
                ] = [
                    "no market "
                    "returned "
                    "bars"
                ]

                records.append(
                    rec
                )

                print(
                    "  FAIL: no bars"
                )

                continue

            if (
                want_market
                not in (
                    frames
                )
            ):

                rec[
                    "status"
                ] = (
                    "FAIL"
                )

                rec[
                    "errors"
                ] = [
                    f"expected "
                    f"market "
                    f"{want_market} "
                    f"not among "
                    f"{resolved}"
                ]

                records.append(
                    rec
                )

                print(
                    f"  FAIL: expected "
                    f"market "
                    f"{want_market} "
                    f"not in "
                    f"{resolved}"
                )

                continue

            df = frames[
                want_market
            ].copy()

            rec[
                "market"
            ] = int(
                want_market
            )

            rec[
                "rows"
            ] = int(
                len(
                    df
                )
            )

            errors = []

            if (
                len(
                    df
                )
                <= 0
            ):

                errors.append(
                    "rows == 0"
                )

            rec[
                "has_trade"
            ] = bool(
                "trade"
                in df.columns
            )

            rec[
                "has_position"
            ] = bool(
                "position"
                in df.columns
            )

            if not rec[
                "has_trade"
            ]:
                errors.append(
                    "trade "
                    "missing"
                )

            if not rec[
                "has_position"
            ]:
                errors.append(
                    "position "
                    "missing"
                )

            if (
                rec[
                    "has_trade"
                ]
            ):

                trade = pd.to_numeric(
                    df[
                        "trade"
                    ],
                    errors=(
                        "coerce"
                    ),
                )

                rec[
                    "trade_nonnull"
                ] = int(
                    trade.notna().sum()
                )

                rec[
                    "trade_min"
                ] = float(
                    trade.min()
                )

                rec[
                    "trade_max"
                ] = float(
                    trade.max()
                )

                if (
                    trade.notna().sum()
                    == 0
                ):
                    errors.append(
                        "trade "
                        "all null"
                    )

                if (
                    trade.dropna()
                    < 0
                ).any():
                    errors.append(
                        "negative "
                        "trade"
                    )

            if (
                rec[
                    "has_position"
                ]
            ):

                pos = pd.to_numeric(
                    df[
                        "position"
                    ],
                    errors=(
                        "coerce"
                    ),
                )

                rec[
                    "position_nonnull"
                ] = int(
                    pos.notna().sum()
                )

                rec[
                    "position_min"
                ] = float(
                    pos.min()
                )

                rec[
                    "position_max"
                ] = float(
                    pos.max()
                )

                if (
                    pos.notna().sum()
                    == 0
                ):
                    errors.append(
                        "position "
                        "all null"
                    )

            # ---- event-time reconstruction ----

            df[
                "datetime"
            ] = pd.to_datetime(
                df[
                    "datetime"
                ],
                errors=(
                    "coerce"
                ),
            )

            rec[
                "raw_latest"
            ] = str(
                df[
                    "datetime"
                ].max()
            )

            norm = normalize_tdx_bars(
                df,
                period_minutes=5,
            )

            rec[
                "normalized_rows"
            ] = int(
                len(
                    norm
                )
            )

            # The server serves the session it is in, so the
            # newest trading day is partial and can run ahead of
            # this machine's clock. A partially served day must
            # not be left in a research sample.
            norm = drop_incomplete_tail(
                norm,
                now=now,
            )

            rec[
                "after_tail_drop_rows"
            ] = int(
                len(
                    norm
                )
            )

            rec[
                "tail_drop_removed"
            ] = (
                int(
                    rec[
                        "normalized_rows"
                    ]
                )
                - int(
                    len(
                        norm
                    )
                )
            )

            if (
                len(
                    norm
                )
                == 0
            ):

                errors.append(
                    "reconstruction "
                    "produced no "
                    "rows"
                )

            else:

                ev = norm[
                    "event_datetime"
                ]

                n_nan = int(
                    ev.isna().sum()
                )

                rec[
                    "event_nan"
                ] = n_nan

                if n_nan:
                    errors.append(
                        f"{n_nan} "
                        "event NaN"
                    )

                future = (
                    ev
                    > now
                    + pd.Timedelta(
                        minutes=(
                            5
                        )
                    )
                )

                rec[
                    "future_rows"
                ] = int(
                    future.sum()
                )

                if (
                    future.any()
                ):
                    errors.append(
                        f"{int(future.sum())} "
                        "future rows"
                    )

                mono = bool(
                    ev.sort_values()
                    .is_monotonic_increasing
                )

                rec[
                    "monotonic"
                ] = (
                    mono
                )

                if not mono:
                    errors.append(
                        "not "
                        "monotonic"
                    )

                rec[
                    "event_latest"
                ] = str(
                    ev.max()
                )

                rec[
                    "event_earliest"
                ] = str(
                    ev.min()
                )

                lag_days = float(
                    (
                        now
                        - ev.max()
                    ).total_seconds()
                    / 86400.0
                )

                rec[
                    "lag_days"
                ] = round(
                    lag_days,
                    3,
                )

                # Sanity on RECONSTRUCTED time: the newest bar
                # must not be absurdly stale for a liquid
                # continuous series.
                if (
                    lag_days
                    > 30
                ):
                    errors.append(
                        f"latest bar "
                        f"{lag_days:.1f} "
                        f"days old"
                    )

            rec[
                "status"
            ] = (
                "FAIL"
                if errors
                else "PASS"
            )

            rec[
                "errors"
            ] = (
                errors
            )

            records.append(
                rec
            )

            print(
                f"  market={rec.get('market')} "
                f"rows={rec['rows']} "
                f"trade={rec.get('trade_nonnull')} "
                f"position="
                f"{rec.get('position_nonnull')}"
            )

            print(
                f"  raw latest   : "
                f"{rec['raw_latest']}"
            )

            print(
                f"  event latest : "
                f"{rec.get('event_latest')}"
                f"  (lag "
                f"{rec.get('lag_days')}d)"
            )

            print(
                f"  status       : "
                f"{rec['status']} "
                f"{rec['errors']}"
            )

    (
        OUT
        / "instrument_capability.json"
    ).write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    cap = pd.DataFrame(
        records
    )

    cap.to_csv(
        OUT
        / "instrument_capability.csv",
        index=False,
    )

    failed = [
        r[
            "root"
        ]
        for r in records
        if r[
            "status"
        ]
        != "PASS"
    ]

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PREFLIGHT SUMMARY"
    )

    print(
        "=" * 60
    )

    with pd.option_context(
        "display.width",
        250,
    ):
        print(
            cap[
                [
                    "root",
                    "code",
                    "market",
                    "rows",
                    "trade_nonnull",
                    "position_nonnull",
                    "event_latest",
                    "lag_days",
                    "status",
                ]
            ].to_string(
                index=False
            )
        )

    if failed:
        raise RuntimeError(
            f"preflight failed: "
            f"{failed}"
        )

    print(
        "\nV3R_PREFLIGHT_PASS"
    )


if __name__ == "__main__":
    main()
