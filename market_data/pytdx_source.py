"""PyTDX market-data source layer.

This is the ONLY market-data acquisition module in the repository.
There is no second provider and no provider abstraction.

Pipeline:

    PyTDX
      -> acquire 5m *L8 bars
      -> normalize verified TDX timestamp semantics
      -> offline CSV
      -> research

Responsibilities are limited to:

    connect()
    download_5m_l8()
    normalize_tdx_bars()

Strategies, models and indicators must never call this module.
They consume offline CSV through the offline store.

================================================================
Validated TDX semantics -- do not change without re-validation
================================================================

1. TDX bar datetime is partly TRADING-DAY based. The night session
   that precedes a trading day is stamped with that trading day's
   date, not with its own calendar date.

   Example, D = Monday 2026-09-07, previous trading day P = Friday
   2026-09-04:

       raw 2026-09-07 21:30  ->  real 2026-09-04 21:30
       raw 2026-09-07 01:30  ->  real 2026-09-05 01:30
       raw 2026-09-07 10:00  ->  real 2026-09-07 10:00

   The previous trading day is taken from the observed trading-day
   sequence, NOT by a naive `hour >= 21 -> minus one day`, which
   breaks across weekends and holidays.

2. TDX bar datetime is the INTERVAL END. A bar labelled T covers
   [T - period, T).

   Verified against the historical transaction stream: after
   shifting the corrected bar time back by one period, the median
   of (sum of transaction volume) / (bar trade) is exactly 1.0000
   for AG2610 / I2701 / SC2610, and on the minutes where volume
   matches exactly, bar high/low match transaction high/low 100%.

3. Field meaning:

       trade     = bar volume
       position  = bar-end open interest

4. Untrustworthy fields -- never use:

       amount        protocol decode artifact (values ~1e-40)
       zengcang      UNTRUSTED (contradicts bar position change)
       nature        UNTRUSTED (id-like, 139-180 classes)
       nature_name   NOT a current feature
       direction     NOT a current feature

   The historical transaction stream is also incomplete in
   high-activity periods, so it is not an approved feature source
   in the current experiment phase.

5. L8 series are vendor-defined continuous main series. They are
   the current continuous-market research series, but they are not
   treated as exact executable-contract history and are not used
   for contract-roll PnL accounting.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pytdx.exhq import TdxExHq_API


# ============================================================
# Fixed connection
# ============================================================
#
# One verified server only. No server pool, no speed test, no
# circuit breaker. If it does not connect, the run fails.

TDX_HOST = "112.74.214.43"

TDX_PORT = 7727

TDX_TIMEOUT = 5


# TDX extended-market frequency ids, aligned with chanlun-pro.
FREQ_1M = 8
FREQ_5M = 0
FREQ_15M = 1
FREQ_30M = 2
FREQ_60M = 3

PERIOD_MINUTES = {
    FREQ_1M: 1,
    FREQ_5M: 5,
    FREQ_15M: 15,
    FREQ_30M: 30,
    FREQ_60M: 60,
}

PAGE_SIZE = 700


# ============================================================
# Instruments
# ============================================================

INSTRUMENTS = {
    "AG": {
        "market": 30,
        "code": "AGL8",
        "name": "白银主连",
    },
    "CU": {
        "market": 30,
        "code": "CUL8",
        "name": "沪铜主连",
    },
    "AL": {
        "market": 30,
        "code": "ALL8",
        "name": "沪铝主连",
    },
    "SN": {
        "market": 30,
        "code": "SNL8",
        "name": "沪锡主连",
    },
    "I": {
        "market": 29,
        "code": "IL8",
        "name": "铁矿主连",
    },
    "SC": {
        "market": 30,
        "code": "SCL8",
        "name": "原油主连",
    },
    "M": {
        "market": 29,
        "code": "ML8",
        "name": "豆粕主连",
    },
    "CF": {
        "market": 28,
        "code": "CFL8",
        "name": "郑棉主连",
    },
}


# ============================================================
# Connection
# ============================================================

def connect():
    """Open a PyTDX extended-market API.

    No retry pool and no fallback server. A connection failure is
    a hard failure so the experiment never silently runs on stale
    or partial data.
    """

    api = TdxExHq_API(
        raise_exception=True,
        auto_retry=True,
    )

    api.connect(
        TDX_HOST,
        TDX_PORT,
        time_out=TDX_TIMEOUT,
    )

    return api


# ============================================================
# Timestamp normalization
# ============================================================

def reconstruct_event_datetime(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Recover real event time from TDX trading-day datetime.

    Adds, without ever overwriting the raw label:

        tdx_datetime_raw
        trading_day
        prev_trading_day
        event_datetime      (= bar END time in real calendar time)

    Rows belonging to the earliest trading day in the frame have no
    observable previous trading day, so their event_datetime is NaT
    and must be dropped by the caller.
    """

    x = df.copy()

    raw = pd.to_datetime(
        x["datetime"],
        errors="raise",
    )

    x[
        "tdx_datetime_raw"
    ] = raw

    x[
        "trading_day"
    ] = raw.dt.normalize()

    trading_days = sorted(
        x[
            "trading_day"
        ]
        .dropna()
        .unique()
    )

    prev_map = {
        trading_days[i]: trading_days[
            i - 1
        ]
        for i in range(
            1,
            len(trading_days),
        )
    }

    x[
        "prev_trading_day"
    ] = (
        x[
            "trading_day"
        ].map(prev_map)
    )

    hour = raw.dt.hour

    event_day = x[
        "trading_day"
    ].copy()

    # Night session before midnight belongs to the calendar date
    # of the previous trading day.
    mask_evening = hour >= 21

    event_day.loc[
        mask_evening
    ] = x.loc[
        mask_evening,
        "prev_trading_day",
    ]

    # After-midnight continuation of that evening session.
    mask_overnight = hour < 8

    event_day.loc[
        mask_overnight
    ] = (
        x.loc[
            mask_overnight,
            "prev_trading_day",
        ]
        + pd.Timedelta(
            days=1
        )
    )

    time_part = (
        raw
        - raw.dt.normalize()
    )

    x[
        "event_datetime"
    ] = (
        event_day
        + time_part
    )

    return x


def normalize_tdx_bars(
    df: pd.DataFrame,
    *,
    period_minutes: int,
) -> pd.DataFrame:
    """Apply the verified bar-time contract to a raw TDX frame.

    Produces a frame indexed for research on `bar_start_time`:

        bar_end_time        corrected real calendar time of the
                            interval end (== TDX label corrected
                            for the trading-day convention)
        bar_start_time      bar_end_time - period
        availability_time   == bar_end_time; the moment this bar's
                            close / volume / OI actually becomes
                            known, and therefore the correct key
                            for causal joins and target
                            availability

    Rows on the earliest trading day are dropped because their
    event time cannot be reconstructed.
    """

    x = reconstruct_event_datetime(
        df
    )

    x = x[
        x[
            "event_datetime"
        ].notna()
    ].copy()

    x = x.sort_values(
        "event_datetime"
    )

    x[
        "bar_end_time"
    ] = x[
        "event_datetime"
    ]

    x[
        "bar_start_time"
    ] = (
        x[
            "bar_end_time"
        ]
        - pd.Timedelta(
            minutes=(
                period_minutes
            )
        )
    )

    x[
        "availability_time"
    ] = x[
        "bar_end_time"
    ]

    return x.reset_index(
        drop=True
    )


# ============================================================
# Download
# ============================================================

def fetch_bars(
    api,
    market: int,
    code: str,
    frequency: int = FREQ_5M,
    max_pages: int = 300,
) -> pd.DataFrame:
    """Fetch bars page by page until the server stops returning.

    Pagination is verified: at 256 pages AGL8 5m returned 179,200
    unique bars reaching back to 2019-10-11. The loop ends when a
    short or empty page comes back.
    """

    frames = []

    for page in range(
        max_pages
    ):

        raw = api.get_instrument_bars(
            frequency,
            int(market),
            str(code),
            page * PAGE_SIZE,
            PAGE_SIZE,
        )

        df = api.to_df(
            raw
        )

        if (
            df is None
            or len(df) == 0
        ):
            break

        df[
            "_page"
        ] = page

        frames.append(
            df
        )

        if (
            len(df)
            < PAGE_SIZE
        ):
            break

    if not frames:
        return pd.DataFrame()

    out = pd.concat(
        frames,
        ignore_index=True,
    )

    out[
        "datetime"
    ] = pd.to_datetime(
        out[
            "datetime"
        ],
        errors="coerce",
    )

    out = (
        out
        .drop_duplicates(
            "datetime",
            keep="last",
        )
        .sort_values(
            "datetime"
        )
        .reset_index(
            drop=True
        )
    )

    return out


def download_5m_l8(
    instrument: str,
    *,
    api=None,
    max_pages: int = 300,
) -> pd.DataFrame:
    """Download and normalize one L8 5m series.

    Returns a frame carrying the verified bar-time contract plus
    the raw label, and the two trusted payload fields:

        trade      = bar volume
        position   = bar-end open interest
    """

    spec = INSTRUMENTS.get(
        str(
            instrument
        ).upper()
    )

    if spec is None:
        raise KeyError(
            f"Unknown instrument: "
            f"{instrument}"
        )

    owns_api = (
        api is None
    )

    if owns_api:
        api = connect()

    try:

        raw = fetch_bars(
            api,
            spec[
                "market"
            ],
            spec[
                "code"
            ],
            FREQ_5M,
            max_pages,
        )

    finally:

        if owns_api:
            api.close()

    if raw.empty:
        return raw

    out = normalize_tdx_bars(
        raw,
        period_minutes=(
            PERIOD_MINUTES[
                FREQ_5M
            ]
        ),
    )

    out[
        "instrument"
    ] = str(
        instrument
    ).upper()

    out[
        "tdx_market"
    ] = spec[
        "market"
    ]

    out[
        "tdx_code"
    ] = spec[
        "code"
    ]

    return out


# ============================================================
# Refresh entrypoint
# ============================================================

def refresh_offline_market_data() -> None:
    """Market-data source layer entrypoint.

    Governance boundary: strategy modules must not call this
    function and must not import PyTDX directly.
    """

    raise NotImplementedError(
        "Offline refresh is owned by the current experiment "
        "download script. This module exposes connect(), "
        "download_5m_l8() and normalize_tdx_bars() only."
    )
