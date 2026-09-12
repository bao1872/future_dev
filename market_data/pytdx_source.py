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

import time
from pathlib import Path

import pandas as pd

from pytdx.errors import TdxConnectionError, TdxFunctionCallError
from pytdx.exhq import TdxExHq_API


# ============================================================
# Connection & Server Pool
# ============================================================
# Aligned with chanlun-pro candidate server pool and IP selection mechanism.

TDX_SERVERS = [
    {"ip": "116.205.143.214", "port": 7727, "name": "扩展市场广州双线1"},
    {"ip": "112.74.214.43", "port": 7727, "name": "扩展市场深圳双线1"},
    {"ip": "120.25.218.6", "port": 7727, "name": "扩展市场深圳双线2"},
    {"ip": "43.139.173.246", "port": 7727, "name": "扩展市场深圳双线3"},
    {"ip": "159.75.90.107", "port": 7727, "name": "扩展市场深圳双线4"},
    {"ip": "106.52.170.195", "port": 7727, "name": "扩展市场深圳双线5"},
    {"ip": "139.9.191.175", "port": 7727, "name": "扩展市场广州双线3"},
    {"ip": "175.24.47.69", "port": 7727, "name": "扩展市场上海双线7"},
    {"ip": "150.158.9.199", "port": 7727, "name": "扩展市场上海双线1"},
    {"ip": "150.158.20.127", "port": 7727, "name": "扩展市场上海双线2"},
    {"ip": "49.235.119.116", "port": 7727, "name": "扩展市场上海双线3"},
    {"ip": "49.234.13.160", "port": 7727, "name": "扩展市场上海双线4"},
    {"ip": "124.71.223.19", "port": 7727, "name": "扩展市场广州双线2"},
    {"ip": "113.45.175.47", "port": 7727, "name": "扩展市场广州双线4"},
    {"ip": "123.60.173.210", "port": 7727, "name": "扩展市场上海双线5"},
    {"ip": "118.89.69.202", "port": 7727, "name": "扩展市场上海双线6"},
]

TDX_HOST = "116.205.143.214"

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
    # Precious
    "AG": {
        "market": 30,
        "code": "AGL8",
        "name": "白银主连",
    },
    "AU": {
        "market": 30,
        "code": "AUL8",
        "name": "黄金主连",
    },
    # Non-ferrous
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
    "NI": {
        "market": 30,
        "code": "NIL8",
        "name": "沪镍主连",
    },
    # Ferrous
    "RB": {
        "market": 30,
        "code": "RBL8",
        "name": "螺纹主连",
    },
    "I": {
        "market": 29,
        "code": "IL8",
        "name": "铁矿主连",
    },
    # Energy / chemical
    "SC": {
        "market": 30,
        "code": "SCL8",
        "name": "原油主连",
    },
    "RU": {
        "market": 30,
        "code": "RUL8",
        "name": "橡胶主连",
    },
    "MA": {
        "market": 28,
        "code": "MAL8",
        "name": "甲醇主连",
    },
    "TA": {
        "market": 28,
        "code": "TAL8",
        "name": "PTA主连",
    },
    # Agriculture
    "M": {
        "market": 29,
        "code": "ML8",
        "name": "豆粕主连",
    },
    "P": {
        "market": 29,
        "code": "PL8",
        "name": "棕榈油主连",
    },
    "CF": {
        "market": 28,
        "code": "CFL8",
        "name": "郑棉主连",
    },
    # New energy (GFEX)
    "LC": {
        "market": 66,
        "code": "LCL8",
        "name": "碳酸锂主连",
    },
}


# ============================================================
# Connection & Server Failover
# ============================================================

def ping_server(
    ip: str,
    port: int = 7727,
    timeout: float = 1.5,
) -> float | None:
    """Probe an extended-market server using live bar request.

    Aligned with chanlun-pro tdx_best_ip.py: tests whether
    get_instrument_bars responds with non-empty bar data within timeout.
    """
    t0 = time.perf_counter()
    api = TdxExHq_API(raise_exception=True)
    try:
        if api.connect(ip, port, time_out=timeout):
            bars = api.get_instrument_bars(9, 74, "AAPL", 0, 10)
            if bars and len(bars) > 0:
                return time.perf_counter() - t0
    except Exception:
        pass
    finally:
        try:
            api.disconnect()
        except Exception:
            pass
    return None


def select_best_server(
    servers: list[dict] | None = None,
    timeout: float = 1.5,
) -> tuple[str, int]:
    """Select the fastest responsive TDX extended-market server.

    Aligned with chanlun-pro commits cbb5075 and 89a0ae7.
    Updates global TDX_HOST and TDX_PORT upon selection.
    """
    global TDX_HOST, TDX_PORT

    candidates = servers or TDX_SERVERS
    scored = []
    for s in candidates:
        latency = ping_server(s["ip"], s["port"], timeout=timeout)
        if latency is not None:
            scored.append((latency, s["ip"], s["port"], s.get("name", "")))

    if not scored:
        raise TdxConnectionError(
            f"No responsive TDX server found among {len(candidates)} candidates."
        )

    scored.sort(key=lambda x: x[0])
    best = scored[0]
    TDX_HOST = best[1]
    TDX_PORT = best[2]
    return best[1], best[2]


def connect(
    host: str | None = None,
    port: int | None = None,
    time_out: int = TDX_TIMEOUT,
    auto_failover: bool = True,
):
    """Open a PyTDX extended-market API.

    If the target server fails or raises TdxFunctionCallError/TdxConnectionError,
    automatically reselects the optimal responsive server from TDX_SERVERS.
    """
    global TDX_HOST, TDX_PORT

    target_host = host or TDX_HOST
    target_port = port or TDX_PORT

    api = TdxExHq_API(
        raise_exception=True,
        auto_retry=True,
    )

    try:
        api.connect(
            target_host,
            target_port,
            time_out=time_out,
        )
        # Verify function call capability, aligned with chanlun-pro
        probe = api.get_instrument_bars(9, 74, "AAPL", 0, 10)
        if probe is None or len(probe) == 0:
            raise TdxFunctionCallError("Probe bar call returned empty response")
        return api
    except (TdxConnectionError, TdxFunctionCallError, Exception):
        if not auto_failover:
            raise
        try:
            api.disconnect()
        except Exception:
            pass
        best_host, best_port = select_best_server()
        api = TdxExHq_API(
            raise_exception=True,
            auto_retry=True,
        )
        api.connect(
            best_host,
            best_port,
            time_out=time_out,
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


def drop_incomplete_tail(
    df: pd.DataFrame,
    *,
    now=None,
    tolerance_minutes: int = 0,
) -> pd.DataFrame:
    """Drop bars whose event time is beyond the causal cutoff.

    The TDX bar label is the END of the interval, so a bar whose
    event_datetime is at or before "now" is already closed and is
    knowable. A bar whose event_datetime is after "now" has not
    finished, so its OHLC is still moving.

    The default tolerance is therefore zero. A positive tolerance
    would admit a bar that has not closed yet, which is exactly the
    thing this function exists to prevent. If a machine clock is
    behind the market clock, that is a clock problem and should be
    reported as clock skew, not absorbed by a hidden margin.

    Clock skew is reported explicitly rather than tolerated, so a
    misconfigured clock shows up in the log instead of silently
    admitting unfinished bars.

    The TDX server serves the session it is currently in, and its
    market date can run ahead of the machine that is downloading,
    so the newest bars can carry an event time beyond "now". Those
    bars are not knowable and must not enter any sample.

    Removal unit is the BAR, not the trading day.

    This matters because a TDX trading day is not a calendar day.
    A trading day labelled Monday can contain bars that really
    happened on Friday evening, Saturday after midnight, and Monday
    daytime. Deleting the whole trading day because its Monday
    daytime bar is in the future would also throw away the Friday
    night bars, which are already complete and already in the past.

    Data validity and experiment endpoint are separate concerns:

        validity      event_datetime <= now
        endpoint      chosen by the experiment, not here
    """

    if df.empty:
        return df

    now = (
        pd.Timestamp(
            now
        )
        if now
        is not None
        else pd.Timestamp.now()
    )

    limit = (
        now
        + pd.Timedelta(
            minutes=(
                tolerance_minutes
            )
        )
    )

    ev = pd.to_datetime(
        df[
            "event_datetime"
        ],
        errors=(
            "coerce"
        ),
    )

    keep = (
        ev <= limit
    ) | ev.isna()

    out = df[
        keep
    ]

    dropped = int(
        (
            ~keep
        ).sum()
    )

    if dropped:

        skew = (
            ev.max()
            - now
        )

        print(
            f"    drop_incomplete_tail: "
            f"dropped {dropped} bar(s); "
            f"max event time is "
            f"{skew} beyond now "
            f"(clock skew if positive "
            f"and large)"
        )

    return out.reset_index(
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
    not_before=None,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch bars page by page until the server stops returning.

    Pagination is verified: at 256 pages AGL8 5m returned 179,200
    unique bars reaching back to 2019-10-11. The loop ends when a
    short or empty page comes back, or when `not_before` has been
    reached. Pages come back newest-first, so early stopping is
    safe.

    Retries on TdxFunctionCallError / TdxConnectionError, aligned with
    chanlun-pro commit cbb5075.
    """
    global TDX_HOST, TDX_PORT

    frames = []

    for page in range(
        max_pages
    ):

        raw = None
        for attempt in range(max_retries):
            try:
                raw = api.get_instrument_bars(
                    frequency,
                    int(market),
                    str(code),
                    page * PAGE_SIZE,
                    PAGE_SIZE,
                )
                if raw is not None:
                    break
            except (TdxConnectionError, TdxFunctionCallError, Exception):
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (attempt + 1))
                try:
                    api.disconnect()
                except Exception:
                    pass
                try:
                    api.connect(TDX_HOST, TDX_PORT, time_out=TDX_TIMEOUT)
                except Exception:
                    best_host, best_port = select_best_server()
                    api.connect(best_host, best_port, time_out=TDX_TIMEOUT)

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

        if not_before is not None:

            seen = pd.to_datetime(
                df[
                    "datetime"
                ],
                errors=(
                    "coerce"
                ),
            )

            if (
                seen.min()
                <= not_before
            ):
                break

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
    not_before=None,
) -> pd.DataFrame:
    """Download and normalize one L8 5m series.

    `not_before` stops the pagination as soon as the fetched bars
    reach back far enough, so a 20-month experiment window does not
    pull the full multi-year history.

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

    cutoff = (
        pd.Timestamp(
            not_before
        )
        if not_before
        is not None
        else None
    )

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
            not_before=cutoff,
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
