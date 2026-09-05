#!/usr/bin/env python3
"""Download the KQ.m@SHFE.ag 5m series and validate it against the current 15m file.

Baseline note
-------------
The current TqSdk account does not have the paid "historical download"
entitlement, so `tqsdk.tools.DataDownloader` refuses to construct. The 5m
series is therefore taken from `TqApi.get_kline_serial`, which is the same
mechanism the existing 15m/1h/4h downloader uses.

That mechanism is hard-capped at 10000 bars per serial, so the 5m history is
~4.5 months, not the ~13.5 months covered by the current 15m file. The 15m
research window is therefore RE-BASED onto whatever the 5m series can cover:

    a 15m bar is usable only when
      * its [start, next_start) bucket contains exactly three 5m bars, and
      * at least WARMUP_5M_BARS (96) 5m bars exist before that bucket start

The usable range is required to be contiguous. Existing 15m / 1h / 4h files
are NOT re-downloaded and NOT modified.

Everything else (OHLC validation, 5m -> 15m exact aggregation, atomic
replacement) follows the same rules as download_silver_main_tqsdk.py.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from tqsdk import TqApi, TqAuth


ROOT = Path(__file__).resolve().parent

SYMBOL = "KQ.m@SHFE.ag"

RAW_15M = (
    ROOT
    / "silver_main_data"
    / "silver_main_15m.csv"
)

OUT_5M = (
    ROOT
    / "silver_main_data"
    / "silver_main_5m.csv"
)

EXPORT_DIR = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_data"
)

VALIDATION_PATH = (
    EXPORT_DIR
    / "5m_download_validation.json"
)

TMP_NORMALIZED = (
    ROOT
    / "silver_main_data"
    / ".silver_main_5m.normalized.csv"
)

FIVE_MIN_SECONDS = 5 * 60
FIFTEEN_MIN_SECONDS = 15 * 60

NS_PER_SECOND = 1_000_000_000

SERIAL_BARS = 10000

# 96 x 5m = 8 trading hours of high-frequency warmup. Required before the
# first research decision so the 8h realized window is complete from row 0.
WARMUP_5M_BARS = 96

FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_oi",
    "close_oi",
]


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(
        encoding="utf-8"
    ).splitlines():

        line = raw_line.strip()

        if (
            not line
            or line.startswith("#")
            or "=" not in line
        ):
            continue

        key, _, value = line.partition("=")

        key = key.strip()

        value = (
            value
            .strip()
            .strip('"')
            .strip("'")
        )

        if key and key not in os.environ:
            os.environ[key] = value


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(
                1024 * 1024
            ),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def build_five_frame(klines: pd.DataFrame) -> pd.DataFrame:
    """Normalize a TqSdk kline serial into the repo's raw CSV schema."""
    ns = (
        klines["datetime"]
        .astype("int64")
        .to_numpy(
            dtype=np.int64
        )
    )

    display = (
        pd.to_datetime(
            ns,
            unit="ns",
            utc=True,
        )
        .tz_convert(
            "Asia/Shanghai"
        )
        .strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    )

    out = pd.DataFrame(
        {
            "datetime": display,
            "datetime_ns": ns,
            "symbol": SYMBOL,
            "timeframe": "5m",
        }
    )

    for field in FIELDS:
        out[field] = pd.to_numeric(
            klines[field],
            errors="raise",
        ).to_numpy(
            dtype=float
        )

    out = (
        out
        .sort_values(
            "datetime_ns"
        )
        .drop_duplicates(
            "datetime_ns",
            keep="last",
        )
        .reset_index(
            drop=True
        )
    )

    return out


def drop_forming_tail(
    df: pd.DataFrame,
    duration_seconds: int,
) -> pd.DataFrame:
    """Drop the still-forming tail bar.

    Same rule as download_silver_main_tqsdk.py: a bar whose nominal end
    (start + duration) is later than now has not closed yet.
    """
    if len(df) < 2:
        raise RuntimeError(
            "5m insufficient bars"
        )

    now_ns = pd.Timestamp.now(
        tz="UTC"
    ).value

    last_start_ns = int(
        df.iloc[-1][
            "datetime_ns"
        ]
    )

    nominal_end_ns = (
        last_start_ns
        + duration_seconds
        * NS_PER_SECOND
    )

    if nominal_end_ns > now_ns:
        dropped = df.iloc[-1][
            "datetime"
        ]

        print(
            "[5m] drop forming bar: "
            f"{dropped}"
        )

        return (
            df.iloc[:-1]
            .copy()
            .reset_index(
                drop=True
            )
        )

    return df


def validate_ohlc(
    df: pd.DataFrame,
) -> None:

    if df.empty:
        raise RuntimeError(
            "5m data is empty"
        )

    if not df[
        "datetime_ns"
    ].is_monotonic_increasing:
        raise RuntimeError(
            "5m datetime is not monotonic"
        )

    if df[
        "datetime_ns"
    ].duplicated().any():
        raise RuntimeError(
            "5m duplicate datetime"
        )

    if df[
        FIELDS
    ].isna().any().any():
        raise RuntimeError(
            "5m OHLCV/OI contains NaN"
        )

    if not (
        df["high"]
        >= df[
            ["open", "close"]
        ].max(axis=1)
    ).all():
        raise RuntimeError(
            "5m invalid high"
        )

    if not (
        df["low"]
        <= df[
            ["open", "close"]
        ].min(axis=1)
    ).all():
        raise RuntimeError(
            "5m invalid low"
        )

    for col in (
        "volume",
        "open_oi",
        "close_oi",
    ):
        if (
            df[col] < 0
        ).any():
            raise RuntimeError(
                f"5m negative {col}"
            )


def scan_alignment(
    five: pd.DataFrame,
    fifteen: pd.DataFrame,
) -> list[tuple[int, int, int]]:
    """For every current 15m bar return (index, 5m count, 5m left position).

    Bucket is [15m start, next 15m start); for the final bar
    [start, start + 15min). This survives lunch/night/weekend gaps.
    """

    five_times = (
        five["datetime_ns"]
        .to_numpy(
            dtype=np.int64
        )
    )

    rows: list[
        tuple[int, int, int]
    ] = []

    for i in range(
        len(fifteen)
    ):
        start_ns = int(
            fifteen.iloc[i][
                "datetime_ns"
            ]
        )

        if i + 1 < len(
            fifteen
        ):
            end_ns = int(
                fifteen.iloc[
                    i + 1
                ][
                    "datetime_ns"
                ]
            )
        else:
            end_ns = (
                start_ns
                + FIFTEEN_MIN_SECONDS
                * NS_PER_SECOND
            )

        left = int(
            np.searchsorted(
                five_times,
                start_ns,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                five_times,
                end_ns,
                side="left",
            )
        )

        rows.append(
            (
                i,
                int(
                    right - left
                ),
                left,
            )
        )

    return rows


def validate_5m_to_15m(
    five: pd.DataFrame,
    fifteen: pd.DataFrame,
) -> dict:
    """Align the 15m window onto the 5m coverage and validate aggregation.

    The usable window is the contiguous run of 15m bars that are exactly
    covered by three 5m bars AND have enough 5m warmup behind them.
    """

    coverage = scan_alignment(
        five,
        fifteen,
    )

    non_three = [
        i
        for i, count, _left in (
            coverage
        )
        if count != 3
    ]

    usable = [
        i
        for i, count, left in (
            coverage
        )
        if (
            count == 3
            and left
            >= WARMUP_5M_BARS
        )
    ]

    if not usable:
        raise RuntimeError(
            "No 15m bar is fully covered by "
            "5m data with sufficient warmup"
        )

    first = usable[0]
    last = usable[-1]

    if usable != list(
        range(
            first,
            last + 1,
        )
    ):
        raise RuntimeError(
            "Usable 15m range is not "
            "contiguous"
        )

    five_times = (
        five["datetime_ns"]
        .to_numpy(
            dtype=np.int64
        )
    )

    errors: list[dict] = []

    compared = 0

    for i in range(
        first,
        last + 1,
    ):
        h = fifteen.iloc[i]

        start_ns = int(
            h["datetime_ns"]
        )

        if i + 1 < len(
            fifteen
        ):
            end_ns = int(
                fifteen.iloc[
                    i + 1
                ][
                    "datetime_ns"
                ]
            )
        else:
            end_ns = (
                start_ns
                + FIFTEEN_MIN_SECONDS
                * NS_PER_SECOND
            )

        left = int(
            np.searchsorted(
                five_times,
                start_ns,
                side="left",
            )
        )

        right = int(
            np.searchsorted(
                five_times,
                end_ns,
                side="left",
            )
        )

        sub = five.iloc[
            left:right
        ]

        if len(sub) != 3:
            errors.append(
                {
                    "datetime": str(
                        h[
                            "datetime"
                        ]
                    ),
                    "type": (
                        "five_minute_count"
                    ),
                    "expected": 3,
                    "actual": int(
                        len(sub)
                    ),
                }
            )
            continue

        expected = {
            "open": float(
                sub.iloc[0][
                    "open"
                ]
            ),
            "high": float(
                sub[
                    "high"
                ].max()
            ),
            "low": float(
                sub[
                    "low"
                ].min()
            ),
            "close": float(
                sub.iloc[-1][
                    "close"
                ]
            ),
            "volume": float(
                sub[
                    "volume"
                ].sum()
            ),
            "open_oi": float(
                sub.iloc[0][
                    "open_oi"
                ]
            ),
            "close_oi": float(
                sub.iloc[-1][
                    "close_oi"
                ]
            ),
        }

        for field, exp in (
            expected.items()
        ):
            actual = float(
                h[field]
            )

            if not np.isclose(
                actual,
                exp,
                rtol=0.0,
                atol=1e-9,
            ):
                errors.append(
                    {
                        "datetime": str(
                            h[
                                "datetime"
                            ]
                        ),
                        "type": (
                            "aggregation_mismatch"
                        ),
                        "field": field,
                        "expected": exp,
                        "actual": actual,
                    }
                )

        compared += 1

    if errors:
        preview = errors[:20]

        raise RuntimeError(
            "5m -> 15m aggregation "
            "FAILED. total errors="
            f"{len(errors)}\n"
            f"first errors={preview}"
        )

    window_start_ns = int(
        fifteen.iloc[first][
            "datetime_ns"
        ]
    )

    window_end_ns = int(
        fifteen.iloc[last][
            "datetime_ns"
        ]
    ) + (
        FIFTEEN_MIN_SECONDS
        * NS_PER_SECOND
    )

    in_window = five[
        (
            five[
                "datetime_ns"
            ]
            >= window_start_ns
        )
        &
        (
            five[
                "datetime_ns"
            ]
            < window_end_ns
        )
    ]

    expected_count = (
        last
        - first
        + 1
    ) * 3

    if len(in_window) != expected_count:
        raise RuntimeError(
            "5m aligned-window count "
            f"{len(in_window)} != "
            f"{expected_count}"
        )

    return {
        "fifteen_bars_total": int(
            len(fifteen)
        ),

        "fifteen_bars_with_non_three_5m": int(
            len(non_three)
        ),

        "fifteen_bars_covered_by_5m": int(
            sum(
                1
                for _i, c, _l in (
                    coverage
                )
                if c == 3
            )
        ),

        "usable_15m_bars": int(
            len(usable)
        ),

        "first_15m_index": int(
            first
        ),

        "last_15m_index": int(
            last
        ),

        "usable_window_start": str(
            fifteen.iloc[first][
                "datetime"
            ]
        ),

        "usable_window_end": str(
            fifteen.iloc[last][
                "datetime"
            ]
        ),

        "five_bars_in_aligned_window": int(
            len(in_window)
        ),

        "expected_five_bars": int(
            expected_count
        ),

        "compared_15m_bars": int(
            compared
        ),

        "count_mismatch": 0,

        "ohlcv_oi_mismatch": 0,

        "aggregation_pass": True,

        "warmup_5m_bars_before_window": int(
            len(
                five[
                    five[
                        "datetime_ns"
                    ]
                    < window_start_ns
                ]
            )
        ),
    }


def main() -> None:

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
        print(
            "Missing TQ_USER / "
            "TQ_PASSWORD",
            file=sys.stderr,
        )
        sys.exit(1)

    if not RAW_15M.is_file():
        raise RuntimeError(
            f"Missing current 15m: "
            f"{RAW_15M}"
        )

    fifteen = pd.read_csv(
        RAW_15M,
        low_memory=False,
    )

    fifteen = (
        fifteen
        .sort_values(
            "datetime_ns"
        )
        .reset_index(
            drop=True
        )
    )

    if len(fifteen) != 10000:
        raise RuntimeError(
            "Current 15m row count "
            f"changed: {len(fifteen)}"
        )

    OUT_5M.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    EXPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TMP_NORMALIZED.exists():
        TMP_NORMALIZED.unlink()

    print(
        f"Downloading {SYMBOL} 5m "
        f"via get_kline_serial "
        f"(data_length="
        f"{SERIAL_BARS})"
    )

    api = None

    try:
        api = TqApi(
            auth=TqAuth(
                user,
                password,
            )
        )

        klines = api.get_kline_serial(
            SYMBOL,
            FIVE_MIN_SECONDS,
            data_length=SERIAL_BARS,
        )

        while not api.is_serial_ready(
            klines
        ):
            api.wait_update()

        print(
            f"[5m] serial ready: "
            f"{len(klines)} bars"
        )

        five = build_five_frame(
            klines
        )

    finally:
        if api is not None:
            api.close()

    five = drop_forming_tail(
        five,
        FIVE_MIN_SECONDS,
    )

    validate_ohlc(five)

    aggregation = (
        validate_5m_to_15m(
            five,
            fifteen,
        )
    )

    five.to_csv(
        TMP_NORMALIZED,
        index=False,
    )

    # Atomic replacement only after all
    # validation has passed.
    TMP_NORMALIZED.replace(
        OUT_5M
    )

    validation = {
        "symbol": SYMBOL,
        "timeframe": "5m",

        "acquisition": {
            "method": (
                "TqApi.get_kline_serial"
            ),
            "requested_data_length": (
                SERIAL_BARS
            ),
            "note": (
                "DataDownloader unavailable: "
                "account has no paid "
                "historical-download "
                "entitlement. Serial is "
                "hard-capped at 10000 bars, "
                "so the 15m research window "
                "is re-based onto the 5m "
                "coverage."
            ),
        },

        "rows": int(len(five)),

        "start": str(
            five.iloc[0][
                "datetime"
            ]
        ),

        "end": str(
            five.iloc[-1][
                "datetime"
            ]
        ),

        "warmup_rows_before_15m": (
            aggregation[
                "warmup_5m_bars_before_window"
            ]
        ),

        "sha256": sha256_file(
            OUT_5M
        ),

        "aggregation": aggregation,
    }

    VALIDATION_PATH.write_text(
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
        "DOWNLOAD_5M_VALIDATION_PASS"
    )


if __name__ == "__main__":
    main()
