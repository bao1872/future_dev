#!/usr/bin/env python3
"""Download multi-instrument 5m / 15m series for Quantile V2 robustness research.

Acquisition path is identical to download_silver_5m_tqsdk.py:
`TqApi.get_kline_serial` with data_length=10000. `DataDownloader` requires a
paid historical-download entitlement this account does not have, and the
serial is hard-capped at 10000 bars.

Validation helpers are copied unchanged from download_silver_5m_tqsdk.py so
the 5m -> 15m aggregation contract is exactly the same across instruments.

AG is NOT re-downloaded and NOT duplicated: it reuses the existing
silver_main_data/silver_main_5m.csv and silver_main_15m.csv and only gains a
new robustness validation/reference record.
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


ROOT = Path(__file__).resolve().parents[1]

INSTRUMENTS = {
    "AG": "KQ.m@SHFE.ag",
    "CU": "KQ.m@SHFE.cu",
    "AL": "KQ.m@SHFE.al",
    "SN": "KQ.m@SHFE.sn",
    "I": "KQ.m@DCE.i",
    "SC": "KQ.m@INE.sc",
    "M": "KQ.m@DCE.m",
    "CF": "KQ.m@CZCE.CF",
}

SERIAL_BARS = 10000

DURATIONS = {
    "5m": 300,
    "15m": 900,
}

OUT_ROOT = (
    ROOT
    / "research"
    / "robustness_data"
    / "raw"
)

# AG raw data is the existing authority. Do not duplicate it.
AG_5M = (
    ROOT
    / "silver_main_data"
    / "silver_main_5m.csv"
)

AG_15M = (
    ROOT
    / "silver_main_data"
    / "silver_main_15m.csv"
)

FIFTEEN_MIN_SECONDS = 15 * 60

NS_PER_SECOND = 1_000_000_000

# 96 x 5m = 8 trading hours of high-frequency warmup.
WARMUP_5M_BARS = 96

MIN_5M_ROWS = 9500

FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_oi",
    "close_oi",
]


# ============================================================
# Helpers copied from download_silver_5m_tqsdk.py
# ============================================================

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


def build_serial_frame(
    klines: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
) -> pd.DataFrame:
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
            "symbol": symbol,
            "timeframe": timeframe,
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

    Same rule as download_silver_main_tqsdk.py and
    download_silver_5m_tqsdk.py: a bar whose nominal end
    (start + duration) is later than now has not closed yet.
    """

    if len(df) < 2:
        raise RuntimeError(
            "insufficient bars"
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
    label: str,
) -> None:

    if df.empty:
        raise RuntimeError(
            f"{label}: empty"
        )

    if not df[
        "datetime_ns"
    ].is_monotonic_increasing:
        raise RuntimeError(
            f"{label}: datetime "
            "not monotonic"
        )

    if df[
        "datetime_ns"
    ].duplicated().any():
        raise RuntimeError(
            f"{label}: duplicate "
            "datetime"
        )

    if df[
        FIELDS
    ].isna().any().any():
        raise RuntimeError(
            f"{label}: NaN in "
            "OHLCV/OI"
        )

    if not (
        df["high"]
        >= df[
            ["open", "close"]
        ].max(axis=1)
    ).all():
        raise RuntimeError(
            f"{label}: invalid high"
        )

    if not (
        df["low"]
        <= df[
            ["open", "close"]
        ].min(axis=1)
    ).all():
        raise RuntimeError(
            f"{label}: invalid low"
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
                f"{label}: negative "
                f"{col}"
            )


def scan_alignment(
    five: pd.DataFrame,
    fifteen: pd.DataFrame,
) -> list[tuple[int, int, int]]:
    """For every 15m bar return (index, 5m count, 5m left position)."""

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
    """Align the 15m window onto the 5m coverage and validate aggregation."""

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
            "No 15m bar is fully covered "
            "by 5m data with sufficient "
            "warmup"
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

    window_end_ns = (
        int(
            fifteen.iloc[
                last
            ][
                "datetime_ns"
            ]
        )
        + (
            FIFTEEN_MIN_SECONDS
            * NS_PER_SECOND
        )
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


# ============================================================
# Per-instrument pipeline
# ============================================================

def read_existing(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path,
        low_memory=False,
    )

    return (
        df
        .sort_values(
            "datetime_ns"
        )
        .reset_index(
            drop=True
        )
    )


def fetch_serial(
    api,
    symbol: str,
    duration_seconds: int,
) -> pd.DataFrame:

    klines = api.get_kline_serial(
        symbol,
        duration_seconds,
        data_length=SERIAL_BARS,
    )

    while not api.is_serial_ready(
        klines
    ):
        api.wait_update()

    return klines


def process_instrument(
    code: str,
    symbol: str,
    api,
) -> dict:

    inst_dir = (
        OUT_ROOT
        / code
    )

    inst_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    reuse_ag = (
        code == "AG"
    )

    if reuse_ag:

        five = read_existing(
            AG_5M
        )

        fifteen = read_existing(
            AG_15M
        )

        five_path = AG_5M
        fifteen_path = AG_15M

        source = (
            "reuse_silver_main_data"
        )

    else:

        five = build_serial_frame(
            fetch_serial(
                api,
                symbol,
                DURATIONS[
                    "5m"
                ],
            ),
            symbol=symbol,
            timeframe="5m",
        )

        fifteen = (
            build_serial_frame(
                fetch_serial(
                    api,
                    symbol,
                    DURATIONS[
                        "15m"
                    ],
                ),
                symbol=symbol,
                timeframe="15m",
            )
        )

        five = drop_forming_tail(
            five,
            DURATIONS[
                "5m"
            ],
        )

        fifteen = (
            drop_forming_tail(
                fifteen,
                DURATIONS[
                    "15m"
                ],
            )
        )

        five_path = (
            inst_dir
            / "5m.csv"
        )

        fifteen_path = (
            inst_dir
            / "15m.csv"
        )

        source = "downloaded"

    validate_ohlc(
        five,
        f"{code} 5m",
    )

    validate_ohlc(
        fifteen,
        f"{code} 15m",
    )

    aggregation = (
        validate_5m_to_15m(
            five,
            fifteen,
        )
    )

    insufficient = (
        len(five)
        < MIN_5M_ROWS
    )

    if not reuse_ag:

        five.to_csv(
            five_path,
            index=False,
        )

        fifteen.to_csv(
            fifteen_path,
            index=False,
        )

    record = {
        "instrument": code,

        "symbol": symbol,

        "source": source,

        "status": (
            "EXCLUDED_DATA_INSUFFICIENT"
            if insufficient
            else "PASS"
        ),

        "five_minute": {
            "rows": int(
                len(five)
            ),

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

            "path": str(
                five_path.relative_to(
                    ROOT
                )
            ),

            "sha256": sha256_file(
                five_path
            ),
        },

        "fifteen_minute": {
            "rows": int(
                len(fifteen)
            ),

            "start": str(
                fifteen.iloc[
                    0
                ][
                    "datetime"
                ]
            ),

            "end": str(
                fifteen.iloc[
                    -1
                ][
                    "datetime"
                ]
            ),

            "path": str(
                fifteen_path.relative_to(
                    ROOT
                )
            ),

            "sha256": sha256_file(
                fifteen_path
            ),
        },

        "aggregation": aggregation,
    }

    (
        inst_dir
        / "validation.json"
    ).write_text(
        json.dumps(
            record,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return record


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

    OUT_ROOT.mkdir(
        parents=True,
        exist_ok=True
    )

    records = {}

    api = None

    try:
        api = TqApi(
            auth=TqAuth(
                user,
                password,
            )
        )

        for code, symbol in (
            INSTRUMENTS.items()
        ):

            print(
                f"\n=== {code} "
                f"({symbol}) ===",
                flush=True,
            )

            try:

                record = (
                    process_instrument(
                        code,
                        symbol,
                        api,
                    )
                )

                records[
                    code
                ] = record

                print(
                    json.dumps(
                        {
                            "status": (
                                record[
                                    "status"
                                ]
                            ),
                            "5m_rows": (
                                record[
                                    "five_minute"
                                ][
                                    "rows"
                                ]
                            ),
                            "15m_rows": (
                                record[
                                    "fifteen_minute"
                                ][
                                    "rows"
                                ]
                            ),
                            "usable_15m": (
                                record[
                                    "aggregation"
                                ][
                                    "usable_15m_bars"
                                ]
                            ),
                            "window": [
                                record[
                                    "aggregation"
                                ][
                                    "usable_window_start"
                                ],
                                record[
                                    "aggregation"
                                ][
                                    "usable_window_end"
                                ],
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

            except Exception as exc:

                print(
                    f"{code} FAILED: "
                    f"{type(exc).__name__}: "
                    f"{exc}",
                    flush=True,
                )

                records[
                    code
                ] = {
                    "instrument": code,
                    "symbol": symbol,
                    "status": (
                        "FAILED"
                    ),
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }

    finally:
        if api is not None:
            api.close()

    index_path = (
        OUT_ROOT
        / "download_index.json"
    )

    index_path.write_text(
        json.dumps(
            records,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n=== DOWNLOAD SUMMARY ===",
        flush=True,
    )

    for code, rec in (
        records.items()
    ):
        print(
            f"{code:<3} "
            f"{rec.get('status')}",
            flush=True,
        )

    print(
        "DOWNLOAD_ROBUSTNESS_FUTURES_DONE",
        flush=True,
    )


if __name__ == "__main__":
    main()
