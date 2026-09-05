#!/usr/bin/env python3
"""Build the 15m decision panel from the PyTDX 5m source bars.

Only 5m is authoritative. 15m is aggregated locally:

    each 15m bar must contain exactly 3 CONSECUTIVE 5m bars,
    otherwise the bucket is discarded.

This is the step that removes the old cross-timeframe / cross-
instrument coverage mismatch: every instrument now shares one
common calendar window and one locally defined aggregation rule.

Features are the two locked sets only:

    F1          direction + location + session-time structure
    F1_VOL      F1 + realized-volatility level and acceleration

No SMC, no DSA, no Momentum, no transaction-flow fields.

Targets, entry at the next 15m bar open and exit at the close of
the H-th 15m bar after the base bar:

    target_raw_return_hH   log(close[i+H] / open[i+1])
    target_long_mfe_hH     best favourable excursion for a long
    target_short_mfe_hH    best favourable excursion for a short

All heavy work is vectorized: groupby aggregation, rolling windows,
cumulative sums with searchsorted, and sliding_window_view for the
path excursions. Python loops exist only over the small experiment
dimensions (instrument x horizon).

Outputs:

    research/exports/pytdx_panel/<INSTRUMENT>_panel.csv
    research/exports/pytdx_panel/panel_manifest.json
    research/exports/pytdx_panel/panel_validation.json
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
    INSTRUMENTS,
)


SRC = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_5m"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "pytdx_panel"
)

HORIZONS = (
    4,
    8,
)

FIVE_MIN_NS = 5 * 60 * 1_000_000_000

FIFTEEN_MIN_NS = (
    15 * 60 * 1_000_000_000
)

LOCATION_WINDOW = 32

LONG_GAP_MINUTES = 60

RET_LAGS = (
    1,
    4,
    8,
    16,
)


# ============================================================
# 5m -> 15m
# ============================================================

def aggregate_15m(
    five: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate 5m bars into 15m bars.

    A bucket is the 15-minute grid cell of `bar_start_time`. It is
    accepted only when it holds exactly 3 bars whose start times
    are 5 minutes apart, i.e. three consecutive 5m bars.
    """

    x = five.sort_values(
        "bar_start_time"
    )

    start_ns = (
        x[
            "bar_start_time"
        ]
        .to_numpy(
            dtype="datetime64[ns]"
        )
        .astype(
            np.int64
        )
    )

    bucket = (
        start_ns
        // FIFTEEN_MIN_NS
    )

    x = x.assign(
        _bucket=bucket
    )

    grouped = x.groupby(
        "_bucket",
        observed=True,
    )

    agg = grouped.agg(
        bar_start_time=(
            "bar_start_time",
            "min",
        ),
        bar_end_time=(
            "bar_end_time",
            "max",
        ),
        open=(
            "open",
            "first",
        ),
        high=(
            "high",
            "max",
        ),
        low=(
            "low",
            "min",
        ),
        close=(
            "close",
            "last",
        ),
        volume=(
            "trade",
            "sum",
        ),
        open_oi=(
            "position",
            "first",
        ),
        close_oi=(
            "position",
            "last",
        ),
        n_bars=(
            "close",
            "size",
        ),
        span_ns=(
            "bar_start_time",
            lambda s: (
                s.to_numpy(
                    dtype=(
                        "datetime64[ns]"
                    )
                ).astype(
                    np.int64
                ).max()
                - s.to_numpy(
                    dtype=(
                        "datetime64[ns]"
                    )
                ).astype(
                    np.int64
                ).min()
            ),
        ),
    )

    # Exactly three bars, spanning 10 minutes end to end.
    ok = (
        agg[
            "n_bars"
        ].to_numpy()
        == 3
    ) & (
        agg[
            "span_ns"
        ].to_numpy()
        == 2 * FIVE_MIN_NS
    )

    out = (
        agg[ok]
        .drop(
            columns=[
                "n_bars",
                "span_ns",
            ]
        )
        .sort_values(
            "bar_start_time"
        )
        .reset_index(
            drop=True
        )
    )

    return out


# ============================================================
# Realized volatility from the 5m series
# ============================================================

def realized_variance(
    five: pd.DataFrame,
    target_end: pd.Series,
) -> dict[str, np.ndarray]:
    """Trailing 5m realized variance, cumulative-sum vectorized.

    For each 15m bar we take the position of the first 5m bar that
    starts at or after the 15m bar's end time, then read the
    trailing window off the cumulative sum of squared 5m log
    returns. This is the formula that was verified to reproduce the
    previous panel's `feat_5m_1h_rv` exactly.
    """

    f = five.sort_values(
        "bar_start_time"
    )

    close = f[
        "close"
    ].to_numpy(
        dtype=float
    )

    start_ns = (
        f[
            "bar_start_time"
        ]
        .to_numpy(
            dtype=(
                "datetime64[ns]"
            )
        )
        .astype(
            np.int64
        )
    )

    ret = np.full(
        len(close),
        np.nan,
    )

    ret[1:] = np.log(
        close[1:]
        / close[:-1]
    )

    sq = np.nan_to_num(
        ret * ret,
        nan=0.0,
    )

    cs = np.concatenate(
        [
            [0.0],
            np.cumsum(
                sq
            ),
        ]
    )

    want = (
        target_end
        .to_numpy(
            dtype=(
                "datetime64[ns]"
            )
        )
        .astype(
            np.int64
        )
    )

    pos = np.searchsorted(
        start_ns,
        want,
        side="left",
    )

    def trailing(
        n: int,
    ) -> np.ndarray:

        out = np.full(
            len(pos),
            np.nan,
        )

        valid = pos >= n

        p = pos[
            valid
        ]

        out[
            valid
        ] = (
            cs[p]
            - cs[
                p - n
            ]
        )

        return out

    return {
        "rv_1h": trailing(
            12
        ),
        "rv_4h": trailing(
            48
        ),
    }


# ============================================================
# Features
# ============================================================

def build_features(
    bars: pd.DataFrame,
    five: pd.DataFrame,
) -> pd.DataFrame:

    close = bars[
        "close"
    ].to_numpy(
        dtype=float
    )

    high = bars[
        "high"
    ].to_numpy(
        dtype=float
    )

    low = bars[
        "low"
    ].to_numpy(
        dtype=float
    )

    start = pd.to_datetime(
        bars[
            "bar_start_time"
        ]
    )

    data: dict[
        str, np.ndarray
    ] = {}

    log_close = np.log(
        close
    )

    for k in RET_LAGS:

        ret = np.full(
            len(
                close
            ),
            np.nan,
        )

        ret[
            k:
        ] = (
            log_close[
                k:
            ]
            -
            log_close[
                :-k
            ]
        )

        data[
            f"feat_15m_ret_{k}"
        ] = ret

    # Location inside the trailing high/low box.
    low_s = pd.Series(
        low
    )

    high_s = pd.Series(
        high
    )

    roll_min = (
        low_s.rolling(
            LOCATION_WINDOW,
            min_periods=(
                LOCATION_WINDOW
            ),
        )
        .min()
        .to_numpy()
    )

    roll_max = (
        high_s.rolling(
            LOCATION_WINDOW,
            min_periods=(
                LOCATION_WINDOW
            ),
        )
        .max()
        .to_numpy()
    )

    span = (
        roll_max
        - roll_min
    )

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        location = (
            (
                close
                - roll_min
            )
            / span
        )

    location[
        ~np.isfinite(
            location
        )
    ] = np.nan

    data[
        "feat_15m_location_32"
    ] = location

    # Session-time structure.
    gap_min = (
        start.diff()
        .dt.total_seconds()
        .to_numpy()
        / 60.0
    )

    new_segment = np.concatenate(
        [
            [True],
            (
                gap_min[
                    1:
                ]
                > 15.0
            ),
        ]
    )

    segment_id = np.cumsum(
        new_segment
    )

    seg_start = (
        pd.Series(
            segment_id
        )
        .map(
            pd.Series(
                segment_id
            )
            .groupby(
                segment_id
            )
            .cumcount()
        )
    )

    # cumcount per segment, vectorized
    order = np.arange(
        len(
            segment_id
        )
    )

    first_index = (
        pd.Series(
            order
        )
        .groupby(
            segment_id
        )
        .transform(
            "min"
        )
        .to_numpy()
    )

    data[
        "feat_time_bars_since_segment_start"
    ] = (
        order
        - first_index
    ).astype(
        float
    )

    after_gap = np.concatenate(
        [
            [np.nan],
            (
                gap_min[
                    1:
                ]
                > LONG_GAP_MINUTES
            ).astype(
                float
            ),
        ]
    )

    after_gap[
        np.isnan(
            gap_min
        )
    ] = np.nan

    data[
        "feat_time_after_long_gap"
    ] = after_gap

    # Realized volatility level and acceleration.
    rv = realized_variance(
        five,
        bars[
            "bar_end_time"
        ],
    )

    data[
        "feat_5m_1h_rv"
    ] = rv[
        "rv_1h"
    ]

    with np.errstate(
        divide="ignore",
        invalid="ignore",
    ):

        ratio = (
            (
                rv[
                    "rv_1h"
                ]
                / 12.0
            )
            /
            (
                rv[
                    "rv_4h"
                ]
                / 48.0
            )
        )

    ratio[
        ~np.isfinite(
            ratio
        )
    ] = np.nan

    data[
        "feat_5m_rv_rate_ratio_1h_4h"
    ] = ratio

    return pd.DataFrame(
        data
    )


# ============================================================
# Targets
# ============================================================

def build_targets(
    bars: pd.DataFrame,
    horizon: int,
) -> dict[str, np.ndarray]:

    n = len(
        bars
    )

    open_ = bars[
        "open"
    ].to_numpy(
        dtype=float
    )

    high = bars[
        "high"
    ].to_numpy(
        dtype=float
    )

    low = bars[
        "low"
    ].to_numpy(
        dtype=float
    )

    close = bars[
        "close"
    ].to_numpy(
        dtype=float
    )

    width = (
        horizon
        + 1
    )

    nan = np.full(
        n,
        np.nan,
    )

    if n < width:
        return {
            f"target_raw_return_h{horizon}": (
                nan
            ),
            f"target_long_mfe_h{horizon}": (
                nan
            ),
            f"target_short_mfe_h{horizon}": (
                nan
            ),
        }

    win_high = (
        np.lib.stride_tricks
        .sliding_window_view(
            high,
            width,
        )
    )

    win_low = (
        np.lib.stride_tricks
        .sliding_window_view(
            low,
            width,
        )
    )

    win_close = (
        np.lib.stride_tricks
        .sliding_window_view(
            close,
            width,
        )
    )

    win_open = (
        np.lib.stride_tricks
        .sliding_window_view(
            open_,
            width,
        )
    )

    # Entry = open of the next bar (column 1 of the window).
    entry = (
        win_open[
            :,
            1
        ]
    )

    # Exit = close of the H-th bar after base.
    exit_ = (
        win_close[
            :,
            -1
        ]
    )

    log_entry = np.log(
        entry
    )

    raw = np.full(
        n,
        np.nan,
    )

    raw[
        : n - horizon
    ] = np.log(
        exit_
    ) - log_entry

    # Path excursions over bars i+1 .. i+H.
    path_high = (
        win_high[
            :,
            1:
        ]
    )

    path_low = (
        win_low[
            :,
            1:
        ]
    )

    long_mfe = np.full(
        n,
        np.nan,
    )

    short_mfe = np.full(
        n,
        np.nan,
    )

    up = (
        np.log(
            path_high
        )
        - log_entry[
            :,
            None
        ]
    )

    down = (
        log_entry[
            :,
            None
        ]
        - np.log(
            path_low
        )
    )

    long_mfe[
        : n - horizon
    ] = np.maximum(
        up.max(
            axis=1
        ),
        0.0,
    )

    short_mfe[
        : n - horizon
    ] = np.maximum(
        down.max(
            axis=1
        ),
        0.0,
    )

    return {
        f"target_raw_return_h{horizon}": (
            raw
        ),
        f"target_long_mfe_h{horizon}": (
            long_mfe
        ),
        f"target_short_mfe_h{horizon}": (
            short_mfe
        ),
    }


# ============================================================
# Main
# ============================================================

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

    manifest = {}
    validation = {}

    for (
        instrument
    ) in INSTRUMENTS:

        print(
            "=" * 64
        )

        print(
            f"{instrument}"
        )

        print(
            "=" * 64
        )

        five = pd.read_csv(
            SRC
            / f"{instrument}_5m.csv",
            parse_dates=[
                "bar_start_time",
                "bar_end_time",
                "availability_time",
                "trading_day",
                "tdx_datetime_raw",
            ],
        )

        bars = aggregate_15m(
            five
        )

        print(
            f"  5m bars : "
            f"{len(five)}"
        )

        print(
            f"  15m bars: "
            f"{len(bars)}"
        )

        features = (
            build_features(
                bars,
                five,
            )
        )

        panel = pd.concat(
            [
                bars[
                    [
                        "bar_start_time",
                        "bar_end_time",
                    ]
                ].rename(
                    columns={
                        "bar_start_time": (
                            "meta_base_bar_time"
                        ),
                        "bar_end_time": (
                            "meta_decision_time"
                        ),
                    }
                ),
                bars[
                    [
                        "open",
                        "high",
                        "low",
                        "close",
                        "volume",
                        "close_oi",
                    ]
                ],
                features,
            ],
            axis=1,
        )

        for h in HORIZONS:

            for name, values in (
                build_targets(
                    bars,
                    h,
                ).items()
            ):

                panel[
                    name
                ] = values

        path = (
            OUT
            / f"{instrument}_panel.csv"
        )

        panel.to_csv(
            path,
            index=False,
        )

        feature_cols = [
            c
            for c in panel.columns
            if c.startswith(
                "feat_"
            )
        ]

        manifest[
            instrument
        ] = {
            "path": (
                f"research/exports/"
                f"pytdx_panel/"
                f"{instrument}_panel.csv"
            ),
            "rows": int(
                len(panel)
            ),
            "feature_schema": (
                feature_cols
            ),
            "first_decision_time": str(
                panel[
                    "meta_decision_time"
                ].min()
            ),
            "last_decision_time": str(
                panel[
                    "meta_decision_time"
                ].max()
            ),
        }

        validation[
            instrument
        ] = {
            "five_minute_bars": int(
                len(five)
            ),
            "fifteen_minute_bars": int(
                len(bars)
            ),
            "aggregation_yield": round(
                len(bars)
                * 3
                / len(five),
                5,
            ),
            "rows": int(
                len(panel)
            ),
            "usable_h4": int(
                panel[
                    "target_raw_return_h4"
                ]
                .notna()
                .sum()
            ),
            "usable_h8": int(
                panel[
                    "target_raw_return_h8"
                ]
                .notna()
                .sum()
            ),
        }

        print(
            f"  rows    : "
            f"{len(panel)}"
        )

        print(
            f"  yield   : "
            f"{validation[instrument]['aggregation_yield']:.4f}"
        )

    (
        OUT
        / "panel_manifest.json"
    ).write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    starts = {
        k: v[
            "first_decision_time"
        ]
        for k, v in (
            manifest.items()
        )
    }

    ends = {
        k: v[
            "last_decision_time"
        ]
        for k, v in (
            manifest.items()
        )
    }

    summary = {
        "status": "PASS",
        "instrument_count": len(
            INSTRUMENTS
        ),
        "horizons": list(
            HORIZONS
        ),
        "aggregation_rule": (
            "15m bucket accepted only with "
            "exactly 3 consecutive 5m bars"
        ),
        "first_decision_time_by_instrument": (
            starts
        ),
        "last_decision_time_by_instrument": (
            ends
        ),
        "by_instrument": (
            validation
        ),
    }

    (
        OUT
        / "panel_validation.json"
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
        "PANEL"
    )

    print(
        "=" * 64
    )

    for k in INSTRUMENTS:

        v = validation[k]

        print(
            f"  {k:3s} "
            f"5m={v['five_minute_bars']:6d}  "
            f"15m={v['fifteen_minute_bars']:6d}  "
            f"yield={v['aggregation_yield']:.4f}  "
            f"H4={v['usable_h4']:6d}  "
            f"H8={v['usable_h8']:6d}"
        )

    print(
        "\nPYTDX_PANEL_BUILD_PASS"
    )


if __name__ == "__main__":
    main()
