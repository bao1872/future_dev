#!/usr/bin/env python3
"""Offline preparation of the Quantile V2 research panel.

Layout
------
  5m  = high-frequency observation layer
        realized variance, signed semivariance, bipower variation,
        jump proxy, gap contribution, volume/OI state

  15m = decision layer
        price / activity / volume / OI state and session-time structure

  target = future 15m-bar return over horizons 4 / 8 / 16  (1h / 2h / 4h)

This script never imports TqSdk and never touches the network.

Window note
-----------
The 5m serial is hard-capped at 10000 bars (~4.5 months), so the 15m
research window is re-based onto the range that the 5m series can cover
with full warmup. The exact range comes from
`5m_download_validation.json` written by download_silver_5m_tqsdk.py.

SMC / Momentum / BOS / CHoCH are deliberately absent.
The DP Oracle is a hindsight reference column only, never a feature.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

RAW_5M = (
    ROOT
    / "silver_main_data"
    / "silver_main_5m.csv"
)

RAW_15M = (
    ROOT
    / "silver_main_data"
    / "silver_main_15m.csv"
)

DOWNLOAD_VALIDATION = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_data"
    / "5m_download_validation.json"
)

OLD_RESEARCH = (
    ROOT
    / "research"
    / "exports"
    / "state_research"
    / "research_panel_15m.csv"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_data"
)

PANEL_PATH = (
    OUT
    / "quantile_panel_15m.csv"
)

DICTIONARY_PATH = (
    OUT
    / "feature_dictionary.csv"
)

SUMMARY_PATH = (
    OUT
    / "realized_measure_summary.csv"
)

VALIDATION_PATH = (
    OUT
    / "validation_summary.json"
)

MANIFEST_PATH = (
    OUT
    / "manifest.json"
)

HORIZONS_15M = (
    4,   # 1h
    8,   # 2h
    16,  # 4h
)

REALIZED_WINDOWS_5M = {
    12: "1h",
    24: "2h",
    48: "4h",
    96: "8h",
}

FIVE_MIN_NS = (
    5
    * 60
    * 1_000_000_000
)


def sha256_file(
    path: Path,
) -> str:

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


def safe_log_ratio(
    a,
    b,
):
    a = np.asarray(
        a,
        dtype=float,
    )

    b = np.asarray(
        b,
        dtype=float,
    )

    out = np.full(
        len(a),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(a)
        & np.isfinite(b)
        & (a > 0)
        & (b > 0)
    )

    out[valid] = np.log(
        a[valid]
        / b[valid]
    )

    return out


def rolling_location(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    window: int,
) -> pd.Series:

    hi = high.rolling(
        window,
        min_periods=window,
    ).max()

    lo = low.rolling(
        window,
        min_periods=window,
    ).min()

    den = (
        hi - lo
    ).replace(
        0.0,
        np.nan,
    )

    return (
        close - lo
    ) / den


def bars_since_segment_start(
    times: pd.Series,
) -> np.ndarray:

    diff_minutes = (
        times.diff()
        .dt.total_seconds()
        / 60.0
    )

    segment_start = (
        diff_minutes.isna()
        | (diff_minutes > 20.0)
    )

    out = np.zeros(
        len(times),
        dtype=int,
    )

    age = 0

    for i, is_start in enumerate(
        segment_start
    ):
        if bool(is_start):
            age = 0
        else:
            age += 1

        out[i] = age

    return out


def window_metrics(
    returns: np.ndarray,
    gap_flag: np.ndarray,
    volume: np.ndarray,
    open_oi: np.ndarray,
    close_oi: np.ndarray,
    end_pos: int,
    n: int,
) -> dict[str, float]:

    start = (
        end_pos
        - n
        + 1
    )

    if start < 1:
        return {}

    r = returns[
        start:
        end_pos + 1
    ]

    gaps = gap_flag[
        start:
        end_pos + 1
    ]

    vol = volume[
        start:
        end_pos + 1
    ]

    if (
        len(r) != n
        or not np.isfinite(
            r
        ).all()
    ):
        return {}

    rv = float(
        np.sum(
            r * r
        )
    )

    pos = r[
        r > 0
    ]

    neg = r[
        r < 0
    ]

    rsv_pos = float(
        np.sum(
            pos * pos
        )
    )

    rsv_neg = float(
        np.sum(
            neg * neg
        )
    )

    # Gap component.
    gap_sq = float(
        np.sum(
            (r * r)[
                gaps
            ]
        )
    )

    continuous = (
        ~gaps
    )

    rv_cont = float(
        np.sum(
            (r * r)[
                continuous
            ]
        )
    )

    # Bipower variation using only adjacent,
    # non-gap contaminated returns.
    pair_ok = (
        (~gaps[1:])
        &
        (~gaps[:-1])
    )

    if pair_ok.any():
        bpv = float(
            (math.pi / 2.0)
            * np.sum(
                np.abs(
                    r[1:][
                        pair_ok
                    ]
                )
                *
                np.abs(
                    r[:-1][
                        pair_ok
                    ]
                )
            )
        )
    else:
        bpv = float("nan")

    jump_var = (
        max(
            rv_cont - bpv,
            0.0,
        )
        if np.isfinite(
            bpv
        )
        else float("nan")
    )

    jump_share = (
        jump_var
        / rv_cont
        if (
            np.isfinite(
                jump_var
            )
            and rv_cont > 0
        )
        else float("nan")
    )

    first_oi = float(
        open_oi[start]
    )

    last_oi = float(
        close_oi[end_pos]
    )

    oi_change = (
        math.log(
            last_oi
            / first_oi
        )
        if (
            first_oi > 0
            and last_oi > 0
        )
        else float("nan")
    )

    mean_volume = float(
        np.mean(
            vol
        )
    )

    volume_cv = (
        float(
            np.std(
                vol,
                ddof=0,
            )
            / mean_volume
        )
        if mean_volume > 0
        else float("nan")
    )

    total_semivar = (
        rsv_pos
        + rsv_neg
    )

    neg_semivar_share = (
        rsv_neg
        / total_semivar
        if total_semivar > 0
        else float("nan")
    )

    return {
        "ret_sum": float(
            np.sum(r)
        ),

        "rv": rv,

        "realized_vol": (
            math.sqrt(rv)
        ),

        "rsv_pos": rsv_pos,

        "rsv_neg": rsv_neg,

        "signed_semivariance": (
            rsv_pos
            - rsv_neg
        ),

        "neg_semivar_share": (
            neg_semivar_share
        ),

        "gap_sq": gap_sq,

        "rv_continuous": (
            rv_cont
        ),

        "bpv_continuous": (
            bpv
        ),

        "jump_var": (
            jump_var
        ),

        "jump_share": (
            jump_share
        ),

        "max_abs_return": float(
            np.max(
                np.abs(r)
            )
        ),

        "positive_return_share": float(
            np.mean(
                r > 0
            )
        ),

        "log_volume_mean": float(
            np.log1p(
                mean_volume
            )
        ),

        "volume_cv": (
            volume_cv
        ),

        "oi_log_change": (
            oi_change
        ),
    }


def build_15m_features(
    fifteen: pd.DataFrame,
) -> pd.DataFrame:

    close = (
        fifteen["close"]
        .astype(float)
    )

    open_ = (
        fifteen["open"]
        .astype(float)
    )

    high = (
        fifteen["high"]
        .astype(float)
    )

    low = (
        fifteen["low"]
        .astype(float)
    )

    volume = (
        fifteen["volume"]
        .astype(float)
    )

    oi = (
        fifteen["close_oi"]
        .astype(float)
    )

    out = pd.DataFrame(
        index=fifteen.index
    )

    log_close = np.log(
        close
    )

    for k in (
        1,
        2,
        4,
        8,
        16,
        32,
    ):
        out[
            f"feat_15m_ret_{k}"
        ] = (
            log_close
            - log_close.shift(k)
        )

    out[
        "feat_15m_range_pct"
    ] = (
        high - low
    ) / close.replace(
        0.0,
        np.nan,
    )

    out[
        "feat_15m_body_pct"
    ] = (
        close - open_
    ) / open_.replace(
        0.0,
        np.nan,
    )

    den = (
        high - low
    ).replace(
        0.0,
        np.nan,
    )

    out[
        "feat_15m_bar_close_location"
    ] = (
        close - low
    ) / den

    for window in (
        16,
        32,
        64,
    ):
        out[
            f"feat_15m_location_{window}"
        ] = rolling_location(
            close,
            high,
            low,
            window,
        )

        mean = close.rolling(
            window,
            min_periods=window,
        ).mean()

        out[
            f"feat_15m_vs_mean_{window}"
        ] = (
            close
            / mean
            - 1.0
        )

    vol_mean32 = (
        volume.rolling(
            32,
            min_periods=32,
        ).mean()
    )

    out[
        "feat_15m_volume_ratio_32"
    ] = (
        volume
        / vol_mean32.replace(
            0.0,
            np.nan,
        )
    )

    for k in (
        1,
        4,
        16,
    ):
        out[
            f"feat_15m_oi_log_change_{k}"
        ] = (
            np.log(
                oi.replace(
                    0.0,
                    np.nan,
                )
            )
            -
            np.log(
                oi.shift(
                    k
                ).replace(
                    0.0,
                    np.nan,
                )
            )
        )

    return out


def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not RAW_5M.is_file():
        raise RuntimeError(
            "silver_main_5m.csv missing"
        )

    if not RAW_15M.is_file():
        raise RuntimeError(
            "silver_main_15m.csv missing"
        )

    if not DOWNLOAD_VALIDATION.is_file():
        raise RuntimeError(
            "5m_download_validation.json "
            "missing; run "
            "download_silver_5m_tqsdk.py "
            "first"
        )

    download_info = json.loads(
        DOWNLOAD_VALIDATION.read_text(
            encoding="utf-8"
        )
    )

    alignment = download_info[
        "aggregation"
    ]

    first_index = int(
        alignment[
            "first_15m_index"
        ]
    )

    last_index = int(
        alignment[
            "last_15m_index"
        ]
    )

    five = pd.read_csv(
        RAW_5M,
        parse_dates=[
            "datetime"
        ],
        low_memory=False,
    )

    fifteen = pd.read_csv(
        RAW_15M,
        parse_dates=[
            "datetime"
        ],
        low_memory=False,
    )

    five = (
        five
        .sort_values(
            "datetime_ns"
        )
        .reset_index(
            drop=True
        )
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
            f"15m rows changed: "
            f"{len(fifteen)}"
        )

    # --------------------------------------------------
    # Re-base the decision layer onto the 5m coverage.
    # Rolling 15m features still read the FULL history so
    # that windows at first_index are causally complete.
    # --------------------------------------------------

    if not (
        0
        <= first_index
        <= last_index
        < len(fifteen)
    ):
        raise RuntimeError(
            "Alignment indices out of range"
        )

    fifteen_a = (
        fifteen
        .iloc[
            first_index:
            last_index + 1
        ]
        .reset_index(
            drop=True
        )
    )

    aligned_len = len(
        fifteen_a
    )

    base_count = (
        aligned_len
        - 1
    )

    if base_count < 2:
        raise RuntimeError(
            "Aligned window too small"
        )

    base = (
        fifteen_a
        .iloc[
            :base_count
        ]
        .copy()
    )

    decision = (
        fifteen_a
        .iloc[
            1:
            base_count + 1
        ]
        .copy()
    )

    # --------------------------------------------------
    # 5m primitive series
    # --------------------------------------------------

    five_close = (
        five["close"]
        .to_numpy(
            dtype=float
        )
    )

    five_volume = (
        five["volume"]
        .to_numpy(
            dtype=float
        )
    )

    five_open_oi = (
        five["open_oi"]
        .to_numpy(
            dtype=float
        )
    )

    five_close_oi = (
        five["close_oi"]
        .to_numpy(
            dtype=float
        )
    )

    five_times_ns = (
        five["datetime_ns"]
        .to_numpy(
            dtype=np.int64
        )
    )

    five_ret = np.full(
        len(five),
        np.nan,
        dtype=float,
    )

    five_ret[1:] = np.log(
        five_close[1:]
        / five_close[:-1]
    )

    time_diff_ns = np.full(
        len(five),
        0,
        dtype=np.int64,
    )

    time_diff_ns[1:] = (
        five_times_ns[1:]
        - five_times_ns[:-1]
    )

    # Normal contiguous interval = 5 minutes.
    # Anything >10 minutes is treated as a
    # session/break gap return.
    five_gap = (
        time_diff_ns
        > 10
        * 60
        * 1_000_000_000
    )

    # --------------------------------------------------
    # Base 15m decision rows
    # Last aligned 15m bar has no next decision time.
    # --------------------------------------------------

    panel = pd.DataFrame(
        {
            "meta_base_bar_index": (
                np.arange(
                    first_index,
                    first_index
                    + base_count,
                    dtype=int,
                )
            ),

            "meta_base_bar_time": (
                base[
                    "datetime"
                ].to_numpy()
            ),

            "meta_decision_time": (
                decision[
                    "datetime"
                ].to_numpy()
            ),

            "meta_entry_bar_index": (
                np.arange(
                    first_index + 1,
                    first_index
                    + 1
                    + base_count,
                    dtype=int,
                )
            ),
        }
    )

    # --------------------------------------------------
    # 15m direct statistical state
    # --------------------------------------------------

    feat15 = (
        build_15m_features(
            fifteen
        )
        .iloc[
            first_index:
            first_index
            + base_count
        ]
        .reset_index(
            drop=True
        )
    )

    panel = pd.concat(
        [
            panel.reset_index(
                drop=True
            ),
            feat15,
        ],
        axis=1,
    )

    # --------------------------------------------------
    # Generic session/time descriptors
    # --------------------------------------------------

    decision_time = pd.to_datetime(
        panel[
            "meta_decision_time"
        ]
    )

    base_time = pd.to_datetime(
        panel[
            "meta_base_bar_time"
        ]
    )

    minute_of_day = (
        decision_time.dt.hour
        * 60
        + decision_time.dt.minute
    )

    dow = (
        decision_time.dt.dayofweek
    )

    panel[
        "feat_time_tod_sin"
    ] = np.sin(
        2.0
        * np.pi
        * minute_of_day
        / 1440.0
    )

    panel[
        "feat_time_tod_cos"
    ] = np.cos(
        2.0
        * np.pi
        * minute_of_day
        / 1440.0
    )

    panel[
        "feat_time_dow_sin"
    ] = np.sin(
        2.0
        * np.pi
        * dow
        / 7.0
    )

    panel[
        "feat_time_dow_cos"
    ] = np.cos(
        2.0
        * np.pi
        * dow
        / 7.0
    )

    decision_gap_minutes = (
        decision_time
        - base_time
    ).dt.total_seconds() / 60.0

    panel[
        "feat_time_decision_gap_minutes"
    ] = (
        decision_gap_minutes
    )

    panel[
        "feat_time_after_long_gap"
    ] = (
        decision_gap_minutes
        > 20.0
    ).astype(int)

    fifteen_times = pd.to_datetime(
        fifteen[
            "datetime"
        ]
    )

    segment_age_full = (
        bars_since_segment_start(
            fifteen_times
        )
    )

    panel[
        "feat_time_bars_since_segment_start"
    ] = segment_age_full[
        first_index:
        first_index
        + base_count
    ]

    # --------------------------------------------------
    # Map each decision time to latest completed 5m bar
    # --------------------------------------------------

    decision_ns = (
        decision[
            "datetime_ns"
        ]
        .to_numpy(
            dtype=np.int64
        )
    )

    last_5m_pos = (
        np.searchsorted(
            five_times_ns,
            decision_ns,
            side="left",
        )
        - 1
    )

    if (
        last_5m_pos < 0
    ).any():
        raise RuntimeError(
            "No 5m history before some decision"
        )

    feature_last_5m_ns = (
        five_times_ns[
            last_5m_pos
        ]
    )

    if not (
        feature_last_5m_ns
        < decision_ns
    ).all():
        raise RuntimeError(
            "5m feature time leaks into decision"
        )

    panel[
        "meta_feature_last_5m_time"
    ] = pd.to_datetime(
        feature_last_5m_ns,
        unit="ns",
        utc=True,
    ).tz_convert(
        "Asia/Shanghai"
    ).tz_localize(
        None
    )

    # --------------------------------------------------
    # Realized measures from trailing 5m returns
    # --------------------------------------------------

    realized_columns = []

    for n5, label in (
        REALIZED_WINDOWS_5M.items()
    ):

        metrics_per_row = []

        for end_pos in last_5m_pos:

            metrics_per_row.append(
                window_metrics(
                    five_ret,
                    five_gap,
                    five_volume,
                    five_open_oi,
                    five_close_oi,
                    int(end_pos),
                    int(n5),
                )
            )

        metric_names = sorted(
            {
                key
                for row in metrics_per_row
                for key in row
            }
        )

        for metric in metric_names:

            col = (
                f"feat_5m_{label}_"
                f"{metric}"
            )

            panel[col] = [
                row.get(
                    metric,
                    np.nan,
                )
                for row in (
                    metrics_per_row
                )
            ]

            realized_columns.append(
                col
            )

    # --------------------------------------------------
    # Cross-scale realized-volatility ratios
    # HAR-style multi-scale state
    # --------------------------------------------------

    def rv_rate(
        label: str,
        n: int,
    ):
        return (
            panel[
                f"feat_5m_{label}_rv"
            ]
            / float(n)
        )

    panel[
        "feat_5m_rv_rate_ratio_1h_4h"
    ] = (
        rv_rate(
            "1h",
            12,
        )
        /
        rv_rate(
            "4h",
            48,
        ).replace(
            0.0,
            np.nan,
        )
    )

    panel[
        "feat_5m_rv_rate_ratio_2h_4h"
    ] = (
        rv_rate(
            "2h",
            24,
        )
        /
        rv_rate(
            "4h",
            48,
        ).replace(
            0.0,
            np.nan,
        )
    )

    panel[
        "feat_5m_rv_rate_ratio_4h_8h"
    ] = (
        rv_rate(
            "4h",
            48,
        )
        /
        rv_rate(
            "8h",
            96,
        ).replace(
            0.0,
            np.nan,
        )
    )

    # --------------------------------------------------
    # Future returns for quantile forecasting.
    #
    # Features known at decision_time.
    # Entry = next observed 15m open.
    # Exit = close of H-th 15m bar after base.
    # --------------------------------------------------

    open15 = (
        fifteen_a[
            "open"
        ].to_numpy(
            dtype=float
        )
    )

    close15 = (
        fifteen_a[
            "close"
        ].to_numpy(
            dtype=float
        )
    )

    high15 = (
        fifteen_a[
            "high"
        ].to_numpy(
            dtype=float
        )
    )

    low15 = (
        fifteen_a[
            "low"
        ].to_numpy(
            dtype=float
        )
    )

    for h in HORIZONS_15M:

        raw_return = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        norm_return = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        long_mfe = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        long_mae = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        short_mfe = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        short_mae = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        for i in range(
            base_count
        ):
            entry_i = (
                i + 1
            )

            exit_i = (
                i + h
            )

            if exit_i >= len(
                fifteen_a
            ):
                continue

            entry_price = float(
                open15[
                    entry_i
                ]
            )

            exit_price = float(
                close15[
                    exit_i
                ]
            )

            raw_return[i] = (
                math.log(
                    exit_price
                    / entry_price
                )
            )

            window_high = float(
                np.max(
                    high15[
                        entry_i:
                        exit_i + 1
                    ]
                )
            )

            window_low = float(
                np.min(
                    low15[
                        entry_i:
                        exit_i + 1
                    ]
                )
            )

            long_mfe[i] = math.log(
                window_high
                / entry_price
            )

            long_mae[i] = math.log(
                window_low
                / entry_price
            )

            short_mfe[i] = math.log(
                entry_price
                / window_low
            )

            short_mae[i] = math.log(
                entry_price
                / window_high
            )

        panel[
            f"target_raw_return_h{h}"
        ] = raw_return

        # Scale by causal 4h 5m realized variance.
        rv4 = panel[
            "feat_5m_4h_rv"
        ].to_numpy(
            dtype=float
        )

        # 4h RV = 48 x 5m observations.
        # H x 15m = 3H x 5m observations.
        scale = np.sqrt(
            (
                rv4 / 48.0
            )
            * (
                3.0 * h
            )
        )

        valid_scale = (
            np.isfinite(
                raw_return
            )
            &
            np.isfinite(
                scale
            )
            &
            (
                scale
                > 1e-12
            )
        )

        norm_return[
            valid_scale
        ] = (
            raw_return[
                valid_scale
            ]
            /
            scale[
                valid_scale
            ]
        )

        panel[
            f"target_norm_return_h{h}"
        ] = norm_return

        # Evaluation references only.
        panel[
            f"target_long_mfe_h{h}"
        ] = long_mfe

        panel[
            f"target_long_mae_h{h}"
        ] = long_mae

        panel[
            f"target_short_mfe_h{h}"
        ] = short_mfe

        panel[
            f"target_short_mae_h{h}"
        ] = short_mae

    # --------------------------------------------------
    # Optional DP Oracle reference.
    # Never a feature.
    # --------------------------------------------------

    if OLD_RESEARCH.is_file():

        old = pd.read_csv(
            OLD_RESEARCH,
            usecols=[
                "base_bar_index",
                "target_15m__oracle_consensus",
                "target_15m__oracle_unanimous_direction",
            ],
            low_memory=False,
        )

        old = (
            old
            .sort_values(
                "base_bar_index"
            )
            .reset_index(
                drop=True
            )
        )

        mask = (
            old[
                "base_bar_index"
            ]
            >= first_index
        ) & (
            old[
                "base_bar_index"
            ]
            < first_index
            + base_count
        )

        old = old.loc[
            mask
        ].reset_index(
            drop=True
        )

        if len(old) != base_count:
            raise RuntimeError(
                "Old research panel "
                "row mismatch: "
                f"{len(old)} != "
                f"{base_count}"
            )

        if not np.array_equal(
            old[
                "base_bar_index"
            ].to_numpy(
                dtype=int
            ),
            np.arange(
                first_index,
                first_index
                + base_count,
                dtype=int,
            ),
        ):
            raise RuntimeError(
                "Old Oracle reference "
                "bar index mismatch"
            )

        panel[
            "reference_dp_oracle_consensus"
        ] = old[
            "target_15m__oracle_consensus"
        ].to_numpy()

        panel[
            "reference_dp_oracle_unanimous"
        ] = old[
            "target_15m__oracle_unanimous_direction"
        ].to_numpy()

    # --------------------------------------------------
    # Leakage / integrity validation
    # --------------------------------------------------

    feature_cols = [
        col
        for col in panel.columns
        if col.startswith(
            "feat_"
        )
    ]

    forbidden = (
        "smc",
        "momentum",
        "sqz",
        "oracle",
        "target",
    )

    bad_features = [
        col
        for col in feature_cols
        if any(
            token
            in col.lower()
            for token in forbidden
        )
    ]

    if bad_features:
        raise RuntimeError(
            "Forbidden features: "
            f"{bad_features}"
        )

    numeric = (
        panel
        .select_dtypes(
            include=[
                np.number
            ]
        )
    )

    if np.isinf(
        numeric.to_numpy(
            dtype=float
        )
    ).any():
        raise RuntimeError(
            "Panel contains +/-inf"
        )

    if not (
        pd.to_datetime(
            panel[
                "meta_feature_last_5m_time"
            ]
        )
        <
        pd.to_datetime(
            panel[
                "meta_decision_time"
            ]
        )
    ).all():
        raise RuntimeError(
            "5m feature leakage detected"
        )

    expected_valid_targets = {}

    for h in HORIZONS_15M:

        valid_count = int(
            panel[
                f"target_raw_return_h{h}"
            ].notna().sum()
        )

        expected = (
            aligned_len
            - h
        )

        if valid_count != expected:
            raise RuntimeError(
                f"h={h}: target count "
                f"{valid_count} != "
                f"{expected}"
            )

        expected_valid_targets[
            str(h)
        ] = valid_count

    # Need full 96-bar high-frequency history
    # from the very first research decision.
    critical = [
        "feat_5m_8h_rv",
        "feat_5m_8h_rsv_pos",
        "feat_5m_8h_rsv_neg",
    ]

    if panel.iloc[0][
        critical
    ].isna().any():
        raise RuntimeError(
            "5m warmup insufficient "
            "for first research row"
        )

    # --------------------------------------------------
    # Persist panel
    # --------------------------------------------------

    panel.to_csv(
        PANEL_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Feature dictionary
    # --------------------------------------------------

    dictionary_rows = []

    for col in panel.columns:

        if col.startswith(
            "feat_"
        ):
            role = "FEATURE"

        elif col.startswith(
            "target_"
        ):
            role = "TARGET"

        elif col.startswith(
            "reference_"
        ):
            role = (
                "HINDSIGHT_REFERENCE"
            )

        else:
            role = "METADATA"

        if col.startswith(
            "feat_5m_"
        ):
            family = (
                "5m_realized_state"
            )

        elif col.startswith(
            "feat_15m_"
        ):
            family = (
                "15m_price_activity"
            )

        elif col.startswith(
            "feat_time_"
        ):
            family = (
                "time_session"
            )

        elif col.startswith(
            "target_"
        ):
            family = (
                "future_outcome"
            )

        elif col.startswith(
            "reference_"
        ):
            family = (
                "oracle_reference"
            )

        else:
            family = "metadata"

        dictionary_rows.append(
            {
                "column": col,
                "role": role,
                "family": family,
                "causal_feature": (
                    role == "FEATURE"
                ),
            }
        )

    dictionary = pd.DataFrame(
        dictionary_rows
    )

    dictionary.to_csv(
        DICTIONARY_PATH,
        index=False,
    )

    # --------------------------------------------------
    # Realized-measure descriptive summary
    # --------------------------------------------------

    realized_feature_cols = [
        col
        for col in feature_cols
        if col.startswith(
            "feat_5m_"
        )
    ]

    summary_rows = []

    for col in (
        realized_feature_cols
    ):
        s = pd.to_numeric(
            panel[col],
            errors="coerce",
        )

        x = s.dropna()

        summary_rows.append(
            {
                "feature": col,
                "non_na": int(
                    len(x)
                ),
                "mean": float(
                    x.mean()
                ),
                "std": float(
                    x.std(
                        ddof=0
                    )
                ),
                "p01": float(
                    x.quantile(
                        0.01
                    )
                ),
                "p25": float(
                    x.quantile(
                        0.25
                    )
                ),
                "p50": float(
                    x.quantile(
                        0.50
                    )
                ),
                "p75": float(
                    x.quantile(
                        0.75
                    )
                ),
                "p99": float(
                    x.quantile(
                        0.99
                    )
                ),
            }
        )

    realized_summary = (
        pd.DataFrame(
            summary_rows
        )
    )

    realized_summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    validation = {
        "panel_rows": int(
            len(panel)
        ),
        "panel_columns": int(
            len(
                panel.columns
            )
        ),
        "feature_count": int(
            len(
                feature_cols
            )
        ),
        "realized_feature_count": int(
            len(
                realized_feature_cols
            )
        ),

        "aligned_15m_window": {
            "first_15m_index": (
                first_index
            ),
            "last_15m_index": (
                last_index
            ),
            "aligned_15m_bars": (
                aligned_len
            ),
            "start": str(
                fifteen_a.iloc[
                    0
                ][
                    "datetime"
                ]
            ),
            "end": str(
                fifteen_a.iloc[
                    -1
                ][
                    "datetime"
                ]
            ),
            "base_rows": int(
                base_count
            ),
        },

        "feature_uses_smc": False,
        "feature_uses_momentum": False,
        "feature_uses_oracle": False,
        "five_minute_feature_causality": True,
        "future_target_valid_rows": (
            expected_valid_targets
        ),
        "no_infinity": True,
    }

    VALIDATION_PATH.write_text(
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "purpose": (
            "5m realized-state + 15m "
            "decision dataset for "
            "short-horizon conditional "
            "quantile forecasting."
        ),

        "instrument": (
            "KQ.m@SHFE.ag"
        ),

        "decision_timeframe": (
            "15m"
        ),

        "high_frequency_source": (
            "5m"
        ),

        "window_rebasing": {
            "reason": (
                "TqSdk 5m serial is hard-capped "
                "at 10000 bars (~4.5 months), "
                "shorter than the current 15m "
                "file (~13.5 months). The 15m "
                "research window is re-based "
                "onto the range the 5m series "
                "can cover with >=96 bars of "
                "high-frequency warmup."
            ),
            "first_15m_index": (
                first_index
            ),
            "last_15m_index": (
                last_index
            ),
            "aligned_15m_bars": (
                aligned_len
            ),
            "15m_bars_available": int(
                len(fifteen)
            ),
        },

        "prediction_horizons": {
            "4": "1h",
            "8": "2h",
            "16": "4h",
        },

        "realized_windows_5m": {
            str(k): v
            for k, v in (
                REALIZED_WINDOWS_5M.items()
            )
        },

        "target_semantics": {
            "decision": (
                "next observed 15m "
                "bar start"
            ),

            "entry_price": (
                "next observed 15m "
                "bar open"
            ),

            "exit_price": (
                "close at base index "
                "+ horizon"
            ),

            "quantile_target": (
                "future raw log return"
            ),

            "mfe_mae": (
                "evaluation reference "
                "only"
            ),

            "dp_oracle": (
                "hindsight reference "
                "only, never feature"
            ),
        },

        "feature_semantics": {
            "SMC": False,
            "Momentum": False,
            "BOS_CHoCH": False,

            "families": [
                "15m returns / geometry",
                "15m volume / OI",
                "5m realized variance",
                "5m positive / negative semivariance",
                "5m bipower variation",
                "5m jump proxy",
                "5m volume / OI state",
                "generic session/time state",
            ],
        },

        "sources": {
            "silver_main_5m.csv": {
                "rows": int(
                    len(five)
                ),
                "sha256": sha256_file(
                    RAW_5M
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
            },

            "silver_main_15m.csv": {
                "rows": int(
                    len(fifteen)
                ),
                "rows_used": int(
                    aligned_len
                ),
                "sha256": sha256_file(
                    RAW_15M
                ),
                "start": str(
                    fifteen.iloc[0][
                        "datetime"
                    ]
                ),
                "end": str(
                    fifteen.iloc[-1][
                        "datetime"
                    ]
                ),
            },
        },

        "outputs": {
            PANEL_PATH.name: {
                "rows": int(
                    len(panel)
                ),
                "columns": int(
                    len(
                        panel.columns
                    )
                ),
                "sha256": sha256_file(
                    PANEL_PATH
                ),
            },

            DICTIONARY_PATH.name: {
                "rows": int(
                    len(
                        dictionary
                    )
                ),
                "sha256": sha256_file(
                    DICTIONARY_PATH
                ),
            },

            SUMMARY_PATH.name: {
                "rows": int(
                    len(
                        realized_summary
                    )
                ),
                "sha256": sha256_file(
                    SUMMARY_PATH
                ),
            },
        },

        "validation": validation,
    }

    MANIFEST_PATH.write_text(
        json.dumps(
            manifest,
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
        "QUANTILE_V2_DATA_PREP_PASS"
    )


if __name__ == "__main__":
    main()
