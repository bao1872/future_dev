#!/usr/bin/env python3
"""Build compact multi-instrument panels for Quantile V2 robustness research.

Only the 11 F3_VOL features plus the three future-return horizons and their
MFE/MAE evaluation references. No SMC, no Momentum, no Oracle, no
semivariance / jump, no 1h or 4h K-line series.

Feature and target definitions are imported from
`prepare_quantile_v2_data.py` so the semantics are byte-identical to the
Silver Quantile V2 panel; nothing is re-derived here.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

if str(
    ROOT
) not in sys.path:
    sys.path.insert(
        0,
        str(ROOT),
    )

from research.prepare_quantile_v2_data import (  # noqa: E402
    bars_since_segment_start,
    rolling_location,
    sha256_file,
    window_metrics,
)


RAW_ROOT = (
    ROOT
    / "research"
    / "robustness_data"
    / "raw"
)

DOWNLOAD_INDEX = (
    RAW_ROOT
    / "download_index.json"
)

OUT = (
    ROOT
    / "research"
    / "exports"
    / "quantile_v2_robustness_data"
)


# ============================================================
# Locked model schema
# ============================================================

MODEL_FEATURES = [
    "feat_15m_ret_1",
    "feat_15m_ret_4",
    "feat_15m_ret_8",
    "feat_15m_ret_16",

    "feat_15m_location_32",

    "feat_time_bars_since_segment_start",
    "feat_time_after_long_gap",

    "feat_15m_volume_ratio_32",
    "feat_15m_oi_log_change_4",

    "feat_5m_1h_rv",
    "feat_5m_rv_rate_ratio_1h_4h",
]

HORIZONS = (4, 8, 16)

# 12 x 5m = 1h, 48 x 5m = 4h
REALIZED_WINDOWS = {
    "1h": 12,
    "4h": 48,
}

MIN_5M_ROWS = 9500

MIN_PANEL_ROWS = 3000

# Instruments that must PASS, otherwise the whole round fails.
REQUIRED_INSTRUMENTS = {
    "AG",
    "CU",
    "I",
    "SC",
    "M",
}

MIN_PASS_INSTRUMENTS = 6

TEN_MIN_NS = (
    10
    * 60
    * 1_000_000_000
)

FORBIDDEN_TOKENS = (
    "smc",
    "momentum",
    "sqz",
    "oracle",
    "target",
    "semivar",
    "jump",
)


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
        4,
        8,
        16,
    ):
        out[
            f"feat_15m_ret_{k}"
        ] = (
            log_close
            - log_close.shift(k)
        )

    out[
        "feat_15m_location_32"
    ] = rolling_location(
        close,
        high,
        low,
        32,
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

    oi_safe = (
        oi.replace(
            0.0,
            np.nan,
        )
    )

    out[
        "feat_15m_oi_log_change_4"
    ] = (
        np.log(
            oi_safe
        )
        -
        np.log(
            oi_safe.shift(
                4
            )
        )
    )

    return out


def build_panel(
    code: str,
    record: dict,
) -> tuple[pd.DataFrame, dict]:

    five_path = (
        ROOT
        / record[
            "five_minute"
        ][
            "path"
        ]
    )

    fifteen_path = (
        ROOT
        / record[
            "fifteen_minute"
        ][
            "path"
        ]
    )

    five = pd.read_csv(
        five_path,
        parse_dates=[
            "datetime"
        ],
        low_memory=False,
    )

    fifteen = pd.read_csv(
        fifteen_path,
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

    aggregation = record[
        "aggregation"
    ]

    first_index = int(
        aggregation[
            "first_15m_index"
        ]
    )

    last_index = int(
        aggregation[
            "last_15m_index"
        ]
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

    # --------------------------------------------------------
    # 5m primitive series
    # --------------------------------------------------------

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

    # Same rule as prepare_quantile_v2_data.py:
    # anything >10 minutes is a session/break gap return.
    five_gap = (
        time_diff_ns
        > TEN_MIN_NS
    )

    # --------------------------------------------------------
    # Base rows
    # --------------------------------------------------------

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
        }
    )

    # --------------------------------------------------------
    # 15m features (history read from the FULL 15m series)
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Session / time descriptors
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Map each decision to the latest completed 5m bar
    # --------------------------------------------------------

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
            f"{code}: no 5m history "
            "before some decision"
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
            f"{code}: 5m feature "
            "leaks into decision"
        )

    # --------------------------------------------------------
    # Realized variance from trailing 5m returns
    # --------------------------------------------------------

    rv_columns = {}

    for label, n5 in (
        REALIZED_WINDOWS.items()
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

        rv_columns[
            label
        ] = [
            row.get(
                "rv",
                np.nan,
            )
            for row in (
                metrics_per_row
            )
        ]

    panel[
        "feat_5m_1h_rv"
    ] = rv_columns[
        "1h"
    ]

    panel[
        "feat_5m_4h_rv"
    ] = rv_columns[
        "4h"
    ]

    rv_rate_1h = (
        panel[
            "feat_5m_1h_rv"
        ]
        / float(
            REALIZED_WINDOWS[
                "1h"
            ]
        )
    )

    rv_rate_4h = (
        panel[
            "feat_5m_4h_rv"
        ]
        / float(
            REALIZED_WINDOWS[
                "4h"
            ]
        )
    )

    panel[
        "feat_5m_rv_rate_ratio_1h_4h"
    ] = (
        rv_rate_1h
        /
        rv_rate_4h.replace(
            0.0,
            np.nan,
        )
    )

    # --------------------------------------------------------
    # Future returns + MFE / MAE references
    # --------------------------------------------------------

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

    target_counts = {}

    for h in HORIZONS:

        raw_return = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        long_mfe = np.full(
            base_count,
            np.nan,
            dtype=float,
        )

        short_mfe = np.full(
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

            short_mfe[i] = math.log(
                entry_price
                / window_low
            )

        panel[
            f"target_raw_return_h{h}"
        ] = raw_return

        panel[
            f"target_long_mfe_h{h}"
        ] = long_mfe

        panel[
            f"target_short_mfe_h{h}"
        ] = short_mfe

        target_counts[
            str(h)
        ] = int(
            np.isfinite(
                raw_return
            ).sum()
        )

    # --------------------------------------------------------
    # Leakage / integrity validation
    # --------------------------------------------------------

    # The forbidden-token guard applies to FEATURE columns only.
    # Target columns legitimately carry the "target_" prefix and
    # MFE/MAE references are evaluation-only references.
    feature_cols = [
        col
        for col in panel.columns
        if col.startswith(
            "feat_"
        )
    ]

    forbidden_hits = [
        col
        for col in feature_cols
        if any(
            token
            in col.lower()
            for token in FORBIDDEN_TOKENS
        )
    ]

    if forbidden_hits:
        raise RuntimeError(
            f"{code}: forbidden "
            f"feature columns: "
            f"{forbidden_hits}"
        )

    # Targets must exist and must be the only target_* columns.
    expected_targets = {
        f"target_raw_return_h{h}"
        for h in HORIZONS
    } | {
        f"target_long_mfe_h{h}"
        for h in HORIZONS
    } | {
        f"target_short_mfe_h{h}"
        for h in HORIZONS
    }

    actual_targets = {
        col
        for col in panel.columns
        if col.startswith(
            "target_"
        )
    }

    if (
        actual_targets
        != expected_targets
    ):
        raise RuntimeError(
            f"{code}: unexpected "
            f"target columns: "
            f"{sorted(actual_targets)}"
        )

    numeric = (
        panel.select_dtypes(
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
            f"{code}: +/-inf"
        )

    for feature in MODEL_FEATURES:

        if feature not in (
            panel.columns
        ):
            raise RuntimeError(
                f"{code}: missing "
                f"{feature}"
            )

    critical = [
        "feat_5m_1h_rv",
        "feat_5m_4h_rv",
    ]

    if panel.iloc[0][
        critical
    ].isna().any():
        raise RuntimeError(
            f"{code}: 5m warmup "
            "insufficient for "
            "first row"
        )

    sufficient = (
        len(five)
        >= MIN_5M_ROWS
        and len(panel)
        >= MIN_PANEL_ROWS
    )

    summary = {
        "instrument": code,

        "symbol": record[
            "symbol"
        ],

        "source": record[
            "source"
        ],

        "5m_rows": int(
            len(five)
        ),

        "5m_start": str(
            five.iloc[0][
                "datetime"
            ]
        ),

        "5m_end": str(
            five.iloc[-1][
                "datetime"
            ]
        ),

        "15m_total_rows": int(
            len(fifteen)
        ),

        "usable_15m_rows": int(
            aligned_len
        ),

        "panel_rows": int(
            len(panel)
        ),

        "target_h4_rows": (
            target_counts[
                "4"
            ]
        ),

        "target_h8_rows": (
            target_counts[
                "8"
            ]
        ),

        "target_h16_rows": (
            target_counts[
                "16"
            ]
        ),

        "5m_15m_count_mismatch": int(
            aggregation[
                "count_mismatch"
            ]
        ),

        "OHLC_mismatch": int(
            aggregation[
                "ohlcv_oi_mismatch"
            ]
        ),

        "volume_mismatch": int(
            aggregation[
                "ohlcv_oi_mismatch"
            ]
        ),

        "OI_mismatch": int(
            aggregation[
                "ohlcv_oi_mismatch"
            ]
        ),

        "5m_sha256": record[
            "five_minute"
        ][
            "sha256"
        ],

        "15m_sha256": record[
            "fifteen_minute"
        ][
            "sha256"
        ],

        "status": (
            "PASS"
            if sufficient
            else (
                "EXCLUDED_DATA_"
                "INSUFFICIENT"
            )
        ),
    }

    return (
        panel,
        summary,
    )


def main() -> None:

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not DOWNLOAD_INDEX.is_file():
        raise RuntimeError(
            "download_index.json "
            "missing; run "
            "download_robustness_"
            "futures.py first"
        )

    records = json.loads(
        DOWNLOAD_INDEX.read_text(
            encoding="utf-8"
        )
    )

    summaries = []

    panels = {}

    for code, record in (
        records.items()
    ):

        if (
            record.get(
                "status"
            )
            != "PASS"
        ):

            print(
                f"[skip] {code}: "
                f"{record.get('status')}",
                flush=True,
            )

            continue

        print(
            f"[build] {code}",
            flush=True,
        )

        try:

            panel, summary = (
                build_panel(
                    code,
                    record,
                )
            )

            summaries.append(
                summary
            )

            panels[
                code
            ] = panel

        except Exception as exc:

            print(
                f"{code} FAILED: "
                f"{type(exc).__name__}: "
                f"{exc}",
                flush=True,
            )

            summaries.append(
                {
                    "instrument": code,
                    "symbol": record[
                        "symbol"
                    ],
                    "status": "FAILED",
                    "error": (
                        f"{type(exc).__name__}: "
                        f"{exc}"
                    ),
                }
            )

    instrument_summary = pd.DataFrame(
        summaries
    )

    # --------------------------------------------------------
    # Identical schema check
    # --------------------------------------------------------

    reference = None

    schemas = {}

    for code, panel in (
        panels.items()
    ):

        schemas[
            code
        ] = list(
            panel.columns
        )

        if reference is None:
            reference = code

    identical = True

    if schemas:

        base_schema = schemas[
            reference
        ]

        identical = all(
            value
            == base_schema
            for value
            in schemas.values()
        )

    if not identical:
        raise RuntimeError(
            f"Panel schema differs "
            f"from {reference}"
        )

    # --------------------------------------------------------
    # Persist panels
    # --------------------------------------------------------

    file_sizes = {}

    for code, panel in (
        panels.items()
    ):

        path = (
            OUT
            / f"{code}_panel.csv"
        )

        panel.to_csv(
            path,
            index=False,
        )

        size_mb = (
            path.stat().st_size
            / 1024
            / 1024
        )

        file_sizes[
            path.name
        ] = round(
            size_mb,
            4,
        )

        if size_mb > 50:
            raise RuntimeError(
                f"{path.name} "
                ">50MB"
            )

    instrument_summary.to_csv(
        OUT
        / "instrument_summary.csv",
        index=False,
    )

    size_mb = (
        (
            OUT
            / "instrument_summary.csv"
        )
        .stat()
        .st_size
        / 1024
        / 1024
    )

    file_sizes[
        "instrument_summary.csv"
    ] = round(
        size_mb,
        4,
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    passed = [
        row[
            "instrument"
        ]
        for row in summaries
        if row.get(
            "status"
        )
        == "PASS"
    ]

    missing_required = sorted(
        REQUIRED_INSTRUMENTS
        - set(passed)
    )

    ok = (
        len(passed)
        >= MIN_PASS_INSTRUMENTS
        and not missing_required
    )

    validation = {
        "status": (
            "PASS"
            if ok
            else "FAIL"
        ),

        "pass_instrument_count": int(
            len(passed)
        ),

        "pass_instruments": (
            sorted(
                passed
            )
        ),

        "min_pass_instruments": (
            MIN_PASS_INSTRUMENTS
        ),

        "required_instruments": (
            sorted(
                REQUIRED_INSTRUMENTS
            )
        ),

        "missing_required": (
            missing_required
        ),

        "identical_feature_schema": (
            identical
        ),

        "model_feature_count": int(
            len(
                MODEL_FEATURES
            )
        ),

        "uses_smc": False,
        "uses_momentum": False,
        "uses_oracle": False,
        "uses_semivariance_or_jump": False,
        "uses_1h_or_4h_series": False,

        "min_5m_rows": (
            MIN_5M_ROWS
        ),

        "min_panel_rows": (
            MIN_PANEL_ROWS
        ),

        "file_sizes_mb": (
            file_sizes
        ),
    }

    (
        OUT
        / "validation.json"
    ).write_text(
        json.dumps(
            validation,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    manifest = {
        "purpose": (
            "Compact multi-instrument "
            "panels for Quantile V2 "
            "robustness replication."
        ),

        "feature_schema": (
            MODEL_FEATURES
        ),

        "horizons": list(
            HORIZONS
        ),

        "realized_windows_5m": {
            label: n
            for label, n in (
                REALIZED_WINDOWS.items()
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

            "mfe": (
                "evaluation reference "
                "only"
            ),
        },

        "excluded_by_design": [
            "SMC",
            "Momentum",
            "Oracle",
            "semivariance",
            "jump",
            "1h series",
            "4h series",
        ],

        "instrument_notes": (
            "Wall-clock coverage differs "
            "per instrument because the "
            "5m serial is capped at "
            "10000 bars. Requirement is "
            "comparable observation "
            "counts, not identical dates."
        ),

        "validation": validation,
    }

    (
        OUT
        / "manifest.json"
    ).write_text(
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
        ),
        flush=True,
    )

    print(
        instrument_summary.to_string(
            index=False
        ),
        flush=True,
    )

    print(
        "ROBUSTNESS_DATA_PREP_"
        + (
            "PASS"
            if ok
            else "FAIL"
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
