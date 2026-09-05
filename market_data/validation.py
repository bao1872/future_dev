from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .config import TIMEFRAMES
from .offline_store import load_bars


def validate_frame(df: pd.DataFrame, timeframe: str) -> list[str]:
    errors: list[str] = []
    if df.empty:
        return [f"[{timeframe}] empty frame"]

    ns = df["datetime_ns"]
    if not ns.is_monotonic_increasing:
        errors.append(f"[{timeframe}] datetime_ns not monotonic")
    if ns.duplicated().any():
        errors.append(f"[{timeframe}] duplicate datetime_ns")

    numeric = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
    if df[numeric].isna().any().any():
        errors.append(f"[{timeframe}] NaN in OHLCV/OI")

    if not (df["high"] >= df[["open", "close"]].max(axis=1)).all():
        errors.append(f"[{timeframe}] high < max(open, close)")
    if not (df["low"] <= df[["open", "close"]].min(axis=1)).all():
        errors.append(f"[{timeframe}] low > min(open, close)")

    for col in ("volume", "open_oi", "close_oi"):
        if not (df[col] >= 0).all():
            errors.append(f"[{timeframe}] negative {col}")

    return errors


def validate_current_offline_data(*, include_cross_tf: bool = True) -> dict[str, Any]:
    report: dict[str, Any] = {"ok": True, "timeframes": {}, "cross_tf": {}}

    frames = {}
    for tf in TIMEFRAMES:
        try:
            df = load_bars(tf)
            frames[tf] = df
            errs = validate_frame(df, tf)
            report["timeframes"][tf] = {
                "ok": not errs,
                "errors": errs,
                "rows": len(df),
                "start": str(df["datetime"].iloc[0]),
                "end": str(df["datetime"].iloc[-1]),
            }
            if errs:
                report["ok"] = False
        except Exception as exc:
            report["timeframes"][tf] = {"ok": False, "errors": [str(exc)]}
            report["ok"] = False

    pairs = [
        ("15m", "1h"),
        ("1h", "4h"),
    ]

    if include_cross_tf:
        for lower_tf, higher_tf in pairs:
            if (
                lower_tf not in frames
                or higher_tf not in frames
            ):
                continue

            key = (
                f"{lower_tf}->{higher_tf}"
            )

            try:
                errors, stats = (
                    validate_aggregation(
                        frames[
                            lower_tf
                        ],
                        frames[
                            higher_tf
                        ],
                        lower_tf,
                        higher_tf,
                    )
                )

                report[
                    "cross_tf"
                ][
                    key
                ] = {
                    "ok": not errors,
                    "stats": stats,
                    "errors": errors[
                        :20
                    ],
                }

                if errors:
                    report[
                        "ok"
                    ] = False

            except Exception as exc:
                report[
                    "cross_tf"
                ][key] = {
                    "ok": False,
                    "errors": [
                        str(exc)
                    ],
                }
                report["ok"] = False

    return report


def validate_aggregation(
    lower: pd.DataFrame,
    higher: pd.DataFrame,
    lower_tf: str,
    higher_tf: str,
) -> tuple[list[str], dict]:
    """Check that `higher` is exactly the aggregation of `lower`.

    Self-contained definition: each higher bar's window is derived by
    flooring the lower bars' epoch nanoseconds to the higher period,
    then OHLC / volume / OI are re-aggregated and compared.

    Self-contained so the validation layer has no dependency on
    any acquisition module.
    """

    from .config import (
        TIMEFRAMES,
    )

    errors: list[str] = []

    period_ns = int(
        TIMEFRAMES[
            higher_tf
        ]
    ) * 1_000_000_000

    lo = lower.copy()
    hi = higher.copy()

    lo[
        "_bucket"
    ] = (
        lo[
            "datetime_ns"
        ].to_numpy()
        // period_ns
    ) * period_ns

    hi[
        "_bucket"
    ] = (
        hi[
            "datetime_ns"
        ].to_numpy()
        // period_ns
    ) * period_ns

    agg = (
        lo.groupby(
            "_bucket"
        )
        .agg(
            agg_open=(
                "open",
                "first",
            ),
            agg_high=(
                "high",
                "max",
            ),
            agg_low=(
                "low",
                "min",
            ),
            agg_close=(
                "close",
                "last",
            ),
            agg_volume=(
                "volume",
                "sum",
            ),
            agg_oi=(
                "close_oi",
                "last",
            ),
            lower_bars=(
                "close",
                "size",
            ),
        )
        .reset_index()
    )

    merged = agg.merge(
        hi,
        on="_bucket",
        how="inner",
    )

    if merged.empty:
        return (
            [
                f"[{lower_tf}->{higher_tf}] "
                "no overlapping bars"
            ],
            {},
        )

    stats = {
        "compared_bars": int(
            len(
                merged
            )
        ),
        "lower_bars_per_higher": (
            float(
                merged[
                    "lower_bars"
                ].median()
            )
        ),
    }

    checks = {
        "open": (
            merged[
                "agg_open"
            ],
            merged["open"],
        ),
        "high": (
            merged[
                "agg_high"
            ],
            merged["high"],
        ),
        "low": (
            merged[
                "agg_low"
            ],
            merged["low"],
        ),
        "close": (
            merged[
                "agg_close"
            ],
            merged[
                "close"
            ],
        ),
        "close_oi": (
            merged[
                "agg_oi"
            ],
            merged[
                "close_oi"
            ],
        ),
    }

    for name, (
        left,
        right,
    ) in checks.items():

        rate = float(
            np.isclose(
                left,
                right,
                rtol=1e-9,
                atol=1e-9,
            ).mean()
        )

        stats[
            f"{name}_match_rate"
        ] = rate

        if rate < 1.0:
            errors.append(
                f"[{lower_tf}->{higher_tf}] "
                f"{name} mismatch rate "
                f"{rate:.6f}"
            )

    vol_rate = float(
        np.isclose(
            merged[
                "agg_volume"
            ],
            merged[
                "volume"
            ],
            rtol=1e-6,
            atol=1e-6,
        ).mean()
    )

    stats[
        "volume_match_rate"
    ] = vol_rate

    if vol_rate < 1.0:
        errors.append(
            f"[{lower_tf}->{higher_tf}] "
            f"volume mismatch rate "
            f"{vol_rate:.6f}"
        )

    return errors, stats
