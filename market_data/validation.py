from __future__ import annotations

from typing import Any

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

    if include_cross_tf and all(tf in frames for tf in ("15m", "1h", "4h")):
        # Reuse the established downloader aggregation semantics rather than creating
        # a second definition here.
        try:
            from download_silver_main_tqsdk import validate_aggregation, REQUESTS

            for lower_tf, higher_tf in (("15m", "1h"), ("1h", "4h")):
                errors, stats = validate_aggregation(
                    frames[lower_tf],
                    frames[higher_tf],
                    lower_tf,
                    higher_tf,
                    REQUESTS[higher_tf]["duration_seconds"],
                )
                key = f"{lower_tf}->{higher_tf}"
                report["cross_tf"][key] = {
                    "ok": not errors,
                    "stats": stats,
                    "errors": errors[:20],
                }
                if errors:
                    report["ok"] = False
        except Exception as exc:
            report["cross_tf"]["error"] = str(exc)
            report["ok"] = False

    return report
