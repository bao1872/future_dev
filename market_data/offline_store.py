from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import pandas as pd

from .config import CURRENT_FILES, DISPLAY_NAME, SOURCE, SYMBOL, TIMEFRAMES

REQUIRED_COLUMNS = {
    "datetime",
    "datetime_ns",
    "symbol",
    "timeframe",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_oi",
    "close_oi",
}


def available_timeframes() -> list[str]:
    return [tf for tf in TIMEFRAMES if CURRENT_FILES[tf].is_file()]


def _read_file(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df = df.sort_values("datetime_ns").reset_index(drop=True)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="raise")
    df["datetime_ns"] = pd.to_numeric(df["datetime_ns"], errors="raise").astype("int64")
    return df


@lru_cache(maxsize=8)
def _load_cached(tf: str, mtime_ns: int) -> pd.DataFrame:
    del mtime_ns  # cache key only
    return _read_file(CURRENT_FILES[tf])


def load_bars(
    timeframe: str,
    *,
    start=None,
    end=None,
    copy: bool = True,
) -> pd.DataFrame:
    """Load the current valid offline market data.

    This is the default research/strategy data entrypoint. It never connects to TqSdk.
    """
    if timeframe not in TIMEFRAMES:
        raise ValueError(f"Unsupported timeframe: {timeframe!r}; expected {list(TIMEFRAMES)}")

    path = CURRENT_FILES[timeframe]
    if not path.is_file():
        raise FileNotFoundError(f"Offline market data not found: {path}")

    df = _load_cached(timeframe, path.stat().st_mtime_ns)
    out = df
    if start is not None:
        start_ts = pd.Timestamp(start)
        out = out[out["datetime"] >= start_ts]
    if end is not None:
        end_ts = pd.Timestamp(end)
        out = out[out["datetime"] <= end_ts]
    return out.copy() if copy else out


def load_bundle(
    timeframes: Iterable[str] = ("15m", "1h", "4h"),
    *,
    start=None,
    end=None,
) -> dict[str, pd.DataFrame]:
    return {
        tf: load_bars(tf, start=start, end=end)
        for tf in timeframes
    }


def to_indicator_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert offline table to a DatetimeIndex frame for canonical indicators."""
    out = df.copy()
    out = out.set_index(pd.DatetimeIndex(out["datetime"]))
    return out


def get_market_status() -> dict:
    items = {}
    for tf in TIMEFRAMES:
        path = CURRENT_FILES[tf]
        if not path.is_file():
            continue
        try:
            df = load_bars(tf, copy=False)
            items[tf] = {
                "path": str(path),
                "rows": int(len(df)),
                "start": str(df["datetime"].iloc[0]) if len(df) else None,
                "end": str(df["datetime"].iloc[-1]) if len(df) else None,
            }
        except Exception as exc:
            items[tf] = {"path": str(path), "error": str(exc)}

    return {
        "source": SOURCE,
        "symbol": SYMBOL,
        "display_name": DISPLAY_NAME,
        "available_timeframes": available_timeframes(),
        "timeframes": items,
    }
