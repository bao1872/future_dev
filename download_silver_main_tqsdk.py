#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download SHFE Silver main-continuous historical K-lines from TqSdk
and save them as offline CSV files.

Symbol:
    KQ.m@SHFE.ag  (SHFE Silver main continuous)

Fixed-window design (current target):
    三个周期（15m / 1h / 4h）各自从 TqSdk 取 data_length = 10000 根。
    15m 的 [window_start, window_end] 作为统一对齐窗口；1h / 4h 直接按
    15m 的真实时间戳窗口截断，保证三份数据覆盖完全相同的 [win_min, win_max]。

    截断后根数（约，取决于服务端实际返回）：
        15m : 10000
        1h  : 由 15m 窗口内的 1h bar 数决定
        4h  : 由 15m 窗口内的 4h bar 数决定

Install:
    python -m pip install -U tqsdk pandas

Before running:
    Export your Kuaiqi (快期) account as environment variables, or put them
    into a .env file next to this script (this file is git-ignored):

        TQ_USER=你的快期账号
        TQ_PASSWORD=你的快期密码

Run:
    python download_silver_main_tqsdk.py

Output:
    ./silver_main_data/silver_main_15m.csv
    ./silver_main_data/silver_main_1h.csv
    ./silver_main_data/silver_main_4h.csv
"""

import os
import sys
from pathlib import Path

import pandas as pd
from tqsdk import TqApi, TqAuth


# =========================
# User settings
# =========================

def load_dotenv(path: Path) -> None:
    """极简 .env 加载器：只读取 KEY=VALUE 行，不覆盖已有的环境变量。"""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


# 凭据只从环境变量 / .env 读取，绝不写死在代码里
load_dotenv(Path(__file__).resolve().parent / ".env")
TQ_USER = os.environ.get("TQ_USER", "").strip()
TQ_PASSWORD = os.environ.get("TQ_PASSWORD", "").strip()

# 上期所白银主连
SYMBOL = "KQ.m@SHFE.ag"

# 输出目录：脚本所在目录下的 silver_main_data/
OUTPUT_DIR = Path(__file__).resolve().parent / "silver_main_data"

# 每个周期取 data_length = 10000 根；之后 1h / 4h 按 15m 窗口截断
REQUESTS = {
    "15m": {"duration_seconds": 15 * 60, "data_length": 10000},
    "1h":  {"duration_seconds": 60 * 60, "data_length": 10000},
    "4h":  {"duration_seconds": 4 * 60 * 60, "data_length": 10000},
}

# CSV保留的行情字段
MARKET_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "open_oi",
    "close_oi",
]


def prepare_for_csv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert a TqSdk K-line DataFrame to a clean offline CSV table.

    Keeps:
      - original nanosecond timestamp as datetime_ns
      - human-readable Beijing time as datetime
      - OHLCV + open/close OI
      - symbol / timeframe metadata
    """
    out = df.copy()

    # Keep only valid rows. During initial loading TqSdk may temporarily
    # expose placeholder rows before is_serial_ready() becomes True.
    out = out[out["datetime"].notna()].copy()
    out = out[out["datetime"] > 0].copy()

    # Preserve original TqSdk nanosecond timestamp as int64.
    out.rename(columns={"datetime": "datetime_ns"}, inplace=True)
    out["datetime_ns"] = out["datetime_ns"].astype("int64")

    # TqSdk serials carry their own `symbol` column. On TqSdk 3.x it merely
    # repeats the main-continuous id, so drop it; keep it as
    # `underlying_symbol` only when it really resolves to distinct contracts.
    if "symbol" in out.columns:
        if len(out) > 0 and out["symbol"].nunique(dropna=False) == 1 \
                and out["symbol"].iloc[0] == SYMBOL:
            out.drop(columns=["symbol"], inplace=True)
        else:
            out.rename(columns={"symbol": "underlying_symbol"}, inplace=True)

    # Unix epoch timestamp -> Beijing time.
    beijing_time = (
        pd.to_datetime(out["datetime_ns"].astype("int64"), unit="ns", utc=True)
        .dt.tz_convert("Asia/Shanghai")
    )
    out.insert(0, "datetime", beijing_time.dt.strftime("%Y-%m-%d %H:%M:%S"))

    out.insert(1, "symbol", SYMBOL)
    out.insert(2, "timeframe", timeframe)

    wanted = [
        "datetime",
        "datetime_ns",
        "symbol",
        "timeframe",
        *MARKET_COLUMNS,
    ]
    wanted = [col for col in wanted if col in out.columns]
    out = out[wanted].reset_index(drop=True)
    return out


def _fmt_ns(ns: int) -> str:
    return (pd.to_datetime(ns, unit="ns", utc=True)
            .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"))


def validate(df: pd.DataFrame, tf: str) -> None:
    """Minimal data validation per audit spec. Raises AssertionError on failure."""
    assert len(df) > 0, f"[{tf}] empty frame"
    assert df["datetime_ns"].is_monotonic_increasing, f"[{tf}] datetime_ns not monotonic increasing"
    assert not df["datetime_ns"].duplicated().any(), f"[{tf}] duplicate datetime_ns"

    req = ["open", "high", "low", "close", "volume", "open_oi", "close_oi"]
    assert not df[req].isna().any().any(), f"[{tf}] NaN in OHLCV/OI"

    assert (df["high"] >= df[["open", "close"]].max(axis=1)).all(), \
        f"[{tf}] high < max(open, close)"
    assert (df["low"] <= df[["open", "close"]].min(axis=1)).all(), \
        f"[{tf}] low > min(open, close)"


def main() -> None:
    if not TQ_USER or not TQ_PASSWORD:
        print(
            "\nERROR: Please fill in TQ_USER and TQ_PASSWORD at the top of this script.\n"
            "快期账户和密码目前是空的，请先填写后再运行。\n",
            file=sys.stderr,
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    api = None
    try:
        print("Connecting to TqSdk...")
        print(f"Symbol: {SYMBOL}")
        print(f"Output directory: {OUTPUT_DIR}\n")

        api = TqApi(auth=TqAuth(TQ_USER, TQ_PASSWORD))

        # Subscribe all three series first.
        # data_length=10000 is the single API assumption being verified this run.
        series = {}
        for timeframe, cfg in REQUESTS.items():
            print(
                f"Requesting {timeframe}: "
                f"{cfg['data_length']} bars, "
                f"{cfg['duration_seconds']} sec/bar"
            )
            series[timeframe] = api.get_kline_serial(
                SYMBOL,
                duration_seconds=cfg["duration_seconds"],
                data_length=cfg["data_length"],
            )

        # Wait until every requested history series is fully received.
        print("\nWaiting for all historical K-line series to finish loading...")
        while not all(api.is_serial_ready(df) for df in series.values()):
            api.wait_update()

        print("All requested K-line series are ready.\n")

        # 1) 15m 作为统一对齐窗口
        clean_15m = prepare_for_csv(series["15m"], "15m")
        win_min = int(clean_15m["datetime_ns"].min())
        win_max = int(clean_15m["datetime_ns"].max())
        print(f"对齐窗口 (以 15m 为准): {_fmt_ns(win_min)} -> {_fmt_ns(win_max)}\n")

        frames = {"15m": clean_15m}
        for timeframe in ("1h", "4h"):
            clean = prepare_for_csv(series[timeframe], timeframe)
            before = len(clean)
            mask = (clean["datetime_ns"] >= win_min) & (clean["datetime_ns"] <= win_max)
            clean = clean[mask].reset_index(drop=True)
            dropped = before - len(clean)
            print(
                f"[{timeframe}] 截断到对齐窗口: {before} -> {len(clean)} 根 "
                f"(丢弃窗口外 {dropped} 根)"
            )
            frames[timeframe] = clean

        # 2) 校验
        print("\nValidating...")
        for tf, df in frames.items():
            validate(df, tf)
            print(f"  [{tf}] validation PASS ({len(df)} bars)")

        # 3) 写出三份 CSV
        for timeframe, clean_df in frames.items():
            output_file = OUTPUT_DIR / f"silver_main_{timeframe}.csv"
            clean_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"[{timeframe}] saved: {output_file.name}")

        # 4) 报告
        print("\n================ REPORT ================")
        for tf in ("15m", "1h", "4h"):
            df = frames[tf]
            print(f"{tf}:")
            print(f"  bars  = {len(df)}")
            print(f"  start = {df.iloc[0]['datetime']}")
            print(f"  end   = {df.iloc[-1]['datetime']}")
        print("共同窗口:")
        print(f"  {_fmt_ns(win_min)} -> {_fmt_ns(win_max)}")
        print("========================================")

        print("\nDone.")

    finally:
        if api is not None:
            api.close()


if __name__ == "__main__":
    main()
