#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download SHFE Silver main-continuous historical K-lines from TqSdk
and save them as offline CSV files.

Symbol:
    KQ.m@SHFE.ag  (SHFE Silver main continuous)

Fixed-window design (current target):
    三个周期（15m / 1h / 4h）各自从 TqSdk 取 data_length = 10000 根。
    按“bar 闭合覆盖”对齐三周期窗口（而非只看 bar start 是否落在 15m 区间）：
        共同起点 = 三周期第一根 bar start 的最大值
        共同终点 = 三周期最后一根 bar 的 bar_end（start + 周期时长）的最小值
    每个周期只保留 bar_start >= 共同起点 且 bar_end <= 共同终点的行，
    保证三份数据覆盖完全相同的“已闭合”时间区间（避免某周期多带一根尚未
    闭合的尾巴，或把窗口起点前的不完整段算进来）。

    每个周期会丢弃末尾尚未走完的 forming bar（其名义结束时间晚于当前
    UTC 时间），并做 15m→1h、1h→4h 的跨周期聚合一致性校验（含覆盖统计，
    缺失低周期数据的 higher bar 直接记 error，不再静默跳过）后才落盘。

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

# 1 秒 = 10^9 纳秒，用于把 bar 起点时间换算为名义结束时间
NS_PER_SECOND = 1_000_000_000

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


def drop_unclosed_tail(
    df: pd.DataFrame,
    duration_seconds: int,
    tf: str,
) -> pd.DataFrame:
    """
    丢弃序列末尾尚未走完的 forming bar。

    TqSdk 会把“当前正在形成的那根 K 线”也放进序列，其名义结束时间
    （bar 起点 + 周期时长）晚于当前 UTC 时间即判定为未完成，必须剔除，
    否则会被当成一个 volume 异常小、与下一周期同时间戳的假完整 bar。
    """
    if len(df) < 2:
        raise AssertionError(f"[{tf}] insufficient bars")

    now_ns = pd.Timestamp.now(tz="UTC").value

    last_start_ns = int(df.iloc[-1]["datetime_ns"])
    nominal_end_ns = last_start_ns + duration_seconds * NS_PER_SECOND

    if nominal_end_ns > now_ns:
        print(f"[{tf}] drop forming bar: {df.iloc[-1]['datetime']}")
        return df.iloc[:-1].copy()

    return df


def align_common_closed_window(
    prepared: dict,
    requests: dict,
):
    """
    按“bar 闭合覆盖”对齐三周期到完全相同的窗口。

    共同起点 = 三周期第一根 bar start 的最大值。
    共同终点 = 三周期最后一根 bar 的 bar_end（start + 周期时长）的最小值。

    每个周期只保留 bar_start >= common_start 且 bar_end <= common_end 的行。
    这样三份数据覆盖的是同一段“所有周期均已完整闭合”的时间区间，而不是
    仅仅把各周期 bar start 截到 15m 的 [min,max]（那样会让 15m 多带一根
    bar_end 超出共同终点的尾巴，或把起点前的不完整段算进来）。

    返回 (aligned: dict, common_start_ns: int, common_end_ns: int)。
    """
    starts = {
        tf: int(df.iloc[0]["datetime_ns"])
        for tf, df in prepared.items()
    }
    ends = {
        tf: int(df.iloc[-1]["datetime_ns"]) + requests[tf]["duration_seconds"] * NS_PER_SECOND
        for tf, df in prepared.items()
    }

    common_start = max(starts.values())
    common_end = min(ends.values())

    aligned = {}
    for tf, df in prepared.items():
        duration_ns = requests[tf]["duration_seconds"] * NS_PER_SECOND
        bar_end = df["datetime_ns"] + duration_ns
        x = df[
            (df["datetime_ns"] >= common_start)
            & (bar_end <= common_end)
        ].copy()
        aligned[tf] = x.reset_index(drop=True)

    return aligned, common_start, common_end


def validate_aggregation(
    lower: pd.DataFrame,
    higher: pd.DataFrame,
    lower_tf: str,
    higher_tf: str,
    higher_duration_seconds: int,
):
    """
    用“高周期实际相邻 timestamp”作为 bucket 边界，验证低周期聚合是否
    与高周期完全一致。中国期货交易时段不连续，不能用简单 resample。

    对 higher 的每一根 bar：
      - 非最后一根：bucket = [start, 下一根 start)
      - 最后一根（edge）：bucket = [start, start + 周期时长)
        （窗口已按“闭合覆盖”对齐，最后一根必然完整闭合，可用名义结束时间
        作为上界，这样最后一根不再被无声跳过）
    任意一根 higher bar 若在其 bucket 内完全找不到低周期数据，直接记
    error（不再静默 continue），避免 false-green。

    返回 (errors, stats)。stats 含：
        higher_total / compared / skipped_edge / empty / mismatch_count
    """
    errors = []
    higher_total = len(higher)
    compared = 0
    skipped_edge = 0
    empty = 0
    mismatch_count = 0

    for i in range(higher_total):
        h = higher.iloc[i]
        start_ns = int(h["datetime_ns"])

        if i < higher_total - 1:
            end_ns = int(higher.iloc[i + 1]["datetime_ns"])
        else:
            # 最后一根：用名义结束时间作上界（闭合窗口内必然完整）
            end_ns = start_ns + higher_duration_seconds * NS_PER_SECOND
            skipped_edge += 1

        sub = lower[
            (lower["datetime_ns"] >= start_ns)
            & (lower["datetime_ns"] < end_ns)
        ]

        if sub.empty:
            empty += 1
            errors.append({
                "datetime": h["datetime"],
                "type": "missing_lower_bars",
                "from": lower_tf,
                "to": higher_tf,
            })
            continue

        compared += 1

        expected = {
            "open": float(sub.iloc[0]["open"]),
            "high": float(sub["high"].max()),
            "low": float(sub["low"].min()),
            "close": float(sub.iloc[-1]["close"]),
            "volume": float(sub["volume"].sum()),
            "open_oi": float(sub.iloc[0]["open_oi"]),
            "close_oi": float(sub.iloc[-1]["close_oi"]),
        }

        for field, exp in expected.items():
            actual = float(h[field])
            if abs(actual - exp) > 1e-9:
                mismatch_count += 1
                errors.append({
                    "datetime": h["datetime"],
                    "from": lower_tf,
                    "to": higher_tf,
                    "field": field,
                    "expected": exp,
                    "actual": actual,
                })

    stats = {
        "higher_total": higher_total,
        "compared": compared,
        "skipped_edge": skipped_edge,
        "empty": empty,
        "mismatch_count": mismatch_count,
    }
    return errors, stats


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

    # 成交量与持仓量必须非负
    assert (df["volume"] >= 0).all(), f"[{tf}] negative volume"
    assert (df["open_oi"] >= 0).all(), f"[{tf}] negative open_oi"
    assert (df["close_oi"] >= 0).all(), f"[{tf}] negative close_oi"


def main() -> None:
    if not TQ_USER or not TQ_PASSWORD:
        print(
            "\nERROR: 未检测到快期账号密码（TQ_USER / TQ_PASSWORD 均为空）。\n"
            "请通过环境变量，或在脚本同目录的 .env 文件（由 .env.example 复制）中注入，\n"
            "切勿将凭据写入代码。\n",
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

        # 1) 每个周期先 prepare，并丢弃未完成尾部 bar
        prepared = {}
        for tf, cfg in REQUESTS.items():
            df = prepare_for_csv(series[tf], tf)
            df = drop_unclosed_tail(df, cfg["duration_seconds"], tf)
            prepared[tf] = df

        # 2) 按“bar 闭合覆盖”把三周期对齐到完全相同窗口
        frames, common_start, common_end = align_common_closed_window(prepared, REQUESTS)
        print(
            f"共同闭合窗口: {_fmt_ns(common_start)} -> {_fmt_ns(common_end)}\n"
        )
        for tf in ("15m", "1h", "4h"):
            before = len(prepared[tf])
            print(
                f"[{tf}] 闭合窗口对齐: {before} -> {len(frames[tf])} 根 "
                f"(丢弃 {before - len(frames[tf])} 根窗口外/未闭合)"
            )

        # 3) 单周期内部校验
        print("\nValidating...")
        for tf, df in frames.items():
            validate(df, tf)
            print(f"  [{tf}] internal validation PASS ({len(df)} bars)")

        # 4) 跨周期聚合一致性校验（带覆盖统计，不再静默跳过缺失）
        print("\nCross-timeframe aggregation check...")
        errors_15m_1h, stats_15m_1h = validate_aggregation(
            frames["15m"], frames["1h"], "15m", "1h",
            REQUESTS["1h"]["duration_seconds"],
        )
        errors_1h_4h, stats_1h_4h = validate_aggregation(
            frames["1h"], frames["4h"], "1h", "4h",
            REQUESTS["4h"]["duration_seconds"],
        )

        def _print_stats(label, stats):
            print(f"  {label}:")
            print(f"    higher_total = {stats['higher_total']}")
            print(f"    compared     = {stats['compared']}")
            print(f"    skipped_edge = {stats['skipped_edge']}")
            print(f"    empty        = {stats['empty']}")
            print(f"    mismatches   = {stats['mismatch_count']}")
            if stats["empty"]:
                print("    EMPTY (missing lower bars):", errors_15m_1h if "15m" in label else errors_1h_4h)
            if stats["mismatch_count"]:
                sample = (errors_15m_1h if "15m" in label else errors_1h_4h)[:10]
                print("    sample:", sample)

        _print_stats("15m -> 1h", stats_15m_1h)
        _print_stats("1h  -> 4h", stats_1h_4h)

        assert stats_15m_1h["empty"] == 0, "15m->1h has higher bars with no lower coverage"
        assert stats_1h_4h["empty"] == 0, "1h->4h has higher bars with no lower coverage"
        assert stats_15m_1h["mismatch_count"] == 0, "15m->1h aggregation mismatches found"
        assert stats_1h_4h["mismatch_count"] == 0, "1h->4h aggregation mismatches found"
        print("  cross-timeframe aggregation PASS")

        # 5) 写出三份 CSV
        for timeframe, clean_df in frames.items():
            output_file = OUTPUT_DIR / f"silver_main_{timeframe}.csv"
            clean_df.to_csv(output_file, index=False, encoding="utf-8-sig")
            print(f"[{timeframe}] saved: {output_file.name}")

        # 6) 报告
        print("\n================ REPORT ================")
        print("COMMON CLOSED WINDOW")
        print(f"  start = {_fmt_ns(common_start)}")
        print(f"  end   = {_fmt_ns(common_end)}")
        for tf in ("15m", "1h", "4h"):
            df = frames[tf]
            print(f"{tf} bars = {len(df)}")
            print(f"  start = {df.iloc[0]['datetime']}")
            print(f"  end   = {df.iloc[-1]['datetime']}")
        print("\n15m -> 1h:")
        print(f"  higher_total = {stats_15m_1h['higher_total']}")
        print(f"  compared     = {stats_15m_1h['compared']}")
        print(f"  empty        = {stats_15m_1h['empty']}")
        print(f"  mismatches   = {stats_15m_1h['mismatch_count']}")
        print("1h -> 4h:")
        print(f"  higher_total = {stats_1h_4h['higher_total']}")
        print(f"  compared     = {stats_1h_4h['compared']}")
        print(f"  empty        = {stats_1h_4h['empty']}")
        print(f"  mismatches   = {stats_1h_4h['mismatch_count']}")
        print("========================================")

        print("\nDone.")

    finally:
        if api is not None:
            api.close()


if __name__ == "__main__":
    main()
