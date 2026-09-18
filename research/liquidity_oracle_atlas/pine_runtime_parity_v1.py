"""
R2B-Prep: TradingView parity harness (DTP only this round).

科学目标（本轮不验证任何 TradingView parity）:

    TradingView 真实 CSV
        -> 数据身份检查 (OHLC)
        -> 时间轴检查
        -> Python 对应值 (调用已审核 owner)
        -> 逐 timestamp 比较
        -> mismatch 报告

只有外部 TradingView CSV 真正进入以后，才能得出 parity 结论。
本脚本本身**不会**生成 TradingView reference CSV，也**不会**用 Python
CSV 冒充 TradingView CSV。

复用（不得重新实现）来自 experiment_structural_reversion_pgm_v1:
    rolling_sma, true_range, pine_rma, trend_state_from_score,
    resample_causal, PINE_SOURCE_REFS, verify_pine_source_refs, git_head

DTP 指标 (5m 真实 raw bars):
    SMA50 / ATR200 / avg_diff / p100(rolling max 500) / avg_col / trend

本轮范围: 仅 DTP。
不处理: SR / Liquidity / G0-G6 / PGM / parameter scan / TB3 / 收益。

状态声明:
    TradingView runtime parity NOT RUN
    Reason: external TradingView CSV not yet supplied
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_structural_reversion_pgm_v1 import (
    PINE_SOURCE_REFS,
    verify_pine_source_refs,
    git_head,
    rolling_sma,
    true_range,
    pine_rma,
    trend_state_from_score,
    resample_causal,
)

# ---------------------------------------------------------------------------
# paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER_5M = REPO_ROOT / "silver_main_data" / "silver_main_5m.csv"
DTP_SOURCE_SHA = PINE_SOURCE_REFS["DeviationTrendProfile.pine"]

SMA_LEN = 50
ATR_LEN = 200
AVG_DIFF_LAG = 5
P100_WINDOW = 500
TREND_SWITCH = 0.1

# 项目 TDX 5m bar 时间约定: 交易所本地时间 (SHFE, Asia/Shanghai)。
# naive 5m bar 以此 localize 后转 UTC 才能与 TradingView 导出对齐。
PY_TZ = "Asia/Shanghai"

# 真实 TradingView 导出列 (允许 indicator prefix, 用 contains 规范化)
TV_INDICATOR_FIELDS = {
    "R2B_SMA50": "sma50",
    "R2B_ATR200": "atr200",
    "R2B_AVG_DIFF": "avg_diff",
    "R2B_P100": "p100",
    "R2B_AVG_COL": "avg_col",
    "R2B_TREND": "trend",
}

# HTF day/segment: 本 round 仅 reserve; 占位实现, 不得据此宣称 HTF parity。
# 真实 day/segment 必须与 experiment 规范交易日历对齐后才可用。
HTF_DAY_SEGMENT_CANONICAL = False

OHLC_COLS = ["open", "high", "low", "close"]


# ---------------------------------------------------------------------------
# timestamp / timeline
# ---------------------------------------------------------------------------

def to_utc_timestamp(
    ts: pd.Series,
    tz: str | None,
) -> pd.Series:
    """
    naive datetime: 必须显式提供 tz, 禁止猜。
    tz-aware: 直接转 UTC。
    """

    ts = pd.to_datetime(ts)

    if ts.dt.tz is not None:
        return ts.dt.tz_convert("UTC")

    if tz is None:
        raise SystemExit("STOP_R2B_TV_TIMEZONE_REQUIRED")

    return ts.dt.tz_localize(tz).dt.tz_convert("UTC")


def check_timeline(ts: pd.Series) -> None:

    if ts.duplicated().any():
        raise SystemExit("STOP_R2B_DUPLICATE_TIMESTAMP")

    if not ts.is_monotonic_increasing:
        raise SystemExit("STOP_R2B_NON_MONOTONIC_TIMESTAMP")


def find_col(
    df: pd.DataFrame,
    needle: str,
) -> str | None:
    """
    exact match 优先; 否则 contains (fail-closed: 多匹配 -> STOP)。
    """

    cols = list(df.columns)

    exact = [c for c in cols if c.lower() == needle.lower()]
    if exact:
        return exact[0]

    contains = [c for c in cols if needle.lower() in c.lower()]

    if len(contains) == 0:
        return None

    if len(contains) > 1:
        raise SystemExit(
            f"STOP_R2B_TV_COLUMN_AMBIGUOUS:{needle}:{contains}"
        )

    return contains[0]


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def load_python_5m(symbol: str) -> pd.DataFrame:
    """
    5m 数据直接使用项目真实 raw bars (silver_main_5m.csv)。
    """

    df = pd.read_csv(SILVER_5M)

    if "symbol" in df.columns:
        want = f"KQ.m@SHFE.{symbol.lower()}"
        if want in set(df["symbol"].astype(str)):
            df = df[df["symbol"].astype(str) == want]

    df["time"] = to_utc_timestamp(df["datetime"], PY_TZ)

    df = df.sort_values("time").set_index("time")

    return df[OHLC_COLS]


def load_tv_csv(
    path: str,
    tv_timezone: str | None,
) -> pd.DataFrame:
    """
    读取 TradingView 导出 CSV, 规范化 OHLC + indicator 列到 UTC index。
    """

    df = pd.read_csv(path)

    time_col = find_col(df, "time")
    if time_col is None:
        raise SystemExit("STOP_R2B_TV_MISSING_TIME_COLUMN")

    df["time"] = to_utc_timestamp(df[time_col], tv_timezone)
    df = df.sort_values("time").set_index("time")

    out = pd.DataFrame(index=df.index)

    for col in OHLC_COLS:
        c = find_col(df, col)
        if c is None:
            raise SystemExit(f"STOP_R2B_TV_MISSING_OHLC_COLUMN:{col}")
        out[col] = df[c]

    for key, canon in TV_INDICATOR_FIELDS.items():
        c = find_col(df, key)
        if c is not None:
            out[canon] = df[c]

    return out


def build_htf_raw(py5m: pd.DataFrame) -> pd.DataFrame:
    """
    HTF resample 需要的 raw frame (time/trading_day/segment/open/high/low/close/disc)。

    注意: trading_day / segment 为占位实现 (HTF_DAY_SEGMENT_CANONICAL=False)。
    segment = 时间间隙 > 5min 处断开; trading_day = bar 日历日期。
    真实 parity 前必须与 experiment 规范交易日历对齐。
    """

    if not HTF_DAY_SEGMENT_CANONICAL:
        print(
            "[WARN] HTF trading_day/segment is PLACEHOLDER; "
            "do NOT claim HTF parity until reconciled with canonical calendar.",
            file=sys.stderr,
        )

    raw = py5m.reset_index()[["time"] + OHLC_COLS].copy()

    gap = raw["time"].diff()
    disc = (gap != pd.Timedelta(minutes=5)).fillna(False).to_numpy(bool)

    raw["disc"] = disc
    raw["segment"] = np.cumsum(disc.astype("int64"))
    raw["trading_day"] = raw["time"].dt.floor("D")

    return raw


# ---------------------------------------------------------------------------
# DTP reference (Python, 调用已审核 owner)
# ---------------------------------------------------------------------------

def compute_dtp(df: pd.DataFrame) -> pd.DataFrame:
    """
    与 dtp_r2b_reference.pine 对应的 Python 参考实现。

    avg      = ta.sma(close, 50)
    atr      = ta.atr(200)
    avg_diff = avg - avg[5]
    p100     = rolling max 500            (Pine UNVERIFIED: percentile_linear_interpolation p100)
    avg_col  = avg_diff / p100
    trend    = trend_state_from_score     (false=-1, true=+1)
    """

    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)

    n = len(close)

    avg = rolling_sma(close, SMA_LEN)
    tr = true_range(high, low, close)
    atr = pine_rma(tr, ATR_LEN)

    avg_diff = np.full(n, np.nan, dtype=float)
    if n > AVG_DIFF_LAG:
        avg_diff[AVG_DIFF_LAG:] = (
            avg[AVG_DIFF_LAG:]
            - avg[: n - AVG_DIFF_LAG]
        )

    p100 = (
        pd.Series(avg_diff)
        .rolling(P100_WINDOW, min_periods=P100_WINDOW)
        .max()
        .to_numpy(dtype=float)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        avg_col = avg_diff / p100

    avg_col = np.where(
        np.isfinite(avg_col),
        avg_col,
        np.nan,
    )

    trend = trend_state_from_score(avg_col, TREND_SWITCH).astype("int8")

    out = pd.DataFrame(index=df.index)
    out["sma50"] = avg
    out["atr200"] = atr
    out["avg_diff"] = avg_diff
    out["p100"] = p100
    out["avg_col"] = avg_col
    out["trend"] = trend

    return out


# ---------------------------------------------------------------------------
# comparator
# ---------------------------------------------------------------------------

def identity_gate(
    py: pd.DataFrame,
    tv: pd.DataFrame,
    atol: float = 1e-9,
) -> dict:
    """
    TradingView OHLC == Python raw OHLC ?
    比较 key = UTC timestamp. 默认严格 isclose(rtol=0, atol=1e-9)。
    """

    n_python = len(py)
    n_tv = len(tv)
    overlap = py.index.intersection(tv.index)
    n_overlap = len(overlap)

    out = {
        "n_python": n_python,
        "n_tv": n_tv,
        "n_timestamp_overlap": n_overlap,
        "timestamp_overlap_rate": n_overlap / max(n_python, n_tv, 1),
    }

    only_py = py.index.difference(tv.index)
    only_tv = tv.index.difference(py.index)
    first_ts_mismatch = None
    if len(only_py):
        first_ts_mismatch = str(only_py.min())
    elif len(only_tv):
        first_ts_mismatch = str(only_tv.min())
    out["first_timestamp_mismatch"] = first_ts_mismatch

    if n_overlap == 0:
        for col in OHLC_COLS:
            out[f"{col}_mismatch"] = 0
            out[f"max_abs_{col}_error"] = None
        out["first_ohlc_mismatch"] = None
        out["data_identity_pass"] = False
        return out

    p = py.loc[overlap]
    t = tv.loc[overlap]

    first_ohlc = None

    for col in OHLC_COLS:
        a = p[col].to_numpy(dtype=float)
        b = t[col].to_numpy(dtype=float)

        both = np.isfinite(a) & np.isfinite(b)
        diff = np.abs(a - b)

        mm = np.zeros(len(a), dtype=bool)
        mm[both] = diff[both] > atol

        one_nan = (
            np.isfinite(a) & ~np.isfinite(b)
        ) | (
            ~np.isfinite(a) & np.isfinite(b)
        )
        mm = mm | one_nan

        cnt = int(mm.sum())
        out[f"{col}_mismatch"] = cnt

        if both.any():
            out[f"max_abs_{col}_error"] = float(diff[both].max())
        else:
            out[f"max_abs_{col}_error"] = None

        if cnt and first_ohlc is None:
            first_ohlc = str(overlap[int(np.argmax(mm))])

    out["first_ohlc_mismatch"] = first_ohlc
    out["data_identity_pass"] = all(
        out[f"{col}_mismatch"] == 0
        for col in OHLC_COLS
    )

    return out


def compare_field(
    py: pd.Series,
    tv: pd.Series,
    name: str,
    rtol: float = 1e-10,
    atol: float = 1e-10,
    integer: bool = False,
) -> dict:
    """
    逐 timestamp 比较单个指标字段。
    trend 用 integer=True (完全整数一致)。
    """

    a = py.to_numpy(dtype=float)
    b = tv.to_numpy(dtype=float)

    both = np.isfinite(a) & np.isfinite(b)
    n_compared = len(a)
    n_both_finite = int(both.sum())

    mismatch_rows = []

    if integer:
        mm = both & (a.astype("int64") != b.astype("int64"))
        if both.any():
            max_abs = 0.0
            mean_abs = 0.0
        else:
            max_abs = None
            mean_abs = None
    else:
        diff = np.abs(a - b)
        tol = atol + rtol * np.abs(b)
        mm = both & (diff > tol)
        if both.any():
            max_abs = float(diff[both].max())
            mean_abs = float(diff[both].mean())
        else:
            max_abs = None
            mean_abs = None

    one_nan = (
        np.isfinite(a) & ~np.isfinite(b)
    ) | (
        ~np.isfinite(a) & np.isfinite(b)
    )
    mm = mm | one_nan

    n_mismatch = int(mm.sum())
    mismatch_rate = n_mismatch / max(n_both_finite, 1)

    first_ts = None
    py_val = None
    tv_val = None

    if n_mismatch:
        idx = int(np.argmax(mm))
        first_ts = str(py.index[idx])
        py_val = float(a[idx])
        tv_val = float(b[idx])
        for i in np.where(mm)[0]:
            mismatch_rows.append(
                (str(py.index[int(i)]), float(a[int(i)]), float(b[int(i)]))
            )

    first_py_finite = (
        str(py.index[int(np.argmax(np.isfinite(a)))])
        if a.size and np.isfinite(a).any()
        else None
    )
    first_tv_finite = (
        str(tv.index[int(np.argmax(np.isfinite(b)))])
        if b.size and np.isfinite(b).any()
        else None
    )

    return {
        "name": name,
        "n_compared": n_compared,
        "n_both_finite": n_both_finite,
        "n_mismatch": n_mismatch,
        "mismatch_rate": mismatch_rate,
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
        "first_mismatch_timestamp": first_ts,
        "python_value": py_val,
        "tradingview_value": tv_val,
        "first_python_finite": first_py_finite,
        "first_tv_finite": first_tv_finite,
        "warmup_mismatch": first_py_finite != first_tv_finite,
        "_mismatch_rows": mismatch_rows,
    }


# ---------------------------------------------------------------------------
# run parity
# ---------------------------------------------------------------------------

def run_parity(
    tv_csv: str,
    symbol: str,
    timeframe: int,
    tv_timezone: str | None,
    output_dir: str,
) -> None:

    # 1) 先证明审核的正是 pin 住的 DTP 源文件
    verify_pine_source_refs()

    # 2) Python 参考 (5m raw bars)
    py = load_python_5m(symbol)
    check_timeline(py.index.to_series())

    if timeframe == 5:
        py_ohlc = py[OHLC_COLS]
        py_dtp = compute_dtp(py)
    else:
        raw = build_htf_raw(py)
        res = resample_causal(raw, timeframe)
        py_ohlc = res[OHLC_COLS]
        py_dtp = compute_dtp(res)

    # 3) TradingView CSV
    tv = load_tv_csv(tv_csv, tv_timezone)
    check_timeline(tv.index.to_series())

    # 4) Data Identity Gate
    gate = identity_gate(py_ohlc, tv[OHLC_COLS])

    tv_csv_sha = hashlib.sha256(Path(tv_csv).read_bytes()).hexdigest()
    timezone_reported = tv_timezone if tv_timezone else "UTC(tz-aware)"

    summary = {
        "source_sha": DTP_SOURCE_SHA,
        "experiment_sha": git_head(),
        "symbol": symbol,
        "timeframe": timeframe,
        "tv_csv_sha256": tv_csv_sha,
        "timezone": timezone_reported,
        "n_overlap": gate["n_timestamp_overlap"],
        "data_identity_pass": gate["data_identity_pass"],
        "indicator_parity_pass": "NOT_RUN",
    }

    if not gate["data_identity_pass"]:

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "r2b_identity_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )

        raise SystemExit("STOP_R2B_DATA_IDENTITY_MISMATCH")

    # 5) indicator compare
    idx = py_dtp.index.intersection(tv.index)

    rows = []
    mismatches = []

    for key, canon in TV_INDICATOR_FIELDS.items():

        if canon not in tv.columns:
            print(
                f"[WARN] TV missing indicator column for {key}; skipped",
                file=sys.stderr,
            )
            continue

        rep = compare_field(
            py_dtp[canon].loc[idx],
            tv[canon].loc[idx],
            canon,
            integer=(canon == "trend"),
        )
        rows.append(rep)

        for ts, pv, tvv in rep["_mismatch_rows"]:
            mismatches.append(
                {
                    "timestamp": ts,
                    "field": canon,
                    "python_value": pv,
                    "tradingview_value": tvv,
                }
            )

    indicator_rows = [
        {k: v for k, v in r.items() if k != "_mismatch_rows"}
        for r in rows
    ]

    indicator_parity_pass = (
        "PASS"
        if rows and all(r["n_mismatch"] == 0 for r in rows)
        else "FAIL"
    )

    summary["indicator_parity_pass"] = indicator_parity_pass

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "r2b_identity_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    pd.DataFrame(indicator_rows).to_csv(
        out_dir / "r2b_indicator_summary.csv",
        index=False,
    )
    pd.DataFrame(
        mismatches,
        columns=["timestamp", "field", "python_value", "tradingview_value"],
    ).to_csv(out_dir / "r2b_mismatches.csv", index=False)

    print(
        f"[PARITY] data_identity_pass={summary['data_identity_pass']} "
        f"indicator_parity_pass={indicator_parity_pass}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# synthetic comparator tests (验证 comparator, 非 TradingView parity)
# ---------------------------------------------------------------------------

def _utc_index(*strings: str) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [pd.Timestamp(s) for s in strings]
    ).tz_localize("UTC")


def run_comparator_tests() -> None:

    print("[SELFTEST] comparator tests", flush=True)

    # --- timestamp ---
    a_idx = _utc_index("2026-01-01 00:00", "2026-01-01 00:05")
    b_idx = a_idx
    assert a_idx.difference(b_idx).empty and b_idx.difference(a_idx).empty
    print("[SELFTEST] timestamp identical: PASS", flush=True)

    c_idx = _utc_index("2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10")
    assert c_idx.difference(a_idx).size == 1
    print("[SELFTEST] timestamp one-missing: PASS", flush=True)

    dup = pd.Series(
        _utc_index("2026-01-01 00:00", "2026-01-01 00:00")
    )
    try:
        check_timeline(dup)
        raise SystemExit("SELFTEST_FAIL: duplicate not detected")
    except SystemExit as e:
        assert "DUPLICATE" in str(e)
    print("[SELFTEST] timestamp duplicate: PASS", flush=True)

    sh = pd.Series(
        pd.to_datetime(
            ["2026-01-01 08:00", "2026-01-01 08:05"]
        ).tz_localize("Asia/Shanghai")
    )
    utc = pd.Series(
        pd.to_datetime(
            ["2026-01-01 00:00", "2026-01-01 00:05"]
        ).tz_localize("UTC")
    )
    assert (
        to_utc_timestamp(sh, None).equals(to_utc_timestamp(utc, None))
    )
    print("[SELFTEST] timestamp timezone-shift: PASS", flush=True)

    # --- OHLC ---
    def ohlc_df(idx, vals):
        return pd.DataFrame(
            {
                "open": vals,
                "high": vals,
                "low": vals,
                "close": vals,
            },
            index=idx,
        )

    o_idx = _utc_index(
        "2026-01-01 00:00", "2026-01-01 00:05",
        "2026-01-01 00:10", "2026-01-01 00:15",
    )
    base = np.array([1.0, 2.0, 3.0, 4.0])
    py = ohlc_df(o_idx, base)
    tv = ohlc_df(o_idx, base)
    g = identity_gate(py, tv)
    assert g["data_identity_pass"]
    assert all(g[f"{c}_mismatch"] == 0 for c in OHLC_COLS)
    print("[SELFTEST] OHLC exact: PASS", flush=True)

    tv10 = ohlc_df(o_idx, base + 1e-10)
    g10 = identity_gate(py, tv10)
    assert g10["data_identity_pass"], g10
    print("[SELFTEST] OHLC 1e-10 (within atol): PASS", flush=True)

    tv5 = ohlc_df(o_idx, base + 1e-5)
    g5 = identity_gate(py, tv5)
    assert not g5["data_identity_pass"]
    assert g5["first_ohlc_mismatch"] is not None
    print("[SELFTEST] OHLC 1e-5 (gate fail): PASS", flush=True)

    # --- indicator ---
    s_idx = _utc_index(
        "2026-01-01 00:00", "2026-01-01 00:05",
        "2026-01-01 00:10", "2026-01-01 00:15",
    )
    pyv = pd.Series([1.0, 2.0, 3.0, 4.0], index=s_idx)
    tvv = pd.Series([1.0, 2.0, 3.0, 4.0], index=s_idx)
    r = compare_field(pyv, tvv, "x")
    assert r["n_mismatch"] == 0
    print("[SELFTEST] indicator exact: PASS", flush=True)

    tvv2 = pd.Series([1.0, 2.0, 9.0, 4.0], index=s_idx)
    r2 = compare_field(pyv, tvv2, "x")
    assert r2["n_mismatch"] == 1
    assert r2["first_mismatch_timestamp"] == str(s_idx[2])
    print("[SELFTEST] indicator one-mismatch: PASS", flush=True)

    # different warmup start
    pyw = pd.Series([1.0, 2.0, 3.0, 4.0], index=s_idx)
    tvw = pd.Series([np.nan, np.nan, 3.0, 4.0], index=s_idx)
    rw = compare_field(pyw, tvw, "x")
    assert rw["warmup_mismatch"]
    print("[SELFTEST] indicator different-warmup: PASS", flush=True)

    # trend state mismatch (integer)
    pt = pd.Series([-1.0, 1.0, 1.0, -1.0], index=s_idx)
    tt = pd.Series([-1.0, 1.0, -1.0, -1.0], index=s_idx)
    rt = compare_field(pt, tt, "trend", integer=True)
    assert rt["n_mismatch"] == 1
    assert rt["first_mismatch_timestamp"] == str(s_idx[2])
    print("[SELFTEST] indicator trend-state-mismatch: PASS", flush=True)

    print("[SELFTEST] ALL PASS", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:

    ap = argparse.ArgumentParser(
        description="R2B-Prep TradingView parity harness (DTP only)"
    )
    ap.add_argument("--tv-csv", default=None)
    ap.add_argument("--symbol", default="AG")
    ap.add_argument("--timeframe", type=int, default=5)
    ap.add_argument("--tv-timezone", default=None)
    ap.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "artifacts" / "pine_runtime_parity"),
    )
    ap.add_argument("--self-test", action="store_true")

    args = ap.parse_args()

    if args.self_test:
        run_comparator_tests()
        return

    if args.tv_csv is None:
        print(
            "no --tv-csv supplied; nothing to compare. "
            "Use --self-test for comparator tests.",
            file=sys.stderr,
        )
        return

    run_parity(
        tv_csv=args.tv_csv,
        symbol=args.symbol,
        timeframe=args.timeframe,
        tv_timezone=args.tv_timezone,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
