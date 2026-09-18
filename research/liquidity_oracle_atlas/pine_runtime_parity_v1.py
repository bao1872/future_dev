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
    PINE_SOURCE_REFS, verify_pine_source_refs, git_head

raw 5m bars 复用仓库官方 owner:
    research.export_ob_trigger_execution_v21.load_raw_5m
    (路径合同 research/exports/v3r_5m/{SYMBOL}_5m.csv; R2B 用 bar_start_time)

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
import tempfile
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
)

# 仓库官方 raw 5m owner (唯一 raw loader; 禁止第三套).
from research.export_ob_trigger_execution_v21 import load_raw_5m

# ---------------------------------------------------------------------------
# paths / constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
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
    复用仓库官方 raw 5m owner (export_ob_trigger_execution_v21.load_raw_5m)。

    该 owner 读取 research/exports/v3r_5m/{SYMBOL}_5m.csv, 并负责
    parse bar_start_time / duplicate 5m gate / sort。
    symbol 找不到 -> owner 抛 FileNotFoundError (fail, 不吞)。

    R2B K 线 identity 使用 bar_start_time 作为 chart bar timestamp
    (TDX 合同: bar_start_time = interval_end - period)。
    naive -> Asia/Shanghai localize -> UTC。
    """

    five = load_raw_5m(symbol.upper())

    idx = to_utc_timestamp(five["bar_start_time"], PY_TZ)

    return pd.DataFrame(
        {
            "open": five["open"].to_numpy(dtype=float),
            "high": five["high"].to_numpy(dtype=float),
            "low": five["low"].to_numpy(dtype=float),
            "close": five["close"].to_numpy(dtype=float),
        },
        index=idx,
    )


def load_tv_csv(
    path: str,
    tv_timezone: str | None,
) -> pd.DataFrame:
    """
    读取 TradingView 导出 CSV, 规范化 OHLC + indicator 列到 UTC index。

    timeline gate 顺序 (fail-closed):
        parse timestamp -> check duplicate -> check monotonic -> set_index
    禁止先 sort 再检查 (否则 NON_MONOTONIC 永远抓不到)。
    """

    df = pd.read_csv(path)

    time_col = find_col(df, "time")
    if time_col is None:
        raise SystemExit("STOP_R2B_TV_MISSING_TIME_COLUMN")

    df["time"] = to_utc_timestamp(df[time_col], tv_timezone)

    # timeline gate BEFORE any sort / set_index
    check_timeline(df["time"])

    df = df.set_index("time")

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

    timestamp contract (防止 false PASS):
        TV 是 Python 历史的子区间, 允许 Python 拥有更多历史。
        但 TV 的每一根 bar 都必须存在于 Python: n_overlap == n_tv。
        TV 出现 Python 不存在的 timestamp -> FAIL (比较交集却声称身份一致)。
        TV 还必须是 Python row sequence 的连续子区间
        (n_internal_python_rows_skipped == 0); 否则 TV 中间漏 bar 仍可 false PASS。
    """

    n_python = len(py)
    n_tv = len(tv)
    overlap = py.index.intersection(tv.index)
    n_overlap = len(overlap)

    only_tv = tv.index.difference(py.index)
    only_py = py.index.difference(tv.index)

    n_tv_only = len(only_tv)
    n_python_only = len(only_py)

    tv_timestamp_coverage = (n_overlap / n_tv) if n_tv else 0.0
    python_timestamp_coverage = (n_overlap / n_python) if n_python else 0.0

    # TV-only = 会导致 identity failure 的 timestamp (证据).
    # Python-only = 合法额外历史, 不得称为 mismatch.
    first_tv_only_timestamp = str(only_tv.min()) if n_tv_only else None
    first_python_only_timestamp = (
        str(only_py.min()) if n_python_only else None
    )

    out = {
        "n_python": n_python,
        "n_tv": n_tv,
        "n_timestamp_overlap": n_overlap,
        "timestamp_overlap_rate": n_overlap / max(n_python, n_tv, 1),
        "n_python_only": n_python_only,
        "n_tv_only": n_tv_only,
        "tv_timestamp_coverage": tv_timestamp_coverage,
        "python_timestamp_coverage": python_timestamp_coverage,
        "first_tv_only_timestamp": first_tv_only_timestamp,
        "first_python_only_timestamp": first_python_only_timestamp,
    }

    # TV -> Python positional continuity (report only; 不因正常休市自动 FAIL)
    positions = py.index.get_indexer(tv.index)
    pos_ok = positions[positions >= 0]
    if len(pos_ok) > 1:
        tv_python_position_monotonic = bool(np.all(np.diff(pos_ok) >= 0))
        n_internal_skipped = int(
            np.sum(np.maximum(np.diff(pos_ok) - 1, 0))
        )
    else:
        tv_python_position_monotonic = True
        n_internal_skipped = 0
    out["tv_python_position_monotonic"] = tv_python_position_monotonic
    out["n_internal_python_rows_skipped"] = n_internal_skipped

    # TV 必须是 Python row sequence 的连续子区间:
    #   1) TV 每根都在 Python (n_overlap == n_tv)
    #   2) TV 中间不能漏 bar (n_internal_python_rows_skipped == 0)
    # 正常午休 / 夜盘断点下 Python 也没有那些行, row 仍连续, 不触发.
    continuous_subsequence_pass = bool(
        n_tv > 0
        and n_overlap == n_tv
        and n_internal_skipped == 0
    )
    out["continuous_subsequence_pass"] = continuous_subsequence_pass

    if n_overlap == 0:
        for col in OHLC_COLS:
            out[f"{col}_mismatch"] = 0
            out[f"max_abs_{col}_error"] = None
        out["first_ohlc_mismatch"] = None
        out["continuous_subsequence_pass"] = False
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

        out[f"{col}_mismatch"] = int(mm.sum())

        if both.any():
            out[f"max_abs_{col}_error"] = float(diff[both].max())
        else:
            out[f"max_abs_{col}_error"] = None

        if out[f"{col}_mismatch"] and first_ohlc is None:
            first_ohlc = str(overlap[int(np.argmax(mm))])

    out["first_ohlc_mismatch"] = first_ohlc
    ohlc_pass = all(out[f"{col}_mismatch"] == 0 for col in OHLC_COLS)
    out["data_identity_pass"] = bool(
        continuous_subsequence_pass and ohlc_pass
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

    diff = np.abs(a - b)
    if integer:
        mm = both & (a.astype("int64") != b.astype("int64"))
    else:
        tol = atol + rtol * np.abs(b)
        mm = both & (diff > tol)

    # error statistics 在 jointly finite rows 上真实计算 (含 integer/trend):
    # 不把 mismatch 判定与误差统计混为一谈 (trend +1 vs -1 的误差应记 2).
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
    evaluable = both | one_nan
    n_evaluable = int(evaluable.sum())
    mismatch_rate = (n_mismatch / n_evaluable) if n_evaluable else np.nan

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
        "n_evaluable": n_evaluable,
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

    # 0) HTF 当前硬阻断: canonical trading_day/segment/disc 未接, 不得 PASS
    if timeframe != 5 and not HTF_DAY_SEGMENT_CANONICAL:
        raise SystemExit(
            f"STOP_R2B_HTF_CANONICAL_CALENDAR_NOT_READY:{timeframe}"
        )

    # 1) 先证明审核的正是 pin 住的 DTP 源文件
    verify_pine_source_refs()

    # 2) Python 参考 (5m raw bars, 复用官方 owner)
    py = load_python_5m(symbol)
    check_timeline(py.index.to_series())

    py_ohlc = py[OHLC_COLS]
    py_dtp = compute_dtp(py)

    # 3) TradingView CSV
    tv = load_tv_csv(tv_csv, tv_timezone)
    check_timeline(tv.index.to_series())

    # 4) Data Identity Gate
    gate = identity_gate(py_ohlc, tv[OHLC_COLS])

    tv_csv_sha = hashlib.sha256(Path(tv_csv).read_bytes()).hexdigest()
    timezone_reported = tv_timezone if tv_timezone else "UTC(tz-aware)"

    summary = {
        "source_sha": DTP_SOURCE_SHA,
        "harness_sha": git_head(),
        "symbol": symbol,
        "timeframe": timeframe,
        "tv_csv_sha256": tv_csv_sha,
        "timezone": timezone_reported,
        **gate,
        "indicator_parity_pass": "NOT_RUN",
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 5) identity FAIL 仍先保存完整证据, 再 STOP (不继续 DTP)
    if not gate["data_identity_pass"]:
        (out_dir / "r2b_identity_summary.json").write_text(
            json.dumps(summary, indent=2, default=str)
        )
        raise SystemExit("STOP_R2B_DATA_IDENTITY_MISMATCH")

    # 6) Required indicator contract: 六列必须全部存在, 否则 fail closed
    _require_all_indicator_columns(tv)

    # 7) indicator compare
    idx = py_dtp.index.intersection(tv.index)

    rows = []
    mismatches = []

    for key, canon in TV_INDICATOR_FIELDS.items():
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

    indicator_parity_pass = decide_indicator_parity(rows)
    summary["indicator_parity_pass"] = indicator_parity_pass
    summary["insufficient_finite_overlap"] = any(
        r["n_both_finite"] <= 0 for r in rows
    )

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


def _require_all_indicator_columns(tv: pd.DataFrame) -> None:
    """
    DTP parity 固定要求六列全部存在。
    缺任意一个 -> fail closed, 禁止 WARN + skip / 部分字段 PASS。
    """

    for key, canon in TV_INDICATOR_FIELDS.items():
        if canon not in tv.columns:
            raise SystemExit(
                f"STOP_R2B_TV_MISSING_INDICATOR_COLUMN:{key}"
            )


def decide_indicator_parity(rows: list[dict]) -> str:
    """
    indicator_parity_pass 判定 (独立可测):
        实际比较字段 == required(6)
        AND 每个字段 n_both_finite > 0 (双方全 NaN 不能算 PASS)
        AND 所有 n_mismatch == 0
        AND 所有 warmup_mismatch == False
    否则 FAIL。
    """

    required = {"sma50", "atr200", "avg_diff", "p100", "avg_col", "trend"}
    compared = {r["name"] for r in rows}

    if compared != required:
        return "FAIL"
    for r in rows:
        # 双方都是 NaN -> n_both_finite=0, 实际没比较任何有效数值, 不能 PASS.
        if r["n_both_finite"] <= 0:
            return "FAIL"
        if r["n_mismatch"] != 0:
            return "FAIL"
        if r["warmup_mismatch"]:
            return "FAIL"

    return "PASS"


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

    # --- OHLC identity (含 timestamp coverage contract) ---
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

    # --- Regression Test 1: TV extra timestamp -> MUST FAIL ---
    # 比较了交集却声称身份一致 = false PASS
    tv_idx3 = _utc_index(
        "2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:10",
    )
    py1 = ohlc_df(o_idx[:2], base[:2])          # 00:00, 00:05
    tv1 = ohlc_df(tv_idx3, base[:3])            # 00:00, 00:05, 00:10 (前两根相同)
    g1 = identity_gate(py1, tv1)
    assert not g1["data_identity_pass"], g1
    assert g1["n_tv_only"] == 1, g1
    assert abs(g1["tv_timestamp_coverage"] - 2 / 3) < 1e-9, g1
    print("[SELFTEST] R1 TV extra timestamp -> FAIL: PASS", flush=True)

    # --- Regression Test 2: Python extra history allowed -> PASS ---
    extra_idx = _utc_index(
        "2025-12-31 23:55", "2026-01-01 00:00", "2026-01-01 00:05",
    )
    py2 = ohlc_df(extra_idx, [0.5, 1.0, 2.0])
    tv2 = ohlc_df(o_idx[:2], base[:2])          # 00:00, 00:05
    g2 = identity_gate(py2, tv2)
    assert g2["data_identity_pass"], g2
    assert g2["n_python_only"] == 1, g2
    assert g2["n_tv_only"] == 0, g2
    print("[SELFTEST] R2 Python extra history -> PASS: PASS", flush=True)

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

    # --- Regression Test 3: loader non-monotonic (real path, no pre-sort) ---
    csv3 = (
        "time,open,high,low,close\n"
        "2026-01-01 00:00,1,1,1,1\n"
        "2026-01-01 00:10,2,2,2,2\n"
        "2026-01-01 00:05,3,3,3,3\n"
    )
    p3 = Path(tempfile.mktemp(suffix=".csv"))
    p3.write_text(csv3)
    try:
        load_tv_csv(str(p3), "UTC")
        raise SystemExit("SELFTEST_FAIL: non-monotonic not detected")
    except SystemExit as e:
        assert "NON_MONOTONIC" in str(e), e
    print("[SELFTEST] R3 loader non-monotonic -> STOP: PASS", flush=True)

    # --- Regression Test 4: missing required indicator -> fail closed ---
    csv4 = (
        "time,open,high,low,close,R2B_SMA50,R2B_ATR200,"
        "R2B_AVG_DIFF,R2B_AVG_COL,R2B_TREND\n"
        "2026-01-01 00:00,1,1,1,1,1,1,0,0,-1\n"
        "2026-01-01 00:05,2,2,2,2,2,2,0,0,1\n"
    )
    p4 = Path(tempfile.mktemp(suffix=".csv"))
    p4.write_text(csv4)
    tv4 = load_tv_csv(str(p4), "UTC")
    try:
        _require_all_indicator_columns(tv4)
        raise SystemExit("SELFTEST_FAIL: missing indicator not detected")
    except SystemExit as e:
        assert "MISSING_INDICATOR_COLUMN:R2B_P100" in str(e), e
    print("[SELFTEST] R4 missing required indicator -> STOP: PASS", flush=True)

    # --- Regression Test 5: partial indicator cannot PASS ---
    s2_idx = s_idx[:2]
    partial_rows = [
        compare_field(
            pd.Series([1.0, 2.0], index=s2_idx),
            pd.Series([1.0, 2.0], index=s2_idx),
            "sma50",
        ),
        compare_field(
            pd.Series([1.0, 2.0], index=s2_idx),
            pd.Series([1.0, 2.0], index=s2_idx),
            "atr200",
        ),
    ]
    assert decide_indicator_parity(partial_rows) == "FAIL"

    full_rows = [
        compare_field(
            pd.Series([1.0, 2.0], index=s2_idx),
            pd.Series([1.0, 2.0], index=s2_idx),
            canon,
            integer=(canon == "trend"),
        )
        for canon in (
            "sma50", "atr200", "avg_diff", "p100", "avg_col", "trend"
        )
    ]
    assert decide_indicator_parity(full_rows) == "PASS"
    print("[SELFTEST] R5 partial indicator cannot PASS: PASS", flush=True)

    # --- Regression Test 6: HTF hard stop ---
    for tf in (15, 60, 240):
        try:
            run_parity(
                "--nonexistent.csv", "AG", tf, None,
                tempfile.mkdtemp(),
            )
            raise SystemExit("SELFTEST_FAIL: HTF not blocked")
        except SystemExit as e:
            assert (
                f"HTF_CANONICAL_CALENDAR_NOT_READY:{tf}" in str(e)
            ), e
    print("[SELFTEST] R6 HTF hard stop: PASS", flush=True)

    # --- Regression Test 7: required field all-NaN cannot PASS ---
    nan_idx = _utc_index("2026-01-01 00:00", "2026-01-01 00:05")
    nan_a = pd.Series([np.nan, np.nan], index=nan_idx)
    nan_b = pd.Series([np.nan, np.nan], index=nan_idx)
    r7_rows = [compare_field(nan_a, nan_b, "sma50")]
    for canon in ("atr200", "avg_diff", "p100", "avg_col", "trend"):
        r7_rows.append(
            compare_field(
                pd.Series([1.0, 2.0], index=nan_idx),
                pd.Series([1.0, 2.0], index=nan_idx),
                canon,
                integer=(canon == "trend"),
            )
        )
    assert decide_indicator_parity(r7_rows) == "FAIL"
    sma_row = r7_rows[0]
    assert sma_row["n_both_finite"] == 0, sma_row
    assert sma_row["n_mismatch"] == 0, sma_row
    assert sma_row["warmup_mismatch"] is False, sma_row
    print("[SELFTEST] R7 all-NaN required field -> FAIL: PASS", flush=True)

    # --- Regression Test 8: TV internal missing bar ---
    py8_idx = _utc_index(
        "2026-01-01 00:00", "2026-01-01 00:05",
        "2026-01-01 00:10", "2026-01-01 00:15",
    )
    tv8_idx = _utc_index(
        "2026-01-01 00:00", "2026-01-01 00:05", "2026-01-01 00:15",
    )
    v8 = np.array([1.0, 2.0, 3.0, 4.0])
    py8 = ohlc_df(py8_idx, v8)
    tv8 = ohlc_df(tv8_idx, [1.0, 2.0, 4.0])
    g8 = identity_gate(py8, tv8)
    assert g8["n_tv_only"] == 0, g8
    assert g8["n_timestamp_overlap"] == 3, g8
    assert g8["n_internal_python_rows_skipped"] == 1, g8
    assert g8["continuous_subsequence_pass"] is False, g8
    assert g8["data_identity_pass"] is False, g8
    print("[SELFTEST] R8 TV internal missing bar -> FAIL: PASS", flush=True)

    # --- Regression Test 9: normal session gap must NOT false fail ---
    gap_idx = _utc_index(
        "2026-01-01 10:00", "2026-01-01 10:05",
        "2026-01-01 13:30", "2026-01-01 13:35",
    )
    v9 = np.array([1.0, 2.0, 3.0, 4.0])
    py9 = ohlc_df(gap_idx, v9)
    tv9 = ohlc_df(gap_idx, v9)
    g9 = identity_gate(py9, tv9)
    assert g9["n_internal_python_rows_skipped"] == 0, g9
    assert g9["continuous_subsequence_pass"] is True, g9
    assert g9["data_identity_pass"] is True, g9
    print("[SELFTEST] R9 session gap row-continuous -> PASS: PASS", flush=True)

    # --- Regression Test 10: python-only history evidence semantics ---
    py10_idx = _utc_index(
        "2025-12-31 23:55", "2026-01-01 00:00", "2026-01-01 00:05",
    )
    tv10_idx = _utc_index("2026-01-01 00:00", "2026-01-01 00:05")
    py10 = ohlc_df(py10_idx, [0.5, 1.0, 2.0])
    tv10 = ohlc_df(tv10_idx, [1.0, 2.0])
    g10 = identity_gate(py10, tv10)
    assert g10["data_identity_pass"] is True, g10
    assert g10["first_tv_only_timestamp"] is None, g10
    assert g10["first_python_only_timestamp"] is not None, g10
    assert "first_timestamp_mismatch" not in g10, g10
    print("[SELFTEST] R10 python-only timestamp evidence: PASS", flush=True)

    # --- Regression Test 11: trend error statistics real ---
    t_idx = _utc_index("2026-01-01 00:00", "2026-01-01 00:05")
    pt = pd.Series([-1.0, 1.0], index=t_idx)
    tt = pd.Series([-1.0, -1.0], index=t_idx)
    rt = compare_field(pt, tt, "trend", integer=True)
    assert rt["n_mismatch"] == 1, rt
    assert rt["max_abs_error"] == 2.0, rt
    assert rt["mean_abs_error"] == 1.0, rt
    print("[SELFTEST] R11 trend error statistics: PASS", flush=True)

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
