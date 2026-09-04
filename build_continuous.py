#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 TqSdk 主连（KQ.m@SHFE.ag）的拼接序列转换为前复权连续序列。

背景
----
TqSdk 的 KQ.m@ 主连是若干个真实月份合约按时间段直接拼接出来的，换月处
存在合约价差造成的跳空，直接拿去回测会污染收益率序列。

本脚本做的事：
  1. 逐 bar 把主连序列与候选月份合约做精确匹配（close 与 close_oi 同时相等），
     得到「哪根 bar 来自哪个合约」的归属表。
  2. 在每次换月的临界点，用新旧两个合约在切换前一根 bar 的收盘价计算价差比例。
  3. 以最新合约为基准做前复权（最新价格不变，历史价格按累计比例缩放）。

为什么不用持仓量跳变来检测换月
------------------------------
TqSdk 在换月时会把新合约的 open_oi 接成旧合约的 close_oi，导致
open_oi[t] == close_oi[t-1] 在全序列上恒成立，该信号完全不可用。
单 bar 内持仓量变化率虽能抓到大部分换月（+30%~+57% 的极端离群），
但实测会漏掉价差较小的换月（2025-12-29 那次仅 4.8%，淹没在噪声里）。
因此这里采用逐 bar 精确匹配，结果可审计。

用法
----
    python build_continuous.py --refresh   # 联网重建归属表（需快期账号）
    python build_continuous.py             # 用缓存离线重算复权序列

输出
----
    silver_main_data/adjusted/silver_main_{15m,1h,4h}_adj.csv
    silver_main_data/rollover_owner.csv    逐bar归属表
    silver_main_data/rollover_segments.csv 主连合约构成图
    silver_main_data/rollover_report.csv   每次换月的价差与累计复权因子
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "silver_main_data"
ADJ_DIR = DATA_DIR / "adjusted"
OWNER_FILE = DATA_DIR / "rollover_owner.csv"
SEG_FILE = DATA_DIR / "rollover_segments.csv"
CLOSE_FILE = DATA_DIR / "contract_closes.csv"
REPORT_FILE = DATA_DIR / "rollover_report.csv"

SYMBOL = "KQ.m@SHFE.ag"
TIMEFRAMES = ("15m", "1h", "4h")
PRICE_COLUMNS = ("open", "high", "low", "close")
DURATION = {"15m": 15 * 60, "1h": 60 * 60, "4h": 4 * 60 * 60}
NS_PER_SEC = 1_000_000_000


# =========================
# 凭据
# =========================

def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def fmt_ns(ns: int) -> str:
    return (pd.to_datetime(ns, unit="ns", utc=True)
            .tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S"))


# =========================
# 步骤 1：主连 -> 合约 归属表
# =========================

def _prep(df: pd.DataFrame) -> pd.DataFrame:
    d = df[(df["datetime"].notna()) & (df["datetime"] > 0)].copy()
    d["ns"] = d["datetime"].astype("int64")
    return d.set_index("ns")[["close", "close_oi"]]


def candidate_months(start_ns: int, end_ns: int) -> list[str]:
    """生成覆盖数据区间、前后各留 2 个月余量的 SHFE 白银合约代码。"""
    start = (pd.to_datetime(start_ns, unit="ns", utc=True)
             .tz_convert("Asia/Shanghai").tz_localize(None).to_period("M"))
    end = (pd.to_datetime(end_ns, unit="ns", utc=True)
           .tz_convert("Asia/Shanghai").tz_localize(None).to_period("M"))
    codes, cur = [], start - 2
    last = end + 2
    while cur <= last:
        codes.append(f"SHFE.ag{cur.year % 100:02d}{cur.month:02d}")
        cur += 1
    return codes


def build_owner_map() -> tuple[pd.DataFrame, pd.DataFrame]:
    """联网构建归属表。返回 (owner_df[ns,contract], closes_df[ns,contract...])。"""
    from tqsdk import TqApi, TqAuth

    load_dotenv(BASE_DIR / ".env")
    user, pwd = os.environ.get("TQ_USER", ""), os.environ.get("TQ_PASSWORD", "")
    if not user or not pwd:
        raise SystemExit("ERROR: 未设置 TQ_USER / TQ_PASSWORD（写入 .env 或导出环境变量）")

    api = TqApi(auth=TqAuth(user, pwd))
    try:
        main_raw = api.get_kline_serial(SYMBOL, DURATION["15m"], data_length=8000)
        while not api.is_serial_ready(main_raw):
            api.wait_update()
        main = _prep(main_raw)
        print(f"主连: {len(main)} 根 {fmt_ns(main.index[0])} -> {fmt_ns(main.index[-1])}")

        owner = pd.Series(index=main.index, dtype="object")
        closes: dict[str, pd.Series] = {}

        for code in candidate_months(main.index[0], main.index[-1]):
            try:
                raw = api.get_kline_serial(code, DURATION["15m"], data_length=8000)
                while not api.is_serial_ready(raw):
                    api.wait_update()
            except Exception as exc:       # 未上市/已摘牌合约会超时或报错
                print(f"  {code}: 跳过 ({type(exc).__name__})")
                continue
            sub = _prep(raw)
            if len(sub) == 0:
                continue
            joined = main.join(sub, how="inner", lsuffix="_m", rsuffix="_s")
            hit = ((joined["close_m"] == joined["close_s"])
                   & (joined["close_oi_m"] == joined["close_oi_s"]))
            hit = hit[hit]
            if len(hit) == 0:
                continue
            owner.loc[hit.index] = code
            closes[code] = sub["close"]
            print(f"  {code}: 命中 {len(hit)} 根 "
                  f"{fmt_ns(hit.index[0])} -> {fmt_ns(hit.index[-1])}")

        owner = _fill_isolated_gaps(owner)
        unmatched = int(owner.isna().sum())
        print(f"未匹配 bar: {unmatched} / {len(owner)}")
        return (owner.rename("contract").rename_axis("ns").reset_index(),
                pd.DataFrame(closes))
    finally:
        api.close()


def _fill_isolated_gaps(owner: pd.Series, max_gap: int = 2) -> pd.Series:
    """
    填补长度 <= max_gap 的孤立未匹配 bar。
    两端分别处理：序列末尾没有"后一段"、开头没有"前一段"，以存在的一侧为准。
    """
    out = owner.copy()
    isna = out.isna().values
    i, n = 0, len(out)
    while i < n:
        if not isna[i]:
            i += 1
            continue
        j = i
        while j < n and isna[j]:
            j += 1
        if (j - i) > max_gap:
            i = j
            continue
        prev = out.iloc[i - 1] if i > 0 else None
        nxt = out.iloc[j] if j < n else None
        if prev is not None and nxt is not None:
            if prev == nxt:
                out.iloc[i:j] = prev
        elif prev is not None:      # 末尾缺口：沿用前一段
            out.iloc[i:j] = prev
        elif nxt is not None:       # 开头缺口：沿用后一段
            out.iloc[i:j] = nxt
        i = j
    return out


def to_segments(owner_df: pd.DataFrame) -> pd.DataFrame:
    seg = owner_df["contract"].fillna("<UNKNOWN>")
    rows = []
    for _, part in owner_df.groupby((seg != seg.shift()).cumsum()):
        rows.append({
            "contract": seg.iloc[part.index].iloc[0],
            "start_ns": int(part["ns"].iloc[0]),
            "end_ns": int(part["ns"].iloc[-1]),
            "bars": int(len(part)),
        })
    df = pd.DataFrame(rows)
    df.insert(1, "start", df["start_ns"].map(fmt_ns))
    df.insert(2, "end", df["end_ns"].map(fmt_ns))
    return df


# =========================
# 步骤 2：换月价差与复权因子
# =========================

def compute_factors(segments: pd.DataFrame,
                    closes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    前复权：以最新合约为基准，历史各段按累计价差比例缩放。
    ratio_j = 新合约收盘价 / 旧合约收盘价，取切换前一根 bar（此时两合约都在交易）。
    """
    n = len(segments)
    ratios = [1.0] * n
    report = []

    for j in range(n - 1):
        old_c, new_c = segments.loc[j, "contract"], segments.loc[j + 1, "contract"]
        ref_ns = segments.loc[j, "end_ns"]        # 切换前最后一根 bar
        if old_c not in closes.columns or new_c not in closes.columns:
            raise SystemExit(f"ERROR: 缺少 {old_c} 或 {new_c} 的价格数据")
        p_old, p_new = closes[old_c].get(ref_ns), closes[new_c].get(ref_ns)
        if pd.isna(p_old) or pd.isna(p_new):
            raise SystemExit(f"ERROR: {fmt_ns(ref_ns)} 缺少 {old_c}/{new_c} 收盘价")
        p_old, p_new = float(p_old), float(p_new)
        ratios[j] = p_new / p_old
        report.append({
            "switch_at": segments.loc[j + 1, "start"],
            "from_contract": old_c,
            "to_contract": new_c,
            "ref_bar": fmt_ns(ref_ns),
            "old_close": p_old,
            "new_close": p_new,
            "spread_pct": (p_new / p_old - 1) * 100,
            "ratio": ratios[j],
        })

    factors = [1.0] * n
    for j in range(n - 2, -1, -1):
        factors[j] = factors[j + 1] * ratios[j]

    report_df = pd.DataFrame(report)
    if len(report_df):
        report_df["cum_factor"] = [factors[j] for j in range(len(report_df))]
    segments = segments.copy()
    segments["factor"] = factors
    return segments, report_df


# =========================
# 步骤 3：按 bar 实际覆盖映射到合约
# =========================

def assign_contracts(bar_ns: pd.Series, owner_df: pd.DataFrame,
                     tf: str) -> tuple[pd.Series, int]:
    """
    按 bar 的时间窗 [t, t_next) 内实际包含的 15m bar 归属，取占比最高的合约。
    返回 (每根bar的合约, 跨合约bar数量)。
    """
    all_ns = owner_df["ns"].values
    owners = owner_df["contract"].values
    result, straddled = [], 0

    for i, ns in enumerate(bar_ns.values):
        t_next = (bar_ns.values[i + 1] if i + 1 < len(bar_ns)
                  else ns + DURATION[tf] * NS_PER_SEC)
        lo = int(pd.Series(all_ns).searchsorted(ns, side="left"))
        hi = int(pd.Series(all_ns).searchsorted(t_next, side="left"))
        if hi <= lo:                              # 窗口内无 15m bar，回退到最后已知段
            k = max(0, int(pd.Series(all_ns).searchsorted(ns, side="right")) - 1)
            result.append(owners[k] if k < len(owners) else "<UNKNOWN>")
            continue
        vals = owners[lo:hi]
        counts = pd.Series(vals).value_counts()
        if len(counts) > 1:
            straddled += 1
        result.append(counts.idxmax())

    return pd.Series(result, index=bar_ns.index), straddled


def adjust_timeframe(tf: str, segments: pd.DataFrame,
                     owner_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"silver_main_{tf}.csv", parse_dates=["datetime"])
    # 直接用 TqSdk 原始纳秒时间戳，不要从 datetime 列反推：
    # pandas 3.0 起默认时间精度为微秒，astype("int64") 得到的是微秒，会差 1000 倍。
    bar_ns = df["datetime_ns"].astype("int64")

    owners, straddled = assign_contracts(bar_ns, owner_df, tf)
    factor_map = segments.set_index("contract")["factor"].to_dict()
    factors = owners.map(factor_map)
    if factors.isna().any():
        raise SystemExit(f"ERROR: {tf} 有 {int(factors.isna().sum())} 根 bar 未匹配到合约")

    out = df.copy()
    for col in PRICE_COLUMNS:
        out[col] = (out[col] * factors).round(2)
    out["contract"] = owners.values
    out["adj_factor"] = factors.round(8).values

    raw_max = df["close"].pct_change().abs().max() * 100
    adj_max = out["close"].pct_change().abs().max() * 100
    print(f"[{tf}] {len(out)} 根 | 跨合约bar {straddled} | "
          f"单bar最大波动 {raw_max:.2f}% -> {adj_max:.2f}%")
    return out


# =========================
# main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="联网重建主连合约归属表")
    args = ap.parse_args()

    ADJ_DIR.mkdir(parents=True, exist_ok=True)

    if args.refresh or not OWNER_FILE.is_file():
        print("== 构建主连合约归属表 ==")
        owner_df, closes = build_owner_map()
        owner_df.to_csv(OWNER_FILE, index=False)
        closes.to_csv(CLOSE_FILE, index_label="ns")
        print(f"已缓存 {OWNER_FILE.name}, {CLOSE_FILE.name}\n")
    else:
        owner_df = pd.read_csv(OWNER_FILE)
        closes = pd.read_csv(CLOSE_FILE, index_col=0)
        # 缓存中可能残留未匹配 bar（例如当时正在形成的最后一根），这里补一次
        owner_df["contract"] = _fill_isolated_gaps(owner_df["contract"])
        print(f"== 使用缓存归属表 {OWNER_FILE.name} ==")

    segments = to_segments(owner_df)
    print("\n=== 主连合约构成 ===")
    print(segments[["contract", "start", "end", "bars"]].to_string(index=False))

    segments, report = compute_factors(segments, closes)
    print("\n=== 换月价差与复权因子 ===")
    if len(report):
        print(report.to_string(index=False,
                               formatters={"spread_pct": "{:+.3f}".format,
                                           "ratio": "{:.6f}".format,
                                           "cum_factor": "{:.6f}".format}))
    report.to_csv(REPORT_FILE, index=False, encoding="utf-8-sig")
    segments.to_csv(SEG_FILE, index=False, encoding="utf-8-sig")

    print("\n=== 生成前复权序列 ===")
    for tf in TIMEFRAMES:
        out = adjust_timeframe(tf, segments, owner_df)
        dst = ADJ_DIR / f"silver_main_{tf}_adj.csv"
        out.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"    -> {dst}")

    print("\n注意: volume / open_oi / close_oi 未做复权（成交量与持仓量不可按比例缩放）。")
    print(f"复权明细: {REPORT_FILE}")


if __name__ == "__main__":
    main()
