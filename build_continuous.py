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
import time
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "silver_main_data"
ADJ_DIR = DATA_DIR / "adjusted"


def owner_file(tf: str) -> Path:
    return DATA_DIR / f"rollover_owner_{tf}.csv"


def close_file(tf: str) -> Path:
    return DATA_DIR / f"contract_closes_{tf}.csv"


def seg_file(tf: str) -> Path:
    return DATA_DIR / f"rollover_segments_{tf}.csv"


def report_file(tf: str) -> Path:
    return DATA_DIR / f"rollover_report_{tf}.csv"

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


def build_owner_map(tf: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    联网构建某个周期的归属表。归属表与目标序列同周期，bar 一一对应。
    返回 (owner_df[ns,contract], closes_df[ns,contract...])。
    """
    from tqsdk import TqApi, TqAuth

    load_dotenv(BASE_DIR / ".env")
    user, pwd = os.environ.get("TQ_USER", ""), os.environ.get("TQ_PASSWORD", "")
    if not user or not pwd:
        raise SystemExit("ERROR: 未设置 TQ_USER / TQ_PASSWORD（写入 .env 或导出环境变量）")

    # 用已下载、已对齐到 15m 窗口的原始 CSV 确定候选合约范围，
    # 避免对全历史合约逐个试探（1h/4h 取满 8000 根会回溯到 2018/2023）。
    raw_path = DATA_DIR / f"silver_main_{tf}.csv"
    if not raw_path.is_file():
        raise SystemExit(f"ERROR: 缺少 {raw_path.name}，请先运行 download_silver_main_tqsdk.py")
    raw_ns = pd.read_csv(raw_path)["datetime_ns"]
    win_min = int(pd.to_numeric(raw_ns).min())
    win_max = int(pd.to_numeric(raw_ns).max())
    codes = candidate_months(win_min, win_max)

    api = TqApi(auth=TqAuth(user, pwd))
    try:
        main_raw = api.get_kline_serial(SYMBOL, DURATION[tf], data_length=8000)
        while not api.is_serial_ready(main_raw):
            api.wait_update()
        main = _prep(main_raw)
        # 只保留对齐窗口内的 bar 用于匹配（与原始 CSV 一一对应）
        main = main[(main.index >= win_min) & (main.index <= win_max)]
        print(f"[{tf}] 主连(窗口内): {len(main)} 根 "
              f"{fmt_ns(main.index[0])} -> {fmt_ns(main.index[-1])}")

        owner = pd.Series(index=main.index, dtype="object")
        closes: dict[str, pd.Series] = {}
        t0 = time.time()

        for n_done, code in enumerate(codes, 1):
            if n_done % 20 == 0 or n_done == len(codes):
                print(f"    扫描合约 {n_done}/{len(codes)} "
                      f"({time.time() - t0:.0f}s, 已匹配 {int(owner.notna().sum())} 根)")
            try:
                raw = api.get_kline_serial(code, DURATION[tf], data_length=8000)
                while not api.is_serial_ready(raw):
                    api.wait_update()
            except Exception as exc:       # 未上市/无数据合约会报错或超时
                continue
            sub = _prep(raw)
            if len(sub) == 0:
                continue
            joined = main.join(sub, how="inner", lsuffix="_m", rsuffix="_s")
            # 仅按收盘价精确匹配：TqSdk 换月时会把新合约的 open_oi 接成旧合约的
            # close_oi，导致 (close, close_oi) 同时相等在换月临界 bar 上必然失败；
            # 而同一时间戳只有一个合约在交易，收盘价本身即可唯一确定归属。
            hit = (joined["close_m"] == joined["close_s"])
            hit = hit[hit]
            if len(hit) == 0:
                continue
            owner.loc[hit.index] = code
            closes[code] = sub["close"]
            print(f"    {code}: 命中 {len(hit)} 根 "
                  f"{fmt_ns(hit.index[0])} -> {fmt_ns(hit.index[-1])}")

        owner = _fill_isolated_gaps(owner)
        unmatched = int(owner.isna().sum())
        print(f"[{tf}] 未匹配 bar: {unmatched} / {len(owner)} "
              f"(耗时 {time.time() - t0:.0f}s)")
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
# 步骤 3：映射到合约并施加复权因子
# =========================

def adjust_timeframe(tf: str, segments: pd.DataFrame,
                     owner_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"silver_main_{tf}.csv", parse_dates=["datetime"])
    # 直接用 TqSdk 原始纳秒时间戳，不要从 datetime 列反推：
    # pandas 3.0 起默认时间精度为微秒，astype("int64") 得到的是微秒，会差 1000 倍。
    df["_ns"] = df["datetime_ns"].astype("int64")

    # 归属表与该周期的目标序列同频率，按 ns 直接左连接即可一一对应。
    merged = df.merge(owner_df[["ns", "contract"]], how="left",
                      left_on="_ns", right_on="ns")
    missing = int(merged["contract"].isna().sum())
    if missing:
        raise SystemExit(
            f"ERROR: {tf} 有 {missing} 根 bar 不在归属表内，"
            f"请先执行 --refresh 重建（数据区间可能已扩展）")

    factor_map = segments.set_index("contract")["factor"].to_dict()
    factors = merged["contract"].map(factor_map)
    if factors.isna().any():
        unknown = sorted(set(merged.loc[factors.isna(), "contract"]))
        raise SystemExit(f"ERROR: {tf} 有合约缺少复权因子: {unknown}")

    out = merged.drop(columns=["_ns", "ns"]).copy()
    for col in PRICE_COLUMNS:
        out[col] = (out[col] * factors).round(2)
    out["adj_factor"] = factors.round(8).values

    raw_max = df["close"].pct_change().abs().max() * 100
    adj_max = out["close"].pct_change().abs().max() * 100
    print(f"[{tf}] {len(out)} 根 | {out['contract'].nunique()} 个合约 | "
          f"单bar最大波动 {raw_max:.2f}% -> {adj_max:.2f}%")
    return out


# =========================
# main
# =========================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="联网重建主连合约归属表")
    ap.add_argument("--tf", nargs="+", choices=list(TIMEFRAMES), default=list(TIMEFRAMES),
                    help="只处理指定周期（默认全部）")
    args = ap.parse_args()

    ADJ_DIR.mkdir(parents=True, exist_ok=True)

    print("=== 生成前复权序列 ===")
    for tf in args.tf:
        print(f"\n---- {tf} ----")
        if args.refresh or not owner_file(tf).is_file():
            owner_df, closes = build_owner_map(tf)
            owner_df.to_csv(owner_file(tf), index=False)
            closes.to_csv(close_file(tf), index_label="ns")
        else:
            owner_df = pd.read_csv(owner_file(tf))
            closes = pd.read_csv(close_file(tf), index_col=0)
            # 缓存中可能残留未匹配 bar（例如当时正在形成的最后一根），补一次
            owner_df["contract"] = _fill_isolated_gaps(owner_df["contract"])

        segments = to_segments(owner_df)
        print(f"  合约构成: {len(segments)} 段")
        segments, report = compute_factors(segments, closes)
        report.to_csv(report_file(tf), index=False, encoding="utf-8-sig")
        segments.to_csv(seg_file(tf), index=False, encoding="utf-8-sig")
        if len(report):
            top = report.reindex(report["spread_pct"].abs().nlargest(3).index)
            worst = top["spread_pct"].abs().max()
            print(f"  换月 {len(report)} 次 | 最大价差 {worst:.3f}% | "
                  f"累计因子 {segments['factor'].min():.4f} ~ {segments['factor'].max():.4f}")

        out = adjust_timeframe(tf, segments, owner_df)
        dst = ADJ_DIR / f"silver_main_{tf}_adj.csv"
        out.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"  -> {dst}")

    print("\n注意: volume / open_oi / close_oi 未做复权（成交量与持仓量不可按比例缩放）。")
    print("归属表与换月明细: silver_main_data/rollover_*_{tf}.csv")


if __name__ == "__main__":
    main()
