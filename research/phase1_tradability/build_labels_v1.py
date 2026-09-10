"""Phase 1 标签构造：方向中性 1:2.5 Tradability（first-passage barrier）。

决策时点：
    decision_bar_index = touch_5m_bar_index
    reference_price    = close[decision_bar_index]
    R_ref              = ATR5（decision bar 及之前）
    start_idx          = decision_bar_index + 1

不使用固定 12 根 horizon，采用 first-passage barrier resolution。
UP / DOWN 两个纯标签假设；同 bar 同时触及 target 与 stop → AMBIGUOUS_INTRABAR
（不再用保守 stop 强行标 0，标签质量优先）。

输出：
    labels_v1.parquet（可再生，不入库）
    label_profile.csv
    resolution_profile.csv
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research.phase1_tradability.phase1_contract_v1 import (
    RESULTS, ROLL_GAP_ATR_THRESHOLD, STOP_R, SYMBOLS, TARGET_R, compute_atr5,
    discontinuity_flags, get_bars,
)

FIVE_MIN = np.timedelta64(5, "m")


@dataclass
class TradabilityResult:
    label: int | None
    status: str
    opportunity_side: str | None
    resolution_bar_index: int | None
    bars_to_resolution: int | None


def label_tradability_event(
    start_idx: int,
    reference_price: float,
    r_ref: float,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    discontinuity_before_bar: np.ndarray,
) -> TradabilityResult:
    up_target = reference_price + TARGET_R * r_ref
    up_stop = reference_price - STOP_R * r_ref
    down_target = reference_price - TARGET_R * r_ref
    down_stop = reference_price + STOP_R * r_ref

    up_alive = True
    down_alive = True
    n = len(open_)

    for i in range(start_idx, n):
        if discontinuity_before_bar[i]:
            return TradabilityResult(None, "ROLL_CENSORED", None, i,
                                     i - start_idx)
        o, h, l = open_[i], high[i], low[i]

        # 1) 开盘跳空优先
        if up_alive:
            if o >= up_target:
                return TradabilityResult(1, "RESOLVED", "UP", i,
                                         i - start_idx + 1)
            if o <= up_stop:
                up_alive = False
        if down_alive:
            if o <= down_target:
                return TradabilityResult(1, "RESOLVED", "DOWN", i,
                                         i - start_idx + 1)
            if o >= down_stop:
                down_alive = False
        if not up_alive and not down_alive:
            return TradabilityResult(0, "RESOLVED", None, i,
                                     i - start_idx + 1)

        # 2) bar 内 high / low
        if up_alive:
            hit_t = h >= up_target
            hit_s = l <= up_stop
            if hit_t and hit_s:
                return TradabilityResult(None, "AMBIGUOUS_INTRABAR", None, i,
                                         i - start_idx + 1)
            if hit_t:
                return TradabilityResult(1, "RESOLVED", "UP", i,
                                         i - start_idx + 1)
            if hit_s:
                up_alive = False
        if down_alive:
            hit_t = l <= down_target
            hit_s = h >= down_stop
            if hit_t and hit_s:
                return TradabilityResult(None, "AMBIGUOUS_INTRABAR", None, i,
                                         i - start_idx + 1)
            if hit_t:
                return TradabilityResult(1, "RESOLVED", "DOWN", i,
                                         i - start_idx + 1)
            if hit_s:
                down_alive = False
        if not up_alive and not down_alive:
            return TradabilityResult(0, "RESOLVED", None, i,
                                     i - start_idx + 1)

    return TradabilityResult(None, "END_OF_DATA_CENSORED", None, None, None)


def build(symbols=SYMBOLS, threshold: float = ROLL_GAP_ATR_THRESHOLD,
          quiet: bool = False) -> pd.DataFrame:
    cand = pd.read_parquet(RESULTS / "candidates_v1.parquet")
    recs = []
    for sym in symbols:
        t0 = time.perf_counter()
        bars = get_bars(sym)
        atr = compute_atr5(bars)
        disc = discontinuity_flags(sym, threshold=threshold)
        o, h, l, c, t = (bars["open"], bars["high"], bars["low"],
                         bars["close"], bars["time"])
        n = len(o)
        sub = cand[cand["symbol"] == sym]
        for cid, gid, tf, di in zip(
                sub["candidate_id"], sub["candidate_group_id"],
                sub["source_tf"], sub["touch_5m_bar_index"]):
            di = int(di)
            if di < 0 or di >= n:
                recs.append(dict(candidate_id=cid, symbol=sym, source_tf=tf,
                                 candidate_group_id=gid, label=None,
                                 status="BAD_BAR_INDEX", opportunity_side=None,
                                 bars_to_resolution=None,
                                 resolution_bar_index=None,
                                 decision_time=None, resolution_time=None,
                                 reference_price=None, R_ref=None))
                continue
            p0 = float(c[di])
            r = float(atr[di])
            if not np.isfinite(r) or r <= 0:
                recs.append(dict(candidate_id=cid, symbol=sym, source_tf=tf,
                                 candidate_group_id=gid, label=None,
                                 status="NO_ATR", opportunity_side=None,
                                 bars_to_resolution=None,
                                 resolution_bar_index=None,
                                 decision_time=t[di] + FIVE_MIN,
                                 resolution_time=None,
                                 reference_price=p0, R_ref=r))
                continue
            res = label_tradability_event(di + 1, p0, r, o, h, l, disc)
            rt = (t[res.resolution_bar_index] + FIVE_MIN
                  if res.resolution_bar_index is not None else None)
            recs.append(dict(
                candidate_id=cid, symbol=sym, source_tf=tf,
                candidate_group_id=gid, label=res.label, status=res.status,
                opportunity_side=res.opportunity_side,
                bars_to_resolution=res.bars_to_resolution,
                resolution_bar_index=res.resolution_bar_index,
                decision_time=t[di] + FIVE_MIN, resolution_time=rt,
                reference_price=p0, R_ref=r))
        if not quiet:
            print(f"[labels] {sym}: {len(sub)} events in "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
    return pd.DataFrame(recs)


def smoke_ag(lab: pd.DataFrame, cand: pd.DataFrame):
    """AG only 人工可读抽查：5 个 label=1、5 个 label=0、最多 5 个 censored。"""
    print("\n=== AG SMOKE TEST ===")
    ag = lab[(lab["symbol"] == "AG") & (lab["status"] == "RESOLVED")]
    for want in (1, 0):
        sub = ag[ag["label"] == want].head(5)
        print(f"\n--- label={want} ---")
        for _, r in sub.iterrows():
            print(f"  {r['candidate_id']} tf={r['source_tf']} "
                  f"P0={r['reference_price']:.1f} R={r['R_ref']:.4f} "
                  f"side={r['opportunity_side']} "
                  f"bars={r['bars_to_resolution']}")
    odd = lab[(lab["symbol"] == "AG")
              & (lab["status"].isin(["AMBIGUOUS_INTRABAR", "ROLL_CENSORED",
                                     "END_OF_DATA_CENSORED"]))].head(5)
    print("\n--- censored / ambiguous ---")
    for _, r in odd.iterrows():
        print(f"  {r['candidate_id']} status={r['status']} "
              f"bars={r['bars_to_resolution']}")

    # 逐根打印一个 label=1 的命中序列确认状态机没写反
    print("\n--- 单事件命中序列 ---")
    bars = get_bars("AG")
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    ex = ag[ag["label"] == 1].iloc[0]
    row = cand[cand["candidate_id"] == ex["candidate_id"]].iloc[0]
    di = int(row["touch_5m_bar_index"])
    p0, rr = float(ex["reference_price"]), float(ex["R_ref"])
    print(f"  {ex['candidate_id']} P0={p0:.1f} R={rr:.4f} "
          f"up_target={p0+2.5*rr:.1f} up_stop={p0-rr:.1f} "
          f"down_target={p0-2.5*rr:.1f} down_stop={p0+rr:.1f}")
    for i in range(di + 1, min(di + 1 + int(ex["bars_to_resolution"]) + 1,
                               len(o))):
        print(f"    bar{i}: O={o[i]:.1f} H={h[i]:.1f} L={l[i]:.1f} C={c[i]:.1f}")


def main():
    lab = build(["AG"])          # 1) AG smoke
    smoke_ag(lab, pd.read_parquet(RESULTS / "candidates_v1.parquet"))
    lab = build()                # 2) 全 4 品种
    lab.to_parquet(RESULTS / "labels_v1.parquet", index=False)

    # 3) 标签画像
    rows = [dict(维度="总计", 取值="ALL", 事件数=len(lab))]
    for k, v in lab["status"].value_counts().items():
        rows.append(dict(维度="status", 取值=str(k), 事件数=int(v)))
    for k, v in lab["label"].value_counts(dropna=False).items():
        rows.append(dict(维度="label", 取值=str(k), 事件数=int(v)))
    for dim, col in (("symbol", "symbol"), ("source_tf", "source_tf")):
        for k, g in lab.groupby(col):
            rows.append(dict(维度=f"{dim}×status", 取值=f"{k}|RESOLVED",
                             事件数=int((g["status"] == "RESOLVED").sum())))
            rows.append(dict(维度=f"{dim}×status", 取值=f"{k}|AMBIGUOUS",
                             事件数=int((g["status"] == "AMBIGUOUS_INTRABAR").sum())))
    pd.DataFrame(rows).to_csv(RESULTS / "label_profile.csv", index=False,
                              encoding="utf-8-sig")

    # 4) resolution 画像
    res = lab[lab["status"] == "RESOLVED"]
    rr = []
    q = res["bars_to_resolution"].quantile([.25, .5, .75, .90, .95, .99])
    base = dict(维度="全部", 取值="ALL", 事件数=len(res))
    base.update({f"p{int(p*100)}": float(v) for p, v in q.items()})
    base["max"] = float(res["bars_to_resolution"].max())
    base["mean_bars"] = float(res["bars_to_resolution"].mean())
    rr.append(base)
    for k, g in res.groupby("source_tf"):
        qq = g["bars_to_resolution"].quantile([.25, .5, .75, .90, .95, .99])
        d = dict(维度="source_tf", 取值=str(k), 事件数=len(g))
        d.update({f"p{int(p*100)}": float(v) for p, v in qq.items()})
        d["max"] = float(g["bars_to_resolution"].max())
        d["mean_bars"] = float(g["bars_to_resolution"].mean())
        rr.append(d)
    for k, g in res.groupby("symbol"):
        qq = g["bars_to_resolution"].quantile([.25, .5, .75, .9, .95, .99])
        d = dict(维度="symbol", 取值=str(k), 事件数=len(g))
        d.update({f"p{int(p*100)}": float(v) for p, v in qq.items()})
        d["max"] = float(g["bars_to_resolution"].max())
        d["mean_bars"] = float(g["bars_to_resolution"].mean())
        rr.append(d)
    pdf = pd.DataFrame(rr)
    pdf["median_hours"] = (pdf["p50"] * 5 / 60).round(3)
    pdf.to_csv(RESULTS / "resolution_profile.csv", index=False,
               encoding="utf-8-sig")

    print("\n=== 标签画像 ===")
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n=== resolution（bars）===")
    print(pdf.round(3).to_string(index=False))
    print("\nLABELS_DONE")


if __name__ == "__main__":
    main()
