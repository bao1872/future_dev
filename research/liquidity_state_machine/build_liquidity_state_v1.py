"""L1 — Liquidity map + MTF state snapshot + interactions + state machine。

严守 REVIEWER_DECISION_V1_1：
  - 4h 只作 CLOCK_4H_ENVIRONMENT（env_direction_4h），无 4h 结构/流动性
  - trend_struct = swing_bias（截断测试已证明 causal）
  - Displacement 已删除，不检验、不留占位
  - competing events：reclaim vs (penetration 方向 BOS)；
    post-reclaim：reversal MSS vs penetration 方向 BOS
  - 全部在 5m 解析第一层
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from research.build_ob_candidate_universe_v3 import aggregate_4h_from_1h
from research.build_pytdx_panel import aggregate_15m
from research.dsa_adapter import compute_dsa_canonical
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.ob_trigger_snapshot import (aggregate_1h_from_15m,
                                          build_full_ob_smc_tf)
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags

RESULTS = "research/analysis_results/liquidity_state_machine_v1"
TF_PERIOD = {"5m": pd.Timedelta(minutes=5), "15m": pd.Timedelta(minutes=15),
             "1h": pd.Timedelta(hours=1)}


def infer_tick(closes: np.ndarray) -> float:
    d = np.abs(np.diff(closes))
    d = d[d > 0]
    return float(np.min(d)) if len(d) else 1.0


def prep(sym: str):
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    five["volume"] = five["trade"].astype(float)
    five["trading_day"] = five["trading_day"].astype(str)
    fifteen = aggregate_15m(five)
    oneh = aggregate_1h_from_15m(fifteen)
    for d in (fifteen, oneh):
        if "volume" not in d.columns and "trade" in d.columns:
            d["volume"] = d["trade"].astype(float)
    env4 = aggregate_4h_from_1h(oneh)
    return five, fifteen, oneh, env4


def bar_end(times: pd.Series, tf: str) -> pd.Series:
    return pd.to_datetime(times) + TF_PERIOD[tf]


def trend_frame(bars, tf):
    smc = build_full_ob_smc_tf(bars.copy())
    st = pd.DataFrame(smc["state_timeline"])
    t = pd.to_datetime(bars["bar_start_time"])
    et = t.iloc[st["bar_index"].to_numpy()] + TF_PERIOD[tf]
    return pd.DataFrame(dict(
        available_time=et.to_numpy(),
        swing_bias=st["swing_bias"].to_numpy(int),
        internal_bias=st["internal_bias"].to_numpy(int),
    )), smc


def env4_frame(env4, oneh):
    """env_direction_4h：只用 last completed CLOCK_4H_ENVIRONMENT。"""
    e = env4.copy()
    dsa = compute_dsa_canonical(e)
    out = pd.DataFrame(dict(
        bucket_start=pd.to_datetime(e["bar_start_time"]).to_numpy(),
        bucket_end=pd.to_datetime(e["bar_end_time"]).to_numpy(),
        env_direction_4h=pd.to_numeric(
            dsa["dsa_direction"], errors="coerce").to_numpy(),
        component_1h_count=pd.to_numeric(
            e.get("component_1h_count", np.nan), errors="coerce").to_numpy(),
    ))
    # bucket 完整可知 = 其最后一根成分 1h bar 结束
    out["env_available_time"] = out["bucket_end"]
    return out


def liquidity_levels(sym, five, fifteen, oneh, smcs, tick):
    rows = []
    # --- canonical structural ---
    for tf, bars in (("5m", five), ("15m", fifteen), ("1h", oneh)):
        smc = smcs[tf]
        t = pd.to_datetime(bars["bar_start_time"])
        per = TF_PERIOD[tf]

        for p in smc["pivots"]:
            ty = p["type"]
            if ty not in ("swing_high", "swing_low"):
                continue
            ci = int(p["confirmed_index"])
            if ci >= len(t):
                continue
            rows.append(dict(
                symbol=sym, source_tf=tf,
                liquidity_type=("CONFIRMED_SWING_HIGH" if ty == "swing_high"
                                else "CONFIRMED_SWING_LOW"),
                liquidity_scope=tf,
                side=+1 if ty == "swing_high" else -1,
                price=float(p["level"]),
                origin_time=pd.Timestamp(p["anchor_time"]),
                available_time=pd.Timestamp(t.iloc[ci]) + per,
            ))
        for q in smc.get("equal_highs_lows", []):
            ty = q.get("type")
            if ty not in ("EQH", "EQL"):
                continue
            ci = int(q["confirmed_index"])
            if ci >= len(t):
                continue
            rows.append(dict(
                symbol=sym, source_tf=tf,
                liquidity_type=("CANONICAL_EQH" if ty == "EQH"
                                else "CANONICAL_EQL"),
                liquidity_scope=tf,
                side=+1 if ty == "EQH" else -1,
                price=float(q["level"]),
                origin_time=pd.Timestamp(q["anchor_time"]),
                available_time=pd.Timestamp(t.iloc[ci]) + per,
            ))

    # --- research-derived time liquidity ---
    f = five
    day = f["trading_day"].to_numpy()
    t5 = pd.to_datetime(f["bar_start_time"])
    hi, lo = f["high"].to_numpy(float), f["low"].to_numpy(float)

    # prev trading day
    dg = pd.DataFrame(dict(d=day, h=hi, l=lo)).groupby("d")
    days = sorted(dg.groups.keys())
    for i in range(1, len(days)):
        prev, cur = days[i - 1], days[i]
        g = dg.get_group(prev)
        first_cur = np.flatnonzero(day == cur)[0]
        av = t5.iloc[first_cur]
        rows.append(dict(symbol=sym, source_tf="TRADING_DAY",
                         liquidity_type="PREV_TRADING_DAY_HIGH",
                         liquidity_scope="TRADING_DAY", side=+1,
                         price=float(g["h"].max()),
                         origin_time=av, available_time=av))
        rows.append(dict(symbol=sym, source_tf="TRADING_DAY",
                         liquidity_type="PREV_TRADING_DAY_LOW",
                         liquidity_scope="TRADING_DAY", side=-1,
                         price=float(g["l"].min()),
                         origin_time=av, available_time=av))

    # prev contiguous session segment (5m 断口分段)
    brk = t5.diff() > pd.Timedelta(minutes=5)
    seg = brk.cumsum().to_numpy()
    segs = sorted(set(seg))
    for i in range(1, len(segs)):
        ps, cs = segs[i - 1], segs[i]
        m = seg == ps
        first_cur = np.flatnonzero(seg == cs)[0]
        av = t5.iloc[first_cur]
        rows.append(dict(symbol=sym, source_tf="CONTIG_SESSION",
                         liquidity_type="PREV_CONTIG_SESSION_HIGH",
                         liquidity_scope="CONTIG_SESSION", side=+1,
                         price=float(hi[m].max()),
                         origin_time=av, available_time=av))
        rows.append(dict(symbol=sym, source_tf="CONTIG_SESSION",
                         liquidity_type="PREV_CONTIG_SESSION_LOW",
                         liquidity_scope="CONTIG_SESSION", side=-1,
                         price=float(lo[m].min()),
                         origin_time=av, available_time=av))

    # prev trading week (ISO year+week of trading_day)
    iso = pd.to_datetime(pd.Series(days)).dt.isocalendar()
    wk = list(zip(iso["year"].tolist(), iso["week"].tolist()))
    day2wk = dict(zip(days, wk))
    uniq_wk = sorted(set(wk))
    for i in range(1, len(uniq_wk)):
        pw, cw = uniq_wk[i - 1], uniq_wk[i]
        pdays = [d for d in days if day2wk[d] == pw]
        cdays = [d for d in days if day2wk[d] == cw]
        m = np.isin(day, pdays)
        first_cur = np.flatnonzero(np.isin(day, cdays))[0]
        av = t5.iloc[first_cur]
        rows.append(dict(symbol=sym, source_tf="TRADING_WEEK",
                         liquidity_type="PREV_TRADING_WEEK_HIGH",
                         liquidity_scope="TRADING_WEEK", side=+1,
                         price=float(hi[m].max()),
                         origin_time=av, available_time=av))
        rows.append(dict(symbol=sym, source_tf="TRADING_WEEK",
                         liquidity_type="PREV_TRADING_WEEK_LOW",
                         liquidity_scope="TRADING_WEEK", side=-1,
                         price=float(lo[m].min()),
                         origin_time=av, available_time=av))

    lv = pd.DataFrame(rows)
    lv["liquidity_id"] = (
        lv["symbol"] + "|" + lv["liquidity_type"] + "|"
        + lv["source_tf"] + "|"
        + lv["available_time"].astype(str) + "|"
        + lv["price"].astype(str))
    return lv.drop_duplicates(subset=["liquidity_id"]).reset_index(drop=True)


def build_interactions(sym, five, lv, trends, env4d, tick):
    t5 = pd.to_datetime(five["bar_start_time"]).to_numpy()
    hi = five["high"].to_numpy(float)
    lo = five["low"].to_numpy(float)
    cl = five["close"].to_numpy(float)
    op = five["open"].to_numpy(float)
    disc = discontinuity_flags(sym)
    n = len(five)
    t5s = pd.Series(t5)

    # 5m 结构事件（BOS/CHoCH）按 available_time
    ev = trends["5m"]["events"]
    evs = []
    for e in ev:
        ci = int(e["confirmed_index"])
        if ci >= n:
            continue
        evs.append(dict(available_time=t5[ci] + TF_PERIOD["5m"],
                        kind=e["type"], bias=int(e["bias"])))
    evd = pd.DataFrame(evs).sort_values("available_time") if evs else \
        pd.DataFrame(columns=["available_time", "kind", "bias"])

    rows = []
    for r in lv.itertuples(index=False):
        side = int(r.side)
        lvl = float(r.price)
        av = pd.Timestamp(r.available_time)
        start = int(np.searchsorted(t5, np.datetime64(av), side="left"))
        if start >= n:
            continue
        # first valid touch
        ti = None
        for i in range(start, n):
            touched = (hi[i] >= lvl) if side == +1 else (lo[i] <= lvl)
            if touched:
                ti = i
                break
        if ti is None:
            continue
        pen = ((hi[ti] >= lvl + tick) if side == +1
               else (lo[ti] <= lvl - tick))
        rec = dict(
            interaction_id=f"{r.liquidity_id}|{ti}",
            symbol=sym, liquidity_id=r.liquidity_id,
            liquidity_type=r.liquidity_type,
            liquidity_source_tf=r.source_tf,
            liquidity_scope=r.liquidity_scope,
            liquidity_side=side, level_price=lvl,
            level_available_time=av,
            interaction_time=t5[ti], touch_ordinal=1,
            touch=True, penetrated=bool(pen),
            penetration_direction=side,
        )
        # depth
        if pen:
            if side == +1:
                rec["penetration_depth_ticks"] = (hi[ti] - lvl) / tick
            else:
                rec["penetration_depth_ticks"] = (lvl - lo[ti]) / tick
        # competing resolution on 5m
        state, stime, amb = "PENETRATED_NO_RESOLUTION", None, False
        if pen:
            # acceptance 必须是 penetration bar 之后**新出现**的 BOS，
            # 不能把历史上任何同向 BOS 当成 acceptance（曾导致 ACCEPTED 虚高）
            pen_av = t5[ti] + TF_PERIOD["5m"]
            for j in range(ti, n):
                if disc[j]:
                    state, stime = "ROLL_CENSORED", t5[j]
                    break
                reclaimed = (cl[j] < lvl) if side == +1 else (cl[j] > lvl)
                at = t5[j] + TF_PERIOD["5m"]
                g = evd[(evd["available_time"] > pen_av)
                        & (evd["available_time"] <= at)]
                acc = bool(((g["kind"] == "BOS")
                            & (g["bias"] == side)).any())
                if reclaimed and acc:
                    state, stime, amb = "AMBIGUOUS_INTRABAR_ORDER", t5[j], True
                    break
                if reclaimed:
                    state, stime = "RECLAIMED", t5[j]
                    break
                if acc:
                    state, stime = "ACCEPTED", t5[j]
                    break
            else:
                state = "END_OF_DATA_CENSORED"
        rec["competing_state"] = state
        rec["resolution_time"] = stime
        rec["order_resolved"] = (not amb)
        rec["reclaimed"] = state == "RECLAIMED"
        rec["accepted"] = state == "ACCEPTED"
        rec["reversal_direction"] = -side

        # post-reclaim competing
        if state == "RECLAIMED":
            st = pd.Timestamp(stime)
            sub = evd[evd["available_time"] >= st]
            post, ptime = "END_OR_ROLL_CENSORED", None
            for k in range(int(np.searchsorted(t5, np.datetime64(st))), n):
                if disc[k]:
                    post, ptime = "ROLL_CENSORED", t5[k]
                    break
                at = t5[k] + TF_PERIOD["5m"]
                g = sub[sub["available_time"] <= at]
                rev = g[(g["kind"] == "CHoCH") & (g["bias"] == -side)]
                res = g[(g["kind"] == "BOS") & (g["bias"] == side)]
                if len(rev) and len(res):
                    post, ptime = "AMBIGUOUS_SAME_TIMESTAMP", t5[k]
                    break
                if len(rev):
                    post, ptime = "REVERSAL_MSS_CONFIRMED", t5[k]
                    break
                if len(res):
                    post, ptime = "REJECTION_FAILED_REACCEPTED", t5[k]
                    break
            rec["post_reclaim_state"] = post
            rec["post_reclaim_time"] = ptime
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    import os
    os.makedirs(RESULTS, exist_ok=True)
    all_lv, all_it = [], []
    t0 = time.perf_counter()
    for sym in ("AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P",
                "RB", "RU", "SC", "SN", "TA"):
        five, fifteen, oneh, env4 = prep(sym)
        tick = infer_tick(five["close"].to_numpy(float))
        smcs, trends = {}, {}
        for tf, bars in (("5m", five), ("15m", fifteen), ("1h", oneh)):
            tf_df, smc = trend_frame(bars, tf)
            smcs[tf] = smc
            trends[tf] = dict(frame=tf_df, events=smc["events"])
        env4d = env4_frame(env4, oneh)

        lv = liquidity_levels(sym, five, fifteen, oneh, smcs, tick)
        it = build_interactions(sym, five, lv, trends, env4d, tick)

        # MTF as-of join
        def asof(df, right, col):
            l = df.sort_values("interaction_time")
            r = right.sort_values("available_time")
            out = pd.merge_asof(l, r, left_on="interaction_time",
                                right_on="available_time",
                                direction="backward",
                                allow_exact_matches=True)
            assert (out["available_time"].isna()
                    | (out["available_time"]
                       <= out["interaction_time"])).all()
            return out
        if len(it):
            for tf in ("5m", "15m", "1h"):
                r = trends[tf]["frame"].rename(columns={
                    "swing_bias": f"trend_struct_{tf}",
                    "internal_bias": f"internal_bias_{tf}"})
                it = asof(it, r, tf)
                it = it.drop(columns=["available_time"], errors="ignore")
            it = asof(it, env4d.rename(columns={
                "env_available_time": "available_time"}), "env4")
            it = it.drop(columns=["available_time"], errors="ignore")

        all_lv.append(lv)
        all_it.append(it)
        print(f"  {sym}: levels={len(lv)} interactions={len(it)} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)

    lv = pd.concat(all_lv, ignore_index=True)
    it = pd.concat(all_it, ignore_index=True)
    lv.to_parquet(f"{RESULTS}/liquidity_levels.parquet", index=False)
    it.to_parquet(f"{RESULTS}/interactions_primary.parquet", index=False)
    it.to_parquet(f"{RESULTS}/interactions_all.parquet", index=False)
    print(f"\nlevels={len(lv)} interactions={len(it)}")
    print(it["competing_state"].value_counts().to_string())
    print("\nL1_DONE")


if __name__ == "__main__":
    main()
