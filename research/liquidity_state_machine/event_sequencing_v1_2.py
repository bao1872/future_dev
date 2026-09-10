"""Event Sequencing Audit v1.2。

修正 v1.1 的事件时间语义：
  1. canonical trading_day（不是自然日）
  2. pre-interaction roll censor
  3. penetration = 严格空间穿透（inferred tick 只作描述）
  4. penetration_bar_start / penetration_available_time 分离
  5. Stage 1 同 bar 分类（SAME_BAR_RECLAIM / CLOSE_BEYOND / CLOSE_AT_LEVEL）
  6. Stage 2 仅 CLOSE_BEYOND 进入对称 competing risks
  7. post-reclaim 严格 available_time > reclaim_available_time
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
FIVE = TF_PERIOD["5m"]


def infer_tick(closes):
    d = np.abs(np.diff(closes))
    d = d[d > 0]
    return float(np.min(d)) if len(d) else 1.0


def prep(sym):
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    five["volume"] = five["trade"].astype(float)
    five["trading_day"] = five["trading_day"].astype(str)
    fifteen = aggregate_15m(five)
    oneh = aggregate_1h_from_15m(fifteen)
    env4 = aggregate_4h_from_1h(oneh)
    return five, fifteen, oneh, env4


def build_symbol(sym, lv):
    five, fifteen, oneh, env4 = prep(sym)
    tick = infer_tick(five["close"].to_numpy(float))
    t5 = pd.to_datetime(five["bar_start_time"])
    t5n = t5.to_numpy()
    hi = five["high"].to_numpy(float)
    lo = five["low"].to_numpy(float)
    cl = five["close"].to_numpy(float)
    day = five["trading_day"].to_numpy()
    disc = discontinuity_flags(sym)
    n = len(five)

    # 5m canonical structural events with available_time
    smc5 = build_full_ob_smc_tf(five.copy())
    evs = []
    for e in smc5["events"]:
        ci = int(e["confirmed_index"])
        if ci >= n:
            continue
        evs.append(dict(available_time=t5n[ci] + FIVE,
                        kind=e["type"], bias=int(e["bias"])))
    evd = (pd.DataFrame(evs).sort_values("available_time")
           if evs else pd.DataFrame(columns=["available_time", "kind",
                                             "bias"]))
    ev_at = evd["available_time"].to_numpy()
    ev_kind = evd["kind"].to_numpy()
    ev_bias = evd["bias"].to_numpy()

    # trend frames
    trends = {}
    for tf, bars in (("5m", five), ("15m", fifteen), ("1h", oneh)):
        smc = build_full_ob_smc_tf(bars.copy())
        st = pd.DataFrame(smc["state_timeline"])
        bt = pd.to_datetime(bars["bar_start_time"])
        trends[tf] = pd.DataFrame(dict(
            available_time=(bt.iloc[st["bar_index"].to_numpy()]
                            + TF_PERIOD[tf]).to_numpy(),
            swing_bias=st["swing_bias"].to_numpy(int),
            internal_bias=st["internal_bias"].to_numpy(int)))

    dsa = compute_dsa_canonical(env4)
    envd = pd.DataFrame(dict(
        available_time=pd.to_datetime(env4["bar_end_time"]).to_numpy(),
        env_direction_4h=pd.to_numeric(dsa["dsa_direction"],
                                       errors="coerce").to_numpy()))

    rows = []
    cens = []
    lvs = lv[lv["symbol"] == sym]
    for r in lvs.itertuples(index=False):
        side = int(r.side)
        lvl = float(r.price)
        start = int(np.searchsorted(t5n, np.datetime64(r.available_time),
                                    side="left"))
        if start >= n:
            continue
        ti, pre_state = None, None
        for i in range(start, n):
            if disc[i]:
                pre_state = "ROLL_CENSORED_PRE_INTERACTION"
                break
            reached = (hi[i] >= lvl) if side == +1 else (lo[i] <= lvl)
            if reached:
                ti = i
                break
        if pre_state is not None:
            cens.append(dict(symbol=sym, liquidity_type=r.liquidity_type,
                             state=pre_state))
            continue
        if ti is None:
            continue

        # --- penetration: strict spatial ---
        pen = (hi[ti] > lvl) if side == +1 else (lo[ti] < lvl)
        rec = dict(
            interaction_id=f"{r.liquidity_id}|{ti}",
            symbol=sym, liquidity_id=r.liquidity_id,
            liquidity_type=r.liquidity_type,
            liquidity_source_tf=r.source_tf,
            liquidity_scope=r.liquidity_scope,
            liquidity_side=side, level_price=lvl,
            level_available_time=r.available_time,
            interaction_time=t5n[ti], touch_ordinal=1,
            penetration_bar_start=t5n[ti],
            penetration_available_time=t5n[ti] + FIVE,
            penetrated=bool(pen),
            penetration_direction=side,
            trading_day=str(day[ti]),           # canonical
        )
        if pen:
            rec["penetration_depth_inferred_ticks"] = (
                (hi[ti] - lvl) / tick if side == +1
                else (lvl - lo[ti]) / tick)
        if not pen:
            rec["stage1"] = "TOUCH_NO_PENETRATION"
            rows.append(rec)
            continue

        # --- Stage 1: same-bar classification (symmetric) ---
        if side == +1:
            sb = cl[ti] < lvl
            cb = cl[ti] > lvl
        else:
            sb = cl[ti] > lvl
            cb = cl[ti] < lvl
        at_lvl = (not sb) and (not cb)
        if sb:
            rec["stage1"] = "SAME_BAR_RECLAIM"
            rec["reclaim_bar_start"] = t5n[ti]
            rec["reclaim_available_time"] = t5n[ti] + FIVE
        elif cb:
            rec["stage1"] = "CLOSE_BEYOND"
        else:
            rec["stage1"] = "CLOSE_AT_LEVEL"

        # --- Stage 2: only CLOSE_BEYOND ---
        if cb:
            pen_av = t5n[ti] + FIVE
            st2, st2t = "END_OF_DATA_CENSORED", None
            for j in range(ti + 1, n):
                if disc[j]:
                    st2, st2t = "ROLL_CENSORED", t5n[j]
                    break
                at = t5n[j] + FIVE
                later = (cl[j] < lvl) if side == +1 else (cl[j] > lvl)
                g = (ev_at > pen_av) & (ev_at <= at)
                acc = bool(((ev_kind[g] == "BOS")
                            & (ev_bias[g] == side)).any())
                if later and acc:
                    st2, st2t = "AMBIGUOUS_SAME_TIMESTAMP", t5n[j]
                    break
                if later:
                    st2, st2t = "LATER_RECLAIM", t5n[j]
                    break
                if acc:
                    st2, st2t = "STRUCTURAL_ACCEPTANCE", t5n[j]
                    break
            rec["stage2"] = st2
            rec["stage2_time"] = st2t
            if st2t is not None:
                rec["bars_to_stage2"] = int(
                    np.searchsorted(t5n, np.datetime64(st2t)) - ti)

        # --- post-reclaim strict ordering ---
        rt = rec.get("reclaim_available_time")
        if rt is not None:
            sub = evd[evd["available_time"] > rt]
            if len(sub):
                k0 = sub.iloc[0]["kind"]
                b0 = int(sub.iloc[0]["bias"])
                rev = (k0 == "CHoCH") and (b0 == -side)
                res = (k0 == "BOS") and (b0 == side)
                rec["post_reclaim_state"] = (
                    "REVERSAL_MSS_CONFIRMED" if rev else
                    "REJECTION_FAILED_REACCEPTED" if res else "OTHER")
                rec["post_reclaim_time"] = sub.iloc[0]["available_time"]
            else:
                rec["post_reclaim_state"] = "NO_LATER_EVENT"
        rows.append(rec)

    it = pd.DataFrame(rows) if rows else pd.DataFrame()
    if len(it):
        for tf in ("5m", "15m", "1h"):
            r = trends[tf].rename(columns={
                "swing_bias": f"trend_struct_{tf}",
                "internal_bias": f"internal_bias_{tf}"})
            it = pd.merge_asof(it.sort_values("interaction_time"), r,
                               left_on="interaction_time",
                               right_on="available_time",
                               direction="backward",
                               allow_exact_matches=True).drop(
                columns=["available_time"], errors="ignore")
        it = pd.merge_asof(it.sort_values("interaction_time"), envd,
                           left_on="interaction_time",
                           right_on="available_time",
                           direction="backward",
                           allow_exact_matches=True).drop(
            columns=["available_time"], errors="ignore")
        assert (it["interaction_time"].notna()).all()
    return it, pd.DataFrame(cens)


def main():
    lv = pd.read_parquet(f"{RESULTS}/liquidity_levels.parquet")
    lv["available_time"] = pd.to_datetime(lv["available_time"])
    allit, allc = [], []
    t0 = time.perf_counter()
    for sym in sorted(lv["symbol"].unique()):
        it, c = build_symbol(sym, lv)
        allit.append(it)
        allc.append(c)
        print(f"  {sym}: interactions={len(it)} censored={len(c)} "
              f"({time.perf_counter()-t0:.0f}s)", flush=True)
    it = pd.concat([x for x in allit if len(x)], ignore_index=True)
    cens = pd.concat(allc, ignore_index=True) if any(
        len(x) for x in allc) else pd.DataFrame()
    it.to_parquet(f"{RESULTS}/interactions_v1_2.parquet", index=False)
    cens.to_csv(f"{RESULTS}/pre_interaction_roll_censor.csv", index=False,
                encoding="utf-8-sig")
    print(f"\ninteractions={len(it)}  "
          f"pre_interaction_roll_censored="
          f"{len(cens)} ({len(cens)/max(1,len(lv))*100:.2f}% of levels)")
    if len(it):
        print("\nstage1:"); print(it["stage1"].value_counts().to_string())
        print("\nstage2:"); print(it["stage2"].value_counts(
            dropna=False).to_string())
    print("\nV1_2_DONE")


if __name__ == "__main__":
    main()
