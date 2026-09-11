"""SMC Oracle Atlas v1.2 —— Oracle 标签语义冻结。

相对 v1.1 的修复：
  1. active liquidity 边界改为 first_penetration_time <= decision_time
     （同 bar 被扫的 liquidity 在收盘时均已 consumed）
  2. Oracle 每行显式不确定性区间：
       best_R_lower / best_R_upper / best_R_is_exact / resolution_class
     censored -> upper = NaN（未知），绝不把下界当完整 Oracle
     NO_ACTIVE_TARGET -> lower = upper = NaN（不是 R=0）
  3. target cluster 保留完整 scope 组成（min/max/count/各周期标志）
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import (
    RESULTS,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1 import (
    BIN_EDGES, BIN_LABELS, RISK_ATR_GRID, SCOPES,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_1 import (
    _ob_fields, build_symbol as _unused,
)
from research.liquidity_state_machine.build_boundary_preoutcome_v3_2 import (
    attach_trend_multi,
)
from research.liquidity_state_machine.build_ob_confluence_v3 import (
    build_ob_map,
)
from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    SymbolData,
)
from research.liquidity_state_machine.match_liquidity_placebo_v2 import rel
from research.phase1_tradability.phase1_contract_v1 import (
    compute_atr5, discontinuity_flags,
)

SCOPE_ORDER = ["5m", "15m", "1h", "CONTIG_SESSION", "TRADING_DAY",
               "TRADING_WEEK"]


def active_mask(master_sym, decision_time):
    """active = 已可用 AND 尚未在 decision_time（含）之前被消费。

    v1.2：decision_time 是 bar 结束，故同 bar 发生的 penetration
    也算已消费（<= 而非 <）。
    """
    av = pd.to_datetime(master_sym["available_time"]).to_numpy() <= decision_time
    fp = pd.to_datetime(master_sym["first_penetration_time"]).to_numpy()
    consumed_by_decision = (~pd.isna(fp)) & (fp <= decision_time)
    return av & (~consumed_by_decision)


def oracle_direction(H, L, entry, atr0, d, tp_sorted, path_censored):
    N = len(H)
    if N == 0:
        return []
    cH = np.maximum.accumulate(H)
    cL = np.minimum.accumulate(L)
    reach = (np.searchsorted(cH, tp_sorted, side="left") if d == +1
             else np.searchsorted(-cL, -tp_sorted, side="left"))
    out = []
    for risk in RISK_ATR_GRID:
        rpx = risk * atr0
        stop = entry - d * rpx
        sm = (cL <= stop) if d == +1 else (cH >= stop)
        si = int(np.argmax(sm)) if sm.any() else -1
        stopped = bool(si >= 0 and sm[si])
        stop_idx = si if stopped else N
        valid = reach < stop_idx
        opt = reach <= stop_idx
        row = dict(risk_ATR=risk, stop_hit=stopped,
                   bars_to_stop=(int(si + 1) if stopped else None))
        for tag, vm in (("conservative", valid), ("optimistic", opt)):
            if vm.any():
                k = int(np.flatnonzero(vm)[-1])
                row[f"{tag}_best_R"] = round(
                    abs(float(tp_sorted[k]) - entry) / rpx, 4)
                row[f"{tag}_best_target_price"] = float(tp_sorted[k])
                row[f"bars_to_best_{tag}"] = int(reach[k] + 1)
                row[f"best_cluster_index_{tag}"] = k
            else:
                row[f"{tag}_best_R"] = 0.0
                row[f"{tag}_best_target_price"] = None
                row[f"bars_to_best_{tag}"] = None
                row[f"best_cluster_index_{tag}"] = None
        amb = bool(stopped and np.any(reach == stop_idx))
        row["ambiguous_intrabar"] = amb

        # ---- 区间语义 ----
        low = row["conservative_best_R"]
        if amb:
            row["best_R_lower"] = low
            row["best_R_upper"] = row["optimistic_best_R"]
            row["best_R_is_exact"] = False
            row["resolution_class"] = "AMBIGUOUS_INTERVAL"
            row["status"] = "AMBIGUOUS_INTRABAR_ORDER"
        elif stopped:
            row["best_R_lower"] = low
            row["best_R_upper"] = low
            row["best_R_is_exact"] = True
            row["status"] = ("TARGET_REACHED_THEN_STOPPED" if low > 0
                             else "STOPPED")
        else:
            # 右删失：只有下界，上界未知
            row["best_R_lower"] = low
            row["best_R_upper"] = np.nan
            row["best_R_is_exact"] = False
            row["status"] = ("TARGET_REACHED_CENSORED" if low > 0
                             else "CENSORED_NO_TARGET")
        row["resolution_class"] = row.get(
            "resolution_class",
            "EXACT_RESOLVED" if stopped else "CENSORED_LOWER_BOUND")
        row["bars_to_best_lower"] = row.get("bars_to_best_conservative")
        row["bars_to_best_upper"] = (
            row.get("bars_to_best_conservative") if row["best_R_is_exact"]
            else (row.get("bars_to_best_optimistic") if amb else np.nan))
        for m in (1, 2, 3):
            pm = entry + d * m * rpx
            mi = (int(np.searchsorted(cH, pm, side="left")) if d == +1
                  else int(np.searchsorted(-cL, -pm, side="left")))
            if mi >= N:
                row[f"first_{m}R_bar"] = None
                row[f"first_{m}R_state"] = "CENSORED_NOT_REACHED"
            elif not stopped:
                row[f"first_{m}R_bar"] = mi + 1
                row[f"first_{m}R_state"] = "REACHED_BEFORE_CENSOR"
            elif mi < stop_idx:
                row[f"first_{m}R_bar"] = mi + 1
                row[f"first_{m}R_state"] = "REACHED_BEFORE_STOP"
            elif mi > stop_idx:
                row[f"first_{m}R_bar"] = None
                row[f"first_{m}R_state"] = "NOT_REACHED_BEFORE_STOP"
            else:
                row[f"first_{m}R_bar"] = None
                row[f"first_{m}R_state"] = "AMBIGUOUS_SAME_BAR"
        out.append(row)
    return out


def build_symbol(sym, contacts, master, obm):
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    t = pd.to_datetime(five["bar_start_time"]).to_numpy()
    hi = five["high"].to_numpy(float)
    lo = five["low"].to_numpy(float)
    cl = five["close"].to_numpy(float)
    atr = compute_atr5(dict(open=five["open"].to_numpy(float), high=hi,
                            low=lo, close=cl, time=t, n=len(five)))
    disc = discontinuity_flags(sym)
    n = len(five)

    ms = master[master["symbol"] == sym].reset_index(drop=True)
    mp = ms["price"].to_numpy(float)
    mtype = ms["liquidity_type"].astype(str).to_numpy()
    mscope = ms["liquidity_scope"].astype(str).to_numpy()
    mid = ms["liquidity_id"].astype(str).to_numpy()
    ms_av = pd.to_datetime(ms["available_time"]).to_numpy()
    ms_fp = pd.to_datetime(ms["first_penetration_time"]).to_numpy()

    cc = contacts[contacts["symbol"] == sym].copy()
    cc["interaction_time"] = pd.to_datetime(cc["decision_time"])
    cc = attach_trend_multi(cc, SymbolData(sym))

    if obm is None or not len(obm):
        obm = build_ob_map(sym)
    ob_av = pd.to_datetime(obm["ob_available_time"]).to_numpy()
    ob_in = pd.to_datetime(obm["ob_inactive_time"]).to_numpy()
    ob_bias = obm["ob_bias"].to_numpy(int)
    ob_zl = obm["zone_low"].to_numpy(float)
    ob_zh = obm["zone_high"].to_numpy(float)
    ob_tf = obm["ob_source_tf"].to_numpy(str)
    ob_ent_arr = np.empty(len(obm), dtype=object)
    ob_ent_arr[:] = obm["ob_enter_times"].tolist()
    o_av = np.sort(ob_av)

    state_rows, oracle_rows = [], []
    for r in cc.itertuples(index=False):
        j = int(r.contact_bar_index)
        if j >= n:
            continue
        entry = float(r.entry_reference)
        a0 = float(r.atr0)
        side = int(r.side)
        lvl = float(r.liquidity_price)
        if not (np.isfinite(a0) and a0 > 0):
            continue
        dt = pd.Timestamp(r.decision_time)
        dtn = np.datetime64(dt)

        am = active_mask(ms, dtn)
        av = ms_av <= dtn
        vp, vs, vt = mp[am], mscope[am], mtype[am]
        dp = (vp - entry) / a0
        de = side * (vp - lvl) / a0

        frow = dict(liquidity_id=r.liquidity_id,
                    contact_number=r.contact_number)
        for sc in SCOPES:
            m = vs == sc
            cnt = (np.histogram(dp[m], bins=BIN_EDGES)[0] if m.any()
                   else np.zeros(len(BIN_LABELS), int))
            for b, c in zip(BIN_LABELS, cnt):
                frow[f"{sc}_bin_{b}"] = int(c)
        up, dn = dp[dp > 0], dp[dp < 0]
        ah, bh = de[de > 0], de[de < 0]
        frow["nearest_above_R"] = round(float(up.min()), 4) if len(up) else np.nan
        frow["nearest_below_R"] = round(float(-dn.max()), 4) if len(dn) else np.nan
        frow["nearest_ahead_R"] = round(float(ah.min()), 4) if len(ah) else np.nan
        frow["nearest_behind_R"] = round(float(-bh.max()), 4) if len(bh) else np.nan
        same = np.isclose(vp, lvl, rtol=0.0, atol=1e-12)
        frow["same_price_identity_count"] = int(same.sum())
        frow["same_price_scopes"] = "|".join(sorted(set(vs[same])))
        frow["active_visible_count"] = int(am.sum())
        frow["historical_visible_count"] = int(av.sum())
        frow["same_price_identity_count_v11"] = int(
            np.isclose(mp[(ms_av <= dtn)
                          & ~((~pd.isna(ms_fp)) & (ms_fp < dtn))],
                       lvl, rtol=0.0, atol=1e-12).sum())

        k = int(np.searchsorted(o_av, dtn, side="right"))
        act = np.zeros(k, bool)
        if k:
            ina = ob_in[:k]
            act = pd.isna(ina) | (ina > dtn)
        zl, zh, bias, tf = (ob_zl[:k][act], ob_zh[:k][act],
                            ob_bias[:k][act], ob_tf[:k][act])
        gid = np.flatnonzero(act)
        ent_k = ob_ent_arr[:k][act]
        frow.update(_ob_fields(zl, zh, bias, tf, ent_k, gid, act, side, lvl,
                               a0, dtn))
        frow.update(dict(
            contact_type=r.contact_type, liquidity_scope=r.liquidity_scope,
            side=side, symbol=sym, contact_bar_index=j, decision_time=dt,
            entry_reference=entry, atr0=a0,
            penetration_depth_R=r.penetration_depth_R,
            close_relative_to_level_R=r.close_relative_to_level_R,
            trend_struct_5m=getattr(r, "trend_struct_5m", np.nan),
            trend_struct_15m=getattr(r, "trend_struct_15m", np.nan),
            trend_struct_1h=getattr(r, "trend_struct_1h", np.nan),
            env_direction_4h=getattr(r, "env_direction_4h", np.nan),
        ))
        frow["sweep_vs_5m"] = rel(side, frow["trend_struct_5m"])
        frow["sweep_vs_15m"] = rel(side, frow["trend_struct_15m"])
        frow["sweep_vs_1h"] = rel(side, frow["trend_struct_1h"])
        frow["env4h_vs_1h"] = rel(frow["env_direction_4h"],
                                  frow["trend_struct_1h"])
        frow["trend_1h_vs_15m"] = rel(frow["trend_struct_1h"],
                                      frow["trend_struct_15m"])
        frow["trend_15m_vs_5m"] = rel(frow["trend_struct_15m"],
                                      frow["trend_struct_5m"])
        state_rows.append(frow)

        di = np.flatnonzero(disc[j + 1:])
        end_i = (j + 1 + int(di[0])) if len(di) else n
        path_censored = "ROLL_CENSORED" if len(di) else "DATA_END_CENSORED"
        H, L = hi[j + 1:end_i], lo[j + 1:end_i]
        plen = len(H)

        for d, dname in ((+1, "LONG"), (-1, "SHORT")):
            sgn = d * (vp - entry)
            ok = sgn > 0
            if not ok.any():
                for risk in RISK_ATR_GRID:
                    oracle_rows.append(dict(
                        liquidity_id=r.liquidity_id,
                        contact_number=r.contact_number, symbol=sym,
                        direction=dname, risk_ATR=risk,
                        best_R_lower=np.nan, best_R_upper=np.nan,
                        best_R_is_exact=False,
                        resolution_class="NO_ACTIVE_TARGET",
                        conservative_best_R=0.0, optimistic_best_R=0.0,
                        stop_hit=False, bars_to_stop=None,
                        bars_to_best_conservative=None,
                        bars_to_best_optimistic=None,
                        bars_to_best_lower=None, bars_to_best_upper=None,
                        best_target_price=None, n_targets=0,
                        ambiguous_intrabar=False, status="NO_ACTIVE_TARGET",
                        path_censor=path_censored, path_len=plen))
                continue
            tp, ts_ = vp[ok], vs[ok]
            up_, inv = np.unique(tp, return_inverse=True)
            order = np.argsort(d * (up_ - entry))
            up_s = up_[order]
            for row in oracle_direction(H, L, entry, a0, d, up_s,
                                        path_censored):
                kc = row.get("best_cluster_index_conservative")
                if kc is not None:
                    gi = int(order[kc])
                    m = inv == gi
                    scopes = sorted(set(ts_[m]))
                    row["best_target_price"] = float(up_[gi])
                    row["best_target_cluster_size"] = int(m.sum())
                    row["best_target_scopes"] = "|".join(scopes)
                    row["best_target_scope_count"] = len(scopes)
                    row["best_target_min_scope"] = min(
                        scopes, key=lambda x: SCOPE_ORDER.index(x)
                        if x in SCOPE_ORDER else 99)
                    row["best_target_max_scope"] = max(
                        scopes, key=lambda x: SCOPE_ORDER.index(x)
                        if x in SCOPE_ORDER else -1)
                    for s_ in ("5m", "15m", "1h", "CONTIG_SESSION",
                               "TRADING_DAY", "TRADING_WEEK"):
                        row[f"best_target_has_{s_.lower()}"] = bool(
                            s_ in scopes)
                row.update(dict(liquidity_id=r.liquidity_id,
                                contact_number=r.contact_number, symbol=sym,
                                direction=dname, n_targets=int(len(up_)),
                                path_censor=path_censored, path_len=plen))
                oracle_rows.append(row)
    return pd.DataFrame(state_rows), pd.DataFrame(oracle_rows)


def main():
    master = pd.read_parquet(RESULTS / "liquidity_master_v1_1.parquet")
    contacts = pd.read_parquet(RESULTS / "liquidity_contacts_v1_1.parquet")
    S, O = [], []
    for sym in sorted(contacts["symbol"].unique()):
        t0 = time.perf_counter()
        obm = build_ob_map(sym)
        s, o = build_symbol(sym, contacts, master, obm)
        S.append(s)
        O.append(o)
        print(f"  {sym}: state={len(s)} oracle={len(o)} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    S = pd.concat(S, ignore_index=True)
    O = pd.concat(O, ignore_index=True)
    S.to_parquet(RESULTS / "liquidity_state_snapshot_v1_2.parquet",
                 index=False)
    O.to_parquet(RESULTS / "oracle_risk_frontier_v1_2.parquet", index=False)
    print(f"\nstate={len(S)} oracle={len(O)}")
    print("\nresolution_class:")
    print(O["resolution_class"].value_counts().to_string())
    print("\nstatus:")
    print(O["status"].value_counts().to_string())
    print("ORACLE_V1_2_DONE")


if __name__ == "__main__":
    main()
