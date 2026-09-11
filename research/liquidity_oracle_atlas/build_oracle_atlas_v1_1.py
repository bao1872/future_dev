"""SMC Oracle Atlas v1.1 —— 第二层（修复版）：状态/活性场/路径 Oracle。

相对 v1.0 的修复：
  1. liquidity field 与 Oracle target 只使用 **decision_time 仍 active**
     的 liquidity（已 consumed 的历史 level 不再进入）
  2. Oracle 路径终点 = min(下一个 discontinuity, data end)，
     **取消 500 bar 主 horizon**，并审计 500 截断影响
  3. target 不再是裸价格数组，而是**价格簇**，保留 identity/scope/type
  4. first_1R/2R/3R 必须相对于 stop first passage 判定
  5. 输出显式 censor status（STOPPED / ROLL / DATA_END / AMBIGUOUS …）
  6. 审计 20ATR / 400 target 截断影响（不作为 primary 限制）
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
from research.liquidity_oracle_atlas.build_oracle_atlas_v1 import (
    BIN_EDGES, BIN_LABELS, RISK_ATR_GRID, SCOPES,
)

AUDIT_MAXB = 500            # 仅用于审计 v1.0 的截断影响
AUDIT_DIST_ATR = 20.0       # 仅用于审计
AUDIT_MAX_N = 400           # 仅用于审计


def active_mask(master_sym, decision_time):
    """active = 已可用 AND 尚未在 decision_time 之前被消费。"""
    av = pd.to_datetime(master_sym["available_time"]).to_numpy()
    fp = pd.to_datetime(master_sym["first_penetration_time"]).to_numpy()
    ok = av <= decision_time
    consumed_before = (~pd.isna(fp)) & (fp < decision_time)
    return ok & (~consumed_before)


def oracle_direction(H, L, entry, atr0, d, tp_sorted, path_censored):
    """tp_sorted: 按利润方向距离升序的 target 价格簇。"""
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
        row = dict(risk_ATR=risk, stop_price=float(stop), stop_hit=stopped,
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
        if amb:
            row["status"] = "AMBIGUOUS_INTRABAR_ORDER"
        elif stopped:
            row["status"] = ("TARGET_REACHED_THEN_STOPPED"
                             if row["conservative_best_R"] > 0 else "STOPPED")
        else:
            row["status"] = ("TARGET_REACHED_CENSORED"
                             if row["conservative_best_R"] > 0
                             else "CENSORED_NO_TARGET")
            row["censor_reason"] = path_censored
        # first_mR 必须相对 stop first passage
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

    state_rows, oracle_rows, audit_rows = [], [], []
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

        # ---------- active liquidity ----------
        av = pd.to_datetime(ms["available_time"]).to_numpy() <= dt
        am = active_mask(ms, np.datetime64(dt))
        vp, vs, vt, vid = mp[am], mscope[am], mtype[am], mid[am]
        vph = mp[av]                       # 历史可见（v1.0 口径，仅审计）
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
        # v1.0 口径（历史可见）的 same-price，用于审计差异
        frow["same_price_identity_count_v10"] = int(
            np.isclose(vph, lvl, rtol=0.0, atol=1e-12).sum())

        # ---------- OB field ----------
        k = int(np.searchsorted(o_av, np.datetime64(dt), side="right"))
        act = np.zeros(k, bool)
        if k:
            ina = ob_in[:k]
            act = pd.isna(ina) | (ina > np.datetime64(dt))
        zl, zh, bias, tf = (ob_zl[:k][act], ob_zh[:k][act],
                            ob_bias[:k][act], ob_tf[:k][act])
        gid = np.flatnonzero(act)
        ent_k = ob_ent_arr[:k][act]
        frow.update(_ob_fields(zl, zh, bias, tf, ent_k, gid, act, side, lvl,
                               a0, np.datetime64(dt)))
        frow.update(dict(
            contact_type=r.contact_type, liquidity_scope=r.liquidity_scope,
            side=side, symbol=sym, contact_bar_index=j,
            decision_time=dt, entry_reference=entry, atr0=a0,
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

        # ---------- Oracle 路径（到 roll 或 data end） ----------
        di = np.flatnonzero(disc[j + 1:])
        end_i = (j + 1 + int(di[0])) if len(di) else n
        path_censored = "ROLL_CENSORED" if len(di) else "DATA_END_CENSORED"
        H, L = hi[j + 1:end_i], lo[j + 1:end_i]
        plen = len(H)

        # 当前 liquidity 若本次已穿透，不得再作为 target
        cur_pen = bool(r.is_penetration)
        for d, dname in ((+1, "LONG"), (-1, "SHORT")):
            sgn = d * (vp - entry)
            ok = sgn > 0
            if cur_pen:
                ok &= (vid != r.liquidity_id)
            if not ok.any():
                for risk in RISK_ATR_GRID:
                    oracle_rows.append(dict(
                        liquidity_id=r.liquidity_id,
                        contact_number=r.contact_number, symbol=sym,
                        direction=dname, risk_ATR=risk,
                        conservative_best_R=0.0, optimistic_best_R=0.0,
                        stop_hit=False, bars_to_stop=None,
                        bars_to_best_conservative=None,
                        bars_to_best_optimistic=None,
                        best_target_price=None, n_targets=0,
                        ambiguous_intrabar=False, status="NO_ACTIVE_TARGET",
                        path_censor=path_censored, path_len=plen))
                continue
            tp, ts_, tt_, ti_ = vp[ok], vs[ok], vt[ok], vid[ok]
            # 价格簇：同价合并
            up_, inv, cnt = np.unique(tp, return_inverse=True,
                                      return_counts=True)
            order = np.argsort(d * (up_ - entry))
            up_s = up_[order]
            n_clip_d = int((np.abs(up_ - entry) / a0 > AUDIT_DIST_ATR).sum())
            n_clip_n = max(0, len(up_) - AUDIT_MAX_N)
            for row in oracle_direction(H, L, entry, a0, d, up_s,
                                        path_censored):
                kc = row.get("best_cluster_index_conservative")
                if kc is not None:
                    gi = int(order[kc])
                    m = inv == gi
                    row["best_target_price"] = float(up_[gi])
                    row["best_target_cluster_size"] = int(cnt[gi])
                    row["best_target_scopes"] = "|".join(sorted(set(ts_[m])))
                    row["best_target_types"] = "|".join(sorted(set(tt_[m])))
                    row["best_target_has_1h"] = bool("1h" in set(ts_[m]))
                    row["best_target_has_day"] = bool(
                        any(x in ("TRADING_DAY", "TRADING_WEEK",
                                  "CONTIG_SESSION") for x in set(ts_[m])))
                row.update(dict(liquidity_id=r.liquidity_id,
                                contact_number=r.contact_number, symbol=sym,
                                direction=dname, n_targets=int(len(up_)),
                                n_target_clipped_dist=n_clip_d,
                                n_target_clipped_count=n_clip_n,
                                path_censor=path_censored, path_len=plen))
                oracle_rows.append(row)
        # 500 bar 截断审计
        audit_rows.append(dict(
            liquidity_id=r.liquidity_id, contact_number=r.contact_number,
            symbol=sym, path_len=plen, path_censor=path_censored,
            active_visible=int(am.sum()), historical_visible=int(av.sum())))
    return (pd.DataFrame(state_rows), pd.DataFrame(oracle_rows),
            pd.DataFrame(audit_rows))


def _ob_fields(zl, zh, bias, tf, ent_k, gid, act, side, lvl, a0, dt):
    out = {}
    for tag, want in (("opposing", -side), ("same_direction", side)):
        m = bias == want
        keys = [f"nearest_{tag}_ob_distance_R", f"nearest_{tag}_ob_width_R",
                f"nearest_{tag}_ob_source_tf", f"nearest_{tag}_ob_freshness",
                f"nearest_{tag}_ob_prior_enter_count"]
        if not m.any():
            out.update({k: (np.nan if "R" in k else None) for k in keys})
            continue
        if side == +1:
            ok = zl[m] > lvl
            dist = (zl[m] - lvl) / a0
        else:
            ok = zh[m] < lvl
            dist = (lvl - zh[m]) / a0
        if not ok.any():
            out.update({k: (np.nan if "R" in k else None) for k in keys})
            continue
        dd = dist[ok]
        bi = int(np.argmin(dd))
        # 索引必须在"已按 act 过滤"的空间内，gid 仅用于映射回原始 enter_times
        g = int(np.flatnonzero(m)[ok][bi])
        raw = int(gid[g])
        n_ent = sum(1 for x in (ent_k[g] or []) if x <= dt)
        out[keys[0]] = round(float(dd[bi]), 4)
        out[keys[1]] = round(float(abs(zh[g] - zl[g]) / a0), 4)
        out[keys[2]] = str(tf[g])
        out[keys[3]] = "FRESH" if n_ent == 0 else "RETESTED"
        out[keys[4]] = n_ent
    return out


def main():
    master = pd.read_parquet(RESULTS / "liquidity_master_v1_1.parquet")
    contacts = pd.read_parquet(RESULTS / "liquidity_contacts_v1_1.parquet")
    S, O, A = [], [], []
    for sym in sorted(contacts["symbol"].unique()):
        t0 = time.perf_counter()
        obm = build_ob_map(sym)
        s, o, a = build_symbol(sym, contacts, master, obm)
        S.append(s)
        O.append(o)
        A.append(a)
        print(f"  {sym}: state={len(s)} oracle={len(o)} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    S, O, A = (pd.concat(S, ignore_index=True), pd.concat(O, ignore_index=True),
               pd.concat(A, ignore_index=True))
    S.to_parquet(RESULTS / "liquidity_state_snapshot_v1_1.parquet",
                 index=False)
    S.to_parquet(RESULTS / "liquidity_field_snapshot_v1_1.parquet",
                 index=False)
    O.to_parquet(RESULTS / "oracle_risk_frontier_v1_1.parquet", index=False)
    A.to_parquet(RESULTS / "oracle_path_audit_v1_1.parquet", index=False)
    print(f"\nstate={len(S)} oracle={len(O)}")
    print("\nstatus:")
    print(O["status"].value_counts().to_string())
    print("\npath_censor:")
    print(O["path_censor"].value_counts().to_string())
    print("ORACLE_V1_1_DONE")


if __name__ == "__main__":
    main()
