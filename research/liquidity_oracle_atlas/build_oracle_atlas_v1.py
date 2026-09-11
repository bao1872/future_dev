"""SMC Oracle Atlas v1.0 —— 第二层：状态快照 / 流动性场 / 路径 Oracle。

Oracle 使用**事后动态规划**给每个接触时点计算"如果当时做多/做空，
理论上能得到什么样的风险—收益路径"。

    Oracle 标签可以使用未来；但**绝不能回到事前特征表**。

到达查询用 cummax/cummin + searchsorted，做到 O(log N)：
    LONG : reach target p  <=> cummax(high)[i] >= p
           stop hit        <=> cummin(low)[i]  <= stop
    SHORT 镜像。
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_state_machine.build_ob_confluence_v3 import build_ob_map
from research.build_pytdx_panel import aggregate_15m
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1 import (
    RESULTS,
)
from research.liquidity_state_machine.build_boundary_preoutcome_v3_2 import (
    attach_trend_multi, build_trend_series,
)
from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    SymbolData,
)
from research.liquidity_state_machine.match_liquidity_placebo_v2 import rel
from research.ob_trigger_snapshot import aggregate_1h_from_15m
from research.phase1_tradability.phase1_contract_v1 import compute_atr5

RISK_ATR_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
MAXB = 500                 # Oracle 向后扫描上限（bar）
TARGET_MAX_DIST_ATR = 20.0
TARGET_MAX_N = 400

BIN_EDGES = [-np.inf, -4, -2, -1, -0.5, 0.0, 0.5, 1, 2, 4, np.inf]
BIN_LABELS = ["(-inf,-4]", "(-4,-2]", "(-2,-1]", "(-1,-0.5]", "(-0.5,0]",
              "(0,0.5]", "(0.5,1]", "(1,2]", "(2,4]", "(4,+inf)"]
SCOPES = ["5m", "15m", "1h", "CONTIG_SESSION", "TRADING_DAY", "TRADING_WEEK"]


def _first_true(mask):
    idx = int(np.argmax(mask))
    return idx if mask[idx] else -1


def oracle_direction(H, L, entry, atr0, d, tp_sorted):
    """对单方向 + 全部 risk grid 计算风险—RR 前沿。

    tp_sorted: 已按"利润方向距离"升序排好的事前 target 价格数组。
    返回 list[dict]，每个 risk 一行。
    """
    N = len(H)
    if N == 0:
        return []
    cH = np.maximum.accumulate(H)
    cL = np.minimum.accumulate(L)
    risk_px_base = atr0
    out = []
    # target 首次到达 bar（O(log N)）
    if d == +1:
        reach = np.searchsorted(cH, tp_sorted, side="left")
    else:
        reach = np.searchsorted(-cL, -tp_sorted, side="left")
    for risk in RISK_ATR_GRID:
        rpx = risk * risk_px_base
        stop = entry - d * rpx
        if d == +1:
            sm = cL <= stop
        else:
            sm = cH >= stop
        si = _first_true(sm)
        stop_idx = si if si >= 0 else N
        stopped = bool(si >= 0)
        # 保守：target 必须严格早于 stop bar
        valid = reach < stop_idx
        # 乐观：允许同 bar 先到 target
        opt_valid = reach <= stop_idx
        row = dict(risk_ATR=risk, stop_price=float(stop),
                   stop_hit=stopped,
                   bars_to_stop=(int(si + 1) if stopped else None))
        for tag, vm in (("conservative", valid), ("optimistic", opt_valid)):
            if vm.any():
                k = int(np.flatnonzero(vm)[-1])
                bp = float(tp_sorted[k])
                row[f"{tag}_best_R"] = round(abs(bp - entry) / rpx, 4)
                row[f"{tag}_best_target_price"] = bp
                row[f"bars_to_best_{tag}"] = int(reach[k] + 1)
            else:
                row[f"{tag}_best_R"] = 0.0
                row[f"{tag}_best_target_price"] = None
                row[f"bars_to_best_{tag}"] = None
        # 同 bar 同时触发 -> 顺序不可知
        row["ambiguous_intrabar"] = bool(
            stopped and np.any(reach == stop_idx))
        if row["ambiguous_intrabar"]:
            row["status"] = "AMBIGUOUS_INTRABAR_ORDER"
        elif stopped:
            row["status"] = "STOPPED"
        else:
            row["status"] = "CENSORED_NO_STOP"
        # R 倍数首次到达（与 target 无关）
        for m in (1, 2, 3):
            pm = entry + d * m * rpx
            i_m = (int(np.searchsorted(cH, pm, side="left")) if d == +1
                   else int(np.searchsorted(-cL, -pm, side="left")))
            row[f"first_{m}R_bar"] = (i_m + 1) if i_m < N else None
        out.append(row)
    return out


def build_symbol(sym, contacts, master, obm):
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    t = pd.to_datetime(five["bar_start_time"]).to_numpy()
    hi = five["high"].to_numpy(float)
    lo = five["low"].to_numpy(float)
    op = five["open"].to_numpy(float)
    cl = five["close"].to_numpy(float)
    atr = compute_atr5(dict(open=op, high=hi, low=lo, close=cl, time=t,
                            n=len(five)))
    n = len(five)

    # liquidity universe（按 scope 分组，便于分箱）
    lv = master[master["symbol"] == sym]
    lv_price = lv["price"].to_numpy(float)
    lv_avail_i = lv["available_bar_index"].to_numpy(float)
    lv_scope = lv["liquidity_scope"].to_numpy(str)

    # ---- trend as-of（整表一次） ----
    sd = SymbolData(sym)
    cc = contacts[contacts["symbol"] == sym].copy()
    cc["interaction_time"] = pd.to_datetime(cc["decision_time"])
    cc = attach_trend_multi(cc, sd)

    # ---- OB map ----
    if obm is None or len(obm) == 0:
        obm = build_ob_map(sym)
    ob_av = pd.to_datetime(obm["ob_available_time"]).to_numpy()
    ob_in = pd.to_datetime(obm["ob_inactive_time"]).to_numpy()
    ob_bias = obm["ob_bias"].to_numpy(int)
    ob_zl = obm["zone_low"].to_numpy(float)
    ob_zh = obm["zone_high"].to_numpy(float)
    ob_tf = obm["ob_source_tf"].to_numpy(str)
    ob_ent = obm["ob_enter_times"].tolist()
    o_av = np.sort(ob_av)

    state_rows, field_rows, oracle_rows = [], [], []
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
        dt = np.datetime64(pd.Timestamp(r.decision_time))

        # ---------- 可见 liquidity ----------
        vis = lv_avail_i <= j
        vp = lv_price[vis]
        vs = lv_scope[vis]

        # ---------- liquidity field ----------
        dp = (vp - entry) / a0                       # 坐标系 A
        de = side * (vp - lvl) / a0                  # 坐标系 B
        frow = dict(liquidity_id=r.liquidity_id, contact_number=r.contact_number)
        for sc in SCOPES:
            m = vs == sc
            if not m.any():
                for b in BIN_LABELS:
                    frow[f"{sc}_bin_{b}"] = 0
                continue
            d_sc = dp[m]
            cnt = np.histogram(d_sc, bins=BIN_EDGES)[0]
            for b, c in zip(BIN_LABELS, cnt):
                frow[f"{sc}_bin_{b}"] = int(c)
        up = dp[dp > 0]
        dn = dp[dp < 0]
        frow["nearest_above_R"] = round(float(up.min()), 4) if len(up) else np.nan
        frow["nearest_below_R"] = round(float(-dn.max()), 4) if len(dn) else np.nan
        ah = de[de > 0]
        bh = de[de < 0]
        frow["nearest_ahead_R"] = round(float(ah.min()), 4) if len(ah) else np.nan
        frow["nearest_behind_R"] = round(float(-bh.max()), 4) if len(bh) else np.nan
        same = np.isclose(vp, lvl, rtol=0.0, atol=1e-12)
        frow["same_price_identity_count"] = int(same.sum())
        frow["same_price_scopes"] = "|".join(sorted(set(vs[same])))
        frow["n_visible_liquidity"] = int(vis.sum())
        field_rows.append(frow)

        # ---------- OB field ----------
        k = int(np.searchsorted(o_av, dt, side="right"))
        act = np.zeros(k, dtype=bool)
        if k:
            ina = ob_in[:k]
            act = pd.isna(ina) | (ina > dt)
        zl, zh, bias, tf = ob_zl[:k], ob_zh[:k], ob_bias[:k], ob_tf[:k]
        zl, zh, bias, tf = zl[act], zh[act], bias[act], tf[act]
        orow = dict(liquidity_id=r.liquidity_id,
                    contact_number=r.contact_number)
        for tag, want in (("opposing", -side), ("same_direction", side)):
            m = bias == want
            if not m.any():
                orow[f"nearest_{tag}_ob_distance_R"] = np.nan
                orow[f"nearest_{tag}_ob_width_R"] = np.nan
                orow[f"nearest_{tag}_ob_source_tf"] = None
                orow[f"nearest_{tag}_ob_freshness"] = None
                orow[f"nearest_{tag}_ob_prior_enter_count"] = None
                continue
            # 沿 side 方向、位于 liquidity 外侧
            if side == +1:
                ok = zl[m] > lvl
                dist = (zl[m] - lvl) / a0
            else:
                ok = zh[m] < lvl
                dist = (lvl - zh[m]) / a0
            if not ok.any():
                orow[f"nearest_{tag}_ob_distance_R"] = np.nan
                orow[f"nearest_{tag}_ob_width_R"] = np.nan
                orow[f"nearest_{tag}_ob_source_tf"] = None
                orow[f"nearest_{tag}_ob_freshness"] = None
                orow[f"nearest_{tag}_ob_prior_enter_count"] = None
                continue
            dd = dist[ok]
            bi = int(np.argmin(dd))
            gidx = np.flatnonzero(m)[ok][bi]
            orow[f"nearest_{tag}_ob_distance_R"] = round(float(dd[bi]), 4)
            orow[f"nearest_{tag}_ob_width_R"] = round(
                float(abs(zh[gidx] - zl[gidx]) / a0), 4)
            orow[f"nearest_{tag}_ob_source_tf"] = str(tf[gidx])
            n_ent = sum(1 for x in (ob_ent[gidx] or []) if x <= dt)
            orow[f"nearest_{tag}_ob_freshness"] = ("FRESH" if n_ent == 0
                                                   else "RETESTED")
            orow[f"nearest_{tag}_ob_prior_enter_count"] = n_ent
        orow.update(field_rows[-1])
        # 合并进 state 行以便后续画像
        orow["contact_type"] = r.contact_type
        orow["liquidity_scope"] = r.liquidity_scope
        orow["side"] = side
        orow["symbol"] = sym
        orow["contact_bar_index"] = j
        orow["decision_time"] = r.decision_time
        orow["trend_struct_5m"] = getattr(r, "trend_struct_5m", np.nan)
        orow["trend_struct_15m"] = getattr(r, "trend_struct_15m", np.nan)
        orow["trend_struct_1h"] = getattr(r, "trend_struct_1h", np.nan)
        orow["env_direction_4h"] = getattr(r, "env_direction_4h", np.nan)
        orow["internal_bias_5m"] = getattr(r, "internal_bias_5m", np.nan)
        orow["internal_bias_15m"] = getattr(r, "internal_bias_15m", np.nan)
        orow["internal_bias_1h"] = getattr(r, "internal_bias_1h", np.nan)
        orow["sweep_vs_5m"] = rel(side, orow["trend_struct_5m"])
        orow["sweep_vs_15m"] = rel(side, orow["trend_struct_15m"])
        orow["sweep_vs_1h"] = rel(side, orow["trend_struct_1h"])
        orow["env4h_vs_1h"] = rel(orow["env_direction_4h"],
                                  orow["trend_struct_1h"])
        orow["trend_1h_vs_15m"] = rel(orow["trend_struct_1h"],
                                      orow["trend_struct_15m"])
        orow["trend_15m_vs_5m"] = rel(orow["trend_struct_15m"],
                                      orow["trend_struct_5m"])
        orow["penetration_depth_R"] = r.penetration_depth_R
        orow["close_relative_to_level_R"] = r.close_relative_to_level_R
        state_rows.append(orow)

        # ---------- Oracle ----------
        H = hi[j + 1: j + 1 + MAXB]
        L = lo[j + 1: j + 1 + MAXB]
        for d, dname in ((+1, "LONG"), (-1, "SHORT")):
            # 事前 target：利润方向、可见、距离上限
            sgn = d * (vp - entry)
            ok = (sgn > 0) & (sgn / a0 <= TARGET_MAX_DIST_ATR)
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
                        ambiguous_intrabar=False,
                        status="NO_TARGET", n_targets=0))
                continue
            tp = vp[ok]
            order = np.argsort(d * (tp - entry))
            tp = tp[order][:TARGET_MAX_N]
            for row in oracle_direction(H, L, entry, a0, d, tp):
                row.update(dict(liquidity_id=r.liquidity_id,
                                contact_number=r.contact_number, symbol=sym,
                                direction=dname,
                                n_targets=int(len(tp))))
                oracle_rows.append(row)
    return pd.DataFrame(state_rows), pd.DataFrame(field_rows), \
        pd.DataFrame(oracle_rows)


def main():
    master = pd.read_parquet(RESULTS / "liquidity_master_v1.parquet")
    contacts = pd.read_parquet(RESULTS / "liquidity_contacts_v1.parquet")
    S, F, O = [], [], []
    for sym in sorted(contacts["symbol"].unique()):
        t0 = time.perf_counter()
        obm = build_ob_map(sym)
        s, f, o = build_symbol(sym, contacts, master, obm)
        S.append(s)
        F.append(f)
        O.append(o)
        print(f"  {sym}: state={len(s)} oracle={len(o)} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    S = pd.concat(S, ignore_index=True)
    F = pd.concat(F, ignore_index=True)
    O = pd.concat(O, ignore_index=True)
    S.to_parquet(RESULTS / "liquidity_state_snapshot.parquet", index=False)
    F.to_parquet(RESULTS / "liquidity_field_snapshot.parquet", index=False)
    O.to_parquet(RESULTS / "oracle_risk_frontier.parquet", index=False)
    print(f"\nstate={len(S)} field={len(F)} oracle={len(O)}")
    print(O.groupby("direction")["conservative_best_R"].describe()
          .round(3).to_string())
    print("ORACLE_DONE")


if __name__ == "__main__":
    main()
