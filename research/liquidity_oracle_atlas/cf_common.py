"""Continuous Frontier 共享语义 helper（P4a / P4b 共用）。

设计原则（用户 P4a-1）：
- 不修改冻结 producer build_oracle_atlas_v1_2.py（否则改变历史 artifact）。
- 把冻结的“纯函数语义”原样抽到这里，由 P4a 的连续重建与（交叉校验用的）冻结
  oracle_direction 共同调用；用逐字段 100% 一致证明语义等价。
- path_geometry() / reconstruct_oracle_grid() / build_target_reachability() 是
  P4 的“连续前沿”新基础设施；在 7 档点回放必须与冻结 oracle_direction 逐位一致。

关键证明：reconstruct_oracle_grid 用 favorable=(cummax(high)-entry)/atr0、
adverse=(entry-cummin(low))/atr0（ATR 单位）后，对 favorable 做
searchsorted(target_distance_ATR, "left") 与冻结 oracle_direction 对 cH 做
searchsorted(tp, "left") 是同一组调用（atr0>0 的线性变换不改变 searchsorted 结果）；
stop_idx 用 searchsorted(adverse, risk, "left") 等价于冻结 cL<=stop 的首位。
故 7 档回放逐位复现冻结值。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import (
    compute_atr5, discontinuity_flags,
)
from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import (
    RESULTS,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1 import (
    BIN_EDGES, BIN_LABELS, RISK_ATR_GRID, SCOPES,
)
from research.liquidity_oracle_atlas.build_oracle_atlas_v1_2 import (
    active_mask, oracle_direction,
)

SCOPE_ORDER = ["5m", "15m", "1h", "CONTIG_SESSION", "TRADING_DAY",
               "TRADING_WEEK"]


def path_geometry(high, low, entry, atr0, direction):
    """用户 P4a-2：连续路径几何，返回 (favorable, adverse)，单位 ATR。

    favorable = 沿 trade 方向的有利 excursion（>=0）；
    adverse    = 朝 stop 方向的不利 excursion（>=0）。
    """
    H = np.asarray(high, float)
    L = np.asarray(low, float)
    if direction == 1:
        favorable = (np.maximum.accumulate(H) - entry) / atr0
        adverse = (entry - np.minimum.accumulate(L)) / atr0
    else:
        favorable = (entry - np.minimum.accumulate(L)) / atr0
        adverse = (np.maximum.accumulate(H) - entry) / atr0
    return (np.maximum(favorable, 0.0), np.maximum(adverse, 0.0))


def _cluster_meta(up_s, order, k, ts_, inv):
    """复刻冻结 producer 的 best target cluster 元数据。

    k = 在 up_s（已排序唯一价）中的索引；
    order = argsort(d*(up_-entry))；inv = np.unique 的 inverse。
    返回 (best_target_price, cluster_size, scopes_str, scope_count,
          min_scope, max_scope, has_{scope} dict)。
    """
    gi = int(order[k])
    m = inv == gi
    scopes = sorted(set(ts_[m]))
    return dict(
        best_target_price=float(up_s[gi]),
        best_target_cluster_size=int(m.sum()),
        best_target_scopes="|".join(scopes),
        best_target_scope_count=len(scopes),
        best_target_min_scope=min(scopes, key=lambda x: SCOPE_ORDER.index(x)
                                  if x in SCOPE_ORDER else 99),
        best_target_max_scope=max(scopes, key=lambda x: SCOPE_ORDER.index(x)
                                  if x in SCOPE_ORDER else -1),
        **{f"best_target_has_{s_.lower()}": bool(s_ in scopes)
           for s_ in ("5m", "15m", "1h", "CONTIG_SESSION",
                      "TRADING_DAY", "TRADING_WEEK")},
    )


def reconstruct_oracle_grid(H, L, entry, a0, d, up_s, ts_, inv, order,
                            path_censored):
    """用户 P4a-6：用连续 favorable/adverse 在 7 档 RISK_ATR_GRID 回放冻结语义。

    返回 list[dict]，每档一行，含 best_R_lower/upper/is_exact/resolution_class/
    status + best target cluster 元数据。与冻结 oracle_direction 逐位等价。
    """
    N = len(H)
    fav, adv = path_geometry(H, L, entry, a0, d)
    n_t = len(up_s)
    target_dist = np.abs(up_s - entry) / a0          # ATR 单位，>0
    reach = np.searchsorted(fav, target_dist, side="left")
    reached = reach < N
    out = []
    for risk in RISK_ATR_GRID:
        rpx = risk * a0
        si = int(np.searchsorted(adv, risk, side="left"))
        stopped = bool(si < N)
        stop_idx = si if stopped else N
        valid = reached & (reach < stop_idx)
        opt = reached & (reach <= stop_idx)
        amb = bool(stopped and np.any(reach == stop_idx))
        row = dict(direction=("LONG" if d == 1 else "SHORT"),
                   risk_ATR=risk, path_censor=path_censored,
                   n_targets=int(n_t),
                   conservative_best_R=0.0, optimistic_best_R=0.0,
                   stop_hit=stopped,
                   bars_to_stop=(int(si + 1) if stopped else None),
                   ambiguous_intrabar=amb)
        if valid.any():
            k = int(np.flatnonzero(valid)[-1])
            low = round(float(abs(up_s[k] - entry)) / rpx, 4)
            row["conservative_best_R"] = low
            row["best_cluster_index_conservative"] = k
            row["bars_to_best_conservative"] = int(reach[k] + 1)
        else:
            low = 0.0
            row["bars_to_best_conservative"] = None
        if opt.any():
            kk = int(np.flatnonzero(opt)[-1])
            upR = round(float(abs(up_s[kk] - entry)) / rpx, 4)
            row["optimistic_best_R"] = upR
            row["best_cluster_index_optimistic"] = kk
        else:
            upR = 0.0
        if amb:
            row["best_R_lower"] = low
            row["best_R_upper"] = upR
            row["best_R_is_exact"] = False
            row["resolution_class"] = "AMBIGUOUS_INTERVAL"
            row["status"] = "AMBIGUOUS_INTRABAR_ORDER"
            row["bars_to_best_lower"] = (int(reach[kk] + 1)
                                         if opt.any() else None)
        elif stopped:
            row["best_R_lower"] = low
            row["best_R_upper"] = low
            row["best_R_is_exact"] = True
            row["resolution_class"] = "EXACT_RESOLVED"
            row["status"] = ("TARGET_REACHED_THEN_STOPPED" if low > 0
                             else "STOPPED")
            row["bars_to_best_lower"] = row.get("bars_to_best_conservative")
        else:
            row["best_R_lower"] = low
            row["best_R_upper"] = np.nan
            row["best_R_is_exact"] = False
            row["resolution_class"] = "CENSORED_LOWER_BOUND"
            row["status"] = ("TARGET_REACHED_CENSORED" if low > 0
                             else "CENSORED_NO_TARGET")
            row["bars_to_best_lower"] = None
        # cluster meta（保守最优 target）
        kc = row.get("best_cluster_index_conservative")
        if kc is not None:
            row.update(_cluster_meta(up_s, order, kc, ts_, inv))
        else:
            row.update(dict(best_target_price=None,
                            best_target_cluster_size=0,
                            best_target_scopes="", best_target_scope_count=0,
                            best_target_min_scope=None,
                            best_target_max_scope=None,
                            **{f"best_target_has_{s_.lower()}": False
                               for s_ in ("5m", "15m", "1h", "CONTIG_SESSION",
                                          "TRADING_DAY", "TRADING_WEEK")}))
        out.append(row)
    return out


def build_target_reachability(H, L, entry, a0, d, up_s, ts_, inv, order,
                             path_censored):
    """用户 P4a-3：每个 target 的连续临界风险（P4b 连续前沿基础）。

    保存 required_risk_before_target_ATR / required_risk_through_target_bar_ATR。
    严格不等号语义见用户 P4a-3。
    """
    N = len(H)
    fav, adv = path_geometry(H, L, entry, a0, d)
    target_dist = np.abs(up_s - entry) / a0
    reach = np.searchsorted(fav, target_dist, side="left")
    rows = []
    for k in range(len(up_s)):
        ri = int(reach[k])
        if ri >= N:                       # 观测路径内未触达
            rows.append(dict(
                target_price=float(up_s[k]),
                target_distance_ATR=round(float(target_dist[k]), 6),
                cluster_size=int((inv == order[k]).sum()),
                scopes="|".join(sorted(set(ts_[inv == order[k]]))),
                reached=False,
                required_risk_before_target_ATR=np.nan,
                required_risk_through_target_bar_ATR=np.nan,
                censor=path_censored))
            continue
        req_before = float(adv[ri - 1]) if ri > 0 else 0.0
        req_through = float(adv[ri])
        rows.append(dict(
            target_price=float(up_s[k]),
            target_distance_ATR=round(float(target_dist[k]), 6),
            cluster_size=int((inv == order[k]).sum()),
            scopes="|".join(sorted(set(ts_[inv == order[k]]))),
            reached=True,
            required_risk_before_target_ATR=round(req_before, 6),
            required_risk_through_target_bar_ATR=round(req_through, 6),
            censor=path_censored))
    return rows


def replay_symbol(sym, contacts, master):
    """对单个品种复现 oracle frontier（连续重建）+ target reachability。

    返回 (oracle_df, target_reach_df)。oracle_df 行 = contact×direction×risk_ATR，
    字段与冻结 oracle_risk_frontier_v1_2 的核心列对齐。
    """
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
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

    oracle_rows, target_rows = [], []
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
        vp, vs = mp[am], mscope[am]
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
                        ambiguous_intrabar=False,
                        status="NO_ACTIVE_TARGET",
                        bars_to_best_conservative=None,
                        bars_to_best_lower=None,
                        path_censor=path_censored, path_len=plen,
                        n_targets=0, best_target_price=None,
                        best_target_cluster_size=0,
                        best_target_scopes="",
                        best_target_scope_count=0,
                        best_target_min_scope=None,
                        best_target_max_scope=None,
                        **{f"best_target_has_{s_.lower()}": False
                           for s_ in ("5m", "15m", "1h", "CONTIG_SESSION",
                                      "TRADING_DAY", "TRADING_WEEK")}))
                continue
            tp, ts_ = vp[ok], vs[ok]
            up_, inv = np.unique(tp, return_inverse=True)
            order = np.argsort(d * (up_ - entry))
            up_s = up_[order]
            for row in reconstruct_oracle_grid(
                    H, L, entry, a0, d, up_s, ts_, inv, order,
                    path_censored):
                row.update(dict(liquidity_id=r.liquidity_id,
                                contact_number=r.contact_number, symbol=sym,
                                path_len=plen))
                oracle_rows.append(row)
            for trow in build_target_reachability(
                    H, L, entry, a0, d, up_s, ts_, inv, order,
                    path_censored):
                trow.update(dict(liquidity_id=r.liquidity_id,
                                 contact_number=r.contact_number, symbol=sym,
                                 direction=dname))
                target_rows.append(trow)
    return pd.DataFrame(oracle_rows), pd.DataFrame(target_rows)


def classify_rr_direction(O):
    """复刻冻结 profile_oracle_atlas_v1_2 的 np.select 组合逻辑。"""
    idx = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
    cols = ["best_R_lower", "best_R_upper", "resolution_class",
            "bars_to_best_lower"]
    Lg = O[O["direction"] == "LONG"].set_index(idx)[cols]
    Sg = O[O["direction"] == "SHORT"].set_index(idx)[cols]
    B = Lg.join(Sg, lsuffix="_l", rsuffix="_s", how="inner").reset_index()
    ll, lu = B["best_R_lower_l"], B["best_R_upper_l"]
    sl, su = B["best_R_lower_s"], B["best_R_upper_s"]
    notg = ((B["resolution_class_l"] == "NO_ACTIVE_TARGET")
            | (B["resolution_class_s"] == "NO_ACTIVE_TARGET"))
    unres = (~notg) & (lu.isna() | su.isna())
    cmp_ok = (~notg) & (~unres)
    B["rr_direction"] = np.select(
        [notg, unres, cmp_ok & (ll > su), cmp_ok & (sl > lu)],
        ["NO_COMPARABLE_TARGET", "UNRESOLVED_CENSOR",
         "LONG_DOMINATES", "SHORT_DOMINATES"],
        default="TRADEOFF_OR_OVERLAP")
    RD = B[idx + ["rr_direction", "best_R_lower_l", "best_R_upper_l",
                  "best_R_lower_s", "best_R_upper_s"]].rename(
        columns={"best_R_lower_l": "long_R_lower",
                 "best_R_upper_l": "long_R_upper",
                 "best_R_lower_s": "short_R_lower",
                 "best_R_upper_s": "short_R_upper"})
    return RD


def replay_contact(sym, r, ms, mp, mscope, hi, lo, disc, n):
    """单 contact：返回 oracle_rows + 紧凑 per-target 表（含 req_before/req_through）。

    紧凑 target 字段（用户 P4c-1，reached 与 unreached 都保存）：
      reached, dist, price, cluster_size, scopes,
      required_before_ATR, required_through_ATR
    - required_through = adv[reach_i]      （保守：risk>它⇒target 先于 stop）
    - required_before  = adv[reach_i-1]    （乐观：risk>它⇒target bar 前未 stop）
    二者独立；连续 evaluator 必须分别使用。
    另返回 long_max_adverse / short_max_adverse（= adv[-1]，ATR 单位）。
    """
    j = int(r.contact_bar_index)
    if j >= n:
        return None
    entry = float(r.entry_reference)
    a0 = float(r.atr0)
    if not (np.isfinite(a0) and a0 > 0):
        return None
    dt = pd.Timestamp(r.decision_time)
    dtn = np.datetime64(dt)
    am = active_mask(ms, dtn)
    vp, vs = mp[am], mscope[am]
    di = np.flatnonzero(disc[j + 1:])
    end_i = (j + 1 + int(di[0])) if len(di) else n
    path_censored = "ROLL_CENSORED" if len(di) else "DATA_END_CENSORED"
    H, L = hi[j + 1:end_i], lo[j + 1:end_i]
    plen = len(H)
    oracle_rows, long_targets, short_targets = [], [], []
    long_max_adverse = short_max_adverse = 0.0
    for d, dname in ((+1, "LONG"), (-1, "SHORT")):
        sgn = d * (vp - entry)
        ok = sgn > 0
        if not ok.any():
            for risk in RISK_ATR_GRID:
                oracle_rows.append(dict(
                    liquidity_id=r.liquidity_id, contact_number=r.contact_number,
                    symbol=sym, direction=dname, risk_ATR=risk,
                    best_R_lower=np.nan, best_R_upper=np.nan,
                    best_R_is_exact=False, resolution_class="NO_ACTIVE_TARGET",
                    conservative_best_R=0.0, optimistic_best_R=0.0,
                    stop_hit=False, bars_to_stop=None, ambiguous_intrabar=False,
                    status="NO_ACTIVE_TARGET", path_censor=path_censored,
                    path_len=plen, n_targets=0, best_target_price=None,
                    best_target_cluster_size=0, best_target_scopes="",
                    best_target_scope_count=0, best_target_min_scope=None,
                    best_target_max_scope=None,
                    **{f"best_target_has_{s_.lower()}": False
                       for s_ in ("5m", "15m", "1h", "CONTIG_SESSION",
                                  "TRADING_DAY", "TRADING_WEEK")}))
            continue
        tp, ts_ = vp[ok], vs[ok]
        up_, inv = np.unique(tp, return_inverse=True)
        order = np.argsort(d * (up_ - entry))
        up_s = up_[order]
        for row in reconstruct_oracle_grid(
                H, L, entry, a0, d, up_s, ts_, inv, order, path_censored):
            row.update(dict(liquidity_id=r.liquidity_id,
                            contact_number=r.contact_number, symbol=sym,
                            path_len=plen))
            oracle_rows.append(row)
        # 紧凑 target 表（P4c：两个临界值都保存，reached/unreached 都保留）
        fav, adv = path_geometry(H, L, entry, a0, d)
        max_adverse = float(adv[-1]) if len(adv) else 0.0
        if d == 1:
            long_max_adverse = max_adverse
        else:
            short_max_adverse = max_adverse
        target_dist = np.abs(up_s - entry) / a0
        reach = np.searchsorted(fav, target_dist, side="left")
        reached = reach < len(H)
        req_before = np.where(reached & (reach > 0),
                              adv[np.maximum(reach - 1, 0)], 0.0)
        req_through = np.full(len(up_s), np.nan)
        req_through[reached] = adv[reach[reached]]
        tlist = long_targets if d == 1 else short_targets
        for k in range(len(up_s)):
            gi = int(order[k])
            m = inv == gi
            scopes = sorted(set(ts_[m]))
            if reached[k]:
                tlist.append(dict(
                    reached=True, dist=float(target_dist[k]),
                    price=float(up_s[k]), cluster_size=int(m.sum()),
                    scopes="|".join(scopes),
                    required_before_ATR=float(req_before[k]),
                    required_through_ATR=float(req_through[k])))
            else:
                # 有 active target 但没达到 != NO_ACTIVE；保留以备完整语义
                tlist.append(dict(
                    reached=False, dist=float(target_dist[k]),
                    price=float(up_s[k]), cluster_size=int(m.sum()),
                    scopes="|".join(scopes),
                    required_before_ATR=np.nan,
                    required_through_ATR=np.nan))
    return dict(symbol=sym, liquidity_id=r.liquidity_id,
                contact_number=r.contact_number, path_censor=path_censored,
                long_max_adverse=long_max_adverse,
                short_max_adverse=short_max_adverse,
                oracle_rows=oracle_rows,
                long_targets=long_targets, short_targets=short_targets)


def replay_symbol_continuous(sym, contacts, master):
    """全品种连续前沿 pilot 入口：返回 (oracle_df, contact_targets_list)。"""
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
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
    mscope = ms["liquidity_scope"].astype(str).to_numpy()
    cc = contacts[contacts["symbol"] == sym].copy()
    cc["interaction_time"] = pd.to_datetime(cc["decision_time"])
    oracle_parts, cparts = [], []
    for r in cc.itertuples(index=False):
        rec = replay_contact(sym, r, ms, mp, mscope, hi, lo, disc, n)
        if rec is None:
            continue
        oracle_parts.append(pd.DataFrame(rec["oracle_rows"]))
        cparts.append({k: rec[k] for k in
                       ("symbol", "liquidity_id", "contact_number",
                        "path_censor", "long_max_adverse", "short_max_adverse",
                        "long_targets", "short_targets")})
    O = pd.concat(oracle_parts, ignore_index=True) if oracle_parts else \
        pd.DataFrame()
    return O, cparts


# ---------------------------------------------------------------------------
# P4c 连续 evaluator（用户 P4c-2/3/4）：逐位复刻冻结 v1.2 uncertainty/censor 语义
# ---------------------------------------------------------------------------
def prep_side(targets, max_adverse):
    """预计算单方向阈值数组，供 bounds_at 快速查询。

    - 若有 active target（targets 非空）则 has_active=True，即使全 unreached；
      仅当 targets 为空（无 active liquidity）才 NO_ACTIVE_TARGET。
    - thr/bef 仅来自 reached target 的 required_through / required_before。
    """
    if not targets:
        return dict(has_active=False, max_adverse=float(max_adverse))
    reached = [t for t in targets if t["reached"]]
    thr = sorted((float(t["required_through_ATR"]), float(t["dist"]))
                 for t in reached
                 if np.isfinite(t.get("required_through_ATR")))
    bef = sorted((float(t["required_before_ATR"]), float(t["dist"]))
                 for t in reached
                 if np.isfinite(t.get("required_before_ATR")))

    def pm(arr):
        out, c = [], 0.0
        for _, d in arr:
            c = max(c, d)
            out.append(c)
        return out

    return dict(has_active=True, max_adverse=float(max_adverse),
                thr=[x[0] for x in thr], thr_pm=pm(thr),
                bef=[x[0] for x in bef], bef_pm=pm(bef))


def bounds_at(prep, r):
    """单方向在 risk=r 的 (state, lower, upper)，复刻冻结 oracle_direction。

    - r > max_adverse ⇒ stop 未在观测路径内发生 ⇒ CENSORED_LOWER_BOUND，
      lower = 已达最大 target 距离，upper = NaN。
    - 否则 lower = max{ dist : required_through < r }（保守），
      upper = max{ dist : required_before < r }（乐观）。
    因 adv 单调非减，required_through<r ⇔ reach<stop_idx，
    required_before<r ⇔ reach<=stop_idx，与冻结一致。
    """
    import bisect
    if not prep["has_active"]:
        return dict(state="NO_ACTIVE_TARGET", lower=np.nan, upper=np.nan)
    if r > prep["max_adverse"]:
        low = prep["thr_pm"][-1] if prep["thr_pm"] else 0.0
        return dict(state="CENSORED_LOWER_BOUND", lower=low, upper=np.nan)
    ilo = bisect.bisect_left(prep["thr"], r)
    lo = prep["thr_pm"][ilo - 1] if ilo > 0 else 0.0
    ihi = bisect.bisect_left(prep["bef"], r)
    up = prep["bef_pm"][ihi - 1] if ihi > 0 else 0.0
    state = "EXACT_RESOLVED" if lo == up else "AMBIGUOUS_INTERVAL"
    return dict(state=state, lower=lo, upper=up)


def continuous_pair_direction(bl, bs, r):
    """复刻冻结 np.select（用户 P4c-4）。比较用 4 位 RR rounding 以匹配冻结。"""
    if (bl["state"] == "NO_ACTIVE_TARGET"
            or bs["state"] == "NO_ACTIVE_TARGET"):
        return "NO_COMPARABLE_TARGET"
    ll = round(bl["lower"] / r, 4) if np.isfinite(bl["lower"]) else np.nan
    lu = round(bl["upper"] / r, 4) if np.isfinite(bl["upper"]) else np.nan
    sl = round(bs["lower"] / r, 4) if np.isfinite(bs["lower"]) else np.nan
    su = round(bs["upper"] / r, 4) if np.isfinite(bs["upper"]) else np.nan
    if np.isnan(lu) or np.isnan(su):
        return "UNRESOLVED_CENSOR"
    if ll > su:
        return "LONG_DOMINATES"
    if sl > lu:
        return "SHORT_DOMINATES"
    return "TRADEOFF_OR_OVERLAP"


def _transition_type(f, t):
    if {f, t} == {"LONG_DOMINATES", "SHORT_DOMINATES"}:
        return "DIRECT"
    if "UNRESOLVED_CENSOR" in (f, t):
        return "CENSOR_MEDIATED"
    if ("TRADEOFF_OR_OVERLAP" in (f, t)
            or "NO_COMPARABLE_TARGET" in (f, t)):
        return "OVERLAP_MEDIATED"
    return "OTHER"


def continuous_frontier_analyze(long_targets, short_targets,
                                long_max_adverse, short_max_adverse,
                                path_censor, risk_max=None):
    """用户 P4c-5..14：连续方向前沿 + transition 分类（5 态序列）。

    返回 dict：seq（开区间 5 态）、transitions（每对相邻态变化）、
    contact_class、overlap_bands、各标志位。
    """
    if risk_max is None:
        risk_max = float(max(RISK_ATR_GRID))
    lp = prep_side(long_targets, long_max_adverse)
    sp = prep_side(short_targets, short_max_adverse)

    def side_at(side_prep, r):
        return bounds_at(side_prep, r)

    # breakpoint = 所有 required_before/through + max_adverse，限 (0, risk_max)
    bps = []
    for t in long_targets + short_targets:
        for key in ("required_before_ATR", "required_through_ATR"):
            v = t.get(key)
            if np.isfinite(v) and 0.0 < v < risk_max:
                bps.append(v)
    for mv in (long_max_adverse, short_max_adverse):
        if 0.0 < mv < risk_max:
            bps.append(mv)
    bps = sorted(set(round(x, 9) for x in bps))
    edges = [0.0] + bps + [risk_max]

    seq = []
    for i in range(len(edges) - 1):
        a, b = edges[i], edges[i + 1]
        r = (a + b) / 2.0
        bl = side_at(lp, r)
        bs = side_at(sp, r)
        st = continuous_pair_direction(bl, bs, r)
        seq.append(dict(a=a, b=b, state=st,
                        ll=bl["lower"], lu=bl["upper"],
                        sl=bs["lower"], su=bs["upper"]))

    transitions = []
    n = len(seq)
    for i in range(n - 1):
        s1, s2 = seq[i]["state"], seq[i + 1]["state"]
        if s1 == s2:
            continue
        bp = seq[i + 1]["a"]
        unl = []
        for side, tlist in (("LONG", long_targets), ("SHORT", short_targets)):
            for t in tlist:
                for key, tt in (("required_before_ATR", "before"),
                                ("required_through_ATR", "through")):
                    v = t.get(key)
                    if np.isfinite(v) and abs(v - bp) < 1e-9:
                        unl.append(dict(side=side, price=t["price"],
                                        dist=t["dist"],
                                        cluster_size=t["cluster_size"],
                                        scopes=t["scopes"],
                                        threshold_type=tt,
                                        threshold_ATR=v))
        transitions.append(dict(from_state=s1, to_state=s2,
                               critical_risk_ATR=bp,
                               transition_type=_transition_type(s1, s2),
                               unlock_targets=unl))

    # overlap band：TRADEOFF 连续区间，且左右邻均为 dominance
    overlap_bands = []
    i = 0
    while i < n:
        if seq[i]["state"] == "TRADEOFF_OR_OVERLAP":
            j = i
            while j < n and seq[j]["state"] == "TRADEOFF_OR_OVERLAP":
                j += 1
            start, end = seq[i]["a"], seq[j - 1]["b"]
            left = seq[i - 1]["state"] if i > 0 else None
            right = seq[j]["state"] if j < n else None
            if left in ("LONG_DOMINATES", "SHORT_DOMINATES") and \
               right in ("LONG_DOMINATES", "SHORT_DOMINATES"):
                overlap_bands.append(dict(start=start, end=end,
                                          width=end - start,
                                          center=(start + end) / 2.0,
                                          left=left, right=right))
            i = j
        else:
            i += 1

    states = [s["state"] for s in seq]
    has_long = "LONG_DOMINATES" in states
    has_short = "SHORT_DOMINATES" in states
    has_trade = "TRADEOFF_OR_OVERLAP" in states
    has_cens = "UNRESOLVED_CENSOR" in states
    direct = any(tr["transition_type"] == "DIRECT" for tr in transitions)
    if not has_long and not has_short:
        contact_class = "NO_CONTINUOUS_DIRECTION_CHANGE"
    elif has_long and has_short:
        if direct:
            contact_class = "DIRECT_CONTINUOUS_SWITCH"
        elif has_cens and not has_trade:
            contact_class = "CENSOR_MEDIATED_UNRESOLVED"
        else:
            contact_class = "OVERLAP_MEDIATED_TRANSITION"
    else:
        contact_class = "SINGLE_SIDE_DOMINANCE"

    return dict(seq=seq, transitions=transitions,
                contact_class=contact_class, overlap_bands=overlap_bands,
                has_long=has_long, has_short=has_short,
                has_tradeoff=has_trade, has_censor=has_cens,
                direct=direct, n_transitions=len(transitions))
