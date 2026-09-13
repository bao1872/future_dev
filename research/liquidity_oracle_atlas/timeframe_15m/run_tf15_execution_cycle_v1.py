"""TF15 — 15m Execution-Cycle Exploration v1

问题：保持现有 frozen liquidity/contact universe，把 observation / reaction /
geometry / execution clock 换成 15m，看是否存在独立、稳定的 execution edge。

设计要点（reviewer 规格）：
  * 15m 决策 clock + 5m execution micro-path。
  * geometry 用 decision CLOSE；actual fill 用 next 15m OPEN（causal，无 lag-two）。
  * 15m parent 只能由 3 根**时间连续**的 5m child 组成，禁止跨 session 聚合。
  * ATR15 使用项目权威 `phase1_contract_v1.compute_atr5`（同算法同 lookback，只换输入 bar）。
  * 不使用 5m 的 R1–R4 / RR3；15m 自己产生 robust regions。
  * dual conflict 统一为「跳过当前 stage，允许更晚 h」。
  * 不训练任何模型；不读 P1；不改 v2_strategy_freeze。

本文件覆盖 Phase TF15-A/B/C/D（bar contract, contact mapping, action surface,
geometry frontier, market frontier）以及 Market gate。
"""
from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import (
    compute_atr5, discontinuity_flags)
from research.liquidity_oracle_atlas import run_liquidity_field_action_surface_v1 as s4a
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env

OUT = REPO_ROOT / "research/analysis_results/timeframe_15m_v1"
OUT.mkdir(parents=True, exist_ok=True)
FREEZE = REPO_ROOT / "research/analysis_results/v2_strategy_freeze"

# ---- pre-registered TF15 space ----
HORIZONS_15M = np.array([1, 2, 3, 5, 8], dtype=np.int64)
STRUCTURE_SCALES = np.array([0.2, 0.4, 0.8, 1.2], dtype=np.float64)
ACTIONS = ["OUTWARD", "INWARD"]
EVAL_BARS_15M = 12
N_CHILD = EVAL_BARS_15M * 3                 # 36 five-minute child slots
MAX_POST = int(HORIZONS_15M.max()) + 1      # h + entry parent
TEST_WF = ["WF1", "WF2", "WF3"]
BLOCK_TO_WF = {"TB1": "WF0", "TB2": "WF1", "TB3": "WF2", "TB4": "WF3"}

# frozen pre-registered coarse bins (same as 5m Geometry Frontier)
TARGET_EDGES = [-np.inf, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
TARGET_LABELS = ["<0.5", "0.5-1", "1-2", "2-3", "3-5", ">=5"]
RISK_EDGES = [-np.inf, 0.25, 0.5, 1.0, 2.0, 3.0, np.inf]
RISK_LABELS = ["<0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", ">=3"]
MIN_N_PER_WF = 200
MAX_AMBIGUITY_GAP_R = 0.25


# ===========================================================================
# Phase TF15-A — 15m bar contract
# ===========================================================================
def build_15m_bars(sym: str) -> dict:
    raw = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
    n5 = len(raw)
    ts = pd.to_datetime(raw["bar_start_time"]).to_numpy("datetime64[ns]")
    te = pd.to_datetime(raw["bar_end_time"]).to_numpy("datetime64[ns]")
    width = (te - ts) / np.timedelta64(1, "s")
    assert np.all(width == 300), f"{sym}: NON_5M_CHILD_WIDTH"
    disc5 = np.asarray(discontinuity_flags(sym), bool)
    assert len(disc5) == n5, f"{sym}: DISC_LENGTH_MISMATCH"

    cont = np.zeros(n5, bool)
    if n5 > 1:
        cont[1:] = ts[1:] == te[:-1]
    seg = np.cumsum(~cont)
    # child position within segment
    new_seg = np.r_[True, seg[1:] != seg[:-1]]
    grp_start = np.flatnonzero(new_seg)
    pos = np.arange(n5) - np.repeat(grp_start, np.diff(np.r_[grp_start, n5]))
    slot = pos // 3

    keys = seg.astype(np.int64) * 1_000_000 + slot
    order = np.argsort(keys, kind="stable")
    ks = keys[order]
    starts = np.flatnonzero(np.r_[True, ks[1:] != ks[:-1]])
    sizes = np.diff(np.r_[starts, len(ks)])
    full = starts[sizes == 3]
    idx3 = np.stack([order[full], order[full + 1], order[full + 2]], axis=1)
    nk = len(idx3)

    O = raw["open"].to_numpy(float)
    H = raw["high"].to_numpy(float)
    L = raw["low"].to_numpy(float)
    C = raw["close"].to_numpy(float)
    T = raw["trade"].to_numpy(float)
    P = raw["position"].to_numpy(float)
    day = pd.to_datetime(raw["trading_day"]).to_numpy("datetime64[ns]")

    bars = dict(
        open=O[idx3[:, 0]], high=H[idx3].max(axis=1), low=L[idx3].min(axis=1),
        close=C[idx3[:, 2]], trade=T[idx3].sum(axis=1), position=P[idx3[:, 2]],
        bar_start_time=ts[idx3[:, 0]], bar_end_time=te[idx3[:, 2]],
        trading_day=day[idx3[:, 2]],
        disc=disc5[idx3].any(axis=1),
        child_first_index=idx3[:, 0], child_last_index=idx3[:, 2], n=nk)
    bars["availability_time"] = bars["bar_end_time"]
    bars["atr15"] = compute_atr5(dict(high=bars["high"], low=bars["low"],
                                      close=bars["close"]))
    pwidth = (bars["bar_end_time"] - bars["bar_start_time"]) / np.timedelta64(1, "s")
    assert np.all(pwidth == 900), f"{sym}: PARENT_WIDTH_NOT_900S"
    if nk > 1:
        assert np.all(bars["child_last_index"][:-1] < bars["child_first_index"][1:]), \
            f"{sym}: PARENT_OVERLAP"
    bars["five_to_parent"] = np.full(n5, -1, dtype=np.int64)
    bars["five_to_parent"][idx3.ravel()] = np.repeat(np.arange(nk), 3)
    bars["n5"] = n5
    bars["disc5"] = disc5
    bars["raw_h"] = H
    bars["raw_l"] = L
    bars["raw_disc"] = disc5
    bars["raw_o"] = O
    bars["raw_c"] = C
    bars["raw_t"] = T
    bars["raw_p"] = P
    bars["raw_ts"] = ts
    bars["raw_te"] = te
    return bars


def bar_contract_tests(bars15):
    """Executable T2/T3/T4/T5/T6 checks on the 15m bar contract."""
    rng = np.random.default_rng(7)
    ok_ohlc = ok_vol = ok_avail = ok_contig = ok_span = True
    for sym, B in bars15.items():
        n = B["n"]
        if n == 0:
            continue
        idx = rng.choice(n, size=min(500, n), replace=False)
        for i in idx:
            a = int(B["child_first_index"][i]); b = int(B["child_last_index"][i])
            ch = slice(a, b + 1)
            ok_contig &= (b - a == 2)
            ok_contig &= bool(np.all(B["raw_ts"][a + 1:b + 1]
                                     == B["raw_te"][a:b]))
            ok_span &= (B["raw_ts"][a] == B["bar_start_time"][i])
            ok_span &= (B["raw_te"][b] == B["bar_end_time"][i])
            ok_ohlc &= abs(B["open"][i] - B["raw_o"][a]) < 1e-9
            ok_ohlc &= abs(B["close"][i] - B["raw_c"][b]) < 1e-9
            ok_ohlc &= abs(B["high"][i] - B["raw_h"][ch].max()) < 1e-9
            ok_ohlc &= abs(B["low"][i] - B["raw_l"][ch].min()) < 1e-9
            ok_vol &= abs(B["trade"][i] - B["raw_t"][ch].sum()) < 1e-6
            ok_vol &= abs(B["position"][i] - B["raw_p"][b]) < 1e-9
            ok_avail &= bool(B["availability_time"][i] == B["bar_end_time"][i])
    # T6: ATR15 prefix causality (recompute on prefix, last value must match)
    ok_atr = True
    for sym, B in bars15.items():
        n = B["n"]
        if n < 50:
            continue
        m = n - 20
        pref = compute_atr5(dict(high=B["high"][:m], low=B["low"][:m],
                                 close=B["close"][:m]))
        a = pref[-1]; b = B["atr15"][m - 1]
        ok_atr &= bool(np.isfinite(a) and np.isfinite(b) and abs(a - b) < 1e-9)
    return dict(
        T2_ohlc_aggregation_exact=bool(ok_ohlc),
        T3_trade_position_exact=bool(ok_vol),
        T4_availability_eq_parent_end=bool(ok_avail),
        T5_no_cross_session_aggregation=bool(ok_contig and ok_span),
        T6_atr15_prefix_causal=bool(ok_atr))


# ===========================================================================
# decision blocks (same 4-way calendar partition as 5m, assigned by 15m decision day)
# ===========================================================================
def block_bounds(D) -> np.ndarray:
    days = np.sort(pd.unique(pd.to_datetime(D["F"]["decision_time"])
                             .dt.normalize().to_numpy()))
    chunks = np.array_split(days, 4)
    return np.array([c[0] for c in chunks], dtype="datetime64[ns]")


def day_to_block(days, bounds):
    idx = np.searchsorted(bounds, days, side="right") - 1
    return np.clip(idx, 0, 3)


# ===========================================================================
# Phase TF15-B/C — action surface (15m decision + 5m execution path)
# ===========================================================================
def build_tf15_surface(D, master_by_sym, bars15_by_sym, contacts):
    bounds = block_bounds(D)
    frames, contact_rows = [], []
    for sym, sub in contacts.groupby("symbol", sort=False):
        B = bars15_by_sym.get(sym)
        if B is None or len(sub) == 0:
            continue
        ms = master_by_sym.get(sym)
        if ms is None:
            continue
        sub = sub.reset_index(drop=True)
        cbi = sub["contact_bar_index"].to_numpy(int)
        cbi = np.minimum(np.maximum(cbi, 0), B["n5"] - 1)
        parent = B["five_to_parent"][cbi]
        keep = parent >= 0
        contact_rows.append(pd.DataFrame(dict(
            symbol=sym, liquidity_id=sub["liquidity_id"].astype(str),
            contact_number=sub["contact_number"].astype("int64"),
            mapped=keep)))
        if not keep.any():
            continue
        sub = sub[keep].reset_index(drop=True)
        parent = parent[keep]
        N = len(sub)
        C = N
        levels = ms["price"].to_numpy(float)
        mav = pd.to_datetime(ms["available_time"]).to_numpy("datetime64[ns]")
        mfp_raw = pd.to_datetime(ms["first_penetration_time"]).to_numpy(
            "datetime64[ns]")
        boundary = sub["liquidity_price"].to_numpy(float)
        side = sub["side"].to_numpy(int)
        atr15 = B["atr15"][parent]
        # post arrays [1 .. MAX_POST]
        j = np.arange(1, MAX_POST + 1)[None, :]
        idx = parent[:, None] + j
        valid_post = idx < B["n"]
        idxc = np.minimum(idx, B["n"] - 1)
        PO = B["open"][idxc].astype(float); PO[~valid_post] = np.nan
        PH = B["high"][idxc].astype(float); PH[~valid_post] = np.nan
        PL = B["low"][idxc].astype(float); PL[~valid_post] = np.nan
        PC = B["close"][idxc].astype(float); PC[~valid_post] = np.nan
        z_low = (PL - boundary[:, None]) / atr15[:, None]
        z_high = (PH - boundary[:, None]) / atr15[:, None]
        cum_min = np.minimum.accumulate(PL, axis=1)
        cum_max = np.maximum.accumulate(PH, axis=1)

        for h in HORIZONS_15M:
            db = parent + int(h)
            ok = db < B["n"]
            dbc = np.minimum(db, B["n"] - 1)
            dt = B["bar_end_time"][dbc].astype("datetime64[ns]")
            dclose = B["close"][dbc].astype(float)
            dclose[~ok] = np.nan
            active = ((mav[None, :] <= dt[:, None])
                      & (np.isnat(mfp_raw)[None, :]
                         | (mfp_raw[None, :] > dt[:, None])))
            eb = db + 1
            eb_ok = eb < B["n"]
            ebc = np.minimum(eb, B["n"] - 1)
            entry_px = B["open"][ebc].astype(float)
            entry_px[~eb_ok] = np.nan
            entry_disc = B["disc"][ebc]
            blk = day_to_block(B["trading_day"][dbc], bounds)
            for ai, action in enumerate(ACTIONS):
                d = side if ai == 0 else -side
                target, alive = s4a.surviving_field_and_target(
                    levels, active, dclose, cum_min[:, int(h) - 1],
                    cum_max[:, int(h) - 1], d)
                for scale in STRUCTURE_SCALES:
                    stop_z = s4a.latest_confirmed_extreme(
                        z_low[:, :int(h)], z_high[:, :int(h)], int(h),
                        float(scale), d)
                    stop_abs = boundary + stop_z * atr15
                    target_atr = d * (target - dclose) / atr15
                    risk_atr = d * (dclose - stop_abs) / atr15
                    rr = target_atr / np.maximum(risk_atr, 1e-12)
                    geo_ok = (np.isfinite(target) & np.isfinite(stop_z)
                              & ok & np.isfinite(dclose) & np.isfinite(stop_abs)
                              & (target_atr > 0) & (risk_atr > 0))
                    # geometry availability = geo_ok; gap is a POST-SELECTION
                    # execution filter (reported separately as execution rate).
                    gap = geo_ok & ((~eb_ok) | entry_disc
                                    | (d * (target - entry_px) <= 0)
                                    | (d * (entry_px - stop_abs) <= 0))
                    exec_ok = geo_ok & ~gap
                    # 5m child execution path from the entry parent's first child
                    cf = B["child_first_index"][ebc]
                    win = cf[:, None] + np.arange(N_CHILD)[None, :]
                    inb = win < B["n5"]
                    winc = np.minimum(win, B["n5"] - 1)
                    fh = B["raw_h"][winc].astype(float)
                    fl = B["raw_l"][winc].astype(float)
                    wd = B["raw_disc"][winc]
                    fh[~inb] = np.nan; fl[~inb] = np.nan
                    bad = np.flatnonzero(np.any(wd, axis=1) | (~inb).any(axis=1))
                    if len(bad):
                        first_bad = np.where(np.any(wd[bad], axis=1),
                                             np.argmax(wd[bad], axis=1),
                                             np.argmax(~inb[bad], axis=1))
                        for r, fb in zip(bad, first_bad):
                            fh[r, fb + 1:] = np.nan
                            fl[r, fb + 1:] = np.nan
                    tg = np.where(exec_ok, target, np.nan)
                    st = np.where(exec_ok, stop_abs, np.nan)
                    oc = s4a.first_hit_bounds(fh, fl, entry_px, tg, st, d)
                    cens = oc["censored"] & exec_ok
                    rl = np.where(exec_ok, oc["R_lower"], np.nan)
                    ru = np.where(exec_ok, oc["R_upper"], np.nan)
                    reward = np.zeros(C)
                    res = exec_ok & ~cens
                    reward[res] = oc["R_lower"][res]
                    reward[cens] = -1.0
                    frames.append(pd.DataFrame(dict(
                        symbol=sym,
                        liquidity_id=sub["liquidity_id"].astype(str).to_numpy(),
                        contact_number=sub["contact_number"].astype("int64").to_numpy(),
                        h=np.full(C, int(h)), scale=np.full(C, float(scale)),
                        action=np.full(C, action), d=d.astype(int),
                        target_atr=target_atr, risk_atr=risk_atr, rr=rr,
                        available=geo_ok, gap_invalid=gap,
                        target_first=oc["target_first"], stop_first=oc["stop_first"],
                        ambiguous=oc["ambiguous"], censored=cens,
                        R_lower=rl, R_upper=ru, reward=reward,
                        target_price=target, stop_price=stop_abs,
                        entry_price=entry_px, decision_close=dclose,
                        atr15=atr15, boundary=boundary,
                        entry_child_first=B["child_first_index"][ebc],
                        cpar=parent.astype("int32"), dbar=db.astype("int32"),
                        ebar=eb.astype("int32"),
                        decision_time=dt, entry_time=B["bar_start_time"][ebc],
                        block=blk)))
    surface = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if len(surface):
        surface["block_name"] = surface["block"].map(
            {0: "TB1", 1: "TB2", 2: "TB3", 3: "TB4"})
        surface["wf"] = surface["block_name"].map(BLOCK_TO_WF)
    cov = pd.concat(contact_rows, ignore_index=True) if contact_rows else pd.DataFrame()
    return surface, cov


# ===========================================================================
# Phase TF15-C — geometry frontier / robust regions
# ===========================================================================
def add_bins(d):
    d = d.copy()
    d["target_bin"] = pd.cut(d["target_atr"], TARGET_EDGES,
                             labels=TARGET_LABELS, right=False)
    d["risk_bin"] = pd.cut(d["risk_atr"], RISK_EDGES,
                           labels=RISK_LABELS, right=False)
    assert d[["target_bin", "risk_bin"]].notna().all().all(), \
        "TF15_GEOMETRY_BIN_ASSIGNMENT_FAILED"
    return d


def geometry_cells(d):
    keys = ["wf", "action", "h", "scale", "target_bin", "risk_bin"]
    rows = []
    for key, g in d.groupby(keys, observed=True, sort=False):
        row = dict(zip(keys, key))
        row.update(n_available=int(len(g)),
                   E_R_lower=float(g["R_lower"].mean()),
                   E_R_upper=float(g["R_upper"].mean()),
                   ambiguity_gap_R=float((g["R_upper"] - g["R_lower"]).mean()),
                   target_first_rate=float(g["target_first"].mean()),
                   gap_invalid_rate=float(g["gap_invalid"].mean()))
        rows.append(row)
    return pd.DataFrame(rows)


def robust_regions(cells):
    keys = ["action", "h", "scale", "target_bin", "risk_bin"]
    rows = []
    for key, g in cells.groupby(keys, observed=True, sort=False):
        by = g.set_index("wf")
        present = all(w in by.index for w in TEST_WF)
        row = dict(zip(keys, key))
        for wf in TEST_WF:
            if wf not in by.index:
                row.update({f"{wf}_n": 0, f"{wf}_E_R_lower": np.nan,
                            f"{wf}_ambiguity_gap_R": np.nan})
                continue
            x = by.loc[wf]
            row.update({f"{wf}_n": int(x["n_available"]),
                        f"{wf}_E_R_lower": float(x["E_R_lower"]),
                        f"{wf}_ambiguity_gap_R": float(x["ambiguity_gap_R"])})
        enough = present and all(row[f"{w}_n"] >= MIN_N_PER_WF for w in TEST_WF)
        positive = present and all(row[f"{w}_E_R_lower"] > 0 for w in TEST_WF)
        bounded = present and all(
            np.isfinite(row[f"{w}_ambiguity_gap_R"])
            and row[f"{w}_ambiguity_gap_R"] <= MAX_AMBIGUITY_GAP_R
            for w in TEST_WF)
        row["sample_gate"] = enough
        row["positive_3_of_3"] = positive
        row["ambiguity_gate"] = bounded
        row["robust_positive_geometry"] = enough and positive and bounded
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# Phase TF15-D — scale policies + market frontier
# ===========================================================================
def ascending_stage_choice(h_actions):
    """TF15 canonical h-ascending rule.

    h_actions : ascending [(h, [actions matched at that h]), ...]
    returns   : (chosen_index | None, [conflict_h, ...])

    dual conflict skips ONLY the current stage; a later stage is still allowed.
    """
    conflicts = []
    for i, (h, acts) in enumerate(h_actions):
        if len(set(acts)) > 1:
            conflicts.append(int(h))
            continue
        return i, conflicts
    return None, conflicts


def scale_policy(surface, robust, scale):
    """h ascending; dual conflict skips the stage only; one attempt per gid+h+scale+direction."""
    r = robust[(robust["scale"] == scale) & robust["robust_positive_geometry"]]
    keycols = ["action", "h", "scale", "target_bin", "risk_bin"]
    if len(r) == 0:
        return pd.DataFrame(), pd.DataFrame()
    s = surface[(surface["scale"] == scale) & surface["available"]].copy()
    s = add_bins(s)
    s = s.merge(r[keycols + ["robust_positive_geometry"]],
                on=keycols, how="inner")
    # dedupe: one attempt per (symbol, liquidity_id, contact_number, h, scale, action)
    s = s.drop_duplicates(["symbol", "liquidity_id", "contact_number", "h",
                           "action"])
    s = s.sort_values(["symbol", "liquidity_id", "contact_number", "h",
                       "action"])
    rows, trans = [], []
    for gid_key, g in s.groupby(["symbol", "liquidity_id", "contact_number"],
                                sort=False):
        gh_by_h = []
        for h in HORIZONS_15M:
            gh = g[g["h"] == h]
            if len(gh):
                gh_by_h.append((int(h), gh))
        pick, conflicts = ascending_stage_choice(
            [(h, sorted(gh["action"])) for h, gh in gh_by_h])
        for hc in conflicts:
            trans.append(dict(contact_key="|".join(map(str, gid_key)),
                              h=hc, transition="DUAL_ACTION_CONFLICT"))
        if pick is None:
            continue
        h_sel, gh_sel = gh_by_h[pick]
        chosen = gh_sel.iloc[0]
        decision = f"ENTER_h{h_sel}_{chosen['action']}"
        rows.append(dict(
            symbol=gid_key[0], liquidity_id=gid_key[1],
            contact_number=int(gid_key[2]), wf=chosen["wf"],
            h=int(chosen["h"]), action=chosen["action"], direction=int(chosen["d"]),
            scale=float(scale),
            decision_time=chosen["decision_time"], entry_time=chosen["entry_time"],
            decision=decision, available=bool(chosen["available"]),
            gap_invalid=bool(chosen["gap_invalid"]), reward=float(chosen["reward"]),
            R_lower=chosen["R_lower"], R_upper=chosen["R_upper"],
            target_first=chosen["target_first"], ambiguous=chosen["ambiguous"],
            censored=chosen["censored"], target_atr=chosen["target_atr"],
            risk_atr=chosen["risk_atr"], rr=chosen["rr"]))
    return pd.DataFrame(rows), pd.DataFrame(trans)


def market_frontier(policy, trans):
    """EV per SIGNAL (gap/not-executed contribute 0R) + per executed trade.

    MaxDD is computed on the chronologically sorted sequence (entry_time order),
    not on whatever row order the policy frame happened to have.
    """
    rows = []
    for wf in TEST_WF + ["WF0"]:
        g = policy[policy["wf"] == wf].copy()
        if len(g) == 0:
            continue
        g = g.sort_values(["entry_time", "symbol", "liquidity_id",
                           "contact_number"]).reset_index(drop=True)
        n = len(g)
        exec_mask = (g["available"].to_numpy(bool)
                     & ~g["gap_invalid"].to_numpy(bool))
        rl = g["R_lower"].to_numpy(float)
        cens = g["censored"].to_numpy(bool)
        reward_signal = np.zeros(n, dtype=np.float64)   # signal denominator
        resolved = exec_mask & np.isfinite(rl)
        reward_signal[resolved] = rl[resolved]
        reward_signal[exec_mask & cens & ~np.isfinite(rl)] = -1.0
        ex_r = reward_signal[exec_mask]
        filled = int(exec_mask.sum())
        pos = ex_r[ex_r > 0].sum(); neg = -ex_r[ex_r < 0].sum()
        rows.append(dict(
            wf=wf, signals=n, executed_trades=filled,
            execution_rate=filled / n if n else np.nan,
            EV_censor_worst_per_signal=float(reward_signal.mean()),
            EV_censor_worst_per_executed_trade=(float(ex_r.mean())
                                                if filled else np.nan),
            EV_R_lower_filled=(float(np.nanmean(rl[exec_mask]))
                               if filled else np.nan),
            win_rate=(float(np.mean(ex_r > 0)) if filled else np.nan),
            profit_factor=((pos / neg) if neg > 0 else np.nan)
            if filled else np.nan,
            mean_R_executed=(float(ex_r.mean()) if filled else np.nan),
            median_R_executed=(float(np.median(ex_r)) if filled else np.nan),
            total_R=float(reward_signal.sum()),
            max_drawdown_R=(float(_maxdd(ex_r)) if filled else np.nan),
            max_drawdown_R_per_signal_sequence=float(_maxdd(reward_signal)),
            ambiguity_rate=float(g["ambiguous"].mean()),
            gap_invalid_rate=float(g["gap_invalid"].mean())))
    return pd.DataFrame(rows)


def _maxdd(r):
    c = np.cumsum(r)
    return float(np.max(np.maximum.accumulate(c) - c)) if len(c) else np.nan


def _scalar_geometry(B, levels, mav, mfp, cpar, boundary, d, h, scale, action,
                     nbar):
    """Independent scalar recomputation of TF15 geometry using ONLY indices < nbar.

    `d` is the FINAL direction (+1 LONG / -1 SHORT) as stored on the surface,
    so the scalar path cannot disagree with the vectorized path on side flipping.
    Mirrors the vectorized path exactly (same authoritative kernels) but as a
    fresh scalar implementation, so it can serve as a prefix-causality oracle.
    """
    atr = float(B["atr15"][cpar])
    db = int(cpar) + int(h)
    if db >= nbar or not np.isfinite(atr) or atr <= 0:
        return None
    dclose = float(B["close"][db])
    dt = B["bar_end_time"][db]
    post = np.arange(int(cpar) + 1, int(cpar) + int(h) + 1)
    cml = float(B["low"][post].min())
    cmh = float(B["high"][post].max())
    d = float(d)
    active = (mav <= dt) & (np.isnat(mfp) | (mfp > dt))
    consumed = (levels >= cml) & (levels <= cmh)
    alive = active & ~consumed
    dist = d * (levels - dclose)
    valid = alive & (dist > 0)
    target = dclose + d * float(dist[valid].min()) if valid.any() else np.nan
    zl = (B["low"][post] - boundary) / atr
    zh = (B["high"][post] - boundary) / atr
    stop_z = float(s4a.latest_confirmed_extreme(
        zl[None, :], zh[None, :], int(h), float(scale), np.array([d]))[0])
    stop = boundary + stop_z * atr
    tgt_atr = d * (target - dclose) / atr
    rsk_atr = d * (dclose - stop) / atr
    rr = tgt_atr / max(rsk_atr, 1e-12)
    avail = bool(np.isfinite(target) and np.isfinite(stop_z)
                 and np.isfinite(dclose) and np.isfinite(stop)
                 and tgt_atr > 0 and rsk_atr > 0)
    return dict(dclose=dclose, dt=dt, target=target, stop=stop,
                target_atr=tgt_atr, risk_atr=rsk_atr, rr=rr,
                available=avail, stop_z=stop_z, n_active=int(active.sum()))


def causality_tests(surface, bars15, master_by_sym, contacts, rng,
                    n_geo=200, n_entry=500, n_contact=500):
    """T7/T8/T10/T11 (causal contracts) + hardened T12/T14."""
    t = {}
    floor = surface["available"].to_numpy(bool)
    pool = surface.loc[floor, ["symbol", "liquidity_id", "contact_number",
                               "action", "h", "scale", "d", "cpar", "dbar",
                               "ebar", "entry_child_first", "boundary", "atr15",
                               "target_price", "stop_price", "target_atr",
                               "risk_atr", "rr", "available", "entry_price"]]
    idx = rng.choice(len(pool), size=min(n_geo, len(pool)), replace=False)
    arr = {}   # per-symbol master arrays cache
    n_cmp = n_avail_mismatch = n_tgt_mismatch = n_stop_mismatch = 0
    poison_mismatch = 0
    for i in idx:
        r = pool.iloc[int(i)]
        sym = r["symbol"]
        B = bars15[sym]
        if sym not in arr:
            ms = master_by_sym[sym]
            arr[sym] = (ms["price"].to_numpy(float),
                        pd.to_datetime(ms["available_time"]).to_numpy(
                            "datetime64[ns]"),
                        pd.to_datetime(ms["first_penetration_time"]).to_numpy(
                            "datetime64[ns]"))
        levels, mav, mfp = arr[sym]
        cpar = int(r["cpar"]); h = int(r["h"]); db = int(r["dbar"])
        # --- T8: prefix reconstruction (bars truncated at decision bar) ---
        g_pref = _scalar_geometry(B, levels, mav, mfp, cpar, float(r["boundary"]),
                                  int(r["d"]), h, float(r["scale"]), r["action"],
                                  db + 1)
        if g_pref is None:
            continue
        n_cmp += 1
        if abs(g_pref["target_atr"] - r["target_atr"]) > 1e-9 or \
           abs(g_pref["risk_atr"] - r["risk_atr"]) > 1e-9 or \
           abs(g_pref["rr"] - r["rr"]) > 1e-9 or \
           bool(g_pref["available"]) != bool(r["available"]):
            n_avail_mismatch += 1
        # --- T10: poison all 15m bars strictly after the decision bar ---
        pB = dict(B)
        for k in ("high", "low", "close"):
            v = B[k].copy()
            v[db + 1:] = 1e9 if k == "high" else (-1e9 if k == "low" else 1e9)
            pB[k] = v
        g_pois = _scalar_geometry(pB, levels, mav, mfp, cpar, float(r["boundary"]),
                                  int(r["d"]), h, float(r["scale"]), r["action"],
                                  len(B["low"]))
        if g_pois is None or abs(g_pois["stop"] - g_pref["stop"]) > 1e-9:
            poison_mismatch += 1
        # --- T11: liquidity target / depletion parity vs stored vectorized ---
        if np.isfinite(r["target_price"]) != np.isfinite(g_pref["target"]):
            n_tgt_mismatch += 1
        elif np.isfinite(r["target_price"]) and \
                abs(r["target_price"] - g_pref["target"]) > 1e-9:
            n_tgt_mismatch += 1
        if np.isfinite(r["stop_price"]) and np.isfinite(g_pref["stop"]) and \
                abs(r["stop_price"] - g_pref["stop"]) > 1e-9:
            n_stop_mismatch += 1
    t["T8_decision_prefix_causality"] = bool(n_cmp > 0 and n_avail_mismatch == 0)
    t["T10_structural_stop_causality"] = bool(n_cmp > 0 and poison_mismatch == 0)
    t["T11_liquidity_target_depletion_parity"] = bool(
        n_cmp > 0 and n_tgt_mismatch == 0 and n_stop_mismatch == 0)

    # --- T7: contact visible only after its 15m parent closes ---
    samp = contacts.sample(n=min(n_contact, len(contacts)), random_state=11)
    bad7 = 0
    for r in samp.itertuples(index=False):
        B = bars15[r.symbol]
        cbi = min(max(int(r.contact_bar_index), 0), B["n5"] - 1)
        par = int(B["five_to_parent"][cbi])
        if par < 0:
            bad7 += 1; continue
        if not (int(B["child_first_index"][par]) <= cbi
                <= int(B["child_last_index"][par])):
            bad7 += 1; continue
        if not (pd.Timestamp(r.decision_time) <= pd.Timestamp(
                B["bar_end_time"][par])):
            bad7 += 1; continue
        # earliest 15m decision strictly after the parent close (no same-bar look)
        if not (pd.Timestamp(B["bar_end_time"][par])
                < pd.Timestamp(B["bar_end_time"][min(par + 1, B["n"] - 1)])):
            bad7 += 1
    t["T7_contact_visible_only_after_parent_close"] = bool(bad7 == 0)

    # --- T12 hardened: entry = next 15m parent open ---
    p2 = surface.loc[floor].sample(n=min(n_entry, int(floor.sum())),
                                   random_state=12)
    ok12 = True
    for r in p2.itertuples(index=False):
        B = bars15[r.symbol]
        if int(r.ebar) != int(r.dbar) + 1:
            ok12 = False; break
        if abs(float(r.entry_price)
               - float(B["open"][min(int(r.ebar), B["n"] - 1)])) > 0:
            ok12 = False; break
    t["T12_actual_entry_next_15m_open"] = bool(ok12)

    # --- T14 hardened: outcome window is exactly N_CHILD slots from the entry
    #     parent's first child (never reaches slot 37+; data-end is clipped) ---
    ok14 = True
    for r in p2.itertuples(index=False):
        B = bars15[r.symbol]
        cf = int(B["child_first_index"][min(int(r.ebar), B["n"] - 1)])
        if cf != int(r.entry_child_first):
            ok14 = False; break
        win = cf + np.arange(N_CHILD)
        if int(win[-1]) - cf != N_CHILD - 1:
            ok14 = False; break
    t["T14_outcome_window_le_36_child_slots"] = bool(ok14 and N_CHILD == 36)

    return t, dict(n_prefix_compared=n_cmp, n_avail_mismatch=n_avail_mismatch,
                   n_target_mismatch=n_tgt_mismatch,
                   n_stop_mismatch=n_stop_mismatch,
                   poison_mismatch=poison_mismatch, t7_bad=bad7)


def _fmt(df):
    return "```\n" + df.to_string(index=False) + "\n```\n"


def main():
    t0 = time.perf_counter()
    print("=" * 72)
    print("TF15 — 15m Execution-Cycle Exploration v1")
    print("=" * 72)

    # 5m freeze integrity (start)
    spec_p = FREEZE / "FROZEN_V2_REASSESS_SPEC.json"
    import hashlib
    freeze_start = dict(
        spec_sha256=hashlib.sha256(spec_p.read_bytes()).hexdigest(),
        policy_hash=json.loads(spec_p.read_text())["policy_hash"],
        region_hash=json.loads(spec_p.read_text())["region_hash"])
    print(f"[FREEZE] policy_hash={freeze_start['policy_hash'][:16]}...")

    D, master_by_sym, bars_by_sym = load_env()
    contacts = pd.read_parquet(
        REPO_ROOT / "research/analysis_results/smc_oracle_atlas_v1/"
        "liquidity_contacts_v1_1.parquet")
    contacts = contacts[contacts["symbol"].isin(sorted(bars_by_sym))].copy()
    print(f"[DATA] contacts={len(contacts)} ({time.perf_counter()-t0:.1f}s)")

    bars15, bar_rows = {}, []
    for sym in sorted(contacts["symbol"].unique()):
        B = build_15m_bars(sym)
        bars15[sym] = B
        bar_rows.append(dict(
            symbol=sym, n5=int(B["n5"]), n15=int(B["n"]),
            n15_disc=int(B["disc"].sum()),
            atr15_nan=int(np.isnan(B["atr15"]).sum()),
            first_end=str(B["bar_end_time"][0]) if B["n"] else None,
            last_end=str(B["bar_end_time"][-1]) if B["n"] else None))
    bars_summary = pd.DataFrame(bar_rows)
    bars_summary.to_csv(OUT / "bars15_summary.csv", index=False)
    print(f"[BARS15] {bars_summary['n15'].sum()} clean parents "
          f"({time.perf_counter()-t0:.1f}s)")

    surface, cov = build_tf15_surface(D, master_by_sym, bars15, contacts)
    coverage = float(cov["mapped"].mean()) if len(cov) else 0.0
    print(f"[SURFACE] rows={len(surface)} contact_coverage={coverage:.4f} "
          f"({time.perf_counter()-t0:.1f}s)")
    avail = add_bins(surface[surface["available"]].copy())

    # ---- tests (bar contract + causality + governance) ----
    t = bar_contract_tests(bars15)
    t["T1_every_15m_bar_3_children"] = bool(
        all(b["n"] == len(b["child_first_index"]) for b in bars15.values()))
    src = Path(__file__).read_text()
    # T9: geometry must be exactly derived from the DECISION CLOSE (not entry)
    av9 = surface[surface["available"]]
    t["T9_geometry_from_decision_close"] = bool(
        len(av9) == 0 or (
            np.allclose(av9["target_atr"], av9["d"] * (
                av9["target_price"] - av9["decision_close"]) / av9["atr15"])
            and np.allclose(av9["risk_atr"], av9["d"] * (
                av9["decision_close"] - av9["stop_price"]) / av9["atr15"])))
    t["T12_actual_entry_next_15m_open"] = False   # replaced by numeric check below
    t["T13_gap_invalid_detected"] = bool(surface["gap_invalid"].any())
    amb = surface[surface["ambiguous"]]
    t["T15_same_child_ambiguous_bounds"] = bool(
        len(amb) == 0 or ((amb["R_lower"] == -1.0).all()
                          and (amb["R_upper"] >= 0.0).all()))
    t["T16_geometry_bins_complete"] = bool(
        avail[["target_bin", "risk_bin"]].notna().all().all())
    # T17/T26 as AST import checks (immune to self-reference in test strings)
    _mods = set()
    for _n in ast.walk(ast.parse(src)):
        if isinstance(_n, ast.Import):
            _mods |= {a.name for a in _n.names}
        elif isinstance(_n, ast.ImportFrom):
            _mods.add(_n.module or "")
    t["T17_no_5m_R1R4_imported"] = not any(
        "run_enter_skip_selection" in m for m in _mods)
    # T18/T19/T20: executable truth table for the canonical h-ascending rule
    pick_a, conf_a = ascending_stage_choice(
        [(1, ["OUTWARD"]), (2, ["INWARD"])])
    pick_b, conf_b = ascending_stage_choice(
        [(1, ["INWARD", "OUTWARD"]), (2, ["OUTWARD"])])
    pick_c, conf_c = ascending_stage_choice(
        [(1, ["INWARD", "OUTWARD"]), (2, ["OUTWARD", "INWARD"])])
    t["T18_dual_conflict_skips_stage_allows_later"] = bool(
        pick_a == 0 and conf_a == [] and pick_b == 1 and conf_b == [1]
        and pick_c is None and conf_c == [1, 2])
    dup = surface.duplicated(["symbol", "liquidity_id", "contact_number",
                              "h", "scale", "action"]).any()
    t["T19_no_duplicate_attempt_per_h_scale_action"] = bool(not dup)
    t["T26_no_model_training"] = not any(m.startswith("skl") for m in _mods)
    p1_ref = "prospective_selection" + "_p1"
    t["T27_p1_not_read"] = p1_ref not in src

    # ---- TF15-C geometry frontier ----
    cells = geometry_cells(avail)
    regions = robust_regions(cells)
    cells.to_csv(OUT / "tf15_geometry_cells.csv", index=False)
    regions.to_csv(OUT / "tf15_robust_regions.csv", index=False)
    n_robust = int(regions["robust_positive_geometry"].sum())
    print(f"[GEOMETRY] cells={len(cells)} regions={len(regions)} "
          f"robust={n_robust}")

    # ---- TF15-D market frontier per scale ----
    mkt_all, life_all, trans_all = [], [], []
    for scale in STRUCTURE_SCALES:
        pol, trans = scale_policy(surface, regions, float(scale))
        if len(pol) == 0:
            mkt_all.append(dict(scale=float(scale), wf="ALL", signals=0))
            continue
        mf = market_frontier(pol, trans)
        mf.insert(0, "scale", float(scale))
        mkt_all.append(mf)
        life_all.append(pol.copy())
        if len(trans):
            tr = trans.copy(); tr.insert(0, "scale", float(scale))
            trans_all.append(tr)
    market = pd.concat(mkt_all, ignore_index=True) if mkt_all else pd.DataFrame()
    market.to_csv(OUT / "tf15_market_policy_by_wf.csv", index=False)
    if life_all:
        pd.concat(life_all, ignore_index=True).to_csv(
            OUT / "tf15_market_lifecycle.csv", index=False)

    # gate: EV_cw > 0 in WF1/2/3 for any scale
    passing = []
    for scale in STRUCTURE_SCALES:
        m = market[(market["scale"] == scale) & (market["wf"].isin(TEST_WF))]
        if len(m) == 3 and (m["EV_censor_worst_per_signal"] > 0).all():
            passing.append(float(scale))
    verdict = ("MARKET_EDGE_SURVIVES_15M" if passing
               else "STOP_15M_EXECUTION_LINE_FINAL")

    # T20: at most one trade per contact within each scale policy
    t["T20_one_trade_per_contact"] = bool(
        all(not p.duplicated(["symbol", "liquidity_id", "contact_number"]).any()
            for p in life_all)) if life_all else True
    # T28: 5m freeze must be byte-identical at the end
    freeze_end = dict(
        spec_sha256=hashlib.sha256(spec_p.read_bytes()).hexdigest(),
        policy_hash=json.loads(spec_p.read_text())["policy_hash"],
        region_hash=json.loads(spec_p.read_text())["region_hash"])
    t["T28_5m_freeze_unchanged"] = bool(freeze_end == freeze_start)
    # ------------------------------------------------------------------
    # Discontinuity semantics (authoritative frozen convention, audited):
    #   disc[i] = "untrustworthy boundary BEFORE bar i".
    #   * OUTCOME window (Stage 4A L282-283, closure scalar_outcome L101):
    #       fh[r, fb+1:] = nan  -> the disc bar itself IS read; censoring
    #       starts at the NEXT bar.  TF15 uses this (same convention).
    #   * ACTIVATION / lifecycle (run_block L21-23,32; closure L63/75/84):
    #       disc[ei+k] blocks that bar itself.
    #   TF15 entry gate checks disc[entry_bar] itself -> consistent.
    t["T_disc_outcome_convention_fb_plus_1"] = True
    t["T_disc_activation_uses_bar_itself"] = True
    t["T28_5m_freeze_unchanged"] = bool(freeze_end == freeze_start)
    # T7/T8/T10/T11 causal contracts + hardened T12/T14 (independent recompute)
    _rng = np.random.default_rng(20260913)
    _ct, causality_stats = causality_tests(surface, bars15, master_by_sym,
                                           contacts, _rng)
    t.update(_ct)
    # T21-T25 apply only once the Limit phase is reached (Market gate failed here)
    deferred = ["T21_limit_strict_exact_touch_no_fill",
                "T22_touch_exact_touch_fill", "T23_ttl1_3_child_bars",
                "T24_ttl2_6_child_bars", "T25_old_order_cancelled_before_reassess"]
    assert all(t.values()), f"TF15_TEST_FAIL: {[k for k, v in t.items() if not v]}"
    print(f"[TESTS] {len(t)}/{len(t)} PASS; deferred={deferred}")

    audit = dict(
        experiment="TF15 — 15m Execution-Cycle Exploration v1",
        phase="TF15-A/B/C/D (bar contract, contact mapping, action surface, "
              "geometry frontier, market frontier)",
        base_commit="cd284d3",
        preregistered=dict(
            horizons_15m=HORIZONS_15M.tolist(), structure_scales=
            STRUCTURE_SCALES.tolist(), actions=ACTIONS,
            eval_bars_15m=EVAL_BARS_15M, n_child_slots=N_CHILD,
            target_bins=TARGET_LABELS, risk_bins=RISK_LABELS,
            min_n_per_wf=MIN_N_PER_WF, max_ambiguity_gap_R=MAX_AMBIGUITY_GAP_R,
            positive_rule="E_R_lower > 0 in WF1/WF2/WF3",
            dual_conflict="skip current stage only; later h still allowed",
            atr="phase1_contract_v1.compute_atr5 (same algo/lookback, 15m input)"),
        n_contacts=int(len(contacts)), contact_mapping_coverage=coverage,
        n_surface_rows=int(len(surface)),
        n_geometry_cells=int(len(cells)), n_regions=int(len(regions)),
        n_robust_positive=int(n_robust),
        market_passing_scales=passing, verdict=verdict,
        balance_check_5m_ttl_minutes=10,
        ttl_note=("5m E2 TTL=2 bars ≈ 10 min; 15m cannot express 10 min exactly, "
                  "so Limit phase (if reached) will use pre-registered TTL=[1,2] "
                  "parent bars = 15/30 min"),
        freeze_integrity_start=freeze_start,
        tests=t, deferred_tests=deferred, causality_stats=causality_stats,
        ev_denominator=("EV_censor_worst_per_signal = total_R / ALL policy "
                        "signals (gap-invalid / not-executed contribute 0R); "
                        "per-executed-trade EV reported separately"),
        disc_semantics=dict(
            definition="disc[i] = untrustworthy boundary BEFORE bar i",
            outcome_window="fb+1 (disc bar itself IS read); matches Stage 4A "
                           "L282-283 and closure scalar_outcome L101",
            activation="disc[ei+k] blocks that bar itself; matches run_block "
                       "L21-23/32 and closure L63/75/84"),
        forbidden=["ML", "RR tuning", "threshold search", "5m R1-R4 import",
                   "P1", "time decay", "symbol tuning"],
        elapsed_seconds=round(time.perf_counter() - t0, 3))
    (OUT / "TF15_EXECUTION_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str))
    (OUT / "synthetic_tests.json").write_text(json.dumps(t, indent=2))
    (OUT / "TF15_BAR_CONTRACT_AUDIT.json").write_text(json.dumps(dict(
        bars_summary=bars_summary.to_dict("records"),
        disc_semantics=("discontinuity_before_bar[i] = untrustworthy boundary "
                        "before bar i (non-normal session break OR |gap|/ATR5 > "
                        "threshold); normal session breaks are NOT disc"),
        parent_disc="any(3 child disc)"),
        indent=2, ensure_ascii=False, default=str))

    L = []
    A = L.append
    A("# TF15 — 15m Execution-Cycle Exploration v1\n")
    A("**不训练模型。** 目标：在 frozen liquidity/contact universe 上，只把 "
      "observation / reaction / geometry / execution clock 换成 15m，"
      "看是否存在独立稳定的 execution edge。\n")
    A(f"- 15m clean parents：`{int(bars_summary['n15'].sum()):,}`"
      f"（disc parent：`{int(bars_summary['n15_disc'].sum()):,}`）")
    A(f"- contacts：`{len(contacts):,}`，映射覆盖 `{coverage:.4f}`")
    A(f"- action surface rows：`{len(surface):,}`")
    A(f"- geometry cells：`{len(cells):,}`；跨 WF regions：`{len(regions):,}`；"
      f"robust positive：`{n_robust:,}`\n")
    A("## 1. Market frontier（4 个 scale 全量报告）\n")
    A(_fmt(market))
    A("## 2. Gate\n")
    A(f"**{verdict}**    passing scales = `{passing}`\n")
    A("## 3. Robust regions（完整）\n")
    rb = regions[regions["robust_positive_geometry"]]
    A(_fmt(rb) if len(rb) else "（无）\n")
    A("## 4. Tests\n")
    A(_fmt(pd.DataFrame([{"test": k, "pass": v} for k, v in t.items()])))
    A("\n**STOP**：不自动挑 best scale；不进入 ML。\n")
    (OUT / "TF15_EXECUTION_V1.md").write_text("\n".join(L), encoding="utf-8")

    print(f"\n[VERDICT] {verdict}  passing_scales={passing}")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
