"""Robust 5m Trade Oracle via Finite-Horizon Dynamic Programming (v1.1).

FUTURE-ORACLE-R1.1-CORRECTNESS-HARDENING

Purpose
-------
Environment-independent trade oracle: for every 5m decision close `t`, solve a
finite-horizon DP that returns the optimal action

    A*(t, H) in {Long, Short, Wait, Tie}

and its value (in gross price points, Cost = 0), for horizons H = 6 / 12 / 24
(30 / 60 / 120 minutes). The oracle uses ONLY 5m OHLC path + execution
constraints. It is FORBIDDEN from using DTP / SR / Liquidity / 15m / 1h / 4h /
any indicator state. See AGENTS.md and the task spec.

This v1.1 hardens correctness issues found in v1.0 (R1):

A. The ENTIRE oracle horizon may not cross a discontinuity. `Q_W(t,h)=0` when
   `disc[t+1]=1` (not just the held position). Wait recursion therefore cannot
   borrow value from a later discontinuity segment.
B. `decision_time` = bar_start_time + 5min (the close), not the bar start.
   `decision_bar_start_time` retained for audit.
C. `segment = cumsum(disc)` (canonical owner semantics); bar i belongs to the
   NEW segment when `disc[i]=1` (entering bar i has a boundary before it).
D. Tie handling uses a numeric tolerance, never code-order bias.
   best_action_set = {a : |Q_a - Q_max| <= eps}; only n_best==1 yields a unique
   action, otherwise Tie. stable_action needs all 3 horizons = SAME UNIQUE.
E. `label_available_time` per H: the last bar whose close is needed to compute
   the label (NOT the optimal exit time). reason: HORIZON_END / DISCONTINUITY /
   DATA_END. Required before any tree model to avoid label leakage.
F. Raw price-point Q/Edge/MFE/MAE retained AND ATR5-normalized versions added
   (canonical causal `compute_atr5`). ATR only rescales units; it does NOT enter
   argmax / exit selection / DP recursion, so directions are unchanged.
G. Exclusion accounting: input = output + excluded_tail + excluded_disc_before
   + excluded_no_valid_roundtrip, mutually exclusive; plus tie_rows.
H. Per-symbol raw 5m is loaded exactly once; samples reuse that load (no second
   reload).

Execution contract (frozen)
---------------------------
- Decision at 5m close of bar `t`.
- Entry fills at the open of the next valid 5m bar `e = t+1`
  (next-open, never same-bar hindsight).
- Exit decision at close of bar `e+h-1`, fill at open of bar `e+h`
  (consistent next-open). Holding length `h` in 1..H.
- Single position, fixed 1 unit, no add, no simultaneous long/short.
- **At most one round-trip trade per Oracle horizon.** Exit does not re-enter
  within the same horizon. (This is the intended oracle: "is THIS 5m bar a good
  trade opportunity?", not "how many times can we scalp if we knew the future".)
- No crossing a discontinuity (holding OR waiting).
- Horizon `H` caps the holding length; paths are also capped by data end.
- Utility v1 = GrossPnL - Cost, with Cost = 0 (no cost owner in project).
  MAE / MFE / holding bars recorded as outcomes, NOT penalised.

DP semantics (the part the reviewer audits)
-------------------------------------------
Flat value (option value of being flat at t with h steps left):

    V_flat[t][h] = max( Q_L(t,h), Q_S(t,h), Q_W(t,h) )

where
    Q_L(t,h) = max_{1<=k<=h} (open[e+k] - open[e])           [no-disc span]
    Q_S(t,h) = max_{1<=k<=h} (open[e]   - open[e+k])
    Q_W(t,h) = V_flat[t+1][h-1]        if disc[t+1]==False
             = 0                        if disc[t+1]==True      (Fix A)
    Q_W(t,h) owns genuine future option value: it is NOT forced to 0.
A*(t,h) = argmax over {Long, Short, Wait}; edge = best - second.

All three horizons share ONE data load + ONE segment build + ONE forward
precompute of QL/QS + ONE backward flat DP. H=6/12/24 read from tables.

Reused (narrow helpers only, NO second 5m loader):
    research.export_ob_trigger_execution_v21.load_raw_5m
    research.phase1_tradability.phase1_contract_v1.discontinuity_flags
    research.phase1_tradability.phase1_contract_v1.compute_atr5  (Fix F units)
Forbidden to reuse: the fixed stop / liquidity target of
run_fixed_execution_baseline_v1.py.

Outputs (artifacts/robust_trade_oracle_dp_v1/):
    robust_trade_oracle_rows.parquet   row-level label table (downstream-ready)
    robust_trade_oracle_summary.json   T2 summary stats
    ROBUST_TRADE_ORACLE_PROTOCOL.json   frozen contract (v1.1)
    ROBUST_TRADE_ORACLE_AUDIT.json      evidence packet
    robust_trade_oracle_report.md       human report
"""
from __future__ import annotations

import argparse
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# --- narrow reusable owners (no second 5m loader) --------------------------
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import research.export_ob_trigger_execution_v21 as _raw
from research.phase1_tradability.phase1_contract_v1 import (
    discontinuity_flags,
    compute_atr5,
)

HMAX = 24                 # max horizon scanned; output uses 6/12/24
HORIZONS = (6, 12, 24)
SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]
NEG = -np.inf
EPS_TIE = 1e-6            # numeric tie tolerance for Q values (price points)
OUT = Path("artifacts/robust_trade_oracle_dp_v1")
OUT.mkdir(parents=True, exist_ok=True)
ACTIONS = ("Long", "Short", "Wait")


# ===========================================================================
# data owner
# ===========================================================================
def build_bars(sym: str) -> dict:
    """Raw 5m owner (single source). Returns numpy-backed bar dict + ATR5."""
    raw = _raw.load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    o = raw["open"].to_numpy(float)
    h = raw["high"].to_numpy(float)
    l = raw["low"].to_numpy(float)
    c = raw["close"].to_numpy(float)
    t = pd.to_datetime(raw["bar_start_time"]).to_numpy()
    disc = np.asarray(discontinuity_flags(sym), bool)
    # Fix C: canonical segment = cumsum(disc); bar i is in the NEW segment when
    # disc[i]=1 (a boundary occurs before entering bar i).
    seg = np.cumsum(disc.astype(np.int64))
    # Fix B: decision happens at the 5m close (bar_start + 5min).
    decision_time = t + pd.Timedelta(minutes=5)
    # Fix F: canonical causal ATR5 as unit scale (high/low/close contract).
    atr5 = compute_atr5(dict(high=h, low=l, close=c))
    return dict(symbol=sym, o=o, h=h, l=l, c=c, t=t, disc=disc, seg=seg,
                decision_time=decision_time, atr5=atr5, n=len(raw))


# ===========================================================================
# core DP
# ===========================================================================
def oracle_core(bars: dict, horizons=HORIZONS, hmax=HMAX) -> dict:
    """Shared one-pass computation of QL/QS flat-DP tables (v1.1).

    Returns arrays (n x (hmax+1)) plus validity, best-hold, has_roundtrip.
    """
    o, h, l, disc = bars["o"], bars["h"], bars["l"], bars["disc"]
    n = bars["n"]

    # valid decision bar: entry bar e=t+1 tradeable, and at least one hold
    # (open[e+1] must exist) => t+2 < n.
    disc_next = np.zeros(n, bool)
    if n > 1:
        disc_next[:-1] = disc[1:]
    valid = (~disc_next) & (np.arange(n) + 2 < n)

    ql = np.full((n, hmax + 1), NEG, dtype=float)
    qs = np.full((n, hmax + 1), NEG, dtype=float)
    best_h_l = np.zeros((n, hmax + 1), dtype=int)
    best_h_s = np.zeros((n, hmax + 1), dtype=int)

    for t in range(n):
        if not valid[t]:
            continue
        e = t + 1
        # first discontinuity bar index > e (held window may not cross it)
        nxt = np.flatnonzero(disc[e + 1:])
        m = (e + 1 + int(nxt[0])) if len(nxt) else n
        cur_l, cur_s = NEG, NEG
        bh_l = bh_s = 0
        for hh in range(1, hmax + 1):
            ex = e + hh
            if ex > n - 1:
                break                 # exit open beyond data
            if ex >= m:
                break                 # hold would exit at / cross discontinuity
            pnl_l = o[ex] - o[e]
            pnl_s = o[e] - o[ex]
            if pnl_l > cur_l:
                cur_l, bh_l = pnl_l, hh
            if pnl_s > cur_s:
                cur_s, bh_s = pnl_s, hh
            ql[t, hh] = cur_l
            qs[t, hh] = cur_s
            best_h_l[t, hh] = bh_l
            best_h_s[t, hh] = bh_s
        # Columns beyond the truncation keep the last running max (monotone).
        for hh in range(1, hmax + 1):
            if not np.isfinite(ql[t, hh]):
                ql[t, hh] = cur_l
                qs[t, hh] = cur_s
                best_h_l[t, hh] = bh_l
                best_h_s[t, hh] = bh_s

    # backward flat DP (Fix A: Wait cannot borrow a later discontinuity segment)
    Vflat = np.zeros((n + 1, hmax + 1), dtype=float)   # Vflat[n][*] = 0
    for t in range(n - 1, -1, -1):
        disc_next_t = (t + 1 < n) and bool(disc[t + 1])
        a_l = ql[t] if valid[t] else None
        a_s = qs[t] if valid[t] else None
        for hh in range(1, hmax + 1):
            wait_val = 0.0 if disc_next_t else Vflat[t + 1, hh - 1]
            al = a_l[hh] if a_l is not None else NEG
            aS = a_s[hh] if a_s is not None else NEG
            if al >= aS and al >= wait_val:
                b = "Long"
            elif aS >= al and aS >= wait_val:
                b = "Short"
            else:
                b = "Wait"
            Vflat[t, hh] = (al if b == "Long" else aS if b == "Short"
                            else wait_val)

    # has_roundtrip[t]: does ANY trade (now or after waiting) exist within the
    # maximal horizon? Used for exclusion accounting (Fix G).
    has_roundtrip = np.zeros(n, bool)
    hmax_scan = max(horizons)
    for t in range(n):
        if not valid[t]:
            continue
        if (np.any(np.isfinite(ql[t])) or np.any(np.isfinite(qs[t]))
                or (t + 1 < n and Vflat[t + 1, hmax_scan] > 0)):
            has_roundtrip[t] = True

    return dict(ql=ql, qs=qs, Vflat=Vflat, valid=valid,
                best_h_l=best_h_l, best_h_s=best_h_s,
                has_roundtrip=has_roundtrip, n=n, horizons=tuple(horizons))


# ===========================================================================
# tie-aware action classification (Fix D)
# ===========================================================================
def _classify(qvals: dict, eps: float = EPS_TIE):
    """qvals: {Long, Short, Wait} -> (unique_action, best_set, n_best).

    unique_action is "Tie" unless exactly one action ties the max within eps.
    """
    Qmax = max(qvals.values())
    if not np.isfinite(Qmax):
        return "Tie", [], 0
    best = [a for a in ACTIONS if abs(qvals[a] - Qmax) <= eps]
    n = len(best)
    unique = best[0] if n == 1 else "Tie"
    return unique, best, n


# ===========================================================================
# label availability (Fix E)
# ===========================================================================
def _label_available(e, H, first_disc, n):
    """Return (last_needed_bar, reason)."""
    horizon_end = e + H
    last_needed = min(horizon_end, first_disc - 1, n - 1)
    disc_within = (first_disc <= horizon_end) and (first_disc < n)
    capped_data = horizon_end > (n - 1)
    reason = ("DISCONTINUITY" if disc_within
              else "DATA_END" if capped_data else "HORIZON_END")
    return int(last_needed), reason


# ===========================================================================
# row-level label table
# ===========================================================================
def _norm(x, scale):
    if scale is None or not np.isfinite(scale) or scale <= 0:
        return None
    return float(x) / float(scale)


def build_rows(bars: dict, core: dict) -> pd.DataFrame:
    o, h, l, disc = bars["o"], bars["h"], bars["l"], bars["disc"]
    tt, dt, seg, atr = (bars["t"], bars["decision_time"], bars["seg"],
                         bars["atr5"])
    n = bars["n"]
    valid = core["valid"]
    ql, qs, Vflat = core["ql"], core["qs"], core["Vflat"]
    best_h_l, best_h_s = core["best_h_l"], core["best_h_s"]
    has_roundtrip = core["has_roundtrip"]
    horizons = core["horizons"]

    rows = []
    for t in np.flatnonzero(valid):
        t = int(t)
        if not has_roundtrip[t]:
            continue                    # excluded_no_valid_roundtrip (Fix G)
        e = t + 1
        nxt = np.flatnonzero(disc[e + 1:])
        first_disc = (e + 1 + int(nxt[0])) if len(nxt) else n
        scale = float(atr[t]) if np.isfinite(atr[t]) else None

        rec = dict(
            symbol=bars["symbol"],
            decision_bar_start_time=pd.Timestamp(tt[t]),
            decision_time=pd.Timestamp(dt[t]),
            decision_bar_index=int(t),
            entry_bar_index=int(e),
            segment=int(seg[t]),
            entry_valid=True,
            atr5_t=(None if scale is None else float(atr[t])),
        )
        actions = []
        for H in horizons:
            qlt = ql[t, H]
            qst = qs[t, H]
            # Fix A: Wait recursion cannot borrow a later discontinuity segment
            qwt = (0.0 if (t + 1 < n and bool(disc[t + 1]))
                   else Vflat[t + 1, H - 1])
            a_l = qlt if np.isfinite(qlt) else NEG
            a_s = qst if np.isfinite(qst) else NEG
            vals = {"Long": a_l, "Short": a_s, "Wait": qwt}
            order = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
            best_a, best_v = order[0]
            second_v = order[1][1]
            edge = (best_v - second_v) if np.isfinite(second_v) else None
            unique, best_set, n_best = _classify(vals)
            rec[f"QL_{H}"] = (None if not np.isfinite(qlt) else float(qlt))
            rec[f"QS_{H}"] = (None if not np.isfinite(qst) else float(qst))
            rec[f"QW_{H}"] = float(qwt)
            rec[f"QL_{H}_ATR"] = _norm(qlt, scale)
            rec[f"QS_{H}_ATR"] = _norm(qst, scale)
            rec[f"QW_{H}_ATR"] = _norm(qwt, scale)
            rec[f"edge_{H}"] = (None if edge is None else float(edge))
            rec[f"edge_{H}_ATR"] = _norm(edge, scale)
            rec[f"action_{H}"] = unique
            rec[f"best_action_set_{H}"] = (
                "Tie" if unique == "Tie" else ",".join(best_set))
            rec[f"n_best_actions_{H}"] = int(n_best)
            # Fix E: label availability
            lab_bar, reason = _label_available(e, H, first_disc, n)
            rec[f"label_available_bar_{H}"] = lab_bar
            rec[f"label_available_time_{H}"] = (
                pd.Timestamp(tt[lab_bar]) + pd.Timedelta(minutes=5))
            rec[f"oracle_terminal_reason_{H}"] = reason
            if unique == "Long":
                bh = int(best_h_l[t, H])
                xb = e + bh
                rec[f"exit_bars_{H}"] = int(xb)
                rec[f"holding_bars_{H}"] = bh
                rec[f"realized_move_{H}"] = float(o[xb] - o[e])
                win = max(float(h[e:xb].max()), float(o[xb])) - float(o[e])
                adve = float(o[e]) - min(float(l[e:xb].min()), float(o[xb]))
                rec[f"MFE_{H}"] = float(win)
                rec[f"MAE_{H}"] = float(adve)
                rec[f"MFE_{H}_ATR"] = _norm(win, scale)
                rec[f"MAE_{H}_ATR"] = _norm(adve, scale)
            elif unique == "Short":
                bh = int(best_h_s[t, H])
                xb = e + bh
                rec[f"exit_bars_{H}"] = int(xb)
                rec[f"holding_bars_{H}"] = bh
                rec[f"realized_move_{H}"] = float(o[e] - o[xb])
                win = float(o[e]) - min(float(l[e:xb].min()), float(o[xb]))
                adve = max(float(h[e:xb].max()), float(o[xb])) - float(o[e])
                rec[f"MFE_{H}"] = float(win)
                rec[f"MAE_{H}"] = float(adve)
                rec[f"MFE_{H}_ATR"] = _norm(win, scale)
                rec[f"MAE_{H}_ATR"] = _norm(adve, scale)
            else:
                rec[f"exit_bars_{H}"] = None
                rec[f"holding_bars_{H}"] = None
                rec[f"realized_move_{H}"] = None
                rec[f"MFE_{H}"] = None
                rec[f"MAE_{H}"] = None
                rec[f"MFE_{H}_ATR"] = None
                rec[f"MAE_{H}_ATR"] = None
            actions.append(unique)
        # agreement / stable label (Fix D)
        c = Counter(actions)
        top, cnt = c.most_common(1)[0]
        rec["agreement"] = round(cnt / len(horizons), 4)
        rec["stable_action"] = (top if (cnt == len(horizons) and top != "Tie")
                                else "Ambiguous")
        rec["_any_tie"] = int("Tie" in actions)
        rec["_agree23"] = int(cnt >= 2)
        rows.append(rec)

    df = pd.DataFrame(rows)
    front = ["symbol", "decision_bar_start_time", "decision_time",
             "decision_bar_index", "entry_bar_index", "segment", "entry_valid",
             "atr5_t", "agreement", "stable_action", "_any_tie", "_agree23"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    return df


def compute_oracle(bars: dict, horizons=HORIZONS, hmax=HMAX):
    core = oracle_core(bars, horizons, hmax)
    df = build_rows(bars, core)
    return df, core


# ===========================================================================
# summary (Fix G, cross-horizon transitions, edge_ATR primary)
# ===========================================================================
def _quant(s, col):
    if s is None or len(s) == 0 or col not in s:
        return None
    v = pd.to_numeric(s[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(v) == 0:
        return None
    return dict(n=int(len(v)),
                p10=round(float(v.quantile(.1)), 5),
                p25=round(float(v.quantile(.25)), 5),
                p50=round(float(v.quantile(.5)), 5),
                p75=round(float(v.quantile(.75)), 5),
                p90=round(float(v.quantile(.9)), 5),
                mean=round(float(v.mean()), 5))


def _transitions(df, a, b):
    cats = ["Long", "Short", "Wait", "Tie"]
    ct = pd.crosstab(df[f"action_{a}"], df[f"action_{b}"])
    ct = ct.reindex(index=cats, columns=cats, fill_value=0)
    counts = ct.to_numpy().tolist()
    pct = (ct / max(1, ct.values.sum())).round(4).to_numpy().tolist()
    return dict(labels=cats, counts=counts, pct=pct)


def summarize(df: pd.DataFrame, excl: dict, horizons=HORIZONS) -> dict:
    n = len(df)
    s = dict(n_decisions=int(n))
    for a in ("Long", "Short", "Wait", "Ambiguous"):
        s[f"stable_{a}_pct"] = round(
            float((df["stable_action"] == a).mean()) * 100, 3) if n else 0.0
    s["agreement_3of3_pct"] = round(
        float((df["agreement"] == 1.0).mean()) * 100, 3) if n else 0.0
    s["agreement_2of3_pct"] = round(
        float(df["_agree23"].mean()) * 100, 3) if n else 0.0
    s["agreement_mean"] = round(float(df["agreement"].mean()), 4) if n else 0.0
    for H in horizons:
        for a in ("Long", "Short", "Wait", "Tie"):
            s[f"action_{H}_{a}_pct"] = round(
                float((df[f"action_{H}"] == a).mean()) * 100, 3) if n else 0.0
    # tie & zero-edge diagnostics
    s["tie_count_H6"] = int((df["action_6"] == "Tie").sum())
    s["tie_count_H12"] = int((df["action_12"] == "Tie").sum())
    s["tie_count_H24"] = int((df["action_24"] == "Tie").sum())
    s["tie_rows"] = int(df["_any_tie"].sum())
    stable_trade = df[df["stable_action"].isin(["Long", "Short"])]
    s["stable_trade_zero_edge_count"] = int(
        (pd.to_numeric(stable_trade["edge_24"], errors="coerce") == 0).sum())
    # by symbol
    by_sym = []
    for sym, g in df.groupby("symbol"):
        e = excl[sym]
        by_sym.append(dict(
            symbol=sym, input_5m_rows=e["input_5m_rows"],
            output_decision_rows=int(len(g)),
            excluded_tail=e["excluded_tail"],
            excluded_disc_before_entry=e["excluded_discontinuity_before_entry"],
            excluded_no_valid_roundtrip=e["excluded_no_valid_roundtrip"],
            tie_rows=int(g["_any_tie"].sum()),
            stable_Long_pct=round(
                float((g["stable_action"] == "Long").mean()) * 100, 3),
            stable_Short_pct=round(
                float((g["stable_action"] == "Short").mean()) * 100, 3),
            stable_Wait_pct=round(
                float((g["stable_action"] == "Wait").mean()) * 100, 3),
            stable_Ambiguous_pct=round(
                float((g["stable_action"] == "Ambiguous").mean()) * 100, 3),
            agreement_3of3_pct=round(
                float((g["agreement"] == 1.0).mean()) * 100, 3),
        ))
    s["by_symbol"] = by_sym
    # by calendar-month time block ("TB block" mapped to month)
    blk = pd.to_datetime(df["decision_time"]).dt.strftime("%Y-%m")
    by_month = []
    for m, g in df.groupby(blk):
        by_month.append(dict(
            time_block=m, n=int(len(g)),
            stable_Long_pct=round(
                float((g["stable_action"] == "Long").mean()) * 100, 3),
            stable_Short_pct=round(
                float((g["stable_action"] == "Short").mean()) * 100, 3),
            stable_Wait_pct=round(
                float((g["stable_action"] == "Wait").mean()) * 100, 3),
            stable_Ambiguous_pct=round(
                float((g["stable_action"] == "Ambiguous").mean()) * 100, 3)))
    s["by_time_block"] = by_month
    # distributions: report edge_ATR primarily, keep raw
    s["edge_ATR"] = _quant(df, "edge_24_ATR")
    s["edge_raw"] = _quant(df, "edge_24")
    s["holding"] = _quant(df, "holding_bars_24")
    s["mfe_ATR"] = _quant(df, "MFE_24_ATR")
    s["mfe_raw"] = _quant(df, "MFE_24")
    s["mae_ATR"] = _quant(df, "MAE_24_ATR")
    s["mae_raw"] = _quant(df, "MAE_24")
    # stable-class edge stats
    for a in ("Long", "Short", "Wait"):
        sub = df[df["stable_action"] == a]
        s[f"stable_{a}_median_edge_ATR"] = _med(sub, "edge_24_ATR")
        s[f"stable_{a}_median_edge_raw"] = _med(sub, "edge_24")
        s[f"stable_{a}_median_holding"] = _med(sub, "holding_bars_24")
    # cross-horizon transition matrices
    s["transition_6_12"] = _transitions(df, 6, 12)
    s["transition_12_24"] = _transitions(df, 12, 24)
    s["transition_6_24"] = _transitions(df, 6, 24)
    # exclusion totals
    s["exclusion_total"] = dict(
        input_5m_rows=int(sum(e["input_5m_rows"] for e in excl.values())),
        output_decision_rows=int(n),
        excluded_tail=int(sum(e["excluded_tail"] for e in excl.values())),
        excluded_discontinuity_before_entry=int(sum(
            e["excluded_discontinuity_before_entry"] for e in excl.values())),
        excluded_no_valid_roundtrip=int(sum(
            e["excluded_no_valid_roundtrip"] for e in excl.values())),
        tie_rows=int(df["_any_tie"].sum()),
    )
    return s


def _med(s, col):
    v = pd.to_numeric(s[col], errors="coerce").dropna()
    return round(float(v.median()), 5) if len(v) else None


# ===========================================================================
# human oracle samples (5)
# ===========================================================================
def pick_samples(df: pd.DataFrame, bars_by_sym: dict) -> list:
    out = []
    longs = df[df["stable_action"] == "Long"]
    if len(longs):
        r = longs.loc[longs["edge_24"].idxmax()]
        out.append(("stable_Long", _sample(r)))
    shorts = df[df["stable_action"] == "Short"]
    if len(shorts):
        r = shorts.loc[shorts["edge_24"].idxmax()]
        out.append(("stable_Short", _sample(r)))
    waits = df[df["stable_action"] == "Wait"]
    if len(waits):
        r = waits.loc[waits["QW_24"].idxmax()]
        out.append(("stable_Wait", _sample(r)))
    ties = df[df["action_24"] == "Tie"]
    if len(ties):
        r = ties.iloc[len(ties) // 2]
        out.append(("tie", _sample(r)))
    disc = df[df["oracle_terminal_reason_24"] == "DISCONTINUITY"]
    if len(disc):
        r = disc.iloc[len(disc) // 2]
        out.append(("discontinuity_near", _sample(r)))
    return out


def _sample(r) -> dict:
    d = dict(symbol=r["symbol"],
             decision_bar_start_time=str(r["decision_bar_start_time"]),
             decision_time=str(r["decision_time"]),
             decision_bar_index=int(r["decision_bar_index"]),
             segment=int(r["segment"]), agreement=float(r["agreement"]),
             stable_action=r["stable_action"], atr5_t=r["atr5_t"])
    for H in HORIZONS:
        d[f"H{H}"] = dict(
            QL=r[f"QL_{H}"], QS=r[f"QS_{H}"], QW=r[f"QW_{H}"],
            QL_ATR=r[f"QL_{H}_ATR"], QS_ATR=r[f"QS_{H}_ATR"],
            QW_ATR=r[f"QW_{H}_ATR"],
            action=r[f"action_{H}"], edge=r[f"edge_{H}"],
            edge_ATR=r[f"edge_{H}_ATR"],
            exit_bars=r[f"exit_bars_{H}"], holding_bars=r[f"holding_bars_{H}"],
            realized_move=r[f"realized_move_{H}"],
            MFE=r[f"MFE_{H}"], MAE=r[f"MAE_{H}"],
            label_available_time=str(r[f"label_available_time_{H}"]),
            terminal_reason=r[f"oracle_terminal_reason_{H}"])
    return d


# ===========================================================================
# runner
# ===========================================================================
def _sha256(path: Path) -> str:
    hsh = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hsh.update(chunk)
    return hsh.hexdigest()


def _exclusion_accounting(bars, core):
    """Mutually-exclusive per-symbol exclusion counts (Fix G)."""
    n = bars["n"]
    disc = bars["disc"]
    valid = core["valid"]
    has_roundtrip = core["has_roundtrip"]
    disc_excl = tail_excl = no_rt_excl = 0
    for i in range(n):
        is_disc = (i + 1 < n) and bool(disc[i + 1])
        is_tail = (i + 2 >= n)
        if is_disc:
            disc_excl += 1
        elif is_tail:
            tail_excl += 1
        elif not has_roundtrip[i]:
            no_rt_excl += 1
    return dict(input_5m_rows=int(n),
                excluded_tail=int(tail_excl),
                excluded_discontinuity_before_entry=int(disc_excl),
                excluded_no_valid_roundtrip=int(no_rt_excl))


def run_all(symbols=SYMBOLS, tail_bars=None):
    t0 = time.perf_counter()
    mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    all_rows = []
    excl = {}
    bars_by_sym = {}
    per_sym = {}
    for sym in symbols:
        bt = time.perf_counter()
        bars = build_bars(sym)
        if tail_bars is not None:
            keep = min(tail_bars, bars["n"])
            for k in ("o", "h", "l", "c", "disc", "seg", "atr5"):
                bars[k] = bars[k][-keep:]
            bars["t"] = bars["t"][-keep:]
            bars["decision_time"] = bars["decision_time"][-keep:]
            bars["n"] = keep
        bars_by_sym[sym] = bars                       # Fix H: single load
        core = oracle_core(bars)
        df = build_rows(bars, core)
        excl[sym] = _exclusion_accounting(bars, core)
        per_sym[sym] = dict(n=int(len(df)),
                            sec=round(time.perf_counter() - bt, 2))
        all_rows.append(df)
        print(f"[{sym}] rows={len(df)} tail={excl[sym]['excluded_tail']} "
              f"disc={excl[sym]['excluded_discontinuity_before_entry']} "
              f"nort={excl[sym]['excluded_no_valid_roundtrip']} "
              f"({time.perf_counter()-bt:.1f}s)")
    rows = pd.concat(all_rows, ignore_index=True)
    mem1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summ = summarize(rows, excl)
    summ["runtime_sec"] = round(time.perf_counter() - t0, 2)
    summ["peak_rss_mb"] = round(mem1 / (1024 * 1024), 1)
    summ["per_symbol_timing"] = per_sym
    summ["horizons"] = list(HORIZONS)
    summ["n_symbols"] = len(symbols)

    rows_path = OUT / "robust_trade_oracle_rows.parquet"
    rows.to_parquet(rows_path, index=False)
    rows_sha = _sha256(rows_path)

    # samples reuse loaded bars (no second reload)
    samples = pick_samples(rows, bars_by_sym)

    cost = dict(
        canonical_table_found=False,
        scanned=["tick_size", "contract_multiplier", "commission",
                 "exchange_fee", "broker_fee", "slippage"],
        NET_PNL="UNAVAILABLE_COST_METADATA",
        rule="禁止凭记忆填写手续费/滑点；只报 GROSS price-point PnL (ATR-normalized 供跨品种比较)",
        break_even_note=("utility v1 = GrossPnL - Cost, Cost=0; "
                         "net PnL requires per-symbol tick value + fee table "
                         "which is not present in the project. Do NOT claim "
                         "net profitable."),
    )

    protocol = dict(
        experiment="Robust 5m Trade Oracle DP v1.1",
        task_id="FUTURE-ORACLE-R1.1-CORRECTNESS-HARDENING",
        version="1.1",
        horizons=list(HORIZONS),
        hmax=HMAX,
        utility="U = GrossPnL - Cost; Cost = 0 (no cost owner)",
        environment_independent=True,
        oracle_semantics=("at most one round-trip trade per horizon; "
                          "Long/Short/Wait; Wait owns future option value"),
        forbidden_inputs=["DTP", "SR", "Liquidity", "4H", "1H", "15m",
                          "5m indicator state"],
        allowed_inputs=["5m OHLC path", "execution constraints",
                        "discontinuity flags", "causal ATR5 (units only)"],
        fills="next valid 5m bar open (entry and exit); no same-bar hindsight",
        single_position=True,
        no_discontinuity_cross=True,
        fix_A="Q_W(t,h)=0 when disc[t+1]=1; whole horizon banned from crossing",
        fix_B="decision_time = bar_start_time + 5min (close)",
        fix_C="segment = cumsum(disc) (canonical owner)",
        fix_D="tie via numeric eps; n_best==1 -> unique else Tie; stable needs "
              "all 3 horizons SAME UNIQUE action",
        fix_E="label_available_time per H (reason HORIZON_END/DISCONTINUITY/"
               "DATA_END); not the optimal exit time",
        fix_F="raw price points retained + ATR5-normalized (units only, "
              "does not enter argmax/exit/DP)",
        fix_G="exclusion accounting input=output+tail+disc+noroundtrip",
        fix_H="per-symbol raw 5m loaded once; samples reuse it",
        reused=["load_raw_5m", "discontinuity_flags", "compute_atr5"],
        not_reused=["fixed stop", "liquidity target of "
                    "run_fixed_execution_baseline_v1.py"],
        shared_compute="one load + one segment build + one QL/QS precompute "
                       "+ one backward flat DP; H=6/12/24 read from tables",
    )
    audit = dict(
        experiment="Robust 5m Trade Oracle DP v1.1",
        task_id="FUTURE-ORACLE-R1.1-CORRECTNESS-HARDENING",
        rows_sha256=rows_sha,
        rows_path=str(rows_path),
        n_decisions=int(len(rows)),
        summary=summ,
        cost=cost,
        samples=samples,
    )

    json.dump(summ, open(OUT / "robust_trade_oracle_summary.json", "w"),
              indent=2, default=str)
    json.dump(protocol, open(OUT / "ROBUST_TRADE_ORACLE_PROTOCOL.json", "w"),
              indent=2, default=str)
    json.dump(audit, open(OUT / "ROBUST_TRADE_ORACLE_AUDIT.json", "w"),
              indent=2, default=str)
    _write_report(rows, summ, protocol, audit, samples)
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s rows={len(rows)} "
          f"sha={rows_sha[:12]} -> {OUT}")
    return dict(rows=rows, summary=summ, audit=audit, protocol=protocol,
                rows_sha=rows_sha)


def _write_report(rows, summ, protocol, audit, samples):
    def tbl_block(title, rows_):
        head = "| " + " | ".join(rows_[0].keys()) + " |"
        sep = "|" + "|".join(["---"] * len(rows_[0])) + "|"
        body = "\n".join("| " + " | ".join(str(v) for v in r.values()) + " |"
                         for r in rows_)
        return f"### {title}\n\n{head}\n{sep}\n{body}\n"

    md = f"""# Robust 5m Trade Oracle DP v1.1 (Correctness Hardening)

> **Status**: `PROVISIONAL_PENDING_USER_AUDIT`. No model yet (R1/R1.1 = label
> stability & correctness only). Utility = gross price-point PnL, cost = 0.
> Built on R1 after user review found 4 substantive label-semantics issues.

**Task**: `FUTURE-ORACLE-R1.1-CORRECTNESS-HARDENING`
**Horizons**: {list(HORIZONS)} (30/60/120 min)
**Environment-independent**: only 5m OHLC + discontinuity + causal ATR5 (units).
No DTP / SR / Liquidity / HTF / indicators.

## 0. Frozen semantics

- Decision at 5m close `t`; entry fill = open of next valid bar `e=t+1`.
- Exit fill = open of bar `e+h`; holding `h` in 1..H.
- **At most one round-trip trade per Oracle horizon** (exit does not re-enter
  in the same horizon). Oracle answers "is THIS bar a good trade opportunity?".
- Single position, fixed 1 unit, no add, no hedge.
- **Whole horizon banned from crossing a discontinuity** (Fix A).
- `V_flat[t][h] = max(Q_L, Q_S, Q_W(t,h))`; `Q_W=0` if `disc[t+1]=1`;
  `Q_W=V_flat[t+1][h-1]` otherwise.
- Ties use numeric eps; only n_best==1 yields a unique action (else Tie).
  stable_action needs all 3 horizons = SAME UNIQUE.
- Shared compute: one load + one segment + one QL/QS precompute + one backward
  flat DP; H=6/12/24 read from tables.

## 1. Sample counts

| metric | value |
|---|---:|
| n_decisions | {summ['n_decisions']} |
| runtime_sec | {summ['runtime_sec']} |
| peak_rss_mb | {summ['peak_rss_mb']} |
| n_symbols | {summ['n_symbols']} |

## 2. Stable action distribution

| class | pct |
|---|---:|
| Long | {summ['stable_Long_pct']} |
| Short | {summ['stable_Short_pct']} |
| Wait | {summ['stable_Wait_pct']} |
| Ambiguous | {summ['stable_Ambiguous_pct']} |

Agreement: 3/3 = {summ['agreement_3of3_pct']}% ; 2/3+ = {summ['agreement_2of3_pct']}%
Tie rows = {summ['tie_rows']}; stable_trade_zero_edge_count = {summ['stable_trade_zero_edge_count']}

## 3. By symbol (with exclusion accounting)

{tbl_block('symbol', summ['by_symbol'])}

## 4. By time block (calendar month; "TB block" mapped to month)

{tbl_block('time_block', summ['by_time_block'])}

## 5. Distribution quantiles (H=24, ATR-normalized primary)

- edge_ATR: {summ['edge_ATR']}
- holding (trade actions): {summ['holding']}
- MFE_ATR: {summ['mfe_ATR']}
- MAE_ATR: {summ['mae_ATR']}

Raw (per-symbol scale, not cross-comparable): edge_raw {summ['edge_raw']},
MFE_raw {summ['mfe_raw']}, MAE_raw {summ['mae_raw']}.

## 6. Stable-class edge / holding (ATR-normalized)

| class | median_edge_ATR | median_holding |
|---|---:|---:|
| Long | {summ['stable_Long_median_edge_ATR']} | {summ['stable_Long_median_holding']} |
| Short | {summ['stable_Short_median_edge_ATR']} | {summ['stable_Short_median_holding']} |
| Wait | {summ['stable_Wait_median_edge_ATR']} | {summ['stable_Wait_median_holding']} |

## 7. Cross-horizon transition (Long/Short/Wait/Tie)

- 6->12: {summ['transition_6_12']['counts']}
- 12->24: {summ['transition_12_24']['counts']}
- 6->24: {summ['transition_6_24']['counts']}

## 8. Exclusion accounting (input = output + excluded, mutually exclusive)

{tbl_block('symbol', [summ['exclusion_total']])}

## 9. Human oracle samples (5)

```json
{json.dumps(samples, indent=2, default=str)}
```

## 10. Cost metadata

```json
{json.dumps(audit['cost'], indent=2, default=str)}
```

## 11. Next

User audits R1.1 fixes (A-H). Then the real Oracle robustness experiment:
scan `risk penalty x time penalty x friction hurdle` over the SAME precomputed
future paths and measure how many Long/Short/Wait labels stay stable. Only
after that is the Oracle a credible label for Environment -> Oracle mapping.
"""
    open(OUT / "robust_trade_oracle_report.md", "w",
         encoding="utf-8-sig").write(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--tail-bars", type=int, default=None,
                    help="only use last N bars per symbol (debug / T1-like)")
    args = ap.parse_args()
    run_all(symbols=args.symbols, tail_bars=args.tail_bars)


if __name__ == "__main__":
    main()
