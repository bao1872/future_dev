"""Robust 5m Trade Oracle via Finite-Horizon Dynamic Programming (v1.0).

FUTURE-ORACLE-R1-ROBUST-5M-DP

Purpose
-------
Build an **environment-independent** trade oracle: for every 5m decision close
`t`, solve a finite-horizon DP that returns the optimal action

    A*(t, H) in {Long, Short, Wait}

and its value (in gross price points, cost = 0), for horizons H = 6 / 12 / 24
(30 / 60 / 120 minutes). The oracle uses **only 5m OHLC path + execution
constraints**. It is forbidden from using DTP / SR / Liquidity / 15m / 1h / 4h
/ any indicator state. See AGENTS.md and the task spec.

Execution contract (frozen)
---------------------------
- Decision at 5m close of bar `t`.
- Entry fills at the **open of the next valid 5m bar** `e = t+1`
  (next-open, never same-bar hindsight).
- Exit decision at close of bar `e+h-1`, fill at open of bar `e+h`
  (consistent next-open convention). Holding length `h` in 1..H.
- Single position, fixed 1 unit, no add, no simultaneous long/short.
- **No crossing a discontinuity**: the held window's exit bar must not be a
  discontinuity bar; the walk stops at the first discontinuity after entry.
- Horizon `H` caps the holding length; paths are also capped by data end.
- Utility v1 = GrossPnL - Cost, with Cost = 0 (no cost owner in project).
  MAE / MFE / holding bars are recorded as outcomes, NOT penalised.

DP semantics (the part the reviewer audits)
-------------------------------------------
Flat value (option value of being flat at t with h steps left):

    V_flat[t][h] = max( Q_L(t,h), Q_S(t,h), V_flat[t+1][h-1] )

where
    Q_L(t,h) = max_{1<=k<=h} (open[e+k] - open[e])           [no-disc span]
    Q_S(t,h) = max_{1<=k<=h} (open[e]   - open[e+k])
    Q_W(t,h) = V_flat[t+1][h-1]        (value of waiting = keeping option)

`WAIT` therefore owns genuine future option value: it is not forced to 0.
A*(t,h) = argmax over {Long, Short, Wait}; edge = best - second.

All three horizons share ONE data load + ONE segment build + ONE forward
precompute of QL/QS + ONE backward flat DP. Nothing is recomputed per H.

Reused (narrow helpers only, NO second 5m loader):
    research.export_ob_trigger_execution_v21.load_raw_5m
    research.phase1_tradability.phase1_contract_v1.discontinuity_flags
Forbidden to reuse: the fixed stop / liquidity target of
run_fixed_execution_baseline_v1.py.

Outputs (artifacts/robust_trade_oracle_dp_v1/):
    robust_trade_oracle_rows.parquet   row-level label table (downstream-ready)
    robust_trade_oracle_summary.json   T2 summary stats
    ROBUST_TRADE_ORACLE_PROTOCOL.json   frozen contract
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
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags

HMAX = 24                 # max horizon scanned; output uses 6/12/24
HORIZONS = (6, 12, 24)
SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]
NEG = -np.inf
OUT = Path("artifacts/robust_trade_oracle_dp_v1")
OUT.mkdir(parents=True, exist_ok=True)


# ===========================================================================
# data owner
# ===========================================================================
def build_bars(sym: str) -> dict:
    """Raw 5m owner (single source). Returns numpy-backed bar dict."""
    raw = _raw.load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    o = raw["open"].to_numpy(float)
    h = raw["high"].to_numpy(float)
    l = raw["low"].to_numpy(float)
    c = raw["close"].to_numpy(float)
    t = pd.to_datetime(raw["bar_start_time"]).to_numpy()
    disc = np.asarray(discontinuity_flags(sym), bool)
    # segment id: cumulative discontinuity count up to and including bar i
    seg = np.concatenate([[0], np.cumsum(disc)]).astype(int)[:-1]
    return dict(symbol=sym, o=o, h=h, l=l, c=c, t=t, disc=disc, seg=seg,
                n=len(raw))


# ===========================================================================
# core DP
# ===========================================================================
def oracle_core(bars: dict, horizons=HORIZONS, hmax=HMAX) -> dict:
    """Shared one-pass computation of QL/QS flat-DP tables.

    Returns arrays (n x (hmax+1)) plus validity and entry index.
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
        # first discontinuity bar index > e (the held window may not cross it)
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
        # Columns beyond the truncation (data end / discontinuity) keep the
        # last running max: QL(t,H)/QS(t,H) are monotone in H.
        for hh in range(1, hmax + 1):
            if not np.isfinite(ql[t, hh]):
                ql[t, hh] = cur_l
                qs[t, hh] = cur_s
                best_h_l[t, hh] = bh_l
                best_h_s[t, hh] = bh_s

    # backward flat DP
    Vflat = np.zeros((n + 1, hmax + 1), dtype=float)   # Vflat[n][*] = 0
    bestA = np.empty((n, hmax + 1), dtype=object)
    for t in range(n - 1, -1, -1):
        qlt = ql[t] if valid[t] else None
        qst = qs[t] if valid[t] else None
        for hh in range(1, hmax + 1):
            wait_val = Vflat[t + 1, hh - 1]
            a_l = qlt[hh] if qlt is not None else NEG
            a_s = qst[hh] if qst is not None else NEG
            # best of three actions
            if a_l >= a_s and a_l >= wait_val:
                b = "Long"
            elif a_s >= a_l and a_s >= wait_val:
                b = "Short"
            else:
                b = "Wait"
            Vflat[t, hh] = (a_l if b == "Long" else a_s if b == "Short"
                            else wait_val)
            bestA[t, hh] = b

    return dict(ql=ql, qs=qs, Vflat=Vflat, bestA=bestA, valid=valid,
                best_h_l=best_h_l, best_h_s=best_h_s, n=n,
                horizons=tuple(horizons))


# ===========================================================================
# row-level label table
# ===========================================================================
def build_rows(bars: dict, core: dict) -> pd.DataFrame:
    o, h, l, disc, tt, seg = (bars["o"], bars["h"], bars["l"], bars["disc"],
                               bars["t"], bars["seg"])
    n = bars["n"]
    valid = core["valid"]
    ql, qs, Vflat, bestA = core["ql"], core["qs"], core["Vflat"], core["bestA"]
    best_h_l, best_h_s = core["best_h_l"], core["best_h_s"]
    horizons = core["horizons"]

    rows = []
    idx = np.flatnonzero(valid)
    for ti in idx:
        t = int(ti)
        e = t + 1
        rec = dict(
            symbol=bars["symbol"],
            decision_time=pd.Timestamp(tt[t]),
            decision_bar_index=int(t),
            entry_bar_index=int(e),
            segment=int(seg[t]),
            entry_valid=True,
        )
        actions = []
        for H in horizons:
            qlt = ql[t, H]
            qst = qs[t, H]
            qwt = Vflat[t + 1, H - 1]
            a_l = qlt if np.isfinite(qlt) else NEG
            a_s = qst if np.isfinite(qst) else NEG
            # best + second
            vals = {"Long": a_l, "Short": a_s, "Wait": qwt}
            order = sorted(vals.items(), key=lambda kv: kv[1], reverse=True)
            best_a, best_v = order[0]
            second_v = order[1][1]
            # edge is undefined when the runner-up action is impossible (-inf)
            edge = (best_v - second_v) if np.isfinite(second_v) else None
            rec[f"QL_{H}"] = (None if not np.isfinite(qlt) else float(qlt))
            rec[f"QS_{H}"] = (None if not np.isfinite(qst) else float(qst))
            rec[f"QW_{H}"] = float(qwt)
            rec[f"action_{H}"] = best_a
            rec[f"edge_{H}"] = (None if edge is None else float(edge))
            if best_a == "Long":
                bh = int(best_h_l[t, H])
                xb = e + bh
                rec[f"exit_bars_{H}"] = int(xb)
                rec[f"holding_bars_{H}"] = bh
                rec[f"realized_move_{H}"] = float(o[xb] - o[e])
                win = max(float(h[e:xb].max()), float(o[xb])) - float(o[e])
                adve = float(o[e]) - min(float(l[e:xb].min()), float(o[xb]))
                rec[f"MFE_{H}"] = float(win)
                rec[f"MAE_{H}"] = float(adve)
            elif best_a == "Short":
                bh = int(best_h_s[t, H])
                xb = e + bh
                rec[f"exit_bars_{H}"] = int(xb)
                rec[f"holding_bars_{H}"] = bh
                rec[f"realized_move_{H}"] = float(o[e] - o[xb])
                win = float(o[e]) - min(float(l[e:xb].min()), float(o[xb]))
                adve = max(float(h[e:xb].max()), float(o[xb])) - float(o[e])
                rec[f"MFE_{H}"] = float(win)
                rec[f"MAE_{H}"] = float(adve)
            else:
                rec[f"exit_bars_{H}"] = None
                rec[f"holding_bars_{H}"] = None
                rec[f"realized_move_{H}"] = None
                rec[f"MFE_{H}"] = None
                rec[f"MAE_{H}"] = None
            actions.append(best_a)
        # cross-horizon agreement
        c = Counter(actions)
        top, cnt = c.most_common(1)[0]
        rec["agreement"] = round(cnt / len(horizons), 4)
        rec["stable_action"] = top if cnt == len(horizons) else "Ambiguous"
        rec["_agree23"] = int(cnt >= 2)   # 2/3 or better
        rows.append(rec)

    df = pd.DataFrame(rows)
    # stable columns first
    front = ["symbol", "decision_time", "decision_bar_index", "entry_bar_index",
             "segment", "entry_valid", "agreement", "stable_action", "_agree23"]
    rest = [c for c in df.columns if c not in front]
    df = df[front + rest]
    return df


def compute_oracle(bars: dict, horizons=HORIZONS, hmax=HMAX):
    core = oracle_core(bars, horizons, hmax)
    df = build_rows(bars, core)
    return df, core


# ===========================================================================
# summary
# ===========================================================================
def summarize(df: pd.DataFrame, horizons=HORIZONS) -> dict:
    n = len(df)
    s = dict(n_decisions=int(n))
    # stable action distribution
    sa = df["stable_action"].value_counts()
    for a in ("Long", "Short", "Wait", "Ambiguous"):
        s[f"stable_{a}_pct"] = round(float((sa.get(a, 0) / n)) * 100, 3) if n \
            else 0.0
    # agreement
    s["agreement_3of3_pct"] = round(float((df["agreement"] == 1.0).mean()) * 100,
                                    3) if n else 0.0
    s["agreement_2of3_pct"] = round(float(df["_agree23"].mean()) * 100, 3) if n \
        else 0.0
    s["agreement_mean"] = round(float(df["agreement"].mean()), 4) if n else 0.0
    # per-horizon action distribution
    for H in horizons:
        ac = df[f"action_{H}"].value_counts()
        for a in ("Long", "Short", "Wait"):
            s[f"action_{H}_{a}_pct"] = round(
                float(ac.get(a, 0) / n) * 100, 3) if n else 0.0
    # by symbol
    by_sym = []
    for sym, g in df.groupby("symbol"):
        by_sym.append(dict(
            symbol=sym, n=int(len(g)),
            stable_Long_pct=round(float((g["stable_action"] == "Long").mean())
                                  * 100, 3),
            stable_Short_pct=round(float((g["stable_action"] == "Short").mean())
                                   * 100, 3),
            stable_Wait_pct=round(float((g["stable_action"] == "Wait").mean())
                                  * 100, 3),
            stable_Ambiguous_pct=round(
                float((g["stable_action"] == "Ambiguous").mean()) * 100, 3),
            agreement_3of3_pct=round(float((g["agreement"] == 1.0).mean())
                                     * 100, 3),
        ))
    s["by_symbol"] = by_sym
    # by calendar-month time block
    blk = pd.to_datetime(df["decision_time"]).dt.strftime("%Y-%m")
    by_month = []
    for m, g in df.groupby(blk):
        by_month.append(dict(
            time_block=m, n=int(len(g)),
            stable_Long_pct=round(float((g["stable_action"] == "Long").mean())
                                  * 100, 3),
            stable_Short_pct=round(float((g["stable_action"] == "Short").mean())
                                   * 100, 3),
            stable_Wait_pct=round(float((g["stable_action"] == "Wait").mean())
                                  * 100, 3),
            stable_Ambiguous_pct=round(
                float((g["stable_action"] == "Ambiguous").mean()) * 100, 3),
        ))
    s["by_time_block"] = by_month
    # edge / holding / MFE / MAE distributions (over trade actions)
    trade = df[df["stable_action"].isin(["Long", "Short"])].copy()
    s["edge"] = _quant(df, "edge_24")
    s["holding"] = _quant(trade, "holding_bars_24")
    s["mfe"] = _quant(trade, "MFE_24")
    s["mae"] = _quant(trade, "MAE_24")
    # stable-class edge stats
    for a in ("Long", "Short", "Wait"):
        sub = df[df["stable_action"] == a]
        s[f"stable_{a}_median_edge"] = _med(sub, "edge_24")
        s[f"stable_{a}_min_edge"] = _min(sub, "edge_24")
        s[f"stable_{a}_median_holding"] = _med(sub, "holding_bars_24")
    return s


def _quant(df, col):
    if df is None or len(df) == 0 or col not in df:
        return None
    v = pd.to_numeric(df[col], errors="coerce").replace(
        [np.inf, -np.inf], np.nan).dropna()
    if len(v) == 0:
        return None
    return dict(n=int(len(v)),
                p10=round(float(v.quantile(.1)), 4),
                p25=round(float(v.quantile(.25)), 4),
                p50=round(float(v.quantile(.5)), 4),
                p75=round(float(v.quantile(.75)), 4),
                p90=round(float(v.quantile(.9)), 4),
                mean=round(float(v.mean()), 4))


def _med(df, col):
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return round(float(v.median()), 4) if len(v) else None


def _min(df, col):
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    return round(float(v.min()), 4) if len(v) else None


# ===========================================================================
# human oracle samples (5)
# ===========================================================================
def pick_samples(df: pd.DataFrame, bars_by_sym: dict) -> list:
    out = []
    # 1 stable Long, high edge
    longs = df[df["stable_action"] == "Long"]
    if len(longs):
        r = longs.loc[longs["edge_24"].idxmax()]
        out.append(("stable_Long_high_edge", _sample(r)))
    # 2 stable Short, high edge
    shorts = df[df["stable_action"] == "Short"]
    if len(shorts):
        r = shorts.loc[shorts["edge_24"].idxmax()]
        out.append(("stable_Short_high_edge", _sample(r)))
    # 3 stable Wait
    waits = df[df["stable_action"] == "Wait"]
    if len(waits):
        r = waits.loc[waits["QW_24"].idxmax()]
        out.append(("stable_Wait", _sample(r)))
    # 4 Ambiguous
    amb = df[df["stable_action"] == "Ambiguous"]
    if len(amb):
        r = amb.iloc[len(amb) // 2]
        out.append(("ambiguous", _sample(r)))
    # 5 single high-edge example with full Q trace
    if len(df):
        r = df.loc[df["edge_24"].idxmax()]
        out.append(("max_edge_any", _sample(r)))
    return out


def _sample(r) -> dict:
    d = dict(symbol=r["symbol"], decision_time=str(r["decision_time"]),
             decision_bar_index=int(r["decision_bar_index"]),
             segment=int(r["segment"]), agreement=float(r["agreement"]),
             stable_action=r["stable_action"])
    for H in HORIZONS:
        d[f"H{H}"] = dict(
            QL=r[f"QL_{H}"], QS=r[f"QS_{H}"], QW=r[f"QW_{H}"],
            action=r[f"action_{H}"], edge=r[f"edge_{H}"],
            exit_bars=r[f"exit_bars_{H}"],
            realized_move=r[f"realized_move_{H}"],
            MFE=r[f"MFE_{H}"], MAE=r[f"MAE_{H}"])
    return d


# ===========================================================================
# runner
# ===========================================================================
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_all(symbols=SYMBOLS, tail_bars=None):
    t0 = time.perf_counter()
    mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    all_rows = []
    per_sym = {}
    for sym in symbols:
        bt = time.perf_counter()
        bars = build_bars(sym)
        if tail_bars is not None:
            keep = min(tail_bars, bars["n"])
            for k in ("o", "h", "l", "c", "disc", "seg"):
                bars[k] = bars[k][-keep:]
            bars["t"] = bars["t"][-keep:]
            bars["n"] = keep
        df, _ = compute_oracle(bars)
        per_sym[sym] = dict(n=int(len(df)),
                            sec=round(time.perf_counter() - bt, 2))
        all_rows.append(df)
        print(f"[{sym}] rows={len(df)} "
              f"({time.perf_counter()-bt:.1f}s)")
    rows = pd.concat(all_rows, ignore_index=True)
    mem1 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    summ = summarize(rows)
    summ["runtime_sec"] = round(time.perf_counter() - t0, 2)
    summ["peak_rss_mb"] = round(mem1 / (1024 * 1024), 1)
    summ["per_symbol_timing"] = per_sym
    summ["horizons"] = list(HORIZONS)
    summ["n_symbols"] = len(symbols)

    # write rows
    rows_path = OUT / "robust_trade_oracle_rows.parquet"
    rows.to_parquet(rows_path, index=False)
    rows_sha = _sha256(rows_path)

    # samples
    bars_by_sym = {s: build_bars(s) for s in symbols}
    # build a per-symbol rows lookup for sampling context (reuse all_rows)
    samples = pick_samples(rows, bars_by_sym)

    # cost metadata (mirror canonical audit: no cost owner in project)
    cost = dict(
        canonical_table_found=False,
        scanned=["tick_size", "contract_multiplier", "commission",
                 "exchange_fee", "broker_fee", "slippage"],
        NET_PNL="UNAVAILABLE_COST_METADATA",
        rule="禁止凭记忆填写手续费/滑点；只报 GROSS price-point PnL",
        break_even_note=("utility v1 = GrossPnL - Cost, Cost=0; "
                         "net PnL requires per-symbol tick value + fee table "
                         "which is not present in the project. Do NOT claim "
                         "net profitable."),
    )

    protocol = dict(
        experiment="Robust 5m Trade Oracle DP v1.0",
        task_id="FUTURE-ORACLE-R1-ROBUST-5M-DP",
        horizons=list(HORIZONS),
        hmax=HMAX,
        utility="U = GrossPnL - Cost; Cost = 0 (no cost owner)",
        environment_independent=True,
        forbidden_inputs=["DTP", "SR", "Liquidity", "4H", "1H", "15m",
                          "5m indicator state"],
        allowed_inputs=["5m OHLC path", "execution constraints",
                        "discontinuity flags"],
        fill="next valid 5m bar open (entry and exit); no same-bar hindsight",
        single_position=True,
        no_discontinuity_cross=True,
        wait_has_option_value=True,
        reused=["load_raw_5m", "discontinuity_flags"],
        not_reused=["fixed stop", "liquidity target of "
                    "run_fixed_execution_baseline_v1.py"],
        shared_compute="one load + one segment build + one QL/QS precompute "
                       "+ one backward flat DP; H=6/12/24 read from tables",
    )
    audit = dict(
        experiment="Robust 5m Trade Oracle DP v1.0",
        task_id="FUTURE-ORACLE-R1-ROBUST-5M-DP",
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

    md = f"""# Robust 5m Trade Oracle DP v1.0

> **Status**: `PROVISIONAL_PENDING_USER_AUDIT`. No model yet (R1 = label
> stability only). Utility = gross price-point PnL, cost = 0.

**Task**: `FUTURE-ORACLE-R1-ROBUST-5M-DP`
**Horizons**: {list(HORIZONS)} (30/60/120 min)
**Environment-independent**: uses only 5m OHLC + discontinuity flags.
No DTP / SR / Liquidity / HTF / indicators.

## 0. Contract (frozen)

- Decision at 5m close `t`; entry fill = open of next valid bar `e=t+1`.
- Exit fill = open of bar `e+h` (next-open convention); holding `h` in 1..H.
- Single position, fixed 1 unit, no add, no same-symbol hedging.
- No crossing discontinuity; horizon caps holding; data end caps path.
- `V_flat[t][h] = max(Q_L, Q_S, V_flat[t+1][h-1])`; `Q_W = V_flat[t+1][h-1]`.
- Shared compute: one load + one segment build + one QL/QS precompute + one
  backward flat DP. H=6/12/24 read from tables (never re-run data).

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

## 3. By symbol

{tbl_block('symbol', summ['by_symbol'])}

## 4. By time block (calendar month; "TB block" mapped to month)

{tbl_block('time_block', summ['by_time_block'])}

## 5. Distribution quantiles (H=24)

- edge: {summ['edge']}
- holding (trade actions): {summ['holding']}
- MFE (trade actions): {summ['mfe']}
- MAE (trade actions): {summ['mae']}

## 6. Stable-class edge / holding

| class | median_edge | min_edge | median_holding |
|---|---:|---:|---:|
| Long | {summ['stable_Long_median_edge']} | {summ['stable_Long_min_edge']} | {summ['stable_Long_median_holding']} |
| Short | {summ['stable_Short_median_edge']} | {summ['stable_Short_min_edge']} | {summ['stable_Short_median_holding']} |
| Wait | {summ['stable_Wait_median_edge']} | {summ['stable_Wait_min_edge']} | {summ['stable_Wait_median_holding']} |

## 7. Human oracle samples (5)

```json
{json.dumps(samples, indent=2, default=str)}
```

## 8. Cost metadata

```json
{json.dumps(audit['cost'], indent=2, default=str)}
```

## 9. Next

User audits DP Bellman, Wait option value, next-open execution, discontinuity
boundary, horizon off-by-one, MFE/MAE interval, H=6/12/24 shared compute, and
row-level artifact completeness. Then decides Oracle R2 (risk/time penalty,
longer horizon, 2/3 consensus) or proceeds to Environment -> Oracle mapping.
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
