"""Oracle Constraint Robustness experiment (v1).

FUTURE-ORACLE-R2.1-CORRECTNESS-CLOSURE

Purpose
-------
Hold R1.1's oracle structure FIXED (5m decision clock, H = 6/12/24, next-open
execution, at most one round-trip, discontinuity contract, Long/Short/Wait/Tie
semantics). Only change the oracle's RISK preference, TIME preference and TRADE
FRICTION hurdle, and measure how stable the action labels are.

This is a SENSITIVITY / ROBUSTNESS surface, NOT parameter tuning. The code never
selects a "best lambda". It only reports facts.

Critical technical rule (user-locked)
-------------------------------------
The DP recursion is performed ENTIRELY in price points. Risk / time / friction
parameters are converted to price points and added to the per-trade utility:

    U(t,k,d; theta) = R(t,k,d) - lambda_R*MAE - c*ATR_t - lambda_T*ATR_t*k

where ATR_t is the ATR at the DECISION bar t (a constant for that decision).
We NEVER normalize V_{t+1} by ATR_{t+1} and feed it back into the Bellman
equation, because ATR_t != ATR_{t+1}. Only at output time do we divide by ATR_t
to enable cross-symbol comparison (Q^ATR = Q / ATR_t).

Utility (theta = (lambda_R, lambda_T, c))
-----------------------------------------
Immediately trade:
    U(t,k,d) = d*(O_{e+k} - O_e) - lambda_R*MAE(t,k,d)
               - c*ATR_t - lambda_T*ATR_t*k
Wait:
    Q_W(t,h) = 0                                  if disc[t+1]
             = max(0, V_flat(t+1,h-1) - lambda_T*ATR_t)   otherwise
Bellman:
    V_flat(t,h) = max( Q_L(t,h), Q_S(t,h), Q_W(t,h) )

MAE (price points):
    MAE^L = O_e - min(L_e .. L_{e+k-1}, O_{e+k})
    MAE^S = max(H_e .. H_{e+k-1}, O_{e+k}) - O_e

Frozen (do NOT change): H in {6,12,24}, decision_time, segment, entry, exit,
discontinuity, tie semantics, label availability, ATR5 owner. No DTP/SR/
Liquidity/HTF/GBDT/MoE/PGM, no real-but-unverified fee data.

Reused narrow owners: build_bars + oracle_core + _classify from
build_robust_trade_oracle_dp_v1 (R1.1 baseline), discontinuity_flags,
compute_atr5.
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# narrow reusable owners (same source as R1.1 -> guarantees data-level parity)
from research.liquidity_oracle_atlas.build_robust_trade_oracle_dp_v1 import (  # noqa: E402
    build_bars,
    oracle_core as r1_oracle_core,
    _classify,
    HORIZONS,
    HMAX,
)
from research.phase1_tradability.phase1_contract_v1 import (  # noqa: E402
    discontinuity_flags,
    compute_atr5,
)

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU", "MA",
           "TA", "M", "P", "CF"]
NEG = -np.inf
EPS_TIE = 1e-6
OUT = Path("artifacts/oracle_constraint_robustness_v1")
OUT.mkdir(parents=True, exist_ok=True)

# action encoding (int8 compact matrix)
A_LONG, A_SHORT, A_WAIT, A_TIE = 1, -1, 0, 2

# Core grid: 3 x 3 x 3 = 27
LAM_R = [0.0, 0.25, 0.50]
LAM_T = [0.0, 0.005, 0.010]
C_FR = [0.0, 0.025, 0.050]
CORE = [(r, t, c, "CORE") for r in LAM_R for t in LAM_T for c in C_FR]
# Stress grid: 2^3 corners minus the all-zero baseline already in CORE -> 7
STRESS_POINTS = [
    (1.00, 0.0, 0.0), (1.00, 0.020, 0.0), (1.00, 0.0, 0.100),
    (0.0, 0.020, 0.0), (0.0, 0.020, 0.100), (0.0, 0.0, 0.100),
    (1.00, 0.020, 0.100),
]
STRESS = [(r, t, c, "STRESS") for (r, t, c) in STRESS_POINTS]
THETAS = CORE + STRESS  # 34 unique points; baseline = index 0 = (0,0,0,CORE)
BASELINE_IDX = 0
N_CORE = len(CORE)


def theta_id(r, t, c, grid):
    return f"r{r}_t{t}_c{c}_{grid}".replace(".", "p")


def assert_numeric_parity(a, b, tol=1e-9):
    """Fail-closed Q/V parity check.

    Fix R2.1: a naively computed ``max(abs(A-B))`` returns NaN whenever either
    array contains -inf cells, and ``NaN > tol`` is False -> a finite mismatch
    could pass silently. This helper requires (1) identical finite masks,
    (2) identical -inf masks, (3) allclose on the finite-finite cells only.
    """
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    assert np.array_equal(np.isfinite(a), np.isfinite(b)), "finite mask mismatch"
    assert np.array_equal(np.isneginf(a), np.isneginf(b)), "-inf mask mismatch"
    m = np.isfinite(a) & np.isfinite(b)
    assert np.allclose(a[m], b[m], atol=tol, rtol=0), "finite-values mismatch"


def _parity_mismatch(a, b, tol=1e-9):
    """Return 1 if a/b are not numerically identical (per assert_numeric_parity),
    0 otherwise. Does not raise."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if not (np.array_equal(np.isfinite(a), np.isfinite(b))
            and np.array_equal(np.isneginf(a), np.isneginf(b))):
        return 1
    m = np.isfinite(a) & np.isfinite(b)
    return 0 if np.allclose(a[m], b[m], atol=tol, rtol=0) else 1


# ---------------------------------------------------------------------------
# candidate-path precompute (ONE per symbol, price points only)
# ---------------------------------------------------------------------------
def precompute_candidate_paths(bars, hmax=HMAX):
    """Return [n, hmax] price-point tensors; never recomputed per theta."""
    o = bars["o"]; h = bars["h"]; l = bars["l"]; disc = bars["disc"]
    # Sanitize ATR: warmup/edge bars yield NaN/Inf from compute_atr5. We must
    # NOT carry that into the DP, because 0*NaN = NaN would poison the penalty
    # terms even at theta=0. Non-finite ATR -> 0 (no ATR-scaled penalty at that
    # bar; edge_ATR is reported None where atr<=0 downstream).
    atr = np.nan_to_num(np.asarray(bars["atr5"], float),
                        nan=0.0, posinf=0.0, neginf=0.0)
    n = bars["n"]
    K = np.arange(1, hmax + 1)                      # 1..hmax (holding length)
    t_idx = np.arange(n)[:, None]
    e = t_idx + 1                                   # entry bar index e = t+1
    ex = e + K[None, :]                             # exit-open index = e+k
    col = np.arange(hmax)[None, :]                  # 0..hmax-1
    idx_low = e + col                               # l[e .. e+hmax-1]

    inb_ex = ex <= (n - 1)
    e_safe = np.minimum(e, n - 1)                    # clip entry index for safe use
    o_ex = np.where(inb_ex, o[np.minimum(ex, n - 1)], 0.0)
    idx_low_safe = np.minimum(idx_low, n - 1)        # clip for safe indexing
    inb_low = idx_low < n
    Lmat = np.where(inb_low, l[idx_low_safe], np.inf)
    Hmat = np.where(inb_low, h[idx_low_safe], -np.inf)

    pnl_long = o_ex - o[e_safe]
    pnl_short = o[e_safe] - o_ex
    cummin_L = np.minimum.accumulate(Lmat, axis=1)  # min(L_e .. L_{e+k-1})
    cummax_H = np.maximum.accumulate(Hmat, axis=1)
    mae_long = o[e_safe] - np.minimum(cummin_L, o_ex)
    mae_short = np.maximum(cummax_H, o_ex) - o[e_safe]

    # valid exit: e+k <= n-1 AND no discontinuity in [e, e+k]
    dc = np.concatenate([[0], np.cumsum(disc.astype(np.int64))])
    # clip ex+1 to dc bounds; out-of-range (inb_ex False) handled by valid_exit
    no_disc_span = (dc[np.minimum(ex + 1, n)] - dc[e]) == 0
    valid_exit = inb_ex & no_disc_span

    return dict(o=o, h=h, l=l, disc=disc, atr_t=atr, n=n, K=K,
                pnl_long=pnl_long, pnl_short=pnl_short,
                mae_long=mae_long, mae_short=mae_short,
                valid_exit=valid_exit)


# ---------------------------------------------------------------------------
# core DP at a single theta (price points; vectorized backward)
# ---------------------------------------------------------------------------
def oracle_core_theta(cand, lam_r, lam_t, c, hmax=HMAX):
    n = cand["n"]; atr = cand["atr_t"]; disc = cand["disc"]
    K = cand["K"]
    atr_col = atr[:, None]
    u_l = (cand["pnl_long"] - lam_r * cand["mae_long"]
           - c * atr_col - lam_t * atr_col * K[None, :])
    u_s = (cand["pnl_short"] - lam_r * cand["mae_short"]
           - c * atr_col - lam_t * atr_col * K[None, :])
    u_l = np.where(cand["valid_exit"], u_l, NEG)
    u_s = np.where(cand["valid_exit"], u_s, NEG)

    QLa = np.maximum.accumulate(u_l, axis=1)        # max over k<=h
    QSa = np.maximum.accumulate(u_s, axis=1)
    QL = np.full((n, hmax + 1), NEG); QL[:, 1:] = QLa
    QS = np.full((n, hmax + 1), NEG); QS[:, 1:] = QSa
    # H-specific optimal holding (1-based k), indexed by H in [1..hmax].
    # Fix R2.1: the optimal exit must be argmax over k<=H, NOT the global
    # argmax over all k<=hmax (which previously reused the HMAX optimum for
    # every horizon). hold_l[:, H] is the best exit length within horizon H.
    n = cand["n"]
    hold_l = np.zeros((n, hmax + 1), dtype=np.int8)
    hold_s = np.zeros((n, hmax + 1), dtype=np.int8)
    for h in range(1, hmax + 1):
        valid_l = np.any(cand["valid_exit"][:, :h], axis=1)
        valid_s = np.any(cand["valid_exit"][:, :h], axis=1)
        kl = np.argmax(u_l[:, :h], axis=1) + 1
        ks = np.argmax(u_s[:, :h], axis=1) + 1
        hold_l[:, h] = np.where(valid_l, kl, 0)
        hold_s[:, h] = np.where(valid_s, ks, 0)

    Vflat = np.zeros((n + 1, hmax + 1))
    QW = np.zeros((n, hmax + 1))
    for hh in range(1, hmax + 1):
        qw = np.zeros(n)
        if n - 1 > 0:
            base = Vflat[1:n, hh - 1] - lam_t * atr[:-1]   # t = 0..n-2
            qw_inner = np.maximum(0.0, base)
            nd = ~disc[1:]                                # disc[t+1] for t<=n-2
            qw[:n - 1] = np.where(nd, qw_inner, 0.0)
            # t = n-1 -> qw = 0 (cannot wait past data)
        QW[:, hh] = qw
        Vflat[:n, hh] = np.maximum.reduce([QL[:, hh], QS[:, hh], qw])
    return dict(QL=QL, QS=QS, QW=QW, Vflat=Vflat,
                hold_l=hold_l, hold_s=hold_s, n=n, hmax=hmax)


# ---------------------------------------------------------------------------
# batch action classification (matches R1.1 _classify semantics)
# ---------------------------------------------------------------------------
def classify_batch(a_l, a_s, a_w, eps=EPS_TIE):
    """a_l/a_s/a_w: [n] arrays (NEG for invalid). Returns (action_code, edge)."""
    a_l = np.where(np.isfinite(a_l), a_l, NEG)
    a_s = np.where(np.isfinite(a_s), a_s, NEG)
    a_w = np.where(np.isfinite(a_w), a_w, NEG)
    amax = np.maximum.reduce([a_l, a_s, a_w])
    close_l = np.abs(a_l - amax) <= eps
    close_s = np.abs(a_s - amax) <= eps
    close_w = np.abs(a_w - amax) <= eps
    n_best = close_l.astype(int) + close_s.astype(int) + close_w.astype(int)
    action = np.full(a_l.shape[0], A_TIE, dtype=np.int8)
    l_only = (n_best == 1) & close_l & ~close_s & ~close_w
    s_only = (n_best == 1) & close_s & ~close_l & ~close_w
    w_only = (n_best == 1) & close_w & ~close_l & ~close_s
    action = np.where(l_only, A_LONG, action)
    action = np.where(s_only, A_SHORT, action)
    action = np.where(w_only, A_WAIT, action)
    vstack = np.stack([a_l, a_s, a_w], axis=1)
    sdesc = np.sort(vstack, axis=1)[:, ::-1]
    best = sdesc[:, 0]; second = sdesc[:, 1]
    edge = best - second
    edge = np.where(np.isfinite(best), edge, np.nan)
    return action, edge


def _action_shares(ca, n_core):
    """Explicit Long/Short/Wait/Tie shares + consensus from action codes.

    Fix R2.1: previous code relied on implicit ``ca + 1`` bincount indices and
    mislabelled Wait/Tie. Here every code is compared explicitly so the mapping
    cannot drift.
    """
    sh_L = float(np.mean(ca == A_LONG))
    sh_S = float(np.mean(ca == A_SHORT))
    sh_W = float(np.mean(ca == A_WAIT))
    sh_T = float(np.mean(ca == A_TIE))
    counts = {
        "Long": int((ca == A_LONG).sum()),
        "Short": int((ca == A_SHORT).sum()),
        "Wait": int((ca == A_WAIT).sum()),
        "Tie": int((ca == A_TIE).sum()),
    }
    # Fix R2.1b: a vote tie across multiple actions is NOT a consensus. "Tie"
    # means a Q-value tie WITHIN one Oracle parameter; "Ambiguous" means the
    # multi-theta vote is split. Never pick by action order (would fake Long).
    top_count = max(counts.values())
    winners = [a for a in ("Long", "Short", "Wait", "Tie") if counts[a] == top_count]
    cons = winners[0] if len(winners) == 1 else "Ambiguous"
    cons_rate = top_count / n_core
    return sh_L, sh_S, sh_W, sh_T, cons, cons_rate


def actions_for_theta(core, cand=None, hmax=HMAX):
    """Return per-H actions / edges / best-values / holdings (H order 6,12,24).

    R2.1: holding is H-specific (argmax over k<=H), stored as
    ``core["hold_l"][:, H]``. If ``cand`` is supplied, also returns the
    ATR-normalized edge (``edge_atr``) and the raw price-point edge
    (``edge_points``); otherwise those keys are omitted.
    """
    atr = cand["atr_t"] if cand is not None else None
    out = {}
    for H in HORIZONS:
        a_l = core["QL"][:, H]
        a_s = core["QS"][:, H]
        a_w = core["QW"][:, H]
        act, edge = classify_batch(a_l, a_s, a_w)
        best_val = np.maximum.reduce([a_l, a_s, a_w])
        d = dict(action=act, edge=edge, best=best_val,
                 hold_l=core["hold_l"][:, H], hold_s=core["hold_s"][:, H])
        if atr is not None:
            edge_atr = np.full_like(edge, np.nan, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                np.divide(edge, atr, out=edge_atr, where=atr > 0)
            d["edge_atr"] = edge_atr
            d["edge_points"] = edge
        out[H] = d
    return out


# ---------------------------------------------------------------------------
# independent brute-force reference (R2 math) for differential test
# ---------------------------------------------------------------------------
def _mae_long(o, l, ee, k):
    ex = ee + k
    m = min(float(l[ee:ex].min()), float(o[ex]))
    return float(o[ee]) - m


def _mae_short(o, h, ee, k):
    ex = ee + k
    m = max(float(h[ee:ex].max()), float(o[ex]))
    return m - float(o[ee])


def brute_force_oracle_r2(bars, t, H, lam_r, lam_t, c, eps=EPS_TIE):
    """Independent enumerator: trade now / wait k then one trade / nothing."""
    o, h, l, disc = bars["o"], bars["h"], bars["l"], bars["disc"]
    # Sanitize exactly like the main DP: warmup/edge ATR is NaN/Inf and must
    # become 0 so the friction/time penalties match oracle_core_theta.
    atr = np.nan_to_num(np.asarray(bars["atr5"], float),
                        nan=0.0, posinf=0.0, neginf=0.0)
    n = bars["n"]; e = t + 1
    atr_t = float(atr[t])

    def trade_util(tt, k, d):
        ee = tt + 1; ex = ee + k
        if ex > n - 1:
            return NEG
        if disc[ee:ex + 1].any():          # disc in [ee, ex] (entry or within)
            return NEG
        pnl = (o[ex] - o[ee]) if d == 1 else (o[ee] - o[ex])
        if d == 1:
            mae = _mae_long(o, l, ee, k)
        else:
            mae = _mae_short(o, h, ee, k)
        pen = lam_r * mae + c * float(atr[tt]) + lam_t * float(atr[tt]) * k
        return pnl - pen

    bestL, bestS = NEG, NEG
    for k in range(1, H + 1):
        uL = trade_util(t, k, 1); uS = trade_util(t, k, -1)
        if uL > bestL:
            bestL = uL
        if uS > bestS:
            bestS = uS
    bestW = 0.0  # nothing
    for k in range(1, H):
        if disc[t + 1:t + k + 1].any():    # cross discontinuity while waiting
            continue
        if t + k >= n:
            continue
        wait_pen = lam_t * float(sum(atr[t:t + k]))   # atr[t .. t+k-1]
        for kk in range(1, H - k + 1):
            uL = trade_util(t + k, kk, 1) - wait_pen
            uS = trade_util(t + k, kk, -1) - wait_pen
            if uL > bestW:
                bestW = uL
            if uS > bestW:
                bestW = uS
    vals = {"Long": bestL if np.isfinite(bestL) else NEG,
            "Short": bestS if np.isfinite(bestS) else NEG,
            "Wait": bestW}
    act, _, _ = _classify(vals, eps)
    return dict(QL=bestL, QS=bestS, QW=bestW, action=act)


# ---------------------------------------------------------------------------
# baseline action from R1.1 oracle_core (for parity + as baseline label)
# ---------------------------------------------------------------------------
def r1_baseline_actions(bars):
    core = r1_oracle_core(bars, horizons=HORIZONS, hmax=HMAX)
    n = bars["n"]; disc = bars["disc"]
    out = {}
    for H in HORIZONS:
        acts = np.full(n, A_TIE, dtype=np.int8)
        edges = np.full(n, np.nan)
        for t in range(n):
            qlt = core["ql"][t, H]; qst = core["qs"][t, H]
            qwt = (0.0 if (t + 1 < n and bool(disc[t + 1]))
                   else core["Vflat"][t + 1, H - 1])
            a_l = qlt if np.isfinite(qlt) else NEG
            a_s = qst if np.isfinite(qst) else NEG
            act, _, _ = _classify({"Long": a_l, "Short": a_s, "Wait": qwt})
            code = {"Long": A_LONG, "Short": A_SHORT,
                    "Wait": A_WAIT, "Tie": A_TIE}[act]
            acts[t] = code
            best = max(a_l, a_s, qwt)
            sec = sorted([a_l, a_s, qwt], reverse=True)[1]
            edges[t] = (best - sec) if np.isfinite(best) else np.nan
        out[H] = dict(action=acts, edge=edges)
    return out


# ---------------------------------------------------------------------------
# per-symbol robustness evaluation
# ---------------------------------------------------------------------------
def evaluate_symbol(bars, cand, thetas=THETAS, hmax=HMAX, keep_idx=None):
    """Run all thetas on one symbol; return row-level + matrix + counters."""
    n = cand["n"]; atr = cand["atr_t"]
    # restrict to valid decisions (same exclusion as R1.1: has_roundtrip)
    base_core = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    # reuse R1.1 has_roundtrip via oracle_core
    r1c = r1_oracle_core(bars, horizons=HORIZONS, hmax=hmax)
    valid = r1c["valid"] & r1c["has_roundtrip"]
    vidx = np.flatnonzero(valid)
    nv = len(vidx)

    # baseline (theta 0) actions / edges per H (from R2 theta 0 for consistency)
    base = actions_for_theta(base_core, cand)
    base_act = {H: base[H]["action"] for H in HORIZONS}

    # containers
    core_act = np.zeros((nv, N_CORE, 3), dtype=np.int8)
    core_edge = np.zeros((nv, N_CORE, 3), dtype=np.float32)
    core_val = np.zeros((nv, N_CORE, 3), dtype=np.float32)
    # per-theta optimal holding (for baseline-direction holding-time analysis)
    core_hold_l = np.zeros((nv, N_CORE, 3), dtype=np.int8)
    core_hold_s = np.zeros((nv, N_CORE, 3), dtype=np.int8)
    full_act = np.zeros((nv, len(thetas), 3), dtype=np.int8)

    # parameter-level counters (per theta x H)
    # action histogram [theta, H, 4]; transition vs baseline [theta,H,4,4];
    # edge histogram [theta,H,bins]; holding histogram [theta,H,24]
    # Fix R2.1: the edge histogram is built from the ATR-normalized edge
    # (dimensionless multiples of ATR), so the bins must live on that scale.
    EBINS_ATR = 801
    ELO_ATR, EHI_ATR = -100.0, 300.0
    eedges_atr = np.linspace(ELO_ATR, EHI_ATR, EBINS_ATR)
    act_hist = np.zeros((len(thetas), 3, 4), dtype=np.int64)
    trans = np.zeros((len(thetas), 3, 4, 4), dtype=np.int64)
    edge_hist = np.zeros((len(thetas), 3, EBINS_ATR - 1), dtype=np.int64)
    hold_hist = np.zeros((len(thetas), 3, 24), dtype=np.int64)

    # by-symbol accumulation (counts) for by_symbol summary
    sym_act = np.zeros((len(thetas), 3, 4), dtype=np.int64)
    sym_flip = np.zeros((len(thetas), 3), dtype=np.int64)   # baseline L/S -> opp
    sym_supp = np.zeros((len(thetas), 3), dtype=np.int64)   # baseline L/S -> W/T
    sym_crea = np.zeros((len(thetas), 3), dtype=np.int64)   # baseline W -> L/S

    for ti, th in enumerate(thetas):
        lam_r, lam_t, c, grid = th
        core = oracle_core_theta(cand, lam_r, lam_t, c)
        aH = actions_for_theta(core, cand)
        is_core = ti < N_CORE
        for hi, H in enumerate(HORIZONS):
            act = aH[H]["action"]; edge = aH[H]["edge"]; best = aH[H]["best"]
            edge_atr = aH[H]["edge_atr"]
            act_v = act[vidx]
            full_act[:, ti, hi] = act_v
            bact = base_act[H][vidx]
            # action histogram (over valid decisions only)
            for code, ci in ((A_LONG, 0), (A_SHORT, 1), (A_WAIT, 2), (A_TIE, 3)):
                cnt = int((act_v == code).sum())
                act_hist[ti, hi, ci] += cnt
                sym_act[ti, hi, ci] += cnt
            # transition vs baseline
            for bci in range(4):
                mask_b = (bact == (A_LONG if bci == 0 else A_SHORT if bci == 1
                                   else A_WAIT if bci == 2 else A_TIE))
                for cci in range(4):
                    trans[ti, hi, bci, cci] += int(((act_v == (A_LONG if cci == 0
                            else A_SHORT if cci == 1 else A_WAIT if cci == 2
                            else A_TIE)) & mask_b).sum())
            # flips / suppression / creation vs baseline
            if is_core:
                core_act[:, ti, hi] = act[vidx]
                core_hold_l[:, ti, hi] = aH[H]["hold_l"][vidx]
                core_hold_s[:, ti, hi] = aH[H]["hold_s"][vidx]
            # edge / value histograms (only for valid finite edges)
            # edge_ATR = ATR-normalized edge (Fix R2.1); used for parameter median
            emask = np.isfinite(edge_atr[vidx])
            ev = np.clip(edge_atr[vidx][emask], ELO_ATR, EHI_ATR)
            if len(ev):
                eh = np.histogram(ev, bins=eedges_atr)[0]
                edge_hist[ti, hi] += eh.astype(np.int64)
            if is_core:
                core_edge[:, ti, hi] = edge_atr[vidx].astype(np.float32)
                core_val[:, ti, hi] = (best[vidx] / np.where(
                    atr[vidx] > 0, atr[vidx], np.nan)).astype(np.float32)
            # holding histogram for trade actions
            hold = np.where(act == A_LONG, aH[H]["hold_l"],
                            np.where(act == A_SHORT, aH[H]["hold_s"],
                                     np.zeros_like(act)))
            hv = hold[vidx]
            hv = hv[(hv >= 1) & (hv <= 24)]
            if len(hv):
                hold_hist[ti, hi, :] += np.bincount(
                    hv.astype(int) - 1, minlength=24).astype(np.int64)
            # flips/suppression/creation per theta (use full act vs baseline)
            if hi == 0:   # accumulate once per (theta) using H=6 baseline? use each H
                pass
            # per-H flip/supp/crea counters (sum across H later)
            bl = (bact == A_LONG); bs = (bact == A_SHORT); bw = (bact == A_WAIT)
            sym_flip[ti, hi] += int((bl & (act_v == A_SHORT)).sum()
                                     + (bs & (act_v == A_LONG)).sum())
            sym_supp[ti, hi] += int(((bl | bs) & ((act_v == A_WAIT) | (act_v == A_TIE))).sum())
            sym_crea[ti, hi] += int((bw & ((act_v == A_LONG) | (act_v == A_SHORT))).sum())

    # row-level reductions across the 27 core thetas
    Hn = len(HORIZONS)
    rows = []
    for j, t in enumerate(vidx):
        rec = dict(symbol=bars["symbol"],
                   decision_bar_index=int(t),
                   decision_bar_start_time=pd.Timestamp(bars["t"][t]),
                   decision_time=pd.Timestamp(bars["decision_time"][t]),
                   atr5_t=(None if not np.isfinite(atr[t]) else float(atr[t])))
        # baseline stable action across 3 H
        bacts = [int(base_act[H][t]) for H in HORIZONS]
        bset = set(bacts)
        if len(bset) == 1 and bacts[0] != A_TIE:
            bstable = bacts[0]
        else:
            bstable = 0 if A_WAIT in bset and len(bset) == 1 else (
                9 if A_TIE in bset else 0)
            # 9 = Ambiguous sentinel; wait-only-stable -> Wait(0)
            if bset == {A_WAIT}:
                bstable = A_WAIT
            elif bset == {A_TIE}:
                bstable = A_TIE
            elif len(bset) > 1:
                bstable = 9
        rec["baseline_stable_action"] = ("Long" if bstable == A_LONG else
                                         "Short" if bstable == A_SHORT else
                                         "Wait" if bstable == A_WAIT else
                                         "Tie" if bstable == A_TIE else
                                         "Ambiguous")
        joint_match = 0; joint_tot = N_CORE * Hn
        joint_opp = 0; joint_tie = 0
        for hi, H in enumerate(HORIZONS):
            ca = core_act[j, :, hi]            # 27 codes
            ce = core_edge[j, :, hi]           # 27 edge_ATR
            cv = core_val[j, :, hi]            # 27 value_ATR
            sh_L, sh_S, sh_W, sh_T, cons, cons_rate = _action_shares(ca, N_CORE)
            bact = int(base_act[H][t])
            breten = float((ca == bact).mean())
            opp_rate = (float((bact == A_LONG) and (ca == A_SHORT).mean()
                              if bact == A_LONG else
                              (ca == A_LONG).mean() if bact == A_SHORT else 0.0))
            if bact in (A_LONG, A_SHORT):
                supp = float(((ca == A_WAIT) | (ca == A_TIE)).mean())
                flip = opp_rate
                crea = 0.0
            else:
                supp = 0.0
                flip = 0.0
                crea = float(((ca == A_LONG) | (ca == A_SHORT)).mean())
            rec[f"Long_share_H{H}"] = round(float(sh_L), 4)
            rec[f"Short_share_H{H}"] = round(float(sh_S), 4)
            rec[f"Wait_share_H{H}"] = round(float(sh_W), 4)
            rec[f"Tie_share_H{H}"] = round(float(sh_T), 4)
            rec[f"consensus_action_H{H}"] = cons
            rec[f"consensus_rate_H{H}"] = round(float(cons_rate), 4)
            rec[f"baseline_retention_H{H}"] = round(float(breten), 4)
            rec[f"opposite_flip_H{H}"] = round(float(flip), 4)
            rec[f"suppression_creation_H{H}"] = round(float(supp + crea), 4)
            rec[f"edge_ATR_min_H{H}"] = round(float(np.nanmin(ce)), 5) \
                if np.any(np.isfinite(ce)) else None
            rec[f"edge_ATR_median_H{H}"] = round(float(np.nanmedian(ce)), 5) \
                if np.any(np.isfinite(ce)) else None
            rec[f"edge_ATR_max_H{H}"] = round(float(np.nanmax(ce)), 5) \
                if np.any(np.isfinite(ce)) else None
            rec[f"value_ATR_min_H{H}"] = round(float(np.nanmin(cv)), 5) \
                if np.any(np.isfinite(cv)) else None
            rec[f"value_ATR_median_H{H}"] = round(float(np.nanmedian(cv)), 5) \
                if np.any(np.isfinite(cv)) else None
            rec[f"value_ATR_max_H{H}"] = round(float(np.nanmax(cv)), 5) \
                if np.any(np.isfinite(cv)) else None
            # joint stats
            joint_match += int((ca == bact).sum())
            joint_tie += int((ca == A_TIE).sum())
            if bact in (A_LONG, A_SHORT):
                joint_opp += int((ca == (-bact)).sum())
        rec["joint_retention"] = round(joint_match / joint_tot, 4)
        rec["joint_opposite_flip_rate"] = round(joint_opp / joint_tot, 4)
        rec["joint_tie_rate"] = round(joint_tie / joint_tot, 4)
        # Fix R2.1b: stable-cohort joint retention headline. For decisions whose
        # baseline action is itself stable across the 3 H (Long/Short/Wait), the
        # three H baselines coincide, so joint_match/joint_tot already equals the
        # stable-cohort definition #{ (theta,H): A = A_stable } / 81. Ambiguous /
        # Tie baselines are excluded from the stable cohort.
        rec["joint_retention_stable"] = (
            round(joint_match / joint_tot, 4)
            if bstable in (A_LONG, A_SHORT, A_WAIT) else None)
        # --- item 5: holding-time distribution WHEN DIRECTION is unchanged ---
        # Gather optimal holding across the 81 (theta,H) cells whose action keeps
        # the baseline direction. This separates "direction stable but exit time
        # jumpy" from "direction itself flips".
        if bstable in (A_LONG, A_SHORT):
            hcols = []
            for hi in range(3):
                ca = core_act[j, :, hi]
                hl = core_hold_l[j, :, hi]
                hs = core_hold_s[j, :, hi]
                hv = np.where(ca == A_LONG, hl, hs)
                hcols.append(hv[ca == bstable])
            hv_all = (np.concatenate(hcols) if hcols
                      else np.array([], dtype=np.int64))
            if len(hv_all):
                rec["baseline_dir_holding_p10"] = round(float(np.percentile(hv_all, 10)), 2)
                rec["baseline_dir_holding_median"] = round(float(np.median(hv_all)), 2)
                rec["baseline_dir_holding_p90"] = round(float(np.percentile(hv_all, 90)), 2)
                rec["baseline_dir_holding_span"] = int(hv_all.max() - hv_all.min())
                rec["baseline_dir_holding_std"] = round(float(np.std(hv_all)), 3)
            else:
                rec["baseline_dir_holding_p10"] = None
                rec["baseline_dir_holding_median"] = None
                rec["baseline_dir_holding_p90"] = None
                rec["baseline_dir_holding_span"] = None
                rec["baseline_dir_holding_std"] = None
        else:
            rec["baseline_dir_holding_p10"] = None
            rec["baseline_dir_holding_median"] = None
            rec["baseline_dir_holding_p90"] = None
            rec["baseline_dir_holding_span"] = None
            rec["baseline_dir_holding_std"] = None
        # strict robust action: all 81 (theta,H) identical and != Tie?
        all_act = full_act[j, :N_CORE, :].reshape(-1)
        uniq = np.unique(all_act)
        if len(uniq) == 1:
            c = int(uniq[0])
            rec["strict_robust_action"] = ("Long" if c == A_LONG else
                                           "Short" if c == A_SHORT else
                                           "Wait" if c == A_WAIT else "Tie")
        else:
            rec["strict_robust_action"] = ""
        rows.append(rec)

    # strip unused locals to avoid lint noise
    _ = (sym_flip, sym_supp, sym_crea, keep_idx)

    counters = dict(act_hist=act_hist, trans=trans, edge_hist=edge_hist,
                    hold_hist=hold_hist, sym_act=sym_act,
                    sym_flip=sym_flip, sym_supp=sym_supp, sym_crea=sym_crea,
                    eedges=eedges_atr, n_valid=int(nv))
    # full_act is already in valid-decision space (axis0 == nv == len(vidx)).
    return rows, full_act, counters, nv, vidx


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------
def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _assert_matrix_key_alignment(rows_df, matrix_df):
    """Fail-closed: every (symbol, decision_bar_index, decision_time) key in the
    row-level table must appear exactly once in the action matrix and vice versa.
    R2.1 fixes the previous bug where the matrix key was np.arange(nv) and could
    silently misalign whenever valid decisions are non-contiguous.
    """
    key = ["symbol", "decision_bar_index", "decision_time"]
    rk = rows_df[key]
    mk = matrix_df[key]
    assert not rk.duplicated().any(), "duplicate row-level keys"
    assert not mk.duplicated().any(), "duplicate matrix keys"
    merged = rk.merge(mk, on=key, how="outer", indicator=True)
    bad = int((merged["_merge"] != "both").sum())
    assert bad == 0, f"{bad} key(s) missing/extra between rows and matrix"


def run_all(symbols=SYMBOLS, tail_bars=None, out_dir=None):
    out_dir = Path(out_dir) if out_dir else OUT
    t0 = time.perf_counter()
    mem0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rows_all = []
    matrix_frames = []
    all_counters = []
    per_sym = {}
    raw_load = 0
    precomp = 0
    for sym in symbols:
        bt = time.perf_counter()
        bars = build_bars(sym)
        raw_load += 1
        if tail_bars is not None:
            keep = min(tail_bars, bars["n"])
            for k in ("o", "h", "l", "c", "disc", "seg", "atr5"):
                bars[k] = bars[k][-keep:]
            bars["t"] = bars["t"][-keep:]
            bars["decision_time"] = bars["decision_time"][-keep:]
            bars["n"] = keep
        cand = precompute_candidate_paths(bars)
        precomp += 1
        rows, mat, counters, nv, vidx = evaluate_symbol(bars, cand)
        per_sym[sym] = dict(n=int(nv),
                             sec=round(time.perf_counter() - bt, 2))
        rows_all.append(pd.DataFrame(rows))
        mdf = pd.DataFrame({
            "symbol": sym,
            "decision_bar_index": vidx,
            "decision_time": pd.to_datetime(bars["decision_time"][vidx]),
        })
        # matrix: columns per theta (H6/H12/H24 action codes)
        cols = {}
        for ti, th in enumerate(THETAS):
            tid = theta_id(*th)
            cols[f"{tid}_H6"] = mat[:, ti, 0]
            cols[f"{tid}_H12"] = mat[:, ti, 1]
            cols[f"{tid}_H24"] = mat[:, ti, 2]
        mdf = pd.concat([mdf, pd.DataFrame(cols)], axis=1)
        matrix_frames.append(mdf)
        all_counters.append(counters)
        print(f"[{sym}] rows={nv} "
              f"({time.perf_counter()-bt:.1f}s)")

    rows_df = pd.concat(rows_all, ignore_index=True)
    matrix_df = pd.concat(matrix_frames, ignore_index=True)
    _assert_matrix_key_alignment(rows_df, matrix_df)

    # aggregate counters across symbols
    agg = _aggregate_counters(all_counters)
    summary = _build_summary(rows_df, agg, per_sym, t0, mem0, raw_load, precomp)
    rows_sha = _write_artifacts(rows_df, matrix_df, summary, agg, per_sym, out_dir)
    print(f"\n[DONE] {time.perf_counter()-t0:.1f}s rows={len(rows_df)} "
          f"sha={rows_sha[:12]} -> {out_dir}")
    return dict(rows=rows_df, summary=summary, rows_sha=rows_sha)


def _aggregate_counters(counters):
    n_th = len(THETAS); nH = 3
    act_hist = np.zeros((n_th, nH, 4), dtype=np.int64)
    trans = np.zeros((n_th, nH, 4, 4), dtype=np.int64)
    edge_hist = np.zeros((n_th, nH, len(counters[0]["edge_hist"][0, 0])),
                         dtype=np.int64)
    hold_hist = np.zeros((n_th, nH, 24), dtype=np.int64)
    for c in counters:
        act_hist += c["act_hist"]
        trans += c["trans"]
        edge_hist += c["edge_hist"]
        hold_hist += c["hold_hist"]
    return dict(act_hist=act_hist, trans=trans, edge_hist=edge_hist,
                hold_hist=hold_hist, eedges=counters[0]["eedges"])


def _hist_median(hist, edges):
    c = np.cumsum(hist)
    if c[-1] == 0:
        return None
    mid = c[-1] / 2.0
    idx = np.searchsorted(c, mid)
    idx = min(idx, len(edges) - 2)
    return float(edges[idx])


def _build_summary(rows_df, agg, per_sym, t0, mem0, raw_load_count=0,
                  precompute_count=0):
    s = dict(experiment="Oracle Constraint Robustness v1",
             task_id="FUTURE-ORACLE-R2.1-CORRECTNESS-CLOSURE",
             horizons=list(HORIZONS), n_symbols=len(per_sym),
             n_core_thetas=N_CORE, n_stress_thetas=len(THETAS) - N_CORE,
             n_thetas=len(THETAS),
             n_decisions=int(len(rows_df)),
             runtime_sec=round(time.perf_counter() - t0, 2),
             peak_rss_mb=round(
                 resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1048576, 1),
             per_symbol_timing=per_sym,
             raw_load_count=int(raw_load_count),
             precompute_count=int(precompute_count))
    # Fix R2.1b: explicitly separate the all-row DIAGNOSTIC from the
    # stable-cohort HEADLINE. The stable cohort only includes decisions whose
    # baseline action is itself stable (Long/Short/Wait) across all three H.
    def _jr_dist(jr):
        if len(jr) == 0:
            return None
        return dict(
            n=int(len(jr)),
            p10=round(float(jr.quantile(.1)), 4),
            p25=round(float(jr.quantile(.25)), 4),
            p50=round(float(jr.quantile(.5)), 4),
            p75=round(float(jr.quantile(.75)), 4),
            p90=round(float(jr.quantile(.9)), 4),
            mean=round(float(jr.mean()), 4),
            frac_full=round(float((jr >= 0.999).mean()), 4),
            frac_above_0_8=round(float((jr >= 0.8).mean()), 4))

    def _jr_thr(jr):
        if len(jr) == 0:
            return None
        return dict(
            ge_0_999=round(float((jr >= 0.999).mean()), 4),
            ge_0_9=round(float((jr >= 0.9).mean()), 4),
            ge_0_8=round(float((jr >= 0.8).mean()), 4),
            ge_0_7=round(float((jr >= 0.7).mean()), 4),
            ge_0_5=round(float((jr >= 0.5).mean()), 4))

    jr_all = pd.to_numeric(rows_df["joint_retention"], errors="coerce").dropna()
    s["joint_retention_all_distribution"] = _jr_dist(jr_all)
    s["joint_retention_all_thresholds"] = _jr_thr(jr_all)
    jr_stable = pd.to_numeric(rows_df["joint_retention_stable"],
                              errors="coerce").dropna()
    s["joint_retention_stable_distribution"] = _jr_dist(jr_stable)
    s["joint_retention_stable_thresholds"] = _jr_thr(jr_stable)
    s["stable_cohort_n"] = int(len(jr_stable))
    # joint opposite flip / tie rate
    s["joint_opposite_flip_rate_mean"] = round(
        float(pd.to_numeric(rows_df["joint_opposite_flip_rate"],
                            errors="coerce").mean()), 4)
    s["joint_tie_rate_mean"] = round(
        float(pd.to_numeric(rows_df["joint_tie_rate"],
                            errors="coerce").mean()), 4)
    # strict robust counts
    sr = rows_df["strict_robust_action"].fillna("")
    s["strict_robust_counts"] = sr.value_counts().to_dict()
    # Fix R2.1b: Core and Stress parameter regions MUST be reported SEPARATELY.
    # The headline robustness question is the Core (reasonable) region; Stress
    # only characterizes behaviour under extreme conditions.
    core_tr = agg["trans"][:N_CORE]
    stress_tr = agg["trans"][N_CORE:]

    def _direction_vs_suppression(tr):
        L, S, W, T = 0, 1, 2, 3
        flip_LS = float(tr[:, :, L, S].sum())
        flip_SL = float(tr[:, :, S, L].sum())
        supp_L = float(tr[:, :, L, W].sum() + tr[:, :, L, T].sum())
        supp_S = float(tr[:, :, S, W].sum() + tr[:, :, S, T].sum())
        crea = float(tr[:, :, W, L].sum() + tr[:, :, W, S].sum())
        cells_L = float(tr[:, :, L, :].sum()) or 1.0
        cells_S = float(tr[:, :, S, :].sum()) or 1.0
        cells_W = float(tr[:, :, W, :].sum()) or 1.0
        return dict(
            baseline_Long_cells=int(cells_L),
            baseline_Short_cells=int(cells_S),
            baseline_Wait_cells=int(cells_W),
            Long_to_Short_count=int(flip_LS),
            Short_to_Long_count=int(flip_SL),
            Long_to_Short_rate=round(flip_LS / cells_L, 4),
            Short_to_Long_rate=round(flip_SL / cells_S, 4),
            Long_to_WaitTie_count=int(supp_L),
            Short_to_WaitTie_count=int(supp_S),
            Long_to_WaitTie_rate=round(supp_L / cells_L, 4),
            Short_to_WaitTie_rate=round(supp_S / cells_S, 4),
            Wait_to_Trade_count=int(crea),
            Wait_to_Trade_rate=round(crea / cells_W, 4))

    s["direction_vs_suppression_core"] = _direction_vs_suppression(core_tr)
    s["direction_vs_suppression_stress"] = _direction_vs_suppression(stress_tr)
    # ---- item 2: baseline-trade retention distribution (stable Long / Short) --
    ret = {}
    for bsa, label in (("Long", "stable_Long"), ("Short", "stable_Short")):
        sub = pd.to_numeric(
            rows_df.loc[rows_df["baseline_stable_action"] == bsa,
                        "baseline_retention_H6"], errors="coerce").dropna()
        if len(sub):
            ret[label] = dict(
                n=int(len(sub)),
                p10=round(float(sub.quantile(.1)), 4),
                p25=round(float(sub.quantile(.25)), 4),
                median=round(float(sub.median()), 4),
                p75=round(float(sub.quantile(.75)), 4),
                p90=round(float(sub.quantile(.9)), 4),
                mean=round(float(sub.mean()), 4),
                frac_above_0_8=round(float((sub >= 0.8).mean()), 4),
                frac_below_0_5=round(float((sub < 0.5).mean()), 4))
        else:
            ret[label] = dict(n=0)
    s["baseline_trade_retention_distribution"] = ret
    # ---- item 5: holding-time distribution when DIRECTION unchanged -----------
    hm = pd.to_numeric(rows_df["baseline_dir_holding_median"], errors="coerce").dropna()
    hspan = pd.to_numeric(rows_df["baseline_dir_holding_span"], errors="coerce").dropna()
    if len(hm):
        s["baseline_direction_holding"] = dict(
            n=int(len(hm)),
            median_of_medians=round(float(hm.median()), 2),
            p10_of_medians=round(float(hm.quantile(.1)), 2),
            p90_of_medians=round(float(hm.quantile(.9)), 2),
            frac_span_ge_12=round(float((hspan >= 12).mean()), 4),
            frac_span_ge_6=round(float((hspan >= 6).mean()), 4),
            mean_span=round(float(hspan.mean()), 2))
    else:
        s["baseline_direction_holding"] = dict(n=0)
    # ATR warmup note (explicitly recorded per user request)
    s["atr_warmup_note"] = ("ATR warmup NaN -> 0 only for penalty scaling in the "
                            "R2 oracle implementation; not applied to entry/exit "
                            "prices. If early-bar effects matter, exclude them "
                            "separately in a later round (not changed this round).")
    # parameter summary table
    s["parameter_summary"] = _parameter_table(agg)
    # one-factor sensitivity
    s["one_factor_sensitivity"] = _one_factor(agg)
    # core vs stress headline comparison (joint retention subset)
    return s


def _parameter_table(agg):
    act_hist = agg["act_hist"]; trans = agg["trans"]; nH = 3
    tbl = []
    for ti, th in enumerate(THETAS):
        r, t, c, grid = th
        tid = theta_id(*th)
        row = dict(theta_id=tid, lambda_r=r, lambda_t=t,
                   friction_hurdle_atr=c, grid_type=grid)
        tot = act_hist[ti, 0].sum()
        for hi, H in enumerate(HORIZONS):
            a = act_hist[ti, hi]
            tot_h = a.sum() or 1
            row[f"H{H}_Long_pct"] = round(100 * a[0] / tot_h, 3)
            row[f"H{H}_Short_pct"] = round(100 * a[1] / tot_h, 3)
            row[f"H{H}_Wait_pct"] = round(100 * a[2] / tot_h, 3)
            row[f"H{H}_Tie_pct"] = round(100 * a[3] / tot_h, 3)
        # baseline agreement across H (use transition baseline vs (0,0,0))
        if ti == BASELINE_IDX:
            row["baseline_action_agreement"] = 1.0
        else:
            # Agreement = fraction of (decision, H) cells where this theta's
            # action equals the baseline theta's action. Derived from the
            # transition tensor: trace(T_{theta,H}) / sum(T_{theta,H}) averaged
            # over H (== #matches / (N*3)). Fix R2.1: previously always None.
            num = 0.0
            den = 0.0
            for hi in range(nH):
                tr = trans[ti, hi]
                num += float(np.trace(tr))
                den += float(tr.sum())
            row["baseline_action_agreement"] = (
                round(num / den, 6) if den > 0 else None)
        # median edge_ATR from histogram
        meds = [_hist_median(agg["edge_hist"][ti, hi], agg["eedges"])
                for hi in range(nH)]
        for hi, H in enumerate(HORIZONS):
            row[f"H{H}_median_edge_ATR"] = (round(meds[hi], 5)
                                            if meds[hi] is not None else None)
        tbl.append(row)
    return tbl


def _one_factor(agg):
    """Extract risk-only / time-only / friction-only sweeps from CORE grid."""
    def sub(rf, tf, cf):
        # find theta index in CORE matching
        for ti, (r, t, c, g) in enumerate(CORE):
            if abs(r - rf) < 1e-12 and abs(t - tf) < 1e-12 and abs(c - cf) < 1e-12:
                return ti
        return None
    res = {}
    # risk only: lambda_t=0, c=0
    res["risk_only"] = _sweep(agg, [(r, 0.0, 0.0) for r in LAM_R])
    # time only
    res["time_only"] = _sweep(agg, [(0.0, t, 0.0) for t in LAM_T])
    # friction only
    res["friction_only"] = _sweep(agg, [(0.0, 0.0, c) for c in C_FR])
    return res


def _sweep(agg, pts):
    """One-factor sweep over CORE thetas, with CONDITIONAL denominators.

    Fix R2.1b: rates are conditioned on the relevant baseline-action cells, NOT
    all decision cells. Numerators/denominators are also returned for audit.
      - flip   P(L<->S | baseline trade)        denom = baseline L + S cells
      - supp   P(Wait/Tie | baseline trade)     denom = baseline L + S cells
      - crea   P(L/S | baseline Wait)           denom = baseline Wait cells
    """
    rows = []
    for (r, t, c) in pts:
        ti = None
        for k, (rr, tt, cc, g) in enumerate(CORE):
            if abs(rr - r) < 1e-12 and abs(tt - t) < 1e-12 and abs(cc - c) < 1e-12:
                ti = k; break
        if ti is None:
            continue
        flip_num = 0.0; supp_num = 0.0; crea_num = 0.0
        flip_den = 0.0; supp_den = 0.0; crea_den = 0.0
        for hi in range(3):
            # transition[ti, hi, bci, cci]; Long=0, Short=1, Wait=2, Tie=3
            tr = agg["trans"][ti, hi]
            base_L = float(tr[0, :].sum())   # baseline Long cells
            base_S = float(tr[1, :].sum())   # baseline Short cells
            base_W = float(tr[2, :].sum())   # baseline Wait cells
            # direction flip (Long<->Short) vs baseline trade
            flip_num += tr[0, 1] + tr[1, 0]
            flip_den += base_L + base_S
            # trade -> Wait/Tie suppression vs baseline trade
            supp_num += tr[0, 2] + tr[0, 3] + tr[1, 2] + tr[1, 3]
            supp_den += base_L + base_S
            # Wait -> trade creation vs baseline Wait
            crea_num += tr[2, 0] + tr[2, 1]
            crea_den += base_W
        flip = flip_num / flip_den if flip_den else 0.0
        supp = supp_num / supp_den if supp_den else 0.0
        crea = crea_num / crea_den if crea_den else 0.0
        rows.append(dict(lambda_r=r, lambda_t=t, friction_hurdle_atr=c,
                         opposite_flip_rate=round(flip, 4),
                         opposite_flip_count=int(flip_num),
                         baseline_trade_count=int(round(flip_den)),
                         trade_suppression_rate=round(supp, 4),
                         trade_suppression_count=int(supp_num),
                         trade_creation_rate=round(crea, 4),
                         trade_creation_count=int(crea_num),
                         baseline_wait_count=int(round(crea_den))))
    return rows


def _write_artifacts(rows_df, matrix_df, summary, agg, per_sym, out_dir=OUT):
    rows_path = out_dir / "oracle_constraint_rows.parquet"
    rows_df.to_parquet(rows_path, index=False)
    rows_sha = _sha256(rows_path)

    matrix_path = out_dir / "oracle_constraint_action_matrix.parquet"
    matrix_df.to_parquet(matrix_path, index=False)

    # parameter summary csv (flattened)
    psum = summary["parameter_summary"]
    pd.DataFrame(psum).to_csv(out_dir / "oracle_constraint_parameter_summary.csv",
                              index=False)
    # by symbol: robustness aggregation directly from the row-level table
    # (no DP re-run). Fix R2.1: previously only a runtime table.
    by_sym = []
    for sym in per_sym.keys():
        g = rows_df[rows_df["symbol"] == sym]
        jr = pd.to_numeric(g["joint_retention"], errors="coerce").dropna()
        jrs = pd.to_numeric(g["joint_retention_stable"], errors="coerce").dropna()
        jf = pd.to_numeric(g["joint_opposite_flip_rate"],
                           errors="coerce").dropna()
        jt = pd.to_numeric(g["joint_tie_rate"], errors="coerce").dropna()
        sr = g["strict_robust_action"].fillna("")
        bsa = g["baseline_stable_action"].fillna("")
        by_sym.append(dict(
            symbol=sym,
            n_decisions=int(len(g)),
            sec=per_sym[sym]["sec"],
            joint_retention_mean=(round(float(jr.mean()), 4) if len(jr) else None),
            joint_retention_ge_0_9=(round(float((jr >= 0.9).mean()), 4)
                                   if len(jr) else None),
            joint_retention_ge_0_8=(round(float((jr >= 0.8).mean()), 4)
                                   if len(jr) else None),
            joint_retention_stable_mean=(round(float(jrs.mean()), 4)
                                         if len(jrs) else None),
            joint_retention_stable_ge_0_9=(round(float((jrs >= 0.9).mean()), 4)
                                           if len(jrs) else None),
            joint_retention_stable_ge_0_8=(round(float((jrs >= 0.8).mean()), 4)
                                           if len(jrs) else None),
            joint_opposite_flip_mean=(round(float(jf.mean()), 4)
                                     if len(jf) else None),
            joint_tie_mean=(round(float(jt.mean()), 4) if len(jt) else None),
            strict_Long_n=int((sr == "Long").sum()),
            strict_Short_n=int((sr == "Short").sum()),
            strict_Wait_n=int((sr == "Wait").sum()),
            strict_Tie_n=int((sr == "Tie").sum()),
            strict_nonrobust_n=int((sr == "").sum()),
            baseline_stable_Long_n=int((bsa == "Long").sum()),
            baseline_stable_Short_n=int((bsa == "Short").sum()),
            baseline_stable_Wait_n=int((bsa == "Wait").sum()),
        ))
    pd.DataFrame(by_sym).to_csv(out_dir / "oracle_constraint_by_symbol.csv",
                                index=False)
    # by month (calendar month of decision_time)
    blk = pd.to_datetime(rows_df["decision_time"]).dt.strftime("%Y-%m")
    by_month = []
    for m, g in rows_df.groupby(blk):
        by_month.append(dict(time_block=m, n=int(len(g)),
                             joint_retention_mean=round(
                                 float(g["joint_retention"].mean()), 4),
                             joint_opposite_flip_mean=round(
                                 float(g["joint_opposite_flip_rate"].mean()), 4),
                             joint_tie_mean=round(
                                 float(g["joint_tie_rate"].mean()), 4)))
    pd.DataFrame(by_month).to_csv(out_dir / "oracle_constraint_by_month.csv",
                                  index=False)

    json.dump(summary, open(out_dir / "oracle_constraint_summary.json", "w"),
              indent=2, default=str)

    protocol = dict(
        experiment="Oracle Constraint Robustness v1",
        task_id="FUTURE-ORACLE-R2.1-CORRECTNESS-CLOSURE",
        version="1.0",
        base_commit="0bee4029a40a8985f1be6da7431d2edeb0109dd6",
        frozen=["H in {6,12,24}", "decision_time", "segment", "entry", "exit",
                "discontinuity", "tie semantics", "label availability",
                "ATR5 owner"],
        forbidden=["DTP", "SR", "Liquidity", "15m/1H/4H", "GBDT", "MoE", "PGM",
                  "real-but-unverified fee data"],
        utility=("U = R - lambda_R*MAE - c*ATR_t - lambda_T*ATR_t*k "
                 "(price points; ATR only used at output as Q/ATR)"),
        dp_rule=("Bellman in price points; NEVER normalize V by ATR before "
                 "recursion. Penalties use decision-time ATR_t."),
        thetas_core=27, thetas_stress=len(THETAS) - N_CORE,
        total_thetas=len(THETAS),
        core_grid=dict(lambda_r=LAM_R, lambda_t=LAM_T,
                       friction_hurdle_atr=C_FR),
        stress_points=STRESS_POINTS,
        friction_hurdle_note=("c*ATR_t is a FRICTION HURDLE, NOT actual "
                              "transaction cost. FRICTION_HURDLE_NOT_ACTUAL_"
                              "TRANSACTION_COST"),
        one_load_per_symbol=True,
        batched_theta_eval=True,
        outputs=["oracle_constraint_rows.parquet",
                 "oracle_constraint_action_matrix.parquet",
                 "oracle_constraint_parameter_summary.csv",
                 "oracle_constraint_by_symbol.csv",
                 "oracle_constraint_by_month.csv",
                 "oracle_constraint_summary.json",
                 "ORACLE_CONSTRAINT_PROTOCOL.json",
                 "ORACLE_CONSTRAINT_AUDIT.json",
                 "oracle_constraint_report.md"],
    )
    json.dump(protocol, open(out_dir / "ORACLE_CONSTRAINT_PROTOCOL.json", "w"),
              indent=2, default=str)

    audit = dict(
        experiment="Oracle Constraint Robustness v1",
        task_id="FUTURE-ORACLE-R2.1-CORRECTNESS-CLOSURE",
        rows_sha256=rows_sha,
        rows_path=str(rows_path),
        n_decisions=int(len(rows_df)),
        raw_load_count=summary.get("raw_load_count"),
        precompute_count=summary.get("precompute_count"),
        grid=dict(n_core=N_CORE, n_stress=len(THETAS) - N_CORE,
                  total=len(THETAS),
                  core=[(r, t, c) for (r, t, c, g) in CORE],
                  stress=STRESS_POINTS),
        cost=dict(
            canonical_table_found=False,
            NET_PNL="UNAVAILABLE_COST_METADATA",
            rule=("friction hurdle c*ATR_t is NOT a real fee; only gross "
                  "price-point utility + risk/time/friction penalties"),
            FRICTION_HURDLE_NOT_ACTUAL_TRANSACTION_COST=True,
        ),
    )
    json.dump(audit, open(out_dir / "ORACLE_CONSTRAINT_AUDIT.json", "w"),
              indent=2, default=str)

    _write_report(rows_df, summary, protocol, audit, out_dir)
    return rows_sha


def _write_report(rows_df, summary, protocol, audit, out_dir=OUT):
    md = f"""# Oracle Constraint Robustness v1 (R2.1)

> **Status**: `PROVISIONAL_PENDING_USER_AUDIT`. This is a SENSITIVITY /
> ROBUSTNESS surface, not parameter tuning. No "best lambda" is selected.

**Task**: `FUTURE-ORACLE-R2.1-CORRECTNESS-CLOSURE`
**Base**: {protocol['base_commit']} (R1.1)
**Horizons**: {list(HORIZONS)}
**Grid**: {summary['n_core_thetas']} core + {summary['n_stress_thetas']} stress
= {summary['n_thetas']} thetas. Core = λR∈{0,.25,.5} × λT∈{0,.005,.010} ×
c∈{0,.025,.05}.

## 0. Frozen contract (unchanged from R1.1)
5m decision clock; H=6/12/24; next-open execution; at most one round-trip;
discontinuity ban; Long/Short/Wait/Tie. Only risk/time/friction change.

## 1. Critical DP rule
Bellman is performed ENTIRELY in price points. Penalties use the DECISION-time
ATR_t. We never feed ATR-normalized V back into the recursion.

## 2. Sample counts
- n_decisions = {summary['n_decisions']}
- runtime_sec = {summary['runtime_sec']}
- peak_rss_mb = {summary['peak_rss_mb']}

## 3. Headline robustness (Core region only)
- **Direction robustness** (Long<->Short direct flip, Core grid):
  joint_opposite_flip_rate_mean = {summary['joint_opposite_flip_rate_mean']}
- **Opportunity robustness** (trade -> Wait/Tie suppression, Core): see
  direction_vs_suppression_core + trade_suppression_rate in one_factor_sensitivity.
- **Timing robustness** (action same but holding/exit sensitive):
  edge_ATR / value_ATR min-median-max per (theta, H) in rows + parameter_summary.
- NOTE: Stress region (direction_vs_suppression_stress) is reported separately
  and only characterizes extreme-condition behaviour, never the headline.

## 4. Joint retention distribution
Headline = **stable-cohort** joint retention (baseline action itself stable across
H6/H12/H24; Ambiguous/Tie baselines excluded). All-row diagnostic retained.

**Stable cohort** (n = {summary['stable_cohort_n']}):
{summary['joint_retention_stable_distribution']}

**All rows (diagnostic only, NOT the headline)**:
{summary['joint_retention_all_distribution']}

## 5. Strict robust action counts
{summary['strict_robust_counts']}

## 6. One-factor sensitivity
- risk only: {summary['one_factor_sensitivity']['risk_only']}
- time only: {summary['one_factor_sensitivity']['time_only']}
- friction only: {summary['one_factor_sensitivity']['friction_only']}

## 7. Cost metadata
```json
{audit['cost']}
```
"""
    open(out_dir / "oracle_constraint_report.md", "w",
         encoding="utf-8-sig").write(md)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=SYMBOLS)
    ap.add_argument("--tail-bars", type=int, default=None,
                    help="only use last N bars per symbol (T1-like debug)")
    args = ap.parse_args()
    run_all(symbols=args.symbols, tail_bars=args.tail_bars)


if __name__ == "__main__":
    main()
