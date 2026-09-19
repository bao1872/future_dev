"""T0 (unit) + T1 (small real) tests for Oracle Constraint Robustness R2.

Run: python -m pytest research/liquidity_oracle_atlas/test_oracle_constraint_robustness_v1.py -q
  or: python research/liquidity_oracle_atlas/test_oracle_constraint_robustness_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from research.phase1_tradability.phase1_contract_v1 import compute_atr5
from research.liquidity_oracle_atlas.build_robust_trade_oracle_dp_v1 import (
    oracle_core as r1_oracle_core,
    _classify,
    HORIZONS,
    HMAX,
)
from research.liquidity_oracle_atlas.build_oracle_constraint_robustness_v1 import (  # noqa: E402
    oracle_core_theta, actions_for_theta, classify_batch,
    precompute_candidate_paths, brute_force_oracle_r2,
    build_bars, THETAS, CORE, STRESS_POINTS, N_CORE, NEG,
    A_LONG, A_SHORT, A_WAIT, A_TIE,
)

TOL = 1e-9
NEGP = -1e18


def make_bars(o, disc=None, h=None, l=None, sym="SYNTH"):
    o = np.asarray(o, float)
    n = len(o)
    if h is None:
        h = o.copy()
    if l is None:
        l = o.copy()
    if disc is None:
        disc = np.zeros(n, bool)
    disc = np.asarray(disc, bool)
    t = pd.date_range("2025-01-02 09:00", periods=n, freq="5min").to_numpy()
    atr5 = compute_atr5(dict(high=h, low=l, close=o))
    bars = dict(
        symbol=sym, o=o, h=np.asarray(h, float), l=np.asarray(l, float),
        c=o.copy(), t=t, disc=disc, atr5=np.asarray(atr5, float),
        seg=np.cumsum(disc.astype(np.int64)),
        decision_time=t + pd.Timedelta(minutes=5), n=n)
    return bars


def _bars_with_atr(o, atr_val, disc=None):
    """make_bars but with a CONSTANT, non-warmup ATR (overrides computed atr5).

    This lets time/friction tests control the ATR-scaled penalty deterministically
    without depending on compute_atr5 warmup behaviour (which yields atr=0 at the
    first bars and would neutralize the time penalty).
    """
    bars = make_bars(o, disc=disc)
    bars["atr5"] = np.full(bars["n"], float(atr_val), dtype=float)
    return bars


def r1_actions(bars):
    core = r1_oracle_core(bars, horizons=HORIZONS, hmax=HMAX)
    n = bars["n"]; disc = bars["disc"]
    out = np.full((n, 3), A_TIE, dtype=np.int8)
    for hi, H in enumerate(HORIZONS):
        for t in range(n):
            qlt = core["ql"][t, H]; qst = core["qs"][t, H]
            qwt = (0.0 if (t + 1 < n and bool(disc[t + 1]))
                   else core["Vflat"][t + 1, H - 1])
            a_l = qlt if np.isfinite(qlt) else NEG
            a_s = qst if np.isfinite(qst) else NEG
            act, _, _ = _classify({"Long": a_l, "Short": a_s, "Wait": qwt})
            out[t, hi] = {"Long": A_LONG, "Short": A_SHORT,
                          "Wait": A_WAIT, "Tie": A_TIE}[act]
    return out


def r2_theta0_actions(bars):
    cand = precompute_candidate_paths(bars)
    core = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    aH = actions_for_theta(core)
    out = np.full((bars["n"], 3), A_TIE, dtype=np.int8)
    for hi, H in enumerate(HORIZONS):
        out[:, hi] = aH[H]["action"]
    return out


# ---- T0.0 BASELINE PARITY GATE (hard stop if it fails) -----------------------
def _parity(bars, tag):
    core1 = r1_oracle_core(bars, horizons=HORIZONS, hmax=HMAX)
    cand = precompute_candidate_paths(bars)
    core2 = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    qm = int(np.abs(core1["ql"] - core2["QL"]).max() > TOL)
    sm = int(np.abs(core1["qs"] - core2["QS"]).max() > TOL)
    vm = int(np.abs(core1["Vflat"] - core2["Vflat"]).max() > TOL)
    a1 = r1_actions(bars)
    a2 = r2_theta0_actions(bars)
    am = int((a1 != a2).any())
    # stable_action parity
    def stable(a):
        s = []
        for t in range(a.shape[0]):
            u = np.unique(a[t])
            if len(u) == 1 and u[0] != A_TIE:
                s.append(u[0])
            else:
                s.append(9)
        return np.array(s)
    sm2 = int((stable(a1) != stable(a2)).any())
    print(f"[PARITY {tag}] Q_mismatch={qm} S_mismatch={sm} V_mismatch={vm} "
          f"action_mismatch={am} stable_mismatch={sm2}")
    return dict(qm=qm, sm=sm, vm=vm, am=am, sm2=sm2)


def test_t0_baseline_parity_synthetic():
    rng = np.random.default_rng(42)
    o = np.cumsum(rng.normal(0, 1, size=50)) + 100
    disc = rng.random(50) < 0.1
    bars = make_bars(o, disc=disc)
    r = _parity(bars, "synth")
    assert r["qm"] == 0 and r["sm"] == 0 and r["vm"] == 0, "QL/QS/Vflat mismatch"
    assert r["am"] == 0, "baseline action mismatch"
    assert r["sm2"] == 0, "baseline stable_action mismatch"


def test_t0_baseline_parity_real_ag():
    bars = build_bars("AG")
    keep = min(600, bars["n"])
    for k in ("o", "h", "l", "c", "disc", "atr5"):
        bars[k] = bars[k][-keep:]
    bars["t"] = bars["t"][-keep:]
    bars["decision_time"] = bars["decision_time"][-keep:]
    bars["n"] = keep
    r = _parity(bars, "AG")
    assert r["qm"] == 0 and r["sm"] == 0 and r["vm"] == 0
    assert r["am"] == 0 and r["sm2"] == 0


# ---- T0.1 risk penalty suppresses a large-MAE Long ---------------------------
def test_t0_risk_penalty_synthetic():
    # Long now: enter 100, bar-1 low=50 (MAE=50), exit 120 -> gross +20.
    # Waiting cannot beat it (later bars flat at 110 -> Short only +10).
    # theta=0 -> Long (+20 > Wait +10). Strong risk penalty -> Long U<0 -> Wait.
    o = np.array([100.0, 100, 120, 110, 110, 110, 110, 110])
    h = np.array([100.0, 100, 120, 110, 110, 110, 110, 110])
    l = np.array([100.0, 50, 120, 110, 110, 110, 110, 110])  # bar-1 low=50
    bars = make_bars(o, h=h, l=l)
    cand = precompute_candidate_paths(bars)
    c0 = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    cR = oracle_core_theta(cand, 1.00, 0.0, 0.0)  # strong risk penalty
    a0 = actions_for_theta(c0)[6]["action"][0]
    aR = actions_for_theta(cR)[6]["action"][0]
    assert a0 == A_LONG, f"baseline should be Long, got {a0}"
    assert aR != A_LONG, f"risk penalty must suppress the high-MAE Long, got {aR}"


# ---- T0.2 time penalty erodes value (vT <= v0, strict somewhere) -----------
def test_t0_time_penalty_synthetic():
    rng = np.random.default_rng(123)
    o = np.cumsum(rng.normal(0, 2.0, size=18)) + 100
    bars = make_bars(o)
    cand = precompute_candidate_paths(bars)
    c0 = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    cT = oracle_core_theta(cand, 0.0, 0.020, 0.0)
    v0 = np.maximum.reduce([c0["QL"][:, 12], c0["QS"][:, 12], c0["QW"][:, 12]])
    vT = np.maximum.reduce([cT["QL"][:, 12], cT["QS"][:, 12], cT["QW"][:, 12]])
    # time penalty never increases value; and strictly erodes where atr>0
    assert np.all(vT <= v0 + 1e-9), "time penalty cannot increase value"
    assert np.any(vT < v0 - 1e-6), "time penalty must strictly erode value somewhere"


# ---- T0.3 friction hurdle suppresses marginal trades -----------------------
def test_t0_friction_hurdle_synthetic():
    rng = np.random.default_rng(321)
    o = np.cumsum(rng.normal(0, 3.0, size=18)) + 100
    bars = make_bars(o)
    cand = precompute_candidate_paths(bars)
    c0 = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    cF = oracle_core_theta(cand, 0.0, 0.0, 0.100)  # big friction hurdle
    a0 = actions_for_theta(c0)[6]["action"]
    aF = actions_for_theta(cF)[6]["action"]
    # friction never increases value; and flips at least one trade->Wait/Tie
    v0 = np.maximum.reduce([c0["QL"][:, 12], c0["QS"][:, 12], c0["QW"][:, 12]])
    vF = np.maximum.reduce([cF["QL"][:, 12], cF["QS"][:, 12], cF["QW"][:, 12]])
    assert np.all(vF <= v0 + 1e-9)
    flipped = np.sum((a0 != A_TIE) & ((aF == A_WAIT) | (aF == A_TIE)))
    assert flipped > 0, "friction hurdle must suppress at least one trade"


# ---- T0.4 Wait time penalty: earlier opportunity preferred (holding) --------
def test_t0_wait_time_prefers_earlier():
    # Equal gross return (+30) at exit k=2 and at exit k=10, both MAE=0.
    # The time penalty is lambda_T*ATR_t*k, so the per-exit utility strictly
    # DECREASES with the holding length k. The oracle must therefore choose the
    # EARLIER exit (k=2), never the later k=10.
    n = 16
    o = np.ones(n) * 100.0
    o[3] = 130.0    # exit-open index (t+1)+k = 1+2 = 3 -> +30 at k=2
    o[11] = 130.0   # (t+1)+k = 1+10 = 11      -> +30 at k=10
    bars = _bars_with_atr(o, 10.0)   # constant ATR=10, no warmup zero
    cand = precompute_candidate_paths(bars)
    c0 = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    cT = oracle_core_theta(cand, 0.0, 0.010, 0.0)
    # chosen holding at decision t=0 must be the earlier exit (k=2)
    assert c0["hold_l"][0] == 2, f"baseline holding must be 2, got {c0['hold_l'][0]}"
    assert cT["hold_l"][0] == 2, f"time penalty must keep earlier exit, got {cT['hold_l'][0]}"
    # raw per-exit utilities confirm the later exit is strictly penalized
    u2 = 30.0 - 0.010 * 10.0 * 2
    u10 = 30.0 - 0.010 * 10.0 * 10
    assert u10 < u2 - 1e-9, "later exit must be strictly worse under time penalty"
    assert abs(cT["QL"][0, 2] - u2) < 1e-6


# ---- T0.5 discontinuity blocks holding --------------------------------------
def test_t0_discontinuity():
    n = 21
    o = np.ones(n) * 100.0
    o[10:] = 200.0
    disc = np.zeros(n, bool)
    disc[10] = True
    bars = make_bars(o, disc=disc)
    cand = precompute_candidate_paths(bars)
    c0 = oracle_core_theta(cand, 0.0, 0.0, 0.0)
    # held path cannot cross disc[10]; QL at t=0 stays 0
    assert c0["QL"][0, 24] == 0.0
    assert c0["Vflat"][0, 24] == 0.0


# ---- T0.6 flat market -> Tie (all thetas) -----------------------------------
def test_t0_tie_flat():
    o = np.ones(20) * 50.0
    bars = make_bars(o)
    cand = precompute_candidate_paths(bars)
    for th in THETAS:
        c = oracle_core_theta(cand, *th[:3])
        a = actions_for_theta(c)[6]["action"][0]
        assert a == A_TIE, f"flat market must be Tie for {th}"


# ---- T0.7 symmetry under price negation -------------------------------------
def test_t0_symmetry():
    rng = np.random.default_rng(7)
    o = np.cumsum(rng.normal(0, 1, size=60)) + 100
    bars = make_bars(o)
    cb = precompute_candidate_paths(bars)
    cnb = precompute_candidate_paths(make_bars(-o))
    opp = {A_LONG: A_SHORT, A_SHORT: A_LONG, A_WAIT: A_WAIT, A_TIE: A_TIE}
    for th in CORE:
        a = actions_for_theta(oracle_core_theta(cb, *th[:3]))[6]["action"]
        b = actions_for_theta(oracle_core_theta(cnb, *th[:3]))[6]["action"]
        assert np.array_equal(a, [opp[x] for x in b]) or True  # per-row check below
        for t in range(bars["n"]):
            assert a[t] == opp[b[t]], (th, t, a[t], b[t])


# ---- T0.8 ATR-scale invariance (uniform price scale -> same actions) --------
def test_t0_atr_scale_invariance():
    rng = np.random.default_rng(11)
    o = np.cumsum(rng.normal(0, 2, size=40)) + 500
    bars = make_bars(o)
    cand = precompute_candidate_paths(bars)
    f = 3.7  # scale prices AND atr by same factor
    bars_s = make_bars(o * f)
    # make_bars recomputes atr5 from scaled bars -> atr also scales by f
    cand_s = precompute_candidate_paths(bars_s)
    for th in CORE:
        a = actions_for_theta(oracle_core_theta(cand, *th[:3]))[6]["action"]
        b = actions_for_theta(oracle_core_theta(cand_s, *th[:3]))[6]["action"]
        assert np.array_equal(a, b), (th, "scale broke action invariance")


# ---- T0.9 brute-force differential (>=100 random, multi theta) --------------
def test_t0_brute_force_differential():
    rng = np.random.default_rng(2026)
    mism = 0
    checked = 0
    for _ in range(120):
        n = int(rng.integers(10, 21))
        H = int(rng.integers(3, 7))
        o = np.cumsum(rng.normal(0, 1.0, size=n)) + 100
        disc = rng.random(n) < 0.12
        bars = make_bars(o, disc=disc)
        th = THETAS[int(rng.integers(0, len(THETAS)))]
        lam_r, lam_t, c, _ = th
        core = oracle_core_theta(precompute_candidate_paths(bars),
                                 lam_r, lam_t, c)
        aH = actions_for_theta(core)
        for _ in range(3):
            t = int(rng.integers(0, max(1, n - 2)))
            if t + 2 >= n:
                continue
            bf = brute_force_oracle_r2(bars, t, H, lam_r, lam_t, c)
            # R2 Q values at this (t,H)
            ql = core["QL"][t, H]
            qs = core["QS"][t, H]
            qw = core["QW"][t, H]
            a_l = ql if np.isfinite(ql) else NEG
            a_s = qs if np.isfinite(qs) else NEG
            act_code = classify_batch(np.array([a_l]), np.array([a_s]),
                                     np.array([qw]))[0][0]
            code_map = {A_LONG: "Long", A_SHORT: "Short",
                        A_WAIT: "Wait", A_TIE: "Tie"}
            act_prod = code_map[act_code]
            checked += 1

            def _close(a, b):
                if not np.isfinite(a) and not np.isfinite(b):
                    return True
                return abs(a - b) <= 1e-6

            if not (_close(ql, bf["QL"]) and _close(qs, bf["QS"])
                    and _close(qw, bf["QW"]) and act_prod == bf["action"]):
                mism += 1
    assert checked >= 100, f"too few: {checked}"
    assert mism == 0, f"brute-force mismatch = {mism} / {checked}"


# ---- T0.10 time preference: equal return, earlier strictly preferred --------
def test_t0_time_prefers_earlier_equal_return():
    # Two identical +30 opportunities, but one is reachable EARLY (exit k=2) and
    # the other only LATE (exit k=10). With lambda_T>0 the oracle's value at the
    # decision bar must be strictly higher for the early case, because the wait
    # chain pays lambda_T*ATR per waited bar. With lambda_T=0 both equal +30.
    def build(near):
        n = 16
        o = np.ones(n) * 100.0
        if near:
            o[3] = 130.0     # early opportunity (exit k=2)
        else:
            o[11] = 130.0    # late opportunity (exit k=10)
        return _bars_with_atr(o, 10.0)

    barsN = build(near=True)
    barsF = build(near=False)
    candN = precompute_candidate_paths(barsN)
    candF = precompute_candidate_paths(barsF)
    c0N = oracle_core_theta(candN, 0.0, 0.0, 0.0)
    c0F = oracle_core_theta(candF, 0.0, 0.0, 0.0)
    cTN = oracle_core_theta(candN, 0.0, 0.010, 0.0)
    cTF = oracle_core_theta(candF, 0.0, 0.010, 0.0)
    # lambda_T=0: identical value (both capture +30, no time cost)
    assert abs(c0N["Vflat"][0, 12] - c0F["Vflat"][0, 12]) < 1e-6
    # lambda_T>0: early opportunity strictly dominates the late one
    vN = cTN["Vflat"][0, 12]
    vF = cTF["Vflat"][0, 12]
    assert vN > vF + 1e-6, f"earlier opportunity must dominate: {vN} vs {vF}"


# ---- T1 real AG, all 34 thetas, all 3 horizons ------------------------------
def test_t1_real_ag_all_thetas():
    from research.liquidity_oracle_atlas.build_oracle_constraint_robustness_v1 import (
        evaluate_symbol)
    bars = build_bars("AG")
    keep = min(800, bars["n"])
    for k in ("o", "h", "l", "c", "disc", "atr5"):
        bars[k] = bars[k][-keep:]
    bars["t"] = bars["t"][-keep:]
    bars["decision_time"] = bars["decision_time"][-keep:]
    bars["n"] = keep
    cand = precompute_candidate_paths(bars)
    rows, mat, counters, nv = evaluate_symbol(bars, cand)
    assert nv > 0
    assert mat.shape == (nv, len(THETAS), 3)
    df = pd.DataFrame(rows)
    # sample categories required by spec
    cat = dict(
        robust_Long=int((df["strict_robust_action"] == "Long").sum()),
        robust_Short=int((df["strict_robust_action"] == "Short").sum()),
        robust_Wait=int((df["strict_robust_action"] == "Wait").sum()),
        baseline_Long_suppressed=int(((df["baseline_stable_action"] == "Long") &
                                      (df["Long_share_H24"] < 1.0)).sum()),
        baseline_Wait_stays=int(((df["baseline_stable_action"] == "Wait") &
                                 (df["Wait_share_H24"] >= 0.5)).sum()),
    )
    print(f"[T1] AG tail={keep} rows={nv} samples={cat}")
    assert cat["robust_Long"] + cat["robust_Short"] + cat["robust_Wait"] > 0
    # all theta count sanity
    assert counters["act_hist"].shape == (len(THETAS), 3, 4)


if __name__ == "__main__":
    test_t0_baseline_parity_synthetic()
    test_t0_baseline_parity_real_ag()
    test_t0_risk_penalty_synthetic()
    test_t0_time_penalty_synthetic()
    test_t0_friction_hurdle_synthetic()
    test_t0_wait_time_prefers_earlier()
    test_t0_discontinuity()
    test_t0_tie_flat()
    test_t0_symmetry()
    test_t0_atr_scale_invariance()
    test_t0_brute_force_differential()
    test_t0_time_prefers_earlier_equal_return()
    test_t1_real_ag_all_thetas()
    print("ALL T0/T1 PASSED")
