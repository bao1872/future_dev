"""T0 (unit) + T1 (small real) tests for Robust 5m Trade Oracle DP v1.0.

Run: python -m pytest research/liquidity_oracle_atlas/test_robust_trade_oracle_dp_v1.py -q
  or: python research/liquidity_oracle_atlas/test_robust_trade_oracle_dp_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_robust_trade_oracle_dp_v1 import (
    oracle_core, build_rows, compute_oracle, HORIZONS, HMAX,
)

TOL = 1e-6


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
    seg = np.concatenate([[0], np.cumsum(disc)]).astype(int)[:-1]
    t = pd.date_range("2025-01-02 09:00", periods=n, freq="5min").to_numpy()
    return dict(symbol=sym, o=o, h=h, l=l, c=o.copy(), t=t, disc=disc, seg=seg,
                n=n)


def _check(core, t, H, ql, qs, action, vflat=None):
    assert core["ql"][t, H] == np.float64(ql), \
        f"QL({t},{H})={core['ql'][t,H]} != {ql}"
    assert core["qs"][t, H] == np.float64(qs), \
        f"QS({t},{H})={core['qs'][t,H]} != {qs}"
    assert core["bestA"][t, H] == action, \
        f"action({t},{H})={core['bestA'][t,H]} != {action}"
    if vflat is not None:
        assert core["Vflat"][t, H] == np.float64(vflat), \
            f"Vflat({t},{H})={core['Vflat'][t,H]} != {vflat}"


# ---- T0.1 monotone up -> Long ----------------------------------------------
def test_t0_monotone_up():
    o = np.arange(1, 31, dtype=float)            # 1..30
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(6,), hmax=HMAX)
    # t=0: e=1, o[e]=2; QL(0,6)=o[7]-o[1]=8-2=6;
    # QS(0,6)=best short exit = immediate h=1 -> 2-3 = -1 (price only rises);
    # QW(0,6)=Vflat[1][5]=QL(1,5)=o[7]-o[2]=8-3=5. best=Long, Vflat=6,
    # edge = 6 - max(QS=-1, QW=5) = 1.
    _check(core, 0, 6, 6.0, -1.0, "Long", vflat=6.0)
    df = build_rows(bars, core)
    r = df[df["decision_bar_index"] == 0].iloc[0]
    assert r["action_6"] == "Long"
    assert abs(r["QL_6"] - 6.0) < TOL
    assert abs(r["QS_6"] + 1.0) < TOL
    assert abs(r["edge_6"] - 1.0) < TOL
    # next-open execution + MFE/MAE interval (monotone: MAE==0, MFE==realized)
    assert abs(r["realized_move_6"] - (o[int(r["exit_bars_6"])] - o[1])) < TOL
    assert abs(r["MFE_6"] - r["realized_move_6"]) < TOL
    assert abs(r["MAE_6"] - 0.0) < TOL


# ---- T0.2 monotone down -> Short -------------------------------------------
def test_t0_monotone_down():
    o = np.arange(30, 0, -1, dtype=float)         # 30..1
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(6,), hmax=HMAX)
    # t=0: e=1, o[e]=29; price only falls: QL best long exit = immediate h=1
    # -> 28-29 = -1; QS(0,6)=29-o[7]=29-23=6; action Short; Vflat=6.
    _check(core, 0, 6, -1.0, 6.0, "Short", vflat=6.0)


# ---- T0.3 oscillate-then-rise -> Wait owns option value --------------------
def test_t0_wait_option_value():
    # enter now Long loses a little; enter now Short gains a little; but
    # waiting to enter Long at the dip (98) then exit at 200 yields +102,
    # which beats both immediate actions at t=0.
    o = np.array([100.0, 100, 98, 98, 200, 200, 200, 200])
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(4,), hmax=HMAX)
    # QL(0,4): enter 100, best exit 200 (h=3) -> +100
    # QS(0,4): short enter 100 exit 98 -> +2
    # QW(0,4)=Vflat[1][3]=102 (delayed Long: enter 98 exit 200)
    assert core["ql"][0, 4] == 100.0
    assert core["qs"][0, 4] == 2.0
    assert core["Vflat"][1, 3] == 102.0
    assert core["bestA"][0, 4] == "Wait", core["bestA"][0, 4]
    assert core["Vflat"][0, 4] == 102.0
    df = build_rows(bars, core)
    r = df[df["decision_bar_index"] == 0].iloc[0]
    assert r["action_4"] == "Wait"
    assert abs(r["QW_4"] - 102.0) < TOL


# ---- T0.4 discontinuity blocks the path ------------------------------------
def test_t0_discontinuity_blocks():
    n = 21
    o = np.ones(n) * 100.0
    o[10:] = 200.0
    disc = np.zeros(n, bool)
    disc[10] = True
    bars = make_bars(o, disc=disc)
    core = oracle_core(bars, horizons=(24,), hmax=HMAX)
    # at t=0 the held window cannot cross disc[10]; best exit stays flat -> 0
    assert core["ql"][0, 24] == 0.0, core["ql"][0, 24]
    # control: no discontinuity -> captures the 200 jump (+100)
    bars2 = make_bars(o, disc=np.zeros(n, bool))
    core2 = oracle_core(bars2, horizons=(24,), hmax=HMAX)
    assert core2["ql"][0, 24] == 100.0, core2["ql"][0, 24]


# ---- T0.5 symmetry under price negation ------------------------------------
def test_t0_symmetry():
    rng = np.random.default_rng(0)
    o = np.cumsum(rng.normal(0, 1, size=60)) + 100
    bars = make_bars(o)
    core = oracle_core(bars, horizons=HORIZONS, hmax=HMAX)
    neg = make_bars(-o)
    cneg = oracle_core(neg, horizons=HORIZONS, hmax=HMAX)
    opp = {"Long": "Short", "Short": "Long", "Wait": "Wait"}
    for t in range(bars["n"]):
        for H in HORIZONS:
            a1, b1 = cneg["ql"][t, H], core["qs"][t, H]
            a2, b2 = cneg["qs"][t, H], core["ql"][t, H]
            # both -inf (no valid hold) counts as equal
            if not (np.isneginf(a1) and np.isneginf(b1)):
                assert abs(a1 - b1) < 1e-6, (t, H, a1, b1)
            if not (np.isneginf(a2) and np.isneginf(b2)):
                assert abs(a2 - b2) < 1e-6, (t, H, a2, b2)
            assert cneg["bestA"][t, H] == opp[core["bestA"][t, H]], (t, H)


# ---- T0.6 horizon tail / data end does not crash ---------------------------
def test_t0_horizon_tail_no_crash():
    o = np.arange(1, 6, dtype=float)             # n=5
    bars = make_bars(o)
    core = oracle_core(bars, horizons=HORIZONS, hmax=HMAX)
    df = build_rows(bars, core)
    assert len(df) > 0
    for _, r in df.iterrows():
        for H in HORIZONS:
            a = r[f"action_{H}"]
            assert a in ("Long", "Short", "Wait")
            assert np.isfinite(r[f"edge_{H}"])
            # edge == best - second
            vals = {"Long": r[f"QL_{H}"], "Short": r[f"QS_{H}"],
                    "Wait": r[f"QW_{H}"]}
            vals = {k: (v if v is not None else -np.inf) for k, v in
                    vals.items()}
            best = max(vals.values())
            second = sorted(vals.values(), reverse=True)[1]
            assert abs((best - second) - r[f"edge_{H}"]) < TOL


# ---- T1 real data, 1 symbol, short window, all horizons --------------------
def test_t1_real_ag_invariant():
    from research.liquidity_oracle_atlas.build_robust_trade_oracle_dp_v1 \
        import build_bars
    sym = "AG"
    bars = build_bars(sym)
    keep = min(800, bars["n"])
    for k in ("o", "h", "l", "c", "disc", "seg"):
        bars[k] = bars[k][-keep:]
    bars["t"] = bars["t"][-keep:]
    bars["n"] = keep

    df, core = compute_oracle(bars, horizons=HORIZONS, hmax=HMAX)
    assert len(df) > 0, "no rows produced"
    disc = bars["disc"]
    o = bars["o"]

    for _, r in df.iterrows():
        e = int(r["entry_bar_index"])
        for H in HORIZONS:
            a = r[f"action_{H}"]
            assert a in ("Long", "Short", "Wait")
            # edge == best - second (recompute from Q columns)
            vals = {"Long": r[f"QL_{H}"], "Short": r[f"QS_{H}"],
                    "Wait": r[f"QW_{H}"]}
            vals = {k: (v if v is not None else -np.inf) for k, v in
                    vals.items()}
            best = max(vals.values())
            second = sorted(vals.values(), reverse=True)[1]
            assert abs((best - second) - r[f"edge_{H}"]) < TOL, \
                f"edge mismatch H={H}"
            if a in ("Long", "Short"):
                xb = int(r[f"exit_bars_{H}"])
                # no discontinuity crossed in [e, xb]
                assert not disc[e:xb + 1].any(), \
                    f"crossed discontinuity e={e} xb={xb}"
                exp = (o[xb] - o[e]) if a == "Long" else (o[e] - o[xb])
                assert abs(exp - r[f"realized_move_{H}"]) < TOL, \
                    "next-open execution mismatch"
                assert r[f"MFE_{H}"] >= r[f"realized_move_{H}"] - TOL, \
                    "MFE must bracket realized"
                hb = int(r[f"holding_bars_{H}"])
                assert 1 <= hb <= H, f"holding_bars out of range: {hb}"
            else:
                assert pd.isna(r[f"exit_bars_{H}"])
                assert pd.isna(r[f"holding_bars_{H}"])
                assert pd.isna(r[f"realized_move_{H}"])
        # agreement / stable label well-formed (rounded to 4 dp)
        assert r["agreement"] in (0.3333, 0.6667, 1.0)
        assert r["stable_action"] in ("Long", "Short", "Wait", "Ambiguous")
    print(f"[T1] AG tail={keep} rows={len(df)} "
          f"stable={df['stable_action'].value_counts().to_dict()}")


if __name__ == "__main__":
    test_t0_monotone_up()
    test_t0_monotone_down()
    test_t0_wait_option_value()
    test_t0_discontinuity_blocks()
    test_t0_symmetry()
    test_t0_horizon_tail_no_crash()
    test_t1_real_ag_invariant()
    print("ALL T0/T1 TESTS PASSED")
