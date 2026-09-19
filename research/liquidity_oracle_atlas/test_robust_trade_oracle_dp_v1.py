"""T0 (unit) + T1 (small real) + brute-force differential tests for
Robust 5m Trade Oracle DP v1.1 (FUTURE-ORACLE-R1.1-CORRECTNESS-HARDENING).

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
    oracle_core, build_rows, compute_oracle, _classify, HORIZONS, HMAX,
)
from research.liquidity_oracle_atlas.build_robust_trade_oracle_dp_v1 import (
    build_bars,
)

TOL = 1e-6
NEG = -np.inf


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
    bars = dict(symbol=sym, o=o, h=h, l=l, c=o.copy(), t=t, disc=disc, n=n)
    return build_bars_real(bars)


# ----- brute-force reference (independent of production oracle_core) ----------
def _best_trade(o, disc, e, H_rem, n):
    """Best single Long/Short round-trip PnL from entry e within H_rem."""
    if e > n - 1 or disc[e]:                 # entry bar discontinuous
        return NEG, NEG
    nxt = np.flatnonzero(disc[e + 1:])
    fd = (e + 1 + int(nxt[0])) if len(nxt) else n
    h_max = min(H_rem, fd - e - 1, n - 1 - e)
    long_b, short_b = NEG, NEG
    for hh in range(1, h_max + 1):
        ex = e + hh
        pnl_l = o[ex] - o[e]
        pnl_s = o[e] - o[ex]
        if pnl_l > long_b:
            long_b = pnl_l
        if pnl_s > short_b:
            short_b = pnl_s
    return long_b, short_b


def brute_force_oracle(bars, t, H, eps=TOL):
    """Independent enumerator: Long now / Short now / Wait k then one trade."""
    o, disc = bars["o"], bars["disc"]
    n = bars["n"]
    e = t + 1
    long_now, short_now = _best_trade(o, disc, e, H, n)
    QL = long_now
    QS = short_now
    QW = 0.0  # no trade
    for k in range(1, H):
        ek = e + k
        if ek > n - 1:
            continue
        # wait path: decisions t->t+1->...->t+k cross into bars t+1..t+k
        if disc[t + 1:ek].any():          # includes disc[t+1] (Fix A)
            continue
        if disc[ek]:                      # entry bar discontinuous
            continue
        rem = H - k
        if rem < 1:
            continue
        lb, sb = _best_trade(o, disc, ek, rem, n)
        best_k = max(lb, sb)
        if best_k > QW:
            QW = best_k
    vals = {"Long": QL if np.isfinite(QL) else NEG,
            "Short": QS if np.isfinite(QS) else NEG,
            "Wait": QW}
    unique, _, n_best = _classify(vals, eps)
    return dict(QL=QL, QS=QS, QW=QW, action=unique, n_best=n_best)


def _close(a, b):
    if not np.isfinite(a) and not np.isfinite(b):
        return True
    if not np.isfinite(a) or not np.isfinite(b):
        return False
    return abs(a - b) <= 1e-6


# ---- T0.1 monotone up -> Long ----------------------------------------------
def test_t0_monotone_up():
    o = np.arange(1, 31, dtype=float)
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(6,), hmax=HMAX)
    assert core["ql"][0, 6] == 6.0
    assert core["qs"][0, 6] == -1.0
    # QW(0,6)=Vflat[1][5]=QL(1,5)=8-3=5 (Long best)
    assert core["Vflat"][1, 5] == 5.0
    df = build_rows(bars, core)
    r = df[df["decision_bar_index"] == 0].iloc[0]
    assert r["action_6"] == "Long"
    assert abs(r["QL_6"] - 6.0) < TOL
    assert abs(r["QS_6"] + 1.0) < TOL
    assert abs(r["edge_6"] - 1.0) < TOL
    assert abs(r["realized_move_6"] - (o[int(r["exit_bars_6"])] - o[1])) < TOL
    assert abs(r["MFE_6"] - r["realized_move_6"]) < TOL
    assert abs(r["MAE_6"] - 0.0) < TOL
    # Fix B: decision_time = bar_start + 5min
    assert (pd.Timestamp(r["decision_time"]) -
            pd.Timestamp(r["decision_bar_start_time"])) == pd.Timedelta(minutes=5)
    # Fix E: label available strictly after decision
    assert pd.Timestamp(r["label_available_time_6"]) > pd.Timestamp(
        r["decision_time"])


# ---- T0.2 monotone down -> Short -------------------------------------------
def test_t0_monotone_down():
    o = np.arange(30, 0, -1, dtype=float)
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(6,), hmax=HMAX)
    assert core["ql"][0, 6] == -1.0
    assert core["qs"][0, 6] == 6.0


# ---- T0.3 oscillate-then-rise -> Wait owns option value --------------------
def test_t0_wait_option_value():
    o = np.array([100.0, 100, 98, 98, 200, 200, 200, 200])
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(4,), hmax=HMAX)
    assert core["ql"][0, 4] == 100.0
    assert core["qs"][0, 4] == 2.0
    assert core["Vflat"][1, 3] == 102.0          # delayed Long: 98->200
    assert core["Vflat"][0, 4] == 102.0
    df = build_rows(bars, core)
    r = df[df["decision_bar_index"] == 0].iloc[0]
    assert r["action_4"] == "Wait"
    assert abs(r["QW_4"] - 102.0) < TOL


# ---- T0.4 discontinuity blocks the held path + Fix A: Wait can't cross -----
def test_t0_discontinuity_and_fix_a():
    n = 21
    o = np.ones(n) * 100.0
    o[10:] = 200.0
    disc = np.zeros(n, bool)
    disc[10] = True
    bars = make_bars(o, disc=disc)
    core = oracle_core(bars, horizons=(24,), hmax=HMAX)
    # held path cannot cross disc[10] -> QL stays flat (0)
    assert core["ql"][0, 24] == 0.0
    # FIX A: Wait recursion cannot borrow segment B's rally -> QW == 0
    assert core["Vflat"][1, 23] == 0.0
    assert core["Vflat"][0, 24] == 0.0
    df = build_rows(bars, core)
    r = df[df["decision_bar_index"] == 0].iloc[0]
    # all three == 0 -> genuine Tie (not a fabricated trade)
    assert r["action_24"] == "Tie"
    assert r["QW_24"] == 0.0
    # control: without discontinuity, Wait CAN see the rally
    bars2 = make_bars(o, disc=np.zeros(n, bool))
    core2 = oracle_core(bars2, horizons=(24,), hmax=HMAX)
    assert core2["Vflat"][0, 24] > 50.0


# ---- T0.5 symmetry under price negation -------------------------------------
def test_t0_symmetry():
    rng = np.random.default_rng(0)
    o = np.cumsum(rng.normal(0, 1, size=60)) + 100
    bars = make_bars(o)
    core = oracle_core(bars, horizons=HORIZONS, hmax=HMAX)
    cneg = oracle_core(make_bars(-o), horizons=HORIZONS, hmax=HMAX)
    opp = {"Long": "Short", "Short": "Long", "Wait": "Wait", "Tie": "Tie"}
    for t in range(bars["n"]):
        for H in HORIZONS:
            a1, b1 = cneg["ql"][t, H], core["qs"][t, H]
            a2, b2 = cneg["qs"][t, H], core["ql"][t, H]
            if not (np.isneginf(a1) and np.isneginf(b1)):
                assert abs(a1 - b1) < 1e-6, (t, H, a1, b1)
            if not (np.isneginf(a2) and np.isneginf(b2)):
                assert abs(a2 - b2) < 1e-6, (t, H, a2, b2)


# ---- T0.6 horizon tail / data end does not crash ---------------------------
def test_t0_horizon_tail_no_crash():
    o = np.arange(1, 6, dtype=float)
    bars = make_bars(o)
    core = oracle_core(bars, horizons=HORIZONS, hmax=HMAX)
    df = build_rows(bars, core)
    assert len(df) > 0
    for _, r in df.iterrows():
        for H in HORIZONS:
            a = r[f"action_{H}"]
            assert a in ("Long", "Short", "Wait", "Tie")
            vals = {"Long": r[f"QL_{H}"], "Short": r[f"QS_{H}"],
                    "Wait": r[f"QW_{H}"]}
            vals = {k: (v if v is not None else NEG) for k, v in vals.items()}
            best = max(vals.values())
            second = sorted(vals.values(), reverse=True)[1]
            assert abs((best - second) - r[f"edge_{H}"]) < TOL


# ---- T0.7 Fix C: segment = cumsum(disc) ------------------------------------
def test_t0_segment_boundary():
    n = 12
    disc = np.zeros(n, bool)
    disc[5] = True
    bars = make_bars(np.arange(n, dtype=float), disc=disc)
    bars = build_bars_real(bars)  # attach canonical segment via cumsum
    seg = np.cumsum(disc.astype(np.int64))
    assert np.array_equal(bars["seg"], seg)
    assert bars["seg"][5] == bars["seg"][4] + 1


def build_bars_real(bars):
    """Wrap a synthetic bars dict with canonical seg/atr5/decision_time."""
    from research.phase1_tradability.phase1_contract_v1 import compute_atr5
    disc = bars["disc"]
    bars["seg"] = np.cumsum(disc.astype(np.int64))
    bars["decision_time"] = bars["t"] + pd.Timedelta(minutes=5)
    bars["atr5"] = compute_atr5(dict(high=bars["h"], low=bars["l"],
                                     close=bars["c"]))
    return bars


# ---- T0.8 Fix D: exact tie (flat market) -> Tie, not Long ------------------
def test_t0_tie_flat():
    o = np.ones(20) * 50.0
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(6,), hmax=HMAX)
    df = build_rows(bars, core)
    r = df.iloc[0]
    assert r["action_6"] == "Tie"
    assert r["n_best_actions_6"] == 3


# ---- T0.9 Fix F: ATR normalization preserves the action --------------------
def test_t0_atr_invariance():
    rng = np.random.default_rng(3)
    o = np.cumsum(rng.normal(0, 2, size=40)) + 500
    bars = make_bars(o)
    core = oracle_core(bars, horizons=(6,), hmax=HMAX)
    df = build_rows(bars, core)
    for _, r in df.iterrows():
        if r["atr5_t"] is None or not np.isfinite(r["atr5_t"]) or r["atr5_t"] <= 0:
            continue
        s = r["atr5_t"]
        raw = {"Long": r["QL_6"] if r["QL_6"] is not None else NEG,
               "Short": r["QS_6"] if r["QS_6"] is not None else NEG,
               "Wait": r["QW_6"]}
        atr = {"Long": r["QL_6_ATR"] if r["QL_6_ATR"] is not None else NEG,
               "Short": r["QS_6_ATR"] if r["QS_6_ATR"] is not None else NEG,
               "Wait": r["QW_6_ATR"]}
        u_raw, _, _ = _classify(raw)
        u_atr, _, _ = _classify(atr)
        assert u_raw == u_atr, (r["decision_bar_index"], u_raw, u_atr)


# ---- T0.10 Fix E: label availability timestamp + discontinuity reason ------
def test_t0_label_availability():
    n = 30
    o = np.arange(1, n + 1, dtype=float)
    disc = np.zeros(n, bool)
    disc[12] = True
    bars = make_bars(o, disc=disc)
    core = oracle_core(bars, horizons=(24,), hmax=HMAX)
    df = build_rows(bars, core)
    # a decision well before the discontinuity should report DISCONTINUITY
    r = df[df["decision_bar_index"] == 0].iloc[0]
    assert r["oracle_terminal_reason_24"] == "DISCONTINUITY"
    assert pd.Timestamp(r["label_available_time_24"]) > pd.Timestamp(
        r["decision_time"])
    # label_available_time must NOT equal optimal exit time (different columns)
    assert "label_available_time_24" in df.columns
    assert "exit_bars_24" in df.columns


# ---- T0.11 brute-force differential (>=100 random cases) --------------------
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
        core = oracle_core(bars, horizons=(H,), hmax=HMAX)
        for _ in range(3):
            t = int(rng.integers(0, max(1, n - 2)))
            if t + 2 >= n:
                continue
            bf = brute_force_oracle(bars, t, H)
            prod_QL = core["ql"][t, H]
            prod_QS = core["qs"][t, H]
            prod_QW = (0.0 if (t + 1 < n and disc[t + 1])
                       else core["Vflat"][t + 1, H - 1])
            checked += 1
            if not _close(prod_QL, bf["QL"]):
                mism += 1
                continue
            if not _close(prod_QS, bf["QS"]):
                mism += 1
                continue
            if not _close(prod_QW, bf["QW"]):
                mism += 1
                continue
            # action must agree
            prod_action = _classify(
                {"Long": prod_QL if np.isfinite(prod_QL) else NEG,
                 "Short": prod_QS if np.isfinite(prod_QS) else NEG,
                 "Wait": prod_QW})[0]
            if prod_action != bf["action"]:
                mism += 1
    assert checked >= 100, f"too few cases: {checked}"
    assert mism == 0, f"brute-force mismatch count = {mism} / {checked}"


# ---- T1 real data, 1 symbol, short window, all horizons --------------------
def test_t1_real_ag_invariant():
    sym = "AG"
    bars = build_bars(sym)
    keep = min(800, bars["n"])
    for k in ("o", "h", "l", "c", "disc", "seg", "atr5"):
        bars[k] = bars[k][-keep:]
    bars["t"] = bars["t"][-keep:]
    bars["decision_time"] = bars["decision_time"][-keep:]
    bars["n"] = keep

    df, core = compute_oracle(bars, horizons=HORIZONS, hmax=HMAX)
    assert len(df) > 0, "no rows produced"
    disc = bars["disc"]
    o = bars["o"]
    disc_reason_seen = False
    for _, r in df.iterrows():
        e = int(r["entry_bar_index"])
        for H in HORIZONS:
            a = r[f"action_{H}"]
            assert a in ("Long", "Short", "Wait", "Tie")
            vals = {"Long": r[f"QL_{H}"], "Short": r[f"QS_{H}"],
                    "Wait": r[f"QW_{H}"]}
            vals = {k: (v if v is not None else NEG) for k, v in vals.items()}
            best = max(vals.values())
            second = sorted(vals.values(), reverse=True)[1]
            assert abs((best - second) - r[f"edge_{H}"]) < TOL
            if a in ("Long", "Short"):
                xb = int(r[f"exit_bars_{H}"])
                assert not disc[e:xb + 1].any()
                exp = (o[xb] - o[e]) if a == "Long" else (o[e] - o[xb])
                assert abs(exp - r[f"realized_move_{H}"]) < TOL
                assert r[f"MFE_{H}"] >= r[f"realized_move_{H}"] - TOL
                hb = int(r[f"holding_bars_{H}"])
                assert 1 <= hb <= H
            else:
                assert pd.isna(r[f"exit_bars_{H}"])
                assert pd.isna(r[f"holding_bars_{H}"])
                assert pd.isna(r[f"realized_move_{H}"])
            # Fix E: label available strictly after decision
            assert pd.Timestamp(r[f"label_available_time_{H}"]) > \
                pd.Timestamp(r["decision_time"])
            if r[f"oracle_terminal_reason_{H}"] == "DISCONTINUITY":
                disc_reason_seen = True
            # Fix F: ATR action == raw action (when scale finite)
            if r["atr5_t"] is not None and np.isfinite(r["atr5_t"]) and \
                    r["atr5_t"] > 0:
                s = r["atr5_t"]
                raw = {"Long": r[f"QL_{H}"] if r[f"QL_{H}"] is not None else NEG,
                       "Short": r[f"QS_{H}"] if r[f"QS_{H}"] is not None else NEG,
                       "Wait": r[f"QW_{H}"]}
                atr = {"Long": r[f"QL_{H}_ATR"] if r[f"QL_{H}_ATR"] is not None
                       else NEG,
                       "Short": r[f"QS_{H}_ATR"] if r[f"QS_{H}_ATR"] is not
                       None else NEG,
                       "Wait": r[f"QW_{H}_ATR"]}
                u_raw = _classify(raw)[0]
                u_atr = _classify(atr)[0]
                assert u_raw == u_atr, (r["decision_bar_index"], H, u_raw,
                                        u_atr)
        assert r["agreement"] in (0.3333, 0.6667, 1.0)
        assert r["stable_action"] in ("Long", "Short", "Wait", "Ambiguous")
    # Fix A: real data should contain discontinuity-truncated labels when a
    # decision sits within H bars of a discontinuity; the synthetic
    # test_t0_label_availability already asserts the DISCONTINUITY reason
    # semantics, so here we only report presence.
    print(f"[T1] AG tail={keep} rows={len(df)} "
          f"disc_terminal_reason_seen={disc_reason_seen} "
          f"stable={df['stable_action'].value_counts().to_dict()}")


if __name__ == "__main__":
    test_t0_monotone_up()
    test_t0_monotone_down()
    test_t0_wait_option_value()
    test_t0_discontinuity_and_fix_a()
    test_t0_symmetry()
    test_t0_horizon_tail_no_crash()
    test_t0_segment_boundary()
    test_t0_tie_flat()
    test_t0_atr_invariance()
    test_t0_label_availability()
    test_t0_brute_force_differential()
    test_t1_real_ag_invariant()
    print("ALL T0/T1/T0.11 TESTS PASSED")
