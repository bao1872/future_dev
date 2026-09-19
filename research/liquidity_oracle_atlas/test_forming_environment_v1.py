"""
T0 / T1 / T1.5 tests for R3A forming MTF environment.

Task: FUTURE-ENV-R3A-FORMING-MTF-ENVIRONMENT
Base: a84e74407bdd303d1ef7a236654141bc919331d0

These tests verify CAUSAL CORRECTNESS of the forming environment (not predictive
power). The slow reference (independent reconstruction from raw<=t) is the
oracle for the production-vs-reference differential (T1).

Run:  python3.11 research/liquidity_oracle_atlas/test_forming_environment_v1.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    DTP_AUDIT_COLS,
    DTP_COLS,
    LIQ_COLS,
    LIQ_PRICE_COLS,
    SR_COLS,
    SR_PRICE_COLS,
    FormingEnvironmentBuilder,
)
import research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 as pgm

ABS_TOL = 1e-6

# columns compared per tf in the differential
COLS_TF = DTP_COLS + DTP_AUDIT_COLS + SR_COLS + SR_PRICE_COLS + LIQ_COLS + LIQ_PRICE_COLS
DISCRETE_COLS = [
    "trend_state", "sr_in_zone", "sr_broken_up", "sr_broken_down", "sr_n_channels",
    "liq_breach_up", "liq_breach_down", "liq_last_breach_side",
    "liq_last_zone_active", "liq_up_count", "liq_down_count",
]

# numeric feature columns across all four TFs (excludes timestamps / bool / int metadata)
FEAT_COLS = [f"{tf}_{c}" for tf in ["m5", "m15", "h1", "h4"] for c in COLS_TF]

# ---------------------------------------------------------------- helpers
def make_base(times, o, h, l, c, segment=0, day=None, disc=None):
    n = len(times)
    if day is None:
        day = [pd.Timestamp("2025-01-02")] * n
    if disc is None:
        disc = [False] * n
    return pd.DataFrame({
        "time": pd.to_datetime(times),
        "trading_day": pd.to_datetime(day),
        "segment": np.int64(segment),
        "open": np.asarray(o, float),
        "high": np.asarray(h, float),
        "low": np.asarray(l, float),
        "close": np.asarray(c, float),
        "disc": np.asarray(disc, bool),
    })


def _compare(tf, prod_row, ref_row, tol=ABS_TOL):
    cells = disc_mis = 0
    max_err = 0.0
    first = None
    for c in COLS_TF:
        pv = prod_row.get(f"{tf}_{c}")
        rv = ref_row.get(c)
        cells += 1
        pnan, rnan = pd.isna(pv), pd.isna(rv)
        if pnan or rnan:
            if pnan != rnan:
                disc_mis += 1
                first = first or (tf, c, "nan_mismatch")
            continue
        if c in DISCRETE_COLS:
            if int(pv) != int(rv):
                disc_mis += 1
                first = first or (tf, c, f"{int(pv)}!={int(rv)}")
        else:
            e = abs(float(pv) - float(rv))
            if e > max_err:
                max_err = e
            if e > tol and first is None:
                first = (tf, c, round(e, 9))
    return cells, disc_mis, max_err, first


# ---------------------------------------------------------------- T0.1
def T0_1_forming_ohlc_handcalc():
    """Hand-computed forming OHLC for a mid-bucket 15m decision."""
    t = pd.date_range("2025-01-02 10:00", periods=6, freq="5min")
    o = [100, 101, 104, 105, 106, 108]
    h = [102, 105, 106, 107, 109, 110]
    l = [99, 100, 103, 104, 105, 107]
    c = [101, 104, 105, 106, 108, 109]
    base = make_base(t, o, h, l, c)
    b = FormingEnvironmentBuilder("SYN", tf_minutes={"m5": 5, "m15": 15}, max_bars=None)
    b.set_raw_frame(base).prepare()
    # decision 1 (10:05) is the 2nd base bar of the 10:00 15m bucket
    fb = b._forming_bar("m15", 1)
    assert fb["bucket_start"] == pd.Timestamp("2025-01-02 10:00"), fb
    assert fb["open"] == 100.0, fb
    assert fb["high"] == 105.0, fb  # max(102,105)
    assert fb["low"] == 99.0, fb    # min(99,100)
    assert fb["close"] == 104.0, fb
    assert fb["n_base"] == 2, fb
    # decision 2 (10:10) closes the bucket -> n_base==3, high includes 106
    fb2 = b._forming_bar("m15", 2)
    assert fb2["n_base"] == 3 and fb2["high"] == 106.0, fb2
    return "T0.1 forming OHLC hand-calc OK"


# ---------------------------------------------------------------- T0.2
def T0_2_no_partial_duplication():
    """Consecutive decisions in one bucket must not duplicate the forming bar
    into history; the completed portion length stays constant within a bucket."""
    t = pd.date_range("2025-01-02 10:00", periods=9, freq="5min")  # 3 x 15m buckets
    o = np.arange(9) * 1.0 + 100
    h = o + 2.0
    l = o - 1.0
    c = o + 1.0
    base = make_base(t, o, h, l, c)
    b = FormingEnvironmentBuilder("SYN", tf_minutes={"m5": 5, "m15": 15})
    b.set_raw_frame(base).prepare()
    # bucket 0 = decisions 0,1,2 ; bucket 1 = 3,4,5 ; bucket 2 = 6,7,8
    for first, last in [(0, 2), (3, 5), (6, 8)]:
        lens = []
        for i in range(first, last + 1):
            pos = b._pos["m15"][(b.base.iloc[i]["trading_day"], int(b.base.iloc[i]["segment"]),
                                 pd.Timestamp(b._bstart["m15"][i]))]
            seg_start = b._seg_start["m15"][int(b.base.iloc[i]["segment"])]
            lo = max(seg_start, pos - b._tail)
            lens.append(pos - lo)
        assert len(set(lens)) == 1, f"completed length varied within bucket: {lens}"
        # n_base grows 1,2,3 -> forming evolves, not duplicated as history
        nbs = [b._forming_bar("m15", i)["n_base"] for i in range(first, last + 1)]
        assert nbs == list(range(1, last - first + 2)), nbs
    return "T0.2 no partial duplication OK"


# ---------------------------------------------------------------- T0.3
def T0_3_future_mutation():
    """Future OHLC changes must not affect the snapshot at a past decision."""
    b = FormingEnvironmentBuilder("AG", max_bars=500)
    b.load_raw().prepare()
    df, _ = b.run()
    i = 250  # a mid-series decision
    before = df.iloc[i][FEAT_COLS]
    # append extreme future bars
    extra = b.base.iloc[i + 1:].copy()
    fut = extra.copy()
    fut = fut.assign(open=fut["open"] * 1000, high=fut["high"] * 1000,
                     low=fut["low"] * 1000, close=fut["close"] * 1000)
    new_base = pd.concat([b.base.iloc[: i + 1], fut], ignore_index=True)
    b2 = FormingEnvironmentBuilder("AG", max_bars=None)
    b2.set_raw_frame(new_base).prepare()
    df2, _ = b2.run()
    after = df2.iloc[i][FEAT_COLS]
    diff = (pd.to_numeric(before, errors="coerce") - pd.to_numeric(after, errors="coerce")).abs().max()
    assert pd.isna(diff) or diff < 1e-9, f"future mutation leaked: {diff}"
    return f"T0.3 future mutation OK (max diff {diff:.2e})"


# ---------------------------------------------------------------- T0.4
def T0_4_completed_boundary_parity():
    """At a bucket's final 5m decision the forming feature == canonical completed."""
    b = FormingEnvironmentBuilder("AG", max_bars=600)
    b.load_raw().prepare()
    df, _ = b.run()
    worst = 0.0
    for tf in ["m5", "m15", "h1", "h4"]:
        closes = [i for i in range(len(df)) if df[f"{tf}_is_complete"].iloc[i]]
        for i in closes[:20]:
            key = (b.base.iloc[i]["trading_day"], int(b.base.iloc[i]["segment"]),
                   pd.Timestamp(b._bstart[tf][i]))
            pos = b._pos[tf][key]
            canon = b._htf_feat[tf].iloc[pos]
            prod = df.iloc[i]
            _, _, mx, _ = _compare(tf, prod, canon, tol=1e-9)
            worst = max(worst, mx)
    assert worst < 1e-6, f"completed-boundary parity failed: {worst}"
    return f"T0.4 completed-boundary parity OK (worst {worst:.2e})"


# ---------------------------------------------------------------- T0.5
def T0_5_segment_reset():
    """Segment B must not inherit D/SR/L state from segment A."""
    # segment A: strong uptrend, big SR/liq
    tA = pd.date_range("2025-01-02 09:00", periods=60, freq="5min")
    oA = 100.0 + np.arange(60) * 1.0
    hA = oA + 3.0
    lA = oA - 1.0
    cA = oA + 1.0
    # segment B: completely different regime, starts fresh
    tB = pd.date_range("2025-01-02 14:00", periods=60, freq="5min")
    oB = 500.0 - np.arange(60) * 0.5
    hB = oB + 1.0
    lB = oB - 3.0
    cB = oB - 1.0
    base = pd.concat([
        make_base(tA, oA, hA, lA, cA, segment=0),
        make_base(tB, oB, hB, lB, cB, segment=1, disc=[True] + [False] * 59),
    ], ignore_index=True)

    b = FormingEnvironmentBuilder("SYN", tf_minutes={"m5": 5, "m15": 15})
    b.set_raw_frame(base).prepare()
    df, _ = b.run()
    # first decision of segment B
    b_idx = int((b.base["segment"] == 1).idxmax())
    full_row = df.iloc[b_idx]
    # isolated segment B
    b2 = FormingEnvironmentBuilder("SYN", tf_minutes={"m5": 5, "m15": 15})
    b2.set_raw_frame(base.iloc[b_idx:].reset_index(drop=True)).prepare()
    df2, _ = b2.run()
    iso_row = df2.iloc[0]
    # features at the start of B must match an isolated B (no A inheritance)
    for col in ["m5_dev", "m5_slope_atr", "m5_trend_score", "m5_trend_state",
                "m15_dev", "m15_slope_atr", "m15_trend_score", "m15_trend_state"]:
        a, c = full_row[col], iso_row[col]
        if pd.isna(a) and pd.isna(c):
            continue
        assert abs(float(a) - float(c)) < 1e-9, f"{col}: {a} vs {c} (segment leak)"
    return "T0.5 segment reset OK"


# ---------------------------------------------------------------- T0.6
def T0_6_pivot_known_time():
    """Pivot causality: only known at t_pivot + right (inherited from canonical).

    Verified two ways:
      (a) confirmed_pivots writes the pivot value only at index p+right;
      (b) build_sr_features must NOT surface the resistance before confirmation
          (truncated series) but must surface it afterwards (full series).
    """
    n = 120
    high = np.full(n, 100.0)
    p = 30
    high[p] = 200.0  # clear high pivot
    low = np.full(n, 100.0)
    close = np.full(n, 100.0)
    atr = np.full(n, 1.0)
    left = right = 10
    piv = pgm.confirmed_pivots(pd.Series(high), left, right, "high")
    assert pd.isna(piv[p]), "pivot visible immediately (known-time violated)"
    assert pd.isna(piv[p + right - 1]), "pivot known too early"
    assert not pd.isna(piv[p + right]), "pivot not known at p+right"
    # (b) an UNCONFIRMED spike must have ZERO effect on SR output: the output must
    # be identical to a series where the spike is simply absent (i.e. invisible).
    high_a = high.copy()                 # spike present, confirmed only at p+right
    high_b = high.copy()
    high_b[p] = 100.0                    # spike absent
    A = pgm.build_sr_features(pd.Series(high_a), pd.Series(low),
                              pd.Series(close), pd.Series(atr), pgm.PINE_DEFAULT)
    B = pgm.build_sr_features(pd.Series(high_b), pd.Series(low),
                              pd.Series(close), pd.Series(atr), pgm.PINE_DEFAULT)
    cols = ["sr_resistance_price", "sr_support_price",
            "sr_resistance_dist_atr", "sr_support_dist_atr"]
    for col in cols:
        a, b = A[col][: p + right], B[col][: p + right]
        for i in range(len(a)):
            av, bv = a[i], b[i]
            if np.isnan(av) or np.isnan(bv):
                assert np.isnan(av) == np.isnan(bv), \
                    f"{col} diverged before confirmation at {i} (known-time violated)"
            else:
                assert av == bv, \
                    f"{col} diverged before confirmation at {i} (known-time violated)"
    return "T0.6 pivot known-time OK"


# ---------------------------------------------------------------- T0.7
def T0_7_5m_sr_enabled():
    """5m SR must actually be computed (legacy pipeline disabled it)."""
    b = FormingEnvironmentBuilder("AG", max_bars=800)
    b.load_raw().prepare()
    df, _ = b.run()
    assert int((df["m5_sr_n_channels"] > 0).sum()) > 0, "5m SR produced no channels"
    assert df["m5_sr_support_dist_atr"].notna().any(), "5m SR dist all NaN"
    return "T0.7 5m SR enabled OK"


# ---------------------------------------------------------------- T0.8
def T0_8_empty_warmup_nan():
    """Warmup must remain NaN (no imputation / fillna / dropna)."""
    t = pd.date_range("2025-01-02 09:00", periods=20, freq="5min")  # < sma_len(50)
    o = np.arange(20) + 100.0
    h = o + 1.0
    l = o - 1.0
    c = o + 0.5
    base = make_base(t, o, h, l, c)
    b = FormingEnvironmentBuilder("SYN", tf_minutes={"m5": 5, "m15": 15})
    b.set_raw_frame(base).prepare()
    df, aud = b.run()
    # DTP needs 50 bars -> dev/slope_atr NaN during warmup
    assert df["m5_dev"].isna().all(), "DTP dev should be NaN in warmup"
    assert aud["warmup_missing_count"]["m5"] > 0, "warmup missing not counted"
    return "T0.8 empty/warmup/NaN OK"


# ---------------------------------------------------------------- T1
def T1_differential_small_sample():
    """Production forming vs slow reference, >=100 sampled decisions, 4 TFs."""
    b = FormingEnvironmentBuilder("AG", max_bars=1200)
    b.load_raw().prepare()
    df, _ = b.run()
    n = len(df)
    # sample decisions covering bucket first/middle/final, warmup, discontinuity
    idxs = set(range(0, min(n, 200), 2))  # bucket positions + warmup
    disc = df.index[df["segment"].diff() != 0]
    idxs.update(int(d) for d in disc if d < n)
    idxs = sorted(i for i in idxs if i < n)
    while len(idxs) < 120:  # pad to >=100
        idxs.append(len(idxs))
    idxs = sorted(set(idxs))[:200]

    ev = {"cells": 0, "discrete_mismatch": 0, "max_abs_error": 0.0, "first_mismatch": None}
    for i in idxs:
        for tf in ["m5", "m15", "h1", "h4"]:
            ref = b.slow_forming_snapshot_reference(i, tf)
            if ref is None:
                continue
            cells, dm, mx, first = _compare(tf, df.iloc[i], ref)
            ev["cells"] += cells
            ev["discrete_mismatch"] += dm
            ev["max_abs_error"] = max(ev["max_abs_error"], mx)
            if first and ev["first_mismatch"] is None:
                ev["first_mismatch"] = first
    assert ev["discrete_mismatch"] == 0, f"discrete mismatch: {ev}"
    assert ev["max_abs_error"] < 1e-6, f"float mismatch: {ev}"
    return (f"T1 differential OK (sampled {len(idxs)} decisions, "
            f"{ev['cells']} cells, disc_mis=0, max_err={ev['max_abs_error']:.2e})"), ev


# ---------------------------------------------------------------- T1.5
def T1_5_end_to_end(out_dir="artifacts/forming_environment_v1", max_bars=2500):
    """End-to-end on 2 real objects; structural invariants only (no model)."""
    import json
    syms = ["AG", "CU"]
    summary = {}
    all_dup = 0
    all_unsorted = 0
    future_viol = 0
    for sym in syms:
        b = FormingEnvironmentBuilder(sym, max_bars=max_bars)
        b.load_raw().prepare()
        df, aud = b.run()
        # structural invariants
        key = ["data_object", "decision_bar_index", "decision_time"]
        dup = int(df.duplicated(subset=key).sum())
        unsorted = int((df["decision_time"].diff().dropna() < pd.Timedelta(0)).sum())
        all_dup += dup
        all_unsorted += unsorted
        # future violation spot-check: 5 random decisions.
        # Build the mutated base from b.base (carries the time/trading_day/
        # segment/disc schema expected by set_raw_frame) so the two builders
        # share one canonical source. Only rows AFTER ki are multiplied; the
        # forming environment at decision ki must be invariant to that change.
        base0 = b.base
        rng = np.random.default_rng(0)
        for _ in range(5):
            ki = int(rng.integers(50, len(base0) - 50))
            before = df.iloc[ki][FEAT_COLS]
            fut = base0.iloc[ki + 1:].copy()
            fut = fut.assign(open=fut["open"] * 1e6, high=fut["high"] * 1e6,
                             low=fut["low"] * 1e6, close=fut["close"] * 1e6)
            new_base = pd.concat([base0.iloc[: ki + 1], fut], ignore_index=True)
            b2 = FormingEnvironmentBuilder(sym, max_bars=None)
            b2.set_raw_frame(new_base).prepare()
            df2, _ = b2.run()
            after = df2.iloc[int(ki)][FEAT_COLS]
            d = (before.astype(float) - after.astype(float)).abs().max()
            if not (pd.isna(d) or d < 1e-9):
                future_viol += 1
        # write temp parquet + audit (large parquet NOT committed to git)
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        df.to_parquet(d / f"forming_environment_{sym}.parquet", index=False)
        with open(d / f"forming_audit_{sym}.json", "w") as f:
            json.dump(aud, f, indent=2, default=str)
        summary[sym] = {
            "rows": int(len(df)), "cols": int(df.shape[1]),
            "dup_keys": dup, "unsorted": unsorted,
            "raw_load_count": aud["stats"]["raw_load_count"],
            "coverage_by_tf": aud["coverage_by_tf"],
            "runtime_sec": aud["stats"]["runtime_sec"],
            "peak_rss_mb": aud["stats"]["peak_rss_mb"],
        }
    assert all_dup == 0, f"duplicate semantic keys: {all_dup}"
    assert all_unsorted == 0, f"unsorted decision_time: {all_unsorted}"
    assert future_viol == 0, f"future violations: {future_viol}"
    return (f"T1.5 end-to-end OK (objects={syms}, dup_keys=0, unsorted=0, "
            f"future_violations=0)"), {"summary": summary,
                                        "dup_keys": all_dup, "unsorted": all_unsorted,
                                        "future_violations": future_viol}


# ---------------------------------------------------------------- runner
def run_all_tests():
    results = []
    # T0 suite (fast, synthetic + small real)
    t0 = [T0_1_forming_ohlc_handcalc, T0_2_no_partial_duplication, T0_3_future_mutation,
          T0_4_completed_boundary_parity, T0_5_segment_reset, T0_6_pivot_known_time,
          T0_7_5m_sr_enabled, T0_8_empty_warmup_nan]
    for fn in t0:
        try:
            msg = fn()
            results.append(("PASS", fn.__name__, msg, None))
            print(f"[PASS] {fn.__name__}: {msg}")
        except AssertionError as e:
            results.append(("FAIL", fn.__name__, str(e), None))
            print(f"[FAIL] {fn.__name__}: {e}")
    # T1
    try:
        msg, ev = T1_differential_small_sample()
        results.append(("PASS", "T1_differential", msg, ev))
        print(f"[PASS] {msg}")
    except AssertionError as e:
        results.append(("FAIL", "T1_differential", str(e), None))
        print(f"[FAIL] T1_differential: {e}")
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("R3A Forming MTF Environment — test runner")
    print("=" * 70)
    res = run_all_tests()
    # T1.5 is the long candidate run; run separately / explicitly
    DO_T15 = "--t15" in sys.argv
    if DO_T15:
        try:
            msg, ev = T1_5_end_to_end()
            print(f"[PASS] {msg}")
            print("  T1.5 summary:", json.dumps(ev["summary"], indent=2, default=str))
        except AssertionError as e:
            print(f"[FAIL] T1.5: {e}")
    else:
        print("\n(skipping T1.5 heavy run; pass --t15 to execute)")
    npass = sum(1 for r in res if r[0] == "PASS")
    print(f"\nT0/T1: {npass}/{len(res)} passed")
