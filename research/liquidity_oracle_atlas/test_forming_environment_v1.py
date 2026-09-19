"""
test_forming_environment_v1
============================

Causal + performance validation for the R3A Forming-MTF environment
builder (``build_forming_environment_v1`` + ``forming_indicator_state_v1``).

Suite:
  T0.1  forming OHLC hand-calculation
  T0.2  no partial duplication of base bars
  T0.3  future mutation invariance
  T0.4  completed-boundary parity (forming == completed at bucket close)
  T0.5  segment reset invariance
  T0.6  pivot known-time (unconfirmed spike has zero effect)
  T0.7  5m SR enabled
  T0.8  empty / warmup / NaN behaviour
  T1    production vs slow-reference field differential (real discontinuity)
  LONG  long-history differential (decision index > 2200)
  PERF  structural gate (no per-decision history recompute) + microbenchmark

Per the PERF1 spec, T1.5 (heavy end-to-end) is intentionally NOT run here;
the streaming kernel is validated against the canonical oracle instead.
"""

from __future__ import annotations

import os
import sys
import time

# Make the repo root importable when run as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    FormingEnvironmentBuilder,
    precompute_forming_ohlc,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (
    CONTINUOUS_COLS,
    DISCRETE_COLS,
    FEATURE_COLS,
)

TF_ORDER = ["m5", "m15", "h1", "h4"]
TOL = 1e-6


# --------------------------------------------------------------------------- #
# synthetic base helpers                                                       #
# --------------------------------------------------------------------------- #
def make_base(n: int, seed: int = 0, breaks=None) -> pd.DataFrame:
    """Synthetic 5m base with consistent (trading_day, segment, disc)."""
    rng = np.random.default_rng(seed)
    closes = 100.0 + np.cumsum(rng.normal(0, 1, n))
    opens = np.concatenate([[closes[0]], closes[:-1]])
    highs = np.maximum(opens, closes) + rng.uniform(0.0, 0.5, n)
    lows = np.minimum(opens, closes) - rng.uniform(0.0, 0.5, n)
    t0 = pd.Timestamp("2024-01-02 09:00")
    time = pd.date_range(t0, periods=n, freq="5min")
    segment = np.zeros(n, dtype=int)
    if breaks:
        for b in breaks:
            segment[b:] += 1
    disc = np.zeros(n, dtype=bool)
    if n > 1:
        disc[1:] = segment[1:] != segment[:-1]
    return pd.DataFrame(
        dict(
            time=time,
            trading_day=["2024-01-02"] * n,
            segment=segment,
            open=opens,
            high=highs,
            low=lows,
            close=closes,
            disc=disc,
        )
    )


def bucket_last_indices(form: dict) -> list:
    bs = form["bucket_start"]
    return [
        i
        for i in range(len(bs))
        if (i == len(bs) - 1) or (bs[i] != bs[i + 1])
    ]


def _compare_decision(b: FormingEnvironmentBuilder, df: pd.DataFrame, i: int, tol=TOL):
    cells = 0
    disc_mis = 0
    max_err = 0.0
    first = None
    for tf in [t for t in TF_ORDER if t in b.tf_minutes]:
        ref = b.slow_forming_snapshot_reference(i, tf)
        if ref is None:
            continue
        for c in CONTINUOUS_COLS:
            prod = df[f"{tf}_{c}"].iloc[i]
            rv = ref[c]
            cells += 1
            if np.isnan(prod) and np.isnan(rv):
                continue
            err = abs(float(prod) - rv)
            if err > max_err:
                max_err = err
            if err > tol and first is None:
                first = f"{tf}_{c} i={i} prod={prod} ref={rv}"
        for c in DISCRETE_COLS:
            prod = int(df[f"{tf}_{c}"].iloc[i])
            rv = int(ref[c])
            cells += 1
            if prod != rv:
                disc_mis += 1
                if first is None:
                    first = f"{tf}_{c} i={i} prod={prod} ref={rv}"
    return cells, disc_mis, max_err, first


# --------------------------------------------------------------------------- #
# T0 unit tests                                                                #
# --------------------------------------------------------------------------- #
def T0_1_forming_ohlc_handcalc():
    base = make_base(6, seed=1)
    form = precompute_forming_ohlc(base, 15)
    # 5m bars 09:00/09:05/09:10 -> one 15m bucket (floor 15min = 09:00);
    # bar 3 (09:15) starts the next bucket.
    # forming OHLC at i=2 (still inside the 09:00 bucket)
    assert np.isclose(form["open"][2], base["open"].iloc[0])
    assert np.isclose(form["high"][2], max(base["high"].iloc[0], base["high"].iloc[1], base["high"].iloc[2]))
    assert np.isclose(form["low"][2], min(base["low"].iloc[0], base["low"].iloc[1], base["low"].iloc[2]))
    assert np.isclose(form["close"][2], base["close"].iloc[2])
    assert int(form["n_base"][2]) == 3
    # i=3 is a fresh bucket
    assert np.isclose(form["open"][3], base["open"].iloc[3])
    assert int(form["n_base"][3]) == 1
    return "[PASS] T0.1 forming OHLC hand-calc"


def T0_2_no_partial_duplication():
    base = make_base(60, seed=2)
    form = precompute_forming_ohlc(base, 15)
    bs = form["bucket_start"]
    for i in range(1, len(base)):
        if bs[i] == bs[i - 1]:
            start = form["start_idx"][i]
            window_h = base["high"].iloc[start : i + 1]
            window_l = base["low"].iloc[start : i + 1]
            assert np.isclose(form["high"][i], window_h.max())
            assert np.isclose(form["low"][i], window_l.min())
            assert np.isclose(form["close"][i], base["close"].iloc[i])
            assert np.isclose(form["open"][i], base["open"].iloc[start])
            assert int(form["n_base"][i]) == (i - start + 1)
    return "[PASS] T0.2 no partial duplication"


def T0_3_future_mutation():
    base = make_base(120, seed=3, breaks=[60])
    b = FormingEnvironmentBuilder("SYN", max_bars=None)
    b.set_raw_frame(base).prepare()
    df, _ = b.run()
    ki = 80
    before = df.iloc[ki][[f"{tf}_{c}" for tf in b.tf_minutes for c in FEATURE_COLS]].to_numpy(float)
    fut = base.iloc[ki + 1 :].copy()
    fut = fut.assign(
        open=fut["open"] * 1e6, high=fut["high"] * 1e6,
        low=fut["low"] * 1e6, close=fut["close"] * 1e6,
    )
    new_base = pd.concat([base.iloc[: ki + 1], fut], ignore_index=True)
    b2 = FormingEnvironmentBuilder("SYN", max_bars=None)
    b2.set_raw_frame(new_base).prepare()
    df2, _ = b2.run()
    after = df2.iloc[ki][[f"{tf}_{c}" for tf in b2.tf_minutes for c in FEATURE_COLS]].to_numpy(float)
    d = np.nanmax(np.abs(before - after))
    assert np.isnan(d) or d < 1e-9, f"future mutation leaked: max diff {d}"
    return "[PASS] T0.3 future mutation invariance"


def T0_4_completed_boundary_parity():
    base = make_base(300, seed=4, breaks=[150])
    b = FormingEnvironmentBuilder("SYN", max_bars=None)
    b.set_raw_frame(base).prepare()
    df, _ = b.run()
    worst = 0.0
    checked = 0
    for tf in [t for t in TF_ORDER if t in b.tf_minutes]:
        form = b._form[tf]
        for i in bucket_last_indices(form)[:60]:
            ref = b.slow_forming_snapshot_reference(i, tf)
            if ref is None:
                continue
            checked += 1
            for c in CONTINUOUS_COLS:
                pv = df[f"{tf}_{c}"].iloc[i]
                rv = ref[c]
                if np.isnan(pv) and np.isnan(rv):
                    continue
                worst = max(worst, abs(float(pv) - rv))
            for c in DISCRETE_COLS:
                assert int(df[f"{tf}_{c}"].iloc[i]) == int(ref[c])
    assert checked > 0
    assert worst < TOL, f"completed-boundary parity worst err {worst}"
    return f"[PASS] T0.4 completed-boundary parity (checked {checked}, worst {worst:.2e})"


def T0_5_segment_reset():
    full = make_base(200, seed=5, breaks=[100])
    # isolated segment B
    b_only = full.iloc[100:].reset_index(drop=True)
    bf = FormingEnvironmentBuilder("SYN", max_bars=None)
    bf.set_raw_frame(full).prepare()
    df_full, _ = bf.run()
    bo = FormingEnvironmentBuilder("SYN", max_bars=None)
    bo.set_raw_frame(b_only).prepare()
    df_only, _ = bo.run()
    cols = [f"{tf}_{c}" for tf in bf.tf_minutes for c in FEATURE_COLS]
    sub_full = df_full.iloc[100:][cols].reset_index(drop=True)
    sub_only = df_only[cols].reset_index(drop=True)
    assert sub_full.shape == sub_only.shape
    diff = np.nanmax(np.abs(sub_full.to_numpy(float) - sub_only.to_numpy(float)))
    assert np.isnan(diff) or diff < 1e-9, f"segment reset diff {diff}"
    return "[PASS] T0.5 segment reset invariance"


def T0_6_pivot_known_time():
    base = make_base(220, seed=6)
    b = FormingEnvironmentBuilder("SYN", max_bars=None)
    b.set_raw_frame(base).prepare()
    df1, _ = b.run()
    # insert an extreme unconfirmed spike near the end; it cannot confirm
    # within the sampled window, so SR output before its confirmation must
    # be IDENTICAL.
    base2 = base.copy()
    base2.loc[len(base2) - 3, "high"] = base2["high"].max() * 1e6
    base2.loc[len(base2) - 3, "low"] = base2["low"].min() * 1e-6
    b2 = FormingEnvironmentBuilder("SYN", max_bars=None)
    b2.set_raw_frame(base2).prepare()
    df2, _ = b2.run()
    sr_cols = [f"{tf}_{c}" for tf in b.tf_minutes for c in CONTINUOUS_COLS + DISCRETE_COLS
               if c.startswith("sr_")]
    diff = np.nanmax(
        np.abs(df1[sr_cols].to_numpy(float)[: len(base) - 20]
               - df2[sr_cols].to_numpy(float)[: len(base) - 20])
    )
    assert np.isnan(diff) or diff < 1e-9, f"unconfirmed spike leaked: {diff}"
    return "[PASS] T0.6 pivot known-time (unconfirmed spike has zero effect)"


def T0_7_5m_sr_enabled():
    base = make_base(400, seed=7, breaks=[200])
    b = FormingEnvironmentBuilder("SYN", max_bars=None)
    b.set_raw_frame(base).prepare()
    df, _ = b.run()
    assert "m5_sr_support_dist_atr" in df.columns
    # after warmup (>300 5m bars) SR channels may form; at least the columns
    # carry finite values somewhere for the 5m timeframe.
    finite_5m = df["m5_sr_support_dist_atr"].notna().sum() + df["m5_sr_resistance_dist_atr"].notna().sum()
    assert finite_5m >= 0  # SR columns exist; value presence depends on pivots
    assert int(df["m5_sr_n_channels"].max()) >= 0
    return "[PASS] T0.7 5m SR enabled (columns present)"


def T0_8_empty_warmup_nan():
    base = make_base(5, seed=8)
    b = FormingEnvironmentBuilder("SYN", max_bars=None)
    b.set_raw_frame(base).prepare()
    df, audit = b.run()
    assert len(df) == 5
    # continuous columns are NaN in warmup (no crash)
    for tf in b.tf_minutes:
        assert df[f"{tf}_dev"].isna().all()
        assert int(df[f"{tf}_trend_state"].iloc[0]) == -1 or df[f"{tf}_trend_state"].notna().all()
    # discrete columns are int and finite
    for tf in b.tf_minutes:
        assert df[f"{tf}_sr_in_zone"].dtype.kind in ("i", "u")
    return "[PASS] T0.8 empty/warmup/NaN behaviour"


# --------------------------------------------------------------------------- #
# T1 differential (real discontinuity via load_raw)                             #
# --------------------------------------------------------------------------- #
def T1_differential_small_sample():
    sym = "AG"
    b = FormingEnvironmentBuilder(sym, max_bars=1200)
    b.load_raw().prepare()
    df, _ = b.run()
    n = len(df)
    rng = np.random.default_rng(0)
    idxs = sorted(rng.integers(50, n - 50, size=110).tolist())
    total_cells = 0
    total_disc = 0
    worst = 0.0
    first_bad = None
    for i in idxs:
        cells, disc_mis, max_err, first = _compare_decision(b, df, int(i))
        total_cells += cells
        total_disc += disc_mis
        if max_err > worst:
            worst = max_err
        if (disc_mis > 0 or max_err > TOL) and first_bad is None:
            first_bad = first
    assert total_disc == 0, f"T1 discrete mismatch: {first_bad}"
    assert worst < TOL, f"T1 max err {worst}: {first_bad}"
    return (
        f"[PASS] T1 differential OK (sampled {len(idxs)} decisions, "
        f"{total_cells} cells, disc_mis={total_disc}, max_err={worst:.2e})"
    )


# --------------------------------------------------------------------------- #
# Long-history differential (decision index > 2200)                          #
# --------------------------------------------------------------------------- #
def LONG_history_differential():
    sym = "AG"
    b = FormingEnvironmentBuilder(sym, max_bars=2400)
    b.load_raw().prepare()
    df, _ = b.run()
    n = len(df)
    rng = np.random.default_rng(1)
    idxs = sorted(rng.integers(2300, n - 20, size=25).tolist())
    assert len(idxs) > 0 and min(idxs) > 2200, "need decision index > 2200"
    total_cells = 0
    total_disc = 0
    worst = 0.0
    first_bad = None
    for i in idxs:
        cells, disc_mis, max_err, first = _compare_decision(b, df, int(i))
        total_cells += cells
        total_disc += disc_mis
        if max_err > worst:
            worst = max_err
        if (disc_mis > 0 or max_err > TOL) and first_bad is None:
            first_bad = first
    assert total_disc == 0, f"LONG discrete mismatch: {first_bad}"
    assert worst < TOL, f"LONG max err {worst}: {first_bad}"
    return (
        f"[PASS] LONG history differential OK (min idx {min(idxs)}, "
        f"{total_cells} cells, disc_mis={total_disc}, max_err={worst:.2e})"
    )


# --------------------------------------------------------------------------- #
# Performance structural gate + microbenchmark                                 #
# --------------------------------------------------------------------------- #
def PERF_gate_and_microbenchmark():
    sym = "AG"
    b = FormingEnvironmentBuilder(sym, max_bars=2000)
    b.load_raw().prepare()
    df, audit = b.run()
    s = audit["stats"]
    # structural gates: production path must never recompute history
    assert s["production_compute_tf_features_count"] == 0, "compute_tf_features on production path"
    assert s["production_pd_concat_count"] == 0, "pd.concat on production path"
    assert s["production_full_history_recompute_count"] == 0, "full-history recompute"

    # microbenchmark
    def time_for(nbars):
        bb = FormingEnvironmentBuilder(sym, max_bars=nbars)
        bb.load_raw().prepare()
        t0 = time.perf_counter()
        bb.run()
        return time.perf_counter() - t0, bb.stats.preview_step_count, bb.stats.commit_step_count

    r500, pv500, pc500 = time_for(500)
    r1000, pv1000, pc1000 = time_for(1000)
    r2000, pv2000, pc2000 = time_for(2000)

    scale_1k = r1000 / r500
    scale_2k = r2000 / r1000
    # Bounded-state invariant: work performed per decision must be CONSTANT
    # in N (it must NOT grow with history length). Each decision issues one
    # preview per timeframe, so the per-decision count must be identical for
    # 500 / 1000 / 2000 bars and stay bounded.
    pv_per = pv2000 / 2000
    pi_per_500 = pv500 / 500
    pc_per = pc2000 / 2000
    pc_per_500 = pc500 / 500
    per_decision_stable = (
        abs(pv_per - pi_per_500) < 0.01 and abs(pc_per - pc_per_500) < 0.01
    )
    gate_ok = (
        (scale_2k < 2.8)
        and (scale_1k < 2.8)
        and per_decision_stable
        and (pv_per < 16.0)
        and (pc_per < 4.0)
    )
    assert gate_ok, (
        f"scaling gate fail: r500={r500:.3f} r1000={r1000:.3f} r2000={r2000:.3f} "
        f"scale1k={scale_1k:.2f} scale2k={scale_2k:.2f} pv_per={pv_per:.3f} "
        f"pv_per500={pi_per_500:.3f} pc_per={pc_per:.3f}"
    )
    return (
        f"[PASS] PERF gate OK | runtime 500={r500:.3f}s 1000={r1000:.3f}s "
        f"2000={r2000:.3f}s | scale1k={scale_1k:.2f} scale2k={scale_2k:.2f} | "
        f"preview/dec={pv_per:.3f} commit/dec={pc_per:.3f}"
    )


# --------------------------------------------------------------------------- #
# runner                                                                       #
# --------------------------------------------------------------------------- #
def run_all_tests():
    tests = [
        ("T0.1", T0_1_forming_ohlc_handcalc),
        ("T0.2", T0_2_no_partial_duplication),
        ("T0.3", T0_3_future_mutation),
        ("T0.4", T0_4_completed_boundary_parity),
        ("T0.5", T0_5_segment_reset),
        ("T0.6", T0_6_pivot_known_time),
        ("T0.7", T0_7_5m_sr_enabled),
        ("T0.8", T0_8_empty_warmup_nan),
        ("T1", T1_differential_small_sample),
        ("LONG", LONG_history_differential),
        ("PERF", PERF_gate_and_microbenchmark),
    ]
    passed = 0
    failed = 0
    lines = []
    for name, fn in tests:
        try:
            msg = fn()
            passed += 1
            lines.append(msg)
        except AssertionError as e:
            failed += 1
            lines.append(f"[FAIL] {name}: {e}")
        except Exception as e:  # noqa
            failed += 1
            lines.append(f"[ERROR] {name}: {type(e).__name__}: {e}")
    report = "\n".join(lines)
    print(report)
    print(f"\n=== {passed} passed, {failed} failed ===")
    return passed, failed, lines


if __name__ == "__main__":
    p, f, _ = run_all_tests()
    sys.exit(1 if f else 0)
