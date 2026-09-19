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
  T0.7  5m SR enabled (non-vacuous: canonical must actually form channels)
  T0.8  empty / warmup / NaN behaviour
  T0.9  decision clock (C_t = T_t + 5min) + output schema
  T1    production vs slow-reference field differential (real discontinuity)
  LONG  long-history differential (decision index > 2200)
  MATURE mature IndicatorState differential (direct state-level, no 5m rebuild)
  PREVIEW preview() does not mutate committed state
  PERF  fail-closed structural gate + microbenchmark
  T1.5  efficient end-to-end engineering chain (Raw->Prepare->Streaming->
        Environment->Artifact->Audit). Engineering/artifact/schema validation
        only; does NOT re-verify indicator math (T0/T1/MATURE/PREVIEW do that),
        no future-mutation reruns, no Oracle join. One production run/object.

Per the PERF1 spec, T1.5 (heavy end-to-end) is intentionally NOT run here;
the streaming kernel is validated against the canonical oracle instead.
"""

from __future__ import annotations

import hashlib
import importlib
import json
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
    IndicatorState,
)
from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    PINE_DEFAULT,
    compute_tf_features,
    confirmed_pivots,
    raw_frame_from_owner,
)

TF_ORDER = ["m5", "m15", "h1", "h4"]
TOL = 1e-6

# T1.5 artifact staging (temp/test only; parquet is NOT committed)
_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
T15_OUT = os.path.join(
    _REPO_ROOT, "artifacts", "forming_environment_v1", "t15"
)

# Semantic row key + mandatory protocol metadata
KEY_COLS = ["data_object", "decision_bar_index", "decision_time"]
META_COLS = [
    "data_object",
    "decision_bar_index",
    "decision_bar_start_time",
    "decision_time",
    "segment",
    "trading_day",
]
# Future-derived model fields that must NOT be emitted
FORBIDDEN_FUTURE_COLS = ["tf_is_complete"]


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


def make_tent_bars(n: int, period: int = 40, top: float = 110.0, bot: float = 90.0) -> dict:
    """Deterministic tent (triangle) wave with IDENTICAL repeated extremes.

    Every apex sits at ``idx % period == period//2`` and every trough at
    ``idx % period == 0``, at exactly the same price. With
    ``sr_pivot_left == sr_pivot_right == 10`` (window 21) each apex is the
    STRICT unique max of its +/-10 window, so canonical SR reliably produces
    pivots and therefore channels. Used by the non-vacuous SR tests and the
    pivot known-time test.
    """
    idx = np.arange(n)
    ph = idx % period
    half = period // 2
    frac = np.where(ph <= half, ph / half, (period - ph) / half)
    close = bot + (top - bot) * frac
    opens = np.concatenate([[close[0]], close[:-1]])
    high = close + 0.5
    low = close - 0.5
    t = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    return dict(
        n=n,
        t=t.to_numpy(),
        day=np.full(n, np.datetime64("2024-01-02")),
        disc=np.zeros(n, dtype=bool),
        o=opens,
        h=high,
        l=low,
        c=close,
    )


def make_rich_bars(n: int = 1000, seed: int = 11) -> dict:
    """Deterministic but richly varying single-segment series.

    Trend + slow cycle + noise; exercises mature DTP (sma/atr/trend 500),
    SR channel lifecycle and liquidity breach/reclaim. Reproducible via seed.
    """
    rng = np.random.default_rng(seed)
    i = np.arange(n)
    close = (
        100.0
        + 8.0 * np.sin(2 * np.pi * i / 220.0)
        + np.cumsum(rng.normal(0.0, 0.35, n))
    )
    opens = np.concatenate([[close[0]], close[:-1]])
    spread = rng.uniform(0.1, 0.9, n)
    high = np.maximum(opens, close) + spread
    low = np.minimum(opens, close) - spread
    t = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    return dict(
        n=n,
        t=t.to_numpy(),
        day=np.full(n, np.datetime64("2024-01-02")),
        disc=np.zeros(n, dtype=bool),
        o=opens,
        h=high,
        l=low,
        c=close,
    )


def _tf_frame_from_bars(bars: dict) -> pd.DataFrame:
    """Canonical single-segment TF frame (same contract production uses when
    it calls ``compute_tf_features``): adds ``available_time`` (bar close) and
    ``n_base``.
    """
    frame = raw_frame_from_owner(bars)
    frame["available_time"] = pd.to_datetime(frame["time"]) + pd.Timedelta(minutes=5)
    frame["n_base"] = 1
    return frame


def bucket_last_indices(form: dict) -> list:
    bs = form["bucket_start"]
    return [
        i
        for i in range(len(bs))
        if (i == len(bs) - 1) or (bs[i] != bs[i + 1])
    ]


def _compare_decision(b: FormingEnvironmentBuilder, df: pd.DataFrame, i: int, tol=TOL):
    """Compare production row `i` against the canonical slow oracle.

    Returns explicit coverage accounting so a large cell count cannot be
    manufactured by NaN==NaN agreement:
        cells         total (continuous + discrete) cells examined
        finite_cells  continuous cells where BOTH sides are non-NaN (real evidence)
        nan_pairs     continuous cells where both sides are NaN (no evidence)
        half_nan      continuous cells where exactly one side is NaN (FATAL)
        disc_cells    discrete cells examined
        disc_mis      discrete mismatches
        max_err       worst absolute error over finite cells
    """
    cells = 0
    finite_cells = 0
    nan_pairs = 0
    half_nan = 0
    disc_cells = 0
    disc_mis = 0
    max_err = 0.0
    first = None
    for tf in [t for t in TF_ORDER if t in b.tf_minutes]:
        ref = b.slow_forming_snapshot_reference(i, tf)
        if ref is None:
            continue
        for c in CONTINUOUS_COLS:
            prod = float(df[f"{tf}_{c}"].iloc[i])
            rv = ref[c]
            cells += 1
            pn, rn = np.isnan(prod), np.isnan(rv)
            if pn and rn:
                nan_pairs += 1
                continue
            if pn != rn:
                half_nan += 1
                if first is None:
                    first = f"{tf}_{c} i={i} NaN mismatch prod={prod} ref={rv}"
                continue
            finite_cells += 1
            err = abs(prod - rv)
            if err > max_err:
                max_err = err
            if err > tol and first is None:
                first = f"{tf}_{c} i={i} prod={prod} ref={rv}"
        for c in DISCRETE_COLS:
            prod = int(df[f"{tf}_{c}"].iloc[i])
            rv = int(ref[c])
            cells += 1
            disc_cells += 1
            if prod != rv:
                disc_mis += 1
                if first is None:
                    first = f"{tf}_{c} i={i} prod={prod} ref={rv}"
    return cells, finite_cells, nan_pairs, half_nan, disc_cells, disc_mis, max_err, first


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
    """Precise known-time boundary: a pivot whose centre is at index p must
    be INVISIBLE at p .. p+R-1 and become visible FIRST at p+R.

    Verified against BOTH owners:
      * canonical ``confirmed_pivots``
      * the streaming ``IndicatorState`` (pivot enters ``sr.pivots``)
    Uses a deterministic tent wave so apex positions are known exactly.
    """
    n = 600
    bars = make_tent_bars(n=n, period=40)
    frame = _tf_frame_from_bars(bars)
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)
    opens = frame["open"].to_numpy(float)
    closes = frame["close"].to_numpy(float)
    L = int(PINE_DEFAULT.sr_pivot_left)
    R = int(PINE_DEFAULT.sr_pivot_right)

    # apexes sit at idx % period == period//2 (strict unique max of +/-L window)
    half = 40 // 2
    apexes = [i for i in range(n) if (i % 40) == half and (i + R) < n]
    assert len(apexes) >= 5, "synthetic produced too few apexes"

    # ---- canonical: confirmed_pivots ----
    cph = confirmed_pivots(high, L, R, "high")
    for p in apexes:
        for j in range(p, p + R):
            assert not np.isfinite(cph[j]), (
                f"canonical pivot visible too early: apex {p}, index {j}"
            )
        assert np.isfinite(cph[p + R]) and abs(cph[p + R] - high[p]) < 1e-12, (
            f"canonical pivot not visible at p+R: apex {p}, got {cph[p + R]}"
        )

    # ---- streaming: pivot must enter sr.pivots exactly at step p+R ----
    st = IndicatorState(PINE_DEFAULT, include_sr=True)
    first_seen = {}
    for k in range(n):
        st.step(k, opens[k], high[k], low[k], closes[k])
        for p in apexes:
            if p in first_seen:
                continue
            for (jj, vv) in st.sr.pivots:
                if jj == p + R and abs(vv - high[p]) < 1e-12:
                    first_seen[p] = k
                    break
    for p in apexes:
        assert p in first_seen, f"streaming never confirmed apex {p}"
        assert first_seen[p] == p + R, (
            f"streaming apex {p} first visible at {first_seen[p]}, expected {p + R}"
        )
    return (
        f"[PASS] T0.6 pivot known-time (apexes={len(apexes)}, first visible "
        f"exactly at p+{R}, canonical+streaming)"
    )


def T0_7_5m_sr_enabled():
    """Non-vacuous 5m SR: canonical MUST actually form channels on a
    deterministic periodic series, production must form the same ones and
    must equal canonical on every 5m feature column.

    (The previous version asserted `count >= 0`, which is true by definition
    and therefore proved nothing.)
    """
    frame = _tf_frame_from_bars(make_tent_bars(n=800, period=40))
    b = FormingEnvironmentBuilder("SYN", max_bars=None)
    b.set_raw_frame(frame).prepare()
    df, _ = b.run()
    canonical = compute_tf_features(frame, PINE_DEFAULT, include_sr=True)

    canon_max = int(canonical["sr_n_channels"].max())
    prod_max = int(df["m5_sr_n_channels"].max())
    assert canon_max > 0, "canonical formed no 5m SR channel (synthetic too weak)"
    assert prod_max > 0, "production formed no 5m SR channel"

    for c in CONTINUOUS_COLS:
        pv = df[f"m5_{c}"].to_numpy(float)
        rv = canonical[c].to_numpy(float)
        assert np.array_equal(np.isnan(pv), np.isnan(rv)), f"m5_{c} NaN pattern differs"
        m = ~np.isnan(pv)
        if m.any():
            worst = float(np.max(np.abs(pv[m] - rv[m])))
            assert worst < TOL, f"m5_{c} differs from canonical (worst {worst:.2e})"
    for c in DISCRETE_COLS:
        assert np.array_equal(
            df[f"m5_{c}"].to_numpy(int), canonical[c].to_numpy(int)
        ), f"m5_{c} discrete mismatch vs canonical"
    return (
        f"[PASS] T0.7 5m SR enabled (canonical channels={canon_max}, "
        f"production channels={prod_max}, prod==canonical)"
    )


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


def T0_9_decision_clock_and_schema():
    """R3A frozen semantics: the decision instant is the 5m bar CLOSE
    C_t = T_t + 5min. `decision_bar_start_time` is T_t (bar start),
    `decision_time` is C_t. Both, plus segment/trading_day, must be emitted
    and must agree with the canonical base frame.
    """
    base = make_base(300, seed=9, breaks=[150])
    b = FormingEnvironmentBuilder("SYN", max_bars=None)
    b.set_raw_frame(base).prepare()
    df, _ = b.run()

    required = [
        "data_object",
        "decision_bar_index",
        "decision_bar_start_time",
        "decision_time",
        "segment",
        "trading_day",
    ]
    missing = [c for c in required if c not in df.columns]
    assert not missing, f"missing schema columns: {missing}"

    # decision clock: exactly +5min on EVERY row
    delta = df["decision_time"] - df["decision_bar_start_time"]
    bad = int((delta != pd.Timedelta(minutes=5)).sum())
    assert bad == 0, f"decision_time != bar start + 5min on {bad} rows"

    # bar start == canonical base time; metadata == canonical base
    assert np.array_equal(
        df["decision_bar_start_time"].to_numpy(), base["time"].to_numpy()
    ), "decision_bar_start_time != canonical base time"
    assert np.array_equal(
        df["segment"].to_numpy(int), base["segment"].to_numpy(int)
    ), "segment != canonical base segment"
    assert np.array_equal(
        pd.to_datetime(df["trading_day"]).to_numpy(),
        pd.to_datetime(base["trading_day"]).to_numpy(),
    ), "trading_day != canonical base trading_day"
    return (
        f"[PASS] T0.9 decision clock + schema (rows={len(df)}, "
        f"clock violations=0, cols={len(required)})"
    )


def _run_differential(
    b: FormingEnvironmentBuilder,
    df: pd.DataFrame,
    idxs,
    label: str,
    min_finite: int = 1,
):
    """Aggregate the differential over `idxs` and enforce real evidence."""
    total_cells = 0
    finite_cells = 0
    nan_pairs = 0
    half_nan = 0
    disc_cells = 0
    disc_mis = 0
    worst = 0.0
    first_bad = None
    for i in idxs:
        cells, fc, npr, hn, dc, dm, me, first = _compare_decision(b, df, int(i))
        total_cells += cells
        finite_cells += fc
        nan_pairs += npr
        half_nan += hn
        disc_cells += dc
        disc_mis += dm
        if me > worst:
            worst = me
        if (dm > 0 or me > TOL or hn > 0) and first_bad is None:
            first_bad = first
    assert half_nan == 0, f"{label} NaN-pattern mismatch half_nan={half_nan}: {first_bad}"
    assert disc_mis == 0, f"{label} discrete mismatch: {first_bad}"
    assert worst < TOL, f"{label} max err {worst}: {first_bad}"
    assert finite_cells >= min_finite, (
        f"{label} insufficient real evidence: finite_cells={finite_cells} "
        f"(nan_pairs={nan_pairs} carry no evidence)"
    )
    return dict(
        decisions=len(list(idxs)),
        cells=total_cells,
        finite=finite_cells,
        nan_pairs=nan_pairs,
        disc=disc_cells,
        disc_mis=disc_mis,
        max_err=worst,
    )


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
    r = _run_differential(b, df, idxs, "T1")
    return (
        f"[PASS] T1 differential OK (sampled {r['decisions']} decisions, "
        f"{r['cells']} cells | finite={r['finite']} nan_pairs={r['nan_pairs']} "
        f"disc={r['disc']} disc_mis={r['disc_mis']} max_err={r['max_err']:.2e})"
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
    r = _run_differential(b, df, idxs, "LONG")
    return (
        f"[PASS] LONG history differential OK (min idx {min(idxs)}, "
        f"{r['cells']} cells | finite={r['finite']} nan_pairs={r['nan_pairs']} "
        f"disc={r['disc']} disc_mis={r['disc_mis']} max_err={r['max_err']:.2e})"
    )


# --------------------------------------------------------------------------- #
# Mature-state IndicatorState differential (direct, no 5m reconstruction)       #
# --------------------------------------------------------------------------- #
def MATURE_state_differential():
    """Validate the streaming IndicatorState on MATURE high-timeframe state
    without manufacturing tens of thousands of 5m bars.

    A single-segment synthetic TF frame of 1000 bars is fed directly:
      reference  = ONE compute_tf_features call (canonical batch owner)
      production = IndicatorState stepped once per bar
    Only MATURE rows (past every warmup window: trend 500 / atr 200 /
    sr width 300) carry real evidence, so rows [warm, n) are compared.
    """
    n = 1000
    frame = _tf_frame_from_bars(make_rich_bars(n=n, seed=11))
    O = frame["open"].to_numpy(float)
    H = frame["high"].to_numpy(float)
    L = frame["low"].to_numpy(float)
    C = frame["close"].to_numpy(float)

    canonical = compute_tf_features(frame, PINE_DEFAULT, include_sr=True)
    st = IndicatorState(PINE_DEFAULT, include_sr=True)
    prod = [st.step(k, O[k], H[k], L[k], C[k]) for k in range(n)]

    warm = max(
        int(PINE_DEFAULT.trend_norm_lookback),
        int(PINE_DEFAULT.sr_width_lookback),
        int(PINE_DEFAULT.atr_len),
    )
    rows = list(range(warm, n))
    assert len(rows) >= 300, f"mature window too small: {len(rows)}"

    finite_cells = 0
    nan_pairs = 0
    half_nan = 0
    disc_cells = 0
    disc_mis = 0
    max_err = 0.0
    first_bad = None
    for k in rows:
        for c in CONTINUOUS_COLS:
            pv = float(prod[k][c])
            rv = float(canonical[c].iloc[k])
            pn, rn = np.isnan(pv), np.isnan(rv)
            if pn and rn:
                nan_pairs += 1
                continue
            if pn != rn:
                half_nan += 1
                if first_bad is None:
                    first_bad = f"{c} k={k} NaN mismatch prod={pv} ref={rv}"
                continue
            finite_cells += 1
            e = abs(pv - rv)
            if e > max_err:
                max_err = e
            if e > TOL and first_bad is None:
                first_bad = f"{c} k={k} prod={pv} ref={rv}"
        for c in DISCRETE_COLS:
            disc_cells += 1
            if int(prod[k][c]) != int(canonical[c].iloc[k]):
                disc_mis += 1
                if first_bad is None:
                    first_bad = (
                        f"{c} k={k} prod={prod[k][c]} ref={canonical[c].iloc[k]}"
                    )

    assert half_nan == 0, f"MATURE NaN pattern mismatch: {first_bad}"
    assert disc_mis == 0, f"MATURE discrete mismatch: {first_bad}"
    assert max_err < TOL, f"MATURE max err {max_err}: {first_bad}"
    assert finite_cells > 0, "MATURE produced no finite comparison cells"
    return (
        f"[PASS] MATURE state differential OK (mature rows={len(rows)} "
        f"from {warm}, finite={finite_cells} nan_pairs={nan_pairs} "
        f"disc={disc_cells} disc_mis={disc_mis} max_err={max_err:.2e})"
    )


# --------------------------------------------------------------------------- #
# Preview non-mutation                                                         #
# --------------------------------------------------------------------------- #
def PREVIEW_non_mutation():
    """The design is committed-state -> preview forming -> discard. Prove
    preview does NOT mutate the committed state:
      * repeated preview of the same bar is idempotent (p1 == p2)
      * committing that bar yields exactly the previewed features (c == p1)
      * subsequent steps still match canonical (no residue)
    """
    n = 900
    frame = _tf_frame_from_bars(make_rich_bars(n=n, seed=13))
    O = frame["open"].to_numpy(float)
    H = frame["high"].to_numpy(float)
    L = frame["low"].to_numpy(float)
    C = frame["close"].to_numpy(float)
    canonical = compute_tf_features(frame, PINE_DEFAULT, include_sr=True)

    K = 600
    st = IndicatorState(PINE_DEFAULT, include_sr=True)
    for k in range(K):
        st.step(k, O[k], H[k], L[k], C[k])

    p1 = st.preview(K, O[K], H[K], L[K], C[K])
    p2 = st.preview(K, O[K], H[K], L[K], C[K])
    rep_mis = 0
    for c in FEATURE_COLS:
        a, bb = float(p1[c]), float(p2[c])
        an, bn = np.isnan(a), np.isnan(bb)
        if an != bn or (not an and abs(a - bb) > TOL):
            rep_mis += 1
    assert rep_mis == 0, f"repeated preview differs on {rep_mis} fields"

    c1 = st.step(K, O[K], H[K], L[K], C[K])
    pv_mis = 0
    for c in FEATURE_COLS:
        a, bb = float(c1[c]), float(p1[c])
        an, bn = np.isnan(a), np.isnan(bb)
        if an != bn or (not an and abs(a - bb) > TOL):
            pv_mis += 1
    assert pv_mis == 0, f"commit != preview on {pv_mis} fields"

    next_mis = 0
    worst = 0.0
    for k in (K + 1, K + 2):
        f = st.step(k, O[k], H[k], L[k], C[k])
        for c in CONTINUOUS_COLS:
            pv = float(f[c])
            rv = float(canonical[c].iloc[k])
            if np.isnan(pv) and np.isnan(rv):
                continue
            if np.isnan(pv) != np.isnan(rv):
                next_mis += 1
                continue
            e = abs(pv - rv)
            worst = max(worst, e)
            if e > TOL:
                next_mis += 1
        for c in DISCRETE_COLS:
            if int(f[c]) != int(canonical[c].iloc[k]):
                next_mis += 1
    assert next_mis == 0, f"post-preview steps diverge from canonical ({next_mis})"
    return (
        f"[PASS] PREVIEW non-mutation OK (rep_mis=0, preview==commit, "
        f"next-step worst={worst:.2e}, K={K})"
    )


# --------------------------------------------------------------------------- #
# Performance structural gate + microbenchmark                                 #
# --------------------------------------------------------------------------- #
def PERF_gate_and_microbenchmark():
    sym = "AG"
    build_mod = importlib.import_module(
        "research.liquidity_oracle_atlas.build_forming_environment_v1"
    )

    def forbidden(*a, **k):
        raise AssertionError("FORBIDDEN_SLOW_CALL_ON_PRODUCTION_PATH")

    # ---- negative control: the gate itself MUST be able to fail ----
    try:
        forbidden()
    except AssertionError:
        pass
    else:
        raise AssertionError("negative control failed: forbidden() did not raise")

    class _ForbidConcat:
        """Delegate every pandas attribute EXCEPT concat (hot-path poison)."""

        def __init__(self, real):
            object.__setattr__(self, "_real", real)

        def __getattr__(self, name):
            if name == "concat":
                raise AssertionError("FORBIDDEN pd.concat on production hot path")
            return getattr(object.__getattribute__(self, "_real"), name)

    real_ctf = build_mod.compute_tf_features
    real_pd = build_mod.pd

    b = FormingEnvironmentBuilder(sym, max_bars=2000)
    # prepare() may legitimately use pd.concat / canonical aggregation
    b.load_raw().prepare()
    build_mod.compute_tf_features = forbidden
    build_mod.pd = _ForbidConcat(real_pd)
    violations = 0
    try:
        df, audit = b.run(profile_memory=False)
    except AssertionError:
        violations += 1
        raise
    finally:
        build_mod.compute_tf_features = real_ctf
        build_mod.pd = real_pd

    # counters are telemetry only; the monkeypatch above is the real gate
    s = audit["stats"]
    assert s["production_compute_tf_features_count"] == 0
    assert s["production_pd_concat_count"] == 0
    assert s["production_full_history_recompute_count"] == 0

    n_decisions = len(df)
    n_tf = len([t for t in TF_ORDER if t in b.tf_minutes])
    expected_preview = n_decisions * n_tf
    assert b.stats.preview_step_count == expected_preview, (
        f"preview_step_count {b.stats.preview_step_count} != "
        f"n_decisions*n_tf {expected_preview}"
    )

    # microbenchmark (tracemalloc OFF).
    # Single-shot timings at these sizes are noise-dominated, so take the
    # MIN of several repetitions (first rep also acts as a warmup). The
    # scaling assertion below is unchanged; only measurement noise is reduced.
    reps = 3

    def time_for(nbars):
        bb = FormingEnvironmentBuilder(sym, max_bars=nbars)
        bb.load_raw().prepare()
        best = None
        pv = pc = 0
        for _ in range(reps):
            bb.stats.preview_step_count = 0
            bb.stats.commit_step_count = 0
            t0 = time.perf_counter()
            bb.run(profile_memory=False)
            dt = time.perf_counter() - t0
            best = dt if best is None else min(best, dt)
            pv = bb.stats.preview_step_count
            pc = bb.stats.commit_step_count
        exp = bb.n * len([t for t in TF_ORDER if t in bb.tf_minutes])
        assert pv == exp, f"nbars={nbars} preview {pv} != {exp}"
        return best, pv, pc

    r500, pv500, pc500 = time_for(500)
    r1000, pv1000, pc1000 = time_for(1000)
    r2000, pv2000, pc2000 = time_for(2000)

    scale_1k = r1000 / r500
    scale_2k = r2000 / r1000
    assert scale_1k < 2.8 and scale_2k < 2.8, (
        f"scaling gate fail: r500={r500:.3f} r1000={r1000:.3f} r2000={r2000:.3f} "
        f"scale1k={scale_1k:.2f} scale2k={scale_2k:.2f}"
    )
    return (
        f"[PASS] PERF gate OK | runtime 500={r500:.3f}s 1000={r1000:.3f}s "
        f"2000={r2000:.3f}s | scale1k={scale_1k:.2f} scale2k={scale_2k:.2f} | "
        f"preview={pv2000} expected={2000 * n_tf} | commit={pc2000} | "
        f"forbidden violations={violations}"
    )


# --------------------------------------------------------------------------- #
# T1.5 efficient end-to-end engineering-chain validation                       #
# --------------------------------------------------------------------------- #
def _t15_window_frame(symbol: str, start: int, end: int):
    """Test-only WINDOW selection (raw[start:end]).

    Uses the documented ``set_raw_frame`` hook; production is NOT modified.
    Needed because ``max_bars`` is prefix-only, and the only object carrying a
    real discontinuity (SC) has it at index 30496 — a prefix window would need
    ~30.5k bars, which conflicts with the ~2500-3000 bar scale. Returns
    (frame, real_discontinuity_count).
    """
    from research.export_ob_trigger_execution_v21 import load_raw_5m
    from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags

    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    disc = np.asarray(discontinuity_flags(symbol), dtype=bool)
    if len(raw) != len(disc):
        raise SystemExit("STOP_T15_RAW_DISC_LENGTH_MISMATCH")
    r = raw.iloc[start:end]
    d = disc[start:end]
    bars = dict(
        n=len(r),
        t=pd.to_datetime(r["bar_start_time"]).to_numpy(),
        day=pd.to_datetime(r["trading_day"]).to_numpy(),
        disc=d,
        o=r["open"].to_numpy(float),
        h=r["high"].to_numpy(float),
        l=r["low"].to_numpy(float),
        c=r["close"].to_numpy(float),
    )
    return raw_frame_from_owner(bars), int(d.sum())


def T1_5_efficient_e2e():
    """End-to-end engineering chain ONLY:

        Raw -> Prepare -> Streaming Environment -> Artifact -> Audit

    Deliberately does NOT re-verify DTP/SR/Liquidity math, mature-state
    differential or preview math (already covered by T0/T1/MATURE/PREVIEW).
    No future-mutation reruns (T0 covers it), no Oracle join, no model fit.
    Each object runs production EXACTLY ONCE.
    """
    os.makedirs(T15_OUT, exist_ok=True)

    # per-TF required engineering columns (presence only; math already proven)
    tf_cols = []
    for tf in TF_ORDER:
        tf_cols += [
            f"{tf}_bucket_start",
            f"{tf}_n_base_known",
            f"{tf}_dev",              # DTP present
            f"{tf}_sr_n_channels",    # SR present
            f"{tf}_liq_up_count",     # Liquidity present
        ]
    required = list(META_COLS) + tf_cols

    specs = [
        ("AG", ("prefix", 3000)),
        ("SC", ("window", 29000, 32000)),
    ]

    frames = []
    objs = []
    total_runtime = 0.0
    rng = np.random.default_rng(7)

    for sym, mode in specs:
        b = FormingEnvironmentBuilder(
            sym, max_bars=(mode[1] if mode[0] == "prefix" else None)
        )
        if mode[0] == "prefix":
            b.load_raw()
            window_note = f"prefix[0:{mode[1]}]"
        else:
            frame, _ndisc = _t15_window_frame(sym, mode[1], mode[2])
            b.set_raw_frame(frame)
            window_note = f"window[{mode[1]}:{mode[2]}]"
        b.prepare()

        # ---- production runs EXACTLY ONCE per object ----
        t0 = time.perf_counter()
        df, audit = b.run(profile_memory=False)
        dt = time.perf_counter() - t0
        total_runtime += dt
        if dt > 30.0:
            raise SystemExit(
                f"STOP_UNEXPECTED_RUNTIME {sym} {dt:.1f}s > 30s "
                "(check for code drift / slow reference / profiling)"
            )

        n = len(df)
        n_tf = len([t for t in TF_ORDER if t in b.tf_minutes])
        st = audit["stats"]

        # ---- hard counts: one raw load, one production run ----
        assert st["raw_load_count"] == 1, f"{sym} raw_load_count={st['raw_load_count']}"
        expected_preview = n * n_tf
        assert st["preview_step_count"] == expected_preview, (
            f"{sym} preview {st['preview_step_count']} != {expected_preview}"
        )

        # ---- schema / forbidden future columns ----
        missing = [c for c in required if c not in df.columns]
        present_future = [c for c in FORBIDDEN_FUTURE_COLS if c in df.columns]

        # ---- decision clock ----
        delta = df["decision_time"] - df["decision_bar_start_time"]
        clock_viol = int((delta != pd.Timedelta(minutes=5)).sum())

        # ---- ordering (strictly increasing) ----
        bi = df["decision_bar_index"].to_numpy()
        dtimes = df["decision_time"].to_numpy()
        bi_ok = bool(np.all(np.diff(bi) > 0))
        dt_ok = bool(np.all(np.diff(dtimes) > np.timedelta64(0, "ns")))
        unsorted_rows = 0 if (bi_ok and dt_ok) else n

        # ---- duplicate semantic keys ----
        dup = int(df.duplicated(subset=KEY_COLS).sum())

        # ---- discontinuity / segments ----
        real_disc = int(b.base["disc"].sum())
        seg_count = int(b.base["segment"].nunique())

        # ---- artifact write ----
        ppath = os.path.join(T15_OUT, f"forming_environment_{sym}.parquet")
        apath = os.path.join(T15_OUT, f"forming_environment_{sym}_audit.json")
        df.to_parquet(ppath, index=False)
        with open(apath, "w") as fh:
            json.dump(audit, fh, indent=2, default=str)
        raw_bytes = open(ppath, "rb").read()
        sha = hashlib.sha256(raw_bytes).hexdigest()

        # ---- artifact round-trip (this is what T1.5 should prove) ----
        rt = pd.read_parquet(ppath)
        assert len(rt) == n, f"{sym} roundtrip row count {len(rt)} != {n}"
        assert list(rt.columns) == list(df.columns), f"{sym} roundtrip columns differ"
        rt_key_mismatch = 0
        for c in KEY_COLS + ["decision_bar_start_time"]:
            if not rt[c].equals(df[c]):
                rt_key_mismatch += 1
        rtd = rt["decision_time"] - rt["decision_bar_start_time"]
        rt_clock_viol = int((rtd != pd.Timedelta(minutes=5)).sum())
        # 20-row sample: dtype/schema integrity + value fidelity
        samp = rng.choice(n, size=min(20, n), replace=False)
        dtype_bad = 0
        val_bad = 0
        for tf in TF_ORDER:
            for c in DISCRETE_COLS:
                if rt[f"{tf}_{c}"].dtype.kind not in ("i", "u"):
                    dtype_bad += 1
            for c in CONTINUOUS_COLS:
                col = f"{tf}_{c}"
                if rt[col].dtype.kind != "f":
                    dtype_bad += 1
                    continue
                a = rt[col].to_numpy(float)[samp]
                bb = df[col].to_numpy(float)[samp]
                if not np.array_equal(np.isnan(a), np.isnan(bb)):
                    val_bad += 1
                    continue
                m = ~np.isnan(a)
                if m.any() and not np.all(np.abs(a[m] - bb[m]) <= 1e-12):
                    val_bad += 1

        objs.append(
            dict(
                object=sym,
                window=window_note,
                rows=n,
                columns=len(df.columns),
                segments=seg_count,
                real_discontinuity_count=real_disc,
                raw_load_count=int(st["raw_load_count"]),
                preview_step_count=int(st["preview_step_count"]),
                expected_preview_count=expected_preview,
                commit_step_count=int(st["commit_step_count"]),
                duplicate_keys=dup,
                decision_clock_violations=clock_viol,
                unsorted_rows=unsorted_rows,
                schema_missing_columns=missing,
                forbidden_future_columns=present_future,
                runtime_sec=round(dt, 3),
                parquet=dict(
                    path=os.path.relpath(ppath, _REPO_ROOT),
                    rows=n,
                    size_bytes=len(raw_bytes),
                    sha256=sha,
                ),
                roundtrip=dict(
                    rows=len(rt),
                    columns=len(rt.columns),
                    key_mismatch=rt_key_mismatch,
                    clock_violations=rt_clock_viol,
                    dtype_bad=dtype_bad,
                    sample_value_mismatch=val_bad,
                ),
            )
        )
        frames.append(df)

    # ---- multi-object merge smoke (artifact aggregation; pd.concat allowed) ----
    merged = pd.concat(frames, ignore_index=True)
    global_dup = int(merged.duplicated(subset=["data_object", "decision_bar_index"]).sum())
    global_dup_t = int(merged.duplicated(subset=["data_object", "decision_time"]).sum())

    # ---- acceptance ----
    for o in objs:
        assert o["duplicate_keys"] == 0, f"{o['object']} duplicate keys"
        assert o["decision_clock_violations"] == 0, f"{o['object']} clock violations"
        assert o["unsorted_rows"] == 0, f"{o['object']} unsorted rows"
        assert not o["schema_missing_columns"], (
            f"{o['object']} missing {o['schema_missing_columns']}"
        )
        assert not o["forbidden_future_columns"], (
            f"{o['object']} future columns {o['forbidden_future_columns']}"
        )
        assert o["raw_load_count"] == 1, f"{o['object']} raw_load != 1"
        assert o["preview_step_count"] == o["expected_preview_count"]
        assert o["roundtrip"]["key_mismatch"] == 0, f"{o['object']} roundtrip key mismatch"
        assert o["roundtrip"]["clock_violations"] == 0
        assert o["roundtrip"]["dtype_bad"] == 0
        assert o["roundtrip"]["sample_value_mismatch"] == 0
    assert len(objs) == 2, "need 2 objects"
    assert global_dup == 0 and global_dup_t == 0, "merged semantic key duplicates"
    assert sum(o["real_discontinuity_count"] for o in objs) > 0, (
        "no object covered a real discontinuity"
    )

    summary = dict(
        task_id="FUTURE-ENV-R3A-T1.5-EFFICIENT-E2E",
        objects=[o["object"] for o in objs],
        total_runtime_sec=round(total_runtime, 3),
        merged_rows=len(merged),
        merged_duplicate_keys=global_dup,
        objects_detail=objs,
        acceptance=dict(
            objects_completed=len(objs),
            duplicate_keys=0,
            decision_clock_violations=0,
            unsorted_rows=0,
            schema_missing=0,
            future_only_columns=0,
            raw_load_per_object=1,
            preview_equals_rows_x_ntf=True,
            roundtrip_key_mismatch=0,
            unexpected_runtime_stop=0,
        ),
    )
    spath = os.path.join(T15_OUT, "FORMING_ENVIRONMENT_T15_SUMMARY.json")
    with open(spath, "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    disc_total = sum(o["real_discontinuity_count"] for o in objs)
    return (
        f"[PASS] T1.5 e2e OK (objects={len(objs)}, rows={[o['rows'] for o in objs]}, "
        f"segments={[o['segments'] for o in objs]}, real_disc={disc_total}, "
        f"runtime/obj={[o['runtime_sec'] for o in objs]}s total={total_runtime:.2f}s, "
        f"preview={[o['preview_step_count'] for o in objs]}==expected, "
        f"dup=0 clock_viol=0 unsorted=0, roundtrip_ok, merged_dup={global_dup})"
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
        ("T0.9", T0_9_decision_clock_and_schema),
        ("T1", T1_differential_small_sample),
        ("LONG", LONG_history_differential),
        ("MATURE", MATURE_state_differential),
        ("PREVIEW", PREVIEW_non_mutation),
        ("PERF", PERF_gate_and_microbenchmark),
        ("T1.5", T1_5_efficient_e2e),
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
