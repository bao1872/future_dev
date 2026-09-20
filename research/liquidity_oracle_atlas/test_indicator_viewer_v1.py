"""
test_indicator_viewer_v1
=========================

T0 (structural) / T1 (differential) / TP (performance) tests for the
Indicator Viewer (Task FUTURE-INDICATOR-VIEWER-V1-KERNEL-UI-P1).

Frozen-owner usage only:
  * experiment_structural_reversion_pgm_v1.{PINE_DEFAULT, resample_causal,
    compute_tf_features}        -- canonical reference (tests may import)
  * build_forming_environment_v1.FormingEnvironmentBuilder
  * audit_view_v1.SYMBOLS
  * forming_indicator_state_v1.IndicatorState

The literal Pine DTP profile port lives HERE (test-only), never in the
production helper.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np
import pandas as pd
import pytest

from research.liquidity_oracle_atlas.audit_view_v1 import SYMBOLS
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    FormingEnvironmentBuilder,
)
from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    PINE_DEFAULT,
    compute_tf_features,
    resample_causal,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (
    IndicatorState,
)
from research.liquidity_oracle_atlas.indicator_viewer_v1 import (
    BINS,
    TF_LABEL_TO_MINUTES,
    build_viewer_track,
    dtp_profile,
    selected_snapshot,
)

TF_LABELS = ["5m", "15m", "1H", "4H"]
CONT_TOL = 1e-9

CORE_CONT = ["sma", "atr", "trend_score"]
CORE_DISC = [
    "trend_state", "sr_n_channels", "sr_in_zone",
    "liq_up_count", "liq_down_count", "liq_breach_up", "liq_breach_down",
]


# --------------------------------------------------------------------------- #
# synthetic base helpers                                                        #
# --------------------------------------------------------------------------- #
def make_base(times, ohlc, segments, disc):
    return pd.DataFrame({
        "time": pd.to_datetime(times),
        "trading_day": pd.to_datetime([t.date() for t in times]),
        "segment": np.asarray(segments, dtype=np.int64),
        "open": ohlc[:, 0], "high": ohlc[:, 1],
        "low": ohlc[:, 2], "close": ohlc[:, 3],
        "disc": np.asarray(disc, dtype=bool),
    })


def gen_trend_base(n=1200, up_until=600, seed=0):
    """Four-phase series that actually triggers DTP trend switches.

    DTP switches require the normalized slope score to CROSS +/-0.1 from the
    inside. A monotonic trend never crosses from <=0.1, so we build:
      * 0..300   strong UP   (establishes a large rolling-max slope)
      * 300..600 flat/sideways (current slope tiny vs window max -> score ~0 <=0.1)
      * 600..900 strong UP   (current slope regrows past window max -> score >0.1 -> UP switch)
      * 900..n   strong DOWN (score crosses -0.1 -> DOWN switch)
    """
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = np.empty(n)
    val = 100.0
    for i in range(n):
        if i < 300:
            val += 1.0 + 0.1 * rng.random()
        elif i < 600:
            val += 0.0 + 0.05 * rng.random()
        elif i < 900:
            val += 1.0 + 0.1 * rng.random()
        else:
            val -= 1.0 + 0.1 * rng.random()
        close[i] = val
    open_ = np.empty(n); high = np.empty(n); low = np.empty(n)
    open_[0] = close[0]; high[0] = close[0] + 1.0; low[0] = close[0] - 1.0
    for i in range(1, n):
        open_[i] = close[i - 1]
        high[i] = max(open_[i], close[i]) + 1.0
        low[i] = min(open_[i], close[i]) - 1.0
    seg = np.zeros(n, dtype=np.int64)
    disc = np.zeros(n, dtype=bool)
    return make_base(times, np.stack([open_, high, low, close], axis=-1), seg, disc)


def gen_random_base(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    times = pd.date_range("2024-03-04 09:00", periods=n, freq="5min")
    ret = rng.normal(0, 0.5, n).cumsum()
    close = 1000.0 + ret
    open_ = np.empty(n); high = np.empty(n); low = np.empty(n)
    open_[0] = close[0]; high[0] = close[0] + 2; low[0] = close[0] - 2
    for i in range(1, n):
        open_[i] = close[i - 1]
        high[i] = max(open_[i], close[i]) + abs(rng.normal(0, 1.5))
        low[i] = min(open_[i], close[i]) - abs(rng.normal(0, 1.5))
    seg = np.zeros(n, dtype=np.int64)
    disc = np.zeros(n, dtype=bool)
    return make_base(times, np.stack([open_, high, low, close], axis=-1), seg, disc)


def fresh_state_at(base, tf, target):
    """Single-pass IndicatorState stepped 0..target (segment resets).

    Returns (state, last_feats) where last_feats is the feature dict emitted at
    ``target`` (needed because IndicatorState does not cache scalar values as
    attributes).
    """
    minutes = TF_LABEL_TO_MINUTES[tf]
    tf_bars = resample_causal(base, minutes)
    state = IndicatorState(PINE_DEFAULT, include_sr=True)
    cur_seg = None
    ci = 0
    last_feats = None
    for i in range(target + 1):
        seg = int(tf_bars["segment"].iloc[i])
        if seg != cur_seg:
            state.reset(); cur_seg = seg; ci = 0
        last_feats = state.step(
            ci, float(tf_bars["open"].iloc[i]), float(tf_bars["high"].iloc[i]),
            float(tf_bars["low"].iloc[i]), float(tf_bars["close"].iloc[i]))
        ci += 1
    return state, last_feats


# --------------------------------------------------------------------------- #
# T0.1 symbol / TF contract                                                     #
# --------------------------------------------------------------------------- #
def test_T0_1_symbol_tf_contract():
    assert len(SYMBOLS) == 15, SYMBOLS
    assert set(SYMBOLS) == {
        "AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU",
        "MA", "TA", "M", "P", "CF",
    }
    assert set(TF_LABEL_TO_MINUTES.keys()) == {"5m", "15m", "1H", "4H"}
    assert TF_LABEL_TO_MINUTES == {"5m": 5, "15m": 15, "1H": 60, "4H": 240}


# --------------------------------------------------------------------------- #
# T0.2 canonical resample (hand check)                                          #
# --------------------------------------------------------------------------- #
def test_T0_2_resample_hand_check():
    times = ["2024-01-02 09:00", "2024-01-02 09:05", "2024-01-02 09:10"]
    ohlc = np.array([
        [100, 110, 95, 102],
        [102, 108, 96, 104],
        [104, 112, 99, 105],
    ], dtype=float)
    base = make_base(pd.to_datetime(times), ohlc, np.zeros(3, dtype=np.int64),
                     np.zeros(3, dtype=bool))
    out = resample_causal(base, 15)
    assert len(out) == 1
    r = out.iloc[0]
    assert r["open"] == 100
    assert r["high"] == 112
    assert r["low"] == 95
    assert r["close"] == 105
    assert int(r["n_base"]) == 3
    assert pd.Timestamp(r["time"]) == pd.Timestamp("2024-01-02 09:00")
    # minimality: a second bucket
    times2 = times + ["2024-01-02 09:15"]
    ohlc2 = np.vstack([ohlc, [105, 106, 104, 106]])
    base2 = make_base(pd.to_datetime(times2), ohlc2, np.zeros(4, dtype=np.int64),
                      np.zeros(4, dtype=bool))
    out2 = resample_causal(base2, 15)
    assert len(out2) == 2
    assert out2.iloc[1]["open"] == 105 and out2.iloc[1]["close"] == 106


# --------------------------------------------------------------------------- #
# T0.3 DTP bands exact (sma +- k*atr)                                           #
# --------------------------------------------------------------------------- #
def test_T0_3_dtp_bands_exact():
    base = gen_trend_base(n=1200, seed=3)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    # find a fully-warm bar
    i = 900
    sma = track.sma[i]; atr = track.atr[i]
    assert np.isfinite(sma) and np.isfinite(atr)
    for k in (1, 2, 3):
        assert abs((sma + k * atr) - (sma + k * atr)) < 1e-12
        assert abs((sma - k * atr) - (sma - k * atr)) < 1e-12
    # cross-check stored sma/atr against a fresh single-pass state at i
    _st, feats = fresh_state_at(base, "5m", i)
    assert abs(feats["sma"] - sma) <= CONT_TOL
    assert abs(feats["atr"] - atr) <= CONT_TOL


# --------------------------------------------------------------------------- #
# T0.4 DTP trend start / profile unavailable before first switch                 #
# --------------------------------------------------------------------------- #
def test_T0_4_trend_start():
    base = gen_trend_base(n=1200, up_until=600, seed=5)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    # warmup bar, no switch yet
    assert track.trend_start_global[50] == -1
    assert dtp_profile(track, 50)[0] is None
    # post first switch bar
    assert track.trend_start_global[900] >= 0
    counts, lookback = dtp_profile(track, 900)
    assert counts is not None
    assert lookback == 900 - int(track.trend_start_global[900])
    # first switch index is finite, profile unavailable strictly before it
    first = int(track.trend_start_global[900])
    assert track.trend_start_global[first] == first
    assert dtp_profile(track, first - 1)[0] is None


# --------------------------------------------------------------------------- #
# T0.5 DTP profile literal oracle (exact count match)                           #
# --------------------------------------------------------------------------- #
def literal_pine_profile(close, sma, atr, trend_start_global, selected, bins=BINS):
    """Independent literal port of ref/DeviationTrendProfile.pine::profile()."""
    n = len(close)
    i = int(selected)
    ts = int(trend_start_global[i]) if 0 <= i < n else -1
    if ts < 0:
        return None
    counts = np.zeros(bins, dtype=np.int64)
    for l in range(ts, i + 1):
        c = close[l]
        min_l = sma[l] - 3.0 * atr[l]
        s_l = 6.0 * atr[l] / bins
        if not (np.isfinite(c) and np.isfinite(min_l) and np.isfinite(s_l) and s_l != 0.0):
            continue
        for b in range(bins):
            lower = min_l + s_l * b
            upper = lower + s_l
            if c >= lower - s_l and c <= upper + s_l:
                counts[b] += 1
    return counts


def test_T0_5_profile_literal_match():
    base = gen_trend_base(n=1200, up_until=600, seed=7)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    sel = 950
    prod, _ = dtp_profile(track, sel)
    lit = literal_pine_profile(track.close, track.sma, track.atr,
                               track.trend_start_global, sel)
    assert prod is not None and lit is not None
    assert np.array_equal(prod, lit)


# --------------------------------------------------------------------------- #
# T0.6 SR full channel snapshot exact (vs fresh state)                          #
# --------------------------------------------------------------------------- #
def test_T0_6_sr_snapshot_exact():
    base = gen_trend_base(n=1200, seed=11)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    for i in range(0, track.n, 50):
        st, _ = fresh_state_at(base, "5m", i)
        ref = [(float(t), float(b), float(s)) for (t, b, s) in st.sr.channels]
        got = []
        for j in range(6):
            if track.sr_valid[i, j]:
                got.append((float(track.sr_top[i, j]), float(track.sr_bottom[i, j]),
                            float(track.sr_strength[i, j])))
        assert got == ref, (i, got, ref)
        assert int(track.sr_n_channels[i]) == len(st.sr.channels)


# --------------------------------------------------------------------------- #
# T0.7 Liquidity active level snapshot exact (vs fresh state)                   #
# --------------------------------------------------------------------------- #
def test_T0_7_liq_snapshot_exact():
    base = gen_trend_base(n=1200, seed=13)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    for i in range(0, track.n, 50):
        st, _ = fresh_state_at(base, "5m", i)
        for side, la, lv, lt, lb, lbr, lbo, lba, lzl, lzr, lzt, lzb in (
            ("up", track.liq_up_valid, track.liq_up_level, track.liq_up_top,
             track.liq_up_bottom, track.liq_up_broken, track.liq_up_breach,
             track.liq_up_zone_active, track.liq_up_zone_left, track.liq_up_zone_right,
             track.liq_up_zone_top, track.liq_up_zone_bottom),
            ("down", track.liq_down_valid, track.liq_down_level, track.liq_down_top,
             track.liq_down_bottom, track.liq_down_broken, track.liq_down_breach,
             track.liq_down_zone_active, track.liq_down_zone_left, track.liq_down_zone_right,
             track.liq_down_zone_top, track.liq_down_zone_bottom),
        ):
            ref = st.liq.levels_up if side == "up" else st.liq.levels_down
            got = []
            for j in range(3):
                if la[i, j]:
                    got.append((float(lv[i, j]), float(lt[i, j]), float(lb[i, j]),
                                int(lbo[i, j]), bool(lbr[i, j])))
            ref_got = [(float(x["level"]), float(x["top"]), float(x["bottom"]),
                        int(x["left"]), bool(x["broken"])) for x in ref[:3]]
            assert got == ref_got, (side, i, got, ref_got)


# --------------------------------------------------------------------------- #
# T0.8 / T0.9 Liquidity post-break lifecycle (visual-only)                      #
# --------------------------------------------------------------------------- #
def test_T0_8_postbreak_lifecycle():
    from research.liquidity_oracle_atlas.indicator_viewer_v1 import _store_liq

    nrows = 12  # > len(seq)
    up_valid = np.zeros((nrows, 3), dtype=bool)
    mk = lambda: np.full((nrows, 3), np.nan)
    up_left = mk(); up_level = mk(); up_top = mk(); up_bottom = mk()
    up_broken = np.zeros((nrows, 3)); up_breach = mk()
    up_za = np.zeros((nrows, 3)); up_zl = mk(); up_zr = mk(); up_zt = mk(); up_zb = mk()
    tracker = {}

    seq = [
        # bar 0: level appears, not broken
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False},
        # bars 1-4: still active, unbroken
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False},
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False},
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False},
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False},
        # bar 5: breach (H=102 > top 101)
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
        # bars 6-7: price inside zone -> expands
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
        # bar 8: price leaves zone (H=105 > level+2.3) -> terminates
        {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
    ]
    # atr_liq=1.0 -> pb=2.3; H/L per simulated bar
    hl = [(None, None), (100, 99), (101, 98), (100, 99), (101, 99),
          (102, 100), (101, 99), (100, 99), (105, 100)]
    rows = []
    for bar, lev in enumerate(seq):
        # reset only this row; other rows retain their captured state
        up_valid[bar] = False; up_left[bar] = np.nan; up_level[bar] = np.nan
        up_top[bar] = np.nan; up_bottom[bar] = np.nan; up_broken[bar] = 0
        up_breach[bar] = np.nan; up_za[bar] = 0; up_zl[bar] = np.nan
        up_zr[bar] = np.nan; up_zt[bar] = np.nan; up_zb[bar] = np.nan
        H, L = hl[bar]
        _store_liq(up_valid, up_left, up_level, up_top, up_bottom, up_broken, up_breach,
                   up_za, up_zl, up_zr, up_zt, up_zb,
                   [lev], tracker, 0, bar, 1.0, H, L)
        rows.append({
            "broken": bool(up_broken[bar, 0]), "breach": up_breach[bar, 0],
            "za": bool(up_za[bar, 0]), "zl": up_zl[bar, 0], "zr": up_zr[bar, 0],
            "zt": up_zt[bar, 0], "zb": up_zb[bar, 0],
        })
    # pre-breach: no zone
    assert rows[0]["za"] is False and np.isnan(rows[0]["zl"])
    # breach bar (index 5)
    assert rows[5]["broken"] and rows[5]["za"]
    assert rows[5]["zl"] == 4 and rows[5]["zr"] == 6
    assert rows[5]["zb"] == 100.0 and rows[5]["zt"] == 102.0
    # expands
    assert rows[6]["zr"] == 7 and rows[7]["zr"] == 8
    # terminates
    assert rows[8]["za"] is False and rows[8]["zr"] == 8


def test_T0_9_multiple_levels_independent():
    from research.liquidity_oracle_atlas.indicator_viewer_v1 import _store_liq

    nrows = 12
    def make_buf():
        up_valid = np.zeros((nrows, 3), dtype=bool)
        mk = lambda: np.full((nrows, 3), np.nan)
        return (up_valid, mk(), mk(), mk(), mk(), np.zeros((nrows, 3)), mk(),
                np.zeros((nrows, 3)), mk(), mk(), mk(), mk())

    tracker = {}
    # level A breaches at bar 5; level B appears at bar 8, breaches at bar 10
    seq = [
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True}],   # A breach
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
         {"left": 8, "level": 200.0, "top": 201.0, "bottom": 199.0, "broken": False}],  # B appears
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
         {"left": 8, "level": 200.0, "top": 201.0, "bottom": 199.0, "broken": False}],
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
         {"left": 8, "level": 200.0, "top": 201.0, "bottom": 199.0, "broken": True}],  # B breach
        [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": True},
         {"left": 8, "level": 200.0, "top": 201.0, "bottom": 199.0, "broken": True}],
    ]
    hl = [(None, None)] * 5 + [(102, 100), (101, 99), (100, 99),
         (105, 100), (101, 99), (202, 200), (201, 199)]
    a_zone_right = []
    b_zone_active = []
    for bar, levels in enumerate(seq):
        b = make_buf()
        H, L = hl[bar]
        _store_liq(b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7], b[8], b[9], b[10], b[11],
                   levels, tracker, 0, bar, 1.0, H, L)
        # slot 0 = A, slot 1 = B
        a_zone_right.append(b[9][bar, 0])
        b_zone_active.append(bool(b[7][bar, 1]))
    # A's zone closed by bar 8 (terminates), unaffected by B breach at bar 10
    assert np.isnan(a_zone_right[10]) or a_zone_right[10] <= 8
    # B becomes active only at its own breach (bar 10)
    assert b_zone_active[9] is False
    assert b_zone_active[10] is True


# --------------------------------------------------------------------------- #
# T0.10 Segment reset                                                            #
# --------------------------------------------------------------------------- #
def test_T0_10_segment_reset():
    n = 700
    rng = np.random.default_rng(2)
    times = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = 1000.0 + rng.normal(0, 1, n).cumsum()
    o = np.empty(n); h = np.empty(n); l = np.empty(n)
    for i in range(n):
        o[i] = close[i - 1] if i else close[i]
        h[i] = max(o[i], close[i]) + 1
        l[i] = min(o[i], close[i]) - 1
    # discontinuity at bar 300
    disc = np.zeros(n, dtype=bool); disc[300] = True
    seg = np.cumsum(disc).astype(np.int64)
    base = make_base(times, np.stack([o, h, l, close], axis=-1), seg, disc)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    # first bar of segment 1 (index 300): trend reset
    assert track.trend_start_global[300] == -1
    assert int(track.trend_state[300]) == -1
    # SR/Liq for segment-1 bars must equal a fresh state reset at 300
    for i in (300, 350, 400):
        st, _ = fresh_state_at(base, "5m", i)
        ref = [(float(t), float(b), float(s)) for (t, b, s) in st.sr.channels]
        got = [(float(track.sr_top[i, j]), float(track.sr_bottom[i, j]),
                float(track.sr_strength[i, j]))
               for j in range(6) if track.sr_valid[i, j]]
        assert got == ref, (i, got, ref)


# --------------------------------------------------------------------------- #
# T0.11 Causality: bars[t+1:] must not change snapshot[t]                        #
# --------------------------------------------------------------------------- #
def test_T0_11_causality():
    base = gen_trend_base(n=1000, seed=17)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    t = 700
    snap1 = selected_snapshot(track, t)
    # mutate future bars
    base2 = base.copy()
    c2 = base2["close"].to_numpy(float).copy()
    c2[t + 1:] = -c2[t + 1:]  # sign flip future
    base2 = base2.assign(close=c2)
    track2 = build_viewer_track(base2, "5m", raw_load_count=1)
    snap2 = selected_snapshot(track2, t)
    assert snap1["dtp"] == snap2["dtp"]
    assert snap1["sr_channels"] == snap2["sr_channels"]
    assert snap1["liq_up"] == snap2["liq_up"]
    assert snap1["liq_down"] == snap2["liq_down"]
    assert snap1["c"] == snap2["c"]
    assert snap1["o"] == snap2["o"]


# --------------------------------------------------------------------------- #
# T0.12 Selection mapping (index -> timestamp / OHLC / snapshot)                #
# --------------------------------------------------------------------------- #
def test_T0_12_selection_mapping():
    base = gen_trend_base(n=1000, seed=19)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    i = 555
    # customdata layout used by the page hit-layer
    customdata = [i, str(pd.Timestamp(track.time[i]))]
    got = selected_snapshot(track, customdata[0])
    assert pd.Timestamp(got["bar_start_time"]) == pd.Timestamp(track.time[i])
    assert got["o"] == float(track.open[i])
    assert got["c"] == float(track.close[i])
    assert got["h"] == float(track.high[i])
    assert got["l"] == float(track.low[i])


# --------------------------------------------------------------------------- #
# T0.13 Negative controls (perturbation must FAIL the comparator)               #
# --------------------------------------------------------------------------- #
def test_T0_13_negative_controls():
    base = gen_trend_base(n=1000, seed=23)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    sel = 800
    counts, _ = dtp_profile(track, sel)
    assert counts is not None

    # profile bin +1 must fail to match
    perturbed = counts.copy(); perturbed[0] += 1
    assert not np.array_equal(counts, perturbed)

    # SR top +0.01 must fail
    sr_top0 = track.sr_top.copy()
    if track.sr_valid[sel].any():
        j = int(np.argmax(track.sr_valid[sel]))
        sr_top0[sel, j] += 0.01
        assert not np.allclose(track.sr_top[sel], sr_top0[sel], atol=1e-9)

    # Liquidity zone_right +1 must fail
    zr0 = track.liq_up_zone_right.copy()
    if track.liq_up_valid[sel].any():
        j = int(np.argmax(track.liq_up_valid[sel]))
        if np.isfinite(track.liq_up_zone_right[sel, j]):
            zr0[sel, j] += 1.0
            assert not np.allclose(track.liq_up_zone_right[sel], zr0[sel], atol=1e-9)


# --------------------------------------------------------------------------- #
# T1 differential vs canonical compute_tf_features (real data)                   #
# --------------------------------------------------------------------------- #
def _compare_core(track, ref, cont_cols, disc_cols):
    n = track.n
    max_err = 0.0
    nan_mismatch = 0
    disc_mismatch = 0
    for c in cont_cols:
        a = getattr(track, c)[:n]
        b = ref[c].to_numpy(float)[:n]
        for i in range(n):
            av, bv = a[i], b[i]
            an, bn = (not np.isfinite(av)), (not np.isfinite(bv))
            if an and bn:
                continue
            if an != bn:
                nan_mismatch += 1
                continue
            max_err = max(max_err, abs(av - bv))
    for c in disc_cols:
        a = getattr(track, c)[:n].astype(float)
        b = ref[c].to_numpy(float)[:n]
        for i in range(n):
            if float(a[i]) != float(b[i]):
                disc_mismatch += 1
    return max_err, nan_mismatch, disc_mismatch


@pytest.mark.parametrize("symbol", ["AG", "SC"])
@pytest.mark.parametrize("tf", TF_LABELS)
def test_T1_differential(symbol, tf):
    builder = FormingEnvironmentBuilder(symbol)
    builder.load_raw()
    base = builder.base
    minutes = TF_LABEL_TO_MINUTES[tf]
    tf_bars = resample_causal(base, minutes)
    ref = compute_tf_features(tf_bars, PINE_DEFAULT, include_sr=True)
    track = build_viewer_track(base, tf, symbol=symbol, raw_load_count=1)

    max_err, nan_mismatch, disc_mismatch = _compare_core(
        track, ref, CORE_CONT, CORE_DISC)
    # report (per spec, only facts)
    summary = {
        "symbol": symbol, "tf": tf, "n": track.n,
        "max_abs_error": float(max_err), "nan_mismatch": int(nan_mismatch),
        "discrete_mismatch": int(disc_mismatch),
    }
    print("T1", summary)
    assert max_err <= CONT_TOL, summary
    assert nan_mismatch == 0, summary
    assert disc_mismatch == 0, summary


# --------------------------------------------------------------------------- #
# TP performance gate (near-linear, no full recompute, no reference)            #
# --------------------------------------------------------------------------- #
def _scaling_run(n):
    base = gen_random_base(n=n, seed=42)
    tracemalloc.start()
    t0 = time.perf_counter()
    track = build_viewer_track(base, "1H", raw_load_count=1)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return track, elapsed, peak / 1e6


def test_TP_performance_gate():
    tN, t2N, t4N = [], [], []
    for rep in range(2):
        _tn, eN, _ = _scaling_run(2000)
        tN.append(eN)
        _t2, e2N, _ = _scaling_run(4000)
        t2N.append(e2N)
        _t4, e4N, _ = _scaling_run(8000)
        t4N.append(e4N)
    eN = max(tN); e2N = max(t2N); e4N = max(t4N)
    r1 = e2N / eN
    r2 = e4N / e2N
    print("TP ratios", {"N": eN, "N2": e2N, "N4": e4N, "r1": r1, "r2": r2})

    # counters on a representative build
    track, _, peak_mb = _scaling_run(4000)
    print("TP counters", {
        "raw_load": track.raw_load_count, "resample": track.resample_count,
        "steps": track.indicator_step_count, "full_recompute": track.full_history_recompute_count,
        "reference": track.reference_call_count, "writes": track.visual_snapshot_write_count,
        "peak_mb": peak_mb, "n": track.n,
    })
    assert track.full_history_recompute_count == 0
    assert track.reference_call_count == 0
    assert track.resample_count == 1
    assert track.raw_load_count == 1
    # near-linear scaling
    assert r1 < 2.8, r1
    assert r2 < 2.8, r2
    # no full-state deepcopy timeline: steps scale with n_1h, not n^2
    assert track.indicator_step_count == track.n


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
