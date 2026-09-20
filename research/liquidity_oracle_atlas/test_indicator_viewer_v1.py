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

# Page module file is `pages/6_Indicator_Viewer.py` (digit-leading name cannot
# be imported via `from ... import` syntax) -> load via importlib.
import importlib as _il
import itertools
import os
import re
import time
import tracemalloc
import types
from statistics import median

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
    LIQ_VISIBLE,
    PROFILE_OFFSET,
    SR_MAX,
    TF_LABEL_TO_MINUTES,
    VIEW_BARS,
    _store_liq,
    build_viewer_track,
    compute_viewport,
    dtp_profile,
    selected_snapshot,
)
from research.liquidity_oracle_atlas.liquidity_source_semantic_oracle_v1 import (
    atr_pine,
    run_liquidity_state_machine,
    unique_confirmed_liq_pivots,
)
from research.liquidity_oracle_atlas.sr_source_semantic_oracle_v1 import (
    run_sr_state_machine,
    unique_confirmed_pivots,
)

_page_mod = _il.import_module("pages.6_Indicator_Viewer")
_parse_selection = _page_mod._parse_selection
build_figure = _page_mod.build_figure
dtp_box_x = _page_mod.dtp_box_x
first_seen_i = _page_mod.first_seen_i
resolve_selection = _page_mod.resolve_selection
segment_start_global = _page_mod.segment_start_global
segment_local_to_global = _page_mod.segment_local_to_global

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

    Returns (state, last_feats).
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


def _run_store(seq, side, atr=1.0, nrows=None):
    """Drive _store_liq over a synthetic sequence.

    ``seq`` is a list of (levels_list, H, L) where each level is a dict with
    keys left/level/top/bottom/broken. Returns per-bar row dicts.
    """
    nrows = nrows or (len(seq) + 2)
    mk = lambda: np.full((nrows, 3), np.nan)
    valid = np.zeros((nrows, 3), dtype=bool)
    left = mk(); level = mk(); top = mk(); bottom = mk()
    broken = np.zeros((nrows, 3)); breach = mk()
    ze = np.zeros((nrows, 3)); za = np.zeros((nrows, 3))
    zl = mk(); zr = mk(); zt = mk(); zb = mk()
    tracker = {}
    rows = []
    for bar, (levels, H, L) in enumerate(seq):
        # Compute breach from the bar's H/L using the SAME strict rule as
        # IndicatorState (H > top for buyside, L < bottom for sellside) and
        # latch it so a breached level stays broken for the rest of the run.
        breached = []
        for lev in levels:
            b = bool(lev.get("broken", False))
            if not b and (
                (side > 0 and H > float(lev["top"]))
                or (side < 0 and L < float(lev["bottom"]))
            ):
                b = True
            if b:
                lev["broken"] = True
            breached.append(dict(lev, broken=b))
        valid[bar] = False; left[bar] = np.nan; level[bar] = np.nan
        top[bar] = np.nan; bottom[bar] = np.nan; broken[bar] = 0
        breach[bar] = np.nan; ze[bar] = 0; za[bar] = 0
        zl[bar] = np.nan; zr[bar] = np.nan; zt[bar] = np.nan; zb[bar] = np.nan
        _store_liq(valid, left, level, top, bottom, broken, breach, ze, za,
                   zl, zr, zt, zb, breached, tracker, 0, bar, side, atr, H, L)
        rows.append(types.SimpleNamespace(
            zone_exists=bool(ze[bar, 0]), zone_active=bool(za[bar, 0]),
            zl=zl[bar, 0], zr=zr[bar, 0], zt=zt[bar, 0], zb=zb[bar, 0],
            broken=bool(broken[bar, 0]), breach=breach[bar, 0],
            za_all=[bool(za[bar, j]) for j in range(3)],
            zr_all=[None if not np.isfinite(zr[bar, j]) else float(zr[bar, j])
                    for j in range(3)],
        ))
    return rows


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
    i = 900
    sma = track.sma[i]; atr = track.atr[i]
    assert np.isfinite(sma) and np.isfinite(atr)
    for k in (1, 2, 3):
        assert abs((sma + k * atr) - (sma + k * atr)) < 1e-12
        assert abs((sma - k * atr) - (sma - k * atr)) < 1e-12
    _st, feats = fresh_state_at(base, "5m", i)
    assert abs(feats["sma"] - sma) <= CONT_TOL
    assert abs(feats["atr"] - atr) <= CONT_TOL


# --------------------------------------------------------------------------- #
# T0.4 DTP trend start / profile unavailable before first switch                 #
# --------------------------------------------------------------------------- #
def test_T0_4_trend_start():
    base = gen_trend_base(n=1200, up_until=600, seed=5)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    assert track.trend_start_global[50] == -1
    assert dtp_profile(track, 50)[0] is None
    assert track.trend_start_global[900] >= 0
    counts, lookback = dtp_profile(track, 900)
    assert counts is not None
    assert lookback == 900 - int(track.trend_start_global[900])
    first = int(track.trend_start_global[900])
    assert track.trend_start_global[first] == first
    assert dtp_profile(track, first - 1)[0] is None


# --------------------------------------------------------------------------- #
# T0.5 DTP profile literal oracle (exact count match)                           #
# --------------------------------------------------------------------------- #
def literal_pine_profile(close, sma, atr, trend_start_global, selected, bins=BINS):
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
             track.liq_up_zone_exists, track.liq_up_zone_left, track.liq_up_zone_right,
             track.liq_up_zone_top, track.liq_up_zone_bottom),
            ("down", track.liq_down_valid, track.liq_down_level, track.liq_down_top,
             track.liq_down_bottom, track.liq_down_broken, track.liq_down_breach,
             track.liq_down_zone_exists, track.liq_down_zone_left, track.liq_down_zone_right,
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
# T0.8 Liquidity post-break lifecycle (BUYSIDE, visual-only)                    #
# --------------------------------------------------------------------------- #
def test_T0_8_postbreak_lifecycle():
    lvl = [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}]
    seq = [(lvl, 100.0, 100.0)] * 5          # appear, unbroken
    seq += [(lvl, 102.0, 100.0)]             # bar 5 breach (H=102 > top 101)
    seq += [(lvl, 100.0, 99.0)]              # bar 6 inside -> expands
    seq += [(lvl, 100.0, 99.0)]              # bar 7 inside -> expands
    seq += [(lvl, 105.0, 100.0)]             # bar 8 leaves (H=105 > level+2.3) -> terminates
    rows = _run_store(seq, side=+1, atr=1.0)
    assert rows[0].zone_exists is False and np.isnan(rows[0].zl)
    assert rows[5].broken and rows[5].zone_active
    assert rows[5].zl == 4 and rows[5].zr == 6
    assert rows[5].zb == 100.0 and rows[5].zt == 102.0
    assert rows[6].zr == 7 and rows[7].zr == 8
    assert rows[8].zone_active is False and rows[8].zr == 8


# --------------------------------------------------------------------------- #
# T0.9 Multiple levels independent lifecycle (BUYSIDE)                          #
# --------------------------------------------------------------------------- #
def test_T0_9_multiple_levels_independent():
    A = {"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}
    B = {"left": 8, "level": 200.0, "top": 201.0, "bottom": 199.0, "broken": False}
    seq = [( [A], 100.0, 100.0)] * 5
    seq += [( [A], 102.0, 100.0)]            # A breach at bar 5
    seq += [( [A], 101.0, 99.0)] * 2
    seq += [( [A, B], 105.0, 99.0)]          # A leaves band -> closes (8); B appears
    seq += [( [A, B], 101.0, 99.0)]          # B not breached yet (9)
    seq += [( [A, B], 202.0, 200.0)]         # B breach at bar 10
    seq += [( [A, B], 201.0, 199.0)]
    rows = _run_store(seq, side=+1, atr=1.0)
    a_zone_right = [r.zr for r in rows]
    b_zone_active = [r.za_all[1] for r in rows]
    assert a_zone_right[10] <= 8
    assert b_zone_active[9] is False
    assert b_zone_active[10] is True


# --------------------------------------------------------------------------- #
# T0.10 Segment reset on a MATURE (switched) segment                            #
# --------------------------------------------------------------------------- #
def test_T0_10_segment_reset_mature():
    n = 1000
    rng = np.random.default_rng(2)
    times = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    # segment 0 (0..950): up -> flat -> up  (mirrors gen_trend_base, whose
    # sandwiched flat region makes the DTP score cross +sw and fire a real UP
    # switch well before the boundary). Segment length 950 > 900 (mature).
    close = np.full(n, 100.0)
    close[0:300] += np.arange(300) * 1.0 + rng.random(300) * 0.2 - 0.1
    close[300:600] += 300.0 + (rng.random(300) * 0.2 - 0.1)
    close[600:950] += 300.0 + np.arange(350) * 1.0 + rng.random(350) * 0.2 - 0.1
    # segment 1 (950..1000): flat
    close[950:] = close[950] + (rng.random(50) * 0.2 - 0.1)

    o = np.empty(n); h = np.empty(n); l = np.empty(n)
    o[0] = close[0]; h[0] = close[0] + 1; l[0] = close[0] - 1
    for i in range(1, n):
        o[i] = close[i - 1]
        h[i] = max(o[i], close[i]) + 1.0
        l[i] = min(o[i], close[i]) - 1.0
    boundary = 950
    disc = np.zeros(n, dtype=bool); disc[boundary] = True
    seg = np.cumsum(disc).astype(np.int64)
    base = make_base(times, np.stack([o, h, l, close], axis=-1), seg, disc)
    track = build_viewer_track(base, "5m", raw_load_count=1)

    # segment 0 length = 950 > 900, with a real switch well before boundary
    assert int(seg[boundary - 1]) == 0
    assert track.trend_start_global[boundary - 1] >= 0
    assert dtp_profile(track, boundary - 1)[0] is not None

    # hard reset at the discontinuity: new segment must NOT inherit old trend
    assert track.trend_start_global[boundary] == -1
    assert dtp_profile(track, boundary)[0] is None
    # segment-1 bars have no switch yet (only 50 bars after reset)
    for i in range(boundary, min(n, boundary + 50)):
        assert track.trend_start_global[i] == -1
        assert dtp_profile(track, i)[0] is None

    # SR/Liq for a segment-1 bar equals a fresh state reset at boundary
    for i in (boundary, boundary + 30, boundary + 49):
        st, _ = fresh_state_at(base, "5m", i)
        ref = [(float(t), float(b), float(s)) for (t, b, s) in st.sr.channels]
        got = [(float(track.sr_top[i, j]), float(track.sr_bottom[i, j]),
                float(track.sr_strength[i, j]))
               for j in range(6) if track.sr_valid[i, j]]
        assert got == ref, (i, got, ref)


# --------------------------------------------------------------------------- #
# T0.SELL.1  Sellside breach geometry (zone_top = level, zone_bottom = max)     #
# --------------------------------------------------------------------------- #
def test_T0_SELL_1_breach_geometry():
    lvl = [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}]
    seq = [(lvl, 100.0, 100.0)] * 5
    seq += [(lvl, 102.0, 98.5)]    # breach bar: H=102>top, L=98.5<bottom
    rows = _run_store(seq, side=-1, atr=1.0)
    r = rows[len(seq) - 1]
    # Sellside: zone_top = level, zone_bottom = max(level - 2.3*atr, L)
    assert r.zt == 100.0
    assert r.zb == max(100.0 - 2.3 * 1.0, 98.5)   # max(97.7, 98.5) = 98.5
    assert r.zone_exists and r.zone_active


# --------------------------------------------------------------------------- #
# T0.SELL.2  Sellside active extension (zone_bottom = min(prev, L))             #
# --------------------------------------------------------------------------- #
def test_T0_SELL_2_active_extension():
    lvl = [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}]
    seq = [(lvl, 100.0, 100.0)] * 5
    seq += [(lvl, 102.0, 98.5)]      # breach (bar 5): L=98.5<99 -> zb=max(97.7,98.5)=98.5
    seq += [(lvl, 100.0, 98.0)]      # active (bar 6): L=98 -> zb=min(98.5,98)=98
    seq += [(lvl, 100.0, 98.5)]      # active (bar 7): L=98.5 -> zb=min(98,98.5)=98
    rows = _run_store(seq, side=-1, atr=1.0)
    assert rows[5].zb == 98.5
    assert rows[6].zb == 98.0
    assert rows[6].zr == 7
    assert rows[7].zb == 98.0
    assert rows[7].zr == 8


# --------------------------------------------------------------------------- #
# T0.SELL.3 + T0.ZONE.PERSIST  Sellside termination keeps frozen geometry        #
# --------------------------------------------------------------------------- #
def test_T0_SELL_3_zone_persist():
    lvl = [{"left": 0, "level": 100.0, "top": 101.0, "bottom": 99.0, "broken": False}]
    seq = [(lvl, 100.0, 100.0)] * 5
    seq += [(lvl, 102.0, 98.5)]      # breach (5): zb=98.5
    seq += [(lvl, 100.0, 98.0)]      # active (6): zb=98
    seq += [(lvl, 100.0, 98.5)]      # active (7): zb=98
    seq += [(lvl, 100.0, 90.0)]      # terminate (8): L=90 < 97.7 -> closed
    rows = _run_store(seq, side=-1, atr=1.0)
    r8 = rows[8]
    assert r8.zone_active is False          # closed
    assert r8.zone_exists is True           # geometry persists
    assert r8.zr == 8                       # frozen at last active
    assert r8.zb == 98.0                    # frozen bottom
    # active -> closed transition preserves existence
    assert rows[7].zone_active is True and rows[7].zone_exists is True


# --------------------------------------------------------------------------- #
# T0.11 Causality: bars[t+1:] must not change snapshot[t]                        #
# --------------------------------------------------------------------------- #
def test_T0_11_causality():
    base = gen_trend_base(n=1000, seed=17)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    t = 700
    snap1 = selected_snapshot(track, t)
    base2 = base.copy()
    c2 = base2["close"].to_numpy(float).copy()
    c2[t + 1:] = -c2[t + 1:]
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
    perturbed = counts.copy(); perturbed[0] += 1
    assert not np.array_equal(counts, perturbed)
    sr_top0 = track.sr_top.copy()
    if track.sr_valid[sel].any():
        j = int(np.argmax(track.sr_valid[sel]))
        sr_top0[sel, j] += 0.01
        assert not np.allclose(track.sr_top[sel], sr_top0[sel], atol=1e-9)
    zr0 = track.liq_up_zone_right.copy()
    if track.liq_up_valid[sel].any():
        j = int(np.argmax(track.liq_up_valid[sel]))
        if np.isfinite(track.liq_up_zone_right[sel, j]):
            zr0[sel, j] += 1.0
            assert not np.allclose(track.liq_up_zone_right[sel], zr0[sel], atol=1e-9)


# --------------------------------------------------------------------------- #
# T0.NO_FUTURE_RENDER  figure must never draw beyond selected bar              #
# --------------------------------------------------------------------------- #
def test_T0_NO_FUTURE_RENDER():
    base = gen_trend_base(n=1000, seed=29)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    for sel in (100, 500, 999):
        fig = build_figure(track, sel, True, True, True)
        candle = fig.data[0]
        assert max(candle.x) <= sel, (sel, max(candle.x))
        # all shapes stay within [lo, selected + right decoration margin]
        _lo, _hi = compute_viewport(track, sel)
        right = sel + PROFILE_OFFSET + 40
        for sh in fig.layout.shapes:
            assert sh.x0 <= right, (sh.x0, right)


# --------------------------------------------------------------------------- #
# T0.CLICK  selection-state parser (simulated Streamlit event)                  #
# --------------------------------------------------------------------------- #
def test_T0_CLICK_selection_parser():
    assert _parse_selection(
        {"selection": {"points": [{"customdata": [123, "2024-01-02 09:00:00"]}]}}
    ) == 123
    # PlotlyState-like object with .selection attribute
    class E:
        def __init__(self):
            self.selection = {"points": [{"customdata": [456, "x"]}]}
    assert _parse_selection(E()) == 456
    assert _parse_selection(None) is None
    assert _parse_selection({"selection": {"points": []}}) is None
    assert _parse_selection({"selection": {}}) is None


# --------------------------------------------------------------------------- #
# T0.SR_SOURCE / T0.LIQ_SOURCE  source-semantic visual differential             #
# --------------------------------------------------------------------------- #
def _notie_base(n=400, seed=7):
    """Deterministic NO-TIE synthetic base (no plateau / equal extrema).

    Strict-unique pivot builtins cannot pick a deterministic winner inside a
    plateau, so those bars are legitimately unverified. This generator keeps
    every window a strict extremum, so no tie masking is needed.
    """
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.35, n))
    o = np.empty(n); h = np.empty(n); l = np.empty(n)
    o[0] = close[0]; h[0] = close[0] + 0.4; l[0] = close[0] - 0.4
    for i in range(1, n):
        o[i] = close[i - 1]
        hi = max(o[i], close[i]); lo = min(o[i], close[i])
        h[i] = hi + 0.05 + 0.10 * rng.random()
        l[i] = lo - 0.05 - 0.10 * rng.random()
    t = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    return make_base(t, np.stack([o, h, l, close], axis=-1),
                     np.zeros(n, dtype=np.int64), np.zeros(n, dtype=bool))


def _two_segment_base(n=1000, boundary=700, seed=13):
    """Two-segment synthetic base: segment 0 = global [0, boundary), 1 = [boundary, n)."""
    rng = np.random.default_rng(seed)
    close = 100.0 + np.cumsum(rng.normal(0, 0.35, n))
    o = np.empty(n); h = np.empty(n); l = np.empty(n)
    o[0] = close[0]; h[0] = close[0] + 0.4; l[0] = close[0] - 0.4
    for i in range(1, n):
        o[i] = close[i - 1]
        hi = max(o[i], close[i]); lo = min(o[i], close[i])
        h[i] = hi + 0.05 + 0.10 * rng.random()
        l[i] = lo - 0.05 - 0.10 * rng.random()
    seg = np.zeros(n, dtype=np.int64)
    seg[boundary:] = 1
    disc = np.zeros(n, dtype=bool)
    disc[boundary] = True
    t = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    return make_base(t, np.stack([o, h, l, close], axis=-1), seg, disc)


def _record_mismatch(rep, field, production, oracle, bar, side):
    """Record first mismatch + bump counter (module-level: no loop closure)."""
    rep["mismatch_count"] += 1
    if rep["first_mismatch"] is None:
        rep["first_mismatch"] = {
            "bar": int(bar), "side": side, "field": field,
            "production": production, "oracle": oracle,
        }


# ---- SR: exact ordered channel differential ------------------------------ #
def _sr_channels(track, i):
    return [
        {"top": float(track.sr_top[i, j]),
         "bottom": float(track.sr_bottom[i, j]),
         "strength": float(track.sr_strength[i, j])}
        for j in range(SR_MAX) if bool(track.sr_valid[i, j])
    ]


def _sr_bar_aligned(base, tf, checkpoints):
    """SR source differential: EXACT ordered channel collection per checkpoint.

    Not `source subset of production`: count / order / top / bottom / strength
    must all match within 1e-9 at every checkpoint.
    """
    track = build_viewer_track(base, tf, raw_load_count=1)
    h = track.high; l = track.low; o = track.open; c = track.close
    ph, pl, _tie = unique_confirmed_pivots(h, l, o, c)
    osr = run_sr_state_machine(ph, pl, h, l, c)
    missing = [f for f in ("hi", "lo", "strength")
               if osr[-1]["channels"] and f not in osr[-1]["channels"][0]]
    rep = {"checkpoints_compared": 0, "channels_compared": 0,
           "fields_compared": ["count", "top", "bottom", "strength"],
           "oracle_missing_fields": missing,
           "mismatch_count": 0, "first_mismatch": None}
    for i in checkpoints:
        P = _sr_channels(track, i)
        O = [{"top": float(x["hi"]), "bottom": float(x["lo"]),
              "strength": float(x["strength"])} for x in osr[i]["channels"]]
        rep["checkpoints_compared"] += 1
        rep["channels_compared"] += max(len(P), len(O))
        if len(P) != len(O):
            _record_mismatch(rep, "CHANNEL_COUNT", len(P), len(O), i, "sr")
            continue
        for k, (a, b) in enumerate(zip(P, O)):
            for f in ("top", "bottom", "strength"):
                if abs(a[f] - b[f]) > CONT_TOL:
                    _record_mismatch(rep, f"{f}[{k}]", a[f], b[f], i, "sr")
    return rep


def test_T0_SR_SOURCE_differential():
    base = _notie_base(n=500, seed=31)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    cps = [int(track.n * f) for f in (0.5, 0.7, 0.85, 0.99)] + [track.n - 1]
    rep = _sr_bar_aligned(base, "5m", cps)
    print("SR_SOURCE", rep)
    assert rep["oracle_missing_fields"] == [], (
        f"source oracle lacks fields: {rep['oracle_missing_fields']}")
    # guard against a vacuous pass (fixture drift -> 0 objects -> 0 mismatch)
    assert rep["checkpoints_compared"] > 0, rep
    assert rep["channels_compared"] > 0, rep
    assert rep["mismatch_count"] == 0, rep


# ---- Liquidity: bar-aligned differential (NO union / subset / superset) --- #
LIQ_FIELDS = ("left", "level", "top", "bottom", "brL", "brZ", "breach_i")


def _liq_levels(track, i, up):
    if up:
        V = track.liq_up_valid; Lft = track.liq_up_left; Lev = track.liq_up_level
        Tp = track.liq_up_top; Bt = track.liq_up_bottom
        Br = track.liq_up_broken; Brch = track.liq_up_breach; Za = track.liq_up_zone_active
    else:
        V = track.liq_down_valid; Lft = track.liq_down_left; Lev = track.liq_down_level
        Tp = track.liq_down_top; Bt = track.liq_down_bottom
        Br = track.liq_down_broken; Brch = track.liq_down_breach; Za = track.liq_down_zone_active
    out = []
    for j in range(LIQ_VISIBLE):
        if not V[i, j]:
            continue
        bi = float(Brch[i, j])
        out.append({"left": int(Lft[i, j]), "level": float(Lev[i, j]),
                    "top": float(Tp[i, j]), "bottom": float(Bt[i, j]),
                    "brL": bool(Br[i, j]), "brZ": bool(Za[i, j]),
                    "breach_i": int(bi) if np.isfinite(bi) else None})
    return out


def _liq_bar_aligned(base, tf):
    """BAR-ALIGNED Liquidity differential against the pinned source oracle.

    Answers: at bar t, are the currently visible levels (and their flags) the
    SAME as TradingView would show? Therefore:
      * compare per bar, in order, field by field;
      * NEVER use union / subset / superset / "appeared at some bar";
      * feed the oracle the PINNED Pine ATR atr_pine(length=10) -- the same
        input production uses (a hand-made np.ones() ATR is not comparable).
    """
    track = build_viewer_track(base, tf, raw_load_count=1)
    h = track.high; l = track.low; o = track.open; c = track.close
    atr = atr_pine(h, l, c, 10)
    ph, pl, tie = unique_confirmed_liq_pivots(h, l, o, c)
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    tie_bars = {int(d["confirm_bar"]) for d in tie}
    rep = {"bars_compared": 0, "levels_compared": 0,
           "fields_compared": list(LIQ_FIELDS),
           "mismatch_count": 0, "first_mismatch": None, "tie_masked_rows": 0}
    for i in range(track.n):
        if not (np.isfinite(atr[i]) and np.isfinite(track.atr_liq[i])):
            continue
        if i in tie_bars:
            rep["tie_masked_rows"] += 1
            continue
        for up in (True, False):
            key = "vis_up" if up else "vis_down"
            side = "up" if up else "down"
            P = _liq_levels(track, i, up)
            O = [{"left": int(x["left"]), "level": float(x["level"]),
                  "top": float(x["top"]), "bottom": float(x["bottom"]),
                  "brL": bool(x["brL"]), "brZ": bool(x["brZ"]),
                  "breach_i": int(x["breach_i"]) if x["breach_i"] is not None else None}
                 for x in sm[i][key]]
            rep["bars_compared"] += 1
            rep["levels_compared"] += max(len(P), len(O))
            if len(P) != len(O):
                _record_mismatch(rep, "LEVEL_COUNT",
                                 [round(q["level"], 6) for q in P],
                                 [round(q["level"], 6) for q in O], i, side)
                continue
            for k, (a, b) in enumerate(zip(P, O)):
                for f in LIQ_FIELDS:
                    va, vb = a[f], b[f]
                    if isinstance(va, float) and isinstance(vb, float):
                        ok = abs(va - vb) <= CONT_TOL
                    else:
                        ok = (va == vb)
                    if not ok:
                        _record_mismatch(rep, f"{f}[{k}]", va, vb, i, side)
    return rep


def test_T0_LIQ_SOURCE_differential():
    base = _notie_base(n=400, seed=7)
    rep = _liq_bar_aligned(base, "5m")
    print("LIQ_SOURCE", rep)
    # guard against a vacuous pass (fixture drift -> 0 objects -> 0 mismatch)
    assert rep["bars_compared"] > 0, rep
    assert rep["levels_compared"] > 0, rep
    assert rep["mismatch_count"] == 0, rep


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
    max_err, nan_mismatch, disc_mismatch = _compare_core(track, ref, CORE_CONT, CORE_DISC)
    summary = {"symbol": symbol, "tf": tf, "n": track.n, "max_abs_error": float(max_err),
                   "nan_mismatch": int(nan_mismatch), "discrete_mismatch": int(disc_mismatch)}
    print("T1", summary)
    assert max_err <= CONT_TOL, summary
    assert nan_mismatch == 0, summary
    assert disc_mismatch == 0, summary


# --------------------------------------------------------------------------- #
# TP performance gate (near-linear, no full recompute, no reference)             #
# --------------------------------------------------------------------------- #
def _scaling_run(n):
    """Build a track and split runtime into canonical resample / viewer work.

    Returns ``(track, t_resample, t_viewer, t_total, peak_mb)`` so the frozen
    complexity gate can be reported per component instead of hiding everything
    inside one opaque number.
    """
    import research.liquidity_oracle_atlas.indicator_viewer_v1 as iv

    base = gen_random_base(n=n, seed=42)
    acc = {"resample": 0.0}
    orig = iv.resample_causal

    def timed_resample(b, minutes):
        t0 = time.perf_counter()
        out = orig(b, minutes)
        acc["resample"] += time.perf_counter() - t0
        return out

    iv.resample_causal = timed_resample
    try:
        tracemalloc.start()
        t0 = time.perf_counter()
        track = build_viewer_track(base, "1H", raw_load_count=1)
        total = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    finally:
        iv.resample_causal = orig
    viewer = max(total - acc["resample"], 0.0)
    return track, acc["resample"], viewer, total, peak / 1e6


# The frozen contract is 2.8 and is NOT adjustable here.
TP_THRESHOLD = 2.8


def test_TP_performance_gate():
    # warmup first (allocator / import / branch cache), then 5 reps + median.
    # We REDUCE MEASUREMENT NOISE instead of relaxing the frozen threshold.
    _scaling_run(2000)

    rows = {}
    for tag, n in (("N", 2000), ("2N", 4000), ("4N", 8000)):
        rs, vs, ts = [], [], []
        for _rep in range(5):
            tr, r, v, t, _p = _scaling_run(n)
            rs.append(r); vs.append(v); ts.append(t)
        rows[tag] = {"resample": median(rs), "viewer": median(vs),
                     "total": median(ts), "n_tf": tr.n}

    eN = rows["N"]["total"]; e2N = rows["2N"]["total"]; e4N = rows["4N"]["total"]
    r1 = e2N / eN
    r2 = e4N / e2N
    print("TP timings", rows)
    print("TP ratios", {"r1": r1, "r2": r2, "threshold": TP_THRESHOLD})

    tr, _r, _v, _t, peak_mb = _scaling_run(4000)
    print("TP counters", {
        "raw_load": tr.raw_load_count, "resample": tr.resample_count,
        "steps": tr.indicator_step_count, "full_recompute": tr.full_history_recompute_count,
        "reference": tr.reference_call_count, "writes": tr.visual_snapshot_write_count,
        "peak_mb": peak_mb, "n": tr.n})
    assert tr.full_history_recompute_count == 0
    assert tr.reference_call_count == 0
    assert tr.resample_count == 1
    assert tr.raw_load_count == 1
    assert tr.indicator_step_count == tr.n
    # frozen complexity contract (NOT relaxed)
    assert r1 < TP_THRESHOLD, {"r1": r1, "rows": rows}
    assert r2 < TP_THRESHOLD, {"r2": r2, "rows": rows}


def test_TP_spy_counters(monkeypatch):
    """Real instrumentation: spy resample_causal and IndicatorState.step,
    and prove the production path never calls slow references / oracles."""
    import research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 as pgm
    import research.liquidity_oracle_atlas.indicator_viewer_v1 as iv
    import research.liquidity_oracle_atlas.liquidity_source_semantic_oracle_v1 as lro
    import research.liquidity_oracle_atlas.sr_source_semantic_oracle_v1 as sro
    from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (
        IndicatorState,
    )

    calls = {"resample": 0, "step": 0}

    orig_res = iv.resample_causal
    def spy_res(base, minutes):
        calls["resample"] += 1
        return orig_res(base, minutes)
    monkeypatch.setattr(iv, "resample_causal", spy_res)

    orig_step = IndicatorState.step
    def spy_step(self, *a, **k):
        calls["step"] += 1
        return orig_step(self, *a, **k)
    monkeypatch.setattr(IndicatorState, "step", spy_step)

    def boom(*a, **k):
        raise AssertionError("slow reference / oracle called from production path")
    monkeypatch.setattr(sro, "run_sr_state_machine", boom)
    monkeypatch.setattr(lro, "run_liquidity_state_machine", boom)
    monkeypatch.setattr(pgm, "compute_tf_features", boom)

    base = gen_random_base(2000)
    track = iv.build_viewer_track(base, "1H", raw_load_count=1)
    # resample called exactly once for the whole build
    assert calls["resample"] == 1, calls
    # selecting a bar must NOT call resample / step / any slow reference
    iv.selected_snapshot(track, 100)
    iv.dtp_profile(track, 100)
    assert calls["resample"] == 1, calls
    # step count equals number of TF bars (single streaming pass)
    assert calls["step"] == track.n, calls


def test_TP_render_perf_real_ag_5m():
    """Render benchmark on the REAL AG / 5m track.

    The track is built ONCE; only ``build_figure()`` is benchmarked, so track
    construction time is never mixed into the render number.
    """
    builder = FormingEnvironmentBuilder("AG")
    builder.load_raw()
    base = builder.base
    t_build0 = time.perf_counter()
    track = build_viewer_track(base, "5m", symbol="AG", raw_load_count=1)
    track_build_sec = time.perf_counter() - t_build0

    def bench(sel):
        ts = []
        for _rep in range(3):
            t0 = time.perf_counter()
            build_figure(track, sel, True, True, True)
            ts.append(time.perf_counter() - t0)
        return min(ts)

    n = track.n
    report = {"symbol": track.symbol, "tf": track.tf_label, "full_N": n,
              "track_build_sec": track_build_sec, "viewport_limit": VIEW_BARS,
              "benchmarks": []}
    for sel in (n - 1, n // 2):
        sec = bench(sel)
        lo, hi = compute_viewport(track, sel)
        fig = build_figure(track, sel, True, True, True)
        bars = hi - lo + 1
        entry = {"selected": sel, "viewport_bars": bars, "traces": len(fig.data),
                 "shapes": len(fig.layout.shapes), "render_sec": sec}
        report["benchmarks"].append(entry)
        assert bars <= VIEW_BARS, entry
    print("TP_RENDER_AG5M", report)


# --------------------------------------------------------------------------- #
# T0.CLICK -- direct K-line selection through the REAL figure hit trace          #
# --------------------------------------------------------------------------- #
def test_T0_CLICK_hit_trace():
    """Prove the figure really exposes a selectable hit target AND that the
    parser round-trips a selection built from that trace's own customdata.

    Unlike the parser-only test, this does NOT hand-write a fake
    ``{"customdata": [123, ...]}`` event.
    """
    base = _notie_base(n=320, seed=11)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    sel = track.n - 1
    fig = build_figure(track, sel, True, True, True)
    lo, hi = compute_viewport(track, sel)
    rendered = hi - lo + 1

    hits = [d for d in fig.data if getattr(d, "name", None) == "iv_hit"]
    assert len(hits) == 1, "figure has no 'iv_hit' selectable trace"
    hit = hits[0]
    # covers exactly the viewport -> one target per rendered candle
    assert len(hit.x) == rendered, (len(hit.x), rendered)
    assert len(hit.customdata) == rendered, (len(hit.customdata), rendered)
    assert len(hit.x) == len(fig.data[0].x), "hit layer must span the candles"
    # each point carries its own TF bar index
    for k in range(rendered):
        assert hit.customdata[k][0] == hit.x[k], k
    assert max(c[0] for c in hit.customdata) == sel

    # round-trip REAL customdata through the parser
    mid = rendered // 2
    real_point = list(hit.customdata[mid])
    parsed = _parse_selection({"selection": {"points": [{"customdata": real_point}]}})
    assert parsed == real_point[0], (parsed, real_point)
    assert parsed == hit.x[mid]


# --------------------------------------------------------------------------- #
# DTP profile visual geometry (Pine box.new orientation + count-driven gradient)  #
# --------------------------------------------------------------------------- #
def test_T0_DTP_PROFILE_GEOMETRY():
    # Pine: start = bar_index + offset ; box.new(start-val, upper, start, lower)
    #       -> the profile box extends to the LEFT of `start`.
    assert dtp_box_x(7, 130) == (123, 130)
    assert dtp_box_x(0, 130) == (130, 130)
    assert dtp_box_x(25, 100) == (75, 100)

    base = gen_trend_base(n=1200, seed=5)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    sel = 900
    fig = build_figure(track, sel, True, False, False)
    start = sel + PROFILE_OFFSET
    boxes = [s for s in fig.layout.shapes
             if s.type == "rect" and s.x1 is not None and int(s.x1) == start]
    assert boxes, "no DTP profile boxes rendered"
    for s in boxes:
        assert int(s.x1) == start, (s.x0, s.x1)
        cnt = start - int(s.x0)
        assert dtp_box_x(cnt, start) == (int(s.x0), int(s.x1)), (s.x0, s.x1)
    # gradient driver must be the bin COUNT (box width), not the bin index:
    # a wider box (more bars) must never be fainter than a narrower one.
    def alpha(s):
        # fillcolor looks like "rgba(18, 209, 235, 0.5)" -> take trailing alpha
        return float(str(s.fillcolor).split(",")[-1].strip(" )"))
    pairs = sorted(((start - int(s.x0), alpha(s)) for s in boxes),
                   key=lambda p: p[0])
    for (c0, a0), (c1, a1) in itertools.pairwise(pairs):
        if c1 > c0:
            assert a1 >= a0, ("opacity must rise with count", pairs)


# --------------------------------------------------------------------------- #
# FIX3: Liquidity `left` is SEGMENT-LOCAL -> must map to the global Plotly x     #
# --------------------------------------------------------------------------- #
def test_FIX3_segment_local_to_global():
    """S = segment start, L = segment-local index -> expected global = S + L.

    Covers BOTH the first segment (S == 0) and a later segment (S > 0).
    """
    base = _two_segment_base(n=1000, boundary=700)
    track = build_viewer_track(base, "5m", raw_load_count=1)

    # --- first segment: S = 0 ---
    S0 = segment_start_global(track, 0)
    assert S0 == 0, S0
    for L in (0, 5, 120, 300):
        assert segment_local_to_global(track, 100, L) == S0 + L, (S0, L)

    # --- later segment: S = 700 ---
    for i in (700, 780, 850, 999):
        assert segment_start_global(track, i) == 700, i
    for L in (0, 80, 250):
        assert segment_local_to_global(track, 800, L) == 700 + L, L

    # identity: ci == global - segment_start_global
    for i in (700, 780, 999):
        assert i - segment_start_global(track, i) == i - 700


def test_FIX3_liq_left_renders_global_x():
    """A liquidity level with left_local=80 in segment 1 (global start 700)
    must be drawn at x0 == 780 -- NOT 80, and NOT collapsed to viewport lo."""
    n, boundary = 1000, 700
    base = _two_segment_base(n=n, boundary=boundary)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    sel = 800
    assert int(track.segment[sel]) == 1
    S = segment_start_global(track, sel)
    assert S == boundary

    LOCAL = 80
    LVL = 123.0
    expected = S + LOCAL  # 780

    # Force a known level (visual-coordinate test, independent of detection).
    track.liq_up_valid[:] = False
    track.liq_down_valid[:] = False
    track.liq_up_valid[sel, 0] = True
    track.liq_up_left[sel, 0] = LOCAL
    track.liq_up_level[sel, 0] = LVL
    track.liq_up_top[sel, 0] = LVL + 0.5
    track.liq_up_bottom[sel, 0] = LVL - 0.5
    track.liq_up_broken[sel, 0] = 0
    track.liq_up_breach[sel, 0] = np.nan
    track.liq_up_zone_exists[sel, 0] = 0
    track.liq_up_zone_active[sel, 0] = 0

    fig = build_figure(track, sel, False, False, True)
    lo, _hi = compute_viewport(track, sel)
    assert lo < expected, {"lo": lo, "expected": expected}

    solids = [s for s in fig.layout.shapes
              if s.type == "line" and (s.line is None or s.line.dash is None)
              and abs(float(s.y0) - LVL) < 1e-9]
    assert solids, "no solid Liquidity line was drawn"
    rendered = sorted(int(s.x0) for s in solids)
    assert expected in rendered, {"expected_x0": expected, "rendered_x0": rendered,
                                  "lo": lo, "S": S, "left_local": LOCAL}
    # must not leak the raw segment-local value, nor collapse to the viewport clamp
    assert LOCAL not in rendered, ("raw segment-local x leaked", rendered)
    assert lo not in rendered, ("collapsed to viewport start", rendered)


# --------------------------------------------------------------------------- #
# Symbol / Timeframe selection contract (pure, no Streamlit)                     #
# --------------------------------------------------------------------------- #
def test_T0_SYMBOL_TF_selection_contract():
    # switch -> latest bar
    assert resolve_selection("AG", "1H", 1000, 500, ("AG", "5m")) == (999, ("AG", "1H"))
    # same context -> keep previous selection
    assert resolve_selection("AG", "1H", 1000, 500, ("AG", "1H")) == (500, ("AG", "1H"))
    # clamped into [0, n-1]
    assert resolve_selection("AG", "1H", 1000, -5, ("AG", "1H"))[0] == 0
    assert resolve_selection("AG", "1H", 1000, 99999, ("AG", "1H"))[0] == 999
    # first visit -> latest bar
    assert resolve_selection("AG", "1H", 1000, None, None)[0] == 999


# --------------------------------------------------------------------------- #
# App navigation regression: every referenced local page path must exist        #
# --------------------------------------------------------------------------- #
def test_app_nav_paths_exist():
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    app_py = os.path.join(repo_root, "app.py")
    with open(app_py) as f:
        src = f.read()
    paths = re.findall(r'st\.Page\(\s*"([^"]+)"', src)
    assert paths, "no st.Page(...) found in app.py"
    # the viewer entry must be present and point at the new page
    assert "pages/6_Indicator_Viewer.py" in paths
    # the alternating page must NOT be referenced (scope contamination removed)
    assert "pages/6_Alternating_Label_Audit.py" not in paths
    for p in paths:
        assert os.path.exists(os.path.join(repo_root, p)), p


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
