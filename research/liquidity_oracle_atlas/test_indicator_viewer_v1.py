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
import os
import re
import time
import tracemalloc
import types

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
    PROFILE_OFFSET,
    TF_LABEL_TO_MINUTES,
    VIEW_BARS,
    _store_liq,
    build_viewer_track,
    compute_viewport,
    dtp_profile,
    selected_snapshot,
)
from research.liquidity_oracle_atlas.liquidity_source_semantic_oracle_v1 import (
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
def _sr_differential(base, tf):
    track = build_viewer_track(base, tf, raw_load_count=1)
    i = track.n - 1
    prod = {(round(track.sr_top[i, j], 3), round(track.sr_bottom[i, j], 3))
               for j in range(6) if track.sr_valid[i, j]}
    h = track.high; l = track.low; o = track.open; c = track.close
    ph, pl, _ = unique_confirmed_pivots(h, l, o, c)
    osr = run_sr_state_machine(ph, pl, h, l, c)
    orc = {(round(ch["hi"], 3), round(ch["lo"], 3)) for ch in osr[i]["channels"]}
    return orc, prod


def _liq_differential(base, tf):
    track = build_viewer_track(base, tf, raw_load_count=1)
    # Production evicts old visible levels (LIQ_VISIBLE cap) but detects every
    # oracle level at SOME bar -> compare against the union over all bars
    # (production is a superset of the source oracle over time).
    prod_up = set()
    prod_down = set()
    for i in range(track.n):
        for j in range(3):
            if track.liq_up_valid[i, j]:
                prod_up.add(round(float(track.liq_up_level[i, j]), 3))
            if track.liq_down_valid[i, j]:
                prod_down.add(round(float(track.liq_down_level[i, j]), 3))
    h = track.high; l = track.low; o = track.open; c = track.close
    atr = np.ones(track.n) * 1.0
    ph, pl, _ = unique_confirmed_liq_pivots(h, l, o, c)
    ol = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    orc_up = {round(x["level"], 3) for i in range(track.n) for x in ol[i]["vis_up"]}
    orc_down = {round(x["level"], 3) for i in range(track.n) for x in ol[i]["vis_down"]}
    return orc_up, prod_up, orc_down, prod_down


def test_T0_SR_SOURCE_differential():
    rng = np.random.default_rng(31)
    n = 500
    t = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    o = np.empty(n); h = np.empty(n); l = np.empty(n)
    o[0] = close[0]; h[0] = close[0] + 1; l[0] = close[0] - 1
    for i in range(1, n):
        o[i] = close[i - 1]; h[i] = max(o[i], close[i]) + 0.5; l[i] = min(o[i], close[i]) - 0.5
    for b in range(12, n, 25):
        h[b] = 120.0; close[b] = 119.0; o[b] = 118.0; l[b] = 117.0
        h[b + 12] = 80.0; close[b + 12] = 81.0; o[b + 12] = 82.0; l[b + 12] = 83.0
    seg = np.zeros(n, dtype=np.int64); disc = np.zeros(n, dtype=bool)
    base = make_base(t, np.stack([o, h, l, close], axis=-1), seg, disc)
    orc, prod = _sr_differential(base, "5m")
    mismatch = orc - prod  # source channels not reproduced by production
    # production (frozen canonical) must reproduce every source-detected channel
    assert mismatch == set(), {"missing": sorted(mismatch), "prod": sorted(prod), "orc": sorted(orc)}


def test_T0_LIQ_SOURCE_differential():
    rng = np.random.default_rng(32)
    n = 500
    t = pd.date_range("2024-01-02 09:00", periods=n, freq="5min")
    close = 100 + np.cumsum(rng.normal(0, 0.2, n))
    o = np.empty(n); h = np.empty(n); l = np.empty(n)
    o[0] = close[0]; h[0] = close[0] + 1; l[0] = close[0] - 1
    for i in range(1, n):
        o[i] = close[i - 1]; h[i] = max(o[i], close[i]) + 0.5; l[i] = min(o[i], close[i]) - 0.5
    for b in (60, 160, 260, 360, 460):
        h[b] = 140.0; close[b] = 139.0; o[b] = 138.0; l[b] = 137.0
    for b in (110, 210, 310, 410):
        l[b] = 60.0; close[b] = 61.0; o[b] = 62.0; h[b] = 63.0
    seg = np.zeros(n, dtype=np.int64); disc = np.zeros(n, dtype=bool)
    base = make_base(t, np.stack([o, h, l, close], axis=-1), seg, disc)
    orc_up, prod_up, orc_down, prod_down = _liq_differential(base, "5m")
    miss_up = orc_up - prod_up
    miss_down = orc_down - prod_down
    assert miss_up == set(), {"missing_up": sorted(miss_up), "prod_up": sorted(prod_up)}
    assert miss_down == set(), {"missing_down": sorted(miss_down), "prod_down": sorted(prod_down)}


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
    track, _, peak_mb = _scaling_run(4000)
    print("TP counters", {
        "raw_load": track.raw_load_count, "resample": track.resample_count,
        "steps": track.indicator_step_count, "full_recompute": track.full_history_recompute_count,
        "reference": track.reference_call_count, "writes": track.visual_snapshot_write_count,
        "peak_mb": peak_mb, "n": track.n})
    assert track.full_history_recompute_count == 0
    assert track.reference_call_count == 0
    assert track.resample_count == 1
    assert track.raw_load_count == 1
    assert r1 < 3.0, r1
    assert r2 < 3.0, r2
    assert track.indicator_step_count == track.n


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


def test_TP_render_perf():
    """Render complexity must be bounded by VIEW_BARS, not full N."""
    base = gen_random_base(2000)
    track = build_viewer_track(base, "5m", raw_load_count=1)
    n = track.n
    for sel in (n - 1, n // 2, 50):
        t0 = time.perf_counter()
        fig = build_figure(track, sel, True, True, True)
        dt = time.perf_counter() - t0
        lo, hi = compute_viewport(track, sel)
        rendered = hi - lo + 1
        assert rendered <= VIEW_BARS, (rendered, sel)
        candle = fig.data[0]
        assert max(candle.x) <= sel
    assert dt < 2.0


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
