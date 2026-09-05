"""Panji canonical indicator calculation bundle (standalone).

Extracted from bao1872/market_dev branch dev at commit:
    4902f8ab4fc65024e2259afe90ebcbc8b9d319f1

Calculation owners copied/inlined:
- dynamic_swing_anchored_vwap.py  (DSA VWAP kernel)
- atr_rope_event_factor_lab_v4.py (ATR Rope dependency used by DSA)
- dsa_selector.py                  (DSA history/bundle SSOT; runtime class excluded)
- smc_pine_core.py                 (SMC + BOS/CHoCH + Order Block lifecycle)
- smc_indicator.py                 (public SMC wrapper)
- sqzmom_lb.py                     (SQZMOM momentum + momentum history)
- first_pyramid_semantics.py       (only MomentumDirection/MomentumChange enums)

Only project/runtime, datasource, CLI and plotting dependencies were removed.  The
indicator formulas, default parameters, causal state transitions, warm-up/NaN
behaviour, BOS/CHoCH semantics, Order Block lifecycle, and SQZMOM trigger logic
are kept from the canonical Panji sources.

Standalone dependencies:
    numpy, pandas
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Any, Literal

import numpy as np
import pandas as pd

logger = logging.getLogger("panji_indicators")

# =============================================================================
# DSA kernel: dynamic_swing_anchored_vwap.py
# =============================================================================

@dataclass
class DSAConfig:
    prd: int = 50
    baseAPT: float = 20.0
    useAdapt: bool = False
    volBias: float = 10.0
    atrLen: int = 50
    line_width: int = 2


def hlc3(df: pd.DataFrame) -> pd.Series:
    return (df["high"] + df["low"] + df["close"]) / 3.0


def atr_wilder(df: pd.DataFrame, n: int) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def alpha_from_apt(apt: float) -> float:
    apt = max(1.0, float(apt))
    decay = np.exp(-np.log(2.0) / apt)
    return float(1.0 - decay)


def format_dsa_time(x) -> str:
    ts = pd.Timestamp(x)
    if (
        ts.hour == 0
        and ts.minute == 0
        and ts.second == 0
        and ts.microsecond == 0
    ):
        return ts.strftime("%Y-%m-%d")
    return ts.isoformat()


def _make_segment(direction: int, points_x: list, points_y: list) -> dict:
    points = [
        {"time": format_dsa_time(x), "value": float(y)}
        for x, y in zip(points_x, points_y)
        if pd.notna(y)
    ]
    return {"direction": int(direction), "points": points}


def dynamic_swing_anchored_vwap(df: pd.DataFrame, cfg: DSAConfig):
    d = df.copy()
    d["hlc3"] = hlc3(d)

    atr = atr_wilder(d, cfg.atrLen)
    atr_avg = atr.ewm(alpha=1 / cfg.atrLen, adjust=False).mean()
    ratio = np.where(atr_avg.values > 0, atr.values / atr_avg.values, 1.0)

    if cfg.useAdapt:
        apt_raw = cfg.baseAPT / np.power(ratio, cfg.volBias)
    else:
        apt_raw = np.full(len(d), cfg.baseAPT, dtype=float)

    apt_clamped = np.clip(apt_raw, 5.0, 300.0)
    apt_series = np.rint(apt_clamped).astype(int)

    high = d["high"].to_numpy(float)
    low = d["low"].to_numpy(float)
    volu = d["volume"].to_numpy(float)
    h3 = d["hlc3"].to_numpy(float)

    n = len(d)
    if n < 2:
        raise ValueError("数据长度太短")

    ph = np.nan
    pl = np.nan
    phL = 0
    plL = 0

    prev = np.nan
    ph_prev_store = np.nan
    pl_prev_store = np.nan

    p = h3[0] * volu[0]
    v = volu[0]

    vwap_out = np.full(n, np.nan, dtype=float)
    dir_out = np.full(n, np.nan, dtype=float)
    pivot_labels = []

    segments = []
    cur_points_x: list[pd.Timestamp] = []
    cur_points_y: list[float] = []
    cur_dir = None
    last_dir = None

    for t in range(n):
        st = 0 if (t - cfg.prd + 1) < 0 else (t - cfg.prd + 1)
        win_h = high[st : t + 1]
        win_l = low[st : t + 1]

        if np.isfinite(high[t]) and high[t] == np.nanmax(win_h):
            ph = high[t]
            phL = t
        if np.isfinite(low[t]) and low[t] == np.nanmin(win_l):
            pl = low[t]
            plL = t

        dir_ = 1 if phL > plL else -1
        dir_out[t] = dir_

        if last_dir is None:
            last_dir = dir_
        if cur_dir is None:
            cur_dir = dir_

        if dir_ != last_dir:
            if len(cur_points_x) >= 2:
                segments.append(_make_segment(last_dir, cur_points_x, cur_points_y))

            x_anchor = plL if dir_ > 0 else phL
            y_anchor = pl if dir_ > 0 else ph

            txt = ""
            if dir_ > 0:
                if np.isfinite(prev):
                    if np.isfinite(y_anchor) and y_anchor < prev:
                        txt = "LL"
                    elif np.isfinite(y_anchor) and y_anchor > prev:
                        txt = "HL"
            else:
                if np.isfinite(prev):
                    if np.isfinite(y_anchor) and y_anchor < prev:
                        txt = "LH"
                    elif np.isfinite(y_anchor) and y_anchor > prev:
                        txt = "HH"

            pivot_labels.append({
                "t": int(x_anchor),
                "x": d.index[x_anchor],
                "y": float(y_anchor) if np.isfinite(y_anchor) else np.nan,
                "text": txt,
                "dir": int(dir_),
            })

            prev = ph_prev_store if dir_ > 0 else pl_prev_store
            p = y_anchor * volu[x_anchor]
            v = volu[x_anchor]

            cur_points_x = []
            cur_points_y = []
            cur_dir = dir_

            for k in range(x_anchor, t + 1):
                alpha = alpha_from_apt(float(apt_series[k]))
                pxv = h3[k] * volu[k]
                v_i = volu[k]
                p = (1.0 - alpha) * p + alpha * pxv
                v = (1.0 - alpha) * v + alpha * v_i
                vv = (p / v) if v > 0 else np.nan
                vwap_out[k] = vv
                cur_points_x.append(d.index[k])
                cur_points_y.append(vv)

            last_dir = dir_
        else:
            alpha = alpha_from_apt(float(apt_series[t]))
            pxv = h3[t] * volu[t]
            v0 = volu[t]
            p = (1.0 - alpha) * p + alpha * pxv
            v = (1.0 - alpha) * v + alpha * v0
            vv = (p / v) if v > 0 else np.nan
            vwap_out[t] = vv
            cur_points_x.append(d.index[t])
            cur_points_y.append(vv)

        ph_prev_store = ph
        pl_prev_store = pl

    if len(cur_points_x) >= 2:
        direction = last_dir if last_dir is not None else cur_dir
        if direction is not None:
            segments.append(_make_segment(direction, cur_points_x, cur_points_y))

    vwap_series = pd.Series(vwap_out, index=d.index, name="DSA_VWAP")
    dir_series = pd.Series(dir_out, index=d.index, name="DSA_DIR")
    return vwap_series, dir_series, pivot_labels, segments


# =============================================================================
# ATR Rope dependency: atr_rope_event_factor_lab_v4.py
# =============================================================================

@dataclass
class ATRRopeConfig:
    length: int = 14
    multi: float = 1.5
    source: str = "close"
    show_ranges: bool = True
    show_atr_channel: bool = False
    up_color: str = "#3daa45"
    down_color: str = "#ff033e"
    flat_color: str = "#004d92"
    range_color: str = "rgba(0,77,146,0.20)"
    regime_lookback: int = 20
    regime_threshold: float = 0.55


def _pine_true_range_np(df: pd.DataFrame) -> np.ndarray:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    pc = np.roll(c, 1)
    pc[0] = np.nan
    tr = np.nanmax(np.vstack([(h - l), np.abs(h - pc), np.abs(l - pc)]), axis=0)
    if len(tr) > 0 and not np.isfinite(tr[0]):
        tr[0] = h[0] - l[0]
    return tr


def _pine_rma_np(values: np.ndarray, length: int) -> np.ndarray:
    n = len(values)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0:
        return out
    vals = np.asarray(values, dtype=float)
    valid_idx = np.where(np.isfinite(vals))[0]
    if len(valid_idx) < length:
        return out
    start = valid_idx[length - 1]
    window_idx = valid_idx[:length]
    out[start] = float(np.nanmean(vals[window_idx]))
    alpha = 1.0 / length
    for i in range(start + 1, n):
        if np.isfinite(vals[i]):
            out[i] = alpha * vals[i] + (1.0 - alpha) * out[i - 1]
        else:
            out[i] = out[i - 1]
    return out


def _ta_cross_np(a: np.ndarray, b: np.ndarray, i: int) -> bool:
    if i <= 0:
        return False
    if not (np.isfinite(a[i]) and np.isfinite(b[i]) and np.isfinite(a[i - 1]) and np.isfinite(b[i - 1])):
        return False
    return (a[i] > b[i] and a[i - 1] <= b[i - 1]) or (a[i] < b[i] and a[i - 1] >= b[i - 1])


def compute_atr_rope(df: pd.DataFrame, cfg: ATRRopeConfig) -> pd.DataFrame:
    out = df.copy().sort_index()
    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"df 缺少字段: {col}")
    if cfg.source not in out.columns:
        raise ValueError(f"source={cfg.source!r} 不在数据列中，可用列: {list(out.columns)}")

    src = out[cfg.source].to_numpy(float)
    src = np.nan_to_num(src, nan=0.0)
    h = out["high"].to_numpy(float)
    l = out["low"].to_numpy(float)
    c = out["close"].to_numpy(float)
    n = len(out)

    tr = _pine_true_range_np(out)
    atr_raw = _pine_rma_np(tr, int(cfg.length))
    atr = atr_raw * float(cfg.multi)

    rope = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    lower = np.full(n, np.nan)
    dir_arr = np.zeros(n, dtype=int)
    c_hi = np.full(n, np.nan)
    c_lo = np.full(n, np.nan)
    h_sum = 0.0
    l_sum = 0.0
    c_count = 0
    ff = True
    ff_arr = np.full(n, False)

    evt_dir_change = np.zeros(n, dtype=bool)
    evt_red_to_blue = np.zeros(n, dtype=bool)
    evt_blue_to_green = np.zeros(n, dtype=bool)
    evt_green_to_blue = np.zeros(n, dtype=bool)
    evt_blue_to_red = np.zeros(n, dtype=bool)
    evt_red_to_green = np.zeros(n, dtype=bool)
    evt_green_to_red = np.zeros(n, dtype=bool)
    evt_turn_up = np.zeros(n, dtype=bool)
    evt_turn_down = np.zeros(n, dtype=bool)
    evt_turn_flat = np.zeros(n, dtype=bool)
    evt_cross_rope = np.zeros(n, dtype=bool)
    evt_line_touch_rope = np.zeros(n, dtype=bool)
    evt_line_cross_up = np.zeros(n, dtype=bool)
    evt_line_cross_down = np.zeros(n, dtype=bool)
    evt_line_retest_green = np.zeros(n, dtype=bool)
    evt_line_retest_red = np.zeros(n, dtype=bool)
    evt_range_start = np.zeros(n, dtype=bool)
    evt_range_touch_high = np.zeros(n, dtype=bool)
    evt_range_touch_low = np.zeros(n, dtype=bool)
    evt_range_break_up = np.zeros(n, dtype=bool)
    evt_range_break_down = np.zeros(n, dtype=bool)
    evt_range_reenter_from_above = np.zeros(n, dtype=bool)
    evt_range_reenter_from_below = np.zeros(n, dtype=bool)

    for i in range(n):
        if i == 0 or not np.isfinite(rope[i - 1]):
            prev_rope = src[i]
        else:
            prev_rope = rope[i - 1]

        threshold_nz = atr[i] if np.isfinite(atr[i]) else 0.0
        move = src[i] - prev_rope
        rope_i = prev_rope + max(abs(move) - threshold_nz, 0.0) * np.sign(move)
        rope[i] = rope_i
        upper[i] = rope_i + atr[i] if np.isfinite(atr[i]) else np.nan
        lower[i] = rope_i - atr[i] if np.isfinite(atr[i]) else np.nan

        prev_dir = int(dir_arr[i - 1]) if i > 0 else 0
        d = prev_dir
        if i > 0 and np.isfinite(rope[i - 1]):
            if rope[i] > rope[i - 1]:
                d = 1
            elif rope[i] < rope[i - 1]:
                d = -1

        crossed = _ta_cross_np(src, rope, i)
        if crossed:
            d = 0
            evt_cross_rope[i] = True
        dir_arr[i] = d

        if i > 0 and d != prev_dir:
            evt_dir_change[i] = True
            if prev_dir == -1 and d == 0:
                evt_red_to_blue[i] = True
            elif prev_dir == 0 and d == 1:
                evt_blue_to_green[i] = True
            elif prev_dir == 1 and d == 0:
                evt_green_to_blue[i] = True
            elif prev_dir == 0 and d == -1:
                evt_blue_to_red[i] = True
            elif prev_dir == -1 and d == 1:
                evt_red_to_green[i] = True
            elif prev_dir == 1 and d == -1:
                evt_green_to_red[i] = True

        if i > 0:
            if d == 1 and prev_dir != 1:
                evt_turn_up[i] = True
            if d == -1 and prev_dir != -1:
                evt_turn_down[i] = True
            if d == 0 and prev_dir != 0:
                evt_turn_flat[i] = True

        if d == 0:
            if i > 0 and prev_dir != 0:
                h_sum = 0.0
                l_sum = 0.0
                c_count = 0
                ff = not ff
                evt_range_start[i] = True
            if np.isfinite(h_sum) and np.isfinite(upper[i]):
                h_sum += upper[i]
            else:
                h_sum = np.nan
            if np.isfinite(l_sum) and np.isfinite(lower[i]):
                l_sum += lower[i]
            else:
                l_sum = np.nan
            c_count += 1
            c_hi[i] = h_sum / c_count if c_count > 0 and np.isfinite(h_sum) else np.nan
            c_lo[i] = l_sum / c_count if c_count > 0 and np.isfinite(l_sum) else np.nan
        else:
            if i > 0:
                c_hi[i] = c_hi[i - 1]
                c_lo[i] = c_lo[i - 1]
        ff_arr[i] = ff

        if np.isfinite(rope[i]):
            evt_line_touch_rope[i] = bool(l[i] <= rope[i] <= h[i])
            if i > 0 and np.isfinite(rope[i - 1]):
                prev_rel = c[i - 1] - rope[i - 1]
                cur_rel = c[i] - rope[i]
                evt_line_cross_up[i] = bool(cur_rel > 0 and prev_rel <= 0)
                evt_line_cross_down[i] = bool(cur_rel < 0 and prev_rel >= 0)
            evt_line_retest_green[i] = bool(d == 1 and evt_line_touch_rope[i])
            evt_line_retest_red[i] = bool(d == -1 and evt_line_touch_rope[i])

        rh = c_hi[i]
        rl = c_lo[i]
        if np.isfinite(rh) and np.isfinite(rl) and rh > rl:
            evt_range_touch_high[i] = bool(l[i] <= rh <= h[i])
            evt_range_touch_low[i] = bool(l[i] <= rl <= h[i])
            if i > 0 and np.isfinite(c_hi[i - 1]) and np.isfinite(c_lo[i - 1]) and c_hi[i - 1] > c_lo[i - 1]:
                prev_rh = c_hi[i - 1]
                prev_rl = c_lo[i - 1]
                prev_close = c[i - 1]
                cur_close = c[i]
                evt_range_break_up[i] = bool(cur_close > rh and prev_close <= prev_rh)
                evt_range_break_down[i] = bool(cur_close < rl and prev_close >= prev_rl)
                evt_range_reenter_from_above[i] = bool(prev_close > prev_rh and rl <= cur_close <= rh)
                evt_range_reenter_from_below[i] = bool(prev_close < prev_rl and rl <= cur_close <= rh)

    dir_prev = np.roll(dir_arr, 1)
    if n:
        dir_prev[0] = 0
    dir_bars = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0 or dir_arr[i] != dir_arr[i - 1]:
            dir_bars[i] = 1
        else:
            dir_bars[i] = dir_bars[i - 1] + 1

    regime_lookback = max(1, int(cfg.regime_lookback))
    regime_threshold = float(cfg.regime_threshold)
    dir_s = pd.Series(dir_arr, index=out.index)
    green_ratio = (dir_s == 1).rolling(regime_lookback, min_periods=1).mean().to_numpy(float)
    red_ratio = (dir_s == -1).rolling(regime_lookback, min_periods=1).mean().to_numpy(float)
    blue_ratio = (dir_s == 0).rolling(regime_lookback, min_periods=1).mean().to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rope_slope_pct = rope / np.roll(rope, regime_lookback) - 1.0
    if n > 0:
        rope_slope_pct[:regime_lookback] = np.nan

    regime = np.zeros(n, dtype=int)
    bull_mask = (green_ratio >= regime_threshold) & (rope_slope_pct > 0)
    bear_mask = (red_ratio >= regime_threshold) & (rope_slope_pct < 0)
    regime[bull_mask] = 1
    regime[bear_mask] = -1

    regime_prev = np.roll(regime, 1)
    if n:
        regime_prev[0] = 0
    regime_bars = np.zeros(n, dtype=int)
    for i in range(n):
        if i == 0 or regime[i] != regime[i - 1]:
            regime_bars[i] = 1
        else:
            regime_bars[i] = regime_bars[i - 1] + 1

    evt_regime_change = np.zeros(n, dtype=bool)
    evt_regime_to_bull = np.zeros(n, dtype=bool)
    evt_regime_to_bear = np.zeros(n, dtype=bool)
    evt_regime_to_range = np.zeros(n, dtype=bool)
    if n > 1:
        evt_regime_change[1:] = regime[1:] != regime[:-1]
        evt_regime_to_bull[1:] = (regime[1:] == 1) & (regime[:-1] != 1)
        evt_regime_to_bear[1:] = (regime[1:] == -1) & (regime[:-1] != -1)
        evt_regime_to_range[1:] = (regime[1:] == 0) & (regime[:-1] != 0)

    regime_strength = np.where(
        regime > 0,
        green_ratio - red_ratio,
        np.where(regime < 0, red_ratio - green_ratio, blue_ratio),
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        rope_dev_pct = c / rope - 1.0
        rope_dev_atr = (c - rope) / atr
        range_mid = (c_hi + c_lo) / 2.0
        range_width = c_hi - c_lo
        range_pos_01 = (c - c_lo) / range_width
        range_width_pct = c_hi / c_lo - 1.0
        range_width_atr = range_width / atr

    out["atr_rope_tr"] = tr
    out["atr_rope_atr_raw"] = atr_raw
    out["atr_rope_atr"] = atr
    out["atr_rope_rope"] = rope
    out["atr_rope_upper"] = upper
    out["atr_rope_lower"] = lower
    out["atr_rope_dir"] = dir_arr
    out["atr_rope_c_hi"] = c_hi
    out["atr_rope_c_lo"] = c_lo
    out["atr_rope_ff"] = ff_arr

    out["evt_atr_rope_dir_change"] = evt_dir_change
    out["evt_atr_rope_dir_red_to_blue"] = evt_red_to_blue
    out["evt_atr_rope_dir_blue_to_green"] = evt_blue_to_green
    out["evt_atr_rope_dir_green_to_blue"] = evt_green_to_blue
    out["evt_atr_rope_dir_blue_to_red"] = evt_blue_to_red
    out["evt_atr_rope_dir_red_to_green"] = evt_red_to_green
    out["evt_atr_rope_dir_green_to_red"] = evt_green_to_red
    out["evt_atr_rope_turn_up"] = evt_turn_up
    out["evt_atr_rope_turn_down"] = evt_turn_down
    out["evt_atr_rope_turn_flat"] = evt_turn_flat

    out["evt_atr_rope_cross_rope"] = evt_cross_rope
    out["evt_atr_rope_line_touch_rope"] = evt_line_touch_rope
    out["evt_atr_rope_line_cross_up"] = evt_line_cross_up
    out["evt_atr_rope_line_cross_down"] = evt_line_cross_down
    out["evt_atr_rope_line_retest_green"] = evt_line_retest_green
    out["evt_atr_rope_line_retest_red"] = evt_line_retest_red

    out["evt_atr_rope_range_start"] = evt_range_start
    out["evt_atr_rope_range_touch_high"] = evt_range_touch_high
    out["evt_atr_rope_range_touch_low"] = evt_range_touch_low
    out["evt_atr_rope_range_break_up"] = evt_range_break_up
    out["evt_atr_rope_range_break_down"] = evt_range_break_down
    out["evt_atr_rope_range_reenter_from_above"] = evt_range_reenter_from_above
    out["evt_atr_rope_range_reenter_from_below"] = evt_range_reenter_from_below

    out["factor_atr_rope_state_dir"] = dir_arr
    out["factor_atr_rope_state_dir_prev"] = dir_prev
    out["factor_atr_rope_state_dir_bars"] = dir_bars
    out["evt_atr_rope_regime_change"] = evt_regime_change
    out["evt_atr_rope_regime_to_bull"] = evt_regime_to_bull
    out["evt_atr_rope_regime_to_bear"] = evt_regime_to_bear
    out["evt_atr_rope_regime_to_range"] = evt_regime_to_range
    out["factor_atr_rope_regime"] = regime
    out["factor_atr_rope_regime_prev"] = regime_prev
    out["factor_atr_rope_regime_bars"] = regime_bars
    out[f"factor_atr_rope_green_ratio_{regime_lookback}"] = green_ratio
    out[f"factor_atr_rope_red_ratio_{regime_lookback}"] = red_ratio
    out[f"factor_atr_rope_blue_ratio_{regime_lookback}"] = blue_ratio
    out[f"factor_atr_rope_slope_pct_{regime_lookback}"] = rope_slope_pct
    out["factor_atr_rope_regime_strength"] = regime_strength
    out["factor_atr_rope_line_dev_pct"] = rope_dev_pct
    out["factor_atr_rope_line_dev_atr"] = rope_dev_atr
    out["factor_atr_rope_range_high"] = c_hi
    out["factor_atr_rope_range_low"] = c_lo
    out["factor_atr_rope_range_mid"] = range_mid
    out["factor_atr_rope_range_pos_01"] = range_pos_01
    out["factor_atr_rope_range_width_pct"] = range_width_pct
    out["factor_atr_rope_range_width_atr"] = range_width_atr
    return out


# =============================================================================
# DSA SSOT post-processing: dsa_selector.py (calculation layer only)
# =============================================================================

MIN_DIR_BARS = 50


def _remove_dsa_lookahead(
    daily_df: pd.DataFrame,
    vwap_series: pd.Series,
    dir_series: pd.Series,
    cfg: DSAConfig | None = None,
) -> tuple[pd.Series, pd.Series]:
    dir_vals = dir_series.fillna(0).astype(int)
    flip_mask = dir_vals != dir_vals.shift(1)
    flip_mask.iloc[0] = False
    flip_indices = daily_df.index[flip_mask].tolist()
    if not flip_indices:
        return vwap_series, dir_series
    if cfg is None:
        cfg = DSAConfig()
    vwap_corrected = vwap_series.copy()
    dir_corrected = dir_series.copy()
    for flip_idx in flip_indices:
        loc = daily_df.index.get_loc(flip_idx)
        if loc < 2:
            continue
        truncated_df = daily_df.iloc[:loc]
        try:
            vwap_trunc, dir_trunc, _, _ = dynamic_swing_anchored_vwap(truncated_df, cfg)
        except Exception as exc:
            logger.debug("截断 DSA 计算异常 flip_idx=%s: %s", flip_idx, exc)
            continue
        common_idx = vwap_trunc.index.intersection(vwap_corrected.index)
        trunc_vals = vwap_trunc.loc[common_idx].astype(float)
        corrected_vals = vwap_corrected.loc[common_idx].astype(float)
        valid_mask = trunc_vals.notna() & corrected_vals.notna()
        diff_mask = (trunc_vals[valid_mask] - corrected_vals[valid_mask]).abs() > 0.001
        replace_idx = diff_mask[diff_mask].index
        vwap_corrected.loc[replace_idx] = trunc_vals.loc[replace_idx]
        dir_corrected.loc[replace_idx] = dir_trunc.loc[replace_idx]
    return vwap_corrected, dir_corrected


def _safe_float(val: Any) -> float | None:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    try:
        f = float(val)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def _safe_date(val: Any) -> str | None:
    if val is None or pd.isna(val):
        return None
    if isinstance(val, pd.Timestamp):
        return val.date().isoformat()
    if isinstance(val, date):
        return val.isoformat()
    try:
        ts = pd.to_datetime(val)
        return ts.date().isoformat() if pd.notna(ts) else None
    except (TypeError, ValueError):
        return None


def _detect_cross_events(close: pd.Series, line: pd.Series, group_id: pd.Series):
    cross_up = pd.Series(False, index=close.index)
    cross_down = pd.Series(False, index=close.index)
    close_prev = close.shift(1)
    line_prev = line.shift(1)
    valid = close.notna() & close_prev.notna() & line.notna() & line_prev.notna()
    cross_up[valid] = (close > line) & (close_prev <= line_prev)
    cross_down[valid] = (close < line) & (close_prev >= line_prev)
    cross_up_count = cross_up.groupby(group_id).cumsum().astype(int)
    cross_down_count = cross_down.groupby(group_id).cumsum().astype(int)

    cross_date = pd.Series(pd.NaT, index=close.index)
    cross_price = pd.Series(np.nan, index=close.index)
    cross_date[cross_up] = close.index[cross_up]
    cross_price[cross_up] = close[cross_up]
    last_cross_up_date = cross_date.groupby(group_id).ffill()
    last_cross_up_price = cross_price.groupby(group_id).ffill()

    cross_date = pd.Series(pd.NaT, index=close.index)
    cross_price = pd.Series(np.nan, index=close.index)
    cross_date[cross_down] = close.index[cross_down]
    cross_price[cross_down] = close[cross_down]
    last_cross_down_date = cross_date.groupby(group_id).ffill()
    last_cross_down_price = cross_price.groupby(group_id).ffill()
    return (
        cross_up, cross_down, cross_up_count, cross_down_count,
        last_cross_up_date, last_cross_up_price,
        last_cross_down_date, last_cross_down_price,
    )


@dataclass
class _DSAHistoryComputation:
    history: pd.DataFrame
    corrected_dir: pd.Series
    pivot_labels: list[dict]
    visual_segments: list[Any]
    effective_frame: pd.DataFrame


def _compute_dsa_history_artifact(bars: pd.DataFrame, config: dict[str, Any]) -> _DSAHistoryComputation:
    empty = _DSAHistoryComputation(
        history=pd.DataFrame(), corrected_dir=pd.Series(dtype=float),
        pivot_labels=[], visual_segments=[], effective_frame=pd.DataFrame(),
    )
    if bars is None or bars.empty or len(bars) < 60:
        return empty

    df = bars.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    dsa_config = config.get("dsa_config", DSAConfig())
    rope_config = config.get("rope_config", ATRRopeConfig(regime_lookback=55))
    min_dir_bars = int(config.get("min_dir_bars", MIN_DIR_BARS))
    lookback = config.get("lookback")
    if lookback is not None and len(df) > lookback:
        df = df.tail(lookback)

    vwap_series, dir_series, pivot_labels, segments = dynamic_swing_anchored_vwap(df, dsa_config)
    vwap_series, dir_series = _remove_dsa_lookahead(df, vwap_series, dir_series, dsa_config)
    dir_vals = dir_series.fillna(0).astype(int)

    change_mask = dir_vals != dir_vals.shift(1)
    change_mask.iloc[0] = True
    group_id = change_mask.cumsum()
    count = group_id.groupby(group_id).cumcount() + 1
    dsa_bars = (count * dir_vals).astype(int)

    regime = pd.Series(0, index=df.index, dtype=int)
    regime[dsa_bars > min_dir_bars] = 1
    regime[dsa_bars < -min_dir_bars] = -1

    trend_transition = pd.Series("NONE", index=df.index, dtype=object)
    _flip_mask = dir_vals != dir_vals.shift(1)
    _flip_mask.iloc[0] = False
    trend_transition[regime == 1] = "UP_CONFIRMED"
    trend_transition[regime == -1] = "DOWN_CONFIRMED"
    _prev_regime = regime.shift(1).fillna(0).astype(int)
    _cur_dir = dir_vals
    _up_to_down = _flip_mask & (_prev_regime == 1) & (_cur_dir == -1)
    _down_to_up = _flip_mask & (_prev_regime == -1) & (_cur_dir == 1)
    _up_broken = _flip_mask & (_cur_dir == -1) & (~_up_to_down) & (regime != -1)
    _down_broken = _flip_mask & (_cur_dir == 1) & (~_down_to_up) & (regime != 1)
    trend_transition[_up_to_down] = "UP_TO_DOWN"
    trend_transition[_down_to_up] = "DOWN_TO_UP"
    trend_transition[_up_broken] = "UP_BROKEN"
    trend_transition[_down_broken] = "DOWN_BROKEN"

    vwap_vals = vwap_series.astype(float)
    vwap_start = vwap_vals.groupby(group_id).transform("first")
    trend_strength = pd.Series(0.0, index=df.index)
    valid_ts = (count > 1) & vwap_start.notna() & vwap_vals.notna() & (vwap_start != 0)
    trend_strength[valid_ts] = (vwap_vals[valid_ts] / vwap_start[valid_ts] - 1) / count[valid_ts]

    close = df["close"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        offset_rate = (close - vwap_vals) / vwap_vals
    offset_rate = offset_rate.replace([np.inf, -np.inf], np.nan)
    offset_mean = offset_rate.groupby(group_id).expanding().mean().reset_index(level=0, drop=True)
    offset_std = offset_rate.groupby(group_id).expanding().std(ddof=0).reset_index(level=0, drop=True)
    offset_percentile = pd.Series(np.nan, index=df.index, dtype=float)
    valid_pct = offset_rate.notna() & offset_mean.notna() & offset_std.notna() & (offset_std > 0)
    zero_std_mask = offset_rate.notna() & offset_mean.notna() & offset_std.notna() & (offset_std == 0)
    if valid_pct.any():
        x = offset_rate[valid_pct].to_numpy()
        mu = offset_mean[valid_pct].to_numpy()
        sigma = offset_std[valid_pct].to_numpy()
        z_scores = (x - mu) / (sigma * math.sqrt(2.0))
        cdf_vals = np.array([0.5 * (1.0 + math.erf(z)) for z in z_scores])
        offset_percentile[valid_pct] = cdf_vals
    if zero_std_mask.any():
        offset_percentile[zero_std_mask] = 0.5

    vwap_ret_total = pd.Series(np.nan, index=df.index, dtype=float)
    vwap_ret_avg = pd.Series(np.nan, index=df.index, dtype=float)
    vwap_ret_5 = pd.Series(np.nan, index=df.index, dtype=float)
    vwap_ret_10 = pd.Series(np.nan, index=df.index, dtype=float)
    vwap_ret_20 = pd.Series(np.nan, index=df.index, dtype=float)
    _pos = group_id.groupby(group_id).cumcount() + 1
    _grsize = group_id.groupby(group_id).transform("size")
    _first = vwap_vals.where(_pos == 1).groupby(group_id).transform("first")
    _finite_first = pd.Series(np.isfinite(_first.to_numpy()), index=_first.index)
    _valid = (_grsize >= 2) & _finite_first & (_first != 0)
    if _valid.any():
        with np.errstate(divide="ignore", invalid="ignore"):
            _total = vwap_vals / _first - 1.0
            _avg = _total / _pos
            _r5 = vwap_vals / vwap_vals.groupby(group_id).shift(5) - 1.0
            _r10 = vwap_vals / vwap_vals.groupby(group_id).shift(10) - 1.0
            _r20 = vwap_vals / vwap_vals.groupby(group_id).shift(20) - 1.0
        vwap_ret_total.loc[_valid] = _total.loc[_valid]
        vwap_ret_avg.loc[_valid] = _avg.loc[_valid]
        vwap_ret_5.loc[_valid] = _r5.loc[_valid]
        vwap_ret_10.loc[_valid] = _r10.loc[_valid]
        vwap_ret_20.loc[_valid] = _r20.loc[_valid]

    with np.errstate(divide="ignore", invalid="ignore"):
        dsa_vwap_dev_pct = (close - vwap_vals) / vwap_vals * 100.0
    change_pct = close.pct_change() * 100.0
    volume = df["volume"].astype(float)
    vol_mean_20 = volume.rolling(window=20, min_periods=1).mean()
    vol_std_20 = volume.rolling(window=20, min_periods=1).std(ddof=0)
    vol_zscore = pd.Series(np.nan, index=df.index, dtype=float)
    valid_vol = vol_std_20 > 0
    vol_zscore[valid_vol] = (volume[valid_vol] - vol_mean_20[valid_vol]) / vol_std_20[valid_vol]
    amount = df["amount"].astype(float)
    avg_amount_20d = amount.rolling(window=20, min_periods=1).mean()

    bar_position = pd.Series(np.arange(len(df), dtype=int), index=df.index)
    segment_id = (group_id - 1).astype(int)
    segment_start_bar_index = bar_position.groupby(group_id).transform("first").astype(int)
    segment_end_bar_index = bar_position.astype(int)
    segment_start_time = pd.Series(df.index, index=df.index).groupby(group_id).transform("first")
    segment_end_time = pd.Series(df.index, index=df.index)
    segment_start_price = close.groupby(group_id).transform("first")
    segment_end_price = close
    segment_bars = count.astype(int)
    with np.errstate(divide="ignore", invalid="ignore"):
        segment_change_pct = (segment_end_price / segment_start_price - 1.0) * 100.0
        segment_slope = segment_change_pct / segment_bars.replace(0, np.nan)

    segment_group_summary = pd.DataFrame({
        "segment_id": segment_id.groupby(group_id).first(),
        "direction": dir_vals.groupby(group_id).first(),
        "start_bar_index": segment_start_bar_index.groupby(group_id).first(),
        "end_bar_index": segment_end_bar_index.groupby(group_id).last(),
        "start_time": segment_start_time.groupby(group_id).first(),
        "end_time": segment_end_time.groupby(group_id).last(),
        "start_price": segment_start_price.groupby(group_id).first(),
        "end_price": segment_end_price.groupby(group_id).last(),
        "bars": segment_bars.groupby(group_id).last(),
        "change_pct": segment_change_pct.groupby(group_id).last(),
        "slope": segment_slope.groupby(group_id).last(),
    })
    previous_segment_summary = segment_group_summary.shift(1)
    def _map_previous(field: str) -> pd.Series:
        return group_id.map(previous_segment_summary[field])
    prev_segment_id = _map_previous("segment_id")
    prev_segment_direction = _map_previous("direction")
    prev_segment_start_bar_index = _map_previous("start_bar_index")
    prev_segment_end_bar_index = _map_previous("end_bar_index")
    prev_segment_start_time = _map_previous("start_time")
    prev_segment_end_time = _map_previous("end_time")
    prev_segment_start_price = _map_previous("start_price")
    prev_segment_end_price = _map_previous("end_price")
    prev_segment_bars = _map_previous("bars")
    prev_segment_change_pct = _map_previous("change_pct")
    prev_segment_slope = _map_previous("slope")

    current_segment_volume_sum = volume.groupby(group_id).cumsum()
    current_segment_amount_sum = amount.groupby(group_id).cumsum()
    current_segment_volume_mean = current_segment_volume_sum / count.replace(0, np.nan)
    current_segment_amount_mean = current_segment_amount_sum / count.replace(0, np.nan)
    _grp_volume_totals = volume.groupby(group_id).sum()
    _grp_amount_totals = amount.groupby(group_id).sum()
    _grp_counts = group_id.groupby(group_id).count()
    _prev_vol_total_by_gid = _grp_volume_totals.shift(1).fillna(0.0)
    _prev_amt_total_by_gid = _grp_amount_totals.shift(1).fillna(0.0)
    _prev_count_by_gid = _grp_counts.shift(1).fillna(0)
    prev_segment_volume_sum = group_id.map(_prev_vol_total_by_gid).astype(float)
    prev_segment_amount_sum = group_id.map(_prev_amt_total_by_gid).astype(float)
    prev_segment_count = group_id.map(_prev_count_by_gid).astype(float)
    prev_segment_volume_mean = prev_segment_volume_sum / prev_segment_count.replace(0, np.nan)
    prev_segment_amount_mean = prev_segment_amount_sum / prev_segment_count.replace(0, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        current_vs_prev_volume_mean_ratio = current_segment_volume_mean / prev_segment_volume_mean.replace(0.0, np.nan)
        current_vs_prev_amount_mean_ratio = current_segment_amount_mean / prev_segment_amount_mean.replace(0.0, np.nan)
        current_vs_prev_volume_ratio = current_segment_volume_sum / prev_segment_volume_sum.replace(0.0, np.nan)
        current_vs_prev_amount_ratio = current_segment_amount_sum / prev_segment_amount_sum.replace(0.0, np.nan)

    rope_dir1_pct = pd.Series(np.nan, index=df.index, dtype=float)
    rope_dir0_pct = pd.Series(np.nan, index=df.index, dtype=float)
    rope_dir_neg1_pct = pd.Series(np.nan, index=df.index, dtype=float)
    touch_rope = pd.Series(False, index=df.index)
    touch_vwap = pd.Series(False, index=df.index)
    atr_rope_rope = pd.Series(np.nan, index=df.index, dtype=float)
    try:
        atr_rope_df = compute_atr_rope(df, rope_config)
        if atr_rope_df is not None and not atr_rope_df.empty:
            atr_rope_dir = atr_rope_df["atr_rope_dir"]
            atr_rope_rope = atr_rope_df["atr_rope_rope"]
            dir1_cum = (atr_rope_dir == 1).groupby(group_id).cumsum()
            dir0_cum = (atr_rope_dir == 0).groupby(group_id).cumsum()
            dir_neg1_cum = (atr_rope_dir == -1).groupby(group_id).cumsum()
            seg_valid_count = atr_rope_dir.notna().groupby(group_id).cumsum()
            safe_seg_count = seg_valid_count.replace(0, np.nan)
            rope_dir1_pct = (dir1_cum / safe_seg_count) * 100.0
            rope_dir0_pct = (dir0_cum / safe_seg_count) * 100.0
            rope_dir_neg1_pct = (dir_neg1_cum / safe_seg_count) * 100.0
            low = df["low"].astype(float)
            valid_touch = atr_rope_rope.notna() & low.notna()
            touch_rope[valid_touch] = low[valid_touch] <= atr_rope_rope[valid_touch]
            touch_vwap[valid_touch] = low[valid_touch] <= vwap_vals[valid_touch]
    except Exception as exc:
        logger.debug("ATR Rope 计算异常: %s", exc)

    (_, _, cross_up_count, cross_down_count,
     last_cross_up_date, last_cross_up_price,
     last_cross_down_date, last_cross_down_price) = _detect_cross_events(close, vwap_vals, group_id)
    (_, _, rope_cross_up_count, rope_cross_down_count,
     rope_cross_up_date, rope_cross_up_price,
     rope_cross_down_date, rope_cross_down_price) = _detect_cross_events(close, atr_rope_rope, group_id)

    result = pd.DataFrame({
        "regime_value": regime,
        "regime_strength": trend_strength,
        "dsa_dir_bars": dsa_bars,
        "trend_transition": trend_transition,
        "offset_rate": offset_rate,
        "offset_mean": offset_mean,
        "offset_std": offset_std,
        "offset_percentile": offset_percentile,
        "vwap_ret_avg": vwap_ret_avg,
        "vwap_ret_total": vwap_ret_total,
        "vwap_ret_5": vwap_ret_5,
        "vwap_ret_10": vwap_ret_10,
        "vwap_ret_20": vwap_ret_20,
        "dsa_vwap": vwap_vals,
        "dsa_vwap_dev_pct": dsa_vwap_dev_pct,
        "change_pct": change_pct,
        "vol_zscore": vol_zscore,
        "avg_amount_20d": avg_amount_20d,
        "segment_id": segment_id,
        "segment_direction": dir_vals,
        "segment_start_bar_index": segment_start_bar_index,
        "segment_end_bar_index": segment_end_bar_index,
        "segment_start_time": segment_start_time,
        "segment_end_time": segment_end_time,
        "segment_start_price": segment_start_price,
        "segment_end_price": segment_end_price,
        "segment_bars": segment_bars,
        "segment_change_pct": segment_change_pct,
        "segment_slope": segment_slope,
        "prev_segment_id": prev_segment_id,
        "prev_segment_direction": prev_segment_direction,
        "prev_segment_start_bar_index": prev_segment_start_bar_index,
        "prev_segment_end_bar_index": prev_segment_end_bar_index,
        "prev_segment_start_time": prev_segment_start_time,
        "prev_segment_end_time": prev_segment_end_time,
        "prev_segment_start_price": prev_segment_start_price,
        "prev_segment_end_price": prev_segment_end_price,
        "prev_segment_bars": prev_segment_bars,
        "prev_segment_change_pct": prev_segment_change_pct,
        "prev_segment_slope": prev_segment_slope,
        "current_segment_volume_sum": current_segment_volume_sum,
        "current_segment_amount_sum": current_segment_amount_sum,
        "current_segment_volume_mean": current_segment_volume_mean,
        "current_segment_amount_mean": current_segment_amount_mean,
        "prev_segment_volume_sum": prev_segment_volume_sum,
        "prev_segment_amount_sum": prev_segment_amount_sum,
        "prev_segment_volume_mean": prev_segment_volume_mean,
        "prev_segment_amount_mean": prev_segment_amount_mean,
        "current_vs_prev_volume_mean_ratio": current_vs_prev_volume_mean_ratio,
        "current_vs_prev_amount_mean_ratio": current_vs_prev_amount_mean_ratio,
        "current_vs_prev_volume_ratio": current_vs_prev_volume_ratio,
        "current_vs_prev_amount_ratio": current_vs_prev_amount_ratio,
        "rope_dir1_pct": rope_dir1_pct,
        "rope_dir0_pct": rope_dir0_pct,
        "rope_dir_neg1_pct": rope_dir_neg1_pct,
        "touch_rope": touch_rope,
        "touch_vwap": touch_vwap,
        "last_cross_up_date": last_cross_up_date,
        "last_cross_up_price": last_cross_up_price,
        "last_cross_down_date": last_cross_down_date,
        "last_cross_down_price": last_cross_down_price,
        "cross_up_count": cross_up_count,
        "cross_down_count": cross_down_count,
        "rope_cross_up_date": rope_cross_up_date,
        "rope_cross_up_price": rope_cross_up_price,
        "rope_cross_down_date": rope_cross_down_date,
        "rope_cross_down_price": rope_cross_down_price,
        "rope_cross_up_count": rope_cross_up_count,
        "rope_cross_down_count": rope_cross_down_count,
    }, index=df.index)
    offset_variance_rate = pd.Series(np.nan, index=df.index, dtype=float)
    valid_var = offset_mean.notna() & offset_std.notna() & offset_mean.abs().gt(1e-10)
    offset_variance_rate[valid_var] = offset_std[valid_var] / offset_mean[valid_var].abs()
    result["offset_variance_rate"] = offset_variance_rate

    return _DSAHistoryComputation(
        history=result, corrected_dir=dir_series, pivot_labels=pivot_labels,
        visual_segments=segments, effective_frame=df,
    )


def compute_dsa_history(bars: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    return _compute_dsa_history_artifact(bars, config).history


def _history_row_to_metrics(row: pd.Series) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "dsa_dir_bars": int(row["dsa_dir_bars"]) if pd.notna(row["dsa_dir_bars"]) else 0,
        "vwap_ret_avg": _safe_float(row["vwap_ret_avg"]),
        "vwap_ret_total": _safe_float(row["vwap_ret_total"]),
        "offset_mean": _safe_float(row["offset_mean"]),
        "offset_std": _safe_float(row["offset_std"]),
        "offset_variance_rate": _safe_float(row["offset_variance_rate"]),
        "offset_percentile": _safe_float(row["offset_percentile"]),
        "regime_value": int(row["regime_value"]) if pd.notna(row["regime_value"]) else 0,
        "regime_strength": _safe_float(row["regime_strength"]),
        "trend_transition": str(row["trend_transition"]) if pd.notna(row["trend_transition"]) else "NONE",
        "offset_rate": _safe_float(row["offset_rate"]),
        "change_pct": _safe_float(row["change_pct"]),
        "touch_rope": bool(row["touch_rope"]) if pd.notna(row["touch_rope"]) else False,
        "touch_vwap": bool(row["touch_vwap"]) if pd.notna(row["touch_vwap"]) else False,
        "rope_dir1_pct": _safe_float(row["rope_dir1_pct"]),
        "rope_dir0_pct": _safe_float(row["rope_dir0_pct"]),
        "rope_dir_neg1_pct": _safe_float(row["rope_dir_neg1_pct"]),
        "cross_up_count": int(row["cross_up_count"]) if pd.notna(row["cross_up_count"]) else 0,
        "cross_down_count": int(row["cross_down_count"]) if pd.notna(row["cross_down_count"]) else 0,
        "last_cross_up_date": _safe_date(row["last_cross_up_date"]),
        "last_cross_down_date": _safe_date(row["last_cross_down_date"]),
        "vwap_ret_5": _safe_float(row["vwap_ret_5"]),
        "vwap_ret_10": _safe_float(row["vwap_ret_10"]),
        "vwap_ret_20": _safe_float(row["vwap_ret_20"]),
        "dsa_vwap": _safe_float(row["dsa_vwap"]),
        "dsa_vwap_dev_pct": _safe_float(row["dsa_vwap_dev_pct"]),
        "vol_zscore": _safe_float(row["vol_zscore"]),
        "avg_amount_20d": _safe_float(row["avg_amount_20d"]),
        "segment_id": int(row["segment_id"]) if pd.notna(row["segment_id"]) else None,
        "segment_direction": int(row["segment_direction"]) if pd.notna(row["segment_direction"]) else None,
        "segment_start_bar_index": int(row["segment_start_bar_index"]) if pd.notna(row["segment_start_bar_index"]) else None,
        "segment_end_bar_index": int(row["segment_end_bar_index"]) if pd.notna(row["segment_end_bar_index"]) else None,
        "segment_start_time": _safe_date(row["segment_start_time"]),
        "segment_end_time": _safe_date(row["segment_end_time"]),
        "segment_start_price": _safe_float(row["segment_start_price"]),
        "segment_end_price": _safe_float(row["segment_end_price"]),
        "segment_bars": int(row["segment_bars"]) if pd.notna(row["segment_bars"]) else None,
        "segment_change_pct": _safe_float(row["segment_change_pct"]),
        "segment_slope": _safe_float(row["segment_slope"]),
        "prev_segment_id": int(row["prev_segment_id"]) if pd.notna(row["prev_segment_id"]) else None,
        "prev_segment_direction": int(row["prev_segment_direction"]) if pd.notna(row["prev_segment_direction"]) else None,
        "prev_segment_start_bar_index": int(row["prev_segment_start_bar_index"]) if pd.notna(row["prev_segment_start_bar_index"]) else None,
        "prev_segment_end_bar_index": int(row["prev_segment_end_bar_index"]) if pd.notna(row["prev_segment_end_bar_index"]) else None,
        "prev_segment_start_time": _safe_date(row["prev_segment_start_time"]),
        "prev_segment_end_time": _safe_date(row["prev_segment_end_time"]),
        "prev_segment_start_price": _safe_float(row["prev_segment_start_price"]),
        "prev_segment_end_price": _safe_float(row["prev_segment_end_price"]),
        "prev_segment_bars": int(row["prev_segment_bars"]) if pd.notna(row["prev_segment_bars"]) else None,
        "prev_segment_change_pct": _safe_float(row["prev_segment_change_pct"]),
        "prev_segment_slope": _safe_float(row["prev_segment_slope"]),
        "current_segment_volume_sum": _safe_float(row["current_segment_volume_sum"]),
        "current_segment_amount_sum": _safe_float(row["current_segment_amount_sum"]),
        "current_segment_volume_mean": _safe_float(row["current_segment_volume_mean"]),
        "current_segment_amount_mean": _safe_float(row["current_segment_amount_mean"]),
        "prev_segment_volume_sum": _safe_float(row["prev_segment_volume_sum"]),
        "prev_segment_amount_sum": _safe_float(row["prev_segment_amount_sum"]),
        "prev_segment_volume_mean": _safe_float(row["prev_segment_volume_mean"]),
        "prev_segment_amount_mean": _safe_float(row["prev_segment_amount_mean"]),
        "current_vs_prev_volume_mean_ratio": _safe_float(row["current_vs_prev_volume_mean_ratio"]),
        "current_vs_prev_amount_mean_ratio": _safe_float(row["current_vs_prev_amount_mean_ratio"]),
        "current_vs_prev_volume_ratio": _safe_float(row["current_vs_prev_volume_ratio"]),
        "current_vs_prev_amount_ratio": _safe_float(row["current_vs_prev_amount_ratio"]),
        "rope_cross_up_date": _safe_date(row["rope_cross_up_date"]),
        "rope_cross_down_date": _safe_date(row["rope_cross_down_date"]),
        "rope_cross_up_price": _safe_float(row["rope_cross_up_price"]),
        "rope_cross_down_price": _safe_float(row["rope_cross_down_price"]),
        "rope_cross_up_count": int(row["rope_cross_up_count"]) if pd.notna(row["rope_cross_up_count"]) else 0,
        "rope_cross_down_count": int(row["rope_cross_down_count"]) if pd.notna(row["rope_cross_down_count"]) else 0,
    }
    return metrics


def compute_dsa_bundle(bars: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    artifact = _compute_dsa_history_artifact(bars, config)
    if artifact.history.empty:
        return {
            "factor_per_bar": pd.DataFrame(), "visual_segments": [],
            "factor_time": pd.DatetimeIndex([]), "pivot_labels": [],
            "anchor": {}, "last_row_metrics": {}, "per_bar": pd.DataFrame(),
        }
    history = artifact.history
    df = artifact.effective_frame
    pivot_labels = artifact.pivot_labels
    segments = artifact.visual_segments
    last_row_metrics = _history_row_to_metrics(history.iloc[-1])
    per_bar = history.copy()
    n = len(per_bar)
    dir_vals = artifact.corrected_dir.fillna(0).astype(int)
    per_bar["dsa_dir"] = dir_vals.values
    per_bar["regime_id"] = per_bar["segment_id"].astype(int)
    pivot_type: list[str | None] = [None] * n
    pivot_price: list[float | None] = [None] * n
    anchor_time: list[str | None] = [None] * n
    index_list = list(df.index)
    for label in pivot_labels:
        t = int(label["t"])
        if 0 <= t < n:
            txt = label.get("text")
            if txt in {"HH", "HL", "LH", "LL"}:
                pivot_type[t] = txt
                pivot_price[t] = float(label["y"]) if np.isfinite(label["y"]) else None
            anchor_time[t] = index_list[t].isoformat()
    per_bar["pivot_type"] = pivot_type
    per_bar["pivot_price"] = pivot_price
    per_bar["anchor_time"] = anchor_time
    anchor = {
        "time": [format_dsa_time(lab["x"]) for lab in pivot_labels],
        "price": [float(lab["y"]) if np.isfinite(lab["y"]) else None for lab in pivot_labels],
        "direction": [int(lab["dir"]) for lab in pivot_labels],
        "type": [lab["text"] or None for lab in pivot_labels],
    }
    return {
        "factor_per_bar": per_bar,
        "visual_segments": segments,
        "factor_time": per_bar.index,
        "pivot_labels": pivot_labels,
        "anchor": anchor,
        "last_row_metrics": last_row_metrics,
        "per_bar": per_bar,
    }


# =============================================================================
# SMC Pine semantic core: smc_pine_core.py + smc_indicator.py
# Order Blocks are part of this canonical core; there is no separate OB formula.
# =============================================================================

BULLISH = 1
BEARISH = -1
BULLISH_LEG = 1
BEARISH_LEG = 0

ATR = "Atr"
RANGE = "Cumulative Mean Range"
CLOSE = "Close"
HIGHLOW = "High/Low"

DEFAULT_PARAMS: dict[str, Any] = {
    "swings_length": 50,
    "equal_length": 3,
    "equal_threshold": 0.1,
    "internal_filter_confluence": False,
    "internal_ob_size": 5,
    "swing_ob_size": 5,
    "order_block_filter": ATR,
    "order_block_mitigation": HIGHLOW,
    "show_internal_order_blocks": True,
    "show_swing_order_blocks": False,
    "show_equal_hl": True,
    "show_high_low_swings": True,
    "show_swings": False,
    "show_internals": True,
    "show_structure": True,
    "show_trend": False,
}


def pine_rma(src: list[float], length: int) -> list[float]:
    n = len(src)
    if n == 0 or length <= 0:
        return [float("nan")] * n
    result = [float("nan")] * n
    if n >= length:
        if length == 1:
            result[0] = src[0]
        else:
            sma_seed = sum(src[:length]) / length
            result[length - 1] = sma_seed
        alpha = 1.0 / length
        for i in range(length, n):
            result[i] = alpha * src[i] + (1.0 - alpha) * result[i - 1]
    return result


def pine_true_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    n = len(highs)
    if n == 0:
        return []
    tr = [0.0] * n
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        prev_close = closes[i - 1]
        tr[i] = max(
            highs[i] - lows[i],
            abs(highs[i] - prev_close),
            abs(lows[i] - prev_close),
        )
    return tr


def pine_atr(highs: list[float], lows: list[float], closes: list[float], length: int) -> list[float]:
    return pine_rma(pine_true_range(highs, lows, closes), length)


def pine_cumulative_mean_range(highs: list[float], lows: list[float], closes: list[float]) -> list[float]:
    tr = pine_true_range(highs, lows, closes)
    n = len(tr)
    if n == 0:
        return []
    result = [float("nan")] * n
    cumsum = 0.0
    for i in range(n):
        cumsum += tr[i]
        if i > 0:
            result[i] = cumsum / i
    return result


def pine_highest(src: list[float], length: int, ref_i: int) -> float:
    start = max(0, ref_i + 1)
    end = min(len(src), ref_i + length + 1)
    if start >= end:
        return src[ref_i] if 0 <= ref_i < len(src) else float("nan")
    return max(src[start:end])


def pine_lowest(src: list[float], length: int, ref_i: int) -> float:
    start = max(0, ref_i + 1)
    end = min(len(src), ref_i + length + 1)
    if start >= end:
        return src[ref_i] if 0 <= ref_i < len(src) else float("nan")
    return min(src[start:end])


def pine_crossover(a_curr: float, a_prev: float, b_curr: float, b_prev: float) -> bool:
    return a_curr > b_curr and a_prev <= b_prev


def pine_crossunder(a_curr: float, a_prev: float, b_curr: float, b_prev: float) -> bool:
    return a_curr < b_curr and a_prev >= b_prev


@dataclass
class _Pivot:
    current_level: float = float("nan")
    last_level: float = float("nan")
    crossed: bool = False
    bar_time: str | None = None
    bar_index: int | None = None


@dataclass
class _Trend:
    bias: int = 0


@dataclass
class _TrailingExtremes:
    top: float = float("nan")
    bottom: float = float("nan")
    bar_time: str | None = None
    bar_index: int | None = None
    last_top_time: str | None = None
    last_bottom_time: str | None = None


@dataclass
class _OrderBlock:
    bar_high: float
    bar_low: float
    bar_time: str
    bar_index: int
    bias: int
    confirmed_index: int
    confirmed_time: str
    mitigated: bool = False
    mitigated_index: int | None = None
    mitigated_time: str | None = None
    entered: bool = False
    enter_index: int | None = None
    enter_time: str | None = None


class _SMCPineState:
    def __init__(
        self,
        opens: list[float],
        highs: list[float],
        lows: list[float],
        closes: list[float],
        times: list[str],
        params: dict[str, Any],
        emit_timeline: bool = False,
    ) -> None:
        self.opens = opens
        self.highs = highs
        self.lows = lows
        self.closes = closes
        self.times = times
        self.params = params
        self.n = len(closes)
        self._emit_timeline = emit_timeline

        self.tr = pine_true_range(highs, lows, closes)
        self.atr200 = pine_rma(self.tr, 200)
        self.cmr = pine_cumulative_mean_range(highs, lows, closes)
        self.volatility_measure = self.atr200 if params["order_block_filter"] == ATR else self.cmr
        self.parsed_highs, self.parsed_lows = self._compute_parsed_high_low()

        self.swing_high = _Pivot()
        self.swing_low = _Pivot()
        self.internal_high = _Pivot()
        self.internal_low = _Pivot()
        self.equal_high = _Pivot()
        self.equal_low = _Pivot()

        self.swing_trend = _Trend()
        self.internal_trend = _Trend()
        self.trailing = _TrailingExtremes()
        self.internal_order_blocks: list[_OrderBlock] = []
        self.swing_order_blocks: list[_OrderBlock] = []
        self.events: list[dict[str, Any]] = []
        self.equal_highs_lows: list[dict[str, Any]] = []
        self.pivots: list[dict[str, Any]] = []
        self.order_blocks_output: list[dict[str, Any]] = []
        self.ob_lifecycle_events: list[dict[str, Any]] = []
        self.state_timeline: list[dict[str, Any]] = []
        self.leg_states: dict[tuple[str, int], dict[int, int]] = {}

    def _compute_parsed_high_low(self) -> tuple[list[float], list[float]]:
        n = self.n
        parsed_high = [0.0] * n
        parsed_low = [0.0] * n
        for i in range(n):
            vol = self.volatility_measure[i]
            if vol != vol:
                high_vol_bar = False
            else:
                high_vol_bar = (self.highs[i] - self.lows[i]) >= 2.0 * vol
            if high_vol_bar:
                parsed_high[i] = self.lows[i]
                parsed_low[i] = self.highs[i]
            else:
                parsed_high[i] = self.highs[i]
                parsed_low[i] = self.lows[i]
        return parsed_high, parsed_low

    def leg(self, i: int, size: int, lane: str) -> int:
        if i < size:
            return 0
        state_map = self.leg_states.setdefault((lane, size), {})
        if i in state_map:
            return state_map[i]
        prev = state_map.get(i - 1, 0)
        ref_i = i - size
        new_leg_high = self.highs[ref_i] > pine_highest(self.highs, size, ref_i)
        new_leg_low = self.lows[ref_i] < pine_lowest(self.lows, size, ref_i)
        out = prev
        if new_leg_high:
            out = BEARISH_LEG
        elif new_leg_low:
            out = BULLISH_LEG
        state_map[i] = out
        return out

    def start_of_new_leg(self, i: int, size: int, lane: str) -> bool:
        return i >= size and self.leg(i, size, lane) != self.leg(i - 1, size, lane)

    def start_of_bearish_leg(self, i: int, size: int, lane: str) -> bool:
        return i >= size and (self.leg(i, size, lane) - self.leg(i - 1, size, lane) == -1)

    def start_of_bullish_leg(self, i: int, size: int, lane: str) -> bool:
        return i >= size and (self.leg(i, size, lane) - self.leg(i - 1, size, lane) == 1)

    def get_current_structure(
        self,
        i: int,
        size: int,
        equal_high_low: bool = False,
        internal: bool = False,
    ) -> None:
        if i < size:
            return

        lane = "equal" if equal_high_low else "internal" if internal else "swing"
        new_pivot = self.start_of_new_leg(i, size, lane)
        pivot_low = self.start_of_bullish_leg(i, size, lane)
        pivot_high = self.start_of_bearish_leg(i, size, lane)
        atr_measure = self.atr200[i] if i < self.n else float("nan")
        if not new_pivot:
            return
        ref_i = i - size

        if pivot_low:
            piv = self.equal_low if equal_high_low else self.internal_low if internal else self.swing_low
            level = self.lows[ref_i]
            if (
                equal_high_low
                and piv.current_level == piv.current_level
                and atr_measure == atr_measure
                and abs(piv.current_level - level) < self.params["equal_threshold"] * atr_measure
            ):
                self.equal_highs_lows.append({
                    "type": "EQL",
                    "anchor_index": piv.bar_index,
                    "anchor_time": piv.bar_time,
                    "second_pivot_index": ref_i,
                    "second_pivot_time": self.times[ref_i],
                    "confirmed_index": i,
                    "confirmed_time": self.times[i],
                    "level": level,
                    "prev_level": piv.current_level,
                })
            self._record_pivot(piv, level, ref_i, i, "low", internal, equal_high_low)
            piv.last_level = piv.current_level
            piv.current_level = level
            piv.crossed = False
            piv.bar_time = self.times[ref_i]
            piv.bar_index = ref_i
            if not equal_high_low and not internal:
                self.trailing.bottom = piv.current_level
                self.trailing.bar_time = piv.bar_time
                self.trailing.bar_index = piv.bar_index
                self.trailing.last_bottom_time = piv.bar_time

        elif pivot_high:
            piv = self.equal_high if equal_high_low else self.internal_high if internal else self.swing_high
            level = self.highs[ref_i]
            if (
                equal_high_low
                and piv.current_level == piv.current_level
                and atr_measure == atr_measure
                and abs(piv.current_level - level) < self.params["equal_threshold"] * atr_measure
            ):
                self.equal_highs_lows.append({
                    "type": "EQH",
                    "anchor_index": piv.bar_index,
                    "anchor_time": piv.bar_time,
                    "second_pivot_index": ref_i,
                    "second_pivot_time": self.times[ref_i],
                    "confirmed_index": i,
                    "confirmed_time": self.times[i],
                    "level": level,
                    "prev_level": piv.current_level,
                })
            self._record_pivot(piv, level, ref_i, i, "high", internal, equal_high_low)
            piv.last_level = piv.current_level
            piv.current_level = level
            piv.crossed = False
            piv.bar_time = self.times[ref_i]
            piv.bar_index = ref_i
            if not equal_high_low and not internal:
                self.trailing.top = piv.current_level
                self.trailing.bar_time = piv.bar_time
                self.trailing.bar_index = piv.bar_index
                self.trailing.last_top_time = piv.bar_time

    def _record_pivot(
        self,
        piv: _Pivot,
        level: float,
        ref_i: int,
        confirmed_i: int,
        kind: Literal["high", "low"],
        internal: bool,
        equal_high_low: bool,
    ) -> None:
        pivot_type = (
            "equal_high" if equal_high_low and kind == "high"
            else "equal_low" if equal_high_low
            else "internal_high" if internal and kind == "high"
            else "internal_low" if internal
            else "swing_high" if kind == "high"
            else "swing_low"
        )
        self.pivots.append({
            "type": pivot_type,
            "anchor_index": ref_i,
            "anchor_time": self.times[ref_i],
            "confirmed_index": confirmed_i,
            "confirmed_time": self.times[confirmed_i],
            "level": level,
            "last_level": piv.current_level if piv.current_level == piv.current_level else None,
        })

    def display_structure(
        self,
        i: int,
        internal: bool = False,
        prev_levels: dict[str, float] | None = None,
    ) -> None:
        if i <= 0 or i >= self.n:
            return
        close_prev = self.closes[i - 1]
        close_curr = self.closes[i]
        bullish_bar = True
        bearish_bar = True
        if self.params["internal_filter_confluence"]:
            row_high = self.highs[i]
            row_low = self.lows[i]
            row_open = self.opens[i]
            row_close = self.closes[i]
            bullish_bar = (row_high - max(row_close, row_open)) > min(row_close, row_open - row_low)
            bearish_bar = (row_high - max(row_close, row_open)) < min(row_close, row_open - row_low)

        piv_high = self.internal_high if internal else self.swing_high
        trd = self.internal_trend if internal else self.swing_trend
        level_curr_high = piv_high.current_level
        level_prev_high = (
            prev_levels["internal_high" if internal else "swing_high"]
            if prev_levels is not None else level_curr_high
        )
        extra_condition = (
            (piv_high.current_level != self.swing_high.current_level) and bullish_bar
            if internal else True
        )
        if (
            piv_high.current_level == piv_high.current_level
            and pine_crossover(close_curr, close_prev, level_curr_high, level_prev_high)
            and not piv_high.crossed
            and extra_condition
        ):
            tag = "CHoCH" if trd.bias == BEARISH else "BOS"
            piv_high.crossed = True
            trd.bias = BULLISH
            self.events.append({
                "type": tag,
                "internal": internal,
                "bullish": True,
                "bias": BULLISH,
                "anchor_index": piv_high.bar_index,
                "anchor_time": piv_high.bar_time,
                "confirmed_index": i,
                "confirmed_time": self.times[i],
                "level": piv_high.current_level,
            })
            self.store_order_block(piv_high, i, internal, BULLISH)

        piv_low = self.internal_low if internal else self.swing_low
        level_curr_low = piv_low.current_level
        level_prev_low = (
            prev_levels["internal_low" if internal else "swing_low"]
            if prev_levels is not None else level_curr_low
        )
        extra_condition = (
            (piv_low.current_level != self.swing_low.current_level) and bearish_bar
            if internal else True
        )
        if (
            piv_low.current_level == piv_low.current_level
            and pine_crossunder(close_curr, close_prev, level_curr_low, level_prev_low)
            and not piv_low.crossed
            and extra_condition
        ):
            tag = "CHoCH" if trd.bias == BULLISH else "BOS"
            piv_low.crossed = True
            trd.bias = BEARISH
            self.events.append({
                "type": tag,
                "internal": internal,
                "bullish": False,
                "bias": BEARISH,
                "anchor_index": piv_low.bar_index,
                "anchor_time": piv_low.bar_time,
                "confirmed_index": i,
                "confirmed_time": self.times[i],
                "level": piv_low.current_level,
            })
            self.store_order_block(piv_low, i, internal, BEARISH)

    def store_order_block(self, piv: _Pivot, current_i: int, internal: bool, bias: int) -> None:
        if piv.bar_index is None:
            return
        if internal and not self.params["show_internal_order_blocks"]:
            return
        if (not internal) and not self.params["show_swing_order_blocks"]:
            return
        start = piv.bar_index
        end = current_i
        if end <= start:
            return
        if bias == BEARISH:
            arr = self.parsed_highs[start:end]
            if not arr:
                return
            local_idx = arr.index(max(arr))
        else:
            arr = self.parsed_lows[start:end]
            if not arr:
                return
            local_idx = arr.index(min(arr))
        parsed_index = start + local_idx
        ob = _OrderBlock(
            bar_high=float(self.parsed_highs[parsed_index]),
            bar_low=float(self.parsed_lows[parsed_index]),
            bar_time=self.times[parsed_index],
            bar_index=parsed_index,
            bias=bias,
            confirmed_index=current_i,
            confirmed_time=self.times[current_i],
        )
        target = self.internal_order_blocks if internal else self.swing_order_blocks
        if len(target) >= 100:
            target.pop()
        target.insert(0, ob)
        self.ob_lifecycle_events.append({
            "type": "OB_CREATED",
            "internal": internal,
            "bias": bias,
            "anchor_index": parsed_index,
            "anchor_time": self.times[parsed_index],
            "confirmed_index": current_i,
            "confirmed_time": self.times[current_i],
            "bar_high": ob.bar_high,
            "bar_low": ob.bar_low,
            "structure_level": "internal" if internal else "swing",
        })
        self.order_blocks_output.insert(0, {
            "internal": internal,
            "bias": bias,
            "anchor_index": parsed_index,
            "anchor_time": self.times[parsed_index],
            "confirmed_index": current_i,
            "confirmed_time": self.times[current_i],
            "bar_high": ob.bar_high,
            "bar_low": ob.bar_low,
            "mitigated": False,
            "mitigated_index": None,
            "mitigated_time": None,
            "entered": False,
            "enter_index": None,
            "enter_time": None,
            "_ob_ref": id(ob),
        })

    def delete_order_blocks(self, i: int, internal: bool = False) -> None:
        obs = self.internal_order_blocks if internal else self.swing_order_blocks
        mitigation_src_high = self.closes[i] if self.params["order_block_mitigation"] == CLOSE else self.highs[i]
        mitigation_src_low = self.closes[i] if self.params["order_block_mitigation"] == CLOSE else self.lows[i]
        kept: list[_OrderBlock] = []
        for ob in obs:
            crossed = False
            if mitigation_src_high > ob.bar_high and ob.bias == BEARISH:
                crossed = True
            elif mitigation_src_low < ob.bar_low and ob.bias == BULLISH:
                crossed = True
            if crossed:
                ob.mitigated = True
                ob.mitigated_index = i
                ob.mitigated_time = self.times[i]
                self.ob_lifecycle_events.append({
                    "type": "OB_MITIGATED",
                    "internal": internal,
                    "bias": ob.bias,
                    "anchor_index": ob.bar_index,
                    "anchor_time": ob.bar_time,
                    "confirmed_index": ob.confirmed_index,
                    "confirmed_time": ob.confirmed_time,
                    "mitigated_index": i,
                    "mitigated_time": self.times[i],
                    "bar_high": ob.bar_high,
                    "bar_low": ob.bar_low,
                    "structure_level": "internal" if internal else "swing",
                    "entered_before_mitigation": ob.entered,
                    "enter_index": ob.enter_index,
                    "enter_time": ob.enter_time,
                })
                for out in self.order_blocks_output:
                    if out.get("_ob_ref") == id(ob):
                        out["mitigated"] = True
                        out["mitigated_index"] = i
                        out["mitigated_time"] = self.times[i]
                        break
            else:
                kept.append(ob)
        if internal:
            self.internal_order_blocks = kept
        else:
            self.swing_order_blocks = kept

    def check_ob_entered(self, i: int, internal: bool = False) -> None:
        if i < 1:
            return
        obs = self.internal_order_blocks if internal else self.swing_order_blocks
        prev_low = self.lows[i - 1]
        prev_high = self.highs[i - 1]
        cur_low = self.lows[i]
        cur_high = self.highs[i]
        structure_level = "internal" if internal else "swing"
        for ob in obs:
            if ob.entered or ob.mitigated:
                continue
            if i <= ob.confirmed_index:
                continue
            prev_no_overlap = (prev_low > ob.bar_high) or (prev_high < ob.bar_low)
            if not prev_no_overlap:
                continue
            cur_overlap = (cur_low <= ob.bar_high) and (cur_high >= ob.bar_low)
            if not cur_overlap:
                continue
            ob.entered = True
            ob.enter_index = i
            ob.enter_time = self.times[i]
            self.ob_lifecycle_events.append({
                "type": "OB_ENTERED",
                "internal": internal,
                "bias": ob.bias,
                "anchor_index": ob.bar_index,
                "anchor_time": ob.bar_time,
                "confirmed_index": ob.confirmed_index,
                "confirmed_time": ob.confirmed_time,
                "enter_index": i,
                "enter_time": self.times[i],
                "bar_high": ob.bar_high,
                "bar_low": ob.bar_low,
                "structure_level": structure_level,
            })
            for out in self.order_blocks_output:
                if out.get("_ob_ref") == id(ob):
                    out["entered"] = True
                    out["enter_index"] = i
                    out["enter_time"] = self.times[i]
                    break

    def update_trailing_extremes(self, i: int) -> None:
        if self.trailing.bar_index is None:
            return
        if i >= self.n:
            return
        if self.trailing.top == self.trailing.top:
            if self.highs[i] >= self.trailing.top:
                self.trailing.top = self.highs[i]
                self.trailing.last_top_time = self.times[i]
        if self.trailing.bottom == self.trailing.bottom:
            if self.lows[i] <= self.trailing.bottom:
                self.trailing.bottom = self.lows[i]
                self.trailing.last_bottom_time = self.times[i]

    def run(self) -> None:
        swings_length = self.params["swings_length"]
        equal_length = self.params["equal_length"]
        show_equal_hl = self.params["show_equal_hl"]
        show_high_low_swings = self.params["show_high_low_swings"]
        show_internal_order_blocks = self.params["show_internal_order_blocks"]
        show_swing_order_blocks = self.params["show_swing_order_blocks"]
        show_internals = self.params.get("show_internals", True)
        show_structure = self.params.get("show_structure", True)
        show_trend = self.params.get("show_trend", True)
        internal_gate = show_internals or show_internal_order_blocks or show_trend
        swing_gate = show_structure or show_swing_order_blocks or show_high_low_swings

        for i in range(self.n):
            prev_levels: dict[str, float] = {
                "swing_high": self.swing_high.current_level,
                "swing_low": self.swing_low.current_level,
                "internal_high": self.internal_high.current_level,
                "internal_low": self.internal_low.current_level,
            }
            if show_high_low_swings and self.trailing.bar_index is not None:
                self.update_trailing_extremes(i)
            self.get_current_structure(i, swings_length, False, False)
            self.get_current_structure(i, 5, False, True)
            if show_equal_hl:
                self.get_current_structure(i, equal_length, True, False)
            if internal_gate:
                self.display_structure(i, True, prev_levels)
            if swing_gate:
                self.display_structure(i, False, prev_levels)
            if show_internal_order_blocks:
                self.check_ob_entered(i, True)
            if show_swing_order_blocks:
                self.check_ob_entered(i, False)
            if show_internal_order_blocks:
                self.delete_order_blocks(i, True)
            if show_swing_order_blocks:
                self.delete_order_blocks(i, False)
            if self._emit_timeline:
                self.state_timeline.append({
                    "bar_index": i,
                    "time": self.times[i],
                    "swing_bias": self.swing_trend.bias,
                    "internal_bias": self.internal_trend.bias,
                    "active_internal_ob_count": len(self.internal_order_blocks),
                    "active_swing_ob_count": len(self.swing_order_blocks),
                })
        for out in self.order_blocks_output:
            out.pop("_ob_ref", None)


def compute_smc_pine(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    times: list[str],
    params: dict[str, Any] | None = None,
    *,
    emit_timeline: bool = False,
) -> dict[str, Any]:
    actual_params = {**DEFAULT_PARAMS, **(params or {})}
    n = len(closes)
    if not (len(opens) == len(highs) == len(lows) == len(times) == n):
        raise ValueError(
            f"输入序列长度不一致: opens={len(opens)} highs={len(highs)} "
            f"lows={len(lows)} closes={n} times={len(times)}"
        )
    if n == 0:
        return {
            "events": [],
            "order_blocks": [],
            "equal_highs_lows": [],
            "trailing": {
                "top": None, "bottom": None, "bar_time": None, "bar_index": None,
                "last_top_time": None, "last_bottom_time": None,
            },
            "swing_bias": 0,
            "internal_bias": 0,
            "pivots": [],
            "time": [],
            "params": actual_params,
            "ob_lifecycle_events": [],
            "state_timeline": [] if emit_timeline else None,
        }
    state = _SMCPineState(opens, highs, lows, closes, times, actual_params, emit_timeline)
    state.run()
    result: dict[str, Any] = {
        "events": state.events,
        "order_blocks": state.order_blocks_output,
        "equal_highs_lows": state.equal_highs_lows,
        "trailing": {
            "top": state.trailing.top if state.trailing.top == state.trailing.top else None,
            "bottom": state.trailing.bottom if state.trailing.bottom == state.trailing.bottom else None,
            "bar_time": state.trailing.bar_time,
            "bar_index": state.trailing.bar_index,
            "last_top_time": state.trailing.last_top_time,
            "last_bottom_time": state.trailing.last_bottom_time,
        },
        "swing_bias": state.swing_trend.bias,
        "internal_bias": state.internal_trend.bias,
        "pivots": state.pivots,
        "time": list(times),
        "params": actual_params,
        "ob_lifecycle_events": state.ob_lifecycle_events,
    }
    if emit_timeline:
        result["state_timeline"] = state.state_timeline
    return result


def compute_smc_indicators(
    opens: list[float],
    highs: list[float],
    lows: list[float],
    closes: list[float],
    times: list[str],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return compute_smc_pine(opens, highs, lows, closes, times, params)


# =============================================================================
# SQZMOM_LB + momentum history: sqzmom_lb.py
# =============================================================================

class MomentumDirection(StrEnum):
    EXPANDING = "expanding"
    CONTRACTING = "contracting"
    FLAT = "flat"


class MomentumChange(StrEnum):
    ENHANCING = "enhancing"
    WEAKENING = "weakening"
    FLAT = "flat"


_COLOR_LIME = "lime"
_COLOR_GREEN = "green"
_COLOR_RED = "red"
_COLOR_MAROON = "maroon"
_COLOR_BLUE = "blue"
_COLOR_BLACK = "black"
_COLOR_GRAY = "gray"

_SQZMOM_DEFAULT_PARAMS: dict[str, Any] = {
    "length": 20,
    "mult": 2.0,
    "lengthKC": 20,
    "multKC": 1.5,
    "useTrueRange": True,
}


def _sma(values: np.ndarray, length: int) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.rolling(window=length, min_periods=length).mean().to_numpy()


def _stdev_biased(values: np.ndarray, length: int) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.rolling(window=length, min_periods=length).std(ddof=0).to_numpy()


def _highest(values: np.ndarray, length: int) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.rolling(window=length, min_periods=length).max().to_numpy()


def _lowest(values: np.ndarray, length: int) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.rolling(window=length, min_periods=length).min().to_numpy()


def _true_range(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> np.ndarray:
    n = len(highs)
    tr = np.empty(n, dtype=float)
    if n == 0:
        return tr
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        prev_close = closes[i - 1]
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - prev_close)
        lc = abs(lows[i] - prev_close)
        tr[i] = max(hl, hc, lc)
    return tr


def _linreg_pine(source: np.ndarray, length: int, offset: int = 0) -> np.ndarray:
    n = len(source)
    result = np.full(n, np.nan)
    if n < length or length < 2:
        return result
    x = np.arange(length, dtype=float)
    sum_x = float(x.sum())
    sum_x2 = float((x * x).sum())
    denom = length * sum_x2 - sum_x * sum_x
    if denom == 0:
        return result
    target_x = length - 1 - offset
    for i in range(length - 1, n):
        y = source[i - length + 1:i + 1]
        if np.isnan(y).any():
            continue
        sum_y = float(y.sum())
        sum_xy = float((x * y).sum())
        slope = (length * sum_xy - sum_x * sum_y) / denom
        intercept = (sum_y - slope * sum_x) / length
        result[i] = intercept + slope * target_x
    return result


def _to_float_or_none(arr: np.ndarray) -> list[float | None]:
    result: list[float | None] = []
    for v in arr:
        if v is None:
            result.append(None)
        elif isinstance(v, float) and (np.isnan(v) or not np.isfinite(v)):
            result.append(None)
        else:
            result.append(float(v))
    return result


def _to_bool_list(arr: np.ndarray) -> list[bool]:
    result: list[bool] = []
    for v in arr:
        if v is None:
            result.append(False)
        elif isinstance(v, float) and np.isnan(v):
            result.append(False)
        else:
            result.append(bool(v))
    return result


def compute_sqzmom_lb(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    p = {**_SQZMOM_DEFAULT_PARAMS, **(params or {})}
    length = int(p["length"])
    mult = float(p["mult"])
    length_kc = int(p["lengthKC"])
    mult_kc = float(p["multKC"])
    use_true_range = bool(p["useTrueRange"])
    n = len(closes)

    if n < max(length, length_kc):
        bcolor_list = [_COLOR_MAROON] * n
        scolor_list = [_COLOR_BLUE] * n
        return {
            "val": [None] * n,
            "sqzOn": [False] * n,
            "sqzOff": [False] * n,
            "noSqz": [True] * n,
            "bcolor": bcolor_list,
            "scolor": scolor_list,
            "params": {
                "length": length,
                "mult": mult,
                "lengthKC": length_kc,
                "multKC": mult_kc,
                "useTrueRange": use_true_range,
                "bb_dev_uses": "multKC",
            },
            "_debug_bb_kc": {
                "basis": [None] * n,
                "dev": [None] * n,
                "upperBB": [None] * n,
                "lowerBB": [None] * n,
                "ma": [None] * n,
                "rangema": [None] * n,
                "upperKC": [None] * n,
                "lowerKC": [None] * n,
            },
        }

    closes_f = np.asarray(closes, dtype=float)
    highs_f = np.asarray(highs, dtype=float)
    lows_f = np.asarray(lows, dtype=float)

    basis = _sma(closes_f, length)
    stdev_arr = _stdev_biased(closes_f, length)
    # Keep the canonical Pine behaviour: BB dev uses multKC, not mult.
    dev = mult_kc * stdev_arr
    upper_bb = basis + dev
    lower_bb = basis - dev

    ma = _sma(closes_f, length_kc)
    if use_true_range:
        range_val = _true_range(highs_f, lows_f, closes_f)
    else:
        range_val = highs_f - lows_f
    rangema = _sma(range_val, length_kc)
    upper_kc = ma + rangema * mult_kc
    lower_kc = ma - rangema * mult_kc

    sqz_on_raw = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off_raw = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    sqz_on_arr = np.where(np.isnan(sqz_on_raw), False, sqz_on_raw)
    sqz_off_arr = np.where(np.isnan(sqz_off_raw), False, sqz_off_raw)
    no_sqz_arr = (~sqz_on_arr.astype(bool)) & (~sqz_off_arr.astype(bool))

    highest_high = _highest(highs_f, length_kc)
    lowest_low = _lowest(lows_f, length_kc)
    avg_hl = (highest_high + lowest_low) / 2.0
    sma_close_kc = _sma(closes_f, length_kc)
    midline = (avg_hl + sma_close_kc) / 2.0
    delta = closes_f - midline
    val = _linreg_pine(delta, length_kc, offset=0)

    prev_val = np.empty(n, dtype=float)
    if n != 0:
        prev_val[0] = 0.0
        prev_val[1:] = val[:-1]
        prev_val = np.where(np.isnan(prev_val), 0.0, prev_val)

    bcolor_arr: list[str] = []
    for i in range(n):
        v = val[i]
        pv = prev_val[i]
        if np.isnan(v):
            bcolor_arr.append(_COLOR_MAROON)
        elif v > 0:
            if v > pv:
                bcolor_arr.append(_COLOR_LIME)
            else:
                bcolor_arr.append(_COLOR_GREEN)
        else:
            if v < pv:
                bcolor_arr.append(_COLOR_RED)
            else:
                bcolor_arr.append(_COLOR_MAROON)

    scolor_arr: list[str] = []
    for i in range(n):
        if no_sqz_arr[i]:
            scolor_arr.append(_COLOR_BLUE)
        elif sqz_on_arr[i]:
            scolor_arr.append(_COLOR_BLACK)
        else:
            scolor_arr.append(_COLOR_GRAY)

    return {
        "val": _to_float_or_none(val),
        "sqzOn": _to_bool_list(sqz_on_arr),
        "sqzOff": _to_bool_list(sqz_off_arr),
        "noSqz": _to_bool_list(no_sqz_arr),
        "bcolor": bcolor_arr,
        "scolor": scolor_arr,
        "params": {
            "length": length,
            "mult": mult,
            "lengthKC": length_kc,
            "multKC": mult_kc,
            "useTrueRange": use_true_range,
            "bb_dev_uses": "multKC",
        },
        "_debug_bb_kc": {
            "basis": _to_float_or_none(basis),
            "dev": _to_float_or_none(dev),
            "upperBB": _to_float_or_none(upper_bb),
            "lowerBB": _to_float_or_none(lower_bb),
            "ma": _to_float_or_none(ma),
            "rangema": _to_float_or_none(rangema),
            "upperKC": _to_float_or_none(upper_kc),
            "lowerKC": _to_float_or_none(lower_kc),
        },
    }


def build_momentum_history(
    sqzmom_result: dict[str, Any],
    volume_series: list[float] | np.ndarray | None = None,
    *,
    times: list[str] | None = None,
) -> dict[str, Any]:
    val_list = sqzmom_result.get("val", []) or []
    sqz_on_list = sqzmom_result.get("sqzOn", []) or []
    sqz_off_list = sqzmom_result.get("sqzOff", []) or []
    n = len(val_list)
    vol_arr = np.asarray(volume_series, dtype=float) if volume_series is not None else None

    daily_state: list[dict[str, Any]] = []
    sqz_release_events: list[dict[str, Any]] = []
    momentum_zero_cross_events: list[dict[str, Any]] = []

    for i in range(n):
        v = val_list[i]
        v_prev = val_list[i - 1] if i > 0 else None
        if sqz_on_list[i]:
            phase = "squeeze_on"
        elif sqz_off_list[i]:
            phase = "squeeze_off"
        else:
            phase = "no_squeeze"

        if v is None or (isinstance(v, float) and np.isnan(v)):
            direction = None
        elif v > 0:
            direction = MomentumDirection.EXPANDING.value
        elif v < 0:
            direction = MomentumDirection.CONTRACTING.value
        else:
            direction = MomentumDirection.FLAT.value

        if v is None or v_prev is None or (
            isinstance(v, float) and np.isnan(v)
        ) or (
            isinstance(v_prev, float) and np.isnan(v_prev)
        ):
            delta = None
            change = None
        else:
            delta = float(v) - float(v_prev)
            if delta > 0:
                change = MomentumChange.ENHANCING.value
            elif delta < 0:
                change = MomentumChange.WEAKENING.value
            else:
                change = MomentumChange.FLAT.value

        squeeze_period_volume_mean = None
        release_vol_ratio = None

        if sqz_on_list[i]:
            seg_start = i
            while seg_start > 0 and sqz_on_list[seg_start - 1]:
                seg_start -= 1
            squeeze_len = i - seg_start + 1
            if vol_arr is not None:
                seg = vol_arr[seg_start:i + 1]
                valid = seg[np.isfinite(seg)]
                if len(valid) > 0:
                    squeeze_period_volume_mean = float(np.mean(valid))
                    release_vol_ratio = None
        elif i > 0 and sqz_on_list[i - 1] and sqz_off_list[i]:
            seg_start = i - 1
            while seg_start > 0 and sqz_on_list[seg_start - 1]:
                seg_start -= 1
            squeeze_len = i - seg_start
            if vol_arr is not None:
                seg = vol_arr[seg_start:i]
                valid = seg[np.isfinite(seg)]
                if len(valid) > 0:
                    squeeze_mean = float(np.mean(valid))
                    squeeze_period_volume_mean = squeeze_mean
                    if np.isfinite(vol_arr[i]) and vol_arr[i] > 0:
                        release_vol_ratio = squeeze_mean / float(vol_arr[i])
        else:
            seg_start = None
            squeeze_len = None

        daily_state.append({
            "bar_index": i,
            "time": times[i] if times else None,
            "volatility_phase": phase,
            "momentum_direction": direction,
            "momentum_change": change,
            "sqzmom_delta": delta,
            "sqzmom_val": float(v) if v is not None and not (
                isinstance(v, float) and np.isnan(v)
            ) else None,
            "squeeze_period_volume_mean": squeeze_period_volume_mean,
            "release_volume_ratio": release_vol_ratio,
        })

        if i > 0 and sqz_on_list[i - 1] and sqz_off_list[i]:
            if v is None or (isinstance(v, float) and np.isnan(v)):
                release_dir = "null"
            elif v > 0:
                release_dir = "up"
            elif v < 0:
                release_dir = "down"
            else:
                release_dir = "null"
            sqz_release_events.append({
                "type": "SQZ_RELEASE",
                "bar_index": i,
                "time": times[i] if times else None,
                "direction": release_dir,
                "squeeze_start_index": seg_start,
                "squeeze_length": squeeze_len,
                "squeeze_period_volume_mean": squeeze_period_volume_mean,
                "release_volume_ratio": release_vol_ratio,
                "sqzmom_val": float(v) if v is not None and not (
                    isinstance(v, float) and np.isnan(v)
                ) else None,
            })

        if i > 0:
            prev_v = val_list[i - 1]
            cur_v = val_list[i]
            def _safe(x: Any) -> float | None:
                if x is None or (isinstance(x, float) and np.isnan(x)):
                    return None
                return float(x)
            pv = _safe(prev_v)
            cv = _safe(cur_v)
            if pv is not None and cv is not None:
                if pv <= 0 and cv > 0:
                    cross_type = "ZERO_CROSS_UP"
                elif pv >= 0 and cv < 0:
                    cross_type = "ZERO_CROSS_DOWN"
                else:
                    cross_type = None
                if cross_type:
                    momentum_zero_cross_events.append({
                        "type": cross_type,
                        "bar_index": i,
                        "time": times[i] if times else None,
                        "from_val": pv,
                        "to_val": cv,
                    })

    return {
        "daily_state": daily_state,
        "sqz_release_events": sqz_release_events,
        "momentum_zero_cross_events": momentum_zero_cross_events,
    }


# =============================================================================
# Thin standalone helpers (glue only; do not alter canonical calculations)
# =============================================================================

def compute_smc_from_frame(df: pd.DataFrame, params: dict[str, Any] | None = None, *, emit_timeline: bool = False) -> dict[str, Any]:
    """Call the canonical SMC core from an OHLC DataFrame."""
    frame = df.sort_index()
    times = [pd.Timestamp(x).isoformat() for x in frame.index]
    return compute_smc_pine(
        frame["open"].astype(float).tolist(),
        frame["high"].astype(float).tolist(),
        frame["low"].astype(float).tolist(),
        frame["close"].astype(float).tolist(),
        times,
        params,
        emit_timeline=emit_timeline,
    )


def compute_momentum_from_frame(df: pd.DataFrame, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call SQZMOM and its canonical history projection from an OHLCV DataFrame."""
    frame = df.sort_index()
    sqz = compute_sqzmom_lb(
        frame["open"].to_numpy(float),
        frame["high"].to_numpy(float),
        frame["low"].to_numpy(float),
        frame["close"].to_numpy(float),
        params,
    )
    times = [pd.Timestamp(x).isoformat() for x in frame.index]
    history = build_momentum_history(
        sqz,
        frame["volume"].to_numpy(float) if "volume" in frame.columns else None,
        times=times,
    )
    return {"sqzmom": sqz, "history": history}


__all__ = [
    "DSAConfig",
    "ATRRopeConfig",
    "dynamic_swing_anchored_vwap",
    "compute_atr_rope",
    "compute_dsa_history",
    "compute_dsa_bundle",
    "DEFAULT_PARAMS",
    "compute_smc_pine",
    "compute_smc_indicators",
    "compute_smc_from_frame",
    "compute_sqzmom_lb",
    "build_momentum_history",
    "compute_momentum_from_frame",
    "MomentumDirection",
    "MomentumChange",
]
