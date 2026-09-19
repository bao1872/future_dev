"""
forming_indicator_state_v1
==========================

Streaming, single-pass indicator kernel for the R3A Forming-MTF
environment.

This module is the PERFORMANCE-CRITICAL re-implementation of the
canonical indicator math in ``experiment_structural_reversion_pgm_v1``
(``compute_segment_features`` / ``build_sr_features`` /
``build_liquidity_features`` / ``confirmed_pivots`` / ``pine_rma`` /
``rolling_sma`` / ``true_range`` / ``trend_state_from_score``).

Design contract (R3A PERF1):
  * NO per-decision recomputation of historical indicator state.
  * Each HTF bar updates the committed state exactly ONCE (``commit``).
  * A forming (in-progress) HTF bar is evaluated by ``preview`` which
    copies the bounded committed state, steps the forming bar once,
    and discards the copy. Preview never mutates committed state.
  * The math is a *mechanical* streaming extraction of the canonical
    batch math: given the same ordered sequence of HTF bars, the last
    emitted feature row is bit-for-bit the same as
    ``compute_tf_features`` on that same sequence (validated by the
    T1 / long-history differential tests).

The canonical owner module is NOT modified. This module only imports
parameter definitions from it.

Complexity per decision, per timeframe:  O(bounded_state), independent
of total history length. There is no ``_tail`` truncation: the committed
state carries genuine RMA / trend / liquidity-recurrence state forward.
"""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    IndicatorParams,
    PINE_DEFAULT,
)

_NAN = float("nan")


# --------------------------------------------------------------------------- #
# DTP (Deviation Trend Profile)                                               #
# --------------------------------------------------------------------------- #
class DTPState:
    """Streaming Deviation-Trend-Profile state.

    Emits, per HTF bar: sma, atr (RMA len=atr_len), atr_liq (RMA
    len=liq_atr_len), dev, slope_raw, slope_atr, trend_score, trend_state.
    """

    def __init__(self, p: IndicatorParams):
        self.p = p
        self.sma_len = int(p.sma_len)
        self.atr_len = int(p.atr_len)
        self.liq_atr_len = int(p.liq_atr_len)
        self.trend_slope_lag = int(p.trend_slope_lag)
        self.trend_norm_lookback = int(p.trend_norm_lookback)
        self.trend_switch = float(p.trend_switch)
        self.atr_alpha = 1.0 / float(p.atr_len)
        self.liq_atr_alpha = 1.0 / float(p.liq_atr_len)
        self.reset()

    def reset(self) -> None:
        self.sma_deque: deque = deque(maxlen=self.sma_len)
        self.prev_close = _NAN
        # main ATR (Wilder RMA)
        self.atr_deque: deque = deque(maxlen=self.atr_len)
        self.atr_seeded = False
        self.atr_val = _NAN
        self.atr_finite = False
        # liquidity ATR (Wilder RMA, short)
        self.atr_liq_deque: deque = deque(maxlen=self.liq_atr_len)
        self.atr_liq_seeded = False
        self.atr_liq_val = _NAN
        self.atr_liq_finite = False
        # sma history for slope
        self.sma_hist: deque = deque(maxlen=self.trend_slope_lag + 1)
        # rolling max of slope_raw
        self.slope_deque: deque = deque(maxlen=self.trend_norm_lookback)
        self.slope_finite = 0
        # trend state machine
        self.trend_state_int = -1
        self.prev_score = _NAN
        self.prev_score_finite = False
        # last emitted values (so commit/preview share one path)
        self.last_atr = _NAN
        self.last_atr_liq = _NAN

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _rma_step(
        tr: float,
        buf: deque,
        seeded: bool,
        val: float,
        finite: bool,
        length: int,
        alpha: float,
    ) -> Tuple[bool, float, bool]:
        """One Wilder-RMA update. Mirrors ``pine_rma`` exactly.

        Returns (seeded, val, finite).
        """
        buf.append(tr)
        all_finite = len(buf) == length and np.all(np.isfinite(np.asarray(buf)))
        if not seeded:
            if all_finite:
                return True, float(np.mean(np.asarray(buf))), True
            return False, _NAN, False
        if not np.isfinite(tr):
            return True, _NAN, False
        if finite:
            return True, val + alpha * (tr - val), True
        # reseed from current window
        if all_finite:
            return True, float(np.mean(np.asarray(buf))), True
        return True, _NAN, False

    # -- step ------------------------------------------------------------- #
    def step(self, k: int, O, H, L, C) -> Dict[str, float]:
        # SMA (rolling mean of close over sma_len)
        self.sma_deque.append(float(C))
        sma = (
            float(np.mean(np.asarray(self.sma_deque)))
            if len(self.sma_deque) >= self.sma_len
            else _NAN
        )

        # True range
        if not np.isfinite(self.prev_close):
            tr = float(H) - float(L)
        else:
            tr = max(
                float(H) - float(L),
                abs(float(H) - self.prev_close),
                abs(float(L) - self.prev_close),
            )
        self.prev_close = float(C)

        # ATR (main) and ATR (liquidity)
        self.atr_seeded, self.atr_val, self.atr_finite = self._rma_step(
            tr, self.atr_deque, self.atr_seeded, self.atr_val,
            self.atr_finite, self.atr_len, self.atr_alpha,
        )
        self.atr_liq_seeded, self.atr_liq_val, self.atr_liq_finite = self._rma_step(
            tr, self.atr_liq_deque, self.atr_liq_seeded, self.atr_liq_val,
            self.atr_liq_finite, self.liq_atr_len, self.liq_atr_alpha,
        )

        atr = self.atr_val
        atr_liq = self.atr_liq_val

        # dev
        dev = (float(C) - sma) / atr if (np.isfinite(sma) and np.isfinite(atr)) else _NAN

        # slope_raw = sma - sma[trend_slope_lag]
        self.sma_hist.append(sma)
        if len(self.sma_hist) <= self.trend_slope_lag:
            slope_raw = _NAN
        else:
            prev_sma = self.sma_hist[0]
            slope_raw = (
                sma - prev_sma
                if (np.isfinite(prev_sma) and np.isfinite(sma))
                else _NAN
            )

        slope_atr = (
            slope_raw / atr
            if (np.isfinite(slope_raw) and np.isfinite(atr))
            else _NAN
        )

        # trend denominator = rolling max of slope_raw over trend_norm_lookback
        # (min_periods == window, so requires all window finite)
        if len(self.slope_deque) == self.trend_norm_lookback and np.isfinite(
            self.slope_deque[0]
        ):
            self.slope_finite -= 1
        self.slope_deque.append(slope_raw)
        if np.isfinite(slope_raw):
            self.slope_finite += 1
        if self.slope_finite == self.trend_norm_lookback:
            trend_denom = float(np.nanmax(np.asarray(self.slope_deque)))
        else:
            trend_denom = _NAN

        # trend score
        if np.isfinite(trend_denom) and trend_denom != 0.0 and np.isfinite(slope_raw):
            trend_score = slope_raw / trend_denom
        else:
            trend_score = _NAN

        # trend state machine (matches trend_state_from_score)
        if np.isfinite(trend_score) and self.prev_score_finite:
            sw = self.trend_switch
            cu = self.prev_score <= sw and trend_score > sw
            cd = self.prev_score >= -sw and trend_score < -sw
            if cu and self.trend_state_int == -1:
                self.trend_state_int = 1
            elif cd and self.trend_state_int == 1:
                self.trend_state_int = -1
        self.prev_score = trend_score
        self.prev_score_finite = np.isfinite(trend_score)

        self.last_atr = atr
        self.last_atr_liq = atr_liq
        return {
            "sma": sma,
            "atr": atr,
            "atr_liq": atr_liq,
            "dev": dev,
            "slope_atr": slope_atr,
            "trend_score": trend_score,
            "trend_state": float(self.trend_state_int),
        }

    # -- snapshot (for preview) ------------------------------------------ #
    def snapshot(self) -> "DTPState":
        s = DTPState(self.p)
        s.sma_deque = self.sma_deque.copy()
        s.prev_close = self.prev_close
        s.atr_deque = self.atr_deque.copy()
        s.atr_seeded = self.atr_seeded
        s.atr_val = self.atr_val
        s.atr_finite = self.atr_finite
        s.atr_liq_deque = self.atr_liq_deque.copy()
        s.atr_liq_seeded = self.atr_liq_seeded
        s.atr_liq_val = self.atr_liq_val
        s.atr_liq_finite = self.atr_liq_finite
        s.sma_hist = self.sma_hist.copy()
        s.slope_deque = self.slope_deque.copy()
        s.slope_finite = self.slope_finite
        s.trend_state_int = self.trend_state_int
        s.prev_score = self.prev_score
        s.prev_score_finite = self.prev_score_finite
        s.last_atr = self.last_atr
        s.last_atr_liq = self.last_atr_liq
        return s


# --------------------------------------------------------------------------- #
# SR (Support / Resistance)                                                   #
# --------------------------------------------------------------------------- #
class SRState:
    """Streaming Support/Resistance channel state.

    Mechanical per-bar extraction of ``build_sr_features``. Channels are
    rebuilt ONLY when a new pivot confirms; between pivots the channel set
    is unchanged, so per-bar cost is O(channels) plus O(1) buffer updates.
    """

    def __init__(self, p: IndicatorParams):
        self.p = p
        self.left = int(p.sr_pivot_left)
        self.right = int(p.sr_pivot_right)
        self.width = self.left + self.right + 1
        self.pct = float(p.sr_channel_width_pct)
        self.wlb = int(p.sr_width_lookback)
        self.loopback = int(p.sr_loopback)
        self.min_strength = int(p.sr_min_strength)
        self.max_channels = int(p.sr_max_channels)
        self.reset()

    def reset(self) -> None:
        self.k = -1
        self.prev_close = _NAN
        self.high_buf: deque = deque(maxlen=self.width)
        self.low_buf: deque = deque(maxlen=self.width)
        self.hh_deque: deque = deque(maxlen=self.wlb)
        self.ll_deque: deque = deque(maxlen=self.wlb)
        self.hist_high: deque = deque(maxlen=self.loopback + 1)
        self.hist_low: deque = deque(maxlen=self.loopback + 1)
        self.pivots: List[Tuple[int, float]] = []
        self.channels: List[Tuple[float, float, float]] = []

    def _pivot_value(self) -> Tuple[float, float]:
        """Return (ph, pl) confirmed at the current bar, using the sliding
        buffer. Center sits at buffer index ``self.left``."""
        ph = _NAN
        pl = _NAN
        if len(self.high_buf) == self.width:
            hb = np.asarray(self.high_buf, dtype=float)
            lb = np.asarray(self.low_buf, dtype=float)
            center_h = hb[self.left]
            wmax = np.nanmax(hb)
            if np.isfinite(center_h) and center_h >= wmax and center_h != 0.0:
                ph = float(center_h)
            center_l = lb[self.left]
            wmin = np.nanmin(lb)
            if np.isfinite(center_l) and center_l <= wmin and center_l != 0.0:
                pl = float(center_l)
        return ph, pl

    def _rebuild_channels(self, k: int, width_i: float) -> None:
        pivots = self.pivots
        hh_arr = np.asarray(self.hist_high, dtype=float)
        ll_arr = np.asarray(self.hist_low, dtype=float)
        candidates: List[Tuple[float, float, float]] = []
        for _, seed in pivots:
            loz = float(seed)
            hiz = float(seed)
            pivot_count = 0
            for _, cpp in pivots:
                wt = (hiz - cpp) if cpp <= hiz else (cpp - loz)
                if wt <= width_i:
                    loz = min(loz, cpp)
                    hiz = max(hiz, cpp)
                    pivot_count += 1
            # Canonical counts ONE touch per bar (high OR low inside the
            # channel), NOT high + low separately. A fully contained bar must
            # contribute exactly 1.
            touches = int(np.sum(((hh_arr <= hiz) & (hh_arr >= loz)) | ((ll_arr <= hiz) & (ll_arr >= loz))))
            strength = float(pivot_count * 20 + touches)
            candidates.append((hiz, loz, strength))

        selected: List[Tuple[float, float, float]] = []
        alive = [True] * len(candidates)
        for _ in range(min(10, len(candidates))):
            ids = [j for j, a in enumerate(alive) if a]
            if not ids:
                break
            strengths = np.array([candidates[j][2] for j in ids], dtype=float)
            kk = ids[int(np.argmax(strengths))]
            hiz, loz, strength = candidates[kk]
            if strength < self.min_strength * 20:
                break
            selected.append((hiz, loz, strength))
            for j, (ch, cl, _) in enumerate(candidates):
                if not alive[j]:
                    continue
                overlap = (loz <= ch <= hiz) or (loz <= cl <= hiz)
                if overlap:
                    alive[j] = False
        self.channels = selected[: self.max_channels]

    def step(self, k: int, H, L, C, atr) -> Dict[str, float]:
        self.k = k
        self.high_buf.append(float(H))
        self.low_buf.append(float(L))
        self.hh_deque.append(float(H))
        self.ll_deque.append(float(L))
        self.hist_high.append(float(H))
        self.hist_low.append(float(L))

        feats: Dict[str, float] = {
            "sr_support_dist_atr": _NAN,
            "sr_resistance_dist_atr": _NAN,
            "sr_support_price": _NAN,
            "sr_resistance_price": _NAN,
            "sr_support_strength": _NAN,
            "sr_resistance_strength": _NAN,
            "sr_in_zone": 0.0,
            "sr_zone_strength": 0.0,
            "sr_broken_up": 0.0,
            "sr_broken_down": 0.0,
            "sr_n_channels": 0.0,
        }

        ph, pl = self._pivot_value()
        ph_truthy = np.isfinite(ph) and ph != 0.0
        pl_truthy = np.isfinite(pl) and pl != 0.0
        if ph_truthy or pl_truthy:
            pivot_value = float(ph) if ph_truthy else float(pl)
            self.pivots.insert(0, (k, pivot_value))
            self.pivots = [(j, p_) for j, p_ in self.pivots if k - j <= self.loopback]
            width_i = _NAN
            if len(self.hh_deque) == self.wlb:
                hh = float(np.nanmax(np.asarray(self.hh_deque)))
                ll = float(np.nanmin(np.asarray(self.ll_deque)))
                if np.isfinite(hh) and np.isfinite(ll):
                    width_i = (hh - ll) * self.pct / 100.0
            if np.isfinite(width_i) and self.pivots:
                self._rebuild_channels(k, width_i)

        feats["sr_n_channels"] = float(len(self.channels))

        c = float(C)
        a = atr if (np.isfinite(atr) and atr > 0) else _NAN
        containing = [z for z in self.channels if z[1] <= c <= z[0]]
        if containing:
            feats["sr_in_zone"] = 1.0
            feats["sr_zone_strength"] = max(z[2] for z in containing)
        supports = [z for z in self.channels if z[0] < c]
        resistances = [z for z in self.channels if z[1] > c]
        if supports:
            z = max(supports, key=lambda q: q[0])
            feats["sr_support_price"] = z[0]
            feats["sr_support_strength"] = z[2]
            if np.isfinite(a):
                feats["sr_support_dist_atr"] = (c - z[0]) / a
        elif containing:
            z = max(containing, key=lambda q: q[2])
            feats["sr_support_price"] = c
            feats["sr_support_strength"] = z[2]
            feats["sr_support_dist_atr"] = 0.0
        if resistances:
            z = min(resistances, key=lambda q: q[1])
            feats["sr_resistance_price"] = z[1]
            feats["sr_resistance_strength"] = z[2]
            if np.isfinite(a):
                feats["sr_resistance_dist_atr"] = (z[1] - c) / a
        elif containing:
            z = max(containing, key=lambda q: q[2])
            feats["sr_resistance_price"] = c
            feats["sr_resistance_strength"] = z[2]
            feats["sr_resistance_dist_atr"] = 0.0
        if k > 0 and not containing and np.isfinite(self.prev_close):
            pc = self.prev_close
            for hi0, lo0, _ in self.channels:
                if pc <= hi0 and c > hi0:
                    feats["sr_broken_up"] = 1.0
                if pc >= lo0 and c < lo0:
                    feats["sr_broken_down"] = 1.0
        self.prev_close = c
        return feats

    def snapshot(self) -> "SRState":
        s = SRState(self.p)
        s.k = self.k
        s.prev_close = self.prev_close
        s.high_buf = self.high_buf.copy()
        s.low_buf = self.low_buf.copy()
        s.hh_deque = self.hh_deque.copy()
        s.ll_deque = self.ll_deque.copy()
        s.hist_high = self.hist_high.copy()
        s.hist_low = self.hist_low.copy()
        s.pivots = list(self.pivots)
        s.channels = list(self.channels)
        return s


# --------------------------------------------------------------------------- #
# Liquidity                                                                    #
# --------------------------------------------------------------------------- #
class LiquidityState:
    """Streaming Liquidity state. Mechanical per-bar extraction of
    ``build_liquidity_features``."""

    def __init__(self, p: IndicatorParams):
        self.p = p
        self.left = int(p.liq_left)
        self.right = int(p.liq_right)
        self.width = self.left + self.right + 1
        self.liq_mar = 10.0 / float(p.liq_margin_input)
        self.visible = int(p.liq_visible)
        self.postbreak = float(p.liq_postbreak_margin_atr)
        self.reset()

    def reset(self) -> None:
        self.k = -1
        self.prev_close = _NAN
        self.high_buf: deque = deque(maxlen=self.width)
        self.low_buf: deque = deque(maxlen=self.width)
        self.zz: List[Dict[str, Any]] = []
        self.levels_up: List[Dict[str, Any]] = []
        self.levels_down: List[Dict[str, Any]] = []
        self.last_breach: Optional[Dict[str, Any]] = None

    # -- zz / level helpers (ported verbatim from canonical) -------------- #
    def _update_zz(self, side: int, idx: int, price: float) -> None:
        if not self.zz or int(self.zz[0]["dir"]) != side:
            self.zz.insert(0, dict(dir=side, x=idx, y=float(price)))
        else:
            better = (
                price > self.zz[0]["y"] if side > 0 else price < self.zz[0]["y"]
            )
            if better:
                self.zz[0] = dict(dir=side, x=idx, y=float(price))
        self.zz = self.zz[:50]

    def _maybe_create_level(self, side: int, pivot: float, atr_i: float) -> None:
        if not np.isfinite(atr_i) or atr_i <= 0:
            return
        margin = atr_i / self.liq_mar
        count = 0
        start_bar = None
        level_price = _NAN
        cluster_max = 0.0
        cluster_min = 1e7
        for z in self.zz:
            if int(z["dir"]) != side:
                continue
            y = float(z["y"])
            if side > 0:
                if y > pivot + margin:
                    break
            else:
                if y < pivot - margin:
                    break
            inside = (pivot - margin < y < pivot + margin)
            if inside:
                count += 1
                start_bar = int(z["x"])
                level_price = y
                cluster_max = max(cluster_max, y)
                cluster_min = min(cluster_min, y)
        if count <= 2 or start_bar is None or not np.isfinite(level_price):
            return
        center = 0.5 * (cluster_max + cluster_min)
        obj = dict(
            left=start_bar,
            level=float(level_price),
            top=float(center + margin),
            bottom=float(center - margin),
            broken=False,
            breach_i=None,
        )
        target = self.levels_up if side > 0 else self.levels_down
        if target and int(target[0]["left"]) == start_bar:
            target[0]["top"] = obj["top"]
            target[0]["bottom"] = obj["bottom"]
        else:
            target.insert(0, obj)
            del target[self.visible:]

    def step(self, k: int, H, L, C, atr_liq) -> Dict[str, float]:
        self.k = k
        self.high_buf.append(float(H))
        self.low_buf.append(float(L))
        feats: Dict[str, float] = {
            "liq_up_dist_atr": _NAN,
            "liq_down_dist_atr": _NAN,
            "liq_up_level_price": _NAN,
            "liq_down_level_price": _NAN,
            "liq_breach_up": 0.0,
            "liq_breach_down": 0.0,
            "liq_last_breach_side": 0.0,
            "liq_last_breach_age": _NAN,
            "liq_last_accept": 0.0,
            "liq_last_reclaim": 0.0,
            "liq_last_zone_active": 0.0,
            "liq_up_count": 0.0,
            "liq_down_count": 0.0,
        }
        atr_i = atr_liq if np.isfinite(atr_liq) else _NAN

        ph, pl = self._pivot_value()
        ph_truthy = np.isfinite(ph) and ph != 0.0
        pl_truthy = np.isfinite(pl) and pl != 0.0
        if ph_truthy:
            self._update_zz(+1, k - self.right, float(ph))
            self._maybe_create_level(+1, float(ph), atr_i)
        if pl_truthy:
            self._update_zz(-1, k - self.right, float(pl))
            self._maybe_create_level(-1, float(pl), atr_i)

        # breach detection
        for lev in self.levels_up:
            if (not lev["broken"]) and float(H) > float(lev["top"]):
                lev["broken"] = True
                lev["breach_i"] = k
                feats["liq_breach_up"] = 1.0
                self.last_breach = dict(
                    side=+1, level=float(lev["level"]), i=k, zone_active=True
                )
        for lev in self.levels_down:
            if (not lev["broken"]) and float(L) < float(lev["bottom"]):
                lev["broken"] = True
                lev["breach_i"] = k
                feats["liq_breach_down"] = 1.0
                self.last_breach = dict(
                    side=-1, level=float(lev["level"]), i=k, zone_active=True
                )

        # active (un-broken, still beyond price) levels
        c = float(C)
        active_up = [z for z in self.levels_up if (not z["broken"]) and z["bottom"] > c]
        active_down = [z for z in self.levels_down if (not z["broken"]) and z["top"] < c]
        if active_up and np.isfinite(atr_i) and atr_i > 0:
            z = min(active_up, key=lambda q: q["bottom"])
            feats["liq_up_dist_atr"] = (float(z["bottom"]) - c) / atr_i
            feats["liq_up_level_price"] = float(z["level"])
        if active_down and np.isfinite(atr_i) and atr_i > 0:
            z = max(active_down, key=lambda q: q["top"])
            feats["liq_down_dist_atr"] = (c - float(z["top"])) / atr_i
            feats["liq_down_level_price"] = float(z["level"])
        # Canonical counts only UNBROKEN levels (sum(not z["broken"])), not
        # the raw list length.
        feats["liq_up_count"] = float(sum(1 for q in self.levels_up if not q["broken"]))
        feats["liq_down_count"] = float(sum(1 for q in self.levels_down if not q["broken"]))

        if self.last_breach is not None:
            side = int(self.last_breach["side"])
            level = float(self.last_breach["level"])
            age = k - int(self.last_breach["i"])
            feats["liq_last_breach_side"] = float(side)
            feats["liq_last_breach_age"] = float(age)
            if (
                bool(self.last_breach.get("zone_active", False))
                and np.isfinite(atr_i)
                and atr_i > 0
            ):
                inside_zone = (
                    float(L) > level - self.postbreak * atr_i
                    and float(H) < level + self.postbreak * atr_i
                )
                if not inside_zone:
                    self.last_breach["zone_active"] = False
            feats["liq_last_zone_active"] = float(
                bool(self.last_breach.get("zone_active", False))
            )
            if side > 0:
                accepted = c > level
                reclaimed = c <= level
            else:
                accepted = c < level
                reclaimed = c >= level
            feats["liq_last_accept"] = float(accepted)
            feats["liq_last_reclaim"] = float(reclaimed)
        self.prev_close = c
        return feats

    def _pivot_value(self) -> Tuple[float, float]:
        ph = _NAN
        pl = _NAN
        if len(self.high_buf) == self.width:
            hb = np.asarray(self.high_buf, dtype=float)
            lb = np.asarray(self.low_buf, dtype=float)
            center_h = hb[self.left]
            wmax = np.nanmax(hb)
            if np.isfinite(center_h) and center_h >= wmax and center_h != 0.0:
                ph = float(center_h)
            center_l = lb[self.left]
            wmin = np.nanmin(lb)
            if np.isfinite(center_l) and center_l <= wmin and center_l != 0.0:
                pl = float(center_l)
        return ph, pl

    def snapshot(self) -> "LiquidityState":
        s = LiquidityState(self.p)
        s.k = self.k
        s.prev_close = self.prev_close
        s.high_buf = self.high_buf.copy()
        s.low_buf = self.low_buf.copy()
        s.zz = copy.deepcopy(self.zz)
        s.levels_up = copy.deepcopy(self.levels_up)
        s.levels_down = copy.deepcopy(self.levels_down)
        s.last_breach = copy.deepcopy(self.last_breach)
        return s


# --------------------------------------------------------------------------- #
# Combined IndicatorState (per timeframe)                                      #
# --------------------------------------------------------------------------- #
class IndicatorState:
    """Combined DTP + SR + Liquidity streaming state for one timeframe.

    ``commit`` persists a COMPLETED HTF bar. ``preview`` evaluates a forming
    (in-progress) HTF bar without mutating committed state.
    """

    def __init__(self, params: IndicatorParams, include_sr: bool = True):
        self.p = params
        self.include_sr = include_sr
        self.dtp = DTPState(params)
        self.sr = SRState(params) if include_sr else None
        self.liq = LiquidityState(params)

    def reset(self) -> None:
        self.dtp.reset()
        if self.sr is not None:
            self.sr.reset()
        self.liq.reset()

    def step(self, k: int, O, H, L, C) -> Dict[str, float]:
        dtp = self.dtp.step(k, O, H, L, C)
        atr = self.dtp.last_atr
        atr_liq = self.dtp.last_atr_liq
        feats: Dict[str, float] = dict(dtp)
        if self.sr is not None:
            feats.update(self.sr.step(k, H, L, C, atr))
        feats.update(self.liq.step(k, H, L, C, atr_liq))
        return feats

    def preview(self, k: int, O, H, L, C) -> Dict[str, float]:
        c = self.snapshot()
        return c.step(k, O, H, L, C)

    def snapshot(self) -> "IndicatorState":
        n = IndicatorState.__new__(IndicatorState)
        n.p = self.p
        n.include_sr = self.include_sr
        n.dtp = self.dtp.snapshot()
        n.sr = self.sr.snapshot() if self.sr is not None else None
        n.liq = self.liq.snapshot()
        return n


# --------------------------------------------------------------------------- #
# Feature column contract                                                      #
# --------------------------------------------------------------------------- #
# Continuous (float) canonical feature columns, in emission order.
CONTINUOUS_COLS = [
    "dev",
    "slope_atr",
    "trend_score",
    "sma",
    "atr",
    "sr_support_dist_atr",
    "sr_resistance_dist_atr",
    "sr_support_price",
    "sr_resistance_price",
    "sr_support_strength",
    "sr_resistance_strength",
    "sr_zone_strength",
    "liq_up_dist_atr",
    "liq_down_dist_atr",
    "liq_up_level_price",
    "liq_down_level_price",
    "liq_last_breach_age",
]

# Discrete (int) canonical feature columns.
DISCRETE_COLS = [
    "trend_state",
    "sr_in_zone",
    "sr_broken_up",
    "sr_broken_down",
    "sr_n_channels",
    "liq_breach_up",
    "liq_breach_down",
    "liq_last_breach_side",
    "liq_last_accept",
    "liq_last_reclaim",
    "liq_last_zone_active",
    "liq_up_count",
    "liq_down_count",
]

FEATURE_COLS = CONTINUOUS_COLS + DISCRETE_COLS

DISCRETE_DTYPES = {
    "trend_state": "int8",
    "sr_in_zone": "int8",
    "sr_broken_up": "int8",
    "sr_broken_down": "int8",
    "sr_n_channels": "int16",
    "liq_breach_up": "int8",
    "liq_breach_down": "int8",
    "liq_last_breach_side": "int8",
    "liq_last_accept": "int8",
    "liq_last_reclaim": "int8",
    "liq_last_zone_active": "int8",
    "liq_up_count": "int16",
    "liq_down_count": "int16",
}
