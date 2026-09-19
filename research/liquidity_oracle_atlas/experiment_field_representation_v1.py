#!/usr/bin/env python3
"""
experiment_field_representation_v1
==================================

R3C-E1 COMPACT FIELD REPRESENTATION.

Question:

    Does a full low-dimensional FIELD representation of the same market
    state contain more OOS information than the current
    nearest-object SUMMARY representation?

        E_old    = [DTP_summary, SR_nearest, LQ_nearest]_4TF        (96)
        E_field  = [Trend_path, SR_field, LQ_field+event]_4TF
                   + VolRegime + Maturity                          (155)

The mathematics, field bins, weights, feature families, variants and the
frozen models are all specified by the task and implemented here verbatim.
This file does NOT modify:
    build_forming_environment_v1.py
    forming_indicator_state_v1.py
Canonical liquidity/SR/DTP behaviour is inherited unchanged; the liquidity
subclass only retains extra experimental METADATA (cluster_count).

One environment pass per object -> one join -> six variants evaluated from
the SAME joined dataset (environment is never recomputed per variant).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys
import time
from collections import deque

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from research.liquidity_oracle_atlas.build_forming_environment_v1 import (  # noqa: E402
    FormingEnvironmentBuilder,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (  # noqa: E402
    DTPState,
    IndicatorState,
    LiquidityState,
    SRState,
)

# --------------------------------------------------------------------------- #
# frozen contract                                                              #
# --------------------------------------------------------------------------- #
SYMBOLS = [
    "AG", "AU", "CU", "AL", "SN",
    "NI", "RB", "I", "SC", "RU",
    "MA", "TA", "M", "P", "CF",
]

R1_REL = os.path.join(
    "artifacts", "robust_trade_oracle_dp_v1", "robust_trade_oracle_rows.parquet"
)
R1_SHA = "ea30c67db0464da9fe4f80ede8dd597336ef6d69e08220ed6a4bb94210fcacfa"
R2_REL = os.path.join(
    "artifacts", "oracle_constraint_robustness_v1", "oracle_constraint_rows.parquet"
)
R2_SHA = "3980177b78d46cbc4ab7ffcd84dedebee3c5c62e396b11ec0da478b463e3f265"

KEY = ["symbol", "decision_bar_index", "decision_time"]
TF = ["m5", "m15", "h1", "h4"]
N_TF = len(TF)

OLD_T = ["dev", "slope_atr", "trend_score", "trend_state"]

OLD_SR = [
    "sr_support_dist_atr", "sr_resistance_dist_atr",
    "sr_support_strength", "sr_resistance_strength",
    "sr_in_zone", "sr_zone_strength",
    "sr_broken_up", "sr_broken_down", "sr_n_channels",
]

OLD_LIQ = [
    "liq_up_dist_atr", "liq_down_dist_atr",
    "liq_breach_up", "liq_breach_down",
    "liq_last_breach_side", "liq_last_breach_age",
    "liq_last_accept", "liq_last_reclaim", "liq_last_zone_active",
    "liq_up_count", "liq_down_count",
]

TREND_NEW = [
    "dev", "slope_atr", "trend_v10_atr", "trend_v20_atr",
    "trend_eff20", "trend_score", "trend_state",
]

SR_NEW = [
    "sr_mass_lt_m2", "sr_mass_m2_m1", "sr_mass_m1_m05", "sr_mass_m05_0",
    "sr_mass_0_p05", "sr_mass_p05_p1", "sr_mass_p1_p2", "sr_mass_gt_p2",
    "sr_at_price_mass", "sr_nearest_gap_atr",
    "sr_n_channels", "sr_broken_up", "sr_broken_down",
]

LIQ_NEW = [
    "liq_mass_lt_m2", "liq_mass_m2_m1", "liq_mass_m1_m05", "liq_mass_m05_0",
    "liq_mass_0_p05", "liq_mass_p05_p1", "liq_mass_p1_p2", "liq_mass_gt_p2",
    "liq_at_price_mass", "liq_at_price_side_mass", "liq_nearest_gap_atr",
    "liq_breach_up", "liq_breach_down", "liq_last_breach_side",
    "liq_last_signed_dist_atr", "liq_last_log_age", "liq_last_zone_active",
]

VOL_NEW = ["vol_regime_log_ratio"]
MATURITY_COLS = ["m15_maturity", "h1_maturity", "h4_maturity"]

TRADE_ACTIONS = ["Long", "Short", "Wait"]
DIRECTION_ACTIONS = ["Long", "Short"]

DISCRETE_SUFFIXES = {
    "trend_state",
    "sr_in_zone", "sr_broken_up", "sr_broken_down", "sr_n_channels",
    "liq_breach_up", "liq_breach_down", "liq_last_breach_side",
    "liq_last_accept", "liq_last_reclaim", "liq_last_zone_active",
    "liq_up_count", "liq_down_count",
}

FIELD_EDGES = np.array(
    [-np.inf, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.inf],
    dtype=np.float64,
)
FIELD_NAMES = [
    "lt_m2", "m2_m1", "m1_m05", "m05_0",
    "0_p05", "p05_p1", "p1_p2", "gt_p2",
]

# experimental metadata counter (must stay 0 on the real path)
MISSING_CLUSTER = {"count": 0}


def _stop(msg: str):
    raise SystemExit(f"STOP: {msg}")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def prefixed(block):
    return [f"{tf}_{c}" for tf in TF for c in block]


VARIANTS = {
    "V0_OLD96": prefixed(OLD_T + OLD_SR + OLD_LIQ),
    "V1_TREND_REPR": prefixed(TREND_NEW + OLD_SR + OLD_LIQ),
    "V2_SR_REPR": prefixed(OLD_T + SR_NEW + OLD_LIQ),
    "V3_LIQ_REPR": prefixed(OLD_T + OLD_SR + LIQ_NEW),
    "V4_FIELD_CORE": prefixed(TREND_NEW + SR_NEW + LIQ_NEW),
    "V5_FIELD_FULL": (
        prefixed(TREND_NEW + SR_NEW + LIQ_NEW + VOL_NEW) + MATURITY_COLS
    ),
}

assert len(VARIANTS["V0_OLD96"]) == 96
assert len(VARIANTS["V1_TREND_REPR"]) == 108
assert len(VARIANTS["V2_SR_REPR"]) == 112
assert len(VARIANTS["V3_LIQ_REPR"]) == 120
assert len(VARIANTS["V4_FIELD_CORE"]) == 148
assert len(VARIANTS["V5_FIELD_FULL"]) == 155

# --- E1.1 attribution variants (auxiliary-state decomposition) ------------ #
FIELD_CORE_COLS = prefixed(TREND_NEW + SR_NEW + LIQ_NEW)
VOL_COLS = prefixed(VOL_NEW)
MAT_COLS = list(MATURITY_COLS)

VARIANTS["V6_OLD96_AUX"] = VARIANTS["V0_OLD96"] + VOL_COLS + MAT_COLS
VARIANTS["V7_FIELD_VOL"] = FIELD_CORE_COLS + VOL_COLS
VARIANTS["V8_FIELD_MAT"] = FIELD_CORE_COLS + MAT_COLS

assert len(VARIANTS["V6_OLD96_AUX"]) == 103
assert len(VARIANTS["V7_FIELD_VOL"]) == 152
assert len(VARIANTS["V8_FIELD_MAT"]) == 151


# --------------------------------------------------------------------------- #
# §15 field integration (frozen)                                               #
# --------------------------------------------------------------------------- #
def interval_field_mass(
    lo: np.ndarray,
    hi: np.ndarray,
    weight: np.ndarray,
    edges: np.ndarray = FIELD_EDGES,
) -> np.ndarray:
    """Vectorized interval-field integration.

    rho(x) = sum_j w_j * 1(lo_j <= x <= hi_j)
    Returns the integral of rho over every fixed bin. No division by zone
    width, so position / width / strength all enter the field.

    A zero-width interval is treated as a Dirac-like point mass whose full
    weight goes to the bin containing its centre.
    """
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float64)

    out = np.zeros(len(edges) - 1, dtype=np.float64)

    valid = (
        np.isfinite(lo) & np.isfinite(hi) & np.isfinite(weight) & (weight > 0)
    )
    if not np.any(valid):
        return out

    lo = lo[valid]
    hi = hi[valid]
    weight = weight[valid]

    swap = lo > hi
    if np.any(swap):
        tmp = lo[swap].copy()
        lo[swap] = hi[swap]
        hi[swap] = tmp

    width = hi - lo
    regular = width > 1e-12

    if np.any(regular):
        l = lo[regular, None]
        h = hi[regular, None]
        w = weight[regular, None]
        left = edges[:-1][None, :]
        right = edges[1:][None, :]
        overlap = np.maximum(
            0.0, np.minimum(h, right) - np.maximum(l, left)
        )
        out += np.sum(w * overlap, axis=0)

    if np.any(~regular):
        centers = 0.5 * (lo[~regular] + hi[~regular])
        bins = np.searchsorted(edges, centers, side="right") - 1
        bins = np.clip(bins, 0, len(out) - 1)
        np.add.at(out, bins, weight[~regular])

    return out


def nearest_signed_gap(lo: np.ndarray, hi: np.ndarray) -> float:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)

    valid = np.isfinite(lo) & np.isfinite(hi)
    if not np.any(valid):
        return np.nan

    lo = lo[valid]
    hi = hi[valid]

    swap = lo > hi
    if np.any(swap):
        tmp = lo[swap].copy()
        lo[swap] = hi[swap]
        hi[swap] = tmp

    gaps = np.where(
        hi < 0.0,
        hi,
        np.where(lo > 0.0, lo, 0.0),
    )
    return float(gaps[np.argmin(np.abs(gaps))])


# --------------------------------------------------------------------------- #
# §17 liquidity metadata subclass (canonical behaviour unchanged)              #
# --------------------------------------------------------------------------- #
class FieldLiquidityState(LiquidityState):
    def _cluster_meta(self, side: int, pivot: float, atr_i: float):
        if not np.isfinite(atr_i) or atr_i <= 0:
            return 0, None

        margin = atr_i / self.liq_mar
        count = 0
        start_bar = None

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

            if pivot - margin < y < pivot + margin:
                count += 1
                start_bar = int(z["x"])

        return count, start_bar

    def _maybe_create_level(self, side, pivot, atr_i):
        count, start_bar = self._cluster_meta(side, pivot, atr_i)

        # canonical behaviour remains owned by the base class
        super()._maybe_create_level(side, pivot, atr_i)

        if count <= 2 or start_bar is None:
            return

        target = self.levels_up if side > 0 else self.levels_down

        for lev in target:
            if int(lev["left"]) == int(start_bar):
                lev["cluster_count"] = int(count)
                break

    def snapshot(self):
        s = FieldLiquidityState(self.p)

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
# §18-22 field indicator state                                                 #
# --------------------------------------------------------------------------- #
class FieldIndicatorState(IndicatorState):
    def __init__(self, params):
        super().__init__(params, include_sr=True)

        # replace metadata-dropping liquidity state
        self.liq = FieldLiquidityState(params)

        # enough for SMA[t-20]
        self.sma_path = deque(maxlen=21)

    def reset(self):
        self.dtp.reset()
        self.sr.reset()
        self.liq.reset()
        self.sma_path.clear()

    def snapshot(self):
        n = FieldIndicatorState.__new__(FieldIndicatorState)

        n.p = self.p
        n.include_sr = True

        n.dtp = self.dtp.snapshot()
        n.sr = self.sr.snapshot()
        n.liq = self.liq.snapshot()
        n.sma_path = self.sma_path.copy()

        return n

    def preview(self, k, O, H, L, C):
        s = self.snapshot()
        return s.step(k, O, H, L, C, emit_field=True)

    def step(self, k, O, H, L, C, emit_field=False):
        base = super().step(k, O, H, L, C)

        sma = float(base["sma"]) if np.isfinite(base["sma"]) else np.nan
        self.sma_path.append(sma)

        if not emit_field:
            return base

        out = dict(base)
        out.update(self._trend_field(base))
        out.update(self._sr_field(C))
        out.update(self._liq_field(C))
        out.update(self._vol_regime())
        return out

    # -- §19 ---------------------------------------------------------------- #
    def _trend_field(self, base):
        a = float(self.dtp.last_atr)
        hist = np.asarray(self.sma_path, dtype=np.float64)

        def velocity(lag):
            if (
                len(hist) < lag + 1
                or not np.isfinite(a)
                or a <= 0
                or not np.isfinite(hist[-1])
                or not np.isfinite(hist[-lag - 1])
            ):
                return np.nan
            return float((hist[-1] - hist[-lag - 1]) / a)

        v10 = velocity(10)
        v20 = velocity(20)

        eff = np.nan
        if len(hist) >= 21:
            z = hist[-21:]
            if np.all(np.isfinite(z)):
                path = np.diff(z)
                denom = float(np.sum(np.abs(path)))
                if denom == 0.0:
                    eff = 0.0
                else:
                    eff = float((z[-1] - z[0]) / denom)

        return {
            "trend_v10_atr": v10,
            "trend_v20_atr": v20,
            "trend_eff20": eff,
        }

    # -- §20 ---------------------------------------------------------------- #
    def _sr_field(self, close):
        out = {f"sr_mass_{name}": 0.0 for name in FIELD_NAMES}
        out["sr_at_price_mass"] = 0.0
        out["sr_nearest_gap_atr"] = np.nan

        atr = float(self.dtp.last_atr)

        if (
            self.sr is None
            or not self.sr.channels
            or not np.isfinite(atr)
            or atr <= 0
        ):
            return out

        channels = np.asarray(self.sr.channels, dtype=np.float64)
        hi = channels[:, 0]
        lo = channels[:, 1]
        strength = channels[:, 2]

        lo_n = (lo - float(close)) / atr
        hi_n = (hi - float(close)) / atr

        weight = np.log1p(np.maximum(strength, 0.0))

        mass = interval_field_mass(lo_n, hi_n, weight)

        for name, value in zip(FIELD_NAMES, mass):
            out[f"sr_mass_{name}"] = float(value)

        contains = (lo_n <= 0.0) & (hi_n >= 0.0)
        out["sr_at_price_mass"] = float(np.sum(weight[contains]))
        out["sr_nearest_gap_atr"] = nearest_signed_gap(lo_n, hi_n)

        return out

    # -- §21 ---------------------------------------------------------------- #
    def _liq_field(self, close):
        out = {f"liq_mass_{name}": 0.0 for name in FIELD_NAMES}
        out.update({
            "liq_at_price_mass": 0.0,
            "liq_at_price_side_mass": 0.0,
            "liq_nearest_gap_atr": np.nan,
            "liq_last_signed_dist_atr": np.nan,
            "liq_last_log_age": np.nan,
        })

        atr = float(self.dtp.last_atr_liq)

        if not np.isfinite(atr) or atr <= 0:
            return out

        rows = []
        for side, levels in ((+1, self.liq.levels_up), (-1, self.liq.levels_down)):
            for z in levels:
                if bool(z.get("broken", False)):
                    continue

                if "cluster_count" not in z:
                    MISSING_CLUSTER["count"] += 1

                count = int(z.get("cluster_count", 3))

                rows.append((
                    float(z["bottom"]),
                    float(z["top"]),
                    float(np.log1p(max(count, 1))),
                    float(side),
                ))

        if rows:
            q = np.asarray(rows, dtype=np.float64)
            lo_n = (q[:, 0] - float(close)) / atr
            hi_n = (q[:, 1] - float(close)) / atr
            weight = q[:, 2]
            side = q[:, 3]

            mass = interval_field_mass(lo_n, hi_n, weight)
            for name, value in zip(FIELD_NAMES, mass):
                out[f"liq_mass_{name}"] = float(value)

            contains = (lo_n <= 0.0) & (hi_n >= 0.0)
            out["liq_at_price_mass"] = float(np.sum(weight[contains]))
            out["liq_at_price_side_mass"] = float(
                np.sum(weight[contains] * side[contains])
            )
            out["liq_nearest_gap_atr"] = nearest_signed_gap(lo_n, hi_n)

        last = self.liq.last_breach
        if last is not None:
            side = float(last["side"])
            level = float(last["level"])

            out["liq_last_signed_dist_atr"] = float(
                side * (float(close) - level) / atr
            )

            age = max(0, int(self.liq.k) - int(last["i"]))
            out["liq_last_log_age"] = float(np.log1p(age))

        return out

    # -- §22 ---------------------------------------------------------------- #
    def _vol_regime(self):
        a_long = float(self.dtp.last_atr)
        a_short = float(self.dtp.last_atr_liq)

        if (
            np.isfinite(a_long) and a_long > 0
            and np.isfinite(a_short) and a_short > 0
        ):
            v = float(np.log(a_short / a_long))
        else:
            v = np.nan

        return {"vol_regime_log_ratio": v}


# --------------------------------------------------------------------------- #
# §23 field environment extractor                                              #
# --------------------------------------------------------------------------- #
EXTRACT_COLS = sorted(set(
    OLD_T + OLD_SR + OLD_LIQ + TREND_NEW + SR_NEW + LIQ_NEW + VOL_NEW
))


def build_field_environment(symbol, max_bars=None):
    b = FormingEnvironmentBuilder(symbol, max_bars=max_bars)
    b.load_raw()
    b.prepare()

    base = b.base
    n = b.n
    seg_arr = base["segment"].to_numpy(np.int64)

    out = {
        "symbol": np.repeat(symbol, n),
        "decision_bar_index": np.arange(n, dtype=np.int64),
        "decision_time": (
            pd.DatetimeIndex(base["time"]).to_numpy()
            + np.timedelta64(5, "m")
        ),
        "trading_day": base["trading_day"].to_numpy(),
    }

    for tf, minutes in b.tf_minutes.items():
        state = FieldIndicatorState(b.params)
        form = b._form[tf]
        seg_completed = b._seg_completed[tf]

        cur_seg = None
        ci = 0
        seg_list = []

        # NOTE: float64 (not float32) so the OLD96 parity gate (<=1e-9) is
        # satisfiable; float32 storage precision would be ~1e-5 on values ~1e2.
        tf_arr = {
            c: np.full(n, np.nan, dtype=np.float64) for c in EXTRACT_COLS
        }

        for i in range(n):
            seg = int(seg_arr[i])

            if seg != cur_seg:
                state.reset()
                cur_seg = seg
                ci = 0
                seg_list = seg_completed.get(seg, [])

            while ci < len(seg_list) and seg_list[ci][0] < i:
                bar = seg_list[ci][1]
                state.step(
                    ci,
                    bar["open"], bar["high"], bar["low"], bar["close"],
                    emit_field=False,
                )
                ci += 1

            feats = state.preview(
                ci,
                form["open"][i], form["high"][i],
                form["low"][i], form["close"][i],
            )

            for c in EXTRACT_COLS:
                if c in feats:
                    tf_arr[c][i] = feats[c]

        for c, arr in tf_arr.items():
            out[f"{tf}_{c}"] = arr

        if tf != "m5":
            expected = minutes // 5
            out[f"{tf}_maturity"] = np.minimum(
                form["n_base"].astype(np.float64) / float(expected), 1.0
            )

    return pd.DataFrame(out)


# --------------------------------------------------------------------------- #
# §25 synthetic field math test                                                #
# --------------------------------------------------------------------------- #
def synthetic_field_test():
    lo = np.array([-1.0, 0.25])
    hi = np.array([-0.5, 0.75])
    w = np.array([2.0, 3.0])
    got = interval_field_mass(lo, hi, w)

    # [-1,-0.5] -> bin "m1_m05" (index 2): 2 * 0.5 = 1
    # [0.25,0.75] -> bins "0_p05" (0.25) and "p05_p1" (0.25): 3 * 0.25 each
    expected = np.array([0.0, 0.0, 1.0, 0.0, 0.75, 0.75, 0.0, 0.0])
    worst = float(np.max(np.abs(got - expected)))
    if worst > 1e-12:
        _stop(f"SYNTHETIC_FIELD_FAIL got={got.tolist()} expected={expected.tolist()}")
    print(f"[synthetic] 2-interval case OK (worst {worst:.2e})", flush=True)

    got2 = interval_field_mass(
        np.array([1.5]), np.array([1.5]), np.array([4.0])
    )
    expected2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0, 0.0])
    worst2 = float(np.max(np.abs(got2 - expected2)))
    if worst2 > 1e-12:
        _stop(f"SYNTHETIC_POINT_FAIL got={got2.tolist()}")
    print(f"[synthetic] zero-width point case OK (worst {worst2:.2e})", flush=True)
    return dict(two_interval_worst=worst, point_mass_worst=worst2)


# --------------------------------------------------------------------------- #
# §24 OLD96 parity gate                                                        #
# --------------------------------------------------------------------------- #
def old96_parity(symbol="AG", max_bars=2000):
    old_df, _ = (
        FormingEnvironmentBuilder(symbol, max_bars=max_bars)
        .load_raw()
        .prepare()
        .run(profile_memory=False)
    )
    new_df = build_field_environment(symbol, max_bars=max_bars)

    if len(old_df) != len(new_df):
        _stop(f"PARITY_ROW_COUNT {len(old_df)} != {len(new_df)}")

    cont_cells = 0
    max_err = 0.0
    disc_cells = 0
    disc_mis = 0
    first_bad = None

    for c in VARIANTS["V0_OLD96"]:
        if c not in old_df.columns or c not in new_df.columns:
            _stop(f"PARITY_MISSING_COLUMN {c}")
        bare = c.split("_", 1)[1]

        if bare in DISCRETE_SUFFIXES:
            a = old_df[c].to_numpy()
            bb = new_df[c].to_numpy()
            disc_cells += int(len(a))
            m = int(np.sum(a != bb))
            if m:
                disc_mis += m
                if first_bad is None:
                    first_bad = f"{c} {m} rows differ"
            continue

        a = old_df[c].to_numpy(float)
        bb = new_df[c].to_numpy(float)
        cont_cells += int(len(a))
        both = np.isnan(a) & np.isnan(bb)
        if not np.array_equal(np.isnan(a), np.isnan(bb)):
            if first_bad is None:
                first_bad = f"{c} NaN pattern differs"
            continue
        m = ~both
        if m.any():
            e = float(np.max(np.abs(a[m] - bb[m])))
            if e > max_err:
                max_err = e
            if e > 1e-9 and first_bad is None:
                first_bad = f"{c} max_err {e:.3e}"

    if disc_mis != 0:
        _stop(f"PARITY_DISCRETE_MISMATCH {disc_mis}: {first_bad}")
    if max_err > 1e-9:
        _stop(f"PARITY_CONTINUOUS_MAX_ERR {max_err:.3e}: {first_bad}")

    return dict(
        symbol=symbol,
        max_bars=max_bars,
        rows=int(len(old_df)),
        continuous_cells=cont_cells,
        max_abs_error=max_err,
        discrete_cells=disc_cells,
        discrete_mismatch=disc_mis,
    )


# --------------------------------------------------------------------------- #
# §26 feature invariants                                                       #
# --------------------------------------------------------------------------- #
def feature_invariants(joined):
    neg_sr = 0
    neg_lq = 0
    eff_viol = 0
    mat_viol = 0
    side_viol = 0

    for tf in TF:
        for name in FIELD_NAMES:
            c = f"{tf}_sr_mass_{name}"
            if c in joined.columns:
                v = joined[c].to_numpy(float)
                neg_sr += int(np.sum(np.isfinite(v) & (v < -1e-12)))
            c = f"{tf}_liq_mass_{name}"
            if c in joined.columns:
                v = joined[c].to_numpy(float)
                neg_lq += int(np.sum(np.isfinite(v) & (v < -1e-12)))

        c = f"{tf}_trend_eff20"
        if c in joined.columns:
            v = joined[c].to_numpy(float)
            eff_viol += int(np.sum(np.isfinite(v) & ((v < -1 - 1e-9) | (v > 1 + 1e-9))))

        c = f"{tf}_liq_at_price_mass"
        cs = f"{tf}_liq_at_price_side_mass"
        if c in joined.columns and cs in joined.columns:
            m = joined[c].to_numpy(float)
            s = joined[cs].to_numpy(float)
            ok = np.isfinite(m) & np.isfinite(s)
            side_viol += int(np.sum(ok & (m < np.abs(s) - 1e-12)))

    for c in MATURITY_COLS:
        if c in joined.columns:
            v = joined[c].to_numpy(float)
            mat_viol += int(np.sum(np.isfinite(v) & ((v <= 0) | (v > 1 + 1e-9))))

    return dict(
        negative_sr_mass_count=neg_sr,
        negative_liq_mass_count=neg_lq,
        trend_eff20_violations=eff_viol,
        maturity_violations=mat_viol,
        at_price_side_violations=side_viol,
        liq_missing_cluster_count=int(MISSING_CLUSTER["count"]),
    )


# --------------------------------------------------------------------------- #
# metrics / preprocessing (frozen from R3B-E0)                                 #
# --------------------------------------------------------------------------- #
def clf_metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1 - 1e-12)
    out = dict(n=int(len(y)), class_share=float(np.mean(y)))
    out["log_loss"] = float(log_loss(y, p, labels=[0, 1]))
    try:
        out["roc_auc"] = float(roc_auc_score(y, p))
    except Exception:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y, p))
    except Exception:
        out["pr_auc"] = float("nan")
    out["balanced_accuracy"] = float(balanced_accuracy_score(y, (p >= 0.5).astype(int)))
    out["brier"] = float(brier_score_loss(y, p))
    return out


def reg_metrics(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    out = dict(n=int(len(y)))
    try:
        out["r2"] = float(r2_score(y, p))
    except Exception:
        out["r2"] = float("nan")
    out["mae"] = float(mean_absolute_error(y, p))
    try:
        out["spearman"] = float(spearmanr(y, p).statistic)
    except Exception:
        out["spearman"] = float("nan")
    return out


def split_cont_disc(cols):
    cont, disc = [], []
    for c in cols:
        bare = c.split("_", 1)[1]
        (disc if bare in DISCRETE_SUFFIXES else cont).append(c)
    return cont, disc


def make_preprocessor(cols):
    cont, disc = split_cont_disc(cols)
    transformers = []
    if cont:
        transformers.append(("c", Pipeline([
            ("imp", SimpleImputer(strategy="median", add_indicator=True)),
            ("sc", StandardScaler()),
        ]), cont))
    if disc:
        transformers.append(("d", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), disc))
    return ColumnTransformer(transformers)


def decile_table(df, pred_col, actual_col, extra_rate_cols, label):
    d = df[[pred_col, actual_col] + extra_rate_cols].dropna(subset=[pred_col]).copy()
    if len(d) < 10:
        return []
    try:
        d["_bin"] = pd.qcut(d[pred_col], 10, labels=False, duplicates="drop")
    except Exception:
        return []
    rows = []
    for bval, g in d.groupby("_bin"):
        rec = dict(
            variant=label, decile=int(bval), n=int(len(g)),
            predicted_mean=float(g[pred_col].mean()),
            actual_mean=float(g[actual_col].mean()),
        )
        for c in extra_rate_cols:
            rec[c + "_rate"] = float(np.mean(g[c])) if len(g) else float("nan")
        rows.append(rec)
    return rows


# --------------------------------------------------------------------------- #
# oracle                                                                       #
# --------------------------------------------------------------------------- #
def load_oracle():
    r1_path = os.path.join(_REPO_ROOT, R1_REL)
    r2_path = os.path.join(_REPO_ROOT, R2_REL)
    for p, expect, tag in ((r1_path, R1_SHA, "R1.1"), (r2_path, R2_SHA, "R2")):
        if not os.path.exists(p):
            _stop(f"{tag} oracle parquet missing: {p}")
        got = sha256_file(p)
        if got != expect:
            _stop(f"STOP_ORACLE_ARTIFACT_IDENTITY_MISMATCH {tag}: {got} != {expect}")
        print(f"[oracle] {tag} sha256 OK", flush=True)

    r1 = pd.read_parquet(r1_path, columns=[
        "symbol", "decision_bar_index", "decision_time", "stable_action",
        "QL_24_ATR", "QS_24_ATR", "QW_24_ATR",
        "label_available_time_6", "label_available_time_12",
        "label_available_time_24",
    ])
    r2 = pd.read_parquet(r2_path, columns=[
        "symbol", "decision_bar_index", "decision_time",
        "baseline_stable_action", "joint_retention_stable",
    ])

    d1 = int(r1.duplicated(subset=KEY).sum())
    d2 = int(r2.duplicated(subset=KEY).sum())
    if d1 or d2:
        _stop(f"STOP_ORACLE_DUPLICATE_KEYS R1={d1} R2={d2}")
    if len(r1) != len(r2):
        _stop(f"STOP_R1_R2_KEY_MISMATCH {len(r1)} != {len(r2)}")

    # Row-count equality alone does NOT imply semantic-key equality.
    k1 = set(map(tuple, r1[KEY].to_numpy(dtype=object).tolist()))
    k2 = set(map(tuple, r2[KEY].to_numpy(dtype=object).tolist()))
    if k1 != k2:
        _stop(
            "STOP_R1_R2_KEY_MISMATCH "
            f"r1_only={len(k1 - k2)} r2_only={len(k2 - k1)}"
        )
    print(
        f"[oracle] R1/R2 rows={len(r1)} dup=0/0 key_sets_identical=True",
        flush=True,
    )
    return r1, r2


# --------------------------------------------------------------------------- #
# main                                                                         #
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="AG,AU")
    ap.add_argument("--out-dir", default=os.path.join(
        _REPO_ROOT, "artifacts", "environment_field_representation_v1"))
    ap.add_argument("--results-name",
                    default="environment_field_representation_small_results_v1.json")
    ap.add_argument("--deciles-name",
                    default="environment_field_representation_small_deciles_v1.csv")
    ap.add_argument("--parity-bars", type=int, default=2000)
    args = ap.parse_args(argv)

    os.makedirs(args.out_dir, exist_ok=True)
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    t_all = time.perf_counter()

    # ---- §25 synthetic field math ----
    synth = synthetic_field_test()

    # ---- §24 OLD96 parity ----
    print("[parity] running OLD96 parity gate...", flush=True)
    parity = old96_parity("AG", max_bars=args.parity_bars)
    print(
        f"[parity] OK continuous_cells={parity['continuous_cells']} "
        f"max_err={parity['max_abs_error']:.2e} disc_mismatch={parity['discrete_mismatch']}",
        flush=True,
    )

    # ---- oracle ----
    r1, r2 = load_oracle()

    # ---- one environment pass per object + join ----
    parts = []
    env_rows_total = 0
    t_field = 0.0
    t_old = 0.0
    for i, sym in enumerate(symbols, start=1):
        t0 = time.perf_counter()
        env = build_field_environment(sym, max_bars=None)
        t_field += time.perf_counter() - t0

        t1 = time.perf_counter()
        _b = (FormingEnvironmentBuilder(sym, max_bars=None)
              .load_raw().prepare())
        _b.run(profile_memory=False)
        t_old += time.perf_counter() - t1

        env_rows_total += len(env)

        o1 = r1[r1["symbol"] == sym]
        o2 = r2[r2["symbol"] == sym]

        env_bi = env["decision_bar_index"].astype("int64").to_numpy()
        env_ti = env["decision_time"].astype("int64").to_numpy()
        env_keys = set(zip(env_bi.tolist(), env_ti.tolist()))

        for tag, o in (("R1", o1), ("R2", o2)):
            orc = set(zip(
                o["decision_bar_index"].to_numpy().tolist(),
                o["decision_time"].astype("int64").to_numpy().tolist(),
            ))
            only = orc - env_keys
            if only:
                _stop(f"STOP_ORACLE_KEYS_NOT_SUBSET {sym}/{tag}: {len(only)}")

        # merge the field environment directly with the oracle columns
        # (both frames carry `trading_day`, so an intermediate frame holding it
        # would collide and be suffixed away)
        full = env.merge(o1, on=KEY, how="inner").merge(
            o2[KEY + ["baseline_stable_action", "joint_retention_stable"]],
            on=KEY, how="inner",
        )
        parts.append(full)
        print(
            f"[{i}/{len(symbols)}] {sym} env_rows={len(env)} joined={len(full)} "
            f"field_sec={time.perf_counter() - t0:.2f}",
            flush=True,
        )

    joined = pd.concat(parts, ignore_index=True)

    oracle_rows = int(len(r1[r1["symbol"].isin(symbols)]))
    ratio = (t_field / t_old) if t_old > 0 else float("inf")
    print(
        f"[perf] old_env_sec={t_old:.2f} field_env_sec={t_field:.2f} "
        f"ratio={ratio:.3f}",
        flush=True,
    )
    if ratio >= 2.0:
        _stop(f"STOP_PERFORMANCE_REGRESSION ratio={ratio:.3f} >= 2.0")

    # ---- targets (frozen) ----
    joined["QL"] = joined["QL_24_ATR"].astype(float)
    joined["QS"] = joined["QS_24_ATR"].astype(float)
    joined["QW"] = joined["QW_24_ATR"].astype(float)
    joined["Y_opp"] = np.maximum(joined["QL"], joined["QS"]) - joined["QW"]
    joined["Y_dir"] = joined["QL"] - joined["QS"]
    joined["LabelAvailableTime"] = pd.concat([
        joined["label_available_time_6"],
        joined["label_available_time_12"],
        joined["label_available_time_24"],
    ], axis=1).max(axis=1)
    a = joined["stable_action"].astype(str)
    joined["Y_trade"] = np.where(
        a.isin(["Long", "Short"]), 1.0, np.where(a == "Wait", 0.0, np.nan))
    joined["Y_long"] = np.where(
        a == "Long", 1.0, np.where(a == "Short", 0.0, np.nan))

    # ---- label consistency (frozen rule) ----
    a2 = joined["baseline_stable_action"].astype(str)
    mismatch_all = int((a != a2).sum())
    used = a.isin(TRADE_ACTIONS) | a2.isin(TRADE_ACTIONS)
    mismatch_used = int((a[used] != a2[used]).sum())
    if mismatch_used != 0:
        _stop(f"STOP_ORACLE_BASELINE_MISMATCH used={mismatch_used}")

    # ---- §26 invariants ----
    inv = feature_invariants(joined)
    print(f"[invariants] {inv}", flush=True)
    if inv["negative_sr_mass_count"] or inv["negative_liq_mass_count"]:
        _stop(f"STOP_NEGATIVE_FIELD_MASS {inv}")
    if inv["trend_eff20_violations"] or inv["maturity_violations"]:
        _stop(f"STOP_RANGE_VIOLATION {inv}")
    if inv["at_price_side_violations"]:
        _stop(f"STOP_SIDE_MASS_VIOLATION {inv}")
    if inv["liq_missing_cluster_count"] > 0:
        _stop(f"STOP_MISSING_CLUSTER_COUNT {inv['liq_missing_cluster_count']}")

    # ---- split by trading days (frozen 60/20/20) ----
    days = np.sort(pd.to_datetime(joined["trading_day"]).dt.normalize().unique())
    n_d = len(days)
    n_tr, n_va = int(n_d * 0.6), int(n_d * 0.2)
    tr_days, va_days, te_days = days[:n_tr], days[n_tr:n_tr + n_va], days[n_tr + n_va:]
    dcol = pd.to_datetime(joined["trading_day"]).dt.normalize()
    m_tr, m_va, m_te = (
        dcol.isin(tr_days), dcol.isin(va_days), dcol.isin(te_days))
    lat = joined["LabelAvailableTime"]
    va_start, te_start = va_days[0], te_days[0]

    print(
        f"[split] train {pd.Timestamp(tr_days[0]).date()}.."
        f"{pd.Timestamp(tr_days[-1]).date()} n={int(m_tr.sum())} | "
        f"val {pd.Timestamp(va_start).date()}..{pd.Timestamp(va_days[-1]).date()} "
        f"n={int(m_va.sum())} | test {pd.Timestamp(te_start).date()}.."
        f"{pd.Timestamp(te_days[-1]).date()} n={int(m_te.sum())}",
        flush=True,
    )

    # ---- E1.1 §12/§13 univariate diagnostics: maturity + vol regime vs Yopp
    univariate = {}
    te_y = joined.loc[m_te, "Y_opp"].to_numpy(float)
    for c in MATURITY_COLS + [f"{tf}_vol_regime_log_ratio" for tf in TF]:
        if c not in joined.columns:
            continue
        x = joined.loc[m_te, c].to_numpy(float)
        ok = np.isfinite(x) & np.isfinite(te_y)
        univariate[c] = (
            float(spearmanr(x[ok], te_y[ok]).statistic) if int(ok.sum()) >= 3 else None
        )
    print(f"[univariate] {univariate}", flush=True)

    results = dict(
        task_id="FUTURE-ENV-R3C-E1-COMPACT-FIELD-REPRESENTATION",
        symbols=symbols,
        synthetic_field_test=synth,
        old96_parity=parity,
        join=dict(
            environment_rows=int(env_rows_total),
            oracle_rows=oracle_rows,
            matched_rows=int(len(joined)),
            environment_only_rows=int(env_rows_total - len(joined)),
            oracle_only_rows=0,
        ),
        label=dict(mismatch_all_rows=mismatch_all,
                   mismatch_used_vocabulary=mismatch_used),
        split=dict(
            train_start=str(tr_days[0]), train_end=str(tr_days[-1]),
            val_start=str(va_start), val_end=str(va_days[-1]),
            test_start=str(te_start), test_end=str(te_days[-1]),
            n_train=int(m_tr.sum()), n_val=int(m_va.sum()), n_test=int(m_te.sum()),
        ),
        invariants=inv,
        variants={},
    )

    deciles = []
    # E1.1: only TEST Yopp deciles, only for the attribution chain
    DECILE_VARIANTS = {
        "V0_OLD96", "V4_FIELD_CORE", "V5_FIELD_FULL",
        "V6_OLD96_AUX", "V7_FIELD_VOL", "V8_FIELD_MAT",
    }

    stages = [("VAL", m_tr, m_va, va_start), ("TEST", (m_tr | m_va), m_te, te_start)]

    for vname, cols in VARIANTS.items():
        missing = [c for c in cols if c not in joined.columns]
        if missing:
            _stop(f"{vname} missing columns {missing[:5]}")

        vres = dict(dimensions=len(cols))
        for stage_name, train_mask, eval_mask, boundary in stages:
            tr_p = joined[train_mask & ~(lat >= boundary)]
            purge = int((lat[train_mask] >= boundary).sum())
            ev = joined[eval_mask]

            pre = make_preprocessor(cols)
            Xtr = pre.fit_transform(tr_p[cols])
            Xev = pre.transform(ev[cols])

            fit_counts = dict(
                preprocessor_fit_count=1,
                train_transform_count=1,
                eval_transform_count=1,
                classification_fit_count=0,
                regression_fit_count=0,
            )
            sres = dict(purged_rows=purge, eval_rows=int(len(ev)),
                        fit_counts=fit_counts)

            train_action = tr_p["stable_action"].astype(str).to_numpy()
            eval_action = ev["stable_action"].astype(str).to_numpy()
            preds = {}

            for task, ycol, vocab, pname in (
                ("TaskA", "Y_trade", TRADE_ACTIONS, "pred_trade"),
                ("TaskB", "Y_long", DIRECTION_ACTIONS, "pred_direction"),
            ):
                ytr = tr_p[ycol].to_numpy(float)
                yev = ev[ycol].to_numpy(float)
                ok_tr = np.isin(train_action, vocab) & ~np.isnan(ytr)
                ok_ev = np.isin(eval_action, vocab) & ~np.isnan(yev)
                if ok_tr.sum() == 0 or ok_ev.sum() == 0:
                    continue
                clf = LogisticRegression(max_iter=2000)
                clf.fit(Xtr[ok_tr], ytr[ok_tr].astype(int))
                fit_counts["classification_fit_count"] += 1
                p = clf.predict_proba(Xev[ok_ev])[:, 1]

                full = np.full(len(ev), np.nan)
                full[np.flatnonzero(ok_ev)] = p
                preds[pname] = full

                grate = float(np.mean(ytr[ok_tr]))
                per = pd.Series(ytr[ok_tr]).groupby(
                    tr_p["symbol"].to_numpy()[ok_tr]).mean().to_dict()
                sp = np.array([per.get(s, grate)
                               for s in ev["symbol"].to_numpy()[ok_ev]], float)
                sres[task] = dict(
                    n=int(ok_ev.sum()),
                    global_prior=clf_metrics(
                        yev[ok_ev].astype(int), np.full(int(ok_ev.sum()), grate)),
                    symbol_prior=clf_metrics(yev[ok_ev].astype(int), sp),
                    logistic=clf_metrics(yev[ok_ev].astype(int), p),
                )

            for task, ycol, pname in (
                ("Yopp", "Y_opp", "pred_yopp"), ("Ydir", "Y_dir", "pred_ydir")):
                ytr = tr_p[ycol].to_numpy(float)
                yev = ev[ycol].to_numpy(float)
                mtr = np.isfinite(ytr)
                mev = np.isfinite(yev)
                if mtr.sum() == 0 or mev.sum() == 0:
                    continue
                rg = Ridge(alpha=1.0)
                rg.fit(Xtr[mtr], ytr[mtr])
                fit_counts["regression_fit_count"] += 1
                p = rg.predict(Xev[mev])

                full = np.full(len(ev), np.nan)
                full[np.flatnonzero(mev)] = p
                preds[pname] = full

                gmean = float(np.mean(ytr[mtr]))
                per = pd.Series(ytr[mtr]).groupby(
                    tr_p["symbol"].to_numpy()[mtr]).mean().to_dict()
                sm = np.array([per.get(s, gmean)
                               for s in ev["symbol"].to_numpy()[mev]], float)
                sres[task] = dict(
                    n=int(mev.sum()),
                    global_mean=reg_metrics(yev[mev], np.full(int(mev.sum()), gmean)),
                    symbol_mean=reg_metrics(yev[mev], sm),
                    ridge=reg_metrics(yev[mev], p),
                )

                if (
                    stage_name == "TEST"
                    and vname in DECILE_VARIANTS
                    and task == "Yopp"
                ):
                    dd = ev.copy()
                    dd["_pred"] = full
                    extra = ["Y_trade"] if task == "Yopp" else ["Y_long"]
                    deciles += decile_table(
                        dd.rename(columns={ycol: "_act"}), "_pred", "_act",
                        extra, f"{vname}_{task}")

            vres[stage_name] = sres

            del Xtr, Xev, pre

        results["variants"][vname] = vres
        print(
            f"[variant] {vname} dim={len(cols)} "
            f"TEST_A_auc={vres.get('TEST', {}).get('TaskA', {}).get('logistic', {}).get('roc_auc')} "
            f"TEST_Yopp_spear={vres.get('TEST', {}).get('Yopp', {}).get('ridge', {}).get('spearman')}",
            flush=True,
        )

    # ---- E1.1 §5/§10 attribution decomposition ----
    def _m(v, st, task, model, met):
        return results["variants"][v][st].get(task, {}).get(model, {}).get(
            met, float("nan"))

    attrib = {}
    for st in ("VAL", "TEST"):
        for label, task, model, met in (
            ("TaskA_auc", "TaskA", "logistic", "roc_auc"),
            ("TaskB_auc", "TaskB", "logistic", "roc_auc"),
            ("Yopp_spearman", "Yopp", "ridge", "spearman"),
            ("Ydir_spearman", "Ydir", "ridge", "spearman"),
        ):
            v0 = _m("V0_OLD96", st, task, model, met)
            v4 = _m("V4_FIELD_CORE", st, task, model, met)
            v5 = _m("V5_FIELD_FULL", st, task, model, met)
            v6 = _m("V6_OLD96_AUX", st, task, model, met)
            v7 = _m("V7_FIELD_VOL", st, task, model, met)
            v8 = _m("V8_FIELD_MAT", st, task, model, met)
            attrib.setdefault(st, {})[label] = dict(
                field_core_effect_V4_minus_V0=v4 - v0,
                aux_combined_V5_minus_V4=v5 - v4,
                aux_on_old96_V6_minus_V0=v6 - v0,
                vol_effect_V7_minus_V4=v7 - v4,
                maturity_effect_V8_minus_V4=v8 - v4,
                interaction_V5_minus_V7_minus_V8_plus_V4=v5 - v7 - v8 + v4,
            )
    results["attribution"] = attrib
    results["univariate_test_spearman_vs_Yopp"] = univariate

    results["runtime"] = dict(
        old_environment_sec=round(t_old, 3),
        field_environment_sec=round(t_field, 3),
        ratio=round(ratio, 3),
        total_sec=round(time.perf_counter() - t_all, 3),
    )

    rpath = os.path.join(args.out_dir, args.results_name)
    with open(rpath, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    dpath = os.path.join(args.out_dir, args.deciles_name)
    pd.DataFrame(deciles).to_csv(dpath, index=False)

    print(f"[save] results={rpath}", flush=True)
    print(f"[save] deciles={dpath}", flush=True)
    print(f"[done] total_sec={results['runtime']['total_sec']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
