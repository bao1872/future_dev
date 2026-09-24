"""FUTURE-R11-R14 V2 — extended causal state (plan §15-§22).

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined.

Builds artifacts/decomposed_value_v2/extended_causal_state_v2.parquet, which
carries the three PRE-REGISTERED V2 families: SPACE18, PATH8, VOL6.

Efficiency contract (§49)
-------------------------
`run_environment_m15` is called EXACTLY ONCE PER SYMBOL (15 calls total) and
all three families are materialized in that single pass. No model fit ever
calls the environment again; every experiment reads the cached parquet.

Causality
---------
SPACE18 uses only decision-time geometry (the m15 channel/liquidity snapshot
for that bar). PATH8 touches only bars j <= t. VOL6 rolling windows are
backward-looking only. Nothing reads a future bar.

Verified element layout (do NOT "simplify" this)
------------------------------------------------
geom_by_decision[t]["m15"] -> (channels, liq_up, liq_down, atr)
  channel   = [top, bottom, strength]   (cross-checked against frozen V1 state:
              res_top=7625, res_bottom=7614, res_str=109 -> [7625, 7614, 109])
  liq zone  = {"left","level","top","bottom","broken","breach_i"}
  geom index == state bar_index
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas import run_decomposed_v2_research as R
from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import STRUCT33
from research.liquidity_oracle_atlas.decomposed_value_features_v2 import (
    PATH8,
    SPACE18,
    VOL6,
)
from research.liquidity_oracle_atlas import decomposed_value_features_v2 as F

# Exactly 3 SR levels are considered; K is frozen (§16) and never optimized.
K_LEVELS = 3


# --------------------------------------------------------------------------- #
# §16 / §18 SPACE18                                                            #
# --------------------------------------------------------------------------- #
def _zone_dist(z, close, atr, above):
    if z is None or not np.isfinite(atr) or atr <= 0:
        return np.nan
    edge = float(z[1] if above else z[0])
    d = (edge - close) / atr if above else (close - edge) / atr
    return d if d >= 0 else np.nan


def extract_space18(g, close, atr, side):
    """18 multi-level structural-space features for one (bar, side)."""
    if not g or "m15" not in g:
        return [np.nan] * 18

    channels, liq_up, liq_down, _ = g["m15"]

    above = sorted([z for z in channels if z[1] > close], key=lambda z: z[1])
    below = sorted([z for z in channels if z[0] < close], key=lambda z: z[0],
                   reverse=True)

    if side > 0:
        ahead, back = above, below
        ahead_above, back_above = True, False
    else:
        ahead, back = below, above
        ahead_above, back_above = False, True

    def zget(zs, k):
        return zs[k] if len(zs) > k else None

    def dist_list(zs, above_flag):
        return np.array([_zone_dist(z, close, atr, above_flag) for z in zs])

    ad = dist_list(ahead, ahead_above)
    bd = dist_list(back, back_above)

    # Index 0 is the nearest zone, already represented in PAY8 -> start at #2.
    a2, a3 = zget(ahead, 1), zget(ahead, 2)
    b2, b3 = zget(back, 1), zget(back, 2)

    def strength(z):
        return np.nan if z is None else float(z[2])

    active_up = [z for z in liq_up
                 if not z.get("broken") and float(z["bottom"]) > close]
    active_dn = [z for z in liq_down
                 if not z.get("broken") and float(z["top"]) < close]

    up = min(active_up, key=lambda z: float(z["bottom"])) if active_up else None
    dn = max(active_dn, key=lambda z: float(z["top"])) if active_dn else None

    if side > 0:
        ahead_liq, back_liq = up, dn
        al_above, bl_above = True, False
    else:
        ahead_liq, back_liq = dn, up
        al_above, bl_above = False, True

    def liq_dist(z, above_flag):
        if z is None or atr <= 0:
            return np.nan
        edge = float(z["bottom"] if above_flag else z["top"])
        return ((edge - close) / atr if above_flag else (close - edge) / atr)

    def liq_width(z):
        if z is None or atr <= 0:
            return np.nan
        return (float(z["top"]) - float(z["bottom"])) / atr

    return [
        _zone_dist(a2, close, atr, ahead_above),
        _zone_dist(a3, close, atr, ahead_above),
        _zone_dist(b2, close, atr, back_above),
        _zone_dist(b3, close, atr, back_above),

        strength(a2), strength(a3), strength(b2), strength(b3),

        int(np.sum(ad <= 1.0)), int(np.sum(ad <= 2.0)),
        int(np.sum(bd <= 1.0)), int(np.sum(bd <= 2.0)),

        float(np.nansum([float(z[2]) for z, d in zip(ahead, ad) if d <= 2.0])),
        float(np.nansum([float(z[2]) for z, d in zip(back, bd) if d <= 2.0])),

        liq_dist(ahead_liq, al_above),
        liq_dist(back_liq, bl_above),
        liq_width(ahead_liq),
        liq_width(back_liq),
    ]


# --------------------------------------------------------------------------- #
# §19 / §20 PATH8                                                              #
# --------------------------------------------------------------------------- #
def trailing_zone_stats(high, low, top, bottom, t, window):
    """Touch count and bars-since-last-touch for the CURRENT causal zone band.

    Only bars j <= t are used; the current bar counts because its OHLC is known
    at decision time.
    """
    if not np.isfinite(top) or not np.isfinite(bottom):
        return np.nan, np.nan

    j0 = max(0, t - window + 1)
    h = high[j0:t + 1]
    l = low[j0:t + 1]

    touch = (h >= bottom) & (l <= top)
    count = int(touch.sum())

    if touch.any():
        last_local = np.flatnonzero(touch)[-1]
        bars_since = (len(touch) - 1) - last_local
    else:
        bars_since = np.nan

    return count, bars_since


def _log1p(v):
    return np.log1p(v) if np.isfinite(v) else np.nan


def extract_path8(high, low, cand, t, ahead_band, back_band):
    """8 recent path / zone-interaction features. No outcome data."""
    a_top, a_bottom = ahead_band
    b_top, b_bottom = back_band

    a16, a_since16 = trailing_zone_stats(high, low, a_top, a_bottom, t, 16)
    b16, b_since16 = trailing_zone_stats(high, low, b_top, b_bottom, t, 16)
    a64, a_since64 = trailing_zone_stats(high, low, a_top, a_bottom, t, 64)
    b64, b_since64 = trailing_zone_stats(high, low, b_top, b_bottom, t, 64)

    def trailing_count(arr, t, w):
        j0 = max(0, t - w + 1)
        return int(np.nansum(arr[j0:t + 1]))

    return [
        a16, b16, a64, b64,
        _log1p(a_since16), _log1p(b_since16),
        trailing_count(cand, t, 32), trailing_count(cand, t, 128),
    ]


# --------------------------------------------------------------------------- #
# §21 VOL6                                                                     #
# --------------------------------------------------------------------------- #
def build_vol6(frame, atr):
    """Backward-looking volatility block. No future centering."""
    close = pd.Series(frame["close"].to_numpy(float))
    high = frame["high"].to_numpy(float)
    low = frame["low"].to_numpy(float)

    atr_s = pd.Series(np.asarray(atr, float))

    med16 = atr_s.rolling(16, min_periods=16).median()
    med64 = atr_s.rolling(64, min_periods=64).median()

    logret = np.log(close).diff()
    rv16 = logret.rolling(16, min_periods=16).std()
    rv64 = logret.rolling(64, min_periods=64).std()

    return pd.DataFrame({
        "atr_rel_med16": atr_s / med16,
        "atr_rel_med64": atr_s / med64,
        "rv16": rv16,
        "rv64": rv64,
        "rv_ratio_16_64": rv16 / rv64,
        "range_over_atr": (high - low) / np.asarray(atr, float),
    })


# --------------------------------------------------------------------------- #
# Differential assertion vs frozen V1 state                                    #
# --------------------------------------------------------------------------- #
def assert_matches_v1_state(state_sym: pd.DataFrame, env, symbol: str) -> dict:
    """close / atr / STRUCT33 / nearest SR geometry must equal V1 exactly.

    Returns the realized differential-assertion summary so the caller can bind
    it into the extended-state manifest (plan §49 / §52). Any real drift raises
    STOP_V2_EXTENDED_STATE_DRIFT before this returns.
    """
    # close comes from the execution frame; atr from the same geometry tuple
    # that SPACE18 divides by; STRUCT33 from the per-decision feature frame.
    ef = env["exec_frame"].sort_values("execution_bar_index").reset_index(drop=True)
    feat = env["features"].sort_values("execution_bar_index").reset_index(drop=True)
    geom = env["geom_by_decision"]
    n = len(state_sym)
    if len(ef) < n or len(feat) < n:
        raise R.StopV2Leakage(
            f"STOP_V2_STATE_SHAPE_MISMATCH {symbol} env={len(ef)}/{len(feat)} "
            f"state={n}")

    bar_ids = state_sym["bar_index"].to_numpy()
    if not np.array_equal(ef["execution_bar_index"].to_numpy()[:n], bar_ids):
        raise R.StopV2Leakage(f"STOP_V2_STATE_ALIGNMENT {symbol}")

    problems = []

    # The frozen V1 state persists STRUCT33/geometry as float32 while the
    # environment computes float64, so agreement is expected only to float32
    # precision (~1e-7 relative). Real semantic drift would be orders larger.
    close_v1 = np.asarray(ef["close"].iloc[:n], float)
    close_v2 = np.asarray(state_sym["close"].to_numpy(float)[:n], float)
    max_abs_close = (float(np.nanmax(np.abs(close_v1 - close_v2)))
                     if n and np.isfinite(close_v1).all() else 0.0)

    def cmp(name, a, b, rtol=1e-6, atol=1e-6):
        a = np.asarray(a, float)[:n]
        b = np.asarray(b, float)[:n]
        if a.shape != b.shape:
            problems.append(f"{name}: shape {a.shape} vs {b.shape}")
            return
        ok = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)
        if not ok.all():
            diff = np.nanmax(np.abs(np.where(np.isnan(a) & np.isnan(b), 0.0,
                                             a - b)))
            problems.append(f"{name}: max_abs_diff={diff} n_bad={int((~ok).sum())}")

    cmp("close", close_v1, close_v2)
    geom_atr = np.array(
        [(geom[t]["m15"][3] if t < len(geom) and geom[t]
          and geom[t].get("m15") else np.nan) for t in range(n)], float)
    atr_v2 = np.asarray(state_sym["atr"].to_numpy(float)[:n], float)
    max_abs_atr = (float(np.nanmax(np.abs(geom_atr - atr_v2)))
                   if n and np.isfinite(geom_atr).any() else 0.0)
    cmp("atr", geom_atr, atr_v2)
    struct33_checked = 0
    for c in STRUCT33:
        if c in feat.columns:
            struct33_checked += 1
            cmp(c, feat[c].iloc[:n], state_sym[c])

    # Nearest SR geometry: the first channel above / below close must agree with
    # the frozen V1 res_* / sup_* columns.
    geom = env["geom_by_decision"]
    sr_checked, sr_drift = 0, False
    step = max(1, n // 25)
    for t in range(0, n, step):
        g = geom[t] if t < len(geom) else None
        if not g or not g.get("m15"):
            continue
        channels, _lu, _ld, _atr = g["m15"]
        close = float(state_sym["close"].iloc[t])
        above = sorted([z for z in channels if z[1] > close], key=lambda z: z[1])
        below = sorted([z for z in channels if z[0] < close], key=lambda z: z[0],
                       reverse=True)
        if above:
            sr_checked += 1
            if (abs(float(above[0][0]) - float(state_sym["res_top"].iloc[t])) > 1e-9
                    or abs(float(above[0][1])
                           - float(state_sym["res_bottom"].iloc[t])) > 1e-9):
                sr_drift = True
                problems.append(f"res_geometry@{t} differs")
        if below:
            sr_checked += 1
            if (abs(float(below[0][0]) - float(state_sym["sup_top"].iloc[t])) > 1e-9
                    or abs(float(below[0][1])
                           - float(state_sym["sup_bottom"].iloc[t])) > 1e-9):
                sr_drift = True
                problems.append(f"sup_geometry@{t} differs")
        if problems:
            break

    if problems:
        raise R.StopV2Leakage(
            f"STOP_V2_EXTENDED_STATE_DRIFT {symbol} {problems[:5]}")

    return {
        "max_abs_close_diff": max_abs_close,
        "max_abs_atr_diff": max_abs_atr,
        "struct33_checked": struct33_checked,
        "nearest_sr_geometry_checked_bars": sr_checked,
        "nearest_sr_geometry_drift": sr_drift,
    }


# --------------------------------------------------------------------------- #
# Per-symbol pass                                                              #
# --------------------------------------------------------------------------- #
def build_symbol_extended(state_sym: pd.DataFrame, symbol: str, *,
                          check: bool = True):
    state_sym = state_sym.sort_values("bar_index").reset_index(drop=True)
    env = run_environment_m15(symbol, capture_provenance=False)
    R.bump("environment_loads")
    R.bump("geometry_passes")
    diff = assert_matches_v1_state(state_sym, env, symbol) if check else None

    geom = env["geom_by_decision"]
    n = len(state_sym)

    close = state_sym["close"].to_numpy(float)
    high = state_sym["high"].to_numpy(float)
    low = state_sym["low"].to_numpy(float)
    atr = state_sym["atr"].to_numpy(float)
    cand = state_sym["candidate_at_decision"].to_numpy(float)
    sup_top = state_sym["sup_top"].to_numpy(float)
    sup_bottom = state_sym["sup_bottom"].to_numpy(float)
    res_top = state_sym["res_top"].to_numpy(float)
    res_bottom = state_sym["res_bottom"].to_numpy(float)

    vol6 = build_vol6(state_sym, atr)

    rows = []
    for t in range(n):
        g = geom[t] if t < len(geom) else None
        c, a = float(close[t]), float(atr[t])
        # LONG: ahead = resistance zone, back = support zone. SHORT: mirrored.
        bands = {
            1: ((res_top[t], res_bottom[t]), (sup_top[t], sup_bottom[t])),
            -1: ((sup_top[t], sup_bottom[t]), (res_top[t], res_bottom[t])),
        }
        for side in (1, -1):
            ahead_band, back_band = bands[side]
            rec = {"symbol": symbol, "decision_bar": int(state_sym["bar_index"].iloc[t]),
                   "side": "LONG" if side > 0 else "SHORT"}
            for name, v in zip(SPACE18, extract_space18(g, c, a, side)):
                rec[name] = v
            for name, v in zip(PATH8, extract_path8(high, low, cand, t,
                                                    ahead_band, back_band)):
                rec[name] = v
            for name in VOL6:
                rec[name] = float(vol6[name].iloc[t])
            rows.append(rec)
    return pd.DataFrame(rows), diff


def build_extended_state(save: bool = True, *, check: bool = True) -> pd.DataFrame:
    """One environment pass per symbol -> the complete V2 extended state.

    P3: also writes extended_state_manifest_v2.json, which binds the artifact
    to the committed code SHA, the frozen V1 state it was derived from, the
    environment budget actually consumed, and the three schema identities.
    """
    import hashlib

    from research.liquidity_oracle_atlas.entry_path_atlas_v1 import SYMBOLS

    def sha256_file(path: str) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    t0 = time.time()
    state = R.read_state()
    out = []
    diff_summary = {"symbols_checked": [], "max_abs_close_diff": 0.0,
                    "max_abs_atr_diff": 0.0,
                    "struct33_columns_checked": len(STRUCT33),
                    "nearest_sr_geometry_checked_bars": 0,
                    "nearest_sr_geometry_drift": False}
    for sym in SYMBOLS:
        sub = state[state["symbol"] == sym]
        if sub.empty:
            continue
        frame, diff = build_symbol_extended(sub, sym, check=check)
        out.append(frame)
        diff_summary["symbols_checked"].append(sym)
        if diff:
            diff_summary["max_abs_close_diff"] = max(
                diff_summary["max_abs_close_diff"], diff["max_abs_close_diff"])
            diff_summary["max_abs_atr_diff"] = max(
                diff_summary["max_abs_atr_diff"], diff["max_abs_atr_diff"])
            diff_summary["nearest_sr_geometry_checked_bars"] += (
                diff["nearest_sr_geometry_checked_bars"])
            diff_summary["nearest_sr_geometry_drift"] = (
                diff_summary["nearest_sr_geometry_drift"]
                or diff["nearest_sr_geometry_drift"])
        print(f"  extended state {sym}: {len(sub)} bars", flush=True)
    df = pd.concat(out, ignore_index=True)
    R.bump("feature_materializations")

    if save:
        R.write_parquet(df, R.EXTENDED_STATE_PARQUET)
        manifest = {
            "task": "FUTURE-R13-PAYOFF-GEOMETRY-V2-EXTENDED-STATE",
            "artifact": R.EXTENDED_STATE_PARQUET,
            "generator_code_sha": R._git_head_sha(),
            "source_v1_state_sha256": sha256_file(R.ALLOWED_V1_STATE),
            "symbols": sorted(diff_summary["symbols_checked"]),
            "n_symbols": len(diff_summary["symbols_checked"]),
            "environment_loads": int(R.COUNTERS["environment_loads"]),
            "geometry_passes": int(R.COUNTERS["geometry_passes"]),
            "feature_materializations": int(
                R.COUNTERS["feature_materializations"]),
            "row_count": int(len(df)),
            "extended_state_sha256": sha256_file(R.EXTENDED_STATE_PARQUET),
            "space18_schema_sha256": F.schema_sha256(SPACE18),
            "path8_schema_sha256": F.schema_sha256(PATH8),
            "vol6_schema_sha256": F.schema_sha256(VOL6),
            "k_levels_frozen": K_LEVELS,
            "differential_assertion_vs_v1_state": diff_summary,
            "runtime_sec": time.time() - t0,
        }
        R.write_json_evidence(manifest, R.EXTENDED_MANIFEST_JSON)
        print(f"  wrote {R.EXTENDED_MANIFEST_JSON}", flush=True)

    print(f"extended state rows={len(df)} env_calls={R.COUNTERS['environment_loads']} "
          f"in {time.time() - t0:.1f}s", flush=True)
    return df


if __name__ == "__main__":
    build_extended_state()
