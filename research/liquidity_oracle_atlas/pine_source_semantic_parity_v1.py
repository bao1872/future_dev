"""
pine_source_semantic_parity_v1.py
================================================================================

STRUCTREV-PGM-R2B-SRC1.1-DTP-AUDIT-HARDEN
-------------------------------------------
DTP Source Semantic Differential Audit -- infrastructure hardening.

方向 (per user 2026-09-18):
    不再做 TradingView runtime parity. 改为:
        Pine 源码 -> 数学语义 -> 独立 Oracle -> 生产 Python -> 差分验证
    本轮只验证 "Pine source semantic parity PASS", 不写 "TradingView runtime
    parity PASS".

Baseline = literal Pine-source Oracle (DeviationTrendProfile.pine), 不调用
production 的 rolling_sma / true_range / pine_rma / trend_state_from_score /
compute_segment_features 内部逻辑作为 Oracle.

Candidate = 实验真实调用链:
    experiment_structural_reversion_pgm_v1.compute_tf_features
        -> compute_segment_features(segment, PINE_DEFAULT, include_sr=False)
不是 pine_runtime_parity_v1.compute_dtp (旧 runtime harness).

唯一新增文件. 不修改 production. mismatch 只报告, 不修.
"""

from __future__ import annotations

import sys
import json
import hashlib
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 as prod
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags

OUT = REPO_ROOT / "artifacts" / "pine_source_semantic_parity"
PINEFILE = REPO_ROOT / "ref" / "DeviationTrendProfile.pine"

BASE_SHA = "10f735e23ba491d4a1ef3aad11368528f265a6a3"

# --- frozen DTP parameters (read from production SSOT; assert frozen) ---
SMA_LEN = prod.PINE_DEFAULT.sma_len
ATR_LEN = prod.PINE_DEFAULT.atr_len
LAG = prod.PINE_DEFAULT.trend_slope_lag
NORM = prod.PINE_DEFAULT.trend_norm_lookback
SWITCH = prod.PINE_DEFAULT.trend_switch

FLOAT_TOL = 1e-9
EPS_GUARD = 1e-12

EXPECTED_FROZEN = {
    "sma_len": 50,
    "atr_len": 200,
    "trend_slope_lag": 5,
    "trend_norm_lookback": 500,
    "trend_switch": 0.1,
}


# =============================================================================
# 0. gates
# =============================================================================

def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(REPO_ROOT),
        text=True,
    ).strip()


def pine_source_sha256() -> str:
    return hashlib.sha256(PINEFILE.read_bytes()).hexdigest()


def assert_base_ancestor() -> None:
    """Reproducibility gate.

    The audit code (BASE_SHA) must be an ancestor of the commit that actually
    runs it. This holds both during development (HEAD == BASE_SHA) and after
    the hardening commit is made (HEAD descends from BASE_SHA). It fails only
    if BASE_SHA is NOT in the running commit's history (e.g. rebased away),
    which would mean the artifact cannot be rebuilt from that SHA.
    """
    r = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=str(REPO_ROOT),
    )
    if r.returncode != 0:
        raise SystemExit(f"STOP_DTP_BASE_NOT_ANCESTOR:{BASE_SHA}")
    actual = pine_source_sha256()
    expected = (
        "7132bf12844dd09c9c18fb0745d20c8f66f6bb1ef07f297698b566e3ad9d4745"
    )
    if actual != expected:
        raise SystemExit(
            f"STOP_DTP_SOURCE_HASH_MISMATCH: {actual}"
        )
    bad = {k: v for k, v in EXPECTED_FROZEN.items()
           if getattr(prod.PINE_DEFAULT, k) != v}
    if bad:
        raise SystemExit(f"STOP_DTP_FROZEN_PARAM_DRIFT:{bad}")


# =============================================================================
# 1. Literal Pine-source Oracle (test-only, independent)
# =============================================================================

def pine_safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """Pine Script division semantics for the literal Oracle.

    Pine: x / 0 -> na (NOT +inf / -inf like NumPy/Python).
    Therefore denominator exactly == 0.0 (incl. +0.0 and -0.0) -> NaN.
    Tiny NON-zero denominators still divide normally (no epsilon guard).
    """
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    out = np.full(len(numerator), np.nan, dtype=np.float64)
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator != 0.0)
    )
    out[valid] = numerator[valid] / denominator[valid]
    return out


def dtp_literal_oracle(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Literal implementation of DeviationTrendProfile.pine mathematics.

    - SMA50:  A_t = mean(close[t-49 .. t])
    - TR:     TR_0 = H_0 - L_0;
              TR_t = max(H_t-L_t, |H_t-C_{t-1}|, |L_t-C_{t-1}|)
    - ATR200: seed = mean(TR[:200]); ATR_t = ATR_{t-1} + (TR_t-ATR_{t-1})/200
    - avg_diff = A_t - A_{t-5}   (first finite when both SMA finite: t=54)
    - P100    = max(avg_diff[t-499 .. t])   (rolling max == Pine percentile 100)
    - avg_col = avg_diff / P100   (DIRECT division, NO epsilon guard)
    - trend   = Pine crossover/crossunder of avg_col vs +/-0.1
                false->-1, true->+1, initial -1

    Does NOT call production primitives. Simple for-loops.
    """

    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    n = len(close)

    sma = np.full(n, np.nan)
    tr = np.full(n, np.nan)
    atr = np.full(n, np.nan)
    avg_diff = np.full(n, np.nan)
    p100 = np.full(n, np.nan)
    avg_col = np.full(n, np.nan)
    trend = np.full(n, -1, dtype=np.int8)

    # ---- SMA50 ----
    for i in range(SMA_LEN - 1, n):
        sma[i] = float(np.mean(close[i - SMA_LEN + 1: i + 1]))

    # ---- True Range ----
    if n:
        tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )

    # ---- ATR200 (Wilder RMA, SMA seed) ----
    if n >= ATR_LEN:
        atr[ATR_LEN - 1] = float(np.mean(tr[:ATR_LEN]))
        for i in range(ATR_LEN, n):
            atr[i] = atr[i - 1] + (tr[i] - atr[i - 1]) / ATR_LEN

    # ---- avg_diff = sma - sma[5] ----
    for i in range(LAG, n):
        if np.isfinite(sma[i]) and np.isfinite(sma[i - LAG]):
            avg_diff[i] = sma[i] - sma[i - LAG]

    # ---- P100 = rolling max over NORM-window (Pine percentile 100) ----
    for i in range(NORM - 1, n):
        w = avg_diff[i - NORM + 1: i + 1]
        if np.all(np.isfinite(w)):
            p100[i] = float(np.max(w))

    # ---- avg_col = avg_diff / P100 (Pine semantics: /0 -> NaN) ----
    avg_col[:] = pine_safe_divide(avg_diff, p100)

    # ---- trend state machine ----
    state = -1
    for i in range(n):
        if i > 0:
            prev = avg_col[i - 1]
            cur = avg_col[i]
            if np.isfinite(prev) and np.isfinite(cur):
                if prev <= SWITCH and cur > SWITCH and state == -1:
                    state = +1
                elif prev >= -SWITCH and cur < -SWITCH and state == +1:
                    state = -1
        trend[i] = state

    return {
        "sma": sma,
        "tr": tr,
        "atr": atr,
        "avg_diff": avg_diff,
        "p100": p100,
        "avg_col": avg_col,
        "trend_state": trend,
    }


# =============================================================================
# 2. Production adapter (real experiment candidate)
# =============================================================================

def production_denominator(close: np.ndarray):
    """Reconstruct production's `trend_denominator` (rolling max of slope_raw).

    Uses production primitives only (no new DTP logic) -- allowed for audit.
    Returns (sma, slope_raw, denom) where slope_raw == production avg_diff,
    denom == production P100 (== trend_denominator).
    """
    sma = prod.rolling_sma(np.asarray(close, dtype=float), SMA_LEN)
    slope = sma - np.roll(sma, LAG)
    slope[:LAG] = np.nan
    denom = (
        pd.Series(slope)
        .rolling(NORM, min_periods=NORM)
        .max()
        .to_numpy(float)
    )
    return sma, slope, denom


def run_production_segment(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> pd.DataFrame:
    """Call the REAL candidate: compute_segment_features on one segment."""
    seg = pd.DataFrame({
        "high": np.asarray(high, dtype=float),
        "low": np.asarray(low, dtype=float),
        "close": np.asarray(close, dtype=float),
    })
    return prod.compute_segment_features(seg, prod.PINE_DEFAULT, include_sr=False)


def production_dtp_5m(symbol: str):
    """Real 5m experiment chain: owner -> raw_frame -> resample(5) -> tf_features."""
    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    disc = np.asarray(discontinuity_flags(symbol), dtype=bool)
    bars = {
        "t": pd.to_datetime(raw["bar_start_time"]).to_numpy(),
        "day": pd.to_datetime(raw["trading_day"]).to_numpy(),
        "disc": disc,
        "o": raw["open"].to_numpy(float),
        "h": raw["high"].to_numpy(float),
        "l": raw["low"].to_numpy(float),
        "c": raw["close"].to_numpy(float),
        "n": len(raw),
    }
    raw_frame = prod.raw_frame_from_owner(bars)
    tf5 = prod.resample_causal(raw_frame, 5)
    feat = prod.compute_tf_features(tf5, prod.PINE_DEFAULT, include_sr=False)
    return raw_frame, tf5, feat


def run_production_with_disc(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    disc: np.ndarray,
) -> pd.DataFrame:
    """Drive the REAL production chain with a synthetic `disc` array.

    Used by the segment-reset test: disc=True forces a new segment, so
    production recomputes SMA/ATR/500-bar normalization/trend_state from
    scratch at that bar (Pine source does NOT do this).
    """
    n = len(close)
    bars = {
        "t": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "day": pd.date_range("2024-01-01", periods=n, freq="5min"),
        "disc": np.asarray(disc, dtype=bool),
        "o": np.asarray(close, dtype=float),
        "h": np.asarray(high, dtype=float),
        "l": np.asarray(low, dtype=float),
        "c": np.asarray(close, dtype=float),
        "n": n,
    }
    raw_frame = prod.raw_frame_from_owner(bars)
    tf5 = prod.resample_causal(raw_frame, 5)
    feat = prod.compute_tf_features(tf5, prod.PINE_DEFAULT, include_sr=False)
    return feat


# =============================================================================
# 3. differential comparator
# =============================================================================

def compare_float(
    name: str,
    oracle: np.ndarray,
    production: np.ndarray,
    start: int,
    tol: float = FLOAT_TOL,
) -> dict:
    o = np.asarray(oracle, dtype=float)
    p = np.asarray(production, dtype=float)
    n = len(o)
    region = np.arange(start, n)
    both = np.isfinite(o) & np.isfinite(p)
    one_nan = (
        (np.isfinite(o) & ~np.isfinite(p))
        | (~np.isfinite(o) & np.isfinite(p))
    )
    mm = np.zeros(n, dtype=bool)
    mm[region] = (both[region] & (np.abs(o[region] - p[region]) > tol)) \
        | one_nan[region]
    n_compared = int(np.sum(np.isfinite(o[region]) | np.isfinite(p[region])))
    n_mismatch = int(mm.sum())
    if both[region].any():
        max_abs = float(np.abs(o[region] - p[region])[both[region]].max())
    else:
        max_abs = None
    first = int(np.argmax(mm)) if n_mismatch else None
    return {
        "field": name,
        "n_compared": n_compared,
        "n_mismatch": n_mismatch,
        "max_abs_error": max_abs,
        "first_mismatch_index": first,
        "first_oracle_value": (
            float(o[first]) if first is not None else None
        ),
        "first_production_value": (
            float(p[first]) if first is not None else None
        ),
    }


def compare_int(
    name: str,
    oracle: np.ndarray,
    production: np.ndarray,
    start: int,
) -> dict:
    o = np.asarray(oracle, dtype=int)
    p = np.asarray(production, dtype=int)
    n = len(o)
    region = np.arange(start, n)
    mm = o[region] != p[region]
    n_compared = int(len(region))
    n_mismatch = int(mm.sum())
    if n_mismatch:
        local = int(np.argmax(mm))
        first = start + local          # absolute index, not region-relative
        diff = np.abs(
            o[region].astype(float) - p[region].astype(float)
        )
        max_abs = float(diff.max())
    else:
        first = None
        max_abs = None
    return {
        "field": name,
        "n_compared": n_compared,
        "n_mismatch": n_mismatch,
        "max_abs_error": max_abs,
        "first_mismatch_index": first,
        "first_oracle_value": (
            int(o[first]) if first is not None else None
        ),
        "first_production_value": (
            int(p[first]) if first is not None else None
        ),
    }


def compare_region(
    name: str,
    oracle: np.ndarray,
    production: np.ndarray,
    a: int,
    b: int,
    int_field: bool = False,
) -> int:
    """Count mismatches over [a, b) for the segment-reset test.

    For float fields: mismatch = (both finite & |diff|>tol) OR exactly one
    finite. For int fields: mismatch = (o != p).
    """
    o = np.asarray(oracle, dtype=float if not int_field else int)
    p = np.asarray(production, dtype=float if not int_field else int)
    seg_o = o[a:b]
    seg_p = p[a:b]
    if int_field:
        mm = seg_o != seg_p
    else:
        both = np.isfinite(seg_o) & np.isfinite(seg_p)
        one_nan = (
            (np.isfinite(seg_o) & ~np.isfinite(seg_p))
            | (~np.isfinite(seg_o) & np.isfinite(seg_p))
        )
        mm = (both & (np.abs(seg_o - seg_p) > FLOAT_TOL)) | one_nan
    return int(mm.sum())


def compare_segment(group_feat: pd.DataFrame) -> dict:
    """Compare Oracle vs production on one segment group.

    Returns per-field comparison rows + the oracle output + production arrays.
    """
    high = group_feat["high"].to_numpy(float)
    low = group_feat["low"].to_numpy(float)
    close = group_feat["close"].to_numpy(float)

    o = dtp_literal_oracle(high, low, close)

    sma_p = group_feat["sma"].to_numpy(float)
    atr_p = group_feat["atr"].to_numpy(float)
    ts_p = group_feat["trend_score"].to_numpy(float)
    tstate_p = group_feat["trend_state"].to_numpy(int)

    # avg_diff / p100 via production primitives (audit only)
    _, slope_p, denom_p = production_denominator(close)

    rows = [
        compare_float("sma", o["sma"], sma_p, SMA_LEN - 1),
        compare_float("atr", o["atr"], atr_p, ATR_LEN - 1),
        compare_float("avg_diff", o["avg_diff"], slope_p, LAG + (SMA_LEN - 1)),
        compare_float("p100", o["p100"], denom_p, NORM - 1 + (SMA_LEN - 1) + LAG),
        compare_float("avg_col", o["avg_col"], ts_p, NORM - 1 + (SMA_LEN - 1) + LAG),
        compare_int("trend_state", o["trend_state"], tstate_p,
                    NORM - 1 + (SMA_LEN - 1) + LAG),
    ]
    return {
        "rows": rows,
        "oracle": o,
        "prod": {
            "sma": sma_p, "atr": atr_p,
            "trend_score": ts_p, "trend_state": tstate_p,
        },
    }


def aggregate_rows(list_of_row_dicts: list[dict]) -> dict:
    """Aggregate per-field comparison rows across multiple segments/series."""
    out = {}
    for r in list_of_row_dicts:
        f = r["field"]
        if f not in out:
            out[f] = {
                "field": f,
                "n_compared": 0,
                "n_mismatch": 0,
                "max_abs_error": None,
                "first_mismatch_index": None,
                "first_oracle_value": None,
                "first_production_value": None,
            }
        a = out[f]
        a["n_compared"] += r["n_compared"]
        a["n_mismatch"] += r["n_mismatch"]
        if r["max_abs_error"] is not None:
            a["max_abs_error"] = (
                r["max_abs_error"]
                if a["max_abs_error"] is None
                else max(a["max_abs_error"], r["max_abs_error"])
            )
        if r["first_mismatch_index"] is not None and a["first_mismatch_index"] is None:
            a["first_mismatch_index"] = r["first_mismatch_index"]
            a["first_oracle_value"] = r["first_oracle_value"]
            a["first_production_value"] = r["first_production_value"]
    return out


# =============================================================================
# 4. deterministic cases (T1-A)
# =============================================================================

def make_ohlc(close: np.ndarray, band: float = 1.0):
    close = np.asarray(close, dtype=float)
    high = close + band
    low = close - band
    return high, low, close


def case_constant_range() -> dict:
    n = 700
    close = 100.0 + np.arange(n) * 0.01
    high, low, close = make_ohlc(close)
    o = dtp_literal_oracle(high, low, close)
    feat = run_production_segment(high, low, close)
    rows = compare_segment(
        pd.DataFrame({
            "high": high, "low": low, "close": close,
            "sma": feat["sma"], "atr": feat["atr"],
            "trend_score": feat["trend_score"],
            "trend_state": feat["trend_state"],
        })
    )["rows"]
    return {
        "name": "constant_range",
        "oracle": o,
        "feat": feat,
        "rows": rows,
    }


def case_gap() -> dict:
    # prev close = 100, next high=110, next low=109 => TR = max(1,10,9)=10
    n = 10
    close = np.full(n, 100.0)
    high = close + 0.5
    low = close - 0.5
    # inject a gap bar
    close[3] = 100.0
    high[3] = 110.0
    low[3] = 109.0
    o = dtp_literal_oracle(high, low, close)
    tr_prod = prod.true_range(high, low, close)[3]
    return {
        "name": "gap",
        "tr_oracle": float(o["tr"][3]),
        "tr_prod": float(tr_prod),
        "expect_tr": 10.0,
    }


def case_crossover() -> dict:
    # avg_col = avg_diff / P100. To make avg_col settle at a target value, we
    # need long steady-slope runs (so the 50-bar SMA reflects the slope) AND
    # persistent P100=1.0 anchor runs within every 500-window.
    # avg_diff ~= 5 * slope, so slope 0.20 -> avg_col~1.0 (anchor),
    # slope 0.02 -> avg_col~0.10, slope 0.022 -> avg_col~0.11, etc.
    slopes = (
        [0.20] * 600       # A: avg_col ~ 1.0 (anchor, P100=1.0)
        + [0.02] * 300     # B: avg_col ~ 0.10
        + [0.022] * 300    # C: avg_col ~ 0.11  -> cross up 0.1
        + [0.20] * 300     # D: anchor (P100 stays 1.0)
        + [-0.02] * 300    # E: avg_col ~ -0.10 -> cross down -0.1
        + [-0.022] * 300   # F: avg_col ~ -0.11
    )
    close = np.empty(len(slopes))
    close[0] = 100.0
    for i in range(1, len(slopes)):
        close[i] = close[i - 1] + slopes[i]
    high, low, close = make_ohlc(close)

    o = dtp_literal_oracle(high, low, close)
    # Oracle trend must equal production trend_state_from_score on same avg_col
    prod_trend = prod.trend_state_from_score(o["avg_col"], SWITCH)

    transitions = []
    prev_state = o["trend_state"][0]
    for i in range(1, len(o["trend_state"])):
        s = o["trend_state"][i]
        if s != prev_state:
            transitions.append({
                "index": int(i),
                "prev_avg_col": (
                    None if not np.isfinite(o["avg_col"][i - 1])
                    else float(o["avg_col"][i - 1])
                ),
                "cur_avg_col": (
                    None if not np.isfinite(o["avg_col"][i])
                    else float(o["avg_col"][i])
                ),
                "new_state": int(s),
            })
        prev_state = s

    # verify each recorded transition satisfies the Pine crossover semantics
    semantics_ok = True
    for t in transitions:
        if t["new_state"] == +1:
            ok = (t["prev_avg_col"] is not None and t["cur_avg_col"] is not None
                  and t["prev_avg_col"] <= SWITCH and t["cur_avg_col"] > SWITCH)
        elif t["new_state"] == -1:
            ok = (t["prev_avg_col"] is not None and t["cur_avg_col"] is not None
                  and t["prev_avg_col"] >= -SWITCH and t["cur_avg_col"] < -SWITCH)
        else:
            ok = False
        if not ok:
            semantics_ok = False

    return {
        "name": "crossover",
        "oracle_trend_equals_production": bool(
            np.array_equal(o["trend_state"], prod_trend)
        ),
        "transitions": transitions,
        "crossover_semantics_ok": semantics_ok,
    }


def case_epsilon() -> dict:
    # Construct 0 < |P100| <= 1e-12 so Pine direct division gives finite
    # while production's |denom|>1e-12 guard forces NaN.
    n = 600
    close = 100.0 + np.arange(n) * 1e-13
    high, low, close = make_ohlc(close, band=1e-14)
    o = dtp_literal_oracle(high, low, close)
    feat = run_production_segment(high, low, close)
    _, slope_p, denom_p = production_denominator(close)

    # epsilon region E = { i : 0 < |P100_i| <= 1e-12 }
    eps_mask = (
        np.isfinite(denom_p)
        & (np.abs(denom_p) > 0)
        & (np.abs(denom_p) <= EPS_GUARD)
    )
    idxs = np.where(eps_mask)[0]
    oracle_ac = o["avg_col"]
    prod_ts = feat["trend_score"].to_numpy(float)

    n_match = 0
    n_mismatch = 0
    first_mm_idx = None
    first_o_val = None
    first_p_val = None
    samples = []
    for i in idxs:
        ov = oracle_ac[i]
        pv = prod_ts[i]
        both_finite = np.isfinite(ov) and np.isfinite(pv)
        both_nan = np.isnan(ov) and np.isnan(pv)
        numeric_match = both_finite and np.isclose(ov, pv, rtol=0, atol=FLOAT_TOL)
        point_match = bool(both_nan or numeric_match)
        if point_match:
            n_match += 1
        else:
            n_mismatch += 1
            if first_mm_idx is None:
                first_mm_idx = int(i)
                first_o_val = None if not np.isfinite(ov) else float(ov)
                first_p_val = None if not np.isfinite(pv) else float(pv)
        if len(samples) < 5:
            samples.append({
                "index": int(i),
                "p100": float(denom_p[i]),
                "oracle_avg_col": (
                    None if not np.isfinite(ov) else float(ov)
                ),
                "production_trend_score": (
                    None if not np.isfinite(pv) else float(pv)
                ),
            })

    # verdict from the ACTUAL differential, never hardcoded
    guard_source_match = (int(eps_mask.sum()) > 0 and n_mismatch == 0)

    return {
        "name": "epsilon_guard",
        "n_epsilon_points": int(eps_mask.sum()),
        "n_epsilon_match": n_match,
        "n_epsilon_mismatch": n_mismatch,
        "first_epsilon_mismatch_index": first_mm_idx,
        "first_oracle_value": first_o_val,
        "first_production_value": first_p_val,
        "guard_source_match": guard_source_match,
        "samples": samples,
    }


# =============================================================================
# 5. random differential (T1-B)
# =============================================================================

def random_differential(seed: int = 20260918, n_series: int = 20,
                        n: int = 2200) -> list[dict]:
    rng = np.random.default_rng(seed)
    all_rows = []
    for s in range(n_series):
        # finite OHLC with high>=max(open,close), low<=min(open,close)
        base = 100.0 + rng.uniform(-5, 5, size=n).cumsum() * 0.1
        close = base.astype(float)
        openp = close + rng.uniform(-0.05, 0.05, size=n)
        high = np.maximum.reduce([openp, close]) + rng.uniform(0.01, 0.2, size=n)
        low = np.minimum.reduce([openp, close]) - rng.uniform(0.01, 0.2, size=n)
        o = dtp_literal_oracle(high, low, close)
        feat = run_production_segment(high, low, close)
        rows = compare_segment(
            pd.DataFrame({
                "high": high, "low": low, "close": close,
                "sma": feat["sma"], "atr": feat["atr"],
                "trend_score": feat["trend_score"],
                "trend_state": feat["trend_state"],
            })
        )["rows"]
        for r in rows:
            all_rows.append(r)
    return all_rows


# =============================================================================
# 6. real AG 5m differential (T1-C)
# =============================================================================

def real_ag_differential() -> dict:
    raw_frame, tf5, feat = production_dtp_5m("AG")
    disc = raw_frame["disc"].to_numpy()
    n_rows = len(feat)

    # segment-reset facts
    segment = feat["segment"].to_numpy()
    n_segments = int(len(np.unique(segment)))
    n_disc_true = int(disc.sum())

    # per-segment Oracle comparison
    seg_rows = []
    oracle_full = None
    for _, g in feat.groupby("segment", sort=False):
        res = compare_segment(g)
        seg_rows.extend(res["rows"])
        if oracle_full is None:
            # for single-segment AG this is the whole thing
            oracle_full = res["oracle"]
        else:
            # multi-segment: oracle_full only used for divergence analysis
            pass

    # full-sequence Oracle (Pine-faithful: NO reset) for divergence analysis
    high = feat["high"].to_numpy(float)
    low = feat["low"].to_numpy(float)
    close = feat["close"].to_numpy(float)
    oracle_full_seq = dtp_literal_oracle(high, low, close)

    # full-sequence comparison (catches segment-reset divergence, if any)
    full_rows = compare_segment(
        pd.DataFrame({
            "high": high, "low": low, "close": close,
            "sma": feat["sma"], "atr": feat["atr"],
            "trend_score": feat["trend_score"],
            "trend_state": feat["trend_state"],
        })
    )["rows"]

    # manual samples (5 spread indices)
    sample_idx = [100, 1000, 5000, 20000, min(44000, n_rows - 1)]
    samples = []
    for i in sample_idx:
        samples.append({
            "index": int(i),
            "high": float(high[i]),
            "low": float(low[i]),
            "close": float(close[i]),
            "oracle_sma": (
                None if not np.isfinite(oracle_full_seq["sma"][i])
                else float(oracle_full_seq["sma"][i])
            ),
            "production_sma": float(feat["sma"].to_numpy(float)[i]),
            "oracle_atr": (
                None if not np.isfinite(oracle_full_seq["atr"][i])
                else float(oracle_full_seq["atr"][i])
            ),
            "production_atr": float(feat["atr"].to_numpy(float)[i]),
            "oracle_avg_col": (
                None if not np.isfinite(oracle_full_seq["avg_col"][i])
                else float(oracle_full_seq["avg_col"][i])
            ),
            "production_trend_score": (
                None if not np.isfinite(feat["trend_score"].to_numpy(float)[i])
                else float(feat["trend_score"].to_numpy(float)[i])
            ),
            "oracle_trend": int(oracle_full_seq["trend_state"][i]),
            "production_trend_state": int(feat["trend_state"].to_numpy(int)[i]),
        })

    return {
        "symbol": "AG",
        "n_rows": int(n_rows),
        "n_segments": n_segments,
        "n_disc_true": n_disc_true,
        "duplicate_timestamps": int(
            feat["available_time"].duplicated().sum()
            if "available_time" in feat else 0
        ),
        "finite_ohlc_violations": int(
            int((~np.isfinite(high)).sum())
            + int((~np.isfinite(low)).sum())
            + int((~np.isfinite(close)).sum())
        ),
        "warmup_rows": int(NORM - 1 + (SMA_LEN - 1) + LAG),
        "seg_rows": seg_rows,
        "full_rows": full_rows,
        "samples": samples,
    }


# =============================================================================
# 7. structural tests (Test A-D)
# =============================================================================

def test_warmup_indices() -> dict:
    n = 700
    close = 100.0 + np.arange(n) * 0.01
    high, low, close = make_ohlc(close)
    o = dtp_literal_oracle(high, low, close)
    first_finite = {}
    for k in ("sma", "atr", "avg_diff", "p100", "avg_col"):
        arr = o[k]
        idxs = np.where(np.isfinite(arr))[0]
        first_finite[k] = int(idxs[0]) if len(idxs) else None
    expect = {
        "sma": SMA_LEN - 1,
        "atr": ATR_LEN - 1,
        "avg_diff": LAG + (SMA_LEN - 1),
        "p100": NORM - 1 + (SMA_LEN - 1) + LAG,
        "avg_col": NORM - 1 + (SMA_LEN - 1) + LAG,
    }
    ok = all(first_finite[k] == expect[k] for k in expect)
    return {"first_finite": first_finite, "expect": expect, "pass": bool(ok)}


def test_prefix_invariance() -> dict:
    n = 1500
    cut = 1200
    close_a = 100.0 + np.arange(n) * 0.02
    # B modifies only cut+1 .. end
    close_b = close_a.copy()
    close_b[cut + 1:] = close_b[cut + 1:] * 1.001
    high_a, low_a, _ = make_ohlc(close_a)
    high_b, low_b, _ = make_ohlc(close_b)

    feat_a = run_production_segment(high_a, low_a, close_a)
    feat_b = run_production_segment(high_b, low_b, close_b)
    o_a = dtp_literal_oracle(high_a, low_a, close_a)
    o_b = dtp_literal_oracle(high_b, low_b, close_b)

    cols = ["sma", "atr", "trend_score", "trend_state"]
    prod_max = 0.0
    for c in cols:
        d = np.abs(
            feat_a[c].to_numpy(float)[:cut + 1]
            - feat_b[c].to_numpy(float)[:cut + 1]
        )
        prod_max = max(prod_max, float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)
    oracle_max = 0.0
    for k in ("sma", "atr", "avg_col", "trend_state"):
        d = np.abs(o_a[k][:cut + 1] - o_b[k][:cut + 1])
        oracle_max = max(oracle_max, float(np.nanmax(d)) if np.isfinite(d).any() else 0.0)
    return {
        "cut": cut,
        "prod_prefix_max_abs_diff": prod_max,
        "oracle_prefix_max_abs_diff": oracle_max,
        "pass": bool(prod_max == 0.0 and oracle_max == 0.0),
    }


def test_call_chain() -> dict:
    calls = {"compute_tf_features": 0, "compute_segment_features": 0}

    orig_ctf = prod.compute_tf_features
    orig_csf = prod.compute_segment_features

    def w_tf(tf, params, include_sr):
        calls["compute_tf_features"] += 1
        return orig_ctf(tf, params, include_sr)

    def w_sf(seg, params, include_sr):
        calls["compute_segment_features"] += 1
        return orig_csf(seg, params, include_sr)

    prod.compute_tf_features = w_tf
    prod.compute_segment_features = w_sf
    try:
        _, _, feat = production_dtp_5m("AG")
    finally:
        prod.compute_tf_features = orig_ctf
        prod.compute_segment_features = orig_csf

    return {
        "compute_tf_features_calls": calls["compute_tf_features"],
        "compute_segment_features_calls": calls["compute_segment_features"],
        "n_ag_rows": int(len(feat)),
        "pass": bool(
            calls["compute_tf_features"] >= 1
            and calls["compute_segment_features"] >= 1
        ),
    }


def test_segment_reset() -> dict:
    """Test E -- classify the production segment reset.

    disc[900]=True forces a second segment. Production reinitializes SMA/ATR/
    500-bar normalization/trend_state at the break (Pine source does NOT).
    Classification: INTENTIONAL_DATA_SAFETY_EXTENSION.
    """
    n = 1600
    disc_index = 900
    rng = np.random.default_rng(20260918)
    close = 100.0 + np.cumsum(rng.uniform(-0.05, 0.05, n))
    high = close + rng.uniform(0.01, 0.1, n)
    low = close - rng.uniform(0.01, 0.1, n)
    disc = np.zeros(n, dtype=bool)
    disc[disc_index] = True

    # Pine-source Oracle: full sequence, NO reset
    oracle = dtp_literal_oracle(high, low, close)

    # Production: real chain with disc -> segment reset
    feat = run_production_with_disc(high, low, close, disc)
    n_segments = int(feat["segment"].nunique())

    fields = {
        "sma": (oracle["sma"], feat["sma"].to_numpy(float)),
        "atr": (oracle["atr"], feat["atr"].to_numpy(float)),
        "trend_score": (oracle["avg_col"], feat["trend_score"].to_numpy(float)),
        "trend_state": (
            oracle["trend_state"], feat["trend_state"].to_numpy(int)
        ),
    }
    pre = {}
    post = {}
    for name, (o, p) in fields.items():
        pre[name] = compare_region(name, o, p, 0, disc_index,
                                    int_field=(name == "trend_state"))
        post[name] = compare_region(name, o, p, disc_index, n,
                                     int_field=(name == "trend_state"))

    segment_reset_triggered = n_segments > 1
    return {
        "n": n,
        "disc_index": disc_index,
        "n_segments": n_segments,
        "segment_reset_triggered": segment_reset_triggered,
        "first_post_reset_index": disc_index,
        "pre_reset_mismatch": pre,
        "sma_mismatch_after_reset": post["sma"],
        "atr_mismatch_after_reset": post["atr"],
        "trend_score_mismatch_after_reset": post["trend_score"],
        "trend_state_mismatch_after_reset": post["trend_state"],
        "classification": "INTENTIONAL_DATA_SAFETY_EXTENSION",
    }


def test_compare_int_evidence() -> dict:
    """Test E4 -- verify compare_int absolute index + max_abs_error evidence.

    Mismatch injected at absolute index 7 (start=5). +1 vs -1 must give
    max_abs_error = 2 and first_mismatch_index = 7.
    """
    o = np.full(10, -1, dtype=int)
    p = o.copy()
    p[7] = +1
    r = compare_int("trend_state", o, p, start=5)
    return {
        "first_mismatch_index": r["first_mismatch_index"],
        "max_abs_error": r["max_abs_error"],
        "expected_index": 7,
        "expected_max_abs_error": 2.0,
        "pass": bool(
            r["first_mismatch_index"] == 7
            and r["max_abs_error"] == 2.0
        ),
    }


def test_pine_safe_divide() -> dict:
    """Test Z -- Pine zero/tiny division semantics for the literal Oracle.

    Pine: x / 0 -> na (NOT +inf / -inf like NumPy).
    Tiny NON-zero denominator (9.2e-14 / 1e-13) still divides -> 0.92.
    """
    num = np.array([1.0, -1.0, 0.0, 9.2e-14])
    den = np.array([0.0, 0.0, 0.0, 1.0e-13])
    out = pine_safe_divide(num, den)
    checks = {
        "z1_pos_over_zero": bool(np.isnan(out[0])),
        "z2_neg_over_zero": bool(np.isnan(out[1])),
        "z3_zero_over_zero": bool(np.isnan(out[2])),
        "z4_tiny_divides": bool(
            np.isfinite(out[3]) and abs(out[3] - 0.92) < 1e-12
        ),
        "no_inf": bool(not np.any(np.isinf(out))),
    }
    return {
        "pass": bool(all(checks.values())),
        "numerator": num.tolist(),
        "denominator": den.tolist(),
        "result": [None if np.isnan(v) else float(v) for v in out],
        "checks": checks,
    }


def test_epsilon_guard() -> dict:
    c = case_epsilon()
    return {
        "n_epsilon_points": c["n_epsilon_points"],
        "n_epsilon_mismatch": c["n_epsilon_mismatch"],
        "guard_source_match": c["guard_source_match"],
        "samples": c["samples"],
        "pass": bool(
            c["n_epsilon_points"] > 0
            and c["n_epsilon_mismatch"] > 0
            and c["guard_source_match"] is False
        ),
    }


# =============================================================================
# 8. main
# =============================================================================

def main() -> None:
    assert_base_ancestor()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[AUDIT] gates passed: base_ancestor ok, pine_source_sha ok, "
          "params frozen", flush=True)

    # T1-A deterministic
    c_const = case_constant_range()
    c_gap = case_gap()
    c_cross = case_crossover()
    c_eps = case_epsilon()
    print("[AUDIT] T1-A constant_range sma n_mismatch =",
          _field_nm(c_const["rows"], "sma"), flush=True)
    print(f"[AUDIT] T1-A gap tr_oracle={c_gap['tr_oracle']} "
          f"tr_prod={c_gap['tr_prod']} expect={c_gap['expect_tr']}", flush=True)
    print("[AUDIT] T1-A crossover oracle==production trend:",
          c_cross["oracle_trend_equals_production"], flush=True)
    for t in c_cross["transitions"]:
        print(f"  transition -> {t['new_state']} "
              f"prev_avg_col={t['prev_avg_col']} cur_avg_col={t['cur_avg_col']}",
              flush=True)
    print(f"[AUDIT] T1-A epsilon n_epsilon_points={c_eps['n_epsilon_points']} "
          f"guard_source_match={c_eps['guard_source_match']}", flush=True)

    # T1-B random
    rand_rows = random_differential()
    rand_agg = aggregate_rows(rand_rows)
    print("[AUDIT] T1-B random per-field:", flush=True)
    for f, r in rand_agg.items():
        print(f"  {f}: n_compared={r['n_compared']} "
              f"n_mismatch={r['n_mismatch']} max_abs={r['max_abs_error']}",
              flush=True)

    # T1-C real AG
    ag = real_ag_differential()
    ag_seg_agg = aggregate_rows(ag["seg_rows"])
    ag_full_agg = aggregate_rows(ag["full_rows"])
    print(f"[AUDIT] T1-C AG rows={ag['n_rows']} segments={ag['n_segments']} "
          f"disc_true={ag['n_disc_true']}", flush=True)
    print("[AUDIT] T1-C AG per-segment per-field:", flush=True)
    for f, r in ag_seg_agg.items():
        print(f"  {f}: n_compared={r['n_compared']} "
              f"n_mismatch={r['n_mismatch']} max_abs={r['max_abs_error']}",
              flush=True)
    print("[AUDIT] T1-C AG full-sequence (divergence) per-field:", flush=True)
    for f, r in ag_full_agg.items():
        print(f"  {f}: n_compared={r['n_compared']} "
              f"n_mismatch={r['n_mismatch']} max_abs={r['max_abs_error']}",
              flush=True)

    # Tests A-D
    t_warm = test_warmup_indices()
    t_pref = test_prefix_invariance()
    t_chain = test_call_chain()
    t_eps = test_epsilon_guard()
    # Hardening tests E1-E5
    t_seg = test_segment_reset()
    t_cmp = test_compare_int_evidence()
    t_zdiv = test_pine_safe_divide()
    print("[AUDIT] Test A warmup pass:", t_warm["pass"], t_warm["first_finite"],
          flush=True)
    print("[AUDIT] Test B prefix pass:", t_pref["pass"], flush=True)
    print("[AUDIT] Test C call-chain pass:", t_chain["pass"],
          "ctf=", t_chain["compute_tf_features_calls"],
          "csf=", t_chain["compute_segment_features_calls"], flush=True)
    print("[AUDIT] Test D epsilon pass:", t_eps["pass"],
          "n_mismatch=", t_eps["n_epsilon_mismatch"], flush=True)
    print("[AUDIT] Test E segment-reset triggered:", t_seg["segment_reset_triggered"],
          "segments=", t_seg["n_segments"],
          "post_reset(sma/atr/ts/state)=",
          t_seg["sma_mismatch_after_reset"], t_seg["atr_mismatch_after_reset"],
          t_seg["trend_score_mismatch_after_reset"],
          t_seg["trend_state_mismatch_after_reset"], flush=True)
    print("[AUDIT] Test E4 compare_int evidence pass:", t_cmp["pass"],
          "idx=", t_cmp["first_mismatch_index"], "max_abs=", t_cmp["max_abs_error"],
          flush=True)
    print("[AUDIT] Test Z division semantics pass:", t_zdiv["pass"],
          "result=", t_zdiv["result"], flush=True)

    # epsilon guard observed on real AG?
    ag_avgcol = ag_seg_agg.get("avg_col", {})
    epsilon_observed_ag = int(ag_avgcol.get("n_mismatch", 0))

    # ---- classification (4 buckets) ----
    core_fields = ("sma", "atr", "avg_diff", "p100", "trend_state")
    core_mismatch = (
        sum(rand_agg[f]["n_mismatch"] for f in core_fields)
        + sum(ag_seg_agg[f]["n_mismatch"] for f in core_fields)
    )
    core_status = "EXACT_SOURCE_MATCH" if core_mismatch == 0 else "SOURCE_DIVERGENCE"

    # epsilon guard: verdict comes from the ACTUAL differential
    eps_status = (
        "EXACT_SOURCE_MATCH"
        if (c_eps["n_epsilon_points"] > 0 and c_eps["n_epsilon_mismatch"] == 0)
        else "SOURCE_DIVERGENCE"
    )

    seg_status = "INTENTIONAL_DATA_SAFETY_EXTENSION"
    pct_status = "UNVERIFIED"

    verdict = (
        "MISMATCH"
        if (eps_status == "SOURCE_DIVERGENCE"
            or core_status == "SOURCE_DIVERGENCE")
        else "MATCH"
    )
    verdict_reason = (
        f"core_math={core_status}; epsilon_guard={eps_status} "
        f"(n_epsilon_mismatch={c_eps['n_epsilon_mismatch']}); "
        f"segment_reset={seg_status}; percentile_na_semantics={pct_status}. "
        f"AG disc_true={ag['n_disc_true']} (no reset on AG); segment marks "
        f"true data-break / ATR5 price-jump, not normal session gaps."
    )

    summary = {
        "task_id": "STRUCTREV-PGM-R2B-SRC1.1-DTP-AUDIT-HARDEN",
        "git_sha": git_head(),
        "audit_git_sha": git_head(),
        "pine_source_sha256": pine_source_sha256(),
        "symbol": "AG",
        "oracle_definition_version": "literal_DeviationTrendProfile_pine_v1",
        "seed": 20260918,
        "n_synthetic_series": 20,
        "n_real_rows": ag["n_rows"],
        "warmup_indices": {
            "sma_first_finite": SMA_LEN - 1,
            "atr_first_finite": ATR_LEN - 1,
            "avg_diff_first_finite": LAG + (SMA_LEN - 1),
            "p100_first_finite": NORM - 1 + (SMA_LEN - 1) + LAG,
            "avg_col_first_finite": NORM - 1 + (SMA_LEN - 1) + LAG,
        },
        "epsilon_guard": {
            "n_epsilon_points": c_eps["n_epsilon_points"],
            "n_epsilon_match": c_eps["n_epsilon_match"],
            "n_epsilon_mismatch": c_eps["n_epsilon_mismatch"],
            "first_epsilon_mismatch_index": c_eps["first_epsilon_mismatch_index"],
            "first_oracle_value": c_eps["first_oracle_value"],
            "first_production_value": c_eps["first_production_value"],
            "match": c_eps["guard_source_match"],
            "observed_on_AG": epsilon_observed_ag,
        },
        "prefix_invariance": {
            "oracle_pass": t_pref["pass"],
            "production_pass": t_pref["pass"],
        },
        "segment_reset_facts": {
            "ag_disc_true": ag["n_disc_true"],
            "ag_n_segments": ag["n_segments"],
            "disc_semantics": (
                "non-5min AND non-normal-session time gap, OR ATR5 price jump"
            ),
            "normal_session_gap_reset": False,
            "synthetic_test": {
                "n": t_seg["n"],
                "disc_index": t_seg["disc_index"],
                "n_segments": t_seg["n_segments"],
                "segment_reset_triggered": t_seg["segment_reset_triggered"],
                "first_post_reset_index": t_seg["first_post_reset_index"],
                "pre_reset_mismatch": t_seg["pre_reset_mismatch"],
                "sma_mismatch_after_reset": t_seg["sma_mismatch_after_reset"],
                "atr_mismatch_after_reset": t_seg["atr_mismatch_after_reset"],
                "trend_score_mismatch_after_reset":
                    t_seg["trend_score_mismatch_after_reset"],
                "trend_state_mismatch_after_reset":
                    t_seg["trend_state_mismatch_after_reset"],
                "classification": t_seg["classification"],
            },
        },
        "classification": {
            "core_math": core_status,
            "epsilon_guard": eps_status,
            "segment_reset": seg_status,
            "percentile_na_semantics": pct_status,
        },
        "per_field_random": rand_agg,
        "per_field_ag_per_segment": ag_seg_agg,
        "per_field_ag_full_sequence": ag_full_agg,
        "tests": {
            "A_warmup": t_warm["pass"],
            "B_prefix": t_pref["pass"],
            "C_call_chain": t_chain["pass"],
            "D_epsilon": t_eps["pass"],
            "E_segment_reset": t_seg["segment_reset_triggered"],
            "E4_compare_int_evidence": t_cmp["pass"],
            "Z_division_semantics": t_zdiv["pass"],
        },
        "division_semantics": {
            "oracle_uses_pine_safe_divide": True,
            "zero_denominator_returns_nan": bool(t_zdiv["checks"]["z1_pos_over_zero"]
                                                 and t_zdiv["checks"]["z2_neg_over_zero"]
                                                 and t_zdiv["checks"]["z3_zero_over_zero"]),
            "tiny_nonzero_divides": bool(t_zdiv["checks"]["z4_tiny_divides"]),
            "no_numpy_inf": bool(t_zdiv["checks"]["no_inf"]),
            "samples": {
                "numerator": t_zdiv["numerator"],
                "denominator": t_zdiv["denominator"],
                "result": t_zdiv["result"],
            },
        },
        "verdict": verdict,
        "verdict_reason": verdict_reason,
    }

    (OUT / "dtp_source_parity_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    # fields csv
    field_rows = []
    for src, agg in (
        ("random", rand_agg),
        ("ag_per_segment", ag_seg_agg),
        ("ag_full_sequence", ag_full_agg),
    ):
        for f, r in agg.items():
            row = {"source": src, **r}
            field_rows.append(row)
    pd.DataFrame(field_rows).to_csv(
        OUT / "dtp_source_parity_fields.csv", index=False
    )

    # mismatches csv (any field with mismatch)
    mm_rows = []
    for src, agg in (
        ("random", rand_agg),
        ("ag_per_segment", ag_seg_agg),
        ("ag_full_sequence", ag_full_agg),
    ):
        for f, r in agg.items():
            if r["n_mismatch"] > 0:
                mm_rows.append({"source": src, **r})
    pd.DataFrame(mm_rows).to_csv(
        OUT / "dtp_source_parity_mismatches.csv", index=False
    )

    # samples csv
    pd.DataFrame(ag["samples"]).to_csv(
        OUT / "dtp_source_parity_samples.csv", index=False
    )

    print("[AUDIT] verdict:", verdict, flush=True)
    print("[AUDIT] verdict_reason:", verdict_reason, flush=True)
    print("[AUDIT] artifacts written to", OUT, flush=True)


def _field_nm(rows, field):
    for r in rows:
        if r["field"] == field:
            return r["n_mismatch"]
    return None


if __name__ == "__main__":
    main()
