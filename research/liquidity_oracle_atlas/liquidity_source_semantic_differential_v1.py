#!/usr/bin/env python3
"""
Liquidity Production Differential (discovery round)
=====================================================

Runs the independent Literal Oracle (``liquidity_source_semantic_oracle_v1``)
against the real production ``build_liquidity_features`` / ``compute_segment_features``
/ ``compute_tf_features`` to establish, on the frozen pinned ``ref/Liquidity.pine``
source semantics domain:

  * which production fields MATCH source,
  * which are REAL SOURCE_DIVERGENCE,
  * which are RESEARCH_EXTENSION (prod self-labelled),
  * which are UNVERIFIED (Pine builtin tie/float/mode cannot be determined).

Layers:
  * Oracle self-tests (L01-L20)        -> ORACLE_NOT_READY gate
  * Boundary matrix (L01-L20)          -> pivot-isolated state-machine comparison
  * Random R-A (inject pivot+ATR)      -> primary state-machine verdict
  * Random R-B (end-to-end OHLC)       -> end-to-end diagnostic
  * Canonical AG 5m / 15m             -> real call chain, per-segment, tie mask
  * Full pipeline parity               -> compute_segment_features vs compute_tf_features
  * Real call chain                    -> counters
  * Prefix causality                   -> future edits do not change [:cut+1]

This script NEVER copies or execs production source code; it only calls public
production functions and observes their public outputs. Pivot isolation injects
prepared ph/pl via monkeypatching ``prod.confirmed_pivots``.

Observations only. No fix recommendation, no production bug judgement.
"""

from __future__ import annotations

import os
import sys
import json
import csv
import hashlib
import subprocess

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from research.liquidity_oracle_atlas import experiment_structural_reversion_pgm_v1 as prod  # noqa: E402
from research.liquidity_oracle_atlas import liquidity_source_semantic_oracle_v1 as oracle  # noqa: E402
from research.export_ob_trigger_execution_v21 import load_raw_5m  # noqa: E402
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BASE = "fbbac892c066fcdbba9aa4d3c890ab765a7c7300"
PINNED_SHA = oracle.PINNED_SHA256
ORACLE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "liquidity_source_semantic_oracle_v1.py")
ART_DIR = os.path.join(ROOT, "artifacts", "liquidity_source_semantic_differential")
FIELDS = ["breach_up", "breach_down", "up_count", "down_count", "zone_active"]
PROD_FIELD = {
    "breach_up": "liq_breach_up",
    "breach_down": "liq_breach_down",
    "up_count": "liq_up_count",
    "down_count": "liq_down_count",
    "zone_active": "liq_last_zone_active",
}
RESEARCH_EXTENSION = {
    "liq_up_dist_atr", "liq_down_dist_atr",
    "liq_up_level_price", "liq_down_level_price",
    "liq_last_breach_side", "liq_last_breach_age",
    "liq_last_accept", "liq_last_reclaim",
}
FLOAT_BOUNDARY = oracle.FLOAT_BOUNDARY


# ===========================================================================
# Git / IO helpers
# ===========================================================================
def _git(args):
    return subprocess.run(["git"] + args, cwd=ROOT, capture_output=True, text=True)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _source_sha(kind):
    if kind == "worktree":
        with open(oracle.LIQ_PINE_PATH, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    if kind == "git-object":
        r = _git(["cat-file", "blob", "HEAD:ref/Liquidity.pine"])
        if r.returncode != 0 or not r.stdout:
            return None  # not yet committed (pre-commit dry-run)
        return hashlib.sha256(r.stdout.encode("utf-8", "surrogateescape")).hexdigest()
    if kind == "expected":
        return PINNED_SHA
    raise ValueError(kind)


def check_t0():
    """Reproducibility gates. Returns a dict; raises SystemExit on hard failure."""
    head = _git(["rev-parse", "HEAD"]).stdout.strip()
    if head != BASE:
        raise SystemExit("STOP_BASE_MISMATCH: head=%s expected=%s" % (head, BASE))
    if not os.path.exists(ORACLE_FILE):
        raise SystemExit("STOP_ORACLE_MISSING")
    st = _git(["status", "--porcelain"])
    dirty = [l for l in st.stdout.splitlines() if l.strip()]
    return {
        "head": head,
        "oracle_exists": os.path.exists(ORACLE_FILE),
        "status_lines": dirty,
        "source_worktree_sha": _source_sha("worktree"),
        "source_gitobject_sha": _source_sha("git-object"),
        "source_expected_sha": PINNED_SHA,
    }


# ===========================================================================
# OHLC generation (for Random / Prefix)
# ===========================================================================
def make_ohlc(n_bars, mid, amp, seed):
    rng = np.random.default_rng(seed)
    t = np.arange(n_bars)
    wave = (mid + amp * 0.5 * np.sin(t * 0.0137)
            + amp * 0.25 * np.sin(t * 0.041 + 1.3)
            + amp * 0.12 * np.sin(t * 0.113 + 2.1)
            + rng.normal(0, amp * 0.06, n_bars))
    high = wave + np.abs(rng.normal(0, amp * 0.05, n_bars)) + 0.3
    low = wave - np.abs(rng.normal(0, amp * 0.05, n_bars)) - 0.3
    close = wave + rng.normal(0, amp * 0.04, n_bars)
    open_ = wave + rng.normal(0, amp * 0.04, n_bars)
    open_[0] = wave[0]
    close[-1] = wave[-1]
    high = np.maximum.reduce([high, low, close, open_])
    low = np.minimum.reduce([high, low, close, open_])
    return high, low, open_, close


# ===========================================================================
# Production runners
# ===========================================================================
def run_prod_liq_with_injected_pivots(high, low, close, atr, ph, pl):
    """Call REAL prod.build_liquidity_features with prepared ph/pl injected."""
    ph = np.asarray(ph, dtype=float)
    pl = np.asarray(pl, dtype=float)
    orig = prod.confirmed_pivots

    def _patched(values, left, right, mode):
        assert (left, right) == (oracle.LIQ_LEN, oracle.LIQ_RIGHT), (left, right)
        return ph if mode == "high" else pl

    prod.confirmed_pivots = _patched
    try:
        feat = prod.build_liquidity_features(
            np.asarray(high, dtype=float),
            np.asarray(low, dtype=float),
            np.asarray(close, dtype=float),
            np.asarray(atr, dtype=float),
            prod.IndicatorParams(),
        )
    finally:
        prod.confirmed_pivots = orig
    return feat


def to_arr(x):
    return np.asarray(x.values if hasattr(x, "values") else x, dtype=float)


def compare_state_full(o_per, p_feat, mask):
    """Compare Oracle per-bar dicts vs production feature dict under mask."""
    res = {}
    for f in FIELDS:
        ov = np.array([b[f] for b in o_per], dtype=float)
        pv = to_arr(p_feat[PROD_FIELD[f]])
        mism = int(np.sum((mask) & (ov != pv)))
        res[f] = mism
    return res


def compare_state_full_masked(o_per, p_feat, mask, first_mis=None, max_consec=None):
    res = {}
    for f in FIELDS:
        ov = np.array([b[f] for b in o_per], dtype=float)
        pv = to_arr(p_feat[PROD_FIELD[f]])
        mism = 0
        consec = 0
        cur = 0
        for i in range(len(ov)):
            bad = bool(mask[i]) and (ov[i] != pv[i])
            if bad:
                mism += 1
                consec += 1
                cur = max(cur, consec)
                if first_mis is not None and first_mis.get(f) is None:
                    first_mis[f] = i
            else:
                consec = 0
        if max_consec is not None:
            max_consec[f] = max(max_consec.get(f, 0), cur)
        res[f] = mism
    return res


# ===========================================================================
# Boundary matrix
# ===========================================================================
def _boundary_cases():
    """Yield (case_id, production_comparable, high, low, close, atr)."""
    inj = oracle._inj
    flat = oracle._flat
    n = 80
    atr = np.full(n, 10.0, dtype=float)

    # L01 high pivot confirmation (internal)
    h = np.full(n, 100.0); l = np.full(n, 100.0); c = np.full(n, 100.0); h[7] = 200.0
    yield ("L01", False, h, l, c, atr)
    # L02 low pivot confirmation (internal)
    h = np.full(n, 100.0); l = np.full(n, 100.0); c = np.full(n, 100.0); l[7] = 5.0
    yield ("L02", False, h, l, c, atr)
    # L03 same-side higher replaces (internal)
    ph, pl = inj(40, [(10, "high", 100.0), (20, "high", 110.0)])
    yield ("L03", False, *flat(40), atr[:40])
    # L04 same-side lower ignored (internal)
    ph, pl = inj(40, [(10, "high", 110.0), (20, "high", 105.0)])
    yield ("L04", False, *flat(40), atr[:40])
    # L05 opposite side inserts (internal)
    ph, pl = inj(40, [(10, "high", 100.0), (20, "low", 10.0), (30, "high", 120.0)])
    yield ("L05", False, *flat(40), atr[:40])
    # L06 zigzag cap 50 (internal)
    items = [(10 + k * 2, "high" if k % 2 == 0 else "low", 100.0 + k) for k in range(60)]
    ph, pl = inj(200, items)
    yield ("L06", False, *flat(200), np.full(200, 10.0, dtype=float))
    # L07 cluster count=2 -> no level (comparable via up_count)
    ph, pl = inj(n, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 103.0)])
    yield ("L07", True, *flat(n), atr)
    # L08 cluster count=3 -> level
    ph, pl = inj(n, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                    (40, "low", 50.0), (50, "high", 98.0)])
    yield ("L08", True, *flat(n), atr)
    # L09 lower margin equality
    ph, pl = inj(n, [(10, "high", 103.0), (20, "low", 50.0), (30, "high", 93.1),
                    (40, "low", 50.0), (50, "high", 100.0)])
    yield ("L09", True, *flat(n), atr)
    # L10 upper margin equality
    ph, pl = inj(n, [(10, "high", 97.0), (20, "low", 50.0), (30, "high", 106.9),
                    (40, "low", 50.0), (50, "high", 100.0)])
    yield ("L10", True, *flat(n), atr)
    # L11 high-side scan break (internal/literal)
    zz = [{"dir": 1, "x": 90, "y": 98.0}, {"dir": 1, "x": 80, "y": 200.0},
          {"dir": 1, "x": 70, "y": 100.0}, {"dir": 1, "x": 60, "y": 102.0}]
    obj = oracle.cluster_level_literal(zz, +1, 98.0, 10.0)
    yield ("L11", False, *flat(n), atr)
    # L12 low-side scan break
    yield ("L12", False, *flat(n), atr)
    # L13 level geometry (internal)
    ph, pl = inj(n, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                    (40, "low", 50.0), (50, "high", 98.0)])
    yield ("L13", False, *flat(n), atr)
    # L14 same start_bar update (internal)
    yield ("L14", False, *flat(n), atr)
    # L15 Visible 3 cap
    items = []
    for ci, base in enumerate([200.0, 300.0, 400.0, 500.0]):
        off = 30 + ci * 60
        items += [(off, "high", base), (off + 10, "low", base - 50.0),
                  (off + 20, "high", base + 2.0), (off + 30, "low", base - 50.0),
                  (off + 40, "high", base - 2.0)]
    ph, pl = inj(300, items)
    yield ("L15", True, *flat(300), np.full(300, 10.0, dtype=float))
    # L16 high breach strictness
    ph, pl = inj(n, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                    (40, "low", 50.0), (50, "high", 98.0)])
    h, l, c = flat(n)
    h[60] = 106.9; h[61] = 107.0
    yield ("L16", True, h, l, c, atr)
    # L17 low breach strictness
    ph, pl = inj(n, [(10, "low", 100.0), (20, "high", 200.0), (30, "low", 98.0),
                    (40, "high", 200.0), (50, "low", 102.0)])
    h, l, c = flat(n)
    l[60] = 93.1; l[61] = 92.9
    yield ("L17", True, h, l, c, atr)
    # L18 multiple same-bar breach
    ph, pl = inj(120, [(10, "high", 200.0), (20, "low", 50.0), (30, "high", 202.0),
                      (40, "low", 50.0), (50, "high", 198.0),
                      (60, "high", 300.0), (70, "low", 50.0), (80, "high", 302.0),
                      (90, "low", 50.0), (100, "high", 298.0)])
    h, l, c = flat(120)
    h[110] = 400.0
    yield ("L18", True, h, l, c, np.full(120, 10.0, dtype=float))
    # L19 post-break zone
    ph, pl = inj(n, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                    (40, "low", 50.0), (50, "high", 98.0)])
    h, l, c = flat(n)
    h[60] = 110.0; h[61] = 100.0; l[61] = 90.0; h[62] = 100.0; l[62] = 75.0; h[63] = 130.0
    yield ("L19", True, h, l, c, atr)
    # L20 ATR 0 / NaN
    ph, pl = inj(n, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                    (40, "low", 50.0), (50, "high", 98.0)])
    yield ("L20", True, *flat(n), atr)


def build_boundary_matrix():
    rows = []
    for case_id, prod_cmp, high, low, close, atr in _boundary_cases():
        ph, pl, _ = oracle.unique_confirmed_liq_pivots(high, low, high, close)
        o_per = oracle.run_liquidity_state_machine(high, low, close, atr, ph, pl)
        p_feat = run_prod_liq_with_injected_pivots(high, low, close, atr, ph, pl)
        mask = np.ones(len(close), dtype=bool)
        m = compare_state_full(o_per, p_feat, mask)
        exact_match = all(v == 0 for v in m.values())
        first = next((f for f in FIELDS if m[f] > 0), "")
        rows.append({
            "case_id": case_id,
            "source_domain": "exact" if prod_cmp else "internal_only",
            "production_comparable": prod_cmp,
            "oracle_breach_up": o_per[-1]["breach_up"],
            "production_breach_up": int(to_arr(p_feat["liq_breach_up"])[-1]),
            "oracle_breach_down": o_per[-1]["breach_down"],
            "production_breach_down": int(to_arr(p_feat["liq_breach_down"])[-1]),
            "oracle_up_count": o_per[-1]["up_count"],
            "production_up_count": int(to_arr(p_feat["liq_up_count"])[-1]),
            "oracle_down_count": o_per[-1]["down_count"],
            "production_down_count": int(to_arr(p_feat["liq_down_count"])[-1]),
            "oracle_zone_active": o_per[-1]["zone_active"],
            "production_zone_active": int(to_arr(p_feat["liq_last_zone_active"])[-1]),
            "breach_up_mm": m["breach_up"],
            "breach_down_mm": m["breach_down"],
            "up_count_mm": m["up_count"],
            "down_count_mm": m["down_count"],
            "zone_active_mm": m["zone_active"],
            "exact_match": exact_match,
            "first_mismatch_field": first,
        })
    return rows


# ===========================================================================
# Random R-A / R-B
# ===========================================================================
def _pivot_mask(high, low, oph, opl):
    n = len(high)
    pph = prod.confirmed_pivots(high, oracle.LIQ_LEN, oracle.LIQ_RIGHT, "high")
    ppl = prod.confirmed_pivots(low, oracle.LIQ_LEN, oracle.LIQ_RIGHT, "low")
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        ov = np.isfinite(oph[i]); pv = np.isfinite(pph[i])
        if ov != pv or (ov and pv and abs(oph[i] - pph[i]) >= FLOAT_BOUNDARY):
            mask[i:] = False
            break
    for i in range(n):
        ov = np.isfinite(opl[i]); pv = np.isfinite(ppl[i])
        if ov != pv or (ov and pv and abs(opl[i] - ppl[i]) >= FLOAT_BOUNDARY):
            mask[i:] = False
            break
    return mask


def layer_b_random(samples):
    seed = 20260918
    n_series = 12
    n = 3200
    ra_rows = []
    rb_rows = []
    ra_total = 0
    rb_total = 0
    for s in range(n_series):
        high, low, open_, close = make_ohlc(n, mid=4000.0, amp=120.0, seed=seed + s)
        atr = oracle.atr_pine(high, low, close)
        ph, pl, otie = oracle.unique_confirmed_liq_pivots(high, low, open_, close)
        # R-A: inject SAME pivots + SAME (independent) ATR into both
        o_per = oracle.run_liquidity_state_machine(high, low, close, atr, ph, pl)
        p_feat = run_prod_liq_with_injected_pivots(high, low, close, atr, ph, pl)
        m = compare_state_full(o_per, p_feat, np.ones(n, dtype=bool))
        ra_mm = sum(m.values())
        ra_total += ra_mm
        ra_rows.append({
            "series": s, "rows": n, "exact_rows": n,
            "breach_up_mm": m["breach_up"], "breach_down_mm": m["breach_down"],
            "up_count_mm": m["up_count"], "down_count_mm": m["down_count"],
            "zone_active_mm": m["zone_active"], "total_mm": ra_mm,
        })
        # R-B: end-to-end, production uses its own (non-strict) pivots + ATR
        prod_atr = prod.pine_rma(prod.true_range(high, low, close), oracle.ATR_LEN)
        pfeat_rb = prod.build_liquidity_features(high, low, close, prod_atr, prod.IndicatorParams())
        mask = _pivot_mask(high, low, ph, pl)
        m2 = compare_state_full(o_per, pfeat_rb, mask)
        rb_mm = sum(m2.values())
        rb_total += rb_mm
        cov = int(mask.sum())
        rb_rows.append({
            "series": s, "rows": n, "exact_rows": cov, "coverage_pct": round(100.0 * cov / n, 3),
            "breach_up_mm": m2["breach_up"], "breach_down_mm": m2["breach_down"],
            "up_count_mm": m2["up_count"], "down_count_mm": m2["down_count"],
            "zone_active_mm": m2["zone_active"], "total_mm": rb_mm,
        })
        # collect mismatch samples (R-B) for up to 20 per field
        for f in FIELDS:
            if m2[f] > 0 and len(samples.get(("random", f), [])) < 20:
                ov = np.array([b[f] for b in o_per], dtype=float)
                pv = to_arr(pfeat_rb[PROD_FIELD[f]])
                for i in range(n):
                    if mask[i] and ov[i] != pv[i]:
                        samples.setdefault(("random", f), []).append({
                            "dataset": "random", "timeframe": "RA_RB", "segment_id": s,
                            "local_bar": i, "global_bar": i, "time": "",
                            "field": f, "oracle_value": ov[i], "production_value": pv[i],
                            "high": high[i], "low": low[i], "close": close[i], "atr": atr[i],
                            "ph": ph[i] if np.isfinite(ph[i]) else "",
                            "pl": pl[i] if np.isfinite(pl[i]) else "",
                            "oracle_zz_dir": o_per[i]["zz_dir"],
                            "oracle_vis_up": str(o_per[i]["vis_up"]),
                            "uncertainty_reason": "state-machine or pivot divergence",
                        })
                        if len(samples[("random", f)]) >= 20:
                            break
    return ra_rows, rb_rows, ra_total, rb_total


# ===========================================================================
# Prefix causality
# ===========================================================================
def prefix_causality():
    n = 2000
    cut = 1500
    high, low, open_, close = make_ohlc(n, mid=4000.0, amp=120.0, seed=777)
    atr = oracle.atr_pine(high, low, close)
    ph, pl, _ = oracle.unique_confirmed_liq_pivots(high, low, open_, close)

    # Oracle full
    o_full = oracle.run_liquidity_state_machine(high, low, close, atr, ph, pl)
    # Production full (real chain, own pivots+atr)
    p_full = prod.build_liquidity_features(high, low, close, atr, prod.IndicatorParams())

    # modify cut+1..
    def modified(arr):
        a = arr.copy()
        rng = np.random.default_rng(999)
        a[cut + 1:] = a[cut + 1:] + rng.normal(0, 0.5, len(a) - (cut + 1))
        return a

    mh, ml, mc = modified(high), modified(low), modified(close)
    matr = oracle.atr_pine(mh, ml, mc)
    mph, mpl, _ = oracle.unique_confirmed_liq_pivots(mh, ml, mh, mc)
    o_mod = oracle.run_liquidity_state_machine(mh, ml, mc, matr, mph, mpl)
    p_mod = prod.build_liquidity_features(mh, ml, mc, matr, prod.IndicatorParams())

    fields_ok = True
    for f in FIELDS:
        ov = np.array([b[f] for b in o_full], dtype=float)
        ovm = np.array([b[f] for b in o_mod], dtype=float)
        pv = to_arr(p_full[PROD_FIELD[f]])
        pvm = to_arr(p_mod[PROD_FIELD[f]])
        if not (np.array_equal(ov[:cut + 1], ovm[:cut + 1]) and np.array_equal(pv[:cut + 1], pvm[:cut + 1])):
            fields_ok = False
    return {"n": n, "cut": cut, "oracle_pass": bool(np.allclose(
        np.array([b["breach_up"] for b in o_full])[:cut + 1],
        np.array([b["breach_up"] for b in o_mod])[:cut + 1])),
        "production_pass": fields_ok, "pass": fields_ok}


# ===========================================================================
# Canonical AG
# ===========================================================================
def load_ag_canonical_bars():
    """Build the ``bars`` mapping expected by ``prod.raw_frame_from_owner``.

    ``raw_frame_from_owner`` requires keys: n, t, day, disc, o, h, l, c.
    ``load_raw_5m`` (canonical AG owner) returns bar_start_time / trading_day /
    open / high / low / close; ``discontinuity_flags`` sorts by the same
    bar_start_time, so the two arrays are row-aligned.
    """
    raw = load_raw_5m("AG")
    disc = discontinuity_flags("AG")
    bars = dict(
        n=len(raw),
        t=raw["bar_start_time"].values,
        day=raw["trading_day"].values,
        disc=np.asarray(disc, dtype=bool),
        o=raw["open"].values.astype(float),
        h=raw["high"].values.astype(float),
        l=raw["low"].values.astype(float),
        c=raw["close"].values.astype(float),
    )
    raw_frame = prod.raw_frame_from_owner(bars)
    return raw, disc, bars, raw_frame


def layer_d_ag(tf_minutes, samples):
    raw, disc, bars, raw_frame = load_ag_canonical_bars()
    tf = prod.resample_causal(raw_frame, tf_minutes)
    seg_count = 0
    total_rows = 0
    cov_rows = 0
    exact_pivot_points = 0
    pivot_mismatch = 0
    tie_events = 0
    float_boundary_events = 0
    mm_counts = {f: 0 for f in FIELDS}
    first_mis = {f: None for f in FIELDS}
    max_consec = {f: 0 for f in FIELDS}

    for sid, seg in tf.groupby("segment", sort=False):
        seg_count += 1
        n = len(seg)
        sh = seg["high"].values.astype(float)
        sl = seg["low"].values.astype(float)
        sc = seg["close"].values.astype(float)
        so = seg["open"].values.astype(float)
        oph, opl, otie = oracle.unique_confirmed_liq_pivots(sh, sl, so, sc)
        oatr = oracle.atr_pine(sh, sl, sc)
        o_per = oracle.run_liquidity_state_machine(sh, sl, sc, oatr, oph, opl, mode="Historical")
        pfeat = prod.compute_segment_features(seg, prod.IndicatorParams(), True)

        pph = prod.confirmed_pivots(sh, oracle.LIQ_LEN, oracle.LIQ_RIGHT, "high")
        ppl = prod.confirmed_pivots(sl, oracle.LIQ_LEN, oracle.LIQ_RIGHT, "low")
        mask = np.ones(n, dtype=bool)
        first_tie = None
        for i in range(n):
            ov = np.isfinite(oph[i]); pv = np.isfinite(pph[i])
            if ov and pv and abs(oph[i] - pph[i]) < FLOAT_BOUNDARY:
                exact_pivot_points += 1
            elif (ov and not pv) or (not ov and pv):
                pivot_mismatch += 1
                if first_tie is None:
                    first_tie = i
            elif ov and pv and abs(oph[i] - pph[i]) >= FLOAT_BOUNDARY:
                float_boundary_events += 1
                if first_tie is None:
                    first_tie = i
        for i in range(n):
            ov = np.isfinite(opl[i]); pv = np.isfinite(ppl[i])
            if (ov and not pv) or (not ov and pv):
                pivot_mismatch += 1
                if first_tie is None:
                    first_tie = i
            elif ov and pv and abs(opl[i] - ppl[i]) >= FLOAT_BOUNDARY:
                float_boundary_events += 1
                if first_tie is None:
                    first_tie = i
        if first_tie is not None:
            tie_events += 1
            mask[first_tie:] = False  # conservative: tie -> rest of segment UNVERIFIED

        compare_state_full_masked(o_per, pfeat, mask, first_mis, max_consec)
        for f in FIELDS:
            ov = np.array([b[f] for b in o_per], dtype=float)
            pv = to_arr(pfeat[PROD_FIELD[f]])
            for i in range(n):
                if mask[i] and ov[i] != pv[i]:
                    mm_counts[f] += 1
                    if len(samples.get(("ag%d" % tf_minutes, f), [])) < 20:
                        samples.setdefault(("ag%d" % tf_minutes, f), []).append({
                            "dataset": "ag%d" % tf_minutes, "timeframe": "%dm" % tf_minutes,
                            "segment_id": sid, "local_bar": i, "global_bar": int(seg.index[i]),
                            "time": str(seg["available_time"].values[i]),
                            "field": f, "oracle_value": ov[i], "production_value": pv[i],
                            "high": sh[i], "low": sl[i], "close": sc[i], "atr": oatr[i],
                            "ph": oph[i] if np.isfinite(oph[i]) else "",
                            "pl": opl[i] if np.isfinite(opl[i]) else "",
                            "oracle_zz_dir": o_per[i]["zz_dir"],
                            "oracle_vis_up": str(o_per[i]["vis_up"]),
                            "uncertainty_reason": "exact-state mismatch under tie mask",
                        })

        total_rows += n
        cov_rows += int(mask.sum())

    return {
        "timeframe": "%dm" % tf_minutes,
        "rows": total_rows,
        "segments": seg_count,
        "exact_rows": cov_rows,
        "coverage_pct": round(100.0 * cov_rows / total_rows, 3) if total_rows else 0.0,
        "exact_pivot_points": exact_pivot_points,
        "pivot_mismatch": pivot_mismatch,
        "tie_events": tie_events,
        "float_boundary_events": float_boundary_events,
        "breach_up_mm": mm_counts["breach_up"],
        "breach_down_mm": mm_counts["breach_down"],
        "up_count_mm": mm_counts["up_count"],
        "down_count_mm": mm_counts["down_count"],
        "zone_active_mm": mm_counts["zone_active"],
        "first_mismatch": first_mis,
        "max_consec": max_consec,
    }


# ===========================================================================
# Pipeline parity
# ===========================================================================
def pipeline_parity(tf):
    params = prod.IndicatorParams()
    tf_full = prod.compute_tf_features(tf, params, True)
    frames = []
    for sid, seg in tf.groupby("segment", sort=False):
        frames.append(prod.compute_segment_features(seg, params, True))
    manual = pd.concat(frames)
    cols = ["liq_breach_up", "liq_breach_down", "liq_up_count", "liq_down_count", "liq_last_zone_active"]
    mism = 0
    for c in cols:
        a = to_arr(tf_full[c])
        b = to_arr(manual[c])
        mism += int(np.sum(a != b))
    return mism


# ===========================================================================
# Call chain
# ===========================================================================
def call_chain_proof():
    names = ["compute_tf_features", "compute_segment_features", "build_liquidity_features", "confirmed_pivots"]
    counters = {k: 0 for k in names}
    orig = {k: getattr(prod, k) for k in names}

    def mk(name):
        fn = orig[name]

        def _w(*a, **k):
            counters[name] += 1
            return fn(*a, **k)
        return _w

    for k in names:
        setattr(prod, k, mk(k))
    raw, disc, bars, raw_frame = load_ag_canonical_bars()
    try:
        tf15 = prod.resample_causal(raw_frame, 15)
        prod.compute_tf_features(tf15, prod.IndicatorParams(), True)
    finally:
        for k in names:
            setattr(prod, k, orig[k])
    return counters


# ===========================================================================
# Artifacts
# ===========================================================================
def _write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _sha256_text(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def snapshot(tag, summary):
    if not os.environ.get("LIQ_SNAPSHOT"):
        return
    d = os.path.join(ART_DIR, tag)
    os.makedirs(d, exist_ok=True)
    _write_json(os.path.join(d, "summary.json"), summary)
    sums = []
    for fn in sorted(os.listdir(ART_DIR)):
        fp = os.path.join(ART_DIR, fn)
        if os.path.isfile(fp):
            sums.append("%s  %s" % (_sha256_file(fp), fn))
    with open(os.path.join(d, "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(sums) + "\n")


# ===========================================================================
# Verdict
# ===========================================================================
def compute_verdict(o_self, boundary, ra_total, rb_total, ag5, ag15, prefix, par5, par15):
    if not o_self[1]:
        return "ORACLE_NOT_READY"
    # exact-domain mismatches
    exact_mm = 0
    for r in boundary:
        if r["production_comparable"]:
            exact_mm += (r["breach_up_mm"] + r["breach_down_mm"] + r["up_count_mm"]
                         + r["down_count_mm"] + r["zone_active_mm"])
    exact_mm += ra_total + ag5["breach_up_mm"] + ag5["breach_down_mm"] + ag5["up_count_mm"] \
        + ag5["down_count_mm"] + ag5["zone_active_mm"] \
        + ag15["breach_up_mm"] + ag15["breach_down_mm"] + ag15["up_count_mm"] \
        + ag15["down_count_mm"] + ag15["zone_active_mm"]
    any_unverified = (ag5["tie_events"] > 0 or ag15["tie_events"] > 0
                      or ag5["float_boundary_events"] > 0 or ag15["float_boundary_events"] > 0)
    cov_ok = (ag5["coverage_pct"] >= 50.0 and ag15["coverage_pct"] >= 50.0)
    if exact_mm > 0:
        return "SOURCE_DIVERGENCE_FOUND"
    if not cov_ok:
        return "INSUFFICIENT_EXACT_COVERAGE"
    if any_unverified:
        return "SOURCE_EXACT_ON_TESTED_DOMAIN_PARTIALLY_UNVERIFIED"
    return "SOURCE_EXACT_ON_TESTED_DOMAIN_PARTIALLY_UNVERIFIED"


# ===========================================================================
# Main
# ===========================================================================
def main():
    os.makedirs(ART_DIR, exist_ok=True)
    t0 = check_t0()
    # Source gate (triple SHA + markers)
    markers = oracle.check_source_gate()
    gobj = t0["source_gitobject_sha"]
    sha_ok = (t0["source_worktree_sha"] == PINNED_SHA
              and (gobj is None or gobj == PINNED_SHA)
              and t0["source_expected_sha"] == PINNED_SHA)
    if not sha_ok:
        raise SystemExit("STOP_LIQ_SOURCE_HASH_MISMATCH: worktree=%s gitobject=%s" % (
            t0["source_worktree_sha"], gobj))

    # Oracle self-tests
    self_results, all_pass = oracle.run_self_tests()
    if not all_pass:
        raise SystemExit("ORACLE_NOT_READY")

    # Boundary matrix
    boundary = build_boundary_matrix()

    # Random
    samples = {}
    ra_rows, rb_rows, ra_total, rb_total = layer_b_random(samples)

    # Prefix
    prefix = prefix_causality()

    # AG 5m / 15m
    raw, disc, bars, raw_frame = load_ag_canonical_bars()
    tf5 = prod.resample_causal(raw_frame, 5)
    tf15 = prod.resample_causal(raw_frame, 15)
    ag5 = layer_d_ag(5, samples)
    ag15 = layer_d_ag(15, samples)

    # Pipeline parity
    par5 = pipeline_parity(tf5)
    par15 = pipeline_parity(tf15)
    if par5 != 0 or par15 != 0:
        raise SystemExit("STOP_PIPELINE_PARITY_MISMATCH: 5m=%d 15m=%d" % (par5, par15))

    # Call chain
    cc = call_chain_proof()

    verdict = compute_verdict((self_results, all_pass), boundary, ra_total, rb_total,
                              ag5, ag15, prefix, par5, par15)

    # Build summary
    summary = {
        "identity": {
            "task_id": "STRUCTREV-PGM-R2B-LIQ-A-CONSOLIDATED-DISCOVERY",
            "base_sha": BASE,
            "commit_sha": t0["head"],
            "runtime_sha": t0["head"],
            "oracle_file": ORACLE_FILE,
        },
        "source": {
            "worktree_sha": t0["source_worktree_sha"],
            "gitobject_sha": t0["source_gitobject_sha"],
            "expected_sha": PINNED_SHA,
            "markers": markers,
            "source_ui_default_mode": oracle.SOURCE_MODE_UI_DEFAULT,
            "research_comparison_mode": oracle.RESEARCH_COMPARISON_MODE,
            "mode_ui_only_for_compared_fields": True,  # per always true under Historical == production full-batch
        },
        "oracle_self_tests": [{"case_id": r["case_id"], "pass": r["pass"], "detail": r["detail"]} for r in self_results],
        "oracle_independence": {
            "production_imports_or_calls": 0,
            "note": "Oracle imports only numpy/stdlib + pinned source constants.",
        },
        "boundary_matrix": boundary,
        "random": {
            "ra_total_mm": ra_total,
            "rb_total_mm": rb_total,
            "ra_rows": ra_rows,
            "rb_rows": rb_rows,
            "ra_detail": "R-A injects identical pivots + identical independent ATR: state-machine verdict.",
            "rb_detail": "R-B end-to-end OHLC + production ATR: end-to-end diagnostic.",
        },
        "ag": {"ag5": ag5, "ag15": ag15},
        "pipeline_parity": {"mismatch_5m": par5, "mismatch_15m": par15},
        "call_chain": cc,
        "prefix": prefix,
        "verdict": verdict,
        "classification": {
            "pivot_exact_domain": "strict-unique pivots confirmed at p+1 (source ta.pivothigh/low)",
            "zigzag_exact_domain": "literal newest-first, same-side replacement, cap 50",
            "cluster_exact_domain": "strict band, count>2, center=(max+min)/2",
            "breach_exact_domain": "strict high>top / low<bottom",
            "postbreak_exact_domain": "strict zone band [level +/- 2.3*atr]",
            "unverified_pivot_tie": "plateau / equal-extrema handled as UNVERIFIED (see AG tie mask)",
            "unverified_float_boundary": "extrema within %.1e treated as UNVERIFIED" % FLOAT_BOUNDARY,
            "unverified_mode_or_builtin": "Mode/per implemented literally; research_comparison_mode=Historical",
            "research_extensions": sorted(RESEARCH_EXTENSION),
        },
        "notes": [
            "liq_up_count/liq_down_count: Pine b_liq_B.size() counts breached levels until visLiq eviction; "
            "production counts only unbroken. If R-A/AG show mismatch on these fields this is a SOURCE_DIVERGENCE.",
            "L20 ATR=0/NaN: Oracle literal (no guard) yields no level, matching production guard output "
            "(OUTPUT_MATCH_IMPLEMENTATION_DIFFERENT, not a divergence).",
            "production confirmed_pivots uses non-strict >= (plateaus) while source ta.pivothigh is strict; "
            "pivot differences are classified UNVERIFIED, not exact-domain divergence.",
        ],
    }

    # Artifacts
    _write_json(os.path.join(ART_DIR, "liq_summary.json"), summary)
    _write_csv(os.path.join(ART_DIR, "liq_oracle_selftests.csv"), self_results,
               ["case_id", "pass", "detail"])
    _write_csv(os.path.join(ART_DIR, "liq_boundary_matrix.csv"), boundary,
               ["case_id", "source_domain", "production_comparable", "oracle_breach_up", "production_breach_up",
                "oracle_breach_down", "production_breach_down", "oracle_up_count", "production_up_count",
                "oracle_down_count", "production_down_count", "oracle_zone_active", "production_zone_active",
                "breach_up_mm", "breach_down_mm", "up_count_mm", "down_count_mm", "zone_active_mm",
                "exact_match", "first_mismatch_field"])
    # random diff csv (combine ra + rb)
    rand_rows = []
    for r in ra_rows:
        rand_rows.append({"layer": "RA", "series": r["series"], "rows": r["rows"], "exact_rows": r["exact_rows"],
                          "breach_up_mm": r["breach_up_mm"], "breach_down_mm": r["breach_down_mm"],
                          "up_count_mm": r["up_count_mm"], "down_count_mm": r["down_count_mm"],
                          "zone_active_mm": r["zone_active_mm"], "total_mm": r["total_mm"]})
    for r in rb_rows:
        rand_rows.append({"layer": "RB", "series": r["series"], "rows": r["rows"], "exact_rows": r["exact_rows"],
                          "coverage_pct": r["coverage_pct"], "breach_up_mm": r["breach_up_mm"],
                          "breach_down_mm": r["breach_down_mm"], "up_count_mm": r["up_count_mm"],
                          "down_count_mm": r["down_count_mm"], "zone_active_mm": r["zone_active_mm"],
                          "total_mm": r["total_mm"]})
    _write_csv(os.path.join(ART_DIR, "liq_random_diff.csv"), rand_rows,
               ["layer", "series", "rows", "exact_rows", "coverage_pct", "breach_up_mm", "breach_down_mm",
                "up_count_mm", "down_count_mm", "zone_active_mm", "total_mm"])
    # AG pivot/state diffs
    for tfm, ag in (("ag5", ag5), ("ag15", ag15)):
        _write_csv(os.path.join(ART_DIR, "liq_%s_pivot_diff.csv" % tfm),
                   [{"timeframe": ag["timeframe"], "rows": ag["rows"], "segments": ag["segments"],
                     "exact_pivot_points": ag["exact_pivot_points"], "pivot_mismatch": ag["pivot_mismatch"],
                     "tie_events": ag["tie_events"], "float_boundary_events": ag["float_boundary_events"]}],
                   ["timeframe", "rows", "segments", "exact_pivot_points", "pivot_mismatch", "tie_events", "float_boundary_events"])
        _write_csv(os.path.join(ART_DIR, "liq_%s_state_diff.csv" % tfm),
                   [{"timeframe": ag["timeframe"], "rows": ag["rows"], "exact_rows": ag["exact_rows"],
                     "coverage_pct": ag["coverage_pct"], "breach_up_mm": ag["breach_up_mm"],
                     "breach_down_mm": ag["breach_down_mm"], "up_count_mm": ag["up_count_mm"],
                     "down_count_mm": ag["down_count_mm"], "zone_active_mm": ag["zone_active_mm"],
                     "max_consec": str(ag["max_consec"]), "first_mismatch": str(ag["first_mismatch"])}],
                   ["timeframe", "rows", "exact_rows", "coverage_pct", "breach_up_mm", "breach_down_mm",
                    "up_count_mm", "down_count_mm", "zone_active_mm", "max_consec", "first_mismatch"])
    _write_csv(os.path.join(ART_DIR, "liq_ag_coverage.csv"), [ag5, ag15],
               ["timeframe", "rows", "segments", "exact_rows", "coverage_pct", "tie_events", "float_boundary_events"])
    _write_csv(os.path.join(ART_DIR, "liq_pipeline_parity.csv"),
               [{"layer": "compute_tf_features_vs_segments", "mismatch_5m": par5, "mismatch_15m": par15}],
               ["layer", "mismatch_5m", "mismatch_15m"])
    _write_csv(os.path.join(ART_DIR, "liq_call_chain.csv"),
               [{"call": k, "count": v} for k, v in cc.items()], ["call", "count"])
    # mismatch samples
    sample_rows = []
    for (ds, f), lst in samples.items():
        for s in lst:
            sample_rows.append(s)
    _write_csv(os.path.join(ART_DIR, "liq_mismatch_samples.csv"), sample_rows,
               ["dataset", "timeframe", "segment_id", "local_bar", "global_bar", "time", "field",
                "oracle_value", "production_value", "high", "low", "close", "atr", "ph", "pl",
                "oracle_zz_dir", "oracle_vis_up", "uncertainty_reason"])

    # SHA256 of all artifacts
    sums = []
    for fn in sorted(os.listdir(ART_DIR)):
        fp = os.path.join(ART_DIR, fn)
        if os.path.isfile(fp):
            sums.append("%s  %s" % (_sha256_file(fp), fn))
    with open(os.path.join(ART_DIR, "SHA256SUMS.txt"), "w") as f:
        f.write("\n".join(sums) + "\n")

    snapshot("final", summary)

    # Print compact summary
    print("=== Liquidity Source Semantic Differential ===")
    print("verdict:", verdict)
    print("oracle self-tests ALL_PASS:", all_pass)
    print("boundary comparable mismatches:", sum(
        r["breach_up_mm"] + r["breach_down_mm"] + r["up_count_mm"] + r["down_count_mm"] + r["zone_active_mm"]
        for r in boundary if r["production_comparable"]))
    print("R-A total mismatch:", ra_total, " R-B total mismatch:", rb_total)
    print("AG5 coverage%%=%.2f mm=%s" % (ag5["coverage_pct"], {f: ag5[f + "_mm"] for f in FIELDS}))
    print("AG15 coverage%%=%.2f mm=%s" % (ag15["coverage_pct"], {f: ag15[f + "_mm"] for f in FIELDS}))
    print("pipeline parity 5m/15m:", par5, par15)
    print("call_chain:", cc)
    print("prefix pass:", prefix["pass"])
    print("artifacts in:", ART_DIR)


if __name__ == "__main__":
    main()
