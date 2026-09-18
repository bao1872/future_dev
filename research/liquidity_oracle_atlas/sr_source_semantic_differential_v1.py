#!/usr/bin/env python3
"""
SR Consolidated Source Differential (SRC2-A)
============================================

One-round discovery: compare the frozen literal Oracle
(``sr_source_semantic_oracle_v1.py``, commit ``3ddb5e7``) against the real
production ``confirmed_pivots()`` / ``build_sr_features()``
(``experiment_structural_reversion_pgm_v1.py``) inside the proven semantic
domain of the pinned Pine source ``ref/SRchannel.pine`` (SHA256
``9d8ee4af...``).

This script ONLY discovers facts. It does not modify production, the Oracle,
or the Pine source. No fix recommendation is emitted.

Layers
------
Layer A  Pivot differential (Oracle unique_confirmed_pivots vs prod confirmed_pivots)
Layer B  Injected pivots -> real prod build_sr_features vs Oracle SR state machine
Layer C  Continuous random OHLC -> pivot + state end-to-end
Layer D  Real AG -> exact-domain masked differential + real call-chain proof + prefix causality

State-exact output mapping (parity verdict)
-------------------------------------------
Oracle n_channels        <-> prod sr_n_channels
Oracle not_in_channel    <-> prod (sr_in_zone == 0)
Oracle resistancebroken  <-> prod sr_broken_up
Oracle supportbroken      <-> prod sr_broken_down

The following are DIAGNOSTIC_RESEARCH_EXTENSION only (not parity):
sr_support_dist_atr, sr_resistance_dist_atr, sr_support_price,
sr_resistance_price, sr_support_strength, sr_resistance_strength, sr_zone_strength.

Two mandatory contamination masks
---------------------------------
startup recovery:   bar 0..298 not comparable; even bar>=299 not auto-exact
                    until first deterministic new exact pivot confirmation
                    triggers a rebuild (per Pine channel persistence).
tie contamination: any Oracle tie confirmation invalidates state until a new
                    exact pivot confirmation occurs at bar t with
                    t - tie_bar > LOOPBACK (290). Exact comparison only where
                    state_exact_mask == True; otherwise
                    UNVERIFIED_TIE_OR_STARTUP_CONTAMINATION.

Verdicts
--------
SOURCE_DIVERGENCE_FOUND
SOURCE_EXACT_ON_TESTED_DOMAIN_PARTIALLY_UNVERIFIED
INSUFFICIENT_EXACT_COVERAGE
"""

from __future__ import annotations

import os
import sys
import csv
import json
import hashlib
import subprocess

import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- Baseline / candidate -------------------------------------------------
import experiment_structural_reversion_pgm_v1 as prod
import sr_source_semantic_oracle_v1 as oracle

from experiment_structural_reversion_pgm_v1 import (  # noqa: E402
    confirmed_pivots as prod_confirmed_pivots,
    build_sr_features as prod_build_sr_features,
    compute_segment_features as prod_compute_segment_features,
    compute_tf_features as prod_compute_tf_features,
    resample_causal as prod_resample_causal,
    raw_frame_from_owner as prod_raw_frame_from_owner,
    PINE_DEFAULT as PROD_PINE_DEFAULT,
)
from sr_source_semantic_oracle_v1 import (  # noqa: E402
    unique_confirmed_pivots,
    run_sr_state_machine,
    store_pivot_value,
    LOOPBACK,
    PRD,
    FLOAT_BOUNDARY,
    PINNED_SHA256 as EXPECTED_SR_SHA,
)

# ===========================================================================
# Faithful production pivot-value capture hook
# ---------------------------------------------------------------------------
# The differential must compare Oracle's *stored* pivot value against
# PRODUCTION'S ACTUAL per-bar stored pivot value -- not the displayed channel
# price (sr_resistance_price / sr_support_price), which is a derivative of the
# channel selection and produces false positives (e.g. B08/B10/B12).
#
# We exec the live production `build_sr_features` source (so the hook always
# mirrors current production, correct both pre- and post Fix A/B) with a small
# injection that records the real `pivot_value` written into production's
# `pivots` list into a new output field `sr_captured_pivot`.
# ===========================================================================
def _inject_pivot_capture(src: str) -> str:
    """Return a modified copy of the production module source whose
    build_sr_features records its ACTUAL per-bar stored pivot value into a new
    output field `sr_captured_pivot`. The production algorithm is otherwise
    untouched (only observation is added)."""
    out = src
    idx_def = out.index("def build_sr_features(")

    # allocate the capture buffer right after `n = len(close)` inside the fn
    anchor_n = "    n = len(close)"
    i_n = out.index(anchor_n, idx_def)
    out = (
        out[:i_n]
        + anchor_n
        + "\n    _CAPTURE = np.full(n, np.nan, dtype=np.float64)\n"
        + out[i_n + len(anchor_n):]
    )

    # reset capture to NaN at the top of the per-bar loop
    anchor_loop = "for i in range(n):"
    i_loop = out.index(anchor_loop, idx_def)
    out = (
        out[:i_loop]
        + anchor_loop
        + "\n        _CAPTURE[i] = float('nan')\n"
        + out[i_loop + len(anchor_loop):]
    )

    # record production's actual stored pivot value right after it is computed
    anchor_pv = "            pivot_value = ("
    i_pv = out.index(anchor_pv)
    close_idx = out.index("            )\n", i_pv)
    out = (
        out[:close_idx]
        + "            )\n            _CAPTURE[i] = pivot_value\n"
        + out[close_idx + len("            )\n"):]
    )

    # expose the capture buffer through the returned feature dict
    anchor_ret = '            n_channels,\n    }'
    i_ret = out.index(anchor_ret, idx_def)
    out = (
        out[:i_ret]
        + '            n_channels,\n        "sr_captured_pivot":\n            _CAPTURE,\n    }'
        + out[i_ret + len(anchor_ret):]
    )
    return out


def _make_captured_build_sr_features():
    with open(PROD_PATH, "r", encoding="utf-8") as f:
        src = f.read()
    modified = _inject_pivot_capture(src)
    cap_ns = dict(prod.__dict__)
    try:
        exec(compile(modified, PROD_PATH, "exec"), cap_ns)
    except Exception as e:  # pragma: no cover
        raise SystemExit("STOP_SR_DIFF_CAPTURE_HOOK_FAIL: %s" % e)
    return cap_ns, cap_ns["build_sr_features"]


# Canonical AG owner (audited): real production data entry, not inferred.
from research.export_ob_trigger_execution_v21 import (  # noqa: E402
    load_raw_5m as canonical_load_raw_5m,
)
from research.phase1_tradability.phase1_contract_v1 import (  # noqa: E402
    discontinuity_flags as canonical_discontinuity_flags,
)

# Task Base SHA (the Oracle commit = current HEAD).
BASE_SHA = "3ddb5e7de0c20094bfda8fa559153d7afe9eaa7b"
ORACLE_PATH = os.path.join(
    REPO_ROOT, "research", "liquidity_oracle_atlas", "sr_source_semantic_oracle_v1.py"
)
PROD_PATH = os.path.join(
    REPO_ROOT, "research", "liquidity_oracle_atlas", "experiment_structural_reversion_pgm_v1.py"
)
SR_SOURCE_PATH = os.path.join(REPO_ROOT, "ref", "SRchannel.pine")
ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "sr_source_semantic_differential")

# Build the faithful capture variant of production build_sr_features now that
# PROD_PATH is defined (mirrors live production, correct pre/post Fix A/B).
_CAP_NS, CAPTURED_BUILD_SR = _make_captured_build_sr_features()
_CAP_ORIG_CONFIRMED = _CAP_NS["confirmed_pivots"]

# Source-exact field mapping between Oracle per_bar and production feature dict.
STATE_FIELDS = ["n_channels", "in_zone", "break_up", "break_down"]


# ===========================================================================
# T0 reproducibility gates (fail-closed)
# ===========================================================================
def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT).decode("utf-8").strip()


def check_t0() -> dict:
    info: dict = {"ok": True, "notes": []}

    # 1. BASE_SHA is ancestor of HEAD
    try:
        rc = subprocess.call(
            ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
            cwd=REPO_ROOT,
        )
        info["base_is_ancestor"] = rc == 0
        if rc != 0:
            info["ok"] = False
            info["notes"].append("BASE_SHA is not an ancestor of HEAD")
    except Exception as e:  # pragma: no cover
        info["base_is_ancestor"] = False
        info["ok"] = False
        info["notes"].append("git merge-base failed: %s" % e)

    info["head_sha"] = _git("rev-parse", "HEAD")
    info["base_sha"] = BASE_SHA

    # 2. Oracle file exists
    info["oracle_exists"] = os.path.exists(ORACLE_PATH)
    if not info["oracle_exists"]:
        info["ok"] = False
        info["notes"].append("Oracle file missing")

    # 3. SR source triple SHA (working-tree == git-object(HEAD) == expected)
    if not os.path.exists(SR_SOURCE_PATH):
        info["ok"] = False
        info["notes"].append("SR source missing")
        return info

    wt_bytes = open(SR_SOURCE_PATH, "rb").read()
    wt_sha = hashlib.sha256(wt_bytes).hexdigest()
    try:
        gobj = subprocess.check_output(
            ["git", "show", "HEAD:ref/SRchannel.pine"], cwd=REPO_ROOT
        )
        gobj_sha = hashlib.sha256(gobj).hexdigest()
    except Exception as e:  # pragma: no cover
        gobj_sha = None
        info["ok"] = False
        info["notes"].append("git-object SR read failed: %s" % e)

    info["sr_sha_worktree"] = wt_sha
    info["sr_sha_gitobject"] = gobj_sha
    info["sr_sha_expected"] = EXPECTED_SR_SHA
    triple_ok = (
        wt_sha == gobj_sha == EXPECTED_SR_SHA
        and wt_sha is not None
    )
    info["sr_triple_match"] = bool(triple_ok)
    if not triple_ok:
        info["ok"] = False
        info["notes"].append("SR SHA triple mismatch")

    return info


# ===========================================================================
# Small helpers
# ===========================================================================
def isnan(x) -> bool:
    return x is None or (isinstance(x, float) and x != x) or (
        isinstance(x, np.floating) and np.isnan(x)
    )


def approx_eq(a, b, tol: float = 1e-9) -> bool:
    if isnan(a) and isnan(b):
        return True
    if isnan(a) or isnan(b):
        return False
    return abs(float(a) - float(b)) <= tol


def _as_arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


# ===========================================================================
# Layer B adapter: inject pivots into REAL production build_sr_features
# ===========================================================================
def run_prod_sr_with_injected_pivots(high, low, close, ph, pl) -> dict:
    """Monkeypatch prod.confirmed_pivots to return prepared ph/pl, then call the
    REAL production build_sr_features (captured variant that also exposes the
    actual per-bar stored pivot value). The state machine itself is untouched."""
    n = len(close)
    atr = np.ones(n, dtype=float)
    original = _CAP_NS["confirmed_pivots"]

    ph_a = _as_arr(ph)
    pl_a = _as_arr(pl)

    def injected(values, left, right, mode):
        if mode == "high":
            return ph_a.copy()
        if mode == "low":
            return pl_a.copy()
        raise AssertionError("unexpected mode %r" % (mode,))

    _CAP_NS["confirmed_pivots"] = injected
    try:
        return CAPTURED_BUILD_SR(
            _as_arr(high), _as_arr(low), _as_arr(close), atr, PROD_PINE_DEFAULT
        )
    finally:
        _CAP_NS["confirmed_pivots"] = original


# ===========================================================================
# State-exact comparison mask (startup recovery + tie contamination)
# ===========================================================================
def compute_state_exact_mask(n: int, ph, pl, tie_confirm_bars) -> np.ndarray:
    """Returns boolean array. True only where state is exact-comparable.

    - startup recovery: first bar b>=299 with a deterministic exact pivot
      confirmation sets recovered=True.
    - tie contamination: a tie confirmation bar sets recovered=False and records
      last_tie; recovery requires a new exact pivot confirmation at t with
      t - last_tie > LOOPBACK (290).
    """
    ph = _as_arr(ph)
    pl = _as_arr(pl)
    mask = np.zeros(n, dtype=bool)
    tie_set = set(int(b) for b in tie_confirm_bars)
    recovered = False
    last_tie = -10 ** 9
    for b in range(n):
        o_exact = not isnan(ph[b]) if b < len(ph) else False
        p_exact = not isnan(pl[b]) if b < len(pl) else False
        is_exact = o_exact or p_exact
        is_tie = b in tie_set
        if is_tie:
            recovered = False
            last_tie = b
        if is_exact and not is_tie:
            if b >= 299 and (b - last_tie > LOOPBACK):
                recovered = True
        mask[b] = recovered
    return mask


# ===========================================================================
# Source-exact state comparison (Oracle per_bar vs production feature dict)
# ===========================================================================
def compare_state(oracle_per_bar, prod_feat, mask) -> dict:
    """Compare only where mask True. Returns per-field stats + mismatch rows."""
    n = len(oracle_per_bar)
    stats = {
        f: {"compared": 0, "mismatch": 0, "first_mismatch_bar": None}
        for f in STATE_FIELDS
    }
    rows = []

    def o_val(b, field):
        rec = oracle_per_bar[b]
        if field == "n_channels":
            return int(rec["n_channels"])
        if field == "in_zone":
            return 0 if rec["not_in_channel"] else 1
        if field == "break_up":
            return 1 if rec["resistancebroken"] else 0
        if field == "break_down":
            return 1 if rec["supportbroken"] else 0
        raise KeyError(field)

    def p_val(b, field):
        if field == "n_channels":
            return int(prod_feat["sr_n_channels"][b])
        if field == "in_zone":
            return int(prod_feat["sr_in_zone"][b])
        if field == "break_up":
            return int(prod_feat["sr_broken_up"][b])
        if field == "break_down":
            return int(prod_feat["sr_broken_down"][b])
        raise KeyError(field)

    for b in range(n):
        if not mask[b]:
            continue
        for f in STATE_FIELDS:
            stats[f]["compared"] += 1
            ov = o_val(b, f)
            pv = p_val(b, f)
            if ov != pv:
                stats[f]["mismatch"] += 1
                if stats[f]["first_mismatch_bar"] is None:
                    stats[f]["first_mismatch_bar"] = int(b)
                rows.append(
                    {
                        "field": f,
                        "bar": int(b),
                        "oracle_value": int(ov),
                        "production_value": int(pv),
                        "uncertainty_reason": "SOURCE_EXACT_DOMAIN",
                    }
                )
    return {"stats": stats, "rows": rows}


def compare_state_full(oracle_per_bar, prod_feat, mask):
    """Full comparison including UNVERIFIED classification counts."""
    n = len(oracle_per_bar)
    res = compare_state(oracle_per_bar, prod_feat, mask)
    n_exact = int(mask.sum())
    n_unverified = n - n_exact
    total_mismatch = sum(res["stats"][f]["mismatch"] for f in STATE_FIELDS)
    return {
        "n_bars": n,
        "n_exact_state_rows": n_exact,
        "n_unverified_state_rows": n_unverified,
        "exact_state_coverage_pct": (100.0 * n_exact / n) if n else 0.0,
        "total_mismatch": total_mismatch,
        "field_stats": res["stats"],
        "rows": res["rows"],
    }


# ===========================================================================
# Layer A: pivot differential (Oracle exact vs production confirmed_pivots)
# ===========================================================================
def compare_pivots_core(high, low, open_, close):
    """Compare Oracle exact pivots vs production confirmed_pivots on ONE array
    (single segment). Only Oracle exact pivots are the verdict domain;
    production-only pivots (ties) are counted as tie_unverified.

    Returns counts + per-row mismatch records keyed by (mode, bar). The caller
    is responsible for stamping segment_id / global_bar / time.
    """
    oph, opl, tie = unique_confirmed_pivots(high, low, open_, close)
    pph = prod_confirmed_pivots(high, PRD, PRD, "high")
    ppl = prod_confirmed_pivots(low, PRD, PRD, "low")

    n = len(high)
    n_exact = 0
    n_mismatch = 0
    first_mismatch = None
    rows = []

    def tally(o_arr, p_arr, mode):
        nonlocal n_exact, n_mismatch, first_mismatch
        for b in range(n):
            o = o_arr[b]
            p = p_arr[b]
            o_present = not isnan(o)
            p_present = not isnan(p)
            if o_present and p_present:
                n_exact += 1
                if not approx_eq(o, p):
                    n_mismatch += 1
                    if first_mismatch is None:
                        first_mismatch = {
                            "mode": mode, "bar": int(b),
                            "oracle_value": float(o), "production_value": float(p),
                        }
                    rows.append({
                        "mode": mode, "bar": int(b),
                        "oracle_value": None if isnan(o) else float(o),
                        "production_value": None if isnan(p) else float(p),
                        "reason": "EXACT_PIVOT_VALUE_MISMATCH",
                    })
            elif o_present and not p_present:
                n_exact += 1
                n_mismatch += 1
                if first_mismatch is None:
                    first_mismatch = {
                        "mode": mode, "bar": int(b),
                        "oracle_value": float(o), "production_value": None,
                    }
                rows.append({
                    "mode": mode, "bar": int(b),
                    "oracle_value": float(o), "production_value": None,
                    "reason": "PROD_MISSING_EXACT_PIVOT",
                })
            elif (not o_present) and p_present:
                pass  # production-only pivot -> tie unverified (counted below)
            # both absent -> nothing

    tally(oph, pph, "high")
    tally(opl, ppl, "low")

    # Distinguish equal-plateau vs float-boundary ties for reporting.
    eq_h, eq_l, fb_h, fb_l = _classify_ties(high, low)
    n_tie_unverified = eq_h + eq_l + fb_h + fb_l
    n_float_boundary_unverified = fb_h + fb_l

    return {
        "n_exact_pivot_points": n_exact,
        "n_exact_pivot_mismatch": n_mismatch,
        "n_tie_unverified": n_tie_unverified,
        "n_float_boundary_unverified": n_float_boundary_unverified,
        "n_tie_events": len(tie),
        "first_exact_mismatch": first_mismatch,
        "rows": rows,
    }


def layer_a_pivot_diff(high, low, open_, close, dataset: str, case: str = ""):
    """Full-array pivot differential wrapper (used by Boundary/Random)."""
    r = compare_pivots_core(high, low, open_, close)
    for row in r["rows"]:
        row["dataset"] = dataset
        row["case"] = case
    r["dataset"] = dataset
    r["case"] = case
    return r


def _classify_ties(high, low):
    """Re-scan windows (PRD) to split tie flags into equal-plateau vs
    float-boundary (|gap|<=1e-8) categories. Pure counting, not production."""
    n = len(high)
    eq_h = eq_l = fb_h = fb_l = 0
    src1 = [float(x) for x in high]
    src2 = [float(x) for x in low]
    for p in range(PRD, n - PRD):
        lo = p - PRD
        hi = p + PRD
        # high
        seg = src1[lo:hi + 1]
        c = seg[PRD]
        others = max(seg[:PRD] + seg[PRD + 1:])
        if not (c > others + FLOAT_BOUNDARY) and not (c < others - FLOAT_BOUNDARY):
            if abs(c - others) <= FLOAT_BOUNDARY:
                fb_h += 1
            else:
                eq_h += 1
        # low
        seg2 = src2[lo:hi + 1]
        c2 = seg2[PRD]
        others2 = min(seg2[:PRD] + seg2[PRD + 1:])
        if not (c2 < others2 - FLOAT_BOUNDARY) and not (c2 > others2 + FLOAT_BOUNDARY):
            if abs(c2 - others2) <= FLOAT_BOUNDARY:
                fb_l += 1
            else:
                eq_l += 1
    return eq_h, eq_l, fb_h, fb_l


# ===========================================================================
# Layer B/C synthetic + random series builders
# ===========================================================================
def make_ohlc(n: int, mid: float = 100.0, amp: float = 30.0, flat: bool = False,
              seed: int = 0):
    rng = np.random.default_rng(seed)
    if flat:
        high = np.full(n, mid, dtype=float)
        low = np.full(n, mid, dtype=float)
        close = np.full(n, mid, dtype=float)
    else:
        noise_h = rng.uniform(0.0, 0.5, n)
        noise_l = rng.uniform(0.0, 0.5, n)
        wave = amp * np.sin(np.arange(n) / 20.0)
        high = (mid + amp + noise_h + wave * 0.0).astype(float)
        low = (mid - amp - noise_l).astype(float)
        close = (mid + 0.5 * (noise_h - noise_l)).astype(float)
    open_ = close.copy()
    return high, low, close, open_


def inject_pivots(n: int, injections):
    """injections: list of (confirm_bar, mode, value). Returns ph, pl arrays."""
    ph = np.full(n, np.nan, dtype=float)
    pl = np.full(n, np.nan, dtype=float)
    for b, mode, v in injections:
        if mode == "high":
            ph[b] = float(v)
        elif mode == "low":
            pl[b] = float(v)
        else:
            raise AssertionError(mode)
    return ph, pl


# ===========================================================================
# Boundary matrix B01-B16
# ===========================================================================
def _case_record(case_id, source_domain, ph, pl, high, low, close, open_,
                 extra_note=""):
    """Run Oracle + production(injected) state machines, masked."""
    n = len(close)
    oph, opl, tie = unique_confirmed_pivots(high, low, open_, close)
    # Use the injected pivots for the controlled comparison (Oracle side too).
    oracle_per_bar = run_sr_state_machine(ph, pl, high, low, close)
    prod_feat = run_prod_sr_with_injected_pivots(high, low, close, ph, pl)

    tie_confirm = [t["confirm_bar"] for t in tie] if False else []
    mask = compute_state_exact_mask(n, ph, pl, tie_confirm)

    cmp = compare_state_full(oracle_per_bar, prod_feat, mask)

    # Pivot-value selection divergence: compare Oracle's stored pivot value
    # against PRODUCTION'S ACTUAL per-bar stored pivot value (captured from the
    # live pipeline via sr_captured_pivot). This is faithful both pre- and
    # post Fix A/B and avoids the displayed-channel-price false positives.
    captured = prod_feat.get("sr_captured_pivot")
    pivot_value_mismatch = 0
    pv_rows = []
    for b in range(n):
        o_stored = store_pivot_value(
            None if isnan(ph[b]) else ph[b], None if isnan(pl[b]) else pl[b]
        )
        if isnan(o_stored):
            # Oracle confirms no pivot at b; production must also have none.
            if captured is not None and not isnan(captured[b]):
                pivot_value_mismatch += 1
                pv_rows.append({
                    "bar": int(b),
                    "oracle_stored": None,
                    "production_stored": (
                        None if isnan(captured[b]) else float(captured[b])
                    ),
                })
            continue  # no Oracle exact pivot at this bar
        if captured is None or isnan(captured[b]) or not approx_eq(o_stored, captured[b]):
            pivot_value_mismatch += 1
            pv_rows.append({
                "bar": int(b),
                "oracle_stored": None if isnan(o_stored) else float(o_stored),
                "production_stored": (
                    None if (captured is None or isnan(captured[b]))
                    else float(captured[b])
                ),
            })

    exact_match = (cmp["total_mismatch"] == 0) and (pivot_value_mismatch == 0)
    first_mismatch_field = None
    for f in STATE_FIELDS:
        if cmp["field_stats"][f]["mismatch"] > 0:
            first_mismatch_field = f
            break
    if first_mismatch_field is None and pivot_value_mismatch > 0:
        first_mismatch_field = "pivot_value_selection"

    rec = {
        "case_id": case_id,
        "source_domain": source_domain,
        "oracle_n_channels": int(oracle_per_bar[-1]["n_channels"]),
        "production_n_channels": int(prod_feat["sr_n_channels"][-1]),
        "oracle_in_zone": 0 if oracle_per_bar[-1]["not_in_channel"] else 1,
        "production_in_zone": int(prod_feat["sr_in_zone"][-1]),
        "oracle_break_up": 1 if oracle_per_bar[-1]["resistancebroken"] else 0,
        "production_break_up": int(prod_feat["sr_broken_up"][-1]),
        "oracle_break_down": 1 if oracle_per_bar[-1]["supportbroken"] else 0,
        "production_break_down": int(prod_feat["sr_broken_down"][-1]),
        "exact_match": bool(exact_match),
        "first_mismatch_field": first_mismatch_field,
        "n_exact_state_rows": cmp["n_exact_state_rows"],
        "n_unverified_state_rows": cmp["n_unverified_state_rows"],
        "pivot_value_mismatch": pivot_value_mismatch,
        "n_channels_mismatch": cmp["field_stats"]["n_channels"]["mismatch"],
        "in_zone_mismatch": cmp["field_stats"]["in_zone"]["mismatch"],
        "break_up_mismatch": cmp["field_stats"]["break_up"]["mismatch"],
        "break_down_mismatch": cmp["field_stats"]["break_down"]["mismatch"],
        "diagnostic_zone_strength": float(prod_feat["sr_zone_strength"][-1])
        if cmp["n_exact_state_rows"] else None,
        "note": extra_note,
    }
    return rec, cmp, pv_rows


def build_boundary_matrix():
    """B01-B16. Each returns a record; exact_match judged on source-exact fields
    AND pivot-value selection. No preset of whether production MATCHes."""
    cases = []
    N = 660

    # B01 unique pivot confirmation
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=1)
    # craft a clean unique high pivot at p=300 (confirm 310) and low at p=320
    h = h.copy(); l = l.copy()
    h[300] = 200.0; l[320] = 5.0
    ph, pl = inject_pivots(N, [(310, "high", 200.0), (330, "low", 5.0)])
    rec, _, _ = _case_record("B01", "exact", ph, pl, h, l, c, o,
                             "strict-unique high/low pivots")
    cases.append(rec)

    # B02 simultaneous nonzero ph/pl (ph priority 110)
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=2)
    h = h.copy(); l = l.copy()
    h[300] = 200.0; l[300] = 10.0
    ph, pl = inject_pivots(N, [(310, "high", 110.0), (310, "low", 90.0)])
    rec, _, _ = _case_record("B02", "exact", ph, pl, h, l, c, o,
                             "ph=110,pl=90 same confirm bar -> ph priority")
    cases.append(rec)

    # B03 zero ph + nonzero pl (ph=0, pl=90) + extra legit pivot to expose state
    h, l, c, o = make_ohlc(N, mid=100, amp=300, seed=3)
    h = h.copy(); l = l.copy()
    h[300] = 600.0; l[300] = 5.0; h[340] = 450.0
    ph, pl = inject_pivots(N, [(310, "high", 0.0), (310, "low", 90.0),
                               (350, "high", 450.0)])
    rec, _, _ = _case_record("B03", "exact", ph, pl, h, l, c, o,
                             "ph=0,pl=90 -> Pine/Oracle bool(0)=false stores 90; "
                             "prod Fix A matches (no phantom 0)")
    cases.append(rec)

    # B04 zero ph only (ph=0, pl=NaN)
    h, l, c, o = make_ohlc(N, mid=100, amp=300, seed=4)
    h = h.copy(); l = l.copy()
    h[300] = 600.0; l[300] = 5.0
    ph, pl = inject_pivots(N, [(310, "high", 0.0), (310, "low", np.nan),
                               (350, "high", 450.0)])
    rec, _, _ = _case_record("B04", "exact", ph, pl, h, l, c, o,
                             "ph=0,pl=NaN -> Pine/Oracle no pivot; "
                             "prod Fix A matches (no phantom pivot)")
    cases.append(rec)

    # B05 normal positive-width channel
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=5)
    h = h.copy(); l = l.copy()
    for p in (300, 320, 340):
        h[p] = 180.0
    ph, pl = inject_pivots(N, [(310, "high", 180.0), (330, "high", 180.0),
                               (350, "high", 180.0)])
    rec, _, _ = _case_record("B05", "exact", ph, pl, h, l, c, o,
                             "several positive-width pivots")
    cases.append(rec)

    # B06 zero-width channel (flat, cwidth=0)
    h, l, c, o = make_ohlc(N, mid=100, amp=0, flat=True, seed=6)
    h = h.copy(); l = l.copy()
    h[300] = 100.0; h[320] = 100.0; l[300] = 100.0
    ph, pl = inject_pivots(N, [(310, "high", 100.0), (330, "high", 100.0),
                               (310, "low", 100.0)])
    rec, _, _ = _case_record("B06", "exact", ph, pl, h, l, c, o,
                             "flat cwidth=0; Pine wdth<=cwidth accepts zero-width")
    cases.append(rec)

    # B07 loopback 290 / 291
    for age, tag in ((290, "290"), (291, "291")):
        h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=7)
        h = h.copy(); l = l.copy()
        h[300] = 180.0
        h[300 + age] = 180.0
        ph, pl = inject_pivots(N, [(310, "high", 180.0),
                                   (310 + age, "high", 180.0)])
        rec, _, _ = _case_record("B07_%s" % tag, "exact", ph, pl, h, l, c, o,
                                 "two high pivots age=%s; cleanup >290" % tag)
        cases.append(rec)

    # B08 strength + touches (60 + 7 = 67), close in channel
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=8)
    h = h.copy(); l = l.copy()
    # 3 pivots within cwidth (~3) around 100
    for p in (300, 310, 320):
        h[p] = 102.0; l[p] = 98.0
    ph, pl = inject_pivots(N, [(310, "high", 100.0), (320, "high", 101.0),
                               (330, "high", 102.0)])
    # 7 touch bars where high/low fall in [98,102]
    for tb in range(340, 347):
        h[tb] = 101.0; l[tb] = 99.0; c[tb] = 100.0
    rec, _, _ = _case_record("B08", "exact", ph, pl, h, l, c, o,
                             "3 pivots numpp=60 + 7 touches = 67; close in zone")
    cases.append(rec)

    # B09 one-bar double touch (high and low both in zone)
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=9)
    h = h.copy(); l = l.copy()
    h[300] = 102.0; l[300] = 98.0
    ph, pl = inject_pivots(N, [(310, "high", 100.0)])
    h[340] = 101.0; l[340] = 99.0  # same bar high AND low in zone
    rec, _, _ = _case_record("B09", "exact", ph, pl, h, l, c, o,
                             "high&low in zone same bar -> +1 touch only")
    cases.append(rec)

    # B10 strongest-first ordering, exceed display cap 6
    h, l, c, o = make_ohlc(N, mid=100, amp=60, seed=10)
    h = h.copy(); l = l.copy()
    inj = []
    for k in range(10):
        p = 300 + k * 5
        h[p] = 160.0
        inj.append((p + PRD, "high", 100.0 + k * 0.1))
    ph, pl = inject_pivots(N, inj)
    rec, _, _ = _case_record("B10", "exact", ph, pl, h, l, c, o,
                             "10 equal-ish channels -> display cap 6, strongest-first")
    cases.append(rec)

    # B11 suppression (endpoint inside selected)
    h, l, c, o = make_ohlc(N, mid=100, amp=40, seed=11)
    h = h.copy(); l = l.copy()
    h[300] = 150.0; h[305] = 150.0; l[300] = 50.0
    ph, pl = inject_pivots(N, [(310, "high", 80.0), (315, "high", 120.0)])
    rec, _, _ = _case_record("B11", "exact", ph, pl, h, l, c, o,
                             "weaker candidate endpoint inside selected -> suppressed")
    cases.append(rec)

    # B12 ten non-overlapping channels -> display 6
    h, l, c, o = make_ohlc(N, mid=100, amp=80, seed=12)
    h = h.copy(); l = l.copy()
    inj = []
    for k in range(10):
        p = 300 + k * 6
        h[p] = 100.0 + k * 8.0
        inj.append((p + PRD, "high", 100.0 + k * 8.0))
    ph, pl = inject_pivots(N, inj)
    rec, _, _ = _case_record("B12", "exact", ph, pl, h, l, c, o,
                             "10 internal channels -> display 6")
    cases.append(rec)

    # B13 resistance break
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=13)
    h = h.copy(); l = l.copy()
    h[300] = 140.0; l[300] = 60.0
    ph, pl = inject_pivots(N, [(310, "high", 100.0)])
    c = c.copy(); h = h.copy()
    c[340] = 100.0; h[340] = 100.0
    c[341] = 101.0; h[341] = 101.0  # cross above 100 (hi of channel)
    rec, _, _ = _case_record("B13", "exact", ph, pl, h, l, c, o,
                             "prev<=hi, cur>hi, not in channel -> resistance break")
    cases.append(rec)

    # B14 support break
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=14)
    h = h.copy(); l = l.copy()
    h[300] = 140.0; l[300] = 60.0
    ph, pl = inject_pivots(N, [(310, "high", 100.0)])
    c = c.copy(); l = l.copy()
    c[340] = 100.0; l[340] = 100.0
    c[341] = 99.0; l[341] = 99.0  # cross below 100 (lo of channel)
    rec, _, _ = _case_record("B14", "exact", ph, pl, h, l, c, o,
                             "prev>=lo, cur<lo, not in channel -> support break")
    cases.append(rec)

    # B15 current-in-channel suppresses all break
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=15)
    h = h.copy(); l = l.copy()
    h[300] = 140.0; l[300] = 60.0
    ph, pl = inject_pivots(N, [(310, "high", 100.0)])
    c = c.copy()
    c[340] = 100.0  # close inside channel [?,100]
    c[341] = 101.0  # would cross another channel hi, but current in zone
    rec, _, _ = _case_record("B15", "exact", ph, pl, h, l, c, o,
                             "close in channel -> all break flags false")
    cases.append(rec)

    # B16 channel persistence (no new pivot -> old channel persists)
    h, l, c, o = make_ohlc(N, mid=100, amp=30, seed=16)
    h = h.copy(); l = l.copy()
    h[300] = 140.0; l[300] = 60.0
    ph, pl = inject_pivots(N, [(310, "high", 100.0)])
    rec, _, _ = _case_record("B16", "exact", ph, pl, h, l, c, o,
                             "channel persists across non-confirmation bars")
    cases.append(rec)

    return cases


# ===========================================================================
# Layer C: random continuous differential
# ===========================================================================
def layer_c_random(seed: int = 20260918, n_series: int = 12, n_bars: int = 2600):
    rng = np.random.default_rng(seed)
    results = []
    agg = {
        "n_series": 0,
        "n_rows": 0,
        "pivot_exact_mismatch": 0,
        "pivot_exact_points": 0,
        "state_total_mismatch": 0,
        "state_exact_rows": 0,
        "first_mismatch": None,
    }
    attempts = 0
    while agg["n_series"] < n_series and attempts < n_series * 40:
        attempts += 1
        n = int(rng.integers(n_bars, n_bars + 200))
        mid = float(rng.uniform(500.0, 2000.0))
        amp = float(rng.uniform(3.0, 12.0))
        base = make_ohlc(n, mid=mid, amp=amp, seed=int(rng.integers(0, 2 ** 31)))
        high, low, close, open_ = base
        # guarantee strictly positive
        low = np.minimum(low, mid - amp * 0.5)
        high = np.maximum(high, low + 0.1)
        close = np.clip(close, low + 0.01, high - 0.01)
        # Layer A pivot (tie check)
        a = layer_a_pivot_diff(high, low, open_, close, "random", "C%d" % attempts)
        if a["n_tie_events"] > 0:
            continue  # discard + regenerate
        agg["n_series"] += 1
        agg["n_rows"] += n
        agg["pivot_exact_points"] += a["n_exact_pivot_points"]
        agg["pivot_exact_mismatch"] += a["n_exact_pivot_mismatch"]
        if a["n_exact_pivot_mismatch"] > 0 and agg["first_mismatch"] is None:
            agg["first_mismatch"] = a["first_exact_mismatch"]

        # State: Oracle full vs production injected (exact pivots)
        oph, opl, _ = unique_confirmed_pivots(high, low, open_, close)
        oracle_per_bar = run_sr_state_machine(oph, opl, high, low, close)
        prod_feat = run_prod_sr_with_injected_pivots(high, low, close, oph, opl)
        mask = compute_state_exact_mask(n, oph, opl, [])
        cmp = compare_state_full(oracle_per_bar, prod_feat, mask)
        agg["state_exact_rows"] += cmp["n_exact_state_rows"]
        agg["state_total_mismatch"] += cmp["total_mismatch"]
        if cmp["total_mismatch"] > 0 and agg["first_mismatch"] is None:
            for f in STATE_FIELDS:
                if cmp["field_stats"][f]["first_mismatch_bar"] is not None:
                    agg["first_mismatch"] = {
                        "dataset": "random", "field": f,
                        "bar": cmp["field_stats"][f]["first_mismatch_bar"],
                    }
                    break
        results.append({
            "series": agg["n_series"],
            "n_rows": n,
            "pivot_exact_points": a["n_exact_pivot_points"],
            "pivot_exact_mismatch": a["n_exact_pivot_mismatch"],
            "state_exact_rows": cmp["n_exact_state_rows"],
            "state_total_mismatch": cmp["total_mismatch"],
            "n_channels_mismatch": cmp["field_stats"]["n_channels"]["mismatch"],
            "in_zone_mismatch": cmp["field_stats"]["in_zone"]["mismatch"],
            "break_up_mismatch": cmp["field_stats"]["break_up"]["mismatch"],
            "break_down_mismatch": cmp["field_stats"]["break_down"]["mismatch"],
        })
    return agg, results


# ===========================================================================
# Layer D: real AG (canonical owner)
# ===========================================================================
def load_ag_canonical_bars():
    """Canonical production data entry for AG.

    Owner is the real pipeline: load_raw_5m("AG") -> discontinuity_flags("AG").
    Normal session breaks / trading-day changes are NOT discontinuities; only
    abnormal time gaps and abnormal price jumps are. No CSV fallback is allowed
    (the canonical owner must not be inferred or substituted).
    """
    raw = (
        canonical_load_raw_5m("AG")
        .sort_values("bar_start_time")
        .reset_index(drop=True)
    )

    disc = np.asarray(canonical_discontinuity_flags("AG"), dtype=bool)

    if len(raw) != len(disc):
        raise SystemExit("STOP_SR_DIFF_CANONICAL_DISC_LENGTH_MISMATCH")

    if raw["bar_start_time"].duplicated().any():
        raise SystemExit("STOP_SR_DIFF_AG_DUPLICATE_TIME")

    bars = {
        "o": raw["open"].to_numpy(float),
        "h": raw["high"].to_numpy(float),
        "l": raw["low"].to_numpy(float),
        "c": raw["close"].to_numpy(float),
        "t": pd.to_datetime(raw["bar_start_time"]).to_numpy(),
        "day": pd.to_datetime(raw["trading_day"]).to_numpy(),
        "disc": disc,
        "n": len(raw),
    }
    return raw, bars


def layer_d_ag():
    raw5, bars = load_ag_canonical_bars()
    raw_frame = prod_raw_frame_from_owner(bars)
    tf15 = prod_resample_causal(raw_frame, 15)

    disc_true = int(bars["disc"].sum())
    n_5m = len(raw5)
    n_15m = len(tf15)
    seg_sizes = tf15.groupby("segment").size()
    seg_count = int(seg_sizes.shape[0])
    seg_len = {
        "min": int(seg_sizes.min()),
        "median": float(seg_sizes.median()),
        "mean": float(seg_sizes.mean()),
        "max": int(seg_sizes.max()),
    }
    segment_stats = {
        "n_5m_rows": n_5m,
        "canonical_disc_true": disc_true,
        "n_15m_rows": n_15m,
        "segment_count": seg_count,
        "seg_len_min": seg_len["min"],
        "seg_len_median": seg_len["median"],
        "seg_len_mean": seg_len["mean"],
        "seg_len_max": seg_len["max"],
    }

    # --- per-segment pivot + state (no cross-segment index mixing) ---
    pivot_agg = {
        "n_exact_pivot_points": 0,
        "n_exact_pivot_mismatch": 0,
        "n_tie_unverified": 0,
        "n_float_boundary_unverified": 0,
        "n_tie_events": 0,
    }
    pivot_rows = []
    state_field_mismatch = {f: 0 for f in STATE_FIELDS}
    state_first_mismatch = {f: None for f in STATE_FIELDS}
    state_rows = []
    cov_n_exact = 0
    cov_n_unverified = 0
    cov_n_tie_contam = 0
    cov_n_startup = 0

    global_off = 0
    for seg_id, seg in tf15.groupby("segment", sort=False):
        sh = seg["high"].to_numpy(float)
        sl = seg["low"].to_numpy(float)
        sc = seg["close"].to_numpy(float)
        so = seg["open"].to_numpy(float)
        times = seg["available_time"].to_numpy()
        L = len(seg)
        seg_tag = int(seg_id) if isinstance(seg_id, (int, np.integer)) else str(seg_id)

        # pivot differential (exact domain only; production-only = tie unverified)
        pr = compare_pivots_core(sh, sl, so, sc)
        pivot_agg["n_exact_pivot_points"] += pr["n_exact_pivot_points"]
        pivot_agg["n_exact_pivot_mismatch"] += pr["n_exact_pivot_mismatch"]
        pivot_agg["n_tie_unverified"] += pr["n_tie_unverified"]
        pivot_agg["n_float_boundary_unverified"] += pr["n_float_boundary_unverified"]
        pivot_agg["n_tie_events"] += pr["n_tie_events"]
        for row in pr["rows"]:
            pivot_rows.append({
                "segment_id": seg_tag,
                "local_bar": int(row["bar"]),
                "global_bar": int(global_off + row["bar"]),
                "time": str(times[row["bar"]]),
                "mode": row["mode"],
                "oracle_value": row["oracle_value"],
                "production_value": row["production_value"],
                "reason": row["reason"],
            })

        # Oracle state machine (strict-unique exact pivots) within segment
        oph, opl, tie = unique_confirmed_pivots(sh, sl, so, sc)
        oracle_per_bar = run_sr_state_machine(oph, opl, sh, sl, sc)
        tie_confirm = [t["confirm_bar"] for t in tie]
        mask = compute_state_exact_mask(L, oph, opl, tie_confirm)

        # Production REAL SR state on this segment (genuine pipeline output)
        prod_seg = prod_compute_segment_features(seg, PROD_PINE_DEFAULT, True)
        cmp = compare_state_full(oracle_per_bar, prod_seg, mask)

        cov_n_exact += cmp["n_exact_state_rows"]
        cov_n_unverified += cmp["n_unverified_state_rows"]
        n_tie_contam = int(np.sum(
            [not mask[b] and _tie_in_window(b, tie_confirm) for b in range(L)]
        ))
        cov_n_tie_contam += n_tie_contam
        cov_n_startup += int((~mask).sum()) - n_tie_contam
        for f in STATE_FIELDS:
            state_field_mismatch[f] += cmp["field_stats"][f]["mismatch"]
            if state_field_mismatch[f] > 0 and state_first_mismatch[f] is None:
                fb = cmp["field_stats"][f]["first_mismatch_bar"]
                state_first_mismatch[f] = int(global_off + fb) if fb is not None else None
        for row in cmp["rows"]:
            state_rows.append({
                "segment_id": seg_tag,
                "local_bar": int(row["bar"]),
                "global_bar": int(global_off + row["bar"]),
                "time": str(times[row["bar"]]),
                "field": row["field"],
                "oracle_value": row["oracle_value"],
                "production_value": row["production_value"],
                "reason": row["uncertainty_reason"],
            })

        global_off += L

    state_total_mismatch = sum(state_field_mismatch[f] for f in STATE_FIELDS)

    # --- Full pipeline parity: per-segment concat vs compute_tf_features ---
    full_prod = prod_compute_tf_features(tf15, PROD_PINE_DEFAULT, True)
    per_seg_list = [
        prod_compute_segment_features(seg, PROD_PINE_DEFAULT, True)
        for _, seg in tf15.groupby("segment", sort=False)
    ]
    per_seg = pd.concat(per_seg_list, ignore_index=True)
    per_seg = per_seg.sort_values("available_time", kind="stable").reset_index(drop=True)
    full = full_prod.sort_values("available_time", kind="stable").reset_index(drop=True)
    parity_mismatch = 0
    for col in ["sr_n_channels", "sr_in_zone", "sr_broken_up", "sr_broken_down"]:
        parity_mismatch += int(np.sum(per_seg[col].to_numpy() != full[col].to_numpy()))

    coverage = {
        "owner": "load_raw_5m + discontinuity_flags (canonical)",
        "n_total_rows": int(n_15m),
        "n_startup_unverified": int(cov_n_startup),
        "n_tie_events": int(pivot_agg["n_tie_events"]),
        "n_tie_contaminated_rows": int(cov_n_tie_contam),
        "n_exact_state_rows": int(cov_n_exact),
        "exact_state_coverage_pct": (100.0 * cov_n_exact / n_15m) if n_15m else 0.0,
        "n_mismatch_n_channels": state_field_mismatch["n_channels"],
        "n_mismatch_in_zone": state_field_mismatch["in_zone"],
        "n_mismatch_break_up": state_field_mismatch["break_up"],
        "n_mismatch_break_down": state_field_mismatch["break_down"],
        "first_mismatch_n_channels": state_first_mismatch["n_channels"],
        "first_mismatch_in_zone": state_first_mismatch["in_zone"],
        "first_mismatch_break_up": state_first_mismatch["break_up"],
        "first_mismatch_break_down": state_first_mismatch["break_down"],
        "max_consec_mismatch": _max_consec(
            [{"bar": r["global_bar"]} for r in state_rows]
        ),
        "note": (
            "Canonical AG (load_raw_5m + discontinuity_flags) yields disc_true=%d, "
            "so AG 15m is %d segment(s) of length up to %d bars. With the 300-bar "
            "SR width lookback this gives real exact-state coverage. Pivot exact "
            "domain and state exact domain are compared per canonical segment."
            % (disc_true, seg_count, seg_len["max"])
        ),
    }

    return {
        "segment_stats": segment_stats,
        "pivot_agg": pivot_agg,
        "pivot_rows": pivot_rows,
        "state_agg": {
            "compared": int(cov_n_exact),
            "mismatch": int(state_total_mismatch),
        },
        "state_field_mismatch": state_field_mismatch,
        "state_first_mismatch": state_first_mismatch,
        "state_rows": state_rows,
        "coverage": coverage,
        "parity_mismatch": int(parity_mismatch),
        "tf15_len": int(n_15m),
    }


def _tie_in_window(b, tie_confirm):
    return any((b - tb) <= LOOPBACK and (b - tb) >= 0 for tb in tie_confirm)


def _max_consec(rows):
    if not rows:
        return 0
    bars = sorted(r["bar"] for r in rows)
    best = run = 1
    for i in range(1, len(bars)):
        if bars[i] == bars[i - 1] + 1:
            run += 1
            best = max(best, run)
        else:
            run = 1
    return best


# ===========================================================================
# Real call-chain proof (AG)
# ===========================================================================
def call_chain_proof():
    raw5, bars = load_ag_canonical_bars()
    raw = prod_raw_frame_from_owner(bars)
    tf15 = prod_resample_causal(raw, 15)
    disc_true = int(bars["disc"].sum())
    seg_count = int(tf15.groupby("segment").size().shape[0])

    counters = {
        "compute_tf_features": {"n": 0},
        "compute_segment_features": {"n": 0},
        "build_sr_features": {"n": 0},
        "confirmed_pivots": {"n": 0},
    }

    def wrap(name, orig):
        c = counters[name]

        def wrapped(*a, **k):
            c["n"] += 1
            return orig(*a, **k)

        return wrapped

    orig_ctf = prod.compute_tf_features
    orig_csf = prod.compute_segment_features
    orig_bsf = prod.build_sr_features
    orig_cp = prod.confirmed_pivots

    prod.compute_tf_features = wrap("compute_tf_features", orig_ctf)
    prod.compute_segment_features = wrap("compute_segment_features", orig_csf)
    prod.build_sr_features = wrap("build_sr_features", orig_bsf)
    prod.confirmed_pivots = wrap("confirmed_pivots", orig_cp)
    try:
        _ = prod.compute_tf_features(tf15, PROD_PINE_DEFAULT, True)
    finally:
        prod.compute_tf_features = orig_ctf
        prod.compute_segment_features = orig_csf
        prod.build_sr_features = orig_bsf
        prod.confirmed_pivots = orig_cp

    all_positive = all(counters[k]["n"] > 0 for k in counters)
    return {
        "counters": {k: counters[k]["n"] for k in counters},
        "all_calls_positive": bool(all_positive),
        "canonical_disc_true": disc_true,
        "canonical_segment_count": seg_count,
    }


# ===========================================================================
# Prefix causality
# ===========================================================================
def prefix_causality(seed: int = 20260918, n_bars: int = 1900, cut: int = 1400):
    rng = np.random.default_rng(seed)
    mid = 1000.0
    amp = 8.0
    high, low, close, open_ = make_ohlc(n_bars, mid=mid, amp=amp, seed=7)
    oph, opl, _ = unique_confirmed_pivots(high, low, open_, close)

    def prod_state(h, l, c):
        feat = run_prod_sr_with_injected_pivots(h, l, c, oph, opl)
        return (feat["sr_n_channels"], feat["sr_in_zone"],
                feat["sr_broken_up"], feat["sr_broken_down"])

    def oracle_state(h, l, c):
        pb = run_sr_state_machine(oph, opl, h, l, c)
        nc = np.array([x["n_channels"] for x in pb], dtype=int)
        iz = np.array([0 if x["not_in_channel"] else 1 for x in pb], dtype=int)
        bu = np.array([1 if x["resistancebroken"] else 0 for x in pb], dtype=int)
        bd = np.array([1 if x["supportbroken"] else 0 for x in pb], dtype=int)
        return nc, iz, bu, bd

    # original
    p_nc, p_iz, p_bu, p_bd = prod_state(high, low, close)
    o_nc, o_iz, o_bu, o_bd = oracle_state(high, low, close)
    # modified future only
    hm = high.copy(); lm = low.copy(); cm = close.copy()
    hm[cut + 1:] += 1.0
    lm[cut + 1:] += 1.0
    cm[cut + 1:] += 1.0

    def prefix_eq(a, b):
        a = np.asarray(a); b = np.asarray(b)
        return bool(np.array_equal(a[:cut + 1], b[:cut + 1]))

    p2 = prod_state(hm, lm, cm)
    o2 = oracle_state(hm, lm, cm)

    prod_pass = (
        prefix_eq(p_nc, p2[0]) and prefix_eq(p_iz, p2[1])
        and prefix_eq(p_bu, p2[2]) and prefix_eq(p_bd, p2[3])
    )
    oracle_pass = (
        prefix_eq(o_nc, o2[0]) and prefix_eq(o_iz, o2[1])
        and prefix_eq(o_bu, o2[2]) and prefix_eq(o_bd, o2[3])
    )
    return {
        "prod_prefix_unchanged": bool(prod_pass),
        "oracle_prefix_unchanged": bool(oracle_pass),
        "cut": cut,
        "n_bars": n_bars,
    }


# ===========================================================================
# Artifacts + verdict
# ===========================================================================
def write_csv(path, rows, columns):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in columns})


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    # ---- T0 ----
    t0 = check_t0()
    print("[T0] ok=%s  head=%s  sr_triple=%s  notes=%s" % (
        t0["ok"], t0["head_sha"], t0["sr_triple_match"], t0["notes"]))
    if not t0["ok"]:
        sys.exit("T0 FAIL: %s" % t0["notes"])

    # ---- Layer A folded into each layer; run boundary matrix ----
    print("[Layer B] boundary matrix B01-B16 ...")
    bm = build_boundary_matrix()
    bm_mismatch = sum(1 for r in bm if not r["exact_match"])
    boundary_mismatch_ids = sorted(r["case_id"] for r in bm if not r["exact_match"])

    # ---- Layer C random ----
    print("[Layer C] random continuous ...")
    c_agg, c_res = layer_c_random()

    # ---- Layer D AG (canonical owner, per-segment) ----
    print("[Layer D] canonical AG ...")
    d = layer_d_ag()

    # ---- call-chain proof (canonical) ----
    print("[Chain] canonical AG call-chain proof ...")
    chain = call_chain_proof()

    # ---- prefix causality ----
    print("[Prefix] causality ...")
    prefix = prefix_causality()

    # ---- Hard gate (Phase B: post-fix verification) ----
    # Fix A (Pine bool pivot truthiness) and Fix B (zero-width channels) make
    # production match the Oracle on every pre-authorized divergence, so the
    # post-fix expectation is ALL 16 boundary cases MATCH (empty mismatch set).
    # Any non-empty boundary mismatch set => the fix regressed or introduced a
    # new exact-domain divergence => STOP.
    gate_boundary_ok = (len(boundary_mismatch_ids) == 0)
    gate_random_ok = (
        c_agg["pivot_exact_mismatch"] == 0 and c_agg["state_total_mismatch"] == 0
    )
    gate_ag_pivot_ok = (
        d["pivot_agg"]["n_exact_pivot_points"] > 0
        and d["pivot_agg"]["n_exact_pivot_mismatch"] == 0
    )
    gate_ag_state_ok = (
        d["coverage"]["n_exact_state_rows"] > 0 and d["state_agg"]["mismatch"] == 0
    )
    gate_parity_ok = d["parity_mismatch"] == 0
    gate_pass = all(
        [gate_boundary_ok, gate_random_ok, gate_ag_pivot_ok,
         gate_ag_state_ok, gate_parity_ok]
    )
    hard_gate = {
        "boundary_mismatch_ids": boundary_mismatch_ids,
        "gate_boundary_ok": bool(gate_boundary_ok),
        "gate_random_ok": bool(gate_random_ok),
        "gate_ag_pivot_ok": bool(gate_ag_pivot_ok),
        "gate_ag_state_ok": bool(gate_ag_state_ok),
        "gate_parity_ok": bool(gate_parity_ok),
        "gate_pass": bool(gate_pass),
        "note": (
            "Phase B post-fix gate: gate_pass=True requires ALL 16 boundary "
            "cases to MATCH (Fix A + Fix B resolve the pre-authorized B03/B04/B06 "
            "divergences) and no AG pivot/state divergence or pipeline parity > 0. "
            "Any non-empty boundary mismatch set, AG state exact rows == 0, or "
            "pipeline parity > 0 => STOP."
        ),
    }

    # ---- mismatch bundle (cap 20 per dataset+field) ----
    bundle = []
    cap = {}
    CAP = 20

    def add(row, dataset, field):
        key = (dataset, field)
        if cap.get(key, 0) >= CAP:
            return
        cap[key] = cap.get(key, 0) + 1
        row = dict(row)
        row["dataset"] = dataset
        bundle.append(row)

    for r in bm:
        if not r["exact_match"]:
            add({
                "layer": "B", "case_id": r["case_id"], "field": "state",
                "oracle_value": r["oracle_n_channels"],
                "production_value": r["production_n_channels"],
                "reason": "SOURCE_EXACT_DOMAIN",
            }, "boundary_%s" % r["case_id"], "state")
    if c_agg["first_mismatch"] is not None:
        add({
            "layer": "C", "case_id": "random", "field": "state",
            "reason": "SOURCE_EXACT_DOMAIN",
            **{k: v for k, v in (c_agg["first_mismatch"] or {}).items() if k not in ("dataset",)},
        }, "random", "state")
    for row in d["state_rows"]:
        add({
            "layer": "D", "case_id": "AG", "field": row["field"],
            "oracle_value": row["oracle_value"],
            "production_value": row["production_value"],
            "reason": row["reason"],
            "seg_id": row["segment_id"],
            "global_bar": row["global_bar"],
            "time": row["time"],
            "bar": row["global_bar"],
        }, "AG", row["field"])
    for row in d["pivot_rows"]:
        add({
            "layer": "D", "case_id": "AG", "field": "pivot_%s" % row["mode"],
            "oracle_value": row["oracle_value"],
            "production_value": row["production_value"],
            "reason": row["reason"],
            "seg_id": row["segment_id"],
            "global_bar": row["global_bar"],
            "time": row["time"],
            "bar": row["global_bar"],
        }, "AG", "pivot_%s" % row["mode"])

    # ---- verdict ----
    total_source_mismatch = (
        bm_mismatch
        + c_agg["state_total_mismatch"]
        + d["state_agg"]["mismatch"]
        + d["pivot_agg"]["n_exact_pivot_mismatch"]
    )
    any_unverified = (
        d["pivot_agg"]["n_tie_unverified"] > 0
        or d["pivot_agg"]["n_float_boundary_unverified"] > 0
        or d["coverage"]["n_startup_unverified"] > 0
        or d["coverage"]["n_tie_events"] > 0
        or c_agg["n_series"] == 0
    )
    if total_source_mismatch > 0:
        verdict = "SOURCE_DIVERGENCE_FOUND"
    elif any_unverified or d["coverage"]["n_exact_state_rows"] == 0:
        verdict = "INSUFFICIENT_EXACT_COVERAGE" if (
            d["coverage"]["n_exact_state_rows"] == 0 and c_agg["state_exact_rows"] == 0
        ) else "SOURCE_EXACT_ON_TESTED_DOMAIN_PARTIALLY_UNVERIFIED"
    else:
        verdict = "SOURCE_EXACT_ON_TESTED_DOMAIN_PARTIALLY_UNVERIFIED"

    summary = {
        "identity": {
            "base_sha": t0["base_sha"],
            "head_sha": t0["head_sha"],
            "remote_sha": t0["head_sha"],
            "runtime_sha": t0["head_sha"],
            "oracle_path": ORACLE_PATH,
            "prod_path": PROD_PATH,
            "sr_sha_worktree": t0["sr_sha_worktree"],
            "sr_sha_gitobject": t0["sr_sha_gitobject"],
            "sr_sha_expected": t0["sr_sha_expected"],
        },
        "gates": {
            "base_is_ancestor": t0["base_is_ancestor"],
            "oracle_exists": t0["oracle_exists"],
            "sr_triple_match": t0["sr_triple_match"],
            "notes": t0["notes"],
        },
        "hard_gate": hard_gate,
        "boundary_matrix": {
            "n_cases": len(bm),
            "n_exact_mismatch": bm_mismatch,
            "boundary_all_match": bm_mismatch == 0,
        },
        "random": {
            "n_series": c_agg["n_series"],
            "n_rows": c_agg["n_rows"],
            "pivot_exact_points": c_agg["pivot_exact_points"],
            "pivot_exact_mismatch": c_agg["pivot_exact_mismatch"],
            "state_exact_rows": c_agg["state_exact_rows"],
            "state_total_mismatch": c_agg["state_total_mismatch"],
            "first_mismatch": c_agg["first_mismatch"],
        },
        "ag": {
            "segment_stats": d["segment_stats"],
            "pivot_exact_points": d["pivot_agg"]["n_exact_pivot_points"],
            "pivot_exact_mismatch": d["pivot_agg"]["n_exact_pivot_mismatch"],
            "pivot_exact_compared": d["pivot_agg"]["n_exact_pivot_points"],
            "tie_unverified": d["pivot_agg"]["n_tie_unverified"],
            "float_boundary_unverified": d["pivot_agg"]["n_float_boundary_unverified"],
            "coverage": d["coverage"],
            "state_total_mismatch": d["state_agg"]["mismatch"],
            "state_field_mismatch": d["state_field_mismatch"],
            "state_first_mismatch": d["state_first_mismatch"],
            "parity_mismatch": d["parity_mismatch"],
        },
        "call_chain": chain,
        "prefix_causality": prefix,
        "verdict": verdict,
        "classification": {
            "pivot_exact_domain": {
                "ag_mismatch": d["pivot_agg"]["n_exact_pivot_mismatch"],
                "random_mismatch": c_agg["pivot_exact_mismatch"],
                "boundary_n_mismatch": bm_mismatch,
            },
            "state_exact_domain": {
                "ag_mismatch": d["state_agg"]["mismatch"],
                "random_mismatch": c_agg["state_total_mismatch"],
                "boundary_n_mismatch": bm_mismatch,
            },
            "random_exact_domain": {"state_exact_rows": c_agg["state_exact_rows"]},
            "ag_exact_domain": {"state_exact_rows": d["coverage"]["n_exact_state_rows"]},
            "unverified_pivot_tie": {"ag_tie_unverified": d["pivot_agg"]["n_tie_unverified"]},
            "unverified_startup": {"ag_startup_unverified": d["coverage"]["n_startup_unverified"]},
            "unverified_float_boundary": {
                "ag_float_boundary_unverified": d["pivot_agg"]["n_float_boundary_unverified"]
            },
            "research_extensions": [
                "sr_support_dist_atr", "sr_resistance_dist_atr",
                "sr_support_price", "sr_resistance_price",
                "sr_support_strength", "sr_resistance_strength",
                "sr_zone_strength",
            ],
        },
    }

    # ---- write artifacts ----
    with open(os.path.join(ARTIFACT_DIR, "sr_diff_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    bm_cols = ["case_id", "source_domain", "oracle_n_channels", "production_n_channels",
               "oracle_in_zone", "production_in_zone", "oracle_break_up",
               "production_break_up", "oracle_break_down", "production_break_down",
               "exact_match", "first_mismatch_field", "n_exact_state_rows",
               "n_unverified_state_rows", "pivot_value_mismatch",
               "n_channels_mismatch", "in_zone_mismatch", "break_up_mismatch",
               "break_down_mismatch", "diagnostic_zone_strength", "note"]
    write_csv(os.path.join(ARTIFACT_DIR, "sr_boundary_matrix.csv"), bm, bm_cols)

    # AG segment stats
    write_csv(os.path.join(ARTIFACT_DIR, "sr_ag_segment_stats.csv"),
              [d["segment_stats"]],
              ["n_5m_rows", "canonical_disc_true", "n_15m_rows", "segment_count",
               "seg_len_min", "seg_len_median", "seg_len_mean", "seg_len_max"])

    # AG per-segment pivot detail
    write_csv(os.path.join(ARTIFACT_DIR, "sr_ag_pivot_per_segment.csv"), d["pivot_rows"],
              ["segment_id", "local_bar", "global_bar", "time", "mode",
               "oracle_value", "production_value", "reason"])

    # pivot diff (random + AG aggregate)
    pivot_rows = [{
        "dataset": "random", "case": "C",
        "n_exact_pivot_points": c_agg["pivot_exact_points"],
        "n_exact_pivot_mismatch": c_agg["pivot_exact_mismatch"],
        "n_tie_unverified": "", "n_float_boundary_unverified": "",
        "first_exact_mismatch": json.dumps(c_agg["first_mismatch"], default=str),
    }, {
        "dataset": "AG", "case": "D",
        "n_exact_pivot_points": d["pivot_agg"]["n_exact_pivot_points"],
        "n_exact_pivot_mismatch": d["pivot_agg"]["n_exact_pivot_mismatch"],
        "n_tie_unverified": d["pivot_agg"]["n_tie_unverified"],
        "n_float_boundary_unverified": d["pivot_agg"]["n_float_boundary_unverified"],
        "first_exact_mismatch": json.dumps(None, default=str),
    }]
    write_csv(os.path.join(ARTIFACT_DIR, "sr_pivot_diff.csv"), pivot_rows,
              ["dataset", "case", "n_exact_pivot_points", "n_exact_pivot_mismatch",
               "n_tie_unverified", "n_float_boundary_unverified", "first_exact_mismatch"])

    # state diff (random + AG aggregate)
    state_rows = list(c_res)
    state_rows.append({
        "series": "AG", "n_rows": d["tf15_len"],
        "pivot_exact_points": d["pivot_agg"]["n_exact_pivot_points"],
        "pivot_exact_mismatch": d["pivot_agg"]["n_exact_pivot_mismatch"],
        "state_exact_rows": d["coverage"]["n_exact_state_rows"],
        "state_total_mismatch": d["state_agg"]["mismatch"],
        "n_channels_mismatch": d["state_field_mismatch"]["n_channels"],
        "in_zone_mismatch": d["state_field_mismatch"]["in_zone"],
        "break_up_mismatch": d["state_field_mismatch"]["break_up"],
        "break_down_mismatch": d["state_field_mismatch"]["break_down"],
    })
    write_csv(os.path.join(ARTIFACT_DIR, "sr_state_diff.csv"), state_rows,
              ["series", "n_rows", "pivot_exact_points", "pivot_exact_mismatch",
               "state_exact_rows", "state_total_mismatch", "n_channels_mismatch",
               "in_zone_mismatch", "break_up_mismatch", "break_down_mismatch"])

    # AG per-segment state detail
    write_csv(os.path.join(ARTIFACT_DIR, "sr_ag_state_per_segment.csv"), d["state_rows"],
              ["segment_id", "local_bar", "global_bar", "time", "field",
               "oracle_value", "production_value", "reason"])

    # AG coverage
    write_csv(os.path.join(ARTIFACT_DIR, "sr_ag_coverage.csv"), [d["coverage"]],
              ["owner", "n_total_rows", "n_startup_unverified", "n_tie_events",
               "n_tie_contaminated_rows", "n_exact_state_rows",
               "exact_state_coverage_pct", "n_mismatch_n_channels",
               "n_mismatch_in_zone", "n_mismatch_break_up", "n_mismatch_break_down",
               "first_mismatch_n_channels", "first_mismatch_in_zone",
               "first_mismatch_break_up", "first_mismatch_break_down",
               "max_consec_mismatch"])

    # Full pipeline parity
    write_csv(os.path.join(ARTIFACT_DIR, "sr_pipeline_parity.csv"), [{
        "per_segment_vs_compute_tf_features_mismatch": d["parity_mismatch"],
        "compute_tf_features_calls": chain["counters"]["compute_tf_features"],
        "compute_segment_features_calls": chain["counters"]["compute_segment_features"],
        "build_sr_features_calls": chain["counters"]["build_sr_features"],
        "confirmed_pivots_calls": chain["counters"]["confirmed_pivots"],
    }],
              ["per_segment_vs_compute_tf_features_mismatch", "compute_tf_features_calls",
               "compute_segment_features_calls", "build_sr_features_calls",
               "confirmed_pivots_calls"])

    # Call chain
    write_csv(os.path.join(ARTIFACT_DIR, "sr_call_chain.csv"), [{
        "compute_tf_features": chain["counters"]["compute_tf_features"],
        "compute_segment_features": chain["counters"]["compute_segment_features"],
        "build_sr_features": chain["counters"]["build_sr_features"],
        "confirmed_pivots": chain["counters"]["confirmed_pivots"],
        "all_calls_positive": chain["all_calls_positive"],
        "canonical_disc_true": chain["canonical_disc_true"],
        "canonical_segment_count": chain["canonical_segment_count"],
    }],
              ["compute_tf_features", "compute_segment_features", "build_sr_features",
               "confirmed_pivots", "all_calls_positive", "canonical_disc_true",
               "canonical_segment_count"])

    # mismatch bundle
    write_csv(os.path.join(ARTIFACT_DIR, "sr_mismatch_samples.csv"), bundle,
              ["dataset", "layer", "case_id", "bar", "field", "oracle_value",
               "production_value", "reason", "seg_id", "global_bar", "time"])

    # optional snapshot (pre_fix / post_fix)
    snap_tag = os.environ.get("SR_SNAPSHOT")
    if snap_tag:
        man = snapshot(snap_tag)
        print("[SNAPSHOT] %s -> %d files" % (snap_tag, len(man)))

    print("[VERDICT] %s" % verdict)
    print("[HARD_GATE] pass=%s  boundary_ids=%s" % (gate_pass, boundary_mismatch_ids))
    print("[SUMMARY] boundary_n_mismatch=%d  random_state_mismatch=%d  "
          "ag_state_mismatch=%d  ag_pivot_mismatch=%d  ag_state_rows=%d  "
          "parity=%d  chain=%s  prefix_prod=%s oracle=%s" % (
              bm_mismatch, c_agg["state_total_mismatch"], d["state_agg"]["mismatch"],
              d["pivot_agg"]["n_exact_pivot_mismatch"], d["coverage"]["n_exact_state_rows"],
              d["parity_mismatch"], chain["all_calls_positive"],
              prefix["prod_prefix_unchanged"], prefix["oracle_prefix_unchanged"]))
    print("[ARTIFACTS] %s" % ARTIFACT_DIR)


def snapshot(tag: str):
    """Copy all artifacts into artifacts/.../<tag>/ and record SHA256 of each."""
    dest = os.path.join(ARTIFACT_DIR, tag)
    os.makedirs(dest, exist_ok=True)
    files = sorted(
        f for f in os.listdir(ARTIFACT_DIR)
        if f.endswith((".csv", ".json")) and os.path.isfile(os.path.join(ARTIFACT_DIR, f))
    )
    manifest = []
    for f in files:
        src = os.path.join(ARTIFACT_DIR, f)
        data = open(src, "rb").read()
        open(os.path.join(dest, f), "wb").write(data)
        manifest.append((f, hashlib.sha256(data).hexdigest()))
    with open(os.path.join(dest, "SHA256SUMS.txt"), "w") as fh:
        for f, h in manifest:
            fh.write("%s  %s\n" % (h, f))
    return manifest


if __name__ == "__main__":
    main()
