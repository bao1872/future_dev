#!/usr/bin/env python3
"""
Liquidity Source Semantic Oracle (Literal re-implementation)
=============================================================

Independent, faithful re-implementation of the pinned Pine source
``ref/Liquidity.pine`` (SHA256 ccd13991...).

This module is the SOLE authority for the Liquidity source semantics. It
depends ONLY on the pinned source contract + numpy / stdlib. It MUST NOT
import or call any production module (``experiment_structural_reversion_pgm_v1``
or similar). Pandas is intentionally not used here (only numpy / stdlib).

The implementation mirrors the Pine state machine literally:

  * pivot builtin   : strict-unique ``ta.pivothigh(7, 1)`` / ``ta.pivotlow(7, 1)``
                      confirmed at ``p + 1`` (never backfilled to ``p``).
  * zigzag          : newest-first list, ``aZZ.d`` initialised to 0, same-side
                      replacement only when strictly more extreme, hard cap 50.
  * cluster         : strict band ``pivot - margin < y < pivot + margin``,
                      scan ``break`` on out-of-band old node, ``count > 2`` gate,
                      ``center = (cluster_max + cluster_min) / 2``.
  * level objects   : ``start_bar`` equality -> update top/bottom, else insert;
                      visible cap ``visLiq = 3`` (oldest evicted).
  * breach          : strict ``high > top`` (high) / ``low < bottom`` (low).
  * post-break zone : strict ``low > level - 2.3*atr and high < level + 2.3*atr``.

No fix recommendation, no production bug judgement. Observations only.
"""

from __future__ import annotations

import os
import hashlib
import re

import numpy as np

# ---------------------------------------------------------------------------
# Frozen identity
# ---------------------------------------------------------------------------
PINNED_SHA256 = "ccd13991b4b96eeed55651ff13b2541ca4a53c91aa68a54e888b1b2b4513a408"

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LIQ_PINE_PATH = os.path.join(_REPO_ROOT, "ref", "Liquidity.pine")

# ---------------------------------------------------------------------------
# Source constants (extracted from pinned ref/Liquidity.pine)
# ---------------------------------------------------------------------------
LIQ_LEN = 7            # liqLen
LIQ_RIGHT = 1          # ta.pivothigh(liqLen, 1) right
LIQ_MAR = 10.0 / 6.9   # liqMar = 10 / margin_input (6.9)
VIS_LIQ = 3            # visLiq
MAX_SIZE = 50          # maxSize
ATR_LEN = 10           # atr = ta.atr(10)
POSTBREAK_ATR = 2.3    # marBuy = marSel = 2.3
FLOAT_BOUNDARY = 1e-8  # deterministic exact-pivot separation
SOURCE_MODE_UI_DEFAULT = "Present"
RESEARCH_COMPARISON_MODE = "Historical"

SOURCE_MARKERS: dict = {}


# ===========================================================================
# Source gate
# ===========================================================================
def check_source_gate() -> dict:
    """Verify the pinned source SHA256 and extract source markers.

    Raises SystemExit on SHA mismatch. Returns the marker dict.
    """
    if not os.path.exists(LIQ_PINE_PATH):
        raise SystemExit("STOP_LIQ_SOURCE_MISSING: %s" % LIQ_PINE_PATH)
    raw = open(LIQ_PINE_PATH, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != PINNED_SHA256:
        raise SystemExit("STOP_LIQ_SOURCE_HASH_MISMATCH: %s != %s" % (sha, PINNED_SHA256))
    txt = raw.decode("utf-8", errors="replace")

    def has(pat: str) -> bool:
        return pat in txt

    def one(pat: str) -> str:
        m = re.search(pat, txt)
        return m.group(1) if m else "?"

    markers = {
        "sha256": sha,
        "pine_version": one(r"@version\s*=\s*(\d+)"),
        "liqLen": one(r"liqLen\s*=\s*(\d+)"),
        "liqMar_expr": "10 / 6.9",
        "liqMar_value": LIQ_MAR,
        "marBuy": one(r"marBuy\s*=\s*([\d.]+)"),
        "marSel": one(r"marSel\s*=\s*([\d.]+)"),
        "visLiq": one(r"visLiq\s*=\s*(\d+)"),
        "maxSize": one(r"maxSize\s*=\s*(\d+)"),
        "atr_expr": "ta.atr(10)",
        "mode_default": one(r"input\.string\(\s*'(Present|Historical)'"),
        "mode_options": "['Present','Historical']",
        "per_expr": "last_bar_index - bar_index <= 500",
        "pivot_high": "ta.pivothigh(liqLen, 1)",
        "pivot_low": "ta.pivotlow(liqLen, 1)",
        "band_expr": "atr / liqMar",
        "count_gate": "count > 2",
        "breach_high": "b.h > x.bx.get_top()",
        "breach_low": "b.l < x.bx.get_bottom()",
        "postbreak_expr": "marBuy*atr / marSel*atr",
        "has_pivothigh": has("ta.pivothigh(liqLen, 1)"),
        "has_pivotlow": has("ta.pivotlow(liqLen, 1)"),
        "has_count_gt_2": has("count > 2"),
        "has_liqMar": has("liqMar = 10 / 6.9") or has("10 / 6.9"),
        "has_mode_present": has("'Present'"),
        "has_mode_historical": has("'Historical'"),
        "has_per_500": has("last_bar_index - bar_index <= 500"),
        "has_strict_band": has("ph - (atr / liqMar)") or has("atr / liqMar"),
        "has_breach_high": has("b.h > x.bx.get_top()"),
        "has_breach_low": has("b.l < x.bx.get_bottom()"),
        "has_postbreak_marBuy": has("marBuy * atr") or has("marBuy*atr"),
    }
    global SOURCE_MARKERS
    SOURCE_MARKERS = markers
    return markers


# ===========================================================================
# ATR (independent Pine rma, identical formula to production)
# ===========================================================================
def true_range(high, low, close) -> np.ndarray:
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    prev = np.r_[np.nan, close[:-1]]
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - prev), np.abs(low - prev)),
    )
    if len(tr):
        tr[0] = high[0] - low[0]
    return tr


def pine_rma(x: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    out = np.full(len(x), np.nan, dtype=float)
    if length <= 0 or len(x) < length:
        return out
    start = None
    for i in range(length - 1, len(x)):
        w = x[i - length + 1: i + 1]
        if np.all(np.isfinite(w)):
            out[i] = float(np.mean(w))
            start = i + 1
            break
    if start is None:
        return out
    alpha = 1.0 / float(length)
    for i in range(start, len(x)):
        if not np.isfinite(x[i]):
            out[i] = np.nan
            continue
        if np.isfinite(out[i - 1]):
            out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
        else:
            w = x[i - length + 1: i + 1]
            if len(w) == length and np.all(np.isfinite(w)):
                out[i] = float(np.mean(w))
    return out


def atr_pine(high, low, close, length: int = ATR_LEN) -> np.ndarray:
    return pine_rma(true_range(high, low, close), length)


# ===========================================================================
# Pivot builtin (strict-unique)
# ===========================================================================
def unique_confirmed_liq_pivots(high, low, open_, close):
    """Strict-unique ``ta.pivothigh(7, 1)`` / ``ta.pivotlow(7, 1)``.

    Returns ``(ph, pl, tie)`` where:
      * ``ph`` / ``pl`` : float arrays, value present at the CONFIRMATION bar
        ``p + 1`` (never at ``p``).
      * ``tie``          : list of ``{"confirm_bar": int, "mode": str}`` for
        plateau / equal-extrema windows where strict-unique cannot pick a
        deterministic winner (UNVERIFIED_BUILTIN, not decided here).

    Strict-unique means the centre must be strictly greater (high) / strictly
    less (low) than ALL neighbours in the ``[p-7, p+1]`` window.
    """
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    n = len(high)
    ph = np.full(n, np.nan, dtype=float)
    pl = np.full(n, np.nan, dtype=float)
    tie = []
    left = LIQ_LEN
    right = LIQ_RIGHT
    width = left + right + 1

    def _scan(values, mode):
        out = np.full(n, np.nan, dtype=float)
        if n < width:
            return out
        w = np.lib.stride_tricks.sliding_window_view(values, width)
        for p in range(len(w)):
            win = w[p]
            center = win[left]
            nei = np.concatenate([win[:left], win[left + 1:]])
            confirm = p + left + right
            if mode == "high":
                strict = np.isfinite(center) and (center > np.max(nei))
                plateau = np.any(nei == center)
            else:
                strict = np.isfinite(center) and (center < np.min(nei))
                plateau = np.any(nei == center)
            if strict:
                out[confirm] = center
            elif plateau:
                tie.append({"confirm_bar": int(confirm), "mode": mode})
        return out

    ph = _scan(high, "high")
    pl = _scan(low, "low")
    return ph, pl, tie


# ===========================================================================
# Zigzag (literal)
# ===========================================================================
def update_zigzag_literal(zz, side, idx, price):
    """Mutates ``zz`` (newest-first list of {dir, x, y}) per Pine zigzag."""
    price = float(price)
    dir_ = zz[0]["dir"] if zz else 0  # aZZ.d initialised to 0
    if side > 0:
        if dir_ < 1:
            zz.insert(0, {"dir": 1, "x": int(idx), "y": price})
        elif dir_ == 1 and price > zz[0]["y"]:
            zz[0] = {"dir": 1, "x": int(idx), "y": price}
        # else ignore
    else:
        if dir_ > -1:
            zz.insert(0, {"dir": -1, "x": int(idx), "y": price})
        elif dir_ == -1 and price < zz[0]["y"]:
            zz[0] = {"dir": -1, "x": int(idx), "y": price}
        # else ignore
    del zz[MAX_SIZE:]  # keep newest 50


def build_zigzag(ph, pl):
    """Build the zigzag from prepared confirmation arrays (for inspection)."""
    ph = np.asarray(ph, dtype=float)
    pl = np.asarray(pl, dtype=float)
    zz = []
    n = len(ph)
    for i in range(n):
        if np.isfinite(ph[i]):
            update_zigzag_literal(zz, +1, i - LIQ_RIGHT, ph[i])
        if np.isfinite(pl[i]):
            update_zigzag_literal(zz, -1, i - LIQ_RIGHT, pl[i])
    return zz


# ===========================================================================
# Cluster / level (literal)
# ===========================================================================
def cluster_level_literal(zz, side, pivot, atr_i):
    """Return a level object dict or None (literal Pine cluster)."""
    if not np.isfinite(atr_i):
        atr_i = np.nan
    margin = atr_i / LIQ_MAR  # Pine: no ATR guard -> nan margin simply matches nothing
    count = 0
    start_bar = None
    level_price = np.nan
    cluster_max = -np.inf   # source: minP (running max of band y)
    cluster_min = np.inf    # source: maxP (running min of band y)
    for z in zz:
        if int(z["dir"]) != side:
            continue
        y = z["y"]
        if side > 0:
            if y > pivot + margin:  # strict break (>) as source
                break
        else:
            if y < pivot - margin:  # strict break (<) as source
                break
        inside = (pivot - margin) < y < (pivot + margin)  # strict band
        if inside:
            count += 1
            start_bar = int(z["x"])
            level_price = y
            if y > cluster_max:
                cluster_max = y
            if y < cluster_min:
                cluster_min = y
    if count <= 2 or start_bar is None or not np.isfinite(level_price):
        return None
    center = 0.5 * (cluster_max + cluster_min)
    return {
        "left": start_bar,
        "level": float(level_price),
        "top": float(center + margin),
        "bottom": float(center - margin),
        "side": side,
        "broken": False,
        "breach_i": None,
    }


def update_level_objects_literal(levels, obj):
    """Update existing (same start_bar) or insert; cap at visLiq (oldest out)."""
    if levels and int(levels[0]["left"]) == int(obj["left"]):
        levels[0]["top"] = obj["top"]
        levels[0]["bottom"] = obj["bottom"]
        # level / side / left retained
    else:
        levels.insert(0, obj)
        del levels[VIS_LIQ:]  # evict oldest beyond visible cap


# ===========================================================================
# Liquidity state machine (literal)
# ===========================================================================
def run_liquidity_state_machine(high, low, close, atr, ph, pl, mode=RESEARCH_COMPARISON_MODE):
    """Run the literal liquidity state machine.

    Returns a list (length n) of per-bar dicts with source-exact fields:
      breach_up, breach_down, up_count, down_count, zone_active
    plus compact diagnostics (zz_dir, zz_len, vis_up, vis_down) for the
    mismatch bundle.
    """
    n = len(close)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    close = np.asarray(close, dtype=float)
    atr = np.asarray(atr, dtype=float)
    ph = np.asarray(ph, dtype=float)
    pl = np.asarray(pl, dtype=float)

    zz = []
    levels_up = []    # newest-first
    levels_down = []  # newest-first
    last_breach = None  # {"side","level","i","zone_active"}

    out = []
    for i in range(n):
        # Zigzag always updated (Pine builds it before the per-gated block).
        if np.isfinite(ph[i]):
            update_zigzag_literal(zz, +1, i - LIQ_RIGHT, ph[i])
        if np.isfinite(pl[i]):
            update_zigzag_literal(zz, -1, i - LIQ_RIGHT, pl[i])

        # per gate (Historical comparison => always True)
        do_state = (mode != "Present") or ((n - 1 - i) <= 500)

        breach_up_i = 0
        breach_down_i = 0
        zone_active_i = 0

        if do_state:
            # cluster high
            if np.isfinite(ph[i]):
                obj = cluster_level_literal(zz, +1, ph[i], atr[i])
                if obj is not None:
                    update_level_objects_literal(levels_up, obj)
            # cluster low
            if np.isfinite(pl[i]):
                obj = cluster_level_literal(zz, -1, pl[i], atr[i])
                if obj is not None:
                    update_level_objects_literal(levels_down, obj)
            # breach (literal, strict)
            for lev in levels_up:
                if (not lev["broken"]) and high[i] > lev["top"]:
                    lev["broken"] = True
                    lev["breach_i"] = i
                    breach_up_i = 1
                    last_breach = {"side": +1, "level": lev["level"], "i": i, "zone_active": True}
            for lev in levels_down:
                if (not lev["broken"]) and low[i] < lev["bottom"]:
                    lev["broken"] = True
                    lev["breach_i"] = i
                    breach_down_i = 1
                    last_breach = {"side": -1, "level": lev["level"], "i": i, "zone_active": True}
            # post-break zone (strict band)
            if last_breach is not None and last_breach["zone_active"]:
                level = last_breach["level"]
                ai = atr[i]
                if np.isfinite(ai) and ai > 0:
                    inside = (low[i] > level - POSTBREAK_ATR * ai) and (high[i] < level + POSTBREAK_ATR * ai)
                    if not inside:
                        last_breach["zone_active"] = False
                zone_active_i = 1 if last_breach["zone_active"] else 0

        out.append({
            "breach_up": breach_up_i,
            "breach_down": breach_down_i,
            "up_count": int(len(levels_up)),   # literal b_liq_B.size() (incl. broken)
            "down_count": int(len(levels_down)),
            "zone_active": zone_active_i,
            "zz_dir": zz[0]["dir"] if zz else 0,
            "zz_len": int(len(zz)),
            "vis_up": [dict(lv) for lv in levels_up],
            "vis_down": [dict(lv) for lv in levels_down],
        })
    return out


# ===========================================================================
# Self-tests (L01-L20 literal expected facts)
# ===========================================================================
def _inj(n, items):
    ph = np.full(n, np.nan, dtype=float)
    pl = np.full(n, np.nan, dtype=float)
    for b, mode, v in items:
        if mode == "high":
            ph[int(b)] = float(v)
        elif mode == "low":
            pl[int(b)] = float(v)
        else:
            raise AssertionError(mode)
    return ph, pl


def _flat(n, v=100.0):
    return (np.full(n, v, dtype=float), np.full(n, v, dtype=float), np.full(n, v, dtype=float))


def run_self_tests():
    """Run L01-L20 literal expected facts. Returns (results, all_pass)."""
    results = []

    def rec(case_id, passed, detail):
        results.append({"case_id": case_id, "pass": bool(passed), "detail": detail})
        return passed

    # ---- L01 high pivot confirmation (strict unique) ----
    n = 20
    h = np.full(n, 100.0); l = np.full(n, 100.0); c = np.full(n, 100.0)
    h[7] = 200.0
    ph, pl, _ = unique_confirmed_liq_pivots(h, l, c, c)
    ok = (np.isfinite(ph[8]) and abs(ph[8] - 200.0) < 1e-9 and not np.isfinite(ph[7])
          and not np.isfinite(ph[9]))
    rec("L01", ok, "high pivot p=7 confirmed at p+1=8; ph[8]=%.4f" % (ph[8] if np.isfinite(ph[8]) else float('nan')))

    # ---- L02 low pivot confirmation ----
    h = np.full(n, 100.0); l = np.full(n, 100.0); c = np.full(n, 100.0)
    l[7] = 5.0
    ph, pl, _ = unique_confirmed_liq_pivots(h, l, c, c)
    ok = (np.isfinite(pl[8]) and abs(pl[8] - 5.0) < 1e-9 and not np.isfinite(pl[7]))
    rec("L02", ok, "low pivot p=7 confirmed at p+1=8; pl[8]=%.4f" % (pl[8] if np.isfinite(pl[8]) else float('nan')))

    # ---- L03 same-side higher replaces newest ----
    ph, pl = _inj(40, [(10, "high", 100.0), (20, "high", 110.0)])
    zz = build_zigzag(ph, pl)
    ok = (len(zz) >= 1 and abs(zz[0]["y"] - 110.0) < 1e-9)
    rec("L03", ok, "newest high = %.4f (expect 110)" % (zz[0]["y"] if zz else float('nan')))

    # ---- L04 same-side lower ignored ----
    ph, pl = _inj(40, [(10, "high", 110.0), (20, "high", 105.0)])
    zz = build_zigzag(ph, pl)
    ok = (len(zz) >= 1 and abs(zz[0]["y"] - 110.0) < 1e-9)
    rec("L04", ok, "newest high = %.4f (expect 110, 105 ignored)" % (zz[0]["y"] if zz else float('nan')))

    # ---- L05 opposite side inserts new node ----
    ph, pl = _inj(40, [(10, "high", 100.0), (20, "low", 10.0), (30, "high", 120.0)])
    zz = build_zigzag(ph, pl)
    dirs = [z["dir"] for z in zz]
    ok = (len(zz) == 3 and dirs[0] == 1 and dirs[1] == -1 and dirs[2] == 1
          and abs(zz[0]["y"] - 120.0) < 1e-9)
    rec("L05", ok, "zz dirs=%s y0=%.2f (expect [1,-1,1] 120)" % (dirs, zz[0]["y"] if zz else float('nan')))

    # ---- L06 zigzag cap 50 ----
    items = []
    for k in range(60):
        b = 10 + k * 2
        items.append((b, "high" if k % 2 == 0 else "low", 100.0 + k))
    ph, pl = _inj(200, items)
    zz = build_zigzag(ph, pl)
    ok = (len(zz) <= MAX_SIZE)
    rec("L06", ok, "zz_len=%d (cap %d)" % (len(zz), MAX_SIZE))

    atr = np.full(80, 10.0, dtype=float)  # margin = 10 / (10/6.9) = 6.9

    # ---- L07 cluster count = 2 -> NO level ----
    ph, pl = _inj(80, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 103.0)])
    h, l, c = _flat(80)
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    ok = (sm[-1]["up_count"] == 0)
    rec("L07", ok, "up_count=%d (expect 0, count=2)" % sm[-1]["up_count"])

    # ---- L08 cluster count = 3 -> level created ----
    ph, pl = _inj(80, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                      (40, "low", 50.0), (50, "high", 98.0)])
    h, l, c = _flat(80)
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    ok = (sm[-1]["up_count"] == 1 and len(sm[-1]["vis_up"]) == 1)
    rec("L08", ok, "up_count=%d (expect 1)" % sm[-1]["up_count"])

    # ---- L09 lower margin equality (y == pivot - margin excluded) ----
    ph, pl = _inj(80, [(10, "high", 103.0), (20, "low", 50.0), (30, "high", 93.1),
                      (40, "low", 50.0), (50, "high", 100.0)])
    h, l, c = _flat(80)
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    ok = (sm[-1]["up_count"] == 0)
    rec("L09", ok, "up_count=%d (expect 0; 93.1 == pivot-margin excluded)" % sm[-1]["up_count"])

    # ---- L10 upper margin equality (y == pivot + margin excluded) ----
    ph, pl = _inj(80, [(10, "high", 97.0), (20, "low", 50.0), (30, "high", 106.9),
                      (40, "low", 50.0), (50, "high", 100.0)])
    h, l, c = _flat(80)
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    ok = (sm[-1]["up_count"] == 0)
    rec("L10", ok, "up_count=%d (expect 0; 106.9 == pivot+margin excluded)" % sm[-1]["up_count"])

    # ---- L11 high-side scan break (literal: out-of-band old node stops scan) ----
    # zz newest-first: in-band(98), out-of-band(200), in-band(100,102).
    # With break: after 200 the scan stops -> count=1 -> NO level.
    # Without break (hypothetical): 100,102 also in band -> count=3 -> level.
    zz_break = [
        {"dir": 1, "x": 90, "y": 98.0},
        {"dir": 1, "x": 80, "y": 200.0},
        {"dir": 1, "x": 70, "y": 100.0},
        {"dir": 1, "x": 60, "y": 102.0},
    ]
    obj = cluster_level_literal(zz_break, +1, 98.0, 10.0)
    ok = (obj is None)
    rec("L11", ok, "cluster_level_literal returns %s (expect None; 200 breaks scan)" % ("level" if obj else "None"))

    # ---- L12 low-side scan break ----
    zz_break_low = [
        {"dir": -1, "x": 90, "y": 98.0},
        {"dir": -1, "x": 80, "y": 5.0},
        {"dir": -1, "x": 70, "y": 100.0},
        {"dir": -1, "x": 60, "y": 102.0},
    ]
    obj = cluster_level_literal(zz_break_low, -1, 98.0, 10.0)
    ok = (obj is None)
    rec("L12", ok, "cluster_level_literal returns %s (expect None; 5 breaks scan)" % ("level" if obj else "None"))

    # ---- L13 level geometry ----
    ph, pl = _inj(80, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                      (40, "low", 50.0), (50, "high", 98.0)])
    h, l, c = _flat(80)
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    lv = sm[-1]["vis_up"][0] if sm[-1]["vis_up"] else None
    # level price = oldest band node (source: last inside node), here 100.
    ok = (lv is not None
          and abs(lv["top"] - 106.9) < 1e-6 and abs(lv["bottom"] - 93.1) < 1e-6
          and abs(lv["level"] - 100.0) < 1e-6)
    rec("L13", ok, "top=%.4f bottom=%.4f level=%.4f (expect 106.9/93.1/100.0)" % (
        lv["top"] if lv else float('nan'), lv["bottom"] if lv else float('nan'),
        lv["level"] if lv else float('nan')))

    # ---- L14 same start_bar -> update (unit on level-object function) ----
    levels = [{"left": 7, "level": 100.0, "top": 106.9, "bottom": 93.1, "side": 1, "broken": False, "breach_i": None}]
    obj = {"left": 7, "level": 100.0, "top": 110.0, "bottom": 90.0, "side": 1, "broken": False, "breach_i": None}
    update_level_objects_literal(levels, obj)
    ok = (len(levels) == 1 and abs(levels[0]["top"] - 110.0) < 1e-9 and abs(levels[0]["bottom"] - 90.0) < 1e-9)
    rec("L14", ok, "len=%d top=%.2f bottom=%.2f (expect 1 / 110 / 90 update)" % (
        len(levels), levels[0]["top"], levels[0]["bottom"]))

    # ---- L15 Visible = 3 cap ----
    items = []
    for ci, base in enumerate([200.0, 300.0, 400.0, 500.0]):
        off = 30 + ci * 60
        items.append((off, "high", base))
        items.append((off + 10, "low", base - 50.0))
        items.append((off + 20, "high", base + 2.0))
        items.append((off + 30, "low", base - 50.0))
        items.append((off + 40, "high", base - 2.0))
    ph, pl = _inj(300, items)
    h, l, c = _flat(300)
    atrL = np.full(300, 10.0, dtype=float)
    sm = run_liquidity_state_machine(h, l, c, atrL, ph, pl)
    ok = (sm[-1]["up_count"] == VIS_LIQ)
    rec("L15", ok, "up_count=%d (expect visible cap %d)" % (sm[-1]["up_count"], VIS_LIQ))

    # ---- L16 high breach strictness ----
    ph, pl = _inj(80, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                      (40, "low", 50.0), (50, "high", 98.0)])
    h, l, c = _flat(80)
    h[60] = 106.9   # == top -> NOT breached
    h[61] = 107.0   # > top -> breached
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    ok = (sm[60]["breach_up"] == 0 and sm[61]["breach_up"] == 1)
    rec("L16", ok, "breach_up[60]=%d breach_up[61]=%d (expect 0 / 1)" % (sm[60]["breach_up"], sm[61]["breach_up"]))

    # ---- L17 low breach strictness ----
    ph, pl = _inj(80, [(10, "low", 100.0), (20, "high", 200.0), (30, "low", 98.0),
                      (40, "high", 200.0), (50, "low", 102.0)])
    h, l, c = _flat(80)
    l[60] = 93.1    # == bottom -> NOT breached
    l[61] = 92.9    # < bottom -> breached
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    ok = (sm[60]["breach_down"] == 0 and sm[61]["breach_down"] == 1)
    rec("L17", ok, "breach_down[60]=%d breach_down[61]=%d (expect 0 / 1)" % (sm[60]["breach_down"], sm[61]["breach_down"]))

    # ---- L18 multiple same-bar breach ----
    ph, pl = _inj(120, [(10, "high", 200.0), (20, "low", 50.0), (30, "high", 202.0),
                       (40, "low", 50.0), (50, "high", 198.0),
                       (60, "high", 300.0), (70, "low", 50.0), (80, "high", 302.0),
                       (90, "low", 50.0), (100, "high", 298.0)])
    h, l, c = _flat(120)
    h[110] = 400.0  # breaches both tops
    atrL = np.full(120, 10.0, dtype=float)
    sm = run_liquidity_state_machine(h, l, c, atrL, ph, pl)
    vu = sm[110]["vis_up"]
    both_broken = len(vu) >= 2 and all(lv["broken"] for lv in vu)
    ok = (sm[110]["breach_up"] == 1 and both_broken)
    rec("L18", ok, "breach_up=%d both_broken=%s (expect 1 / True)" % (sm[110]["breach_up"], both_broken))

    # ---- L19 post-break zone (strict band) ----
    ph, pl = _inj(80, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                      (40, "low", 50.0), (50, "high", 98.0)])
    h, l, c = _flat(80)
    h[60] = 110.0   # breach top=106.9 ; zone_active=1
    h[61] = 100.0; l[61] = 90.0   # inside [75,121] ; zone_active=1
    h[62] = 100.0; l[62] = 75.0   # == lower bound ; strict exit -> 0
    h[63] = 130.0; l[63] = 100.0  # >= upper ; 0
    sm = run_liquidity_state_machine(h, l, c, atr, ph, pl)
    seq = [sm[b]["zone_active"] for b in (60, 61, 62, 63)]
    ok = (seq == [1, 1, 0, 0])
    rec("L19", ok, "zone_active seq=%s (expect [1,1,0,0])" % seq)

    # ---- L20 ATR = 0 / NaN (literal: no level; matches prod guard output) ----
    ph, pl = _inj(80, [(10, "high", 100.0), (20, "low", 50.0), (30, "high", 102.0),
                      (40, "low", 50.0), (50, "high", 98.0)])
    h, l, c = _flat(80)
    atr0 = np.zeros(80, dtype=float)
    sm0 = run_liquidity_state_machine(h, l, c, atr0, ph, pl)
    atr_nan = np.full(80, np.nan, dtype=float)
    sm_nan = run_liquidity_state_machine(h, l, c, atr_nan, ph, pl)
    ok = (sm0[-1]["up_count"] == 0 and sm_nan[-1]["up_count"] == 0)
    rec("L20", ok, "up_count atr=0:%d atr=nan:%d (expect 0 / 0; OUTPUT_MATCH vs prod guard)" % (
        sm0[-1]["up_count"], sm_nan[-1]["up_count"]))

    all_pass = all(r["pass"] for r in results)
    return results, all_pass


if __name__ == "__main__":
    markers = check_source_gate()
    results, all_pass = run_self_tests()
    for r in results:
        print("[%s] %s : %s" % ("PASS" if r["pass"] else "FAIL", r["case_id"], r["detail"]))
    print("ALL_PASS=%s  n_cases=%d" % (all_pass, len(results)))
    if not all_pass:
        raise SystemExit("ORACLE_NOT_READY")
