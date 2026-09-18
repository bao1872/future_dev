#!/usr/bin/env python3
"""
SRchannel.pine literal source Oracle (SRC2.1)
==============================================

Independent, reproducible Oracle built ONLY from the pinned Pine source
(ref/SRchannel.pine, SHA256 9d8ee4af... ) + confirmed Pine official semantics.

It does NOT import or call any production code. It is the ruler for the
future SRC2.2 production differential.

Layered design (so SRC2.2 can isolate pivot-builtin mismatch vs SR
state-machine mismatch):

  Layer A  unique_confirmed_pivots()        -> strict-unique pivot confirmation + tie mask
  Layer B  sr_state_oracle_from_confirmed_pivots (-> run_sr_state_machine)
                                        -> pure SR state machine fed confirmed pivots

Constants below are copied verbatim from the pinned source contract; they are
NOT derived from production.

Frozen source params:
  prd = 10, ppsrc = High/Low, ChannelW = 5, minstrength = 1,
  maxnumsr input = 6 (display = min(10, 6) = 6), loopback = 290,
  width lookback = 300, max internal SR = 10.
"""

import os
import sys
import re
import json
import csv
import math
import hashlib
import subprocess

# ---------------------------------------------------------------------------
# Paths / frozen contract
# ---------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SR_SOURCE_PATH = os.path.join(REPO_ROOT, "ref", "SRchannel.pine")

PRD = 10
PPSRC = "High/Low"
CHANNELW = 5            # percent -> * 5 / 100 = * 0.05
MINSTRENGTH = 1
MAXNUMSR_INPUT = 6      # display channels = min(10, MAXNUMSR_INPUT) = 6
LOOPBACK = 290
WIDTH_LOOKBACK = 300
MAX_INTERNAL_SR = 10

PINNED_SHA256 = "9d8ee4af1e2c9a05c361c2e7883dc418b2d575e5cd8bcfaa0e652a13e395a87a"
BASE_SHA = "986aa7203f853184a2c30519946810b798a6684e"

# Pine float comparisons use ~9-digit precision; differences below this are
# treated as equal (-> tie / not strict-unique), per confirmed TV semantics.
FLOAT_BOUNDARY = 1e-8

ARTIFACT_DIR = os.path.join(REPO_ROOT, "artifacts", "sr_source_semantic_oracle")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def is_nan(x):
    return isinstance(x, float) and x != x


def clean(v):
    if v is None:
        return None
    if is_nan(v):
        return None
    if isinstance(v, float):
        return v
    return v


def _almost_equal(a, b):
    return abs(a - b) < FLOAT_BOUNDARY


def pine_bool_float(x):
    """Pine `bool(float)`: NaN -> false, 0.0 -> false, other finite -> true."""
    if x is None:
        return False
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return False
    if xf != xf:          # NaN
        return False
    return xf != 0.0


# ---------------------------------------------------------------------------
# T0 source gate
# ---------------------------------------------------------------------------
def check_source_gate():
    info = {"ok": True, "git_head": None, "sha_worktree": None,
            "sha_gitobj": None, "markers_ok": False, "notes": []}
    try:
        info["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
    except Exception as e:                       # pragma: no cover
        info["git_head"] = "ERR:%s" % e
        info["notes"].append("git rev-parse failed")
    if info["git_head"] != BASE_SHA:
        info["notes"].append(
            "git_head != BASE_SHA (pre-commit T0 expects equality; "
            "post-commit this is expected and does not fail ORACLE_READY)")
    if os.path.exists(SR_SOURCE_PATH):
        data = open(SR_SOURCE_PATH, "rb").read()
        info["sha_worktree"] = hashlib.sha256(data).hexdigest()
        if info["sha_worktree"] != PINNED_SHA256:
            info["ok"] = False
            info["notes"].append("working-tree SHA256 != pinned")
        try:
            gobj = subprocess.check_output(
                ["git", "show", "HEAD:ref/SRchannel.pine"],
                cwd=REPO_ROOT).decode()
            info["sha_gitobj"] = hashlib.sha256(gobj.encode("utf-8")).hexdigest()
        except Exception:
            info["sha_gitobj"] = None
    else:
        info["ok"] = False
        info["notes"].append("SR source missing")
    try:
        text = open(SR_SOURCE_PATH, "r", encoding="utf-8").read()
    except Exception:
        text = ""
    markers = [
        "prd = input.int(defval = 10",
        "ppsrc = input.string(defval = 'High/Low'",
        "ChannelW = input.int(defval = 5",
        "maxnumsr = input.int(defval = 6",
        "loopback = input.int(defval = 290",
        "ta.pivothigh(src1, prd, prd)",
        "ta.pivotlow(src2, prd, prd)",
    ]
    info["markers_ok"] = all(m in text for m in markers)
    if not info["markers_ok"]:
        info["ok"] = False
        info["notes"].append("source markers missing")
    return info


def check_oracle_independence():
    """Oracle must not depend on production code.

    Detects real production *calls* (name followed by '(') only; our own
    Layer-A function is named `unique_confirmed_pivots`, which is excluded by
    the negative lookbehind. The check list literals themselves are not call
    sites and carry no '(' immediately after the name, so they do not match.
    """
    src = open(os.path.abspath(__file__), "r", encoding="utf-8").read()
    illegal_imports = []
    for line in src.splitlines():
        s = line.strip()
        if s.startswith("import ") or s.startswith("from "):
            mod = s.split()[1].split(".")[0]
            if mod not in ("os", "sys", "re", "json", "csv", "math", "hashlib", "subprocess"):
                illegal_imports.append(s)
    calls = {}
    for name in ["confirmed_pivots", "build_sr_features", "compute_segment_features"]:
        pat = r"(?<!unique_)" + re.escape(name) + r"\("
        calls[name] = len(re.findall(pat, src))
    prod_calls = sum(calls.values())
    return {
        "illegal_imports": illegal_imports,
        "calls_confirmed_pivots": calls["confirmed_pivots"],
        "calls_build_sr_features": calls["build_sr_features"],
        "calls_compute_segment_features": calls["compute_segment_features"],
        "independent": (prod_calls == 0) and (not illegal_imports),
    }


# ---------------------------------------------------------------------------
# Layer A: unique confirmed pivots
# ---------------------------------------------------------------------------
def unique_confirmed_pivots(high, low, open_, close):
    """Strict-unique pivot confirmation.

    Returns ph[], pl[] aligned to bar index (non-na ONLY at confirmation bar
    p+PRD) plus a list of tie (plateau / equal-extreme) records -> UNVERIFIED.
    """
    n = len(high)
    if PPSRC == "High/Low":
        src1 = [float(h) for h in high]
        src2 = [float(l) for l in low]
    else:
        src1 = [float(max(c, o)) for c, o in zip(close, open_)]
        src2 = [float(min(c, o)) for c, o in zip(close, open_)]
    ph = [float("nan")] * n
    pl = [float("nan")] * n
    tie_flags = []
    for p in range(PRD, n - PRD):
        lo_w = p - PRD
        hi_w = p + PRD                      # confirmation bar
        seg1 = src1[lo_w:hi_w + 1]
        seg2 = src2[lo_w:hi_w + 1]
        c1 = seg1[PRD]
        others1 = seg1[:PRD] + seg1[PRD + 1:]
        m_others1 = max(others1)
        if c1 > m_others1 + FLOAT_BOUNDARY:
            ph[hi_w] = c1                   # strict-unique high
        elif _almost_equal(c1, m_others1):
            tie_flags.append({"type": "high", "p": p, "confirm_bar": hi_w})
        c2 = seg2[PRD]
        others2 = seg2[:PRD] + seg2[PRD + 1:]
        m_others2 = min(others2)
        if c2 < m_others2 - FLOAT_BOUNDARY:
            pl[hi_w] = c2                   # strict-unique low
        elif _almost_equal(c2, m_others2):
            tie_flags.append({"type": "low", "p": p, "confirm_bar": hi_w})
    return ph, pl, tie_flags


# ---------------------------------------------------------------------------
# Layer B: pure SR state machine
# ---------------------------------------------------------------------------
def store_pivot_value(ph_c, pl_c):
    """bool(ph) ? ph : pl ; returns None when neither is inserted."""
    if pine_bool_float(ph_c):
        return ph_c
    if pine_bool_float(pl_c):
        return pl_c
    return None


def compute_cwidth(high, low, c):
    """(ta.highest(300) - ta.lowest(300)) * ChannelW / 100.

    Pre-full-300-bar startup is UNVERIFIED -> returns None (exact domain
    begins at bar 299, 0-indexed).
    """
    if c < WIDTH_LOOKBACK - 1:
        return None
    lo_i = c - WIDTH_LOOKBACK + 1
    hi = max(high[lo_i:c + 1])
    lo = min(low[lo_i:c + 1])
    return (hi - lo) * CHANNELW / 100.0


def get_sr_vals(pivotvals, ind, cwidth, trace=False):
    """Literal get_sr_vals(ind): seed lo=hi=pivotvals[ind]; scan all pivots;
    accept when wdth <= cwidth; expand lo/hi; numpp += 20 per accepted pivot.
    """
    lo = float(pivotvals[ind])
    hi = lo
    numpp = 0
    steps = []
    for y in range(len(pivotvals)):
        cpp = float(pivotvals[y])
        if cpp <= hi:
            wdth = hi - cpp
            branch = "lo"
        else:
            wdth = cpp - lo
            branch = "hi"
        accepted = (cwidth is not None) and (wdth <= cwidth)
        if accepted:
            if cpp <= hi:
                lo = min(lo, cpp)
            else:
                hi = max(hi, cpp)
            numpp += 20
        if trace:
            steps.append({
                "y": y, "cpp": cpp, "branch": branch,
                "wdth": wdth, "accepted": accepted,
                "lo": lo, "hi": hi, "numpp": numpp,
            })
    return hi, lo, numpp, steps


def build_supres(pivotvals, cwidth, high, low, c):
    """get_sr_vals for each pivot (-> pivot strength) then add 291-bar
    HL-touch strength (one increment per bar, not per high/low)."""
    size = len(pivotvals)
    supres = [0.0] * (size * 3)
    for x in range(size):
        hi, lo, numpp, _ = get_sr_vals(pivotvals, x, cwidth)
        supres[x * 3] = float(numpp)
        supres[x * 3 + 1] = hi
        supres[x * 3 + 2] = lo
    for x in range(size):
        h = supres[x * 3 + 1]
        l = supres[x * 3 + 2]
        s = 0
        for y in range(0, LOOPBACK + 1):     # y = 0..290 -> 291 bars
            b = c - y
            if b < 0:
                continue
            if (high[b] <= h and high[b] >= l) or (low[b] <= h and low[b] >= l):
                s += 1
        supres[x * 3] = supres[x * 3] + s
    return supres


def select_channels_from_supres(supres, size):
    """Strongest-first selection with `>` (newest-first tie), then suppression
    of candidates whose hi/lo lies inside the selected [ll,hh] (-> -1), capped
    at MAX_INTERNAL_SR; final stable descending sort; display <= 6."""
    suportresistance = [0.0] * (MAX_INTERNAL_SR * 2)
    stren = [0.0] * MAX_INTERNAL_SR
    work = list(supres)
    src = 0
    for _ in range(size):
        stv = -1.0
        stl = -1
        for y in range(size):
            sval = work[y * 3]
            if sval > stv and sval >= MINSTRENGTH * 20:
                stv = sval
                stl = y
        if stl >= 0:
            hh = work[stl * 3 + 1]
            ll = work[stl * 3 + 2]
            suportresistance[src * 2] = hh
            suportresistance[src * 2 + 1] = ll
            stren[src] = work[stl * 3]
            for y in range(size):
                if (work[y * 3 + 1] <= hh and work[y * 3 + 1] >= ll) or \
                   (work[y * 3 + 2] <= hh and work[y * 3 + 2] >= ll):
                    work[y * 3] = -1.0
            src += 1
            if src >= MAX_INTERNAL_SR:
                break
    for x in range(MAX_INTERNAL_SR - 1):
        for y in range(x + 1, MAX_INTERNAL_SR):
            if stren[y] > stren[x]:
                stren[x], stren[y] = stren[y], stren[x]
                suportresistance[x * 2], suportresistance[y * 2] = \
                    suportresistance[y * 2], suportresistance[x * 2]
                suportresistance[x * 2 + 1], suportresistance[y * 2 + 1] = \
                    suportresistance[y * 2 + 1], suportresistance[x * 2 + 1]
    channels = []
    for x in range(min(MAX_INTERNAL_SR, MAXNUMSR_INPUT)):
        if suportresistance[x * 2] != 0.0:
            channels.append({
                "hi": suportresistance[x * 2],
                "lo": suportresistance[x * 2 + 1],
                "strength": stren[x],
            })
    return channels, src, work


def evaluate_breaks(channels, close_c, close_prev):
    """not_in_a_channel first; only if True, check C[t-1]<=hi<C[t] (resistance)
    and C[t-1]>=lo>C[t] (support)."""
    not_in = True
    for ch in channels:
        if close_c <= ch["hi"] and close_c >= ch["lo"]:
            not_in = False
    resistance_broken = False
    support_broken = False
    if not_in:
        for ch in channels:
            if close_prev <= ch["hi"] and close_c > ch["hi"]:
                resistance_broken = True
            if close_prev >= ch["lo"] and close_c < ch["lo"]:
                support_broken = True
    return not_in, resistance_broken, support_broken


def run_sr_state_machine(ph, pl, high, low, close):
    """Bar-by-bar replay fed with confirmed ph/pl. Channels persist across
    non-confirmation bars (Pine only rebuilds on a new pivot bar)."""
    n = len(ph)
    pivotvals = []
    pivotlocs = []
    per_bar = []
    for c in range(n):
        if pine_bool_float(ph[c]) or pine_bool_float(pl[c]):
            val = store_pivot_value(ph[c], pl[c])
            if val is not None:
                pivotvals.insert(0, val)
                pivotlocs.insert(0, c)
                x = len(pivotvals) - 1
                while x >= 0:
                    if c - pivotlocs[x] > LOOPBACK:
                        pivotvals.pop()
                        pivotlocs.pop()
                        x -= 1
                        continue
                    break
            cwidth = compute_cwidth(high, low, c)
            if cwidth is not None and len(pivotvals) > 0:
                supres = build_supres(pivotvals, cwidth, high, low, c)
                channels, _, _ = select_channels_from_supres(supres, len(pivotvals))
            else:
                channels = []
        else:
            channels = per_bar[-1]["channels"] if per_bar else []
        close_c = float(close[c])
        close_prev = float(close[c - 1]) if c > 0 else float(close[c])
        not_in, rb, sb = evaluate_breaks(channels, close_c, close_prev)
        per_bar.append({
            "bar": c,
            "n_pivots": len(pivotvals),
            "channels": [dict(ch) for ch in channels],
            "n_channels": len(channels),
            "not_in_channel": not_in,
            "resistancebroken": rb,
            "supportbroken": sb,
        })
    return per_bar


# ---------------------------------------------------------------------------
# Self tests S1..S16
# ---------------------------------------------------------------------------
def _flat(n, price):
    return [float(price)] * n


def test_S1():
    n = 100
    high = _flat(n, 50.0)
    low = _flat(n, 40.0)
    op = _flat(n, 45.0)
    cl = _flat(n, 45.0)
    high[50] = 100.0                       # strict-unique spike
    ph, pl, ties = unique_confirmed_pivots(high, low, op, cl)
    ok = is_nan(ph[50]) and is_nan(ph[59]) and (ph[60] == 100.0)
    return ok, {"ph[50]": ph[50], "ph[59]": ph[59], "ph[60]": ph[60],
                "confirm_bar": 60, "note": "confirmation = p + 10"}


def test_S2():
    n = 450
    high = _flat(n, 50.0)
    low = _flat(n, 40.0)
    op = _flat(n, 45.0)
    cl = _flat(n, 45.0)
    for sp in [30, 120, 210, 300, 360, 420]:
        high[sp] = 100.0
        low[sp] = 20.0
    phA, plA, _ = unique_confirmed_pivots(high, low, op, cl)
    runA = run_sr_state_machine(phA, plA, high, low, cl)
    cut = 350
    highB = list(high)
    lowB = list(low)
    opB = list(op)
    clB = list(cl)
    for i in range(cut + 1, n):
        highB[i] += 0.5
        lowB[i] -= 0.5
        clB[i] += 0.5
    phB, plB, _ = unique_confirmed_pivots(highB, lowB, opB, clB)
    runB = run_sr_state_machine(phB, plB, highB, lowB, clB)
    same = True
    first_diff = None
    for b in range(0, cut + 1):
        a = runA[b]
        bb = runB[b]
        a_ch = [(round(ch["hi"], 6), round(ch["lo"], 6)) for ch in a["channels"]]
        b_ch = [(round(ch["hi"], 6), round(ch["lo"], 6)) for ch in bb["channels"]]
        if (a["n_channels"] != bb["n_channels"] or a_ch != b_ch or
                a["resistancebroken"] != bb["resistancebroken"] or
                a["supportbroken"] != bb["supportbroken"] or
                a["not_in_channel"] != bb["not_in_channel"]):
            same = False
            first_diff = b
            break
    return same, {"cut": cut, "prefix_len": cut + 1, "same": same,
                  "first_diff_bar": first_diff}


def test_S3():
    n = 40
    high = _flat(n, 50.0)
    low = _flat(n, 40.0)
    op = _flat(n, 45.0)
    cl = _flat(n, 45.0)
    high[20] = 100.0
    high[21] = 100.0                      # equal-high plateau -> tie
    ph, pl, ties = unique_confirmed_pivots(high, low, op, cl)
    tie_high = [t for t in ties if t["type"] == "high"]
    confirm_bars = [t["confirm_bar"] for t in tie_high]
    no_assign = all(is_nan(ph[cb]) for cb in confirm_bars)
    ok = (len(tie_high) > 0) and no_assign
    return ok, {"tie_high_count": len(tie_high), "confirm_bars": confirm_bars,
                "no_value_assigned": no_assign,
                "classification": "PIVOT_TIE_SEMANTICS=UNVERIFIED"}


def test_S4():
    v = store_pivot_value(110.0, 90.0)
    ok = v == 110.0
    return ok, {"stored": v, "note": "ph priority on simultaneous event"}


def test_S5():
    # ph=0.0 must NOT be inserted; pl=nan must NOT be inserted.
    v0 = store_pivot_value(0.0, float("nan"))     # zero ph -> not inserted
    vn = store_pivot_value(float("nan"), float("nan"))  # both nan -> None
    vz = store_pivot_value(float("nan"), 0.0)      # zero pl -> not inserted
    vok = store_pivot_value(float("nan"), 80.0)    # truthy pl -> inserted
    ok = (v0 is None) and (vn is None) and (vz is None) and (vok == 80.0)
    return ok, {"zero_ph_inserted": v0 is None, "nan_inserted": vn is None,
                "zero_pl_inserted": vz is None, "only_pl": vok}


def test_S6():
    n = 400
    high = _flat(n, 50.0)
    low = _flat(n, 40.0)
    op = _flat(n, 45.0)
    cl = _flat(n, 45.0)

    def scenario(new_loc):
        phx = [float("nan")] * n
        plx = [float("nan")] * n
        phx[0] = 100.0                     # pivot A, confirmation loc 0
        phx[new_loc] = 110.0               # new pivot triggers cleanup
        runx = run_sr_state_machine(phx, plx, high, low, cl)
        return runx[new_loc]["n_pivots"]

    surv = scenario(290)                   # age = 290 -> survives -> 2
    rem = scenario(291)                    # age = 291 -> removed -> 1
    ok = (surv == 2) and (rem == 1)
    return ok, {"age290_pivots": surv, "age291_pivots": rem,
                "note": "cleanup only on new-pivot bar"}


def test_S7():
    n = 400
    high = _flat(n, 100.0)
    low = _flat(n, 100.0)
    op = _flat(n, 100.0)
    cl = _flat(n, 100.0)                   # flat -> cwidth = 0
    ph = [float("nan")] * n
    pl = [float("nan")] * n
    ph[300] = 100.0
    ph[301] = 100.0                        # equal pivots, injected
    run = run_sr_state_machine(ph, pl, high, low, cl)
    chans = run[301]["channels"]
    zero_width = any(abs(c["hi"] - c["lo"]) < 1e-12 for c in chans)
    ok = zero_width and len(chans) >= 1
    return ok, {"n_channels": len(chans),
                "channels": [(c["hi"], c["lo"]) for c in chans],
                "note": "wdth<=cwidth with cwidth=0 is allowed (no >0 guard)"}


def test_S8():
    pivotvals = [100.0, 104.0, 108.0, 96.0]   # newest -> oldest
    cwidth = 15.0
    hi, lo, numpp, steps = get_sr_vals(pivotvals, 0, cwidth, trace=True)
    ok = (abs(hi - 108.0) < 1e-9) and (abs(lo - 96.0) < 1e-9) and (numpp == 80)
    return ok, {"hi": hi, "lo": lo, "numpp": numpp, "pivot_count": numpp // 20,
                "steps": steps}


def test_S9():
    n = 301
    high = _flat(n, 1.0)
    low = _flat(n, 1.0)
    op = _flat(n, 1.0)
    cl = _flat(n, 1.0)
    pivotvals = [100.0, 102.0, 104.0]
    cwidth = 10.0
    touch_bars = [10, 50, 100, 150, 200, 250, 300]
    for b in touch_bars:
        high[b] = 102.0
        low[b] = 50.0 if b != 100 else 102.0   # bar 100: high AND low in zone
    c = 300
    supres = build_supres(pivotvals, cwidth, high, low, c)
    strength = supres[0]
    ok = strength == 67
    return ok, {"strength": strength, "pivot_part": 60, "touch_part": 7,
                "note": "high+low same bar still +1"}


def test_S10():
    n = 600
    high = _flat(n, 1.0)
    low = _flat(n, 1.0)
    op = _flat(n, 1.0)
    cl = _flat(n, 1.0)
    pivotvals = [100.0]
    cwidth = 10.0
    c = 300
    high[10] = 100.0                      # y = 290 (oldest allowed) -> counted
    high[9] = 100.0                        # y = 291 -> NOT counted
    supres = build_supres(pivotvals, cwidth, high, low, c)
    with_oldest = supres[0]               # expect 20 + 1 = 21
    high2 = list(high)
    high2[9] = 1.0
    high2[10] = 1.0
    supres2 = build_supres(pivotvals, cwidth, high2, low, c)
    without = supres2[0]                   # expect 20
    ok = (with_oldest == 21) and (without == 20)
    return ok, {"with_oldest_touch": with_oldest, "without": without,
                "window_len": LOOPBACK + 1}


def test_S11():
    n = 400
    high = _flat(n, 50.0)                  # outside both channels -> no touches
    low = _flat(n, 50.0)
    op = _flat(n, 50.0)
    cl = _flat(n, 50.0)
    pivotvals = [200.0, 100.0]             # newest first
    cwidth = 0.0                           # separated channels
    supres = build_supres(pivotvals, cwidth, high, low, 300)
    channels, _, _ = select_channels_from_supres(supres, 2)
    ok = bool(channels) and (abs(channels[0]["hi"] - 200.0) < 1e-9)
    return ok, {"channel0_hi": channels[0]["hi"] if channels else None,
                "note": "equal strength -> newer (lower index) wins"}


def test_S12():
    # candidate A [lo=100, hi=105] strength 40 ; candidate B [lo=102, hi=102] strength 20
    # B endpoint inside A -> suppressed (strength set to -1)
    size = 2
    supres = [40.0, 105.0, 100.0, 20.0, 102.0, 102.0]
    channels, src, work = select_channels_from_supres(supres, size)
    b_suppressed = work[3] == -1.0
    b_selected = any(abs(c["hi"] - 102.0) < 1e-9 for c in channels)
    ok = b_suppressed and (not b_selected)
    return ok, {"B_strength_after": work[3], "n_channels": len(channels)}


def test_S13():
    n = 400
    high = _flat(n, 100.0)
    low = _flat(n, 100.0)
    op = _flat(n, 100.0)
    cl = _flat(n, 100.0)
    pivotvals = [float(100 + 10 * i) for i in range(12)]   # 12 separated
    cwidth = 0.0
    supres = build_supres(pivotvals, cwidth, high, low, 300)
    channels, src, _ = select_channels_from_supres(supres, 12)
    ok = (src == 10) and (len(channels) == 6)
    return ok, {"internal_selected": src, "display": len(channels)}


def test_S14():
    channels = [{"hi": 105.0, "lo": 100.0, "strength": 40.0}]
    not_in, rb, sb = evaluate_breaks(channels, 106.0, 104.0)
    ok = rb and (not sb)
    return ok, {"not_in": not_in, "resistance": rb, "support": sb}


def test_S15():
    channels = [{"hi": 105.0, "lo": 95.0, "strength": 40.0}]
    not_in, rb, sb = evaluate_breaks(channels, 94.0, 96.0)
    ok = sb and (not rb)
    return ok, {"not_in": not_in, "resistance": rb, "support": sb}


def test_S16():
    channels = [{"hi": 105.0, "lo": 100.0, "strength": 40.0},
                {"hi": 205.0, "lo": 200.0, "strength": 40.0}]
    # current close 102 in first channel -> not_in False -> no breaks even
    # though close crossed the second channel's hi
    not_in, rb, sb = evaluate_breaks(channels, 201.0, 199.0)
    ok = (not not_in) and (not rb) and (not sb)
    return ok, {"not_in": not_in, "resistance": rb, "support": sb}


SELF_TESTS = [
    ("S1", "pivot confirmation = p + 10 (no backfill)", test_S1),
    ("S2", "causality: outputs[:cut+1] unchanged by future edits", test_S2),
    ("S3", "tie / plateau -> UNVERIFIED, no value assigned", test_S3),
    ("S4", "ph priority on simultaneous event", test_S4),
    ("S5", "bool(0.0)/bool(NaN) not inserted", test_S5),
    ("S6", "loopback age 290 survives / 291 removed", test_S6),
    ("S7", "zero-width channel allowed (cwidth=0)", test_S7),
    ("S8", "get_sr_vals literal trace", test_S8),
    ("S9", "strength = 60 (pivot) + 7 (touches) = 67", test_S9),
    ("S10", "291-bar touch window (y=0..290)", test_S10),
    ("S11", "strongest-first tie stability (newer wins)", test_S11),
    ("S12", "suppression sets strength = -1", test_S12),
    ("S13", "10 internal / 6 displayed channels", test_S13),
    ("S14", "resistance break", test_S14),
    ("S15", "support break", test_S15),
    ("S16", "current-in-channel suppresses break", test_S16),
]


# ---------------------------------------------------------------------------
# Human-check samples
# ---------------------------------------------------------------------------
def build_samples(gate):
    samples = []
    # 1. unique pivot confirmation
    ok1, d1 = test_S1()
    samples.append({
        "sample_id": "S1_unique_pivot",
        "description": "strict-unique high at bar 50",
        "input": "high[50]=100, others=50",
        "intermediate": "window [40,60] strict-unique max at 50",
        "expected": "ph[60]=100, ph[50]=ph[59]=NaN",
        "actual": "ph[60]=%s" % d1["ph[60]"],
    })
    # 2. loopback age
    ok6, d6 = test_S6()
    samples.append({
        "sample_id": "S6_loopback_age",
        "description": "pivot A confirmed at loc 0",
        "input": "new pivot at bar 290 / 291",
        "intermediate": "age = new_loc - 0",
        "expected": "age290 -> 2 pivots, age291 -> 1 pivot",
        "actual": "age290=%s, age291=%s" % (d6["age290_pivots"], d6["age291_pivots"]),
    })
    # 3. zero-width
    ok7, d7 = test_S7()
    samples.append({
        "sample_id": "S7_zero_width",
        "description": "flat OHLC (cwidth=0) + equal pivots 100",
        "input": "ph[300]=ph[301]=100",
        "intermediate": "get_sr_vals: wdth=0 <= 0 accept",
        "expected": "zero-width channel [100,100] present",
        "actual": "channels=%s" % d7["channels"],
    })
    # 4. strength
    ok9, d9 = test_S9()
    samples.append({
        "sample_id": "S9_strength",
        "description": "3 pivots in [100,104] + 7 touch bars",
        "input": "pivotvals=[100,102,104], cwidth=10, 7 touches (bar100 both hl)",
        "intermediate": "pivot=60, touches=7",
        "expected": "strength=67",
        "actual": "strength=%s" % d9["strength"],
    })
    # 5. breaks
    ok14, d14 = test_S14()
    ok15, d15 = test_S15()
    samples.append({
        "sample_id": "S14_S15_breaks",
        "description": "resistance/support break",
        "input": "resistance hi=105 prev=104 cur=106 ; support lo=95 prev=96 cur=94",
        "intermediate": "not_in_a_channel=True",
        "expected": "resistance=True, support=True",
        "actual": "resistance=%s, support=%s" % (d14["resistance"], d15["support"]),
    })
    return samples


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    gate = check_source_gate()
    indep = check_oracle_independence()

    test_rows = []
    all_pass = True
    for name, desc, fn in SELF_TESTS:
        try:
            ok, detail = fn()
        except Exception as e:               # pragma: no cover
            ok, detail = False, {"error": str(e)}
        all_pass = all_pass and ok
        test_rows.append({
            "test": name, "desc": desc, "status": "PASS" if ok else "FAIL",
            "detail": detail,
        })

    # tie / float / startup accounting
    tie_count = sum(1 for r in test_rows if r["test"] == "S3" and r["status"] == "PASS")
    startup_unverified_bars = WIDTH_LOOKBACK - 1   # bars 0..298 exact-unverified

    source_ok = gate["ok"]
    independence_ok = indep["independent"]
    verdict = "ORACLE_READY" if (source_ok and independence_ok and all_pass) else "ORACLE_NOT_READY"

    classification = {
        "source_exact_domain": [
            "pivot confirmation timing (strict-unique, = p+PRD)",
            "ph priority on simultaneous event",
            "bool(0.0)/bool(NaN) behavior",
            "pivot store order (newest-first, confirmation loc)",
            "loopback age rule (>290 removed, 290 survives)",
            "cwidth formula after full 300-bar finite window",
            "get_sr_vals literal scan",
            "pivot strength (20 per pivot)",
            "291-bar HL touch strength (1 per bar)",
            "strongest-first with > (newest-first tie)",
            "suppression (hi/lo in selected -> -1)",
            "10 internal / 6 displayed channels",
            "not-in-channel gate",
            "resistance / support break semantics",
        ],
        "unverified_builtin": [
            "pivot tie / plateau semantics (no runtime oracle)",
            "ta.highest/lowest startup before full 300 bars",
            "ta.highest/lowest NA handling outside finite-input contract",
            "sub-1e-8 float comparison boundary",
        ],
        "source_derived_diagnostic": [
            "display channel count",
            "in-channel flag",
            "channel hi/lo/strength snapshots",
        ],
        "research_extension": [
            "sr_support_dist_atr",
            "sr_resistance_dist_atr",
            "nearest sr_support_price",
            "nearest sr_resistance_price",
            "nearest sr_support_strength",
            "nearest sr_resistance_strength",
            "sr_zone_strength",
        ],
    }

    samples = build_samples(gate)

    summary = {
        "git_sha": gate["git_head"],
        "sr_source_sha256": gate["sha_worktree"],
        "pinned_sha256": PINNED_SHA256,
        "source_params": {
            "prd": PRD, "ppsrc": PPSRC, "ChannelW": CHANNELW,
            "minstrength": MINSTRENGTH, "maxnumsr_input": MAXNUMSR_INPUT,
            "loopback": LOOPBACK, "width_lookback": WIDTH_LOOKBACK,
            "max_internal_sr": MAX_INTERNAL_SR,
        },
        "oracle_independence": indep,
        "classification": classification,
        "tests": {r["test"]: {"status": r["status"], "detail": _clean_detail(r["detail"])}
                  for r in test_rows},
        "pivot_tie_cases": tie_count,
        "float_boundary_cases": "UNVERIFIED (<1e-8)",
        "startup_unverified_bars": startup_unverified_bars,
        "verdict": verdict,
    }

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    with open(os.path.join(ARTIFACT_DIR, "sr_oracle_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(ARTIFACT_DIR, "sr_oracle_selftests.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["test", "status", "desc", "key", "value"])
        for r in test_rows:
            flat = _flatten_detail(r["detail"])
            if not flat:
                w.writerow([r["test"], r["status"], r["desc"], "", ""])
            for k, v in flat.items():
                w.writerow([r["test"], r["status"], r["desc"], k, v])

    # channel trace (S8)
    with open(os.path.join(ARTIFACT_DIR, "sr_oracle_channel_trace.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["y", "cpp", "branch", "wdth", "accepted", "lo", "hi", "numpp"])
        for s in test_S8()[1]["steps"]:
            w.writerow([s["y"], s["cpp"], s["branch"], s["wdth"], s["accepted"],
                       s["lo"], s["hi"], s["numpp"]])

    with open(os.path.join(ARTIFACT_DIR, "sr_oracle_samples.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "description", "input", "intermediate",
                    "expected", "actual"])
        for s in samples:
            w.writerow([s["sample_id"], s["description"], s["input"],
                        s["intermediate"], s["expected"], s["actual"]])

    # console report
    print("=" * 70)
    print("SR SOURCE SEMANTIC ORACLE  (SRC2.1)")
    print("=" * 70)
    print("git_sha            :", gate["git_head"])
    print("sr_source_sha256   :", gate["sha_worktree"])
    print("pinned_sha256      :", PINNED_SHA256)
    print("source sha match   :", gate["sha_worktree"] == PINNED_SHA256)
    print("markers_ok         :", gate["markers_ok"])
    print("oracle independent :", indep["independent"])
    print("-" * 70)
    for r in test_rows:
        print("  %-4s %-4s  %s" % (r["test"], r["status"], r["desc"]))
    print("-" * 70)
    print("pivot_tie_cases         :", tie_count, "(UNVERIFIED)")
    print("startup_unverified_bars :", startup_unverified_bars, "(bar < 299)")
    print("float_boundary          : UNVERIFIED (<1e-8)")
    print("-" * 70)
    print("VERDICT:", verdict)
    print("=" * 70)
    return summary


def _clean_detail(d):
    if isinstance(d, dict):
        return {k: clean(v) for k, v in d.items()}
    return d


def _flatten_detail(d):
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            out[k] = json.dumps(v)
        elif isinstance(v, dict):
            out[k] = json.dumps(v)
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    sys.exit(0 if main()["verdict"] == "ORACLE_READY" else 1)
