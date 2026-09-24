"""FUTURE-R7-M15-AHEAD-SR-NONACCEPTANCE-TP-V1 (Amendment A1: GROSS-primary).

Tests whether realizing profit at the FIRST encounter with the frozen ahead-SR
target zone -- when that zone is touched but NOT closed through on the same
completed 15m bar, and the trade is already profitable there -- improves the
PAIRED GROSS TD5 outcome vs carrying the same trade to the frozen TD horizon.

Governance (Amendment A1, carried forward unchanged from R6):
  * No canonical real transaction-cost / slippage owner exists in this repo
    (canonical_table_found=False, REALISTIC_NET_PNL=UNAVAILABLE_COST_METADATA).
  * Primary quantity is therefore GROSS paired policy effect, NOT a NET verdict.
  * No friction proxy is used for the verdict.

Downstream STOP policy is FROZEN by R6:

    NO_STRUCTURAL_STOP

R7 studies Take-Profit ONLY and never applies the R6 Structural Stop.

Rule (frozen, exactly one Primary TP rule, no tunable parameter):
  Target = the frozen AHEAD SR zone taken from the Candidate decision/entry env
    LONG : resistance zone [raw_res_bottom, raw_res_top]
    SHORT: support    zone [raw_sup_bottom, raw_sup_top]
  touch        LONG: High_t  >= zone_bottom   ; SHORT: Low_t <= zone_top
  close-through LONG: Close_t >  zone_top     ; SHORT: Close_t < zone_bottom
  The FIRST encounter owns the TP decision permanently:
    accepted_same_bar  (cross == touch) -> NO TP ever from this frozen zone
    r_touch_close <= 0                  -> NO TP ever from this frozen zone
    otherwise                           -> TP at the OPEN of the NEXT valid
                                           canonical 15m execution bar, only if
                                           same hard segment AND next bar occurs
                                           before the Baseline_H exit.
  `r_touch_close > 0` is NOT a tuned threshold: zero is the semantic boundary
  between taking profit and exiting a non-profitable trade.
"""

import json
import os
import time

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.entry_path_atlas_v1 import (
    SYMBOLS,
    TD_CHECKPOINTS,
    load_symbol_state,
    build_horizon_indices,
)
from research.liquidity_oracle_atlas.structural_stop_v1 import (
    paired_gid_bootstrap,
    weighted_mean,
    weighted_quantile,
    sha256_file,
    write_csv,
    write_json,
)

# --------------------------------------------------------------------------- #
# 0. Constants / governance                                                    #
# --------------------------------------------------------------------------- #
TASK_ID = "FUTURE-R7-M15-AHEAD-SR-NONACCEPTANCE-TP-V1"
# Lineage is explicit at BOTH levels.
BASE_SHA = "273be9d514259d2a7a585cc27b0531f6f51476d7"
# RC-R7-14: reviewed_parent_sha is the immediate reviewed R7 parent
# (f28f029 = the PRE-T2 T1.5 evidence that this patch corrects).
REVIEWED_PARENT = "f28f02948efc52d91da8d6196c8e9fbabeab5b2a"

ARTIFACT_DIR = os.path.join("artifacts", "ahead_sr_take_profit_v1")
EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")

# §31: stage identity is unambiguous. The PRE-T2 (T1.5) integration stage owns
# stage-explicit `*_t1_5_*` names; the canonical un-prefixed names are reserved
# for the Formal stage only.
T1_5_PRIMARY_CSV = os.path.join(EVIDENCE_DIR,
                                "ahead_sr_take_profit_v1_t1_5_primary.csv")
T1_5_SIDE_STATS_CSV = os.path.join(EVIDENCE_DIR,
                                   "ahead_sr_take_profit_v1_t1_5_side_stats.csv")
T1_5_GROUP_STATS_CSV = os.path.join(EVIDENCE_DIR,
                                    "ahead_sr_take_profit_v1_t1_5_group_stats.csv")
T1_5_EVENTS_CSV = os.path.join(EVIDENCE_DIR,
                               "ahead_sr_take_profit_v1_t1_5_events.csv")
T1_5_SUMMARY_JSON = os.path.join(EVIDENCE_DIR,
                                 "ahead_sr_take_profit_v1_t1_5_summary.json")
T1_5_MANIFEST_JSON = os.path.join(EVIDENCE_DIR,
                                  "ahead_sr_take_profit_v1_t1_5_manifest.json")

# Canonical Formal evidence (FULL 13773-Candidate population only).
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "ahead_sr_take_profit_v1_manifest.json")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "ahead_sr_take_profit_v1_summary.json")
PRIMARY_CSV = os.path.join(EVIDENCE_DIR, "ahead_sr_take_profit_v1_primary.csv")
GROUP_STATS_CSV = os.path.join(EVIDENCE_DIR,
                               "ahead_sr_take_profit_v1_group_stats.csv")
EVENTS_CSV = os.path.join(EVIDENCE_DIR, "ahead_sr_take_profit_v1_events.csv")
SIDE_STATS_CSV = os.path.join(EVIDENCE_DIR,
                              "ahead_sr_take_profit_v1_side_stats.csv")

STAGE_T1_5 = "t1_5_integration"
STAGE_FORMAL = "formal_r7"
ENV_CONTRACT_ID = "FUTURE-R4-M15-ENVIRONMENT-V1"

L2_PARQUET = os.path.join("artifacts", "entry_path_atlas_v1",
                          "entry_path_row_metrics_v1.parquet")
R6_MANIFEST = os.path.join(EVIDENCE_DIR, "structural_stop_v1_manifest.json")
R6_SUMMARY = os.path.join(EVIDENCE_DIR, "structural_stop_v1_summary.json")

FROZEN_INPUTS = {
    L2_PARQUET:
        "110d5990fd9b3bc17aca65a92eb172e0de088635e89c364428b8c9bfccf94d04",
    R6_MANIFEST:
        "85c60fa975bb7cf77ce1613f716b0652a751b8d659a352bfcf2ca63e5147179c",
    R6_SUMMARY:
        "6776ffd056f084f7186b7f3734d551644d9cfd5f2216cd9f43d01835bcedf85e",
}

HORIZONS = ("td1", "td3", "td5")
PRIMARY_HORIZON = "td5"
ROBUSTNESS_HORIZONS = ("td1", "td3")
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_B = 2000

# cost governance (Amendment A1, unchanged)
COST_METADATA_STATUS = "UNAVAILABLE_COST_METADATA"
REALISTIC_NET_PNL_STATUS = "NOT_ESTIMATED"
FORMAL_PRIMARY_BASIS = "GROSS_PAIRED_POLICY_EFFECT"
FRICTION_PROXY_USED_FOR_VERDICT = False
ENTRY_COST_ESTIMATED = False
EXIT_COST_ESTIMATED = False

# §2 / §23: R6 closed the Stop line; R7 consumes the verdict mechanically.
R6_DOWNSTREAM_POLICY_REQUIRED = {
    "stage": "formal_r6",
    "all_pass": True,
    "formal_verdict": "NO_IDENTIFIABLE_GROSS_STOP_EDGE",
}
FROZEN_DOWNSTREAM_STOP = "NO_STRUCTURAL_STOP"

TP_STATES = ("not_eligible", "no_touch", "accepted_breakout",
             "touch_not_profitable", "no_signal_before_baseline_exit",
             "no_next_bar", "segment_change", "beyond_baseline_exit",
             "tp_executed")

COUNTERS = {"r5_l2_loads": 0, "execution_frame_loads": 0, "direction_reruns": 0,
            "sr_recompute_count": 0, "path_atlas_rescans": 0,
            "full_history_recompute_count": 0, "reference_calls": 0,
            "r6_policy_verifications": 0, "production_candidate_views": 0}


def _bump(name, n=1):
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def reset_counters():
    for k in COUNTERS:
        COUNTERS[k] = 0


def verify_frozen_inputs():
    """Fail closed if any required frozen upstream artifact is missing/mismatched."""
    out = {}
    for path, want in FROZEN_INPUTS.items():
        if not os.path.exists(path):
            raise RuntimeError(
                f"STOP_R7_FROZEN_INPUT_ARTIFACT_MISMATCH missing={path}")
        got = sha256_file(path)
        if got != want:
            raise RuntimeError(
                f"STOP_R7_FROZEN_INPUT_ARTIFACT_MISMATCH path={path} "
                f"got={got} want={want}")
        out[path] = got
    return out


# --------------------------------------------------------------------------- #
# 1. R6 downstream-policy verification (§2)                                     #
# --------------------------------------------------------------------------- #
def verify_r6_downstream_policy():
    """Mechanical proof that the downstream STOP policy is NO_STRUCTURAL_STOP."""
    _bump("r6_policy_verifications")
    if not os.path.exists(R6_MANIFEST):
        raise RuntimeError(
            f"STOP_R7_R6_DOWNSTREAM_POLICY_MISSING missing={R6_MANIFEST}")
    with open(R6_MANIFEST) as f:
        man = json.load(f)
    mism = {k: (man.get(k), v) for k, v in R6_DOWNSTREAM_POLICY_REQUIRED.items()
            if man.get(k) != v}
    if mism:
        raise RuntimeError(f"STOP_R7_R6_DOWNSTREAM_POLICY_MISMATCH {mism}")
    return {
        "downstream_stop_policy": FROZEN_DOWNSTREAM_STOP,
        "apply_r6_structural_stop": False,
        "r6_manifest_path": R6_MANIFEST,
        "r6_stage": man["stage"],
        "r6_all_pass": bool(man["all_pass"]),
        "r6_verdict": man["formal_verdict"],
        "r6_authorized_review_sha": man.get("authorized_review_sha"),
        "r6_artifact_sha256": man.get("artifact_sha256", {}),
    }


# --------------------------------------------------------------------------- #
# 2. Frozen R5 L2 loader                                                        #
# --------------------------------------------------------------------------- #
L2_COLUMNS = [
    "semantic_key", "symbol", "gid", "direction_system", "direction",
    "sample_weight_raw", "direction_correct", "oracle_direction",
    "fill_time", "entry_price", "ATR0",
    "raw_sup_top", "raw_sup_bottom", "raw_sup_strength",
    "raw_res_top", "raw_res_bottom", "raw_res_strength",
    "raw_liq_up_top", "raw_liq_up_bottom", "raw_liq_dn_top", "raw_liq_dn_bottom",
    "first_ahead_sr_touch", "first_ahead_sr_cross",
    "first_ahead_liq_touch", "first_ahead_liq_cross",
    "mfe_at_first_ahead_sr", "mae_before_first_ahead_sr",
    "mfe_final", "mae_final",
    "td1_mfe", "td1_mae", "td1_r",
    "td3_mfe", "td3_mae", "td3_r",
    "td5_mfe", "td5_mae", "td5_r",
]


def load_r5_l2():
    _bump("r5_l2_loads")
    return pd.read_parquet(L2_PARQUET, columns=L2_COLUMNS)


# --------------------------------------------------------------------------- #
# 3. Frozen ahead-SR geometry (§4 / §5)                                         #
# --------------------------------------------------------------------------- #
def ahead_sr_zone(is_long, sup_top, sup_bottom, sup_strength,
                  res_top, res_bottom, res_strength):
    """Frozen AHEAD SR TARGET ZONE, taken from the Candidate decision environment.

    LONG  -> resistance zone [raw_res_bottom, raw_res_top]
    SHORT -> support    zone [raw_sup_bottom, raw_sup_top]
    The zone is consumed exactly as frozen: never recomputed, never moved.
    """
    zone_bottom = np.where(is_long, res_bottom, sup_bottom)
    zone_top = np.where(is_long, res_top, sup_top)
    strength = np.where(is_long, res_strength, sup_strength)
    return zone_bottom, zone_top, strength


def ahead_touch_cross_boundaries(is_long, zone_bottom, zone_top):
    """Near edge is TOUCHED first; far edge must be CLOSED THROUGH (§5).

    LONG : touch = High  >= zone_bottom ; cross = Close > zone_top
    SHORT: touch = Low   <= zone_top    ; cross = Close < zone_bottom
    This reproduces the frozen R5 channels
      ahead_sr_touch = w(res_bottom, sup_top)
      ahead_sr_cross = w(res_top,    sup_bottom).
    """
    touch_b = np.where(is_long, zone_bottom, zone_top)
    cross_b = np.where(is_long, zone_top, zone_bottom)
    return touch_b, cross_b


def ahead_liq_zone(is_long, liq_up_top, liq_up_bottom,
                   liq_dn_top, liq_dn_bottom):
    zone_bottom = np.where(is_long, liq_up_bottom, liq_dn_bottom)
    zone_top = np.where(is_long, liq_up_top, liq_dn_top)
    return zone_bottom, zone_top


# --------------------------------------------------------------------------- #
# 4. Production TP signal (§26)                                                 #
# --------------------------------------------------------------------------- #
def production_tp_signal_step(touch_step, cross_step, side, entry_idx,
                              entry_price, atr0, close, eligible):
    """Production derivation from the frozen R5 first-passage event steps (O(N))."""
    touch = np.asarray(touch_step, np.int64)
    cross = np.asarray(cross_step, np.int64)

    # Structural event ordering invariant (§5).
    bad = (cross >= 0) & ((touch < 0) | (cross < touch))
    if bad.any():
        raise RuntimeError("STOP_R7_AHEAD_SR_EVENT_ORDER_MISMATCH")

    has_touch = eligible & (touch >= 0)

    # First touch that closes through the full zone means acceptance (§7).
    accepted_same_bar = has_touch & (cross == touch)

    safe_touch_idx = np.where(has_touch, entry_idx + touch, 0)

    r_touch_close = np.where(
        has_touch, side * (close[safe_touch_idx] - entry_price) / atr0, np.nan)

    profitable = has_touch & np.isfinite(r_touch_close) & (r_touch_close > 0.0)

    signal = has_touch & ~accepted_same_bar & profitable

    signal_step = np.where(signal, touch, -1).astype(np.int64)

    return {
        "signal_step": signal_step,
        "has_touch": has_touch,
        "accepted_same_bar": accepted_same_bar,
        "r_touch_close": r_touch_close,
        "profitable_at_touch": profitable,
    }


def signal_level_reason(eligible, touch_observed, accepted, profitable, signal):
    """Terminal classification of the FIRST encounter, for ONE observation window.

    RC-R7-1/RC-R7-3: `touch_observed` / `accepted` / `profitable` must already be
    HORIZON-CENSORED. A first touch occurring after Baseline_H exit is not
    observed at H, so such a row is classified `no_touch` at H -- never
    `accepted_breakout` and never `touch_not_profitable`.
    """
    touch_observed = np.asarray(touch_observed, bool)
    eligible = np.asarray(eligible, bool)
    accepted = np.asarray(accepted, bool)
    profitable = np.asarray(profitable, bool)
    signal = np.asarray(signal, bool)
    r = np.full(len(touch_observed), "", dtype=object)
    r[~eligible] = "not_eligible"
    r[eligible & ~touch_observed] = "no_touch"
    r[eligible & touch_observed & accepted] = "accepted_breakout"
    r[eligible & touch_observed & ~accepted & ~profitable] = \
        "touch_not_profitable"
    r[eligible & touch_observed & ~accepted & profitable & signal] = "tp_signal"
    return r


def horizon_censor(frozen_touch, frozen_cross, exit_step):
    """RC-R7-1: horizon-causal view of the frozen first-encounter event steps.

    touch_observed_H = (frozen_touch >= 0) AND (frozen_touch <= exit_step_H)
    cross_observed_H = (frozen_cross >= 0) AND (frozen_cross <= exit_step_H)

    A first touch that occurs AFTER Baseline_H exit is invisible in the H row.
    """
    frozen_touch = np.asarray(frozen_touch, np.int64)
    frozen_cross = np.asarray(frozen_cross, np.int64)
    exit_step = np.asarray(exit_step, np.int64)
    touch_observed = (frozen_touch >= 0) & (frozen_touch <= exit_step)
    cross_observed = (frozen_cross >= 0) & (frozen_cross <= exit_step)
    return touch_observed, cross_observed


# --------------------------------------------------------------------------- #
# 5. Reference TP signal (§27)                                                  #
# --------------------------------------------------------------------------- #
def reference_tp_signal_step(side, entry_idx, end_idx, entry_price, atr0,
                             high, low, close, segment, entry_segment,
                             ahead_bottom, ahead_top):
    """One candidate: independent literal bar scan of the frozen touch/cross rule."""
    if not (np.isfinite(ahead_bottom) and np.isfinite(ahead_top)):
        return -1, "not_eligible"
    s = float(side)
    for step in range(int(end_idx) - int(entry_idx) + 1):
        j = int(entry_idx) + step
        if j >= len(close) or j < 0:
            break
        if segment[j] != entry_segment:
            break
        if s > 0:
            touched = high[j] >= ahead_bottom
            crossed = close[j] > ahead_top
        else:
            touched = low[j] <= ahead_top
            crossed = close[j] < ahead_bottom
        if not touched:
            continue
        # First encounter owns the decision permanently.
        if crossed:
            return -1, "accepted_breakout"
        r_close = s * (close[j] - entry_price) / atr0
        if r_close <= 0.0:
            return -1, "touch_not_profitable"
        return step, "tp_signal"
    return -1, "no_touch"


def scan_tp_reference(case):
    """Reference (bar-scan) signal for ONE symbol/system case. T0/T1/T1.5 only."""
    _bump("reference_calls")
    n = int(len(case["entry_idx"]))
    sides = np.asarray(case["side"], float)
    e = np.asarray(case["entry_idx"], np.int64)
    end = np.asarray(case["end_idx"], np.int64)
    atr0 = np.asarray(case["atr0"], float)
    p0 = np.asarray(case["entry_price"], float)
    ent_seg = np.asarray(case["entry_segment"], np.int64)
    hi = np.asarray(case["high"], float)
    lo = np.asarray(case["low"], float)
    cl = np.asarray(case["close"], float)
    seg = np.asarray(case["segment"], np.int64)
    tb = np.asarray(case["ahead_touch_boundary"], float)
    cb = np.asarray(case["ahead_cross_boundary"], float)
    # §27 names the geometric ZONE edges; the touch/cross comparison flips by side.
    zb = np.asarray(case["zone_bottom"], float)
    zt = np.asarray(case["zone_top"], float)
    eligible = np.isfinite(tb) & np.isfinite(cb)

    sig = np.full(n, -1, np.int64)
    touch = np.full(n, -1, np.int64)
    cross = np.full(n, -1, np.int64)
    accepted = np.zeros(n, bool)
    profitable = np.zeros(n, bool)
    r_close = np.full(n, np.nan)
    reason = np.full(n, "", dtype=object)

    for i in range(n):
        step, why = reference_tp_signal_step(
            side=sides[i], entry_idx=e[i], end_idx=end[i], entry_price=p0[i],
            atr0=atr0[i], high=hi, low=lo, close=cl, segment=seg,
            entry_segment=ent_seg[i], ahead_bottom=zb[i], ahead_top=zt[i])
        reason[i] = why
        if not eligible[i]:
            continue
        # Independent re-scan of the two frozen event STEPS (for §32 comparison).
        t = -1
        c = -1
        for st_ in range(int(end[i]) - int(e[i]) + 1):
            j = int(e[i]) + st_
            if j >= len(cl) or seg[j] != ent_seg[i]:
                break
            if sides[i] > 0:
                if t < 0 and hi[j] >= tb[i]:
                    t = st_
                if c < 0 and cl[j] > cb[i]:
                    c = st_
            else:
                if t < 0 and lo[j] <= tb[i]:
                    t = st_
                if c < 0 and cl[j] < cb[i]:
                    c = st_
            if t >= 0 and c >= 0:
                break
        touch[i] = t
        cross[i] = c
        if t >= 0:
            safe_j = int(e[i]) + int(t)
            rc = sides[i] * (cl[safe_j] - p0[i]) / atr0[i]
            r_close[i] = rc
            accepted[i] = bool(c == t)
            profitable[i] = bool(np.isfinite(rc) and rc > 0.0)
        if why == "tp_signal":
            sig[i] = step
    return {
        "tp_eligible": eligible,
        "first_touch_step": touch,
        "first_cross_step": cross,
        "accepted_same_bar": accepted,
        "r_touch_close": r_close,
        "profitable_at_touch": profitable,
        "signal_step": sig,
        "signal_reason": reason,
    }


def scan_tp_production(case):
    """Vectorized production kernel, reusing the frozen R5 event steps (§25)."""
    n = int(len(case["entry_idx"]))
    _bump("production_candidate_views", n)
    eligible = case["tp_eligible"]
    frozen_touch = np.asarray(case["frozen_touch_step"], np.int64)
    frozen_cross = np.asarray(case["frozen_cross_step"], np.int64)
    out = production_tp_signal_step(
        touch_step=frozen_touch, cross_step=frozen_cross,
        side=np.asarray(case["side"], float),
        entry_idx=np.asarray(case["entry_idx"], np.int64),
        entry_price=np.asarray(case["entry_price"], float),
        atr0=np.asarray(case["atr0"], float),
        close=np.asarray(case["close"], float), eligible=eligible)
    out["first_touch_step"] = np.where(eligible, frozen_touch, -1).astype(np.int64)
    out["first_cross_step"] = np.where(eligible, frozen_cross, -1).astype(np.int64)
    out["tp_eligible"] = eligible
    out["per_h"] = {}
    for name in HORIZONS:
        exit_step = np.asarray(case["td_ends"][name], np.int64) \
            - np.asarray(case["entry_idx"], np.int64)
        out["per_h"][name] = resolve_tp_fill(
            signal_step=out["signal_step"],
            entry_idx=np.asarray(case["entry_idx"], np.int64),
            side=np.asarray(case["side"], float),
            entry_price=np.asarray(case["entry_price"], float),
            atr0=np.asarray(case["atr0"], float),
            open_px=np.asarray(case["open"], float),
            segment=np.asarray(case["segment"], np.int64),
            entry_segment=np.asarray(case["entry_segment"], np.int64),
            exit_step_h=exit_step)
        out["per_h"][name]["exit_step"] = exit_step
    return out


# --------------------------------------------------------------------------- #
# 6. Horizon-causal next-open fill (§9 / §10 / §28)                             #
# --------------------------------------------------------------------------- #
def resolve_tp_fill(signal_step, entry_idx, side, entry_price, atr0,
                    open_px, segment, entry_segment, exit_step_h):
    """Causal next-valid-open TP fill for ONE horizon H."""
    signal_step = np.asarray(signal_step, np.int64)
    exit_step_h = np.asarray(exit_step_h, np.int64)

    observed = (signal_step >= 0) & (signal_step <= exit_step_h)

    signal_idx = np.where(observed, entry_idx + signal_step, -1)

    next_idx = np.where(observed, signal_idx + 1, -1)

    in_range = observed & (next_idx >= 0) & (next_idx < len(open_px))

    safe_next = np.where(in_range, next_idx, 0)

    same_segment = in_range & (segment[safe_next] == entry_segment)

    before_baseline = same_segment & (signal_step + 1 <= exit_step_h)

    executable = before_baseline

    fill_open = np.where(executable, open_px[safe_next], np.nan)

    tp_return = np.where(
        executable, side * (fill_open - entry_price) / atr0, np.nan)

    return {
        "signal_observed": observed,
        "executable": executable,
        "fill_step": np.where(executable, signal_step + 1, -1),
        "fill_idx": np.where(executable, next_idx, -1),
        "fill_open": fill_open,
        "tp_return": tp_return,
    }


def horizon_reason(sig_reason0, observed, in_range, same_segment, executable):
    """§28 reason taxonomy, resolved per horizon row."""
    r = np.array(sig_reason0, dtype=object).copy()
    m = (r == "tp_signal")
    r[m & ~observed] = "no_signal_before_baseline_exit"
    r[m & observed & ~in_range] = "no_next_bar"
    r[m & observed & in_range & ~same_segment] = "segment_change"
    r[m & observed & in_range & same_segment & ~executable] = \
        "beyond_baseline_exit"
    r[m & executable] = "tp_executed"
    return r


# --------------------------------------------------------------------------- #
# 7. Row artifact (§30)                                                         #
# --------------------------------------------------------------------------- #
ROW_COLUMNS = [
    "semantic_key", "symbol", "gid", "direction_system", "direction",
    "sample_weight_raw", "direction_correct",
    "entry_time", "entry_price", "ATR0",
    "evaluation_horizon",
    "baseline_exit_time", "baseline_exit_price", "baseline_gross_return_atr",
    "tp_eligible",
    "first_ahead_sr_touch_step", "first_ahead_sr_cross_step",
    "first_touch_accepted_breakout",
    "tp_signal_observed", "tp_signal_step", "tp_signal_time",
    "r_touch_close", "mfe_at_first_ahead_sr", "mfe_final",
    "first_encounter_class_global_audit",
    "tp_executable", "tp_fill_step", "tp_fill_time", "tp_fill_open",
    "tp_gross_return_atr", "paired_delta_gross_atr",
    "saved_giveback_atr", "lost_continuation_atr",
    "frozen_ahead_sr_bottom", "frozen_ahead_sr_top", "frozen_ahead_sr_strength",
    "ahead_liq_available", "ahead_liq_touched_by_signal",
    "ahead_liq_crossed_by_signal",
    "tp_reason",
]


def _prepare_symbol(df, st):
    """Align frozen fill_time -> canonical execution-frame index."""
    n_bars = int(st.n_bars)
    fill_t = pd.to_datetime(df["fill_time"]).to_numpy("datetime64[ns]")
    bar_t = pd.to_datetime(st.bar_start_time).to_numpy("datetime64[ns]")
    lookup = {int(t.astype("int64")): i for i, t in enumerate(bar_t)}
    entry_idx = np.array([lookup.get(int(t.astype("int64")), -1) for t in fill_t],
                         dtype=np.int64)
    if (entry_idx < 0).any():
        raise RuntimeError("STOP_R7_FILL_TIME_NOT_IN_FRAME")
    entry_price = df["entry_price"].to_numpy(float)
    if not np.allclose(st.open[entry_idx], entry_price, atol=1e-9):
        raise RuntimeError("STOP_R7_ENTRY_PRICE_ALIGNMENT_MISMATCH")
    end_idx, td_ends = build_horizon_indices(st, entry_idx)
    is_long = (df["direction"].to_numpy(object) == "LONG")
    zb, zt, zstrength = ahead_sr_zone(
        is_long,
        df["raw_sup_top"].to_numpy(float), df["raw_sup_bottom"].to_numpy(float),
        df["raw_sup_strength"].to_numpy(float),
        df["raw_res_top"].to_numpy(float), df["raw_res_bottom"].to_numpy(float),
        df["raw_res_strength"].to_numpy(float))
    tb, cb = ahead_touch_cross_boundaries(is_long, zb, zt)
    lqb, lqt = ahead_liq_zone(
        is_long,
        df["raw_liq_up_top"].to_numpy(float),
        df["raw_liq_up_bottom"].to_numpy(float),
        df["raw_liq_dn_top"].to_numpy(float),
        df["raw_liq_dn_bottom"].to_numpy(float))
    eligible = np.isfinite(tb) & np.isfinite(cb)
    return {
        "df": df, "entry_idx": entry_idx, "end_idx": end_idx, "td_ends": td_ends,
        "side": np.where(is_long, 1.0, -1.0), "is_long": is_long,
        "entry_price": entry_price, "atr0": df["ATR0"].to_numpy(float),
        "entry_segment": st.segment[entry_idx],
        "open": st.open, "high": st.high, "low": st.low, "close": st.close,
        "segment": st.segment, "n_bars": n_bars,
        "zone_bottom": zb, "zone_top": zt, "zone_strength": zstrength,
        "ahead_touch_boundary": tb, "ahead_cross_boundary": cb,
        "liq_zone_bottom": lqb, "liq_zone_top": lqt,
        "tp_eligible": eligible,
        "frozen_touch_step": df["first_ahead_sr_touch"].to_numpy(np.int64),
        "frozen_cross_step": df["first_ahead_sr_cross"].to_numpy(np.int64),
        "liq_available": np.isfinite(lqb) & np.isfinite(lqt),
    }


def treatment_rows_for_symbol(df_l2, st, direction_system):
    """Production: per (Candidate x horizon) treatment rows for one direction system."""
    sub = df_l2[df_l2["direction_system"] == direction_system].sort_values(
        "semantic_key").reset_index(drop=True)
    if len(sub) == 0:
        return pd.DataFrame(columns=ROW_COLUMNS), None
    case = _prepare_symbol(sub, st)
    prod = scan_tp_production(case)
    N = int(len(sub))
    bst = pd.to_datetime(st.bar_start_time)
    decision_time = (bst + pd.Timedelta(minutes=15)).to_numpy("datetime64[ns]")
    bst_np = bst.to_numpy("datetime64[ns]")
    n_bars = case["n_bars"]

    eligible = prod["tp_eligible"]
    frozen_touch = np.asarray(prod["first_touch_step"], np.int64)
    frozen_cross = np.asarray(prod["first_cross_step"], np.int64)
    r_close_global = prod["r_touch_close"]
    mfe_touch_global = sub["mfe_at_first_ahead_sr"].to_numpy(float)

    # RC-R7-3: the five-day global classification is retained ONLY as an
    # explicitly AUDIT_ONLY column; it never drives a horizon row's reason.
    global_audit = signal_level_reason(
        eligible, frozen_touch >= 0, prod["accepted_same_bar"],
        prod["profitable_at_touch"], prod["signal_step"] >= 0)

    frames = []
    for H in HORIZONS:
        m = prod["per_h"][H]
        exit_step = m["exit_step"].astype(np.int64)

        # ---- RC-R7-1: horizon-causal first-encounter view ----
        touch_obs, cross_obs = horizon_censor(frozen_touch, frozen_cross,
                                              exit_step)
        touch_step_H = np.where(touch_obs, frozen_touch, -1).astype(np.int64)
        cross_step_H = np.where(cross_obs, frozen_cross, -1).astype(np.int64)
        accepted_H = touch_obs & (frozen_cross == frozen_touch)
        r_close_H = np.where(touch_obs, r_close_global, np.nan)
        mfe_touch_H = np.where(touch_obs, mfe_touch_global, np.nan)
        profitable_H = (touch_obs & np.isfinite(r_close_H)
                        & (r_close_H > 0.0))
        signal_H = prod["signal_step"] >= 0
        sig_reason0 = signal_level_reason(
            eligible, touch_obs, accepted_H, profitable_H, signal_H)
        # §11: baseline exit price is the ACTUAL canonical frame close.
        safe_exit = np.clip(case["entry_idx"] + exit_step, 0, n_bars - 1)
        base_close = case["close"][safe_exit]
        base_r = sub[f"{H}_r"].to_numpy(float)
        recon = case["side"] * (base_close - case["entry_price"]) / case["atr0"]
        if not np.allclose(recon, base_r, atol=1e-12, equal_nan=True):
            raise RuntimeError("STOP_R7_BASELINE_EXIT_ALIGNMENT_MISMATCH")
        base_time = bst[safe_exit].to_numpy("datetime64[ns]") \
            + np.timedelta64(15, "m")

        fill_open = m["fill_open"]
        tp_gross = np.where(
            m["executable"],
            case["side"] * (fill_open - case["entry_price"]) / case["atr0"],
            np.nan)
        tp_ret = np.where(m["executable"], tp_gross, base_r)
        delta = tp_ret - base_r

        sig_idx = np.where(m["signal_observed"],
                           case["entry_idx"] + touch_step_H, -1)
        safe_sig = np.clip(sig_idx, 0, n_bars - 1)
        sig_time = np.where(m["signal_observed"], decision_time[safe_sig],
                            np.datetime64("NaT", "ns"))
        next_idx = np.where(m["signal_observed"], sig_idx + 1, -1)
        in_range = m["signal_observed"] & (next_idx >= 0) & (next_idx < n_bars)
        safe_next = np.where(in_range, next_idx, 0)
        same_segment = in_range & (case["segment"][safe_next]
                                   == case["entry_segment"])
        fill_idx = np.where(m["executable"], m["fill_idx"], -1)
        safe_fill = np.clip(fill_idx, 0, n_bars - 1)
        fill_time = np.where(m["executable"], bst_np[safe_fill],
                             np.datetime64("NaT", "ns"))
        rsn = horizon_reason(sig_reason0, m["signal_observed"], in_range,
                             same_segment, m["executable"])

        liq_touch = sub["first_ahead_liq_touch"].to_numpy(np.int64)
        liq_cross = sub["first_ahead_liq_cross"].to_numpy(np.int64)
        tp_sig_step = np.where(m["signal_observed"], touch_step_H, -1)

        rec = pd.DataFrame({
            "semantic_key": sub["semantic_key"].to_numpy(object),
            "symbol": sub["symbol"].to_numpy(object),
            "gid": sub["gid"].to_numpy(object),
            "direction_system": direction_system,
            "direction": sub["direction"].to_numpy(object),
            "sample_weight_raw": sub["sample_weight_raw"].to_numpy(float),
            "direction_correct": sub["direction_correct"].to_numpy(np.uint8),
            "entry_time": pd.to_datetime(
                sub["fill_time"]).to_numpy("datetime64[ns]"),
            "entry_price": case["entry_price"], "ATR0": case["atr0"],
            "evaluation_horizon": H,
            "baseline_exit_time": base_time,
            "baseline_exit_price": base_close,
            "baseline_gross_return_atr": base_r,
            "tp_eligible": prod["tp_eligible"],
            "first_ahead_sr_touch_step": touch_step_H,
            "first_ahead_sr_cross_step": cross_step_H,
            "first_touch_accepted_breakout": accepted_H,
            "tp_signal_observed": m["signal_observed"],
            "tp_signal_step": tp_sig_step.astype(np.int64),
            "tp_signal_time": sig_time,
            "r_touch_close": r_close_H,
            "mfe_at_first_ahead_sr": mfe_touch_H,
            # RC-R7-13: `mfe_final` is a retrospective TD5/path outcome carried
            # ONLY for the pre-registered RemainingMFE descriptive. AUDIT ONLY.
            "mfe_final": sub["mfe_final"].to_numpy(float),
            # RC-R7-3: five-day global classification, AUDIT ONLY, never used by
            # the horizon-specific reason `tp_reason` or by any H-specific rate.
            "first_encounter_class_global_audit": global_audit,
            "tp_executable": m["executable"],
            "tp_fill_step": m["fill_step"].astype(np.int64),
            "tp_fill_time": fill_time.astype("datetime64[ns]"),
            "tp_fill_open": fill_open,
            "tp_gross_return_atr": tp_ret,
            "paired_delta_gross_atr": delta,
            "saved_giveback_atr": np.maximum(delta, 0.0),
            "lost_continuation_atr": np.maximum(-delta, 0.0),
            "frozen_ahead_sr_bottom": case["zone_bottom"],
            "frozen_ahead_sr_top": case["zone_top"],
            "frozen_ahead_sr_strength": case["zone_strength"],
            "ahead_liq_available": case["liq_available"],
            "ahead_liq_touched_by_signal":
                (liq_touch >= 0) & (liq_touch <= tp_sig_step),
            "ahead_liq_crossed_by_signal":
                (liq_cross >= 0) & (liq_cross <= tp_sig_step),
            "tp_reason": rsn,
        })[ROW_COLUMNS]
        frames.append(rec)
    return pd.concat(frames, ignore_index=True), case


# --------------------------------------------------------------------------- #
# 8. Effect estimation (§14 / §15 / §16)                                        #
# --------------------------------------------------------------------------- #
def touch_masks(q):
    """Signal-level masks shared by every effect/diagnostic consumer.

    NOT_PROFITABLE is the horizon-independent terminal classification of the
    FIRST encounter (§7): touched, NOT accepted through, r_touch_close <= 0.
    """
    touch_pos = (q["first_ahead_sr_touch_step"].to_numpy(np.int64) >= 0)
    accepted = q["first_touch_accepted_breakout"].to_numpy(bool)
    r_close = q["r_touch_close"].to_numpy(float)
    not_profitable = touch_pos & ~accepted & np.isfinite(r_close) & (r_close <= 0)
    return touch_pos, accepted, r_close, not_profitable


def _not_profitable_mask(q):
    return touch_masks(q)[3]


def _boot_sub(vals, gid, w, mask, name):
    if not np.asarray(mask, bool).any():
        return {name: float("nan"), f"{name}_ci_low": float("nan"),
                f"{name}_ci_high": float("nan"), f"{name}_n_rows": 0,
                f"{name}_n_gids": 0}
    b = paired_gid_bootstrap(vals[mask], np.asarray(gid, object)[mask], w[mask],
                             B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    return {name: b["point"], f"{name}_ci_low": b["ci_low"],
            f"{name}_ci_high": b["ci_high"],
            f"{name}_n_rows": b["n_rows"], f"{name}_n_gids": b["n_gids"]}


def _effect(big, system, horizon, side=None):
    """Weighted, whole-gid-bootstrapped GROSS TP effect + decomposition."""
    q = big[(big.direction_system == system)
            & (big.evaluation_horizon == horizon)]
    if side is not None:
        q = q[q.direction == side]
    if len(q) == 0:
        return None
    base = q["baseline_gross_return_atr"].to_numpy(float)
    tp = q["tp_gross_return_atr"].to_numpy(float)
    w = q["sample_weight_raw"].to_numpy(float)
    gid = q["gid"].to_numpy(object)
    corr = q["direction_correct"].to_numpy(bool)
    delta = tp - base
    sg = np.maximum(delta, 0.0)
    lc = np.maximum(-delta, 0.0)
    dm = ~corr
    cm = corr

    ev = paired_gid_bootstrap(delta, gid, w, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    row = {
        "system": system, "horizon": horizon, "side": side or "ALL",
        "n_rows": int(len(q)), "n_gids": int(ev["n_gids"]),
        "delta_ev_gross": ev["point"],
        "delta_ev_ci_low": ev["ci_low"], "delta_ev_ci_high": ev["ci_high"],
        "break_even_incremental_exit_cost_atr": ev["point"],
        "break_even_incremental_exit_cost_ci_low": ev["ci_low"],
        "break_even_incremental_exit_cost_ci_high": ev["ci_high"],
        "has_positive_gross_cost_capacity": bool(ev["point"] > 0),
        **_boot_sub(sg, gid, w, np.ones(len(q), bool), "saved_giveback"),
        **_boot_sub(lc, gid, w, np.ones(len(q), bool), "lost_continuation"),
        **_boot_sub(delta, gid, w, cm, "delta_correct_gross"),
        **_boot_sub(delta, gid, w, dm, "delta_wrong_gross"),
        "tp_eligible_rate": weighted_mean(q["tp_eligible"].to_numpy(float), w),
        "first_touch_rate": weighted_mean(
            (q["first_ahead_sr_touch_step"] >= 0).astype(float), w),
        "first_touch_accepted_breakout_rate": weighted_mean(
            q["first_touch_accepted_breakout"].to_numpy(float), w),
        "first_touch_not_profitable_rate": weighted_mean(
            _not_profitable_mask(q).astype(float), w),
        "tp_signal_rate": weighted_mean(
            q["tp_signal_observed"].to_numpy(float), w),
        "tp_executable_rate": weighted_mean(
            q["tp_executable"].to_numpy(float), w),
        "tp_hit_rate_correct": weighted_mean(
            q["tp_executable"].to_numpy(float)[cm], w[cm])
            if cm.any() else float("nan"),
        "tp_hit_rate_wrong": weighted_mean(
            q["tp_executable"].to_numpy(float)[dm], w[dm])
            if dm.any() else float("nan"),
        "raw_n_correct": int(cm.sum()), "raw_n_wrong": int(dm.sum()),
    }
    return row


GROUP_STAT_COLUMNS = [
    "symbol", "direction_system", "evaluation_horizon", "n_rows", "n_gids",
    "delta_ev_gross", "baseline_gross_mean", "tp_gross_mean",
    "tp_eligible_rate", "tp_signal_rate", "tp_executable_rate",
    "saved_giveback", "lost_continuation",
    "raw_n_rows", "raw_delta_ev_gross", "raw_baseline_gross_mean",
    "raw_tp_gross_mean",
]


def group_stat_rows(big):
    """Per-symbol/group evidence under the frozen FORMAL (weighted) estimator."""
    rows = []
    for sym in sorted(big.symbol.unique()):
        for ds in ("A9", "E9"):
            for H in HORIZONS:
                q = big[(big.symbol == sym) & (big.direction_system == ds)
                        & (big.evaluation_horizon == H)]
                if len(q) == 0:
                    continue
                w = q["sample_weight_raw"].to_numpy(float)
                delta = q["paired_delta_gross_atr"].to_numpy(float)
                base = q["baseline_gross_return_atr"].to_numpy(float)
                tp = q["tp_gross_return_atr"].to_numpy(float)
                rows.append({
                    "symbol": sym, "direction_system": ds,
                    "evaluation_horizon": H,
                    "n_rows": int(len(q)), "n_gids": int(q["gid"].nunique()),
                    "delta_ev_gross": weighted_mean(delta, w),
                    "baseline_gross_mean": weighted_mean(base, w),
                    "tp_gross_mean": weighted_mean(tp, w),
                    "tp_eligible_rate": weighted_mean(
                        q["tp_eligible"].to_numpy(float), w),
                    "tp_signal_rate": weighted_mean(
                        q["tp_signal_observed"].to_numpy(float), w),
                    "tp_executable_rate": weighted_mean(
                        q["tp_executable"].to_numpy(float), w),
                    "saved_giveback": weighted_mean(
                        np.maximum(delta, 0.0), w),
                    "lost_continuation": weighted_mean(
                        np.maximum(-delta, 0.0), w),
                    "raw_n_rows": int(len(q)),
                    "raw_delta_ev_gross": float(delta.mean()) if len(q) else float("nan"),
                    "raw_baseline_gross_mean": float(base.mean()) if len(q) else float("nan"),
                    "raw_tp_gross_mean": float(tp.mean()) if len(q) else float("nan"),
                })
    return rows


EVENT_COLUMNS = [
    "semantic_key", "symbol", "gid", "direction_system", "direction",
    "evaluation_horizon", "direction_correct",
    "first_ahead_sr_touch_step", "first_ahead_sr_cross_step",
    "first_touch_accepted_breakout", "r_touch_close", "mfe_at_first_ahead_sr",
    "tp_signal_step", "tp_signal_time", "tp_executable", "tp_fill_step",
    "tp_fill_time", "tp_fill_open", "tp_reason",
    "baseline_gross_return_atr", "tp_gross_return_atr",
    "paired_delta_gross_atr",
    "frozen_ahead_sr_bottom", "frozen_ahead_sr_top", "frozen_ahead_sr_strength",
    "ahead_liq_available", "ahead_liq_touched_by_signal",
    "ahead_liq_crossed_by_signal",
]


def events_frame(big):
    """Observed TP-signal event table (both stages share it)."""
    ev = big[big["tp_signal_observed"]]
    return ev[EVENT_COLUMNS] if len(ev) else pd.DataFrame(columns=EVENT_COLUMNS)


# --------------------------------------------------------------------------- #
# 9. TP diagnostics (§17 / §18 / §19)                                           #
# --------------------------------------------------------------------------- #
def formal_tp_diagnostics(big, system="E9", horizon="td5"):
    """Weighted diagnostics for the primary cell. AUDIT ONLY."""
    q = big[(big.direction_system == system)
            & (big.evaluation_horizon == horizon)]
    w = q["sample_weight_raw"].to_numpy(float)
    corr = q["direction_correct"].to_numpy(bool)
    ex = q["tp_executable"].to_numpy(bool)
    sig = q["tp_signal_observed"].to_numpy(bool)
    elig = q["tp_eligible"].to_numpy(bool)
    touch_pos, accepted, r_close, not_profitable = touch_masks(q)
    base = q["baseline_gross_return_atr"].to_numpy(float)
    tp = q["tp_gross_return_atr"].to_numpy(float)
    d = tp - base
    ss = q["tp_signal_step"].to_numpy(float)
    fs = q["tp_fill_step"].to_numpy(float)
    mfe_t = q["mfe_at_first_ahead_sr"].to_numpy(float)
    mfe_fin = q["mfe_final"].to_numpy(float)

    out = {
        "system": system, "horizon": horizon,
        "tp_eligible_rate": weighted_mean(elig.astype(float), w),
        "first_touch_rate": weighted_mean(touch_pos.astype(float), w),
        "first_touch_accepted_breakout_rate": weighted_mean(accepted.astype(float), w),
        "first_touch_not_profitable_rate": weighted_mean(
            not_profitable.astype(float), w),
        "tp_signal_rate": weighted_mean(sig.astype(float), w),
        "tp_executable_rate": weighted_mean(ex.astype(float), w),
        "tp_hit_rate_correct": weighted_mean(ex.astype(float)[corr], w[corr])
            if corr.any() else float("nan"),
        "tp_hit_rate_wrong": weighted_mean(ex.astype(float)[~corr], w[~corr])
            if (~corr).any() else float("nan"),
        "signal_step_p25": weighted_quantile(ss[sig], w[sig], 0.25),
        "signal_step_median": weighted_quantile(ss[sig], w[sig], 0.50),
        "signal_step_p75": weighted_quantile(ss[sig], w[sig], 0.75),
        "fill_step_p25": weighted_quantile(fs[ex], w[ex], 0.25),
        "fill_step_median": weighted_quantile(fs[ex], w[ex], 0.50),
        "fill_step_p75": weighted_quantile(fs[ex], w[ex], 0.75),
    }
    # signal -> fill WALL-CLOCK elapsed time (executed signals only)
    sig_t = pd.to_datetime(q["tp_signal_time"])
    fil_t = pd.to_datetime(q["tp_fill_time"])
    wc_min = ((fil_t - sig_t).dt.total_seconds() / 60.0).to_numpy(float)
    out["signal_to_fill_wall_clock_minutes_p25"] = weighted_quantile(
        wc_min[ex], w[ex], 0.25)
    out["signal_to_fill_wall_clock_minutes_median"] = weighted_quantile(
        wc_min[ex], w[ex], 0.50)
    out["signal_to_fill_wall_clock_minutes_p75"] = weighted_quantile(
        wc_min[ex], w[ex], 0.75)
    # signal-bar close -> next-open execution gap, in ATR0 units (executed only)
    side_arr = np.where(q["direction"].to_numpy(object) == "LONG", 1.0, -1.0)
    touch_close_px = _touch_close_price(q)
    gap_atr = side_arr * (q["tp_fill_open"].to_numpy(float) - touch_close_px) \
        / q["ATR0"].to_numpy(float)
    out["signal_close_to_next_open_gap_atr_p25"] = weighted_quantile(
        gap_atr[ex], w[ex], 0.25)
    out["signal_close_to_next_open_gap_atr_median"] = weighted_quantile(
        gap_atr[ex], w[ex], 0.50)
    out["signal_close_to_next_open_gap_atr_p75"] = weighted_quantile(
        gap_atr[ex], w[ex], 0.75)

    out["r_touch_close_p25"] = weighted_quantile(r_close[touch_pos], w[touch_pos], 0.25)
    out["r_touch_close_median"] = weighted_quantile(r_close[touch_pos], w[touch_pos], 0.50)
    out["r_touch_close_p75"] = weighted_quantile(r_close[touch_pos], w[touch_pos], 0.75)
    out["mfe_touch_p25"] = weighted_quantile(mfe_t[touch_pos], w[touch_pos], 0.25)
    out["mfe_touch_median"] = weighted_quantile(mfe_t[touch_pos], w[touch_pos], 0.50)
    out["mfe_touch_p75"] = weighted_quantile(mfe_t[touch_pos], w[touch_pos], 0.75)

    out["tp_gross_mean_executed"] = weighted_mean(tp[ex], w[ex])
    out["baseline_gross_mean_executed"] = weighted_mean(base[ex], w[ex])
    out["paired_delta_mean_executed"] = weighted_mean(d[ex], w[ex])
    out["saved_giveback"] = weighted_mean(np.maximum(d, 0.0), w)
    out["lost_continuation"] = weighted_mean(np.maximum(-d, 0.0), w)
    out["frac_executed_delta_positive"] = weighted_mean((d[ex] > 0).astype(float), w[ex]) \
        if ex.any() else float("nan")
    out["frac_executed_delta_negative"] = weighted_mean((d[ex] < 0).astype(float), w[ex]) \
        if ex.any() else float("nan")

    okm = touch_pos & np.isfinite(mfe_t) & (mfe_t > 0)
    cap_tp = np.where(okm, tp / np.where(okm, mfe_t, 1.0), np.nan)
    cap_b = np.where(okm, base / np.where(okm, mfe_t, 1.0), np.nan)
    rem = np.where(touch_pos, mfe_fin - mfe_t, np.nan)
    give_b = np.where(touch_pos, mfe_t - base, np.nan)
    out["capture_tp_p25"] = weighted_quantile(cap_tp[okm], w[okm], 0.25)
    out["capture_tp_median"] = weighted_quantile(cap_tp[okm], w[okm], 0.50)
    out["capture_tp_p75"] = weighted_quantile(cap_tp[okm], w[okm], 0.75)
    out["capture_b_p25"] = weighted_quantile(cap_b[okm], w[okm], 0.25)
    out["capture_b_median"] = weighted_quantile(cap_b[okm], w[okm], 0.50)
    out["capture_b_p75"] = weighted_quantile(cap_b[okm], w[okm], 0.75)
    out["remaining_mfe_p25"] = weighted_quantile(rem[touch_pos], w[touch_pos], 0.25)
    out["remaining_mfe_median"] = weighted_quantile(rem[touch_pos], w[touch_pos], 0.50)
    out["remaining_mfe_p75"] = weighted_quantile(rem[touch_pos], w[touch_pos], 0.75)
    out["baseline_giveback_from_touch_peak_median"] = weighted_quantile(
        give_b[touch_pos], w[touch_pos], 0.50)

    stg = q["frozen_ahead_sr_strength"].to_numpy(float)
    out["frozen_ahead_sr_strength_p25"] = weighted_quantile(stg, w, 0.25)
    out["frozen_ahead_sr_strength_median"] = weighted_quantile(stg, w, 0.50)
    out["frozen_ahead_sr_strength_p75"] = weighted_quantile(stg, w, 0.75)

    # §19: Liquidity is SECONDARY DESCRIPTIVE ONLY, reported CONDITIONAL ON the
    # ahead-SR TP signal (never a TP confirmation).
    out["liq_*_denominator"] = "observed_ahead_sr_tp_signal"
    for col, name in (("ahead_liq_available", "ahead_liq_available"),
                      ("ahead_liq_touched_by_signal", "ahead_liq_touched"),
                      ("ahead_liq_crossed_by_signal", "ahead_liq_crossed")):
        v = q[col].to_numpy(float)
        out[f"{name}_given_signal"] = (
            weighted_mean(v[sig], w[sig]) if sig.any() else float("nan"))
        out[f"{name}_population_incidence"] = weighted_mean(v, w)
    return out


def _touch_close_price(q):
    """Reconstruct the touch-bar CLOSE price from the frozen identity.

    r_touch_close = sign * (close_touch - entry_price) / ATR0, so:
        close_touch = entry_price + r_touch_close * ATR0 / sign
    This avoids carrying a second price column for the same quantity.
    """
    side_arr = np.where(q["direction"].to_numpy(object) == "LONG", 1.0, -1.0)
    r_close = q["r_touch_close"].to_numpy(float)
    return (q["entry_price"].to_numpy(float)
            + r_close * q["ATR0"].to_numpy(float) / side_arr)


# --------------------------------------------------------------------------- #
# 10. Reference / Production differential (§32)                                 #
# --------------------------------------------------------------------------- #
# §32 comparison surface (names as returned by the kernels).
DIFF_DISCRETE = ("tp_eligible", "first_touch_step", "first_cross_step",
                 "accepted_same_bar", "signal_step", "signal_observed",
                 "executable", "fill_step", "fill_idx")
DIFF_NUMERIC = ("r_touch_close", "fill_open", "tp_return")


def differential_symbol(case):
    """Reference (bar-scan) vs Production (frozen-step) on the SAME case."""
    ref = scan_tp_reference(case)
    prod = scan_tp_production(case)
    n = int(len(case["entry_idx"]))
    out = {"n": n, "mismatch": 0, "max_abs_error": 0.0, "first_mismatch": None}

    def _note(key):
        out["mismatch"] += 1
        if out["first_mismatch"] is None:
            out["first_mismatch"] = key

    if not np.array_equal(ref["tp_eligible"], prod["tp_eligible"]):
        _note("tp_eligible")
    for f in ("first_touch_step", "first_cross_step", "accepted_same_bar"):
        if not np.array_equal(ref[f], prod[f]):
            _note(f)
    if not (np.array_equal(np.isfinite(ref["r_touch_close"]),
                           np.isfinite(prod["r_touch_close"]))):
        _note("r_touch_close.mask")
    else:
        m = np.isfinite(ref["r_touch_close"])
        if m.any():
            e = float(np.max(np.abs(ref["r_touch_close"][m]
                                    - prod["r_touch_close"][m])))
            out["max_abs_error"] = max(out["max_abs_error"], e)
            if e > 1e-12:
                _note("r_touch_close")
    if not np.array_equal(ref["signal_step"], prod["signal_step"]):
        _note("signal_step")

    for H in HORIZONS:
        rp = prod["per_h"][H]
        exit_step = rp["exit_step"]
        rr = resolve_tp_fill(
            signal_step=ref["signal_step"],
            entry_idx=np.asarray(case["entry_idx"], np.int64),
            side=np.asarray(case["side"], float),
            entry_price=np.asarray(case["entry_price"], float),
            atr0=np.asarray(case["atr0"], float),
            open_px=np.asarray(case["open"], float),
            segment=np.asarray(case["segment"], np.int64),
            entry_segment=np.asarray(case["entry_segment"], np.int64),
            exit_step_h=exit_step)
        for f in ("signal_observed", "executable", "fill_step", "fill_idx"):
            if not np.array_equal(rr[f], rp[f]):
                _note(f"{H}.{f}")
        for f in ("fill_open", "tp_return"):
            x, y = rr[f], rp[f]
            if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                _note(f"{H}.{f}.mask")
            elif np.isfinite(x).any():
                e = float(np.max(np.abs(x[np.isfinite(x)]
                                        - y[np.isfinite(y)])))
                out["max_abs_error"] = max(out["max_abs_error"], e)
                if e > 1e-12:
                    _note(f"{H}.{f}")
    return out


# --------------------------------------------------------------------------- #
# 11. Formal verdict (§22)                                                      #
# --------------------------------------------------------------------------- #
GROSS_VERDICTS = ("GROSS_TP_EDGE_SUPPORTED_UNIVERSAL", "GROSS_TP_HARMFUL",
                  "GROSS_SIDE_HETEROGENEITY_REQUIRES_FOLLOWUP",
                  "NO_IDENTIFIABLE_GROSS_TP_EDGE")


def formal_verdict(primary_row, side_rows):
    """Exactly four frozen categories. No post-hoc category may be created."""
    long_row = next(r for r in side_rows if r["side"] == "LONG")
    short_row = next(r for r in side_rows if r["side"] == "SHORT")
    long_neg = long_row["delta_ev_ci_high"] < 0
    short_neg = short_row["delta_ev_ci_high"] < 0
    long_pos = long_row["delta_ev_ci_low"] > 0
    short_pos = short_row["delta_ev_ci_low"] > 0
    if (primary_row["delta_ev_ci_low"] > 0
            and primary_row["delta_correct_gross_ci_low"] > 0
            and not long_neg and not short_neg):
        return "GROSS_TP_EDGE_SUPPORTED_UNIVERSAL"
    if primary_row["delta_ev_ci_high"] < 0:
        return "GROSS_TP_HARMFUL"
    if (long_pos and short_neg) or (short_pos and long_neg):
        return "GROSS_SIDE_HETEROGENEITY_REQUIRES_FOLLOWUP"
    return "NO_IDENTIFIABLE_GROSS_TP_EDGE"


# --------------------------------------------------------------------------- #
# 12. T1.5 integration driver (§34 / §35)                                       #
# --------------------------------------------------------------------------- #
T1_5_PER_SYMBOL = 50

# RC-R7-8 / RC-R7-12: the four horizon-causal first-encounter incidence series.
# Each consumes ONLY horizon-censored columns.
CAUSAL_SERIES = (
    ("first_touch", lambda q: q["first_ahead_sr_touch_step"].to_numpy(np.int64) >= 0),
    ("accepted_breakout", lambda q: q["first_touch_accepted_breakout"].to_numpy(bool)),
    ("touch_not_profitable", lambda q: _not_profitable_mask(q)),
    ("tp_signal", lambda q: q["tp_signal_observed"].to_numpy(bool)),
)


def _perf_counters(t_start):
    perf = dict(COUNTERS)
    perf["runtime_sec"] = time.time() - t_start
    return perf


def run_t1_5(verbose=True):
    t_start = time.time()
    reset_counters()
    frozen = verify_frozen_inputs()
    policy = verify_r6_downstream_policy()
    l2 = load_r5_l2()

    all_rows = []
    diffs = []
    per_symbol = []
    shared_frame = []
    for sym in SYMBOLS:
        sub_sym = l2[l2["symbol"] == sym]
        if len(sub_sym) == 0:
            continue
        keys = sorted(sub_sym["semantic_key"].unique())[:T1_5_PER_SYMBOL]
        sub = sub_sym[sub_sym["semantic_key"].isin(keys)]
        st = load_symbol_state(sym)          # ONE frame load per symbol
        _bump("execution_frame_loads")
        sym_diff = {}
        for ds in ("A9", "E9"):              # A9 and E9 SHARE that one frame
            rows, case = treatment_rows_for_symbol(sub, st, ds)
            if case is None or len(rows) == 0:
                continue
            all_rows.append(rows)
            sym_diff[ds] = differential_symbol(case)
        diffs.append({"symbol": sym, **sym_diff})
        per_symbol.append({"symbol": sym, "n_keys": int(len(keys)),
                           "n_A9": int((sub.direction_system == "A9").sum()),
                           "n_E9": int((sub.direction_system == "E9").sum())})
        shared_frame.append({"symbol": sym,
                             "execution_frame_loads": int(
                                 COUNTERS["execution_frame_loads"]),
                             "systems_sharing_frame": sorted(sym_diff.keys())})
    big = pd.concat(all_rows, ignore_index=True)

    # RC-R7-8: horizon-causal incidence monotonicity. Every first-encounter
    # series must satisfy TD1 <= TD3 <= TD5 per symbol x system. With the
    # horizon censor in place, a touch occurring after Baseline_H contributes
    # ZERO to the H-specific incidence.
    causality = []
    caus_ok = True
    for sym in sorted(big.symbol.unique()):
        for ds in ("A9", "E9"):
            rec = {"symbol": sym, "system": ds}
            for name, fn in CAUSAL_SERIES:
                c = {}
                for H in HORIZONS:
                    q = big[(big.symbol == sym) & (big.direction_system == ds)
                            & (big.evaluation_horizon == H)]
                    c[H] = int(np.asarray(fn(q)).sum()) if len(q) else 0
                ok = bool(c["td1"] <= c["td3"] <= c["td5"])
                caus_ok = caus_ok and ok
                rec[name] = c
                rec[f"{name}_ok"] = ok
            causality.append(rec)

    primary = []
    for ds in ("A9", "E9"):
        for H in HORIZONS:
            e = _effect(big, ds, H)
            if e:
                e["is_primary"] = bool(ds == "E9" and H == PRIMARY_HORIZON)
                e["is_robustness"] = bool(ds == "E9" and H in ROBUSTNESS_HORIZONS)
                primary.append(e)
    primary_df = pd.DataFrame(primary)

    side_rows = []
    for ds in ("A9", "E9"):
        for H in HORIZONS:
            for sd in ("LONG", "SHORT"):
                e = _effect(big, ds, H, side=sd)
                if e:
                    side_rows.append(e)
    side_df = pd.DataFrame(side_rows)

    group_df = pd.DataFrame(group_stat_rows(big), columns=GROUP_STAT_COLUMNS)
    events_df = events_frame(big)

    max_abs = 0.0
    for d in diffs:
        for ds in ("A9", "E9"):
            if ds in d:
                max_abs = max(max_abs, d[ds]["max_abs_error"])
    n_mismatch = sum(d[ds]["mismatch"] for d in diffs for ds in ("A9", "E9")
                     if ds in d)

    write_csv(T1_5_PRIMARY_CSV, primary_df, list(primary_df.columns))
    write_csv(T1_5_SIDE_STATS_CSV, side_df, list(side_df.columns))
    write_csv(T1_5_GROUP_STATS_CSV, group_df, list(GROUP_STAT_COLUMNS))
    write_csv(T1_5_EVENTS_CSV, events_df, EVENT_COLUMNS)

    perf = _perf_counters(t_start)
    summary = {
        "task_id": TASK_ID, "stage": STAGE_T1_5,
        "scientific_status": "NON_SCIENTIFIC_INTEGRATION_ONLY",
        "n_symbols": len(per_symbol),
        "n_base_semantic_keys": int(sum(p["n_keys"] for p in per_symbol)),
        "n_direction_system_views": int(sum(
            p["n_A9"] + p["n_E9"] for p in per_symbol)),
        "n_horizon_rows": int(len(big)),
        "per_symbol": per_symbol,
        "downstream_policy": policy,
        "differential": {"n_symbols_with_reference": len(diffs),
                         "n_mismatch": int(n_mismatch),
                         "max_abs_error": float(max_abs)},
        "per_symbol_differential": diffs,
        "horizon_causality": {"ok": bool(caus_ok), "per_symbol_system": causality},
        "shared_frame": shared_frame,
        "primary_table": primary,
        "performance": perf,
        "cost_governance": {
            "cost_metadata_status": COST_METADATA_STATUS,
            "realistic_net_pnl_status": REALISTIC_NET_PNL_STATUS,
            "formal_primary_basis": FORMAL_PRIMARY_BASIS,
            "friction_proxy_used_for_verdict": FRICTION_PROXY_USED_FOR_VERDICT,
            "entry_cost_estimated": ENTRY_COST_ESTIMATED,
            "exit_cost_estimated": EXIT_COST_ESTIMATED,
        },
        "unverified_items": [
            "Full 13773-Candidate Formal R7 NOT run (not authorized).",
            "No NET verdict; GROSS paired policy effect only.",
            "Friction proxy not used for any verdict.",
            "T1.5 numbers are integration validation only, not a research result.",
            "T1.5 per-symbol Delta must NOT be interpreted economically.",
        ],
    }
    write_json(T1_5_SUMMARY_JSON, summary)

    artifact_shas = {os.path.basename(p): sha256_file(p)
                     for p in (T1_5_PRIMARY_CSV, T1_5_SIDE_STATS_CSV,
                               T1_5_GROUP_STATS_CSV, T1_5_EVENTS_CSV,
                               T1_5_SUMMARY_JSON)}
    manifest = {
        "task_id": TASK_ID, "stage": STAGE_T1_5,
        "base_sha": BASE_SHA, "reviewed_parent_sha": REVIEWED_PARENT,
        "generator_code_sha": _git_head_sha(),
        "frozen_inputs_sha256": frozen,
        "downstream_policy": policy,
        "cost_governance": summary["cost_governance"],
        "differential": summary["differential"],
        "horizon_causality": {"ok": bool(caus_ok)},
        "performance": perf,
        "artifact_sha256": artifact_shas,
        "serialization_manifest_last": True,
        "authorized_review_sha_note":
            "R7 PRE-T2 authorized under FUTURE-R6/R7 Amendment A1 "
            "(GROSS primary, NO_STRUCTURAL_STOP downstream).",
    }
    write_json(T1_5_MANIFEST_JSON, manifest)
    if verbose:
        print(json.dumps({"differential": summary["differential"],
                          "population": {"horseizon_rows": int(len(big)),
                                         "views": summary["n_direction_system_views"]},
                          "performance": perf}, indent=2))
    return summary


# --------------------------------------------------------------------------- #
# 13. Formal gates (§37 / §38) — IMPLEMENTED, NOT RUN                           #
# --------------------------------------------------------------------------- #
FORMAL_ARTIFACT = os.path.join(ARTIFACT_DIR,
                               "ahead_sr_take_profit_row_metrics_v1.parquet")
FORMAL_MANIFEST = MANIFEST_JSON
FORMAL_SUMMARY = SUMMARY_JSON

FROZEN_FORMAL = {"symbols": 15, "semantic_keys": 13773, "gids": 638,
                 "long_gids": 319, "short_gids": 319,
                 "base_rows": 27546, "base_per_system": 13773,
                 "horizon_rows": 27546, "rows_per_system": 41319,
                 "total_rows": 82638}

PERF_EXPECTED = {"r5_l2_loads": 1, "execution_frame_loads": 15,
                 "direction_reruns": 0, "sr_recompute_count": 0,
                 "path_atlas_rescans": 0, "full_history_recompute_count": 0,
                 "reference_calls": 0, "r6_policy_verifications": 1,
                 "production_candidate_views": FROZEN_FORMAL["base_rows"]}


def _per_system_gid_weight_gate(l2, atol=1e-9):
    """§37: A9 and E9 each carry the frozen Candidate weight independently.

    Per SYSTEM every gid weight must sum to 1. The merged A9+E9 gid weight
    legitimately sums to ~2 and is NEVER gated.
    """
    rep = {}
    for ds, key in (("A9", "a9"), ("E9", "e9")):
        g = l2[l2.direction_system == ds]
        gw = g.groupby("gid")["sample_weight_raw"].sum().to_numpy(float)
        rep[f"{key}_gid_weight_ok"] = bool(
            len(gw) == FROZEN_FORMAL["gids"] and np.allclose(gw, 1.0, atol=atol))
        rep[f"{key}_n_gids"] = int(len(gw))
        rep[f"{key}_gid_weight_max_abs_dev"] = (
            float(np.max(np.abs(gw - 1.0))) if len(gw) else float("inf"))
    merged = l2.groupby("gid")["sample_weight_raw"].sum().to_numpy(float)
    rep["merged_gid_weight_sum_mean"] = (
        float(np.mean(merged)) if len(merged) else float("nan"))
    rep["merged_gid_weight_is_not_gated"] = True
    return rep


def _per_system_gid_weight_mismatches(l2, atol=1e-9):
    rep = _per_system_gid_weight_gate(l2, atol=atol)
    return {f"{k}_gid_weight": rep[f"{k}_gid_weight_max_abs_dev"]
            for k in ("a9", "e9") if not rep[f"{k}_gid_weight_ok"]}


def _base_l2_population_gates(l2):
    mism = {}
    syms = sorted(str(s) for s in l2.symbol.unique())
    if syms != sorted(SYMBOLS):
        mism["symbols"] = syms
    if int(len(l2)) != FROZEN_FORMAL["base_rows"]:
        mism["base_l2_rows"] = int(len(l2))
    nk = int(l2.semantic_key.nunique())
    if nk != FROZEN_FORMAL["semantic_keys"]:
        mism["semantic_keys"] = nk
    bad_sys = sorted(set(str(s) for s in l2.direction_system.unique())
                     - {"A9", "E9"})
    if bad_sys:
        mism["direction_system_values"] = bad_sys
    for ds in ("A9", "E9"):
        n = int((l2.direction_system == ds).sum())
        if n != FROZEN_FORMAL["base_per_system"]:
            mism[f"base_{ds}_rows"] = n
    npairs = int(l2[["semantic_key", "direction_system"]].drop_duplicates().shape[0])
    if npairs != FROZEN_FORMAL["base_rows"]:
        mism["semantic_key_system_pairs"] = npairs
    per = l2.groupby(["semantic_key", "direction_system"]).size()
    if not bool((per == 1).all()):
        mism["semantic_key_system_duplicate_rows"] = int((per != 1).sum())
    nsys = l2.groupby("semantic_key")["direction_system"].nunique()
    if not bool((nsys == 2).all()):
        mism["semantic_key_incomplete_system_pair"] = int((nsys != 2).sum())
    ndir = l2.groupby("gid")["oracle_direction"].nunique()
    if not bool((ndir == 1).all()):
        mism["gid_oracle_direction_multivalued"] = int((ndir != 1).sum())
    gid_dir = l2.drop_duplicates("gid").set_index("gid")["oracle_direction"]
    if int(len(gid_dir)) != FROZEN_FORMAL["gids"]:
        mism["gids"] = int(len(gid_dir))
    bad_dir = sorted(set(str(d) for d in gid_dir.unique()) - {"LONG", "SHORT"})
    if bad_dir:
        mism["oracle_direction_values"] = bad_dir
    long_g = set(gid_dir[gid_dir == "LONG"].index.tolist())
    short_g = set(gid_dir[gid_dir == "SHORT"].index.tolist())
    if len(long_g) != FROZEN_FORMAL["long_gids"]:
        mism["long_gids"] = len(long_g)
    if len(short_g) != FROZEN_FORMAL["short_gids"]:
        mism["short_gids"] = len(short_g)
    if long_g & short_g:
        mism["gid_direction_overlap_n"] = len(long_g & short_g)
    all_g = set(gid_dir.index.tolist())
    if (long_g | short_g) != all_g:
        mism["gid_direction_union_missing_n"] = len(all_g - (long_g | short_g))
    return mism


def _formal_population_gates(l2, big=None):
    """BASE frozen-input gate (§37): R5 L2 identity + per-system gid weights.

    RC-R7-10: this is the BASE gate only. The generated 82638-row R7 treatment
    frame is verified separately by `_generated_population_gate`.
    """
    mism = {}
    mism.update(_base_l2_population_gates(l2))
    mism.update(_per_system_gid_weight_mismatches(l2))
    return mism


def _generated_population_gate(big):
    """RC-R7-9: pre-write gate on the IN-MEMORY generated Formal R7 frame.

    The canonical parquet writer must NOT be called when this fails.
    """
    mism = {}
    n = int(len(big))
    if n != FROZEN_FORMAL["total_rows"]:
        mism["rows"] = n
    nk = int(big.semantic_key.nunique())
    if nk != FROZEN_FORMAL["semantic_keys"]:
        mism["semantic_keys"] = nk
    bad_sys = sorted(set(str(s) for s in big.direction_system.unique())
                     - {"A9", "E9"})
    if bad_sys:
        mism["direction_system_values"] = bad_sys
    bad_h = sorted(set(str(h) for h in big.evaluation_horizon.unique())
                   - set(HORIZONS))
    if bad_h:
        mism["evaluation_horizon_values"] = bad_h
    for ds in ("A9", "E9"):
        c = int((big.direction_system == ds).sum())
        if c != FROZEN_FORMAL["rows_per_system"]:
            mism[f"{ds}_rows"] = c
    for H in HORIZONS:
        c = int((big.evaluation_horizon == H).sum())
        if c != FROZEN_FORMAL["horizon_rows"]:
            mism[f"{H}_rows"] = c
    keycols = ["semantic_key", "direction_system", "evaluation_horizon"]
    uq = int(big[keycols].drop_duplicates().shape[0])
    if uq != FROZEN_FORMAL["total_rows"]:
        mism["unique_keys"] = uq
    if n - uq != 0:
        mism["duplicate_keys"] = int(n - uq)
    combo = (big["semantic_key"].astype(str) + "|"
             + big["direction_system"].astype(str) + "|"
             + big["evaluation_horizon"].astype(str))
    if int(combo.nunique()) != FROZEN_FORMAL["total_rows"]:
        mism["unique_system_horizon_keys"] = int(combo.nunique())
    per = big.groupby("semantic_key").size()
    if not bool((per == 6).all()):
        mism["rows_per_semantic_key"] = int((per != 6).sum())
    return mism


def _write_formal_artifact(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _verify_formal_artifact(path, frozen_keys):
    """§38: post-write key verification (key columns only)."""
    t = pd.read_parquet(path, columns=["semantic_key", "direction_system",
                                       "evaluation_horizon"])
    n = int(len(t))
    uniq = int(t.drop_duplicates().shape[0])
    combo = (t["semantic_key"].astype(str) + "|" + t["direction_system"].astype(str)
             + "|" + t["evaluation_horizon"].astype(str))
    rep = {
        "rows": n, "unique_keys": uniq, "duplicate_keys": int(n - uniq),
        "a9_rows": int((t.direction_system == "A9").sum()),
        "e9_rows": int((t.direction_system == "E9").sum()),
        "invalid_system_rows": int((~t.direction_system.isin(["A9", "E9"])).sum()),
        "invalid_horizon_rows": int((~t.evaluation_horizon.isin(list(HORIZONS))).sum()),
        "unknown_semantic_key_rows": int((~t.semantic_key.astype(str).isin(frozen_keys)).sum()),
        "unique_system_horizon_keys": int(combo.nunique()),
        "rows_per_semantic_key_max": int(t.groupby("semantic_key").size().max())
        if n else 0,
        "rows_per_semantic_key_min": int(t.groupby("semantic_key").size().min())
        if n else 0,
    }
    rep["six_combos_exactly_once"] = bool(
        rep["unique_system_horizon_keys"] == FROZEN_FORMAL["total_rows"])
    rep["pass"] = bool(
        rep["rows"] == FROZEN_FORMAL["total_rows"]
        and rep["unique_keys"] == FROZEN_FORMAL["total_rows"]
        and rep["duplicate_keys"] == 0
        and rep["a9_rows"] == FROZEN_FORMAL["rows_per_system"]
        and rep["e9_rows"] == FROZEN_FORMAL["rows_per_system"]
        and rep["invalid_system_rows"] == 0 and rep["invalid_horizon_rows"] == 0
        and rep["unknown_semantic_key_rows"] == 0
        and rep["rows_per_semantic_key_max"] == 6
        and rep["rows_per_semantic_key_min"] == 6
        and rep["six_combos_exactly_once"])
    return rep


PROVENANCE_FIELDS = ("environment_contract_id", "cache_schema_version", "identity",
                     "code_identity", "raw_sha256", "execution_frame_sha256",
                     "sha256", "rows", "max_bars")


def _env_provenance_records(states):
    recs = []
    for st in states:
        p = getattr(st, "env_provenance", None)
        p = p if isinstance(p, dict) else {}
        rec = {"symbol": getattr(st, "symbol", p.get("symbol"))}
        rec.update({k: p.get(k) for k in PROVENANCE_FIELDS})
        recs.append(rec)
    return recs


def _env_provenance_gate(recs):
    mism = {}
    if len(recs) != FROZEN_FORMAL["symbols"]:
        mism["n_symbols"] = len(recs)
    syms = sorted(str(r.get("symbol")) for r in recs)
    if syms != sorted(SYMBOLS):
        mism["symbols"] = syms
    for r in recs:
        sym = r.get("symbol")
        if r.get("environment_contract_id") != ENV_CONTRACT_ID:
            mism.setdefault("environment_contract_id", []).append(
                (sym, r.get("environment_contract_id")))
        if r.get("max_bars") is not None:
            mism.setdefault("max_bars", []).append((sym, r.get("max_bars")))
        for k in PROVENANCE_FIELDS:
            if k != "max_bars" and r.get(k) is None:
                mism.setdefault(f"missing_{k}", []).append(sym)
    return mism


def run_formal_r7(*, allow_full=False, authorized_review_sha=None,
                  write_artifacts=True, verbose=False):
    """§37/§38: frozen Formal R7 path behind an explicit authorization gate.

    Frozen ordering (RC-R7-11):
      1. authorized SHA
      2. frozen artifact SHA
      3. R6 downstream-policy verification
      4. R5 base L2 gate                      <-- base_population_gates
      5. compute all R7 production rows
      6. generated-population gate            <-- generated_population_gates
      7. environment-provenance gate
      8. performance gate
      9. canonical parquet write              <-- only reachable if 4/6/7/8 PASS
     10. actual parquet key verification
     11. Formal statistics / evidence
     12. artifact SHA256
     13. Formal manifest LAST
    """
    if allow_full is not True:
        raise RuntimeError("STOP_R7_FULL_POPULATION_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R7_AUTHORIZED_REVIEW_SHA_REQUIRED")
    if write_artifacts is not True:
        raise RuntimeError("STOP_R7_FORMAL_ARTIFACT_WRITE_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R7_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")
    t_start = time.time()
    reset_counters()
    frozen = verify_frozen_inputs()            # 2
    policy = verify_r6_downstream_policy()     # 3
    l2 = load_r5_l2()

    # 4: BASE gate on the frozen R5 L2, before ANY R7 row is generated.
    base_mismatch = _formal_population_gates(l2)
    gid_weight_report = _per_system_gid_weight_gate(l2)
    if base_mismatch:
        raise RuntimeError(
            f"STOP_R7_FORMAL_BASE_POPULATION_GATE {base_mismatch}")

    # 5: compute all R7 production rows.
    all_rows = []
    states = []
    for sym in SYMBOLS:
        sub = l2[l2.symbol == sym]
        if len(sub) == 0:
            continue
        st = load_symbol_state(sym)
        states.append(st)
        _bump("execution_frame_loads")
        for ds in ("A9", "E9"):         # A9 and E9 share ONE frame per symbol
            rows, _ = treatment_rows_for_symbol(sub, st, ds)
            all_rows.append(rows)
    big = pd.concat(all_rows, ignore_index=True)
    frozen_keys = set(l2.semantic_key.astype(str).unique())

    # 6: generated-population gate (in-memory, PRE-WRITE).
    gen_mismatch = _generated_population_gate(big)
    # 7: environment provenance gate.
    env_records = _env_provenance_records(states)
    env_mismatch = _env_provenance_gate(env_records)
    # 8: performance gate.
    perf = _perf_counters(t_start)
    perf_mismatch = {k: (perf[k], v) for k, v in PERF_EXPECTED.items()
                     if perf.get(k) != v}
    if gen_mismatch:
        raise RuntimeError(
            f"STOP_R7_FORMAL_GENERATED_POPULATION_GATE {gen_mismatch}")
    if env_mismatch:
        raise RuntimeError(
            f"STOP_R7_FORMAL_ENVIRONMENT_PROVENANCE_GATE {env_mismatch}")
    if perf_mismatch:
        raise RuntimeError(f"STOP_R7_FORMAL_PERFORMANCE_GATE {perf_mismatch}")

    base_gates = {"pass": not base_mismatch, "mismatch": base_mismatch,
                  "gid_weight": gid_weight_report}
    gen_gates = {"pass": not gen_mismatch, "mismatch": gen_mismatch,
                 "expected": {"rows": FROZEN_FORMAL["total_rows"],
                              "semantic_keys": FROZEN_FORMAL["semantic_keys"],
                              "a9_rows": FROZEN_FORMAL["rows_per_system"],
                              "e9_rows": FROZEN_FORMAL["rows_per_system"],
                              "horizon_rows": FROZEN_FORMAL["horizon_rows"]}}

    _write_formal_artifact(big, FORMAL_ARTIFACT)
    artifact_report = _verify_formal_artifact(FORMAL_ARTIFACT, frozen_keys)
    if not artifact_report.get("pass"):
        raise RuntimeError(
            f"STOP_R7_FORMAL_ARTIFACT_KEY_MISMATCH {artifact_report}")

    primary_row = _effect(big, "E9", PRIMARY_HORIZON)
    side_rows = [_effect(big, "E9", PRIMARY_HORIZON, side=s)
                 for s in ("LONG", "SHORT")]
    verdict = formal_verdict(primary_row, side_rows)
    diag = formal_tp_diagnostics(big, "E9", PRIMARY_HORIZON)
    primary_df = pd.DataFrame(
        [e for ds in ("A9", "E9") for H in HORIZONS
         if (e := _effect(big, ds, H)) is not None])
    side_df = pd.DataFrame(
        [e for ds in ("A9", "E9") for H in HORIZONS for sd in ("LONG", "SHORT")
         if (e := _effect(big, ds, H, side=sd)) is not None])
    group_df = pd.DataFrame(group_stat_rows(big), columns=GROUP_STAT_COLUMNS)
    events_df = events_frame(big)

    write_csv(PRIMARY_CSV, primary_df, list(primary_df.columns))
    write_csv(SIDE_STATS_CSV, side_df, list(side_df.columns))
    write_csv(GROUP_STATS_CSV, group_df, list(GROUP_STAT_COLUMNS))
    write_csv(EVENTS_CSV, events_df, EVENT_COLUMNS)

    summary = {
        "task_id": TASK_ID, "stage": STAGE_FORMAL,
        "authorized_review_sha": authorized_review_sha,
        "generator_code_sha": head, "reviewed_parent_sha": head,
        "population": "full", "n_horizon_rows": int(len(big)),
        "formal_verdict": verdict,
        "primary_row": primary_row, "side_rows": side_rows,
        "tp_diagnostics": diag,
        "downstream_policy": policy,
        # RC-R7-10: base and generated gates are persisted SEPARATELY.
        "base_population_gates": base_gates,
        "generated_population_gates": gen_gates,
        "performance_gates": {"pass": not perf_mismatch,
                              "mismatch": perf_mismatch,
                              "expected": PERF_EXPECTED},
        "artifact_integrity_gates": artifact_report,
        "environment_provenance_gates": {"pass": not env_mismatch,
                                         "mismatch": env_mismatch},
        "performance": perf,
        "cost_governance": {
            "cost_metadata_status": COST_METADATA_STATUS,
            "realistic_net_pnl_status": REALISTIC_NET_PNL_STATUS,
            "formal_primary_basis": FORMAL_PRIMARY_BASIS,
            "friction_proxy_used_for_verdict": FRICTION_PROXY_USED_FOR_VERDICT,
            "entry_cost_estimated": ENTRY_COST_ESTIMATED,
            "exit_cost_estimated": EXIT_COST_ESTIMATED,
        },
        "unverified_items": [
            "Real transaction cost / slippage metadata still unavailable.",
            "Verdict is GROSS policy effect, NOT a NET-profitability claim.",
            "A9-vs-E9 system comparison reserved for a later frozen ablation.",
        ],
    }
    write_json(FORMAL_SUMMARY, summary)

    formal_artifacts = [FORMAL_ARTIFACT, PRIMARY_CSV, SIDE_STATS_CSV,
                        GROUP_STATS_CSV, EVENTS_CSV, FORMAL_SUMMARY]
    art_shas = {os.path.basename(p): sha256_file(p) for p in formal_artifacts}
    manifest = {
        "task_id": TASK_ID, "stage": STAGE_FORMAL,
        "base_sha": BASE_SHA,
        "authorized_review_sha": authorized_review_sha,
        "generator_code_sha": head, "reviewed_parent_sha": head,
        "population": "full", "n_horizon_rows": int(len(big)),
        "frozen_inputs_sha256": frozen,
        "base_population_gates": base_gates,
        "generated_population_gates": gen_gates,
        "performance_gates": summary["performance_gates"],
        "artifact_integrity_gates": artifact_report,
        "environment_provenance_gates": summary["environment_provenance_gates"],
        "environment_provenance": env_records,
        "downstream_policy": policy,
        "formal_verdict": verdict,
        "cost_governance": summary["cost_governance"],
        "artifact_sha256": art_shas,
        "serialization_manifest_last": True,
        # RC-R7-10: Formal acceptance requires BOTH population gates.
        "all_pass": bool(base_gates["pass"] and gen_gates["pass"]
                         and not perf_mismatch and not env_mismatch
                         and artifact_report.get("pass")),
    }
    write_json(FORMAL_MANIFEST, manifest)   # written LAST
    if verbose:
        print(json.dumps({"verdict": verdict, "artifact": artifact_report,
                          "perf": perf}, indent=2))
    return summary


def _git_head_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


if __name__ == "__main__":
    run_t1_5()
