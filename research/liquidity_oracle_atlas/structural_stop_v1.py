"""FUTURE-R6-M15-STRUCTURAL-STOP-V1 (Amendment A1: GROSS-primary).

Tests whether a causal, observable SR structural-invalidation stop improves the
PAIRED GROSS price outcome vs carrying the same trade to a frozen TD horizon.

Governance (Amendment A1):
  * No canonical real transaction-cost / slippage owner exists in this repo
    (canonical_table_found=False, REALISTIC_NET_PNL=UNAVAILABLE_COST_METADATA).
  * Primary quantity is therefore GROSS paired policy effect, NOT a NET verdict.
  * `kappa*ATR` is a FRICTION PROXY only and never determines the verdict.

Rule (frozen):
  LONG : B = frozen raw_sup_bottom ; invalidation = first completed 15m bar t with
         low_t < B AND close_t < B   (== pierced and NOT same-bar reclaimed)
  SHORT: B = frozen raw_res_top   ; invalidation = first completed 15m bar t with
         high_t > B AND close_t > B
  Fill : OPEN of the NEXT valid canonical 15m execution bar, ONLY IF that bar is in
         the SAME hard segment as the entry bar AND occurs before the baseline exit.
        The execution-frame `segment` owner is authoritative; R6 never redefines
        session boundaries and never requires same-trading-day.
"""

import json
import os
import hashlib
import time

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.entry_path_atlas_v1 import (
    SYMBOLS,
    TD_CHECKPOINTS,
    load_symbol_state,
    build_horizon_indices,
)

# --------------------------------------------------------------------------- #
# 0. Constants / governance                                                    #
# --------------------------------------------------------------------------- #
TASK_ID = "FUTURE-R6-M15-STRUCTURAL-STOP-V1"
# FG-R6-13: keep BOTH lineage levels explicit. BASE_SHA is the original R6 base;
# REVIEWED_PARENT is the immediate reviewed R6 parent of this PRE-T2 evidence.
BASE_SHA = "c0126551040f3e758ba5352ecb693bcbff0ad8b0"
REVIEWED_PARENT = "9700d0c629274000f67ce3ecbacca753f668f2a8"

ARTIFACT_DIR = os.path.join("artifacts", "structural_stop_v1")
EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")

# FG-R6-5: stage identity must be UNAMBIGUOUS. The PRE-T2 (T1.5) integration stage
# owns stage-explicit `*_t1_5_*` names; the canonical (un-prefixed) names below
# belong to the Formal FULL-population stage only.
T1_5_PRIMARY_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_t1_5_primary.csv")
T1_5_SIDE_STATS_CSV = os.path.join(EVIDENCE_DIR,
                                   "structural_stop_v1_t1_5_side_stats.csv")
T1_5_GROUP_STATS_CSV = os.path.join(EVIDENCE_DIR,
                                    "structural_stop_v1_t1_5_group_stats.csv")
T1_5_STOP_EVENTS_CSV = os.path.join(EVIDENCE_DIR,
                                    "structural_stop_v1_t1_5_stop_events.csv")
T1_5_SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "structural_stop_v1_t1_5_summary.json")
T1_5_MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "structural_stop_v1_t1_5_manifest.json")

# Canonical Formal evidence (FULL 13773-Candidate population ONLY).
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "structural_stop_v1_manifest.json")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "structural_stop_v1_summary.json")
PRIMARY_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_primary.csv")
GROUP_STATS_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_group_stats.csv")
STOP_EVENTS_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_stop_events.csv")
SIDE_STATS_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_side_stats.csv")

STAGE_T1_5 = "t1_5_integration"
STAGE_FORMAL = "formal_r6"
ENV_CONTRACT_ID = "FUTURE-R4-M15-ENVIRONMENT-V1"

FROZEN_INPUTS = {
    os.path.join("artifacts", "entry_path_atlas_v1", "entry_path_row_metrics_v1.parquet"):
        "110d5990fd9b3bc17aca65a92eb172e0de088635e89c364428b8c9bfccf94d04",
    os.path.join("artifacts", "entry_path_atlas_v1", "entry_path_curve_v1.parquet"):
        "fd8e8d352bc519baae0ee94088ab03ee76a942e1d0746a74bee005bb26f8dff0",
    os.path.join("artifacts", "entry_path_atlas_v1", "e9_direction_state_v1.parquet"):
        "1778eeeaf97c0f6b9496fa785d730bc7c23f2e910ddbdc5cfb4772b121e5f2fd",
}
L2_PARQUET = os.path.join("artifacts", "entry_path_atlas_v1",
                          "entry_path_row_metrics_v1.parquet")

HORIZONS = ("td1", "td3", "td5")
PRIMARY_HORIZON = "td5"
ROBUSTNESS_HORIZONS = ("td1", "td3")
BOOTSTRAP_SEED = 20260924
BOOTSTRAP_B = 2000

# cost governance (Amendment A1)
COST_METADATA_STATUS = "UNAVAILABLE_COST_METADATA"
REALISTIC_NET_PNL_STATUS = "NOT_ESTIMATED"
FORMAL_PRIMARY_BASIS = "GROSS_PAIRED_POLICY_EFFECT"
FRICTION_PROXY_USED_FOR_VERDICT = False
ENTRY_COST_ESTIMATED = False
EXIT_COST_ESTIMATED = False

COUNTERS = {"r5_artifact_loads": 0, "execution_frame_loads": 0,
            "direction_artifact_sha_verifications": 0, "sr_recompute_count": 0,
            "full_history_recompute_count": 0, "reference_calls": 0,
            "production_candidate_views": 0}


def _bump(name, n=1):
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def reset_counters():
    for k in COUNTERS:
        COUNTERS[k] = 0


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_frozen_inputs():
    """Fail closed if any required frozen R5 artifact is missing or SHA-mismatched."""
    out = {}
    for path, want in FROZEN_INPUTS.items():
        if not os.path.exists(path):
            raise RuntimeError(
                f"STOP_R6_FROZEN_INPUT_ARTIFACT_MISMATCH missing={path}")
        got = sha256_file(path)
        if got != want:
            raise RuntimeError(
                f"STOP_R6_FROZEN_INPUT_ARTIFACT_MISMATCH path={path} "
                f"got={got} want={want}")
        out[path] = got
    return out


# --------------------------------------------------------------------------- #
# 1. Frozen R5 L2 loader                                                        #
# --------------------------------------------------------------------------- #
L2_COLUMNS = [
    "semantic_key", "symbol", "gid", "direction_system", "direction",
    "sample_weight_raw", "direction_correct", "oracle_direction",
    "entry_price", "ATR0", "fill_time",
    "raw_sup_bottom", "raw_sup_top", "raw_sup_strength",
    "raw_res_bottom", "raw_res_top", "raw_res_strength",
    "raw_liq_dn_bottom", "raw_liq_up_top",
    "sr_first_pierce", "sr_same_bar_reclaim", "sr_first_failed_reclaim",
    "lb_first_touch", "lb_first_pierce", "lb_first_reclaim",
    "td1_r", "td3_r", "td5_r",
]


def load_r5_l2():
    _bump("r5_artifact_loads")
    df = pd.read_parquet(L2_PARQUET, columns=L2_COLUMNS)
    return df


# --------------------------------------------------------------------------- #
# 2. Invalidation step (frozen rule)                                            #
# --------------------------------------------------------------------------- #
def production_invalidation_step(pi, same_bar, failed_reclaim):
    """Production derivation from the frozen R5 first-passage event steps (O(N)).

    Equivalent to the literal bar rule:  first t with (low<B and close<B).
      * first pierce p; if same-bar reclaimed -> first later close beyond (= FFR)
        else -> p.
    """
    pi = np.asarray(pi, dtype=np.int64)
    sbr = np.asarray(same_bar, dtype=bool)
    ffr = np.asarray(failed_reclaim, dtype=np.int64)
    return np.where(pi < 0, -1, np.where(sbr, ffr, pi))


def reference_invalidation_step(side, lo, hi, cl, entry_idx, end_idx, B):
    """Reference (bar-scan) derivation for ONE candidate; T0/T1 + differential."""
    if not np.isfinite(B):
        return -1
    for t in range(0, int(end_idx - entry_idx) + 1):
        j = int(entry_idx + t)
        if side > 0:
            if lo[j] < B and cl[j] < B:
                return t
        else:
            if hi[j] > B and cl[j] > B:
                return t
    return -1


# --------------------------------------------------------------------------- #
# 3. Fill resolution (shared)                                                   #
# --------------------------------------------------------------------------- #
def boundary_of(direction_long, raw_sup_bottom, raw_res_top):
    return np.where(direction_long, raw_sup_bottom, raw_res_top)


def zone_edges(direction_long, raw_sup_top, raw_sup_bottom,
               raw_res_top, raw_res_bottom):
    top = np.where(direction_long, raw_sup_top, raw_res_top)
    bot = np.where(direction_long, raw_sup_bottom, raw_res_bottom)
    return top, bot


def resolve_treatment(*, inv_step, side, entry_idx, entry_price, atr0,
                      open_px, segment, n_bars, exit_step_H, eligible):
    """Vectorized stop fill resolution for ONE horizon H.

    Returns dict with stop_signal_observed/step, stop_executable, stop_fill_step,
    stop_reason, and the resulting stop gross return is derived by caller.
    """
    N = len(inv_step)
    inv_step = np.asarray(inv_step, np.int64)
    exit_step_H = np.asarray(exit_step_H, np.int64)
    # RC-R6-1: horizon-causal. A signal is OBSERVED for horizon H only if the
    # invalidation occurs at/before that horizon's baseline exit bar. An
    # invalidation after the exit must never appear in the H-specific row.
    signal_observed = eligible & (inv_step >= 0) & (inv_step <= exit_step_H)
    signal_idx = np.where(signal_observed, entry_idx + inv_step, -1)
    next_idx = np.where(signal_observed, signal_idx + 1, -1)
    in_range = signal_observed & (next_idx >= 0) & (next_idx < n_bars)
    safe_next = np.where(in_range, next_idx, 0)
    same_seg = in_range & (segment[safe_next] == segment[entry_idx])
    before_exit = same_seg & (inv_step + 1 <= exit_step_H)
    executable = before_exit  # same hard segment + before baseline exit

    reason = np.full(N, "", dtype=object)
    reason[~eligible] = "not_eligible"
    reason[eligible & (inv_step < 0)] = "no_signal"
    reason[eligible & (inv_step >= 0) & (inv_step > exit_step_H)] = \
        "no_signal_before_baseline_exit"
    reason[signal_observed & ~in_range] = "no_next_bar"
    reason[in_range & ~same_seg] = "segment_change"
    reason[same_seg & ~before_exit] = "beyond_baseline_exit"
    reason[executable] = "structural_invalidation"

    fill_step = np.where(executable, inv_step + 1, -1)
    fill_idx = np.where(executable, next_idx, -1)
    safe_fill = np.where(executable, next_idx, 0)
    fill_open = np.where(executable, open_px[safe_fill], np.nan)
    gross = np.where(executable,
                     side * (fill_open - entry_price) / atr0, np.nan)
    return {
        "stop_signal_observed": signal_observed,
        # RC-R6-1: the horizon row must never expose a future invalidation step.
        "stop_signal_step": np.where(signal_observed, inv_step, -1).astype(np.int64),
        "stop_signal_idx": signal_idx,
        "stop_executable": executable,
        "stop_fill_step": fill_step.astype(np.int64),
        "stop_fill_idx": fill_idx,
        "stop_fill_open": fill_open,
        "stop_reason": reason,
        "stop_gross": gross,
    }


# --------------------------------------------------------------------------- #
# 4. Reference kernel (per-candidate, T0/T1 + differential)                     #
# --------------------------------------------------------------------------- #
def scan_structural_stop_reference(case):
    """Reference: explicit per-candidate loop over the frozen rule.

    case keys (arrays, all length N unless noted):
      side, entry_idx, end_idx, atr0, entry_price, entry_segment,
      open, high, low, close, segment, n_bars,
      raw_sup_top, raw_sup_bottom, raw_res_top, raw_res_bottom,
      td_ends (dict name->array)
    """
    _bump("reference_calls")
    n = len(case["entry_idx"])
    sides = np.asarray(case["side"], float)
    e = np.asarray(case["entry_idx"], np.int64)
    end = np.asarray(case["end_idx"], np.int64)
    atr0 = np.asarray(case["atr0"], float)
    p0 = np.asarray(case["entry_price"], float)
    seg = np.asarray(case["segment"], np.int64)
    ent_seg = np.asarray(case["entry_segment"], np.int64)
    o = np.asarray(case["open"], float)
    h = np.asarray(case["high"], float)
    l = np.asarray(case["low"], float)
    c = np.asarray(case["close"], float)
    n_bars = int(case["n_bars"])
    is_long = sides > 0
    B = boundary_of(is_long, np.asarray(case["raw_sup_bottom"], float),
                    np.asarray(case["raw_res_top"], float))
    top_ = np.asarray(case["raw_sup_top"], float)
    bot_ = np.asarray(case["raw_sup_bottom"], float)
    rtop = np.asarray(case["raw_res_top"], float)
    rbot = np.asarray(case["raw_res_bottom"], float)
    zone_top, zone_bot = zone_edges(is_long, top_, bot_, rtop, rbot)

    out = {
        "inv_step": np.full(n, -1, np.int64),
        "stop_eligible": np.isfinite(B) & np.isfinite(zone_top) & np.isfinite(zone_bot),
        "per_h": {},
    }
    for name in HORIZONS:
        out["per_h"][name] = {
            "stop_signal_observed": np.zeros(n, bool),
            "stop_signal_step": np.full(n, -1, np.int64),
            "stop_executable": np.zeros(n, bool),
            "stop_fill_step": np.full(n, -1, np.int64),
            "stop_fill_idx": np.full(n, -1, np.int64),
            "stop_fill_open": np.full(n, np.nan),
            "stop_gross": np.full(n, np.nan),
            "stop_reason": np.full(n, "", object),
            "exit_step": np.full(n, -1, np.int64),
        }
    for i in range(n):
        inv = reference_invalidation_step(
            sides[i], l, h, c, e[i], end[i], B[i]) if out["stop_eligible"][i] else -1
        out["inv_step"][i] = inv
        for name in HORIZONS:
            m = out["per_h"][name]
            ex = int(case["td_ends"][name][i] - e[i])
            m["exit_step"][i] = ex
            if inv < 0:
                m["stop_reason"][i] = ("not_eligible" if not out["stop_eligible"][i]
                                       else "no_signal")
                continue
            if inv > ex:
                # RC-R6-1: invalidation happens after this horizon's baseline exit.
                m["stop_reason"][i] = "no_signal_before_baseline_exit"
                continue
            m["stop_signal_observed"][i] = True
            m["stop_signal_step"][i] = inv
            nxt = e[i] + inv + 1
            if nxt >= n_bars:
                m["stop_reason"][i] = "no_next_bar"
                continue
            if seg[nxt] != ent_seg[i]:
                m["stop_reason"][i] = "segment_change"
                continue
            if inv + 1 > ex:
                m["stop_reason"][i] = "beyond_baseline_exit"
                continue
            m["stop_executable"][i] = True
            m["stop_fill_step"][i] = inv + 1
            m["stop_fill_idx"][i] = nxt
            m["stop_fill_open"][i] = o[nxt]
            m["stop_gross"][i] = sides[i] * (o[nxt] - p0[i]) / atr0[i]
            m["stop_reason"][i] = "structural_invalidation"
    return out


# --------------------------------------------------------------------------- #
# 5. Production kernel (vectorized, per symbol)                                 #
# --------------------------------------------------------------------------- #
def scan_structural_stop_production(case):
    """Vectorized production kernel. Reuses frozen R5 event steps for the rule."""
    n = len(case["entry_idx"])
    _bump("production_candidate_views", n)
    sides = np.asarray(case["side"], float)
    e = np.asarray(case["entry_idx"], np.int64)
    atr0 = np.asarray(case["atr0"], float)
    p0 = np.asarray(case["entry_price"], float)
    seg = np.asarray(case["segment"], np.int64)
    ent_seg = np.asarray(case["entry_segment"], np.int64)
    o = np.asarray(case["open"], float)
    n_bars = int(case["n_bars"])
    is_long = sides > 0
    B = boundary_of(is_long, np.asarray(case["raw_sup_bottom"], float),
                    np.asarray(case["raw_res_top"], float))
    zone_top, zone_bot = zone_edges(
        is_long, np.asarray(case["raw_sup_top"], float),
        np.asarray(case["raw_sup_bottom"], float),
        np.asarray(case["raw_res_top"], float),
        np.asarray(case["raw_res_bottom"], float))
    eligible = np.isfinite(B) & np.isfinite(zone_top) & np.isfinite(zone_bot)
    inv = np.where(eligible, np.asarray(case["inv_step"], np.int64), -1)

    out = {"inv_step": inv, "stop_eligible": eligible, "per_h": {}}
    for name in HORIZONS:
        exit_step = np.asarray(case["td_ends"][name], np.int64) - e
        r = resolve_treatment(
            inv_step=inv, side=sides, entry_idx=e, entry_price=p0, atr0=atr0,
            open_px=o, segment=seg, n_bars=n_bars, exit_step_H=exit_step,
            eligible=eligible)
        r["exit_step"] = exit_step
        out["per_h"][name] = r
    return out


# --------------------------------------------------------------------------- #
# 6. Paired whole-gid bootstrap                                                 #
# --------------------------------------------------------------------------- #
def paired_gid_bootstrap(delta, gid, weight, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED,
                         batch=200):
    """Paired (candidate-level) whole-gid bootstrap of the weighted mean delta."""
    delta = np.asarray(delta, float)
    weight = np.asarray(weight, float)
    ok = np.isfinite(delta) & np.isfinite(weight)
    delta = delta[ok]; weight = weight[ok]; gid = np.asarray(gid, object)[ok]
    ug, inv = np.unique(gid, return_inverse=True)
    G = len(ug)
    num = np.bincount(inv, weights=delta * weight, minlength=G)   # per-gid sum
    den = np.bincount(inv, weights=weight, minlength=G)
    point = float(num.sum() / den.sum())
    rng = np.random.default_rng(seed)
    reps = np.empty(B, float)
    for b0 in range(0, B, batch):
        b1 = min(B, b0 + batch)
        idx = rng.integers(0, G, size=(b1 - b0, G))
        reps[b0:b1] = num[idx].sum(1) / den[idx].sum(1)
    lo, hi = np.quantile(reps, [0.025, 0.975])
    return {"point": point, "ci_low": float(lo), "ci_high": float(hi),
            "n_gids": int(G), "n_rows": int(len(delta)), "reps": reps}


def weighted_mean(x, w):
    """RC-R6-2: weighted mean over rows with finite x and positive weight."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    return float((x[ok] * w[ok]).sum() / w[ok].sum())


def weighted_quantile(x, w, q):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    ok = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return float("nan")
    x = x[ok]; w = w[ok]
    o = np.argsort(x); x = x[o]; w = w[o]
    cw = np.cumsum(w)
    pos = (cw - 0.5 * w) / cw[-1]
    return float(np.interp(q, pos, x, left=x[0], right=x[-1]))


def decomposition_weighted(baseline, stop, correct, weight):
    """RC-R6-2: weighted GROSS mechanism decomposition (correctness AUDIT_ONLY).

    Weights are the frozen `sample_weight_raw`; they are used as-is on the
    subgroup (never globally renormalized before filtering).
    """
    base = np.asarray(baseline, float)
    stp = np.asarray(stop, float)
    w = np.asarray(weight, float)
    corr = np.asarray(correct, bool)
    ok = np.isfinite(base) & np.isfinite(stp) & np.isfinite(w)
    d = stp - base
    mw = (~corr) & ok
    mc = corr & ok
    out = {"n_rows": int(ok.sum()), "n_wrong": int(mw.sum()),
           "n_correct": int(mc.sum())}
    out["delta_ev_gross"] = weighted_mean(d[ok], w[ok])
    out["delta_wrong_gross"] = weighted_mean(d[mw], w[mw])
    out["delta_correct_gross"] = weighted_mean(d[mc], w[mc])
    wlr = np.maximum(-base, 0.0) - np.maximum(-stp, 0.0)
    out["wlr_gross"] = weighted_mean(wlr[mw], w[mw])
    cpe = np.maximum(base, 0.0) - np.maximum(stp, 0.0)
    out["cpe_gross"] = weighted_mean(cpe[mc], w[mc])
    return out


# --------------------------------------------------------------------------- #
# 7. Per-symbol treatment table (production)                                    #
# --------------------------------------------------------------------------- #
ROW_COLUMNS = [
    "semantic_key", "symbol", "gid", "direction_system", "direction",
    "sample_weight_raw", "direction_correct",
    "entry_time", "entry_price", "ATR0",
    "evaluation_horizon",
    "baseline_exit_time", "baseline_exit_price",
    "baseline_gross_return_atr", "stop_gross_return_atr", "paired_delta_gross_atr",
    "stop_eligible", "stop_signal_observed", "stop_signal_step", "stop_signal_time",
    "stop_signal_bar_close",
    "stop_executable", "stop_fill_step", "stop_fill_time", "stop_fill_open",
    "stop_reason",
    "frozen_sr_bottom", "frozen_sr_top", "frozen_sr_strength",
    "lb_available", "lb_touched_by_signal", "lb_pierced_by_signal",
    "lb_broken_unreclaimed_at_signal",
]


def _prepare_symbol(case_from_l2, st):
    """Build the shared per-symbol case dict from frozen L2 rows + SymbolState."""
    df = case_from_l2
    n_bars = int(st.n_bars)
    fill_t = pd.to_datetime(df["fill_time"]).to_numpy("datetime64[ns]")
    bar_t = pd.to_datetime(st.bar_start_time).to_numpy("datetime64[ns]")
    lookup = {int(t.astype("int64")): i for i, t in enumerate(bar_t)}
    entry_idx = np.array([lookup.get(int(t.astype("int64")), -1) for t in fill_t],
                         dtype=np.int64)
    if (entry_idx < 0).any():
        raise RuntimeError("STOP_R6_FILL_TIME_NOT_IN_FRAME")
    entry_price = df["entry_price"].to_numpy(float)
    if not np.allclose(st.open[entry_idx], entry_price, atol=1e-9):
        raise RuntimeError("STOP_R6_ENTRY_PRICE_ALIGNMENT_MISMATCH")
    end_idx, td_ends = build_horizon_indices(st, entry_idx)
    is_long = (df["direction"].to_numpy(object) == "LONG")
    return {
        "df": df, "entry_idx": entry_idx, "end_idx": end_idx, "td_ends": td_ends,
        "side": np.where(is_long, 1.0, -1.0), "is_long": is_long,
        "entry_price": entry_price, "atr0": df["ATR0"].to_numpy(float),
        "entry_segment": st.segment[entry_idx],
        "open": st.open, "high": st.high, "low": st.low, "close": st.close,
        "segment": st.segment, "n_bars": n_bars,
        "raw_sup_top": df["raw_sup_top"].to_numpy(float),
        "raw_sup_bottom": df["raw_sup_bottom"].to_numpy(float),
        "raw_res_top": df["raw_res_top"].to_numpy(float),
        "raw_res_bottom": df["raw_res_bottom"].to_numpy(float),
        "inv_step": production_invalidation_step(
            df["sr_first_pierce"].to_numpy(np.int64),
            df["sr_same_bar_reclaim"].to_numpy(bool),
            df["sr_first_failed_reclaim"].to_numpy(np.int64)),
    }


def treatment_rows_for_symbol(df_l2, st, direction_system):
    """Production: per (candidate x horizon) treatment rows for one system."""
    sub = df_l2[df_l2["direction_system"] == direction_system].sort_values(
        "semantic_key").reset_index(drop=True)
    if len(sub) == 0:
        return pd.DataFrame(columns=ROW_COLUMNS), None
    case = _prepare_symbol(sub, st)
    prod = scan_structural_stop_production(case)
    is_long = case["is_long"]
    zone_top, zone_bot = zone_edges(
        is_long, case["raw_sup_top"], case["raw_sup_bottom"],
        case["raw_res_top"], case["raw_res_bottom"])
    strength = np.where(is_long, sub["raw_sup_strength"].to_numpy(float),
                        sub["raw_res_strength"].to_numpy(float))
    lb_behind_finite = np.where(
        is_long, np.isfinite(sub["raw_liq_dn_bottom"].to_numpy(float)),
        np.isfinite(sub["raw_liq_up_top"].to_numpy(float)))
    lb_touch = sub["lb_first_touch"].to_numpy(np.int64)
    lb_pierce = sub["lb_first_pierce"].to_numpy(np.int64)
    lb_reclaim = sub["lb_first_reclaim"].to_numpy(np.int64)
    bst = pd.to_datetime(st.bar_start_time)
    decision_time = bst + pd.Timedelta(minutes=15)

    frames = []
    for H in HORIZONS:
        m = prod["per_h"][H]
        base_r = sub[f"{H}_r"].to_numpy(float)
        exit_idx = case["entry_idx"] + m["exit_step"]
        safe_exit = np.clip(exit_idx, 0, case["n_bars"] - 1)
        # RC-R6-5: baseline exit price is the ACTUAL canonical frame close; assert it
        # reproduces the frozen R5 checkpoint return within 1e-12.
        base_close = case["close"][safe_exit]
        recon = case["side"] * (base_close - case["entry_price"]) / case["atr0"]
        if not np.allclose(recon, base_r, atol=1e-12, equal_nan=True):
            raise RuntimeError("STOP_R6_BASELINE_EXIT_ALIGNMENT_MISMATCH")
        base_time = bst[safe_exit] + pd.Timedelta(minutes=15)
        stop_r = np.where(m["stop_executable"], m["stop_gross"], base_r)
        sig_idx = np.where(m["stop_signal_observed"], m["stop_signal_idx"], -1)
        safe_sig = np.clip(sig_idx, 0, case["n_bars"] - 1)
        sig_time = np.where(m["stop_signal_observed"],
                            decision_time[safe_sig].to_numpy("datetime64[ns]"),
                            np.datetime64("NaT", "ns"))
        fill_idx = np.where(m["stop_executable"], m["stop_fill_idx"], -1)
        safe_fill = np.clip(fill_idx, 0, case["n_bars"] - 1)
        fill_time = np.where(m["stop_executable"],
                             bst[safe_fill].to_numpy(), np.datetime64("NaT", "ns"))
        rec = pd.DataFrame({
            "semantic_key": sub["semantic_key"].to_numpy(object),
            "symbol": sub["symbol"].to_numpy(object),
            "gid": sub["gid"].to_numpy(object),
            "direction_system": direction_system,
            "direction": sub["direction"].to_numpy(object),
            "sample_weight_raw": sub["sample_weight_raw"].to_numpy(float),
            "direction_correct": sub["direction_correct"].to_numpy(np.uint8),
            "entry_time": pd.to_datetime(sub["fill_time"]).to_numpy("datetime64[ns]"),
            "entry_price": case["entry_price"], "ATR0": case["atr0"],
            "evaluation_horizon": H,
            "baseline_exit_time": base_time.to_numpy("datetime64[ns]"),
            "baseline_exit_price": base_close,
            "baseline_gross_return_atr": base_r,
            "stop_gross_return_atr": stop_r,
            "paired_delta_gross_atr": stop_r - base_r,
            "stop_eligible": prod["stop_eligible"],
            "stop_signal_observed": m["stop_signal_observed"],
            "stop_signal_step": m["stop_signal_step"].astype(np.int64),
            "stop_signal_time": sig_time,
            "stop_signal_bar_close": np.where(m["stop_signal_observed"],
                                              case["close"][safe_sig], np.nan),
            "stop_executable": m["stop_executable"],
            "stop_fill_step": m["stop_fill_step"].astype(np.int64),
            "stop_fill_time": fill_time.astype("datetime64[ns]"),
            "stop_fill_open": m["stop_fill_open"],
            "stop_reason": m["stop_reason"],
            "frozen_sr_bottom": zone_bot, "frozen_sr_top": zone_top,
            "frozen_sr_strength": strength,
            "lb_available": lb_behind_finite,
            "lb_touched_by_signal": (lb_touch >= 0) & (lb_touch <= m["stop_signal_step"]),
            "lb_pierced_by_signal": (lb_pierce >= 0) & (lb_pierce <= m["stop_signal_step"]),
            "lb_broken_unreclaimed_at_signal":
                (lb_pierce >= 0) & (lb_pierce <= m["stop_signal_step"])
                & ((lb_reclaim < 0) | (lb_reclaim > m["stop_signal_step"])),
        })[ROW_COLUMNS]
        frames.append(rec)
    return pd.concat(frames, ignore_index=True), case


# --------------------------------------------------------------------------- #
# 8. Evidence writers                                                           #
# --------------------------------------------------------------------------- #
def write_csv(path, rows, columns):
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows, columns=columns)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(obj), f, indent=2)


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


# --------------------------------------------------------------------------- #
# 9. T0 / T1 reference-vs-production differential (one symbol)                  #
# --------------------------------------------------------------------------- #
DIFF_FIELDS = ("stop_eligible", "inv_step", "stop_signal_observed",
               "stop_signal_step", "stop_executable", "stop_fill_step",
               "stop_fill_idx", "stop_fill_open", "stop_gross")


def differential_symbol(case, system="E9"):
    """Reference (bar-scan) vs Production (L2-derived) on the SAME case."""
    ref = scan_structural_stop_reference(case)
    prod = scan_structural_stop_production(case)
    out = {"n": int(len(case["entry_idx"])), "mismatch": 0,
           "max_abs_error": 0.0, "first_mismatch": None}
    if not np.array_equal(ref["stop_eligible"], prod["stop_eligible"]):
        out["mismatch"] += 1
        out["first_mismatch"] = out["first_mismatch"] or "stop_eligible"
    if not np.array_equal(ref["inv_step"], prod["inv_step"]):
        out["mismatch"] += 1
        out["first_mismatch"] = out["first_mismatch"] or "inv_step"
    for H in HORIZONS:
        a, b = ref["per_h"][H], prod["per_h"][H]
        for f in ("stop_signal_observed", "stop_signal_step", "stop_executable",
                  "stop_fill_step", "stop_fill_idx"):
            if not np.array_equal(a[f], b[f]):
                out["mismatch"] += 1
                out["first_mismatch"] = out["first_mismatch"] or f"{H}.{f}"
        for f in ("stop_fill_open", "stop_gross"):
            x, y = a[f], b[f]
            m = np.isfinite(x) & np.isfinite(y)
            if not np.array_equal(np.isfinite(x), np.isfinite(y)):
                out["mismatch"] += 1
                out["first_mismatch"] = out["first_mismatch"] or f"{H}.{f}.mask"
            elif m.any():
                e = float(np.max(np.abs(x[m] - y[m])))
                out["max_abs_error"] = max(out["max_abs_error"], e)
                if e > 1e-12:
                    out["mismatch"] += 1
                    out["first_mismatch"] = out["first_mismatch"] or f"{H}.{f}"
    return out


# --------------------------------------------------------------------------- #
# 10. T1.5 integration driver (15 symbols, first 50 candidates)                  #
# --------------------------------------------------------------------------- #
T1_5_PER_SYMBOL = 50


def _boot_sub(vals, gid, w, mask, name):
    """RC-R6-3: whole-gid paired bootstrap restricted to a subgroup mask."""
    if not np.asarray(mask, bool).any():
        return {name: float("nan"), f"{name}_ci_low": float("nan"),
                f"{name}_ci_high": float("nan")}
    b = paired_gid_bootstrap(vals[mask], np.asarray(gid, object)[mask],
                             w[mask])
    return {name: b["point"], f"{name}_ci_low": b["ci_low"],
            f"{name}_ci_high": b["ci_high"]}


def _effect(big, system, horizon, side=None):
    """RC-R6-2/3/4/6: weighted, whole-gid-bootstrapped GROSS effect row."""
    q = big[(big.direction_system == system) & (big.evaluation_horizon == horizon)]
    if side is not None:
        q = q[q.direction == side]
    if len(q) == 0:
        return None
    base = q["baseline_gross_return_atr"].to_numpy(float)
    stp = q["stop_gross_return_atr"].to_numpy(float)
    w = q["sample_weight_raw"].to_numpy(float)
    gid = q["gid"].to_numpy(object)
    corr = q["direction_correct"].to_numpy(bool)
    delta = stp - base
    wlr = np.maximum(-base, 0.0) - np.maximum(-stp, 0.0)
    cpe = np.maximum(base, 0.0) - np.maximum(stp, 0.0)
    dm = ~corr
    cm = corr
    ev = paired_gid_bootstrap(delta, gid, w)
    row = {
        "system": system, "horizon": horizon, "side": side or "ALL",
        "n_rows": int(len(q)), "n_gids": int(ev["n_gids"]),
        "delta_ev_gross": ev["point"],
        "delta_ev_ci_low": ev["ci_low"], "delta_ev_ci_high": ev["ci_high"],
        "break_even_incremental_exit_cost_atr": ev["point"],
        "break_even_incremental_exit_cost_ci_low": ev["ci_low"],
        "break_even_incremental_exit_cost_ci_high": ev["ci_high"],
        "has_positive_gross_cost_capacity": bool(ev["point"] > 0),
        **_boot_sub(delta, gid, w, dm, "delta_wrong_gross"),
        **_boot_sub(delta, gid, w, cm, "delta_correct_gross"),
        **_boot_sub(wlr, gid, w, dm, "wlr_gross"),
        **_boot_sub(cpe, gid, w, cm, "cpe_gross"),
        "stop_eligible_rate": weighted_mean(q["stop_eligible"].to_numpy(float), w),
        "stop_signal_rate": weighted_mean(q["stop_signal_observed"].to_numpy(float), w),
        "stop_executable_rate": weighted_mean(q["stop_executable"].to_numpy(float), w),
        "stop_hit_rate_correct": weighted_mean(
            q["stop_executable"].to_numpy(float)[cm], w[cm]) if cm.any() else float("nan"),
        "stop_hit_rate_wrong": weighted_mean(
            q["stop_executable"].to_numpy(float)[dm], w[dm]) if dm.any() else float("nan"),
        "raw_n_correct": int(cm.sum()), "raw_n_wrong": int(dm.sum()),
    }
    return row


GROUP_STAT_COLUMNS = [
    "symbol", "direction_system", "evaluation_horizon", "n_rows", "n_gids",
    "delta_ev_gross", "baseline_gross_mean", "stop_gross_mean",
    "stop_eligible_rate", "stop_signal_rate", "stop_executable_rate",
    "raw_n_rows", "raw_delta_ev_gross", "raw_baseline_gross_mean",
    "raw_stop_gross_mean",
]


def group_stat_rows(big):
    """FG-R6-7: per-symbol/group evidence under the FORMAL (weighted) estimator.

    Every effect / rate column is the frozen `sample_weight_raw`-weighted mean.
    Raw (unweighted) row means are reported ONLY as additionally-named columns.
    """
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
                stp = q["stop_gross_return_atr"].to_numpy(float)
                rows.append({
                    "symbol": sym, "direction_system": ds, "evaluation_horizon": H,
                    "n_rows": int(len(q)), "n_gids": int(q["gid"].nunique()),
                    "delta_ev_gross": weighted_mean(delta, w),
                    "baseline_gross_mean": weighted_mean(base, w),
                    "stop_gross_mean": weighted_mean(stp, w),
                    "stop_eligible_rate": weighted_mean(
                        q["stop_eligible"].to_numpy(float), w),
                    "stop_signal_rate": weighted_mean(
                        q["stop_signal_observed"].to_numpy(float), w),
                    "stop_executable_rate": weighted_mean(
                        q["stop_executable"].to_numpy(float), w),
                    "raw_n_rows": int(len(q)),
                    "raw_delta_ev_gross": float(delta.mean()) if len(q) else float("nan"),
                    "raw_baseline_gross_mean": float(base.mean()) if len(q) else float("nan"),
                    "raw_stop_gross_mean": float(stp.mean()) if len(q) else float("nan"),
                })
    return rows


EVENT_COLUMNS = ["semantic_key", "symbol", "gid", "direction_system", "direction",
                 "evaluation_horizon", "direction_correct", "stop_signal_step",
                 "stop_signal_time", "stop_executable", "stop_fill_step",
                 "stop_fill_time", "stop_fill_open", "stop_reason",
                 "baseline_gross_return_atr", "stop_gross_return_atr",
                 "paired_delta_gross_atr", "frozen_sr_bottom", "frozen_sr_top",
                 "frozen_sr_strength", "lb_available", "lb_touched_by_signal",
                 "lb_pierced_by_signal", "lb_broken_unreclaimed_at_signal"]


def stop_events_frame(big):
    """Observed-stop event table (shared by the T1.5 and Formal stages)."""
    ev = big[big["stop_signal_observed"]]
    return ev[EVENT_COLUMNS] if len(ev) else pd.DataFrame(columns=EVENT_COLUMNS)


def formal_stop_diagnostics(big, system="E9", horizon="td5"):
    """RC-R6-7 / FG-R6-8: weighted stop diagnostics for the primary cell."""
    q = big[(big.direction_system == system) & (big.evaluation_horizon == horizon)]
    w = q["sample_weight_raw"].to_numpy(float)
    corr = q["direction_correct"].to_numpy(bool)
    ex = q["stop_executable"].to_numpy(bool)
    sig = q["stop_signal_observed"].to_numpy(bool)
    stopped = ex
    base = q["baseline_gross_return_atr"].to_numpy(float)
    stp = q["stop_gross_return_atr"].to_numpy(float)
    d = stp - base
    ss = q["stop_signal_step"].to_numpy(float)
    fs = q["stop_fill_step"].to_numpy(float)
    out = {
        "system": system, "horizon": horizon,
        "stop_eligible_rate": weighted_mean(q["stop_eligible"].to_numpy(float), w),
        "stop_signal_rate": weighted_mean(sig.astype(float), w),
        "stop_executable_rate": weighted_mean(ex.astype(float), w),
        "stop_hit_rate_correct": weighted_mean(ex.astype(float)[corr], w[corr]) if corr.any() else float("nan"),
        "stop_hit_rate_wrong": weighted_mean(ex.astype(float)[~corr], w[~corr]) if (~corr).any() else float("nan"),
        "signal_step_p25": weighted_quantile(ss[sig], w[sig], 0.25),
        "signal_step_median": weighted_quantile(ss[sig], w[sig], 0.50),
        "signal_step_p75": weighted_quantile(ss[sig], w[sig], 0.75),
        "fill_step_p25": weighted_quantile(fs[stopped], w[stopped], 0.25),
        "fill_step_median": weighted_quantile(fs[stopped], w[stopped], 0.50),
        "fill_step_p75": weighted_quantile(fs[stopped], w[stopped], 0.75),
    }
    # signal-bar close -> next-open execution gap, in ATR0 units (executed stops)
    side_arr = np.where(q["direction"].to_numpy(object) == "LONG", 1.0, -1.0)
    sbc = q["stop_signal_bar_close"].to_numpy(float)
    gap_atr = side_arr * (q["stop_fill_open"].to_numpy(float) - sbc) \
        / q["ATR0"].to_numpy(float)
    out["signal_close_to_next_open_gap_atr_p25"] = weighted_quantile(
        gap_atr[stopped], w[stopped], 0.25)
    out["signal_close_to_next_open_gap_atr_median"] = weighted_quantile(
        gap_atr[stopped], w[stopped], 0.50)
    out["signal_close_to_next_open_gap_atr_p75"] = weighted_quantile(
        gap_atr[stopped], w[stopped], 0.75)
    # FG-R6-8: signal -> fill WALL-CLOCK elapsed time, for executable stops.
    sig_t = pd.to_datetime(q["stop_signal_time"])
    fil_t = pd.to_datetime(q["stop_fill_time"])
    wc_min = ((fil_t - sig_t).dt.total_seconds() / 60.0).to_numpy(float)
    out["signal_to_fill_wall_clock_minutes_p25"] = weighted_quantile(
        wc_min[stopped], w[stopped], 0.25)
    out["signal_to_fill_wall_clock_minutes_median"] = weighted_quantile(
        wc_min[stopped], w[stopped], 0.50)
    out["signal_to_fill_wall_clock_minutes_p75"] = weighted_quantile(
        wc_min[stopped], w[stopped], 0.75)
    out["baseline_gross_mean_stopped"] = weighted_mean(base[stopped], w[stopped])
    out["stop_gross_mean_stopped"] = weighted_mean(stp[stopped], w[stopped])
    out["paired_gross_improvement_stopped"] = weighted_mean(d[stopped], w[stopped])
    if (stopped & corr).any():
        out["frac_stopped_correct_baseline_td5_profitable"] = weighted_mean(
            (base[stopped & corr] > 0).astype(float), w[stopped & corr])
    else:
        out["frac_stopped_correct_baseline_td5_profitable"] = float("nan")
    if (stopped & ~corr).any():
        out["frac_stopped_wrong_loss_reduced"] = weighted_mean(
            (np.maximum(-stp, 0.0) < np.maximum(-base, 0.0)).astype(float)[stopped & ~corr],
            w[stopped & ~corr])
    else:
        out["frac_stopped_wrong_loss_reduced"] = float("nan")
    st = q["frozen_sr_strength"].to_numpy(float)
    out["frozen_sr_strength_p25"] = weighted_quantile(st, w, 0.25)
    out["frozen_sr_strength_median"] = weighted_quantile(st, w, 0.50)
    out["frozen_sr_strength_p75"] = weighted_quantile(st, w, 0.75)
    # FG-R6-8: the Liquidity-behind question is "when the SR stop signal fires,
    # is LB simultaneously touched / pierced / broken-unreclaimed?". That is a
    # rate CONDITIONAL ON an observed SR stop signal, NOT a population incidence.
    out["lb_*_denominator"] = "observed_sr_stop_signal"
    cond = sig
    for col, name in (("lb_available", "lb_available"),
                      ("lb_touched_by_signal", "lb_touched_by_signal"),
                      ("lb_pierced_by_signal", "lb_pierced_by_signal"),
                      ("lb_broken_unreclaimed_at_signal",
                       "lb_broken_unreclaimed_at_signal")):
        v = q[col].to_numpy(float)
        out[f"{name}_given_signal"] = (
            weighted_mean(v[cond], w[cond]) if cond.any() else float("nan"))
    # Optional, explicitly-named population incidence (NOT the primary question).
    out["lb_available_population_incidence"] = weighted_mean(
        q["lb_available"].to_numpy(float), w)
    out["lb_touched_by_signal_population_incidence"] = weighted_mean(
        q["lb_touched_by_signal"].to_numpy(float), w)
    out["lb_pierced_by_signal_population_incidence"] = weighted_mean(
        q["lb_pierced_by_signal"].to_numpy(float), w)
    out["lb_broken_unreclaimed_at_signal_population_incidence"] = weighted_mean(
        q["lb_broken_unreclaimed_at_signal"].to_numpy(float), w)
    return out


GROSS_VERDICTS = ("GROSS_STOP_EDGE_SUPPORTED_UNIVERSAL", "GROSS_STOP_HARMFUL",
                  "GROSS_SIDE_HETEROGENEITY_REQUIRES_FOLLOWUP",
                  "NO_IDENTIFIABLE_GROSS_STOP_EDGE")


def formal_verdict(primary_row, side_rows):
    """RC-R6-12: Amendment-A1 frozen GROSS verdict categories."""
    long_row = next(r for r in side_rows if r["side"] == "LONG")
    short_row = next(r for r in side_rows if r["side"] == "SHORT")
    long_neg = long_row["delta_ev_ci_high"] < 0
    short_neg = short_row["delta_ev_ci_high"] < 0
    long_pos = long_row["delta_ev_ci_low"] > 0
    short_pos = short_row["delta_ev_ci_low"] > 0
    if (primary_row["delta_ev_ci_low"] > 0
            and primary_row["delta_wrong_gross_ci_low"] > 0
            and primary_row["wlr_gross"] > 0
            and not long_neg and not short_neg):
        return "GROSS_STOP_EDGE_SUPPORTED_UNIVERSAL"
    if primary_row["delta_ev_ci_high"] < 0:
        return "GROSS_STOP_HARMFUL"
    if (long_pos and short_neg) or (short_pos and long_neg):
        return "GROSS_SIDE_HETEROGENEITY_REQUIRES_FOLLOWUP"
    return "NO_IDENTIFIABLE_GROSS_STOP_EDGE"


def run_t1_5(verbose=True):
    t_start = time.time()
    reset_counters()
    frozen = verify_frozen_inputs()
    l2 = load_r5_l2()
    _bump("direction_artifact_sha_verifications")

    all_rows = []
    diffs = []
    per_symbol = []
    for sym in SYMBOLS:
        sub_sym = l2[l2["symbol"] == sym]
        if len(sub_sym) == 0:
            continue
        keys = sorted(sub_sym["semantic_key"].unique())[:T1_5_PER_SYMBOL]
        sub = sub_sym[sub_sym["semantic_key"].isin(keys)]
        st = load_symbol_state(sym)
        _bump("execution_frame_loads")
        sym_diff = {}
        for ds in ("A9", "E9"):
            rows, case = treatment_rows_for_symbol(sub, st, ds)
            if case is None or len(rows) == 0:
                continue
            all_rows.append(rows)
            sym_diff[ds] = differential_symbol(case, ds)
        diffs.append({"symbol": sym, **sym_diff})
        per_symbol.append({"symbol": sym, "n_keys": int(len(keys)),
                           "n_A9": int((sub.direction_system == "A9").sum()),
                           "n_E9": int((sub.direction_system == "E9").sum())})
    big = pd.concat(all_rows, ignore_index=True)

    # RC-R6-13: horizon causality — TD1 signal count <= TD3 <= TD5 (non-strict)
    causality = []
    caus_ok = True
    for sym in sorted(big.symbol.unique()):
        for ds in ("A9", "E9"):
            c = {H: int(big[(big.symbol == sym) & (big.direction_system == ds)
                            & (big.evaluation_horizon == H)]["stop_signal_observed"].sum())
                 for H in HORIZONS}
            ok = c["td1"] <= c["td3"] <= c["td5"]
            caus_ok = caus_ok and ok
            causality.append({"symbol": sym, "system": ds, **c, "ok": bool(ok)})

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

    # FG-R6-7: the SAME weighted group implementation is used by BOTH stages.
    group_df = pd.DataFrame(group_stat_rows(big), columns=GROUP_STAT_COLUMNS)
    events_df = stop_events_frame(big)

    max_abs = 0.0
    for d in diffs:
        for ds in ("A9", "E9"):
            if ds in d:
                max_abs = max(max_abs, d[ds]["max_abs_error"])
    n_mismatch = sum(d[ds]["mismatch"] for d in diffs for ds in ("A9", "E9") if ds in d)

    write_csv(T1_5_PRIMARY_CSV, primary_df, list(primary_df.columns))
    write_csv(T1_5_SIDE_STATS_CSV, side_df, list(side_df.columns))
    write_csv(T1_5_GROUP_STATS_CSV, group_df, list(group_df.columns))
    write_csv(T1_5_STOP_EVENTS_CSV, events_df, EVENT_COLUMNS)

    perf = {
        "r5_artifact_loads": int(COUNTERS["r5_artifact_loads"]),
        "execution_frame_loads": int(COUNTERS["execution_frame_loads"]),
        "direction_artifact_sha_verifications":
            int(COUNTERS["direction_artifact_sha_verifications"]),
        "sr_recompute_count": int(COUNTERS["sr_recompute_count"]),
        "full_history_recompute_count": int(COUNTERS["full_history_recompute_count"]),
        "reference_calls": int(COUNTERS["reference_calls"]),
        "production_candidate_views": int(COUNTERS["production_candidate_views"]),
        "runtime_sec": time.time() - t_start,
    }
    summary = {
        "task_id": TASK_ID, "stage": STAGE_T1_5,
        "scientific_status": "NON_SCIENTIFIC_INTEGRATION_ONLY",
        "n_symbols": len(per_symbol), "n_rows_total": int(len(big)),
        "per_symbol": per_symbol,
        "differential": {"n_symbols_with_reference": len(diffs),
                         "n_mismatch": int(n_mismatch),
                         "max_abs_error": float(max_abs)},
        "per_symbol_differential": diffs,
        "horizon_causality": {"ok": bool(caus_ok), "per_symbol_system": causality},
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
            "Full 13773-Candidate Formal R6 NOT run (not authorized).",
            "No NET verdict; GROSS paired policy effect only.",
            "Friction proxy not used for any verdict.",
            "T1.5 numbers are integration validation only, not a research result.",
        ],
    }
    write_json(T1_5_SUMMARY_JSON, summary)

    artifact_shas = {os.path.basename(p): sha256_file(p)
                     for p in (T1_5_PRIMARY_CSV, T1_5_SIDE_STATS_CSV,
                               T1_5_GROUP_STATS_CSV, T1_5_STOP_EVENTS_CSV,
                               T1_5_SUMMARY_JSON)}
    manifest = {
        "task_id": TASK_ID, "stage": STAGE_T1_5,
        "base_sha": BASE_SHA, "reviewed_parent_sha": REVIEWED_PARENT,
        "generator_code_sha": _git_head_sha(),
        "frozen_inputs_sha256": frozen,
        "cost_governance": summary["cost_governance"],
        "differential": summary["differential"],
        "performance": perf,
        "artifact_sha256": artifact_shas,
        "serialization_manifest_last": True,
        "authorized_review_sha_note":
            "R6 PRE-T2 authorized under FUTURE-R6 Amendment A1 (GROSS primary).",
    }
    write_json(T1_5_MANIFEST_JSON, manifest)
    if verbose:
        print(json.dumps({"summary": summary["differential"],
                          "performance": perf}, indent=2))
    return summary


FORMAL_ARTIFACT = os.path.join(ARTIFACT_DIR, "structural_stop_row_metrics_v1.parquet")
# FG-R6-5: Formal writes the CANONICAL un-prefixed evidence names.
FORMAL_MANIFEST = MANIFEST_JSON
FORMAL_SUMMARY = SUMMARY_JSON

FROZEN_FORMAL = {"symbols": 15, "semantic_keys": 13773, "gids": 638,
                 "long_gids": 319, "short_gids": 319,
                 "base_rows": 27546, "base_per_system": 13773,
                 "rows_per_system": 41319, "total_rows": 82638}

# FG-R6-9: exact frozen Formal performance budget.
PERF_EXPECTED = {"r5_artifact_loads": 1, "execution_frame_loads": 15,
                 "direction_artifact_sha_verifications": 1,
                 "sr_recompute_count": 0, "full_history_recompute_count": 0,
                 "reference_calls": 0,
                 "production_candidate_views":
                     FROZEN_FORMAL["base_rows"]}


def _per_system_gid_weight_gate(l2, atol=1e-9):
    """FG-R6-1: per-SYSTEM gid weight identity.

    A9 and E9 each carry the frozen Candidate weight INDEPENDENTLY, and the A9/E9
    duplication is NOT halved. The correct identity is therefore PER SYSTEM:

        within A9: every gid weight sum == 1
        within E9: every gid weight sum == 1

    The MERGED (A9 + E9) per-gid weight legitimately sums to ~2 and is NEVER gated.
    Both per-system flags are persisted and required by the Formal population gate.
    """
    rep = {}
    for ds, key in (("A9", "a9"), ("E9", "e9")):
        g = l2[l2.direction_system == ds]
        gw = g.groupby("gid")["sample_weight_raw"].sum().to_numpy(float)
        ok = bool(len(gw) == FROZEN_FORMAL["gids"]
                  and np.allclose(gw, 1.0, atol=atol))
        rep[f"{key}_gid_weight_ok"] = ok
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
    """FG-R6-2: mechanical base-L2 population identity."""
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
    # ---- gid / oracle_direction identity ----
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
    """FG-R6-1/2: frozen Formal population hard gates (pre-write).

    `big` is accepted for call-shape stability but the frozen R5 identity lives in
    L2; treatment-row population is verified mechanically on the written artifact
    (FG-R6-10) as well.
    """
    mism = {}
    mism.update(_base_l2_population_gates(l2))
    mism.update(_per_system_gid_weight_mismatches(l2))
    return mism


def _write_formal_artifact(df, path):
    """FG-R6-3: the ONE canonical Formal parquet writer (monkeypatchable)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)


def _verify_formal_artifact(path, frozen_keys):
    """RC-R6-11 / FG-R6-10: post-write key verification (key columns only)."""
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
    # FG-R6-10: exactly the six (system x horizon) rows per semantic_key.
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


# --------------------------------------------------------------------------- #
# 12. FG-R6-11 Formal environment provenance                                    #
# --------------------------------------------------------------------------- #
PROVENANCE_FIELDS = ("environment_contract_id", "cache_schema_version", "identity",
                     "code_identity", "raw_sha256", "execution_frame_sha256",
                     "sha256", "rows", "max_bars")


def _env_provenance_records(states):
    """FG-R6-11: canonical SymbolState environment provenance, one record/symbol."""
    recs = []
    for st in states:
        p = getattr(st, "env_provenance", None)
        p = p if isinstance(p, dict) else {}
        rec = {"symbol": getattr(st, "symbol", p.get("symbol"))}
        rec.update({k: p.get(k) for k in PROVENANCE_FIELDS})
        recs.append(rec)
    return recs


def _env_provenance_gate(recs):
    """FG-R6-11: exact 15-symbol coverage + frozen environment contract."""
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


def _formal_perf_counters(t_start):
    return {
        "r5_artifact_loads": int(COUNTERS["r5_artifact_loads"]),
        "execution_frame_loads": int(COUNTERS["execution_frame_loads"]),
        "direction_artifact_sha_verifications":
            int(COUNTERS["direction_artifact_sha_verifications"]),
        "sr_recompute_count": int(COUNTERS["sr_recompute_count"]),
        "full_history_recompute_count": int(COUNTERS["full_history_recompute_count"]),
        "reference_calls": int(COUNTERS["reference_calls"]),
        "production_candidate_views": int(COUNTERS["production_candidate_views"]),
        "runtime_sec": time.time() - t_start,
    }


def run_formal_r6(*, allow_full=False, authorized_review_sha=None,
                  write_artifacts=True, verbose=False):
    """FG-R6-3: frozen Formal R6 path behind an explicit authorization gate.

    Frozen ordering:
      1. verify authorized SHA
      2. verify frozen inputs
      3. compute full Production rows
      4. population gates
      5. performance gates
      6. ONLY if both PASS: write canonical Formal parquet
      7. mechanically verify the written parquet
      8. compute Formal statistics / evidence
      9. write Formal evidence files (canonical, FULL population)
     10. compute SHA256 of every Formal artifact
     11. write Formal manifest LAST
    """
    if allow_full is not True:
        raise RuntimeError("STOP_R6_FULL_POPULATION_NOT_AUTHORIZED")
    if not authorized_review_sha:
        raise RuntimeError("STOP_R6_AUTHORIZED_REVIEW_SHA_REQUIRED")
    # FG-R6-4: a Formal manifest must never describe a run without a canonical,
    # written and parquet-verified artifact.
    if write_artifacts is not True:
        raise RuntimeError("STOP_R6_FORMAL_ARTIFACT_WRITE_REQUIRED")
    head = _git_head_sha()
    if head != authorized_review_sha:
        raise RuntimeError(
            f"STOP_R6_GENERATOR_SHA_MISMATCH head={head} "
            f"authorized_review_sha={authorized_review_sha}")
    t_start = time.time()
    reset_counters()
    frozen = verify_frozen_inputs()
    l2 = load_r5_l2()
    _bump("direction_artifact_sha_verifications")

    # ---- step 3: full Production rows. ONE frame object per symbol, reused by
    # A9 and E9 (FG-R6-9).
    all_rows = []
    states = []
    for sym in SYMBOLS:
        sub = l2[l2.symbol == sym]
        if len(sub) == 0:
            continue
        st = load_symbol_state(sym)
        states.append(st)
        _bump("execution_frame_loads")
        for ds in ("A9", "E9"):
            rows, _ = treatment_rows_for_symbol(sub, st, ds)
            all_rows.append(rows)
    big = pd.concat(all_rows, ignore_index=True)
    frozen_keys = set(l2.semantic_key.astype(str).unique())

    # ---- steps 4-5: gates MUST pass before any canonical artifact is written.
    pop_mismatch = _formal_population_gates(l2, big)
    gid_weight_report = _per_system_gid_weight_gate(l2)
    env_records = _env_provenance_records(states)
    env_mismatch = _env_provenance_gate(env_records)
    perf = _formal_perf_counters(t_start)
    perf_mismatch = {k: (perf[k], v) for k, v in PERF_EXPECTED.items()
                     if perf[k] != v}
    if pop_mismatch:
        raise RuntimeError(f"STOP_R6_FORMAL_POPULATION_GATE {pop_mismatch}")
    if perf_mismatch:
        raise RuntimeError(f"STOP_R6_FORMAL_PERFORMANCE_GATE {perf_mismatch}")
    if env_mismatch:
        raise RuntimeError(
            f"STOP_R6_FORMAL_ENVIRONMENT_PROVENANCE_GATE {env_mismatch}")

    # ---- step 6-7: canonical artifact write + mechanical key verification.
    _write_formal_artifact(big, FORMAL_ARTIFACT)
    artifact_report = _verify_formal_artifact(FORMAL_ARTIFACT, frozen_keys)
    if not artifact_report.get("pass"):
        raise RuntimeError(
            f"STOP_R6_FORMAL_ARTIFACT_KEY_MISMATCH {artifact_report}")

    # ---- step 8: Formal statistics / evidence (FULL population, weighted).
    primary_row = _effect(big, "E9", PRIMARY_HORIZON)
    side_rows = [_effect(big, "E9", PRIMARY_HORIZON, side=s) for s in ("LONG", "SHORT")]
    verdict = formal_verdict(primary_row, side_rows)
    diag = formal_stop_diagnostics(big, "E9", PRIMARY_HORIZON)
    primary_df = pd.DataFrame(
        [e for ds in ("A9", "E9") for H in HORIZONS
         if (e := _effect(big, ds, H)) is not None])
    side_df = pd.DataFrame(
        [e for ds in ("A9", "E9") for H in HORIZONS for sd in ("LONG", "SHORT")
         if (e := _effect(big, ds, H, side=sd)) is not None])
    group_df = pd.DataFrame(group_stat_rows(big), columns=GROUP_STAT_COLUMNS)
    events_df = stop_events_frame(big)

    # ---- step 9: canonical Formal evidence files (never the T1.5 subset).
    write_csv(PRIMARY_CSV, primary_df, list(primary_df.columns))
    write_csv(SIDE_STATS_CSV, side_df, list(side_df.columns))
    write_csv(GROUP_STATS_CSV, group_df, list(GROUP_STAT_COLUMNS))
    write_csv(STOP_EVENTS_CSV, events_df, EVENT_COLUMNS)

    summary = {
        "task_id": TASK_ID, "stage": STAGE_FORMAL,
        "authorized_review_sha": authorized_review_sha,
        "generator_code_sha": head, "reviewed_parent_sha": head,
        "population": "full", "n_primary_rows": int(len(big)),
        "formal_verdict": verdict,
        "primary_row": primary_row, "side_rows": side_rows,
        "stop_diagnostics": diag,
        "population_gates": {"pass": not pop_mismatch, "mismatch": pop_mismatch,
                             "gid_weight": gid_weight_report},
        "performance_gates": {"pass": not perf_mismatch,
                              "mismatch": perf_mismatch, "expected": PERF_EXPECTED},
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

    # ---- step 10-11: SHA256 of EVERY Formal artifact, then manifest LAST.
    formal_artifacts = [FORMAL_ARTIFACT, PRIMARY_CSV, SIDE_STATS_CSV,
                        GROUP_STATS_CSV, STOP_EVENTS_CSV, FORMAL_SUMMARY]
    art_shas = {os.path.basename(p): sha256_file(p) for p in formal_artifacts}
    manifest = {
        "task_id": TASK_ID, "stage": STAGE_FORMAL,
        "base_sha": BASE_SHA,
        "authorized_review_sha": authorized_review_sha,
        "generator_code_sha": head, "reviewed_parent_sha": head,
        "population": "full", "n_primary_rows": int(len(big)),
        "frozen_inputs_sha256": frozen,
        "population_gates": summary["population_gates"],
        "performance_gates": summary["performance_gates"],
        "artifact_integrity_gates": artifact_report,
        "environment_provenance_gates": summary["environment_provenance_gates"],
        "environment_provenance": env_records,
        "formal_verdict": verdict,
        "cost_governance": summary["cost_governance"],
        "artifact_sha256": art_shas,
        "serialization_manifest_last": True,
        "all_pass": bool(not pop_mismatch and not perf_mismatch and not env_mismatch
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

