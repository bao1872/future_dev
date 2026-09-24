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
BASE_SHA = "c0126551040f3e758ba5352ecb693bcbff0ad8b0"
REVIEWED_PARENT = "c0126551040f3e758ba5352ecb693bcbff0ad8b0"

ARTIFACT_DIR = os.path.join("artifacts", "structural_stop_v1")
EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "structural_stop_v1_manifest.json")
SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "structural_stop_v1_summary.json")
PRIMARY_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_primary.csv")
GROUP_STATS_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_group_stats.csv")
STOP_EVENTS_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_stop_events.csv")
SIDE_STATS_CSV = os.path.join(EVIDENCE_DIR, "structural_stop_v1_side_stats.csv")

STAGE_T1_5 = "t1_5_integration"

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
            "direction_artifact_loads": 0, "sr_recompute_count": 0,
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
    "sample_weight_raw", "direction_correct", "entry_price", "ATR0", "fill_time",
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
    signal_observed = eligible & (inv_step >= 0)
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
        "stop_signal_step": inv_step.astype(np.int64),
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


def decomposition(baseline, stop, correct):
    """GROSS mechanism decomposition (correctness AUDIT_ONLY, never feeds the rule)."""
    base = np.asarray(baseline, float)
    stp = np.asarray(stop, float)
    correct = np.asarray(correct, bool)
    ok = np.isfinite(base) & np.isfinite(stp)
    d = (stp - base)[ok]
    c = correct[ok]
    out = {"n_rows": int(ok.sum()), "n_wrong": int((~c).sum()),
           "n_correct": int(c.sum())}
    out["delta_ev_gross"] = float(np.mean(d)) if d.size else float("nan")
    out["delta_wrong_gross"] = (float(np.mean(d[~c])) if (~c).any() else float("nan"))
    out["delta_correct_gross"] = (float(np.mean(d[c])) if c.any() else float("nan"))
    # wrong-loss reduction: Loss(baseline) - Loss(stop) among wrong
    if (~c).any():
        out["wlr_gross"] = float(np.mean(
            np.maximum(-base[ok][~c], 0.0) - np.maximum(-stp[ok][~c], 0.0)))
    else:
        out["wlr_gross"] = float("nan")
    # correct-profit erosion: Profit(baseline) - Profit(stop) among correct
    if c.any():
        out["cpe_gross"] = float(np.mean(
            np.maximum(base[ok][c], 0.0) - np.maximum(stp[ok][c], 0.0)))
    else:
        out["cpe_gross"] = float("nan")
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
        base_time = bst[safe_exit] + pd.Timedelta(minutes=15)
        stop_r = np.where(m["stop_executable"], m["stop_gross"], base_r)
        sig_idx = np.where(m["stop_signal_observed"], m["stop_signal_idx"], -1)
        safe_sig = np.clip(sig_idx, 0, case["n_bars"] - 1)
        sig_time = np.where(m["stop_signal_observed"],
                            decision_time[safe_sig].to_numpy("datetime64[ns]"),
                            np.datetime64("NaT"))
        fill_idx = np.where(m["stop_executable"], m["stop_fill_idx"], -1)
        safe_fill = np.clip(fill_idx, 0, case["n_bars"] - 1)
        fill_time = np.where(m["stop_executable"],
                             bst[safe_fill].to_numpy(), np.datetime64("NaT"))
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
            "baseline_exit_price": case["entry_price"]
                + case["side"] * base_r * case["atr0"],
            "baseline_gross_return_atr": base_r,
            "stop_gross_return_atr": stop_r,
            "paired_delta_gross_atr": stop_r - base_r,
            "stop_eligible": prod["stop_eligible"],
            "stop_signal_observed": m["stop_signal_observed"],
            "stop_signal_step": m["stop_signal_step"].astype(np.int64),
            "stop_signal_time": sig_time,
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


def _effect(big, system, horizon, side=None):
    q = big[(big.direction_system == system) & (big.evaluation_horizon == horizon)]
    if side is not None:
        q = q[q.direction == side]
    if len(q) == 0:
        return None
    boot = paired_gid_bootstrap(
        q["paired_delta_gross_atr"].to_numpy(float),
        q["gid"].to_numpy(object), q["sample_weight_raw"].to_numpy(float))
    dec = decomposition(q["baseline_gross_return_atr"].to_numpy(float),
                        q["stop_gross_return_atr"].to_numpy(float),
                        q["direction_correct"].to_numpy(bool))
    return {"system": system, "horizon": horizon, "side": side or "ALL",
            "n_rows": int(len(q)), "n_gids": boot["n_gids"],
            "delta_ev_gross": boot["point"],
            "ci_low": boot["ci_low"], "ci_high": boot["ci_high"],
            **{k: dec[k] for k in ("delta_wrong_gross", "wlr_gross",
                                   "delta_correct_gross", "cpe_gross")},
            "stop_eligible_rate": float(q["stop_eligible"].mean()),
            "stop_signal_rate": float(q["stop_signal_observed"].mean()),
            "stop_executable_rate": float(q["stop_executable"].mean())}


def run_t1_5(verbose=True):
    t_start = time.time()
    reset_counters()
    frozen = verify_frozen_inputs()
    l2 = load_r5_l2()
    _bump("direction_artifact_loads")

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

    group_rows = []
    for sym in sorted(big.symbol.unique()):
        for ds in ("A9", "E9"):
            for H in HORIZONS:
                q = big[(big.symbol == sym) & (big.direction_system == ds)
                        & (big.evaluation_horizon == H)]
                if len(q) == 0:
                    continue
                group_rows.append({
                    "symbol": sym, "direction_system": ds, "evaluation_horizon": H,
                    "n_rows": int(len(q)),
                    "stop_eligible_rate": float(q.stop_eligible.mean()),
                    "stop_signal_rate": float(q.stop_signal_observed.mean()),
                    "stop_executable_rate": float(q.stop_executable.mean()),
                    "delta_ev_gross": float(q.paired_delta_gross_atr.mean()),
                    "baseline_gross_mean": float(q.baseline_gross_return_atr.mean()),
                    "stop_gross_mean": float(q.stop_gross_return_atr.mean())})
    group_df = pd.DataFrame(group_rows)

    ev = big[big.stop_signal_observed].copy()
    ev_cols = ["semantic_key", "symbol", "gid", "direction_system", "direction",
               "evaluation_horizon", "direction_correct", "stop_signal_step",
               "stop_signal_time", "stop_executable", "stop_fill_step",
               "stop_fill_time", "stop_fill_open", "stop_reason",
               "baseline_gross_return_atr", "stop_gross_return_atr",
               "paired_delta_gross_atr", "frozen_sr_bottom", "frozen_sr_top",
               "frozen_sr_strength", "lb_available", "lb_touched_by_signal",
               "lb_pierced_by_signal", "lb_broken_unreclaimed_at_signal"]
    events_df = ev[ev_cols] if len(ev) else pd.DataFrame(columns=ev_cols)

    max_abs = 0.0
    for d in diffs:
        for ds in ("A9", "E9"):
            if ds in d:
                max_abs = max(max_abs, d[ds]["max_abs_error"])
    n_mismatch = sum(d[ds]["mismatch"] for d in diffs for ds in ("A9", "E9") if ds in d)

    write_csv(PRIMARY_CSV, primary_df, list(primary_df.columns))
    write_csv(SIDE_STATS_CSV, side_df, list(side_df.columns))
    write_csv(GROUP_STATS_CSV, group_df, list(group_df.columns))
    write_csv(STOP_EVENTS_CSV, events_df, ev_cols)

    perf = {
        "r5_artifact_loads": int(COUNTERS["r5_artifact_loads"]),
        "execution_frame_loads": int(COUNTERS["execution_frame_loads"]),
        "direction_artifact_loads": int(COUNTERS["direction_artifact_loads"]),
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
    write_json(SUMMARY_JSON, summary)

    artifact_shas = {os.path.basename(p): sha256_file(p)
                     for p in (PRIMARY_CSV, SIDE_STATS_CSV, GROUP_STATS_CSV,
                               STOP_EVENTS_CSV, SUMMARY_JSON)}
    manifest = {
        "task_id": TASK_ID, "stage": STAGE_T1_5,
        "base_sha": BASE_SHA, "reviewed_parent_sha": REVIEWED_PARENT,
        "generator_code_sha": _git_head_sha(),
        "frozen_inputs_sha256": frozen,
        "cost_governance": summary["cost_governance"],
        "differential": summary["differential"],
        "performance": perf,
        "artifact_sha256": artifact_shas,
        "authorized_review_sha_note":
            "R6 PRE-T2 authorized under FUTURE-R6 Amendment A1 (GROSS primary).",
    }
    write_json(MANIFEST_JSON, manifest)
    if verbose:
        print(json.dumps({"summary": summary["differential"],
                          "performance": perf}, indent=2))
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

