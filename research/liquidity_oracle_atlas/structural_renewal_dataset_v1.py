"""FUTURE-R8-M15-STRUCTURAL-RENEWAL-DATASET-V1.

R8 materializes the structural-renewal dataset for the Opportunity Value
program. It is LABEL / DATA engineering only: no model, no policy, no verdict.

Frozen upstream (do NOT redesign):
    DIRECTION_GATED_EXPERTS_V1 = CLOSED / FROZEN
    ENTRY_PATH_ATLAS_V1        = CLOSED / PASS
    STRUCTURAL_STOP_V1         = NO_IDENTIFIABLE_GROSS_STOP_EDGE -> NO_STRUCTURAL_STOP
    AHEAD_SR_NONACCEPTANCE_TP  = NO_IDENTIFIABLE_GROSS_TP_EDGE   -> NO_TP

Semantics (§1/§2): a structure touch is NOT a Stop and NOT a Take-Profit. It
means the previous local opportunity has ENDED and the market must be
re-evaluated. R8 labels the ONE-STEP economic outcome from a decision epoch to
the next structural decision epoch (or the frozen terminal horizon).

Barriers (§3) are the NEAR edges of the canonical m15 SR zones only:
    LONG : favorable = resistance_bottom ; adverse = support_top
    SHORT: favorable = support_top       ; adverse = resistance_bottom
No far-edge optimization, no strength threshold, no ATR multiplier.

Governance: all economics are GROSS (Amendment A1 carried forward).
"""

import hashlib
import json
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
    derive_m15_candidate_gate,
)
from research.liquidity_oracle_atlas.entry_path_atlas_v1 import (
    SYMBOLS,
    extract_zone_geometry,
)
from research.liquidity_oracle_atlas.build_struct33_dataset_v1 import STRUCT33
from research.liquidity_oracle_atlas.direction_gated_experts_v1 import (
    orient_struct33_router_side,
)

TASK_ID = "FUTURE-R8-M15-STRUCTURAL-RENEWAL-DATASET-V1"
BASE_SHA = "f3ca3a04317126e8f35afe7430d7aea202a9175a"

ARTIFACT_DIR = os.path.join("artifacts", "opportunity_value_v1")
STATE_PARQUET = os.path.join(ARTIFACT_DIR, "state_v1.parquet")
SIDE_FEATURES_PARQUET = os.path.join(ARTIFACT_DIR, "side_features_v1.parquet")
LABEL_PARQUETS = {
    "train": os.path.join(ARTIFACT_DIR, "labels_train_v1.parquet"),
    "val": os.path.join(ARTIFACT_DIR, "labels_val_v1.parquet"),
    "test": os.path.join(ARTIFACT_DIR, "labels_test_v1.parquet"),
}
MANIFEST_JSON = os.path.join(ARTIFACT_DIR, "manifest_v1.json")

HORIZONS = ("td1", "td3", "td5")
HORIZON_DAYS = {"td1": 1, "td3": 3, "td5": 5}
PRIMARY_HORIZON = "td5"

# §6: 15m OHLC does NOT reveal intrabar order. BOTH stays an independent class.
EVENT_NONE = np.int8(0)
EVENT_FAV = np.int8(1)
EVENT_ADV = np.int8(2)
EVENT_BOTH = np.int8(3)
EVENT_CLASS_NAME = {EVENT_NONE: "NONE", EVENT_FAV: "FAVORABLE_FIRST",
                    EVENT_ADV: "ADVERSE_FIRST", EVENT_BOTH: "BOTH_SAME_BAR"}

STATE_SIDE = ("LONG", "SHORT")
# Join key carried (as leading columns) by side_features_v1.parquet so R9/R10
# never rely on positional alignment.
SIDE_KEY = ("symbol", "decision_bar", "side")

# §14: OPP36 = canonical STRUCT33 (side-oriented) + exactly three structural
# features. No feature search, no symbol feature.
OPP36 = tuple(STRUCT33) + ("reward_distance_atr", "risk_distance_atr",
                           "log_structural_rr")
N_OPP36 = len(OPP36)                      # 36

# §15: these may exist ONLY in LABEL/AUDIT artifacts, never in OPP36.
FORBIDDEN_FEATURE_TOKENS = (
    "oracle_direction", "direction_correct", "oracle_entry_fill_time",
    "oracle_exit_fill_time", "event_class", "first_touch", "first_cross",
    "mfe", "mae", "future_return", "bars_to_event", "label_available_time",
    "episode_return", "win", "loss_magnitude", "win_magnitude",
)

COST_METADATA_STATUS = "UNAVAILABLE_COST_METADATA"
REALISTIC_NET_PNL_STATUS = "NOT_ESTIMATED"
FORMAL_PRIMARY_BASIS = "GROSS_PAIRED_POLICY_EFFECT"

COUNTERS = {
    "environment_loads": 0,
    "geometry_extract_calls": 0,
    "candidate_gate_derivations": 0,
    "feature_matrix_builds": 0,
    "side_feature_builds": 0,
    "structural_stream_scans": 0,
    "direction_reruns": 0,
    "candidate_python_loops": 0,
    "side_python_loops": 0,
    "horizon_path_rescans": 0,
    "full_history_recompute_count": 0,
}


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


def opp36_schema_sha256():
    return hashlib.sha256("\n".join(OPP36).encode()).hexdigest()


# --------------------------------------------------------------------------- #
# 1. State cache (§11 / §12)                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class OpportunityState:
    symbol: str
    n_bars: int

    bar_index: np.ndarray
    bar_start_time: np.ndarray
    decision_time: np.ndarray
    trading_day: np.ndarray
    segment: np.ndarray

    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray

    X33: np.ndarray

    atr: np.ndarray
    sup_top: np.ndarray
    sup_bottom: np.ndarray
    sup_strength: np.ndarray
    res_top: np.ndarray
    res_bottom: np.ndarray
    res_strength: np.ndarray

    candidate_at_decision: np.ndarray
    candidate_trigger_bits: np.ndarray


def shift_candidate_clock(candidate_any: np.ndarray, trigger_bits: np.ndarray):
    """§12: canonical R4 Candidate is defined on the NEXT bar.

    `derive_m15_candidate_gate` sets candidate_any[i] from the touch state of bar
    i-1, i.e. the causal decision is made at the CLOSE of bar i-1 and the
    Candidate fill occurs at Open_i. Therefore the decision-bar mask is the
    candidate mask shifted BACK by exactly one bar.

    Returns (candidate_at_decision, candidate_trigger_bits_at_decision).
    """
    n = len(candidate_any)
    at_decision = np.zeros(n, dtype=bool)
    at_decision[:-1] = np.asarray(candidate_any, bool)[1:]
    bits = np.zeros(n, dtype=np.uint8)
    bits[:-1] = np.asarray(trigger_bits, np.uint8)[1:]
    return at_decision, bits


def horizon_end_indices(trading_day, segment, n_bars, days):
    """End_H = min(TD_H, hard-segment end, data end), evaluated per bar (§7)."""
    day_arr = np.asarray(trading_day)
    seg_arr = np.asarray(segment, np.int64)
    uniq_days = pd.Index(day_arr).unique()
    day_ord = np.empty(n_bars, dtype=np.int64)
    ord_last = {}
    for o, d in enumerate(uniq_days):
        pos = np.flatnonzero(day_arr == d)
        day_ord[pos] = o
        ord_last[o] = int(pos[-1])
    max_ord = len(uniq_days) - 1

    seg_last = np.empty(n_bars, dtype=np.int64)
    for s in np.unique(seg_arr):
        pos = np.flatnonzero(seg_arr == s)
        seg_last[pos] = int(pos[-1])

    out = {}
    for k in days:
        tgt = np.clip(day_ord + (int(k) - 1), 0, max_ord)
        last = np.array([ord_last[int(o)] for o in tgt], dtype=np.int64)
        out[int(k)] = np.minimum(np.minimum(last, seg_last), n_bars - 1)
    return out


def load_symbol_state(symbol: str) -> OpportunityState:
    """ONE environment load, ONE geometry extraction, ONE gate derivation (§10)."""
    env = run_environment_m15(symbol, capture_provenance=False)
    _bump("environment_loads")

    frame = env["exec_frame"]
    feats = env["features"]
    geom = env["geom_by_decision"]
    touch_bits = env["touch_bits"]

    n = int(len(frame))
    close = frame["close"].to_numpy(np.float64)
    seg = frame["segment"].to_numpy(np.int64)
    day = frame["trading_day"].to_numpy(object)

    def col(name):
        return (feats[name].to_numpy(np.float64) if name in feats.columns
                else np.full(n, np.nan))

    sr_sup_ref = col("m15_sr_support_price")
    sr_res_ref = col("m15_sr_resistance_price")
    up_level_ref = col("m15_liq_up_level_price")
    dn_level_ref = col("m15_liq_down_level_price")

    zg = extract_zone_geometry(geom, n, close, sr_sup_ref, sr_res_ref,
                               up_level_ref, dn_level_ref)
    _bump("geometry_extract_calls")

    x33 = feats[list(STRUCT33)].to_numpy(np.float32)
    _bump("feature_matrix_builds")

    gate = derive_m15_candidate_gate(touch_bits, seg, day)
    _bump("candidate_gate_derivations")
    cand_at_dec, cand_bits = shift_candidate_clock(
        gate["candidate_any"], gate["candidate_trigger_bits"])

    return OpportunityState(
        symbol=symbol, n_bars=n,
        bar_index=frame["execution_bar_index"].to_numpy(np.int64),
        bar_start_time=pd.to_datetime(
            frame["bar_start_time"]).to_numpy("datetime64[ns]"),
        decision_time=pd.to_datetime(
            frame["decision_time"]).to_numpy("datetime64[ns]"),
        trading_day=day, segment=seg,
        open=frame["open"].to_numpy(np.float64),
        high=frame["high"].to_numpy(np.float64),
        low=frame["low"].to_numpy(np.float64),
        close=close, X33=x33,
        atr=col("m15_atr"),
        sup_top=np.asarray(zg["sup_top"], float),
        sup_bottom=np.asarray(zg["sup_bottom"], float),
        sup_strength=np.asarray(zg["sup_strength"], float),
        res_top=np.asarray(zg["res_top"], float),
        res_bottom=np.asarray(zg["res_bottom"], float),
        res_strength=np.asarray(zg["res_strength"], float),
        candidate_at_decision=cand_at_dec,
        candidate_trigger_bits=cand_bits)


# --------------------------------------------------------------------------- #
# 2. Barriers, side views and OPP36 (§3 / §4 / §13 / §14)                       #
# --------------------------------------------------------------------------- #
def structural_barriers(is_long, sup_top, res_bottom):
    """NEAR edges of the next canonical m15 SR zones (§3).

    LONG : favorable = resistance_bottom ; adverse = support_top
    SHORT: favorable = support_top       ; adverse = resistance_bottom
    """
    favorable = np.where(is_long, res_bottom, sup_top)
    adverse = np.where(is_long, sup_top, res_bottom)
    return favorable, adverse


def bracket_metrics(side, close, favorable, adverse, atr):
    """§4: G = d(F - C)/ATR ; L = d(C - A)/ATR ; eligible = G>0 and L>0."""
    g = side * (favorable - close) / atr
    l = side * (close - adverse) / atr
    finite = np.isfinite(g) & np.isfinite(l)
    eligible = finite & (g > 0.0) & (l > 0.0)
    return g, l, eligible


def orient_struct33_side(X33: np.ndarray, side_is_long: np.ndarray) -> np.ndarray:
    """§14: R8 wrapper over the frozen router-side orientation semantics."""
    return orient_struct33_router_side(X33, side_is_long)


def build_side_views(state: OpportunityState):
    """§13: every decision bar produces at most two counterfactual side views.

    Returns a dict of stacked (2N) arrays: LONG block then SHORT block.
    """
    n = state.n_bars
    is_long = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    X33 = np.concatenate([state.X33, state.X33], axis=0)
    close = np.concatenate([state.close, state.close])
    atr = np.concatenate([state.atr, state.atr])
    sup_top = np.concatenate([state.sup_top, state.sup_top])
    res_bottom = np.concatenate([state.res_bottom, state.res_bottom])
    favorable, adverse = structural_barriers(is_long, sup_top, res_bottom)
    side = np.where(is_long, 1.0, -1.0)
    g, l, eligible = bracket_metrics(side, close, favorable, adverse, atr)
    X36 = np.concatenate([
        orient_struct33_side(X33, is_long),
        np.stack([g, l, _log_rr(g, l)], axis=1).astype(np.float32)], axis=1)
    _bump("side_feature_builds")
    return {
        "is_long": is_long, "side": side, "close": close, "atr": atr,
        "favorable": favorable, "adverse": adverse,
        "G": g, "L": l, "eligible": eligible, "X36": X36,
        "decision_bar": np.concatenate([np.arange(n), np.arange(n)]),
    }


def _log_rr(g, l):
    """§4: log RR instead of clipping the raw ratio."""
    ok = np.isfinite(g) & np.isfinite(l) & (g > 0) & (l > 0)
    out = np.full(np.shape(g), np.nan, dtype=float)
    out[ok] = np.log(g[ok]) - np.log(l[ok])
    return out


# --------------------------------------------------------------------------- #
# 3. Vectorized structural first-event scan (§16)                               #
# --------------------------------------------------------------------------- #
def scan_first_structural_event(*, entry_idx, end_idx, side, favorable_boundary,
                                adverse_boundary, high, low, segment,
                                entry_segment, eligible):
    """ONE time-step loop, vectorized over ALL stacked side views."""
    entry_idx = np.asarray(entry_idx, np.int64)
    end_idx = np.asarray(end_idx, np.int64)
    side = np.asarray(side, np.float64)

    n = len(entry_idx)
    event_step = np.full(n, -1, np.int32)
    event_code = np.zeros(n, np.int8)

    active = np.asarray(eligible, bool).copy()
    n_bars = len(high)
    if n == 0 or not active.any():
        return event_step, event_code

    max_h = int(np.max(np.where(active, end_idx - entry_idx, 0)))

    for h in range(max_h + 1):
        if not active.any():
            break
        j = entry_idx + h
        valid = active & (j >= 0) & (j <= end_idx) & (j < n_bars)
        safe_j = np.where(valid, j, 0)
        valid &= (segment[safe_j] == entry_segment)

        fav = valid & np.where(side > 0,
                               high[safe_j] >= favorable_boundary,
                               low[safe_j] <= favorable_boundary)
        adv = valid & np.where(side > 0,
                               low[safe_j] <= adverse_boundary,
                               high[safe_j] >= adverse_boundary)

        first = active & (fav | adv)
        both = first & fav & adv
        fav_only = first & fav & ~adv
        adv_only = first & adv & ~fav

        event_step[first] = h
        event_code[fav_only] = EVENT_FAV
        event_code[adv_only] = EVENT_ADV
        event_code[both] = EVENT_BOTH

        active[first] = False
        exhausted = active & ((entry_idx + h) >= end_idx)
        active[exhausted] = False
    return event_step, event_code


# --------------------------------------------------------------------------- #
# 4. Episode label (§17 / §18)                                                  #
# --------------------------------------------------------------------------- #
def episode_label(*, event_step, event_code, entry_idx, end_idx_H, side,
                  entry_open, atr0, open_px, close_px, segment, entry_segment):
    event_step = np.asarray(event_step, np.int64)
    end_idx_H = np.asarray(end_idx_H, np.int64)
    n_bars = len(open_px)

    event_idx = np.where(event_step >= 0, entry_idx + event_step, -1)
    observed = (event_step >= 0) & (event_idx <= end_idx_H)

    next_idx = np.where(observed, event_idx + 1, -1)
    in_range = observed & (next_idx >= 0) & (next_idx < n_bars)
    safe_next = np.where(in_range, next_idx, 0)
    renewable = (in_range
                 & (segment[safe_next] == entry_segment)
                 & (next_idx <= end_idx_H))

    safe_end = np.clip(end_idx_H, 0, n_bars - 1)
    exit_price = np.where(renewable, open_px[safe_next], close_px[safe_end])

    y = side * (exit_price - entry_open) / atr0
    event_class = np.where(
        observed,
        np.where(event_code == EVENT_FAV, "FAVORABLE_FIRST",
                 np.where(event_code == EVENT_ADV, "ADVERSE_FIRST",
                          "BOTH_SAME_BAR")),
        "NONE")
    return {
        "event_step": event_step,
        "event_code": event_code,
        "event_observed": observed,
        "renewal_executable": renewable,
        "renewal_decision_idx": np.where(observed, event_idx, -1).astype(np.int64),
        "renewal_fill_idx": np.where(renewable, next_idx, -1).astype(np.int64),
        "episode_exit_price": exit_price,
        "episode_return_atr": y,
        "win": y > 0,
        "win_magnitude": np.maximum(y, 0.0),
        "loss_magnitude": np.maximum(-y, 0.0),
        "event_class": event_class,
    }


# --------------------------------------------------------------------------- #
# 5. Per-symbol materialization (§7 / §19 / §20)                                #
# --------------------------------------------------------------------------- #
def _epoch_weights(eligible_long, eligible_short):
    """§20: every decision epoch contributes total model-training weight 1."""
    both = eligible_long & eligible_short
    only_long = eligible_long & ~eligible_short
    only_short = eligible_short & ~eligible_long
    w_long = np.where(both, 0.5, np.where(only_long, 1.0, 0.0))
    w_short = np.where(both, 0.5, np.where(only_short, 1.0, 0.0))
    return w_long, w_short


def build_symbol_dataset(symbol: str, split=None, verbose: bool = False):
    st = load_symbol_state(symbol)
    n = st.n_bars
    views = build_side_views(st)

    entry_idx_all = np.concatenate([np.arange(n), np.arange(n)]) + 1
    has_entry = entry_idx_all < n
    entry_idx = np.where(has_entry, entry_idx_all, 0)
    entry_open = st.open[entry_idx]
    atr0 = st.atr[entry_idx]
    entry_segment = st.segment[entry_idx]
    eligible = views["eligible"] & has_entry

    # §7: ONE scan only, through the maximum (TD5) horizon. TD1/TD3 come from
    # censoring the same first-event result -- never a rescan (§17).
    ends = horizon_end_indices(st.trading_day, st.segment, n, (1, 3, 5))
    end_td5 = np.concatenate([ends[5], ends[5]])

    event_step, event_code = scan_first_structural_event(
        entry_idx=entry_idx, end_idx=end_td5, side=views["side"],
        favorable_boundary=views["favorable"], adverse_boundary=views["adverse"],
        high=st.high, low=st.low, segment=st.segment,
        entry_segment=entry_segment, eligible=eligible)
    _bump("structural_stream_scans")

    w_long, w_short = _epoch_weights(
        eligible[:n] & (np.arange(n) + 1 < n),
        eligible[n:] & (np.arange(n) + 1 < n))

    label_rows = []
    for H in HORIZONS:
        k = HORIZON_DAYS[H]
        end_H = np.concatenate([ends[k], ends[k]])
        lab = episode_label(
            event_step=event_step, event_code=event_code, entry_idx=entry_idx,
            end_idx_H=end_H, side=views["side"], entry_open=entry_open,
            atr0=atr0, open_px=st.open, close_px=st.close, segment=st.segment,
            entry_segment=entry_segment)
        lab["horizon"] = H
        lab["end_idx"] = end_H
        label_rows.append(lab)

    side_df = pd.DataFrame({
        "symbol": symbol,
        "decision_bar": views["decision_bar"],
        "side": np.where(views["is_long"], "LONG", "SHORT"),
        "candidate_at_decision": np.concatenate([st.candidate_at_decision,
                                                 st.candidate_at_decision]),
        "bracket_eligible": eligible,
        "G": views["G"], "L": views["L"]})
    # §21: side_features carries the OPP36 matrix ONCE (not once per horizon).
    # The three leading columns are JOIN KEYS only; the OPP36 schema SHA is
    # computed over the 36 model columns alone.
    side_feat = pd.DataFrame(views["X36"], columns=list(OPP36))
    side_feat.insert(0, "side", np.where(views["is_long"], "LONG", "SHORT"))
    side_feat.insert(0, "decision_bar", views["decision_bar"])
    side_feat.insert(0, "symbol", symbol)
    SIDE_KEY = ("symbol", "decision_bar", "side")

    state_df = pd.DataFrame({
        "symbol": symbol,
        "bar_index": st.bar_index,
        "bar_start_time": st.bar_start_time,
        "decision_time": st.decision_time,
        "trading_day": st.trading_day,
        "segment": st.segment,
        "open": st.open, "high": st.high, "low": st.low, "close": st.close,
        "atr": st.atr,
        "sup_top": st.sup_top, "sup_bottom": st.sup_bottom,
        "res_top": st.res_top, "res_bottom": st.res_bottom,
        "candidate_at_decision": st.candidate_at_decision,
        "candidate_trigger_bits": st.candidate_trigger_bits})

    label_df = _assemble_labels(symbol, st, views, label_rows, eligible,
                                entry_idx, w_long, w_short, split)
    if verbose:
        print(f"{symbol}: bars={n} labels={len(label_df)}", flush=True)
    return state_df, side_df, side_feat, label_df


def _assemble_labels(symbol, st, views, label_rows, eligible, entry_idx,
                     w_long, w_short, split):
    n = st.n_bars
    frames = []
    decision_bar = np.concatenate([np.arange(n), np.arange(n)])
    side_name = np.where(views["is_long"], "LONG", "SHORT")
    weight = np.concatenate([w_long, w_short])
    for lab in label_rows:
        H = lab["horizon"]
        frames.append(pd.DataFrame({
            "symbol": symbol,
            "decision_bar": decision_bar,
            "entry_bar": entry_idx,
            "side": side_name,
            "horizon": H,
            "decision_time": st.decision_time[decision_bar],
            "label_available_time": st.decision_time[np.clip(
                np.where(lab["renewal_executable"], lab["renewal_fill_idx"],
                         lab["end_idx"]), 0, n - 1)],
            "bracket_eligible": eligible,
            "G": views["G"], "L": views["L"],
            "log_structural_rr": _log_rr(views["G"], views["L"]),
            "event_step": lab["event_step"],
            "event_class": lab["event_class"],
            "event_observed": lab["event_observed"],
            "renewal_executable": lab["renewal_executable"],
            "renewal_decision_idx": lab["renewal_decision_idx"],
            "renewal_fill_idx": lab["renewal_fill_idx"],
            "episode_exit_price": lab["episode_exit_price"],
            "episode_return_atr": lab["episode_return_atr"],
            "win": lab["win"],
            "win_magnitude": lab["win_magnitude"],
            "loss_magnitude": lab["loss_magnitude"],
            "sample_weight": weight,
        }))
    out = pd.concat(frames, ignore_index=True)
    if split is not None:
        out = assign_split(out, split)
    return out


def assign_split(df: pd.DataFrame, split: dict) -> pd.DataFrame:
    """§19: split purity. A label is usable only if the episode RESOLVES inside
    its own split window."""
    cal = split["cal"]
    cuts = cal["cuts"] if "cuts" in cal else split["cuts"]
    t1 = np.datetime64(cuts[0], "ns")
    t2 = np.datetime64(cuts[1], "ns")
    end = np.datetime64(pd.Timestamp(cal["end"]).to_datetime64(), "ns")

    dt = df["decision_time"].to_numpy("datetime64[ns]")
    lav = df["label_available_time"].to_numpy("datetime64[ns]")

    stage = np.full(len(df), "", dtype=object)
    stage[(dt < t1) & (lav < t1)] = "train"
    stage[(dt >= t1) & (dt < t2) & (lav < t2)] = "val"
    stage[(dt >= t2) & (lav <= end)] = "test"
    df = df.copy()
    df["split"] = stage
    return df[stage != ""]


# --------------------------------------------------------------------------- #
# 6. Full 15-symbol materialization (§21 / §22)                                 #
# --------------------------------------------------------------------------- #
def materialize(symbols=SYMBOLS, split=None, verbose: bool = True,
                save: bool = True):
    t0 = time.time()
    reset_counters()
    if split is None:
        from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
            build_frozen_split)
        split = build_frozen_split()

    states, sides, side_feats, labels = [], [], [], []
    for sym in symbols:
        s_df, sd_df, sf_df, l_df = build_symbol_dataset(sym, split=split)
        states.append(s_df)
        sides.append(sd_df)
        side_feats.append(sf_df)
        labels.append(l_df)
    state_df = pd.concat(states, ignore_index=True)
    side_df = pd.concat(sides, ignore_index=True)
    side_feat_df = pd.concat(side_feats, ignore_index=True)
    label_df = pd.concat(labels, ignore_index=True)

    parts = {}
    for stage in ("train", "val", "test"):
        parts[stage] = label_df[label_df["split"] == stage].reset_index(drop=True)

    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    if save:
        state_df.to_parquet(STATE_PARQUET, index=False)
        side_feat_df.to_parquet(SIDE_FEATURES_PARQUET, index=False)
        for stage, df in parts.items():
            df.to_parquet(LABEL_PARQUETS[stage], index=False)

    manifest = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "generator_code_sha": _git_head_sha(),
        "symbols": list(symbols),
        "opp36_schema_sha256": opp36_schema_sha256(),
        "n_opp36": N_OPP36,
        "state_rows": int(len(state_df)),
        "side_rows": int(len(side_df)),
        "side_feature_rows": int(len(side_feat_df)),
        "label_rows_total": int(len(label_df)),
        "label_rows": {k: int(len(v)) for k, v in parts.items()},
        "per_horizon_counts": {H: int((label_df.horizon == H).sum())
                               for H in HORIZONS},
        "event_class_counts": {k: int(v) for k, v in
                               label_df["event_class"].value_counts().items()},
        "bracket_coverage": float(side_df["bracket_eligible"].mean()),
        "cost_governance": {
            "cost_metadata_status": COST_METADATA_STATUS,
            "realistic_net_pnl_status": REALISTIC_NET_PNL_STATUS,
            "formal_primary_basis": FORMAL_PRIMARY_BASIS},
        "performance": dict(COUNTERS),
        "runtime_sec": time.time() - t0,
        "artifact_sha256": {},
    }
    if save:
        manifest["artifact_sha256"] = {
            os.path.basename(STATE_PARQUET): sha256_file(STATE_PARQUET),
            os.path.basename(SIDE_FEATURES_PARQUET):
                sha256_file(SIDE_FEATURES_PARQUET)}
        for stage, p in LABEL_PARQUETS.items():
            manifest["artifact_sha256"][os.path.basename(p)] = sha256_file(p)
        with open(MANIFEST_JSON, "w") as f:
            json.dump(manifest, f, indent=2)
    if verbose:
        print(json.dumps({k: v for k, v in manifest.items()
                          if k != "artifact_sha256"}, indent=2, default=str))
    return {"state": state_df, "side": side_df, "side_features": side_feat_df,
            "labels": label_df, "parts": parts, "manifest": manifest,
            "split": split}


def _git_head_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# 7. Reference (candidate-by-candidate) kernel — T0/T1 only (§49)               #
# --------------------------------------------------------------------------- #
def reference_episode(*, entry_idx, end_idx, side, favorable, adverse,
                      high, low, open_px, close_px, segment, entry_segment):
    """Literal per-episode scan. Reference only; never used in production."""
    if not (np.isfinite(favorable) and np.isfinite(adverse)):
        return None
    step = -1
    code = EVENT_NONE
    for h in range(int(end_idx) - int(entry_idx) + 1):
        j = int(entry_idx) + h
        if j >= len(high):
            break
        if segment[j] != entry_segment:
            break
        if side > 0:
            fav = high[j] >= favorable
            adv = low[j] <= adverse
        else:
            fav = low[j] <= favorable
            adv = high[j] >= adverse
        if fav or adv:
            step = h
            code = EVENT_BOTH if (fav and adv) else (EVENT_FAV if fav
                                                     else EVENT_ADV)
            break
    if step < 0:
        return {"event_step": -1, "event_code": EVENT_NONE}
    return {"event_step": step, "event_code": code}


if __name__ == "__main__":
    materialize()
