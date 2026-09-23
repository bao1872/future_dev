"""entry_path_atlas_v1
======================

FUTURE-R5-M15-ENTRY-PATH-ATLAS-V1

PATH INFORMATION experiment (kernel).

After the FROZEN E9 Direction decision is made at a Candidate entry, do the
subsequently OBSERVABLE price path and the frozen-at-entry SR/Liquidity interaction
states stably separate

    E9 Direction ultimately CORRECT      vs      E9 Direction ultimately WRONG ?

This is NOT a stop-loss optimization, NOT a take-profit optimization and NOT a
Direction redesign. No thresholds, stop rules, TP rules, architecture, Direction
threshold/features, outer split, Teacher, Candidate gate or dataset definition may
be tuned here.

Causality contract
------------------
Everything derived from the future (oracle_direction, oracle_exit_fill_time,
entry_quality_atr, final MFE/MAE, event completion) is AUDIT_ONLY /
FORBIDDEN_REALTIME_FEATURE. It may be a label, audit field, scoring outcome or
stratification variable, but it must never enter a real-time feature or state
transition before it becomes observable. The observation horizon therefore stops at
the EARLIEST of (end of 5th trading day, hard-segment boundary, data end) and never
at oracle_exit_fill_time.

Kernels
-------
Reference  -- deliberately slow and obvious (per-candidate loop). T0/T1 only.
Production -- loop over TIME STEP, vectorized over all Candidates. Never a Python
              loop over Candidate rows. O(N*H*Z) time, O(N*Z) memory, streaming
              accumulators plus checkpoint snapshots only.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.train_direction_model_15sym_v1 import (
    SYMBOLS,
    verify_manifest,
)
from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
    build_frozen_split,
)
from research.liquidity_oracle_atlas.direction_gated_experts_v1 import (
    build_direction_expert_data,
    run_chain,
)

TASK_ID = "FUTURE-R5-M15-ENTRY-PATH-ATLAS-V1"
BASE_SHA = "cb9261d4498d328e6dd8edca5d655eaf609b0611"
STAGE = "kernel_checkpoint"

TF_ORDER = ("m15", "h1", "h4")

# Fixed bar-count checkpoints. Step index is 0-based: step s means (s+1) bars have
# been observed since the fill bar, so 1 bar = step 0, 4 bars = step 3, 16 bars = step 15.
BAR_CHECKPOINTS = ((0, "m15"), (3, "h1"), (15, "h4"))
BARS_IN_CHECKPOINT = {"m15": 1, "h1": 4, "h4": 16}
# Trading-day checkpoints: end of the Nth trading day (fill day counts as day 1).
TD_CHECKPOINTS = ((1, "td1"), (3, "td3"), (5, "td5"))
CHECKPOINT_NAMES = tuple(n for _, n in BAR_CHECKPOINTS) + tuple(n for _, n in TD_CHECKPOINTS)
PRIMARY_CHECKPOINT = "h4"          # 4h = 16 valid 15m bars
HORIZON_TRADING_DAYS = 5

# Frozen TEST population from the frozen upstream Direction experiment.
FROZEN_TEST = {"test_rows": 13773, "test_trades": 638,
               "long_trades": 319, "short_trades": 319}

ARTIFACT_DIR = os.path.join("artifacts", "entry_path_atlas_v1")
E9_STATE_PARQUET = os.path.join(ARTIFACT_DIR, "e9_direction_state_v1.parquet")
ANCHORS_PARQUET = os.path.join(ARTIFACT_DIR, "entry_path_anchors_v1.parquet")
ROW_METRICS_PARQUET = os.path.join(ARTIFACT_DIR, "entry_path_row_metrics_v1.parquet")
EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "entry_path_atlas_v1_manifest.json")

# Fields whose value is only knowable AFTER the fact. Never a real-time input.
AUDIT_ONLY_FIELDS = (
    "direction_correct",
    "oracle_direction",
    "oracle_entry_quality_atr",
    "e9_teacher_exit_return_atr",
    "oracle_exit_fill_time",
    "final_mfe",
    "final_mae",
)
FORBIDDEN_REALTIME_FEATURE = "FORBIDDEN_REALTIME_FEATURE"

SEMANTIC_KEY_FIELDS = ("symbol", "oracle_trade_id",
                       "candidate_decision_time", "candidate_fill_time")

# --------------------------------------------------------------------------- #
# Performance / structural counters (TP performance gate)                      #
# --------------------------------------------------------------------------- #
COUNTER_NAMES = (
    "raw_exec_load_count",
    "direction_chain_run_count",
    "atr_precompute_count",
    "sr_liq_precompute_count",
    "full_history_recompute_count",
    "reference_call_count_production",
    "candidate_python_loop_count",
    "hotloop_dataframe_concat_count",
    "path_step_count",
)
COUNTERS = {k: 0 for k in COUNTER_NAMES}


def bump(name: str, n: int = 1) -> None:
    if name not in COUNTERS:
        raise KeyError(f"unknown counter: {name}")
    COUNTERS[name] += int(n)


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0


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


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_semantic_key(df: pd.DataFrame) -> np.ndarray:
    parts = [df[c].astype(str) for c in SEMANTIC_KEY_FIELDS]
    out = parts[0]
    for p in parts[1:]:
        out = out + "::" + p
    return out.to_numpy(object)


# --------------------------------------------------------------------------- #
# 1. Materialize the frozen row-level E9 Direction state (ONCE, no LOSO)       #
# --------------------------------------------------------------------------- #
def materialize_e9_direction_state(save: bool = True, verbose: bool = True):
    """Run the frozen upstream pooled chain ONCE to obtain row-level E9 TEST state.

    Direction is frozen: this reuses run_chain verbatim and never retrains or alters
    it. LOSO is deliberately NOT run.
    """
    def log(*a):
        if verbose:
            print(*a, file=sys.stderr, flush=True)

    log("verify manifest (fail-closed) ...")
    verify_manifest(SYMBOLS)

    log("build frozen split ...")
    split = build_frozen_split()
    ds = split["ds"]
    train_idx, val_idx, test_idx = (split["train_idx"], split["val_idx"],
                                    split["test_idx"])

    data = build_direction_expert_data(ds)
    gid_all = data.gid[test_idx]
    n_rows = int(test_idx.size)
    n_trades = int(np.unique(gid_all).size)
    n_long = int(np.unique(gid_all[data.y[test_idx] == 1]).size)
    n_short = int(np.unique(gid_all[data.y[test_idx] == 0]).size)
    if (n_rows != FROZEN_TEST["test_rows"] or n_trades != FROZEN_TEST["test_trades"]
            or n_long != FROZEN_TEST["long_trades"]
            or n_short != FROZEN_TEST["short_trades"]):
        raise RuntimeError(
            f"STOP_PATH_ATLAS_POPULATION_DRIFT rows={n_rows} trades={n_trades} "
            f"long={n_long} short={n_short}")

    log("run frozen pooled chain ONCE (E9 row-level materialization) ...")
    chain = run_chain(data, ds, train_idx, val_idx, test_idx)
    bump("direction_chain_run_count")

    e9 = np.asarray(chain["E9"], dtype=np.uint8)
    router = np.asarray(chain["router_te"], dtype=np.int8)
    if e9.size != n_rows:
        raise RuntimeError("STOP_PATH_ATLAS_E9_SHAPE_MISMATCH")

    sub = ds.iloc[test_idx]
    df = pd.DataFrame({
        "semantic_key": build_semantic_key(sub),
        "symbol": data.symbol[test_idx],
        "oracle_trade_id": sub["oracle_trade_id"].to_numpy(object),
        "gid": gid_all,
        "candidate_decision_index": sub["candidate_decision_index"].to_numpy(np.int64),
        "candidate_decision_time": sub["candidate_decision_time"].to_numpy(object),
        "candidate_fill_index": sub["candidate_fill_index"].to_numpy(np.int64),
        "candidate_fill_time": sub["candidate_fill_time"].to_numpy(object),
        "candidate_fill_price": sub["candidate_fill_price"].to_numpy(np.float64),
        "sample_weight_raw": data.w[test_idx],
        # ---- frozen Direction state (decision-time, real-time legal) ----
        "e9_direction": np.where(e9 == 1, "LONG", "SHORT"),
        "e9_side": np.where(e9 == 1, 1.0, -1.0),
        "router_direction": np.where(router == 1, "LONG", "SHORT"),
        "router_p_long": np.asarray(chain["p_te"], dtype=np.float64),
        "e9_p_correct": np.asarray(chain["p_score"]["E9"], dtype=np.float64),
        # ---- AUDIT ONLY ----
        "oracle_direction": sub["oracle_direction"].to_numpy(object),
        "oracle_entry_quality_atr": sub["entry_quality_atr"].to_numpy(np.float64),
        "oracle_exit_fill_time": sub["oracle_exit_fill_time"].to_numpy(object),
    })
    df["direction_correct"] = (
        df["e9_direction"].to_numpy(object)
        == df["oracle_direction"].to_numpy(object)
    ).astype(np.uint8)
    df["e9_teacher_exit_return_atr"] = np.where(
        df["direction_correct"].to_numpy() == 1,
        df["oracle_entry_quality_atr"],
        -df["oracle_entry_quality_atr"],
    )

    if not df["semantic_key"].is_unique:
        raise RuntimeError("STOP_PATH_ATLAS_SEMANTIC_KEY_NOT_UNIQUE")

    if save:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        df.to_parquet(E9_STATE_PARQUET, index=False)
        log(f"e9 direction state -> {E9_STATE_PARQUET}")
    return df


# --------------------------------------------------------------------------- #
# 2. Canonical decision-time environment state (ATR0 + SR/Liquidity zones)     #
# --------------------------------------------------------------------------- #
@dataclass
class SymbolState:
    symbol: str
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    open: np.ndarray
    segment: np.ndarray
    trading_day: np.ndarray
    bar_start_time: np.ndarray
    execution_bar_index: np.ndarray
    atr0_col: np.ndarray                 # canonical m15_atr per bar
    sr_support_price: np.ndarray
    sr_resistance_price: np.ndarray
    liq_up_edge: np.ndarray              # near edge (zone bottom) of liq_up
    liq_down_edge: np.ndarray            # near edge (zone top) of liq_down
    n_bars: int
    env_complete: bool
    env_note: str = ""


def _nearest_edge_levels(geom_by_decision, n_bars, price_arr):
    """Extract liquidity near-edge prices from the canonical zone owner.

    Canonical distance semantics (forming_indicator_state_v1):
        liq_up_dist_atr   = (zone.bottom - close) / atr
        liq_down_dist_atr = (close - zone.top)    / atr
    so the TRADEABLE boundary is the zone's near edge (bottom for up, top for down),
    NOT the zone centre (which is what *_level_price reports). We select the nearest
    zone by |edge - price| and preserve the zone's exact top/bottom later.
    """
    up_edge = np.full(n_bars, np.nan, dtype=np.float64)
    dn_edge = np.full(n_bars, np.nan, dtype=np.float64)
    for i in range(n_bars):
        g = geom_by_decision[i]
        if not g or "m15" not in g:
            continue
        channels, liq_up, liq_down = g["m15"][0], g["m15"][1], g["m15"][2]
        px = price_arr[i]
        best = None
        for z in liq_up:
            e = float(z["bottom"])
            if not np.isfinite(e):
                continue
            d = abs(e - px)
            if best is None or d < best[0]:
                best = (d, e)
        if best is not None:
            up_edge[i] = best[1]
        best = None
        for z in liq_down:
            e = float(z["top"])
            if not np.isfinite(e):
                continue
            d = abs(e - px)
            if best is None or d < best[0]:
                best = (d, e)
        if best is not None:
            dn_edge[i] = best[1]
        del channels
    return up_edge, dn_edge


def load_symbol_state(symbol: str) -> SymbolState:
    """Load the m15 execution frame and canonical decision-time environment ONCE.

    run_environment_m15 is the canonical owner for exec frame, features (m15_atr,
    SR/Liquidity prices) and geom_by_decision (SR channels + liquidity zones).
    capture_provenance=False so the persisted cache is used rather than recomputed.
    """
    from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
        run_environment_m15,
    )
    env = run_environment_m15(symbol, capture_provenance=False)
    bump("raw_exec_load_count")
    bump("atr_precompute_count")
    bump("sr_liq_precompute_count")

    frame = env["exec_frame"]
    feats = env["features"]
    geom = env["geom_by_decision"]
    n_frame = len(frame)
    n_feat = len(feats)

    # fail-closed: a partial/smoke env cache would silently truncate candidate paths
    env_complete = (n_feat == n_frame) and (len(geom) == n_frame)
    note = ""
    if not env_complete:
        note = (f"env incomplete: features={n_feat} geom={len(geom)} frame={n_frame}; "
                "symbol excluded from real-data validation")

    high = frame["high"].to_numpy(np.float64)
    low = frame["low"].to_numpy(np.float64)
    close = frame["close"].to_numpy(np.float64)
    open_ = frame["open"].to_numpy(np.float64)
    segment = frame["segment"].to_numpy(np.int64)
    trading_day = frame["trading_day"].to_numpy(object)
    bar_start = frame["bar_start_time"].to_numpy(object)
    ebi = frame["execution_bar_index"].to_numpy(np.int64)

    def col(name):
        return (feats[name].to_numpy(np.float64) if name in feats.columns
                else np.full(n_frame, np.nan))

    atr0 = col("m15_atr")
    sr_sup = col("m15_sr_support_price")
    sr_res = col("m15_sr_resistance_price")
    up_edge, dn_edge = _nearest_edge_levels(geom, n_frame, close)
    # fall back to the canonical centre level only if no zone was resolvable
    up_edge = np.where(np.isnan(up_edge), col("m15_liq_up_level_price"), up_edge)
    dn_edge = np.where(np.isnan(dn_edge), col("m15_liq_down_level_price"), dn_edge)

    return SymbolState(symbol, high, low, close, open_, segment, trading_day,
                       bar_start, ebi, atr0, sr_sup, sr_res, up_edge, dn_edge,
                       n_frame, env_complete, note)


# --------------------------------------------------------------------------- #
# 3. Horizon construction (segment / trading day / data end)                   #
# --------------------------------------------------------------------------- #
def build_horizon_indices(state: SymbolState, fill_idx: np.ndarray):
    """Return (end_idx, td_end_idx dict) -- earliest of 5th-TD end, segment, data end.

    Never uses oracle_exit_fill_time.
    """
    n = state.n_bars
    seg = state.segment
    day = state.trading_day

    # last bar index of the segment containing each bar
    seg_last = np.empty(n, dtype=np.int64)
    for s in np.unique(seg):
        pos = np.flatnonzero(seg == s)
        seg_last[pos] = pos[-1]

    # trading-day ordinal -> last bar index with that ordinal
    uniq_days = pd.Index(day).unique()
    day_ord = np.empty(n, dtype=np.int64)
    ord_last = {}
    for o, d in enumerate(uniq_days):
        pos = np.flatnonzero(np.asarray(day) == d)
        day_ord[pos] = o
        ord_last[o] = pos[-1]
    max_ord = len(uniq_days) - 1

    def last_of(ord_arr):
        out = np.empty(len(ord_arr), dtype=np.int64)
        for i, o in enumerate(ord_arr):
            o = int(min(max(o, 0), max_ord))
            out[i] = ord_last[o]
        return out

    f_ord = day_ord[fill_idx]
    td_ends = {}
    for k, name in TD_CHECKPOINTS:
        td_ends[name] = last_of(f_ord + (k - 1))

    seg_end = seg_last[fill_idx]
    data_end = np.full(len(fill_idx), n - 1, dtype=np.int64)
    end = np.minimum(np.minimum(td_ends["td5"], seg_end), data_end)
    # trading-day checkpoints must also respect the segment/data truncation
    for name in td_ends:
        td_ends[name] = np.minimum(np.minimum(td_ends[name], seg_end), data_end)
    return end, td_ends


# --------------------------------------------------------------------------- #
# 4. Side-relative VIEW (raw identities preserved separately)                  #
# --------------------------------------------------------------------------- #
def side_view(e9_long: np.ndarray, sr_support, sr_resistance,
              liq_up, liq_down):
    """Raw identities (support/resistance/liq_up/liq_down) are NEVER collapsed.

    This only builds the side-relative projection used by the event state machine.
    E9 LONG : backstop=support, ahead=resistance, liq ahead=up,    behind=down
    E9 SHORT: backstop=resistance, ahead=support, liq ahead=down,  behind=up
    """
    is_long = np.asarray(e9_long, dtype=bool)
    backstop = np.where(is_long, sr_support, sr_resistance)
    ahead_sr = np.where(is_long, sr_resistance, sr_support)
    ahead_liq = np.where(is_long, liq_up, liq_down)
    behind_liq = np.where(is_long, liq_down, liq_up)
    return backstop, ahead_sr, ahead_liq, behind_liq


def build_anchors_for_symbol(e9_df: pd.DataFrame, state: SymbolState) -> dict:
    """L1 anchors: join frozen Candidates to canonical decision-time state.

    Runs the HARD fail-closed alignment gate (ruling 3): candidate_fill_index must be
    a valid positional index into the symbol frame with matching execution_bar_index,
    open price and bar_start_time. Any mismatch is STOP.

    All frozen-at-decision state (ATR0, SR, Liquidity) is taken at the DECISION index,
    matching the canonical struct33 convention `candidate_atr = m15_atr[decision_idx]`.
    """
    sel = e9_df[e9_df["symbol"] == state.symbol].reset_index(drop=True)
    if len(sel) == 0:
        return None
    fill = sel["candidate_fill_index"].to_numpy(np.int64)
    dec = sel["candidate_decision_index"].to_numpy(np.int64)
    if int(fill.max()) >= state.n_bars or int(dec.max()) >= state.n_bars:
        raise RuntimeError("STOP_PATH_ATLAS_FILL_INDEX_OUT_OF_RANGE")
    if not np.array_equal(state.execution_bar_index[fill], fill):
        raise RuntimeError("STOP_PATH_ATLAS_FILL_ALIGNMENT_MISMATCH:index")
    if not np.allclose(state.open[fill],
                       sel["candidate_fill_price"].to_numpy(np.float64), atol=1e-9):
        raise RuntimeError("STOP_PATH_ATLAS_FILL_ALIGNMENT_MISMATCH:price")
    ft = pd.to_datetime(sel["candidate_fill_time"]).to_numpy("datetime64[ns]")
    bt = pd.to_datetime(state.bar_start_time[fill]).to_numpy("datetime64[ns]")
    if not np.array_equal(ft, bt):
        raise RuntimeError("STOP_PATH_ATLAS_FILL_ALIGNMENT_MISMATCH:time")

    e9_long = (sel["e9_direction"].to_numpy(object) == "LONG")
    sr_sup = state.sr_support_price[dec]
    sr_res = state.sr_resistance_price[dec]
    liq_up = state.liq_up_edge[dec]
    liq_dn = state.liq_down_edge[dec]
    backstop, ahead_sr, ahead_liq, behind_liq = side_view(
        e9_long, sr_sup, sr_res, liq_up, liq_dn)
    end_idx, td_ends = build_horizon_indices(state, fill)
    return {
        "df": sel,
        "entry_idx": fill,
        "end_idx": end_idx,
        "entry_price": sel["candidate_fill_price"].to_numpy(np.float64),
        "atr0": state.atr0_col[dec],
        "side": np.where(e9_long, 1.0, -1.0),
        "entry_segment": state.segment[fill],
        "backstop_level": backstop,
        "ahead_sr_level": ahead_sr,
        "ahead_liq_level": ahead_liq,
        "behind_liq_level": behind_liq,
        "td_ends": td_ends,
        "raw_sr_support": sr_sup,
        "raw_sr_resistance": sr_res,
        "raw_liq_up": liq_up,
        "raw_liq_down": liq_dn,
    }


def run_symbol_paths(state: SymbolState, anchors: dict) -> dict:
    """Production path scan for one symbol (arrays already loaded once)."""
    return scan_paths_streaming(
        entry_idx=anchors["entry_idx"],
        end_idx=anchors["end_idx"],
        entry_price=anchors["entry_price"],
        atr0=anchors["atr0"],
        side=anchors["side"],
        high=state.high,
        low=state.low,
        close=state.close,
        segment=state.segment,
        entry_segment=anchors["entry_segment"],
        backstop_level=anchors["backstop_level"],
        ahead_sr_level=anchors["ahead_sr_level"],
        ahead_liq_level=anchors["ahead_liq_level"],
        td_ends=anchors["td_ends"],
    )


# --------------------------------------------------------------------------- #
# 5. Production kernel: loop over TIME, vectorize over Candidates              #
# --------------------------------------------------------------------------- #
def scan_paths_streaming(*, entry_idx, end_idx, entry_price, atr0, side,
                         high, low, close, segment, entry_segment,
                         backstop_level, ahead_sr_level, ahead_liq_level,
                         td_ends=None, counters=None):
    """Streaming path/event scan. No Python loop over Candidate rows.

    Returns MFE/MAE accumulators, event first-passage indices, checkpoint snapshots.
    Convention: not triggered = -1. A NaN zone means the event is UNAVAILABLE and is
    never fabricated (level comparisons against NaN are always False).
    """
    n = len(entry_idx)
    mfe = np.zeros(n, dtype=np.float64)
    mae = np.zeros(n, dtype=np.float64)

    first_touch = np.full(n, -1, dtype=np.int32)
    first_pierce = np.full(n, -1, dtype=np.int32)
    first_reclaim = np.full(n, -1, dtype=np.int32)
    first_failed_reclaim = np.full(n, -1, dtype=np.int32)
    first_ahead_sr = np.full(n, -1, dtype=np.int32)
    first_ahead_sr_cross = np.full(n, -1, dtype=np.int32)
    first_ahead_liq = np.full(n, -1, dtype=np.int32)
    first_ahead_liq_cross = np.full(n, -1, dtype=np.int32)

    same_bar_reclaim = np.zeros(n, dtype=bool)
    pierced = np.zeros(n, dtype=bool)
    reclaimed = np.zeros(n, dtype=bool)

    mfe_at_sr = np.full(n, np.nan, dtype=np.float64)
    mae_before_sr = np.full(n, np.nan, dtype=np.float64)

    snaps = {name: {"mfe": None, "mae": None, "r": None} for name in CHECKPOINT_NAMES}
    bar_steps = dict(BAR_CHECKPOINTS)

    n_bars = len(close)
    max_step = int(np.max(end_idx - entry_idx)) if n else 0
    bump("path_step_count", max_step + 1)

    for step in range(max_step + 1):
        j = entry_idx + step
        valid = (j <= end_idx) & (j < n_bars)
        safe_j = np.minimum(j, n_bars - 1)
        valid &= segment[safe_j] == entry_segment
        if not valid.any():
            continue

        hi = high[safe_j]
        lo = low[safe_j]
        cl = close[safe_j]

        fav = np.where(side > 0, hi - entry_price, entry_price - lo) / atr0
        adv = np.where(side > 0, entry_price - lo, hi - entry_price) / atr0
        mfe[valid] = np.maximum(mfe[valid], fav[valid])
        mae[valid] = np.maximum(mae[valid], adv[valid])

        B = backstop_level
        # touch: price reaches the backstop without piercing
        touch_now = valid & (first_touch < 0) & np.where(
            side > 0, lo <= B, hi >= B)
        first_touch[touch_now] = step

        pierce_now = valid & ~pierced & np.where(side > 0, lo < B, hi > B)
        first_pierce[pierce_now] = step
        pierced[pierce_now] = True

        valid_close = np.where(side > 0, cl >= B, cl <= B)

        sbr = pierce_now & valid_close
        same_bar_reclaim[sbr] = True
        first_reclaim[sbr] = step
        reclaimed[sbr] = True

        late = valid & pierced & ~reclaimed & valid_close
        first_reclaim[late] = step
        reclaimed[late] = True

        fail = valid & reclaimed & (first_failed_reclaim < 0) & (~valid_close)
        first_failed_reclaim[fail] = step

        # ahead SR / liquidity: touch = reach, cross = strictly through
        sr_touch = valid & (first_ahead_sr < 0) & np.where(
            side > 0, hi >= ahead_sr_level, lo <= ahead_sr_level)
        first_ahead_sr[sr_touch] = step
        mfe_at_sr[sr_touch] = mfe[sr_touch]
        mae_before_sr[sr_touch] = mae[sr_touch]

        sr_cross = valid & (first_ahead_sr_cross < 0) & np.where(
            side > 0, cl > ahead_sr_level, cl < ahead_sr_level)
        first_ahead_sr_cross[sr_cross] = step

        liq_touch = valid & (first_ahead_liq < 0) & np.where(
            side > 0, hi >= ahead_liq_level, lo <= ahead_liq_level)
        first_ahead_liq[liq_touch] = step

        liq_cross = valid & (first_ahead_liq_cross < 0) & np.where(
            side > 0, cl > ahead_liq_level, cl < ahead_liq_level)
        first_ahead_liq_cross[liq_cross] = step

        if step in bar_steps:
            name = bar_steps[step]
            if snaps[name]["mfe"] is None:
                snaps[name] = {"mfe": mfe.copy(), "mae": mae.copy(),
                               "r": (side * (cl - entry_price) / atr0).copy()}
        if td_ends is not None:
            for name, arr in td_ends.items():
                hit = valid & (j == arr)
                if hit.any() and snaps[name]["mfe"] is None:
                    snaps[name] = {"mfe": np.where(hit, mfe, np.nan),
                                   "mae": np.where(hit, mae, np.nan),
                                   "r": np.where(hit, side * (cl - entry_price) / atr0,
                                                 np.nan)}

    bars_to_reclaim = np.where(
        (first_pierce >= 0) & (first_reclaim >= 0),
        first_reclaim - first_pierce, -1).astype(np.int32)
    break_continue = (first_pierce >= 0) & (first_reclaim < 0)

    return {
        "mfe_final": mfe,
        "mae_final": mae,
        "first_touch": first_touch,
        "first_pierce": first_pierce,
        "first_reclaim": first_reclaim,
        "same_bar_reclaim": same_bar_reclaim,
        "bars_to_reclaim": bars_to_reclaim,
        "first_failed_reclaim": first_failed_reclaim,
        "break_continue": break_continue,
        "first_ahead_sr_touch": first_ahead_sr,
        "first_ahead_sr_cross": first_ahead_sr_cross,
        "first_ahead_liq_touch": first_ahead_liq,
        "first_ahead_liq_cross": first_ahead_liq_cross,
        "mfe_at_first_ahead_sr": mfe_at_sr,
        "mae_before_first_ahead_sr": mae_before_sr,
        "checkpoints": snaps,
    }


# --------------------------------------------------------------------------- #
# 6. Reference kernel (T0/T1 only -- never in the production call chain)       #
# --------------------------------------------------------------------------- #
def scan_paths_reference(*, entry_idx, end_idx, entry_price, atr0, side,
                         high, low, close, segment, entry_segment,
                         backstop_level, ahead_sr_level, ahead_liq_level,
                         td_ends=None):
    """Deliberately slow per-candidate reference. Allowed in T0/T1 only.

    Must never bump production counters: `candidate_python_loop_count` and
    `reference_call_count_production` are PRODUCTION invariants and stay 0.
    """
    n = len(entry_idx)
    out = {k: (np.zeros(n) if k in ("mfe_final", "mae_final")
               else (np.zeros(n, bool) if k == "same_bar_reclaim" or k == "break_continue"
                     else np.full(n, -1, dtype=np.int32)))
           for k in ("mfe_final", "mae_final", "first_touch", "first_pierce",
                     "first_reclaim", "same_bar_reclaim", "bars_to_reclaim",
                     "first_failed_reclaim", "break_continue",
                     "first_ahead_sr_touch", "first_ahead_sr_cross",
                     "first_ahead_liq_touch", "first_ahead_liq_cross")}
    mfe_at = np.full(n, np.nan)
    mae_before = np.full(n, np.nan)
    snaps = {name: {"mfe": np.full(n, np.nan), "mae": np.full(n, np.nan),
                    "r": np.full(n, np.nan)} for name in CHECKPOINT_NAMES}
    bar_steps = dict(BAR_CHECKPOINTS)

    for i in range(n):                      # per-candidate loop (reference only)
        a = int(entry_idx[i]); e = int(end_idx[i]); seg = entry_segment[i]
        p0 = float(entry_price[i]); atr = float(atr0[i]); s = float(side[i])
        B = float(backstop_level[i]); AS = float(ahead_sr_level[i])
        AL = float(ahead_liq_level[i])
        mfe = 0.0; mae = 0.0
        pierced = False; reclaimed = False
        for step in range(int(e - a) + 1):
            j = a + step
            if j > e or j >= len(close) or segment[j] != seg:
                continue
            hi = float(high[j]); lo = float(low[j]); cl = float(close[j])
            fav = ((hi - p0) if s > 0 else (p0 - lo)) / atr
            adv = ((p0 - lo) if s > 0 else (hi - p0)) / atr
            mfe = max(mfe, fav); mae = max(mae, adv)

            if out["first_touch"][i] < 0:
                if (s > 0 and lo <= B) or (s < 0 and hi >= B):
                    out["first_touch"][i] = step
            pierce_now = (not pierced) and ((s > 0 and lo < B) or (s < 0 and hi > B))
            if pierce_now:
                out["first_pierce"][i] = step
                pierced = True
            valid_close = (cl >= B) if s > 0 else (cl <= B)
            if pierce_now and valid_close:
                out["same_bar_reclaim"][i] = True
                out["first_reclaim"][i] = step
                reclaimed = True
            elif pierced and (not reclaimed) and valid_close:
                out["first_reclaim"][i] = step
                reclaimed = True
            if reclaimed and out["first_failed_reclaim"][i] < 0 and (not valid_close):
                out["first_failed_reclaim"][i] = step

            if out["first_ahead_sr_touch"][i] < 0 and (
                    (s > 0 and hi >= AS) or (s < 0 and lo <= AS)):
                out["first_ahead_sr_touch"][i] = step
                mfe_at[i] = mfe; mae_before[i] = mae
            if out["first_ahead_sr_cross"][i] < 0 and (
                    (s > 0 and cl > AS) or (s < 0 and cl < AS)):
                out["first_ahead_sr_cross"][i] = step
            if out["first_ahead_liq_touch"][i] < 0 and (
                    (s > 0 and hi >= AL) or (s < 0 and lo <= AL)):
                out["first_ahead_liq_touch"][i] = step
            if out["first_ahead_liq_cross"][i] < 0 and (
                    (s > 0 and cl > AL) or (s < 0 and cl < AL)):
                out["first_ahead_liq_cross"][i] = step

            if step in bar_steps:
                snaps[bar_steps[step]]["mfe"][i] = mfe
                snaps[bar_steps[step]]["mae"][i] = mae
                snaps[bar_steps[step]]["r"][i] = s * (cl - p0) / atr
            if td_ends is not None:
                for name, arr in td_ends.items():
                    if j == int(arr[i]):
                        snaps[name]["mfe"][i] = mfe
                        snaps[name]["mae"][i] = mae
                        snaps[name]["r"][i] = s * (cl - p0) / atr

        out["mfe_final"][i] = mfe
        out["mae_final"][i] = mae
        out["bars_to_reclaim"][i] = (
            (out["first_reclaim"][i] - out["first_pierce"][i])
            if (out["first_pierce"][i] >= 0 and out["first_reclaim"][i] >= 0) else -1)
        out["break_continue"][i] = (
            out["first_pierce"][i] >= 0 and out["first_reclaim"][i] < 0)

    out["mfe_at_first_ahead_sr"] = mfe_at
    out["mae_before_first_ahead_sr"] = mae_before
    out["checkpoints"] = snaps
    return out


# --------------------------------------------------------------------------- #
# 7. Statistics: whole-gid cluster bootstrap                                   #
# --------------------------------------------------------------------------- #
def cluster_bootstrap_gid(values, gid, w=None, B=2000, seed=20260924):
    """Resample WHOLE Oracle opportunities (gid), never rows.

    Within a trade the value is aggregated with the canonical sample_weight_raw; each
    opportunity then counts EXACTLY ONCE in the resample (canonical sample_weight_raw
    sums to 1 per opportunity, so opportunities are equally weighted). Candidate rows
    are never independent units.
    """
    values = np.asarray(values, dtype=np.float64)
    gid = np.asarray(gid, dtype=object)
    w = np.ones(len(values), dtype=np.float64) if w is None else np.asarray(w, np.float64)
    ug, inv = np.unique(gid, return_inverse=True)
    num = np.bincount(inv, weights=values * w, minlength=len(ug))
    den = np.bincount(inv, weights=w, minlength=len(ug))
    trade_mean = num / np.where(den > 0, den, 1.0)
    k = len(ug)
    if k == 0:
        return 0.0, 0.0, 0.0
    rng = np.random.default_rng(seed)
    means = np.empty(B, dtype=np.float64)
    for b in range(B):
        means[b] = trade_mean[rng.integers(0, k, size=k)].mean()
    return (float(trade_mean.mean()),
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def group_stats(values, gid, w):
    """Weighted mean / median / quantiles with canonical sample_weight_raw."""
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return {"n_rows": 0, "mean": None, "median": None}
    vv, ww = v[ok], w[ok]
    mean, lo, hi = cluster_bootstrap_gid(
        vv, np.asarray(gid, dtype=object)[ok], w=ww)
    order = np.argsort(vv)
    vs, ws = vv[order], ww[order]
    cw = np.cumsum(ws)
    cut = 0.5 * cw[-1]
    med = float(vs[int(np.searchsorted(cw, cut))])
    qs = {}
    for q in (10, 25, 50, 75, 90):
        c = (q / 100.0) * cw[-1]
        qs[f"p{q}"] = float(vs[min(int(np.searchsorted(cw, c)), len(vs) - 1)])
    return {"n_rows": int(ok.sum()), "mean": mean, "ci_low": lo, "ci_high": hi,
            "weighted_median": med, **qs}


# --------------------------------------------------------------------------- #
# 8. Differential harness (shared by T1 tests and the manifest generator)      #
# --------------------------------------------------------------------------- #
DIFF_FIELDS = ("mfe_final", "mae_final", "first_touch", "first_pierce",
               "first_reclaim", "same_bar_reclaim", "bars_to_reclaim",
               "first_failed_reclaim", "break_continue",
               "first_ahead_sr_touch", "first_ahead_sr_cross",
               "first_ahead_liq_touch", "first_ahead_liq_cross")


def diff_reference_vs_production(case: dict) -> dict:
    """Compare Reference vs Production field by field."""
    p = scan_paths_streaming(**case)
    r = scan_paths_reference(**case)
    rows = 0
    cells = 0
    mismatch = 0
    maxerr = 0.0
    first = None
    for f in DIFF_FIELDS:
        a = np.asarray(p[f])
        b = np.asarray(r[f])
        rows = len(a)
        cells += int(a.size)
        if a.dtype == bool or np.issubdtype(a.dtype, np.integer):
            d = a != b
            if d.any():
                mismatch += int(d.sum())
                if first is None:
                    first = (f, int(np.flatnonzero(d)[0]))
        else:
            e = float(np.nanmax(np.abs(a - b))) if a.size else 0.0
            maxerr = max(maxerr, e)
    return {"rows": rows, "cells": cells, "mismatch": mismatch,
            "max_abs_error": maxerr, "first_mismatch": first}


def make_synthetic_case(N: int, H: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_bars = H + 8
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n_bars))
    high = close + np.abs(rng.normal(0, 0.3, n_bars))
    low = close - np.abs(rng.normal(0, 0.3, n_bars))
    entry_idx = rng.integers(0, n_bars - H - 1, size=N)
    return dict(
        entry_idx=entry_idx.astype(np.int64),
        end_idx=(entry_idx + H).astype(np.int64),
        entry_price=close[entry_idx],
        atr0=np.full(N, 1.0),
        side=np.where(rng.random(N) < 0.5, 1.0, -1.0),
        high=high, low=low, close=close,
        segment=np.zeros(n_bars, dtype=np.int64),
        entry_segment=np.zeros(N, dtype=np.int64),
        backstop_level=close[entry_idx] - 1.0,
        ahead_sr_level=close[entry_idx] + 1.0,
        ahead_liq_level=close[entry_idx] + 2.0,
        td_ends=None,
    )


def tp_scaling_benchmark(base_n=200, horizon=32, seed=0) -> dict:
    def timed(N, H):
        c = make_synthetic_case(N, H, seed)
        t = time.perf_counter()
        scan_paths_streaming(**c)
        return time.perf_counter() - t

    t1 = timed(base_n, horizon)
    t2 = timed(base_n * 2, horizon)
    t4 = timed(base_n * 4, horizon)
    h1 = timed(base_n, horizon)
    h2 = timed(base_n, horizon * 2)
    safe = lambda x, y: (y / x) if x > 0 else None
    return {
        "candidate_scaling": {"N": base_n, "2N": base_n * 2, "4N": base_n * 4,
                              "H": horizon,
                              "t_N": t1, "t_2N": t2, "t_4N": t4,
                              "ratio_2N": safe(t1, t2), "ratio_4N": safe(t1, t4)},
        "horizon_scaling": {"H": horizon, "2H": horizon * 2, "N": base_n,
                            "t_H": h1, "t_2H": h2, "ratio_2H": safe(h1, h2)},
    }


# --------------------------------------------------------------------------- #
# 9. Kernel Checkpoint driver                                                  #
# --------------------------------------------------------------------------- #
def run_kernel_checkpoint(symbols=("AG",), n_subset=150, verbose=True) -> dict:
    """Run the authorized kernel gates and write the Kernel Checkpoint manifest.

    Deliberately NOT the full experiment: no T1.5, no T2, no statistics over the whole
    population, no primary-endpoint estimate.
    """
    def log(*a):
        if verbose:
            print(*a, file=sys.stderr, flush=True)

    t_start = time.time()
    reset_counters()
    log("materialize frozen E9 direction state (run_chain ONCE) ...")
    e9 = materialize_e9_direction_state(save=True, verbose=verbose)

    pop = {
        "test_rows": int(len(e9)),
        "test_trades": int(e9["gid"].nunique()),
        "long_trades": int(e9.loc[e9["oracle_direction"] == "LONG", "gid"].nunique()),
        "short_trades": int(e9.loc[e9["oracle_direction"] == "SHORT", "gid"].nunique()),
        "e9_predicted_long_rows": int((e9["e9_direction"] == "LONG").sum()),
        "e9_predicted_short_rows": int((e9["e9_direction"] == "SHORT").sum()),
        "direction_correct_rate": float(e9["direction_correct"].mean()),
    }

    blocks = {}
    for sym in symbols:
        log(f"symbol {sym}: load canonical state ...")
        st = load_symbol_state(sym)
        if not st.env_complete:
            blocks[sym] = {"env_complete": False, "note": st.env_note}
            continue
        anc = build_anchors_for_symbol(e9, st)
        n = len(anc["entry_idx"])
        full = dict(entry_idx=anc["entry_idx"], end_idx=anc["end_idx"],
                    entry_price=anc["entry_price"], atr0=anc["atr0"],
                    side=anc["side"], high=st.high, low=st.low, close=st.close,
                    segment=st.segment, entry_segment=anc["entry_segment"],
                    backstop_level=anc["backstop_level"],
                    ahead_sr_level=anc["ahead_sr_level"],
                    ahead_liq_level=anc["ahead_liq_level"],
                    td_ends=anc["td_ends"])
        t = time.perf_counter()
        scan_paths_streaming(**full)
        rt_full = time.perf_counter() - t
        sub = {k: (v[:n_subset] if isinstance(v, np.ndarray) else
                   {kk: vv[:n_subset] for kk, vv in v.items()})
               for k, v in full.items()}
        rep = diff_reference_vs_production(sub)
        blocks[sym] = {
            "env_complete": True,
            "n_candidates": n,
            "production_runtime_sec_full": rt_full,
            "t1": rep,
        }

    log("TP scaling benchmark ...")
    scaling = tp_scaling_benchmark()
    try:
        import resource as _res
        rss = float(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss)
    except Exception:
        rss = None

    artifacts = {}
    for p in (E9_STATE_PARQUET, ANCHORS_PARQUET, ROW_METRICS_PARQUET):
        if os.path.exists(p):
            artifacts[p] = _sha256_file(p)

    extra = {
        "branch": "entry-path-atlas-v1",
        "population": pop,
        "symbols_evaluated": list(symbols),
        "symbol_blocks": blocks,
        "tp": {"counters": dict(COUNTERS),
               "scaling": scaling,
               "peak_rss_bytes": rss},
        "artifacts": artifacts,
        "runtime_sec": time.time() - t_start,
        "unverified_items": [
            "Full 15-symbol canonical environment materialization DEFERRED to T1.5 "
            "(only symbols with a complete persisted R4 m15 env cache are evaluated).",
            "Primary endpoint delta_PS_4h is NOT estimated here (full experiment not authorized).",
            "L2 row metrics parquet is not produced at kernel stage.",
        ],
    }
    return write_manifest(extra=extra)


# --------------------------------------------------------------------------- #
# 10. Manifest                                                                 #
# --------------------------------------------------------------------------- #
def write_manifest(*, stage=STAGE, extra=None, path=MANIFEST_JSON):
    payload = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "stage": stage,
        "primary_endpoint": {
            "name": "delta_PS_4h",
            "definition": "E[PS_4h | direction_correct=1] - E[PS_4h | direction_correct=0]",
            "ps": "MFE - MAE (side-normalized, ATR0 units)",
            "checkpoint": PRIMARY_CHECKPOINT,
            "bars": 16,
            "inference": "whole-gid cluster bootstrap",
        },
        "audit_only_fields": list(AUDIT_ONLY_FIELDS),
        "realtime_policy": FORBIDDEN_REALTIME_FEATURE,
        "counters": dict(COUNTERS),
        "checkpoints": list(CHECKPOINT_NAMES),
        "horizon": {
            "trading_days": HORIZON_TRADING_DAYS,
            "stops_at": "earliest of 5th trading day end / hard segment / data end",
            "never_stops_at_oracle_exit": True,
        },
    }
    if extra:
        payload.update(extra)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_clean(payload), f, indent=2)
    return payload
