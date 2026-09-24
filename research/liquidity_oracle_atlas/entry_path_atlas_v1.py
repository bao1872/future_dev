"""entry_path_atlas_v1
======================

FUTURE-R5-M15-ENTRY-PATH-ATLAS-V1

PATH INFORMATION experiment (kernel).

After the FROZEN E9 Direction decision is made at a Candidate entry, do the
subsequently OBSERVABLE price path and the frozen-at-entry SR/Liquidity interaction
states stably separate

    E9 Direction ultimately CORRECT      vs      E9 Direction ultimately WRONG ?

This is NOT a stop-loss optimization, NOT a take-profit optimization and NOT a
Direction redesign.

ZONE SEMANTICS (RC1/RC2)
------------------------
Canonical SR owner stores (top, bottom, strength) and canonical Liquidity owner stores
(left, level, top, bottom, broken, breach_i). A zone is a BAND, not a point:

    LONG  backstop = support zone : touch/enter = low  <= support_top
                                    pierce-through     = low  <  support_bottom
                                    reclaim boundary   = close >= support_bottom
    SHORT backstop = resistance   : touch/enter = high >= resistance_bottom
                                    pierce-through     = high >  resistance_top
                                    reclaim boundary   = close <= resistance_top

A price merely ENTERING a zone is NEVER a pierce-through. Both structural backstops
(SR and Liquidity-behind) carry independent state machines (RC3).

Causality contract
------------------
Everything derived from the future is AUDIT_ONLY / FORBIDDEN_REALTIME_FEATURE. The
observation horizon stops at the EARLIEST of (end of 5th trading day, hard-segment
boundary, data end) and never at oracle_exit_fill_time.

Checkpoint contract (RC4)
-------------------------
Checkpoint arrays are pre-filled with NaN. A Candidate truncated before a checkpoint
has NO observation there (NaN), and never inherits a shorter-horizon accumulator.
Trading-day checkpoints fill per-Candidate at each Candidate's own day end.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import sys
import time
from dataclasses import dataclass

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
REVIEWED_SHA = "6376c63f973d8c5f6705ccaf421abc8740c7b0ef"
STAGE = "kernel_checkpoint"

TF_ORDER = ("m15", "h1", "h4")

# Step index is 0-based: step s means (s+1) bars observed since the fill bar.
BAR_CHECKPOINTS = ((0, "m15"), (3, "h1"), (15, "h4"))
BARS_IN_CHECKPOINT = {"m15": 1, "h1": 4, "h4": 16}
TD_CHECKPOINTS = ((1, "td1"), (3, "td3"), (5, "td5"))
CHECKPOINT_NAMES = (tuple(n for _, n in BAR_CHECKPOINTS)
                    + tuple(n for _, n in TD_CHECKPOINTS))
PRIMARY_CHECKPOINT = "h4"
HORIZON_TRADING_DAYS = 5

FROZEN_TEST = {"test_rows": 13773, "test_trades": 638,
               "long_trades": 319, "short_trades": 319}

ARTIFACT_DIR = os.path.join("artifacts", "entry_path_atlas_v1")
E9_STATE_PARQUET = os.path.join(ARTIFACT_DIR, "e9_direction_state_v1.parquet")
ANCHORS_PARQUET = os.path.join(ARTIFACT_DIR, "entry_path_anchors_v1.parquet")
ROW_METRICS_PARQUET = os.path.join(ARTIFACT_DIR, "entry_path_row_metrics_v1.parquet")
EVIDENCE_DIR = os.path.join("research", "liquidity_oracle_atlas", "evidence")
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "entry_path_atlas_v1_manifest.json")
R4_ENV_DIR = os.path.join("artifacts", "candidate_gate_r4_m15_touch_nextbar_v1")
R4_ENV_MANIFEST = os.path.join(R4_ENV_DIR, "r4_env_manifest.json")

AUDIT_ONLY_FIELDS = (
    "direction_correct", "e9_direction_correct", "a9_direction_correct",
    "oracle_direction", "oracle_entry_quality_atr",
    "e9_teacher_exit_return_atr", "a9_teacher_exit_return_atr", "oracle_exit_fill_time",
    "final_mfe", "final_mae",
)
FORBIDDEN_REALTIME_FEATURE = "FORBIDDEN_REALTIME_FEATURE"
SEMANTIC_KEY_FIELDS = ("symbol", "oracle_trade_id",
                       "candidate_decision_time", "candidate_fill_time")

COUNTER_NAMES = (
    "raw_exec_load_count", "direction_chain_run_count", "atr_precompute_count",
    "sr_liq_precompute_count", "full_history_recompute_count",
    "reference_call_count_production", "candidate_python_loop_count",
    "hotloop_dataframe_concat_count", "path_step_count", "path_scan_count",
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


def _git_head_sha() -> str:
    """Current git HEAD SHA (identity of the code that produced this manifest)."""
    try:
        import subprocess
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=os.path.dirname(__file__))
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "UNKNOWN"


def _arr_eq(a, b):
    """NaN-aware array equality: equal shapes, equal finite masks, equal finite values."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return False
    fa, fb = np.isfinite(a), np.isfinite(b)
    if not np.array_equal(fa, fb):
        return False
    return np.array_equal(a[fa], b[fb])


def build_semantic_key(df: pd.DataFrame) -> np.ndarray:
    parts = [df[c].astype(str) for c in SEMANTIC_KEY_FIELDS]
    out = parts[0]
    for p in parts[1:]:
        out = out + "::" + p
    return out.to_numpy(object)


# --------------------------------------------------------------------------- #
# 1. Frozen row-level E9 Direction state (materialized ONCE, no LOSO)          #
# --------------------------------------------------------------------------- #
def materialize_e9_direction_state(save: bool = True, verbose: bool = True):
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

    log("run frozen pooled chain ONCE ...")
    chain = run_chain(data, ds, train_idx, val_idx, test_idx)
    bump("direction_chain_run_count")
    e9 = np.asarray(chain["E9"], dtype=np.uint8)
    router = np.asarray(chain["router_te"], dtype=np.int8)
    if e9.size != n_rows:
        raise RuntimeError("STOP_PATH_ATLAS_E9_SHAPE_MISMATCH")

    # A9 := frozen direct DTP9 router (A). Naming only; the upstream chain already
    # freezes A == router_te. We assert it hard so a silent divergence fails closed.
    a9 = np.asarray(chain["A"], dtype=np.uint8)
    if a9.size != n_rows:
        raise RuntimeError("STOP_PATH_ATLAS_A9_SHAPE_MISMATCH")
    if not np.array_equal(a9, np.asarray(chain["router_te"], dtype=np.uint8)):
        raise RuntimeError(
            "STOP_PATH_ATLAS_A9_ROUTER_MISMATCH "
            "(frozen router A must equal router_te exactly)")
    a9_p_long = np.asarray(chain["A_p"], dtype=np.float64)
    # router_p_long is a DISTINCT probability artifact (chain['p_te']); it must NOT
    # be substituted for a9_p_long even though the hard labels agree.

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
        # A9 (frozen DTP9 router) direction-state artifact
        "a9_direction": np.where(a9 == 1, "LONG", "SHORT"),
        "a9_side": np.where(a9 == 1, 1.0, -1.0),
        "a9_p_long": a9_p_long,
        # E9 (gated experts) direction-state artifact
        "e9_direction": np.where(e9 == 1, "LONG", "SHORT"),
        "e9_side": np.where(e9 == 1, 1.0, -1.0),
        "router_direction": np.where(router == 1, "LONG", "SHORT"),
        "router_p_long": np.asarray(chain["p_te"], dtype=np.float64),
        "e9_p_correct": np.asarray(chain["p_score"]["E9"], dtype=np.float64),
        "oracle_direction": sub["oracle_direction"].to_numpy(object),
        "oracle_entry_quality_atr": sub["entry_quality_atr"].to_numpy(np.float64),
        "oracle_exit_fill_time": sub["oracle_exit_fill_time"].to_numpy(object),
    })
    df["e9_direction_correct"] = (
        df["e9_direction"].to_numpy(object)
        == df["oracle_direction"].to_numpy(object)).astype(np.uint8)
    df["a9_direction_correct"] = (
        df["a9_direction"].to_numpy(object)
        == df["oracle_direction"].to_numpy(object)).astype(np.uint8)
    df["direction_correct"] = df["e9_direction_correct"]  # legacy alias
    df["e9_teacher_exit_return_atr"] = np.where(
        df["e9_direction_correct"].to_numpy() == 1,
        df["oracle_entry_quality_atr"], -df["oracle_entry_quality_atr"])
    df["a9_teacher_exit_return_atr"] = np.where(
        df["a9_direction_correct"].to_numpy() == 1,
        df["oracle_entry_quality_atr"], -df["oracle_entry_quality_atr"])
    if not df["semantic_key"].is_unique:
        raise RuntimeError("STOP_PATH_ATLAS_SEMANTIC_KEY_NOT_UNIQUE")
    if save:
        os.makedirs(ARTIFACT_DIR, exist_ok=True)
        df.to_parquet(E9_STATE_PARQUET, index=False)
        log(f"e9 direction state -> {E9_STATE_PARQUET}")
    return df


# --------------------------------------------------------------------------- #
# 2. Canonical zone geometry (RC1 / RC2)                                       #
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
    atr0_col: np.ndarray
    # canonical SR zone geometry (selected per canonical scalar identity)
    sup_top: np.ndarray
    sup_bottom: np.ndarray
    sup_strength: np.ndarray
    res_top: np.ndarray
    res_bottom: np.ndarray
    res_strength: np.ndarray
    # canonical Liquidity zone geometry (canonical active-level selection)
    liq_up_top: np.ndarray
    liq_up_bottom: np.ndarray
    liq_up_level: np.ndarray
    liq_dn_top: np.ndarray
    liq_dn_bottom: np.ndarray
    liq_dn_level: np.ndarray
    # canonical reference features for differential assertions
    ref_sr_support: np.ndarray
    ref_sr_resistance: np.ndarray
    ref_liq_up_dist: np.ndarray
    ref_liq_dn_dist: np.ndarray
    n_bars: int
    env_complete: bool
    env_note: str = ""


def _max_dev(a, b):
    """Max absolute deviation over rows where BOTH arrays are finite.

    A row where one is finite and the other NaN is a genuine disagreement and
    returns inf (so the audit fails loudly instead of silently producing NaN).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    fin = np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        if (np.isfinite(a) != np.isfinite(b)).any():
            return float("inf")
        return 0.0
    if (np.isfinite(a) != np.isfinite(b)).any():
        return float("inf")
    return float(np.nanmax(np.abs(a[fin] - b[fin])))


def extract_zone_geometry(geom, n_bars, close, sr_sup_ref, sr_res_ref,
                          liq_up_level_ref, liq_dn_level_ref):
    """Resolve decision-time SR and Liquidity ZONES by REPLICATING the canonical
    SRState.step / LiquidityState.step selection over the canonical geom.

    This is mechanically identical to the canonical owner (same channels, same
    close, same rules) so the selected band is exactly the canonical channel:

      SR support    = max-top channel entirely below close (z[0] < c); else, when
                      price is inside a channel, the max-strength containing channel
                      and sr_support_price := close.
      SR resistance = min-bottom channel entirely above close (z[1] > c); else the
                      max-strength containing channel and sr_resistance_price := close.
      Liquidity up  = min-bottom UNBROKEN level with bottom > close.
      Liquidity down= max-top   UNBROKEN level with top   < close.

    The band (top, bottom, strength / level) of the selected zone is preserved
    verbatim -- never collapsed to a single price point.
    """
    sup_top = np.full(n_bars, np.nan)
    sup_bot = np.full(n_bars, np.nan)
    sup_str = np.full(n_bars, np.nan)
    res_top = np.full(n_bars, np.nan)
    res_bot = np.full(n_bars, np.nan)
    res_str = np.full(n_bars, np.nan)
    up_top = np.full(n_bars, np.nan)
    up_bot = np.full(n_bars, np.nan)
    up_lvl = np.full(n_bars, np.nan)
    dn_top = np.full(n_bars, np.nan)
    dn_bot = np.full(n_bars, np.nan)
    dn_lvl = np.full(n_bars, np.nan)
    # re-derived canonical scalar/level, used only for the audit
    recon_sup = np.full(n_bars, np.nan)
    recon_res = np.full(n_bars, np.nan)
    recon_up = np.full(n_bars, np.nan)
    recon_dn = np.full(n_bars, np.nan)

    for i in range(n_bars):
        g = geom[i]
        if not g or "m15" not in g:
            continue
        channels, liq_up, liq_down, _atr = g["m15"]
        c = float(close[i])

        # ---- SR: replicate canonical SRState.step selection ----
        containing = [z for z in channels if z[1] <= c <= z[0]]
        supports = [z for z in channels if z[0] < c]
        resistances = [z for z in channels if z[1] > c]

        sup_sel = None
        if supports:
            sup_sel = max(supports, key=lambda q: q[0])
            recon_sup[i] = float(sup_sel[0])
        elif containing:
            sup_sel = max(containing, key=lambda q: q[2])
            recon_sup[i] = c
        if sup_sel is not None:
            sup_top[i], sup_bot[i], sup_str[i] = (
                float(sup_sel[0]), float(sup_sel[1]), float(sup_sel[2]))

        res_sel = None
        if resistances:
            res_sel = min(resistances, key=lambda q: q[1])
            recon_res[i] = float(res_sel[1])
        elif containing:
            res_sel = max(containing, key=lambda q: q[2])
            recon_res[i] = c
        if res_sel is not None:
            res_top[i], res_bot[i], res_str[i] = (
                float(res_sel[0]), float(res_sel[1]), float(res_sel[2]))

        # ---- Liquidity: replicate canonical LiquidityState.step selection ----
        active_up = [z for z in liq_up
                     if (not z.get("broken")) and float(z["bottom"]) > c]
        if active_up:
            u = min(active_up, key=lambda q: float(q["bottom"]))
            up_top[i], up_bot[i], up_lvl[i] = (
                float(u["top"]), float(u["bottom"]), float(u["level"]))
            recon_up[i] = float(u["level"])
        active_down = [z for z in liq_down
                       if (not z.get("broken")) and float(z["top"]) < c]
        if active_down:
            d = max(active_down, key=lambda q: float(q["top"]))
            dn_top[i], dn_bot[i], dn_lvl[i] = (
                float(d["top"]), float(d["bottom"]), float(d["level"]))
            recon_dn[i] = float(d["level"])

    audit = {
        "sr_support_price_max_dev": _max_dev(recon_sup, sr_sup_ref),
        "sr_resistance_price_max_dev": _max_dev(recon_res, sr_res_ref),
        "liq_up_level_max_dev": _max_dev(recon_up, liq_up_level_ref),
        "liq_dn_level_max_dev": _max_dev(recon_dn, liq_dn_level_ref),
    }
    return dict(sup_top=sup_top, sup_bottom=sup_bot, sup_strength=sup_str,
                res_top=res_top, res_bottom=res_bot, res_strength=res_str,
                liq_up_top=up_top, liq_up_bottom=up_bot, liq_up_level=up_lvl,
                liq_dn_top=dn_top, liq_dn_bottom=dn_bot, liq_dn_level=dn_lvl,
                audit=audit)


def _normalize_max_bars(mb):
    """The R4 env manifest stores max_bars as the string 'None' for a full run."""
    if mb is None:
        return None
    if isinstance(mb, str):
        return None if mb == "None" else int(mb)
    return int(mb)


def load_env_provenance(symbol):
    """Record the R4 environment cache provenance (RC9).

    The canonical R4 owner stores ``environment_contract_id`` and
    ``cache_schema_version`` (NOT ``contract_id`` / ``cache_version``). Fail closed
    if any required key is missing, and reject smoke caches (max_bars != None).
    """
    out = {"symbol": symbol, "max_bars": None}
    try:
        with open(R4_ENV_MANIFEST) as f:
            man = json.load(f)
        ent = man.get(symbol) or (man.get("symbols") or {}).get(symbol)
        if not isinstance(ent, dict):
            raise RuntimeError(
                f"STOP_PATH_ATLAS_ENV_PROVENANCE_MISSING symbol={symbol}")
        required = ("environment_contract_id", "cache_schema_version", "identity",
                    "code_identity", "raw_sha256", "execution_frame_sha256",
                    "max_bars")
        missing = [k for k in required if k not in ent]
        if missing:
            raise RuntimeError(
                f"STOP_PATH_ATLAS_ENV_PROVENANCE_MISSING_KEYS symbol={symbol} "
                f"keys={missing}")
        mb = _normalize_max_bars(ent["max_bars"])
        if mb is not None:
            # smoke / partial cache is not acceptable for formal use
            raise RuntimeError(
                f"STOP_PATH_ATLAS_ENV_SMOKE_CACHE symbol={symbol} max_bars={mb}")
        out.update({k: ent.get(k) for k in required})
        out["max_bars"] = None
        out["sha256"] = ent.get("sha256")
        out["rows"] = ent.get("rows")
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(f"STOP_PATH_ATLAS_ENV_PROVENANCE_ERROR {exc}")
    return out


def load_symbol_state(symbol: str) -> SymbolState:
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
    env_complete = (len(feats) == n_frame) and (len(geom) == n_frame)
    note = ""
    if not env_complete:
        note = (f"env incomplete: features={len(feats)} geom={len(geom)} "
                f"frame={n_frame}; symbol excluded from real-data validation")

    def col(name):
        return (feats[name].to_numpy(np.float64) if name in feats.columns
                else np.full(n_frame, np.nan))

    close = frame["close"].to_numpy(np.float64)
    atr = col("m15_atr")
    sr_sup_ref = col("m15_sr_support_price")
    sr_res_ref = col("m15_sr_resistance_price")
    up_level_ref = col("m15_liq_up_level_price")
    dn_level_ref = col("m15_liq_down_level_price")
    zg = extract_zone_geometry(geom, n_frame, close, sr_sup_ref, sr_res_ref,
                               up_level_ref, dn_level_ref)

    st = SymbolState(
        symbol, frame["high"].to_numpy(np.float64), frame["low"].to_numpy(np.float64),
        close, frame["open"].to_numpy(np.float64), frame["segment"].to_numpy(np.int64),
        frame["trading_day"].to_numpy(object),
        frame["bar_start_time"].to_numpy(object),
        frame["execution_bar_index"].to_numpy(np.int64),
        atr,
        zg["sup_top"], zg["sup_bottom"], zg["sup_strength"],
        zg["res_top"], zg["res_bottom"], zg["res_strength"],
        zg["liq_up_top"], zg["liq_up_bottom"], zg["liq_up_level"],
        zg["liq_dn_top"], zg["liq_dn_bottom"], zg["liq_dn_level"],
        sr_sup_ref, sr_res_ref, up_level_ref, dn_level_ref,
        n_frame, env_complete, note)
    st.env_provenance = load_env_provenance(symbol)
    st.zone_audit = zg["audit"]
    return st


# --------------------------------------------------------------------------- #
# 3. Horizon                                                                   #
# --------------------------------------------------------------------------- #
def build_horizon_indices(state: SymbolState, fill_idx: np.ndarray):
    n = state.n_bars
    seg = state.segment
    day = state.trading_day

    seg_last = np.empty(n, dtype=np.int64)
    for s in np.unique(seg):
        pos = np.flatnonzero(seg == s)
        seg_last[pos] = pos[-1]

    day_arr = np.asarray(day)
    uniq_days = pd.Index(day).unique()
    day_ord = np.empty(n, dtype=np.int64)
    ord_last = {}
    for o, d in enumerate(uniq_days):
        pos = np.flatnonzero(day_arr == d)
        day_ord[pos] = o
        ord_last[o] = int(pos[-1])
    max_ord = len(uniq_days) - 1

    def last_of(ord_arr):
        out = np.empty(len(ord_arr), dtype=np.int64)
        for i, o in enumerate(ord_arr):
            o = int(min(max(o, 0), max_ord))
            out[i] = ord_last[o]
        return out

    f_ord = day_ord[fill_idx]
    td_ends = {name: last_of(f_ord + (k - 1)) for k, name in TD_CHECKPOINTS}
    seg_end = seg_last[fill_idx]
    data_end = np.full(len(fill_idx), n - 1, dtype=np.int64)
    end = np.minimum(np.minimum(td_ends["td5"], seg_end), data_end)
    for name in td_ends:
        td_ends[name] = np.minimum(np.minimum(td_ends[name], seg_end), data_end)
    return end, td_ends


# --------------------------------------------------------------------------- #
# 4. Side-relative boundaries (RC1 / RC3)                                      #
# --------------------------------------------------------------------------- #
def side_boundaries(e9_long, sup_top, sup_bottom, res_top, res_bottom,
                    liq_up_top, liq_up_bottom, liq_dn_top, liq_dn_bottom):
    """Return the boundary triple for each structural backstop plus ahead channels.

    LONG : SR backstop = support      ; liquidity-behind = LOWER liquidity
    SHORT: SR backstop = resistance   ; liquidity-behind = UPPER liquidity
    """
    is_long = np.asarray(e9_long, dtype=bool)
    w = lambda a, b: np.where(is_long, a, b)
    return {
        "sr_enter": w(sup_top, res_bottom),
        "sr_pierce": w(sup_bottom, res_top),
        "sr_reclaim": w(sup_bottom, res_top),
        "lb_enter": w(liq_dn_top, liq_up_bottom),
        "lb_pierce": w(liq_dn_bottom, liq_up_top),
        "lb_reclaim": w(liq_dn_bottom, liq_up_top),
        "ahead_sr_touch": w(res_bottom, sup_top),
        "ahead_sr_cross": w(res_top, sup_bottom),
        "ahead_liq_touch": w(liq_up_bottom, liq_dn_top),
        "ahead_liq_cross": w(liq_up_top, liq_dn_bottom),
    }


# --------------------------------------------------------------------------- #
# 5. Alignment gates (RC6)                                                     #
# --------------------------------------------------------------------------- #
def build_base_anchors(e9_df_symbol: pd.DataFrame, state: SymbolState) -> dict:
    """Direction-INVARIANT anchor base (RC D).

    Contains the positional + alignment identity, entry geometry, horizon indices,
    raw canonical zone identities and BOTH direction systems' labels. It does NOT
    contain any side-relative boundary. One base is built ONCE per symbol; the
    canonical environment / zone geometry is consumed exactly once.
    """
    sel = e9_df_symbol[e9_df_symbol["symbol"] == state.symbol].reset_index(drop=True)
    if len(sel) == 0:
        return None
    fill = sel["candidate_fill_index"].to_numpy(np.int64)
    dec = sel["candidate_decision_index"].to_numpy(np.int64)
    if int(fill.max()) >= state.n_bars or int(dec.max()) >= state.n_bars:
        raise RuntimeError("STOP_PATH_ATLAS_FILL_INDEX_OUT_OF_RANGE")

    # (a) positional identity
    if not np.array_equal(state.execution_bar_index[fill], fill):
        raise RuntimeError("STOP_ENTRY_PATH_FILL_ALIGNMENT_MISMATCH")
    # (b) open price at the fill bar
    if not np.allclose(state.open[fill],
                       sel["candidate_fill_price"].to_numpy(np.float64), atol=1e-9):
        raise RuntimeError("STOP_ENTRY_PATH_FILL_ALIGNMENT_MISMATCH")
    # (c) INDEPENDENT timestamp -> row lookup (does not reuse fill_index)
    bar_ns = pd.to_datetime(state.bar_start_time).to_numpy("datetime64[ns]")
    lookup = {}
    for i, ts in enumerate(bar_ns):
        lookup[int(ts.astype("int64"))] = i
    fill_ns = pd.to_datetime(sel["candidate_fill_time"]).to_numpy("datetime64[ns]")
    resolved = np.array([lookup.get(int(t.astype("int64")), -1) for t in fill_ns],
                        dtype=np.int64)
    if not np.array_equal(resolved, fill):
        raise RuntimeError("STOP_ENTRY_PATH_FILL_ALIGNMENT_MISMATCH")
    # (d) decision-time environment alignment
    if not np.array_equal(fill, dec + 1):
        raise RuntimeError("STOP_ENTRY_PATH_FILL_ALIGNMENT_MISMATCH")
    dec_ns = pd.to_datetime(sel["candidate_decision_time"]).to_numpy("datetime64[ns]")
    expect_dec = bar_ns[dec] + np.timedelta64(15, "m")
    if not np.array_equal(dec_ns.astype("int64"), expect_dec.astype("int64")):
        raise RuntimeError("STOP_ENTRY_PATH_DECISION_ALIGNMENT_MISMATCH")

    end_idx, td_ends = build_horizon_indices(state, fill)
    return {
        "df": sel,
        "entry_idx": fill,
        "dec_idx": dec,
        "end_idx": end_idx,
        "entry_price": sel["candidate_fill_price"].to_numpy(np.float64),
        "atr0": state.atr0_col[dec],
        "entry_segment": state.segment[fill],
        "td_ends": td_ends,
        # both direction systems' labels (direction-invariant storage)
        "a9_direction": sel["a9_direction"].to_numpy(object),
        "a9_side": sel["a9_side"].to_numpy(np.float64),
        "a9_direction_correct": sel["a9_direction_correct"].to_numpy(np.uint8),
        "e9_direction": sel["e9_direction"].to_numpy(object),
        "e9_side": sel["e9_side"].to_numpy(np.float64),
        "e9_direction_correct": sel["e9_direction_correct"].to_numpy(np.uint8),
        "oracle_direction": sel["oracle_direction"].to_numpy(object),
        "semantic_key": sel["semantic_key"].to_numpy(object),
        "gid": sel["gid"].to_numpy(object),
        # raw canonical identities preserved (never collapsed)
        "raw_sup_top": state.sup_top[dec], "raw_sup_bottom": state.sup_bottom[dec],
        "raw_res_top": state.res_top[dec], "raw_res_bottom": state.res_bottom[dec],
        "raw_liq_up_top": state.liq_up_top[dec],
        "raw_liq_up_bottom": state.liq_up_bottom[dec],
        "raw_liq_dn_top": state.liq_dn_top[dec],
        "raw_liq_dn_bottom": state.liq_dn_bottom[dec],
    }


def make_direction_view(system: str, base: dict, state: SymbolState) -> dict:
    """Mechanically produce the side-relative boundary triple + SR/Liq backstops +
    ahead channels for ONE direction system (RC D).

    A9 uses the frozen DTP9 router thesis; E9 uses the gated experts. The canonical
    environment / zone geometry is NEVER recomputed here.
    """
    if system not in ("A9", "E9"):
        raise ValueError(f"unknown direction system: {system}")
    if system == "A9":
        is_long = base["a9_direction"] == "LONG"
    else:
        is_long = base["e9_direction"] == "LONG"
    dec = base["dec_idx"]
    b = side_boundaries(
        is_long,
        state.sup_top[dec], state.sup_bottom[dec],
        state.res_top[dec], state.res_bottom[dec],
        state.liq_up_top[dec], state.liq_up_bottom[dec],
        state.liq_dn_top[dec], state.liq_dn_bottom[dec])
    return {
        **base,
        "system": system,
        "side": np.where(is_long, 1.0, -1.0),
        **b,
    }


def build_anchors_for_symbol(e9_df: pd.DataFrame, state: SymbolState) -> dict:
    """Backward-compatible E9-view anchor builder (used by RC1-RC8 tests)."""
    base = build_base_anchors(e9_df, state)
    if base is None:
        return None
    return make_direction_view("E9", base, state)


# --------------------------------------------------------------------------- #
# 6. Production kernel                                                         #
# --------------------------------------------------------------------------- #
def _new_backstop_state(n):
    return dict(first_touch=np.full(n, -1, np.int32),
                first_pierce=np.full(n, -1, np.int32),
                first_reclaim=np.full(n, -1, np.int32),
                first_failed=np.full(n, -1, np.int32),
                same_bar=np.zeros(n, bool),
                pierced=np.zeros(n, bool),
                reclaimed=np.zeros(n, bool))


def _update_backstop(st, valid, step, side, lo, hi, cl,
                     enter_b, pierce_b, reclaim_b):
    # touch/ENTER the zone
    t = valid & (st["first_touch"] < 0) & np.where(side > 0, lo <= enter_b, hi >= enter_b)
    st["first_touch"][t] = step
    # PIERCE THROUGH the zone (never the same thing as entering it)
    p = valid & ~st["pierced"] & np.where(side > 0, lo < pierce_b, hi > pierce_b)
    st["first_pierce"][p] = step
    st["pierced"][p] = True
    valid_close = np.where(side > 0, cl >= reclaim_b, cl <= reclaim_b)
    sbr = p & valid_close
    st["same_bar"][sbr] = True
    st["first_reclaim"][sbr] = step
    st["reclaimed"][sbr] = True
    late = valid & st["pierced"] & ~st["reclaimed"] & valid_close
    st["first_reclaim"][late] = step
    st["reclaimed"][late] = True
    f = valid & st["reclaimed"] & (st["first_failed"] < 0) & (~valid_close)
    st["first_failed"][f] = step


def _finalize_backstop(st):
    return {
        "first_touch": st["first_touch"],
        "first_pierce": st["first_pierce"],
        "same_bar_reclaim": st["same_bar"],
        "first_reclaim": st["first_reclaim"],
        "bars_to_reclaim": np.where(
            (st["first_pierce"] >= 0) & (st["first_reclaim"] >= 0),
            st["first_reclaim"] - st["first_pierce"], -1).astype(np.int32),
        "first_failed_reclaim": st["first_failed"],
        "break_continue": (st["first_pierce"] >= 0) & (st["first_reclaim"] < 0),
    }


def scan_paths_streaming(*, entry_idx, end_idx, entry_price, atr0, side,
                         high, low, close, segment, entry_segment,
                         sr_enter, sr_pierce, sr_reclaim,
                         lb_enter, lb_pierce, lb_reclaim,
                         ahead_sr_touch, ahead_sr_cross,
                         ahead_liq_touch, ahead_liq_cross,
                         td_ends=None):
    """Production: loop over TIME, vectorized over Candidates. No Candidate loop."""
    bump("path_scan_count")
    n = len(entry_idx)
    mfe = np.zeros(n, dtype=np.float64)
    mae = np.zeros(n, dtype=np.float64)
    sr = _new_backstop_state(n)
    lb = _new_backstop_state(n)

    a_sr_touch = np.full(n, -1, np.int32)
    a_sr_cross = np.full(n, -1, np.int32)
    a_liq_touch = np.full(n, -1, np.int32)
    a_liq_cross = np.full(n, -1, np.int32)
    mfe_at_sr = np.full(n, np.nan)
    mae_before_sr = np.full(n, np.nan)
    mfe_at_liq = np.full(n, np.nan)
    mae_before_liq = np.full(n, np.nan)

    # RC4: NaN-filled; a truncated Candidate stays NaN (never inherits a shorter path)
    snaps = {name: {"mfe": np.full(n, np.nan), "mae": np.full(n, np.nan),
                    "r": np.full(n, np.nan)} for name in CHECKPOINT_NAMES}
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

        _update_backstop(sr, valid, step, side, lo, hi, cl,
                         sr_enter, sr_pierce, sr_reclaim)
        _update_backstop(lb, valid, step, side, lo, hi, cl,
                         lb_enter, lb_pierce, lb_reclaim)

        t1 = valid & (a_sr_touch < 0) & np.where(
            side > 0, hi >= ahead_sr_touch, lo <= ahead_sr_touch)
        a_sr_touch[t1] = step
        mfe_at_sr[t1] = mfe[t1]
        mae_before_sr[t1] = mae[t1]
        c1 = valid & (a_sr_cross < 0) & np.where(
            side > 0, cl > ahead_sr_cross, cl < ahead_sr_cross)
        a_sr_cross[c1] = step

        t2 = valid & (a_liq_touch < 0) & np.where(
            side > 0, hi >= ahead_liq_touch, lo <= ahead_liq_touch)
        a_liq_touch[t2] = step
        mfe_at_liq[t2] = mfe[t2]
        mae_before_liq[t2] = mae[t2]
        c2 = valid & (a_liq_cross < 0) & np.where(
            side > 0, cl > ahead_liq_cross, cl < ahead_liq_cross)
        a_liq_cross[c2] = step

        if step in bar_steps:
            name = bar_steps[step]
            snaps[name]["mfe"][valid] = mfe[valid]
            snaps[name]["mae"][valid] = mae[valid]
            snaps[name]["r"][valid] = (side * (cl - entry_price) / atr0)[valid]
        if td_ends is not None:
            for name, arr in td_ends.items():
                hit = valid & (j == arr)
                if hit.any():
                    snaps[name]["mfe"][hit] = mfe[hit]
                    snaps[name]["mae"][hit] = mae[hit]
                    snaps[name]["r"][hit] = (side * (cl - entry_price) / atr0)[hit]

    out = {"mfe_final": mfe, "mae_final": mae}
    for pfx, st in (("sr", sr), ("lb", lb)):
        for k, v in _finalize_backstop(st).items():
            out[f"{pfx}_{k}"] = v
    out.update({
        "first_ahead_sr_touch": a_sr_touch, "first_ahead_sr_cross": a_sr_cross,
        "first_ahead_liq_touch": a_liq_touch, "first_ahead_liq_cross": a_liq_cross,
        "mfe_at_first_ahead_sr": mfe_at_sr, "mae_before_first_ahead_sr": mae_before_sr,
        "mfe_at_first_ahead_liq": mfe_at_liq, "mae_before_first_ahead_liq": mae_before_liq,
        "checkpoints": snaps,
    })
    return out


def run_symbol_paths(state: SymbolState, anchors: dict) -> dict:
    return scan_paths_streaming(
        entry_idx=anchors["entry_idx"], end_idx=anchors["end_idx"],
        entry_price=anchors["entry_price"], atr0=anchors["atr0"],
        side=anchors["side"], high=state.high, low=state.low, close=state.close,
        segment=state.segment, entry_segment=anchors["entry_segment"],
        sr_enter=anchors["sr_enter"], sr_pierce=anchors["sr_pierce"],
        sr_reclaim=anchors["sr_reclaim"],
        lb_enter=anchors["lb_enter"], lb_pierce=anchors["lb_pierce"],
        lb_reclaim=anchors["lb_reclaim"],
        ahead_sr_touch=anchors["ahead_sr_touch"],
        ahead_sr_cross=anchors["ahead_sr_cross"],
        ahead_liq_touch=anchors["ahead_liq_touch"],
        ahead_liq_cross=anchors["ahead_liq_cross"],
        td_ends=anchors["td_ends"])


# --------------------------------------------------------------------------- #
# 6b. Dual direction-view batch (A9 + E9)                                       #
# --------------------------------------------------------------------------- #
_DUAL_BOUNDARY_KEYS = (
    "entry_idx", "end_idx", "entry_price", "atr0", "entry_segment",
    "sr_enter", "sr_pierce", "sr_reclaim",
    "lb_enter", "lb_pierce", "lb_reclaim",
    "ahead_sr_touch", "ahead_sr_cross",
    "ahead_liq_touch", "ahead_liq_cross",
)


def build_dual_case(state: SymbolState, v_a9: dict, v_e9: dict) -> dict:
    """Stack A9 and E9 views into ONE 2N Candidate-view batch (RC E).

    The canonical environment / ATR / zone geometry are consumed exactly once (the
    shared ``state``); only the direction-relative boundary arrays are doubled. The
    time-step loop in scan_paths_streaming remains the single hot loop.
    """
    cat = lambda k: np.concatenate([v_a9[k], v_e9[k]])
    case = {
        "entry_idx": cat("entry_idx"),
        "end_idx": cat("end_idx"),
        "entry_price": cat("entry_price"),
        "atr0": cat("atr0"),
        "side": cat("side"),
        "high": state.high, "low": state.low, "close": state.close,
        "segment": state.segment,
        "entry_segment": cat("entry_segment"),
        "sr_enter": cat("sr_enter"), "sr_pierce": cat("sr_pierce"),
        "sr_reclaim": cat("sr_reclaim"),
        "lb_enter": cat("lb_enter"), "lb_pierce": cat("lb_pierce"),
        "lb_reclaim": cat("lb_reclaim"),
        "ahead_sr_touch": cat("ahead_sr_touch"),
        "ahead_sr_cross": cat("ahead_sr_cross"),
        "ahead_liq_touch": cat("ahead_liq_touch"),
        "ahead_liq_cross": cat("ahead_liq_cross"),
        "td_ends": {k: np.concatenate([v_a9["td_ends"][k], v_e9["td_ends"][k]])
                    for k in v_a9["td_ends"]},
    }
    return case


def _slice_dual(out: dict, n: int) -> dict:
    a9 = {k: (v[:n] if k != "checkpoints" else v) for k, v in out.items()}
    e9 = {k: (v[n:] if k != "checkpoints" else v) for k, v in out.items()}
    # split checkpoints (nested) by candidate index
    a9c = {nm: {m: out["checkpoints"][nm][m][:n] for m in ("mfe", "mae", "r")}
           for nm in out["checkpoints"]}
    e9c = {nm: {m: out["checkpoints"][nm][m][n:] for m in ("mfe", "mae", "r")}
           for nm in out["checkpoints"]}
    a9["checkpoints"] = a9c
    e9["checkpoints"] = e9c
    return a9, e9


def run_symbol_paths_dual(state: SymbolState, base: dict) -> dict:
    """Build A9 + E9 views, scan them as ONE 2N batch, return split outputs."""
    if base is None:
        return None
    v_a9 = make_direction_view("A9", base, state)
    v_e9 = make_direction_view("E9", base, state)
    case = build_dual_case(state, v_a9, v_e9)
    full = scan_paths_streaming(**{k: case[k] for k in case
                                   if k in inspect.signature(scan_paths_streaming).parameters})
    n = len(base["entry_idx"])
    a9_out, e9_out = _slice_dual(full, n)
    return {"n": n, "full": full, "A9": a9_out, "E9": e9_out,
            "v_a9": v_a9, "v_e9": v_e9, "case": case}


# Fields compared by the agreement invariant (RC F): every side-relative output
# must be IDENTICAL for candidates where A9 and E9 disagree on nothing else but
# the direction label (i.e. agreement rows).
_AGREEMENT_FIELDS = (
    "mfe_final", "mae_final",
    "sr_first_touch", "sr_first_pierce", "sr_same_bar_reclaim", "sr_first_reclaim",
    "sr_bars_to_reclaim", "sr_first_failed_reclaim", "sr_break_continue",
    "lb_first_touch", "lb_first_pierce", "lb_same_bar_reclaim", "lb_first_reclaim",
    "lb_bars_to_reclaim", "lb_first_failed_reclaim", "lb_break_continue",
    "first_ahead_sr_touch", "first_ahead_sr_cross",
    "first_ahead_liq_touch", "first_ahead_liq_cross",
    "mfe_at_first_ahead_sr", "mae_before_first_ahead_sr",
    "mfe_at_first_ahead_liq", "mae_before_first_ahead_liq",
)


def check_agreement_invariant(a9_out, e9_out, agreement_mask, symbol="?"):
    """HARD gate (RC F): for every agreement candidate, A9 and E9 path outputs must
    be identical (NaN-mask equal)."""
    idx = np.flatnonzero(agreement_mask)
    if idx.size == 0:
        return
    for f in _AGREEMENT_FIELDS:
        if not _arr_eq(a9_out[f][idx], e9_out[f][idx]):
            raise RuntimeError(
                f"STOP_A9_E9_AGREEMENT_INVARIANT_FAILED symbol={symbol} field={f}")
    for nm in CHECKPOINT_NAMES:
        for m in ("mfe", "mae", "r"):
            if not _arr_eq(a9_out["checkpoints"][nm][m][idx],
                           e9_out["checkpoints"][nm][m][idx]):
                raise RuntimeError(
                    f"STOP_A9_E9_AGREEMENT_INVARIANT_FAILED symbol={symbol} "
                    f"checkpoint={nm}.{m}")


def decompose_disagreement(base, symbol="?"):
    """RC G: classify disagreement rows into E9_FIX / E9_BREAK and verify exactly
    one of {A9, E9} is correct (binary oracle)."""
    a9_c = np.asarray(base["a9_direction_correct"], dtype=np.uint8)
    e9_c = np.asarray(base["e9_direction_correct"], dtype=np.uint8)
    agreement = (a9_c == e9_c)
    disagreement = ~agreement
    # binary oracle => on disagreement exactly one is correct
    bad = disagreement & (a9_c + e9_c != 1)
    if bad.any():
        raise RuntimeError(
            f"STOP_A9_E9_DISAGREEMENT_PARITY symbol={symbol} "
            f"rows={int(bad.sum())}")
    e9_fix = disagreement & (a9_c == 0) & (e9_c == 1)
    e9_break = disagreement & (a9_c == 1) & (e9_c == 0)
    return {
        "n": int(len(a9_c)),
        "agreement": int(agreement.sum()),
        "disagreement": int(disagreement.sum()),
        "e9_fix": int(e9_fix.sum()),
        "e9_break": int(e9_break.sum()),
        "agreement_rate": float(agreement.mean()) if len(a9_c) else 0.0,
    }


# --------------------------------------------------------------------------- #
# 7. Reference kernel (T0/T1 only)                                             #
# --------------------------------------------------------------------------- #
def scan_paths_reference(*, entry_idx, end_idx, entry_price, atr0, side,
                         high, low, close, segment, entry_segment,
                         sr_enter, sr_pierce, sr_reclaim,
                         lb_enter, lb_pierce, lb_reclaim,
                         ahead_sr_touch, ahead_sr_cross,
                         ahead_liq_touch, ahead_liq_cross,
                         td_ends=None):
    """Deliberately slow per-candidate reference. T0/T1 only."""
    n = len(entry_idx)
    res = {"mfe_final": np.zeros(n), "mae_final": np.zeros(n)}
    for pfx in ("sr", "lb"):
        res[f"{pfx}_first_touch"] = np.full(n, -1, np.int32)
        res[f"{pfx}_first_pierce"] = np.full(n, -1, np.int32)
        res[f"{pfx}_first_reclaim"] = np.full(n, -1, np.int32)
        res[f"{pfx}_first_failed_reclaim"] = np.full(n, -1, np.int32)
        res[f"{pfx}_same_bar_reclaim"] = np.zeros(n, bool)
        res[f"{pfx}_bars_to_reclaim"] = np.full(n, -1, np.int32)
        res[f"{pfx}_break_continue"] = np.zeros(n, bool)
    for k in ("first_ahead_sr_touch", "first_ahead_sr_cross",
              "first_ahead_liq_touch", "first_ahead_liq_cross"):
        res[k] = np.full(n, -1, np.int32)
    for k in ("mfe_at_first_ahead_sr", "mae_before_first_ahead_sr",
              "mfe_at_first_ahead_liq", "mae_before_first_ahead_liq"):
        res[k] = np.full(n, np.nan)
    snaps = {name: {"mfe": np.full(n, np.nan), "mae": np.full(n, np.nan),
                    "r": np.full(n, np.nan)} for name in CHECKPOINT_NAMES}
    bar_steps = dict(BAR_CHECKPOINTS)

    for i in range(n):
        a = int(entry_idx[i]); e = int(end_idx[i]); seg = int(entry_segment[i])
        p0 = float(entry_price[i]); atr = float(atr0[i]); s = float(side[i])
        st = {"sr": {"pierced": False, "reclaimed": False},
              "lb": {"pierced": False, "reclaimed": False}}
        mfe = 0.0
        mae = 0.0
        for step in range(int(e - a) + 1):
            j = a + step
            if j > e or j >= len(close) or segment[j] != seg:
                continue
            hi = float(high[j]); lo = float(low[j]); cl = float(close[j])
            mfe = max(mfe, ((hi - p0) if s > 0 else (p0 - lo)) / atr)
            mae = max(mae, ((p0 - lo) if s > 0 else (hi - p0)) / atr)

            for pfx, (eb, pb, rb) in (
                    ("sr", (float(sr_enter[i]), float(sr_pierce[i]), float(sr_reclaim[i]))),
                    ("lb", (float(lb_enter[i]), float(lb_pierce[i]), float(lb_reclaim[i])))):
                if res[f"{pfx}_first_touch"][i] < 0 and (
                        (s > 0 and lo <= eb) or (s < 0 and hi >= eb)):
                    res[f"{pfx}_first_touch"][i] = step
                pierce_now = (not st[pfx]["pierced"]) and (
                    (s > 0 and lo < pb) or (s < 0 and hi > pb))
                if pierce_now:
                    res[f"{pfx}_first_pierce"][i] = step
                    st[pfx]["pierced"] = True
                vc = (cl >= rb) if s > 0 else (cl <= rb)
                if pierce_now and vc:
                    res[f"{pfx}_same_bar_reclaim"][i] = True
                    res[f"{pfx}_first_reclaim"][i] = step
                    st[pfx]["reclaimed"] = True
                elif st[pfx]["pierced"] and not st[pfx]["reclaimed"] and vc:
                    res[f"{pfx}_first_reclaim"][i] = step
                    st[pfx]["reclaimed"] = True
                if (st[pfx]["reclaimed"]
                        and res[f"{pfx}_first_failed_reclaim"][i] < 0 and not vc):
                    res[f"{pfx}_first_failed_reclaim"][i] = step

            if res["first_ahead_sr_touch"][i] < 0 and (
                    (s > 0 and hi >= float(ahead_sr_touch[i]))
                    or (s < 0 and lo <= float(ahead_sr_touch[i]))):
                res["first_ahead_sr_touch"][i] = step
                res["mfe_at_first_ahead_sr"][i] = mfe
                res["mae_before_first_ahead_sr"][i] = mae
            if res["first_ahead_sr_cross"][i] < 0 and (
                    (s > 0 and cl > float(ahead_sr_cross[i]))
                    or (s < 0 and cl < float(ahead_sr_cross[i]))):
                res["first_ahead_sr_cross"][i] = step
            if res["first_ahead_liq_touch"][i] < 0 and (
                    (s > 0 and hi >= float(ahead_liq_touch[i]))
                    or (s < 0 and lo <= float(ahead_liq_touch[i]))):
                res["first_ahead_liq_touch"][i] = step
                res["mfe_at_first_ahead_liq"][i] = mfe
                res["mae_before_first_ahead_liq"][i] = mae
            if res["first_ahead_liq_cross"][i] < 0 and (
                    (s > 0 and cl > float(ahead_liq_cross[i]))
                    or (s < 0 and cl < float(ahead_liq_cross[i]))):
                res["first_ahead_liq_cross"][i] = step

            if step in bar_steps:
                nm = bar_steps[step]
                snaps[nm]["mfe"][i] = mfe
                snaps[nm]["mae"][i] = mae
                snaps[nm]["r"][i] = s * (cl - p0) / atr
            if td_ends is not None:
                for nm, arr in td_ends.items():
                    if j == int(arr[i]):
                        snaps[nm]["mfe"][i] = mfe
                        snaps[nm]["mae"][i] = mae
                        snaps[nm]["r"][i] = s * (cl - p0) / atr

        res["mfe_final"][i] = mfe
        res["mae_final"][i] = mae
        for pfx in ("sr", "lb"):
            res[f"{pfx}_bars_to_reclaim"][i] = (
                res[f"{pfx}_first_reclaim"][i] - res[f"{pfx}_first_pierce"][i]
                if (res[f"{pfx}_first_pierce"][i] >= 0
                    and res[f"{pfx}_first_reclaim"][i] >= 0) else -1)
            res[f"{pfx}_break_continue"][i] = (
                res[f"{pfx}_first_pierce"][i] >= 0
                and res[f"{pfx}_first_reclaim"][i] < 0)
    res["checkpoints"] = snaps
    return res


# --------------------------------------------------------------------------- #
# 8. Statistics                                                                #
# --------------------------------------------------------------------------- #
def cluster_bootstrap_gid(values, gid, w=None, B=2000, seed=20260924):
    values = np.asarray(values, dtype=np.float64)
    gid = np.asarray(gid, dtype=object)
    w = np.ones(len(values)) if w is None else np.asarray(w, np.float64)
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
    return (float(trade_mean.mean()), float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)))


def delta_ps_cluster_bootstrap(ps, correct, gid, w, B=2000, seed=20260924):
    """Primary contrast delta_PS = weighted_mean(PS|correct) - weighted_mean(PS|wrong).

    Bootstrap unit is the WHOLE gid: every replicate resamples gids with replacement,
    carries all Candidate rows of a sampled gid together, and recomputes BOTH the
    correct and the wrong weighted mean inside the SAME replicate before differencing.
    This preserves the cluster dependence of a gid that contributes both correct and
    wrong Candidates. The two groups are never bootstrapped independently.
    """
    ps = np.asarray(ps, dtype=np.float64)
    correct = np.asarray(correct).astype(bool)
    gid = np.asarray(gid, dtype=object)
    w = np.asarray(w, dtype=np.float64)
    ok = np.isfinite(ps) & np.isfinite(w) & (w > 0)
    ps, correct, gid, w = ps[ok], correct[ok], gid[ok], w[ok]
    if ps.size == 0:
        return {"point": None, "ci_low": None, "ci_high": None,
                "n_gids": 0, "correct_mass": 0.0, "wrong_mass": 0.0}

    ug, inv = np.unique(gid, return_inverse=True)
    wc = np.where(correct, w, 0.0)
    ww = np.where(~correct, w, 0.0)
    num_c = np.bincount(inv, weights=ps * wc, minlength=len(ug))
    den_c = np.bincount(inv, weights=wc, minlength=len(ug))
    num_w = np.bincount(inv, weights=ps * ww, minlength=len(ug))
    den_w = np.bincount(inv, weights=ww, minlength=len(ug))

    k = len(ug)
    rng = np.random.default_rng(seed)
    deltas = np.empty(B, dtype=np.float64)
    for b in range(B):
        idx = rng.integers(0, k, size=k)
        dc = den_c[idx].sum()
        dw = den_w[idx].sum()
        mc = num_c[idx].sum() / dc if dc > 0 else np.nan
        mw = num_w[idx].sum() / dw if dw > 0 else np.nan
        deltas[b] = mc - mw
    Dc = float(den_c.sum())
    Dw = float(den_w.sum())
    point = ((num_c.sum() / Dc) if Dc > 0 else np.nan) - (
        (num_w.sum() / Dw) if Dw > 0 else np.nan)
    return {"point": float(point),
            "ci_low": float(np.nanquantile(deltas, 0.025)),
            "ci_high": float(np.nanquantile(deltas, 0.975)),
            "n_gids": int(k), "correct_mass": Dc, "wrong_mass": Dw,
            "n_rows": int(ps.size)}


def group_stats(values, gid, w):
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
    med = float(vs[int(np.searchsorted(cw, 0.5 * cw[-1]))])
    qs = {}
    for q in (10, 25, 50, 75, 90):
        c = (q / 100.0) * cw[-1]
        qs[f"p{q}"] = float(vs[min(int(np.searchsorted(cw, c)), len(vs) - 1)])
    return {"n_rows": int(ok.sum()), "mean": mean, "ci_low": lo, "ci_high": hi,
            "weighted_median": med, **qs}


# --------------------------------------------------------------------------- #
# 9. Differential harness (RC5)                                                #
# --------------------------------------------------------------------------- #
DIFF_FIELDS = (
    "mfe_final", "mae_final",
    "sr_first_touch", "sr_first_pierce", "sr_same_bar_reclaim", "sr_first_reclaim",
    "sr_bars_to_reclaim", "sr_first_failed_reclaim", "sr_break_continue",
    "lb_first_touch", "lb_first_pierce", "lb_same_bar_reclaim", "lb_first_reclaim",
    "lb_bars_to_reclaim", "lb_first_failed_reclaim", "lb_break_continue",
    "first_ahead_sr_touch", "first_ahead_sr_cross",
    "first_ahead_liq_touch", "first_ahead_liq_cross",
    "mfe_at_first_ahead_sr", "mae_before_first_ahead_sr",
    "mfe_at_first_ahead_liq", "mae_before_first_ahead_liq",
)


def _cmp(a, b, out):
    """Compare two arrays; NaN/value mask mismatch counts as a mismatch."""
    a = np.asarray(a)
    b = np.asarray(b)
    cells = int(a.size)
    out["cells"] += cells
    if np.issubdtype(a.dtype, np.bool_) or np.issubdtype(a.dtype, np.integer):
        d = a != b
    else:
        na, nb = np.isnan(a), np.isnan(b)
        if np.array_equal(na, nb):
            d = np.zeros(a.shape, bool)
            if (~na).any():
                d = np.zeros(a.shape, bool)
                d[~na] = np.abs(a[~na] - b[~na]) > 1e-12
        else:
            d = na != nb
    if d.any():
        out["mismatch"] += int(d.sum())
        if out["first_mismatch"] is None:
            out["first_mismatch"] = int(np.flatnonzero(d)[0])
    return out


def diff_reference_vs_production(case: dict) -> dict:
    _sig = inspect.signature(scan_paths_streaming)
    clean = {k: v for k, v in case.items() if k in _sig.parameters}
    p = scan_paths_streaming(**clean)
    r = scan_paths_reference(**clean)
    out = {"rows": len(p["mfe_final"]), "cells": 0, "mismatch": 0,
           "max_abs_error": 0.0, "first_mismatch": None}
    for f in DIFF_FIELDS:
        _cmp(p[f], r[f], out)
    for name in CHECKPOINT_NAMES:
        for m in ("mfe", "mae", "r"):
            _cmp(p["checkpoints"][name][m], r["checkpoints"][name][m], out)
    # finite float error
    for f in ("mfe_final", "mae_final", "mfe_at_first_ahead_sr",
              "mae_before_first_ahead_sr", "mfe_at_first_ahead_liq",
              "mae_before_first_ahead_liq"):
        a = np.asarray(p[f], dtype=float)
        b = np.asarray(r[f], dtype=float)
        m = np.isfinite(a) & np.isfinite(b)
        if m.any():
            out["max_abs_error"] = max(out["max_abs_error"],
                                       float(np.max(np.abs(a[m] - b[m]))))
    for name in CHECKPOINT_NAMES:
        for k in ("mfe", "mae", "r"):
            a = np.asarray(p["checkpoints"][name][k], dtype=float)
            b = np.asarray(r["checkpoints"][name][k], dtype=float)
            m = np.isfinite(a) & np.isfinite(b)
            if m.any():
                out["max_abs_error"] = max(out["max_abs_error"],
                                           float(np.max(np.abs(a[m] - b[m]))))
    return out


def make_synthetic_case(N: int, H: int, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    n_bars = H + 8
    close = 100.0 + np.cumsum(rng.normal(0, 0.3, n_bars))
    high = close + np.abs(rng.normal(0, 0.3, n_bars))
    low = close - np.abs(rng.normal(0, 0.3, n_bars))
    entry_idx = rng.integers(0, n_bars - H - 1, size=N)
    p0 = close[entry_idx]
    return dict(
        entry_idx=entry_idx.astype(np.int64),
        end_idx=(entry_idx + H).astype(np.int64),
        entry_price=p0, atr0=np.full(N, 1.0),
        side=np.where(rng.random(N) < 0.5, 1.0, -1.0),
        high=high, low=low, close=close,
        segment=np.zeros(n_bars, dtype=np.int64),
        entry_segment=np.zeros(N, dtype=np.int64),
        sr_enter=np.where(rng.random(N) < 0.5, p0 - 1.0, p0 + 1.0),
        sr_pierce=np.where(rng.random(N) < 0.5, p0 - 2.0, p0 + 2.0),
        sr_reclaim=np.where(rng.random(N) < 0.5, p0 - 2.0, p0 + 2.0),
        lb_enter=p0 - 3.0, lb_pierce=p0 - 4.0, lb_reclaim=p0 - 4.0,
        ahead_sr_touch=p0 + 1.5, ahead_sr_cross=p0 + 2.5,
        ahead_liq_touch=p0 + 3.0, ahead_liq_cross=p0 + 4.0,
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
    r = lambda x, y: (y / x) if x > 0 else None
    return {
        "candidate_scaling": {"N": base_n, "2N": base_n * 2, "4N": base_n * 4,
                              "H": horizon, "t_N": t1, "t_2N": t2, "t_4N": t4,
                              "ratio_2N": r(t1, t2), "ratio_4N": r(t1, t4)},
        "horizon_scaling": {"H": horizon, "2H": horizon * 2, "N": base_n,
                            "t_H": h1, "t_2H": h2, "ratio_2H": r(h1, h2)},
    }


# --------------------------------------------------------------------------- #
# 10. Kernel Checkpoint driver                                                 #
# --------------------------------------------------------------------------- #
def run_kernel_checkpoint(symbols=SYMBOLS, n_subset=50, verbose=True) -> dict:
    """T1.5 integration: canonical environment materialization + A9/E9 dual-view
    path scan over a small deterministic subset per symbol.

    Per symbol gates (RC L):
      * env complete
      * environment provenance valid (canonical R4 keys; fail-closed on smoke cache)
      * anchor alignment pass (RC6)
      * zone audit pass
      * Reference vs Production mismatch = 0 (both A9 + E9 via 2N batch)
      * max abs error <= 1e-12
      * A9/E9 agreement invariant pass (RC F)
    """
    def log(*a):
        if verbose:
            print(*a, file=sys.stderr, flush=True)

    t_start = time.time()
    reset_counters()
    log("materialize frozen E9 (+A9 alias) direction state (run_chain ONCE) ...")
    e9 = materialize_e9_direction_state(save=True, verbose=verbose)

    pop = {
        "test_rows": int(len(e9)),
        "test_trades": int(e9["gid"].nunique()),
        "long_trades": int(e9.loc[e9["oracle_direction"] == "LONG", "gid"].nunique()),
        "short_trades": int(e9.loc[e9["oracle_direction"] == "SHORT", "gid"].nunique()),
        "e9_predicted_long_rows": int((e9["e9_direction"] == "LONG").sum()),
        "e9_predicted_short_rows": int((e9["e9_direction"] == "SHORT").sum()),
        "a9_predicted_long_rows": int((e9["a9_direction"] == "LONG").sum()),
        "a9_predicted_short_rows": int((e9["a9_direction"] == "SHORT").sum()),
        "e9_direction_correct_rate": float(e9["e9_direction_correct"].mean()),
        "a9_direction_correct_rate": float(e9["a9_direction_correct"].mean()),
        "a9_equals_router_te": bool(np.array_equal(
            e9["a9_direction"].to_numpy(object),
            e9["router_direction"].to_numpy(object))),
    }

    blocks = {}
    total_candidates = 0
    for sym in symbols:
        log(f"symbol {sym}: load canonical state (materialize env if missing) ...")
        st = load_symbol_state(sym)
        if not st.env_complete:
            blocks[sym] = {"env_complete": False, "note": st.env_note}
            continue
        # deterministic subset: first min(n_subset, N) by canonical semantic-key order
        sel = e9[e9["symbol"] == sym].sort_values("semantic_key")
        k = int(min(n_subset, len(sel)))
        base = build_base_anchors(sel.head(k), st)
        if base is None:
            blocks[sym] = {"env_complete": True, "n_candidates": 0}
            continue
        total_candidates += k
        dual = run_symbol_paths_dual(st, base)
        rep = diff_reference_vs_production(dual["case"])
        agreement_mask = (np.asarray(base["a9_direction"]) == np.asarray(base["e9_direction"]))
        agreement_pass = True
        agr_msg = ""
        try:
            check_agreement_invariant(dual["A9"], dual["E9"], agreement_mask, sym)
        except RuntimeError as e:
            agreement_pass = False
            agr_msg = str(e)
        decomp = decompose_disagreement(base, sym)
        za = st.zone_audit
        zone_audit_max = max(za.values()) if za else 0.0
        blocks[sym] = {
            "env_complete": True,
            "frame_rows": int(st.n_bars),
            "n_candidates": k,
            "env_provenance": st.env_provenance,
            "anchor_alignment_pass": True,   # build_base_anchors raised otherwise
            "zone_audit_pass": zone_audit_max < 1e-9,
            "zone_audit_max_dev": zone_audit_max,
            "t1_mismatch": rep["mismatch"],
            "t1_max_abs_error": rep["max_abs_error"],
            "reference_vs_production_pass": (rep["mismatch"] == 0
                                            and rep["max_abs_error"] <= 1e-12),
            "agreement_invariant_pass": agreement_pass,
            "agreement_invariant_msg": agr_msg,
            "a9_e9_counts": decomp,
        }

    log("TP scaling benchmark ...")
    scaling = tp_scaling_benchmark()
    try:
        import resource as _res
        rss = float(_res.getrusage(_res.RUSAGE_SELF).ru_maxrss)
    except Exception:
        rss = None

    # aggregate environment provenance summary (RC9) across evaluated symbols
    env_summary = {}
    for sym, blk in blocks.items():
        if blk.get("env_complete") and "env_provenance" in blk:
            pv = blk["env_provenance"]
            env_summary[sym] = {
                "environment_contract_id": pv.get("environment_contract_id"),
                "cache_schema_version": pv.get("cache_schema_version"),
                "max_bars": pv.get("max_bars"),
                "rows": pv.get("rows"),
            }

    artifacts = {}
    for p in (E9_STATE_PARQUET, ANCHORS_PARQUET, ROW_METRICS_PARQUET):
        if os.path.exists(p):
            artifacts[p] = _sha256_file(p)

    extra = {
        "branch": "entry-path-atlas-v1",
        "code_sha": _git_head_sha(),
        "reviewed_parent_sha": REVIEWED_SHA,
        "population": pop,
        "symbols_evaluated": list(symbols),
        "n_symbols_env_complete": int(sum(1 for b in blocks.values()
                                          if b.get("env_complete"))),
        "symbol_blocks": blocks,
        "environment_provenance_summary": env_summary,
        "a9_comparator": {
            "role": "PRE_REGISTERED_SECONDARY_COMPARATOR",
            "identity": "A9 := chain['A'] (frozen direct DTP9 router)",
            "primary_remains": "delta_PS_4h_E9",
            "note": "A9 does NOT promote to a second primary; no A9-vs-E9 verdict here.",
        },
        "agreement_disagreement_total": {
            "agreement": int(sum(b["a9_e9_counts"]["agreement"] for b in blocks.values()
                                 if "a9_e9_counts" in b)),
            "disagreement": int(sum(b["a9_e9_counts"]["disagreement"] for b in blocks.values()
                                    if "a9_e9_counts" in b)),
            "e9_fix": int(sum(b["a9_e9_counts"]["e9_fix"] for b in blocks.values()
                              if "a9_e9_counts" in b)),
            "e9_break": int(sum(b["a9_e9_counts"]["e9_break"] for b in blocks.values()
                                if "a9_e9_counts" in b)),
        },
        "zone_semantics": {
            "long_backstop": "support: touch=low<=support_top, "
                             "pierce=low<support_bottom, reclaim=close>=support_bottom",
            "short_backstop": "resistance: touch=high>=resistance_bottom, "
                              "pierce=high>resistance_top, reclaim=close<=resistance_top",
            "rule": "entering a zone is NEVER a pierce-through",
        },
        "checkpoint_semantics": {
            "nan_filled": True,
            "truncated_candidate_checkpoints": "unavailable (NaN), never inherited",
        },
        "tp": {
            "counters": dict(COUNTERS),
            "scaling": scaling,
            "peak_rss_bytes": rss,
            "performance": {
                "total_environment_loads": int(COUNTERS["raw_exec_load_count"]),
                "direction_chain_runs": int(COUNTERS["direction_chain_run_count"]),
                "path_scans": int(COUNTERS["path_scan_count"]),
                "candidate_views_processed": int(2 * total_candidates),
                "runtime_sec": time.time() - t_start,
            },
        },
        "artifacts": artifacts,
        "runtime_sec": time.time() - t_start,
        "unverified_items": [
            "Full primary endpoint delta_PS_4h (E9 and A9) is NOT estimated at T1.5.",
            "L2 row metrics parquet (semantic_key x direction_system) not produced.",
            "Stop-Loss / Take-Profit experiments NOT run (frozen later).",
            "FUTURE-RX-DIRECTION-LAYER-ABLATION-V1 pre-registered; NOT run here.",
        ],
    }
    return write_manifest(extra=extra)


# --------------------------------------------------------------------------- #
# 11. Manifest                                                                 #
# --------------------------------------------------------------------------- #
def write_manifest(*, stage=STAGE, extra=None, path=MANIFEST_JSON):
    payload = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "reviewed_parent_sha": REVIEWED_SHA,
        "stage": stage,
        "primary_endpoint": {
            "name": "delta_PS_4h",
            "definition": "E[PS_4h | direction_correct=1] - E[PS_4h | direction_correct=0]",
            "ps": "MFE - MAE (side-normalized, ATR0 units)",
            "checkpoint": PRIMARY_CHECKPOINT,
            "bars": 16,
            "inference": "whole-gid cluster bootstrap (correct and wrong means "
                         "recomputed inside the SAME replicate)",
        },
        "audit_only_fields": list(AUDIT_ONLY_FIELDS),
        "realtime_policy": FORBIDDEN_REALTIME_FEATURE,
        "counters": dict(COUNTERS),
        "checkpoints": list(CHECKPOINT_NAMES),
        "diff_fields": list(DIFF_FIELDS),
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
