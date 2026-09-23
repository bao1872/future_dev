"""build_teacher_oracle_dp_m15_overnight_v1
===========================================

Overnight-allowing TEACHER Oracle for STRUCT33 direction/quality study
(TASK ID: FUTURE-R4-M15-STRUCT33-DIRECTION-QUALITY-V1).

This module is a SEPARATE artifact from the frozen day-flat 15m mechanical R2
Oracle (``build_trade_oracle_dp_m15_one_entry_proximity_v1``). It is intentionally
decoupled so the original frozen baseline stays byte-frozen and can always be
used for differential testing.

Division of ownership
----------------------
* REUSED (imported, never redefined here): the frozen 6-state Bellman kernel
  ``solve_day_dp_v2`` / ``_solve_unit_v2``, the action selector, the new-entry
  predicate, the exhausive reference, helpers (``_trade_excursion``,
  ``_transition_label``, ``_walk_unit_path``), the DP-proximity owner and the
  proximity-episode id computer.
* OWNED HERE (the only semantic delta vs the frozen baseline):
  - unit construction = one (hard) SEGMENT only; the ordinary trading-day
    boundary is NOT a terminal, so positions may be carried overnight;
  - terminal policy: a unit terminates only at a hard segment discontinuity
    (``HARD_BOUNDARY``) or the true data end (``DATA_END``);
  - per-trade exit_reason: ``HARD_BOUNDARY`` / ``DATA_END`` (forced,
    training_eligible=False), ``REVERSAL``, ``OPTIMAL_FLAT`` (both eligible);
  - Teacher artifact metadata + its own artifact root.

The frozen baseline's ``(trading_day, segment)`` day-flat units are NOT present
in this module at all.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 import (
    BASE_ALLOWED,
    EPS,
    INVALID_ACTION,
    MATH_VERSION,
    NEW_ENTRY,
    POS,
    ACT,
    S2I,
    STATE_LABELS,
    STATES,
    TURNOVER,
    choose_actions_v2,
    compute_proximity_episode_id,
    exhaustive_reference_v2,
    is_new_entry,
    solve_day_dp_v2,
    _solve_unit_v2,
    _walk_unit_path,
    _trade_excursion,
    _transition_label,
    build_dp_proximity_m15,
    build_artifact_frames_v1,
)
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    ENTRY_PROX_ATR,
    KernelCounters,
)


# --------------------------------------------------------------------------- #
# Teacher-owned terminal vocabulary                                            #
# --------------------------------------------------------------------------- #
DATA_END = "DATA_END"
HARD_BOUNDARY = "HARD_BOUNDARY"
OPTIMAL_FLAT = "OPTIMAL_FLAT"
REVERSAL = "REVERSAL"

TASK_ID = "FUTURE-R4-M15-STRUCT33-DIRECTION-QUALITY-V1-TEACHER"
# Clean Teacher contract id, separated from the legacy pseudo-SHA field.
TEACHER_CONTRACT_ID = "FUTURE-R4-M15-OVERNIGHT-TEACHER-V1"
ARTIFACT_ROOT = "artifacts/teacher_oracle_dp_m15_overnight_v1"

ORACLE_ACTIONS_FILE = "oracle_actions.parquet"
ORACLE_TRADES_FILE = "oracle_trades.parquet"
ORACLE_METADATA_FILE = "metadata.json"

# columns the Viewer overlay requires on the trades table (V2 superset)
ORACLE_VIEWER_TRADE_COLUMNS = (
    "direction",
    "entry_fill_time",
    "entry_fill_price",
    "exit_fill_time",
    "exit_fill_price",
    "trade_id",
    "entry_proximity_bits",
    "entry_proximity_episode_id",
    "entry_armed_before",
)


# --------------------------------------------------------------------------- #
# Teacher-owned unit construction (SEGMENT only, overnight)                    #
# --------------------------------------------------------------------------- #
def build_intraday_units(segment: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Overnight Teacher units: a unit is a single (hard) SEGMENT.

    The ordinary trading-day boundary is NOT a terminal, so positions may be
    carried overnight (e.g. Monday night open -> Tuesday close). Only a segment
    change (canonical hard discontinuity, ``segment = cumsum(disc)``) or the true
    data end terminates a unit.
    """
    segment = np.asarray(segment)
    boundary = np.empty(len(segment), dtype=bool)
    boundary[0] = True
    if len(segment) > 1:
        boundary[1:] = segment[1:] != segment[:-1]
    starts = np.flatnonzero(boundary)
    ends = np.r_[starts[1:] - 1, len(segment) - 1]
    return starts, ends


def _unit_terminal_reason(end: int, n: int, seg_arr: np.ndarray) -> str:
    """Overnight Teacher unit terminal.

    In segment-only units every non-data-end unit end is a hard segment change,
    which is the canonical hard boundary.
    """
    if end >= n - 1:
        return DATA_END
    return HARD_BOUNDARY


# --------------------------------------------------------------------------- #
# Trade reconstruction (per-trade exit_reason + eligibility)                    #
# --------------------------------------------------------------------------- #
def _append_unit_trades(
    trades: List[Dict[str, Any]],
    symbol: str,
    td_arr: np.ndarray,
    times: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    proximity_bits: np.ndarray,
    prox_ep_id: np.ndarray,
    cost: np.ndarray,
    path: List[Dict[str, Any]],
    terminal_reason: str,
    training_eligible: bool,
    end: int,
) -> None:
    """Reconstruct complete trades from the unit's optimal (t, p, a, q) path.

    Each trade's own exit reason is resolved from how it actually closed:
      * close at the unit's forced-flat decision (t == end-1) => the unit
        terminal reason (HARD_BOUNDARY / DATA_END), training_eligible=False
      * close by reversal (new entry) => REVERSAL, training_eligible=True
      * close by voluntary flat      => OPTIMAL_FLAT, training_eligible=True
    """
    open_trade: Optional[Dict[str, Any]] = None

    def start(d: Dict[str, Any]) -> Dict[str, Any]:
        fill = int(d["t"]) + 1
        direction = "LONG" if d["pa"] == 1 else "SHORT"
        return {
            "trade_id": f"{symbol}_{direction}_{fill}",
            "symbol": symbol,
            "trading_day": td_arr[d["t"]],
            "direction": direction,
            "entry_decision_index": int(d["t"]),
            "entry_fill_index": fill,
            "entry_fill_time": times[fill],
            "entry_fill_price": float(opens[fill]),
            "entry_proximity_bits": int(proximity_bits[d["t"]]),
            "entry_proximity_episode_id": int(prox_ep_id[d["t"]]),
            "entry_armed_before": int(d["qb"]),
            "terminal_reason": terminal_reason,
            "training_eligible": bool(training_eligible),
        }

    def close(tr: Dict[str, Any], d: Dict[str, Any]) -> Dict[str, Any]:
        fill = int(d["t"]) + 1
        tr = dict(tr)
        tr.update(
            {
                "exit_decision_index": int(d["t"]),
                "exit_fill_index": fill,
                "exit_fill_time": times[fill],
                "exit_fill_price": float(opens[fill]),
            }
        )
        s = 1 if tr["direction"] == "LONG" else -1
        entry = float(tr["entry_fill_price"])
        exit_p = float(tr["exit_fill_price"])
        tr["holding_bars"] = int(fill - int(tr["entry_fill_index"]))
        tr["gross_points"] = s * (exit_p - entry)
        tr["cost_points"] = float(cost[int(tr["entry_decision_index"])]) + float(
            cost[int(d["t"])]
        )
        tr["net_points"] = tr["gross_points"] - tr["cost_points"]
        mfe, mae = _trade_excursion(
            s, entry, exit_p, tr["entry_fill_index"], tr["exit_fill_index"],
            highs, lows,
        )
        tr["MFE"] = mfe
        tr["MAE"] = mae
        is_boundary = int(d["t"]) == end - 1
        if is_boundary:
            treason, telig = terminal_reason, False
        elif d["ne"]:
            treason, telig = REVERSAL, True
        else:
            treason, telig = OPTIMAL_FLAT, True
        tr["terminal_reason"] = treason
        tr["training_eligible"] = bool(telig)
        return tr

    for d in path:
        if open_trade is None:
            if d["ne"]:
                open_trade = start(d)
        else:
            if d["pa"] == 0:
                trades.append(close(open_trade, d))
                open_trade = None
            elif d["ne"]:
                trades.append(close(open_trade, d))
                open_trade = start(d)

    if open_trade is not None:
        raise AssertionError("unit DP left an unclosed trade")


# --------------------------------------------------------------------------- #
# Teacher DP driver (reuses frozen Bellman kernel via _solve_unit_v2)           #
# --------------------------------------------------------------------------- #
def _dp_from_proximity(
    symbol: str,
    prox: pd.DataFrame,
    counters: KernelCounters,
    *,
    cost_points: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    n = int(len(prox))
    opens = prox["open"].to_numpy(float)[:n]
    highs = prox["high"].to_numpy(float)[:n]
    lows = prox["low"].to_numpy(float)[:n]
    times = prox["bar_start_time"].to_numpy()[:n]
    seg_arr = prox["segment"].to_numpy(np.int64)[:n]
    td_arr = pd.to_datetime(prox["trading_day"]).to_numpy()[:n]
    proximity_bits = np.asarray(prox["dp_proximity_bits"], dtype=np.int64)[:n]
    proximity_any = np.asarray(prox["dp_proximity_any"], dtype=bool)[:n]
    if cost_points is None:
        cost = np.zeros(n, dtype=float)
    else:
        cost = np.asarray(cost_points, dtype=float)

    starts, ends = build_intraday_units(seg_arr)
    prox_ep_id = compute_proximity_episode_id(proximity_any, starts, n)

    dec: Dict[str, np.ndarray] = {
        "proximity_bits": np.zeros(n, dtype=np.int64),
        "proximity_any": np.zeros(n, dtype=bool),
        "proximity_episode_id": np.full(n, -1, dtype=np.int64),
        "position_before": np.zeros(n, dtype=np.int8),
        "position_after": np.zeros(n, dtype=np.int8),
        "entry_right_before": np.zeros(n, dtype=np.int8),
        "entry_right_after": np.zeros(n, dtype=np.int8),
        "new_entry_consumed": np.zeros(n, dtype=bool),
        "transition": np.empty(n, dtype=object),
    }
    for label in STATE_LABELS:
        for a_idx, a_letter in enumerate(("s", "f", "l")):
            dec[f"q_{label}_{a_letter}"] = np.full(n, np.nan, dtype=float)
        dec[f"best_{label}"] = np.zeros(n, dtype=np.int8)
        dec[f"edge_{label}"] = np.full(n, np.nan, dtype=float)
        dec[f"amb_{label}"] = np.zeros(n, dtype=bool)
    dec["ambiguous"] = np.zeros(n, dtype=bool)
    dec["terminal_reason"] = np.empty(n, dtype=object)
    dec["training_eligible"] = np.zeros(n, dtype=bool)
    dec["label_available_time"] = np.full(
        n, np.datetime64("NaT", "ns"), dtype="datetime64[ns]"
    )

    trades: List[Dict[str, Any]] = []
    units: List[Dict[str, Any]] = []
    unit_values: List[float] = []
    valid = np.zeros(n, dtype=bool)

    for s, e in zip((int(x) for x in starts), (int(x) for x in ends)):
        term = _unit_terminal_reason(e, n, seg_arr)
        eligible = term != DATA_END
        # 15m clock availability: decision close i -> known at i close (+15min)
        lav = times[e] + np.timedelta64(15, "m")
        units.append(
            {
                "seg_start": s,
                "seg_end": e,
                "terminal_reason": term,
                "training_eligible": eligible,
            }
        )
        if e - s < 2:
            continue

        core, length = _solve_unit_v2(opens, proximity_any, cost, s, e)
        counters.dp_state_count += (e - s) * len(STATES)
        actions = core["actions"]
        Q = core["Q"]
        edges = core["edges"]
        ambg = core["ambiguous"]
        unit_values.append(float(np.nanmax(Q[0, S2I[(0, 1)], :])))

        # optimal (p, q) path from (Flat, armed=1); rewire q via global prox
        p = 0
        q = 1
        path: List[Dict[str, Any]] = []
        for local in range(length):
            si = S2I[(p, q)]
            a = int(actions[local, si])
            if a == int(INVALID_ACTION):
                raise RuntimeError(f"missing DP action at t={s + local}, state {(p, q)}")
            new_entry = is_new_entry(p, a)
            q_after = 0 if new_entry else q
            path.append(
                {
                    "t": s + local,
                    "pb": p,
                    "pa": a,
                    "qb": q,
                    "qa": q_after,
                    "ne": new_entry,
                    "si": si,
                    "local": local,
                }
            )
            p = a
            if (s + local) < e - 1:
                q = 1 if not bool(proximity_any[s + local + 1]) else q_after
        if p != 0:
            raise AssertionError("unit did not terminate flat")

        for d in path:
            t = int(d["t"])
            si = int(d["si"])
            local = int(d["local"])
            valid[t] = True
            dec["proximity_bits"][t] = int(proximity_bits[t])
            dec["proximity_any"][t] = bool(proximity_any[t])
            dec["proximity_episode_id"][t] = int(prox_ep_id[t])
            dec["position_before"][t] = d["pb"]
            dec["position_after"][t] = d["pa"]
            dec["entry_right_before"][t] = d["qb"]
            dec["entry_right_after"][t] = d["qa"]
            dec["new_entry_consumed"][t] = d["ne"]
            dec["transition"][t] = _transition_label(d["pb"], d["pa"])
            for sj, label in enumerate(STATE_LABELS):
                for a_idx, a_letter in enumerate(("s", "f", "l")):
                    dec[f"q_{label}_{a_letter}"][t] = float(Q[local, sj, a_idx])
                dec[f"best_{label}"][t] = actions[local, sj]
                dec[f"edge_{label}"][t] = edges[local, sj]
                dec[f"amb_{label}"][t] = bool(ambg[local, sj])
            dec["ambiguous"][t] = bool(ambg[local, si])
            dec["terminal_reason"][t] = term
            dec["training_eligible"][t] = eligible
            dec["label_available_time"][t] = lav

        _append_unit_trades(
            trades,
            symbol,
            td_arr,
            times,
            opens,
            highs,
            lows,
            proximity_bits,
            prox_ep_id,
            cost,
            path,
            term,
            eligible,
            e,
        )

    sel = np.flatnonzero(valid)
    return {
        "symbol": symbol,
        "n": n,
        "open": opens,
        "high": highs,
        "low": lows,
        "time": times,
        "trading_day": td_arr,
        "segment": seg_arr,
        "proximity_bits": proximity_bits,
        "proximity_any": proximity_any,
        "proximity_episode_id": prox_ep_id,
        "cost_points": cost,
        "sel": sel,
        "decision": dec,
        "starts": starts,
        "ends": ends,
        "units": units,
        "unit_values": unit_values,
        "trades": trades,
    }


def run_dp_m15_overnight_teacher(
    symbol: str,
    *,
    cost_points: Optional[np.ndarray] = None,
    max_bars: Optional[int] = None,
    counters: Optional[KernelCounters] = None,
) -> Dict[str, Any]:
    """Single-pass overnight Teacher runner for one symbol."""
    counters = counters or KernelCounters()
    prox = build_dp_proximity_m15(symbol, max_bars)
    return _dp_from_proximity(symbol, prox, counters, cost_points=cost_points)


def map_candidates_to_trades(
    candidates: pd.DataFrame,
    trades: pd.DataFrame,
    segment: np.ndarray,
) -> pd.DataFrame:
    """Map each 15m Candidate to the first valid Oracle trade that ends after it.

    Rule (spec #8 / #21): within the SAME hard segment, map a Candidate to the FIRST
    valid Oracle trade whose ``exit_fill_index`` is strictly greater than the
    Candidate's ``candidate_fill_index``.

    A trade is "valid" for training iff ``training_eligible`` is True
    (i.e. terminal_reason in {OPTIMAL_FLAT, REVERSAL}); HARD_BOUNDARY / DATA_END
    exits are excluded.

    CRITICAL: the search is scoped per hard segment, so a Candidate can NEVER map to
    a trade in a different segment. This enforces "mapping may never cross segment".

    Returns a DataFrame aligned to ``candidates`` (same row order) with columns:
        mapped, trade_id, direction, exit_fill_index, exit_fill_price,
        terminal_reason, training_eligible, drop_reason.
    """
    cand = candidates.reset_index(drop=True)
    cand_fill = cand["candidate_fill_index"].to_numpy(np.int64)
    cand_seg = segment[cand_fill]

    tdf = trades.reset_index(drop=True)
    t_entry = tdf["entry_fill_index"].to_numpy(np.int64)
    t_exit = tdf["exit_fill_index"].to_numpy(np.int64)
    t_seg = segment[t_entry]  # == segment[t_exit] by cross_segment invariant

    order = np.argsort(t_exit, kind="stable")
    t_exit_sorted = t_exit[order]

    rows = []
    for i in range(len(cand)):
        ci = int(cand_fill[i])
        cs = int(cand_seg[i])
        same = np.flatnonzero(t_seg == cs)
        if same.size == 0:
            rows.append((False, None, None, None, None, None, False, "CENSORED_NO_FUTURE_TEACHER"))
            continue
        se = t_exit_sorted  # already sorted globally; restrict below
        so = order
        # restrict to same segment
        same_mask = t_seg[so] == cs
        se_s = se[same_mask]
        so_s = so[same_mask]
        j = np.searchsorted(se_s, ci, side="right")  # first exit > ci
        if j >= len(se_s):
            rows.append((False, None, None, None, None, None, False, "CENSORED_NO_FUTURE_TEACHER"))
            continue
        k = int(so_s[j])
        tr = tdf.iloc[k]
        elig = bool(tr["training_eligible"])
        reason = str(tr["terminal_reason"])
        if not elig:
            rows.append((False, str(tr["trade_id"]), str(tr["direction"]),
                         int(tr["exit_fill_index"]), float(tr["exit_fill_price"]),
                         reason, elig, "INELIGIBLE_TEACHER_EXIT"))
        else:
            rows.append((True, str(tr["trade_id"]), str(tr["direction"]),
                         int(tr["exit_fill_index"]), float(tr["exit_fill_price"]),
                         reason, elig, "OK"))

    cols = ["mapped", "trade_id", "direction", "exit_fill_index", "exit_fill_price",
            "terminal_reason", "training_eligible", "drop_reason"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Artifact frames / metadata / writer / reader (Teacher-owned)                 #
# --------------------------------------------------------------------------- #
# Frame construction reuses the frozen baseline's build_artifact_frames_v1 so the
# Teacher artifact schema is byte-identical to the proven Phase-0 format; only the
# Teacher-owned metadata (TASK_ID / objective / terminal_policy) differs.
build_artifact_frames = build_artifact_frames_v1


def artifact_metadata(
    oracle_source_sha: str,
    symbol: str,
    data_start: Any = None,
    data_end: Any = None,
    *,
    cost_mode: str = "zero_cost",
    generated_at: Optional[str] = None,
    row_count_actions: int = 0,
    row_count_trades: int = 0,
    teacher_contract_id: str = TASK_ID,
    teacher_source_git_sha: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "task_id": TASK_ID,
        "math_version": MATH_VERSION,
        "oracle_source_sha": oracle_source_sha,
        "teacher_contract_id": teacher_contract_id,
        "teacher_source_git_sha": teacher_source_git_sha or oracle_source_sha,
        "symbol": symbol,
        "data_start": str(data_start),
        "data_end": str(data_end),
        "objective": "gross_open_to_open_pnl_overnight_teacher",
        "execution_clock": "15m",
        "cost_mode": cost_mode,
        "entry_proximity_atr": ENTRY_PROX_ATR,
        "tf_universe": ["m15", "h1", "h4"],
        "entry_semantics": (
            "pre_existing_SR_LIQ_within_alpha_ATR; "
            "one_new_entry_per_continuous_proximity_episode"
        ),
        "terminal_policy": (
            "overnight_teacher: unit = hard segment; "
            "HARD_BOUNDARY/DATA_END forced exits are ineligible"
        ),
        "generated_at": generated_at or datetime.now(timezone.utc).isoformat(),
        "row_count_actions": int(row_count_actions),
        "row_count_trades": int(row_count_trades),
    }


def write_oracle_artifact(
    result: Dict[str, Any],
    root: Any,
    *,
    oracle_source_sha: str,
    generated_at: Optional[str] = None,
) -> Path:
    symbol = result["symbol"]
    outdir = Path(root) / symbol
    outdir.mkdir(parents=True, exist_ok=True)

    frames = build_artifact_frames(result)
    actions = frames["oracle_actions"]
    trades = frames["oracle_trades"]
    actions.to_parquet(outdir / ORACLE_ACTIONS_FILE, index=False)
    trades.to_parquet(outdir / ORACLE_TRADES_FILE, index=False)

    times = result["time"]
    meta = artifact_metadata(
        oracle_source_sha, symbol,
        data_start=pd.Timestamp(times[0]), data_end=pd.Timestamp(times[-1]),
        generated_at=generated_at,
        row_count_actions=len(actions), row_count_trades=len(trades),
        teacher_contract_id=TEACHER_CONTRACT_ID,
        teacher_source_git_sha=oracle_source_sha,
    )
    (outdir / ORACLE_METADATA_FILE).write_text(json.dumps(meta, indent=2, default=str))
    return outdir


def load_oracle_artifact(
    root: Any,
    symbol: str,
    *,
    expected_math_version: Optional[str] = MATH_VERSION,
    expected_source_sha: Optional[str] = None,
) -> Dict[str, Any]:
    def _fail(reason: str) -> Dict[str, Any]:
        return {"ok": False, "reason": reason, "actions": None, "trades": None,
                "metadata": None}

    outdir = Path(root) / symbol
    ap = outdir / ORACLE_ACTIONS_FILE
    tp = outdir / ORACLE_TRADES_FILE
    mp = outdir / ORACLE_METADATA_FILE
    if not (ap.exists() and tp.exists() and mp.exists()):
        return _fail("missing_artifact")

    try:
        meta = json.loads(mp.read_text())
    except (OSError, ValueError):
        return _fail("missing_metadata")
    if not isinstance(meta, dict) or "math_version" not in meta:
        return _fail("missing_metadata")
    if expected_math_version is not None and meta.get("math_version") != expected_math_version:
        return _fail("math_version_mismatch")
    if meta.get("symbol") != symbol:
        return _fail("symbol_mismatch")
    if expected_source_sha is not None and meta.get("oracle_source_sha") != expected_source_sha:
        return _fail("source_sha_mismatch")

    try:
        actions = pd.read_parquet(ap)
        trades = pd.read_parquet(tp)
    except (OSError, ValueError):
        return _fail("unreadable_artifact")

    rc_actions = meta.get("row_count_actions")
    rc_trades = meta.get("row_count_trades")
    if rc_actions is not None and len(actions) != int(rc_actions):
        return _fail("row_count_mismatch")
    if rc_trades is not None and len(trades) != int(rc_trades):
        return _fail("row_count_mismatch")

    missing = [c for c in ORACLE_VIEWER_TRADE_COLUMNS if c not in trades.columns]
    if missing:
        return _fail("missing_trade_columns")

    return {"ok": True, "reason": None, "actions": actions, "trades": trades,
            "metadata": meta}


# --------------------------------------------------------------------------- #
# Mechanical invariant checker (used by tests + audits)                         #
# --------------------------------------------------------------------------- #
def check_oracle_invariants(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return counts of invariant violations (all should be 0).

    Overnight variant: cross_day is EXPECTED (overnight holding), so it is only
    reported as an informational count, not a violation. cross_segment remains a
    hard invariant (a trade must never span a hard segment boundary).

    ``nonflat_terminal`` and ``pnl_mismatch_flag`` are REAL mechanical checks:
      * nonflat_terminal: every unit's forced-flat decision (at e-1) must have
        position_after == Flat; otherwise the Teacher leaked a non-flat terminal.
      * pnl_mismatch_flag: the DP total value (sum of per-unit optimal values)
        must equal the sum of realized trade gross points, within 1e-6.
    """
    dec = result["decision"]
    prox = result["proximity_any"]
    ep = dec["proximity_episode_id"]
    trades = result["trades"]

    out: Dict[str, Any] = {k: 0 for k in (
        "new_entry_outside_proximity",
        "new_entry_armed_zero",
        "illegal_reversal",
        "cross_day",
        "cross_segment",
        "nonflat_terminal",
        "pnl_mismatch_flag",
    )}

    sel = result["sel"]
    per_ep: Dict[int, int] = {}
    for t in sel:
        t = int(t)
        ne = bool(dec["new_entry_consumed"][t])
        if not ne:
            continue
        if not bool(prox[t]):
            out["new_entry_outside_proximity"] += 1
        if int(dec["entry_right_before"][t]) != 1:
            out["new_entry_armed_zero"] += 1
        eid = int(ep[t])
        per_ep[eid] = per_ep.get(eid, 0) + 1
        pb = int(dec["position_before"][t])
        pa = int(dec["position_after"][t])
        if pb != 0 and pa == -pb:
            if not (bool(prox[t]) and int(dec["entry_right_before"][t]) == 1):
                out["illegal_reversal"] += 1

    out["max_new_entries_per_episode"] = max(per_ep.values()) if per_ep else 0

    if trades:
        td = pd.to_datetime(result["trading_day"]).to_numpy()
        seg = result["segment"]
        for tr in trades:
            ei, xi = int(tr["entry_fill_index"]), int(tr["exit_fill_index"])
            if td[ei] != td[xi]:
                out["cross_day"] += 1
            if seg[ei] != seg[xi]:
                out["cross_segment"] += 1

    # non-flat unit terminals: the forced-flat decision is at e-1 of each unit.
    for u in result["units"]:
        e = int(u["seg_end"])
        if e - int(u["seg_start"]) < 2:
            continue
        if int(dec["position_after"][e - 1]) != 0:
            out["nonflat_terminal"] += 1

    # PnL reconstruction: the Bellman total value (sum of per-unit optimal
    # values) must equal the sum of realized trade NET points (gross - cost).
    # Under zero cost gross == net, but the canonical Bellman is PnL - cost, so
    # this reconciles against net_points so it stays correct once cost is real.
    uv = result.get("unit_values")
    total_val = float(np.sum(np.asarray(uv, dtype=float))) if uv else 0.0
    total_gross = float(sum(float(t["gross_points"]) for t in trades))
    total_cost = float(sum(float(t["cost_points"]) for t in trades))
    total_net = float(sum(float(t["net_points"]) for t in trades))
    out["gross_points_total"] = total_gross
    out["cost_points_total"] = total_cost
    out["net_points_total"] = total_net
    out["pnl_abs_diff"] = abs(total_val - total_net)
    if out["pnl_abs_diff"] > 1e-6:
        out["pnl_mismatch_flag"] = 1

    return out


if __name__ == "__main__":
    import sys
    syms = sys.argv[1:] or ["AG"]
    for s in syms:
        r = run_dp_m15_overnight_teacher(s)
        inv = check_oracle_invariants(r)
        print(f"[{s}] trades={len(r['trades'])} "
              f"cross_segment={inv['cross_segment']} "
              f"new_entry_outside_prox={inv['new_entry_outside_proximity']} "
              f"illegal_reversal={inv['illegal_reversal']}")
