"""dp_label_lifecycle_v1
=========================

FUT-M15-DP-LABEL-VIZ-GATE-01 — DP-label lifecycle logic + audit (PURE / read-only).

This module is a VISUALIZATION + LOGIC-CONFIRMATION helper only. It:

  * NEVER recomputes the DP (build_trade_oracle_dp_m15_one_entry_proximity_v1).
  * NEVER changes the DP math, the candidate gate, or any Layer-2 / label logic.
  * ONLY reads the canonical DP label artifact (oracle_trades / oracle_actions
    parquet) and derives the trade-lifecycle view required by the task.

Owner mapping (frozen, documented in the Evidence Packet):
  * "Candidate Region" in THIS task == the DP's OWN proximity episode that
    authorized the entry. The DP module (`build_trade_oracle_dp_m15_one_entry_proximity_v1`)
    owns a SINGLE-POSITION Bellman path; by construction at most ONE new entry
    is allowed per continuous proximity episode, and a new entry can only occur
    after the position is flat (p == 0) or via an atomic reversal. Therefore:
        - one candidate region  -> exactly one entry          (HARD RULE #2)
        - Exit_i <= Entry_{i+1}  (no overlapping positions)    (HARD RULE #1/#4)
        - same-direction continuation (LONG->LONG) is allowed    (task #7)
  * The R4 Candidate Trading Zones (candidate_gate_r4_m15) are a SEPARATE
    subsystem and are NOT consumed or referenced here (the DP docstring is
    explicit). We do not couple them; the "candidate_region_id" below is the
    DP's `entry_proximity_episode_id`.

The canonical entry/exit owners are reused verbatim:
  * Entry owner  : build_trade_oracle_dp_m15_one_entry_proximity_v1._append_unit_trades.start
  * Exit owner   : build_trade_oracle_dp_m15_one_entry_proximity_v1._append_unit_trades.close
  * DP lifecycle : build_trade_oracle_dp_m15_one_entry_proximity_v1._dp_from_proximity
  * Label artifact: build_artifact_frames_v1 + write_oracle_artifact
                    (artifacts/trade_oracle_dp_m15_one_entry_proximity_v1/<sym>/)
  * Viewer owner : pages/6_Indicator_Viewer.py (Indicator Viewer engine
                   indicator_viewer_v1.add_dp_oracle_overlay / build_figure)

Exit-reason vocabulary is STRICTLY limited to what the canonical DP actually
produces (`transition` at the exit decision): FLAT_EXIT (LONG_EXIT / SHORT_EXIT)
or REVERSAL (LONG_TO_SHORT / SHORT_TO_LONG). This DP has NO TP / STOP / TARGET /
NEXT_CANDIDATE_BOUNDARY reasons — those are NOT invented.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Canonical exit-reason mapping (ONLY values the DP actually emits)             #
# --------------------------------------------------------------------------- #
_FLAT_TRANSITIONS = {"LONG_EXIT", "SHORT_EXIT"}
_REVERSAL_TRANSITIONS = {"LONG_TO_SHORT", "SHORT_TO_LONG"}


def map_exit_reason(transition: Optional[str]) -> str:
    """Map the canonical DP `transition` at the exit decision to a label.

    Only FLAT_EXIT / REVERSAL exist in this DP. No TP/STOP/TARGET invented.
    """
    if transition in _FLAT_TRANSITIONS:
        return "FLAT_EXIT"
    if transition in _REVERSAL_TRANSITIONS:
        return "REVERSAL"
    return str(transition) if transition is not None else "UNKNOWN"


# --------------------------------------------------------------------------- #
# Lifecycle record                                                              #
# --------------------------------------------------------------------------- #
@dataclass
class LifecycleRecord:
    candidate_id: int                      # sequential "Candidate #k"
    candidate_region_id: int              # DP entry_proximity_episode_id
    candidate_time: Any                   # entry decision time
    candidate_anchor_price: float         # reused DP entry_fill_price (canonical anchor)
    direction: str
    entry_time: Any
    entry_price: float
    number_of_entries: int                # == 1 by construction
    exit_time: Any
    exit_price: float
    exit_reason: str                      # FLAT_EXIT / REVERSAL (canonical)
    prev_candidate_id: Optional[int] = None
    prev_candidate_time: Any = None
    prev_exit_time: Any = None
    next_candidate_id: Optional[int] = None
    next_candidate_time: Any = None
    next_entry_time: Any = None
    overlap_with_next: bool = False
    # chart geometry (DP proximity-episode bar span on the decision axis)
    band_lo: int = -1
    band_hi: int = -1
    entry_fill_index: int = -1
    exit_fill_index: int = -1
    valid: bool = True


def _actions_lookup(actions: pd.DataFrame) -> Dict[int, Dict[str, Any]]:
    """Index the canonical actions frame by decision_bar_index."""
    if actions is None or len(actions) == 0:
        return {}
    key = "decision_bar_index"
    if key not in actions.columns:
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for rec in actions.to_dict("records"):
        out[int(rec[key])] = rec
    return out


def build_lifecycle_records(
    trades: pd.DataFrame,
    actions: Optional[pd.DataFrame] = None,
) -> List[LifecycleRecord]:
    """Derive the ordered lifecycle records (one per DP trade = one candidate).

    Pure: reads only the canonical artifact tables; performs no DP math.
    """
    if trades is None or len(trades) == 0:
        return []
    t = trades.sort_values(["entry_fill_index"]).reset_index(drop=True)
    adict = _actions_lookup(actions)

    # entry_proximity_episode_id belongs to the trade row (canonical).
    ep = t["entry_proximity_episode_id"].to_numpy()
    # band span per episode from the canonical actions frame
    band_lo: Dict[int, int] = {}
    band_hi: Dict[int, int] = {}
    if adict:
        # actions carries dp_proximity_episode_id + decision_bar_index
        col = "dp_proximity_episode_id"
        if col in actions.columns:
            a = actions.copy()
            a[col] = a[col].astype("int64")
            for eid, grp in a.groupby(col):
                if int(eid) < 0:
                    continue
                idxs = grp["decision_bar_index"].to_numpy().astype(int)
                band_lo[int(eid)] = int(idxs.min())
                band_hi[int(eid)] = int(idxs.max())

    recs: List[LifecycleRecord] = []
    n = len(t)
    for i in range(n):
        row = t.iloc[i]
        eid = int(row["entry_proximity_episode_id"])
        # entry decision time from actions (canonical) when available
        cand_time = row.get("entry_fill_time")
        if int(row["entry_decision_index"]) in adict:
            cand_time = adict[int(row["entry_decision_index"])].get(
                "decision_time", cand_time
            )
        exit_trans = None
        if int(row["exit_decision_index"]) in adict:
            exit_trans = adict[int(row["exit_decision_index"])].get("transition")
        rec = LifecycleRecord(
            candidate_id=i + 1,
            candidate_region_id=eid,
            candidate_time=pd.Timestamp(cand_time),
            candidate_anchor_price=float(row["entry_fill_price"]),
            direction=str(row["direction"]),
            entry_time=pd.Timestamp(row["entry_fill_time"]),
            entry_price=float(row["entry_fill_price"]),
            number_of_entries=1,
            exit_time=pd.Timestamp(row["exit_fill_time"]),
            exit_price=float(row["exit_fill_price"]),
            exit_reason=map_exit_reason(exit_trans),
            band_lo=band_lo.get(eid, -1),
            band_hi=band_hi.get(eid, -1),
            entry_fill_index=int(row["entry_fill_index"]),
            exit_fill_index=int(row["exit_fill_index"]),
        )
        recs.append(rec)

    # prev / next linkages + overlap flag
    for i, rec in enumerate(recs):
        if i > 0:
            p = recs[i - 1]
            rec.prev_candidate_id = p.candidate_id
            rec.prev_candidate_time = p.candidate_time
            rec.prev_exit_time = p.exit_time
        if i < n - 1:
            nx = recs[i + 1]
            rec.next_candidate_id = nx.candidate_id
            rec.next_candidate_time = nx.candidate_time
            rec.next_entry_time = nx.entry_time
            rec.overlap_with_next = bool(rec.exit_fill_index > nx.entry_fill_index)
    return recs


# --------------------------------------------------------------------------- #
# Auto assertions (task #12: A / B / C / D / E)                                 #
# --------------------------------------------------------------------------- #
def _entries_per_candidate(trades: pd.DataFrame) -> Dict[int, int]:
    if trades is None or len(trades) == 0:
        return {}
    g = trades.groupby("entry_proximity_episode_id").size()
    return {int(k): int(v) for k, v in g.items()}


def run_lifecycle_assertions(
    trades: pd.DataFrame,
    records: List[LifecycleRecord],
) -> Dict[str, Any]:
    """Return pass/fail for gates A..E plus supporting detail."""
    out: Dict[str, Any] = {}
    epc = _entries_per_candidate(trades)
    # A. one candidate -> one entry
    max_ep = max(epc.values()) if epc else 0
    viol_a = [k for k, v in epc.items() if v != 1]
    out["A_one_entry_per_candidate"] = (len(viol_a) == 0)
    out["A_max_entries_per_candidate"] = max_ep
    out["A_violations"] = viol_a

    # ordered fill indices
    t = trades.sort_values(["entry_fill_index"]).reset_index(drop=True)
    ex = t["exit_fill_index"].to_numpy().astype(int)
    en = t["entry_fill_index"].to_numpy().astype(int)
    diro = t["direction"].to_numpy()

    # B. no overlapping trades: exit[i] <= entry[i+1]
    overlap_idx: List[int] = []
    for i in range(len(t) - 1):
        if ex[i] > en[i + 1]:
            overlap_idx.append(int(i))
    out["B_no_overlapping_trades"] = (len(overlap_idx) == 0)
    out["B_overlap_indices"] = overlap_idx

    # C. previous trade closed before new entry (same condition as B, framed as
    #    "the prior trade must be flat before the next entry is allowed")
    out["C_previous_closed_before_new_entry"] = (len(overlap_idx) == 0)
    out["C_violations"] = overlap_idx

    # D. same-direction continuation must be ALLOWED (not flagged as a violation)
    same_dir: List[int] = []
    for i in range(len(t) - 1):
        if str(diro[i]) == str(diro[i + 1]):
            same_dir.append(int(i))
    out["D_same_direction_continuation_count"] = len(same_dir)
    out["D_same_direction_present"] = (len(same_dir) > 0)
    # gate D passes if same-direction continuation exists AND it does not trip B/C
    d_ok = True
    for i in same_dir:
        if ex[i] > en[i + 1]:
            d_ok = False
            break
    out["D_allowed"] = bool(out["D_same_direction_present"] and d_ok)

    # E. one label per candidate (== one trade row per entry_proximity_episode_id)
    label_viol = [k for k, v in epc.items() if v != 1]
    out["E_one_label_per_candidate"] = (len(label_viol) == 0)
    out["E_violations"] = label_viol

    out["all_pass"] = bool(
        out["A_one_entry_per_candidate"]
        and out["B_no_overlapping_trades"]
        and out["C_previous_closed_before_new_entry"]
        and out["D_allowed"]
        and out["E_one_label_per_candidate"]
    )
    return out


# --------------------------------------------------------------------------- #
# FSM re-validation (canonical event sort -> validate_execution_events)         #
# --------------------------------------------------------------------------- #
def event_sort_key(e: Tuple[int, str]) -> Tuple[int, int]:
    """EXIT before ENTRY when both land on the same fill (task #9)."""
    return (int(e[0]), 0 if "EXIT" in e[1] else 1)


def build_execution_events(records: List[LifecycleRecord]) -> List[Tuple[int, str]]:
    """Canonical Entry/Exit event list for the FSM (reused from DP module)."""
    from research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 import (
        validate_execution_events,
    )

    events: List[Tuple[int, str]] = []
    for r in records:
        kind_e = "LONG_ENTRY" if r.direction == "LONG" else "SHORT_ENTRY"
        kind_x = "LONG_EXIT" if r.direction == "LONG" else "SHORT_EXIT"
        events.append((r.entry_fill_index, kind_e))
        events.append((r.exit_fill_index, kind_x))
    events.sort(key=event_sort_key)
    return events, validate_execution_events(events)


# --------------------------------------------------------------------------- #
# Representative manual cases (task #13: Case 1..4)                             #
# --------------------------------------------------------------------------- #
def find_representative_cases(
    trades: pd.DataFrame,
    records: List[LifecycleRecord],
) -> Dict[str, Any]:
    """Return candidate_ids that best demonstrate the four manual cases.

    Case 1 : previous trade exited, THEN next candidate appears (large gap).
    Case 2 : next candidate appears quickly after previous exit (small gap).
    Case 3 : same-direction continuation (LONG->LONG or SHORT->SHORT).
    Case 4 : candidate region re-entered by price many times (longest proximity
             episode) yet produced exactly one entry.
    """
    if not records:
        return {}
    # gaps between consecutive exits and next entries
    gaps = []
    for i in range(len(records) - 1):
        gap = records[i + 1].entry_fill_index - records[i].exit_fill_index
        gaps.append((gap, records[i + 1].candidate_id, i))

    case1 = None  # largest positive gap (exit well before next entry)
    case2 = None  # smallest positive gap
    if gaps:
        pos = [g for g in gaps if g[0] >= 0]
        if pos:
            case1 = max(pos, key=lambda x: x[0])[1]
            case2 = min(pos, key=lambda x: x[0])[1]

    # Case 3: same-direction continuation
    case3 = None
    for i in range(len(records) - 1):
        if records[i].direction == records[i + 1].direction:
            case3 = records[i + 1].candidate_id
            break

    # Case 4: longest proximity episode (most bars in band) -> one entry
    case4 = None
    best_len = -1
    for r in records:
        if r.band_lo >= 0 and r.band_hi >= 0:
            ln = r.band_hi - r.band_lo + 1
            if ln > best_len:
                best_len = ln
                case4 = r.candidate_id

    return {
        "case1_previous_exits_then_next": case1,
        "case2_next_quick_after_exit": case2,
        "case3_same_direction_continuation": case3,
        "case4_longest_proximity_episode_one_entry": case4,
        "case4_episode_bars": best_len if best_len >= 0 else None,
    }


# --------------------------------------------------------------------------- #
# Negative controls (task #16: NC1..NC5)                                       #
# --------------------------------------------------------------------------- #
def _copy_trades(trades: pd.DataFrame) -> pd.DataFrame:
    return trades.copy().reset_index(drop=True)


def _inject_second_entry_same_episode(
    trades: pd.DataFrame, episode_id: int
) -> pd.DataFrame:
    """NC1/NC3: synthesize a SECOND entry in an existing candidate region."""
    t = _copy_trades(trades)
    mask = t["entry_proximity_episode_id"] == episode_id
    if not mask.any():
        return t
    base = t[mask].iloc[0].to_dict()
    # second entry: later fill, same episode, same direction (re-entry concept)
    base["entry_fill_index"] = int(base["entry_fill_index"]) + 1
    base["exit_fill_index"] = int(base["exit_fill_index"]) + 1
    base["entry_fill_time"] = pd.Timestamp(base["entry_fill_time"]) + pd.Timedelta(minutes=15)
    base["exit_fill_time"] = pd.Timestamp(base["exit_fill_time"]) + pd.Timedelta(minutes=15)
    base["trade_id"] = str(base["trade_id"]) + "_INJECTED"
    t = pd.concat([t, pd.DataFrame([base])], ignore_index=True)
    return t


def _inject_overlap(trades: pd.DataFrame, idx_pair: Tuple[int, int]) -> pd.DataFrame:
    """NC2/NC4: pull a later trade's entry BEFORE the previous trade's exit."""
    t = _copy_trades(trades)
    i, j = idx_pair
    # ensure j > i by entry order
    t = t.sort_values(["entry_fill_index"]).reset_index(drop=True)
    ei = t["entry_fill_index"].to_numpy().astype(int)
    # find positions
    pos_i = int(np.flatnonzero(ei == i)[0]) if (ei == i).any() else 0
    # simpler: shift trade j's entry/exit earlier than trade i's exit
    # pick first two trades by entry order
    a = t.iloc[0]
    b = t.iloc[1]
    t.loc[t.index[1], "entry_fill_index"] = int(a["exit_fill_index"]) - 1
    t.loc[t.index[1], "exit_fill_index"] = int(a["exit_fill_index"]) + 1
    t.loc[t.index[1], "entry_fill_time"] = (
        pd.Timestamp(a["exit_fill_time"]) - pd.Timedelta(minutes=15)
    )
    t.loc[t.index[1], "exit_fill_time"] = (
        pd.Timestamp(a["exit_fill_time"]) + pd.Timedelta(minutes=15)
    )
    return t


def run_negative_controls(
    trades: pd.DataFrame,
    records: List[LifecycleRecord],
) -> List[Dict[str, Any]]:
    """Construct corrupted data and confirm the gate behaves correctly.

    NC1/NC3 : a second entry in the same candidate region -> A MUST FAIL.
    NC2/NC4 : overlapping trades -> B/C MUST FAIL.
    NC5     : genuine LONG->LONG continuation -> D MUST PASS (not flagged as
              a direction-alternation violation).
    """
    results: List[Dict[str, Any]] = []

    # pick an episode to corrupt (first trade's episode)
    if len(records):
        eid = records[0].candidate_region_id
        corrupted = _inject_second_entry_same_episode(trades, eid)
        c_recs = build_lifecycle_records(corrupted)
        c_ass = run_lifecycle_assertions(corrupted, c_recs)
        results.append({
            "name": "NC1_two_entries_one_candidate",
            "expect": "A_FAIL",
            "gate_fired": (not c_ass["A_one_entry_per_candidate"]),
            "passed": (not c_ass["A_one_entry_per_candidate"]),
        })
        results.append({
            "name": "NC3_same_candidate_reentry_second_entry",
            "expect": "A_FAIL",
            "gate_fired": (not c_ass["A_one_entry_per_candidate"]),
            "passed": (not c_ass["A_one_entry_per_candidate"]),
        })

    # NC2 / NC4 overlapping
    if len(trades) >= 2:
        corrupted = _inject_overlap(trades, (0, 1))
        c_recs = build_lifecycle_records(corrupted)
        c_ass = run_lifecycle_assertions(corrupted, c_recs)
        overlap_fired = (not c_ass["B_no_overlapping_trades"])
        results.append({
            "name": "NC2_next_entry_before_prev_exit",
            "expect": "B/C_FAIL",
            "gate_fired": overlap_fired,
            "passed": overlap_fired,
        })
        results.append({
            "name": "NC4_overlapping_time_intervals",
            "expect": "B/C_FAIL",
            "gate_fired": overlap_fired,
            "passed": overlap_fired,
        })

    # NC5 same-direction continuation must PASS on the ORIGINAL data
    ass = run_lifecycle_assertions(trades, records)
    results.append({
        "name": "NC5_same_direction_continuation_allowed",
        "expect": "D_PASS",
        "gate_fired": ass["D_allowed"],
        "passed": ass["D_allowed"],
    })
    return results


# --------------------------------------------------------------------------- #
# Evidence-packet summary                                                       #
# --------------------------------------------------------------------------- #
def evidence_summary(
    trades: pd.DataFrame,
    records: List[LifecycleRecord],
    assertions: Dict[str, Any],
    cases: Dict[str, Any],
    neg: List[Dict[str, Any]],
) -> Dict[str, Any]:
    epc = _entries_per_candidate(trades)
    t = trades.sort_values(["entry_fill_index"]).reset_index(drop=True)
    diro = t["direction"].to_numpy()
    same_dir = sum(
        1 for i in range(len(t) - 1) if str(diro[i]) == str(diro[i + 1])
    )
    return {
        "candidate_region_count": int(len(records)),
        "entry_count": int(len(trades)),
        "max_entries_per_candidate": int(max(epc.values())) if epc else 0,
        "overlapping_trade_count": int(
            0 if assertions["B_no_overlapping_trades"] else len(assertions["B_overlap_indices"])
        ),
        "same_direction_continuation_count": int(same_dir),
        "assertions_all_pass": bool(assertions["all_pass"]),
        "negative_controls_all_pass": all(r["passed"] for r in neg),
        "representative_cases": cases,
    }
