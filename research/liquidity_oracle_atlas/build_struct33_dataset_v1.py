"""build_struct33_dataset_v1
===========================

Phase 1 dataset builder: ONE AG Candidate -> Teacher label table.

EFFICIENCY CONTRACT (enforced by structure, not just docs):
  1. run_environment_m15("AG", capture_provenance=False) EXACTLY ONCE.
  2. derive_m15_candidate_gate ONCE from that environment's touch_bits.
  3. load_oracle_artifact (verified overnight Teacher). NO DP rerun.
  4. No recompute of ATR / DTP / SR / Liquidity / 15m bars.
  5. STRUCT33 = direct columns from the canonical environment (3 TF x 11).
  6. All alignment via integer execution indices; no DataFrame merge.
  7. Candidate -> Teacher mapping uses per-segment np.searchsorted.
  8. map_candidates_to_trades() (in the Teacher module) is reference-only and
     is NOT imported here; the differential test compares against it.
  9. NumPy gather for Teacher fields + label construction (no per-row iloc).
 10. Weight via np.unique(return_inverse, return_counts). No per-trade loop.

This module deliberately does NOT import run_dp_m15_overnight_teacher; loading
the artifact is the only Teacher interaction. (Guarded by
test_no_dp_rerun_inside_builder.)

Thoroughness rule: this is an index / map / label-assembly task. It computes no
indicator math, trains nothing, and does not modify Teacher or Candidate.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    compute_candidate_episode_id,
    derive_m15_candidate_gate,
    run_environment_m15,
)
from research.liquidity_oracle_atlas.build_teacher_oracle_dp_m15_overnight_v1 import (
    ARTIFACT_ROOT,
    load_oracle_artifact,
)

TASK_ID = "FUTURE-R4-M15-STRUCT33-DATASET-V1-PHASE1-AG"
BASE_SHA = "29e52876fb894b40046fcd2e4035cb66eb931c1d"

TF_ORDER = ("m15", "h1", "h4")

FEATURES_PER_TF = (
    "trend_state",
    "slope_atr",
    "dev",
    "sr_support_dist_atr",
    "sr_resistance_dist_atr",
    "sr_support_strength",
    "sr_resistance_strength",
    "liq_up_dist_atr",
    "liq_down_dist_atr",
    "liq_up_count",
    "liq_down_count",
)

STRUCT33 = tuple(
    f"{tf}_{name}" for tf in TF_ORDER for name in FEATURES_PER_TF
)
DTP9 = tuple(
    f"{tf}_{name}"
    for tf in TF_ORDER
    for name in ("trend_state", "slope_atr", "dev")
)

assert len(DTP9) == 9
assert len(STRUCT33) == 33
assert set(DTP9).issubset(STRUCT33)

# Second layer of defense: no model / path / trigger column may sneak in.
# NOTE: "q_" is intentionally NOT here -- the canonical liq_up_*/liq_down_* STRUCT33
# fields contain "q_". The authoritative guard is the exact-schema assertion
# `assert tuple(X.columns) == STRUCT33` immediately below, which would fail on any
# non-STRUCT33 column (including any Q-value column) regardless.
FORBIDDEN_FEATURE_SUBSTRINGS = (
    "m5", "oracle", "future", "entry_quality", "bars_to_", "dp_proximity",
    "candidate_", "trigger", "episode", "wick", "body", "breach",
    "accept", "reclaim",
)


# --------------------------------------------------------------------------- #
# Vectorized Candidate -> Teacher mapper (production path)                       #
# --------------------------------------------------------------------------- #
def map_candidates_fast(
    candidate_fill_idx: np.ndarray,
    candidate_valid: np.ndarray,
    segment: np.ndarray,
    trades: pd.DataFrame,
) -> np.ndarray:
    """Map each Candidate to the first future Teacher trade in the same segment.

    Returns the mapped Teacher row index; -1 means no future Teacher trade in the
    same hard segment (censored).

    Semantics (matches the reference helper):
        first trade with exit_fill_index > candidate_fill_idx,
        restricted to the same hard segment.

    Complexity ~ O(T log T + C log T) via per-segment np.searchsorted, NOT the
    O(C x T) per-Candidate scan of the reference implementation.
    """
    m = len(candidate_fill_idx)
    mapped = np.full(m, -1, dtype=np.int64)

    t_entry = trades["entry_fill_index"].to_numpy(np.int64)
    t_exit = trades["exit_fill_index"].to_numpy(np.int64)
    # Teacher invariant: entry/exit share a hard segment.
    t_seg = segment[t_entry]

    cand_seg = np.full(m, -1, dtype=np.int64)
    v = np.flatnonzero(candidate_valid)
    cand_seg[v] = segment[candidate_fill_idx[v]]

    common_segments = np.intersect1d(
        np.unique(cand_seg[v]),
        np.unique(t_seg),
        assume_unique=False,
    )

    # Only loop over hard segments; in AG this is effectively one iteration.
    for seg_id in common_segments:
        ci = np.flatnonzero(candidate_valid & (cand_seg == seg_id))
        ti = np.flatnonzero(t_seg == seg_id)
        if ci.size == 0 or ti.size == 0:
            continue

        order = np.argsort(t_exit[ti], kind="stable")
        ti_sorted = ti[order]
        exits = t_exit[ti_sorted]

        pos = np.searchsorted(exits, candidate_fill_idx[ci], side="right")
        ok = pos < len(exits)
        mapped[ci[ok]] = ti_sorted[pos[ok]]

    return mapped


# --------------------------------------------------------------------------- #
# Statistics helpers                                                            #
# --------------------------------------------------------------------------- #
def _quantiles(arr: np.ndarray, qs) -> dict:
    a = arr[np.isfinite(arr)]
    if a.size == 0:
        return {f"p{int(q * 100)}": None for q in qs}
    return {f"p{int(q * 100)}": float(np.percentile(a, q * 100)) for q in qs}


# --------------------------------------------------------------------------- #
# Main builder                                                                 #
# --------------------------------------------------------------------------- #
def build_struct33_dataset(
    symbol: str,
    *,
    teacher_root: str = ARTIFACT_ROOT,
    out_dir: str = "artifacts/struct33_dataset_v1",
    evidence_dir: str = "research/liquidity_oracle_atlas/evidence",
    random_state: int = 20260923,
) -> dict:
    # --- 1. environment run ONCE ------------------------------------------
    env = run_environment_m15(symbol, capture_provenance=False)
    exec_frame = env["exec_frame"]
    features = env["features"]
    touch_bits = env["touch_bits"]

    n = len(exec_frame)
    segment = exec_frame["segment"].to_numpy(np.int64)
    trading_day = exec_frame["trading_day"].to_numpy()

    bar_start = pd.to_datetime(exec_frame["bar_start_time"]).to_numpy()
    open_px = exec_frame["open"].to_numpy(float)

    # --- 2. candidate gate ONCE (no second Candidate source) --------------
    gate = derive_m15_candidate_gate(touch_bits, segment, trading_day)
    candidate_any = gate["candidate_any"]
    candidate_idx = np.flatnonzero(candidate_any)          # == decision indices
    m = len(candidate_idx)
    candidate_trigger_bits = gate["candidate_trigger_bits"][candidate_idx]
    episode_id = compute_candidate_episode_id(
        candidate_any, gate["same_unit"]
    )[candidate_idx]

    decision_idx = candidate_idx
    fill_idx = decision_idx + 1
    has_next_bar = fill_idx < n

    fill_price = np.full(m, np.nan, dtype=float)
    fill_time = np.full(m, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    same_segment_fill = np.zeros(m, dtype=bool)
    safe = np.flatnonzero(has_next_bar)
    fill_price[safe] = open_px[fill_idx[safe]]
    fill_time[safe] = bar_start[fill_idx[safe]]
    same_segment_fill[safe] = (
        segment[decision_idx[safe]] == segment[fill_idx[safe]]
    )

    # --- 3. load verified Teacher (NO DP rerun) ---------------------------
    art = load_oracle_artifact(teacher_root, symbol)
    if not art["ok"]:
        raise RuntimeError(f"STOP_PHASE1_TEACHER_ARTIFACT: {art['reason']}")
    trades = art["trades"].reset_index(drop=True)

    # --- 4. vectorized mapping --------------------------------------------
    mapped = map_candidates_fast(fill_idx, has_next_bar, segment, trades)

    # --- 5. gather Teacher fields (NumPy) ---------------------------------
    t_trade_id = trades["trade_id"].to_numpy(object)
    t_direction = trades["direction"].to_numpy(object)
    t_entry_idx = trades["entry_fill_index"].to_numpy(np.int64)
    t_exit_idx = trades["exit_fill_index"].to_numpy(np.int64)
    t_entry_price = trades["entry_fill_price"].to_numpy(float)
    t_exit_price = trades["exit_fill_price"].to_numpy(float)
    t_entry_time = trades["entry_fill_time"].to_numpy()
    t_exit_time = trades["exit_fill_time"].to_numpy()
    t_terminal = trades["terminal_reason"].to_numpy(object)
    t_eligible = trades["training_eligible"].to_numpy(bool)

    mapped_ok = mapped >= 0
    j = np.flatnonzero(mapped_ok)
    k = mapped[j]

    teacher_row_index = np.full(m, -1, dtype=np.int64)
    oracle_trade_id = np.full(m, None, dtype=object)
    oracle_direction = np.full(m, None, dtype=object)
    oracle_entry_idx = np.full(m, -1, dtype=np.int64)
    oracle_exit_idx = np.full(m, -1, dtype=np.int64)
    oracle_entry_price = np.full(m, np.nan, dtype=float)
    oracle_exit_price = np.full(m, np.nan, dtype=float)
    oracle_entry_time = np.full(m, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    oracle_exit_time = np.full(m, np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    oracle_terminal_reason = np.full(m, None, dtype=object)
    teacher_eligible = np.zeros(m, dtype=bool)

    teacher_row_index[j] = k
    oracle_trade_id[j] = t_trade_id[k]
    oracle_direction[j] = t_direction[k]
    oracle_entry_idx[j] = t_entry_idx[k]
    oracle_exit_idx[j] = t_exit_idx[k]
    oracle_entry_price[j] = t_entry_price[k]
    oracle_exit_price[j] = t_exit_price[k]
    oracle_entry_time[j] = t_entry_time[k]
    oracle_exit_time[j] = t_exit_time[k]
    oracle_terminal_reason[j] = t_terminal[k]
    teacher_eligible[j] = t_eligible[k]

    # --- 6. EntryQualityATR (vectorized) ---------------------------------
    candidate_atr = features["m15_atr"].to_numpy(float)[decision_idx]
    direction_sign = np.zeros(m, dtype=float)
    direction_sign[oracle_direction == "LONG"] = 1.0
    direction_sign[oracle_direction == "SHORT"] = -1.0

    bad_atr = ~np.isfinite(candidate_atr) | (candidate_atr <= 0)
    entry_quality_atr = np.full(m, np.nan, dtype=float)
    bars_to_oracle_entry = np.full(m, np.nan, dtype=float)
    bars_to_oracle_exit = np.full(m, np.nan, dtype=float)

    label_ok = (
        mapped_ok
        & teacher_eligible
        & same_segment_fill
        & np.isfinite(candidate_atr)
        & (candidate_atr > 0)
    )
    e = np.flatnonzero(label_ok)
    entry_quality_atr[e] = (
        direction_sign[e]
        * (oracle_exit_price[e] - fill_price[e])
        / candidate_atr[e]
    )
    bars_to_oracle_entry[e] = oracle_entry_idx[e] - fill_idx[e]
    bars_to_oracle_exit[e] = oracle_exit_idx[e] - fill_idx[e]

    # --- 7. weights (np.unique, no per-trade loop) -----------------------
    sample_weight_raw = np.full(m, np.nan, dtype=float)
    if e.size:
        teacher_rows = mapped[e]
        _, inv, counts = np.unique(
            teacher_rows, return_inverse=True, return_counts=True
        )
        sample_weight_raw[e] = 1.0 / counts[inv]
        weight_sum_by_trade = np.bincount(inv, weights=sample_weight_raw[e])
        assert np.allclose(weight_sum_by_trade, 1.0, atol=1e-12)

    # --- 8. STRUCT33 direct selection (integer alignment, no merge) ------
    X = (
        features.loc[:, list(STRUCT33)]
        .iloc[decision_idx]
        .reset_index(drop=True)
    )
    assert tuple(X.columns) == STRUCT33
    for sub in FORBIDDEN_FEATURE_SUBSTRINGS:
        assert not any(sub in c for c in X.columns), f"forbidden feature: {sub}"

    # --- 9. label_status fail-closed -------------------------------------
    status = np.full(m, "OK", dtype=object)
    status[~has_next_bar] = "CENSORED_NO_NEXT_BAR"
    status[has_next_bar & ~same_segment_fill] = "CENSORED_FILL_CROSSES_HARD_SEGMENT"
    status[same_segment_fill & ~mapped_ok] = "CENSORED_NO_FUTURE_TEACHER"
    status[mapped_ok & ~teacher_eligible] = "INELIGIBLE_TEACHER_EXIT"
    status[mapped_ok & teacher_eligible & same_segment_fill & bad_atr] = (
        "CENSORED_BAD_ATR"
    )
    label_eligible = status == "OK"

    # --- 10. assemble dataset --------------------------------------------
    ds = pd.DataFrame(
        {
            "symbol": symbol,
            "candidate_decision_index": decision_idx.astype(np.int64),
            "candidate_decision_time": bar_start[decision_idx],
            "candidate_fill_index": fill_idx.astype(np.int64),
            "candidate_fill_time": fill_time,
            "candidate_fill_price": fill_price,
            "candidate_episode_id": episode_id.astype(np.int64),
            "candidate_trigger_bits": candidate_trigger_bits.astype(np.uint8),
            "teacher_row_index": teacher_row_index,
            "oracle_trade_id": oracle_trade_id,
            "oracle_direction": oracle_direction,
            "oracle_entry_fill_index": oracle_entry_idx,
            "oracle_entry_fill_time": oracle_entry_time,
            "oracle_entry_fill_price": oracle_entry_price,
            "oracle_exit_fill_index": oracle_exit_idx,
            "oracle_exit_fill_time": oracle_exit_time,
            "oracle_exit_fill_price": oracle_exit_price,
            "oracle_terminal_reason": oracle_terminal_reason,
            "bars_to_oracle_entry": bars_to_oracle_entry,
            "bars_to_oracle_exit": bars_to_oracle_exit,
            "entry_quality_atr": entry_quality_atr,
            "sample_weight_raw": sample_weight_raw,
            "label_eligible": label_eligible,
            "label_status": status,
        }
    )
    ds = pd.concat([ds, X], axis=1)

    # --- 11. statistics (label health only, no model metrics) -------------
    stats = _compute_stats(
        m=m,
        has_next_bar=has_next_bar,
        same_segment_fill=same_segment_fill,
        mapped=mapped,
        mapped_ok=mapped_ok,
        teacher_eligible=teacher_eligible,
        label_eligible=label_eligible,
        label_ok=label_ok,
        oracle_trade_id=oracle_trade_id,
        oracle_direction=oracle_direction,
        entry_quality_atr=entry_quality_atr,
        bars_to_oracle_entry=bars_to_oracle_entry,
        bars_to_oracle_exit=bars_to_oracle_exit,
        sample_weight_raw=sample_weight_raw,
        X=X,
    )

    # --- 12. deterministic review sample (48 rows) ------------------------
    review_sample = _build_review_sample(
        ds, label_ok, random_state=random_state
    )

    # --- 13. write artifacts (parquet not committed) ----------------------
    out_path = os.path.join(out_dir, symbol)
    os.makedirs(out_path, exist_ok=True)
    parquet_path = os.path.join(out_path, "candidate_teacher_dataset.parquet")
    ds.to_parquet(parquet_path, index=False)

    meta = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "symbol": symbol,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "teacher_artifact": {
            "math_version": art["metadata"].get("math_version"),
            "oracle_source_sha": art["metadata"].get("oracle_source_sha"),
            "data_start": art["metadata"].get("data_start"),
            "data_end": art["metadata"].get("data_end"),
            "row_count_trades": art["metadata"].get("row_count_trades"),
        },
        "struct33": list(STRUCT33),
        "dtp9": list(DTP9),
        "n_candidates": m,
        "n_label_eligible": int(label_eligible.sum()),
        "efficiency": {
            "environment_runs": 1,
            "dp_reruns": 0,
            "teacher_reloaded": True,
            "candidate_source": "run_environment_m15 touch_bits (single pass)",
        },
    }
    with open(os.path.join(out_path, "metadata.json"), "w") as fh:
        json.dump(meta, fh, indent=2, default=str)

    os.makedirs(evidence_dir, exist_ok=True)
    summary_path = os.path.join(
        evidence_dir, f"phase1_{symbol}_summary.json"
    )
    with open(summary_path, "w") as fh:
        json.dump(
            {"meta": meta, "stats": stats}, fh, indent=2, default=str
        )
    review_path = os.path.join(
        evidence_dir, f"phase1_{symbol}_review_sample.csv"
    )
    review_sample.to_csv(review_path, index=False)

    return {
        "dataset": ds,
        "stats": stats,
        "meta": meta,
        "review_sample": review_sample,
        "paths": {
            "parquet": parquet_path,
            "metadata": os.path.join(out_path, "metadata.json"),
            "summary": summary_path,
            "review_sample": review_path,
        },
    }


def _compute_stats(
    *,
    m,
    has_next_bar,
    same_segment_fill,
    mapped,
    mapped_ok,
    teacher_eligible,
    label_eligible,
    label_ok,
    oracle_trade_id,
    oracle_direction,
    entry_quality_atr,
    bars_to_oracle_entry,
    bars_to_oracle_exit,
    sample_weight_raw,
    X,
):
    n_mapped = int(mapped_ok.sum())
    n_no_future = int((mapped == -1).sum())
    n_ineligible = int((mapped_ok & ~teacher_eligible).sum())
    n_valid_fill = int(has_next_bar.sum())
    n_hardseg_censored = int((has_next_bar & ~same_segment_fill).sum())

    dir_counts = {}
    if label_eligible.any():
        d = oracle_direction[label_eligible]
        for v in ("LONG", "SHORT"):
            dir_counts[v] = int((d == v).sum())
        dir_counts["ratio_LONG"] = (
            dir_counts["LONG"] / label_eligible.sum()
        )

    unique_trades_mapped = (
        int(len(pd.unique(oracle_trade_id[mapped_ok])))
        if n_mapped else 0
    )
    unique_trades_eligible = (
        int(len(pd.unique(oracle_trade_id[label_eligible])))
        if label_eligible.any() else 0
    )

    # density: candidates per Oracle trade (trades that have >=1 candidate)
    density = {f"p{int(q * 100)}": None for q in (0.10, 0.25, 0.50, 0.75, 0.90)}
    density["max"] = None
    if n_mapped:
        per = np.bincount(mapped[mapped_ok])
        nz = per[per > 0].astype(float)
        density = _quantiles(nz, (0.10, 0.25, 0.50, 0.75, 0.90))
        density["max"] = int(nz.max())

    eql = _quantiles(
        entry_quality_atr[label_eligible],
        (0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99),
    )

    be = bars_to_oracle_entry[label_ok]
    boe = bars_to_oracle_exit[label_ok]
    be_sign = {
        "neg": int((be < 0).sum()),
        "zero": int((be == 0).sum()),
        "pos": int((be > 0).sum()),
    } if be.size else {"neg": 0, "zero": 0, "pos": 0}
    boe_q = _quantiles(boe, (0.10, 0.25, 0.50, 0.75, 0.90))

    w = sample_weight_raw[label_eligible]
    weight_min = float(w.min()) if w.size else None
    weight_max = float(w.max()) if w.size else None

    nonnull = {c: float(X[c].notna().mean()) for c in X.columns}

    return {
        "n_candidates": int(m),
        "fill": {
            "valid_next_fill": n_valid_fill,
            "hard_segment_censored": n_hardseg_censored,
        },
        "mapping": {
            "mapped": n_mapped,
            "censored_no_future_teacher": n_no_future,
            "ineligible_teacher_exit": n_ineligible,
        },
        "direction_label_eligible": dir_counts,
        "opportunity": {
            "unique_oracle_trades_mapped": unique_trades_mapped,
            "unique_oracle_trades_eligible": unique_trades_eligible,
        },
        "density_candidates_per_oracle_trade": density,
        "entry_quality_atr": eql,
        "timing_bars_to_oracle_entry_sign": be_sign,
        "timing_bars_to_oracle_exit": boe_q,
        "weight": {"min": weight_min, "max": weight_max},
        "feature_nonnull_rate": nonnull,
    }


def _build_review_sample(ds: pd.DataFrame, label_ok: np.ndarray, *, random_state: int):
    rng = np.random.RandomState(random_state)

    def _pick(mask, k):
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return np.array([], dtype=np.int64)
        if idx.size <= k:
            return idx
        return rng.choice(idx, size=k, replace=False)

    be = ds["bars_to_oracle_entry"].to_numpy()
    boe = ds["bars_to_oracle_exit"].to_numpy()

    g_before = _pick((be > 0) & label_ok, 12)
    g_near = _pick((np.abs(be) <= 2) & label_ok, 12)
    g_in = _pick((be < 0) & (boe > 0) & label_ok, 12)
    g_exit = _pick((boe <= 4) & (boe > 0) & label_ok, 12)

    sel = np.concatenate([g_before, g_near, g_in, g_exit])
    grp = (
        ["before_entry"] * len(g_before)
        + ["near_entry"] * len(g_near)
        + ["in_position"] * len(g_in)
        + ["near_exit"] * len(g_exit)
    )

    cols = [
        "candidate_decision_index",
        "candidate_decision_time",
        "candidate_fill_time",
        "oracle_direction",
        "candidate_fill_price",
        "oracle_entry_fill_time",
        "oracle_entry_fill_price",
        "oracle_exit_fill_time",
        "oracle_exit_fill_price",
        "bars_to_oracle_entry",
        "bars_to_oracle_exit",
        "entry_quality_atr",
    ]
    sample = ds.iloc[sel][cols].copy()
    sample["group"] = grp
    return sample.reset_index(drop=True)


if __name__ == "__main__":
    r = build_struct33_dataset("AG")
    print(json.dumps(r["stats"], indent=2, default=str))
    print("paths:", r["paths"])
