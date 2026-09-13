"""5M-GPM1 — G2: Graph Probability -> Target Decision v1

Core question:
    Does G0's graph-probability increment actually help the SAME 5m signals pick a
    BETTER target, and form a BETTER return curve?

This is a TARGET-SELECTION mechanism experiment only. It does NOT search new
features; G0 is frozen. It builds a fully pre-decision full-node table (vectorized),
trains G0 (reused from TB12_HARDENED transition cache) and an Independent EV model
(I0) on full nodes, then compares three target selectors:

    T0  Nearest          (first node in distance order)
    T1  Independent EV   (I0: per-node TARGET/LOSS/CENSOR multinomial, no chain)
    T2  Graph EV         (G0 chain propagation, full graph probability)

Hard governance:
  * TB12_HARDENED transition cache is used ONLY to train G0 + purge. It must NOT be
    used to pick TB2 test targets (it is a survival risk-set: path stops at LOSS/
    CENSOR). The full-node table is built fully pre-decision from `master`.
  * feature/outcome tables are separate; selectors never read outcome columns.
  * No SKIP/enter-skip research; a selector must always pick one target.
  * Stop after WF1. No WF2/WF3, selector, GNN, RL, G1d, parameter/threshold search.

Vectorization contract:
  * The heavy path (signal x price active mask, node expansion, segment math, EV,
    first-hit reward) is numpy-vectorized.
  * Only the final "one position per symbol" state machine is a small Python loop
    (inherent temporal dependency).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# Frozen helpers reused from the G0 runner (NOT modified here).
from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars, load_master, load_contacts, assign_blocks,
    fit_multinomial, purge_train, G0_NUM, G0_CAT, NEXT, LOSS, CENSOR,
    active_mask_at, signal_clusters, label_window, G0_HARDENED_BASELINE,
    signal_metrics,
)

warnings_simple = None

# ---------------------------------------------------------------------------
# Frozen constants (NEVER tuned by results)
# ---------------------------------------------------------------------------
OUT = REPO / "research/analysis_results/5m_graph_target_decision_v1"
CACHE = OUT / "cache"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)

MASTER_PATH = REPO / "research/analysis_results/smc_oracle_atlas_v1/liquidity_master_v1_1.parquet"
CONTACTS_PATH = REPO / "research/analysis_results/smc_oracle_atlas_v1/liquidity_contacts_v1_1.parquet"

OUTCOME_WINDOW = 34
RISK_ATR = 1.0  # frozen structural stop = decision_close - direction*atr0

# Independent-candidate (T1) model — fixed.
I0_NUM = ["cum_distance_R", "risk_R", "rr_ref"]
I0_CAT = ["direction", "symbol", "contact_type"]

FULL_UNIV = ["AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P",
             "RB", "RU", "SC", "SN", "TA"]
BUILDER_VERSION = "G2-VEC-1"
CHUNK_SIGNALS = 512

TARGET, LOSS_I, CENSOR_I = 0, 1, 2  # I0 independent label encoding

# ---------------------------------------------------------------------------
# G0 Feature Attribution — fixed nested multinomials (NLL/Brier only).
# NOT part of the G2 economic PASS/FAIL gate. Answers WHERE G0's statistical
# value comes from, on the same full-node table:
#   F0 = base distance              (cum_distance_R, risk_R)
#   F1 = F0 + local spacing         (+ delta_R)
#   F2 = F0 + chain position        (+ edge_index)
#   G0 = F0 + spacing + position    (+ delta_R + edge_index)
# Common categorical features: direction, symbol, contact_type.
# All four use the SAME uniform Logistic parameters (fit_multinomial default).
# ---------------------------------------------------------------------------
COMMON_CAT = ["direction", "symbol", "contact_type"]
F0_NUM = ["cum_distance_R", "risk_R"]
F0_CAT = COMMON_CAT
F1_NUM = ["cum_distance_R", "risk_R", "delta_R"]
F1_CAT = COMMON_CAT
F2_NUM = ["cum_distance_R", "risk_R"]
F2_CAT = COMMON_CAT + ["edge_index"]
# G0 attribution entry MUST match the hardened G0 spec EXACTLY — same feature set AND
# same column order as the G0 module's G0_NUM/G0_CAT — so the reproduction guard
# (jNLL~1.6809, brier~0.22866) holds to 1e-6. sklearn LogisticRegression(lbfgs, L2) is
# sensitive to numeric-column order within tol, so we reuse G0_NUM/G0_CAT verbatim.
G0_ATTR_NUM = list(G0_NUM)
G0_ATTR_CAT = list(G0_CAT)
ATTR_SPECS = [
    ("F0", F0_NUM, F0_CAT),
    ("F1", F1_NUM, F1_CAT),
    ("F2", F2_NUM, F2_CAT),
    ("G0", G0_ATTR_NUM, G0_ATTR_CAT),
]


# ---------------------------------------------------------------------------
# Segmented numpy helpers (vectorized; no per-group python loops)
# ---------------------------------------------------------------------------
def segment_meta(group_id: np.ndarray):
    """Return (is_start, starts, lengths, edge_index) for each equal group in group_id.

    Uses np.unique inverse so it is ROBUST to gaps (e.g. after a TB2 subset) and does
    NOT require group_id to be a contiguous 0..K-1 sequence. Nodes of the same group_id
    are contiguous as long as group_id is sorted; callers sort by signal_gid.
    """
    group_id = np.asarray(group_id)
    _, inv = np.unique(group_id, return_inverse=True)  # inv contiguous 0..K-1
    is_start = np.r_[True, inv[1:] != inv[:-1]]
    starts = np.flatnonzero(is_start)
    ends = np.r_[starts[1:], len(inv)]
    lengths = ends - starts
    edge_index = np.arange(len(inv)) - np.repeat(starts, lengths)
    return is_start, starts, lengths, edge_index


def segmented_cumsum(x: np.ndarray, starts: np.ndarray, lengths: np.ndarray):
    cs = np.cumsum(x)
    offsets = np.zeros(len(starts), dtype=np.float64)
    m = starts > 0
    offsets[m] = cs[starts[m] - 1]
    return cs - np.repeat(offsets, lengths)


def segmented_cumprod(p: np.ndarray, starts: np.ndarray, lengths: np.ndarray):
    p = np.clip(p, 1e-12, 1.0)
    logp = np.log(p)
    csum = segmented_cumsum(logp, starts, lengths)
    return np.exp(csum)


def segmented_argmax(value: np.ndarray, starts: np.ndarray, lengths: np.ndarray):
    """Index (global) of the max value per segment; ties -> earliest position
    (which, with nodes sorted by distance, is the NEAREST target)."""
    grp = np.repeat(np.arange(len(starts)), lengths)
    pos = np.arange(len(value), dtype=np.int64)
    # sort primarily by value descending, secondarily by position ascending
    order = np.lexsort((pos, -value))
    grp_order = grp[order]
    first = np.unique(grp_order, return_index=True)[1]
    return order[first]


def first_true_index(mask: np.ndarray, axis: int = 0):
    any_hit = mask.any(axis=axis)
    idx = mask.argmax(axis=axis)
    sentinel = mask.shape[axis]
    return np.where(any_hit, idx, sentinel)


# ---------------------------------------------------------------------------
# Master price-group prep (one pass per symbol)
# ---------------------------------------------------------------------------
def prepare_master_price_groups(master_sym: pd.DataFrame) -> dict:
    price = master_sym["price"].to_numpy(np.float64)
    order = np.argsort(price, kind="stable")
    price_sorted = price[order]
    starts = np.r_[0, np.flatnonzero(np.diff(price_sorted) != 0) + 1]
    unique_price = price_sorted[starts]
    available = pd.to_datetime(master_sym["available_time"]).to_numpy()[order]
    fp = pd.to_datetime(master_sym["first_penetration_time"]).to_numpy()[order]
    return dict(unique_price=unique_price, identity_order=order,
                group_starts=starts, available_time=available,
                first_penetration_time=fp)


def active_prices_chunk(decision_time: np.ndarray, master_info: dict) -> np.ndarray:
    """(chunk, n_unique_price) boolean: is the price-cluster active at decision time."""
    av = master_info["available_time"]
    fp = master_info["first_penetration_time"]
    active_identity = (
        (av[None, :] <= decision_time[:, None])
        & (np.isnat(fp)[None, :] | (fp[None, :] > decision_time[:, None]))
    )
    active_price = np.logical_or.reduceat(active_identity, master_info["group_starts"], axis=1)
    return active_price


# ---------------------------------------------------------------------------
# Full-node builder (vectorized), one symbol
# ---------------------------------------------------------------------------
def build_full_nodes_symbol(sym, contacts_sym, master_sym, bars, builder_version):
    """Return (features_df, outcomes_df) for ALL active directional nodes per signal.

    Features carry ONLY decision-time-visible info. Outcomes carry ONLY the realized
    label. They are saved to separate files.
    """
    t0 = time.perf_counter()
    ms_info = prepare_master_price_groups(master_sym)
    price = ms_info["unique_price"]
    n_price = len(price)

    sig = contacts_sym.reset_index(drop=True)
    N = len(sig)
    if N == 0:
        return pd.DataFrame(), pd.DataFrame()

    # decision-time arrays
    dt = pd.to_datetime(sig["decision_time"]).to_numpy()
    dec = sig["decision_price"].to_numpy(np.float64)
    atr0 = sig["atr0"].to_numpy(np.float64)
    stop = sig["stop_price"].to_numpy(np.float64)
    direction = sig["direction"].to_numpy(np.int64)
    j = sig["contact_bar_index"].to_numpy(np.int64)
    symbol = sig["symbol"].to_numpy(object)
    block = sig["block"].to_numpy(object) if "block" in sig.columns else np.array(["TB?"] * N, object)
    tday = sig["dt_day"].to_numpy(object)
    ctype = sig["contact_type"].to_numpy(object)
    sid = (sig["liquidity_id"].astype(str) + "|" + sig["contact_number"].astype(str)).to_numpy(object)

    # per-signal future window (entry bar = j+1)
    start = j + 1
    di = [np.flatnonzero(bars["disc"][start[k]:]) for k in range(N)]
    end = np.array([start[k] + (int(di[k][0]) if len(di[k]) else bars["n"]) for k in range(N)])
    W = np.minimum(end - start, OUTCOME_WINDOW)
    valid_sig = (start >= 0) & (start < bars["n"]) & (W > 0)

    # chunked active mask + node expansion
    feats = []
    outs = []
    for lo in range(0, N, CHUNK_SIGNALS):
        hi = min(lo + CHUNK_SIGNALS, N)
        sl = slice(lo, hi)
        ap = active_prices_chunk(dt[sl], ms_info)            # (chunk, n_price)
        dchunk = direction[sl][:, None]
        dist_px = dchunk * (price[None, :] - dec[sl][:, None])
        valid = ap & (dist_px > 0)
        sr, pc = np.nonzero(valid)
        if len(sr) == 0:
            continue
        sr = sr + lo  # global signal index
        dist_R = dist_px[sr - lo, pc] / atr0[sr]
        target_price = price[pc]
        # sort within signal by ascending distance_R (nearest first)
        ord_ = np.lexsort((dist_R, sr))
        sr = sr[ord_]
        pc = pc[ord_]
        dist_R = dist_R[ord_]
        target_price = target_price[ord_]
        is_start, starts, lengths, edge_index = segment_meta(sr)
        prev = np.empty_like(dist_R)
        prev[is_start] = 0.0
        idx = np.flatnonzero(~is_start)
        prev[idx] = dist_R[idx - 1]
        delta_R = dist_R - prev
        cum_R = dist_R
        risk_R = (direction * (dec - stop) / atr0)[sr]  # per-signal, same for all nodes
        rr_ref = cum_R / risk_R

        # outcomes: label each node over its signal's future window
        f_ti = []
        f_si = []
        tstate = []
        amb = []
        # group by signal for labeling
        us, inv = np.unique(sr, return_inverse=True)
        for ui, gidx in enumerate(us):
            gmask = inv == ui
            k = int(gidx)
            w = int(W[k])
            if w <= 0:
                f_ti.append(np.full(gmask.sum(), OUTCOME_WINDOW))
                f_si.append(OUTCOME_WINDOW)
                tstate.append(np.full(gmask.sum(), CENSOR_I))
                amb.append(np.zeros(gmask.sum(), dtype=int))
                continue
            Hp = bars["h"][start[k]:start[k] + w]
            Lp = bars["l"][start[k]:start[k] + w]
            tgt = target_price[gmask]
            lab = label_window(Hp, Lp, float(dec[k]), float(stop[k]), int(direction[k]), tgt)
            f_ti.append(np.asarray(lab["first_target_index"]))
            f_si.append(int(lab["first_stop_index"]))
            st = np.where(lab["target_first"], TARGET,
                          np.where(lab["stop_first"] | lab["ambiguous"], LOSS_I, CENSOR_I))
            tstate.append(st.astype(int))
            amb.append(lab["ambiguous"].astype(int))

        f_ti = np.concatenate(f_ti)
        f_si = np.array(f_si, dtype=int)
        tstate = np.concatenate(tstate)
        amb = np.concatenate(amb)

        gid = sr  # signal_gid == global index within this symbol build
        feats.append(pd.DataFrame(dict(
            signal_gid=gid, signal_id=sid[sr], symbol=symbol[sr],
            decision_time=dt[sr], trading_day=tday[sr], block=block[sr],
            direction=direction[sr], contact_type=ctype[sr],
            atr0=atr0[sr], decision_price=dec[sr], stop_price=stop[sr],
            contact_bar_index=j[sr], entry_bar=start[sr], W=W[sr],
            edge_index=edge_index.astype(int), target_price=target_price,
            delta_R=delta_R, cum_distance_R=cum_R, risk_R=risk_R, rr_ref=rr_ref,
        )))
        outs.append(pd.DataFrame(dict(
            signal_gid=gid, edge_index=edge_index.astype(int),
            target_state=tstate, first_target_index=f_ti,
            first_stop_index=np.repeat(f_si, lengths), was_ambiguous=amb,
        )))

    if not feats:
        return pd.DataFrame(), pd.DataFrame()
    features = pd.concat(feats, ignore_index=True)
    outcomes = pd.concat(outs, ignore_index=True)
    print(f"[NODE] sym={sym} signals={N} (valid={int(valid_sig.sum())}) "
          f"nodes={len(features)} nodes/sig={len(features)/max(N,1):.2f} "
          f"seconds={time.perf_counter()-t0:.1f}")
    return features, outcomes


# ---------------------------------------------------------------------------
# EV computation
# ---------------------------------------------------------------------------
def resolve_classes(pipe):
    classes = pipe.named_steps["clf"].classes_
    return {int(c): i for i, c in enumerate(classes)}


def predict_reordered(pipe, df, num, cat):
    proba = np.asarray(pipe.predict_proba(df[num + cat]), dtype=float)
    pos = resolve_classes(pipe)
    order = [pos[NEXT], pos[LOSS], pos[CENSOR]]
    return proba[:, order]


def compute_graph_ev(p_next, p_loss, p_censor, signal_gid, rr_ref):
    is_start, starts, lengths, _ = segment_meta(signal_gid)
    p_reach = segmented_cumprod(p_next, starts, lengths)
    p_next_safe = np.clip(p_next, 1e-12, 1.0)
    survival_before = p_reach / p_next_safe
    loss_mass_edge = survival_before * p_loss
    p_loss_before = segmented_cumsum(loss_mass_edge, starts, lengths)
    censor_mass_edge = survival_before * p_censor
    p_censor_before = segmented_cumsum(censor_mass_edge, starts, lengths)
    mass = p_reach + p_loss_before + p_censor_before
    assert np.allclose(mass, 1.0, atol=1e-6), "GRAPH_MASS_SUM_NOT_1"
    # P1-5: within a signal segment, p_reach is non-increasing across edge_index (later
    # nodes are reached at most as often). Vectorized real check, not a no-op.
    same_segment = signal_gid[1:] == signal_gid[:-1]
    assert np.all(np.diff(p_reach)[same_segment] <= 1e-9), "STOP_GRAPH_REACH_NONMONOTONIC"
    graph_ev = p_reach * rr_ref - p_loss_before
    return graph_ev, p_reach, p_loss_before, p_censor_before


# ---------------------------------------------------------------------------
# Selectors -> one target per signal
# ---------------------------------------------------------------------------
def select_targets(features: pd.DataFrame, graph_ev: np.ndarray, indep_ev: np.ndarray):
    """Return DataFrame indexed by signal_gid with chosen target per selector.

    IMPORTANT: this function receives ONLY features (no outcome columns). It returns
    the selected node row indices for T0/T1/T2.
    """
    assert "target_state" not in features.columns, "LEAKAGE_SELECTOR_READS_OUTCOME"
    assert "first_target_index" not in features.columns, "LEAKAGE_SELECTOR_READS_OUTCOME"
    sig = features["signal_gid"].to_numpy()
    is_start, starts, lengths, _ = segment_meta(sig)
    t0_idx = starts
    t1_idx = segmented_argmax(indep_ev, starts, lengths)
    t2_idx = segmented_argmax(graph_ev, starts, lengths)
    return dict(t0=t0_idx, t1=t1_idx, t2=t2_idx, starts=starts,
                lengths=lengths, signal_gid=sig)


# ---------------------------------------------------------------------------
# Vectorized reward (frozen execution contract)
# ---------------------------------------------------------------------------
def reward_for_signal(direction, entry_open, stop_px, target_px, Op, Hp, Lp):
    """Frozen execution contract (matches run_fixed_execution_baseline_v1.execute_path).

    Returns (realized_R, outcome_str, exit_bar_index, exit_px).
      exit_bar_index: index within the future path (0 == entry bar); -1 if no exit.
      exit_px:        realized fill price.

    P0-1: on the bar where target/stop is first hit, the fill uses that bar's OPEN, NOT its
    High/Low. A gap through the level is filled at the open (the executable price given the
    position opens at entry_open and the next bar gaps). Same-bar target&stop hit resolves
    STOP_FIRST (frozen default).
    """
    d = int(direction)
    W = len(Hp)
    if not (entry_open is not None and np.isfinite(entry_open)):
        return 0.0, "GAP_INVALID", -1, np.nan
    if not (d * (entry_open - stop_px) > 0):
        return 0.0, "ENTRY_BEYOND_STOP", -1, np.nan
    if not (d * (target_px - entry_open) > 0):
        return 0.0, "TARGET_PASSED_BEFORE_ENTRY", -1, np.nan
    if d > 0:
        sm = Lp <= stop_px
        tm = Hp >= target_px
    else:
        sm = Hp >= stop_px
        tm = Lp <= target_px
    si = int(np.argmax(sm)) if sm.any() else W
    ti = int(np.argmax(tm)) if tm.any() else W
    if si >= W and ti >= W:
        return 0.0, "CENSOR", -1, np.nan
    if si < W and ti < W and si == ti:
        # same-bar ambiguous -> STOP_FIRST (conservative, frozen default)
        k = si
        ex = min(stop_px, Op[k]) if d > 0 else max(stop_px, Op[k])
        risk_px = float(d * (entry_open - stop_px))
        return float(d * (ex - entry_open) / risk_px), "STOP", k, float(ex)
    if ti < si:
        k = ti
        ex = max(target_px, Op[k]) if d > 0 else min(target_px, Op[k])
        risk_px = float(d * (entry_open - stop_px))
        return float(d * (ex - entry_open) / risk_px), "TARGET", k, float(ex)
    else:
        k = si
        ex = min(stop_px, Op[k]) if d > 0 else max(stop_px, Op[k])
        risk_px = float(d * (entry_open - stop_px))
        return float(d * (ex - entry_open) / risk_px), "STOP", k, float(ex)


def compute_rewards(features: pd.DataFrame, sel: dict, bars_by_sym: dict):
    """Compute realized_R + outcome for T0/T1/T2, paired (same entry/stop/future).

    Returns ONE per-signal DataFrame (one row per signal) so reward / trading_day /
    decision_time columns are structurally guaranteed to share length = n_signals.
    """
    sig = features["signal_gid"].to_numpy()
    is_start, starts, lengths, _ = segment_meta(sig)
    sym = features["symbol"].to_numpy(object)
    direction = features["direction"].to_numpy(np.int64)
    stop = features["stop_price"].to_numpy(np.float64)
    entry_bar = features["entry_bar"].to_numpy(np.int64)
    target_price = features["target_price"].to_numpy(np.float64)
    W = features["W"].to_numpy(np.int64)
    tday = features["trading_day"].to_numpy()
    dtime = features["decision_time"].to_numpy()
    idx = {k: sel[k] for k in ("t0", "t1", "t2")}

    # Iterate SEGMENTS (aligned with select_targets, which returns one index per
    # segment). gids are globally unique & contiguous per signal, so segment == signal.
    rows = []
    for ui in range(len(starts)):
        k = int(starts[ui])            # first node row of this segment
        gid = int(sig[k])
        b = bars_by_sym[sym[k]]
        eb = int(entry_bar[k])
        row = dict(signal_gid=gid, symbol=sym[k], direction=int(direction[k]),
                   entry_bar=eb, W=int(W[k]), stop_price=float(stop[k]),
                   trading_day=tday[k], decision_time=dtime[k])
        if eb >= b["n"] or eb < 0 or bool(b["disc"][eb]):
            for s in ("t0", "t1", "t2"):
                row[f"target_price_{s}"] = float(target_price[idx[s][ui]])
                row[f"reward_{s}"] = 0.0
                row[f"outcome_{s}"] = "GAP_INVALID"
            rows.append(row)
            continue
        w = int(W[k])
        Hp = b["h"][eb:eb + w] if w > 0 else np.array([])
        Lp = b["l"][eb:eb + w] if w > 0 else np.array([])
        Op = b["o"][eb:eb + w] if w > 0 else np.array([])
        entry_open = float(b["o"][eb])
        row["entry_open"] = entry_open
        for s in ("t0", "t1", "t2"):
            tp = float(target_price[idx[s][ui]])
            r, o, _k, _ex = reward_for_signal(int(direction[k]), entry_open,
                                             float(stop[k]), tp, Op, Hp, Lp)
            row[f"target_price_{s}"] = tp
            row[f"reward_{s}"] = r
            row[f"outcome_{s}"] = o
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Sequential (one position per symbol) state machine — only allowed python loop
# ---------------------------------------------------------------------------
def sequential_curve(features: pd.DataFrame, sel: dict, bars_by_sym: dict, which="t2"):
    # P0-4: sequential WF1 curve MUST run on TB2-only test signals.
    assert (features["block"] == "TB2").all(), "STOP_SEQUENTIAL_NON_TB2_INPUT"
    sig = features["signal_gid"].to_numpy()
    is_start, starts, lengths, _ = segment_meta(sig)
    usig, inv = np.unique(sig, return_inverse=True)
    sym = features["symbol"].to_numpy(object)
    direction = features["direction"].to_numpy(np.int64)
    stop = features["stop_price"].to_numpy(np.float64)
    entry_bar = features["entry_bar"].to_numpy(np.int64)
    target_price = features["target_price"].to_numpy(np.float64)
    W = features["W"].to_numpy(np.int64)
    tday = features["trading_day"].to_numpy(object)
    idx = sel[which]

    # sort signals chronologically by entry_bar ascending per symbol
    rows = []
    open_until = {}
    meta = {}
    for ui in range(len(usig)):
        k = int(starts[ui])  # row index of this signal's first node
        b = bars_by_sym[sym[k]]
        eb = int(entry_bar[k])
        entry_time = pd.Timestamp(b["t"][eb]) if 0 <= eb < b["n"] else pd.NaT
        meta[ui] = (eb, entry_time)
    order = sorted(range(len(usig)), key=lambda ui: (meta[ui][1] if meta[ui][1] is not None else pd.NaT))
    for ui in order:
        k = int(starts[ui])
        b = bars_by_sym[sym[k]]
        eb, entry_time = meta[ui]
        if eb >= b["n"] or eb < 0 or bool(b["disc"][eb]):
            rows.append(dict(signal_gid=k, executed=False, skip_reason="GAP_INVALID",
                             realized_R=0.0, outcome="GAP_INVALID",
                             entry_time=entry_time, symbol=sym[k], direction=int(direction[k]),
                             trading_day=tday[k]))
            continue
        # selected node row index for this signal is idx[ui]
        node_abs = idx[ui]
        tp = float(target_price[node_abs])
        entry_open = float(b["o"][eb])
        if not (int(direction[k]) * (entry_open - stop[k]) > 0):
            rows.append(dict(signal_gid=k, executed=False, skip_reason="ENTRY_BEYOND_STOP",
                             realized_R=0.0, outcome="ENTRY_BEYOND_STOP",
                             entry_time=entry_time, symbol=sym[k], direction=int(direction[k]),
                             trading_day=tday[k]))
            continue
        if not (int(direction[k]) * (tp - entry_open) > 0):
            rows.append(dict(signal_gid=k, executed=False, skip_reason="TARGET_PASSED_BEFORE_ENTRY",
                             realized_R=0.0, outcome="TARGET_PASSED_BEFORE_ENTRY",
                             entry_time=entry_time, symbol=sym[k], direction=int(direction[k]),
                             trading_day=tday[k]))
            continue
        if sym[k] in open_until and entry_time <= open_until[sym[k]]:
            rows.append(dict(signal_gid=k, executed=False, skip_reason="BLOCKED_BY_OPEN_POSITION",
                             realized_R=0.0, outcome="BLOCKED_BY_OPEN_POSITION",
                             entry_time=entry_time, symbol=sym[k], direction=int(direction[k]),
                             trading_day=tday[k]))
            continue
        w = int(W[k])
        Hp = b["h"][eb:eb + w] if w > 0 else np.array([])
        Lp = b["l"][eb:eb + w] if w > 0 else np.array([])
        Op = b["o"][eb:eb + w] if w > 0 else np.array([])
        # P0-3: SAME execution kernel as paired; exit_bar_index drives exit_time (no 2nd argmax).
        r, o, exit_idx, _ex = reward_for_signal(int(direction[k]), entry_open,
                                               float(stop[k]), tp, Op, Hp, Lp)
        if o == "CENSOR":
            last = min(eb + w - 1, b["n"] - 1) if w > 0 else eb
            exit_time = pd.Timestamp(b["t"][last]) if 0 <= last < b["n"] else entry_time
        else:
            ei = int(eb + exit_idx) if (exit_idx >= 0 and 0 <= eb + exit_idx < b["n"]) else eb
            exit_time = pd.Timestamp(b["t"][ei])
        open_until[sym[k]] = exit_time
        rows.append(dict(signal_gid=k, executed=True, skip_reason="", realized_R=float(r),
                         outcome=o, entry_time=entry_time, exit_time=exit_time,
                         symbol=sym[k], direction=int(direction[k]), trading_day=tday[k]))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(rr, decision_day=None, label="signals"):
    """label: 'signals' for paired counterfactual (no one-position filter),
    'trades' for sequential (one position per symbol). P1-3."""
    rr = np.asarray(rr, dtype=float)
    rr = rr[np.isfinite(rr)]
    n = len(rr)
    if n == 0:
        return dict(n=0)
    wins = rr[rr > 0]
    losses = rr[rr < 0]
    mw = float(wins.mean()) if len(wins) else np.nan
    ml = float(losses.mean()) if len(losses) else np.nan
    sw = float(wins.sum()) if len(wins) else 0.0
    sl = float(abs(losses.sum())) if len(losses) else 0.0
    payoff = (mw / abs(ml)) if (len(wins) and len(losses) and ml != 0) else np.nan
    pf = (sw / sl) if sl > 0 else np.nan
    ev = float(rr.mean())
    total_R = float(rr.sum())
    # P1-1: primary drawdown in R units (cumulative sum of per-signal R).
    cum_R = np.r_[0.0, np.cumsum(rr)]
    peak_R = np.maximum.accumulate(cum_R)
    dd_R = cum_R - peak_R
    max_DD_R = float(-dd_R.min())
    # P1-2: 0.5%-risk normalized account drawdown — SEPARATE metric, NOT a real margin
    # backtest. Kept only for descriptive comparison.
    equity = np.cumprod(1.0 + 0.005 * rr)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    max_DD_pct_0p5risk = float(dd.min() * 100)  # NOT_REAL_MARGIN_BACKTEST
    m = dict(n=int(n), win_rate=float((rr > 0).mean()),
             avg_win_R=mw, avg_loss_R=ml, payoff_ratio=payoff,
             profit_factor=pf, EV_per_signal=ev, total_R=total_R,
             max_DD_R=max_DD_R,
             max_DD_pct_0p5risk=max_DD_pct_0p5risk,  # NOT_REAL_MARGIN_BACKTEST
             totalR_over_maxDD=(total_R / max_DD_R if max_DD_R > 0 else np.nan))
    if decision_day is not None:
        days = pd.to_datetime(np.asarray(decision_day)).values
        if len(days):
            t0 = pd.Timestamp(days.min()); t1 = pd.Timestamp(days.max())
            span = max((t1 - t0).days, 1)
            # P1-3: paired counterfactual has no one-position execution filter -> "signals";
            # sequential (one position per symbol) -> "trades".
            m[f"{label}_per_100_days"] = round(n / span * 100, 3)
            m["R_per_100_days"] = round(total_R / span * 100, 3)
            months = pd.to_datetime(days).values.astype("datetime64[M]")
            by_m = {}
            for mo, v in zip(months, rr):
                by_m.setdefault(mo, []).append(v)
            month_R = {mo: float(np.sum(v)) for mo, v in by_m.items()}
            vals = list(month_R.values())
            m["positive_month_ratio"] = float(np.mean([1 if v > 0 else 0 for v in vals])) if vals else np.nan
            m["best_month_R"] = float(max(vals)) if vals else np.nan
            m["worst_month_R"] = float(min(vals)) if vals else np.nan
    return m


def rolling_metrics(rr, window):
    rr = np.asarray(rr, dtype=float)
    out = []
    for i in range(len(rr)):
        s = rr[max(0, i - window + 1):i + 1]
        if len(s) >= 1:
            out.append(float(np.mean(s[np.isfinite(s)])))
        else:
            out.append(np.nan)
    return np.array(out)


# ---------------------------------------------------------------------------
# Bootstrap (paired by trading day)
# ---------------------------------------------------------------------------
def paired_bootstrap_by_day(diff, decision_day, n=500):
    day = pd.to_datetime(np.asarray(decision_day)).values
    udays, inv = np.unique(day, return_inverse=True)
    D = len(udays)
    day_sum = np.array([diff[inv == d].sum() for d in range(D)])
    day_n = np.array([int((inv == d).sum()) for d in range(D)])
    rng = np.random.default_rng(0)
    num = np.empty(n)
    den = np.empty(n)
    for b in range(n):
        idx = rng.integers(0, D, size=D)
        num[b] = day_sum[idx].sum()
        den[b] = day_n[idx].sum()
    ratios = num / np.where(den == 0, np.nan, den)
    ratios = ratios[np.isfinite(ratios)]
    lo = float(np.percentile(ratios, 2.5))
    hi = float(np.percentile(ratios, 97.5))
    return dict(mean=float(np.mean(ratios)), ci_lo=lo, ci_hi=hi, p=float((ratios > 0).mean()))


# ---------------------------------------------------------------------------
# Caching (per symbol, stage-separated)
# ---------------------------------------------------------------------------
def _hash_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def build_or_load_nodes(sym, contacts_sym, master_sym, bars, force=False, cache=True):
    # Cache key MUST include the exact signal set. Otherwise a stale subset cache
    # (e.g. smoke's 100 signals) would be silently reused by a fuller run (pilot 500
    # or WF1 all-signals), under-sampling the universe without any error.
    sids = (contacts_sym["liquidity_id"].astype(str) + "|"
            + contacts_sym["contact_number"].astype(str))
    sig_hash = hashlib.sha256(
        str(tuple(sorted(sids.tolist()))).encode()).hexdigest()[:12]
    fpath = CACHE / f"g2_full_nodes_features_{sym}_TB12_{sig_hash}.parquet"
    opath = CACHE / f"g2_full_nodes_outcomes_{sym}_TB12_{sig_hash}.parquet"
    meta = CACHE / f"g2_nodes_meta_{sym}_TB12_{sig_hash}.json"
    if cache and fpath.exists() and opath.exists() and not force:
        try:
            m = json.loads(meta.read_text())
            if (m.get("builder_version") == BUILDER_VERSION and
                    m.get("master_hash") == _hash_file(MASTER_PATH) and
                    m.get("contacts_hash") == _hash_file(CONTACTS_PATH) and
                    m.get("sig_hash") == sig_hash):
                return pd.read_parquet(fpath), pd.read_parquet(opath)
        except Exception:
            pass
    feats, outs = build_full_nodes_symbol(sym, contacts_sym, master_sym, bars, BUILDER_VERSION)
    if cache:
        feats.to_parquet(fpath, index=False)
        outs.to_parquet(opath, index=False)
        meta.write_text(json.dumps(dict(builder_version=BUILDER_VERSION,
                                        master_hash=_hash_file(MASTER_PATH),
                                        contacts_hash=_hash_file(CONTACTS_PATH),
                                        sig_hash=sig_hash,
                                        n_signals=int(len(contacts_sym)),
                                        source_commit="0a048fd", G0_spec_hash="frozen")))
    return feats, outs


# ---------------------------------------------------------------------------
# G0 / I0 training + scoring
# ---------------------------------------------------------------------------
def train_g0(scope_tag="TB12_HARDENED"):
    tr = load_transitions_g0(scope_tag)
    from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import load_transitions
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag=scope_tag)
    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_purged, purg = purge_train(tr, test_start)
    print("  [PURGE] " + ", ".join(f"{k}={v}" for k, v in purg.items()))
    assert purg["n_train_signals_before_purge"] == G0_HARDENED_BASELINE["before"]
    assert purg["n_train_signals_after_purge"] == G0_HARDENED_BASELINE["after"]
    assert purg["n_purged_signals"] == G0_HARDENED_BASELINE["purged"]
    pipe = fit_multinomial(tr_purged, G0_NUM, G0_CAT)
    return pipe, purg


def load_transitions_g0(scope_tag="TB12_HARDENED"):
    from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import load_transitions
    return load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag=scope_tag)


def score_g0_nodes(pipe, features: pd.DataFrame):
    proba = predict_reordered(pipe, features, G0_NUM, G0_CAT)
    p_next, p_loss, p_censor = proba[:, 0], proba[:, 1], proba[:, 2]
    ev, p_reach, p_loss_b, p_censor_b = compute_graph_ev(
        p_next, p_loss, p_censor, features["signal_gid"].to_numpy(),
        features["rr_ref"].to_numpy())
    return ev, p_reach, p_loss_b, p_censor_b


def train_i0(features: pd.DataFrame, outcomes: pd.DataFrame):
    """Train I0 on TB1 full nodes (purged), return pipe."""
    df = features.merge(outcomes, on=["signal_gid", "edge_index"], how="inner")
    df = df[df["block"] == "TB1"]
    df = df.copy()
    df["state_code"] = df["target_state"].astype(int)
    # purge: use resolution_end_time from G0 transition cache per signal
    tr = load_transitions_g0()
    res_end = (tr.groupby("signal_id")["signal_resolution_end_time"].max()
               .rename("res_end").reset_index())
    df = df.merge(res_end, left_on="signal_id", right_on="signal_id", how="left")
    test_start = pd.Timestamp("2025-06-11 00:00:00")
    df = df[df["res_end"] < test_start]
    pipe = fit_multinomial(df, I0_NUM, I0_CAT)
    return pipe


def score_i0_nodes(pipe, features: pd.DataFrame):
    proba = predict_reordered(pipe, features, I0_NUM, I0_CAT)
    p_target, p_loss, p_censor = proba[:, 0], proba[:, 1], proba[:, 2]
    indep_ev = p_target * features["rr_ref"].to_numpy() - p_loss
    return indep_ev, p_target, p_loss, p_censor


# ---------------------------------------------------------------------------
# Hardened whole-signal purge — SAME canonical training sample as hardened G0
# ---------------------------------------------------------------------------
def hardened_purged_train_signals(scope_tag="TB12_HARDENED"):
    """Canonical G0 training sample: TB1 transition risk-set, purged so that no train
    signal resolves at/after the TB2 test start. F0/F1/F2/G0 attribution MUST train on
    exactly this set to stay apples-to-apples with hardened G0.

    Contract (must reproduce): before=21165, after=21135, purged=30.
    Otherwise STOP_G2_ATTRIBUTION_SAMPLE_DRIFT.
    """
    from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import load_transitions
    tr = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag=scope_tag)
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag=scope_tag)
    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_purged, purg = purge_train(tr, test_start)
    assert purg["n_train_signals_before_purge"] == G0_HARDENED_BASELINE["before"], \
        "STOP_G2_ATTRIBUTION_SAMPLE_DRIFT"
    assert purg["n_train_signals_after_purge"] == G0_HARDENED_BASELINE["after"], \
        "STOP_G2_ATTRIBUTION_SAMPLE_DRIFT"
    assert purg["n_purged_signals"] == G0_HARDENED_BASELINE["purged"], \
        "STOP_G2_ATTRIBUTION_SAMPLE_DRIFT"
    return set(tr_purged["signal_id"].unique()), purg


# ---------------------------------------------------------------------------
# G0 Feature Attribution (fixed nested models; NLL/Brier only; NOT in PASS/FAIL)
# ---------------------------------------------------------------------------
def g0_feature_attribution():
    """FIXED feature attribution (Section A): F0/F1/F2/G0 nested multinomials evaluated
    on the FROZEN G0 probability task — the hardened competing-risk transition risk-set
    (transitions_*_TB12_HARDENED.parquet) — NOT the G2 full-node table.

    Why this matters: the G0 probability task is a per-signal ordered transition chain
    (path stops at first LOSS/CENSOR). Evaluating on the full-node table (all active
    candidate nodes, ~20x edges/signal) inflates joint NLL and is a DIFFERENT task. Here
    train = TB1 (hardened-purged, identical canonical sample as hardened G0); test = TB2
    transition risk-set. G0 must therefore reproduce the committed baseline
    (joint NLL/signal ~1.6809, equal-signal Brier ~0.22866).

    The full-node table is reserved for T0/T1/T2 target selection only.
    delta convention: delta = baseline_metric - augmented_metric (positive => augmented better).
    NOT part of the G2 economic PASS/FAIL gate.
    """
    from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (
        load_transitions, fit_multinomial, signal_metrics, purge_train,
        G0_HARDENED_BASELINE,
    )
    tr = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag="TB12_HARDENED")
    te = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag="TB12_HARDENED")
    test_start = pd.Timestamp(te["signal_trading_day"].min())
    tr_purged, purg = purge_train(tr, test_start)
    # Guard 3: exact hardened whole-signal purge contract
    assert purg["n_train_signals_before_purge"] == G0_HARDENED_BASELINE["before"], \
        "STOP_G2_ATTRIBUTION_SAMPLE_DRIFT"
    assert purg["n_train_signals_after_purge"] == G0_HARDENED_BASELINE["after"], \
        "STOP_G2_ATTRIBUTION_SAMPLE_DRIFT"
    assert purg["n_purged_signals"] == G0_HARDENED_BASELINE["purged"], \
        "STOP_G2_ATTRIBUTION_SAMPLE_DRIFT"
    tr_purged = tr_purged.copy()
    tr_purged["state_code"] = tr_purged["state_code"].astype(int)
    te = te.copy()
    te["state_code"] = te["state_code"].astype(int)
    print("  [ATTR] hardened transition purge: " + ", ".join(f"{k}={v}" for k, v in purg.items()))
    print(f"  [ATTR] test signals={te['signal_id'].nunique()} edges={len(te)}")

    rows = []
    metrics = {}
    for name, num, cat in ATTR_SPECS:
        pipe = fit_multinomial(tr_purged, num, cat)
        m = signal_metrics(pipe, te)
        metrics[name] = m
        rows.append(dict(
            model=name, n_train_edges=int(len(tr_purged)), n_test_edges=int(len(te)),
            joint_nll_per_signal=float(m["joint_nll_per_signal"]),
            joint_nll_per_edge=float(m["joint_nll_per_edge"]),
            brier_signal_equal=float(m["brier_signal_equal"]),
            brier_path_sum=float(m["brier_path_sum"]),
        ))
        print(f"  [ATTR] {name}: jNLL/sig={m['joint_nll_per_signal']:.4f} "
              f"brier_eq={m['brier_signal_equal']:.5f} "
              f"(+delta_R={'delta_R' in num}) (+edge_index={'edge_index' in cat})")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "g2_feature_attribution.csv", index=False)

    def _d(baseline, aug, key):
        return metrics[baseline][key] - metrics[aug][key]

    deltas = pd.DataFrame([
        dict(contrast="F1-F0 (local spacing: +delta_R)",
             d_joint_nll=_d("F0", "F1", "joint_nll_per_signal"),
             d_brier=_d("F0", "F1", "brier_signal_equal")),
        dict(contrast="F2-F0 (chain position: +edge_index)",
             d_joint_nll=_d("F0", "F2", "joint_nll_per_signal"),
             d_brier=_d("F0", "F2", "brier_signal_equal")),
        dict(contrast="G0-F1 (edge_index | spacing)",
             d_joint_nll=_d("F1", "G0", "joint_nll_per_signal"),
             d_brier=_d("F1", "G0", "brier_signal_equal")),
        dict(contrast="G0-F2 (delta_R | chain position)",
             d_joint_nll=_d("F2", "G0", "joint_nll_per_signal"),
             d_brier=_d("F2", "G0", "brier_signal_equal")),
    ])
    deltas.to_csv(OUT / "g2_feature_attribution_deltas.csv", index=False)
    print("[ATTR] deltas (positive => augmented model better; lower NLL/Brier is better):")
    for _, r in deltas.iterrows():
        print(f"    {r['contrast']:34s} dNLL={r['d_joint_nll']:+.4f} "
              f"dBrier={r['d_brier']:+.5f}")
    return df, deltas, metrics


# ---------------------------------------------------------------------------
# T2 Node EV decomposition (reward_term / loss_term / graph_ev)
# ---------------------------------------------------------------------------
def t2_node_decomposition(test_feats: pd.DataFrame, g0_pipe, rewards_test: pd.DataFrame,
                         sel_test: dict):
    """For each TB2 signal's T2-selected node, save the EV decomposition:
        reward_term  = p_reach * rr_ref
        loss_term    = -p_loss_before
        graph_ev     = reward_term + loss_term
    plus p_censor_before. Attaches realized R / outcome of the chosen node.
    """
    ev, p_reach, p_loss_b, p_censor_b = score_g0_nodes(g0_pipe, test_feats)
    rr = test_feats["rr_ref"].to_numpy()
    t2_idx = sel_test["t2"]  # row indices into test_feats (one per signal)
    dec = test_feats.iloc[np.asarray(t2_idx)].reset_index(drop=True)
    rows = pd.DataFrame({
        "signal_gid": dec["signal_gid"].to_numpy(),
        "signal_id": dec["signal_id"].to_numpy(),
        "symbol": dec["symbol"].to_numpy(),
        "direction": dec["direction"].to_numpy(),
        "edge_index": dec["edge_index"].to_numpy(),
        "target_price": dec["target_price"].to_numpy(),
        "cum_distance_R": dec["cum_distance_R"].to_numpy(),
        "delta_R": dec["delta_R"].to_numpy(),
        "rr_ref": rr[t2_idx],
        "p_reach": p_reach[t2_idx],
        "p_loss_before": p_loss_b[t2_idx],
        "p_censor_before": p_censor_b[t2_idx],
    })
    rows["reward_term"] = rows["p_reach"] * rows["rr_ref"]
    rows["loss_term"] = -rows["p_loss_before"]
    rows["graph_ev"] = rows["reward_term"] + rows["loss_term"]
    rows["realized_R"] = rewards_test["reward_t2"].to_numpy()
    rows["outcome"] = rewards_test["outcome_t2"].to_numpy()
    rows.to_csv(OUT / "g2_t2_node_decomposition.csv", index=False)
    print(f"[DECOMP] T2 node decomposition rows={len(rows)} "
          f"mean reward_term={rows['reward_term'].mean():.4f} "
          f"mean loss_term={rows['loss_term'].mean():.4f} "
          f"mean graph_ev={rows['graph_ev'].mean():.4f}")
    return rows


# ---------------------------------------------------------------------------
# Same-Bar Multi-Node Diagnostic (Section D)
# ---------------------------------------------------------------------------
def samebar_multinode_diagnostic(test_feats: pd.DataFrame, outcomes: pd.DataFrame):
    """POST-SELECTION diagnostic. Reads ONLY the outcome table (first_target_index /
    first_stop_index); NEVER passed to any selector.

    Meaning (correct interpretation): in the FUTURE realized 5m path, how many adjacent
    liquidity nodes have their FIRST target hit on the SAME 5m bar — i.e. one bar crosses
    L1..Lk at once. Strict pre-stop reach: first_target_index < first_stop_index. CENSOR
    (both == W) is NOT a crossing. ambiguous_with_stop = target & stop hit same bar (< W).
    """
    oc = outcomes.merge(test_feats[["signal_gid", "edge_index", "W"]],
                        on=["signal_gid", "edge_index"], how="inner")
    sig = oc["signal_gid"].to_numpy()
    fti = oc["first_target_index"].to_numpy().astype(np.int64)
    fsi = oc["first_stop_index"].to_numpy().astype(np.int64)
    W = oc["W"].to_numpy().astype(np.int64)
    reached = fti < fsi                                   # strict pre-stop reach
    amb_stop = (fti == fsi) & (fti < W)                   # same-bar target&stop (both < W)

    n_nodes = len(oc)
    n_reached = int(reached.sum())
    amb_stop_rate = float(amb_stop.mean()) if n_nodes else np.nan

    # same-bar crossing batches: key = (signal_gid, first_target_index) among reached.
    # fti < 1000 (within-window index), so the integer composite is collision-free.
    key = sig[reached].astype(np.int64) * 1000 + fti[reached].astype(np.int64)
    codes, _ = pd.factorize(key)
    batch_sizes = np.bincount(codes)

    # per-signal max batch size (fully vectorized, no groupby.apply)
    uniq_sig = np.unique(sig)
    sig_pos = np.searchsorted(uniq_sig, sig[reached])
    max_batch = np.zeros(len(uniq_sig))
    np.maximum.at(max_batch, sig_pos, batch_sizes[codes])

    n_test_signals = len(uniq_sig)
    rate_ge2 = float((max_batch >= 2).mean())
    rate_ge3 = float((max_batch >= 3).mean())
    rate_ge4 = float((max_batch >= 4).mean())

    n_batches = len(batch_sizes)
    cnt_eq1 = int((batch_sizes == 1).sum())
    cnt_eq2 = int((batch_sizes == 2).sum())
    cnt_eq3 = int((batch_sizes == 3).sum())
    cnt_ge4 = int((batch_sizes >= 4).sum())
    nodes_in_multi = int(batch_sizes[batch_sizes >= 2].sum())
    frac_multi = nodes_in_multi / n_reached if n_reached else np.nan

    summ = dict(
        n_test_signals=n_test_signals,
        n_test_nodes=n_nodes,
        n_reached_nodes=n_reached,
        rate_max_batch_ge_2=rate_ge2,
        rate_max_batch_ge_3=rate_ge3,
        rate_max_batch_ge_4=rate_ge4,
        batch_size_1=cnt_eq1,
        batch_size_2=cnt_eq2,
        batch_size_3=cnt_eq3,
        batch_size_ge4=cnt_ge4,
        fraction_reached_nodes_in_multinode_batch=frac_multi,
        ambiguous_with_stop_node_rate=amb_stop_rate,
    )
    pd.DataFrame([summ]).to_csv(OUT / "g2_samebar_multinode_diagnostic.csv", index=False)
    print("[SAME-BAR] future-path multi-node crossing (post-selection, outcome-only):")
    print(f"    n_test_signals={n_test_signals} n_reached_nodes={n_reached}")
    print(f"    rate max_batch>=2={rate_ge2:.3f} >=3={rate_ge3:.3f} >=4={rate_ge4:.3f}")
    print(f"    batch sizes 1/2/3/>=4 = {cnt_eq1}/{cnt_eq2}/{cnt_eq3}/{cnt_ge4}")
    print(f"    fraction reached nodes in multi-node batch={frac_multi:.3f}")
    print(f"    ambiguous_with_stop_node_rate(fti==fsi<W)={amb_stop_rate:.3f}")
    return summ


# ---------------------------------------------------------------------------
# Candidate density diagnostic (renamed from old same-bar; per-signal node count)
# ---------------------------------------------------------------------------
def candidate_density_diagnostic(test_feats: pd.DataFrame, rewards_test: pd.DataFrame,
                                 sel_test: dict):
    """Secondary descriptive view: per-signal active candidate-node count (all active
    nodes of a signal share the same decision/entry bar). NOT the same-bar crossing
    diagnostic. Shows how T0/T1/T2 diverge as the candidate set grows."""
    sig = test_feats["signal_gid"].to_numpy()
    _, starts, lengths, _ = segment_meta(sig)
    n_nodes = lengths
    ei = test_feats["edge_index"].to_numpy()
    e_t1 = ei[sel_test["t1"]]
    e_t2 = ei[sel_test["t2"]]
    r_t0 = rewards_test["reward_t0"].to_numpy()
    r_t1 = rewards_test["reward_t1"].to_numpy()
    r_t2 = rewards_test["reward_t2"].to_numpy()

    frac_t1_nonnearest = float((e_t1 > 0).mean())
    frac_t2_nonnearest = float((e_t2 > 0).mean())

    q = np.quantile(n_nodes, [0.25, 0.5, 0.75])
    bucket = np.digitize(n_nodes, q)
    recs = []
    for b in range(4):
        m = bucket == b
        if m.sum() == 0:
            continue
        recs.append(dict(
            bucket=f"Q{b+1}", n_signals=int(m.sum()),
            n_nodes_lo=int(n_nodes[m].min()), n_nodes_hi=int(n_nodes[m].max()),
            frac_t1_nonnearest=float((e_t1[m] > 0).mean()),
            frac_t2_nonnearest=float((e_t2[m] > 0).mean()),
            mean_R_t0=float(np.nanmean(r_t0[m])),
            mean_R_t1=float(np.nanmean(r_t1[m])),
            mean_R_t2=float(np.nanmean(r_t2[m])),
            mean_T2_minus_T0=float(np.nanmean((r_t2 - r_t0)[m])),
        ))
    df = pd.DataFrame(recs)
    df.to_csv(OUT / "g2_candidate_density_diagnostic.csv", index=False)
    corr = float(np.corrcoef(n_nodes, r_t2 - r_t0)[0, 1]) if len(n_nodes) > 1 else np.nan
    print(f"[CANDIDATE-DENSITY] n_nodes median={int(np.median(n_nodes))} "
          f"p25/p75={int(np.quantile(n_nodes,0.25))}/{int(np.quantile(n_nodes,0.75))}")
    print(f"    frac T1(non-nearest)={frac_t1_nonnearest:.3f} "
          f"frac T2(non-nearest)={frac_t2_nonnearest:.3f} corr(n_nodes,T2-T0)={corr:+.3f}")
    return df


# ---------------------------------------------------------------------------
# Component test harness (test mode) — no economics
# ---------------------------------------------------------------------------
def _segmented_cumsum_scalar(x, starts, lengths):
    out = np.empty_like(x)
    for i in range(len(starts)):
        s, l = int(starts[i]), int(lengths[i])
        out[s:s + l] = np.cumsum(x[s:s + l])
    return out


def _segmented_argmax_scalar(value, starts, lengths):
    out = np.empty(len(starts), dtype=np.int64)
    for i in range(len(starts)):
        s, l = int(starts[i]), int(lengths[i])
        out[i] = s + int(np.argmax(value[s:s + l]))
    return out


# ---------------------------------------------------------------------------
# Closure audit — reproducibility fix + deterministic economic replay.
# Reuses TB12_HARDENED transition cache + G2 full-node feature cache.
# Does NOT rebuild master field / full nodes / scan liquidity / future paths.
# ---------------------------------------------------------------------------
def _sig_hash_for_symbol(contacts_sym):
    sids = (contacts_sym["liquidity_id"].astype(str) + "|"
            + contacts_sym["contact_number"].astype(str))
    return hashlib.sha256(str(tuple(sorted(sids.tolist()))).encode()).hexdigest()[:12]


def load_full_node_tb2(force_rebuild=False):
    """P0-8: load the EXACT WF1 full-universe full-node caches by signal-set hash.
    NO 'pick largest cache' heuristic, NO rebuild. If the exact cache for any symbol is
    missing, or its builder_version/master_hash/contacts_hash/sig_hash do not match the
    current data, STOP_CLOSURE_CACHE_MISSING (reviewer must rebuild explicitly).
    Returns (features, outcomes, provenance). signal_gid is globalized via the SAME
    helper run_g2 uses (P0-6)."""
    master = load_master()
    contacts = load_contacts()
    contacts["block"] = assign_blocks(contacts)
    feats_all, outs_all = [], []
    offset = 0
    provenance = {}
    for sym in FULL_UNIV:
        cs = contacts[contacts["symbol"] == sym]
        expected_hash = _sig_hash_for_symbol(cs)
        fpath = CACHE / f"g2_full_nodes_features_{sym}_TB12_{expected_hash}.parquet"
        opath = CACHE / f"g2_full_nodes_outcomes_{sym}_TB12_{expected_hash}.parquet"
        meta = CACHE / f"g2_nodes_meta_{sym}_TB12_{expected_hash}.json"
        if not (fpath.exists() and opath.exists() and meta.exists()):
            print("STOP_CLOSURE_CACHE_MISSING", sym, expected_hash)
            return None, None, None
        m = json.loads(meta.read_text())
        if not (m.get("builder_version") == BUILDER_VERSION
                and m.get("master_hash") == _hash_file(MASTER_PATH)
                and m.get("contacts_hash") == _hash_file(CONTACTS_PATH)
                and m.get("sig_hash") == expected_hash):
            print("STOP_CLOSURE_CACHE_HASH_MISMATCH", sym, expected_hash, m)
            return None, None, None
        feats = pd.read_parquet(fpath)
        outs = pd.read_parquet(opath)
        # P0-6: same globalize helper as run_g2.
        feats, outs, offset = globalize_node_ids(feats, outs, offset)
        feats_all.append(feats)
        outs_all.append(outs)
        provenance[sym] = dict(expected_hash=expected_hash,
                               builder_version=m.get("builder_version"),
                               master_hash=m.get("master_hash"),
                               contacts_hash=m.get("contacts_hash"),
                               sig_hash=m.get("sig_hash"),
                               n_signals=int(m.get("n_signals", -1)))
    features = pd.concat(feats_all, ignore_index=True)
    outcomes = pd.concat(outs_all, ignore_index=True)
    # P0-7: node-key uniqueness guards (one-to-one, no cross-symbol collision).
    feat_key = ["signal_gid", "edge_index"]
    assert not features.duplicated(feat_key).any(), "STOP_G2_NODE_KEY_COLLISION"
    assert not outcomes.duplicated(feat_key).any(), "STOP_G2_NODE_KEY_COLLISION"
    assert len(features) == len(outcomes), "STOP_G2_NODE_KEY_COLLISION"
    features.merge(outcomes, on=feat_key, validate="one_to_one")
    return features, outcomes, provenance


def run_g2_closure_audit():
    """G2 closure audit (per user spec sections 1-14).

    Guards:
      A purge baseline (21165->21135, purge 30) on TB1 transitions
      B test universe (n_test_signals=24954, n_test_edges=55945) on TB2 transitions
      C G0 reproduction: T2 scoring model == hardened G0 (jNLL~1.6809, brier~0.22866)
      D model contract: T2 scoring == attribution G0 (features/classes/coef/train hash)
      E F0/F1/F2/G0 attribution on transition risk-set (G0 entry reproduces baseline)
      F zero-node universe: 24954 - 24700 == n_zero_node_signals (per symbol)
      G economic replay: T0/T1/T2 EV + T2-T0 bootstrap reproduce g2_summary/g2_bootstrap
      H same-bar diagnostic deterministically reconfirmed
    """
    from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (
        load_transitions, fit_multinomial, signal_metrics, purge_train,
        G0_HARDENED_BASELINE,
    )
    HARD_PATH = REPO / "research/analysis_results/5m_graph_probability_v1/graph_necessity_HARDENED_PASS.csv"
    HARD = pd.read_csv(HARD_PATH).iloc[0]
    HARDENED_G0_NLL = float(HARD["g0_joint_nll"])
    HARDENED_G0_BRIER = float(HARD["g0_brier_eqsig"])
    HARDENED_N_TEST_EDGES = int(HARD["n_test_edges"])
    HARDENED_N_TEST_SIGNALS = int(HARD["n_test_signals"])

    guards = {}
    t0 = time.perf_counter()

    # ---- transition risk-set (TB12_HARDENED) ----
    tr_all = load_transitions(FULL_UNIV, blocks=["TB1"], scope_tag="TB12_HARDENED")
    te_all = load_transitions(FULL_UNIV, blocks=["TB2"], scope_tag="TB12_HARDENED")
    test_start = pd.Timestamp(te_all["signal_trading_day"].min())
    tr_purged, purg = purge_train(tr_all, test_start)

    # Guard A: hardened purge baseline
    gA = (purg["n_train_signals_before_purge"] == G0_HARDENED_BASELINE["before"]
          and purg["n_train_signals_after_purge"] == G0_HARDENED_BASELINE["after"]
          and purg["n_purged_signals"] == G0_HARDENED_BASELINE["purged"])
    guards["A_purge_baseline"] = dict(pass_=bool(gA), **purg,
                                      baseline=dict(G0_HARDENED_BASELINE))

    # Guard B: test universe size (transition TB2)
    n_test_sig = int(te_all["signal_id"].nunique())
    n_test_edge = int(len(te_all))
    gB = (n_test_sig == HARDENED_N_TEST_SIGNALS and n_test_edge == HARDENED_N_TEST_EDGES)
    guards["B_test_universe"] = dict(pass_=bool(gB), n_test_signals=n_test_sig,
        hardened_n_test_signals=HARDENED_N_TEST_SIGNALS, n_test_edges=n_test_edge,
        hardened_n_test_edges=HARDENED_N_TEST_EDGES)

    # ---- Guard C: G0 reproduction (the T2 scoring model IS the hardened G0) ----
    t2_pipe, _ = train_g0()   # exact model T2 uses to score full nodes
    m_t2 = signal_metrics(t2_pipe, te_all)
    g0_nll = float(m_t2["joint_nll_per_signal"])
    g0_brier = float(m_t2["brier_signal_equal"])
    repro_ok = (abs(g0_nll - HARDENED_G0_NLL) < 1e-6
                and abs(g0_brier - HARDENED_G0_BRIER) < 1e-6)
    guards["C_g0_reproduction"] = dict(
        pass_=bool(repro_ok),
        g0_joint_nll=g0_nll, hardened_g0_joint_nll=HARDENED_G0_NLL,
        g0_brier_eqsig=g0_brier, hardened_g0_brier_eqsig=HARDENED_G0_BRIER,
        abs_d_nll=abs(g0_nll - HARDENED_G0_NLL),
        abs_d_brier=abs(g0_brier - HARDENED_G0_BRIER))
    if not repro_ok:
        print("STOP_G2_G0_REPRODUCTION_FAIL")
        print(json.dumps(guards, indent=2, default=str))
        return dict(verdict="STOP_G2_G0_REPRODUCTION_FAIL", guards=guards)

    # ---- Guard D: estimator contract (T2 scoring == attribution G0) ----
    attr_g0_pipe = fit_multinomial(tr_purged, G0_NUM, G0_CAT)

    def _hash(df, cols):
        return hashlib.sha256(df[cols].astype(str).to_csv(index=False).encode()).hexdigest()[:16]

    train_sig_hash = _hash(tr_purged, ["signal_id"])
    train_edge_hash = _hash(tr_purged, ["signal_id", "edge_index", "state_code"])
    feat_spec_hash = hashlib.sha256(
        (",".join(G0_NUM) + "|" + ",".join(G0_CAT)).encode()).hexdigest()[:16]
    coef_match = np.allclose(t2_pipe.named_steps["clf"].coef_,
                             attr_g0_pipe.named_steps["clf"].coef_)
    contract_match = (list(t2_pipe.feature_names_in_) == list(attr_g0_pipe.feature_names_in_)
                      and list(t2_pipe.named_steps["clf"].classes_) == list(attr_g0_pipe.named_steps["clf"].classes_)
                      and bool(coef_match))
    guards["D_model_contract"] = dict(
        pass_=bool(contract_match),
        g0_train_signal_hash=train_sig_hash,
        g0_train_edge_hash=train_edge_hash,
        g0_feature_spec_hash=feat_spec_hash,
        g0_model_contract_match=bool(contract_match),
        t2_features=list(t2_pipe.feature_names_in_),
        g0_features=list(attr_g0_pipe.feature_names_in_),
        t2_classes=list(map(int, t2_pipe.named_steps["clf"].classes_)))

    # ---- Guard E: F0/F1/F2/G0 attribution on transition risk-set ----
    tr_p = tr_purged.copy()
    tr_p["state_code"] = tr_p["state_code"].astype(int)
    te = te_all.copy()
    te["state_code"] = te["state_code"].astype(int)
    attr_rows = []
    metrics = {}
    for name, num, cat in ATTR_SPECS:
        pipe = fit_multinomial(tr_p, num, cat)
        m = signal_metrics(pipe, te)
        metrics[name] = m
        attr_rows.append(dict(model=name, n_train_edges=int(len(tr_p)), n_test_edges=int(len(te)),
            joint_nll_per_signal=float(m["joint_nll_per_signal"]),
            joint_nll_per_edge=float(m["joint_nll_per_edge"]),
            brier_signal_equal=float(m["brier_signal_equal"]),
            brier_path_sum=float(m["brier_path_sum"])))
    # explicit G0 reproduction (Guard C within attribution)
    assert abs(float(metrics["G0"]["joint_nll_per_signal"]) - HARDENED_G0_NLL) < 1e-6, \
        "STOP_G2_G0_REPRODUCTION_FAIL"
    assert abs(float(metrics["G0"]["brier_signal_equal"]) - HARDENED_G0_BRIER) < 1e-6, \
        "STOP_G2_G0_REPRODUCTION_FAIL"
    attr_df = pd.DataFrame(attr_rows)
    attr_df.to_csv(OUT / "g2_feature_attribution.csv", index=False)

    def _dd(b, a, k):
        return metrics[b][k] - metrics[a][k]

    deltas = pd.DataFrame([
        dict(contrast="F1-F0 (+delta_R)", d_joint_nll=_dd("F0", "F1", "joint_nll_per_signal"),
             d_brier=_dd("F0", "F1", "brier_signal_equal")),
        dict(contrast="F2-F0 (+edge_index)", d_joint_nll=_dd("F0", "F2", "joint_nll_per_signal"),
             d_brier=_dd("F0", "F2", "brier_signal_equal")),
        dict(contrast="G0-F1 (edge_index|spacing)", d_joint_nll=_dd("F1", "G0", "joint_nll_per_signal"),
             d_brier=_dd("F1", "G0", "brier_signal_equal")),
        dict(contrast="G0-F2 (delta_R|chain)", d_joint_nll=_dd("F2", "G0", "joint_nll_per_signal"),
             d_brier=_dd("F2", "G0", "brier_signal_equal")),
    ])
    deltas.to_csv(OUT / "g2_feature_attribution_deltas.csv", index=False)
    guards["E_attribution"] = dict(pass_=True,
        g0_joint_nll=float(metrics["G0"]["joint_nll_per_signal"]),
        g0_brier=float(metrics["G0"]["brier_signal_equal"]),
        reported=attr_rows)

    # ---- Guard F: exact full-node cache provenance (P0-8) ----
    # load_full_node_tb2 STOP_CLOSURE_CACHE_MISSING if any symbol's exact cache is
    # missing or builder_version/master_hash/contacts_hash/sig_hash mismatch.
    features, outcomes, provenance = load_full_node_tb2()
    if provenance is None:
        return dict(verdict="STOP_CLOSURE_CACHE_MISSING", guards=guards)
    cache_ok = bool(provenance) and all(
        p["builder_version"] == BUILDER_VERSION for p in provenance.values())
    guards["F_cache_provenance"] = dict(pass_=bool(cache_ok), n_symbols=len(provenance),
        provenance={s: {k: p[k] for k in ("builder_version", "master_hash",
                  "contacts_hash", "sig_hash", "n_signals")} for s, p in provenance.items()})
    if not cache_ok:
        print("STOP_CLOSURE_CACHE_PROVENANCE_FAIL")
        print(json.dumps(guards["F_cache_provenance"], indent=2, default=str))
        return dict(verdict="STOP_CLOSURE_CACHE_PROVENANCE_FAIL", guards=guards)

    # ---- Guard G: global node-key uniqueness (P0-7) ----
    # Enforced by assertions inside load_full_node_tb2 (raises STOP_G2_NODE_KEY_COLLISION
    # if violated). Reaching here means the one-to-one merge succeeded.
    feat_key = ["signal_gid", "edge_index"]
    g_ok = (not features.duplicated(feat_key).any()
            and not outcomes.duplicated(feat_key).any()
            and len(features) == len(outcomes))
    guards["G_key_uniqueness"] = dict(pass_=bool(g_ok), n_nodes=int(len(features)),
                                      n_outcomes=int(len(outcomes)))

    # ---- Guards H/I: deterministic economic replay vs freshly written g2_summary ----
    bars_cache = {sym: load_raw_bars(sym) for sym in FULL_UNIV}
    i0_pipe = train_i0(features, outcomes)
    ev_g, _, _, _ = score_g0_nodes(t2_pipe, features)
    ev_i, _, _, _ = score_i0_nodes(i0_pipe, features)
    features = features.copy()
    features["graph_ev"] = ev_g
    features["indep_ev"] = ev_i
    test_mask = features["block"].to_numpy() == "TB2"
    test_feats = features[test_mask].reset_index(drop=True)
    assert (test_feats["block"] == "TB2").all(), "STOP_SEQUENTIAL_NON_TB2_INPUT"
    sel_test = select_targets(test_feats, test_feats["graph_ev"].to_numpy(),
                              test_feats["indep_ev"].to_numpy())
    rewards_test = compute_rewards(test_feats, sel_test, bars_cache)
    r_t0 = rewards_test["reward_t0"].to_numpy()
    r_t1 = rewards_test["reward_t1"].to_numpy()
    r_t2 = rewards_test["reward_t2"].to_numpy()
    dd = rewards_test["trading_day"].to_numpy()
    ev_replay = dict(t0=float(r_t0.mean()), t1=float(r_t1.mean()), t2=float(r_t2.mean()))
    bs_replay = paired_bootstrap_by_day(r_t2 - r_t0, dd)
    # compare to g2_summary.csv that run_g2 just regenerated in THIS session
    prev = pd.read_csv(OUT / "g2_summary.csv")
    prev_ev = dict(zip(prev["selector"], prev["EV_per_signal"]))
    h_match = abs(ev_replay["t1"] - prev_ev["T1_IndependentEV"]) < 1e-9
    i_match = (abs(ev_replay["t0"] - prev_ev["T0_Nearest"]) < 1e-9
               and abs(ev_replay["t2"] - prev_ev["T2_GraphEV"]) < 1e-9)
    prev_bs = pd.read_csv(OUT / "g2_bootstrap.csv")
    prev_bs_t20 = prev_bs[prev_bs["compare"] == "T2-T0"].iloc[0]
    bs_match = (abs(bs_replay["ci_lo"] - float(prev_bs_t20["ci_lo"])) < 1e-6
               and abs(bs_replay["ci_hi"] - float(prev_bs_t20["ci_hi"])) < 1e-6)
    guards["H_t1_replay"] = dict(pass_=bool(h_match), replay_t1=ev_replay["t1"],
                                 g2_summary_t1=prev_ev["T1_IndependentEV"])
    guards["I_t0_t2_replay"] = dict(pass_=bool(i_match and bs_match),
        replay_t0=ev_replay["t0"], g2_summary_t0=prev_ev["T0_Nearest"],
        replay_t2=ev_replay["t2"], g2_summary_t2=prev_ev["T2_GraphEV"],
        replay_T2T0_ci=[bs_replay["ci_lo"], bs_replay["ci_hi"]],
        g2_summary_T2T0_ci=[float(prev_bs_t20["ci_lo"]), float(prev_bs_t20["ci_hi"])],
        ev_match=bool(i_match), bs_match=bool(bs_match))
    rewards_test.to_csv(OUT / "g2_closure_replay_rewards.csv", index=False)

    # ---- Guard J: same-bar diagnostic signal universe (P0-10) ----
    sb = samebar_multinode_diagnostic(test_feats, outcomes)
    j_ok = (int(sb["n_test_signals"]) == HARDENED_N_TEST_SIGNALS
            and int(sb["n_test_signals"]) == int(test_feats["signal_gid"].nunique()))
    guards["J_samebar_signal_universe"] = dict(pass_=bool(j_ok), **sb,
        expected_n_test_signals=HARDENED_N_TEST_SIGNALS)

    # ---- Guard K: execution oracle parity (P0-2) ----
    k_ok = run_execution_parity(n_random=600, seed=1)
    guards["K_execution_parity"] = dict(pass_=bool(k_ok))

    # ---- Guard L: WF1 sequential TB2-only (P0-4) ----
    seq_executed = 0
    for name in ("t0", "t1", "t2"):
        sc = sequential_curve(test_feats, sel_test, bars_cache, which=name)
        seq_executed += int(sc["executed"].sum())
    l_ok = (int(test_feats["signal_gid"].nunique()) == HARDENED_N_TEST_SIGNALS)
    guards["L_sequential_tb2_only"] = dict(pass_=bool(l_ok),
        n_input_signals=int(test_feats["signal_gid"].nunique()),
        expected_n_test_signals=HARDENED_N_TEST_SIGNALS, n_sequential_executed=seq_executed)

    # ---- final closure judgement ----
    all_pass = all(g.get("pass_", False) for g in guards.values())
    if all_pass:
        # economic verdict from FRESHLY computed T0/T1/T2 (no presupposition of old result)
        A = ev_replay["t2"] > ev_replay["t0"]
        B = ev_replay["t2"] > ev_replay["t1"]
        C = bs_replay["ci_lo"] > 0
        D = (paired_bootstrap_by_day(r_t2 - r_t1, dd)["ci_lo"] > 0)
        if A and B and C and D:
            verdict = "GRAPH_TARGET_SELECTION_WF1_PASS"
        elif (A and B) and not (C and D):
            verdict = "GRAPH_TARGET_SELECTION_WF1_WEAK"
        else:
            verdict = "NO_GRAPH_TARGET_ECONOMIC_INCREMENT_WF1"
        closure = "CLOSED"
    else:
        verdict = "CLOSURE_FAIL"
        closure = "CLOSURE_FAIL"
    report = dict(
        closure=closure,
        verdict=verdict,
        g0_probability_structure="GRAPH_STRUCTURE_NECESSITY_WF1_HARDENED_PASS",
        economic_decision=verdict,
        guards=guards,
        seconds=time.perf_counter() - t0,
    )
    pd.DataFrame([dict(guard=k, pass_=bool(v.get("pass_", False)),
                       detail=str({kk: vv for kk, vv in v.items() if kk != "pass_"}))
                  for k, v in guards.items()]).to_csv(OUT / "g2_closure_guards.csv", index=False)
    with open(OUT / "g2_closure_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    print("\n=== G2 CLOSURE AUDIT ===")
    for k, v in guards.items():
        print(f"  [{k}] {'PASS' if v.get('pass_') else 'FAIL'}")
    print(f"  closure={closure} verdict={verdict}")
    print(f"  G0 reproduction: jNLL={g0_nll:.6f} (hardened {HARDENED_G0_NLL:.6f}) "
          f"brier={g0_brier:.6f} (hardened {HARDENED_G0_BRIER:.6f})")
    print(f"  economic replay EV/sig: T0={ev_replay['t0']:+.5f} "
          f"T1={ev_replay['t1']:+.5f} T2={ev_replay['t2']:+.5f}")
    print(f"  T2-T0 bootstrap CI=[{bs_replay['ci_lo']:+.4f},{bs_replay['ci_hi']:+.4f}]")
    print(f"  [{time.perf_counter() - t0:.1f}s]")
    return report


def run_execution_parity(n_random=600, seed=1):
    """P0-2: scalar-oracle execution parity vs frozen run_fixed_execution_baseline_v1.execute_path.
    Returns True iff ZERO mismatches across random + synthetic cases."""
    from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import execute_path
    rng = np.random.default_rng(seed)
    fails = []

    def check(direction, entry_open, stop_px, target_px, O, H, L, tag):
        W = len(O)
        bars = dict(o=O, h=H, l=L, c=np.empty(W), n=W,
                    t=np.arange(W), disc=np.zeros(W, dtype=bool))
        ref = execute_path(bars, 0, W, direction, stop_px, target_px, stop_first=True)
        R, out, k, ex = reward_for_signal(direction, entry_open, stop_px, target_px, O, H, L)
        if ref is None:
            if out != "CENSOR":
                fails.append(f"{tag}: oracle None but out={out}")
            return
        if out != ref["outcome"]:
            fails.append(f"{tag}: outcome {out} != {ref['outcome']}")
            return
        if k != ref["exit_bar"]:
            fails.append(f"{tag}: exit_bar {k} != {ref['exit_bar']}")
        if abs(ex - ref["exit_px"]) > 1e-9:
            fails.append(f"{tag}: exit_px {ex} != {ref['exit_px']}")
        risk_px = direction * (entry_open - stop_px)
        if abs(risk_px) > 0:
            Rref = direction * (ref["exit_px"] - entry_open) / risk_px
            if abs(R - Rref) > 1e-9:
                fails.append(f"{tag}: R {R} != {Rref}")
        if ref["same_bar_ambiguous"] and out != "STOP":
            fails.append(f"{tag}: ambiguous but outcome {out}")

    # 500+ random cases
    for _ in range(n_random):
        direction = 1 if rng.random() < 0.5 else -1
        entry_open = rng.uniform(100, 200)
        atr = rng.uniform(0.5, 5)
        stop_px = entry_open - direction * atr
        tgt_dist = rng.uniform(atr * 1.1, atr * 5)
        target_px = entry_open + direction * tgt_dist
        W = int(rng.integers(2, 40))
        base = rng.uniform(90, 210, size=W)
        o = rng.uniform(95, 205, size=W)
        h = np.maximum(o, base) + rng.uniform(0, 3, size=W)
        l = np.minimum(o, base) - rng.uniform(0, 3, size=W)
        check(direction, entry_open, stop_px, target_px, o, h, l, "rand")

    # synthetic hand-built cases (P0-2 explicit list)
    cases = [
        (1, 100.0, 99.0, 101.0, [100,100,100], [100,101.5,100], [100,99,100], "L_target_normal"),
        (1, 100.0, 99.0, 101.0, [100,102,100], [100,102,100], [100,101,100], "L_target_gap"),
        (1, 100.0, 99.0, 101.0, [100,100,100], [100,100,100], [100,98,100], "L_stop_normal"),
        (1, 100.0, 99.0, 101.0, [100,97,100], [100,97,100], [100,97,100], "L_stop_gap"),
        (-1, 100.0, 101.0, 99.0, [100,100,100], [100,100,100], [100,98,100], "S_target_normal"),
        (-1, 100.0, 101.0, 99.0, [100,98,100], [100,98,100], [100,98,100], "S_target_gap"),
        (-1, 100.0, 101.0, 99.0, [100,100,100], [100,102,100], [100,100,100], "S_stop_normal"),
        (-1, 100.0, 101.0, 99.0, [100,103,100], [100,103,100], [100,103,100], "S_stop_gap"),
        (1, 100.0, 99.0, 101.0, [100,100,100], [100,101.5,100], [100,98,100], "samebar_stop_first"),
        (1, 100.0, 99.0, 101.0, [100,100,100], [100,100,100], [100,100,100], "censor"),
    ]
    for direction, eo, sp, tp, o, h, l, tag in cases:
        check(direction, eo, sp, tp, np.array(o, dtype=float), np.array(h, dtype=float),
              np.array(l, dtype=float), tag)

    if fails:
        print("STOP_G2_EXECUTION_PARITY_FAIL")
        for f in fails[:30]:
            print("  " + f)
        return False
    print(f"EXECUTION PARITY OK: reward_for_signal matches frozen execute_path "
          f"({n_random} random + {len(cases)} synthetic cases, 0 mismatch)")
    return True


def run_component_tests():
    """Lightweight correctness harness (no economics). Returns True if all pass."""
    rng = np.random.default_rng(0)
    fails = []
    # 1) segmented math vs scalar loop
    for _ in range(30):
        n = int(rng.integers(5, 300))
        g = rng.integers(0, 6, size=n)
        _, starts, lengths, _ = segment_meta(g)
        x = rng.random(n)
        if not np.allclose(segmented_cumsum(x, starts, lengths),
                           _segmented_cumsum_scalar(x, starts, lengths)):
            fails.append("segmented_cumsum mismatch"); break
    for _ in range(30):
        n = int(rng.integers(5, 300))
        g = rng.integers(0, 6, size=n)
        _, starts, lengths, _ = segment_meta(g)
        v = rng.random(n)
        if not np.array_equal(segmented_argmax(v, starts, lengths),
                              _segmented_argmax_scalar(v, starts, lengths)):
            fails.append("segmented_argmax mismatch"); break
    for _ in range(30):
        n = int(rng.integers(5, 300))
        g = rng.integers(0, 6, size=n)
        _, starts, lengths, _ = segment_meta(g)
        p = rng.random(n) * 0.9 + 0.05
        cpv = segmented_cumprod(p, starts, lengths)
        cps = np.exp(_segmented_cumsum_scalar(np.log(p), starts, lengths))
        if not np.allclose(cpv, cps):
            fails.append("segmented_cumprod mismatch"); break
    # 2) graph probability invariants
    for _ in range(30):
        n = int(rng.integers(3, 80))
        g = np.repeat(np.arange(n // 4 + 1), 4)[:n]
        _, starts, lengths, _ = segment_meta(g)
        p_next = rng.random(n) * 0.9 + 0.05
        p_loss = rng.random(n) * 0.4
        p_censor = rng.random(n) * 0.4
        s = p_next + p_loss + p_censor
        p_next /= s; p_loss /= s; p_censor /= s
        rr = rng.random(n) * 3
        ev, p_reach, plb, pcb = compute_graph_ev(p_next, p_loss, p_censor, g, rr)
        if not np.allclose(p_reach + plb + pcb, 1.0, atol=1e-6):
            fails.append("graph mass != 1"); break
        for i in range(len(starts)):
            s0, l0 = int(starts[i]), int(lengths[i])
            if l0 > 1 and not np.all(np.diff(p_reach[s0:s0 + l0]) <= 1e-9):
                fails.append("p_reach not monotonic"); break
    # 3) node-builder parity vs scalar oracle (P0)
    if not run_p0_parity(symbols=("AG",), first_n=100):
        fails.append("P0 node-builder parity FAIL")
    # 4) execution parity vs scalar oracle (P0-2)
    if not run_execution_parity(n_random=600, seed=1):
        fails.append("execution parity FAIL")
    if fails:
        print("STOP_COMPONENT_TEST_FAIL")
        for f in fails[:10]:
            print("  " + f)
        return False
    print("COMPONENT_TESTS OK (segmented math / graph invariants / node-builder parity)")
    return True


# ---------------------------------------------------------------------------
# P0 parity vs scalar reference (signal_clusters)
# ---------------------------------------------------------------------------
def run_p0_parity(symbols=("AG",), first_n=100):
    master = load_master()
    contacts = load_contacts()
    contacts["block"] = assign_blocks(contacts)
    bars_cache = {}
    fails = []
    for sym in symbols:
        cs = contacts[contacts["symbol"] == sym].reset_index(drop=True)
        if first_n:
            cs = cs.iloc[:first_n].reset_index(drop=True)
        ms = master[master["symbol"] == sym]
        bars = bars_cache.get(sym) or load_raw_bars(sym)
        bars_cache[sym] = bars
        from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import build_enrichment
        enr = build_enrichment(master, bars_cache, symbols=[sym])
        # vectorized
        feats, _ = build_full_nodes_symbol(sym, cs, ms, bars, BUILDER_VERSION)
        vec = feats.sort_values(["signal_gid", "edge_index"]).reset_index(drop=True)
        # oracle (scalar) — only for signals that vectorized produced (active>0)
        oracle_rows = []
        for i in range(len(cs)):
            r = cs.iloc[i]
            cl = signal_clusters(sym, r, ms, enr, bars)
            if cl is None:
                continue
            f = cl["feats"]
            order = np.argsort(f["distance_R"], kind="stable")
            prev_d = 0.0
            for k in order:
                dr = float(f["distance_R"][k])
                delta_R = dr - prev_d
                prev_d = dr
                oracle_rows.append(dict(
                    signal_gid=i, edge_index=int(k),
                    price=float(f["price"][k]), distance_R=dr,
                    delta_R=delta_R, cum_distance_R=dr,
                ))
        ora = pd.DataFrame(oracle_rows)
        if len(ora) == 0:
            continue
        # G2-01 candidate set parity (per signal, set of prices)
        for g in ora["signal_gid"].unique():
            op = set(np.round(ora[ora.signal_gid == g]["price"].to_numpy(), 6))
            vp = set(np.round(vec[vec.signal_gid == g]["target_price"].to_numpy(), 6))
            if op != vp:
                fails.append(f"G2-01 candidate_set sym={sym} gid={g} "
                             f"missing={op ^ vp}")
                break
        # G2-02 same-price collapse count parity
        if len(ora) != len(vec):
            fails.append(f"G2-02 node_count sym={sym} oracle={len(ora)} vec={len(vec)}")
        # G2-04 edge_index parity + G2-05 delta_R parity
        m = ora.merge(vec, on=["signal_gid", "edge_index"], suffixes=("_o", "_v"))
        if not np.allclose(m["delta_R_o"], m["delta_R_v"], atol=1e-9):
            fails.append(f"G2-05 delta_R mismatch sym={sym}")
        if not np.allclose(m["distance_R"], m["cum_distance_R_v"], atol=1e-9):
            fails.append(f"G2-03 distanceR mismatch sym={sym}")
    if fails:
        print("STOP_G2_NODE_BUILDER_PARITY_FAIL")
        for f in fails[:20]:
            print("  " + f)
        return False
    print("P0 parity OK: vectorized full-node builder matches scalar reference "
          f"({first_n} signals x {symbols})")
    return True


# ---------------------------------------------------------------------------
# Global signal key (P0-5/P0-6): ONE shared helper for main run and closure
# ---------------------------------------------------------------------------
def globalize_node_ids(feats, outs, offset):
    """Make signal_gid globally unique across symbols. offset advances by
    (max local gid + 1), NOT nunique, so zero-node / gap gids stay unique.
    Returns (feats, outs, new_offset)."""
    feats = feats.copy()
    outs = outs.copy()
    local_max = int(feats["signal_gid"].max())
    feats["signal_gid"] = feats["signal_gid"].astype(np.int64) + offset
    outs["signal_gid"] = outs["signal_gid"].astype(np.int64) + offset
    return feats, outs, offset + local_max + 1


# ---------------------------------------------------------------------------
# Core run
# ---------------------------------------------------------------------------
def run_g2(symbols, max_signals, force_rebuild, cache_nodes):
    master = load_master()
    contacts = load_contacts()
    contacts["block"] = assign_blocks(contacts)
    if max_signals:
        parts = []
        for s in contacts["symbol"].unique():
            g = contacts[contacts["symbol"] == s].sort_values("decision_time")
            n = len(g)
            if n > max_signals:
                # spread across the date range so all TB blocks are represented
                idx = np.linspace(0, n - 1, max_signals).round().astype(int)
                g = g.iloc[idx]
            parts.append(g)
        contacts = pd.concat(parts, ignore_index=True)
    bars_cache = {}
    feats_all = []
    outs_all = []
    offset = 0
    for sym in symbols:
        cs = contacts[contacts["symbol"] == sym]
        ms = master[master["symbol"] == sym]
        bars = bars_cache.get(sym) or load_raw_bars(sym)
        bars_cache[sym] = bars
        feats, outs = build_or_load_nodes(sym, cs, ms, bars, force=force_rebuild,
                                          cache=cache_nodes)
        # P0-5/P0-6: one shared helper to globalize signal_gid across symbols.
        feats, outs, offset = globalize_node_ids(feats, outs, offset)
        feats_all.append(feats)
        outs_all.append(outs)
    features = pd.concat(feats_all, ignore_index=True)
    outcomes = pd.concat(outs_all, ignore_index=True)
    # P0-7: node-key uniqueness guards (one-to-one, no cross-symbol collision).
    feat_key = ["signal_gid", "edge_index"]
    assert not features.duplicated(feat_key).any(), "STOP_G2_NODE_KEY_COLLISION"
    assert not outcomes.duplicated(feat_key).any(), "STOP_G2_NODE_KEY_COLLISION"
    assert len(features) == len(outcomes), "STOP_G2_NODE_KEY_COLLISION"
    features.merge(outcomes, on=feat_key, validate="one_to_one")

    # G0 (frozen) + I0 (full nodes)
    g0_pipe, _ = train_g0()
    i0_pipe = train_i0(features, outcomes)

    ev_graph, p_reach, p_loss_b, p_censor_b = score_g0_nodes(g0_pipe, features)
    ev_indep, p_target, p_loss_i, p_censor_i = score_i0_nodes(i0_pipe, features)
    features = features.copy()
    features["graph_ev"] = ev_graph
    features["indep_ev"] = ev_indep

    sel = select_targets(features, ev_graph, ev_indep)

    # restrict to TB2 test for the primary gate
    test_mask = features["block"].to_numpy() == "TB2"
    test_feats = features[test_mask].reset_index(drop=True)
    if len(test_feats) == 0:
        print("[G2] WARNING: empty TB2 test set — skip gate (mechanics-only run)")
        return dict(verdict="NO_TEST_SET", paired={}, seq={}, bootstrap=None, ev=None)
    # recompute sel on test subset
    ev_g_test = test_feats["graph_ev"].to_numpy()
    ev_i_test = test_feats["indep_ev"].to_numpy()
    sel_test = select_targets(test_feats, ev_g_test, ev_i_test)
    rewards_test = compute_rewards(test_feats, sel_test, bars_cache)

    # ---- paired metrics ----
    paired = {}
    for name in ("t0", "t1", "t2"):
        paired[name] = compute_metrics(rewards_test[f"reward_{name}"].to_numpy(),
                                       rewards_test["trading_day"].to_numpy())

    # ---- primary gate (paired counterfactual) ----
    r_t0 = rewards_test["reward_t0"].to_numpy()
    r_t1 = rewards_test["reward_t1"].to_numpy()
    r_t2 = rewards_test["reward_t2"].to_numpy()
    dd = rewards_test["trading_day"].to_numpy()
    ev_t0, ev_t1, ev_t2 = r_t0.mean(), r_t1.mean(), r_t2.mean()
    bs_t2_t0 = paired_bootstrap_by_day(r_t2 - r_t0, dd)
    bs_t2_t1 = paired_bootstrap_by_day(r_t2 - r_t1, dd)
    A = ev_t2 > ev_t0
    B = ev_t2 > ev_t1
    C = bs_t2_t0["ci_lo"] > 0
    D = bs_t2_t1["ci_lo"] > 0
    if A and B and C and D:
        verdict = "GRAPH_TARGET_SELECTION_WF1_PASS"
    elif (A and B) and (not (C and D)):
        verdict = "GRAPH_TARGET_SELECTION_WF1_WEAK"
    else:
        verdict = "NO_GRAPH_TARGET_ECONOMIC_INCREMENT_WF1"

    # ---- sequential (one position per symbol) — P0-4: TB2-only ----
    assert (test_feats["block"] == "TB2").all(), "STOP_SEQUENTIAL_NON_TB2_INPUT"
    print(f"[SEQ] n_input_signals(TB2)={int(test_feats['signal_gid'].nunique())}")
    seq = {}
    for name in ("t0", "t1", "t2"):
        sc = sequential_curve(test_feats, sel_test, bars_cache, which=name)
        ex = sc[sc["executed"]]
        seq[name] = compute_metrics(ex["realized_R"].to_numpy(), ex["trading_day"].to_numpy(),
                                     label="trades")

    # ---- G0 Feature Attribution (fixed nested models; EXACT hardened transition
    #      risk-set; NLL/Brier only; NOT in economic PASS/FAIL) ----
    g0_feature_attribution()

    # ---- T2 node EV decomposition (reward_term / loss_term / graph_ev) ----
    t2_node_decomposition(test_feats, g0_pipe, rewards_test, sel_test)

    # ---- Section D: Same-Bar Multi-Node Diagnostic (future-path crossing; reads
    #      outcome table ONLY, post-selection, never enters selector) ----
    samebar_multinode_diagnostic(test_feats, outcomes)

    # ---- candidate density diagnostic (renamed from old same-bar; per-signal node
    #      count, kept as a secondary descriptive view) ----
    candidate_density_diagnostic(test_feats, rewards_test, sel_test)

    # ---- save artifacts ----
    summ = pd.DataFrame([
        dict(selector="T0_Nearest", **paired["t0"]),
        dict(selector="T1_IndependentEV", **paired["t1"]),
        dict(selector="T2_GraphEV", **paired["t2"]),
    ])
    summ.to_csv(OUT / "g2_summary.csv", index=False)

    sseq = pd.DataFrame([
        dict(selector="T0_Nearest", **seq["t0"]),
        dict(selector="T1_IndependentEV", **seq["t1"]),
        dict(selector="T2_GraphEV", **seq["t2"]),
    ])
    sseq.to_csv(OUT / "g2_sequential_summary.csv", index=False)

    boot = pd.DataFrame([
        dict(compare="T2-T0", **bs_t2_t0),
        dict(compare="T2-T1", **bs_t2_t1),
    ])
    boot.to_csv(OUT / "g2_bootstrap.csv", index=False)

    # equity curves (paired, chronological by decision_time) — per-signal aligned
    eq = OUT / "g2_paired_equity_curve.csv"
    dt0 = pd.to_datetime(rewards_test["decision_time"]).to_numpy()
    order = np.argsort(dt0)
    pd.DataFrame({
        "decision_time": dt0[order],
        "T0": np.cumprod(1.0 + 0.005 * rewards_test["reward_t0"].to_numpy()[order]),
        "T1": np.cumprod(1.0 + 0.005 * rewards_test["reward_t1"].to_numpy()[order]),
        "T2": np.cumprod(1.0 + 0.005 * rewards_test["reward_t2"].to_numpy()[order]),
    }).to_csv(eq, index=False)

    scope = "WF1" if set(symbols) == set(FULL_UNIV) else f"SUBSET({len(symbols)}sym)"
    print(f"[G2] verdict={verdict} [{scope}; not authoritative until full 15-symbol WF1]")
    print(f"    EV/sig  T0={ev_t0:+.5f} T1={ev_t1:+.5f} T2={ev_t2:+.5f}")
    print(f"    T2-T0 boot CI=[{bs_t2_t0['ci_lo']:+.4f},{bs_t2_t0['ci_hi']:+.4f}] "
          f"T2-T1 CI=[{bs_t2_t1['ci_lo']:+.4f},{bs_t2_t1['ci_hi']:+.4f}]")
    for name in ("t0", "t1", "t2"):
        print(f"    [{name}] paired win={paired[name].get('win_rate'):.3f} "
              f"pf={paired[name].get('profit_factor')} totalR={paired[name].get('total_R'):.1f} "
              f"maxDD={paired[name].get('max_DD_R'):.1f}")
    return dict(verdict=verdict, paired=paired, seq=seq, bootstrap=boot,
                ev=(ev_t0, ev_t1, ev_t2))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["p0", "smoke", "pilot", "wf1", "test", "closure"])
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--max-signals", type=int, default=None)
    ap.add_argument("--force-rebuild", action="store_true")
    ap.add_argument("--no-cache-nodes", dest="cache_nodes", action="store_false")
    ap.set_defaults(cache_nodes=True)
    args = ap.parse_args()

    t_total = time.perf_counter()
    if args.mode == "p0":
        ok = run_p0_parity(symbols=("AG",), first_n=100)
        print("P0_PARITY_OK" if ok else "P0_PARITY_FAIL")
        return

    if args.mode == "smoke":
        univ = args.symbols or ["AG"]
        cap = args.max_signals or 100
        run_g2(univ, cap, args.force_rebuild, args.cache_nodes)
    elif args.mode == "pilot":
        univ = args.symbols or ["AG", "CU", "RB", "MA"]
        cap = args.max_signals or 500
        run_g2(univ, cap, args.force_rebuild, args.cache_nodes)
    elif args.mode == "wf1":
        univ = args.symbols or list(FULL_UNIV)
        cap = args.max_signals  # None = full
        run_g2(univ, cap, args.force_rebuild, args.cache_nodes)
    elif args.mode == "test":
        # full component test harness: segmented math vs scalar, graph invariants,
        # node-builder parity vs scalar oracle (P0 also exercises execution sanity).
        # NO economics.
        ok = run_component_tests()
        print("RESULT", "OK" if ok else "FAIL")
        return

    if args.mode == "closure":
        # reproducibility closure audit: reuse transition + full-node caches, NO rebuild.
        rep = run_g2_closure_audit()
        print("CLOSURE", rep.get("closure"))
        return

    print(f"[DONE] g2 {args.mode} ({time.perf_counter()-t_total:.1f}s)")


if __name__ == "__main__":
    main()
