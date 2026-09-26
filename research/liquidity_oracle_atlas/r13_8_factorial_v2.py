"""FUTURE-R13.8-V2 -- Causal A9/E9 x Value-Gate factorial experiment.

SUPERSEDES the prior E9-only R13.8.

Scientific question (plan §0-§3):
    The Direction layer is NOT resolved. A9 (frozen DTP9 router) and E9
    (gated META10 LONG/SHORT correctness experts) must BOTH be evaluated
    under the same causal TRAIN-OOF framework. The value gate (P vs PGM3 vs
    EV_C control) is a SEPARATE factor. The experiment decomposes:

        Direction in {A9, E9}
        x Value architecture in {A0_V1_DISJOINT, A1_SHARE_TO_WIN}
        x Gate in {P, PGM3, EV_C_NEGATIVE_CONTROL}
        = 12 cells

    with pre-registered contrasts:
        dGate | A9  = R(A9,PGM) - R(A9,P)
        dGate | E9  = R(E9,PGM) - R(E9,P)
        dDir  | P   = R(E9,P)   - R(A9,P)
        dDir  | PGM = R(E9,PGM) - R(A9,PGM)
        Interaction I = dGate|E9 - dGate|A9

Hard constraints (plan §30, §51-§56):
    * Only p_win/mu_win/mu_loss enter PGM3 (reuse R13.7 PGM3 verbatim).
    * A9 == chain["A"]; A9 probability == chain["A_p"] (NOT p_te).
    * E9 uses META10 gated experts; NO M9/M33/E33 fits (governance §73).
    * Causal Direction axis: 5 chain runs, each produces BOTH A9 and E9.
    * No DEV VAL, no old TEST, no closed-TEST Direction axis, no base refit.
    * Prediction artifacts contain NO outcome (episode_return_atr/win/
      teacher/oracle direction).
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
    build_frozen_split,
    _fit_m0,
    _predict_m0,
)
from research.liquidity_oracle_atlas.direction_gated_experts_v1 import (
    fit_fixed_router,
    build_prequential_router_oof,
    build_meta_features,
    fit_expert_pair,
    predict_experts,
    prepare_xy,
    DTP9,
    build_direction_expert_data,
)
from research.liquidity_oracle_atlas import run_decomposed_v2_research as R
from research.liquidity_oracle_atlas.decomposed_value_closure_audit_v1 import (
    load_td5_oof,
    sha256_of,
)
from research.liquidity_oracle_atlas.meta_output_integration_v1 import (
    PGM3,
    meta_state,
)

# --------------------------------------------------------------------------- #
# Constants                                                                    #
# --------------------------------------------------------------------------- #
HORIZON = "td5"
N_FOLDS = 5
BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_BLOCK = 5

REVIEWED_PARENT_SHA = "047342d15d22d028a732c54e364bad14c1a6283a"
R13_7_CODE_SHA = "98e499ce356a466f1160b4c55290d5106bc6e5e5"
R13_7_EVIDENCE_SHA = "047342d15d22d028a732c54e364bad14c1a6283a"

ARCHS = {"A0": "A0_V1_DISJOINT", "A1": "A1_SHARE_TO_WIN"}
DIR_SYSTEMS = ["A9", "E9"]
GATES = ["P", "PGM3", "EV_C"]
KEY = ["symbol", "decision_bar", "side", "horizon"]

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
ARTIFACT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "decomposed_value_v2")
EVIDENCE_DIR = R.EVIDENCE_DIR
if not os.path.isabs(EVIDENCE_DIR):
    EVIDENCE_DIR = os.path.join(PROJECT_ROOT, EVIDENCE_DIR)

DIRECTION_AXIS_PARQUET = os.path.join(
    ARTIFACT_DIR, "a9_e9_train_oof_axis_v2.parquet")
FACTORIAL_OOF_PARQUET = os.path.join(
    ARTIFACT_DIR, "direction_value_factorial_oof_v2.parquet")

DIR_SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "r13_8_direction_summary_v2.json")
DIR_FOLD_CSV = os.path.join(EVIDENCE_DIR, "r13_8_direction_fold_v2.csv")
DIR_DISAGREE_CSV = os.path.join(EVIDENCE_DIR, "r13_8_direction_disagreement_v2.csv")
FACT_MODELS_CSV = os.path.join(EVIDENCE_DIR, "r13_8_factorial_models_v2.csv")
FACT_FOLD_CSV = os.path.join(EVIDENCE_DIR, "r13_8_factorial_fold_v2.csv")
FACT_MATCHED_CSV = os.path.join(EVIDENCE_DIR, "r13_8_factorial_matched_coverage_v2.csv")
FACT_CONTRAST_CSV = os.path.join(EVIDENCE_DIR, "r13_8_factorial_contrasts_v2.csv")
PGM_REGIMES_CSV = os.path.join(EVIDENCE_DIR, "r13_8_pgm_regimes_v2.csv")
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "r13_8_manifest_v2.json")
# ---- Reviewer repair (NEED_REVISION) evidence paths ----
PAIR_AUDIT_JSON = os.path.join(EVIDENCE_DIR, "r13_8_pair_universe_audit_v2.json")
ROUTER_GATE_CSV = os.path.join(EVIDENCE_DIR, "r13_8_router_probability_gate_v2.csv")
MATCHEDN_CSV = os.path.join(EVIDENCE_DIR, "r13_8_matchedN_p_vs_pgm_v2.csv")


# --------------------------------------------------------------------------- #
# Direction canonical API (thin, frozen-A + E9-only; no M9/M33/E33)            #
# --------------------------------------------------------------------------- #
def run_axis_chain(data, ds, train_idx, val_idx, test_idx, n_estimators=None):
    """Reproduce chain["A"] and chain["E9"] ONLY (canonical frozen machinery).

    Reuses fit_fixed_router / build_prequential_router_oof / fit_expert_pair /
    predict_experts verbatim -> identical A/E9 outputs to direction_gated_experts_v1,
    but never fits M9/M33/E33 (governance §73: those fits = 0).
    """
    m0 = _fit_m0(ds, train_idx, val_idx)
    if n_estimators is None:
        n_estimators = int(m0.best_iteration_)
    pred_a, p_a = _predict_m0(m0, ds, test_idx)

    Xtr9, ytr, wtr = prepare_xy(ds, train_idx, DTP9)
    router = fit_fixed_router(Xtr9, ytr, wtr, n_estimators)
    Xte9, _, _ = prepare_xy(ds, test_idx, DTP9)
    Xva9, _, _ = prepare_xy(ds, val_idx, DTP9)
    p_te = router.predict_proba(Xte9)[:, 1]
    p_va = router.predict_proba(Xva9)[:, 1]
    router_te = (p_te >= 0.5).astype(np.int8)
    router_va = (p_va >= 0.5).astype(np.int8)

    # fail-closed: fixed router must reproduce frozen A on test
    if not np.array_equal(router_te, np.asarray(pred_a, dtype=np.int8)):
        raise RuntimeError("STOP_FIXED_ROUTER_A_PREDICTION_MISMATCH")

    oof = build_prequential_router_oof(ds, data, train_idx, n_estimators,
                                       return_audit=True)
    avail_tr = oof["available"][train_idx]
    tr_gate = train_idx[avail_tr]
    oof_pred = oof["pred"][tr_gate]
    oof_p = oof["p_long"][tr_gate]

    m10_te, _, _ = build_meta_features(data.X9[test_idx], data.X33[test_idx],
                                       router_te, p_te)
    m10_tr, _, _ = build_meta_features(data.X9[tr_gate], data.X33[tr_gate],
                                       oof_pred, oof_p)
    m10_va, _, _ = build_meta_features(data.X9[val_idx], data.X33[val_idx],
                                       router_va, p_va)
    gid_tr = data.gid[tr_gate]
    gid_va = data.gid[val_idx]
    y_meta_tr = (np.asarray(oof_pred, dtype=np.int8) == data.y[tr_gate]).astype(np.uint8)
    y_meta_va = (np.asarray(router_va, dtype=np.int8) == data.y[val_idx]).astype(np.uint8)

    e9 = fit_expert_pair(m10_tr, y_meta_tr, gid_tr, oof_pred,
                         m10_va, y_meta_va, gid_va, router_va)
    fin_e9, pe9 = predict_experts(e9[0], e9[1], m10_te, router_te)

    return {
        "n_estimators": int(n_estimators),
        "A": np.asarray(pred_a, dtype=np.uint8),
        "E9": np.asarray(fin_e9, dtype=np.uint8),
        "A_p": np.asarray(p_a, dtype=np.float64),
        "router_p_long": np.asarray(p_te, dtype=np.float64),
        "router_te": router_te,
        "e9_p_correct": np.asarray(pe9, dtype=np.float64),
        "models": {"router": router, "e9": e9},
        "oof": oof,
    }


# --------------------------------------------------------------------------- #
# Value ROOT universe + Direction crosswalk                                    #
# --------------------------------------------------------------------------- #
def load_value_root(arch: str) -> pd.DataFrame:
    oof = load_td5_oof(arch)
    lab = R.read_train_labels(
        columns=["symbol", "decision_bar", "side", "horizon",
                 "label_available_time", "win"])
    lab = lab[lab["horizon"] == HORIZON]
    base = oof.merge(
        lab[["symbol", "decision_bar", "side", "horizon",
             "label_available_time", "win"]],
        on=KEY, how="left", validate="one_to_one")
    st = R.read_state(columns=["symbol", "bar_index", "trading_day",
                              "candidate_at_decision"]).rename(
        columns={"bar_index": "decision_bar"})
    base = base.merge(st, on=["symbol", "decision_bar"], how="left",
                      validate="many_to_one")
    root = base[base["candidate_at_decision"] == True].reset_index(drop=True)
    return root


def build_direction_assets():
    split = build_frozen_split()
    ds = split["ds"]
    data = build_direction_expert_data(ds)
    cand = pd.to_datetime(ds["candidate_decision_time"]).to_numpy("datetime64[ns]")
    dir_index = pd.DataFrame({
        "symbol": ds["symbol"].to_numpy(object).astype(str),
        "cand_ns": cand.astype("int64"),
        "dir_row": np.arange(len(ds)),
    })
    return split, ds, data, dir_index


def map_epochs(value_root: pd.DataFrame, dir_index: pd.DataFrame) -> pd.DataFrame:
    v = value_root.copy()
    v["dt_ns"] = pd.to_datetime(v["decision_time"]).to_numpy("datetime64[ns]").astype("int64")
    ep = v.drop_duplicates(["symbol", "decision_bar"]).copy()
    ep = ep.merge(dir_index, left_on=["symbol", "dt_ns"],
                  right_on=["symbol", "cand_ns"], how="left")
    return ep


def verify_crosswalk(ep: pd.DataFrame, dir_index: pd.DataFrame) -> dict:
    n_value = len(ep)
    n_matched = int(ep["dir_row"].notna().sum())
    n_missing = int(ep["dir_row"].isna().sum())
    n_dup_value = int(ep.duplicated(["symbol", "decision_bar"]).sum())
    n_dup_dir = int(dir_index.duplicated(["symbol", "cand_ns"]).sum())
    if n_missing != 0 or n_dup_value != 0 or n_dup_dir != 0:
        raise RuntimeError("STOP_R13_8_DIRECTION_VALUE_CROSSWALK")
    return {
        "n_value_epochs": n_value,
        "n_direction_matches": n_matched,
        "n_missing": n_missing,
        "n_duplicate_value_keys": n_dup_value,
        "n_duplicate_direction_keys": n_dup_dir,
    }


def build_direction_axis(ds, data, dir_index, ep: pd.DataFrame,
                         train_idx=None) -> pd.DataFrame:
    """Run 5 causal Direction chains; each produces BOTH A9 and E9.

    The historical FIT/VAL pool is explicitly intersected with the frozen TRAIN
    index (reviewer repair section 5) so that no old VAL/TEST row can enter the
    Direction fit.
    """
    cand = pd.to_datetime(ds["candidate_decision_time"]).to_numpy("datetime64[ns]").astype("int64")
    oexit = pd.to_datetime(ds["oracle_exit_fill_time"]).to_numpy("datetime64[ns]").astype("int64")

    rows = []
    provenance = []
    for k in range(N_FOLDS):
        target = ep[ep["fold"] == k].copy()
        if len(target) == 0:
            continue
        target_rows = target["dir_row"].to_numpy(int)
        T_ns = int(pd.to_datetime(target["decision_time"]).min().value)

        fit_idx, val_idx = direction_history_pool(data, ds, T_ns, train_idx)
        n_preq = _prequential_oof_fit_count(data, ds, fit_idx)
        chain = run_axis_chain(data, ds, fit_idx, val_idx, target_rows)

        # chain outputs are positioned by test_idx order == target_rows order
        a9 = np.asarray(chain["A"]).astype(np.uint8)
        e9 = np.asarray(chain["E9"]).astype(np.uint8)
        # canonical assertion: A9 == frozen router
        np.testing.assert_array_equal(a9, np.asarray(chain["router_te"]).astype(np.uint8))

        rows.append(pd.DataFrame({
            "symbol": target["symbol"].to_numpy(object),
            "decision_bar": target["decision_bar"].to_numpy(),
            "decision_time": target["decision_time"].to_numpy(),
            "trading_day": target["trading_day"].to_numpy(),
            "fold": k,
            "dir_row": target_rows,
            "a9_side": np.where(a9 == 1, 1, -1),
            "e9_side": np.where(e9 == 1, 1, -1),
            "a9_direction": a9.astype(np.int8),
            "e9_direction": e9.astype(np.int8),
            "a9_p_long": np.asarray(chain["A_p"]),
            "e9_p_correct": np.asarray(chain["e9_p_correct"]),
            "router_direction": np.asarray(chain["router_te"]).astype(np.int8),
            "router_p_long": np.asarray(chain["router_p_long"]),
        }))
        provenance.append({
            "fold": k,
            "cutoff": str(pd.to_datetime(T_ns)),
            "n_direction_fit": int(fit_idx.size),
            "n_direction_val": int(val_idx.size),
            "n_target": int(target_rows.size),
            "max_fit_decision_time": str(pd.to_datetime(np.max(cand[fit_idx]))),
            "max_fit_oracle_exit_fill_time": str(pd.to_datetime(np.max(oexit[fit_idx]))),
            "max_val_decision_time": str(pd.to_datetime(np.max(cand[val_idx]))),
            "max_val_oracle_exit_fill_time": str(pd.to_datetime(np.max(oexit[val_idx]))),
            "n_estimators": chain["n_estimators"],
            "n_prequential_oof_router_fits": n_preq,
            # 1 (m0) + 1 (fixed router) + N (prequential OOF router) + 2 (E9 LONG/SHORT experts)
            "n_underlying_model_fits": 4 + n_preq,
        })
        # hard assert: max label availability < cutoff
        assert np.max(cand[fit_idx]) < T_ns and np.max(oexit[fit_idx]) < T_ns
        assert np.max(cand[val_idx]) < T_ns and np.max(oexit[val_idx]) < T_ns

    axis = pd.concat(rows, ignore_index=True)
    prov = pd.DataFrame(provenance)
    return axis, prov


def split_85_15(data, available_idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dec = data.decision_time_ns[available_idx]
    day = dec // 86_400_000_000_000  # integer day index (ns -> days)
    uniq = np.sort(pd.unique(day))
    cut = int(len(uniq) * 0.85)
    fit_days = set(uniq[:cut].tolist())
    is_fit = np.array([d in fit_days for d in day])
    return available_idx[is_fit], available_idx[~is_fit]


def direction_history_pool(data, ds, T_ns: int, train_idx=None) -> Tuple[np.ndarray, np.ndarray]:
    """Build the causal Direction FIT/VAL history pool for one target fold.

    The pool is the set of bars with candidate_decision_time < T and
    oracle_exit_fill_time < T, optionally intersected with the frozen TRAIN index
    (reviewer repair section 5). This is the single source of truth used by both
    build_direction_axis and the audit tests.
    """
    cand = pd.to_datetime(ds["candidate_decision_time"]).to_numpy("datetime64[ns]").astype("int64")
    oexit = pd.to_datetime(ds["oracle_exit_fill_time"]).to_numpy("datetime64[ns]").astype("int64")
    hist_pool = np.where((cand < T_ns) & (oexit < T_ns))[0]
    if train_idx is not None:
        hist_pool = np.intersect1d(np.asarray(train_idx), hist_pool)
    return split_85_15(data, hist_pool)


def _prequential_oof_fit_count(data, ds, train_idx) -> int:
    """Count the number of prequential OOF router fits (reviewer repair section 6).

    Mirrors the canonical OOF quarter loop in direction_gated_experts_v1 (every
    quarter after the first warm-up quarter with a non-empty, non-straddling
    prediction block triggers exactly one fixed-router fit). This is a PURE count
    over the canonical block construction -- no model is fitted.
    """
    dec = data.decision_time_ns
    oex = data.oexit_time_ns
    gid = data.gid
    t = pd.to_datetime(dec).astype("datetime64[ns]")
    pidx = pd.PeriodIndex(t, freq="Q")
    qord = (pidx.year * 4 + (pidx.quarter - 1)).to_numpy()
    train_idx = np.asarray(train_idx)
    trq = qord[train_idx]
    tq = np.unique(trq)
    dfq = pd.DataFrame({"gid": gid[train_idx], "q": trq})
    nq = dfq.groupby("gid", sort=False)["q"].nunique()
    straddling = nq[nq > 1].index.to_numpy()
    straddle_mask = (np.isin(gid, straddling) if straddling.size
                     else np.zeros(len(gid), dtype=bool))
    n_fits = 0
    for q in tq[1:]:
        block_all = train_idx[trq == q]
        pred_idx = block_all[~straddle_mask[block_all]]
        if pred_idx.size == 0:
            continue
        n_fits += 1
    return int(n_fits)


# --------------------------------------------------------------------------- #
# Direction-conditioned Value population                                       #
# --------------------------------------------------------------------------- #
def direction_conditioned_value(value_root: pd.DataFrame, axis_df: pd.DataFrame,
                               dir_system: str) -> pd.DataFrame:
    side_col = "a9_side" if dir_system == "A9" else "e9_side"
    sel = axis_df[["symbol", "decision_bar", side_col]].copy()
    sel["chosen_side"] = np.where(sel[side_col].to_numpy() > 0, "LONG", "SHORT")
    m = value_root.merge(sel, on=["symbol", "decision_bar"], how="inner",
                         validate="many_to_one")
    chosen = m[m["side"] == m["chosen_side"]].copy()
    if chosen.duplicated(["symbol", "decision_bar"]).any():
        raise RuntimeError("STOP_R13_8_MULTIPLE_CHOSEN_SIDE")
    chosen["epoch_weight"] = 1.0
    return chosen


def gate_scores(frame: pd.DataFrame):
    """Return per-row score for each gate (history thresholds computed by caller)."""
    p = np.clip(frame["p_win"].to_numpy(float), 0.0, 1.0)
    mw = np.maximum(frame["mu_win"].to_numpy(float), 0.0)
    ml = np.maximum(frame["mu_loss"].to_numpy(float), 0.0)
    evc = p * mw - (1.0 - p) * ml
    return p, evc


def pgm_crossfit(frame: pd.DataFrame):
    """Folds 1..4 second-level crossfit. Returns per-(D,V) arrays + regime probs."""
    parts = []
    regime_probs = None
    scores = {"P": [], "EV_C": [], "PGM3": []}
    selects = {"P": [], "EV_C": [], "PGM3": []}
    q80s = {"P": [], "EV_C": [], "PGM3": []}
    meta = []
    for k in range(1, N_FOLDS):
        target = frame[frame["fold"] == k].copy()
        T = pd.to_datetime(target["decision_time"]).min()
        hist = frame[frame["fold"] < k].copy()
        hist = hist[pd.to_datetime(hist["decision_time"]) < T]
        hist = hist[pd.to_datetime(hist["label_available_time"]) < T]
        yh = hist["episode_return_atr"].to_numpy(float)
        wh = hist["epoch_weight"].to_numpy(float)

        p_h, evc_h = gate_scores(hist)
        model = PGM3().fit(meta_state(hist), yh, wh)
        pgm_h = model.predict(meta_state(hist))

        p_t, evc_t = gate_scores(target)
        pgm_t = model.predict(meta_state(target))
        rp = model.regime_prob(meta_state(target))

        q80_p = float(np.quantile(p_h, 0.80))
        q80_e = float(np.quantile(evc_h, 0.80))
        q80_g = float(np.quantile(pgm_h, 0.80))

        out = target[["symbol", "decision_bar", "fold", "decision_time",
                      "trading_day", "epoch_weight", "episode_return_atr",
                      "side", "p_win", "mu_win", "mu_loss"]].copy()
        out["score_P"] = p_t
        out["score_EV_C"] = evc_t
        out["score_PGM3"] = pgm_t
        out["q80_P"] = q80_p
        out["q80_EV_C"] = q80_e
        out["q80_PGM3"] = q80_g
        out["select20_P"] = p_t >= q80_p
        out["select20_EV_C"] = evc_t >= q80_e
        out["select20_PGM3"] = pgm_t >= q80_g
        parts.append(out)

        if regime_probs is None:
            regime_probs = rp
        else:
            regime_probs = np.concatenate([regime_probs, rp], axis=0)
        meta.append({
            "eval_fold": k, "n_hist": int(len(hist)),
            "q80_P": q80_p, "q80_EV_C": q80_e, "q80_PGM3": q80_g,
            "pgm_components": int(model.gmm.n_components),
        })

    # Keep axis (epoch) order: do NOT sort, so regime_probs stays row-aligned
    # with block_idx built from the same axis order.
    res = pd.concat(parts, ignore_index=True)
    return res, regime_probs, pd.DataFrame(meta)


# --------------------------------------------------------------------------- #
# Block bootstrap                                                              #
# --------------------------------------------------------------------------- #
def block_index_map(trading_days: np.ndarray, block: int):
    days = np.sort(pd.unique(trading_days))
    n_complete = len(days) // block
    days = days[: n_complete * block]
    pos = {d: np.where(trading_days == d)[0] for d in days}
    blocks = [days[i * block:(i + 1) * block] for i in range(n_complete)]
    return [np.concatenate([pos[d] for d in b]) for b in blocks]


def _wmean(y, w):
    return float(np.average(y, weights=w)) if w.sum() > 0 else np.nan


def boot_cells(cells: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]],
               block_idx, B: int, seed: int) -> Dict[str, np.ndarray]:
    """cells: key -> (y, w, selected_bool). Returns per-cell R20 bootstrap dist."""
    rng = np.random.default_rng(seed)
    n_blocks = len(block_idx)
    out = {k: np.empty(B) for k in cells}
    for b in range(B):
        idx = np.concatenate([block_idx[rng.integers(0, n_blocks)]
                              for _ in range(n_blocks)])
        for k, (y, w, sel) in cells.items():
            m = sel[idx]
            yy = y[idx][m]
            ww = w[idx][m]
            out[k][b] = _wmean(yy, ww)
    return {k: v[np.isfinite(v)] for k, v in out.items()}


def boot_contrasts(cells, contrast_fns, block_idx, B, seed):
    rng = np.random.default_rng(seed)
    n_blocks = len(block_idx)
    out = {name: np.empty(B) for name in contrast_fns}
    for b in range(B):
        idx = np.concatenate([block_idx[rng.integers(0, n_blocks)]
                              for _ in range(n_blocks)])
        rr = {}
        for k, (y, w, sel) in cells.items():
            m = sel[idx]
            rr[k] = _wmean(y[idx][m], w[idx][m])
        for name, fn in contrast_fns.items():
            out[name][b] = fn(rr)
    return {name: v[np.isfinite(v)] for name, v in out.items()}


def boot_matched(score, y, w, fold, block_idx, k_per_fold, B, seed):
    rng = np.random.default_rng(seed)
    n_blocks = len(block_idx)
    out = np.empty(B)
    folds = np.unique(fold)
    for b in range(B):
        idx = np.concatenate([block_idx[rng.integers(0, n_blocks)]
                              for _ in range(n_blocks)])
        yy = y[idx]
        ww = w[idx]
        ss = score[idx]
        ff = fold[idx]
        tot = 0.0
        totw = 0.0
        for f in folds:
            fm = ff == f
            if not fm.any():
                continue
            k = k_per_fold.get(int(f), 0)
            if k <= 0:
                continue
            sfold = ss[fm]
            order = np.argsort(-sfold)[:k]
            tot += np.sum(ww[fm][order] * yy[fm][order])
            totw += np.sum(ww[fm][order])
        out[b] = tot / totw if totw > 0 else np.nan
    return out[np.isfinite(out)]


def _ci(v):
    if v.size == 0:
        return (np.nan, np.nan, np.nan)
    return (float(np.nanmean(v)), float(np.nanpercentile(v, 2.5)),
            float(np.nanpercentile(v, 97.5)))


# --------------------------------------------------------------------------- #
# Direction audit (PART C)                                                     #
# --------------------------------------------------------------------------- #
def direction_audit_table(value_root: pd.DataFrame, axis_df: pd.DataFrame):
    pv = value_root.pivot_table(index=["symbol", "decision_bar", "fold",
                                       "trading_day"],
                                columns="side",
                                values="episode_return_atr").reset_index()
    d = pv.merge(axis_df[["symbol", "decision_bar", "a9_side", "e9_side"]],
                 on=["symbol", "decision_bar"], validate="one_to_one")
    yl = d["LONG"].to_numpy(float)
    ys = d["SHORT"].to_numpy(float)
    a9_long = d["a9_side"].to_numpy() > 0
    e9_long = d["e9_side"].to_numpy() > 0
    d["y_a9"] = np.where(a9_long, yl, ys)
    d["y_e9"] = np.where(e9_long, yl, ys)
    d["y_a9_opp"] = np.where(a9_long, ys, yl)
    d["y_e9_opp"] = np.where(e9_long, ys, yl)
    d["a9_adv"] = d["y_a9"] - d["y_a9_opp"]
    d["e9_adv"] = d["y_e9"] - d["y_e9_opp"]
    d["e9_minus_a9"] = d["y_e9"] - d["y_a9"]
    d["agree"] = d["a9_side"] == d["e9_side"]
    d["avg_side"] = (yl + ys) / 2.0
    d["oracle_side"] = np.maximum(yl, ys)
    return d


def direction_summary(d: pd.DataFrame, block_idx):
    w = np.ones(len(d))
    cells = {
        "A9": (d["y_a9"].to_numpy(float), w, np.ones(len(d), bool)),
        "E9": (d["y_e9"].to_numpy(float), w, np.ones(len(d), bool)),
    }
    boots = boot_cells(cells, block_idx, BOOTSTRAP_B, BOOTSTRAP_SEED)
    a9_mean = _wmean(d["y_a9"].to_numpy(float), w)
    e9_mean = _wmean(d["y_e9"].to_numpy(float), w)
    a9_adv = _wmean(d["a9_adv"].to_numpy(float), w)
    e9_adv = _wmean(d["e9_adv"].to_numpy(float), w)
    delta = _wmean(d["e9_minus_a9"].to_numpy(float), w)
    delta_boot = boots["E9"] - boots["A9"]
    dlo, dhi = np.nanpercentile(delta_boot, 2.5), np.nanpercentile(delta_boot, 97.5)
    if dlo > 0:
        verdict = "E9_SUPPORTED_OVER_A9"
    elif dhi < 0:
        verdict = "A9_SUPPORTED_OVER_E9"
    else:
        verdict = "A9_E9_DIRECTION_UNRESOLVED"
    agree = float(d["agree"].mean())
    n_dis = int((~d["agree"]).sum())
    dis = d[~d["agree"]]
    e9_win = int((dis["y_e9"] > dis["y_a9"]).sum())
    a9_win = int((dis["y_a9"] > dis["y_e9"]).sum())
    tie = int((dis["y_e9"] == dis["y_a9"]).sum())
    return {
        "A9_selected_mean_return": round(a9_mean, 6),
        "A9_opposite_mean_return": round(a9_mean - a9_adv, 6),
        "A9_direction_advantage": round(a9_adv, 6),
        "E9_selected_mean_return": round(e9_mean, 6),
        "E9_opposite_mean_return": round(e9_mean - e9_adv, 6),
        "E9_direction_advantage": round(e9_adv, 6),
        "E9_minus_A9_point": round(delta, 6),
        "E9_minus_A9_ci_lo": round(float(dlo), 6),
        "E9_minus_A9_ci_hi": round(float(dhi), 6),
        "direction_verdict": verdict,
        "agreement_rate": round(agree, 6),
        "disagreement_count": n_dis,
        "E9_wins_disagreement": e9_win,
        "A9_wins_disagreement": a9_win,
        "ties_disagreement": tie,
    }


# --------------------------------------------------------------------------- #
# Reviewer repair (NEED_REVISION): frozen-artifact evidence repair helpers       #
# --------------------------------------------------------------------------- #
def pair_universe_audit(value_root: pd.DataFrame) -> dict:
    """Prove every (symbol,decision_bar,fold) has both LONG & SHORT with pair
    sample_weight summing to 1.0 (reviewer repair section 2)."""
    grp = value_root.groupby(["symbol", "decision_bar", "fold"], sort=False)
    n_rows = len(value_root)
    n_epochs = value_root.drop_duplicates(["symbol", "decision_bar", "fold"]).shape[0]
    incomplete = 0
    max_err = 0.0
    for _, g in grp:
        if set(g["side"].tolist()) != {"LONG", "SHORT"}:
            incomplete += 1
        sw = g["sample_weight"].to_numpy(float)
        err = abs(float(sw.sum()) - 1.0)
        if err > max_err:
            max_err = err
    if incomplete > 0:
        raise RuntimeError("STOP_R13_8_INCOMPLETE_TWO_SIDE_EPOCH")
    if max_err > 1e-12:
        raise RuntimeError("STOP_R13_8_PAIR_WEIGHT_DRIFT")
    return {"rows": int(n_rows), "epochs": int(n_epochs),
            "incomplete_epochs": int(incomplete),
            "max_pair_weight_error": float(max_err)}


def epoch_universe_identity(vr_a0: pd.DataFrame, vr_a1: pd.DataFrame) -> dict:
    """Require A0 and A1 epoch universe (symbol,decision_bar,fold,decision_time,
    trading_day) to be identical (reviewer repair section 3)."""
    keys = ["symbol", "decision_bar", "fold", "decision_time", "trading_day"]
    s0 = set(map(tuple, vr_a0[keys].drop_duplicates().to_numpy()))
    s1 = set(map(tuple, vr_a1[keys].drop_duplicates().to_numpy()))
    if s0 != s1:
        raise RuntimeError("STOP_R13_8_A0_A1_EPOCH_UNIVERSE_DRIFT")
    return {"a0_epochs": len(s0), "a1_epochs": len(s1), "identical": True}


def router_probability_gate(axis: pd.DataFrame) -> pd.DataFrame:
    """Per-fold canonical gate: max |router_p_long - a9_p_long| <= 1e-4
    (reviewer repair section 4). No prediction regeneration."""
    rows = []
    for k in range(N_FOLDS):
        a = axis[axis["fold"] == k]
        dev = float(np.max(np.abs(
            a["router_p_long"].to_numpy(float) - a["a9_p_long"].to_numpy(float))))
        if dev > 1e-4:
            raise RuntimeError("STOP_R13_8_FIXED_ROUTER_A_PROBABILITY_MISMATCH")
        rows.append({"fold": int(k), "max_abs_a9_minus_router_p": dev})
    return pd.DataFrame(rows)


def corrected_fit_counts(axis: pd.DataFrame, ds, data) -> pd.DataFrame:
    """Recompute Direction fit counts from the FROZEN axis (reviewer repair section 6)."""
    cand = pd.to_datetime(ds["candidate_decision_time"]).to_numpy("datetime64[ns]").astype("int64")
    oexit = pd.to_datetime(ds["oracle_exit_fill_time"]).to_numpy("datetime64[ns]").astype("int64")
    rows = []
    for k in range(N_FOLDS):
        target = axis[axis["fold"] == k]
        if len(target) == 0:
            continue
        T_ns = int(pd.to_datetime(target["decision_time"]).min().value)
        avail = np.where((cand < T_ns) & (oexit < T_ns))[0]
        fit_idx, val_idx = split_85_15(data, avail)
        n_preq = _prequential_oof_fit_count(data, ds, fit_idx)
        rows.append({
            "fold": int(k),
            "cutoff": str(pd.to_datetime(T_ns)),
            "n_direction_fit": int(fit_idx.size),
            "n_direction_val": int(val_idx.size),
            "n_target": int(target.shape[0]),
            "max_fit_decision_time": str(pd.to_datetime(np.max(cand[fit_idx]))),
            "max_fit_oracle_exit_fill_time": str(pd.to_datetime(np.max(oexit[fit_idx]))),
            "max_val_decision_time": str(pd.to_datetime(np.max(cand[val_idx]))),
            "max_val_oracle_exit_fill_time": str(pd.to_datetime(np.max(oexit[val_idx]))),
            "n_prequential_oof_router_fits": int(n_preq),
            "n_underlying_model_fits": 4 + int(n_preq),
        })
        assert np.max(cand[fit_idx]) < T_ns and np.max(oexit[fit_idx]) < T_ns
        assert np.max(cand[val_idx]) < T_ns and np.max(oexit[val_idx]) < T_ns
    return pd.DataFrame(rows)


def boot_matched_pair(score_base, score_chall, y, w, fold, block_idx, k_per_fold, B, seed):
    """Paired bootstrap: top-k by base-score return vs top-k by challenger-score
    return, using identical block draws (reviewer repair section 7)."""
    rng = np.random.default_rng(seed)
    n_blocks = len(block_idx)
    folds = np.unique(fold)
    p_ret = np.empty(B)
    g_ret = np.empty(B)
    for b in range(B):
        idx = np.concatenate([block_idx[rng.integers(0, n_blocks)]
                              for _ in range(n_blocks)])
        yy = y[idx]; ww = w[idx]
        sb = score_base[idx]; sc = score_chall[idx]; ff = fold[idx]
        tp = 0.0; twp = 0.0; tg = 0.0; twg = 0.0
        for f in folds:
            fm = ff == f
            if not fm.any():
                continue
            k = k_per_fold.get(int(f), 0)
            if k <= 0:
                continue
            ob = np.argsort(-sb[fm])[:k]
            oc = np.argsort(-sc[fm])[:k]
            tp += float(np.sum(ww[fm][ob] * yy[fm][ob])); twp += float(ww[fm][ob].sum())
            tg += float(np.sum(ww[fm][oc] * yy[fm][oc])); twg += float(ww[fm][oc].sum())
        p_ret[b] = tp / twp if twp > 0 else np.nan
        g_ret[b] = tg / twg if twg > 0 else np.nan
    return p_ret[np.isfinite(p_ret)], g_ret[np.isfinite(g_ret)]


# --------------------------------------------------------------------------- #
# Main orchestration (reviewer repair: reuse frozen prediction artifacts)         #
# --------------------------------------------------------------------------- #
def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    os.makedirs(EVIDENCE_DIR, exist_ok=True)

    # ---- Phase 0: canonical Direction spec print ----
    print("A9_CANONICAL = system A = frozen direct DTP9 router (chain['A'])")
    print("A9_PROBABILITY = chain['A_p'] (NOT p_te)")
    print("E9_CANONICAL = gated META10 LONG/SHORT correctness experts")
    print("DIRECTION_LABEL_HORIZON = oracle_exit_fill_time")

    # ---- Phase 1: crosswalk (no model fit) ----
    split, ds, data, dir_index = build_direction_assets()
    train_idx = split["train_idx"]
    value_roots = {a: load_value_root(arch) for a, arch in ARCHS.items()}
    epochs = {a: map_epochs(vr, dir_index) for a, vr in value_roots.items()}
    xwalk = verify_crosswalk(epochs["A0"], dir_index)

    # ---- Phase 2: verify frozen prediction artifacts (reviewer section 1) ----
    with open(MANIFEST_JSON) as f:
        manifest_prev = json.load(f)
    axis_sha = sha256_of(DIRECTION_AXIS_PARQUET)
    fact_sha = sha256_of(FACTORIAL_OOF_PARQUET)
    if (axis_sha != manifest_prev.get("direction_axis_sha256") or
            fact_sha != manifest_prev.get("factorial_oof_sha256")):
        raise RuntimeError("STOP_R13_8_REPAIR_FROZEN_ARTIFACT_UNAVAILABLE")
    axis = pd.read_parquet(DIRECTION_AXIS_PARQUET)
    fact_ledger = pd.read_parquet(FACTORIAL_OOF_PARQUET)
    print(f"[R13.8] frozen axis sha={axis_sha}")
    print(f"[R13.8] frozen factorial oof sha={fact_sha}")

    # ---- Phase 3: Direction audit (frozen axis, all 5 folds) ----
    d_audit = direction_audit_table(value_roots["A0"], axis)
    block_idx_all = block_index_map(d_audit["trading_day"].to_numpy(), BOOTSTRAP_BLOCK)
    dir_summary = direction_summary(d_audit, block_idx_all)
    R.write_json_evidence({**xwalk, **dir_summary}, DIR_SUMMARY_JSON)

    fold_rows = []
    for k in range(N_FOLDS):
        dk = d_audit[d_audit["fold"] == k]
        w = np.ones(len(dk))
        fold_rows.append({
            "eval_fold": k,
            "A9_mean_return": round(_wmean(dk["y_a9"], w), 6),
            "E9_mean_return": round(_wmean(dk["y_e9"], w), 6),
            "E9_minus_A9": round(_wmean(dk["e9_minus_a9"], w), 6),
            "agreement_rate": round(float(dk["agree"].mean()), 6),
        })
    R.write_csv_evidence(pd.DataFrame(fold_rows), DIR_FOLD_CSV)
    dis = d_audit[~d_audit["agree"]]
    R.write_csv_evidence(dis[["symbol", "decision_bar", "fold", "y_a9", "y_e9",
                              "y_a9_opp", "y_e9_opp", "e9_minus_a9"]]
                         .rename(columns={"y_a9": "y_A9", "y_e9": "y_E9"}),
                         DIR_DISAGREE_CSV)

    # ---- Phase 4: pair + universe audits (reviewer sections 2, 3) ----
    pair_audit = {a: pair_universe_audit(value_roots[a]) for a in ARCHS}
    R.write_json_evidence({"A0": pair_audit["A0"], "A1": pair_audit["A1"]},
                          PAIR_AUDIT_JSON)
    universe = epoch_universe_identity(value_roots["A0"], value_roots["A1"])
    sel_counts = {}
    for D in DIR_SYSTEMS:
        for Va in ARCHS:
            chosen = direction_conditioned_value(value_roots[Va], axis, D)
            n = len(chosen)
            if n != 21341:
                raise RuntimeError(f"STOP_R13_8_SELECTION_EPOCH_COUNT:{D}_{Va}_{n}")
            if chosen.duplicated(["symbol", "decision_bar"]).any():
                raise RuntimeError("STOP_R13_8_MULTIPLE_CHOSEN_SIDE")
            sel_counts[f"{D}_{Va}"] = int(n)

    # ---- Phase 5: router probability gate (reviewer section 4) ----
    router_gate = router_probability_gate(axis)
    R.write_csv_evidence(router_gate, ROUTER_GATE_CSV)

    # ---- Phase 6: corrected Direction fit counts (reviewer section 6) ----
    fit_counts = corrected_fit_counts(axis, ds, data)
    R.write_csv_evidence(fit_counts, os.path.join(
        EVIDENCE_DIR, "r13_8_direction_provenance_v2.csv"))

    # ---- Phase 7: reconstruct analysis frames from FROZEN fact_ledger ----
    # No PGM re-fit, no prediction regeneration: scores/trade20 come from the
    # frozen artifact; outcomes come from frozen TRAIN labels via value roots.
    cond = {}
    for Va in ARCHS:
        for D in DIR_SYSTEMS:
            led = fact_ledger[(fact_ledger["direction"] == D) &
                              (fact_ledger["value_arch"] == Va)].copy()
            chosen = direction_conditioned_value(value_roots[Va], axis, D)
            # Only pull outcome from chosen; led already carries epoch_weight /
            # trading_day / fold / score / trade20 (avoids pandas _x/_y suffixing).
            ch = chosen[["symbol", "decision_bar", "fold", "episode_return_atr"]].copy()
            merged = led.merge(ch, on=["symbol", "decision_bar", "fold"], how="left")
            if merged["episode_return_atr"].isna().any():
                raise RuntimeError("STOP_R13_8_OUTCOME_JOIN_MISSING")
            cond[(D, Va)] = merged

    cells = {}
    block_idx_map = {}
    for (D, Va), merged in cond.items():
        c = {}
        for G in GATES:
            s = merged[merged["gate"] == G].reset_index(drop=True)
            c[G] = {
                "y": s["episode_return_atr"].to_numpy(float),
                "w": s["epoch_weight"].to_numpy(float),
                "sel": s["trade20"].to_numpy(bool),
                "score": s["score"].to_numpy(float),
                "fold": s["fold"].to_numpy(),
                "trading_day": s["trading_day"].to_numpy(),
            }
        bi = block_index_map(c["P"]["trading_day"], BOOTSTRAP_BLOCK)
        cells[(D, Va)] = (c, bi)

    # ---- Phase 8: cell metrics (fact_models / fact_fold) ----
    fact_models = []
    fact_fold = []
    for (D, Va), (c, bi) in cells.items():
        for G in GATES:
            y = c[G]["y"]; w = c[G]["w"]; sel = c[G]["sel"]
            sc = c[G]["score"]; fold = c[G]["fold"]
            k20 = {int(f): max(1, int(np.ceil(0.2 * (fold == f).sum())))
                   for f in np.unique(fold)}
            m20 = boot_matched(sc, y, w, fold, bi, k20, BOOTSTRAP_B, BOOTSTRAP_SEED)
            mean_ret = _wmean(y[sel], w[sel]) if w[sel].sum() > 0 else np.nan
            cov = float(w[sel].sum() / w.sum()) if w.sum() > 0 else np.nan
            ys = y[sel]; ws = w[sel]; wins = ys > 0
            win_rate = float(wins.mean()) if wins.size > 0 else np.nan
            aw = _wmean(ys[wins], ws[wins]) if wins.any() else np.nan
            al = _wmean(ys[~wins], ws[~wins]) if (~wins).any() else np.nan
            pr = (aw / abs(al)) if (al and not np.isnan(al) and al != 0) else np.nan
            fact_models.append({
                "direction": D, "value_arch": Va, "gate": G,
                "causal_q80_coverage": round(cov, 6),
                "selected_mean_return": round(mean_ret, 6),
                "win_rate": round(win_rate, 6),
                "avg_win": round(aw, 6) if not np.isnan(aw) else np.nan,
                "avg_loss": round(al, 6) if not np.isnan(al) else np.nan,
                "payoff_ratio": round(pr, 6) if not np.isnan(pr) else np.nan,
                "matched20_return": round(float(np.nanmean(m20)), 6),
            })
            for kf in range(1, N_FOLDS):
                fm = fold == kf
                fsel = fm & sel
                fr = _wmean(y[fsel], w[fsel]) if w[fsel].sum() > 0 else np.nan
                fcov = float(w[fsel].sum() / w[fm].sum()) if w[fm].sum() > 0 else np.nan
                fact_fold.append({
                    "direction": D, "value_arch": Va, "gate": G,
                    "eval_fold": kf,
                    "TRADE20_coverage": round(fcov, 6),
                    "TRADE20_mean_return": round(fr, 6),
                })

    # ---- Phase 9: matched-N (P baseline vs PGM3 challenger), reviewer section 7 ----
    matched_rows = []
    for (D, Va), (c, bi) in cells.items():
        yP, wP, selP, scP = c["P"]["y"], c["P"]["w"], c["P"]["sel"], c["P"]["score"]
        yG, wG, selG, scG = c["PGM3"]["y"], c["PGM3"]["w"], c["PGM3"]["sel"], c["PGM3"]["score"]
        fold = c["P"]["fold"]
        folds = np.unique(fold)
        kP = {int(f): int(((fold == f) & selP).sum()) for f in folds}
        k20d = {int(f): max(1, int(np.ceil(0.2 * (fold == f).sum()))) for f in folds}
        P_ret20 = _wmean(yP[selP], wP[selP])
        G_ret20 = _wmean(yG[selG], wG[selG])
        p20, g20 = boot_matched_pair(scP, scG, yP, wP, fold, bi, k20d,
                                     BOOTSTRAP_B, BOOTSTRAP_SEED)
        pN, gN = boot_matched_pair(scP, scG, yP, wP, fold, bi, kP,
                                   BOOTSTRAP_B, BOOTSTRAP_SEED)
        lo20, hi20 = (np.nanpercentile(g20 - p20, 2.5),
                      np.nanpercentile(g20 - p20, 97.5))
        loN, hiN = (np.nanpercentile(gN - pN, 2.5),
                    np.nanpercentile(gN - pN, 97.5))
        nstr = ",".join(f"{int(f)}={kP[int(f)]}" for f in folds)
        G_retN = _wmean(yG[selG], wG[selG])
        matched_rows.append({
            "direction": D, "value_arch": Va, "match_kind": "matched20",
            "P_return": round(P_ret20, 6), "PGM_return": round(G_ret20, 6),
            "PGM_minus_P_point": round(G_ret20 - P_ret20, 6),
            "ci_lo": round(lo20, 6), "ci_hi": round(hi20, 6), "n_per_fold": nstr})
        matched_rows.append({
            "direction": D, "value_arch": Va, "match_kind": "matchedN",
            "P_return": round(P_ret20, 6), "PGM_return": round(G_retN, 6),
            "PGM_minus_P_point": round(G_retN - P_ret20, 6),
            "ci_lo": round(loN, 6), "ci_hi": round(hiN, 6), "n_per_fold": nstr})

    # ---- Phase 10: contrasts A0 + A1 (reviewer sections 8, 9) ----
    flat = {}
    obs = {}
    for (D, Va), (c, bi) in cells.items():
        for G in ("P", "PGM3"):
            flat[f"{D}_{Va}_{G}"] = (c[G]["y"], c[G]["w"], c[G]["sel"])
            obs[f"{D}_{Va}_{G}"] = _wmean(c[G]["y"][c[G]["sel"]], c[G]["w"][c[G]["sel"]])

    contrast_defs = []
    for Va in ("A0", "A1"):
        contrast_defs += [
            (f"PGM_minus_P_given_A9", Va,
             lambda rr, Va=Va: rr[f"A9_{Va}_PGM3"] - rr[f"A9_{Va}_P"]),
            (f"PGM_minus_P_given_E9", Va,
             lambda rr, Va=Va: rr[f"E9_{Va}_PGM3"] - rr[f"E9_{Va}_P"]),
            (f"E9_minus_A9_given_P", Va,
             lambda rr, Va=Va: rr[f"E9_{Va}_P"] - rr[f"A9_{Va}_P"]),
            (f"E9_minus_A9_given_PGM", Va,
             lambda rr, Va=Va: rr[f"E9_{Va}_PGM3"] - rr[f"A9_{Va}_PGM3"]),
        ]
    for Va in ("A0", "A1"):
        contrast_defs.append((
            "direction_gate_interaction", Va,
            lambda rr, Va=Va: (rr[f"E9_{Va}_PGM3"] - rr[f"E9_{Va}_P"])
            - (rr[f"A9_{Va}_PGM3"] - rr[f"A9_{Va}_P"])))

    def _obs_point(name, Va):
        if name == "PGM_minus_P_given_A9":
            return obs[f"A9_{Va}_PGM3"] - obs[f"A9_{Va}_P"]
        if name == "PGM_minus_P_given_E9":
            return obs[f"E9_{Va}_PGM3"] - obs[f"E9_{Va}_P"]
        if name == "E9_minus_A9_given_P":
            return obs[f"E9_{Va}_P"] - obs[f"A9_{Va}_P"]
        if name == "E9_minus_A9_given_PGM":
            return obs[f"E9_{Va}_PGM3"] - obs[f"A9_{Va}_PGM3"]
        if name == "direction_gate_interaction":
            return (obs[f"E9_{Va}_PGM3"] - obs[f"E9_{Va}_P"]) - \
                   (obs[f"A9_{Va}_PGM3"] - obs[f"A9_{Va}_P"])
        raise KeyError(name)

    contrast_rows = []
    for Va in ("A0", "A1"):
        bi = cells[("A9", Va)][1]
        flat_Va = {k: v for k, v in flat.items() if f"_{Va}_" in k}
        Va_fns = {name: fn for (name, v, fn) in contrast_defs if v == Va}
        cb = boot_contrasts(flat_Va, Va_fns, bi, BOOTSTRAP_B, BOOTSTRAP_SEED)
        for (name, v, fn) in contrast_defs:
            if v != Va:
                continue
            dist = cb[name]
            lo = float(np.nanpercentile(dist, 2.5))
            hi = float(np.nanpercentile(dist, 97.5))
            pt = _obs_point(name, Va)
            contrast_rows.append({
                "contrast": name, "value_arch": Va,
                "point": round(pt, 6), "ci_lo": round(lo, 6), "ci_hi": round(hi, 6),
            })

    # ---- Phase 11: write evidence (frozen parquet artifacts left untouched) ----
    R.write_csv_evidence(pd.DataFrame(fact_models), FACT_MODELS_CSV)
    R.write_csv_evidence(pd.DataFrame(fact_fold), FACT_FOLD_CSV)
    R.write_csv_evidence(pd.DataFrame(matched_rows), MATCHEDN_CSV)
    R.write_csv_evidence(pd.DataFrame(contrast_rows), FACT_CONTRAST_CSV)
    # pgm_regimes_v2 + factorial ledger are reused from the frozen artifact set;
    # they are NOT regenerated (no PGM re-fit, no prediction regeneration).

    # ---- manifest ----
    canon_mod = os.path.join(HERE, "direction_gated_experts_v1.py")
    a0_shards = [sha256_of(R._unit_paths("A0_V1_DISJOINT", f, HORIZON)[0])
                 for f in range(N_FOLDS)]
    a1_shards = [sha256_of(R._unit_paths("A1_SHARE_TO_WIN", f, HORIZON)[0])
                 for f in range(N_FOLDS)]
    fit_total = int(fit_counts["n_underlying_model_fits"].sum())
    lineage = {
        "r13_8_code_sha": os.environ.get("R13_8_CODE_SHA", "PENDING_COMMIT"),
        "canonical_direction_module_sha256": sha256_of(canon_mod),
        "a0_oof_shard_sha256": a0_shards,
        "a1_oof_shard_sha256": a1_shards,
        "train_labels_sha256": sha256_of(R.ALLOWED_V1_LABELS_TRAIN),
        "state_sha256": sha256_of(R.ALLOWED_V1_STATE),
        "direction_dataset_rows": int(len(ds)),
        "original_direction_model_byte_sha_available": False,
    }
    manifest = {
        "experiment": "FUTURE-R13.8-V2",
        "reviewed_parent_sha": REVIEWED_PARENT_SHA,
        "r13_7_code_sha": R13_7_CODE_SHA,
        "r13_7_evidence_sha": R13_7_EVIDENCE_SHA,
        "evidence_repair": "NEED_REVISION__EVIDENCE_REPAIR_ONLY",
        "direction_axis_sha256": axis_sha,
        "factorial_oof_sha256": fact_sha,
        "lineage": lineage,
        "crosswalk": xwalk,
        "governance": {
            "dev_val_reads": 0, "old_test_label_reads": 0,
            "closed_test_direction_axis_reads": 0,
            "direction_chain_runs_reused": N_FOLDS,
            "direction_model_refits": 0,
            "prediction_refits": 0,
            "E33_fits": 0, "M33_fits": 0, "M9_fits": 0,
            "base_value_model_fits": 0, "pgm_fits": 16,
            "threshold_mining": False, "symbol_filtering": False,
            "regime_filtering": False,
            "direction_underlying_model_fits_total": fit_total,
            "direction_underlying_fits_per_fold":
                fit_counts[["fold", "n_underlying_model_fits"]].to_dict("records"),
        },
        "direction_audit": dir_summary,
        "pair_universe_audit": {"A0": pair_audit["A0"], "A1": pair_audit["A1"]},
        "epoch_universe_identity": universe,
        "direction_selection_epoch_counts": sel_counts,
        "router_probability_gate": router_gate.to_dict("records"),
        "contrasts_A0": {r["contrast"]: r for r in contrast_rows if r["value_arch"] == "A0"},
        "contrasts_A1": {r["contrast"]: r for r in contrast_rows if r["value_arch"] == "A1"},
        "stop": "AWAITING_R13_8_V2_REPAIR_REVIEW",
    }
    R.write_json_evidence(manifest, MANIFEST_JSON)
    print(f"[R13.8] evidence REPAIR written. direction verdict = {dir_summary['direction_verdict']}")
    print(f"[R13.8] STOP: AWAITING_R13_8_V2_REPAIR_REVIEW")


if __name__ == "__main__":
    main()
