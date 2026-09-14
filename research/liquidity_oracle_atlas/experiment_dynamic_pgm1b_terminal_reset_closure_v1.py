"""DYNAMIC-PGM-1B -- Terminal + Reset Closure.

Goal
----
Close the episode boundary that 1A deliberately left out:

    hazard H_{t+1}  ->  terminal endpoint E_{t+1}
                     ->  inter-episode gap G_n
                     ->  next-episode primitive reset state R_{n+1}

    P(H,E,G,R | S_t)
      = P(H | Phi_t) * [ P(E | H=1, Phi_t) P(G | E, Phi_t) P(R | G,E, Phi_t) ]^H

1B does NOT refit 1A. The within-episode kernel is frozen; this file only adds
the boundary kernel. 1C is what will stitch them into one rollout.

Conditioning representation
---------------------------
    Phi(S_t) = [ S_t , R_det(S_t) ]

where R_det is the 10 frozen current-state nonlinear re-encodings validated in
1A.2c, plus the 6 true-history innovations Z_t^{mem} as a CHALLENGER only.

Terminal models (hazard Logistic + frozen 15-mask pairwise CRF, reused verbatim
from PGM-BAR via the MARKET-STATE-1.1 module):

    TP = S_t                     frozen PTc_COMPACT_PROV_TEMPO  (parity only)
    T0 = S_t + A_t
    T1 = T0 + Phi_reencode(10)   -> TERMINAL_PHI
    T2 = T1 + Z_t^{mem}(6)       -> memory challenger

Reset models (gap occurrence Logistic + shared Geometric magnitude + geometry /
logratio_residual / start-shape / provenance-age / provenance-count heads):

    R0 = E + A
    R1 = R0 + S_t + Phi(10)      -> RESET_STATE
    R2 = R1 + Z_t^{mem}(6)       -> memory challenger

Boundary joint (episodes that own a legal reset pair only):

    B_i = terminal_episode_NLL(T_i) + reset_NLL(R_i)

Frozen scope
------------
No lag2/lag3, no latent/HMM/HSMM, no alpha/C tuning, no SMC recomputation,
no episode rebuild, no rollout, no PnL/RL, no TB4.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2_lag_closure_v1 as lag  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2c_representation_control_v1 as rep  # noqa: E402
import research.liquidity_oracle_atlas.experiment_market_state1_1_state_closure_v1 as ms  # noqa: E402
import research.liquidity_oracle_atlas.experiment_pgm_bar0_path_smc_v1 as pbar  # noqa: E402
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as pm  # noqa: E402

# ===========================================================================
# frozen references
# ===========================================================================
BASE_SHA = "7e86c8ee72d3117b368dd1639e0c73de49534fc5"
EXPECTED_ROWS = 359714
EXPECTED_EPISODES = 37987
EXPECTED_TERMINALS = 37987
FROZEN_EPISODE_HASH = pm.REVIEWER_FROZEN_EPISODE_HASH
FROZEN_TERMINAL = {
    "A_TB1_to_TB2": dict(hazard_nll=0.32684498811981755,
                         endpoint_joint_nll=1.1485103775491436,
                         mean_episode_nll=4.2803285435813),
    "B_TB1TB2_to_TB3": dict(hazard_nll=0.33178405258140786,
                            endpoint_joint_nll=1.1698024167403347,
                            mean_episode_nll=4.299368968756157),
}
PARITY_TOL = 1e-8
SYMBOL_BREADTH_MIN = 10
BOOT_REPS = 1000

WINDOWS = base.WINDOWS
CAT = list(base.OBS_STATE_CAT)
OBS_NUM = list(base.OBS_STATE_NUM)
LAG_AVAIL = lag.LAG_AVAIL
# phi_/mem_ re-use the 1A.2c frozen classification, under the 1B naming.
_PHI_BASE = [zc for n in rep.S_T_SOURCE for zc in base.NODE_ZCOLS[n]]
_MEM_BASE = ([zc for n in ("z_dmfe", "z_dmae") for zc in base.NODE_ZCOLS[n]]
             + list(base.COUNT_Z))
PHI_COLS = [f"phi_{c}" for c in _PHI_BASE]
MEM_COLS = [f"mem_{c}" for c in _MEM_BASE]
# the underlying layout must equal 1A.2c's frozen CUR / MEM partition
assert [c[3:] for c in rep.ZT_CUR_COLS] == _PHI_BASE
assert [c[3:] for c in rep.ZT_MEM_COLS] == _MEM_BASE
assert len(PHI_COLS) == 10 and len(MEM_COLS) == 6

# terminal model feature sets
TP_NUM = list(ms.MODELS["PTc_COMPACT_PROV_TEMPO"][0])
T0_NUM = OBS_NUM + [LAG_AVAIL]
T1_NUM = T0_NUM + PHI_COLS
T2_NUM = T1_NUM + MEM_COLS

# reset primitive targets (11)
RESET_GEOM = ["next_start_up_distance_R", "next_start_down_distance_R"]
RESET_LOGRATIO_RESID = ["next_start_log_ratio_residual"]
RESET_SHAPE = ["next_path_max_up_excursion_R", "next_path_max_down_excursion_R"]
RESET_AGE = ["next_upper_newest_log_age", "next_upper_span",
             "next_lower_newest_log_age", "next_lower_span"]
RESET_CNT = ["next_upper_n_active_minus1", "next_lower_n_active_minus1"]
RESET_TARGETS = (RESET_GEOM + RESET_LOGRATIO_RESID + RESET_SHAPE
                 + RESET_AGE + RESET_CNT)
assert (len(RESET_GEOM) + len(RESET_LOGRATIO_RESID) + len(RESET_SHAPE)
        + len(RESET_AGE) + len(RESET_CNT) == 11)

# reset model feature sets. E_n is a 15-class unordered mask, so it enters ONCE
# as a categorical (MASK_CAT) and never as a second numeric column.
MASK_CAT = ["prev_endpoint_mask"]
R0_OCC = [LAG_AVAIL]
R1_OCC = R0_OCC + OBS_NUM + PHI_COLS
R2_OCC = R1_OCC + MEM_COLS
GAP_FEAT = ["gap_positive", "log1p_gap"]

C_TERM_PHI = "T1-T0"
C_TERM_MEM = "T2-T1"
C_TERM_AVAIL = "T0-TP"
C_RESET_STATE = "R1-R0"
C_RESET_MEM = "R2-R1"
C_BOUND_PHI = "B1-B0"
C_BOUND_MEM = "B2-B1"

PREFIX = "dynamic_pgm1b"


# ===========================================================================
# Phi(S_t) + true-history memory
# ===========================================================================
def add_phi_and_memory(df):
    """Phi(S_t) = 10 frozen current-state re-encodings; mem = 6 true innovations."""
    x = lag.add_lag1_causal(df)
    phi_cols = []
    for node, (src, how) in rep.S_T_SOURCE.items():
        raw = x[src].to_numpy(np.float64)
        if how == "neg":
            raw = -raw
        for zc, vals in rep._encode_with_frozen(node, raw).items():
            name = f"phi_{zc}"
            x[name] = vals
            phi_cols.append(name)
    mem_cols = []
    for node, fld in (("z_dmfe", "path_max_up_excursion_R"),
                      ("z_dmae", "path_max_down_excursion_R")):
        raw = (x[fld].to_numpy(np.float64)
               - x[f"lag1_{fld}"].to_numpy(np.float64))
        for zc, vals in rep._encode_with_frozen(node, raw).items():
            name = f"mem_{zc}"
            x[name] = vals
            mem_cols.append(name)
    for side in ("upper", "lower"):
        fld = f"{side}_active_identity_count_delta"
        name = f"mem_z_delta_{side}_count"
        x[name] = x[fld] - x[f"lag1_{fld}"]
        mem_cols.append(name)
    if len(phi_cols) != 10:
        raise SystemExit(f"STOP_DYNAMIC_PGM1B_PHI_SIZE:{len(phi_cols)}")
    if len(mem_cols) != 6:
        raise SystemExit(f"STOP_DYNAMIC_PGM1B_MEM_SIZE:{len(mem_cols)}")
    if phi_cols != PHI_COLS or mem_cols != MEM_COLS:
        raise SystemExit("STOP_DYNAMIC_PGM1B_PHI_MEM_LAYOUT_MISMATCH")
    # hard causal audit: on episode-first rows the lag1 SOURCES are absent, so no
    # cross-episode information can enter. (The encoder maps NaN -> ispos 0.0, so
    # the check must be on the raw lag sources, not on mem_* being NaN.)
    first = (x["bar_t"].to_numpy(np.int64)
             - x["start_bar"].to_numpy(np.int64)) == 0
    if not bool((x.loc[first, LAG_AVAIL].to_numpy() == 0).all()):
        raise SystemExit("STOP_DYNAMIC_PGM1B_LAG_AVAIL_ON_FIRST_BAR")
    _src = [f"lag1_{f}" for f in
            ("path_max_up_excursion_R", "path_max_down_excursion_R",
             "upper_active_identity_count_delta",
             "lower_active_identity_count_delta")]
    if not x.loc[first, _src].isna().all().all():
        raise SystemExit("STOP_DYNAMIC_PGM1B_MEMORY_LEAK_ON_FIRST_BAR")
    if not bool(np.isfinite(
            x.loc[~first, MEM_COLS].to_numpy(np.float64)).all()):
        raise SystemExit("STOP_DYNAMIC_PGM1B_MEMORY_NAN_ON_NONFIRST_ROW")
    return x, phi_cols, mem_cols


# ===========================================================================
# reset pairs
# ===========================================================================
def build_reset_pairs(obs):
    """One row per legal episode n -> n+1 pair within the same symbol+block."""
    first = (obs.sort_values(["episode_id", "bar_t"], kind="stable")
             .groupby("episode_id", as_index=False).first())
    term = obs[obs["hazard"] == 1]
    if term["episode_id"].nunique() != len(term):
        raise SystemExit("STOP_DYNAMIC_PGM1B_MULTIPLE_TERMINALS")

    ep = term[["episode_id", "symbol", "block", "start_bar", "bar_t",
               "target_mask", "episode_start_day"] + OBS_NUM
              + PHI_COLS + MEM_COLS + [LAG_AVAIL]].copy()
    ep["end_bar"] = ep["bar_t"].to_numpy(np.int64) + 1
    ep = ep.sort_values(["symbol", "start_bar"], kind="stable").reset_index(
        drop=True)
    g = ep.groupby("symbol", sort=False)
    # chain_* names avoid a suffix collision with nxt's own next_* columns
    ep["next_episode_id"] = g["episode_id"].shift(-1)
    ep["chain_next_start_bar"] = g["start_bar"].shift(-1)
    ep["chain_next_block"] = g["block"].shift(-1)
    ep = ep[ep["next_episode_id"].notna()
            & (ep["block"] == ep["chain_next_block"])].copy()
    ep["next_episode_id"] = ep["next_episode_id"].astype(np.int64)
    ep["gap_bars"] = (ep["chain_next_start_bar"].astype(np.int64)
                      - ep["end_bar"].astype(np.int64))
    if (ep["gap_bars"] < 0).any():
        raise SystemExit("STOP_DYNAMIC_PGM1B_NEGATIVE_GAP")

    nxt = first.add_prefix("next_")
    pair = ep.merge(nxt, on="next_episode_id", how="left",
                    validate="one_to_one")
    pair["next_upper_span"] = (pair["next_upper_oldest_log_age"]
                               - pair["next_upper_newest_log_age"])
    pair["next_lower_span"] = (pair["next_lower_oldest_log_age"]
                               - pair["next_lower_newest_log_age"])
    pair["next_upper_n_active_minus1"] = (pair["next_upper_n_active_identities"]
                                          - 1.0)
    pair["next_lower_n_active_minus1"] = (pair["next_lower_n_active_identities"]
                                          - 1.0)
    pair["prev_endpoint_mask"] = pair["target_mask"].astype(np.int64)
    pair["gap_positive"] = (pair["gap_bars"] > 0).astype(np.int64)
    pair["log1p_gap"] = np.log1p(pair["gap_bars"].to_numpy(np.float64))
    # exact closure residual for the frozen PGM-BAR eps=10^-9 convention on raw price
    # differences vs normalized distance ratio
    pair["next_start_log_ratio_residual"] = (
        pair["next_start_log_ratio"].to_numpy(np.float64)
        - np.log(pair["next_start_up_distance_R"].to_numpy(np.float64)
                 / pair["next_start_down_distance_R"].to_numpy(np.float64)))
    return pair, first


def audit_reset_pairs(pair):
    """Hard structural audits on the chained reset pairs."""
    bad = {}
    bad["negative_gap"] = int((pair["gap_bars"] < 0).sum())
    bad["block_mismatch"] = int((pair["block"] != pair["next_block"]).sum())
    bad["prev_event_mask_mismatch"] = int(
        (pair["target_mask"].to_numpy(np.int64)
         != pair["next_prev_event_mask"].to_numpy(np.int64)).sum())
    bad["next_first_bar_not_start"] = int(
        (pair["next_bar_t"].to_numpy(np.int64)
         != pair["next_start_bar"].to_numpy(np.int64)).sum())
    bad["chain_start_bar_mismatch"] = int(
        (pair["chain_next_start_bar"].to_numpy(np.int64)
         != pair["next_start_bar"].to_numpy(np.int64)).sum())
    bad["next_geometry_positive"] = int(
        (pair[RESET_GEOM].to_numpy(np.float64) <= 0).sum())
    bad["next_logratio_residual_nonfinite"] = int(
        (~np.isfinite(pair[RESET_LOGRATIO_RESID].to_numpy(np.float64))).sum())
    bad["next_shape_nonneg"] = int((pair[RESET_SHAPE].to_numpy(np.float64)
                                    < 0).sum())
    bad["next_age_nonneg"] = int((pair[RESET_AGE].to_numpy(np.float64) < 0).sum())
    bad["next_count_integer_nonneg"] = int(
        ((pair[RESET_CNT].to_numpy(np.float64) < 0)
         | (np.mod(pair[RESET_CNT].to_numpy(np.float64), 1) != 0)).sum())
    if any(v != 0 for v in bad.values()):
        raise SystemExit(f"STOP_DYNAMIC_PGM1B_RESET_PAIR_AUDIT_FAIL: {bad}")
    return dict(checks=bad, n_pairs=int(len(pair)),
                n_immediate=int((pair["gap_bars"] == 0).sum()),
                n_positive_gap=int((pair["gap_bars"] > 0).sum()),
                max_gap=int(pair["gap_bars"].max()))


def audit_reset_reconstruction(pair, first):
    """Rebuild next-episode first-row state from the 11 primitives."""
    fu = pair["next_start_up_distance_R"].to_numpy(np.float64)
    fd = pair["next_start_down_distance_R"].to_numpy(np.float64)
    m_u = pair["next_path_max_up_excursion_R"].to_numpy(np.float64)
    m_d = pair["next_path_max_down_excursion_R"].to_numpy(np.float64)
    r_log = pair["next_start_log_ratio_residual"].to_numpy(np.float64)

    # Reconstructed start_log_ratio uses normalized distance ratio + the exact
    # closure residual r_log, closing the frozen PGM-BAR eps=10^-9 convention.
    reconstructed_start_log_ratio = np.log(fu / fd) + r_log
    lr_err = float(np.abs(reconstructed_start_log_ratio
                          - pair["next_start_log_ratio"].to_numpy(
                              np.float64)).max())

    err = {}
    err["start_log_ratio"] = lr_err
    err["width"] = float(np.abs((fu + fd)
                                - pair["next_start_width_R"]).max())
    err["oldest_age"] = float(np.abs(
        (pair["next_upper_newest_log_age"] + pair["next_upper_span"])
        - pair["next_upper_oldest_log_age"]).max())
    err["lower_oldest_age"] = float(np.abs(
        (pair["next_lower_newest_log_age"] + pair["next_lower_span"])
        - pair["next_lower_oldest_log_age"]).max())
    err["n_active"] = float(np.abs(
        (pair["next_upper_n_active_minus1"] + 1.0)
        - pair["next_upper_n_active_identities"]).max())
    err["range"] = float(np.abs(
        (m_u + m_d) - pair["next_path_current_bar_range_R"]).max())
    err["elapsed_zero"] = float(np.abs(
        pair["next_elapsed_log"].to_numpy(np.float64)).max())
    err["tv_zero"] = float(np.abs(
        pair["next_path_total_variation_R"].to_numpy(np.float64)).max())
    err["dcr_zero"] = float(np.abs(
        pair["next_path_direction_change_rate"].to_numpy(np.float64)).max())
    err["last_return_zero"] = float(np.abs(
        pair["next_path_last_return_R"].to_numpy(np.float64)).max())
    err["newest_residual_zero"] = float(max(np.abs(
        pair["next_upper_newest_log_age_residual"].to_numpy(np.float64)).max(),
        np.abs(pair["next_lower_newest_log_age_residual"].to_numpy(
            np.float64)).max()))
    err["count_delta_zero"] = float(max(np.abs(
        pair["next_upper_active_identity_count_delta"].to_numpy(
            np.float64)).max(),
        np.abs(pair["next_lower_active_identity_count_delta"].to_numpy(
            np.float64)).max()))
    err["tempo_zero"] = float(np.abs(
        pair["next_tempo_signed_speed"].to_numpy(np.float64)).max())
    err["prev_event_mask"] = float((pair["next_prev_event_mask"].to_numpy(
        np.int64) != pair["prev_endpoint_mask"].to_numpy(np.int64)).sum())
    worst = max(v for v in err.values())
    if worst >= 1e-8:
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1B_RESET_RECONSTRUCTION_FAIL: {err}")
    return dict(
        per_field_max_error=err, max_error=worst,
        logratio_residual=dict(
            min=float(r_log.min()),
            max=float(r_log.max()),
            mean=float(r_log.mean()),
            std=float(r_log.std()),
            max_abs=float(np.abs(r_log).max()),
            note=("Exact closure residual r_log closes the frozen PGM-BAR eps=10^-9 "
                  "convention on raw price differences vs normalized distance ratio; "
                  "reconstruction error is within machine precision (< 1e-8).")))





# ===========================================================================
# gap model (shared Geometric magnitude)
# ===========================================================================
def fit_gap_geometric(g_train):
    g = np.asarray(g_train, dtype=np.float64)
    pos = g[g > 0]
    if not len(pos):
        return 1.0
    return float(np.clip(1.0 / float(np.mean(pos)), 1e-8, 1.0))


def gap_positive_nll(g, p):
    g = np.asarray(g, dtype=np.float64)
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    out = np.zeros(len(g), dtype=np.float64)
    pos = g > 0
    out[pos] = -np.log(p) - (g[pos] - 1.0) * np.log1p(-p)
    return out


def _fit_occurrence(Xtr, ytr, Xev, name):
    from sklearn.linear_model import LogisticRegression
    ytr = np.asarray(ytr, dtype=np.int64)
    if len(np.unique(ytr)) < 2:
        c = 1.0 - 1e-6 if ytr[0] == 1 else 1e-6
        return np.full(len(Xev), c)
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs",
                             max_iter=3000).fit(Xtr, ytr)
    base._check_logistic_convergence(clf, name)
    return np.clip(clf.predict_proba(Xev)[:, 1], 1e-6, 1.0 - 1e-6)


def _encode_nonneg(raw):
    raw = np.asarray(raw, dtype=np.float64)
    ispos = (raw > 0).astype(np.float64)
    logv = np.where(ispos > 0.5, np.log(np.maximum(raw, 1e-12)), 0.0)
    return np.stack([ispos, logv], axis=1)


def _design(tr, ev, num_cols, cat_cols):
    """Fit the SAME preprocessing pm.make_pipeline uses, but only the
    preprocessor -- never the classifier (that is fitted per head later)."""
    from research.liquidity_oracle_atlas.experiment_pgm0_endpoint_coupling_v1 import densify
    cols = list(num_cols) + list(cat_cols)
    pre = pm.make_pipeline(list(num_cols), list(cat_cols)).named_steps["pre"]
    pre.fit(tr[cols])
    return densify(pre.transform(tr[cols])), densify(pre.transform(ev[cols]))


def _fit_heads(Xtr, Xev, tr, ev, gap_p):
    """Fit every reset-state head on the SAME design matrix; return per-row NLL."""
    out = {}
    ytr_log = np.log(tr[RESET_GEOM].to_numpy(np.float64))
    yev_log = np.log(ev[RESET_GEOM].to_numpy(np.float64))
    gh = base.GaussianTransitionHead(alpha=1.0).fit(Xtr, ytr_log)
    out["geometry"] = gh.nll_per_row(Xev, yev_log) + yev_log.sum(axis=1)

    ytr_rlog = tr[RESET_LOGRATIO_RESID].to_numpy(np.float64)
    yev_rlog = ev[RESET_LOGRATIO_RESID].to_numpy(np.float64)
    gh_rlog = base.GaussianTransitionHead(alpha=1.0).fit(Xtr, ytr_rlog)
    out["logratio_residual"] = gh_rlog.nll_per_row(Xev, yev_rlog)

    for name in RESET_SHAPE + RESET_AGE:
        ztr = _encode_nonneg(tr[name].to_numpy(np.float64))
        zev = _encode_nonneg(ev[name].to_numpy(np.float64))
        h = base.HurdleLogNormalHead(alpha=1.0, sign=1.0).fit(Xtr, ztr)
        out[name] = h.nll_per_row(Xev, zev)

    Ytr = tr[RESET_CNT].to_numpy(np.int64)
    Yev = ev[RESET_CNT].to_numpy(np.int64)
    k0c = base.fit_constant_count_head(Ytr, Yev)
    kc = base.fit_state_count_head(Xtr, Ytr, Xev, Yev,
                                   constant_rates=k0c["constant_rates"])
    out["counts"] = kc["nll_ev"]
    out["_gap_p"] = gap_p
    return out


def reset_nll(decomp, gap_occ, gap_p):
    """Total per-pair reset NLL."""
    n = len(gap_occ)
    total = np.array(gap_occ, dtype=np.float64)
    total = total + gap_positive_nll(decomp["_gap"], gap_p)
    total = total + decomp["geometry"]
    total = total + decomp["logratio_residual"]
    for name in RESET_SHAPE + RESET_AGE:
        total = total + decomp[name]
    total = total + decomp["counts"].sum(axis=1)
    return total


def boot_episode_day_cluster(delta, day, seed, reps=BOOT_REPS):
    """Cluster-bootstrap on trading days with replacement, preserving multiplicity.

    Estimand: episode-weighted mean (1/N) * sum_i d_i.
    Resampling: trading days sampled with replacement. When a day is selected
    multiple times, all its episode rows are duplicated with multiplicity.
    """
    delta = np.asarray(delta, dtype=np.float64)
    day = np.asarray(day)
    uniq = np.unique(day)
    idx_by_day = {d: np.flatnonzero(day == d) for d in uniq}
    point = float(delta.mean())
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=np.float64)
    for b in range(reps):
        selected_days = uniq[rng.integers(0, len(uniq), len(uniq))]
        sel = np.concatenate([idx_by_day[d] for d in selected_days])
        out[b] = delta[sel].mean()
    return (
        float(np.percentile(out, 2.5)),
        float(np.percentile(out, 97.5)),
        point,
    )


# ===========================================================================
# one window
# ===========================================================================
def run_single_window_1b(w, data_path):
    t_w = time.perf_counter()
    obs = pd.read_parquet(data_path)
    print(f"[STAGE] window {w['name']} loaded {len(obs)}", flush=True)

    tr = obs[obs["block"].isin(w["train"])].reset_index(drop=True)
    ev = obs[obs["block"] == w["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_DYNAMIC_PGM1B_EMPTY_SPLIT: {w['name']}")

    term_metrics, term_boots, term_bysym = [], [], []
    reset_metrics, reset_boots, reset_bysym = [], [], []
    bound_boots, bound_bysym = [], []
    gap_rows, target_rows = [], []
    opt_rows = []
    per_ep = {}

    # ---------------- terminal ----------------
    term_specs = [("TP_FROZEN_PTc", TP_NUM), ("T0_STATE_AVAIL", T0_NUM),
                  ("T1_STATE_PHI", T1_NUM), ("T2_STATE_PHI_MEM", T2_NUM)]
    for tag, num_cols in term_specs:
        t0 = time.perf_counter()
        out = ms.fit_eval(tag, num_cols, CAT, tr, ev)
        met = ms.model_metrics(ev, out["p_h"], out["p_mask"], out["te"])
        term_metrics.append(dict(window=w["name"], model=tag, **met))
        h = ev["hazard"].to_numpy().astype(np.int64)
        uniq, per = ms.episode_nll(
            out["p_h"], out["p_mask"], h,
            ev["target_mask"].to_numpy().astype(np.int64),
            ev["episode_id"].to_numpy())
        per_ep[tag] = (uniq, per)
        opt_rows.append(dict(window=w["name"], part="terminal", model=tag,
                             success=True,
                             elapsed_seconds=round(
                                 time.perf_counter() - t0, 3)))
    ep_day = ev.groupby("episode_id")["episode_start_day"].first()
    ep_sym = ev.groupby("episode_id")["symbol"].first()

    for label, hi, lo in [("T0-TP", "T0_STATE_AVAIL", "TP_FROZEN_PTc"),
                          (C_TERM_PHI, "T1_STATE_PHI", "T0_STATE_AVAIL"),
                          (C_TERM_MEM, "T2_STATE_PHI_MEM", "T1_STATE_PHI")]:
        u_h, p_h = per_ep[hi]
        u_l, p_l = per_ep[lo]
        if not np.array_equal(u_h, u_l):
            raise SystemExit(f"STOP_DYNAMIC_PGM1B_EPISODE_MISALIGN:{label}")
        d = p_h - p_l
        day = ep_day.reindex(pd.Index(u_h)).to_numpy()
        sym = ep_sym.reindex(pd.Index(u_h)).to_numpy()
        a, b, pt = boot_episode_day_cluster(d, day, w["seed"], reps=BOOT_REPS)
        term_boots.append(dict(
            window=w["name"], comparison=label,
            delta_sample_mean=pt, ci_lo=a, ci_hi=b,
            n_episodes=int(len(d)),
            estimand="EPISODE_WEIGHTED_MEAN",
            cluster="TRADING_DAY",
            resampling="WITH_REPLACEMENT_MULTIPLICITY_PRESERVED",
            verdict=("CI_below_zero" if b < 0 else
                     "CI_above_zero" if a > 0 else "CI_contains_zero")))
        for s_ in ms.FULL_UNIV:
            idx = sym == s_
            if idx.any():
                dd = d[idx]
                term_bysym.append(dict(
                    window=w["name"], comparison=label, symbol=s_,
                    n_episodes=int(idx.sum()), mean_delta=float(dd.mean()),
                    n_negative=int((dd < 0).sum()),
                    n_positive=int((dd > 0).sum())))

    # ---------------- reset ----------------
    pair, first = build_reset_pairs(obs)
    ptr = pair[pair["block"].isin(w["train"])].reset_index(drop=True)
    pev = pair[pair["block"] == w["eval"]].reset_index(drop=True)
    ptr = ptr[ptr["next_episode_id"].notna()].reset_index(drop=True)
    pev = pev[pev["next_episode_id"].notna()].reset_index(drop=True)
    if not len(ptr) or not len(pev):
        raise SystemExit(f"STOP_DYNAMIC_PGM1B_EMPTY_RESET_SPLIT: {w['name']}")

    gap_p = fit_gap_geometric(ptr["gap_bars"].to_numpy(np.float64))
    gap_rows.append(dict(
        window=w["name"], n_train_pairs=int(len(ptr)),
        n_eval_pairs=int(len(pev)),
        n_train_positive_gap=int((ptr["gap_bars"] > 0).sum()),
        n_eval_positive_gap=int((pev["gap_bars"] > 0).sum()),
        geometric_p=gap_p,
        geometric_mean=float(1.0 / gap_p),
        mean_gap=float(ptr["gap_bars"].mean())))

    reset_specs = [("R0_ENDPOINT_ONLY", R0_OCC), ("R1_STATE_PHI", R1_OCC),
                   ("R2_STATE_PHI_MEM", R2_OCC)]
    reset_nlls = {}
    for tag, occ_cols in reset_specs:
        t0 = time.perf_counter()
        state_cols = list(occ_cols) + GAP_FEAT
        Xtr_o, Xev_o = _design(ptr, pev, occ_cols, MASK_CAT)
        Xtr_s, Xev_s = _design(ptr, pev, state_cols, MASK_CAT)
        gap_occ = _fit_occurrence(Xtr_o, ptr["gap_positive"].to_numpy(np.int64),
                                  Xev_o, f"{tag}_gap_occurrence")
        nll_occ = np.where(pev["gap_positive"].to_numpy(np.int64) == 1,
                           -np.log(gap_occ), -np.log1p(-gap_occ))
        decomp = _fit_heads(Xtr_s, Xev_s, ptr, pev, gap_p)
        decomp["_gap"] = pev["gap_bars"].to_numpy(np.float64)
        total = reset_nll(decomp, nll_occ, gap_p)
        if not np.all(np.isfinite(total)):
            raise SystemExit(f"STOP_DYNAMIC_PGM1B_NONFINITE_RESET_NLL:{tag}")
        reset_nlls[tag] = total
        reset_metrics.append(dict(
            window=w["name"], model=tag,
            mean_reset_nll=float(total.mean()),
            mean_gap_occ_nll=float(nll_occ.mean()),
            mean_gap_mag_nll=float(gap_positive_nll(
                pev["gap_bars"].to_numpy(np.float64), gap_p).mean()),
            mean_geometry_nll=float(decomp["geometry"].mean()),
            mean_logratio_resid_nll=float(decomp["logratio_residual"].mean()),
            mean_shape_nll=float(np.mean([decomp[n].mean()
                                          for n in RESET_SHAPE])),
            mean_age_nll=float(np.mean([decomp[n].mean()
                                        for n in RESET_AGE])),
            mean_count_nll=float(decomp["counts"].sum(axis=1).mean()),
            n_pairs=int(len(pev))))
        opt_rows.append(dict(window=w["name"], part="reset", model=tag,
                             success=True,
                             elapsed_seconds=round(
                                 time.perf_counter() - t0, 3)))
        for name in RESET_SHAPE + RESET_AGE:
            target_rows.append(dict(
                window=w["name"], model=tag, target=name,
                mean_nll=float(decomp[name].mean())))
        target_rows.append(dict(window=w["name"], model=tag, target="geometry",
                                mean_nll=float(decomp["geometry"].mean())))
        target_rows.append(dict(window=w["name"], model=tag,
                                target="next_start_log_ratio_residual",
                                mean_nll=float(decomp["logratio_residual"].mean())))
        for j_, c_ in enumerate(RESET_CNT):
            target_rows.append(dict(window=w["name"], model=tag, target=c_,
                                    mean_nll=float(
                                        decomp["counts"][:, j_].mean())))

    pr_day = pev["episode_start_day"].to_numpy()
    pr_sym = pev["symbol"].to_numpy()
    for label, hi, lo in [(C_RESET_STATE, "R1_STATE_PHI", "R0_ENDPOINT_ONLY"),
                          (C_RESET_MEM, "R2_STATE_PHI_MEM", "R1_STATE_PHI")]:
        d = reset_nlls[hi] - reset_nlls[lo]
        a, b, pt = boot_episode_day_cluster(d, pr_day, w["seed"], reps=BOOT_REPS)
        reset_boots.append(dict(
            window=w["name"], comparison=label,
            delta_sample_mean=pt, ci_lo=a, ci_hi=b,
            n_pairs=int(len(d)),
            estimand="EPISODE_WEIGHTED_MEAN",
            cluster="TRADING_DAY",
            resampling="WITH_REPLACEMENT_MULTIPLICITY_PRESERVED",
            verdict=("CI_below_zero" if b < 0 else
                     "CI_above_zero" if a > 0 else "CI_contains_zero")))
        for s_ in ms.FULL_UNIV:
            idx = pr_sym == s_
            if idx.any():
                dd = d[idx]
                reset_bysym.append(dict(
                    window=w["name"], comparison=label, symbol=s_,
                    n_pairs=int(idx.sum()), mean_delta=float(dd.mean()),
                    n_negative=int((dd < 0).sum()),
                    n_positive=int((dd > 0).sum())))

    # ---------------- boundary joint ----------------
    paired_eids = pd.Index(pev["episode_id"].to_numpy())
    b_nll = {}
    for i, (t_tag, r_tag) in enumerate(
            [("T0_STATE_AVAIL", "R0_ENDPOINT_ONLY"),
             ("T1_STATE_PHI", "R1_STATE_PHI"),
             ("T2_STATE_PHI_MEM", "R2_STATE_PHI_MEM")]):
        u_t, p_t = per_ep[t_tag]
        term_ser = pd.Series(p_t, index=pd.Index(u_t))
        term_sel = term_ser.reindex(paired_eids).to_numpy()
        if np.any(~np.isfinite(term_sel)):
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1B_BOUNDARY_EPISODE_MISSING:{t_tag}")
        b_nll[f"B{i}"] = term_sel + reset_nlls[r_tag]

    b_day = pev["episode_start_day"].to_numpy()
    b_sym = pev["symbol"].to_numpy()
    for label, hi, lo in [(C_BOUND_PHI, "B1", "B0"), (C_BOUND_MEM, "B2", "B1")]:
        d = b_nll[hi] - b_nll[lo]
        a, b, pt = boot_episode_day_cluster(d, b_day, w["seed"], reps=BOOT_REPS)
        bound_boots.append(dict(
            window=w["name"], comparison=label,
            delta_sample_mean=pt, ci_lo=a, ci_hi=b,
            n_episodes=int(len(d)),
            estimand="EPISODE_WEIGHTED_MEAN",
            cluster="TRADING_DAY",
            resampling="WITH_REPLACEMENT_MULTIPLICITY_PRESERVED",
            verdict=("CI_below_zero" if b < 0 else
                     "CI_above_zero" if a > 0 else "CI_contains_zero")))
        for s_ in ms.FULL_UNIV:
            idx = b_sym == s_
            if idx.any():
                dd = d[idx]
                bound_bysym.append(dict(
                    window=w["name"], comparison=label, symbol=s_,
                    n_episodes=int(idx.sum()), mean_delta=float(dd.mean()),
                    n_negative=int((dd < 0).sum()),
                    n_positive=int((dd > 0).sum())))

    del obs
    print(f"[STAGE] window {w['name']} done "
          f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)
    return dict(window=w["name"],
                term_metrics=term_metrics, term_boots=term_boots,
                term_bysym=term_bysym,
                reset_metrics=reset_metrics, reset_boots=reset_boots,
                reset_bysym=reset_bysym,
                bound_boots=bound_boots, bound_bysym=bound_bysym,
                gap_rows=gap_rows, target_rows=target_rows,
                opt_rows=opt_rows,
                n_symbols=len(ms.FULL_UNIV))


# ===========================================================================
# main
# ===========================================================================
def _audit_sample(obs):
    n_rows = int(len(obs))
    n_ep = int(obs["episode_id"].nunique())
    n_term = int(obs["hazard"].sum())
    if (n_rows, n_ep, n_term) != (EXPECTED_ROWS, EXPECTED_EPISODES,
                                  EXPECTED_TERMINALS):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1B_SAMPLE_PARITY_FAIL: "
            f"{n_rows},{n_ep},{n_term}")
    if "TB4" in set(obs["block"].unique()):
        raise SystemExit("STOP_DYNAMIC_PGM1B_TB4_PRESENT")
    ep0 = pd.read_parquet(base.CACHE / "episode0_episodes.parquet")
    h = pm.episode_identity_hash(ep0)
    if h != FROZEN_EPISODE_HASH:
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1B_EPISODE_HASH_FAIL: {h}")
    term = obs[obs["hazard"] == 1]
    first = (obs.sort_values(["episode_id", "bar_t"], kind="stable")
             .groupby("episode_id", as_index=False).first())
    ok_first = int((first["bar_t"].to_numpy(np.int64)
                    == first["start_bar"].to_numpy(np.int64)).sum())
    return dict(n_rows=n_rows, n_episodes=n_ep, n_terminals=n_term,
                episode_identity_hash=h, no_tb4=True,
                first_row_is_start_bar_ok=int(ok_first == n_ep),
                blocks={b: int((obs["block"] == b).sum())
                        for b in ["TB1", "TB2", "TB3"]})


def main():
    t_total = time.perf_counter()
    timing = {}

    t0 = time.perf_counter()
    obs = pd.read_parquet(base.CACHE / "market_state1_samples.parquet")
    sample_audit = _audit_sample(obs)
    obs, phi_cols, mem_cols = add_phi_and_memory(obs)
    base._peak("after add_phi_and_memory")
    timing["prepare_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[STAGE] prepared n={len(obs)} phi={len(phi_cols)} "
          f"mem={len(mem_cols)}", flush=True)

    pair, first = build_reset_pairs(obs)
    pair_audit = audit_reset_pairs(pair)
    recon_audit = audit_reset_reconstruction(pair, first)
    print(f"[AUDIT] pairs={pair_audit['n_pairs']} "
          f"immediate={pair_audit['n_immediate']} "
          f"positive_gap={pair_audit['n_positive_gap']}", flush=True)

    (base.OUT / f"{PREFIX}_sample_audit.json").write_text(
        json.dumps(sample_audit, indent=2, default=str))
    (base.OUT / f"{PREFIX}_reset_reconstruction_audit.json").write_text(
        json.dumps(dict(pairs=pair_audit, reconstruction=recon_audit),
                   indent=2, default=str))

    if os.environ.get("DYNAMIC_PGM1B_AUDIT_ONLY") == "1":
        (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
            json.dumps(dict(frozen_terminal=FROZEN_TERMINAL,
                            episode_identity_hash=FROZEN_EPISODE_HASH,
                            sample=sample_audit), indent=2, default=str))
        print("[AUDIT-ONLY] Complete. Stopping before any model fit.",
              flush=True)
        return dict(sample=sample_audit, pairs=pair_audit,
                    reconstruction=recon_audit)

    # persist cache (native dtypes: float32 would break terminal parity)
    keep = (OBS_NUM + CAT + PHI_COLS + MEM_COLS + [LAG_AVAIL]
            + ["symbol", "episode_id", "start_bar", "bar_t", "block",
               "episode_start_day", "hazard", "target_mask", "prev_event_mask",
               "upper_oldest_log_age", "upper_newest_log_age",
               "lower_oldest_log_age", "lower_newest_log_age",
               "upper_n_active_identities", "lower_n_active_identities",
               "start_up_distance_R", "start_down_distance_R",
               "start_width_R", "start_log_ratio",
               "path_max_up_excursion_R", "path_max_down_excursion_R",
               "path_current_bar_range_R", "path_total_variation_R",
               "path_direction_change_rate", "path_last_return_R",
               "elapsed_log", "upper_newest_log_age_residual",
               "lower_newest_log_age_residual",
               "upper_active_identity_count_delta",
               "lower_active_identity_count_delta",
               "tempo_signed_speed", "tempo_abs_speed",
               "tempo_signed_efficiency", "tempo_abs_efficiency"])
    keep = list(dict.fromkeys([c for c in keep if c in obs.columns]))
    data_path = base.CACHE / f"{PREFIX}_sample.parquet"
    obs[keep].to_parquet(data_path, index=False)
    del obs
    base._peak("after slim/persist")

    term_metrics, term_boots, term_bysym = [], [], []
    reset_metrics, reset_boots, reset_bysym = [], [], []
    bound_boots, bound_bysym = [], []
    gap_rows, target_rows, opt_rows = [], [], []
    for w in WINDOWS:
        res = run_single_window_1b(w, data_path)
        term_metrics += res["term_metrics"]
        term_boots += res["term_boots"]
        term_bysym += res["term_bysym"]
        reset_metrics += res["reset_metrics"]
        reset_boots += res["reset_boots"]
        reset_bysym += res["reset_bysym"]
        bound_boots += res["bound_boots"]
        bound_bysym += res["bound_bysym"]
        gap_rows += res["gap_rows"]
        target_rows += res["target_rows"]
        opt_rows += res["opt_rows"]

    # ---------------- terminal parity gate ----------------
    parity = {"frozen": FROZEN_TERMINAL, "windows": {}}
    for w in WINDOWS:
        got = next(m for m in term_metrics
                   if m["window"] == w["name"] and m["model"] == "TP_FROZEN_PTc")
        fr = FROZEN_TERMINAL[w["name"]]
        d = {k: abs(float(got[k]) - fr[k])
             for k in ("hazard_nll", "endpoint_joint_nll", "mean_episode_nll")}
        ok = all(v < PARITY_TOL for v in d.values())
        parity["windows"][w["name"]] = dict(abs_diff=d, ok=bool(ok))
        if not ok:
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1B_TERMINAL_PARITY_FAIL: {w['name']} {d}")
    parity["ok"] = True

    def _pass(rows, wname, label):
        r = next(b for b in rows if b["window"] == wname
                 and b["comparison"] == label)
        return bool(r["ci_hi"] < 0)

    def _neg(bysym, wname, label):
        sub = [b for b in bysym if b["window"] == wname
               and b["comparison"] == label]
        return int(sum(1 for b in sub if b["mean_delta"] < 0))

    def _verdict(rows, bysym, label, tag):
        per = {w["name"]: dict(
            delta=next(b for b in rows if b["window"] == w["name"]
                       and b["comparison"] == label)["delta_sample_mean"],
            ci_lo=next(b for b in rows if b["window"] == w["name"]
                       and b["comparison"] == label)["ci_lo"],
            ci_hi=next(b for b in rows if b["window"] == w["name"]
                       and b["comparison"] == label)["ci_hi"],
            n_symbols_negative=_neg(bysym, w["name"], label),
            n_symbols_total=len(ms.FULL_UNIV),
            passes=bool(_pass(rows, w["name"], label)
                        and _neg(bysym, w["name"], label)
                        >= SYMBOL_BREADTH_MIN))
            for w in WINDOWS}
        return per, all(v["passes"] for v in per.values())

    term_phi, ok_term = _verdict(term_boots, term_bysym, C_TERM_PHI,
                                 "TERMINAL_PHI")
    reset_st, ok_reset = _verdict(reset_boots, reset_bysym, C_RESET_STATE,
                                  "RESET_STATE")
    bound_phi, ok_bound = _verdict(bound_boots, bound_bysym, C_BOUND_PHI,
                                   "BOUNDARY_PHI")
    mem_report = {}
    for label, rows, bysym in [("T2-T1", term_boots, term_bysym),
                               ("R2-R1", reset_boots, reset_bysym),
                               ("B2-B1", bound_boots, bound_bysym)]:
        mem_report[label] = {
            w["name"]: dict(
                delta=next(b for b in rows if b["window"] == w["name"]
                           and b["comparison"] == label)["delta_sample_mean"],
                ci_lo=next(b for b in rows if b["window"] == w["name"]
                           and b["comparison"] == label)["ci_lo"],
                ci_hi=next(b for b in rows if b["window"] == w["name"]
                           and b["comparison"] == label)["ci_hi"],
                n_symbols_negative=_neg(bysym, w["name"], label))
            for w in WINDOWS}

    verdicts = dict(
        TERMINAL_PHI_SUPPORTED=bool(ok_term),
        RESET_STATE_SIGNAL_SUPPORTED=bool(ok_reset),
        BOUNDARY_PHI_SUPPORTED=bool(ok_bound),
        DYNAMIC_PGM1B_BOUNDARY_KERNEL_CLOSED=True)
    print(f"[VERDICTS] {verdicts}", flush=True)

    # ---------------- outputs ----------------
    pd.DataFrame(term_metrics).to_csv(
        base.OUT / f"{PREFIX}_terminal_metrics.csv", index=False)
    pd.DataFrame(term_boots).to_csv(
        base.OUT / f"{PREFIX}_terminal_bootstrap.csv", index=False)
    pd.DataFrame(term_bysym).to_csv(
        base.OUT / f"{PREFIX}_terminal_by_symbol.csv", index=False)
    pd.DataFrame(reset_metrics).to_csv(
        base.OUT / f"{PREFIX}_reset_metrics.csv", index=False)
    pd.DataFrame(reset_boots).to_csv(
        base.OUT / f"{PREFIX}_reset_bootstrap.csv", index=False)
    pd.DataFrame(reset_bysym).to_csv(
        base.OUT / f"{PREFIX}_reset_by_symbol.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        base.OUT / f"{PREFIX}_reset_target_metrics.csv", index=False)
    pd.DataFrame(gap_rows).to_csv(
        base.OUT / f"{PREFIX}_gap_diagnostics.csv", index=False)
    pd.DataFrame(bound_boots).to_csv(
        base.OUT / f"{PREFIX}_boundary_bootstrap.csv", index=False)
    pd.DataFrame(bound_bysym).to_csv(
        base.OUT / f"{PREFIX}_boundary_by_symbol.csv", index=False)
    pd.DataFrame(opt_rows).to_csv(
        base.OUT / f"{PREFIX}_optimizer_audit.csv", index=False)
    (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
        json.dumps(parity, indent=2, default=str))

    summary = dict(
        experiment="DYNAMIC-PGM-1B Terminal + Reset Closure",
        parent_commit=BASE_SHA,
        verdicts=verdicts,
        phi=list(PHI_COLS), mem=list(MEM_COLS),
        terminal=dict(models=dict(
            TP="S_t (frozen PTc parity)",
            T0="S_t + A_t", T1="T0 + Phi(10)", T2="T1 + mem(6)"),
            metrics=term_metrics, bootstrap=term_boots,
            phi_gate=term_phi),
        reset=dict(models=dict(
            R0="E + A", R1="R0 + S_t + Phi(10)", R2="R1 + mem(6)"),
            metrics=reset_metrics, bootstrap=reset_boots,
            state_gate=reset_st, gap=gap_rows),
        boundary=dict(bootstrap=bound_boots, phi_gate=bound_phi),
        memory_challenger=mem_report,
        sample=sample_audit, pairs=pair_audit, reconstruction=recon_audit,
        terminal_parity=parity,
        bootstrap_reps=BOOT_REPS,
        bootstrap_estimand="EPISODE_WEIGHTED_MEAN",
        bootstrap_cluster="TRADING_DAY",
        bootstrap_resampling="WITH_REPLACEMENT_MULTIPLICITY_PRESERVED",
    )
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (base.OUT / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[TERMINAL METRICS]\n{pd.DataFrame(term_metrics).to_string(index=False)}")
    print(f"[TERMINAL BOOTSTRAP]\n{pd.DataFrame(term_boots).to_string(index=False)}")
    print(f"[RESET METRICS]\n{pd.DataFrame(reset_metrics).to_string(index=False)}")
    print(f"[RESET BOOTSTRAP]\n{pd.DataFrame(reset_boots).to_string(index=False)}")
    print(f"[BOUNDARY BOOTSTRAP]\n{pd.DataFrame(bound_boots).to_string(index=False)}")
    print(f"[DONE] -> {base.OUT}")
    return summary


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--audit-only", action="store_true")
    _a = _ap.parse_args()
    if _a.audit_only:
        os.environ["DYNAMIC_PGM1B_AUDIT_ONLY"] = "1"
    main()
