"""DYNAMIC-PGM-1A.2 -- Lag Closure.

Question
--------
Given the current observed Market State S_t, does knowing the previous bar's
dynamic state S_{t-1} still add STABLE extra information for predicting Z_{t+1}?

This decides whether the world model can be approximated as first-order Markov

    P(S_{t+1} | S_t)

or needs at least

    P(S_{t+1} | S_t, S_{t-1}).

Design
------
Everything frozen from 1A.1b is REUSED by importing it as `base`:

    transition sample, support-correct heads, reconstruction,
    count magnitude closure, bootstrap, invariants.

The only new objects in this file are:

    1. lag1 construction (causal, episode-internal) + `lag1_available`
    2. K1A and K2 (two new models)

Four nested models
------------------
    K0  = P(Z)                        no state
    K1  = P(Z | S_t)                  frozen 1A.1b baseline
    K1A = P(Z | S_t, A_t)             CONTROL: availability indicator only
    K2  = P(Z | S_t, A_t, S_{t-1})    lag1 block

Why K1A exists (confound control)
---------------------------------
`lag1_available` is essentially an "is this the episode's first bar" indicator.
Although it is in principle derivable from S_t, our heads are linear
(Logistic / Ridge), so a 0/1 column also acts as an extra nonlinear basis
function. Without a control we could not tell whether a K2-K1 gain came from
genuine one-step memory or from the model more easily identifying episode
starts. Therefore:

    K1A - K1   -> episode-start / representation effect      (reported only)
    K2  - K1A  -> genuine S_{t-1} residual memory            (PRIMARY gate)

The lag verdict reads K2-K1A ONLY.

K0 and K1 are re-fitted here and MUST reproduce the frozen 1A.1b numbers
(rows / z_feature_hash / joint NLL) -- see PARITY gate below.

Frozen scope (do NOT expand in this experiment)
-----------------------------------------------
no latent/HMM/HSMM, no lag2/lag3, no alpha/C tuning, no SMC/OB,
no terminal/reset, no rollout, no PnL/RL, no new feature mining.

Count Magnitude Closure
-----------------------
P(A|S) = P(A>0|S) * P(A|A>0).  Occurrence is state-dependent Logistic and may
differ across K1 / K1A / K2.  Magnitude is a state-independent exact ZTP with a
TRAIN-ONLY constant lambda SHARED IDENTICALLY by K0, K1, K1A and K2.  Therefore
both K2-K1A and K2-K1 count gains can only come from predicting *whether* an
activation happens, never from predicting its magnitude.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             log_loss)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base  # noqa: E402

# ===========================================================================
# frozen 1A.1b reference (parity gate)
# ===========================================================================
BASE_SHA = "16ab731ac099ffcec7c2089640b95b5555184853"
PARENT_SHA = base.BASE_SHA
EXPECTED_TRANSITIONS = base.EXPECTED_TRANSITIONS          # 321727
FROZEN_Z_FEATURE_HASH = (
    "ae9bcf818d9f8528409af1e11692b949fb6710e7f59fac73369a92630608ab30")
# frozen 1A.1b mean joint NLL (dynamic_pgm1a1b_model_metrics.csv @ BASE_SHA)
FROZEN_JOINT = {
    "A_TB1_to_TB2": dict(K0_UNCONDITIONAL=3.045832444543554,
                         K1_STATE=1.3807112331885323),
    "B_TB1TB2_to_TB3": dict(K0_UNCONDITIONAL=3.0543425100566743,
                            K1_STATE=1.412292223427112),
}
PARITY_TOL = 1e-8
SYMBOL_BREADTH_MIN = 10          # >= 10 / 15 symbols negative

# ===========================================================================
# lag1 block (identical to base.LAG_BASE; re-declared for explicitness)
#   3 dynamic location + 6 dynamic path + 6 compact liquidity provenance = 15
# Explicitly EXCLUDED: start geometry, previous endpoint, episode context,
# TEMPO (derived representation of current state), lag2, static context.
# ===========================================================================
LAG_BASE = [
    "cur_up_distance_R", "cur_down_distance_R", "cur_log_ratio",
    "path_total_variation_R", "path_max_up_excursion_R",
    "path_max_down_excursion_R", "path_direction_change_rate",
    "path_last_return_R", "path_current_bar_range_R",
] + list(base.COMPACT_PROV)

LAG_COLS = [f"lag1_{c}" for c in LAG_BASE]
LAG_AVAIL = "lag1_available"
# K1A control: ONLY the availability indicator, never any lag VALUE.
K1A_EXTRA = [LAG_AVAIL]
# K2: availability indicator + the 15 previous-bar dynamic states.
K2_EXTRA = list(K1A_EXTRA) + LAG_COLS     # 16 new columns

# model tags / comparisons
MODEL_K0 = "K0_UNCONDITIONAL"       # P(Z)
MODEL_K1 = "K1_STATE"               # P(Z | S_t)
MODEL_K1A = "K1A_STATE_AVAIL"       # P(Z | S_t, A_t)          <-- control
MODEL_K2 = "K2_STATE_LAG1"          # P(Z | S_t, A_t, S_{t-1})

# PRIMARY lag-memory comparison. It isolates S_{t-1} from the episode-start
# representation effect carried by `lag1_available`.
PRIMARY_COMPARISON = "K2-K1A"
SECONDARY_COMPARISONS = ["K1A-K1", "K2-K1", "K1-K0"]

if LAG_BASE != list(base.LAG_BASE):
    raise SystemExit("STOP_DYNAMIC_PGM1A2_LAG_BASE_MISMATCH")
# K1A must contain the indicator only -- no lag values may leak into it.
if set(K1A_EXTRA) & set(LAG_COLS):
    raise SystemExit("STOP_DYNAMIC_PGM1A2_K1A_CONTAINS_LAG_VALUES")

# ===========================================================================
# attribution blocks (1A.2 splits Count OUT of LiquidityComposition)
# ===========================================================================
ATTRIB_NODE_BLOCKS = {
    "Location": ["z_d_up"],
    "Path": ["z_dmfe", "z_dmae", "z_dcr", "z_range"],
    "LiquidityResidual": ["z_uresid", "z_lresid"],
}
COUNT_OCCURRENCE_BLOCK = "CountOccurrence"

# output namespace (never overwrite 1A.1b)
PREFIX = "dynamic_pgm1a2"


# ===========================================================================
# lag1 construction
# ===========================================================================
def add_lag1_causal(df):
    """Add episode-internal lag1 of LAG_BASE plus `lag1_available`.

    Causality rules (all enforced):
      * groupby(episode_id).shift(1) -- never crosses episode boundaries;
      * no rolling, no forward fill;
      * `lag1_available` = previous row exists within the SAME episode;
      * hard audit: lag1 column == that same shift(1) exactly;
      * hard audit: the shifted row is the true PREVIOUS BAR (bar_t - 1), i.e.
        the filtered transition frame has no bar_t gaps inside an episode.
        (Verified on real data: 0 gaps out of 321,727 transitions.)
    """
    x = df.sort_values(["episode_id", "bar_t"], kind="stable").copy()
    g = x.groupby("episode_id", sort=False)
    prev_bar = g["bar_t"].shift(1)

    for c in LAG_BASE:
        x[f"lag1_{c}"] = g[c].shift(1)
    x[LAG_AVAIL] = prev_bar.notna().astype(np.int8)

    # ---- hard audit 1: lag1 is exactly the episode-internal previous row
    avail = x[LAG_AVAIL].eq(1)
    for c in LAG_BASE:
        a = x.loc[avail, f"lag1_{c}"].to_numpy(dtype=np.float64)
        b = g[c].shift(1).loc[avail].to_numpy(dtype=np.float64)
        if not np.array_equal(a, b, equal_nan=True):
            raise SystemExit(f"STOP_DYNAMIC_PGM1A2_LAG_CAUSALITY_FAIL:{c}")

    # ---- hard audit 2: previous row is the true previous BAR (no gaps)
    same_ep = g["episode_id"].shift(1).eq(x["episode_id"])
    gap = (prev_bar.notna() & same_ep
           & (x["bar_t"].to_numpy() != prev_bar.to_numpy() + 1))
    if int(gap.sum()) != 0:
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A2_LAG_BAR_GAP:{int(gap.sum())}")
    return x


def audit_lag_causality(cur):
    """Non-fatal lag audit report (counts + gap diagnostic)."""
    g = cur.groupby("episode_id", sort=False)
    prev_bar = g["bar_t"].shift(1)
    same_ep = g["episode_id"].shift(1).eq(cur["episode_id"])
    gap = (prev_bar.notna() & same_ep
           & (cur["bar_t"].to_numpy() != prev_bar.to_numpy() + 1))
    avail = cur[LAG_AVAIL].to_numpy()
    return dict(
        n_rows=int(len(cur)),
        n_episodes=int(cur["episode_id"].nunique()),
        n_lag_columns=len(LAG_COLS),
        lag_columns=LAG_COLS,
        lag1_available_column=LAG_AVAIL,
        n_lag1_available=int((avail == 1).sum()),
        n_lag1_missing=int((avail == 0).sum()),
        bar_t_gap_count=int(gap.sum()),
        cross_episode_leakage=bool(int((~same_ep & prev_bar.notna()).sum()) != 0),
        no_forward_fill=True,
        no_rolling=True,
    )


# ===========================================================================
# design matrix
# ===========================================================================
def _make_ct(num_cols):
    """ColumnTransformer with the SAME column ordering base uses for K1
    (categorical block first, then numeric) -- required for exact K1 parity."""
    return ColumnTransformer([
        ("cat", Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore"))]),
         list(base.OBS_STATE_CAT)),
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), list(num_cols)),
    ])


def _occ_nll(Y, p0):
    """Occurrence-only Hurdle-Poisson NLL (magnitude term excluded).

    Matches base._hurdle_nll's occurrence part exactly:
        y > 0 : -log(clip(p0, 1e-12, 1))
        y == 0: -log1p(-p0)
    """
    Y = np.asarray(Y, dtype=np.float64)
    p0 = np.asarray(p0, dtype=np.float64)
    out = np.zeros(p0.shape, dtype=np.float64)
    for j in range(p0.shape[1]):
        pos = Y[:, j] > 0
        out[:, j] = np.where(
            pos,
            -np.log(np.clip(p0[:, j], 1e-12, 1.0)),
            -np.log1p(-p0[:, j]))
    return out


def _node_eval_nll(nodes):
    return np.sum([nodes[n]["ev"] for n, _ in base.Z_LAYOUT], axis=0)


# ===========================================================================
# one window: K0 / K1 / K2
# ===========================================================================
def run_single_window_1a2(w, data_path):
    t_w = time.perf_counter()
    cur = pd.read_parquet(data_path)
    tr = cur[cur["block"].isin(w["train"])].reset_index(drop=True)
    ev = cur[cur["block"] == w["eval"]].reset_index(drop=True)
    if not len(tr) or not len(ev):
        raise SystemExit(f"STOP_DYNAMIC_PGM1A_EMPTY_SPLIT: {w['name']}")
    del cur
    print(f"[STAGE] window {w['name']} n_train={len(tr)} n_eval={len(ev)}",
          flush=True)

    Zc_tr = tr[base.ALL_Z_COLS].to_numpy(np.float32)
    Zc_ev = ev[base.ALL_Z_COLS].to_numpy(np.float32)
    yd_tr = tr[base.DISC_Z].to_numpy(np.int64)
    yd_ev = ev[base.DISC_Z].to_numpy(np.int64)
    Yc_tr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yc_ev = ev[base.COUNT_Z].to_numpy(np.int64)
    sym_ev = ev["symbol"].to_numpy()
    day_ev = ev["episode_start_day"].to_numpy()
    eid_ev = ev["episode_id"].to_numpy()

    model_metrics, opt_rows, block_rows, target_rows = [], [], [], []
    count_occ_rows, boots, bysym = [], [], []

    # ---------------- K0 ----------------
    t_k0 = time.perf_counter()
    k0 = base.fit_constant_heads(Zc_tr, Zc_ev, yd_tr, yd_ev)
    k0_count = base.fit_constant_count_head(Yc_tr, Yc_ev)
    opt_rows.append(dict(window=w["name"], model="K0_UNCONDITIONAL",
                         n_params=k0["n_params"] + k0_count["n_params"],
                         success=True,
                         elapsed_seconds=round(time.perf_counter() - t_k0, 3)))

    # ------------- K1 / K1A / K2 design matrices (INDEPENDENT transformers) --
    #   X1  = OBSERVED_STATE
    #   X1A = OBSERVED_STATE + lag1_available                (control)
    #   X2  = OBSERVED_STATE + lag1_available + 15 lag vars
    obs_num = list(base.OBS_STATE_NUM)
    num1 = obs_num
    num1a = obs_num + list(K1A_EXTRA)
    num2 = obs_num + list(K2_EXTRA)

    t_k1 = time.perf_counter()
    ct1 = _make_ct(num1)
    Xtr1 = ct1.fit_transform(tr).astype(np.float32)
    Xev1 = ct1.transform(ev).astype(np.float32)
    t_k1a = time.perf_counter()
    ct1a = _make_ct(num1a)
    Xtr1a = ct1a.fit_transform(tr).astype(np.float32)
    Xev1a = ct1a.transform(ev).astype(np.float32)
    t_k2 = time.perf_counter()
    ct2 = _make_ct(num2)
    Xtr2 = ct2.fit_transform(tr).astype(np.float32)
    Xev2 = ct2.transform(ev).astype(np.float32)

    # K1 / K1A / K2 MUST see identical rows
    if not (Xev1.shape[0] == Xev1a.shape[0] == Xev2.shape[0] == len(ev)):
        raise SystemExit("STOP_DYNAMIC_PGM1A2_EVAL_ROW_MISMATCH")
    # nested design dimensions
    if not (Xtr1a.shape[1] == Xtr1.shape[1] + len(K1A_EXTRA)
            and Xtr2.shape[1] == Xtr1a.shape[1] + len(LAG_COLS)):
        raise SystemExit("STOP_DYNAMIC_PGM1A2_NESTED_DESIGN_MISMATCH")

    # ---------------- K1 ----------------
    k1 = base.fit_state_heads(Xtr1, Xev1, Zc_tr, Zc_ev, yd_tr, yd_ev)
    k1_count = base.fit_state_count_head(
        Xtr1, Yc_tr, Xev1, Yc_ev,
        constant_rates=k0_count["constant_rates"])
    opt_rows.append(dict(window=w["name"], model=MODEL_K1,
                         n_params=k1["n_params"] + k1_count["n_params"],
                         success=True,
                         elapsed_seconds=round(time.perf_counter() - t_k1, 3)))

    # ---------------- K1A (control: availability indicator only) -------------
    k1a = base.fit_state_heads(Xtr1a, Xev1a, Zc_tr, Zc_ev, yd_tr, yd_ev)
    k1a_count = base.fit_state_count_head(
        Xtr1a, Yc_tr, Xev1a, Yc_ev,
        constant_rates=k0_count["constant_rates"])
    opt_rows.append(dict(window=w["name"], model=MODEL_K1A,
                         n_params=k1a["n_params"] + k1a_count["n_params"],
                         success=True,
                         elapsed_seconds=round(time.perf_counter() - t_k1a, 3)))

    # ---------------- K2 ----------------
    k2 = base.fit_state_heads(Xtr2, Xev2, Zc_tr, Zc_ev, yd_tr, yd_ev)
    k2_count = base.fit_state_count_head(
        Xtr2, Yc_tr, Xev2, Yc_ev,
        constant_rates=k0_count["constant_rates"])
    opt_rows.append(dict(window=w["name"], model=MODEL_K2,
                         n_params=k2["n_params"] + k2_count["n_params"],
                         success=True,
                         elapsed_seconds=round(time.perf_counter() - t_k2, 3)))

    # ---------------- joint NLL ----------------
    def _joint(k, kc):
        cont = _node_eval_nll(k["nodes"])
        disc = k["disc_ev"]
        cnt = kc["nll_ev"].sum(axis=1)
        return cont, disc, cnt, cont + disc + cnt

    (cont0, disc0, cnt0, j0) = _joint(k0, k0_count)
    (cont1, disc1, cnt1, j1) = _joint(k1, k1_count)
    (cont1a, disc1a, cnt1a, j1a) = _joint(k1a, k1a_count)
    (cont2, disc2, cnt2, j2) = _joint(k2, k2_count)

    for tag, cont, disc, cnt, j in [
            (MODEL_K0, cont0, disc0, cnt0, j0),
            (MODEL_K1, cont1, disc1, cnt1, j1),
            (MODEL_K1A, cont1a, disc1a, cnt1a, j1a),
            (MODEL_K2, cont2, disc2, cnt2, j2)]:
        m_c, m_d, m_n, m_j = (float(np.mean(cont)), float(np.mean(disc)),
                              float(np.mean(cnt)), float(np.mean(j)))
        assert abs(m_j - m_c - m_d - m_n) < 1e-10
        assert np.all(np.isfinite(j))
        model_metrics.append(dict(window=w["name"], model=tag,
                                  mean_joint_nll=m_j, mean_cont_nll=m_c,
                                  mean_disc_nll=m_d, mean_count_nll=m_n,
                                  n_rows=int(len(ev))))

    # ---------------- Count Magnitude Closure hard assertions ----------------
    # lambda_K0 == lambda_K1 == lambda_K1A == lambda_K2 (train-only constant)
    for j in range(len(base.COUNT_Z)):
        r0 = k0_count["rate_ev"][:, j]
        for other in (k1_count["rate_ev"][:, j], k1a_count["rate_ev"][:, j],
                      k2_count["rate_ev"][:, j]):
            if not np.array_equal(r0, other):
                raise SystemExit(
                    "STOP_DYNAMIC_PGM1A2_COUNT_MAGNITUDE_NOT_CANCELLED")

    occ0 = _occ_nll(Yc_ev, k0_count["p0_ev"])
    occ1 = _occ_nll(Yc_ev, k1_count["p0_ev"])
    occ1a = _occ_nll(Yc_ev, k1a_count["p0_ev"])
    occ2 = _occ_nll(Yc_ev, k2_count["p0_ev"])

    # PRIMARY: K2 - K1A count delta must come from occurrence only
    d_count_21a = cnt2 - cnt1a
    d_occ_21a = occ2.sum(axis=1) - occ1a.sum(axis=1)
    max_cancel_err = float(np.max(np.abs(d_count_21a - d_occ_21a)))
    if max_cancel_err >= 1e-10:
        raise SystemExit(
            "STOP_DYNAMIC_PGM1A2_COUNT_MAGNITUDE_NOT_CANCELLED:"
            f"{max_cancel_err}")

    # ---------------- deltas + bootstrap ----------------
    # PRIMARY   : K2-K1A  (isolates S_{t-1} from the availability indicator)
    # SECONDARY : K1A-K1 (episode-start representation effect),
    #             K2-K1  (total lag-block gain), K1-K0 (state baseline)
    delta10 = j1 - j0            # K1  - K0
    delta_a1 = j1a - j1          # K1A - K1
    delta_21a = j2 - j1a         # K2  - K1A   <-- PRIMARY
    delta21 = j2 - j1            # K2  - K1    <-- secondary (total)

    for label, d in (("K1-K0", delta10), ("K1A-K1", delta_a1),
                     ("K2-K1A", delta_21a), ("K2-K1", delta21)):
        lo, hi, point = base.boot_cluster(d, day_ev, eid_ev, w["seed"])
        boots.append(dict(window=w["name"], comparison=label,
                          delta_sample_mean=point, ci_lo=lo, ci_hi=hi,
                          verdict=("CI_below_zero" if hi < 0
                                   else "CI_above_zero" if lo > 0
                                   else "CI_contains_zero")))

        df_d = pd.DataFrame({"symbol": sym_ev, "eid": eid_ev, "delta": d})
        ep_d = df_d.groupby(["symbol", "eid"], as_index=False)["delta"].mean()
        for sym, grp in ep_d.groupby("symbol"):
            raw = df_d[df_d["symbol"] == sym]
            d_raw = raw["delta"].to_numpy()
            m_ep = float(grp["delta"].mean())
            bysym.append(dict(window=w["name"], comparison=label, symbol=sym,
                              n_episodes=int(len(grp)), n_rows=int(len(raw)),
                              mean_delta_episode=m_ep,
                              mean_delta_row=float(d_raw.mean()),
                              mean_delta=m_ep,
                              n_negative=int((d_raw < 0).sum()),
                              n_positive=int((d_raw > 0).sum())))

    # ---------------- block attribution (PRIMARY column: K2 - K1A) -----------
    def _blk(k, nodes):
        return np.sum([k["nodes"][n]["ev"] for n in nodes], axis=0)

    for bname, nodes in ATTRIB_NODE_BLOCKS.items():
        s0, s1 = _blk(k0, nodes), _blk(k1, nodes)
        s1a, s2 = _blk(k1a, nodes), _blk(k2, nodes)
        block_rows.append(dict(
            window=w["name"], block=bname,
            mean_nll_k0=float(np.mean(s0)), mean_nll_k1=float(np.mean(s1)),
            mean_nll_k1a=float(np.mean(s1a)), mean_nll_k2=float(np.mean(s2)),
            delta_k2_minus_k1a=float(np.mean(s2 - s1a)),
            delta_k1a_minus_k1=float(np.mean(s1a - s1)),
            delta_k2_minus_k1=float(np.mean(s2 - s1)),
            delta_k1_minus_k0=float(np.mean(s1 - s0))))
    o0, o1 = occ0.sum(axis=1), occ1.sum(axis=1)
    o1a, o2 = occ1a.sum(axis=1), occ2.sum(axis=1)
    block_rows.append(dict(
        window=w["name"], block=COUNT_OCCURRENCE_BLOCK,
        mean_nll_k0=float(np.mean(o0)), mean_nll_k1=float(np.mean(o1)),
        mean_nll_k1a=float(np.mean(o1a)), mean_nll_k2=float(np.mean(o2)),
        delta_k2_minus_k1a=float(np.mean(o2 - o1a)),
        delta_k1a_minus_k1=float(np.mean(o1a - o1)),
        delta_k2_minus_k1=float(np.mean(o2 - o1)),
        delta_k1_minus_k0=float(np.mean(o1 - o0))))

    # ---------------- per-target ----------------
    for nm, _cols in base.Z_LAYOUT:
        n0 = k0["nodes"][nm]["ev"]
        n1 = k1["nodes"][nm]["ev"]
        n1a = k1a["nodes"][nm]["ev"]
        n2 = k2["nodes"][nm]["ev"]
        target_rows.append(dict(
            window=w["name"], target=nm, kind="node",
            mean_nll_k0=float(np.mean(n0)), mean_nll_k1=float(np.mean(n1)),
            mean_nll_k1a=float(np.mean(n1a)), mean_nll_k2=float(np.mean(n2)),
            delta_k1_minus_k0=float(np.mean(n1 - n0)),
            delta_k1a_minus_k1=float(np.mean(n1a - n1)),
            delta_k2_minus_k1a=float(np.mean(n2 - n1a)),
            delta_k2_minus_k1=float(np.mean(n2 - n1))))
    for j, c in enumerate(base.COUNT_Z):
        n0 = k0_count["nll_ev"][:, j]
        n1 = k1_count["nll_ev"][:, j]
        n1a = k1a_count["nll_ev"][:, j]
        n2 = k2_count["nll_ev"][:, j]
        target_rows.append(dict(
            window=w["name"], target=c, kind="discrete_count",
            mean_nll_k0=float(np.mean(n0)), mean_nll_k1=float(np.mean(n1)),
            mean_nll_k1a=float(np.mean(n1a)), mean_nll_k2=float(np.mean(n2)),
            delta_k1_minus_k0=float(np.mean(n1 - n0)),
            delta_k1a_minus_k1=float(np.mean(n1a - n1)),
            delta_k2_minus_k1a=float(np.mean(n2 - n1a)),
            delta_k2_minus_k1=float(np.mean(n2 - n1))))
        # count occurrence diagnostics (+ closure evidence)
        ye = Yc_ev[:, j]
        ze = (ye > 0).astype(int)
        p1 = k1_count["p0_ev"][:, j]
        p1a = k1a_count["p0_ev"][:, j]
        p2 = k2_count["p0_ev"][:, j]
        count_occ_rows.append(dict(
            window=w["name"], target=c,
            constant_ztp_lambda=float(k0_count["constant_rates"][j]),
            state_dependent_magnitude=False,
            k1_occ_logloss=float(log_loss(ze, np.clip(p1, 1e-6, 1 - 1e-6),
                                          labels=[0, 1])),
            k1a_occ_logloss=float(log_loss(ze, np.clip(p1a, 1e-6, 1 - 1e-6),
                                           labels=[0, 1])),
            k2_occ_logloss=float(log_loss(ze, np.clip(p2, 1e-6, 1 - 1e-6),
                                          labels=[0, 1])),
            k1_occ_brier=float(brier_score_loss(ze, p1)),
            k1a_occ_brier=float(brier_score_loss(ze, p1a)),
            k2_occ_brier=float(brier_score_loss(ze, p2)),
            k1_occ_pr_auc=(float(average_precision_score(ze, p1))
                           if len(np.unique(ze)) > 1 else float("nan")),
            k1a_occ_pr_auc=(float(average_precision_score(ze, p1a))
                            if len(np.unique(ze)) > 1 else float("nan")),
            k2_occ_pr_auc=(float(average_precision_score(ze, p2))
                           if len(np.unique(ze)) > 1 else float("nan")),
            mean_occ_nll_k1=float(np.mean(occ1[:, j])),
            mean_occ_nll_k1a=float(np.mean(occ1a[:, j])),
            mean_occ_nll_k2=float(np.mean(occ2[:, j])),
            delta_occ_nll_k2_minus_k1a=float(np.mean(occ2[:, j] - occ1a[:, j])),
            delta_count_nll_k2_minus_k1a=float(np.mean(n2 - n1a)),
            delta_occ_nll_k1a_minus_k1=float(np.mean(occ1a[:, j] - occ1[:, j])),
            delta_occ_nll_k2_minus_k1=float(np.mean(occ2[:, j] - occ1[:, j])),
            delta_count_nll_k2_minus_k1=float(np.mean(n2 - n1)),
            count_minus_occurrence_residual=float(np.max(np.abs(
                (n2 - n1a) - (occ2[:, j] - occ1a[:, j])))),
        ))

    # ---------------- covariance PD audit ----------------
    def _min_eig(h):
        for cand in (h, getattr(h, "g", None)):
            if cand is not None and hasattr(cand, "cov"):
                return float(np.linalg.eigvalsh(
                    np.asarray(cand.cov, dtype=np.float64)).min())
        return None

    cov_audit = {w["name"]: {}}
    for nm, d in k2["nodes"].items():
        e = _min_eig(d["head"])
        cov_audit[w["name"]][nm] = dict(
            min_eig=(e if e is not None else "n/a"),
            pd=bool(e is not None and e > 0))

    del (Xtr1, Xev1, Xtr1a, Xev1a, Xtr2, Xev2, k1, k1a, k2, k0, tr, ev)
    print(f"[STAGE] window {w['name']} done "
          f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)
    return dict(window=w["name"], model_metrics=model_metrics,
                opt_rows=opt_rows, boots=boots, bysym=bysym,
                block_rows=block_rows, target_rows=target_rows,
                count_occ_rows=count_occ_rows, cov_audit=cov_audit,
                n_symbols=int(len(pd.unique(sym_ev))))


# ===========================================================================
# main
# ===========================================================================
def main():
    t_total = time.perf_counter()
    timing = {}

    t0 = time.perf_counter()
    keep_cols = (list(base.OBS_STATE_NUM) + list(base.OBS_STATE_CAT)
                 + list(LAG_BASE) + base.BUILD_SRC_COLS
                 + ["hazard", "block", "symbol", "episode_id",
                    "episode_start_day", "bar_t"])
    keep_cols = list(dict.fromkeys(keep_cols))
    df = pd.read_parquet(base.CACHE / "market_state1_samples.parquet",
                         columns=keep_cols)
    build_set = set(base.BUILD_SRC_COLS)
    for c in df.columns:
        if c not in build_set and str(df[c].dtype).startswith("float64"):
            df[c] = df[c].astype(np.float32)
    cur, nxt = base.build_transition_sample(df)
    base._peak("after build_transition_sample")
    cur = add_lag1_causal(cur)
    base._peak("after add_lag1_causal")
    del df
    timing["build_transition_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[STAGE] build done n={len(cur)}", flush=True)

    # ---------------- sample parity (rows + z_feature_hash) ----------------
    hash_cols = base.ALL_Z_COLS + [base.DISC_Z] + base.COUNT_Z
    h = hashlib.sha256(
        pd.util.hash_pandas_object(cur[hash_cols], index=False)
        .values.tobytes()).hexdigest()
    sample_audit = dict(
        n_transition_rows=int(len(cur)),
        expected_transitions=EXPECTED_TRANSITIONS,
        count_ok=bool(len(cur) == EXPECTED_TRANSITIONS),
        z_feature_hash=h,
        blocks={b: int((cur["block"] == b).sum())
                for b in ["TB1", "TB2", "TB3"]},
        n_episodes=int(cur["episode_id"].nunique()),
    )
    lag_audit = audit_lag_causality(cur)
    parity_audit = dict(
        parent_commit=BASE_SHA,
        rows_ok=bool(len(cur) == EXPECTED_TRANSITIONS),
        z_feature_hash_ok=bool(h == FROZEN_Z_FEATURE_HASH),
        z_feature_hash_expected=FROZEN_Z_FEATURE_HASH,
        z_feature_hash_actual=h,
        joint_nll_parity={},
        joint_nll_tol=PARITY_TOL,
    )
    if not (parity_audit["rows_ok"] and parity_audit["z_feature_hash_ok"]):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A2_SAMPLE_PARITY_FAIL: {parity_audit}")

    # ---------------- invariants (frozen from 1A.1b) ----------------
    invariants = dict(
        tv_update_exact=bool(np.allclose(
            (nxt["path_total_variation_R"].to_numpy(float)
             - cur["path_total_variation_R"].to_numpy(float)),
            np.abs(cur["z_d_up"].to_numpy(float)), atol=1e-8)),
        last_return_exact=bool(np.allclose(
            nxt["path_last_return_R"].to_numpy(float),
            -cur["z_d_up"].to_numpy(float), atol=1e-8)),
        no_tb4=bool("TB4" not in set(cur["block"].unique())),
        lag1_from_past_only=bool(
            (cur.loc[cur[LAG_AVAIL] == 0, LAG_COLS].isna().all(axis=None))
            and (cur.loc[cur[LAG_AVAIL] == 1, LAG_COLS].notna().any(axis=1)
                 .all())),
    )
    if not all(invariants.values()):
        raise SystemExit(f"STOP_DYNAMIC_PGM1A_INVARIANT_FAIL: {invariants}")

    # ---------------- support + reconstruction (reuse 1A.1b) ----------------
    support_audit = dict(
        upper_residual=base.audit_support_residual(
            nxt["upper_newest_log_age_residual"].to_numpy(float),
            name="upper_residual"),
        lower_residual=base.audit_support_residual(
            nxt["lower_newest_log_age_residual"].to_numpy(float),
            name="lower_residual"),
        range=base.audit_support_range(
            nxt["path_current_bar_range_R"].to_numpy(float), name="range"),
    )
    full_recon = base.audit_full_reconstruction(cur, cur, nxt)
    agezero_audit = base.audit_agezero_reconstruction(cur, cur, nxt)
    if (full_recon["state_max_error"] >= 1e-8
            or full_recon["derived_representation_max_error"] >= 1e-8
            or full_recon["discrete_mismatch"] != 0):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A1_STATE_RECONSTRUCTION_FAIL: {full_recon}")
    if agezero_audit["upper"]["n_mismatch"] or agezero_audit["lower"]["n_mismatch"]:
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A1_AGEZERO_RECON_MISMATCH: {agezero_audit}")
    base.configure_child_semantics(agezero_deterministic=True)
    if base.disc_spec() != []:
        raise SystemExit("STOP_DYNAMIC_PGM1A1_DISC_SPEC_NONEMPTY")

    (base.OUT / f"{PREFIX}_lag_audit.json").write_text(
        json.dumps(dict(lag=lag_audit, sample=sample_audit,
                        support=support_audit,
                        reconstruction=full_recon), indent=2, default=str))
    print(f"[AUDIT] lag={lag_audit['n_lag1_available']}/"
          f"{lag_audit['n_rows']} available, bar_t_gap="
          f"{lag_audit['bar_t_gap_count']}", flush=True)

    del nxt

    if os.environ.get("DYNAMIC_PGM1A2_AUDIT_ONLY") == "1":
        (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
            json.dumps(parity_audit, indent=2, default=str))
        print("[AUDIT-ONLY] Complete. Stopping before any model fit.",
              flush=True)
        return dict(parity=parity_audit, lag=lag_audit, sample=sample_audit)

    # ---------------- persist transition cache (WITH lag1 columns) ----------
    needed_cols = (list(base.OBS_STATE_NUM) + list(base.OBS_STATE_CAT)
                   + base.ALL_Z_COLS + [base.DISC_Z] + base.COUNT_Z
                   + list(K2_EXTRA)
                   + ["episode_id", "symbol", "block", "episode_start_day"])
    for c in cur.columns:
        if str(cur[c].dtype).startswith("float64"):
            cur[c] = cur[c].astype(np.float32)
    cur = cur[list(dict.fromkeys(needed_cols))].copy()
    data_path = base.CACHE / f"{PREFIX}_transitions.parquet"
    cur.to_parquet(data_path, index=False)
    del cur
    base._peak("after slim cur")

    # ---------------- windows (subprocess isolation) ----------------
    win_env = dict(os.environ)
    for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        win_env[_bt] = "1"

    model_metrics, opt_rows, boots, bysym = [], [], [], []
    block_rows, target_rows, count_occ_rows = [], [], []
    cov_audit, n_symbols = {}, 0
    for w in base.WINDOWS:
        result_path = base.CACHE / f"_{PREFIX}_window_{w['name']}.json"
        cmd = [sys.executable, str(Path(__file__)),
               "--window-json", json.dumps(w),
               "--data", str(data_path), "--result", str(result_path),
               "--agezero-deterministic"]
        print(f"[STAGE] window {w['name']} subprocess start", flush=True)
        t_w = time.perf_counter()
        r = subprocess.run(cmd, env=win_env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout, file=sys.stderr)
            print(r.stderr, file=sys.stderr)
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A_WINDOW_FAIL: {w['name']} rc={r.returncode}")
        res = json.loads(Path(result_path).read_text())
        model_metrics += res["model_metrics"]
        opt_rows += res["opt_rows"]
        boots += res["boots"]
        bysym += res["bysym"]
        block_rows += res["block_rows"]
        target_rows += res["target_rows"]
        count_occ_rows += res.get("count_occ_rows", [])
        cov_audit.update(res["cov_audit"])
        n_symbols = max(n_symbols, int(res.get("n_symbols", 0)))
        print(f"[STAGE] window {w['name']} subprocess done "
              f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)

    # ---------------- K0 / K1 joint-NLL parity gate ----------------
    for w in base.WINDOWS:
        got = {m["model"]: m["mean_joint_nll"] for m in model_metrics
               if m["window"] == w["name"]}
        froz = FROZEN_JOINT[w["name"]]
        d0 = abs(got["K0_UNCONDITIONAL"] - froz["K0_UNCONDITIONAL"])
        d1 = abs(got["K1_STATE"] - froz["K1_STATE"])
        parity_audit["joint_nll_parity"][w["name"]] = dict(
            k0_expected=froz["K0_UNCONDITIONAL"], k0_actual=got["K0_UNCONDITIONAL"],
            k0_abs_diff=d0, k1_expected=froz["K1_STATE"],
            k1_actual=got["K1_STATE"], k1_abs_diff=d1,
            ok=bool(d0 < PARITY_TOL and d1 < PARITY_TOL))
        if d0 >= PARITY_TOL or d1 >= PARITY_TOL:
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A2_K1_PARITY_FAIL: {w['name']} "
                f"dK0={d0} dK1={d1}")
    parity_audit["joint_nll_parity_ok"] = True

    # ---------------- verdict ----------------
    def _row(wname, label):
        return next(b for b in boots if b["window"] == wname
                    and b["comparison"] == label)

    def _sym_neg(wname, label):
        sub = [b for b in bysym if b["window"] == wname
               and b["comparison"] == label]
        return int(sum(1 for b in sub if b["mean_delta_episode"] < 0))

    windows_report = {}
    for w in base.WINDOWS:
        # ---- PRIMARY: K2 - K1A. This is the ONLY comparison the lag verdict
        # may read; it isolates S_{t-1} from the `lag1_available` indicator.
        r_21a = _row(w["name"], PRIMARY_COMPARISON)
        n_neg = _sym_neg(w["name"], PRIMARY_COMPARISON)
        # ---- secondary (reported only)
        r10 = _row(w["name"], "K1-K0")
        r_a1 = _row(w["name"], "K1A-K1")
        r21 = _row(w["name"], "K2-K1")
        denom = abs(r10["delta_sample_mean"])
        ratio = (abs(r_21a["delta_sample_mean"]) / denom
                 if denom != 0 else float("nan"))
        total_ratio = (abs(r21["delta_sample_mean"]) / denom
                       if denom != 0 else float("nan"))
        windows_report[w["name"]] = dict(
            delta_k2_minus_k1a=r_21a["delta_sample_mean"],
            ci_lo=r_21a["ci_lo"], ci_hi=r_21a["ci_hi"],
            n_symbols_negative=n_neg, n_symbols_total=n_symbols,
            delta_k1_minus_k0=r10["delta_sample_mean"],
            increment_ratio=ratio,
            total_increment_ratio=total_ratio,
            secondary=dict(
                delta_k1a_minus_k1=r_a1["delta_sample_mean"],
                k1a_minus_k1_ci_lo=r_a1["ci_lo"],
                k1a_minus_k1_ci_hi=r_a1["ci_hi"],
                delta_k2_minus_k1=r21["delta_sample_mean"],
                k2_minus_k1_ci_lo=r21["ci_lo"],
                k2_minus_k1_ci_hi=r21["ci_hi"]),
            passes=bool(r_21a["ci_hi"] < 0 and n_neg >= SYMBOL_BREADTH_MIN))
    n_pass = sum(1 for v in windows_report.values() if v["passes"])
    if n_pass == len(base.WINDOWS):
        verdict = "LAG1_RESIDUAL_SUPPORTED"
    elif n_pass == 0:
        verdict = "NO_STABLE_LAG1_INCREMENT_DETECTED"
    else:
        verdict = "LAG1_INCREMENT_NOT_STABLE"

    # ---------------- outputs ----------------
    pd.DataFrame(model_metrics).to_csv(
        base.OUT / f"{PREFIX}_model_metrics.csv", index=False)
    pd.DataFrame(boots).to_csv(
        base.OUT / f"{PREFIX}_bootstrap.csv", index=False)
    pd.DataFrame(bysym).to_csv(
        base.OUT / f"{PREFIX}_by_symbol.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        base.OUT / f"{PREFIX}_target_metrics.csv", index=False)
    pd.DataFrame(block_rows).to_csv(
        base.OUT / f"{PREFIX}_lag_block_attribution.csv", index=False)
    pd.DataFrame(count_occ_rows).to_csv(
        base.OUT / f"{PREFIX}_count_occurrence.csv", index=False)
    pd.DataFrame(opt_rows).to_csv(
        base.OUT / f"{PREFIX}_optimizer_audit.csv", index=False)
    (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
        json.dumps(parity_audit, indent=2, default=str))

    summary = dict(
        experiment="DYNAMIC-PGM-1A.2 Lag Closure",
        parent_commit=BASE_SHA,
        grandparent_commit=PARENT_SHA,
        question=("Given S_t, does S_{t-1} add stable extra information "
                  "for Z_{t+1}?"),
        verdict=verdict,
        primary_comparison=PRIMARY_COMPARISON,
        secondary_comparisons=SECONDARY_COMPARISONS,
        models=dict(
            K0_UNCONDITIONAL="P(Z)",
            K1_STATE="P(Z | S_t)",
            K1A_STATE_AVAIL="P(Z | S_t, A_t)  [control: availability only]",
            K2_STATE_LAG1="P(Z | S_t, A_t, S_{t-1})"),
        windows=windows_report,
        symbol_breadth_min=SYMBOL_BREADTH_MIN,
        lag_block=dict(n_lag_columns=len(LAG_COLS), columns=LAG_COLS,
                       availability_column=LAG_AVAIL,
                       k1a_extra=K1A_EXTRA, k2_extra=K2_EXTRA,
                       excluded=["start_geometry", "previous_endpoint",
                                 "episode_context", "TEMPO", "lag2",
                                 "static_context"]),
        confound_control=dict(
            issue=("lag1_available is essentially an 'episode first bar' "
                   "indicator; as a 0/1 column it also acts as an extra "
                   "nonlinear basis function for the linear heads."),
            control_model="K1A_STATE_AVAIL = OBSERVED_STATE + lag1_available",
            resolution=("K1A-K1 measures the episode-start representation "
                        "effect; K2-K1A isolates genuine S_{t-1} memory and "
                        "is the ONLY input to the lag verdict.")),
        count_magnitude_closure=dict(
            magnitude=("state-independent train-constant exact ZTP shared "
                       "by K0/K1/K1A/K2"),
            occurrence="state-dependent Logistic (may differ across K1/K1A/K2)",
            k2_minus_k1a_count_gain_source="occurrence only"),
        model_metrics=model_metrics,
        bootstrap=boots,
        block_attribution=block_rows,
        per_target=target_rows,
        covariance_audit=cov_audit,
        sample=sample_audit,
        parity=parity_audit,
        bootstrap_reps=base.BOOTSTRAP_REPS,
        frozen_scope=["no latent/HMM/HSMM", "no lag2/lag3",
                      "no alpha/C tuning", "no SMC/OB",
                      "no terminal/reset", "no rollout", "no PnL/RL"],
    )
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (base.OUT / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[MODEL METRICS]\n{pd.DataFrame(model_metrics).to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{pd.DataFrame(boots).to_string(index=False)}")
    print(f"[BLOCK ATTRIBUTION]\n{pd.DataFrame(block_rows).to_string(index=False)}")
    print(f"[VERDICT] {verdict}")
    print("[PRIMARY K2-K1A] " + ", ".join(
        f"{k}: {v['delta_k2_minus_k1a']:.4f} "
        f"CI=[{v['ci_lo']:.4f},{v['ci_hi']:.4f}] "
        f"neg={v['n_symbols_negative']}/{v['n_symbols_total']}"
        for k, v in windows_report.items()))
    print("[SECONDARY] " + ", ".join(
        f"{k}: K1A-K1={v['secondary']['delta_k1a_minus_k1']:.4f} "
        f"K2-K1={v['secondary']['delta_k2_minus_k1']:.4f} "
        f"K1-K0={v['delta_k1_minus_k0']:.4f}"
        for k, v in windows_report.items()))
    print("[INCREMENT RATIO] " + ", ".join(
        f"{k}: {v['increment_ratio']:.4f} "
        f"(total {v['total_increment_ratio']:.4f})"
        for k, v in windows_report.items()))
    print(f"[DONE] -> {base.OUT}")
    return summary


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--window-json")
    _ap.add_argument("--data")
    _ap.add_argument("--result")
    _ap.add_argument("--agezero-deterministic", action="store_true",
                     help="Child mode: age-zero is a deterministic derived "
                          "state, drop the 4-class discrete node.")
    _ap.add_argument("--audit-only", action="store_true",
                     help="Run real-data audits (sample parity, lag "
                          "causality, reconstruction) and stop; no model fit.")
    _a = _ap.parse_args()
    if _a.audit_only:
        os.environ["DYNAMIC_PGM1A2_AUDIT_ONLY"] = "1"
    if _a.window_json:
        if not _a.agezero_deterministic:
            raise SystemExit(
                "STOP_DYNAMIC_PGM1A2_CHILD_AGEZERO_STATE_NOT_PROPAGATED")
        base.configure_child_semantics(agezero_deterministic=True)
        if base.disc_spec() != []:
            raise SystemExit("STOP_DYNAMIC_PGM1A2_CHILD_DISC_SPEC_NONEMPTY")
        _w = json.loads(_a.window_json)
        Path(_a.result).write_text(
            json.dumps(run_single_window_1a2(_w, _a.data), default=str))
    else:
        main()
