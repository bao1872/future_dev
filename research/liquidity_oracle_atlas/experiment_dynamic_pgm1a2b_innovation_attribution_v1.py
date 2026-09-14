"""DYNAMIC-PGM-1A.2b -- Innovation Attribution.

Context (1A.2, frozen at b12778e1159e322fc9f2de6937a00eb0b1bf6260)
-----------------------------------------------------------------
1A.2 established LAG1_RESIDUAL_SUPPORTED:

    K2 - K1A  =  -0.06333 (A, CI [-0.06847, -0.05854])
              =  -0.06319 (B, CI [-0.06838, -0.05826])
    15/15 symbols negative in both windows.

and the gain is ~97-98% concentrated in the Path block (row-weighted:
Path -0.0729 of total -0.0741 in A; -0.0687 of -0.0710 in B).

But that does NOT mean the world model must carry the full previous state.

Question
--------
1A.2b asks a compression question, not a feature-importance question:

    Is the information carried by S_{t-1} really just the most recent
    innovation Z_t?

Because S_t = F(S_{t-1}, Z_t), the two competing hypotheses are

    H_compress : P(Z_{t+1} | S_t, A_t, Z_t)      absorbs the whole lag gain
    H_full     : P(Z_{t+1} | S_t, A_t, S_{t-1})  retains residual info beyond Z_t

Models
------
    M0  = P(Z_{t+1} | S_t, A_t)                 frozen 1A.2 K1A   (baseline)
    MF  = P(Z_{t+1} | S_t, A_t, S_{t-1})        frozen 1A.2 K2    (full lag)
    MZ  = P(Z_{t+1} | S_t, A_t, Z_t)            NEW               (innovation)

K0 and K1 are re-fitted only as parity references / ratio denominators.

Gates
-----
    GATE 1  MZ - M0      : does Z_t itself carry residual information?
    GATE 2  MF - MZ      : after knowing Z_t, does the full S_{t-1} still help?

    GATE 1 pass and GATE 2 not stable  -> LAG1_MEMORY_COMPRESSIBLE_TO_INNOVATION
    GATE 1 pass and GATE 2 stable <0   -> FULL_LAG_STATE_RETAINS_RESIDUAL_INFORMATION
    GATE 1 fails                       -> INNOVATION_ATTRIBUTION_NOT_SUPPORTED

    R = |MZ-M0| / |MF-M0|  (retention) is EXPLANATORY ONLY, never a gate.

Z_t definition (FULL stochastic innovation)
-------------------------------------------
In the transition frame, row r's encoded innovation columns ARE Z_{r+1}.
Therefore Z_t for row r is the previous row's encoded innovation vector; with
bar_t gaps proven == 0 that previous row is exactly bar_t - 1. Episode-first
rows have no Z_t (NaN -> train-median imputed, flagged by `lag1_available`).

Z_t must cover EVERY stochastic node of the 1A.1b kernel:

    7 support-correct nodes   -> 14 encoded columns (base.ALL_Z_COLS)
    2 hurdle-Poisson counts   ->  2 encoded columns (base.COUNT_Z)
    -------------------------------------------------------------
    9 native stochastic nodes -> 16 encoded predictor columns

Testing only the 14 continuous columns would silently evaluate
Z_t^{continuous} instead of the pre-registered Z_t^{full}.

On "compression"
----------------
The compression being tested is SEMANTIC, not a raw column-count comparison:

    full previous-state memory  S_{t-1}   (15 history state variables)
      ->  most recent transition innovation  Z_t   (9 native stochastic nodes)

The design matrices are MF = +16 columns vs MZ = +17 columns, because the
9 native innovations are expanded to 16 encoded columns to reuse the existing
support representation. NO compactness claim is made from column counts; the
structural simplification is that 15 remembered state variables are replaced
by the 9 native innovations that generated the current state.

Frozen scope
------------
No lag2/lag3, no latent/HMM/HSMM, no alpha/C tuning, no SMC/OB,
no terminal/reset, no rollout, no PnL/RL, no new feature mining.
Count Magnitude Closure is inherited unchanged (one shared train-only
constant ZTP lambda across every model).
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

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2_lag_closure_v1 as lag  # noqa: E402

# ===========================================================================
# frozen references
# ===========================================================================
BASE_SHA = "b12778e1159e322fc9f2de6937a00eb0b1bf6260"   # 1A.2 result commit
PARENT_1A1B_SHA = "16ab731ac099ffcec7c2089640b95b5555184853"
EXPECTED_TRANSITIONS = base.EXPECTED_TRANSITIONS
FROZEN_Z_FEATURE_HASH = lag.FROZEN_Z_FEATURE_HASH
# frozen 1A.1b (grandparent) K0 / K1
FROZEN_1A1B_JOINT = lag.FROZEN_JOINT
# frozen 1A.2 (parent) M0 == K1A_STATE_AVAIL, MF == K2_STATE_LAG1
FROZEN_1A2_JOINT = {
    "A_TB1_to_TB2": dict(K1A_STATE_AVAIL=1.3738896895068666,
                         K2_STATE_LAG1=1.299807223748625),
    "B_TB1TB2_to_TB3": dict(K1A_STATE_AVAIL=1.4054025921992275,
                            K2_STATE_LAG1=1.3344161873975653),
}
PARITY_TOL = 1e-8
SYMBOL_BREADTH_MIN = lag.SYMBOL_BREADTH_MIN

# ===========================================================================
# model definitions
# ===========================================================================
MODEL_K0 = "K0_UNCONDITIONAL"        # parity / ratio denominator
MODEL_K1 = "K1_STATE"                # parity / ratio denominator
MODEL_M0 = "M0_STATE_AVAIL"          # == 1A.2 K1A   (baseline)
MODEL_MF = "MF_STATE_LAG1"           # == 1A.2 K2    (full lag)
MODEL_MZ = "MZ_STATE_INNOV"          # NEW: innovation only
# explanatory input-side variants (never gated)
MODEL_MZP = "MZ_PATH_INNOV"          # Path Z_t inputs only
MODEL_MZN = "MZ_NONPATH_INNOV"       # non-Path Z_t inputs only

LAG_AVAIL = lag.LAG_AVAIL

# ---------------------------------------------------------------------------
# Z_t = the FULL stochastic innovation of the transition S_{t-1} -> S_t.
# It must cover EVERY stochastic node of the 1A.1b transition kernel:
#     * 7 support-correct nodes, encoded as 14 columns  (base.ALL_Z_COLS)
#     * 2 hurdle-Poisson count nodes                    (base.COUNT_Z)
# Omitting the count nodes would silently test Z_t^{continuous} instead of the
# pre-registered Z_t^{full}, making a negative MF-MZ uninterpretable.
# ---------------------------------------------------------------------------
INNOV_SOURCE_COLS = list(base.ALL_Z_COLS) + list(base.COUNT_Z)
ZT_COLS = [f"zt_{c}" for c in INNOV_SOURCE_COLS]

# native stochastic objects vs encoded predictor columns
N_NATIVE_STOCHASTIC_NODES = len(base.NODE_SPECS) + len(base.COUNT_Z)   # 9
N_ENCODED_INNOV_COLS = len(ZT_COLS)                                    # 16

if len(base.ALL_Z_COLS) != 14:
    raise SystemExit("STOP_DYNAMIC_PGM1A2B_INNOV_SOURCE_UNEXPECTED:ALL_Z!=14")
if len(base.COUNT_Z) != 2:
    raise SystemExit("STOP_DYNAMIC_PGM1A2B_INNOV_SOURCE_UNEXPECTED:COUNT!=2")
if len(INNOV_SOURCE_COLS) != 16 or len(ZT_COLS) != 16:
    raise SystemExit("STOP_DYNAMIC_PGM1A2B_INNOV_SOURCE_UNEXPECTED:ZT!=16")
if N_NATIVE_STOCHASTIC_NODES != 9:
    raise SystemExit("STOP_DYNAMIC_PGM1A2B_NATIVE_NODE_COUNT_UNEXPECTED")

# Path innovation block (dmfe, dmae, dcr, range) vs everything else.
# NOTE: the count innovations fall in the NON-Path side.
_PATH_NODES = lag.ATTRIB_NODE_BLOCKS["Path"]
PATH_ZT_COLS = [f"zt_{c}" for n in _PATH_NODES for c in base.NODE_ZCOLS[n]]
NONPATH_ZT_COLS = [c for c in ZT_COLS if c not in set(PATH_ZT_COLS)]
if sorted(PATH_ZT_COLS + NONPATH_ZT_COLS) != sorted(ZT_COLS):
    raise SystemExit("STOP_DYNAMIC_PGM1A2B_INNOV_SPLIT_NOT_A_PARTITION")
if set(PATH_ZT_COLS) & set(NONPATH_ZT_COLS):
    raise SystemExit("STOP_DYNAMIC_PGM1A2B_INNOV_SPLIT_OVERLAP")
if len(PATH_ZT_COLS) != 9 or len(NONPATH_ZT_COLS) != 7:
    raise SystemExit("STOP_DYNAMIC_PGM1A2B_INNOV_SPLIT_SIZE_UNEXPECTED")

M0_EXTRA = [LAG_AVAIL]                            #  1
MF_EXTRA = [LAG_AVAIL] + list(lag.LAG_COLS)       # 16  (15 lag states)
MZ_EXTRA = [LAG_AVAIL] + list(ZT_COLS)            # 17  (16 encoded innov)
MZP_EXTRA = [LAG_AVAIL] + PATH_ZT_COLS            # 10
MZN_EXTRA = [LAG_AVAIL] + NONPATH_ZT_COLS         #  8

# comparisons
C_REF = "K1-K0"        # state baseline (ratio denominator)
C_FULL = "MF-M0"       # full-lag gain   G_F
C_INNOV = "MZ-M0"      # innovation gain G_Z   -> GATE 1
C_RESID = "MF-MZ"      # residual after Z_t    -> GATE 2

PREFIX = "dynamic_pgm1a2b"


def add_zt_causal(df):
    """Add Z_t = the FULL realized innovation that produced S_t from S_{t-1}.

    Row r's encoded innovation columns (14 support-correct + 2 count) are
    Z_{r+1}, so Z_t is the previous row's encoded innovation vector
    (bar_t gaps are proven == 0, so that row is exactly bar_t - 1).
    Episode-first rows have no Z_t -> NaN (imputed later), and availability
    is identical to `lag1_available`.
    """
    x = df.sort_values(["episode_id", "bar_t"], kind="stable").copy()
    g = x.groupby("episode_id", sort=False)
    prev_bar = g["bar_t"].shift(1)
    for c in INNOV_SOURCE_COLS:
        x[f"zt_{c}"] = g[c].shift(1)

    # hard audit: Z_t is exactly the episode-internal previous row's vector
    avail = prev_bar.notna().to_numpy()
    for c in INNOV_SOURCE_COLS:
        a = x.loc[avail, f"zt_{c}"].to_numpy(dtype=np.float64)
        b = g[c].shift(1).loc[avail].to_numpy(dtype=np.float64)
        if not np.array_equal(a, b, equal_nan=True):
            raise SystemExit(f"STOP_DYNAMIC_PGM1A2B_ZT_CAUSALITY_FAIL:{c}")
    # hard audit: the shifted row must be the true previous BAR (bar_t - 1).
    # Z_t correctness depends on it -- a gap would silently make "Z_t" the
    # innovation of some older bar.
    same_ep = g["episode_id"].shift(1).eq(x["episode_id"]).to_numpy()
    gap = (avail & same_ep
           & (x["bar_t"].to_numpy() != prev_bar.to_numpy() + 1))
    if int(gap.sum()) != 0:
        raise SystemExit(f"STOP_DYNAMIC_PGM1A2B_ZT_BAR_GAP:{int(gap.sum())}")

    # availability must agree with the lag1 availability flag
    if LAG_AVAIL in x.columns:
        if not np.array_equal(avail, x[LAG_AVAIL].to_numpy().astype(bool)):
            raise SystemExit("STOP_DYNAMIC_PGM1A2B_ZT_AVAILABILITY_MISMATCH")
    return x


def audit_zt(cur):
    g = cur.groupby("episode_id", sort=False)
    prev = g["bar_t"].shift(1)
    same_ep = g["episode_id"].shift(1).eq(cur["episode_id"])
    gap = (prev.notna() & same_ep
           & (cur["bar_t"].to_numpy() != prev.to_numpy() + 1))
    avail = cur[LAG_AVAIL].to_numpy().astype(bool)
    return dict(
        n_rows=int(len(cur)),
        # native stochastic objects vs encoded design-matrix columns
        n_native_stochastic_nodes=N_NATIVE_STOCHASTIC_NODES,
        native_node_names=[n for n, *_ in base.NODE_SPECS] + list(base.COUNT_Z),
        n_encoded_innov_columns=N_ENCODED_INNOV_COLS,
        n_support_encoded_columns=len(base.ALL_Z_COLS),
        n_count_innovation_columns=len(base.COUNT_Z),
        n_zt_columns=len(ZT_COLS),
        zt_columns=ZT_COLS,
        n_zt_available=int(avail.sum()),
        n_zt_missing=int((~avail).sum()),
        bar_t_gap_count=int(gap.sum()),
        path_zt_columns=PATH_ZT_COLS,
        nonpath_zt_columns=NONPATH_ZT_COLS,
        n_path_encoded=len(PATH_ZT_COLS),
        n_nonpath_encoded=len(NONPATH_ZT_COLS),
        cross_episode_leakage=bool(int((~same_ep & prev.notna()).sum()) != 0),
    )


def _node_eval_nll(nodes):
    return np.sum([nodes[n]["ev"] for n, _ in base.Z_LAYOUT], axis=0)


# ===========================================================================
# one window
# ===========================================================================
def run_single_window_1a2b(w, data_path):
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

    # ---------------- shared constant count magnitude (K0) ----------------
    t0 = time.perf_counter()
    k0 = base.fit_constant_heads(Zc_tr, Zc_ev, yd_tr, yd_ev)
    k0_count = base.fit_constant_count_head(Yc_tr, Yc_ev)
    opt_rows.append(dict(window=w["name"], model=MODEL_K0,
                         n_params=k0["n_params"] + k0_count["n_params"],
                         success=True,
                         elapsed_seconds=round(time.perf_counter() - t0, 3)))

    obs = list(base.OBS_STATE_NUM)
    specs = [
        (MODEL_K1, obs, t0),
        (MODEL_M0, obs + M0_EXTRA, None),
        (MODEL_MF, obs + MF_EXTRA, None),
        (MODEL_MZ, obs + MZ_EXTRA, None),
        (MODEL_MZP, obs + MZP_EXTRA, None),
        (MODEL_MZN, obs + MZN_EXTRA, None),
    ]
    fitted = {}
    for tag, num_cols, _ in specs:
        t_m = time.perf_counter()
        ct = lag._make_ct(num_cols)
        Xtr = ct.fit_transform(tr).astype(np.float32)
        Xev = ct.transform(ev).astype(np.float32)
        if Xev.shape[0] != len(ev):
            raise SystemExit("STOP_DYNAMIC_PGM1A2B_EVAL_ROW_MISMATCH")
        k = base.fit_state_heads(Xtr, Xev, Zc_tr, Zc_ev, yd_tr, yd_ev)
        kc = base.fit_state_count_head(
            Xtr, Yc_tr, Xev, Yc_ev,
            constant_rates=k0_count["constant_rates"])
        fitted[tag] = (k, kc)
        opt_rows.append(dict(window=w["name"], model=tag,
                             n_params=k["n_params"] + kc["n_params"],
                             success=True,
                             elapsed_seconds=round(time.perf_counter() - t_m, 3)))
        del Xtr, Xev, ct

    # every model must be fitted and evaluated on the identical eval rows
    assert set(fitted) == {MODEL_K1, MODEL_M0, MODEL_MF, MODEL_MZ,
                           MODEL_MZP, MODEL_MZN}

    # ---------------- joint NLL ----------------
    def _joint(k, kc):
        cont = _node_eval_nll(k["nodes"])
        disc = k["disc_ev"]
        cnt = kc["nll_ev"].sum(axis=1)
        return cont, disc, cnt, cont + disc + cnt

    J = {}
    J[MODEL_K0] = _joint(k0, k0_count)
    for tag in (MODEL_K1, MODEL_M0, MODEL_MF, MODEL_MZ, MODEL_MZP, MODEL_MZN):
        J[tag] = _joint(*fitted[tag])

    for tag, (cont, disc, cnt, j) in J.items():
        m_c, m_d, m_n, m_j = (float(np.mean(cont)), float(np.mean(disc)),
                              float(np.mean(cnt)), float(np.mean(j)))
        assert abs(m_j - m_c - m_d - m_n) < 1e-10
        assert np.all(np.isfinite(j))
        model_metrics.append(dict(window=w["name"], model=tag,
                                  mean_joint_nll=m_j, mean_cont_nll=m_c,
                                  mean_disc_nll=m_d, mean_count_nll=m_n,
                                  n_rows=int(len(ev))))

    # ---------------- count magnitude closure (all models) ----------------
    for j in range(len(base.COUNT_Z)):
        r0 = k0_count["rate_ev"][:, j]
        for tag in (MODEL_K1, MODEL_M0, MODEL_MF, MODEL_MZ, MODEL_MZP, MODEL_MZN):
            if not np.array_equal(r0, fitted[tag][1]["rate_ev"][:, j]):
                raise SystemExit(
                    "STOP_DYNAMIC_PGM1A2B_COUNT_MAGNITUDE_NOT_CANCELLED")

    p0 = {tag: fitted[tag][1]["p0_ev"] for tag in
          (MODEL_M0, MODEL_MF, MODEL_MZ, MODEL_MZP, MODEL_MZN)}
    occ = {tag: lag._occ_nll(Yc_ev, p) for tag, p in p0.items()}
    # count NLL difference (index 2 of _joint is the count component)
    d_cnt = J[MODEL_MZ][2] - J[MODEL_M0][2]
    d_occ = occ[MODEL_MZ].sum(axis=1) - occ[MODEL_M0].sum(axis=1)
    if float(np.max(np.abs(d_cnt - d_occ))) >= 1e-10:
        raise SystemExit(
            "STOP_DYNAMIC_PGM1A2B_COUNT_MAGNITUDE_NOT_CANCELLED")

    # ---------------- comparisons ----------------
    j = {tag: J[tag][3] for tag in J}
    deltas = {
        C_REF: j[MODEL_K1] - j[MODEL_K0],
        C_FULL: j[MODEL_MF] - j[MODEL_M0],
        C_INNOV: j[MODEL_MZ] - j[MODEL_M0],
        C_RESID: j[MODEL_MF] - j[MODEL_MZ],
    }
    for label, d in deltas.items():
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

    # ---------------- target-block attribution (explanatory) --------------
    for bname, nodes in lag.ATTRIB_NODE_BLOCKS.items():
        def s(tag):
            return np.sum([fitted[tag][0]["nodes"][n]["ev"] for n in nodes],
                          axis=0)
        s0, s1 = s(MODEL_M0), s(MODEL_MF)
        sz = s(MODEL_MZ)
        block_rows.append(dict(
            window=w["name"], block=bname,
            mean_nll_m0=float(np.mean(s0)), mean_nll_mf=float(np.mean(s1)),
            mean_nll_mz=float(np.mean(sz)),
            delta_mf_minus_m0=float(np.mean(s1 - s0)),
            delta_mz_minus_m0=float(np.mean(sz - s0)),
            delta_mf_minus_mz=float(np.mean(s1 - sz))))
    o0 = occ[MODEL_M0].sum(axis=1)
    of, oz = occ[MODEL_MF].sum(axis=1), occ[MODEL_MZ].sum(axis=1)
    block_rows.append(dict(
        window=w["name"], block=lag.COUNT_OCCURRENCE_BLOCK,
        mean_nll_m0=float(np.mean(o0)), mean_nll_mf=float(np.mean(of)),
        mean_nll_mz=float(np.mean(oz)),
        delta_mf_minus_m0=float(np.mean(of - o0)),
        delta_mz_minus_m0=float(np.mean(oz - o0)),
        delta_mf_minus_mz=float(np.mean(of - oz))))

    # ---------------- per-target ----------------
    for nm, _cols in base.Z_LAYOUT:
        n0 = k0["nodes"][nm]["ev"]
        n1 = fitted[MODEL_M0][0]["nodes"][nm]["ev"]
        nf = fitted[MODEL_MF][0]["nodes"][nm]["ev"]
        nz = fitted[MODEL_MZ][0]["nodes"][nm]["ev"]
        target_rows.append(dict(
            window=w["name"], target=nm, kind="node",
            mean_nll_m0=float(np.mean(n1)), mean_nll_mf=float(np.mean(nf)),
            mean_nll_mz=float(np.mean(nz)),
            delta_mf_minus_m0=float(np.mean(nf - n1)),
            delta_mz_minus_m0=float(np.mean(nz - n1)),
            delta_mf_minus_mz=float(np.mean(nf - nz))))
    for j2, c in enumerate(base.COUNT_Z):
        n0 = k0_count["nll_ev"][:, j2]
        n1 = fitted[MODEL_M0][1]["nll_ev"][:, j2]
        nf = fitted[MODEL_MF][1]["nll_ev"][:, j2]
        nz = fitted[MODEL_MZ][1]["nll_ev"][:, j2]
        target_rows.append(dict(
            window=w["name"], target=c, kind="discrete_count",
            mean_nll_m0=float(np.mean(n1)), mean_nll_mf=float(np.mean(nf)),
            mean_nll_mz=float(np.mean(nz)),
            delta_mf_minus_m0=float(np.mean(nf - n1)),
            delta_mz_minus_m0=float(np.mean(nz - n1)),
            delta_mf_minus_mz=float(np.mean(nf - nz))))
        ye = Yc_ev[:, j2]
        ze = (ye > 0).astype(int)
        count_occ_rows.append(dict(
            window=w["name"], target=c,
            constant_ztp_lambda=float(k0_count["constant_rates"][j2]),
            state_dependent_magnitude=False,
            mean_occ_nll_m0=float(np.mean(occ[MODEL_M0][:, j2])),
            mean_occ_nll_mf=float(np.mean(occ[MODEL_MF][:, j2])),
            mean_occ_nll_mz=float(np.mean(occ[MODEL_MZ][:, j2])),
            delta_occ_nll_mz_minus_m0=float(np.mean(
                occ[MODEL_MZ][:, j2] - occ[MODEL_M0][:, j2])),
            delta_count_nll_mz_minus_m0=float(np.mean(nz - n1)),
            count_minus_occurrence_residual=float(np.max(np.abs(
                (nz - n1) - (occ[MODEL_MZ][:, j2] - occ[MODEL_M0][:, j2])))),
        ))

    # ---------------- explanatory input-side innovation split -------------
    gF = float(np.mean(j[MODEL_MF] - j[MODEL_M0]))
    gZ = float(np.mean(j[MODEL_MZ] - j[MODEL_M0]))
    innov_rows = [dict(
        window=w["name"],
        gain_full_lag_rowmean=gF,
        gain_innovation_rowmean=gZ,
        retention_abs_ratio=abs(gZ) / abs(gF) if gF != 0 else float("nan"),
        gain_innov_path_rowmean=float(np.mean(j[MODEL_MZP] - j[MODEL_M0])),
        gain_innov_nonpath_rowmean=float(np.mean(j[MODEL_MZN] - j[MODEL_M0])),
        n_zt_path_columns=len(PATH_ZT_COLS),
        n_zt_nonpath_columns=len(NONPATH_ZT_COLS),
        note=("input-side split is EXPLANATORY ONLY (no CI, no gate); "
              "component gains are not additive"),
    )]

    del fitted, tr, ev
    print(f"[STAGE] window {w['name']} done "
          f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)
    return dict(window=w["name"], model_metrics=model_metrics,
                opt_rows=opt_rows, boots=boots, bysym=bysym,
                block_rows=block_rows, target_rows=target_rows,
                count_occ_rows=count_occ_rows, innov_rows=innov_rows,
                n_symbols=int(len(pd.unique(sym_ev))))


# ===========================================================================
# main
# ===========================================================================
def main():
    t_total = time.perf_counter()
    timing = {}

    t0 = time.perf_counter()
    keep_cols = (list(base.OBS_STATE_NUM) + list(base.OBS_STATE_CAT)
                 + list(lag.LAG_BASE) + base.BUILD_SRC_COLS
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
    cur = lag.add_lag1_causal(cur)
    cur = add_zt_causal(cur)
    base._peak("after add_zt_causal")
    del df
    timing["build_transition_seconds"] = round(time.perf_counter() - t0, 2)
    print(f"[STAGE] build done n={len(cur)}", flush=True)

    # ---------------- sample parity ----------------
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
    zt_audit = audit_zt(cur)
    parity_audit = dict(
        parent_commit=BASE_SHA,
        grandparent_commit=PARENT_1A1B_SHA,
        rows_ok=bool(len(cur) == EXPECTED_TRANSITIONS),
        z_feature_hash_ok=bool(h == FROZEN_Z_FEATURE_HASH),
        z_feature_hash_expected=FROZEN_Z_FEATURE_HASH,
        z_feature_hash_actual=h,
        joint_nll_parity_1a1b={},
        joint_nll_parity_1a2={},
        joint_nll_tol=PARITY_TOL,
    )
    if not (parity_audit["rows_ok"] and parity_audit["z_feature_hash_ok"]):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A2B_SAMPLE_PARITY_FAIL: {parity_audit}")

    # ---------------- invariants + reconstruction (reuse 1A.1b) -----------
    invariants = dict(
        tv_update_exact=bool(np.allclose(
            (nxt["path_total_variation_R"].to_numpy(float)
             - cur["path_total_variation_R"].to_numpy(float)),
            np.abs(cur["z_d_up"].to_numpy(float)), atol=1e-8)),
        last_return_exact=bool(np.allclose(
            nxt["path_last_return_R"].to_numpy(float),
            -cur["z_d_up"].to_numpy(float), atol=1e-8)),
        no_tb4=bool("TB4" not in set(cur["block"].unique())),
        zt_present_where_lag1_available=bool(
            cur.loc[cur[LAG_AVAIL] == 1, ZT_COLS].notna().all().all()),
        zt_missing_where_lag1_unavailable=bool(
            cur.loc[cur[LAG_AVAIL] == 0, ZT_COLS].isna().all().all()),
    )
    if not all(invariants.values()):
        raise SystemExit(f"STOP_DYNAMIC_PGM1A_INVARIANT_FAIL: {invariants}")

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

    (base.OUT / f"{PREFIX}_zt_audit.json").write_text(
        json.dumps(dict(zt=zt_audit, sample=sample_audit,
                        support=support_audit,
                        reconstruction=full_recon), indent=2, default=str))
    print(f"[AUDIT] zt={zt_audit['n_zt_available']}/{zt_audit['n_rows']} "
          f"available, bar_t_gap={zt_audit['bar_t_gap_count']}", flush=True)

    del nxt

    if os.environ.get("DYNAMIC_PGM1A2B_AUDIT_ONLY") == "1":
        (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
            json.dumps(parity_audit, indent=2, default=str))
        print("[AUDIT-ONLY] Complete. Stopping before any model fit.",
              flush=True)
        return dict(parity=parity_audit, zt=zt_audit, sample=sample_audit)

    # ---------------- persist cache (with lag1 + Z_t columns) -------------
    needed_cols = (list(base.OBS_STATE_NUM) + list(base.OBS_STATE_CAT)
                   + base.ALL_Z_COLS + [base.DISC_Z] + base.COUNT_Z
                   + list(lag.LAG_COLS) + [LAG_AVAIL] + ZT_COLS
                   + ["episode_id", "symbol", "block", "episode_start_day"])
    for c in cur.columns:
        if str(cur[c].dtype).startswith("float64"):
            cur[c] = cur[c].astype(np.float32)
    cur = cur[list(dict.fromkeys(needed_cols))].copy()
    data_path = base.CACHE / f"{PREFIX}_transitions.parquet"
    cur.to_parquet(data_path, index=False)
    del cur
    base._peak("after slim cur")

    # ---------------- windows ----------------
    win_env = dict(os.environ)
    for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        win_env[_bt] = "1"

    model_metrics, opt_rows, boots, bysym = [], [], [], []
    block_rows, target_rows, count_occ_rows, innov_rows = [], [], [], []
    n_symbols = 0
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
        innov_rows += res.get("innov_rows", [])
        n_symbols = max(n_symbols, int(res.get("n_symbols", 0)))
        print(f"[STAGE] window {w['name']} subprocess done "
              f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)

    # ---------------- parity gates ----------------
    for w in base.WINDOWS:
        got = {m["model"]: m["mean_joint_nll"] for m in model_metrics
               if m["window"] == w["name"]}
        f1 = FROZEN_1A1B_JOINT[w["name"]]
        d0 = abs(got[MODEL_K0] - f1["K0_UNCONDITIONAL"])
        d1 = abs(got[MODEL_K1] - f1["K1_STATE"])
        parity_audit["joint_nll_parity_1a1b"][w["name"]] = dict(
            k0_abs_diff=d0, k1_abs_diff=d1,
            ok=bool(d0 < PARITY_TOL and d1 < PARITY_TOL))
        f2 = FROZEN_1A2_JOINT[w["name"]]
        dm0 = abs(got[MODEL_M0] - f2["K1A_STATE_AVAIL"])
        dmf = abs(got[MODEL_MF] - f2["K2_STATE_LAG1"])
        parity_audit["joint_nll_parity_1a2"][w["name"]] = dict(
            m0_abs_diff=dm0, mf_abs_diff=dmf,
            ok=bool(dm0 < PARITY_TOL and dmf < PARITY_TOL))
        if (d0 >= PARITY_TOL or d1 >= PARITY_TOL
                or dm0 >= PARITY_TOL or dmf >= PARITY_TOL):
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A2B_PARITY_FAIL: {w['name']} "
                f"dK0={d0} dK1={d1} dM0={dm0} dMF={dmf}")

    # ---------------- verdict ----------------
    def _row(wname, label):
        return next(b for b in boots if b["window"] == wname
                    and b["comparison"] == label)

    def _sym_neg(wname, label):
        sub = [b for b in bysym if b["window"] == wname
               and b["comparison"] == label]
        return int(sum(1 for b in sub if b["mean_delta_episode"] < 0))

    def _passes(wname, label):
        r = _row(wname, label)
        return bool(r["ci_hi"] < 0 and _sym_neg(wname, label) >= SYMBOL_BREADTH_MIN)

    windows_report = {}
    for w in base.WINDOWS:
        rz, rr, rf, ref = (_row(w["name"], C_INNOV), _row(w["name"], C_RESID),
                           _row(w["name"], C_FULL), _row(w["name"], C_REF))
        denom = abs(ref["delta_sample_mean"])
        gF, gZ = rf["delta_sample_mean"], rz["delta_sample_mean"]
        windows_report[w["name"]] = dict(
            delta_mz_minus_m0=gZ, mz_ci_lo=rz["ci_lo"], mz_ci_hi=rz["ci_hi"],
            n_symbols_negative=_sym_neg(w["name"], C_INNOV),
            n_symbols_total=n_symbols,
            delta_mf_minus_m0=gF,
            delta_mf_minus_mz=rr["delta_sample_mean"],
            resid_ci_lo=rr["ci_lo"], resid_ci_hi=rr["ci_hi"],
            retention_abs_ratio=(abs(gZ) / abs(gF) if gF != 0 else float("nan")),
            increment_vs_state_ratio=(abs(gZ) / denom if denom != 0
                                      else float("nan")),
            gate1_innovation_carries_info=_passes(w["name"], C_INNOV),
            gate2_full_lag_retains_residual=_passes(w["name"], C_RESID))
    g1 = all(v["gate1_innovation_carries_info"] for v in windows_report.values())
    g2 = all(v["gate2_full_lag_retains_residual"] for v in windows_report.values())
    if not g1:
        verdict = "INNOVATION_ATTRIBUTION_NOT_SUPPORTED"
    elif g2:
        verdict = "FULL_LAG_STATE_RETAINS_RESIDUAL_INFORMATION"
    else:
        verdict = "LAG1_MEMORY_COMPRESSIBLE_TO_INNOVATION"

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
        base.OUT / f"{PREFIX}_block_attribution.csv", index=False)
    pd.DataFrame(innov_rows).to_csv(
        base.OUT / f"{PREFIX}_innovation_attribution.csv", index=False)
    pd.DataFrame(count_occ_rows).to_csv(
        base.OUT / f"{PREFIX}_count_occurrence.csv", index=False)
    pd.DataFrame(opt_rows).to_csv(
        base.OUT / f"{PREFIX}_optimizer_audit.csv", index=False)
    (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
        json.dumps(parity_audit, indent=2, default=str))

    summary = dict(
        experiment="DYNAMIC-PGM-1A.2b Innovation Attribution",
        parent_commit=BASE_SHA,
        grandparent_commit=PARENT_1A1B_SHA,
        question=("Is the information in S_{t-1} really just the most recent "
                  "innovation Z_t?"),
        verdict=verdict,
        models=dict(
            M0_STATE_AVAIL="P(Z_{t+1} | S_t, A_t)  [= frozen 1A.2 K1A]",
            MF_STATE_LAG1="P(Z_{t+1} | S_t, A_t, S_{t-1})  [= frozen 1A.2 K2]",
            MZ_STATE_INNOV="P(Z_{t+1} | S_t, A_t, Z_t)  [NEW]"),
        gates=dict(
            gate1="MZ-M0 : does Z_t carry residual information?",
            gate2="MF-MZ : does full S_{t-1} still help beyond Z_t?",
            gate1_passed=bool(g1), gate2_passed=bool(g2),
            breadth_min=SYMBOL_BREADTH_MIN),
        windows=windows_report,
        innovation_attribution=innov_rows,
        block_attribution=block_rows,
        per_target=target_rows,
        model_metrics=model_metrics,
        bootstrap=boots,
        sample=sample_audit,
        zt_audit=zt_audit,
        parity=parity_audit,
        zt_block=dict(
            n_native_stochastic_nodes=N_NATIVE_STOCHASTIC_NODES,
            n_encoded_innov_columns=N_ENCODED_INNOV_COLS,
            n_support_encoded_columns=len(base.ALL_Z_COLS),
            n_count_innovation_columns=len(base.COUNT_Z),
            n_zt_columns=len(ZT_COLS), zt_columns=ZT_COLS,
            path_zt_columns=PATH_ZT_COLS,
            nonpath_zt_columns=NONPATH_ZT_COLS,
            n_path_encoded=len(PATH_ZT_COLS),
            n_nonpath_encoded=len(NONPATH_ZT_COLS),
            availability_column=LAG_AVAIL,
            note=("9 native stochastic nodes (7 support-correct + 2 count) "
                  "expanded to 16 encoded predictor columns to reuse the "
                  "existing support representation")),
        count_magnitude_closure=dict(
            magnitude=("state-independent train-constant exact ZTP shared by "
                       "K0/K1/M0/MF/MZ/MZ_PATH/MZ_NONPATH"),
            occurrence="state-dependent Logistic (may differ across models)"),
        frozen_scope=["no lag2/lag3", "no latent/HMM/HSMM",
                      "no alpha/C tuning", "no SMC/OB", "no terminal/reset",
                      "no rollout", "no PnL/RL"],
        bootstrap_reps=base.BOOTSTRAP_REPS,
    )
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (base.OUT / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[MODEL METRICS]\n{pd.DataFrame(model_metrics).to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{pd.DataFrame(boots).to_string(index=False)}")
    print(f"[INNOVATION ATTRIBUTION]\n{pd.DataFrame(innov_rows).to_string(index=False)}")
    print(f"[VERDICT] {verdict}")
    print("[GATE1 MZ-M0] " + ", ".join(
        f"{k}: {v['delta_mz_minus_m0']:.5f} "
        f"CI=[{v['mz_ci_lo']:.5f},{v['mz_ci_hi']:.5f}] "
        f"neg={v['n_symbols_negative']}/{v['n_symbols_total']}"
        for k, v in windows_report.items()))
    print("[GATE2 MF-MZ] " + ", ".join(
        f"{k}: {v['delta_mf_minus_mz']:.5f} "
        f"CI=[{v['resid_ci_lo']:.5f},{v['resid_ci_hi']:.5f}]"
        for k, v in windows_report.items()))
    print("[RETENTION |MZ-M0|/|MF-M0|] " + ", ".join(
        f"{k}: {v['retention_abs_ratio']:.4f}"
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
                     help="Run real-data audits (sample parity, Z_t "
                          "causality, reconstruction) and stop.")
    _a = _ap.parse_args()
    if _a.audit_only:
        os.environ["DYNAMIC_PGM1A2B_AUDIT_ONLY"] = "1"
    if _a.window_json:
        if not _a.agezero_deterministic:
            raise SystemExit(
                "STOP_DYNAMIC_PGM1A2B_CHILD_AGEZERO_STATE_NOT_PROPAGATED")
        base.configure_child_semantics(agezero_deterministic=True)
        if base.disc_spec() != []:
            raise SystemExit("STOP_DYNAMIC_PGM1A2B_CHILD_DISC_SPEC_NONEMPTY")
        _w = json.loads(_a.window_json)
        Path(_a.result).write_text(
            json.dumps(run_single_window_1a2b(_w, _a.data), default=str))
    else:
        main()
