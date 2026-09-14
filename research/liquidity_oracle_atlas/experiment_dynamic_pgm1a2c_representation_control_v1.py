"""DYNAMIC-PGM-1A.2c -- Representation Control.

Why this exists
---------------
1A.2b produced LAG1_MEMORY_COMPRESSIBLE_TO_INNOVATION:

    MZ - M0 = -0.27247 (A), -0.25673 (B)   15/15 symbols, CI deep below 0
    MF - MZ = +0.20915 (A), +0.19354 (B)   CI entirely ABOVE 0

But MZ's advantage is suspiciously large (~4x the full-lag gain of 1A.2), and
there is a concrete confound: several Z_t encoded columns are NOT memory at
all -- they are nonlinear re-encodings of quantities ALREADY present in S_t.

Because every head is linear (Ridge / Logistic), handing it a re-encoded copy
of a current-state field is a free nonlinear basis function, not history.

So 1A.2b's result strictly supports only:

    "S_t + Z_t representation dominates S_t + S_{t-1} representation"

It does NOT yet establish that recent innovation carries genuine history.

What 1A.2c does
---------------
1. A DETERMINISTIC recoverability audit (algebraic, NOT statistical):
   for every Z_t encoded column, ask "is this strictly computable from S_t
   alone under the frozen definitions?"

       CURRENT_STATE_REENCODING  -> already in S_t (or a frozen identity of it)
       TRUE_HISTORY_INNOVATION   -> requires S_{t-1}, i.e. real memory

   Every classification is VERIFIED by exact equality against the frozen
   encoder (base.build_z_columns); a failed verification is a hard STOP.

2. A control model using ONLY the re-encoding half:

       M_C = P(Z_{t+1} | S_t, A_t, Z_t^{cur})

   so that

       M_C - M0   =  pure current-state re-encoding gain
       MZ  - M_C  =  genuine recent-innovation memory  <-- PRIMARY GATE

Models
------
    M0    = S_t + A_t                     (frozen 1A.2 K1A)
    MF    = S_t + A_t + S_{t-1}           (frozen 1A.2 K2)
    MZ    = S_t + A_t + Z_t (16)          (frozen 1A.2b)
    M_C   = S_t + A_t + Z_t^{cur}         (NEW control)
    M_MEM = S_t + A_t + Z_t^{mem}         (NEW, explanatory only)

Gate
----
    PRIMARY: MZ - M_C
      both windows CI_hi < 0 and >= 10/15 symbols negative
        -> RECENT_INNOVATION_MEMORY_SUPPORTED
      otherwise
        -> RECENT_INNOVATION_MEMORY_NOT_SUPPORTED_GAIN_IS_REENCODING

    M_C - M0, MZ - M0, MF - MZ, MF - M0 are reported, never gated here.

Frozen scope
------------
No new feature mining, no lag2, no latent state, no alpha/C tuning,
no terminal/reset, no rollout, no PnL/RL. Count Magnitude Closure inherited
unchanged (one shared train-only constant ZTP lambda across all models).
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
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2b_innovation_attribution_v1 as prev  # noqa: E402

# ===========================================================================
# frozen references
# ===========================================================================
BASE_SHA = "1089fd1622b4b9de7743a8180a40f5d89176c8c4"   # 1A.2b result commit
P_1A2_SHA = "b12778e1159e322fc9f2de6937a00eb0b1bf6260"
P_1A1B_SHA = "16ab731ac099ffcec7c2089640b95b5555184853"
EXPECTED_TRANSITIONS = base.EXPECTED_TRANSITIONS
FROZEN_Z_FEATURE_HASH = lag.FROZEN_Z_FEATURE_HASH
FROZEN_1A1B_JOINT = lag.FROZEN_JOINT
FROZEN_1A2_JOINT = prev.FROZEN_1A2_JOINT
# frozen 1A.2b MZ
FROZEN_1A2B_JOINT = {
    "A_TB1_to_TB2": dict(MZ_STATE_INNOV=0.956009745907654),
    "B_TB1TB2_to_TB3": dict(MZ_STATE_INNOV=1.0336791275514023),
}
PARITY_TOL = 1e-8
SYMBOL_BREADTH_MIN = lag.SYMBOL_BREADTH_MIN

# ===========================================================================
# models
# ===========================================================================
MODEL_K0 = "K0_UNCONDITIONAL"
MODEL_K1 = "K1_STATE"
MODEL_M0 = "M0_STATE_AVAIL"
MODEL_MF = "MF_STATE_LAG1"
MODEL_MZ = "MZ_STATE_INNOV"
MODEL_MC = "MC_STATE_CURREENCODING"      # NEW control: Z_t^{cur} only
MODEL_MMEM = "MMEM_STATE_TRUE_INNOV"     # NEW, explanatory: Z_t^{mem} only

LAG_AVAIL = lag.LAG_AVAIL
ZT_COLS = list(prev.ZT_COLS)

CURRENT_STATE_REENCODING = "CURRENT_STATE_REENCODING"
TRUE_HISTORY_INNOVATION = "TRUE_HISTORY_INNOVATION"

# ---------------------------------------------------------------------------
# Recoverability map: node -> (S_t source column, transform).
# Derived from the FROZEN definitions, then verified exactly below.
#   * value nodes (z_range, z_dcr, z_uresid, z_lresid): the row-(t-1) encoding
#     is the encoder applied to that field's value at time t, and that field is
#     already a column of OBS_STATE_NUM.
#   * z_d_up: the frozen build asserts  nxt.last_return == -z_d_up, hence
#     z_d_up(t-1 -> t) == -S_t["path_last_return_R"], already in S_t.
#   * z_dmfe / z_dmae: cumulative-max INCREMENTS -> need MFE(t-1) / MAE(t-1),
#     which are NOT in S_t (only the lag1_* copies carry them).
#   * count increments: cumulative differences -> need cumulative(t-1). NOT in S_t.
# ---------------------------------------------------------------------------
S_T_SOURCE = {
    "z_d_up": ("path_last_return_R", "neg"),
    "z_dcr": ("path_direction_change_rate", "id"),
    "z_range": ("path_current_bar_range_R", "id"),
    "z_uresid": ("upper_newest_log_age_residual", "id"),
    "z_lresid": ("lower_newest_log_age_residual", "id"),
}
ZT_CUR_COLS = [f"zt_{c}" for n in S_T_SOURCE for c in base.NODE_ZCOLS[n]]
ZT_MEM_COLS = [c for c in ZT_COLS if c not in set(ZT_CUR_COLS)]

M0_EXTRA = [LAG_AVAIL]
MF_EXTRA = [LAG_AVAIL] + list(lag.LAG_COLS)
MZ_EXTRA = [LAG_AVAIL] + ZT_COLS
MC_EXTRA = [LAG_AVAIL] + ZT_CUR_COLS
MMEM_EXTRA = [LAG_AVAIL] + ZT_MEM_COLS

C_REENC = "MC-M0"        # pure current-state re-encoding gain
C_MEM = "MZ-MC"          # PRIMARY: genuine innovation memory
C_INNOV = "MZ-M0"        # frozen 1A.2b reference
C_RESID = "MF-MZ"        # frozen 1A.2b reference
C_FULL = "MF-M0"         # frozen 1A.2 reference
C_REF = "K1-K0"          # ratio denominator

PREFIX = "dynamic_pgm1a2c"


# ===========================================================================
# deterministic recoverability audit
# ===========================================================================
# Comparison tolerance for the recoverability audit. It matches the tolerance
# under which the frozen build ITSELF asserts the identity last_return = -z_d_up
# (build_transition_sample checks it with atol=1e-8). Recoverability is therefore
# "deterministic up to the representation noise the frozen pipeline already
# accepts", and each column's max error is reported for transparency.
REENCODE_TOL = 1e-8


def _encode_with_frozen(node, raw):
    """Re-encode a raw value through the FROZEN support encoder.

    base.build_z_columns covers gaussian_delta / hurdle_ln_value / hurdle_ln_neg /
    ln_value / signed_hurdle. The remaining two kinds are built INLINE in
    base.build_transition_sample (z_dmfe / z_dmae as hurdle_ln_delta, z_dcr as
    zero_interior_one); we replicate that inline logic verbatim here so the
    audit uses exactly the frozen encoding, not a re-implementation guess.
    """
    kind = base.NODE_KIND[node]
    raw = np.asarray(raw, dtype=np.float64)
    cols = base.NODE_ZCOLS[node]
    if kind == "hurdle_ln_delta":
        ispos = (raw > 0).astype(float)
        logv = np.where(ispos > 0.5, np.log(np.maximum(raw, 1e-12)), 0.0)
        return {cols[0]: ispos, cols[1]: logv}
    if kind == "zero_interior_one":
        is0 = (raw <= 1e-9).astype(float)
        is1 = (raw >= 1 - 1e-9).astype(float)
        interior = (raw > 1e-9) & (raw < 1 - 1e-9)
        logit = np.where(interior, np.log(raw / (1.0 - raw)), 0.0)
        return {cols[0]: is0, cols[1]: is1, cols[2]: logit}
    return base.build_z_columns(node, kind, raw)


def _cmp_avail(vals, actual, avail):
    """Compare only where Z_t exists; return (ok, max_abs_error)."""
    a = np.asarray(vals, dtype=np.float64)[avail]
    b = np.asarray(actual, dtype=np.float64)[avail]
    na, nb = np.isnan(a), np.isnan(b)
    if np.any(na != nb):
        return False, float("nan")
    m = ~na
    err = float(np.max(np.abs(a[m] - b[m]))) if m.any() else 0.0
    return bool(err <= REENCODE_TOL), err


def audit_zt_recoverability(cur):
    """Classify every Z_t column and VERIFY each classification exactly.

    Returns (classification, report). A classification that cannot be
    reproduced exactly is a hard STOP -- we never guess by column name.
    """
    cls, checks = {}, []
    for node in S_T_SOURCE:
        src_col, how = S_T_SOURCE[node]
        if src_col not in base.OBS_STATE_NUM:
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A2C_SOURCE_NOT_IN_STATE:{src_col}")
        raw = cur[src_col].to_numpy(dtype=np.float64)
        if how == "neg":
            raw = -raw
        enc = _encode_with_frozen(node, raw)
        for zc, vals in enc.items():
            col = f"zt_{zc}"
            actual = cur[col].to_numpy(dtype=np.float64)
            avail = cur[LAG_AVAIL].to_numpy().astype(bool)
            ok, err = _cmp_avail(vals, actual, avail)
            if not ok:
                raise SystemExit(
                    f"STOP_DYNAMIC_PGM1A2C_REENCODING_VERIFY_FAIL:{col}:{err}")
            cls[col] = CURRENT_STATE_REENCODING
            checks.append(dict(node=node, zt_column=col,
                               source=src_col, transform=how,
                               classification=CURRENT_STATE_REENCODING,
                               max_abs_error=err, exact=bool(err == 0.0)))

    # everything else must be a genuine cross-time quantity: verify that it is
    # reproducible ONLY from (S_t, S_{t-1}) -- i.e. it really needs the past.
    for node in ("z_dmfe", "z_dmae"):
        fld = ("path_max_up_excursion_R" if node == "z_dmfe"
               else "path_max_down_excursion_R")
        delta = (cur[fld].to_numpy(dtype=np.float64)
                 - cur[f"lag1_{fld}"].to_numpy(dtype=np.float64))
        enc = _encode_with_frozen(node, delta)
        for zc, vals in enc.items():
            col = f"zt_{zc}"
            actual = cur[col].to_numpy(dtype=np.float64)
            avail = cur[LAG_AVAIL].to_numpy().astype(bool)
            ok, err = _cmp_avail(vals, actual, avail)
            if not ok:
                raise SystemExit(
                    f"STOP_DYNAMIC_PGM1A2C_INNOV_VERIFY_FAIL:{col}:{err}")
            cls[col] = TRUE_HISTORY_INNOVATION
            checks.append(dict(node=node, zt_column=col,
                               source=f"{fld} - lag1_{fld}",
                               transform="delta",
                               classification=TRUE_HISTORY_INNOVATION,
                               max_abs_error=err, exact=bool(err == 0.0)))
    for side, node in (("upper", "z_delta_upper_count"),
                       ("lower", "z_delta_lower_count")):
        fld = f"{side}_active_identity_count_delta"
        delta = (cur[fld].to_numpy(dtype=np.float64)
                 - cur[f"lag1_{fld}"].to_numpy(dtype=np.float64))
        col = f"zt_{node}"
        actual = cur[col].to_numpy(dtype=np.float64)
        avail = cur[LAG_AVAIL].to_numpy().astype(bool)
        ok, err = _cmp_avail(delta, actual, avail)
        if not ok:
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A2C_INNOV_VERIFY_FAIL:{col}:{err}")
        cls[col] = TRUE_HISTORY_INNOVATION
        checks.append(dict(node=node, zt_column=col,
                           source=f"{fld} - lag1_{fld}",
                           transform="increment",
                           classification=TRUE_HISTORY_INNOVATION,
                           max_abs_error=err, exact=bool(err == 0.0)))

    if set(cls) != set(ZT_COLS):
        raise SystemExit("STOP_DYNAMIC_PGM1A2C_CLASSIFICATION_INCOMPLETE")
    n_cur = sum(1 for v in cls.values() if v == CURRENT_STATE_REENCODING)
    n_mem = len(cls) - n_cur
    if n_cur != len(ZT_CUR_COLS) or n_mem != len(ZT_MEM_COLS):
        raise SystemExit("STOP_DYNAMIC_PGM1A2C_SPLIT_SIZE_MISMATCH")
    report = dict(
        n_zt_columns=len(ZT_COLS),
        n_current_state_reencoding=n_cur,
        n_true_history_innovation=n_mem,
        zt_cur_columns=ZT_CUR_COLS,
        zt_mem_columns=ZT_MEM_COLS,
        classification=cls,
        checks=checks,
        method=("deterministic algebraic audit: each column re-encoded from "
                "the frozen definitions and compared by exact equality; "
                "no statistical screening, no feature selection"),
    )
    return cls, report


def _node_eval_nll(nodes):
    return np.sum([nodes[n]["ev"] for n, _ in base.Z_LAYOUT], axis=0)


# ===========================================================================
# one window
# ===========================================================================
def run_single_window_1a2c(w, data_path):
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

    model_metrics, opt_rows, target_rows = [], [], []
    count_occ_rows, boots, bysym = [], [], []

    t0 = time.perf_counter()
    k0 = base.fit_constant_heads(Zc_tr, Zc_ev, yd_tr, yd_ev)
    k0_count = base.fit_constant_count_head(Yc_tr, Yc_ev)
    opt_rows.append(dict(window=w["name"], model=MODEL_K0,
                         n_params=k0["n_params"] + k0_count["n_params"],
                         success=True,
                         elapsed_seconds=round(time.perf_counter() - t0, 3)))

    obs = list(base.OBS_STATE_NUM)
    specs = [(MODEL_K1, obs), (MODEL_M0, obs + M0_EXTRA),
             (MODEL_MF, obs + MF_EXTRA), (MODEL_MZ, obs + MZ_EXTRA),
             (MODEL_MC, obs + MC_EXTRA), (MODEL_MMEM, obs + MMEM_EXTRA)]
    fitted = {}
    for tag, num_cols in specs:
        t_m = time.perf_counter()
        ct = lag._make_ct(num_cols)
        Xtr = ct.fit_transform(tr).astype(np.float32)
        Xev = ct.transform(ev).astype(np.float32)
        if Xev.shape[0] != len(ev):
            raise SystemExit("STOP_DYNAMIC_PGM1A2C_EVAL_ROW_MISMATCH")
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

    assert set(fitted) == {MODEL_K1, MODEL_M0, MODEL_MF, MODEL_MZ,
                           MODEL_MC, MODEL_MMEM}

    def _joint(k, kc):
        cont = _node_eval_nll(k["nodes"])
        disc = k["disc_ev"]
        cnt = kc["nll_ev"].sum(axis=1)
        return cont, disc, cnt, cont + disc + cnt

    J = {MODEL_K0: _joint(k0, k0_count)}
    for tag, _ in specs:
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
        for tag, _ in specs:
            if not np.array_equal(r0, fitted[tag][1]["rate_ev"][:, j]):
                raise SystemExit(
                    "STOP_DYNAMIC_PGM1A2C_COUNT_MAGNITUDE_NOT_CANCELLED")

    occ = {tag: lag._occ_nll(Yc_ev, fitted[tag][1]["p0_ev"])
           for tag in (MODEL_M0, MODEL_MF, MODEL_MZ, MODEL_MC, MODEL_MMEM)}
    d_cnt = J[MODEL_MZ][2] - J[MODEL_M0][2]
    d_occ = occ[MODEL_MZ].sum(axis=1) - occ[MODEL_M0].sum(axis=1)
    if float(np.max(np.abs(d_cnt - d_occ))) >= 1e-10:
        raise SystemExit(
            "STOP_DYNAMIC_PGM1A2C_COUNT_MAGNITUDE_NOT_CANCELLED")

    # ---------------- comparisons ----------------
    j = {tag: J[tag][3] for tag in J}
    deltas = {
        C_REF: j[MODEL_K1] - j[MODEL_K0],
        C_FULL: j[MODEL_MF] - j[MODEL_M0],
        C_INNOV: j[MODEL_MZ] - j[MODEL_M0],
        C_RESID: j[MODEL_MF] - j[MODEL_MZ],
        C_REENC: j[MODEL_MC] - j[MODEL_M0],
        C_MEM: j[MODEL_MZ] - j[MODEL_MC],
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

    # ---------------- explanatory decomposition ----------------
    gZ = float(np.mean(j[MODEL_MZ] - j[MODEL_M0]))
    gC = float(np.mean(j[MODEL_MC] - j[MODEL_M0]))
    gM = float(np.mean(j[MODEL_MMEM] - j[MODEL_M0]))
    decom_rows = [dict(
        window=w["name"],
        gain_mz_minus_m0_rowmean=gZ,
        gain_reencoding_mc_minus_m0_rowmean=gC,
        gain_trueinnov_mmem_minus_m0_rowmean=gM,
        reencoding_share_of_mz_gain=(gC / gZ if gZ != 0 else float("nan")),
        mem_share_of_mz_gain=((gZ - gC) / gZ if gZ != 0 else float("nan")),
        n_cur_encoding_columns=len(ZT_CUR_COLS),
        n_true_innovation_columns=len(ZT_MEM_COLS),
        note=("explanatory only; component gains are not additive and "
              "row-weighted (gates use episode-weighted bootstrap)"),
    )]

    # ---------------- per-target ----------------
    for nm, _cols in base.Z_LAYOUT:
        n1 = fitted[MODEL_M0][0]["nodes"][nm]["ev"]
        nf = fitted[MODEL_MF][0]["nodes"][nm]["ev"]
        nz = fitted[MODEL_MZ][0]["nodes"][nm]["ev"]
        nc = fitted[MODEL_MC][0]["nodes"][nm]["ev"]
        target_rows.append(dict(
            window=w["name"], target=nm, kind="node",
            mean_nll_m0=float(np.mean(n1)), mean_nll_mf=float(np.mean(nf)),
            mean_nll_mz=float(np.mean(nz)), mean_nll_mc=float(np.mean(nc)),
            delta_mz_minus_m0=float(np.mean(nz - n1)),
            delta_mc_minus_m0=float(np.mean(nc - n1)),
            delta_mz_minus_mc=float(np.mean(nz - nc))))
    for j2, c in enumerate(base.COUNT_Z):
        n1 = fitted[MODEL_M0][1]["nll_ev"][:, j2]
        nf = fitted[MODEL_MF][1]["nll_ev"][:, j2]
        nz = fitted[MODEL_MZ][1]["nll_ev"][:, j2]
        nc = fitted[MODEL_MC][1]["nll_ev"][:, j2]
        target_rows.append(dict(
            window=w["name"], target=c, kind="discrete_count",
            mean_nll_m0=float(np.mean(n1)), mean_nll_mf=float(np.mean(nf)),
            mean_nll_mz=float(np.mean(nz)), mean_nll_mc=float(np.mean(nc)),
            delta_mz_minus_m0=float(np.mean(nz - n1)),
            delta_mc_minus_m0=float(np.mean(nc - n1)),
            delta_mz_minus_mc=float(np.mean(nz - nc))))
        ye = Yc_ev[:, j2]
        count_occ_rows.append(dict(
            window=w["name"], target=c,
            constant_ztp_lambda=float(k0_count["constant_rates"][j2]),
            state_dependent_magnitude=False,
            mean_occ_nll_m0=float(np.mean(occ[MODEL_M0][:, j2])),
            mean_occ_nll_mz=float(np.mean(occ[MODEL_MZ][:, j2])),
            mean_occ_nll_mc=float(np.mean(occ[MODEL_MC][:, j2])),
            delta_occ_nll_mz_minus_m0=float(np.mean(
                occ[MODEL_MZ][:, j2] - occ[MODEL_M0][:, j2])),
            delta_count_nll_mz_minus_m0=float(np.mean(nz - n1)),
            count_minus_occurrence_residual=float(np.max(np.abs(
                (nz - n1) - (occ[MODEL_MZ][:, j2] - occ[MODEL_M0][:, j2])))),
        ))

    del fitted, tr, ev
    print(f"[STAGE] window {w['name']} done "
          f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)
    return dict(window=w["name"], model_metrics=model_metrics,
                opt_rows=opt_rows, boots=boots, bysym=bysym,
                target_rows=target_rows, count_occ_rows=count_occ_rows,
                decom_rows=decom_rows,
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
    cur = prev.add_zt_causal(cur)
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
    parity_audit = dict(
        parent_commit=BASE_SHA, p_1a2_commit=P_1A2_SHA,
        grandparent_commit=P_1A1B_SHA,
        rows_ok=bool(len(cur) == EXPECTED_TRANSITIONS),
        z_feature_hash_ok=bool(h == FROZEN_Z_FEATURE_HASH),
        z_feature_hash_expected=FROZEN_Z_FEATURE_HASH,
        z_feature_hash_actual=h,
        joint_nll_parity_1a1b={}, joint_nll_parity_1a2={},
        joint_nll_parity_1a2b={}, joint_nll_tol=PARITY_TOL,
    )
    if not (parity_audit["rows_ok"] and parity_audit["z_feature_hash_ok"]):
        raise SystemExit(
            f"STOP_DYNAMIC_PGM1A2C_SAMPLE_PARITY_FAIL: {parity_audit}")

    # ---------------- deterministic recoverability audit ----------------
    cls, zt_class = audit_zt_recoverability(cur)
    print(f"[AUDIT] Z_t split: reencoding={zt_class['n_current_state_reencoding']}"
          f" true_innovation={zt_class['n_true_history_innovation']}", flush=True)

    # ---------------- invariants + reconstruction ----------------
    invariants = dict(
        tv_update_exact=bool(np.allclose(
            (nxt["path_total_variation_R"].to_numpy(float)
             - cur["path_total_variation_R"].to_numpy(float)),
            np.abs(cur["z_d_up"].to_numpy(float)), atol=1e-8)),
        last_return_exact=bool(np.allclose(
            nxt["path_last_return_R"].to_numpy(float),
            -cur["z_d_up"].to_numpy(float), atol=1e-8)),
        no_tb4=bool("TB4" not in set(cur["block"].unique())),
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

    (base.OUT / f"{PREFIX}_representation_audit.json").write_text(
        json.dumps(dict(zt_classification=zt_class, sample=sample_audit,
                        support=support_audit,
                        reconstruction=full_recon), indent=2, default=str))

    del nxt

    if os.environ.get("DYNAMIC_PGM1A2C_AUDIT_ONLY") == "1":
        (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
            json.dumps(parity_audit, indent=2, default=str))
        print("[AUDIT-ONLY] Complete. Stopping before any model fit.",
              flush=True)
        return dict(parity=parity_audit, zt_classification=zt_class,
                    sample=sample_audit)

    # ---------------- persist cache ----------------
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
    target_rows, count_occ_rows, decom_rows = [], [], []
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
        target_rows += res["target_rows"]
        count_occ_rows += res.get("count_occ_rows", [])
        decom_rows += res.get("decom_rows", [])
        n_symbols = max(n_symbols, int(res.get("n_symbols", 0)))
        print(f"[STAGE] window {w['name']} subprocess done "
              f"took={round(time.perf_counter() - t_w, 2)}s", flush=True)

    # ---------------- parity gates (three ancestors) ----------------
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
        f3 = FROZEN_1A2B_JOINT[w["name"]]
        dmz = abs(got[MODEL_MZ] - f3["MZ_STATE_INNOV"])
        parity_audit["joint_nll_parity_1a2b"][w["name"]] = dict(
            mz_abs_diff=dmz, ok=bool(dmz < PARITY_TOL))
        if (d0 >= PARITY_TOL or d1 >= PARITY_TOL or dm0 >= PARITY_TOL
                or dmf >= PARITY_TOL or dmz >= PARITY_TOL):
            raise SystemExit(
                f"STOP_DYNAMIC_PGM1A2C_PARITY_FAIL: {w['name']} "
                f"dK0={d0} dK1={d1} dM0={dm0} dMF={dmf} dMZ={dmz}")

    # ---------------- verdict: PRIMARY is MZ - MC ----------------
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
        rmem = _row(w["name"], C_MEM)
        rre = _row(w["name"], C_REENC)
        rinnov = _row(w["name"], C_INNOV)
        windows_report[w["name"]] = dict(
            delta_mz_minus_mc=rmem["delta_sample_mean"],
            mem_ci_lo=rmem["ci_lo"], mem_ci_hi=rmem["ci_hi"],
            n_symbols_negative=_sym_neg(w["name"], C_MEM),
            n_symbols_total=n_symbols,
            delta_mc_minus_m0=rre["delta_sample_mean"],
            reenc_ci_lo=rre["ci_lo"], reenc_ci_hi=rre["ci_hi"],
            delta_mz_minus_m0=rinnov["delta_sample_mean"],
            passes=_passes(w["name"], C_MEM))
    if all(v["passes"] for v in windows_report.values()):
        verdict = "RECENT_INNOVATION_MEMORY_SUPPORTED"
    else:
        verdict = "RECENT_INNOVATION_MEMORY_NOT_SUPPORTED_GAIN_IS_REENCODING"

    # ---------------- outputs ----------------
    pd.DataFrame(model_metrics).to_csv(
        base.OUT / f"{PREFIX}_model_metrics.csv", index=False)
    pd.DataFrame(boots).to_csv(
        base.OUT / f"{PREFIX}_bootstrap.csv", index=False)
    pd.DataFrame(bysym).to_csv(
        base.OUT / f"{PREFIX}_by_symbol.csv", index=False)
    pd.DataFrame(target_rows).to_csv(
        base.OUT / f"{PREFIX}_target_metrics.csv", index=False)
    pd.DataFrame(decom_rows).to_csv(
        base.OUT / f"{PREFIX}_gain_decomposition.csv", index=False)
    pd.DataFrame(count_occ_rows).to_csv(
        base.OUT / f"{PREFIX}_count_occurrence.csv", index=False)
    pd.DataFrame(opt_rows).to_csv(
        base.OUT / f"{PREFIX}_optimizer_audit.csv", index=False)
    (base.OUT / f"{PREFIX}_parity_audit.json").write_text(
        json.dumps(parity_audit, indent=2, default=str))

    summary = dict(
        experiment="DYNAMIC-PGM-1A.2c Representation Control",
        parent_commit=BASE_SHA,
        question=("After controlling for current-state nonlinear re-encoding, "
                  "does the recent innovation still carry genuine history?"),
        verdict=verdict,
        primary_comparison=C_MEM,
        models=dict(
            M0="P(Z|S_t,A_t)", MF="P(Z|S_t,A_t,S_{t-1})",
            MZ="P(Z|S_t,A_t,Z_t)", MC="P(Z|S_t,A_t,Z_t^{cur})  [control]",
            MMEM="P(Z|S_t,A_t,Z_t^{mem}) [explanatory]"),
        zt_classification=zt_class,
        windows=windows_report,
        gain_decomposition=decom_rows,
        per_target=target_rows,
        model_metrics=model_metrics,
        bootstrap=boots,
        sample=sample_audit,
        parity=parity_audit,
        count_magnitude_closure=dict(
            magnitude=("state-independent train-constant exact ZTP shared by "
                       "all models"),
            occurrence="state-dependent Logistic"),
        bootstrap_reps=base.BOOTSTRAP_REPS,
    )
    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)
    summary["timing"] = timing
    (base.OUT / f"{PREFIX}_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    print(f"[MODEL METRICS]\n{pd.DataFrame(model_metrics).to_string(index=False)}")
    print(f"[BOOTSTRAP]\n{pd.DataFrame(boots).to_string(index=False)}")
    print(f"[GAIN DECOMPOSITION]\n{pd.DataFrame(decom_rows).to_string(index=False)}")
    print(f"[VERDICT] {verdict}")
    print("[PRIMARY MZ-MC] " + ", ".join(
        f"{k}: {v['delta_mz_minus_mc']:.5f} "
        f"CI=[{v['mem_ci_lo']:.5f},{v['mem_ci_hi']:.5f}] "
        f"neg={v['n_symbols_negative']}/{v['n_symbols_total']}"
        for k, v in windows_report.items()))
    print("[REENCODING MC-M0] " + ", ".join(
        f"{k}: {v['delta_mc_minus_m0']:.5f}" for k, v in windows_report.items()))
    print(f"[DONE] -> {base.OUT}")
    return summary


if __name__ == "__main__":
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--window-json")
    _ap.add_argument("--data")
    _ap.add_argument("--result")
    _ap.add_argument("--agezero-deterministic", action="store_true")
    _ap.add_argument("--audit-only", action="store_true",
                     help="Run the deterministic Z_t recoverability audit, "
                          "sample parity and reconstruction, then stop.")
    _a = _ap.parse_args()
    if _a.audit_only:
        os.environ["DYNAMIC_PGM1A2C_AUDIT_ONLY"] = "1"
    if _a.window_json:
        if not _a.agezero_deterministic:
            raise SystemExit(
                "STOP_DYNAMIC_PGM1A2C_CHILD_AGEZERO_STATE_NOT_PROPAGATED")
        base.configure_child_semantics(agezero_deterministic=True)
        if base.disc_spec() != []:
            raise SystemExit("STOP_DYNAMIC_PGM1A2C_CHILD_DISC_SPEC_NONEMPTY")
        _w = json.loads(_a.window_json)
        Path(_a.result).write_text(
            json.dumps(run_single_window_1a2c(_w, _a.data), default=str))
    else:
        main()
