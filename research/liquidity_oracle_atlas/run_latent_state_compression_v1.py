"""Stage 4B — Latent-State Compression Gate v1.0

研究目的（reviewer 规格）：
  1) 状态本身能不能低维压缩（Reaction PCA / Liquidity PCA + NMF secondary）；
  2) 压缩后经济信息有没有丢（当前 action outcome + WAIT value）；
  3) 是否真的有证据值得进入 nonlinear compression。

本实验不修改 Stage 4A 的 geometry / stop / target / horizon / scale 定义。
几何核（surviving_field_and_target / latest_confirmed_extreme /
first_hit_bounds / build_post）全部直接 import Stage 4A 权威实现。

与 Stage 4A 的唯一结构性差异（必要且已加复现断言）：
  Stage 4A 的 action surface 只含 TB2/TB3/TB4（TB1 被 dropna 丢弃）；
  本实验的 WF 方案要求 train-only 拟合（WF1: TB1→TB2），因此本地构造
  全 block(TB1..TB4) surface，TB1 标为 WF0（仅训练）。
  对 TB2-TB4 子集用 Stage 4A 的 aggregate_surface 与既有
  action_surface_by_wf.csv 做 HARD 复现断言。

样本单位：state = contact × decision_horizon（不含 action/scale 重复）。
Geometry 永不参与 PCA。
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import subspace_angles
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr
from sklearn.decomposition import NMF, PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (brier_score_loss, log_loss, mean_absolute_error,
                             r2_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.liquidity_oracle_atlas import run_liquidity_field_action_surface_v1 as s4a

OUT = Path("research/analysis_results/latent_state_compression_v1")
OUT.mkdir(parents=True, exist_ok=True)
S4A_OUT = Path("research/analysis_results/liquidity_field_action_surface_v1")
REACT_CSV = S4A_OUT.parent / "liquidity_field_reaction_model_v1" / "reaction_episode_features.csv"
FIELD_CSV = S4A_OUT.parent / "liquidity_field_reaction_model_v1" / "liquidity_field_snapshot.csv"
REF_SURF = S4A_OUT / "action_surface_by_wf.csv"

BASE_COMMIT = s4a.BASE_COMMIT
HORIZONS = s4a.HORIZONS                       # [1,2,3,5,8,13]
STRUCTURE_SCALES = s4a.STRUCTURE_SCALES
ACTIONS = s4a.ACTIONS
EVAL_BARS = s4a.EVAL_BARS
T_POST = s4a.T_POST
CHUNK = s4a.CHUNK

# block -> wf 标签（TB1 仅训练）
BLOCK_TO_WF = {"TB1": "WF0", "TB2": "WF1", "TB3": "WF2", "TB4": "WF3"}
TEST_WF = ["WF1", "WF2", "WF3"]
TEST_BLOCK_OF_WF = {"WF1": "TB2", "WF2": "TB3", "WF3": "TB4"}
TRAIN_WF_OF = {"WF1": ["WF0"], "WF2": ["WF0", "WF1"], "WF3": ["WF0", "WF1", "WF2"]}

FIELD_KEY = ["symbol", "liquidity_id", "contact_number"]
REACTION_KEY = FIELD_KEY + ["h"]
STATE_KEY = REACTION_KEY                       # contact × decision_horizon

# --- Reaction block（decision h 已可见的 prefix 特征；禁 future） ---
REACTION_FEATURES = [
    "path_efficiency", "total_variation_atr", "amplitude_atr",
    "cross_count", "boundary_occupancy", "boundary_touch_bars",
    "dc_pivots_0p20", "dc_pivots_0p40", "dc_pivots_0p80", "dc_pivots_1p20",
    "outward_mfe_atr", "inward_mae_atr",
]
REACTION_LOG1P = ["cross_count", "boundary_touch_bars", "dc_pivots_0p20",
                  "dc_pivots_0p40", "dc_pivots_0p80", "dc_pivots_1p20"]
# 禁止入模（未来信息）
FORBIDDEN_REACTION = ["persistent_escape", "persistent_escape_bar",
                      "post_escape_efficiency", "retracement_ratio",
                      "fp_out_0.25", "fp_out_0.50", "fp_out_1.00",
                      "fp_in_0.25", "fp_in_0.50", "fp_in_1.00",
                      "target_first", "future_target_hit"]

# --- Liquidity block ---
LAMBDAS = ["0p5", "1p0", "2p0", "4p0"]
LIQUIDITY_FEATURES = ([f"density_{l}" for l in LAMBDAS]
                      + [f"imbalance_{l}" for l in LAMBDAS]
                      + ["room_up_atr", "room_down_atr", "depletion_share"])
# DEFERRED（见 audit）：remaining_liq_imbalance_1p0 需要新的 surviving-field
# imbalance 定义（decision h 参考价 + 带权强度），不在 frozen snapshot 合同内。
DEFERRED_FEATURES = {"remaining_liq_imbalance_1p0"}

# NMF 非负子空间
NMF_FEATURES = ([f"density_{l}" for l in LAMBDAS]
                + [f"liq_intensity_up_{l}" for l in LAMBDAS]
                + [f"liq_intensity_dn_{l}" for l in LAMBDAS])

# NMF 为控制成本使用固定子样本（随机种子固定，非 outcome 选择）
NMF_SUBSAMPLE = 150_000
NMF_KS = [2, 3, 4]


# ===========================================================================
# 1. 全 block action surface（几何核复用 Stage 4A，仅保留 TB1 为训练块）
# ===========================================================================
def compute_action_surface_all_blocks(D, master_by_sym, bars_by_sym):
    F = D["F"].copy().reset_index(drop=True)
    bm = s4a._assign_blocks(D["F"])
    days = pd.to_datetime(F["decision_time"]).dt.normalize()
    F["block"] = days.map(bm).astype(str)
    F["wf"] = F["block"].map(BLOCK_TO_WF)
    assert F["block"].isin(list(BLOCK_TO_WF)).all(), "BLOCK_ASSIGN_FAIL"

    syms = sorted(F["symbol"].unique())
    frames, gid_maps, gid = [], [], 0
    for sym in syms:
        sub = F[F["symbol"] == sym].reset_index(drop=True)
        if len(sub) == 0:
            continue
        bars = bars_by_sym.get(sym)
        ms = master_by_sym.get(sym)
        if bars is None or ms is None:
            continue
        post = s4a.build_post(bars, sub)
        N = len(sub)
        mp = ms["price"].to_numpy(float)
        mav = pd.to_datetime(ms["available_time"]).to_numpy()
        mfp_raw = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
        cid_str = (sub["symbol"].to_numpy().astype(str) + "|"
                   + sub["liquidity_id"].to_numpy().astype(str) + "|"
                   + sub["contact_number"].to_numpy().astype(str))
        gids = np.arange(gid, gid + N)
        gid_maps.append(pd.DataFrame(dict(
            gid=gids, contact_id=cid_str,
            symbol=sub["symbol"].to_numpy().astype(str),
            liquidity_id=sub["liquidity_id"].to_numpy().astype(str),
            contact_number=sub["contact_number"].to_numpy().astype("int64"))))
        gid += N
        wf_c_all = sub["wf"].to_numpy()
        blk_c_all = sub["block"].to_numpy()

        for st in range(0, N, CHUNK):
            en = min(st + CHUNK, N)
            c = en - st
            dtc = pd.to_datetime(sub["decision_time"].to_numpy()[st:en]).to_numpy()
            av_le = mav[None, :] <= dtc[:, None]
            mfp = mfp_raw
            fp_gt = np.isnat(mfp)[None, :] | (mfp[None, :] > dtc[:, None])
            active = av_le & fp_gt
            side_c = post["side"][st:en]
            boundary_c = post["boundary"][st:en]
            atr0_c = post["atr0"][st:en]
            O_c = post["O"][st:en]
            H_c = post["H"][st:en]
            L_c = post["L"][st:en]
            z_low_c = post["z_low"][st:en]
            z_high_c = post["z_high"][st:en]
            disc_c = post["disc"][st:en]
            wf_blk = wf_c_all[st:en]
            blk = blk_c_all[st:en]
            gids_c = gids[st:en]
            for h in HORIZONS:
                cml = post["cum_min_low"][st:en, h - 1]
                cmh = post["cum_max_high"][st:en, h - 1]
                entry = O_c[:, h]
                fh = H_c[:, h:h + EVAL_BARS].copy()
                fl = L_c[:, h:h + EVAL_BARS].copy()
                wdisc = disc_c[:, h:h + EVAL_BARS]
                fbad = np.flatnonzero(np.any(wdisc, axis=1))
                if len(fbad):
                    first_bad = np.argmax(wdisc[fbad], axis=1)
                    for r, fb in zip(fbad, first_bad):
                        fh[r, fb + 1:] = np.nan
                        fl[r, fb + 1:] = np.nan
                entry_valid = (~disc_c[:, h]) & np.isfinite(entry)
                for ai, action in enumerate(ACTIONS):
                    d = side_c if ai == 0 else -side_c
                    target, alive = s4a.surviving_field_and_target(
                        mp, active, entry, cml, cmh, d)
                    for scale in STRUCTURE_SCALES:
                        stop_z = s4a.latest_confirmed_extreme(
                            z_low_c[:, :h], z_high_c[:, :h], int(h), scale, d)
                        stop_abs = boundary_c + stop_z * atr0_c
                        risk = np.abs(entry - stop_abs)
                        reward = np.abs(target - entry)
                        rr = reward / np.maximum(risk, 1e-12)
                        struct_ok = np.isfinite(stop_z)
                        has_tgt = np.isfinite(target)
                        geo_ok = (has_tgt & struct_ok & entry_valid
                                  & np.isfinite(stop_abs)
                                  & (np.abs(target - entry) > 1e-9)
                                  & (np.abs(stop_abs - entry) > 1e-9))
                        tgt_use = np.where(geo_ok, target, np.nan)
                        stop_use = np.where(geo_ok, stop_abs, np.nan)
                        oc = s4a.first_hit_bounds(fh, fl, entry, tgt_use,
                                                  stop_use, d)
                        geook = geo_ok
                        frames.append(pd.DataFrame(dict(
                            gid=gids_c, wf=wf_blk, block=blk,
                            h=np.full(c, int(h), dtype=np.int64),
                            scale=np.full(c, float(scale)),
                            action=np.full(c, action),
                            d=d.astype(int),
                            rr=np.where(geook, rr, np.nan),
                            target_atr=np.where(geook, reward / atr0_c, np.nan),
                            risk_atr=np.where(geook, risk / atr0_c, np.nan),
                            entry_atr=np.where(geook, entry / atr0_c, np.nan),
                            target_first=oc["target_first"],
                            stop_first=oc["stop_first"],
                            ambiguous=oc["ambiguous"],
                            censored=oc["censored"],
                            R_lower=np.where(geook, oc["R_lower"], np.nan),
                            R_upper=np.where(geook, oc["R_upper"], np.nan),
                            available=geook,
                            remaining_count=np.asarray(alive).sum(axis=1).astype(int),
                            n_active=np.asarray(active).sum(axis=1).astype(int))))
    df = pd.concat(frames, ignore_index=True)
    gmap = pd.concat(gid_maps, ignore_index=True)
    return df, gmap


def assert_reproduce_stage4a(df: pd.DataFrame) -> dict:
    """HARD：TB2-TB4 子集必须复现 Stage 4A 已发布聚合表。"""
    ref = pd.read_csv(REF_SURF)
    cur = s4a.aggregate_surface(
        df[df["wf"].isin(TEST_WF)].rename(columns={"scale": "scale"}))
    m = cur.merge(ref, on=["wf", "h", "structure_scale", "action"],
                  suffixes=("_new", "_ref"))
    assert len(m) == len(ref) == len(cur), (
        f"[FATAL] REPRODUCE_CELL_COUNT_MISMATCH cur={len(cur)} "
        f"ref={len(ref)} merged={len(m)}")
    bad = {}
    for col in ["n_contacts", "n_action_available", "availability",
                "median_target_atr", "median_risk_atr", "median_RR",
                "target_first_rate", "stop_first_rate",
                "same_bar_ambiguous_rate", "censored_rate",
                "E_R_lower", "E_R_upper"]:
        a = m[f"{col}_new"].to_numpy(float)
        b = m[f"{col}_ref"].to_numpy(float)
        both_nan = np.isnan(a) & np.isnan(b)
        d = np.where(both_nan, 0.0, np.abs(np.nan_to_num(a) - np.nan_to_num(b)))
        if d.max() > 1e-9:
            bad[col] = float(d.max())
    assert not bad, f"[FATAL] STAGE4A_REPRODUCTION_FAIL: {bad}"
    return dict(n_cells=len(m), max_abs_diff=0.0)


# ===========================================================================
# 2. State 表：contact × decision_horizon（唯一）
# ===========================================================================
def _norm_keys(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["symbol"] = d["symbol"].astype(str)
    d["liquidity_id"] = d["liquidity_id"].astype(str)
    d["contact_number"] = d["contact_number"].astype("int64")
    return d


def build_state_table(D, df, gmap) -> pd.DataFrame:
    F = _norm_keys(D["F"])
    bm = s4a._assign_blocks(D["F"])
    days = pd.to_datetime(F["decision_time"]).dt.normalize()
    F["block"] = days.map(bm).astype(str)
    F["wf"] = F["block"].map(BLOCK_TO_WF)
    base = F[FIELD_KEY + ["block", "wf"]].drop_duplicates(FIELD_KEY)
    base = base.reset_index(drop=True)
    H = pd.DataFrame({"h": np.array(HORIZONS, dtype="int64")})
    state = base.merge(H, how="cross")

    # ---- reaction（prefix，decision h 可见）----
    react = _norm_keys(pd.read_csv(REACT_CSV))
    if "horizon" in react.columns and "h" not in react.columns:
        react = react.rename(columns={"horizon": "h"})
    react["h"] = react["h"].astype("int64")
    react = react[react["h"].isin(np.array(HORIZONS, dtype="int64").tolist())]
    leaks = [c for c in FORBIDDEN_REACTION if c in react.columns]
    assert not react.duplicated(REACTION_KEY).any(), "REACTION_KEY_NOT_UNIQUE"
    rj = react[REACTION_KEY + REACTION_FEATURES].copy()
    rj["_r"] = 1                       # join 存在性指示（与特征 NaN 无关）
    state = state.merge(rj, on=REACTION_KEY, how="left")
    match_react = float(state["_r"].notna().mean())
    state = state.drop(columns=["_r"])

    # ---- liquidity（t0 frozen field）----
    field = _norm_keys(pd.read_csv(FIELD_CSV))
    assert not field.duplicated(FIELD_KEY).any(), "FIELD_KEY_NOT_UNIQUE"
    fld_cols = (["room_up_atr", "room_down_atr", "field_position"]
                + [f"liq_intensity_up_{l}" for l in LAMBDAS]
                + [f"liq_intensity_dn_{l}" for l in LAMBDAS])
    fj = field[FIELD_KEY + fld_cols].copy()
    fj["_f"] = 1
    state = state.merge(fj, on=FIELD_KEY, how="left")
    match_field = float(state["_f"].notna().mean())
    state = state.drop(columns=["_f"])
    for l in LAMBDAS:
        up = state[f"liq_intensity_up_{l}"].to_numpy(float)
        dn = state[f"liq_intensity_dn_{l}"].to_numpy(float)
        state[f"density_{l}"] = np.log1p(up + dn)
        state[f"imbalance_{l}"] = (up - dn) / (up + dn + 1e-12)
    # ---- depletion_share（action/scale 无关，仅依赖 contact × h）----
    dep = df.drop_duplicates(["gid", "h"])[
        ["gid", "h", "remaining_count", "n_active"]]
    dep = dep.merge(gmap, on="gid", how="left")
    assert not dep.duplicated(REACTION_KEY).any(), "DEPLETION_KEY_NOT_UNIQUE"
    dj = dep[REACTION_KEY + ["remaining_count", "n_active"]].copy()
    dj["_d"] = 1
    state = state.merge(dj, on=REACTION_KEY, how="left")
    match_dep = float(state["_d"].notna().mean())
    state = state.drop(columns=["_d"])
    na = state["n_active"].to_numpy(float)
    rc = state["remaining_count"].to_numpy(float)
    state["depletion_share"] = 1.0 - np.where(
        na > 0, rc / np.maximum(na, 1e-12), np.nan)

    assert not state.duplicated(STATE_KEY).any(), "STATE_KEY_NOT_UNIQUE"
    for name, rate in [("reaction", match_react), ("field", match_field),
                       ("depletion", match_dep)]:
        assert rate == 1.0, f"[FATAL] JOIN_MATCH_RATE_FAIL {name}={rate}"
    return state, dict(reaction=match_react, field=match_field,
                       depletion=match_dep, leaks_present=leaks,
                       n_state=len(state))


# ===========================================================================
# 3. PCA（train-only）
# ===========================================================================
def fit_pca(state, feats, log_cols, train_mask, max_k=8):
    X = state[feats].to_numpy(np.float64).copy()
    if log_cols:
        idx = [feats.index(c) for c in log_cols]
        X[:, idx] = np.log1p(np.clip(X[:, idx], 0.0, None))
    X[~np.isfinite(X)] = np.nan
    med = np.nanmedian(X[train_mask], axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    X = np.where(np.isfinite(X), X, med)
    sc = StandardScaler().fit(X[train_mask])
    Z = sc.transform(X)
    k = min(len(feats), max_k)
    pca = PCA(n_components=k, svd_solver="full")
    pca.fit(Z[train_mask])
    scores = pca.transform(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    k80 = int(np.searchsorted(cum, 0.80) + 1)
    k90 = int(np.searchsorted(cum, 0.90) + 1)
    k95 = int(np.searchsorted(cum, 0.95) + 1)
    return dict(scores=scores, loadings=pca.components_, evr=evr, cum=cum,
                k80=k80, k90=k90, k95=k95, k_primary=min(k90, 6), k_fit=k)


def pca_variance_rows(block, res_by_wf, feats):
    rows = []
    for wf, r in res_by_wf.items():
        for i, e in enumerate(r["evr"]):
            rows.append(dict(block=block, wf=wf, component=i + 1,
                             explained_variance_ratio=float(e),
                             cumulative_variance=float(r["cum"][i])))
    return pd.DataFrame(rows)


def pca_loading_rows(block, res_by_wf, feats):
    rows = []
    for wf, r in res_by_wf.items():
        L = r["loadings"]
        for i in range(L.shape[0]):
            for j, f in enumerate(feats):
                rows.append(dict(block=block, wf=wf, component=i + 1,
                                 feature=f, loading=float(L[i, j])))
    return pd.DataFrame(rows)


def stability_rows(block, res_by_wf, feats, k=3):
    """Hungarian 匹配 + principal angles（禁止简单 PC1↔PC1）。"""
    out = []
    wfs = [w for w in ["WF1", "WF2", "WF3"] if w in res_by_wf]
    for a in range(len(wfs)):
        for b in range(a + 1, len(wfs)):
            wa, wb = wfs[a], wfs[b]
            La = res_by_wf[wa]["loadings"][:k].T      # (p, k)
            Lb = res_by_wf[wb]["loadings"][:k].T
            na = La / np.maximum(np.linalg.norm(La, axis=0), 1e-12)
            nb = Lb / np.maximum(np.linalg.norm(Lb, axis=0), 1e-12)
            S = np.abs(na.T @ nb)                      # (k, k) |cos|
            ri, ci = linear_sum_assignment(-S)
            ang = np.degrees(subspace_angles(La, Lb))
            for i, j in zip(ri, ci):
                out.append(dict(
                    block=block, wf_a=wa, wf_b=wb, k=k,
                    component_a=int(i + 1), component_b=int(j + 1),
                    abs_cosine=float(S[i, j]),
                    max_principal_angle_deg=float(ang.max()),
                    mean_principal_angle_deg=float(ang.mean())))
            out.append(dict(block=block, wf_a=wa, wf_b=wb, k=k,
                            component_a=0, component_b=0,
                            abs_cosine=float(S[ri, ci].mean()),
                            max_principal_angle_deg=float(ang.max()),
                            mean_principal_angle_deg=float(ang.mean())))
    return pd.DataFrame(out)


# ===========================================================================
# 4. NMF（secondary，非负子空间）
# ===========================================================================
def run_nmf_for_wf(state, wf, train_mask):
    X = state.loc[train_mask, NMF_FEATURES].to_numpy(np.float64).copy()
    X[~np.isfinite(X)] = np.nan
    X = np.where(np.isfinite(X), X, 0.0)
    X = np.clip(X, 0.0, None)
    if len(X) > NMF_SUBSAMPLE:
        rs = np.random.RandomState(42)
        idx = rs.choice(len(X), size=NMF_SUBSAMPLE, replace=False)
        Xs = X[idx]
    else:
        Xs = X
    mx = Xs.max()
    if mx <= 0:
        return [], []
    Xs = Xs / mx
    out = []
    recon = []
    for k in NMF_KS:
        nm = NMF(n_components=k, random_state=42, max_iter=1000, init="nndsvd")
        W = nm.fit_transform(Xs)
        H = nm.components_
        err = float(nm.reconstruction_err_)
        for i in range(k):
            d = dict(wf=wf, k=k, component=i + 1, reconstruction_error=err,
                     n_rows=int(len(Xs)))
            for j, f in enumerate(NMF_FEATURES):
                d[f] = float(H[i, j])
            out.append(d)
        recon.append(dict(wf=wf, k=k, reconstruction_error=err,
                          n_rows=int(len(Xs))))
    return out, recon


# ===========================================================================
# 5. 模型：当前 action outcome（LogisticRegression，固定超参）
# ===========================================================================
GEO_COLS = ["target_distance_atr", "risk_distance_atr", "rr", "h",
            "structure_scale", "action_is_outward"]


def run_action_models(df, gmap, state, res_r, res_l):
    ev = df[df["available"] & (~df["ambiguous"]) & (~df["censored"])].copy()
    ev = ev.merge(gmap, on="gid", how="left")
    ev["target_distance_atr"] = ev["target_atr"].astype(float)
    ev["risk_distance_atr"] = ev["risk_atr"].astype(float)
    ev["structure_scale"] = ev["scale"].astype(float)
    ev["action_is_outward"] = (ev["action"] == "OUTWARD").astype(float)
    coverage = dict(overall=len(ev) / len(df))

    rows, cov_rows = [], []
    for wf in TEST_WF:
        test_block = TEST_BLOCK_OF_WF[wf]
        train_wf = TRAIN_WF_OF[wf]
        needed = set(train_wf) | {wf}
        sub = ev[ev["wf"].isin(needed)]
        rpc = [f"rpc{j+1}_{wf}" for j in range(res_r[wf]["k_primary"])]
        lpc = [f"lpc{j+1}_{wf}" for j in range(res_l[wf]["k_primary"])]
        cols = (REACTION_FEATURES + LIQUIDITY_FEATURES + rpc + lpc)
        sub = sub.merge(state[STATE_KEY + cols], on=STATE_KEY, how="left")
        tr = sub[sub["wf"].isin(train_wf)]
        te = sub[sub["wf"] == wf]
        cov_rows.append(dict(wf=wf, test_block=test_block,
                             n_test_rows=len(te),
                             n_test_evaluable=len(te),
                             n_train_evaluable=len(tr),
                             evaluable_coverage_overall=coverage["overall"]))
        ytr = tr["target_first"].astype(int).to_numpy()
        yte = te["target_first"].astype(int).to_numpy()
        specs = {
            "M0": GEO_COLS,
            "M1": GEO_COLS + REACTION_FEATURES,
            "M2": GEO_COLS + rpc,
            "M3": GEO_COLS + LIQUIDITY_FEATURES,
            "M4": GEO_COLS + lpc,
            "M5": GEO_COLS + rpc + lpc,
        }
        for name, feats in specs.items():
            Xtr = tr[feats].to_numpy(np.float64)
            Xte = te[feats].to_numpy(np.float64)
            Xtr[~np.isfinite(Xtr)] = 0.0
            Xte[~np.isfinite(Xte)] = 0.0
            sc = StandardScaler().fit(Xtr)
            clf = LogisticRegression(max_iter=200, C=1.0)
            clf.fit(sc.transform(Xtr), ytr)
            p = clf.predict_proba(sc.transform(Xte))[:, 1]
            rows.append(dict(wf=wf, model=name, n_train=len(tr),
                             n_test=len(te),
                             roc_auc=float(roc_auc_score(yte, p)),
                             log_loss=float(log_loss(yte, p)),
                             brier=float(brier_score_loss(yte, p))))
        print(f"  [ACTION] {wf} done (train={len(tr)}, test={len(te)})")
    info = pd.DataFrame(rows)
    piv = info.pivot_table(index="wf", columns="model", values="roc_auc")
    deltas = []
    for wf in TEST_WF:
        a = piv.loc[wf]
        deltas.append(dict(wf=wf, block="action",
                           d_M1_M0=float(a["M1"] - a["M0"]),
                           d_M2_M0=float(a["M2"] - a["M0"]),
                           d_M1_M2=float(a["M1"] - a["M2"]),
                           d_M3_M0=float(a["M3"] - a["M0"]),
                           d_M4_M0=float(a["M4"] - a["M0"]),
                           d_M3_M4=float(a["M3"] - a["M4"]),
                           d_M5_M0=float(a["M5"] - a["M0"])))
    return info, pd.DataFrame(deltas), pd.DataFrame(cov_rows)


# ===========================================================================
# 6. WAIT-value 模型（Ridge，固定 alpha）
# ===========================================================================
WAIT_TARGETS = ["delta_E_R_lower", "delta_E_R_upper", "delta_RR",
                "delta_target_distance_atr", "delta_risk_distance_atr"]
WAIT_GEO = ["target_atr_b", "risk_atr_b", "rr_b", "structure_scale",
            "base_h", "action_is_outward", "later_h"]


def run_wait_models(df, gmap, state, res_r, res_l):
    agg, raw = s4a._matched_waiting(df, gmap, "first")
    w = raw.copy()
    w["action_is_outward"] = (w["action"] == "OUTWARD").astype(float)
    w = w.rename(columns={"base_h": "h"})
    rows, delta_tf = [], []
    for wf in TEST_WF:
        train_wf = TRAIN_WF_OF[wf]
        needed = set(train_wf) | {wf}
        sub_all = w[w["wf"].isin(needed)]
        rpc = [f"rpc{j+1}_{wf}" for j in range(res_r[wf]["k_primary"])]
        lpc = [f"lpc{j+1}_{wf}" for j in range(res_l[wf]["k_primary"])]
        cols = REACTION_FEATURES + LIQUIDITY_FEATURES + rpc + lpc
        sub = sub_all.merge(state[STATE_KEY + cols], on=STATE_KEY, how="left")
        sub = sub.rename(columns={"h": "base_h"})
        tr = sub[sub["wf"].isin(train_wf)]
        te = sub[sub["wf"] == wf]
        delta_tf.append(dict(wf=wf, n=len(te),
                             mean_delta_target_first=float(
                                 te["delta_target_first"].mean()),
                             mean_delta_E_R_lower=float(
                                 te["delta_E_R_lower"].mean())))
        specs = {
            "W0": WAIT_GEO,
            "W1": WAIT_GEO + REACTION_FEATURES,
            "W2": WAIT_GEO + rpc,
            "W3": WAIT_GEO + LIQUIDITY_FEATURES,
            "W4": WAIT_GEO + lpc,
            "W5": WAIT_GEO + rpc + lpc,
        }
        for tgt in WAIT_TARGETS:
            mtr = tr[tgt].to_numpy(float)
            mte = te[tgt].to_numpy(float)
            ok_tr = np.isfinite(mtr)
            ok_te = np.isfinite(mte)
            for name, feats in specs.items():
                Xtr = tr[feats].to_numpy(np.float64)[ok_tr]
                Xte = te[feats].to_numpy(np.float64)[ok_te]
                Xtr[~np.isfinite(Xtr)] = 0.0
                Xte[~np.isfinite(Xte)] = 0.0
                sc = StandardScaler().fit(Xtr)
                m = Ridge(alpha=1.0)
                m.fit(sc.transform(Xtr), mtr[ok_tr])
                p = m.predict(sc.transform(Xte))
                yt = mte[ok_te]
                sp = float(spearmanr(yt, p).statistic) if len(yt) > 2 else np.nan
                rows.append(dict(wf=wf, target=tgt, model=name,
                                 n_train=int(ok_tr.sum()), n_test=len(yt),
                                 r2=float(r2_score(yt, p)),
                                 mae=float(mean_absolute_error(yt, p)),
                                 spearman=sp))
        print(f"  [WAIT] {wf} done (train={len(tr)}, test={len(te)})")
    info = pd.DataFrame(rows)
    prim = info[info["target"] == "delta_E_R_lower"]
    piv = prim.pivot_table(index="wf", columns="model", values="r2")
    pivs = prim.pivot_table(index="wf", columns="model", values="spearman")
    deltas = []
    for wf in TEST_WF:
        a, s = piv.loc[wf], pivs.loc[wf]
        deltas.append(dict(wf=wf, block="wait",
                           d_W1_W0_r2=float(a["W1"] - a["W0"]),
                           d_W2_W0_r2=float(a["W2"] - a["W0"]),
                           d_W1_W2_r2=float(a["W1"] - a["W2"]),
                           d_W3_W0_r2=float(a["W3"] - a["W0"]),
                           d_W4_W0_r2=float(a["W4"] - a["W0"]),
                           d_W3_W4_r2=float(a["W3"] - a["W4"]),
                           d_W5_W0_r2=float(a["W5"] - a["W0"]),
                           d_W1_W0_spearman=float(s["W1"] - s["W0"]),
                           d_W1_W2_spearman=float(s["W1"] - s["W2"]),
                           d_W3_W0_spearman=float(s["W3"] - s["W0"]),
                           d_W3_W4_spearman=float(s["W3"] - s["W4"])))
    return info, pd.DataFrame(deltas), pd.DataFrame(delta_tf)


# ===========================================================================
# 7. 报告
# ===========================================================================
def _fmt(df, n=40):
    return "```\n" + df.head(n).to_string(index=False) + "\n```\n"


def write_report(path, manifest, res_r, res_l, vr, vl, action_info,
                 action_delta, wait_info, wait_delta, wait_tf, stab_r,
                 stab_l, nmf_c, gate, join_rate, repro):
    L = []
    A = L.append
    A("# Stage 4B — Latent-State Compression Gate v1.0\n")
    A("样本单位：**contact × decision_horizon**（不含 action/scale 重复）。"
      "Geometry 永不参与 PCA。\n")
    A(f"- state rows：`{manifest['n_state']:,}`；"
      f"join match rate = `{join_rate}`（reaction/field/depletion 均须 1.0）")
    A(f"- Stage 4A 复现：cells=`{repro['n_cells']}`，max|diff|="
      f"`{repro['max_abs_diff']}`\n")

    A("## Q1/Q2 — 有效维度（k80/k90/k95）\n")
    A("Reaction：\n")
    A(_fmt(pd.DataFrame([
        dict(block=b, wf=wf, k80=res_r[wf]["k80"], k90=res_r[wf]["k90"],
             k95=res_r[wf]["k95"], k_primary=res_r[wf]["k_primary"],
             pc1=round(float(res_r[wf]["evr"][0]), 4),
             pc2=round(float(res_r[wf]["evr"][1]), 4),
             pc3=round(float(res_r[wf]["evr"][2]), 4))
        for b, wf in [("reaction", w) for w in TEST_WF]])))
    A("Liquidity：\n")
    A(_fmt(pd.DataFrame([
        dict(block=b, wf=wf, k80=res_l[wf]["k80"], k90=res_l[wf]["k90"],
             k95=res_l[wf]["k95"], k_primary=res_l[wf]["k_primary"],
             pc1=round(float(res_l[wf]["evr"][0]), 4),
             pc2=round(float(res_l[wf]["evr"][1]), 4),
             pc3=round(float(res_l[wf]["evr"][2]), 4))
        for b, wf in [("liquidity", w) for w in TEST_WF]])))

    A("## Q3 — 跨 WF 稳定性（Hungarian 匹配，k=3）\n")
    A(_fmt(stab_r))
    A(_fmt(stab_l))

    A("## Q4/Q5 — 当前 action outcome：压缩后是否保留信息（target_first）\n")
    A(_fmt(action_info.pivot_table(index="wf", columns="model",
                                   values="roc_auc").reset_index()))
    A("ΔAUC：\n")
    A(_fmt(action_delta))

    A("## Q6 — base state 能否解释 WAIT value\n")
    A("primary target = `delta_E_R_lower`（连续，未二值化）：\n")
    A(_fmt(wait_info[wait_info["target"] == "delta_E_R_lower"].pivot_table(
        index="wf", columns="model", values="r2").reset_index()))
    A("Spearman（primary target）：\n")
    A(_fmt(wait_info[wait_info["target"] == "delta_E_R_lower"].pivot_table(
        index="wf", columns="model", values="spearman").reset_index()))
    A("Δ（R² / Spearman）：\n")
    A(_fmt(wait_delta))
    A("`delta_target_first` 仅作 group-level 汇总（不建模为单样本概率差）：\n")
    A(_fmt(wait_tf))

    A("## Q7 — nonlinear compression gate\n")
    A(f"判定：**{gate['verdict']}**\n")
    A(_fmt(pd.DataFrame(gate["evidence"])))
    A("### 结论\n")
    for q in gate["answers"]:
        A(f"- {q}")
    A("\n**DEFERRED**：`remaining_liq_imbalance_1p0` 未纳入 primary liquidity "
      "block——它需要新的 surviving-field imbalance 定义（decision-h 参考价 + "
      "带权强度），不在 frozen snapshot 合同内，不在本轮自行发明公式。\n")
    A("**STOP**：本轮不进入 Autoencoder / Kernel PCA / UMAP / t-SNE / DP / RL。\n")
    Path(path).write_text("\n".join(L), encoding="utf-8")


# ===========================================================================
def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = s4a.load_env()
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")

    df, gmap = compute_action_surface_all_blocks(D, master_by_sym, bars_by_sym)
    print(f"[SURFACE-ALL] rows={len(df)} contacts={df['gid'].nunique()} "
          f"({time.perf_counter()-t0:.1f}s)")
    repro = assert_reproduce_stage4a(df)
    print(f"[REPRO] Stage 4A TB2-TB4 reproduced: cells={repro['n_cells']} "
          f"max_diff={repro['max_abs_diff']}")

    state, join = build_state_table(D, df, gmap)
    join_rate = 1.0
    print(f"[STATE] rows={len(state)} join={join} ({time.perf_counter()-t0:.1f}s)")

    res_r, res_l = {}, {}
    for wf in TEST_WF:
        train_wf = TRAIN_WF_OF[wf]
        tm = state["wf"].isin(train_wf).to_numpy()
        res_r[wf] = fit_pca(state, REACTION_FEATURES, REACTION_LOG1P, tm)
        res_l[wf] = fit_pca(state, LIQUIDITY_FEATURES, [], tm)
        for j in range(res_r[wf]["k_primary"]):
            state[f"rpc{j+1}_{wf}"] = res_r[wf]["scores"][:, j].astype("float32")
        for j in range(res_l[wf]["k_primary"]):
            state[f"lpc{j+1}_{wf}"] = res_l[wf]["scores"][:, j].astype("float32")
        print(f"[PCA] {wf} r_k90={res_r[wf]['k90']} r_kp={res_r[wf]['k_primary']} "
              f"l_k90={res_l[wf]['k90']} l_kp={res_l[wf]['k_primary']} "
              f"({time.perf_counter()-t0:.1f}s)")

    vr = pd.concat([pca_variance_rows("reaction", res_r, REACTION_FEATURES)]
                   ).assign(k80=lambda d: d["wf"].map(
                       {w: res_r[w]["k80"] for w in TEST_WF}),
                       k90=lambda d: d["wf"].map(
                           {w: res_r[w]["k90"] for w in TEST_WF}),
                       k95=lambda d: d["wf"].map(
                           {w: res_r[w]["k95"] for w in TEST_WF}),
                       k_primary=lambda d: d["wf"].map(
                           {w: res_r[w]["k_primary"] for w in TEST_WF}))
    vl = pd.concat([pca_variance_rows("liquidity", res_l, LIQUIDITY_FEATURES)]
                   ).assign(k80=lambda d: d["wf"].map(
                       {w: res_l[w]["k80"] for w in TEST_WF}),
                       k90=lambda d: d["wf"].map(
                           {w: res_l[w]["k90"] for w in TEST_WF}),
                       k95=lambda d: d["wf"].map(
                           {w: res_l[w]["k95"] for w in TEST_WF}),
                       k_primary=lambda d: d["wf"].map(
                           {w: res_l[w]["k_primary"] for w in TEST_WF}))
    lr = pca_loading_rows("reaction", res_r, REACTION_FEATURES)
    ll = pca_loading_rows("liquidity", res_l, LIQUIDITY_FEATURES)
    stab_r = stability_rows("reaction", res_r, REACTION_FEATURES, k=3)
    stab_l = stability_rows("liquidity", res_l, LIQUIDITY_FEATURES, k=3)

    nmf_rows, nmf_recon = [], []
    for wf in TEST_WF:
        tm = state["wf"].isin(TRAIN_WF_OF[wf]).to_numpy()
        c, r = run_nmf_for_wf(state, wf, tm)
        nmf_rows += c
        nmf_recon += r
        print(f"[NMF] {wf} done ({time.perf_counter()-t0:.1f}s)")
    nmf_c = pd.DataFrame(nmf_rows)

    print("[ACTION] fitting ...")
    action_info, action_delta, action_cov = run_action_models(
        df, gmap, state, res_r, res_l)
    print("[WAIT] fitting ...")
    wait_info, wait_delta, wait_tf = run_wait_models(
        df, gmap, state, res_r, res_l)

    # ---- §14 nonlinear compression gate ----
    A_T, W_R2, W_SP = 0.01, 0.005, 0.02
    ev = []
    ad = action_delta.set_index("wf")
    wd = wait_delta.set_index("wf")
    r_gain = int((ad["d_M1_M0"] >= A_T).sum())
    r_loss = int((ad["d_M1_M2"] >= A_T).sum())
    l_gain = int((ad["d_M3_M0"] >= A_T).sum())
    l_loss = int((ad["d_M3_M4"] >= A_T).sum())
    wr_gain = int(((wd["d_W1_W0_r2"] >= W_R2)
                   | (wd["d_W1_W0_spearman"] >= W_SP)).sum())
    wr_loss = int(((wd["d_W1_W2_r2"] >= W_R2)
                   | (wd["d_W1_W2_spearman"] >= W_SP)).sum())
    wl_gain = int(((wd["d_W3_W0_r2"] >= W_R2)
                   | (wd["d_W3_W0_spearman"] >= W_SP)).sum())
    wl_loss = int(((wd["d_W3_W4_r2"] >= W_R2)
                   | (wd["d_W3_W4_spearman"] >= W_SP)).sum())
    ev = [dict(channel="action_reaction", raw_gain_wf=r_gain,
               pca_loss_wf=r_loss),
          dict(channel="action_liquidity", raw_gain_wf=l_gain,
               pca_loss_wf=l_loss),
          dict(channel="wait_reaction", raw_gain_wf=wr_gain, pca_loss_wf=wr_loss),
          dict(channel="wait_liquidity", raw_gain_wf=wl_gain, pca_loss_wf=wl_loss)]
    worth = any(e["raw_gain_wf"] >= 2 and e["pca_loss_wf"] >= 2 for e in ev)
    verdict = ("NONLINEAR_COMPRESSION_WORTH_TESTING" if worth
               else "STOP_BEFORE_NONLINEAR_COMPRESSION")
    raw_g = ad["d_M1_M0"].mean()
    pca_g = ad["d_M2_M0"].mean()
    lraw_g = ad["d_M3_M0"].mean()
    lpca_g = ad["d_M4_M0"].mean()
    wr2 = wd["d_W1_W0_r2"].mean()
    wr2p = wd["d_W2_W0_r2"].mean()
    answers = [
        f"Q1 Reaction：k90 = {[res_r[w]['k90'] for w in TEST_WF]}，"
        f"k_primary = {[res_r[w]['k_primary'] for w in TEST_WF]}。",
        f"Q2 Liquidity：k90 = {[res_l[w]['k90'] for w in TEST_WF]}，"
        f"k_primary = {[res_l[w]['k_primary'] for w in TEST_WF]}。",
        f"Q3 稳定性：reaction mean matched |cos| = "
        f"{stab_r[stab_r['component_a'] == 0]['abs_cosine'].mean():.3f}；"
        f"liquidity = {stab_l[stab_l['component_a'] == 0]['abs_cosine'].mean():.3f}。",
        f"Q4 reaction 压缩保留：raw ΔAUC 均值 {raw_g:+.4f} vs PCA ΔAUC 均值 "
        f"{pca_g:+.4f}。",
        f"Q5 liquidity 压缩保留：raw ΔAUC 均值 {lraw_g:+.4f} vs PCA ΔAUC 均值 "
        f"{lpca_g:+.4f}。",
        f"Q6 base state → WAIT value：delta_E_R_lower raw R² 增益均值 "
        f"{wr2:+.5f}，PCA 版 {wr2p:+.5f}。",
        f"Q7 判定 {verdict}。"]

    manifest = dict(
        experiment="Stage 4B Latent-State Compression Gate v1.0",
        base_commit=BASE_COMMIT,
        state_key=STATE_KEY, n_state=int(len(state)),
        n_action_rows=int(len(df)),
        reaction_features=REACTION_FEATURES,
        reaction_log1p=REACTION_LOG1P,
        forbidden_reaction=FORBIDDEN_REACTION,
        liquidity_features=LIQUIDITY_FEATURES,
        deferred_liquidity=sorted(DEFERRED_FEATURES),
        nmf_features=NMF_FEATURES, nmf_ks=NMF_KS,
        nmf_subsample=NMF_SUBSAMPLE,
        geometry_never_in_pca=GEO_COLS,
        wf_scheme={w: dict(train=TRAIN_WF_OF[w], test=TEST_BLOCK_OF_WF[w])
                   for w in TEST_WF},
        preprocessing=dict(
            log1p_then_standard=REACTION_LOG1P,
            density="log1p(up+dn) constructed once; then StandardScaler(train)",
            imbalance="(up-dn)/(up+dn+1e-12)",
            scaler_pca="strict train-only per WF"),
        join_match_rate=join,
        reproduction=repro)
    json.dump(manifest, open(OUT / "latent_feature_manifest.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    vr.to_csv(OUT / "reaction_pca_variance_by_wf.csv", index=False)
    vl.to_csv(OUT / "liquidity_pca_variance_by_wf.csv", index=False)
    lr.to_csv(OUT / "reaction_pca_loadings_by_wf.csv", index=False)
    ll.to_csv(OUT / "liquidity_pca_loadings_by_wf.csv", index=False)
    stab_r.to_csv(OUT / "reaction_pca_stability.csv", index=False)
    stab_l.to_csv(OUT / "liquidity_pca_stability.csv", index=False)
    nmf_c.to_csv(OUT / "liquidity_nmf_components.csv", index=False)
    action_info.to_csv(OUT / "latent_action_information_by_wf.csv", index=False)
    action_delta.to_csv(OUT / "latent_action_delta_auc_by_wf.csv", index=False)
    action_cov.to_csv(OUT / "latent_action_coverage_by_wf.csv", index=False)
    wait_info.to_csv(OUT / "latent_wait_information_by_wf.csv", index=False)
    wait_delta.to_csv(OUT / "latent_wait_delta_by_wf.csv", index=False)
    wait_tf.to_csv(OUT / "latent_wait_target_first_diagnostic.csv", index=False)

    gate = dict(verdict=verdict, evidence=ev, answers=answers,
                thresholds=dict(action_auc=A_T, wait_r2=W_R2, wait_spearman=W_SP),
                forbidden_next=["Autoencoder", "Kernel PCA", "UMAP feature",
                                "t-SNE feature", "DP", "RL"])
    json.dump(dict(manifest=manifest, gate=gate),
              open(OUT / "LATENT_STATE_COMPRESSION_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)

    write_report(OUT / "LATENT_STATE_COMPRESSION_V1.md", manifest, res_r, res_l,
                 vr, vl, action_info, action_delta, wait_info, wait_delta,
                 wait_tf, stab_r, stab_l, nmf_c, gate, join_rate, repro)
    print(f"\n[VERDICT] {verdict}")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
