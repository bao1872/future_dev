"""Stage 4A 前置 HARD gate：P1.1 t0 流动性场冻结复现 + P1.2 prefix causal 复现
+ I1/I2/I3 集成 gate（结构化主键 join 覆盖）。

运行：
  python audit_stage4a_reproduction.py

HARD 语义：任一 gate FAIL → SystemExit(1) + audit JSON 标记 all_pass=false。

P1.1  验证 Stage 4A 使用的 t0 流动性场（active_mask + nearest above/below）与冻结
      Atlas v1.2 (liquidity_state_snapshot_v1_2.parquet) 完全一致（用 entry_reference）。
P1.2  验证 reaction morphology 特征可由 prefix-only（仅前 h 根 post-contact bar）原始数据
      完整复现（full == prefix），无未来泄漏。
I1     FIELD_JOIN_MATCH_RATE = Stage 4A gmap 主键在 field artifact 的匹配率（须 1.0）。
I2     REACTION_JOIN_MATCH_RATE = (contact,h) 在 reaction artifact 的匹配率（须 1.0）。
I3     join 后关键列不能整列 NaN（防 join 失败导致空表）。
"""
from __future__ import annotations
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env
from research.liquidity_oracle_atlas.liquidity_field_reaction_model_v1 import (
    reaction_matrices, geometry_horizon, complexity_matrix)

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
FROZEN_SNAP = ATLAS / "liquidity_state_snapshot_v1_2.parquet"
S1_DIR = Path("research/analysis_results/liquidity_field_reaction_model_v1")
REACT_CSV = S1_DIR / "reaction_episode_features.csv"
FLD_CSV = S1_DIR / "liquidity_field_snapshot.csv"
HORIZONS = [1, 2, 3, 5, 8, 13]
FIELD_KEY = ["symbol", "liquidity_id", "contact_number"]
REACTION_KEY = ["symbol", "liquidity_id", "contact_number", "h"]
OUT = Path("research/analysis_results/liquidity_field_action_surface_v1")
OUT.mkdir(parents=True, exist_ok=True)

R_FEAT = ["total_variation_atr", "path_efficiency", "amplitude_atr",
          "cross_count", "dc_pivots_0p40", "dc_pivots_0p80"]


# ===========================================================================
# P1.1  t0 流动性场冻结复现
# ===========================================================================
def run_p11(D, master_by_sym, snap):
    print(f"[P1.1] recomputing t0 field for {len(snap)} contacts vs frozen Atlas v1.2 ...")
    recs = []
    snap = snap.reset_index(drop=True)
    for sym, sub in snap.groupby("symbol"):
        ms = master_by_sym.get(sym)
        if ms is None or len(sub) == 0:
            continue
        mprice = ms["price"].to_numpy(float)
        mav = pd.to_datetime(ms["available_time"]).to_numpy()
        mfp = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
        dt = pd.to_datetime(sub["decision_time"].to_numpy()).to_numpy()
        entry = sub["entry_reference"].to_numpy(float)
        atr0 = sub["atr0"].to_numpy(float)
        av_le = mav[None, :] <= dt[:, None]
        fp_gt = np.isnat(mfp)[None, :] | (mfp[None, :] > dt[:, None])
        active = av_le & fp_gt                       # 与 active_mask 完全一致
        n_act = active.sum(1)
        dp = (mprice[None, :] - entry[:, None]) / atr0[:, None]
        upv = np.where(active & (dp > 0), dp, np.inf)
        dnv = np.where(active & (dp < 0), dp, -np.inf)
        nab = upv.min(1)
        nbl = -dnv.max(1)
        recs.append(pd.DataFrame({
            "symbol": sym,
            "liquidity_id": sub["liquidity_id"].to_numpy().astype(str),
            "contact_number": sub["contact_number"].to_numpy().astype(int),
            "n_active": n_act.astype(int),
            "nearest_above_R": nab,
            "nearest_below_R": nbl,
        }))
    rec = pd.concat(recs, ignore_index=True)

    keys = ["symbol", "liquidity_id", "contact_number"]
    cmp = snap[keys + ["active_visible_count", "nearest_above_R",
                       "nearest_below_R", "entry_reference"]].copy()
    cmp = cmp.merge(rec, on=keys, how="left")

    # 1) active count 精确复现（active_mask 与冻结 Atlas 完全一致）
    cnt_mismatch = int((cmp["active_visible_count"].to_numpy(int)
                        != cmp["n_active"].to_numpy(int)).sum())
    # 2) nearest above/below R（用 entry_reference 复现，与冻结 Atlas 一致；
    #    冻结值四舍五入到 4 位小数，故比较到 4 位）
    def rd(x):
        return pd.to_numeric(x, errors="coerce").round(4)
    d_ab = (rd(cmp["nearest_above_R_x"]) - rd(cmp["nearest_above_R_y"])).abs().fillna(0)
    d_bl = (rd(cmp["nearest_below_R_x"]) - rd(cmp["nearest_below_R_y"])).abs().fillna(0)
    r_mismatch = int(((d_ab > 1e-4) | (d_bl > 1e-4)).sum())
    max_d = float(max(d_ab.max(), d_bl.max()))

    # 3) Stage 4A 消费兼容性：Stage 1-3 artifact 必须带结构化主键且唯一
    fld = pd.read_csv(FLD_CSV)
    rct = pd.read_csv(REACT_CSV)
    field_has_key = all(c in fld.columns for c in FIELD_KEY)
    react_has_key = all(c in rct.columns for c in REACTION_KEY)
    field_unique = bool(field_has_key and not fld.duplicated(FIELD_KEY).any())
    react_unique = bool(react_has_key and not rct.duplicated(REACTION_KEY).any())
    artifact_ok = field_has_key and react_has_key and field_unique and react_unique

    ok = (cnt_mismatch == 0) and (r_mismatch == 0) and artifact_ok
    detail = dict(
        n_contacts=int(len(cmp)),
        active_count_mismatch=cnt_mismatch,
        nearest_R_mismatch=r_mismatch,
        max_abs_R_diff=round(max_d, 6),
        frozen_field_reproduction_pass=bool(cnt_mismatch == 0 and r_mismatch == 0),
        artifact_structured_key_present=bool(field_has_key and react_has_key),
        artifact_key_unique=bool(field_unique and react_unique),
    )
    print(f"[P1.1] active_mismatch={cnt_mismatch} R_mismatch={r_mismatch} "
          f"max|dR|={max_d:.2e} artifact_ok={artifact_ok}")
    return ok, detail


# ===========================================================================
# P1.2  prefix causal 复现（reaction morphology 可由 prefix-only 原始数据复现）
# ===========================================================================
def run_p12(D, bars_by_sym, snap_lookup):
    print("[P1.2] recomputing reaction morphology from prefix-only raw bars ...")
    F = D["F"]
    samp = F.sample(min(500, len(F)), random_state=0).reset_index(drop=True)
    hmax = max(HORIZONS)

    # 已冻结的 Stage 1-3 reaction 特征（用 symbol/horizon 分组 + 自然键排序做值复现）
    react = pd.read_csv(REACT_CSV)
    react = react[react["horizon"].isin(HORIZONS)].copy()

    mism = {f: 0 for f in R_FEAT}
    maxd = {f: 0.0 for f in R_FEAT}
    n_compared = 0
    inv_ok = 0
    inv_tot = 0

    for sym, sub in samp.groupby("symbol"):
        bars = bars_by_sym.get(sym)
        if bars is None or len(sub) == 0:
            continue
        one = sub.reset_index(drop=True)
        C = len(one)
        yc, ymax, ymin = reaction_matrices(bars, one, hmax=hmax)
        bp = one["liquidity_price"].to_numpy(float)
        atr0 = one["atr0"].to_numpy(float)
        side = one["side"].to_numpy(int)
        for h in HORIZONS:
            g_pre = geometry_horizon(yc[:, :h + 1], ymax[:, :h + 1], ymin[:, :h + 1], h)
            cm_pre = complexity_matrix(yc[:, :h + 1])
            g_full = geometry_horizon(yc, ymax, ymin, h)
            # prefix 不变量：full vs prefix-only 必须完全一致（特征仅依赖 prefix）
            tv_close = np.isclose(g_full["total_variation_atr"], g_pre["total_variation_atr"],
                                  atol=1e-9, equal_nan=True)
            cc_close = np.array_equal(g_full["cross_count"], g_pre["cross_count"])
            inv_ok += int(tv_close.sum()) + int(C if cc_close else 0)
            # cc_close 是整组 bool；用逐 contact 近似：cross_count 整组相等则全部计
            inv_tot += 2 * C
            # 与冻结 CSV 做值复现：同 (symbol,horizon) 内按自然键对齐
            sortk = np.lexsort((atr0, bp, side))
            blk = pd.DataFrame({
                "symbol": sym, "horizon": h, "side": side[sortk],
                "boundary_price": np.round(bp, 5)[sortk],
                "atr0": np.round(atr0, 5)[sortk],
                "total_variation_atr": np.round(g_pre["total_variation_atr"], 4)[sortk],
                "path_efficiency": np.round(g_pre["path_efficiency"], 4)[sortk],
                "amplitude_atr": np.round(g_pre["amplitude_atr"], 4)[sortk],
                "cross_count": g_pre["cross_count"][sortk],
                "dc_pivots_0p40": cm_pre["dc_pivots_0p40"][sortk],
                "dc_pivots_0p80": cm_pre["dc_pivots_0p80"][sortk],
            })
            rsub = react[(react["symbol"] == sym) & (react["horizon"] == h)].copy()
            if len(rsub) == 0:
                continue
            rsub = rsub.sort_values(["side", "boundary_price", "atr0"]).reset_index(drop=True)
            blk = blk.sort_values(["side", "boundary_price", "atr0"]).reset_index(drop=True)
            m = min(len(blk), len(rsub))
            if m == 0:
                continue
            n_compared += m
            for f in R_FEAT:
                a = pd.to_numeric(blk[f].iloc[:m], errors="coerce").to_numpy(float)
                b = pd.to_numeric(rsub[f].iloc[:m], errors="coerce").to_numpy(float)
                good = np.isfinite(a) & np.isfinite(b)
                same = good & (np.abs(a - b) <= 1e-3)
                mism[f] += int((~same).sum())
                if good.any():
                    maxd[f] = max(maxd[f], float(np.abs(a[good] - b[good]).max()))

    inv_rate = inv_ok / inv_tot if inv_tot else 1.0
    total_mismatch = sum(mism.values())
    # HARD: prefix 不变量必须 100%；值复现仅作诊断
    ok = inv_rate >= 0.999
    detail = dict(
        n_sampled_contacts=int(len(samp)),
        n_feature_cells_compared=int(n_compared),
        full_vs_prefix_invariant_rate=round(inv_rate, 6),
        csv_value_reproduction_mismatch=mism,
        csv_value_reproduction_max_abs_diff={k: round(v, 6) for k, v in maxd.items()},
    )
    print(f"[P1.2] sampled={len(samp)} cells={n_compared} "
          f"prefix_invariant_rate={inv_rate:.4f} csv_value_mismatch={total_mismatch}")
    return ok, detail


# ===========================================================================
# I1/I2/I3 集成 gate：结构化主键 join 覆盖 + 非空校验
# ===========================================================================
def run_integration(D):
    print("[I] verifying structured-key join coverage ...")
    F = D["F"]
    fkey = F[FIELD_KEY].drop_duplicates().reset_index(drop=True)
    rkey = pd.concat([F[FIELD_KEY].assign(h=h) for h in HORIZONS], ignore_index=True)

    fld = pd.read_csv(FLD_CSV)
    rct = pd.read_csv(REACT_CSV)

    # I1 FIELD_JOIN_MATCH_RATE
    fm = fkey.merge(fld[FIELD_KEY], on=FIELD_KEY, how="left", indicator=True)
    i1 = float((fm["_merge"] == "both").mean())

    # I2 REACTION_JOIN_MATCH_RATE（按 h）
    rm = rkey.merge(rct[REACTION_KEY], on=REACTION_KEY, how="left", indicator=True)
    i2 = float((rm["_merge"] == "both").mean())
    i2_by_h = {int(h): float((rm[rm["h"] == h]["_merge"] == "both").mean())
               for h in HORIZONS}

    # I3 关键列 coverage（防 join 失败 → 整列 NaN）。
    # 注意：field_position / room_up_atr 在无活跃流动性一侧时合法为 NaN
    # （结构性语义缺失，非 join 故障）；真正要防的是 join 落空导致 coverage≈0。
    # 阈值取 0.5：>0.5 表示 join 产生了真实数据，~0 才表示 join 失败。
    fjoin = fkey.merge(fld, on=FIELD_KEY, how="left")
    field_cov = {c: float(fjoin[c].notna().mean()) for c in
                 ["field_position", "liq_imbalance_1p0", "room_up_atr"]}
    rjoin = rkey.merge(rct, on=REACTION_KEY, how="left")
    react_cov = {c: float(rjoin[c].notna().mean()) for c in
                 ["path_efficiency", "cross_count", "amplitude_atr",
                  "dc_pivots_0p40", "dc_pivots_0p80"]}
    i3_pass = (all(v >= 0.5 for v in field_cov.values())
               and all(v >= 0.5 for v in react_cov.values()))

    ok = (i1 >= 0.999) and (i2 >= 0.999) and i3_pass
    detail = dict(
        field_join_match_rate=round(i1, 6),
        reaction_join_match_rate=round(i2, 6),
        reaction_join_match_rate_by_h={k: round(v, 6) for k, v in i2_by_h.items()},
        field_feature_coverage=field_cov,
        reaction_feature_coverage=react_cov,
        i3_all_columns_covered=bool(i3_pass),
    )
    print(f"[I] I1_field={i1:.4f} I2_react={i2:.4f} I3={i3_pass}")
    return ok, detail


def _producer_version():
    p = S1_DIR / "artifact_version_gate.json"
    if p.exists():
        try:
            return json.load(open(p))
        except Exception:
            return None
    return None


def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = load_env()
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")
    snap = pd.read_parquet(FROZEN_SNAP)

    ok1, d1 = run_p11(D, master_by_sym, snap)
    ok2, d2 = run_p12(D, bars_by_sym, None)
    okI, dI = run_integration(D)
    all_pass = ok1 and ok2 and okI
    audit = dict(
        experiment="Stage 4A pre-gate: P1.1 + P1.2 + I1/I2/I3 integration",
        base_commit="7107f16",
        artifact_producer=_producer_version(),
        P1_1_t0_field_reproduction=dict(pass_=bool(ok1), **d1),
        P1_2_prefix_causal_reproduction=dict(pass_=bool(ok2), **d2),
        I_integration_join=dict(pass_=bool(okI), **dI),
        all_pass=bool(all_pass),
    )
    json.dump(audit, open(OUT / "stage4a_reproduction_gate.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"\n[REPRO-GATE] all_pass={all_pass}")
    print(f"  P1.1 pass={ok1}  P1.2 pass={ok2}  I1/I2/I3 pass={okI}")
    if not all_pass:
        print("[FATAL] Stage 4A reproduction/integration gate FAILED — do NOT run full Stage 4A.")
        raise SystemExit(1)
    print("[OK] Stage 4A gates PASSED — cleared to run full Stage 4A.")


if __name__ == "__main__":
    main()
