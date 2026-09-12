"""Stage 4A P0 HARD AUDIT — structure_scale δ vs 实际 stop 语义。

P0.1 代码审查结论（见 run_liquidity_field_action_surface_v1.py:289-292）：
    stop_abs = boundary_c + stop_z * atr0_c
    stop_z = latest_confirmed_extreme(z_low, z_high, h, scale, d)
    risk = abs(entry - stop_abs);  entry = O_c[:, h]
即 scale 仅作为 δ-confirmation 阈值传入 latest_confirmed_extreme；
实际 stop = 已 δ-confirmed 的 structural extremum。属于 Case 1（正确）。

本脚本做实证交叉验证：
P0.2 独立标量重建 STOP_PRICE_MATCH_RATE（应与存储 risk_atr 完全一致）。
P0.3 逐 scale 分布 sanity：若 risk_atr 恒等于 scale，说明退化成固定 ATR stop（Case 2）。
"""
import sys, time, json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env
from research.liquidity_oracle_atlas.run_liquidity_field_action_surface_v1 import (
    compute_action_surface, build_post, HORIZONS, STRUCTURE_SCALES)


def indep_latest_confirmed_extreme(zl, zh, h, delta, d):
    """独立标量重实现（与向量化版本逻辑一致：返回最晚确认的 extreme）。"""
    best = np.nan
    if d == 1:                       # LONG → confirmed LOW (support)
        for i in range(h):
            later = zh[i + 1:]
            if len(later) and (later.max() - zl[i]) >= delta:
                best = zl[i]         # 持续覆盖 → 最终为最晚确认
        return best
    else:                            # SHORT → confirmed HIGH (resistance)
        for i in range(h):
            later = zl[i + 1:]
            if len(later) and (zh[i] - later.min()) >= delta:
                best = zh[i]
        return best


def main():
    t0 = time.perf_counter()
    D, mb, bb = load_env()
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")
    df, gmap = compute_action_surface(D, mb, bb, sample=3000)
    dg = df.merge(gmap, on="gid", how="left")
    print(f"[P0] df rows={len(df)} available={int(df['available'].sum())}")

    # ---- P0.2 独立标量重建 STOP_PRICE_MATCH_RATE ----
    # 注意：存储列 'risk_atr' 实际是 abs(entry - stop_abs)（价格单位，未除以 atr0）。
    # 重建也在价格单位比较；同时记录 ATR 归一化风险用于 P0.3。
    print("\n[P0.2] independent scalar stop reconstruction vs stored risk_atr (ATR units)")
    post_cache, sub_cache = {}, {}
    av = dg[dg["available"]]
    samp = av.sample(min(2000, len(av)), random_state=1)
    recs_atr = []
    risk_norm_by_scale = {str(sc): [] for sc in STRUCTURE_SCALES}
    for _, r in samp.iterrows():
        sym = r["symbol"]; lid = r["liquidity_id"]; cn = int(r["contact_number"])
        if sym not in post_cache:
            sub_all = D["F"][D["F"]["symbol"] == sym]
            post_cache[sym] = build_post(bb[sym], sub_all)
            sub_cache[sym] = sub_all.reset_index(drop=True)
        post = post_cache[sym]; sub = sub_cache[sym]
        idx = sub[(sub["liquidity_id"] == lid) & (sub["contact_number"] == cn)].index
        if len(idx) == 0:
            continue
        k = int(idx[0]); h = int(r["h"]); scale = float(r["scale"]); d = int(r["d"])
        entry = float(post["O"][k, h])
        boundary = float(post["boundary"][k])
        atr0 = float(post["atr0"][k])
        zl = post["z_low"][k, :h]; zh = post["z_high"][k, :h]
        stop_z = indep_latest_confirmed_extreme(zl, zh, h, scale, d)
        if not np.isfinite(stop_z):
            continue
        stop_abs = boundary + stop_z * atr0
        risk_atr_ref = abs(entry - stop_abs) / atr0     # ATR 单位（与存储列一致）
        recs_atr.append((risk_atr_ref, float(r["risk_atr"])))
        risk_norm_by_scale[str(scale)].append(risk_atr_ref)
    recs_atr = np.array(recs_atr)
    match_rate = float(np.mean(np.abs(recs_atr[:, 0] - recs_atr[:, 1]) < 1e-6))
    max_abs = float(np.max(np.abs(recs_atr[:, 0] - recs_atr[:, 1]))) if len(recs_atr) else float("nan")
    print(f"  n_reconstructed={len(recs_atr)} STOP_PRICE_MATCH_RATE={match_rate:.6f} "
          f"max_abs_diff={max_abs:.2e}")

    # ---- P0.3 逐 scale ATR 归一化风险分布 sanity ----
    # 若某 scale 的 risk_norm 恒等于 scale → 退化成固定 ATR stop（Case 2）。
    print("\n[P0.3] per-scale ATR-normalized risk vs scale（若 share≈1.0 则退化）")
    p03 = {}
    for sc in STRUCTURE_SCALES:
        arr = np.array(risk_norm_by_scale[str(sc)])
        if len(arr) == 0:
            p03[str(sc)] = dict(n=0)
            continue
        share_eq = float((np.abs(arr - sc) < 1e-6).mean())
        p03[str(sc)] = dict(
            n=int(len(arr)),
            share_risk_norm_eq_scale=round(share_eq, 6),
            median_risk_norm=round(float(np.median(arr)), 5),
            p25=round(float(np.percentile(arr, 25)), 5),
            p75=round(float(np.percentile(arr, 75)), 5),
            std=round(float(np.std(arr)), 5))
        print(f"  scale={sc}: n={len(arr)} share(risk_norm≈scale)={share_eq:.4f} "
              f"median={np.median(arr):.4f} p25={np.percentile(arr,25):.4f} "
              f"p75={np.percentile(arr,75):.4f} std={np.std(arr):.4f}")

    # ---- P0 verdict ----
    degraded = any(v.get("share_risk_norm_eq_scale") is not None
                   and v["share_risk_norm_eq_scale"] >= 0.99
                   for v in p03.values() if v.get("n", 0) > 0)
    p0_pass = (match_rate >= 0.999) and (not degraded)
    print(f"\n[P0 VERDICT] pass={p0_pass} "
          f"stop_semantics_valid={match_rate >= 0.999} degraded_to_fixed_atr_stop={degraded}")
    audit = dict(
        experiment="Stage 4A P0 stop-semantics audit",
        code_review="stop_abs = boundary + latest_confirmed_extreme(...) * atr0 "
                     "(scale = δ-confirmation threshold, NOT fixed risk distance) → Case 1",
        P0_2_stop_price_match_rate=round(match_rate, 6),
        P0_2_max_abs_diff=round(max_abs, 9),
        P0_3_per_scale=p03,
        degraded_to_fixed_atr_stop=bool(degraded),
        stop_semantics_valid=bool(match_rate >= 0.999),
        P0_pass=bool(p0_pass),
    )
    out = Path("research/analysis_results/liquidity_field_action_surface_v1")
    out.mkdir(parents=True, exist_ok=True)
    json.dump(audit, open(out / "stage4a_p0_stop_audit.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    if not p0_pass:
        print("[FATAL] P0 FAIL → STAGE4A_ECONOMIC_RESULTS_INVALID; stop before re-interpretation.")
        raise SystemExit(1)
    print("[OK] P0 PASS → stop = confirmed structural extremum; only report wording needs fix.")


if __name__ == "__main__":
    main()
