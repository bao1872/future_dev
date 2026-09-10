"""订单块边界局部反应实验 v3.1 —— 阶段二：解锁 outcome 并估计局部边界效应。

模型（透明局部线性概率模型，非机器学习）：
    Y = α + τ·D + β1·x + β2·D·x + e
    x = ob_margin_R ；D = 1{x >= 0} ；三角核权重 w = max(1 - |x|/h, 0)
    τ = 边界处"刚进入 OB"相对"刚没进入 OB"的局部概率差异

bootstrap：canonical trading_day 整块有放回，500 次，每次重新拟合。
WLS 使用等价的 numpy 实现（环境无 statsmodels）。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    SymbolData,
)
from research.liquidity_state_machine.match_liquidity_placebo_v2 import (
    resolve_stage2,
)

RESULTS = Path("research/analysis_results/liquidity_ob_boundary_v3_1")
BANDWIDTHS_R = [0.10, 0.20, 0.30, 0.50]
BOOT = 500
MIN_SIDE_INFER = 200
OVERALL_INTERPRETABLE = True

SUBGROUPS = [
    ("sweep_vs_1h", ["AGAINST_TREND", "WITH_TREND"], "1h趋势关系"),
    ("env4h_vs_1h", ["AGAINST_TREND", "WITH_TREND"], "4h×1h"),
    ("trend_1h_vs_15m", ["AGAINST_TREND", "WITH_TREND"], "1h×15m"),
    ("liquidity_source_tf_group", ["5m", "15m", "1h", "TIME"],
     "被扫liquidity级别"),
    ("nearest_ob_source_tf", ["5m", "15m", "1h"], "OB周期"),
    ("nearest_ob_freshness", ["FRESH", "RETESTED"], "OB新鲜度"),
    ("liquidity_overlap", ["SINGLE", "MULTI_IDENTITY"], "同价位叠加"),
]
SECONDARY_DIMS = ["symbol", "fold"]


def _wls(y, X, w):
    """加权最小二乘，返回 (params, pvalues)。"""
    y = np.asarray(y, float)
    X = np.asarray(X, float)
    w = np.asarray(w, float)
    sw = np.sqrt(w)
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    n, k = X.shape
    resid = y - X @ beta
    sigma2 = float(np.sum(w * resid ** 2) / max(n - k, 1))
    try:
        cov = sigma2 * np.linalg.inv(X.T @ (X * w[:, None]))
        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
    except np.linalg.LinAlgError:
        se = np.full(k, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        tval = beta / se
    return beta, 2.0 * (1.0 - _st.norm.cdf(np.abs(tval)))


def _fit_core(x, y, h):
    """局部线性 WLS（三角核），返回 τ 等。"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    D = (x >= 0).astype(float)
    w = np.maximum(1.0 - np.abs(x) / h, 0.0)
    if len(x) < 20 or w.sum() <= 0:
        return None
    X = np.column_stack([np.ones(len(x)), D, x, D * x])
    try:
        beta, _ = _wls(y, X, w)
    except Exception:
        return None
    return dict(n=len(x), n_left=int((x < 0).sum()),
                n_right=int((x >= 0).sum()),
                p_left=float(y[x < 0].mean()) if (x < 0).any() else np.nan,
                p_right=float(y[x >= 0].mean()) if (x >= 0).any() else np.nan,
                tau=float(beta[1]))


def fit_local(g, h):
    return _fit_core(g["ob_margin_R"].to_numpy(float),
                     g["later_reclaim"].to_numpy(float), h)


def boot_tau(g, h, days, rng):
    x = g["ob_margin_R"].to_numpy(float)
    y = g["later_reclaim"].to_numpy(float)
    td = g["trading_day"].to_numpy()
    idx = {d: np.flatnonzero(td == d) for d in days}
    taus = []
    for _ in range(BOOT):
        pick = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([idx[d] for d in pick])
        r = _fit_core(x[rows], y[rows], h)
        if r is not None:
            taus.append(r["tau"])
    if len(taus) < 50:
        return (np.nan, np.nan)
    return (float(np.quantile(taus, .025)), float(np.quantile(taus, .975)))


def main():
    global OVERALL_INTERPRETABLE
    gate = json.load(open(RESULTS / "AUDIT_V3_1.json"))
    h = float(gate["primary_bandwidth"])
    if not gate.get("gate_pass"):
        print("LOCAL_CONTEXT_CONTINUITY_FAIL —— 总体 τ 不做强解释；"
              "仅执行预注册 subgroup 边界分析（§13 第二步）")
        OVERALL_INTERPRETABLE = False
    else:
        print(f"GATE PASS —— PRIMARY_BANDWIDTH = {h}，解锁 outcome")

    d = pd.read_parquet(RESULTS / "boundary_preoutcome.parquet")
    assert not ({"later_reclaim", "stage2", "stage2_state",
                 "structural_acceptance"} & set(d.columns))

    need = d[d["ob_margin_R"].abs() <= max(BANDWIDTHS_R)].copy()
    print(f"需解析 outcome 的事件 = {len(need)}")

    sd_map, rec = {}, []
    for sym, g in need.groupby("symbol"):
        sd = sd_map.setdefault(sym, SymbolData(sym))
        for r in g.itertuples(index=False):
            st, _ = resolve_stage2(sd, int(r.interaction_i), int(r.side),
                                   float(r.price))
            rec.append((r.level_key, int(st == "LATER_RECLAIM"),
                        int(st == "STRUCTURAL_ACCEPTANCE")))
        print(f"  [outcome] {sym} {len(g)}", flush=True)
    O = pd.DataFrame(rec, columns=["level_key", "later_reclaim",
                                   "structural_acceptance"])
    d = d.merge(O, on="level_key", how="left")
    d.to_parquet(RESULTS / "boundary_with_outcome.parquet", index=False)

    rng = np.random.default_rng(7)
    days_all = d["trading_day"].dropna().unique()

    # ---------- 第一层：四个 bandwidth ----------
    rows = []
    for bh in BANDWIDTHS_R:
        g = d[d["ob_margin_R"].abs() <= bh].dropna(subset=["later_reclaim"])
        if len(g) < 40:
            continue
        r = fit_local(g, bh)
        if r is None:
            continue
        lo, hi = boot_tau(g, bh, days_all, rng)
        rows.append(dict(bandwidth=bh, primary=bool(bh == h), **r,
                         ci_low=round(lo, 4) if np.isfinite(lo) else None,
                         ci_high=round(hi, 4) if np.isfinite(hi) else None,
                         raw_diff_pp=round((r["p_right"] - r["p_left"]) * 100, 2),
                         exact_zero_in_band=int((g["ob_margin_R"] == 0).sum()),
                         overall_interpretable=OVERALL_INTERPRETABLE))
    BE = pd.DataFrame(rows)
    BE.to_csv(RESULTS / "local_boundary_effect.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 局部边界效应（四个 bandwidth） ===")
    print(BE.to_string(index=False))

    gg = d[d["ob_margin_R"].abs() <= h].dropna(subset=["later_reclaim"])
    lo, hi = boot_tau(gg, h, days_all, rng)
    pd.DataFrame([dict(bandwidth=h, tau=fit_local(gg, h)["tau"],
                       ci_low=lo, ci_high=hi, n=len(gg))]).to_csv(
        RESULTS / "bootstrap_boundary_effect.csv", index=False,
        encoding="utf-8-sig")
    print(f"\nPRIMARY τ = {fit_local(gg, h)['tau']:.4f}  "
          f"95%CI [{lo:.4f}, {hi:.4f}]")

    # ---------- 边界响应曲线 ----------
    edges = np.arange(-0.50, 0.501, 0.10)
    c = d.dropna(subset=["later_reclaim"]).copy()
    c["mb"] = pd.cut(c["ob_margin_R"], edges, include_lowest=True)
    cur = (c.groupby("mb", observed=True)
           .agg(n=("later_reclaim", "size"),
                later_reclaim_rate=("later_reclaim", "mean"),
                structural_acceptance_rate=("structural_acceptance", "mean"))
           .reset_index())
    cur["n"] = cur["n"].astype(int)
    for k in ("later_reclaim_rate", "structural_acceptance_rate"):
        cur[k] = cur[k].round(4)
    cur.to_csv(RESULTS / "margin_response_curve.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== 边界响应曲线 ===")
    print(cur.to_string(index=False))

    # ---------- 第二层：预注册异质性 ----------
    sub = []
    for col, levels, label in SUBGROUPS:
        if col not in d.columns:
            continue
        for lv in levels:
            s = d[(d[col].astype(str) == lv)
                  & (d["ob_margin_R"].abs() <= h)].dropna(
                      subset=["later_reclaim"])
            if len(s) < 40:
                sub.append(dict(dim=label, level=lv, n=len(s),
                                note="样本不足"))
                continue
            r = fit_local(s, h)
            if r is None:
                continue
            nl, nr = r["n_left"], r["n_right"]
            if nl >= MIN_SIDE_INFER and nr >= MIN_SIDE_INFER:
                lo2, hi2 = boot_tau(s, h, s["trading_day"].dropna().unique(),
                                    rng)
            else:
                lo2, hi2 = np.nan, np.nan
            sub.append(dict(dim=label, level=lv, n=len(s), n_left=nl,
                            n_right=nr, p_left=round(r["p_left"], 4),
                            p_right=round(r["p_right"], 4),
                            tau=round(r["tau"], 4),
                            ci_low=round(lo2, 4) if np.isfinite(lo2) else None,
                            ci_high=round(hi2, 4) if np.isfinite(hi2) else None,
                            infer=bool(nl >= MIN_SIDE_INFER
                                       and nr >= MIN_SIDE_INFER)))
    SB = pd.DataFrame(sub)
    SB.to_csv(RESULTS / "boundary_subgroups.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 预注册异质性 ===")
    print(SB.to_string(index=False))

    # ---------- 次级：symbol / fold ----------
    sec = []
    for col in SECONDARY_DIMS:
        for lv in sorted(d[col].astype(str).unique()):
            s = d[(d[col].astype(str) == lv)
                  & (d["ob_margin_R"].abs() <= h)].dropna(
                      subset=["later_reclaim"])
            if len(s) < 40:
                continue
            r = fit_local(s, h)
            if r is None:
                continue
            nl, nr = r["n_left"], r["n_right"]
            if nl >= MIN_SIDE_INFER and nr >= MIN_SIDE_INFER:
                lo2, hi2 = boot_tau(s, h, s["trading_day"].dropna().unique(),
                                    rng)
            else:
                lo2, hi2 = np.nan, np.nan
            sec.append(dict(dim=col, level=lv, n=len(s), n_left=nl,
                            n_right=nr, p_left=round(r["p_left"], 4),
                            p_right=round(r["p_right"], 4),
                            tau=round(r["tau"], 4),
                            ci_low=round(lo2, 4) if np.isfinite(lo2) else None,
                            ci_high=round(hi2, 4) if np.isfinite(hi2) else None,
                            infer=bool(nl >= MIN_SIDE_INFER
                                       and nr >= MIN_SIDE_INFER)))
    SE = pd.DataFrame(sec)
    SE.to_csv(RESULTS / "boundary_secondary.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 按品种 / 折 ===")
    print(SE.to_string(index=False))

    # ---------- state transition matrix ----------
    keys = [k for k in ["sweep_vs_1h", "trend_1h_vs_15m",
                        "liquidity_source_tf_group", "nearest_ob_source_tf",
                        "nearest_ob_freshness", "liquidity_overlap"]
            if k in d.columns]
    mrows = []
    for kk, s in d[d["ob_margin_R"].abs() <= h].dropna(
            subset=["later_reclaim"]).groupby(keys):
        if not isinstance(kk, tuple):
            kk = (kk,)
        r = fit_local(s, h)
        if r is None:
            continue
        mrows.append(dict(**dict(zip(keys, kk)), n=len(s),
                          n_left=r["n_left"], n_right=r["n_right"],
                          p_left=round(r["p_left"], 4),
                          p_right=round(r["p_right"], 4),
                          tau=round(r["tau"], 4),
                          infer=bool(r["n_left"] >= MIN_SIDE_INFER
                                     and r["n_right"] >= MIN_SIDE_INFER)))
    MT = pd.DataFrame(mrows).sort_values("n", ascending=False)
    MT.to_csv(RESULTS / "state_transition_matrix.csv", index=False,
              encoding="utf-8-sig")
    print(f"\nstate_transition_matrix 行数 = {len(MT)}（n 小者仅画像）")

    # ---------- 三级交互模型 ----------
    inter = []
    for col, base in (("nearest_ob_source_tf", "5m"),
                      ("nearest_ob_freshness", "FRESH"),
                      ("sweep_vs_1h", "WITH_TREND")):
        if col not in d.columns:
            continue
        s = d[d["ob_margin_R"].abs() <= h].dropna(
            subset=["later_reclaim"]).copy()
        D = (s["ob_margin_R"] >= 0).astype(float).to_numpy()
        x = s["ob_margin_R"].to_numpy(float)
        w = np.maximum(1.0 - np.abs(x) / h, 0.0)
        lv = sorted(v for v in s[col].astype(str).unique() if v != base)
        cols = ["const", "D", "x", "Dx"]
        M = [np.ones(len(s)), D, x, D * x]
        for v in lv:
            gv = (s[col].astype(str) == v).astype(float).to_numpy()
            M.append(gv)
            M.append(D * gv)
            cols += [f"G_{v}", f"DG_{v}"]
        try:
            beta, pv = _wls(s["later_reclaim"].to_numpy(float),
                            np.column_stack(M), w)
        except Exception:
            continue
        pm = dict(zip(cols, beta))
        pp = dict(zip(cols, pv))
        row = dict(model=f"D × {col}", base=base, n=len(s),
                   tau_base=round(float(pm["D"]), 4))
        for v in lv:
            row[f"delta_{v}"] = round(float(pm[f"DG_{v}"]), 4)
            row[f"p_{v}"] = round(float(pp[f"DG_{v}"]), 4)
        inter.append(row)
    IT = pd.DataFrame(inter)
    IT.to_csv(RESULTS / "boundary_interaction_models.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 交互模型 ===")
    print(IT.to_string(index=False))
    print("\nBOUNDARY_DONE")


if __name__ == "__main__":
    main()
