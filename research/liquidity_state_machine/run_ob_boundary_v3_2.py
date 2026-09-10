"""订单块边界实验 v3.2 —— 阶段二：Gate PASS 后解锁 outcome。

主模型（透明局部线性概率模型）：
    Y = α + τD + β1x + β2Dx + e
    x = ob_margin_R
    D = 1{ob_boundary_state == "ENTERED_OB"}
    EDGE_TOUCH 完全排除于局部模型与左右支持样本
    三角核 w = max(1 - |x|/h, 0)

subgroup 推断资格不再只看样本量，必须自身通过连续性 Gate。
bootstrap：canonical trading_day 整块有放回，500 次，每次重拟合。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as _st

from research.liquidity_state_machine.build_boundary_preoutcome_v3_2 import (
    CATEG_AUDIT, KEY_NUMERIC, NUMERIC_AUDIT, smd,
)
from research.liquidity_state_machine.liquidity_specificity_placebo_v2 import (
    SymbolData,
)
from research.liquidity_state_machine.match_liquidity_placebo_v2 import (
    resolve_stage2,
)

RESULTS = Path("research/analysis_results/liquidity_ob_boundary_v3_2")
BANDWIDTHS_R = [0.10, 0.20, 0.30, 0.50]
BOOT = 500
MIN_SIDE = 200

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
# subgroup 内仍需检查的分类混淆变量
SUB_CATEG = ["symbol", "liquidity_type", "liquidity_source_tf",
             "nearest_ob_source_tf", "nearest_ob_freshness", "sweep_vs_1h"]


def _wls(y, X, w):
    y, X, w = (np.asarray(y, float), np.asarray(X, float),
               np.asarray(w, float))
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
        t = beta / se
    return beta, 2.0 * (1.0 - _st.norm.cdf(np.abs(t)))


def _fit_core(x, D, y, h):
    w = np.maximum(1.0 - np.abs(x) / h, 0.0)
    if len(x) < 20 or w.sum() <= 0:
        return None
    X = np.column_stack([np.ones(len(x)), D, x, D * x])
    try:
        beta, _ = _wls(y, X, w)
    except Exception:
        return None
    return dict(n=len(x), n_left=int((D == 0).sum()),
                n_right=int((D == 1).sum()),
                p_left=float(y[D == 0].mean()) if (D == 0).any() else np.nan,
                p_right=float(y[D == 1].mean()) if (D == 1).any() else np.nan,
                tau=float(beta[1]))


def _arrays(g):
    """返回局部模型所需数组（D=ENTERED_OB，y=later_reclaim）。"""
    return (g["ob_margin_R"].to_numpy(float),
            (g["ob_boundary_state"] == "ENTERED_OB").astype(float).to_numpy(),
            g["later_reclaim"].to_numpy(float))


def fit_local(g, h):
    return _fit_core(*_arrays(g), h)


def boot_tau(g, h, days, rng):
    x, D, y = _arrays(g)
    td = g["trading_day"].to_numpy()
    idx = {d: np.flatnonzero(td == d) for d in days}
    taus = []
    for _ in range(BOOT):
        rows = np.concatenate([idx[d] for d in
                               rng.choice(days, size=len(days),
                                          replace=True)])
        r = _fit_core(x[rows], D[rows], y[rows], h)
        if r is not None:
            taus.append(r["tau"])
    if len(taus) < 50:
        return (np.nan, np.nan)
    return (float(np.quantile(taus, .025)), float(np.quantile(taus, .975)))


def band(df, h):
    """Primary 分析样本：|margin| <= h 且排除 EDGE_TOUCH。"""
    return df[(df["ob_margin_R"].abs() <= h)
              & (df["ob_boundary_state"] != "EDGE_TOUCH")]


def subgroup_gate(s, h):
    """subgroup 自身的连续性 Gate。返回 (pass, max_smd_all, max_smd_key,
    max_cat_pp)。"""
    L = s[s["ob_boundary_state"] == "NEAR_MISS"]
    R = s[s["ob_boundary_state"] == "ENTERED_OB"]
    m_all, m_key = 0.0, 0.0
    for c in NUMERIC_AUDIT:
        if c not in s.columns:
            continue
        a = pd.to_numeric(L[c], errors="coerce")
        b = pd.to_numeric(R[c], errors="coerce")
        if a.notna().sum() < 30 or b.notna().sum() < 30:
            continue
        v = abs(smd(a, b))
        m_all = max(m_all, v)
        if c in KEY_NUMERIC:
            m_key = max(m_key, v)
    m_cat = 0.0
    for c in SUB_CATEG:
        if c not in s.columns:
            continue
        pl = L[c].astype(str).value_counts(normalize=True)
        pr = R[c].astype(str).value_counts(normalize=True)
        for k in set(pl.index) | set(pr.index):
            m_cat = max(m_cat, abs(float(pl.get(k, 0))
                                   - float(pr.get(k, 0))) * 100)
    return (bool(m_all <= 0.15 and m_key <= 0.10 and m_cat <= 8.0),
            round(m_all, 4), round(m_key, 4), round(m_cat, 2))


def main():
    gate = json.load(open(RESULTS / "AUDIT_V3_2.json"))
    if not gate.get("gate_pass"):
        print("总体 Gate 未通过 —— 最终分类 D，停止正式 outcome 推断")
        return
    h = float(gate["primary_bandwidth"])
    print(f"总体 Gate PASS —— PRIMARY_BANDWIDTH = {h}，解锁 outcome")

    d = pd.read_parquet(RESULTS / "boundary_preoutcome_v3_2.parquet")
    assert not ({"later_reclaim", "stage2", "stage2_state",
                 "structural_acceptance"} & set(d.columns))

    need = d[d["ob_margin_R"].abs() <= max(BANDWIDTHS_R)].copy()
    print(f"需解析 outcome = {len(need)}")
    sd_map, rec = {}, []
    for sym, g in need.groupby("symbol"):
        sd = sd_map.setdefault(sym, SymbolData(sym))
        for r in g.itertuples(index=False):
            st, _ = resolve_stage2(sd, int(r.interaction_i), int(r.side),
                                   float(r.price))
            rec.append((r.level_key, int(st == "LATER_RECLAIM"),
                        int(st == "STRUCTURAL_ACCEPTANCE")))
        print(f"  [outcome] {sym} {len(g)}", flush=True)
    d = d.merge(pd.DataFrame(rec, columns=[
        "level_key", "later_reclaim", "structural_acceptance"]),
        on="level_key", how="left")
    d.to_parquet(RESULTS / "boundary_with_outcome_v3_2.parquet", index=False)

    rng = np.random.default_rng(7)
    days_all = d["trading_day"].dropna().unique()

    # ---------- 三态 raw reclaim ----------
    tri = []
    for st in ("NEAR_MISS", "EDGE_TOUCH", "ENTERED_OB"):
        s = d[(d["ob_margin_R"].abs() <= h)
              & (d["ob_boundary_state"] == st)].dropna(
                  subset=["later_reclaim"])
        if not len(s):
            continue
        tri.append(dict(state=st, n=len(s),
                        raw_reclaim=round(float(s["later_reclaim"].mean()), 4),
                        raw_structural_acceptance=round(
                            float(s["structural_acceptance"].mean()), 4)))
    TRI = pd.DataFrame(tri)
    TRI.to_csv(RESULTS / "boundary_state_outcome.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== 三态 raw outcome（primary 带宽内） ===")
    print(TRI.to_string(index=False))

    # ---------- 四个 bandwidth ----------
    rows = []
    for bh in BANDWIDTHS_R:
        g = band(d, bh).dropna(subset=["later_reclaim"])
        if len(g) < 40:
            continue
        r = fit_local(g, bh)
        if r is None:
            continue
        lo, hi = boot_tau(g, bh, days_all, rng)
        rows.append(dict(bandwidth=bh, primary=bool(bh == h), **r,
                         ci_low=round(lo, 4) if np.isfinite(lo) else None,
                         ci_high=round(hi, 4) if np.isfinite(hi) else None,
                         raw_diff_pp=round((r["p_right"] - r["p_left"]) * 100,
                                           2)))
    BE = pd.DataFrame(rows)
    BE.to_csv(RESULTS / "local_boundary_effect.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 局部边界效应（EDGE_TOUCH 已排除） ===")
    print(BE.to_string(index=False))

    gg = band(d, h).dropna(subset=["later_reclaim"])
    lo, hi = boot_tau(gg, h, days_all, rng)
    pd.DataFrame([dict(bandwidth=h, tau=fit_local(gg, h)["tau"],
                       ci_low=lo, ci_high=hi, n=len(gg))]).to_csv(
        RESULTS / "bootstrap_boundary_effect.csv", index=False,
        encoding="utf-8-sig")
    print(f"\nPRIMARY τ = {fit_local(gg, h)['tau']:.4f}  95%CI "
          f"[{lo:.4f}, {hi:.4f}]")

    # ---------- 响应曲线（EDGE_TOUCH 单独行） ----------
    def _bin(m):
        if m == 0:
            return "EDGE_TOUCH"
        for a, b in ((-0.50, -0.40), (-0.40, -0.30), (-0.30, -0.20),
                     (-0.20, -0.10), (-0.10, 0.0), (0.0, 0.10),
                     (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
                     (0.40, 0.50)):
            if m == 0:
                break
            if m > 0 and (a >= 0) and (a < m <= b):
                return f"({a:.2f},{b:.2f}]"
            if m < 0 and (b <= 0) and (a <= m < b):
                return f"[{a:.2f},{b:.2f})"
        return None

    c = d.dropna(subset=["later_reclaim"]).copy()
    c["bin"] = [ _bin(float(m)) for m in c["ob_margin_R"]]
    c = c[c["bin"].notna()]
    order = [f"[{a:.2f},{b:.2f})" for a, b in
             ((-0.50, -0.40), (-0.40, -0.30), (-0.30, -0.20), (-0.20, -0.10),
              (-0.10, 0.0))] + ["EDGE_TOUCH"] + \
            [f"({a:.2f},{b:.2f}]" for a, b in
             ((0.0, 0.10), (0.10, 0.20), (0.20, 0.30), (0.30, 0.40),
              (0.40, 0.50))]
    cur = (c.groupby("bin").agg(n=("later_reclaim", "size"),
                                later_reclaim_rate=("later_reclaim", "mean"),
                                structural_acceptance_rate=(
                                    "structural_acceptance", "mean"))
           .reindex(order).dropna(how="all").reset_index())
    cur["n"] = cur["n"].astype(int)
    for k in ("later_reclaim_rate", "structural_acceptance_rate"):
        cur[k] = cur[k].round(4)
    cur.to_csv(RESULTS / "margin_response_curve.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== 响应曲线（EDGE_TOUCH 单独行） ===")
    print(cur.to_string(index=False))

    # ---------- subgroup（各自 Gate） ----------
    sub = []
    for col, levels, label in SUBGROUPS:
        if col not in d.columns:
            continue
        for lv in levels:
            s = band(d[d[col].astype(str) == lv], h).dropna(
                subset=["later_reclaim"])
            if len(s) < 40:
                sub.append(dict(dim=label, level=lv, n=len(s),
                                note="样本不足"))
                continue
            r = fit_local(s, h)
            if r is None:
                continue
            nl, nr = r["n_left"], r["n_right"]
            spass = bool(nl >= MIN_SIDE and nr >= MIN_SIDE)
            gp, ma, mk, mc = subgroup_gate(s, h)
            infer = bool(spass and gp)
            if infer:
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
                            support_pass=spass,
                            smd_all=ma, smd_key=mk, cat_pp=mc,
                            subgroup_gate=gp, infer=infer))
    SB = pd.DataFrame(sub)
    SB.to_csv(RESULTS / "boundary_subgroups.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 预注册 subgroup（含各自 Gate） ===")
    print(SB.to_string(index=False))

    # ---------- 次级：symbol / fold ----------
    sec = []
    for col in ("symbol", "fold"):
        for lv in sorted(d[col].astype(str).unique()):
            s = band(d[d[col].astype(str) == lv], h).dropna(
                subset=["later_reclaim"])
            if len(s) < 40:
                continue
            r = fit_local(s, h)
            if r is None:
                continue
            spass = bool(r["n_left"] >= MIN_SIDE and r["n_right"] >= MIN_SIDE)
            gp, ma, mk, mc = subgroup_gate(s, h)
            if spass and gp:
                lo2, hi2 = boot_tau(s, h, s["trading_day"].dropna().unique(),
                                    rng)
            else:
                lo2, hi2 = np.nan, np.nan
            sec.append(dict(dim=col, level=lv, n=len(s),
                            n_left=r["n_left"], n_right=r["n_right"],
                            p_left=round(r["p_left"], 4),
                            p_right=round(r["p_right"], 4),
                            tau=round(r["tau"], 4),
                            ci_low=round(lo2, 4) if np.isfinite(lo2) else None,
                            ci_high=round(hi2, 4) if np.isfinite(hi2) else None,
                            smd_all=ma, smd_key=mk, cat_pp=mc,
                            support_pass=spass, subgroup_gate=gp,
                            infer=bool(spass and gp)))
    SE = pd.DataFrame(sec)
    SE.to_csv(RESULTS / "boundary_secondary.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== 按品种 / 折 ===")
    print(SE.to_string(index=False))
    print("\nBOUNDARY_V3_2_DONE")


if __name__ == "__main__":
    main()
