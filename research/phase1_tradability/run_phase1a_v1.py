"""Phase 1A — Tradability Signal Sanity Check（低成本诊断，不训练新模型）。

只用现有标签 + 现有模型配置重跑一遍确定性折叠拟合以取回逐事件分数，
然后回答：

    A. 被标成 tradable=1 的事件到底多快产生 2.5R？
       （decile × success_by_bar 在 6/12/24/48/96）
    B. 高分事件是否不仅更容易成功、而且更快成功？
       （decile × bars_to_resolution 中位数）
    C. 5m / 15m / 1h 是否表现一致？
       （source_tf 内部分十分位 × success_by_bar）
    D. rollover 跳空阈值 3 / 5 / 10 ATR 会改变多少标签？

不重训模型、不改特征、不调参。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.phase1_tradability.build_features_v1 import build
from research.phase1_tradability.build_labels_v1 import build as build_labels
from research.phase1_tradability.phase1_contract_v1 import RESULTS
from research.phase1_tradability.run_phase1_v1 import (fit_lgbm,
                                                       fit_logistic, prep)

BAR_GRID = (6, 12, 24, 48, 96)


def prep(X: pd.DataFrame, cols: list) -> np.ndarray:
    return X[cols].to_numpy(float)


def run_preds(res, feats, cat_cols):
    """重跑确定性折叠拟合，取回逐事件预测（仅此用途，不调参）。"""
    folds, _ = m2.build_folds(res["trading_day"].to_numpy(), len(res))
    specs = {"Model1_Logistic": (feats, "logistic"),
             "Model2_LightGBM": (feats, "lgbm")}
    out = []
    for fi in range(4):
        tr_d, se_d, te_d = folds[fi]
        tr = res[res["trading_day"].isin(tr_d)]
        se = res[res["trading_day"].isin(se_d)]
        te = res[res["trading_day"].isin(te_d)]
        tr_ok = tr[tr["resolution_time"] < se["decision_time"].min()]
        if len(tr_ok) < 200 or len(te) < 100:
            continue
        ytr = tr_ok["tradable"].to_numpy(int)
        for name, (cols, kind) in specs.items():
            mdl = (fit_logistic(prep(tr_ok, cols), ytr) if kind == "logistic"
                   else fit_lgbm(prep(tr_ok, cols), ytr))
            p = mdl.predict_proba(prep(te, cols))[:, 1]
            out.append(pd.DataFrame(dict(
                candidate_id=te["candidate_id"].to_numpy(),
                折=f"F{fi+1}", 模型=name, pred=p,
                actual=te["tradable"].to_numpy(int))))
    return pd.concat(out, ignore_index=True)


def add_deciles(g, col="pred"):
    g = g.copy()
    g["decile"] = pd.qcut(g[col].rank(method="first"), 10,
                          labels=False) + 1
    return g


def success_table(g, by=("decile",), label=""):
    """每个 (by) 分组在 BAR_GRID 上的累计成功发生率。"""
    rows = []
    keys = list(by)
    for k, gg in g.groupby(keys):
        k = k if isinstance(k, tuple) else (k,)
        d = dict(zip(keys, k))
        d["事件数"] = len(gg)
        succ = (gg["actual"] == 1).to_numpy()
        bars = gg["bars_to_resolution"].to_numpy(float)
        for t in BAR_GRID:
            rate = float(((succ) & (bars <= t)).mean())
            d[f"success_by_{t}bar"] = round(rate, 4)
        d["最终tradable率"] = round(float(succ.mean()), 4)
        if label:
            d["维度"] = label
        rows.append(d)
    return pd.DataFrame(rows)


def add_lift(tb, g, keys):
    """把 success_by_Xbar 转成相对同期全体的 Lift。"""
    succ = (g["actual"] == 1).to_numpy()
    bars = g["bars_to_resolution"].to_numpy(float)
    base = {t: float(((succ) & (bars <= t)).mean()) for t in BAR_GRID}
    for t in BAR_GRID:
        b = base[t]
        tb[f"Lift_by_{t}bar"] = (tb[f"success_by_{t}bar"] / b).round(4) \
            if b > 0 else np.nan
    return tb


def main():
    lab = pd.read_parquet(RESULTS / "labels_v1.parquet")
    cand = pd.read_parquet(RESULTS / "candidates_v1.parquet")
    X, contract, feats = build()

    df = (lab.merge(cand[["candidate_id", "trading_day"]],
                    on="candidate_id", how="left", validate="one_to_one")
             .merge(X, on="candidate_id", how="inner", validate="one_to_one")
             .rename(columns={"label": "tradable"}))
    res = df[df["status"] == "RESOLVED"].copy()
    res["symbol_code"] = res["symbol"].astype("category").cat.codes
    res["source_tf_code"] = res["source_tf"].astype("category").cat.codes

    P = run_preds(res, feats, ["symbol_code", "source_tf_code"])
    P = P.merge(
        res[["candidate_id", "symbol", "source_tf", "bars_to_resolution",
             "opportunity_side"]],
        on="candidate_id", how="left", validate="many_to_one")
    P.to_parquet(RESULTS / "phase1a_predictions.parquet", index=False)

    meta = res[["candidate_id", "source_tf", "symbol", "bars_to_resolution",
                "tradable"]]

    out_a, out_b, out_c = [], [], []
    for name, g in P.groupby("模型"):
        g = add_deciles(g)
        tb = success_table(g)
        tb = add_lift(tb, g, ["decile"])
        tb["模型"] = name
        out_a.append(tb)

        # B：成功速度
        rows = []
        for d, gg in g.groupby("decile"):
            s = gg[gg["actual"] == 1]["bars_to_resolution"]
            rows.append(dict(
                模型=name, decile=d, 事件数=len(gg),
                成功事件数=len(s),
                成功中位bars=round(float(s.median()), 2) if len(s) else None,
                成功均值bars=round(float(s.mean()), 2) if len(s) else None,
                失败中位bars=round(float(
                    gg[gg["actual"] == 0]["bars_to_resolution"].median()), 2),
            ))
        out_b.append(pd.DataFrame(rows))

        # C：source_tf 内部十分位
        for tf, gg in g.groupby("source_tf"):
            gg = add_deciles(gg)
            t2 = success_table(gg, label=tf)
            t2 = add_lift(t2, gg, ["decile"])
            t2["模型"] = name
            t2["source_tf"] = tf
            out_c.append(t2)

    A = pd.concat(out_a, ignore_index=True)
    B = pd.concat(out_b, ignore_index=True)
    C = pd.concat(out_c, ignore_index=True)

    # D：rollover 阈值敏感性
    base = lab.set_index("candidate_id")
    rows = []
    for thr in (3, 5, 10):
        l = build_labels(threshold=float(thr), quiet=True)
        l = l.set_index("candidate_id")
        aligned = l.reindex(base.index)
        same_status = (aligned["status"].to_numpy()
                       == base["status"].to_numpy())
        both = (aligned["label"].notna()) & (base["label"].notna())
        same_label = (aligned["label"].to_numpy() == base["label"].to_numpy())
        rows.append(dict(
            rollover_ATR阈值=thr,
            RESOLVED=int((aligned["status"] == "RESOLVED").sum()),
            AMBIGUOUS=int((aligned["status"] == "AMBIGUOUS_INTRABAR").sum()),
            ROLL_CENSORED=int((aligned["status"] == "ROLL_CENSORED").sum()),
            END_CENSORED=int(
                (aligned["status"] == "END_OF_DATA_CENSORED").sum()),
            base_rate=(round(float((aligned["label"] == 1).mean()
                                   / max(1e-12, aligned["label"].notna().mean()
                                         )), 4)),
            状态与10ATR不同=int((~same_status).sum()),
            标签与10ATR不同=int((both & ~same_label).sum()),
        ))
    D = pd.DataFrame(rows)

    A.to_csv(RESULTS / "phase1a_decile_success_by_bar.csv", index=False,
             encoding="utf-8-sig")
    B.to_csv(RESULTS / "phase1a_decile_time_to_success.csv", index=False,
             encoding="utf-8-sig")
    C.to_csv(RESULTS / "phase1a_sourcetf_decile_success_by_bar.csv",
             index=False, encoding="utf-8-sig")
    D.to_csv(RESULTS / "phase1a_rollover_sensitivity.csv", index=False,
             encoding="utf-8-sig")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    lg = A[A["模型"] == "Model1_Logistic"]
    print("\n=== A. decile × 累计成功发生率（Logistic）===")
    print(lg[["decile", "事件数"] + [f"success_by_{t}bar" for t in BAR_GRID]
              + ["最终tradable率"]].to_string(index=False))
    print("\n=== A'. 同期 Lift（相对全体）===")
    print(lg[["decile"] + [f"Lift_by_{t}bar" for t in BAR_GRID]]
          .to_string(index=False))
    print("\n=== B. decile × 成功速度（Logistic）===")
    print(B[B["模型"] == "Model1_Logistic"].to_string(index=False))
    print("\n=== C. source_tf × decile（Logistic, success_by_24bar）===")
    cl = C[C["模型"] == "Model1_Logistic"]
    print(cl.pivot_table(index="decile", columns="source_tf",
                         values="success_by_24bar").round(4).to_string())
    print("\n=== D. rollover 阈值敏感性 ===")
    print(D.to_string(index=False))
    print("\nPHASE1A_DONE")


if __name__ == "__main__":
    main()
