"""Nested metadata+state audit（Phase 1 native-direction 补实验）。

修正此前的非嵌套比较：

    旧：metadata-only  vs  state-only        （不科学）
    新：M1 metadata    vs  M3 metadata+state （嵌套增量）

M0 constant / M1 metadata / M2 state / M3 metadata+state。
zero-shot 版本不使用 symbol。
标签、candidate、feature contract、split 全部冻结，不调参。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import research.m2_nondeep_temporal_v1 as m2
from research.phase1_native_direction.nd_contract_v1 import DEV4, RESULTS

BOOT = 500
SEED = 7
META_ALL = ["symbol", "source_tf", "native_direction"]
META_ZS = ["source_tf", "native_direction"]          # zero-shot 不含 symbol


def fit_log(Xtr, ytr):
    return make_pipeline(
        SimpleImputer(strategy="median"), StandardScaler(),
        LogisticRegression(max_iter=3000, random_state=SEED)).fit(Xtr, ytr)


def _oh(tr, te, cols):
    """train-only vocabulary 的 one-hot。"""
    a = pd.get_dummies(tr[cols].astype(str), columns=cols, dtype=float)
    b = pd.get_dummies(te[cols].astype(str), columns=cols, dtype=float)
    return (a.to_numpy(float),
            b.reindex(columns=a.columns, fill_value=0.0).to_numpy(float))


def design(tr, te, meta_cols, state_cols, use_meta, use_state):
    parts_tr, parts_te = [], []
    if use_meta:
        a, b = _oh(tr, te, meta_cols)
        parts_tr.append(a)
        parts_te.append(b)
    if use_state:
        parts_tr.append(tr[state_cols].to_numpy(float))
        parts_te.append(te[state_cols].to_numpy(float))
    if not parts_tr:
        raise ValueError("empty design")
    return (np.hstack(parts_tr), np.hstack(parts_te))


def metrics(y, p):
    y = np.asarray(y, int)
    p = np.asarray(p, float)
    o = dict(n=len(y), base_rate=round(float(y.mean()), 4))
    o["AUC"] = round(float(roc_auc_score(y, p)), 4) if len(np.unique(y)) > 1 else None
    o["PR_AUC"] = round(float(average_precision_score(y, p)), 4) if len(np.unique(y)) > 1 else None
    o["Brier"] = round(float(brier_score_loss(y, p)), 6)
    m = p >= np.quantile(p, 0.8)
    r = float(y[m].mean())
    o["Top20_rate"] = round(r, 4)
    o["Lift@20"] = round(r / y.mean(), 4)
    o["Top20_uplift"] = round(r - y.mean(), 4)
    return o


def within_stratum(P, col, min_n=100):
    """symbol × source_tf × native_direction 组内排序。"""
    rows = []
    for k, g in P.groupby(["symbol", "source_tf", "native_direction"]):
        if len(g) < min_n:
            rows.append(dict(stratum="|".join(map(str, k)), n=len(g),
                             uplift="LOW_SAMPLE", lift="LOW_SAMPLE",
                             base_rate=round(float(g["y"].mean()), 4)))
            continue
        p = g[col].to_numpy(float)
        y = g["y"].to_numpy(int)
        mk = p >= np.quantile(p, 0.8)
        top = float(y[mk].mean())
        base = float(y.mean())
        rows.append(dict(stratum="|".join(map(str, k)), n=len(g),
                         base_rate=round(base, 4),
                         top20_rate=round(top, 4),
                         uplift=round(top - base, 4),
                         lift=round(top / base, 4) if base > 0 else None))
    t = pd.DataFrame(rows)
    ok = t[t["uplift"] != "LOW_SAMPLE"].copy()
    ok["uplift"] = ok["uplift"].astype(float)
    ok["lift"] = ok["lift"].astype(float)
    agg = dict(
        n_strata=len(t), usable_strata=len(ok),
        low_sample_strata=int(len(t) - len(ok)),
        event_weighted_uplift=round(float(
            (ok["uplift"] * ok["n"]).sum() / ok["n"].sum()), 4),
        macro_mean_uplift=round(float(ok["uplift"].mean()), 4),
        macro_median_uplift=round(float(ok["uplift"].median()), 4),
        event_weighted_lift=round(float(
            (ok["lift"] * ok["n"]).sum() / ok["n"].sum()), 4),
        macro_median_lift=round(float(ok["lift"].median()), 4),
        positive_strata=int((ok["uplift"] > 0).sum()),
    )
    return t, agg


def delta(m3, m1):
    return dict(
        delta_AUC=round(m3["AUC"] - m1["AUC"], 4),
        delta_PR_AUC=round(m3["PR_AUC"] - m1["PR_AUC"], 4),
        delta_Brier=round(m3["Brier"] - m1["Brier"], 6),   # 越低越好
        delta_Top20_uplift=round(m3["Top20_uplift"] - m1["Top20_uplift"], 4),
    )


def paired_boot(P, c3, c1, n=BOOT, seed=SEED):
    """同一 resample 内同时算 M3 与 M1 的增量（paired）。"""
    rng = np.random.default_rng(seed)
    days = P["trading_day"].dropna().unique()
    idx = {d: np.flatnonzero((P["trading_day"] == d).to_numpy())
           for d in days}
    y = P["y"].to_numpy(int)
    p3 = P[c3].to_numpy(float)
    p1 = P[c1].to_numpy(float)
    da, du = [], []
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        r = np.concatenate([idx[d] for d in pick])
        yy, a3, a1 = y[r], p3[r], p1[r]
        if len(np.unique(yy)) < 2:
            continue
        da.append(roc_auc_score(yy, a3) - roc_auc_score(yy, a1))
        u3 = yy[a3 >= np.quantile(a3, .8)].mean() - yy.mean()
        u1 = yy[a1 >= np.quantile(a1, .8)].mean() - yy.mean()
        du.append(u3 - u1)
    return dict(n_rep=len(da),
                delta_AUC_ci=[round(float(np.quantile(da, .025)), 4),
                              round(float(np.quantile(da, .975)), 4)],
                delta_Top20_uplift_ci=[
                    round(float(np.quantile(du, .025)), 4),
                    round(float(np.quantile(du, .975)), 4)])


def main():
    lab = pd.read_parquet(RESULTS / "native_labels_v1.parquet")
    X = pd.read_parquet(RESULTS / "native_features_v1.parquet")
    lab["candidate_id"] = lab["candidate_id"].astype(str)
    X["candidate_id"] = X["candidate_id"].astype(str)
    d = lab.merge(X, on="candidate_id", how="inner", validate="one_to_one")
    d = d[d["native_status"] == "RESOLVED"].copy()
    d["y"] = d["native_label"].astype(int)
    state = [c for c in X.columns if c != "candidate_id"]
    assert not any(c in state for c in
                   ["native_direction", "source_ob_bias", "native_label"])
    d = d.reset_index(drop=True)
    print(f"[nested] resolved={len(d)} state_dim={len(state)}", flush=True)

    folds, _ = m2.build_folds(d["trading_day"].to_numpy(), len(d))
    recs = []
    for fi in range(4):
        tr_d, se_d, te_d = folds[fi]
        tr = d[d["trading_day"].isin(tr_d)]
        se = d[d["trading_day"].isin(se_d)]
        te = d[d["trading_day"].isin(te_d)]
        if len(te) < 100:
            continue
        se_start, te_start = se["decision_time"].min(), te["decision_time"].min()
        tr_ok = tr[tr["resolution_time"] < se_start]
        if len(tr_ok) < 200:
            continue
        # ---- ALL-symbol refit 模型 ----
        for name, meta, um, us in (
            ("M1_meta", META_ALL, True, False),
            ("M2_state", META_ALL, False, True),
            ("M3_meta_state", META_ALL, True, True),
        ):
            Xtr, Xte = design(tr_ok, te, meta, state, um, us)
            m = fit_log(Xtr, tr_ok["y"].to_numpy(int))
            te = te.copy()
            te[f"p_{name}"] = m.predict_proba(Xte)[:, 1]
        # ---- zero-shot：只用 DEV4 训练，不用 symbol ----
        dtr = tr_ok[tr_ok["symbol"].isin(DEV4)]
        for name, um, us in (("ZS_M1", True, False),
                             ("ZS_M3", True, True)):
            Xtr, Xte = design(dtr, te, META_ZS, state, um, us)
            m = fit_log(Xtr, dtr["y"].to_numpy(int))
            te[f"p_{name}"] = m.predict_proba(Xte)[:, 1]
        te["折"] = f"F{fi+1}"
        recs.append(te)
        print(f"  fold F{fi+1} done (test={len(te)})", flush=True)

    P = pd.concat(recs, ignore_index=True)
    P.to_parquet(RESULTS / "nested_predictions.parquet", index=False)

    new = P[~P["symbol"].isin(DEV4)]
    dev = P[P["symbol"].isin(DEV4)]

    # ---------- 主表 ----------
    rows = []
    for uni, g in (("ALL15", P), ("DEV4", dev), ("NEW11", new)):
        br = g["y"].mean()
        rows.append(dict(universe=uni, 模型="M0_constant", **metrics(
            g["y"], np.full(len(g), br))))
        for nm, col in (("M1_metadata", "p_M1_meta"),
                        ("M2_state", "p_M2_state"),
                        ("M3_metadata_state", "p_M3_meta_state")):
            rows.append(dict(universe=uni, 模型=nm,
                             **metrics(g["y"], g[col])))
        rows.append(dict(universe=uni, 模型="ZS_M1_tf_dir",
                         **metrics(g["y"], g["p_ZS_M1"])))
        rows.append(dict(universe=uni, 模型="ZS_M3_tf_dir_state",
                         **metrics(g["y"], g["p_ZS_M3"])))
    main_df = pd.DataFrame(rows)
    main_df.to_csv(RESULTS / "nested_main.csv", index=False,
                   encoding="utf-8-sig")
    print("\n=== nested main ===")
    print(main_df[["universe", "模型", "n", "AUC", "PR_AUC", "Brier",
                   "Lift@20", "Top20_uplift"]].to_string(index=False),
          flush=True)

    # ---------- 增量 ----------
    dl = []
    for uni, g in (("ALL15", P), ("DEV4", dev), ("NEW11_zero_shot", new)):
        m1 = metrics(g["y"], g["p_M1_meta"])
        m3 = metrics(g["y"], g["p_M3_meta_state"])
        d1 = dict(universe=uni, comparison="M3 - M1 (metadata)")
        d1.update(delta(m3, m1))
        dl.append(d1)
        z1 = metrics(g["y"], g["p_ZS_M1"])
        z3 = metrics(g["y"], g["p_ZS_M3"])
        d2 = dict(universe=uni, comparison="ZS_M3 - ZS_M1 (tf+dir)")
        d2.update(delta(z3, z1))
        dl.append(d2)
    ddf = pd.DataFrame(dl)
    ddf.to_csv(RESULTS / "nested_delta.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== 增量（M3 vs M1）===")
    print(ddf.to_string(index=False), flush=True)

    # ---------- 每折 ----------
    fr = []
    for f, g in P.groupby("折"):
        for tag, c1, c3 in (("ALL15", "p_M1_meta", "p_M3_meta_state"),
                            ("NEW11_ZS", "p_ZS_M1", "p_ZS_M3")):
            gg = g if tag == "ALL15" else g[~g["symbol"].isin(DEV4)]
            a, b = metrics(gg["y"], gg[c1]), metrics(gg["y"], gg[c3])
            fr.append(dict(折=f, comparison=tag,
                           M1_AUC=a["AUC"], M3_AUC=b["AUC"],
                           delta_AUC=round(b["AUC"] - a["AUC"], 4),
                           M1_Lift20=a["Lift@20"], M3_Lift20=b["Lift@20"],
                           M1_uplift=a["Top20_uplift"],
                           M3_uplift=b["Top20_uplift"],
                           delta_uplift=round(
                               b["Top20_uplift"] - a["Top20_uplift"], 4)))
    pd.DataFrame(fr).to_csv(RESULTS / "nested_by_fold.csv", index=False,
                            encoding="utf-8-sig")
    print("\n=== by fold ===")
    print(pd.DataFrame(fr).to_string(index=False), flush=True)

    # ---------- within-stratum ----------
    ws_rows = []
    for tag, g, col in (("ALL15_M3", P, "p_M3_meta_state"),
                        ("NEW11_ZS_M3", new, "p_ZS_M3")):
        t, agg = within_stratum(g, col)
        t.to_csv(RESULTS / f"nested_within_stratum_{tag}.csv", index=False,
                 encoding="utf-8-sig")
        ws_rows.append(dict(dataset=tag, **agg))
    # DEDUP
    ded = (P.sort_values(["candidate_group_id", "decision_time"])
            .groupby("candidate_group_id", as_index=False).head(1))
    for tag, g, col in (("DEDUP_M3", ded, "p_M3_meta_state"),):
        t, agg = within_stratum(g, col)
        ws_rows.append(dict(dataset=tag, **agg))
    ws = pd.DataFrame(ws_rows)
    ws.to_csv(RESULTS / "nested_within_stratum.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== within-stratum (symbol x tf x direction) ===")
    print(ws.to_string(index=False), flush=True)

    # ---------- DEDUP ----------
    dr = []
    for tag, g in (("ALL_EVENTS", P), ("DEDUP_EVENTS", ded)):
        for nm, col in (("M1", "p_M1_meta"), ("M3", "p_M3_meta_state")):
            dr.append(dict(dataset=tag, 模型=nm,
                           **metrics(g["y"], g[col])))
    dd = pd.DataFrame(dr)
    dd.to_csv(RESULTS / "nested_dedup.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== DEDUP ===")
    print(dd[["dataset", "模型", "n", "AUC", "Lift@20",
              "Top20_uplift"]].to_string(index=False), flush=True)
    d1 = metrics(ded["y"], ded["p_M1_meta"])
    d3 = metrics(ded["y"], ded["p_M3_meta_state"])
    print("DEDUP delta:", json.dumps(delta(d3, d1), ensure_ascii=False))

    # ---------- paired bootstrap ----------
    bs = []
    bs.append(dict(dataset="ALL15", **paired_boot(
        P, "p_M3_meta_state", "p_M1_meta")))
    bs.append(dict(dataset="NEW11_zero_shot", **paired_boot(
        new, "p_ZS_M3", "p_ZS_M1")))
    bdf = pd.DataFrame(bs)
    bdf.to_csv(RESULTS / "nested_bootstrap.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== paired day-block bootstrap (500) ===")
    print(bdf.to_string(index=False), flush=True)

    (RESULTS / "nested_audit.json").write_text(json.dumps(
        dict(main=main_df.to_dict("records"), delta=ddf.to_dict("records"),
             within=ws.to_dict("records"), bootstrap=bs),
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nNESTED_AUDIT_DONE")


if __name__ == "__main__":
    main()
