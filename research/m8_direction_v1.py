"""M8：FOLLOW / FADE 方向信号可学习性与来源诊断。

彻底停止六动作联合预测，把问题拆成：

    事件 → Q模型：值不值得做 → D模型：FOLLOW 还是 FADE → 固定目标

方向标签（不再用「六动作里收益最高的是哪个」）
------------------------------------------
    follow_mean_R = mean(FOLLOW_1.5R, FOLLOW_2.0R, FOLLOW_2.5R)
    fade_mean_R   = mean(FADE_1.5R,   FADE_2.0R,   FADE_2.5R)
    direction_advantage_R = follow_mean_R - fade_mean_R
    event_quality_R       = (follow_mean_R + fade_mean_R) / 2

事件级输入
----------
一个 candidate_id 一行。优先使用状态表里的**真正事件级市场状态**
（趋势/结构/动量/DSA/OB 属性），以及 62 维中「跨 6 动作不变」的子集。
主方向模型禁止混入 target_R 相关特征（target_fit_* / target_R）
与动作相对特征（*_rel_* / trade_direction / trade_mode）。

模型：仅 LightGBM regression 与 Ridge regression，参数固定不调参。

用法：python -m research.m8_direction_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

import research.m2_nondeep_temporal_v1 as m2
from research.ob_rl_model_view_v0_spec import MODEL_FEATURES_V0
from research.rl_62d_core_v1 import ACTION_NAMES
from research.train_rl_62d_v1 import curve_metrics

OUT = Path("research/analysis_results/m8_direction")
OUT.mkdir(parents=True, exist_ok=True)

STATE = Path(
    "research/analysis_results/ob_rl_dataset_v0/ob_rl_state_v0.parquet")
ACTION = Path(
    "research/analysis_results/ob_rl_dataset_v0/ob_rl_action_v0.parquet")

A6 = list(ACTION_NAMES[1:])
COVERAGES = [0.10, 0.20, 0.30, 0.50]

# 标识 / 时间 / 权重 / 原始价格水平：不作为方向模型输入
EXCLUDE_STATE = {
    "candidate_id", "candidate_group_id", "trading_day", "touch_time",
    "touch_5m_bar_index", "decision_weight",
}
DROP_PATTERNS = ("_zone_low", "_zone_high", "_level_")

# 62 维中明确与 target_R 绑定、不得进入方向模型
TARGET_RELATED = {"target_R", "trade_direction", "trade_mode"} | {
    c for c in MODEL_FEATURES_V0 if c.startswith("target_fit_")
}

# 预注册特征分组（按状态表列名的语义模式）
GROUP_RULES = [
    ("趋势", ("swing_bias_", "internal_bias_", "_structure_bias_")),
    ("结构", ("_structure_type_", "_structure_age_", "structure_class_",
              "source_ob_structure", "source_ob_internal",
              "touch_close_beyond_far_edge", "touch_reclaimed_by_close")),
    ("动量", ("momentum_direction_", "sqzmom_val_", "sqzmom_delta_")),
    ("DSA", ("dsa_direction_", "dsa_raw_dsa_vwap_dev_pct_",
             "dsa_vwap_dev_pct_")),
    ("OB属性", ("source_ob_bias", "above_ob_bias_", "below_ob_bias_",
                "source_ob_width_atr5", "quant_", "touch_behavior",
                "touch_ordinal", "is_first_touch",
                "touch_intrabar_far_edge_breach", "group_",
                "touch_5m_bar_index")),
    ("风险几何", ("_atr_", "ob_above_atr_", "ob_below_atr_", "_relation_")),
]


def group_of(col: str) -> str:
    for g, pats in GROUP_RULES:
        for p in pats:
            if p in col:
                return g
    return "其他"


def perf(realized, day, traded):
    daily = pd.Series(realized, index=day).groupby(level=0).mean().sort_index()
    cm = curve_metrics(daily)
    tr = np.asarray(traded, dtype=bool)
    rtr = np.asarray(realized, dtype=float)[tr]
    return dict(
        opportunity_curve_cumulative_R=float(cm["累计收益"]),
        夏普率=float(cm["夏普率"]),
        最大回撤=float(cm["最大回撤"]),
        利润因子=float(m2._pf(rtr)),
        交易事件数=int(tr.sum()),
        trade_total_R=round(float(rtr.sum()), 4),
        平均R=round(float(rtr.mean()), 6) if len(rtr) else None,
    )


def fit_predict(kind, Xtr, ytr, Xte):
    if kind == "LightGBM":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05,
                              num_leaves=31, random_state=m2.SEED,
                              verbose=-1, n_jobs=1)
        m.fit(Xtr, ytr)
        return m.predict(Xte), m
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      Ridge(alpha=1.0))
    m.fit(Xtr, ytr)
    return m.predict(Xte), m


def _reg(a, p):
    if len(a) < 50 or np.std(p) < 1e-12:
        return dict(事件数=len(a), 皮尔逊=None, 斯皮尔曼=None, MAE=None,
                    RMSE=None)
    return dict(事件数=len(a),
                皮尔逊=round(float(pearsonr(p, a)[0]), 4),
                斯皮尔曼=round(float(spearmanr(p, a)[0]), 4),
                MAE=round(float(np.mean(np.abs(p - a))), 6),
                RMSE=round(float(np.sqrt(np.mean((p - a) ** 2))), 6))


def _cls(yt, score):
    if len(np.unique(yt)) < 2:
        return dict(AUC=None, 准确率=None, FOLLOW精确率=None,
                    FOLLOW召回率=None, FADE精确率=None, FADE召回率=None)
    auc = roc_auc_score(yt, score)
    yp = (np.asarray(score) > 0).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    return dict(AUC=round(float(auc), 4),
                准确率=round(float((yt == yp).mean()), 4),
                FOLLOW精确率=round(tp / (tp + fp), 4) if tp + fp else None,
                FOLLOW召回率=round(tp / (tp + fn), 4) if tp + fn else None,
                FADE精确率=round(tn / (tn + fn), 4) if tn + fn else None,
                FADE召回率=round(tn / (tn + fp), 4) if tn + fp else None)


def main():
    D = m2.load_base()
    n_kept = D["n_kept"]
    keep_ids = D["keep_ids"]
    ev_day, ev_sym = D["ev_day"], D["ev_sym"]
    R = D["R"][:, 1:]                      # (n_kept, 6)

    # ---------------- 方向标签 ----------------
    fo = R[:, 0:3].mean(axis=1)
    fa = R[:, 3:6].mean(axis=1)
    lab = pd.DataFrame({
        "candidate_id": keep_ids,
        "follow_mean_R": fo,
        "fade_mean_R": fa,
        "direction_advantage_R": fo - fa,
        "event_quality_R": (fo + fa) / 2.0,
        "dir_adv_1.5R": R[:, 0] - R[:, 3],
        "dir_adv_2.0R": R[:, 1] - R[:, 4],
        "dir_adv_2.5R": R[:, 2] - R[:, 5],
    })
    for _i, _a in enumerate(A6):
        lab[f"R_{_a}"] = R[:, _i]
    lab["symbol"] = ev_sym
    lab["day"] = ev_day
    print(f"[M8] events={n_kept}, 方向优势 std="
          f"{lab['direction_advantage_R'].std():.4f}", flush=True)

    # ---------------- 事件级特征 ----------------
    st = pd.read_parquet(STATE)
    st["candidate_id"] = st["candidate_id"].astype(str)
    st = st.set_index("candidate_id").loc[keep_ids].reset_index()
    st = st.drop(columns=[c for c in EXCLUDE_STATE if c in st.columns])
    st = st.drop(columns=[c for c in st.columns
                          if any(p in c for p in DROP_PATTERNS)])

    # 62 维中「跨 6 动作不变」的子集（事件级，可安全使用）
    act = pd.read_parquet(ACTION)
    act["candidate_id"] = act["candidate_id"].astype(str)
    act = act.sort_values(["candidate_id", "action"])
    nun = act.groupby("candidate_id")[list(MODEL_FEATURES_V0)].nunique()
    invariant = [c for c in MODEL_FEATURES_V0
                 if (nun[c] <= 1).all() and c not in TARGET_RELATED]
    a_first = act.groupby("candidate_id").first().loc[keep_ids]
    for c in invariant:
        st[c] = a_first[c].to_numpy()

    # 输出完整 feature → group mapping（含 62 维分类）
    mapping = []
    for c in st.columns:
        if c in ("candidate_id",):
            continue
        if c in MODEL_FEATURES_V0:
            kind = ("A_事件级(62维跨动作不变)" if c in invariant
                    else ("C_targetR相关" if c in TARGET_RELATED
                          else "B/D_动作相对或未用"))
        else:
            kind = "事件级状态表"
        mapping.append(dict(特征=c, 来源=kind, 分组=group_of(c)))
    for c in MODEL_FEATURES_V0:            # 全部 62 维都登记分类
        if c in TARGET_RELATED:
            k = "C_targetR相关"
        elif c in invariant:
            k = "A_事件级(跨动作不变)"
        else:
            k = "B/D_动作相对(未用于方向模型)"
        mapping.append(dict(特征=c, 来源=k, 分组=group_of(c)))
    map_df = pd.DataFrame(mapping).drop_duplicates(subset=["特征"])
    map_df.to_csv(OUT / "m8_feature_mapping.csv", index=False,
                  encoding="utf-8-sig")

    # 编码
    feat_cols = [c for c in st.columns if c != "candidate_id"]
    for c in feat_cols:
        if not pd.api.types.is_numeric_dtype(st[c]):
            st[c] = st[c].astype(str).astype("category").cat.codes.astype(
                "int16")
    X = st[feat_cols].to_numpy(float)
    print(f"[M8] event features={len(feat_cols)} "
          f"(62维中事件级不变 {len(invariant)})", flush=True)

    folds, uniq = m2.build_folds(ev_day, n_kept)
    fold_train = [f[0] for f in folds]
    fold_sel = [f[1] for f in folds]
    fold_test = [f[2] for f in folds]

    groups = {c: group_of(c) for c in feat_cols}
    uniq_groups = sorted(set(groups.values()))

    # ---------------- 逐折训练 D / Q ----------------
    preds = []
    for fi in range(4):
        tr_m = np.isin(ev_day, list(fold_train[fi]))
        se_m = np.isin(ev_day, list(fold_sel[fi]))
        te_m = np.isin(ev_day, list(fold_test[fi]))
        Xtr, Xte, Xse = X[tr_m], X[te_m], X[se_m]
        for tgt in ("direction_advantage_R", "event_quality_R"):
            ytr = lab[tgt].to_numpy(float)[tr_m]
            for kind in ("LightGBM", "Ridge"):
                pte, mdl = fit_predict(kind, Xtr, ytr, Xte)
                # 选择段也要预测：Q 的覆盖率阈值必须由选择段冻结
                pse = mdl.predict(Xse)
                for seg, mask, pv in (("sel", se_m, pse), ("test", te_m, pte)):
                    sub = lab[mask].copy()
                    sub["折"] = f"F{fi+1}"
                    sub["segment"] = seg
                    sub["目标"] = tgt
                    sub["模型"] = kind
                    sub["pred"] = pv
                    preds.append(sub)
        print(f"[M8] F{fi+1}: train={int(tr_m.sum())} test={int(te_m.sum())}",
              flush=True)
    P = pd.concat(preds, ignore_index=True)

    reg_rows, dec_rows, cls_rows, econ_rows = [], [], [], []
    fold_rows, sym_rows, imp_rows, abl_rows, tgt_rows = [], [], [], [], []

    # 诊断一律只用测试段；选择段仅用于冻结覆盖率
    Dp = P[(P["目标"] == "direction_advantage_R") & (P["segment"] == "test")]
    Qp = P[(P["目标"] == "event_quality_R") & (P["segment"] == "test")]

    # ---------------- D 模型诊断 ----------------
    for kind, g in Dp.groupby("模型"):
        a = g["direction_advantage_R"].to_numpy(float)
        p = g["pred"].to_numpy(float)
        reg_rows.append(dict(模型=kind, 范围="合并", **_reg(a, p)))
        for f_, gg in g.groupby("折"):
            reg_rows.append(dict(
                模型=kind, 范围=f_,
                **_reg(gg["direction_advantage_R"].to_numpy(float),
                       gg["pred"].to_numpy(float))))
        for q, gg in g.groupby(pd.qcut(g["pred"], 10, labels=False,
                                       duplicates="drop")):
            dec_rows.append(dict(
                模型=kind, 十分位=int(q) + 1, 事件数=len(gg),
                预测方向优势均值=round(float(gg["pred"].mean()), 6),
                真实方向优势均值=round(
                    float(gg["direction_advantage_R"].mean()), 6),
                FOLLOW事后较优比例=round(
                    float((gg["direction_advantage_R"] > 0).mean()), 4)))
        m = ~np.isclose(a, 0, atol=1e-12)
        yt = (a[m] > 0).astype(int)
        cls_rows.append(dict(模型=f"D-{kind}", 样本数=int(m.sum()),
                             **_cls(yt, p[m])))
        # 基线
        cls_rows.append(dict(模型="always_FOLLOW", 样本数=int(m.sum()),
                             **_cls(yt, np.ones(m.sum()))))
        cls_rows.append(dict(模型="always_FADE", 样本数=int(m.sum()),
                             **_cls(yt, -np.ones(m.sum()))))
        rng = np.random.RandomState(m2.SEED).randn(m.sum())
        cls_rows.append(dict(模型="random_50_50", 样本数=int(m.sum()),
                             **_cls(yt, rng)))
        maj = 1 if yt.mean() >= 0.5 else 0
        cls_rows.append(dict(模型="majority_direction", 样本数=int(m.sum()),
                             **_cls(yt, np.full(m.sum(), maj))))
        # 经济方向增量
        chosen = np.where(p > 0, g["follow_mean_R"], g["fade_mean_R"])
        econ_rows.append(dict(
            模型=f"D-{kind}", 事件数=len(g),
            模型方向平均R=round(float(chosen.mean()), 6),
            always_FOLLOW平均R=round(float(g["follow_mean_R"].mean()), 6),
            always_FADE平均R=round(float(g["fade_mean_R"].mean()), 6),
            random方向期望R=round(float((g["follow_mean_R"].mean()
                                         + g["fade_mean_R"].mean()) / 2), 6),
            oracle方向平均R=round(float(
                np.maximum(g["follow_mean_R"], g["fade_mean_R"]).mean()), 6),
            方向命中率=round(float(
                ((p > 0) == (g["direction_advantage_R"] > 0)).mean()), 4),
            相对random增量=round(float(
                chosen.mean() - (g["follow_mean_R"].mean()
                                 + g["fade_mean_R"].mean()) / 2), 6)))
        for f_, gg in g.groupby("折"):
            aa = gg["direction_advantage_R"].to_numpy(float)
            pp = gg["pred"].to_numpy(float)
            ch = np.where(pp > 0, gg["follow_mean_R"], gg["fade_mean_R"])
            fold_rows.append(dict(
                模型=kind, 折=f_,
                **_reg(aa, pp),
                方向命中率=round(float(((pp > 0) == (aa > 0)).mean()), 4),
                模型方向平均R=round(float(ch.mean()), 6),
                相对random增量=round(float(
                    ch.mean() - (gg["follow_mean_R"].mean()
                                 + gg["fade_mean_R"].mean()) / 2), 6)))
        for s, gg in g.groupby("symbol"):
            aa = gg["direction_advantage_R"].to_numpy(float)
            pp = gg["pred"].to_numpy(float)
            ch = np.where(pp > 0, gg["follow_mean_R"], gg["fade_mean_R"])
            sym_rows.append(dict(
                模型=kind, 品种=s,
                **_reg(aa, pp),
                方向命中率=round(float(((pp > 0) == (aa > 0)).mean()), 4),
                模型方向平均R=round(float(ch.mean()), 6),
                相对random增量=round(float(
                    ch.mean() - (gg["follow_mean_R"].mean()
                                 + gg["fade_mean_R"].mean()) / 2), 6)))

        # 三个目标稳健性（同一 D 模型方向，不据此选择目标）
        for tgt in ("1.5R", "2.0R", "2.5R"):
            cf, ca = f"R_FOLLOW_{tgt}", f"R_FADE_{tgt}"
            fv = g[cf].to_numpy(float)
            av = g[ca].to_numpy(float)
            pp = g["pred"].to_numpy(float)
            ch = np.where(pp > 0, fv, av)
            tgt_rows.append(dict(
                模型=kind, 目标R=tgt,
                方向命中率=round(float(
                    ((pp > 0) == ((fv - av) > 0)).mean()), 4),
                模型方向平均R=round(float(ch.mean()), 6),
                always_FOLLOW平均R=round(float(fv.mean()), 6),
                always_FADE平均R=round(float(av.mean()), 6),
                random方向期望R=round(float((fv.mean() + av.mean()) / 2), 6),
                经济方向增量=round(float(
                    ch.mean() - (fv.mean() + av.mean()) / 2), 6)))
        print(f"[M8] D-{kind} done", flush=True)

    # ---------------- Q × D 交叉 ----------------
    cross_rows = []
    for fi in range(4):
        d_te = Dp[(Dp["折"] == f"F{fi+1}") & (Dp["模型"] == "LightGBM")
                  & (Dp["segment"] == "test")]
        q_te = Qp[(Qp["折"] == f"F{fi+1}") & (Qp["模型"] == "LightGBM")
                  & (Qp["segment"] == "test")]
        # 选择段需从未按 segment 过滤的 P 中取（Qp 已限定 test）
        q_se = P[(P["目标"] == "event_quality_R")
                 & (P["折"] == f"F{fi+1}") & (P["模型"] == "LightGBM")
                 & (P["segment"] == "sel")]
        # Q 覆盖率：选择段按事件质量均值选最优，阈值冻结
        qsel = q_se["pred"].to_numpy()
        qual = q_se["event_quality_R"].to_numpy()
        best = None
        for c in COVERAGES:
            cut = np.quantile(qsel, 1 - c)
            sel = qsel > cut
            if sel.sum() == 0:
                continue
            sh = float(qual[sel].mean())
            if best is None or sh > best[0]:
                best = (sh, cut, c)
        cut = best[1]
        for name, use_q, use_d in (
            ("A_无筛选+alwaysFOLLOW", False, False),
            ("B_无筛选+模型方向", False, True),
            ("C_Q筛选+alwaysFOLLOW", True, False),
            ("D_Q筛选+模型方向", True, True),
        ):
            qte = q_te["pred"].to_numpy()
            sel = (qte > cut) if use_q else np.ones(len(q_te), dtype=bool)
            dte = d_te["pred"].to_numpy()
            if use_d:
                r = np.where(dte > 0, d_te["follow_mean_R"].to_numpy(),
                             d_te["fade_mean_R"].to_numpy())
            else:
                r = d_te["follow_mean_R"].to_numpy()
            realized = np.where(sel, r, 0.0)
            cross_rows.append(dict(
                折=f"F{fi+1}", 组合=name,
                **perf(realized, d_te["day"].to_numpy(), sel)))

    # ---------------- 特征组消融（仅 LightGBM，一次删一组）----------------
    def run_ablation(keep_groups):
        cols = [c for c in feat_cols if groups.get(c) in keep_groups]
        idx = [feat_cols.index(c) for c in cols]
        ps, aa = [], []
        for fi in range(4):
            tr_m = np.isin(ev_day, list(fold_train[fi]))
            te_m = np.isin(ev_day, list(fold_test[fi]))
            ytr = lab["direction_advantage_R"].to_numpy(float)[tr_m]
            p, _ = fit_predict("LightGBM", X[np.ix_(tr_m, idx)], ytr,
                               X[np.ix_(te_m, idx)])
            ps.append(p)
            aa.append(lab["direction_advantage_R"].to_numpy(float)[te_m])
        p = np.concatenate(ps)
        a = np.concatenate(aa)
        sp = spearmanr(p, a)[0]
        inc = float(np.where(p > 0, 1, 0).mean() * 0)  # 占位
        return sp

    abl_rows.append(dict(变体="ALL", 特征数=len(feat_cols),
                         方向斯皮尔曼=round(float(run_ablation(
                             set(uniq_groups))), 4)))
    for gname in uniq_groups:
        keep = set(uniq_groups) - {gname}
        n = len([c for c in feat_cols if groups.get(c) in keep])
        abl_rows.append(dict(变体=f"ALL minus {gname}", 特征数=n,
                             方向斯皮尔曼=round(float(run_ablation(keep)), 4)))

    # ---------------- 特征重要性 ----------------
    tr_m0 = np.isin(ev_day, list(fold_train[3]))
    _, mdl = fit_predict("LightGBM", X[tr_m0],
                         lab["direction_advantage_R"].to_numpy(float)[tr_m0],
                         X[tr_m0])
    try:
        for f, v in zip(feat_cols, mdl.feature_importances_):
            imp_rows.append(dict(特征=f, 分组=groups.get(f),
                                 重要性=int(v)))
    except Exception:
        pass

    def dump(rows, name):
        df = pd.DataFrame(rows)
        df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
        return df

    dump(reg_rows, "m8_regression.csv")
    dump(dec_rows, "m8_decile.csv")
    dump(cls_rows, "m8_classification.csv")
    dump(econ_rows, "m8_economic.csv")
    dump(fold_rows, "m8_by_fold.csv")
    dump(sym_rows, "m8_by_symbol.csv")
    dump(cross_rows, "m8_qd_cross.csv")
    dump(tgt_rows, "m8_target_robustness.csv")
    dump(abl_rows, "m8_ablation.csv")
    imp_df = dump(imp_rows, "m8_feature_importance.csv")
    if len(imp_df):
        imp_df.groupby("分组")["重要性"].sum().sort_values(
            ascending=False).to_csv(OUT / "m8_feature_importance_group.csv",
                                    encoding="utf-8-sig")

    (OUT / "m8_audit.json").write_text(json.dumps(dict(
        script="research/m8_direction_v1.py",
        events=int(n_kept), event_features=len(feat_cols),
        invariant_from_62=invariant,
        target_related_excluded=sorted(TARGET_RELATED),
        groups={g: len([c for c in feat_cols if groups.get(c) == g])
                for g in uniq_groups},
        coverages=COVERAGES, seed=m2.SEED,
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    print("\n=== D 模型回归 ===")
    print(pd.DataFrame(reg_rows).round(6).to_string(index=False))
    print("\n=== D 模型十分位 ===")
    print(pd.DataFrame(dec_rows).round(6).to_string(index=False))
    print("\n=== 方向分类 ===")
    print(pd.DataFrame(cls_rows).round(6).to_string(index=False))
    print("\n=== 经济方向增量 ===")
    print(pd.DataFrame(econ_rows).round(6).to_string(index=False))
    print("\n=== Q×D 交叉 ===")
    print(pd.DataFrame(cross_rows).round(6).to_string(index=False))
    print("\n=== 1.5/2.0/2.5R 方向稳健性 ===")
    print(pd.DataFrame(tgt_rows).round(6).to_string(index=False))
    print("\n=== 特征组消融 ===")
    print(pd.DataFrame(abl_rows).round(6).to_string(index=False))
    print("\nM8_DONE")


if __name__ == "__main__":
    main()
