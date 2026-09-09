"""M6：最优停止可学习性实验（直接监督「继续优势」）。

核心问题
--------
当前状态信息能不能预测：

    「继续持有」相对于「现在立即退出」的价值差？

M5 已证明动态退出理论空间极大（2.5R 夏普 1.44 → 13.30，
平均每笔 +0.740R），而现有 FVI 只有 −0.089R/笔。
M6 判断这是「状态没有信息」还是「价值迭代没学到」。

关键建模变化：不用模型自己倒推标签，直接用**真实事后路径**构造监督目标。

对每个真实决策点 step 1..11：

    oracle_future_value_R =
        当前选择 HOLD 后，后续所有合法退出时点与最终终值中的事后最高收益

    oracle_continuation_advantage_R = oracle_future_value_R - exit_value_R

    > 0  → 事后应 HOLD；< 0 → 事后应 EXIT；= 0 → TIE（单独统计）

注意：HOLD 当前 K 线后若立即触发 STOP/TARGET/GAP_*/BOTH，
未来价值直接由该强制终值决定，不越过已 terminal 的路径。

模型
----
- 模型一：全步骤共享 LightGBM（step / bars_remaining 作为状态变量）
- 模型二：线性回归（仅缺失填充 + 标准化）作为最低复杂度基准
- 对照：现有 FVI 结果（不重调，直接引用 rl_exit_v1）

时间切分：事件级先切分，再展开该事件的轨迹状态。
入场方向由每折自己的 LightGBM 快照模型给出（train 段事件需按折重新预测，
因为 m2fix_events 只保存了 sel/test 段）。

用法：python -m research.m6_learnability_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score

import research.m2_nondeep_temporal_v1 as m2
from research.rl_62d_core_v1 import ACTION_NAMES
from research.train_rl_62d_v1 import curve_metrics

OUT = Path("research/analysis_results/m6_learnability")
OUT.mkdir(parents=True, exist_ok=True)

TRAJ = Path(
    "research/analysis_results/rl_exit_v1/rl_exit_trajectory_v1.parquet")
DT_AUDIT = Path(
    "research/analysis_results/m3_direction_target/m3dt_audit.json")

TARGETS = ["1.5R", "2.0R", "2.5R"]
A6 = list(ACTION_NAMES[1:])
FOLLOW_COLS = [a for a in A6 if a.startswith("FOLLOW")]
FADE_COLS = [a for a in A6 if a.startswith("FADE")]
COV_KEY = {"1.5R": "简化A_覆盖率", "2.0R": "简化B_覆盖率",
           "2.5R": "简化C_覆盖率"}
FORCED = {"GAP_STOP", "GAP_TARGET"}

POS_NUM = [
    "current_unrealized_R", "distance_to_stop_R", "distance_to_target_R",
    "bars_held", "bars_remaining", "trade_direction", "target_R",
    "max_favorable_excursion_R_so_far", "max_adverse_excursion_R_so_far",
    "previous_step_price_change_R", "minutes_since_previous_valid_bar",
]
R_BINS = [-np.inf, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 1.00, np.inf]
R_LABELS = ["R<-0.75", "-0.75~-0.50", "-0.50~-0.25", "-0.25~0",
            "0~0.25", "0.25~0.50", "0.50~1.00", "R>=1.00"]
STEP_BINS = [0, 2, 4, 6, 8, 11]
STEP_LABELS = ["1-2", "3-4", "5-6", "7-8", "9-11"]


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
        交易次数=int(tr.sum()),
        trade_total_R=round(float(rtr.sum()), 4),
        平均交易R=round(float(rtr.mean()), 6) if len(rtr) else None,
    )


def build_states(traj, keys, t):
    """keys: DataFrame(candidate_id, action)。构造决策状态 + 事后继续优势。"""
    st = traj.merge(keys, on="candidate_id", how="inner")
    st = st[st["initial_action"] == st["action"]]
    st = st.sort_values(["candidate_id", "step"])
    dec = st[st["is_decision_point"] & ~st["terminal_reason"].isin(FORCED)]
    parts = []
    for cid, g in dec.groupby("candidate_id", sort=False):
        vals = g["exit_value_R"].to_numpy(float)
        term = float(g["all_hold_terminal_R"].iloc[0])
        n = len(vals)
        if n == 0:
            continue
        sm = np.maximum.accumulate(vals[::-1])[::-1]
        suf = np.full(n, -np.inf)
        suf[:-1] = sm[1:]
        future = np.maximum(suf, term)
        gg = g.copy()
        gg["oracle_future_value_R"] = future
        gg["oracle_continuation_advantage_R"] = future - vals
        parts.append(gg)
    out = pd.concat(parts, ignore_index=True)
    out["目标"] = t
    return out


def fit_predict(kind, Xtr, ytr, Xte):
    if kind == "LightGBM":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=m2.SEED, verbose=-1, n_jobs=1)
        m.fit(Xtr, ytr)
        return m.predict(Xte)
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                      Ridge(alpha=1.0))
    m.fit(Xtr, ytr)
    return m.predict(Xte)


def _dist(a):
    a = np.asarray(a, dtype=float)
    return dict(
        平均继续优势=round(float(a.mean()), 6),
        中位数=round(float(np.median(a)), 6),
        标准差=round(float(a.std()), 6),
        HOLD比例=round(float((a > 1e-12).mean()), 4),
        EXIT比例=round(float((a < -1e-12).mean()), 4),
        TIE比例=round(float(np.isclose(a, 0, atol=1e-12).mean()), 4))


def _reg(a, p):
    if len(a) < 50 or np.std(p) < 1e-12:
        return dict(状态数=len(a), 皮尔逊=None, 斯皮尔曼=None, MAE=None,
                    RMSE=None)
    return dict(
        状态数=len(a),
        皮尔逊=round(float(pearsonr(p, a)[0]), 4),
        斯皮尔曼=round(float(spearmanr(p, a)[0]), 4),
        MAE=round(float(np.mean(np.abs(p - a))), 6),
        RMSE=round(float(np.sqrt(np.mean((p - a) ** 2))), 6))


def cls_metrics(y_true_adv, y_pred_adv):
    m = ~np.isclose(y_true_adv, 0, atol=1e-12)
    if m.sum() < 50 or len(np.unique(y_true_adv[m] > 0)) < 2:
        return dict(样本数=int(m.sum()), AUC=None, 准确率=None,
                    HOLD精确率=None, HOLD召回率=None, EXIT精确率=None,
                    EXIT召回率=None)
    yt = (y_true_adv[m] > 0).astype(int)
    yp = (y_pred_adv[m] > 0).astype(int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    return dict(
        样本数=int(m.sum()),
        AUC=round(float(roc_auc_score(yt, y_pred_adv[m])), 4),
        准确率=round(float((yt == yp).mean()), 4),
        HOLD精确率=round(tp / (tp + fp), 4) if tp + fp else None,
        HOLD召回率=round(tp / (tp + fn), 4) if tp + fn else None,
        EXIT精确率=round(tn / (tn + fn), 4) if tn + fn else None,
        EXIT召回率=round(tn / (tn + fp), 4) if tn + fp else None)


def _simple(yt, yp):
    yt = np.asarray(yt, dtype=int)
    yp = np.asarray(yp, dtype=int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    return dict(样本数=len(yt), AUC=None,
                准确率=round(float((yt == yp).mean()), 4),
                HOLD精确率=round(tp / (tp + fp), 4) if tp + fp else None,
                HOLD召回率=round(tp / (tp + fn), 4) if tp + fn else None,
                EXIT精确率=round(tn / (tn + fn), 4) if tn + fn else None,
                EXIT召回率=round(tn / (tn + fp), 4) if tn + fp else None)


def _replay(g, epi):
    """epi: 该折【全部】测试事件（含无决策点的 episode），保证不丢样本。"""
    g = g.sort_values(["candidate_id", "step"])
    out, early, held = {}, {}, {}
    for cid, gg in g.groupby("candidate_id", sort=False):
        term = float(gg["all_hold_terminal_R"].iloc[0])
        o, b, e = term, int(gg["step"].max()), False
        for _, r in gg.iterrows():
            if float(r["pred_adv"]) > 0:
                continue
            o, b, e = float(r["exit_value_R"]), int(r["step"]), True
            break
        out[cid], held[cid], early[cid] = o, b, e
    cids = epi["candidate_id"].to_numpy()
    traded = epi["traded"].to_numpy().astype(bool)
    day = epi["day"].to_numpy()
    term_map = dict(zip(epi["candidate_id"].to_numpy(),
                        epi["term_R"].to_numpy()))
    # 无决策点的 episode 无法提前退出，只能取固定终值
    realized = np.array([out.get(c, term_map[c]) for c in cids])
    realized = np.where(traded, realized, 0.0)
    # 无决策点的 episode：未提前退出、持有 0 根
    ev_arr = np.array([early.get(c, False) for c in cids])[traded]
    hd = np.array([held.get(c, 0) for c in cids])[traded]
    d = perf(realized, day, traded)
    d["提前退出比例"] = round(float(ev_arr.mean()), 4) if len(ev_arr) else None
    d["平均持有K线数"] = round(float(hd.mean()), 4) if len(hd) else None
    return d


def main():
    traj = pd.read_parquet(TRAJ)
    D = m2.load_base()
    n_kept = D["n_kept"]
    ev_day = D["ev_day"]
    ev_sym = D["ev_sym"]
    keep_ids = D["keep_ids"]
    y_rows = D["y_rows"]

    temporal = m2.build_temporal(ev_sym, D["ev_bar"])
    Xsnap, Xtemp, cat = m2.build_feature_matrices(D, temporal)
    folds, uniq = m2.build_folds(ev_day, n_kept)
    fold_train_days = [f[0] for f in folds]
    fold_se_days = [f[1] for f in folds]
    fold_test_days = [f[2] for f in folds]

    audit = json.load(open(DT_AUDIT))
    fold_cov = {t: {int(p["折"]): p[COV_KEY[t]] for p in audit["fold_plan"]}
                for t in TARGETS}

    # 特征列（排除一切泄漏字段）
    feat_cols = (["step", "bars_remaining"]
                 + [c for c in traj.columns if c.startswith("bg_")]
                 + [c for c in traj.columns if c.startswith("dyn_")]
                 + POS_NUM + ["crossed_session_break"])
    for c in feat_cols:
        if not pd.api.types.is_numeric_dtype(traj[c]):
            traj[c] = traj[c].astype(str).astype("category").cat.codes.astype(
                "int16")
    traj["crossed_session_break"] = traj["crossed_session_break"].astype("int8")

    # ---------- 每折：入场方向（train 段事件需按折重新预测）----------
    dir_frames = []
    for fi in range(4):
        tr_mask = m2.day_mask(ev_day, fold_train_days[fi])
        se_mask = m2.day_mask(ev_day, fold_se_days[fi])
        te_mask = m2.day_mask(ev_day, fold_test_days[fi])
        # 选择段也要预测：覆盖率阈值必须由选择段冻结
        use = tr_mask | se_mask | te_mask
        tr_row = np.repeat(tr_mask, 6)
        use_row = np.repeat(use, 6)
        Xsnap_num, Xtemp_num = m2.encode_for_fold(Xsnap, Xtemp, cat, tr_row)
        Xn = Xsnap_num                      # 快照视图（最终入场规则）
        Xtr = Xn.iloc[tr_row].to_numpy(float)
        ytr = y_rows[tr_row]
        yhat = m2.train_reg("lgbm", Xtr, ytr,
                            Xn.iloc[use_row].to_numpy(float), cat, m2.SEED)
        ye = yhat.reshape(-1, 6)
        fs = ye[:, [A6.index(c) for c in FOLLOW_COLS]].mean(axis=1)
        fd = ye[:, [A6.index(c) for c in FADE_COLS]].mean(axis=1)
        isf = fs >= fd
        sub = pd.DataFrame(dict(
            candidate_id=keep_ids[use],
            fold=fi + 1,
            day=ev_day[use],
            is_follow=isf,
            event_score=np.where(isf, fs, fd),
            is_test=te_mask[use],
            is_sel=se_mask[use],
        ))
        dir_frames.append(sub)
        print(f"[M6] F{fi+1}: pool events={len(sub)} (train={int(tr_mask.sum())},"
              f" test={int(te_mask.sum())})", flush=True)
    dirs = pd.concat(dir_frames, ignore_index=True)

    dist_rows, reg_rows, dec_rows, cls_rows = [], [], [], []
    econ_rows, strat_rows, replay_rows, pooled = [], [], [], []

    for t in TARGETS:
        all_states, fold_states, epi_frames = [], [], []
        for fi in range(4):
            d = dirs[dirs["fold"] == fi + 1]
            act = np.where(d["is_follow"].to_numpy(), f"FOLLOW_{t}",
                           f"FADE_{t}")
            keys = pd.DataFrame({"candidate_id": d["candidate_id"].to_numpy(),
                                 "action": act})
            # 覆盖率阈值必须由【选择段】冻结，再应用到测试段
            se = d[d["is_sel"]]
            te = d[d["is_test"]]
            cutoff = np.quantile(se["event_score"].to_numpy(),
                                 1 - fold_cov[t][fi + 1])

            # 该折全部测试事件（含无决策点 episode），用于无偏回放
            te_ids = te["candidate_id"].to_numpy()
            te_act = act[d["is_test"].to_numpy()]
            te_keys = pd.DataFrame({"candidate_id": te_ids, "action": te_act})
            te_tr = traj.merge(te_keys, on="candidate_id", how="inner")
            te_tr = te_tr[te_tr["initial_action"] == te_tr["action"]]
            term_s = te_tr.groupby("candidate_id")["all_hold_terminal_R"].first()
            epi = pd.DataFrame({"candidate_id": te_ids})
            epi["term_R"] = epi["candidate_id"].map(term_s).astype(float)
            epi["traded"] = (te["event_score"].to_numpy() > cutoff)
            epi["day"] = te["day"].to_numpy()
            epi = epi.dropna(subset=["term_R"]).reset_index(drop=True)
            epi["折"] = f"F{fi+1}"
            epi_frames.append(epi)

            st = build_states(traj, keys, t)
            if len(st) == 0:
                continue
            tr_map = dict(zip(te["candidate_id"].to_numpy(),
                              (te["event_score"].to_numpy() > cutoff)))
            st["traded"] = st["candidate_id"].map(tr_map).fillna(False).astype(bool)
            st["折"] = f"F{fi+1}"
            st["方向"] = np.where(st["action"].str.startswith("FOLLOW"),
                                 "FOLLOW", "FADE")

            tr_m = st["trading_day"].isin(fold_train_days[fi]).to_numpy()
            te_m = st["trading_day"].isin(fold_test_days[fi]).to_numpy()
            if tr_m.sum() < 500 or te_m.sum() < 100:
                continue
            Xtr = st.loc[tr_m, feat_cols].to_numpy(float)
            ytr = st.loc[tr_m, "oracle_continuation_advantage_R"].to_numpy(float)
            Xte = st.loc[te_m, feat_cols].to_numpy(float)
            cols = ["candidate_id", "symbol", "trading_day", "step",
                    "exit_value_R", "all_hold_terminal_R",
                    "oracle_continuation_advantage_R", "traded"]
            for kind in ("LightGBM", "线性"):
                pred = fit_predict(kind, Xtr, ytr, Xte)
                sub = st.loc[te_m, cols].copy()
                sub["模型"] = kind
                sub["pred_adv"] = pred
                sub["折"] = f"F{fi+1}"
                sub["目标"] = t
                sub["方向"] = st.loc[te_m, "方向"].to_numpy()
                fold_states.append(sub)
            all_states.append(st)
            print(f"[M6]   {t} F{fi+1}: train={int(tr_m.sum())} "
                  f"test={int(te_m.sum())}", flush=True)

        full = pd.concat(all_states, ignore_index=True)
        dist_rows.append(dict(维度="整体", 取值="ALL", 目标=t, 状态数=len(full),
                              **_dist(full["oracle_continuation_advantage_R"])))
        for s, g in full.groupby("step"):
            dist_rows.append(dict(维度="step", 取值=int(s), 目标=t,
                                  状态数=len(g),
                                  **_dist(g["oracle_continuation_advantage_R"])))
        for d_, g in full.groupby("方向"):
            dist_rows.append(dict(维度="方向", 取值=d_, 目标=t, 状态数=len(g),
                                  **_dist(g["oracle_continuation_advantage_R"])))
        for s, g in full.groupby("symbol"):
            dist_rows.append(dict(维度="品种", 取值=s, 目标=t, 状态数=len(g),
                                  **_dist(g["oracle_continuation_advantage_R"])))
        for fi, g in full.groupby("折"):
            dist_rows.append(dict(维度="折", 取值=fi, 目标=t, 状态数=len(g),
                                  **_dist(g["oracle_continuation_advantage_R"])))

        test = pd.concat(fold_states, ignore_index=True)
        pooled.append(test)

        for kind, g in test.groupby("模型"):
            a = g["oracle_continuation_advantage_R"].to_numpy(float)
            p = g["pred_adv"].to_numpy(float)
            reg_rows.append(dict(目标=t, 模型=kind, 范围="合并", **_reg(a, p)))
            for f_, gg in g.groupby("折"):
                reg_rows.append(dict(
                    目标=t, 模型=kind, 范围=f_,
                    **_reg(gg["oracle_continuation_advantage_R"].to_numpy(float),
                           gg["pred_adv"].to_numpy(float))))
            for q, gg in g.groupby(pd.qcut(g["pred_adv"], 10, labels=False,
                                           duplicates="drop")):
                dec_rows.append(dict(
                    目标=t, 模型=kind, 预测十分位=int(q) + 1, 状态数=len(gg),
                    预测平均=round(float(gg["pred_adv"].mean()), 6),
                    真实平均=round(float(
                        gg["oracle_continuation_advantage_R"].mean()), 6),
                    真实中位数=round(float(
                        gg["oracle_continuation_advantage_R"].median()), 6),
                    真实HOLD比例=round(float(
                        (gg["oracle_continuation_advantage_R"] > 0).mean()), 4)))
            cls_rows.append(dict(目标=t, 模型=kind, 范围="合并",
                                 **cls_metrics(a, p)))
            m = ~np.isclose(a, 0, atol=1e-12)
            if m.sum() >= 50:
                yt = (a[m] > 0).astype(int)
                cls_rows.append(dict(目标=t, 模型=f"{kind}(基准:永远HOLD)",
                                     范围="合并",
                                     **_simple(yt, np.ones(m.sum()))))
                cls_rows.append(dict(目标=t, 模型=f"{kind}(基准:永远EXIT)",
                                     范围="合并",
                                     **_simple(yt, np.zeros(m.sum()))))
                maj = 1 if yt.mean() >= 0.5 else 0
                cls_rows.append(dict(目标=t, 模型=f"{kind}(基准:多数类)",
                                     范围="合并",
                                     **_simple(yt, np.full(m.sum(), maj))))
            pe, ph = g[g["pred_adv"] <= 0], g[g["pred_adv"] > 0]
            econ_rows.append(dict(
                目标=t, 模型=kind,
                预测EXIT_真实优势=(round(float(
                    pe["oracle_continuation_advantage_R"].mean()), 6)
                    if len(pe) else None),
                预测HOLD_真实优势=(round(float(
                    ph["oracle_continuation_advantage_R"].mean()), 6)
                    if len(ph) else None),
                两组差值=(round(float(
                    ph["oracle_continuation_advantage_R"].mean()
                    - pe["oracle_continuation_advantage_R"].mean()), 6)
                    if len(pe) and len(ph) else None)))
            g = g.copy()
            g["浮盈亏桶"] = pd.cut(g["exit_value_R"], bins=R_BINS,
                                  labels=R_LABELS, right=False)
            g["持仓步数桶"] = pd.cut(g["step"], bins=STEP_BINS,
                                   labels=STEP_LABELS, right=True)
            for dim, col in (("浮盈亏", "浮盈亏桶"), ("持仓步数", "持仓步数桶")):
                for b, gg in g.groupby(col, observed=True):
                    if len(gg) == 0:
                        continue
                    yt = gg["oracle_continuation_advantage_R"].to_numpy(float)
                    yp = gg["pred_adv"].to_numpy(float)
                    mm = ~np.isclose(yt, 0, atol=1e-12)
                    acc = (float(((yt > 0) == (yp > 0))[mm].mean())
                           if mm.sum() else None)
                    strat_rows.append(dict(
                        目标=t, 模型=kind, 维度=dim, 分层=str(b),
                        状态数=len(gg),
                        真实HOLD比例=round(float((yt > 0).mean()), 4),
                        模型HOLD比例=round(float((yp > 0).mean()), 4),
                        分类准确率=round(acc, 4) if acc is not None else None,
                        真实平均优势=round(float(yt.mean()), 6),
                        预测平均优势=round(float(yp.mean()), 6)))

        epi_all = pd.concat(epi_frames, ignore_index=True)
        for kind, g in test.groupby("模型"):
            replay_rows.append(dict(目标=t, 退出方式=kind,
                                    **_replay(g, epi_all)))
        base = test[test["模型"] == "LightGBM"].copy()
        base["pred_adv"] = np.inf
        replay_rows.append(dict(目标=t, 退出方式="固定退出(永远HOLD)",
                                **_replay(base, epi_all)))
        oc = test[test["模型"] == "LightGBM"].copy()
        oc["pred_adv"] = oc["oracle_continuation_advantage_R"]
        replay_rows.append(dict(目标=t, 退出方式="事后完美动态退出",
                                **_replay(oc, epi_all)))

    all_test = pd.concat(pooled, ignore_index=True)
    all_test.to_parquet(OUT / "m6_test_states.parquet", index=False)

    def dump(rows, name):
        df = pd.DataFrame(rows)
        df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
        return df

    dist_df = dump(dist_rows, "m6_target_distribution.csv")
    reg_df = dump(reg_rows, "m6_regression.csv")
    dec_df = dump(dec_rows, "m6_decile.csv")
    cls_df = dump(cls_rows, "m6_classification.csv")
    econ_df = dump(econ_rows, "m6_economic_split.csv")
    strat_df = dump(strat_rows, "m6_stratified.csv")
    rep_df = dump(replay_rows, "m6_replay.csv")

    (OUT / "m6_audit.json").write_text(json.dumps(dict(
        script="research/m6_learnability_v1.py",
        entry="LightGBM SNAPSHOT 方向 + 固定目标",
        targets=TARGETS, seed=m2.SEED, features=len(feat_cols),
        note=("直接监督 oracle_continuation_advantage_R；全步骤共享模型；"
              "事件级 4 折时间切分后展开轨迹状态。")),
        ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    print("\n=== 目标分布（整体）===")
    print(dist_df[dist_df["维度"] == "整体"].round(6).to_string(index=False))
    print("\n=== 回归能力（合并）===")
    print(reg_df[reg_df["范围"] == "合并"].round(6).to_string(index=False))
    print("\n=== 预测十分位（2.5R）===")
    print(dec_df[dec_df["目标"] == "2.5R"].round(6).to_string(index=False))
    print("\n=== 分类能力（合并）===")
    print(cls_df[cls_df["范围"] == "合并"].round(6).to_string(index=False))
    print("\n=== 经济分离 ===")
    print(econ_df.round(6).to_string(index=False))
    print("\n=== 策略回放 ===")
    print(rep_df.round(6).to_string(index=False))
    print("\nM6_DONE")


if __name__ == "__main__":
    main()
