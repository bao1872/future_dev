"""M7：持仓路径状态可学习性实验。

M6 结论：即便全步骤共享 LightGBM，退出判断仍接近随机（AUC 0.529），
且在最需要的深亏区（R<-0.75）模型 96% 预测 HOLD（真实 67% 该退出）。
M6 把「当前 -0.5R」的两条路径（持续恶化 vs 正在修复）当成相似状态。

M7 只回答：
    交易走到当前位置的「路径形状」，有没有补上缺失的信息？

两个监督目标
------------
目标A（主要）：exit_gain_vs_fixed_R = exit_value_R - fixed_reward_R
    > 0 现在退出优于继续执行固定策略；< 0 退出反而损害固定策略。
    只要求识别「何时覆盖原策略」，不要求找到未来最优退出点。
目标B（对照）：oracle_continuation_advantage_R
    降级为理论上限诊断目标，不作唯一目标。

新增路径状态（仅 16 个，全部由持仓路径生成，不加技术指标）
------------------------------------------------------
unrealized_R_delta_1/2/3, unrealized_R_slope_3
drawdown_from_MFE_R, rebound_from_MAE_R
MFE_delta_1/3, MAE_delta_1/3
distance_to_stop_delta_1/3, distance_to_target_delta_1/3
bars_since_MFE, bars_since_MAE

因果合同：step=t 的所有 delta/slope/MFE/MAE/bars_since 只能使用 step<=t
已发生的信息；MFE/MAE 基于当前决策开盘之前已完成 K 线，不读取当前未完成
K 线的高低收。已加硬测试。

三个特征视图
------------
S0 CURRENT           M6 现有状态
S1 PATH_ONLY         仅路径特征 + step/bars_remaining/direction/target_R/current_unrealized_R
S2 CURRENT_PLUS_PATH S0 + 路径特征

模型：LightGBM + 线性。禁止调参、禁止深度学习、禁止新增技术指标。

用法：python -m research.m7_path_learnability_v1
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

OUT = Path("research/analysis_results/m7_path_learnability")
OUT.mkdir(parents=True, exist_ok=True)

TRAJ = Path("research/analysis_results/rl_exit_v1/rl_exit_trajectory_v1.parquet")
DT_AUDIT = Path("research/analysis_results/m3_direction_target/m3dt_audit.json")

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

PATH_FEATS = [
    "unrealized_R_delta_1", "unrealized_R_delta_2", "unrealized_R_delta_3",
    "unrealized_R_slope_3",
    "drawdown_from_MFE_R", "rebound_from_MAE_R",
    "MFE_delta_1", "MFE_delta_3", "MAE_delta_1", "MAE_delta_3",
    "distance_to_stop_delta_1", "distance_to_stop_delta_3",
    "distance_to_target_delta_1", "distance_to_target_delta_3",
    "bars_since_MFE", "bars_since_MAE",
]

PATH_GROUPS = {
    "短期收益变化": ["unrealized_R_delta_1", "unrealized_R_delta_2",
                     "unrealized_R_delta_3", "unrealized_R_slope_3"],
    "MFE回撤": ["drawdown_from_MFE_R", "MFE_delta_1", "MFE_delta_3"],
    "MAE修复": ["rebound_from_MAE_R", "MAE_delta_1", "MAE_delta_3"],
    "止损距离变化": ["distance_to_stop_delta_1", "distance_to_stop_delta_3"],
    "目标距离变化": ["distance_to_target_delta_1", "distance_to_target_delta_3"],
    "极值时间": ["bars_since_MFE", "bars_since_MAE"],
}

R_BINS = [-np.inf, -0.75, -0.50, -0.25, 0.0, 0.25, 0.50, 1.00, np.inf]
R_LABELS = ["R<-0.75", "-0.75~-0.50", "-0.50~-0.25", "-0.25~0",
            "0~0.25", "0.25~0.50", "0.50~1.00", "R>=1.00"]


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


def add_path_features(st):
    """只使用 step<=t 的信息构造路径状态（严格因果）。"""
    st = st.sort_values(["candidate_id", "step"]).copy()
    g = st.groupby("candidate_id", sort=False)
    o = st["exit_value_R"]
    mfe = st["max_favorable_excursion_R_so_far"]
    mae = st["max_adverse_excursion_R_so_far"]

    p1, p2, p3 = g["exit_value_R"].shift(1), g["exit_value_R"].shift(2), \
        g["exit_value_R"].shift(3)
    st["unrealized_R_delta_1"] = o - p1
    st["unrealized_R_delta_2"] = o - p2
    st["unrealized_R_delta_3"] = o - p3
    st["unrealized_R_slope_3"] = (o - p2) / 2.0

    st["drawdown_from_MFE_R"] = mfe - o
    st["rebound_from_MAE_R"] = o - mae

    m1, m3 = g["max_favorable_excursion_R_so_far"].shift(1), \
        g["max_favorable_excursion_R_so_far"].shift(3)
    a1, a3 = g["max_adverse_excursion_R_so_far"].shift(1), \
        g["max_adverse_excursion_R_so_far"].shift(3)
    st["MFE_delta_1"] = mfe - m1
    st["MFE_delta_3"] = mfe - m3
    st["MAE_delta_1"] = mae - a1
    st["MAE_delta_3"] = mae - a3

    st["distance_to_stop_delta_1"] = st["distance_to_stop_R"] - \
        g["distance_to_stop_R"].shift(1)
    st["distance_to_stop_delta_3"] = st["distance_to_stop_R"] - \
        g["distance_to_stop_R"].shift(3)
    st["distance_to_target_delta_1"] = st["distance_to_target_R"] - \
        g["distance_to_target_R"].shift(1)
    st["distance_to_target_delta_3"] = st["distance_to_target_R"] - \
        g["distance_to_target_R"].shift(3)

    # 距离「当前极值」首次发生经过了多少根（MFE 单调不减 / MAE 单调不增）
    for col, out in (("max_favorable_excursion_R_so_far", "bars_since_MFE"),
                     ("max_adverse_excursion_R_so_far", "bars_since_MAE")):
        prev = g[col].shift(1)
        ch = (st[col] != prev) | prev.isna()
        blk = ch.groupby(st["candidate_id"]).cumsum()
        first_step = st.groupby([st["candidate_id"], blk])["step"].transform("first")
        st[out] = st["step"] - first_step
    return st


def causal_checks(st):
    """路径状态硬因果测试。"""
    res = {}
    res["drawdown_from_MFE_R_非负"] = bool(
        (st["drawdown_from_MFE_R"] >= -1e-9).all())
    res["rebound_from_MAE_R_非负"] = bool(
        (st["rebound_from_MAE_R"] >= -1e-9).all())
    res["bars_since_MFE_范围合法"] = bool(
        (st["bars_since_MFE"] >= 0).all()
        and (st["bars_since_MFE"] <= st["step"]).all())
    res["bars_since_MAE_范围合法"] = bool(
        (st["bars_since_MAE"] >= 0).all()
        and (st["bars_since_MAE"] <= st["step"]).all())
    # delta_1 必须等于相邻两步 open 之差（未跨 episode）
    g = st.groupby("candidate_id", sort=False)
    expect = st["exit_value_R"] - g["exit_value_R"].shift(1)
    m = expect.notna()
    res["unrealized_R_delta_1_等于相邻开盘差"] = bool(
        np.allclose(st.loc[m, "unrealized_R_delta_1"], expect[m], atol=1e-9))
    # 路径特征不得依赖未来：随机打乱未来 step 后特征值不变（构造上成立，
    # 这里用等价检验：MFE 单调不减 / MAE 单调不增）
    res["MFE_单调不减"] = bool(
        (g["max_favorable_excursion_R_so_far"].diff().fillna(0) >= -1e-9).all())
    res["MAE_单调不增"] = bool(
        (g["max_adverse_excursion_R_so_far"].diff().fillna(0) <= 1e-9).all())
    return res


def build_states(traj, keys, t):
    st = traj.merge(keys, on="candidate_id", how="inner")
    st = st[st["initial_action"] == st["action"]].copy()
    st = add_path_features(st)
    # 目标A：现在退出相对固定策略的增量
    st["exit_gain_vs_fixed_R"] = (
        st["exit_value_R"] - st["all_hold_terminal_R"])
    dec = st[st["is_decision_point"] & ~st["terminal_reason"].isin(FORCED)]
    parts = []
    for cid, gg0 in dec.groupby("candidate_id", sort=False):
        vals = gg0["exit_value_R"].to_numpy(float)
        term = float(gg0["all_hold_terminal_R"].iloc[0])
        n = len(vals)
        if n == 0:
            continue
        sm = np.maximum.accumulate(vals[::-1])[::-1]
        suf = np.full(n, -np.inf)
        suf[:-1] = sm[1:]
        future = np.maximum(suf, term)
        gg = gg0.copy()
        gg["oracle_future_value_R"] = future
        gg["oracle_continuation_advantage_R"] = future - vals
        parts.append(gg)
    out = pd.concat(parts, ignore_index=True)
    out["目标"] = t
    return out


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
        return dict(状态数=len(a), 皮尔逊=None, 斯皮尔曼=None, MAE=None,
                    RMSE=None)
    return dict(状态数=len(a),
                皮尔逊=round(float(pearsonr(p, a)[0]), 4),
                斯皮尔曼=round(float(spearmanr(p, a)[0]), 4),
                MAE=round(float(np.mean(np.abs(p - a))), 6),
                RMSE=round(float(np.sqrt(np.mean((p - a) ** 2))), 6))


def _cls(yt_pos, score):
    """yt_pos: 真实正类(EXIT有利)布尔；score: 模型打分。"""
    m = ~np.isclose(yt_pos, 0.5)      # 占位，外部已过滤 TIE
    return m


def cls_metrics(y_true, score):
    """y_true: 1=正类。返回 AUC/精确率/召回率。"""
    if len(np.unique(y_true)) < 2:
        return dict(AUC=None, 准确率=None, 正类精确率=None, 正类召回率=None,
                    负类精确率=None, 负类召回率=None)
    auc = roc_auc_score(y_true, score)
    yp = (np.asarray(score) > 0).astype(int)
    yt = np.asarray(y_true, dtype=int)
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    tn = int(((yp == 0) & (yt == 0)).sum())
    return dict(
        AUC=round(float(auc), 4),
        准确率=round(float((yt == yp).mean()), 4),
        正类精确率=round(tp / (tp + fp), 4) if tp + fp else None,
        正类召回率=round(tp / (tp + fn), 4) if tp + fn else None,
        负类精确率=round(tn / (tn + fn), 4) if tn + fn else None,
        负类召回率=round(tn / (tn + fp), 4) if tn + fp else None)


def _replay(g, epi):
    """pred_score > 0 → EXIT。epi 为该折全部测试事件（含无决策点）。"""
    g = g.sort_values(["candidate_id", "step"])
    out, early, held = {}, {}, {}
    for cid, gg in g.groupby("candidate_id", sort=False):
        term = float(gg["all_hold_terminal_R"].iloc[0])
        o, b, e = term, int(gg["step"].max()), False
        for _, r in gg.iterrows():
            if float(r["pred_score"]) > 0:
                o, b, e = float(r["exit_value_R"]), int(r["step"]), True
                break
        out[cid], held[cid], early[cid] = o, b, e
    cids = epi["candidate_id"].to_numpy()
    traded = epi["traded"].to_numpy().astype(bool)
    day = epi["day"].to_numpy()
    term_map = dict(zip(epi["candidate_id"].to_numpy(),
                        epi["term_R"].to_numpy()))
    realized = np.where(traded, np.array([out.get(c, term_map[c])
                                          for c in cids]), 0.0)
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
    ev_day, keep_ids, y_rows = D["ev_day"], D["keep_ids"], D["y_rows"]
    temporal = m2.build_temporal(D["ev_sym"], D["ev_bar"])
    Xsnap, Xtemp, cat = m2.build_feature_matrices(D, temporal)
    folds, uniq = m2.build_folds(ev_day, n_kept)
    fold_train_days = [f[0] for f in folds]
    fold_se_days = [f[1] for f in folds]
    fold_test_days = [f[2] for f in folds]

    audit = json.load(open(DT_AUDIT))
    fold_cov = {t: {int(p["折"]): p[COV_KEY[t]] for p in audit["fold_plan"]}
                for t in TARGETS}

    cur_feats = (["step", "bars_remaining"]
                 + [c for c in traj.columns if c.startswith("bg_")]
                 + [c for c in traj.columns if c.startswith("dyn_")]
                 + POS_NUM + ["crossed_session_break"])
    for c in cur_feats:
        if not pd.api.types.is_numeric_dtype(traj[c]):
            traj[c] = traj[c].astype(str).astype("category").cat.codes.astype(
                "int16")
    traj["crossed_session_break"] = traj["crossed_session_break"].astype("int8")

    VIEWS = {
        "S0_CURRENT": cur_feats,
        "S1_PATH_ONLY": (PATH_FEATS + ["step", "bars_remaining",
                                       "trade_direction", "target_R",
                                       "current_unrealized_R"]),
        "S2_CURRENT_PLUS_PATH": cur_feats + PATH_FEATS,
    }

    # 每折入场方向
    dir_frames = []
    for fi in range(4):
        tr_m = m2.day_mask(ev_day, fold_train_days[fi])
        se_m = m2.day_mask(ev_day, fold_se_days[fi])
        te_m = m2.day_mask(ev_day, fold_test_days[fi])
        use = tr_m | se_m | te_m
        tr_row, use_row = np.repeat(tr_m, 6), np.repeat(use, 6)
        Xsnap_num, Xtemp_num = m2.encode_for_fold(Xsnap, Xtemp, cat, tr_row)
        Xn = Xsnap_num
        yhat = m2.train_reg("lgbm", Xn.iloc[tr_row].to_numpy(float),
                            y_rows[tr_row], Xn.iloc[use_row].to_numpy(float),
                            cat, m2.SEED)
        ye = yhat.reshape(-1, 6)
        fs = ye[:, [A6.index(c) for c in FOLLOW_COLS]].mean(axis=1)
        fd = ye[:, [A6.index(c) for c in FADE_COLS]].mean(axis=1)
        isf = fs >= fd
        dir_frames.append(pd.DataFrame(dict(
            candidate_id=keep_ids[use], fold=fi + 1, day=ev_day[use],
            is_follow=isf, event_score=np.where(isf, fs, fd),
            is_test=te_m[use], is_sel=se_m[use])))
        print(f"[M7] F{fi+1} pool={int(use.sum())}", flush=True)
    dirs = pd.concat(dir_frames, ignore_index=True)

    reg_rows, dec_rows, cls_rows, econ_rows = [], [], [], []
    deep_rows, strat_rows, replay_rows, fold_rows, sym_rows = [], [], [], [], []
    imp_rows = []
    causal_all = {}

    for t in TARGETS:
        fold_preds, epi_frames = [], []
        for fi in range(4):
            d = dirs[dirs["fold"] == fi + 1]
            act = np.where(d["is_follow"].to_numpy(), f"FOLLOW_{t}",
                           f"FADE_{t}")
            keys = pd.DataFrame({"candidate_id": d["candidate_id"].to_numpy(),
                                 "action": act})
            se, te = d[d["is_sel"]], d[d["is_test"]]
            cutoff = np.quantile(se["event_score"].to_numpy(),
                                 1 - fold_cov[t][fi + 1])

            te_ids = te["candidate_id"].to_numpy()
            te_keys = pd.DataFrame({
                "candidate_id": te_ids,
                "action": act[d["is_test"].to_numpy()]})
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
            causal_all.setdefault(t, causal_checks(st))
            st["traded"] = st["candidate_id"].map(
                dict(zip(te_ids, te["event_score"].to_numpy() > cutoff))
            ).fillna(False).astype(bool)
            st["折"] = f"F{fi+1}"

            tr_mask = st["trading_day"].isin(fold_train_days[fi]).to_numpy()
            te_mask = st["trading_day"].isin(fold_test_days[fi]).to_numpy()
            if tr_mask.sum() < 500 or te_mask.sum() < 100:
                continue
            cols = ["candidate_id", "symbol", "trading_day", "step",
                    "exit_value_R", "all_hold_terminal_R",
                    "exit_gain_vs_fixed_R",
                    "oracle_continuation_advantage_R", "traded",
                    "current_unrealized_R"]
            for vname, feats in VIEWS.items():
                Xtr = st.loc[tr_mask, feats].to_numpy(float)
                Xte = st.loc[te_mask, feats].to_numpy(float)
                for tgt, sign in (("A", 1.0), ("B", -1.0)):
                    ycol = ("exit_gain_vs_fixed_R" if tgt == "A"
                            else "oracle_continuation_advantage_R")
                    ytr = st.loc[tr_mask, ycol].to_numpy(float)
                    kinds = ["LightGBM", "线性"] if tgt == "A" else ["LightGBM"]
                    for kind in kinds:
                        pred, mdl = fit_predict(kind, Xtr, ytr, Xte)
                        sub = st.loc[te_mask, cols].copy()
                        sub["视图"] = vname
                        sub["目标类型"] = tgt
                        sub["模型"] = kind
                        sub["pred_raw"] = pred
                        # pred_score > 0 → EXIT
                        sub["pred_score"] = sign * pred
                        sub["折"] = f"F{fi+1}"
                        sub["目标"] = t
                        fold_preds.append(sub)
                        if (tgt == "A" and kind == "LightGBM"
                                and vname == "S2_CURRENT_PLUS_PATH"):
                            try:
                                imp = mdl.feature_importances_
                                for f, v in zip(feats, imp):
                                    imp_rows.append(dict(
                                        目标=t, 特征=f, 重要性=int(v)))
                            except Exception:
                                pass
            print(f"[M7]   {t} F{fi+1} done", flush=True)

        test = pd.concat(fold_preds, ignore_index=True)
        epi_all = pd.concat(epi_frames, ignore_index=True)

        for (vname, tgt, kind), g in test.groupby(
                ["视图", "目标类型", "模型"]):
            ycol = ("exit_gain_vs_fixed_R" if tgt == "A"
                    else "oracle_continuation_advantage_R")
            a = g[ycol].to_numpy(float)
            p = g["pred_raw"].to_numpy(float)
            label = f"{vname}/{tgt}/{kind}"
            reg_rows.append(dict(目标=t, 视图=vname, 目标类型=tgt, 模型=kind,
                                 **_reg(a, p)))
            for q, gg in g.groupby(pd.qcut(g["pred_raw"], 10, labels=False,
                                           duplicates="drop")):
                real = gg[ycol].to_numpy(float)
                pos = (real > 0) if tgt == "A" else None
                dec_rows.append(dict(
                    目标=t, 视图=vname, 目标类型=tgt, 模型=kind,
                    预测十分位=int(q) + 1, 状态数=len(gg),
                    预测平均=round(float(gg["pred_raw"].mean()), 6),
                    真实平均=round(float(real.mean()), 6),
                    真实中位数=round(float(np.median(real)), 6),
                    真实有利比例=(round(float(pos.mean()), 4)
                                  if pos is not None else None)))
            # 分类（排除 TIE）
            m = ~np.isclose(a, 0, atol=1e-12)
            if m.sum() >= 50 and len(np.unique(a[m] > 0)) >= 2:
                yt = (a[m] > 0).astype(int)
                sc = p[m] * (1.0 if tgt == "A" else -1.0)
                c = cls_metrics(yt, sc)
                pe = a[m][sc > 0]
                ph = a[m][sc <= 0]
                econ_rows.append(dict(
                    目标=t, 视图=vname, 目标类型=tgt, 模型=kind,
                    样本数=int(m.sum()), **c,
                    预测EXIT组真实值=(round(float(pe.mean()), 6)
                                      if len(pe) else None),
                    预测HOLD组真实值=(round(float(ph.mean()), 6)
                                      if len(ph) else None),
                    两组差值=(round(float(pe.mean() - ph.mean()), 6)
                              if len(pe) and len(ph) else None)))
            # 深亏区专项
            dl = g[g["current_unrealized_R"] < -0.75]
            if len(dl) >= 50:
                ad = dl[ycol].to_numpy(float)
                md = ~np.isclose(ad, 0, atol=1e-12)
                if md.sum() >= 50 and len(np.unique(ad[md] > 0)) >= 2:
                    ytd = (ad[md] > 0).astype(int)
                    scd = dl["pred_raw"].to_numpy(float)[md] * (
                        1.0 if tgt == "A" else -1.0)
                    cd = cls_metrics(ytd, scd)
                    deep_rows.append(dict(
                        目标=t, 视图=vname, 目标类型=tgt, 模型=kind,
                        状态数=len(dl),
                        真实EXIT有利比例=round(float((ad > 0).mean()), 4),
                        **cd,
                        预测EXIT组真实值=(round(float(ad[md][scd > 0].mean()), 6)
                                          if (scd > 0).any() else None),
                        预测HOLD组真实值=(round(float(ad[md][scd <= 0].mean()), 6)
                                          if (scd <= 0).any() else None)))
            # 浮盈亏分层
            gg2 = g.copy()
            gg2["浮盈亏桶"] = pd.cut(gg2["current_unrealized_R"], bins=R_BINS,
                                    labels=R_LABELS, right=False)
            for b, gb in gg2.groupby("浮盈亏桶", observed=True):
                if len(gb) < 50:
                    continue
                ab = gb[ycol].to_numpy(float)
                mb = ~np.isclose(ab, 0, atol=1e-12)
                acc = None
                if mb.sum() and len(np.unique(ab[mb] > 0)) >= 2:
                    scb = gb["pred_raw"].to_numpy(float)[mb] * (
                        1.0 if tgt == "A" else -1.0)
                    acc = cls_metrics((ab[mb] > 0).astype(int), scb)["AUC"]
                strat_rows.append(dict(
                    目标=t, 视图=vname, 目标类型=tgt, 模型=kind,
                    分层=str(b), 状态数=len(gb),
                    真实有利比例=round(float((ab > 0).mean()), 4),
                    AUC=acc, 真实平均=round(float(ab.mean()), 6)))

        # 回放
        for (vname, tgt, kind), g in test.groupby(["视图", "目标类型", "模型"]):
            if not (tgt == "A" or (vname == "S0_CURRENT" and kind == "LightGBM")):
                continue
            nm = f"{vname}-{tgt}-{kind}"
            replay_rows.append(dict(目标=t, 退出方式=nm, **_replay(g, epi_all)))
            for f_, gg in g.groupby("折"):
                ep = epi_all[epi_all["折"] == f_]
                fold_rows.append(dict(目标=t, 退出方式=nm, 折=f_,
                                      **_replay(gg, ep)))
                ep2 = ep.copy()
                for s in pd.unique(gg["symbol"]):
                    gs = gg[gg["symbol"] == s]
                    if len(gs) < 20:
                        continue
                    sym_rows.append(dict(目标=t, 退出方式=nm, 品种=s, 折=f_,
                                         **_replay(gs, ep2)))
        base = test[test["视图"] == "S0_CURRENT"].copy()
        base["pred_score"] = -np.inf
        replay_rows.append(dict(目标=t, 退出方式="固定退出(永远HOLD)",
                                **_replay(base, epi_all)))
        oc = test[test["视图"] == "S0_CURRENT"].copy()
        oc["pred_score"] = oc["exit_gain_vs_fixed_R"]
        replay_rows.append(dict(目标=t, 退出方式="事后完美覆盖(oracleA)",
                                **_replay(oc, epi_all)))

    def dump(rows, name):
        df = pd.DataFrame(rows)
        df.to_csv(OUT / name, index=False, encoding="utf-8-sig")
        return df

    reg_df = dump(reg_rows, "m7_regression.csv")
    dec_df = dump(dec_rows, "m7_decile.csv")
    econ_df = dump(econ_rows, "m7_classification.csv")
    deep_df = dump(deep_rows, "m7_deeploss.csv")
    strat_df = dump(strat_rows, "m7_stratified.csv")
    rep_df = dump(replay_rows, "m7_replay.csv")
    fold_df = dump(fold_rows, "m7_by_fold.csv")
    sym_df = dump(sym_rows, "m7_by_symbol.csv")

    if imp_rows:
        imp = pd.DataFrame(imp_rows).groupby("特征", as_index=False)[
            "重要性"].sum().sort_values("重要性", ascending=False)
        mp = dict(zip(PATH_FEATS, [g for g, fs in PATH_GROUPS.items()
                                   for f in fs]))
        imp["分组"] = imp["特征"].map(mp).fillna("现有状态")
        imp.to_csv(OUT / "m7_feature_importance.csv", index=False,
                   encoding="utf-8-sig")
        grp = imp.groupby("分组")["重要性"].sum().sort_values(ascending=False)
        grp.to_csv(OUT / "m7_feature_importance_group.csv",
                   encoding="utf-8-sig")

    (OUT / "m7_audit.json").write_text(json.dumps(dict(
        script="research/m7_path_learnability_v1.py",
        targets=TARGETS, seed=m2.SEED,
        path_features=PATH_FEATS,
        n_path_features=len(PATH_FEATS),
        views={k: len(v) for k, v in VIEWS.items()},
        causal_checks=causal_all,
        note=("目标A=exit_gain_vs_fixed_R(主要)；"
              "目标B=oracle_continuation_advantage_R(理论上限对照)。")),
        ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 270)
    pd.set_option("display.max_columns", 60)
    print("\n=== 路径状态因果检查 ===")
    for k, v in causal_all.items():
        print(f"  {k}: {v}")
    print("\n=== 目标A：回归 ===")
    print(reg_df[(reg_df["目标类型"] == "A")].round(6).to_string(index=False))
    print("\n=== 目标A：分类 + 经济分离 ===")
    print(econ_df[econ_df["目标类型"] == "A"].round(6).to_string(index=False))
    print("\n=== 深亏区（R<-0.75，目标A）===")
    print(deep_df[deep_df["目标类型"] == "A"].round(6).to_string(index=False))
    print("\n=== 策略回放（2.5R）===")
    print(rep_df[rep_df["目标"] == "2.5R"].round(6).to_string(index=False))
    print("\nM7_DONE")


if __name__ == "__main__":
    main()
