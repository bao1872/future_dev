#!/usr/bin/env python3

"""62维 RL V1 -- 稳健性验证。

目标：判断当前 CatBoost 的正收益是否
  1) 跨时间重复出现（多窗口滚动）；
  2) 显著超过随机模型选择结果（随机置换检验）；
  3) 依赖重复事件（重叠敏感性）。

本轮不调模型、不增加特征、不调整盈利目标、不搜索新覆盖率。
聚焦当前最佳模型 CatBoost（参数冻结）。

主收益曲线口径：每日机会集等权标准化收益。
"""

from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from matplotlib import pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Heiti TC", "Songti SC", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

from research.ob_rl_model_view_v0_spec import MODEL_FEATURES_V0
from research.rl_62d_core_v1 import ACTION_NAMES, NO_TRADE_INDEX
from research.train_rl_62d_v1 import (
    ROLL_GAP_ATR_THRESHOLD,
    curve_metrics,
    daily_curve,
)

RESULTS = Path("research/analysis_results/rl_62d_v1")
ACTION_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/ob_rl_action_v0.parquet"
)

SEED = 20240901
N_PERMUTATIONS = 200
COV = [0.05, 0.10, 0.20, 0.30, 0.50]
THREADS = 1

CATEGORICAL_COLUMNS = (
    "source_tf", "source_ob_structure", "touch_behavior",
    "quant_state", "trade_mode",
    "forward_active_ob_structure_class_5m",
    "forward_active_ob_structure_class_15m",
    "forward_active_ob_structure_class_1h",
    "backward_active_ob_structure_class_5m",
    "backward_active_ob_structure_class_15m",
    "backward_active_ob_structure_class_1h",
    "momentum_direction_5m", "momentum_direction_15m", "momentum_direction_1h",
)


def git_head():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------
# 数据准备
# ------------------------------------------------------------

def build_raw_rows(order_ids):
    action = pd.read_parquet(ACTION_PARQUET)
    sub = action[
        action["candidate_id"].isin(order_ids)
        & (action["action"] != "SKIP")
    ]
    ev = pd.DataFrame(
        {"candidate_id": order_ids, "_row": np.arange(len(order_ids))}
    )
    sub = (
        ev.merge(sub, on="candidate_id", how="left")
        .sort_values(["_row", "action"], kind="mergesort")
        .reset_index(drop=True)
    )
    return sub[list(MODEL_FEATURES_V0)].reset_index(drop=True)


def prepare_catboost(rows):
    x = rows.copy()
    cat = [c for c in CATEGORICAL_COLUMNS if c in x.columns]
    for c in cat:
        x[c] = (
            x[c].astype("object")
            .where(x[c].notna(), "__MISSING__").astype(str)
        )
    for c in x.columns:
        if c not in cat:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    return x, cat


def load_data():
    idx = pd.read_csv(RESULTS / "event_index_v2.csv")
    rewards = np.load(RESULTS / "reward_matrix_v2.npy")
    gaps = np.load(RESULTS / "max_gap_atr_v2.npy")
    Xenc = np.load(RESULTS / "features_v2.npy")

    order = idx["candidate_id"].to_numpy()
    raw = build_raw_rows(order)

    keep = (
        np.nan_to_num(gaps[:, 1:], nan=0.0).max(axis=1)
        <= ROLL_GAP_ATR_THRESHOLD
    )
    ev_of_row = np.repeat(np.arange(len(order)), 6)
    row_keep = keep[ev_of_row]

    Xenc_k = Xenc[row_keep]
    raw_k = raw.loc[row_keep].reset_index(drop=True)
    y_k = rewards[ev_of_row, 1 + (np.arange(len(ev_of_row)) % 6)][row_keep]
    split_k = idx["split"].to_numpy()[ev_of_row][row_keep]
    day_k = idx["trading_day"].to_numpy()[keep].astype(str)
    sym_k = idx["symbol"].to_numpy()[keep]
    rew_k = rewards[keep]

    n_kept = len(day_k)
    # 行级掩码（覆盖事件×6 行）与事件级掩码（每事件一个）两套并存
    split_ev = split_k.reshape(n_kept, 6)[:, 0]
    real_train_ev = split_ev == "TRAIN"
    real_sel_ev = split_ev == "SELECTION"
    real_test_ev = split_ev == "TEST"

    n_kept = len(day_k)
    ek = ev_of_row[row_keep]
    act_k = (np.arange(len(ev_of_row)) % 6)[row_keep]
    keep_idx = np.where(keep)[0]
    remap = -np.ones(len(order), dtype=np.int64)
    remap[keep_idx] = np.arange(n_kept)
    ek2 = remap[ek]

    Xcb, cat_cols = prepare_catboost(raw_k)
    cat_idx = [Xcb.columns.get_loc(c) for c in cat_cols]

    # 重叠敏感性辅助
    st = pd.read_parquet(
        "research/analysis_results/ob_rl_dataset_v0/ob_rl_state_v0.parquet"
    )[["candidate_id", "candidate_group_id"]]
    grp = st.set_index("candidate_id")["candidate_group_id"].reindex(order)
    grp_k = grp.to_numpy()[keep]
    act = pd.read_parquet(ACTION_PARQUET)[
        ["candidate_id", "decision_weight"]
    ].drop_duplicates("candidate_id").set_index("candidate_id")[
        "decision_weight"
    ].reindex(order)
    dw_k = act.to_numpy()[keep]

    return dict(
        rewards=rew_k, day_k=day_k, sym_k=sym_k, split_k=split_k,
        real_train_ev=real_train_ev, real_sel_ev=real_sel_ev,
        real_test_ev=real_test_ev,
        n_kept=n_kept, ek2=ek2, act_k=act_k, Xcb=Xcb, cat_idx=cat_idx,
        grp_k=grp_k, dw_k=dw_k,
    )


# ------------------------------------------------------------
# CatBoost 训练与评分
# ------------------------------------------------------------

def fit_catboost(Xtr, ytr, Xse, yse, cat_idx):
    cb = CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=6, l2_leaf_reg=5.0,
        loss_function="RMSE", random_seed=SEED,
        allow_writing_files=False, verbose=False,
        thread_count=THREADS,
    )
    cb.fit(
        Xtr, ytr, cat_features=cat_idx, eval_set=(Xse, yse),
        early_stopping_rounds=100, use_best_model=True,
    )
    return cb


def train_s6(rew, Xcb, cat_idx, ek2, act_k, n_kept, train_row, sel_row):
    """train_row / sel_row 为行级掩码（覆盖事件×6 行）。"""
    y = rew[:, 1:][ek2, act_k]
    train_rows = np.where(train_row)[0]
    sel_rows = np.where(sel_row)[0]
    cb = fit_catboost(
        Xcb.iloc[train_rows], y[train_rows],
        Xcb.iloc[sel_rows], y[sel_rows], cat_idx,
    )
    p = cb.predict(Xcb)
    s6 = np.full((n_kept, 6), -np.inf)
    s6[ek2, act_k] = p
    return s6


def _profit_factor(r):
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    win = r[r > 0].sum()
    loss = -r[r < 0].sum()
    if loss <= 0:
        return float("inf") if win > 0 else 0.0
    return float(win / loss)


def pipeline(s6, rew, day_k, sel_ev, test_ev):
    """给定每事件六动作评分，在 sel 选覆盖率，在 test 出指标。"""
    mx_sel = s6[sel_ev].max(axis=1)
    cuts = {c: float(np.quantile(mx_sel, 1 - c)) for c in COV}

    best = None
    for c in COV:
        msel = sel_ev & (s6.max(axis=1) >= cuts[c])
        r = np.zeros(sel_ev.sum())
        pos = np.where(msel[sel_ev])[0]
        sa = np.argmax(s6[msel], axis=1)
        got = rew[msel, 1:][np.arange(msel.sum()), sa]
        r[pos] = got
        daily = daily_curve(r, day_k[sel_ev])
        sh = curve_metrics(daily).get("夏普率", -99)
        sh = sh if (sh == sh) else -99
        key = (round(sh, 4), int(msel.sum()))
        if best is None or key > best[0]:
            best = (key, c)
    best_c = best[1]

    m = test_ev & (s6.max(axis=1) >= cuts[best_c])
    sa = np.argmax(s6[m], axis=1)
    got = rew[m, 1:][np.arange(m.sum()), sa]
    r_test = np.zeros(test_ev.sum())
    pos = np.where(m[test_ev])[0]
    r_test[pos] = got
    daily = daily_curve(r_test, day_k[test_ev])
    met = curve_metrics(daily)
    pf = _profit_factor(got)
    met.update({
        "交易次数": int(m.sum()),
        "不交易比例": round(1 - m.sum() / test_ev.sum(), 4),
        "选中平均收益": round(float(got.mean()), 6) if len(got) else None,
        "最佳覆盖率": best_c,
        "利润因子": round(pf, 4) if np.isfinite(pf) else (1e9 if pf > 0 else 0.0),
    })
    return daily, met, best_c, cuts


# ------------------------------------------------------------
# 全局（供进程池）
# ------------------------------------------------------------

G = {}


def _init(Xcb, cat_idx, ek2, act_k, n_kept, day_k, real_train_row,
          real_sel_row, real_sel_ev, real_test_ev, groups, real_rew):
    global G
    G = dict(
        Xcb=Xcb, cat_idx=cat_idx, ek2=ek2, act_k=act_k, n_kept=n_kept,
        day_k=day_k, real_train_row=real_train_row,
        real_sel_row=real_sel_row, real_sel_ev=real_sel_ev,
        real_test_ev=real_test_ev, groups=groups, real_rew=real_rew,
    )


def _perm_task(seed_i):
    rng = np.random.default_rng(seed_i)
    real_rew = G["real_rew"]
    groups = G["groups"]
    out = real_rew.copy()
    for g in groups:
        perm = rng.permutation(len(g))
        out[g, 1:] = real_rew[g[perm], 1:]
    s6 = train_s6(
        out, G["Xcb"], G["cat_idx"], G["ek2"], G["act_k"], G["n_kept"],
        G["real_train_row"], G["real_sel_row"],
    )
    _, met, bc, _ = pipeline(
        s6, out, G["day_k"], G["real_sel_ev"], G["real_test_ev"]
    )
    return (
        met["夏普率"], float(met.get("累计收益", 0)),
        met["利润因子"], bc, met["交易次数"],
        met["选中平均收益"],
    )


# ------------------------------------------------------------

def max_losing_streak(returns):
    if len(returns) == 0:
        return 0
    best = cur = 0
    for x in returns:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    D = load_data()
    rew_k = D["rewards"]
    day_k = D["day_k"]
    sym_k = D["sym_k"]
    n_kept = D["n_kept"]
    ek2, act_k = D["ek2"], D["act_k"]
    Xcb, cat_idx = D["Xcb"], D["cat_idx"]

    real_train_ev = D["real_train_ev"]
    real_sel_ev = D["real_sel_ev"]
    real_test_ev = D["real_test_ev"]
    real_train_row = np.repeat(real_train_ev, 6)
    real_sel_row = np.repeat(real_sel_ev, 6)

    # ---------- 真实基准（60/20/20） ----------
    s6_real = train_s6(
        rew_k, Xcb, cat_idx, ek2, act_k, n_kept,
        real_train_row, real_sel_row
    )
    daily_real, met_real, bc_real, cuts_real = pipeline(
        s6_real, rew_k, day_k, real_sel_ev, real_test_ev
    )
    met_real["策略"] = "CatBoost_分位策略(真实60/20/20)"

    # 最长连亏（按测试段事件时间序）
    m_real = real_test_ev & (s6_real.max(axis=1) >= cuts_real[bc_real])
    sa = np.argmax(s6_real[m_real], axis=1)
    got = rew_k[m_real, 1:][np.arange(m_real.sum()), sa]
    order_t = np.argsort(day_k[m_real])
    streak = max_losing_streak(got[order_t])

    # ---------- 实验 A：滚动窗口 ----------
    uniq = np.array(sorted(set(day_k)))
    parts = np.array_split(uniq, 5)
    wf_dailies = []
    wf_rows = []
    for w in [1, 2, 3]:
        tr_days = list(np.concatenate(parts[:w]).tolist())
        se_days = list(parts[w].tolist())
        te_days = list(parts[w + 1].tolist())
        tr_ev = np.isin(day_k, tr_days)
        se_ev = np.isin(day_k, se_days)
        te_ev = np.isin(day_k, te_days)
        s6 = train_s6(
            rew_k, Xcb, cat_idx, ek2, act_k, n_kept,
            np.repeat(tr_ev, 6), np.repeat(se_ev, 6),
        )
        daily, met, bc, _ = pipeline(s6, rew_k, day_k, se_ev, te_ev)
        wf_dailies.append(daily)
        wf_rows.append({
            "窗口": w,
            "训练起": str(pd.to_datetime(tr_days).min().date()),
            "训练止": str(pd.to_datetime(tr_days).max().date()),
            "选择起": str(pd.to_datetime(se_days).min().date()),
            "选择止": str(pd.to_datetime(se_days).max().date()),
            "测试起": str(pd.to_datetime(te_days).min().date()),
            "测试止": str(pd.to_datetime(te_days).max().date()),
            "训练事件数": int(tr_ev.sum()),
            "选择事件数": int(se_ev.sum()),
            "测试事件数": int(te_ev.sum()),
            "最佳覆盖率": bc,
            "累计收益": round(float(daily.sum()), 4),
            "夏普率": met["夏普率"],
            "最大回撤": met["最大回撤"],
            "利润因子": met["利润因子"],
            "交易次数": met["交易次数"],
            "选中平均收益": met["选中平均收益"],
        })
    combined = pd.concat(wf_dailies).sort_index()
    wf_total = {
        "滚动测试总累计收益": round(float(combined.sum()), 4),
        "滚动测试夏普率": curve_metrics(combined)["夏普率"],
        "滚动测试最大回撤": curve_metrics(combined)["最大回撤"],
        "滚动测试利润因子": _profit_factor(combined.to_numpy()),
        "正收益窗口比例": round(
            float(np.mean([1.0 for d in wf_dailies if d.sum() > 0])), 4
        ),
        "正收益交易日比例": round(
            float((combined > 0).mean()), 4
        ),
        "总交易日": int(len(combined)),
    }
    pd.DataFrame(wf_rows).to_csv(
        RESULTS / "walk_forward_windows.csv",
        index=False, encoding="utf-8-sig",
    )
    combined.to_frame("每日收益").to_csv(
        RESULTS / "walk_forward_oos_curve.csv",
        encoding="utf-8-sig",
    )

    # ---------- 实验 B：随机置换 ----------
    month = pd.to_datetime(day_k).to_period("M").astype(str)
    sym = sym_k
    df = pd.DataFrame({"s": sym, "m": month})
    grp_list = []
    for _, idxs in df.groupby(["s", "m"]).groups.items():
        grp_list.append(np.array(list(idxs)))
    groups_arr = grp_list

    seeds = [SEED + i for i in range(N_PERMUTATIONS)]
    perm_cache = RESULTS / "permutation_results.csv"
    if perm_cache.exists():
        _cached = pd.read_csv(perm_cache)
        if len(_cached) == N_PERMUTATIONS and list(_cached.columns) == [
            "夏普率", "累计收益", "利润因子", "最佳覆盖率",
            "交易次数", "选中平均收益",
        ]:
            perm = _cached
            print("PERM_CACHE_HIT")
        else:
            perm = None
    else:
        perm = None
    if perm is None:
        with ProcessPoolExecutor(
            max_workers=max(1, (os.cpu_count() or 4) // 2),
            initializer=_init,
            initargs=(
                Xcb, cat_idx, ek2, act_k, n_kept, day_k,
                real_train_row, real_sel_row, real_sel_ev, real_test_ev,
                groups_arr, rew_k,
            ),
        ) as ex:
            res = list(ex.map(_perm_task, seeds))
        perm = pd.DataFrame(
            res, columns=[
                "夏普率", "累计收益", "利润因子", "最佳覆盖率",
                "交易次数", "选中平均收益",
            ]
        )
        perm.to_csv(perm_cache, index=False, encoding="utf-8-sig")
    else:
        perm.to_csv(perm_cache, index=False, encoding="utf-8-sig")

    real_sharpe = met_real["夏普率"]
    real_cum = float(daily_real.sum())
    real_pf = met_real["利润因子"]
    pct_sh = float((perm["夏普率"] >= real_sharpe).mean())
    pct_cum = float((perm["累计收益"] >= real_cum).mean())
    perm_summary = {
        "随机夏普均值": round(float(perm["夏普率"].mean()), 6),
        "随机夏普中位数": round(float(perm["夏普率"].median()), 6),
        "随机夏普95分位": round(float(np.percentile(perm["夏普率"], 95)), 6),
        "随机夏普99分位": round(float(np.percentile(perm["夏普率"], 99)), 6),
        "随机夏普最大值": round(float(perm["夏普率"].max()), 6),
        "真实夏普率": round(real_sharpe, 6),
        "真实夏普率经验p值(越大越异常)": round(pct_sh, 6),
        "真实累计收益": round(real_cum, 6),
        "真实累计经验p值": round(pct_cum, 6),
        "置换次数": N_PERMUTATIONS,
    }

    # ---------- 实验 C：重叠敏感性 ----------
    # 原始：事件等权（即 met_real 本身）
    # 加权：按 decision_weight 加权每日收益
    # 去重：每 candidate_group_id 仅保留一个事件
    def strategy_daily(m_ev, weights=None):
        m = real_test_ev & (s6_real.max(axis=1) >= cuts_real[bc_real])
        sel = m & m_ev
        sa = np.argmax(s6_real[sel], axis=1)
        got = rew_k[sel, 1:][np.arange(sel.sum()), sa]
        tday = day_k[sel]
        if weights is None:
            dfp = pd.DataFrame({"g": tday, "r": got})
            return dfp.groupby("g")["r"].mean()
        dfp = pd.DataFrame({"g": tday, "r": got, "w": weights[sel]})
        return dfp.groupby("g").apply(
            lambda x: np.average(x["r"], weights=x["w"])
        )

    orig = daily_real
    weighted = strategy_daily(real_test_ev, weights=D["dw_k"])
    # 去重
    gp = D["grp_k"]
    seen = set()
    keep_dedup = np.zeros(n_kept, dtype=bool)
    for i in range(n_kept):
        g = gp[i]
        if g not in seen:
            seen.add(g)
            keep_dedup[i] = True
    dedup = strategy_daily(keep_dedup)

    def agg(name, series, n_ev):
        s = series.dropna()
        cm = curve_metrics(s)
        return {
            "版本": name,
            "累计收益": round(float(s.sum()), 4),
            "夏普率": cm["夏普率"],
            "最大回撤": cm["最大回撤"],
            "利润因子": _profit_factor(s.to_numpy()),
            "有效事件数": int(n_ev),
        }

    overlap = pd.DataFrame([
        agg("原始事件等权", orig, int(real_test_ev.sum())),
        agg("decision_weight加权", weighted, int(real_test_ev.sum())),
        agg("去重(每订单块组1个)", dedup, int(keep_dedup.sum())),
    ])
    overlap.to_csv(
        RESULTS / "overlap_sensitivity.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---------- 图 ----------
    fig, ax = plt.subplots(figsize=(11, 6))
    cum = np.cumsum(combined.to_numpy())
    ax.plot(range(len(cum)), cum, label="滚动OOS累计")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("多窗口滚动历史测试累计收益（每日机会集等权）")
    ax.set_xlabel("交易日序号")
    ax.set_ylabel("累计 R")
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_walk_forward_oos.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(perm["夏普率"], bins=40, color="steelblue")
    ax.axvline(real_sharpe, color="red", lw=2, label=f"真实 {real_sharpe:.2f}")
    ax.set_title("随机置换收益分布 vs 真实结果")
    ax.set_xlabel("测试段夏普率")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_permutation_null.png", dpi=110)
    plt.close(fig)

    audit = {
        "git_head": git_head(),
        "seed": SEED,
        "n_permutations": N_PERMUTATIONS,
        "covariates_checked": COV,
        "real_reference": {
            "策略": met_real["策略"], "累计收益": round(real_cum, 6),
            "夏普率": round(real_sharpe, 6),
            "最大回撤": met_real["最大回撤"],
            "利润因子": real_pf,
            "交易次数": met_real["交易次数"],
            "最长连续亏损(笔)": streak,
            "选中平均收益": met_real["选中平均收益"],
        },
        "walk_forward_total": wf_total,
        "permutation_summary": perm_summary,
        "overlap_sensitivity": overlap.to_dict(orient="records"),
    }
    (RESULTS / "robustness_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print("RL_62D_ROBUSTNESS_DONE")
    print("real sharpe", round(real_sharpe, 4), "perm p(sharpe)=", round(pct_sh, 4))
    print("walk_forward_total", wf_total)
    print("overlap", overlap.to_string(index=False))


if __name__ == "__main__":
    main()
