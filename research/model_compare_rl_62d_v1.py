#!/usr/bin/env python3

"""62维 RL V1 -- 模型对比与排序能力诊断。

目标不是调神经网络，而是回答：
    1. 62维能不能排序好坏交易？
    2. 62维能不能在同一事件里选对六种动作的相对优劣？
    3. 这种能力能不能转化成正收益曲线？

三模型：岭回归 / CatBoost / 神经网络（MSE）。
同一份事件、同一份特征、同一份标签、同一份时间切分。

裁决全部在模型选择段完成；历史测试段只做冻结应用。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS", "Heiti TC", "Songti SC", "DejaVu Sans"
]
plt.rcParams["axes.unicode_minus"] = False

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor
from scipy.stats import kendalltau, spearmanr
from sklearn.linear_model import Ridge
from torch import nn

from research.ob_rl_model_view_v0_spec import MODEL_FEATURES_V0
from research.rl_62d_core_v1 import (
    ACTION_NAMES,
    DATASET_SKIP_ACTION,
    NO_TRADE_INDEX,
    QNetwork,
    select_actions,
)
from research.train_rl_62d_v1 import (
    ROLL_GAP_ATR_THRESHOLD,
    curve_metrics,
    daily_curve,
    trade_metrics,
)

RESULTS = Path("research/analysis_results/rl_62d_v1")
ACTION_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/"
    "ob_rl_action_v0.parquet"
)

SEED = 20240901
RIDGE_ALPHAS = [0.1, 1.0, 10.0]
COVERAGES = [0.05, 0.10, 0.20, 0.30, 0.50]

# 历史实现：explore_ob_q_relationship_v1.py::prepare_X
CATEGORICAL_COLUMNS = (
    "source_tf",
    "source_ob_structure",
    "touch_behavior",
    "quant_state",
    "trade_mode",
    "forward_active_ob_structure_class_5m",
    "forward_active_ob_structure_class_15m",
    "forward_active_ob_structure_class_1h",
    "backward_active_ob_structure_class_5m",
    "backward_active_ob_structure_class_15m",
    "backward_active_ob_structure_class_1h",
    "momentum_direction_5m",
    "momentum_direction_15m",
    "momentum_direction_1h",
)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def build_raw_rows(order_ids):
    """按事件顺序取 [事件x6, 62] 的原始特征（含类别原值）。"""

    action = pd.read_parquet(ACTION_PARQUET)
    sub = action[
        action["candidate_id"].isin(order_ids)
        & (action["action"] != DATASET_SKIP_ACTION)
    ]
    ev = pd.DataFrame(
        {"candidate_id": order_ids, "_row": np.arange(len(order_ids))}
    )
    sub = (
        ev.merge(sub, on="candidate_id", how="left")
        .sort_values(["_row", "action"], kind="mergesort")
        .reset_index(drop=True)
    )
    assert np.array_equal(
        sub["candidate_id"].to_numpy(), np.repeat(order_ids, 6)
    )
    return sub[list(MODEL_FEATURES_V0)].reset_index(drop=True)


def prepare_catboost(rows):
    """复用历史权威 prepare_X 的类别处理方式。"""

    x = rows.copy()
    cat = [c for c in CATEGORICAL_COLUMNS if c in x.columns]
    for c in cat:
        x[c] = (
            x[c]
            .astype("object")
            .where(x[c].notna(), "__MISSING__")
            .astype(str)
        )
    for c in x.columns:
        if c in cat:
            continue
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x, cat


def diag_table(pred, real):
    from scipy.stats import spearmanr as sp

    pr = float(np.corrcoef(pred, real)[0, 1])
    sr = float(sp(pred, real).statistic)
    return {
        "皮尔逊相关": round(pr, 6),
        "斯皮尔曼相关": round(sr, 6),
        "预测均值": round(float(pred.mean()), 6),
        "真实均值": round(float(real.mean()), 6),
        "预测标准差": round(float(pred.std()), 6),
        "真实标准差": round(float(real.std()), 6),
        "平均绝对误差": round(float(np.abs(pred - real).mean()), 6),
        "均方根误差": round(
            float(np.sqrt(((pred - real) ** 2).mean())), 6
        ),
    }


def within_event_ranking(pred6, real6):
    """同一事件内部六动作的排序能力。"""

    n = len(pred6)
    sp_list, kt_list = [], []
    hit1, hit2 = 0, 0
    dir_ok, tier_ok = 0, 0
    sel_r, mean_r, orc_r = [], [], []

    for i in range(n):
        p, r = pred6[i], real6[i]
        if np.std(p) > 0 and np.std(r) > 0:
            sp_list.append(float(spearmanr(p, r).statistic))
            kt_list.append(float(kendalltau(p, r).statistic))
        s = int(np.argmax(p))
        o = int(np.argmax(r))
        # 并列安全：只要取得最高实际收益就算命中
        if r[s] >= r.max() - 1e-12:
            hit1 += 1
        top2 = np.partition(r, -2)[-2]
        if r[s] >= top2 - 1e-12:
            hit2 += 1
        # 方向：动作 1-3 顺势，4-6 反向
        if (s <= 2) == (o <= 2):
            dir_ok += 1
        if (s % 3) == (o % 3):
            tier_ok += 1
        sel_r.append(r[s])
        mean_r.append(r.mean())
        orc_r.append(r[o])

    return {
        "事件数": n,
        "平均事件内斯皮尔曼": round(float(np.mean(sp_list)), 6)
        if sp_list
        else None,
        "平均事件内肯德尔": round(float(np.mean(kt_list)), 6)
        if kt_list
        else None,
        "第一名命中率": round(hit1 / n, 6),
        "前二名命中率": round(hit2 / n, 6),
        "随机基准第一名命中率": round(1 / 6, 6),
        "方向选择正确率": round(dir_ok / n, 6),
        "目标档位选择正确率": round(tier_ok / n, 6),
        "模型选择动作实际收益": round(float(np.mean(sel_r)), 6),
        "六动作平均实际收益": round(float(np.mean(mean_r)), 6),
        "事后最佳实际收益": round(float(np.mean(orc_r)), 6),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

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
    n_ev_all = len(idx)
    ev_of_row = np.repeat(np.arange(n_ev_all), 6)
    row_keep = keep[ev_of_row]

    Xenc_k = Xenc[row_keep]
    raw_k = raw.loc[row_keep].reset_index(drop=True)
    y_k = rewards[ev_of_row, 1 + (np.arange(len(ev_of_row)) % 6)][
        row_keep
    ]
    sp_k = idx["split"].to_numpy()[ev_of_row][row_keep]

    ek = ev_of_row[row_keep]
    # 动作索引必须来自「行位置」而不是事件索引
    act_k = (np.arange(len(ev_of_row)) % 6)[row_keep]
    keep_idx = np.where(keep)[0]
    remap = -np.ones(n_ev_all, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    ek2 = remap[ek]

    day_k = idx["trading_day"].to_numpy()[keep_idx]
    sym_k = idx["symbol"].to_numpy()[keep_idx]
    split_k = idx["split"].to_numpy()[keep_idx]
    rew_k = rewards[keep_idx]

    m_tr, m_se, m_te = (
        sp_k == "TRAIN",
        sp_k == "SELECTION",
        sp_k == "TEST",
    )

    Xcb, cat_cols = prepare_catboost(raw_k)
    cat_idx = [Xcb.columns.get_loc(c) for c in cat_cols]

    preds = {}

    # ---------------- 岭回归 ----------------
    best_a, best_rmse = None, float("inf")
    for a in RIDGE_ALPHAS:
        m = Ridge(alpha=a, random_state=None)
        m.fit(Xenc_k[m_tr], y_k[m_tr])
        p = m.predict(Xenc_k[m_se])
        r = float(np.sqrt(((p - y_k[m_se]) ** 2).mean()))
        if r < best_rmse:
            best_rmse, best_a = r, a
    ridge = Ridge(alpha=best_a)
    ridge.fit(Xenc_k[m_tr], y_k[m_tr])
    preds["岭回归"] = ridge.predict(Xenc_k)

    # ---------------- CatBoost ----------------
    cb = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        random_seed=SEED,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )
    cb.fit(
        Xcb.iloc[m_tr],
        y_k[m_tr],
        cat_features=cat_idx,
        eval_set=(Xcb.iloc[m_se], y_k[m_se]),
        early_stopping_rounds=100,
        use_best_model=True,
    )
    preds["CatBoost"] = cb.predict(Xcb)

    # ---------------- 神经网络 ----------------
    net = QNetwork(Xenc_k.shape[1])
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xenc_k, dtype=torch.float32)
    yt = torch.tensor(y_k, dtype=torch.float32)
    x_tr, y_tr = Xt[m_tr], yt[m_tr]
    x_se, y_se = Xt[m_se], yt[m_se]
    best_se, best_state, wait = float("inf"), None, 0
    for _ in range(60):
        net.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(perm), 1024):
            b = perm[i : i + 1024]
            loss = nn.functional.mse_loss(net(x_tr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            sl = float(nn.functional.mse_loss(net(x_se), y_se))
        if sl < best_se - 1e-8:
            best_se, wait = sl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            wait += 1
            if wait >= 5:
                break
    net.load_state_dict(best_state)
    net.eval()
    with torch.no_grad():
        preds["神经网络"] = net(Xt).numpy()

    # ---------------- 诊断 ----------------
    n_kept = len(keep_idx)
    diag_sel, diag_te = [], []
    for name, p in preds.items():
        diag_sel.append(
            {"模型": name, **diag_table(p[m_se], y_k[m_se])}
        )
        diag_te.append(
            {"模型": name, **diag_table(p[m_te], y_k[m_te])}
        )
    pd.DataFrame(diag_sel).to_csv(
        RESULTS / "model_diagnostics_selection.csv",
        index=False, encoding="utf-8-sig",
    )
    pd.DataFrame(diag_te).to_csv(
        RESULTS / "model_diagnostics_test.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---------------- 每事件六动作得分 ----------------
    scores = {}
    for name, p in preds.items():
        s6 = np.full((n_kept, 6), -np.inf)
        s6[ek2, act_k] = p
        if not np.isfinite(s6).all():
            raise RuntimeError(f"{name} 事件得分存在空缺")
        scores[name] = s6

    # ---------------- 事件内排序 ----------------
    te = split_k == "TEST"
    rank_rows = []
    for name, s6 in scores.items():
        rank_rows.append(
            {
                "模型": name,
                **within_event_ranking(s6[te], rew_k[te, 1:]),
            }
        )
    rank_df = pd.DataFrame(rank_rows)
    rank_df.to_csv(
        RESULTS / "within_event_ranking_test.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---------------- 遗憾 ----------------
    regret_rows = []
    for name, s6 in scores.items():
        sel = np.argmax(s6[te], axis=1)
        got = rew_k[te, 1:][np.arange(te.sum()), sel]
        orc = rew_k[te, 1:].max(axis=1)
        rg = orc - got
        zero = np.where(s6[te].max(axis=1) > 0, got, 0.0)
        regret_rows.append(
            {
                "模型": name,
                "平均遗憾": round(float(rg.mean()), 6),
                "遗憾中位数": round(float(np.median(rg)), 6),
                "遗憾90分位": round(float(np.percentile(rg, 90)), 6),
                "零阈值策略平均收益": round(float(zero.mean()), 6),
                "事后最佳平均收益": round(float(orc.mean()), 6),
                "模型占事后最佳比例": round(
                    float(zero.mean() / orc.mean()), 6
                )
                if orc.mean() != 0
                else None,
            }
        )
    pd.DataFrame(regret_rows).to_csv(
        RESULTS / "regret_test.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---------------- 基准 ----------------
    train_mean6 = rew_k[split_k == "TRAIN", 1:].mean(axis=0)
    best_fixed = int(np.argmax(train_mean6)) + 1

    # ---------------- 分位阈值（选择段冻结）----
    se = split_k == "SELECTION"
    strat_rows = []
    final_rows = []

    for name, s6 in scores.items():
        mx = s6.max(axis=1)
        cuts = {
            c: float(np.quantile(mx[se], 1 - c)) for c in COVERAGES
        }

        # 选择段评估各覆盖率
        cand = []
        for c in COVERAGES:
            m = se & (mx >= cuts[c])
            sel = np.argmax(s6[m], axis=1)
            got = rew_k[m, 1:][np.arange(m.sum()), sel]
            s = daily_curve(got, day_k[m])
            met = curve_metrics(s)
            cand.append((c, met.get("夏普率", -99), int(m.sum())))
        cand.sort(key=lambda t: (-round(t[1], 4), -t[2]))
        best_c = cand[0][0]

        # 测试段分层
        for c in COVERAGES:
            m = te & (mx >= cuts[c])
            if not m.any():
                continue
            sel = np.argmax(s6[m], axis=1)
            got = rew_k[m, 1:][np.arange(m.sum()), sel]
            s = daily_curve(got, day_k[m])
            tm = trade_metrics(got)
            strat_rows.append(
                {
                    "模型": name,
                    "分位组": f"最高{int(c*100)}%",
                    "阈值(选择段冻结)": round(cuts[c], 6),
                    "事件数": int(m.sum()),
                    "实际平均收益": round(float(got.mean()), 6),
                    **{
                        k: v
                        for k, v in tm.items()
                        if k in ("利润因子", "胜率", "平均盈亏比")
                    },
                    **curve_metrics(s),
                }
            )
        m = te & (mx > 0)
        if m.any():
            sel = np.argmax(s6[m], axis=1)
            got = rew_k[m, 1:][np.arange(m.sum()), sel]
            s = daily_curve(got, day_k[m])
            strat_rows.append(
                {
                    "模型": name,
                    "分位组": "全部预测>0",
                    "阈值(选择段冻结)": 0.0,
                    "事件数": int(m.sum()),
                    "实际平均收益": round(float(got.mean()), 6),
                    **{
                        k: v
                        for k, v in trade_metrics(got).items()
                        if k in ("利润因子", "胜率", "平均盈亏比")
                    },
                    **curve_metrics(s),
                }
            )

        # 两条策略
        for sname, mm in (
            ("零阈值", te & (mx > 0)),
            ("分位策略", te & (mx >= cuts[best_c])),
        ):
            sel = np.argmax(s6[mm], axis=1)
            got = rew_k[mm, 1:][np.arange(mm.sum()), sel]
            daily = np.zeros(te.sum())
            pos = np.where(mm[te])[0]
            daily[pos] = got
            s = daily_curve(daily, day_k[te])
            tm = trade_metrics(got)
            final_rows.append(
                {
                    "模型": name,
                    "策略": sname
                    + (f"(最高{int(best_c*100)}%)" if sname == "分位策略" else ""),
                    **curve_metrics(s),
                    "交易次数": int(len(got)),
                    "不交易比例": round(
                        1 - len(got) / int(te.sum()), 4
                    ),
                    "利润因子": tm.get("利润因子", 0.0),
                }
            )

    # 基准
    for nm, rr in (
        ("最佳固定交易动作", rew_k[te, best_fixed]),
        ("不交易", np.zeros(int(te.sum()))),
    ):
        s = daily_curve(rr, day_k[te])
        tm = trade_metrics(rr)
        final_rows.append(
            {
                "模型": nm,
                "策略": ACTION_NAMES[best_fixed]
                if nm.startswith("最佳")
                else "固定",
                **curve_metrics(s),
                "交易次数": int((rr != 0).sum()),
                "不交易比例": round(float((rr == 0).mean()), 4),
                "利润因子": tm.get("利润因子", 0.0),
            }
        )

    pd.DataFrame(strat_rows).to_csv(
        RESULTS / "prediction_strata_test.csv",
        index=False, encoding="utf-8-sig",
    )
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(
        RESULTS / "final_strategy_compare.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---------------- 最佳模型跨品种 / 跨时间 ----
    tradeable = final_df[~final_df["模型"].isin(["不交易"])].copy()
    tradeable = tradeable.sort_values("夏普率", ascending=False)
    best = tradeable.iloc[0]
    bm, bs = best["模型"], best["策略"]

    cov = None
    if "分位策略" in bs:
        name = bm
        s6 = scores[name]
        mx = s6.max(axis=1)
        cuts = {
            c: float(np.quantile(mx[se], 1 - c)) for c in COVERAGES
        }
        cov = float(
            [c for c in COVERAGES if f"最高{int(c*100)}%" in bs][0]
        )

    def best_daily(mask):
        s6 = scores[bm]
        mx = s6.max(axis=1)
        if cov is None:
            mm = mask & (mx > 0)
        else:
            mm = mask & (mx >= cuts[cov])
        sel = np.argmax(s6[mm], axis=1)
        got = rew_k[mm, 1:][np.arange(mm.sum()), sel]
        daily = np.zeros(mask.sum())
        pos = np.where(mm[mask])[0]
        daily[pos] = got
        return daily_curve(daily, day_k[mask]), int(len(got))

    by_sym = []
    for sym in sorted(set(sym_k[te])):
        m = te & (sym_k == sym)
        s, ntr = best_daily(m)
        by_sym.append(
            {
                "品种": sym, **curve_metrics(s), "交易次数": ntr,
                "样本数": int(m.sum()),
            }
        )
    pd.DataFrame(by_sym).to_csv(
        RESULTS / "best_model_by_symbol.csv",
        index=False, encoding="utf-8-sig",
    )

    test_days = sorted(set(day_k[te]))
    k = max(len(test_days) // 4, 1)
    blocks = [test_days[i * k : (i + 1) * k] for i in range(4)]
    if len(test_days) > 4 * k:
        blocks[3] = test_days[3 * k :]
    by_blk = []
    for bi, bd in enumerate(blocks, start=1):
        m = te & np.isin(day_k, bd)
        if not m.any():
            continue
        s, ntr = best_daily(m)
        by_blk.append(
            {
                "时间块": bi, "起": bd[0], "止": bd[-1],
                **curve_metrics(s), "交易次数": ntr,
                "样本数": int(m.sum()),
            }
        )
    pd.DataFrame(by_blk).to_csv(
        RESULTS / "best_model_by_time_block.csv",
        index=False, encoding="utf-8-sig",
    )

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(figsize=(11, 6))
    for _, r in final_df.iterrows():
        pass
    for nm in scores:
        s6 = scores[nm]
        mx = s6.max(axis=1)
        mm = te & (mx > 0)
        sel = np.argmax(s6[mm], axis=1)
        got = rew_k[mm, 1:][np.arange(mm.sum()), sel]
        daily = np.zeros(te.sum())
        pos = np.where(mm[te])[0]
        daily[pos] = got
        s = daily_curve(daily, day_k[te])
        ax.plot(range(len(s)), np.cumsum(s.to_numpy()), label=nm)
    s = daily_curve(rew_k[te, best_fixed], day_k[te])
    ax.plot(
        range(len(s)),
        np.cumsum(s.to_numpy()),
        label=f"最佳固定交易动作({ACTION_NAMES[best_fixed]})",
    )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("历史测试段 各模型零阈值策略累计收益")
    ax.set_xlabel("交易日序号")
    ax.set_ylabel("累计 R")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_model_compare_curves.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    st = pd.read_csv(RESULTS / "prediction_strata_test.csv")
    for nm in st["模型"].unique():
        d = st[st["模型"] == nm]
        d = d[d["分位组"] != "全部预测>0"]
        ax.plot(
            d["分位组"], d["实际平均收益"], marker="o", label=nm
        )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("预测分位 -> 实际平均收益（历史测试段）")
    ax.set_ylabel("实际平均收益 R")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "fig_prediction_strata.png", dpi=110)
    plt.close(fig)

    audit = {
        "git_head": git_head(),
        "seed": SEED,
        "best_fixed_trade_action": ACTION_NAMES[best_fixed],
        "train_mean_by_action": {
            ACTION_NAMES[1 + i]: round(float(train_mean6[i]), 6)
            for i in range(6)
        },
        "ridge_alpha_selected": best_a,
        "ridge_alpha_candidates": RIDGE_ALPHAS,
        "catboost_iterations_used": int(cb.tree_count_),
        "events_used": int(n_kept),
        "events_dropped_roll_gap": int((~keep).sum()),
        "best_model": bm,
        "best_strategy": bs,
    }
    (RESULTS / "model_compare_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("RL_62D_MODEL_COMPARE_DONE")
    print("best_fixed", ACTION_NAMES[best_fixed], train_mean6.round(4))
    print("ridge_alpha", best_a, "catboost_trees", cb.tree_count_)
    print()
    print("=== 选择段诊断 ===")
    print(pd.DataFrame(diag_sel).to_string(index=False))
    print()
    print("=== 测试段诊断 ===")
    print(pd.DataFrame(diag_te).to_string(index=False))
    print()
    print("=== 事件内排序 ===")
    print(rank_df.to_string(index=False))
    print()
    print("=== 遗憾 ===")
    print(pd.DataFrame(regret_rows).to_string(index=False))
    print()
    print("=== 最终策略 ===")
    print(final_df.to_string(index=False))


if __name__ == "__main__":
    main()
