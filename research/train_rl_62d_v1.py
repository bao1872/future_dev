#!/usr/bin/env python3

"""62维 RL V1 -- 训练 + 历史测试 + 收益曲线。

单步完整反馈决策：不使用目标网络、经验回放、折扣因子、
下一状态与贝尔曼更新。

主收益曲线口径：每日机会集等权标准化收益
（当天所有合格事件等权平均，不交易记 0）。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Heiti TC",
    "Songti SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import pandas as pd
import torch
from torch import nn

from research.rl_62d_core_v1 import (
    ACTION_NAMES,
    N_ACTIONS,
    NO_TRADE_INDEX,
    QNetwork,
    choose_best_fixed_action,
    compute_training_loss,
    oracle_actions,
    realized_policy_rewards,
    select_actions,
)

RESULTS = Path("research/analysis_results/rl_62d_v1")

# ---- 冻结的训练参数（禁止搜索）----
SEED = 20240901
LEARNING_RATE = 1e-3
BATCH_SIZE = 1024
MAX_EPOCHS = 60
EARLY_STOP_PATIENCE = 5

# 换月拼接断层过滤：窗口内最大跳空超过该 ATR 倍数即剔除。
# 该过滤与方向无关，不会系统性偏向多空任何一方。
ROLL_GAP_ATR_THRESHOLD = 10.0

TRADING_DAYS_PER_YEAR = 252

# 损失函数。默认 smooth_l1（合同指定）。
# mse 变体用于与「按期望值决策」的规则对齐：
# 类 L1 损失收敛到条件中位数，而中位数在 R 分布上是止损，
# 会让「最高预测收益 > 0 才交易」退化成永不交易。
LOSS = sys.argv[1] if len(sys.argv) > 1 else "smooth_l1"
SUFFIX = LOSS


def loss_fn(pred, target):
    if LOSS == "mse":
        return nn.functional.mse_loss(pred, target)
    return nn.functional.smooth_l1_loss(pred, target)


def git_head() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------
# 每日机会集等权曲线
# ------------------------------------------------------------


def daily_curve(r: np.ndarray, groups: np.ndarray):
    """按组（交易日 或 交易日x品种）等权平均。"""

    df = pd.DataFrame({"g": groups, "r": r})
    s = df.groupby("g", sort=True)["r"].mean()
    return s


def curve_metrics(s: pd.Series):
    if len(s) == 0:
        return {}
    v = s.to_numpy(float)
    cum = np.cumsum(v)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    std = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    sharpe = (
        float(v.mean()) / std * np.sqrt(TRADING_DAYS_PER_YEAR)
        if std > 0
        else 0.0
    )
    return {
        "days": int(len(v)),
        "累计收益": round(float(cum[-1]), 4),
        "夏普率": round(sharpe, 4),
        "最大回撤": round(float(dd.max()), 4),
        "正收益日比例": round(float((v > 0).mean()), 4),
        "日收益标准差": round(float(std), 6),
    }


def trade_metrics(r: np.ndarray):
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return {}
    win, loss = r[r > 0], r[r < 0]
    gp, gl = float(win.sum()), float(-loss.sum())
    return {
        "交易次数": int(len(r)),
        "胜率": round(float((r > 0).mean()), 4),
        "平均盈利": round(float(win.mean()), 4) if len(win) else 0.0,
        "平均亏损": round(float(loss.mean()), 4) if len(loss) else 0.0,
        "平均盈亏比": (
            round(float(win.mean() / abs(loss.mean())), 4)
            if len(win) and len(loss)
            else 0.0
        ),
        "利润因子": round(gp / gl, 4) if gl > 0 else float("inf"),
        "最大单笔盈利": round(float(r.max()), 4),
        "最大单笔亏损": round(float(r.min()), 4),
    }


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    X = np.load(RESULTS / "features_v2.npy")
    rewards = np.load(RESULTS / "reward_matrix_v2.npy")
    gaps = np.load(RESULTS / "max_gap_atr_v2.npy")
    idx = pd.read_csv(RESULTS / "event_index_v2.csv")

    n_events = len(idx)
    n_rows = n_events * 6
    assert X.shape[0] == n_rows

    # ---- 换月断层过滤（方向对称）----
    keep = (
        np.nan_to_num(gaps[:, 1:], nan=0.0).max(axis=1)
        <= ROLL_GAP_ATR_THRESHOLD
    )
    n_dropped_gap = int((~keep).sum())

    ev_of_row = np.repeat(np.arange(n_events), 6)
    row_keep = keep[ev_of_row]

    split = idx["split"].to_numpy()
    row_split = split[ev_of_row]

    symbol = idx["symbol"].to_numpy()
    day = idx["trading_day"].to_numpy()

    Xk = X[row_keep]
    yk = rewards[ev_of_row, 1 + (np.arange(n_rows) % 6)][row_keep]
    sk = row_split[row_keep]
    ek = ev_of_row[row_keep]

    keep_idx = np.where(keep)[0]
    remap = -np.ones(n_events, dtype=np.int64)
    remap[keep_idx] = np.arange(len(keep_idx))
    ek2 = remap[ek]

    Xt = torch.tensor(Xk, dtype=torch.float32)
    yt = torch.tensor(yk, dtype=torch.float32)

    m_tr = sk == "TRAIN"
    m_se = sk == "SELECTION"
    m_te = sk == "TEST"

    x_tr, y_tr = Xt[m_tr], yt[m_tr]
    x_se, y_se = Xt[m_se], yt[m_se]
    x_te, y_te = Xt[m_te], yt[m_te]

    dim = Xt.shape[1]
    model = QNetwork(dim)
    opt = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_se = float("inf")
    best_epoch = 0
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(x_tr))
        tot = 0.0
        for i in range(0, len(perm), BATCH_SIZE):
            b = perm[i : i + BATCH_SIZE]
            loss = loss_fn(model(x_tr[b]), y_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss) * len(b)
        tr_loss = tot / max(len(perm), 1)

        model.eval()
        with torch.no_grad():
            se_loss = float(loss_fn(model(x_se), y_se))
        history.append(
            {
                "epoch": epoch,
                "train_loss": round(tr_loss, 6),
                "selection_loss": round(se_loss, 6),
            }
        )

        if se_loss < best_se - 1e-8:
            best_se = se_loss
            best_epoch = epoch
            best_state = {
                k: v.clone() for k, v in model.state_dict().items()
            }
            wait = 0
        else:
            wait += 1
            if wait >= EARLY_STOP_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        scores_all = model(Xt).numpy()

    n_kept = len(keep_idx)
    scores_by_event = np.full((n_kept, 6), -np.inf)
    scores_by_event[ek2, ek % 6] = scores_all

    rew_kept = rewards[keep_idx]
    sym_kept = symbol[keep_idx]
    day_kept = day[keep_idx]
    split_kept = split[keep_idx]

    chosen = select_actions(scores_by_event)
    realized_model = rew_kept[np.arange(n_kept), chosen]

    # ---- 预测诊断：判断是「无信号」还是「模型失效」----
    pred_rows = scores_all
    act_of_row = np.arange(len(pred_rows)) % 6
    per_action_corr = {}
    for k in range(6):
        m = act_of_row == k
        if m.sum() > 10:
            per_action_corr[ACTION_NAMES[1 + k]] = round(
                float(np.corrcoef(pred_rows[m], yk[m])[0, 1]), 6
            )
    pred_corr = round(
        float(np.corrcoef(pred_rows, yk)[0, 1]), 6
    )

    fixed_action = choose_best_fixed_action(
        rew_kept[split_kept == "TRAIN"]
    )
    realized_fixed = rew_kept[:, fixed_action]

    orc = oracle_actions(rew_kept)
    realized_oracle = rew_kept[np.arange(n_kept), orc]

    realized_zero = np.zeros(n_kept)

    te = split_kept == "TEST"

    strategies = {
        "不交易": realized_zero,
        f"固定动作({ACTION_NAMES[fixed_action]})": realized_fixed,
        "62维模型": realized_model,
        "事后理论上限": realized_oracle,
    }

    # ---- 主结果表 ----
    rows = []
    for name, r in strategies.items():
        s = daily_curve(r[te], day_kept[te])
        m = curve_metrics(s)
        tm = trade_metrics(r[te][chosen[te] > 0] if name == "62维模型" else r[te])
        rows.append(
            {
                "策略": name,
                **m,
                "交易次数": tm.get("交易次数", 0),
                "不交易比例": (
                    round(float((chosen[te] == NO_TRADE_INDEX).mean()), 4)
                    if name == "62维模型"
                    else round(float((r[te] == 0).mean()), 4)
                ),
                "利润因子": tm.get("利润因子", 0.0),
            }
        )
    core = pd.DataFrame(rows)
    core.to_csv(
        RESULTS / f"test_core_metrics_{SUFFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 分品种 ----
    by_sym = []
    for sym in sorted(set(sym_kept[te])):
        m = (sym_kept == sym) & te
        for name, r in strategies.items():
            if name == "不交易":
                continue
            s = daily_curve(r[m], day_kept[m])
            by_sym.append(
                {
                    "品种": sym,
                    "策略": name,
                    **curve_metrics(s),
                    "交易次数": int((chosen[m] > 0).sum())
                    if name == "62维模型"
                    else int(m.sum()),
                    "利润因子": trade_metrics(r[m]).get("利润因子", 0.0),
                }
            )
    pd.DataFrame(by_sym).to_csv(
        RESULTS / f"test_by_symbol_{SUFFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 四段时间稳定性 ----
    test_days = sorted(set(day_kept[te]))
    k = max(len(test_days) // 4, 1)
    blocks = [
        test_days[i * k : (i + 1) * k] for i in range(4)
    ]
    if len(test_days) > 4 * k:
        blocks[3] = test_days[3 * k :]

    by_block = []
    for bi, bd in enumerate(blocks, start=1):
        m = te & np.isin(day_kept, bd)
        if not m.any():
            continue
        for name, r in strategies.items():
            if name == "不交易":
                continue
            s = daily_curve(r[m], day_kept[m])
            by_block.append(
                {
                    "时间块": bi,
                    "起": bd[0],
                    "止": bd[-1],
                    "策略": name,
                    **curve_metrics(s),
                    "交易次数": int((chosen[m] > 0).sum())
                    if name == "62维模型"
                    else int(m.sum()),
                    "利润因子": trade_metrics(r[m]).get("利润因子", 0.0),
                }
            )
    pd.DataFrame(by_block).to_csv(
        RESULTS / f"test_by_time_block_{SUFFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 交易层指标（模型实际交易）----
    tm = trade_metrics(realized_model[te][chosen[te] > 0])
    tm["不交易比例"] = round(
        float((chosen[te] == NO_TRADE_INDEX).mean()), 4
    )
    tm["总机会数"] = int(te.sum())
    pd.DataFrame([{"项目": k, "数值": v} for k, v in tm.items()]).to_csv(
        RESULTS / f"test_trading_metrics_{SUFFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 每日收益 ----
    dd = pd.DataFrame(
        {
            "trading_day": day_kept[te],
            "symbol": sym_kept[te],
            **{
                name: r[te]
                for name, r in strategies.items()
                if name != "不交易"
            },
            "model_action": chosen[te],
        }
    )
    dd.to_csv(
        RESULTS / f"daily_returns_test_{SUFFIX}.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---- 图 ----
    fig, ax = plt.subplots(figsize=(11, 6))
    for name, r in strategies.items():
        s = daily_curve(r[te], day_kept[te])
        ax.plot(
            range(len(s)), np.cumsum(s.to_numpy()), label=name
        )
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("历史测试段 四策略累计收益（每日机会集等权）")
    ax.set_xlabel("交易日序号")
    ax.set_ylabel("累计 R")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"fig_test_curves_{SUFFIX}.png", dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for sym in sorted(set(sym_kept[te])):
        m = (sym_kept == sym) & te
        s = daily_curve(realized_model[m], day_kept[m])
        ax.plot(range(len(s)), np.cumsum(s.to_numpy()), label=sym)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("62维模型 分品种累计收益")
    ax.set_xlabel("交易日序号")
    ax.set_ylabel("累计 R")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / f"fig_by_symbol_{SUFFIX}.png", dpi=110)
    plt.close(fig)

    blk_vals, blk_names = [], []
    for bi, bd in enumerate(blocks, start=1):
        m = te & np.isin(day_kept, bd)
        if not m.any():
            continue
        s = daily_curve(realized_model[m], day_kept[m])
        blk_vals.append(float(np.sum(s.to_numpy())))
        blk_names.append(f"块{bi}")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(blk_names, blk_vals)
    ax.axhline(0, color="grey", lw=0.8)
    ax.set_title("62维模型 四时间块累计收益")
    ax.set_ylabel("累计 R")
    fig.tight_layout()
    fig.savefig(RESULTS / f"fig_time_blocks_{SUFFIX}.png", dpi=110)
    plt.close(fig)

    s = daily_curve(realized_model[te], day_kept[te])
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(s.to_numpy(), bins=60)
    ax.set_title("62维模型 每日标准化收益分布")
    ax.set_xlabel("日收益 R")
    fig.tight_layout()
    fig.savefig(RESULTS / f"fig_daily_distribution_{SUFFIX}.png", dpi=110)
    plt.close(fig)

    # ---- 审计 ----
    audit = {
        "git_head": git_head(),
        "seed": SEED,
        "training_params": {
            "network": f"{dim}->128->128->64->1",
            "optimizer": "Adam",
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "max_epochs": MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "early_stop_metric": "selection_smooth_l1_loss",
            "best_epoch": best_epoch,
            "loss": LOSS,
            "no_target_network": True,
            "no_replay_buffer": True,
            "no_discount_factor": True,
            "no_next_state": True,
            "no_bellman_update": True,
        },
        "data": {
            "raw_feature_count": 62,
            "encoded_feature_count": int(dim),
            "evaluable_events": int(n_events),
            "roll_gap_atr_threshold": ROLL_GAP_ATR_THRESHOLD,
            "events_dropped_by_roll_gap": n_dropped_gap,
            "events_used": int(n_kept),
        },
        "fixed_action": ACTION_NAMES[fixed_action],
        "prediction_diagnostics": {
            "pred_realized_corr_all_rows": pred_corr,
            "pred_realized_corr_by_action": per_action_corr,
            "predicted_mean": round(float(pred_rows.mean()), 6),
            "predicted_std": round(float(pred_rows.std()), 6),
            "test_events_with_positive_best_prediction": int(
                (scores_by_event.max(axis=1) > 0).sum()
            ),
            "test_events": int(n_kept),
        },
        "curve_definition": (
            "每日机会集等权标准化收益：当天所有合格事件等权平均，"
            "不交易记 0；研究型标准化信号曲线，不是账户资金曲线"
        ),
        "history": history,
    }
    (RESULTS / f"training_audit_v1_{SUFFIX}.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("RL_62D_TRAIN_V1_DONE")
    print("encoded_dim", dim, "best_epoch", best_epoch)
    print("events_used", n_kept, "dropped_by_roll_gap", n_dropped_gap)
    print("fixed_action", ACTION_NAMES[fixed_action])
    print()
    print(core.to_string(index=False))


if __name__ == "__main__":
    main()
