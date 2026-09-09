"""M5：动态退出价值上限诊断（诊断实验，不训练、不优化）。

要回答的核心问题
----------------
1. 动态退出【本身】有没有足够的理论空间？
2. 第一版强化学习失败，是因为没有空间，还是没学到？
3. 提前退出主要在减少左尾，还是在砍掉右尾？
4. 是否有证据支持下一轮研究「受限退出动作空间」？

方法
----
不训练任何模型。对每一笔交易，轨迹已给出：

    step 1..11 每个决策点开盘退出的 exit_value_R
    永远持有的固定终值 fixed_reward_R

事后完美动态退出（理论上限，不可交易）：

    oracle_dynamic_reward_R = max(fixed_reward_R, exit_value_R@step1..11)

对三种固定目标 1.5R / 2.0R / 2.5R 分别计算，避免 2.5R 选择偏差。

指标口径
--------
- 夏普率 / 最大回撤：每日机会集等权曲线（未交易记 0）。
- 交易次数：实际执行交易的事件数（收益为 0 仍计入）。
- trade_total_R：实际交易收益直接求和。
- opportunity_curve_cumulative_R：机会集等权曲线累计值。

用法：python -m research.m5_exit_value_diagnosis_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.rl_62d_core_v1 import ACTION_NAMES
from research.train_rl_62d_v1 import curve_metrics

OUT = Path("research/analysis_results/m5_exit_value")
OUT.mkdir(parents=True, exist_ok=True)

TRAJ = Path(
    "research/analysis_results/rl_exit_v1/rl_exit_trajectory_v1.parquet"
)
EVENTS = Path(
    "research/analysis_results/m2_temporal_fix/m2fix_events.parquet"
)
DT_AUDIT = Path(
    "research/analysis_results/m3_direction_target/m3dt_audit.json"
)
RL_EP = Path("research/analysis_results/rl_exit_v1/rl_episodes.parquet")

TARGETS = ["1.5R", "2.0R", "2.5R"]
A6 = list(ACTION_NAMES[1:])
FOLLOW_COLS = [a for a in A6 if a.startswith("FOLLOW")]
FADE_COLS = [a for a in A6 if a.startswith("FADE")]

ENTRY_MODEL = "LightGBM回归"
ENTRY_VIEW = "SNAPSHOT"

# 每折选择段冻结的覆盖率（来自预注册的 A/B/C 方案）
COV_KEY = {"1.5R": "简化A_覆盖率", "2.0R": "简化B_覆盖率",
           "2.5R": "简化C_覆盖率"}

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


def gain_stats(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(样本数=0)
    return dict(
        样本数=int(len(x)),
        平均理论退出增量=round(float(x.mean()), 6),
        中位数=round(float(np.median(x)), 6),
        p90=round(float(np.percentile(x, 90)), 6),
        正增量比例=round(float((x > 1e-12).mean()), 4),
        零增量比例=round(float(np.isclose(x, 0, atol=1e-12).mean()), 4),
    )


def main():
    traj = pd.read_parquet(TRAJ)
    ev = pd.read_parquet(EVENTS)
    ev = ev[(ev["模型"] == ENTRY_MODEL) & (ev["view"] == ENTRY_VIEW)]
    ev = ev.reset_index(drop=True)
    yh = ev[[f"yhat_{a}" for a in A6]].to_numpy(float)
    ev["follow_score"] = yh[:, [A6.index(c) for c in FOLLOW_COLS]].mean(axis=1)
    ev["fade_score"] = yh[:, [A6.index(c) for c in FADE_COLS]].mean(axis=1)
    ev["is_follow"] = ev["follow_score"] >= ev["fade_score"]
    ev["event_score"] = np.where(
        ev["is_follow"], ev["follow_score"], ev["fade_score"]
    )

    audit = json.load(open(DT_AUDIT))
    fold_cov = {
        t: {int(p["折"]): p[COV_KEY[t]] for p in audit["fold_plan"]}
        for t in TARGETS
    }

    # ------------------------------------------------------------------ #
    # A. 事后完美动态退出理论上限（对三种固定目标）
    # ------------------------------------------------------------------ #
    long = []
    for t in TARGETS:
        for f in sorted(ev["fold"].unique()):
            sel = ev[(ev["fold"] == f) & (ev["segment"] == "sel")]
            te = ev[(ev["fold"] == f) & (ev["segment"] == "test")].copy()
            cutoff = np.quantile(
                sel["event_score"].to_numpy(), 1 - fold_cov[t][int(f)]
            )
            te["traded"] = te["event_score"].to_numpy() > cutoff
            te["action"] = np.where(
                te["is_follow"].to_numpy(), f"FOLLOW_{t}", f"FADE_{t}"
            )
            tr = traj.merge(
                te.loc[te["traded"], ["candidate_id", "action"]],
                on="candidate_id", how="inner",
            )
            tr = tr[tr["initial_action"] == tr["action"]]
            fixed = tr.groupby("candidate_id")["all_hold_terminal_R"].first()
            dec = tr[tr["is_decision_point"]]
            best_exit = dec.groupby("candidate_id")["exit_value_R"].max()
            best_exit = best_exit.reindex(fixed.index)
            oracle = np.maximum(fixed.to_numpy(),
                                best_exit.fillna(-np.inf).to_numpy())

            te["fixed_R"] = te["candidate_id"].map(fixed).astype(float)
            te["oracle_R"] = te["candidate_id"].map(
                pd.Series(oracle, index=fixed.index)).astype(float)
            te["fixed_R"] = te["fixed_R"].fillna(0.0)
            te["oracle_R"] = te["oracle_R"].fillna(0.0)
            te["realized_fixed"] = np.where(te["traded"], te["fixed_R"], 0.0)
            te["realized_oracle"] = np.where(te["traded"], te["oracle_R"], 0.0)
            te["exit_gain"] = np.where(
                te["traded"], te["oracle_R"] - te["fixed_R"], np.nan
            )
            te["目标"] = t
            long.append(te[[
                "candidate_id", "symbol", "day", "fold", "action", "traded",
                "fixed_R", "oracle_R", "realized_fixed", "realized_oracle",
                "exit_gain", "目标",
            ]])
    long = pd.concat(long, ignore_index=True)
    long.to_parquet(OUT / "m5_events.parquet", index=False)

    tgt_rows, fold_rows, sym_rows, space_rows = [], [], [], []
    for t in TARGETS:
        g = long[long["目标"] == t]
        for tag, col in (("固定退出", "realized_fixed"),
                         ("事后动态退出", "realized_oracle")):
            p = perf(g[col].to_numpy(), g["day"].to_numpy(),
                     g["traded"].to_numpy())
            tgt_rows.append(dict(目标=t, 退出方式=tag, **p))
        a = tgt_rows[-2]
        b = tgt_rows[-1]
        space_rows.append(dict(
            目标=t,
            固定退出夏普=a["夏普率"], 事后动态退出夏普=b["夏普率"],
            夏普理论增量=round(b["夏普率"] - a["夏普率"], 4),
            固定机会集累计=a["opportunity_curve_cumulative_R"],
            事后机会集累计=b["opportunity_curve_cumulative_R"],
            **gain_stats(g["exit_gain"]),
        ))
        for f, gg in g.groupby("fold"):
            for tag, col in (("固定退出", "realized_fixed"),
                             ("事后动态退出", "realized_oracle")):
                fold_rows.append(dict(
                    目标=t, 折=f"F{f}", 退出方式=tag,
                    **perf(gg[col].to_numpy(), gg["day"].to_numpy(),
                           gg["traded"].to_numpy())))
        for s, gg in g.groupby("symbol"):
            for tag, col in (("固定退出", "realized_fixed"),
                             ("事后动态退出", "realized_oracle")):
                sym_rows.append(dict(
                    目标=t, 品种=s, 退出方式=tag,
                    **perf(gg[col].to_numpy(), gg["day"].to_numpy(),
                           gg["traded"].to_numpy())))

    tgt_df = pd.DataFrame(tgt_rows)
    space_df = pd.DataFrame(space_rows)
    fold_df = pd.DataFrame(fold_rows)
    sym_df = pd.DataFrame(sym_rows)

    # 理论空间拆分：折 / 品种 / 方向
    split_rows = []
    for t in TARGETS:
        g = long[(long["目标"] == t) & long["traded"]]
        split_rows.append(dict(维度="整体", 取值="ALL", 目标=t,
                               **gain_stats(g["exit_gain"])))
        for f, gg in g.groupby("fold"):
            split_rows.append(dict(维度="折", 取值=f"F{f}", 目标=t,
                                   **gain_stats(gg["exit_gain"])))
        for s, gg in g.groupby("symbol"):
            split_rows.append(dict(维度="品种", 取值=s, 目标=t,
                                   **gain_stats(gg["exit_gain"])))
        for d, gg in g.groupby(g["action"].str.startswith("FOLLOW")):
            split_rows.append(dict(
                维度="方向", 取值="FOLLOW" if d else "FADE", 目标=t,
                **gain_stats(gg["exit_gain"])))
    split_df = pd.DataFrame(split_rows)

    tgt_df.to_csv(OUT / "m5_oracle_by_target.csv", index=False,
                  encoding="utf-8-sig")
    space_df.to_csv(OUT / "m5_oracle_space.csv", index=False,
                    encoding="utf-8-sig")
    split_df.to_csv(OUT / "m5_oracle_space_split.csv", index=False,
                    encoding="utf-8-sig")
    fold_df.to_csv(OUT / "m5_oracle_by_fold.csv", index=False,
                   encoding="utf-8-sig")
    sym_df.to_csv(OUT / "m5_oracle_by_symbol.csv", index=False,
                  encoding="utf-8-sig")

    # ------------------------------------------------------------------ #
    # B. 当前三种强化学习提前退出诊断
    # ------------------------------------------------------------------ #
    if not RL_EP.exists():
        print("[M5] rl_episodes.parquet 缺失，跳过 RL 诊断", flush=True)
        rl_summary = pd.DataFrame()
    else:
        rl = pd.read_parquet(RL_EP)
        rt = rl[rl["traded"]].copy()
        rt["gain"] = rt["rl"] - rt["fixed"]
        rt["方向"] = np.where(rt["action"].str.startswith("FOLLOW"),
                              "FOLLOW", "FADE")
        rt["退出时浮盈亏桶"] = pd.cut(
            rt["rl"], bins=R_BINS, labels=R_LABELS, right=False
        )
        rt["持仓步数桶"] = pd.cut(
            rt["held"], bins=STEP_BINS, labels=STEP_LABELS, right=True
        )

        rl_summary = []
        for k, g in rt.groupby("模型"):
            e = g[g["early"]]
            n_tr, n_e = len(g), len(e)
            if n_e == 0:
                continue
            gain = e["gain"].to_numpy()
            rl_summary.append(dict(
                模型=k, 交易次数=n_tr,
                提前退出次数=n_e,
                提前退出比例=round(n_e / n_tr, 4),
                改善次数=int((gain > 0).sum()),
                改善比例=round(float((gain > 0).mean()), 4),
                恶化次数=int((gain < 0).sum()),
                恶化比例=round(float((gain < 0).mean()), 4),
                平均退出增量=round(float(gain.mean()), 6),
                改善交易平均增量=round(float(gain[gain > 0].mean()), 6)
                if (gain > 0).any() else None,
                恶化交易平均损失=round(float(gain[gain < 0].mean()), 6)
                if (gain < 0).any() else None,
            ))
        rl_summary = pd.DataFrame(rl_summary)
        rl_summary.to_csv(OUT / "m5_rl_early_exit.csv", index=False,
                          encoding="utf-8-sig")

        # 右尾截断：提前退出后，若继续固定持有本来会发生什么
        rt_rows = []
        for k, g in rt.groupby("模型"):
            e = g[g["early"]]
            for reason, gg in e.groupby("reason"):
                rt_rows.append(dict(
                    模型=k, 固定策略后续=reason,
                    提前退出次数=len(gg),
                    占提前退出比例=round(len(gg) / len(e), 4),
                    平均退出增量=round(float(gg["gain"].mean()), 6),
                    改善比例=round(float((gg["gain"] > 0).mean()), 4),
                ))
        rt_df = pd.DataFrame(rt_rows)
        rt_df.to_csv(OUT / "m5_rl_right_tail.csv", index=False,
                     encoding="utf-8-sig")

        # 按退出时浮盈亏分层
        bucket_rows = []
        for k, g in rt.groupby("模型"):
            e = g[g["early"]]
            for b, gg in e.groupby("退出时浮盈亏桶", observed=True):
                if len(gg) == 0:
                    continue
                bucket_rows.append(dict(
                    模型=k, 浮盈亏桶=str(b), 提前退出次数=len(gg),
                    平均退出增量=round(float(gg["gain"].mean()), 6),
                    改善比例=round(float((gg["gain"] > 0).mean()), 4),
                    后续TARGET比例=round(
                        float((gg["reason"] == "TARGET").mean()), 4),
                    后续STOP比例=round(
                        float((gg["reason"] == "STOP").mean()), 4),
                ))
        pd.DataFrame(bucket_rows).to_csv(
            OUT / "m5_rl_by_unrealized.csv", index=False, encoding="utf-8-sig")

        # 按持仓步数分层
        step_rows = []
        for k, g in rt.groupby("模型"):
            e = g[g["early"]]
            for b, gg in e.groupby("持仓步数桶", observed=True):
                if len(gg) == 0:
                    continue
                step_rows.append(dict(
                    模型=k, 持仓步数=str(b), 提前退出次数=len(gg),
                    平均退出增量=round(float(gg["gain"].mean()), 6),
                    改善比例=round(float((gg["gain"] > 0).mean()), 4),
                    后续TARGET比例=round(
                        float((gg["reason"] == "TARGET").mean()), 4),
                    后续STOP比例=round(
                        float((gg["reason"] == "STOP").mean()), 4),
                ))
        pd.DataFrame(step_rows).to_csv(
            OUT / "m5_rl_by_step.csv", index=False, encoding="utf-8-sig")

        # 按折 / 品种
        for dim, col, name in (("折", "折", "m5_rl_by_fold"),
                               ("品种", "symbol", "m5_rl_by_symbol")):
            rows = []
            for k, g in rt.groupby("模型"):
                for v, gg in g.groupby(col):
                    e = gg[gg["early"]]
                    if len(e) == 0:
                        continue
                    rows.append(dict(
                        模型=k, **{dim: v}, 交易次数=len(gg),
                        提前退出次数=len(e),
                        提前退出比例=round(len(e) / len(gg), 4),
                        平均退出增量=round(float(e["gain"].mean()), 6),
                        改善比例=round(float((e["gain"] > 0).mean()), 4),
                    ))
            pd.DataFrame(rows).to_csv(
                OUT / f"{name}.csv", index=False, encoding="utf-8-sig")

    (OUT / "m5_audit.json").write_text(json.dumps(dict(
        script="research/m5_exit_value_diagnosis_v1.py",
        entry_model=ENTRY_MODEL, entry_view=ENTRY_VIEW,
        targets=TARGETS,
        fold_coverage=fold_cov,
        note=(
            "事后完美动态退出为理论上限，不是可交易策略；"
            "夏普/回撤统一使用每日机会集等权曲线；"
            "交易次数按是否实际执行统计（收益为 0 仍计入）。"
        ),
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 60)
    print("\n=== 各固定目标：固定退出 vs 事后动态退出 ===")
    print(tgt_df.round(6).to_string(index=False))
    print("\n=== 动态退出理论空间（按目标）===")
    print(space_df.round(6).to_string(index=False))
    print("\n=== 理论空间拆分（折/品种/方向）===")
    print(split_df.round(6).to_string(index=False))
    if len(rl_summary):
        print("\n=== 当前强化学习提前退出诊断 ===")
        print(rl_summary.round(6).to_string(index=False))
    print("\nM5_DONE")


if __name__ == "__main__":
    main()
