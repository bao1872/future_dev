"""第一版强化学习：有限期限最优停止（HOLD / EXIT 动态退出）。

入场规则（第十部分）
--------------------
修正后时间增强不稳定 → 按「稳定性优先」采用 LightGBM 快照。
方向 × 目标实验显示「模型选方向 + 固定 2.5R」(简化C) 最优，
且四个品种全部改善。因此入场固定为：

    direction = FOLLOW if mean(pred FOLLOW_*) >= mean(pred FADE_*) else FADE
    target    = 2.5R
    coverage  = 每折选择段冻结（沿用 M2 覆盖率集合）

强化学习只负责
--------------
    HOLD / EXIT

不负责是否入场、方向、目标R、止损、加减仓、反手、移动止损。

建模（第十二/十三部分）
----------------------
EXIT 价值直接计算：exit_value_R = 当前开盘平仓的 R。
只学习 continuation_value_R（继续持有的价值）。
倒序训练：step 11 → step 1；step 0 强制 HOLD，不训练退出决策。
gamma = 1.0，不做奖励塑形。

目标：
    若该步 HOLD 后本根 K 线即终值（止损/止盈/同时/timeout）
        target = 真实终值
    否则
        target = V(next) = max(next_exit_value_R, C_hat(next))
    若下一步为强制跳空（agent 无权决策）
        V(next) = 该强制终值

模型：线性(Ridge) / LightGBM / XGBoost。禁止深度学习。

对照（第十四部分）
------------------
完全相同入场事件、方向、目标、止损下：
    固定退出（现有止损/止盈/12根）
    vs 线性动态退出 / LightGBM动态退出 / XGBoost动态退出

用法：python -m research.rl_exit_fvi_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.rl_62d_core_v1 import ACTION_NAMES
from research.train_rl_62d_v1 import curve_metrics

OUT = Path("research/analysis_results/rl_exit_v1")
OUT.mkdir(parents=True, exist_ok=True)

TRAJ = OUT / "rl_exit_trajectory_v1.parquet"
EVENTS = Path(
    "research/analysis_results/m2_temporal_fix/m2fix_events.parquet"
)

ENTRY_MODEL = "LightGBM回归"
ENTRY_VIEW = "SNAPSHOT"
TARGET = "2.5R"                      # 简化C
# 每折选择段冻结的覆盖率（来自 m3dt_audit.json）
FOLD_COVERAGE = {1: 0.2, 2: 0.3, 3: 0.2, 4: 0.1}

A6 = list(ACTION_NAMES[1:])
FOLLOW_COLS = [a for a in A6 if a.startswith("FOLLOW")]
FADE_COLS = [a for a in A6 if a.startswith("FADE")]

POS_NUM = [
    "current_unrealized_R", "distance_to_stop_R", "distance_to_target_R",
    "bars_held", "bars_remaining", "trade_direction", "target_R",
    "max_favorable_excursion_R_so_far", "max_adverse_excursion_R_so_far",
    "previous_step_price_change_R", "minutes_since_previous_valid_bar",
]
FORCED = {"GAP_STOP", "GAP_TARGET"}
MODELS = ("线性", "LightGBM", "XGBoost")


def perf(realized, day, traded):
    """统一口径：夏普/回撤用每日机会集等权曲线；交易次数按是否执行统计。"""
    daily = pd.Series(realized, index=day).groupby(level=0).mean().sort_index()
    cm = curve_metrics(daily)
    tr = np.asarray(traded, dtype=bool)
    rtr = np.asarray(realized, dtype=float)[tr]
    return dict(
        # 主指标：每日机会集等权曲线
        opportunity_curve_cumulative_R=float(cm["累计收益"]),
        夏普率=float(cm["夏普率"]),
        最大回撤=float(cm["最大回撤"]),
        利润因子=float(m2._pf(rtr)),
        # 交易次数 = 实际执行交易的事件数（收益为 0 仍计入）
        交易次数=int(tr.sum()),
        trade_total_R=round(float(rtr.sum()), 4),
        平均交易R=round(float(rtr.mean()), 6) if len(rtr) else None,
    )


def fit_model(kind, X, y, seed=m2.SEED):
    if kind == "线性":
        from sklearn.linear_model import Ridge
        m = Ridge(alpha=1.0, random_state=seed)
        m.fit(X, y)
        return m
    if kind == "LightGBM":
        import lightgbm as lgb
        m = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, num_leaves=31,
            random_state=seed, verbose=-1, n_jobs=1,
        )
        m.fit(X, y)
        return m
    if kind == "XGBoost":
        import xgboost as xgb
        m = xgb.XGBRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            random_state=seed, n_jobs=1, tree_method="hist",
        )
        m.fit(X, y)
        return m
    raise ValueError(kind)


def main():
    traj = pd.read_parquet(TRAJ)
    print(f"[RL] trajectory rows={len(traj)}", flush=True)

    # ---------- 入场规则（测试段）----------
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

    # 每折：选择段定 cutoff（覆盖率冻结），测试段应用
    entry = []
    for f in sorted(ev["fold"].unique()):
        sel = ev[(ev["fold"] == f) & (ev["segment"] == "sel")]
        te = ev[(ev["fold"] == f) & (ev["segment"] == "test")]
        c = FOLD_COVERAGE[int(f)]
        cutoff = np.quantile(sel["event_score"].to_numpy(), 1 - c)
        traded = te["event_score"].to_numpy() > cutoff
        # 保留【全部】测试事件并标记是否交易，未交易在机会集曲线中记 0
        sub = te.copy()
        sub["traded"] = traded
        sub["action"] = np.where(
            sub["is_follow"].to_numpy(), f"FOLLOW_{TARGET}", f"FADE_{TARGET}"
        )
        entry.append(sub[
            ["candidate_id", "symbol", "day", "fold", "action", "traded"]
        ])
    entry = pd.concat(entry, ignore_index=True)
    print(f"[RL] test events={len(entry)}, traded={int(entry['traded'].sum())}",
          flush=True)

    # ---------- 事件 → 折（按交易日，严格先切分再展开轨迹）----------
    D = m2.load_base()
    folds, uniq = m2.build_folds(D["ev_day"], D["n_kept"])
    # 注意：同一交易日在不同折中角色不同，必须按折分别保存，不能压成一个 dict
    fold_train_days = {
        fi + 1: set(tr_d) for fi, (tr_d, se_d, te_d) in enumerate(folds)
    }

    # ---------- 特征列 ----------
    feat_cols = (
        [c for c in traj.columns if c.startswith("bg_")]
        + [c for c in traj.columns if c.startswith("dyn_")]
        + POS_NUM + ["crossed_session_break"]
    )
    # 所有非数值列（category / str / object）统一编码为整数码
    cat_cols = [
        c for c in feat_cols
        if not pd.api.types.is_numeric_dtype(traj[c])
    ]
    for c in cat_cols:
        traj[c] = traj[c].astype(str).astype("category").cat.codes.astype("int16")
    traj["crossed_session_break"] = traj["crossed_session_break"].astype("int8")
    print(f"[RL] features={len(feat_cols)} (categorical encoded={len(cat_cols)})",
          flush=True)

    results = []          # 每 (fold, 模型) 的测试段逐笔结果
    for fi in range(1, 5):
        # ---- 训练集：该折 train 段的 (事件 × {FOLLOW_2.5R, FADE_2.5R}) ----
        tr_days = fold_train_days[fi]
        tr_ep = traj[
            traj["trading_day"].isin(tr_days)
            & traj["initial_action"].isin([f"FOLLOW_{TARGET}", f"FADE_{TARGET}"])
        ].copy()
        print(f"[RL] F{fi}: train rows={len(tr_ep)}", flush=True)

        # NaN 填补：中位数只取自该折训练行（线性模型不接受 NaN）
        _Xtr = tr_ep[feat_cols].to_numpy(float)
        med = np.nanmedian(_Xtr, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)

        def Xof(g, med=med):
            X = g[feat_cols].to_numpy(float)
            return np.where(np.isnan(X), med, X)

        # 按 episode 建立 step → 行 的索引
        tr_ep = tr_ep.sort_values(["candidate_id", "initial_action", "step"])

        # 倒序拟合每一步的 continuation value
        models = {k: {} for k in MODELS}     # model_kind -> {step: model}
        for j in range(11, 0, -1):
            cur = tr_ep[tr_ep["step"] == j]
            cur = cur[cur["is_decision_point"] & ~cur["terminal_reason"].isin(FORCED)]
            if len(cur) < 50:
                print(f"[RL]   step {j}: skip (rows={len(cur)})", flush=True)
                continue
            print(f"[RL]   step {j}: rows={len(cur)}", flush=True)
            nxt = tr_ep[tr_ep["step"] == j + 1].set_index(
                ["candidate_id", "initial_action"]
            )
            cur = cur.set_index(["candidate_id", "initial_action"])

            # target：本步 HOLD 后的价值
            is_term = cur["terminal"].to_numpy()
            term_R = cur["all_hold_terminal_R"].to_numpy()
            aligned = nxt.reindex(cur.index)
            nx_exit = aligned["exit_value_R"].to_numpy()
            nx_forced = aligned["terminal_reason"].isin(FORCED).to_numpy()
            nx_termR = aligned["all_hold_terminal_R"].to_numpy()

            # target：本步 HOLD 后的价值（终值 或 V(next)）
            target = np.where(is_term, term_R, np.nan)
            for k in MODELS:
                v_next = np.full(len(cur), np.nan)
                has_m = (j + 1) in models[k]
                if has_m:
                    m_next = models[k][j + 1]
                    cn = m_next.predict(Xof(nxt.reindex(cur.index)))
                else:
                    cn = np.zeros(len(cur))
                v_next = np.where(nx_forced, nx_termR,
                                  np.maximum(nx_exit, cn))
                y = np.where(is_term, term_R, v_next)
                keep = np.isfinite(y)
                if keep.sum() < 50:
                    continue
                models[k][j] = fit_model(k, Xof(cur[keep]), y[keep])

        # ---- 测试段评估 ----
        ent_f = entry[entry["fold"] == fi]
        # 只对实际交易事件做轨迹模拟；未交易在机会集曲线中记 0
        tr_ent = ent_f[ent_f["traded"]]
        te_ep = traj.merge(
            tr_ent[["candidate_id", "action"]],
            on="candidate_id", how="inner",
        )
        te_ep = te_ep[te_ep["initial_action"] == te_ep["action"]]
        te_ep = te_ep.sort_values(["candidate_id", "step"])
        print(f"[RL] F{fi}: test episodes={te_ep['candidate_id'].nunique()}",
              flush=True)

        # 评估基准 = 该折【全部】测试事件（未交易记 0），保证与 M2/M3 同一
        # 每日机会集等权口径；只对实际交易事件做轨迹模拟。
        full = ent_f.reset_index(drop=True)
        idx_of = {c: i for i, c in enumerate(full["candidate_id"].to_numpy())}

        for k in MODELS:
            rl_v = np.zeros(len(full))
            fx_v = np.zeros(len(full))
            early_v = np.zeros(len(full), dtype=bool)
            held_v = np.full(len(full), -1)
            reason_v = np.array([""] * len(full), dtype=object)
            for cid, g in te_ep.groupby("candidate_id", sort=False):
                g = g.sort_values("step")
                fixed_R = float(g["all_hold_terminal_R"].iloc[0])
                tr = g[g["terminal"]]
                reason = (
                    str(tr["terminal_reason"].iloc[0]) if len(tr) else "TIMEOUT"
                )
                out, bars, is_early = fixed_R, int(g["bars_held"].max()), False
                for _, row in g.iterrows():
                    j = int(row["step"])
                    if j == 0 or not row["is_decision_point"]:
                        continue
                    if row["terminal_reason"] in FORCED:
                        out, bars, is_early = float(row["all_hold_terminal_R"]), j, False
                        break
                    mj = models[k].get(j)
                    if mj is not None:
                        xv = row[feat_cols].to_numpy(float)
                        xv = np.where(np.isnan(xv), med, xv).reshape(1, -1)
                        cv = float(mj.predict(xv)[0])
                    else:
                        cv = float(row["exit_value_R"]) + 1.0
                    if float(row["exit_value_R"]) > cv:
                        out, bars, is_early = float(row["exit_value_R"]), j, True
                        break
                    if row["terminal"]:
                        out, bars, is_early = float(row["all_hold_terminal_R"]), j, False
                        break
                    if j == 11:
                        out, bars, is_early = float(row["all_hold_terminal_R"]), j, False
                        break
                i = idx_of[cid]
                rl_v[i], fx_v[i] = out, fixed_R
                early_v[i], held_v[i], reason_v[i] = is_early, bars, reason
            # 硬断言：未交易事件在机会集曲线中必须记 0
            _nt = ~full["traded"].to_numpy()
            assert np.all(rl_v[_nt] == 0.0) and np.all(fx_v[_nt] == 0.0), (
                f"F{fi}/{k}: 未交易事件的收益不为 0（机会集口径被污染）"
            )
            results.append(dict(
                折=f"F{fi}", 模型=k,
                candidate_id=full["candidate_id"].to_numpy(),
                day=full["day"].to_numpy(),
                symbol=full["symbol"].to_numpy(),
                action=full["action"].to_numpy(),
                traded=full["traded"].to_numpy(),
                rl=rl_v, fixed=fx_v, early=early_v, held=held_v,
                reason=reason_v,
            ))
        print(f"[RL] F{fi} done", flush=True)

    # ---------------- 汇总（长表，跨折正确聚合）----------------
    long_res = pd.concat([
        pd.DataFrame(dict(
            折=r["折"], 模型=r["模型"], symbol=r["symbol"], day=r["day"],
            candidate_id=r["candidate_id"], action=r["action"],
            traded=r["traded"], rl=r["rl"], fixed=r["fixed"],
            early=r["early"], held=r["held"], reason=r["reason"],
        ))
        for r in results
    ], ignore_index=True)
    # 逐事件诊断（M5 用）：可再生，不入库
    long_res.to_parquet(OUT / "rl_episodes.parquet", index=False)

    def agg_by(keys):
        out = []
        for kk, g in long_res.groupby(keys):
            kk = kk if isinstance(kk, tuple) else (kk,)
            d = dict(zip(keys, kk))
            gt = g[g["traded"]]
            for tag, col in (("固定退出", "fixed"), ("动态退出", "rl")):
                p = perf(g[col].to_numpy(), g["day"].to_numpy(),
                         g["traded"].to_numpy())
                out.append(dict(
                    **d, 退出方式=tag, **p,
                    提前退出比例=(
                        round(float(gt["early"].mean()), 4)
                        if tag == "动态退出" else 0.0
                    ),
                    平均持有K线数=(
                        round(float(gt["held"].mean()), 4)
                        if tag == "动态退出" else None
                    ),
                ))
        return pd.DataFrame(out)

    overall = agg_by(["模型"])
    by_fold = agg_by(["折", "模型"])
    by_sym = agg_by(["symbol", "模型"]).rename(columns={"symbol": "品种"})

    # 净增量
    inc = []
    for k in MODELS:
        a = overall[(overall["模型"] == k) & (overall["退出方式"] == "固定退出")].iloc[0]
        b = overall[(overall["模型"] == k) & (overall["退出方式"] == "动态退出")].iloc[0]
        inc.append(dict(
            模型=k,
            固定退出夏普=a["夏普率"], 动态退出夏普=b["夏普率"],
            夏普差值=round(b["夏普率"] - a["夏普率"], 4),
            固定退出机会集累计=a["opportunity_curve_cumulative_R"],
            动态退出机会集累计=b["opportunity_curve_cumulative_R"],
            机会集累计差值=round(
                b["opportunity_curve_cumulative_R"]
                - a["opportunity_curve_cumulative_R"], 4),
            固定退出trade_total_R=a["trade_total_R"],
            动态退出trade_total_R=b["trade_total_R"],
            提前退出比例=b["提前退出比例"], 平均持有K线数=b["平均持有K线数"],
        ))
    inc = pd.DataFrame(inc)

    overall.to_csv(OUT / "rl_overall.csv", index=False, encoding="utf-8-sig")
    by_fold.to_csv(OUT / "rl_by_fold.csv", index=False, encoding="utf-8-sig")
    by_sym.to_csv(OUT / "rl_by_symbol.csv", index=False, encoding="utf-8-sig")
    inc.to_csv(OUT / "rl_increment.csv", index=False, encoding="utf-8-sig")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n=== 强化学习 vs 固定退出（整体）===")
    print(overall.round(6).to_string(index=False))
    print("\n=== 净增量 ===")
    print(inc.round(6).to_string(index=False))
    print("\n=== 分折 ===")
    print(by_fold.round(6).to_string(index=False))
    print("\n=== 分品种 ===")
    print(by_sym.round(6).to_string(index=False))
    print("\nRL_DONE")


if __name__ == "__main__":
    main()
