"""第五部分：方向 × 目标档位简化实验。

动机
----
M3 快照侧（冻结、可信）显示：
    事件筛选 +0.0166R（正但小）
    方向选择 +0.0712R（主要正贡献，约 96%）
    目标档位 −0.0135R（负贡献）

因此假设：六动作模型设计过复杂，应降为「模型只选方向 + 目标档位固定」。

设计
----
不训练任何新模型，直接复用 LightGBM 回归的六动作预测：

    follow_score = mean(pred FOLLOW_1.5R, FOLLOW_2.0R, FOLLOW_2.5R)
    fade_score   = mean(pred FADE_1.5R,   FADE_2.0R,   FADE_2.5R)
    direction    = FOLLOW if follow_score >= fade_score else FADE
    event_score  = max(follow_score, fade_score)

比较策略：
    原模型   六动作直接 argmax
    简化A    模型方向 + 固定 1.5R
    简化B    模型方向 + 固定 2.0R
    简化C    模型方向 + 固定 2.5R
    简化D    模型方向 + 每折选择段从 {1.5,2.0,2.5} 选一个，测试段冻结

覆盖率：沿用 M2 规则 —— 每折在模型选择段按「每日等权夏普率」
从 COVERAGES=[0.10,0.20,0.30,0.50] 中选最佳，测试段冻结。不新增覆盖率。

用法：python -m research.m3_direction_target_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.rl_62d_core_v1 import ACTION_NAMES
from research.train_rl_62d_v1 import curve_metrics

OUT = Path("research/analysis_results/m3_direction_target")
OUT.mkdir(parents=True, exist_ok=True)

EVENTS = Path(
    "research/analysis_results/m2_temporal_fix/m2fix_events.parquet"
)

# 最终入场规则：修正后时间增强不稳定，按「稳定性优先」采用 LightGBM 快照
ENTRY_MODEL = "LightGBM回归"
ENTRY_VIEW = "SNAPSHOT"

A6 = list(ACTION_NAMES[1:])          # 6 个可交易动作
FOLLOW_COLS = [a for a in A6 if a.startswith("FOLLOW")]
FADE_COLS = [a for a in A6 if a.startswith("FADE")]
TARGETS = {"1.5R": 0, "2.0R": 1, "2.5R": 2}   # 方向内偏移


def perf(realized, day, traded):
    """统一口径：夏普/回撤用每日机会集等权曲线；交易次数按是否执行统计。"""
    daily = (
        pd.Series(realized, index=day).groupby(level=0).mean().sort_index()
    )
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


def pick_coverage(event_score, r_if_traded, day):
    """选择段选最佳覆盖率（每日等权夏普率），返回 (cutoff, coverage, sharpe)。"""
    best = None
    for c in m2.COVERAGES:
        cutoff = np.quantile(event_score, 1 - c)
        traded = event_score > cutoff
        if traded.sum() == 0:
            continue
        realized = np.where(traded, r_if_traded, 0.0)
        daily = pd.Series(realized, index=day).groupby(level=0).mean()
        sh = curve_metrics(daily)["夏普率"]
        if best is None or sh > best[0]:
            best = (sh, float(cutoff), c)
    if best is None:
        return None
    return best[1], best[2], best[0]


def main():
    ev = pd.read_parquet(EVENTS)
    ev = ev[(ev["模型"] == ENTRY_MODEL) & (ev["view"] == ENTRY_VIEW)].copy()
    # 位置索引对齐：后续用 .index 作为 rr / yh 的行下标
    ev = ev.reset_index(drop=True)
    print(f"[DT] entry={ENTRY_MODEL}/{ENTRY_VIEW} rows={len(ev)}", flush=True)

    yh = ev[[f"yhat_{a}" for a in A6]].to_numpy(float)
    rr = ev[[f"R_{a}" for a in A6]].to_numpy(float)
    ev["follow_score"] = yh[:, [A6.index(c) for c in FOLLOW_COLS]].mean(axis=1)
    ev["fade_score"] = yh[:, [A6.index(c) for c in FADE_COLS]].mean(axis=1)
    ev["six_argmax"] = yh.argmax(axis=1)
    ev["six_max"] = yh.max(axis=1)

    follow_mean_R = rr[:, [A6.index(c) for c in FOLLOW_COLS]].mean(axis=1)
    fade_mean_R = rr[:, [A6.index(c) for c in FADE_COLS]].mean(axis=1)
    is_follow = ev["follow_score"].to_numpy() >= ev["fade_score"].to_numpy()
    ev["direction"] = np.where(is_follow, "FOLLOW", "FADE")
    ev["event_score"] = np.where(
        is_follow, ev["follow_score"], ev["fade_score"]
    )
    # 方向内偏移：FOLLOW→0, FADE→3
    dir_off = np.where(is_follow, 0, 3)

    recs = []
    audit_folds = []

    for f in sorted(ev["fold"].unique()):
        sel = ev[ev["segment"] == "sel"]
        sel = sel[sel["fold"] == f]
        te = ev[(ev["segment"] == "test") & (ev["fold"] == f)]

        # ---------- 原模型（六动作 argmax）----------
        r_sel = rr[sel.index.to_numpy()][
            np.arange(len(sel)), sel["six_argmax"].to_numpy()
        ]
        cutoff6, cov6, sh6 = pick_coverage(
            sel["six_max"].to_numpy(), r_sel, sel["day"].to_numpy()
        )
        r_te = rr[te.index.to_numpy()][
            np.arange(len(te)), te["six_argmax"].to_numpy()
        ]
        traded6 = te["six_max"].to_numpy() > cutoff6
        realized6 = np.where(traded6, r_te, 0.0)
        recs.append(dict(
            策略="原模型", 方向="六动作直选", 目标="六动作直选", 折=f,
            day=te["day"].to_numpy(), symbol=te["symbol"].to_numpy(),
            traded=traded6, realized=realized6,
        ))

        # ---------- 简化 A/B/C/D ----------
        # 先为每个固定目标在选择段选出最佳覆盖率
        per_target = {}
        for tname, off in TARGETS.items():
            act_sel = dir_off[sel.index.to_numpy()] + off
            r_t_sel = rr[sel.index.to_numpy()][np.arange(len(sel)), act_sel]
            per_target[tname] = pick_coverage(
                sel["event_score"].to_numpy(), r_t_sel, sel["day"].to_numpy()
            )

        # D：选目标（每个目标用其自身最佳覆盖率对应的选择段夏普）
        d_choice = max(
            ((t, per_target[t][2]) for t in TARGETS if per_target[t]),
            key=lambda x: x[1],
        )
        d_target = d_choice[0]

        plans = [
            ("简化A", "1.5R"), ("简化B", "2.0R"), ("简化C", "2.5R"),
            ("简化D", d_target),
        ]
        for sname, tname in plans:
            off = TARGETS[tname]
            cutoff_t, cov_t = per_target[tname][0], per_target[tname][1]
            act_te = dir_off[te.index.to_numpy()] + off
            r_te_t = rr[te.index.to_numpy()][np.arange(len(te)), act_te]
            traded = te["event_score"].to_numpy() > cutoff_t
            realized = np.where(traded, r_te_t, 0.0)
            recs.append(dict(
                策略=sname, 方向="模型方向", 目标=tname, 折=f,
                day=te["day"].to_numpy(), symbol=te["symbol"].to_numpy(),
                traded=traded, realized=realized,
            ))

        audit_folds.append(dict(
            折=int(f),
            原模型_覆盖率=cov6,
            简化A_覆盖率=per_target["1.5R"][1],
            简化B_覆盖率=per_target["2.0R"][1],
            简化C_覆盖率=per_target["2.5R"][1],
            简化D_选中目标=d_target,
            简化D_覆盖率=per_target[d_target][1],
        ))
        print(f"[DT] F{f}: 原模型 cov={cov6}, D 选目标={d_target}", flush=True)

    # ---------------- 汇总 ----------------
    long = []
    for r in recs:
        long.append(pd.DataFrame(dict(
            策略=r["策略"], 方向=r["方向"], 目标=r["目标"], 折=r["折"],
            day=r["day"], symbol=r["symbol"],
            traded=r["traded"], realized=r["realized"],
        )))
    long = pd.concat(long, ignore_index=True)
    long.to_parquet(OUT / "m3dt_events.parquet", index=False)

    main_rows, fold_rows, sym_rows = [], [], []
    for (sname,), g in long.groupby(["策略"]):
        p = perf(g["realized"].to_numpy(), g["day"].to_numpy(),
                 g["traded"].to_numpy())
        main_rows.append(dict(
            策略=sname,
            方向=g["方向"].iloc[0], 目标=g["目标"].iloc[0], **p,
            被选事件平均收益=(
                round(float(g[g["traded"]]["realized"].mean()), 6)
                if g["traded"].any() else None
            ),
        ))
        for f, gg in g.groupby("折"):
            fold_rows.append(dict(策略=sname, 折=f"F{f}", **perf(
                gg["realized"].to_numpy(), gg["day"].to_numpy(),
                gg["traded"].to_numpy())))
        for s, gg in g.groupby("symbol"):
            sym_rows.append(dict(策略=sname, 品种=s, **perf(
                gg["realized"].to_numpy(), gg["day"].to_numpy(),
                gg["traded"].to_numpy())))

    main_df = pd.DataFrame(main_rows)
    fold_df = pd.DataFrame(fold_rows)
    sym_df = pd.DataFrame(sym_rows)
    main_df.to_csv(OUT / "m3dt_main.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUT / "m3dt_by_fold.csv", index=False, encoding="utf-8-sig")
    sym_df.to_csv(OUT / "m3dt_by_symbol.csv", index=False, encoding="utf-8-sig")

    (OUT / "m3dt_audit.json").write_text(json.dumps(dict(
        script="research/m3_direction_target_v1.py",
        entry_model=ENTRY_MODEL, entry_view=ENTRY_VIEW,
        coverages=m2.COVERAGES,
        fold_plan=audit_folds,
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n=== 方向 × 目标档位（整体）===")
    print(main_df.round(6).to_string(index=False))
    print("\n=== 分折 ===")
    print(fold_df.round(6).to_string(index=False))
    print("\n=== 分品种 ===")
    print(sym_df.round(6).to_string(index=False))
    print("\nM3DT_DONE")


if __name__ == "__main__":
    main()
