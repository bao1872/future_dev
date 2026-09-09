"""时间增强广播修复后的重跑（只重跑被污染部分）。

裁决依据
--------
* M2 时间增强特征存在「事件级 → 动作级」位置错位，所有 TEMPORAL 结果作废。
* 快照（SNAPSHOT）结果未受污染，保持冻结，本脚本不重新评估快照。
* 本脚本只重跑 TEMPORAL 四个回归模型；
  额外跑一次 LightGBM 快照，仅为第五部分「方向 × 目标档位」实验提供
  六动作预测（确定性复现，不改动也不覆盖冻结的 M2 快照评估指标）。

所有参数 / 折 / 覆盖率完全沿用 M2，不得调整。

输出
----
research/analysis_results/m2_temporal_fix/
    m2fix_main.csv            主表（含交易次数/最大回撤/利润因子）
    m2fix_by_fold.csv         4 个滚动折
    m2fix_by_symbol.csv       AG/CU/RB/M
    m2fix_compare.csv         快照夏普(冻结) vs 修正后时间增强夏普
    m2fix_m3_levels.csv       M3 四层分解（修正后时间增强）
    m2fix_m3_contrib.csv      M3 三层贡献
    m2fix_direction.csv       经济方向增量 + 事后最佳方向命中率
    m2fix_events.parquet      逐事件六动作预测与真实收益（选择段+测试段）
    m2fix_audit.json

用法：python -m research.m2_temporal_rerun_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.rl_62d_core_v1 import ACTION_NAMES
from research.train_rl_62d_v1 import curve_metrics

OUT = Path("research/analysis_results/m2_temporal_fix")
OUT.mkdir(parents=True, exist_ok=True)

FROZEN_M2 = Path("research/analysis_results/rl_62d_m2/m2_main.csv")
FROZEN_M3_LEVELS = Path("research/analysis_results/m3/m3_core_levels.csv")

RUNS = [
    ("CatBoost回归", "catboost", "TEMPORAL"),
    ("LightGBM回归", "lgbm", "TEMPORAL"),
    ("XGBoost回归", "xgb", "TEMPORAL"),
    ("Ridge回归", "ridge", "TEMPORAL"),
    # 仅为方向×目标简化实验提供预测；不改动冻结的 M2 快照评估
    ("LightGBM回归", "lgbm", "SNAPSHOT"),
]

TARGET_NAMES = ("1.5R", "2.0R", "2.5R")


def perf(g):
    """滚动测试表现（每日机会集等权曲线）。"""
    realized = g["realized"].to_numpy(float)
    traded = g["traded"].to_numpy(bool)
    daily = (
        pd.Series(realized, index=g["day"].to_numpy())
        .groupby(level=0).mean().sort_index()
    )
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


def decompose(g):
    """M3 三层分解。"""
    all_six = float(g["six_mean"].mean())
    sel = g[g["traded"]]
    if len(sel) == 0:
        return None
    sel_six = float(sel["six_mean"].mean())
    dir_mean = float(sel["dir_mean"].mean())
    chosen = float(sel["chosen_r"].mean())
    scr, dirc = sel_six - all_six, dir_mean - sel_six
    tgc, fin = chosen - dir_mean, chosen - all_six
    return dict(
        所有事件六动作平均=all_six, 被选事件六动作平均=sel_six,
        模型所选方向平均=dir_mean, 模型最终动作=chosen,
        事件筛选增量=scr, 方向选择增量=dirc, 目标档位增量=tgc,
        最终增量=fin,
        事件筛选占比=(scr / fin if fin != 0 else np.nan),
        方向选择占比=(dirc / fin if fin != 0 else np.nan),
        目标档位占比=(tgc / fin if fin != 0 else np.nan),
        全部事件数=int(len(g)), 被选事件数=int(len(sel)),
    )


def direction_metrics(g):
    """第六部分：经济方向增量 与 事后最佳方向命中率，并列报告。"""
    sel = g[g["traded"]]
    if len(sel) == 0:
        return None
    econ = float((sel["dir_mean"] - sel["six_mean"]).mean())
    best_dir = np.where(
        sel["follow_mean"].to_numpy() >= sel["fade_mean"].to_numpy(),
        "FOLLOW", "FADE",
    )
    hit = float((sel["chosen_direction"].to_numpy() == best_dir).mean())
    return dict(
        经济方向增量=econ,
        事后最佳方向命中率=hit,
        被选事件数=int(len(sel)),
    )


def main():
    D = m2.load_base()
    n_kept = D["n_kept"]
    R = D["R"][:, 1:]
    y_rows = D["y_rows"]
    ev_sym, ev_day = D["ev_sym"], D["ev_day"]
    keep_ids = D["keep_ids"]

    temporal = m2.build_temporal(ev_sym, D["ev_bar"])
    Xsnap, Xtemp, cat = m2.build_feature_matrices(D, temporal)
    folds, uniq = m2.build_folds(ev_day, n_kept)

    all_recs, test_recs = [], []

    for mname, mkind, vname in RUNS:
        Xn = Xsnap if vname == "SNAPSHOT" else Xtemp
        for fi, (tr_days, se_days, te_days) in enumerate(folds):
            tr_ev = m2.day_mask(ev_day, tr_days)
            se_ev = m2.day_mask(ev_day, se_days)
            te_ev = m2.day_mask(ev_day, te_days)
            tr_row = np.repeat(tr_ev, 6)
            se_row = np.repeat(se_ev, 6)
            te_row = np.repeat(te_ev, 6)

            Xsnap_num, Xtemp_num = m2.encode_for_fold(Xsnap, Xtemp, cat, tr_row)
            Xcur = Xsnap_num if vname == "SNAPSHOT" else Xtemp_num
            Xtr = Xcur.iloc[tr_row].to_numpy(float)
            ytr = y_rows[tr_row]

            yhat_se = m2.train_reg(
                mkind, Xtr, ytr, Xcur.iloc[se_row].to_numpy(float), cat, m2.SEED
            )
            yhat_te = m2.train_reg(
                mkind, Xtr, ytr, Xcur.iloc[te_row].to_numpy(float), cat, m2.SEED
            )
            cutoff, cov = m2.select_coverage(
                yhat_se, R[se_ev], ev_day[se_ev], se_ev
            )

            # ---- 选择段 / 测试段逐事件记录（含六动作预测与真实收益）----
            for seg, mask, yh in (
                ("sel", se_ev, yhat_se), ("test", te_ev, yhat_te)
            ):
                ye = yh.reshape(-1, 6)
                Rm = R[mask]
                rec = pd.DataFrame(
                    ye, columns=[f"yhat_{a}" for a in ACTION_NAMES[1:]]
                )
                for k in range(6):
                    rec[f"R_{ACTION_NAMES[1+k]}"] = Rm[:, k]
                rec.insert(0, "segment", seg)
                rec.insert(0, "fold", fi + 1)
                rec.insert(0, "view", vname)
                rec.insert(0, "模型", mname)
                rec.insert(0, "day", ev_day[mask])
                rec.insert(0, "symbol", ev_sym[mask])
                rec.insert(0, "candidate_id", keep_ids[mask])
                all_recs.append(rec)

            # ---- 测试段评估（M2 规则：覆盖率在每折选择段冻结）----
            ye = yhat_te.reshape(-1, 6)
            ev_score = ye.max(axis=1)
            chosen = ye.argmax(axis=1)
            traded = ev_score > cutoff
            Rte = R[te_ev]
            n_te = int(te_ev.sum())
            six_mean = Rte.mean(axis=1)
            is_follow = chosen < 3
            dir_mean = np.where(
                is_follow, Rte[:, :3].mean(axis=1), Rte[:, 3:].mean(axis=1)
            )
            chosen_r = Rte[np.arange(n_te), chosen]
            t = pd.DataFrame(dict(
                模型=mname, 特征视图=vname, fold=fi + 1,
                symbol=ev_sym[te_ev], day=ev_day[te_ev],
                candidate_id=keep_ids[te_ev],
                traded=traded, chosen_action=chosen,
                chosen_direction=np.where(is_follow, "FOLLOW", "FADE"),
                six_mean=six_mean, dir_mean=dir_mean,
                follow_mean=Rte[:, :3].mean(axis=1),
                fade_mean=Rte[:, 3:].mean(axis=1),
                chosen_r=chosen_r,
                realized=np.where(traded, chosen_r, 0.0),
            ))
            test_recs.append(t)
            print(f"[rerun] {mname}/{vname} F{fi+1}: cov={cov} "
                  f"test={n_te} traded={int(traded.sum())}", flush=True)

    events = pd.concat(all_recs, ignore_index=True)
    events.to_parquet(OUT / "m2fix_events.parquet", index=False)
    tests = pd.concat(test_recs, ignore_index=True)
    tests.to_parquet(OUT / "m2fix_test_events.parquet", index=False)

    # ---------------- 主表 / 分折 / 分品种 ----------------
    main_rows, fold_rows, sym_rows = [], [], []
    m3_levels, m3_contrib, dir_rows = [], [], []

    for (mname, vname), g in tests.groupby(["模型", "特征视图"]):
        p = perf(g)
        main_rows.append(dict(
            模型=mname, 特征视图=vname, **p,
            不交易比例=round(float(1 - g["traded"].mean()), 4),
            选中平均收益=(
                round(float(g[g["traded"]]["chosen_r"].mean()), 6)
                if g["traded"].any() else None
            ),
        ))
        for fi, gg in g.groupby("fold"):
            fold_rows.append(dict(模型=mname, 特征视图=vname, 折=f"F{fi}", **perf(gg)))
        for s, gg in g.groupby("symbol"):
            sym_rows.append(dict(模型=mname, 特征视图=vname, 品种=s, **perf(gg)))

        d = decompose(g)
        if d:
            m3_contrib.append(dict(模型=mname, 特征视图=vname, **d))
            for lvl, val, inc in (
                ("所有事件六动作平均", d["所有事件六动作平均"], np.nan),
                ("被选事件六动作平均", d["被选事件六动作平均"], d["事件筛选增量"]),
                ("模型所选方向平均", d["模型所选方向平均"], d["方向选择增量"]),
                ("模型最终动作", d["模型最终动作"], d["目标档位增量"]),
            ):
                m3_levels.append(dict(
                    模型=mname, 特征视图=vname, 层级=lvl,
                    平均R=val, 相对上一层增量=inc,
                ))
        dm = direction_metrics(g)
        if dm:
            dir_rows.append(dict(模型=mname, 特征视图=vname, **dm))

    main_df = pd.DataFrame(main_rows)
    fold_df = pd.DataFrame(fold_rows)
    sym_df = pd.DataFrame(sym_rows)
    main_df.to_csv(OUT / "m2fix_main.csv", index=False, encoding="utf-8-sig")
    fold_df.to_csv(OUT / "m2fix_by_fold.csv", index=False, encoding="utf-8-sig")
    sym_df.to_csv(OUT / "m2fix_by_symbol.csv", index=False, encoding="utf-8-sig")

    # ---------------- 快照(冻结) vs 修正后时间增强 ----------------
    frozen = pd.read_csv(FROZEN_M2)
    snap_sharpe = {
        r["模型"]: r["夏普率"]
        for _, r in frozen[frozen["特征视图"] == "SNAPSHOT"].iterrows()
    }
    cmp_rows = []
    for _, r in main_df[main_df["特征视图"] == "TEMPORAL"].iterrows():
        s = snap_sharpe.get(r["模型"])
        cmp_rows.append(dict(
            模型=r["模型"], 快照夏普=round(float(s), 4),
            修正后时间增强夏普=round(float(r["夏普率"]), 4),
            差值=round(float(r["夏普率"]) - float(s), 4),
        ))
    cmp_df = pd.DataFrame(cmp_rows)
    cmp_df.to_csv(OUT / "m2fix_compare.csv", index=False, encoding="utf-8-sig")

    pd.DataFrame(m3_levels).to_csv(
        OUT / "m2fix_m3_levels.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(m3_contrib).to_csv(
        OUT / "m2fix_m3_contrib.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(dir_rows).to_csv(
        OUT / "m2fix_direction.csv", index=False, encoding="utf-8-sig")

    audit = dict(
        script="research/m2_temporal_rerun_v1.py",
        fix=(
            "时间增强特征由 pd.concat 位置对齐改为 candidate_id "
            "many-to-one merge，并加入硬断言"
        ),
        runs=[{"模型": m, "view": v} for m, _, v in RUNS],
        seed=m2.SEED,
        coverages=m2.COVERAGES,
        n_kept=int(n_kept),
        note="快照结果沿用冻结的 M2，未重新评估。",
    )
    (OUT / "m2fix_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n=== 主表（修正后）===")
    print(main_df.round(6).to_string(index=False))
    print("\n=== 快照 vs 修正后时间增强 ===")
    print(cmp_df.to_string(index=False))
    print("\n=== M3 四层（修正后）===")
    print(pd.DataFrame(m3_levels).round(6).to_string(index=False))
    print("\n=== M3 三层贡献（修正后）===")
    print(pd.DataFrame(m3_contrib).round(6).to_string(index=False))
    print("\n=== 方向能力（经济增量 vs 事后命中率）===")
    print(pd.DataFrame(dir_rows).round(6).to_string(index=False))
    print("\nRERUN_DONE")


if __name__ == "__main__":
    main()
