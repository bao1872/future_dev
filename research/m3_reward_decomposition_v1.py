"""M3: 收益来源拆解（LightGBM + SNAPSHOT / TEMPORAL）。

目标
----
M2 表明「LightGBM 回归 + TEMPORAL」是滚动向前验证下的当前冠军，
但不知道优势来自哪里。M3 把被选事件的最终收益精确拆成三层：

    全部测试事件六动作平均收益
      --[事件筛选增量]-->  被选事件六动作平均收益
      --[方向选择增量]-->  模型所选方向三目标平均收益
      --[目标档位增量]-->  模型最终选中动作实际收益

约束（严格遵守）
--------------
* 不训练任何新模型、不调参、不换覆盖率、不加特征。
  LightGBM 与 M2 使用完全相同的 train_reg / select_coverage / SEED。
* M2 只保存了聚合表，未保存逐事件预测。因此这里以**完全相同的确定性流程**
  重跑 4 折，仅为取回逐事件 yhat / cutoff / chosen action，
  属于复现而非重新拟合（结果与 M2 主表一致，脚本内做交叉校验）。

输出
----
research/analysis_results/m3/
    m3_core.csv                        四层分解主表（整体）
    m3_by_fold.csv                     每折分解
    m3_by_symbol.csv                   每品种分解
    m3_snapshot_vs_temporal_fold.csv   SNAPSHOT vs TEMPORAL 按折
    m3_snapshot_vs_temporal_symbol.csv SNAPSHOT vs TEMPORAL 按品种
    m3_events.parquet                  逐测试事件记录（审计用）
    m3_audit.json

用法：python -m research.m3_reward_decomposition_v1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.m2_nondeep_temporal_v1 import (
    SEED,
    _pf,
    build_feature_matrices,
    build_folds,
    build_temporal,
    day_mask,
    encode_for_fold,
    load_base,
    select_coverage,
    train_reg,
)
from research.train_rl_62d_v1 import curve_metrics

OUT = Path("research/analysis_results/m3")
OUT.mkdir(parents=True, exist_ok=True)

VIEWS = ("SNAPSHOT", "TEMPORAL")
MODEL_KIND = "lgbm"

# ACTION_NAMES[1:] = FOLLOW_1.5R / FOLLOW_2.0R / FOLLOW_2.5R / FADE_1.5R / ...
# 因此动作下标 <3 为 FOLLOW，>=3 为 FADE。


def collect_view(D, Xsnap, Xtemp, cat, folds, vname):
    """对指定视图跑 4 折，返回逐测试事件记录。"""
    R = D["R"][:, 1:]
    y_rows = D["y_rows"]
    ev_sym, ev_day = D["ev_sym"], D["ev_day"]
    recs = []
    for fi, (tr_days, se_days, te_days) in enumerate(folds):
        tr_ev = day_mask(ev_day, tr_days)
        se_ev = day_mask(ev_day, se_days)
        te_ev = day_mask(ev_day, te_days)
        tr_row = np.repeat(tr_ev, 6)
        se_row = np.repeat(se_ev, 6)
        te_row = np.repeat(te_ev, 6)

        Xsnap_num, Xtemp_num = encode_for_fold(Xsnap, Xtemp, cat, tr_row)
        Xn = Xsnap_num if vname == "SNAPSHOT" else Xtemp_num
        Xtr = Xn.iloc[tr_row].to_numpy(float)
        ytr = y_rows[tr_row]
        Xse = Xn.iloc[se_row].to_numpy(float)
        Xte = Xn.iloc[te_row].to_numpy(float)

        yhat_se = train_reg(MODEL_KIND, Xtr, ytr, Xse, cat, SEED)
        yhat_te = train_reg(MODEL_KIND, Xtr, ytr, Xte, cat, SEED)

        cutoff, cov = select_coverage(
            yhat_se, R[se_ev], ev_day[se_ev], se_ev
        )
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

        recs.append(
            pd.DataFrame(
                dict(
                    view=vname,
                    fold=fi + 1,
                    symbol=ev_sym[te_ev],
                    day=ev_day[te_ev],
                    candidate_id=D["keep_ids"][te_ev],
                    traded=traded,
                    chosen_action=chosen,
                    chosen_direction=np.where(is_follow, "FOLLOW", "FADE"),
                    six_mean=six_mean,
                    dir_mean=dir_mean,
                    chosen_r=chosen_r,
                    realized=np.where(traded, chosen_r, 0.0),
                )
            )
        )
        print(
            f"[M3] {vname} F{fi+1}: cutoff={cutoff:.6f} cov={cov} "
            f"test={n_te} traded={int(traded.sum())}",
            flush=True,
        )
    return pd.concat(recs, ignore_index=True)


def decompose(g, label=None, label_name=None):
    """三层分解。g 为某范围内的全部测试事件（含未交易事件）。"""
    all_six = float(g["six_mean"].mean())
    sel = g[g["traded"]]
    n_sel = len(sel)
    if n_sel == 0:
        return None
    sel_six = float(sel["six_mean"].mean())
    dir_mean = float(sel["dir_mean"].mean())
    chosen = float(sel["chosen_r"].mean())

    scr = sel_six - all_six
    dirc = dir_mean - sel_six
    tgc = chosen - dir_mean
    fin = chosen - all_six
    row = dict(
        所有事件六动作平均=all_six,
        被选事件六动作平均=sel_six,
        模型所选方向平均=dir_mean,
        模型最终动作=chosen,
        事件筛选增量=scr,
        方向选择增量=dirc,
        目标档位增量=tgc,
        最终增量=fin,
        全部事件数=int(len(g)),
        被选事件数=int(n_sel),
    )
    if fin != 0:
        row["事件筛选占比"] = scr / fin
        row["方向选择占比"] = dirc / fin
        row["目标档位占比"] = tgc / fin
    else:
        row["事件筛选占比"] = np.nan
        row["方向选择占比"] = np.nan
        row["目标档位占比"] = np.nan
    if label_name is not None:
        row = {label_name: label, **row}
    return row


def perf(g):
    """该范围内的滚动测试表现（与 M2 口径一致：按日等权平均后算曲线）。"""
    realized = g["realized"].to_numpy(float)
    daily = (
        pd.Series(realized, index=g["day"].to_numpy())
        .groupby(level=0)
        .mean()
        .sort_index()
    )
    cm = curve_metrics(daily)
    nz = realized[realized != 0]
    return dict(
        累计收益=float(cm["累计收益"]),
        夏普率=float(cm["夏普率"]),
        最大回撤=float(cm["最大回撤"]),
        利润因子=float(_pf(nz)),
    )


def rnd(d, n=6):
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


def main():
    D = load_base()
    n_kept = D["n_kept"]
    temporal = build_temporal(D["ev_sym"], D["ev_bar"])
    Xsnap, Xtemp, cat = build_feature_matrices(D, temporal)
    folds, uniq = build_folds(D["ev_day"], n_kept)

    frames = {}
    for v in VIEWS:
        frames[v] = collect_view(D, Xsnap, Xtemp, cat, folds, v)

    all_ev = pd.concat([frames[v] for v in VIEWS], ignore_index=True)
    all_ev.to_parquet(OUT / "m3_events.parquet", index=False)

    # ---------------- 整体四层分解表 ----------------
    core_rows = []
    for v in VIEWS:
        d = decompose(frames[v], label=v, label_name="特征视图")
        core_rows.append(d)
    core = pd.DataFrame(core_rows)
    core.to_csv(OUT / "m3_core.csv", index=False, encoding="utf-8-sig")

    # 四层纵向表（层级 / 平均R / 相对上一层增量）
    level_rows = []
    for v in VIEWS:
        d = core[core["特征视图"] == v].iloc[0]
        level_rows.append(
            dict(特征视图=v, 层级="所有事件六动作平均",
                 平均R=d["所有事件六动作平均"], 相对上一层增量=np.nan)
        )
        level_rows.append(
            dict(特征视图=v, 层级="被选事件六动作平均",
                 平均R=d["被选事件六动作平均"],
                 相对上一层增量=d["事件筛选增量"])
        )
        level_rows.append(
            dict(特征视图=v, 层级="模型所选方向平均",
                 平均R=d["模型所选方向平均"],
                 相对上一层增量=d["方向选择增量"])
        )
        level_rows.append(
            dict(特征视图=v, 层级="模型最终动作",
                 平均R=d["模型最终动作"],
                 相对上一层增量=d["目标档位增量"])
        )
    lvl = pd.DataFrame(level_rows)
    lvl.to_csv(OUT / "m3_core_levels.csv", index=False, encoding="utf-8-sig")

    # ---------------- 每折 / 每品种分解 ----------------
    by_fold, by_sym = [], []
    for v in VIEWS:
        g = frames[v]
        for fi, gg in g.groupby("fold"):
            d = decompose(gg, label=fi, label_name="折")
            if d:
                d["特征视图"] = v
                d.update(perf(gg))
                by_fold.append(d)
        for s, gg in g.groupby("symbol"):
            d = decompose(gg, label=s, label_name="品种")
            if d:
                d["特征视图"] = v
                d.update(perf(gg))
                by_sym.append(d)
    by_fold = pd.DataFrame(by_fold)
    by_sym = pd.DataFrame(by_sym)
    by_fold.to_csv(OUT / "m3_by_fold.csv", index=False, encoding="utf-8-sig")
    by_sym.to_csv(OUT / "m3_by_symbol.csv", index=False, encoding="utf-8-sig")

    # ---------------- SNAPSHOT vs TEMPORAL ----------------
    def compare(col, label, frames_dict, keys):
        rows = []
        for k in keys:
            row = {label: k}
            for v in VIEWS:
                g = frames_dict[v]
                gg = g[g[col] == k]
                if len(gg) == 0:
                    continue
                p = perf(gg)
                d = decompose(gg)
                row[f"{v}_累计收益"] = p["累计收益"]
                row[f"{v}_夏普率"] = p["夏普率"]
                row[f"{v}_利润因子"] = p["利润因子"]
                row[f"{v}_被选事件平均收益"] = (
                    float(gg[gg["traded"]]["chosen_r"].mean())
                    if gg["traded"].any()
                    else np.nan
                )
                row[f"{v}_事件筛选增量"] = d["事件筛选增量"]
                row[f"{v}_方向选择增量"] = d["方向选择增量"]
                row[f"{v}_目标档位增量"] = d["目标档位增量"]
            row["夏普率差值(TEMPORAL-SNAPSHOT)"] = (
                row.get("TEMPORAL_夏普率", np.nan)
                - row.get("SNAPSHOT_夏普率", np.nan)
            )
            rows.append(row)
        return pd.DataFrame(rows)

    fold_cmp = compare(
        "fold", "折", frames, sorted(frames["SNAPSHOT"]["fold"].unique())
    )
    sym_cmp = compare(
        "symbol", "品种", frames, sorted(frames["SNAPSHOT"]["symbol"].unique())
    )
    fold_cmp.to_csv(
        OUT / "m3_snapshot_vs_temporal_fold.csv", index=False,
        encoding="utf-8-sig",
    )
    sym_cmp.to_csv(
        OUT / "m3_snapshot_vs_temporal_symbol.csv", index=False,
        encoding="utf-8-sig",
    )

    # ---------------- 交叉校验：与 M2 主表一致性 ----------------
    m2 = pd.read_csv(
        "research/analysis_results/rl_62d_m2/m2_main.csv"
    )
    checks = []
    for v in VIEWS:
        g = frames[v]
        p = perf(g)
        m = m2[(m2["模型"] == "LightGBM回归") & (m2["特征视图"] == v)]
        if len(m):
            checks.append(
                dict(
                    视图=v,
                    M3_夏普率=round(p["夏普率"], 4),
                    M2_夏普率=float(m["夏普率"].iloc[0]),
                    M3_累计收益=round(p["累计收益"], 4),
                    M2_累计收益=float(m["累计收益"].iloc[0]),
                    M3_交易次数=int((g["realized"] != 0).sum()),
                    M2_交易次数=int(m["交易次数"].iloc[0]),
                )
            )
    checks = pd.DataFrame(checks)
    checks.to_csv(
        OUT / "m3_crosscheck_m2.csv", index=False, encoding="utf-8-sig"
    )

    audit = dict(
        script="research/m3_reward_decomposition_v1.py",
        model="LightGBM回归",
        views=list(VIEWS),
        seed=SEED,
        n_kept=int(n_kept),
        n_test_events={v: int(len(frames[v])) for v in VIEWS},
        n_traded={v: int(frames[v]["traded"].sum()) for v in VIEWS},
        note=(
            "M2 未保存逐事件预测；本脚本以相同确定性流程重跑 4 折取回 "
            "逐事件 yhat/cutoff/chosen，未改模型、参数、覆盖率或特征。"
        ),
    )
    (OUT / "m3_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    print("\n=== M3 四层分解（整体）===")
    print(lvl.round(6).to_string(index=False))
    print("\n=== M3 三层贡献（整体）===")
    print(core.round(6).to_string(index=False))
    print("\n=== 每折 ===")
    print(by_fold.round(6).to_string(index=False))
    print("\n=== 每品种 ===")
    print(by_sym.round(6).to_string(index=False))
    print("\n=== SNAPSHOT vs TEMPORAL 按折 ===")
    print(fold_cmp.round(6).to_string(index=False))
    print("\n=== SNAPSHOT vs TEMPORAL 按品种 ===")
    print(sym_cmp.round(6).to_string(index=False))
    print("\n=== 与 M2 主表交叉校验 ===")
    print(checks.to_string(index=False))
    print("\nM3_DONE")


if __name__ == "__main__":
    main()
