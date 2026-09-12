"""P12.5 -- Matched Waiting Consistency Gate  (Stage 4A 后处理, 只读汇总)

目的（reviewer 规格）：
  用 *现有* Stage 4A 的 matched comparison 原始行，做一个低成本完整汇总，
  回答一个唯一问题：

    「等待后 RR↑、target-first↓、E[R]↓」这个 trade-off，
    是 3 个 WF / 不同 later horizon / OUTWARD・INWARD / 不同 scale
    都大体成立的广泛机制，还是只存在于 h=2→3 的少数 cell？

措辞合同（HARD）：
  只能称 "matched waiting effect"。
  禁止写 "causal waiting effect" —— 能在 later horizon 继续 available 的
  contact 本身仍可能有选择性（selection）。

硬边界：
  不 bootstrap；不调 threshold；不改 Stage 4A geometry / stop / target /
  horizon / scale 定义；不做任何模型。
  本脚本直接复用 run_liquidity_field_action_surface_v1 的权威函数
  （compute_action_surface / value_of_waiting_first_available /
  value_of_waiting_h2_cohort），保证与 Stage 4A 逐字一致，不重实现几何。

输出目录：research/analysis_results/liquidity_field_action_surface_v1/
  p12_5_matched_waiting_cells.csv
  p12_5_direction_consistency.csv
  P12_5_MATCHED_WAITING_CONSISTENCY.md
  P12_5_MATCHED_WAITING_AUDIT.json
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.liquidity_oracle_atlas import run_liquidity_field_action_surface_v1 as s4a

OUT = Path("research/analysis_results/liquidity_field_action_surface_v1")
OUT.mkdir(parents=True, exist_ok=True)

# 三个方向一致性 share（>0 视为正）；target_first / E_R_lower 的「下降」即 share < 0.5
SHARE_SPEC = [
    ("share_delta_RR_positive", "delta_RR"),
    ("share_delta_target_first_positive", "delta_target_first"),
    ("share_delta_E_R_lower_positive", "delta_E_R_lower"),
]

VARIANTS = {
    "first_available": s4a.value_of_waiting_first_available,
    "h2_cohort": s4a.value_of_waiting_h2_cohort,
}


def _prepare_raw(raw: pd.DataFrame, variant: str) -> pd.DataFrame:
    """统一 raw matched 行：加 waiting_distance，并对齐列名。"""
    r = raw.copy()
    r["waiting_distance"] = r["later_h"].astype(int) - r["base_h"].astype(int)
    r["variant"] = variant
    return r


def _cells(r: pd.DataFrame) -> pd.DataFrame:
    """每个 (variant, wf, action, structure_scale, base_h, later_h) 一行。"""
    g = r.groupby(
        ["variant", "wf", "action", "structure_scale", "base_h", "later_h",
         "waiting_distance"], sort=False)
    out = g.agg(
        matched_n=("delta_RR", "size"),
        median_delta_RR=("delta_RR", "median"),
        mean_delta_target_first=("delta_target_first", "mean"),
        median_delta_E_R_lower=("delta_E_R_lower", "median"),
        median_delta_E_R_upper=("delta_E_R_upper", "median"),
        median_delta_target_distance_atr=("delta_target_distance_atr", "median"),
        median_delta_risk_distance_atr=("delta_risk_distance_atr", "median"),
        share_delta_RR_positive=("delta_RR", lambda s: float((s > 0).mean())),
        share_delta_target_first_positive=(
            "delta_target_first", lambda s: float((s > 0).mean())),
        share_delta_E_R_lower_positive=(
            "delta_E_R_lower", lambda s: float((s > 0).mean())),
    )
    return out.reset_index()


def _consistency_block(r: pd.DataFrame, variant: str, dim: str,
                       keys) -> pd.DataFrame:
    """按一个维度汇总方向一致性 share（长表）。dim 仅用于标注。"""
    if keys is None:
        gg = [("ALL", r)]
    else:
        gg = list(r.groupby(keys, sort=True, dropna=False))
    rows = []
    for key, g in gg:
        if not isinstance(key, tuple):
            key = (key,)
        label = "|".join(str(k) for k in key)
        for share_col, src in SHARE_SPEC:
            rows.append(dict(
                variant=variant, group_dim=dim, group_value=label,
                metric=f"share_positive__{src}", n=len(g),
                share=float((g[src] > 0).mean()),
                median=float(g[src].median())))
        rows.append(dict(
            variant=variant, group_dim=dim, group_value=label,
            metric="matched_n", n=len(g), share=np.nan, median=float(len(g))))
    return pd.DataFrame(rows)


def build_consistency(r: pd.DataFrame, variant: str) -> pd.DataFrame:
    parts = [
        _consistency_block(r, variant, "overall", None),
        _consistency_block(r, variant, "wf", ["wf"]),
        _consistency_block(r, variant, "action", ["action"]),
        _consistency_block(r, variant, "structure_scale", ["structure_scale"]),
        _consistency_block(r, variant, "waiting_distance", ["waiting_distance"]),
        _consistency_block(r, variant, "base_h_later_h", ["base_h", "later_h"]),
    ]
    return pd.concat(parts, ignore_index=True)


def _verdict(r: pd.DataFrame) -> dict:
    """广泛 trade-off 判定（无 threshold 调参，仅用多数定义 share>0.5）。"""
    per_wf = []
    for wf, g in r.groupby("wf", sort=True):
        s_rr = float((g["delta_RR"] > 0).mean())
        s_tf = float((g["delta_target_first"] > 0).mean())
        s_er = float((g["delta_E_R_lower"] > 0).mean())
        per_wf.append(dict(wf=wf, n=len(g), share_delta_RR_positive=s_rr,
                           share_delta_target_first_positive=s_tf,
                           share_delta_E_R_lower_positive=s_er,
                           rr_up=s_rr > 0.5, tf_down=s_tf < 0.5,
                           er_down=s_er < 0.5))
    ok = [w for w in per_wf if w["rr_up"] and w["tf_down"] and w["er_down"]]
    overall = dict(
        n=len(r),
        share_delta_RR_positive=float((r["delta_RR"] > 0).mean()),
        share_delta_target_first_positive=float((r["delta_target_first"] > 0).mean()),
        share_delta_E_R_lower_positive=float((r["delta_E_R_lower"] > 0).mean()))
    if len(ok) == len(per_wf) and len(per_wf) == 3:
        v = "TRADEOFF_BROAD"
    elif len(ok) >= 2:
        v = "TRADEOFF_BROAD_WITH_EXCEPTIONS"
    else:
        v = "TRADEOFF_CELL_SPECIFIC"
    return dict(verdict=v, per_wf=per_wf, overall=overall)


def _md_report(cells: pd.DataFrame, cons: pd.DataFrame,
               verdicts: dict, n_raw: dict) -> str:
    L = []
    L.append("# P12.5 — Matched Waiting Consistency Gate\n")
    L.append("**措辞合同**：本页只描述 *matched waiting effect*（同一 contact 的配对"
             "等待效应）。**不是** causal waiting effect——能在 later horizon 继续 "
             "available 的 contact 本身仍可能有选择性。\n")
    L.append("**边界**：不 bootstrap、不调 threshold、不做模型；"
             "geometry/stop/target/horizon/scale 定义与 Stage 4A 完全一致"
             "（直接复用其函数）。\n")
    L.append(f"- raw matched rows：first_available={n_raw['first_available']:,}，"
             f"h2_cohort={n_raw['h2_cohort']:,}\n")

    L.append("## 1. 方向一致性判定（overall + by WF）\n")
    for variant, v in verdicts.items():
        L.append(f"### variant = `{variant}`\n")
        L.append(f"- 判定：**{v['verdict']}**")
        ov = v["overall"]
        L.append(f"- overall n={ov['n']:,}：share(RR↑)={ov['share_delta_RR_positive']:.3f}，"
                 f"share(target_first↑)={ov['share_delta_target_first_positive']:.3f}，"
                 f"share(E[R]_lower↑)={ov['share_delta_E_R_lower_positive']:.3f}\n")
        t = pd.DataFrame(v["per_wf"])[
            ["wf", "n", "share_delta_RR_positive",
             "share_delta_target_first_positive",
             "share_delta_E_R_lower_positive", "rr_up", "tf_down", "er_down"]]
        L.append("```\n" + t.to_string(index=False) + "\n```\n")
        L.append("解读：`share(RR↑) > 0.5` = 多数配对 RR 改善；"
                 "`share(target_first↑) < 0.5` = 多数配对命中概率下降；"
                 "`share(E[R]_lower↑) < 0.5` = 多数配对净 E[R]_lower 下降。\n")

    L.append("## 2. 每个 (base_h → later_h) 的方向一致性 share\n")
    sel = cons[(cons["group_dim"] == "base_h_later_h")
               & (cons["metric"].isin(
                   ["share_positive__delta_RR",
                    "share_positive__delta_target_first",
                    "share_positive__delta_E_R_lower"]))]
    piv = sel.pivot_table(index=["variant", "group_value"], columns="metric",
                          values="share", aggfunc="first")
    nn = cons[(cons["group_dim"] == "base_h_later_h")
              & (cons["metric"] == "matched_n")].set_index(
        ["variant", "group_value"])["n"]
    piv["matched_n"] = nn
    L.append("```\n" + piv.reset_index().to_string(index=False) + "\n```\n")

    L.append("## 3. 按 waiting_distance = later_h − base_h\n")
    wd = cons[(cons["group_dim"] == "waiting_distance")]
    L.append("```\n" + wd.to_string(index=False) + "\n```\n")

    L.append("## 4. 按 action / structure_scale\n")
    for dim in ("action", "structure_scale"):
        sub = cons[cons["group_dim"] == dim]
        L.append(f"### by {dim}\n")
        L.append("```\n" + sub.to_string(index=False) + "\n```\n")

    L.append("## 5. cell 明细（median + share）\n")
    L.append("```\n" + cells.to_string(index=False) + "\n```\n")
    L.append("## 6. 下一步\n")
    L.append("P12.5 完成后不停机，直接进入 Stage 4B "
             "Latent-State Compression Gate v1.0（`run_latent_state_compression_v1.py`）。\n")
    return "\n".join(L)


def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = s4a.load_env()
    print(f"[ENV] loaded ({time.perf_counter()-t0:.1f}s)")
    df, gmap = s4a.compute_action_surface(D, master_by_sym, bars_by_sym)
    df = df[df["wf"].isin(["WF1", "WF2", "WF3"])].reset_index(drop=True)
    print(f"[SURFACE] rows={len(df)} contacts={df['gid'].nunique()} "
          f"({time.perf_counter()-t0:.1f}s)")

    cells_all, cons_all, verdicts, n_raw = [], [], {}, {}
    for variant, fn in VARIANTS.items():
        agg, raw = fn(df, gmap)
        assert len(raw) == int(agg["matched_n"].sum()), (
            f"[FATAL] {variant} matched_n 汇总不一致")
        r = _prepare_raw(raw, variant)
        cells_all.append(_cells(r))
        cons_all.append(build_consistency(r, variant))
        verdicts[variant] = _verdict(r)
        n_raw[variant] = len(r)
        print(f"[P12.5] variant={variant} raw={len(r)} "
              f"verdict={verdicts[variant]['verdict']}")

    cells = pd.concat(cells_all, ignore_index=True)
    cons = pd.concat(cons_all, ignore_index=True)
    cells.to_csv(OUT / "p12_5_matched_waiting_cells.csv", index=False)
    cons.to_csv(OUT / "p12_5_direction_consistency.csv", index=False)
    (OUT / "P12_5_MATCHED_WAITING_CONSISTENCY.md").write_text(
        _md_report(cells, cons, verdicts, n_raw), encoding="utf-8")
    audit = dict(
        gate="P12.5 Matched Waiting Consistency Gate",
        base_commit=s4a.BASE_COMMIT,
        wording_contract="matched waiting effect (NOT causal)",
        reuse="run_liquidity_field_action_surface_v1 (authoritative functions)",
        variants={v: dict(n_raw=n_raw[v], verdict=verdicts[v]["verdict"])
                  for v in VARIANTS},
        verdicts=verdicts,
        no_bootstrap=True, no_threshold_tuning=True,
        stage4a_definitions_unchanged=True)
    json.dump(audit, open(OUT / "P12_5_MATCHED_WAITING_AUDIT.json", "w"),
              indent=2, ensure_ascii=False, default=str)
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
