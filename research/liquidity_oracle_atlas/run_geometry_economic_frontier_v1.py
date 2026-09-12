"""Geometry Economic Frontier v1.0.

唯一问题：在结构第一次可交易时，冻结的 Entry/Stop/Target geometry 是否
已经定义了跨 WF1/WF2/WF3 重复出现的正期望 ENTER region？

这是 Stage 4A/4B 的只读经济面后处理，不训练模型、不搜索阈值、不新增特征，
也不改变 target、stop、horizon、structure scale 或 first-hit 语义。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env
from research.liquidity_oracle_atlas import run_latent_state_compression_v1 as s4b


OUT = Path("research/analysis_results/geometry_economic_frontier_v1")
OUT.mkdir(parents=True, exist_ok=True)

TEST_WF = ["WF1", "WF2", "WF3"]
MIN_N_PER_WF = 200
MAX_AMBIGUITY_GAP_R = 0.25

# 用户预注册的 coarse bins。左闭右开，最后一档包含 +inf。
TARGET_EDGES = [-np.inf, 0.5, 1.0, 2.0, 3.0, 5.0, np.inf]
TARGET_LABELS = ["<0.5", "0.5-1", "1-2", "2-3", "3-5", ">=5"]
RISK_EDGES = [-np.inf, 0.25, 0.5, 1.0, 2.0, 3.0, np.inf]
RISK_LABELS = ["<0.25", "0.25-0.5", "0.5-1", "1-2", "2-3", ">=3"]
RR_EDGES = [-np.inf, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, np.inf]
RR_LABELS = ["<0.5", "0.5-0.75", "0.75-1", "1-1.5", "1.5-2", "2-3", ">=3"]


def first_available_geometry(surface: pd.DataFrame) -> pd.DataFrame:
    """每个 contact/action/scale 只保留预注册 horizon 中第一次 available。"""
    d = surface[surface["wf"].isin(TEST_WF) & surface["available"]].copy()
    key = ["gid", "wf", "action", "scale"]
    d["first_h"] = d.groupby(key, sort=False)["h"].transform("min")
    d = d[d["h"] == d["first_h"]].copy()
    assert not d.duplicated(key).any(), "FIRST_AVAILABLE_KEY_NOT_UNIQUE"
    assert (d["h"] == d["first_h"]).all(), "FIRST_AVAILABLE_H_MISMATCH"
    return d


def add_bins(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["target_bin"] = pd.cut(d["target_atr"], TARGET_EDGES,
                              labels=TARGET_LABELS, right=False)
    d["risk_bin"] = pd.cut(d["risk_atr"], RISK_EDGES,
                            labels=RISK_LABELS, right=False)
    d["rr_bin"] = pd.cut(d["rr"], RR_EDGES, labels=RR_LABELS, right=False)
    assert d[["target_bin", "risk_bin", "rr_bin"]].notna().all().all(), \
        "GEOMETRY_BIN_ASSIGNMENT_FAILED"
    return d


def aggregate_plane(d: pd.DataFrame, plane: str) -> pd.DataFrame:
    if plane == "target_x_risk":
        bins = ["target_bin", "risk_bin"]
    elif plane == "rr_x_target":
        bins = ["rr_bin", "target_bin"]
    else:
        raise ValueError(plane)
    keys = ["wf", "action", "first_h", "scale"] + bins
    rows = []
    for key, g in d.groupby(keys, observed=True, sort=False):
        row = dict(zip(keys, key))
        row.update(
            plane=plane,
            n_available=int(len(g)),
            n_resolved=int(g["R_lower"].notna().sum()),
            target_first_rate=float(g["target_first"].mean()),
            stop_first_rate=float(g["stop_first"].mean()),
            ambiguous_rate=float(g["ambiguous"].mean()),
            censored_rate=float(g["censored"].mean()),
            E_R_lower=float(g["R_lower"].mean()),
            E_R_upper=float(g["R_upper"].mean()),
            ambiguity_gap_R=float((g["R_upper"] - g["R_lower"]).mean()),
            median_target_atr=float(g["target_atr"].median()),
            median_risk_atr=float(g["risk_atr"].median()),
            median_RR=float(g["rr"].median()),
        )
        rows.append(row)
    return pd.DataFrame(rows)


def robust_regions(cells: pd.DataFrame, plane: str) -> pd.DataFrame:
    bins = (["target_bin", "risk_bin"] if plane == "target_x_risk"
            else ["rr_bin", "target_bin"])
    region_key = ["action", "first_h", "scale"] + bins
    rows = []
    for key, g in cells.groupby(region_key, observed=True, sort=False):
        by_wf = g.set_index("wf")
        present = all(w in by_wf.index for w in TEST_WF)
        row = dict(zip(region_key, key))
        row["plane"] = plane
        row["all_wf_present"] = present
        for wf in TEST_WF:
            if wf not in by_wf.index:
                row.update({f"{wf}_n": 0, f"{wf}_E_R_lower": np.nan,
                            f"{wf}_E_R_upper": np.nan,
                            f"{wf}_ambiguity_gap_R": np.nan})
                continue
            x = by_wf.loc[wf]
            row.update({f"{wf}_n": int(x["n_available"]),
                        f"{wf}_E_R_lower": float(x["E_R_lower"]),
                        f"{wf}_E_R_upper": float(x["E_R_upper"]),
                        f"{wf}_ambiguity_gap_R": float(x["ambiguity_gap_R"])})
        enough = present and all(row[f"{w}_n"] >= MIN_N_PER_WF for w in TEST_WF)
        positive = present and all(row[f"{w}_E_R_lower"] > 0 for w in TEST_WF)
        bounded = present and all(
            np.isfinite(row[f"{w}_ambiguity_gap_R"])
            and row[f"{w}_ambiguity_gap_R"] <= MAX_AMBIGUITY_GAP_R
            for w in TEST_WF)
        row["min_n_across_wf"] = min(row[f"{w}_n"] for w in TEST_WF)
        row["min_E_R_lower_across_wf"] = np.nanmin(
            [row[f"{w}_E_R_lower"] for w in TEST_WF]) if present else np.nan
        row["max_ambiguity_gap_R_across_wf"] = np.nanmax(
            [row[f"{w}_ambiguity_gap_R"] for w in TEST_WF]) if present else np.nan
        row["sample_gate"] = enough
        row["positive_3_of_3"] = positive
        row["ambiguity_gate"] = bounded
        row["robust_positive_geometry"] = enough and positive and bounded
        rows.append(row)
    return pd.DataFrame(rows)


def write_report(cells: pd.DataFrame, regions: pd.DataFrame, audit: dict) -> None:
    robust = regions[regions["robust_positive_geometry"]]
    eligible = regions[regions["sample_gate"] & regions["ambiguity_gate"]]
    robust_primary = robust[robust["plane"] == "target_x_risk"]
    robust_secondary = robust[robust["plane"] == "rr_x_target"]
    lines = [
        "# Geometry Economic Frontier v1.0\n",
        "## 判定\n",
        f"**{audit['verdict']}**\n",
        "研究对象是每个 contact/action/scale 在预注册 horizon 中的第一次可交易 geometry。"
        "本实验没有训练模型、搜索阈值或加入 morphology/liquidity 特征。\n",
        "## 预注册合同\n",
        f"- 每个 WF 最低样本量：`{MIN_N_PER_WF}`\n",
        f"- ambiguity gap 上限：`{MAX_AMBIGUITY_GAP_R:.2f}R`\n",
        "- 正期望要求：WF1、WF2、WF3 的 `E[R]lower` 均严格大于 0\n",
        "- 主平面：target distance × risk distance；辅助平面：RR × target distance\n",
        "- region 保留 action、first-available h、structure scale，不跨执行语义合并\n",
        "## 样本与结果\n",
        f"- first-available rows：`{audit['n_first_available_rows']:,}`\n",
        f"- 全量 WF cells：`{len(cells):,}`\n",
        f"- 跨 WF regions：`{len(regions):,}`\n",
        f"- 同时通过样本量与 ambiguity gate：`{len(eligible):,}`\n",
        f"- ROBUST_POSITIVE_GEOMETRY：主平面 `{len(robust_primary):,}`；"
        f"辅助平面 `{len(robust_secondary):,}`\n",
        "两个平面的 region 会重叠，数量不可相加解释为独立 edge。结果是 gross、"
        "cell-level 描述，不含成本，也尚未构成可执行 selection policy。\n",
    ]
    if len(robust):
        cols = ["plane", "action", "first_h", "scale", "target_bin",
                "risk_bin", "rr_bin", "min_n_across_wf",
                "min_E_R_lower_across_wf", "max_ambiguity_gap_R_across_wf"]
        cols = [c for c in cols if c in robust.columns]
        lines += ["## 通过的 regions（完整）\n", "```\n",
                  robust[cols].to_string(index=False), "\n```\n",
                  "下一步允许进入预注册的 ENTER/SKIP selection experiment；"
                  "本报告本身不做 selection。\n"]
    else:
        lines += ["## STOP 结论\n",
                  "没有 region 同时满足 3/3 WF conservative 正期望、最低样本量和"
                  " ambiguity 稳健性。因此当前 Liquidity-Field Reaction execution line "
                  "在 Geometry Frontier 处停止；不进入 ENTER/SKIP、DP 或 RL。\n"]
    (OUT / "GEOMETRY_ECONOMIC_FRONTIER_V1.md").write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="仅用于开发 smoke；正式判定必须全量运行")
    args = ap.parse_args()
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = load_env()
    surface, _ = s4b.compute_action_surface_all_blocks(
        D, master_by_sym, bars_by_sym)
    if args.sample:
        keep = set(surface["gid"].drop_duplicates().sample(
            min(args.sample, surface["gid"].nunique()), random_state=7))
        surface = surface[surface["gid"].isin(keep)].copy()
    else:
        reproduction = s4b.assert_reproduce_stage4a(surface)
    if args.sample:
        reproduction = None
    first = add_bins(first_available_geometry(surface))
    cell_parts, region_parts = [], []
    for plane in ["target_x_risk", "rr_x_target"]:
        c = aggregate_plane(first, plane)
        cell_parts.append(c)
        region_parts.append(robust_regions(c, plane))
    cells = pd.concat(cell_parts, ignore_index=True, sort=False)
    regions = pd.concat(region_parts, ignore_index=True, sort=False)
    cells.to_csv(OUT / "geometry_frontier_cells_by_wf.csv", index=False)
    regions.to_csv(OUT / "geometry_frontier_robust_regions.csv", index=False)
    n_robust = int(regions["robust_positive_geometry"].sum())
    full = args.sample is None
    verdict = ("ROBUST_POSITIVE_GEOMETRY" if n_robust and full else
               "NO_ROBUST_POSITIVE_GEOMETRY_STOP" if full else
               "SAMPLE_MODE_NO_FORMAL_VERDICT")
    audit = dict(
        experiment="Geometry Economic Frontier v1.0",
        base_commit=s4b.BASE_COMMIT,
        full_run=full,
        sample=args.sample,
        n_surface_rows=int(len(surface)),
        n_first_available_rows=int(len(first)),
        n_cells=int(len(cells)),
        n_regions=int(len(regions)),
        n_robust_positive=int(n_robust),
        verdict=verdict,
        stage4a_reproduction=reproduction,
        preregistered=dict(
            target_bins=TARGET_LABELS, risk_bins=RISK_LABELS, rr_bins=RR_LABELS,
            min_n_per_wf=MIN_N_PER_WF,
            max_ambiguity_gap_R=MAX_AMBIGUITY_GAP_R,
            positive_rule="E_R_lower > 0 in WF1, WF2, WF3",
            planes=["target_x_risk", "rr_x_target"],
            region_identity=["action", "first_h", "structure_scale"]),
        forbidden=["ML", "threshold search", "best-cell optimization",
                   "new morphology/liquidity features", "DP", "RL"],
        elapsed_seconds=round(time.perf_counter() - t0, 3))
    (OUT / "GEOMETRY_ECONOMIC_FRONTIER_AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2))
    write_report(cells, regions, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
