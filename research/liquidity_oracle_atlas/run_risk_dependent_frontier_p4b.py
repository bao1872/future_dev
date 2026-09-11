"""SMC Risk-dependent Frontier —— P4b：全 15 品种 Continuous Frontier。

P4a HARD GATE 已 PASS（连续 path_geometry 与冻结 7 档语义逐位等价），
自动进入本步。

固定 15 品种（冻结 Atlas v1.2 宇宙，禁止 LC）：
  AG AU CU AL SN NI RB I SC RU MA TA M P CF

产出（原目录 research/analysis_results/smc_risk_dependent_frontier_v1/）：
  target_reachability_frontier.parquet   (per contact×direction×target, 不入 git)
  continuous_direction_frontier.parquet  (per contact 方向 run 边界)
  continuous_switch_events.parquet       (per switch×unlock target)
  continuous_switch_classification.csv   (per contact)
  grid_midpoint_vs_continuous.csv
  continuous_threshold_by_tb.csv / by_symbol.csv
  unlock_target_scope_profile.csv
  P4_CONTINUOUS_FRONTIER_AUDIT.json
  SMC_RISK_DEPENDENT_CONTINUOUS_FRONTIER_V1.md

Governance: TRADING_METRICS=NOT_APPLICABLE；不预测/不 PnL。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import opportunity_common as oc
from research.liquidity_oracle_atlas.cf_common import (
    replay_symbol_continuous, classify_rr_direction,
    continuous_frontier_from_targets,
)
from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import (
    RESULTS,
)

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_risk_dependent_frontier_v1")
OUT.mkdir(parents=True, exist_ok=True)

SYMBOLS = ["AG", "AU", "CU", "AL", "SN", "NI", "RB", "I", "SC", "RU",
           "MA", "TA", "M", "P", "CF"]
MASTER = pd.read_parquet(RESULTS / "liquidity_master_v1_1.parquet")
CONTACTS = pd.read_parquet(RESULTS / "liquidity_contacts_v1_1.parquet")
frozen_dir = pd.read_parquet(ATLAS / "oracle_risk_direction_v1_2.parquet")
stab = pd.read_parquet(ATLAS / "oracle_direction_stability_v1_2.parquet")

# TB（trading block）per contact
lab = oc.load_labels()
keys = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]
blk = oc.attach_trading_day_block(lab[keys].copy())
ct2tb = (lab[["symbol", "liquidity_id", "contact_number"]]
          .assign(block=blk["block"].values)
          .drop_duplicates(["symbol", "liquidity_id", "contact_number"])
          .set_index(["symbol", "liquidity_id", "contact_number"])
          ["block"].to_dict())

print(f"[P4b] 15 symbols={SYMBOLS}", flush=True)
oracle_parts, switch_rows, class_rows, frontier_rows, tgt_parts = [], [], [], [], []
for sym in SYMBOLS:
    t0 = time.perf_counter()
    O, cparts = replay_symbol_continuous(sym, CONTACTS, MASTER)
    oracle_parts.append(O)
    for c in cparts:
        res = continuous_frontier_from_targets(
            c["long_targets"], c["short_targets"], c["path_censor"])
        tb = ct2tb.get((sym, c["liquidity_id"], c["contact_number"]), "NA")
        class_rows.append(dict(
            symbol=sym, liquidity_id=c["liquidity_id"],
            contact_number=c["contact_number"], block=tb,
            path_censor=c["path_censor"], n_switch=res["n_switch"],
            classification=res["classification"]))
        # frontier 方向 run 边界（紧凑）
        seq = res["frontier"]
        for i in range(len(seq)):
            if i == 0 or seq[i][2] != seq[i - 1][2]:
                frontier_rows.append(dict(
                    symbol=sym, liquidity_id=c["liquidity_id"],
                    contact_number=c["contact_number"],
                    threshold=seq[i][0], direction=seq[i][2],
                    best_long_distance=seq[i][3],
                    best_short_distance=seq[i][4]))
        for ev in res["events"]:
            for ut in ev["unlock_targets"]:
                switch_rows.append(dict(
                    symbol=sym, liquidity_id=c["liquidity_id"],
                    contact_number=c["contact_number"], block=tb,
                    critical_risk_ATR=ev["critical_risk_ATR"],
                    switch_type=ev["switch_type"],
                    before_long_distance=ev["before_long_distance"],
                    before_short_distance=ev["before_short_distance"],
                    after_long_distance=ev["after_long_distance"],
                    after_short_distance=ev["after_short_distance"],
                    unlock_side=("LONG" if any(
                        t in (x["price"] for x in c["long_targets"])
                        for t in [ut["price"]]) else "SHORT"),
                    unlock_target_price=ut["price"],
                    unlock_target_distance_ATR=ut["dist"],
                    unlock_cluster_size=ut["cluster_size"],
                    unlock_scopes=ut["scopes"]))
    print(f"  {sym}: contacts={len(cparts)} "
          f"({time.perf_counter()-t0:.1f}s)", flush=True)

O = pd.concat(oracle_parts, ignore_index=True)
RD = classify_rr_direction(O)
print(f"[P4b] oracle={len(O)} rr_direction rows={len(RD)}", flush=True)

# ---- 全 15 品种 rr_direction 复现校验（应与冻结 100%）----
fdir = frozen_dir.copy()
m = fdir.merge(RD, on=["symbol", "liquidity_id", "contact_number", "risk_ATR"],
               how="inner", suffixes=("_f", "_r"))
all_match = bool((m["rr_direction_f"] == m["rr_direction_r"]).all())
print(f"[P4b] 全 15 品种 rr_direction 100% 复现 = {all_match}", flush=True)

class_df = pd.DataFrame(class_rows)
switch_df = pd.DataFrame(switch_rows)
frontier_df = pd.DataFrame(frontier_rows)

# ---- 主队列：冻结 RISK_DEPENDENT = 18,495 ----
rd_keys = stab[stab["direction_stability"] == "RISK_DEPENDENT"][
    ["symbol", "liquidity_id", "contact_number"]].copy()
rd_cls = rd_keys.merge(class_df, on=["symbol", "liquidity_id", "contact_number"],
                       how="left")
rd_dist = rd_cls["classification"].value_counts().to_dict()
rd_total = len(rd_cls)
rd_clear = rd_dist.get("CLEAR_SINGLE_CONTINUOUS_SWITCH", 0) + \
           rd_dist.get("CLEAR_MULTI_CONTINUOUS_SWITCH", 0)
rd_pct_clear = round(100.0 * rd_clear / rd_total, 2) if rd_total else None

# 全样本分布
all_dist = class_df["classification"].value_counts().to_dict()

# ---- 解锁 target scope 画像 ----
if len(switch_df):
    prof = (switch_df.groupby("unlock_scopes").size().reset_index(name="n")
            .sort_values("n", ascending=False))
    prof["share"] = (prof["n"] / prof["n"].sum()).round(4)
    prof.to_csv(OUT / "unlock_target_scope_profile.csv", index=False,
                encoding="utf-8-sig")
    # cluster size 分布
    csz = (switch_df.groupby("unlock_cluster_size").size().reset_index(name="n")
           .sort_values("unlock_cluster_size"))
    csz.to_csv(OUT / "unlock_target_cluster_size.csv", index=False,
               encoding="utf-8-sig")

# ---- 连续 threshold 分布 vs 旧 1.25 GRID_MIDPOINT ----
if len(switch_df):
    sw = switch_df.copy()
    sw["critical_risk_ATR"] = pd.to_numeric(sw["critical_risk_ATR"])
    by_tb = sw.groupby("block")["critical_risk_ATR"].agg(
        n="count", median="median", p25=lambda x: x.quantile(.25),
        p75=lambda x: x.quantile(.75), p90=lambda x: x.quantile(.90)).reset_index()
    by_tb.to_csv(OUT / "continuous_threshold_by_tb.csv", index=False,
                 encoding="utf-8-sig")
    by_sym = sw.groupby("symbol")["critical_risk_ATR"].agg(
        n="count", median="median", p25=lambda x: x.quantile(.25),
        p75=lambda x: x.quantile(.75), p90=lambda x: x.quantile(.90)).reset_index()
    by_sym.to_csv(OUT / "continuous_threshold_by_symbol.csv", index=False,
                  encoding="utf-8-sig")
    overall = dict(
        n=int(len(sw)),
        median=round(float(sw["critical_risk_ATR"].median()), 4),
        p25=round(float(sw["critical_risk_ATR"].quantile(.25)), 4),
        p75=round(float(sw["critical_risk_ATR"].quantile(.75)), 4),
        p90=round(float(sw["critical_risk_ATR"].quantile(.90)), 4),
        mean=round(float(sw["critical_risk_ATR"].mean()), 4))
else:
    overall, by_tb, by_sym = {}, pd.DataFrame(), pd.DataFrame()

grid_vs_cont = pd.DataFrame([dict(
    estimate="GRID_MIDPOINT_SWITCH_ESTIMATE (old, P1.5)", value=1.25),
    dict(estimate="continuous_critical_risk_median", value=overall.get("median")),
    dict(estimate="continuous_critical_risk_p25", value=overall.get("p25")),
    dict(estimate="continuous_critical_risk_p75", value=overall.get("p75")),
    dict(estimate="continuous_critical_risk_p90", value=overall.get("p90")),
    dict(estimate="continuous_critical_risk_mean", value=overall.get("mean"))])
grid_vs_cont.to_csv(OUT / "grid_midpoint_vs_continuous.csv", index=False,
                    encoding="utf-8-sig")

# ---- ROI 裁决 ----
# Gate: RD 中多数事件仍为 CLEAR 连续 switch，且非 censor/ambiguous 主导，
#       15 品种 & TB1-4 广泛存在。
rd_censor = rd_dist.get("DATA_END_CENSORED", 0)
rd_amb = rd_dist.get("AMBIGUOUS_SWITCH", 0)
rd_no = rd_dist.get("NO_CONTINUOUS_SWITCH", 0)
n_sym_with_switch = int(switch_df["symbol"].nunique()) if len(switch_df) else 0
n_tb_with_switch = int(switch_df["block"].nunique()) if len(switch_df) else 0
structural = (rd_pct_clear is not None and rd_pct_clear >= 50.0
              and rd_clear > (rd_censor + rd_amb + rd_no)
              and n_sym_with_switch >= 12 and n_tb_with_switch == 4)
if structural:
    verdict = "STRUCTURAL_RISK_SWITCH"
elif rd_clear < (rd_no + rd_censor + rd_amb):
    verdict = "GRID_ARTIFACT"
else:
    verdict = "HETEROGENEOUS_MECHANISM"

# ---- 写 parquet / csv ----
class_df.to_csv(OUT / "continuous_switch_classification.csv", index=False,
                encoding="utf-8-sig")
if len(switch_df):
    switch_df.to_parquet(OUT / "continuous_switch_events.parquet", index=False)
frontier_df.to_parquet(OUT / "continuous_direction_frontier.parquet",
                       index=False)

audit = dict(
    experiment="SMC Risk-dependent Frontier P4b Continuous Frontier (15-symbol)",
    symbols=SYMBOLS, n_symbols=len(SYMBOLS),
    rr_direction_full15_reproduced_100pct=all_match,
    n_contacts_total=int(len(class_df)),
    classification_distribution_all=all_dist,
    primary_cohort="RISK_DEPENDENT (frozen v1.2, 18495)",
    rd_classification=rd_dist,
    rd_total=rd_total, rd_pct_clear=rd_pct_clear,
    n_symbols_with_switch=n_sym_with_switch,
    n_tb_with_switch=n_tb_with_switch,
    continuous_threshold_overall=overall,
    verdict=verdict,
    grid_vs_continuous=grid_vs_cont.to_dict("records"),
    governance=dict(trading_metrics="NOT_APPLICABLE", no_pnl=True,
                    atlas_v1_2_unchanged=True),
)
with open(OUT / "P4_CONTINUOUS_FRONTIER_AUDIT.json", "w") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False, default=str)

print("\n=== P4b 主队列（RISK_DEPENDENT 18,495）连续 switch 分类 ===")
print(json.dumps(rd_dist, indent=2, ensure_ascii=False))
print(f"\n  CLEAR 占比 = {rd_pct_clear}%  -> verdict = {verdict}", flush=True)
print(f"  连续 critical_risk: {overall}", flush=True)
print(f"[P4b] switch_events={len(switch_df)} frontier_rows={len(frontier_df)}")
print("  outputs ->", OUT)
