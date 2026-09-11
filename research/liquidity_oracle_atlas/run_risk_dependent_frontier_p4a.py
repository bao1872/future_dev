"""SMC Risk-dependent Frontier —— P4a：3 品种连续前沿语义复现试跑。

固定 AG / RB / I（仅用于验证实现语义，不是性能筛选，不得更换）。

HARD GATE（用户 P4a-6）：用连续 path_geometry 在冻结 7 档回放 best_R_lower/
upper/resolution_class，再重建 rr_direction，与冻结 oracle_risk_direction_v1_2
对比：
  - 冻结 LONG_DOMINATES / SHORT_DOMINATES 行要求 100.000% 一致（一条都不行）；
  - 其它三类也报一致率与 mismatch 原因。

本脚本复用 cf_common 的冻结语义（active_mask / oracle_direction 同义重建）+
新 path_geometry 连续基础设施，证明两者逐位等价。

Governance: TRADING_METRICS=NOT_APPLICABLE；不预测/不 PnL。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.cf_common import (
    replay_symbol, classify_rr_direction, RESULTS,
)
from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import (
    RESULTS as LIFE_RESULTS,
)

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_risk_dependent_frontier_v1")
OUT.mkdir(parents=True, exist_ok=True)

PILOT_SYMBOLS = ["AG", "RB", "I"]
MASTER = pd.read_parquet(RESULTS / "liquidity_master_v1_1.parquet")
CONTACTS = pd.read_parquet(RESULTS / "liquidity_contacts_v1_1.parquet")

frozen_dir = pd.read_parquet(ATLAS / "oracle_risk_direction_v1_2.parquet")
frozen_frontier = pd.read_parquet(ATLAS / "oracle_risk_frontier_v1_2.parquet")

print(f"[P4a] pilot symbols={PILOT_SYMBOLS}", flush=True)
oracle_parts, target_parts = [], []
for sym in PILOT_SYMBOLS:
    t0 = __import__("time").perf_counter()
    odf, tdf = replay_symbol(sym, CONTACTS, MASTER)
    oracle_parts.append(odf)
    target_parts.append(tdf)
    print(f"  {sym}: oracle={len(odf)} target_rows={len(tdf)} "
          f"({__import__('time').perf_counter()-t0:.1f}s)", flush=True)

O = pd.concat(oracle_parts, ignore_index=True)
T = pd.concat(target_parts, ignore_index=True)
RD = classify_rr_direction(O)

# ---------- HARD GATE: rr_direction 与冻结对比 ----------
fdir = frozen_dir[frozen_dir["symbol"].isin(PILOT_SYMBOLS)].copy()
m = fdir.merge(RD, on=["symbol", "liquidity_id", "contact_number", "risk_ATR"],
               how="inner", suffixes=("_frozen", "_recon"))
print(f"\n[P4a] rr_direction 比对行数 = {len(m)}", flush=True)
ct = pd.crosstab(m["rr_direction_frozen"], m["rr_direction_recon"])
ct.to_csv(OUT / "p4a_rr_direction_confusion.csv", encoding="utf-8-sig")
print("\n=== rr_direction confusion (frozen rows × recon cols) ===")
print(ct.to_string(), flush=True)

# 各类精确一致率
classes = ["LONG_DOMINATES", "SHORT_DOMINATES", "TRADEOFF_OR_OVERLAP",
           "UNRESOLVED_CENSOR", "NO_COMPARABLE_TARGET"]
class_stat = {}
for c in classes:
    sub = m[m["rr_direction_frozen"] == c]
    ok = (sub["rr_direction_recon"] == c).sum()
    class_stat[c] = dict(frozen_n=int(len(sub)), match=int(ok),
                         rate=(round(ok / len(sub), 6) if len(sub) else None))

# 致命 HARD GATE
hard = class_stat["LONG_DOMINATES"]["frozen_n"] > 0 and \
       class_stat["LONG_DOMINATES"]["match"] == class_stat["LONG_DOMINATES"]["frozen_n"] and \
       class_stat["SHORT_DOMINATES"]["frozen_n"] > 0 and \
       class_stat["SHORT_DOMINATES"]["match"] == class_stat["SHORT_DOMINATES"]["frozen_n"]
verdict = "P4A_PASS" if hard else "P4A_SEMANTIC_FAIL"
print(f"\n[P4a] HARD GATE (LONG/SHORT 100%): {hard} -> {verdict}", flush=True)

# ---------- 逐方向 oracle 字段对账（P4a-7）----------
ff = frozen_frontier[frozen_frontier["symbol"].isin(PILOT_SYMBOLS)].copy()
of = O.copy()
# 字段对齐：冻结用 best_R_lower/upper/resolution_class
chk = ff.merge(of, on=["symbol", "liquidity_id", "contact_number",
                       "direction", "risk_ATR"],
               how="inner", suffixes=("_f", "_r"))
field_stat = {}
for col in ["best_R_lower", "best_R_upper", "resolution_class"]:
    a = chk[f"{col}_f"]
    b = chk[f"{col}_r"]
    if col == "resolution_class":
        match = (a.astype(str) == b.astype(str))
    else:
        match = np.isclose(a.to_numpy(float), b.to_numpy(float),
                          equal_nan=True)
    field_stat[col] = dict(n=int(len(chk)), match=int(match.sum()),
                           rate=round(float(match.mean()), 6))
print("\n=== 逐方向 oracle 字段对账（冻结 vs 连续重建）===", flush=True)
for k, v in field_stat.items():
    print(f"  {k}: {v['match']}/{v['n']} = {v['rate']}", flush=True)

# mismatch 原因抽样（P4a-6 其它三类）
mism = m[m["rr_direction_frozen"] != m["rr_direction_recon"]].copy()
mism_summary = (mism.groupby(["rr_direction_frozen", "rr_direction_recon"])
                .size().reset_index(name="n").sort_values("n", ascending=False))
mism_summary.to_csv(OUT / "p4a_mismatch_detail.csv", index=False,
                    encoding="utf-8-sig")

# reconstruction audit csv（每个 contact×risk 的重建值）
audit_rows = m[["symbol", "liquidity_id", "contact_number", "risk_ATR",
                "rr_direction_frozen", "rr_direction_recon",
                "long_R_lower_frozen", "long_R_lower_recon",
                "short_R_upper_frozen", "short_R_upper_recon"]].copy()
audit_rows["match"] = (audit_rows["rr_direction_frozen"]
                       == audit_rows["rr_direction_recon"])
audit_rows.to_csv(OUT / "p4a_semantic_reconstruction_audit.csv", index=False,
                  encoding="utf-8-sig")

# 保存 target reachability（P4b 基础设施试跑，3 品种）
T.to_parquet(OUT / "target_reachability_frontier.parquet", index=False)

# ---------- 汇总 ----------
summary = dict(
    experiment="SMC Risk-dependent Frontier P4a Semantic Pilot",
    pilot_symbols=PILOT_SYMBOLS,
    n_contacts_pilot=int(CONTACTS[CONTACTS["symbol"].isin(PILOT_SYMBOLS)]
                         .groupby(["symbol", "liquidity_id",
                                   "contact_number"]).ngroups),
    rr_direction_confusion=ct.to_dict(),
    class_match_rate=class_stat,
    hard_gate_long_short_100pct=bool(hard),
    oracle_field_match=field_stat,
    verdict=verdict,
    governance=dict(trading_metrics="NOT_APPLICABLE", no_pnl=True,
                    atlas_v1_2_unchanged=True),
)
with open(OUT / "P4A_AUDIT.json", "w") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False, default=str)

print(f"\n[P4a] DONE -> verdict={verdict}; "
      f"target_reachability rows={len(T)}", flush=True)
print(f"        confusion -> {OUT/'p4a_rr_direction_confusion.csv'}")
print(f"        audit     -> {OUT/'p4a_semantic_reconstruction_audit.csv'}")
