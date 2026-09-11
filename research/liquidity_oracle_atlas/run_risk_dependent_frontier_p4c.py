"""SMC Risk-dependent Frontier —— P4c 连续语义修复。

用户审计结论：P4a 接受；P4b 连续 evaluator 未通过自己的 HARD GATE
（只存 req_through，没用 req_before；"100% 复现"实际验证的是 7 档
reconstruct_oracle_grid，不是 continuous_frontier_from_targets；censor 未按 risk
区间应用；2.2 统计被 unlock target 数加权；scope 无 enrichment）。

P4c 修复：
- 紧凑 target 同时保存 required_before / required_through（两个独立临界值）；
- continuous evaluator 逐位复刻冻结 v1.2：lower=max{dist:req_through<r}、
  upper=max{dist:req_before<r}、r>max_adverse⇒CENSORED(upper=NaN)；
- 新 HARD GATE：continuous_side_bounds + continuous_pair_direction 直接在 7 档点
  与冻结 rr_direction 比较（678,300 行，要求 100%）；
- transition 用 5 态序列（LONG/SHORT_DOMINATES/TRADEOFF/UNRESOLVED/NO_COMPARABLE），
  区分 DIRECT / OVERLAP_MEDIATED / CENSOR_MEDIATED；
- 统计单位 = transition_event（不再被 unlock target 数加权）；
- scope 做 baseline-normalized enrichment。

Governance: TRADING_METRICS=NOT_APPLICABLE；不预测/不 PnL/不 regime 模型。
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
    prep_side, bounds_at, continuous_pair_direction,
    continuous_frontier_analyze, RISK_ATR_GRID, SCOPE_ORDER,
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

lab = oc.load_labels()
blk = oc.attach_trading_day_block(lab[["symbol", "liquidity_id",
                                       "contact_number"]].copy())
ct2tb = (lab[["symbol", "liquidity_id", "contact_number"]].assign(
    block=blk["block"].values).drop_duplicates(
    ["symbol", "liquidity_id", "contact_number"]).set_index(
    ["symbol", "liquidity_id", "contact_number"])["block"].to_dict())

# primary cohort：冻结 RISK_DEPENDENT（18,495）
rd = stab[stab["direction_stability"] == "RISK_DEPENDENT"][
    ["symbol", "liquidity_id", "contact_number"]].copy()
rd["k"] = list(zip(rd["liquidity_id"].astype(str),
                   rd["contact_number"].astype(str)))
rd_by_sym = {s: set(rd[rd["symbol"] == s]["k"]) for s in SYMBOLS}

frozen_dir["k"] = list(zip(frozen_dir["liquidity_id"].astype(str),
                           frozen_dir["contact_number"].astype(str)))
frozen_dir["risk_ATR"] = frozen_dir["risk_ATR"].astype(float)


def pair_raw(bl, bs, r):
    """未四舍五入的连续方向（用于区分 rounding mismatch）。"""
    if (bl["state"] == "NO_ACTIVE_TARGET"
            or bs["state"] == "NO_ACTIVE_TARGET"):
        return "NO_COMPARABLE_TARGET"
    ll = bl["lower"] / r if np.isfinite(bl["lower"]) else np.nan
    lu = bl["upper"] / r if np.isfinite(bl["upper"]) else np.nan
    sl = bs["lower"] / r if np.isfinite(bs["lower"]) else np.nan
    su = bs["upper"] / r if np.isfinite(bs["upper"]) else np.nan
    if np.isnan(lu) or np.isnan(su):
        return "UNRESOLVED_CENSOR"
    if ll > su:
        return "LONG_DOMINATES"
    if sl > lu:
        return "SHORT_DOMINATES"
    return "TRADEOFF_OR_OVERLAP"


print(f"[P4c] 15 symbols={SYMBOLS}", flush=True)
gate_rows, mism = [], {}
trans_rows, unlock_rows, class_rows, overlap_rows = [], [], [], []
bg_scope = {s: 0 for s in SCOPE_ORDER}
bg_total = 0
trans_scope_inst = {s: 0 for s in SCOPE_ORDER}
trans_total = 0

for sym in SYMBOLS:
    t0 = time.perf_counter()
    O, cparts = replay_symbol_continuous(sym, CONTACTS, MASTER)
    # ---- HARD GATE：连续 evaluator 直接 vs 冻结 rr_direction ----
    sym_gate = []
    for c in cparts:
        lid = str(c["liquidity_id"]); cn = str(c["contact_number"])
        lp = prep_side(c["long_targets"], c["long_max_adverse"])
        sp = prep_side(c["short_targets"], c["short_max_adverse"])
        for risk in RISK_ATR_GRID:
            bl = bounds_at(lp, risk); bs = bounds_at(sp, risk)
            st = continuous_pair_direction(bl, bs, risk)
            sym_gate.append((lid, cn, float(risk), st))
    gcols = ["liquidity_id", "contact_number", "risk_ATR", "cont_rr"]
    gd = pd.DataFrame(sym_gate, columns=gcols)
    fg = frozen_dir[(frozen_dir["symbol"] == sym)][
        ["liquidity_id", "contact_number", "risk_ATR", "rr_direction"]].copy()
    fg["liquidity_id"] = fg["liquidity_id"].astype(str)
    fg["contact_number"] = fg["contact_number"].astype(str)
    gd["liquidity_id"] = gd["liquidity_id"].astype(str)
    gd["contact_number"] = gd["contact_number"].astype(str)
    mg = fg.merge(gd, on=["liquidity_id", "contact_number", "risk_ATR"],
                  how="inner")
    same = (mg["rr_direction"] == mg["cont_rr"])
    gate_rows.append((sym, len(mg), round(100.0 * same.mean(), 4)))
    for fr, cr in zip(mg["rr_direction"], mg["cont_rr"]):
        if fr != cr:
            mism[(sym, fr, cr)] = mism.get((sym, fr, cr), 0) + 1
    # ---- transition 分析 ----
    for c in cparts:
        lid = str(c["liquidity_id"]); cn = str(c["contact_number"])
        k = (lid, cn)
        is_rd = k in rd_by_sym[sym]
        tb = ct2tb.get((sym, c["liquidity_id"], c["contact_number"]), "NA")
        res = continuous_frontier_analyze(
            c["long_targets"], c["short_targets"],
            c["long_max_adverse"], c["short_max_adverse"], c["path_censor"])
        for ti, tr in enumerate(res["transitions"]):
            eid = f"{sym}|{lid}|{cn}|{ti}"
            trans_rows.append(dict(
                symbol=sym, liquidity_id=lid, contact_number=cn, block=tb,
                is_RD=is_rd, transition_index=ti,
                from_state=tr["from_state"], to_state=tr["to_state"],
                transition_type=tr["transition_type"],
                critical_risk_ATR=tr["critical_risk_ATR"]))
            for ut in tr["unlock_targets"]:
                unlock_rows.append(dict(
                    event_id=eid, symbol=sym, liquidity_id=lid,
                    contact_number=cn, block=tb, is_RD=is_rd,
                    side=ut["side"], price=ut["price"], dist=ut["dist"],
                    cluster_size=ut["cluster_size"], scopes=ut["scopes"],
                    threshold_type=ut["threshold_type"],
                    threshold_ATR=ut["threshold_ATR"]))
                if is_rd:
                    for s in SCOPE_ORDER:
                        if s in ut["scopes"].split("|"):
                            trans_scope_inst[s] += 1
                            trans_total += 1
        for ob in res["overlap_bands"]:
            overlap_rows.append(dict(
                symbol=sym, liquidity_id=lid, contact_number=cn, block=tb,
                is_RD=is_rd, start=ob["start"], end=ob["end"],
                width=ob["width"], center=ob["center"],
                left=ob["left"], right=ob["right"]))
        class_rows.append(dict(
            symbol=sym, liquidity_id=lid, contact_number=cn, block=tb,
            path_censor=c["path_censor"], is_RD=is_rd,
            contact_class=res["contact_class"],
            has_long=res["has_long"], has_short=res["has_short"],
            has_tradeoff=res["has_tradeoff"], has_censor=res["has_censor"],
            direct=res["direct"], n_transitions=res["n_transitions"]))
        # background scope（RD contact 内所有 reached target breakpoint）
        if is_rd:
            for t in c["long_targets"] + c["short_targets"]:
                if not t["reached"]:
                    continue
                for key in ("required_before_ATR", "required_through_ATR"):
                    if not np.isfinite(t.get(key)):
                        continue
                    for s in SCOPE_ORDER:
                        if s in t["scopes"].split("|"):
                            bg_scope[s] += 1
                            bg_total += 1
    print(f"  {sym}: contacts={len(cparts)} "
          f"gate={gate_rows[-1][2]}% ({time.perf_counter()-t0:.1f}s)",
          flush=True)

# ---- GATE 汇总 ----
gate_df = pd.DataFrame(gate_rows, columns=["symbol", "n_rows", "match_rate_pct"])
gate_df.to_csv(OUT / "p4c_frozen_grid_reconstruction.csv", index=False,
               encoding="utf-8-sig")
overall_gate = round(100.0 * sum(r[1] for r in gate_rows
                                 if r[1]) / sum(r[1] for r in gate_rows), 4)
gate_pass = all(r[2] == 100.0 for r in gate_rows)
print(f"[P4c] GATE overall match = {overall_gate}%  pass={gate_pass}",
      flush=True)
if mism:
    print(f"  mismatches (top): {sorted(mism.items(), key=lambda x:-x[1])[:10]}",
          flush=True)

# ---- 输出表 ----
class_df = pd.DataFrame(class_rows)
trans_df = pd.DataFrame(trans_rows)
unlock_df = pd.DataFrame(unlock_rows)
overlap_df = pd.DataFrame(overlap_rows)

# primary cohort (RD) 分类分布
rd_class = class_df[class_df["is_RD"]]
rd_dist = rd_class["contact_class"].value_counts().to_dict()
rd_total = len(rd_class)
# secondary（全样本）
all_dist = class_df["contact_class"].value_counts().to_dict()
all_total = len(class_df)

# transition type（RD）分布
rd_trans = trans_df[trans_df["is_RD"]]
trans_type_dist = rd_trans["transition_type"].value_counts().to_dict()

# direct threshold（RD, DIRECT events）
direct = rd_trans[rd_trans["transition_type"] == "DIRECT"]["critical_risk_ATR"]
direct_stats = dict(n=int(len(direct)),
                    median=round(float(direct.median()), 4) if len(direct) else None,
                    p25=round(float(direct.quantile(.25)), 4) if len(direct) else None,
                    p75=round(float(direct.quantile(.75)), 4) if len(direct) else None,
                    p90=round(float(direct.quantile(.90)), 4) if len(direct) else None)

# overlap band（RD）
rd_ov = overlap_df[overlap_df["is_RD"]]
if len(rd_ov):
    overlap_stats = dict(
        n=int(len(rd_ov)),
        start_median=round(float(rd_ov["start"].median()), 4),
        end_median=round(float(rd_ov["end"].median()), 4),
        width_median=round(float(rd_ov["width"].median()), 4),
        center_median=round(float(rd_ov["center"].median()), 4),
        center_p25=round(float(rd_ov["center"].quantile(.25)), 4),
        center_p75=round(float(rd_ov["center"].quantile(.75)), 4))
else:
    overlap_stats = dict(n=0)

# censor transitions（RD）
censor_n = int((rd_trans["transition_type"] == "CENSOR_MEDIATED").sum())

# ---- scope enrichment ----
bg_share = {s: (bg_scope[s] / bg_total if bg_total else 0.0)
            for s in SCOPE_ORDER}
tr_share = {s: (trans_scope_inst[s] / trans_total if trans_total else 0.0)
            for s in SCOPE_ORDER}
enrich = pd.DataFrame({
    "scope": SCOPE_ORDER,
    "background_share": [round(bg_share[s], 4) for s in SCOPE_ORDER],
    "transition_share": [round(tr_share[s], 4) for s in SCOPE_ORDER],
})
enrich["enrichment"] = enrich.apply(
    lambda r: round(r["transition_share"] / r["background_share"], 3)
    if r["background_share"] > 0 else np.nan, axis=1)
enrich.to_csv(OUT / "scope_background_vs_transition.csv", index=False,
              encoding="utf-8-sig")

# ---- 写 parquet / csv ----
class_df.to_csv(OUT / "rd_transition_classification.csv", index=False,
                encoding="utf-8-sig")
if len(trans_df):
    trans_df.to_parquet(OUT / "continuous_transition_events.parquet",
                        index=False)
if len(unlock_df):
    unlock_df.to_parquet(OUT / "continuous_transition_unlock_targets.parquet",
                         index=False)
if len(overlap_df):
    overlap_df.to_parquet(OUT / "overlap_band_profile.parquet", index=False)
pd.DataFrame([direct_stats]).to_csv(
    OUT / "direct_switch_threshold_profile.csv", index=False,
    encoding="utf-8-sig")
pd.DataFrame([overlap_stats]).to_csv(
    OUT / "overlap_band_profile.csv", index=False, encoding="utf-8-sig")
pd.DataFrame([dict(censor_transition_count=censor_n)]).to_csv(
    OUT / "censor_transition_profile.csv", index=False,
    encoding="utf-8-sig")

# ---- 裁决（P4c-14）----
def verdict(rd_dist, rd_total):
    dl = rd_dist.get("DIRECT_CONTINUOUS_SWITCH", 0)
    ov = rd_dist.get("OVERLAP_MEDIATED_TRANSITION", 0)
    ce = rd_dist.get("CENSOR_MEDIATED_UNRESOLVED", 0)
    no = (rd_dist.get("NO_CONTINUOUS_DIRECTION_CHANGE", 0)
          + rd_dist.get("SINGLE_SIDE_DOMINANCE", 0))
    if rd_total == 0:
        return "UNKNOWN"
    if dl > (ov + ce + no):
        return "DIRECT_STRUCTURAL_SWITCH"
    if ov >= dl and ov >= ce and ov >= no:
        return "OVERLAP_MEDIATED_TRANSITION"
    if ce > dl and ce > ov:
        return "CENSOR_DOMINATED"
    if no > (dl + ov + ce):
        return "GRID_SAMPLING_ARTIFACT"
    return "HETEROGENEOUS"

v = verdict(rd_dist, rd_total)

audit = dict(
    experiment="SMC Risk-dependent Frontier P4c Continuous Semantic Repair",
    symbols=SYMBOLS, n_symbols=len(SYMBOLS),
    gate_overall_match_pct=overall_gate, gate_pass=gate_pass,
    gate_per_symbol=gate_rows,
    gate_mismatch_examples=sorted(mism.items(), key=lambda x: -x[1])[:20],
    primary_cohort="RISK_DEPENDENT (frozen v1.2, 18495)",
    rd_total=rd_total, rd_contact_class_distribution=rd_dist,
    all_contact_class_distribution=all_dist,
    rd_transition_type_distribution=trans_type_dist,
    direct_switch_threshold_stats=direct_stats,
    overlap_band_stats=overlap_stats,
    censor_transition_count=censor_n,
    scope_enrichment=enrich.to_dict("records"),
    verdict=v,
    provisional_note=("P4b 的 CLEAR=0.32%/AMBIGUOUS=85.4%/2.2ATR/CONTIG_SESSION 主导"
                     " 均 PROVISIONAL；P4c 已修正连续 evaluator 并重新分类。"),
    governance=dict(trading_metrics="NOT_APPLICABLE", no_pnl=True,
                    no_predict=True, atlas_v1_2_unchanged=True),
)
with open(OUT / "P4C_CONTINUOUS_SEMANTIC_AUDIT.json", "w") as f:
    json.dump(audit, f, indent=2, ensure_ascii=False, default=str)

print("\n=== P4c GATE ===")
print(gate_df.to_string(index=False))
print(f"\n=== RD({rd_total}) contact_class ===")
print(json.dumps(rd_dist, indent=2, ensure_ascii=False))
print(f"\n=== RD transition_type ===\n{json.dumps(trans_type_dist, indent=2)}")
print(f"\n  direct_threshold={direct_stats}")
print(f"  overlap_stats={overlap_stats}")
print(f"  censor_transitions={censor_n}")
print(f"\n  VERDICT = {v}")
print(f"[P4c] trans_events={len(trans_df)} unlock={len(unlock_df)} "
      f"overlap={len(overlap_df)} class={len(class_df)}")
print("  outputs ->", OUT)
