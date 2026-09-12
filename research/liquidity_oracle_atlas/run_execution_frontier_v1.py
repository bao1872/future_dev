"""Execution Semantics Repair v1: E0 reference -> E1 executable Lag-1 market.

R1-R4 membership remains frozen and next-open-conditioned.  Once membership is
known at that open, executable entry is delayed to the following full 5m bar
open.  E2 limit frontier is gated on E1 whole-policy positive expectancy in all
three development WFs.  Prospective P1 is never read.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env
from research.liquidity_oracle_atlas import run_latent_state_compression_v1 as s4b
from research.liquidity_oracle_atlas import run_liquidity_field_action_surface_v1 as s4a
from research.liquidity_oracle_atlas.run_enter_skip_selection_v1 import (
    PRIMARY_REGIONS, build_policy, primary_matches, contribution_and_cost)

OUT=Path("research/analysis_results/execution_frontier_v1"); OUT.mkdir(parents=True,exist_ok=True)
BASE="27a4786d65dfd95b41c9a58748df24e1e5bacd84"
WFS=["WF1","WF2","WF3"]

def metric(g):
    executed=g[g.status=="EXECUTED_LAG1"]; x=executed[executed.R_lower.notna()]; r=x.R_lower.astype(float)
    pos=r[r>0].sum(); neg=-r[r<0].sum(); cum=x.sort_values(
        ["entry_time","symbol","gid"]).R_lower.cumsum(); dd=cum-cum.cummax()
    return dict(n_signals=len(g),n_trades=len(executed),nonentry_n=int((g.status!="EXECUTED_LAG1").sum()),
        resolved_n=len(x),censored_n=int((executed.R_lower.isna()).sum()),
        E_R_lower=float(r.mean()) if len(r) else np.nan,E_R_upper=float(x.R_upper.mean()) if len(x) else np.nan,
        EV_R_lower_per_signal=float(r.sum()/len(g)) if len(g) else np.nan,
        EV_R_lower_censor_worst_per_signal=float((r.sum()-executed.R_lower.isna().sum())/len(g)) if len(g) else np.nan,
        win_rate=float((r>0).mean()) if len(r) else np.nan,
        profit_factor=float(pos/neg) if neg else np.nan,total_R=float(r.sum()),
        max_drawdown_R=float(dd.min()) if len(dd) else np.nan,
        ambiguous_rate=float(g.ambiguous.mean()) if len(g) else np.nan)

def main():
    t0=time.perf_counter(); D,masters,bars=load_env()
    surface,gmap=s4b.compute_action_surface_all_blocks(D,masters,bars)
    repro=s4b.assert_reproduce_stage4a(surface); surface=surface[surface.wf.isin(WFS)].copy()
    F=D["F"].copy(); meta=gmap.merge(F[["symbol","liquidity_id","contact_number",
        "decision_time","contact_bar_index","atr0","side"]],
        on=["symbol","liquidity_id","contact_number"],validate="one_to_one")
    meta=meta.merge(surface[["gid","wf"]].drop_duplicates(),on="gid",validate="one_to_one")
    matches=primary_matches(surface); policy=build_policy(surface,matches,meta)
    if len(policy)!=73885 or int(policy.entered.sum())!=9015:
        raise SystemExit("STOP_EXECUTION_BASELINE_REPRODUCTION_FAIL")
    ref_contrib,_=contribution_and_cost(policy)
    old=pd.read_csv("research/analysis_results/enter_skip_selection_v1/selection_region_contribution.csv")
    cols=["region","wf","n_trades","E_R_lower_resolved"]
    a=ref_contrib[ref_contrib.wf.isin(WFS)][cols].sort_values(cols[:2]).reset_index(drop=True)
    b=old[old.wf.isin(WFS)][cols].sort_values(cols[:2]).reset_index(drop=True)
    if len(a)!=len(b) or np.nanmax(np.abs(a.E_R_lower_resolved-b.E_R_lower_resolved))>=1e-12 or not (a.n_trades==b.n_trades).all():
        raise SystemExit("STOP_EXECUTION_BASELINE_REPRODUCTION_FAIL")

    chosen=policy[policy.entered][["gid","wf","symbol","region","contact_bar_index"]]
    m=chosen.merge(matches,on=["gid","wf","region"],suffixes=("","_s"),validate="one_to_one")
    m=m.merge(meta[["gid","atr0"]],on="gid",validate="one_to_one")
    m["reference_entry"]=m.entry_atr*m.atr0
    m["target_price"]=m.reference_entry+m.d*m.target_atr*m.atr0
    m["stop_price"]=m.reference_entry-m.d*m.risk_atr*m.atr0
    rows=[]
    for sym,g in m.groupby("symbol",sort=False):
        B=bars[sym]; ix=g.index.to_numpy(); gi=g.copy()
        signal_i=gi.contact_bar_index.astype(int).to_numpy()+1+gi.h.astype(int).to_numpy()
        entry_i=signal_i+1
        valid=entry_i<B["n"]; entry=np.full(len(gi),np.nan); entry[valid]=B["o"][entry_i[valid]]
        direction=gi.d.to_numpy(int); target=gi.target_price.to_numpy(float); stop=gi.stop_price.to_numpy(float)
        geometry=valid & (~B["disc"][np.minimum(entry_i,B["n"]-1)]) & (direction*(target-entry)>0) & (direction*(entry-stop)>0)
        for j,r in enumerate(gi.itertuples(index=False)):
            base=dict(gid=r.gid,wf=r.wf,symbol=sym,region=r.region,direction=direction[j],
                signal_bar_index=int(signal_i[j]),entry_bar_index=int(entry_i[j]),
                reference_entry=r.reference_entry,entry_price=entry[j],target_price=target[j],stop_price=stop[j])
            # E1.1 Gate A: during the one full waiting bar the frozen thesis
            # must remain alive.  Since no position exists yet, either boundary
            # invalidates the signal and has zero economic return; ordering is
            # irrelevant when both are touched.
            si=signal_i[j]
            target_touched=(B["h"][si]>=target[j] if direction[j]>0 else B["l"][si]<=target[j])
            stop_touched=(B["l"][si]<=stop[j] if direction[j]>0 else B["h"][si]>=stop[j])
            if target_touched and stop_touched: pre="BOTH_BOUNDARIES_TOUCHED_BEFORE_ENTRY"
            elif target_touched: pre="TARGET_CONSUMED_BEFORE_ENTRY"
            elif stop_touched: pre="STOP_INVALIDATED_BEFORE_ENTRY"
            elif valid[j] and B["disc"][entry_i[j]]: pre="DISCONTINUITY_BEFORE_ENTRY"
            else: pre=None
            if pre: rows.append(dict(**base,status=pre,R_lower=np.nan,R_upper=np.nan,
                ambiguous=False,censored=False,entry_time=pd.NaT)); continue
            if not geometry[j]: rows.append(dict(**base,status="LAG1_GAP_ENTRY_INVALID",R_lower=np.nan,R_upper=np.nan,
                ambiguous=False,censored=False,entry_time=pd.NaT)); continue
            ei=entry_i[j]; fh=B["h"][ei:ei+34][None,:]; fl=B["l"][ei:ei+34][None,:]
            if fh.shape[1]<34:
                fh=np.pad(fh,((0,0),(0,34-fh.shape[1])),constant_values=np.nan)
                fl=np.pad(fl,((0,0),(0,34-fl.shape[1])),constant_values=np.nan)
            disc=B["disc"][ei:ei+34]
            if disc.any():
                k=int(np.argmax(disc)); fh[:,k+1:]=np.nan; fl[:,k+1:]=np.nan
            oc=s4a.first_hit_bounds(fh,fl,np.array([entry[j]]),np.array([target[j]]),np.array([stop[j]]),np.array([direction[j]]))
            rows.append(dict(**base,status="EXECUTED_LAG1",R_lower=oc["R_lower"][0],R_upper=oc["R_upper"][0],
                ambiguous=bool(oc["ambiguous"][0]),censored=bool(oc["censored"][0]),entry_time=pd.Timestamp(B["t"][ei])))
    lag=pd.DataFrame(rows)
    out=[]
    for scope,gg in [("WHOLE_POLICY",lag)]+[(r,lag[lag.region==r]) for r in ["R1","R2","R3","R4"]]:
        for wf in WFS: out.append(dict(execution="E1_LAG1_MARKET",scope=scope,wf=wf,**metric(gg[gg.wf==wf])))
    metrics=pd.DataFrame(out); whole=metrics[metrics.scope=="WHOLE_POLICY"]
    gate=bool((whole.EV_R_lower_per_signal>0).all() &
              (whole.EV_R_lower_censor_worst_per_signal>0).all())
    verdict="LAG1_HARDENED_EXECUTION_EDGE_SURVIVES" if gate else "STOP_NO_ROBUST_LAG1_EXECUTION_EDGE"
    lag.to_parquet(OUT/"execution_lag1_trades.parquet",index=False)
    metrics.to_csv(OUT/"execution_lag1_by_wf_region.csv",index=False)
    audit=dict(experiment="Execution Semantics Repair v1",policy_commit=BASE,
        prospective_holdout_read=False,n_contacts=len(policy),n_reference_enter=int(policy.entered.sum()),
        stage4a_reproduction=repro,reference_max_abs_diff=0.0,execution_lag_bars=1,
        e0_classification="REFERENCE_GEOMETRY_ORACLE_ISH_NOT_LIVE_EXECUTABLE",
        prefill_path_gate=True,
        prefill_status_counts=lag.status.value_counts().to_dict(),
        whole_policy_by_wf=whole.to_dict("records"),gate_E1_positive_3_of_3=gate,
        verdict=verdict,next_stage=("E2_LIMIT_FRONTIER_ALLOWED" if gate else "STOP_BEFORE_LIMIT_FRONTIER"),
        elapsed_seconds=round(time.perf_counter()-t0,3))
    (OUT/"EXECUTION_FRONTIER_AUDIT.json").write_text(json.dumps(audit,indent=2,default=str))
    (OUT/"EXECUTION_FRONTIER_V1.md").write_text(report(audit,metrics))
    print(json.dumps(audit,indent=2,default=str))

def report(a,m):
    return f"""# Execution Semantics Repair v1

E0 is reclassified as **reference geometry**, not a live-executable backtest. R1-R4
membership observes the next open; E1 enters at the following full 5m bar open.

## Gate

**{a['verdict']}** — `{a['next_stage']}`

```
{m.to_string(index=False)}
```

P1 was not read. No signal threshold, region, feature, symbol, or time rule changed.
"""
if __name__=="__main__": main()
