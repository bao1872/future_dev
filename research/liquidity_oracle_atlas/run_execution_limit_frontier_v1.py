"""E2 Limit Execution Frontier: frozen signals, SINGLE_ATTEMPT primary, TTL=2."""
from __future__ import annotations
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env
from research.liquidity_oracle_atlas.run_liquidity_field_action_surface_v1 import first_hit_bounds

OUT=Path("research/analysis_results/execution_frontier_v1")
RRS=np.array([.25,.5,.75,1.,1.25,1.5,2.,3.])
COSTS=[0,.01,.02,.03,.05,.1]; MODELS=["TOUCH_FILL","STRICT_TRADE_THROUGH"]
BENCH={"WF1":.042021741035443656,"WF2":.04751890891070517,"WF3":.07637480131763613}
PRE={"TARGET_CONSUMED_BEFORE_ENTRY","STOP_INVALIDATED_BEFORE_ENTRY","BOTH_BOUNDARIES_TOUCHED_BEFORE_ENTRY"}

def run_block(g,B,rr,model):
    n=len(g); d=g.direction.to_numpy(int); target=g.target_price.to_numpy(float); stop=g.stop_price.to_numpy(float)
    limit=(target+rr*stop)/(1+rr); ei=g.entry_bar_index.to_numpy(int)
    o=B['o'][ei]; H=np.column_stack([B['h'][ei+k] for k in range(2)]); L=np.column_stack([B['l'][ei+k] for k in range(2)])
    O=np.column_stack([B['o'][ei+k] for k in range(2)])
    pre=g.status.isin(PRE).to_numpy(); disc0=B['disc'][ei]
    gap=(d*(target-o)<=0)|(d*(o-stop)<=0)
    market=(~pre)&(~gap)&(~disc0)&np.where(d>0,o<=limit,o>=limit)
    fill=np.where(d[:,None]>0,L<limit[:,None] if model.startswith('STRICT') else L<=limit[:,None],
                  H>limit[:,None] if model.startswith('STRICT') else H>=limit[:,None])
    tt=np.where(d[:,None]>0,H>=target[:,None],L<=target[:,None]); ss=np.where(d[:,None]>0,L<=stop[:,None],H>=stop[:,None])
    active=(~pre)&(~gap)&(~disc0); status=np.full(n,"EXPIRED_UNFILLED",object); status[pre]=g.status.to_numpy()[pre]; status[gap&~pre]="GAP_INVALID_BEFORE_LIMIT_ACTIVATION"; status[disc0&~pre]="DISCONTINUITY_BEFORE_LIMIT_ACTIVATION"
    fill_i=np.full(n,-1); fill_px=np.full(n,np.nan); lower=np.zeros(n); upper=np.zeros(n); amb=np.zeros(n,bool)
    fill_i[market]=0; fill_px[market]=o[market]; status[market]="FILLED_AT_OPEN"
    for k in range(2):
        pending=active&(fill_i<0)&(status=="EXPIRED_UNFILLED")
        dk=B['disc'][ei+k]; status[pending&dk]="DISCONTINUITY_BEFORE_LIMIT_ACTIVATION"; pending &= ~dk
        f=pending&fill[:,k]; t=pending&tt[:,k]; s=pending&ss[:,k]
        ft=f&t; fs=f&s&~t; status[t&~f]="MISS_TARGET_BEFORE_FILL"; status[s&~f&~t]="STOP_INVALIDATED_BEFORE_FILL"; status[t&s&~f]="BOTH_BOUNDARIES_BEFORE_FILL"
        status[ft]="AMBIGUOUS_FILL_TARGET_ORDER"; amb[ft]=True; upper[ft]=rr
        status[fs]="AMBIGUOUS_FILL_STOP_ORDER"; amb[fs]=True; lower[fs]=-1
        clean=f&~t&~s; fill_i[clean]=k; fill_px[clean]=limit[clean]; status[clean]="FILLED"
    filled=fill_i>=0
    if filled.any():
        ids=np.flatnonzero(filled); maxn=34; fh=np.full((len(ids),maxn),np.nan); fl=fh.copy()
        for k in range(maxn):
            idx=ei[ids]+fill_i[ids]+k; ok=idx<B['n']; fh[ok,k]=B['h'][idx[ok]]; fl[ok,k]=B['l'][idx[ok]]
        oc=first_hit_bounds(fh,fl,fill_px[ids],target[ids],stop[ids],d[ids]); lower[ids]=oc['R_lower']; upper[ids]=oc['R_upper']; amb[ids]|=oc['ambiguous']
    actual=d*(target-fill_px)/(d*(fill_px-stop));
    return pd.DataFrame(dict(gid=g.gid.to_numpy(),wf=g.wf.to_numpy(),symbol=g.symbol.to_numpy(),region=g.region.to_numpy(),rr_target=rr,fill_model=model,status=status,filled=filled,fill_bar=fill_i,actual_RR=actual,R_lower=lower,R_upper=upper,ambiguous=amb))

def agg(x,keys):
    rows=[]
    for key,g in x.groupby(keys,sort=False):
        if not isinstance(key,tuple):key=(key,)
        f=g[g.filled]; r=f.R_lower.dropna(); pos=r[r>0].sum();neg=-r[r<0].sum()
        row=dict(zip(keys,key)); row.update(n_signals=len(g),filled_n=len(f),fill_rate_per_signal=g.filled.mean(),
            miss_target_before_fill_rate=g.status.eq('MISS_TARGET_BEFORE_FILL').mean(),stop_invalid_before_fill_rate=g.status.eq('STOP_INVALIDATED_BEFORE_FILL').mean(),
            expired_rate=g.status.eq('EXPIRED_UNFILLED').mean(),ambiguity_rate=g.ambiguous.mean(),median_bars_to_fill=f.fill_bar.median(),
            median_actual_RR=f.actual_RR.median(),mean_actual_RR=f.actual_RR.mean(),filled_win_rate=(r>0).mean(),filled_profit_factor=pos/neg if neg else np.nan,
            E_R_lower_filled=r.mean(),E_R_upper_filled=f.R_upper.mean(),EV_R_lower_per_signal=g.R_lower.sum()/len(g),EV_R_upper_per_signal=g.R_upper.sum()/len(g),
            censor_worst_EV_per_signal=(g.R_lower.sum()-f.R_lower.isna().sum())/len(g))
        rows.append(row)
    return pd.DataFrame(rows)

def main():
    audit=json.load(open(OUT/'EXECUTION_FRONTIER_AUDIT.json'))
    expected={'EXECUTED_LAG1':7097,'TARGET_CONSUMED_BEFORE_ENTRY':1550,'STOP_INVALIDATED_BEFORE_ENTRY':310,'BOTH_BOUNDARIES_TOUCHED_BEFORE_ENTRY':15,'LAG1_GAP_ENTRY_INVALID':43}
    if audit['prefill_status_counts']!=expected:raise SystemExit('STOP_E2_BASELINE_REPRODUCTION_FAIL')
    D,_,bars=load_env(); sig=pd.read_parquet(OUT/'execution_lag1_trades.parquet'); parts=[]
    for rr in RRS:
      for model in MODELS:
       for sym,g in sig.groupby('symbol',sort=False):parts.append(run_block(g,bars[sym],rr,model))
    x=pd.concat(parts,ignore_index=True); wf=agg(x,['wf','rr_target','fill_model']); reg=agg(x,['wf','region','rr_target','fill_model'])
    wf['delta_EV_vs_E1_market']=wf.EV_R_lower_per_signal-wf.wf.map(BENCH)
    gateA=[];gateB=[]
    for (rr,mo),g in wf.groupby(['rr_target','fill_model']):
      if len(g)==3 and (g.EV_R_lower_per_signal>0).all():gateA.append([rr,mo])
      if len(g)==3 and (g.delta_EV_vs_E1_market>0).all():gateB.append([rr,mo])
    cost=pd.concat([wf.assign(cost_R=c,net_EV_per_signal=wf.EV_R_lower_per_signal-c*wf.fill_rate_per_signal) for c in COSTS])
    sym=agg(x,['symbol','region','rr_target','fill_model']);sym=sym[sym.n_signals>=100]
    wf.to_csv(OUT/'execution_limit_frontier_by_wf.csv',index=False);reg.to_csv(OUT/'execution_limit_frontier_by_region.csv',index=False)
    x.groupby(['wf','rr_target','fill_model','status']).size().rename('n').reset_index().to_csv(OUT/'execution_limit_lifecycle_breakdown.csv',index=False)
    cost.to_csv(OUT/'execution_limit_cost_sensitivity.csv',index=False);sym.to_csv(OUT/'execution_limit_symbol_diagnostic.csv',index=False)
    pd.DataFrame(columns=['note']).assign(note=['REASSESS_NEXT_H_DEFERRED_NOT_PRIMARY']).to_csv(OUT/'execution_limit_reassess_diagnostic.csv',index=False)
    au=dict(experiment='E2_LIMIT_FRONTIER',signals=9015,ttl_full_bars=2,tick_metadata='NO_AUTHORITATIVE_TICK_METADATA',P1_read=False,
        gateA_combinations=gateA,gateB_combinations=gateB,ROBUST_LIMIT_EDGE_SURVIVES=bool(gateA),ROBUST_LIMIT_BEATS_MARKET=bool(gateB),
        reassess_status='DEFERRED_SECONDARY',forbidden=['optimal RR selection','P1','symbol tuning','ML','RL'])
    (OUT/'EXECUTION_LIMIT_FRONTIER_AUDIT.json').write_text(json.dumps(au,indent=2))
    (OUT/'EXECUTION_LIMIT_FRONTIER_V1.md').write_text('# E2 Limit Execution Frontier\n\n'+json.dumps(au,indent=2)+'\n\nPrimary SINGLE_ATTEMPT only; no RR selected.\n')
    print(json.dumps(au,indent=2))
if __name__=='__main__':main()
