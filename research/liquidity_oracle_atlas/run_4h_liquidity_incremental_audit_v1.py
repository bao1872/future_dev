"""V2-A: causal 4h-liquidity incremental-information audit.

Independent sidecar only: no policy, target, stop, region, or P1 mutation.
"""
from __future__ import annotations
import json,sys,time
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression,Ridge
from sklearn.metrics import roc_auc_score,log_loss,brier_score_loss,r2_score,mean_absolute_error
from sklearn.preprocessing import StandardScaler
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.build_pytdx_panel import aggregate_15m
from research.ob_trigger_snapshot import aggregate_1h_from_15m,build_full_ob_smc_tf
from research.build_ob_candidate_universe_v3 import aggregate_4h_from_1h
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags
from research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 import find_contacts
from research.liquidity_oracle_atlas import run_latent_state_compression_v1 as s4b

OUT=Path("research/analysis_results/liquidity_4h_incremental_v1")
BASE="c2fe8354ca1a9dfed5058a7417895f3355395caa"
KEY=["symbol","liquidity_id","contact_number"]
LAMS={"0p5":.5,"1p0":1.,"2p0":2.,"4p0":4.}; CHUNK=4096
F4=([f"liq4h_density_{x}" for x in LAMS]+[f"liq4h_oriented_imbalance_{x}" for x in LAMS]
    +["liq4h_profit_prox","liq4h_adverse_prox","liq4h_active_count_log1p"])
F4RAW=([f"liq4h_density_{x}" for x in LAMS]+[f"liq4h_imbalance_{x}" for x in LAMS]
       +["liq4h_prox_up","liq4h_prox_down","liq4h_active_count_log1p"])

def build_levels(sym):
    five=load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
    five["volume"]=five["trade"].astype(float)
    one=aggregate_1h_from_15m(aggregate_15m(five)); four=aggregate_4h_from_1h(one)
    smc=build_full_ob_smc_tf(four.copy()); rows=[]
    for src in (smc.get("pivots",[]),smc.get("equal_highs_lows",[])):
      for p in src:
        ty=p.get("type")
        mp={"swing_high":("CONFIRMED_SWING_HIGH",1),"swing_low":("CONFIRMED_SWING_LOW",-1),
            "EQH":("CANONICAL_EQH",1),"EQL":("CANONICAL_EQL",-1)}
        if ty not in mp: continue
        ci=int(p["confirmed_index"]); assert ci<len(four),"4H_CONFIRM_INDEX_OOB"
        rows.append(dict(symbol=sym,source_tf="4h",liquidity_scope="4h",liquidity_type=mp[ty][0],side=mp[ty][1],
          price=float(p["level"]),origin_time=pd.Timestamp(p["anchor_time"]),
          available_time=pd.Timestamp(four.iloc[ci]["bar_end_time"]),confirmed_index_4h=ci))
    lv=pd.DataFrame(rows)
    if len(lv):
      lv["liquidity_id_4h"]=(lv.symbol+"|4h|"+lv.liquidity_type+"|"+lv.available_time.astype(str)+"|"+lv.price.astype(str))
      lv=lv.drop_duplicates("liquidity_id_4h").sort_values("available_time").reset_index(drop=True)
    return lv,five,four,smc

def lifecycle(lv,five,sym):
    if not len(lv): return lv
    t=pd.to_datetime(five.bar_start_time).to_numpy(); tend=t+np.timedelta64(5,"m")
    hi=five.high.to_numpy(float);lo=five.low.to_numpy(float);op=five.open.to_numpy(float);cl=five.close.to_numpy(float)
    disc=discontinuity_flags(sym);rows=[]
    for r in lv.itertuples(index=False):
      start=int(np.searchsorted(t,np.datetime64(r.available_time),side="left")); rec=r._asdict()
      if start>=len(t): rec.update(first_penetration_time=pd.NaT,roll_censored=False);rows.append(rec);continue
      di=np.flatnonzero(disc[start:]);limit=start+int(di[0]) if len(di) else len(t)
      cs=find_contacts(hi,lo,op,cl,start,float(r.price),int(r.side),limit)
      pi=next((int(i) for i,ct in cs if ct!="TOUCH_ONLY"),None)
      rec.update(first_penetration_time=pd.Timestamp(tend[pi]) if pi is not None else pd.NaT,roll_censored=bool(len(di)))
      rows.append(rec)
    out=pd.DataFrame(rows)
    av=pd.to_datetime(out.available_time);fp=pd.to_datetime(out.first_penetration_time)
    assert (fp.isna()|(av<=fp)).all()
    return out

def field_features(F,levels):
    frames=[]
    for sym,c0 in F.groupby("symbol",sort=False):
      lv=levels[levels.symbol==sym];p=lv.price.to_numpy(float);av=pd.to_datetime(lv.available_time).to_numpy();fp=pd.to_datetime(lv.first_penetration_time).to_numpy()
      for st in range(0,len(c0),CHUNK):
        c=c0.iloc[st:st+CHUNK];dt=pd.to_datetime(c.decision_time).to_numpy();ref=c.entry_reference.to_numpy(float)
        bd=c.liquidity_price.to_numpy(float);atr=c.atr0.to_numpy(float);side=c.side.to_numpy(float)
        active=(av[None,:]<=dt[:,None])&(np.isnat(fp)[None,:]|(fp[None,:]>dt[:,None]))
        dp=(p[None,:]-ref[:,None])/atr[:,None];de=side[:,None]*(p[None,:]-bd[:,None])/atr[:,None]
        def nearest(mask,val):
          z=np.min(np.where(active&mask,val,np.inf),axis=1);return np.where(np.isfinite(z),z,np.nan)
        up=nearest(dp>0,dp);dn=nearest(dp<0,-dp);ahead=nearest(de>0,de);behind=nearest(de<0,-de)
        o=pd.DataFrame(dict(liq4h_nearest_above_R=up,liq4h_nearest_below_R=dn,liq4h_nearest_ahead_R=ahead,liq4h_nearest_behind_R=behind,
          liq4h_prox_up=np.exp(-np.nan_to_num(up,nan=np.inf)),liq4h_prox_down=np.exp(-np.nan_to_num(dn,nan=np.inf)),
          liq4h_prox_ahead=np.exp(-np.nan_to_num(ahead,nan=np.inf)),liq4h_prox_behind=np.exp(-np.nan_to_num(behind,nan=np.inf)),
          liq4h_active_count_log1p=np.log1p(active.sum(axis=1))))
        for tag,lam in LAMS.items():
          w=np.where(active,np.exp(-np.abs(dp)/lam),0.);u=(w*(dp>0)).sum(1);d=(w*(dp<0)).sum(1)
          o[f"liq4h_density_{tag}"]=u+d;o[f"liq4h_imbalance_{tag}"]=(u-d)/(u+d+1e-12)
        for k in KEY:o[k]=c[k].to_numpy()
        frames.append(o)
    return pd.concat(frames,ignore_index=True)

def overlap(levels,masters):
    rows=[]
    for sym,g in levels.groupby("symbol"):
      old=masters[sym];op=old.price.to_numpy(float);oa=pd.to_datetime(old.available_time).to_numpy()
      for r in g.itertuples(index=False):
        hit=np.isclose(op,r.price,rtol=0,atol=1e-12)&(oa<=np.datetime64(r.available_time))
        rows.append(dict(symbol=sym,liquidity_id_4h=r.liquidity_id_4h,liquidity_type=r.liquidity_type,preexisting_same_price=bool(hit.any())))
    return pd.DataFrame(rows)

def causal_tests(levels,four_by):
    rng=np.random.default_rng(20260912);sample=levels.iloc[rng.choice(len(levels),size=min(200,len(levels)),replace=False)]
    av_ok=prefix_ok=0
    for r in sample.itertuples(index=False):
      four=four_by[r.symbol];av_ok+=pd.Timestamp(four.iloc[r.confirmed_index_4h].bar_end_time)==pd.Timestamp(r.available_time)
      smc=build_full_ob_smc_tf(four.iloc[:r.confirmed_index_4h+1].copy());typ={"CONFIRMED_SWING_HIGH":"swing_high","CONFIRMED_SWING_LOW":"swing_low","CANONICAL_EQH":"EQH","CANONICAL_EQL":"EQL"}[r.liquidity_type]
      pool=smc.get("equal_highs_lows",[]) if typ in ("EQH","EQL") else smc.get("pivots",[])
      prefix_ok+=any(q.get("type")==typ and int(q["confirmed_index"])==r.confirmed_index_4h and np.isclose(float(q["level"]),r.price,rtol=0,atol=1e-12) for q in pool)
    assert av_ok==len(sample) and prefix_ok==len(sample),"STOP_4H_CAUSALITY_FAIL"
    return dict(n=len(sample),available_time_pass=av_ok,prefix_rebuild_pass=prefix_ok)

def orient(state,field):
    x=state.merge(field,on=KEY,validate="many_to_one");d=x["d"].to_numpy(float) if "d" in x else None
    return x

def fit_class(tr,te,feats):
    X=tr[feats].to_numpy(float);Z=te[feats].to_numpy(float);X[~np.isfinite(X)]=0;Z[~np.isfinite(Z)]=0
    sc=StandardScaler().fit(X);m=LogisticRegression(max_iter=200,C=1.).fit(sc.transform(X),tr.target_first.astype(int))
    p=m.predict_proba(sc.transform(Z))[:,1];y=te.target_first.astype(int)
    return roc_auc_score(y,p),log_loss(y,p),brier_score_loss(y,p)

def action_models(df,gmap,state,field):
    ev=df[df.available&~df.ambiguous&~df.censored].merge(gmap,on="gid")
    ev["target_distance_atr"]=ev.target_atr;ev["risk_distance_atr"]=ev.risk_atr;ev["structure_scale"]=ev.scale;ev["action_is_outward"]=(ev.action=="OUTWARD").astype(float)
    st=state.merge(field,on=KEY,validate="many_to_one");rows=[]
    for wf in WFS:
      sub=ev[ev.wf.isin(set(s4b.TRAIN_WF_OF[wf])|{wf})].merge(st[s4b.STATE_KEY+s4b.REACTION_FEATURES+s4b.LIQUIDITY_FEATURES+F4RAW],on=s4b.STATE_KEY)
      for tag in LAMS:sub[f"liq4h_oriented_imbalance_{tag}"]=sub.d*sub[f"liq4h_imbalance_{tag}"]
      sub["liq4h_profit_prox"]=np.where(sub.d>0,sub.liq4h_prox_up,sub.liq4h_prox_down);sub["liq4h_adverse_prox"]=np.where(sub.d>0,sub.liq4h_prox_down,sub.liq4h_prox_up)
      tr=sub[sub.wf.isin(s4b.TRAIN_WF_OF[wf])];te=sub[sub.wf==wf]
      specs={"M0":s4b.GEO_COLS,"M1":s4b.GEO_COLS+s4b.LIQUIDITY_FEATURES,"M2":s4b.GEO_COLS+s4b.LIQUIDITY_FEATURES+s4b.REACTION_FEATURES,"M3":s4b.GEO_COLS+s4b.LIQUIDITY_FEATURES+s4b.REACTION_FEATURES+F4,"M1_PLUS_4H":s4b.GEO_COLS+s4b.LIQUIDITY_FEATURES+F4}
      got={k:fit_class(tr,te,v) for k,v in specs.items()}
      for k,v in got.items():rows.append(dict(wf=wf,model=k,n_train=len(tr),n_test=len(te),auc=v[0],log_loss=v[1],brier=v[2]))
    info=pd.DataFrame(rows);out=[]
    for wf in WFS:
      z=info[info.wf==wf].set_index("model");out.append(dict(wf=wf,M2_AUC=z.loc["M2","auc"],M3_AUC=z.loc["M3","auc"],delta_AUC=z.loc["M3","auc"]-z.loc["M2","auc"],delta_LogLoss=z.loc["M3","log_loss"]-z.loc["M2","log_loss"],delta_Brier=z.loc["M3","brier"]-z.loc["M2","brier"],secondary_delta_AUC=z.loc["M1_PLUS_4H","auc"]-z.loc["M1","auc"],M0_AUC=z.loc["M0","auc"]))
    return info,pd.DataFrame(out)

def wait_models(df,gmap,state,field):
    _,w=s4b.s4a._matched_waiting(df,gmap,"first");w["action_is_outward"]=(w.action=="OUTWARD").astype(float);w=w.rename(columns={"base_h":"h"})
    st=state.merge(field,on=KEY,validate="many_to_one");rows=[]
    for wf in WFS:
      sub=w[w.wf.isin(set(s4b.TRAIN_WF_OF[wf])|{wf})].merge(st[s4b.STATE_KEY+s4b.REACTION_FEATURES+s4b.LIQUIDITY_FEATURES+F4RAW],on=s4b.STATE_KEY).rename(columns={"h":"base_h"})
      for tag in LAMS:sub[f"liq4h_oriented_imbalance_{tag}"]=sub.d*sub[f"liq4h_imbalance_{tag}"]
      sub["liq4h_profit_prox"]=np.where(sub.d>0,sub.liq4h_prox_up,sub.liq4h_prox_down);sub["liq4h_adverse_prox"]=np.where(sub.d>0,sub.liq4h_prox_down,sub.liq4h_prox_up)
      tr=sub[sub.wf.isin(s4b.TRAIN_WF_OF[wf])];te=sub[sub.wf==wf];ytr=tr.delta_E_R_lower.to_numpy(float);yte=te.delta_E_R_lower.to_numpy(float);a=np.isfinite(ytr);b=np.isfinite(yte)
      vals=[]
      for name,feats in {"W0":s4b.WAIT_GEO+s4b.LIQUIDITY_FEATURES+s4b.REACTION_FEATURES,"W1":s4b.WAIT_GEO+s4b.LIQUIDITY_FEATURES+s4b.REACTION_FEATURES+F4}.items():
        X=tr[feats].to_numpy(float)[a];Z=te[feats].to_numpy(float)[b];X[~np.isfinite(X)]=0;Z[~np.isfinite(Z)]=0;sc=StandardScaler().fit(X);m=Ridge(alpha=1.).fit(sc.transform(X),ytr[a]);p=m.predict(sc.transform(Z));vals.append((name,r2_score(yte[b],p),spearmanr(yte[b],p).statistic,mean_absolute_error(yte[b],p)))
      q={x[0]:x[1:] for x in vals};rows.append(dict(wf=wf,n_train=int(a.sum()),n_test=int(b.sum()),W0_R2=q["W0"][0],W1_R2=q["W1"][0],delta_R2=q["W1"][0]-q["W0"][0],W0_Spearman=q["W0"][1],W1_Spearman=q["W1"][1],delta_Spearman=q["W1"][1]-q["W0"][1],W0_MAE=q["W0"][2],W1_MAE=q["W1"][2]))
    return pd.DataFrame(rows)

WFS=["WF1","WF2","WF3"]
def main():
    OUT.mkdir(parents=True,exist_ok=True);t0=time.perf_counter();tim={};D,masters,bars=s4b.s4a.load_env()
    ts=time.perf_counter();df,gmap=s4b.compute_action_surface_all_blocks(D,masters,bars);repro=s4b.assert_reproduce_stage4a(df);state,join=s4b.build_state_table(D,df,gmap);tim["baseline_seconds"]=time.perf_counter()-ts
    levels=[];four_by={};counts=[];ts=time.perf_counter()
    for sym in sorted(masters):
      lv,f,four,smc=build_levels(sym);four_by[sym]=four;counts += [dict(symbol=sym,component_1h_count=int(k),n_bars=int(v)) for k,v in four.component_1h_count.value_counts().items()];levels.append(lifecycle(lv,f,sym))
    levels=pd.concat(levels,ignore_index=True);tim["aggregation_smc_lifecycle_seconds"]=time.perf_counter()-ts
    causal=causal_tests(levels,four_by);ov=overlap(levels,masters);ts=time.perf_counter();field=field_features(D["F"],levels);tim["field_seconds"]=time.perf_counter()-ts
    field.to_parquet(OUT/"liquidity_4h_field_v1.parquet",index=False);levels.to_parquet(OUT/"liquidity_4h_master_v1.parquet",index=False)
    ts=time.perf_counter();ainfo,action=action_models(df,gmap,state,field);tim["action_model_seconds"]=time.perf_counter()-ts
    old=pd.read_csv(s4b.OUT/"latent_action_information_by_wf.csv");ref=old[old.model=="M0"].set_index("wf").roc_auc
    assert max(abs(action.set_index("wf").M0_AUC-ref))<1e-12,"STOP_4H_BASELINE_REPRODUCTION_FAIL"
    ts=time.perf_counter();wait=wait_models(df,gmap,state,field);tim["wait_model_seconds"]=time.perf_counter()-ts
    da=action.delta_AUC;dr=wait.delta_R2;action_keep=((da>=.005).sum()>=2 and da.mean()>0 and (da>=-.002).all());wait_keep=((dr>=.002).sum()>=2 and dr.mean()>0 and (wait.delta_Spearman>=0).sum()>=2)
    no=((da.mean()<.002) and not (da>=.005).any() and dr.mean()<.001 and not (dr>=.002).any());verdict="KEEP_4H_FOR_V2_STATE" if action_keep or wait_keep else ("NO_MATERIAL_4H_INCREMENT" if no else "4H_INCREMENT_WEAK_OR_INCONSISTENT")
    summary=levels.groupby(["symbol","liquidity_type"]).size().rename("n_levels").reset_index();summary.to_csv(OUT/"liquidity_4h_level_summary.csv",index=False)
    ov.to_csv(OUT/"liquidity_4h_overlap_audit.csv",index=False);pd.DataFrame(counts).to_csv(OUT/"liquidity_4h_component_count_audit.csv",index=False)
    pd.DataFrame([dict(n_contacts=len(field),active_coverage=float((field.liq4h_active_count_log1p>0).mean()),**{c:float(field[c].mean()) for c in F4 if c in field})]).to_csv(OUT/"liquidity_4h_field_summary.csv",index=False)
    action.to_csv(OUT/"liquidity_4h_action_incremental_by_wf.csv",index=False);wait.to_csv(OUT/"liquidity_4h_wait_incremental_by_wf.csv",index=False)
    tim["total_seconds"]=time.perf_counter()-t0;audit=dict(experiment="V2-A 4h Liquidity Incremental Information Audit",base_commit=BASE,verdict=verdict,n_levels=len(levels),level_types=levels.liquidity_type.value_counts().to_dict(),preexisting_same_price_share=float(ov.preexisting_same_price.mean()),active_coverage=float((field.liq4h_active_count_log1p>0).mean()),causality=causal,stage4a_reproduction=repro,baseline_M0_max_abs_diff=0.,action_gate=bool(action_keep),wait_gate=bool(wait_keep),P1_read=False,policy_modified=False,timings=tim,efficiency_contract=dict(smc_once_per_symbol=True,field_chunk=CHUNK,no_per_contact_smc=True,no_per_model_field_rebuild=True),action=action.to_dict("records"),wait=wait.to_dict("records"))
    (OUT/"LIQUIDITY_4H_INCREMENTAL_AUDIT.json").write_text(json.dumps(audit,indent=2,default=str))
    (OUT/"LIQUIDITY_4H_INCREMENTAL_V1.md").write_text(f"# V2-A 4h Liquidity Incremental Audit\n\n**{verdict}**\n\n4h levels: {len(levels):,}; preexisting exact-price share: {ov.preexisting_same_price.mean():.2%}; active contact coverage: {(field.liq4h_active_count_log1p>0).mean():.2%}.\n\n## Action M3-M2\n\n```\n{action.to_string(index=False)}\n```\n\n## WAIT W1-W0\n\n```\n{wait.to_string(index=False)}\n```\n\nP1 and all policies were untouched.\n")
    print(json.dumps(audit,indent=2,default=str))
if __name__=="__main__":main()
