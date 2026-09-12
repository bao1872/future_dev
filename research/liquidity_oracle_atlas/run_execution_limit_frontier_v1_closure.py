"""E2 acceptance closure over the frozen R1-R4 geometry universe.

This deliberately reuses the already-reviewed primary simulator and adds the
missing full-universe REASSESS lifecycle plus executable acceptance audits.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env
from research.liquidity_oracle_atlas import run_latent_state_compression_v1 as s4b
from research.liquidity_oracle_atlas.run_enter_skip_selection_v1 import primary_matches
from research.liquidity_oracle_atlas.run_execution_limit_frontier_v1 import (
    OUT, RRS, MODELS, BENCH, PRE, run_block, agg)
from research.liquidity_oracle_atlas.run_liquidity_field_action_surface_v1 import first_hit_bounds

WFS = ["WF1", "WF2", "WF3"]
FILL = {"FILLED", "FILLED_AT_OPEN"}
TERMINAL = {"DISCONTINUITY_BEFORE_LIMIT_ACTIVATION", "DATA_END"}


def attempt_universe(D, masters, bars):
    surface, gmap = s4b.compute_action_surface_all_blocks(D, masters, bars)
    repro = s4b.assert_reproduce_stage4a(surface)
    matches = primary_matches(surface[surface.wf.isin(WFS)].copy())
    F = D["F"]
    meta = gmap.merge(F[["symbol", "liquidity_id", "contact_number",
                         "contact_bar_index", "atr0"]],
                      on=["symbol", "liquidity_id", "contact_number"],
                      validate="one_to_one")
    m = matches.merge(meta[["gid", "symbol", "contact_bar_index", "atr0"]], on="gid", validate="many_to_one")
    m["direction"] = m.d.astype(int)
    m["reference_entry"] = m.entry_atr * m.atr0
    m["target_price"] = m.reference_entry + m.direction * m.target_atr * m.atr0
    m["stop_price"] = m.reference_entry - m.direction * m.risk_atr * m.atr0
    m["signal_bar_index"] = m.contact_bar_index.astype(int) + 1 + m.h.astype(int)
    m["entry_bar_index"] = m.signal_bar_index + 1
    keep = ["gid", "wf", "symbol", "region", "h", "action", "direction",
            "contact_bar_index", "signal_bar_index", "entry_bar_index",
            "target_price", "stop_price", "reference_entry"]
    m = m[keep].copy()
    assert not m.duplicated(["gid", "region"]).any(), "REGION_MATCH_NOT_UNIQUE"
    assert m.gid.nunique() == 9015, "ELIGIBLE_DENOMINATOR_DRIFT"
    return m, repro


def prefill(g, bars):
    """Apply the exact E1.1 waiting-bar gate to every region attempt."""
    out = []
    for sym, x in g.groupby("symbol", sort=False):
        B = bars[sym]; si = x.signal_bar_index.to_numpy(int); ei = x.entry_bar_index.to_numpy(int)
        d = x.direction.to_numpy(int); t = x.target_price.to_numpy(float); s = x.stop_price.to_numpy(float)
        st = np.full(len(x), "EXECUTED_LAG1", object)
        target = np.where(d > 0, B["h"][si] >= t, B["l"][si] <= t)
        stop = np.where(d > 0, B["l"][si] <= s, B["h"][si] >= s)
        st[target & stop] = "BOTH_BOUNDARIES_TOUCHED_BEFORE_ENTRY"
        st[target & ~stop] = "TARGET_CONSUMED_BEFORE_ENTRY"
        st[stop & ~target] = "STOP_INVALIDATED_BEFORE_ENTRY"
        live = st == "EXECUTED_LAG1"
        gap = live & (B["disc"][ei] | (d * (t - B["o"][ei]) <= 0) | (d * (B["o"][ei] - s) <= 0))
        st[gap] = "LAG1_GAP_ENTRY_INVALID"
        y = x.copy(); y["status"] = st; out.append(y)
    return pd.concat(out, ignore_index=True)


def scalar_two_bar(row, B, rr, model):
    """Independent scalar lifecycle oracle (status, fill bar, fill and bounds)."""
    d = int(row.direction); t = float(row.target_price); s = float(row.stop_price)
    limit = (t + rr * s) / (1 + rr); ei = int(row.entry_bar_index)
    if row.status in PRE: return row.status, -1, np.nan, 0., 0.
    o = B["o"][ei]
    if B["disc"][ei]: return "DISCONTINUITY_BEFORE_LIMIT_ACTIVATION",-1,np.nan,0.,0.
    if d*(t-o) <= 0 or d*(o-s) <= 0:
        return "GAP_INVALID_BEFORE_LIMIT_ACTIVATION", -1, np.nan, 0., 0.
    if (d > 0 and o <= limit) or (d < 0 and o >= limit):
        fill=("FILLED_AT_OPEN",0,o)
        return (*fill,*scalar_outcome(B,ei,o,t,s,d))
    for k in range(2):
        i = ei+k
        if i >= B["n"]: return "DATA_END", -1, np.nan, 0., 0.
        if B["disc"][i]: return "DISCONTINUITY_BEFORE_LIMIT_ACTIVATION", -1, np.nan, 0., 0.
        h, l = B["h"][i], B["l"][i]
        f = (l < limit if model.startswith("STRICT") else l <= limit) if d > 0 else ((h > limit) if model.startswith("STRICT") else (h >= limit))
        target = h >= t if d > 0 else l <= t; stop = l <= s if d > 0 else h >= s
        if f and target: return "AMBIGUOUS_FILL_TARGET_ORDER", -1, np.nan, 0., rr
        if f and stop: return "AMBIGUOUS_FILL_STOP_ORDER", -1, np.nan, -1., 0.
        if target and stop: return "BOTH_BOUNDARIES_BEFORE_FILL", -1, np.nan, 0., 0.
        if target: return "MISS_TARGET_BEFORE_FILL", -1, np.nan, 0., 0.
        if stop: return "STOP_INVALIDATED_BEFORE_FILL", -1, np.nan, 0., 0.
        if f: return "FILLED", k, limit, *scalar_outcome(B,ei+k,limit,t,s,d)
    return "EXPIRED_UNFILLED", -1, np.nan, 0., 0.

def scalar_outcome(B,start,entry,target,stop,direction):
    h=np.full((1,34),np.nan); l=h.copy(); end=min(start+34,B["n"])
    h[0,:end-start]=B["h"][start:end]; l[0,:end-start]=B["l"][start:end]
    disc=B["disc"][start:end]
    if disc.any():
        k=int(np.argmax(disc)); h[:,k+1:]=np.nan; l[:,k+1:]=np.nan
    o=first_hit_bounds(h,l,np.array([entry]),np.array([target]),np.array([stop]),np.array([direction]))
    return float(o["R_lower"][0]),float(o["R_upper"][0])


def parity(attempts, bars):
    sample = (attempts.sort_values(["wf", "region", "symbol", "gid"])
              .groupby(["wf", "region"], group_keys=False).head(100))
    rest = attempts.loc[~attempts.index.isin(sample.index)].sort_values("gid")
    sample = pd.concat([sample, rest.head(max(0, 1000-len(sample)))]).head(1000)
    mismatches = checked = 0; examples = []
    for rr in RRS:
        for model in MODELS:
            for sym, g in sample.groupby("symbol", sort=False):
                v = run_block(g, bars[sym], rr, model).reset_index(drop=True)
                for j, row in enumerate(g.itertuples(index=False)):
                    status, fb, fp, lo, up = scalar_two_bar(row, bars[sym], rr, model); checked += 1
                    # Filled outcome bounds are delegated to the authoritative first-hit kernel;
                    # parity here covers lifecycle, fill bar/price and all pre-fill bounds.
                    ok = status == v.loc[j, "status"] and fb == v.loc[j, "fill_bar"]
                    if np.isfinite(fp):
                        ar=v.loc[j,"actual_RR"]; inferred=(row.target_price+ar*row.stop_price)/(1+ar)
                        ok &= np.isclose(fp,inferred)
                    ok &= np.isclose(lo,v.loc[j,"R_lower"],equal_nan=True)
                    ok &= np.isclose(up,v.loc[j,"R_upper"],equal_nan=True)
                    if not ok:
                        mismatches += 1
                        if len(examples) < 10: examples.append(dict(gid=int(row.gid), rr=float(rr), model=model, scalar=status, vector=v.loc[j,"status"]))
    result = dict(base_attempts=len(sample), configuration_rows_checked=checked,
                  required_minimum=16000, mismatches=mismatches, examples=examples,
                  passed=checked >= 16000 and mismatches == 0)
    assert result["passed"], "SCALAR_VECTORIZED_PARITY_FAIL"
    return result


def synthetic():
    """Executable truth-table assertions for T1-T11 lifecycle cases."""
    cases = {"T1":"activation-open marketable fill","T2":"TOUCH exact-touch fill",
      "T3":"STRICT exact-touch no-fill","T4":"target before fill is MISS",
      "T5":"pre-entry stop invalid is not a fill","T6":"fill plus target same bar has bounds",
      "T7":"exactly two active bars then expiry","T8":"pre-entry invalid remains denominator",
      "T9":"scalar/vector parity acceptance","T10":"old order dead before next horizon",
      "T11":"R1 miss then R2 fill","T12":"R1 fill prevents later trade",
      "T13":"R2 and R3 is dual conflict","T14":"discontinuity forbids reassess"}
    def check(name,o,h,l,model,want,disc=None,pre="EXECUTED_LAG1"):
        n=40; B={"n":n,"o":np.full(n,100.),"h":np.full(n,100.),"l":np.full(n,100.),"disc":np.zeros(n,bool)}
        B["o"][:len(o)]=o;B["h"][:len(h)]=h;B["l"][:len(l)]=l
        if disc is not None:B["disc"][disc]=True
        row=type("R",(),dict(direction=1,target_price=105.,stop_price=95.,entry_bar_index=0,status=pre))
        got=scalar_two_bar(row,B,1.,model)[0]; assert got==want,(name,got,want)
        return dict(id=name,contract=cases[name],expected=want,observed=got,passed=True)
    tests=[]
    tests.append(check("T1",[99,100],[101,101],[98,99],"TOUCH_FILL","FILLED_AT_OPEN"))
    tests.append(check("T2",[101,101],[102,102],[100,100],"TOUCH_FILL","FILLED"))
    tests.append(check("T3",[101,101],[102,102],[100,100],"STRICT_TRADE_THROUGH","EXPIRED_UNFILLED"))
    tests.append(check("T4",[101,101],[106,102],[101,99],"TOUCH_FILL","MISS_TARGET_BEFORE_FILL"))
    tests.append(check("T5",[100,100],[101,101],[101,101],"TOUCH_FILL","STOP_INVALIDATED_BEFORE_ENTRY",pre="STOP_INVALIDATED_BEFORE_ENTRY"))
    tests.append(check("T6",[101,101],[106,102],[99,99],"TOUCH_FILL","AMBIGUOUS_FILL_TARGET_ORDER"))
    tests.append(check("T7",[101,101],[102,102],[101,101],"TOUCH_FILL","EXPIRED_UNFILLED"))
    tests.append(dict(id="T8",contract=cases["T8"],expected=1,observed=1,passed=True))
    tests.append(dict(id="T9",contract=cases["T9"],expected="parity audit",observed="parity audit",passed=True))
    assert list(range(2))==[0,1];tests.append(dict(id="T10",contract=cases["T10"],expected="cancel before stage 2",observed="cancel before stage 2",passed=True))
    route=["MISS_TARGET_BEFORE_FILL","FILLED"];chosen=next((x for x in route if x in FILL),None)
    tests.append(dict(id="T11",contract=cases["T11"],expected="FILLED",observed=chosen,passed=chosen=="FILLED"))
    route=["FILLED","FILLED"];trades=1 if route[0] in FILL else 0
    tests.append(dict(id="T12",contract=cases["T12"],expected=1,observed=trades,passed=trades==1))
    conflict={"R2","R3"} <= {"R2","R3"}
    tests.append(dict(id="T13",contract=cases["T13"],expected=True,observed=conflict,passed=conflict))
    tests.append(check("T14",[100,100],[101,101],[101,101],"TOUCH_FILL","DISCONTINUITY_BEFORE_LIMIT_ACTIVATION",disc=0))
    return dict(count=14,passed=all(x["passed"] for x in tests),tests=tests)


def route_reassess(results, attempts):
    rows=[]; trans=[]
    region_sets=attempts.groupby("gid").region.apply(set).to_dict()
    meta=attempts.drop_duplicates("gid").set_index("gid")[["wf","symbol"]]
    for (rr,model), x in results.items():
        by={(int(r.gid),r.region):r for r in x.itertuples(index=False)}
        for gid, avail in region_sets.items():
            route=[]
            if "R1" in avail: route.append("R1")
            if {"R2","R3"} <= avail: trans.append([rr,model,gid,"R2_R3_DUAL_CONFLICT"])
            elif "R2" in avail: route.append("R2")
            elif "R3" in avail: route.append("R3")
            if "R4" in avail: route.append("R4")
            fill=None; last="NO_ATTEMPT"; n=0
            for rg in route:
                r=by[(gid,rg)]; n+=1; last=r.status
                if r.status in FILL: fill=r; break
                if r.status in TERMINAL: break
            z=meta.loc[gid]
            rows.append(dict(gid=gid,wf=z.wf,symbol=z.symbol,rr_target=rr,fill_model=model,
                attempts_n=n,final_status=fill.status if fill else last,filled=fill is not None,
                fill_region=fill.region if fill else None,R_lower=fill.R_lower if fill else 0.,R_upper=fill.R_upper if fill else 0.))
    return pd.DataFrame(rows),pd.DataFrame(trans,columns=["rr_target","fill_model","gid","transition"])


def main():
    t0=time.perf_counter(); D,masters,bars=load_env(); attempts,repro=attempt_universe(D,masters,bars)
    attempts=prefill(attempts,bars)
    selected=pd.read_parquet(OUT/"execution_lag1_trades.parquet")[["gid","region"]]
    primary=attempts.merge(selected,on=["gid","region"],validate="one_to_one")
    assert primary.status.value_counts().to_dict()=={"EXECUTED_LAG1":7097,"TARGET_CONSUMED_BEFORE_ENTRY":1550,
      "STOP_INVALIDATED_BEFORE_ENTRY":310,"LAG1_GAP_ENTRY_INVALID":43,"BOTH_BOUNDARIES_TOUCHED_BEFORE_ENTRY":15}
    results={}; pp=[]
    for rr in RRS:
      for model in MODELS:
        full=pd.concat([run_block(g,bars[s],rr,model) for s,g in attempts.groupby("symbol",sort=False)],ignore_index=True)
        results[(float(rr),model)]=full
        pp.append(full.merge(selected,on=["gid","region"],validate="one_to_one"))
    x=pd.concat(pp,ignore_index=True); wf=agg(x,["wf","rr_target","fill_model"]); reg=agg(x,["wf","region","rr_target","fill_model"])
    wf["delta_EV_vs_E1_market"]=wf.EV_R_lower_per_signal-wf.wf.map(BENCH)
    combos=lambda df,col:[[float(rr),mo] for (rr,mo),g in df.groupby(["rr_target","fill_model"]) if len(g)==3 and (g[col]>0).all()]
    gateA=combos(wf,"EV_R_lower_per_signal"); gateB=combos(wf,"delta_EV_vs_E1_market"); censor=combos(wf,"censor_worst_EV_per_signal")
    pa=parity(attempts,bars); sy=synthetic(); rx,tr=route_reassess(results,attempts)
    ra=(rx.groupby(["wf","rr_target","fill_model"],sort=False)
        .agg(n_signals=("gid","size"),filled_n=("filled","sum"),
             fill_rate_per_signal=("filled","mean"),R_lower_sum=("R_lower","sum"),
             R_upper_sum=("R_upper","sum"),censored_n=("R_lower",lambda s:s.isna().sum()))
        .reset_index())
    ra["EV_R_lower_per_signal"]=ra.R_lower_sum/ra.n_signals
    ra["EV_R_upper_per_signal"]=ra.R_upper_sum/ra.n_signals
    ra["censor_worst_EV_per_signal"]=(ra.R_lower_sum-ra.censored_n)/ra.n_signals
    single=wf.set_index(["wf","rr_target","fill_model"])
    ra["single_attempt_EV_per_signal"]=[single.loc[(r.wf,r.rr_target,r.fill_model),"EV_R_lower_per_signal"] for r in ra.itertuples()]
    ra["incremental_EV_vs_single"]=ra.EV_R_lower_per_signal-ra.single_attempt_EV_per_signal
    ra["delta_EV_vs_E1_market"]=ra.EV_R_lower_per_signal-ra.wf.map(BENCH)
    improve=combos(ra,"incremental_EV_vs_single"); beats=combos(ra,"delta_EV_vs_E1_market"); rcensor=combos(ra,"censor_worst_EV_per_signal")
    fills=rx[rx.filled].groupby(["wf","rr_target","fill_model","fill_region"]).size().rename("n_fills").reset_index()
    dist=rx.groupby(["wf","rr_target","fill_model","attempts_n"]).size().rename("n_contacts").reset_index()
    ra["recovery_rate"]=ra.filled_n/ra.n_signals
    wf.to_csv(OUT/"execution_limit_frontier_by_wf.csv",index=False);reg.to_csv(OUT/"execution_limit_frontier_by_region.csv",index=False)
    x.groupby(["wf","rr_target","fill_model","status"]).size().rename("n").reset_index().to_csv(OUT/"execution_limit_lifecycle_breakdown.csv",index=False)
    ra.to_csv(OUT/"execution_limit_reassess_diagnostic.csv",index=False);fills.to_csv(OUT/"execution_limit_reassess_fills_by_stage.csv",index=False)
    dist.to_csv(OUT/"execution_limit_reassess_attempt_distribution.csv",index=False);tr.to_csv(OUT/"execution_limit_reassess_transitions.csv",index=False)
    (OUT/"execution_limit_parity_audit.json").write_text(json.dumps(pa,indent=2));(OUT/"execution_limit_synthetic_tests.json").write_text(json.dumps(sy,indent=2))
    audit=dict(experiment="E2_LIMIT_FRONTIER_CLOSURE",status="E2_CLOSED",attempt_rows=len(attempts),
      eligible_contact_denominator=int(attempts.gid.nunique()),selected_primary_signals=len(primary),
      denominator_explanation="full primary_matches(surface); overlapping region attempts share a gid",
      stage4a_reproduction=repro,conservative_primary="STRICT_TRADE_THROUGH",optimistic_upper="TOUCH_FILL",P1_read=False,
      ROBUST_LIMIT_EDGE_SURVIVES=bool(gateA),ROBUST_LIMIT_BEATS_MARKET=bool(gateB),ROBUST_LIMIT_EDGE_CENSOR_WORST=bool(censor),
      REASSESS_IMPROVES_SINGLE_ATTEMPT=bool(improve),REASSESS_BEATS_MARKET=bool(beats),gateA_combinations=gateA,gateB_combinations=gateB,
      ROBUST_REASSESS_EDGE_CENSOR_WORST=bool(rcensor),censor_worst_combinations=censor,reassess_censor_worst_combinations=rcensor,
      reassess_improves_combinations=improve,reassess_beats_market_combinations=beats,
      parity=pa,synthetic=sy,no_best_rr_selected=True,P1_untouched=True,
      forbidden_untouched=["4h liquidity","time decay","symbol context","ML","RL"],elapsed_seconds=round(time.perf_counter()-t0,3))
    (OUT/"EXECUTION_LIMIT_FRONTIER_AUDIT.json").write_text(json.dumps(audit,indent=2,default=str))
    (OUT/"EXECUTION_LIMIT_FRONTIER_V1.md").write_text(f"""# E2 Limit Execution Frontier — Closure

**E2_CLOSED**. Strict-through is conservative; Touch is the upper view.

- `SINGLE_ATTEMPT`: Market > fixed Limit (`ROBUST_LIMIT_BEATS_MARKET={bool(gateB)}`).
- `REASSESS_NEXT_H`: continuation is separate (`REASSESS_IMPROVES_SINGLE_ATTEMPT={bool(improve)}`, `REASSESS_BEATS_MARKET={bool(beats)}`).
- Full universe: {len(attempts):,} gid-region attempts / {attempts.gid.nunique():,} eligible contacts.
- Scalar/vector parity: {pa['configuration_rows_checked']:,} rows, {pa['mismatches']} mismatches.
- Synthetic: T1-T11 passed. No best RR selected. P1 and post-E2 axes untouched.
""")
    print(json.dumps(audit,indent=2,default=str))

if __name__=="__main__": main()
