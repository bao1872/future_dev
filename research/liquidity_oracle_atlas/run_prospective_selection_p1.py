"""One-shot Prospective Holdout Validation P1.

Two explicit phases:
  prepare  -> isolated download, overlap gate, causal producer replay, boundary
              reproduction, immutable data manifest (no policy outcomes)
  evaluate -> execute commit-27a4786 frozen policy once on the sealed P1 block

P1 is permanently forbidden from training/research after acquisition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from market_data.pytdx_source import download_5m_l8, drop_incomplete_tail
import research.export_ob_trigger_execution_v21 as rawmod
import research.phase1_tradability.phase1_contract_v1 as phase1
import research.liquidity_state_machine.build_liquidity_state_v1 as lsm
import research.liquidity_state_machine.liquidity_specificity_placebo_v2 as spec
import research.liquidity_oracle_atlas.build_liquidity_lifecycle_v1_1 as life
from research.liquidity_oracle_atlas import run_latent_state_compression_v1 as s4b
from research.liquidity_oracle_atlas.run_enter_skip_selection_v1 import (
    COST_R_GRID, PRIMARY_REGIONS, build_policy, primary_matches)

OUT = ROOT / "research/analysis_results/prospective_selection_p1"
HOLD = ROOT / "data/prospective_holdout/P1"
RAW = HOLD / "raw"
NORM = HOLD / "normalized"
ART = HOLD / "manifest"
FROZEN_RAW = ROOT / "research/exports/v3r_5m"
FROZEN_MASTER = ROOT / "research/analysis_results/smc_oracle_atlas_v1/liquidity_master_v1_1.parquet"
CUTOFF = pd.Timestamp("2026-09-04 14:55:00")
POLICY_COMMIT = "27a4786d65dfd95b41c9a58748df24e1e5bacd84"
OVERLAP_START = pd.Timestamp("2026-09-03 00:00:00")
SYMBOLS = ["AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P",
           "RB", "RU", "SC", "SN", "TA"]
KEEP = ["bar_start_time", "bar_end_time", "availability_time", "trading_day",
        "tdx_datetime_raw", "open", "high", "low", "close", "trade", "position"]
PROTOCOL = OUT / "PROSPECTIVE_P1_PROTOCOL.json"
MANIFEST = OUT / "PROSPECTIVE_P1_DATA_MANIFEST.json"
REGISTRY = ROOT / "research/governance/prospective_holdout_registry.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def patch_isolated_loaders():
    rawmod.RAW_5M_ROOT = NORM
    phase1._BAR_CACHE.clear()
    # Imported function objects resolve RAW_5M_ROOT in rawmod's module globals.
    assert rawmod.load_raw_5m.__globals__["RAW_5M_ROOT"] == NORM


def validate_frame(d: pd.DataFrame, sym: str):
    t = pd.to_datetime(d.bar_start_time)
    assert t.is_monotonic_increasing, f"{sym}: NOT_MONOTONIC"
    assert not t.duplicated().any(), f"{sym}: DUPLICATE_TIME"
    assert ((pd.to_datetime(d.bar_end_time)-t).dt.total_seconds() == 300).all()
    for c in ["open", "high", "low", "close"]:
        assert pd.to_numeric(d[c], errors="coerce").notna().all(), f"{sym}: NAN_{c}"
    assert (d.high >= d[["open", "close"]].max(axis=1)).all()
    assert (d.low <= d[["open", "close"]].min(axis=1)).all()
    assert (d.trade >= 0).all() and (d.position >= 0).all()


def continuity(old, new, sym):
    keys = ["bar_start_time"]
    cols = ["open", "high", "low", "close", "trade"]
    a = old[(pd.to_datetime(old.bar_start_time) >= OVERLAP_START)
            & (pd.to_datetime(old.bar_start_time) <= CUTOFF)][keys+cols]
    b = new[(pd.to_datetime(new.bar_start_time) >= OVERLAP_START)
            & (pd.to_datetime(new.bar_start_time) <= CUTOFF)][keys+cols]
    m = a.merge(b, on=keys, suffixes=("_old", "_new"), how="outer", indicator=True)
    both = m._merge.eq("both")
    def rate(cs):
        if not both.any(): return 0.0
        return float(np.logical_and.reduce([
            np.isclose(m.loc[both, f"{c}_old"], m.loc[both, f"{c}_new"],
                       rtol=0, atol=0, equal_nan=True) for c in cs]).mean())
    return dict(symbol=sym, overlap_rows_old=len(a), overlap_rows_new=len(b),
                timestamp_match_rate=float(both.sum()/max(len(m), 1)),
                OHLC_match_rate=rate(["open","high","low","close"]),
                volume_match_rate=rate(["trade"]))


def download_and_seal_raw():
    RAW.mkdir(parents=True, exist_ok=False)
    NORM.mkdir(parents=True, exist_ok=False)
    ART.mkdir(parents=True, exist_ok=False)
    audits, hashes, total_new = [], {}, 0
    for sym in SYMBOLS:
        last_error = None
        for attempt in range(1, 4):
            try:
                # Adapter key is the root symbol; each attempt owns a fresh
                # connection so one timeout cannot poison later instruments.
                fresh = drop_incomplete_tail(download_5m_l8(
                    sym, not_before=OVERLAP_START))
                break
            except Exception as exc:
                last_error = exc
                if attempt == 3:
                    raise
                time.sleep(attempt)
        else:  # defensive: loop either breaks or raises
            raise last_error
        try:
            fresh = fresh[KEEP].copy()
            validate_frame(fresh, sym)
            old = pd.read_csv(FROZEN_RAW/f"{sym}_5m.csv", parse_dates=[
                "bar_start_time","bar_end_time","availability_time","trading_day",
                "tdx_datetime_raw"])
            aud = continuity(old, fresh, sym)
            audits.append(aud)
            if min(aud["timestamp_match_rate"], aud["OHLC_match_rate"],
                   aud["volume_match_rate"]) < 1.0:
                raise SystemExit(f"STOP_PROSPECTIVE_DATA_CONTRACT_MISMATCH: {aud}")
            raw_p = RAW/f"{sym}_overlap_and_p1.parquet"
            fresh.to_parquet(raw_p, index=False)
            hashes[str(raw_p.relative_to(ROOT))] = sha256(raw_p)
            p1 = fresh[pd.to_datetime(fresh.bar_start_time) > CUTOFF]
            total_new += len(p1)
            combined = pd.concat([
                old[pd.to_datetime(old.bar_start_time) <= CUTOFF], p1],
                ignore_index=True).drop_duplicates("bar_start_time", keep="last")
            combined = combined.sort_values("bar_start_time").reset_index(drop=True)
            validate_frame(combined, sym)
            combined.to_csv(NORM/f"{sym}_5m.csv", index=False)
        except Exception:
            # Never resume a partially prepared block; caller discards it.
            raise
    pd.DataFrame(audits).to_json(OUT/"prospective_data_continuity_audit.json",
                                 orient="records", indent=2)
    return hashes, total_new, audits


def rebuild_causal_artifacts():
    patch_isolated_loaders()
    levels, true_res, masters, contacts = [], [], [], []
    for sym in SYMBOLS:
        five, fifteen, oneh, env4 = lsm.prep(sym)
        tick = lsm.infer_tick(five.close.to_numpy(float))
        smcs = {}
        for tf, bars in (("5m",five),("15m",fifteen),("1h",oneh)):
            _, smc = lsm.trend_frame(bars, tf); smcs[tf] = smc
        lv = lsm.liquidity_levels(sym, five, fifteen, oneh, smcs, tick)
        levels.append(lv)
        sd = spec.SymbolData(sym)
        rr = spec.resolve_levels(sd, lv.rename(columns={"liquidity_id":"level_key"}), False)
        true_res.append(rr)
        valid = rr[rr.activation_state == "VALID_AHEAD"]
        m, c = life.build_symbol(sym, valid)
        masters.append(m); contacts.append(c)
        print(f"[CAUSAL] {sym} levels={len(lv)} contacts={len(c)}", flush=True)
    lv = pd.concat(levels, ignore_index=True)
    tr = pd.concat(true_res, ignore_index=True)
    ma = pd.concat(masters, ignore_index=True)
    co = pd.concat(contacts, ignore_index=True)
    lv.to_parquet(ART/"liquidity_levels.parquet", index=False)
    tr.to_parquet(ART/"true_interactions.parquet", index=False)
    ma.to_parquet(ART/"liquidity_master.parquet", index=False)
    co.to_parquet(ART/"liquidity_contacts.parquet", index=False)
    assert (pd.to_datetime(lv.available_time) <= pd.to_datetime(
        lv.available_time)).all()  # explicit producer field existence / parse gate
    return ma, co


def boundary_reproduction(new_master):
    old = pd.read_parquet(FROZEN_MASTER)
    rows, all_pass = [], True
    for sym in SYMBOLS:
        vals = []
        for name, d in (("old",old),("new",new_master)):
            x=d[d.symbol==sym]
            act=x[(pd.to_datetime(x.available_time)<=CUTOFF)
                  & (pd.to_datetime(x.first_penetration_time).isna()
                     | (pd.to_datetime(x.first_penetration_time)>CUTOFF))]
            vals.append((name, act))
        ao, an = vals[0][1], vals[1][1]
        ids_o, ids_n = set(ao.liquidity_id.astype(str)), set(an.liquidity_id.astype(str))
        # Reference price = last frozen close at cutoff.
        bars=pd.read_csv(FROZEN_RAW/f"{sym}_5m.csv")
        b=bars[pd.to_datetime(bars.bar_start_time)<=CUTOFF].iloc[-1]
        px=float(b.close)
        def near(a):
            p=a.price.to_numpy(float); up=p[p>px]; dn=p[p<px]
            return (float(up.min()) if len(up) else None,
                    float(dn.max()) if len(dn) else None)
        ok=(len(ao)==len(an) and ids_o==ids_n and near(ao)==near(an))
        all_pass &= ok
        rows.append(dict(symbol=sym, old_active=len(ao), new_active=len(an),
                         active_ids_match=ids_o==ids_n,
                         old_nearest_above=near(ao)[0],new_nearest_above=near(an)[0],
                         old_nearest_below=near(ao)[1],new_nearest_below=near(an)[1],pass_=ok))
    out=dict(cutoff=str(CUTOFF), all_pass=bool(all_pass), rows=rows)
    (OUT/"prospective_boundary_reproduction.json").write_text(json.dumps(out,indent=2))
    if not all_pass: raise SystemExit("STOP_FROZEN_BOUNDARY_REPRODUCTION_FAIL")
    return out


def prepare():
    if HOLD.exists(): raise SystemExit(f"P1 already exists; refusing overwrite: {HOLD}")
    hashes,total_new,audits=download_and_seal_raw()
    ma,co=rebuild_causal_artifacts()
    boundary=boundary_reproduction(ma)
    p1bars=[]
    for p in RAW.glob("*.parquet"):
        d=pd.read_parquet(p); p1bars.append(d[pd.to_datetime(d.bar_start_time)>CUTOFF])
    allp=pd.concat(p1bars,ignore_index=True)
    manifest=dict(block_id="P1",role="PROSPECTIVE_HOLDOUT",
        start_timestamp=str(pd.to_datetime(allp.bar_start_time).min()),
        end_timestamp=str(pd.to_datetime(allp.bar_start_time).max()),
        raw_file_hashes=hashes,symbols=SYMBOLS,symbol_count=len(SYMBOLS),
        bar_count=int(len(allp)),adapter="market_data.pytdx_source.download_5m_l8",
        adapter_contract="PyTDX 1.72r2 customized; TDX label=interval end",
        policy_commit=POLICY_COMMIT,generated_at=datetime.now().isoformat(timespec="seconds"),
        continuity_all_pass=all(x["timestamp_match_rate"]==1 and x["OHLC_match_rate"]==1
                                and x["volume_match_rate"]==1 for x in audits),
        boundary_reproduction_pass=boundary["all_pass"],sealed=True)
    MANIFEST.write_text(json.dumps(manifest,indent=2))
    reg=json.loads(REGISTRY.read_text()); reg["P1"].update(
        end_inclusive=manifest["end_timestamp"],status="SEALED_NOT_EVALUATED",
        raw_file_hashes=hashes,policy_commit=POLICY_COMMIT)
    REGISTRY.write_text(json.dumps(reg,indent=2))
    print(json.dumps(manifest,indent=2))


def bars_dict(sym):
    patch_isolated_loaders(); raw=rawmod.load_raw_5m(sym)
    disc=phase1.discontinuity_flags(sym)
    return dict(o=raw.open.to_numpy(float),h=raw.high.to_numpy(float),
        l=raw.low.to_numpy(float),c=raw.close.to_numpy(float),
        t=pd.to_datetime(raw.bar_start_time).to_numpy(),
        day=pd.to_datetime(raw.trading_day).to_numpy(),disc=disc,n=len(raw))


def metrics(tr):
    mature=tr[tr.maturity_status=="MATURED"]; resolved=mature[mature.R_lower.notna()]
    r=resolved.R_lower.astype(float); pos=r[r>0].sum(); neg=-r[r<0].sum()
    ordered=resolved.sort_values(["execution_decision_time","symbol","gid"])
    cum=ordered.R_lower.cumsum(); dd=cum-cum.cummax()
    return dict(n_enter=len(tr),matured_trades=len(mature),pending_maturity=int(
        (tr.maturity_status=="PENDING_MATURITY").sum()),resolved_n=len(resolved),
        ambiguous_n=int(mature.ambiguous.sum()),censored_n=int(mature.R_lower.isna().sum()),
        mean_R_lower=float(r.mean()) if len(r) else None,
        median_R_lower=float(r.median()) if len(r) else None,
        profit_factor_lower=float(pos/neg) if neg else None,
        win_rate=float((r>0).mean()) if len(r) else None,total_R_lower=float(r.sum()),
        max_drawdown_R=float(dd.min()) if len(dd) else None,
        E_R_lower_resolved=float(r.mean()) if len(r) else None,
        E_R_lower_censor_worst=float(mature.R_lower.fillna(-1).mean()) if len(mature) else None)


def evaluate():
    manifest=json.loads(MANIFEST.read_text())
    if not manifest.get("sealed"): raise SystemExit("P1_NOT_SEALED")
    reg=json.loads(REGISTRY.read_text())
    if reg["P1"]["status"]!="SEALED_NOT_EVALUATED":
        raise SystemExit("P1 one-shot evaluation already consumed or invalid state")
    patch_isolated_loaders()
    master=pd.read_parquet(ART/"liquidity_master.parquet")
    contacts=pd.read_parquet(ART/"liquidity_contacts.parquet")
    bars={s:bars_dict(s) for s in SYMBOLS}
    D={"F":contacts.copy()}
    surface,gmap=s4b.compute_action_surface_all_blocks(D,
        {s:g.reset_index(drop=True) for s,g in master.groupby("symbol")},bars)
    meta=gmap.merge(contacts[["symbol","liquidity_id","contact_number","decision_time",
        "contact_bar_index"]],on=["symbol","liquidity_id","contact_number"],
        validate="one_to_one")
    pids=set(meta.loc[pd.to_datetime(meta.decision_time)>CUTOFF,"gid"])
    surface=surface[surface.gid.isin(pids)].copy(); surface["wf"]="P1"
    meta=meta[meta.gid.isin(pids)].copy(); meta["wf"]="P1"
    matches=primary_matches(surface); policy=build_policy(surface,matches,meta)
    policy["execution_decision_time"]=pd.NaT; policy["maturity_status"]="NO_ENTRY"
    for i in policy.index[policy.entered]:
        r=policy.loc[i]; ei=int(r.contact_bar_index)+1+int(r.h); b=bars[r.symbol]
        policy.at[i,"execution_decision_time"]=pd.Timestamp(b["t"][ei])
        policy.at[i,"maturity_status"]=("MATURED" if ei+34<=b["n"] else "PENDING_MATURITY")
        if ei+34>b["n"]:
            policy.loc[i,["target_first","stop_first","ambiguous","censored","R_lower","R_upper"]]=np.nan
    tr=policy[policy.entered].copy(); met=metrics(tr)
    met.update(n_contacts=len(policy),trade_rate=len(tr)/len(policy) if len(policy) else None)
    costs=pd.DataFrame([dict(cost_R=c,net_E_R_lower=(met["E_R_lower_resolved"]-c
        if met["E_R_lower_resolved"] is not None else np.nan)) for c in COST_R_GRID])
    regc=[]
    for rg,g in tr[tr.maturity_status=="MATURED"].groupby("region"):
        rr=g[g.R_lower.notna()]; regc.append(dict(region=rg,role="DESCRIPTIVE_ONLY",
            n=len(g),resolved_n=len(rr),E_R_lower=float(rr.R_lower.mean()) if len(rr) else np.nan,
            win_rate=float((rr.R_lower>0).mean()) if len(rr) else np.nan))
    matured=tr[tr.maturity_status=="MATURED"].copy()
    matured["trading_day"]=pd.to_datetime(matured.execution_decision_time).dt.date.astype(str)
    daily=matured.groupby("trading_day").agg(n=("gid","size"),E_R_lower=("R_lower","mean")).reset_index()
    sym=matured.groupby("symbol").agg(n=("gid","size"),E_R_lower=("R_lower","mean")).reset_index()
    conc=dict(unique_trading_days=len(daily),unique_symbols=len(sym),
              largest_day_share=float(daily.n.max()/len(matured)) if len(matured) else None,
              largest_symbol_share=float(sym.n.max()/len(matured)) if len(matured) else None)
    formal=met["resolved_n"]>=200
    verdicts=(["P1_GROSS_PASS" if met["E_R_lower_resolved"]>0 else "P1_GROSS_FAIL",
               "P1_COST_0P01_PASS" if met["E_R_lower_resolved"]-.01>0 else "P1_COST_0P01_FAIL"]
              if formal else ["PROSPECTIVE_P1_EARLY_CHECK","INSUFFICIENT_FOR_FORMAL_VALIDATION"])
    if met["E_R_lower_censor_worst"] is not None and met["E_R_lower_censor_worst"]<=0:
        verdicts.append("CENSORING_SENSITIVE")
    audit=dict(block_id="P1",role="PROSPECTIVE_HOLDOUT",policy_commit=POLICY_COMMIT,
        metrics=met,concentration=conc,resolved_min_gate=200,formal_gate_met=formal,
        verdicts=verdicts,status="BURNED_AND_EVALUATED",adaptive_response="FORBIDDEN")
    tr.to_csv(OUT/"prospective_policy_trades.csv",index=False)
    pd.DataFrame([met]).to_csv(OUT/"prospective_policy_summary.csv",index=False)
    costs.to_csv(OUT/"prospective_cost_sensitivity.csv",index=False)
    pd.DataFrame(regc).to_csv(OUT/"prospective_region_contribution.csv",index=False)
    daily.to_csv(OUT/"prospective_daily_summary.csv",index=False)
    sym.to_csv(OUT/"prospective_symbol_summary.csv",index=False)
    (OUT/"PROSPECTIVE_P1_AUDIT.json").write_text(json.dumps(audit,indent=2))
    (OUT/"PROSPECTIVE_P1_REPORT.md").write_text(report(manifest,audit,costs))
    reg["P1"]["status"]="BURNED_AND_EVALUATED"; reg["P1"]["evaluated_at"]=datetime.now().isoformat(timespec="seconds")
    REGISTRY.write_text(json.dumps(reg,indent=2))
    print(json.dumps(audit,indent=2))


def report(manifest,audit,costs):
    m=audit["metrics"]; c01=float(costs.loc[costs.cost_R==.01,"net_E_R_lower"].iloc[0])
    return f"""# Prospective Holdout Validation P1

**Status: BURNED_AND_EVALUATED**

1. P1 range: `{manifest['start_timestamp']}` to `{manifest['end_timestamp']}`.
2. Adapter continuity: **PASS**.
3. Frozen-boundary reproduction: **PASS**.
4. Contacts `{m['n_contacts']:,}`; ENTER `{m['n_enter']:,}`; matured trades `{m['matured_trades']:,}`; pending `{m['pending_maturity']:,}`.
5. Gross resolved E[R]lower: `{m['E_R_lower_resolved']}`.
6. Net at 0.01R: `{c01}`.
7. Censor-worst: `{m['E_R_lower_censor_worst']}`.
8. Profit factor `{m['profit_factor_lower']}`; win rate `{m['win_rate']}`; max drawdown `{m['max_drawdown_R']}`.
9. Resolved `{m['resolved_n']}`; >=200 gate: `{audit['formal_gate_met']}`.
10. Verdict: `{' / '.join(audit['verdicts'])}`.
11. P1 is permanently `BURNED_AND_EVALUATED` and forbidden from research/training.
12. **STOP**. No policy adaptation or optimization was performed.
"""


if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("phase",choices=["prepare","evaluate"])
    args=ap.parse_args(); prepare() if args.phase=="prepare" else evaluate()
