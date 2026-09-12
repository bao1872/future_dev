"""ENTER / SKIP Selection Gate v1.0.

Freeze the four primary target-x-risk regions discovered at commit 8438456 into
a deterministic one-contact/at-most-one-trade sequential policy.  WF1-WF3 are
development replay only: all three participated in region discovery.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env
from research.liquidity_oracle_atlas import run_latent_state_compression_v1 as s4b
from research.liquidity_oracle_atlas.run_liquidity_field_action_surface_v1 import _assign_blocks


OUT = Path("research/analysis_results/enter_skip_selection_v1")
OUT.mkdir(parents=True, exist_ok=True)
BASE_COMMIT = "8438456982e9be6999a57fee75212fe21771f6bf"
COST_R_GRID = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]
TEST_WF = ["WF1", "WF2", "WF3"]
FIELD_KEY = ["symbol", "liquidity_id", "contact_number"]

PRIMARY_REGIONS = [
    dict(region="R1", action="OUTWARD", h=2, scale=1.2,
         target_lo=3.0, target_hi=5.0, risk_lo=1.0, risk_hi=2.0),
    dict(region="R2", action="OUTWARD", h=5, scale=1.2,
         target_lo=0.5, target_hi=1.0, risk_lo=1.0, risk_hi=2.0),
    dict(region="R3", action="INWARD", h=5, scale=1.2,
         target_lo=0.5, target_hi=1.0, risk_lo=1.0, risk_hi=2.0),
    dict(region="R4", action="OUTWARD", h=8, scale=1.2,
         target_lo=0.5, target_hi=1.0, risk_lo=1.0, risk_hi=2.0),
]


def _canonical_sha(obj) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def _in_region(d: pd.DataFrame, r: dict) -> pd.Series:
    return (d["available"] & (d["action"] == r["action"])
            & (d["h"] == r["h"]) & np.isclose(d["scale"], r["scale"])
            & d["target_atr"].ge(r["target_lo"])
            & d["target_atr"].lt(r["target_hi"])
            & d["risk_atr"].ge(r["risk_lo"])
            & d["risk_atr"].lt(r["risk_hi"]))


def primary_matches(surface: pd.DataFrame) -> pd.DataFrame:
    pieces = []
    for r in PRIMARY_REGIONS:
        x = surface[_in_region(surface, r)].copy()
        x["region"] = r["region"]
        pieces.append(x)
    out = pd.concat(pieces, ignore_index=True)
    assert not out.duplicated(["gid", "region"]).any(), "REGION_MATCH_NOT_UNIQUE"
    return out


def build_policy(surface: pd.DataFrame, matches: pd.DataFrame,
                 contacts: pd.DataFrame) -> pd.DataFrame:
    """R1 at h2; R2/R3 conflict-or-enter at h5; R4 at h8; max one trade."""
    by = {(int(g), r): x.iloc[0] for (g, r), x in
          matches.groupby(["gid", "region"], sort=False)}
    rows = []
    for c in contacts.itertuples(index=False):
        gid = int(c.gid)
        hit = [r for r in ["R1", "R2", "R3", "R4"] if (gid, r) in by]
        if "R1" in hit:
            selected, decision = by[(gid, "R1")], "ENTER_R1"
        elif "R2" in hit and "R3" in hit:
            selected, decision = None, "SKIP_DUAL_ACTION_CONFLICT"
        elif "R2" in hit:
            selected, decision = by[(gid, "R2")], "ENTER_R2"
        elif "R3" in hit:
            selected, decision = by[(gid, "R3")], "ENTER_R3"
        elif "R4" in hit:
            selected, decision = by[(gid, "R4")], "ENTER_R4"
        else:
            selected, decision = None, "SKIP_NO_GEOMETRY_EDGE"
        row = dict(gid=gid, wf=c.wf, symbol=c.symbol,
                   liquidity_id=c.liquidity_id,
                   contact_number=int(c.contact_number),
                   contact_bar_index=int(c.contact_bar_index),
                   decision_time=c.decision_time, decision=decision,
                   matched_regions="|".join(hit), n_region_matches=len(hit),
                   entered=selected is not None)
        if selected is not None:
            for k in ["region", "action", "h", "scale", "target_atr", "risk_atr",
                      "rr", "target_first", "stop_first", "ambiguous", "censored",
                      "R_lower", "R_upper"]:
                row[k] = selected[k]
        rows.append(row)
    out = pd.DataFrame(rows)
    assert not out.duplicated("gid").any(), "POLICY_CONTACT_NOT_UNIQUE"
    assert out.groupby("gid")["entered"].sum().max() <= 1, "MULTIPLE_ENTRY"
    return out


def _metrics(g: pd.DataFrame) -> dict:
    tr = g[g["entered"]].copy()
    resolved = tr[tr["R_lower"].notna()]
    r = resolved["R_lower"].astype(float)
    pos, neg = r[r > 0].sum(), -r[r < 0].sum()
    time_col = ("execution_decision_time" if "execution_decision_time" in resolved
                and resolved["execution_decision_time"].notna().all()
                else "decision_time")
    ordered = resolved.sort_values([time_col, "symbol", "gid"])
    cum = ordered["R_lower"].cumsum()
    dd = cum - cum.cummax()
    worst = tr["R_lower"].fillna(-1.0)
    return dict(
        n_contacts=int(len(g)), n_trades=int(len(tr)),
        trade_rate=float(len(tr) / len(g)) if len(g) else np.nan,
        resolved_n=int(len(resolved)), censored_n=int(tr["R_lower"].isna().sum()),
        censored_rate=float(tr["R_lower"].isna().mean()) if len(tr) else np.nan,
        mean_R_lower=float(r.mean()) if len(r) else np.nan,
        median_R_lower=float(r.median()) if len(r) else np.nan,
        profit_factor_lower=float(pos / neg) if neg > 0 else np.nan,
        win_rate=float((r > 0).mean()) if len(r) else np.nan,
        total_R_lower=float(r.sum()),
        ambiguity_rate=float(tr["ambiguous"].mean()) if len(tr) else np.nan,
        E_R_lower_resolved=float(r.mean()) if len(r) else np.nan,
        E_R_lower_censor_worst=float(worst.mean()) if len(worst) else np.nan,
        max_drawdown_R=float(dd.min()) if len(dd) else np.nan,
        final_cumulative_R=float(cum.iloc[-1]) if len(cum) else np.nan)


def development_replay(policy: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wf in TEST_WF:
        row = dict(scope="DEVELOPMENT_REPLAY", wf=wf)
        row.update(_metrics(policy[policy["wf"] == wf]))
        rows.append(row)
    return pd.DataFrame(rows)


def contribution_and_cost(policy: pd.DataFrame):
    contrib, costs = [], []
    tr = policy[policy["entered"]]
    scopes = [("WHOLE_POLICY", "ALL", tr)]
    scopes += [("WHOLE_POLICY", wf, tr[tr["wf"] == wf]) for wf in TEST_WF]
    for region in [r["region"] for r in PRIMARY_REGIONS]:
        rg = tr[tr["region"] == region]
        scopes += [(region, wf, rg[rg["wf"] == wf]) for wf in TEST_WF]
    for name, wf, g in scopes:
        resolved = g[g["R_lower"].notna()]
        base = _metrics(pd.concat([g], ignore_index=True).assign(entered=True))
        contrib.append(dict(region=name, wf=wf, **base))
        gross_l = float(resolved["R_lower"].mean()) if len(resolved) else np.nan
        gross_u = float(resolved["R_upper"].mean()) if len(resolved) else np.nan
        for cost in COST_R_GRID:
            costs.append(dict(
                region=name, wf=wf, cost_R=cost, n=int(len(g)),
                resolved_n=int(len(resolved)), gross_E_R_lower=gross_l,
                net_E_R_lower=gross_l - cost, gross_E_R_upper=gross_u,
                net_E_R_upper=gross_u - cost,
                target_first_rate=float(g["target_first"].mean()) if len(g) else np.nan,
                stop_first_rate=float(g["stop_first"].mean()) if len(g) else np.nan,
                ambiguous_rate=float(g["ambiguous"].mean()) if len(g) else np.nan,
                censored_rate=float(g["censored"].mean()) if len(g) else np.nan,
                break_even_cost_R=gross_l))
    return pd.DataFrame(contrib), pd.DataFrame(costs)


def overlap_audit(policy: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for wf in ["ALL"] + TEST_WF:
        p = policy if wf == "ALL" else policy[policy["wf"] == wf]
        m = matches if wf == "ALL" else matches[matches["wf"] == wf]
        rows += [dict(row_type="summary", wf=wf, item=k, value=v) for k, v in {
            "n_contacts": len(p), "n_enter": int(p.entered.sum()),
            "n_skip": int((~p.entered).sum()), "enter_rate": float(p.entered.mean()),
            "n_R1": int((m.region == "R1").sum()), "n_R2": int((m.region == "R2").sum()),
            "n_R3": int((m.region == "R3").sum()), "n_R4": int((m.region == "R4").sum()),
            "n_multiple_region_match": int((p.n_region_matches > 1).sum()),
            "n_dual_action_conflict": int((p.decision == "SKIP_DUAL_ACTION_CONFLICT").sum()),
            "n_earlier_region_supersedes_later": int(
                ((p.matched_regions.str.contains("R1")
                  & p.matched_regions.str.contains("R[234]", regex=True))
                 | (p.matched_regions.str.contains("R[23]", regex=True)
                    & p.matched_regions.str.contains("R4"))).sum()),
            "unique_contact_coverage": int(p.entered.sum()),
        }.items()]
        sets = {r: set(m.loc[m.region == r, "gid"]) for r in ["R1", "R2", "R3", "R4"]}
        for a, b in combinations(sets, 2):
            rows.append(dict(row_type="region_pair", wf=wf, item=f"{a}&{b}",
                             value=len(sets[a] & sets[b])))
    return pd.DataFrame(rows)


def auxiliary_shadow(surface: pd.DataFrame, primary_policy: pd.DataFrame,
                     frontier_regions: pd.DataFrame) -> pd.DataFrame:
    aux = frontier_regions[(frontier_regions["plane"] == "rr_x_target")
                           & frontier_regions["robust_positive_geometry"]]
    first = surface[surface["available"]].copy()
    key = ["gid", "wf", "action", "scale"]
    first["first_h"] = first.groupby(key, sort=False)["h"].transform("min")
    first = first[first["h"] == first["first_h"]]
    pieces = []
    for i, r in aux.reset_index(drop=True).iterrows():
        rr_lo, rr_hi = {"0.5-0.75": (0.5, .75), "1-1.5": (1., 1.5)}[r.rr_bin]
        t_lo, t_hi = {"0.5-1": (.5, 1.), "2-3": (2., 3.)}[r.target_bin]
        x = first[(first.action == r.action) & (first.h == r.first_h)
                  & np.isclose(first.scale, r.scale) & first.rr.ge(rr_lo)
                  & first.rr.lt(rr_hi) & first.target_atr.ge(t_lo)
                  & first.target_atr.lt(t_hi)].copy()
        x["aux_region"] = f"A{i+1}"
        pieces.append(x)
    m = pd.concat(pieces, ignore_index=True)
    primary_ids = set(primary_policy.loc[primary_policy.entered, "gid"])
    rows = []
    for (region, wf), g in m.groupby(["aux_region", "wf"], sort=False):
        resolved = g[g.R_lower.notna()]
        gross = float(resolved.R_lower.mean()) if len(resolved) else np.nan
        for cost in COST_R_GRID:
            rows.append(dict(aux_region=region, wf=wf, cost_R=cost,
                             n_rows=len(g), unique_contacts=g.gid.nunique(),
                             overlap_with_primary_contacts=len(set(g.gid) & primary_ids),
                             gross_E_R_lower=gross, net_E_R_lower=gross-cost))
    # Contact-level four-way coverage, repeated as explicit summary rows.
    aux_ids, all_ids = set(m.gid), set(primary_policy.gid)
    for item, n in {"primary_only": len(primary_ids-aux_ids),
                    "auxiliary_only": len(aux_ids-primary_ids),
                    "both": len(primary_ids & aux_ids),
                    "neither": len(all_ids-(primary_ids | aux_ids))}.items():
        rows.append(dict(aux_region="COVERAGE", wf="ALL", cost_R=np.nan,
                         n_rows=n, unique_contacts=n,
                         overlap_with_primary_contacts=np.nan,
                         gross_E_R_lower=np.nan, net_E_R_lower=np.nan,
                         coverage_class=item))
    return pd.DataFrame(rows)


def main():
    t0 = time.perf_counter()
    D, master_by_sym, bars_by_sym = load_env()
    surface, gmap = s4b.compute_action_surface_all_blocks(D, master_by_sym, bars_by_sym)
    reproduction = s4b.assert_reproduce_stage4a(surface)
    surface = surface[surface["wf"].isin(TEST_WF)].copy()
    F = D["F"].copy()
    bm = _assign_blocks(F)
    F["block"] = pd.to_datetime(F.decision_time).dt.normalize().map(bm)
    F["wf"] = F.block.map({"TB2": "WF1", "TB3": "WF2", "TB4": "WF3"})
    contact_meta = gmap.merge(F[FIELD_KEY + ["decision_time", "contact_bar_index"]], on=FIELD_KEY,
                              how="left", validate="one_to_one")
    contact_meta = contact_meta.merge(surface[["gid", "wf"]].drop_duplicates(),
                                      on="gid", how="inner", validate="one_to_one")
    assert contact_meta.decision_time.notna().all(), "DECISION_TIME_JOIN_FAILED"
    matches = primary_matches(surface)
    policy = build_policy(surface, matches, contact_meta)
    policy["execution_decision_time"] = pd.NaT
    entered_idx = policy.index[policy.entered]
    for i in entered_idx:
        r = policy.loc[i]
        bar_i = int(r.contact_bar_index) + 1 + int(r.h)
        policy.at[i, "execution_decision_time"] = pd.Timestamp(
            bars_by_sym[r.symbol]["t"][bar_i])
    assert policy.loc[policy.entered, "execution_decision_time"].notna().all(), \
        "EXECUTION_DECISION_TIME_MISSING"
    overlap = overlap_audit(policy, matches)
    replay = development_replay(policy)
    contribution, costs = contribution_and_cost(policy)
    frontier = pd.read_csv("research/analysis_results/geometry_economic_frontier_v1/geometry_frontier_robust_regions.csv")
    auxiliary = auxiliary_shadow(surface, policy, frontier)

    # Four equal chronological blocks exhaust the current frozen contact data.
    block_ranges = F.groupby("block").decision_time.agg(["min", "max", "count"])
    latest_block = str(block_ranges.index[-1])
    untouched = bool((pd.to_datetime(F.decision_time) > pd.to_datetime(
        block_ranges.loc["TB4", "max"])).any())
    region_sha = _canonical_sha(PRIMARY_REGIONS)
    policy_definition = dict(order=["h2:R1", "h5:R2/R3 conflict", "h8:R4"],
                             max_trades_per_contact=1,
                             dual_action="SKIP_DUAL_ACTION_CONFLICT",
                             no_match="SKIP_NO_GEOMETRY_EDGE")
    policy_sha = _canonical_sha(dict(regions=PRIMARY_REGIONS, policy=policy_definition))
    manifest = dict(experiment="ENTER / SKIP Selection Gate v1.0",
                    authoritative_base_commit=BASE_COMMIT,
                    status="FROZEN_CANDIDATE_POLICY",
                    primary_regions=PRIMARY_REGIONS, auxiliary_plane="SHADOW_ONLY",
                    policy_definition=policy_definition, cost_R_grid=COST_R_GRID,
                    region_definition_sha=region_sha, policy_sha=policy_sha,
                    forbidden=["bin search", "ML", "new features", "DP", "RL"])
    prospective = dict(
        verdict="NO_UNTOUCHED_PROSPECTIVE_BLOCK", exists=False,
        reason="TB1-TB4 are four equal chronological partitions of all current frozen contacts",
        current_data_start=str(pd.to_datetime(F.decision_time).min()),
        current_data_end=str(pd.to_datetime(F.decision_time).max()),
        latest_block=latest_block,
        blocks={b: {"start_time": str(x["min"]), "end_time": str(x["max"]),
                    "n_contacts": int(x["count"])} for b, x in block_ranges.iterrows()},
        policy_sha=policy_sha, region_definition_sha=region_sha)
    whole = contribution[(contribution.region == "WHOLE_POLICY")
                         & contribution.wf.isin(TEST_WF)]
    min_gross = float(whole.E_R_lower_resolved.min())
    capacity = max([c for c in COST_R_GRID if min_gross - c > 0], default=None)
    censoring_sensitive = bool((whole.E_R_lower_censor_worst <= 0).any())
    audit = dict(
        experiment=manifest["experiment"], base_commit=BASE_COMMIT,
        stage4a_reproduction=reproduction, n_surface_rows=len(surface),
        n_contacts=len(policy), n_enter=int(policy.entered.sum()),
        n_skip=int((~policy.entered).sum()),
        n_dual_action_conflict=int((policy.decision == "SKIP_DUAL_ACTION_CONFLICT").sum()),
        n_multiple_region_match=int((policy.n_region_matches > 1).sum()),
        min_gross_E_R_lower_across_wf=min_gross,
        min_censor_worst_E_R_lower_across_wf=float(
            whole.E_R_lower_censor_worst.min()),
        censoring_sensitive=censoring_sensitive,
        whole_policy_cost_capacity_R=capacity,
        cost_metadata="NO_AUTHORITATIVE_METADATA; generic R-cost sensitivity only",
        prospective_verdict=prospective["verdict"],
        final_verdict="FROZEN_CANDIDATE_POLICY",
        elapsed_seconds=round(time.perf_counter()-t0, 3))

    (OUT / "selection_policy_manifest.json").write_text(json.dumps(manifest, indent=2))
    overlap.to_csv(OUT / "selection_overlap_audit.csv", index=False)
    replay.to_csv(OUT / "selection_development_replay_by_wf.csv", index=False)
    costs.to_csv(OUT / "selection_cost_sensitivity.csv", index=False)
    contribution.to_csv(OUT / "selection_region_contribution.csv", index=False)
    auxiliary.to_csv(OUT / "selection_auxiliary_shadow.csv", index=False)
    (OUT / "prospective_block_audit.json").write_text(json.dumps(prospective, indent=2))
    (OUT / "ENTER_SKIP_SELECTION_AUDIT.json").write_text(json.dumps(audit, indent=2))
    write_report(audit, replay, overlap, prospective, capacity)
    print(json.dumps(audit, indent=2))


def write_report(audit, replay, overlap, prospective, capacity):
    all_summary = overlap[(overlap.row_type == "summary") & (overlap.wf == "ALL")]
    sm = dict(zip(all_summary.item, all_summary.value))
    md = f"""# ENTER / SKIP Selection Gate v1.0

## Verdict

**FROZEN_CANDIDATE_POLICY / NO_UNTOUCHED_PROSPECTIVE_BLOCK**

WF1-WF3 participated in region discovery. Results below are **DEVELOPMENT_REPLAY**,
not OOS or prospective confirmation. The auxiliary RR x target plane is shadow-only.

## Frozen sequential policy

- R1 at h=2 enters OUTWARD; otherwise wait.
- At h=5, R2 enters OUTWARD or R3 enters INWARD; simultaneous match is
  `SKIP_DUAL_ACTION_CONFLICT`.
- At h=8, R4 enters OUTWARD; otherwise `SKIP_NO_GEOMETRY_EDGE`.
- One liquidity contact can enter at most once.

## Answers

1. Primary unique contact coverage: **{int(sm['unique_contact_coverage']):,}** of
   {int(sm['n_contacts']):,} contacts ({sm['enter_rate']:.2%}).
2. Multiple-region matches: **{int(sm['n_multiple_region_match']):,}**;
   dual-action conflicts: **{int(sm['n_dual_action_conflict']):,}**;
   earlier-region supersedes a later match: **{int(sm['n_earlier_region_supersedes_later']):,}**.
3. Whole-policy development-replay minimum gross `E[R]lower` across WF:
   **{audit['min_gross_E_R_lower_across_wf']:+.6f}R**. Under the -1R censor
   stress the minimum is **{audit['min_censor_worst_E_R_lower_across_wf']:+.6f}R**,
   so the result is **CENSORING_SENSITIVE**.
4. Generic cost capacity on the preregistered grid: **{capacity}R**. This is not a
   real transaction-cost estimate; the repository has no authoritative cost metadata.
5. Untouched prospective block: **NO**. Current data ends at
   `{prospective['current_data_end']}` and TB1-TB4 exhaust it.
6. No prospective PASS/FAIL can be issued.
7. Stop at **FROZEN_CANDIDATE_POLICY** until genuinely untouched data exists.

## Development replay by WF

```
{replay.to_string(index=False)}
```

## Censoring

`E_R_lower_resolved` preserves the frozen outcome semantics. The separate
`E_R_lower_censor_worst` stress assigns every censored trade -1R; it does not alter
the stored outcomes.

## STOP

No bin changes, classifier, morphology/liquidity search, Autoencoder, DP, or RL.
"""
    (OUT / "ENTER_SKIP_SELECTION_V1.md").write_text(md)


if __name__ == "__main__":
    main()
