"""Structural-OB 标签上的 nested metadata vs metadata+state（只跑 Logistic）。

与 run_nested_audit_v1 完全同构，只把标签换成 structural label，
universe 排除 INVALID_AT_DECISION。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

import research.m2_nondeep_temporal_v1 as m2
from research.phase1_native_direction.nd_contract_v1 import DEV4, RESULTS
from research.phase1_native_direction.run_nested_audit_v1 import (
    BOOT, META_ALL, META_ZS, delta, design, fit_log, metrics, paired_boot,
    within_stratum,
)


def main():
    lab = pd.read_parquet(
        RESULTS / "native_structural_labels_v1.parquet")
    X = pd.read_parquet(RESULTS / "native_features_v1.parquet")
    lab["candidate_id"] = lab["candidate_id"].astype(str)
    X["candidate_id"] = X["candidate_id"].astype(str)
    # 去掉与特征表重名的列，避免 merge 产生 _x/_y 后缀
    dup = [c for c in lab.columns
           if c != "candidate_id" and c in set(X.columns)]
    if dup:
        print(f"[struct-nested] drop duplicate cols from labels: {dup}")
        lab = lab.drop(columns=dup)
    d = lab.merge(X, on="candidate_id", how="inner", validate="one_to_one")
    d = d[d["struct_status"] == "RESOLVED"].copy()
    d["y"] = d["struct_label"].astype(int)
    state = [c for c in X.columns if c != "candidate_id"]
    assert not any(c in state for c in
                   ["native_direction", "source_ob_bias", "struct_label"])
    d = d.reset_index(drop=True)
    print(f"[struct-nested] resolved={len(d)} state_dim={len(state)}",
          flush=True)

    folds, _ = m2.build_folds(d["trading_day"].to_numpy(), len(d))
    recs = []
    for fi in range(4):
        tr_d, se_d, te_d = folds[fi]
        tr = d[d["trading_day"].isin(tr_d)]
        se = d[d["trading_day"].isin(se_d)]
        te = d[d["trading_day"].isin(te_d)]
        if len(te) < 100:
            continue
        se_start = se["decision_time"].min()
        tr_ok = tr[tr["resolution_time"] < se_start]
        if len(tr_ok) < 200:
            continue
        te = te.copy()
        for name, meta, um, us in (
            ("M1_meta", META_ALL, True, False),
            ("M3_meta_state", META_ALL, True, True),
        ):
            Xtr, Xte = design(tr_ok, te, meta, state, um, us)
            m = fit_log(Xtr, tr_ok["y"].to_numpy(int))
            te[f"p_{name}"] = m.predict_proba(Xte)[:, 1]
        dtr = tr_ok[tr_ok["symbol"].isin(DEV4)]
        for name, um, us in (("ZS_M1", True, False), ("ZS_M3", True, True)):
            Xtr, Xte = design(dtr, te, META_ZS, state, um, us)
            m = fit_log(Xtr, dtr["y"].to_numpy(int))
            te[f"p_{name}"] = m.predict_proba(Xte)[:, 1]
        te["折"] = f"F{fi+1}"
        recs.append(te)
    P = pd.concat(recs, ignore_index=True)
    P.to_parquet(RESULTS / "structural_nested_predictions.parquet",
                 index=False)

    new = P[~P["symbol"].isin(DEV4)]
    rows = []
    for uni, g in (("ALL15", P), ("NEW11_zero_shot", new)):
        rows.append(dict(universe=uni, 模型="M0_constant",
                         **metrics(g["y"], np.full(len(g), g["y"].mean()))))
        for nm, col in (("M1_metadata", "p_M1_meta"),
                        ("M3_metadata_state", "p_M3_meta_state"),
                        ("ZS_M1", "p_ZS_M1"), ("ZS_M3", "p_ZS_M3")):
            rows.append(dict(universe=uni, 模型=nm,
                             **metrics(g["y"], g[col])))
    md = pd.DataFrame(rows)
    md.to_csv(RESULTS / "structural_nested_main.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== structural nested main ===")
    print(md[["universe", "模型", "n", "AUC", "PR_AUC", "Brier",
              "Lift@20", "Top20_uplift"]].to_string(index=False), flush=True)

    dl = []
    for uni, g in (("ALL15", P), ("NEW11_zero_shot", new)):
        dl.append(dict(universe=uni, comparison="M3 - M1", **delta(
            metrics(g["y"], g["p_M3_meta_state"]),
            metrics(g["y"], g["p_M1_meta"]))))
        dl.append(dict(universe=uni, comparison="ZS_M3 - ZS_M1", **delta(
            metrics(g["y"], g["p_ZS_M3"]),
            metrics(g["y"], g["p_ZS_M1"]))))
    ddf = pd.DataFrame(dl)
    ddf.to_csv(RESULTS / "structural_nested_delta.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== 增量 ===")
    print(ddf.to_string(index=False), flush=True)

    ws = []
    for tag, g, col in (("ALL15_M3", P, "p_M3_meta_state"),
                        ("NEW11_ZS_M3", new, "p_ZS_M3")):
        t, agg = within_stratum(g, col)
        ws.append(dict(dataset=tag, **agg))
    wsd = pd.DataFrame(ws)
    wsd.to_csv(RESULTS / "structural_within_stratum.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== within-stratum ===")
    print(wsd.to_string(index=False), flush=True)

    bs = [dict(dataset="ALL15",
               **paired_boot(P, "p_M3_meta_state", "p_M1_meta")),
          dict(dataset="NEW11_zero_shot",
               **paired_boot(new, "p_ZS_M3", "p_ZS_M1"))]
    bd = pd.DataFrame(bs)
    bd.to_csv(RESULTS / "structural_nested_bootstrap.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== paired bootstrap ===")
    print(bd.to_string(index=False), flush=True)

    (RESULTS / "structural_nested_audit.json").write_text(json.dumps(
        dict(main=md.to_dict("records"), delta=ddf.to_dict("records"),
             within=wsd.to_dict("records"), bootstrap=bs),
        indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nSTRUCTURAL_NESTED_DONE")


if __name__ == "__main__":
    main()
