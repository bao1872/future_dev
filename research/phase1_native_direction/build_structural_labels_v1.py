"""Structural-OB label：用 OB 自己的失效边界定义风险，而不是 ATR5。

    R_struct = |close - far_edge|          （方向化后 > 0 表示 OB 仍有效）
    target   = close + d * 2.5 * R_struct
    stop     = close - d * 1.0 * R_struct == far_edge   （即打穿失效边）

因为 stop 恰等于 far_edge，本标签在数学上与既有 first-passage 状态机
完全同构，只是把 r_ref 从 ATR5 换成 R_struct —— 因此直接复用
label_native_direction_event，不另写一套语义。

R_struct <= 0（trigger close 时 OB 已失效）记 INVALID_AT_DECISION，
label=None，不进入模型 universe（避免制造机械可识别的 0 标签）。
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from research.phase1_native_direction.nd_contract_v1 import (
    FIVE_MIN, RESULTS, SYMBOLS, compute_atr5, disc_flags, get_bars,
)
from research.phase1_native_direction.nd_label_v1 import (
    label_native_direction_event,
)


def build():
    e = pd.read_csv(
        RESULTS / "reference_semantics_events.csv", low_memory=False)
    e["candidate_id"] = e["candidate_id"].astype(str)
    lab = pd.read_parquet(RESULTS / "native_labels_v1.parquet")
    lab["candidate_id"] = lab["candidate_id"].astype(str)
    d = e.merge(
        lab[["candidate_id", "trading_day", "touch_5m_bar_index",
             "decision_time", "touch_time", "candidate_group_id"]],
        on="candidate_id", how="left", validate="one_to_one")

    d["R_struct"] = (d["dist_close_to_far_edge_R"] * d["ATR5"]).astype(float)
    d["R_struct_over_ATR5"] = d["dist_close_to_far_edge_R"].astype(float)

    out = []
    for sym, g in d.groupby("symbol"):
        t0 = time.perf_counter()
        bars = get_bars(sym)
        disc = disc_flags(sym)
        o, h, l, c, t = (bars["open"], bars["high"], bars["low"],
                         bars["close"], bars["time"])
        n = len(o)
        for cid, di, nd, p0, rs, dt in zip(
                g["candidate_id"], g["touch_5m_bar_index"],
                g["native_direction"], g["close"], g["R_struct"],
                g["decision_time"]):
            di = int(di)
            if not np.isfinite(rs) or rs <= 0:
                out.append(dict(candidate_id=cid, symbol=sym,
                                struct_label=None,
                                struct_status="INVALID_AT_DECISION",
                                struct_bars=None, struct_res_type=None,
                                flipped_struct_label=None,
                                flipped_struct_status="INVALID_AT_DECISION",
                                flipped_struct_bars=None,
                                decision_time=dt, resolution_time=None))
                continue
            if di < 0 or di >= n:
                out.append(dict(candidate_id=cid, symbol=sym,
                                struct_label=None,
                                struct_status="BAD_BAR_INDEX",
                                struct_bars=None, struct_res_type=None,
                                flipped_struct_label=None,
                                flipped_struct_status="BAD_BAR_INDEX",
                                flipped_struct_bars=None,
                                decision_time=dt, resolution_time=None))
                continue
            rec = dict(candidate_id=cid, symbol=sym)
            for flip, pre in ((False, "struct"), (True, "flipped_struct")):
                res = label_native_direction_event(
                    start_idx=di + 1, reference_price=float(p0),
                    r_ref=float(rs),
                    native_direction=(-int(nd) if flip else int(nd)),
                    open_=o, high=h, low=l, discontinuity_before_bar=disc)
                rec[f"{pre}_label"] = res.label
                rec[f"{pre}_status"] = res.status
                rec[f"{pre}_bars"] = res.bars_to_resolution
                rec[f"{pre}_res_type"] = res.resolution_type
                if not flip:
                    rt = (t[res.resolution_bar_index] + FIVE_MIN
                          if res.resolution_bar_index is not None else None)
                    rec["resolution_time"] = rt
            rec["decision_time"] = dt
            out.append(rec)
        print(f"  [struct] {sym}: {len(g)} in "
              f"{time.perf_counter()-t0:.1f}s", flush=True)

    s = pd.DataFrame(out)
    full = d.merge(s, on="candidate_id", how="left",
                   validate="one_to_one", suffixes=("", "_s"))
    keep = [c for c in full.columns if not c.endswith("_s")]
    full = full[keep]
    full.to_parquet(RESULTS / "native_structural_labels_v1.parquet",
                    index=False)
    return full


def report(full):
    print("\n=== structural label profile ===")
    vc = full["struct_status"].value_counts()
    print(vc.to_string())
    res = full[full["struct_status"] == "RESOLVED"]
    br = float(res["struct_label"].mean())
    print(f"\nRESOLVED n={len(res)}  native structural base rate = {br:.4f}")
    print(f"  target wins={int((res['struct_label']==1).sum())}  "
          f"stop wins={int((res['struct_label']==0).sum())}")

    print("\n=== INVALID_AT_DECISION ===")
    inv = full["struct_status"] == "INVALID_AT_DECISION"
    print(f"  n={int(inv.sum())}  share={float(inv.mean()):.4f}")

    # native vs flipped placebo
    both = full[(full["struct_status"] == "RESOLVED")
                & (full["flipped_struct_status"] == "RESOLVED")].copy()
    ns = float(both["struct_label"].mean())
    fs = float(both["flipped_struct_label"].mean())
    print("\n=== native vs flipped (structural) ===")
    print(f"  n={len(both)}  native={ns:.4f}  flipped={fs:.4f}  "
          f"diff={ns-fs:+.4f}")

    rows = []
    for dim, col in (("direction", "native_direction"),
                     ("source_tf", "source_tf"), ("symbol", "symbol")):
        for k, g in both.groupby(col):
            a = float(g["struct_label"].mean())
            b = float(g["flipped_struct_label"].mean())
            rows.append(dict(维度=dim, 取值=str(k), n=len(g),
                             native=round(a, 4), flipped=round(b, 4),
                             diff=round(a - b, 4)))
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "structural_native_vs_flip.csv", index=False,
             encoding="utf-8-sig")
    print(t.to_string(index=False))

    # paired block bootstrap
    rng = np.random.default_rng(7)
    days = both["trading_day"].dropna().unique()
    idx = {dd: np.flatnonzero((both["trading_day"] == dd).to_numpy())
           for dd in days}
    nv = both["struct_label"].to_numpy(float)
    fv = both["flipped_struct_label"].to_numpy(float)
    diffs = []
    for _ in range(500):
        pick = rng.choice(days, size=len(days), replace=True)
        r_ = np.concatenate([idx[dd] for dd in pick])
        diffs.append(nv[r_].mean() - fv[r_].mean())
    ci = [round(float(np.quantile(diffs, .025)), 4),
          round(float(np.quantile(diffs, .975)), 4)]
    print(f"\n  paired block bootstrap 95%CI = {ci}")
    pd.DataFrame([dict(diff=round(ns - fs, 4), ci_low=ci[0],
                       ci_high=ci[1], n=len(both))]).to_csv(
        RESULTS / "structural_native_vs_flip_bootstrap.csv", index=False,
        encoding="utf-8-sig")

    # resolution profile
    rr = []
    for dim, col, keys in (
            ("ALL", None, [("ALL", res)]),
            ("direction", "native_direction", list(res.groupby("native_direction"))),
            ("source_tf", "source_tf", list(res.groupby("source_tf")))):
        for k, g in keys:
            q = g["struct_bars"].quantile([.25, .5, .75, .90, .95, .99])
            rr.append(dict(维度=dim, 取值=str(k), n=len(g),
                           p25=float(q[.25]), median=float(q[.5]),
                           p75=float(q[.75]), p90=float(q[.90]),
                           p95=float(q[.95]), p99=float(q[.99]),
                           max=float(g["struct_bars"].max()),
                           mean=float(g["struct_bars"].mean())))
    pdf = pd.DataFrame(rr)
    pdf.to_csv(RESULTS / "structural_resolution_profile.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== structural resolution (bars) ===")
    print(pdf.round(2).to_string(index=False))


def main():
    t0 = time.perf_counter()
    full = build()
    print(f"build {time.perf_counter()-t0:.0f}s")
    report(full)
    print("\nSTRUCTURAL_LABELS_DONE")


if __name__ == "__main__":
    main()
