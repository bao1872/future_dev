"""Step 3-7：AG smoke test、LONG/SHORT 人工 spot-check、全品种标签、
flipped-direction placebo、label/resolution profile。

reference_price = trigger_bar_close（测量坐标原点，不是 entry_price）
start_idx       = touch_5m_bar_index + 1（只用事件确认之后的价格）
R_ref           = ATR5_at_trigger（统计测量尺，不是最终止损）
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from research.phase1_native_direction.nd_contract_v1 import (
    BEARISH, BULLISH, FIVE_MIN, RESULTS, SYMBOLS, compute_atr5, disc_flags,
    get_bars, load_candidates,
)
from research.phase1_native_direction.nd_label_v1 import (
    label_native_direction_event,
)


def _scan(cand: pd.DataFrame, sym: str, flip: bool) -> list:
    bars = get_bars(sym)
    atr = compute_atr5(bars)
    disc = disc_flags(sym)
    o, h, l, c, t = (bars["open"], bars["high"], bars["low"],
                     bars["close"], bars["time"])
    n = len(o)
    sub = cand[cand["symbol"] == sym]
    recs = []
    for cid, di, nd in zip(sub["candidate_id"],
                           sub["touch_5m_bar_index"],
                           sub["native_direction"]):
        di = int(di)
        dt = t[di] + FIVE_MIN
        if di < 0 or di >= n:
            recs.append((cid, None, "BAD_BAR_INDEX", None, None, None,
                         dt, None, None, None))
            continue
        p0, r = float(c[di]), float(atr[di])
        if not np.isfinite(r) or r <= 0:
            recs.append((cid, None, "NO_ATR", None, None, None,
                         dt, None, None, None))
            continue
        d = -int(nd) if flip else int(nd)
        res = label_native_direction_event(
            start_idx=di + 1, reference_price=p0, r_ref=r,
            native_direction=d, open_=o, high=h, low=l,
            discontinuity_before_bar=disc)
        rt = (t[res.resolution_bar_index] + FIVE_MIN
              if res.resolution_bar_index is not None else None)
        recs.append((cid, res.label, res.status, res.resolution_bar_index,
                     res.bars_to_resolution, res.resolution_price,
                     dt, rt, res.resolution_type, p0))
    return recs


COLS = ["candidate_id", "label", "status", "resolution_bar_index",
        "bars_to_resolution", "resolution_price", "decision_time",
        "resolution_time", "resolution_type", "reference_price"]


def build_all(cand: pd.DataFrame, syms=SYMBOLS):
    frames = {}
    for flip in (False, True):
        recs = []
        for sym in syms:
            t0 = time.perf_counter()
            recs += _scan(cand, sym, flip)
            print(f"  [labels] {sym} flip={flip}: "
                  f"{time.perf_counter()-t0:.1f}s", flush=True)
        frames[flip] = pd.DataFrame(recs, columns=COLS)

    lab = cand[["candidate_id", "symbol", "source_tf", "trading_day",
                "native_direction", "touch_5m_bar_index",
                "candidate_group_id", "touch_time"]].copy()
    nat = frames[False].rename(columns={
        "label": "native_label", "status": "native_status",
        "resolution_bar_index": "native_res_bar",
        "bars_to_resolution": "native_bars",
        "resolution_price": "native_res_price",
        "resolution_type": "native_res_type"})
    flp = frames[True].rename(columns={
        "label": "flipped_label", "status": "flipped_status",
        "resolution_bar_index": "flipped_res_bar",
        "bars_to_resolution": "flipped_bars",
        "resolution_type": "flipped_res_type"})
    lab = lab.merge(nat, on="candidate_id", how="left", validate="one_to_one")
    lab = lab.merge(flp[["candidate_id", "flipped_label", "flipped_status",
                         "flipped_bars", "flipped_res_type"]],
                    on="candidate_id", how="left", validate="one_to_one")
    return lab


def smoke_ag(lab: pd.DataFrame):
    print("\n=== AG SMOKE TEST (native) ===")
    ag = lab[(lab["symbol"] == "AG") & (lab["native_status"] == "RESOLVED")]
    for want in (1, 0):
        print(f"\n--- native_label={want} ---")
        for _, r in ag[ag["native_label"] == want].head(5).iterrows():
            print(f"  {r['candidate_id']} tf={r['source_tf']} "
                  f"dir={r['native_direction']} "
                  f"P0={r['reference_price']:.1f} "
                  f"bars={r['native_bars']} type={r['native_res_type']}")


def spot_check(lab: pd.DataFrame, direction: int, n_each: int = 5):
    """逐根打印 future bar 直到 resolution，人工核对方向逻辑。"""
    name = "BULLISH/LONG" if direction == BULLISH else "BEARISH/SHORT"
    print(f"\n{'='*70}\nSPOT CHECK — {name}\n{'='*70}")
    sub = lab[(lab["native_direction"] == direction)
              & (lab["native_status"].isin(["RESOLVED", "AMBIGUOUS_INTRABAR"]))]
    for kind, val in (("SUCCESS", 1), ("FAILURE", 0), ("AMBIGUOUS", None)):
        if val is None:
            g = sub[sub["native_status"] == "AMBIGUOUS_INTRABAR"]
        else:
            g = sub[(sub["native_status"] == "RESOLVED")
                    & (sub["native_label"] == val)]
        print(f"\n--- {kind} (show {min(n_each, len(g))}) ---")
        for _, r in g.head(n_each).iterrows():
            bars = get_bars(r["symbol"])
            di = int(r["touch_5m_bar_index"])
            p0 = r["reference_price"]
            rr = _atr_at(r["symbol"], di)
            d = int(r["native_direction"])
            tgt = p0 + d * 2.5 * rr
            stp = p0 - d * 1.0 * rr
            print(f"\n  {r['candidate_id']} sym={r['symbol']} "
                  f"tf={r['source_tf']} dir={d}")
            print(f"  trigger={r['touch_time']} P0={p0:.2f} ATR5={rr:.4f}"
                  f"  target={tgt:.2f} stop={stp:.2f}")
            nb = r["native_bars"]
            end = di + 1 + (int(nb) if pd.notna(nb) else 1)
            for i in range(di + 1, min(end, len(bars["open"]))):
                print(f"    bar{i}: t={bars['time'][i]} "
                      f"O={bars['open'][i]:.2f} H={bars['high'][i]:.2f} "
                      f"L={bars['low'][i]:.2f} C={bars['close'][i]:.2f}")
            print(f"    => status={r['native_status']} "
                  f"label={r['native_label']} type={r['native_res_type']}")


_ATR_CACHE: dict = {}


def _atr_at(sym: str, idx: int) -> float:
    if sym not in _ATR_CACHE:
        _ATR_CACHE[sym] = compute_atr5(get_bars(sym))
    return float(_ATR_CACHE[sym][idx])


def label_profile(lab: pd.DataFrame):
    rows = []
    for dim, col in (("ALL", None), ("direction", "native_direction"),
                     ("source_tf", "source_tf"), ("symbol", "symbol")):
        groups = [("ALL", lab)] if col is None else list(lab.groupby(col))
        for k, g in groups:
            res = g[g["native_status"] == "RESOLVED"]
            rows.append(dict(
                维度=dim, 取值=str(k), 事件数=len(g),
                resolved=len(res),
                target_wins=int((res["native_label"] == 1).sum()),
                stop_wins=int((res["native_label"] == 0).sum()),
                ambiguous=int((g["native_status"]
                               == "AMBIGUOUS_INTRABAR").sum()),
                roll_censored=int((g["native_status"]
                                   == "ROLL_CENSORED").sum()),
                end_censored=int((g["native_status"]
                                  == "END_OF_DATA_CENSORED").sum()),
                native_base_rate=(round(float(res["native_label"].mean()), 4)
                                  if len(res) else None),
            ))
    p = pd.DataFrame(rows)
    p.to_csv(RESULTS / "label_profile.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== label profile ===")
    print(p.to_string(index=False))
    return p


def resolution_profile(lab: pd.DataFrame):
    res = lab[lab["native_status"] == "RESOLVED"]
    rows = []

    def add(dim, key, g):
        q = g["native_bars"].quantile([.10, .25, .5, .75, .90, .95, .99])
        rows.append(dict(
            维度=dim, 取值=str(key), n=len(g),
            p10=float(q[.10]), p25=float(q[.25]), median=float(q[.5]),
            p75=float(q[.75]), p90=float(q[.90]), p95=float(q[.95]),
            p99=float(q[.99]), max=float(g["native_bars"].max()),
            mean=float(g["native_bars"].mean()),
            median_hours=round(float(q[.5]) * 5 / 60, 3),
        ))

    add("ALL", "ALL", res)
    for k, g in res.groupby("native_direction"):
        add("direction", "LONG" if k == 1 else "SHORT", g)
    for k, g in res.groupby("source_tf"):
        add("source_tf", k, g)
    for k, g in res.groupby("symbol"):
        add("symbol", k, g)
    p = pd.DataFrame(rows)
    p.to_csv(RESULTS / "resolution_profile.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== resolution profile (bars) ===")
    print(p.round(2).to_string(index=False))
    return p


def native_vs_flip(lab: pd.DataFrame):
    both = lab[(lab["native_status"] == "RESOLVED")
               & (lab["flipped_status"] == "RESOLVED")].copy()
    rows = []

    def add(dim, key, g):
        ns = float(g["native_label"].mean())
        fs = float(g["flipped_label"].mean())
        rows.append(dict(维度=dim, 取值=str(key), n=len(g),
                         native_success_rate=round(ns, 4),
                         flipped_success_rate=round(fs, 4),
                         native_minus_flipped=round(ns - fs, 4)))

    add("ALL", "ALL", both)
    for k, g in both.groupby("native_direction"):
        add("direction", "LONG" if k == 1 else "SHORT", g)
    for k, g in both.groupby("symbol"):
        add("symbol", k, g)
    for k, g in both.groupby("source_tf"):
        add("source_tf", k, g)
    t = pd.DataFrame(rows)
    t.to_csv(RESULTS / "native_vs_flip.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== native vs flipped placebo ===")
    print(t.to_string(index=False))

    # paired block bootstrap by trading_day
    rng = np.random.default_rng(7)
    out = []

    def boot(mask, label):
        # mask 决定样本子集；block 索引必须在该子集内重建，
        # 否则各子集会退回全样本（曾导致 LONG/SHORT 数值完全相同）。
        g = both[mask]
        days = g["trading_day"].dropna().unique()
        idx = {d: np.flatnonzero((g["trading_day"] == d).to_numpy())
               for d in days}
        nv = g["native_label"].to_numpy(float)
        fv = g["flipped_label"].to_numpy(float)
        diffs = []
        for _ in range(500):
            pick = rng.choice(days, size=len(days), replace=True)
            rows_i = np.concatenate([idx[d] for d in pick])
            diffs.append(nv[rows_i].mean() - fv[rows_i].mean())
        return dict(维度=label, n=len(g), n_days=len(days),
                    diff=round(float(nv.mean() - fv.mean()), 4),
                    ci_low=round(float(np.quantile(diffs, .025)), 4),
                    ci_high=round(float(np.quantile(diffs, .975)), 4))

    allm = np.ones(len(both), dtype=bool)
    out.append(boot(allm, "ALL"))
    for k in (1, -1):
        out.append(boot((both["native_direction"] == k).to_numpy(),
                        "LONG" if k == 1 else "SHORT"))
    b = pd.DataFrame(out)
    b.to_csv(RESULTS / "native_vs_flip_bootstrap.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== paired block bootstrap (trading_day, 500) ===")
    print(b.to_string(index=False))
    return t, b


def main():
    cand = load_candidates()

    # Step 3: AG smoke
    ag = cand[cand["symbol"] == "AG"]
    lab_ag = build_all(ag, ["AG"])
    smoke_ag(lab_ag)

    # Step 5: full build
    print("\n=== full label build (15 symbols, native + flipped) ===",
          flush=True)
    t0 = time.perf_counter()
    lab = build_all(cand)
    print(f"build done {time.perf_counter()-t0:.0f}s", flush=True)

    # hard assertions
    res = lab["native_status"].eq("RESOLVED")
    assert lab["candidate_id"].is_unique
    assert set(lab["native_direction"].dropna().unique()).issubset({-1, 1})
    assert lab.loc[res, "native_label"].isin([0, 1]).all()
    assert (lab.loc[res, "resolution_time"]
            > lab.loc[res, "decision_time"]).all()
    print("hard assertions PASSED")

    lab.to_parquet(RESULTS / "native_labels_v1.parquet", index=False)

    label_profile(lab)
    resolution_profile(lab)

    # Step 4: LONG/SHORT spot-check
    spot_check(lab, BULLISH)
    spot_check(lab, BEARISH)

    # Step 6: placebo
    native_vs_flip(lab)
    print("\nNATIVE_LABELS_DONE")


if __name__ == "__main__":
    main()
