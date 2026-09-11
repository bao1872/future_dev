"""SMC Oracle Atlas v1.0 —— 第一层：liquidity 生命周期与接触事件。

研究单位不是"一条 liquidity 一行"，而是：
    liquidity_master    : 一个事前形成的 liquidity identity 一行
    liquidity_contacts  : liquidity_id × contact_number 一行

接触判定严格在价格空间：
    BSL(side=+1) 从下方接近 -> 触及条件 high >= level
    SSL(side=-1) 从上方接近 -> 触及条件 low  <= level

接触分类（BSL / SSL 镜像）：
    TOUCH_ONLY              high == level（恰好到，未穿）
    PENETRATE_RECLAIM       穿过但 close 回到内侧
    PENETRATE_CLOSE_AT      穿过且 close == level
    PENETRATE_CLOSE_BEYOND  穿过且 close 站到外侧
    GAP_CROSS               open 已越过 level

首次严格穿透（high > level / low < level）后视为该 liquidity 被消费，
不再记录新的"未扫原始 contact"。
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import compute_atr5

RESULTS = Path("research/analysis_results/smc_oracle_atlas_v1")
RESULTS.mkdir(parents=True, exist_ok=True)
MAX_CONTACTS = 25


def find_contacts(hi, lo, op, cl, start, level, side, n):
    """返回 [(bar_idx, contact_type)]，首次穿透后停止。"""
    out = []
    i = int(start)
    while i < n and len(out) < MAX_CONTACTS:
        if side == +1:
            hit = np.flatnonzero(hi[i:] >= level)
        else:
            hit = np.flatnonzero(lo[i:] <= level)
        if not len(hit):
            break
        j = i + int(hit[0])
        if side == +1:
            if op[j] > level:
                ct = "GAP_CROSS"
            elif hi[j] > level:
                ct = ("PENETRATE_RECLAIM" if cl[j] < level else
                      "PENETRATE_CLOSE_AT" if cl[j] == level else
                      "PENETRATE_CLOSE_BEYOND")
            else:
                ct = "TOUCH_ONLY"
        else:
            if op[j] < level:
                ct = "GAP_CROSS"
            elif lo[j] < level:
                ct = ("PENETRATE_RECLAIM" if cl[j] > level else
                      "PENETRATE_CLOSE_AT" if cl[j] == level else
                      "PENETRATE_CLOSE_BEYOND")
            else:
                ct = "TOUCH_ONLY"
        out.append((j, ct))
        if ct != "TOUCH_ONLY":
            break  # 首次严格穿透 -> consumed
        # 找离开 level 的第一根 bar
        if side == +1:
            lv = np.flatnonzero(hi[j + 1:] < level)
        else:
            lv = np.flatnonzero(lo[j + 1:] > level)
        if not len(lv):
            break
        i = j + 1 + int(lv[0])
    return out


def build_symbol(sym, lv):
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    t = pd.to_datetime(five["bar_start_time"]).to_numpy()
    hi = five["high"].to_numpy(float)
    lo = five["low"].to_numpy(float)
    op = five["open"].to_numpy(float)
    cl = five["close"].to_numpy(float)
    atr = compute_atr5(dict(open=op, high=hi, low=lo, close=cl, time=t,
                            n=len(five)))
    n = len(five)

    mrows, crows = [], []
    for r in lv.itertuples(index=False):
        level = float(r.price)
        side = int(r.side)
        a = (int(r.activation_i) + 1 if pd.notna(r.activation_i)
             else int(np.searchsorted(t, np.datetime64(r.available_time),
                                      side="right")))
        if a >= n:
            mrows.append(dict(liquidity_id=r.level_key, symbol=sym,
                              liquidity_type=r.liquidity_type,
                              liquidity_scope=r.liquidity_scope, side=side,
                              price=level, available_time=r.available_time,
                              n_contacts=0, censored=True,
                              censor_reason="NO_BAR_AFTER_AVAILABLE"))
            continue
        cs = find_contacts(hi, lo, op, cl, a, level, side, n)
        if not cs:
            mrows.append(dict(liquidity_id=r.level_key, symbol=sym,
                              liquidity_type=r.liquidity_type,
                              liquidity_scope=r.liquidity_scope, side=side,
                              price=level, available_time=r.available_time,
                              n_contacts=0, censored=True,
                              censor_reason="NEVER_CONTACTED"))
            continue
        first_idx = cs[0][0]
        pen = [x for x in cs if x[1] != "TOUCH_ONLY"]
        fp_idx = pen[0][0] if pen else None
        last_type = cs[-1][1]
        mrows.append(dict(
            liquidity_id=r.level_key, symbol=sym,
            liquidity_type=r.liquidity_type,
            liquidity_scope=r.liquidity_scope, side=side, price=level,
            available_time=r.available_time,
            available_bar_index=a - 1,
            first_contact_bar_index=int(first_idx),
            first_contact_time=pd.Timestamp(t[first_idx]),
            bars_to_first_contact=int(first_idx - (a - 1)),
            first_penetration_bar_index=(int(fp_idx) if fp_idx is not None
                                         else None),
            first_penetration_time=(pd.Timestamp(t[fp_idx])
                                    if fp_idx is not None else pd.NaT),
            bars_to_first_penetration=(int(fp_idx - (a - 1))
                                       if fp_idx is not None else None),
            n_contacts=len(cs),
            n_contacts_before_penetration=(len(cs) - 1 if pen else len(cs)),
            consumed=bool(pen),
            final_contact_type=last_type,
            censored=not bool(pen),
            censor_reason=(None if pen else "NO_PENETRATION_BEFORE_END")))

        prev_idx = None
        for k, (j, ct) in enumerate(cs, start=1):
            a5 = float(atr[j]) if np.isfinite(atr[j]) and atr[j] > 0 else np.nan
            pen_r = (side * (hi[j] - level) / a5 if side == +1
                     else side * (level - lo[j]) / a5) if a5 == a5 else np.nan
            pen_r = max(pen_r, 0.0) if pen_r == pen_r else np.nan
            crows.append(dict(
                liquidity_id=r.level_key, symbol=sym, contact_number=k,
                contact_bar_index=int(j),
                contact_time=pd.Timestamp(t[j]),
                decision_time=pd.Timestamp(t[j]),
                contact_type=ct,
                liquidity_type=r.liquidity_type,
                liquidity_scope=r.liquidity_scope, side=side,
                liquidity_price=level,
                entry_reference=float(cl[j]),
                atr0=a5,
                bars_since_available=int(j - (a - 1)),
                bars_since_previous_contact=(None if prev_idx is None
                                             else int(j - prev_idx)),
                penetration_depth_R=round(float(pen_r), 6)
                if pen_r == pen_r else np.nan,
                close_relative_to_level_R=round(
                    float(side * (cl[j] - level) / a5), 6)
                if a5 == a5 else np.nan,
                bar_range_R=round(float((hi[j] - lo[j]) / a5), 6)
                if a5 == a5 else np.nan,
                abs_return_R=round(float(abs(cl[j] - op[j]) / a5), 6)
                if a5 == a5 else np.nan,
                is_first_contact=bool(k == 1),
                is_penetration=bool(ct != "TOUCH_ONLY"),
            ))
            prev_idx = j
    return pd.DataFrame(mrows), pd.DataFrame(crows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default=None)
    ap.add_argument("--out", default="v1")
    args = ap.parse_args()

    tr = pd.read_parquet(
        "research/analysis_results/liquidity_specificity_v2/"
        "true_interactions.parquet")
    lv = tr[tr["activation_state"] == "VALID_AHEAD"].copy()
    syms = (args.symbols.split(",") if args.symbols
            else sorted(lv["symbol"].unique().tolist()))

    M, C = [], []
    for sym in syms:
        t0 = time.perf_counter()
        m, c = build_symbol(sym, lv[lv["symbol"] == sym])
        M.append(m)
        C.append(c)
        print(f"  {sym}: levels={len(m)} contacts={len(c)} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)
    M = pd.concat(M, ignore_index=True)
    C = pd.concat(C, ignore_index=True)
    M.to_parquet(RESULTS / f"liquidity_master_{args.out}.parquet",
                 index=False)
    C.to_parquet(RESULTS / f"liquidity_contacts_{args.out}.parquet",
                 index=False)
    print(f"\nmaster={len(M)} contacts={len(C)}")
    if len(C):
        print("\ncontact_type:")
        print(C["contact_type"].value_counts().to_string())
        print("\ncontact_number:")
        print(C["contact_number"].value_counts().sort_index().head(8)
              .to_string())
    print("LIFECYCLE_DONE")


if __name__ == "__main__":
    main()
