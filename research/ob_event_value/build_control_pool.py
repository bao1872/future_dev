"""Step 2：Control pool —— 没有任何 source_tf OB_ENTERED 的普通时点。

Primary control 条件：
  1. 当前 bar 是有效 5m 交易 bar
  2. t0 无任何 source_tf 的 canonical OB_ENTERED
  3. 有足够 pre-history（t0 >= 21）
  4. 后面有完整 24 根 valid 5m bars
  5. 未来 24 根不跨无法可靠处理的 roll / discontinuity
  6. R0 = ATR5[t0-1] 有限且 > 0

明确禁止：要求"未来 24 根也不能出现 OB"——那是用未来信息选 control，
会造成 post-treatment selection bias。

预注册 sensitivity：CLEAN_CONTROL_3 —— t0 / t0-1 / t0-2 三根内均无
OB_ENTERED（只使用 control 时点及过去信息，合法）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ob_event_value.ev_contract_v1 import (
    HORIZONS, MATCH_COLS, RESULTS, SYMBOLS, future_clean, get_bars,
    pre_event_state,
)
from research.ob_event_value.build_event_bars import load_candidates


def main():
    cand = load_candidates()
    cand["candidate_id"] = cand["candidate_id"].astype(str)

    # 每个 (symbol, bar) 是否有 OB_ENTERED（忽略 source_tf）
    entered_any = {}
    for (sym, t0), _sub in cand.groupby(["symbol", "touch_5m_bar_index"]):
        entered_any[(sym, int(t0))] = True

    rows = []
    for sym in SYMBOLS:
        bars = get_bars(sym)
        n = bars["n"]
        st = pre_event_state(bars)
        day = bars["trading_day"]
        if day is None:
            raise RuntimeError(f"{sym}: 无 trading_day")
        uday = {d: i for i, d in enumerate(sorted(set(day.tolist())))}
        day_ord = np.array([uday[d] for d in day.tolist()])

        has = np.zeros(n, dtype=bool)
        for (s, t0), _v in entered_any.items():
            if s == sym and 0 <= t0 < n:
                has[t0] = True
        clean3 = ~has.copy()
        for sh in (1, 2):
            shifted = np.zeros(n, dtype=bool)
            shifted[sh:] = has[:n - sh]
            clean3 &= ~shifted

        ok_pre = np.arange(n) >= 21
        ok_fut = np.arange(n) + 24 <= n
        r0 = st["R0"]
        ok_r0 = np.isfinite(r0) & (r0 > 0)
        ok_disc = np.array([
            (i + 24 <= n) and future_clean(bars, i, max(HORIZONS))
            for i in range(n)])
        ok_state = np.ones(n, dtype=bool)
        for c in MATCH_COLS:
            ok_state &= np.isfinite(st[c])

        m = (~has) & ok_pre & ok_fut & ok_r0 & ok_disc & ok_state
        idx = np.flatnonzero(m)
        rows.append(pd.DataFrame(dict(
            control_id=[f"{sym}|{int(i)}" for i in idx],
            symbol=sym, t0=idx,
            session_type=bars["session_type"][idx],
            time_bucket_30m=bars["time_bucket_30m"][idx],
            trading_day=day[idx],
            trading_day_ord=day_ord[idx],
            clean3=clean3[idx],
            **{c: st[c][idx] for c in MATCH_COLS},
            R0=r0[idx],
        )))
        print(f"  [control] {sym}: {len(idx)} / {n} bars", flush=True)

    pool = pd.concat(rows, ignore_index=True)
    pool.to_parquet(RESULTS / "control_pool.parquet", index=False)

    aud = (pool.groupby("symbol")
               .agg(controls=("control_id", "size"),
                    clean3=("clean3", "sum")).reset_index())
    aud.to_csv(RESULTS / "control_pool_audit.csv", index=False,
               encoding="utf-8-sig")

    print("\n=== control pool ===")
    print(f"  total controls   : {len(pool)}")
    print(f"  CLEAN_CONTROL_3  : {int(pool['clean3'].sum())}")
    print("\n=== 硬断言 ===")
    assert (~pool["t0"].isna()).all()
    assert (pool["R0"] > 0).all()
    print("  control R0 > 0  : OK")
    print("  no OB at t0     : OK (构造保证)")
    print("\nCONTROL_POOL_DONE")


if __name__ == "__main__":
    main()
