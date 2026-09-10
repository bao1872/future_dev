"""GATE A — 5m/15m/1h structural trend causality。

黄金标准检验：**截断测试**。
在 bars[:i+1] 上重建 SMC，bar i 的 swing_bias / internal_bias
必须与在全序列上得到的完全一致。若未来数据回流，截断后状态会改变。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.ob_trigger_snapshot import build_full_ob_smc_tf
from research.build_pytdx_panel import aggregate_15m
from research.ob_trigger_snapshot import aggregate_1h_from_15m

OUT = "research/liquidity_state_machine/trend_causality_audit.csv"
N_SAMPLE = 40


def prep(sym: str):
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
        drop=True)
    five["volume"] = five["trade"].astype(float)
    fifteen = aggregate_15m(five)
    oneh = aggregate_1h_from_15m(fifteen)
    for d in (five, fifteen, oneh):
        if "volume" not in d.columns and "trade" in d.columns:
            d["volume"] = d["trade"].astype(float)
    return {"5m": five, "15m": fifteen, "1h": oneh}


def state_at_i(bars, i):
    smc = build_full_ob_smc_tf(bars.iloc[: i + 1].copy())
    st = pd.DataFrame(smc["state_timeline"])
    row = st[st["bar_index"] == i]
    if row.empty:
        return None
    return (int(row["swing_bias"].iloc[0]),
            int(row["internal_bias"].iloc[0]))


def main():
    rng = np.random.default_rng(11)
    rows = []
    for sym in ("AG", "CU", "M", "RB", "SC", "TA"):
        bars_by_tf = prep(sym)
        for tf, bars in bars_by_tf.items():
            n = len(bars)
            if n < 200:
                continue
            full = build_full_ob_smc_tf(bars.copy())
            stf = pd.DataFrame(full["state_timeline"])
            idxs = sorted(set(int(x) for x in
                              rng.integers(120, n - 1, N_SAMPLE)))
            bad = 0
            checked = 0
            for i in idxs:
                a = stf[stf["bar_index"] == i]
                if a.empty:
                    continue
                want = (int(a["swing_bias"].iloc[0]),
                        int(a["internal_bias"].iloc[0]))
                got = state_at_i(bars, i)
                checked += 1
                if got != want:
                    bad += 1
            rows.append(dict(symbol=sym, tf=tf, n_bars=n,
                             checked=checked, mismatches=bad,
                             causal=bool(bad == 0)))
            print(f"  {sym} {tf}: checked={checked} mismatches={bad}",
                  flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print("\n=== TREND CAUSALITY AUDIT (truncation test) ===")
    print(d.to_string(index=False))
    ok = bool(d["causal"].all())
    print(f"\nGATE A: {'PASS' if ok else 'FAIL — TREND_CAUSALITY_GATE_FAIL'}")
    print("TREND_CAUSALITY_AUDIT_DONE")


if __name__ == "__main__":
    main()
