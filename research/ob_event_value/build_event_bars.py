"""Step 1：重建 event-bar treatment unit（去伪重复）。

同一 (symbol, source_tf, trigger_5m_bar_index) 可能有多个 candidate_id，
其未来路径完全相同，必须 collapse 成 1 个 treatment event，
否则造成 pseudo-replication。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.ob_event_value.ev_contract_v1 import (
    RESULTS, SYMBOLS, V3_DIR, get_bars,
)


def load_candidates() -> pd.DataFrame:
    parts = []
    for s in SYMBOLS:
        c = pd.read_csv(V3_DIR / f"{s}_candidates.csv", low_memory=False)
        c["symbol"] = s
        parts.append(c[["candidate_id", "symbol", "source_tf",
                        "source_ob_bias", "touch_5m_bar_index"]])
    return pd.concat(parts, ignore_index=True)


def main():
    cand = load_candidates()
    cand["candidate_id"] = cand["candidate_id"].astype(str)

    g = cand.groupby(["symbol", "source_tf", "touch_5m_bar_index"])
    rows = []
    for (sym, tf, t0), sub in g:
        biases = sorted(set(sub["source_ob_bias"].astype(int).tolist()))
        if biases == [1]:
            comp = "BULLISH"
            nd = 1
        elif biases == [-1]:
            comp = "BEARISH"
            nd = -1
        else:
            comp = "MIXED"
            nd = 0
        rows.append(dict(
            event_bar_id=f"{sym}|{tf}|{int(t0)}",
            symbol=sym, source_tf=tf, t0=int(t0),
            n_entered_obs=len(sub), bias_composition=comp,
            native_direction=nd,
            candidate_ids=",".join(sorted(sub["candidate_id"].tolist())),
        ))
    ev = pd.DataFrame(rows)
    assert ev["event_bar_id"].is_unique, "event_bar_id 不唯一"

    # 过滤：需要足够 pre-history、完整未来、且不跨不可信边界
    keep = []
    for sym, sub in ev.groupby("symbol"):
        bars = get_bars(sym)
        n = bars["n"]
        t0s = sub["t0"].to_numpy()
        ok_hist = t0s >= 21                       # t0-20 .. t0-1 需要存在
        ok_fut = t0s + 24 <= n
        ok_r0 = np.isfinite(bars["atr5"][np.maximum(t0s - 1, 0)]) & \
            (bars["atr5"][np.maximum(t0s - 1, 0)] > 0)
        ok_disc = np.array([_clean(bars, int(t)) for t in t0s])
        m = np.asarray(ok_hist & ok_fut & ok_r0 & ok_disc, dtype=bool)
        keep.append(sub[m])
    ev = pd.concat(keep, ignore_index=True)
    assert ev["event_bar_id"].is_unique

    # ANY_TF_BAR_DEDUP：同一 (symbol, t0) 只保留一条
    any_tf = (ev.sort_values(["symbol", "t0", "source_tf"])
                .groupby(["symbol", "t0"], as_index=False).head(1))

    ev.to_parquet(RESULTS / "event_bars.parquet", index=False)
    any_tf.to_parquet(RESULTS / "event_bars_anytf_dedup.parquet",
                      index=False)

    aud = (ev.groupby(["symbol", "source_tf"])
             .agg(event_bars=("event_bar_id", "size"),
                  n_candidates=("n_entered_obs", "sum"))
             .reset_index())
    aud.to_csv(RESULTS / "event_bar_audit.csv", index=False,
               encoding="utf-8-sig")

    print("=== treatment universe ===")
    print(f"  candidate rows        : {len(cand)}")
    print(f"  event_bar (sym×tf×bar): {len(ev)}")
    print(f"  ANY_TF dedup (sym×bar): {len(any_tf)}")
    print(f"  collapsed away        : {len(cand) - len(ev)}")
    print(f"  any_tf collapsed      : {len(ev) - len(any_tf)}")
    print("\n=== by source_tf ===")
    print(ev.groupby("source_tf").size().to_string())
    print("\n=== bias composition ===")
    print(ev["bias_composition"].value_counts().to_string())
    print("\nEVENT_BARS_DONE")


def _clean(bars, t0):
    from research.ob_event_value.ev_contract_v1 import future_clean
    return future_clean(bars, t0, 24)


if __name__ == "__main__":
    main()
