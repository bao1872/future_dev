"""Phase 1B 数据准备：16 品种候选 + 多 horizon 标签 + 105 维特征。

全部复用 Phase 1 已冻结的语义（canonical OB entered candidate、ATR5、
2.5R / 1R barrier、first-passage），只把品种从 DEV4 扩到 16。

产出（research/analysis_results/phase1_tradability_v1/）：
    candidates_v16.parquet
    labels_v16.parquet       tradable_12 / 24 / 48 / unbounded
    features_v16.parquet
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.phase1_tradability.build_features_v1 import (
    build as build_feats,
)
from research.phase1_tradability.build_labels_v1 import (
    FIVE_MIN, label_tradability_event,
)
from research.phase1_tradability.phase1_contract_v1 import (
    RESULTS, ROLL_GAP_ATR_THRESHOLD, compute_atr5, discontinuity_flags,
    get_bars,
)

STATE16 = Path(
    "research/analysis_results/ob_rl_dataset_v0_16/ob_rl_state_v0.parquet")

HORIZONS = ((12, "12"), (24, "24"), (48, "48"), (None, "unb"))


def build_candidates_16() -> pd.DataFrame:
    st = pd.read_parquet(STATE16)
    st["candidate_id"] = st["candidate_id"].astype(str)
    assert st["candidate_id"].is_unique, "16品种 candidate_id 不唯一"
    keep = st[[
        "candidate_id", "candidate_group_id", "symbol", "source_tf",
        "touch_5m_bar_index", "trading_day", "touch_time",
    ]].copy()
    keep.to_parquet(RESULTS / "candidates_v16.parquet", index=False)
    print(f"[cand16] rows={len(keep)} "
          f"symbols={sorted(keep['symbol'].unique().tolist())}", flush=True)
    return keep


def _label_one(cand: pd.DataFrame, syms, max_bars) -> pd.DataFrame:
    recs = []
    for sym in syms:
        t0 = time.perf_counter()
        bars = get_bars(sym)
        atr = compute_atr5(bars)
        disc = discontinuity_flags(sym, threshold=ROLL_GAP_ATR_THRESHOLD)
        o, h, l, c, t = (bars["open"], bars["high"], bars["low"],
                         bars["close"], bars["time"])
        n = len(o)
        sub = cand[cand["symbol"] == sym]
        for cid, di in zip(sub["candidate_id"],
                           sub["touch_5m_bar_index"]):
            di = int(di)
            dt = t[di] + FIVE_MIN
            if di < 0 or di >= n:
                recs.append((cid, None, "BAD_BAR_INDEX", None, dt, None))
                continue
            p0, r = float(c[di]), float(atr[di])
            if not np.isfinite(r) or r <= 0:
                recs.append((cid, None, "NO_ATR", None, dt, None))
                continue
            res = label_tradability_event(di + 1, p0, r, o, h, l, disc,
                                          max_bars=max_bars)
            rt = (t[res.resolution_bar_index] + FIVE_MIN
                  if res.resolution_bar_index is not None else None)
            recs.append((cid, res.label, res.status, res.bars_to_resolution,
                         dt, rt))
        print(f"  [labels] {sym} h={max_bars}: {len(sub)} in "
              f"{time.perf_counter()-t0:.1f}s", flush=True)
    return pd.DataFrame(recs, columns=[
        "candidate_id", "label", "status", "bars_to_resolution",
        "decision_time", "resolution_time"])


def build_labels_16(cand: pd.DataFrame) -> pd.DataFrame:
    syms = sorted(cand["symbol"].unique().tolist())
    out = cand[["candidate_id", "symbol", "source_tf"]].copy()
    for max_bars, name in HORIZONS:
        f = _label_one(cand, syms, max_bars)
        f = f.rename(columns={
            "label": f"tradable_{name}",
            "status": f"status_{name}",
            "bars_to_resolution": f"bars_{name}"})
        if name == "24":
            out = out.merge(f, on="candidate_id", how="left",
                            validate="one_to_one")
        else:
            out = out.merge(
                f.drop(columns=["decision_time", "resolution_time"]),
                on="candidate_id", how="left", validate="one_to_one")
    out.to_parquet(RESULTS / "labels_v16.parquet", index=False)
    print("\n[labels16] 各 horizon 分布：")
    for _, name in HORIZONS:
        vc = out[f"status_{name}"].value_counts()
        base = out.loc[out[f"status_{name}"] == "RESOLVED",
                       f"tradable_{name}"].mean()
        print(f"  h={name:>3}: resolved={int(vc.get('RESOLVED', 0))} "
              f"base_rate={base:.4f}  "
              f"{dict(vc)}")
    return out


def build_features_16() -> pd.DataFrame:
    # build_features_v1 在 import 时就把 STATE_PARQUET 绑定到自己的名字空间，
    # 必须改它自己的绑定（改 contract 模块的属性无效）。
    import research.phase1_tradability.build_features_v1 as BF
    orig = BF.STATE_PARQUET
    BF.STATE_PARQUET = STATE16
    try:
        X, contract, included = build_feats()
    finally:
        BF.STATE_PARQUET = orig
    X.to_parquet(RESULTS / "features_v16.parquet", index=False)
    print(f"[feat16] PHASE1_FEATURES_V1 = {len(included)} rows={len(X)}")
    return X


def main():
    t0 = time.perf_counter()
    cand = build_candidates_16()
    build_labels_16(cand)
    build_features_16()
    print(f"\nPREPARE16_DONE {time.perf_counter()-t0:.0f}s")


if __name__ == "__main__":
    main()
