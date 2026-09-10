"""Phase 1 事件级特征：PHASE1_FEATURES_V1。

只允许事件发生时的绝对状态。硬性排除：
    trade_direction / trade_mode / target_R
    target_fit_*
    任何 *_rel_*（依赖候选交易方向）
    stop_structure_*（依赖具体 action）
    任何 reward / future / resolution 字段
    *_zone_low / *_zone_high / *_level_（原始价格水平）

一个 candidate_id 只有一行，不再膨胀成 action rows。
输出：
    phase1_feature_contract_v1.csv
    features_v1.parquet（可再生，不入库）
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.phase1_tradability.phase1_contract_v1 import (
    RESULTS, STATE_PARQUET, group_of, is_excluded,
)
from research.ob_rl_model_view_v0_spec import MODEL_FEATURES_V0

# 62 维中「跨 6 动作不变」的事件级子集（M8 已用 action-invariance 验证）
INVARIANT_FROM_62 = [
    "source_tf", "source_ob_structure", "source_ob_bias",
    "source_ob_width_atr5", "touch_behavior", "touch_ordinal",
    "momentum_direction_5m", "momentum_direction_15m",
    "momentum_direction_1h", "quant_state",
    "quant_width_percentile_train",
]


def build() -> tuple[pd.DataFrame, pd.DataFrame, list]:
    st = pd.read_parquet(STATE_PARQUET)
    st["candidate_id"] = st["candidate_id"].astype(str)
    st = st.set_index("candidate_id")

    cols = [c for c in st.columns if c not in
            ("candidate_group_id", "trading_day", "touch_time",
             "touch_5m_bar_index", "decision_weight")]
    rows = []
    included = []
    for c in cols:
        ex, reason = is_excluded(c)
        if c in ("symbol", "source_tf"):
            reason = "分组变量(Baseline1专用)，非事件状态特征"
        g = group_of(c)
        rows.append(dict(feature=c, group=g, source="state_parquet",
                         causal_status="EVENT_LEVEL", included=(not ex),
                         exclusion_reason=reason))
        if not ex:
            included.append(c)
    # 62 维事件级子集（显式登记，_rel_ 一律排除）
    for c in MODEL_FEATURES_V0:
        ex, reason = is_excluded(c)
        if c not in INVARIANT_FROM_62:
            ex, reason = True, "62维中与action相关或未能证明事件级"
        rows.append(dict(feature=c, group=group_of(c), source="MODEL_FEATURES_V0",
                         causal_status=("EVENT_LEVEL" if not ex
                                        else "ACTION_RELATIVE"),
                         included=(not ex), exclusion_reason=reason))
    contract = pd.DataFrame(rows).drop_duplicates(subset=["feature"])
    contract.to_csv(RESULTS / "phase1_feature_contract_v1.csv", index=False,
                    encoding="utf-8-sig")

    X = st[included].copy()
    for c in included:
        if not pd.api.types.is_numeric_dtype(X[c]):
            X[c] = X[c].astype(str).astype("category").cat.codes.astype("int16")
    X = X.reset_index()
    X.to_parquet(RESULTS / "features_v1.parquet", index=False)
    return X, contract, included


def main():
    X, contract, included = build()
    print(f"[features] PHASE1_FEATURES_V1 = {len(included)}")
    print(contract.groupby("group")["included"].agg(["sum", "count"]).rename(
        columns={"sum": "纳入", "count": "登记"}).to_string())
    excluded = contract[~contract["included"]]
    print(f"\n排除 {len(excluded)} 项，示例：")
    print(excluded[["feature", "exclusion_reason"]].head(12).to_string(
        index=False))
    print("\nFEATURES_DONE")


if __name__ == "__main__":
    main()
