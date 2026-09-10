"""Step 8-9：feature contract 重新审计 + 方向对齐特征构建。

transform 逐项显式声明，禁止按列名启发式猜测。
语义依据来自原始字段取值（见 _verify_semantics），不是列名：

  signed_align  : 字段取值为带符号的牛/熊量（±1 或可正可负的连续量）
  invariant     : 无牛熊符号含义（位置/距离/类别/年龄/宽度/计数）
  categorical   : 字符串类别，语义本身已与方向无关（above/below 是位置不是多空）
  exclude       : 方向本身 / 零方差 / 标识
"""
from __future__ import annotations

import pandas as pd

from research.phase1_native_direction.nd_contract_v1 import (
    RESULTS, STATE16, load_candidates,
)

TF = ("5m", "15m", "1h")

# ---------- 显式声明：带符号（顺 OB 为正）----------
SIGNED_BASES = (
    "swing_bias",                    # ±1 趋势偏向（含 0）
    "internal_bias",                 # ±1
    "last_swing_structure_bias",     # ±1（含 0）
    "last_internal_structure_bias",  # ±1
    "dsa_direction",                 # ±1（含 0）
    "sqzmom_val",                    # 有正负的连续动量值
    "sqzmom_delta",                  # 有正负的连续动量变化
    "dsa_raw_dsa_vwap_dev_pct",      # 有正负的 VWAP 偏离百分比
    "dsa_vwap_dev_pct",              # 有正负的 VWAP 偏离百分比
)
SIGNED = tuple(f"{b}_{t}" for b in SIGNED_BASES for t in TF)

# ---------- 显式声明：字符串类别（原值为位置/类型，不是多空符号）----------
CATEGORICAL = (
    "source_ob_structure",                                   # internal/swing
    "touch_behavior",            # NO_BREACH/BREACH_RECLAIM/CLOSE_BEYOND（已相对 OB）
    "quant_state",                                           # HIGH/LOW/MID/UNKNOWN
    "source_tf",                                             # 5m/15m/1h
    # 注意：momentum_direction_* 原值是 contracting/expanding/flat，
    # 是类别而非 ±1 符号。若按列名猜成 signed 会被错误翻转。
    *tuple(f"momentum_direction_{t}" for t in TF),
    *tuple(f"last_swing_structure_type_{t}" for t in TF),    # BOS/CHoCH/NONE
    *tuple(f"last_internal_structure_type_{t}" for t in TF),  # BOS/CHoCH
    *tuple(f"above_ob_structure_class_{t}" for t in TF),     # internal/swing
    *tuple(f"below_ob_structure_class_{t}" for t in TF),     # internal/swing
    *tuple(f"swing_high_relation_{t}" for t in TF),          # above/below/overlap
    *tuple(f"swing_low_relation_{t}" for t in TF),           # above/below/overlap
    *tuple(f"internal_high_relation_{t}" for t in TF),       # above/below/overlap
    *tuple(f"internal_low_relation_{t}" for t in TF),        # above/below/overlap
)

# ---------- 显式声明：排除 ----------
EXCLUDE_EXPLICIT = {
    "source_ob_bias": "native_direction 本身，主模型不得直接使用",
    # 主模型不得直接看到 raw native_direction（仅允许进入 baseline）
    "native_direction": "native_direction 本身，主模型不得直接使用",
}
ZERO_VAR_CANDIDATES = tuple(
    f"{p}_{t}" for p in ("above_ob_bias", "below_ob_bias") for t in TF
)

# ---------- 硬性排除（因果/标识/未来）----------
HARD_EXCLUDE = {
    "candidate_id", "candidate_group_id", "trading_day", "touch_time",
    "touch_5m_bar_index", "decision_weight",
    "native_label", "flipped_label", "native_status", "flipped_status",
    "native_bars", "flipped_bars", "native_res_type", "flipped_res_type",
    "resolution_time", "decision_time", "resolution_bar_index",
    "bars_to_resolution", "resolution_type", "reference_price",
    "native_res_bar", "flipped_res_bar", "native_res_price",
    "symbol",          # 主模型不使用（仅 baseline 用）
}


def build_contract(X: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in X.columns:
        if c in HARD_EXCLUDE:
            rows.append(dict(feature=c, raw_source="state_parquet",
                             group="—", causal_status="EXCLUDED_HARD",
                             direction_semantics="n/a",
                             transform="exclude", included=False,
                             exclusion_reason="标识/未来/动作字段，硬性排除"))
            continue
        if c in EXCLUDE_EXPLICIT:
            rows.append(dict(feature=c, raw_source="state_parquet",
                             group="OB属性",
                             causal_status="EVENT_LEVEL",
                             direction_semantics="方向本身",
                             transform="exclude", included=False,
                             exclusion_reason=EXCLUDE_EXPLICIT[c]))
            continue
        if c in ZERO_VAR_CANDIDATES:
            nun = X[c].nunique(dropna=True)
            if nun <= 1:
                rows.append(dict(feature=c, raw_source="state_parquet",
                                 group="结构", causal_status="EVENT_LEVEL",
                                 direction_semantics="零方差",
                                 transform="exclude", included=False,
                                 exclusion_reason=f"零方差常量(unique={nun})，无信息"))
                continue
        if c in SIGNED:
            rows.append(dict(feature=c, raw_source="state_parquet",
                             group=_group(c), causal_status="EVENT_LEVEL",
                             direction_semantics="signed(bull+/bear-)",
                             transform="signed_align", included=True,
                             exclusion_reason=""))
            continue
        if c in CATEGORICAL:
            rows.append(dict(feature=c, raw_source="state_parquet",
                             group=_group(c), causal_status="EVENT_LEVEL",
                             direction_semantics="categorical/positional",
                             transform="categorical_align", included=True,
                             exclusion_reason=""))
            continue
        rows.append(dict(feature=c, raw_source="state_parquet",
                         group=_group(c), causal_status="EVENT_LEVEL",
                         direction_semantics="invariant",
                         transform="invariant", included=True,
                         exclusion_reason=""))
    return pd.DataFrame(rows)


def _group(c: str) -> str:
    for g, pats in (
        ("OB属性", ("source_ob_", "touch_", "is_first_touch", "group_")),
        ("趋势", ("swing_bias", "internal_bias", "_structure_bias")),
        ("结构", ("_structure_type", "_structure_age", "structure_class")),
        ("动量", ("momentum_", "sqzmom_")),
        ("DSA", ("dsa_")),
        ("风险几何", ("_atr_", "ob_above_atr", "ob_below_atr", "_relation_")),
        ("Quantile", ("quant_")),
    ):
        if any(p in c for p in pats):
            return g
    return "其他"


def build():
    """返回 (X_aligned, contract, feature_names)。"""
    st = pd.read_parquet(STATE16)
    st["candidate_id"] = st["candidate_id"].astype(str)
    cand = load_candidates()[["candidate_id", "native_direction"]]
    st = st.merge(cand, on="candidate_id", how="inner",
                  validate="one_to_one")

    X = st.drop(columns=[c for c in HARD_EXCLUDE if c in st.columns])
    contract = build_contract(X)
    contract.to_csv(RESULTS / "phase1_native_feature_contract_v1.csv",
                    index=False, encoding="utf-8-sig")

    d = st["native_direction"].to_numpy(float)
    inc = contract.loc[contract["included"], "feature"].tolist()
    out = pd.DataFrame({"candidate_id": st["candidate_id"].to_numpy()})
    for c in inc:
        v = pd.to_numeric(X[c], errors="coerce")
        tf = contract.loc[contract["feature"] == c, "transform"].iloc[0]
        if tf == "signed_align":
            out[c] = (v.to_numpy(float) * d)
        else:
            out[c] = v.to_numpy(float)
    # 类别字段：显式 one-hot（语义已与方向无关，不翻转）
    cats = [c for c in inc if c in CATEGORICAL]
    if cats:
        oh = pd.get_dummies(
            X[cats].astype("category"), prefix=cats, dtype=float)
        out = pd.concat([out, oh.reset_index(drop=True)], axis=1)
        out = out.drop(columns=cats)

    # 零方差列剔除
    nz = [c for c in out.columns
          if c != "candidate_id" and out[c].nunique(dropna=True) > 1]
    dropped = [c for c in out.columns if c != "candidate_id" and c not in nz]
    out = out[["candidate_id"] + nz]
    return out, contract, nz, dropped


def main():
    X, contract, feats, dropped = build()
    X.to_parquet(RESULTS / "native_features_v1.parquet", index=False)
    print("=== transform 分布 ===")
    print(contract["transform"].value_counts().to_string())
    print(f"\nincluded(raw)={int(contract['included'].sum())} "
          f"final feature dim={len(feats)} "
          f"dropped_zero_var={len(dropped)}")
    if dropped:
        print("dropped:", dropped)
    print("\n=== signed_align 字段 ===")
    print(contract.loc[contract["transform"] == "signed_align", "feature"]
          .tolist())
    print("\nFEATURES_DONE")


if __name__ == "__main__":
    main()
