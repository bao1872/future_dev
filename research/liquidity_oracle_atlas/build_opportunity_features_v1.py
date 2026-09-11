"""SMC Structural Delivery Opportunity Study v1.0 —— 特征构造 + 协议落盘。

§11 顺序：先冻结 feature_manifest.json + hash，再产生特征，之后才画像/建模。
本脚本只做：写 feature_manifest.json / OPPORTUNITY_PROTOCOL.json /
opportunity_features.parquet。不训练。
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import opportunity_common as oc

OUT = oc.OUT
OUT.mkdir(parents=True, exist_ok=True)

print("[F1] 冻结 feature_manifest.json + hash ...")
mhash = oc.manifest_hash()
manifest = dict(oc.FEATURE_MANIFEST)
manifest["manifest_hash"] = mhash
with open(OUT / "feature_manifest.json", "w") as f:
    json.dump(manifest, f, ensure_ascii=False, indent=2)
print(f"    manifest_hash = {mhash[:16]}...")

print("[F2] 构造特征 opportunity_features.parquet ...")
F = oc.build_features()
# 行对齐检查
lab = oc.load_labels()
assert len(F) == len(lab), f"特征行数 {len(F)} != 标签行数 {len(lab)}"
F.to_parquet(OUT / "opportunity_features.parquet", index=False)
print(f"    features shape = {F.shape}")
print(f"    B0={len(oc.B0_COLS)} B1={len(oc.B1_COLS)} "
      f"B2={len(oc.B2_COLS)} B3={len(oc.B3_COLS)} B4={len(oc.B4_COLS)}")

print("[F3] 写 OPPORTUNITY_PROTOCOL.json ...")
protocol = {
    "study": "SMC Structural Delivery Opportunity Study v1.0",
    "atlas_freeze_commit": oc.ATLAS_FREEZE_COMMIT,
    "temporal_block_definition_hash": oc.TB_HASH,
    "feature_manifest_hash": mhash,
    "risk_grid": oc.RISK_GRID,
    "prospective_oos_start_trading_day": oc.PROSPECTIVE_OOS_START,
    "evidence_level": ("描述性 + 内部因果时间泛化(walk-forward TB1->TB4)；"
                       "非独立 OOS validation。"),
    "preregistered_hypotheses": {
        "H1": "B2(liquidity/target geometry) 应是 Opportunity 最主要信息块之一。",
        "H2": "B1(contact geometry) 应提供额外信息。",
        "H3": "B3(trend) 对‘有没有 delivery’的增量不作预设（方向未知）。",
        "H4": ("若 M_ob-M0>0 但 Full-Full_minus_OB≈0，则 OB 主要代理 "
               "liquidity/geometry，无独立上下文价值。"),
    },
    "verdict_criteria": {
        "LEARNABLE": ("Full 相对 M0 主要 ranking metric 跨 WF 方向一致；"
                      "Brier/LogLoss 不系统恶化；Bottom20 delivery rate "
                      "稳定低于 base；多数品种方向一致；"
                      "bootstrap delta 非单一时期贡献。"),
        "WEAK": "存在少量增量但不跨时期/品种稳定。",
        "NOT_LEARNABLE": "Full≈M0 且 Bottom20 几乎不能富集 NO_DELIVERY。",
        "REGIME_DEPENDENT": "不同 TB/WF 明显方向变化。",
    },
    "forbidden_claims": [
        "PnL / 策略回测", "最佳止损", "Long/Short 正式方向模型",
        "LightGBM/CatBoost/SHAP/自动特征搜索",
    ],
}
with open(OUT / "OPPORTUNITY_PROTOCOL.json", "w") as f:
    json.dump(protocol, f, ensure_ascii=False, indent=2)
print("[DONE] 特征与协议已落盘:", OUT)
