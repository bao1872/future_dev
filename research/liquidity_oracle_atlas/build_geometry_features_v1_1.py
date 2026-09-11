"""SMC Opportunity Geometry Decomposition v1.1 —— 特征契约 + 特征构造。

- 复用 opportunity_common.build_features（已修复 OB freshness 分类 bug）。
- 冻结 feature_manifest_v1_1.json，诚实标注 G4（多周期距离分箱）在冻结 Atlas v1.2 中
  不存在（扫描全部 20 个冻结文件确认），故 G4 = UNAVAILABLE。
- 输出写入 research/analysis_results/smc_opportunity_geometry_v1_1/
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import opportunity_common as oc

OUT = Path("research/analysis_results/smc_opportunity_geometry_v1_1")
OUT.mkdir(parents=True, exist_ok=True)

# ---- 子块定义（仅使用冻结 Atlas v1.2 真实存在的列）----
G1 = ["nearest_above_R", "nearest_below_R"]
G2 = ["n_targets_L", "n_targets_S"]
G3 = ["nearest_ahead_R", "nearest_behind_R"]
G4 = []   # UNAVAILABLE：冻结 Atlas v1.2 无 5m/15m/1h/session/day/week 距离分箱列
G5 = ["same_price_identity_count"]

B0 = oc.B0_COLS
B1 = oc.B1_COLS
B2 = oc.B2_COLS          # 实际 = G1+G2+G3+G5（B2_BIN_COLS 不存在）
B3 = oc.B3_COLS
B4 = oc.B4_COLS

FEATURE_MANIFEST_V11 = {
    "study": "SMC Opportunity Geometry Decomposition v1.1",
    "atlas_freeze_commit": oc.ATLAS_FREEZE_COMMIT,
    "tb_hash": oc.TB_HASH,
    "prospective_oos_start": oc.PROSPECTIVE_OOS_START,
    "blocks": {
        "B0_metadata": B0,
        "B1_contact": B1,
        "B2_liquidity_geometry": B2,
        "B3_trend": B3,
        "B4_ob_context": B4,
        "G1_nearest_price": G1,
        "G2_target_count": G2,
        "G3_event_relative": G3,
        "G4_multitf_field": G4,   # UNAVAILABLE
        "G5_same_price": G5,
    },
    "g4_status": ("UNAVAILABLE: no per-scope (5m/15m/1h/CONTIG_SESSION/"
                  "TRADING_DAY/TRADING_WEEK) distance-bin-count columns exist "
                  "in frozen Atlas v1.2 (verified across all 20 frozen parquet "
                  "files). v1.0 B2_BIN_COLS were hand-named and never matched "
                  "real columns, so G4 was silently empty. Cannot test "
                  "multitf liquidity field vs simple geometry this round."),
    "unavailable": [
        "session_type", "minute_from_session_open",
        "internal_bias_1h/15m/5m", "pre_* (pre_ret_3_R etc.)",
        "pre_contact_same_price_*",
        "B8 per-bin/per-scope active target counts",
        "G4 multi-timeframe distance bins",
    ],
    "freshness_fix": ("nearest_opposing_ob_freshness / "
                      "nearest_same_direction_ob_freshness now treated as "
                      "categorical (FRESH/RETESTED/missing) instead of coerced "
                      "to NaN."),
}


def manifest_hash():
    s = json.dumps(FEATURE_MANIFEST_V11, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def main():
    # 复用构建（freshness 已修复于 opportunity_common）
    oc.OUT = OUT
    # labels 与 v1.0 完全一致，直接复用其落盘 parquet（避免重跑 Stage 0）
    src_lab = Path("research/analysis_results/smc_opportunity_v1") / "opportunity_labels.parquet"
    import shutil
    shutil.copyfile(src_lab, OUT / "opportunity_labels.parquet")
    F = oc.build_features()
    F.to_parquet(OUT / "opportunity_features_v1_1.parquet", index=False)
    h = manifest_hash()
    FEATURE_MANIFEST_V11["manifest_hash"] = h
    with open(OUT / "feature_manifest_v1_1.json", "w", encoding="utf-8") as f:
        json.dump(FEATURE_MANIFEST_V11, f, ensure_ascii=False, indent=2)
    proto = {
        "study": "SMC Opportunity Geometry Decomposition v1.1",
        "baseline_commit": "a746e3e5402e71a44cf57dbf1b718f4610f8e2bd",
        "atlas_freeze_commit": oc.ATLAS_FREEZE_COMMIT,
        "tb_hash": oc.TB_HASH,
        "prospective_oos_start": oc.PROSPECTIVE_OOS_START,
        "feature_manifest_hash": h,
        "models": "fixed LogisticRegression(penalty=l2,C=1.0,solver=lbfgs,"
                  "max_iter=3000); OneHotEncoder(handle_unknown=ignore); "
                  "median impute + StandardScaler (numeric); SplineTransformer"
                  "(n_knots=4,degree=2,knots=quantile) for nonlinear distances",
        "walk_forward": {
            "WF1": "train TB1 / test TB2",
            "WF2": "train TB1+TB2 / test TB3",
            "WF3": "train TB1+TB2+TB3 / test TB4",
        },
        "bootstrap": "canonical trading_day block, 500 resamples, 95% CI",
        "g4_status": FEATURE_MANIFEST_V11["g4_status"],
        "note": ("Primary risk for mechanism audit: 0.25/0.50/0.75/1.00 ATR; "
                 "1.5/2/3 ATR secondary. No best-stop selection. No PnL."),
    }
    with open(OUT / "GEOMETRY_PROTOCOL.json", "w", encoding="utf-8") as f:
        json.dump(proto, f, ensure_ascii=False, indent=2)
    print(f"[F1] features shape = {F.shape}")
    print(f"[F2] manifest_hash = {h}")
    g4n = len(G4)
    print(f"[F3] G1={len(G1)} G2={len(G2)} G3={len(G3)} G4={g4n}(UNAVAILABLE) "
          f"G5={len(G5)}")
    print(f"[DONE] v1.1 特征与契约已落盘: {OUT}")


if __name__ == "__main__":
    main()
