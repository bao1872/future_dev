"""SMC Structural Delivery Opportunity Study v1.0 —— 共享定义与特征构造。

- 冻结 Atlas v1.2（不修改）。
- FEATURE_MANIFEST 是特征契约：在跑任何画像/模型前冻结并 hash。
- 所有特征必须满足 available_time <= decision_time（决策时可观测）。
- Oracle / path / 未来字段（best_R_*, resolution_class, direction_stability…）
  一律排除出 X。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ATLAS = Path("research/analysis_results/smc_oracle_atlas_v1")
OUT = Path("research/analysis_results/smc_opportunity_v1")
OUT.mkdir(parents=True, exist_ok=True)

RISK_GRID = [0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]
N_BLOCKS = 4
BLOCKS = [f"TB{i+1}" for i in range(N_BLOCKS)]

ATLAS_FREEZE_COMMIT = "7ae7c1ae57b45bbc8bade1b5742fdc0353859c83"
TB_HASH = ("b51fe7952bc1311aed83409889cd6706a6bc173cd4ed2f6e96570340183ac3d6")
PROSPECTIVE_OOS_START = "2026-09-07"

# 决策时可观测、且确实存在于冻结 Atlas v1.2 状态快照/contact 表的字段。
# 不在其中的字段标为 UNAVAILABLE（见 FEATURE_MANIFEST["unavailable"]）。

B0_COLS = ["symbol", "side", "liquidity_type", "liquidity_scope", "atr0"]

B1_COLS = [
    "contact_number", "contact_type",
    "bars_since_available", "bars_since_previous_contact",
    "penetration_depth_R", "close_relative_to_level_R",
    "bar_range_R", "abs_return_R",
    "is_first_contact", "is_penetration",
]

# 标准化距离分箱（每 scope 11 个 bin），归并为 正侧 5 + 负侧 5
_SCOPES = ["5m", "15m", "1h", "CONTIG_SESSION", "TRADING_DAY", "TRADING_WEEK"]
_POS_BINS = ["(0,0.5]", "(0.5,1]", "(1,2]", "(2,4]", "(4,+inf)"]
_NEG_BINS = ["(-inf,-4]", "(-4,-2]", "(-2,-1]", "(-1,-0.5]", "(-0.5,0]"]


def _bin_cols():
    out = []
    for s in _SCOPES:
        for b in _POS_BINS:
            out.append(f"{s}_pos_{b}")
        for b in _NEG_BINS:
            out.append(f"{s}_neg_{b}")
    return out


B2_BIN_COLS = _bin_cols()
B2_BASE_COLS = [
    "nearest_above_R", "nearest_below_R", "nearest_ahead_R", "nearest_behind_R",
    "same_price_identity_count",
    "n_targets_L", "n_targets_S",  # B8 active target geometry（cluster count 可用）
]
B2_COLS = B2_BASE_COLS + B2_BIN_COLS

B3_COLS = [
    "env_direction_4h", "trend_struct_1h", "trend_struct_15m", "trend_struct_5m",
    "sweep_vs_5m", "sweep_vs_15m", "sweep_vs_1h",
    "env4h_vs_1h", "trend_1h_vs_15m", "trend_15m_vs_5m",
]

B4_COLS = [
    "nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
    "nearest_opposing_ob_source_tf", "nearest_opposing_ob_freshness",
    "nearest_opposing_ob_prior_enter_count",
    "nearest_same_direction_ob_distance_R", "nearest_same_direction_ob_width_R",
    "nearest_same_direction_ob_source_tf",
    "nearest_same_direction_ob_freshness",
    "nearest_same_direction_ob_prior_enter_count",
]

# 数值型且明显非线性的字段 → SplineTransformer
SPLINE_NUMERIC = [
    "penetration_depth_R", "close_relative_to_level_R", "bar_range_R",
    "abs_return_R", "nearest_above_R", "nearest_below_R",
    "nearest_ahead_R", "nearest_behind_R",
    "nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
    "nearest_same_direction_ob_distance_R", "nearest_same_direction_ob_width_R",
    "atr0",
] + B2_BIN_COLS  # 分箱计数亦作数值

FEATURE_MANIFEST = {
    "study": "SMC Structural Delivery Opportunity Study v1.0",
    "atlas_freeze_commit": ATLAS_FREEZE_COMMIT,
    "tb_definition_hash": TB_HASH,
    "risk_grid": RISK_GRID,
    "prospective_oos_start_trading_day": PROSPECTIVE_OOS_START,
    "blocks": {
        "B0_metadata": B0_COLS,
        "B1_contact_lifecycle": B1_COLS,
        "B2_liquidity_target_field": B2_COLS,
        "B3_trend": B3_COLS,
        "B4_ob_context": B4_COLS,
    },
    "spline_numeric": SPLINE_NUMERIC,
    "model_contract": {
        "classifier": "LogisticRegression",
        "penalty": "l2", "C": 1.0, "solver": "lbfgs", "max_iter": 3000,
        "categorical": "OneHotEncoder(handle_unknown='ignore')",
        "numeric": "median imputer + StandardScaler",
        "nonlinear_numeric": "SplineTransformer(n_knots=4, degree=2, knots='quantile', include_bias=False)",
        "fit_scope": "train_only",
    },
    "unavailable_in_frozen_atlas_v1_2": {
        "B0": ["session_type", "minute_from_session_open"],
        "B1": ["atr_rel_pre", "pre_ret_3_R", "pre_ret_12_R",
                "pre_rv_12_R", "pre_range_12_R",
                "volume_z_20_pre", "volume_z_t0"],
        "B3": ["internal_bias_1h", "internal_bias_15m", "internal_bias_5m"],
        "B8": ["per_bin_target_cluster_counts",
                "per_scope_target_distances",
                "pre_contact_same_price_identity_count",
                "pre_contact_same_price_scopes"],
        "note": "以上字段不在冻结 Atlas v1.2；session/volume/OB-freshness 等可由 raw bars 因果重建，"
                "但本轮不在 feature contract 内，避免引入事后重建偏差。",
    },
    "excluded_oracle_path_fields": [
        "best_R_lower", "best_R_upper", "resolution_class", "direction_stability",
        "direction", "status", "path_censor", "n_both_pos",
    ],
}


def manifest_json_str() -> str:
    m = {k: v for k, v in FEATURE_MANIFEST.items()}
    # 去掉不可 hash 的大对象（无），直接 json
    return json.dumps(m, ensure_ascii=False, sort_keys=True)


def manifest_hash() -> str:
    return hashlib.sha256(manifest_json_str().encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------
# 加载
# ----------------------------------------------------------------------
def load_labels() -> pd.DataFrame:
    return pd.read_parquet(OUT / "opportunity_labels.parquet")


def _build_bin_features(st: pd.DataFrame) -> pd.DataFrame:
    """把 11-bin 列归并为 正侧 5 + 负侧 5。"""
    out = pd.DataFrame(index=st.index)
    rename = {}
    for s in _SCOPES:
        for b in _POS_BINS:
            rename[f"{s}_bin_{b}"] = f"{s}_pos_{b}"
        for b in _NEG_BINS:
            rename[f"{s}_bin_{b}"] = f"{s}_neg_{b}"
    sel = st[[c for c in rename if c in st.columns]].rename(columns=rename)
    for c in sel.columns:
        sel[c] = pd.to_numeric(sel[c], errors="coerce").fillna(0)
    return sel


def build_features() -> pd.DataFrame:
    """返回 contact×risk 行 × 全部 block 特征，索引与 load_labels() 对齐。"""
    lab = load_labels()
    keys = ["symbol", "liquidity_id", "contact_number", "risk_ATR"]

    st = pd.read_parquet(ATLAS / "liquidity_state_snapshot_v1_2.parquet")
    con = pd.read_parquet(ATLAS / "liquidity_contacts_v1_1.parquet")

    # 决策时快照是 contact 级别，与 risk_ATR 无关 → 直接按 contact 键合并
    ckeys = ["symbol", "liquidity_id", "contact_number"]
    stf = _build_bin_features(st).reset_index(drop=True)
    st_meta = st[ckeys].reset_index(drop=True)
    st_all = pd.concat([st_meta, stf], axis=1)
    # OB / trend / nearest 等也在 st 中
    extra = [c for c in (B2_BASE_COLS + B3_COLS + B4_COLS)
             if c in st.columns and c not in st_all.columns]
    if extra:
        st_all = st_all.merge(st[ckeys + extra], on=ckeys, how="left")

    con_meta = con[ckeys + [c for c in (B0_COLS + B1_COLS)
                            if c in con.columns and c not in ckeys]].copy()

    # 合并到 contact×risk
    F = lab[keys + ["n_targets_L", "n_targets_S"]].copy()
    F = F.merge(con_meta, on=ckeys, how="left")
    F = F.merge(st_all, on=ckeys, how="left")

    # 仅对数值列做 to_numeric；分类列（symbol/side/trend/ob_source_tf 等字符串）
    # 必须保持 object，交给 pipeline 的 OneHotEncoder，绝不能强转数值（会变成 NaN）。
    NUMERIC = (["atr0", "contact_number", "bars_since_available",
                "bars_since_previous_contact", "penetration_depth_R",
                "close_relative_to_level_R", "bar_range_R", "abs_return_R",
                "n_targets_L", "n_targets_S", "nearest_above_R", "nearest_below_R",
                "nearest_ahead_R", "nearest_behind_R", "same_price_identity_count"]
               + B2_BIN_COLS
               + ["nearest_opposing_ob_distance_R", "nearest_opposing_ob_width_R",
                  "nearest_opposing_ob_freshness", "nearest_opposing_ob_prior_enter_count",
                  "nearest_same_direction_ob_distance_R",
                  "nearest_same_direction_ob_width_R",
                  "nearest_same_direction_ob_freshness",
                  "nearest_same_direction_ob_prior_enter_count"])
    for c in NUMERIC:
        if c in F.columns:
            F[c] = pd.to_numeric(F[c], errors="coerce")
    # 类别字段保持 object（pipeline 内 OneHot）
    return F


if __name__ == "__main__":
    print("manifest_hash =", manifest_hash())
    print("blocks:")
    for b, cols in FEATURE_MANIFEST["blocks"].items():
        print(f"  {b}: {len(cols)} cols")
    F = build_features()
    print("features shape:", F.shape)


# ----------------------------------------------------------------------
# 时间块（复用已冻结 temporal_blocks_v1.json）
# ----------------------------------------------------------------------
def load_blocks():
    blk = json.load(open(
        Path("research/analysis_results/smc_oracle_label_temporal_stability_v1")
        / "temporal_blocks_v1.json"))
    days = {tb: blk[tb]["exact_days"] for tb in BLOCKS}
    d2b = {d: tb for tb, ds in days.items() for d in ds}
    return d2b


def attach_trading_day_block(keys_df: pd.DataFrame) -> pd.DataFrame:
    """给 contact 键表补 trading_day + block（因果可用，来自 raw bars）。"""
    from research.export_ob_trigger_execution_v21 import load_raw_5m
    con = pd.read_parquet(ATLAS / "liquidity_contacts_v1_1.parquet")
    syms = sorted(con["symbol"].unique().tolist())
    td_map = {}
    for s in syms:
        five = load_raw_5m(s)
        td_arr = five["trading_day"].astype(str).to_numpy()
        c = con[con["symbol"] == s]
        idx = c["contact_bar_index"].to_numpy()
        ok = (idx >= 0) & (idx < len(td_arr))
        m = pd.Series(np.nan, index=c.index, dtype=object)
        m[pd.Series(ok, index=c.index)] = td_arr[idx[ok]]
        td_map[s] = m
    con["trading_day"] = pd.concat(td_map.values()).reindex(con.index)
    d2b = load_blocks()
    con["block"] = con["trading_day"].map(d2b)
    out = keys_df.merge(
        con[["symbol", "liquidity_id", "contact_number", "trading_day", "block"]],
        on=["symbol", "liquidity_id", "contact_number"], how="left")
    return out
