#!/usr/bin/env python3

"""62维状态 + 单步离线强化学习 V1 -- 第一阶段数据与实验合同审计。

本阶段只做审计，不训练模型，不运行最终收益回测。

产出：
    research/analysis_results/rl_62d_v1/feature_manifest_62.csv
    research/analysis_results/rl_62d_v1/feature_causality_audit.csv
    research/analysis_results/rl_62d_v1/candidate_audit.csv
    research/analysis_results/rl_62d_v1/reward_matrix_audit.csv
    research/analysis_results/rl_62d_v1/time_split_audit.json
    research/analysis_results/rl_62d_v1/experiment_contract.json
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from research.ob_rl_dataset_v0_spec import (
    ACTIONS,
    PRIMARY_HORIZON,
    REWARD_VERSION,
    SAME_BAR_POLICY,
    STOP_ATR,
    TARGET_R,
    VALIDATED_TFS,
)
from research.ob_rl_model_view_v0_spec import (
    CATEGORICAL_FEATURES_V0,
    MODEL_FEATURES_V0,
)
from research.rl_62d_core_v1 import (
    ACTION_NAMES,
    BAR_MINUTES,
    DATASET_SKIP_ACTION,
    MAX_FUTURE_OBSERVATION_BARS,
    MAX_FUTURE_OBSERVATION_MINUTES,
    N_ACTIONS,
    NO_TRADE_INDEX,
)

RESULTS = Path("research/analysis_results/rl_62d_v1")

STATE_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/"
    "ob_rl_state_v0.parquet"
)
ACTION_PARQUET = Path(
    "research/analysis_results/ob_rl_dataset_v0/"
    "ob_rl_action_v0.parquet"
)

# 上一轮被证伪的静态分组，本轮严禁作为过滤条件。
FORBIDDEN_BUCKET_FACTORS = (
    "source_tf",
    "touch_bin",
    "ob_width_bucket",
)

# 已知历史参考不变量，只用于核对，不用于硬编码样本。
REFERENCE_CANDIDATE_COUNT = 21_481
REFERENCE_ELIGIBLE_COUNT = 20_013

# 物理分组名称（按当前代码真实语义）
G_TREND = "趋势"
G_STRUCTURE = "结构"
G_MOMENTUM = "动量"
G_VOLUME = "成交量或成交参与"
G_VOLATILITY = "波动"
G_OB = "订单块或结构位置"
G_OTHER = "其他"

SPEC_MODULE = "research/ob_rl_model_view_v0_spec.py"
DATASET_MODULE = "research/ob_rl_dataset_v0_spec.py"

# 决策时点可用性
DTA_YES = "YES"
DTA_ACTION_DEPENDENT = "ACTION_DEPENDENT"


# ------------------------------------------------------------
# 62 维静态语义表
# ------------------------------------------------------------


def _event_specs():
    return [
        (
            "source_tf",
            G_OB,
            "订单块来源周期 5m/15m/1h",
        ),
        (
            "source_ob_structure",
            G_OB,
            "订单块结构类型 internal/swing",
        ),
        (
            "source_ob_bias",
            G_OB,
            "订单块方向 +1 多头 / -1 空头",
        ),
        (
            "source_ob_width_atr5",
            G_OB,
            "订单块宽度，按 ATR5 归一化",
        ),
        (
            "touch_behavior",
            G_OB,
            "触碰行为 NO_BREACH/BREACH_RECLAIM/CLOSE_BEYOND",
        ),
        (
            "touch_ordinal",
            G_OB,
            "该订单块被触碰的序号",
        ),
    ]


def _smc_specs(tf):
    return [
        (
            f"internal_bias_rel_{tf}",
            G_TREND,
            f"{tf} 内部结构方向，按交易方向旋转",
        ),
        (
            f"swing_bias_rel_{tf}",
            G_TREND,
            f"{tf} 摆动结构方向，按交易方向旋转",
        ),
        (
            f"target_fit_internal_{tf}",
            G_STRUCTURE,
            f"{tf} 前方内部结构距离 / 目标距离，随 target_R 变化",
        ),
        (
            f"target_fit_swing_{tf}",
            G_STRUCTURE,
            f"{tf} 前方摆动结构距离 / 目标距离，随 target_R 变化",
        ),
        (
            f"target_fit_ob_{tf}",
            G_OB,
            f"{tf} 前方活跃订单块距离 / 目标距离，随 target_R 变化",
        ),
        (
            f"stop_structure_internal_{tf}",
            G_STRUCTURE,
            f"{tf} 后方内部结构距离 / 止损距离，止损固定 1 ATR，随方向翻转",
        ),
        (
            f"stop_structure_swing_{tf}",
            G_STRUCTURE,
            f"{tf} 后方摆动结构距离 / 止损距离，止损固定 1 ATR，随方向翻转",
        ),
        (
            f"stop_structure_ob_{tf}",
            G_OB,
            f"{tf} 后方活跃订单块距离 / 止损距离，止损固定 1 ATR，随方向翻转",
        ),
        (
            f"forward_active_ob_bias_rel_{tf}",
            G_OB,
            f"{tf} 前方最近活跃订单块方向，按交易方向旋转",
        ),
        (
            f"forward_active_ob_structure_class_{tf}",
            G_OB,
            f"{tf} 前方最近活跃订单块结构类别，缺失为 NO_OB",
        ),
        (
            f"backward_active_ob_bias_{tf}",
            G_OB,
            f"{tf} 后方最近活跃订单块方向",
        ),
        (
            f"backward_active_ob_structure_class_{tf}",
            G_OB,
            f"{tf} 后方最近活跃订单块结构类别，缺失为 NO_OB",
        ),
    ]


def _dsa_specs(tf):
    return [
        (
            f"dsa_alignment_{tf}",
            G_TREND,
            f"{tf} DSA 已确认方向与交易方向的一致性，随动作变化",
        ),
        (
            f"dsa_vwap_dev_rel_{tf}",
            G_VOLUME,
            f"{tf} VWAP 偏离，按交易方向旋转；DSA 未确认时为 NaN",
        ),
    ]


def _momentum_specs(tf):
    return [
        (
            f"momentum_direction_{tf}",
            G_MOMENTUM,
            f"{tf} SQZMOM 方向 expanding/contracting/flat",
        ),
        (
            f"momentum_value_rel_{tf}",
            G_MOMENTUM,
            f"{tf} SQZMOM 强度，按交易方向旋转",
        ),
        (
            f"momentum_delta_rel_{tf}",
            G_MOMENTUM,
            f"{tf} SQZMOM 变化量，按交易方向旋转",
        ),
    ]


def _quantile_specs():
    return [
        (
            "quant_state",
            G_VOLATILITY,
            "分位机会宽度分位状态 LOW/MID/HIGH/UNKNOWN",
        ),
        (
            "quant_width_percentile_train",
            G_VOLATILITY,
            "分位机会宽度预测值相对训练分布的分位",
        ),
    ]


def _action_specs():
    return [
        (
            "trade_mode",
            G_OTHER,
            "动作标识 FOLLOW/FADE，非市场状态",
        ),
        (
            "trade_direction",
            G_OTHER,
            "动作标识 = source_ob_bias x 方向系数，非市场状态",
        ),
        (
            "target_R",
            G_OTHER,
            "动作标识 盈利目标倍数，非市场状态",
        ),
    ]


def build_static_spec():
    """按权威 MODEL_FEATURES_V0 的顺序拼出静态语义表。"""

    specs = []
    specs += [("EVENT",) + s for s in _event_specs()]
    for tf in VALIDATED_TFS:
        specs += [(f"SMC_{tf}",) + s for s in _smc_specs(tf)]
    for tf in VALIDATED_TFS:
        specs += [(f"DSA_{tf}",) + s for s in _dsa_specs(tf)]
    for tf in VALIDATED_TFS:
        specs += [
            (f"MOMENTUM_{tf}",) + s for s in _momentum_specs(tf)
        ]
    specs += [("QUANTILE",) + s for s in _quantile_specs()]
    specs += [("ACTION",) + s for s in _action_specs()]

    got = [s[1] for s in specs]
    if got != list(MODEL_FEATURES_V0):
        raise RuntimeError(
            "静态语义表与权威 MODEL_FEATURES_V0 不一致"
        )
    return specs


# ------------------------------------------------------------
# 数据装载
# ------------------------------------------------------------


def load_state_action():
    state = pd.read_parquet(STATE_PARQUET)
    action = pd.read_parquet(ACTION_PARQUET)
    return state, action


def compute_analyzable_mask(action: pd.DataFrame) -> pd.Series:
    """候选在主视野下 6 个交易动作收益是否全部有效。

    连续性要求由权威模拟器保证：前向 12 根 K 线必须连续，
    否则该候选在主视野下不可评估。
    """

    trade = action[action["action"] != DATASET_SKIP_ACTION]
    return trade.groupby("candidate_id")["primary_reward_R"].apply(
        lambda x: bool(np.isfinite(x.to_numpy(float)).all())
    )


def build_event_universe(state, action):
    """按合同构造事件全集：合格候选 + 主视野可评估。"""

    eligible = ~state["touch_close_beyond_far_edge"].astype(bool)
    analyzable = state["candidate_id"].map(
        compute_analyzable_mask(action)
    ).fillna(False).astype(bool)

    universe = state.loc[eligible & analyzable].copy()
    universe = universe.sort_values(
        ["trading_day", "symbol", "touch_5m_bar_index"],
        kind="mergesort",
    ).reset_index(drop=True)

    # 上一轮静态分组严禁作为过滤条件。
    # 注意 source_tf 本身是 62 维特征之一，必须保留为特征，
    # 只是不得用于筛选样本。校验方式：合格候选中出现的每个取值，
    # 都必须在最终事件集中仍然存在（不得被筛成空集）。
    for col in FORBIDDEN_BUCKET_FACTORS:
        if col not in state.columns:
            continue
        before = set(
            state.loc[eligible, col].dropna().unique().tolist()
        )
        after = set(universe[col].dropna().unique().tolist())
        if not before.issubset(after):
            raise RuntimeError(
                f"检测到分组过滤: {col} 缺失取值 "
                f"{sorted(before - after)}"
            )

    return universe, eligible, analyzable


def build_reward_matrix(action, universe) -> np.ndarray:
    """把权威动作表透视成 [事件数, 7] 的真实收益矩阵。"""

    trade = action[
        action["candidate_id"].isin(
            universe["candidate_id"].to_numpy()
        )
    ]

    piv = trade.pivot(
        index="candidate_id",
        columns="action",
        values="primary_reward_R",
    )

    # 数据集的不交易动作名为 SKIP，对应本实验的 NO_TRADE。
    cols = [
        DATASET_SKIP_ACTION if a == "NO_TRADE" else a
        for a in ACTION_NAMES
    ]
    missing = [c for c in cols if c not in piv.columns]
    if missing:
        raise RuntimeError(f"动作缺失: {missing}")

    piv = piv.reindex(universe["candidate_id"].to_numpy())[cols]
    matrix = piv.to_numpy(dtype=float)

    # 不交易真实收益恒为 0。
    matrix[:, NO_TRADE_INDEX] = 0.0

    if matrix.shape != (len(universe), N_ACTIONS):
        raise RuntimeError(
            f"奖励矩阵形状错误: {matrix.shape}"
        )
    if not np.isfinite(matrix).all():
        raise RuntimeError("奖励矩阵存在非有限值")
    return matrix


# ------------------------------------------------------------
# 特征清单 / 因果审计
# ------------------------------------------------------------


def build_feature_manifest(action, universe):
    specs = build_static_spec()

    usable_ids = set(universe["candidate_id"].to_numpy())
    sub = action[
        action["candidate_id"].isin(usable_ids)
        & (action["action"] != DATASET_SKIP_ACTION)
    ]

    # 动作依赖性：同一候选在 6 个交易动作下取值是否恒定。
    grp = sub.groupby("candidate_id")
    vary = {}
    for name in MODEL_FEATURES_V0:
        vary[name] = int(
            (grp[name].nunique(dropna=False) > 1).sum()
        )

    cat = set(CATEGORICAL_FEATURES_V0)
    rows = []
    for i, (family, name, group, note) in enumerate(specs, start=1):
        col = sub[name]
        rows.append(
            {
                "feature_index": i,
                "feature_name": name,
                "physical_group": group,
                "source_module": SPEC_MODULE,
                "source_field": name,
                "decision_time_available": (
                    DTA_YES
                    if vary[name] == 0
                    else DTA_ACTION_DEPENDENT
                ),
                "future_data_risk": (
                    "REVIEW_REQUIRED"
                    if name.startswith("quant_")
                    else "NONE"
                ),
                "missing_rate": round(
                    float(col.isna().mean()), 6
                ),
                "dtype": str(col.dtype),
                "notes": note,
                "family": family,
                "is_categorical": name in cat,
                "varies_across_actions": vary[name] > 0,
                "candidates_with_variation": vary[name],
                "row_grain": "candidate_x_action",
            }
        )

    df = pd.DataFrame(rows)
    if len(df) != 62:
        raise RuntimeError(f"特征数不是 62: {len(df)}")
    if df["feature_name"].nunique() != 62:
        raise RuntimeError("特征名不唯一")
    return df


def build_causality_audit(manifest):
    rows = []
    for _, r in manifest.iterrows():
        name = r["feature_name"]
        group = r["physical_group"]
        dta = r["decision_time_available"]

        if name.startswith("quant_"):
            reason = (
                "分位机会宽度是决策时点的模型输出，分位参考分布"
                "按 fold 拟合；需复核分位模型输入是否全部决策时点可知"
            )
            suggested = "保持使用，训练前人工复核分位模型输入"
        elif dta == DTA_ACTION_DEPENDENT:
            reason = (
                "取值依赖被评估的动作（按交易方向旋转或除以目标距离），"
                "不存在单一事件状态取值"
            )
            suggested = (
                "需裁决输入口径后再训练，见 experiment_contract.json"
            )
        else:
            reason = "决策时点由已冻结的当期结构与指标直接得到"
            suggested = "保持使用"

        rows.append(
            {
                "feature_name": name,
                "physical_group": group,
                "decision_time_available": dta,
                "lookahead_bars": 0,
                "future_data_risk": r["future_data_risk"],
                "risk_reason": reason,
                "confirmed_leak": False,
                "suggested_action": suggested,
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 候选审计
# ------------------------------------------------------------


def build_candidate_audit(state, eligible, analyzable, universe):
    rows = [
        {
            "level": "STAGE",
            "key": "frozen_candidates",
            "rule": "冻结 V3 候选全集",
            "count": int(len(state)),
        },
        {
            "level": "STAGE",
            "key": "excluded_touch_bar_close_beyond",
            "rule": "剔除触碰当根收盘已越过远端边界",
            "count": int((~eligible).sum()),
        },
        {
            "level": "STAGE",
            "key": "eligible_candidates",
            "rule": "合格候选",
            "count": int(eligible.sum()),
        },
        {
            "level": "STAGE",
            "key": "excluded_non_contiguous_h12",
            "rule": (
                f"剔除主视野 h{PRIMARY_HORIZON} 前向 12 根 K 线不连续"
            ),
            "count": int((eligible & ~analyzable).sum()),
        },
        {
            "level": "STAGE",
            "key": "final_usable_events",
            "rule": "合格且在 h12 可评估的最终事件集",
            "count": int(len(universe)),
        },
        {
            "level": "REFERENCE",
            "key": "reference_candidate_count",
            "rule": "历史参考不变量",
            "count": REFERENCE_CANDIDATE_COUNT,
        },
        {
            "level": "REFERENCE",
            "key": "reference_eligible_count",
            "rule": "历史参考不变量",
            "count": REFERENCE_ELIGIBLE_COUNT,
        },
    ]

    for sym, n in universe["symbol"].value_counts().items():
        rows.append(
            {
                "level": "SYMBOL",
                "key": f"final_usable_events_{sym}",
                "rule": "最终事件集分品种",
                "count": int(n),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 奖励矩阵审计
# ------------------------------------------------------------


def build_reward_matrix_audit(matrix, action, universe):
    per_action = []
    for i, name in enumerate(ACTION_NAMES):
        col = matrix[:, i]
        per_action.append(
            {
                "action_index": i,
                "action_name": name,
                "mean_R": round(float(np.mean(col)), 6),
                "std_R": round(float(np.std(col)), 6),
                "min_R": round(float(np.min(col)), 6),
                "max_R": round(float(np.max(col)), 6),
                "nan_count": int(np.isnan(col).sum()),
                "is_all_zero": bool(np.all(col == 0.0)),
            }
        )

    trade = action[
        action["candidate_id"].isin(
            universe["candidate_id"].to_numpy()
        )
        & (action["action"] != DATASET_SKIP_ACTION)
    ]

    rows = [
        {"check": "n_events", "value": int(matrix.shape[0])},
        {"check": "n_action_columns", "value": int(matrix.shape[1])},
        {
            "check": "no_trade_all_zero",
            "value": bool(
                np.all(matrix[:, NO_TRADE_INDEX] == 0.0)
            ),
        },
        {
            "check": "all_finite",
            "value": bool(np.isfinite(matrix).all()),
        },
        {
            "check": "primary_horizon",
            "value": int(PRIMARY_HORIZON),
        },
        {"check": "reward_version", "value": REWARD_VERSION},
        {"check": "stop_atr", "value": float(STOP_ATR)},
        {
            "check": "target_R",
            "value": ",".join(str(t) for t in TARGET_R),
        },
        {"check": "same_bar_policy", "value": SAME_BAR_POLICY},
        {
            "check": "simulator_module",
            "value": "research/analyze_ob_candidate_v3_phase1.py",
        },
        {
            "check": "simulator_functions",
            "value": "simulate + apply_policy",
        },
        {
            "check": "reward_source_column",
            "value": "primary_reward_R",
        },
        {
            "check": "reward_equals_gross_R_h12",
            "value": bool(
                np.allclose(
                    trade["primary_reward_R"].to_numpy(float),
                    trade["gross_R_h12"].to_numpy(float),
                    equal_nan=True,
                )
            ),
        },
    ]
    df = pd.DataFrame(rows)
    df = pd.concat(
        [
            df,
            pd.DataFrame(
                [
                    {
                        "check": f"action_{r['action_name']}",
                        "value": (
                            f"mean_R={r['mean_R']},"
                            f"std_R={r['std_R']},"
                            f"nan={r['nan_count']}"
                        ),
                    }
                    for r in per_action
                ]
            ),
        ],
        ignore_index=True,
    )
    return df


# ------------------------------------------------------------
# 时间切分
# ------------------------------------------------------------


def build_time_split(universe):
    days = sorted(pd.to_datetime(universe["trading_day"]).unique())
    n_days = len(days)

    # 隔离带：最大未来观察 12 根 5 分钟 K 线 = 60 分钟。
    # 60 分钟可能跨自然日（夜盘），因此以「交易日」为单位取 1 天，
    # 保证前一段最后一个事件的前向窗口不可能进入后一段。
    embargo_days = 1

    b1 = int(n_days * 0.60)
    b2 = int(n_days * 0.80)

    train_days = days[: max(b1 - embargo_days, 0)]
    select_days = days[b1 : max(b2 - embargo_days, b1)]
    test_days = days[b2:]

    def mask(dl):
        return pd.to_datetime(universe["trading_day"]).isin(
            dl
        ).to_numpy()

    m_train = mask(train_days)
    m_select = mask(select_days)
    m_test = mask(test_days)

    # 同一事件不得跨区重复。
    if (m_train & m_select).any() or (m_select & m_test).any():
        raise RuntimeError("时间段存在事件重复")
    if (m_train & m_test).any():
        raise RuntimeError("时间段存在事件重复")

    def day_str(dl):
        if not len(dl):
            return None
        return (
            str(pd.Timestamp(dl[0]).date()),
            str(pd.Timestamp(dl[-1]).date()),
        )

    def block(m, dl):
        sub = universe.loc[m]
        return {
            "start_date": day_str(dl)[0],
            "end_date": day_str(dl)[1],
            "trading_days": int(len(dl)),
            "events": int(m.sum()),
            "by_symbol": {
                str(k): int(v)
                for k, v in sub["symbol"]
                .value_counts()
                .items()
            },
        }

    return (
        {
            "split_basis": "trading_day_chronological_60_20_20",
            "max_future_observation_bars": (
                MAX_FUTURE_OBSERVATION_BARS
            ),
            "max_future_observation_minutes": (
                MAX_FUTURE_OBSERVATION_MINUTES
            ),
            "bar_minutes": BAR_MINUTES,
            "embargo_trading_days": embargo_days,
            "embargo_rationale": (
                "最大未来观察长度为入场后 12 根连续 5 分钟 K 线"
                "（60 分钟）；60 分钟可能跨越自然日夜盘边界，"
                "因此以交易日为最小单位取 1 个交易日的隔离带，"
                "确保前段事件的前向窗口不会读取后段行情"
            ),
            "total_trading_days": int(n_days),
            "train": block(m_train, train_days),
            "selection": block(m_select, select_days),
            "test": block(m_test, test_days),
            "covered_events": int(
                m_train.sum() + m_select.sum() + m_test.sum()
            ),
            "universe_events": int(len(universe)),
            "dropped_by_embargo_events": int(
                len(universe)
                - (m_train.sum() + m_select.sum() + m_test.sum())
            ),
        },
        m_train,
        m_select,
        m_test,
    )


# ------------------------------------------------------------
# 主流程
# ------------------------------------------------------------


def git_head() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    state, action = load_state_action()
    universe, eligible, analyzable = build_event_universe(
        state, action
    )
    matrix = build_reward_matrix(action, universe)

    manifest = build_feature_manifest(action, universe)
    causality = build_causality_audit(manifest)
    cand = build_candidate_audit(
        state, eligible, analyzable, universe
    )
    rew = build_reward_matrix_audit(matrix, action, universe)
    split, m_train, m_select, m_test = build_time_split(universe)

    manifest.to_csv(
        RESULTS / "feature_manifest_62.csv",
        index=False,
        encoding="utf-8-sig",
    )
    causality.to_csv(
        RESULTS / "feature_causality_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cand.to_csv(
        RESULTS / "candidate_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rew.to_csv(
        RESULTS / "reward_matrix_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (RESULTS / "time_split_audit.json").write_text(
        json.dumps(split, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    n_state = int((~manifest["varies_across_actions"]).sum())
    n_action_rel = int(manifest["varies_across_actions"].sum())

    contract = {
        "experiment": "rl_62d_v1",
        "stage": "AUDIT_ONLY",
        "git_head": git_head(),
        "research_question": (
            "62维决策时点市场状态是否能帮助模型在7个冻结动作中"
            "做出选择，并在历史测试段形成优于固定动作基准的收益曲线"
        ),
        "feature_contract": {
            "authority": SPEC_MODULE + "::MODEL_FEATURES_V0",
            "total_features": 62,
            "unique_features": 62,
            "single_state_features": n_state,
            "action_relative_features": n_action_rel,
            "physical_groups": {
                k: int(v)
                for k, v in manifest["physical_group"]
                .value_counts()
                .items()
            },
            "missing_rate_max": float(
                manifest["missing_rate"].max()
            ),
        },
        "blocking_conflict": {
            "id": "Q_HEAD_INPUT_GRAIN_CONFLICT",
            "severity": "BLOCKING",
            "summary": (
                "权威 62 维是动作相对编码（candidate x action 粒度），"
                f"其中 {n_action_rel} 维在同一事件的不同动作下取值不同，"
                f"只有 {n_state} 维是事件级单一状态。"
                "7 头 Q 网络要求一个事件一个状态，两者不可直接兼容。"
            ),
            "evidence": {
                "state_table_rows": int(len(state)),
                "action_table_rows": int(len(action)),
                "action_table_grain": "candidate_x_action",
                "no_trade_row_target_fit_nan_rate": 1.0,
            },
            "options": [
                {
                    "id": "P1",
                    "desc": (
                        "改用事件级绝对状态（STATE 表 137 列中的"
                        "action-free 编码），重新定义一版 62 维或更少"
                    ),
                    "impact": "需要新增一轮特征工程，不是复用现有 62 维",
                },
                {
                    "id": "P2",
                    "desc": (
                        "保留 62 维动作相对编码，改为 Q(s,a) 单头网络："
                        "把动作字段置为各动作取值后分别前向，取 7 个标量"
                    ),
                    "impact": (
                        "保留 62 维定义，但架构不同于 7 头向量输出；"
                        "且 NO_TRADE 无交易方向，动作相对字段在其上未定义"
                    ),
                },
                {
                    "id": "P3",
                    "desc": "只用 11 维事件级纯状态",
                    "impact": "维度大幅缩水，偏离 62 维研究目标",
                },
            ],
            "resolution": "PENDING_USER_DECISION",
        },
        "action_space": {
            "names": list(ACTION_NAMES),
            "n_actions": N_ACTIONS,
            "frozen": True,
            "no_trade_index": NO_TRADE_INDEX,
            "dataset_skip_action": DATASET_SKIP_ACTION,
        },
        "reward_contract": {
            "simulator": (
                "research/analyze_ob_candidate_v3_phase1.py"
                "::simulate + apply_policy"
            ),
            "reward_column": "primary_reward_R",
            "reward_version": REWARD_VERSION,
            "primary_horizon": PRIMARY_HORIZON,
            "stop_atr": STOP_ATR,
            "target_R": list(TARGET_R),
            "same_bar_policy": SAME_BAR_POLICY,
            "gross_only": True,
            "no_commission_slippage": True,
        },
        "candidate_contract": {
            "candidate_count": int(len(state)),
            "eligible_candidates": int(eligible.sum()),
            "reference_candidate_count": REFERENCE_CANDIDATE_COUNT,
            "reference_eligible_count": REFERENCE_ELIGIBLE_COUNT,
            "matches_reference": bool(
                len(state) == REFERENCE_CANDIDATE_COUNT
                and int(eligible.sum())
                == REFERENCE_ELIGIBLE_COUNT
            ),
            "final_usable_events": int(len(universe)),
            "excluded_reason_non_contiguous": (
                f"主视野 h{PRIMARY_HORIZON} 前向 12 根 K 线不连续"
            ),
            "bucket_filters_used": [],
            "forbidden_bucket_factors": list(
                FORBIDDEN_BUCKET_FACTORS
            ),
        },
        "baselines": [
            "ZERO_NO_TRADE",
            "FIXED_BEST_ACTION_FROM_TRAIN",
            "RL_62D",
            "ORACLE_PERFECT_HINDSIGHT",
        ],
        "model_contract": {
            "module": "research/rl_62d_core_v1.py",
            "class": "QNetwork",
            "n_features": 62,
            "n_actions": N_ACTIONS,
            "structure_search_forbidden": True,
            "grid_search_forbidden": True,
        },
        "time_split": split,
    }

    (RESULTS / "experiment_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("RL_62D_V1_AUDIT_DONE")
    print(
        "features",
        len(manifest),
        "single_state",
        n_state,
        "action_relative",
        n_action_rel,
    )
    print(
        "candidates",
        len(state),
        "eligible",
        int(eligible.sum()),
        "usable_events",
        len(universe),
    )
    print("reward_matrix", matrix.shape)


if __name__ == "__main__":
    main()
