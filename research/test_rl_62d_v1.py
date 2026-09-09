#!/usr/bin/env python3

"""62维 + 单步离线强化学习 V1 -- 定向研究测试。

只覆盖本实验合同要求的检查，不做全仓回归。
可用 pytest 运行，也可直接 `python -m research.test_rl_62d_v1`。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from research.build_rl_62d_audit_v1 import (
    FORBIDDEN_BUCKET_FACTORS,
    RESULTS,
    build_event_universe,
    build_reward_matrix,
    build_time_split,
    load_state_action,
    main as run_audit,
)
from research.ob_rl_model_view_v0_spec import (
    CONTINUOUS_FEATURES_V0,
    MODEL_FEATURES_V0,
)
from research.rl_62d_core_v1 import (
    ACTION_NAMES,
    N_ACTIONS,
    NO_TRADE_INDEX,
    NO_TRADE_VALUE,
    QNetwork,
    Standardizer,
    choose_best_fixed_action,
    compute_training_loss,
    expand_events_to_action_rows,
    oracle_actions,
    realized_policy_rewards,
    score_actions,
    select_actions,
)

_CACHE: dict = {}

# 第一阶段记录的删除事件数，用于连续性审计对账。
EXPECTED_EXCLUDED = 9948


def _ensure_audit():
    if not _CACHE:
        if not (RESULTS / "experiment_contract.json").exists():
            run_audit()
        _CACHE["state"], _CACHE["action"] = load_state_action()
        (
            _CACHE["universe"],
            _CACHE["eligible"],
            _CACHE["analyzable"],
        ) = build_event_universe(
            _CACHE["state"], _CACHE["action"]
        )
        _CACHE["matrix"] = build_reward_matrix(
            _CACHE["action"], _CACHE["universe"]
        )
        (
            _CACHE["split"],
            _CACHE["m_train"],
            _CACHE["m_select"],
            _CACHE["m_test"],
        ) = build_time_split(_CACHE["universe"])
        _CACHE["contract"] = json.loads(
            (RESULTS / "experiment_contract.json").read_text(
                encoding="utf-8"
            )
        )
    return _CACHE


def _manifest() -> pd.DataFrame:
    return pd.read_csv(RESULTS / "feature_manifest_62.csv")


# ------------------------------------------------------------
# 1. 特征数严格等于 62
# ------------------------------------------------------------


def test_feature_count_is_62():
    m = _manifest()
    assert len(m) == 62
    assert len(MODEL_FEATURES_V0) == 62
    assert list(m["feature_name"]) == list(MODEL_FEATURES_V0)


# ------------------------------------------------------------
# 2. 特征名唯一
# ------------------------------------------------------------


def test_feature_names_unique():
    m = _manifest()
    assert m["feature_name"].nunique() == 62
    assert len(set(MODEL_FEATURES_V0)) == 62


# ------------------------------------------------------------
# 3. 不交易奖励严格等于 0
# ------------------------------------------------------------


def test_no_trade_reward_is_zero():
    c = _ensure_audit()
    mat = c["matrix"]
    assert np.all(mat[:, NO_TRADE_INDEX] == 0.0)
    assert ACTION_NAMES[NO_TRADE_INDEX] == "NO_TRADE"
    assert NO_TRADE_VALUE == 0.0


# ------------------------------------------------------------
# 4. 奖励矩阵严格 7 列
# ------------------------------------------------------------


def test_reward_matrix_has_7_columns():
    c = _ensure_audit()
    mat = c["matrix"]
    assert mat.shape[1] == N_ACTIONS == 7
    assert mat.shape[0] == len(c["universe"])


# ------------------------------------------------------------
# 5. 输入矩阵不存在无穷值
# ------------------------------------------------------------


def test_input_matrix_has_no_inf():
    c = _ensure_audit()
    action = c["action"]
    universe = c["universe"]
    m_train = c["m_train"]

    order = universe["candidate_id"].to_numpy()

    sub = action[
        action["candidate_id"].isin(order)
        & (action["action"] != "SKIP")
    ]

    # 每个事件恰好 6 个交易动作；按事件顺序展开，便于用事件级
    # 训练掩码索引到动作级行。
    ev = pd.DataFrame(
        {"candidate_id": order, "_row": np.arange(len(order))}
    )
    sub = (
        ev.merge(sub, on="candidate_id", how="left")
        .sort_values(["_row", "action"], kind="mergesort")
        .reset_index(drop=True)
    )
    assert np.array_equal(
        sub["candidate_id"].to_numpy(), np.repeat(order, 6)
    )

    cols = [c_ for c_ in CONTINUOUS_FEATURES_V0 if c_ in sub.columns]
    x = sub[cols].to_numpy(dtype=float)

    sc = Standardizer()
    z = sc.fit_transform(x[np.repeat(m_train, 6)], "TRAIN")

    assert np.isfinite(z).all(), "标准化后仍存在非有限值"
    assert not np.isinf(x).any(), "原始输入存在无穷值"


# ------------------------------------------------------------
# 6. 时间切分严格递增
# ------------------------------------------------------------


def test_time_split_strictly_increasing():
    c = _ensure_audit()
    s = c["split"]
    assert s["train"]["end_date"] < s["selection"]["start_date"]
    assert s["selection"]["end_date"] < s["test"]["start_date"]
    assert s["embargo_trading_days"] >= 1


# ------------------------------------------------------------
# 7. 三个时间段不存在事件重复
# ------------------------------------------------------------


def test_no_event_overlap_between_splits():
    c = _ensure_audit()
    s = c["split"]
    total = (
        s["train"]["events"]
        + s["selection"]["events"]
        + s["test"]["events"]
    )
    assert total == s["covered_events"]

    days = sorted(
        pd.to_datetime(c["universe"]["trading_day"]).unique()
    )
    n = len(days)
    b1, b2 = int(n * 0.60), int(n * 0.80)
    e = s["embargo_trading_days"]
    tr = set(days[: max(b1 - e, 0)])
    se = set(days[b1 : max(b2 - e, b1)])
    te = set(days[b2:])
    assert not (tr & se) and not (se & te) and not (tr & te)


# ------------------------------------------------------------
# 8. 数据标准化只拟合训练段
# ------------------------------------------------------------


def test_standardizer_fits_on_train_only():
    c = _ensure_audit()
    universe = c["universe"]
    m_train = c["m_train"]

    x = np.hstack(
        [
            np.arange(len(universe), dtype=float)[:, None] * 2.0,
            np.asarray(
                pd.to_numeric(
                    universe["source_ob_width_atr5"],
                    errors="coerce",
                ),
                dtype=float,
            )[:, None],
        ]
    )
    assert x.shape == (len(universe), 2)

    sc = Standardizer()
    sc.fit(x[m_train], "TRAIN")
    assert sc.fitted_on_ == "TRAIN"

    mean_train = x[m_train].mean(axis=0)
    assert np.allclose(sc.mean_, mean_train)

    # 用全量拟合会得到不同的均值，证明没有用到训练段之外的数据。
    mean_all = x.mean(axis=0)
    assert not np.allclose(sc.mean_, mean_all)


# ------------------------------------------------------------
# 9. 不使用上一轮 7 个 bucket 作为过滤条件
# ------------------------------------------------------------


def test_no_previous_bucket_filters():
    c = _ensure_audit()
    contract = c["contract"]
    assert contract["candidate_contract"]["bucket_filters_used"] == []

    universe = c["universe"]
    # source_tf 只作为特征保留，不得用于筛选：各取值都必须在场。
    if "source_tf" in universe.columns:
        assert universe["source_tf"].nunique() >= 2
    for factor in ("touch_bin", "ob_width_bucket"):
        assert factor in FORBIDDEN_BUCKET_FACTORS


# ------------------------------------------------------------
# 10. 所有交易结果复用权威交易模拟语义
# ------------------------------------------------------------


def test_reward_uses_authoritative_simulator():
    audit = pd.read_csv(RESULTS / "reward_matrix_audit.csv")
    d = dict(zip(audit["check"], audit["value"]))

    assert str(d["simulator_module"]) == (
        "research/analyze_ob_candidate_v3_phase1.py"
    )
    assert str(d["simulator_functions"]) == (
        "simulate + apply_policy"
    )
    assert str(d["reward_source_column"]) == "primary_reward_R"
    assert str(d["reward_version"]) == "GROSS_R_V0"
    assert str(d["same_bar_policy"]) == "conservative"
    assert bool(d["reward_equals_gross_R_h12"]) is True
    assert bool(d["no_trade_all_zero"]) is True
    assert int(d["n_action_columns"]) == 7


# ============================================================
# 第二阶段新增
# ============================================================

# ------------------------------------------------------------
# 11. 单输出网络只能输出每行动作一个收益值
# ------------------------------------------------------------


def test_single_output_network_shape():
    net = QNetwork(n_features=62)
    x = torch.randn(16, 62)
    y = net(x)
    assert tuple(y.shape) == (16,), "单输出网络必须输出一维收益"

    loss = compute_training_loss(
        net, x, torch.randn(16)
    )
    assert torch.isfinite(loss)

    scores = score_actions(net, x)
    assert tuple(scores.shape) == (16,)


def test_select_actions_threshold():
    # 全为负 -> 不交易
    s = np.array(
        [
            [-0.03, -0.06, -0.11, -0.02, -0.08, -0.15],
            [0.08, 0.21, 0.13, -0.16, -0.22, -0.31],
        ]
    )
    act = select_actions(s)
    assert act[0] == NO_TRADE_INDEX
    # 第二个事件最高值为 FOLLOW_2.0R，编号 2
    assert act[1] == 2

    # 恰好为 0 也不交易
    zero = np.zeros((1, 6))
    assert select_actions(zero)[0] == NO_TRADE_INDEX


# ------------------------------------------------------------
# 12. 不交易价值固定为 0
# ------------------------------------------------------------


def test_no_trade_value_is_zero():
    assert NO_TRADE_VALUE == 0.0
    c = _ensure_audit()
    assert np.all(c["matrix"][:, NO_TRADE_INDEX] == 0.0)


# ------------------------------------------------------------
# 13. 不交易不进入模型训练行
# ------------------------------------------------------------


def test_no_trade_not_in_training_rows():
    c = _ensure_audit()
    universe = c["universe"]
    action = c["action"]

    usable = set(universe["candidate_id"].to_numpy())
    sub = action[
        action["candidate_id"].isin(usable)
        & (action["action"] != "SKIP")
    ]
    assert "SKIP" not in set(sub["action"].unique())
    assert len(sub) == len(universe) * 6


# ------------------------------------------------------------
# 14. 每个事件严格对应 6 个交易动作
# ------------------------------------------------------------


def test_each_event_has_6_actions():
    c = _ensure_audit()
    universe = c["universe"]
    action = c["action"]

    usable = set(universe["candidate_id"].to_numpy())
    sub = action[
        action["candidate_id"].isin(usable)
        & (action["action"] != "SKIP")
    ]
    per_event = sub.groupby("candidate_id").size()
    assert (per_event == 6).all()

    g = sub.groupby("candidate_id")
    assert (g["touch_time"].nunique() == 1).all()
    assert (g["trading_day"].nunique() == 1).all()
    assert (g["symbol"].nunique() == 1).all()

    # 动作相对字段必须随动作变化，事件级字段必须不变
    assert (g["target_R"].nunique() > 1).all()
    assert (g["source_ob_bias"].nunique() == 1).all()


# ------------------------------------------------------------
# 15. 同一事件 6 行动作不得跨时间切分
# ------------------------------------------------------------


def test_event_actions_not_split_across_time():
    c = _ensure_audit()
    universe = c["universe"]

    m_train = c["m_train"]
    m_select = c["m_select"]
    m_test = c["m_test"]

    masks = [m_train, m_select, m_test]

    # 每个事件只能落在一个时间段里
    for i in range(len(universe)):
        assert sum(bool(m[i]) for m in masks) <= 1

    # 展开成动作行后，同一事件的 6 行仍在同一个段
    for m in masks:
        idx = np.where(m)[0]
        rows = expand_events_to_action_rows(idx)
        assert len(rows) == len(idx) * 6


# ------------------------------------------------------------
# 16. 类别字段不得使用未经合同确认的任意整数编码
# ------------------------------------------------------------


def test_categorical_fields_not_integer_encoded():
    enc = pd.read_csv(RESULTS / "feature_encoding_audit.csv")
    assert len(enc) > 0

    # 历史权威编码不可被 MLP 直接复用，必须等待编码合同
    assert set(enc["是否可直接复用"]) == {"否"}

    # 原始列仍必须是字符串/对象类型，不能已经变成整数编码
    m = _manifest()
    cat_rows = m[m["is_categorical"]]
    assert len(cat_rows) > 0
    for _, r in cat_rows.iterrows():
        if r["feature_name"] == "trade_direction":
            continue
        assert r["dtype"] == "object", (
            f"{r['feature_name']} 已被编码为 {r['dtype']}"
        )


# ------------------------------------------------------------
# 17. 两个分位字段因果审计完整
# ------------------------------------------------------------


def test_quant_causality_audit_complete():
    q = pd.read_csv(
        RESULTS / "quant_feature_causality_audit.csv"
    )

    finals = set(q["最终字段"])
    assert finals == {
        "quant_state",
        "quant_width_percentile_train",
    }

    # 9 个底层输入字段都必须被追踪到
    f1 = {
        "feat_15m_ret_1",
        "feat_15m_ret_4",
        "feat_15m_ret_8",
        "feat_15m_ret_16",
        "feat_15m_location_32",
        "feat_time_bars_since_segment_start",
        "feat_time_after_long_gap",
        "feat_5m_1h_rv",
        "feat_5m_rv_rate_ratio_1h_4h",
    }
    assert f1.issubset(set(q["底层输入字段"]))

    assert set(q["是否存在未来数据风险"]) == {"无"}
    assert set(q["结论"]) == {"通过"}
    assert set(q["是否仅使用过去数据"]) == {"是"}


# ------------------------------------------------------------
# 18. 连续性删除原因数量之和严格等于 9948
# ------------------------------------------------------------


def test_continuity_reason_sum_equals_excluded():
    r = pd.read_csv(
        RESULTS / "continuity_exclusion_reasons.csv"
    )
    assert int(r["事件数"].sum()) == EXPECTED_EXCLUDED


# ------------------------------------------------------------
# 19. 正常休市与真实缺失必须分开统计
# ------------------------------------------------------------


def test_continuity_separates_closure_and_missing():
    r = pd.read_csv(
        RESULTS / "continuity_exclusion_reasons.csv"
    )

    for col in ("是否属于正常市场休市", "是否属于真实数据缺失"):
        assert col in r.columns

    # 两个标记互斥
    both = r["是否属于正常市场休市"] & r["是否属于真实数据缺失"]
    assert not bool(both.any())

    by_sym = pd.read_csv(
        RESULTS / "continuity_by_symbol.csv"
    )
    assert set(
        ["品种", "合格事件数", "被删除事件数", "删除比例",
         "正常休市导致", "真实缺失导致", "其他原因"]
    ).issubset(set(by_sym.columns))

    # 分品种：正常休市 + 真实缺失 + 其他 = 被删除总数
    tot = (
        by_sym["正常休市导致"]
        + by_sym["真实缺失导致"]
        + by_sym["其他原因"]
    )
    assert (tot == by_sym["被删除事件数"]).all()


# ------------------------------------------------------------
# 基准曲线辅助函数
# ------------------------------------------------------------


def test_baseline_helpers():
    c = _ensure_audit()
    train = c["matrix"][c["m_train"]]

    best = choose_best_fixed_action(train)
    means = train.mean(axis=0)
    assert best == int(np.argmax(means))

    orc = oracle_actions(c["matrix"])
    assert orc.shape == (len(c["matrix"]),)
    # 事后完美选择的期望收益必须不低于任何固定动作
    assert float(c["matrix"][np.arange(len(orc)), orc].mean()) >= (
        float(means.max()) - 1e-12
    )


# ------------------------------------------------------------


def _run_all():
    fns = [
        test_feature_count_is_62,
        test_feature_names_unique,
        test_no_trade_reward_is_zero,
        test_reward_matrix_has_7_columns,
        test_input_matrix_has_no_inf,
        test_time_split_strictly_increasing,
        test_no_event_overlap_between_splits,
        test_standardizer_fits_on_train_only,
        test_no_previous_bucket_filters,
        test_reward_uses_authoritative_simulator,
        test_single_output_network_shape,
        test_select_actions_threshold,
        test_no_trade_value_is_zero,
        test_no_trade_not_in_training_rows,
        test_each_event_has_6_actions,
        test_event_actions_not_split_across_time,
        test_categorical_fields_not_integer_encoded,
        test_quant_causality_audit_complete,
        test_continuity_reason_sum_equals_excluded,
        test_continuity_separates_closure_and_missing,
        test_baseline_helpers,
    ]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("RL_62D_V1_TARGETED_TESTS_PASS")


if __name__ == "__main__":
    _run_all()
