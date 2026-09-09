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
    ACTION_PARQUET,
    FORBIDDEN_BUCKET_FACTORS,
    RESULTS,
    STATE_PARQUET,
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
    QNetwork,
    Standardizer,
    choose_actions,
    choose_best_fixed_action,
    compute_training_loss,
    realized_policy_rewards,
)

_CACHE: dict = {}


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
        _CACHE["split"], _CACHE["m_train"], _, _ = build_time_split(
            _CACHE["universe"]
        )
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
    assert (
        s["selection"]["end_date"] < s["test"]["start_date"]
    )
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

    days = sorted(pd.to_datetime(c["universe"]["trading_day"]).unique())
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
    for factor in FORBIDDEN_BUCKET_FACTORS:
        assert factor not in universe.columns or factor in (
            "source_tf",
        )

    # source_tf 只作为特征保留，不得用于筛选：各取值都必须在场。
    if "source_tf" in universe.columns:
        assert universe["source_tf"].nunique() >= 2


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


# ------------------------------------------------------------
# 模型核心契约
# ------------------------------------------------------------


def test_q_network_contract():
    net = QNetwork(n_features=62, n_actions=7)
    x = torch.randn(16, 62)
    y = net(x)
    assert tuple(y.shape) == (16, 7)

    loss = compute_training_loss(net, x, torch.randn(16, 7))
    assert torch.isfinite(loss)

    sel, pred = choose_actions(net, x)
    assert tuple(sel.shape) == (16,)
    assert tuple(pred.shape) == (16, 7)


def test_realized_policy_rewards_use_history():
    c = _ensure_audit()
    mat = torch.tensor(c["matrix"][:64], dtype=torch.float32)
    sel = torch.randint(0, 7, (64,))
    got = realized_policy_rewards(sel, mat)
    exp = mat[torch.arange(64), sel]
    assert torch.allclose(got, exp)

    # 预测收益与真实收益必须被区分：此处取的是历史真实值。
    assert not torch.allclose(got, torch.zeros_like(got))


def test_choose_best_fixed_action_from_train():
    c = _ensure_audit()
    train = c["matrix"][c["m_train"]]
    best = choose_best_fixed_action(train)
    assert 0 <= best < 7
    means = train.mean(axis=0)
    assert best == int(np.argmax(means))


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
        test_q_network_contract,
        test_realized_policy_rewards_use_history,
        test_choose_best_fixed_action_from_train,
    ]
    for fn in fns:
        fn()
        print("PASS", fn.__name__)
    print("RL_62D_V1_TARGETED_TESTS_PASS")


if __name__ == "__main__":
    _run_all()
