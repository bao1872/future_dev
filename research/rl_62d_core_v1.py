#!/usr/bin/env python3

"""62维状态 + 单步离线强化学习 V1 -- 模型核心（第二阶段修订）。

实验目标
--------
验证 62 维决策时点描述，是否能帮助模型在 7 个冻结动作中做出选择，
并在历史测试段形成优于固定动作基准的收益曲线。

模型结构裁决（第二阶段）
------------------------
权威 62 维是「动作相对」编码：同一事件在不同候选交易方案下取值不同。
因此模型不再是「一个状态同时预测 7 个动作」，而是：

    一个事件 x 一个候选交易方案 -> 一行 62 维 -> 一个收益预测值

即状态—动作价值网络 Q(s, a)，输入 62 维，输出 1 个标量。

「不交易」不进入网络，其价值固定为 0（NO_TRADE_VALUE）。
决策时：对 6 个真实交易方案分别评分，取最高；若最高值 <= 0 则不交易。

关键边界
--------
* 一个交易事件 = 一个单步决策。不构造下一状态，不建多步马尔可夫链。
* 收益回测必须使用模型选择动作后**实际发生的历史收益**，
  严禁使用模型预测收益。
* 第一版禁止模型结构搜索、禁止网格搜索、禁止调整收益门槛。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

# ------------------------------------------------------------
# 动作空间（本轮彻底冻结，禁止增删）
# ------------------------------------------------------------

ACTION_NAMES = (
    "NO_TRADE",
    "FOLLOW_1.5R",
    "FOLLOW_2.0R",
    "FOLLOW_2.5R",
    "FADE_1.5R",
    "FADE_2.0R",
    "FADE_2.5R",
)

# 进入价值网络的 6 个真实交易动作（不含不交易）。
TRADE_ACTION_NAMES = ACTION_NAMES[1:]

N_ACTIONS = len(ACTION_NAMES)
N_TRADE_ACTIONS = len(TRADE_ACTION_NAMES)

NO_TRADE_INDEX = 0

# 不交易的价值恒为 0，不参与网络训练。
NO_TRADE_VALUE = 0.0

# 数据集里「不交易」的历史动作名是 SKIP，两者是同一个动作。
DATASET_SKIP_ACTION = "SKIP"

# 主视野：12 根 5 分钟 K 线 = 60 分钟，与数据集 PRIMARY_HORIZON 一致。
PRIMARY_HORIZON = 12

# 单根 K 线分钟数，用于把隔离带换算成时间长度。
BAR_MINUTES = 5

# 最大未来观察长度：入场后向未来读取 12 根连续 5 分钟 K 线。
MAX_FUTURE_OBSERVATION_BARS = PRIMARY_HORIZON
MAX_FUTURE_OBSERVATION_MINUTES = (
    MAX_FUTURE_OBSERVATION_BARS * BAR_MINUTES
)

# 训练粒度合同：一个事件 x 6 个真实交易动作。
TRAINING_ROWS_PER_EVENT = N_TRADE_ACTIONS


# ------------------------------------------------------------
# 网络
# ------------------------------------------------------------


class QNetwork(nn.Module):
    """状态—动作价值网络。

    每一行输入代表：
    当前市场状态与某一个候选交易动作组成的62维描述。

    输出代表：
    当前候选交易动作的预计收益。
    """

    def __init__(self, encoded_feature_count: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(encoded_feature_count, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def compute_training_loss(
    model,
    features,
    realized_rewards,
):
    """学习当前62维状态—动作组合对应的历史实际收益。"""

    predicted_rewards = model(features)

    return nn.functional.smooth_l1_loss(
        predicted_rewards,
        realized_rewards,
    )


# ------------------------------------------------------------
# 决策
# ------------------------------------------------------------


@torch.no_grad()
def score_actions(model, action_features):
    """对 [事件数 x 6, 62] 的动作行逐行打分。

    返回 [事件数 x 6] 的预测收益（按事件顺序展开）。
    """

    model.eval()
    return model(action_features)


def select_actions(scores) -> np.ndarray:
    """把每个事件的 6 个交易动作得分转成最终动作编号。

    scores 形状为 [事件数, 6]。

    规则：
        1. 取预测收益最高的交易动作；
        2. 若最高预测收益 <= 0，则不交易（编号 0）；
        3. 否则返回该交易动作在 ACTION_NAMES 中的编号。

    第一版不设置任何额外门槛。
    """

    if isinstance(scores, torch.Tensor):
        s = scores.detach().cpu().numpy()
    else:
        s = np.asarray(scores, dtype=float)

    if s.ndim != 2 or s.shape[1] != N_TRADE_ACTIONS:
        raise RuntimeError(
            f"动作得分形状错误: {s.shape}, 期望 (N, 6)"
        )

    best_local = np.argmax(s, axis=1)
    best_value = s[np.arange(s.shape[0]), best_local]

    # 交易动作编号从 1 开始，因为 0 号是不交易。
    action = best_local + 1
    action = np.where(
        best_value > 0.0, action, NO_TRADE_INDEX
    )

    return action.astype(int)


def realized_policy_rewards(
    selected_actions,
    reward_matrix,
):
    """根据模型实际选择的动作，取得该动作后来真实发生的历史收益。

    收益回测禁止使用模型预测收益。
    """

    row_index = torch.arange(
        len(selected_actions),
        device=selected_actions.device,
    )

    return reward_matrix[
        row_index,
        selected_actions,
    ]


def choose_best_fixed_action(train_reward_matrix):
    """只使用训练数据，选择历史平均收益最高的固定动作。

    后面的历史测试阶段不得重新选择。
    矩阵含 7 列，不交易列恒为 0，因此若所有交易动作平均为负，
    最优固定动作就是「不交易」。
    """

    mean_rewards = np.asarray(
        train_reward_matrix, dtype=float
    ).mean(axis=0)

    return int(np.argmax(mean_rewards))


def oracle_actions(reward_matrix: np.ndarray) -> np.ndarray:
    """事后选择真实收益最高的动作。

    该基准不可实盘，只用于衡量 7 动作空间的理论上限。
    """

    m = np.asarray(reward_matrix, dtype=float)
    return np.argmax(m, axis=1)


# ------------------------------------------------------------
# 动作粒度：先按事件切分，再展开成 6 行动作
# ------------------------------------------------------------


def expand_events_to_action_rows(event_index: np.ndarray):
    """事件索引 -> 动作行索引映射。

    时间切分必须先按事件完成，再展开成动作行，
    禁止把同一事件的 6 行动作拆到不同时间段。
    """

    event_index = np.asarray(event_index)
    return np.repeat(event_index, TRAINING_ROWS_PER_EVENT)


def assert_action_grain(n_events: int, action_rows: int) -> None:
    expected = n_events * TRAINING_ROWS_PER_EVENT
    if action_rows != expected:
        raise RuntimeError(
            f"动作粒度错误: {action_rows} != {expected}"
        )


# ------------------------------------------------------------
# 标准化：只能在训练段拟合
# ------------------------------------------------------------


class Standardizer:
    """均值/标准差标准化器。

    拟合只允许发生在训练段。缺失值填充值同样只从训练段取，
    禁止用全量数据提前计算。
    """

    def __init__(self):
        self.mean_ = None
        self.std_ = None
        self.fill_ = None
        self.fitted_on_ = None

    def fit(self, x: np.ndarray, split_name: str) -> "Standardizer":
        arr = np.asarray(x, dtype=float)

        # 中位数只在训练段计算，避免把后段分布带进前段。
        self.fill_ = np.nanmedian(arr, axis=0)
        self.fill_ = np.where(
            np.isfinite(self.fill_), self.fill_, 0.0
        )

        filled = np.where(np.isfinite(arr), arr, self.fill_)

        self.mean_ = filled.mean(axis=0)
        self.std_ = filled.std(axis=0)

        # 常量列标准差为 0，退化为 1 以免除零。
        self.std_ = np.where(self.std_ > 0, self.std_, 1.0)

        self.fitted_on_ = split_name
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean_ is None:
            raise RuntimeError("standardizer not fitted")

        arr = np.asarray(x, dtype=float)
        arr = np.where(np.isfinite(arr), arr, self.fill_)

        return (arr - self.mean_) / self.std_

    def fit_transform(
        self, x: np.ndarray, split_name: str
    ) -> np.ndarray:
        return self.fit(x, split_name).transform(x)
