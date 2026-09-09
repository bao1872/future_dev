#!/usr/bin/env python3

"""62维状态 + 单步离线强化学习 V1 -- 模型核心。

实验目标
--------
验证 62 维决策时点市场状态，是否能帮助模型在 7 个冻结动作中做出
选择，并在历史测试段形成优于固定动作基准的收益曲线。

关键边界
--------
* 一个交易事件 = 一个单步决策。不人为构造下一状态，不建立多步
  马尔可夫链。
* 模型输出 7 个动作各自的预期收益，最终选择预期收益最高的动作。
* ``NO_TRADE`` 是正式动作之一，其真实收益恒为 0。
* 收益回测必须使用模型选择动作后**实际发生的历史收益**，
  严禁使用模型预测收益。
* 第一版禁止模型结构搜索、禁止网格搜索。

重要未决问题（见 experiment_contract.json）
------------------------------------------
权威 62 维（``MODEL_FEATURES_V0``）是**动作相对**编码：62 维中有
51 维在同一事件的不同动作下取值不同，无法作为「一个事件一个状态」
直接输入 7 头 Q 网络。本模块按用户给定合同实现
``n_features=62, n_actions=7``，但输入口径需先裁决后再训练。
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

# 数据集里「不交易」的历史动作名是 SKIP，两者是同一个动作。
DATASET_SKIP_ACTION = "SKIP"

N_ACTIONS = len(ACTION_NAMES)

NO_TRADE_INDEX = ACTION_NAMES.index("NO_TRADE")

# 主视野：12 根 5 分钟 K 线 = 60 分钟，与数据集 PRIMARY_HORIZON 一致。
PRIMARY_HORIZON = 12

# 单根 K 线分钟数，用于把隔离带换算成时间长度。
BAR_MINUTES = 5

# 最大未来观察长度：入场后向未来读取 12 根连续 5 分钟 K 线。
MAX_FUTURE_OBSERVATION_BARS = PRIMARY_HORIZON
MAX_FUTURE_OBSERVATION_MINUTES = (
    MAX_FUTURE_OBSERVATION_BARS * BAR_MINUTES
)


# ------------------------------------------------------------
# 网络
# ------------------------------------------------------------


class QNetwork(nn.Module):
    """62维状态价值网络。

    输入为交易决策时刻的62维市场状态。
    输出为7个冻结动作各自的预期收益。
    """

    def __init__(
        self,
        n_features: int = 62,
        n_actions: int = N_ACTIONS,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_features, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


def compute_training_loss(
    model,
    features,
    realized_rewards,
):
    """使用7个动作的历史真实收益，同时训练7个动作价值输出。"""

    predicted_rewards = model(features)

    return nn.functional.smooth_l1_loss(
        predicted_rewards,
        realized_rewards,
    )


@torch.no_grad()
def choose_actions(
    model,
    features,
):
    """为每个市场状态选择预期收益最高的动作。"""

    model.eval()

    predicted_rewards = model(features)

    selected_actions = torch.argmax(
        predicted_rewards,
        dim=1,
    )

    return selected_actions, predicted_rewards


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
    """

    mean_rewards = np.asarray(
        train_reward_matrix, dtype=float
    ).mean(axis=0)

    return int(np.argmax(mean_rewards))


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


# ------------------------------------------------------------
# 事后完美选择基准
# ------------------------------------------------------------


def oracle_actions(reward_matrix: np.ndarray) -> np.ndarray:
    """事后选择真实收益最高的动作。

    该基准不可实盘，只用于衡量 7 动作空间的理论上限。
    """

    m = np.asarray(reward_matrix, dtype=float)
    return np.argmax(m, axis=1)
