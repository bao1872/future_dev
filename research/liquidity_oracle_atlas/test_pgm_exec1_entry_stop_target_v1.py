"""
test_pgm_exec1_entry_stop_target_v1.py

PGM-EXEC-1 测试套件：
- Future Path Tensor: 高级索引、多空对称、越界/discontinuity/时间间隔/跨日排除、基线对齐
- MAE / MFE 数学模型: 手工已知路径数值精确校验、多空对称
- 成交模型 (Fill): k=0 首根即刻成交、pullback 挂单等待成交、未成交、多空对称
- 止损模型 (Stop): 正常盘中止损、穿空止损 (gap loss)、入场根止损、止损先于止盈、同根双触 STOP FIRST
- 止盈模型 (Target): 次根止盈、穿空止盈（无 favorable improvement）、入场根严禁止盈、NONE 绝不止盈
- 超时模型 (Timeout): 无止损止盈下第 fill_idx+5 根收盘平仓、晚入场保证 6 根持仓
- 向量化 vs 标量基准对照 (Vectorized vs Scalar Reference): 随机样本全要素 100% 精确对齐 (atol <= 1e-12)
- 125 策略表面与 Core 数量: exact 125 策略、36 core、12 stop-only core、TB3 表面禁令
- 稳健邻域选择 (Robust Selection): 抵御单点孤立噪点、邻域中位数、正邻域比例、最小成交门限、确定性 Tie-Break
- Bootstrap 对比: 四项对比日聚类 Paired Bootstrap 接线
- 治理门禁与产物校验器: 环境变量授权、旧产物阻断、Fail-closed 校验与变异测试
- 性能与源码审计: 核心热路径禁止逐行/逐笔循环
- 真实管线单次拟合测试: session fixture 保证真实 Window A/B 仅拟合一次
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import sys
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_pgm_exec1_entry_stop_target_v1 as x1
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0d_acceleration_terminal_outcome_v1 as d0
import research.liquidity_oracle_atlas.experiment_pgm_native0e_consensus_acceleration_v1 as e0


# ===========================================================================
# 辅助函数：构造合成数据 (Synthetic Data Helpers)
# ===========================================================================
def make_synthetic_tensor(
    n: int = 10,
    n_future: int = 8,
    direction: float = 1.0,
    atr0: float = 10.0,
    base_price: float = 100.0,
) -> Dict[str, Any]:
    """生成简单合成未来路径张量字典。"""
    O = np.full((n, n_future), base_price, dtype=np.float64)
    H = np.full((n, n_future), base_price + 2.0, dtype=np.float64)
    L = np.full((n, n_future), base_price - 2.0, dtype=np.float64)
    C = np.full((n, n_future), base_price + 1.0, dtype=np.float64)
    T = np.full((n, n_future), np.datetime64("2024-01-02T09:30:00"), dtype="datetime64[ns]")
    DAY = np.full((n, n_future), np.datetime64("2024-01-02"), dtype="datetime64[ns]")
    DISC = np.zeros((n, n_future), dtype=bool)

    dir_arr = np.full(n, direction, dtype=np.float64)
    atr_arr = np.full(n, atr0, dtype=np.float64)
    hazard_arr = np.zeros(n, dtype=np.int64)
    entry_day = np.array([f"2024-01-{(i % 10) + 1:02d}" for i in range(n)])
    symbol = np.array([f"SYM_{i % 3}" for i in range(n)])
    path_valid = np.ones(n, dtype=bool)

    return dict(
        O=O,
        H=H,
        L=L,
        C=C,
        T=T,
        DAY=DAY,
        DISC=DISC,
        direction=dir_arr,
        atr0=atr_arr,
        hazard=hazard_arr,
        entry_day=entry_day,
        symbol=symbol,
        path_valid=path_valid,
        disc_excluded=np.zeros(n, dtype=bool),
        gap_excluded=np.zeros(n, dtype=bool),
        day_excluded=np.zeros(n, dtype=bool),
    )


# ===========================================================================
# A. Future Tensor Tests
# ===========================================================================
def test_future_tensor_shape_and_advanced_indexing():
    """测试 build_future_tensor 高级索引提取与张量形状 (N, 8)。"""
    # 构造单个 symbol 的 20 根 K 线
    sym = "AGL8"
    n_bars = 20
    t_arr = pd.date_range("2024-01-02 09:00:00", periods=n_bars, freq="5min").to_numpy(dtype="datetime64[ns]")
    day_arr = np.full(n_bars, np.datetime64("2024-01-02", "ns"), dtype="datetime64[ns]")
    disc_arr = np.zeros(n_bars, dtype=bool)
    o_arr = np.arange(100.0, 100.0 + n_bars)
    h_arr = o_arr + 2.0
    l_arr = o_arr - 2.0
    c_arr = o_arr + 1.0

    bars_by_sym = {
        sym: dict(
            o=o_arr, h=h_arr, l=l_arr, c=c_arr,
            t=t_arr, day=day_arr, disc=disc_arr, n=n_bars,
        )
    }

    eval_df = pd.DataFrame(
        {
            "symbol": [sym, sym],
            "entry_bar": [2, 5],
            "base_action": [1.0, -1.0],
            "atr0": [5.0, 5.0],
            "hazard": [0, 1],
            "entry_day": ["2024-01-02", "2024-01-02"],
        }
    )

    tensor = x1.build_future_tensor(eval_df, bars_by_sym, n_future=8)
    assert tensor["O"].shape == (2, 8)
    assert tensor["H"].shape == (2, 8)
    assert tensor["L"].shape == (2, 8)
    assert tensor["C"].shape == (2, 8)
    assert tensor["path_valid"].all()

    # 验证第一行 (entry_bar=2)：未来 8 根对应原数组 index 2..9
    np.testing.assert_array_equal(tensor["O"][0], o_arr[2:10])
    np.testing.assert_array_equal(tensor["C"][0], c_arr[2:10])
    # 验证第二行 (entry_bar=5)：未来 8 根对应原数组 index 5..13
    np.testing.assert_array_equal(tensor["O"][1], o_arr[5:13])


def test_future_tensor_out_of_range_invalid():
    """测试未来路径越界时标记 path_valid=False。"""
    sym = "AGL8"
    n_bars = 10
    t_arr = pd.date_range("2024-01-02 09:00:00", periods=n_bars, freq="5min").to_numpy(dtype="datetime64[ns]")
    bars_by_sym = {
        sym: dict(
            o=np.ones(n_bars), h=np.ones(n_bars), l=np.ones(n_bars), c=np.ones(n_bars),
            t=t_arr, day=np.full(n_bars, np.datetime64("2024-01-02", "ns"), dtype="datetime64[ns]"),
            disc=np.zeros(n_bars, dtype=bool), n=n_bars,
        )
    }

    # entry_bar=5 + 8 = 13 > 10 => 越界；entry_bar=0 => t = -1 => 越界
    eval_df = pd.DataFrame(
        {
            "symbol": [sym, sym, sym],
            "entry_bar": [1, 5, 0],
            "base_action": [1.0, 1.0, 1.0],
            "atr0": [1.0, 1.0, 1.0],
            "hazard": [0, 0, 0],
            "entry_day": ["2024-01-02"] * 3,
        }
    )

    tensor = x1.build_future_tensor(eval_df, bars_by_sym, n_future=8)
    # 仅第 0 行 (entry_bar=1, 1+8=9 <= 10, e>=1) 合法
    assert tensor["path_valid"][0] == True
    assert tensor["path_valid"][1] == False
    assert tensor["path_valid"][2] == False


def test_future_tensor_discontinuity_and_gap_and_day_exclusions():
    """测试遇到 discontinuity、时间间隔 >10min 或跨交易日时的排除机制。"""
    sym = "AGL8"
    n_bars = 15
    t_arr = pd.date_range("2024-01-02 09:00:00", periods=n_bars, freq="5min").to_numpy(dtype="datetime64[ns]")
    day_arr = np.full(n_bars, np.datetime64("2024-01-02", "ns"), dtype="datetime64[ns]")
    disc_arr = np.zeros(n_bars, dtype=bool)

    # 1. 设置 discontinuity
    disc_arr[4] = True

    # 2. 设置时间间隔 > 10 min (设置第 10 根开始延迟 15 min)
    t_arr[10:] = t_arr[10:] + np.timedelta64(15, "m")

    # 3. 设置跨日
    day_arr[13:] = np.datetime64("2024-01-03", "ns")

    bars_by_sym = {
        sym: dict(
            o=np.ones(n_bars), h=np.ones(n_bars), l=np.ones(n_bars), c=np.ones(n_bars),
            t=t_arr, day=day_arr, disc=disc_arr, n=n_bars,
        )
    }

    eval_df = pd.DataFrame(
        {
            "symbol": [sym, sym, sym],
            "entry_bar": [1, 4, 6],  # 1 跨越 disc 4; 4 跨越 gap 10; 6 跨越 day 13
            "base_action": [1.0, 1.0, 1.0],
            "atr0": [1.0, 1.0, 1.0],
            "hazard": [0, 0, 0],
            "entry_day": ["2024-01-02"] * 3,
        }
    )

    tensor = x1.build_future_tensor(eval_df, bars_by_sym, n_future=8)
    assert tensor["disc_excluded"][0] == True
    assert tensor["path_valid"][0] == False

    assert tensor["gap_excluded"][1] == True
    assert tensor["path_valid"][1] == False

    assert tensor["day_excluded"][2] == True
    assert tensor["path_valid"][2] == False


def test_assert_allowed_blocks_forbidden_tb4():
    """测试严格禁止 TB4。"""
    df_clean = pd.DataFrame({"block": ["TB1", "TB2", "TB3"]})
    x1.assert_allowed_blocks(df_clean)  # 应该正常通过

    df_tb4 = pd.DataFrame({"block": ["TB1", "TB4"]})
    with pytest.raises(SystemExit) as exc_info:
        x1.assert_allowed_blocks(df_tb4)
    assert "STOP_PGM_EXEC1_FORBIDDEN_TB4" in str(exc_info.value)


def test_baseline_parity_hard_gate():
    """测试基线硬对齐门禁：d_t * (C_{t+1} - O_{t+1}) / ATR0 与 pi 的 max |diff| <= 1e-12。"""
    tensor = make_synthetic_tensor(n=5)
    # baseline_gross = direction * (C[:, 0] - O[:, 0]) / atr0
    # direction=1, C=101, O=100, atr0=10 => (101 - 100)/10 = 0.1
    pi_exact = np.full(5, 0.1, dtype=np.float64)
    err = x1.verify_baseline_parity(tensor, pi_exact)
    assert err <= 1e-12

    # 微扰导致失败
    pi_bad = pi_exact.copy()
    pi_bad[0] += 1e-4
    with pytest.raises(SystemExit) as exc_info:
        x1.verify_baseline_parity(tensor, pi_bad)
    assert "STOP_PGM_EXEC1_BASELINE_PARITY_FAIL" in str(exc_info.value)


# ===========================================================================
# B. MAE / MFE Tests
# ===========================================================================
def test_mae_mfe_handcrafted_path_exact():
    """手工已知 6-bar 路径严格测试 MAE / MFE 数值正确性与多空对称性。"""
    # 构造多头单 (d = +1, E0 = 100, ATR0 = 10)
    # Bar 0: O=100, H=105, L=98, C=102 -> F0=0.5, A0=-0.2
    # Bar 1: O=102, H=112, L=101, C=110 -> F1=1.2, A1=+0.1
    # Bar 2: O=110, H=115, L=95, C=105 -> F2=1.5, A2=-0.5
    # Bar 3..5: 保持不变
    O_long = np.array([[100, 102, 110, 105, 105, 105, 105, 105]], dtype=float)
    H_long = np.array([[105, 112, 115, 106, 106, 106, 106, 106]], dtype=float)
    L_long = np.array([[98, 101, 95, 104, 104, 104, 104, 104]], dtype=float)
    C_long = np.array([[102, 110, 105, 105, 105, 105, 105, 105]], dtype=float)

    tensor_long = dict(
        O=O_long, H=H_long, L=L_long, C=C_long,
        direction=np.array([1.0]), atr0=np.array([10.0]),
        hazard=np.array([0]), entry_day=np.array(["2024-01-02"]),
        symbol=np.array(["AGL8"]), path_valid=np.array([True]),
    )

    res_long = x1.compute_mae_mfe(tensor_long)

    # h = 1: MFE = 0.5, MAE = 0.2
    assert pytest.approx(res_long[1]["mfe"][0], 1e-12) == 0.5
    assert pytest.approx(res_long[1]["mae"][0], 1e-12) == 0.2

    # h = 3: MFE = max(0.5, 1.2, 1.5) = 1.5; MAE = max(0, -min(-0.2, 0.1, -0.5)) = 0.5
    assert pytest.approx(res_long[3]["mfe"][0], 1e-12) == 1.5
    assert pytest.approx(res_long[3]["mae"][0], 1e-12) == 0.5

    # 构造完全对称空头单 (d = -1, E0 = 100, ATR0 = 10)
    # 多头价格偏离 Delta = P - 100, 空头价格偏离为 -Delta => P_short = 100 - (P_long - 100) = 200 - P_long
    # 空头 High 对应 200 - L_long, 空头 Low 对应 200 - H_long
    O_short = 200.0 - O_long
    H_short = 200.0 - L_long
    L_short = 200.0 - H_long
    C_short = 200.0 - C_long

    tensor_short = dict(
        O=O_short, H=H_short, L=L_short, C=C_short,
        direction=np.array([-1.0]), atr0=np.array([10.0]),
        hazard=np.array([0]), entry_day=np.array(["2024-01-02"]),
        symbol=np.array(["AGL8"]), path_valid=np.array([True]),
    )

    res_short = x1.compute_mae_mfe(tensor_short)
    assert pytest.approx(res_short[1]["mfe"][0], 1e-12) == 0.5
    assert pytest.approx(res_short[1]["mae"][0], 1e-12) == 0.2
    assert pytest.approx(res_short[3]["mfe"][0], 1e-12) == 1.5
    assert pytest.approx(res_short[3]["mae"][0], 1e-12) == 0.5


# ===========================================================================
# C. Fill Tests
# ===========================================================================
def test_fill_k0_first_bar_immediate():
    """测试 k=0 时 limit 即为次根 open，必在 bar 0 立即成交。"""
    tensor = make_synthetic_tensor(n=4, direction=1.0)
    rel = x1.precompute_relative_path_for_k(tensor, k=0.0)
    assert rel["filled"].all()
    np.testing.assert_array_equal(rel["fill_idx"], np.zeros(4, dtype=int))


def test_fill_pullback_timing_and_no_fill():
    """测试 pullback 挂单成交时序 (bar 0, 1, 2) 以及超时未成交。"""
    # 4 个样本，多头 (d=1, O0=100, ATR0=10, k=0.10 => Ek = 100 - 1 = 99)
    # Row 0: Bar 0 Low=98 <= 99 => fill_idx=0
    # Row 1: Bar 0 Low=100 > 99; Bar 1 Low=98.5 <= 99 => fill_idx=1
    # Row 2: Bar 0 Low=100, Bar 1 Low=100; Bar 2 Low=99.0 <= 99 => fill_idx=2
    # Row 3: 前 3 根 Low 均 >= 99.5; Bar 3 Low=95 (但已超 3 根等待期) => NO_FILL
    O = np.full((4, 8), 100.0)
    H = np.full((4, 8), 105.0)
    L = np.full((4, 8), 100.0)
    C = np.full((4, 8), 101.0)

    L[0, 0] = 98.0
    L[1, 1] = 98.5
    L[2, 2] = 99.0
    L[3, 3] = 95.0  # 第 4 根才跌破，但此时已超时

    tensor = dict(
        O=O, H=H, L=L, C=C, direction=np.ones(4), atr0=np.full(4, 10.0),
        hazard=np.zeros(4, dtype=int), entry_day=np.array(["2024-01-02"] * 4),
        symbol=np.array(["AGL8"] * 4), path_valid=np.ones(4, dtype=bool),
    )

    rel = x1.precompute_relative_path_for_k(tensor, k=0.10)
    assert rel["filled"][0] == True and rel["fill_idx"][0] == 0
    assert rel["filled"][1] == True and rel["fill_idx"][1] == 1
    assert rel["filled"][2] == True and rel["fill_idx"][2] == 2
    assert rel["filled"][3] == False and rel["fill_idx"][3] == -1


# ===========================================================================
# D. Stop Tests
# ===========================================================================
def test_stop_normal_intrabar_vs_gap_through():
    """测试盘中止损 (收益 -s) 与开盘穿空止损 (承担 gap loss)。"""
    # 2 个样本，多头 d=1, O0=100, ATR0=10, k=0 => Ek=100. stop_s = 0.20 => 止损价 98.0
    # Row 0: Bar 1 Open=99, Low=97.5 <= 98 => 普通盘中止损，收益固定为 -0.20
    # Row 1: Bar 1 Open=97.0 <= 98 => 开盘穿空止损，收益为 (97.0 - 100)/10 = -0.30
    O = np.full((2, 8), 100.0)
    H = np.full((2, 8), 102.0)
    L = np.full((2, 8), 99.5)
    C = np.full((2, 8), 100.5)

    # Row 0
    O[0, 1] = 99.0
    L[0, 1] = 97.5

    # Row 1
    O[1, 1] = 97.0
    L[1, 1] = 96.5

    tensor = dict(
        O=O, H=H, L=L, C=C, direction=np.ones(2), atr0=np.full(2, 10.0),
        hazard=np.zeros(2, dtype=int), entry_day=np.array(["2024-01-02"] * 2),
        symbol=np.array(["AGL8"] * 2), path_valid=np.ones(2, dtype=bool),
    )

    rel = x1.precompute_relative_path_for_k(tensor, k=0.0)
    sim = x1.simulate_policy(rel, stop_s=0.20, target_code="NONE")

    assert sim["exit_reason"][0] == x1.STOP
    assert pytest.approx(sim["gross_return"][0], 1e-12) == -0.20

    assert sim["exit_reason"][1] == x1.STOP
    assert pytest.approx(sim["gross_return"][1], 1e-12) == -0.30


def test_stop_on_entry_bar():
    """测试入场根触碰止损立即触发 STOP，holding_bars=1。"""
    # 多头 d=1, O0=100, ATR0=10, k=0 => Ek=100. stop_s = 0.15 => 止损价 98.5
    # Bar 0 Low=98.0 <= 98.5
    O = np.full((1, 8), 100.0)
    H = np.full((1, 8), 101.0)
    L = np.full((1, 8), 99.5)
    C = np.full((1, 8), 100.0)
    L[0, 0] = 98.0

    tensor = dict(
        O=O, H=H, L=L, C=C, direction=np.ones(1), atr0=np.full(1, 10.0),
        hazard=np.zeros(1, dtype=int), entry_day=np.array(["2024-01-02"]),
        symbol=np.array(["AGL8"]), path_valid=np.ones(1, dtype=bool),
    )
    rel = x1.precompute_relative_path_for_k(tensor, k=0.0)
    sim = x1.simulate_policy(rel, stop_s=0.15, target_code="2.0R")

    assert sim["exit_reason"][0] == x1.STOP
    assert sim["exit_idx"][0] == 0
    assert sim["holding_bars"][0] == 1
    assert pytest.approx(sim["gross_return"][0], 1e-12) == -0.15


def test_same_bar_stop_and_target_stop_first():
    """测试后续同一根 bar 同时触碰止损与止盈时，强制判定 STOP FIRST。"""
    # 多头 d=1, O0=100, ATR0=10, k=0, stop_s=0.20, target="1.5R" (q = 0.30 => 103.0)
    # Bar 0 成交安全
    # Bar 1 同时触碰止损 (Low=97.0 <= 98.0) 和止盈 (High=104.0 >= 103.0)
    O = np.full((1, 8), 100.0)
    H = np.full((1, 8), 101.0)
    L = np.full((1, 8), 99.5)
    C = np.full((1, 8), 100.0)

    O[0, 1] = 100.0
    H[0, 1] = 104.0
    L[0, 1] = 97.0

    tensor = dict(
        O=O, H=H, L=L, C=C, direction=np.ones(1), atr0=np.full(1, 10.0),
        hazard=np.zeros(1, dtype=int), entry_day=np.array(["2024-01-02"]),
        symbol=np.array(["AGL8"]), path_valid=np.ones(1, dtype=bool),
    )
    rel = x1.precompute_relative_path_for_k(tensor, k=0.0)
    sim = x1.simulate_policy(rel, stop_s=0.20, target_code="1.5R")

    # 必须裁定为 STOP FIRST
    assert sim["exit_reason"][0] == x1.STOP
    assert sim["exit_idx"][0] == 1
    assert pytest.approx(sim["gross_return"][0], 1e-12) == -0.20


# ===========================================================================
# E. Target Tests
# ===========================================================================
def test_target_entry_bar_forbidden_and_gap_through_no_improvement():
    """测试入场根严禁止盈，以及跳空超过止盈位时不给 favorable improvement。"""
    # 2 个样本，多头 d=1, O0=100, ATR0=10, k=0, stop_s=0.20, target="2.0R" (q = 0.40 => 104.0)
    # Row 0: Bar 0 High=105.0 >= 104.0 (入场根达到止盈位，但严禁止盈！)，Bar 1 High=100.5，Bar 2 High=104.5 (此时才止盈)
    # Row 1: Bar 1 Open=106.0 >= 104.0 (开盘大幅跳空超过 target)，收益严格锁定为 +q = +0.40，不授予 0.60
    O = np.full((2, 8), 100.0)
    H = np.full((2, 8), 100.5)
    L = np.full((2, 8), 99.5)
    C = np.full((2, 8), 100.0)

    # Row 0
    H[0, 0] = 105.0
    H[0, 2] = 104.5

    # Row 1
    O[1, 1] = 106.0
    H[1, 1] = 107.0

    tensor = dict(
        O=O, H=H, L=L, C=C, direction=np.ones(2), atr0=np.full(2, 10.0),
        hazard=np.zeros(2, dtype=int), entry_day=np.array(["2024-01-02"] * 2),
        symbol=np.array(["AGL8"] * 2), path_valid=np.ones(2, dtype=bool),
    )
    rel = x1.precompute_relative_path_for_k(tensor, k=0.0)
    sim = x1.simulate_policy(rel, stop_s=0.20, target_code="2.0R")

    # Row 0 必须在 Bar 2 退出，而不是 Bar 0
    assert sim["exit_reason"][0] == x1.TARGET
    assert sim["exit_idx"][0] == 2
    assert pytest.approx(sim["gross_return"][0], 1e-12) == 0.40

    # Row 1 退出收益严格为 +q = 0.40
    assert sim["exit_reason"][1] == x1.TARGET
    assert sim["exit_idx"][1] == 1
    assert pytest.approx(sim["gross_return"][1], 1e-12) == 0.40


# ===========================================================================
# F. Timeout Tests
# ===========================================================================
def test_timeout_and_holding_bars():
    """测试未触碰止损止盈时在第 fill_idx + 5 根收盘平仓，持有 6 根 K 线。"""
    # 多头 d=1, O0=100, ATR0=10, k=0.10 (Ek=99.0).
    # Bar 0, 1 未成交，Bar 2 成交 (fill_idx=2)
    # 持仓期为 bar 2..7，退出 bar 为 2 + 5 = 7，收盘价 C[7] = 103.0
    # 收益为 (103 - 99)/10 = 0.40，holding_bars = 6
    O = np.full((1, 8), 100.0)
    H = np.full((1, 8), 101.0)
    L = np.full((1, 8), 99.5)
    C = np.full((1, 8), 100.0)

    L[0, 2] = 99.0
    C[0, 7] = 103.0

    tensor = dict(
        O=O, H=H, L=L, C=C, direction=np.ones(1), atr0=np.full(1, 10.0),
        hazard=np.zeros(1, dtype=int), entry_day=np.array(["2024-01-02"]),
        symbol=np.array(["AGL8"]), path_valid=np.ones(1, dtype=bool),
    )
    rel = x1.precompute_relative_path_for_k(tensor, k=0.10)
    sim = x1.simulate_policy(rel, stop_s=0.30, target_code="NONE")

    assert sim["exit_reason"][0] == x1.TIMEOUT
    assert sim["exit_idx"][0] == 7
    assert sim["holding_bars"][0] == 6
    assert pytest.approx(sim["gross_return"][0], 1e-12) == 0.40


# ===========================================================================
# G. Vectorized vs Scalar Reference Parity Tests (最关键正确性测试)
# ===========================================================================
def test_vectorized_vs_scalar_reference_parity_random_sample():
    """生成包含多空、跳空、同根冲突的大随机样本，验证向量化引擎与标量参考逻辑完全对齐。"""
    rng = np.random.default_rng(20260916)
    N = 250
    dirs = rng.choice([-1.0, 1.0], size=N)
    atr0s = rng.uniform(5.0, 20.0, size=N)

    # 生成价格游走
    O = np.zeros((N, 8), dtype=float)
    H = np.zeros((N, 8), dtype=float)
    L = np.zeros((N, 8), dtype=float)
    C = np.zeros((N, 8), dtype=float)

    start_p = rng.uniform(50.0, 500.0, size=N)
    for i in range(N):
        curr = start_p[i]
        for b in range(8):
            o_b = curr + rng.normal(0, 0.5 * atr0s[i])
            c_b = o_b + rng.normal(0, 0.5 * atr0s[i])
            high_cand = max(o_b, c_b) + abs(rng.normal(0, 0.5 * atr0s[i]))
            low_cand = min(o_b, c_b) - abs(rng.normal(0, 0.5 * atr0s[i]))
            O[i, b] = o_b
            C[i, b] = c_b
            H[i, b] = high_cand
            L[i, b] = low_cand
            curr = c_b

    tensor = dict(
        O=O, H=H, L=L, C=C, direction=dirs, atr0=atr0s,
        hazard=rng.integers(0, 2, size=N), entry_day=np.array(["2024-01-02"] * N),
        symbol=np.array(["AGL8"] * N), path_valid=np.ones(N, dtype=bool),
    )

    # 抽取多组 (k, s, target) 测试
    test_configs = [
        (0.00, 0.15, "1.5R"),
        (0.05, 0.20, "2.0R"),
        (0.10, 0.25, "NONE"),
        (0.15, 0.10, "3.0R"),
        (0.20, 0.30, "1.0R"),
    ]

    for k, s, target in test_configs:
        rel = x1.precompute_relative_path_for_k(tensor, k)
        sim_vec = x1.simulate_policy(rel, s, target)

        for row_i in range(N):
            ref = x1.simulate_policy_scalar_reference(tensor, row_i, k, s, target)
            assert sim_vec["filled"][row_i] == ref["filled"], f"Row {row_i} filled mismatch"
            assert sim_vec["fill_idx"][row_i] == ref["fill_idx"], f"Row {row_i} fill_idx mismatch"
            assert sim_vec["exit_reason"][row_i] == ref["exit_reason"], f"Row {row_i} exit_reason mismatch"
            assert sim_vec["exit_idx"][row_i] == ref["exit_idx"], f"Row {row_i} exit_idx mismatch"
            assert sim_vec["holding_bars"][row_i] == ref["holding_bars"], f"Row {row_i} holding_bars mismatch"
            assert pytest.approx(sim_vec["gross_return"][row_i], abs=1e-12) == ref["gross_return"]
            assert pytest.approx(sim_vec["net_return"][row_i], abs=1e-12) == ref["net_return"]


# ===========================================================================
# H. Surface & Core Counts Tests
# ===========================================================================
def test_surface_exact_125_and_core_counts():
    """测试全空间表面必须 exact 125 策略，核心策略 exact 36，仅止损核心 exact 12。"""
    tensor = make_synthetic_tensor(n=20)
    df_surf = x1.evaluate_surface(tensor, eval_block=x1.TB2_BLOCK)
    assert len(df_surf) == 125
    assert int(df_surf["is_core"].sum()) == 36

    so_core_count = int(
        (df_surf["is_core"] & (df_surf["target_code"] == "NONE")).sum()
    )
    assert so_core_count == 12

    # 检查无重复组合键
    keys = list(zip(df_surf["k"], df_surf["stop_s"], df_surf["target_code"]))
    assert len(keys) == len(set(keys))


def test_surface_rejects_tb3_block():
    """测试 evaluate_surface 严格禁止在 TB3 上运行。"""
    tensor = make_synthetic_tensor(n=10)
    with pytest.raises(SystemExit) as exc_info:
        x1.evaluate_surface(tensor, eval_block="TB3")
    assert "STOP_PGM_EXEC1_SURFACE_FORBIDDEN_ON_BLOCK" in str(exc_info.value)


# ===========================================================================
# I. Robust Neighborhood Selection Tests
# ===========================================================================
def test_robust_selection_prefers_neighborhood_over_isolated_outlier():
    """测试稳健选择抵御孤立尖锐极值，选择平滑正邻域策略。"""
    tensor = make_synthetic_tensor(n=50)
    df_surf = x1.evaluate_surface(tensor, eval_block=x1.TB2_BLOCK)

    # 人为修改 EV 构造对抗场景：
    # 让策略 A (k=0.0, s=0.15, t="1.5R") 自身具有暴利 EV (+10.0)，但其周围邻域全部为大亏 (-5.0)
    # 让策略 B (k=0.05, s=0.20, t="2.0R") 自身及全部邻域稳定盈利 (+0.20)
    df_surf["net_EV_per_decision"] = -0.5
    df_surf["n_trades"] = 6000

    # 给策略 B 设置稳定正收益
    mask_b = (
        (np.abs(df_surf["k"] - 0.05) < 1e-5)
        & (df_surf["stop_s"].round(4).isin([0.15, 0.20, 0.25]))
        & (df_surf["target_code"].isin(["1.5R", "2.0R", "3.0R"]))
    )
    df_surf.loc[mask_b, "net_EV_per_decision"] = 0.20

    # 策略 A 自身单点暴利
    mask_a = (
        (np.abs(df_surf["k"] - 0.0) < 1e-5)
        & (np.abs(df_surf["stop_s"] - 0.15) < 1e-5)
        & (df_surf["target_code"] == "1.5R")
    )
    df_surf.loc[mask_a, "net_EV_per_decision"] = 10.0

    # 重新计算稳健指标
    stop_idx_map = {round(val, 4): idx for idx, val in enumerate(x1.STOP_GRID)}
    target_idx_map = {code: idx for idx, code in enumerate(x1.TARGET_CODES)}
    r_scores = []
    p_rates = []
    so_r_scores = []
    so_p_rates = []

    for _, r in df_surf.iterrows():
        curr_k = r["k"]
        curr_s = round(r["stop_s"], 4)
        curr_t = r["target_code"]
        s_idx = stop_idx_map[curr_s]
        t_idx = target_idx_map[curr_t]
        s_nbr_indices = set(range(max(0, s_idx - 1), min(len(x1.STOP_GRID), s_idx + 2)))
        t_nbr_indices = set(range(max(0, t_idx - 1), min(len(x1.TARGET_CODES), t_idx + 2)))
        s_nbr_vals = {round(x1.STOP_GRID[i], 4) for i in s_nbr_indices}
        t_nbr_codes = {x1.TARGET_CODES[i] for i in t_nbr_indices}

        nbr_mask = (
            (np.abs(df_surf["k"] - curr_k) < 1e-5)
            & df_surf["stop_s"].round(4).isin(s_nbr_vals)
            & df_surf["target_code"].isin(t_nbr_codes)
        )
        nbr_evs = df_surf.loc[nbr_mask, "net_EV_per_decision"].to_numpy(float)
        r_scores.append(float(np.median(nbr_evs)))
        p_rates.append(float(np.mean(nbr_evs > 0.0)))
        so_r_scores.append(np.nan)
        so_p_rates.append(np.nan)

    df_surf["robust_score"] = r_scores
    df_surf["positive_neighbor_rate"] = p_rates
    df_surf["stop_only_robust_score"] = so_r_scores
    df_surf["stop_only_positive_neighbor_rate"] = so_p_rates

    sel = x1.select_policies(df_surf)
    primary = sel["primary"]

    # 策略 A 单点 EV=10.0 绝不能获胜，获胜者必须是平滑稳健的策略 B
    assert primary["k"] == 0.05
    assert primary["stop_s"] == 0.20
    assert primary["target_code"] == "2.0R"
    assert primary["selection_pass"] == True


# ===========================================================================
# J. Bootstrap Tests
# ===========================================================================
def test_bootstrap_contrasts_wiring():
    """测试日聚类 Paired Bootstrap 接线与 4 项对比输出格式。"""
    n_days = 20
    rows_per_day = 10
    N = n_days * rows_per_day

    entry_days = np.repeat([f"2024-01-{d+1:02d}" for d in range(n_days)], rows_per_day)
    base_net = np.random.normal(0.01, 0.1, size=N)
    pri_net = base_net + np.random.normal(0.02, 0.05, size=N)
    so_net = base_net + np.random.normal(0.01, 0.05, size=N)

    returns_dict = dict(BASE=base_net, PRIMARY=pri_net, STOP_ONLY=so_net)
    df_boot = x1.run_bootstrap_contrasts(
        entry_days, returns_dict, n_boot=200, seed=20260916
    )

    assert len(df_boot) == 4
    expected_contrasts = [
        "PRIMARY_NET",
        "PRIMARY_MINUS_BASE",
        "STOP_ONLY_NET",
        "STOP_ONLY_MINUS_BASE",
    ]
    assert df_boot["contrast"].tolist() == expected_contrasts
    for c in ["point", "ci95_lower", "ci95_upper", "p_pos"]:
        assert c in df_boot.columns
        assert np.isfinite(df_boot[c]).all()


# ===========================================================================
# K. Governance & Artifact Validation Tests
# ===========================================================================
def test_formal_authorization_gate(monkeypatch):
    """测试未授权时阻断正式运行。"""
    monkeypatch.delenv("AUTHORIZE_PGM_EXEC1_FULL_EXPLORATORY", raising=False)
    with pytest.raises(SystemExit) as exc_info:
        x1.require_full_authorization()
    assert "STOP_PGM_EXEC1_FULL_NOT_AUTHORIZED" in str(exc_info.value)

    monkeypatch.setenv("AUTHORIZE_PGM_EXEC1_FULL_EXPLORATORY", "0")
    with pytest.raises(SystemExit) as exc_info:
        x1.require_full_authorization()
    assert "STOP_PGM_EXEC1_FULL_NOT_AUTHORIZED" in str(exc_info.value)

    monkeypatch.setenv("AUTHORIZE_PGM_EXEC1_FULL_EXPLORATORY", "1")
    x1.require_full_authorization()  # 正常通过


def test_stale_artifact_detection(tmp_path, monkeypatch):
    """测试若存在旧 artifacts 则立即 fail-closed 阻断。"""
    monkeypatch.setattr(x1, "OUT_DIR", tmp_path)
    # 无文件时正常
    x1.assert_no_stale_artifacts()

    # 放入一个旧文件
    (tmp_path / "pgm_exec1_stale.csv").touch()
    with pytest.raises(SystemExit) as exc_info:
        x1.assert_no_stale_artifacts()
    assert "STOP_PGM_EXEC1_ARTIFACT_ALREADY_EXISTS" in str(exc_info.value)


def test_artifact_validator_fail_closed():
    """测试产物校验器对缺失产物、不满足 exact 数量的变异测试。"""
    # 构造假产物映射
    valid_map = {
        x1.ARTIFACT_MAE_MFE_QUANTILES: pd.DataFrame({"block": ["TB2"]}),
        x1.ARTIFACT_MAE_MFE_THRESHOLDS: pd.DataFrame({"block": ["TB2"]}),
        x1.ARTIFACT_ENTRY_ONLY: pd.DataFrame({"block": ["TB2"]}),
        x1.ARTIFACT_SURFACE_TB2: pd.DataFrame(
            {
                "k": np.zeros(125),
                "is_core": [True] * 36 + [False] * 89,
                "target_code": ["NONE"] * 12 + ["1.5R"] * 113,
            }
        ),
        x1.ARTIFACT_SELECTED_POLICIES: pd.DataFrame(
            {"policy_type": ["PRIMARY", "STOP_ONLY_SECONDARY"]}
        ),
        x1.ARTIFACT_TB3_VALIDATION: pd.DataFrame(
            {"policy_name": ["BASE", "PRIMARY", "STOP_ONLY_SECONDARY"]}
        ),
        x1.ARTIFACT_BOOTSTRAP: pd.DataFrame({"contrast": ["A", "B", "C", "D"]}),
        x1.ARTIFACT_FORMAL_SUMMARY: {
            "run_head": "abc", "base_sha": "abc", "sample_hash": "abc", "transition_hash": "abc",
            "counts": {}, "symbols": [], "blocks": [], "window_A": "", "window_B": "",
            "score_owner_parity": {}, "atr0_parity": 0.0, "path_contract": {},
            "FUTURE_BARS": 8, "ENTRY_WAIT_BARS": 3, "MAX_HOLD_BARS": 6, "MAX_GAP_MINUTES": 10,
            "ENTRY_GRID": [], "STOP_GRID": [], "TARGET_CODES": [], "core_grids": {},
            "cost": 0.01, "bootstrap_N": 2000, "bootstrap_seed": 1,
            "tb2_path_valid_count": 1, "tb3_path_valid_count": 1, "baseline_parity_errors": {},
            "selected_PRIMARY": {}, "selected_STOP_ONLY_SECONDARY": {}, "tb2_selection_pass": True,
            "tb3_PRIMARY_metrics": {}, "tb3_bootstrap": [], "formal_verdict": "",
            "known_limitations": [], "timing": {}, "artifact_files": [],
        },
    }

    # 正常通过
    x1.validate_in_memory_artifacts(valid_map)

    # 变异 1：缺少一个文件
    mutated_1 = valid_map.copy()
    del mutated_1[x1.ARTIFACT_BOOTSTRAP]
    with pytest.raises(SystemExit) as exc:
        x1.validate_in_memory_artifacts(mutated_1)
    assert "STOP_PGM_EXEC1_ARTIFACT_KEYS_MISMATCH" in str(exc.value)

    # 变异 2：surface 不是 125 行
    mutated_2 = valid_map.copy()
    mutated_2[x1.ARTIFACT_SURFACE_TB2] = pd.DataFrame({"k": np.zeros(120)})
    with pytest.raises(SystemExit) as exc:
        x1.validate_in_memory_artifacts(mutated_2)
    assert "STOP_PGM_EXEC1_SURFACE_NOT_125" in str(exc.value)


# ===========================================================================
# L. Performance Tests (源码审计)
# ===========================================================================
def test_source_code_audit_no_forbidden_loops():
    """源码静态审计：确保核心函数内部绝无 iterrows/itertuples/逐行循环。"""
    x1.audit_forbidden_patterns_in_hot_loops()


# ===========================================================================
# M. Real Pipeline Fixture & Integration Tests
# ===========================================================================
@pytest.fixture(scope="module")
def real_scored_bundle():
    """模块级 fixture：真实 Window A 与 Window B 全局仅拟合一次。"""
    bundle = x1.load_and_score()
    return bundle


def test_real_pipeline_fit_once_and_baseline_parity(real_scored_bundle):
    """真实数据端到端检验：验证 Window A/B 分值所有权以及 TB2/TB3 基线对齐 (max diff <= 1e-12)。"""
    bundle = real_scored_bundle
    scored_A = bundle["scored_A"]
    scored_B = bundle["scored_B"]
    bars_by_sym = bundle["bars_by_sym"]

    # 验证分值所有权
    d_a = d0.verify_window_score_owner(scored_A, bundle["fit_A"], x1.TB2_BLOCK)
    assert d_a <= 1e-12

    d_b = d0.verify_window_score_owner(scored_B, bundle["fit_B"], x1.TB3_BLOCK)
    assert d_b <= 1e-12

    # 验证 TB2 基线对齐
    eval_tb2 = x1.extract_evaluation_sample(scored_A, x1.TB2_BLOCK)
    t_tb2 = x1.build_future_tensor(eval_tb2, bars_by_sym)
    eval_tb2_val = eval_tb2[t_tb2["path_valid"]].reset_index(drop=True)
    t_tb2_val = x1.slice_future_tensor(t_tb2, t_tb2["path_valid"])
    err_tb2 = x1.verify_baseline_parity(t_tb2_val, eval_tb2_val["pi"].to_numpy(float))
    assert err_tb2 <= 1e-12

    # 验证 TB3 基线对齐
    eval_tb3 = x1.extract_evaluation_sample(scored_B, x1.TB3_BLOCK)
    t_tb3 = x1.build_future_tensor(eval_tb3, bars_by_sym)
    eval_tb3_val = eval_tb3[t_tb3["path_valid"]].reset_index(drop=True)
    t_tb3_val = x1.slice_future_tensor(t_tb3, t_tb3["path_valid"])
    err_tb3 = x1.verify_baseline_parity(t_tb3_val, eval_tb3_val["pi"].to_numpy(float))
    assert err_tb3 <= 1e-12
