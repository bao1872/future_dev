"""
build_m2_temporal_panel.py
=========================

为 M2 实验构建【逐 5 分钟底层市场状态时间面板】，确定性数据产品，只生成一次。

设计（严格遵循用户数据流约束）：
  - 每个品种完整 5 分钟历史：
        load_raw_5m(sym)
        → build_momentum_frame  一次   （冻结、只读）
        → build_dsa_frame       一次   （冻结、只读）
        → 向量化生成时间变化特征
        → 形成逐 5 分钟 TEMPORAL 面板
    AG / CU / RB / M 各一次，总调用：build_momentum_frame = 4 次，build_dsa_frame = 4 次。
    绝不按事件（~20008）或按动作（~120048）重复计算指标。
  - 连续字段时间特征全部向量化（shift / 闭式斜率），无逐 K 线 Python 循环、无 np.polyfit 每窗口、无 apply(axis=1)。
  - 离散状态变化向量化（changed.rolling.max + cumsum.groupby.cumcount）。
  - 只使用绝对底层市场状态（momentum / DSA / 基础波动 / 参与），动作相对转换留到模型脚本。
  - 每个品种单独打印耗时，定位真正瓶颈。

输出：
  research/analysis_results/m2/temporal_market_panel.parquet
  列：symbol, bar_index, bar_time, [原始底层状态], [时间变化特征(__*)]

用法：
  python -m research.build_m2_temporal_panel            # 四品种全量
  python -m research.build_m2_temporal_panel AG        # 仅 AG（验收）
  python -m research.build_m2_temporal_panel AG CU     # 指定品种
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.build_ob_rl_dataset_v0 import build_momentum_frame, build_dsa_frame

PANEL_OUT = Path("research/analysis_results/m2/temporal_market_panel.parquet")
ALL_SYMS = ["AG", "CU", "RB", "M"]

# 连续底层市场状态字段（用于 delta / slope）
CONT = ["sqzmom_val", "sqzmom_delta", "dsa_raw_dsa_vwap_dev_pct", "vol20", "vol_part"]
# 离散底层市场状态字段（用于状态切换）
DISC = ["momentum_direction", "dsa_direction"]


def _add_temporal_numeric(df, columns):
    """向量化生成连续字段时间变化特征。仅 shift + 闭式斜率，无逐行循环。"""
    out = df.copy()
    for col in columns:
        s = pd.to_numeric(out[col], errors="coerce")
        out[f"{col}__delta_1"] = s - s.shift(1)
        out[f"{col}__delta_3"] = s - s.shift(3)
        out[f"{col}__delta_6"] = s - s.shift(6)
        # 最近6根线性斜率（闭式，等价于对 [t-5..t] 做线性回归斜率）
        out[f"{col}__slope_6"] = (
            -2.5 * s.shift(5)
            - 1.5 * s.shift(4)
            - 0.5 * s.shift(3)
            + 0.5 * s.shift(2)
            + 1.5 * s.shift(1)
            + 2.5 * s
        ) / 17.5
    return out


def _add_temporal_state(df, column):
    """向量化计算离散市场状态变化。"""
    state = df[column]
    changed = state.ne(state.shift(1))
    df[f"{column}__changed_last_3"] = (
        changed.rolling(3, min_periods=1).max().astype("int8")
    )
    change_group = changed.cumsum()
    df[f"{column}__bars_since_change"] = change_group.groupby(change_group).cumcount()
    return df


def build_panel(sym):
    """对单个品种构建逐 5 分钟 TEMPORAL 面板（指标仅计算一次）。"""
    t0 = time.perf_counter()
    five = load_raw_5m(sym)
    t_load = time.perf_counter() - t0

    t0 = time.perf_counter()
    mom = build_momentum_frame(five)
    t_mom = time.perf_counter() - t0

    t0 = time.perf_counter()
    dsa = build_dsa_frame(five)
    t_dsa = time.perf_counter() - t0

    # 基础波动 / 参与（从原始 5 分钟 K 线向量化得到，因果：shift(1) 只用过去）
    ret = np.log(five["close"] / five["close"].shift(1))
    vol20 = ret.rolling(20).std().shift(1)
    vmean = five["volume"].rolling(20).mean().shift(1)
    vol_part = five["volume"] / vmean

    # 组装逐 bar 表（bar_index 0..n-1，与 touch_5m_bar_index 同空间）
    df = pd.DataFrame({"bar_index": np.arange(len(five))})
    df["bar_time"] = five["bar_start_time"].values
    df["momentum_direction"] = mom["momentum_direction"].values
    df["sqzmom_val"] = mom["sqzmom_val"].values
    df["sqzmom_delta"] = mom["sqzmom_delta"].values
    df["dsa_direction"] = dsa["dsa_direction"].values
    df["dsa_raw_dsa_vwap_dev_pct"] = dsa["dsa_raw_dsa_vwap_dev_pct"].values
    df["vol20"] = vol20.values
    df["vol_part"] = vol_part.values
    df = df.set_index("bar_index")

    t0 = time.perf_counter()
    df = _add_temporal_numeric(df, CONT)
    for col in DISC:
        df = _add_temporal_state(df, col)
    t_feat = time.perf_counter() - t0

    df = df.reset_index()
    df.insert(0, "symbol", sym)
    return df, dict(load=t_load, mom=t_mom, dsa=t_dsa, feat=t_feat, bars=len(df))


def main():
    syms = [s for s in sys.argv[1:] if s] or ALL_SYMS
    t_all = time.perf_counter()
    parts = []
    for sym in syms:
        df, info = build_panel(sym)
        parts.append(df)
        print(
            f"[{sym}] bars={info['bars']} "
            f"load={info['load']:.2f}s momentum={info['mom']:.2f}s "
            f"dsa={info['dsa']:.2f}s features={info['feat']:.2f}s"
        )
    panel = pd.concat(parts, ignore_index=True)
    PANEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(PANEL_OUT, index=False)
    print(f"\nTOTAL panels built in {time.perf_counter() - t_all:.2f}s")
    print(f"panel rows={len(panel)} cols={panel.shape[1]}")
    print(f"saved -> {PANEL_OUT}")
    tcols = [c for c in panel.columns if "__" in c]
    print(f"temporal feature cols ({len(tcols)}): {tcols}")


if __name__ == "__main__":
    main()
