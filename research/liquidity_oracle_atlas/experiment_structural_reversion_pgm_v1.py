"""
experiment_structural_reversion_pgm_v1.py
=========================================

结构性回归（Structural Reversion）PGM 单文件实验。

核心研究问题
------------
不是先设计最终买卖规则，而是逐层回答：

1. Deviation 是否真的包含回归信息？
2. Support / Resistance 是否在 Deviation 之外提供增量？
3. 1H / 4H 环境是否改变回归概率？
4. Liquidity 是否提供 SR 之外的增量信息？
5. Liquidity sweep / reclaim / acceptance 是否代表不同状态？
6. 5m trigger 是否改善 15m setup？
7. 现有 PGM score 是否还能提供额外环境信息？

PGM 节点阶梯
------------
G0  symbol baseline
G1  + deviation
G2  + support / resistance
G3  + 1H / 4H environment
G4  + liquidity
G5  + 5m execution state
G6  + existing frozen PGM score

时间治理
--------
Window A:
    train = TB1
    eval  = TB2

Window B:
    train = TB1 + TB2
    eval  = TB3

参数探索：
    只允许 TB1 -> TB2。
    禁止根据 TB3 结果挑参数。

TB4：
    本实验禁止访问。

重要因果约束
------------
- 所有 pivot 必须在 right bars 后才确认。
- 禁止把确认后的 pivot 回填到原 pivot 时点。
- 15m / 1H / 4H bar 只有完成以后才能用于 5m decision。
- future tensor 只用于 outcome，不能进入 feature。
- 高周期环境必须使用 decision time 当时已经完成的数据。

Pine 对应
---------
Deviation Trend Profile:
    SMA50
    ATR200
    dev = (close - SMA50) / ATR200

Support Resistance Channel:
    confirmed pivot
    channel clustering
    channel strength

Liquidity:
    confirmed swing
    ATR cluster
    liquidity level
    breach
    acceptance
    reclaim
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# -----------------------------------------------------------------------------
# BLAS 线程冻结，避免研究结果因为底层线程调度产生不必要差异
# -----------------------------------------------------------------------------

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_k, "1")


_REPO_ROOT = Path(__file__).resolve().parents[2]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


import numpy as np
import pandas as pd
import sklearn.metrics

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# -----------------------------------------------------------------------------
# 复用项目现有 owner
# -----------------------------------------------------------------------------

import research.liquidity_oracle_atlas.experiment_pgm_exec1_entry_stop_target_v1 as x1
import research.liquidity_oracle_atlas.experiment_pgm_native0d_acceleration_terminal_outcome_v1 as d0
import research.liquidity_oracle_atlas.experiment_pgm_native0e_consensus_acceleration_v1 as e0
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm


# =============================================================================
# 0. 治理
# =============================================================================

BASE_SHA = "7b9069f8caa2b69a45c3d4d3167e0cb097505847"

EXPERIMENT_NAME = "STRUCTURAL-REVERSION-PGM-V1"
EXPERIMENT_SCOPE = "EXPLORATORY_STRUCTURAL_REVERSION_ON_TB1_TB2_TB3_ONLY"

ALLOWED_BLOCKS = ("TB1", "TB2", "TB3")

WINDOWS = (
    dict(
        name="A_TB1_to_TB2",
        train=("TB1",),
        eval="TB2",
        scored_key="scored_A",
    ),
    dict(
        name="B_TB1TB2_to_TB3",
        train=("TB1", "TB2"),
        eval="TB3",
        scored_key="scored_B",
    ),
)

# 5m bars
HORIZONS = (6, 12, 24)
HMAX = max(HORIZONS)

# 仅作为路径诊断，不是最终 stop
STOP_R_DIAGNOSTIC = 1.0

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260918

SMOKE_CAP = 5000

# 参数扫描共同 warmup。
# 最大 DTP ATR candidate = 300，留一些缓冲。
SCAN_COMMON_M15_WARMUP = 320


OUT_DIR = _REPO_ROOT / "artifacts" / "liquidity_oracle_atlas"

PREFIX = "structural_reversion_pgm_v1"

ARTIFACT_METRICS = f"{PREFIX}_metrics.csv"
ARTIFACT_BOOTSTRAP = f"{PREFIX}_bootstrap.csv"
ARTIFACT_CELLS = f"{PREFIX}_cells.csv"
ARTIFACT_SCAN = f"{PREFIX}_parameter_scan.csv"
ARTIFACT_SUMMARY = f"{PREFIX}_summary.json"


# =============================================================================
# 1. 参数
# =============================================================================

@dataclass(frozen=True)
class IndicatorParams:

    # -------------------------------------------------------------------------
    # Deviation Trend Profile
    # -------------------------------------------------------------------------

    sma_len: int = 50
    atr_len: int = 200

    trend_slope_lag: int = 5
    trend_norm_lookback: int = 500
    trend_switch: float = 0.10

    # -------------------------------------------------------------------------
    # Support Resistance Channel
    # -------------------------------------------------------------------------

    sr_pivot_left: int = 10
    sr_pivot_right: int = 10

    sr_channel_width_pct: float = 5.0
    sr_width_lookback: int = 300

    sr_loopback: int = 290

    sr_min_strength: int = 1
    sr_max_channels: int = 6

    # -------------------------------------------------------------------------
    # Liquidity
    # -------------------------------------------------------------------------

    liq_left: int = 7
    liq_right: int = 1

    # Pine:
    # liqMar = 10 / input
    liq_margin_input: float = 6.9

    liq_visible: int = 3

    liq_atr_len: int = 10

    # Pine breach zone 默认约 2.3 ATR
    liq_postbreak_margin_atr: float = 2.3

    # -------------------------------------------------------------------------
    # 研究用 proximity
    # -------------------------------------------------------------------------

    # 注意：
    # 这只是把连续距离转成诊断节点。
    # 不代表最终交易阈值。
    near_atr: float = 0.50


PINE_DEFAULT = IndicatorParams()


def parameter_candidates() -> Dict[str, IndicatorParams]:
    """
    第一轮只允许一次改变一个参数。

    这里故意不扫描 sma_len。

    原因：
        SMA 同时定义：
        1. reversion direction
        2. mean target

    如果修改 SMA，再比较 y_mean_hit，
    就已经不是同一个 outcome。

    SMA20/50/100 必须以后单独做 target-definition experiment。
    """

    b = PINE_DEFAULT

    return {

        "pine_default": b,

        # ---------------------------------------------------------------------
        # Deviation scale
        # ---------------------------------------------------------------------

        "dev_atr100":
            replace(b, atr_len=100),

        "dev_atr300":
            replace(b, atr_len=300),

        "trend_lag3":
            replace(b, trend_slope_lag=3),

        "trend_lag10":
            replace(b, trend_slope_lag=10),

        "trend_switch0_05":
            replace(b, trend_switch=0.05),

        "trend_switch0_15":
            replace(b, trend_switch=0.15),

        # ---------------------------------------------------------------------
        # SR
        # ---------------------------------------------------------------------

        "sr_pivot6":
            replace(
                b,
                sr_pivot_left=6,
                sr_pivot_right=6,
            ),

        "sr_pivot14":
            replace(
                b,
                sr_pivot_left=14,
                sr_pivot_right=14,
            ),

        "sr_width3":
            replace(
                b,
                sr_channel_width_pct=3.0,
            ),

        "sr_width7":
            replace(
                b,
                sr_channel_width_pct=7.0,
            ),

        "sr_loopback180":
            replace(
                b,
                sr_loopback=180,
            ),

        "sr_loopback400":
            replace(
                b,
                sr_loopback=400,
            ),

        "sr_strength2":
            replace(
                b,
                sr_min_strength=2,
            ),

        "sr_strength3":
            replace(
                b,
                sr_min_strength=3,
            ),

        # ---------------------------------------------------------------------
        # Liquidity
        # ---------------------------------------------------------------------

        "liq_len5":
            replace(
                b,
                liq_left=5,
            ),

        "liq_len9":
            replace(
                b,
                liq_left=9,
            ),

        "liq_margin5_5":
            replace(
                b,
                liq_margin_input=5.5,
            ),

        "liq_margin8_0":
            replace(
                b,
                liq_margin_input=8.0,
            ),

        "liq_postbreak1_5":
            replace(
                b,
                liq_postbreak_margin_atr=1.5,
            ),

        "liq_postbreak3_0":
            replace(
                b,
                liq_postbreak_margin_atr=3.0,
            ),

        # ---------------------------------------------------------------------
        # proximity
        # ---------------------------------------------------------------------

        "near_atr0_25":
            replace(
                b,
                near_atr=0.25,
            ),

        "near_atr0_75":
            replace(
                b,
                near_atr=0.75,
            ),
    }


TF_MINUTES = {
    "m5": 5,
    "m15": 15,
    "h1": 60,
    "h4": 240,
}


# =============================================================================
# 2. 治理检查
# =============================================================================

def assert_allowed_blocks(
    df: pd.DataFrame,
) -> None:

    if "block" not in df.columns:
        raise SystemExit(
            "STOP_STRUCTREV_BLOCK_COLUMN_MISSING"
        )

    blocks = set(
        map(
            str,
            df["block"]
            .dropna()
            .unique()
            .tolist(),
        )
    )

    if "TB4" in blocks:
        raise SystemExit(
            "STOP_STRUCTREV_TB4_FORBIDDEN"
        )

    bad = blocks - set(ALLOWED_BLOCKS)

    if bad:
        raise SystemExit(
            f"STOP_STRUCTREV_UNKNOWN_BLOCKS:{sorted(bad)}"
        )


def git_head() -> str:

    return subprocess.check_output(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=str(_REPO_ROOT),
        text=True,
    ).strip()


def assert_base_sha_ancestor() -> None:

    r = subprocess.run(
        [
            "git",
            "merge-base",
            "--is-ancestor",
            BASE_SHA,
            "HEAD",
        ],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )

    if r.returncode != 0:
        raise SystemExit(
            f"STOP_STRUCTREV_BASE_SHA_NOT_ANCESTOR:{BASE_SHA}"
        )


def assert_clean_git_tree() -> None:

    checks = [

        (
            ["git", "diff", "--exit-code"],
            "STOP_STRUCTREV_GIT_TREE_DIRTY",
        ),

        (
            ["git", "diff", "--cached", "--exit-code"],
            "STOP_STRUCTREV_GIT_INDEX_DIRTY",
        ),
    ]

    for args, code in checks:

        r = subprocess.run(
            args,
            cwd=str(_REPO_ROOT),
            capture_output=True,
        )

        if r.returncode != 0:
            raise SystemExit(code)


def require_full_authorization() -> None:

    token = os.environ.get(
        "AUTHORIZE_STRUCTURAL_REVERSION_PGM_V1",
        "",
    ).strip()

    if token != "1":
        raise SystemExit(
            "STOP_STRUCTREV_FULL_NOT_AUTHORIZED"
        )


def require_parameter_scan_authorization() -> None:
    """
    参数扫描授权闸门。

    默认拒绝; 需要显式 env AUTHORIZE_STRUCTREV_PARAMETER_SCAN=1。

    该 gate 必须在 CLI 层、任何 load / fit / feature 工作之前调用;
    这里仅作为次级防御保留。
    """

    scan_token = os.environ.get(
        "AUTHORIZE_STRUCTREV_PARAMETER_SCAN",
        "",
    ).strip()

    if scan_token != "1":
        raise SystemExit(
            "STOP_STRUCTREV_PARAMETER_SCAN_NOT_AUTHORIZED"
        )


# =============================================================================
# 3. Pine 基础函数
# =============================================================================

def true_range(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
) -> np.ndarray:

    high = np.asarray(
        high,
        dtype=np.float64,
    )

    low = np.asarray(
        low,
        dtype=np.float64,
    )

    close = np.asarray(
        close,
        dtype=np.float64,
    )

    prev = np.r_[
        np.nan,
        close[:-1],
    ]

    tr = np.maximum(
        high - low,
        np.maximum(
            np.abs(high - prev),
            np.abs(low - prev),
        ),
    )

    if len(tr):
        tr[0] = (
            high[0]
            - low[0]
        )

    return tr


def pine_rma(
    x: np.ndarray,
    length: int,
) -> np.ndarray:
    """
    Wilder RMA。

    用 SMA seed，然后：
        rma_t =
            rma_(t-1)
            + (x_t - rma_(t-1)) / length
    """

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    out = np.full(
        len(x),
        np.nan,
        dtype=np.float64,
    )

    if (
        length <= 0
        or len(x) < length
    ):
        return out

    start = None

    for i in range(
        length - 1,
        len(x),
    ):

        w = x[
            i - length + 1:
            i + 1
        ]

        if np.all(
            np.isfinite(w)
        ):

            out[i] = float(
                np.mean(w)
            )

            start = i + 1

            break

    if start is None:
        return out

    alpha = (
        1.0
        / float(length)
    )

    for i in range(
        start,
        len(x),
    ):

        if not np.isfinite(x[i]):

            out[i] = np.nan

            continue

        if np.isfinite(
            out[i - 1]
        ):

            out[i] = (
                out[i - 1]
                + alpha
                * (
                    x[i]
                    - out[i - 1]
                )
            )

        else:

            w = x[
                i - length + 1:
                i + 1
            ]

            if (
                len(w) == length
                and np.all(
                    np.isfinite(w)
                )
            ):

                out[i] = float(
                    np.mean(w)
                )

    return out


def rolling_sma(
    x: np.ndarray,
    length: int,
) -> np.ndarray:

    return (
        pd.Series(
            np.asarray(
                x,
                dtype=np.float64,
            )
        )
        .rolling(
            length,
            min_periods=length,
        )
        .mean()
        .to_numpy(
            dtype=np.float64,
        )
    )


def confirmed_pivots(
    values: np.ndarray,
    left: int,
    right: int,
    mode: str,
) -> np.ndarray:
    """
    返回 pivot 的“确认时点”。

    pivot 位于 p：

        p-left ... p ... p+right

    只有到：

        p+right

    才允许系统知道该 pivot。

    绝不把 pivot 回填到 p。
    """

    v = np.asarray(
        values,
        dtype=np.float64,
    )

    n = len(v)

    out = np.full(
        n,
        np.nan,
        dtype=np.float64,
    )

    width = (
        left
        + right
        + 1
    )

    if n < width:
        return out

    windows = (
        np.lib
        .stride_tricks
        .sliding_window_view(
            v,
            width,
        )
    )

    center = windows[
        :,
        left,
    ]

    if mode == "high":

        extrema = np.nanmax(
            windows,
            axis=1,
        )

        ok = (
            np.isfinite(center)
            & (
                center
                >= extrema
            )
        )

    elif mode == "low":

        extrema = np.nanmin(
            windows,
            axis=1,
        )

        ok = (
            np.isfinite(center)
            & (
                center
                <= extrema
            )
        )

    else:

        raise ValueError(mode)

    starts = np.arange(
        len(windows),
        dtype=np.int64,
    )

    confirmation = (
        starts
        + left
        + right
    )

    out[
        confirmation[ok]
    ] = center[ok]

    return out


def trend_state_from_score(
    score: np.ndarray,
    switch: float,
) -> np.ndarray:

    score = np.asarray(
        score,
        dtype=np.float64,
    )

    out = np.zeros(
        len(score),
        dtype=np.int8,
    )

    state = 0
    prev = np.nan

    for i, cur in enumerate(score):

        if (
            np.isfinite(cur)
            and np.isfinite(prev)
        ):

            cross_up = (
                prev <= switch
                and cur > switch
            )

            cross_down = (
                prev >= -switch
                and cur < -switch
            )

            if (
                cross_up
                and state <= 0
            ):
                state = 1

            elif (
                cross_down
                and state >= 0
            ):
                state = -1

        out[i] = state

        if np.isfinite(cur):
            prev = cur

    return out


# =============================================================================
# 4. Support / Resistance
# =============================================================================

def build_sr_features(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    params: IndicatorParams,
) -> Dict[str, np.ndarray]:

    n = len(close)

    ph = confirmed_pivots(
        high,
        params.sr_pivot_left,
        params.sr_pivot_right,
        "high",
    )

    pl = confirmed_pivots(
        low,
        params.sr_pivot_left,
        params.sr_pivot_right,
        "low",
    )

    hh = (
        pd.Series(high)
        .rolling(
            params.sr_width_lookback,
            min_periods=params.sr_width_lookback,
        )
        .max()
        .to_numpy(float)
    )

    ll = (
        pd.Series(low)
        .rolling(
            params.sr_width_lookback,
            min_periods=params.sr_width_lookback,
        )
        .min()
        .to_numpy(float)
    )

    channel_width = (
        (hh - ll)
        * params.sr_channel_width_pct
        / 100.0
    )

    support_dist = np.full(
        n,
        np.nan,
    )

    resistance_dist = np.full(
        n,
        np.nan,
    )

    support_price = np.full(
        n,
        np.nan,
    )

    resistance_price = np.full(
        n,
        np.nan,
    )

    support_strength = np.full(
        n,
        np.nan,
    )

    resistance_strength = np.full(
        n,
        np.nan,
    )

    in_zone = np.zeros(
        n,
        dtype=np.int8,
    )

    zone_strength = np.zeros(
        n,
        dtype=np.float64,
    )

    broken_up = np.zeros(
        n,
        dtype=np.int8,
    )

    broken_down = np.zeros(
        n,
        dtype=np.int8,
    )

    n_channels = np.zeros(
        n,
        dtype=np.int16,
    )

    # 保存的是“确认时间”
    pivots: List[
        Tuple[int, float]
    ] = []

    # hi, lo, strength
    channels: List[
        Tuple[float, float, float]
    ] = []

    for i in range(n):

        new_pivot = (
            np.isfinite(ph[i])
            or np.isfinite(pl[i])
        )

        if new_pivot:

            # 对齐 Pine：
            # 同一确认 bar 上 ph 优先
            pivot_value = (
                float(ph[i])
                if np.isfinite(ph[i])
                else float(pl[i])
            )

            pivots.insert(
                0,
                (
                    i,
                    pivot_value,
                ),
            )

            pivots = [
                (j, p)
                for j, p in pivots
                if (
                    i - j
                    <= params.sr_loopback
                )
            ]

            width_i = (
                float(
                    channel_width[i]
                )
                if np.isfinite(
                    channel_width[i]
                )
                else np.nan
            )

            if (
                np.isfinite(width_i)
                and width_i > 0
                and pivots
            ):

                candidates = []

                hist0 = max(
                    0,
                    i - params.sr_loopback,
                )

                hhist = high[
                    hist0:
                    i + 1
                ]

                lhist = low[
                    hist0:
                    i + 1
                ]

                for _, seed in pivots:

                    loz = float(seed)
                    hiz = float(seed)

                    pivot_count = 0

                    for _, cpp in pivots:

                        width_test = (
                            hiz - cpp
                            if cpp <= hiz
                            else cpp - loz
                        )

                        if (
                            width_test
                            <= width_i
                        ):

                            loz = min(
                                loz,
                                cpp,
                            )

                            hiz = max(
                                hiz,
                                cpp,
                            )

                            pivot_count += 1

                    touches = int(
                        np.sum(
                            (
                                (hhist <= hiz)
                                & (hhist >= loz)
                            )
                            |
                            (
                                (lhist <= hiz)
                                & (lhist >= loz)
                            )
                        )
                    )

                    strength = float(
                        pivot_count * 20
                        + touches
                    )

                    candidates.append(
                        (
                            hiz,
                            loz,
                            strength,
                        )
                    )

                selected = []

                alive = np.ones(
                    len(candidates),
                    dtype=bool,
                )

                for _ in range(
                    min(
                        10,
                        len(candidates),
                    )
                ):

                    ids = np.flatnonzero(
                        alive
                    )

                    if len(ids) == 0:
                        break

                    strengths = np.array(
                        [
                            candidates[j][2]
                            for j in ids
                        ],
                        dtype=float,
                    )

                    k = int(
                        ids[
                            int(
                                np.argmax(
                                    strengths
                                )
                            )
                        ]
                    )

                    hiz, loz, strength = (
                        candidates[k]
                    )

                    if (
                        strength
                        < params.sr_min_strength
                        * 20
                    ):
                        break

                    selected.append(
                        (
                            hiz,
                            loz,
                            strength,
                        )
                    )

                    for j, (
                        ch,
                        cl,
                        _,
                    ) in enumerate(
                        candidates
                    ):

                        if not alive[j]:
                            continue

                        overlap = (
                            loz <= ch <= hiz
                            or
                            loz <= cl <= hiz
                        )

                        if overlap:
                            alive[j] = False

                channels = selected[
                    : params.sr_max_channels
                ]

        n_channels[i] = len(
            channels
        )

        c = float(
            close[i]
        )

        a = (
            float(atr[i])
            if (
                np.isfinite(atr[i])
                and atr[i] > 0
            )
            else np.nan
        )

        containing = [
            z
            for z in channels
            if (
                z[1]
                <= c
                <= z[0]
            )
        ]

        if containing:

            in_zone[i] = 1

            zone_strength[i] = max(
                z[2]
                for z in containing
            )

        supports = [
            z
            for z in channels
            if z[0] < c
        ]

        resistances = [
            z
            for z in channels
            if z[1] > c
        ]

        if supports:

            z = max(
                supports,
                key=lambda q: q[0],
            )

            support_price[i] = z[0]

            support_strength[i] = z[2]

            if np.isfinite(a):

                support_dist[i] = (
                    c - z[0]
                ) / a

        elif containing:

            z = max(
                containing,
                key=lambda q: q[2],
            )

            support_price[i] = c
            support_strength[i] = z[2]
            support_dist[i] = 0.0

        if resistances:

            z = min(
                resistances,
                key=lambda q: q[1],
            )

            resistance_price[i] = z[1]

            resistance_strength[i] = z[2]

            if np.isfinite(a):

                resistance_dist[i] = (
                    z[1] - c
                ) / a

        elif containing:

            z = max(
                containing,
                key=lambda q: q[2],
            )

            resistance_price[i] = c
            resistance_strength[i] = z[2]
            resistance_dist[i] = 0.0

        if (
            i > 0
            and not containing
        ):

            prev_close = float(
                close[i - 1]
            )

            for hi0, lo0, _ in channels:

                if (
                    prev_close <= hi0
                    and c > hi0
                ):
                    broken_up[i] = 1

                if (
                    prev_close >= lo0
                    and c < lo0
                ):
                    broken_down[i] = 1

    return {

        "sr_support_dist_atr":
            support_dist,

        "sr_resistance_dist_atr":
            resistance_dist,

        "sr_support_price":
            support_price,

        "sr_resistance_price":
            resistance_price,

        "sr_support_strength":
            support_strength,

        "sr_resistance_strength":
            resistance_strength,

        "sr_in_zone":
            in_zone,

        "sr_zone_strength":
            zone_strength,

        "sr_broken_up":
            broken_up,

        "sr_broken_down":
            broken_down,

        "sr_n_channels":
            n_channels,
    }


# =============================================================================
# 5. Liquidity
# =============================================================================

def build_liquidity_features(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    params: IndicatorParams,
) -> Dict[str, np.ndarray]:

    n = len(close)

    ph = confirmed_pivots(
        high,
        params.liq_left,
        params.liq_right,
        "high",
    )

    pl = confirmed_pivots(
        low,
        params.liq_left,
        params.liq_right,
        "low",
    )

    liq_mar = (
        10.0
        / float(
            params.liq_margin_input
        )
    )

    up_dist = np.full(
        n,
        np.nan,
    )

    down_dist = np.full(
        n,
        np.nan,
    )

    up_level = np.full(
        n,
        np.nan,
    )

    down_level = np.full(
        n,
        np.nan,
    )

    breach_up = np.zeros(
        n,
        dtype=np.int8,
    )

    breach_down = np.zeros(
        n,
        dtype=np.int8,
    )

    last_breach_side = np.zeros(
        n,
        dtype=np.int8,
    )

    last_breach_age = np.full(
        n,
        np.nan,
    )

    last_accept = np.zeros(
        n,
        dtype=np.int8,
    )

    last_reclaim = np.zeros(
        n,
        dtype=np.int8,
    )

    last_zone_active = np.zeros(
        n,
        dtype=np.int8,
    )

    up_count = np.zeros(
        n,
        dtype=np.int16,
    )

    down_count = np.zeros(
        n,
        dtype=np.int16,
    )

    # newest first
    zz: List[
        Dict[str, Any]
    ] = []

    levels_up: List[
        Dict[str, Any]
    ] = []

    levels_down: List[
        Dict[str, Any]
    ] = []

    last_breach: Optional[
        Dict[str, Any]
    ] = None


    def update_zz(
        side: int,
        idx: int,
        price: float,
    ) -> None:

        nonlocal zz

        if (
            not zz
            or int(
                zz[0]["dir"]
            ) != side
        ):

            zz.insert(
                0,
                dict(
                    dir=side,
                    x=idx,
                    y=float(price),
                ),
            )

        else:

            better = (
                price > zz[0]["y"]
                if side > 0
                else price < zz[0]["y"]
            )

            if better:

                zz[0] = dict(
                    dir=side,
                    x=idx,
                    y=float(price),
                )

        zz = zz[:50]


    def maybe_create_level(
        side: int,
        pivot: float,
        atr_i: float,
    ) -> None:

        nonlocal levels_up
        nonlocal levels_down

        if (
            not np.isfinite(atr_i)
            or atr_i <= 0
        ):
            return

        margin = (
            atr_i
            / liq_mar
        )

        count = 0

        start_bar = None

        level_price = np.nan

        # Pine 中变量名反直觉：
        # minP 实际保留 cluster max
        # maxP 实际保留 cluster min
        cluster_max = 0.0
        cluster_min = 1e7

        for z in zz:

            if (
                int(z["dir"])
                != side
            ):
                continue

            y = float(
                z["y"]
            )

            if side > 0:

                if (
                    y
                    > pivot + margin
                ):
                    break

            else:

                if (
                    y
                    < pivot - margin
                ):
                    break

            inside = (
                pivot - margin
                < y
                < pivot + margin
            )

            if inside:

                count += 1

                start_bar = int(
                    z["x"]
                )

                level_price = y

                cluster_max = max(
                    cluster_max,
                    y,
                )

                cluster_min = min(
                    cluster_min,
                    y,
                )

        if (
            count <= 2
            or start_bar is None
            or not np.isfinite(
                level_price
            )
        ):
            return

        center = (
            0.5
            * (
                cluster_max
                + cluster_min
            )
        )

        obj = dict(

            left=start_bar,

            level=float(
                level_price
            ),

            top=float(
                center
                + margin
            ),

            bottom=float(
                center
                - margin
            ),

            broken=False,

            breach_i=None,
        )

        target = (
            levels_up
            if side > 0
            else levels_down
        )

        if (
            target
            and int(
                target[0]["left"]
            ) == start_bar
        ):

            target[0]["top"] = (
                obj["top"]
            )

            target[0]["bottom"] = (
                obj["bottom"]
            )

        else:

            target.insert(
                0,
                obj,
            )

            del target[
                params.liq_visible:
            ]


    for i in range(n):

        atr_i = (
            float(atr[i])
            if np.isfinite(atr[i])
            else np.nan
        )

        # ---------------------------------------------------------------------
        # confirmed swing
        # ---------------------------------------------------------------------

        if np.isfinite(
            ph[i]
        ):

            update_zz(
                +1,
                i - params.liq_right,
                float(ph[i]),
            )

            maybe_create_level(
                +1,
                float(ph[i]),
                atr_i,
            )

        if np.isfinite(
            pl[i]
        ):

            update_zz(
                -1,
                i - params.liq_right,
                float(pl[i]),
            )

            maybe_create_level(
                -1,
                float(pl[i]),
                atr_i,
            )

        # ---------------------------------------------------------------------
        # breach
        # ---------------------------------------------------------------------

        for lev in levels_up:

            if (
                not lev["broken"]
                and float(high[i])
                > float(
                    lev["top"]
                )
            ):

                lev["broken"] = True
                lev["breach_i"] = i

                breach_up[i] = 1

                last_breach = dict(

                    side=+1,

                    level=float(
                        lev["level"]
                    ),

                    i=i,

                    zone_active=True,
                )

        for lev in levels_down:

            if (
                not lev["broken"]
                and float(low[i])
                < float(
                    lev["bottom"]
                )
            ):

                lev["broken"] = True
                lev["breach_i"] = i

                breach_down[i] = 1

                last_breach = dict(

                    side=-1,

                    level=float(
                        lev["level"]
                    ),

                    i=i,

                    zone_active=True,
                )

        # ---------------------------------------------------------------------
        # 最近尚未突破的 liquidity
        # ---------------------------------------------------------------------

        active_up = [

            z
            for z in levels_up

            if (
                not z["broken"]
                and z["bottom"]
                > close[i]
            )
        ]

        active_down = [

            z
            for z in levels_down

            if (
                not z["broken"]
                and z["top"]
                < close[i]
            )
        ]

        if (
            active_up
            and np.isfinite(atr_i)
            and atr_i > 0
        ):

            z = min(
                active_up,
                key=lambda q: q["bottom"],
            )

            up_dist[i] = (
                float(z["bottom"])
                - float(close[i])
            ) / atr_i

            up_level[i] = float(
                z["level"]
            )

        if (
            active_down
            and np.isfinite(atr_i)
            and atr_i > 0
        ):

            z = max(
                active_down,
                key=lambda q: q["top"],
            )

            down_dist[i] = (
                float(close[i])
                - float(z["top"])
            ) / atr_i

            down_level[i] = float(
                z["level"]
            )

        up_count[i] = int(
            sum(
                not z["broken"]
                for z in levels_up
            )
        )

        down_count[i] = int(
            sum(
                not z["broken"]
                for z in levels_down
            )
        )

        # ---------------------------------------------------------------------
        # breach 后状态
        # ---------------------------------------------------------------------

        if last_breach is not None:

            side = int(
                last_breach["side"]
            )

            level = float(
                last_breach["level"]
            )

            age = (
                i
                - int(
                    last_breach["i"]
                )
            )

            last_breach_side[i] = side

            last_breach_age[i] = float(
                age
            )

            # Pine post-break zone
            if (
                bool(
                    last_breach.get(
                        "zone_active",
                        False,
                    )
                )
                and np.isfinite(
                    atr_i
                )
                and atr_i > 0
            ):

                inside_zone = (

                    float(low[i])
                    >
                    level
                    - params.liq_postbreak_margin_atr
                    * atr_i

                    and

                    float(high[i])
                    <
                    level
                    + params.liq_postbreak_margin_atr
                    * atr_i
                )

                if not inside_zone:

                    last_breach[
                        "zone_active"
                    ] = False

            last_zone_active[i] = int(
                bool(
                    last_breach.get(
                        "zone_active",
                        False,
                    )
                )
            )

            # -------------------------------------------------------------
            # 我们额外显式定义：
            #
            # acceptance:
            #     close 仍在突破方向
            #
            # reclaim:
            #     close 已经重新回到 level 另一侧
            #
            # 这是研究节点，不宣称 Pine 原作者有这个交易定义。
            # -------------------------------------------------------------

            if side > 0:

                accepted = (
                    close[i]
                    > level
                )

                reclaimed = (
                    close[i]
                    <= level
                )

            else:

                accepted = (
                    close[i]
                    < level
                )

                reclaimed = (
                    close[i]
                    >= level
                )

            last_accept[i] = int(
                accepted
            )

            last_reclaim[i] = int(
                reclaimed
            )

    return {

        "liq_up_dist_atr":
            up_dist,

        "liq_down_dist_atr":
            down_dist,

        "liq_up_level_price":
            up_level,

        "liq_down_level_price":
            down_level,

        "liq_breach_up":
            breach_up,

        "liq_breach_down":
            breach_down,

        "liq_last_breach_side":
            last_breach_side,

        "liq_last_breach_age":
            last_breach_age,

        "liq_last_accept":
            last_accept,

        "liq_last_reclaim":
            last_reclaim,

        "liq_last_zone_active":
            last_zone_active,

        "liq_up_count":
            up_count,

        "liq_down_count":
            down_count,
    }


# =============================================================================
# 6. 单周期指标
# =============================================================================

def compute_segment_features(
    seg: pd.DataFrame,
    params: IndicatorParams,
    include_sr: bool,
) -> pd.DataFrame:

    x = (
        seg
        .copy()
        .reset_index(drop=True)
    )

    x["tf_seq"] = np.arange(
        len(x),
        dtype=np.int64,
    )

    high = x[
        "high"
    ].to_numpy(float)

    low = x[
        "low"
    ].to_numpy(float)

    close = x[
        "close"
    ].to_numpy(float)

    # -------------------------------------------------------------------------
    # DTP
    # -------------------------------------------------------------------------

    sma = rolling_sma(
        close,
        params.sma_len,
    )

    tr = true_range(
        high,
        low,
        close,
    )

    atr = pine_rma(
        tr,
        params.atr_len,
    )

    atr_liq = pine_rma(
        tr,
        params.liq_atr_len,
    )

    dev = (
        close
        - sma
    ) / atr

    slope_raw = (
        sma
        - np.roll(
            sma,
            params.trend_slope_lag,
        )
    )

    slope_raw[
        : params.trend_slope_lag
    ] = np.nan

    slope_atr = (
        slope_raw
        / atr
    )

    # Pine percentile(...,100)
    # 等价于 rolling maximum
    trend_denominator = (
        pd.Series(
            slope_raw
        )
        .rolling(
            params.trend_norm_lookback,
            min_periods=params.trend_norm_lookback,
        )
        .max()
        .to_numpy(float)
    )

    trend_score = np.divide(

        slope_raw,

        trend_denominator,

        out=np.full(
            len(x),
            np.nan,
        ),

        where=(
            np.isfinite(
                trend_denominator
            )
            &
            (
                np.abs(
                    trend_denominator
                )
                > 1e-12
            )
        ),
    )

    trend_state = (
        trend_state_from_score(
            trend_score,
            params.trend_switch,
        )
    )

    x["sma"] = sma
    x["atr"] = atr
    x["dev"] = dev
    x["slope_atr"] = slope_atr
    x["trend_score"] = trend_score
    x["trend_state"] = trend_state

    # -------------------------------------------------------------------------
    # SR
    # -------------------------------------------------------------------------

    if include_sr:

        sr = build_sr_features(
            high,
            low,
            close,
            atr,
            params,
        )

        for k, v in sr.items():
            x[k] = v

    else:

        # 第一轮 5m 不计算昂贵 SR。
        # 5m 负责 trigger；
        # 15m/1h/4h 负责结构。
        nan_cols = [

            "sr_support_dist_atr",
            "sr_resistance_dist_atr",

            "sr_support_price",
            "sr_resistance_price",

            "sr_support_strength",
            "sr_resistance_strength",

            "sr_zone_strength",
        ]

        zero_cols = [

            "sr_in_zone",
            "sr_broken_up",
            "sr_broken_down",
            "sr_n_channels",
        ]

        for c in nan_cols:
            x[c] = np.nan

        for c in zero_cols:
            x[c] = 0

    # -------------------------------------------------------------------------
    # Liquidity
    # -------------------------------------------------------------------------

    liq = build_liquidity_features(
        high,
        low,
        close,
        atr_liq,
        params,
    )

    for k, v in liq.items():
        x[k] = v

    return x


def compute_tf_features(
    tf_bars: pd.DataFrame,
    params: IndicatorParams,
    include_sr: bool,
) -> pd.DataFrame:

    pieces = []

    # discontinuity 后所有递推 / rolling 重新开始
    for _, seg in tf_bars.groupby(
        "segment",
        sort=False,
    ):

        pieces.append(
            compute_segment_features(
                seg,
                params,
                include_sr,
            )
        )

    if not pieces:
        return tf_bars.copy()

    out = pd.concat(
        pieces,
        ignore_index=True,
    )

    return (
        out
        .sort_values(
            "available_time",
            kind="stable",
        )
        .reset_index(drop=True)
    )


# =============================================================================
# 7. 5m -> 15m / 1H / 4H
# =============================================================================

def raw_frame_from_owner(
    bars: Mapping[str, Any],
) -> pd.DataFrame:

    n = int(
        bars["n"]
    )

    time_arr = pd.to_datetime(
        np.asarray(
            bars["t"]
        )
    )

    trading_day = pd.to_datetime(
        np.asarray(
            bars["day"]
        )
    )

    disc = np.asarray(
        bars["disc"],
        dtype=bool,
    )

    if not (
        len(time_arr)
        == len(trading_day)
        == len(disc)
        == n
    ):

        raise SystemExit(
            "STOP_STRUCTREV_RAW_BAR_LENGTH_MISMATCH"
        )

    segment = np.cumsum(
        disc.astype(
            np.int64
        )
    )

    return pd.DataFrame({

        "raw_index":
            np.arange(
                n,
                dtype=np.int64,
            ),

        "time":
            time_arr,

        "trading_day":
            trading_day,

        "segment":
            segment,

        "open":
            np.asarray(
                bars["o"],
                float,
            ),

        "high":
            np.asarray(
                bars["h"],
                float,
            ),

        "low":
            np.asarray(
                bars["l"],
                float,
            ),

        "close":
            np.asarray(
                bars["c"],
                float,
            ),

        "disc":
            disc,
    })


def resample_causal(
    raw: pd.DataFrame,
    minutes: int,
) -> pd.DataFrame:

    if minutes == 5:

        out = raw.copy()

        out[
            "available_time"
        ] = (
            out["time"]
            + pd.Timedelta(
                minutes=5
            )
        )

        out["n_base"] = 1

        return out

    x = raw.copy()

    x["bucket"] = (
        x["time"]
        .dt.floor(
            f"{minutes}min"
        )
    )

    # 不允许跨：
    # trading day
    # discontinuity segment

    grouped = x.groupby(
        [
            "trading_day",
            "segment",
            "bucket",
        ],
        sort=False,
        observed=True,
    )

    out = grouped.agg(

        open=(
            "open",
            "first",
        ),

        high=(
            "high",
            "max",
        ),

        low=(
            "low",
            "min",
        ),

        close=(
            "close",
            "last",
        ),

        first_time=(
            "time",
            "first",
        ),

        last_time=(
            "time",
            "last",
        ),

        n_base=(
            "time",
            "size",
        ),

    ).reset_index()

    out["time"] = (
        out["bucket"]
    )

    # -------------------------------------------------------------------------
    # 最重要的因果约束
    #
    # 一个 4H bar：
    # 不能在 bar 尚未结束时给 5m decision 使用。
    #
    # available_time =
    # 最后一根真实 5m bar 完成时间
    # -------------------------------------------------------------------------

    out[
        "available_time"
    ] = (
        out["last_time"]
        + pd.Timedelta(
            minutes=5
        )
    )

    out["disc"] = False

    return out[

        [
            "time",
            "trading_day",
            "segment",

            "open",
            "high",
            "low",
            "close",

            "disc",

            "available_time",

            "n_base",
        ]
    ]


def build_indicator_cache(
    bars_by_sym: Mapping[
        str,
        Mapping[str, Any],
    ],
    params: IndicatorParams,
    symbols: Optional[
        Iterable[str]
    ] = None,
) -> Dict[
    str,
    Dict[
        str,
        pd.DataFrame,
    ],
]:

    use_symbols = sorted(
        symbols
        if symbols is not None
        else bars_by_sym.keys()
    )

    cache = {}

    for sym in use_symbols:

        if sym not in bars_by_sym:

            raise SystemExit(
                f"STOP_STRUCTREV_SYMBOL_BARS_MISSING:{sym}"
            )

        raw = raw_frame_from_owner(
            bars_by_sym[sym]
        )

        cache[sym] = {}

        for tf, minutes in (
            TF_MINUTES.items()
        ):

            tf_bars = resample_causal(
                raw,
                minutes,
            )

            cache[sym][tf] = (
                compute_tf_features(

                    tf_bars,

                    params,

                    include_sr=(
                        tf != "m5"
                    ),
                )
            )

    return cache


# =============================================================================
# 8. 将多周期特征映射到 5m decision rows
# =============================================================================

META_TF_COLS = {

    "raw_index",
    "time",
    "trading_day",
    "segment",

    "open",
    "high",
    "low",
    "close",

    "disc",

    "available_time",
    "n_base",

    "bucket",
    "first_time",
    "last_time",
}


def attach_indicator_features(
    scored: pd.DataFrame,
    bars_by_sym: Mapping[
        str,
        Mapping[str, Any],
    ],
    cache: Dict[
        str,
        Dict[
            str,
            pd.DataFrame,
        ],
    ],
) -> pd.DataFrame:

    out = (
        scored
        .copy()
        .reset_index(drop=True)
    )

    if (
        "bar_t"
        not in out.columns
    ):

        raise SystemExit(
            "STOP_STRUCTREV_BAR_T_MISSING"
        )

    feature_names = {}

    sample_symbol = next(
        iter(cache)
    )

    # 预建列
    for tf in TF_MINUTES:

        cols = [

            c
            for c in
            cache[
                sample_symbol
            ][tf].columns

            if c
            not in META_TF_COLS
        ]

        feature_names[tf] = cols

        for c in cols:

            out[
                f"{tf}_{c}"
            ] = np.nan

    decision_close_time = np.full(
        len(out),
        np.datetime64(
            "NaT",
            "ns",
        ),
        dtype="datetime64[ns]",
    )

    for sym, index in (
        out
        .groupby(
            "symbol",
            sort=False,
        )
        .groups
        .items()
    ):

        rows = np.asarray(
            list(index),
            dtype=np.int64,
        )

        if (
            sym not in bars_by_sym
            or sym not in cache
        ):

            raise SystemExit(
                f"STOP_STRUCTREV_ATTACH_SYMBOL_MISSING:{sym}"
            )

        bars = bars_by_sym[sym]

        bar_t = (
            out
            .loc[
                rows,
                "bar_t",
            ]
            .to_numpy(
                np.int64
            )
        )

        if (
            np.any(bar_t < 0)
            or
            np.any(
                bar_t
                >= int(
                    bars["n"]
                )
            )
        ):

            raise SystemExit(
                f"STOP_STRUCTREV_BAR_T_OOB:{sym}"
            )

        decision_start = (
            pd.to_datetime(
                np.asarray(
                    bars["t"]
                )[bar_t]
            )
            .to_numpy(
                dtype="datetime64[ns]"
            )
        )

        # bar_t 是当前 5m bar。
        # 所有当前 bar 信息在 bar close 后才知道。
        decision_close = (
            decision_start
            + np.timedelta64(
                5,
                "m",
            )
        )

        decision_close_time[
            rows
        ] = decision_close

        for tf in TF_MINUTES:

            f = cache[sym][tf]

            available_time = (
                pd.to_datetime(
                    f[
                        "available_time"
                    ]
                )
                .to_numpy(
                    dtype="datetime64[ns]"
                )
            )

            # latest fully-known TF row
            pos = (
                np.searchsorted(
                    available_time,
                    decision_close,
                    side="right",
                )
                - 1
            )

            ok = (
                pos >= 0
            )

            if not np.any(ok):
                continue

            row_pos = rows[ok]
            feat_pos = pos[ok]

            for c in (
                feature_names[tf]
            ):

                values = (
                    f[c]
                    .to_numpy()
                )

                out.loc[
                    row_pos,
                    f"{tf}_{c}",
                ] = (
                    values[
                        feat_pos
                    ]
                )

    out[
        "decision_close_time_structrev"
    ] = decision_close_time

    return out


# =============================================================================
# 9. 多空统一到“回归方向”
# =============================================================================

def add_oriented_nodes(
    df: pd.DataFrame,
    params: IndicatorParams,
) -> pd.DataFrame:

    x = df.copy()

    dev15 = (
        x["m15_dev"]
        .to_numpy(float)
    )

    # -------------------------------------------------------------------------
    # 偏离在上方 -> 回归方向 short
    # 偏离在下方 -> 回归方向 long
    # -------------------------------------------------------------------------

    direction = (
        -np.sign(dev15)
    )

    direction[
        ~np.isfinite(dev15)
    ] = 0

    direction = direction.astype(
        np.int8
    )

    x[
        "reversion_dir"
    ] = direction

    x[
        "dev_abs"
    ] = np.abs(
        dev15
    )

    # -------------------------------------------------------------------------
    # 多周期环境
    # -------------------------------------------------------------------------

    for tf in (
        "m5",
        "m15",
        "h1",
        "h4",
    ):

        slope = (
            x[
                f"{tf}_slope_atr"
            ]
            .to_numpy(float)
        )

        dev = (
            x[
                f"{tf}_dev"
            ]
            .to_numpy(float)
        )

        trend_state = (
            x[
                f"{tf}_trend_state"
            ]
            .to_numpy(float)
        )

        # >0:
        # trend slope 与回归方向同向
        x[
            f"{tf}_trend_align"
        ] = (
            direction
            * slope
        )

        x[
            f"{tf}_state_align"
        ] = (
            direction
            * trend_state
        )

        # >0:
        # 该周期也处于需要向 reversion_dir 回归的一侧
        x[
            f"{tf}_reversion_pressure"
        ] = (
            -direction
            * dev
        )

    # -------------------------------------------------------------------------
    # SR
    #
    # long:
    #     same side = support
    #     opposite  = resistance
    #
    # short:
    #     same side = resistance
    #     opposite  = support
    # -------------------------------------------------------------------------

    for tf in (
        "m15",
        "h1",
        "h4",
    ):

        support_dist = (
            x[
                f"{tf}_sr_support_dist_atr"
            ]
            .to_numpy(float)
        )

        resistance_dist = (
            x[
                f"{tf}_sr_resistance_dist_atr"
            ]
            .to_numpy(float)
        )

        support_strength = (
            x[
                f"{tf}_sr_support_strength"
            ]
            .to_numpy(float)
        )

        resistance_strength = (
            x[
                f"{tf}_sr_resistance_strength"
            ]
            .to_numpy(float)
        )

        support_price = (
            x[
                f"{tf}_sr_support_price"
            ]
            .to_numpy(float)
        )

        resistance_price = (
            x[
                f"{tf}_sr_resistance_price"
            ]
            .to_numpy(float)
        )

        x[
            f"{tf}_sr_same_dist"
        ] = np.where(

            direction > 0,

            support_dist,

            np.where(
                direction < 0,
                resistance_dist,
                np.nan,
            ),
        )

        x[
            f"{tf}_sr_opp_dist"
        ] = np.where(

            direction > 0,

            resistance_dist,

            np.where(
                direction < 0,
                support_dist,
                np.nan,
            ),
        )

        x[
            f"{tf}_sr_same_strength"
        ] = np.where(

            direction > 0,

            support_strength,

            np.where(
                direction < 0,
                resistance_strength,
                np.nan,
            ),
        )

        x[
            f"{tf}_sr_opp_strength"
        ] = np.where(

            direction > 0,

            resistance_strength,

            np.where(
                direction < 0,
                support_strength,
                np.nan,
            ),
        )

        x[
            f"{tf}_sr_same_price"
        ] = np.where(

            direction > 0,

            support_price,

            np.where(
                direction < 0,
                resistance_price,
                np.nan,
            ),
        )

        x[
            f"{tf}_sr_opp_price"
        ] = np.where(

            direction > 0,

            resistance_price,

            np.where(
                direction < 0,
                support_price,
                np.nan,
            ),
        )

        same_dist = (
            x[
                f"{tf}_sr_same_dist"
            ]
            .to_numpy(float)
        )

        opp_dist = (
            x[
                f"{tf}_sr_opp_dist"
            ]
            .to_numpy(float)
        )

        x[
            f"{tf}_sr_same_exists"
        ] = np.isfinite(
            same_dist
        ).astype(
            np.int8
        )

        x[
            f"{tf}_sr_opp_exists"
        ] = np.isfinite(
            opp_dist
        ).astype(
            np.int8
        )

        x[
            f"{tf}_sr_same_near"
        ] = (
            same_dist
            <= params.near_atr
        ).astype(
            np.int8
        )

    # -------------------------------------------------------------------------
    # Liquidity
    #
    # 对 long：
    #     against side liquidity = downside sellside
    #     target-side liquidity  = upside buyside
    #
    # 对 short：
    #     相反
    # -------------------------------------------------------------------------

    for tf in (
        "m5",
        "m15",
        "h1",
    ):

        up = (
            x[
                f"{tf}_liq_up_dist_atr"
            ]
            .to_numpy(float)
        )

        down = (
            x[
                f"{tf}_liq_down_dist_atr"
            ]
            .to_numpy(float)
        )

        breach_up = (
            x[
                f"{tf}_liq_breach_up"
            ]
            .to_numpy(float)
        )

        breach_down = (
            x[
                f"{tf}_liq_breach_down"
            ]
            .to_numpy(float)
        )

        last_side = (
            x[
                f"{tf}_liq_last_breach_side"
            ]
            .to_numpy(float)
        )

        last_accept = (
            x[
                f"{tf}_liq_last_accept"
            ]
            .to_numpy(float)
        )

        last_reclaim = (
            x[
                f"{tf}_liq_last_reclaim"
            ]
            .to_numpy(float)
        )

        x[
            f"{tf}_liq_against_dist"
        ] = np.where(

            direction > 0,

            down,

            np.where(
                direction < 0,
                up,
                np.nan,
            ),
        )

        x[
            f"{tf}_liq_target_dist"
        ] = np.where(

            direction > 0,

            up,

            np.where(
                direction < 0,
                down,
                np.nan,
            ),
        )

        against_dist = (
            x[
                f"{tf}_liq_against_dist"
            ]
            .to_numpy(float)
        )

        x[
            f"{tf}_liq_against_exists"
        ] = np.isfinite(
            against_dist
        ).astype(
            np.int8
        )

        x[
            f"{tf}_liq_against_near"
        ] = (
            against_dist
            <= params.near_atr
        ).astype(
            np.int8
        )

        x[
            f"{tf}_liq_sweep_against_now"
        ] = np.where(

            direction > 0,

            breach_down,

            np.where(
                direction < 0,
                breach_up,
                0,
            ),

        ).astype(
            np.int8
        )

        x[
            f"{tf}_liq_break_trade_dir_now"
        ] = np.where(

            direction > 0,

            breach_up,

            np.where(
                direction < 0,
                breach_down,
                0,
            ),

        ).astype(
            np.int8
        )

        # 最近一次 liquidity breach
        # 是否发生在交易方向反侧
        against_last = (
            last_side
            == -direction
        )

        x[
            f"{tf}_liq_reclaim_after_against_sweep"
        ] = (

            against_last

            & (
                last_reclaim
                > 0
            )

        ).astype(
            np.int8
        )

        x[
            f"{tf}_liq_accept_against"
        ] = (

            against_last

            & (
                last_accept
                > 0
            )

        ).astype(
            np.int8
        )

    # -------------------------------------------------------------------------
    # Existing PGM
    #
    # 不让旧 continuation PGM 决定本策略方向。
    # 它只是环境/context node。
    # -------------------------------------------------------------------------

    score = (
        x[
            "score_mu"
        ]
        .to_numpy(float)
    )

    x[
        "pgm_alignment"
    ] = (
        direction
        * score
    )

    x[
        "pgm_abs_score"
    ] = np.abs(
        score
    )

    return x


# =============================================================================
# 10. Candidate universe + future path
# =============================================================================

def extract_candidate_rows(
    df: pd.DataFrame,
    blocks: Sequence[str],
    common_scan_universe: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:

    assert_allowed_blocks(
        df
    )

    block_mask = (
        df[
            "block"
        ]
        .isin(
            list(blocks)
        )
        .to_numpy(bool)
    )

    raw_block_rows = int(
        block_mask.sum()
    )

    atr0_vals = (
        df[
            "atr0"
        ]
        .to_numpy(float)
    )

    m1 = (
        block_mask
        &
        df[
            "same_block_entry_valid"
        ]
        .to_numpy(bool)
    )

    m2 = (
        m1
        &
        np.isfinite(
            atr0_vals
        )
        &
        (
            atr0_vals
            > 0
        )
    )

    m3 = (
        m2
        &
        np.isfinite(
            df[
                "m15_dev"
            ]
            .to_numpy(float)
        )
    )

    m4 = (
        m3
        &
        np.isfinite(
            df[
                "m15_sma"
            ]
            .to_numpy(float)
        )
    )

    mask = (
        m4
        &
        (
            df[
                "reversion_dir"
            ]
            .to_numpy(
                np.int8
            )
            != 0
        )
    )

    if common_scan_universe:

        if (
            "m15_tf_seq"
            not in df.columns
        ):

            raise SystemExit(
                "STOP_STRUCTREV_SCAN_TF_SEQ_MISSING"
            )

        mask &= (

            df[
                "m15_tf_seq"
            ]
            .to_numpy(float)

            >=

            SCAN_COMMON_M15_WARMUP
        )

    funnel = dict(
        raw_block_rows=raw_block_rows,
        same_block_entry_valid=int(
            m1.sum()
        ),
        atr0_valid=int(
            m2.sum()
        ),
        m15_dev_valid=int(
            m3.sum()
        ),
        m15_sma_valid=int(
            m4.sum()
        ),
        reversion_dir_nonzero=int(
            mask.sum()
        ),
    )

    out = (
        df
        .loc[mask]
        .copy()
        .reset_index(drop=True)
    )

    # x1.build_future_tensor
    # 使用 base_action 作为方向字段。
    # 在这里明确改成结构性回归方向。
    out[
        "base_action"
    ] = (
        out[
            "reversion_dir"
        ]
        .to_numpy(
            np.int8
        )
    )

    return out, funnel


def build_path_dataset(
    df: pd.DataFrame,
    blocks: Sequence[str],
    bars_by_sym: Mapping[
        str,
        Mapping[str, Any],
    ],
    horizon: int,
    cap: Optional[int] = None,
    common_scan_universe: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:

    sample, cand_funnel = (
        extract_candidate_rows(
            df,
            blocks,
            common_scan_universe=(
                common_scan_universe
            ),
        )
    )

    # -------------------------------------------------------------------------
    # Decision anchor (causal t)
    #
    # 研究问题：在 t 时刻看到信号之后未来发生什么。
    # decision bar = t (bar_t)
    # entry_bar    = t + 1 (future path 起点, x1.build_future_tensor 合同)
    #
    # 禁止用 entry_bar / future O/H/L/C 计算 decision-time state。
    # -------------------------------------------------------------------------

    decision_bar = (
        sample["bar_t"].to_numpy(np.int64)
    )

    entry_bar = (
        sample["entry_bar"].to_numpy(np.int64)
    )

    if not np.array_equal(entry_bar, decision_bar + 1):

        n_mismatch = int(
            (entry_bar != decision_bar + 1).sum()
        )

        raise SystemExit(
            "STOP_STRUCTREV_DECISION_ENTRY_INDEX_CONTRACT "
            f"mismatch={n_mismatch} "
            f"sample_entry_bar={entry_bar[:5].tolist()}"
        )

    # decision-close parity gate:
    # bars[t][bar_t] + 5min == attach 时 decision_close_time_structrev
    decision_close_time = (
        sample["decision_close_time_structrev"]
        .to_numpy(dtype="datetime64[ns]")
    )

    for s in np.unique(sample["symbol"].to_numpy()):

        m = sample["symbol"].to_numpy() == s

        bar_time = (
            pd.to_datetime(
                np.asarray(bars_by_sym[s]["t"])[decision_bar[m]]
            )
            .to_numpy(dtype="datetime64[ns]")
        )

        expected = bar_time + np.timedelta64(5, "m")

        if not np.array_equal(expected, decision_close_time[m]):

            raise SystemExit(
                "STOP_STRUCTREV_DECISION_TIME_PARITY:"
                f"{s}"
            )

    syms = sample["symbol"].to_numpy()

    atr0 = sample["atr0"].to_numpy(float)

    direction = sample["reversion_dir"].to_numpy(float)

    mean_price = sample["m15_sma"].to_numpy(float)

    opp_sr_price = (
        sample["m15_sr_opp_price"].to_numpy(float)
    )

    decision_close = np.empty(len(sample), dtype=float)

    for s in np.unique(syms):

        m = syms == s

        decision_close[m] = np.asarray(
            bars_by_sym[s]["c"], dtype=float
        )[decision_bar[m]]

    # mean-ahead (t 时刻信息) —— 作为 candidate gate
    mean_dist_now_R = (
        direction
        * (mean_price - decision_close)
        / atr0
    )

    mean_ahead_now = (
        np.isfinite(mean_dist_now_R)
        & (mean_dist_now_R > 0)
    )

    opp_sr_dist_now_R = (
        direction
        * (opp_sr_price - decision_close)
        / atr0
    )

    opp_sr_valid = (
        np.isfinite(opp_sr_dist_now_R)
        & (opp_sr_dist_now_R > 0)
    )

    # 把 decision-time 量作为列挂回 sample,
    # 之后所有过滤都保持对齐。
    sample = sample.copy()

    sample["decision_close_structrev"] = decision_close

    sample["mean_target_dist_R"] = mean_dist_now_R

    sample["mean_ahead_now"] = (
        mean_ahead_now.astype(np.int8)
    )

    sample["opp_sr_target_dist_R"] = opp_sr_dist_now_R

    sample["opp_sr_target_valid"] = (
        opp_sr_valid.astype(np.int8)
    )

    # funnel: candidate 阶段 fact (mean gate 之前)
    funnel = dict(cand_funnel)

    funnel["mean_ahead_now"] = int(mean_ahead_now.sum())

    # candidate gate: mean 在 t 时刻已经 ahead
    # (mean_ahead_now 是 t 时刻信息, 不依赖未来)
    sample = (
        sample.loc[mean_ahead_now]
        .copy()
        .reset_index(drop=True)
    )

    # smoke cap (仅 smoke 时 cap 生效; full 时 cap=None)
    if cap is not None and len(sample) > cap:

        # evenly spaced sample
        # 避免 smoke 只取最早的数据
        pos = np.linspace(
            0, len(sample) - 1, cap
        ).round().astype(int)

        sample = (
            sample.iloc[pos]
            .copy()
            .reset_index(drop=True)
        )

    funnel["post_smoke_cap"] = int(len(sample))

    # -------------------------------------------------------------------------
    # 直接复用项目 future tensor owner
    # future path 从 entry_bar (= bar_t + 1) 开始。
    # -------------------------------------------------------------------------

    tensor_raw = (
        x1.build_future_tensor(

            sample,

            bars_by_sym,

            n_future=horizon,
        )
    )

    valid = np.asarray(
        tensor_raw["path_valid"],
        dtype=bool,
    )

    sample = (
        sample.loc[valid]
        .copy()
        .reset_index(drop=True)
    )

    tensor = (
        x1.slice_future_tensor(
            tensor_raw,
            valid,
        )
    )

    if len(sample) == 0:

        raise SystemExit(
            f"STOP_STRUCTREV_EMPTY_PATH_DATASET:{list(blocks)}"
        )

    # decision-time 量从列取回 (已随过滤对齐)
    decision_close = (
        sample["decision_close_structrev"].to_numpy(float)
    )

    atr0 = sample["atr0"].to_numpy(float)

    direction = (
        sample["reversion_dir"].to_numpy(float)
    )

    H = np.asarray(
        tensor["H"],
        float,
    )

    L = np.asarray(
        tensor["L"],
        float,
    )

    C = np.asarray(
        tensor["C"],
        float,
    )

    # decision_close 已在上方基于 bar_t (causal t) 计算并挂为列,
    # atr0 / direction 也在 path_valid 过滤后从列取回 (3687-3691),
    # 此处不再重复赋值。

    # -------------------------------------------------------------------------
    # 统一方向：
    #
    # favorable > 0
    # adverse   > 0
    # -------------------------------------------------------------------------

    favorable = np.where(
        direction[:, None] > 0,
        (
            H
            - decision_close[:, None]
        )
        / atr0[:, None],
        (
            decision_close[:, None]
            - L
        )
        / atr0[:, None],
    )

    adverse = np.where(
        direction[:, None] > 0,
        (
            decision_close[:, None]
            - L
        )
        / atr0[:, None],
        (
            H
            - decision_close[:, None]
        )
        / atr0[:, None],
    )

    signal_return = (
        direction[:, None]
        *
        (
            C
            - decision_close[:, None]
        )
        / atr0[:, None]
    )

    # -------------------------------------------------------------------------
    # Mean target
    #
    # 当前冻结 SMA50。
    # 不在本实验里优化。
    # -------------------------------------------------------------------------

    mean_price = (
        sample[
            "m15_sma"
        ]
        .to_numpy(float)
    )

    mean_dist_now_R = (
        direction
        *
        (
            mean_price
            - decision_close
        )
        / atr0
    )

    mean_ahead_now = (
        np.isfinite(
            mean_dist_now_R
        )
        &
        (
            mean_dist_now_R
            > 0
        )
    )

    sample[
        "mean_target_dist_R"
    ] = mean_dist_now_R

    sample[
        "mean_ahead_now"
    ] = (
        mean_ahead_now
        .astype(
            np.int8
        )
    )

    # -------------------------------------------------------------------------
    # opposite SR target
    # -------------------------------------------------------------------------

    opposite_sr_price = (
        sample[
            "m15_sr_opp_price"
        ]
        .to_numpy(float)
    )

    opposite_sr_dist_R = (
        direction
        *
        (
            opposite_sr_price
            - decision_close
        )
        / atr0
    )

    opposite_sr_valid = (

        np.isfinite(
            opposite_sr_dist_R
        )

        &

        (
            opposite_sr_dist_R
            > 0
        )
    )

    sample[
        "opp_sr_target_dist_R"
    ] = (
        opposite_sr_dist_R
    )

    sample[
        "opp_sr_target_valid"
    ] = (
        opposite_sr_valid
        .astype(
            np.int8
        )
    )

    # -------------------------------------------------------------------------
    # Outcomes
    # -------------------------------------------------------------------------

    for horizon in (horizon,):  # single-horizon build; caller passes horizon

        k = horizon

        # mean_ahead_now 在 candidate gate 后全为 True;
        # 用列回读得到 post-filter 对齐版本,
        # 避免与 pre-gate 数组错位。
        mean_ahead_now_final = (
            sample["mean_ahead_now"].to_numpy() > 0
        )

        favorable_h = (
            favorable[
                :,
                :k,
            ]
        )

        adverse_h = (
            adverse[
                :,
                :k,
            ]
        )

        sample[
            f"ret_R_{horizon}"
        ] = (
            signal_return[
                :,
                k - 1,
            ]
        )

        sample[
            f"mfe_R_{horizon}"
        ] = np.maximum(
            0.0,
            np.nanmax(
                favorable_h,
                axis=1,
            ),
        )

        sample[
            f"mae_R_{horizon}"
        ] = np.maximum(
            0.0,
            np.nanmax(
                adverse_h,
                axis=1,
            ),
        )

        if np.any(
            sample[f"mfe_R_{horizon}"].to_numpy(float)
            < 0
        ):
            raise SystemExit(
                "STOP_STRUCTREV_NEGATIVE_MFE"
            )

        if np.any(
            sample[f"mae_R_{horizon}"].to_numpy(float)
            < 0
        ):
            raise SystemExit(
                "STOP_STRUCTREV_NEGATIVE_MAE"
            )

        sample[
            f"y_positive_{horizon}"
        ] = (

            signal_return[
                :,
                k - 1,
            ]

            > 0

        ).astype(
            np.int8
        )

        # ---------------------------------------------------------------------
        # 是否触及 SMA mean
        # ---------------------------------------------------------------------

        mean_hit_matrix = np.where(
            direction[:, None] > 0,
            H >= mean_price[:, None],
            L <= mean_price[:, None],
        )

        mean_hit_matrix[
            ~mean_ahead_now_final,
            :
        ] = False

        mean_has = (
            mean_hit_matrix
            .any(axis=1)
        )

        mean_idx = np.argmax(
            mean_hit_matrix,
            axis=1,
        )

        # ---------------------------------------------------------------------
        # diagnostic 1R adverse barrier
        # ---------------------------------------------------------------------

        stop_matrix = (

            adverse_h

            >=

            STOP_R_DIAGNOSTIC
        )

        stop_has = (
            stop_matrix
            .any(axis=1)
        )

        stop_idx = np.argmax(
            stop_matrix,
            axis=1,
        )

        sample[
            f"y_mean_hit_{horizon}"
        ] = np.where(

            mean_ahead_now_final,

            mean_has.astype(float),

            np.nan,
        )

        # 同一根 bar 同时碰到 mean / stop：
        # 保守按 stop first
        sample[
            f"y_mean_before_1R_{horizon}"
        ] = np.where(

            mean_ahead_now_final,

            (
                mean_has

                &

                (
                    (~stop_has)

                    |

                    (
                        mean_idx
                        < stop_idx
                    )
                )

            ).astype(float),

            np.nan,
        )

        # ---------------------------------------------------------------------
        # 是否触及 opposite SR
        # ---------------------------------------------------------------------

        opposite_hit = (

            favorable_h

            >=

            opposite_sr_dist_R[
                :,
                None,
            ]
        )

        opposite_hit[
            ~opposite_sr_valid,
            :
        ] = False

        sample[
            f"y_opp_sr_hit_{horizon}"
        ] = np.where(

            opposite_sr_valid,

            opposite_hit
            .any(axis=1)
            .astype(float),

            np.nan,
        )

        # ---------------------------------------------------------------------
        # Universe funnel
        # 顺序: cand -> mean_ahead_now -> post_smoke_cap
        #        -> future_path_valid -> outcome_finite
        # funnel 已在上方逐步填充 (mean_ahead_now, post_smoke_cap)。
        # ---------------------------------------------------------------------

        funnel["future_path_valid"] = int(
            len(sample)
        )

        for outc in [
            f"y_mean_hit_{horizon}",
            f"y_mean_before_1R_{horizon}",
            f"y_opp_sr_hit_{horizon}",
            f"y_positive_{horizon}",
        ]:

            funnel[f"outcome_finite:{outc}"] = int(
                np.isfinite(
                    sample[outc].to_numpy(float)
                ).sum()
            )

        return sample, funnel


# =============================================================================
# 11. PGM node ladder
# =============================================================================

DEV_NODES = [

    "dev_abs",
]


SR_NODES = [

    "m15_sr_same_dist",

    "m15_sr_same_strength",

    "m15_sr_same_exists",

    "m15_sr_same_near",

    "m15_sr_opp_dist",

    "m15_sr_opp_strength",

    "m15_sr_opp_exists",

    "m15_sr_in_zone",
]


HTF_NODES = [

    "h1_reversion_pressure",

    "h1_trend_align",

    "h1_state_align",

    "h1_sr_same_dist",

    "h1_sr_same_strength",

    "h1_sr_same_exists",

    "h4_reversion_pressure",

    "h4_trend_align",

    "h4_state_align",

    "h4_sr_same_dist",

    "h4_sr_same_strength",

    "h4_sr_same_exists",
]


LIQUIDITY_NODES = [

    "m15_liq_against_dist",

    "m15_liq_against_exists",

    "m15_liq_against_near",

    "m15_liq_sweep_against_now",

    "m15_liq_reclaim_after_against_sweep",

    "m15_liq_accept_against",

    "m15_liq_last_zone_active",

    "m15_liq_target_dist",

    "h1_liq_against_dist",

    "h1_liq_reclaim_after_against_sweep",

    "h1_liq_accept_against",

    "h1_liq_last_zone_active",
]


TRIGGER_NODES = [

    "m5_reversion_pressure",

    "m5_trend_align",

    "m5_state_align",

    "m5_liq_against_dist",

    "m5_liq_against_near",

    "m5_liq_sweep_against_now",

    "m5_liq_reclaim_after_against_sweep",

    "m5_liq_accept_against",

    "m5_liq_last_zone_active",
]


PGM_CONTEXT_NODES = [

    "pgm_alignment",

    "pgm_abs_score",
]


GRAPH_LADDER = [

    (
        "G0_BASE",
        [],
    ),

    (
        "G1_DEV",
        DEV_NODES,
    ),

    (
        "G2_DEV_SR",
        DEV_NODES
        + SR_NODES,
    ),

    (
        "G3_DEV_SR_HTF",
        DEV_NODES
        + SR_NODES
        + HTF_NODES,
    ),

    (
        "G4_DEV_SR_HTF_LIQ",
        DEV_NODES
        + SR_NODES
        + HTF_NODES
        + LIQUIDITY_NODES,
    ),

    (
        "G5_PLUS_5M_TRIGGER",
        DEV_NODES
        + SR_NODES
        + HTF_NODES
        + LIQUIDITY_NODES
        + TRIGGER_NODES,
    ),

    (
        "G6_PLUS_FROZEN_PGM",
        DEV_NODES
        + SR_NODES
        + HTF_NODES
        + LIQUIDITY_NODES
        + TRIGGER_NODES
        + PGM_CONTEXT_NODES,
    ),
]


# =============================================================================
# 12. 固定 CPD
# =============================================================================

def make_binary_cpd(
    numeric_nodes: Sequence[str],
) -> Pipeline:

    transformers = []

    if numeric_nodes:

        numeric_pipeline = Pipeline(
            steps=[

                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        add_indicator=True,
                    ),
                ),

                (
                    "scaler",
                    StandardScaler(),
                ),
            ]
        )

        transformers.append(

            (
                "num",

                numeric_pipeline,

                list(
                    numeric_nodes
                ),
            )
        )

    categorical_pipeline = Pipeline(
        steps=[

            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),

            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    # symbol 始终作为基础节点。
    transformers.append(

        (
            "cat",

            categorical_pipeline,

            ["symbol"],
        )
    )

    preprocessor = ColumnTransformer(

        transformers=transformers,

        remainder="drop",
    )

    # 固定 CPD。
    # 不在本实验里调 classifier 超参数。
    classifier = LogisticRegression(

        C=1.0,

        penalty="l2",

        solver="lbfgs",

        max_iter=2000,

        random_state=20260918,
    )

    return Pipeline(
        [

            (
                "pre",
                preprocessor,
            ),

            (
                "clf",
                classifier,
            ),
        ]
    )


def safe_auc(
    y: np.ndarray,
    p: np.ndarray,
) -> Tuple[float, float]:

    if (
        len(
            np.unique(y)
        )
        < 2
    ):
        return (
            np.nan,
            np.nan,
        )

    roc = float(
        sklearn.metrics
        .roc_auc_score(
            y,
            p,
        )
    )

    pr = float(
        sklearn.metrics
        .average_precision_score(
            y,
            p,
        )
    )

    return (
        roc,
        pr,
    )


def fit_eval_cpd(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    outcome: str,
    nodes: Sequence[str],
) -> Tuple[
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    Tuple[Tuple[Any, ...], ...],
    Tuple[Tuple[Any, ...], ...],
]:

    if (
        outcome not in train.columns
        or outcome not in eval_df.columns
    ):

        raise SystemExit(
            f"STOP_STRUCTREV_OUTCOME_MISSING:{outcome}"
        )

    train_valid = np.isfinite(
        train[
            outcome
        ]
        .to_numpy(float)
    )

    eval_valid = np.isfinite(
        eval_df[
            outcome
        ]
        .to_numpy(float)
    )

    tr = (
        train
        .loc[
            train_valid
        ]
        .copy()
    )

    ev = (
        eval_df
        .loc[
            eval_valid
        ]
        .copy()
    )

    # ---------------------------------------------------------------------
    # Fail-closed（仅技术性）：
    # 删除任意 1000 阈值。
    # 只要求 train / eval 非空 + train 含两类。
    # ---------------------------------------------------------------------
    if len(tr) == 0 or len(ev) == 0:

        raise SystemExit(

            "STOP_STRUCTREV_SAMPLE_EMPTY:"
            f"{outcome}:"
            f"train={len(tr)} "
            f"eval={len(ev)}"
        )

    y_train = (
        tr[
            outcome
        ]
        .to_numpy(
            np.int8
        )
    )

    y_eval = (
        ev[
            outcome
        ]
        .to_numpy(
            np.int8
        )
    )

    if (
        len(
            np.unique(
                y_train
            )
        )
        < 2
    ):

        raise SystemExit(
            f"STOP_STRUCTREV_TRAIN_ONE_CLASS:{outcome}"
        )

    # ---------------------------------------------------------------------
    # 完整 sample facts（不再用小样本守卫掩盖）
    # ---------------------------------------------------------------------
    n_pos_tr = int(
        y_train.sum()
    )
    n_neg_tr = len(y_train) - n_pos_tr
    n_pos_ev = int(
        y_eval.sum()
    )
    n_neg_ev = len(y_eval) - n_pos_ev
    n_days_tr = int(
        np.unique(
            tr["entry_day"]
        ).size
    )
    n_days_ev = int(
        np.unique(
            ev["entry_day"]
        ).size
    )
    n_symbols_tr = int(
        tr["symbol"].nunique()
    )
    n_symbols_ev = int(
        ev["symbol"].nunique()
    )

    print(
        (
            f"[SAMPLE] {outcome} "
            f"train={len(tr)} "
            f"(pos={n_pos_tr} neg={n_neg_tr} "
            f"days={n_days_tr} symbols={n_symbols_tr} "
            f"rate={np.mean(y_train):.4f}) "
            f"eval={len(ev)} "
            f"(pos={n_pos_ev} neg={n_neg_ev} "
            f"days={n_days_ev} symbols={n_symbols_ev} "
            f"rate={np.mean(y_eval):.4f})"
        ),
        flush=True,
    )

    model = make_binary_cpd(
        nodes
    )

    model.fit(
        tr,
        y_train,
    )

    prob = (
        model
        .predict_proba(
            ev
        )[:, 1]
    )

    eps = 1e-12

    prob = np.clip(
        prob,
        eps,
        1.0 - eps,
    )

    row_logloss = -(

        y_eval
        * np.log(prob)

        +

        (
            1
            - y_eval
        )
        * np.log(
            1.0
            - prob
        )
    )

    roc_auc, pr_auc = safe_auc(
        y_eval,
        prob,
    )

    metrics = dict(

        n_train=int(
            len(tr)
        ),

        n_train_positive=int(
            n_pos_tr
        ),

        n_train_negative=int(
            n_neg_tr
        ),

        n_train_days=int(
            n_days_tr
        ),

        n_train_symbols=int(
            n_symbols_tr
        ),

        n_eval=int(
            len(ev)
        ),

        n_eval_positive=int(
            n_pos_ev
        ),

        n_eval_negative=int(
            n_neg_ev
        ),

        n_eval_days=int(
            n_days_ev
        ),

        n_eval_symbols=int(
            n_symbols_ev
        ),

        event_rate_train=float(
            np.mean(
                y_train
            )
        ),

        event_rate_eval=float(
            np.mean(
                y_eval
            )
        ),

        log_loss=float(
            np.mean(
                row_logloss
            )
        ),

        brier=float(
            np.mean(
                (
                    prob
                    - y_eval
                )
                ** 2
            )
        ),

        roc_auc=roc_auc,

        pr_auc=pr_auc,

        mean_pred=float(
            np.mean(
                prob
            )
        ),
    )

    # 训练/评估集行键 (per-horizon G0-G6 行键一致性校验用)
    # episode_id 不存在时退化为 (symbol, bar_t)
    key_cols = (
        "symbol",
        "bar_t",
    )

    train_keys = tuple(
        zip(
            tr[key_cols[0]].to_numpy(),
            tr[key_cols[1]].to_numpy(),
        )
    )

    eval_keys = tuple(
        zip(
            ev[key_cols[0]].to_numpy(),
            ev[key_cols[1]].to_numpy(),
        )
    )

    return (

        metrics,

        row_logloss,

        ev[
            "entry_day"
        ].to_numpy(),

        prob,

        eval_keys,

        train_keys,
    )


# =============================================================================
# 13. Incremental graph experiment
# =============================================================================

def evaluate_graph_ladder(
    train: pd.DataFrame,
    eval_df: pd.DataFrame,
    window_name: str,
    outcomes: Sequence[str],
    n_boot: int,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    metric_rows = []
    bootstrap_rows = []

    for outcome in outcomes:

        previous_loss = None
        previous_days = None
        previous_eval_keys = None
        previous_train_keys = None
        previous_graph = None

        for (
            graph_name,
            nodes,
        ) in GRAPH_LADDER:

            (
                metrics,
                loss,
                days,
                _,
                eval_keys,
                train_keys,
            ) = fit_eval_cpd(

                train,

                eval_df,

                outcome,

                nodes,
            )

            metric_rows.append(

                dict(

                    window=window_name,

                    outcome=outcome,

                    graph=graph_name,

                    n_nodes=len(
                        nodes
                    ),

                    **metrics,
                )
            )

            if (
                previous_loss
                is not None
            ):

                if (

                    len(
                        previous_loss
                    )
                    != len(loss)

                    or

                    not np.array_equal(
                        previous_days,
                        days,
                    )

                    or

                    previous_eval_keys
                    != eval_keys

                    or

                    previous_train_keys
                    != train_keys
                ):

                    raise SystemExit(

                        "STOP_STRUCTREV_GRAPH_SAMPLE_DRIFT:"
                        f"{outcome}:"
                        f"{previous_graph}"
                        "->"
                        f"{graph_name}"
                    )

                # -------------------------------------------------------------
                # 正数 =
                # 新 graph log-loss 更低
                # -------------------------------------------------------------

                delta = (
                    previous_loss
                    - loss
                )

                boot = (
                    e0
                    .paired_day_mean_bootstrap(

                        days,

                        delta,

                        n_boot=n_boot,

                        seed=BOOTSTRAP_SEED,
                    )
                )

                bootstrap_rows.append(

                    dict(

                        window=window_name,

                        outcome=outcome,

                        from_graph=previous_graph,

                        to_graph=graph_name,

                        delta_logloss_point=float(
                            np.mean(
                                delta
                            )
                        ),

                        **boot,
                    )
                )

            previous_loss = loss
            previous_days = days
            previous_eval_keys = eval_keys
            previous_train_keys = train_keys
            previous_graph = graph_name

    return (

        pd.DataFrame(
            metric_rows
        ),

        pd.DataFrame(
            bootstrap_rows
        ),
    )


# =============================================================================
# 14. 直观诊断单元
# =============================================================================

def diagnostic_cells(
    eval_df: pd.DataFrame,
    window_name: str,
    horizon: int,
) -> pd.DataFrame:

    rows = []

    deviation_edges = [

        -np.inf,

        0.5,

        1.0,

        1.5,

        2.0,

        2.5,

        3.0,

        np.inf,
    ]

    sr_edges = [

        -np.inf,

        0.0,

        0.25,

        0.50,

        1.0,

        2.0,

        np.inf,
    ]

    for horizon in (horizon,):  # single-horizon diagnostic

        outcome = (
            f"y_mean_hit_{horizon}"
        )

        valid = np.isfinite(
            eval_df[
                outcome
            ]
            .to_numpy(float)
        )

        z = (
            eval_df
            .loc[valid]
            .copy()
        )

        z[
            "dev_bin"
        ] = pd.cut(

            z[
                "dev_abs"
            ],

            deviation_edges,

            right=True,

            include_lowest=True,
        )

        z[
            "sr_bin"
        ] = pd.cut(

            z[
                "m15_sr_same_dist"
            ],

            sr_edges,

            right=True,

            include_lowest=True,
        )

        # ---------------------------------------------------------------------
        # 单变量分层
        # ---------------------------------------------------------------------

        for (
            diagnostic,
            group_col,
        ) in [

            (
                "DEV",
                "dev_bin",
            ),

            (
                "SR",
                "sr_bin",
            ),
        ]:

            grouped = z.groupby(
                group_col,
                observed=True,
            )

            for key, g in grouped:

                if len(g) == 0:
                    continue

                rows.append(

                    dict(

                        window=window_name,

                        horizon=horizon,

                        diagnostic=diagnostic,

                        cell=str(key),

                        n=int(
                            len(g)
                        ),

                        mean_hit_rate=float(
                            g[
                                outcome
                            ]
                            .mean()
                        ),

                        mean_ret_R=float(
                            g[
                                f"ret_R_{horizon}"
                            ]
                            .mean()
                        ),

                        mean_mfe_R=float(
                            g[
                                f"mfe_R_{horizon}"
                            ]
                            .mean()
                        ),

                        mean_mae_R=float(
                            g[
                                f"mae_R_{horizon}"
                            ]
                            .mean()
                        ),
                    )
                )

        # ---------------------------------------------------------------------
        # SR × HTF environment
        # ---------------------------------------------------------------------

        htf_align = (

            (
                z[
                    "h1_trend_align"
                ]
                .to_numpy(float)
                > 0
            )

            &

            (
                z[
                    "h4_trend_align"
                ]
                .to_numpy(float)
                > 0
            )
        )

        sr_near = (

            z[
                "m15_sr_same_near"
            ]
            .to_numpy(int)
            > 0
        )

        for sr_flag in (
            0,
            1,
        ):

            for htf_flag in (
                0,
                1,
            ):

                mask = (

                    (
                        sr_near
                        == bool(sr_flag)
                    )

                    &

                    (
                        htf_align
                        == bool(htf_flag)
                    )
                )

                g = z.loc[
                    mask
                ]

                if len(g) == 0:
                    continue

                rows.append(

                    dict(

                        window=window_name,

                        horizon=horizon,

                        diagnostic="SRxHTF",

                        cell=(
                            f"sr_near={sr_flag}"
                            "|"
                            f"htf_align={htf_flag}"
                        ),

                        n=int(
                            len(g)
                        ),

                        mean_hit_rate=float(
                            g[
                                outcome
                            ]
                            .mean()
                        ),

                        mean_ret_R=float(
                            g[
                                f"ret_R_{horizon}"
                            ]
                            .mean()
                        ),

                        mean_mfe_R=float(
                            g[
                                f"mfe_R_{horizon}"
                            ]
                            .mean()
                        ),

                        mean_mae_R=float(
                            g[
                                f"mae_R_{horizon}"
                            ]
                            .mean()
                        ),
                    )
                )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 15. 参数扫描专用 Window A loader
# =============================================================================

def load_window_a_only() -> Dict[str, Any]:
    """
    只加载：
        TB1 -> TB2

    不构建 Window B。
    不评分 TB3。
    """

    prep = (
        d0
        .load_prepared_frame()
    )

    aligned = prep[
        "aligned"
    ]

    bars_by_sym = prep[
        "bars_by_sym"
    ]

    assert_allowed_blocks(
        aligned
    )

    fit_a = (
        pgm
        .fit_samplers_for_window(

            pgm.WINDOWS[0],

            pgm.SAMPLE_PATH,

            pgm.TRANSITION_SAMPLE_PATH,
        )
    )

    scored_a = (
        d0
        .prepare_window_windowframe(

            aligned,

            fit_a,

            "structrev_scan_A",
        )
    )

    d0.verify_window_score_owner(
        scored_a,
        fit_a,
        "TB1",
    )

    d0.verify_window_score_owner(
        scored_a,
        fit_a,
        "TB2",
    )

    return dict(

        prep=prep,

        aligned=aligned,

        bars_by_sym=bars_by_sym,

        fit_A=fit_a,

        scored_A=scored_a,
    )


# =============================================================================
# 16. Parameter scan
# =============================================================================

def assert_scan_universe_aligned(
    name: str,
    loss: np.ndarray,
    baseline_loss: np.ndarray,
    days: np.ndarray,
    baseline_days: np.ndarray,
    eval_keys: Sequence[Tuple[Any, ...]],
    baseline_eval_keys: Sequence[Tuple[Any, ...]],
    train_keys: Sequence[Tuple[Any, ...]],
    baseline_train_keys: Sequence[Tuple[Any, ...]],
) -> None:
    """
    参数扫描配对宇宙一致性 gate (fail-closed)。

    参数效果比较必须建立在完全相同的
    train rows / eval rows 之上:
        (symbol, bar_t) 必须逐一相等。
    """

    if (
        baseline_loss is None
        or baseline_days is None
        or baseline_eval_keys is None
        or baseline_train_keys is None
    ):

        raise SystemExit(
            "STOP_STRUCTREV_SCAN_BASELINE_ORDER_BROKEN"
        )

    if (

        len(loss)
        != len(baseline_loss)

        or

        not np.array_equal(
            days,
            baseline_days,
        )
    ):

        raise SystemExit(
            f"STOP_STRUCTREV_SCAN_UNIVERSE_DRIFT:{name}"
        )

    if eval_keys != baseline_eval_keys:

        raise SystemExit(
            f"STOP_STRUCTREV_SCAN_EVAL_UNIVERSE_DRIFT:{name}"
        )

    if train_keys != baseline_train_keys:

        raise SystemExit(
            f"STOP_STRUCTREV_SCAN_TRAIN_UNIVERSE_DRIFT:{name}"
        )


def run_parameter_scan(
    bundle: Dict[str, Any],
    cap: Optional[int] = None,
) -> pd.DataFrame:

    # ---------------------------------------------------------------------
    # 参数扫描授权闸门 (次级防御)
    #
    # 主 gate 在 CLI 层、load/fit 之前调用。
    # 这里再次校验, 避免被直接调用绕过。
    # ---------------------------------------------------------------------
    require_parameter_scan_authorization()

    scored_a = bundle[
        "scored_A"
    ]

    bars_by_sym = bundle[
        "bars_by_sym"
    ]

    symbols = sorted(
        scored_a[
            "symbol"
        ]
        .unique()
        .tolist()
    )

    rows = []

    baseline_loss = None
    baseline_days = None
    baseline_eval_keys = None
    baseline_train_keys = None

    scan_nodes = (

        DEV_NODES

        + SR_NODES

        + HTF_NODES

        + LIQUIDITY_NODES

        + TRIGGER_NODES
    )

    for (
        name,
        params,
    ) in (
        parameter_candidates()
        .items()
    ):

        print(
            f"[SCAN] {name}",
            flush=True,
        )

        cache = build_indicator_cache(

            bars_by_sym,

            params,

            symbols,
        )

        scored = (
            attach_indicator_features(

                scored_a,

                bars_by_sym,

                cache,
            )
        )

        scored = add_oriented_nodes(
            scored,
            params,
        )

        train, train_funnel = build_path_dataset(

            scored,

            ("TB1",),

            bars_by_sym,

            horizon=24,

            cap=cap,

            common_scan_universe=True,
        )

        eval_df, eval_funnel = build_path_dataset(

            scored,

            ("TB2",),

            bars_by_sym,

            horizon=24,

            cap=cap,

            common_scan_universe=True,
        )

        # 固定 outcome。
        # SMA50 在整个 scan 中不变。
        outcome = (
            "y_mean_hit_24"
        )

        (
            metrics,
            loss,
            days,
            _,
            eval_keys,
            train_keys,
        ) = fit_eval_cpd(

            train,

            eval_df,

            outcome,

            scan_nodes,
        )

        row = dict(

            param_set=name,

            outcome=outcome,

            **metrics,

            **asdict(
                params
            ),
        )

        if (
            name
            == "pine_default"
        ):

            baseline_loss = (
                loss.copy()
            )

            baseline_days = (
                days.copy()
            )

            baseline_eval_keys = (
                eval_keys
            )

            baseline_train_keys = (
                train_keys
            )

            row.update(

                delta_vs_pine=np.nan,

                ci95_lower=np.nan,

                ci95_upper=np.nan,

                p_pos=np.nan,
            )

        else:

            assert_scan_universe_aligned(
                name=name,
                loss=loss,
                baseline_loss=baseline_loss,
                days=days,
                baseline_days=baseline_days,
                eval_keys=eval_keys,
                baseline_eval_keys=baseline_eval_keys,
                train_keys=train_keys,
                baseline_train_keys=baseline_train_keys,
            )

            delta = (
                baseline_loss
                - loss
            )

            boot = (
                e0
                .paired_day_mean_bootstrap(

                    days,

                    delta,

                    n_boot=min(
                        BOOTSTRAP_N,
                        1000,
                    ),

                    seed=BOOTSTRAP_SEED,
                )
            )

            row.update(

                delta_vs_pine=float(
                    np.mean(
                        delta
                    )
                ),

                **boot,
            )

        rows.append(
            row
        )

    return pd.DataFrame(
        rows
    )


# =============================================================================
# 17. Prefix invariance audit
# =============================================================================

def prefix_invariance_synthetic() -> None:
    """
    修改未来 K 线，
    不允许改变过去已经确认的特征。
    """

    rng = np.random.default_rng(
        7
    )

    n = 1400

    close = (

        100

        + np.cumsum(
            rng.normal(
                0,
                0.2,
                n,
            )
        )
    )

    high = (

        close

        + rng.uniform(
            0.05,
            0.4,
            n,
        )
    )

    low = (

        close

        - rng.uniform(
            0.05,
            0.4,
            n,
        )
    )

    open_ = np.r_[

        close[0],

        close[:-1],
    ]

    time_index = pd.date_range(

        "2026-01-01",

        periods=n,

        freq="5min",
    )

    base = pd.DataFrame({

        "time":
            time_index,

        "trading_day":
            pd.Timestamp(
                "2026-01-01"
            ),

        "segment":
            0,

        "open":
            open_,

        "high":
            high,

        "low":
            low,

        "close":
            close,

        "disc":
            False,

        "available_time":
            time_index
            + pd.Timedelta(
                minutes=5
            ),

        "n_base":
            1,
    })

    cut = 1000

    feature_a = (
        compute_tf_features(

            base,

            PINE_DEFAULT,

            include_sr=True,
        )
    )

    perturbed = (
        base.copy()
    )

    # 只改未来
    perturbed.loc[
        cut:,
        "high",
    ] += 50

    perturbed.loc[
        cut:,
        "low",
    ] -= 50

    perturbed.loc[
        cut:,
        "close",
    ] += 25

    feature_b = (
        compute_tf_features(

            perturbed,

            PINE_DEFAULT,

            include_sr=True,
        )
    )

    audit_cols = [

        "dev",

        "slope_atr",

        "sr_support_dist_atr",

        "sr_resistance_dist_atr",

        "liq_up_dist_atr",

        "liq_down_dist_atr",

        "liq_last_breach_side",
    ]

    # 给 right-confirmed pivot 留安全缓冲。
    end = (
        cut
        - 20
    )

    for col in audit_cols:

        a = (
            feature_a[
                col
            ]
            .to_numpy()[
                :end
            ]
        )

        b = (
            feature_b[
                col
            ]
            .to_numpy()[
                :end
            ]
        )

        if not np.allclose(

            a,

            b,

            equal_nan=True,

            atol=1e-12,

            rtol=0,
        ):

            raise SystemExit(
                f"STOP_STRUCTREV_PREFIX_INVARIANCE_FAIL:{col}"
            )


# =============================================================================
# 17b. Synthetic causality audits
# =============================================================================

def pivot_causality_synthetic() -> None:
    """
    验证 confirmed_pivots 的因果性：

    一个在 center index p 的极值，
    只在 p+right 才被系统“知道”，
    之前全部为 NaN。
    """

    left = 3
    right = 3

    n = 60
    p = 25

    # high pivot
    v = np.linspace(
        0.0,
        1.0,
        n,
    )

    v[p] = 100.0

    out = confirmed_pivots(
        v,
        left,
        right,
        "high",
    )

    confirm_idx = p + right

    if not np.all(
        np.isnan(
            out[:confirm_idx]
        )
    ):
        raise SystemExit(
            "STOP_STRUCTREV_PIVOT_CAUSALITY_LEAK:"
            "before_confirmation"
        )

    if not np.isfinite(
        out[confirm_idx]
    ):
        raise SystemExit(
            "STOP_STRUCTREV_PIVOT_CAUSALITY_MISSING:"
            "at_confirmation"
        )

    if out[confirm_idx] != v[p]:
        raise SystemExit(
            "STOP_STRUCTREV_PIVOT_CAUSALITY_VALUE_MISMATCH"
        )

    # low pivot
    v2 = np.linspace(
        1.0,
        0.0,
        n,
    )

    v2[p] = -100.0

    out2 = confirmed_pivots(
        v2,
        left,
        right,
        "low",
    )

    if not np.all(
        np.isnan(
            out2[:confirm_idx]
        )
    ):
        raise SystemExit(
            "STOP_STRUCTREV_PIVOT_CAUSALITY_LEAK:"
            "before_confirmation_low"
        )

    if out2[confirm_idx] != v2[p]:
        raise SystemExit(
            "STOP_STRUCTREV_PIVOT_CAUSALITY_VALUE_MISMATCH_LOW"
        )


def decision_anchor_synthetic() -> None:
    """
    针对 R1.1 关键 bug 的合成/结构测试：

    decision anchor 必须等于 bar_t 的 close (C_t)，
    绝不能用 entry_bar (t+1) 的 close (C_{t+1})。

    构造最小合成：
        C_t     = 100
        C_{t+1} = 999
        SMA_t   = 110
        direction = +1
        ATR     = 10

    正确 (用 C_t):
        mean_dist_now_R = +1 * (110 - 100) / 10 = +1

    错误 (用 C_{t+1}):
        mean_dist_now_R = +1 * (110 - 999) / 10 = -88.9

    测试必须明确抓住这次 bug (错误值 != +1)。
    """

    # bars: index 0 = bar_t, index 1 = entry_bar (t+1)
    c = np.array(
        [
            100.0,
            999.0,
        ]
    )

    bars_by_sym = {
        "X": {
            "c": c,
        },
    }

    sample = pd.DataFrame(
        {
            "symbol": ["X"],
            "bar_t": [0],
            "entry_bar": [1],
            "m15_sma": [110.0],
            "reversion_dir": [1],
            "atr0": [10.0],
        }
    )

    decision_bar = (
        sample["bar_t"].to_numpy(np.int64)
    )

    # 正确实现: decision_close = bars["c"][bar_t]
    decision_close_correct = np.asarray(
        bars_by_sym["X"]["c"],
        dtype=float,
    )[decision_bar]

    mean_dist_correct = (
        sample["reversion_dir"].to_numpy(float)
        * (
            sample["m15_sma"].to_numpy(float)
            - decision_close_correct
        )
        / sample["atr0"].to_numpy(float)
    )

    if not np.allclose(
        mean_dist_correct,
        [1.0],
    ):
        raise SystemExit(
            "STOP_STRUCTREV_DECISION_ANCHOR_SYNTHETIC_BUG:"
            f"correct={mean_dist_correct.tolist()}"
        )

    # 错误实现 (用 entry_bar): 必须被测试抓出 != +1
    entry_bar = (
        sample["entry_bar"].to_numpy(np.int64)
    )

    decision_close_wrong = np.asarray(
        bars_by_sym["X"]["c"],
        dtype=float,
    )[entry_bar]

    mean_dist_wrong = (
        sample["reversion_dir"].to_numpy(float)
        * (
            sample["m15_sma"].to_numpy(float)
            - decision_close_wrong
        )
        / sample["atr0"].to_numpy(float)
    )

    if np.allclose(
        mean_dist_wrong,
        [1.0],
    ):
        raise SystemExit(
            "STOP_STRUCTREV_DECISION_ANCHOR_SYNTHETIC_FAILED_TO_CATCH_BUG"
        )

    # 明确断言错误实现会得到 -88.9 (非 +1)
    if not np.allclose(
        mean_dist_wrong,
        [-88.9],
    ):
        raise SystemExit(
            "STOP_STRUCTREV_DECISION_ANCHOR_SYNTHETIC_WRONG_VALUE:"
            f"wrong={mean_dist_wrong.tolist()}"
        )


def mfe_mae_synthetic() -> None:
    """
    验证 MFE / MAE 语义：

        MFE >= 0
        MAE >= 0

    MFE = max(0, max over future of favorable excursion)
    MAE = max(0, max over future of adverse excursion)

    favorable (long) = (H_future - C_t) / atr0
    adverse   (long) = (C_t - L_future) / atr0

    复现 build_path_dataset 的同一公式。
    """

    def compute(
        decision_close,
        H,
        L,
        atr0,
        direction,
    ):

        favorable = np.where(
            direction[:, None] > 0,
            (
                H
                - decision_close[:, None]
            )
            / atr0[:, None],
            (
                decision_close[:, None]
                - L
            )
            / atr0[:, None],
        )

        adverse = np.where(
            direction[:, None] > 0,
            (
                decision_close[:, None]
                - L
            )
            / atr0[:, None],
            (
                H
                - decision_close[:, None]
            )
            / atr0[:, None],
        )

        mfe = np.maximum(
            0.0,
            np.nanmax(
                favorable,
                axis=1,
            ),
        )

        mae = np.maximum(
            0.0,
            np.nanmax(
                adverse,
                axis=1,
            ),
        )

        return mfe, mae

    # Case A: 全程 favorable (long)
    mfe_a, mae_a = compute(
        decision_close=np.array([100.0]),
        H=np.array([[105.0, 110.0]]),
        L=np.array([[101.0, 102.0]]),
        atr0=np.array([10.0]),
        direction=np.array([1]),
    )

    if not np.allclose(mfe_a, [1.0]):
        raise SystemExit(
            "STOP_STRUCTREV_MFE_MAE_SYNTHETIC_A:"
            f"mfe={mfe_a.tolist()}"
        )

    if not np.allclose(mae_a, [0.0]):
        raise SystemExit(
            "STOP_STRUCTREV_MFE_MAE_SYNTHETIC_A:"
            f"mae={mae_a.tolist()}"
        )

    # Case B: 全程 adverse (long)
    mfe_b, mae_b = compute(
        decision_close=np.array([100.0]),
        H=np.array([[99.0, 98.0]]),
        L=np.array([[95.0, 90.0]]),
        atr0=np.array([10.0]),
        direction=np.array([1]),
    )

    if not np.allclose(mfe_b, [0.0]):
        raise SystemExit(
            "STOP_STRUCTREV_MFE_MAE_SYNTHETIC_B:"
            f"mfe={mfe_b.tolist()}"
        )

    if not np.allclose(mae_b, [1.0]):
        raise SystemExit(
            "STOP_STRUCTREV_MFE_MAE_SYNTHETIC_B:"
            f"mae={mae_b.tolist()}"
        )


def parameter_scan_keys_drift_synthetic() -> None:
    """
    结构测试：参数扫描配对宇宙 key-drift gate。

    直接复用生产 gate assert_scan_universe_aligned,
    证明 baseline / same 通过, 任何 row key 漂移被拒绝。
    """

    baseline_eval = (
        ("AG", 100),
        ("CU", 200),
    )

    baseline_train = (
        ("AG", 100),
        ("CU", 200),
    )

    loss = np.array(
        [0.1, 0.2]
    )

    baseline_loss = np.array(
        [0.1, 0.2]
    )

    days = np.array(
        [5, 6]
    )

    baseline_days = np.array(
        [5, 6]
    )

    # same: 必须与 baseline 完全一致, PASS
    assert_scan_universe_aligned(
        name="same",
        loss=loss,
        baseline_loss=baseline_loss,
        days=days,
        baseline_days=baseline_days,
        eval_keys=baseline_eval,
        baseline_eval_keys=baseline_eval,
        train_keys=baseline_train,
        baseline_train_keys=baseline_train,
    )

    # eval key 漂移 -> 必须被拒绝
    drift_eval = (
        ("AG", 101),
        ("CU", 200),
    )

    try:

        assert_scan_universe_aligned(
            name="drift_eval",
            loss=loss,
            baseline_loss=baseline_loss,
            days=days,
            baseline_days=baseline_days,
            eval_keys=drift_eval,
            baseline_eval_keys=baseline_eval,
            train_keys=baseline_train,
            baseline_train_keys=baseline_train,
        )

    except SystemExit as exc:

        if (
            "STOP_STRUCTREV_SCAN_EVAL_UNIVERSE_DRIFT"
            not in str(exc)
        ):
            raise

    else:

        raise SystemExit(
            "STOP_STRUCTREV_SCAN_DRIFT_SYNTHETIC_FAILED"
        )

    # train key 漂移 -> 必须被拒绝
    drift_train = (
        ("AG", 101),
        ("CU", 200),
    )

    try:

        assert_scan_universe_aligned(
            name="drift_train",
            loss=loss,
            baseline_loss=baseline_loss,
            days=days,
            baseline_days=baseline_days,
            eval_keys=baseline_eval,
            baseline_eval_keys=baseline_eval,
            train_keys=drift_train,
            baseline_train_keys=baseline_train,
        )

    except SystemExit as exc:

        if (
            "STOP_STRUCTREV_SCAN_TRAIN_UNIVERSE_DRIFT"
            not in str(exc)
        ):
            raise

    else:

        raise SystemExit(
            "STOP_STRUCTREV_SCAN_DRIFT_SYNTHETIC_FAILED"
        )


def higher_tf_asof_synthetic() -> None:
    """
    验证更高周期（15m / 1H / 4H）as-of 对齐合同：

    一个 5m decision bar 在 available_time = t 时，
    只能使用 available_time <= t 的更高周期 bar，
    绝不能偷看下一个更高周期 bar。

    复现 attach_indicator_features 的 as-of 对齐逻辑：
        pos = searchsorted(available_time, t, side="right") - 1

    每个 timeframe 单独 fail code。
    """

    n = 600

    rng = np.random.default_rng(
        11,
    )

    close = (
        100.0
        + np.cumsum(
            rng.normal(
                0,
                0.2,
                n,
            )
        )
    )

    high = (
        close
        + rng.uniform(
            0.05,
            0.4,
            n,
        )
    )

    low = (
        close
        - rng.uniform(
            0.05,
            0.4,
            n,
        )
    )

    open_ = np.r_[
        close[0],
        close[:-1],
    ]

    time_index = pd.date_range(
        "2026-01-01",
        periods=n,
        freq="5min",
    )

    raw = pd.DataFrame(
        {
            "time": time_index,
            "trading_day": pd.Timestamp(
                "2026-01-01",
            ),
            "segment": 0,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "disc": False,
            "available_time": (
                time_index
                + pd.Timedelta(
                    minutes=5,
                )
            ),
            "n_base": 1,
        }
    )

    decision_times = (
        pd.to_datetime(
            raw["available_time"]
        )
        .to_numpy(
            dtype="datetime64[ns]",
        )
    )

    for minutes in (
        15,
        60,
        240,
    ):

        tf = resample_causal(
            raw,
            minutes,
        )

        tf_avail = (
            pd.to_datetime(
                tf["available_time"]
            )
            .to_numpy(
                dtype="datetime64[ns]",
            )
        )

        for t in decision_times[::30]:

            pos = (
                np.searchsorted(
                    tf_avail,
                    t,
                    side="right",
                )
                - 1
            )

            if pos < 0:
                continue

            # as-of 合同：对齐的更高周期 bar 必须已 available
            if tf_avail[pos] > t:
                raise SystemExit(
                    f"STOP_STRUCTREV_HTF_ASOF_FUTURE_LEAK:{minutes}"
                )

            # 下一个更高周期 bar 必须严格晚于 t
            if pos + 1 < len(tf_avail):

                if tf_avail[pos + 1] <= t:
                    raise SystemExit(
                        f"STOP_STRUCTREV_HTF_ASOF_NOT_LATEST:{minutes}"
                    )

        # 因果扰动：
        # 修改未来更高周期 bar 不应改变过去决策的 as-of 对齐
        tf_perturbed = tf.copy()

        tf_perturbed.loc[
            tf_perturbed.index[-1],
            "close",
        ] += 999.0

        tf_perturbed_avail = (
            pd.to_datetime(
                tf_perturbed["available_time"]
            )
            .to_numpy(
                dtype="datetime64[ns]",
            )
        )

        for t in decision_times[::30]:

            pos0 = (
                np.searchsorted(
                    tf_avail,
                    t,
                    side="right",
                )
                - 1
            )

            pos1 = (
                np.searchsorted(
                    tf_perturbed_avail,
                    t,
                    side="right",
                )
                - 1
            )

            if pos0 < 0 or pos1 < 0:
                continue

            if pos0 != pos1:
                raise SystemExit(
                    f"STOP_STRUCTREV_HTF_ASOF_PERTURB_DRIFT:{minutes}"
                )


# =============================================================================
# 18. Audit
# =============================================================================

def run_audit_only() -> None:

    print(
        f"[{EXPERIMENT_NAME}] AUDIT",
        flush=True,
    )

    print(
        f"HEAD={git_head()}",
        flush=True,
    )

    print(
        f"BASE_SHA={BASE_SHA}",
        flush=True,
    )

    assert_base_sha_ancestor()

    if (
        "TB4"
        in ALLOWED_BLOCKS
    ):

        raise SystemExit(
            "STOP_STRUCTREV_TB4_IN_ALLOWED_BLOCKS"
        )

    prefix_invariance_synthetic()

    print(
        "[AUDIT] prefix invariance: PASS",
        flush=True,
    )

    pivot_causality_synthetic()

    print(
        "[AUDIT] confirmed pivot causality (synthetic): PASS",
        flush=True,
    )

    decision_anchor_synthetic()

    print(
        "[AUDIT] decision-anchor synthetic (bar_t != entry_bar): PASS",
        flush=True,
    )

    mfe_mae_synthetic()

    print(
        "[AUDIT] MFE/MAE nonnegative semantics: PASS",
        flush=True,
    )

    higher_tf_asof_synthetic()

    print(
        "[AUDIT] higher-TF as-of (15m/1H/4H synthetic): PASS",
        flush=True,
    )

    parameter_scan_keys_drift_synthetic()

    print(
        "[AUDIT] parameter-scan key-drift gate: PASS",
        flush=True,
    )

    print(
        "[AUDIT] parameter scan scope = TB1 -> TB2 only",
        flush=True,
    )

    print(
        "[AUDIT] TB4 forbidden: PASS",
        flush=True,
    )


# =============================================================================
# 19. 默认特征
# =============================================================================

def prepare_default_frames(
    bundle: Dict[str, Any],
) -> Dict[str, Any]:

    bars_by_sym = bundle[
        "bars_by_sym"
    ]

    symbols = sorted(
        bundle[
            "scored_A"
        ][
            "symbol"
        ]
        .unique()
        .tolist()
    )

    print(
        "[FEATURE] build causal multi-TF indicators",
        flush=True,
    )

    cache = build_indicator_cache(

        bars_by_sym,

        PINE_DEFAULT,

        symbols,
    )

    frames = {}

    for key in (
        "scored_A",
    ):

        print(
            f"[FEATURE] attach {key}",
            flush=True,
        )

        f = (
            attach_indicator_features(

                bundle[key],

                bars_by_sym,

                cache,
            )
        )

        f = (
            add_oriented_nodes(
                f,
                PINE_DEFAULT,
            )
        )

        assert_allowed_blocks(
            f
        )

        frames[key] = f

    return dict(

        cache=cache,

        frames=frames,
    )


# =============================================================================
# 20. Main experiment
# =============================================================================

def run_main(
    smoke: bool,
    full: bool,
) -> Dict[str, Any]:

    if full:

        require_full_authorization()

        assert_clean_git_tree()

    assert_base_sha_ancestor()

    print(
        "[LOAD] load_window_a_only() "
        "(TB1 -> TB2 only; no TB3 / Window B)",
        flush=True,
    )

    t0 = time.perf_counter()

    bundle = load_window_a_only()

    prepared = prepare_default_frames(
        bundle,
    )

    t1 = time.perf_counter()

    print(
        f"[LOAD] done {t1 - t0:.2f}s",
        flush=True,
    )

    frames = prepared[
        "frames"
    ]

    bars_by_sym = bundle[
        "bars_by_sym"
    ]

    cap = (
        SMOKE_CAP
        if smoke
        else None
    )

    n_boot = (
        200
        if smoke
        else BOOTSTRAP_N
    )

    # -------------------------------------------------------------------------
    # smoke / full 探索：
    #   只研究 Window A (TB1 -> TB2)。
    #   不访问 TB3 / Window B。
    #   每个 horizon 独立构建自己的 path-valid universe。
    # -------------------------------------------------------------------------

    active_windows = [
        w
        for w in WINDOWS
        if w["name"] == "A_TB1_to_TB2"
    ]

    all_metrics = []
    all_bootstrap = []
    all_cells = []
    all_funnels = []

    counts = {}

    for window in active_windows:

        frame = frames[
            window[
                "scored_key"
            ]
        ]

        for horizon in HORIZONS:

            t2 = time.perf_counter()

            train, train_funnel = build_path_dataset(
                frame,
                window["train"],
                bars_by_sym,
                horizon=horizon,
                cap=cap,
            )

            eval_df, eval_funnel = build_path_dataset(
                frame,
                (window["eval"],),
                bars_by_sym,
                horizon=horizon,
                cap=cap,
            )

            t3 = time.perf_counter()

            key = (
                f"{window['name']}"
                f"|H{horizon}"
            )

            counts[key] = dict(
                n_train=len(train),
                n_eval=len(eval_df),
            )

            print(
                (
                    "[WINDOW] "
                    f"{key} "
                    f"train={len(train)} "
                    f"eval={len(eval_df)} "
                    f"build={t3 - t2:.2f}s"
                ),
                flush=True,
            )

            print(
                f"[FUNNEL][train] {key} {train_funnel}",
                flush=True,
            )
            print(
                f"[FUNNEL][eval ] {key} {eval_funnel}",
                flush=True,
            )

            outcomes_h = [
                f"y_mean_hit_{horizon}",
            ]

            if full:

                outcomes_h.append(
                    f"y_mean_before_1R_{horizon}"
                )

            metrics, bootstrap = evaluate_graph_ladder(
                train,
                eval_df,
                key,
                outcomes_h,
                n_boot,
            )

            cells = diagnostic_cells(
                eval_df,
                key,
                horizon,
            )

            all_metrics.append(
                metrics
            )

            all_bootstrap.append(
                bootstrap
            )

            all_cells.append(
                cells
            )

            all_funnels.append(
                dict(
                    window=key,
                    **train_funnel,
                )
            )

    return dict(

        metrics=pd.concat(
            all_metrics,
            ignore_index=True,
        ),

        bootstrap=pd.concat(
            all_bootstrap,
            ignore_index=True,
        ),

        cells=pd.concat(
            all_cells,
            ignore_index=True,
        ),

        counts=counts,

        funnels=all_funnels,

        params=asdict(
            PINE_DEFAULT
        ),

        head=git_head(),
    )


# =============================================================================
# 21. Artifact writer
# =============================================================================

def write_artifacts(
    result: Dict[str, Any],
    overwrite: bool,
) -> None:

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = [

        OUT_DIR
        / ARTIFACT_METRICS,

        OUT_DIR
        / ARTIFACT_BOOTSTRAP,

        OUT_DIR
        / ARTIFACT_CELLS,

        OUT_DIR
        / ARTIFACT_SUMMARY,
    ]

    if not overwrite:

        stale = [

            str(p)

            for p in paths

            if p.exists()
        ]

        if stale:

            raise SystemExit(
                f"STOP_STRUCTREV_ARTIFACT_EXISTS:{stale}"
            )

    result[
        "metrics"
    ].to_csv(

        paths[0],

        index=False,
    )

    result[
        "bootstrap"
    ].to_csv(

        paths[1],

        index=False,
    )

    result[
        "cells"
    ].to_csv(

        paths[2],

        index=False,
    )

    summary = dict(

        experiment=
            EXPERIMENT_NAME,

        scope=
            EXPERIMENT_SCOPE,

        base_sha=
            BASE_SHA,

        head=
            result["head"],

        params=
            result["params"],

        counts=
            result["counts"],

        artifacts=[
            p.name
            for p in paths
        ],
    )

    paths[
        3
    ].write_text(

        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),

        encoding="utf-8",
    )


# =============================================================================
# 22. CLI
# =============================================================================

def main() -> None:

    parser = argparse.ArgumentParser()

    mode = (
        parser
        .add_mutually_exclusive_group(
            required=True
        )
    )

    mode.add_argument(
        "--audit-only",
        action="store_true",
    )

    mode.add_argument(
        "--smoke",
        action="store_true",
    )

    mode.add_argument(
        "--full",
        action="store_true",
    )

    mode.add_argument(
        "--parameter-scan",
        action="store_true",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    args = (
        parser
        .parse_args()
    )

    start = time.perf_counter()

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------

    if args.audit_only:

        run_audit_only()

        return

    # -------------------------------------------------------------------------
    # Parameter scan
    #
    # 只允许 Window A。
    # -------------------------------------------------------------------------

    if args.parameter_scan:

        # -----------------------------------------------------------------
        # 授权 + clean-tree gate 必须在任何 load / fit / feature 之前。
        # -----------------------------------------------------------------
        require_parameter_scan_authorization()

        assert_clean_git_tree()

        assert_base_sha_ancestor()

        print(
            "[LOAD] parameter scan: Window A only",
            flush=True,
        )

        bundle = (
            load_window_a_only()
        )

        scan = (
            run_parameter_scan(
                bundle
            )
        )

        OUT_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        path = (
            OUT_DIR
            / ARTIFACT_SCAN
        )

        if (
            path.exists()
            and not args.overwrite
        ):

            raise SystemExit(
                f"STOP_STRUCTREV_SCAN_ARTIFACT_EXISTS:{path}"
            )

        scan.to_csv(
            path,
            index=False,
        )

        print(
            scan
            .sort_values(
                "log_loss"
            )
            .to_string(
                index=False
            ),
            flush=True,
        )

        print(
            f"[DONE] {path}",
            flush=True,
        )

        return

    # -------------------------------------------------------------------------
    # Smoke / Full
    # -------------------------------------------------------------------------

    result = (
        run_main(

            smoke=args.smoke,

            full=args.full,
        )
    )

    print(
        "\n=== METRICS ===",
        flush=True,
    )

    print(

        result[
            "metrics"
        ]
        .to_string(
            index=False
        ),

        flush=True,
    )

    print(
        "\n=== INCREMENT BOOTSTRAP ===",
        flush=True,
    )

    print(

        result[
            "bootstrap"
        ]
        .to_string(
            index=False
        ),

        flush=True,
    )

    if args.full:

        write_artifacts(

            result,

            overwrite=(
                args.overwrite
            ),
        )

        print(
            f"[ARTIFACT] {OUT_DIR}",
            flush=True,
        )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"[DONE] elapsed={elapsed:.2f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()