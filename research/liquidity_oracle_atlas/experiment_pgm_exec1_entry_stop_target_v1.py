"""
experiment_pgm_exec1_entry_stop_target_v1.py

PGM-EXEC-1: 在已经冻结的 PGM continuation signal 上研究执行层：
PGM direction -> pullback entry -> fixed stop -> target / timeout

唯一科学问题：
PGM 已存在的 continuation edge，能否通过 entry + risk/reward 转化成正净收益？

本模块实现完整实验框架：
- 未来路径张量（Future Path Tensor, N x 8, 完全向量化）
- MAE / MFE 数学模型与阈值分布
- 125 个执行策略（5 entry x 5 stop x 5 target）向量化评估
- TB2 二维参数邻域稳健选择（Primary 与 Stop-Only Secondary）
- TB3 严格冻结样本验证（仅 BASE / PRIMARY / STOP_ONLY_SECONDARY）
- 基于 entry_day 的日聚合 Paired Bootstrap
- Fail-closed Formal Artifacts 写入与磁盘重读双向对齐校验
"""
from __future__ import annotations

import argparse
import ast
import inspect
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 复用既有所有者（禁止重新发明或复制代码）
# ---------------------------------------------------------------------------
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0d_acceleration_terminal_outcome_v1 as d0
import research.liquidity_oracle_atlas.experiment_pgm_native0e_consensus_acceleration_v1 as e0

# ---------------------------------------------------------------------------
# 0. 治理常量 (Governance Constants)
# ---------------------------------------------------------------------------
BASE_SHA = "cc4544ec8a1cae03def67252df3e2b4c1d4fdf8e"

EXPERIMENT_NAME = "PGM-EXEC-1 -- Entry / MAE-MFE / Stop-Target Surface"
EXPERIMENT_SCOPE = "EXPLORATORY_PGM_EXECUTION_RESEARCH_ON_TB1_TB2_TB3"

ALLOWED_BLOCKS = ["TB1", "TB2", "TB3"]
TB2_BLOCK = "TB2"
TB3_BLOCK = "TB3"

ENTRY_GRID = np.array([0.00, 0.05, 0.10, 0.15, 0.20], dtype=float)
STOP_GRID = np.array([0.10, 0.15, 0.20, 0.25, 0.30], dtype=float)
TARGET_CODES = ["1.0R", "1.5R", "2.0R", "3.0R", "NONE"]
TARGET_MULTIPLES: Dict[str, Optional[float]] = {
    "1.0R": 1.0,
    "1.5R": 1.5,
    "2.0R": 2.0,
    "3.0R": 3.0,
    "NONE": None,
}

CORE_ENTRY = {0.00, 0.05, 0.10, 0.15}
CORE_STOP = {0.15, 0.20, 0.25}
CORE_TARGET = {"1.5R", "2.0R", "NONE"}

ENTRY_WAIT_BARS = 3
MAX_HOLD_BARS = 6
# 3 + 6 - 1
FUTURE_BARS = 8

MAX_GAP_MINUTES = 10
PRIMARY_COST_ATR0 = 0.01

BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260916

SMOKE_BOOTSTRAP_N = 200
SMOKE_EVAL_CAP = 4096

MIN_SELECTION_TRADES = 5000
MIN_POSITIVE_NEIGHBOR_RATE = 0.70

MAE_MFE_HORIZONS = [1, 3, 6]
MAE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
MFE_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.60]

# 退出原因编码
NO_FILL = 0
STOP = 1
TARGET = 2
TIMEOUT = 3

# 文件路径与 Artifacts 前缀
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = _REPO_ROOT / "artifacts" / "liquidity_oracle_atlas"

ARTIFACT_PREFIX = "pgm_exec1_"
ARTIFACT_MAE_MFE_QUANTILES = "pgm_exec1_mae_mfe_quantiles.csv"
ARTIFACT_MAE_MFE_THRESHOLDS = "pgm_exec1_mae_mfe_thresholds.csv"
ARTIFACT_ENTRY_ONLY = "pgm_exec1_entry_only.csv"
ARTIFACT_SURFACE_TB2 = "pgm_exec1_surface_tb2.csv"
ARTIFACT_SELECTED_POLICIES = "pgm_exec1_selected_policies.csv"
ARTIFACT_TB3_VALIDATION = "pgm_exec1_tb3_validation.csv"
ARTIFACT_BOOTSTRAP = "pgm_exec1_bootstrap.csv"
ARTIFACT_FORMAL_SUMMARY = "pgm_exec1_formal_summary.json"

EXACT_ARTIFACTS = [
    ARTIFACT_MAE_MFE_QUANTILES,
    ARTIFACT_MAE_MFE_THRESHOLDS,
    ARTIFACT_ENTRY_ONLY,
    ARTIFACT_SURFACE_TB2,
    ARTIFACT_SELECTED_POLICIES,
    ARTIFACT_TB3_VALIDATION,
    ARTIFACT_BOOTSTRAP,
    ARTIFACT_FORMAL_SUMMARY,
]


# ===========================================================================
# 1. 治理检查与准入规则 (Governance Gates)
# ===========================================================================
def assert_allowed_blocks(df: pd.DataFrame) -> None:
    """严格禁止 TB4 及未许可 block。"""
    blocks = df["block"].unique()
    if "TB4" in blocks:
        raise SystemExit("STOP_PGM_EXEC1_FORBIDDEN_TB4")
    for b in blocks:
        if b not in ALLOWED_BLOCKS:
            raise SystemExit(f"STOP_PGM_EXEC1_FORBIDDEN_BLOCK: {b}")


def require_full_authorization() -> None:
    """正式 Full 运行必须且仅能通过环境变量授权。"""
    token = os.environ.get("AUTHORIZE_PGM_EXEC1_FULL_EXPLORATORY", "").strip()
    if token != "1":
        raise SystemExit("STOP_PGM_EXEC1_FULL_NOT_AUTHORIZED")


def assert_clean_git_tree() -> str:
    """检查 tracked tree 与 staging 必须处于 clean 状态，返回当前 HEAD SHA。"""
    diff_res = subprocess.run(
        ["git", "diff", "--exit-code"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )
    if diff_res.returncode != 0:
        raise SystemExit("STOP_PGM_EXEC1_GIT_TREE_NOT_CLEAN")

    cached_res = subprocess.run(
        ["git", "diff", "--cached", "--exit-code"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )
    if cached_res.returncode != 0:
        raise SystemExit("STOP_PGM_EXEC1_GIT_STAGING_NOT_CLEAN")

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(_REPO_ROOT),
        text=True,
    ).strip()
    return head


def assert_no_stale_artifacts() -> None:
    """Formal 运行前，artifacts 目录中不得存在任何 pgm_exec1_* 文件。"""
    stale = sorted(list(OUT_DIR.glob(f"{ARTIFACT_PREFIX}*")))
    if len(stale) > 0:
        stale_names = [f.name for f in stale]
        raise SystemExit(f"STOP_PGM_EXEC1_ARTIFACT_ALREADY_EXISTS: {stale_names}")


# ATR0 所有者对齐审计的输入契约
ATR0_AUDIT_KEY_COLS = ["symbol", "episode_id", "bar_t"]
ATR0_AUDIT_REQUIRED_BLOCKS = ["TB1", "TB2", "TB3"]


def assert_atr0_audit_universe(obs: pd.DataFrame, cur_truth: pd.DataFrame) -> None:
    """硬性契约：ATR0 所有者对齐审计的输入必须是**全量决策宇宙**。

    该 gate 校验的命题是「cur_truth 中每一条 hazard==0 记录都必须在 obs 中出现且 atr0 一致」，
    因此传入 block 子样本（例如某个 window 的 TB2 path-valid 评测子集）时：
    覆盖的键少于 truth，却仍会走完 inner merge 并返回一个看起来正常的 max_err ——
    这正是「比较了不同 universe 却报告 PASS」的最危险形态。

    因此任何不满足以下条件的输入必须 fail-closed：

    1. 必需列存在（obs: symbol / episode_id / bar_t / atr0 / hazard / block）
    2. obs 的 block 覆盖 TB1 / TB2 / TB3
    3. (symbol, episode_id, bar_t) 在 obs[hazard==0] 与 cur_truth 两侧均唯一
    4. obs[hazard==0] 的键集合与 cur_truth 的键集合完全相等
    5. len(obs[hazard==0]) == len(cur_truth)

    禁止用 block 子样本、path-valid 子样本或任何 derivative 帧替代本参数。
    """
    missing_obs = [
        c for c in ATR0_AUDIT_KEY_COLS + ["atr0", "hazard", "block"] if c not in obs.columns
    ]
    if missing_obs:
        raise SystemExit(
            f"STOP_PGM_EXEC1_ATR0_AUDIT_UNIVERSE_INVALID: obs_missing_columns={missing_obs}"
        )
    missing_truth = [c for c in ATR0_AUDIT_KEY_COLS if c not in cur_truth.columns]
    if missing_truth:
        raise SystemExit(
            f"STOP_PGM_EXEC1_ATR0_AUDIT_UNIVERSE_INVALID: cur_truth_missing_columns={missing_truth}"
        )

    blocks = sorted({str(b) for b in obs["block"].unique().tolist()})
    if not set(ATR0_AUDIT_REQUIRED_BLOCKS).issubset(set(blocks)):
        raise SystemExit(
            "STOP_PGM_EXEC1_ATR0_AUDIT_UNIVERSE_INVALID: "
            f"block_coverage={blocks} does not cover {ATR0_AUDIT_REQUIRED_BLOCKS}"
        )

    obs_h0 = obs.loc[obs["hazard"] == 0, ATR0_AUDIT_KEY_COLS]
    truth_keys_df = cur_truth[ATR0_AUDIT_KEY_COLS]
    if bool(obs_h0.duplicated().any()):
        raise SystemExit("STOP_PGM_EXEC1_ATR0_AUDIT_UNIVERSE_INVALID: obs_hazard0_keys_not_unique")
    if bool(truth_keys_df.duplicated().any()):
        raise SystemExit("STOP_PGM_EXEC1_ATR0_AUDIT_UNIVERSE_INVALID: cur_truth_keys_not_unique")

    obs_keys = set(map(tuple, obs_h0.to_numpy()))
    truth_keys = set(map(tuple, truth_keys_df.to_numpy()))
    if len(obs_h0) != len(cur_truth) or obs_keys != truth_keys:
        raise SystemExit(
            "STOP_PGM_EXEC1_ATR0_AUDIT_UNIVERSE_INVALID: "
            f"obs_hazard0_len={len(obs_h0)} cur_truth_len={len(cur_truth)} "
            f"missing_keys={len(truth_keys - obs_keys)} extra_keys={len(obs_keys - truth_keys)}"
        )


# ===========================================================================
# 2. 数据加载与 Window 评分（全局仅 Fit 一次）
# ===========================================================================
def load_and_score() -> Dict[str, Any]:
    """单次加载并评分 Window A 与 Window B。

    严格遵循 contract：
    - Window A fit 一次 -> 评分 aligned 得到 scored_A (所有权：TB1, TB2)
    - Window B fit 一次 -> 评分 aligned 得到 scored_B (所有权：TB1, TB2, TB3)
    禁止在后续 surface、selection 或 validation 过程中重复 fit。
    """
    prep = d0.load_prepared_frame()
    aligned = prep["aligned"]
    bars_by_sym = prep["bars_by_sym"]
    assert_allowed_blocks(aligned)

    # Window A (覆盖 TB1, TB2)
    fit_A = pgm.fit_samplers_for_window(
        pgm.WINDOWS[0],
        pgm.SAMPLE_PATH,
        pgm.TRANSITION_SAMPLE_PATH,
    )
    scored_A = d0.prepare_window_windowframe(aligned, fit_A, "pgm_exec1_A")
    d0.verify_window_score_owner(scored_A, fit_A, "TB1")
    d0.verify_window_score_owner(scored_A, fit_A, "TB2")

    # Window B (覆盖 TB1, TB2, TB3)
    fit_B = pgm.fit_samplers_for_window(
        pgm.WINDOWS[1],
        pgm.SAMPLE_PATH,
        pgm.TRANSITION_SAMPLE_PATH,
    )
    scored_B = d0.prepare_window_windowframe(aligned, fit_B, "pgm_exec1_B")
    d0.verify_window_score_owner(scored_B, fit_B, "TB1")
    d0.verify_window_score_owner(scored_B, fit_B, "TB2")
    d0.verify_window_score_owner(scored_B, fit_B, "TB3")

    return dict(
        prep=prep,
        aligned=aligned,
        bars_by_sym=bars_by_sym,
        fit_A=fit_A,
        scored_A=scored_A,
        fit_B=fit_B,
        scored_B=scored_B,
    )


def extract_evaluation_sample(scored: pd.DataFrame, eval_block: str) -> pd.DataFrame:
    """提取评测宇宙基础子样本。

    基础过滤条件：
    - block == eval_block
    - same_block_entry_valid == True
    - base_action != 0
    - np.isfinite(atr0)
    - np.isfinite(score_mu)
    """
    assert_allowed_blocks(scored)
    base_mask = (
        (scored["block"] == eval_block)
        & scored["same_block_entry_valid"].to_numpy(bool)
        & (scored["base_action"].to_numpy(float) != 0.0)
        & np.isfinite(scored["atr0"].to_numpy(float))
        & np.isfinite(scored["score_mu"].to_numpy(float))
    )
    sub = scored[base_mask].copy().reset_index(drop=True)
    return sub


# ===========================================================================
# 3. 未来路径张量构建 (Future Path Tensor - 纯向量化，禁止逐行 Python loop)
# ===========================================================================
def build_future_tensor(
    eval_df: pd.DataFrame,
    bars_by_sym: Dict[str, Dict[str, Any]],
    n_future: int = FUTURE_BARS,
) -> Dict[str, Any]:
    """构建未来 n_future=8 根 K 线的未来路径张量。

    严格遵循效率与因果合同：
    - 禁止随 N 增长的 Python 循环，仅允许在 15 个固定 symbol 上做向量分块检索
    - 每个 symbol 使用 NumPy 高级索引一次性提取未来 8 根 K 线
    - 检查未来路径有效性：
      1. 索引不越界 (t >= 0 且 t+8 <= n_bars)
      2. 不跨越 discontinuity: ~DISC.any(axis=1)
      3. 相邻 bar 时间连续: [t, t+1, ..., t+8] 共 8 个步长，0 < Delta t_j <= 10 min
      4. 同 trading_day: 未来 8 根 DAY[:, j] == DAY[:, 0]
    """
    N = len(eval_df)
    O = np.full((N, n_future), np.nan, dtype=np.float64)
    H = np.full((N, n_future), np.nan, dtype=np.float64)
    L = np.full((N, n_future), np.nan, dtype=np.float64)
    C = np.full((N, n_future), np.nan, dtype=np.float64)
    T = np.full((N, n_future), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    DAY = np.full((N, n_future), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    DISC = np.zeros((N, n_future), dtype=bool)

    direction = eval_df["base_action"].to_numpy(np.float64)
    atr0 = eval_df["atr0"].to_numpy(np.float64)
    hazard = eval_df["hazard"].to_numpy(np.int64)
    entry_day = pd.to_datetime(eval_df["entry_day"]).to_numpy()
    symbol = eval_df["symbol"].to_numpy()

    path_valid = np.zeros(N, dtype=bool)
    disc_excluded = np.zeros(N, dtype=bool)
    gap_excluded = np.zeros(N, dtype=bool)
    day_excluded = np.zeros(N, dtype=bool)

    offset_8 = np.arange(n_future, dtype=int)[None, :]  # shape: (1, 8)
    offset_9 = np.arange(-1, n_future, dtype=int)[None, :]  # shape: (1, 9), 从 bar t 到 t+8

    symbols = np.unique(symbol)
    for sym in symbols:
        sym_mask = symbol == sym
        row_pos = np.flatnonzero(sym_mask)
        if len(row_pos) == 0:
            continue
        if sym not in bars_by_sym:
            raise SystemExit(f"STOP_PGM_EXEC1_MISSING_SYMBOL_BARS: {sym}")

        bars = bars_by_sym[sym]
        n_bars = int(bars["n"])
        e = eval_df.loc[eval_df.index[row_pos], "entry_bar"].to_numpy(int)

        # 检查是否越界：必须满足 e >= 1 (保证决策 bar t = e-1 >= 0) 且 e + n_future <= n_bars
        in_range = (e >= 1) & (e + n_future <= n_bars)
        if not np.any(in_range):
            continue

        valid_row_pos = row_pos[in_range]
        e_valid = e[in_range]

        # 高级索引 (N_sub, 8) 与 (N_sub, 9)
        idx_8 = e_valid[:, None] + offset_8
        idx_9 = e_valid[:, None] + offset_9

        o_sub = bars["o"][idx_8]
        h_sub = bars["h"][idx_8]
        l_sub = bars["l"][idx_8]
        c_sub = bars["c"][idx_8]
        t_sub = np.asarray(bars["t"], dtype="datetime64[ns]")[idx_8]
        day_sub = np.asarray(bars["day"], dtype="datetime64[ns]")[idx_8]
        disc_sub = bars["disc"][idx_8]

        O[valid_row_pos] = o_sub
        H[valid_row_pos] = h_sub
        L[valid_row_pos] = l_sub
        C[valid_row_pos] = c_sub
        T[valid_row_pos] = t_sub
        DAY[valid_row_pos] = day_sub
        DISC[valid_row_pos] = disc_sub

        # 7.1 不跨 discontinuity
        has_disc = disc_sub.any(axis=1)

        # 7.2 相邻 bar 时间连续：检查 [t, t+1, ..., t+8] 的 8 个间隔
        t_9 = np.asarray(bars["t"], dtype="datetime64[ns]")[idx_9]
        dt = np.diff(t_9, axis=1)  # shape: (N_sub, 8)
        max_gap = np.timedelta64(MAX_GAP_MINUTES, "m")
        gap_ok = np.all((dt > np.timedelta64(0, "ns")) & (dt <= max_gap), axis=1)

        # 7.3 同 trading_day：未来 8 根 DAY[:, j] == DAY[:, 0] 全部成立
        same_day = np.all(day_sub == day_sub[:, [0]], axis=1)

        disc_excluded[valid_row_pos] = has_disc
        gap_excluded[valid_row_pos] = ~gap_ok
        day_excluded[valid_row_pos] = ~same_day

        path_valid[valid_row_pos] = (~has_disc) & gap_ok & same_day

    return dict(
        O=O,
        H=H,
        L=L,
        C=C,
        T=T,
        DAY=DAY,
        DISC=DISC,
        direction=direction,
        atr0=atr0,
        hazard=hazard,
        entry_day=entry_day,
        symbol=symbol,
        path_valid=path_valid,
        disc_excluded=disc_excluded,
        gap_excluded=gap_excluded,
        day_excluded=day_excluded,
    )


def slice_future_tensor(tensor: Dict[str, Any], mask: np.ndarray) -> Dict[str, Any]:
    """沿着行维度切分未来路径张量字典。"""
    out = {}
    for k, v in tensor.items():
        if isinstance(v, np.ndarray) and len(v) == len(mask):
            out[k] = v[mask]
        else:
            out[k] = v
    return out


# ===========================================================================
# 4. 基线硬对齐门禁 (Baseline Parity Hard Gate)
# ===========================================================================
def verify_baseline_parity(tensor: Dict[str, Any], pi_expected: np.ndarray) -> float:
    """验证 common path-valid sample 上，future tensor 计算的 gross return 与 scored['pi'] 严格对齐。

    d_t * (C_{t+1} - O_{t+1}) / ATR0_t 必须与 pi 满足 max |Delta| <= 1e-12。
    """
    direction = tensor["direction"]
    atr0 = tensor["atr0"]
    o1 = tensor["O"][:, 0]
    c1 = tensor["C"][:, 0]
    baseline_gross_raw = direction * (c1 - o1) / atr0

    diff = np.abs(baseline_gross_raw - pi_expected)
    if not np.all(np.isfinite(diff)):
        raise SystemExit("STOP_PGM_EXEC1_BASELINE_PARITY_NONFINITE")
    max_err = float(np.max(diff))
    if max_err > 1e-12:
        raise SystemExit(
            f"STOP_PGM_EXEC1_BASELINE_PARITY_FAIL: max_abs_diff={max_err:.2e} > 1e-12"
        )
    return max_err


# ===========================================================================
# 5. MAE / MFE 数学模型与阈值分布
# ===========================================================================
def compute_mae_mfe(tensor: Dict[str, Any]) -> Dict[int, Dict[str, np.ndarray]]:
    """在 path-valid sample 上向量化计算 MAE 与 MFE。

    E_0 = O_{t+1}
    Favorable: 多头 High, 空头 Low
    Adverse: 多头 Low, 空头 High
    方向归一化:
    F_j = d_t * (P^{fav}_j - E_0) / ATR0_t
    A_j = d_t * (P^{adv}_j - E_0) / ATR0_t
    MFE_h = max(0, max_{1 <= j <= h} F_j)
    MAE_h = max(0, -min_{1 <= j <= h} A_j)
    """
    d = tensor["direction"][:, None]
    atr0 = tensor["atr0"][:, None]
    E0 = tensor["O"][:, 0:1]

    # 方向归一化价格极值
    p_fav = np.where(d == 1.0, tensor["H"], tensor["L"])
    p_adv = np.where(d == 1.0, tensor["L"], tensor["H"])

    F = d * (p_fav - E0) / atr0
    A = d * (p_adv - E0) / atr0

    out = {}
    for h in MAE_MFE_HORIZONS:
        F_h = F[:, :h]
        A_h = A[:, :h]
        mfe_h = np.maximum(0.0, np.max(F_h, axis=1))
        mae_h = np.maximum(0.0, -np.min(A_h, axis=1))
        out[h] = dict(mae=mae_h, mfe=mfe_h)
    return out


def summarize_mae_mfe(
    mae_mfe_by_h: Dict[int, Dict[str, np.ndarray]],
    hazard: np.ndarray,
    block_name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """生成 MAE/MFE 分位数表与阈值累积分布表。"""
    q_rows = []
    th_rows = []

    groups = {
        "ALL": np.ones(len(hazard), dtype=bool),
        "H0": (hazard == 0),
        "H1": (hazard == 1),
    }

    for grp_name, grp_mask in groups.items():
        n_grp = int(grp_mask.sum())
        if n_grp == 0:
            continue
        for h in MAE_MFE_HORIZONS:
            mae_vals = mae_mfe_by_h[h]["mae"][grp_mask]
            mfe_vals = mae_mfe_by_h[h]["mfe"][grp_mask]

            for metric_name, vals in [("MAE", mae_vals), ("MFE", mfe_vals)]:
                q25, q50, q75, q90 = np.percentile(vals, [25, 50, 75, 90])
                q_rows.append(
                    dict(
                        block=block_name,
                        group=grp_name,
                        horizon=h,
                        metric=metric_name,
                        q25=float(q25),
                        q50=float(q50),
                        q75=float(q75),
                        q90=float(q90),
                        n=n_grp,
                    )
                )

            # Thresholds
            for th in MAE_THRESHOLDS:
                prob = float(np.mean(mae_vals >= th))
                th_rows.append(
                    dict(
                        block=block_name,
                        group=grp_name,
                        horizon=h,
                        metric="MAE",
                        threshold=float(th),
                        prob_ge=prob,
                        n=n_grp,
                    )
                )
            for th in MFE_THRESHOLDS:
                prob = float(np.mean(mfe_vals >= th))
                th_rows.append(
                    dict(
                        block=block_name,
                        group=grp_name,
                        horizon=h,
                        metric="MFE",
                        threshold=float(th),
                        prob_ge=prob,
                        n=n_grp,
                    )
                )

    df_q = pd.DataFrame(q_rows)
    df_th = pd.DataFrame(th_rows)
    return df_q, df_th


# ===========================================================================
# 6. 向量化执行引擎 (Vectorized Execution Engine)
# ===========================================================================
def precompute_relative_path_for_k(
    tensor: Dict[str, Any],
    k: float,
) -> Dict[str, Any]:
    """对每个固定 entry 参数 k，预计算一次相对路径与挂单成交索引。

    E_k = O_{t+1} - d_t * k * ATR0_t
    挂单在未来前 3 根 K 线 (j=0, 1, 2) 等待成交：
    多头成交: Low_j <= E_k
    空头成交: High_j >= E_k
    若跨空穿过 limit，保守合同成交价依然为 E_k（不授予 favorable price improvement）。
    """
    d = tensor["direction"][:, None]
    atr0 = tensor["atr0"][:, None]
    O0 = tensor["O"][:, 0:1]

    # Limit 挂单入场价 (N, 1)
    Ek = O0 - d * k * atr0

    # 挂单等待前 3 根 (j=0, 1, 2)
    # 多头：Low <= Ek；空头：High >= Ek
    fill_mask = np.where(
        d == 1.0,
        tensor["L"][:, :ENTRY_WAIT_BARS] <= Ek,
        tensor["H"][:, :ENTRY_WAIT_BARS] >= Ek,
    )
    filled = fill_mask.any(axis=1)
    fill_idx = np.where(filled, fill_mask.argmax(axis=1), -1)

    p_fav = np.where(d == 1.0, tensor["H"], tensor["L"])
    p_adv = np.where(d == 1.0, tensor["L"], tensor["H"])

    open_rel = d * (tensor["O"] - Ek) / atr0
    fav_rel = d * (p_fav - Ek) / atr0
    adv_rel = d * (p_adv - Ek) / atr0
    close_rel = d * (tensor["C"] - Ek) / atr0

    return dict(
        k=k,
        Ek=Ek,
        filled=filled,
        fill_idx=fill_idx,
        open_rel=open_rel,
        fav_rel=fav_rel,
        adv_rel=adv_rel,
        close_rel=close_rel,
    )


def first_true_index(mask: np.ndarray) -> np.ndarray:
    """向量化查找每行第一个 True 的列下标，若全 False 返回 sentinel。"""
    sentinel = FUTURE_BARS + 10
    cols = np.arange(mask.shape[1], dtype=np.int32)[None, :]
    return np.where(mask, cols, sentinel).min(axis=1)


def simulate_policy(
    rel_k: Dict[str, Any],
    stop_s: Optional[float],
    target_code: str,
) -> Dict[str, Any]:
    """向量化执行单个 (k, stop_s, target_code) 策略。

    持仓期规则：
    - fill bar 为 f_i，持仓范围 j in [f_i, f_i + 5]，最多 6 bars
    - 止损规则：
      * 方向归一化止损位 -s
      * stop_gap: open_rel <= -s (承担 gap loss，收益为 open_rel)
      * stop_touch: adv_rel <= -s (intrabar 止损，收益为 -s)
    - 止盈规则：
      * Entry bar 禁止止盈 (target_active = active & ~is_entry_bar)
      * 若 target != NONE, q = m * s.
      * open_rel >= q 或 fav_rel >= q 触发，收益固定为 +q（不给 gap improvement）
    - Same-bar ambiguity:
      * STOP FIRST: 同一根 bar 同时触碰止损与止盈时，强制判定为 STOP
    - Timeout:
      * 未触发止损与止盈，在 bar f_i + 5 收盘平仓，收益为 CloseRel_{f_i+5}
    """
    filled = rel_k["filled"]
    fill_idx = rel_k["fill_idx"]
    open_rel = rel_k["open_rel"]
    fav_rel = rel_k["fav_rel"]
    adv_rel = rel_k["adv_rel"]
    close_rel = rel_k["close_rel"]

    N = len(filled)
    sentinel = FUTURE_BARS + 10
    j = np.arange(FUTURE_BARS, dtype=np.int32)[None, :]

    active = (
        filled[:, None]
        & (j >= fill_idx[:, None])
        & (j <= fill_idx[:, None] + (MAX_HOLD_BARS - 1))
    )
    is_entry_bar = j == fill_idx[:, None]

    # 1. 止损触发检测
    if stop_s is not None and np.isfinite(stop_s):
        stop_gap = active & (open_rel <= -stop_s)
        stop_touch = active & (adv_rel <= -stop_s)
        stop_hit = stop_gap | stop_touch
        stop_idx = first_true_index(stop_hit)
    else:
        stop_idx = np.full(N, sentinel, dtype=np.int32)

    # 2. 止盈触发检测
    target_mult = TARGET_MULTIPLES.get(target_code)
    if target_mult is not None and stop_s is not None and np.isfinite(stop_s):
        q = target_mult * stop_s
        target_active = active & (~is_entry_bar)
        target_hit = target_active & ((open_rel >= q) | (fav_rel >= q))
        target_idx = first_true_index(target_hit)
    else:
        q = None
        target_idx = np.full(N, sentinel, dtype=np.int32)

    # 3. 超时索引
    timeout_idx = np.where(filled, fill_idx + (MAX_HOLD_BARS - 1), sentinel)

    # 4. 竞争仲裁：STOP FIRST
    stop_wins = filled & (stop_idx < sentinel) & (stop_idx <= target_idx)
    target_wins = filled & (~stop_wins) & (target_idx < sentinel)
    timeout_wins = filled & (~stop_wins) & (~target_wins)

    exit_reason = np.full(N, NO_FILL, dtype=np.int32)
    exit_reason[stop_wins] = STOP
    exit_reason[target_wins] = TARGET
    exit_reason[timeout_wins] = TIMEOUT

    exit_idx = np.full(N, -1, dtype=np.int32)
    exit_idx[stop_wins] = stop_idx[stop_wins]
    exit_idx[target_wins] = target_idx[target_wins]
    exit_idx[timeout_wins] = timeout_idx[timeout_wins]

    gross_return = np.zeros(N, dtype=np.float64)

    # STOP 收益：若 open <= -stop_s 则承担 open gap loss，否则为 -stop_s
    if np.any(stop_wins):
        s_rows = np.flatnonzero(stop_wins)
        s_cols = exit_idx[s_rows]
        s_open = open_rel[s_rows, s_cols]
        gross_return[s_rows] = np.where(s_open <= -stop_s, s_open, -stop_s)

    # TARGET 收益：固定为 +q
    if np.any(target_wins):
        gross_return[target_wins] = q

    # TIMEOUT 收益：第 f_i + 5 根收盘价
    if np.any(timeout_wins):
        t_rows = np.flatnonzero(timeout_wins)
        t_cols = exit_idx[t_rows]
        gross_return[t_rows] = close_rel[t_rows, t_cols]

    holding_bars = np.where(filled, exit_idx - fill_idx + 1, 0)
    net_return = np.where(filled, gross_return - PRIMARY_COST_ATR0, 0.0)

    return dict(
        filled=filled,
        fill_idx=fill_idx,
        exit_reason=exit_reason,
        exit_idx=exit_idx,
        gross_return=gross_return,
        net_return=net_return,
        holding_bars=holding_bars,
    )


def compute_exec_metrics(
    sim: Dict[str, Any],
    tensor: Dict[str, Any],
) -> Dict[str, Any]:
    """复用 d0.strategy_metrics 并补充 EXEC-1 专属执行统计指标。"""
    filled = sim["filled"]
    action = filled.astype(np.float64)
    r_trad = sim["gross_return"]
    entry_day = tensor["entry_day"]
    symbol = tensor["symbol"]

    # 复用 0D 所有者计算基础财务指标
    m = d0.strategy_metrics(
        action=action,
        r_trad=r_trad,
        cost=PRIMARY_COST_ATR0,
        entry_day=entry_day,
        symbol=symbol,
    )

    n_dec = len(filled)
    n_tr = int(filled.sum())
    exit_reason = sim["exit_reason"]
    holding_bars = sim["holding_bars"]

    fill_rate = float(n_tr / n_dec) if n_dec > 0 else 0.0
    stop_rate = float(np.sum(exit_reason == STOP) / n_tr) if n_tr > 0 else 0.0
    target_rate = float(np.sum(exit_reason == TARGET) / n_tr) if n_tr > 0 else 0.0
    timeout_rate = float(np.sum(exit_reason == TIMEOUT) / n_tr) if n_tr > 0 else 0.0
    avg_holding = float(np.mean(holding_bars[filled])) if n_tr > 0 else 0.0

    m["fill_rate"] = fill_rate
    m["stop_rate"] = stop_rate
    m["target_rate"] = target_rate
    m["timeout_rate"] = timeout_rate
    m["avg_holding_bars"] = avg_holding

    return m


# ===========================================================================
# 7. 纯标量基准实现 (用于测试验证对照，禁止在核心热路径中使用)
# ===========================================================================
def simulate_policy_scalar_reference(
    tensor: Dict[str, Any],
    row_idx: int,
    k: float,
    stop_s: Optional[float],
    target_code: str,
) -> Dict[str, Any]:
    """慢速纯 Python 标量逻辑，用于验证向量化执行引擎数学一致性。"""
    d = float(tensor["direction"][row_idx])
    atr = float(tensor["atr0"][row_idx])
    o_future = tensor["O"][row_idx]
    h_future = tensor["H"][row_idx]
    l_future = tensor["L"][row_idx]
    c_future = tensor["C"][row_idx]

    Ek = o_future[0] - d * k * atr

    # 寻找成交
    filled = False
    fill_bar = -1
    for b in range(ENTRY_WAIT_BARS):
        if d == 1.0:
            if l_future[b] <= Ek:
                filled = True
                fill_bar = b
                break
        else:
            if h_future[b] >= Ek:
                filled = True
                fill_bar = b
                break

    if not filled:
        return dict(
            filled=False,
            fill_idx=-1,
            exit_reason=NO_FILL,
            exit_idx=-1,
            gross_return=0.0,
            net_return=0.0,
            holding_bars=0,
        )

    # 模拟持仓期
    target_mult = TARGET_MULTIPLES.get(target_code)
    q = (target_mult * stop_s) if (target_mult is not None and stop_s is not None) else None

    exit_reason = TIMEOUT
    exit_bar = fill_bar + (MAX_HOLD_BARS - 1)
    gross_ret = d * (c_future[exit_bar] - Ek) / atr

    for cur in range(fill_bar, fill_bar + MAX_HOLD_BARS):
        open_rel = d * (o_future[cur] - Ek) / atr
        p_fav = h_future[cur] if d == 1.0 else l_future[cur]
        p_adv = l_future[cur] if d == 1.0 else h_future[cur]
        fav_rel = d * (p_fav - Ek) / atr
        adv_rel = d * (p_adv - Ek) / atr

        # 止损检测
        stop_hit = False
        stop_ret = 0.0
        if stop_s is not None and np.isfinite(stop_s):
            if open_rel <= -stop_s:
                stop_hit = True
                stop_ret = open_rel  # gap loss
            elif adv_rel <= -stop_s:
                stop_hit = True
                stop_ret = -stop_s

        # 止盈检测：entry bar 禁止止盈
        target_hit = False
        if cur > fill_bar and q is not None:
            if (open_rel >= q) or (fav_rel >= q):
                target_hit = True

        # STOP FIRST 仲裁
        if stop_hit:
            exit_reason = STOP
            exit_bar = cur
            gross_ret = stop_ret
            break
        elif target_hit:
            exit_reason = TARGET
            exit_bar = cur
            gross_ret = q
            break

    holding_bars = exit_bar - fill_bar + 1
    net_ret = gross_ret - PRIMARY_COST_ATR0

    return dict(
        filled=True,
        fill_idx=fill_bar,
        exit_reason=exit_reason,
        exit_idx=exit_bar,
        gross_return=gross_ret,
        net_return=net_ret,
        holding_bars=holding_bars,
    )


# ===========================================================================
# 8. 125 策略全空间表面评估 (仅允许在 TB2 上运行)
# ===========================================================================
def is_core_policy(k: float, stop_s: float, target_code: str) -> bool:
    """判定是否为核心候选策略 (4 x 3 x 3 = 36)。"""
    k_ok = any(abs(k - ck) < 1e-5 for ck in CORE_ENTRY)
    s_ok = any(abs(stop_s - cs) < 1e-5 for cs in CORE_STOP)
    t_ok = target_code in CORE_TARGET
    return bool(k_ok and s_ok and t_ok)


def is_stop_only_core(k: float, stop_s: float, target_code: str) -> bool:
    """判定是否为仅止损核心候选策略 (4 x 3 = 12)。"""
    if target_code != "NONE":
        return False
    k_ok = any(abs(k - ck) < 1e-5 for ck in CORE_ENTRY)
    s_ok = any(abs(stop_s - cs) < 1e-5 for cs in CORE_STOP)
    return bool(k_ok and s_ok)


def evaluate_surface(
    tensor: Dict[str, Any],
    eval_block: str = TB2_BLOCK,
) -> pd.DataFrame:
    """在评测张量上评估 125 个策略的全空间表面。

    严格禁止在 TB3 上调用本函数。
    计算：
    - 同一 k 下复用相对路径
    - 统计 125 个策略的全部指标
    - 计算每个候选策略在二维参数空间 (stop +/- 1, target +/- 1) 内的稳健指标
    """
    if eval_block != TB2_BLOCK:
        raise SystemExit(
            f"STOP_PGM_EXEC1_SURFACE_FORBIDDEN_ON_BLOCK: {eval_block} != {TB2_BLOCK}"
        )

    n_k = len(ENTRY_GRID)
    n_s = len(STOP_GRID)
    n_t = len(TARGET_CODES)
    EV_grid = np.zeros((n_k, n_s, n_t), dtype=np.float64)
    matrix = [[[None for _ in range(n_t)] for _ in range(n_s)] for _ in range(n_k)]

    # 5 个 entry 循环
    for k_idx, k in enumerate(ENTRY_GRID):
        rel_k = precompute_relative_path_for_k(tensor, k)
        # 5 个 stop 循环
        for s_idx, s in enumerate(STOP_GRID):
            # 5 个 target 循环
            for t_idx, target in enumerate(TARGET_CODES):
                sim = simulate_policy(rel_k, s, target)
                metrics = compute_exec_metrics(sim, tensor)

                t_mult = TARGET_MULTIPLES[target]
                core_flag = is_core_policy(k, s, target)
                net_ev = metrics["net_EV_per_decision"]
                EV_grid[k_idx, s_idx, t_idx] = net_ev

                row = dict(
                    k=float(k),
                    stop_s=float(s),
                    target_code=target,
                    target_multiple=(float(t_mult) if t_mult is not None else np.nan),
                    n_decisions=metrics["n_decisions"],
                    n_trades=metrics["n_trades"],
                    trade_rate=metrics["trade_rate"],
                    fill_rate=metrics["fill_rate"],
                    gross_total_ATR0=metrics["gross_total_ATR0"],
                    net_total_ATR0=metrics["net_total_ATR0"],
                    gross_EV_per_decision=metrics["gross_EV_per_decision"],
                    net_EV_per_decision=net_ev,
                    net_EV_per_trade=metrics["net_EV_per_trade"],
                    win_rate=metrics["win_rate"],
                    mean_win=metrics["mean_win"],
                    mean_loss=metrics["mean_loss"],
                    payoff_ratio=metrics["payoff_ratio"],
                    profit_factor=metrics["profit_factor"],
                    break_even_cost=metrics["break_even_cost"],
                    daily_sharpe_annualized=metrics["daily_sharpe_annualized"],
                    max_drawdown_ATR0=metrics["max_drawdown_ATR0"],
                    positive_symbol_count=metrics["positive_symbol_count"],
                    top3_profit_share=metrics["top3_profit_share"],
                    stop_rate=metrics["stop_rate"],
                    target_rate=metrics["target_rate"],
                    timeout_rate=metrics["timeout_rate"],
                    avg_holding_bars=metrics["avg_holding_bars"],
                    is_core=core_flag,
                )
                matrix[k_idx][s_idx][t_idx] = row

    # 邻域稳健分数计算：仅使用固定 5x5x5 数组切片，禁止 iterrows / itertuples / append
    flat_rows = [None] * (n_k * n_s * n_t)
    flat_idx = 0
    for k_idx in range(n_k):
        for s_idx in range(n_s):
            s_start = max(0, s_idx - 1)
            s_end = min(n_s, s_idx + 2)
            for t_idx in range(n_t):
                t_start = max(0, t_idx - 1)
                t_end = min(n_t, t_idx + 2)

                # 提取 2D 邻域切片
                nbr_slice = EV_grid[k_idx, s_start:s_end, t_start:t_end]
                r_score = float(np.median(nbr_slice))
                pos_rate = float(np.mean(nbr_slice > 0.0))

                row = matrix[k_idx][s_idx][t_idx]
                row["robust_score"] = r_score
                row["positive_neighbor_rate"] = pos_rate

                # Stop-only: target == "NONE" (t_idx == 4)
                if TARGET_CODES[t_idx] == "NONE":
                    so_slice = EV_grid[k_idx, s_start:s_end, 4]
                    row["stop_only_robust_score"] = float(np.median(so_slice))
                    row["stop_only_positive_neighbor_rate"] = float(np.mean(so_slice > 0.0))
                else:
                    row["stop_only_robust_score"] = np.nan
                    row["stop_only_positive_neighbor_rate"] = np.nan

                flat_rows[flat_idx] = row
                flat_idx += 1

    df_surface = pd.DataFrame(flat_rows)
    if len(df_surface) != 125:
        raise SystemExit(f"STOP_PGM_EXEC1_SURFACE_COUNT_NOT_125: {len(df_surface)}")

    return df_surface


# ===========================================================================
# 9. 稳健策略选择 (TB2 Robust Selection)
# ===========================================================================
def select_policies(surface_df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """在 TB2 表面上选择 PRIMARY 策略与 STOP_ONLY_SECONDARY 策略。

    确定性 Tie-Break 规则：
    1. robust_score desc
    2. positive_neighbor_rate desc
    3. net_EV_per_decision desc
    4. n_trades desc
    5. k asc
    6. stop_s asc
    7. target index asc
    """
    target_idx_map = {code: idx for idx, code in enumerate(TARGET_CODES)}
    df = surface_df.copy()
    df["target_idx"] = df["target_code"].map(target_idx_map)

    # 1. Primary Policy Selection (来自 36 个 core)
    core_df = df[df["is_core"]].copy()
    if len(core_df) != 36:
        raise SystemExit(f"STOP_PGM_EXEC1_CORE_NOT_36: {len(core_df)}")

    valid_core = core_df[core_df["n_trades"] >= MIN_SELECTION_TRADES]
    has_trades_primary = len(valid_core) > 0
    candidate_pool = valid_core if has_trades_primary else core_df

    sorted_primary = candidate_pool.sort_values(
        by=[
            "robust_score",
            "positive_neighbor_rate",
            "net_EV_per_decision",
            "n_trades",
            "k",
            "stop_s",
            "target_idx",
        ],
        ascending=[False, False, False, False, True, True, True],
    )
    primary_row = sorted_primary.iloc[0].to_dict()

    selection_pass = bool(
        has_trades_primary
        and (primary_row["robust_score"] > 0.0)
        and (primary_row["positive_neighbor_rate"] >= MIN_POSITIVE_NEIGHBOR_RATE)
        and (primary_row["n_trades"] >= MIN_SELECTION_TRADES)
    )
    primary_row["selection_pass"] = selection_pass
    primary_row["policy_type"] = "PRIMARY"

    # 2. Secondary Stop-Only Policy Selection (来自 12 个 stop-only core)
    so_core_df = df[
        df["is_core"] & (df["target_code"] == "NONE")
    ].copy()
    if len(so_core_df) != 12:
        raise SystemExit(f"STOP_PGM_EXEC1_STOP_ONLY_CORE_NOT_12: {len(so_core_df)}")

    valid_so = so_core_df[so_core_df["n_trades"] >= MIN_SELECTION_TRADES]
    has_trades_so = len(valid_so) > 0
    so_pool = valid_so if has_trades_so else so_core_df

    sorted_so = so_pool.sort_values(
        by=[
            "stop_only_robust_score",
            "stop_only_positive_neighbor_rate",
            "net_EV_per_decision",
            "n_trades",
            "k",
            "stop_s",
            "target_idx",
        ],
        ascending=[False, False, False, False, True, True, True],
    )
    so_row = sorted_so.iloc[0].to_dict()

    so_pass = bool(
        has_trades_so
        and (so_row["stop_only_robust_score"] > 0.0)
        and (so_row["stop_only_positive_neighbor_rate"] >= MIN_POSITIVE_NEIGHBOR_RATE)
        and (so_row["n_trades"] >= MIN_SELECTION_TRADES)
    )
    so_row["selection_pass"] = so_pass
    so_row["policy_type"] = "STOP_ONLY_SECONDARY"

    return dict(primary=primary_row, stop_only=so_row)


# ===========================================================================
# 10. TB3 严格冻结样本验证 (TB3 Strict Freeze Validation)
# ===========================================================================
def evaluate_tb3_frozen(
    tensor_tb3: Dict[str, Any],
    primary_params: Dict[str, Any],
    stop_only_params: Dict[str, Any],
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """在 TB3 上仅计算 BASE、PRIMARY 与 STOP_ONLY_SECONDARY 三个策略。

    严禁在 TB3 上计算 125 策略表面。
    """
    N = len(tensor_tb3["direction"])
    entry_day = tensor_tb3["entry_day"]
    symbol = tensor_tb3["symbol"]
    hazard = tensor_tb3["hazard"]

    # 1. BASE Policy (0A baseline: next open -> next close, immediate fill)
    d = tensor_tb3["direction"]
    atr0 = tensor_tb3["atr0"]
    o1 = tensor_tb3["O"][:, 0]
    c1 = tensor_tb3["C"][:, 0]
    base_gross = d * (c1 - o1) / atr0
    base_action = np.ones(N, dtype=np.float64)
    base_net = base_gross - PRIMARY_COST_ATR0

    m_base = d0.strategy_metrics(
        action=base_action,
        r_trad=base_gross,
        cost=PRIMARY_COST_ATR0,
        entry_day=entry_day,
        symbol=symbol,
    )
    m_base["fill_rate"] = 1.0
    m_base["stop_rate"] = 0.0
    m_base["target_rate"] = 0.0
    m_base["timeout_rate"] = 1.0
    m_base["avg_holding_bars"] = 1.0

    # 2. PRIMARY Policy
    rel_k_pri = precompute_relative_path_for_k(tensor_tb3, primary_params["k"])
    sim_pri = simulate_policy(
        rel_k_pri,
        primary_params["stop_s"],
        primary_params["target_code"],
    )
    m_pri = compute_exec_metrics(sim_pri, tensor_tb3)

    # 3. STOP_ONLY_SECONDARY Policy
    rel_k_so = precompute_relative_path_for_k(tensor_tb3, stop_only_params["k"])
    sim_so = simulate_policy(
        rel_k_so,
        stop_only_params["stop_s"],
        "NONE",
    )
    m_so = compute_exec_metrics(sim_so, tensor_tb3)

    # 汇总并计算 H0/H1 分组统计
    policies = [
        ("BASE", 0.0, np.nan, "NONE", np.nan, m_base, base_gross, base_net),
        (
            "PRIMARY",
            primary_params["k"],
            primary_params["stop_s"],
            primary_params["target_code"],
            TARGET_MULTIPLES.get(primary_params["target_code"]),
            m_pri,
            sim_pri["gross_return"],
            sim_pri["net_return"],
        ),
        (
            "STOP_ONLY_SECONDARY",
            stop_only_params["k"],
            stop_only_params["stop_s"],
            "NONE",
            np.nan,
            m_so,
            sim_so["gross_return"],
            sim_so["net_return"],
        ),
    ]

    h0_mask = hazard == 0
    h1_mask = hazard == 1
    n_h0 = int(h0_mask.sum())
    n_h1 = int(h1_mask.sum())

    val_rows = []
    for name, k_v, s_v, t_code, t_mult, m, gross_arr, net_arr in policies:
        g_h0 = float(np.mean(gross_arr[h0_mask])) if n_h0 > 0 else np.nan
        n_h0_val = float(np.mean(net_arr[h0_mask])) if n_h0 > 0 else np.nan
        g_h1 = float(np.mean(gross_arr[h1_mask])) if n_h1 > 0 else np.nan
        n_h1_val = float(np.mean(net_arr[h1_mask])) if n_h1 > 0 else np.nan

        row = dict(
            policy_name=name,
            k=float(k_v),
            stop_s=(float(s_v) if np.isfinite(s_v) else np.nan),
            target_code=t_code,
            target_multiple=(float(t_mult) if (t_mult is not None and np.isfinite(t_mult)) else np.nan),
            n_decisions=m["n_decisions"],
            n_trades=m["n_trades"],
            trade_rate=m["trade_rate"],
            fill_rate=m["fill_rate"],
            gross_total_ATR0=m["gross_total_ATR0"],
            net_total_ATR0=m["net_total_ATR0"],
            gross_EV_per_decision=m["gross_EV_per_decision"],
            net_EV_per_decision=m["net_EV_per_decision"],
            net_EV_per_trade=m["net_EV_per_trade"],
            win_rate=m["win_rate"],
            mean_win=m["mean_win"],
            mean_loss=m["mean_loss"],
            payoff_ratio=m["payoff_ratio"],
            profit_factor=m["profit_factor"],
            break_even_cost=m["break_even_cost"],
            daily_sharpe_annualized=m["daily_sharpe_annualized"],
            max_drawdown_ATR0=m["max_drawdown_ATR0"],
            positive_symbol_count=m["positive_symbol_count"],
            top3_profit_share=m["top3_profit_share"],
            stop_rate=m["stop_rate"],
            target_rate=m["target_rate"],
            timeout_rate=m["timeout_rate"],
            avg_holding_bars=m["avg_holding_bars"],
            gross_EV_H0=g_h0,
            net_EV_H0=n_h0_val,
            gross_EV_H1=g_h1,
            net_EV_H1=n_h1_val,
            n_H0=n_h0,
            n_H1=n_h1,
        )
        val_rows.append(row)

    df_tb3_val = pd.DataFrame(val_rows)
    returns_dict = dict(
        BASE=base_net,
        PRIMARY=sim_pri["net_return"],
        STOP_ONLY=sim_so["net_return"],
    )
    return df_tb3_val, returns_dict


# ===========================================================================
# 11. Entry-Only 诊断评估 (Entry-Only Diagnostics)
# ===========================================================================
def evaluate_entry_only_diagnostics(
    tensor: Dict[str, Any],
    block_name: str,
) -> pd.DataFrame:
    """评估每个 k 值仅进行挂单入场、无止损、无止盈、6-bar 超时退出的诊断表现。"""
    rows = []
    for k in ENTRY_GRID:
        rel_k = precompute_relative_path_for_k(tensor, k)
        sim = simulate_policy(rel_k, stop_s=None, target_code="NONE")
        m = compute_exec_metrics(sim, tensor)
        row = dict(
            block=block_name,
            k=float(k),
            n_decisions=m["n_decisions"],
            n_trades=m["n_trades"],
            trade_rate=m["trade_rate"],
            fill_rate=m["fill_rate"],
            gross_total_ATR0=m["gross_total_ATR0"],
            net_total_ATR0=m["net_total_ATR0"],
            gross_EV_per_decision=m["gross_EV_per_decision"],
            net_EV_per_decision=m["net_EV_per_decision"],
            net_EV_per_trade=m["net_EV_per_trade"],
            win_rate=m["win_rate"],
            mean_win=m["mean_win"],
            mean_loss=m["mean_loss"],
            payoff_ratio=m["payoff_ratio"],
            profit_factor=m["profit_factor"],
            break_even_cost=m["break_even_cost"],
            daily_sharpe_annualized=m["daily_sharpe_annualized"],
            max_drawdown_ATR0=m["max_drawdown_ATR0"],
            positive_symbol_count=m["positive_symbol_count"],
            top3_profit_share=m["top3_profit_share"],
            avg_holding_bars=m["avg_holding_bars"],
        )
        rows.append(row)
    return pd.DataFrame(rows)


# ===========================================================================
# 12. 日聚类 Paired Bootstrap (复用 e0.paired_day_mean_bootstrap)
# ===========================================================================
def run_bootstrap_contrasts(
    entry_day: np.ndarray,
    returns_dict: Dict[str, np.ndarray],
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """在 TB3 净收益序列上运行日聚类 Paired Bootstrap。

    计算 4 个对比：
    1. PRIMARY_NET
    2. PRIMARY_MINUS_BASE
    3. STOP_ONLY_NET
    4. STOP_ONLY_MINUS_BASE
    """
    net_pri = returns_dict["PRIMARY"]
    net_base = returns_dict["BASE"]
    net_so = returns_dict["STOP_ONLY"]

    b_pri = e0.paired_day_mean_bootstrap(entry_day, net_pri, n_boot=n_boot, seed=seed)
    b_pri_base = e0.paired_day_mean_bootstrap(
        entry_day, net_pri - net_base, n_boot=n_boot, seed=seed
    )
    b_so = e0.paired_day_mean_bootstrap(entry_day, net_so, n_boot=n_boot, seed=seed)
    b_so_base = e0.paired_day_mean_bootstrap(
        entry_day, net_so - net_base, n_boot=n_boot, seed=seed
    )

    rows = [
        dict(contrast="PRIMARY_NET", **b_pri),
        dict(contrast="PRIMARY_MINUS_BASE", **b_pri_base),
        dict(contrast="STOP_ONLY_NET", **b_so),
        dict(contrast="STOP_ONLY_MINUS_BASE", **b_so_base),
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# 13. 科学结论裁定 (Formal Verdict Determination)
# ===========================================================================
def determine_verdict(
    tb2_primary: Dict[str, Any],
    tb3_primary_metrics: Dict[str, Any],
    bootstrap_df: pd.DataFrame,
) -> str:
    """根据严格协议裁定科学结论。

    仅由 TB2 frozen primary 在 TB3 上的表现裁定：
    Full support:
    - TB2 selection_pass == True
    - CI_lower(NetEV_EXEC1) > 0
    - CI_lower(EXEC1 - BASE) > 0
    - profit_factor > 1.0
    - daily_sharpe_annualized > 0.0
    - n_trades >= 5000
    => PGM_EXECUTION_EDGE_SUPPORTED_EXPLORATORY

    Improvement only:
    - CI_lower(EXEC1 - BASE) > 0
    => PGM_EXECUTION_IMPROVES_BASE_BUT_NOT_PROFITABLE_EXPLORATORY

    Otherwise:
    => PGM_EXECUTION_EDGE_NOT_SUPPORTED_EXPLORATORY
    """
    b_indexed = bootstrap_df.set_index("contrast")
    pri_net_ci_low = float(b_indexed.loc["PRIMARY_NET", "ci95_lower"])
    delta_ci_low = float(b_indexed.loc["PRIMARY_MINUS_BASE", "ci95_lower"])

    tb2_pass = bool(tb2_primary["selection_pass"])
    ci_net_pos = bool(pri_net_ci_low > 0.0)
    ci_delta_pos = bool(delta_ci_low > 0.0)
    pf_gt_1 = bool(tb3_primary_metrics["profit_factor"] > 1.0)
    sharpe_gt_0 = bool(tb3_primary_metrics["daily_sharpe_annualized"] > 0.0)
    trades_ok = bool(tb3_primary_metrics["n_trades"] >= MIN_SELECTION_TRADES)

    if (
        tb2_pass
        and ci_net_pos
        and ci_delta_pos
        and pf_gt_1
        and sharpe_gt_0
        and trades_ok
    ):
        return "PGM_EXECUTION_EDGE_SUPPORTED_EXPLORATORY"
    elif ci_delta_pos:
        return "PGM_EXECUTION_IMPROVES_BASE_BUT_NOT_PROFITABLE_EXPLORATORY"
    else:
        return "PGM_EXECUTION_EDGE_NOT_SUPPORTED_EXPLORATORY"


# ===========================================================================
# 14. 产物校验器 (Fail-Closed Artifacts Validator)
# ===========================================================================
def validate_in_memory_artifacts(artifacts: Dict[str, Any]) -> None:
    """内存产物完整性与格式闭环校验。"""
    # 1. 检查是否存在未授权或缺失产物
    keys = set(artifacts.keys())
    expected = set(EXACT_ARTIFACTS)
    if keys != expected:
        raise SystemExit(
            f"STOP_PGM_EXEC1_ARTIFACT_KEYS_MISMATCH: diff={keys.symmetric_difference(expected)}"
        )

    # 2. surface_tb2 校验：必须 exact 125 unique policies
    df_surf = artifacts[ARTIFACT_SURFACE_TB2]
    if len(df_surf) != 125:
        raise SystemExit(f"STOP_PGM_EXEC1_SURFACE_NOT_125: {len(df_surf)}")
    n_core = int(df_surf["is_core"].sum())
    if n_core != 36:
        raise SystemExit(f"STOP_PGM_EXEC1_CORE_NOT_36: {n_core}")
    n_so_core = int((df_surf["is_core"] & (df_surf["target_code"] == "NONE")).sum())
    if n_so_core != 12:
        raise SystemExit(f"STOP_PGM_EXEC1_STOP_ONLY_CORE_NOT_12: {n_so_core}")

    # 3. selected_policies 校验：必须 exact 2 rows
    df_sel = artifacts[ARTIFACT_SELECTED_POLICIES]
    if len(df_sel) != 2:
        raise SystemExit(f"STOP_PGM_EXEC1_SELECTED_NOT_2: {len(df_sel)}")
    sel_types = sorted(df_sel["policy_type"].tolist())
    if sel_types != ["PRIMARY", "STOP_ONLY_SECONDARY"]:
        raise SystemExit(f"STOP_PGM_EXEC1_SELECTED_TYPES_INVALID: {sel_types}")

    # 4. tb3_validation 校验：必须 exact 3 rows
    df_val = artifacts[ARTIFACT_TB3_VALIDATION]
    if len(df_val) != 3:
        raise SystemExit(f"STOP_PGM_EXEC1_TB3_VAL_NOT_3: {len(df_val)}")
    p_names = df_val["policy_name"].tolist()
    if p_names != ["BASE", "PRIMARY", "STOP_ONLY_SECONDARY"]:
        raise SystemExit(f"STOP_PGM_EXEC1_TB3_VAL_POLICIES_INVALID: {p_names}")

    # 5. bootstrap 校验：必须 exact 4 rows
    df_boot = artifacts[ARTIFACT_BOOTSTRAP]
    if len(df_boot) != 4:
        raise SystemExit(f"STOP_PGM_EXEC1_BOOTSTRAP_NOT_4: {len(df_boot)}")

    # 6. summary 语义字段校验
    summ = artifacts[ARTIFACT_FORMAL_SUMMARY]
    required_keys = [
        "run_head",
        "base_sha",
        "sample_hash",
        "transition_hash",
        "counts",
        "symbols",
        "blocks",
        "window_A",
        "window_B",
        "score_owner_parity",
        "atr0_parity",
        "path_contract",
        "FUTURE_BARS",
        "ENTRY_WAIT_BARS",
        "MAX_HOLD_BARS",
        "MAX_GAP_MINUTES",
        "ENTRY_GRID",
        "STOP_GRID",
        "TARGET_CODES",
        "core_grids",
        "cost",
        "bootstrap_N",
        "bootstrap_seed",
        "tb2_path_valid_count",
        "tb3_path_valid_count",
        "baseline_parity_errors",
        "selected_PRIMARY",
        "selected_STOP_ONLY_SECONDARY",
        "tb2_selection_pass",
        "tb3_PRIMARY_metrics",
        "tb3_bootstrap",
        "formal_verdict",
        "known_limitations",
        "timing",
        "artifact_files",
    ]
    for rk in required_keys:
        if rk not in summ:
            raise SystemExit(f"STOP_PGM_EXEC1_SUMMARY_MISSING_KEY: {rk}")


def write_and_verify_artifacts_on_disk(
    artifacts: Dict[str, Any],
    out_dir: pathlib.Path = OUT_DIR,
) -> None:
    """写入磁盘并重新读取做字节/数值精确双向对齐校验。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    # 写入文件
    for name, content in artifacts.items():
        path = out_dir / name
        if isinstance(content, pd.DataFrame):
            content.to_csv(path, index=False)
        elif isinstance(content, dict):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            raise SystemExit(f"STOP_PGM_EXEC1_UNSUPPORTED_ARTIFACT_TYPE: {type(content)}")

    # 重新从磁盘读取校验
    disk_files = sorted(list(out_dir.glob(f"{ARTIFACT_PREFIX}*")))
    disk_file_names = [f.name for f in disk_files]
    if disk_file_names != sorted(EXACT_ARTIFACTS):
        raise SystemExit(
            f"STOP_PGM_EXEC1_DISK_ARTIFACT_SET_MISMATCH: expected {EXACT_ARTIFACTS}, found {disk_file_names}"
        )

    for name in EXACT_ARTIFACTS:
        path = out_dir / name
        mem_obj = artifacts[name]
        if name.endswith(".csv"):
            disk_df = pd.read_csv(path)
            if list(disk_df.columns) != list(mem_obj.columns):
                raise SystemExit(f"STOP_PGM_EXEC1_DISK_COLUMNS_MISMATCH: {name}")
            if len(disk_df) != len(mem_obj):
                raise SystemExit(f"STOP_PGM_EXEC1_DISK_ROWS_MISMATCH: {name}")
            # 校验数值列
            for col in mem_obj.columns:
                if np.issubdtype(mem_obj[col].dtype, np.number):
                    mem_vals = mem_obj[col].to_numpy(float)
                    disk_vals = disk_df[col].to_numpy(float)
                    mask = np.isfinite(mem_vals) & np.isfinite(disk_vals)
                    if np.any(mask):
                        max_diff = np.max(np.abs(mem_vals[mask] - disk_vals[mask]))
                        if max_diff > 1e-12:
                            raise SystemExit(
                                f"STOP_PGM_EXEC1_DISK_NUMERIC_MISMATCH: {name}.{col} max_diff={max_diff:.2e}"
                            )
        elif name.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                disk_json = json.load(f)
            if set(disk_json.keys()) != set(mem_obj.keys()):
                raise SystemExit(f"STOP_PGM_EXEC1_DISK_JSON_KEYS_MISMATCH: {name}")


# ===========================================================================
# 15. Audit-Only 模式 (无科学结果输出)
# ===========================================================================
def run_audit_only() -> None:
    """仅运行治理审计与数据一致性检查，禁止打印任何科学结论或表面 EV。"""
    print("=" * 60)
    print("PGM-EXEC-1: AUDIT-ONLY MODE")
    print("=" * 60)

    # 1. Git 状态审计
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
    ).strip()
    print(f"[AUDIT] HEAD={head}")
    print(f"[AUDIT] BASE_SHA={BASE_SHA}")
    merge_res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )
    if merge_res.returncode != 0:
        raise SystemExit("STOP_PGM_EXEC1_FREEZE_CHECK_FAIL")

    # 2. 产物 Hash 审计
    hashes = d0.compute_artifact_hashes()
    print(f"[AUDIT] sample_sha256={hashes['sample_artifact_sha256']}")
    print(f"[AUDIT] transition_sha256={hashes['transition_artifact_sha256']}")

    # 3. 数据集与 ATR0 对齐审计
    prep = d0.load_prepared_frame()
    obs = n0a.load_observed_decision_universe()
    assert_allowed_blocks(obs)
    aud = n0a.audit_decision_universe(obs)
    print(f"[AUDIT] n_all_obs={aud['n_all_obs']} H0={aud['n_hazard0']} H1={aud['n_hazard1']}")
    print(f"[AUDIT] symbols_count={len(aud['symbols'])}")
    print(f"[AUDIT] blocks={sorted(obs['block'].unique().tolist())}")

    # ATR0 所有者对齐的输入契约必须先被证明是全量决策宇宙
    cur_truth = n0a.load_transition_truth_audit()["cur"]
    assert_atr0_audit_universe(obs, cur_truth)
    atr0_err = n0a.audit_atr0_owner_parity(obs, cur_truth)
    print(f"[AUDIT] max_abs_atr0_owner_error={atr0_err:.2e}")
    if atr0_err > 1e-12:
        raise SystemExit("STOP_PGM_EXEC1_ATR0_PARITY_FAIL")

    # 4. Window A/B Score 所有权验证
    print("[AUDIT] Fitting Window A & Window B for score owner verification...")
    bundle = load_and_score()
    scored_A = bundle["scored_A"]
    scored_B = bundle["scored_B"]
    bars_by_sym = bundle["bars_by_sym"]

    err_A_tb2 = d0.verify_window_score_owner(scored_A, bundle["fit_A"], "TB2")
    err_B_tb3 = d0.verify_window_score_owner(scored_B, bundle["fit_B"], "TB3")
    print(f"[AUDIT] Window A score owner error on TB2: {err_A_tb2:.2e}")
    print(f"[AUDIT] Window B score owner error on TB3: {err_B_tb3:.2e}")

    # 5. Future Tensor 形状与排除统计 (TB2 与 TB3)
    eval_tb2 = extract_evaluation_sample(scored_A, TB2_BLOCK)
    tensor_tb2 = build_future_tensor(eval_tb2, bars_by_sym)
    n_raw_tb2 = len(eval_tb2)
    n_valid_tb2 = int(tensor_tb2["path_valid"].sum())
    disc_tb2 = int(tensor_tb2["disc_excluded"].sum())
    gap_tb2 = int(tensor_tb2["gap_excluded"].sum())
    day_tb2 = int(tensor_tb2["day_excluded"].sum())

    print(f"[AUDIT] TB2 raw_count={n_raw_tb2} path_valid_count={n_valid_tb2}")
    print(f"[AUDIT] TB2 disc_excluded={disc_tb2} gap_excluded={gap_tb2} day_excluded={day_tb2}")

    # Baseline Parity 门禁 (TB2)
    eval_tb2_valid = eval_tb2[tensor_tb2["path_valid"]].reset_index(drop=True)
    t_tb2_valid = slice_future_tensor(tensor_tb2, tensor_tb2["path_valid"])
    err_base_tb2 = verify_baseline_parity(t_tb2_valid, eval_tb2_valid["pi"].to_numpy(float))
    print(f"[AUDIT] TB2 baseline parity max error: {err_base_tb2:.2e}")

    eval_tb3 = extract_evaluation_sample(scored_B, TB3_BLOCK)
    tensor_tb3 = build_future_tensor(eval_tb3, bars_by_sym)
    n_raw_tb3 = len(eval_tb3)
    n_valid_tb3 = int(tensor_tb3["path_valid"].sum())
    disc_tb3 = int(tensor_tb3["disc_excluded"].sum())
    gap_tb3 = int(tensor_tb3["gap_excluded"].sum())
    day_tb3 = int(tensor_tb3["day_excluded"].sum())

    print(f"[AUDIT] TB3 raw_count={n_raw_tb3} path_valid_count={n_valid_tb3}")
    print(f"[AUDIT] TB3 disc_excluded={disc_tb3} gap_excluded={gap_tb3} day_excluded={day_tb3}")

    eval_tb3_valid = eval_tb3[tensor_tb3["path_valid"]].reset_index(drop=True)
    t_tb3_valid = slice_future_tensor(tensor_tb3, tensor_tb3["path_valid"])
    err_base_tb3 = verify_baseline_parity(t_tb3_valid, eval_tb3_valid["pi"].to_numpy(float))
    print(f"[AUDIT] TB3 baseline parity max error: {err_base_tb3:.2e}")

    # 6. 参数网格计数审计
    print(f"[AUDIT] surface_policy_count={len(ENTRY_GRID) * len(STOP_GRID) * len(TARGET_CODES)} (expected 125)")
    core_count = sum(
        1
        for k in ENTRY_GRID
        for s in STOP_GRID
        for t in TARGET_CODES
        if is_core_policy(k, s, t)
    )
    so_core_count = sum(
        1
        for k in ENTRY_GRID
        for s in STOP_GRID
        for t in TARGET_CODES
        if is_stop_only_core(k, s, t)
    )
    print(f"[AUDIT] core_candidates_count={core_count} (expected 36)")
    print(f"[AUDIT] stop_only_core_count={so_core_count} (expected 12)")

    # 7. 核心热路径源码审计 (禁止 Python 逐行循环)
    audit_forbidden_patterns_in_hot_loops()
    print("[AUDIT] Hot loop source code audit: PASS (no forbidden row/trade loops)")
    print("=" * 60)
    print("[AUDIT] ALL AUDIT CHECKS PASSED. NO SCIENTIFIC OUTCOME REPORTED.")
    print("=" * 60)


def audit_forbidden_patterns_in_hot_loops() -> None:
    """静态检查源码，确保关键热路径没有使用禁止的逐行循环结构。"""
    src_file = pathlib.Path(__file__).resolve()
    with open(src_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(src_file))

    target_funcs = {"build_future_tensor", "simulate_policy", "evaluate_surface"}
    forbidden_tokens = ["iterrows", "itertuples", "append"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_funcs:
            for subnode in ast.walk(node):
                if isinstance(subnode, ast.Attribute) and subnode.attr in forbidden_tokens:
                    raise SystemExit(
                        f"STOP_PGM_EXEC1_FORBIDDEN_HOT_LOOP: {node.name} calls {subnode.attr}"
                    )
                # 检查 for row in ... 或 for trade in ...
                if isinstance(subnode, ast.For):
                    if isinstance(subnode.target, ast.Name):
                        if subnode.target.id in {"row", "trade"}:
                            raise SystemExit(
                                f"STOP_PGM_EXEC1_FORBIDDEN_ROW_LOOP: {node.name} loops over {subnode.target.id}"
                            )


# ===========================================================================
# 16. Smoke 模式 (轻量级端到端测试)
# ===========================================================================
def run_smoke() -> None:
    """运行 Smoke 测试：真实 Window A/B fit，但在上限截断的样本上走通全流程。"""
    print("=" * 60)
    print("PGM-EXEC-1: SMOKE MODE (CAP=4096, BOOTSTRAP=200)")
    print("=" * 60)
    t_start = time.time()

    # 1. 加载并评分 (真实 Window A/B fit 一次)
    t0 = time.time()
    bundle = load_and_score()
    t_load = time.time() - t0
    scored_A = bundle["scored_A"]
    scored_B = bundle["scored_B"]
    bars_by_sym = bundle["bars_by_sym"]

    # 2. 截断样本提取
    eval_tb2 = extract_evaluation_sample(scored_A, TB2_BLOCK)
    if len(eval_tb2) > SMOKE_EVAL_CAP:
        eval_tb2 = eval_tb2.iloc[:SMOKE_EVAL_CAP].copy().reset_index(drop=True)

    eval_tb3 = extract_evaluation_sample(scored_B, TB3_BLOCK)
    if len(eval_tb3) > SMOKE_EVAL_CAP:
        eval_tb3 = eval_tb3.iloc[:SMOKE_EVAL_CAP].copy().reset_index(drop=True)

    # 3. Future Tensor 构建
    t0 = time.time()
    tensor_tb2 = build_future_tensor(eval_tb2, bars_by_sym)
    t_tensor_tb2 = time.time() - t0

    t0 = time.time()
    tensor_tb3 = build_future_tensor(eval_tb3, bars_by_sym)
    t_tensor_tb3 = time.time() - t0

    # 过滤至 path_valid
    eval_tb2_valid = eval_tb2[tensor_tb2["path_valid"]].reset_index(drop=True)
    t_tb2_valid = slice_future_tensor(tensor_tb2, tensor_tb2["path_valid"])
    verify_baseline_parity(t_tb2_valid, eval_tb2_valid["pi"].to_numpy(float))

    eval_tb3_valid = eval_tb3[tensor_tb3["path_valid"]].reset_index(drop=True)
    t_tb3_valid = slice_future_tensor(tensor_tb3, tensor_tb3["path_valid"])
    verify_baseline_parity(t_tb3_valid, eval_tb3_valid["pi"].to_numpy(float))

    print(f"[SMOKE] TB2 path-valid: {len(eval_tb2_valid)} / {len(eval_tb2)}")
    print(f"[SMOKE] TB3 path-valid: {len(eval_tb3_valid)} / {len(eval_tb3)}")

    # 4. MAE / MFE 计算
    t0 = time.time()
    mae_mfe_tb2 = compute_mae_mfe(t_tb2_valid)
    summarize_mae_mfe(mae_mfe_tb2, t_tb2_valid["hazard"], TB2_BLOCK)
    mae_mfe_tb3 = compute_mae_mfe(t_tb3_valid)
    summarize_mae_mfe(mae_mfe_tb3, t_tb3_valid["hazard"], TB3_BLOCK)
    t_mae_mfe = time.time() - t0

    # 5. 评估 TB2 表面 (125 策略)
    t0 = time.time()
    df_surface_tb2 = evaluate_surface(t_tb2_valid, eval_block=TB2_BLOCK)
    t_surface = time.time() - t0

    # 6. 选择策略
    t0 = time.time()
    selections = select_policies(df_surface_tb2)
    primary = selections["primary"]
    stop_only = selections["stop_only"]
    t_selection = time.time() - t0
    print(f"[SMOKE] Selected PRIMARY: k={primary['k']} stop={primary['stop_s']} target={primary['target_code']}")
    print(f"[SMOKE] Selected STOP_ONLY: k={stop_only['k']} stop={stop_only['stop_s']}")

    # 7. TB3 冻结验证 (仅 3 个策略)
    t0 = time.time()
    df_tb3_val, returns_dict = evaluate_tb3_frozen(t_tb3_valid, primary, stop_only)
    t_tb3_val = time.time() - t0

    # 8. Bootstrap
    t0 = time.time()
    df_boot = run_bootstrap_contrasts(
        t_tb3_valid["entry_day"],
        returns_dict,
        n_boot=SMOKE_BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
    )
    t_boot = time.time() - t0

    t_total = time.time() - t_start
    print(f"[SMOKE] Timing: load={t_load:.1f}s tensor_tb2={t_tensor_tb2:.2f}s surface={t_surface:.2f}s boot={t_boot:.2f}s total={t_total:.1f}s")
    print("=" * 60)
    print("SMOKE ONLY")
    print("NO SCIENTIFIC VERDICT")
    print("=" * 60)


# ===========================================================================
# 17. Formal Full 实验主运行流程 (受环境变量与治理协议严格保护)
# ===========================================================================
def run_full_exploratory() -> None:
    """运行 PGM-EXEC-1 正式探索性全量实验流程并生成 8 个标准 Artifacts。"""
    # 1. 严格授权门禁
    require_full_authorization()

    # 2. 检查 Git 状态
    head = assert_clean_git_tree()

    # 3. 检查文件闭环 (不得存在旧 artifacts)
    assert_no_stale_artifacts()

    print("=" * 60)
    print(f"PGM-EXEC-1: FORMAL FULL EXPLORATORY RUN (HEAD={head})")
    print("=" * 60)
    t_all_start = time.time()

    # 4. 加载数据并执行 Window A / B 独立评分 (全局一次)
    t0 = time.time()
    bundle = load_and_score()
    t_load_score = time.time() - t0
    scored_A = bundle["scored_A"]
    scored_B = bundle["scored_B"]
    bars_by_sym = bundle["bars_by_sym"]

    # 4b. ATR0 所有者对齐审计
    # 必须使用 prepared 全量决策宇宙（bundle["prep"]["obs_day"]），
    # 绝不接受 block 子样本 / path-valid 子样本：那会导致「比较了不同 universe 却报告 PASS」。
    # 该 gate 早于任何科学计算执行 (fail-closed before MAE/MFE, surface, selection, TB3)。
    obs_full = bundle["prep"]["obs_day"]
    cur_truth = n0a.load_transition_truth_audit()["cur"]
    assert_atr0_audit_universe(obs_full, cur_truth)
    err_atr0_parity = float(n0a.audit_atr0_owner_parity(obs_full, cur_truth))

    # 5. 提取决策宇宙
    eval_tb2 = extract_evaluation_sample(scored_A, TB2_BLOCK)
    eval_tb3 = extract_evaluation_sample(scored_B, TB3_BLOCK)

    # 6. 构建未来路径张量
    t0 = time.time()
    tensor_tb2_raw = build_future_tensor(eval_tb2, bars_by_sym)
    t_tensor_tb2 = time.time() - t0

    t0 = time.time()
    tensor_tb3_raw = build_future_tensor(eval_tb3, bars_by_sym)
    t_tensor_tb3 = time.time() - t0

    # 过滤至 common path-valid 样本
    eval_tb2 = eval_tb2[tensor_tb2_raw["path_valid"]].reset_index(drop=True)
    tensor_tb2 = slice_future_tensor(tensor_tb2_raw, tensor_tb2_raw["path_valid"])
    err_tb2 = verify_baseline_parity(tensor_tb2, eval_tb2["pi"].to_numpy(float))

    eval_tb3 = eval_tb3[tensor_tb3_raw["path_valid"]].reset_index(drop=True)
    tensor_tb3 = slice_future_tensor(tensor_tb3_raw, tensor_tb3_raw["path_valid"])
    err_tb3 = verify_baseline_parity(tensor_tb3, eval_tb3["pi"].to_numpy(float))

    # 7. MAE / MFE 分布
    t0 = time.time()
    mae_mfe_tb2 = compute_mae_mfe(tensor_tb2)
    q_tb2, th_tb2 = summarize_mae_mfe(mae_mfe_tb2, tensor_tb2["hazard"], TB2_BLOCK)

    mae_mfe_tb3 = compute_mae_mfe(tensor_tb3)
    q_tb3, th_tb3 = summarize_mae_mfe(mae_mfe_tb3, tensor_tb3["hazard"], TB3_BLOCK)

    df_mae_mfe_quantiles = pd.concat([q_tb2, q_tb3], ignore_index=True)
    df_mae_mfe_thresholds = pd.concat([th_tb2, th_tb3], ignore_index=True)
    t_mae_mfe = time.time() - t0

    # 8. Entry-Only 诊断
    eo_tb2 = evaluate_entry_only_diagnostics(tensor_tb2, TB2_BLOCK)
    eo_tb3 = evaluate_entry_only_diagnostics(tensor_tb3, TB3_BLOCK)
    df_entry_only = pd.concat([eo_tb2, eo_tb3], ignore_index=True)

    # 9. TB2 表面评估 (125 策略)
    t0 = time.time()
    df_surface_tb2 = evaluate_surface(tensor_tb2, eval_block=TB2_BLOCK)
    t_surface = time.time() - t0

    # 10. 策略选择与冻结
    t0 = time.time()
    selections = select_policies(df_surface_tb2)
    primary = selections["primary"]
    stop_only = selections["stop_only"]
    t_selection = time.time() - t0

    df_selected = pd.DataFrame(
        [
            dict(
                policy_type="PRIMARY",
                k=primary["k"],
                stop_s=primary["stop_s"],
                target_code=primary["target_code"],
                target_multiple=TARGET_MULTIPLES.get(primary["target_code"]),
                robust_score=primary["robust_score"],
                positive_neighbor_rate=primary["positive_neighbor_rate"],
                n_trades=primary["n_trades"],
                net_EV_per_decision=primary["net_EV_per_decision"],
                selection_pass=primary["selection_pass"],
            ),
            dict(
                policy_type="STOP_ONLY_SECONDARY",
                k=stop_only["k"],
                stop_s=stop_only["stop_s"],
                target_code="NONE",
                target_multiple=np.nan,
                robust_score=stop_only["stop_only_robust_score"],
                positive_neighbor_rate=stop_only["stop_only_positive_neighbor_rate"],
                n_trades=stop_only["n_trades"],
                net_EV_per_decision=stop_only["net_EV_per_decision"],
                selection_pass=stop_only["selection_pass"],
            ),
        ]
    )

    # 11. TB3 严格冻结验证
    t0 = time.time()
    df_tb3_val, returns_dict = evaluate_tb3_frozen(tensor_tb3, primary, stop_only)
    t_tb3_val = time.time() - t0

    # 12. Bootstrap
    t0 = time.time()
    df_bootstrap = run_bootstrap_contrasts(
        tensor_tb3["entry_day"],
        returns_dict,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
    )
    t_boot = time.time() - t0

    # 13. 裁定科学结论
    pri_tb3_metrics = df_tb3_val[df_tb3_val["policy_name"] == "PRIMARY"].iloc[0].to_dict()
    verdict = determine_verdict(primary, pri_tb3_metrics, df_bootstrap)

    # 14. 汇总 Formal Summary JSON
    timing_dict = dict(
        load_and_score_seconds=float(t_load_score),
        future_tensor_tb2_seconds=float(t_tensor_tb2),
        future_tensor_tb3_seconds=float(t_tensor_tb3),
        mae_mfe_seconds=float(t_mae_mfe),
        surface_tb2_seconds=float(t_surface),
        selection_seconds=float(t_selection),
        tb3_validation_seconds=float(t_tb3_val),
        bootstrap_seconds=float(t_boot),
        total_seconds=float(time.time() - t_all_start),
    )

    hashes = d0.compute_artifact_hashes()
    formal_summary = dict(
        run_head=head,
        base_sha=BASE_SHA,
        sample_hash=hashes["sample_artifact_sha256"],
        transition_hash=hashes["transition_artifact_sha256"],
        counts=dict(
            tb2_raw=len(tensor_tb2_raw["path_valid"]),
            tb2_valid=len(eval_tb2),
            tb3_raw=len(tensor_tb3_raw["path_valid"]),
            tb3_valid=len(eval_tb3),
        ),
        symbols=sorted(eval_tb2["symbol"].unique().tolist()),
        blocks=ALLOWED_BLOCKS,
        window_A="W0_FIT_ONCE",
        window_B="W1_FIT_ONCE",
        score_owner_parity=dict(
            tb2_window_A_max_err=float(d0.verify_window_score_owner(scored_A, bundle["fit_A"], "TB2")),
            tb3_window_B_max_err=float(d0.verify_window_score_owner(scored_B, bundle["fit_B"], "TB3")),
        ),
        atr0_parity=float(err_atr0_parity),
        path_contract=dict(
            FUTURE_BARS=FUTURE_BARS,
            ENTRY_WAIT_BARS=ENTRY_WAIT_BARS,
            MAX_HOLD_BARS=MAX_HOLD_BARS,
            MAX_GAP_MINUTES=MAX_GAP_MINUTES,
        ),
        FUTURE_BARS=FUTURE_BARS,
        ENTRY_WAIT_BARS=ENTRY_WAIT_BARS,
        MAX_HOLD_BARS=MAX_HOLD_BARS,
        MAX_GAP_MINUTES=MAX_GAP_MINUTES,
        ENTRY_GRID=ENTRY_GRID.tolist(),
        STOP_GRID=STOP_GRID.tolist(),
        TARGET_CODES=TARGET_CODES,
        core_grids=dict(
            core_entries=sorted(list(CORE_ENTRY)),
            core_stops=sorted(list(CORE_STOP)),
            core_targets=sorted(list(CORE_TARGET)),
        ),
        cost=PRIMARY_COST_ATR0,
        bootstrap_N=BOOTSTRAP_N,
        bootstrap_seed=BOOTSTRAP_SEED,
        tb2_path_valid_count=len(eval_tb2),
        tb3_path_valid_count=len(eval_tb3),
        baseline_parity_errors=dict(tb2=float(err_tb2), tb3=float(err_tb3)),
        selected_PRIMARY=primary,
        selected_STOP_ONLY_SECONDARY=stop_only,
        tb2_selection_pass=bool(primary["selection_pass"]),
        tb3_PRIMARY_metrics=pri_tb3_metrics,
        tb3_bootstrap=df_bootstrap.to_dict(orient="records"),
        formal_verdict=verdict,
        known_limitations=[
            "Exploratory research on TB1/TB2/TB3 only; TB4 remains strictly locked.",
            "Execution assumes 5m continuous limit fill without favorable price improvement.",
            "Cross-session and discontinuity paths are excluded from causal holding periods.",
        ],
        timing=timing_dict,
        artifact_files=EXACT_ARTIFACTS,
    )

    artifacts_map = {
        ARTIFACT_MAE_MFE_QUANTILES: df_mae_mfe_quantiles,
        ARTIFACT_MAE_MFE_THRESHOLDS: df_mae_mfe_thresholds,
        ARTIFACT_ENTRY_ONLY: df_entry_only,
        ARTIFACT_SURFACE_TB2: df_surface_tb2,
        ARTIFACT_SELECTED_POLICIES: df_selected,
        ARTIFACT_TB3_VALIDATION: df_tb3_val,
        ARTIFACT_BOOTSTRAP: df_bootstrap,
        ARTIFACT_FORMAL_SUMMARY: formal_summary,
    }

    # 15. 内存产物语义校验
    validate_in_memory_artifacts(artifacts_map)

    # 16. 写入磁盘并重新读取校验
    write_and_verify_artifacts_on_disk(artifacts_map)

    print(f"[FORMAL] Formal execution completed in {timing_dict['total_seconds']:.1f}s.")
    print(f"[FORMAL] VERDICT: {verdict}")


# ===========================================================================
# 18. CLI 入口
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--audit-only", action="store_true", help="Run audit checks only")
    parser.add_argument("--smoke", action="store_true", help="Run smoke pipeline on capped sample")
    parser.add_argument("--full-exploratory", action="store_true", help="Run full formal exploratory experiment")
    args = parser.parse_args()

    if args.audit_only:
        run_audit_only()
    elif args.smoke:
        run_smoke()
    elif args.full_exploratory:
        run_full_exploratory()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
