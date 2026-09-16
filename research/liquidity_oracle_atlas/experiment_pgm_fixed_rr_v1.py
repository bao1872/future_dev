"""
experiment_pgm_fixed_rr_v1.py

PGM-FIXED-RR-1 -- 固定 2R 收益 / 1R 风险结构下的 PGM 方向优势检验

唯一研究问题：
    不给 execution 任何优化空间（无 stop 搜索、无 target 搜索、无 entry 搜索），
    只采用固定风险收益结构 (Reward/Risk = 2:1)，PGM 的方向优势是否足以产生正期望？

冻结设计（本实验不搜索任何参数）：
    方向      d_t = sign(score_mu,t)        （PGM 冻结所有者）
    Entry     P_0 = O_{t+1}                 （次根开盘，不做 pullback）
    Stop      1 * ATR0_t                    （固定，非搜索量）
    Target    2 * ATR0_t                    （固定，Reward/Risk = 2:1）
    成本      0.01 ATR0                     （每笔一次）

持仓窗口（报告维度，不是选择参数）：6 / 12 / 24 bars（30min / 1h / 2h）

对照组：
    A  PGM       d_t = sign(score_mu)
    B  REVERSE   -d_t
    C  RANDOM    rng.choice([-1, 1], seed=20260916)  （固定 seed，确定性）

复用所有者（禁止重新实现）：
    x1.load_and_score            Window A / B 拟合与评分（各 fit 一次）
    x1.extract_evaluation_sample 评测宇宙过滤
    x1.build_future_tensor        未来路径张量（本实验 n_future = HMAX = 24，只构建一次）
    x1.slice_future_tensor        沿行切片
    x1.verify_baseline_parity     基线经济收益对齐硬门
    d0.strategy_metrics           策略度量所有者（PF / win / payoff / Sharpe / MaxDD）
    e0.paired_day_mean_bootstrap   日聚类 bootstrap 唯一所有者

效率契约：
    未来路径张量一次性构建为 N x 24，三个持仓窗口共用并切片，
    禁止按持仓窗口重复构建；核心模拟器全部 NumPy 向量化，禁止 trade/bar 双重循环。
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_pgm_exec1_entry_stop_target_v1 as x1
import research.liquidity_oracle_atlas.experiment_pgm_native0a_one_step_alpha_v1 as n0a
import research.liquidity_oracle_atlas.experiment_pgm_native0d_acceleration_terminal_outcome_v1 as d0
import research.liquidity_oracle_atlas.experiment_pgm_native0e_consensus_acceleration_v1 as e0

# ---------------------------------------------------------------------------
# 0. 治理与冻结常量
# ---------------------------------------------------------------------------
BASE_SHA = "7b77fe4d1523240b4bda66343b60af636acab7d4"

EXPERIMENT_NAME = "PGM-FIXED-RR-1 -- Fixed 2R Reward / 1R Risk on PGM direction"
EXPERIMENT_SCOPE = "EXPLORATORY_FIXED_RR_ON_TB2_TB3"

ALLOWED_BLOCKS = ["TB1", "TB2", "TB3"]
EVAL_BLOCKS = ["TB2", "TB3"]
TB2_BLOCK = "TB2"
TB3_BLOCK = "TB3"

# 固定风险收益结构（Reward/Risk = 2:1），两个量互相独立，禁止写成 target = m * stop
STOP_R = 1.0
TARGET_R = 2.0
COST_ATR0 = 0.01

# 持仓窗口：报告维度（30min / 1h / 2h）
HOLD_GRID = [6, 12, 24]
HMAX = 24

RANDOM_SEED = 20260916
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260916

# 跳空成交规则（两者均为冻结常量，不存在搜索）
#   IDEAL_FILL                  : 用户设计规定 —— 止损固定 -1R、止盈固定 +2R
#   CONSERVATIVE_GAP_THROUGH    : 与 EXEC-1 §17/§19 一致 —— 开盘已越过止损位时承担 gap loss
GAP_RULE_IDEAL = "IDEAL_FILL"
GAP_RULE_CONSERVATIVE = "CONSERVATIVE_GAP_THROUGH"
GAP_RULES = [GAP_RULE_IDEAL, GAP_RULE_CONSERVATIVE]
PRIMARY_GAP_RULE = GAP_RULE_IDEAL

DIRECTION_PGM = "PGM"
DIRECTION_REVERSE = "REVERSE"
DIRECTION_RANDOM = "RANDOM"
DIRECTION_KINDS = [DIRECTION_PGM, DIRECTION_REVERSE, DIRECTION_RANDOM]

# 退出原因编码
NO_EXIT = 0
STOP_EXIT = 1
TARGET_EXIT = 2
TIMEOUT_EXIT = 3

SMOKE_CAP = 4096

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = _REPO_ROOT / "artifacts" / "liquidity_oracle_atlas"

ARTIFACT_PREFIX = "pgm_fixed_rr1_"
ARTIFACT_RESULTS = "pgm_fixed_rr1_results.csv"
ARTIFACT_BOOTSTRAP = "pgm_fixed_rr1_bootstrap.csv"
ARTIFACT_SUMMARY = "pgm_fixed_rr1_summary.json"
EXACT_ARTIFACTS = [ARTIFACT_RESULTS, ARTIFACT_BOOTSTRAP, ARTIFACT_SUMMARY]


# ===========================================================================
# 1. 治理门禁
# ===========================================================================
def assert_allowed_blocks(df: pd.DataFrame) -> None:
    """严格禁止 TB4 及未许可 block。"""
    blocks = list(df["block"].unique())
    if "TB4" in blocks:
        raise SystemExit("STOP_PGM_FIXEDRR1_FORBIDDEN_TB4")
    for b in blocks:
        if b not in ALLOWED_BLOCKS:
            raise SystemExit(f"STOP_PGM_FIXEDRR1_FORBIDDEN_BLOCK: {b}")


def require_run_authorization() -> None:
    """正式运行必须且仅能通过环境变量授权。"""
    token = os.environ.get("AUTHORIZE_PGM_FIXED_RR1_RUN", "").strip()
    if token != "1":
        raise SystemExit("STOP_PGM_FIXEDRR1_RUN_NOT_AUTHORIZED")


def assert_clean_git_tree() -> str:
    """tracked tree 与 staging 必须 clean，返回 HEAD。"""
    if subprocess.run(["git", "diff", "--exit-code"], cwd=str(_REPO_ROOT), capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_FIXEDRR1_GIT_TREE_NOT_CLEAN")
    if subprocess.run(["git", "diff", "--cached", "--exit-code"], cwd=str(_REPO_ROOT), capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_FIXEDRR1_GIT_STAGING_NOT_CLEAN")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip()


def assert_no_stale_artifacts() -> None:
    """运行前 artifacts 目录不得存在任何 pgm_fixed_rr1_* 文件。"""
    stale = sorted(list(OUT_DIR.glob(f"{ARTIFACT_PREFIX}*")))
    if len(stale) > 0:
        raise SystemExit(f"STOP_PGM_FIXEDRR1_ARTIFACT_ALREADY_EXISTS: {[f.name for f in stale]}")


# ===========================================================================
# 2. 方向构建
# ===========================================================================
def build_direction(score_mu: np.ndarray) -> np.ndarray:
    """PGM 方向：d_t = sign(score_mu,t)。"""
    return np.sign(np.asarray(score_mu, dtype=np.float64)).astype(np.int8)


def reverse_direction(direction: np.ndarray) -> np.ndarray:
    """反向 PGM 对照：d_t -> -d_t。"""
    return (-np.asarray(direction)).astype(np.int8)


def random_direction(n: int, seed: int = RANDOM_SEED) -> np.ndarray:
    """随机方向对照：固定 seed，确定性可复现。"""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=int(n)).astype(np.int8)


def build_all_directions(direction: np.ndarray) -> Dict[str, np.ndarray]:
    """一次性构建 A/B/C 三个方向组。"""
    d = np.asarray(direction).astype(np.int8)
    return {
        DIRECTION_PGM: d,
        DIRECTION_REVERSE: reverse_direction(d),
        DIRECTION_RANDOM: random_direction(len(d), RANDOM_SEED),
    }


# ===========================================================================
# 3. 归一化未来路径（一次计算，三个持仓窗口共用）
# ===========================================================================
def build_relative_path(
    future_open: np.ndarray,
    future_high: np.ndarray,
    future_low: np.ndarray,
    future_close: np.ndarray,
    entry: np.ndarray,
    direction: np.ndarray,
    atr0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """把未来 OHLC 一次性归一化为「相对入场价的方向化 R 倍数」。

    返回 (open_rel, high_rel, low_rel, close_rel)，形状均为 (N, HMAX)。
    注意：比用户设计多返回 open_rel —— 这是 CONSERVATIVE_GAP_THROUGH 规则
    （开盘已越过止损位时承担 gap loss）所必需的量，仅用于该规则。

        X_rel = d * (X - P_0) / ATR0
    """
    d = np.asarray(direction, dtype=np.float64)[:, None]
    scale = d / np.asarray(atr0, dtype=np.float64)[:, None]
    e = np.asarray(entry, dtype=np.float64)[:, None]
    open_rel = (np.asarray(future_open, dtype=np.float64) - e) * scale
    high_rel = (np.asarray(future_high, dtype=np.float64) - e) * scale
    low_rel = (np.asarray(future_low, dtype=np.float64) - e) * scale
    close_rel = (np.asarray(future_close, dtype=np.float64) - e) * scale
    return open_rel, high_rel, low_rel, close_rel


# ===========================================================================
# 4. 向量化固定 RR 模拟器（禁止 trade/bar 双重循环）
# ===========================================================================
def simulate_fixed_rr(
    high_rel: np.ndarray,
    low_rel: np.ndarray,
    close_rel: np.ndarray,
    open_rel: np.ndarray,
    hold: int,
    stop_r: float = STOP_R,
    target_r: float = TARGET_R,
    gap_rule: str = PRIMARY_GAP_RULE,
) -> Dict[str, np.ndarray]:
    """固定 1R 止损 / 2R 止盈的向量化模拟。

    规则（全部冻结）：
      - 止损位 -stop_r，止盈位 +target_r（方向化 R 倍数）
      - 同根同时触发 -> STOP FIRST
      - 未触发 -> 第 hold-1 根收盘平仓（timeout）
      - 跳空：IDEAL_FILL 固定 -stop_r / +target_r；
              CONSERVATIVE_GAP_THROUGH 在开盘已越过止损位时承担 gap loss
    """
    if hold < 1 or hold > HMAX:
        raise SystemExit(f"STOP_PGM_FIXEDRR1_HOLD_OUT_OF_RANGE: {hold}")
    if gap_rule not in GAP_RULES:
        raise SystemExit(f"STOP_PGM_FIXEDRR1_UNKNOWN_GAP_RULE: {gap_rule}")

    h = high_rel[:, :hold]
    l = low_rel[:, :hold]
    o = open_rel[:, :hold]

    stop_hit = (l <= -stop_r) | (o <= -stop_r)
    target_hit = (h >= target_r) | (o >= target_r)

    has_stop = stop_hit.any(axis=1)
    has_target = target_hit.any(axis=1)
    stop_idx = np.argmax(stop_hit, axis=1)
    target_idx = np.argmax(target_hit, axis=1)

    # STOP FIRST
    exit_stop = has_stop & ((~has_target) | (stop_idx <= target_idx))
    exit_target = has_target & (~exit_stop)

    gross = close_rel[:, hold - 1].astype(np.float64, copy=True)
    gross[exit_target] = float(target_r)

    if gap_rule == GAP_RULE_CONSERVATIVE:
        rows = np.flatnonzero(exit_stop)
        cols = stop_idx[rows]
        open_at_stop = o[rows, cols]
        gross[rows] = np.where(open_at_stop <= -stop_r, open_at_stop, -float(stop_r))
    else:
        gross[exit_stop] = -float(stop_r)

    exit_code = np.full(len(gross), TIMEOUT_EXIT, dtype=np.int8)
    exit_code[exit_target] = TARGET_EXIT
    exit_code[exit_stop] = STOP_EXIT

    exit_idx = np.full(len(gross), hold - 1, dtype=np.int32)
    exit_idx[exit_target] = target_idx[exit_target].astype(np.int32)
    exit_idx[exit_stop] = stop_idx[exit_stop].astype(np.int32)

    return dict(
        gross_return=gross,
        net_return=gross - COST_ATR0,
        exit_code=exit_code,
        exit_idx=exit_idx,
        exit_is_stop=exit_stop,
        exit_is_target=exit_target,
        exit_is_timeout=~(exit_stop | exit_target),
    )


def simulate_fixed_rr_scalar_reference(
    high_rel: np.ndarray,
    low_rel: np.ndarray,
    close_rel: np.ndarray,
    open_rel: np.ndarray,
    row_idx: int,
    hold: int,
    stop_r: float = STOP_R,
    target_r: float = TARGET_R,
    gap_rule: str = PRIMARY_GAP_RULE,
) -> Dict[str, Any]:
    """慢速纯 Python 标量参照实现，仅供测试对照，禁止用于热路径。"""
    for j in range(hold):
        lo = float(low_rel[row_idx, j])
        hi = float(high_rel[row_idx, j])
        op = float(open_rel[row_idx, j])
        stop_now = (lo <= -stop_r) or (op <= -stop_r)
        target_now = (hi >= target_r) or (op >= target_r)
        if stop_now:
            if gap_rule == GAP_RULE_CONSERVATIVE and op <= -stop_r:
                gross = op
            else:
                gross = -float(stop_r)
            return dict(gross_return=gross, net_return=gross - COST_ATR0, exit_code=STOP_EXIT, exit_idx=j)
        if target_now:
            gross = float(target_r)
            return dict(gross_return=gross, net_return=gross - COST_ATR0, exit_code=TARGET_EXIT, exit_idx=j)
    gross = float(close_rel[row_idx, hold - 1])
    return dict(gross_return=gross, net_return=gross - COST_ATR0, exit_code=TIMEOUT_EXIT, exit_idx=hold - 1)


# ===========================================================================
# 5. 区块数据准备（张量只构建一次）
# ===========================================================================
def build_block_frame(
    scored: pd.DataFrame,
    block: str,
    bars_by_sym: Dict[str, Any],
    cap: int | None = None,
) -> Dict[str, Any]:
    """准备单个 block 的评测帧与未来路径张量（n_future = HMAX，只构建一次）。"""
    assert_allowed_blocks(scored)
    eval_df = x1.extract_evaluation_sample(scored, block)
    if cap is not None and len(eval_df) > cap:
        eval_df = eval_df.iloc[:cap].copy().reset_index(drop=True)

    tensor_raw = x1.build_future_tensor(eval_df, bars_by_sym, n_future=HMAX)
    valid = tensor_raw["path_valid"]
    eval_valid = eval_df[valid].reset_index(drop=True)
    tensor = x1.slice_future_tensor(tensor_raw, valid)
    parity_err = x1.verify_baseline_parity(tensor, eval_valid["pi"].to_numpy(float))

    direction_owner = eval_valid["base_action"].to_numpy(np.float64)
    direction_derived = build_direction(eval_valid["score_mu"].to_numpy(np.float64)).astype(np.float64)
    if not np.array_equal(direction_owner, direction_derived):
        raise SystemExit("STOP_PGM_FIXEDRR1_DIRECTION_OWNER_MISMATCH")

    return dict(
        block=block,
        eval_df=eval_valid,
        tensor=tensor,
        n_raw=len(eval_df),
        n_valid=int(valid.sum()),
        disc_excluded=int(tensor_raw["disc_excluded"].sum()),
        gap_excluded=int(tensor_raw["gap_excluded"].sum()),
        day_excluded=int(tensor_raw["day_excluded"].sum()),
        baseline_parity_error=float(parity_err),
        direction=direction_owner.astype(np.int8),
        atr0=tensor["atr0"],
        entry=tensor["O"][:, 0],
        entry_day=tensor["entry_day"],
        symbol=tensor["symbol"],
    )


def evaluate_block(
    frame: Dict[str, Any],
    gap_rule: str = PRIMARY_GAP_RULE,
) -> Tuple[pd.DataFrame, Dict[str, Dict[int, np.ndarray]]]:
    """在单个 block 上评估 3 个方向 × 3 个持仓窗口。"""
    directions = build_all_directions(frame["direction"])
    N = len(frame["direction"])
    action = np.ones(N, dtype=np.float64)

    n_rows = len(DIRECTION_KINDS) * len(HOLD_GRID)
    results: List[Dict[str, Any]] = [None] * n_rows
    row_i = 0
    net_by_kind: Dict[str, Dict[int, np.ndarray]] = {k: {} for k in DIRECTION_KINDS}

    for kind, d in directions.items():
        o_rel, h_rel, l_rel, c_rel = build_relative_path(
            frame["tensor"]["O"],
            frame["tensor"]["H"],
            frame["tensor"]["L"],
            frame["tensor"]["C"],
            frame["entry"],
            d,
            frame["atr0"],
        )
        for hold in HOLD_GRID:
            sim = simulate_fixed_rr(h_rel, l_rel, c_rel, o_rel, hold, STOP_R, TARGET_R, gap_rule)
            m = d0.strategy_metrics(
                action=action,
                r_trad=sim["gross_return"],
                cost=COST_ATR0,
                entry_day=frame["entry_day"],
                symbol=frame["symbol"],
            )
            net_by_kind[kind][hold] = sim["net_return"]
            results[row_i] = dict(
                block=frame["block"],
                direction=kind,
                hold_bars=hold,
                gap_rule=gap_rule,
                n_decisions=N,
                n_trades=int(m["n_trades"]),
                gross_EV_per_decision=float(np.mean(sim["gross_return"])),
                net_EV_per_decision=float(np.mean(sim["net_return"])),
                win_rate=float(m["win_rate"]),
                mean_win=float(m["mean_win"]),
                mean_loss=float(m["mean_loss"]),
                payoff_ratio=float(m["payoff_ratio"]),
                profit_factor=float(m["profit_factor"]),
                daily_sharpe_annualized=float(m["daily_sharpe_annualized"]),
                max_drawdown_ATR0=float(m["max_drawdown_ATR0"]),
                gross_total_ATR0=float(m["gross_total_ATR0"]),
                net_total_ATR0=float(m["net_total_ATR0"]),
                positive_symbol_count=int(m["positive_symbol_count"]),
                top3_profit_share=float(m["top3_profit_share"]),
                stop_rate=float(np.mean(sim["exit_code"] == STOP_EXIT)),
                target_rate=float(np.mean(sim["exit_code"] == TARGET_EXIT)),
                timeout_rate=float(np.mean(sim["exit_code"] == TIMEOUT_EXIT)),
            )
            row_i += 1
    return pd.DataFrame(results), net_by_kind


def run_bootstrap_contrasts(
    entry_day: np.ndarray,
    net_by_kind: Dict[str, Dict[int, np.ndarray]],
    block: str,
    gap_rule: str,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    """日聚类 paired bootstrap：PGM 净值、PGM-RANDOM、PGM-REVERSE。"""
    rows = []
    for hold in HOLD_GRID:
        pgm_net = net_by_kind[DIRECTION_PGM][hold]
        rnd_net = net_by_kind[DIRECTION_RANDOM][hold]
        rev_net = net_by_kind[DIRECTION_REVERSE][hold]
        contrasts = [
            ("PGM_NET", pgm_net),
            ("PGM_MINUS_RANDOM", pgm_net - rnd_net),
            ("PGM_MINUS_REVERSE", pgm_net - rev_net),
            ("REVERSE_NET", rev_net),
            ("RANDOM_NET", rnd_net),
        ]
        for name, values in contrasts:
            b = e0.paired_day_mean_bootstrap(entry_day, values, n_boot=n_boot, seed=seed)
            rows.append(dict(block=block, hold_bars=hold, gap_rule=gap_rule, contrast=name, **b))
    return pd.DataFrame(rows)


# ===========================================================================
# 6. 源码热路径审计
# ===========================================================================
def audit_forbidden_patterns_in_hot_loops() -> None:
    """静态检查：核心函数内部禁止 iterrows / itertuples / append / 逐笔循环。"""
    src_file = pathlib.Path(__file__).resolve()
    with open(src_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(src_file))

    target_funcs = {"build_relative_path", "simulate_fixed_rr", "evaluate_block"}
    forbidden_tokens = ["iterrows", "itertuples", "append"]

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in target_funcs:
            for sub in ast.walk(node):
                if isinstance(sub, ast.Attribute) and sub.attr in forbidden_tokens:
                    raise SystemExit(f"STOP_PGM_FIXEDRR1_FORBIDDEN_HOT_LOOP: {node.name} calls {sub.attr}")
                if isinstance(sub, ast.For) and isinstance(sub.target, ast.Name):
                    if sub.target.id in {"row", "trade", "i"}:
                        raise SystemExit(
                            f"STOP_PGM_FIXEDRR1_FORBIDDEN_ROW_LOOP: {node.name} loops over {sub.target.id}"
                        )


# ===========================================================================
# 7. Audit-Only 模式（不输出任何科学结果）
# ===========================================================================
def run_audit_only() -> None:
    """仅做治理与数据一致性审计：hash、样本计数、张量形状、冻结常量、源码审计。"""
    print("=" * 70)
    print("PGM-FIXED-RR-1: AUDIT-ONLY MODE")
    print("=" * 70)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip()
    print(f"[AUDIT] HEAD={head}")
    print(f"[AUDIT] BASE_SHA={BASE_SHA}")
    if subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=str(_REPO_ROOT), capture_output=True).returncode != 0:
        raise SystemExit("STOP_PGM_FIXEDRR1_FREEZE_CHECK_FAIL")

    hashes = d0.compute_artifact_hashes()
    print(f"[AUDIT] sample_sha256={hashes['sample_artifact_sha256']}")
    print(f"[AUDIT] transition_sha256={hashes['transition_artifact_sha256']}")

    obs = n0a.load_observed_decision_universe()
    assert_allowed_blocks(obs)
    atr0_err = n0a.audit_atr0_owner_parity(obs, n0a.load_transition_truth_audit()["cur"])
    print(f"[AUDIT] max_abs_atr0_owner_error={atr0_err:.2e}")
    if atr0_err > 1e-12:
        raise SystemExit("STOP_PGM_FIXEDRR1_ATR0_PARITY_FAIL")

    print("[AUDIT] Fitting Window A & Window B (reuse frozen owners)...")
    bundle = x1.load_and_score()
    scored_by_block = {TB2_BLOCK: bundle["scored_A"], TB3_BLOCK: bundle["scored_B"]}
    bars_by_sym = bundle["bars_by_sym"]

    for blk in EVAL_BLOCKS:
        t0 = time.time()
        frame = build_block_frame(scored_by_block[blk], blk, bars_by_sym)
        t_build = time.time() - t0
        print(
            f"[AUDIT] {blk}: raw={frame['n_raw']} path_valid={frame['n_valid']} "
            f"disc_excl={frame['disc_excluded']} gap_excl={frame['gap_excluded']} "
            f"day_excl={frame['day_excluded']}"
        )
        print(
            f"[AUDIT] {blk}: tensor shapes O/H/L/C={frame['tensor']['O'].shape} "
            f"baseline_parity_err={frame['baseline_parity_error']:.2e} build={t_build:.2f}s"
        )

    print(f"[AUDIT] frozen: STOP_R={STOP_R} TARGET_R={TARGET_R} COST={COST_ATR0} "
          f"HOLD_GRID={HOLD_GRID} HMAX={HMAX} RANDOM_SEED={RANDOM_SEED}")
    print(f"[AUDIT] gap rules={GAP_RULES} primary={PRIMARY_GAP_RULE}")
    audit_forbidden_patterns_in_hot_loops()
    print("[AUDIT] Hot loop source audit: PASS")
    print("=" * 70)
    print("[AUDIT] ALL AUDIT CHECKS PASSED. NO SCIENTIFIC OUTCOME REPORTED.")
    print("=" * 70)


# ===========================================================================
# 8. Smoke 模式（只验证接线与计数，不输出科学指标）
# ===========================================================================
def run_smoke() -> None:
    """真实所有者 + 截断样本的接线验证，只打印结构性计数与耗时。"""
    print("=" * 70)
    print(f"PGM-FIXED-RR-1: SMOKE MODE (CAP={SMOKE_CAP}, WIRING ONLY)")
    print("=" * 70)
    t_start = time.time()
    bundle = x1.load_and_score()
    scored_by_block = {TB2_BLOCK: bundle["scored_A"], TB3_BLOCK: bundle["scored_B"]}
    bars_by_sym = bundle["bars_by_sym"]

    for blk in EVAL_BLOCKS:
        t0 = time.time()
        frame = build_block_frame(scored_by_block[blk], blk, bars_by_sym, cap=SMOKE_CAP)
        t_build = time.time() - t0
        df_res, net_by_kind = evaluate_block(frame, PRIMARY_GAP_RULE)
        t_eval = time.time() - t0 - t_build
        df_boot = run_bootstrap_contrasts(
            frame["entry_day"], net_by_kind, blk, PRIMARY_GAP_RULE, n_boot=200, seed=BOOTSTRAP_SEED
        )
        print(
            f"[SMOKE] {blk}: n={frame['n_valid']} rows={len(df_res)} boot_rows={len(df_boot)} "
            f"build={t_build:.2f}s eval={t_eval:.2f}s"
        )
        shapes_ok = all(
            frame["tensor"][k].shape == (frame["n_valid"], HMAX) for k in ["O", "H", "L", "C"]
        )
        print(f"[SMOKE] {blk}: tensor (N,{HMAX}) shapes ok = {shapes_ok}")
    print(f"[SMOKE] total={time.time() - t_start:.1f}s")
    print("SMOKE ONLY")
    print("NO SCIENTIFIC OUTCOME REPORTED")
    print("=" * 70)


# ===========================================================================
# 9. 正式运行模式（受环境变量保护，本轮不执行）
# ===========================================================================
def run_full() -> None:
    """正式计算并写出 3 个产物。必须由 AUTHORIZE_PGM_FIXED_RR1_RUN=1 显式授权。"""
    require_run_authorization()
    head = assert_clean_git_tree()
    assert_no_stale_artifacts()

    print("=" * 70)
    print(f"PGM-FIXED-RR-1: RUN (HEAD={head})")
    print("=" * 70)
    t_all = time.time()

    bundle = x1.load_and_score()
    scored_by_block = {TB2_BLOCK: bundle["scored_A"], TB3_BLOCK: bundle["scored_B"]}
    bars_by_sym = bundle["bars_by_sym"]

    frames = {}
    for blk in EVAL_BLOCKS:
        frames[blk] = build_block_frame(scored_by_block[blk], blk, bars_by_sym)

    all_results = []
    all_boot = []
    for gap_rule in GAP_RULES:
        for blk in EVAL_BLOCKS:
            df_res, net_by_kind = evaluate_block(frames[blk], gap_rule)
            all_results.append(df_res)
            all_boot.append(
                run_bootstrap_contrasts(frames[blk]["entry_day"], net_by_kind, blk, gap_rule)
            )

    df_results = pd.concat(all_results, ignore_index=True)
    df_bootstrap = pd.concat(all_boot, ignore_index=True)

    summary = dict(
        experiment=EXPERIMENT_NAME,
        scope=EXPERIMENT_SCOPE,
        run_head=head,
        base_sha=BASE_SHA,
        frozen=dict(
            entry="next_bar_open",
            stop_R=STOP_R,
            target_R=TARGET_R,
            reward_risk=TARGET_R / STOP_R,
            cost_ATR0=COST_ATR0,
            hold_grid=HOLD_GRID,
            hmax=HMAX,
            random_seed=RANDOM_SEED,
            gap_rules=GAP_RULES,
            primary_gap_rule=PRIMARY_GAP_RULE,
        ),
        counts={blk: dict(raw=frames[blk]["n_raw"], path_valid=frames[blk]["n_valid"]) for blk in EVAL_BLOCKS},
        baseline_parity_errors={blk: frames[blk]["baseline_parity_error"] for blk in EVAL_BLOCKS},
        bootstrap_N=BOOTSTRAP_N,
        bootstrap_seed=BOOTSTRAP_SEED,
        timing=dict(total_seconds=float(time.time() - t_all)),
        artifact_files=EXACT_ARTIFACTS,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(OUT_DIR / ARTIFACT_RESULTS, index=False)
    df_bootstrap.to_csv(OUT_DIR / ARTIFACT_BOOTSTRAP, index=False)
    with open(OUT_DIR / ARTIFACT_SUMMARY, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    disk_names = sorted(p.name for p in OUT_DIR.glob(f"{ARTIFACT_PREFIX}*"))
    if disk_names != sorted(EXACT_ARTIFACTS):
        raise SystemExit(f"STOP_PGM_FIXEDRR1_DISK_ARTIFACT_SET_MISMATCH: {disk_names}")

    print(df_results.to_string(index=False))
    print(df_bootstrap.to_string(index=False))
    print(f"[RUN] completed in {summary['timing']['total_seconds']:.1f}s")


# ===========================================================================
# 10. CLI
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--audit-only", action="store_true", help="Governance / data audit only")
    parser.add_argument("--smoke", action="store_true", help="Wiring smoke on capped sample")
    parser.add_argument("--run", action="store_true", help="Authorized full run")
    args = parser.parse_args()

    if args.audit_only:
        run_audit_only()
    elif args.smoke:
        run_smoke()
    elif args.run:
        run_full()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
