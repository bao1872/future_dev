"""
experiment_pgm_native0a_one_step_alpha_v1.py
===========================================

PGM-NATIVE-0A -- One-Step Tradable Alpha Probe

PURPOSE
-------
Directly test whether the current DYNAMIC-PGM 5m transition prediction has
independent, tradable economic value after applying next-open execution constraints.

Core research question:
    "After conditioning on state S_t at bar t close, does the PGM's transition
     prediction retain tradeable edge when execution can only occur at bar t+1 open?"

The transition model predicts:
    z_{d, t+1} = d_{U, t+1} - d_{U, t}
    r^{CC}_{t+1} = (C_{t+1} - C_t) / ATR0_t == -z_{d, t+1}

The close-to-close return decomposes into:
    r^{CC}_{t+1} = g_{t+1} + r^{trad}_{t+1}
where:
    g_{t+1} = (O_{t+1} - C_t) / ATR0_t      (untradeable next-open gap)
    r^{trad}_{t+1} = (C_{t+1} - O_{t+1}) / ATR0_t  (tradeable open-to-close return)

Primary PGM score:
    s_t = -mu_{z, t} = -E[z_{d, t+1} | S_t]
Native action:
    a_t = +1 if s_t > 0 else (-1 if s_t < 0 else 0)
Trade return:
    pi_t = a_t * r^{trad}_{t+1}

This experiment does NOT:
  * build a new backtester,
  * train any economic model (no Ridge, no HGB, no RL),
  * search thresholds / stop / target / horizons,
  * prune symbols,
  * use TB3 to tune any parameter,
  * depend on V2 opportunities or R1-R4.

EXPERIMENT SCOPE (hard-coded, never escalated)
------------------------------------------------
EXPERIMENT_SCOPE = "PGM_NATIVE_WITHIN_EPISODE_ONE_STEP_TRADABLE_ALPHA_PROBE"
Forbidden verdicts: {"PROFITABLE_STRATEGY", "LIVE_READY", "FINAL_STRATEGY"}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for _bt in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_bt] = "1"

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import scipy.stats

import research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 as ex0
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1c_free_run_rollout_v1 as pgm
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base


# ===========================================================================
# Governance Constants
# ===========================================================================
BASE_SHA = "2bae5c6a86d8c563027597bd9a54b66fc824b5b4"
EXPERIMENT_NAME = "PGM-NATIVE-0A -- One-Step Tradable Alpha Probe"
EXPERIMENT_SCOPE = "PGM_NATIVE_WITHIN_EPISODE_ONE_STEP_TRADABLE_ALPHA_PROBE"
PREFIX = "pgm_native0a"

FORBIDDEN_VERDICTS = {"PROFITABLE_STRATEGY", "LIVE_READY", "FINAL_STRATEGY"}

COST_ATR0_GRID = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260915

TB2_BLOCK = "TB2"
TB3_BLOCK = "TB3"


# ===========================================================================
# Reuse Map
# ===========================================================================
REUSE_MAP = {
    "transition_sample": "pgm.TRANSITION_SAMPLE_PATH (cache/dynamic_pgm1a2c_transitions.parquet) + base.build_transition_sample(pgm.SAMPLE_PATH)",
    "pgm_fit": "pgm.fit_samplers_for_window(pgm.WINDOWS[0/1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)",
    "deterministic_pgm_score": "mc.analytic_conditional_support(df)['z_d_up_mu'] -> score_mu = -z_d_up_mu; pgm.compute_state_conditional_support_probs -> score_break_skew",
    "raw_bars": "ex0.load_env() -> bars_by_sym",
    "next_open_semantics": "t = bar_t, entry = t + 1, C0 = c[t], O1 = o[t+1], C1 = c[t+1]",
    "discontinuity_semantics": "bars['disc'][t+1] -> ENTRY_UNAVAILABLE",
}


def print_reuse_map() -> None:
    print("==================================================", flush=True)
    print("REUSE MAP:", flush=True)
    for k, v in REUSE_MAP.items():
        print(f"  REUSE {k}: {v}", flush=True)
    print("==================================================", flush=True)


# ===========================================================================
# Transition Decision Universe Loader & Auditor
# ===========================================================================
def load_transition_universe() -> pd.DataFrame:
    """Load within-episode decision universe with exact bar_t, atr0, hazard columns.

    Priority reads pgm.TRANSITION_SAMPLE_PATH. If bar_t / atr0 / hazard are not
    persisted in the parquet, dynamically restores them from base.build_transition_sample(pgm.SAMPLE_PATH).
    Enforces exact 1-to-1 row-by-row alignment.
    """
    trans = pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH)
    if "bar_t" not in trans.columns or "atr0" not in trans.columns or "hazard" not in trans.columns:
        s = pd.read_parquet(pgm.SAMPLE_PATH)
        cur, _ = base.build_transition_sample(s)
        if len(cur) != len(trans) or not (cur["episode_id"].values == trans["episode_id"].values).all():
            raise SystemExit("STOP_PGM_NATIVE_TRANSITION_ALIGNMENT_FAIL")
        trans["bar_t"] = cur["bar_t"].values
        trans["atr0"] = cur["atr0"].values
        trans["hazard"] = cur["hazard"].values
        trans["z_d_up"] = cur["z_d_up"].values
    return trans


def audit_decision_universe(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit mathematical and contractual integrity of the transition decision universe."""
    if not (df["hazard"] == 0).all():
        raise SystemExit("STOP_PGM_NATIVE_HAZARD_NOT_ZERO")

    if not np.all(np.isfinite(df["z_d_up"].to_numpy(float))):
        raise SystemExit("STOP_PGM_NATIVE_ZDUP_NONFINITE")

    atr0 = df["atr0"].to_numpy(float)
    if not np.all(np.isfinite(atr0)) or not (atr0 > 0).all():
        raise SystemExit("STOP_PGM_NATIVE_ATR0_INVALID")

    # Check uniqueness of (symbol, bar_t)
    dup_mask = df.duplicated(subset=["symbol", "bar_t"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        cols = [c for c in ["symbol", "bar_t", "episode_id"] if c in df.columns]
        dups = df[dup_mask][cols].head(10)
        print(f"STOP_PGM_NATIVE_DUPLICATE_DECISION_KEY: found {dup_count} duplicate rows:\n{dups}")
        raise SystemExit("STOP_PGM_NATIVE_DUPLICATE_DECISION_KEY")

    row_counts_by_block = df["block"].value_counts().to_dict()
    symbols = sorted(df["symbol"].unique().tolist())

    return dict(
        n_rows=len(df),
        duplicate_keys=dup_count,
        hazard_zero=(df["hazard"] == 0).all(),
        row_counts_by_block=row_counts_by_block,
        n_symbols=len(symbols),
        symbols=symbols,
    )


# ===========================================================================
# Raw Bar Economic Alignment & Return Decomposition
# ===========================================================================
def align_raw_bars_and_returns(df: pd.DataFrame, bars_by_sym: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align transition decision rows with raw 5m bars, verify return parity and decomposition.

    t = bar_t
    entry_bar = t + 1
    If entry_bar >= n or bars['disc'][entry_bar] -> ENTRY_UNAVAILABLE.
    For valid rows:
      C0 = c[t]
      O1 = o[t+1]
      C1 = c[t+1]
      r_state_CC_ATR0 = (C1 - C0) / atr0
      gap_ATR0        = (O1 - C0) / atr0
      r_trad_OC_ATR0  = (C1 - O1) / atr0
    Hard assertions:
      |r_state_CC_ATR0 - (-z_d_up)| <= 1e-8
      |r_state_CC_ATR0 - (gap_ATR0 + r_trad_OC_ATR0)| <= 1e-10
    """
    out_dfs = []
    n_checked = 0
    n_unavailable = 0
    n_disc = 0
    max_err_cc = 0.0
    max_err_decomp = 0.0

    for sym, g in df.groupby("symbol", sort=False):
        if sym not in bars_by_sym:
            raise SystemExit(f"STOP_PGM_NATIVE_MISSING_SYMBOL_BARS: {sym}")
        bars = bars_by_sym[sym]
        t = g["bar_t"].to_numpy(int)
        e = t + 1
        n_bars = bars["n"]
        disc = bars["disc"]

        in_range = e < n_bars
        is_disc = np.zeros(len(e), dtype=bool)
        is_disc[in_range] = disc[e[in_range]]

        valid = in_range & (~is_disc)
        n_unavailable += int((~valid).sum())
        n_disc += int(is_disc.sum())

        g_out = g.copy()
        g_out["is_entry_valid"] = valid
        g_out["entry_bar"] = np.where(valid, e, -1)
        g_out["entry_day"] = np.where(valid, bars["day"][np.maximum(e, 0)], pd.NaT)

        g_val = g_out[valid]
        t_val = t[valid]
        e_val = e[valid]

        c0 = bars["c"][t_val]
        o1 = bars["o"][e_val]
        c1 = bars["c"][e_val]
        atr = g_val["atr0"].to_numpy(float)

        r_cc = (c1 - c0) / atr
        r_gap = (o1 - c0) / atr
        r_trad = (c1 - o1) / atr

        # Exact parity checks
        z_dup = g_val["z_d_up"].to_numpy(float)
        err_cc = np.abs(r_cc - (-z_dup))
        err_decomp = np.abs(r_cc - (r_gap + r_trad))

        loc_max_cc = float(np.max(err_cc)) if len(err_cc) > 0 else 0.0
        loc_max_decomp = float(np.max(err_decomp)) if len(err_decomp) > 0 else 0.0

        if loc_max_cc > 1e-8:
            raise SystemExit(f"STOP_PGM_NATIVE_RAW_RETURN_SEMANTIC_MISMATCH: max_err={loc_max_cc}")
        if loc_max_decomp > 1e-10:
            raise SystemExit(f"STOP_PGM_NATIVE_RETURN_DECOMPOSITION_MISMATCH: max_err={loc_max_decomp}")

        max_err_cc = max(max_err_cc, loc_max_cc)
        max_err_decomp = max(max_err_decomp, loc_max_decomp)
        n_checked += len(g_val)

        g_out.loc[valid, "r_state_CC_ATR0"] = r_cc
        g_out.loc[valid, "gap_ATR0"] = r_gap
        g_out.loc[valid, "r_trad_OC_ATR0"] = r_trad
        out_dfs.append(g_out)

    res_df = pd.concat(out_dfs, ignore_index=True)
    audit_res = dict(
        n_checked=n_checked,
        n_unavailable=n_unavailable,
        n_disc=n_disc,
        max_err_r_cc=max_err_cc,
        max_err_decomp=max_err_decomp,
    )
    return res_df, audit_res


# ===========================================================================
# PGM Model Scoring
# ===========================================================================
def score_pgm_block(df_block: pd.DataFrame, trans_sampler: Any) -> pd.DataFrame:
    """Compute primary directional score and secondary break skew on block rows."""
    mom = trans_sampler.analytic_conditional_support(df_block)
    risk = pgm.compute_state_conditional_support_probs(df_block, trans_sampler)

    out = df_block.copy()
    out["score_mu"] = -np.asarray(mom["z_d_up_mu"], dtype=np.float64)
    out["score_break_skew"] = (
        np.asarray(risk["p_up_distance_negative"], dtype=np.float64)
        - np.asarray(risk["p_down_distance_negative"], dtype=np.float64)
    )

    # Native action
    s = out["score_mu"].to_numpy(float)
    actions = np.where(s > 0, 1, np.where(s < 0, -1, 0))
    out["action"] = actions

    if "r_trad_OC_ATR0" in out.columns:
        out["strategy_return_ATR0"] = out["action"] * out["r_trad_OC_ATR0"]

    return out


# ===========================================================================
# Metrics, Deciles, Cost Stress, and Clustered Bootstrap
# ===========================================================================
def compute_block_metrics(df_valid: pd.DataFrame) -> Dict[str, Any]:
    """Compute prediction, tradability, gap loss, and sign policy metrics."""
    score = df_valid["score_mu"].to_numpy(float)
    r_cc = df_valid["r_state_CC_ATR0"].to_numpy(float)
    r_trad = df_valid["r_trad_OC_ATR0"].to_numpy(float)
    gap = df_valid["gap_ATR0"].to_numpy(float)
    action = df_valid["action"].to_numpy(int)
    strat_ret = df_valid["strategy_return_ATR0"].to_numpy(float)

    # A. Structural prediction
    spearman_state = float(scipy.stats.spearmanr(score, r_cc).statistic)

    # B. Tradable prediction
    spearman_trad = float(scipy.stats.spearmanr(score, r_trad).statistic)

    # C. Gap capture loss
    spearman_gap = float(scipy.stats.spearmanr(score, gap).statistic)
    mean_abs_gap = float(np.mean(np.abs(gap)))
    mean_abs_trad = float(np.mean(np.abs(r_trad)))

    # D. Sign policy
    n_signals = len(df_valid)
    n_long = int((action == 1).sum())
    n_short = int((action == -1).sum())
    n_skip = int((action == 0).sum())
    n_trades = n_long + n_short

    gross_ev_signal = float(np.mean(strat_ret))
    gross_ev_trade = float(np.mean(strat_ret[action != 0])) if n_trades > 0 else 0.0
    total_atr0 = float(np.sum(strat_ret))

    trade_mask = action != 0
    trade_rets = strat_ret[trade_mask]
    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets < 0]

    win_rate = float(len(wins) / len(trade_rets)) if len(trade_rets) > 0 else 0.0
    mean_win = float(np.mean(wins)) if len(wins) > 0 else 0.0
    mean_loss = float(np.mean(losses)) if len(losses) > 0 else 0.0
    payoff_ratio = float(abs(mean_win / mean_loss)) if abs(mean_loss) > 1e-12 else 0.0
    profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if abs(np.sum(losses)) > 1e-12 else 0.0

    return dict(
        spearman_state=spearman_state,
        spearman_trad=spearman_trad,
        spearman_gap=spearman_gap,
        mean_abs_gap=mean_abs_gap,
        mean_abs_trad=mean_abs_trad,
        n_signals=n_signals,
        n_long=n_long,
        n_short=n_short,
        n_skip=n_skip,
        gross_ev_signal=gross_ev_signal,
        gross_ev_trade=gross_ev_trade,
        total_atr0=total_atr0,
        win_rate=win_rate,
        mean_win=mean_win,
        mean_loss=mean_loss,
        payoff_ratio=payoff_ratio,
        profit_factor=profit_factor,
    )


def compute_decile_edges(scores: np.ndarray) -> np.ndarray:
    """Compute 10 decile edges from calibration scores."""
    quantiles = np.linspace(0.1, 0.9, 9)
    inner_edges = np.quantile(scores, quantiles)
    edges = np.concatenate([[-np.inf], inner_edges, [np.inf]])
    return edges


def evaluate_decile_bins(df_valid: pd.DataFrame, edges: np.ndarray) -> List[Dict[str, Any]]:
    """Evaluate performance across decile bins using fixed edges."""
    score = df_valid["score_mu"].to_numpy(float)
    bins = pd.cut(score, bins=edges, labels=False, include_lowest=True)
    df_eval = df_valid.copy()
    df_eval["decile_bin"] = bins

    rows = []
    for b in range(10):
        sub = df_eval[df_eval["decile_bin"] == b]
        n_b = len(sub)
        if n_b > 0:
            rows.append(dict(
                bin_idx=b,
                n=n_b,
                score_mean=float(sub["score_mu"].mean()),
                r_state_CC_mean=float(sub["r_state_CC_ATR0"].mean()),
                r_trad_OC_mean=float(sub["r_trad_OC_ATR0"].mean()),
                sign_policy_EV=float(sub["strategy_return_ATR0"].mean()),
            ))
        else:
            rows.append(dict(
                bin_idx=b, n=0, score_mean=0.0,
                r_state_CC_mean=0.0, r_trad_OC_mean=0.0, sign_policy_EV=0.0,
            ))
    return rows


def run_cost_stress_grid(df_valid: pd.DataFrame) -> List[Dict[str, Any]]:
    """Evaluate net return across normalized friction grid COST_ATR0_GRID."""
    strat_ret = df_valid["strategy_return_ATR0"].to_numpy(float)
    is_traded = (df_valid["action"].to_numpy(int) != 0).astype(float)
    trade_rate = float(np.mean(is_traded))
    gross_ev = float(np.mean(strat_ret))
    break_even = gross_ev / trade_rate if trade_rate > 1e-12 else 0.0

    rows = []
    for cost in COST_ATR0_GRID:
        net_ret = strat_ret - cost * is_traded
        net_ev_signal = float(np.mean(net_ret))
        net_ev_trade = float(np.mean(net_ret[is_traded == 1])) if is_traded.sum() > 0 else 0.0
        total_net = float(np.sum(net_ret))
        rows.append(dict(
            cost_ATR0=cost,
            net_EV_per_signal_ATR0=net_ev_signal,
            net_EV_per_trade_ATR0=net_ev_trade,
            total_net_ATR0=total_net,
            break_even_cost_ATR0=break_even,
        ))
    return rows


def run_day_clustered_bootstrap(df_valid: pd.DataFrame,
                                n_boot: int = BOOTSTRAP_N,
                                seed: int = BOOTSTRAP_SEED) -> Dict[str, Dict[str, float]]:
    """Trading-day clustered bootstrap for rho_state, rho_trad, and EV_sign.

    Resamples unique trading days with replacement, evaluates the metric on the
    concatenated rows of sampled days. Returns point, CI95 lower, CI95 upper, P(>0).
    """
    days = df_valid["entry_day"].to_numpy()
    score = df_valid["score_mu"].to_numpy(float)
    r_cc = df_valid["r_state_CC_ATR0"].to_numpy(float)
    r_trad = df_valid["r_trad_OC_ATR0"].to_numpy(float)
    strat_ret = df_valid["strategy_return_ATR0"].to_numpy(float)

    # Point estimates
    pt_rho_state = float(scipy.stats.spearmanr(score, r_cc).statistic)
    pt_rho_trad = float(scipy.stats.spearmanr(score, r_trad).statistic)
    pt_ev_sign = float(np.mean(strat_ret))

    # Pre-index days
    unique_days, inverse = np.unique(days, return_inverse=True)
    n_days = len(unique_days)
    day_indices = [np.flatnonzero(inverse == d) for d in range(n_days)]

    rng = np.random.default_rng(seed)

    boot_rho_state = np.empty(n_boot, dtype=np.float64)
    boot_rho_trad = np.empty(n_boot, dtype=np.float64)
    boot_ev_sign = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_d = rng.choice(n_days, size=n_days, replace=True)
        idx_b = np.concatenate([day_indices[d] for d in sampled_d])

        s_b = score[idx_b]
        r_cc_b = r_cc[idx_b]
        r_trad_b = r_trad[idx_b]
        ret_b = strat_ret[idx_b]

        boot_rho_state[b] = scipy.stats.spearmanr(s_b, r_cc_b).statistic
        boot_rho_trad[b] = scipy.stats.spearmanr(s_b, r_trad_b).statistic
        boot_ev_sign[b] = np.mean(ret_b)

    def _summarize(pt: float, dist: np.ndarray) -> Dict[str, float]:
        lo = float(np.percentile(dist, 2.5))
        hi = float(np.percentile(dist, 97.5))
        p_pos = float(np.mean(dist > 0))
        return dict(point=pt, ci95_lower=lo, ci95_upper=hi, p_pos=p_pos)

    return dict(
        rho_state=_summarize(pt_rho_state, boot_rho_state),
        rho_trad=_summarize(pt_rho_trad, boot_rho_trad),
        EV_sign=_summarize(pt_ev_sign, boot_ev_sign),
    )


def compute_symbol_breadth(df_valid: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Compute per-symbol metrics across all 15 instruments."""
    symbols = sorted(df_valid["symbol"].unique().tolist())
    rows = []
    n_pos_ev = 0
    n_pos_rho_trad = 0

    for sym in symbols:
        sub = df_valid[df_valid["symbol"] == sym]
        n_sym = len(sub)
        if n_sym == 0:
            continue

        score = sub["score_mu"].to_numpy(float)
        r_cc = sub["r_state_CC_ATR0"].to_numpy(float)
        r_trad = sub["r_trad_OC_ATR0"].to_numpy(float)
        action = sub["action"].to_numpy(int)
        strat_ret = sub["strategy_return_ATR0"].to_numpy(float)

        rho_state = float(scipy.stats.spearmanr(score, r_cc).statistic)
        rho_trad = float(scipy.stats.spearmanr(score, r_trad).statistic)
        ev_sign = float(np.mean(strat_ret))

        trade_mask = action != 0
        trade_rets = strat_ret[trade_mask]
        wins = trade_rets[trade_rets > 0]
        losses = trade_rets[trade_rets < 0]

        win_rate = float(len(wins) / len(trade_rets)) if len(trade_rets) > 0 else 0.0
        profit_factor = float(np.sum(wins) / abs(np.sum(losses))) if abs(np.sum(losses)) > 1e-12 else 0.0
        long_share = float(np.mean(action == 1))
        short_share = float(np.mean(action == -1))

        if ev_sign > 0:
            n_pos_ev += 1
        if rho_trad > 0:
            n_pos_rho_trad += 1

        rows.append(dict(
            symbol=sym,
            n=n_sym,
            rho_state=rho_state,
            rho_trad=rho_trad,
            EV_sign=ev_sign,
            win_rate=win_rate,
            profit_factor=profit_factor,
            long_share=long_share,
            short_share=short_share,
        ))

    counts = dict(
        n_symbols_total=len(symbols),
        n_symbols_positive_EV=n_pos_ev,
        n_symbols_positive_rho_trad=n_pos_rho_trad,
    )
    return rows, counts


# ===========================================================================
# Pipeline Modes: Audit-Only and Smoke
# ===========================================================================
def run_audit_only() -> None:
    """Execute static & data audit without fitting PGM models."""
    print("==================================================", flush=True)
    print("PGM-NATIVE-0A: AUDIT-ONLY EXECUTION", flush=True)
    print("==================================================", flush=True)

    # 1. Base HEAD verification
    try:
        git_head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True
        ).strip()
    except Exception as e:
        git_head = f"UNKNOWN: {e}"

    print(f"[AUDIT] EXPECTED BASE SHA: {BASE_SHA}")
    print(f"[AUDIT] CURRENT GIT HEAD : {git_head}")
    if git_head != BASE_SHA:
        print(f"WARNING: Current HEAD {git_head} != BASE_SHA {BASE_SHA}")

    # 2. REUSE MAP
    print_reuse_map()

    # 3. Transition artifact audit
    trans_path = pgm.TRANSITION_SAMPLE_PATH
    print(f"[AUDIT] Transition artifact path: {trans_path}")
    if not trans_path.exists():
        raise SystemExit(f"STOP_PGM_NATIVE_TRANSITION_FILE_NOT_FOUND: {trans_path}")

    h_sha = hashlib.sha256(trans_path.read_bytes()).hexdigest()
    print(f"[AUDIT] Transition artifact sha256: {h_sha}")

    trans = load_transition_universe()
    aud_res = audit_decision_universe(trans)
    print(f"[AUDIT] Decision universe rows: {aud_res['n_rows']}")
    print(f"[AUDIT] Duplicate (symbol, bar_t) count: {aud_res['duplicate_keys']}")
    print(f"[AUDIT] All hazard == 0: {aud_res['hazard_zero']}")
    print(f"[AUDIT] Block row counts: {aud_res['row_counts_by_block']}")
    print(f"[AUDIT] Total symbols: {aud_res['n_symbols']} {aud_res['symbols']}")

    # 4. Raw bar economic alignment audit
    print("[AUDIT] Loading bars_by_sym via ex0.load_env()...", flush=True)
    _, _, bars_by_sym = ex0.load_env()

    aligned_df, align_aud = align_raw_bars_and_returns(trans, bars_by_sym)
    print(f"[AUDIT] Checked valid rows: {align_aud['n_checked']}")
    print(f"[AUDIT] Entry unavailable count: {align_aud['n_unavailable']}")
    print(f"[AUDIT] Discontinuity count: {align_aud['n_disc']}")
    print(f"[AUDIT] Max abs error (r_cc == -z_d_up): {align_aud['max_err_r_cc']:.2e} (target <= 1e-8)")
    print(f"[AUDIT] Max abs error (r_cc == gap + trad): {align_aud['max_err_decomp']:.2e} (target <= 1e-10)")

    # 5. Coverage by block x symbol
    cov = aligned_df[aligned_df["is_entry_valid"]].groupby(["block", "symbol"]).size().unstack(fill_value=0)
    print("[AUDIT] Valid coverage by block x symbol:")
    print(cov.to_string())

    # 6. Forbidden dependencies check
    script_content = Path(__file__).read_text()
    forbidden_terms = [
        "execution_" + "lag1_trades.parquet",
        "reward_" + "SKIP",
        "reward_" + "MARKET",
        "reward_" + "LIMIT_RR3",
        "reward_" + "REASSESS_RR3",
        "fit_" + "action_q_models",
        "apply_" + "action_policy",
    ]
    for term in forbidden_terms:
        if term in script_content:
            raise SystemExit(f"STOP_PGM_NATIVE_FORBIDDEN_DEPENDENCY_FOUND: {term}")
    print("[AUDIT] Forbidden dependencies check: ALL CLEAN (no V2/R1-R4/Q-models).")
    print("[AUDIT] Complete. Stopping before model fit.", flush=True)


def run_smoke_test() -> None:
    """Execute lightweight end-to-end smoke test on <= 512 rows per block."""
    print("==================================================", flush=True)
    print("PGM-NATIVE-0A: SMOKE TEST EXECUTION (<=512 rows/block)", flush=True)
    print("==================================================", flush=True)
    t0 = time.perf_counter()

    # 1. Load universe & raw bars
    trans = load_transition_universe()
    _, _, bars_by_sym = ex0.load_env()
    aligned_df, _ = align_raw_bars_and_returns(trans, bars_by_sym)

    # Subsample TB2 and TB3 to 512 rows each for smoke
    tb2_sub = aligned_df[(aligned_df["block"] == TB2_BLOCK) & aligned_df["is_entry_valid"]].sample(n=512, random_state=42).copy().reset_index(drop=True)
    tb3_sub = aligned_df[(aligned_df["block"] == TB3_BLOCK) & aligned_df["is_entry_valid"]].sample(n=512, random_state=42).copy().reset_index(drop=True)

    print(f"[SMOKE] Subsampled TB2: {len(tb2_sub)} rows ({tb2_sub['symbol'].nunique()} syms), TB3: {len(tb3_sub)} rows ({tb3_sub['symbol'].nunique()} syms)")

    # 2. Fit Window A (TB1 -> score TB2)
    print("[SMOKE] Fitting Window A samplers...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc_A = fit_A["trans_samplers"]["MC_STATE_CURREENCODING"]
    tb2_scored = score_pgm_block(tb2_sub, mc_A)

    # 3. Fit Window B (TB1+TB2 -> score TB3)
    print("[SMOKE] Fitting Window B samplers...", flush=True)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc_B = fit_B["trans_samplers"]["MC_STATE_CURREENCODING"]
    tb3_scored = score_pgm_block(tb3_sub, mc_B)

    # 4. Metrics
    m_tb2 = compute_block_metrics(tb2_scored)
    m_tb3 = compute_block_metrics(tb3_scored)
    print(f"[SMOKE TB2] Spearman(state)={m_tb2['spearman_state']:.4f}, Spearman(trad)={m_tb2['spearman_trad']:.4f}, EV_sign={m_tb2['gross_ev_signal']:.4f}")
    print(f"[SMOKE TB3] Spearman(state)={m_tb3['spearman_state']:.4f}, Spearman(trad)={m_tb3['spearman_trad']:.4f}, EV_sign={m_tb3['gross_ev_signal']:.4f}")

    # 5. Decile monotonicity
    edges_tb2 = compute_decile_edges(tb2_scored["score_mu"].to_numpy(float))
    dec_tb2 = evaluate_decile_bins(tb2_scored, edges_tb2)
    dec_tb3 = evaluate_decile_bins(tb3_scored, edges_tb2)
    print(f"[SMOKE] Decile binning verified (10 bins for TB2 and TB3 using TB2 edges).")

    # 6. Cost stress
    stress_tb2 = run_cost_stress_grid(tb2_scored)
    stress_tb3 = run_cost_stress_grid(tb3_scored)
    print(f"[SMOKE] Cost stress grid verified: break-even cost TB2={stress_tb2[0]['break_even_cost_ATR0']:.4f}, TB3={stress_tb3[0]['break_even_cost_ATR0']:.4f}")

    # 7. Day-clustered bootstrap (smoke with 200 reps)
    print("[SMOKE] Running day-clustered bootstrap (200 reps)...", flush=True)
    boot_tb3 = run_day_clustered_bootstrap(tb3_scored, n_boot=200, seed=BOOTSTRAP_SEED)
    print(f"[SMOKE TB3 Boot] rho_state: pt={boot_tb3['rho_state']['point']:.4f}, CI=[{boot_tb3['rho_state']['ci95_lower']:.4f}, {boot_tb3['rho_state']['ci95_upper']:.4f}]")
    print(f"[SMOKE TB3 Boot] rho_trad : pt={boot_tb3['rho_trad']['point']:.4f}, CI=[{boot_tb3['rho_trad']['ci95_lower']:.4f}, {boot_tb3['rho_trad']['ci95_upper']:.4f}]")
    print(f"[SMOKE TB3 Boot] EV_sign  : pt={boot_tb3['EV_sign']['point']:.4f}, CI=[{boot_tb3['EV_sign']['ci95_lower']:.4f}, {boot_tb3['EV_sign']['ci95_upper']:.4f}]")

    # 8. Symbol breadth
    sym_rows, sym_counts = compute_symbol_breadth(tb3_scored)
    print(f"[SMOKE TB3 Breadth] Positive EV symbols: {sym_counts['n_symbols_positive_EV']} / {sym_counts['n_symbols_total']}")

    elapsed = time.perf_counter() - t0
    print(f"[SMOKE COMPLETE] Successfully executed in {elapsed:.2f}s! (No formal verdict emitted)", flush=True)


def main():
    parser = argparse.ArgumentParser(description=EXPERIMENT_NAME)
    parser.add_argument("--audit-only", action="store_true", help="Run static and structural audit without fitting models.")
    parser.add_argument("--smoke", action="store_true", help="Run lightweight end-to-end smoke test.")
    parser.add_argument("--formal", action="store_true", help="Execute formal full evaluation (BLOCKED THIS ROUND).")
    args = parser.parse_args()

    if args.formal:
        raise SystemExit(
            "STOP_PGM_NATIVE_FORMAL_NOT_AUTHORIZED_THIS_ROUND:\n"
            "本轮未授权运行 --formal。请先提交代码与测试由用户完成独立审计，获得明确授权后再运行。"
        )

    if args.audit_only:
        run_audit_only()
        return

    if args.smoke:
        run_smoke_test()
        return

    # Default if no flags: show help
    parser.print_help()


if __name__ == "__main__":
    main()
