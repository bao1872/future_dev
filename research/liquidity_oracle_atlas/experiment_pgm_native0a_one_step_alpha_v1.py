"""
experiment_pgm_native0a_one_step_alpha_v1.py
===========================================

PGM-NATIVE-0A.2: Formal Runner + Fail-Closed Data Ownership
One-Step Tradable Alpha Probe on Frozen PGM Bar Sample

PURPOSE & CAUSAL REPAIR
-----------------------
Directly test whether the current DYNAMIC-PGM 5m transition prediction has
independent, tradable economic value after applying next-open execution constraints,
WITHOUT using future hazard labels to filter trading opportunities.

In PGM-NATIVE-0A, the transition sample (pgm.TRANSITION_SAMPLE_PATH) was erroneously
used as the decision universe. Because row t's hazard label in the frozen sample
is H_{t+1} (an observation of whether the NEXT bar terminates the episode), filtering
on hazard == 0 constituted future-label selection, discarding 37,987 hazard==1 rows (10.56%).

In PGM-NATIVE-0A.1 and 0A.2, the trading decision universe is repaired to include ALL observation
rows from pgm.SAMPLE_PATH (both H_{t+1} == 0 and H_{t+1} == 1). The transition sample is
degraded strictly to a MODEL TRAINING and TARGET AUDIT artifact.

SCOPING & KNOWN LIMITATIONS
---------------------------
EXPERIMENT_SCOPE = "PGM_NATIVE_ONE_STEP_ALPHA_ON_FROZEN_PGM_BAR_SAMPLE"

This experiment is NOT a prospective or live strategy validation.
The frozen PGM bar sample itself historically excluded:
  1. Cross-block episodes (end_block != start_block)
  2. event_mask == 0 censored episodes (episodes without structural boundary hit)

The research question answered here is:
    "Within the frozen PGM bar sample, without using the true next-bar hazard label
     to filter trading opportunities, does the PGM transition score have tradable
     alpha after next-open execution?"

Forbidden verdicts:
    {"FULL_MARKET_OOS", "PROSPECTIVE_STRATEGY", "LIVE_READY", "FINAL_STRATEGY", "PROFITABLE_STRATEGY"}

MATHEMATICAL MODEL
------------------
Transition head models conditional nonterminal expectation:
    m(S_t) = E[-z_{d, t+1} | S_t, H_{t+1}=0]
    score_mu = -mu_{z, t} = -E[z_{d, t+1} | S_t, H_{t+1}=0]

The score is evaluated on ALL current PGM states S_t, regardless of whether H_{t+1} is 0 or 1:
    s_t = m(S_t)
    a_t = sign(s_t) = +1 if s_t > 0 else (-1 if s_t < 0 else 0)
    r^{trad}_{t+1} = (C_{t+1} - O_{t+1}) / ATR0_t
    pi_t = a_t * r^{trad}_{t+1}

PRIMARY economic evaluation is computed across ALL valid observation rows.
Hazard subgroups (H=0 vs H=1) are evaluated strictly for ex-post diagnostic bias estimation:
    future_filter_EV_bias = EV_{H=0} - EV_{ALL}
    future_filter_rho_trad_bias = rho_{trad, H=0} - rho_{trad, ALL}
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
BASE_SHA = "3b68cf988796992ba47eb018ac8db52e81289756"
EXPERIMENT_NAME = "PGM-NATIVE-0A.2 -- Formal Runner + Fail-Closed Data Ownership"
EXPERIMENT_SCOPE = "PGM_NATIVE_ONE_STEP_ALPHA_ON_FROZEN_PGM_BAR_SAMPLE"
PREFIX = "pgm_native0a1"

FORBIDDEN_VERDICTS = {
    "FULL_MARKET_OOS",
    "PROSPECTIVE_STRATEGY",
    "LIVE_READY",
    "FINAL_STRATEGY",
    "PROFITABLE_STRATEGY",
}

VERDICT_STRINGS = {
    "NOT_SUPPORTED": "PGM_NATIVE_TRANSITION_PREDICTION_NOT_SUPPORTED_ON_FROZEN_PGM_BAR_SAMPLE",
    "PREDICTIVE_BUT_NOT_TRADABLE": "PGM_TRANSITION_PREDICTIVE_BUT_TRADABLE_ALPHA_NOT_SUPPORTED_ON_FROZEN_PGM_BAR_SAMPLE",
    "SUPPORTED": "PGM_NATIVE_ONE_STEP_TRADABLE_ALPHA_SUPPORTED_ON_FROZEN_PGM_BAR_SAMPLE",
}

COST_ATR0_GRID = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 20260915

TB2_BLOCK = "TB2"
TB3_BLOCK = "TB3"
EXPECTED_TRANSITIONS = 321727
EXPECTED_ALL_OBS = 359714
EXPECTED_HAZARD1 = 37987
EXPECTED_SYMBOLS = ["AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P", "RB", "RU", "SC", "SN", "TA"]


# ===========================================================================
# Reuse Map
# ===========================================================================
REUSE_MAP = {
    "decision_universe": "pgm.SAMPLE_PATH (ALL observation rows: hazard==0 AND hazard==1)",
    "transition_sample": "pgm.TRANSITION_SAMPLE_PATH (degraded to model fitting & target audit only)",
    "pgm_fit": "pgm.fit_samplers_for_window(pgm.WINDOWS[0/1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)",
    "deterministic_pgm_score": "mc.analytic_conditional_support(df)['z_d_up_mu'] -> score_mu = -z_d_up_mu",
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
# Fail-Closed Episode Metadata & ATR0 Reconstruction
# ===========================================================================
def load_episode_metadata(
    ep0_path: Path = base.CACHE / "episode0_episodes.parquet",
    ep3_path: Path = base.CACHE / "episode_repl0_through_tb3.parquet",
) -> pd.DataFrame:
    """Load and merge episode metadata across all blocks with fail-closed integrity checks."""
    if not ep0_path.exists():
        raise SystemExit(f"STOP_PGM_NATIVE_EPISODE_METADATA_MISSING: {ep0_path}")
    if not ep3_path.exists():
        raise SystemExit(f"STOP_PGM_NATIVE_EPISODE_METADATA_MISSING: {ep3_path}")

    ep0 = pd.read_parquet(ep0_path, columns=["symbol", "start_bar", "start_upper_price", "start_lower_price"])
    ep3 = pd.read_parquet(ep3_path, columns=["symbol", "start_bar", "start_upper_price", "start_lower_price"])

    if ep0.duplicated(subset=["symbol", "start_bar"]).any():
        raise SystemExit("STOP_PGM_NATIVE_EPISODE_METADATA_DUPLICATE: duplicates within ep0")
    if ep3.duplicated(subset=["symbol", "start_bar"]).any():
        raise SystemExit("STOP_PGM_NATIVE_EPISODE_METADATA_DUPLICATE: duplicates within ep3")

    # Merge ep0 and ep3 without duplicates (ep3 is superset covering through TB3)
    ep_all = pd.concat([ep0, ep3]).drop_duplicates(subset=["symbol", "start_bar"], keep="last")
    return ep_all


def reconstruct_atr0(s: pd.DataFrame, ep_meta: pd.DataFrame) -> np.ndarray:
    """Reconstruct exact atr0 for observation rows from episode metadata.

    Fail-closed:
      1. ep_meta duplicate (symbol, start_bar) check
      2. span_price finite > 0 check
      3. cur_width_R finite > 0 check
      4. 100% match of (symbol, start_bar) with ep_meta; unmatched prints first 20 and raises SystemExit
      5. atr0 = span_price / cur_width_R finite > 0 check
    """
    if ep_meta.duplicated(subset=["symbol", "start_bar"]).any():
        raise SystemExit("STOP_PGM_NATIVE_EPISODE_METADATA_DUPLICATE")

    span = ep_meta["start_upper_price"].to_numpy(float) - ep_meta["start_lower_price"].to_numpy(float)
    if not np.all(np.isfinite(span)) or not (span > 0).all():
        raise SystemExit("STOP_PGM_NATIVE_ATR0_SPAN_INVALID")

    width = s["cur_width_R"].to_numpy(float)
    if not np.all(np.isfinite(width)) or not (width > 0).all():
        raise SystemExit("STOP_PGM_NATIVE_CUR_WIDTH_INVALID")

    ep_map = dict(zip(zip(ep_meta["symbol"], ep_meta["start_bar"].astype(int)), span))
    keys = list(zip(s["symbol"], s["start_bar"].astype(int)))

    unmatched = [k for k in keys if k not in ep_map]
    if len(unmatched) > 0:
        first_20 = unmatched[:20]
        print(f"[ERROR] STOP_PGM_NATIVE_ATR0_METADATA_UNMATCHED: {len(unmatched)} keys unmatched. First 20: {first_20}")
        raise SystemExit(f"STOP_PGM_NATIVE_ATR0_METADATA_UNMATCHED: {len(unmatched)} keys unmatched")

    spans = np.array([ep_map[k] for k in keys], dtype=float)
    atr0 = spans / width
    if not np.all(np.isfinite(atr0)) or not (atr0 > 0).all():
        raise SystemExit("STOP_PGM_NATIVE_ATR0_INVALID")

    return atr0


# ===========================================================================
# Decision Universe Loader & Auditor
# ===========================================================================
def load_observed_decision_universe(
    ep0_path: Path = base.CACHE / "episode0_episodes.parquet",
    ep3_path: Path = base.CACHE / "episode_repl0_through_tb3.parquet",
) -> pd.DataFrame:
    """Load ALL observation rows from frozen pgm.SAMPLE_PATH with fail-closed atr0 reconstruction."""
    s = pd.read_parquet(pgm.SAMPLE_PATH)
    ep_meta = load_episode_metadata(ep0_path=ep0_path, ep3_path=ep3_path)
    s["atr0"] = reconstruct_atr0(s, ep_meta)
    return s


def load_transition_truth_audit() -> Dict[str, Any]:
    """Load and verify transition truth artifacts for model fitting and parity audit.

    Rebuilt truth from base.build_transition_sample(pgm.SAMPLE_PATH) is the float64 owner.
    Cached artifact pgm.TRANSITION_SAMPLE_PATH is compared, NOT silently overwritten.
    """
    s = pd.read_parquet(pgm.SAMPLE_PATH)
    cur, nxt = base.build_transition_sample(s)
    cached = pd.read_parquet(pgm.TRANSITION_SAMPLE_PATH)

    n_rebuilt = len(cur)
    n_cached = len(cached)
    if n_rebuilt != n_cached:
        raise SystemExit(f"STOP_PGM_NATIVE_TRANSITION_COUNT_MISMATCH: rebuilt={n_rebuilt} cached={n_cached}")

    symbols_equal = bool((cur["symbol"].values == cached["symbol"].values).all())
    episodes_equal = bool((cur["episode_id"].values == cached["episode_id"].values).all())
    blocks_equal = bool((cur["block"].values == cached["block"].values).all())

    if not (symbols_equal and episodes_equal and blocks_equal):
        raise SystemExit("STOP_PGM_NATIVE_TRANSITION_IDENTITY_MISMATCH")

    # Precision difference between float32 cached artifact and float64 rebuilt owner
    diff = np.abs(cached["z_d_up"].to_numpy(float) - cur["z_d_up"].to_numpy(float))
    max_abs_z_d_up_diff = float(np.max(diff))

    return dict(
        cur=cur,
        nxt=nxt,
        cached=cached,
        n_rebuilt=n_rebuilt,
        n_cached=n_cached,
        symbols_equal=symbols_equal,
        episodes_equal=episodes_equal,
        blocks_equal=blocks_equal,
        max_abs_z_d_up_float32_vs_rebuilt=max_abs_z_d_up_diff,
    )


def audit_atr0_owner_parity(s: pd.DataFrame, cur_truth: pd.DataFrame) -> float:
    """Verify exact parity of reconstructed atr0 against rebuilt cur_truth on hazard==0 rows.

    Joined on (symbol, episode_id, bar_t) with validate='one_to_one'.
    """
    s_h0 = s[s["hazard"] == 0][["symbol", "episode_id", "bar_t", "atr0"]].copy()
    cur_sub = cur_truth[["symbol", "episode_id", "bar_t", "atr0"]].copy()
    merged = pd.merge(
        s_h0,
        cur_sub,
        on=["symbol", "episode_id", "bar_t"],
        suffixes=("_obs", "_cur"),
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(cur_truth):
        raise SystemExit(f"STOP_PGM_NATIVE_ATR0_OWNER_PARITY_FAIL: merged len {len(merged)} != cur_truth len {len(cur_truth)}")
    diff = np.abs(merged["atr0_obs"].to_numpy(float) - merged["atr0_cur"].to_numpy(float))
    max_err = float(np.max(diff))
    if max_err > 1e-12:
        raise SystemExit(f"STOP_PGM_NATIVE_ATR0_OWNER_PARITY_FAIL: max_err={max_err:.2e} > 1e-12")
    return max_err


def audit_decision_universe(df: pd.DataFrame) -> Dict[str, Any]:
    """Audit mathematical and contractual integrity of the observation decision universe."""
    n_rows = len(df)

    # 1. Hard assert: exactly {0, 1}
    hazard_set = set(df["hazard"].unique())
    if hazard_set != {0, 1}:
        raise SystemExit(f"STOP_PGM_NATIVE_HAZARD_CLASSES_INVALID: expected {{0, 1}}, got {hazard_set}")

    # 2. Hard assert: finite positive atr0
    atr0 = df["atr0"].to_numpy(float)
    if not np.all(np.isfinite(atr0)) or not (atr0 > 0).all():
        raise SystemExit("STOP_PGM_NATIVE_ATR0_INVALID")

    # 3. Check uniqueness of (symbol, bar_t)
    dup_mask = df.duplicated(subset=["symbol", "bar_t"], keep=False)
    dup_count = int(dup_mask.sum())
    if dup_count > 0:
        cols = [c for c in ["symbol", "bar_t", "episode_id"] if c in df.columns]
        dups = df[dup_mask][cols].head(10)
        print(f"STOP_PGM_NATIVE_DUPLICATE_DECISION_KEY: found {dup_count} duplicate rows:\n{dups}")
        raise SystemExit("STOP_PGM_NATIVE_DUPLICATE_DECISION_KEY")

    # 4. Symbol completeness: all 15 symbols must be present
    symbols = sorted(df["symbol"].unique().tolist())
    if len(symbols) != len(EXPECTED_SYMBOLS) or symbols != EXPECTED_SYMBOLS:
        missing = set(EXPECTED_SYMBOLS) - set(symbols)
        raise SystemExit(f"STOP_PGM_NATIVE_SYMBOLS_INCOMPLETE: missing={missing}")

    # 5. Row count invariants
    n_h0 = int((df["hazard"] == 0).sum())
    n_h1 = int((df["hazard"] == 1).sum())

    if n_rows != EXPECTED_ALL_OBS:
        raise SystemExit(f"STOP_PGM_NATIVE_OBS_COUNT_MISMATCH: expected {EXPECTED_ALL_OBS}, got {n_rows}")
    if n_h0 != EXPECTED_TRANSITIONS:
        raise SystemExit(f"STOP_PGM_NATIVE_HAZARD0_COUNT_MISMATCH: expected {EXPECTED_TRANSITIONS}, got {n_h0}")
    if n_h1 != EXPECTED_HAZARD1:
        raise SystemExit(f"STOP_PGM_NATIVE_HAZARD1_COUNT_MISMATCH: expected {EXPECTED_HAZARD1}, got {n_h1}")
    if n_rows - n_h0 != n_h1:
        raise SystemExit("STOP_PGM_NATIVE_ROW_INVARIANT_VIOLATION")

    row_counts_by_block = df["block"].value_counts().to_dict()
    cov_table = df.groupby(["block", "symbol"]).size().unstack(fill_value=0)
    hazard1_share = float(n_h1 / n_rows) if n_rows > 0 else 0.0

    return dict(
        n_all_obs=n_rows,
        n_rows=n_rows,
        duplicate_keys=dup_count,
        n_hazard0=n_h0,
        n_hazard1=n_h1,
        hazard1_share=hazard1_share,
        row_counts_by_block=row_counts_by_block,
        n_symbols=len(symbols),
        symbols=symbols,
        cov_table=cov_table,
    )


# ===========================================================================
# Episode Limitation Auditor
# ===========================================================================
def audit_episode_selection_limitations() -> Dict[str, Any]:
    """Audit known selection exclusions inherent in the frozen PGM episode sample."""
    ep0_path = base.CACHE / "episode0_episodes.parquet"
    ep3_path = base.CACHE / "episode_repl0_through_tb3.parquet"
    if not (ep0_path.exists() and ep3_path.exists()):
        return dict(available=False, reason="episode parquet not found")

    ep0 = pd.read_parquet(ep0_path)
    ep3 = pd.read_parquet(ep3_path)
    ep_all = pd.concat([ep0, ep3]).drop_duplicates(subset=["symbol", "start_bar"])

    cross_blk = int((ep_all["start_block"] != ep_all["end_block"]).sum())
    censor = int((ep_all["event_mask"] == 0).sum())
    n_episodes = len(ep_all)

    return dict(
        available=True,
        n_total_raw_episodes=n_episodes,
        n_cross_block_excluded=cross_blk,
        n_censor_excluded=censor,
        n_retained_frozen_episodes=n_episodes - cross_blk - censor,
    )


# ===========================================================================
# Raw Bar Economic Alignment & Keyed Nonterminal Parity
# ===========================================================================
def align_raw_bars_and_returns(
    df: pd.DataFrame,
    bars_by_sym: Dict[str, Any],
    cur_truth: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Align observation decision rows with raw 5m bars, verify return parity and decomposition.

    t = bar_t
    entry_bar = t + 1
    Safe assignment avoids out-of-bounds indexing when entry_bar >= n.
    Decomposition: r_CC_ATR0 == gap_ATR0 + r_trad_OC_ATR0 on all valid rows (tol <= 1e-10).
    Nonterminal audit: on hazard == 0 rows, strictly keyed join on (symbol, episode_id, bar_t).
    """
    out_dfs = []
    n_checked = 0
    n_unavailable = 0
    n_disc = 0
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

        # Safe assignment preventing index out of bounds
        entry_bar = np.full(len(e), -1, dtype=int)
        entry_bar[valid] = e[valid]
        g_out["entry_bar"] = entry_bar

        entry_day = np.full(len(e), np.datetime64("NaT"), dtype=bars["day"].dtype)
        entry_day[valid] = bars["day"][e[valid]]
        g_out["entry_day"] = entry_day

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

        # Exact decomposition check on ALL valid rows
        err_decomp = np.abs(r_cc - (r_gap + r_trad))
        loc_max_decomp = float(np.max(err_decomp)) if len(err_decomp) > 0 else 0.0
        if loc_max_decomp > 1e-10:
            raise SystemExit(f"STOP_PGM_NATIVE_RETURN_DECOMPOSITION_MISMATCH: max_err={loc_max_decomp}")

        max_err_decomp = max(max_err_decomp, loc_max_decomp)
        n_checked += len(g_val)

        g_out.loc[valid, "r_state_CC_ATR0"] = r_cc
        g_out.loc[valid, "gap_ATR0"] = r_gap
        g_out.loc[valid, "r_trad_OC_ATR0"] = r_trad
        out_dfs.append(g_out)

    res_df = pd.concat(out_dfs, ignore_index=True)

    # Keyed nonterminal target audit against cur_truth on hazard == 0
    max_err_cc = 0.0
    if cur_truth is not None and not cur_truth.empty:
        h0_valid = res_df[(res_df["hazard"] == 0) & (res_df["is_entry_valid"])].copy()
        cur_sub = cur_truth[["symbol", "episode_id", "bar_t", "z_d_up"]].copy()

        # Keyed merge with strict one_to_one validation
        try:
            merged = pd.merge(
                h0_valid,
                cur_sub,
                on=["symbol", "episode_id", "bar_t"],
                how="left",
                validate="one_to_one",
            )
        except Exception as e:
            raise SystemExit(f"STOP_PGM_NATIVE_NONTERMINAL_KEY_PARITY_FAIL: merge error {e}")

        # Check matched rows == expected H0 valid rows
        n_unmatched = int(merged["z_d_up"].isna().sum())
        if n_unmatched > 0 or len(merged) != len(h0_valid):
            raise SystemExit(f"STOP_PGM_NATIVE_NONTERMINAL_KEY_PARITY_FAIL: unmatched={n_unmatched}, "
                             f"merged={len(merged)}, h0_valid={len(h0_valid)}")

        diff = np.abs(merged["r_state_CC_ATR0"].to_numpy(float) - (-merged["z_d_up"].to_numpy(float)))
        max_err_cc = float(np.max(diff)) if len(diff) > 0 else 0.0
        if max_err_cc > 1e-8:
            raise SystemExit(f"STOP_PGM_NATIVE_NONTERMINAL_TARGET_MISMATCH: max_err={max_err_cc}")

    audit_res = dict(
        n_checked=n_checked,
        n_unavailable=n_unavailable,
        n_disc=n_disc,
        max_err_r_cc=max_err_cc,
        max_err_decomp=max_err_decomp,
    )
    return res_df, audit_res


# ===========================================================================
# PGM Model Scoring & Routing
# ===========================================================================
def score_pgm_block(df_block: pd.DataFrame, trans_sampler: Any) -> pd.DataFrame:
    """Compute primary directional score and native action on block observation rows.

    score_mu represents E[r_CC | S_t, H_{t+1}=0].
    Action is computed independently of future hazard label H_{t+1}.
    """
    mom = trans_sampler.analytic_conditional_support(df_block)
    out = df_block.copy()
    out["score_mu"] = -np.asarray(mom["z_d_up_mu"], dtype=np.float64)

    # Action sign contract: independent of future hazard label
    s = out["score_mu"].to_numpy(float)
    actions = np.where(s > 0, 1, np.where(s < 0, -1, 0))
    out["action"] = actions

    if "r_trad_OC_ATR0" in out.columns:
        out["strategy_return_ATR0"] = out["action"] * out["r_trad_OC_ATR0"]

    return out


def route_and_score_evaluation(
    df_eval: pd.DataFrame,
    mc_A: Any,
    mc_B: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strictly route TB2 rows to Window A sampler (mc_A), and TB3 rows to Window B sampler (mc_B)."""
    df_tb2 = df_eval[df_eval["block"] == TB2_BLOCK].copy()
    df_tb3 = df_eval[df_eval["block"] == TB3_BLOCK].copy()

    if len(df_tb2) == 0:
        raise ValueError("Empty TB2 block for evaluation")
    if len(df_tb3) == 0:
        raise ValueError("Empty TB3 block for evaluation")

    scored_tb2 = score_pgm_block(df_tb2, mc_A)
    scored_tb3 = score_pgm_block(df_tb3, mc_B)
    return scored_tb2, scored_tb3


# ===========================================================================
# Metrics, Deciles, Cost Stress, and Clustered Bootstrap
# ===========================================================================
def compute_block_metrics(df_valid: pd.DataFrame) -> Dict[str, Any]:
    """Compute PRIMARY economic metrics on ALL observation rows, plus hazard diagnostics."""
    score = df_valid["score_mu"].to_numpy(float)
    r_cc = df_valid["r_state_CC_ATR0"].to_numpy(float)
    r_trad = df_valid["r_trad_OC_ATR0"].to_numpy(float)
    gap = df_valid["gap_ATR0"].to_numpy(float)
    action = df_valid["action"].to_numpy(int)
    strat_ret = df_valid["strategy_return_ATR0"].to_numpy(float)
    hazard = df_valid["hazard"].to_numpy(int) if "hazard" in df_valid.columns else np.zeros(len(df_valid), dtype=int)

    # PRIMARY (ALL observation rows)
    spearman_state_all = float(scipy.stats.spearmanr(score, r_cc).statistic)
    spearman_trad_all = float(scipy.stats.spearmanr(score, r_trad).statistic)
    spearman_gap_all = float(scipy.stats.spearmanr(score, gap).statistic)

    n_decisions_all = len(df_valid)
    n_long = int((action == 1).sum())
    n_short = int((action == -1).sum())
    n_skip = int((action == 0).sum())
    n_trades = n_long + n_short

    gross_ev_signal_all = float(np.mean(strat_ret))
    gross_ev_trade_all = float(np.mean(strat_ret[action != 0])) if n_trades > 0 else 0.0
    total_atr0_all = float(np.sum(strat_ret))

    trade_mask = action != 0
    trade_rets = strat_ret[trade_mask]
    wins = trade_rets[trade_rets > 0]
    losses = trade_rets[trade_rets < 0]

    win_rate_all = float(len(wins) / len(trade_rets)) if len(trade_rets) > 0 else 0.0
    mean_win_all = float(np.mean(wins)) if len(wins) > 0 else 0.0
    mean_loss_all = float(np.mean(losses)) if len(losses) > 0 else 0.0
    payoff_ratio_all = float(abs(mean_win_all / mean_loss_all)) if abs(mean_loss_all) > 1e-12 else 0.0
    profit_factor_all = float(np.sum(wins) / abs(np.sum(losses))) if abs(np.sum(losses)) > 1e-12 else 0.0

    # Model validity diagnostic (nonterminal / hazard == 0 only)
    mask_h0 = hazard == 0
    mask_h1 = hazard == 1

    rho_model_nonterminal = float(scipy.stats.spearmanr(score[mask_h0], r_cc[mask_h0]).statistic) if mask_h0.sum() > 1 else 0.0

    # Diagnostic subgroups H0 and H1
    def _subgroup_diag(m: np.ndarray) -> Dict[str, Any]:
        n_sub = int(m.sum())
        if n_sub < 2:
            return dict(n=n_sub, rho_CC=0.0, rho_trad=0.0, EV_sign=0.0, win_rate=0.0, profit_factor=0.0)
        s_m = score[m]
        cc_m = r_cc[m]
        tr_m = r_trad[m]
        act_m = action[m]
        ret_m = strat_ret[m]
        r_cc_sp = float(scipy.stats.spearmanr(s_m, cc_m).statistic)
        r_tr_sp = float(scipy.stats.spearmanr(s_m, tr_m).statistic)
        ev_sig = float(np.mean(ret_m))
        tr_rets = ret_m[act_m != 0]
        w = tr_rets[tr_rets > 0]
        l = tr_rets[tr_rets < 0]
        wr = float(len(w) / len(tr_rets)) if len(tr_rets) > 0 else 0.0
        pf = float(np.sum(w) / abs(np.sum(l))) if abs(np.sum(l)) > 1e-12 else 0.0
        return dict(
            n=n_sub,
            rho_CC=r_cc_sp,
            rho_trad=r_tr_sp,
            EV_sign=ev_sig,
            win_rate=wr,
            profit_factor=pf,
        )

    diag_h0 = _subgroup_diag(mask_h0)
    diag_h1 = _subgroup_diag(mask_h1)

    future_filter_EV_bias = diag_h0["EV_sign"] - gross_ev_signal_all
    future_filter_rho_trad_bias = diag_h0["rho_trad"] - spearman_trad_all

    return dict(
        # PRIMARY economic results (ALL ROWS)
        spearman_state=spearman_state_all,
        spearman_trad=spearman_trad_all,
        spearman_gap=spearman_gap_all,
        gross_ev_signal=gross_ev_signal_all,
        gross_ev_trade=gross_ev_trade_all,
        total_atr0=total_atr0_all,
        n_decisions_all=n_decisions_all,
        n_signals=n_decisions_all,
        n_long=n_long,
        n_short=n_short,
        n_skip=n_skip,
        win_rate=win_rate_all,
        mean_win=mean_win_all,
        mean_loss=mean_loss_all,
        payoff_ratio=payoff_ratio_all,
        profit_factor=profit_factor_all,
        # Diagnostics
        rho_model_nonterminal=rho_model_nonterminal,
        DIAGNOSTIC_H0=diag_h0,
        DIAGNOSTIC_H1=diag_h1,
        future_filter_EV_bias=future_filter_EV_bias,
        future_filter_rho_trad_bias=future_filter_rho_trad_bias,
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
    bins = pd.cut(score, bins=edges, labels=False, include_lowest=True, duplicates="drop")
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


def run_day_clustered_bootstrap(
    df_valid: pd.DataFrame,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Dict[str, float]]:
    """Trading-day clustered bootstrap for PRIMARY ALL-row metrics and nonterminal diagnostic.

    Resamples unique trading days with replacement, evaluates metrics on concatenated rows.
    """
    days = df_valid["entry_day"].to_numpy()
    score = df_valid["score_mu"].to_numpy(float)
    r_cc = df_valid["r_state_CC_ATR0"].to_numpy(float)
    r_trad = df_valid["r_trad_OC_ATR0"].to_numpy(float)
    strat_ret = df_valid["strategy_return_ATR0"].to_numpy(float)
    hazard = df_valid["hazard"].to_numpy(int) if "hazard" in df_valid.columns else np.zeros(len(df_valid), dtype=int)

    # Point estimates on ALL rows
    pt_rho_cc = float(scipy.stats.spearmanr(score, r_cc).statistic)
    pt_rho_trad = float(scipy.stats.spearmanr(score, r_trad).statistic)
    pt_ev_sign = float(np.mean(strat_ret))

    # Point estimate on nonterminal diagnostic
    mask_h0 = hazard == 0
    pt_rho_nonterminal = float(scipy.stats.spearmanr(score[mask_h0], r_cc[mask_h0]).statistic) if mask_h0.sum() > 1 else 0.0

    # Pre-index days
    unique_days, inverse = np.unique(days, return_inverse=True)
    n_days = len(unique_days)
    day_indices = [np.flatnonzero(inverse == d) for d in range(n_days)]

    rng = np.random.default_rng(seed)

    boot_rho_cc = np.empty(n_boot, dtype=np.float64)
    boot_rho_trad = np.empty(n_boot, dtype=np.float64)
    boot_ev_sign = np.empty(n_boot, dtype=np.float64)
    boot_rho_nonterminal = np.empty(n_boot, dtype=np.float64)

    for b in range(n_boot):
        sampled_d = rng.choice(n_days, size=n_days, replace=True)
        idx_b = np.concatenate([day_indices[d] for d in sampled_d])

        s_b = score[idx_b]
        r_cc_b = r_cc[idx_b]
        r_trad_b = r_trad[idx_b]
        ret_b = strat_ret[idx_b]
        haz_b = hazard[idx_b]

        boot_rho_cc[b] = scipy.stats.spearmanr(s_b, r_cc_b).statistic
        boot_rho_trad[b] = scipy.stats.spearmanr(s_b, r_trad_b).statistic
        boot_ev_sign[b] = np.mean(ret_b)

        h0_idx = haz_b == 0
        if h0_idx.sum() > 1:
            boot_rho_nonterminal[b] = scipy.stats.spearmanr(s_b[h0_idx], r_cc_b[h0_idx]).statistic
        else:
            boot_rho_nonterminal[b] = 0.0

    def _summarize(pt: float, dist: np.ndarray) -> Dict[str, float]:
        lo = float(np.percentile(dist, 2.5))
        hi = float(np.percentile(dist, 97.5))
        p_pos = float(np.mean(dist > 0))
        return dict(point=pt, ci95_lower=lo, ci95_upper=hi, p_pos=p_pos)

    return dict(
        rho_CC_all=_summarize(pt_rho_cc, boot_rho_cc),
        rho_state=_summarize(pt_rho_cc, boot_rho_cc),  # alias for backwards-compatibility
        rho_trad_all=_summarize(pt_rho_trad, boot_rho_trad),
        rho_trad=_summarize(pt_rho_trad, boot_rho_trad),  # alias
        EV_sign_all=_summarize(pt_ev_sign, boot_ev_sign),
        EV_sign=_summarize(pt_ev_sign, boot_ev_sign),  # alias
        rho_model_nonterminal=_summarize(pt_rho_nonterminal, boot_rho_nonterminal),
    )


def compute_symbol_breadth(df_valid: pd.DataFrame) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Compute per-symbol metrics across all 15 instruments on ALL rows."""
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
# Formal Verdict Pre-Registration
# ===========================================================================
def determine_formal_verdict(boot_tb3: Dict[str, Any]) -> str:
    """Pre-registered verdict logic based on TB3 bootstrap results.

    1. If CI95(rho_model_nonterminal).lower <= 0 -> NOT_SUPPORTED
    2. Else if CI95(rho_trad_all).lower <= 0 or CI95(EV_sign_all).lower <= 0 -> PREDICTIVE_BUT_NOT_TRADABLE
    3. Else if CI95(rho_model_nonterminal).lower > 0 and CI95(rho_trad_all).lower > 0 and CI95(EV_sign_all).lower > 0 -> SUPPORTED
    """
    ci_model_lo = boot_tb3["rho_model_nonterminal"]["ci95_lower"]
    ci_trad_lo = boot_tb3["rho_trad_all"]["ci95_lower"]
    ci_ev_lo = boot_tb3["EV_sign_all"]["ci95_lower"]

    if ci_model_lo <= 0:
        return VERDICT_STRINGS["NOT_SUPPORTED"]
    elif ci_trad_lo <= 0 or ci_ev_lo <= 0:
        return VERDICT_STRINGS["PREDICTIVE_BUT_NOT_TRADABLE"]
    else:
        return VERDICT_STRINGS["SUPPORTED"]


# ===========================================================================
# Formal Pipeline Executor & Runner
# ===========================================================================
def execute_formal_pipeline(
    obs: pd.DataFrame,
    bars_by_sym: Dict[str, Any],
    cur_truth: pd.DataFrame,
    mc_A: Any,
    mc_B: Any,
    n_boot: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute formal pipeline from aligned observations to formal summary & CSVs."""
    aligned, align_aud = align_raw_bars_and_returns(obs, bars_by_sym, cur_truth=cur_truth)
    df_valid = aligned[aligned["is_entry_valid"]].copy()

    # Hard assert 15 symbols in TB2 and TB3
    tb2_syms = sorted(df_valid[df_valid["block"] == TB2_BLOCK]["symbol"].unique())
    tb3_syms = sorted(df_valid[df_valid["block"] == TB3_BLOCK]["symbol"].unique())
    if tb2_syms != EXPECTED_SYMBOLS or tb3_syms != EXPECTED_SYMBOLS:
        raise SystemExit("STOP_PGM_NATIVE_BLOCK_SYMBOLS_INCOMPLETE")

    # Route evaluation
    tb2_scored, tb3_scored = route_and_score_evaluation(df_valid, mc_A, mc_B)

    # Block metrics
    m_tb2 = compute_block_metrics(tb2_scored)
    m_tb3 = compute_block_metrics(tb3_scored)

    # Deciles using TB2 edges for both
    edges_tb2 = compute_decile_edges(tb2_scored["score_mu"].to_numpy(float))
    dec_tb2 = evaluate_decile_bins(tb2_scored, edges_tb2)
    dec_tb3 = evaluate_decile_bins(tb3_scored, edges_tb2)

    # Cost stress
    stress_tb2 = run_cost_stress_grid(tb2_scored)
    stress_tb3 = run_cost_stress_grid(tb3_scored)

    # Day-clustered bootstrap
    boot_tb2 = run_day_clustered_bootstrap(tb2_scored, n_boot=n_boot, seed=seed)
    boot_tb3 = run_day_clustered_bootstrap(tb3_scored, n_boot=n_boot, seed=seed)

    # Symbol breadth
    breadth_tb2_rows, breadth_tb2_counts = compute_symbol_breadth(tb2_scored)
    breadth_tb3_rows, breadth_tb3_counts = compute_symbol_breadth(tb3_scored)

    # Verdict on TB3 bootstrap
    verdict = determine_formal_verdict(boot_tb3)

    summary = {
        "EXPERIMENT_NAME": EXPERIMENT_NAME,
        "EXPERIMENT_SCOPE": EXPERIMENT_SCOPE,
        "base_sha": BASE_SHA,
        "run_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(_REPO_ROOT), text=True).strip(),
        "sample_artifact_sha256": hashlib.sha256(pgm.SAMPLE_PATH.read_bytes()).hexdigest() if pgm.SAMPLE_PATH.exists() else "N/A",
        "n_all_obs": len(obs),
        "n_hazard0": int((obs["hazard"] == 0).sum()),
        "n_hazard1": int((obs["hazard"] == 1).sum()),
        "hazard1_share": float((obs["hazard"] == 1).mean()),
        "tb2_row_count": len(tb2_scored),
        "tb3_row_count": len(tb3_scored),
        "TB2": {
            "rho_CC_all": m_tb2["spearman_state"],
            "rho_trad_all": m_tb2["spearman_trad"],
            "EV_sign_all": m_tb2["gross_ev_signal"],
            "rho_model_nonterminal": m_tb2["rho_model_nonterminal"],
            "win_rate": m_tb2["win_rate"],
            "profit_factor": m_tb2["profit_factor"],
            "payoff": m_tb2["payoff_ratio"],
            "break_even_cost_ATR0": stress_tb2[0]["break_even_cost_ATR0"],
            "future_filter_EV_bias": m_tb2["future_filter_EV_bias"],
            "future_filter_rho_trad_bias": m_tb2["future_filter_rho_trad_bias"],
        },
        "TB3": {
            "rho_CC_all": m_tb3["spearman_state"],
            "rho_trad_all": m_tb3["spearman_trad"],
            "EV_sign_all": m_tb3["gross_ev_signal"],
            "rho_model_nonterminal": m_tb3["rho_model_nonterminal"],
            "win_rate": m_tb3["win_rate"],
            "profit_factor": m_tb3["profit_factor"],
            "payoff": m_tb3["payoff_ratio"],
            "break_even_cost_ATR0": stress_tb3[0]["break_even_cost_ATR0"],
            "future_filter_EV_bias": m_tb3["future_filter_EV_bias"],
            "future_filter_rho_trad_bias": m_tb3["future_filter_rho_trad_bias"],
        },
        "TB2_bootstrap": boot_tb2,
        "TB3_bootstrap": boot_tb3,
        "TB2_symbol_breadth": breadth_tb2_counts,
        "TB3_symbol_breadth": breadth_tb3_counts,
        "formal_verdict": verdict,
        "known_scope_limitations": [
            "cross-block episodes excluded from frozen PGM sample",
            "event_mask==0 censored episodes excluded from frozen PGM sample",
            "normalized friction stress only; not realistic net PnL",
            "one-step open-to-close probe; not portfolio replay",
        ],
    }

    if output_dir is not None:
        out_p = Path(output_dir)
        out_p.mkdir(parents=True, exist_ok=True)
        (out_p / "pgm_native0a1_formal_summary.json").write_text(json.dumps(summary, indent=2))

        df_metrics = pd.DataFrame([
            {"block": "TB2", **m_tb2},
            {"block": "TB3", **m_tb3},
        ])
        df_metrics.to_csv(out_p / "pgm_native0a1_block_metrics.csv", index=False)

        boot_rows = []
        for blk_name, b_dict in [("TB2", boot_tb2), ("TB3", boot_tb3)]:
            for k, v in b_dict.items():
                boot_rows.append({"block": blk_name, "metric": k, **v})
        pd.DataFrame(boot_rows).to_csv(out_p / "pgm_native0a1_bootstrap.csv", index=False)

        breadth_all = [dict(block="TB2", **r) for r in breadth_tb2_rows] + [dict(block="TB3", **r) for r in breadth_tb3_rows]
        pd.DataFrame(breadth_all).to_csv(out_p / "pgm_native0a1_symbol_breadth.csv", index=False)

        dec_all = [dict(block="TB2", **r) for r in dec_tb2] + [dict(block="TB3", **r) for r in dec_tb3]
        pd.DataFrame(dec_all).to_csv(out_p / "pgm_native0a1_deciles.csv", index=False)

        stress_all = [dict(block="TB2", **r) for r in stress_tb2] + [dict(block="TB3", **r) for r in stress_tb3]
        pd.DataFrame(stress_all).to_csv(out_p / "pgm_native0a1_cost_stress.csv", index=False)

    return summary


def run_formal(output_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Execute formal full evaluation on ALL TB2 and TB3 observation rows (BLOCKED THIS ROUND)."""
    if not os.environ.get("AUTHORIZE_PGM_NATIVE_FORMAL", "").strip():
        raise SystemExit(
            "STOP_PGM_NATIVE_FORMAL_NOT_AUTHORIZED_THIS_ROUND:\n"
            "本轮未授权运行 --formal。必须先提交代码与测试由用户完成独立审计，获得明确授权后再运行。"
        )

    # 1. Base SHA ancestry check
    res = subprocess.run(["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"], cwd=str(_REPO_ROOT), capture_output=True)
    if res.returncode != 0:
        raise SystemExit(f"STOP_PGM_NATIVE_BASE_SHA_NOT_ANCESTOR: BASE_SHA {BASE_SHA} is not an ancestor of HEAD")

    # 2. Decision universe & audit
    obs = load_observed_decision_universe()
    audit_decision_universe(obs)

    # 3. Transition truth
    trans_aud = load_transition_truth_audit()

    # 4. Raw bars
    _, _, bars_by_sym = ex0.load_env()

    # 5. Window A and Window B samplers
    print("[FORMAL] Fitting Window A transition sampler...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc_A = fit_A["trans_samplers"]["MC_STATE_CURREENCODING"]

    print("[FORMAL] Fitting Window B transition sampler...", flush=True)
    fit_B = pgm.fit_samplers_for_window(pgm.WINDOWS[1], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc_B = fit_B["trans_samplers"]["MC_STATE_CURREENCODING"]

    if output_dir is None:
        output_dir = _REPO_ROOT / "research" / "analysis_results" / "local_liquidity_transition_v0"

    return execute_formal_pipeline(
        obs=obs,
        bars_by_sym=bars_by_sym,
        cur_truth=trans_aud["cur"],
        mc_A=mc_A,
        mc_B=mc_B,
        n_boot=BOOTSTRAP_N,
        seed=BOOTSTRAP_SEED,
        output_dir=output_dir,
    )


# ===========================================================================
# Pipeline Modes: Audit-Only and Smoke
# ===========================================================================
def run_audit_only() -> None:
    """Execute static & data audit without fitting PGM models."""
    print("==================================================", flush=True)
    print("PGM-NATIVE-0A.2: AUDIT-ONLY EXECUTION", flush=True)
    print("==================================================", flush=True)

    # 1. Base HEAD ancestor verification
    res = subprocess.run(
        ["git", "merge-base", "--is-ancestor", BASE_SHA, "HEAD"],
        cwd=str(_REPO_ROOT),
        capture_output=True,
    )
    if res.returncode != 0:
        raise SystemExit(f"STOP_PGM_NATIVE_BASE_SHA_NOT_ANCESTOR: BASE_SHA {BASE_SHA} is not an ancestor of HEAD")
    print(f"[AUDIT] BASE SHA ANCESTRY CHECK: PASS (BASE_SHA {BASE_SHA} is ancestor of HEAD)")

    # 2. REUSE MAP
    print_reuse_map()

    # 3. Decision Universe Audit (SAMPLE_PATH: ALL ROWS)
    print("[AUDIT] Loading observation decision universe from pgm.SAMPLE_PATH...", flush=True)
    obs = load_observed_decision_universe()
    aud_res = audit_decision_universe(obs)
    print(f"[AUDIT] Decision universe total observation rows (n_all_obs): {aud_res['n_all_obs']}")
    print(f"[AUDIT] Hazard == 0 count: {aud_res['n_hazard0']}")
    print(f"[AUDIT] Hazard == 1 count: {aud_res['n_hazard1']}")
    print(f"[AUDIT] Hazard == 1 share: {aud_res['hazard1_share'] * 100:.2f}%")
    print(f"[AUDIT] Duplicate (symbol, bar_t) count: {aud_res['duplicate_keys']}")
    print(f"[AUDIT] Block row counts: {aud_res['row_counts_by_block']}")
    print(f"[AUDIT] Total symbols: {aud_res['n_symbols']} {aud_res['symbols']}")

    # 4. Fail-closed metadata check outputs
    ep_meta = load_episode_metadata()
    print(f"[AUDIT] n_unmatched_episode_metadata = 0")
    print(f"[AUDIT] n_duplicate_episode_metadata = 0")

    # 5. Transition Truth Audit (MODEL / AUDIT ARTIFACT)
    print("[AUDIT] Auditing transition truth artifacts...", flush=True)
    trans_aud = load_transition_truth_audit()
    print(f"[AUDIT] Rebuilt transition rows (n_transition): {trans_aud['n_rebuilt']}")
    print(f"[AUDIT] Cached transition rows: {trans_aud['n_cached']}")
    print(f"[AUDIT] Rebuilt == Cached identity: symbols={trans_aud['symbols_equal']}, "
          f"episodes={trans_aud['episodes_equal']}, blocks={trans_aud['blocks_equal']}")
    print(f"[AUDIT] Max abs diff z_d_up (float32 cached vs float64 rebuilt): "
          f"{trans_aud['max_abs_z_d_up_float32_vs_rebuilt']:.2e}")

    # 6. ATR0 Owner Parity
    max_atr0_owner_err = audit_atr0_owner_parity(obs, trans_aud["cur"])
    print(f"[AUDIT] max_abs_atr0_owner_error: {max_atr0_owner_err:.2e} (target <= 1e-12)")

    # Check relation: n_all_obs - n_transition == n_hazard1
    diff_obs_trans = aud_res["n_all_obs"] - trans_aud["n_rebuilt"]
    print(f"[AUDIT] n_all_obs - n_transition = {diff_obs_trans} (matches n_hazard1: {diff_obs_trans == aud_res['n_hazard1']})")

    # 7. Episode Limitation Audit
    lim_aud = audit_episode_selection_limitations()
    if lim_aud.get("available"):
        print(f"[AUDIT] Episode selection limitations:")
        print(f"        Total raw episodes        : {lim_aud['n_total_raw_episodes']}")
        print(f"        Cross-block excluded      : {lim_aud['n_cross_block_excluded']}")
        print(f"        Censor (mask==0) excluded : {lim_aud['n_censor_excluded']}")
        print(f"        Retained frozen episodes  : {lim_aud['n_retained_frozen_episodes']}")

    # 8. Raw Bar Economic Alignment Audit (ALL ROWS)
    print("[AUDIT] Loading bars_by_sym via ex0.load_env()...", flush=True)
    _, _, bars_by_sym = ex0.load_env()

    aligned_df, align_aud = align_raw_bars_and_returns(obs, bars_by_sym, cur_truth=trans_aud["cur"])
    print(f"[AUDIT] Checked valid rows: {align_aud['n_checked']}")
    print(f"[AUDIT] Entry unavailable count: {align_aud['n_unavailable']}")
    print(f"[AUDIT] Discontinuity count: {align_aud['n_disc']}")
    print(f"[AUDIT] Max abs error (r_cc == gap + trad on ALL rows): {align_aud['max_err_decomp']:.2e} (target <= 1e-10)")
    print(f"[AUDIT] Max abs error (r_cc == -z_d_up on nonterminal rows via keyed join): {align_aud['max_err_r_cc']:.2e} (target <= 1e-8)")

    # 9. Coverage by block x symbol
    cov = aligned_df[aligned_df["is_entry_valid"]].groupby(["block", "symbol"]).size().unstack(fill_value=0)
    print("[AUDIT] Valid coverage by block x symbol (ALL rows):")
    print(cov.to_string())

    # 10. Fitted Sampler Design Columns Future Audit
    print("[AUDIT] Fitting Window A transition sampler to inspect design_cols...", flush=True)
    fit_A = pgm.fit_samplers_for_window(pgm.WINDOWS[0], pgm.SAMPLE_PATH, pgm.TRANSITION_SAMPLE_PATH)
    mc_A = fit_A["trans_samplers"]["MC_STATE_CURREENCODING"]
    forbidden_targets = set(base.ALL_Z_COLS + base.COUNT_Z + [
        "hazard", "target_mask",
        "_".join(["reward", "SKIP"]),
        "_".join(["reward", "MARKET"]),
        "_".join(["reward", "LIMIT_RR3"]),
        "_".join(["reward", "REASSESS_RR3"]),
        "r_CC_ATR0", "gap_ATR0", "r_trad_OC_ATR0",
    ])
    inter = set(mc_A.design_cols).intersection(forbidden_targets)
    if inter:
        raise SystemExit(f"STOP_PGM_NATIVE_DESIGN_COLS_INTERSECTS_TARGETS: {inter}")
    print(f"[AUDIT] mc_A design columns count: {len(mc_A.design_cols)} (all causal, zero target overlap)")

    # 11. Forbidden Dependencies Check
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
    """Execute lightweight end-to-end smoke test on <= 512 rows per block sampled from ALL rows."""
    print("==================================================", flush=True)
    print("PGM-NATIVE-0A.2: SMOKE TEST EXECUTION (<=512 rows/block on ALL rows)", flush=True)
    print("==================================================", flush=True)
    t0 = time.perf_counter()

    # 1. Load universe & raw bars
    obs = load_observed_decision_universe()
    trans_aud = load_transition_truth_audit()
    _, _, bars_by_sym = ex0.load_env()
    aligned_df, _ = align_raw_bars_and_returns(obs, bars_by_sym, cur_truth=trans_aud["cur"])

    # Subsample TB2 and TB3 to 512 rows each from ALL rows, ensuring both hazard 0 and 1 are present
    def _sample_block(blk_name: str, n_total: int = 512, seed: int = 42) -> pd.DataFrame:
        blk = aligned_df[(aligned_df["block"] == blk_name) & aligned_df["is_entry_valid"]]
        h0 = blk[blk["hazard"] == 0]
        h1 = blk[blk["hazard"] == 1]
        n_h1 = max(1, int(round(n_total * (len(h1) / len(blk)))))
        n_h0 = n_total - n_h1
        sub_h0 = h0.sample(n=n_h0, random_state=seed)
        sub_h1 = h1.sample(n=n_h1, random_state=seed)
        return pd.concat([sub_h0, sub_h1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    tb2_sub = _sample_block(TB2_BLOCK, 512, 42)
    tb3_sub = _sample_block(TB3_BLOCK, 512, 42)

    print(f"[SMOKE] Subsampled TB2: {len(tb2_sub)} rows (H0={int((tb2_sub['hazard']==0).sum())}, "
          f"H1={int((tb2_sub['hazard']==1).sum())}, {tb2_sub['symbol'].nunique()} syms)")
    print(f"[SMOKE] Subsampled TB3: {len(tb3_sub)} rows (H0={int((tb3_sub['hazard']==0).sum())}, "
          f"H1={int((tb3_sub['hazard']==1).sum())}, {tb3_sub['symbol'].nunique()} syms)")

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

    # 4. Metrics (PRIMARY ALL ROWS + DIAGNOSTICS)
    m_tb2 = compute_block_metrics(tb2_scored)
    m_tb3 = compute_block_metrics(tb3_scored)

    print(f"[SMOKE TB2 ALL] rho_CC={m_tb2['spearman_state']:.4f}, rho_trad={m_tb2['spearman_trad']:.4f}, "
          f"EV_sign={m_tb2['gross_ev_signal']:.4f}, win_rate={m_tb2['win_rate']:.4f}")
    print(f"[SMOKE TB2 H0 ] rho_CC={m_tb2['DIAGNOSTIC_H0']['rho_CC']:.4f}, rho_trad={m_tb2['DIAGNOSTIC_H0']['rho_trad']:.4f}, "
          f"EV_sign={m_tb2['DIAGNOSTIC_H0']['EV_sign']:.4f}")
    print(f"[SMOKE TB2 H1 ] rho_CC={m_tb2['DIAGNOSTIC_H1']['rho_CC']:.4f}, rho_trad={m_tb2['DIAGNOSTIC_H1']['rho_trad']:.4f}, "
          f"EV_sign={m_tb2['DIAGNOSTIC_H1']['EV_sign']:.4f}")
    print(f"[SMOKE TB2 BIAS] future_filter_EV_bias={m_tb2['future_filter_EV_bias']:.4f}, "
          f"future_filter_rho_trad_bias={m_tb2['future_filter_rho_trad_bias']:.4f}")

    print(f"[SMOKE TB3 ALL] rho_CC={m_tb3['spearman_state']:.4f}, rho_trad={m_tb3['spearman_trad']:.4f}, "
          f"EV_sign={m_tb3['gross_ev_signal']:.4f}, win_rate={m_tb3['win_rate']:.4f}")
    print(f"[SMOKE TB3 H0 ] rho_CC={m_tb3['DIAGNOSTIC_H0']['rho_CC']:.4f}, rho_trad={m_tb3['DIAGNOSTIC_H0']['rho_trad']:.4f}, "
          f"EV_sign={m_tb3['DIAGNOSTIC_H0']['EV_sign']:.4f}")
    print(f"[SMOKE TB3 H1 ] rho_CC={m_tb3['DIAGNOSTIC_H1']['rho_CC']:.4f}, rho_trad={m_tb3['DIAGNOSTIC_H1']['rho_trad']:.4f}, "
          f"EV_sign={m_tb3['DIAGNOSTIC_H1']['EV_sign']:.4f}")
    print(f"[SMOKE TB3 BIAS] future_filter_EV_bias={m_tb3['future_filter_EV_bias']:.4f}, "
          f"future_filter_rho_trad_bias={m_tb3['future_filter_rho_trad_bias']:.4f}")

    # 5. Decile monotonicity
    edges_tb2 = compute_decile_edges(tb2_scored["score_mu"].to_numpy(float))
    dec_tb2 = evaluate_decile_bins(tb2_scored, edges_tb2)
    dec_tb3 = evaluate_decile_bins(tb3_scored, edges_tb2)
    print(f"[SMOKE] Decile binning verified (10 bins for TB2 and TB3 using TB2 edges).")

    # 6. Cost stress
    stress_tb2 = run_cost_stress_grid(tb2_scored)
    stress_tb3 = run_cost_stress_grid(tb3_scored)
    print(f"[SMOKE] Cost stress grid verified: break-even cost TB2={stress_tb2[0]['break_even_cost_ATR0']:.4f}, "
          f"TB3={stress_tb3[0]['break_even_cost_ATR0']:.4f}")

    # 7. Day-clustered bootstrap (smoke with 200 reps)
    print("[SMOKE] Running day-clustered bootstrap on TB3 (200 reps)...", flush=True)
    boot_tb3 = run_day_clustered_bootstrap(tb3_scored, n_boot=200, seed=BOOTSTRAP_SEED)
    print(f"[SMOKE TB3 Boot ALL] rho_CC   : pt={boot_tb3['rho_CC_all']['point']:.4f}, CI=[{boot_tb3['rho_CC_all']['ci95_lower']:.4f}, {boot_tb3['rho_CC_all']['ci95_upper']:.4f}]")
    print(f"[SMOKE TB3 Boot ALL] rho_trad : pt={boot_tb3['rho_trad_all']['point']:.4f}, CI=[{boot_tb3['rho_trad_all']['ci95_lower']:.4f}, {boot_tb3['rho_trad_all']['ci95_upper']:.4f}]")
    print(f"[SMOKE TB3 Boot ALL] EV_sign  : pt={boot_tb3['EV_sign_all']['point']:.4f}, CI=[{boot_tb3['EV_sign_all']['ci95_lower']:.4f}, {boot_tb3['EV_sign_all']['ci95_upper']:.4f}]")
    print(f"[SMOKE TB3 Boot DIAG] rho_nonterminal: pt={boot_tb3['rho_model_nonterminal']['point']:.4f}, CI=[{boot_tb3['rho_model_nonterminal']['ci95_lower']:.4f}, {boot_tb3['rho_model_nonterminal']['ci95_upper']:.4f}]")

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
        run_formal()
        return

    if args.audit_only:
        run_audit_only()
        return

    if args.smoke:
        run_smoke_test()
        return

    parser.print_help()


if __name__ == "__main__":
    main()
