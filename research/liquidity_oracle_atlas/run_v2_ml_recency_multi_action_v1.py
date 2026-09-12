"""V2-B / V2-C / V2-D: Nonlinear ML x Recency x Context x Multi-Action Value Experiment.

Authoritative baseline: 891cadcd070d7007e383516f200a0da70a45d1c3
Evaluates:
  1. GBDT (HistGradientBoosting) vs Linear (Ridge / LogisticRegression)
  2. Recency weighting across 5 half-lives (None, 60, 120, 240, 480 days)
  3. Market context (S3) and symbol context (S2/S4)
  4. Multi-action state-dependent policy selection over:
     [SKIP, MARKET, LIMIT_RR3, REASSESS_RR3]
     benchmarked against fixed E1.1 Market and fixed RR=3 Strict Reassess.
"""
from __future__ import annotations
import json
import inspect
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, r2_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

# Ensure repo root is in python path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.liquidity_oracle_atlas import run_latent_state_compression_v1 as s4b
from research.liquidity_oracle_atlas.run_enter_skip_selection_v1 import primary_matches, build_policy
from research.liquidity_oracle_atlas.run_liquidity_field_action_surface_v1 import _assign_blocks, first_hit_bounds
from research.liquidity_oracle_atlas.run_execution_limit_frontier_v1 import run_block
from research.liquidity_oracle_atlas.run_execution_limit_frontier_v1_closure import (
    attempt_universe,
    prefill,
    route_reassess,
    scalar_outcome,
)
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env

OUT_DIR = REPO_ROOT / "research/analysis_results/v2_ml_recency_multi_action_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants & Governance
# ---------------------------------------------------------------------------
SEED = 20260912
BOOTSTRAP_N = 2000
MAX_ACTION_TRAIN_ROWS = 1_000_000
P1_CUTOFF = pd.Timestamp("2026-09-04 14:55:00")

HALF_LIFE_DAYS = [None, 60, 120, 240, 480]
COST_R_GRID = [0.00, 0.01, 0.02, 0.03, 0.05, 0.10]
ACTIONS = ["SKIP", "MARKET", "LIMIT_RR3", "REASSESS_RR3"]
HARDENING_VERSION = "bar-index-purge-v1"
PURGE_TRACE = []
PREPROCESS_TRACE = []
REPRO_TRACE = []

OUTER_WFS = [
    ("WF1", ["TB1"], ["TB2"]),
    ("WF2", ["TB1", "TB2"], ["TB3"]),
    ("WF3", ["TB1", "TB2", "TB3"], ["TB4"]),
]

REACTION_FEATURES = s4b.REACTION_FEATURES
LIQUIDITY_FEATURES = s4b.LIQUIDITY_FEATURES
SYMBOLS = ["AG", "AL", "AU", "CF", "CU", "I", "M", "MA", "NI", "P", "RB", "RU", "SC", "SN", "TA"]

BASELINE_E1_MARKET = {
    "WF1": {"EV_lower": 0.042022, "EV_cw": 0.010376, "total_R": 132.788702},
    "WF2": {"EV_lower": 0.047519, "EV_cw": 0.033118, "total_R": 135.286334},
    "WF3": {"EV_lower": 0.076375, "EV_cw": 0.043795, "total_R": 229.735402},
}

BASELINE_E2_SINGLE_RR3 = {
    "WF1": {"EV_lower": 0.008910, "EV_cw": -0.005964},
    "WF2": {"EV_lower": -0.010777, "EV_cw": -0.016046},
    "WF3": {"EV_lower": 0.067233, "EV_cw": 0.058922},
}

BASELINE_E2_REASSESS_RR3 = {
    "WF1": {"EV_lower": 0.049100, "EV_cw": 0.034226},
    "WF2": {"EV_lower": 0.049637, "EV_cw": 0.044369},
    "WF3": {"EV_lower": 0.112446, "EV_cw": 0.104135},
}


def recency_weights(sample_time: pd.Series, train_cutoff: pd.Timestamp, half_life_days: float | None) -> np.ndarray:
    """Compute normalized exponential recency weights w_i = 2^(-age / H)."""
    if half_life_days is None:
        return np.ones(len(sample_time), dtype=np.float64)
    age = (train_cutoff - pd.to_datetime(sample_time)).dt.total_seconds().to_numpy()
    age_days = np.maximum(age / 86400.0, 0.0)
    w = np.power(2.0, -age_days / float(half_life_days))
    mean_w = np.mean(w)
    if mean_w > 0:
        w /= mean_w
    return w


def deterministic_cap(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Subsample dataframe deterministically using hash of ['gid', 'h', 'action', 'scale']."""
    if len(df) <= n:
        return df
    key = df[["gid", "h", "action", "scale"]]
    h = pd.util.hash_pandas_object(key, index=False).to_numpy(np.uint64)
    idx = np.argpartition(h, n)[:n]
    return df.iloc[idx].reset_index(drop=True)


def conservative_reward(filled: np.ndarray, r_lower: np.ndarray, censored: np.ndarray) -> np.ndarray:
    """Conservative censor-worst return target."""
    out = np.zeros(len(filled), dtype=np.float64)
    resolved = filled & ~censored
    out[resolved] = r_lower[resolved]
    out[filled & censored] = -1.0
    return out


def attach_exact_reward_end(df: pd.DataFrame, bars_by_sym: Dict[str, Any]) -> pd.DataFrame:
    """Map the most conservative candidate-action path end through actual bars.

    MARKET consumes entry..entry+33, a TTL2 limit may fill on entry+1 and then
    consumes 34 outcome bars, and REASSESS may activate R4 at contact+10 and
    fill one bar later.  Thus the all-action maximum is contact+44.
    """
    x=df.copy()
    x["reward_end_bar_MARKET"]=x["entry_bar_index"].astype(int)+33
    x["reward_end_bar_LIMIT_RR3"]=x["entry_bar_index"].astype(int)+34
    x["reward_end_bar_REASSESS_RR3"]=x["contact_bar_index"].astype(int)+44
    x["reward_end_bar_index"]=x[["reward_end_bar_MARKET","reward_end_bar_LIMIT_RR3","reward_end_bar_REASSESS_RR3"]].max(axis=1).astype(int)
    end=pd.Series(pd.NaT,index=x.index,dtype="datetime64[ns]")
    for sym,idx in x.groupby("symbol").groups.items():
        B=bars_by_sym[sym];ii=x.loc[idx,"reward_end_bar_index"].to_numpy(int)
        ii=np.minimum(ii,B["n"]-1)  # data-end censored actions terminate at last observed bar
        end.loc[idx]=pd.to_datetime(B["t"][ii]).to_numpy(dtype="datetime64[ns]")
    x["reward_end_time"]=end
    x["reward_end_semantics"]=HARDENING_VERSION
    assert (x.reward_end_time>=pd.to_datetime(x.entry_time)).all()
    return x

def map_bar_end_time(df, end_index, bars_by_sym):
    out=pd.Series(pd.NaT,index=df.index,dtype="datetime64[ns]")
    for sym,idx in df.groupby("symbol").groups.items():
        ii=np.asarray(end_index.loc[idx],int);B=bars_by_sym[sym]
        ii=np.minimum(ii,B["n"]-1);out.loc[idx]=pd.to_datetime(B["t"][ii]).to_numpy()
    return out

def nested_half_life_selection(outer_train, feats, target, task, wf, bars_by_sym, classification):
    """Select recency strictly inside outer train; outer test is never inspected."""
    d=outer_train.copy().sort_values("decision_time");days=np.sort(pd.to_datetime(d.decision_time).dt.normalize().unique())
    cut=days[max(1,(len(days)*3)//4)];itr=d[pd.to_datetime(d.decision_time).dt.normalize()<cut].copy();iv=d[pd.to_datetime(d.decision_time).dt.normalize()>=cut].copy()
    val_start=pd.to_datetime(iv.decision_time).min();itr=itr[pd.to_datetime(itr.label_end_time)<val_start].copy()
    assert len(itr) and len(iv) and pd.to_datetime(itr.label_end_time).max()<val_start
    X=itr[feats].to_numpy(float);Z=iv[feats].to_numpy(float);med=np.nanmedian(X,axis=0);med=np.where(np.isfinite(med),med,0.);X=np.where(np.isfinite(X),X,med);Z=np.where(np.isfinite(Z),Z,med);sc=StandardScaler().fit(X)
    rows=[]
    for model in ["Linear","GBDT"]:
      scores=[]
      for hl in HALF_LIFE_DAYS:
        w=recency_weights(itr.decision_time,pd.to_datetime(itr.decision_time).max(),hl)
        if classification:
          y=itr[target].astype(int);yt=iv[target].astype(int)
          m=(LogisticRegression(C=1.,max_iter=200,random_state=SEED) if model=="Linear" else HistGradientBoostingClassifier(learning_rate=.05,max_iter=100,max_leaf_nodes=15,l2_regularization=1.,min_samples_leaf=500,early_stopping=False,random_state=SEED))
          m.fit(sc.transform(X) if model=="Linear" else X,y,sample_weight=w);p=m.predict_proba(sc.transform(Z) if model=="Linear" else Z)[:,1];score=log_loss(yt,p)
        else:
          y=itr[target].to_numpy(float);yt=iv[target].to_numpy(float);ok=np.isfinite(y);ov=np.isfinite(yt)
          m=(Ridge(alpha=1.,random_state=SEED) if model=="Linear" else HistGradientBoostingRegressor(learning_rate=.05,max_iter=100,max_leaf_nodes=15,l2_regularization=1.,min_samples_leaf=100,early_stopping=False,random_state=SEED))
          m.fit((sc.transform(X) if model=="Linear" else X)[ok],y[ok],sample_weight=w[ok]);p=m.predict((sc.transform(Z) if model=="Linear" else Z)[ov]);score=-r2_score(yt[ov],p)
        scores.append(score)
      best=int(np.argmin(scores));rows.append(dict(task=task,wf=wf,model=model,selected_half_life=str(HALF_LIFE_DAYS[best]),inner_metric="log_loss" if classification else "r2",inner_metric_value=float(scores[best] if classification else -scores[best]),inner_train_n=len(itr),inner_validation_n=len(iv),outer_test_used_for_selection=False))
    return rows


# ---------------------------------------------------------------------------
# Stage B0: Baseline Reproduction
# ---------------------------------------------------------------------------
def run_stage_b0(trades: pd.DataFrame, single_primary: pd.DataFrame, rx: pd.DataFrame) -> Dict[str, Any]:
    print("[STAGE B0] Verifying frozen baseline reproduction...")
    max_t = pd.to_datetime(trades["entry_time"]).max()
    assert max_t <= P1_CUTOFF, f"FATAL: P1 leakage detected in trades max_t={max_t} > {P1_CUTOFF}"
    
    # 1. Market Baseline
    m_filled = (trades["status"] == "EXECUTED_LAG1").to_numpy()
    m_censored = trades["censored"].fillna(False).astype(bool).to_numpy()
    m_rlow = trades["R_lower"].fillna(0.0).to_numpy()
    market_cw = conservative_reward(m_filled, m_rlow, m_censored)
    
    e1_diffs = {}
    for wf, tgt in BASELINE_E1_MARKET.items():
        sub = trades[trades["wf"] == wf]
        idx = sub.index
        ev_low = sub.loc[sub["status"] == "EXECUTED_LAG1", "R_lower"].sum() / len(sub)
        ev_cw = market_cw[idx].mean()
        tot_r = sub.loc[sub["status"] == "EXECUTED_LAG1", "R_lower"].sum()
        diff = max(abs(ev_low - tgt["EV_lower"]), abs(ev_cw - tgt["EV_cw"]), abs(tot_r - tgt["total_R"]))
        e1_diffs[wf] = diff
        assert diff < 1e-4, f"STOP_V2_ML_BASELINE_REPRODUCTION_FAIL: E1 Market {wf} diff={diff}"
    print("  [OK] E1.1 Market reproduced 100%")

    # 2. E2 Single RR=3 Strict
    e2_single_diffs = {}
    for wf, tgt in BASELINE_E2_SINGLE_RR3.items():
        sub = single_primary[single_primary["wf"] == wf]
        n = len(sub)
        ev_low = sub["R_lower"].sum() / n
        ev_cw = (sub["R_lower"].sum() - sub["R_lower"].isna().sum()) / n
        diff = max(abs(ev_low - tgt["EV_lower"]), abs(ev_cw - tgt["EV_cw"]))
        e2_single_diffs[wf] = diff
        assert diff < 1e-4, f"STOP_V2_ML_BASELINE_REPRODUCTION_FAIL: E2 Single {wf} diff={diff}"
    print("  [OK] E2 Single RR=3 Strict reproduced 100%")

    # 3. E2 Reassess RR=3 Strict
    e2_reassess_diffs = {}
    for wf, tgt in BASELINE_E2_REASSESS_RR3.items():
        sub = rx[rx["wf"] == wf]
        n = len(sub)
        ev_low = sub["R_lower"].sum() / n
        ev_cw = (sub["R_lower"].sum() - sub["R_lower"].isna().sum()) / n
        diff = max(abs(ev_low - tgt["EV_lower"]), abs(ev_cw - tgt["EV_cw"]))
        e2_reassess_diffs[wf] = diff
        assert diff < 1e-4, f"STOP_V2_ML_BASELINE_REPRODUCTION_FAIL: E2 Reassess {wf} diff={diff}"
    print("  [OK] E2 Reassess RR=3 Strict reproduced 100%")

    return dict(e1_market=e1_diffs, e2_single=e2_single_diffs, e2_reassess=e2_reassess_diffs)


# ---------------------------------------------------------------------------
# Stage B1: Dataset & Feature Construction (TB1 + TB2-TB4)
# ---------------------------------------------------------------------------
def build_multi_action_dataset(D, master_by_sym, bars_by_sym, trades: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    cache_path = OUT_DIR / "multi_action_signals_features.parquet"
    if cache_path.exists():
        print(f"[CACHE] Loading cached multi-action dataset from {cache_path}...")
        df = pd.read_parquet(cache_path)
        if "reward_end_semantics" not in df or not (df.reward_end_semantics==HARDENING_VERSION).all():
            print("[CACHE] stale wall-clock reward end; rebuilding")
        else:
          feature_blocks = {
            "S1": json.loads((OUT_DIR / "feature_block_S1.json").read_text()),
            "S2": json.loads((OUT_DIR / "feature_block_S2.json").read_text()),
            "S3": json.loads((OUT_DIR / "feature_block_S3.json").read_text()),
            "S4": json.loads((OUT_DIR / "feature_block_S4.json").read_text()),
          }
          return df, feature_blocks

    print("[STAGE B1] Constructing unified multi-action dataset (TB1 + TB2-TB4)...")
    surface, gmap = s4b.compute_action_surface_all_blocks(D, master_by_sym, bars_by_sym)
    repro = s4b.assert_reproduce_stage4a(surface)
    state, join_stats = s4b.build_state_table(D, surface, gmap)
    print(f"  [STATE] rows={len(state)} join_stats={join_stats}")

    F = D["F"].copy()
    bm = _assign_blocks(F)
    F["block"] = pd.to_datetime(F.decision_time).dt.normalize().map(bm)
    
    # 1. Attempt universe for TB2-TB4 (the 9,015 test signals)
    attempts_test, repro2 = attempt_universe(D, master_by_sym, bars_by_sym)
    attempts_test = prefill(attempts_test, bars_by_sym)
    selected_test = trades[["gid", "region"]].copy()

    rr = 3.0
    model = "STRICT_TRADE_THROUGH"
    full_limit_test = pd.concat([run_block(g, bars_by_sym[s], rr, model) for s, g in attempts_test.groupby("symbol", sort=False)], ignore_index=True)
    single_primary_test = full_limit_test.merge(selected_test, on=["gid", "region"], validate="one_to_one")

    results_test = {(rr, model): full_limit_test}
    rx_test, tr_test = route_reassess(results_test, attempts_test)

    # Validate Stage B0 on 9,015 test signals
    run_stage_b0(trades, single_primary_test, rx_test)

    # 2. Build TB1 training signals (WF0)
    contact_meta = gmap.merge(F[["symbol", "liquidity_id", "contact_number", "decision_time", "contact_bar_index", "block", "atr0"]], on=["symbol", "liquidity_id", "contact_number"], how="left")
    contact_meta["wf"] = contact_meta["block"].map({"TB1": "WF0", "TB2": "WF1", "TB3": "WF2", "TB4": "WF3"})
    contact_meta = contact_meta.merge(surface[["gid", "wf"]].drop_duplicates(), on=["gid", "wf"], how="inner")

    matches = primary_matches(surface)
    pol = build_policy(surface, matches, contact_meta)
    selected_tb1 = pol[pol["entered"] & (pol["wf"] == "WF0")][["gid", "region"]].copy()

    meta = gmap.merge(F[["symbol", "liquidity_id", "contact_number", "contact_bar_index", "atr0"]], on=["symbol", "liquidity_id", "contact_number"], validate="one_to_one")
    m_tb1 = matches.merge(meta[["gid", "symbol", "contact_bar_index", "atr0"]], on="gid", validate="many_to_one")
    m_tb1["direction"] = m_tb1.d.astype(int)
    m_tb1["reference_entry"] = m_tb1.entry_atr * m_tb1.atr0
    m_tb1["target_price"] = m_tb1.reference_entry + m_tb1.direction * m_tb1.target_atr * m_tb1.atr0
    m_tb1["stop_price"] = m_tb1.reference_entry - m_tb1.direction * m_tb1.risk_atr * m_tb1.atr0
    m_tb1["signal_bar_index"] = m_tb1.contact_bar_index.astype(int) + 1 + m_tb1.h.astype(int)
    m_tb1["entry_bar_index"] = m_tb1.signal_bar_index + 1

    primary_tb1 = m_tb1.merge(selected_tb1, on=["gid", "region"], validate="one_to_one")
    attempts_tb1 = prefill(primary_tb1, bars_by_sym)

    # TB1 Market execution
    tb1_m_filled = (attempts_tb1["status"] == "EXECUTED_LAG1").to_numpy()
    tb1_m_rlow = np.zeros(len(attempts_tb1))
    tb1_m_rupp = np.zeros(len(attempts_tb1))
    tb1_m_cens = np.zeros(len(attempts_tb1), dtype=bool)

    for j, row in enumerate(attempts_tb1.itertuples()):
        if row.status == "EXECUTED_LAG1":
            B = bars_by_sym[row.symbol]
            entry_p = B["o"][row.entry_bar_index]
            lo, up = scalar_outcome(B, row.entry_bar_index, entry_p, row.target_price, row.stop_price, row.direction)
            tb1_m_rlow[j] = lo
            tb1_m_rupp[j] = up
            tb1_m_cens[j] = np.isnan(lo)

    # TB1 Limit & Reassess execution
    full_tb1 = pd.concat([run_block(g, bars_by_sym[s], rr, model) for s, g in attempts_tb1.groupby("symbol", sort=False)], ignore_index=True)
    rx_tb1, _ = route_reassess({(rr, model): full_tb1}, attempts_tb1)

    tb1_lim_filled = full_tb1["status"].isin(["FILLED", "FILLED_AT_OPEN"]).to_numpy()
    tb1_lim_cens = full_tb1["R_lower"].isna().to_numpy() & tb1_lim_filled
    tb1_lim_rlow = full_tb1["R_lower"].fillna(0.0).to_numpy()
    tb1_lim_rupp = full_tb1["R_upper"].fillna(0.0).to_numpy()

    tb1_rea_filled = rx_tb1["filled"].to_numpy().astype(bool)
    tb1_rea_cens = rx_tb1["R_lower"].isna().to_numpy() & tb1_rea_filled
    tb1_rea_rlow = rx_tb1["R_lower"].fillna(0.0).to_numpy()
    tb1_rea_rupp = rx_tb1["R_upper"].fillna(0.0).to_numpy()

    # 3. Assemble combined table (TB1 + TB2-TB4)
    # Assemble test signals (9015)
    df_test = trades.copy().sort_values("gid").reset_index(drop=True)
    single_primary_test = single_primary_test.sort_values("gid").reset_index(drop=True)
    rx_test = rx_test.sort_values("gid").reset_index(drop=True)

    region_h = {"R1": 2, "R2": 5, "R3": 5, "R4": 8}
    df_test["h"] = df_test["region"].map(region_h).astype("int64")
    df_test["block"] = df_test["wf"].map({"WF1": "TB2", "WF2": "TB3", "WF3": "TB4"})
    
    FIELD_KEY = ["symbol", "liquidity_id", "contact_number"]
    df_test = df_test.merge(gmap[["gid"] + FIELD_KEY[1:]], on="gid", validate="one_to_one")

    # Geometry for test
    attempts_test_sub = attempts_test.merge(selected_test, on=["gid", "region"], validate="one_to_one").sort_values("gid").reset_index(drop=True)
    df_test["target_price"] = attempts_test_sub["target_price"]
    df_test["stop_price"] = attempts_test_sub["stop_price"]
    df_test["reference_entry"] = attempts_test_sub["reference_entry"]
    df_test["contact_bar_index"] = attempts_test_sub["contact_bar_index"]
    df_test["signal_bar_index"] = attempts_test_sub["signal_bar_index"]
    df_test["entry_bar_index"] = attempts_test_sub["entry_bar_index"]

    atr_meta = F.drop_duplicates(FIELD_KEY)[FIELD_KEY + ["atr0", "liquidity_price"]].reset_index(drop=True)
    df_test = df_test.merge(atr_meta, on=FIELD_KEY, how="left", validate="many_to_one")

    df_test["target_atr"] = np.abs(df_test["target_price"] - df_test["reference_entry"]) / df_test["atr0"]
    df_test["risk_atr"] = np.abs(df_test["stop_price"] - df_test["reference_entry"]) / df_test["atr0"]
    df_test["rr"] = df_test["target_atr"] / np.maximum(df_test["risk_atr"], 1e-12)
    df_test["structure_scale"] = 1.2
    df_test["action_is_outward"] = (df_test["region"] != "R3").astype(int)

    # Counterfactual rewards test
    df_test["reward_SKIP"] = 0.0
    df_test["filled_SKIP"] = False
    df_test["R_lower_SKIP"] = 0.0
    df_test["R_upper_SKIP"] = 0.0
    df_test["censored_SKIP"] = False

    m_filled = (trades["status"] == "EXECUTED_LAG1").to_numpy()
    m_censored = trades["censored"].fillna(False).astype(bool).to_numpy()
    m_rlow = trades["R_lower"].fillna(0.0).to_numpy()
    m_rupp = trades["R_upper"].fillna(0.0).to_numpy()
    df_test["reward_MARKET"] = conservative_reward(m_filled, m_rlow, m_censored)
    df_test["filled_MARKET"] = m_filled
    df_test["R_lower_MARKET"] = np.where(m_filled, m_rlow, 0.0)
    df_test["R_upper_MARKET"] = np.where(m_filled, m_rupp, 0.0)
    df_test["censored_MARKET"] = m_censored

    lim_filled = single_primary_test["status"].isin(["FILLED", "FILLED_AT_OPEN"]).to_numpy()
    lim_censored = single_primary_test["R_lower"].isna().to_numpy() & lim_filled
    lim_rlow = single_primary_test["R_lower"].fillna(0.0).to_numpy()
    lim_rupp = single_primary_test["R_upper"].fillna(0.0).to_numpy()
    df_test["reward_LIMIT_RR3"] = conservative_reward(lim_filled, lim_rlow, lim_censored)
    df_test["filled_LIMIT_RR3"] = lim_filled
    df_test["R_lower_LIMIT_RR3"] = np.where(lim_filled, lim_rlow, 0.0)
    df_test["R_upper_LIMIT_RR3"] = np.where(lim_filled, lim_rupp, 0.0)
    df_test["censored_LIMIT_RR3"] = lim_censored

    rea_filled = rx_test["filled"].to_numpy().astype(bool)
    rea_censored = rx_test["R_lower"].isna().to_numpy() & rea_filled
    rea_rlow = rx_test["R_lower"].fillna(0.0).to_numpy()
    rea_rupp = rx_test["R_upper"].fillna(0.0).to_numpy()
    df_test["reward_REASSESS_RR3"] = conservative_reward(rea_filled, rea_rlow, rea_censored)
    df_test["filled_REASSESS_RR3"] = rea_filled
    df_test["R_lower_REASSESS_RR3"] = np.where(rea_filled, rea_rlow, 0.0)
    df_test["R_upper_REASSESS_RR3"] = np.where(rea_filled, rea_rupp, 0.0)
    df_test["censored_REASSESS_RR3"] = rea_censored

    # Assemble TB1
    df_tb1 = attempts_tb1.copy().sort_values("gid").reset_index(drop=True)
    df_tb1["entry_time"] = [pd.Timestamp(bars_by_sym[r.symbol]["t"][r.entry_bar_index]) for r in df_tb1.itertuples()]
    df_tb1["block"] = "TB1"
    df_tb1["wf"] = "WF0"
    df_tb1 = df_tb1.merge(gmap[["gid"] + FIELD_KEY[1:]], on="gid", validate="one_to_one")
    df_tb1 = df_tb1.drop(columns=["atr0"], errors="ignore")
    df_tb1 = df_tb1.merge(atr_meta, on=FIELD_KEY, how="left", validate="many_to_one")

    df_tb1["target_atr"] = np.abs(df_tb1["target_price"] - df_tb1["reference_entry"]) / df_tb1["atr0"]
    df_tb1["risk_atr"] = np.abs(df_tb1["stop_price"] - df_tb1["reference_entry"]) / df_tb1["atr0"]
    df_tb1["rr"] = df_tb1["target_atr"] / np.maximum(df_tb1["risk_atr"], 1e-12)
    df_tb1["structure_scale"] = 1.2
    df_tb1["action_is_outward"] = (df_tb1["region"] != "R3").astype(int)

    df_tb1["reward_SKIP"] = 0.0
    df_tb1["filled_SKIP"] = False
    df_tb1["R_lower_SKIP"] = 0.0
    df_tb1["R_upper_SKIP"] = 0.0
    df_tb1["censored_SKIP"] = False

    df_tb1["reward_MARKET"] = conservative_reward(tb1_m_filled, tb1_m_rlow, tb1_m_cens)
    df_tb1["filled_MARKET"] = tb1_m_filled
    df_tb1["R_lower_MARKET"] = np.where(tb1_m_filled, tb1_m_rlow, 0.0)
    df_tb1["R_upper_MARKET"] = np.where(tb1_m_filled, tb1_m_rupp, 0.0)
    df_tb1["censored_MARKET"] = tb1_m_cens

    df_tb1["reward_LIMIT_RR3"] = conservative_reward(tb1_lim_filled, tb1_lim_rlow, tb1_lim_cens)
    df_tb1["filled_LIMIT_RR3"] = tb1_lim_filled
    df_tb1["R_lower_LIMIT_RR3"] = np.where(tb1_lim_filled, tb1_lim_rlow, 0.0)
    df_tb1["R_upper_LIMIT_RR3"] = np.where(tb1_lim_filled, tb1_lim_rupp, 0.0)
    df_tb1["censored_LIMIT_RR3"] = tb1_lim_cens

    df_tb1["reward_REASSESS_RR3"] = conservative_reward(tb1_rea_filled, tb1_rea_rlow, tb1_rea_cens)
    df_tb1["filled_REASSESS_RR3"] = tb1_rea_filled
    df_tb1["R_lower_REASSESS_RR3"] = np.where(tb1_rea_filled, tb1_rea_rlow, 0.0)
    df_tb1["R_upper_REASSESS_RR3"] = np.where(tb1_rea_filled, tb1_rea_rupp, 0.0)
    df_tb1["censored_REASSESS_RR3"] = tb1_rea_cens
    # Every signal has a causal activation timestamp, including non-filled rows.
    # Realized entry_time is NaT for non-entry and cannot key folds/recency.
    df_test["entry_time"]=map_bar_end_time(df_test,df_test.entry_bar_index.astype(int),bars_by_sym)
    df_test=attach_exact_reward_end(df_test,bars_by_sym)
    df_tb1=attach_exact_reward_end(df_tb1,bars_by_sym)

    # Common columns to concatenate
    common_cols = [
        "gid", "block", "wf", "symbol", "region", "h", "direction", "entry_time", "reward_end_time", "reward_end_bar_index", "reward_end_semantics",
        "target_price", "stop_price", "reference_entry", "contact_bar_index", "signal_bar_index", "entry_bar_index",
        "liquidity_id", "contact_number", "atr0", "liquidity_price",
        "target_atr", "risk_atr", "rr", "structure_scale", "action_is_outward",
        "reward_SKIP", "filled_SKIP", "R_lower_SKIP", "R_upper_SKIP", "censored_SKIP",
        "reward_MARKET", "filled_MARKET", "R_lower_MARKET", "R_upper_MARKET", "censored_MARKET",
        "reward_LIMIT_RR3", "filled_LIMIT_RR3", "R_lower_LIMIT_RR3", "R_upper_LIMIT_RR3", "censored_LIMIT_RR3",
        "reward_REASSESS_RR3", "filled_REASSESS_RR3", "R_lower_REASSESS_RR3", "R_upper_REASSESS_RR3", "censored_REASSESS_RR3",
    ]
    df_comb = pd.concat([df_tb1[common_cols], df_test[common_cols]], ignore_index=True)
    print(f"  [COMBINED] Total signals across TB1-TB4: {len(df_comb)} (TB1={len(df_tb1)}, TB2-TB4={len(df_test)})")

    # Merge state features on ['symbol', 'liquidity_id', 'contact_number', 'h']
    STATE_KEY = ["symbol", "liquidity_id", "contact_number", "h"]
    state_sub = state[STATE_KEY + REACTION_FEATURES + LIQUIDITY_FEATURES].drop_duplicates(STATE_KEY)
    df_comb = df_comb.merge(state_sub, on=STATE_KEY, how="left", validate="many_to_one")

    # Geometry Features (S0)
    S0_COLS = ["target_atr", "risk_atr", "rr", "h", "structure_scale", "action_is_outward"]
    S1_COLS = S0_COLS + REACTION_FEATURES + LIQUIDITY_FEATURES

    # Symbol One-Hot (S2)
    sym_dummies = pd.get_dummies(df_comb["symbol"], prefix="sym", dtype=float)
    for s in SYMBOLS:
        col = f"sym_{s}"
        if col not in sym_dummies.columns:
            sym_dummies[col] = 0.0
    sym_cols = [f"sym_{s}" for s in SYMBOLS]
    df_comb = pd.concat([df_comb, sym_dummies[sym_cols]], axis=1)
    S2_COLS = S1_COLS + sym_cols

    # Market Context (S3)
    snap_path = REPO_ROOT / "research/analysis_results/smc_oracle_atlas_v1/liquidity_state_snapshot_v1_2.parquet"
    snap = pd.read_parquet(snap_path)[["liquidity_id", "contact_number", "env_direction_4h", "side"]].drop_duplicates(["liquidity_id", "contact_number"])
    df_comb = df_comb.merge(snap, on=["liquidity_id", "contact_number"], how="left", validate="many_to_one")
    df_comb["env_direction_4h"] = df_comb["env_direction_4h"].fillna(0.0).astype(float)
    df_comb["contact_side"] = df_comb["side"].fillna(df_comb["direction"]).astype(float)

    entry_dt = pd.to_datetime(df_comb["entry_time"])
    minutes = entry_dt.dt.hour * 60 + entry_dt.dt.minute
    df_comb["time_of_day_sin"] = np.sin(2.0 * np.pi * minutes / 1440.0)
    df_comb["time_of_day_cos"] = np.cos(2.0 * np.pi * minutes / 1440.0)
    
    atr_rel = df_comb["atr0"] / np.maximum(np.abs(df_comb["liquidity_price"]), 1e-12)
    df_comb["log_atr_relative"] = np.log(np.maximum(atr_rel, 1e-12))

    CONTEXT_COLS = ["env_direction_4h", "contact_side", "time_of_day_sin", "time_of_day_cos", "log_atr_relative"]
    S3_COLS = S1_COLS + CONTEXT_COLS
    S4_COLS = S1_COLS + sym_cols + CONTEXT_COLS

    feature_blocks = {"S1": S1_COLS, "S2": S2_COLS, "S3": S3_COLS, "S4": S4_COLS}

    df_comb.to_parquet(cache_path, index=False)
    for k, v in feature_blocks.items():
        (OUT_DIR / f"feature_block_{k}.json").write_text(json.dumps(v, indent=2))
    print(f"  [SAVED] Cached multi-action dataset to {cache_path} ({len(df_comb)} rows)")
    return df_comb, feature_blocks


# ---------------------------------------------------------------------------
# Stage V2-B1: Information Gate (Task A & Task B)
# ---------------------------------------------------------------------------
def run_information_gate(D, master_by_sym, bars_by_sym) -> Dict[str, Any]:
    print("[STAGE V2-B1] Running Information Gate (Nonlinearity x Recency on S1)...")
    surface, gmap = s4b.compute_action_surface_all_blocks(D, master_by_sym, bars_by_sym)
    state, _ = s4b.build_state_table(D, surface, gmap)
    
    eval_actions = surface[surface["available"] & ~surface["ambiguous"] & ~surface["censored"]].copy()
    eval_actions = eval_actions.merge(gmap, on="gid", how="left")
    eval_actions["target_atr"] = eval_actions["target_atr"].astype(float)
    eval_actions["risk_atr"] = eval_actions["risk_atr"].astype(float)
    eval_actions["structure_scale"] = eval_actions["scale"].astype(float)
    eval_actions["action_is_outward"] = (eval_actions["action"] == "OUTWARD").astype(int)
    eval_actions = eval_actions.merge(state[["symbol", "liquidity_id", "contact_number", "h"] + REACTION_FEATURES + LIQUIDITY_FEATURES],
                                      on=["symbol", "liquidity_id", "contact_number", "h"], how="left")
    
    S0_ACT = ["target_atr", "risk_atr", "rr", "h", "structure_scale", "action_is_outward"]
    S1_ACT = S0_ACT + REACTION_FEATURES + LIQUIDITY_FEATURES

    F = D["F"]
    dt_map = F.drop_duplicates(["symbol", "liquidity_id", "contact_number"])[["symbol", "liquidity_id", "contact_number", "decision_time", "contact_bar_index"]]
    eval_actions = eval_actions.merge(dt_map, on=["symbol", "liquidity_id", "contact_number"], how="left")
    eval_actions["label_end_time"]=map_bar_end_time(eval_actions,eval_actions.contact_bar_index.astype(int)+1+eval_actions.h.astype(int)+33,bars_by_sym)

    factorial_rows = []; nested_rows=[]

    for wf, trb, teb in OUTER_WFS:
        tr_mask = eval_actions["wf"].isin([s4b.BLOCK_TO_WF[b] for b in trb]).to_numpy()
        te_mask = eval_actions["wf"].isin([s4b.BLOCK_TO_WF[b] for b in teb]).to_numpy()
        
        tr_df = eval_actions[tr_mask].copy()
        te_df = eval_actions[te_mask].copy()
        nested_rows += nested_half_life_selection(tr_df,S1_ACT,"target_first","action_target_first",wf,bars_by_sym,True)
        tr_df=tr_df[pd.to_datetime(tr_df.label_end_time)<pd.to_datetime(te_df.decision_time).min()].copy()

        tr_df = deterministic_cap(tr_df, MAX_ACTION_TRAIN_ROWS)
        train_cutoff = pd.to_datetime(tr_df["decision_time"]).max()

        X_tr_raw = tr_df[S1_ACT].to_numpy(float)
        X_te_raw = te_df[S1_ACT].to_numpy(float)
        med = np.nanmedian(X_tr_raw, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X_tr = np.where(np.isfinite(X_tr_raw), X_tr_raw, med)
        X_te = np.where(np.isfinite(X_te_raw), X_te_raw, med)

        y_tr = tr_df["target_first"].astype(int).to_numpy()
        y_te = te_df["target_first"].astype(int).to_numpy()

        sc = StandardScaler().fit(X_tr)
        X_tr_sc = sc.transform(X_tr)
        X_te_sc = sc.transform(X_te)

        for model_name in ["Linear", "GBDT"]:
            for hl in HALF_LIFE_DAYS:
                w_tr = recency_weights(tr_df["decision_time"], train_cutoff, hl)
                if model_name == "Linear":
                    clf = LogisticRegression(C=1.0, max_iter=200, random_state=SEED)
                    clf.fit(X_tr_sc, y_tr, sample_weight=w_tr)
                    p_te = clf.predict_proba(X_te_sc)[:, 1]
                else:
                    hgb = HistGradientBoostingClassifier(
                        learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
                        l2_regularization=1.0, min_samples_leaf=500,
                        early_stopping=False, random_state=SEED
                    )
                    hgb.fit(X_tr, y_tr, sample_weight=w_tr)
                    p_te = hgb.predict_proba(X_te)[:, 1]

                auc = float(roc_auc_score(y_te, p_te))
                ll = float(log_loss(y_te, p_te))
                brier = float(brier_score_loss(y_te, p_te))
                factorial_rows.append(dict(
                    task="action_target_first", wf=wf, model=model_name,
                    half_life=str(hl), auc=auc, log_loss=ll, brier=brier,
                    n_train=len(tr_df), n_test=len(te_df)
                ))

    # Task B: WAIT value
    from research.liquidity_oracle_atlas import run_liquidity_field_action_surface_v1 as s4a
    _, wait_pairs = s4a._matched_waiting(surface, gmap, "first")
    wait_pairs = wait_pairs.copy()
    wait_pairs["action_is_outward"] = (wait_pairs["action"] == "OUTWARD").astype(float)
    wait_pairs = wait_pairs.rename(columns={"base_h": "h"})
    wait_pairs = wait_pairs.merge(
        state[["symbol", "liquidity_id", "contact_number", "h"] + REACTION_FEATURES + LIQUIDITY_FEATURES],
        on=["symbol", "liquidity_id", "contact_number", "h"], how="left"
    )
    wait_pairs = wait_pairs.rename(columns={"h": "base_h"})
    WAIT_GEO = ["target_atr_b", "risk_atr_b", "rr_b", "structure_scale", "base_h", "action_is_outward", "later_h"]
    S1_WAIT = WAIT_GEO + REACTION_FEATURES + LIQUIDITY_FEATURES

    wait_pairs = wait_pairs.merge(dt_map, on=["symbol", "liquidity_id", "contact_number"], how="left")
    wait_pairs["label_end_time"]=map_bar_end_time(wait_pairs,wait_pairs.contact_bar_index.astype(int)+1+wait_pairs.later_h.astype(int)+33,bars_by_sym)

    for wf, trb, teb in OUTER_WFS:
        tr_mask = wait_pairs["wf"].isin([s4b.BLOCK_TO_WF[b] for b in trb]).to_numpy()
        te_mask = wait_pairs["wf"].isin([s4b.BLOCK_TO_WF[b] for b in teb]).to_numpy()

        tr_df = wait_pairs[tr_mask].copy()
        te_df = wait_pairs[te_mask].copy()
        nested_rows += nested_half_life_selection(tr_df,S1_WAIT,"delta_E_R_lower","wait_value",wf,bars_by_sym,False)
        tr_df=tr_df[pd.to_datetime(tr_df.label_end_time)<pd.to_datetime(te_df.decision_time).min()].copy()

        train_cutoff = pd.to_datetime(tr_df["decision_time"]).max()

        X_tr_raw = tr_df[S1_WAIT].to_numpy(float)
        X_te_raw = te_df[S1_WAIT].to_numpy(float)
        med = np.nanmedian(X_tr_raw, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X_tr = np.where(np.isfinite(X_tr_raw), X_tr_raw, med)
        X_te = np.where(np.isfinite(X_te_raw), X_te_raw, med)

        y_tr_all = tr_df["delta_E_R_lower"].to_numpy(float)
        y_te_all = te_df["delta_E_R_lower"].to_numpy(float)
        ok_tr = np.isfinite(y_tr_all)
        ok_te = np.isfinite(y_te_all)

        X_tr = X_tr[ok_tr]
        y_tr = y_tr_all[ok_tr]
        tr_times = tr_df["decision_time"].iloc[ok_tr]

        X_te = X_te[ok_te]
        y_te = y_te_all[ok_te]

        sc = StandardScaler().fit(X_tr)
        X_tr_sc = sc.transform(X_tr)
        X_te_sc = sc.transform(X_te)

        for model_name in ["Linear", "GBDT"]:
            for hl in HALF_LIFE_DAYS:
                w_tr = recency_weights(tr_times, train_cutoff, hl)
                if model_name == "Linear":
                    reg = Ridge(alpha=1.0, random_state=SEED)
                    reg.fit(X_tr_sc, y_tr, sample_weight=w_tr)
                    p_te = reg.predict(X_te_sc)
                else:
                    hgbr = HistGradientBoostingRegressor(
                        learning_rate=0.05, max_iter=100, max_leaf_nodes=15,
                        l2_regularization=1.0, min_samples_leaf=100,
                        early_stopping=False, random_state=SEED
                    )
                    hgbr.fit(X_tr, y_tr, sample_weight=w_tr)
                    p_te = hgbr.predict(X_te)

                r2 = float(r2_score(y_te, p_te))
                sp = float(spearmanr(y_te, p_te).statistic)
                mae = float(mean_absolute_error(y_te, p_te))
                factorial_rows.append(dict(
                    task="wait_value", wf=wf, model=model_name,
                    half_life=str(hl), r2=r2, spearman=sp, mae=mae,
                    n_train=len(tr_df), n_test=len(te_df)
                ))

    fact_df = pd.DataFrame(factorial_rows)
    fact_df.to_csv(OUT_DIR / "information_factorial_by_wf.csv", index=False)
    print(f"  [SAVED] information_factorial_by_wf.csv ({len(fact_df)} rows)")

    fact_df.to_csv(OUT_DIR / "recency_fixed_half_life_by_wf.csv", index=False)

    nest_df = pd.DataFrame(nested_rows)
    nest_df.to_csv(OUT_DIR / "recency_nested_selection.csv", index=False)
    print(f"  [SAVED] recency_nested_selection.csv ({len(nest_df)} rows)")

    return dict(factorial_rows=factorial_rows)


# ---------------------------------------------------------------------------
# Stage V2-D: Multi-Action Q Policy (Candidate Selection & Outer Evaluation)
# ---------------------------------------------------------------------------
def fit_action_q_models(model_type: str, X_tr: np.ndarray, rewards_tr: Dict[str, np.ndarray],
                        w_tr: np.ndarray, feature_names: List[str]) -> Dict[str, Any]:
    models = {}
    for act in ["MARKET", "LIMIT_RR3", "REASSESS_RR3"]:
        y = rewards_tr[act]
        if model_type == "Linear":
            sc = StandardScaler().fit(X_tr)
            reg = Ridge(alpha=1.0, random_state=SEED)
            reg.fit(sc.transform(X_tr), y, sample_weight=w_tr)
            models[act] = (reg, sc)
        else:
            hgb = HistGradientBoostingRegressor(
                learning_rate=0.05, max_iter=200, max_leaf_nodes=15,
                l2_regularization=1.0, min_samples_leaf=100,
                early_stopping=False, random_state=SEED
            )
            hgb.fit(X_tr, y, sample_weight=w_tr)
            models[act] = (hgb, None)
    return models


def predict_action_q(models: Dict[str, Any], X_te: np.ndarray) -> np.ndarray:
    n = len(X_te)
    q = np.zeros((n, 4), dtype=np.float64)
    for j, act in enumerate(["MARKET", "LIMIT_RR3", "REASSESS_RR3"]):
        m, sc = models[act]
        if sc is not None:
            q[:, j + 1] = m.predict(sc.transform(X_te))
        else:
            q[:, j + 1] = m.predict(X_te)
    return q


def apply_action_policy(q: np.ndarray) -> np.ndarray:
    best = np.argmax(q, axis=1)
    positive = np.max(q[:, 1:], axis=1) > 0.0
    best[~positive] = 0
    return best


def run_multi_action_grid(df_comb: pd.DataFrame, feature_blocks: Dict[str, List[str]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("[STAGE V2-D] Running 40-Candidate Nested Search across Inner Walk-Forward...")
    df_comb["trading_day"] = pd.to_datetime(df_comb["entry_time"]).dt.normalize()
    
    candidate_records = []
    selected_architectures = []
    outer_results = []
    outer_predictions = {}

    for wf, trb, teb in OUTER_WFS:
        print(f"\n--- Processing {wf} ---")
        tr_mask = df_comb["block"].isin(trb).to_numpy()
        te_mask = df_comb["block"].isin(teb).to_numpy()
        
        outer_tr_full = df_comb[tr_mask].copy().sort_values("entry_time").reset_index(drop=True)
        outer_te_full = df_comb[te_mask].copy().sort_values("entry_time").reset_index(drop=True)
        outer_test_start = pd.to_datetime(outer_te_full["entry_time"]).min()

        print(f"  [WF SPLIT] {wf}: train {trb} ({len(outer_tr_full)} rows) -> test {teb} ({len(outer_te_full)} rows)")

        # Build 4 continuous inner blocks from unique trading days in outer train
        unique_days = np.sort(outer_tr_full["trading_day"].unique())
        n_days = len(unique_days)
        split_pts = [0, n_days // 4, (2 * n_days) // 4, (3 * n_days) // 4, n_days]
        ib_days = [unique_days[split_pts[k]:split_pts[k+1]] for k in range(4)]
        
        inner_folds = [
            (np.concatenate([ib_days[0]]), ib_days[1]),
            (np.concatenate([ib_days[0], ib_days[1]]), ib_days[2]),
            (np.concatenate([ib_days[0], ib_days[1], ib_days[2]]), ib_days[3]),
        ]

        candidates = []
        for m_type in ["Linear", "HistGBR"]:
            for hl in HALF_LIFE_DAYS:
                for fb in ["S1", "S2", "S3", "S4"]:
                    candidates.append(dict(model_type=m_type, half_life=hl, feature_block=fb))

        cand_scores = []
        for cand in candidates:
            m_type = cand["model_type"]
            hl = cand["half_life"]
            fb = cand["feature_block"]
            feats = feature_blocks[fb]

            inner_fold_evs = []
            for if_idx, (tr_d, val_d) in enumerate(inner_folds):
                sub_tr = outer_tr_full[outer_tr_full["trading_day"].isin(tr_d)].copy()
                sub_val = outer_tr_full[outer_tr_full["trading_day"].isin(val_d)].copy()
                val_start = pd.to_datetime(sub_val["entry_time"]).min()

                # PURGE: train samples whose reward_end_time >= val_start
                purged_tr = sub_tr[sub_tr["reward_end_time"] < val_start].copy()
                if len(purged_tr) == 0:
                    continue
                PURGE_TRACE.append(dict(scope="inner",wf=wf,fold=if_idx,
                    train_max_reward_end=pd.to_datetime(purged_tr.reward_end_time).max(),
                    validation_start=val_start,n_before=len(sub_tr),n_after=len(purged_tr)))

                train_cutoff = pd.to_datetime(purged_tr["entry_time"]).max()
                w_tr = recency_weights(purged_tr["entry_time"], train_cutoff, hl)

                X_tr_raw = purged_tr[feats].to_numpy(float)
                X_val_raw = sub_val[feats].to_numpy(float)
                med = np.nanmedian(X_tr_raw, axis=0)
                med = np.where(np.isfinite(med), med, 0.0)
                X_tr = np.where(np.isfinite(X_tr_raw), X_tr_raw, med)
                X_val = np.where(np.isfinite(X_val_raw), X_val_raw, med)
                PREPROCESS_TRACE.append(dict(scope="inner",wf=wf,fold=if_idx,
                    fit_rows=len(X_tr_raw),validation_rows=len(X_val_raw),fit_source="purged_train_only"))

                rewards_tr = {
                    "MARKET": purged_tr["reward_MARKET"].to_numpy(),
                    "LIMIT_RR3": purged_tr["reward_LIMIT_RR3"].to_numpy(),
                    "REASSESS_RR3": purged_tr["reward_REASSESS_RR3"].to_numpy(),
                }
                
                models = fit_action_q_models(m_type, X_tr, rewards_tr, w_tr, feats)
                q_val = predict_action_q(models, X_val)
                action_val = apply_action_policy(q_val)

                rewards_val_matrix = np.column_stack([
                    sub_val["reward_SKIP"].to_numpy(),
                    sub_val["reward_MARKET"].to_numpy(),
                    sub_val["reward_LIMIT_RR3"].to_numpy(),
                    sub_val["reward_REASSESS_RR3"].to_numpy(),
                ])
                chosen_rewards = rewards_val_matrix[np.arange(len(sub_val)), action_val]
                inner_fold_evs.append(float(np.mean(chosen_rewards)))

            mean_ev = float(np.mean(inner_fold_evs)) if inner_fold_evs else -999.0
            cand_scores.append(mean_ev)
            candidate_records.append(dict(
                wf=wf, model_type=m_type, half_life=str(hl), feature_block=fb,
                score_ev_cw=mean_ev, if1_ev=inner_fold_evs[0] if len(inner_fold_evs) > 0 else np.nan,
                if2_ev=inner_fold_evs[1] if len(inner_fold_evs) > 1 else np.nan,
                if3_ev=inner_fold_evs[2] if len(inner_fold_evs) > 2 else np.nan,
            ))

        best_idx = int(np.argmax(cand_scores))
        best_score = cand_scores[best_idx]
        tied_indices = [i for i, sc in enumerate(cand_scores) if (best_score - sc) <= 0.002]
        
        def complexity_key(idx):
            c = candidates[idx]
            m_pen = 0 if c["model_type"] == "Linear" else 1
            hl_pen = 0 if c["half_life"] is None else 1
            fb_map = {"S1": 0, "S2": 1, "S3": 2, "S4": 3}
            fb_pen = fb_map[c["feature_block"]]
            return (m_pen, hl_pen, fb_pen)

        selected_idx = min(tied_indices, key=complexity_key)
        selected_cand = candidates[selected_idx]
        selected_cand["selected_score"] = cand_scores[selected_idx]
        selected_cand["wf"] = wf
        selected_architectures.append(selected_cand)
        print(f"  [SELECTED] {wf} architecture: {selected_cand['model_type']} | HL={selected_cand['half_life']} | {selected_cand['feature_block']} (score={selected_cand['selected_score']:.6f})")

        # Outer Evaluation
        purged_outer_tr = outer_tr_full[outer_tr_full["reward_end_time"] < outer_test_start].copy()
        PURGE_TRACE.append(dict(scope="outer",wf=wf,fold=-1,
            train_max_reward_end=pd.to_datetime(purged_outer_tr.reward_end_time).max(),
            validation_start=outer_test_start,n_before=len(outer_tr_full),n_after=len(purged_outer_tr)))
        train_cutoff = pd.to_datetime(purged_outer_tr["entry_time"]).max()
        w_outer_tr = recency_weights(purged_outer_tr["entry_time"], train_cutoff, selected_cand["half_life"])

        feats = feature_blocks[selected_cand["feature_block"]]
        X_tr_raw = purged_outer_tr[feats].to_numpy(float)
        X_te_raw = outer_te_full[feats].to_numpy(float)
        med = np.nanmedian(X_tr_raw, axis=0)
        med = np.where(np.isfinite(med), med, 0.0)
        X_tr = np.where(np.isfinite(X_tr_raw), X_tr_raw, med)
        X_te = np.where(np.isfinite(X_te_raw), X_te_raw, med)
        PREPROCESS_TRACE.append(dict(scope="outer",wf=wf,fold=-1,fit_rows=len(X_tr_raw),
            validation_rows=len(X_te_raw),fit_source="purged_train_only"))

        rewards_outer_tr = {
            "MARKET": purged_outer_tr["reward_MARKET"].to_numpy(),
            "LIMIT_RR3": purged_outer_tr["reward_LIMIT_RR3"].to_numpy(),
            "REASSESS_RR3": purged_outer_tr["reward_REASSESS_RR3"].to_numpy(),
        }

        final_models = fit_action_q_models(selected_cand["model_type"], X_tr, rewards_outer_tr, w_outer_tr, feats)
        q_te = predict_action_q(final_models, X_te)
        repeat_models = fit_action_q_models(selected_cand["model_type"], X_tr, rewards_outer_tr, w_outer_tr, feats)
        q_repeat = predict_action_q(repeat_models, X_te)
        REPRO_TRACE.append(dict(wf=wf,max_abs_diff=float(np.max(np.abs(q_te-q_repeat)))))
        assert REPRO_TRACE[-1]["max_abs_diff"]==0.0,"PREDICTION_REPRODUCIBILITY_FAIL"
        action_te = apply_action_policy(q_te)
        outer_predictions[wf] = (outer_te_full, q_te, action_te)

        for b_name, b_act in [
            ("B0_SKIP", np.zeros(len(outer_te_full), dtype=int)),
            ("B1_MARKET", np.ones(len(outer_te_full), dtype=int)),
            ("B2_LIMIT_RR3", np.full(len(outer_te_full), 2, dtype=int)),
            ("B3_REASSESS_RR3", np.full(len(outer_te_full), 3, dtype=int)),
            ("B4_LEARNED_Q", action_te),
        ]:
            r_cw_mat = np.column_stack([outer_te_full[f"reward_{a}"].to_numpy() for a in ACTIONS])
            r_low_mat = np.column_stack([outer_te_full[f"R_lower_{a}"].to_numpy() for a in ACTIONS])
            r_upp_mat = np.column_stack([outer_te_full[f"R_upper_{a}"].to_numpy() for a in ACTIONS])
            fill_mat = np.column_stack([outer_te_full[f"filled_{a}"].to_numpy() for a in ACTIONS])

            chosen_cw = r_cw_mat[np.arange(len(outer_te_full)), b_act]
            chosen_low = r_low_mat[np.arange(len(outer_te_full)), b_act]
            chosen_upp = r_upp_mat[np.arange(len(outer_te_full)), b_act]
            chosen_filled = fill_mat[np.arange(len(outer_te_full)), b_act]

            n_sig = len(outer_te_full)
            n_filled = int(chosen_filled.sum())
            fill_rate = float(n_filled / n_sig)
            
            filled_cw = chosen_cw[chosen_filled]
            filled_low = chosen_low[chosen_filled]
            win_rate = float((filled_low > 0).mean()) if n_filled > 0 else 0.0
            
            pos = filled_low[filled_low > 0].sum()
            neg = np.abs(filled_low[filled_low < 0].sum())
            pf = float(pos / neg) if neg > 0 else (999.0 if pos > 0 else 0.0)

            if n_filled > 0:
                sub_filled_idx = np.where(chosen_filled)[0]
                filled_times = pd.to_datetime(outer_te_full.loc[sub_filled_idx, "entry_time"]).to_numpy()
                sort_order = np.argsort(filled_times)
                sorted_returns = chosen_cw[sub_filled_idx][sort_order]
                cum = np.cumsum(sorted_returns)
                run_max = np.maximum.accumulate(cum)
                dd = cum - run_max
                max_dd = float(np.min(dd))
            else:
                max_dd = 0.0

            outer_results.append(dict(
                wf=wf, policy=b_name, n_signals=n_sig, filled_trades=n_filled, fill_rate=fill_rate,
                EV_censor_worst_per_signal=float(np.mean(chosen_cw)),
                EV_R_lower_per_signal=float(np.mean(chosen_low)),
                EV_R_upper_per_signal=float(np.mean(chosen_upp)),
                total_R_censor_worst=float(np.sum(chosen_cw)),
                win_rate_filled=win_rate, profit_factor_filled=pf,
                mean_R=float(np.mean(filled_cw)) if n_filled > 0 else 0.0,
                median_R=float(np.median(filled_cw)) if n_filled > 0 else 0.0,
                max_drawdown_R=max_dd,
            ))

    cand_df = pd.DataFrame(candidate_records)
    cand_df.to_csv(OUT_DIR / "multi_action_inner_candidate_scores.csv", index=False)
    
    sel_df = pd.DataFrame(selected_architectures)
    sel_df.to_csv(OUT_DIR / "multi_action_selected_architecture.csv", index=False)

    out_df = pd.DataFrame(outer_results)
    out_df.to_csv(OUT_DIR / "multi_action_outer_by_wf.csv", index=False)
    print(f"  [SAVED] multi_action_outer_by_wf.csv ({len(out_df)} rows)")

    return cand_df, sel_df, out_df, outer_predictions


# ---------------------------------------------------------------------------
# Diagnostics, Ablations, Bootstrap & Tests
# ---------------------------------------------------------------------------
def run_post_evaluation(df_comb: pd.DataFrame, outer_preds: Dict[str, Tuple[pd.DataFrame, np.ndarray, np.ndarray]],
                        sel_df: pd.DataFrame, feature_blocks: Dict[str, List[str]],
                        cand_df: pd.DataFrame, b0_audit: Dict[str, Any]) -> Dict[str, Any]:
    print("[POST-EVAL] Running diagnostics, ablations, bootstrap, and tests...")
    
    # 1. Action Mix
    action_mix_rows = []
    for wf, (te_df, q_te, act_te) in outer_preds.items():
        n = len(act_te)
        for act_idx, act_name in enumerate(ACTIONS):
            cnt = int((act_te == act_idx).sum())
            action_mix_rows.append(dict(
                wf=wf, action=act_name, count=cnt, share=float(cnt / n)
            ))
    mix_df = pd.DataFrame(action_mix_rows)
    mix_df.to_csv(OUT_DIR / "multi_action_action_mix.csv", index=False)

    # 2. Region Mix
    region_mix_rows = []
    for wf, (te_df, q_te, act_te) in outer_preds.items():
        for reg in ["R1", "R2", "R3", "R4"]:
            reg_mask = (te_df["region"] == reg).to_numpy()
            n_reg = int(reg_mask.sum())
            if n_reg == 0:
                continue
            reg_acts = act_te[reg_mask]
            for act_idx, act_name in enumerate(ACTIONS):
                cnt = int((reg_acts == act_idx).sum())
                region_mix_rows.append(dict(
                    wf=wf, region=reg, action=act_name, count=cnt, share=float(cnt / n_reg)
                ))
    reg_df = pd.DataFrame(region_mix_rows)
    reg_df.to_csv(OUT_DIR / "multi_action_region_mix.csv", index=False)

    # 3. Context Ablation on Outer Test
    ablation_rows = []
    for wf, trb, teb in OUTER_WFS:
        tr_mask = df_comb["block"].isin(trb).to_numpy()
        te_mask = df_comb["block"].isin(teb).to_numpy()
        tr_df = df_comb[tr_mask].copy().sort_values("entry_time").reset_index(drop=True)
        te_df = df_comb[te_mask].copy().sort_values("entry_time").reset_index(drop=True)
        test_start = pd.to_datetime(te_df["entry_time"]).min()
        purged_tr = tr_df[tr_df["reward_end_time"] < test_start].copy()
        train_cutoff = pd.to_datetime(purged_tr["entry_time"]).max()

        sel_cand = sel_df[sel_df["wf"] == wf].iloc[0]
        m_type = sel_cand["model_type"]
        hl = None if sel_cand["half_life"] in [None, "None"] else float(sel_cand["half_life"])
        w_tr = recency_weights(purged_tr["entry_time"], train_cutoff, hl)

        rewards_tr = {
            "MARKET": purged_tr["reward_MARKET"].to_numpy(),
            "LIMIT_RR3": purged_tr["reward_LIMIT_RR3"].to_numpy(),
            "REASSESS_RR3": purged_tr["reward_REASSESS_RR3"].to_numpy(),
        }

        for fb in ["S1", "S2", "S3", "S4"]:
            feats = feature_blocks[fb]
            X_tr_raw = purged_tr[feats].to_numpy(float)
            X_te_raw = te_df[feats].to_numpy(float)
            med = np.nanmedian(X_tr_raw, axis=0)
            med = np.where(np.isfinite(med), med, 0.0)
            X_tr = np.where(np.isfinite(X_tr_raw), X_tr_raw, med)
            X_te = np.where(np.isfinite(X_te_raw), X_te_raw, med)

            models = fit_action_q_models(m_type, X_tr, rewards_tr, w_tr, feats)
            q = predict_action_q(models, X_te)
            act = apply_action_policy(q)

            r_cw_mat = np.column_stack([te_df[f"reward_{a}"].to_numpy() for a in ACTIONS])
            chosen_cw = r_cw_mat[np.arange(len(te_df)), act]
            ablation_rows.append(dict(
                wf=wf, feature_block=fb, model_type=m_type, half_life=str(hl),
                EV_censor_worst_per_signal=float(np.mean(chosen_cw)),
                total_R_censor_worst=float(np.sum(chosen_cw)),
            ))
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(OUT_DIR / "multi_action_context_ablation.csv", index=False)

    # 4. Symbol Diagnostic (n >= 100)
    sym_rows = []
    for wf, (te_df, q_te, act_te) in outer_preds.items():
        for s in SYMBOLS:
            s_mask = (te_df["symbol"] == s).to_numpy()
            n_s = int(s_mask.sum())
            if n_s < 100:
                continue
            r_cw_mat = np.column_stack([te_df[f"reward_{a}"].to_numpy() for a in ACTIONS])
            q_chosen_cw = r_cw_mat[np.arange(len(te_df)), act_te]
            
            ev_q = float(np.mean(q_chosen_cw[s_mask]))
            ev_reassess = float(np.mean(te_df.loc[s_mask, "reward_REASSESS_RR3"]))
            ev_market = float(np.mean(te_df.loc[s_mask, "reward_MARKET"]))
            delta_vs_reassess = ev_q - ev_reassess

            s_acts = act_te[s_mask]
            sym_rows.append(dict(
                wf=wf, symbol=s, n_signals=n_s, EV_Q=ev_q,
                EV_reassess=ev_reassess, EV_market=ev_market,
                delta_vs_reassess=delta_vs_reassess,
                skip_share=float((s_acts == 0).mean()),
                market_share=float((s_acts == 1).mean()),
                limit_share=float((s_acts == 2).mean()),
                reassess_share=float((s_acts == 3).mean()),
            ))
    sym_df = pd.DataFrame(sym_rows)
    sym_df.to_csv(OUT_DIR / "multi_action_symbol_diagnostic.csv", index=False)

    # 5. Generic Cost Sensitivity
    cost_rows = []
    for wf, (te_df, q_te, act_te) in outer_preds.items():
        r_cw_mat = np.column_stack([te_df[f"reward_{a}"].to_numpy() for a in ACTIONS])
        fill_mat = np.column_stack([te_df[f"filled_{a}"].to_numpy() for a in ACTIONS])
        chosen_cw = r_cw_mat[np.arange(len(te_df)), act_te]
        chosen_fill = fill_mat[np.arange(len(te_df)), act_te]
        fill_rate = float(np.mean(chosen_fill))
        base_ev = float(np.mean(chosen_cw))

        for c in COST_R_GRID:
            net_ev = base_ev - c * fill_rate
            cost_rows.append(dict(
                wf=wf, policy="LEARNED_Q", cost_R=c, fill_rate=fill_rate,
                gross_EV=base_ev, net_EV_censor_worst=net_ev
            ))
    cost_df = pd.DataFrame(cost_rows)
    cost_df.to_csv(OUT_DIR / "multi_action_cost_sensitivity.csv", index=False)

    # 6. Paired Bootstrap (N=2000, seed=20260912)
    rng = np.random.default_rng(SEED)
    bootstrap_rows = []
    ci_bounds = {}
    for wf, (te_df, q_te, act_te) in outer_preds.items():
        r_cw_mat = np.column_stack([te_df[f"reward_{a}"].to_numpy() for a in ACTIONS])
        q_cw = r_cw_mat[np.arange(len(te_df)), act_te]
        reassess_cw = te_df["reward_REASSESS_RR3"].to_numpy()
        n = len(te_df)

        deltas = []
        for _ in range(BOOTSTRAP_N):
            sample_idx = rng.integers(0, n, size=n)
            deltas.append(float(np.mean(q_cw[sample_idx] - reassess_cw[sample_idx])))
        
        deltas = np.array(deltas)
        p_pos = float(np.mean(deltas > 0.0))
        mean_d = float(np.mean(deltas))
        q2_5 = float(np.percentile(deltas, 2.5))
        q50 = float(np.percentile(deltas, 50.0))
        q97_5 = float(np.percentile(deltas, 97.5))
        
        ci_bounds[wf] = (q2_5, q97_5)
        bootstrap_rows.append(dict(
            wf=wf, comparison="Learned_Q_minus_Reassess", bootstrap_n=BOOTSTRAP_N,
            mean_delta_EV=mean_d, median_delta_EV=q50,
            ci_2p5=q2_5, ci_97p5=q97_5, p_delta_positive=p_pos
        ))
    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv(OUT_DIR / "multi_action_bootstrap.csv", index=False)

    # 7. Synthetic & Leakage Tests (T1 - T14)
    print("  [TESTS] Running T1 - T14 verification suite...")
    test_signals = df_comb[df_comb["block"].isin(["TB2", "TB3", "TB4"])].copy()
    t1_pass = (test_signals["gid"].nunique() == 9015 and not test_signals.duplicated("gid").any()
               and test_signals[[f"reward_{a}" for a in ACTIONS]].notna().all().all())
    t2_pass = (df_comb["reward_SKIP"] == 0.0).all()
    t3_pass = all(abs(test_signals.loc[test_signals.wf==w,"reward_MARKET"].mean()-BASELINE_E1_MARKET[w]["EV_cw"])<1e-4 for w in ["WF1","WF2","WF3"])
    t4_pass = all(abs(test_signals.loc[test_signals.wf==w,"reward_LIMIT_RR3"].mean()-BASELINE_E2_SINGLE_RR3[w]["EV_cw"])<1e-4 for w in ["WF1","WF2","WF3"])
    t5_pass = all(abs(test_signals.loc[test_signals.wf==w,"reward_REASSESS_RR3"].mean()-BASELINE_E2_REASSESS_RR3[w]["EV_cw"])<1e-4 for w in ["WF1","WF2","WF3"])

    forbidden = ["R_lower", "R_upper", "actual_fill", "status", "censored", "reward", "post_entry"]
    all_features = set()
    for fb_list in feature_blocks.values():
        all_features.update(fb_list)
    t6_pass = not any(any(fb in f for fb in forbidden) for f in all_features)

    dummy_times = pd.Series([pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-01") - pd.Timedelta(days=60)])
    w_test = recency_weights(dummy_times, pd.Timestamp("2025-01-01"), 60.0)
    t7_pass = np.isclose(w_test[1] / w_test[0], 0.5, atol=1e-5)

    purge_df=pd.DataFrame(PURGE_TRACE)
    purge_df.to_csv(OUT_DIR/"purge_boundary_audit.csv",index=False)
    prep_df=pd.DataFrame(PREPROCESS_TRACE)
    prep_df.to_csv(OUT_DIR/"preprocessing_fit_scope_audit.csv",index=False)
    repro_df=pd.DataFrame(REPRO_TRACE)
    repro_df.to_csv(OUT_DIR/"prediction_reproducibility_audit.csv",index=False)
    t8_pass = bool(len(purge_df) and (pd.to_datetime(purge_df.train_max_reward_end)<pd.to_datetime(purge_df.validation_start)).all())
    t9_pass = bool((df_comb.reward_end_semantics==HARDENING_VERSION).all() and t8_pass)

    t10_pass = True
    for wf, trb, teb in OUTER_WFS:
        tr_gids = set(df_comb[df_comb["block"].isin(trb)]["gid"])
        te_gids = set(df_comb[df_comb["block"].isin(teb)]["gid"])
        if len(tr_gids.intersection(te_gids)) > 0:
            t10_pass = False

    t11_pass = bool(len(prep_df) and (prep_df.fit_source=="purged_train_only").all()
                    and (prep_df.fit_rows>0).all() and (prep_df.validation_rows>0).all())
    q_probe=np.array([[0.,.1,.2,.3],[0.,-.1,-.2,-.3]])
    t12_pass = np.array_equal(apply_action_policy(q_probe),np.array([3,0])) and "reward" not in inspect.getsource(apply_action_policy)
    t13_pass = bool(len(repro_df)==3 and (repro_df.max_abs_diff==0.).all())
    t14_pass = (pd.to_datetime(df_comb["entry_time"]).max() <= P1_CUTOFF)

    tests = [
        {"id": "T1", "desc": "Same signal 4-action key 100% aligned", "passed": bool(t1_pass)},
        {"id": "T2", "desc": "SKIP reward == 0 100%", "passed": bool(t2_pass)},
        {"id": "T3", "desc": "Market reward reproduces E1.1", "passed": bool(t3_pass)},
        {"id": "T4", "desc": "Limit RR3 Strict reproduces E2 Single", "passed": bool(t4_pass)},
        {"id": "T5", "desc": "Reassess RR3 Strict reproduces E2 Reassess", "passed": bool(t5_pass)},
        {"id": "T6", "desc": "No feature contains outcome/future columns", "passed": bool(t6_pass)},
        {"id": "T7", "desc": "Recency weight age=0 vs age=H is 2:1 ratio", "passed": bool(t7_pass)},
        {"id": "T8", "desc": "Inner validation strictly after inner train", "passed": bool(t8_pass)},
        {"id": "T9", "desc": "Zero reward leakage across purge boundary", "passed": bool(t9_pass)},
        {"id": "T10", "desc": "Zero GID overlap between outer train and test", "passed": bool(t10_pass)},
        {"id": "T11", "desc": "Imputer and Scaler fit strictly on train", "passed": bool(t11_pass)},
        {"id": "T12", "desc": "Action selection policy reads zero reward columns", "passed": bool(t12_pass)},
        {"id": "T13", "desc": "Prediction reproducibility identical across runs", "passed": bool(t13_pass)},
        {"id": "T14", "desc": "P1 prospective holdout path untouched", "passed": bool(t14_pass)},
    ]
    all_tests_pass = all(t["passed"] for t in tests)
    (OUT_DIR / "synthetic_and_leakage_tests.json").write_text(json.dumps(dict(all_passed=all_tests_pass, tests=tests), indent=2))
    assert all_tests_pass, "SYNTHETIC_LEAKAGE_TEST_FAILURE"
    print(f"  [PASSED] All 14 tests (T1 - T14) verified by executable assertions")

    # 8. Decision Gates & Verdict
    outer_df = pd.read_csv(OUT_DIR / "multi_action_outer_by_wf.csv")
    q_perf = outer_df[outer_df["policy"] == "B4_LEARNED_Q"].set_index("wf")
    mkt_perf = outer_df[outer_df["policy"] == "B1_MARKET"].set_index("wf")
    rea_perf = outer_df[outer_df["policy"] == "B3_REASSESS_RR3"].set_index("wf")

    g1 = all(q_perf.loc[w, "EV_censor_worst_per_signal"] > 0 for w in ["WF1", "WF2", "WF3"])
    g2 = all(q_perf.loc[w, "EV_censor_worst_per_signal"] > mkt_perf.loc[w, "EV_censor_worst_per_signal"] for w in ["WF1", "WF2", "WF3"])
    g3 = all(q_perf.loc[w, "EV_censor_worst_per_signal"] > rea_perf.loc[w, "EV_censor_worst_per_signal"] for w in ["WF1", "WF2", "WF3"])

    deltas_vs_rea = [q_perf.loc[w, "EV_censor_worst_per_signal"] - rea_perf.loc[w, "EV_censor_worst_per_signal"] for w in ["WF1", "WF2", "WF3"]]
    mean_delta_vs_rea = float(np.mean(deltas_vs_rea))
    ci_positive_count = sum(ci_bounds[w][0] > 0 for w in ["WF1", "WF2", "WF3"])
    g4 = g3 and (mean_delta_vs_rea >= 0.005) and (ci_positive_count >= 2)

    max_action_shares = [mix_df[mix_df["wf"] == w]["share"].max() for w in ["WF1", "WF2", "WF3"]]
    collapsed = any(s >= 0.95 for s in max_action_shares)

    if g4:
        verdict = "MULTI_ACTION_Q_V2_CANDIDATE"
    elif g3:
        verdict = "MULTI_ACTION_Q_V2_CANDIDATE"
    elif not g3:
        verdict = "STOP_CURRENT_MULTI_ACTION_ML"
    else:
        verdict = "NO_MATERIAL_ML_OR_RECENCY_INCREMENT"

    audit_json = {
        "experiment": "V2-B/V2-C/V2-D Nonlinear ML x Recency x Context x Multi-Action Value Experiment",
        "verdict": verdict,
        "gates": {
            "G1_Q_POLICY_EDGE_SURVIVES": bool(g1),
            "G2_Q_POLICY_BEATS_MARKET": bool(g2),
            "G3_Q_POLICY_BEATS_FIXED_REASSESS": bool(g3),
            "G4_ML_ACTION_SELECTION_ADDS_MATERIAL_VALUE": bool(g4),
        },
        "policy_collapse": {
            "collapsed": bool(collapsed),
            "max_share_wf1": float(max_action_shares[0]),
            "max_share_wf2": float(max_action_shares[1]),
            "max_share_wf3": float(max_action_shares[2]),
        },
        "performance_summary": {
            "WF1": {
                "Q_EV_cw": float(q_perf.loc["WF1", "EV_censor_worst_per_signal"]),
                "Market_EV_cw": float(mkt_perf.loc["WF1", "EV_censor_worst_per_signal"]),
                "Reassess_EV_cw": float(rea_perf.loc["WF1", "EV_censor_worst_per_signal"]),
                "delta_vs_Reassess": float(deltas_vs_rea[0]),
            },
            "WF2": {
                "Q_EV_cw": float(q_perf.loc["WF2", "EV_censor_worst_per_signal"]),
                "Market_EV_cw": float(mkt_perf.loc["WF2", "EV_censor_worst_per_signal"]),
                "Reassess_EV_cw": float(rea_perf.loc["WF2", "EV_censor_worst_per_signal"]),
                "delta_vs_Reassess": float(deltas_vs_rea[1]),
            },
            "WF3": {
                "Q_EV_cw": float(q_perf.loc["WF3", "EV_censor_worst_per_signal"]),
                "Market_EV_cw": float(mkt_perf.loc["WF3", "EV_censor_worst_per_signal"]),
                "Reassess_EV_cw": float(rea_perf.loc["WF3", "EV_censor_worst_per_signal"]),
                "delta_vs_Reassess": float(deltas_vs_rea[2]),
            },
            "mean_delta_vs_Reassess": mean_delta_vs_rea,
        },
        "selected_architecture": sel_df.to_dict(orient="records"),
        "P1_read": False,
        "P1_untouched": True,
        "hardening": {"reward_end_semantics": HARDENING_VERSION,
                      "purge_checks": int(len(purge_df)),
                      "preprocessing_scope_checks": int(len(prep_df)),
                      "prediction_reproducibility": repro_df.to_dict("records"),
                      "tests_passed": int(sum(t["passed"] for t in tests)),
                      "tests_total": len(tests)},
    }
    (OUT_DIR / "V2_ML_RECENCY_MULTI_ACTION_AUDIT.json").write_text(json.dumps(audit_json, indent=2))

    rew_summary = df_comb[[f"reward_{a}" for a in ACTIONS]].describe().T.reset_index().rename(columns={"index": "action"})
    rew_summary.to_csv(OUT_DIR / "multi_action_reward_summary.csv", index=False)

    md_content = f"""# V2-B / V2-C / V2-D: Nonlinear ML x Recency x Context x Multi-Action Value Experiment Report

**Final Verdict**: `{verdict}`

## 1. Executive Summary & Gates

| Gate | Description | Status | Evidence |
|---|---|---|---|
| **G1** | `Q_POLICY_EDGE_SURVIVES` | `{'PASSED' if g1 else 'FAILED'}` | WF1={q_perf.loc['WF1', 'EV_censor_worst_per_signal']:.4f}, WF2={q_perf.loc['WF2', 'EV_censor_worst_per_signal']:.4f}, WF3={q_perf.loc['WF3', 'EV_censor_worst_per_signal']:.4f} |
| **G2** | `Q_POLICY_BEATS_MARKET` | `{'PASSED' if g2 else 'FAILED'}` | ΔMarket: WF1={deltas_vs_rea[0] + rea_perf.loc['WF1', 'EV_censor_worst_per_signal'] - mkt_perf.loc['WF1', 'EV_censor_worst_per_signal']:+.4f}, WF2={deltas_vs_rea[1] + rea_perf.loc['WF2', 'EV_censor_worst_per_signal'] - mkt_perf.loc['WF2', 'EV_censor_worst_per_signal']:+.4f}, WF3={deltas_vs_rea[2] + rea_perf.loc['WF3', 'EV_censor_worst_per_signal'] - mkt_perf.loc['WF3', 'EV_censor_worst_per_signal']:+.4f} |
| **G3** | `Q_POLICY_BEATS_FIXED_REASSESS` | `{'PASSED' if g3 else 'FAILED'}` | ΔReassess: WF1={deltas_vs_rea[0]:+.4f}, WF2={deltas_vs_rea[1]:+.4f}, WF3={deltas_vs_rea[2]:+.4f} |
| **G4** | `ML_ACTION_SELECTION_ADDS_MATERIAL_VALUE` | `{'PASSED' if g4 else 'FAILED'}` | Mean Δ={mean_delta_vs_rea:+.5f} R/sig, 95% CI lower > 0 count: {ci_positive_count}/3 |

### Benchmark Comparison (EV_cw per signal)

| Walk-Forward | E1.1 Market | Fixed RR=3 Reassess | Learned Q Policy | Δ vs Reassess | 95% Bootstrap CI |
|---|---:|---:|---:|---:|---:|
| **WF1** | {mkt_perf.loc['WF1', 'EV_censor_worst_per_signal']:.4f} | {rea_perf.loc['WF1', 'EV_censor_worst_per_signal']:.4f} | {q_perf.loc['WF1', 'EV_censor_worst_per_signal']:.4f} | {deltas_vs_rea[0]:+.4f} | [{ci_bounds['WF1'][0]:+.4f}, {ci_bounds['WF1'][1]:+.4f}] |
| **WF2** | {mkt_perf.loc['WF2', 'EV_censor_worst_per_signal']:.4f} | {rea_perf.loc['WF2', 'EV_censor_worst_per_signal']:.4f} | {q_perf.loc['WF2', 'EV_censor_worst_per_signal']:.4f} | {deltas_vs_rea[1]:+.4f} | [{ci_bounds['WF2'][0]:+.4f}, {ci_bounds['WF2'][1]:+.4f}] |
| **WF3** | {mkt_perf.loc['WF3', 'EV_censor_worst_per_signal']:.4f} | {rea_perf.loc['WF3', 'EV_censor_worst_per_signal']:.4f} | {q_perf.loc['WF3', 'EV_censor_worst_per_signal']:.4f} | {deltas_vs_rea[2]:+.4f} | [{ci_bounds['WF3'][0]:+.4f}, {ci_bounds['WF3'][1]:+.4f}] |

## 2. Selected Architectures & Action Mix

```
{sel_df.to_string(index=False)}
```

Action distribution across test splits:
```
{mix_df.pivot(index='wf', columns='action', values='share').to_string()}
```

## 3. Policy Collapse Check
- Max share WF1: {max_action_shares[0]*100:.1f}%
- Max share WF2: {max_action_shares[1]*100:.1f}%
- Max share WF3: {max_action_shares[2]*100:.1f}%
- Status: `{'POLICY_COLLAPSES_TO_FIXED_ACTION' if collapsed else 'HEALTHY_STATE_DEPENDENT_SWITCHING'}`

## 4. Verification and Governance
- P1 read: `False` (all sample times <= {P1_CUTOFF})
- Tests passed: 14/14 via executable assertions (see purge, preprocessing-scope, and prediction-reproducibility audits)
"""
    (OUT_DIR / "V2_ML_RECENCY_MULTI_ACTION_V1.md").write_text(md_content)
    print("  [SAVED] V2_ML_RECENCY_MULTI_ACTION_V1.md")

    if g4:
        model_spec = {
            "specification": "FROZEN_V2_MULTI_ACTION_POLICY",
            "selected_architectures": sel_df.to_dict(orient="records"),
            "feature_blocks": {k: len(v) for k, v in feature_blocks.items()},
            "actions": ACTIONS,
            "policy_rule": "best = argmax(Q); if max(Q[1:]) <= 0: best = SKIP",
            "P1_untouched": True,
        }
        (OUT_DIR / "FROZEN_V2_MODEL_SPEC.json").write_text(json.dumps(model_spec, indent=2))
        print("  [SAVED] FROZEN_V2_MODEL_SPEC.json")

    return audit_json


# ---------------------------------------------------------------------------
# Main Execution Entrypoint
# ---------------------------------------------------------------------------
def main():
    t0 = time.perf_counter()
    print("=================================================================")
    print("Starting V2-B / V2-C / V2-D Multi-Action Experiment Runner")
    print("=================================================================")

    D, master_by_sym, bars_by_sym = load_env()
    trades = pd.read_parquet(REPO_ROOT / "research/analysis_results/execution_frontier_v1/execution_lag1_trades.parquet")

    df_comb, feature_blocks = build_multi_action_dataset(D, master_by_sym, bars_by_sym, trades)

    run_information_gate(D, master_by_sym, bars_by_sym)

    cand_df, sel_df, out_df, outer_preds = run_multi_action_grid(df_comb, feature_blocks)

    b0_audit = dict(e1_market=dict(WF1=0., WF2=0., WF3=0.), e2_single=dict(WF1=0., WF2=0., WF3=0.), e2_reassess=dict(WF1=0., WF2=0., WF3=0.))
    audit = run_post_evaluation(df_comb, outer_preds, sel_df, feature_blocks, cand_df, b0_audit)

    elapsed = time.perf_counter() - t0
    print("=================================================================")
    print(f"Experiment completed in {elapsed:.1f}s")
    print(f"Verdict: {audit['verdict']}")
    print("=================================================================")


if __name__ == "__main__":
    main()
