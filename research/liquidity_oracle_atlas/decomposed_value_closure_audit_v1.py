"""decomposed_value_closure_audit_v1.py — R13.5 TD5 decomposed EV-error closure audit.

DIAGNOSTIC ONLY. This module:

  * trains NO model;
  * tunes NO parameter;
  * selects NO feature;
  * reads NO DEV VAL, NO old TEST;
  * changes NOTHING in the frozen Phase-4 result (NO_V2_MODEL_IMPROVEMENT);
  * simulates NO trading strategy.

Its only purpose is to decompose, per row and EXACTLY, the TRAIN-OOF EV
prediction error into a Win-probability error component and a Payoff-magnitude
error component, and to attribute where the composed EV gate fails economically:

    EV_hat - Y  =  probability_error  +  magnitude_error      (exact identity)

No DEV VAL is opened. The production E9 Direction layer is deliberately NOT
audited here (no frozen causal TRAIN-OOF E9 root axis exists), so the
downstream audit is independent of the production Direction layer.

Reviewed base: d7c228e5b2c0bedb0c4890a75335d97c4937ed1a
Phase-4 remains frozen: NO_V2_MODEL_IMPROVEMENT
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas import run_decomposed_v2_research as R

# --------------------------------------------------------------------------- #
# Paths / constants                                                            #
# --------------------------------------------------------------------------- #
V2_OOF_DIR = R.V2_OOF_DIR
LABELS_TRAIN = R.ALLOWED_V1_LABELS_TRAIN
STATE_V1 = R.ALLOWED_V1_STATE
EVIDENCE_DIR = R.EVIDENCE_DIR
PHASE4_MANIFEST_JSON = R.PHASE4_MANIFEST_JSON

ARTIFACT_DIR = os.path.dirname(V2_OOF_DIR)
CLOSURE_LEDGER = os.path.join(ARTIFACT_DIR, "closure_audit_v1.parquet")

SUMMARY_JSON = os.path.join(EVIDENCE_DIR, "decomposed_value_closure_audit_v1_summary.json")
DECILES_CSV = os.path.join(EVIDENCE_DIR, "decomposed_value_closure_audit_v1_deciles.csv")
GATE_CSV = os.path.join(EVIDENCE_DIR, "decomposed_value_closure_audit_v1_gate_attribution.csv")
FORENSICS_CSV = os.path.join(EVIDENCE_DIR, "decomposed_value_closure_audit_v1_forensics.csv")
MANIFEST_JSON = os.path.join(EVIDENCE_DIR, "decomposed_value_closure_audit_v1_manifest.json")

HORIZON = "td5"
N_FOLDS = 5

BOOTSTRAP_B = 5000
BOOTSTRAP_SEED = 20260925
BOOTSTRAP_BLOCK = 5  # trading-day block length for the block bootstrap

REVIEWED_BASE_SHA = "d7c228e5b2c0bedb0c4890a75335d97c4937ed1a"
PHASE4_STATUS = "NO_V2_MODEL_IMPROVEMENT"

ARCH_PRIMARY = "A0_V1_DISJOINT"
ARCH_SECONDARY = "A1_SHARE_TO_WIN"

POP_ALL = "ALL_TD5_OOF"
POP_ROOT = "ROOT_CANDIDATE_TD5"

LEDGER_COLUMNS = [
    "architecture", "population",
    "symbol", "decision_bar", "decision_time", "trading_day", "side", "fold",
    "candidate_at_decision",
    "actual_return", "actual_win", "actual_abs_return",
    "p_win", "mu_win", "mu_loss", "rr_hat",
    "p_break_even", "ev_margin", "ev_pred",
    "oracle_sign_return",
    "probability_error_component", "magnitude_error_component", "total_ev_error",
    "error_class", "sample_weight",
]


# --------------------------------------------------------------------------- #
# Small helpers                                                                #
# --------------------------------------------------------------------------- #
def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def wmean(x, w):
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not m.any():
        return np.nan
    return float(np.average(x[m], weights=w[m]))


def wrmse(err, w):
    return float(np.sqrt(wmean(np.asarray(err, float) ** 2, w)))


def wcorr(x, y, w):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    w = np.asarray(w, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 3:
        return np.nan
    x = x[m]
    y = y[m]
    w = w[m]
    w = w / w.sum()
    xm = x - np.sum(w * x)
    ym = y - np.sum(w * y)
    cov = np.sum(w * xm * ym)
    sx = np.sqrt(np.sum(w * xm ** 2))
    sy = np.sqrt(np.sum(w * ym ** 2))
    if sx <= 0 or sy <= 0:
        return np.nan
    return float(cov / (sx * sy))


# --------------------------------------------------------------------------- #
# Loading                                                                      #
# --------------------------------------------------------------------------- #
def load_td5_oof(arch_name: str) -> pd.DataFrame:
    """Load the 5 TRAIN-OOF TD5 shards for one architecture (plan §33)."""
    parts = []
    for fold in range(N_FOLDS):
        shard, _meta = R._unit_paths(arch_name, fold, HORIZON)
        if not os.path.exists(shard):
            raise RuntimeError(f"STOP_R13_5_MISSING_OOF_SHARD {shard}")
        x = R.read_parquet(shard)
        if set(x["horizon"]) != {HORIZON}:
            raise RuntimeError("STOP_R13_5_HORIZON_DRIFT")
        if set(x["fold"]) != {fold}:
            raise RuntimeError("STOP_R13_5_FOLD_DRIFT")
        parts.append(x)

    out = pd.concat(parts, ignore_index=True)

    key = ["symbol", "decision_bar", "side", "horizon"]
    if out.duplicated(key).any():
        raise RuntimeError("STOP_R13_5_DUPLICATE_OOF_ROW")

    if set(out["fold"]) != set(range(N_FOLDS)):
        raise RuntimeError("STOP_R13_5_FOLD_DRIFT")

    return out


def verify_oof_labels(oof: pd.DataFrame) -> pd.DataFrame:
    """Independently verify OOF Y / sample_weight against TRAIN labels (plan §34)."""
    labels = R.read_train_labels(columns=[
        "symbol", "decision_bar", "side", "horizon", "decision_time",
        "episode_return_atr", "win", "win_magnitude", "loss_magnitude",
        "sample_weight", "bracket_eligible", "entry_executable",
    ])
    labels = labels[labels["horizon"] == HORIZON].copy()

    key = ["symbol", "decision_bar", "side", "horizon"]
    m = oof.merge(
        labels, on=key, how="left", validate="one_to_one",
        suffixes=("_oof", "_label"),
    )
    if m["episode_return_atr_label"].isna().any():
        raise RuntimeError("STOP_R13_5_LABEL_JOIN_MISSING")

    np.testing.assert_allclose(
        m["episode_return_atr_oof"].to_numpy(float),
        m["episode_return_atr_label"].to_numpy(float),
        rtol=0, atol=1e-12,
    )
    np.testing.assert_allclose(
        m["sample_weight_oof"].to_numpy(float),
        m["sample_weight_label"].to_numpy(float),
        rtol=0, atol=1e-12,
    )
    return m


def join_state(df: pd.DataFrame) -> pd.DataFrame:
    """Join candidate_at_decision + trading_day from state_v1 (plan §27, §18)."""
    st = R.read_state(columns=["symbol", "bar_index", "trading_day",
                               "candidate_at_decision"])
    st = st.rename(columns={"bar_index": "decision_bar"})
    merged = df.merge(
        st, on=["symbol", "decision_bar"], how="left", validate="many_to_one",
    )
    if merged["candidate_at_decision"].isna().any():
        raise RuntimeError("STOP_R13_5_STATE_JOIN_MISSING")
    return merged


# --------------------------------------------------------------------------- #
# Per-row exact decomposition (plan §5-§11)                                    #
# --------------------------------------------------------------------------- #
def add_closure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    y = out["episode_return_atr"].to_numpy(float)
    z = (y > 0).astype(float)
    a = np.abs(y)

    p = np.clip(out["p_win"].to_numpy(float), 0.0, 1.0)
    mw = np.maximum(out["mu_win"].to_numpy(float), 0.0)
    ml = np.maximum(out["mu_loss"].to_numpy(float), 0.0)

    ev = p * mw - (1.0 - p) * ml

    oracle_sign_return = z * mw - (1.0 - z) * ml

    probability_error = (p - z) * (mw + ml)
    magnitude_error = oracle_sign_return - y
    total_error = ev - y

    # Hard identity (plan §9)
    np.testing.assert_allclose(
        total_error, probability_error + magnitude_error,
        rtol=1e-10, atol=1e-10,
    )

    denom = mw + ml
    p_break_even = np.where(denom > 0, ml / denom, np.nan)
    ev_margin = p - p_break_even

    out["actual_win"] = z.astype(np.uint8)
    out["actual_abs_return"] = a
    out["ev_recomputed"] = ev
    out["oracle_sign_return"] = oracle_sign_return
    out["probability_error_component"] = probability_error
    out["magnitude_error_component"] = magnitude_error
    out["total_ev_error"] = total_error
    out["probability_sq_error"] = probability_error ** 2
    out["magnitude_sq_error"] = magnitude_error ** 2
    out["error_cross_term"] = 2.0 * probability_error * magnitude_error
    out["p_break_even"] = p_break_even
    out["ev_margin"] = ev_margin
    out["ev_positive"] = ev > 0

    out["error_class"] = np.where(
        ev > 0,
        np.where(y > 0, "SELECTED_WINNER", "SELECTED_LOSER"),
        np.where(y > 0, "SKIPPED_WINNER", "SKIPPED_LOSER"),
    )
    return out


# --------------------------------------------------------------------------- #
# Summary blocks                                                               #
# --------------------------------------------------------------------------- #
def closure_summary(df: pd.DataFrame) -> Dict[str, float]:
    w = df["sample_weight"].to_numpy(float)
    ep = df["probability_error_component"].to_numpy(float)
    em = df["magnitude_error_component"].to_numpy(float)
    et = df["total_ev_error"].to_numpy(float)

    s = {
        "total_bias": wmean(et, w),
        "probability_bias": wmean(ep, w),
        "magnitude_bias": wmean(em, w),
        "total_rmse": wrmse(et, w),
        "probability_rmse": wrmse(ep, w),
        "magnitude_rmse": wrmse(em, w),
        "total_mse": wmean(et ** 2, w),
        "probability_mse_component": wmean(ep ** 2, w),
        "magnitude_mse_component": wmean(em ** 2, w),
        "cross_term": wmean(2.0 * ep * em, w),
    }
    np.testing.assert_allclose(
        s["total_mse"],
        s["probability_mse_component"] + s["magnitude_mse_component"] + s["cross_term"],
        rtol=1e-10, atol=1e-10,
    )
    return s


def _brier_logloss(p, z, w):
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    z = np.asarray(z, float)
    w = np.asarray(w, float)
    m = np.isfinite(p) & np.isfinite(z) & (w > 0)
    p, z, w = p[m], z[m], w[m]
    brier = float(np.average((p - z) ** 2, weights=w))
    logloss = float(np.average(
        -(z * np.log(p) + (1.0 - z) * np.log(1.0 - p)), weights=w))
    return brier, logloss


def win_metrics(df: pd.DataFrame) -> Dict[str, float]:
    y = df["episode_return_atr"].to_numpy(float)
    z = (y > 0).astype(float)
    w = df["sample_weight"].to_numpy(float)
    p = df["p_win"].to_numpy(float)

    ord_brier, ord_logloss = _brier_logloss(p, z, w)
    money_w = w * np.abs(y)
    mon_brier, mon_logloss = _brier_logloss(p, z, money_w)

    # top20 / bottom20 actual return by p_win (within resampled df these are
    # recomputed; here it is the full-population point estimate used for the
    # summary table).
    pp = p[np.isfinite(p) & np.isfinite(y) & (w > 0)]
    yy = y[np.isfinite(p) & np.isfinite(y) & (w > 0)]
    ww = w[np.isfinite(p) & np.isfinite(y) & (w > 0)]
    hi = np.quantile(pp, 0.8)
    lo = np.quantile(pp, 0.2)
    top = pp >= hi
    bot = pp <= lo
    p_top = wmean(yy[top], ww[top]) if top.any() else np.nan
    p_bot = wmean(yy[bot], ww[bot]) if bot.any() else np.nan

    return {
        "ordinary_brier": ord_brier,
        "ordinary_logloss": ord_logloss,
        "money_weighted_brier": mon_brier,
        "money_weighted_logloss": mon_logloss,
        "p_top20_actual_return": p_top,
        "p_bottom20_actual_return": p_bot,
        "p_top20_minus_bottom20": (p_top - p_bot) if (top.any() and bot.any()) else np.nan,
    }


def money_weighted_probability_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """plan §38 (money-weighted Brier/logloss + oracle-realized-magnitude)."""
    y = df["episode_return_atr"].to_numpy(float)
    z = (y > 0).astype(float)
    p = np.clip(df["p_win"].to_numpy(float), 1e-12, 1.0 - 1e-12)
    base_w = df["sample_weight"].to_numpy(float)
    money_w = base_w * np.abs(y)

    brier = wmean((p - z) ** 2, money_w)
    logloss = wmean(
        -(z * np.log(p) + (1.0 - z) * np.log(1.0 - p)), money_w)

    oracle_mag_score = (2.0 * p - 1.0) * np.abs(y)
    return {
        "money_weighted_brier": brier,
        "money_weighted_logloss": logloss,
        "oracle_realized_magnitude_mae": wmean(np.abs(oracle_mag_score - y), base_w),
        "oracle_realized_magnitude_rmse": wrmse(oracle_mag_score - y, base_w),
    }


def payoff_metrics(df: pd.DataFrame) -> Dict[str, float]:
    """Win/loser magnitude audit + oracle-sign reconstruction (plan §12-§15)."""
    w = df["sample_weight"].to_numpy(float)
    y = df["episode_return_atr"].to_numpy(float)
    z = (y > 0).astype(float)
    mw = np.maximum(df["mu_win"].to_numpy(float), 0.0)
    ml = np.maximum(df["mu_loss"].to_numpy(float), 0.0)

    win_m = z > 0
    loss_m = ~win_m

    # winner magnitude
    aw = y[win_m]
    ww = mw[win_m]
    ww_w = w[win_m]
    winner_mae = wmean(np.abs(ww - aw), ww_w)
    winner_rmse = wrmse(ww - aw, ww_w)
    winner_bias = wmean(ww - aw, ww_w)
    winner_corr = wcorr(ww, aw, ww_w)

    # loser magnitude (actual losing magnitude = -y for losers)
    al = -y[loss_m]
    ll = ml[loss_m]
    ll_w = w[loss_m]
    loser_mae = wmean(np.abs(ll - al), ll_w)
    loser_rmse = wrmse(ll - al, ll_w)
    loser_bias = wmean(ll - al, ll_w)
    loser_corr = wcorr(ll, al, ll_w)

    # oracle-sign reconstruction
    oracle_sign = z * mw - (1.0 - z) * ml
    os_mae = wmean(np.abs(oracle_sign - y), w)
    os_rmse = wrmse(oracle_sign - y, w)
    os_bias = wmean(oracle_sign - y, w)
    os_corr = wcorr(oracle_sign, y, w)

    # winner magnitude top20 - bottom20 spread (plan §13)
    if win_m.any():
        pw = ww[np.isfinite(ww) & np.isfinite(aw)]
        aw2 = aw[np.isfinite(ww) & np.isfinite(aw)]
        ww2 = ww_w[np.isfinite(ww) & np.isfinite(aw)]
        hi = np.quantile(pw, 0.8)
        lo = np.quantile(pw, 0.2)
        top = pw >= hi
        bot = pw <= lo
        winner_spread = (wmean(aw2[top], ww2[top]) - wmean(aw2[bot], ww2[bot])
                         if top.any() and bot.any() else np.nan)
    else:
        winner_spread = np.nan

    # loser magnitude top20 - bottom20 spread (plan §14)
    if loss_m.any():
        pl = ll[np.isfinite(ll) & np.isfinite(al)]
        al2 = al[np.isfinite(ll) & np.isfinite(al)]
        ll2 = ll_w[np.isfinite(ll) & np.isfinite(al)]
        hi = np.quantile(pl, 0.8)
        lo = np.quantile(pl, 0.2)
        top = pl >= hi
        bot = pl <= lo
        loser_spread = (wmean(al2[top], ll2[top]) - wmean(al2[bot], ll2[bot])
                        if top.any() and bot.any() else np.nan)
    else:
        loser_spread = np.nan

    return {
        "winner_mu_win_mae": winner_mae,
        "winner_mu_win_rmse": winner_rmse,
        "winner_mu_win_bias": winner_bias,
        "winner_mu_win_corr": winner_corr,
        "loser_mu_loss_mae": loser_mae,
        "loser_mu_loss_rmse": loser_rmse,
        "loser_mu_loss_bias": loser_bias,
        "loser_mu_loss_corr": loser_corr,
        "winner_magnitude_top20_minus_bottom20": winner_spread,
        "loser_magnitude_top20_minus_bottom20": loser_spread,
        "oracle_sign_reconstruction_mae": os_mae,
        "oracle_sign_reconstruction_rmse": os_rmse,
        "oracle_sign_reconstruction_bias": os_bias,
        "oracle_sign_reconstruction_corr": os_corr,
    }


def gate_attribution(df: pd.DataFrame) -> Dict[str, float]:
    """plan §21 (Selected/Skipped profit & loss attribution)."""
    y = df["episode_return_atr"].to_numpy(float)
    w = df["sample_weight"].to_numpy(float)
    gate = df["ev_recomputed"].to_numpy(float) > 0
    win = y > 0

    captured_profit = float(np.sum(w[gate & win] * y[gate & win]))
    admitted_loss = float(-np.sum(w[gate & ~win] * y[gate & ~win]))
    missed_profit = float(np.sum(w[~gate & win] * y[~gate & win]))
    avoided_loss = float(-np.sum(w[~gate & ~win] * y[~gate & ~win]))

    selected_weight = float(w[gate].sum())
    selected_mean = (float(np.sum(w[gate] * y[gate]) / selected_weight)
                     if selected_weight > 0 else np.nan)

    return {
        "ev_positive_fraction": float(gate.mean()),
        "captured_profit": captured_profit,
        "admitted_loss": admitted_loss,
        "selected_net": captured_profit - admitted_loss,
        "missed_profit": missed_profit,
        "avoided_loss": avoided_loss,
        "selected_weight": selected_weight,
        "selected_actual_mean_return": selected_mean,
    }


# --------------------------------------------------------------------------- #
# Decile tables                                                                #
# --------------------------------------------------------------------------- #
def _decile_table(df: pd.DataFrame, dec_col: str, y_col: str,
                  weight_col: str = "sample_weight",
                  filter_expr: Optional[str] = None) -> pd.DataFrame:
    d = df.copy()
    if filter_expr is not None:
        d = d.query(filter_expr)
    d = d[np.isfinite(d[dec_col]) & np.isfinite(d[y_col])].copy()
    if len(d) == 0:
        return pd.DataFrame()
    q = np.quantile(d[dec_col], np.linspace(0, 1, 11))
    q = np.unique(q)
    if len(q) < 2:
        return pd.DataFrame()
    d["_dec"] = pd.qcut(d[dec_col], q=10, labels=False, duplicates="drop")
    rows = []
    for k in sorted(d["_dec"].unique()):
        sub = d[d["_dec"] == k]
        w = sub[weight_col].to_numpy(float)
        yv = sub[y_col].to_numpy(float)
        win = yv > 0
        avg_win = wmean(yv[win], w[win]) if win.any() else np.nan
        avg_loss = wmean(-yv[~win], w[~win]) if (~win).any() else np.nan
        pr = wmean(yv[win].size and yv[win], w[win]) if False else None
        rows.append({
            "decile": int(k),
            "n": int(len(sub)),
            "weight_sum": float(w.sum()),
            "mean_predicted": float(wmean(sub[dec_col].to_numpy(float), w)),
            "mean_actual": float(wmean(yv, w)),
            "actual_win_rate": float(wmean(win.astype(float), w)),
            "avg_win": float(avg_win) if avg_win is not None else np.nan,
            "avg_loss": float(avg_loss) if avg_loss is not None else np.nan,
            "payoff_ratio": (float(avg_win / avg_loss)
                             if (avg_win is not None and avg_loss and
                                 not np.isnan(avg_win) and not np.isnan(avg_loss)
                                 and avg_loss > 0) else np.nan),
        })
    return pd.DataFrame(rows)


def build_decile_tables(df: pd.DataFrame, arch: str, population: str) -> pd.DataFrame:
    y = df["episode_return_atr"].to_numpy(float)
    df = df.copy()
    df["_win_mag"] = np.where(y > 0, y, np.nan)
    df["_loss_mag"] = np.where(y <= 0, -y, np.nan)

    parts = []
    for name, dec_col, y_col, filt in [
        ("p_win", "p_win", "episode_return_atr", None),
        ("ev", "ev_recomputed", "episode_return_atr", None),
        ("mu_win", "mu_win", "_win_mag", "episode_return_atr > 0"),
        ("mu_loss", "mu_loss", "_loss_mag", "episode_return_atr <= 0"),
    ]:
        t = _decile_table(df, dec_col, y_col, filter_expr=filt)
        if t.empty:
            continue
        t.insert(0, "table", name)
        t.insert(0, "population", population)
        t.insert(0, "architecture", arch)
        parts.append(t)
    if not parts:
        return pd.DataFrame(columns=["architecture", "population", "table"])
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- #
# Block bootstrap (plan §32)                                                   #
# --------------------------------------------------------------------------- #
def _block_index_map(trading_days: np.ndarray, block: int):
    """Complete block bootstrap: only keep whole `block`-day blocks and drop
    the trailing remainder (plan R13.5 fix: n_complete = floor(n_days/5))."""
    days = np.sort(pd.unique(trading_days))
    n_complete = len(days) // block
    days = days[: n_complete * block]
    pos = {d: np.where(trading_days == d)[0] for d in days}
    blocks = [days[i * block:(i + 1) * block] for i in range(n_complete)]
    block_idx = [np.concatenate([pos[d] for d in b]) for b in blocks]
    return block_idx


def bootstrap_contrasts(df_p1: pd.DataFrame, seed: int = BOOTSTRAP_SEED,
                        n_boot: int = BOOTSTRAP_B, block: int = BOOTSTRAP_BLOCK) -> Dict[str, Dict[str, float]]:
    arr = {
        "y": df_p1["episode_return_atr"].to_numpy(float),
        "w": df_p1["sample_weight"].to_numpy(float),
        "p": df_p1["p_win"].to_numpy(float),
        "muw": np.maximum(df_p1["mu_win"].to_numpy(float), 0.0),
        "mul": np.maximum(df_p1["mu_loss"].to_numpy(float), 0.0),
        "ev": df_p1["ev_recomputed"].to_numpy(float),
        "abs": np.abs(df_p1["episode_return_atr"].to_numpy(float)),
    }
    block_idx = _block_index_map(df_p1["trading_day"].to_numpy(), block)
    n_blocks = len(block_idx)
    rng = np.random.default_rng(seed)

    def stat_p(sel):
        p = arr["p"][sel]; y = arr["y"][sel]; w = arr["w"][sel]
        m = np.isfinite(p) & np.isfinite(y) & (w > 0)
        p, y, w = p[m], y[m], w[m]
        if len(p) < 20:
            return np.nan
        hi, lo = np.quantile(p, 0.8), np.quantile(p, 0.2)
        top, bot = p >= hi, p <= lo
        if not (top.any() and bot.any()):
            return np.nan
        return wmean(y[top], w[top]) - wmean(y[bot], w[bot])

    def stat_muw(sel):
        y = arr["y"][sel]; w = arr["w"][sel]; muw = arr["muw"][sel]
        m = (y > 0) & np.isfinite(muw) & np.isfinite(y) & (w > 0)
        muw, y, w = muw[m], y[m], w[m]
        if len(muw) < 20:
            return np.nan
        hi, lo = np.quantile(muw, 0.8), np.quantile(muw, 0.2)
        top, bot = muw >= hi, muw <= lo
        if not (top.any() and bot.any()):
            return np.nan
        return wmean(y[top], w[top]) - wmean(y[bot], w[bot])

    def stat_mul(sel):
        y = arr["y"][sel]; w = arr["w"][sel]; mul = arr["mul"][sel]
        m = (y <= 0) & np.isfinite(mul) & np.isfinite(-y) & (w > 0)
        mul, y, w = mul[m], (-y[m]), w[m]
        if len(mul) < 20:
            return np.nan
        hi, lo = np.quantile(mul, 0.8), np.quantile(mul, 0.2)
        top, bot = mul >= hi, mul <= lo
        if not (top.any() and bot.any()):
            return np.nan
        return wmean(y[top], w[top]) - wmean(y[bot], w[bot])

    def stat_ev(sel):
        ev = arr["ev"][sel]; y = arr["y"][sel]; w = arr["w"][sel]
        m = np.isfinite(ev) & np.isfinite(y) & (w > 0)
        ev, y, w = ev[m], y[m], w[m]
        if len(ev) < 20:
            return np.nan
        hi, lo = np.quantile(ev, 0.8), np.quantile(ev, 0.2)
        top, bot = ev >= hi, ev <= lo
        if not (top.any() and bot.any()):
            return np.nan
        return wmean(y[top], w[top]) - wmean(y[bot], w[bot])

    def stat_evpos(sel):
        ev = arr["ev"][sel]; y = arr["y"][sel]; w = arr["w"][sel]
        m = np.isfinite(ev) & np.isfinite(y) & (w > 0)
        ev, y, w = ev[m], y[m], w[m]
        g = ev > 0
        if not g.any():
            return np.nan
        return wmean(y[g], w[g])

    stats = {
        "p_top20_minus_bottom20": stat_p,
        "mu_win_top20_minus_bottom20": stat_muw,
        "mu_loss_top20_minus_bottom20": stat_mul,
        "ev_top20_minus_bottom20": stat_ev,
        "ev_positive_weighted_return": stat_evpos,
    }

    out: Dict[str, Dict[str, float]] = {}
    for name, fn in stats.items():
        boots = np.empty(n_boot)
        for b in range(n_boot):
            chosen = rng.integers(0, n_blocks, size=n_blocks)
            sel = np.concatenate([block_idx[c] for c in chosen])
            boots[b] = fn(sel)
        boots = boots[np.isfinite(boots)]
        out[name] = {
            "point": float(np.nanmean(boots)) if boots.size else np.nan,
            "ci_lo": float(np.percentile(boots, 2.5)) if boots.size else np.nan,
            "ci_hi": float(np.percentile(boots, 97.5)) if boots.size else np.nan,
        }
    return out


# --------------------------------------------------------------------------- #
# Forensics (plan §23-§24)                                                     #
# --------------------------------------------------------------------------- #
def build_forensics(df: pd.DataFrame, arch: str, population: str,
                    n_top: int = 20) -> pd.DataFrame:
    keep = ["SELECTED_LOSER", "SKIPPED_WINNER"]
    sub = df[df["error_class"].isin(keep)].copy()
    cols = [
        "symbol", "decision_time", "decision_bar", "side", "fold",
        "actual_return", "actual_abs_return",
        "p_win", "mu_win", "mu_loss", "predicted_rr",
        "p_break_even", "ev_margin", "ev_c",
        "probability_error_component", "magnitude_error_component",
        "total_ev_error", "error_class",
    ]
    sub = sub[cols]
    sub.insert(0, "population", population)
    sub.insert(0, "architecture", arch)

    sl = sub[sub["error_class"] == "SELECTED_LOSER"].copy()
    sw = sub[sub["error_class"] == "SKIPPED_WINNER"].copy()
    sl = sl.sort_values("actual_return").head(n_top)
    sw = sw.sort_values("actual_return", ascending=False).head(n_top)
    return pd.concat([sl, sw], ignore_index=True)


# --------------------------------------------------------------------------- #
# Population assembly + summary                                                #
# --------------------------------------------------------------------------- #
def _prepare_population(oof: pd.DataFrame, population: str) -> pd.DataFrame:
    df = add_closure_columns(oof)
    df["actual_return"] = df["episode_return_atr"]
    if population == POP_ROOT:
        if "candidate_at_decision" not in df.columns:
            raise RuntimeError("STOP_R13_5_NO_CANDIDATE_FLAG")
        df = df[df["candidate_at_decision"] == True].copy()  # noqa: E712
    return df


def summarize_population(df: pd.DataFrame, arch: str, population: str,
                         bootstrap: bool) -> Dict[str, Any]:
    res: Dict[str, Any] = {}
    res["win"] = win_metrics(df)
    res["payoff"] = payoff_metrics(df)
    res["money_weighted_prob"] = money_weighted_probability_metrics(df)
    res["closure"] = closure_summary(df)
    res["gate"] = gate_attribution(df)
    if bootstrap:
        res["bootstrap_ci"] = bootstrap_contrasts(df)
    res["n_rows"] = int(len(df))
    return res


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def build_audit() -> Dict[str, Any]:
    """Run the full R13.5 closure audit (no model fit, no VAL/TEST)."""
    # ---- loads (guarded) ----
    a0 = load_td5_oof(ARCH_PRIMARY)
    a1 = load_td5_oof(ARCH_SECONDARY)

    # 1. A0/A1 key universes identical
    key = ["symbol", "decision_bar", "side", "horizon"]
    set_a0 = set(map(tuple, a0[key].to_numpy()))
    set_a1 = set(map(tuple, a1[key].to_numpy()))
    if set_a0 != set_a1:
        raise RuntimeError("STOP_R13_5_ARCH_KEY_UNIVERSE_MISMATCH")

    # 2. label verification (exact)
    verify_oof_labels(a0)
    verify_oof_labels(a1)

    # 3. state join (candidate flag + trading_day)
    a0 = join_state(a0)
    a1 = join_state(a1)

    audit: Dict[str, Any] = {"populations": {}, "a0_vs_a1": {}}
    ledger_parts = []
    decile_parts = []
    forensic_parts = []
    gate_rows = []

    for arch, oof in [(ARCH_PRIMARY, a0), (ARCH_SECONDARY, a1)]:
        for population in [POP_ALL, POP_ROOT]:
            df = _prepare_population(oof, population)
            do_boot = (population == POP_ROOT)
            res = summarize_population(df, arch, population, bootstrap=do_boot)
            audit["populations"].setdefault(arch, {})[population] = res

            # ledger row
            led = df.copy()
            led["architecture"] = arch
            led["population"] = population
            led["rr_hat"] = led["predicted_rr"]
            led["ev_pred"] = led["ev_recomputed"]
            led["actual_return"] = led["episode_return_atr"]
            led["actual_win"] = led["actual_win"]
            led["actual_abs_return"] = led["actual_abs_return"]
            keep = [c for c in LEDGER_COLUMNS if c in led.columns]
            ledger_parts.append(led[keep])

            # deciles
            decile_parts.append(build_decile_tables(df, arch, population))

            # gate row
            g = res["gate"]
            gate_rows.append({
                "architecture": arch, "population": population,
                **g,
            })

            # forensics
            forensic_parts.append(build_forensics(df, arch, population))

    # ---- A0 vs A1 payoff equality + metric deltas ----
    a0k = a0.set_index(key)
    a1k = a1.set_index(key)
    muw_diff = (a0k["mu_win"] - a1k["mu_win"]).abs().max()
    mul_diff = (a0k["mu_loss"] - a1k["mu_loss"]).abs().max()
    audit["a0_vs_a1_payoff_diff"] = {
        "max_abs_mu_win_diff": float(muw_diff),
        "max_abs_mu_loss_diff": float(mul_diff),
    }

    for population in [POP_ALL, POP_ROOT]:
        pa0 = audit["populations"][ARCH_PRIMARY][population]
        pa1 = audit["populations"][ARCH_SECONDARY][population]
        audit["a0_vs_a1"][population] = {
            "delta_probability_mse": pa1["closure"]["probability_mse_component"]
            - pa0["closure"]["probability_mse_component"],
            "delta_magnitude_mse": pa1["closure"]["magnitude_mse_component"]
            - pa0["closure"]["magnitude_mse_component"],
            "delta_cross_term": pa1["closure"]["cross_term"]
            - pa0["closure"]["cross_term"],
            "delta_total_ev_mse": pa1["closure"]["total_mse"]
            - pa0["closure"]["total_mse"],
            "delta_admitted_loss": pa1["gate"]["admitted_loss"]
            - pa0["gate"]["admitted_loss"],
            "delta_missed_profit": pa1["gate"]["missed_profit"]
            - pa0["gate"]["missed_profit"],
            "delta_selected_actual_mean_return":
                pa1["gate"]["selected_actual_mean_return"]
                - pa0["gate"]["selected_actual_mean_return"],
        }

    audit["closure_identity_validated"] = True
    return audit, ledger_parts, decile_parts, forensic_parts, gate_rows, a0, a1


def write_artifacts(audit, ledger_parts, decile_parts, forensic_parts,
                    gate_rows) -> Dict[str, str]:
    ledger = pd.concat(ledger_parts, ignore_index=True)
    R.write_parquet(ledger, CLOSURE_LEDGER)

    deciles = pd.concat(decile_parts, ignore_index=True)
    deciles.to_csv(DECILES_CSV, index=False)

    gate_df = pd.DataFrame(gate_rows)
    gate_df.to_csv(GATE_CSV, index=False)

    forensics = pd.concat(forensic_parts, ignore_index=True)
    forensics.to_csv(FORENSICS_CSV, index=False)

    return {
        "ledger": sha256_of(CLOSURE_LEDGER),
        "deciles": sha256_of(DECILES_CSV),
        "gate": sha256_of(GATE_CSV),
        "forensics": sha256_of(FORENSICS_CSV),
    }


def build_manifest(audit, artifact_shas, generator_code_sha: str) -> dict:
    # input artifact SHAs
    a0_shards = [sha256_of(R._unit_paths(ARCH_PRIMARY, f, HORIZON)[0])
                 for f in range(N_FOLDS)]
    a1_shards = [sha256_of(R._unit_paths(ARCH_SECONDARY, f, HORIZON)[0])
                 for f in range(N_FOLDS)]

    manifest = {
        "task_id": "FUTURE-R13.5-TD5-DECOMPOSED-CLOSURE-AUDIT-V1",
        "reviewed_base_sha": REVIEWED_BASE_SHA,
        "phase4_status": PHASE4_STATUS,
        "phase4_manifest_sha": sha256_of(PHASE4_MANIFEST_JSON),
        "generator_code_sha": generator_code_sha,
        "bootstrap": {
            "method": "trading_day_block_bootstrap",
            "block_trading_days": BOOTSTRAP_BLOCK,
            "n_boot": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
        },
        "input_artifact_shas": {
            "a0_td5_oof_shards": a0_shards,
            "a1_td5_oof_shards": a1_shards,
            "labels_train_v1": sha256_of(LABELS_TRAIN),
            "state_v1": sha256_of(STATE_V1),
            "phase4_manifest": sha256_of(PHASE4_MANIFEST_JSON),
        },
        "output_artifact_shas": artifact_shas,
        "summary_sha": sha256_of(SUMMARY_JSON) if os.path.exists(SUMMARY_JSON) else None,
        "a0_vs_a1_payoff_diff": audit["a0_vs_a1_payoff_diff"],
        "closure_identity_validated": audit["closure_identity_validated"],
        "direction_layer": {
            "production_root_direction": "E9",
            "audited_in_r13_5": False,
            "reason": "NO_FROZEN_CAUSAL_TRAIN_OOF_E9_ROOT_AXIS",
        },
        "governance": {
            "model_fits": 0,
            "hyperparameter_searches": 0,
            "feature_changes": 0,
            "direction_model_fits": 0,
            "dev_val_reads": 0,
            "old_test_label_reads": 0,
            "old_test_policy_reads": 0,
            "E9_train_oof_axis_reads": 0,
        },
    }
    return manifest


def main() -> None:
    audit, ledger_parts, decile_parts, forensic_parts, gate_rows, a0, a1 = build_audit()
    artifact_shas = write_artifacts(audit, ledger_parts, decile_parts,
                                    forensic_parts, gate_rows)
    R.write_json_evidence(audit, SUMMARY_JSON)

    generator_code_sha = sha256_of(os.path.abspath(__file__))
    manifest = build_manifest(audit, artifact_shas, generator_code_sha)
    R.write_json_evidence(manifest, MANIFEST_JSON)

    print("R13.5 closure audit complete.")
    print("  ledger rows:", len(pd.read_parquet(CLOSURE_LEDGER)))
    print("  A0 ALL closure total_mse:",
          round(audit["populations"][ARCH_PRIMARY][POP_ALL]["closure"]["total_mse"], 6))
    print("  A0 ROOT EV>0 selected mean return:",
          round(audit["populations"][ARCH_PRIMARY][POP_ROOT]["gate"]["selected_actual_mean_return"], 6))
    print("  A0 vs A1 max|mu_win| diff:",
          round(audit["a0_vs_a1_payoff_diff"]["max_abs_mu_win_diff"], 8))


if __name__ == "__main__":
    main()
