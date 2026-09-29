"""PAYOFF-TEMPORAL-01A -- Causal Geometry Trajectory Incremental Information Audit.

Question
--------
Within the frozen canonical TD5 OOF population, does the strictly PAST-ONLY 15m
geometry trajectory add incremental predictive information for conditional
payoff magnitude, beyond the current static PAY8 snapshot?

    M0 = PAY8                 (8 features)
    M1 = PAY8 + TEMP57        (65 features)

The model class is NOT changed: the existing two-head conditional LightGBM
payoff architecture is reused (win head on Y>0, loss head on Y<=0). Only the
feature information differs.

Causality contract
------------------
Every temporal feature is a function of (S_t, S_{t-1}, S_{t-2}, ...) ONLY.
No field from t+1 or later may enter. History may never cross a canonical
causal continuity boundary (symbol / side / fold / trading day / segment /
non-contiguous decision bar / non-15m time gap). No forward fill, no backfill.

TRADING_METRICS: NOT_APPLICABLE
reason: No trading action has been defined. This module measures conditional
payoff-magnitude prediction error only; it produces no policy, no trade ledger
and therefore no win-rate / payoff-ratio / expectancy claim.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_decomposed_value_dataset_v1 import (
    PAY8_COLS,
    build_pay8_from_state,
)
from research.liquidity_oracle_atlas.walkforward_development_v1 import (
    N_OUTER,
    build_fold_plan,
    is_warmup_row,
    iter_fold_splits,
    oof_row_count_check,
)
from research.liquidity_oracle_atlas.payoff_ratio_model_v1 import (
    EARLY_STOPPING_ROUNDS,
    REG_PARAMS,
    fit_bundle,
)
import research.liquidity_oracle_atlas.payoff_ratio_model_v1 as _payoff_v1

TASK_ID = "PAYOFF-TEMPORAL-01A"

# --------------------------------------------------------------------------- #
# Frozen geometry roles inside canonical PAY8                                  #
# --------------------------------------------------------------------------- #
G_COL = "reward_distance_atr"
L_COL = "risk_distance_atr"
Z_COL = "log_structural_rr"

LAGS = (1, 2, 3, 4)          # LAG32
DELTA_LAGS = (1, 2, 4)       # PATH9
SLOPE_WINDOWS = (4, 16)      # SLOPE6
EXTREMA_WINDOWS = (4, 16)    # EXTREMA4 (Z only)
RUN_CAP = 16                 # PERSIST3 cap
SWITCH_WINDOW = 16           # PERSIST3 trailing switch window

# Frozen bootstrap contract (reviewer-frozen for this experiment).
BOOTSTRAP_BLOCK_DAYS = 5
BOOTSTRAP_SEED = 20260929
B_T1_5 = 500
B_T2 = 2000

PRIMARY_HORIZON = "td5"

# --------------------------------------------------------------------------- #
# Canonical artifact paths                                                     #
# --------------------------------------------------------------------------- #
STATE_PARQUET = Path("artifacts/decomposed_value_v1/state_v1.parquet")
LABEL_DIR = Path("artifacts/opportunity_value_v1")
LABEL_TRAIN_PARQUET = LABEL_DIR / "labels_train_v1.parquet"
OOF_DIR = Path("artifacts/decomposed_value_v2/oof")

# Canonical continuity keys. `segment` is the authoritative state segment key;
# the repo exposes no session/roll column, so the conservative observable rules
# below are used (never forward-fill, never cross a boundary).
CONTINUITY_KEYS = ("symbol", "side", "fold", "trading_day", "segment")

# --------------------------------------------------------------------------- #
# Performance / governance counters                                            #
# --------------------------------------------------------------------------- #
COUNTERS: dict[str, int] = {
    "state_load_count": 0,
    "pay8_load_count": 0,
    "label_load_count": 0,
    "temporal_feature_build_count": 0,
    "model_fit_count": 0,
    "hyperparameter_search_count": 0,
    "reference_kernel_call_count": 0,
    "raw_market_rescan_count": 0,
    "environment_recompute_count": 0,
    "geometry_recompute_count": 0,
    "concat_hotloop_count": 0,
}


def _bump(key: str, n: int = 1) -> None:
    COUNTERS[key] = COUNTERS.get(key, 0) + n


def reset_counters() -> None:
    for k in COUNTERS:
        COUNTERS[k] = 0


# --------------------------------------------------------------------------- #
# TEMP57 schema                                                                #
# --------------------------------------------------------------------------- #
def _temp57_columns() -> list[str]:
    cols: list[str] = []
    # A. LAG32 -- 8 PAY8 fields x lag(1,2,3,4)
    for c in PAY8_COLS:
        for k in LAGS:
            cols.append(f"{c}__lag{k}")
    # B. PATH9 -- G,L,Z x delta(1,2,4)
    for name in ("g", "l", "z"):
        for k in DELTA_LAGS:
            cols.append(f"{name}__delta{k}")
    # C. CURV3 -- G,L,Z acceleration
    for name in ("g", "l", "z"):
        cols.append(f"{name}__accel")
    # D. SLOPE6 -- G,L,Z x OLS slope over (4,16)
    for name in ("g", "l", "z"):
        for w in SLOPE_WINDOWS:
            cols.append(f"{name}__slope{w}")
    # E. EXTREMA4 -- Z distance from trailing max/min over (4,16)
    for w in EXTREMA_WINDOWS:
        cols.append(f"z__from_max{w}")
        cols.append(f"z__from_min{w}")
    # F. PERSIST3
    cols.append("z__positive_run")
    cols.append("z__negative_run")
    cols.append(f"z__switch_count{SWITCH_WINDOW}")

    if len(cols) != 57:
        raise AssertionError(f"STOP_TEMP57_COUNT len={len(cols)}")
    if len(set(cols)) != len(cols):
        raise AssertionError("STOP_TEMP57_DUPLICATE_NAMES")
    return cols


TEMP57_COLS: list[str] = _temp57_columns()

M0_COLS: list[str] = list(PAY8_COLS)                 # 8
M1_COLS: list[str] = list(PAY8_COLS) + TEMP57_COLS   # 65
assert len(M0_COLS) == 8 and len(M1_COLS) == 65

# Explicit exclusions (01A must never contain these).
EXCLUDED_TOKENS = (
    "p_win", "win33", "mu_win", "mu_loss", "predicted_rr", "ev_c",
    "mfe", "mae", "oracle", "future", "barrier", "target",
)


# --------------------------------------------------------------------------- #
# Vectorized primitives (production kernel; no per-row Python loop)            #
# --------------------------------------------------------------------------- #
def rolling_ols_slope(y: np.ndarray, window: int) -> np.ndarray:
    """OLS slope of y over the trailing [t-window+1 .. t] window.

    Vectorized via np.correlate. Requires a COMPLETE finite window; positions
    with an incomplete window (or any non-finite member) are NaN. Callers must
    additionally mask positions whose window would cross a causal boundary.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    out = np.full(n, np.nan, dtype=float)
    if n < window:
        return out

    x = np.arange(window, dtype=float)
    sx = x.sum()
    sxx = np.dot(x, x)
    denom = sxx - sx * sx / window

    valid = np.isfinite(y)
    y0 = np.where(valid, y, 0.0)

    count = np.correlate(valid.astype(float), np.ones(window), mode="valid")
    sy = np.correlate(y0, np.ones(window), mode="valid")
    sxy = np.correlate(y0, x, mode="valid")

    slope = (sxy - sx * sy / window) / denom
    slope[count != window] = np.nan
    out[window - 1:] = slope
    return out


def capped_true_run(mask: np.ndarray, cap: int = RUN_CAP) -> np.ndarray:
    """Length of the consecutive True-run ending at each position, capped."""
    mask = np.asarray(mask, dtype=bool)
    idx = np.arange(len(mask))
    last_false = np.maximum.accumulate(np.where(~mask, idx, -1))
    run = idx - last_false
    run[~mask] = 0
    return np.minimum(run, cap)


def trailing_sign_switch_count(delta: np.ndarray, *, pos: np.ndarray | None = None,
                               window: int = SWITCH_WINDOW) -> np.ndarray:
    """Count of non-zero sign switches among the trailing `window` deltas.

    Fully vectorized (cumsum differencing); no Python row loop. When `pos`
    (within-block position) is supplied, positions whose trailing window would
    cross a causal boundary are left NaN.
    """
    d = np.asarray(delta, dtype=float)
    n = len(d)
    out = np.full(n, np.nan, dtype=float)
    if n == 0:
        return out

    s = np.sign(d)
    switch = np.zeros(n, dtype=float)
    if n > 1:
        valid_pair = (np.isfinite(s[1:]) & np.isfinite(s[:-1])
                      & (s[1:] != 0) & (s[:-1] != 0))
        switch[1:] = (valid_pair & (s[1:] != s[:-1])).astype(float)

    # cs[k] = sum(switch[0..k-1]); the trailing W deltas ending at i are
    # switch[i-W+1 .. i] -> cs[i+1] - cs[i-W+1].
    cs = np.concatenate([[0.0], np.cumsum(switch)])
    idx = np.arange(n)
    ok = (idx >= window) if pos is None else (np.asarray(pos) >= window)
    if ok.any():
        out[ok] = cs[idx[ok] + 1] - cs[idx[ok] - window + 1]
    return out


# --------------------------------------------------------------------------- #
# Causal continuity                                                            #
# --------------------------------------------------------------------------- #
def add_continuity_blocks(df: pd.DataFrame, *,
                          enforce_15m: bool = True) -> pd.DataFrame:
    """Sort canonically and attach causal block id + within-block position.

    A new block starts whenever ANY of the following holds:
      symbol / side / fold / trading_day / segment changes,
      decision_bar is not contiguous (+1),
      (optionally) the decision_time gap is not exactly one 15m bar.

    Returns a NEW frame with helper columns `_block` (int64) and `_pos`
    (0-based position inside its block). `_pos` is the single primitive that
    makes every lag/window boundary-safe: a lag k or window w is valid only
    where `_pos >= k` (resp. `_pos >= w-1`), because `_pos` counts consecutive
    same-block rows.
    """
    keys = [k for k in CONTINUITY_KEYS if k in df.columns]
    out = df.sort_values(["symbol", "side", "decision_bar"],
                         kind="stable").reset_index(drop=True)
    n = len(out)
    new = np.zeros(n, dtype=bool)
    if n:
        new[0] = True

    for k in keys:
        # Vectorized, dtype-agnostic, NaN-safe: a change is a genuine value
        # change; NaN -> NaN is NOT a change.
        s = out[k]
        prev = s.shift(1)
        ne = s.ne(prev).to_numpy(dtype=bool)
        both_na = (s.isna().to_numpy(dtype=bool)
                   & prev.isna().to_numpy(dtype=bool))
        new |= ne & ~both_na

    db = out["decision_bar"].to_numpy(dtype=np.int64)
    if n > 1:
        new[1:] |= (db[1:] - db[:-1]) != 1

    if enforce_15m and "decision_time" in out.columns and n > 1:
        dt = pd.to_datetime(out["decision_time"]).to_numpy("datetime64[ns]")
        gap_min = (dt[1:] - dt[:-1]).astype("timedelta64[m]").astype(np.int64)
        new[1:] |= (gap_min != 15)

    block = np.cumsum(new) - 1
    starts = np.maximum.accumulate(np.where(new, np.arange(n), 0))
    out["_block"] = block.astype(np.int64)
    out["_pos"] = (np.arange(n) - starts).astype(np.int64)
    return out


# --------------------------------------------------------------------------- #
# TEMP57 builder                                                               #
# --------------------------------------------------------------------------- #
def build_temp57(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 57 strictly-past-only temporal features.

    `df` must already carry canonical PAY8 columns plus `_pos` from
    `add_continuity_blocks`, sorted by (block, decision_bar). Every lag /
    window is masked to `_pos`, so no value can cross a causal boundary.
    """
    missing = [c for c in PAY8_COLS if c not in df.columns]
    if missing:
        raise AssertionError(f"STOP_PAY8_MISSING {missing}")
    if "_pos" not in df.columns:
        raise AssertionError("STOP_NO_CONTINUITY_POS")

    n = len(df)
    pos = df["_pos"].to_numpy(dtype=np.int64)
    out = pd.DataFrame(index=df.index)

    def lag_of(v: np.ndarray, k: int) -> np.ndarray:
        x = np.full(n, np.nan, dtype=float)
        if n > k:
            x[k:] = v[:-k]
        x[pos < k] = np.nan
        return x

    # -------- A. LAG32 --------
    lag_store: dict[tuple[str, int], np.ndarray] = {}
    for c in PAY8_COLS:
        v = df[c].to_numpy(dtype=float)
        for k in LAGS:
            x = lag_of(v, k)
            lag_store[(c, k)] = x
            out[f"{c}__lag{k}"] = x

    # -------- B. PATH9 + C. CURV3 --------
    series = {"g": df[G_COL].to_numpy(dtype=float),
              "l": df[L_COL].to_numpy(dtype=float),
              "z": df[Z_COL].to_numpy(dtype=float)}
    d1_store: dict[str, np.ndarray] = {}
    for name, v in series.items():
        for k in DELTA_LAGS:
            d = v - lag_of(v, k)
            d[pos < k] = np.nan
            out[f"{name}__delta{k}"] = d
            if k == 1:
                d1_store[name] = d

    # -------- C. CURV3 (all deltas first, then all accelerations) --------
    # accel = d1_t - d1_{t-1}; d1 is already NaN where _pos < 1, so its lag is
    # automatically NaN where _pos < 2.
    for name in ("g", "l", "z"):
        d1 = d1_store[name]
        accel = np.full(n, np.nan, dtype=float)
        if n > 1:
            accel[1:] = d1[1:] - d1[:-1]
        accel[pos < 2] = np.nan
        out[f"{name}__accel"] = accel

    # -------- D. SLOPE6 --------
    for name, v in series.items():
        for w in SLOPE_WINDOWS:
            s = rolling_ols_slope(v, w)
            s[pos < w - 1] = np.nan
            out[f"{name}__slope{w}"] = s

    # -------- E. EXTREMA4 (Z only) --------
    z = series["z"]
    zs = pd.Series(z)
    for w in EXTREMA_WINDOWS:
        rmax = zs.rolling(w, min_periods=w).max().to_numpy(dtype=float)
        rmin = zs.rolling(w, min_periods=w).min().to_numpy(dtype=float)
        hi = z - rmax
        lo = z - rmin
        hi[pos < w - 1] = np.nan
        lo[pos < w - 1] = np.nan
        out[f"z__from_max{w}"] = hi
        out[f"z__from_min{w}"] = lo

    # -------- F. PERSIST3 --------
    dz = out["z__delta1"].to_numpy(dtype=float)
    out["z__positive_run"] = capped_true_run(np.isfinite(dz) & (dz > 0),
                                             cap=RUN_CAP)
    out["z__negative_run"] = capped_true_run(np.isfinite(dz) & (dz < 0),
                                             cap=RUN_CAP)

    out[f"z__switch_count{SWITCH_WINDOW}"] = trailing_sign_switch_count(
        dz, pos=pos, window=SWITCH_WINDOW)

    if list(out.columns) != TEMP57_COLS:
        raise AssertionError("STOP_TEMP57_SCHEMA_ORDER")
    _bump("temporal_feature_build_count")
    return out


def attach_temp57(df: pd.DataFrame, *, enforce_15m: bool = True) -> pd.DataFrame:
    """Canonical entry: sort -> continuity blocks -> TEMP57, concatenated once."""
    blocked = add_continuity_blocks(df, enforce_15m=enforce_15m)
    t57 = build_temp57(blocked)
    return pd.concat([blocked, t57], axis=1)


# --------------------------------------------------------------------------- #
# Common support                                                               #
# --------------------------------------------------------------------------- #
def common_support_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows where all PAY8 and all TEMP57 are finite (M0 and M1 identical)."""
    pay = df[list(PAY8_COLS)].to_numpy(dtype=float)
    ok = np.isfinite(pay).all(axis=1)
    if set(TEMP57_COLS).issubset(df.columns):
        t = df[TEMP57_COLS].to_numpy(dtype=float)
        ok &= np.isfinite(t).all(axis=1)
    return ok


# --------------------------------------------------------------------------- #
# Canonical loaders                                                            #
# --------------------------------------------------------------------------- #
def load_pay8_from_state(state_path: Path | str = STATE_PARQUET) -> pd.DataFrame:
    """Derive canonical PAY8 from the frozen state artifact.

    Uses the frozen owner `build_pay8_from_state`, whose contract is
    "no environment / geometry reload". Emits both LONG and SHORT per bar.
    """
    sdf = pd.read_parquet(state_path)
    _bump("state_load_count")
    frames = []
    for sym, g in sdf.groupby("symbol", sort=True):
        g = g.sort_values("bar_index", kind="stable")
        if (g["bar_index"].to_numpy() != np.arange(len(g))).any():
            raise AssertionError(f"STOP_STATE_BAR_INDEX_NOT_DENSE {sym}")
        frames.append(build_pay8_from_state(g, sym))
    pay8 = pd.concat(frames, ignore_index=True)
    _bump("pay8_load_count")
    return pay8


def load_state_boundaries(state_path: Path | str = STATE_PARQUET) -> pd.DataFrame:
    """Trading day + canonical segment per (symbol, bar_index)."""
    sdf = pd.read_parquet(
        state_path, columns=["symbol", "bar_index", "trading_day", "segment"])
    _bump("state_load_count")
    return sdf


def load_labels_td5(label_path: Path | str = LABEL_TRAIN_PARQUET
                    ) -> pd.DataFrame:
    """Canonical TRAIN labels, primary horizon td5 only. TEST is never read."""
    df = pd.read_parquet(label_path)
    _bump("label_load_count")
    return df[df["horizon"] == PRIMARY_HORIZON].reset_index(drop=True)


SIDE_KEY = ("symbol", "decision_bar", "side")

# Columns carried from the dense bar series onto candidate rows. `decision_time`
# is intentionally excluded: the labels already own it and a collision would
# produce decision_time_x/_y.
BAR_KEEP = ["symbol", "side", "decision_bar", "trading_day", "segment"]


# --------------------------------------------------------------------------- #
# Dense bar-level TEMP57 (the actual 15m market trajectory)                    #
# --------------------------------------------------------------------------- #
def build_bar_level_temp57(state_path: Path | str = STATE_PARQUET, *,
                           enforce_15m: bool = True) -> pd.DataFrame:
    """Build TEMP57 over the FULL canonical 15m bar series, both sides.

    The trajectory must be the MARKET path over consecutive 15m bars. Candidate
    rows are sparsely and irregularly spaced, so lagging across candidate rows
    would not be a 15m-bar trajectory at all (and would leave almost no support
    for a 16-bar window). PAY8 is derived once per symbol by the frozen owner,
    which already emits one row per (symbol, bar, side); TEMP57 is built on that
    dense series and later joined onto candidates.
    """
    sdf = pd.read_parquet(state_path)
    _bump("state_load_count")
    frames = []
    for sym, g in sdf.groupby("symbol", sort=True):
        g = g.sort_values("bar_index", kind="stable")
        if (g["bar_index"].to_numpy() != np.arange(len(g))).any():
            raise AssertionError(f"STOP_STATE_BAR_INDEX_NOT_DENSE {sym}")
        pay = build_pay8_from_state(g, sym)
        bnd = g[["bar_index", "trading_day", "segment"]].rename(
            columns={"bar_index": "decision_bar"})
        frames.append(pay.merge(bnd, on="decision_bar", how="left",
                                validate="m:1"))
    bars = pd.concat(frames, ignore_index=True)
    _bump("pay8_load_count")
    return attach_temp57(bars, enforce_15m=enforce_15m)


# --------------------------------------------------------------------------- #
# Canonical common-support analysis frame                                      #
# --------------------------------------------------------------------------- #
def build_analysis_frame(state_path: Path | str = STATE_PARQUET,
                         label_path: Path | str = LABEL_TRAIN_PARQUET,
                         *, enforce_15m: bool = True
                         ) -> tuple[pd.DataFrame, dict]:
    """Build the canonical TD5 common-support frame ONCE.

    Canonical state is loaded once, PAY8 derived once by the frozen owner, the
    canonical TRAIN td5 labels loaded once, and TEMP57 built once on the dense
    bar series. M0 and M1 then train/evaluate on the SAME rows.
    """
    lab = load_labels_td5(label_path)
    n_lab = int(len(lab))
    bars = build_bar_level_temp57(state_path, enforce_15m=enforce_15m)
    keep = BAR_KEEP + list(PAY8_COLS) + TEMP57_COLS

    # PAY8/TEMP57 are authoritative; drop colliding label-side geometry so the
    # merge cannot emit _x/_y (same rule as payoff_ratio_model_v1).
    lab = lab.drop(columns=[c for c in PAY8_COLS if c in lab.columns])
    df = lab.merge(bars[keep], on=list(SIDE_KEY), how="left",
                   validate="m:1", indicator=True)
    unmatched = int((df["_merge"].to_numpy(object) != "both").sum())
    df = df.drop(columns=["_merge"])
    if unmatched:
        raise AssertionError(f"STOP_TEMP57_JOIN_UNMATCHED n={unmatched}")

    n_before = int(len(df))
    sup = common_support_mask(df)

    out = df[sup].reset_index(drop=True)
    report = {
        "labels_td5_rows": n_lab,
        "rows_before_temporal_support": n_before,
        "rows_after_temporal_support": int(len(out)),
        "rows_removed": int(n_before - len(out)),
        "n_symbols": int(out["symbol"].nunique()),
        "per_symbol_counts": {s: int(c)
                              for s, c in out["symbol"].value_counts().items()},
        "sides": sorted(out["side"].unique().tolist()),
        "n_trading_days": int(out["trading_day"].nunique()),
    }
    return out, report


FORCE_N_JOBS = 1


@contextmanager
def forced_single_thread():
    """Thread-only override while fitting.

    The repo already forces n_jobs=1 for this regressor family as a segfault
    guard (decomposed_models_v2.FORCE_N_JOBS). n_jobs affects only threading,
    not the fitted model; the repo documents that historical best_iteration
    values are unchanged by it.
    """
    old = _payoff_v1.REG_PARAMS.get("n_jobs", None)
    _payoff_v1.REG_PARAMS["n_jobs"] = FORCE_N_JOBS
    try:
        yield
    finally:
        if old is None:
            _payoff_v1.REG_PARAMS.pop("n_jobs", None)
        else:
            _payoff_v1.REG_PARAMS["n_jobs"] = old


# --------------------------------------------------------------------------- #
# M0 / M1 OOF fitting                                                          #
# --------------------------------------------------------------------------- #
def run_arms_oof(frame: pd.DataFrame, plan=None,
                 arm_cols: dict[str, list[str]] | None = None):
    """Fit M0 and M1 per canonical outer fold; emit OOF predictions only.

    Reuses the frozen two-head `fit_bundle` (identical params, identical early
    stopping). 5 folds x 2 arms x 2 heads = 20 fits. No hyperparameter search.
    """
    if plan is None:
        plan = build_fold_plan(frame)
    if arm_cols is None:
        arm_cols = {"M0": M0_COLS, "M1": M1_COLS}

    y = frame["episode_return_atr"].to_numpy(dtype=float)
    w = frame["sample_weight"].to_numpy(dtype=float)
    preds = {a: {"muW": np.full(len(frame), np.nan, dtype=float),
                 "muL": np.full(len(frame), np.nan, dtype=float)}
             for a in arm_cols}

    n_fit = 0
    fold_rows = {}
    with forced_single_thread():
        for split in iter_fold_splits(frame, plan):
            fc, es, ou = split.fit_core, split.es, split.outer
            fold_rows[int(split.fold)] = int(np.count_nonzero(ou))
            for arm, cols in arm_cols.items():
                Xtr = frame.loc[fc, cols].to_numpy(dtype=np.float32)
                Xv = frame.loc[es, cols].to_numpy(dtype=np.float32)
                Xou = frame.loc[ou, cols].to_numpy(dtype=np.float32)
                bundle = fit_bundle(Xtr, y[fc], w[fc], Xv, y[es], w[es],
                                    PRIMARY_HORIZON)
                preds[arm]["muW"][ou] = bundle.win_magnitude_model.predict(Xou)
                preds[arm]["muL"][ou] = bundle.loss_magnitude_model.predict(Xou)
                n_fit += 2

    _bump("model_fit_count", n_fit)
    return preds, plan, n_fit, fold_rows


# --------------------------------------------------------------------------- #
# Paired estimands                                                             #
# --------------------------------------------------------------------------- #
def paired_differences(frame: pd.DataFrame, preds: dict) -> pd.DataFrame:
    """Per-row paired error differences.

    d_win  = |Y  - muW_M0| - |Y  - muW_M1|   (win rows,  Y >  0)
    d_loss = |-Y - muL_M0| - |-Y - muL_M1|   (loss rows, Y <= 0)

    Positive => TEMP57 improved the prediction.
    """
    y = frame["episode_return_atr"].to_numpy(dtype=float)
    w = frame["sample_weight"].to_numpy(dtype=float)
    is_win = y > 0
    muW0, muL0 = preds["M0"]["muW"], preds["M0"]["muL"]
    muW1, muL1 = preds["M1"]["muW"], preds["M1"]["muL"]

    d_win = np.where(is_win, np.abs(y - muW0) - np.abs(y - muW1), np.nan)
    d_loss = np.where(~is_win, np.abs((-y) - muL0) - np.abs((-y) - muL1), np.nan)

    return pd.DataFrame({
        "symbol": frame["symbol"].to_numpy(),
        "side": frame["side"].to_numpy(),
        "trading_day": frame["trading_day"].to_numpy(),
        "y": y,
        "w": w,
        "is_win": is_win,
        "muW_M0": muW0, "muW_M1": muW1,
        "muL_M0": muL0, "muL_M1": muL1,
        "d_win": d_win,
        "d_loss": d_loss,
    })


def block_bootstrap_weighted_mean(d: np.ndarray, w: np.ndarray, days, *,
                                  B: int, seed: int = BOOTSTRAP_SEED,
                                  block: int = BOOTSTRAP_BLOCK_DAYS) -> dict:
    """Paired 5-trading-day block bootstrap of the weighted mean of d.

    Resamples complete day-blocks with replacement (tail remainder dropped, the
    audited-owner rule). Uses per-block weight sums so it is O(B * n_blocks).
    """
    d = np.asarray(d, dtype=float)
    w = np.asarray(w, dtype=float)
    days = np.asarray(days)
    fin = np.isfinite(d) & np.isfinite(w) & (w > 0)
    if not fin.any():
        return {"bootstrap_mean": float("nan"), "bootstrap_ci_low": float("nan"),
                "bootstrap_ci_high": float("nan"), "n_valid_reps": 0,
                "n_blocks": 0, "excluded_tail_days": 0}
    d, w, days = d[fin], w[fin], days[fin]

    order = np.argsort(days, kind="stable")
    uniq, inv = np.unique(days[order], return_inverse=True)
    n_days = len(uniq)
    n_complete = n_days // block
    if n_complete < 1:
        raise AssertionError(f"STOP_NOT_ENOUGH_DAY_BLOCKS n_days={n_days}")

    day_block = np.arange(n_days) // block
    in_complete = np.arange(n_days) < n_complete * block
    row_in = in_complete[inv]
    rb = day_block[inv]

    nb = int(n_complete)
    sw = np.zeros(nb, dtype=float)
    swd = np.zeros(nb, dtype=float)
    np.add.at(sw, rb[row_in], w[order][row_in])
    np.add.at(swd, rb[row_in], (w * d)[order][row_in])

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nb, size=(int(B), nb))
    sws = sw[idx].sum(axis=1)
    swds = swd[idx].sum(axis=1)
    means = np.where(sws > 0, swds / np.where(sws > 0, sws, 1.0), np.nan)

    return {
        "bootstrap_mean": float(np.nanmean(means)),
        "bootstrap_ci_low": float(np.nanpercentile(means, 2.5)),
        "bootstrap_ci_high": float(np.nanpercentile(means, 97.5)),
        "n_valid_reps": int(np.count_nonzero(np.isfinite(means))),
        "n_blocks": nb,
        "excluded_tail_days": int(n_days - n_complete * block),
    }


def estimand(d: np.ndarray, w: np.ndarray, days, *, B: int,
             seed: int = BOOTSTRAP_SEED) -> dict:
    """Observed weighted mean + block-bootstrap CI for one paired difference."""
    fin = np.isfinite(d) & np.isfinite(w)
    sw = float(np.sum(w[fin]))
    observed = float(np.sum(w[fin] * d[fin]) / sw) if sw > 0 else float("nan")
    out = {"observed_D": observed, "n": int(np.count_nonzero(fin)),
           "n_trading_days": int(len(np.unique(np.asarray(days)[fin])))}
    out.update(block_bootstrap_weighted_mean(d, w, days, B=B, seed=seed))
    return out


def estimate_universes(pdiff: pd.DataFrame, *, B: int) -> dict:
    """pooled + every symbol + LONG + SHORT, for both heads."""
    res: dict[str, dict] = {}
    for head in ("win", "loss"):
        col = f"d_{head}"
        res[head] = {"pooled": estimand(pdiff[col].to_numpy(float),
                                        pdiff["w"].to_numpy(float),
                                        pdiff["trading_day"].to_numpy(),
                                        B=B)}
        per_sym = {}
        for sym, g in pdiff.groupby("symbol", sort=True):
            per_sym[str(sym)] = estimand(g[col].to_numpy(float),
                                         g["w"].to_numpy(float),
                                         g["trading_day"].to_numpy(), B=B)
        res[head]["per_symbol"] = per_sym
        by_side = {}
        for side, g in pdiff.groupby("side", sort=True):
            by_side[str(side)] = estimand(g[col].to_numpy(float),
                                          g["w"].to_numpy(float),
                                          g["trading_day"].to_numpy(), B=B)
        res[head]["by_side"] = by_side
    return res
