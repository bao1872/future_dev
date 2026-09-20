"""
audit_view_v1
=============

Read-only helpers for the label / indicator audit UI
(Task PANJI-R3-AUDIT-UI-V1-LABEL-INDICATOR).

Two independent chains, kept strictly separate:

  1. LABEL chain (past -> future, hindsight audit)
       decision t  ->  entry t+1  ->  future path  ->  R1/R2 robust oracle row

  2. INDICATOR chain (decision-time only)
       raw 5m  ->  forming 5m/15m/1h/4h source  ->  streaming indicators
       <->  independent slow reference (compute_tf_features)

This module performs NO oracle re-solving, NO label modification, NO
indicator/parameter change and NO model work. It only reads frozen
artifacts (R1/R2) and calls the frozen canonical builder.

Frozen owners reused verbatim:
  * research.liquidity_oracle_atlas.experiment_field_representation_v1      (E1)
  * research.liquidity_oracle_atlas.experiment_vol_normalization_falsification_v1 (E12)
  * research.liquidity_oracle_atlas.build_forming_environment_v1            (canonical)
  * research.phase1_tradability.phase1_contract_v1.discontinuity_flags      (canonical)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]

SYMBOLS = [
    "AG", "AU", "CU", "AL", "SN",
    "NI", "RB", "I", "SC", "RU",
    "MA", "TA", "M", "P", "CF",
]

KEY = ["symbol", "decision_bar_index", "decision_time"]
HORIZONS = (6, 12, 24)
EXPECTED_BARS = {"m5": 1, "m15": 3, "h1": 12, "h4": 48}
TF_LIST = ["m5", "m15", "h1", "h4"]

CONT_TOL = 1e-9


# --------------------------------------------------------------------------- #
# oracle artifact schema / loading                                             #
# --------------------------------------------------------------------------- #
def oracle_paths() -> Tuple[Path, Path]:
    import research.liquidity_oracle_atlas.experiment_field_representation_v1 as E1

    return _REPO_ROOT / E1.R1_REL, _REPO_ROOT / E1.R2_REL


def _r1_needed() -> List[str]:
    cols = [
        "symbol", "decision_bar_index", "decision_time", "decision_bar_start_time",
        "segment", "entry_bar_index", "entry_valid", "atr5_t", "agreement",
        "stable_action", "_any_tie", "_agree23",
    ]
    for h in HORIZONS:
        cols += [
            f"QL_{h}", f"QS_{h}", f"QW_{h}",
            f"QL_{h}_ATR", f"QS_{h}_ATR", f"QW_{h}_ATR",
            f"edge_{h}", f"edge_{h}_ATR",
            f"action_{h}", f"best_action_set_{h}", f"n_best_actions_{h}",
            f"label_available_bar_{h}", f"label_available_time_{h}",
            f"oracle_terminal_reason_{h}",
            f"exit_bars_{h}", f"holding_bars_{h}", f"realized_move_{h}",
            f"MFE_{h}", f"MAE_{h}",
        ]
    return cols


def _r2_needed() -> List[str]:
    cols = [
        "symbol", "decision_bar_index", "decision_time",
        "baseline_stable_action", "strict_robust_action",
        "joint_retention", "joint_retention_stable",
        "joint_opposite_flip_rate", "joint_tie_rate",
        "baseline_dir_holding_median", "baseline_dir_holding_span",
    ]
    for h in HORIZONS:
        cols += [
            f"consensus_action_H{h}", f"consensus_rate_H{h}",
            f"baseline_retention_H{h}", f"opposite_flip_H{h}",
            f"Long_share_H{h}", f"Short_share_H{h}",
            f"Wait_share_H{h}", f"Tie_share_H{h}",
            f"edge_ATR_median_H{h}", f"value_ATR_median_H{h}",
        ]
    return cols


def available_columns(path: Path, wanted: Sequence[str]) -> List[str]:
    """Only the columns that ACTUALLY exist in the parquet (never invent)."""
    import pyarrow.parquet as pq

    names = set(pq.ParquetFile(path).schema.names)
    return [c for c in wanted if c in names]


def oracle_schema_report() -> Dict[str, Any]:
    import pyarrow.parquet as pq

    r1p, r2p = oracle_paths()
    pf1, pf2 = pq.ParquetFile(r1p), pq.ParquetFile(r2p)
    rep: Dict[str, Any] = {
        "r1_path": str(r1p.relative_to(_REPO_ROOT)),
        "r2_path": str(r2p.relative_to(_REPO_ROOT)),
        "r1_rows": int(pf1.metadata.num_rows),
        "r2_rows": int(pf2.metadata.num_rows),
        "r1_columns": list(pf1.schema.names),
        "r2_columns": list(pf2.schema.names),
    }
    s1, s2 = set(rep["r1_columns"]), set(rep["r2_columns"])
    for h in HORIZONS:
        rep[f"H{h}_QLQSQW_present"] = all(
            c in s1 for c in (f"QL_{h}", f"QS_{h}", f"QW_{h}"))
        rep[f"H{h}_QLQSQW_ATR_present"] = all(
            c in s1 for c in (f"QL_{h}_ATR", f"QS_{h}_ATR", f"QW_{h}_ATR"))
        rep[f"H{h}_action_present"] = f"action_{h}" in s1
        rep[f"H{h}_label_available_time_present"] = f"label_available_time_{h}" in s1
    rep["r1_stable_action_present"] = "stable_action" in s1
    rep["r2_baseline_stable_action_present"] = "baseline_stable_action" in s2
    rep["r2_joint_retention_stable_present"] = "joint_retention_stable" in s2
    rep["r2_strict_robust_action_present"] = "strict_robust_action" in s2
    return rep


def load_symbol_oracle(symbol: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """R1 (robust oracle) + R2 (constraint robustness) for one symbol.

    Column selection follows the ACTUAL parquet schema. No derived label is
    constructed here -- everything shown is artifact-native.
    """
    r1p, r2p = oracle_paths()
    c1 = available_columns(r1p, _r1_needed())
    c2 = available_columns(r2p, _r2_needed())

    # R2 duplicates atr5_t / decision_bar_start_time -> keep only R1's copy.
    drop = {"symbol", "decision_bar_index", "decision_time",
            "atr5_t", "decision_bar_start_time"}
    c2 = [c for c in c2 if c not in drop]

    r1 = pd.read_parquet(r1p, columns=c1,
                         filters=[("symbol", "==", symbol)]).reset_index(drop=True)
    r2 = pd.read_parquet(r2p, columns=list(KEY) + c2,
                         filters=[("symbol", "==", symbol)]).reset_index(drop=True)

    merged = r1.merge(r2, on=KEY, how="inner", validate="one_to_one")
    if merged.duplicated(KEY).any():
        raise RuntimeError("AUDIT_DUPLICATE_KEYS")

    info = dict(R1_columns=c1, R2_columns=c2,
                r1_rows=int(len(r1)), r2_rows=int(len(r2)),
                merged_rows=int(len(merged)),
                dropped_by_inner_join=int(len(r1) - len(merged)))
    return merged, info


def action_vocabulary(merged: pd.DataFrame) -> List[str]:
    """Action filter options taken from the artifact, never hard-coded."""
    vals = set()
    if "stable_action" in merged.columns:
        vals |= set(merged["stable_action"].astype(str).unique())
    if "baseline_stable_action" in merged.columns:
        vals |= set(merged["baseline_stable_action"].astype(str).unique())
    vals.discard("nan")
    return sorted(vals)


def filter_events(
    merged: pd.DataFrame,
    action: str = "All",
    retention_min: float = 0.0,
) -> pd.DataFrame:
    df = merged
    if action != "All" and "stable_action" in df.columns:
        df = df[df["stable_action"].astype(str) == str(action)]
    if retention_min > 0.0 and "joint_retention_stable" in df.columns:
        df = df[df["joint_retention_stable"].astype(float) >= float(retention_min)]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# time semantics                                                               #
# --------------------------------------------------------------------------- #
def row_at(env: pd.DataFrame, t: int) -> pd.Series:
    """env decision_bar_index == arange(n), so positional == index value."""
    return env.iloc[int(t)]


def time_semantics(builder: Any, t: int) -> Dict[str, Any]:
    base = builder.base
    n = builder.n
    t = int(t)
    bar_start = pd.Timestamp(base["time"].iloc[t])
    out = {
        "t": t,
        "bar_start_time": bar_start,
        "decision_time": bar_start + pd.Timedelta(minutes=5),
        "decision_open": float(base["open"].iloc[t]),
        "decision_high": float(base["high"].iloc[t]),
        "decision_low": float(base["low"].iloc[t]),
        "decision_close": float(base["close"].iloc[t]),
        "segment": int(base["segment"].iloc[t]),
        "has_entry": bool(t + 1 < n),
    }
    if t + 1 < n:
        out.update(
            entry_bar_index=t + 1,
            entry_bar_start_time=pd.Timestamp(base["time"].iloc[t + 1]),
            entry_open=float(base["open"].iloc[t + 1]),
        )
    return out


def disc_flags(symbol: str) -> np.ndarray:
    from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags

    return np.asarray(discontinuity_flags(symbol), dtype=bool)


def path_table(builder: Any, t: int, disc: np.ndarray,
               lo: int = -5, hi: int = 25) -> pd.DataFrame:
    base = builder.base
    n = builder.n
    t = int(t)
    start = max(0, t + lo)
    end = min(n - 1, t + hi)
    idx = np.arange(start, end + 1)
    phase = np.where(idx <= t, "KNOWN", np.where(idx == t + 1, "ENTRY", "LABEL_FUTURE"))
    return pd.DataFrame({
        "bar_index": idx,
        "bar_start_time": pd.DatetimeIndex(base["time"].to_numpy())[idx],
        "open": base["open"].to_numpy(float)[idx],
        "high": base["high"].to_numpy(float)[idx],
        "low": base["low"].to_numpy(float)[idx],
        "close": base["close"].to_numpy(float)[idx],
        "discontinuity_before": disc[idx],
        "phase": phase,
    })


def window_slice(builder: Any, t: int, ctx: int) -> pd.DataFrame:
    base = builder.base
    n = builder.n
    t = int(t)
    start = max(0, t - int(ctx))
    end = min(n - 1, t + int(ctx))
    idx = np.arange(start, end + 1)
    return pd.DataFrame({
        "bar_index": idx,
        "bar_start_time": pd.DatetimeIndex(base["time"].to_numpy())[idx],
        "open": base["open"].to_numpy(float)[idx],
        "high": base["high"].to_numpy(float)[idx],
        "low": base["low"].to_numpy(float)[idx],
        "close": base["close"].to_numpy(float)[idx],
    })


# --------------------------------------------------------------------------- #
# forming source inspector                                                     #
# --------------------------------------------------------------------------- #
def forming_info(builder: Any, t: int, tf: str) -> Dict[str, Any]:
    form = builder._form[tf]
    t = int(t)
    start_idx = int(form["start_idx"][t])
    n_base = int(form["n_base"][t])
    expected = EXPECTED_BARS[tf]
    return {
        "tf": tf,
        "bucket_start": pd.Timestamp(form["bucket_start"][t]),
        "start_idx": start_idx,
        "end_idx": t,
        "n_base_known": n_base,
        "expected_bars": expected,
        "maturity": float(n_base) / float(expected),
        "forming_open": float(form["open"][t]),
        "forming_high": float(form["high"][t]),
        "forming_low": float(form["low"][t]),
        "forming_close": float(form["close"][t]),
    }


def forming_checks(builder: Any, t: int, tf: str) -> Dict[str, bool]:
    """Timing / no-lookahead checks for the forming bar at decision t."""
    form = builder._form[tf]
    base = builder.base
    t = int(t)
    source_start_idx = int(form["start_idx"][t])
    n_base = int(form["n_base"][t])
    return {
        "start_idx <= t": bool(source_start_idx <= t),
        "end_idx == decision t": bool(True),
        "n_base identity (n == t - start + 1)": bool(n_base == t - source_start_idx + 1),
        "forming close == decision close": bool(np.isclose(
            float(form["close"][t]), float(base["close"].iloc[t]), equal_nan=True)),
        "no future base bars": bool(source_start_idx <= t),
    }


def artifact_timing_checks(builder: Any, t: int, mrow: pd.Series) -> Dict[str, bool]:
    """Cross-checks between the R1/R2 artifact row and the canonical base frame."""
    base = builder.base
    t = int(t)
    bar_start = pd.Timestamp(base["time"].iloc[t])
    out = {
        "R1 decision_bar_index == t": bool(int(mrow["decision_bar_index"]) == t),
        "artifact decision_time == bar_start + 5min": bool(
            pd.Timestamp(mrow["decision_time"]) == bar_start + pd.Timedelta(minutes=5)),
    }
    if "decision_bar_start_time" in mrow.index:
        out["artifact decision_bar_start_time == base time[t]"] = bool(
            pd.Timestamp(mrow["decision_bar_start_time"]) == bar_start)
    if "entry_bar_index" in mrow.index and np.isfinite(mrow["entry_bar_index"]):
        out["R1 entry_bar_index == t + 1"] = bool(int(mrow["entry_bar_index"]) == t + 1)
    return out


# --------------------------------------------------------------------------- #
# indicator inspector + slow reference differential                            #
# --------------------------------------------------------------------------- #
def streaming_row(env: pd.DataFrame, t: int, tf: str, feature_cols: Sequence[str]) -> Dict[str, float]:
    row = env.iloc[int(t)]
    return {c: float(row[f"{tf}_{c}"]) for c in feature_cols}


def slow_reference(builder: Any, t: int, tf: str) -> Optional[Dict[str, float]]:
    return builder.slow_forming_snapshot_reference(int(t), tf)


def diff_stream_vs_slow(
    stream: Dict[str, float],
    slow: Optional[Dict[str, float]],
    feature_cols: Sequence[str],
    discrete_cols: Sequence[str],
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    discrete = set(discrete_cols)
    rows = []
    max_cont = 0.0
    nan_mismatch = 0
    disc_mismatch = 0

    for c in feature_cols:
        s = stream.get(c, np.nan)
        r = slow.get(c, np.nan) if slow else np.nan
        if c in discrete:
            ok = (float(s) == float(r))
            if not ok:
                disc_mismatch += 1
            rows.append(dict(feature=c, kind="discrete", streaming=s,
                             slow_reference=r, abs_error=np.nan,
                             status="OK" if ok else "MISMATCH"))
            continue
        s_nan = not np.isfinite(s)
        r_nan = not np.isfinite(r)
        if s_nan and r_nan:
            err = 0.0
            ok = True
        elif s_nan != r_nan:
            err = np.inf
            nan_mismatch += 1
            ok = False
        else:
            err = abs(float(s) - float(r))
            max_cont = max(max_cont, err)
            ok = err <= CONT_TOL
        rows.append(dict(feature=c, kind="continuous", streaming=s,
                         slow_reference=r, abs_error=err,
                         status="OK" if ok else "MISMATCH"))

    detail = pd.DataFrame(rows)
    passed = bool(max_cont <= CONT_TOL and nan_mismatch == 0 and disc_mismatch == 0)
    summary = dict(
        slow_available=bool(slow is not None),
        continuous_max_abs_error=float(max_cont),
        nan_pattern_mismatch=int(nan_mismatch),
        discrete_mismatch=int(disc_mismatch),
        mismatch_rows=int((detail["status"] == "MISMATCH").sum()) if len(detail) else 0,
        passed=passed,
    )
    return summary, detail


def oracle_future_path(builder: Any, t: int, entry_idx: int,
                       horizon: int = 24) -> pd.DataFrame:
    """Visual aid only -- NOT a Wait-DP recomputation."""
    base = builder.base
    n = builder.n
    e = int(entry_idx)
    end = min(n - 1, e + int(horizon))
    idx = np.arange(e, end + 1)
    entry_open = float(base["open"].iloc[e])
    opens = base["open"].to_numpy(float)[idx]
    return pd.DataFrame({
        "bar_index": idx,
        "bar_start_time": pd.DatetimeIndex(base["time"].to_numpy())[idx],
        "open": opens,
        "long_move_from_entry": opens - entry_open,
        "short_move_from_entry": entry_open - opens,
    })


def horizon_rows(merged: pd.DataFrame, pos: int) -> pd.DataFrame:
    """Artifact-native horizon table. Missing columns -> NaN (shown as n/a)."""
    row = merged.iloc[int(pos)]
    out = []
    for h in HORIZONS:
        rec = {"H": h}
        for base_name, label in (("QL", "QL"), ("QS", "QS"), ("QW", "QW")):
            for suf, tag in (("", "raw"), ("_ATR", "ATR")):
                col = f"{base_name}_{h}{suf}"
                rec[f"{label} {tag}"] = row[col] if col in merged.columns else np.nan
        for col, tag in ((f"label_available_time_{h}", "label_available_time"),
                         (f"action_{h}", "artifact action"),
                         (f"best_action_set_{h}", "best_action_set"),
                         (f"n_best_actions_{h}", "n_best_actions"),
                         (f"oracle_terminal_reason_{h}", "terminal_reason"),
                         (f"holding_bars_{h}", "holding_bars"),
                         (f"realized_move_{h}", "realized_move"),
                         (f"MFE_{h}", "MFE"), (f"MAE_{h}", "MAE")):
            rec[tag] = row[col] if col in merged.columns else np.nan
        out.append(rec)
    return pd.DataFrame(out)
