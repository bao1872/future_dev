"""FUTURE-R8-M15-DECOMPOSED-VALUE-DATASET-V1.

Builds the decomposed feature / label artifacts for the Win-Probability /
Payoff-Ratio experiment (R9A / R9B / R9C / R10).

Design (supersedes the unified "Opportunity Value Model" framing):

- Reuses the FROZEN R8 structural-renewal labels and renewal axis. The R8
  ``materialize`` step is run first to satisfy the provenance requirement
  (``generator_code_sha == FINAL_CODE_SHA`` in the R8 manifest);
- Derives ``WIN33`` = side-oriented STRUCT33 (exactly 33 features) from the
  per-bar ``X33`` state. It is the Win-Probability feature contract and MUST
  NOT contain any PAY8 field;
- Derives ``PAY8`` = 8 structural / payoff geometry features per
  ``(symbol, decision_bar, side)``. It is the Payoff feature contract and MUST
  NOT contain STRUCT33 / DTP9 / trend / probability / oracle fields.

All artifacts are written to ``artifacts/decomposed_value_v1/``.

Governance:
- TEST labels are materialized but are NEVER read during R9A / R9B fitting
  (§45). They are only read once, post-verdict, by the R10 runner.
- The manifest binds the exact ``generator_code_sha`` and every artifact SHA.
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.structural_renewal_dataset_v1 import (
    ARTIFACT_DIR,
    STATE_PARQUET,
    RENEWAL_AXIS_PARQUET,
    LABEL_PARQUETS,
    SIDE_KEY,
    HORIZONS,
    STRUCT33,
    SYMBOLS,
    load_symbol_state,
    materialize as r8_materialize,
    sha256_file,
    opp36_schema_sha256,
    orient_struct33_router_side,
    structural_barriers,
    bracket_metrics,
    _log_rr,
)
from research.liquidity_oracle_atlas.direction_null_baseline_v1 import (
    build_frozen_split,
)

TASK_ID = "FUTURE-R8-M15-DECOMPOSED-VALUE-DATASET-V1"
BASE_SHA = "f3ca3a04317126e8f35afe7430d7aea202a9175a"

# New experiment artifact directory.
DEC_ARTIFACT_DIR = os.path.join("artifacts", "decomposed_value_v1")
STATE_PARQUET_D = os.path.join(DEC_ARTIFACT_DIR, "state_v1.parquet")
WIN_FEATURES_PARQUET = os.path.join(DEC_ARTIFACT_DIR, "win_features_v1.parquet")
PAYOFF_FEATURES_PARQUET = os.path.join(DEC_ARTIFACT_DIR, "payoff_features_v1.parquet")
LABEL_PARQUETS_D = {
    s: os.path.join(DEC_ARTIFACT_DIR, f"labels_{s}_v1.parquet")
    for s in ("train", "val", "test")
}
RENEWAL_AXIS_PARQUET_D = os.path.join(
    DEC_ARTIFACT_DIR, "renewal_event_axis_v1.parquet")
MANIFEST_JSON_D = os.path.join(DEC_ARTIFACT_DIR, "r8_manifest_v1.json")

# §6 / §7: exact feature contracts.
WIN33_N = 33
PAY8_N = 8
WIN33_COLS = [f"win33_{i:02d}" for i in range(WIN33_N)]
PAY8_COLS = [
    "reward_distance_atr",
    "risk_distance_atr",
    "log_structural_rr",
    "ahead_zone_width_atr",
    "back_zone_width_atr",
    "ahead_zone_strength",
    "back_zone_strength",
    "atr_over_abs_price",
]

COUNTERS = {
    "state_artifact_loads": 0,
    "feature_builds": 0,
    "symbol_loops": 0,
}


def _bump(name, n=1):
    COUNTERS[name] = COUNTERS.get(name, 0) + int(n)


def reset_counters():
    for k in list(COUNTERS):
        COUNTERS[k] = 0


def _git_head_sha():
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(__file__),
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def _schema_sha256(columns):
    h = hashlib.sha256()
    for c in columns:
        h.update(str(c).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# 1. WIN33 builder (side-oriented STRUCT33)                                     #
# --------------------------------------------------------------------------- #
def build_win33_frame(st, symbol: str) -> pd.DataFrame:
    """WIN33 = orient_struct33_router_side(raw X33, is_long). Exactly 33 cols."""
    n = st.n_bars
    x33 = np.concatenate([st.X33, st.X33], axis=0)
    is_long = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])
    win33 = orient_struct33_router_side(x33, is_long)  # [2n, 33]
    assert win33.shape == (2 * n, WIN33_N), win33.shape
    df = pd.DataFrame(win33.astype(np.float32), columns=list(WIN33_COLS))
    df.insert(0, "side", np.where(is_long, "LONG", "SHORT"))
    df.insert(0, "decision_bar", np.tile(np.arange(n), 2))
    df.insert(0, "symbol", symbol)
    _bump("feature_builds")
    return df


# --------------------------------------------------------------------------- #
# 2. PAY8 builder (structural / payoff geometry)                               #
# --------------------------------------------------------------------------- #
def build_pay8_frame(st, symbol: str) -> pd.DataFrame:
    """PAY8 = 8 causal geometry features per (symbol, decision_bar, side).

    LONG : ahead = resistance zone, back = support zone.
    SHORT: ahead = support zone,    back = resistance zone.
    Volatility scale: atr_over_abs_price = atr / |close| (NaN when denom 0).
    """
    n = st.n_bars
    b = np.tile(np.arange(n), 2)
    is_long = np.concatenate([np.ones(n, bool), np.zeros(n, bool)])

    sup_top = st.sup_top[b]
    sup_bottom = st.sup_bottom[b]
    sup_strength = st.sup_strength[b]
    res_top = st.res_top[b]
    res_bottom = st.res_bottom[b]
    res_strength = st.res_strength[b]
    atr = st.atr[b]
    close = st.close[b]

    side = np.where(is_long, 1.0, -1.0)
    fav, adv = structural_barriers(is_long, sup_top, res_bottom)
    g, l, _elig = bracket_metrics(side, close, fav, adv, atr)
    log_rr = _log_rr(g, l)

    ahead_width = np.where(
        is_long, (res_top - res_bottom) / atr, (sup_top - sup_bottom) / atr)
    back_width = np.where(
        is_long, (sup_top - sup_bottom) / atr, (res_top - res_bottom) / atr)
    ahead_strength = np.where(is_long, res_strength, sup_strength)
    back_strength = np.where(is_long, sup_strength, res_strength)
    atr_price = np.where(np.abs(close) > 0, atr / np.abs(close), np.nan)

    x = np.column_stack([
        g, l, log_rr, ahead_width, back_width,
        ahead_strength, back_strength, atr_price,
    ]).astype(np.float32)
    assert x.shape == (2 * n, PAY8_N), x.shape
    df = pd.DataFrame(x, columns=list(PAY8_COLS))
    df.insert(0, "side", np.where(is_long, "LONG", "SHORT"))
    df.insert(0, "decision_bar", b)
    df.insert(0, "symbol", symbol)
    _bump("feature_builds")
    return df


# --------------------------------------------------------------------------- #
# 3. Materialization                                                          #
# --------------------------------------------------------------------------- #
def materialize_decomposed(symbols=SYMBOLS, split=None, verbose: bool = True,
                          save: bool = True, run_r8: bool = True):
    """Build artifacts/decomposed_value_v1/.

    §5: rerun the full R8 materialization first (provenance foundation), then
    derive WIN33 / PAY8 from the per-symbol State and re-save the canonical
    state / labels / renewal axis into the decomposed directory.
    """
    t0 = time.time()
    reset_counters()
    if split is None:
        split = build_frozen_split()
    if run_r8:
        r8_materialize(symbols=symbols, split=split, verbose=verbose, save=True)
        _bump("state_artifact_loads")

    # Load canonical R8 outputs.
    state_df = pd.read_parquet(STATE_PARQUET)
    renewal_axis_df = pd.read_parquet(RENEWAL_AXIS_PARQUET)
    label_parts = {s: pd.read_parquet(LABEL_PARQUETS[s]) for s in LABEL_PARQUETS}
    _bump("state_artifact_loads", 1 + len(label_parts) + 1)

    # Derive WIN33 / PAY8 per symbol from the raw State (X33 + zone geometry).
    win_frames, pay_frames = [], []
    for sym in symbols:
        st = load_symbol_state(sym)
        win_frames.append(build_win33_frame(st, sym))
        pay_frames.append(build_pay8_frame(st, sym))
        _bump("symbol_loops")
    win_df = pd.concat(win_frames, ignore_index=True)
    pay_df = pd.concat(pay_frames, ignore_index=True)

    # Hard contract checks.
    if not set(WIN33_COLS).isdisjoint(set(PAY8_COLS)):
        raise RuntimeError("STOP_DECOMPOSED_FEATURE_OVERLAP")
    if win_df.shape[1] != 3 + WIN33_N:
        raise RuntimeError("STOP_WIN33_COLUMN_COUNT")
    if pay_df.shape[1] != 3 + PAY8_N:
        raise RuntimeError("STOP_PAY8_COLUMN_COUNT")

    # Aggregate label split audit.
    label_df = pd.concat(label_parts.values(), ignore_index=True)
    split_audit = {s: int(len(v)) for s, v in label_parts.items()}
    per_horizon = {H: int((label_df["horizon"] == H).sum()) for H in HORIZONS}

    os.makedirs(DEC_ARTIFACT_DIR, exist_ok=True)
    if save:
        state_df.to_parquet(STATE_PARQUET_D, index=False)
        win_df.to_parquet(WIN_FEATURES_PARQUET, index=False)
        pay_df.to_parquet(PAYOFF_FEATURES_PARQUET, index=False)
        renewal_axis_df.to_parquet(RENEWAL_AXIS_PARQUET_D, index=False)
        for s, df in label_parts.items():
            df.to_parquet(LABEL_PARQUETS_D[s], index=False)

    artifact_sha = {}
    if save:
        for name, path in [
            ("state_v1.parquet", STATE_PARQUET_D),
            ("win_features_v1.parquet", WIN_FEATURES_PARQUET),
            ("payoff_features_v1.parquet", PAYOFF_FEATURES_PARQUET),
            ("renewal_event_axis_v1.parquet", RENEWAL_AXIS_PARQUET_D),
        ]:
            artifact_sha[name] = sha256_file(path)
        for s, p in LABEL_PARQUETS_D.items():
            artifact_sha[f"labels_{s}_v1.parquet"] = sha256_file(p)

    manifest = {
        "task_id": TASK_ID,
        "base_sha": BASE_SHA,
        "generator_code_sha": _git_head_sha(),
        "r8_upstream_manifest": os.path.relpath(
            os.path.join(ARTIFACT_DIR, "manifest_v1.json"),
            start=os.path.dirname(__file__)),
        "symbols": list(symbols),
        "win33_schema_sha256": _schema_sha256(WIN33_COLS),
        "pay8_schema_sha256": _schema_sha256(PAY8_COLS),
        "opp36_schema_sha256": opp36_schema_sha256(),
        "win33_n_features": WIN33_N,
        "pay8_n_features": PAY8_N,
        "win33_disjoint_pay8": True,
        "state_rows": int(len(state_df)),
        "win_feature_rows": int(len(win_df)),
        "payoff_feature_rows": int(len(pay_df)),
        "renewal_axis_rows": int(len(renewal_axis_df)),
        "label_rows": split_audit,
        "per_horizon_counts": per_horizon,
        "win33_feature_names": list(WIN33_COLS),
        "pay8_feature_names": list(PAY8_COLS),
        "split_audit": split_audit,
        "performance": dict(COUNTERS),
        "runtime_sec": time.time() - t0,
        "artifact_sha256": artifact_sha,
    }
    if save:
        with open(MANIFEST_JSON_D, "w") as f:
            json.dump(manifest, f, indent=2)
    if verbose:
        print(json.dumps({k: v for k, v in manifest.items()
                          if k != "artifact_sha256"}, indent=2, default=str))
    return manifest


# --------------------------------------------------------------------------- #
# 4. Loaders (consumed by R9A / R9B / R10)                                     #
# --------------------------------------------------------------------------- #
def load_win_features() -> pd.DataFrame:
    return pd.read_parquet(WIN_FEATURES_PARQUET)


def load_payoff_features() -> pd.DataFrame:
    return pd.read_parquet(PAYOFF_FEATURES_PARQUET)


def load_labels(stage: str) -> pd.DataFrame:
    if stage not in LABEL_PARQUETS_D:
        raise KeyError(stage)
    return pd.read_parquet(LABEL_PARQUETS_D[stage])


def load_renewal_axis() -> pd.DataFrame:
    return pd.read_parquet(RENEWAL_AXIS_PARQUET_D)


def load_state() -> pd.DataFrame:
    return pd.read_parquet(STATE_PARQUET_D)


def load_manifest() -> dict:
    with open(MANIFEST_JSON_D) as f:
        return json.load(f)


if __name__ == "__main__":
    materialize_decomposed(verbose=True)
