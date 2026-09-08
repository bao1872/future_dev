#!/usr/bin/env python3

"""RL-1A: Baseline + SMC x Risk/Reward relationship discovery.

Answers (A) what the unconditional FOLLOW/FADE x RR payoff surface
looks like, (B) whether an SMC state changes the payoff distribution
of the SAME action, and (C) whether SMC mainly selects direction,
participation, or RR.

Rules enforced here
-------------------
* Input authority is the finalized RL0 manifest + action Parquet SHA.
* ``decision_weight`` is used EXACTLY as stored; it is never
  renormalised inside a cell or recomputed.
* Valid-reward coverage is always reported against the whole cell, so
  a state with worse continuity cannot hide behind a smaller sample.
* Only pre-registered contrasts get a bootstrap CI; no p-values.
* SMC discovery runs on H12 only. H6/H24 are baseline diagnostics.
* No sorting by reward, no best/top/recommendation language.
* No MFE / MAE: V0 has none, and this module never fabricates one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASET_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_rl_dataset_v0"
)

OUT_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_rl_relationship_v1"
)

from research.audit_ob_rl_dataset_v0 import (  # noqa: E402
    sha256_file,
)
from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    VALIDATED_TFS,
)
from research.ob_rl_relationship_v1_spec import (  # noqa: E402
    RL1_VERSION,
    PRIMARY_HORIZON,
    BASELINE_HORIZONS,
    SMC_DISCOVERY_HORIZONS,
    BASELINE_VIEWS,
    SMC_VIEWS,
    BOOTSTRAP_REPS,
    BOOTSTRAP_SEED,
    BOOTSTRAP_CI,
    MIN_VALID_ROWS_FOR_CI,
    MIN_TRADING_DAYS_FOR_CI,
    CI_OK,
    CI_LOW_SUPPORT,
    FORBIDDEN_OUTPUT_TOKENS,
    OUT_OF_SCOPE_FAMILIES,
    CELL_COLUMNS,
    CONTRAST_COLUMNS,
    PAIRED_COLUMNS,
    smc_registry,
    bias_state,
    target_fit_state,
    stop_structure_state,
    forward_ob_state,
    assert_no_quarantined_tf,
)

EXIT_GAP_STOP = 2
EXIT_GAP_TARGET = 3
EXIT_BOTH = 4
EXIT_STOP = 5
EXIT_TARGET = 6
EXIT_TIMEOUT = 7

REWARD_COL = "reward"
WEIGHT_COL = "decision_weight"
DAY_COL = "trading_day"


# ============================================================
# Input authority
# ============================================================

def verify_rl0_input(root: Path, manifest: dict) -> dict:
    if manifest.get("rl0_finalized") is not True:
        raise RuntimeError("RL0 dataset not finalized")
    if manifest.get("model_view_version") != "ob_rl_model_view_v0":
        raise RuntimeError("unexpected model view")

    pq = root / "ob_rl_action_v0.parquet"
    if not pq.exists():
        raise RuntimeError("action parquet missing")

    expected = manifest["parquet"]["action"]["parquet_sha256"]
    actual = sha256_file(pq)
    if actual != expected:
        raise RuntimeError("RL0 action parquet SHA drift")

    if manifest["parquet"]["action"]["rows"] != 150_367:
        raise RuntimeError("action row count drift")

    if manifest.get("model_view_feature_count") != 62:
        raise RuntimeError("model view drift")

    return {
        "parquet_sha256": actual,
        "rows": 150_367,
    }


# ============================================================
# Weighted statistics
# ============================================================

def weighted_mean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return np.nan
    return float(np.sum(v[ok] * w[ok]) / np.sum(w[ok]))


def weighted_quantiles(
    values,
    weights,
    probs=(0.10, 0.25, 0.50, 0.75, 0.90),
):
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not ok.any():
        return [np.nan for _ in probs]

    v = v[ok]
    w = w[ok]

    order = np.argsort(v)
    v = v[order]
    w = w[order]

    cdf = (np.cumsum(w) - 0.5 * w) / np.sum(w)

    return np.interp(
        probs, cdf, v, left=v[0], right=v[-1]
    )


def weighted_rate(flag, weights) -> float:
    f = np.asarray(flag, dtype=float)
    w = np.asarray(weights, dtype=float)
    ok = np.isfinite(w) & (w > 0) & np.isfinite(f)
    if not ok.any():
        return np.nan
    return float(np.sum(f[ok] * w[ok]) / np.sum(w[ok]))


# ============================================================
# Cell summarization
# ============================================================

def summarize_cells(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    horizon: int,
) -> pd.DataFrame:
    reward_col = f"gross_R_h{horizon}"
    code_col = f"exit_code_h{horizon}"

    rows = []
    for keys, g in df.groupby(
        group_cols, dropna=False, observed=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))

        r = pd.to_numeric(
            g[reward_col], errors="coerce"
        ).to_numpy(float)
        w = pd.to_numeric(
            g[WEIGHT_COL], errors="coerce"
        ).to_numpy(float)
        c = pd.to_numeric(
            g[code_col], errors="coerce"
        ).to_numpy(float)

        finite = np.isfinite(r)

        base["rows_total"] = int(len(g))
        base["valid_n"] = int(finite.sum())
        base["valid_pct"] = (
            round(float(finite.mean()) * 100.0, 2)
            if len(g)
            else np.nan
        )
        base["weight_valid"] = float(
            np.nansum(w[finite])
        )
        base["trading_days"] = int(
            g.loc[finite, DAY_COL].nunique()
        )

        rv = r[finite]
        wv = w[finite]
        cv = c[finite]

        base["mean_R"] = weighted_mean(rv, wv)
        qs = weighted_quantiles(rv, wv)
        for name, q in zip(
            ("p10", "p25", "median", "p75", "p90"), qs
        ):
            base[name] = float(q)

        base["target_rate"] = weighted_rate(
            np.isin(cv, [EXIT_TARGET, EXIT_GAP_TARGET]), wv
        )
        # conservative same-bar policy: BOTH resolves as a stop loss
        base["stop_rate"] = weighted_rate(
            np.isin(
                cv, [EXIT_STOP, EXIT_GAP_STOP, EXIT_BOTH]
            ),
            wv,
        )
        base["timeout_rate"] = weighted_rate(
            np.isin(cv, [EXIT_TIMEOUT]), wv
        )
        base["both_rate"] = weighted_rate(
            np.isin(cv, [EXIT_BOTH]), wv
        )

        base["horizon"] = horizon
        rows.append(base)

    return pd.DataFrame(rows)


def finalize_cells(
    cells: pd.DataFrame, meta: dict
) -> pd.DataFrame:
    if cells.empty:
        return pd.DataFrame(columns=list(CELL_COLUMNS))
    out = cells.copy()
    for k, v in meta.items():
        out[k] = v
    for c in CELL_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(CELL_COLUMNS)]


# ============================================================
# Cluster bootstrap (trading day)
# ============================================================

def _daily_sums(x: pd.DataFrame, days: list) -> pd.DataFrame:
    y = x.copy()
    y["wr"] = (
        pd.to_numeric(y[WEIGHT_COL], errors="coerce")
        * pd.to_numeric(y[REWARD_COL], errors="coerce")
    )
    return (
        y.groupby(DAY_COL)
        .agg(
            sw=(WEIGHT_COL, "sum"),
            swr=("wr", "sum"),
        )
        .reindex(days, fill_value=0)
    )


def cluster_bootstrap_delta(
    a: pd.DataFrame,
    b: pd.DataFrame,
    *,
    reps: int = BOOTSTRAP_REPS,
    seed: int = BOOTSTRAP_SEED,
):
    """Trading-day cluster bootstrap of (mean A - mean B).

    Rows are never resampled independently: the same trading-day index
    is drawn for both arms, so the counterfactual pairing survives.
    """
    days = sorted(
        set(a[DAY_COL].astype(str))
        | set(b[DAY_COL].astype(str))
    )
    if len(days) < MIN_TRADING_DAYS_FOR_CI:
        return np.nan, np.nan, 0

    A = _daily_sums(a, days)
    B = _daily_sums(b, days)

    rng = np.random.default_rng(seed)
    sw_a = A["sw"].to_numpy(float)
    swr_a = A["swr"].to_numpy(float)
    sw_b = B["sw"].to_numpy(float)
    swr_b = B["swr"].to_numpy(float)

    n = len(days)
    out = []
    for _ in range(reps):
        idx = rng.integers(0, n, size=n)
        aw = sw_a[idx].sum()
        bw = sw_b[idx].sum()
        if aw <= 0 or bw <= 0:
            continue
        out.append(swr_a[idx].sum() / aw - swr_b[idx].sum() / bw)

    if not out:
        return np.nan, np.nan, 0

    lo, hi = np.quantile(out, BOOTSTRAP_CI)
    return float(lo), float(hi), len(out)


def _arm(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            REWARD_COL: pd.to_numeric(
                g[f"gross_R_h{horizon}"], errors="coerce"
            ),
            WEIGHT_COL: pd.to_numeric(
                g[WEIGHT_COL], errors="coerce"
            ),
            DAY_COL: g[DAY_COL].astype(str),
        }
    ).dropna(subset=[REWARD_COL])


# ============================================================
# Baseline + paired contrasts
# ============================================================

def build_baseline_cells(
    df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for view, vcols in BASELINE_VIEWS.items():
        for h in BASELINE_HORIZONS:
            cells = summarize_cells(
                df,
                group_cols=list(vcols)
                + ["trade_mode", "target_R"],
                horizon=h,
            )
            if cells.empty:
                continue
            frames.append(
                finalize_cells(
                    cells,
                    {
                        "analysis_family": "BASELINE",
                        "feature": "action",
                        "env_tf": "na",
                        "state_definition": "action",
                        "view": view,
                    },
                )
            )
    if not frames:
        return pd.DataFrame(columns=list(CELL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def build_paired_action(
    df: pd.DataFrame, *, horizon: int = PRIMARY_HORIZON
) -> pd.DataFrame:
    rc = f"gross_R_h{horizon}"
    # Candidate-level columns live on the LEFT arm only, so the view
    # columns are not duplicated into *_follow / *_fade.
    keep = [
        "candidate_id",
        "target_R",
        "symbol",
        "source_tf",
        "source_ob_structure",
        DAY_COL,
        WEIGHT_COL,
        rc,
    ]
    keep_right = [
        "candidate_id",
        "target_R",
        DAY_COL,
        WEIGHT_COL,
        rc,
    ]
    f = df[df["trade_mode"].astype(str) == "FOLLOW"][keep]
    d = df[df["trade_mode"].astype(str) == "FADE"][
        keep_right
    ]

    pair = f.merge(
        d,
        on=["candidate_id", "target_R"],
        suffixes=("_follow", "_fade"),
        validate="one_to_one",
    )
    pair["delta_R"] = pd.to_numeric(
        pair[f"{rc}_follow"], errors="coerce"
    ) - pd.to_numeric(
        pair[f"{rc}_fade"], errors="coerce"
    )
    return _summarize_pairs(
        pair,
        group_extra=["target_R"],
        horizon=horizon,
        pair_kind="FOLLOW_minus_FADE",
        trade_mode="paired",
        rc=rc,
    )


def build_paired_rr(
    df: pd.DataFrame, *, horizon: int = PRIMARY_HORIZON
) -> pd.DataFrame:
    rc = f"gross_R_h{horizon}"
    keep = [
        "candidate_id",
        "trade_mode",
        "symbol",
        "source_tf",
        "source_ob_structure",
        DAY_COL,
        WEIGHT_COL,
        rc,
    ]
    frames = []
    for lo, hi in ((1.5, 2.0), (2.0, 2.5)):
        keep_right = [
            "candidate_id",
            "trade_mode",
            DAY_COL,
            WEIGHT_COL,
            rc,
        ]
        a = df[
            np.isclose(
                pd.to_numeric(df["target_R"], errors="coerce"),
                lo,
            )
        ][keep]
        b = df[
            np.isclose(
                pd.to_numeric(df["target_R"], errors="coerce"),
                hi,
            )
        ][keep_right]
        pair = a.merge(
            b,
            on=["candidate_id", "trade_mode"],
            suffixes=("_lo", "_hi"),
            validate="one_to_one",
        )
        pair["delta_R"] = pd.to_numeric(
            pair[f"{rc}_hi"], errors="coerce"
        ) - pd.to_numeric(
            pair[f"{rc}_lo"], errors="coerce"
        )
        frames.append(
            _summarize_pairs(
                pair,
                group_extra=["trade_mode"],
                horizon=horizon,
                pair_kind=f"RR_{hi}_minus_{lo}",
                trade_mode=None,
                rr_low=lo,
                rr_high=hi,
                arm_lo=a,
                arm_hi=b,
                rc=rc,
            )
        )
    if not frames:
        return pd.DataFrame(columns=list(PAIRED_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def _pair_col(df: pd.DataFrame, base: str) -> str:
    """Resolve the left arm's column after a suffixed merge."""
    for suffix in ("_lo", "_follow"):
        if f"{base}{suffix}" in df.columns:
            return f"{base}{suffix}"
    if base in df.columns:
        return base
    raise RuntimeError(f"paired merge lost column {base}")


def _summarize_pairs(
    pair: pd.DataFrame,
    *,
    group_extra: list[str],
    horizon: int,
    pair_kind: str,
    trade_mode,
    rr_low=np.nan,
    rr_high=np.nan,
    arm_lo=None,
    arm_hi=None,
    rc=None,
) -> pd.DataFrame:
    """Pair-level delta distribution + trading-day cluster CI."""
    w_col = _pair_col(pair, WEIGHT_COL)
    d_col = _pair_col(pair, DAY_COL)

    valid = np.isfinite(
        pd.to_numeric(
            pair["delta_R"], errors="coerce"
        ).to_numpy(float)
    ) & np.isfinite(
        pd.to_numeric(
            pair[w_col], errors="coerce"
        ).to_numpy(float)
    )
    p = pair.loc[valid].copy()
    p[WEIGHT_COL] = pd.to_numeric(
        p[w_col], errors="coerce"
    )
    p[DAY_COL] = p[d_col].astype(str)

    frames = []
    for view, vcols in BASELINE_VIEWS.items():
        gcols = list(vcols) + group_extra
        for keys, g in p.groupby(
            gcols, dropna=False, observed=True
        ):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(gcols, keys))
            row["pair_kind"] = pair_kind
            row["view"] = view
            row["horizon"] = horizon
            row["rr_low"] = rr_low
            row["rr_high"] = rr_high
            if trade_mode is not None:
                row["trade_mode"] = trade_mode

            w = g[WEIGHT_COL].to_numpy(float)
            dv = pd.to_numeric(
                g["delta_R"], errors="coerce"
            ).to_numpy(float)

            row["n_pairs"] = int(len(g))
            row["weight_total"] = float(np.nansum(w))
            row["trading_days"] = int(g[DAY_COL].nunique())
            row["delta_mean_R"] = weighted_mean(dv, w)
            qs = weighted_quantiles(dv, w)
            for name, q in zip(
                (
                    "delta_p10",
                    "delta_p25",
                    "delta_median",
                    "delta_p75",
                    "delta_p90",
                ),
                qs,
            ):
                row[name] = float(q)

            if arm_lo is not None and arm_hi is not None:
                sub_lo = arm_lo[
                    arm_lo["candidate_id"]
                    .astype(str)
                    .isin(g["candidate_id"].astype(str))
                ]
                sub_hi = arm_hi[
                    arm_hi["candidate_id"]
                    .astype(str)
                    .isin(g["candidate_id"].astype(str))
                ]
                a_arm = _arm(sub_hi, horizon)
                b_arm = _arm(sub_lo, horizon)
            else:
                follow = pd.DataFrame(
                    {
                        REWARD_COL: pd.to_numeric(
                            g[f"{rc}_follow"], errors="coerce"
                        ),
                        WEIGHT_COL: w,
                        DAY_COL: g[DAY_COL].astype(str),
                    }
                ).dropna(subset=[REWARD_COL])
                fade = pd.DataFrame(
                    {
                        REWARD_COL: pd.to_numeric(
                            g[f"{rc}_fade"], errors="coerce"
                        ),
                        WEIGHT_COL: w,
                        DAY_COL: g[DAY_COL].astype(str),
                    }
                ).dropna(subset=[REWARD_COL])
                a_arm, b_arm = follow, fade

            if (
                len(a_arm) >= MIN_VALID_ROWS_FOR_CI
                and len(b_arm) >= MIN_VALID_ROWS_FOR_CI
            ):
                lo, hi, reps = cluster_bootstrap_delta(
                    a_arm, b_arm
                )
                row["ci_low"] = lo
                row["ci_high"] = hi
                row["bootstrap_valid_reps"] = reps
                row["ci_status"] = (
                    CI_OK if reps > 0 else CI_LOW_SUPPORT
                )
            else:
                row["ci_low"] = np.nan
                row["ci_high"] = np.nan
                row["bootstrap_valid_reps"] = 0
                row["ci_status"] = CI_LOW_SUPPORT

            frames.append(row)

    if not frames:
        return pd.DataFrame(columns=list(PAIRED_COLUMNS))
    out = pd.DataFrame(frames)
    for c in PAIRED_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(PAIRED_COLUMNS)]


# ============================================================
# SMC cells + contrasts
# ============================================================

def add_smc_states(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for (
        family,
        feature,
        tf,
        missing_label,
        _contrast,
        kind,
    ) in smc_registry():
        assert_no_quarantined_tf(tf)
        if kind == "bias":
            out[feature] = [
                bias_state(v) for v in out[feature]
            ]
        elif kind == "target_fit":
            out[feature] = [
                target_fit_state(v, missing_label=missing_label)
                for v in out[feature]
            ]
        elif kind == "stop_structure":
            out[feature] = [
                stop_structure_state(
                    v, missing_label=missing_label
                )
                for v in out[feature]
            ]
        elif kind == "forward_ob":
            sc = out[
                f"forward_active_ob_structure_class_{tf}"
            ]
            br = out[f"forward_active_ob_bias_rel_{tf}"]
            out[feature] = [
                forward_ob_state(a, b)
                for a, b in zip(sc, br)
            ]
        else:
            raise RuntimeError(f"unknown state kind {kind}")
    return out


def build_smc_cells(
    df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []
    for (
        family,
        feature,
        tf,
        _missing,
        _contrast,
        kind,
    ) in smc_registry():
        for h in SMC_DISCOVERY_HORIZONS:
            for view, vcols in SMC_VIEWS.items():
                cells = summarize_cells(
                    df,
                    group_cols=list(vcols)
                    + [
                        "trade_mode",
                        "target_R",
                        feature,
                    ],
                    horizon=h,
                )
                if cells.empty:
                    continue
                cells = cells.rename(
                    columns={feature: "state"}
                )
                frames.append(
                    finalize_cells(
                        cells,
                        {
                            "analysis_family": family,
                            "feature": feature,
                            "env_tf": tf,
                            "state_definition": kind,
                            "view": view,
                        },
                    )
                )
    if not frames:
        return pd.DataFrame(columns=list(CELL_COLUMNS))
    return pd.concat(frames, ignore_index=True)


def build_smc_contrasts(
    df: pd.DataFrame, *, horizon: int = PRIMARY_HORIZON
) -> pd.DataFrame:
    rows = []
    for (
        family,
        feature,
        tf,
        _missing,
        contrast,
        kind,
    ) in smc_registry():
        if contrast is None:
            continue
        hi_state, lo_state = contrast

        for view, vcols in SMC_VIEWS.items():
            gcols = list(vcols) + ["trade_mode", "target_R"]
            for keys, g in df.groupby(
                gcols, dropna=False, observed=True
            ):
                if not isinstance(keys, tuple):
                    keys = (keys,)
                base = dict(zip(gcols, keys))

                a = g[g[feature].astype(str) == hi_state]
                b = g[g[feature].astype(str) == lo_state]

                row = dict(base)
                row.update(
                    {
                        "analysis_family": family,
                        "feature": feature,
                        "env_tf": tf,
                        "state_definition": kind,
                        "view": view,
                        "horizon": horizon,
                        "contrast": f"{hi_state}_minus_{lo_state}",
                        "state_high": hi_state,
                        "state_low": lo_state,
                        "n_high": int(len(a)),
                        "n_low": int(len(b)),
                    }
                )

                arm_a = _arm(a, horizon)
                arm_b = _arm(b, horizon)
                if len(arm_a) == 0 or len(arm_b) == 0:
                    row["delta_mean_R"] = np.nan
                    row["ci_low"] = np.nan
                    row["ci_high"] = np.nan
                    row["bootstrap_valid_reps"] = 0
                    row["ci_status"] = CI_LOW_SUPPORT
                    rows.append(row)
                    continue

                row["delta_mean_R"] = weighted_mean(
                    arm_a[REWARD_COL], arm_a[WEIGHT_COL]
                ) - weighted_mean(
                    arm_b[REWARD_COL], arm_b[WEIGHT_COL]
                )

                if (
                    len(arm_a) >= MIN_VALID_ROWS_FOR_CI
                    and len(arm_b) >= MIN_VALID_ROWS_FOR_CI
                ):
                    lo, hi, reps = cluster_bootstrap_delta(
                        arm_a, arm_b
                    )
                    row["ci_low"] = lo
                    row["ci_high"] = hi
                    row["bootstrap_valid_reps"] = reps
                    row["ci_status"] = (
                        CI_OK if reps > 0 else CI_LOW_SUPPORT
                    )
                else:
                    row["ci_low"] = np.nan
                    row["ci_high"] = np.nan
                    row["bootstrap_valid_reps"] = 0
                    row["ci_status"] = CI_LOW_SUPPORT

                rows.append(row)

    if not rows:
        return pd.DataFrame(columns=list(CONTRAST_COLUMNS))
    out = pd.DataFrame(rows)
    for c in CONTRAST_COLUMNS:
        if c not in out.columns:
            out[c] = np.nan
    return out[list(CONTRAST_COLUMNS)]


# ============================================================
# Guards
# ============================================================

def guard_outputs(root: Path) -> None:
    for p in sorted(root.glob("*.csv")):
        txt = p.read_text(encoding="utf-8").lower()
        hits = [
            t
            for t in FORBIDDEN_OUTPUT_TOKENS
            if t in txt
        ]
        if hits:
            raise RuntimeError(
                f"{p.name} contains ranking language: {hits}"
            )


def assert_no_duplicate_key(df: pd.DataFrame) -> None:
    if df.duplicated(["candidate_id", "action"]).any():
        raise RuntimeError("duplicate candidate-action row")


# ============================================================
# Main (Gate B only)
# ============================================================

def read_columns() -> list[str]:
    cols = [
        "candidate_id",
        "symbol",
        "source_tf",
        "source_ob_structure",
        DAY_COL,
        "action",
        "trade_mode",
        "target_R",
        WEIGHT_COL,
        "gross_R_h6",
        "gross_R_h12",
        "gross_R_h24",
        "exit_code_h6",
        "exit_code_h12",
        "exit_code_h24",
    ]
    for (
        _family,
        feature,
        tf,
        _m,
        _c,
        kind,
    ) in smc_registry():
        if kind == "forward_ob":
            cols += [
                f"forward_active_ob_structure_class_{tf}",
                f"forward_active_ob_bias_rel_{tf}",
            ]
        else:
            cols.append(feature)
    return list(dict.fromkeys(cols))


def main() -> None:
    root = DATASET_ROOT
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    manifest = json.loads(
        (root / "dataset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    input_proof = verify_rl0_input(root, manifest)

    df = pd.read_parquet(
        root / "ob_rl_action_v0.parquet",
        columns=read_columns(),
    )
    assert_no_duplicate_key(df)

    # SKIP has reward 0 by construction; it is not a tradeable arm
    # and would only dilute every cell.
    trades = df[
        df["trade_mode"].astype(str) != "SKIP"
    ].copy()

    trades = add_smc_states(trades)

    baseline = build_baseline_cells(trades)
    paired_action = build_paired_action(trades)
    paired_rr = build_paired_rr(trades)
    smc_cells = build_smc_cells(trades)
    smc_contrasts = build_smc_contrasts(trades)

    baseline.to_csv(
        OUT_ROOT / "baseline_cells.csv", index=False
    )
    paired_action.to_csv(
        OUT_ROOT / "paired_action_contrasts.csv", index=False
    )
    paired_rr.to_csv(
        OUT_ROOT / "paired_rr_contrasts.csv", index=False
    )
    smc_cells.to_csv(
        OUT_ROOT / "smc_cells.csv", index=False
    )
    smc_contrasts.to_csv(
        OUT_ROOT / "smc_contrasts.csv", index=False
    )

    guard_outputs(OUT_ROOT)

    rl1_manifest = {
        "rl1_version": RL1_VERSION,
        "scope": {
            "baseline": True,
            "smc_rr": True,
            "dsa": False,
            "momentum": False,
            "quantile_scan": False,
            "interactions": False,
            "ml": False,
        },
        "out_of_scope_families": list(
            OUT_OF_SCOPE_FAMILIES
        ),
        "input_proof": input_proof,
        "horizons": {
            "primary": PRIMARY_HORIZON,
            "baseline": list(BASELINE_HORIZONS),
            "smc_discovery": list(
                SMC_DISCOVERY_HORIZONS
            ),
        },
        "statistics": {
            "weight": "decision_weight as stored "
            "(1 / group_candidate_count), never renormalised",
            "ci": "trading-day cluster bootstrap, "
            f"{BOOTSTRAP_REPS} reps, no p-values",
            "seed": BOOTSTRAP_SEED,
            "min_valid_rows_for_ci": (
                MIN_VALID_ROWS_FOR_CI
            ),
            "min_trading_days_for_ci": (
                MIN_TRADING_DAYS_FOR_CI
            ),
        },
        "registry_features": len(smc_registry()),
        "rows": {
            "baseline_cells": int(len(baseline)),
            "paired_action": int(len(paired_action)),
            "paired_rr": int(len(paired_rr)),
            "smc_cells": int(len(smc_cells)),
            "smc_contrasts": int(len(smc_contrasts)),
        },
    }
    (OUT_ROOT / "rl1_manifest.json").write_text(
        json.dumps(
            rl1_manifest, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )

    readme = f"""# RL-1A - Baseline + SMC x Risk/Reward

Scope: action baseline and SMC relationship discovery only.
DSA, Momentum, Quantile scans, interactions and ML are out of scope.

- Reward: gross_R_h6 / h12 / h24 (GROSS_R_V0). No MFE / MAE in V0.
- SMC discovery horizon: H12 only. H6 / H24 are baseline diagnostics.
- Weight: `decision_weight` used exactly as stored.
- CI: trading-day cluster bootstrap, {BOOTSTRAP_REPS} reps,
  pre-registered contrasts only, no p-values.

Outputs:
- baseline_cells.csv
- paired_action_contrasts.csv  (FOLLOW - FADE, same candidate+RR)
- paired_rr_contrasts.csv      (2.0-1.5, 2.5-2.0, same candidate+mode)
- smc_cells.csv
- smc_contrasts.csv

Cells are ordered by registry / timeframe / action / state, never by
reward. This round reports payoff distributions, not recommendations.
"""
    (OUT_ROOT / "README.md").write_text(
        readme, encoding="utf-8"
    )

    print("RL1_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
