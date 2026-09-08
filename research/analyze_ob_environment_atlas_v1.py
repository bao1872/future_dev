#!/usr/bin/env python3

"""Analyse the OB Environment Atlas V1 payoff distributions.

This module answers:

    given an OB touch, what does the FUTURE PAYOFF DISTRIBUTION look
    like under each complete multi-timeframe environment?

It is deliberately NOT a strategy leaderboard. There is no best/rank/
optimize/select logic anywhere in this module. Outcomes are reported
as full distributions (terminal / MFE / MAE quantiles plus tail
probabilities) so that no single stop-target setting can mislabel an
environment.

Levels
------
    Level 0  baseline (source timeframe)
    Level 1  one environment family at a time
    Level 2  pre-registered family PAIRS only
             (full cartesian product is forbidden)

Every cell passes a sample-size gate before any distribution is
emitted; under-sized cells report coverage only.
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

ATLAS_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_environment_atlas_v1"
)
PHASE1 = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_candidate_v3_phase1"
)

from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    weighted_mean,
    weighted_quantile,
)
from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
    load_raw_five,
)
from research.environment_atlas_spec import (  # noqa: E402
    ATLAS_VERSION,
    ATLAS_BASELINE_SHA,
    VALIDATED_TFS,
    QUARANTINED_TFS,
    HORIZONS,
    DIST_BINS,
    MIN_RAW_N,
    MIN_WEIGHTED_N,
    MIN_TRADING_DAYS,
    ROBUST_RAW_N,
    ROBUST_TRADING_DAYS,
    STATUS_INSUFFICIENT,
    STATUS_EXPLORATORY,
    STATUS_ROBUST,
    LEVEL2_INTERACTIONS,
    FOUR_HOUR_AUTHORITY,
    SOURCE_OWNERS,
    assert_validated_tf,
)

WEIGHT_COL = "decision_weight"

# MFE / MAE tail thresholds reported per (direction, horizon).
MFE_THRESHOLDS = (1.0, 2.0, 3.0)
MAE_THRESHOLDS = (0.5, 1.0, 1.5)

DIRECTIONS = ("follow", "fade")


def value_columns() -> list[str]:
    cols = []
    for h in HORIZONS:
        for d in DIRECTIONS:
            for k in ("terminal_R_atr", "mfe_atr", "mae_atr"):
                cols.append(f"{d}_h{h}_{k}")
    return cols


def weighted_distribution(
    values: np.ndarray,
    weights: np.ndarray,
) -> dict | None:
    ok = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    v = values[ok]
    w = weights[ok]
    if len(v) == 0:
        return None
    return {
        "n": int(len(v)),
        "n_weighted": float(w.sum()),
        "mean": weighted_mean(v, w),
        "p10": weighted_quantile(v, 0.10, w),
        "p25": weighted_quantile(v, 0.25, w),
        "median": weighted_quantile(v, 0.50, w),
        "p75": weighted_quantile(v, 0.75, w),
        "p90": weighted_quantile(v, 0.90, w),
    }


def tail_probability(
    values: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> float:
    ok = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights > 0)
    )
    v = values[ok]
    w = weights[ok]
    if len(v) == 0:
        return np.nan
    hit = (v >= threshold).astype(float)
    denom = float(w.sum())
    if denom <= 0:
        return np.nan
    return float(np.sum(w * hit) / denom)


def summarize_cells(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    value_cols: list[str],
    weight_col: str = WEIGHT_COL,
) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(
        group_cols, dropna=False, observed=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(g)
        nw = float(g[weight_col].sum())
        days = int(g["trading_day"].nunique())

        base = dict(zip(group_cols, keys))
        base.update(
            {"n": n, "n_weighted": nw, "trading_days": days}
        )

        if (
            n < MIN_RAW_N
            or nw < MIN_WEIGHTED_N
            or days < MIN_TRADING_DAYS
        ):
            base["status"] = STATUS_INSUFFICIENT
            rows.append(base)
            continue

        base["status"] = (
            STATUS_ROBUST
            if n >= ROBUST_RAW_N and days >= ROBUST_TRADING_DAYS
            else STATUS_EXPLORATORY
        )

        w = g[weight_col].to_numpy(float)
        for col in value_cols:
            st = weighted_distribution(
                g[col].to_numpy(float), w
            )
            if st is None:
                continue
            for k, v in st.items():
                if k in ("n", "n_weighted"):
                    continue
                base[f"{col}_{k}"] = v

        for h in HORIZONS:
            for d in DIRECTIONS:
                mfe = g[f"{d}_h{h}_mfe_atr"].to_numpy(float)
                mae = g[f"{d}_h{h}_mae_atr"].to_numpy(float)
                for t in MFE_THRESHOLDS:
                    base[
                        f"{d}_h{h}_P_mfe_ge_{t:g}atr"
                    ] = tail_probability(mfe, w, t)
                for t in MAE_THRESHOLDS:
                    base[
                        f"{d}_h{h}_P_mae_ge_{t:g}atr"
                    ] = tail_probability(mae, w, t)

        rows.append(base)

    return pd.DataFrame(rows)


# ============================================================
# Analysis frame assembly
# ============================================================

def pivot_env(
    env_tf: pd.DataFrame,
    fields: tuple[str, ...],
) -> pd.DataFrame:
    sub = env_tf[
        env_tf["context_tf"].astype(str).isin(VALIDATED_TFS)
    ]
    piv = sub.pivot(
        index="candidate_id",
        columns="context_tf",
        values=list(fields),
    )
    piv.columns = [
        f"{a}_{b}" for a, b in piv.columns
    ]
    return piv.reset_index()


def _joint(row_values: list) -> str:
    parts = []
    for v in row_values:
        if v is None or not np.isfinite(v):
            parts.append("NA")
        else:
            parts.append(f"{int(v):+d}")
    return "|".join(parts)


def _merge_checked(
    left: pd.DataFrame,
    right: pd.DataFrame,
    name: str,
    on: str = "candidate_id",
) -> pd.DataFrame:
    """Merge that refuses to silently create _x/_y suffixed columns."""
    dup = (set(right.columns) & set(left.columns)) - {on}
    if dup:
        raise RuntimeError(
            f"{name}: column collision with analysis frame "
            f"(would create _x/_y): {sorted(dup)[:8]}"
        )
    return left.merge(right, on=on, how="left")


def build_analysis_frame(
    candidates: pd.DataFrame,
    env_tf: pd.DataFrame,
    level_map: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    df = candidates.copy()
    df[WEIGHT_COL] = (
        1.0 / df["group_candidate_count"].to_numpy(float)
    )

    df = _merge_checked(df, outcomes, "outcomes")

    env_w = pivot_env(
        env_tf,
        (
            "dsa_direction",
            "internal_bias",
            "swing_bias",
            "momentum_direction",
            "momentum_change",
            "volatility_phase",
        ),
    )
    df = _merge_checked(df, env_w, "env_tf")

    lm = level_map.pivot(
        index="candidate_id",
        columns="context_tf",
        values=[
            "above_nearest_exec_atr",
            "below_nearest_exec_atr",
            "above_count_within_1atr",
            "below_count_within_1atr",
            "above_object_count",
            "below_object_count",
        ],
    )
    lm.columns = [f"{a}_{b}" for a, b in lm.columns]
    df = _merge_checked(
        df, lm.reset_index(), "level_map"
    )

    # ---- environment joint states (relative to the OB direction) ----
    bias = df["source_ob_bias"].to_numpy(float)

    def rel(col_tf: str) -> np.ndarray:
        return df[col_tf].to_numpy(float) * bias

    dsa_cols = [f"dsa_direction_{tf}" for tf in VALIDATED_TFS]
    mom_dir_cols = [
        f"momentum_direction_{tf}" for tf in VALIDATED_TFS
    ]
    mom_chg_cols = [
        f"momentum_change_{tf}" for tf in VALIDATED_TFS
    ]
    vol_cols = [
        f"volatility_phase_{tf}" for tf in VALIDATED_TFS
    ]

    dsa_rel = {
        tf: rel(f"dsa_direction_{tf}") for tf in VALIDATED_TFS
    }
    # joint order: 1h | 15m | 5m
    order = ("1h", "15m", "5m")
    df["dsa_joint"] = [
        _joint([dsa_rel[tf][i] for tf in order])
        for i in range(len(df))
    ]
    df["momentum_dir_joint"] = [
        _joint(
            [
                df[f"momentum_direction_{tf}"].to_numpy(float)[i]
                * bias[i]
                for tf in order
            ]
        )
        for i in range(len(df))
    ]
    df["momentum_change_joint"] = [
        "|".join(
            str(df[f"momentum_change_{tf}"].to_numpy()[i])
            for tf in order
        )
        for i in range(len(df))
    ]
    df["volatility_phase_joint"] = [
        "|".join(
            str(df[f"volatility_phase_{tf}"].to_numpy()[i])
            for tf in order
        )
        for i in range(len(df))
    ]

    df["smc_internal_joint"] = [
        _joint(
            [
                df[f"internal_bias_{tf}"].to_numpy(float)[i]
                * bias[i]
                for tf in order
            ]
        )
        for i in range(len(df))
    ]
    df["smc_swing_joint"] = [
        _joint(
            [
                df[f"swing_bias_{tf}"].to_numpy(float)[i]
                * bias[i]
                for tf in order
            ]
        )
        for i in range(len(df))
    ]

    def _num(col: str) -> np.ndarray:
        return pd.to_numeric(df[col], errors="coerce").to_numpy(
            float
        )

    def support_count(cols: list[str]) -> np.ndarray:
        M = np.column_stack(
            [_num(c) * bias for c in cols]
        )
        return np.nansum((M == 1).astype(float), axis=1).astype(
            int
        )

    df["dsa_support_count"] = support_count(dsa_cols)
    df["momentum_dir_support_count"] = support_count(
        mom_dir_cols
    )
    df["smc_internal_support_count"] = support_count(
        [f"internal_bias_{tf}" for tf in VALIDATED_TFS]
    )
    df["smc_swing_support_count"] = support_count(
        [f"swing_bias_{tf}" for tf in VALIDATED_TFS]
    )

    has5 = df["group_has_5m"].to_numpy(bool)
    has15 = df["group_has_15m"].to_numpy(bool)
    has1 = df["group_has_1h"].to_numpy(bool)
    clab = np.full(len(df), "other", dtype=object)
    clab[has5 & ~has15 & ~has1] = "only_5m"
    clab[~has5 & has15 & ~has1] = "only_15m"
    clab[~has5 & ~has15 & has1] = "only_1h"
    clab[has5 & has15 & ~has1] = "5m+15m"
    clab[has5 & ~has15 & has1] = "5m+1h"
    clab[~has5 & has15 & has1] = "15m+1h"
    clab[has5 & has15 & has1] = "5m+15m+1h"
    df["confluence_label"] = clab

    # ---- pressure / support (direction aware, per TF) ----
    for tf in VALIDATED_TFS:
        up = df[f"above_nearest_exec_atr_{tf}"].to_numpy(float)
        dn = df[f"below_nearest_exec_atr_{tf}"].to_numpy(float)
        room = np.where(bias == 1, up, dn)
        df[f"forward_room_{tf}"] = room
        df[f"forward_room_class_{tf}"] = np.select(
            [room < 1.0, room < 2.0, room >= 2.0],
            ["<1ATR", "1-2ATR", ">=2ATR"],
            default="UNKNOWN",
        )
        df[f"pressure_density_{tf}"] = df[
            f"above_count_within_1atr_{tf}"
        ].to_numpy(float)
        df[f"support_density_{tf}"] = df[
            f"below_count_within_1atr_{tf}"
        ].to_numpy(float)

    room_stack = np.column_stack(
        [df[f"forward_room_{tf}"].to_numpy(float) for tf in VALIDATED_TFS]
    )
    with np.errstate(invalid="ignore"):
        rmin = np.nanmin(
            np.where(np.isfinite(room_stack), room_stack, np.nan),
            axis=1,
        )
    df["forward_room_min_atr"] = np.where(
        np.all(~np.isfinite(room_stack), axis=1), np.nan, rmin
    )
    df["forward_room_class_min"] = np.select(
        [
            df["forward_room_min_atr"] < 1.0,
            df["forward_room_min_atr"] < 2.0,
            df["forward_room_min_atr"] >= 2.0,
        ],
        ["<1ATR", "1-2ATR", ">=2ATR"],
        default="UNKNOWN",
    )

    dens = np.nansum(
        np.column_stack(
            [
                df[f"pressure_density_{tf}"].to_numpy(float)
                for tf in VALIDATED_TFS
            ]
        ),
        axis=1,
    )
    df["pressure_density_multitf"] = dens
    df["pressure_density_class"] = np.select(
        [dens <= 1, dens <= 3, dens > 3],
        ["LOW", "MID", "HIGH"],
        default="UNKNOWN",
    )

    return df


# ============================================================
# Main (Gate B only)
# ============================================================

# Pre-registered Level-2 group-column mapping.
# Coarse support counts are used so cells stay large enough to pass the
# sample-size gate. A full cartesian product is never attempted.
L2_GROUP_COLS = {
    ("DSA", "MOMENTUM"): (
        "dsa_support_count",
        "momentum_dir_support_count",
    ),
    ("DSA", "LEVELS"): (
        "dsa_support_count",
        "forward_room_class_min",
    ),
    ("MOMENTUM", "LEVELS"): (
        "momentum_dir_support_count",
        "forward_room_class_min",
    ),
    ("SMC", "DSA"): (
        "smc_internal_support_count",
        "dsa_support_count",
    ),
    ("SMC", "MOMENTUM"): (
        "smc_internal_support_count",
        "momentum_dir_support_count",
    ),
    ("TOUCH", "DSA"): ("touch_behavior", "dsa_support_count"),
    ("TOUCH", "MOMENTUM"): (
        "touch_behavior",
        "momentum_dir_support_count",
    ),
    ("TOUCH", "LEVELS"): (
        "touch_behavior",
        "forward_room_class_min",
    ),
    ("QUANTILE", "MOMENTUM"): (
        "quant_bin",
        "momentum_dir_support_count",
    ),
    ("QUANTILE", "LEVELS"): (
        "quant_bin",
        "forward_room_class_min",
    ),
}


def main() -> None:
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = load_full_or_chunks("candidates")
    ann = pd.read_csv(PHASE1 / "candidate_annotations.csv")
    candidates = candidates.merge(
        ann,
        on=["candidate_id", "symbol", "source_tf"],
        how="left",
        validate="one_to_one",
    )

    symbols = sorted(candidates["symbol"].unique().tolist())
    raw_five = {s: load_raw_five(s) for s in symbols}
    parts = []
    for s in symbols:
        five = raw_five[s]
        idx = (
            candidates.loc[
                candidates["symbol"] == s,
                "touch_5m_bar_index",
            ]
            .astype(int)
            .to_numpy()
        )
        x = candidates[candidates["symbol"] == s].copy()
        x["trading_day"] = (
            five.iloc[idx]["trading_day"].astype(str).to_numpy()
        )
        parts.append(x)
    candidates = pd.concat(parts, ignore_index=True)

    env_tf = pd.read_csv(ATLAS_ROOT / "candidate_env_tf.csv")
    level_map = pd.read_csv(
        ATLAS_ROOT / "candidate_level_map.csv"
    )
    outcomes = pd.read_csv(
        ATLAS_ROOT / "candidate_outcomes.csv"
    )

    df = build_analysis_frame(
        candidates, env_tf, level_map, outcomes
    )
    vcol = value_columns()
    stats: dict = {}

    def emit(name: str, specs) -> int:
        frames = []
        for facet, gcols in specs:
            t = summarize_cells(
                df, group_cols=list(gcols), value_cols=vcol
            )
            if t.empty:
                continue
            t.insert(0, "facet", facet)
            frames.append(t)
        if not frames:
            return 0
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(ATLAS_ROOT / name, index=False)
        stats[name] = {
            "rows": int(len(out)),
            "robust_cells": int(
                (out["status"] == STATUS_ROBUST).sum()
            ),
            "exploratory_cells": int(
                (out["status"] == STATUS_EXPLORATORY).sum()
            ),
            "insufficient_cells": int(
                (out["status"] == STATUS_INSUFFICIENT).sum()
            ),
        }
        print(name, len(out), flush=True)
        return len(out)

    emit(
        "baseline_distribution.csv",
        [
            ("source_tf", ("source_tf",)),
            ("symbol_x_source_tf", ("symbol", "source_tf")),
        ],
    )
    emit(
        "structure_distribution.csv",
        [
            ("smc_internal_joint", ("smc_internal_joint",)),
            ("smc_swing_joint", ("smc_swing_joint",)),
            (
                "smc_internal_support_count",
                ("smc_internal_support_count",),
            ),
        ],
    )
    emit(
        "dsa_distribution.csv",
        [
            ("dsa_joint", ("dsa_joint",)),
            ("dsa_support_count", ("dsa_support_count",)),
        ],
    )
    emit(
        "momentum_distribution.csv",
        [
            ("momentum_dir_joint", ("momentum_dir_joint",)),
            (
                "momentum_change_joint",
                ("momentum_change_joint",),
            ),
            (
                "volatility_phase_joint",
                ("volatility_phase_joint",),
            ),
            (
                "momentum_dir_support_count",
                ("momentum_dir_support_count",),
            ),
        ],
    )
    emit(
        "level_distribution.csv",
        [
            ("forward_room_min", ("forward_room_class_min",)),
            ("forward_room_5m", ("forward_room_class_5m",)),
            ("forward_room_15m", ("forward_room_class_15m",)),
            ("forward_room_1h", ("forward_room_class_1h",)),
            ("pressure_density", ("pressure_density_class",)),
        ],
    )
    emit(
        "quantile_distribution.csv",
        [("quant_bin", ("quant_bin",))],
    )
    emit(
        "touch_distribution.csv",
        [
            ("touch_behavior", ("touch_behavior",)),
            ("touch_bin", ("touch_bin",)),
            ("confluence", ("confluence_label",)),
        ],
    )

    for pair in LEVEL2_INTERACTIONS:
        gcols = L2_GROUP_COLS.get(pair)
        if gcols is None:
            continue
        fname = (
            "interaction_"
            + pair[0].lower()
            + "_"
            + pair[1].lower()
            + ".csv"
        )
        emit(fname, [("x".join(pair), gcols)])

    summ = {
        "atlas_version": ATLAS_VERSION,
        "baseline_sha": ATLAS_BASELINE_SHA,
        "candidates": int(len(df)),
        "timeframes": list(VALIDATED_TFS),
        "quarantined": list(QUARANTINED_TFS),
        "four_hour_authority": FOUR_HOUR_AUTHORITY,
        "source_owners": SOURCE_OWNERS,
        "gates": {
            "min_raw_n": MIN_RAW_N,
            "min_weighted_n": MIN_WEIGHTED_N,
            "min_trading_days": MIN_TRADING_DAYS,
            "robust_raw_n": ROBUST_RAW_N,
            "robust_trading_days": ROBUST_TRADING_DAYS,
        },
        "tables": stats,
        "level2_interactions": [
            list(p) for p in LEVEL2_INTERACTIONS
        ],
    }
    (ATLAS_ROOT / "atlas_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("ATLAS_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
