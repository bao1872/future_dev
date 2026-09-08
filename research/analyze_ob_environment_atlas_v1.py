#!/usr/bin/env python3

"""Analyse the OB Environment Atlas V1 payoff distributions.

Architecture
------------
The environment mother table is DIRECTION-NEUTRAL (absolute). The
analyzer expands it into

    candidate x trade_mode  (follow / fade)

and only THEN converts environment to "relative to the ACTUAL trade
direction". This is essential: a fade trade lives in the mirror image of
the follow environment, so encoding environment once relative to the OB
direction would silently mislabel every fade row.

Outcomes are reported as full distributions per horizon, with a
sample-size gate evaluated SEPARATELY for each horizon (H24 strict
continuity is ~19%, so a cell that is ROBUST at H12 can be
INSUFFICIENT at H24).

There is no best / rank / optimize / select logic in this module.
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
    MIN_RAW_N,
    MIN_WEIGHTED_N,
    MIN_TRADING_DAYS,
    ROBUST_RAW_N,
    ROBUST_TRADING_DAYS,
    STATUS_INSUFFICIENT,
    STATUS_EXPLORATORY,
    STATUS_ROBUST,
    LEVEL2_INTERACTIONS,
    LEVEL2_TF_EXPANDED,
    FOUR_HOUR_AUTHORITY,
    SOURCE_OWNERS,
    assert_validated_tf,
)

WEIGHT_COL = "decision_weight"

MFE_THRESHOLDS = (1.0, 2.0, 3.0)
MAE_THRESHOLDS = (0.5, 1.0, 1.5)

TRADE_MODES = ("follow", "fade")

# Canonical momentum strings are NEVER cast to float.
MOMENTUM_STRING_FIELDS = (
    "momentum_volatility_phase",
    "momentum_momentum_direction",
    "momentum_momentum_change",
)

LEVEL_MAP_SIDE_FIELDS = (
    "above_nearest_exec_atr",
    "below_nearest_exec_atr",
    "above_nearest_tf_atr",
    "below_nearest_tf_atr",
    "above_count_within_1atr",
    "below_count_within_1atr",
    "above_object_count",
    "below_object_count",
    "above_active_bull_ob_count",
    "above_active_bear_ob_count",
    "below_active_bull_ob_count",
    "below_active_bear_ob_count",
    "above_nearest_active_ob_exec_atr",
    "above_nearest_internal_pivot_exec_atr",
    "above_nearest_swing_pivot_exec_atr",
    "above_nearest_equal_level_exec_atr",
    "below_nearest_active_ob_exec_atr",
    "below_nearest_internal_pivot_exec_atr",
    "below_nearest_swing_pivot_exec_atr",
    "below_nearest_equal_level_exec_atr",
)


# ============================================================
# Distribution helpers
# ============================================================

def weighted_distribution(
    values: np.ndarray, weights: np.ndarray
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
    values: np.ndarray, weights: np.ndarray, threshold: float
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
    denom = float(w.sum())
    if denom <= 0:
        return np.nan
    return float(np.sum(w * (v >= threshold).astype(float)) / denom)


def outcome_gate(
    g: pd.DataFrame, *, horizon: int
) -> dict:
    """Sample-size gate for ONE horizon, on that horizon's valid rows."""
    col = f"h{horizon}_terminal_atr"
    if col not in g.columns:
        raise RuntimeError(f"missing outcome column {col}")

    valid = np.isfinite(g[col].to_numpy(float))
    gv = g.loc[valid]

    n = int(len(gv))
    nw = float(gv[WEIGHT_COL].sum()) if n else 0.0
    days = int(gv["trading_day"].nunique()) if n else 0

    if (
        n < MIN_RAW_N
        or nw < MIN_WEIGHTED_N
        or days < MIN_TRADING_DAYS
    ):
        status = STATUS_INSUFFICIENT
    elif n >= ROBUST_RAW_N and days >= ROBUST_TRADING_DAYS:
        status = STATUS_ROBUST
    else:
        status = STATUS_EXPLORATORY

    return {
        "valid_n": n,
        "valid_n_weighted": nw,
        "valid_trading_days": days,
        "status": status,
    }


def summarize_cells(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    weight_col: str = WEIGHT_COL,
) -> pd.DataFrame:
    rows = []
    for keys, g in df.groupby(
        group_cols, dropna=False, observed=True
    ):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        base["cell_n"] = int(len(g))

        for h in HORIZONS:
            gate = outcome_gate(g, horizon=h)
            base[f"h{h}_status"] = gate["status"]
            base[f"h{h}_valid_n"] = gate["valid_n"]
            base[f"h{h}_valid_n_weighted"] = gate[
                "valid_n_weighted"
            ]
            base[f"h{h}_valid_trading_days"] = gate[
                "valid_trading_days"
            ]

            if gate["status"] == STATUS_INSUFFICIENT:
                continue

            col = f"h{h}_terminal_atr"
            valid = np.isfinite(g[col].to_numpy(float))
            gv = g.loc[valid]
            w = gv[weight_col].to_numpy(float)

            st = weighted_distribution(
                gv[col].to_numpy(float), w
            )
            if st:
                for k in ("mean", "p10", "p25", "median", "p75", "p90"):
                    base[f"h{h}_terminal_{k}"] = st[k]

            mfe = gv[f"h{h}_mfe_atr"].to_numpy(float)
            mae = gv[f"h{h}_mae_atr"].to_numpy(float)
            for k in ("mean", "median", "p75", "p90"):
                m = weighted_distribution(mfe, w)
                a = weighted_distribution(mae, w)
                base[f"h{h}_mfe_{k}"] = m[k] if m else np.nan
                base[f"h{h}_mae_{k}"] = a[k] if a else np.nan

            for t in MFE_THRESHOLDS:
                base[f"h{h}_P_mfe_ge_{t:g}atr"] = (
                    tail_probability(mfe, w, t)
                )
            for t in MAE_THRESHOLDS:
                base[f"h{h}_P_mae_ge_{t:g}atr"] = (
                    tail_probability(mae, w, t)
                )

        rows.append(base)
    return pd.DataFrame(rows)


# ============================================================
# Frame assembly
# ============================================================

def pivot_env(
    env_tf: pd.DataFrame, fields: tuple[str, ...]
) -> pd.DataFrame:
    sub = env_tf[
        env_tf["context_tf"].astype(str).isin(VALIDATED_TFS)
    ]
    piv = sub.pivot(
        index="candidate_id",
        columns="context_tf",
        values=list(fields),
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return piv.reset_index()


def _joint(vals: list) -> str:
    parts = []
    for v in vals:
        if v is None or not np.isfinite(v):
            parts.append("NA")
        else:
            parts.append(f"{int(v):+d}")
    return "|".join(parts)


def _str_joint(values: list) -> str:
    return "|".join("NA" if v is None else str(v) for v in values)


def _merge_checked(
    left: pd.DataFrame,
    right: pd.DataFrame,
    name: str,
    on: str = "candidate_id",
) -> pd.DataFrame:
    dup = (set(right.columns) & set(left.columns)) - {on}
    if dup:
        raise RuntimeError(
            f"{name}: column collision (would create _x/_y): "
            f"{sorted(dup)[:8]}"
        )
    return left.merge(right, on=on, how="left")


def build_analysis_frame(
    candidates: pd.DataFrame,
    env_tf: pd.DataFrame,
    level_map: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Direction-NEUTRAL absolute environment frame (candidate grain)."""
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
            "momentum_sqzmom_sign",
            "momentum_momentum_direction",
            "momentum_momentum_change",
            "momentum_volatility_phase",
        ),
    )
    df = _merge_checked(df, env_w, "env_tf")

    lm_fields = [
        f
        for f in LEVEL_MAP_SIDE_FIELDS
        if f in level_map.columns
    ]
    lm = level_map.pivot(
        index="candidate_id",
        columns="context_tf",
        values=lm_fields,
    )
    lm.columns = [f"{a}_{b}" for a, b in lm.columns]
    df = _merge_checked(df, lm.reset_index(), "level_map")

    order = ("1h", "15m", "5m")

    # Canonical momentum STRINGS: studied as-is, never multiplied.
    for f in MOMENTUM_STRING_FIELDS:
        df[f"{f}_joint"] = [
            _str_joint(
                [df[f"{f}_{tf}"].to_numpy()[i] for tf in order]
            )
            for i in range(len(df))
        ]

    return df


def build_directional_frame(base: pd.DataFrame) -> pd.DataFrame:
    """Expand candidate -> candidate x trade_mode.

    Environment becomes relative to the ACTUAL trade direction here.
    """
    frames = []
    for mode, mult in (("follow", 1), ("fade", -1)):
        x = base.copy()
        x["trade_mode"] = mode
        x["trade_direction"] = (
            x["source_ob_bias"].to_numpy(float) * mult
        )
        td = x["trade_direction"].to_numpy(float)

        for tf in VALIDATED_TFS:
            assert_validated_tf(tf)

            x[f"dsa_rel_{tf}"] = (
                x[f"dsa_direction_{tf}"].to_numpy(float) * td
            )
            x[f"momentum_rel_{tf}"] = (
                x[f"momentum_sqzmom_sign_{tf}"].to_numpy(float)
                * td
            )
            x[f"smc_internal_rel_{tf}"] = (
                x[f"internal_bias_{tf}"].to_numpy(float) * td
            )
            x[f"smc_swing_rel_{tf}"] = (
                x[f"swing_bias_{tf}"].to_numpy(float) * td
            )

            above = x[f"above_nearest_exec_atr_{tf}"].to_numpy(
                float
            )
            below = x[f"below_nearest_exec_atr_{tf}"].to_numpy(
                float
            )
            x[f"forward_nearest_exec_atr_{tf}"] = np.where(
                td == 1, above, below
            )
            x[f"backward_nearest_exec_atr_{tf}"] = np.where(
                td == 1, below, above
            )

            a_n = x[f"above_count_within_1atr_{tf}"].to_numpy(
                float
            )
            b_n = x[f"below_count_within_1atr_{tf}"].to_numpy(
                float
            )
            x[f"forward_density_1atr_{tf}"] = np.where(
                td == 1, a_n, b_n
            )
            x[f"backward_density_1atr_{tf}"] = np.where(
                td == 1, b_n, a_n
            )

            fr = x[f"forward_nearest_exec_atr_{tf}"].to_numpy(
                float
            )
            x[f"forward_room_class_{tf}"] = np.select(
                [fr < 1.0, fr < 2.0, fr >= 2.0],
                ["<1ATR", "1-2ATR", ">=2ATR"],
                default="UNKNOWN",
            )
            fd = x[f"forward_density_1atr_{tf}"].to_numpy(float)
            x[f"forward_density_class_{tf}"] = np.select(
                [fd <= 0, fd <= 2, fd > 2],
                ["LOW", "MID", "HIGH"],
                default="UNKNOWN",
            )

        # Derived multi-TF descriptor (allowed, but Level-2 must not
        # depend on it alone).
        room_stack = np.column_stack(
            [
                x[f"forward_nearest_exec_atr_{tf}"].to_numpy(
                    float
                )
                for tf in VALIDATED_TFS
            ]
        )
        with np.errstate(invalid="ignore"):
            rmin = np.nanmin(
                np.where(
                    np.isfinite(room_stack), room_stack, np.nan
                ),
                axis=1,
            )
        x["forward_room_min_atr"] = np.where(
            np.all(~np.isfinite(room_stack), axis=1), np.nan, rmin
        )
        x["forward_room_class_min"] = np.select(
            [
                x["forward_room_min_atr"] < 1.0,
                x["forward_room_min_atr"] < 2.0,
                x["forward_room_min_atr"] >= 2.0,
            ],
            ["<1ATR", "1-2ATR", ">=2ATR"],
            default="UNKNOWN",
        )

        order = ("1h", "15m", "5m")
        x["dsa_joint_rel"] = [
            _joint([x[f"dsa_rel_{tf}"].to_numpy()[i] for tf in order])
            for i in range(len(x))
        ]
        x["momentum_rel_joint"] = [
            _joint(
                [
                    x[f"momentum_rel_{tf}"].to_numpy()[i]
                    for tf in order
                ]
            )
            for i in range(len(x))
        ]
        x["smc_internal_joint_rel"] = [
            _joint(
                [
                    x[f"smc_internal_rel_{tf}"].to_numpy()[i]
                    for tf in order
                ]
            )
            for i in range(len(x))
        ]
        x["smc_swing_joint_rel"] = [
            _joint(
                [
                    x[f"smc_swing_rel_{tf}"].to_numpy()[i]
                    for tf in order
                ]
            )
            for i in range(len(x))
        ]

        def support_count(cols: list[str]) -> np.ndarray:
            M = np.column_stack(
                [
                    pd.to_numeric(x[c], errors="coerce")
                    .to_numpy(float)
                    for c in cols
                ]
            )
            return np.nansum((M == 1).astype(float), axis=1)

        x["dsa_support_count"] = support_count(
            [f"dsa_rel_{tf}" for tf in VALIDATED_TFS]
        ).astype(int)
        x["momentum_support_count"] = support_count(
            [f"momentum_rel_{tf}" for tf in VALIDATED_TFS]
        ).astype(int)
        x["smc_internal_support_count"] = support_count(
            [f"smc_internal_rel_{tf}" for tf in VALIDATED_TFS]
        ).astype(int)

        for h in HORIZONS:
            x[f"h{h}_terminal_atr"] = x[
                f"{mode}_h{h}_terminal_R_atr"
            ]
            x[f"h{h}_mfe_atr"] = x[f"{mode}_h{h}_mfe_atr"]
            x[f"h{h}_mae_atr"] = x[f"{mode}_h{h}_mae_atr"]

        frames.append(x)

    return pd.concat(frames, ignore_index=True)


# ============================================================
# Main (Gate B only)
# ============================================================

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

    base = build_analysis_frame(
        candidates, env_tf, level_map, outcomes
    )
    df = build_directional_frame(base)

    stats: dict = {}

    def emit(name: str, specs) -> int:
        frames = []
        for facet, gcols in specs:
            t = summarize_cells(df, group_cols=list(gcols))
            if t.empty:
                continue
            t.insert(0, "facet", facet)
            frames.append(t)
        if not frames:
            return 0
        out = pd.concat(frames, ignore_index=True)
        out.to_csv(ATLAS_ROOT / name, index=False)

        robust = {
            f"h{h}_robust_cells": int(
                (out[f"h{h}_status"] == STATUS_ROBUST).sum()
            )
            for h in HORIZONS
            if f"h{h}_status" in out.columns
        }
        stats[name] = {"rows": int(len(out)), **robust}
        print(name, len(out), flush=True)
        return len(out)

    # ---------------- Level 0 ----------------
    emit(
        "baseline_distribution.csv",
        [
            ("source_tf", ("trade_mode", "source_tf")),
            (
                "symbol_x_source_tf",
                ("symbol", "trade_mode", "source_tf"),
            ),
        ],
    )

    # ---------------- Level 1 ----------------
    emit(
        "structure_distribution.csv",
        [
            (
                "smc_internal_joint_rel",
                ("trade_mode", "smc_internal_joint_rel"),
            ),
            (
                "smc_swing_joint_rel",
                ("trade_mode", "smc_swing_joint_rel"),
            ),
            (
                "smc_internal_support_count",
                ("trade_mode", "smc_internal_support_count"),
            ),
        ],
    )
    emit(
        "dsa_distribution.csv",
        [
            ("dsa_joint_rel", ("trade_mode", "dsa_joint_rel")),
            (
                "dsa_support_count",
                ("trade_mode", "dsa_support_count"),
            ),
        ],
    )
    emit(
        "momentum_distribution.csv",
        [
            (
                "momentum_rel_joint",
                ("trade_mode", "momentum_rel_joint"),
            ),
            (
                "momentum_support_count",
                ("trade_mode", "momentum_support_count"),
            ),
            (
                "momentum_direction_canonical",
                (
                    "trade_mode",
                    "momentum_momentum_direction_joint",
                ),
            ),
            (
                "momentum_change_canonical",
                (
                    "trade_mode",
                    "momentum_momentum_change_joint",
                ),
            ),
            (
                "volatility_phase_canonical",
                (
                    "trade_mode",
                    "momentum_volatility_phase_joint",
                ),
            ),
        ],
    )
    emit(
        "level_distribution.csv",
        [
            (
                f"forward_room_{tf}",
                ("trade_mode", f"forward_room_class_{tf}"),
            )
            for tf in VALIDATED_TFS
        ]
        + [
            (
                f"forward_density_{tf}",
                ("trade_mode", f"forward_density_class_{tf}"),
            )
            for tf in VALIDATED_TFS
        ]
        + [
            (
                "forward_room_min_derived",
                ("trade_mode", "forward_room_class_min"),
            )
        ],
    )
    emit(
        "quantile_distribution.csv",
        [("quant_bin", ("trade_mode", "quant_bin"))],
    )
    emit(
        "touch_distribution.csv",
        [
            ("touch_behavior", ("trade_mode", "touch_behavior")),
            ("touch_bin", ("trade_mode", "touch_bin")),
            ("confluence", ("trade_mode", "confluence_label")),
        ],
    )

    # ---------------- Level 2 (pre-registered pairs) ----------------
    for pair in LEVEL2_INTERACTIONS:
        a, b = pair
        if pair in LEVEL2_TF_EXPANDED:
            # LEVELS side expanded per timeframe: TF identity kept.
            for tf in VALIDATED_TFS:
                emit(
                    f"interaction_{a.lower()}_{b.lower()}_{tf}.csv",
                    [
                        (
                            f"{a}_x_{b}_{tf}_room",
                            (
                                "trade_mode",
                                f"{a.lower()}_joint_rel",
                                f"forward_room_class_{tf}",
                            ),
                        ),
                        (
                            f"{a}_x_{b}_{tf}_density",
                            (
                                "trade_mode",
                                f"{a.lower()}_joint_rel",
                                f"forward_density_class_{tf}",
                            ),
                        ),
                    ],
                )
            continue

        colmap = {
            ("DSA", "MOMENTUM"): (
                "dsa_joint_rel",
                "momentum_rel_joint",
            ),
            ("SMC", "DSA"): (
                "smc_internal_joint_rel",
                "dsa_joint_rel",
            ),
            ("SMC", "MOMENTUM"): (
                "smc_internal_joint_rel",
                "momentum_rel_joint",
            ),
            ("TOUCH", "DSA"): ("touch_behavior", "dsa_joint_rel"),
            ("TOUCH", "MOMENTUM"): (
                "touch_behavior",
                "momentum_rel_joint",
            ),
            ("QUANTILE", "MOMENTUM"): (
                "quant_bin",
                "momentum_rel_joint",
            ),
        }
        gc = colmap.get(pair)
        if gc is None:
            continue
        emit(
            f"interaction_{a.lower()}_{b.lower()}.csv",
            [("x".join(pair), ("trade_mode",) + gc)],
        )

    summ = {
        "atlas_version": ATLAS_VERSION,
        "baseline_sha": ATLAS_BASELINE_SHA,
        "candidates": int(len(candidates)),
        "directional_rows": int(len(df)),
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
            "per_horizon": True,
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
