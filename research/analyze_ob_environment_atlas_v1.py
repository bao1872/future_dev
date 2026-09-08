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

source_tf is the CANDIDATE GENERATOR IDENTITY, not an ordinary feature.
A 5m OB touch and a 15m OB touch are different generators that can even
carry opposite mechanisms, so every environment distribution is emitted
conditioned on source_tf first. Pooled versions exist but are
explicitly secondary.

Outcomes are reported as full distributions per horizon, with a
sample-size gate evaluated SEPARATELY for each horizon (H24 strict
continuity is ~19%, so a cell that is ROBUST at H12 can be
INSUFFICIENT at H24).

The Atlas studies a PRE-REGISTERED projection of the environment
(DSA running state, Momentum, Pressure/Support, Structure, Quantile,
Volatility) fixed before any payoff is inspected. Continuous variables
are layered by their own descriptive quintiles, never by hand-picked
thresholds.

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
    DIST_BINS,
    density_field,
    DESCRIPTIVE_Q_BINS,
    DESCRIPTIVE_Q_LABELS,
    DESCRIPTIVE_Q_GROUP,
    LEVEL_TYPE_BUCKETS,
    ACTIVE_OB_COUNT_FIELDS,
    DSA_PROJECTIONS,
    MOMENTUM_TF_PROJECTIONS,
    STRUCTURE_PROJECTIONS,
    VOLATILITY_PROJECTIONS,
    ENV_PIVOT_FIELDS,
    LEVEL2_INTERACTIONS,
    LEVEL2_TF_EXPANDED,
    LEVEL2_STATE_AXIS,
    LEVEL2_LEVEL_STATE_AXIS,
    assert_level2_registry,
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


def level_map_side_fields() -> tuple[str, ...]:
    """Level-map columns carried into the directional frame.

    All four ATR distance bins are carried, so multi-layer
    pressure/support density is genuinely analysed instead of being
    collapsed to a single 1ATR count.
    """
    out: list[str] = []
    for side in ("above", "below"):
        out += [
            f"{side}_nearest_exec_atr",
            f"{side}_nearest_tf_atr",
            f"{side}_object_count",
            f"{side}_active_bull_ob_count",
            f"{side}_active_bear_ob_count",
        ]
        out += [density_field(side, b) for b in DIST_BINS]
        out += [
            f"{side}_nearest_{t}_exec_atr"
            for t in LEVEL_TYPE_BUCKETS
        ]
    out += ["overlap_object_count", "overlap_active_ob_count"]
    return tuple(out)


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
# Environment projection helpers
# ============================================================

def descriptive_quintile(
    df: pd.DataFrame, col: str
) -> pd.Series:
    """Descriptive environment layering by the variable's OWN
    distribution, ranked within (symbol, source_tf, trade_mode).

    This is a description of the environment, made before any payoff
    is inspected. It is NOT a strategy threshold.
    """
    pct = (
        df.groupby(
            list(DESCRIPTIVE_Q_GROUP), observed=True
        )[col]
        .rank(pct=True, method="average")
    )
    return pd.cut(
        pct,
        bins=list(DESCRIPTIVE_Q_BINS),
        labels=list(DESCRIPTIVE_Q_LABELS),
    )


def count_class(values: pd.Series) -> pd.Series:
    """Counts (objects, OBs) become an explicit ordinal class."""
    v = pd.to_numeric(values, errors="coerce")
    out = np.select(
        [v.isna(), v <= 0, v <= 1, v <= 2],
        ["NA", "0", "1", "2"],
        default="3+",
    )
    return pd.Series(out, index=v.index)


def bool_class(values: pd.Series) -> pd.Series:
    s = values
    if s.dtype == bool:
        b = s.to_numpy(bool)
        na = np.zeros(len(s), dtype=bool)
    else:
        t = s.astype("object").where(s.notna(), None)
        na = np.array([v is None for v in t])
        b = np.array(
            [
                str(v).strip().lower()
                in ("true", "1", "yes")
                for v in t
            ]
        )
    return pd.Series(
        np.where(na, "NA", np.where(b, "TRUE", "FALSE")),
        index=s.index,
    )


def project_environment_column(
    df: pd.DataFrame,
    *,
    src_col: str,
    kind: str,
    td: np.ndarray,
) -> pd.Series:
    """One pre-registered environment projection."""
    if kind == "categorical":
        s = df[src_col]
        return pd.Series(
            np.where(s.isna(), "NA", s.astype(str)),
            index=df.index,
        )

    v = pd.to_numeric(
        df[src_col], errors="coerce"
    ).to_numpy(float)

    if kind in ("signed", "signed_cat"):
        v = v * td

    if kind in ("signed", "raw_q"):
        tmp = f"__proj_{src_col}"
        df[tmp] = v
        out = descriptive_quintile(df, tmp)
        df.drop(columns=[tmp], inplace=True)
        return pd.Series(out, index=df.index)

    if kind == "signed_cat":
        return pd.Series(
            np.where(np.isfinite(v), v, np.nan), index=df.index
        )

    raise RuntimeError(f"unknown projection kind {kind!r}")


def add_projection_facets(
    df: pd.DataFrame,
    prefix: str,
    projections: tuple[tuple[str, str, str], ...],
    td: np.ndarray,
) -> tuple[pd.DataFrame, list[tuple[str, tuple[str, ...]]]]:
    """Materialise one projection family for every validated TF."""
    specs: list[tuple[str, tuple[str, ...]]] = []
    for facet_suffix, src_suffix, kind in projections:
        for tf in VALIDATED_TFS:
            assert_validated_tf(tf)
            src = f"{src_suffix}_{tf}"
            if src not in df.columns:
                raise RuntimeError(
                    f"missing environment column {src}"
                )
            col = f"{prefix}_{facet_suffix}_{tf}"
            df[col] = project_environment_column(
                df, src_col=src, kind=kind, td=td
            )
            specs.append((col, (col,)))
    return df, specs


# ============================================================
# source_tf conditioning
# ============================================================

def source_conditioned_specs(
    facet: str,
    env_cols: tuple[str, ...],
) -> list[tuple[str, tuple[str, ...]]]:
    """Primary is ALWAYS conditioned on the candidate generator
    (source_tf); the pooled version is explicitly secondary."""
    return [
        (
            facet + "_by_source_tf",
            ("source_tf", "trade_mode", *env_cols),
        ),
        (
            facet + "_pooled",
            ("trade_mode", *env_cols),
        ),
    ]


def expand_specs(
    specs: list[tuple[str, tuple[str, ...]]],
) -> list[tuple[str, tuple[str, ...]]]:
    out: list[tuple[str, tuple[str, ...]]] = []
    for facet, cols in specs:
        out.extend(source_conditioned_specs(facet, tuple(cols)))
    return out


# ============================================================
# Frame assembly
# ============================================================

def pivot_env(
    env_tf: pd.DataFrame, fields: tuple[str, ...]
) -> pd.DataFrame:
    sub = env_tf[
        env_tf["context_tf"].astype(str).isin(VALIDATED_TFS)
    ]
    missing = [f for f in fields if f not in sub.columns]
    if missing:
        raise RuntimeError(
            "environment columns missing from env_tf: "
            f"{missing}"
        )
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


def add_event_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit event labels (multi-TF co-occurrence on the decision
    bar). Derived here so the analyzer never depends on a label that
    no generator produced."""
    out = df.copy()

    required = {
        "group_has_5m",
        "group_has_15m",
        "group_has_1h",
        "touch_behavior",
        "touch_bin",
        "quant_bin",
    }
    missing = required - set(out.columns)
    if missing:
        raise RuntimeError(
            "event label input missing: "
            f"{sorted(missing)}"
        )

    h5 = out["group_has_5m"].astype(bool)
    h15 = out["group_has_15m"].astype(bool)
    h1 = out["group_has_1h"].astype(bool)

    out["confluence_label"] = np.select(
        [
            h5 & ~h15 & ~h1,
            ~h5 & h15 & ~h1,
            ~h5 & ~h15 & h1,
            h5 & h15 & ~h1,
            h5 & ~h15 & h1,
            ~h5 & h15 & h1,
            h5 & h15 & h1,
        ],
        [
            "only_5m",
            "only_15m",
            "only_1h",
            "5m+15m",
            "5m+1h",
            "15m+1h",
            "5m+15m+1h",
        ],
        default="INVALID",
    )

    if (out["confluence_label"] == "INVALID").any():
        raise RuntimeError("invalid confluence state")

    return out


def build_analysis_frame(
    candidates: pd.DataFrame,
    env_tf: pd.DataFrame,
    level_map: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> pd.DataFrame:
    """Direction-NEUTRAL absolute environment frame (candidate grain)."""
    df = add_event_labels(candidates)
    df[WEIGHT_COL] = (
        1.0 / df["group_candidate_count"].to_numpy(float)
    )
    df = _merge_checked(df, outcomes, "outcomes")

    env_w = pivot_env(env_tf, ENV_PIVOT_FIELDS)
    df = _merge_checked(df, env_w, "env_tf")

    lm_fields = [
        f
        for f in level_map_side_fields()
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


def build_directional_frame(
    base: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, list]]:
    """Expand candidate -> candidate x trade_mode.

    Environment becomes relative to the ACTUAL trade direction here.
    Returns the frame plus the Level-1 facet registry so that main()
    cannot silently study a different projection list.
    """
    frames = []
    reg: dict[str, list[tuple[str, tuple[str, ...]]]] = {}

    for mode, mult in (("follow", 1), ("fade", -1)):
        x = base.copy()
        x["trade_mode"] = mode
        x["trade_direction"] = (
            x["source_ob_bias"].to_numpy(float) * mult
        )
        td = x["trade_direction"].to_numpy(float)

        dsa_rel_specs: list = []
        mom_rel_specs: list = []
        smc_rel_specs: list = []

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

            dsa_rel_specs.append(
                (f"dsa_direction_rel_{tf}", (f"dsa_rel_{tf}",))
            )
            mom_rel_specs.append(
                (
                    f"momentum_sign_rel_{tf}",
                    (f"momentum_rel_{tf}",),
                )
            )
            smc_rel_specs.append(
                (
                    f"smc_internal_rel_{tf}",
                    (f"smc_internal_rel_{tf}",),
                )
            )
            smc_rel_specs.append(
                (
                    f"smc_swing_rel_{tf}",
                    (f"smc_swing_rel_{tf}",),
                )
            )

            # ---- Pressure / support: forward vs backward ----
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

            for t in LEVEL_TYPE_BUCKETS:
                a = x[
                    f"above_nearest_{t}_exec_atr_{tf}"
                ].to_numpy(float)
                b = x[
                    f"below_nearest_{t}_exec_atr_{tf}"
                ].to_numpy(float)
                x[f"forward_nearest_{t}_exec_atr_{tf}"] = (
                    np.where(td == 1, a, b)
                )

            for b in DIST_BINS:
                a_n = x[
                    f"{density_field('above', b)}_{tf}"
                ].to_numpy(float)
                b_n = x[
                    f"{density_field('below', b)}_{tf}"
                ].to_numpy(float)
                x[f"forward_density_{b:g}atr_{tf}"] = np.where(
                    td == 1, a_n, b_n
                )
                x[f"backward_density_{b:g}atr_{tf}"] = np.where(
                    td == 1, b_n, a_n
                )

            for kind in ("bull", "bear"):
                a_c = x[
                    f"above_active_{kind}_ob_count_{tf}"
                ].to_numpy(float)
                b_c = x[
                    f"below_active_{kind}_ob_count_{tf}"
                ].to_numpy(float)
                x[f"forward_{kind}_ob_count_{tf}"] = np.where(
                    td == 1, a_c, b_c
                )
                x[f"backward_{kind}_ob_count_{tf}"] = np.where(
                    td == 1, b_c, a_c
                )

            # Room / density classes (used by Level-2 registry).
            fr = x[f"forward_nearest_exec_atr_{tf}"].to_numpy(
                float
            )
            x[f"forward_room_class_{tf}"] = np.select(
                [fr < 1.0, fr < 2.0, fr >= 2.0],
                ["<1ATR", "1-2ATR", ">=2ATR"],
                default="UNKNOWN",
            )
            fd = x[f"forward_density_{1.0:g}atr_{tf}"].to_numpy(
                float
            )
            x[f"forward_density_class_{tf}"] = np.select(
                [fd <= 0, fd <= 2, fd > 2],
                ["LOW", "MID", "HIGH"],
                default="UNKNOWN",
            )

        # ---- Pre-registered environment projections ----
        x, dsa_proj = add_projection_facets(
            x, "dsa", DSA_PROJECTIONS, td
        )
        x, mom_proj = add_projection_facets(
            x, "momentum", MOMENTUM_TF_PROJECTIONS, td
        )
        x, str_proj = add_projection_facets(
            x, "structure", STRUCTURE_PROJECTIONS, td
        )
        x, vol_proj = add_projection_facets(
            x, "volatility", VOLATILITY_PROJECTIONS, td
        )

        # ---- Levels facets (per TF, never collapsed to min) ----
        level_specs: list = []
        for tf in VALIDATED_TFS:
            for src in (
                f"forward_nearest_exec_atr_{tf}",
                f"backward_nearest_exec_atr_{tf}",
            ):
                q = f"{src}_q"
                x[q] = descriptive_quintile(x, src)
                level_specs.append((q, (q,)))

            for b in DIST_BINS:
                src = f"forward_density_{b:g}atr_{tf}"
                c = f"{src}_class"
                x[c] = count_class(x[src])
                level_specs.append((c, (c,)))

            for t in LEVEL_TYPE_BUCKETS:
                src = f"forward_nearest_{t}_exec_atr_{tf}"
                q = f"{src}_q"
                x[q] = descriptive_quintile(x, src)
                level_specs.append((q, (q,)))

            for nm in (
                "forward_bull_ob_count",
                "forward_bear_ob_count",
                "backward_bull_ob_count",
                "backward_bear_ob_count",
                "overlap_object_count",
                "overlap_active_ob_count",
            ):
                src = f"{nm}_{tf}"
                c = f"{src}_class"
                x[c] = count_class(x[src])
                level_specs.append((c, (c,)))

            level_specs.append(
                (
                    f"forward_room_class_{tf}",
                    (f"forward_room_class_{tf}",),
                )
            )

        # ---- Active OB counts (structure) ----
        active_specs: list = []
        for f in ACTIVE_OB_COUNT_FIELDS:
            for tf in VALIDATED_TFS:
                src = f"{f}_{tf}"
                if src not in x.columns:
                    raise RuntimeError(
                        f"missing active OB count column {src}"
                    )
                c = f"activeob_{f}_{tf}_class"
                x[c] = count_class(x[src])
                active_specs.append((c, (c,)))

        # ---- Quantile facets ----
        x["quant_width_pctile_q"] = descriptive_quintile(
            x, "quant_width_percentile_train"
        )
        x["quant_top30_class"] = bool_class(x["quant_top30_train"])
        x["quant_crossed_class"] = bool_class(x["quant_crossed"])
        qw = pd.to_numeric(
            x["quant_width"], errors="coerce"
        ).to_numpy(float)
        x["quant_known"] = np.where(
            np.isfinite(qw), "KNOWN", "UNKNOWN"
        )
        quant_specs = [
            ("quant_bin", ("quant_bin",)),
            (
                "quant_width_percentile",
                ("quant_width_pctile_q",),
            ),
            ("quant_top30", ("quant_top30_class",)),
            ("quant_crossed", ("quant_crossed_class",)),
            ("quant_coverage", ("quant_known",)),
        ]

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
        x["smc_swing_support_count"] = support_count(
            [f"smc_swing_rel_{tf}" for tf in VALIDATED_TFS]
        ).astype(int)

        for h in HORIZONS:
            x[f"h{h}_terminal_atr"] = x[
                f"{mode}_h{h}_terminal_R_atr"
            ]
            x[f"h{h}_mfe_atr"] = x[f"{mode}_h{h}_mfe_atr"]
            x[f"h{h}_mae_atr"] = x[f"{mode}_h{h}_mae_atr"]

        frames.append(x)

        reg = {
            "dsa": dsa_rel_specs
            + [
                ("dsa_joint_rel", ("dsa_joint_rel",)),
                ("dsa_support_count", ("dsa_support_count",)),
            ]
            + dsa_proj,
            "momentum": mom_rel_specs
            + [
                (
                    "momentum_rel_joint",
                    ("momentum_rel_joint",),
                ),
                (
                    "momentum_support_count",
                    ("momentum_support_count",),
                ),
                (
                    "momentum_direction_canonical",
                    (
                        "momentum_momentum_direction_joint",
                    ),
                ),
                (
                    "momentum_change_canonical",
                    ("momentum_momentum_change_joint",),
                ),
                (
                    "volatility_phase_canonical",
                    ("momentum_volatility_phase_joint",),
                ),
            ]
            + mom_proj,
            "structure": smc_rel_specs
            + [
                (
                    "smc_internal_joint_rel",
                    ("smc_internal_joint_rel",),
                ),
                (
                    "smc_swing_joint_rel",
                    ("smc_swing_joint_rel",),
                ),
                (
                    "smc_internal_support_count",
                    ("smc_internal_support_count",),
                ),
                (
                    "smc_swing_support_count",
                    ("smc_swing_support_count",),
                ),
            ]
            + str_proj
            + active_specs,
            "levels": level_specs,
            "quantile": quant_specs,
            "volatility": vol_proj,
        }

    return pd.concat(frames, ignore_index=True), reg


# ============================================================
# Level-2 registry runner
# ============================================================

def build_registered_interaction_specs(
    pair: tuple[str, str],
    tf: str | None = None,
) -> list[tuple[str, tuple[str, ...]]]:
    """Specs for ONE registered pair (optionally one timeframe).

    State columns come from the explicit registry -- nothing is
    derived by string guessing such as f"{a.lower()}_joint_rel".
    """
    a, b = pair
    if pair in LEVEL2_TF_EXPANDED:
        state_col = LEVEL2_LEVEL_STATE_AXIS[pair]
        if tf is None:
            raise RuntimeError(
                f"TF-expanded pair {pair} requires a timeframe"
            )
        assert_validated_tf(tf)
        return [
            (
                f"{a}_x_{b}_{tf}_room",
                (state_col, f"forward_room_class_{tf}"),
            ),
            (
                f"{a}_x_{b}_{tf}_density",
                (
                    state_col,
                    f"forward_density_class_{tf}",
                ),
            ),
        ]

    if tf is not None:
        raise RuntimeError(
            f"pair {pair} is not timeframe-expanded"
        )
    cols = LEVEL2_STATE_AXIS[pair]
    return [("x".join(pair), tuple(cols))]


def _check_spec_columns(
    df: pd.DataFrame,
    specs,
    pair: tuple[str, str],
) -> None:
    known = {"source_tf", "trade_mode"}
    missing = sorted(
        {c for _, gc in specs for c in gc}
        - set(df.columns)
        - known
    )
    if missing:
        raise RuntimeError(
            f"Level-2 {pair} references missing columns: "
            f"{missing}"
        )


def run_registered_interaction(
    df: pd.DataFrame,
    pair: tuple[str, str],
    emit,
) -> int:
    """Execute one registered Level-2 interaction.

    The synthetic test and the real run share this function, so the
    registry and the executed surface cannot drift apart. TF-expanded
    pairs emit one file per timeframe (12 files); the remaining pairs
    emit one file each (6 files) => 18 executed interactions.
    """
    a, b = pair
    total = 0

    if pair in LEVEL2_TF_EXPANDED:
        for tf in VALIDATED_TFS:
            specs = expand_specs(
                build_registered_interaction_specs(pair, tf)
            )
            _check_spec_columns(df, specs, pair)
            total += emit(
                f"interaction_{a.lower()}_{b.lower()}"
                f"_{tf}.csv",
                specs,
            )
        return total

    specs = expand_specs(
        build_registered_interaction_specs(pair)
    )
    _check_spec_columns(df, specs, pair)
    return emit(
        f"interaction_{a.lower()}_{b.lower()}.csv", specs
    )


# ============================================================
# Main (Gate B only)
# ============================================================

def main() -> None:
    assert_level2_registry()
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
    df, reg = build_directional_frame(base)

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
        stats[name] = {
            "rows": int(len(out)),
            "facets": int(out["facet"].nunique()),
            **robust,
        }
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
        expand_specs(reg["structure"]),
    )
    emit(
        "dsa_distribution.csv",
        expand_specs(reg["dsa"]),
    )
    emit(
        "momentum_distribution.csv",
        expand_specs(reg["momentum"]),
    )
    emit(
        "level_distribution.csv",
        expand_specs(reg["levels"]),
    )
    emit(
        "quantile_distribution.csv",
        expand_specs(reg["quantile"]),
    )
    emit(
        "volatility_distribution.csv",
        expand_specs(reg["volatility"]),
    )
    emit(
        "touch_distribution.csv",
        expand_specs(
            [
                ("touch_behavior", ("touch_behavior",)),
                ("touch_bin", ("touch_bin",)),
                ("confluence", ("confluence_label",)),
            ]
        ),
    )

    # ---------------- Level 2 (registry driven) ----------------
    level2_executed = []
    for pair in LEVEL2_INTERACTIONS:
        n = run_registered_interaction(df, pair, emit)
        level2_executed.append(
            {
                "pair": list(pair),
                "tf_expanded": bool(
                    pair in LEVEL2_TF_EXPANDED
                ),
                "rows": int(n),
            }
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
        "source_tf_conditioning": {
            "primary_suffix": "_by_source_tf",
            "secondary_suffix": "_pooled",
            "rule": (
                "every Level-1 and Level-2 facet is emitted "
                "conditioned on source_tf; pooled is secondary"
            ),
        },
        "projections": {
            "dsa_facets": len(reg["dsa"]),
            "momentum_facets": len(reg["momentum"]),
            "structure_facets": len(reg["structure"]),
            "levels_facets": len(reg["levels"]),
            "quantile_facets": len(reg["quantile"]),
            "volatility_facets": len(reg["volatility"]),
            "descriptive_layering": {
                "bins": list(DESCRIPTIVE_Q_BINS),
                "labels": list(DESCRIPTIVE_Q_LABELS),
                "ranked_within": list(DESCRIPTIVE_Q_GROUP),
                "note": (
                    "descriptive environment layering only; "
                    "never a strategy threshold"
                ),
            },
        },
        "tables": stats,
        "level2_interactions": [
            list(p) for p in LEVEL2_INTERACTIONS
        ],
        "level2_executed": level2_executed,
    }
    (ATLAS_ROOT / "atlas_summary.json").write_text(
        json.dumps(summ, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("ATLAS_ANALYZE_DONE", flush=True)


if __name__ == "__main__":
    main()
