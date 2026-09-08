#!/usr/bin/env python3

"""Build the OB RL Dataset V0 (Gate B component).

Produces two tables:

    ob_rl_state_v0.csv     one row per OB touch (ABSOLUTE state)
    ob_rl_action_v0.csv    candidate x 7 actions (action-relative
                           state + GROSS reward)

Hard rules
----------
* The STATE table stores ABSOLUTE market state only: above / below,
  high / low, bull / bear. Forward / backward / aligned / opposed is
  an ACTION-relative concept and exists only in the action table.
* All structural distances are expressed in 5m execution ATR, because
  stop and target are both defined in that unit.
* Momentum is consumed from the canonical source owner
  (``research.indicator_adapter.compute_smc_momentum_bundle``).
  SQZMOM is never reimplemented and ``momentum_direction`` keeps its
  canonical string semantics.
* Active-OB membership is an EXACT test against the frozen levels
  vocabulary, never a substring match.
* Reward reuses the validated Phase-1 first-hit simulator
  (build_path_arrays / rotated / simulate / apply_policy). No second
  simulator exists in this module.
* Reward is GROSS only. No commission / slippage / tick model exists,
  so no net R may be produced.
* 4h is quarantined: input rows are counted, output rows must be 0.
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

OUT_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_rl_dataset_v0"
)

from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
    load_raw_five,
)
from research.analyze_ob_candidate_v3_phase1 import (  # noqa: E402
    build_path_arrays,
    rotated,
    simulate,
    apply_policy,
)
from research.indicator_adapter import (  # noqa: E402
    compute_smc_momentum_bundle,
)
from research.build_pytdx_panel import aggregate_15m  # noqa: E402
from research.ob_trigger_snapshot import (  # noqa: E402
    aggregate_1h_from_15m,
)
from research.ob_rl_dataset_v0_spec import (  # noqa: E402
    DATASET_VERSION,
    BASELINE_SHA,
    VALIDATED_TFS,
    QUARANTINED_TFS,
    STOP_ATR,
    TARGET_R,
    ACTIONS,
    PRIMARY_HORIZON,
    DIAGNOSTIC_HORIZONS,
    SAME_BAR_POLICY,
    REWARD_VERSION,
    SKIP_EXIT_CODE,
    EVENT_FIELDS,
    SMC_BIAS_FIELDS,
    SMC_EVENT_FIELDS,
    SMC_PIVOT_DISTANCE_MAP,
    ACTIVE_OB_TYPES,
    LEVELS_REQUIRED_COLUMNS,
    DSA_FIELDS,
    DSA_RENAMES,
    MOMENTUM_FIELDS,
    MOMENTUM_JOIN_FIELD,
    QUANTILE_FIELDS,
    QUANT_LOW_MAX,
    QUANT_HIGH_MIN,
    FORBIDDEN_STATE_PREFIXES,
    FORBIDDEN_STATE_COLUMNS,
    assert_validated_tf,
    assert_level_vocabulary,
    assert_action_cardinality,
    parse_action,
)


# ============================================================
# Canonical momentum
# ============================================================

def build_momentum_frame(
    bars: pd.DataFrame,
) -> pd.DataFrame:
    """Canonical SQZMOM momentum state for one timeframe."""
    x = bars.copy()
    x = x.set_index(
        pd.DatetimeIndex(pd.to_datetime(x["bar_start_time"]))
    )
    x = x.sort_index()

    bundle = compute_smc_momentum_bundle(x)
    m = pd.DataFrame(
        bundle.momentum_history["daily_state"]
    )

    required = set(MOMENTUM_FIELDS) | {MOMENTUM_JOIN_FIELD}
    missing = required - set(m.columns)
    if missing:
        raise RuntimeError(
            f"momentum missing: {sorted(missing)}"
        )

    if not np.array_equal(
        m[MOMENTUM_JOIN_FIELD].to_numpy(int),
        np.arange(len(bars)),
    ):
        raise RuntimeError("momentum bar alignment failed")

    return m[
        [MOMENTUM_JOIN_FIELD, *MOMENTUM_FIELDS]
    ].copy()


def attach_momentum(
    ctx: pd.DataFrame,
    momentum_by_tf: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join canonical momentum on the causally-completed bar."""
    parts = []
    for tf, g in ctx.groupby("context_tf", sort=False):
        assert_validated_tf(tf)
        mom = momentum_by_tf[tf].set_index(
            MOMENTUM_JOIN_FIELD
        )
        idx = g["bar_index"].astype(int).to_numpy()
        missing = ~np.isin(idx, mom.index.to_numpy(int))
        if missing.any():
            raise RuntimeError(
                f"{tf}: {int(missing.sum())} context bars "
                "missing momentum"
            )
        x = g.copy()
        mm = mom.loc[idx]
        for c in MOMENTUM_FIELDS:
            x[c] = mm[c].to_numpy()
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


# ============================================================
# Geometry
# ============================================================

def pct_to_exec_atr(
    distance_pct,
    touch_close,
    atr5,
) -> np.ndarray:
    """distance_pct -> distance in 5m execution ATR units.

        distance_atr = distance_pct / 100 * touch_close / atr5

    ``touch_close`` / ``atr5`` may be scalars or per-row arrays.
    """
    d = pd.to_numeric(
        pd.Series(distance_pct).reset_index(drop=True)
        if not isinstance(distance_pct, pd.Series)
        else distance_pct,
        errors="coerce",
    ).to_numpy(float)

    t = np.asarray(touch_close, dtype=float).ravel()
    a = np.asarray(atr5, dtype=float).ravel()
    if t.size == 1:
        t = np.repeat(t, d.shape[0])
    if a.size == 1:
        a = np.repeat(a, d.shape[0])

    out = np.full(d.shape[0], np.nan)
    ok = np.isfinite(d) & np.isfinite(t) & np.isfinite(a) & (a > 0)
    out[ok] = d[ok] / 100.0 * t[ok] / a[ok]
    return out


def pivot_tf(
    df: pd.DataFrame,
    fields,
    *,
    index: str = "candidate_id",
) -> pd.DataFrame:
    missing = [f for f in fields if f not in df.columns]
    if missing:
        raise RuntimeError(
            f"cannot pivot, missing columns: {missing}"
        )
    piv = df.pivot(
        index=index,
        columns="context_tf",
        values=list(fields),
    )
    piv.columns = [f"{a}_{b}" for a, b in piv.columns]
    return piv.reset_index()


# ============================================================
# Active OB map (frozen vocabulary, exact membership)
# ============================================================

def build_active_ob_map(
    levels: pd.DataFrame,
    *,
    touch_close_map: dict[str, float],
    atr5_map: dict[str, float],
) -> pd.DataFrame:
    """Nearest ACTIVE OB above / below per (candidate, timeframe).

    Membership uses ACTIVE_OB_TYPES exactly. Nothing is inferred from
    a substring of object_type.
    """
    missing = [
        c for c in LEVELS_REQUIRED_COLUMNS
        if c not in levels.columns
    ]
    if missing:
        raise RuntimeError(
            f"levels missing columns: {missing}"
        )

    tf_series = levels["timeframe"].astype(str)
    quarantined_input_rows = int(
        tf_series.isin(QUARANTINED_TFS).sum()
    )
    x = levels[tf_series.isin(VALIDATED_TFS)].copy()

    x = x[
        x["object_type"].astype(str).isin(ACTIVE_OB_TYPES)
    ].copy()

    x["distance_pct"] = pd.to_numeric(
        x["distance_pct"], errors="coerce"
    )

    rows = []
    for (cid, tf), g in x.groupby(
        ["event_id", "timeframe"], sort=False
    ):
        cid = str(cid)
        tf = str(tf)
        assert_validated_tf(tf)

        row = {
            "candidate_id": cid,
            "context_tf": tf,
        }

        tc = touch_close_map.get(cid, np.nan)
        a5 = atr5_map.get(cid, np.nan)

        for side in ("above", "below"):
            z = g[g["relation"].astype(str).eq(side)]
            if z.empty:
                row[f"{side}_ob_distance_pct"] = np.nan
                row[f"{side}_ob_bias"] = np.nan
                row[f"{side}_ob_structure_class"] = None
                row[f"{side}_ob_zone_low"] = np.nan
                row[f"{side}_ob_zone_high"] = np.nan
                continue

            d = z["distance_pct"].to_numpy(float)
            ok = np.isfinite(d)
            if not ok.any():
                continue
            j = int(np.nanargmin(d))
            r = z.iloc[j]

            row[f"{side}_ob_distance_pct"] = float(d[j])
            row[f"{side}_ob_bias"] = pd.to_numeric(
                r["bias"], errors="coerce"
            )
            row[f"{side}_ob_structure_class"] = str(
                r["structure_class"]
            )
            row[f"{side}_ob_zone_low"] = pd.to_numeric(
                r["zone_low"], errors="coerce"
            )
            row[f"{side}_ob_zone_high"] = pd.to_numeric(
                r["zone_high"], errors="coerce"
            )

        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out, quarantined_input_rows

    for side in ("above", "below"):
        col = f"{side}_ob_distance_pct"
        if col not in out.columns:
            continue
        out[f"ob_{side}_atr"] = pct_to_exec_atr(
            out[col],
            out["candidate_id"].map(touch_close_map),
            out["candidate_id"].map(atr5_map),
        )
    return out, quarantined_input_rows


# ============================================================
# State table (ABSOLUTE)
# ============================================================

def touch_behavior(df: pd.DataFrame) -> pd.Series:
    return pd.Series(
        np.select(
            [
                df["touch_close_beyond_far_edge"].astype(bool),
                df["touch_intrabar_far_edge_breach"].astype(bool),
            ],
            ["CLOSE_BEYOND", "BREACH_RECLAIM"],
            default="NO_BREACH",
        ),
        index=df.index,
    )


def quant_state(p) -> str:
    if pd.isna(p):
        return "UNKNOWN"
    v = float(p)
    if v <= QUANT_LOW_MAX:
        return "LOW"
    if v >= QUANT_HIGH_MIN:
        return "HIGH"
    return "MID"


def build_state(
    candidates: pd.DataFrame,
    context: pd.DataFrame,
    levels: pd.DataFrame,
    momentum_by_tf: dict[tuple[str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    if candidates["candidate_id"].duplicated().any():
        raise RuntimeError("duplicate candidate_id in candidates")

    tf_series = context["context_tf"].astype(str)
    quarantined_ctx_input = int(
        tf_series.isin(QUARANTINED_TFS).sum()
    )
    ctx = context[tf_series.isin(VALIDATED_TFS)].copy()
    if ctx["context_tf"].astype(str).isin(QUARANTINED_TFS).any():
        raise RuntimeError("quarantined TF leaked into state")

    vocab = assert_level_vocabulary(levels)

    # --- price / ATR reference at the touch bar ---
    c5 = ctx[ctx["context_tf"].astype(str).eq("5m")]
    touch_close_map = (
        c5.set_index("candidate_id")["close"]
        .apply(pd.to_numeric, errors="coerce")
        .to_dict()
    )
    atr5_map = dict(
        zip(
            candidates["candidate_id"].astype(str),
            pd.to_numeric(
                candidates["5m_atr14"], errors="coerce"
            ).to_numpy(float),
        )
    )

    # --- momentum on the completed context bar ---
    mom_frames = []
    for symbol, g in ctx.groupby("symbol", sort=False):
        mom = {
            tf: momentum_by_tf[(symbol, tf)]
            for tf in VALIDATED_TFS
        }
        mom_frames.append(
            attach_momentum(
                g[["candidate_id", "symbol", "context_tf", "bar_index"]],
                mom,
            )
        )
    mom_df = pd.concat(mom_frames, ignore_index=True)

    # --- SMC / DSA pivot to per-TF columns ---
    smc_fields = (
        list(SMC_BIAS_FIELDS)
        + list(SMC_EVENT_FIELDS)
        + list(SMC_PIVOT_DISTANCE_MAP.keys())
        + list(DSA_FIELDS)
    )
    smc_piv = pivot_tf(ctx, smc_fields)

    mom_piv = pivot_tf(mom_df, list(MOMENTUM_FIELDS))

    # --- active OB map ---
    ob_map, quarantined_lv_input = build_active_ob_map(
        levels,
        touch_close_map=touch_close_map,
        atr5_map=atr5_map,
    )
    ob_piv = (
        pivot_tf(
            ob_map,
            [
                c
                for c in ob_map.columns
                if c
                in (
                    "ob_above_atr",
                    "ob_below_atr",
                    "above_ob_bias",
                    "below_ob_bias",
                    "above_ob_structure_class",
                    "below_ob_structure_class",
                    "above_ob_zone_low",
                    "above_ob_zone_high",
                    "below_ob_zone_low",
                    "below_ob_zone_high",
                )
            ],
        )
        if not ob_map.empty
        else pd.DataFrame(columns=["candidate_id"])
    )

    # --- assemble ---
    state = candidates[list(EVENT_FIELDS)].copy()
    state["touch_behavior"] = touch_behavior(state)

    for f in QUANTILE_FIELDS:
        state[f] = pd.to_numeric(
            candidates[f], errors="coerce"
        ).to_numpy()
    state["quant_state"] = [
        quant_state(v)
        for v in state["quant_width_percentile_train"]
    ]

    state = state.merge(smc_piv, on="candidate_id", how="left")
    state = state.merge(mom_piv, on="candidate_id", how="left")
    if len(ob_piv.columns) > 1:
        state = state.merge(ob_piv, on="candidate_id", how="left")

    # --- SMC pivot distances -> execution ATR ---
    tc = pd.to_numeric(
        state["candidate_id"].map(touch_close_map),
        errors="coerce",
    ).to_numpy(float)
    a5 = pd.to_numeric(
        state["candidate_id"].map(atr5_map), errors="coerce"
    ).to_numpy(float)

    for tf in VALIDATED_TFS:
        for src, dst in SMC_PIVOT_DISTANCE_MAP.items():
            col = f"{src}_{tf}"
            if col not in state.columns:
                raise RuntimeError(f"missing SMC column {col}")
            state[f"{dst}_{tf}"] = pct_to_exec_atr(
                state[col], tc, a5
            )
        for src, dst in DSA_RENAMES.items():
            if src == dst:
                continue
            col = f"{src}_{tf}"
            if col in state.columns:
                state[f"{dst}_{tf}"] = state[col]

    state["decision_weight"] = (
        1.0
        / pd.to_numeric(
            state["group_candidate_count"], errors="coerce"
        ).to_numpy(float)
    )

    info = {
        "4h_context_input_rows_seen": quarantined_ctx_input,
        "4h_levels_input_rows_seen": quarantined_lv_input,
        "level_vocabulary": vocab,
    }
    return state, info


# ============================================================
# Action expansion + action-relative state
# ============================================================

def expand_actions(state: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for action in ACTIONS:
        x = state.copy()
        x["action"] = action
        if action == "SKIP":
            x["trade_mode"] = "SKIP"
            x["trade_direction"] = 0
            x["stop_atr"] = 0.0
            x["target_R"] = 0.0
        else:
            mode, target_r = parse_action(action)
            mult = 1 if mode == "FOLLOW" else -1
            x["trade_mode"] = mode
            x["trade_direction"] = (
                x["source_ob_bias"].to_numpy(float) * mult
            )
            x["stop_atr"] = STOP_ATR
            x["target_R"] = target_r
        rows.append(x)
    return pd.concat(rows, ignore_index=True)


def add_relative_bias(
    df: pd.DataFrame, col: str, out: str
) -> None:
    td = df["trade_direction"].to_numpy(float)
    raw = pd.to_numeric(
        df[col], errors="coerce"
    ).to_numpy(float)
    df[out] = np.where(td == 0, 0.0, raw * td)


def directional_pair(
    df: pd.DataFrame,
    *,
    high_col: str,
    low_col: str,
    out_forward: str,
    out_backward: str,
) -> None:
    """high/low -> forward/backward relative to the action."""
    td = df["trade_direction"].to_numpy(float)
    hi = pd.to_numeric(
        df[high_col], errors="coerce"
    ).to_numpy(float)
    lo = pd.to_numeric(
        df[low_col], errors="coerce"
    ).to_numpy(float)
    df[out_forward] = np.where(
        td > 0, hi, np.where(td < 0, lo, np.nan)
    )
    df[out_backward] = np.where(
        td > 0, lo, np.where(td < 0, hi, np.nan)
    )


def _safe_div(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ok = np.isfinite(a) & np.isfinite(b) & (b > 0)
    out = np.full(len(a), np.nan)
    out[ok] = a[ok] / b[ok]
    return out


def add_action_relative(
    action_df: pd.DataFrame,
) -> pd.DataFrame:
    df = action_df
    for tf in VALIDATED_TFS:
        assert_validated_tf(tf)

        for kind in ("internal", "swing"):
            add_relative_bias(
                df, f"{kind}_bias_{tf}", f"{kind}_bias_rel_{tf}"
            )

        directional_pair(
            df,
            high_col=f"internal_high_atr_{tf}",
            low_col=f"internal_low_atr_{tf}",
            out_forward=f"forward_internal_atr_{tf}",
            out_backward=f"backward_internal_atr_{tf}",
        )
        directional_pair(
            df,
            high_col=f"swing_high_atr_{tf}",
            low_col=f"swing_low_atr_{tf}",
            out_forward=f"forward_swing_atr_{tf}",
            out_backward=f"backward_swing_atr_{tf}",
        )
        directional_pair(
            df,
            high_col=f"ob_above_atr_{tf}",
            low_col=f"ob_below_atr_{tf}",
            out_forward=f"forward_active_ob_atr_{tf}",
            out_backward=f"backward_active_ob_atr_{tf}",
        )

        td = df["trade_direction"].to_numpy(float)
        df[f"dsa_alignment_{tf}"] = (
            pd.to_numeric(
                df[f"dsa_direction_{tf}"], errors="coerce"
            ).to_numpy(float)
            * td
        )
        df[f"dsa_vwap_dev_rel_{tf}"] = (
            pd.to_numeric(
                df[f"dsa_vwap_dev_pct_{tf}"], errors="coerce"
            ).to_numpy(float)
            * td
        )
        df[f"momentum_value_rel_{tf}"] = (
            pd.to_numeric(
                df[f"sqzmom_val_{tf}"], errors="coerce"
            ).to_numpy(float)
            * td
        )
        df[f"momentum_delta_rel_{tf}"] = (
            pd.to_numeric(
                df[f"sqzmom_delta_{tf}"], errors="coerce"
            ).to_numpy(float)
            * td
        )

        tgt = df["target_R"].to_numpy(float)
        stp = df["stop_atr"].to_numpy(float)

        df[f"target_fit_internal_{tf}"] = _safe_div(
            df[f"forward_internal_atr_{tf}"].to_numpy(float),
            tgt,
        )
        df[f"target_fit_swing_{tf}"] = _safe_div(
            df[f"forward_swing_atr_{tf}"].to_numpy(float),
            tgt,
        )
        df[f"target_fit_ob_{tf}"] = _safe_div(
            df[f"forward_active_ob_atr_{tf}"].to_numpy(float),
            tgt,
        )

        df[f"stop_structure_internal_{tf}"] = _safe_div(
            df[f"backward_internal_atr_{tf}"].to_numpy(float),
            stp,
        )
        df[f"stop_structure_swing_{tf}"] = _safe_div(
            df[f"backward_swing_atr_{tf}"].to_numpy(float),
            stp,
        )
        df[f"stop_structure_ob_{tf}"] = _safe_div(
            df[f"backward_active_ob_atr_{tf}"].to_numpy(float),
            stp,
        )
    return df


# ============================================================
# Reward (reuses the validated Phase-1 simulator)
# ============================================================

def attach_rewards(
    action_df: pd.DataFrame,
    candidates: pd.DataFrame,
    path: pd.DataFrame,
) -> pd.DataFrame:
    n = len(candidates)
    assert_action_cardinality(n, action_df)

    cid_order = candidates["candidate_id"].astype(str).to_numpy()
    got = action_df["candidate_id"].astype(str).to_numpy()[:n]
    if not np.array_equal(got, cid_order):
        raise RuntimeError(
            "action block order does not match candidates order"
        )

    arrays = build_path_arrays(candidates, path)
    bias = pd.to_numeric(
        candidates["source_ob_bias"], errors="coerce"
    ).to_numpy(float)

    reward_map: dict = {}
    for mode, sign in (("FOLLOW", bias), ("FADE", -bias)):
        rot = rotated(arrays, sign)
        for target_r in TARGET_R:
            for horizon in DIAGNOSTIC_HORIZONS:
                base, code = simulate(
                    rot,
                    arrays["contig"],
                    stop_atr=STOP_ATR,
                    target_r=target_r,
                    horizon=horizon,
                    require_contiguous=True,
                )
                result = apply_policy(
                    base, code, target_r, SAME_BAR_POLICY
                )
                reward_map[(mode, target_r, horizon)] = (
                    np.asarray(result, dtype=float),
                    np.asarray(code, dtype=int),
                )

    m = len(action_df)
    gross = {
        h: np.zeros(m, dtype=float)
        for h in DIAGNOSTIC_HORIZONS
    }
    codes = {
        h: np.full(m, SKIP_EXIT_CODE, dtype=int)
        for h in DIAGNOSTIC_HORIZONS
    }

    for i, action in enumerate(ACTIONS):
        if action == "SKIP":
            continue
        sl = slice(i * n, (i + 1) * n)
        mode, target_r = parse_action(action)
        for h in DIAGNOSTIC_HORIZONS:
            res, code = reward_map[(mode, target_r, h)]
            gross[h][sl] = res
            codes[h][sl] = code

    for h in DIAGNOSTIC_HORIZONS:
        action_df[f"gross_R_h{h}"] = gross[h]
        action_df[f"exit_code_h{h}"] = codes[h]

    action_df["primary_reward_R"] = action_df[
        f"gross_R_h{PRIMARY_HORIZON}"
    ]
    action_df["reward_version"] = REWARD_VERSION
    return action_df


# ============================================================
# Audits
# ============================================================

def audit_state_columns(df: pd.DataFrame) -> None:
    bad = [
        c
        for c in df.columns
        if c.startswith(FORBIDDEN_STATE_PREFIXES)
    ]
    leak = sorted(set(FORBIDDEN_STATE_COLUMNS) & set(df.columns))
    if bad or leak:
        raise RuntimeError(
            "future/execution leakage in state: "
            f"prefixed={bad} explicit={leak}"
        )


def audit_no_quarantined(*frames: pd.DataFrame) -> int:
    hits = 0
    for df in frames:
        for c in df.columns:
            if "4h" in str(c):
                hits += 1
            s = df[c]
            if s.dtype == object:
                if s.astype(str).eq("4h").any():
                    hits += 1
    if hits:
        raise RuntimeError(
            f"quarantined 4h present in output: {hits}"
        )
    return 0


def audit_state_cardinality(
    state: pd.DataFrame, candidates: pd.DataFrame
) -> None:
    if len(state) != len(candidates):
        raise RuntimeError(
            "state cardinality mismatch: "
            f"{len(state)} != {len(candidates)}"
        )
    if not state["candidate_id"].is_unique:
        raise RuntimeError("state candidate_id not unique")


# ============================================================
# Main (Gate B only)
# ============================================================

def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = load_full_or_chunks("candidates")
    context = load_full_or_chunks("context")
    levels = load_full_or_chunks("levels")
    path = load_full_or_chunks("path")

    symbols = sorted(candidates["symbol"].unique().tolist())
    raw_five = {s: load_raw_five(s) for s in symbols}

    momentum_by_tf: dict[tuple[str, str], pd.DataFrame] = {}
    for s in symbols:
        five = raw_five[s]
        fifteen = (
            aggregate_15m(five)
            .sort_values("bar_start_time")
            .reset_index(drop=True)
        )
        one_hour = (
            aggregate_1h_from_15m(fifteen)
            .sort_values("bar_start_time")
            .reset_index(drop=True)
        )
        momentum_by_tf[(s, "5m")] = build_momentum_frame(five)
        momentum_by_tf[(s, "15m")] = build_momentum_frame(
            fifteen
        )
        momentum_by_tf[(s, "1h")] = build_momentum_frame(
            one_hour
        )

    state, info = build_state(
        candidates, context, levels, momentum_by_tf
    )
    audit_state_cardinality(state, candidates)
    audit_state_columns(state)

    action_df = expand_actions(state)
    action_df = add_action_relative(action_df)
    action_df = attach_rewards(action_df, candidates, path)

    assert_action_cardinality(len(state), action_df)
    audit_state_columns(
        action_df.drop(
            columns=[
                c
                for c in (
                    "gross_R_h6",
                    "gross_R_h12",
                    "gross_R_h24",
                    "exit_code_h6",
                    "exit_code_h12",
                    "exit_code_h24",
                    "primary_reward_R",
                    "reward_version",
                )
                if c in action_df.columns
            ]
        )
    )
    audit_no_quarantined(state, action_df)

    state.to_csv(OUT_ROOT / "ob_rl_state_v0.csv", index=False)
    action_df.to_csv(
        OUT_ROOT / "ob_rl_action_v0.csv", index=False
    )

    manifest = {
        "dataset_version": DATASET_VERSION,
        "baseline_sha": BASELINE_SHA,
        "reward_version": REWARD_VERSION,
        "candidates": int(len(candidates)),
        "state_rows": int(len(state)),
        "action_rows": int(len(action_df)),
        "actions": list(ACTIONS),
        "stop_atr": STOP_ATR,
        "target_R": list(TARGET_R),
        "primary_horizon": PRIMARY_HORIZON,
        "diagnostic_horizons": list(DIAGNOSTIC_HORIZONS),
        "same_bar_policy": SAME_BAR_POLICY,
        "timeframes": list(VALIDATED_TFS),
        "quarantined": list(QUARANTINED_TFS),
        "4h_context_input_rows_seen": info[
            "4h_context_input_rows_seen"
        ],
        "4h_levels_input_rows_seen": info[
            "4h_levels_input_rows_seen"
        ],
        "level_vocabulary": info["level_vocabulary"],
        "state_columns": list(state.columns),
        "action_columns": list(action_df.columns),
    }
    (OUT_ROOT / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("OB_RL_DATASET_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
