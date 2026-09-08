#!/usr/bin/env python3

"""Build the OB Environment Atlas V1 tables (Gate B component).

Produces three SEPARATE tables so that environment and outcome can
never contaminate each other:

    candidate_env_tf     candidate x timeframe environment
    candidate_level_map  candidate x timeframe pressure/support map
    candidate_outcomes   candidate-level future payoff distribution

Hard rules
----------
* Momentum is consumed from the canonical source owner
  (``research.indicator_adapter.compute_smc_momentum_bundle``).
  SQZMOM is never reimplemented.
* Momentum is joined on the *already causally completed* context bar
  index, so there is no lookahead.
* The levels primary key is ``event_id``; its mapping to
  ``candidate_id`` is PROVEN at runtime, never assumed.
* Pressure/support keeps timeframe identity: no collapse to a single
  "forward room" number. Nearest object AND multi-bin density AND
  object-type breakdown are all retained.
* Distances are stored in BOTH execution ATR (5m) and TF-local ATR.
* 4h is quarantined and never enters any atlas table.
* No best / rank / optimize / select logic exists in this module.
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
PHASE1 = ROOT / "research" / "analysis_results" / "ob_candidate_v3_phase1"

from research.audit_ob_candidate_v3_posthoc import (  # noqa: E402
    load_full_or_chunks,
    add_true_continuity,
    load_raw_five,
)
from research.indicator_adapter import (  # noqa: E402
    compute_smc_momentum_bundle,
)
from research.build_pytdx_panel import aggregate_15m  # noqa: E402
from research.ob_trigger_snapshot import (  # noqa: E402
    aggregate_1h_from_15m,
)
from research.environment_atlas_spec import (  # noqa: E402
    ATLAS_VERSION,
    ATLAS_BASELINE_SHA,
    VALIDATED_TFS,
    QUARANTINED_TFS,
    HORIZONS,
    DIST_BINS,
    MOMENTUM_STATE_FIELDS,
    MOMENTUM_JOIN_FIELD,
    DSA_ENV_FIELDS,
    STRUCTURE_FIELDS,
    CURRENT_PIVOT_FIELDS,
    OVERLAP_FIELDS,
    NEAREST_FIELDS,
    LEVELS_RELATIONS,
    LEVEL_TYPE_BUCKETS,
    SOURCE_OWNERS,
    FOUR_HOUR_AUTHORITY,
    assert_validated_tf,
)


def _to_canonical_frame(bars: pd.DataFrame) -> pd.DataFrame:
    x = bars.copy()
    x = x.set_index(
        pd.DatetimeIndex(pd.to_datetime(x["bar_start_time"]))
    )
    return x.sort_index()


def build_momentum_frame(bars: pd.DataFrame) -> pd.DataFrame:
    """Canonical SQZMOM momentum history for a timeframe.

    Source owner: panji_indicators.compute_sqzmom_lb +
    build_momentum_history, reached through
    research.indicator_adapter.compute_smc_momentum_bundle.
    """
    bundle = compute_smc_momentum_bundle(
        _to_canonical_frame(bars)
    )
    states = bundle.momentum_history["daily_state"]
    out = pd.DataFrame(states)

    if len(out) != len(bars):
        raise RuntimeError(
            "momentum history/bar length mismatch: "
            f"{len(out)} != {len(bars)}"
        )
    if not np.array_equal(
        out[MOMENTUM_JOIN_FIELD].to_numpy(int),
        np.arange(len(bars)),
    ):
        raise RuntimeError("momentum bar_index alignment failure")
    return out


def attach_momentum(
    env_tf: pd.DataFrame,
    momentum_by_tf: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Join canonical momentum onto the causally-completed context bar."""
    parts = []
    for tf, g in env_tf.groupby("context_tf", sort=False):
        assert_validated_tf(tf)
        mom = momentum_by_tf[tf].set_index(MOMENTUM_JOIN_FIELD)
        idx = g["bar_index"].astype(int)
        x = g.copy()
        m = mom.reindex(idx.to_numpy())
        if m.index.isna().any():
            raise RuntimeError(f"{tf}: missing momentum rows")
        for c in MOMENTUM_STATE_FIELDS:
            x[f"momentum_{c}"] = m[c].to_numpy()
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def resolve_level_join_key(
    candidates: pd.DataFrame,
    levels: pd.DataFrame,
) -> dict:
    """PROVE the mapping between candidate_id and levels event_id.

    Never inferred silently. Three independent checks:
      1. every event_id exists as a candidate_id;
      2. symbol agrees;
      3. levels trigger_time equals that candidate's touch_time.
    """
    ev = levels["event_id"].astype(str)
    cd = candidates["candidate_id"].astype(str)

    missing = set(ev.unique()) - set(cd.unique())
    if missing:
        raise RuntimeError(
            "levels event_id not found in candidates: "
            f"{len(missing)} e.g. {sorted(missing)[:5]}"
        )

    cmap = candidates.set_index("candidate_id")

    sym_lv = levels["symbol"].astype(str).to_numpy()
    sym_cd = (
        cmap.loc[ev.to_numpy(), "symbol"].astype(str).to_numpy()
    )
    sym_mismatch = int((sym_lv != sym_cd).sum())

    tt_lv = pd.to_datetime(levels["trigger_time"]).to_numpy(
        "datetime64[ns]"
    )
    tt_cd = pd.to_datetime(
        cmap.loc[ev.to_numpy(), "touch_time"].to_numpy()
    ).to_numpy("datetime64[ns]")
    time_mismatch = int((tt_lv != tt_cd).sum())

    if sym_mismatch or time_mismatch:
        raise RuntimeError(
            "levels join key proof failed: "
            f"symbol_mismatch={sym_mismatch} "
            f"time_mismatch={time_mismatch}"
        )

    return {
        "proven": True,
        "level_rows": int(len(levels)),
        "distinct_event_ids": int(ev.nunique()),
        "missing_in_candidates": 0,
        "symbol_mismatch": 0,
        "trigger_time_mismatch": 0,
        "method": (
            "event_id -> candidate_id, verified by symbol and by "
            "trigger_time == candidate touch_time on every row"
        ),
    }


def distance_exec_atr(
    object_price: float,
    price: float,
    atr5: float,
) -> float:
    if not np.isfinite(atr5) or atr5 <= 0:
        return np.nan
    return abs(float(object_price) - float(price)) / atr5


def distance_tf_atr(
    object_price: float,
    price: float,
    atr_tf: float,
) -> float:
    if not np.isfinite(atr_tf) or atr_tf <= 0:
        return np.nan
    return abs(float(object_price) - float(price)) / atr_tf


def classify_level_type(
    object_type: object,
    structure_class: object,
) -> str:
    ot = "" if object_type is None else str(object_type)
    sc = "" if structure_class is None else str(structure_class)
    low = ot.lower()
    if "active" in low:
        return "active_ob"
    if "internal" in low:
        return "internal_pivot"
    if "swing" in low:
        return "swing_pivot"
    if "equal" in low or sc.lower() == "equal":
        return "equal_level"
    return "other"


def verify_level_vocabulary(levels: pd.DataFrame) -> dict:
    """Verify classification vocabulary against REAL levels data.

    Unknown (object_type, structure_class) pairs are counted rather
    than silently dropped.
    """
    if levels.empty:
        return {"pairs": 0, "unknown_pairs": 0, "buckets": {}}
    pairs = (
        levels[["object_type", "structure_class"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
    )
    buckets: dict[str, int] = {}
    unknown = 0
    for ot, sc in pairs.itertuples(index=False):
        b = classify_level_type(ot, sc)
        buckets[b] = buckets.get(b, 0) + 1
        if b == "other" and LEVEL_TYPE_BUCKETS:
            unknown += 1
    return {
        "pairs": int(len(pairs)),
        "unknown_pairs": int(unknown),
        "buckets": buckets,
    }


def summarize_level_side(
    levels: pd.DataFrame,
    *,
    price: float,
    atr5: float,
    atr_tf: float,
    side: str,
) -> dict:
    assert side in ("above", "below")

    rel = levels["relation"].astype(str)
    x = levels[rel.eq(side)].copy()

    out = {
        f"{side}_object_count": int(len(x)),
        f"{side}_nearest_exec_atr": np.nan,
        f"{side}_nearest_tf_atr": np.nan,
        f"{side}_nearest_type": None,
        f"{side}_nearest_structure_class": None,
    }
    for b in DIST_BINS:
        out[f"{side}_count_within_{b:g}atr"] = 0
    for b in LEVEL_TYPE_BUCKETS:
        out[f"{side}_{b}_count"] = 0

    if x.empty:
        return out

    center = x["object_price_center"].to_numpy(float)
    d_exec = np.abs(center - price) / atr5 if (
        np.isfinite(atr5) and atr5 > 0
    ) else np.full(len(x), np.nan)
    d_tf = np.abs(center - price) / atr_tf if (
        np.isfinite(atr_tf) and atr_tf > 0
    ) else np.full(len(x), np.nan)
    x = x.assign(
        distance_exec_atr=d_exec,
        distance_tf_atr=d_tf,
    )

    types = [
        classify_level_type(o, s)
        for o, s in zip(
            x["object_type"], x["structure_class"]
        )
    ]
    x = x.assign(level_type=types)

    j = int(np.nanargmin(x["distance_exec_atr"].to_numpy(float)))
    out[f"{side}_nearest_exec_atr"] = float(
        x["distance_exec_atr"].to_numpy(float)[j]
    )
    out[f"{side}_nearest_tf_atr"] = float(
        x["distance_tf_atr"].to_numpy(float)[j]
    )
    out[f"{side}_nearest_type"] = str(
        x["object_type"].to_numpy()[j]
    )
    out[f"{side}_nearest_structure_class"] = str(
        x["structure_class"].to_numpy()[j]
    )

    de = x["distance_exec_atr"].to_numpy(float)
    ok = np.isfinite(de)
    for b in DIST_BINS:
        out[f"{side}_count_within_{b:g}atr"] = int(
            (de[ok] <= b).sum()
        )
    for b in LEVEL_TYPE_BUCKETS:
        out[f"{side}_{b}_count"] = int(
            (np.asarray(types) == b).sum()
        )
    return out


def path_excursion(
    g: pd.DataFrame,
    *,
    direction: int,
    horizon: int,
) -> dict | None:
    x = g[g["step"] <= horizon].sort_values("step")
    if len(x) != horizon:
        return None
    if not bool(x["execution_contiguous"].all()):
        return None

    if direction == 1:
        mfe = x["high_atr"].max()
        mae = -x["low_atr"].min()
        terminal = x.iloc[-1]["close_atr"]
    else:
        mfe = -x["low_atr"].min()
        mae = x["high_atr"].max()
        terminal = -x.iloc[-1]["close_atr"]

    return {
        "terminal_R_atr": float(terminal),
        "mfe_atr": float(max(mfe, 0)),
        "mae_atr": float(max(mae, 0)),
    }


# ============================================================
# Table builders
# ============================================================

def build_candidate_env_tf(
    candidates: pd.DataFrame,
    context: pd.DataFrame,
    momentum_by_tf: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    ctx = context.copy()
    ctx = ctx[
        ctx["context_tf"].astype(str).isin(VALIDATED_TFS)
    ].copy()
    if (ctx["context_tf"].astype(str).isin(QUARANTINED_TFS)).any():
        raise RuntimeError("quarantined 4h entered environment table")

    wanted = (
        set(STRUCTURE_FIELDS)
        | set(CURRENT_PIVOT_FIELDS)
        | set(OVERLAP_FIELDS)
        | set(NEAREST_FIELDS)
        | set(DSA_ENV_FIELDS)
        | {"atr14"}
    )
    cols = [c for c in ctx.columns if c in wanted]
    env = ctx[
        ["candidate_id", "symbol", "context_tf", "bar_index", "bar_end", "lag_minutes"]
        + cols
    ].copy()

    env = env.rename(
        columns={"bar_index": "context_bar_index"}
    )
    env = attach_momentum(env, momentum_by_tf)
    return env


def build_candidate_level_map(
    candidates: pd.DataFrame,
    levels: pd.DataFrame,
    context: pd.DataFrame,
) -> pd.DataFrame:
    join_proof = resolve_level_join_key(candidates, levels)
    vocab = verify_level_vocabulary(levels)

    lv = levels[
        levels["timeframe"].astype(str).isin(VALIDATED_TFS)
    ].copy()
    if (lv["timeframe"].astype(str).isin(QUARANTINED_TFS)).any():
        raise RuntimeError("quarantined 4h entered level map")

    # atr per (candidate, tf) from the context snapshot
    a = context[
        context["context_tf"].astype(str).isin(VALIDATED_TFS)
    ][["candidate_id", "context_tf", "atr14"]].copy()
    atr_map = {
        (c, t): v
        for c, t, v in zip(
            a["candidate_id"].astype(str),
            a["context_tf"].astype(str),
            a["atr14"].to_numpy(float),
        )
    }

    price_map = dict(
        zip(
            candidates["candidate_id"].astype(str),
            candidates["entry_next_5m_open"].to_numpy(float),
        )
    )
    atr5_map = dict(
        zip(
            candidates["candidate_id"].astype(str),
            candidates["5m_atr14"].to_numpy(float),
        )
    )

    rows = []
    for (cid, tf), g in lv.groupby(
        ["event_id", "timeframe"], sort=False
    ):
        cid = str(cid)
        tf = str(tf)
        assert_validated_tf(tf)
        price = price_map.get(cid, np.nan)
        atr5 = atr5_map.get(cid, np.nan)
        atr_tf = atr_map.get((cid, tf), np.nan)

        row = {
            "candidate_id": cid,
            "context_tf": tf,
            "level_object_count": int(len(g)),
        }
        for side in ("above", "below"):
            row.update(
                summarize_level_side(
                    g,
                    price=price,
                    atr5=atr5,
                    atr_tf=atr_tf,
                    side=side,
                )
            )
        ov = g[g["relation"].astype(str).eq("overlap")]
        row["overlap_object_count"] = int(len(ov))
        rows.append(row)

    out = pd.DataFrame(rows)
    return out, join_proof, vocab


def build_candidate_outcomes(
    candidates: pd.DataFrame,
    path: pd.DataFrame,
) -> pd.DataFrame:
    path = add_true_continuity(path)
    bias = dict(
        zip(
            candidates["candidate_id"].astype(str),
            candidates["source_ob_bias"].to_numpy(int),
        )
    )

    rows = []
    for cid, g in path.groupby("candidate_id", sort=False):
        d = bias.get(str(cid), 1)
        row = {"candidate_id": str(cid)}
        for horizon in HORIZONS:
            for dname, direction in (
                ("follow", int(d)),
                ("fade", -int(d)),
            ):
                r = path_excursion(
                    g, direction=direction, horizon=horizon
                )
                pre = f"{dname}_h{horizon}"
                if r is None:
                    row[f"{pre}_terminal_R_atr"] = np.nan
                    row[f"{pre}_mfe_atr"] = np.nan
                    row[f"{pre}_mae_atr"] = np.nan
                else:
                    row[f"{pre}_terminal_R_atr"] = r[
                        "terminal_R_atr"
                    ]
                    row[f"{pre}_mfe_atr"] = r["mfe_atr"]
                    row[f"{pre}_mae_atr"] = r["mae_atr"]
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# Main (Gate B only)
# ============================================================

def main() -> None:
    ATLAS_ROOT.mkdir(parents=True, exist_ok=True)

    candidates = load_full_or_chunks("candidates")
    context = load_full_or_chunks("context")
    levels = load_full_or_chunks("levels")
    path = load_full_or_chunks("path")

    ann = pd.read_csv(
        PHASE1 / "candidate_annotations.csv"
    )
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

    momentum_by_tf = {}
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
        momentum_by_tf[(s, "15m")] = build_momentum_frame(fifteen)
        momentum_by_tf[(s, "1h")] = build_momentum_frame(one_hour)

    env_frames = []
    for s in symbols:
        sub = context[
            context["candidate_id"].isin(
                set(
                    candidates.loc[
                        candidates["symbol"] == s,
                        "candidate_id",
                    ]
                )
            )
        ]
        mom = {
            tf: momentum_by_tf[(s, tf)] for tf in VALIDATED_TFS
        }
        env_frames.append(
            build_candidate_env_tf(
                candidates[candidates["symbol"] == s], sub, mom
            )
        )
    env_tf = pd.concat(env_frames, ignore_index=True)

    level_map, join_proof, vocab = build_candidate_level_map(
        candidates, levels, context
    )
    outcomes = build_candidate_outcomes(candidates, path)

    env_tf.to_csv(
        ATLAS_ROOT / "candidate_env_tf.csv", index=False
    )
    level_map.to_csv(
        ATLAS_ROOT / "candidate_level_map.csv", index=False
    )
    outcomes.to_csv(
        ATLAS_ROOT / "candidate_outcomes.csv", index=False
    )

    coverage = {
        "atlas_version": ATLAS_VERSION,
        "baseline_sha": ATLAS_BASELINE_SHA,
        "candidates": int(len(candidates)),
        "env_tf_rows": int(len(env_tf)),
        "level_map_rows": int(len(level_map)),
        "outcomes_rows": int(len(outcomes)),
        "timeframes": list(VALIDATED_TFS),
        "quarantined": list(QUARANTINED_TFS),
        "levels_join_proof": join_proof,
        "level_vocabulary": vocab,
        "source_owners": SOURCE_OWNERS,
        "four_hour_authority": FOUR_HOUR_AUTHORITY,
    }
    (ATLAS_ROOT / "environment_coverage.json").write_text(
        json.dumps(coverage, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("ATLAS_BUILD_DONE", flush=True)


if __name__ == "__main__":
    main()
