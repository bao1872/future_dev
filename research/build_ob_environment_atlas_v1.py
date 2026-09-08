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
  SQZMOM is never reimplemented. Canonical string states
  (momentum_direction / momentum_change / volatility_phase) are carried
  verbatim; a separate numeric ``momentum_sqzmom_sign`` is derived
  explicitly for directional combination.
* Momentum is joined on the *already causally completed* context bar
  index (``bar_index``), and missing bars raise.
* The levels primary key is ``event_id``; its mapping to
  ``candidate_id`` is PROVEN at runtime, never assumed.
* Pressure/support uses the COMMITTED touch-time zone-edge geometry:
  ``relation`` + ``distance_pct``, ATR-normalised against the 5m
  touch-bar close. Distances are NOT recomputed from object center and
  NOT measured from the next 5m open.
* A candidate x timeframe with no structural object is EXPLICIT (counts
  zero, nearest NaN) -- "no pressure/support" is itself an environment.
* All dsa_raw_* fields are carried (exact parity with committed V3
  schema, asserted at runtime).
* 4h is quarantined: input rows are counted, output rows are asserted 0.
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
    density_field,
    MOMENTUM_STATE_FIELDS,
    MOMENTUM_JOIN_FIELD,
    MOMENTUM_SIGN_FIELD,
    EXPECTED_DSA_RAW_FIELDS,
    DSA_ENV_FIELDS,
    STRUCTURE_FIELDS,
    ACTIVE_OB_COUNT_FIELDS,
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
    """Canonical SQZMOM momentum history for a timeframe."""
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
    """Join canonical momentum onto the causally-completed context bar.

    Uses the real context ``bar_index``. A missing momentum bar is a
    hard error (reindex-based NaN checks are not reliable).
    """
    parts = []
    for tf, g in env_tf.groupby("context_tf", sort=False):
        assert_validated_tf(tf)
        mom = momentum_by_tf[tf].set_index(MOMENTUM_JOIN_FIELD)
        idx = g["bar_index"].astype(int).to_numpy()

        missing = ~np.isin(idx, mom.index.to_numpy(int))
        if missing.any():
            raise RuntimeError(
                f"{tf}: {int(missing.sum())} context bars "
                "missing momentum"
            )

        m = mom.loc[idx]
        x = g.copy()
        for c in MOMENTUM_STATE_FIELDS:
            x[f"momentum_{c}"] = m[c].to_numpy()
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def add_momentum_sign(env_tf: pd.DataFrame) -> pd.DataFrame:
    """Derive numeric SQZMOM sign from canonical sqzmom_val.

    sqzmom_val > 0 -> +1 (expanding)
    sqzmom_val < 0 -> -1 (contracting)
    sqzmom_val = 0 ->  0 (flat)

    Canonical string fields are left untouched.
    """
    out = env_tf.copy()
    v = pd.to_numeric(
        out.get("momentum_sqzmom_val"), errors="coerce"
    ).to_numpy(float)
    sign = np.full(len(out), np.nan)
    ok = np.isfinite(v)
    sign[ok] = np.sign(v[ok])
    out[MOMENTUM_SIGN_FIELD] = sign
    return out


def resolve_level_join_key(
    candidates: pd.DataFrame,
    levels: pd.DataFrame,
) -> dict:
    """PROVE the mapping between candidate_id and levels event_id."""
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
    sym_cd = cmap.loc[ev.to_numpy(), "symbol"].astype(str).to_numpy()
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
            f"trigger_time_mismatch={time_mismatch}"
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
    object_price: float, price: float, atr5: float
) -> float:
    if not np.isfinite(atr5) or atr5 <= 0:
        return np.nan
    return abs(float(object_price) - float(price)) / atr5


def distance_tf_atr(
    object_price: float, price: float, atr_tf: float
) -> float:
    if not np.isfinite(atr_tf) or atr_tf <= 0:
        return np.nan
    return abs(float(object_price) - float(price)) / atr_tf


def classify_level_type(
    object_type: object, structure_class: object
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

    Unmapped pairs are reported with examples; the caller must STOP.
    """
    if levels.empty:
        return {
            "pairs": 0,
            "unknown_pairs": 0,
            "unknown_examples": [],
            "buckets": {},
        }
    pairs = (
        levels[["object_type", "structure_class"]]
        .fillna("")
        .astype(str)
        .drop_duplicates()
    )
    buckets: dict[str, int] = {}
    unknown_pairs = 0
    unknown_examples: list[str] = []
    for ot, sc in pairs.itertuples(index=False):
        b = classify_level_type(ot, sc)
        buckets[b] = buckets.get(b, 0) + 1
        if b == "other":
            unknown_pairs += 1
            unknown_examples.append(f"{ot}/{sc}")
    return {
        "pairs": int(len(pairs)),
        "unknown_pairs": int(unknown_pairs),
        "unknown_examples": unknown_examples[:20],
        "buckets": buckets,
    }


def normalize_level_distance(
    levels: pd.DataFrame,
    *,
    touch_close: float,
    atr5: float,
    atr_tf: float,
) -> pd.DataFrame:
    """ATR-normalise the COMMITTED touch-time zone-edge distance.

    Uses ``distance_pct`` (calculated by the V3 source owner from
    relation_to_price(zone_low, zone_high, trigger_close)). The price
    reference is the 5m touch-bar CLOSE, never the next open, and the
    geometry is the zone edge, never the object center.
    """
    x = levels.copy()
    pct = pd.to_numeric(
        x["distance_pct"], errors="coerce"
    ).to_numpy(float)
    abs_distance = pct / 100.0 * touch_close

    if np.isfinite(atr5) and atr5 > 0:
        x["distance_exec_atr"] = abs_distance / atr5
    else:
        x["distance_exec_atr"] = np.nan

    if np.isfinite(atr_tf) and atr_tf > 0:
        x["distance_tf_atr"] = abs_distance / atr_tf
    else:
        x["distance_tf_atr"] = np.nan
    return x


def summarize_level_side(
    levels: pd.DataFrame,
    *,
    side: str,
) -> dict:
    """Pressure/support summary for one side.

    Expects ``distance_exec_atr`` / ``distance_tf_atr`` / ``level_type``
    already present (see normalize_level_distance).
    """
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
        out[density_field(side, b)] = 0
    for b in LEVEL_TYPE_BUCKETS:
        out[f"{side}_{b}_count"] = 0
        out[f"{side}_nearest_{b}_exec_atr"] = np.nan
    out[f"{side}_active_bull_ob_count"] = 0
    out[f"{side}_active_bear_ob_count"] = 0

    if x.empty:
        return out

    de = x["distance_exec_atr"].to_numpy(float)
    ok = np.isfinite(de)

    if ok.any():
        j = int(np.nanargmin(de))
        out[f"{side}_nearest_exec_atr"] = float(de[j])
        out[f"{side}_nearest_tf_atr"] = float(
            x["distance_tf_atr"].to_numpy(float)[j]
        )
        out[f"{side}_nearest_type"] = str(
            x["object_type"].to_numpy()[j]
        )
        out[f"{side}_nearest_structure_class"] = str(
            x["structure_class"].to_numpy()[j]
        )
        for b in DIST_BINS:
            out[density_field(side, b)] = int(
                (de[ok] <= b).sum()
            )

    ltypes = x["level_type"].astype(str).to_numpy()
    for b in LEVEL_TYPE_BUCKETS:
        m = ltypes == b
        out[f"{side}_{b}_count"] = int(m.sum())
        if m.any():
            sub = de[m]
            sub = sub[np.isfinite(sub)]
            if len(sub):
                out[f"{side}_nearest_{b}_exec_atr"] = float(
                    sub.min()
                )

    am = ltypes == "active_ob"
    if am.any():
        bz = pd.to_numeric(x["bias"], errors="coerce").to_numpy(
            float
        )
        out[f"{side}_active_bull_ob_count"] = int(
            (am & (bz == 1)).sum()
        )
        out[f"{side}_active_bear_ob_count"] = int(
            (am & (bz == -1)).sum()
        )
    return out


def full_candidate_tf_grid(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Every candidate x validated timeframe, so that 'no levels' is
    represented explicitly instead of vanishing."""
    ids = candidates[["candidate_id"]].drop_duplicates()
    tfs = pd.DataFrame({"context_tf": list(VALIDATED_TFS)})
    return ids.merge(tfs, how="cross")


def path_excursion(
    g: pd.DataFrame, *, direction: int, horizon: int
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
    context: pd.DataFrame,
    momentum_by_tf: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict]:
    tf_series = context["context_tf"].astype(str)

    # 4h quarantine: count what was SEEN, then exclude.
    quarantined_input_rows = int(
        tf_series.isin(QUARANTINED_TFS).sum()
    )

    ctx = context[tf_series.isin(VALIDATED_TFS)].copy()

    if ctx["context_tf"].astype(str).isin(QUARANTINED_TFS).any():
        raise RuntimeError("quarantined TF leaked into output")

    # DSA raw parity with the committed V3 schema -- EXACT equality.
    # A newly appearing dsa_raw_* field is just as much a contract
    # break as a missing one, so "expected - actual" is not enough.
    actual_dsa = set(
        c for c in ctx.columns if c.startswith("dsa_raw_")
    )
    expected_dsa = set(EXPECTED_DSA_RAW_FIELDS)
    if actual_dsa != expected_dsa:
        raise RuntimeError(
            "DSA raw schema drift: "
            f"missing={sorted(expected_dsa - actual_dsa)} "
            f"extra={sorted(actual_dsa - expected_dsa)}"
        )

    wanted = (
        set(STRUCTURE_FIELDS)
        | set(CURRENT_PIVOT_FIELDS)
        | set(OVERLAP_FIELDS)
        | set(NEAREST_FIELDS)
        | set(DSA_ENV_FIELDS)
        | set(ACTIVE_OB_COUNT_FIELDS)
        | {"atr14", "close"}
    )
    cols = [c for c in ctx.columns if c in wanted]

    env = ctx[
        [
            "candidate_id",
            "symbol",
            "context_tf",
            "bar_index",
            "bar_end",
            "lag_minutes",
        ]
        + cols
    ].copy()

    # IMPORTANT: momentum joins on the canonical context bar_index, so
    # it must run BEFORE any rename of that column.
    env = attach_momentum(env, momentum_by_tf)
    env = add_momentum_sign(env)

    # ATR must be comparable across symbols and price levels, so the
    # environment carries ATR / price, not ATR alone.
    atr = pd.to_numeric(
        env["atr14"], errors="coerce"
    ).to_numpy(float)
    close = pd.to_numeric(
        env["close"], errors="coerce"
    ).to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        atr_pct = np.where(
            (close > 0) & np.isfinite(close), atr / close, np.nan
        )
    env["atr_pct"] = atr_pct

    env = env.rename(columns={"bar_index": "context_bar_index"})

    info = {
        "4h_input_rows_seen": quarantined_input_rows,
        "4h_output_rows": int(
            env["context_tf"].astype(str)
            .isin(QUARANTINED_TFS).sum()
        ),
        "dsa_raw_expected": len(EXPECTED_DSA_RAW_FIELDS),
        "dsa_raw_carried": len(
            [c for c in env.columns if c.startswith("dsa_raw_")]
        ),
    }
    return env, info


def build_candidate_level_map(
    candidates: pd.DataFrame,
    levels: pd.DataFrame,
    context: pd.DataFrame,
) -> tuple[pd.DataFrame, dict, dict, dict]:
    join_proof = resolve_level_join_key(candidates, levels)
    vocab = verify_level_vocabulary(levels)

    # Unmapped structural vocabulary => STOP, never silently "other".
    if vocab["unknown_pairs"] != 0:
        raise RuntimeError(
            "unmapped level vocabulary: "
            f"{vocab['unknown_examples']}"
        )

    tf_series = levels["timeframe"].astype(str)
    quarantined_input_rows = int(
        tf_series.isin(QUARANTINED_TFS).sum()
    )
    lv = levels[tf_series.isin(VALIDATED_TFS)].copy()
    if lv["timeframe"].astype(str).isin(QUARANTINED_TFS).any():
        raise RuntimeError("quarantined TF leaked into level map")

    # touch_close: 5m context bar close (NOT the next open).
    c5 = context[context["context_tf"].astype(str).eq("5m")]
    touch_close_map = (
        c5.set_index("candidate_id")["close"]
        .apply(pd.to_numeric, errors="coerce")
        .to_dict()
    )
    atr5_map = dict(
        zip(
            candidates["candidate_id"].astype(str),
            candidates["5m_atr14"].to_numpy(float),
        )
    )

    a = context[
        context["context_tf"].astype(str).isin(VALIDATED_TFS)
    ][["candidate_id", "context_tf", "atr14"]].copy()
    atr_map = {
        (c, t): v
        for c, t, v in zip(
            a["candidate_id"].astype(str),
            a["context_tf"].astype(str),
            pd.to_numeric(a["atr14"], errors="coerce").to_numpy(
                float
            ),
        )
    }

    rows = []
    for (cid, tf), g in lv.groupby(
        ["event_id", "timeframe"], sort=False
    ):
        cid = str(cid)
        tf = str(tf)
        assert_validated_tf(tf)

        g2 = normalize_level_distance(
            g,
            touch_close=touch_close_map.get(cid, np.nan),
            atr5=atr5_map.get(cid, np.nan),
            atr_tf=atr_map.get((cid, tf), np.nan),
        )
        g2 = g2.assign(
            level_type=[
                classify_level_type(o, s)
                for o, s in zip(
                    g2["object_type"], g2["structure_class"]
                )
            ]
        )

        row = {
            "candidate_id": cid,
            "context_tf": tf,
            "level_object_count": int(len(g2)),
        }
        for side in ("above", "below"):
            row.update(summarize_level_side(g2, side=side))
        ov = g2[g2["relation"].astype(str).eq("overlap")]
        row["overlap_object_count"] = int(len(ov))
        row["overlap_active_ob_count"] = int(
            (
                ov["level_type"].astype(str).eq("active_ob")
            ).sum()
        )
        rows.append(row)

    grid = full_candidate_tf_grid(candidates)
    out = grid.merge(
        pd.DataFrame(rows),
        on=["candidate_id", "context_tf"],
        how="left",
        validate="one_to_one",
    )

    # Explicit zero for all counts; nearest distances stay NaN.
    count_cols = [
        c for c in out.columns if c.endswith("_count")
    ]
    for c in count_cols:
        out[c] = pd.to_numeric(
            out[c], errors="coerce"
        ).fillna(0).astype(int)

    quar = {
        "4h_input_rows_seen": quarantined_input_rows,
        "4h_output_rows": int(
            out["context_tf"].astype(str)
            .isin(QUARANTINED_TFS).sum()
        ),
    }
    return out, join_proof, vocab, quar


def assert_cardinality(
    candidates: pd.DataFrame,
    env_tf: pd.DataFrame,
    level_map: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> dict:
    """Cheap hard gates that synthetic data satisfies trivially but
    real data must be forced to satisfy.

        env_tf    == candidates x 3 validated TFs
        level_map == candidates x 3 validated TFs
        outcomes  == candidates (unique)
    """
    n = int(candidates["candidate_id"].nunique())
    if candidates["candidate_id"].duplicated().any():
        raise RuntimeError("duplicate candidate_id in candidates")

    expected_tf_rows = n * len(VALIDATED_TFS)

    if len(env_tf) != expected_tf_rows:
        raise RuntimeError(
            "environment cardinality mismatch: "
            f"{len(env_tf)} != {expected_tf_rows}"
        )
    if env_tf.duplicated(["candidate_id", "context_tf"]).any():
        raise RuntimeError(
            "duplicate candidate x TF environment row"
        )
    if len(level_map) != expected_tf_rows:
        raise RuntimeError(
            "level map cardinality mismatch: "
            f"{len(level_map)} != {expected_tf_rows}"
        )
    if level_map.duplicated(["candidate_id", "context_tf"]).any():
        raise RuntimeError(
            "duplicate candidate x TF level map row"
        )
    if len(outcomes) != n:
        raise RuntimeError(
            "outcome cardinality mismatch: "
            f"{len(outcomes)} != {n}"
        )
    if outcomes["candidate_id"].duplicated().any():
        raise RuntimeError("duplicate candidate_id in outcomes")

    return {
        "candidates": n,
        "expected_tf_rows": expected_tf_rows,
        "env_tf_rows": int(len(env_tf)),
        "level_map_rows": int(len(level_map)),
        "outcomes_rows": int(len(outcomes)),
        "duplicate_keys": 0,
    }


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

    env_frames = []
    env_quar = {"4h_input_rows_seen": 0, "4h_output_rows": 0}
    dsa_counts: dict = {}
    for s in symbols:
        ids = set(
            candidates.loc[
                candidates["symbol"] == s, "candidate_id"
            ].astype(str)
        )
        sub = context[
            context["candidate_id"].astype(str).isin(ids)
        ]
        mom = {
            tf: momentum_by_tf[(s, tf)] for tf in VALIDATED_TFS
        }
        e, info = build_candidate_env_tf(sub, mom)
        env_frames.append(e)
        env_quar["4h_input_rows_seen"] += info[
            "4h_input_rows_seen"
        ]
        env_quar["4h_output_rows"] += info["4h_output_rows"]
        dsa_counts[s] = {
            "expected": info["dsa_raw_expected"],
            "carried": info["dsa_raw_carried"],
        }

    env_tf = pd.concat(env_frames, ignore_index=True)

    level_map, join_proof, vocab, quar_lv = (
        build_candidate_level_map(
            candidates, levels, context
        )
    )
    outcomes = build_candidate_outcomes(candidates, path)

    cardinality = assert_cardinality(
        candidates, env_tf, level_map, outcomes
    )

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
        "4h_input_rows_seen": (
            env_quar["4h_input_rows_seen"]
            + quar_lv["4h_input_rows_seen"]
        ),
        "4h_output_rows": (
            env_quar["4h_output_rows"]
            + quar_lv["4h_output_rows"]
        ),
        "dsa_raw_parity": dsa_counts,
        "cardinality": cardinality,
        "atr_pct_carried": bool("atr_pct" in env_tf.columns),
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
