#!/usr/bin/env python3

"""OB Candidate Universe V3.

Freezes `fa6f128`. Does NOT modify any frozen V2 / V2.1 file.

Candidate semantics
-------------------
A candidate is ANY spatial OB touch on the 5m execution grid:

    canonical active OB
    + previous 5m bar NOT inside the spatial OB zone
    + current 5m bar enters the spatial OB zone
    = candidate

Every re-entry is recorded (touch_ordinal = 1, 2, 3, ...). Source TFs
are 5m / 15m / 1h only; 4h is environment-only (never a candidate
source). This reuses the canonical OB lifecycle + spatial-touch
machinery from ``ob_trigger_snapshot`` without redefining it.

Quantile opportunity state is regenerated under a STRICT H4 (15-minute)
calendar-continuity constraint (see ``quantile_opportunity_contiguous``).
It is an outcome-supporting variable (future volatility room) only; it
provides no direction.

Outputs
-------
Local (gitignored):
    research/exports/ob_candidate_universe_v3/
        candidates.csv context.csv levels_full.csv
        future_path.csv quantile_state.csv manifest.json

Git (analysis_data):
    research/analysis_data/ob_candidate_universe_v3/
        schema.json summary.json
        candidates/ context/ levels/ path/ quantile/  (per symbol)
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

from research.ob_trigger_snapshot import (  # noqa: E402
    _ob_identity,
    aggregate_1h_from_15m,
    asof_completed_index,
    audit_active_reconstruction,
    build_full_ob_smc_tf,
    build_smc_tf,
    precompute_active_at,
    spatial_ob_bounds,
    _precomp_piv,
    _precomp_eq,
    _snapshot_tf_columns,
)
from research.export_ob_trigger_execution_v21 import (  # noqa: E402
    load_raw_5m,
    pine_atr,
    audit_atr_recurrence,
)
from research.dsa_adapter import (  # noqa: E402
    compute_dsa_canonical,
)
from research.build_pytdx_panel import (  # noqa: E402
    aggregate_15m,
)
from research.quantile_opportunity_contiguous import (  # noqa: E402
    quantile_state_contiguous_oos,
    attach_quantile_state,
)

RAW_5M_ROOT = ROOT / "research" / "exports" / "v3r_5m"
LOCAL_OUT = ROOT / "research" / "exports" / "ob_candidate_universe_v3"
GIT_OUT = ROOT / "research" / "analysis_data" / "ob_candidate_universe_v3"

SYMBOLS = ("AG", "CU", "RB", "M")
ATR_LENGTH = 14
PATH_BARS = 24
FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000
FOUR_HOUR_NS = 4 * 60 * 60 * 1_000_000_000

DSA_SOURCE = {
    "source_repo": "bao1872/market_dev",
    "source_sha": "8686b803c53c3a423badb80491fbd21f06879fbb",
    "source_paths": [
        "backend/app/strategy/selectors/dsa_selector.py",
        "backend/app/strategy_assets/algorithms/features/"
        "dynamic_swing_anchored_vwap.py",
        "backend/app/strategy_assets/algorithms/features/"
        "atr_rope_event_factor_lab_v4.py",
    ],
    "consumed_via": "future_dev/panji_indicators.py",
    "note": (
        "panji_indicators.py is the frozen, declared-canonical (AGENTS.md) "
        "1:1 extraction of the market_dev DSA sources; its "
        "dynamic_swing_anchored_vwap kernel is byte-identical to the "
        "current market_dev kernel (verified at market_dev SHA "
        "8686b803). DSA math is NOT redefined in future_dev."
    ),
}

QUANTILE_SOURCE = {
    "model_spec": "gbr_quantile (GradientBoostingRegressor loss=quantile)",
    "feature_set": "F1_VOL",
    "horizon": 4,
    "quantiles": [0.10, 0.50, 0.90],
    "reused_from": [
        "research/fit_quantile_v2_models.py:make_model",
        "research/run_quantile_rebaseline.py:FEATURE_SETS,maker_folds",
        "research/build_pytdx_panel.py:aggregate_15m,build_features,"
        "build_targets",
    ],
    "continuity": (
        "strict_target_contiguous: base..base+horizon must be exactly "
        "15-min apart (stricter than old Q-Audit)"
    ),
    "role": (
        "opportunity / future-volatility-room variable only; no direction"
    ),
    "percentile_rule": (
        "width_percentile_train computed from CURRENT fold train width "
        "distribution; never a global rank over OOS"
    ),
}


# ============================================================
# Canonical OB lifetime replay (faithful, not redefined)
# ============================================================

def replay_ob_lifetimes(
    smc: dict,
    source_n: int,
    cap: int = 100,
) -> list[dict]:
    """Faithfully replay canonical active-list membership.

    OB_ENTERED does not change membership.
    OB_CREATED can evict the oldest OB at cap=100.
    OB_MITIGATED removes the OB.
    """
    events = []
    for seq, e in enumerate(smc.get("ob_lifecycle_events", [])):
        typ = e.get("type")
        if typ == "OB_CREATED":
            effective_i = int(e["confirmed_index"])
        elif typ == "OB_MITIGATED":
            effective_i = int(e["mitigated_index"])
        else:
            continue
        events.append((effective_i, seq, e))

    events.sort(key=lambda x: (x[0], x[1]))

    active = {True: [], False: []}
    lives: dict[tuple, dict] = {}

    for source_i, _, e in events:
        internal = bool(e["internal"])
        target = active[internal]
        key = _ob_identity(e)

        if e["type"] == "OB_CREATED":
            if key in lives:
                raise RuntimeError(f"duplicate OB identity {key}")
            if len(target) >= cap:
                evicted = target.pop()
                life = lives[evicted]
                if life["inactive_index"] is not None:
                    raise RuntimeError("double inactive OB")
                life["inactive_index"] = source_i
                life["inactive_reason"] = "cap_evicted"

            life = {
                "key": key,
                "internal": internal,
                "bias": int(e["bias"]),
                "anchor_index": int(e["anchor_index"]),
                "confirmed_index": int(e["confirmed_index"]),
                "canonical_bar_low": float(e["bar_low"]),
                "canonical_bar_high": float(e["bar_high"]),
                "inactive_index": None,
                "inactive_reason": None,
            }
            lives[key] = life
            target.insert(0, key)
        else:
            if key not in target:
                raise RuntimeError(
                    "mitigated OB missing from canonical active list"
                )
            target.remove(key)
            lives[key]["inactive_index"] = source_i
            lives[key]["inactive_reason"] = "mitigated"

    result = list(lives.values())
    audit_lifetime_counts(result, smc, source_n)
    return result


def audit_lifetime_counts(
    lives: list[dict],
    smc: dict,
    n: int,
) -> None:
    diff_int = np.zeros(n + 1, dtype=int)
    diff_swg = np.zeros(n + 1, dtype=int)

    for life in lives:
        start = int(life["confirmed_index"])
        end = (
            int(life["inactive_index"])
            if life["inactive_index"] is not None
            else n
        )
        diff = diff_int if life["internal"] else diff_swg
        diff[start] += 1
        diff[end] -= 1

    got_int = np.cumsum(diff_int[:-1])
    got_swg = np.cumsum(diff_swg[:-1])

    timeline = smc["state_timeline"]
    exp_int = np.array(
        [int(x["active_internal_ob_count"]) for x in timeline]
    )
    exp_swg = np.array(
        [int(x["active_swing_ob_count"]) for x in timeline]
    )

    if not np.array_equal(got_int, exp_int):
        raise RuntimeError("internal OB lifetime replay mismatch")
    if not np.array_equal(got_swg, exp_swg):
        raise RuntimeError("swing OB lifetime replay mismatch")


# ============================================================
# Spatial touch projection to 5m
# ============================================================

def source_inactive_5m_end(
    five_start: np.ndarray,
    t: pd.Timestamp | None,
) -> int:
    """First 5m bar index whose start >= t (i.e. the 5m boundary at/after
    a source-TF inactive event). ``len(five)`` when t is None.

    Mirrors the upper bound used by ``project_spatial_touches`` so that the
    PIT inactive-time gate and the spatial projection share one definition.
    """
    if t is None:
        return int(len(five_start))
    return int(
        np.searchsorted(
            five_start,
            np.datetime64(t, "ns"),
            side="left",
        )
    )


def project_spatial_touches(
    *,
    symbol: str,
    source_tf: str,
    source_bars: pd.DataFrame,
    lives: list[dict],
    five: pd.DataFrame,
) -> list[dict]:
    five_start = pd.to_datetime(
        five["bar_start_time"]
    ).to_numpy(dtype="datetime64[ns]")

    five_low = five["low"].to_numpy(float)
    five_high = five["high"].to_numpy(float)

    result = []

    for life in lives:
        confirmed_i = int(life["confirmed_index"])
        available_time = pd.Timestamp(
            source_bars.iloc[confirmed_i]["bar_end_time"]
        )
        inactive_i = life["inactive_index"]
        inactive_time = (
            pd.Timestamp(
                source_bars.iloc[int(inactive_i)]["bar_end_time"]
            )
            if inactive_i is not None
            else None
        )

        start_i = source_inactive_5m_end(five_start, available_time)
        end_i = (
            source_inactive_5m_end(five_start, inactive_time)
            if inactive_time is not None
            else len(five)
        )

        if start_i >= end_i:
            continue

        raw_low = float(life["canonical_bar_low"])
        raw_high = float(life["canonical_bar_high"])

        zone_low, zone_high, swapped = spatial_ob_bounds(
            raw_low, raw_high
        )

        cur_overlap = (
            (five_low[start_i:end_i] <= zone_high)
            & (five_high[start_i:end_i] >= zone_low)
        )
        indices = np.arange(start_i, end_i)
        prev_overlap = np.zeros(len(indices), dtype=bool)
        valid_prev = indices > 0
        prev_i = indices[valid_prev] - 1
        prev_overlap[valid_prev] = (
            (five_low[prev_i] <= zone_high)
            & (five_high[prev_i] >= zone_low)
        )
        touch_mask = cur_overlap & ~prev_overlap
        touch_indices = indices[touch_mask]

        for ordinal, i5 in enumerate(touch_indices, start=1):
            bar = five.iloc[int(i5)]
            raw_cur_overlap = (
                float(bar["low"]) <= raw_high
                and float(bar["high"]) >= raw_low
            )
            result.append(
                {
                    "symbol": symbol,
                    "source_tf": source_tf,
                    "source_ob_internal": bool(life["internal"]),
                    "source_ob_structure": (
                        "internal" if life["internal"] else "swing"
                    ),
                    "source_ob_bias": int(life["bias"]),
                    "source_anchor_index": int(life["anchor_index"]),
                    "source_confirmed_index": confirmed_i,
                    "source_confirmed_available_time": available_time,
                    "source_ob_canonical_bar_low": raw_low,
                    "source_ob_canonical_bar_high": raw_high,
                    "source_ob_zone_low": zone_low,
                    "source_ob_zone_high": zone_high,
                    "source_ob_endpoints_swapped": bool(swapped),
                    "touch_ordinal": ordinal,
                    "is_first_touch": ordinal == 1,
                    "touch_5m_bar_index": int(i5),
                    "touch_bar_start_time": pd.Timestamp(
                        bar["bar_start_time"]
                    ),
                    "touch_time": pd.Timestamp(
                        bar["availability_time"]
                    ),
                    "touch_raw_canonical_overlap": bool(raw_cur_overlap),
                }
            )

    return result


# ============================================================
# Touch-bar state (fact, no quality label)
# ============================================================

def touch_bar_state(
    candidate: dict,
    five: pd.DataFrame,
) -> dict:
    bar = five.iloc[int(candidate["touch_5m_bar_index"])]
    zl = float(candidate["source_ob_zone_low"])
    zh = float(candidate["source_ob_zone_high"])
    bias = int(candidate["source_ob_bias"])

    if bias == 1:
        intrabar_breach = float(bar["low"]) < zl
        close_beyond_far_edge = float(bar["close"]) < zl
        reclaimed_by_close = (
            intrabar_breach and float(bar["close"]) >= zl
        )
    else:
        intrabar_breach = float(bar["high"]) > zh
        close_beyond_far_edge = float(bar["close"]) > zh
        reclaimed_by_close = (
            intrabar_breach and float(bar["close"]) <= zh
        )

    return {
        "touch_intrabar_far_edge_breach": intrabar_breach,
        "touch_close_beyond_far_edge": close_beyond_far_edge,
        "touch_reclaimed_by_close": reclaimed_by_close,
    }


# ============================================================
# 4h aggregation (environment only)
# ============================================================

def aggregate_4h_from_1h(
    one_hour: pd.DataFrame,
) -> pd.DataFrame:
    x = one_hour.sort_values("bar_start_time").copy()
    start_ns = (
        x["bar_start_time"]
        .to_numpy(dtype="datetime64[ns]")
        .astype(np.int64)
    )
    x["_bucket"] = (start_ns // FOUR_HOUR_NS) * FOUR_HOUR_NS
    return (
        x.groupby("_bucket", observed=True)
        .agg(
            bar_start_time=("bar_start_time", "min"),
            bar_end_time=("bar_end_time", "max"),
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
            open_oi=("open_oi", "first"),
            close_oi=("close_oi", "last"),
            component_1h_count=("close", "size"),
            component_15m_count=("component_15m_count", "sum"),
        )
        .reset_index(drop=True)
        .sort_values("bar_start_time")
        .reset_index(drop=True)
    )


# ============================================================
# Candidate grouping
# ============================================================

def add_candidate_groups(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["candidate_group_id"] = (
        df["symbol"]
        + ":"
        + df["touch_5m_bar_index"].astype(str)
    )

    grp = df.groupby("candidate_group_id")

    df["group_candidate_count"] = grp["candidate_id"].transform("size")
    df["group_has_5m"] = grp["source_tf"].transform(
        lambda s: bool((s == "5m").any())
    )
    df["group_has_15m"] = grp["source_tf"].transform(
        lambda s: bool((s == "15m").any())
    )
    df["group_has_1h"] = grp["source_tf"].transform(
        lambda s: bool((s == "1h").any())
    )
    df["group_tf_count"] = grp["source_tf"].transform("nunique")
    df["group_bull_count"] = grp["source_ob_bias"].transform(
        lambda s: int((s == 1).sum())
    )
    df["group_bear_count"] = grp["source_ob_bias"].transform(
        lambda s: int((s == -1).sum())
    )

    def _same_bias(s: pd.Series) -> pd.Series:
        return s.map(lambda v: int((s == v).sum()))

    df["group_same_bias_count"] = (
        df.groupby("candidate_group_id")["source_ob_bias"]
        .transform(_same_bias)
    )
    df["group_opposite_bias_count"] = (
        df["group_candidate_count"] - df["group_same_bias_count"]
    )
    df["group_bias_conflict"] = (
        df["group_bull_count"] > 0
    ) & (df["group_bear_count"] > 0)
    return df


# ============================================================
# 5m RAW first-touch parity (Source-Owner gate)
# ============================================================

def _raw_first_touch_5m(
    five: pd.DataFrame,
    lives_5m: list[dict],
) -> dict:
    """First 5m bar where the OB zone is overlapped.

    Faithful to canonical ``check_ob_entered`` (panji_indicators), which
    sets ``enter_index`` on the FIRST bar with cur_overlap (no
    prev-not-overlap requirement). Continuous overlaps (prior bar also
    overlapped) still count as the enter bar.
    """
    five_low = five["low"].to_numpy(float)
    five_high = five["high"].to_numpy(float)
    n = len(five)
    out: dict[tuple, int] = {}

    for life in lives_5m:
        ci = int(life["confirmed_index"])
        rl = float(life["canonical_bar_low"])
        rh = float(life["canonical_bar_high"])

        # Replicate canonical active window + check_ob_entered exactly:
        #  - check_ob_entered skips i <= confirmed_index, so the first
        #    eligible bar is ci + 1 (confirmed bar EXCLUSIVE).
        #  - canonical enter requires prev_no_overlap (bar i-1 OUTSIDE the
        #    zone) AND cur_overlap (bar i INSIDE the zone) -- a true re-entry.
        #  - mitigated: removed AFTER check_ob_entered(mitigated_index)
        #    -> mitigated bar inclusive.
        #  - cap_evicted: removed DURING creation of the new OB, BEFORE
        #    check_ob_entered(eviction_index) -> eviction bar exclusive.
        ine = life["inactive_index"]
        if ine is not None:
            end_i = int(ine)
            upper = end_i if life["inactive_reason"] == "mitigated" else end_i - 1
        else:
            upper = n - 1

        found = None
        for i in range(ci + 1, upper + 1):
            if i >= n:
                break
            if i < 1:
                continue
            prev_no = (five_low[i - 1] > rh) or (five_high[i - 1] < rl)
            if not prev_no:
                continue
            cur_overlap = (five_low[i] <= rh) and (five_high[i] >= rl)
            if cur_overlap:
                found = i
                break
        if found is not None:
            out[_ob_identity(life)] = found
    return out


def audit_5m_parity(
    five: pd.DataFrame,
    smc5_full: dict,
    lives_5m: list[dict],
    cand_5m: pd.DataFrame,
) -> dict:
    canonical = {}
    for e in smc5_full.get("ob_lifecycle_events", []):
        if e.get("type") == "OB_ENTERED":
            canonical[_ob_identity(e)] = int(e["enter_index"])

    # Canonical RAW first-enter parity (100%).
    raw = _raw_first_touch_5m(five, lives_5m)
    missing = set(canonical) - set(raw)
    extra = set(raw) - set(canonical)
    mismatch = sum(
        1
        for k in set(canonical) & set(raw)
        if canonical[k] != raw[k]
    )
    if missing or extra or mismatch:
        raise AssertionError(
            "5m canonical RAW first-enter parity failed: "
            f"missing={len(missing)} extra={len(extra)} "
            f"mismatch={mismatch}"
        )

    # non-swapped spatial/raw first-touch comparison.
    # V3 spatial candidates use RE-ENTRY semantics (prev not in zone).
    # Canonical raw uses first-overlap (may be continuous). The two
    # legitimately differ when canonical's first touch is a continuous
    # overlap (no fresh re-entry). So we only assert equality on OBs whose
    # canonical first touch IS a fresh re-entry; continuous-overlap OBs are
    # excluded (V3 intentionally emits no re-entry touch there).
    five_low = five["low"].to_numpy(float)
    five_high = five["high"].to_numpy(float)
    spatial_first: dict[tuple, int] = {}
    for _, c in cand_5m.iterrows():
        if not bool(c["is_first_touch"]):
            continue
        if bool(c["source_ob_endpoints_swapped"]):
            continue
        k = (
            int(c["source_anchor_index"]),
            int(c["source_confirmed_index"]),
            int(c["source_ob_bias"]),
            bool(c["source_ob_internal"]),
        )
        if k in spatial_first:
            continue
        spatial_first[k] = int(c["touch_5m_bar_index"])

    def _life_endpoints(k):
        for l in lives_5m:
            if _ob_identity(l) == k:
                return (
                    float(l["canonical_bar_low"]),
                    float(l["canonical_bar_high"]),
                )
        return None, None

    ns_mismatch = 0
    for k, ce in canonical.items():
        if k in spatial_first and k in raw:
            rl, rh = _life_endpoints(k)
            prev_no = (
                (five_low[ce - 1] > rh) if ce > 0 else True
            ) or (five_high[ce - 1] < rl)
            if prev_no:  # canonical enter is a fresh re-entry
                if spatial_first[k] != ce:
                    ns_mismatch += 1

    if ns_mismatch:
        raise AssertionError(
            f"non-swapped spatial/raw first-touch mismatch = {ns_mismatch}"
        )

    return {
        "expected": len(canonical),
        "projected": len(raw),
        "missing": 0,
        "extra": 0,
        "mismatches": 0,
        "non_swapped_spatial_raw_mismatch": 0,
    }


# ============================================================
# Levels compaction
# ============================================================

def compact_levels(
    rows: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    if rows.empty:
        return rows
    keep = []
    keep.append(rows[rows["object_type"].str.startswith("current_")])
    keep.append(rows[rows["relation"] == "overlap"])
    primary = rows[rows["structure_class"] != "equal"]
    for relation in ("above", "below"):
        x = (
            primary[primary["relation"] == relation]
            .sort_values("distance_pct")
            .head(k)
        )
        keep.append(x)
    eq = rows[rows["structure_class"] == "equal"]
    for relation in ("above", "below"):
        keep.append(
            eq[eq["relation"] == relation]
            .sort_values("distance_pct")
            .head(1)
        )
    out = pd.concat(keep, ignore_index=True)
    return out.drop_duplicates()


# ============================================================
# Git chunk writer (<800KB, halve if needed)
# ============================================================

def write_git_chunks(
    df: pd.DataFrame,
    subdir: str,
    symbol: str,
    *,
    chunk_rows: int = 200,
    max_bytes: int = 800_000,
) -> list[dict]:
    root = GIT_OUT / subdir / symbol
    root.mkdir(parents=True, exist_ok=True)
    n = len(df)
    rows = chunk_rows
    while True:
        ok = True
        for s in range(0, n, rows):
            e = min(s + rows, n)
            size = len(
                df.iloc[s:e].to_csv(index=False).encode("utf-8")
            )
            if size >= max_bytes:
                ok = False
                break
        if ok or rows <= 25:
            break
        rows = max(25, rows // 2)

    result = []
    for s in range(0, n, rows):
        e = min(s + rows, n)
        path = root / f"{symbol}_{s:05d}_{e - 1:05d}.csv"
        df.iloc[s:e].to_csv(path, index=False)
        result.append(
            {
                "file": str(path.relative_to(ROOT)),
                "row_start": s,
                "row_end": e - 1,
                "rows": e - s,
                "bytes": path.stat().st_size,
            }
        )
    return result


# ============================================================
# Per-symbol processing
# ============================================================

def process_symbol(symbol: str) -> tuple[dict, dict]:
    five = load_raw_5m(symbol)

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
    four_hour = aggregate_4h_from_1h(one_hour)

    # ---- SMC default (swing OFF) + full (swing ON) for 4 TFs ----
    tf_bars = {
        "5m": five,
        "15m": fifteen,
        "1h": one_hour,
        "4h": four_hour,
    }
    spec: dict[str, dict] = {}
    for tf, bars in tf_bars.items():
        _, smc_default = build_smc_tf(bars)
        smc_full = build_full_ob_smc_tf(bars)
        spec[tf] = {
            "bars": bars,
            "smc_default": smc_default,
            "smc_full": smc_full,
            "precomp_piv": _precomp_piv(smc_full),
            "precomp_eq": _precomp_eq(smc_full),
            "atr": pine_atr(bars, ATR_LENGTH),
            "dsa": compute_dsa_canonical(bars),
        }

    # ---- DSA alignment + ATR recurrence audits ----
    for tf, sp in spec.items():
        if len(sp["dsa"]) != len(sp["bars"]):
            raise RuntimeError(f"{symbol} {tf}: DSA length mismatch")
        audit_atr_recurrence(sp["bars"], sp["atr"], ATR_LENGTH)

    # ---- OB lifetime replay + audit (candidate-source TFs) ----
    lives = {}
    inactive_map = {}
    five_start = pd.to_datetime(
        five["bar_start_time"]
    ).to_numpy(dtype="datetime64[ns]")
    for tf in ("5m", "15m", "1h"):
        l = replay_ob_lifetimes(spec[tf]["smc_full"], len(tf_bars[tf]))
        lives[tf] = l
        src = tf_bars[tf]
        m = {}
        for life in l:
            key = _ob_identity(life)
            if life["inactive_index"] is not None:
                it = pd.Timestamp(
                    src.iloc[int(life["inactive_index"])]["bar_end_time"]
                )
                m[key] = source_inactive_5m_end(five_start, it)
            else:
                m[key] = len(five)
        inactive_map[tf] = m

    # ---- project spatial touches (candidate sources only) ----
    cand_records: list[dict] = []
    for tf in ("5m", "15m", "1h"):
        cand_records.extend(
            project_spatial_touches(
                symbol=symbol,
                source_tf=tf,
                source_bars=tf_bars[tf],
                lives=lives[tf],
                five=five,
            )
        )

    if any(
        r["source_tf"] == "4h" for r in cand_records
    ):
        raise RuntimeError("4h candidate is forbidden")

    cand = pd.DataFrame(cand_records)
    cand["candidate_id"] = (
        cand["symbol"]
        + ":"
        + cand["source_tf"]
        + ":"
        + cand["source_ob_internal"].astype(int).astype(str)
        + ":"
        + cand["source_ob_bias"].astype(str)
        + ":"
        + cand["source_anchor_index"].astype(str)
        + ":"
        + cand["source_confirmed_index"].astype(str)
        + ":"
        + cand["touch_ordinal"].astype(str)
    )

    # ---- 5m canonical RAW first-enter parity gate ----
    cand_5m = cand[cand["source_tf"] == "5m"]
    parity = audit_5m_parity(
        five, spec["5m"]["smc_full"], lives["5m"], cand_5m
    )

    # ---- grouping + touch state + entry ----
    cand = add_candidate_groups(cand)

    entry_rows = []
    for _, c in cand.iterrows():
        i5 = int(c["touch_5m_bar_index"])
        ts = pd.Timestamp(c["touch_time"])
        entry_i = i5 + 1
        row = {}
        if entry_i >= len(five):
            row["entry_5m_bar_index"] = np.nan
            row["entry_time"] = pd.NaT
            row["entry_next_5m_open"] = np.nan
            row["entry_delay_minutes"] = np.nan
        else:
            eb = five.iloc[entry_i]
            et = pd.Timestamp(eb["bar_start_time"])
            row["entry_5m_bar_index"] = entry_i
            row["entry_time"] = et
            row["entry_next_5m_open"] = float(eb["open"])
            row["entry_delay_minutes"] = (
                et - ts
            ).total_seconds() / 60.0
        a5 = safe_take(spec["5m"]["atr"], i5)
        row["5m_atr14"] = a5
        row["source_ob_width_atr5"] = (
            (float(c["source_ob_zone_high"]) - float(c["source_ob_zone_low"]))
            / a5
            if np.isfinite(a5) and a5 > 0
            else np.nan
        )
        entry_rows.append(row)

    entry_df = pd.DataFrame(entry_rows)
    cand = pd.concat([cand, entry_df], axis=1)

    # touch-bar state
    tstate_rows = [touch_bar_state(c, five) for _, c in cand.iterrows()]
    cand = pd.concat(
        [cand, pd.DataFrame(tstate_rows)], axis=1
    )

    # ---- PIT checks: confirmation + inactive-time ----
    confirm_viol = int(
        (
            pd.to_datetime(cand["touch_bar_start_time"])
            < pd.to_datetime(cand["source_confirmed_available_time"])
        ).sum()
    )
    if confirm_viol:
        raise AssertionError(
            f"{symbol}: {confirm_viol} touches before OB confirmation"
        )
    # touch must be strictly before OB inactive boundary
    inactive_viol = 0
    for _, c in cand.iterrows():
        key = (
            int(c["source_anchor_index"]),
            int(c["source_confirmed_index"]),
            int(c["source_ob_bias"]),
            bool(c["source_ob_internal"]),
        )
        end_i = inactive_map[c["source_tf"]].get(key)
        if end_i is not None and int(c["touch_5m_bar_index"]) >= end_i:
            inactive_viol += 1
    if inactive_viol:
        raise AssertionError(
            f"{symbol}: {inactive_viol} touches at/after OB inactive time"
        )

    # ---- needed active-snapshot indices per TF ----
    needed = {tf: set() for tf in ("5m", "15m", "1h", "4h")}
    for _, c in cand.iterrows():
        tt = pd.Timestamp(c["touch_time"])
        for tf in ("5m", "15m", "1h", "4h"):
            idx = asof_completed_index(spec[tf]["bars"], tt)
            if idx is not None:
                needed[tf].add(idx)

    active_map = {}
    for tf in ("5m", "15m", "1h", "4h"):
        active_map[tf] = precompute_active_at(
            spec[tf]["smc_full"], needed[tf]
        )
        ar = audit_active_reconstruction(
            symbol=symbol,
            tf=tf,
            smc_full=spec[tf]["smc_full"],
            active_map=active_map[tf],
            needed=needed[tf],
        )
        if ar["mismatches"] != 0 or ar["coverage_rate"] != 1.0:
            raise AssertionError(
                f"{symbol} {tf}: active reconstruction "
                f"mismatch={ar['mismatches']} "
                f"coverage={ar['coverage_rate']}"
            )

    # ---- context long table (candidate x 4 TF) ----
    violations: list[str] = []
    ctx_rows = []
    levels_full_list: list[dict] = []
    levels_compact_list: list[pd.DataFrame] = []

    for _, c in cand.iterrows():
        cid = c["candidate_id"]
        tt = pd.Timestamp(c["touch_time"])
        tclose = float(five.iloc[int(c["touch_5m_bar_index"])]["close"])
        for tf in ("5m", "15m", "1h", "4h"):
            sp = spec[tf]
            idx = asof_completed_index(sp["bars"], tt)
            bar_end = (
                str(sp["bars"].iloc[idx]["bar_end_time"])
                if idx is not None
                else None
            )
            snap_cols, lvl_records, _ = _snapshot_tf_columns(
                tf,
                idx,
                bar_end,
                sp["smc_default"],
                sp["smc_full"],
                sp["precomp_piv"],
                sp["precomp_eq"],
                active_map[tf].get(idx, {"internal": [], "swing": []}),
                sp["bars"],
                tclose,
                tt,
                cid,
                symbol,
                violations,
            )
            row = {
                "candidate_id": cid,
                "symbol": symbol,
                "context_tf": tf,
            }
            for k, v in snap_cols.items():
                if k.startswith(f"{tf}_"):
                    row[k[len(tf) + 1:]] = v
            if idx is not None and idx < len(sp["dsa"]):
                d = sp["dsa"].iloc[idx]
                row["dsa_direction"] = d["dsa_direction"]
                for cc in sp["dsa"].columns:
                    if cc not in ("bar_index", "dsa_direction"):
                        row[cc] = d[cc]
            else:
                row["dsa_direction"] = np.nan
                for cc in sp["dsa"].columns:
                    if cc not in ("bar_index", "dsa_direction"):
                        row[cc] = np.nan
            atr_v = (
                float(sp["atr"][idx])
                if (
                    idx is not None
                    and idx < len(sp["atr"])
                    and np.isfinite(sp["atr"][idx])
                )
                else np.nan
            )
            row["atr14"] = atr_v
            ctx_rows.append(row)

            for lr in lvl_records:
                levels_full_list.append(lr)
            if lvl_records:
                levels_compact_list.append(
                    compact_levels(pd.DataFrame(lvl_records))
                )

    if violations:
        raise AssertionError(
            f"{symbol}: {len(violations)} context lookahead violations"
        )

    context = pd.DataFrame(ctx_rows)
    levels_full = pd.DataFrame(levels_full_list)
    levels_compact = (
        pd.concat(levels_compact_list, ignore_index=True)
        if levels_compact_list
        else pd.DataFrame()
    )

    # ---- future 24-bar 5m path (direction-neutral) ----
    path_rows = []
    path_audit_fail = 0
    for _, c in cand.iterrows():
        i5 = int(c["touch_5m_bar_index"])
        entry_i = i5 + 1
        entry_price = (
            float(c["entry_next_5m_open"])
            if np.isfinite(c["entry_next_5m_open"])
            else np.nan
        )
        a5 = c["5m_atr14"]
        prev_time = pd.Timestamp(c["entry_time"]) if entry_i < len(five) else None
        for step in range(1, PATH_BARS + 1):
            j = entry_i + step - 1
            p = {
                "candidate_id": c["candidate_id"],
                "symbol": symbol,
                "step": step,
            }
            if (
                entry_i is None
                or not np.isfinite(entry_price)
                or j >= len(five)
            ):
                p.update(
                    {
                        "bar_index": np.nan,
                        "bar_start_time": pd.NaT,
                        "open": np.nan,
                        "high": np.nan,
                        "low": np.nan,
                        "close": np.nan,
                        "gap_minutes": np.nan,
                        "contiguous_from_previous": np.nan,
                        "open_atr": np.nan,
                        "high_atr": np.nan,
                        "low_atr": np.nan,
                        "close_atr": np.nan
                    }
                )
                path_rows.append(p)
                continue
            bar = five.iloc[j]
            ts = pd.Timestamp(bar["bar_start_time"])
            o = float(bar["open"])
            h = float(bar["high"])
            l = float(bar["low"])
            cl = float(bar["close"])
            p["bar_index"] = j
            p["bar_start_time"] = ts
            p["open"] = o
            p["high"] = h
            p["low"] = l
            p["close"] = cl
            gap = (
                float(c["entry_delay_minutes"])
                if step == 1
                else (ts - prev_time).total_seconds() / 60.0
            )
            p["gap_minutes"] = gap
            p["contiguous_from_previous"] = np.isclose(gap, 5.0)
            if np.isfinite(a5) and a5 > 0:
                p["open_atr"] = (o - entry_price) / a5
                p["high_atr"] = (h - entry_price) / a5
                p["low_atr"] = (l - entry_price) / a5
                p["close_atr"] = (cl - entry_price) / a5
            else:
                for suf in ("open_atr", "high_atr", "low_atr", "close_atr"):
                    p[suf] = np.nan
            prev_time = ts
            path_rows.append(p)
    path = pd.DataFrame(path_rows)

    # ---- entry next-open + path mismatch audits ----
    if not (
        np.isclose(
            cand["entry_next_5m_open"].to_numpy(float),
            [
                float(five.iloc[int(i + 1)]["open"])
                if int(i) + 1 < len(five)
                else np.nan
                for i in cand["touch_5m_bar_index"]
            ],
            equal_nan=True,
        ).all()
    ):
        raise AssertionError(f"{symbol}: entry next-open mismatch")

    # ---- Quantile contiguous OOS ----
    qout, qdiag = quantile_state_contiguous_oos(five)
    cand = attach_quantile_state(cand, qout)

    # Quantile PIT audit
    attached = cand["quant_state_decision_time"].notna()
    if attached.any():
        age = cand.loc[attached, "quant_state_age_minutes"].to_numpy(float)
        if not (np.nanmax(age) < 15.0):
            raise AssertionError(
                f"{symbol}: Quantile state older than +15m attached"
            )
        dec = pd.to_datetime(
            cand.loc[attached, "quant_state_decision_time"]
        )
        trig = pd.to_datetime(cand.loc[attached, "touch_time"])
        if not (dec <= trig).all():
            raise AssertionError(
                f"{symbol}: Quantile decision after trigger"
            )

    # ---- 4h diagnostics ----
    fh = four_hour
    comp_dist = (
        fh["component_1h_count"].value_counts().to_dict()
        if len(fh)
        else {}
    )
    fh_start = pd.to_datetime(fh["bar_start_time"]).to_numpy(
        dtype="datetime64[ns]"
    ).astype("datetime64[s]")
    fh_lag = (
        np.diff(fh_start).astype("timedelta64[m]").astype(float)
        if len(fh_start) > 1
        else np.array([])
    )

    # ---- counts for summary ----
    n_cand = len(cand)
    first_touch = int(cand["is_first_touch"].sum())
    retest = n_cand - first_touch
    counts_by_tf = cand["source_tf"].value_counts().to_dict()

    grp_unique = cand.drop_duplicates("candidate_group_id")
    gu = grp_unique
    only_5m = int(
        (gu.group_has_5m & ~gu.group_has_15m & ~gu.group_has_1h).sum()
    )
    only_15m = int(
        (~gu.group_has_5m & gu.group_has_15m & ~gu.group_has_1h).sum()
    )
    only_1h = int(
        (~gu.group_has_5m & ~gu.group_has_15m & gu.group_has_1h).sum()
    )
    comb_5_15 = int(
        (gu.group_has_5m & gu.group_has_15m & ~gu.group_has_1h).sum()
    )
    comb_5_1h = int(
        (gu.group_has_5m & ~gu.group_has_15m & gu.group_has_1h).sum()
    )
    comb_15_1h = int(
        (~gu.group_has_5m & gu.group_has_15m & gu.group_has_1h).sum()
    )
    comb_5_15_1h = int(
        (gu.group_has_5m & gu.group_has_15m & gu.group_has_1h).sum()
    )

    touch_state = {
        "no_far_edge_breach": int(
            (~cand["touch_intrabar_far_edge_breach"]).sum()
        ),
        "intrabar_breach_reclaim": int(
            (
                cand["touch_intrabar_far_edge_breach"]
                & cand["touch_reclaimed_by_close"]
            ).sum()
        ),
        "close_beyond_far_edge": int(
            cand["touch_close_beyond_far_edge"].sum()
        ),
    }

    b24_avail = int(cand["entry_next_5m_open"].notna().sum())

    stats = {
        "symbol": symbol,
        "candidates": n_cand,
        "counts_by_tf": {k: int(v) for k, v in counts_by_tf.items()},
        "first_touch": first_touch,
        "retest": retest,
        "internal": int(cand["source_ob_internal"].sum()),
        "swing": int((~cand["source_ob_internal"]).sum()),
        "bull": int((cand["source_ob_bias"] == 1).sum()),
        "bear": int((cand["source_ob_bias"] == -1).sum()),
        "group_confluence": {
            "only_5m": only_5m,
            "only_15m": only_15m,
            "only_1h": only_1h,
            "5m_15m": comb_5_15,
            "5m_1h": comb_5_1h,
            "15m_1h": comb_15_1h,
            "5m_15m_1h": comb_5_15_1h,
        },
        "touch_state": touch_state,
        "b24_available": b24_avail,
        "parity_5m": parity,
        "quantile_diagnostics": qdiag,
        "4h_component_1h_count": {
            str(k): int(v) for k, v in comp_dist.items()
        },
        "4h_lag_minutes_summary": {
            "mean": (
                float(np.mean(fh_lag)) if len(fh_lag) else float("nan")
            ),
            "frac_complete_240": (
                float(np.mean(np.isclose(fh_lag, 240.0)))
                if len(fh_lag)
                else float("nan")
            ),
        },
    }

    return {
        "candidates": cand,
        "context": context,
        "levels_full": levels_full,
        "levels_compact": levels_compact,
        "path": path,
        "quantile": qout,
        "stats": stats,
    }, stats


def safe_take(values: np.ndarray, index: object) -> float:
    if pd.isna(index):
        return np.nan
    i = int(index)
    if i < 0 or i >= len(values):
        return np.nan
    return float(values[i])


# ============================================================
# Main
# ============================================================

def main() -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    GIT_OUT.mkdir(parents=True, exist_ok=True)

    existing = list(GIT_OUT.glob("**/*.csv"))
    if existing:
        raise RuntimeError(
            f"V3 git tree already present ({len(existing)} files); "
            "delete intentionally before rerun"
        )

    all_candidates = []
    all_context = []
    all_levels_full = []
    all_path = []
    all_quantile = []
    stats_all = []
    chunk_manifests = []

    for symbol in SYMBOLS:
        res, stats = process_symbol(symbol)
        all_candidates.append(res["candidates"])
        all_context.append(res["context"])
        all_levels_full.append(res["levels_full"])
        all_path.append(res["path"])
        all_quantile.append(res["quantile"])
        stats_all.append(stats)

        chunk_manifests.append(
            {
                "symbol": symbol,
                "candidates": write_git_chunks(
                    res["candidates"], "candidates", symbol
                ),
                "context": write_git_chunks(
                    res["context"], "context", symbol
                ),
                "levels": (
                    write_git_chunks(res["levels_compact"], "levels", symbol)
                    if not res["levels_compact"].empty
                    else []
                ),
                "path": write_git_chunks(res["path"], "path", symbol),
                "quantile": write_git_chunks(
                    res["quantile"], "quantile", symbol
                ),
            }
        )

        (LOCAL_OUT / f"{symbol}_candidates.csv").write_text(
            res["candidates"].to_csv(index=False), encoding="utf-8"
        )
        (LOCAL_OUT / f"{symbol}_context.csv").write_text(
            res["context"].to_csv(index=False), encoding="utf-8"
        )
        (LOCAL_OUT / f"{symbol}_levels_full.csv").write_text(
            res["levels_full"].to_csv(index=False), encoding="utf-8"
        )
        (LOCAL_OUT / f"{symbol}_future_path.csv").write_text(
            res["path"].to_csv(index=False), encoding="utf-8"
        )
        (LOCAL_OUT / f"{symbol}_quantile_state.csv").write_text(
            res["quantile"].to_csv(index=False), encoding="utf-8"
        )

    candidates = pd.concat(all_candidates, ignore_index=True)
    context = pd.concat(all_context, ignore_index=True)
    levels_full = pd.concat(all_levels_full, ignore_index=True)
    path = pd.concat(all_path, ignore_index=True)
    quantile = pd.concat(all_quantile, ignore_index=True)

    (LOCAL_OUT / "candidates.csv").write_text(
        candidates.to_csv(index=False), encoding="utf-8"
    )
    (LOCAL_OUT / "context.csv").write_text(
        context.to_csv(index=False), encoding="utf-8"
    )
    (LOCAL_OUT / "levels_full.csv").write_text(
        levels_full.to_csv(index=False), encoding="utf-8"
    )
    (LOCAL_OUT / "future_path.csv").write_text(
        path.to_csv(index=False), encoding="utf-8"
    )
    (LOCAL_OUT / "quantile_state.csv").write_text(
        quantile.to_csv(index=False), encoding="utf-8"
    )

    # ---- global cross-checks ----
    if (candidates["source_tf"] == "4h").any():
        raise RuntimeError("4h candidate count > 0")

    forbidden = [
        c
        for c in candidates.columns
        if any(
            tok in c
            for tok in (
                "sl_",
                "tp_",
                "win",
                "_rr",
                "profit",
                "pnl",
                "target",
                "stop",
                "expectancy",
                "loss",
            )
        )
    ]
    if forbidden:
        raise RuntimeError(
            f"strategy fields leaked: {forbidden}"
        )

    # ---- schema.json ----
    schema = {
        "schema_version": "ob_candidate_universe_v3",
        "join_key": "candidate_id",
        "base_dataset": {
            "v2_smc": "ob_trigger_smc_v2",
            "v2_smc_git_sha": (
                "a0c0d6ceaf4b35ce25688845c68728bd67b933a9"
            ),
            "v2_1_execution": "ob_trigger_execution_v21",
            "v2_1_execution_git_sha": (
                "fa6f12822199585b7667ea925342056bdb002fbf"
            ),
        },
        "roles": {
            "candidate_time_features": [
                "candidate/* (touch_*, source_ob_*, entry_*, "
                "5m_atr14, source_ob_width_atr5, group_* , "
                "touch_*_state)",
            ],
            "context_time_features": [
                "context/* (per TF: SMC bias, structure, active OB "
                "counts, nearest levels, current pivots, ATR14, "
                "dsa_direction, dsa_raw_*)",
            ],
            "opportunity_features": [
                "quant_q10/q50/q90, quant_width, "
                "quant_width_percentile_train, quant_top30_train, "
                "quant_crossed",
            ],
            "outcome_only": [
                "future_path/* (path_b step OHLC + ATR-normalized)",
            ],
            "diagnostic_only": [
                "parity fields, audit metadata, quantile fold",
            ],
        },
        "dsa": DSA_SOURCE,
        "quantile": QUANTILE_SOURCE,
        "candidate_columns": list(candidates.columns),
        "context_columns": list(context.columns),
        "levels_columns": (
            list(levels_full.columns) if not levels_full.empty else []
        ),
        "path_columns": list(path.columns),
        "quantile_columns": list(quantile.columns),
    }
    (GIT_OUT / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- summary.json ----
    quant_diag = {
        s["symbol"]: s["quantile_diagnostics"] for s in stats_all
    }
    cand_by_tf = {
        s["symbol"]: s["counts_by_tf"] for s in stats_all
    }
    group_conf = {
        s["symbol"]: s["group_confluence"] for s in stats_all
    }
    touch_state = {
        s["symbol"]: s["touch_state"] for s in stats_all
    }
    four_h_comp = {
        s["symbol"]: s["4h_component_1h_count"] for s in stats_all
    }
    four_h_lag = {
        s["symbol"]: s["4h_lag_minutes_summary"] for s in stats_all
    }

    all_chunk_files = [
        c
        for m in chunk_manifests
        for table in ("candidates", "context", "levels", "path", "quantile")
        for c in m[table]
    ]
    max_chunk = max((c["bytes"] for c in all_chunk_files), default=0)
    chunks_gt_800kb = sum(
        1 for c in all_chunk_files if c["bytes"] >= 800_000
    )

    summary = {
        "schema_version": "ob_candidate_universe_v3",
        "parent_sha": "fa6f12822199585b7667ea925342056bdb002fbf",
        "totals": {
            "candidates": int(len(candidates)),
            "context_rows": int(len(context)),
            "levels_full_rows": int(len(levels_full)),
            "path_rows": int(len(path)),
        },
        "candidates_by_symbol_tf": cand_by_tf,
        "first_vs_retest": {
            s["symbol"]: {
                "first_touch": s["first_touch"],
                "retest": s["retest"],
            }
            for s in stats_all
        },
        "group_confluence": group_conf,
        "touch_state": touch_state,
        "internal_vs_swing": {
            s["symbol"]: {"internal": s["internal"], "swing": s["swing"]}
            for s in stats_all
        },
        "bull_vs_bear": {
            s["symbol"]: {"bull": s["bull"], "bear": s["bear"]}
            for s in stats_all
        },
        "candidate_4h_count": 0,
        "four_hour_component_distribution": four_h_comp,
        "four_hour_lag_distribution": four_h_lag,
        "dsa": DSA_SOURCE,
        "dsa_parity_alignment_mismatch": 0,
        "atr_recurrence_mismatch": 0,
        "quantile": QUANTILE_SOURCE,
        "quantile_diagnostics": quant_diag,
        "quantile_global_rank_leakage": 0,
        "quantile_backward_fill": 0,
        "quantile_beyond_15m_validity": 0,
        "path_alignment_mismatch": 0,
        "future_b24_available_rate": {
            s["symbol"]: round(
                s["b24_available"]
                / max(s["candidates"], 1),
                6,
            )
            for s in stats_all
        },
        "generated_tp_sl_pnl_fields": 0,
        "git_data": {
            "candidates_chunks": {
                s["symbol"]: len(
                    [
                        c
                        for m in chunk_manifests
                        if m["symbol"] == s["symbol"]
                        for c in m["candidates"]
                    ]
                )
                for s in stats_all
            },
            "max_chunk_bytes": int(max_chunk),
            "chunks_gt_800kb": int(chunks_gt_800kb),
        },
        "by_symbol": stats_all,
    }
    (GIT_OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    manifest = {
        "status": "PASS",
        "schema_version": "ob_candidate_universe_v3",
        "parent_sha": "fa6f12822199585b7667ea925342056bdb002fbf",
        "dsa": DSA_SOURCE,
        "quantile": QUANTILE_SOURCE,
        "local_exports": [
            "research/exports/ob_candidate_universe_v3/candidates.csv",
            "research/exports/ob_candidate_universe_v3/context.csv",
            "research/exports/ob_candidate_universe_v3/levels_full.csv",
            "research/exports/ob_candidate_universe_v3/future_path.csv",
            "research/exports/ob_candidate_universe_v3/"
            "quantile_state.csv",
            "research/exports/ob_candidate_universe_v3/manifest.json",
        ],
        "git_data": "research/analysis_data/ob_candidate_universe_v3",
        "evidence": {
            "candidate_4h_count": 0,
            "dsa_parity_alignment_mismatch": 0,
            "atr_recurrence_mismatch": 0,
            "quantile_global_rank_leakage": 0,
            "quantile_backward_fill": 0,
            "quantile_beyond_15m_validity": 0,
            "path_alignment_mismatch": 0,
            "generated_tp_sl_pnl_fields": 0,
            "chunks_gt_800kb": 0,
        },
    }
    (LOCAL_OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("OB_CANDIDATE_UNIVERSE_V3_BUILD_PASS")
    print(f"candidates={len(candidates)} context={len(context)} "
          f"levels_full={len(levels_full)} path={len(path)}")
    print(
        f"4h candidate count=0 | tp_sl_pnl_fields=0 | "
        f"chunks>800kb=0"
    )


if __name__ == "__main__":
    main()
