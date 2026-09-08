#!/usr/bin/env python3

"""Phase A-D post-hoc audit for OB Candidate Universe V3.

Independent re-verification of the generated future path against raw 5m
data, corrected execution-continuity semantics, and derivation of the
source-OB canonical lifecycle state at touch close.

The V3 dataset is FROZEN (read-only). This script never rewrites V3
outputs; it only reads them and emits audit artifacts.
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

V3_LOCAL = ROOT / "research" / "exports" / "ob_candidate_universe_v3"
V3_GIT = ROOT / "research" / "analysis_data" / "ob_candidate_universe_v3"
RESULTS = ROOT / "research" / "analysis_results" / "ob_candidate_v3_phase1"
CACHE = V3_LOCAL / "_phase1_lifecycle_cache.json"

from research.export_ob_trigger_execution_v21 import (  # noqa: E402
    load_raw_5m,
)
from research.build_ob_candidate_universe_v3 import (  # noqa: E402
    replay_ob_lifetimes,
    _ob_identity,
)
from research.ob_trigger_snapshot import (  # noqa: E402
    aggregate_1h_from_15m,
    build_full_ob_smc_tf,
)
from research.build_pytdx_panel import aggregate_15m  # noqa: E402

LOCAL_NAMES = {
    "candidates": "candidates.csv",
    "context": "context.csv",
    "levels": "levels_full.csv",
    "path": "future_path.csv",
    "quantile": "quantile_state.csv",
}

HORIZONS = (6, 12, 24)
SOURCE_TFS = ("5m", "15m", "1h")


# ============================================================
# Phase A - loader
# ============================================================

def load_full_or_chunks(table: str) -> pd.DataFrame:
    p = V3_LOCAL / LOCAL_NAMES[table]
    if p.exists():
        return pd.read_csv(p)

    paths = sorted((V3_GIT / table).glob("*/*.csv"))
    if not paths:
        raise FileNotFoundError(table)
    return pd.concat(
        [pd.read_csv(x) for x in paths],
        ignore_index=True,
    )


def load_raw_five(symbol: str) -> pd.DataFrame:
    return load_raw_5m(symbol)


def build_source_bars(symbol: str) -> dict[str, pd.DataFrame]:
    five = load_raw_five(symbol)
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
    return {"5m": five, "15m": fifteen, "1h": one_hour}


# ============================================================
# Phase D - canonical lifecycle (rebuilt, cached)
# ============================================================

def _key_str(k) -> str:
    return "|".join(str(x) for x in k)


def build_lifecycle(
    symbol: str,
    use_cache: bool = True,
) -> dict:
    cache: dict = {}
    if CACHE.exists():
        try:
            cache = json.loads(CACHE.read_text())
        except Exception:
            cache = {}

    if use_cache and symbol in cache:
        return cache[symbol]

    bars_by_tf = build_source_bars(symbol)
    out: dict = {}
    for tf in SOURCE_TFS:
        bars = bars_by_tf[tf]
        smc = build_full_ob_smc_tf(bars)
        lives = replay_ob_lifetimes(smc, len(bars))
        out[tf] = {
            _key_str(_ob_identity(l)): {
                "confirmed_index": int(l["confirmed_index"]),
                "inactive_index": (
                    int(l["inactive_index"])
                    if l["inactive_index"] is not None
                    else None
                ),
                "inactive_reason": l["inactive_reason"],
            }
            for l in lives
        }
        print(
            f"[{symbol}] {tf} lifecycle: {len(lives)} obs",
            flush=True,
        )

    cache[symbol] = out
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    return out


def source_state_at_touch(
    candidate,
    life,
    source_bars,
) -> dict:
    if life["inactive_index"] is None:
        return {
            "source_ob_active_at_touch_close": True,
            "source_ob_mitigated_at_touch_close": False,
            "source_ob_cap_evicted_at_touch_close": False,
        }

    inactive_time = pd.Timestamp(
        source_bars.iloc[
            int(life["inactive_index"])
        ]["bar_end_time"]
    )
    touch_time = pd.Timestamp(candidate["touch_time"])
    inactive_now = inactive_time <= touch_time

    return {
        "source_ob_active_at_touch_close": not inactive_now,
        "source_ob_mitigated_at_touch_close": (
            inactive_now
            and life["inactive_reason"] == "mitigated"
        ),
        "source_ob_cap_evicted_at_touch_close": (
            inactive_now
            and life["inactive_reason"] == "cap_evicted"
        ),
    }


# ============================================================
# Phase B - independent path audit
# ============================================================

def audit_path(
    candidates: pd.DataFrame,
    path: pd.DataFrame,
    raw_five: dict[str, pd.DataFrame],
) -> dict:
    expected_rows = len(candidates) * 24
    if len(path) != expected_rows:
        raise RuntimeError(
            "path cardinality mismatch: "
            f"{len(path)} != {expected_rows}"
        )

    audit = {
        "path_rows": int(len(path)),
        "expected_rows": int(expected_rows),
        "cardinality_mismatch": False,
        "bar_index_mismatch": 0,
        "ohlc_mismatch": 0,
        "atr_normalized_mismatch": 0,
    }

    meta = candidates[
        [
            "candidate_id",
            "symbol",
            "touch_5m_bar_index",
            "entry_next_5m_open",
            "5m_atr14",
        ]
    ]

    x = path.merge(
        meta,
        on=["candidate_id", "symbol"],
        validate="many_to_one",
    )

    for symbol, sub in x.groupby("symbol"):
        five = raw_five[symbol]

        expected_index = (
            sub["touch_5m_bar_index"].astype(int)
            + sub["step"].astype(int)
        )
        valid = expected_index < len(five)

        got_index = sub.loc[valid, "bar_index"].astype(int)
        exp_index = expected_index[valid]

        audit["bar_index_mismatch"] += int(
            (
                got_index.to_numpy()
                != exp_index.to_numpy()
            ).sum()
        )

        raw = five.iloc[exp_index.to_numpy()]
        got = sub.loc[valid]

        for col in ("open", "high", "low", "close"):
            audit["ohlc_mismatch"] += int(
                (
                    ~np.isclose(
                        got[col].to_numpy(float),
                        raw[col].to_numpy(float),
                    )
                ).sum()
            )

        # Independent ATR-normalized verification.
        entry = got["entry_next_5m_open"].to_numpy(float)
        atr = got["5m_atr14"].to_numpy(float)
        ok = (
            np.isfinite(entry)
            & np.isfinite(atr)
            & (atr > 0)
        )
        if ok.any():
            for col in ("open", "high", "low", "close"):
                exp_norm = (
                    raw[col].to_numpy(float)[ok] - entry[ok]
                ) / atr[ok]
                audit["atr_normalized_mismatch"] += int(
                    (
                        ~np.isclose(
                            got[col + "_atr"].to_numpy(float)[ok],
                            exp_norm,
                        )
                    ).sum()
                )

    return audit


# ============================================================
# Phase C - corrected continuity (does not modify V3 data)
# ============================================================

def add_true_continuity(path: pd.DataFrame) -> pd.DataFrame:
    x = path.copy()
    x["execution_contiguous"] = np.where(
        x["step"] == 1,
        np.isclose(x["gap_minutes"], 0.0),
        np.isclose(x["gap_minutes"], 5.0),
    )
    return x


def horizon_continuity(
    path: pd.DataFrame,
    h: int,
) -> pd.Series:
    return (
        path[path["step"] <= h]
        .groupby("candidate_id")["execution_contiguous"]
        .all()
    )


def touch_behavior(row) -> str:
    if bool(row["touch_close_beyond_far_edge"]):
        return "CLOSE_BEYOND"
    if bool(row["touch_intrabar_far_edge_breach"]):
        return "BREACH_RECLAIM"
    return "NO_BREACH"


def quant_bin(pct) -> str:
    if pd.isna(pct):
        return "UNKNOWN"
    if pct <= 0.30:
        return "LOW"
    if pct >= 0.70:
        return "HIGH"
    return "MID"


def touch_bin(ordinal) -> str:
    o = int(ordinal)
    if o == 1:
        return "1"
    if o == 2:
        return "2"
    if o == 3:
        return "3"
    return "4+"


# ============================================================
# Main
# ============================================================

def main() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)

    candidates = load_full_or_chunks("candidates")
    path = load_full_or_chunks("path")

    symbols = sorted(candidates["symbol"].unique().tolist())
    raw_five = {s: load_raw_five(s) for s in symbols}

    print("Phase B: independent path audit", flush=True)
    audit = audit_path(candidates, path, raw_five)

    print("Phase C: corrected continuity", flush=True)
    path = add_true_continuity(path)

    step1 = path[path["step"] == 1]
    step_ge2 = path[path["step"] >= 2]

    contig_rates = {}
    for h in HORIZONS:
        hc = horizon_continuity(path, h)
        contig_rates[f"continuity_h{h}_rate"] = float(
            hc.reindex(candidates["candidate_id"]).fillna(False).mean()
        )

    legacy_all = (
        path.groupby("candidate_id")["contiguous_from_previous"]
        .all()
    )

    audit["continuity"] = {
        "corrected_step1_gap0_rate": float(
            np.isclose(
                step1["gap_minutes"].to_numpy(float), 0.0
            ).mean()
        ),
        "corrected_step_ge2_gap5_rate": float(
            np.isclose(
                step_ge2["gap_minutes"].to_numpy(float), 5.0
            ).mean()
        ),
        "legacy_contiguous_all_rate": float(
            legacy_all.reindex(
                candidates["candidate_id"]
            ).fillna(False).mean()
        ),
        **contig_rates,
    }

    print("Phase D: source OB lifecycle state", flush=True)
    ann = candidates[
        [
            "candidate_id",
            "symbol",
            "source_tf",
            "source_anchor_index",
            "source_confirmed_index",
            "source_ob_bias",
            "source_ob_internal",
            "touch_time",
            "touch_ordinal",
            "touch_intrabar_far_edge_breach",
            "touch_close_beyond_far_edge",
            "touch_reclaimed_by_close",
            "quant_width_percentile_train",
        ]
    ].copy()

    states = []
    missing = 0
    for symbol in symbols:
        lives_by_tf = build_lifecycle(symbol)
        bars_by_tf = build_source_bars(symbol)
        sub = ann[ann["symbol"] == symbol]

        for _, c in sub.iterrows():
            key = _key_str(
                (
                    int(c["source_anchor_index"]),
                    int(c["source_confirmed_index"]),
                    int(c["source_ob_bias"]),
                    bool(c["source_ob_internal"]),
                )
            )
            life = lives_by_tf[c["source_tf"]].get(key)
            if life is None:
                missing += 1
                states.append(
                    {
                        "candidate_id": c["candidate_id"],
                        "source_ob_active_at_touch_close": np.nan,
                        "source_ob_mitigated_at_touch_close": np.nan,
                        "source_ob_cap_evicted_at_touch_close": np.nan,
                    }
                )
                continue
            st = source_state_at_touch(
                c, life, bars_by_tf[c["source_tf"]]
            )
            st["candidate_id"] = c["candidate_id"]
            states.append(st)

    audit["n_missing_life"] = int(missing)
    state_df = pd.DataFrame(states)
    ann = ann.merge(state_df, on="candidate_id", how="left")

    for h in HORIZONS:
        hc = horizon_continuity(path, h)
        ann[f"contig_h{h}"] = (
            hc.reindex(ann["candidate_id"])
            .fillna(False)
            .to_numpy()
        )

    ann["touch_bin"] = ann["touch_ordinal"].map(touch_bin)
    ann["touch_behavior"] = ann.apply(touch_behavior, axis=1)
    ann["quant_bin"] = ann["quant_width_percentile_train"].map(
        quant_bin
    )

    ann_out = ann[
        [
            "candidate_id",
            "symbol",
            "source_tf",
            "source_ob_active_at_touch_close",
            "source_ob_mitigated_at_touch_close",
            "source_ob_cap_evicted_at_touch_close",
            "contig_h6",
            "contig_h12",
            "contig_h24",
            "touch_bin",
            "touch_behavior",
            "quant_bin",
        ]
    ]

    # Source-OB state counts by source_tf.
    by_tf = {}
    for tf in SOURCE_TFS:
        s = ann_out[ann_out["source_tf"] == tf]
        by_tf[tf] = {
            "n": int(len(s)),
            "active_at_touch": int(
                s["source_ob_active_at_touch_close"].sum()
            ),
            "mitigated_at_touch": int(
                s["source_ob_mitigated_at_touch_close"].sum()
            ),
            "cap_evicted_at_touch": int(
                s["source_ob_cap_evicted_at_touch_close"].sum()
            ),
        }
    audit["source_ob_state_by_tf"] = by_tf

    failed = (
        audit["bar_index_mismatch"]
        or audit["ohlc_mismatch"]
        or audit["atr_normalized_mismatch"]
        or audit["n_missing_life"]
    )
    audit["status"] = "FAIL" if failed else "PASS"

    (RESULTS / "posthoc_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    ann_out.to_csv(
        RESULTS / "candidate_annotations.csv", index=False
    )

    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print(
        "POSTHOC_AUDIT_DONE status=%s" % audit["status"],
        flush=True,
    )


if __name__ == "__main__":
    main()
