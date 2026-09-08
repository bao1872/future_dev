#!/usr/bin/env python3

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.build_pytdx_panel import aggregate_15m  # noqa: E402
from research.ob_trigger_snapshot import aggregate_1h_from_15m  # noqa: E402
from research.dsa_adapter import compute_dsa_canonical  # noqa: E402
from panji_indicators import (  # noqa: E402
    ATRRopeConfig,
    DSAConfig,
    MIN_DIR_BARS,
    compute_dsa_history,
)

V2_ROOT = ROOT / "research" / "analysis_data" / "ob_trigger_smc_v2"

RAW_5M_ROOT = ROOT / "research" / "exports" / "v3r_5m"

LOCAL_OUT = ROOT / "research" / "exports" / "ob_trigger_execution_v21"

GIT_OUT = ROOT / "research" / "analysis_data" / "ob_trigger_execution_v21"

SYMBOLS = ("AG", "CU", "RB", "M")

ATR_LENGTH = 14
PATH_BARS = 24

FIVE_MINUTES_NS = 5 * 60 * 1_000_000_000

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
        "1:1 extraction of the market_dev DSA sources above; its "
        "dynamic_swing_anchored_vwap kernel is byte-identical to the "
        "current market_dev kernel (verified at market_dev SHA "
        "8686b803). DSA math is NOT redefined in future_dev."
    ),
    "parity_target": "adapter output == panji_indicators.compute_dsa_history",
}


# ============================================================
# V2 event universe (frozen input)
# ============================================================

def load_v2_symbol(symbol: str) -> pd.DataFrame:
    root = V2_ROOT / symbol
    paths = sorted(root.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"no V2 chunks for {symbol}")

    frames = [
        pd.read_csv(
            p,
            parse_dates=[
                "trigger_time",
                "5m_bar_end",
                "15m_bar_end",
                "1h_bar_end",
            ],
        )
        for p in paths
    ]

    out = pd.concat(frames, ignore_index=True)

    if out["event_id"].duplicated().any():
        dup = (
            out.loc[out["event_id"].duplicated(), "event_id"]
            .head()
            .tolist()
        )
        raise RuntimeError(f"{symbol}: duplicate V2 event_id: {dup}")

    return (
        out.sort_values(["trigger_bar_index", "event_id"])
        .reset_index(drop=True)
    )


# ============================================================
# Raw 5m loader
# ============================================================

def load_raw_5m(symbol: str) -> pd.DataFrame:
    path = RAW_5M_ROOT / f"{symbol}_5m.csv"
    if not path.exists():
        raise FileNotFoundError(path)

    five = pd.read_csv(
        path,
        parse_dates=[
            "bar_start_time",
            "bar_end_time",
            "availability_time",
            "trading_day",
            "tdx_datetime_raw",
        ],
    )
    five = five.sort_values("bar_start_time").reset_index(drop=True)

    if five["bar_start_time"].duplicated().any():
        raise RuntimeError(f"{symbol}: duplicate 5m bars")

    # Keep source columns intact, but provide canonical OHLCV name.
    five["volume"] = five["trade"].astype(float)
    return five


# ============================================================
# ATR: Pine / Wilder RMA (length 14)
# ============================================================

def pine_rma(values: np.ndarray, length: int) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0:
        raise ValueError("length must be positive")
    if n < length:
        return out

    seed = np.mean(x[:length])
    out[length - 1] = seed
    alpha = 1.0 / float(length)

    for i in range(length, n):
        out[i] = alpha * x[i] + (1.0 - alpha) * out[i - 1]
    return out


def true_range(bars: pd.DataFrame) -> np.ndarray:
    high = bars["high"].to_numpy(float)
    low = bars["low"].to_numpy(float)
    close = bars["close"].to_numpy(float)

    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]

    tr = np.maximum.reduce(
        [
            high - low,
            np.abs(high - prev_close),
            np.abs(low - prev_close),
        ]
    )
    return tr


def pine_atr(bars: pd.DataFrame, length: int = 14) -> np.ndarray:
    return pine_rma(true_range(bars), length)


def audit_atr_recurrence(
    bars: pd.DataFrame, atr: np.ndarray, length: int
) -> int:
    tr = true_range(bars)
    if len(tr) < length:
        return 0

    seed_i = length - 1
    expected_seed = float(np.mean(tr[:length]))
    if not np.isclose(atr[seed_i], expected_seed, rtol=1e-12, atol=1e-12):
        raise AssertionError("ATR seed mismatch")

    checks = np.linspace(
        length,
        len(tr) - 1,
        min(1000, len(tr) - length),
        dtype=int,
    )
    for i in checks:
        expected = (atr[i - 1] * (length - 1) + tr[i]) / length
        if not np.isclose(atr[i], expected, rtol=1e-12, atol=1e-12):
            raise AssertionError(f"ATR recurrence mismatch at {i}")
    return int(len(checks))


# ============================================================
# Timeframe construction (no PIT re-join)
# ============================================================

def build_timeframes(
    five: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return fifteen, one_hour


# ============================================================
# ATR snapshot helpers
# ============================================================

def safe_take(values: np.ndarray, index: object) -> float:
    if pd.isna(index):
        return np.nan
    i = int(index)
    if i < 0 or i >= len(values):
        raise IndexError(i)
    return float(values[i])


# ============================================================
# Entry price: lock next 5m open
# ============================================================

def add_entry_fields(
    row: dict, event: pd.Series, five: pd.DataFrame
) -> int | None:
    trigger_i = int(event["trigger_bar_index"])
    entry_i = trigger_i + 1

    if entry_i >= len(five):
        row["entry_bar_index"] = np.nan
        row["entry_time"] = pd.NaT
        row["entry_next_5m_open"] = np.nan
        row["entry_delay_minutes"] = np.nan
        return None

    entry = five.iloc[entry_i]
    entry_time = pd.Timestamp(entry["bar_start_time"])
    trigger_time = pd.Timestamp(event["trigger_time"])

    row["entry_bar_index"] = entry_i
    row["entry_time"] = entry_time
    row["entry_next_5m_open"] = float(entry["open"])
    row["entry_delay_minutes"] = (
        entry_time - trigger_time
    ).total_seconds() / 60.0
    return entry_i


# ============================================================
# Future 24-bar 5m path
# ============================================================

def _write_missing_path_bar(row: dict, step: int) -> None:
    p = f"path_b{step:02d}_"
    for suffix in (
        "bar_index",
        "time",
        "open",
        "high",
        "low",
        "close",
        "gap_minutes",
        "contiguous_from_previous",
        "open_atr",
        "high_atr",
        "low_atr",
        "close_atr",
        "directional_favorable_atr",
        "directional_adverse_atr",
    ):
        row[p + suffix] = np.nan


def add_future_path(
    row: dict,
    *,
    five: pd.DataFrame,
    entry_i: int | None,
    atr5: float,
    direction: int,
) -> None:
    entry_price = row["entry_next_5m_open"]
    if entry_i is None or not np.isfinite(entry_price):
        for step in range(1, PATH_BARS + 1):
            _write_missing_path_bar(row, step)
        return

    if direction not in (-1, 1):
        raise ValueError(f"invalid OB direction: {direction}")

    prev_time = pd.Timestamp(row["entry_time"])

    for step in range(1, PATH_BARS + 1):
        j = entry_i + step - 1
        p = f"path_b{step:02d}_"

        if j >= len(five):
            _write_missing_path_bar(row, step)
            continue

        bar = five.iloc[j]
        ts = pd.Timestamp(bar["bar_start_time"])

        o = float(bar["open"])
        h = float(bar["high"])
        l = float(bar["low"])
        c = float(bar["close"])

        row[p + "bar_index"] = j
        row[p + "time"] = ts
        row[p + "open"] = o
        row[p + "high"] = h
        row[p + "low"] = l
        row[p + "close"] = c

        if step == 1:
            gap_minutes = float(row["entry_delay_minutes"])
        else:
            gap_minutes = (ts - prev_time).total_seconds() / 60.0

        row[p + "gap_minutes"] = gap_minutes
        row[p + "contiguous_from_previous"] = np.isclose(gap_minutes, 5.0)

        if np.isfinite(atr5) and atr5 > 0:
            row[p + "open_atr"] = (o - entry_price) / atr5
            row[p + "high_atr"] = (h - entry_price) / atr5
            row[p + "low_atr"] = (l - entry_price) / atr5
            row[p + "close_atr"] = (c - entry_price) / atr5

            if direction == 1:
                fav = (h - entry_price) / atr5
                adv = (entry_price - l) / atr5
            else:
                fav = (entry_price - l) / atr5
                adv = (h - entry_price) / atr5

            row[p + "directional_favorable_atr"] = fav
            row[p + "directional_adverse_atr"] = adv
        else:
            for suffix in (
                "open_atr",
                "high_atr",
                "low_atr",
                "close_atr",
                "directional_favorable_atr",
                "directional_adverse_atr",
            ):
                row[p + suffix] = np.nan

        prev_time = ts


# ============================================================
# Canonical DSA snapshot
# ============================================================

def add_dsa_snapshot(
    row: dict, *, prefix: str, dsa: pd.DataFrame, bar_index: object
) -> None:
    if pd.isna(bar_index):
        for col in dsa.columns:
            if col == "bar_index":
                continue
            row[f"{prefix}{col}"] = np.nan
        return

    i = int(bar_index)
    d = dsa.iloc[i]
    if int(d["bar_index"]) != i:
        raise AssertionError("DSA bar alignment mismatch")

    for col in dsa.columns:
        if col == "bar_index":
            continue
        row[f"{prefix}{col}"] = d[col]


# ============================================================
# Canonical DSA parity audit (adapter vs panji canonical)
# ============================================================

def audit_dsa_parity(symbol: str, bars: pd.DataFrame) -> int:
    adapter_out = compute_dsa_canonical(bars)

    df = bars.copy()
    for col in ("open", "high", "low", "close", "volume"):
        if col not in df.columns:
            raise ValueError(f"audit_dsa_parity: missing {col}")
    if "amount" not in df.columns:
        df["amount"] = np.nan
    if "bar_start_time" in df.columns:
        df = df.set_index(
            pd.DatetimeIndex(pd.to_datetime(df["bar_start_time"]))
        )
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    config = {
        "dsa_config": DSAConfig(),
        "rope_config": ATRRopeConfig(regime_lookback=55),
        "min_dir_bars": MIN_DIR_BARS,
    }
    hist = compute_dsa_history(df, config).reset_index(drop=True)

    if len(hist) != len(adapter_out):
        raise AssertionError(
            f"{symbol}: DSA parity length mismatch "
            f"{len(hist)} != {len(adapter_out)}"
        )

    exp_dir = hist["regime_value"].fillna(0).astype(int).to_numpy()
    if not np.array_equal(adapter_out["dsa_direction"].to_numpy(), exp_dir):
        raise AssertionError(f"{symbol}: DSA direction parity mismatch")

    mism = 0
    for c in hist.columns:
        if c == "regime_value":
            continue
        exp = hist[c].to_numpy()
        got = adapter_out[f"dsa_raw_{c}"].to_numpy()
        if exp.dtype.kind in "iufc":
            if not np.allclose(exp, got, equal_nan=True, rtol=0, atol=0):
                mism += 1
        else:
            exp_s = pd.Series(exp).fillna("").astype(str).to_numpy()
            got_s = pd.Series(got).fillna("").astype(str).to_numpy()
            if not np.array_equal(exp_s, got_s):
                mism += 1
    if mism:
        raise AssertionError(
            f"{symbol}: DSA canonical parity mismatch = {mism}"
        )
    return int(len(adapter_out))


# ============================================================
# Per-symbol processing
# ============================================================

def process_symbol(symbol: str) -> tuple[pd.DataFrame, dict]:
    events = load_v2_symbol(symbol)
    five = load_raw_5m(symbol)
    fifteen, one_hour = build_timeframes(five)

    # ---------------- ATR ----------------
    atr5 = pine_atr(five, ATR_LENGTH)
    atr15 = pine_atr(fifteen, ATR_LENGTH)
    atr1h = pine_atr(one_hour, ATR_LENGTH)

    audit_atr_recurrence(five, atr5, ATR_LENGTH)
    audit_atr_recurrence(fifteen, atr15, ATR_LENGTH)
    audit_atr_recurrence(one_hour, atr1h, ATR_LENGTH)

    # ---------------- Canonical DSA ----------------
    dsa5 = compute_dsa_canonical(five)
    dsa15 = compute_dsa_canonical(fifteen)
    dsa1h = compute_dsa_canonical(one_hour)

    if not (
        len(dsa5) == len(five)
        and len(dsa15) == len(fifteen)
        and len(dsa1h) == len(one_hour)
    ):
        raise RuntimeError(f"{symbol}: DSA length mismatch")

    rows = []
    for _, event in events.iterrows():
        row = {
            "event_id": event["event_id"],
            "symbol": symbol,
            "trigger_time": event["trigger_time"],
            "trigger_bar_index": int(event["trigger_bar_index"]),
            "trigger_ob_bias": int(event["trigger_ob_bias"]),
            "trigger_ob_structure": event["trigger_ob_structure"],
        }

        i5 = int(event["trigger_bar_index"])
        i15 = event["15m_bar_index"]
        i1h = event["1h_bar_index"]

        a5 = safe_take(atr5, i5)
        a15 = safe_take(atr15, i15)
        a1h = safe_take(atr1h, i1h)

        row["5m_atr14"] = a5
        row["5m_atr14_pct"] = (
            a5 / float(event["5m_close"]) * 100.0
            if np.isfinite(a5)
            else np.nan
        )
        row["15m_atr14"] = a15
        row["15m_atr14_pct"] = (
            a15 / float(event["15m_close"]) * 100.0
            if np.isfinite(a15)
            else np.nan
        )
        row["1h_atr14"] = a1h
        row["1h_atr14_pct"] = (
            a1h / float(event["1h_close"]) * 100.0
            if (np.isfinite(a1h) and pd.notna(event["1h_close"]))
            else np.nan
        )

        # ---- Section 16: ATR / structure execution-scale fields ----
        if np.isfinite(a5) and a5 > 0:
            trigger_width = float(event["trigger_ob_zone_high"]) - float(
                event["trigger_ob_zone_low"]
            )
            row["trigger_ob_width_atr5"] = trigger_width / a5
            for tf in ("5m", "15m", "1h"):
                for side in ("above", "below"):
                    dist_pct = event[
                        f"{tf}_nearest_{side}_distance_pct"
                    ]
                    if pd.isna(dist_pct):
                        row[
                            f"{tf}_nearest_{side}_distance_atr5"
                        ] = np.nan
                        continue
                    price = float(event["5m_close"])
                    distance_price = float(dist_pct) / 100.0 * price
                    row[f"{tf}_nearest_{side}_distance_atr5"] = (
                        distance_price / a5
                    )
        else:
            row["trigger_ob_width_atr5"] = np.nan

        entry_i = add_entry_fields(row, event, five)

        add_dsa_snapshot(row, prefix="5m_", dsa=dsa5, bar_index=i5)
        add_dsa_snapshot(row, prefix="15m_", dsa=dsa15, bar_index=i15)
        add_dsa_snapshot(row, prefix="1h_", dsa=dsa1h, bar_index=i1h)

        add_future_path(
            row,
            five=five,
            entry_i=entry_i,
            atr5=a5,
            direction=int(event["trigger_ob_bias"]),
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    if out["event_id"].duplicated().any():
        raise RuntimeError(f"{symbol}: duplicate supplement event")

    if set(out["event_id"]) != set(events["event_id"]):
        missing = set(events["event_id"]) - set(out["event_id"])
        extra = set(out["event_id"]) - set(events["event_id"])
        raise RuntimeError(
            f"{symbol}: V2 event universe changed "
            f"(missing={len(missing)} extra={len(extra)})"
        )

    # ---- DSA direction distribution (TF x symbol) ----
    dsa_dir_dist = {}
    for tf in ("5m", "15m", "1h"):
        col = f"{tf}_dsa_direction"
        vc = out[col].value_counts(dropna=False).to_dict()
        dsa_dir_dist[tf] = {
            str(int(k)) if pd.notna(k) else "nan": int(v)
            for k, v in vc.items()
        }

    # ---- SMC trigger direction vs DSA direction (counts only) ----
    smc_vs_dsa = {}
    for tf in ("5m", "15m", "1h"):
        ct = out.groupby(
            ["trigger_ob_bias", f"{tf}_dsa_direction"], dropna=False
        ).size().to_dict()
        smc_vs_dsa[tf] = {
            f"{k[0]}/{k[1]}": int(v) for k, v in ct.items()
        }

    stats = {
        "symbol": symbol,
        "events": len(out),
        "atr5_valid": int(out["5m_atr14"].notna().sum()),
        "atr15_valid": int(out["15m_atr14"].notna().sum()),
        "atr1h_valid": int(out["1h_atr14"].notna().sum()),
        "entry_valid": int(out["entry_next_5m_open"].notna().sum()),
        "entry_delay_gt_5m": int(
            (out["entry_delay_minutes"] > 5.0).sum()
        ),
        "b24_available": int(
            out["path_b24_open"].notna().sum()
        ),
        "dsa_direction_distribution": dsa_dir_dist,
        "smc_vs_dsa_direction": smc_vs_dsa,
    }
    return out, stats


# ============================================================
# Path alignment audits (P0)
# ============================================================

def audit_path_alignment(
    supplement: pd.DataFrame, five: pd.DataFrame
) -> int:
    if supplement.empty:
        return 0

    sample_positions = np.linspace(
        0,
        len(supplement) - 1,
        min(1000, len(supplement)),
        dtype=int,
    )

    for k in sample_positions:
        row = supplement.iloc[k]
        ti = int(row["trigger_bar_index"])
        entry_i = ti + 1
        if entry_i >= len(five):
            continue

        expected_entry = five.iloc[entry_i]
        assert int(row["entry_bar_index"]) == entry_i
        assert np.isclose(
            float(row["entry_next_5m_open"]),
            float(expected_entry["open"]),
        )

        for step in (1, 2, 3, 6, 12, 24):
            j = entry_i + step - 1
            p = f"path_b{step:02d}_"
            if j >= len(five):
                assert pd.isna(row[p + "open"])
                continue
            expected = five.iloc[j]
            assert int(row[p + "bar_index"]) == j
            for field in ("open", "high", "low", "close"):
                assert np.isclose(
                    float(row[p + field]),
                    float(expected[field]),
                )
    return int(len(sample_positions))


def audit_normalized_path(supplement: pd.DataFrame) -> int:
    valid = supplement[
        supplement["5m_atr14"].notna()
        & supplement["entry_next_5m_open"].notna()
    ]
    if valid.empty:
        return 0

    sample = valid.iloc[
        np.linspace(
            0, len(valid) - 1, min(1000, len(valid)), dtype=int
        )
    ]
    for _, row in sample.iterrows():
        atr = float(row["5m_atr14"])
        entry = float(row["entry_next_5m_open"])
        for step in (1, 3, 6, 12, 24):
            p = f"path_b{step:02d}_"
            if pd.isna(row[p + "high"]):
                continue
            expected_high = (float(row[p + "high"]) - entry) / atr
            expected_low = (float(row[p + "low"]) - entry) / atr
            assert np.isclose(row[p + "high_atr"], expected_high)
            assert np.isclose(row[p + "low_atr"], expected_low)
    return int(len(sample))


# ============================================================
# Git chunking
# ============================================================

def write_git_chunks(
    df: pd.DataFrame, symbol: str, *, chunk_rows: int = 100
) -> list[dict]:
    root = GIT_OUT / symbol
    root.mkdir(parents=True, exist_ok=True)

    result = []
    for start in range(0, len(df), chunk_rows):
        end = min(start + chunk_rows, len(df))
        path = root / (
            f"{symbol}_{start:04d}_{end - 1:04d}.csv"
        )
        part = df.iloc[start:end]
        part.to_csv(path, index=False)
        size = path.stat().st_size
        if size >= 900_000:
            raise RuntimeError(f"{path} too large: {size} bytes")
        result.append(
            {
                "file": str(path.relative_to(ROOT)),
                "row_start": start,
                "row_end": end - 1,
                "rows": len(part),
                "bytes": size,
            }
        )
    return result


# ============================================================
# Main
# ============================================================

def main() -> None:
    LOCAL_OUT.mkdir(parents=True, exist_ok=True)
    GIT_OUT.mkdir(parents=True, exist_ok=True)

    # hard guard: refuse to overwrite an existing V2.1 tree
    existing_git = list(GIT_OUT.glob("**/*.csv"))
    if existing_git:
        raise RuntimeError(
            f"V2.1 git tree already present ({len(existing_git)} files); "
            "delete intentionally before rerun"
        )

    supplements: list[pd.DataFrame] = []
    stats_all: list[dict] = []
    chunk_manifests: list[dict] = []
    atr_audit = {}
    dsa_audit = {}
    path_audit = {}
    norm_audit = {}

    v2_event_total = 0

    for symbol in SYMBOLS:
        out, stats = process_symbol(symbol)
        five = load_raw_5m(symbol)

        supplements.append(out)
        stats_all.append(stats)
        v2_event_total += len(out)

        # audits
        atr_audit[symbol] = {
            "5m": int(audit_atr_recurrence(five, pine_atr(five, ATR_LENGTH), ATR_LENGTH)),
        }
        dsa_audit[symbol] = int(
            audit_dsa_parity(symbol, five)
        )
        path_audit[symbol] = int(audit_path_alignment(out, five))
        norm_audit[symbol] = int(audit_normalized_path(out))

        chunks = write_git_chunks(out, symbol, chunk_rows=100)
        chunk_manifests.append({"symbol": symbol, "chunks": chunks})

        (LOCAL_OUT / f"{symbol}_execution_supplement.csv").write_text(
            out.to_csv(index=False), encoding="utf-8"
        )

    supplement = pd.concat(supplements, ignore_index=True)

    # ---- local full combined CSV ----
    (LOCAL_OUT / "execution_supplement.csv").write_text(
        supplement.to_csv(index=False), encoding="utf-8"
    )

    total_rows = len(supplement)
    if total_rows != v2_event_total:
        raise RuntimeError(
            f"TOTAL supplement rows {total_rows} != "
            f"V2 event total {v2_event_total}"
        )

    # ---- schema.json ----
    schema = {
        "schema_version": "ob_trigger_execution_v21",
        "join_key": "event_id",
        "base_dataset": {
            "version": "ob_trigger_smc_v2",
            "git_sha": "a0c0d6ceaf4b35ce25688845c68728bd67b933a9",
        },
        "roles": {
            "causal_features": [
                "ATR fields",
                "DSA fields",
                "entry metadata known at entry",
            ],
            "outcome_only": [
                "path_b01_* through path_b24_*",
            ],
        },
        "execution_semantics": {
            "signal_known": "5m trigger bar completion",
            "entry": "next 5m bar open",
            "atr": (
                "Pine/Wilder ATR14 using data available through "
                "the snapshot bar"
            ),
            "path_b01": "entry bar itself",
        },
        "dsa": DSA_SOURCE,
        "columns": list(supplement.columns),
        "dsa_exported_fields": [
            c for c in supplement.columns if c.startswith("dsa_raw_")
        ],
    }
    (GIT_OUT / "schema.json").write_text(
        json.dumps(schema, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- summary.json ----
    atr_valid = {
        s["symbol"]: {
            "5m": s["atr5_valid"],
            "15m": s["atr15_valid"],
            "1h": s["atr1h_valid"],
        }
        for s in stats_all
    }
    entry_delay_gt5 = {
        s["symbol"]: s["entry_delay_gt_5m"] for s in stats_all
    }
    b24_avail = {s["symbol"]: s["b24_available"] for s in stats_all}
    dsa_dist = {
        s["symbol"]: s["dsa_direction_distribution"] for s in stats_all
    }
    smc_dsa = {s["symbol"]: s["smc_vs_dsa_direction"] for s in stats_all}

    max_chunk = max(
        (c["bytes"] for m in chunk_manifests for c in m["chunks"]),
        default=0,
    )
    big_chunks = [
        c["file"]
        for m in chunk_manifests
        for c in m["chunks"]
        if c["bytes"] >= 900_000
    ]

    tp_sl_fields = [
        c
        for c in supplement.columns
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
            )
        )
    ]

    summary = {
        "schema_version": "ob_trigger_execution_v21",
        "base_dataset": {
            "version": "ob_trigger_smc_v2",
            "git_sha": "a0c0d6ceaf4b35ce25688845c68728bd67b933a9",
        },
        "totals": {
            "v2_event_rows": int(v2_event_total),
            "supplement_rows": int(total_rows),
        },
        "event_parity": {
            "missing_v2_event_id": 0,
            "extra_event_id": 0,
            "duplicate_event_id": 0,
        },
        "dsa": DSA_SOURCE,
        "dsa_canonical_parity_mismatch": 0,
        "atr_recurrence_mismatch": 0,
        "atr_valid_by_symbol_tf": atr_valid,
        "next_open_alignment_mismatch": 0,
        "path_alignment_mismatch": 0,
        "atr_normalized_path_mismatch": 0,
        "entry_delay_gt_5m_rate": {
            s["symbol"]: round(
                s["entry_delay_gt_5m"] / max(s["events"], 1), 6
            )
            for s in stats_all
        },
        "entry_delay_gt_5m_count": entry_delay_gt5,
        "future_b24_available_rate": {
            s["symbol"]: round(
                s["b24_available"] / max(s["events"], 1), 6
            )
            for s in stats_all
        },
        "future_b24_available_count": b24_avail,
        "dsa_direction_distribution": dsa_dist,
        "smc_trigger_vs_dsa_direction": smc_dsa,
        "audit_counts": {
            "atr_recurrence_checked": atr_audit,
            "dsa_parity_checked": dsa_audit,
            "path_alignment_checked": path_audit,
            "normalized_path_checked": norm_audit,
        },
        "git_data": {
            "chunks_by_symbol": {
                m["symbol"]: len(m["chunks"]) for m in chunk_manifests
            },
            "max_chunk_bytes": int(max_chunk),
            "chunks_gt_900kb": len(big_chunks),
        },
        "generated_tp_sl_pnl_fields": len(tp_sl_fields),
        "by_symbol": stats_all,
        "chunk_manifests": chunk_manifests,
    }
    (GIT_OUT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- local manifest ----
    manifest = {
        "status": "PASS",
        "schema_version": "ob_trigger_execution_v21",
        "base_dataset_sha": "a0c0d6ceaf4b35ce25688845c68728bd67b933a9",
        "dsa": DSA_SOURCE,
        "local_exports": [
            "research/exports/ob_trigger_execution_v21/execution_supplement.csv",
            "research/exports/ob_trigger_execution_v21/manifest.json",
        ],
        "git_data": "research/analysis_data/ob_trigger_execution_v21",
        "totals": summary["totals"],
        "evidence": {
            "v2_event_rows": int(v2_event_total),
            "supplement_rows": int(total_rows),
            "missing_v2_event_id": 0,
            "extra_event_id": 0,
            "duplicate_event_id": 0,
            "dsa_canonical_parity_mismatch": 0,
            "atr_recurrence_mismatch": 0,
            "next_open_alignment_mismatch": 0,
            "path_alignment_mismatch": 0,
            "atr_normalized_path_mismatch": 0,
            "chunks_gt_900kb": len(big_chunks),
            "generated_tp_sl_pnl_fields": len(tp_sl_fields),
        },
    }
    (LOCAL_OUT / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("OB_TRIGGER_EXECUTION_V21_BUILD_PASS")
    print(
        f"v2_event_rows={v2_event_total} "
        f"supplement_rows={total_rows}"
    )
    print(
        f"dsa_parity_mismatch=0 atr_recurrence=0 "
        f"path_alignment=0 normalized_path=0 "
        f"chunks_gt_900kb={len(big_chunks)} "
        f"tp_sl_pnl_fields={len(tp_sl_fields)}"
    )


if __name__ == "__main__":
    main()
