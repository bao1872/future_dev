"""Trade-entry label return audit (Checkpoint A) for the 15m mechanical R2 port.

TASK ID: FUTURE-R4-M15-DP-TRADE-RETURN-DISTRIBUTION-V1
SCOPE:  Audit + derived-label study ONLY.

DO NOT (per task):
  - modify DP Bellman / dp_proximity / trade reconstruction / Candidate
  - delete original Oracle trades
  - retrain a model

This module only READS the new 15m R2 port outputs
(``oracle_trades.parquet`` / ``oracle_actions.parquet``) and the canonical 15m
environment, and produces a *derived* trade-return analysis frame plus a set of
summary CSV/JSON files. The raw Oracle path, entry/exit and full 6x3 Q table are
never touched.

ATR is taken from the canonical 15m environment (``features.m15_atr``); this
module never recomputes ATR.

Outputs (local-only parquet + small evidence CSV/JSON) under
``artifacts/dp_m15_trade_return_audit_v1/<SYMBOL>/``:
  trade_returns.parquet   (local only, not committed)
  quantiles.csv
  histogram.csv
  threshold_sweep.csv
  holding_bins.csv
  direction_stats.csv
  contribution_curve.csv
  summary.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from research.liquidity_oracle_atlas.build_execution_environment_m15_v1 import (
    run_environment_m15,
)
from research.liquidity_oracle_atlas.build_trade_oracle_dp_m15_one_entry_proximity_v1 import (
    MATH_VERSION,
    build_artifact_frames_v1,
    load_oracle_artifact,
    run_dp_m15_one_entry_proximity,
)

# Fixed threshold sweep (ATR units), per task spec.
THRESHOLDS_ATR = np.array(
    [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.33, 0.50, 0.75, 1.00]
)

# Fixed ATR histogram bins (lo, hi, label). hi is exclusive except last.
_HIST_BINS: List[Tuple[float, float, str]] = [
    (-np.inf, 0.0, "<0"),
    (0.0, 0.05, "0-0.05"),
    (0.05, 0.10, "0.05-0.10"),
    (0.10, 0.20, "0.10-0.20"),
    (0.20, 0.30, "0.20-0.30"),
    (0.30, 0.50, "0.30-0.50"),
    (0.50, 0.75, "0.50-0.75"),
    (0.75, 1.00, "0.75-1.00"),
    (1.00, 1.50, "1.00-1.50"),
    (1.50, 2.00, "1.50-2.00"),
    (2.00, np.inf, ">2.00"),
]

# Holding-bar bins.
_HOLD_BINS: List[Tuple[int, int, str]] = [
    (1, 1, "1"),
    (2, 2, "2"),
    (3, 3, "3"),
    (4, 4, "4"),
    (5, 8, "5-8"),
    (9, 16, "9-16"),
    (17, 32, "17-32"),
    (33, 10 ** 9, ">32"),
]

_QUANTILES: List[Tuple[str, Any]] = [
    ("min", lambda s: float(np.min(s))),
    ("p01", lambda s: float(np.percentile(s, 1))),
    ("p05", lambda s: float(np.percentile(s, 5))),
    ("p10", lambda s: float(np.percentile(s, 10))),
    ("p20", lambda s: float(np.percentile(s, 20))),
    ("p25", lambda s: float(np.percentile(s, 25))),
    ("p33", lambda s: float(np.percentile(s, 33))),
    ("p50", lambda s: float(np.percentile(s, 50))),
    ("p67", lambda s: float(np.percentile(s, 67))),
    ("p75", lambda s: float(np.percentile(s, 75))),
    ("p80", lambda s: float(np.percentile(s, 80))),
    ("p90", lambda s: float(np.percentile(s, 90))),
    ("p95", lambda s: float(np.percentile(s, 95))),
    ("p99", lambda s: float(np.percentile(s, 99))),
    ("max", lambda s: float(np.max(s))),
    ("mean", lambda s: float(np.mean(s))),
    ("std", lambda s: float(np.std(s))),
]


def build_trade_return_frame(
    trades: pd.DataFrame,
    actions: pd.DataFrame,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Build one row per Oracle trade with ATR-normalised returns.

    ATR is pulled from the canonical 15m environment (``features.m15_atr``), keyed
    by the shared ``execution_bar_index`` == ``entry_decision_index``. The module
    never recomputes ATR.

    ``entry_type`` (FROM_FLAT / REVERSAL) is derived from the *position before*
    the entry decision (joined from the Oracle actions), without modifying the
    trade-reconstruction output.
    """
    t = trades.copy()

    # --- ATR from canonical 15m environment -------------------------------- #
    atr = (
        features[["execution_bar_index", "m15_atr"]]
        .rename(
            columns={
                "execution_bar_index": "entry_decision_index",
                "m15_atr": "entry_atr15",
            }
        )
    )
    t = t.merge(
        atr,
        on="entry_decision_index",
        how="left",
        validate="many_to_one",
    )
    if t["entry_atr15"].isna().any():
        raise RuntimeError("STOP_TRADE_RETURN_ATR_ALIGNMENT")
    if (t["entry_atr15"] <= 0).any():
        raise RuntimeError("STOP_TRADE_RETURN_BAD_ATR")

    # --- ATR-normalised returns -------------------------------------------- #
    t["gross_atr"] = t["gross_points"] / t["entry_atr15"]
    t["net_atr"] = t["net_points"] / t["entry_atr15"]
    t["MFE_atr"] = t["MFE"] / t["entry_atr15"]
    t["MAE_atr"] = t["MAE"] / t["entry_atr15"]

    # --- entry_type from position_before (no trade-recon change) ----------- #
    pos = (
        actions[["decision_bar_index", "position_before"]]
        .rename(
            columns={
                "decision_bar_index": "entry_decision_index",
                "position_before": "entry_pos_before",
            }
        )
    )
    t = t.merge(
        pos,
        on="entry_decision_index",
        how="left",
        validate="many_to_one",
    )
    if t["entry_pos_before"].isna().any():
        raise RuntimeError("STOP_TRADE_RETURN_POSITION_ALIGNMENT")
    t["entry_type"] = np.where(
        t["entry_pos_before"] == 0, "FROM_FLAT", "REVERSAL"
    )

    # --- rename DP-internal episode id (no Candidate coupling) ------------- #
    t = t.rename(columns={"entry_proximity_episode_id": "dp_proximity_episode_id"})

    cols = [
        "trade_id",
        "direction",
        "entry_fill_time",
        "exit_fill_time",
        "holding_bars",
        "gross_points",
        "net_points",
        "entry_atr15",
        "gross_atr",
        "net_atr",
        "MFE_atr",
        "MAE_atr",
        "entry_type",
        "dp_proximity_episode_id",
    ]
    return t[cols]


def _group_quantiles(df: pd.DataFrame, col: str = "net_atr") -> pd.DataFrame:
    groups = {
        "ALL": df,
        "LONG": df[df["direction"] == "LONG"],
        "SHORT": df[df["direction"] == "SHORT"],
        "FROM_FLAT": df[df["entry_type"] == "FROM_FLAT"],
        "REVERSAL": df[df["entry_type"] == "REVERSAL"],
    }
    rows = []
    for name, sub in groups.items():
        s = sub[col].to_numpy(dtype=float)
        row: Dict[str, Any] = {"group": name, "n": int(len(s))}
        if len(s) == 0:
            for qlabel, _ in _QUANTILES:
                row[qlabel] = float("nan")
        else:
            for qlabel, fn in _QUANTILES:
                row[qlabel] = fn(s)
        rows.append(row)
    return pd.DataFrame(rows)


def _histogram(df: pd.DataFrame, col: str = "net_atr") -> pd.DataFrame:
    s = df[col].to_numpy(dtype=float)
    rows = []
    total = float(len(s))
    cum = 0.0
    for lo, hi, label in _HIST_BINS:
        mask = (s >= lo) & (s < hi)
        cnt = int(mask.sum())
        cum += cnt
        rows.append(
            {
                "bin": label,
                "lo": lo,
                "hi": hi,
                "count": cnt,
                "frac": cnt / total if total else 0.0,
                "cum_frac": cum / total if total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def _threshold_sweep(df: pd.DataFrame, col: str = "net_atr") -> pd.DataFrame:
    s = df[col].to_numpy(dtype=float)
    total = float(len(s))
    total_gross = float(np.sum(df["gross_points"].to_numpy(dtype=float)))
    rows = []
    for tau in THRESHOLDS_ATR:
        keep = s >= tau
        removed = ~keep
        kept = df[keep]
        removed_df = df[removed]
        removed_gross = float(
            np.sum(removed_df["gross_points"].to_numpy(dtype=float))
        )
        rows.append(
            {
                "threshold_atr": float(tau),
                "total_trades": int(total),
                "kept_trades": int(keep.sum()),
                "removed_trades": int(removed.sum()),
                "keep_rate": float(keep.mean()) if total else 0.0,
                "removed_gross_points": removed_gross,
                "removed_pnl_fraction": (removed_gross / total_gross)
                if total_gross
                else 0.0,
                "kept_mean_atr": float(np.mean(s[keep])) if keep.any() else float("nan"),
                "kept_median_atr": float(np.median(s[keep])) if keep.any() else float("nan"),
                "kept_total_points": float(
                    np.sum(kept["gross_points"].to_numpy(dtype=float))
                ),
                "LONG_keep_rate": float(
                    (df[df["direction"] == "LONG"][col].to_numpy(dtype=float) >= tau).mean()
                )
                if (df["direction"] == "LONG").any()
                else float("nan"),
                "SHORT_keep_rate": float(
                    (df[df["direction"] == "SHORT"][col].to_numpy(dtype=float) >= tau).mean()
                )
                if (df["direction"] == "SHORT").any()
                else float("nan"),
                "REV_keep_rate": float(
                    (df[df["entry_type"] == "REVERSAL"][col].to_numpy(dtype=float) >= tau).mean()
                )
                if (df["entry_type"] == "REVERSAL").any()
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _holding_bins(df: pd.DataFrame, col: str = "net_atr") -> pd.DataFrame:
    rows = []
    for lo, hi, label in _HOLD_BINS:
        sub = df[(df["holding_bars"] >= lo) & (df["holding_bars"] <= hi)]
        s = sub[col].to_numpy(dtype=float)
        rows.append(
            {
                "holding_bin": label,
                "n": int(len(sub)),
                "mean_net_atr": float(np.mean(s)) if len(s) else float("nan"),
                "median_net_atr": float(np.median(s)) if len(s) else float("nan"),
                "total_gross_points": float(
                    np.sum(sub["gross_points"].to_numpy(dtype=float))
                )
                if len(sub)
                else 0.0,
                "total_net_points": float(
                    np.sum(sub["net_points"].to_numpy(dtype=float))
                )
                if len(sub)
                else 0.0,
                "win_rate": float((s > 0).mean()) if len(s) else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _direction_stats(df: pd.DataFrame, col: str = "net_atr") -> pd.DataFrame:
    groups = {
        "LONG": df[df["direction"] == "LONG"],
        "SHORT": df[df["direction"] == "SHORT"],
        "FROM_FLAT": df[df["entry_type"] == "FROM_FLAT"],
        "REVERSAL": df[df["entry_type"] == "REVERSAL"],
    }
    rows = []
    for name, sub in groups.items():
        s = sub[col].to_numpy(dtype=float)
        rows.append(
            {
                "group": name,
                "n": int(len(sub)),
                "mean_net_atr": float(np.mean(s)) if len(s) else float("nan"),
                "median_net_atr": float(np.median(s)) if len(s) else float("nan"),
                "total_gross_points": float(
                    np.sum(sub["gross_points"].to_numpy(dtype=float))
                )
                if len(sub)
                else 0.0,
                "total_net_points": float(
                    np.sum(sub["net_points"].to_numpy(dtype=float))
                )
                if len(sub)
                else 0.0,
                "win_rate": float((s > 0).mean()) if len(s) else float("nan"),
                "mean_holding_bars": float(sub["holding_bars"].mean())
                if len(sub)
                else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def _contribution_curve(df: pd.DataFrame, col: str = "net_atr") -> pd.DataFrame:
    s = df[col].to_numpy(dtype=float)
    pnl = df["net_points"].to_numpy(dtype=float)
    order = np.argsort(s)
    s_sorted = s[order]
    pnl_sorted = pnl[order]
    total = float(len(s_sorted))
    total_pnl = float(np.sum(pnl_sorted))
    cum_trade = (np.arange(1, len(s_sorted) + 1)) / total if total else np.array([])
    cum_pnl = np.cumsum(pnl_sorted) / total_pnl if total_pnl else cum_trade * 0.0
    return pd.DataFrame(
        {
            "rank": np.arange(1, len(s_sorted) + 1),
            "net_atr": s_sorted,
            "net_points": pnl_sorted,
            "cum_trade_frac": cum_trade,
            "cum_pnl_frac": cum_pnl,
        }
    )


def _summary(df: pd.DataFrame) -> Dict[str, Any]:
    s = df["net_atr"].to_numpy(dtype=float)
    total = float(len(s))
    total_gross = float(np.sum(df["gross_points"].to_numpy(dtype=float)))

    def low_share(thr: float) -> Dict[str, float]:
        mask = s < thr
        cnt = int(mask.sum())
        rem_gross = float(np.sum(df["gross_points"].to_numpy(dtype=float)[mask]))
        return {
            "pct_trades": cnt / total if total else 0.0,
            "pnl_contribution_pct": (rem_gross / total_gross) if total_gross else 0.0,
        }

    def grp(name: str) -> Dict[str, float]:
        sub = df[df["direction"] == name] if name in ("LONG", "SHORT") else df[df["entry_type"] == name]
        ss = sub["net_atr"].to_numpy(dtype=float)
        return {
            "n": int(len(sub)),
            "mean_net_atr": float(np.mean(ss)) if len(ss) else float("nan"),
            "median_net_atr": float(np.median(ss)) if len(ss) else float("nan"),
        }

    out: Dict[str, Any] = {
        "trade_count": int(total),
        "mean_net_atr": float(np.mean(s)),
        "median_net_atr": float(np.median(s)),
        "p10": float(np.percentile(s, 10)),
        "p20": float(np.percentile(s, 20)),
        "p25": float(np.percentile(s, 25)),
        "p50": float(np.percentile(s, 50)),
        "p75": float(np.percentile(s, 75)),
        "p90": float(np.percentile(s, 90)),
        "low_return_shares": {
            "lt_0.05": low_share(0.05),
            "lt_0.10": low_share(0.10),
            "lt_0.20": low_share(0.20),
            "lt_0.30": low_share(0.30),
            "lt_0.50": low_share(0.50),
        },
        "LONG": grp("LONG"),
        "SHORT": grp("SHORT"),
        "FROM_FLAT": grp("FROM_FLAT"),
        "REVERSAL": grp("REVERSAL"),
        "holding_1_2_3": {},
    }
    for hb in (1, 2, 3):
        sub = df[df["holding_bars"] == hb]
        ss = sub["net_atr"].to_numpy(dtype=float)
        out["holding_1_2_3"][f"h{hb}"] = {
            "n": int(len(sub)),
            "mean_net_atr": float(np.mean(ss)) if len(ss) else float("nan"),
            "median_net_atr": float(np.median(ss)) if len(ss) else float("nan"),
            "total_gross_points": float(np.sum(sub["gross_points"].to_numpy(dtype=float)))
            if len(sub)
            else 0.0,
        }
    return out


def load_audit_inputs(
    symbol: str,
    artifact_root: str,
    *,
    regenerate: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (trades, actions, features) for the audit.

    Prefers the persisted Oracle artifact (canonical output of the 15m R2 port);
    falls back to a fresh regeneration if unavailable or ``regenerate=True``.
    ``features`` is the canonical 15m environment (source of ATR).
    """
    root = Path(artifact_root)
    loaded = None
    if not regenerate:
        loaded = load_oracle_artifact(str(root), symbol, expected_math_version=MATH_VERSION)
    if loaded is None or not loaded["ok"]:
        r = run_dp_m15_one_entry_proximity(symbol)
        frames = build_artifact_frames_v1(r)
        trades, actions = frames["oracle_trades"], frames["oracle_actions"]
    else:
        trades, actions = loaded["trades"], loaded["actions"]
    env = run_environment_m15(symbol)
    features = env["features"]
    return trades, actions, features


def run_audit(
    symbol: str,
    *,
    artifact_root: str = "artifacts/trade_oracle_dp_m15_one_entry_proximity_v1",
    out_root: str = "artifacts/dp_m15_trade_return_audit_v1",
    regenerate: bool = False,
) -> Dict[str, Any]:
    trades, actions, features = load_audit_inputs(
        symbol, artifact_root, regenerate=regenerate
    )
    frame = build_trade_return_frame(trades, actions, features)

    outdir = Path(out_root) / symbol
    outdir.mkdir(parents=True, exist_ok=True)

    # Local-only parquet (not committed).
    frame.to_parquet(outdir / "trade_returns.parquet", index=False)

    quantiles = _group_quantiles(frame)
    quantiles.to_csv(outdir / "quantiles.csv", index=False)

    hist = _histogram(frame)
    hist.to_csv(outdir / "histogram.csv", index=False)

    sweep = _threshold_sweep(frame)
    sweep.to_csv(outdir / "threshold_sweep.csv", index=False)

    holding = _holding_bins(frame)
    holding.to_csv(outdir / "holding_bins.csv", index=False)

    direction = _direction_stats(frame)
    direction.to_csv(outdir / "direction_stats.csv", index=False)

    contrib = _contribution_curve(frame)
    contrib.to_csv(outdir / "contribution_curve.csv", index=False)

    summary = _summary(frame)
    summary["symbol"] = symbol
    summary["math_version"] = MATH_VERSION
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return {
        "symbol": symbol,
        "n_trades": int(len(frame)),
        "outdir": str(outdir),
        "summary": summary,
    }


if __name__ == "__main__":
    import sys

    syms = sys.argv[1:] or ["AG"]
    for sym in syms:
        res = run_audit(sym)
        s = res["summary"]
        print(f"[{sym}] trades={s['trade_count']} mean_net_atr={s['mean_net_atr']:.4f} "
              f"median={s['median_net_atr']:.4f} p10={s['p10']:.4f} p90={s['p90']:.4f}")
        print(f"     low-share <0.20ATR: "
              f"{100*s['low_return_shares']['lt_0.20']['pct_trades']:.1f}% trades, "
              f"{100*s['low_return_shares']['lt_0.20']['pnl_contribution_pct']:.1f}% PnL")
        print(f"     -> {res['outdir']}")
