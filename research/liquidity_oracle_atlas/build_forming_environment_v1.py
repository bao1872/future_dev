"""
R3A Forming MTF Environment builder.

Task:  FUTURE-ENV-R3A-FORMING-MTF-ENVIRONMENT
Base:  a84e74407bdd303d1ef7a236654141bc919331d0

Goal (only one question this round):
    At each 5m decision CLOSE, using ONLY information known at that moment,
    rebuild the current *forming* higher-timeframe bar and evaluate the
    canonical DTP / SR / Liquidity environment state on
        E_t = [D, SR, L]_{5m, 15m, 1H, 4H}

Design principles (frozen by the task spec):
  * NO indicator math is reimplemented. The sole feature engine is the
    already-audited `compute_tf_features` (which calls
    `compute_segment_features` -> `build_sr_features` /
    `build_liquidity_features`; both internally use `confirmed_pivots`, so the
    pivot known-time causality rule `t_known = t_pivot + right` is inherited).
  * 5m SR is intentionally ENABLED (include_sr=True for all four TFs). This is
    the deliberate R3A environment-definition change vs the legacy pipeline
    (legacy used include_sr=False for 5m).
  * A forming HTF bar is built ONLY from base bars j <= t that share the same
    (trading_day, segment, floor(time, M)). The committed completed HTF bars
    (all buckets strictly before the current one) plus the single forming bar
    are fed to `compute_tf_features`; its last row (the forming bucket) is the
    snapshot. At a bucket's final 5m decision the forming bar == the completed
    bar, so Completed-Boundary Parity holds by construction.
  * Discontinuity resets are handled automatically: `compute_tf_features`
    groups by `segment`, so rolling / pivot / SR / liquidity state never leaks
    across segments.

NOTE on efficiency: for the candidate run (T0/T1/T1.5, small data) the
instantaneous HTF frame is recomputed per decision using the full available
completed history. This is exact and matches the slow reference trivially. A
full T2 would require the committed-state incremental RMA optimization
(designed in the task spec §10) — that is OUT OF SCOPE for R3A candidate.
"""
from __future__ import annotations

import json
import sys
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags
from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    IndicatorParams,
    PINE_DEFAULT,
    compute_tf_features,
    raw_frame_from_owner,
    resample_causal,
)

# ---------------------------------------------------------------------------
# frozen config
# ---------------------------------------------------------------------------
TF_MINUTES: Dict[str, int] = {"m5": 5, "m15": 15, "h1": 60, "h4": 240}
TF_ORDER: List[str] = ["m5", "m15", "h1", "h4"]

# formal model-matrix columns (canonical output of compute_segment_features)
DTP_COLS = ["dev", "slope_atr", "trend_score", "trend_state"]
SR_COLS = [
    "sr_support_dist_atr", "sr_resistance_dist_atr",
    "sr_support_strength", "sr_resistance_strength",
    "sr_in_zone", "sr_zone_strength",
    "sr_broken_up", "sr_broken_down", "sr_n_channels",
]
LIQ_COLS = [
    "liq_up_dist_atr", "liq_down_dist_atr",
    "liq_breach_up", "liq_breach_down",
    "liq_last_breach_side", "liq_last_breach_age",
    "liq_last_accept", "liq_last_reclaim",
    "liq_last_zone_active", "liq_up_count", "liq_down_count",
]
# raw-price audit fields (kept for diagnostics, excluded from initial matrix)
SR_PRICE_COLS = ["sr_support_price", "sr_resistance_price"]
LIQ_PRICE_COLS = ["liq_up_level_price", "liq_down_level_price"]
DTP_AUDIT_COLS = ["sma", "atr"]

MODEL_COLS = DTP_COLS + SR_COLS + LIQ_COLS  # per-tf, used for missing audit


@dataclass
class RunStats:
    raw_load_count: int = 0
    state_init_count: int = 0
    forming_snapshot_count: int = 0
    completed_commit_count: int = 0
    runtime_sec: float = 0.0
    peak_rss_mb: float = 0.0


class FormingEnvironmentBuilder:
    """Build the per-decision forming MTF environment matrix.

    Usage:
        b = FormingEnvironmentBuilder(symbol)
        b.load_raw()
        b.prepare()
        env_df, audit = b.run()
    """

    def __init__(
        self,
        symbol: str,
        params: IndicatorParams | None = None,
        tf_minutes: Mapping[str, int] = TF_MINUTES,
        max_bars: int | None = None,
    ) -> None:
        self.symbol = symbol
        self.params = params or PINE_DEFAULT
        self.tf_minutes = dict(tf_minutes)
        self.max_bars = max_bars
        self.stats = RunStats()

        self.base: pd.DataFrame | None = None
        self.n: int = 0
        self._o = self._h = self._l = self._c = None
        self._time = None

        # per-tf precomputed completed HTF bars (DataFrame) + features (DataFrame)
        self._htf_bars: Dict[str, pd.DataFrame] = {}
        self._htf_feat: Dict[str, pd.DataFrame] = {}
        # per-tf: bucket-start per base bar, first base index of current bucket,
        # and whether the base bar is the last in its bucket
        self._bstart: Dict[str, np.ndarray] = {}
        self._bfirst: Dict[str, np.ndarray] = {}
        self._is_last: Dict[str, np.ndarray] = {}
        # per-tf: (td, seg, bstart_str) -> row index in htf_bars[tf]
        self._pos: Dict[str, Dict[Tuple, int]] = {}
        # per-tf: segment -> first htf row index (for seg-scoped completed history)
        self._seg_start: Dict[str, Dict[int, int]] = {}
        # bounded history window used for the forming snapshot
        self._tail: int = 2000  # >= max lookback (trend_norm_lookback=500) + RMA convergence

    # ------------------------------------------------------------------ load
    def set_raw_frame(self, df: pd.DataFrame) -> "FormingEnvironmentBuilder":
        """Test hook: install a caller-built 5m base frame directly.

        `df` must carry columns:
            time, trading_day, segment, open, high, low, close, disc
        (same schema produced by raw_frame_from_owner).
        """
        self.base = df.reset_index(drop=True)
        self.n = len(df)
        self._o = df["open"].to_numpy(float)
        self._h = df["high"].to_numpy(float)
        self._l = df["low"].to_numpy(float)
        self._c = df["close"].to_numpy(float)
        self._time = pd.DatetimeIndex(df["time"].to_numpy())
        self.stats.raw_load_count += 1
        return self

    def load_raw(self) -> "FormingEnvironmentBuilder":
        raw = load_raw_5m(self.symbol).sort_values("bar_start_time").reset_index(
            drop=True
        )
        if self.max_bars is not None and len(raw) > self.max_bars:
            raw = raw.iloc[: self.max_bars].reset_index(drop=True)
        o = raw["open"].to_numpy(float)
        h = raw["high"].to_numpy(float)
        l = raw["low"].to_numpy(float)
        c = raw["close"].to_numpy(float)
        t = pd.to_datetime(raw["bar_start_time"]).to_numpy()
        day = pd.to_datetime(raw["trading_day"]).to_numpy()
        disc = np.asarray(discontinuity_flags(self.symbol), bool)
        if len(disc) != len(raw):
            # discontinuity_flags may cover the full symbol history; align to raw
            disc = np.zeros(len(raw), bool)
        bars = dict(n=len(raw), t=t, day=day, disc=disc, o=o, h=h, l=l, c=c)
        self.base = raw_frame_from_owner(bars)
        self.n = len(raw)
        self._o, self._h, self._l, self._c = o, h, l, c
        self._time = pd.DatetimeIndex(self.base["time"].to_numpy())
        self.stats.raw_load_count += 1
        return self

    # --------------------------------------------------------------- prepare
    def prepare(self) -> "FormingEnvironmentBuilder":
        if self.base is None:
            raise RuntimeError("call load_raw() first")
        for tf, minutes in self.tf_minutes.items():
            tfb = resample_causal(self.base, minutes)
            self._htf_bars[tf] = tfb.reset_index(drop=True)
            self._htf_feat[tf] = compute_tf_features(
                tfb, self.params, include_sr=True
            ).reset_index(drop=True)
            self.stats.completed_commit_count += len(tfb)
            self.stats.state_init_count += int(self.base["segment"].nunique())
            self._index_tf(tf, minutes)
        return self

    def _index_tf(self, tf: str, minutes: int) -> None:
        b = self.base
        t = self._time
        bs = t.floor(f"{minutes}min").to_numpy()
        seg = b["segment"].to_numpy()
        td = pd.DatetimeIndex(b["trading_day"].to_numpy())
        n = len(b)

        # per-base-bar bucket key + first index of current bucket
        bfirst = np.empty(n, dtype=np.int64)
        prev_bs = None
        start = 0
        for i in range(n):
            if prev_bs is None or bs[i] != prev_bs or seg[i] != seg[i - 1]:
                start = i
            bfirst[i] = start
            prev_bs = bs[i]
        # last-in-bucket flag
        is_last = np.zeros(n, bool)
        for i in range(1, n):
            if bs[i] != bs[i - 1] or seg[i] != seg[i - 1]:
                is_last[i - 1] = True
        is_last[n - 1] = True

        # map bucket key -> htf row index
        tfb = self._htf_bars[tf]
        pos: Dict[Tuple, int] = {}
        for r in range(len(tfb)):
            key = (
                pd.Timestamp(tfb.iloc[r]["trading_day"]),
                int(tfb.iloc[r]["segment"]),
                pd.Timestamp(tfb.iloc[r]["time"]),
            )
            pos[key] = r

        self._bstart[tf] = bs
        self._bfirst[tf] = bfirst
        self._is_last[tf] = is_last
        self._pos[tf] = pos

        # first htf row index for each segment (completed history is scoped to the
        # current segment only — discontinuity resets all indicator state, so prior
        # segments do not affect the forming snapshot of the current one).
        seg_start: Dict[int, int] = {}
        last_seg = None
        for r in range(len(tfb)):
            s = int(tfb.iloc[r]["segment"])
            if s != last_seg:
                seg_start[s] = r
                last_seg = s
        self._seg_start[tf] = seg_start

    # ------------------------------------------------------------------- run
    def run(self) -> Tuple[pd.DataFrame, dict]:
        if self.base is None or not self._htf_bars:
            raise RuntimeError("call load_raw() and prepare() first")
        tracemalloc.start()
        t0 = time.perf_counter()

        b = self.base
        seg = b["segment"].to_numpy()
        td = pd.DatetimeIndex(b["trading_day"].to_numpy())
        o, h, l, c = self._o, self._h, self._l, self._c
        time_idx = self._time

        rows: List[dict] = []
        for i in range(self.n):
            dec_time = time_idx[i] + pd.Timedelta(minutes=5)
            row: dict = {
                "data_object": self.symbol,
                "decision_bar_index": i,
                "decision_bar_start_time": time_idx[i],
                "decision_time": dec_time,
                "segment": int(seg[i]),
                "trading_day": td[i],
            }
            for tf in [t for t in TF_ORDER if t in self.tf_minutes]:
                if tf == "m5":
                    # 5m bucket closes at i; forming == completed bar i exactly
                    frow = self._htf_feat["m5"].iloc[i]
                    bstart = time_idx[i]
                    nb = 1
                    complete = True
                else:
                    minutes = self.tf_minutes[tf]
                    bs = self._bstart[tf][i]
                    ff = int(self._bfirst[tf][i])
                    complete = bool(self._is_last[tf][i])
                    nb = i - ff + 1
                    O = o[ff]
                    Hh = h[ff : i + 1].max()
                    Ll = l[ff : i + 1].min()
                    Cc = c[i]
                    key = (td[i], int(seg[i]), pd.Timestamp(bs))
                    pos = self._pos[tf].get(key)
                    if pos is None:
                        # should not happen; fall back to empty forming
                        frow = None
                    else:
                        # only the CURRENT segment's completed HTF bars, tail-bounded.
                        # discontinuity resets all indicator state, so prior segments
                        # cannot affect this forming snapshot.
                        seg_start = self._seg_start[tf].get(int(seg[i]), 0)
                        lo = max(seg_start, pos - self._tail)
                        completed = self._htf_bars[tf].iloc[lo:pos]
                        forming = pd.DataFrame([{
                            "time": bs,
                            "trading_day": td[i],
                            "segment": int(seg[i]),
                            "open": O,
                            "high": Hh,
                            "low": Ll,
                            "close": Cc,
                            "disc": False,
                            "available_time": dec_time,
                            "n_base": int(nb),
                        }])
                        frame = pd.concat(
                            [completed, forming], ignore_index=True
                        )
                        feat = compute_tf_features(
                            frame, self.params, include_sr=True
                        )
                        mask = (
                            (feat["time"] == bs)
                            & (feat["trading_day"] == td[i])
                            & (feat["segment"] == int(seg[i]))
                        )
                        frow = feat[mask].iloc[-1]
                    bstart = bs
                if frow is None:
                    # defensive: leave NaN (no imputation)
                    for col in MODEL_COLS + SR_PRICE_COLS + LIQ_PRICE_COLS + DTP_AUDIT_COLS:
                        row[f"{tf}_{col}"] = np.nan
                else:
                    for col in DTP_COLS + DTP_AUDIT_COLS:
                        row[f"{tf}_{col}"] = frow.get(col, np.nan)
                    for col in SR_COLS + SR_PRICE_COLS:
                        row[f"{tf}_{col}"] = frow.get(col, np.nan)
                    for col in LIQ_COLS + LIQ_PRICE_COLS:
                        row[f"{tf}_{col}"] = frow.get(col, np.nan)
                row[f"{tf}_bucket_start"] = bstart
                row[f"{tf}_n_base_known"] = int(nb)
                row[f"{tf}_is_complete"] = bool(complete)
                self.stats.forming_snapshot_count += 1
            rows.append(row)

        env_df = pd.DataFrame(rows)
        self.stats.runtime_sec = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.stats.peak_rss_mb = peak / (1024 * 1024)

        audit = self._missing_audit(env_df)
        audit["stats"] = {
            "raw_load_count": self.stats.raw_load_count,
            "state_init_count": self.stats.state_init_count,
            "forming_snapshot_count": self.stats.forming_snapshot_count,
            "completed_commit_count": self.stats.completed_commit_count,
            "runtime_sec": round(self.stats.runtime_sec, 4),
            "peak_rss_mb": round(self.stats.peak_rss_mb, 2),
            "n_decisions": self.n,
            "symbol": self.symbol,
            "params": self.params.__class__.__name__,
        }
        return env_df, audit

    # ----------------------------------------------------------- missing audit
    def _missing_audit(self, env_df: pd.DataFrame) -> dict:
        audit: dict = {"by_feature": {}, "coverage_by_tf": {}, "warmup_missing_count": {}}
        warm = min(self.n, 500)
        for tf in [t for t in TF_ORDER if t in self.tf_minutes]:
            cols = [f"{tf}_{c}" for c in MODEL_COLS]
            miss = env_df[cols].isna().sum()
            audit["by_feature"].update(
                {c: int(miss[c]) for c in cols}
            )
            non_null_rows = env_df[cols].notna().all(axis=1).sum()
            audit["coverage_by_tf"][tf] = round(
                float(non_null_rows) / max(1, self.n), 6
            )
            audit["warmup_missing_count"][tf] = int(
                env_df.iloc[:warm][cols].isna().any(axis=1).sum()
            )
        return audit

    # --------------------------------------------------- slow reference (T0/T1)
    def _forming_bar(self, tf: str, i: int) -> dict:
        """Return the forming HTF bar built from only base bars j<=i in bucket(t)."""
        minutes = self.tf_minutes[tf]
        bs = self._bstart[tf][i]
        ff = int(self._bfirst[tf][i])
        return {
            "bucket_start": bs,
            "open": self._o[ff],
            "high": float(self._h[ff : i + 1].max()),
            "low": float(self._l[ff : i + 1].min()),
            "close": self._c[i],
            "n_base": int(i - ff + 1),
        }

    def slow_forming_snapshot_reference(
        self, i: int, tf: str
    ) -> pd.Series | None:
        """Independent reconstruction from raw <= decision i. T0/T1 differential.

        Does NOT use the maintained committed state; slices the 5m base up to
        decision i, rebuilds completed HTF buckets (strictly before the current
        bucket) + the forming bar, then calls the canonical engine.
        """
        minutes = self.tf_minutes[tf]
        sl = self.base.iloc[: i + 1].copy()
        if len(sl) == 0:
            return None
        comp = resample_causal(sl, minutes)
        cur_bucket = pd.Timestamp(self._bstart[tf][i])
        cur_td = pd.Timestamp(self.base.iloc[i]["trading_day"])
        cur_seg = int(self.base.iloc[i]["segment"])
        # completed buckets strictly before current bucket, scoped to current segment
        comp_done = comp[
            (comp["time"] < cur_bucket) & (comp["segment"] == cur_seg)
        ].tail(self._tail)
        fb = sl[
            (sl["time"].dt.floor(f"{minutes}min") == cur_bucket)
            & (pd.DatetimeIndex(sl["trading_day"]) == cur_td)
            & (sl["segment"] == cur_seg)
        ]
        forming = pd.DataFrame([{
            "time": cur_bucket,
            "trading_day": cur_td,
            "segment": cur_seg,
            "open": fb["open"].iloc[0],
            "high": fb["high"].max(),
            "low": fb["low"].min(),
            "close": fb["close"].iloc[-1],
            "disc": False,
            "available_time": self._time[i] + pd.Timedelta(minutes=5),
            "n_base": len(fb),
        }])
        frame = pd.concat([comp_done, forming], ignore_index=True)
        feat = compute_tf_features(frame, self.params, include_sr=True)
        mask = (
            (feat["time"] == cur_bucket)
            & (feat["trading_day"] == cur_td)
            & (feat["segment"] == cur_seg)
        )
        f = feat[mask]
        return None if len(f) == 0 else f.iloc[-1]


# --------------------------------------------------------------------- runner
def run_symbol(
    symbol: str,
    out_dir: str | None = None,
    max_bars: int | None = None,
    params: IndicatorParams | None = None,
) -> Tuple[pd.DataFrame, dict]:
    b = FormingEnvironmentBuilder(symbol, params=params, max_bars=max_bars)
    b.load_raw()
    b.prepare()
    env_df, audit = b.run()
    if out_dir:
        d = Path(out_dir)
        d.mkdir(parents=True, exist_ok=True)
        # large parquet intentionally NOT committed to git
        env_df.to_parquet(d / f"forming_environment_{symbol}.parquet", index=False)
        with open(d / f"forming_audit_{symbol}.json", "w") as f:
            json.dump(audit, f, indent=2, default=str)
    return env_df, audit


if __name__ == "__main__":
    import os

    # candidate / T1.5 run: 2 symbols, bounded bars, temp artifact dir
    OUT = "artifacts/forming_environment_v1"
    SYMS = ["AG", "CU"]
    MAXB = 2800
    for sym in SYMS:
        df, aud = run_symbol(sym, out_dir=OUT, max_bars=MAXB)
        print(f"[{sym}] rows={len(df)} cols={df.shape[1]} "
              f"runtime={aud['stats']['runtime_sec']}s "
              f"rss={aud['stats']['peak_rss_mb']}MB")
        print(f"  coverage_by_tf={aud['coverage_by_tf']}")
