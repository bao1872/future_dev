"""
build_forming_environment_v1
============================

Production builder for the R3A *Forming* Multi-Timeframe environment.

At every 5m decision close C_t, using ONLY information known at that time,
it reconstructs the DTP / SR / Liquidity environment state E_t on
5m / 15m / 1H / 4H.

Performance design (R3A PERF1)
-------------------------------
* The indicator math is a strict streaming re-implementation
  (``forming_indicator_state_v1``). There is NO per-decision
  recomputation of historical indicator state.
* Each completed HTF bar is committed to the per-timeframe state exactly
  ONCE. A forming (still-in-progress) HTF bar is evaluated with
  ``IndicatorState.preview`` which copies the bounded committed state,
  steps the forming bar once, and discards the copy — committed state is
  never mutated by a preview.
* Complexity is O(N x TF x bounded_state), independent of total history
  length. No ``_tail`` truncation; genuine RMA / trend / liquidity
  recurrence state is carried forward by the committed state.

The canonical owner module (``experiment_structural_reversion_pgm_v1``) is
NOT modified. Only ``resample_causal`` / ``raw_frame_from_owner`` (pure
aggregation) and ``compute_tf_features`` (ORACLE, slow reference only) are
imported from it.

``slow_forming_snapshot_reference`` is the independent oracle used by the
differential tests; it is NOT on the production path.
"""

from __future__ import annotations

import json
import time
import tracemalloc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags
from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    PINE_DEFAULT,
    IndicatorParams,
    compute_tf_features,
    raw_frame_from_owner,
    resample_causal,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import (
    CONTINUOUS_COLS,
    DISCRETE_COLS,
    DISCRETE_DTYPES,
    FEATURE_COLS,
    IndicatorState,
)

TF_ORDER = ["m5", "m15", "h1", "h4"]
FEATURE_SET = set(FEATURE_COLS)


# --------------------------------------------------------------------------- #
# Vectorized forming-OHLC precomputation                                       #
# --------------------------------------------------------------------------- #
def precompute_forming_ohlc(base: pd.DataFrame, minutes: int) -> Dict[str, np.ndarray]:
    """One-pass vectorized forming-bar OHLC for every base (5m) bar.

    For bucket(t) = floor(time(t), minutes) within the same
    (trading_day, segment), the forming bar at base bar i aggregates
    base[first(i) .. i]:
        O = first open
        H = running max high
        L = running min low
        C = close(i)
        n_base = i - first(i) + 1
    This matches ``resample_causal`` for a CLOSED bucket, so at bucket
    close the forming bar equals the committed HTF bar (completed-boundary
    parity by construction).
    """
    time = pd.DatetimeIndex(base["time"])
    if minutes == 5:
        bucket = time.to_numpy()
    else:
        bucket = time.floor(f"{minutes}min").to_numpy()

    seg = base["segment"].to_numpy(np.int64)
    day = pd.to_datetime(base["trading_day"]).to_numpy()

    n = len(base)
    group_start = np.ones(n, dtype=bool)
    if n > 1:
        group_start[1:] = (
            (bucket[1:] != bucket[:-1])
            | (seg[1:] != seg[:-1])
            | (day[1:] != day[:-1])
        )
    gid = np.cumsum(group_start) - 1
    idx = np.arange(n)
    start_idx = np.maximum.accumulate(np.where(group_start, idx, 0))

    out = {
        "bucket_start": bucket,
        "group_id": gid,
        "start_idx": start_idx,
        "open": base["open"].to_numpy(float)[start_idx],
        "high": base["high"].groupby(gid).cummax().to_numpy(float),
        "low": base["low"].groupby(gid).cummin().to_numpy(float),
        "close": base["close"].to_numpy(float),
        "n_base": (idx - start_idx + 1).astype(np.int64),
    }
    return out


# --------------------------------------------------------------------------- #
# Run statistics                                                               #
# --------------------------------------------------------------------------- #
@dataclass
class RunStats:
    raw_load_count: int = 0
    state_init_count: int = 0
    forming_snapshot_count: int = 0
    completed_commit_count: int = 0
    preview_step_count: int = 0
    commit_step_count: int = 0
    # --- PERF1 structural gates (must be 0 on the production path) ---
    production_compute_tf_features_count: int = 0
    production_pd_concat_count: int = 0
    production_full_history_recompute_count: int = 0
    runtime_sec: float = 0.0
    # NOTE: this is a `tracemalloc` traced-allocation peak, NOT process RSS.
    peak_tracemalloc_mb: float = 0.0
    profile_memory_enabled: bool = False


# --------------------------------------------------------------------------- #
# Production builder                                                           #
# --------------------------------------------------------------------------- #
class FormingEnvironmentBuilder:
    def __init__(
        self,
        symbol: str,
        max_bars: Optional[int] = None,
        tf_minutes: Optional[Dict[str, int]] = None,
    ):
        self.symbol = symbol
        self.params: IndicatorParams = PINE_DEFAULT
        self.tf_minutes = dict(tf_minutes) if tf_minutes else {
            "m5": 5, "m15": 15, "h1": 60, "h4": 240,
        }
        self.max_bars = max_bars
        self.stats = RunStats()

        self.base: Optional[pd.DataFrame] = None
        self.n: int = 0
        self._o = self._h = self._l = self._c = None
        self._time: Optional[pd.DatetimeIndex] = None

        self._form: Dict[str, Dict[str, np.ndarray]] = {}
        self._completed: Dict[str, pd.DataFrame] = {}
        self._seg_completed: Dict[str, Dict[int, List[Tuple[int, Dict[str, float]]]]] = {}
        self._last_idx: Dict[str, Dict[Any, int]] = {}

    # ------------------------------------------------------------------ load
    def set_raw_frame(self, df: pd.DataFrame) -> "FormingEnvironmentBuilder":
        """Test hook: install a caller-built 5m base frame directly.

        `df` must carry columns:
            time, trading_day, segment, open, high, low, close, disc
        The provided discontinuity flags are trusted (NOT zeroed).
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
        raw_full = (
            load_raw_5m(self.symbol)
            .sort_values("bar_start_time")
            .reset_index(drop=True)
        )
        disc_full = np.asarray(discontinuity_flags(self.symbol), dtype=bool)

        # P0 correctness: a silent fallback to all-False would silently erase
        # real discontinuity. The lengths MUST agree.
        if len(raw_full) != len(disc_full):
            raise SystemExit("STOP_RAW_DISC_LENGTH_MISMATCH")

        if self.max_bars is not None:
            n = min(self.max_bars, len(raw_full))
            raw = raw_full.iloc[:n].reset_index(drop=True)
            disc = disc_full[:n]
        else:
            raw = raw_full
            disc = disc_full

        bars = dict(
            n=len(raw),
            t=pd.to_datetime(raw["bar_start_time"]).to_numpy(),
            day=pd.to_datetime(raw["trading_day"]).to_numpy(),
            disc=disc,
            o=raw["open"].to_numpy(float),
            h=raw["high"].to_numpy(float),
            l=raw["low"].to_numpy(float),
            c=raw["close"].to_numpy(float),
        )
        self.base = raw_frame_from_owner(bars)
        self.n = len(raw)
        self._o = bars["o"]
        self._h = bars["h"]
        self._l = bars["l"]
        self._c = bars["c"]
        self._time = pd.DatetimeIndex(self.base["time"].to_numpy())
        self.stats.raw_load_count += 1
        return self

    # --------------------------------------------------------------- prepare
    def prepare(self) -> "FormingEnvironmentBuilder":
        if self.base is None:
            raise RuntimeError("call load_raw() first")
        base = self.base
        n = self.n
        td_arr = pd.to_datetime(base["trading_day"]).to_numpy()
        seg_arr = base["segment"].to_numpy(np.int64)

        self._form = {}
        self._completed = {}
        self._seg_completed = {}
        self._last_idx = {}

        for tf, minutes in self.tf_minutes.items():
            form = precompute_forming_ohlc(base, minutes)
            self._form[tf] = form
            bucket_arr = form["bucket_start"]

            li: Dict[Any, int] = {}
            for j in range(n):
                key = (td_arr[j], seg_arr[j], bucket_arr[j])
                li[key] = j  # later overwrites -> last base index of bucket
            self._last_idx[tf] = li

            completed = resample_causal(base, minutes)
            self._completed[tf] = completed

            seg_completed: Dict[int, List[Tuple[int, Dict[str, float]]]] = {}
            for _, row in completed.iterrows():
                key = (row["trading_day"], row["segment"], row["time"])
                lj = li.get(key)
                if lj is None:
                    continue
                seg_completed.setdefault(int(row["segment"]), []).append(
                    (
                        lj,
                        dict(
                            open=float(row["open"]),
                            high=float(row["high"]),
                            low=float(row["low"]),
                            close=float(row["close"]),
                        ),
                    )
                )
            for s in seg_completed:
                seg_completed[s].sort(key=lambda x: x[0])
            self._seg_completed[tf] = seg_completed
            self.stats.state_init_count += int(base["segment"].nunique())
        return self

    # ------------------------------------------------------------------- run
    def run(self, profile_memory: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        if self.base is None:
            raise RuntimeError("call load_raw() first")
        n = self.n
        if n == 0:
            return pd.DataFrame(), {}

        base = self.base
        seg_arr = base["segment"].to_numpy(np.int64)
        tfs = [t for t in TF_ORDER if t in self.tf_minutes]

        out_cols = ["data_object", "decision_bar_index", "decision_time"]
        for tf in tfs:
            for c in FEATURE_COLS:
                out_cols.append(f"{tf}_{c}")
            out_cols.append(f"{tf}_bucket_start")
            out_cols.append(f"{tf}_n_base_known")

        arr: Dict[str, np.ndarray] = {}
        for tf in tfs:
            for c in CONTINUOUS_COLS:
                arr[f"{tf}_{c}"] = np.full(n, np.nan, dtype=float)
            for c in DISCRETE_COLS:
                arr[f"{tf}_{c}"] = np.zeros(n, dtype=float)
            arr[f"{tf}_bucket_start"] = np.empty(n, dtype="datetime64[ns]")
            arr[f"{tf}_n_base_known"] = np.zeros(n, dtype=np.int64)

        self.stats.profile_memory_enabled = bool(profile_memory)
        self.stats.peak_tracemalloc_mb = 0.0
        if profile_memory:
            tracemalloc.start()
        t0 = time.perf_counter()

        for tf in tfs:
            minutes = self.tf_minutes[tf]
            form = self._form[tf]
            seg_completed = self._seg_completed[tf]
            committed = IndicatorState(self.params, include_sr=True)
            cur_seg: Optional[int] = None
            ci = 0
            seg_list: List[Tuple[int, Dict[str, float]]] = []

            for i in range(n):
                seg = int(seg_arr[i])
                if seg != cur_seg:
                    committed.reset()
                    cur_seg = seg
                    ci = 0
                    seg_list = seg_completed.get(seg, [])

                # advance commits: completed bars whose coverage ended before i
                while ci < len(seg_list) and seg_list[ci][0] < i:
                    bar = seg_list[ci][1]
                    committed.step(
                        ci, bar["open"], bar["high"], bar["low"], bar["close"]
                    )
                    self.stats.commit_step_count += 1
                    ci += 1

                # preview the forming bar (does NOT mutate committed state)
                fo = form["open"][i]
                fh = form["high"][i]
                fl = form["low"][i]
                fc = form["close"][i]
                feats = committed.preview(ci, fo, fh, fl, fc)
                self.stats.preview_step_count += 1
                for c, v in feats.items():
                    if c in FEATURE_SET:
                        arr[f"{tf}_{c}"][i] = v
                arr[f"{tf}_bucket_start"][i] = form["bucket_start"][i]
                arr[f"{tf}_n_base_known"][i] = int(form["n_base"][i])

        self.stats.runtime_sec = time.perf_counter() - t0
        if profile_memory:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.stats.peak_tracemalloc_mb = peak / 1e6

        # R3A frozen semantics: the decision instant is the 5m bar CLOSE
        # C_t = T_t + 5min, where T_t is the 5m bar start. `self._time` is
        # the bar START time, so decision_time must be start + 5min.
        start_time = self._time.to_numpy()
        data = {
            "data_object": [self.symbol] * n,
            "decision_bar_index": np.arange(n),
            "decision_bar_start_time": start_time,
            "decision_time": start_time + np.timedelta64(5, "m"),
            "segment": base["segment"].to_numpy(np.int64),
            "trading_day": base["trading_day"].to_numpy(),
        }
        for col, a in arr.items():
            data[col] = a
        df = pd.DataFrame(data)
        for tf in tfs:
            for c in DISCRETE_COLS:
                col = f"{tf}_{c}"
                df[col] = df[col].astype(DISCRETE_DTYPES[c])

        audit = self._missing_audit(df, tfs)
        audit["stats"] = {
            "raw_load_count": self.stats.raw_load_count,
            "state_init_count": self.stats.state_init_count,
            "commit_step_count": self.stats.commit_step_count,
            "preview_step_count": self.stats.preview_step_count,
            "production_compute_tf_features_count": (
                self.stats.production_compute_tf_features_count
            ),
            "production_pd_concat_count": self.stats.production_pd_concat_count,
            "production_full_history_recompute_count": (
                self.stats.production_full_history_recompute_count
            ),
            "runtime_sec": self.stats.runtime_sec,
            "peak_tracemalloc_mb": self.stats.peak_tracemalloc_mb,
            "profile_memory_enabled": self.stats.profile_memory_enabled,
        }
        return df, audit

    # -------------------------------------------------------------- audit
    def _missing_audit(self, df: pd.DataFrame, tfs: List[str]) -> Dict[str, Any]:
        coverage_by_tf: Dict[str, Any] = {}
        missing_count_by_feature: Dict[str, int] = {}
        warmup_missing = 0
        for tf in tfs:
            tf_cov: Dict[str, float] = {}
            for c in FEATURE_COLS:
                col = f"{tf}_{c}"
                nonnan = int(df[col].notna().sum())
                cov = nonnan / len(df) if len(df) else 0.0
                tf_cov[c] = cov
                miss = len(df) - nonnan
                missing_count_by_feature[col] = miss
                warmup_missing += miss
            coverage_by_tf[tf] = tf_cov
        return {
            "coverage_by_tf": coverage_by_tf,
            "missing_count_by_feature": missing_count_by_feature,
            "warmup_missing_count": warmup_missing,
        }

    # ----------------------------------------------------------- oracle
    def slow_forming_snapshot_reference(
        self, i: int, tf: str
    ) -> Optional[Dict[str, float]]:
        """INDEPENDENT oracle. Rebuilds the canonical feature row for the
        forming bar at decision i from raw<=t only, via the canonical batch
        ``compute_tf_features``. Used ONLY by differential tests; never on
        the production path.
        """
        minutes = self.tf_minutes[tf]
        base = self.base
        form = self._form[tf]
        completed = self._completed[tf]
        seg = int(base["segment"].iloc[i])
        cur_bucket = form["bucket_start"][i]

        comp_done = completed[
            (completed["segment"] == seg) & (completed["time"] < cur_bucket)
        ]
        row = dict(
            time=pd.Timestamp(cur_bucket),
            trading_day=base["trading_day"].iloc[i],
            segment=seg,
            open=float(form["open"][i]),
            high=float(form["high"][i]),
            low=float(form["low"][i]),
            close=float(form["close"][i]),
            disc=False,
            available_time=pd.Timestamp(self._time[i]) + pd.Timedelta(minutes=5),
            n_base=int(form["n_base"][i]),
        )
        forming = pd.DataFrame([row])
        seq = pd.concat([comp_done, forming], ignore_index=True)
        out = compute_tf_features(seq, self.params, include_sr=True)
        if out.empty:
            return None
        last = out.iloc[-1]
        result: Dict[str, float] = {}
        for c in FEATURE_COLS:
            v = last[c]
            if c in DISCRETE_COLS:
                result[c] = float(int(v)) if np.isfinite(v) else 0.0
            else:
                result[c] = float(v) if np.isfinite(v) else np.nan
        return result
