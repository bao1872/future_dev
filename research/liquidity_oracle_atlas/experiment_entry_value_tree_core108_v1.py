"""
experiment_entry_value_tree_core108_v1
======================================

FUTURE-ENTRY-VALUE-TREE-CORE108-V1  —  Phase 1 feature kernel (checkpoint).

Research question (frozen): within existing DP V2 global proximity episodes, can
the causal 4-timeframe market environment and the path-to-date features predict
the counterfactual incremental value of entering LONG/SHORT now versus remaining
FLAT and preserving the right to enter later in the same episode? Which features
matter?

This module is the EXPERIMENT-SPECIFIC production feature kernel only. It does
NOT train a model, does NOT run the DP oracle, does NOT modify any canonical
owner. Phase 1 produces CORE108 feature rows + labels + a small checkpoint
dataset, validated by T0/T1/TP. Model training (T1.5) and the 15-symbol full
run (T2) are OUT OF SCOPE for this file.

Canonical owners reused (NOT modified):
  * research.export_ob_trigger_execution_v21.load_raw_5m
  * research.phase1_tradability.phase1_contract_v1.discontinuity_flags
  * research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1
        PINE_DEFAULT, raw_frame_from_owner, resample_causal
  * research.liquidity_oracle_atlas.build_forming_environment_v1
        precompute_forming_ohlc
  * research.liquidity_oracle_atlas.forming_indicator_state_v1
        IndicatorState  (DTP trend_state/slope_atr/dev/atr + preview/snapshot)
  * research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1
        TF_ORDER, ROLES, REV_SIGN, ENTRY_PROX_ATR, NEAR_ATR, MASK_BIT,
        build_base_from_arrays, proximity_bits_from_prev_geometry,
        select_target, in_proximity, start_episode, classify_bar,
        recedes_without_touch
  * research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2
        load_oracle_artifact_v2, MATH_VERSION

--------------------------------------------------------------------------
IMPORTANT SEMANTIC REVISIONS (per reviewer REQUIRED CHANGES)
--------------------------------------------------------------------------
RC1  ATR source. There is NO ``base['atr']``. The 5m ATR comes from the m5
     ``IndicatorState.step()`` DTP feature ``feats["atr"]`` produced during the
     single-pass streaming. It is used ONLY for Y normalization + audit and
     never enters CORE108.

RC2  The canonical ``Episode.approach_velocity`` STOPS updating after the first
     interaction event. Our research question needs every 5m bar to describe
     the episode-to-date path. So we add ``Core108PathTracker`` whose
     ``approach_velocity``/``path_efficiency`` are EPISODE-TO-DATE statistics
     (updated until the role episode ends). We reuse the canonical ``Episode``
     lifecycle / ``classify_bar`` phase but the path math is new. This is
     documented in code + metadata.

RC3  ``OUTSIDE`` keeps the REAL signed distance; 0 is reserved for "at the
     boundary". No structure -> phase=NO_STRUCTURE, distance_atr=NaN, rest 0.
     Structure exists but not yet in an interaction episode -> phase=OUTSIDE.
     distance_atr is the REAL signed distance scaled ONLY by the same-TF
     previous-geometry ATR (one consistent unit per column); when that per-TF ATR
     is unavailable it is NaN (no cross-TF / 5m fallback). rest 0.

RC4  Two distinct ATRs:
       * Path ATR  A^{path} = ATR_{TF, tau}, frozen at role-episode start and
         taken from the previous-close geometry (canonical near-edge/proximity
         uses the TF ATR at the decision bar). Used to scale d_t.
       * Target ATR ATR_{5m,t} = current m5 DTP atr. Used ONLY for Y_L/Y_S.

RC5  All episode aggregation uses composite key
     ``GlobalEpisodeKey = (symbol, proximity_episode_id)`` (R2 ids are
     per-symbol). Never group by proximity_episode_id alone.

RC6  The production kernel recomputes ``proximity_bits_from_prev_geometry``
     every bar (structure gate only, NOT a DP re-solve). T1 proves they equal
     the R2 artifact's proximity_bits/any bit-for-bit.

RC7  The Reference kernel (T1) must NOT call ``Core108PathTracker``; it replays
     the episode independently with the raw formulas.

RC8  Real checkpoint dataset and synthetic fixtures are kept separate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags
from research.liquidity_oracle_atlas.experiment_structural_reversion_pgm_v1 import (
    PINE_DEFAULT,
    raw_frame_from_owner,
    resample_causal,
)
from research.liquidity_oracle_atlas.build_forming_environment_v1 import (
    precompute_forming_ohlc,
)
from research.liquidity_oracle_atlas.forming_indicator_state_v1 import IndicatorState
from research.liquidity_oracle_atlas.experiment_structure_interaction_entry_v1 import (
    TF_ORDER,
    TF_MINUTES,
    ROLES,
    REV_SIGN,
    NEAR_ATR,
    ENTRY_PROX_ATR,
    MASK_BIT,
    build_base_from_arrays,
    proximity_bits_from_prev_geometry,
    select_target,
    in_proximity,
    start_episode,
    classify_bar,
    recedes_without_touch,
)
from research.liquidity_oracle_atlas.build_structure_constrained_trade_oracle_dp_v2 import (
    load_oracle_artifact_v2,
    MATH_VERSION,
)

_NAN = float("nan")

DEFAULT_ARTIFACT_ROOT = Path("artifacts/intraday_dp_oracle_r2_one_entry_proximity")

# 4 TF x (3 DTP + 4 roles x 6 path fields) = 108
_PATH_FIELDS = (
    "distance_atr",
    "phase",
    "episode_age_5m",
    "approach_velocity",
    "path_efficiency",
    "max_penetration_atr",
)

# Columns explicitly forbidden from CORE108 (audit/leak/cost columns live
# elsewhere). Used by T0 negative control.
FORBIDDEN_FUTURE_COLUMNS = {
    "best_F1",
    "edge_F1",
    "amb_F1",
    "Q_F1_L",
    "Q_F1_F",
    "Q_F1_S",
    "label_available_time",
    "proximity_episode_id",
    "proximity_bits",
    "proximity_any",
    "decision_time",
    "training_eligible",
    "outcome_ret",
    "mfe",
    "mae",
    "next_bar_direction",
    "oracle_path_action",
    "oracle_position",
    "volume",
    "strength",
    "zone_width",
}


# --------------------------------------------------------------------------- #
# Counters                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class Core108Counters:
    """Phase-1 performance / integrity counters."""

    raw_load_count: int = 0
    resample_count: int = 0
    feature_precompute_count: int = 0
    path_step_count: int = 0
    feature_row_count: int = 0
    full_history_recompute_count: int = 0
    reference_call_count: int = 0
    concat_count: int = 0
    oracle_recompute_count: int = 0
    disk_read_count: int = 0


# --------------------------------------------------------------------------- #
# Experiment-specific path tracker (RC2)                                        #
# --------------------------------------------------------------------------- #
class Core108PathTracker:
    """Per-(tf, role) episode-to-date path state.

    IMPORTANT (RC2): this is NOT the canonical ``Episode.approach_velocity``,
    which freezes after the first interaction event. Here ``approach_velocity``
    and ``path_efficiency`` are updated on EVERY bar of the role episode until
    it ends. The phase label is still taken from the canonical ``Episode``
    lifecycle (driven by ``classify_bar``), so only the path SCALARS are new.

    State (frozen at episode start):
      tau     : bar index of episode start
      sE      : oriented near edge  (s * near_edge)
      A_path  : frozen path ATR = ATR_{TF, tau}
    """

    __slots__ = (
        "active",
        "tau",
        "sE",
        "A_path",
        "d_start",
        "d_prev",
        "age",
        "tv",
        "max_pen",
        "phase",
    )

    def __init__(self) -> None:
        self.active = False
        self.tau = 0
        self.sE = 0.0
        self.A_path = 0.0
        self.d_start = _NAN
        self.d_prev = _NAN
        self.age = 0
        self.tv = 0.0
        self.max_pen = 0.0
        self.phase = "OUTSIDE"

    def start(self, tau: int, sE: float, A_path: float) -> None:
        self.active = True
        self.tau = tau
        self.sE = sE
        self.A_path = A_path
        self.d_start = _NAN
        self.d_prev = _NAN
        self.age = 0
        self.tv = 0.0
        self.max_pen = 0.0
        self.phase = "APPROACH"

    def reset(self) -> None:
        self.active = False
        self.tau = 0
        self.sE = 0.0
        self.A_path = 0.0
        self.d_start = _NAN
        self.d_prev = _NAN
        self.age = 0
        self.tv = 0.0
        self.max_pen = 0.0
        self.phase = "OUTSIDE"

    def update(self, sC: float, sH: float, sL: float, phase: str) -> None:
        """Advance the episode-to-date state for the current bar.

        sC/sH/sL are the REV_SIGN-oriented coordinates of the current 5m OHLC.
        d_t = (sC - sE) / A_path.
        u^min = min(sH, sL).  pen = max(0, (sE - u^min)/A_path).
        """
        A = self.A_path
        if not (math.isfinite(A) and A > 0):
            self.d_start = _NAN
            self.d_prev = _NAN
            self.age = 1 if self.age == 0 else self.age + 1
            self.phase = phase
            return

        d = (sC - self.sE) / A
        u_min = min(sH, sL)
        pen = max(0.0, (self.sE - u_min) / A)

        if self.age == 0:
            self.d_start = d
            self.d_prev = d
            self.age = 1
            self.tv = 0.0
            self.max_pen = pen
        else:
            self.tv += abs(d - self.d_prev)
            self.d_prev = d
            self.age += 1
            if pen > self.max_pen:
                self.max_pen = pen
        self.phase = phase

    def features(self) -> Tuple[float, str, int, float, float, float]:
        """Return (distance_atr, phase, episode_age_5m, approach_velocity,
        path_efficiency, max_penetration_atr)."""
        if not self.active or self.age == 0:
            return (0.0, "OUTSIDE", 0, 0.0, 0.0, 0.0)
        d_start = self.d_start
        d_cur = self.d_prev
        age = self.age
        denom = max(1, age - 1)
        if math.isfinite(d_start) and math.isfinite(d_cur):
            vel = (d_start - d_cur) / denom
            eps = 1e-9
            eff = (d_start - d_cur) / (self.tv + eps)
        else:
            vel = 0.0
            eff = 0.0
        return (d_cur, self.phase, age, vel, eff, self.max_pen)


# --------------------------------------------------------------------------- #
# CORE108 schema                                                               #
# --------------------------------------------------------------------------- #
def core108_columns() -> List[str]:
    cols: List[str] = []
    for tf in TF_ORDER:
        cols += [f"{tf}_trend_state", f"{tf}_slope_atr", f"{tf}_dev"]
        for role in ROLES:
            for f in _PATH_FIELDS:
                cols.append(f"{tf}_{role}_{f}")
    return cols


def _path_block(role: str) -> List[str]:
    return [f"{tf}_{role}_{f}" for tf in TF_ORDER for f in _PATH_FIELDS]


# --------------------------------------------------------------------------- #
# Base frame builder (true prefix support)                                      #
# --------------------------------------------------------------------------- #
def build_base_prefix(
    symbol: str,
    counters: Core108Counters,
    max_bars: Optional[int] = None,
) -> Dict[str, Any]:
    """Load raw 5m once (canonical owner) + build per-TF forming frames.

    If ``max_bars`` is given the raw is sliced to a TRUE prefix (bar 0..N-1) so
    DTP / structure state is identical to the full run up to that bar (RC8 /
    reviewer: arbitrary history tail truncation is forbidden).
    """
    counters.raw_load_count += 1
    raw = load_raw_5m(symbol).sort_values("bar_start_time").reset_index(drop=True)
    disc = np.asarray(discontinuity_flags(symbol), dtype=bool)
    if max_bars is not None and int(max_bars) < len(raw):
        raw = raw.iloc[: int(max_bars)].reset_index(drop=True)
        disc = disc[: int(max_bars)]
    if len(raw) != len(disc):
        raise SystemExit("STOP_RAW_DISC_LENGTH_MISMATCH")

    info = build_base_from_arrays(
        pd.to_datetime(raw["bar_start_time"]).to_numpy(),
        pd.to_datetime(raw["trading_day"]).to_numpy(),
        raw["open"].to_numpy(float),
        raw["high"].to_numpy(float),
        raw["low"].to_numpy(float),
        raw["close"].to_numpy(float),
        disc,
        counters,
    )
    counters.feature_precompute_count += 1
    info["base_time"] = pd.to_datetime(raw["bar_start_time"]).to_numpy()
    return info


# --------------------------------------------------------------------------- #
# Production streaming kernel                                                   #
# --------------------------------------------------------------------------- #
def run_streaming(
    info: Dict[str, Any],
    counters: Core108Counters,
    symbol: str,
    max_bars: Optional[int] = None,
    geom_mutator: Optional[Callable[[int, Dict[str, Tuple]], Dict[str, Tuple]]] = None,
) -> pd.DataFrame:
    """Single-pass streaming: 4TF IndicatorState + 16 Core108PathTrackers.

    Returns a DataFrame with ONE ROW PER proximity_any bar (the candidate set),
    carrying the 108 CORE108 features + audit columns. O(N x 4 x 4) linear.

    ``geom_mutator`` is a TEST-ONLY seam: when provided, it receives
    ``(bar_index, geom_by_tf)`` after the per-TF preview and returns the geometry
    dict used for the path/structure interaction. It never affects production
    (defaults to None -> identity). It lets a test scramble the CURRENT bar's
    geometry to prove the path consumes ``prev_geom`` (geometry_{t-1}), not
    ``geom_by_tf`` (geometry_t).
    """
    base = info["base"]
    form = info["form"]
    seg_completed = info["seg_completed"]
    n = len(base)
    if max_bars is not None:
        n = min(n, int(max_bars))

    seg_arr = base["segment"].to_numpy(np.int64)
    time_arr = base["time"].to_numpy()
    O5 = base["open"].to_numpy(float)[:n]
    H5 = base["high"].to_numpy(float)[:n]
    L5 = base["low"].to_numpy(float)[:n]
    C5 = base["close"].to_numpy(float)[:n]
    base_time = info["base_time"][:n]

    # per-TF committed/preview indicator state
    tf_states = {tf: IndicatorState(PINE_DEFAULT, include_sr=True) for tf in TF_ORDER}
    cur_seg_per_tf = {tf: None for tf in TF_ORDER}
    ci_per_tf = {tf: 0 for tf in TF_ORDER}
    seg_list_per_tf = {tf: [] for tf in TF_ORDER}

    # per-(tf, role) state: canonical Episode (phase lifecycle) + our tracker
    active: Dict[Tuple[str, str], Any] = {
        (tf, role): None for tf in TF_ORDER for role in ROLES
    }
    trackers = {(tf, role): Core108PathTracker() for tf in TF_ORDER for role in ROLES}

    # DTP context per TF (trend_state/slope_atr/dev) + m5 atr for Y
    dtp_ctx = {
        tf: {
            "dev": np.full(n, _NAN),
            "slope_atr": np.full(n, _NAN),
            "trend_state": np.full(n, _NAN),
        }
        for tf in TF_ORDER
    }
    atr5m = np.full(n, _NAN)

    sr_first_seen: Dict[Tuple, int] = {}

    prev_geom_by_tf: Optional[Dict[str, Tuple]] = None
    prev_seg = None

    # columnar outputs
    cols = core108_columns()
    rows: List[Dict[str, Any]] = []
    audit_bar: List[int] = []
    audit_time: List[Any] = []
    audit_prox_bits: List[int] = []
    audit_atr5m: List[float] = []

    for i in range(n):
        seg = int(seg_arr[i])
        if seg != prev_seg:
            for key in active:
                if active[key] is not None:
                    active[key].termination_reason = "SEGMENT_END"
                    active[key] = None
                trackers[key].reset()
            for tf in TF_ORDER:
                tf_states[tf].reset()
                cur_seg_per_tf[tf] = seg
                ci_per_tf[tf] = 0
                seg_list_per_tf[tf] = seg_completed[tf].get(seg, [])
            prev_geom_by_tf = None
            prev_seg = seg

        # ---- per-TF commit + preview (geometry snapshot) ----
        geom_by_tf: Dict[str, Tuple] = {}
        for tf in TF_ORDER:
            minutes = TF_MINUTES[tf]
            f = form[tf]
            seg_list = seg_list_per_tf[tf]
            ci = ci_per_tf[tf]
            while ci < len(seg_list) and seg_list[ci][0] < i:
                bar = seg_list[ci][1]
                tf_states[tf].step(ci, bar["open"], bar["high"], bar["low"], bar["close"])
                ci += 1
            ci_per_tf[tf] = ci

            fo = float(f["open"][i])
            fh = float(f["high"][i])
            fl = float(f["low"][i])
            fc = float(f["close"][i])
            pv = tf_states[tf].snapshot()
            feats = pv.step(ci, fo, fh, fl, fc)

            atr_tf = feats["atr"]
            dtp_ctx[tf]["dev"][i] = feats["dev"]
            dtp_ctx[tf]["slope_atr"][i] = feats["slope_atr"]
            dtp_ctx[tf]["trend_state"][i] = feats["trend_state"]
            if tf == "m5":
                atr5m[i] = atr_tf
            channels = list(pv.sr.channels)
            liq_up = [dict(x) for x in pv.liq.levels_up]
            liq_down = [dict(x) for x in pv.liq.levels_down]
            geom_by_tf[tf] = (channels, liq_up, liq_down, atr_tf)

        # TEST-ONLY seam: optionally scramble the CURRENT bar's geometry so a test
        # can prove the path consumes prev_geom (geometry_{t-1}), not this one.
        if geom_mutator is not None:
            geom_by_tf = geom_mutator(i, geom_by_tf)

        # ---- recompute proximity from PREVIOUS close geometry (RC6) ----
        pb = int(
            proximity_bits_from_prev_geometry(
                float(L5[i]), float(H5[i]), prev_geom_by_tf or {}, alpha=ENTRY_PROX_ATR
            )
        )
        pa = bool(pb != 0)

        # ---- CORE108 path interacts with PREVIOUS-close known structure ----
        # The structure-to-path interaction (select_target / in_proximity /
        # owner-change / OUTSIDE distance / episode start ATR) must use the
        # geometry that existed BEFORE this bar started (interaction_geom_by_tf ==
        # prev_geom_by_tf). geom_by_tf (current preview) is used only for DTP,
        # forming MTF, and to become the NEXT bar's prev_geom_by_tf. This is RC4
        # (frozen A_path comes from previous geometry) and avoids "self-made
        # structure" (forming a structure this bar then claiming to touch it).
        interaction_geom_by_tf = prev_geom_by_tf

        # ---- per (tf, role) episode lifecycle + path update (RC2/RC3/RC4) ----
        O = O5[i]
        H = H5[i]
        L = L5[i]
        C = C5[i]
        path_feats: Dict[Tuple[str, str], Tuple[float, str, int, float, float, float]] = {}

        for tf in TF_ORDER:
            _g = interaction_geom_by_tf.get(tf) if interaction_geom_by_tf else None
            if _g is not None:
                channels, liq_up, liq_down, atr_tf = _g
            else:
                channels, liq_up, liq_down, atr_tf = None, _NAN, _NAN, _NAN
            for role in ROLES:
                key = (tf, role)
                ep = active[key]
                cand = (
                    select_target(
                        role, channels, liq_up, liq_down, C, atr_tf, tf, seg, i, sr_first_seen
                    )
                    if channels is not None
                    else None
                )

                if ep is None:
                    if cand is not None:
                        prox = in_proximity(cand, O, H, L, C, atr_tf)
                        # FIX-C: a new (tf, role) episode requires a defined per-TF
                        # ATR (previous-geometry). Without it the proximity radius
                        # and the frozen A_path are undefined, so no episode may
                        # start. (in_proximity's touch branch can be True even with
                        # atr_tf=NaN, hence the explicit guard.) The bar falls
                        # through to OUTSIDE/NaN below.
                        if prox and np.isfinite(atr_tf) and atr_tf > 0:
                            ep = start_episode(
                                symbol, tf, role, cand, i, seg, time_arr[i], C
                            )
                            active[key] = ep
                            trackers[key].start(i, ep.near_edge * REV_SIGN[role], ep.start_atr)
                else:
                    if cand is not None and cand["structure_id"] != ep.structure_id:
                        ep.termination_reason = "STRUCTURE_OWNER_CHANGED"
                        active[key] = None
                        trackers[key].reset()
                        ep = None
                        if cand is not None:
                            prox = in_proximity(cand, O, H, L, C, atr_tf)
                            if prox and np.isfinite(atr_tf) and atr_tf > 0:
                                ep = start_episode(
                                    symbol, tf, role, cand, i, seg, time_arr[i], C
                                )
                                active[key] = ep
                                trackers[key].start(
                                    i, ep.near_edge * REV_SIGN[role], ep.start_atr
                                )

                if ep is not None:
                    classify_bar(ep, O, H, L, C)
                    tr = trackers[key]
                    s = ep.rev_sign
                    sC = s * C
                    sH = max(s * H, s * L)
                    sL = min(s * H, s * L)
                    tr.update(sC, sH, sL, ep.phase)
                    counters.path_step_count += 1
                    cur = tr.features()

                    # termination (never on the start bar)
                    if i != ep.start_i:
                        s_atr = ep.rev_sign
                        u = s_atr * C
                        u_near = s_atr * ep.near_edge
                        u_far = s_atr * ep.far_edge
                        atr = ep.start_atr
                        term = False
                        if i == n - 1:
                            term = True
                        elif recedes_without_touch(ep, C, atr):
                            term = True
                        elif u - u_near > NEAR_ATR * atr:
                            term = True
                        elif u_far - u > NEAR_ATR * atr:
                            term = True
                        if term:
                            active[key] = None
                            trackers[key].reset()
                            # keep the pre-termination state for THIS bar
                    path_feats[key] = cur
                else:
                    if cand is not None:
                        s = REV_SIGN[role]
                        sE = s * cand["near_edge"]
                        # FIX-A/B: OUTSIDE distance is scaled ONLY by the same-TF
                        # previous-geometry ATR (atr_tf). No current-TF / 5m ATR
                        # fallback: a missing per-TF ATR means the column unit is
                        # undefined, so it stays NaN (LightGBM handles missing
                        # natively). This keeps every {tf}_*_distance_atr in one
                        # consistent unit (never a mix of 4h-ATR and 5m-ATR rows).
                        d_out = (
                            (s * C - sE) / atr_tf
                            if (np.isfinite(atr_tf) and atr_tf > 0)
                            else _NAN
                        )
                        path_feats[key] = (d_out, "OUTSIDE", 0, 0.0, 0.0, 0.0)
                    else:
                        path_feats[key] = (_NAN, "NO_STRUCTURE", 0, 0.0, 0.0, 0.0)

        # advance: current preview geometry becomes the NEXT bar's previous-known
        # structure. The path of bar t used prev_geom_by_tf (geometry_{t-1});
        # geometry_t is only now promoted, so bar t+1's path will see it.
        prev_geom_by_tf = {
            tf: (
                geom_by_tf[tf][0],
                geom_by_tf[tf][1],
                geom_by_tf[tf][2],
                geom_by_tf[tf][3],
            )
            for tf in TF_ORDER
        }

        if not pa:
            continue

        # ---- assemble CORE108 row (only on proximity_any) ----
        row: Dict[str, Any] = {}
        for tf in TF_ORDER:
            row[f"{tf}_trend_state"] = dtp_ctx[tf]["trend_state"][i]
            row[f"{tf}_slope_atr"] = dtp_ctx[tf]["slope_atr"][i]
            row[f"{tf}_dev"] = dtp_ctx[tf]["dev"][i]
            for role in ROLES:
                d, ph, age, vel, eff, mpen = path_feats[(tf, role)]
                pre = f"{tf}_{role}_"
                row[pre + "distance_atr"] = d
                row[pre + "phase"] = ph
                row[pre + "episode_age_5m"] = age
                row[pre + "approach_velocity"] = vel
                row[pre + "path_efficiency"] = eff
                row[pre + "max_penetration_atr"] = mpen

        rows.append(row)
        audit_bar.append(i)
        audit_time.append(pd.Timestamp(base_time[i]) + pd.Timedelta(minutes=5))
        audit_prox_bits.append(pb)
        audit_atr5m.append(atr5m[i])

    counters.feature_row_count = len(rows)
    out = pd.DataFrame(rows, columns=cols)
    out["symbol"] = symbol
    out["decision_bar_index"] = audit_bar
    out["decision_time"] = audit_time
    out["prox_bits_kernel"] = audit_prox_bits
    out["atr5m"] = audit_atr5m
    return out


# --------------------------------------------------------------------------- #
# Reference kernel (RC7) — independent slow replay, no Core108PathTracker        #
# --------------------------------------------------------------------------- #
def reference_replay(
    info: Dict[str, Any],
    counters: Core108Counters,
    targets: List[Dict[str, Any]],
    symbol: str = "SYNTH",
) -> Dict[Tuple[int, str, str], Tuple[float, str, int, float, float, float]]:
    """Independent per-(tf, role) path replay for a SMALL set of (t, tf, role).

    It does NOT call ``Core108PathTracker``; it re-derives the episode lifecycle
    with canonical helpers and computes the episode-to-date scalars INLINE with
    the raw formulas, so T1 validates the production path math independently.
    """
    counters.reference_call_count += 1
    base = info["base"]
    form = info["form"]
    seg_completed = info["seg_completed"]
    max_t = max(t["t"] for t in targets) + 1
    n = min(len(base), max_t)

    seg_arr = base["segment"].to_numpy(np.int64)
    time_arr = base["time"].to_numpy()
    O5 = base["open"].to_numpy(float)[:n]
    H5 = base["high"].to_numpy(float)[:n]
    L5 = base["low"].to_numpy(float)[:n]
    C5 = base["close"].to_numpy(float)[:n]

    tf_states = {tf: IndicatorState(PINE_DEFAULT, include_sr=True) for tf in TF_ORDER}
    cur_seg_per_tf = {tf: None for tf in TF_ORDER}
    ci_per_tf = {tf: 0 for tf in TF_ORDER}
    seg_list_per_tf = {tf: [] for tf in TF_ORDER}

    active: Dict[Tuple[str, str], Any] = {
        (tf, role): None for tf in TF_ORDER for role in ROLES
    }
    # inline path numbers (independent of Core108PathTracker)
    inline: Dict[Tuple[str, str], Dict[str, Any]] = {
        (tf, role): {
            "active": False,
            "d_start": _NAN,
            "d_prev": _NAN,
            "age": 0,
            "tv": 0.0,
            "max_pen": 0.0,
            "sE": 0.0,
            "A": 0.0,
        }
        for tf in TF_ORDER
        for role in ROLES
    }

    sr_first_seen: Dict[Tuple, int] = {}
    prev_seg = None
    prev_geom_by_tf: Optional[Dict[str, Tuple]] = None

    result: Dict[Tuple[int, str, str], Tuple[float, str, int, float, float, float]] = {}
    target_set = {(t["t"], t["tf"], t["role"]) for t in targets}

    for i in range(n):
        seg = int(seg_arr[i])
        if seg != prev_seg:
            for key in active:
                active[key] = None
                inline[key] = {
                    "active": False,
                    "d_start": _NAN,
                    "d_prev": _NAN,
                    "age": 0,
                    "tv": 0.0,
                    "max_pen": 0.0,
                    "sE": 0.0,
                    "A": 0.0,
                }
            for tf in TF_ORDER:
                tf_states[tf].reset()
                cur_seg_per_tf[tf] = seg
                ci_per_tf[tf] = 0
                seg_list_per_tf[tf] = seg_completed[tf].get(seg, [])
            prev_geom_by_tf = None
            prev_seg = seg

        geom_by_tf: Dict[str, Tuple] = {}
        for tf in TF_ORDER:
            minutes = TF_MINUTES[tf]
            f = form[tf]
            seg_list = seg_list_per_tf[tf]
            ci = ci_per_tf[tf]
            while ci < len(seg_list) and seg_list[ci][0] < i:
                bar = seg_list[ci][1]
                tf_states[tf].step(ci, bar["open"], bar["high"], bar["low"], bar["close"])
                ci += 1
            ci_per_tf[tf] = ci
            fo = float(f["open"][i])
            fh = float(f["high"][i])
            fl = float(f["low"][i])
            fc = float(f["close"][i])
            pv = tf_states[tf].snapshot()
            feats = pv.step(ci, fo, fh, fl, fc)
            atr_tf = feats["atr"]
            channels = list(pv.sr.channels)
            liq_up = [dict(x) for x in pv.liq.levels_up]
            liq_down = [dict(x) for x in pv.liq.levels_down]
            geom_by_tf[tf] = (channels, liq_up, liq_down, atr_tf)

        O = O5[i]
        H = H5[i]
        L = L5[i]
        C = C5[i]

        # CORE108 path must interact with PREVIOUS-close known structure
        # (mirrors production: geometry_{t-1} + Bar_t -> Path_t).
        interaction_geom_by_tf = prev_geom_by_tf

        for tf in TF_ORDER:
            _g = interaction_geom_by_tf.get(tf) if interaction_geom_by_tf else None
            if _g is not None:
                channels, liq_up, liq_down, atr_tf = _g
            else:
                channels, liq_up, liq_down, atr_tf = None, _NAN, _NAN, _NAN
            for role in ROLES:
                key = (tf, role)
                ep = active[key]
                st = inline[key]
                cand = (
                    select_target(
                        role, channels, liq_up, liq_down, C, atr_tf, tf, seg, i, sr_first_seen
                    )
                    if channels is not None
                    else None
                )
                if ep is None:
                    if cand is not None:
                        prox = in_proximity(cand, O, H, L, C, atr_tf)
                        # FIX-C: require a defined per-TF ATR to start a new episode
                        # (mirrors production; in_proximity's touch branch is True
                        # even with atr_tf=NaN).
                        if prox and math.isfinite(atr_tf) and atr_tf > 0:
                            ep = start_episode(
                                symbol, tf, role, cand, i, seg, time_arr[i], C
                            )
                            active[key] = ep
                            st["active"] = True
                            st["sE"] = ep.near_edge * REV_SIGN[role]
                            st["A"] = ep.start_atr
                            st["d_start"] = _NAN
                            st["d_prev"] = _NAN
                            st["age"] = 0
                            st["tv"] = 0.0
                            st["max_pen"] = 0.0
                else:
                    if cand is not None and cand["structure_id"] != ep.structure_id:
                        active[key] = None
                        st["active"] = False
                        ep = None
                        if cand is not None:
                            prox = in_proximity(cand, O, H, L, C, atr_tf)
                            if prox and math.isfinite(atr_tf) and atr_tf > 0:
                                ep = start_episode(
                                    symbol, tf, role, cand, i, seg, time_arr[i], C
                                )
                                active[key] = ep
                                st["active"] = True
                                st["sE"] = ep.near_edge * REV_SIGN[role]
                                st["A"] = ep.start_atr
                                st["d_start"] = _NAN
                                st["d_prev"] = _NAN
                                st["age"] = 0
                                st["tv"] = 0.0
                                st["max_pen"] = 0.0

                if ep is not None:
                    classify_bar(ep, O, H, L, C)
                    s = ep.rev_sign
                    sC = s * C
                    sH = max(s * H, s * L)
                    sL = min(s * H, s * L)
                    A = st["A"]
                    if math.isfinite(A) and A > 0:
                        d = (sC - st["sE"]) / A
                        u_min = min(sH, sL)
                        pen = max(0.0, (st["sE"] - u_min) / A)
                    else:
                        d = _NAN
                        pen = 0.0
                    if st["age"] == 0:
                        st["d_start"] = d
                        st["d_prev"] = d
                        st["age"] = 1
                        st["tv"] = 0.0
                        st["max_pen"] = pen
                    else:
                        st["tv"] += abs(d - st["d_prev"])
                        st["d_prev"] = d
                        st["age"] += 1
                        if pen > st["max_pen"]:
                            st["max_pen"] = pen
                    phase = ep.phase

                    # capture pre-termination path state (mirrors production
                    # Core108PathTracker.features(), which is read BEFORE the
                    # termination reset so the terminating bar keeps its state)
                    d_start = st["d_start"]
                    d_cur = st["d_prev"]
                    age = st["age"]
                    if st["active"] and age > 0 and math.isfinite(d_start) and math.isfinite(d_cur):
                        vel = (d_start - d_cur) / max(1, age - 1)
                        eff = (d_start - d_cur) / (st["tv"] + 1e-9)
                    else:
                        vel = 0.0
                        eff = 0.0
                    dist = d_cur if (st["active"] and age > 0) else 0.0
                    feat_out = (dist, phase, age, vel, eff, st["max_pen"])

                    if (i, tf, role) in target_set:
                        result[(i, tf, role)] = feat_out

                    # termination (after capture)
                    if i != ep.start_i:
                        u = s * C
                        u_near = s * ep.near_edge
                        u_far = s * ep.far_edge
                        atr = ep.start_atr
                        term = False
                        if i == n - 1:
                            term = True
                        elif recedes_without_touch(ep, C, atr):
                            term = True
                        elif u - u_near > NEAR_ATR * atr:
                            term = True
                        elif u_far - u > NEAR_ATR * atr:
                            term = True
                        if term:
                            active[key] = None
                            st["active"] = False
                else:
                    if (i, tf, role) in target_set:
                        if cand is not None:
                            s = REV_SIGN[role]
                            sE = s * cand["near_edge"]
                            # FIX-A/B: OUTSIDE distance scaled ONLY by the
                            # same-TF previous-geometry ATR; no cross-TF
                            # fallback (matches production).
                            d_out = (
                                (s * C - sE) / atr_tf
                                if (np.isfinite(atr_tf) and atr_tf > 0)
                                else _NAN
                            )
                            result[(i, tf, role)] = (d_out, "OUTSIDE", 0, 0.0, 0.0, 0.0)
                        else:
                            result[(i, tf, role)] = (_NAN, "NO_STRUCTURE", 0, 0.0, 0.0, 0.0)

        # advance: current preview geometry becomes the NEXT bar's previous-known
        # structure (mirrors production).
        prev_geom_by_tf = {
            tf: (
                geom_by_tf[tf][0],
                geom_by_tf[tf][1],
                geom_by_tf[tf][2],
                geom_by_tf[tf][3],
            )
            for tf in TF_ORDER
        }
    return result


# --------------------------------------------------------------------------- #
# Semantic join to R2 Q artifact + labels + weights                             #
# --------------------------------------------------------------------------- #
def join_with_r2(
    feature_df: pd.DataFrame,
    symbol: str,
    counters: Core108Counters,
    artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Join CORE108 rows to the persisted R2 Q artifact on (symbol, decision_time).

    Produces Y_L / Y_S, sample weights via composite GlobalEpisodeKey, and an
    integrity report. Never joins by row position (RC5).
    """
    counters.disk_read_count += 1
    art = load_oracle_artifact_v2(artifact_root, symbol)
    if not art["ok"]:
        raise SystemExit(f"STOP_ARTIFACT_LOAD_FAILED: {art.get('reason')}")
    actions = art["actions"].copy()
    actions["symbol"] = symbol

    df = feature_df.merge(
        actions[
            [
                "symbol",
                "decision_time",
                "proximity_episode_id",
                "proximity_any",
                "training_eligible",
                "label_available_time",
                "Q_F1_L",
                "Q_F1_F",
                "Q_F1_S",
            ]
        ],
        on=["symbol", "decision_time"],
        how="left",
        validate="one_to_one",
    )

    integrity: Dict[str, Any] = {}
    integrity["feature_rows"] = len(feature_df)
    integrity["unmatched_feature_rows"] = int(df["proximity_episode_id"].isna().sum())
    integrity["oracle_rows"] = len(actions)
    integrity["duplicate_feature_keys"] = int(
        feature_df.duplicated(subset=["symbol", "decision_time"]).sum()
    )
    integrity["duplicate_oracle_keys"] = int(
        actions.duplicated(subset=["symbol", "decision_time"]).sum()
    )

    ql = df["Q_F1_L"].to_numpy(float)
    qf = df["Q_F1_F"].to_numpy(float)
    qs = df["Q_F1_S"].to_numpy(float)
    a5 = df["atr5m"].to_numpy(float)
    elig = df["training_eligible"].to_numpy(bool) if "training_eligible" in df else np.ones(len(df), bool)
    prox_any = df["proximity_any"].fillna(False).to_numpy(bool)

    f1_full = np.isfinite(ql) & np.isfinite(qf) & np.isfinite(qs)
    atr_ok = np.isfinite(a5) & (a5 > 0)
    candidate = f1_full & atr_ok & elig & prox_any

    df["Y_L"] = np.where(candidate, (ql - qf) / a5, _NAN)
    df["Y_S"] = np.where(candidate, (qs - qf) / a5, _NAN)
    df["is_candidate"] = candidate
    df["sample_weight"] = 0.0
    df["sample_weight_norm"] = 0.0

    cand_idx = np.where(candidate)[0]
    if len(cand_idx) > 0:
        ep_id = df["proximity_episode_id"].to_numpy()
        df_cand = df.iloc[cand_idx]
        counts = df_cand.groupby(["symbol", "proximity_episode_id"]).size()
        # raw weight 1/N_e
        w_raw = (1.0 / counts.reindex(
            list(zip(df_cand["symbol"], df_cand["proximity_episode_id"]))
        ).to_numpy())
        df.loc[df_cand.index, "sample_weight"] = w_raw
        # normalize to mean 1 across candidate rows
        mean_w = w_raw.mean()
        if mean_w > 0:
            df.loc[df_cand.index, "sample_weight_norm"] = w_raw / mean_w

    integrity["candidate_rows"] = int(candidate.sum())
    integrity["nan_by_family"] = _nan_by_family(df, candidate)
    integrity["feature_count"] = len(core108_columns())
    integrity["math_version"] = art["metadata"].get("math_version")
    integrity["oracle_source_sha"] = art["metadata"].get("oracle_source_sha")
    integrity["cost_mode"] = art["metadata"].get("cost_mode")
    return {"df": df, "integrity": integrity}


def _nan_by_family(df: pd.DataFrame, candidate: np.ndarray) -> Dict[str, int]:
    out: Dict[str, int] = {}
    cand_df = df.iloc[np.where(candidate)[0]]
    for fam in ("distance_atr", "phase", "episode_age_5m", "approach_velocity",
                "path_efficiency", "max_penetration_atr", "trend_state",
                "slope_atr", "dev"):
        cols = [c for c in cand_df.columns if c.endswith("_" + fam)]
        if cols:
            out[fam] = int(cand_df[cols].isna().any(axis=1).sum())
    return out


# --------------------------------------------------------------------------- #
# Orchestration                                                                #
# --------------------------------------------------------------------------- #
def run_production_kernel(
    symbol: str,
    counters: Optional[Core108Counters] = None,
    max_bars: Optional[int] = None,
    artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
    join: bool = True,
    geom_mutator: Optional[Callable[[int, Dict[str, Tuple]], Dict[str, Tuple]]] = None,
) -> Dict[str, Any]:
    counters = counters or Core108Counters()
    info = build_base_prefix(symbol, counters, max_bars=max_bars)
    feature_df = run_streaming(
        info, counters, symbol, max_bars=max_bars, geom_mutator=geom_mutator
    )
    out: Dict[str, Any] = {
        "symbol": symbol,
        "feature_df": feature_df,
        "counters": counters,
        "n": len(info["base"]),
    }
    if join:
        jr = join_with_r2(feature_df, symbol, counters, artifact_root=artifact_root)
        out["joined"] = jr["df"]
        out["integrity"] = jr["integrity"]
    return out


def write_checkpoint_dataset(
    symbol: str,
    out_path: Any,
    prefix_bars: int = 10000,
    artifact_root: Any = DEFAULT_ARTIFACT_ROOT,
) -> Dict[str, Any]:
    """Write a SMALL, REAL checkpoint dataset (RC8).

    ONLY real market data from a TRUE prefix (bar 0..prefix_bars-1) is included;
    synthetic fixtures are never mixed in. Candidates are the labeled rows ready
    for the T1.5 model. Audit/label columns are kept; the model selects X from
    ``core108_columns()`` later.
    """
    import hashlib

    counters = Core108Counters()
    r = run_production_kernel(
        symbol, counters, max_bars=prefix_bars, artifact_root=artifact_root
    )
    df = r["joined"]
    cand = df[df["is_candidate"]].copy()
    keep = core108_columns() + [
        "Y_L",
        "Y_S",
        "sample_weight",
        "sample_weight_norm",
        "symbol",
        "decision_bar_index",
        "decision_time",
        "proximity_episode_id",
        "is_candidate",
    ]
    cand = cand[[c for c in keep if c in cand.columns]]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(out_path, index=False)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    return {
        "path": str(out_path),
        "rows": int(len(cand)),
        "columns": list(cand.columns),
        "bytes": int(out_path.stat().st_size),
        "sha256": sha,
        "counters": counters,
        "integrity": r["integrity"],
    }
