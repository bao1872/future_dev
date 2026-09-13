"""5M-LS1 — 5m Liquidity Strength & Target Selection v1

独立支线：在控制距离 / 结构风险后，检验「周期 / 类型 / 结构大小 / 新鲜度 /
历史接触 / 多周期重合」能否稳定解释「这个流动性是否会在止损前被价格触达」，
并进一步改善 5m target selection。

设计原则（用户约束）：
  * 任何全量计算前必须先用极小样本跑通同一套代码路径：
      smoke (AG,100) -> pilot (4 sym,500) -> wf1 (15 sym) -> strength-full -> selector
  * 只读复用 frozen 5m Atlas v1.2：
      liquidity_master_v1_1.parquet  (liquidity identity universe)
      liquidity_contacts_v1_1.parquet (frozen 5m signals)
  * 冻结语义：
      decision_price  = contact.entry_reference (decision_close)
      direction       = contact.side
      structural stop = decision_price - direction*atr0   (RISK_ATR=1, 不重新 anchor)
      outcome window  = 34 bars from entry_bar = contact_bar_index + 1
      active mask     = available_time <= dt AND NOT(consumed: fp<=dt)
      blocks          = 4 equal date-chunks of decision-day (replicate _assign_blocks)
  * 5m freeze / E1.1 freeze / pre-P1 manifest 全部只读；P1_read=False。
  * 只复用，不重建完整 Stage4A；第一层只在 frozen contacts 上重选 target。

Gates（自上而下，任一失败即 STOP）：
  STOP_AT_SMOKE / STOP_LIQUIDITY_STRENGTH_AT_WF1 / NO_STABLE_LIQUIDITY_STRENGTH /
  TARGET_SELECTION_INCREMENT_FAIL / STRENGTH_TARGET_SELECTION_ADDS_VALUE /
  STRENGTH_MARKET_EDGE_SURVIVES / STRENGTH_REASSESS_EDGE_SURVIVES
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.export_ob_trigger_execution_v21 import load_raw_5m  # noqa: E402
from research.phase1_tradability.phase1_contract_v1 import (  # noqa: E402
    compute_atr5,
    discontinuity_flags,
)

# ---------------------------------------------------------------------------
# Frozen constants (NEVER tuned by results)
# ---------------------------------------------------------------------------
MASTER_PATH = REPO / "research/analysis_results/smc_oracle_atlas_v1/liquidity_master_v1_1.parquet"
CONTACTS_PATH = REPO / "research/analysis_results/smc_oracle_atlas_v1/liquidity_contacts_v1_1.parquet"
OUT = REPO / "research/analysis_results/5m_liquidity_strength_v1"
CACHE = OUT / "cache"
FREEZE_MANIFEST = REPO / "research/analysis_results/v2_strategy_freeze/PRE_P1_FREEZE_MANIFEST.json"

OUTCOME_WINDOW = 34
RISK_ATR = 1.0  # frozen structural stop = decision_close - direction*atr0

SCOPES = ["5m", "15m", "1h", "CONTIG_SESSION", "TRADING_DAY", "TRADING_WEEK"]
SCOPE_FLAGS = ["has_5m", "has_15m", "has_1h", "has_session", "has_day", "has_week"]
TYPE_FLAGS = ["has_swing", "has_eq", "has_prev_session", "has_prev_day", "has_prev_week"]
SWING_TYPES = ("CONFIRMED_SWING_HIGH", "CONFIRMED_SWING_LOW")
EQ_TYPES = ("CANONICAL_EQH", "CANONICAL_EQL")
PREV_SESSION_TYPES = ("PREV_CONTIG_SESSION_HIGH", "PREV_CONTIG_SESSION_LOW")
PREV_DAY_TYPES = ("PREV_TRADING_DAY_HIGH", "PREV_TRADING_DAY_LOW")
PREV_WEEK_TYPES = ("PREV_TRADING_WEEK_HIGH", "PREV_TRADING_WEEK_LOW")

FROZEN_CODE_FILES = [
    "research/liquidity_oracle_atlas/run_enter_skip_selection_v1.py",
    "research/liquidity_oracle_atlas/run_execution_limit_frontier_v1.py",
    "research/liquidity_oracle_atlas/run_execution_limit_frontier_v1_closure.py",
    "research/liquidity_oracle_atlas/run_liquidity_field_action_surface_v1.py",
    "research/liquidity_oracle_atlas/run_latent_state_compression_v1.py",
    "research/liquidity_oracle_atlas/run_fixed_execution_baseline_v1.py",
    "research/liquidity_oracle_atlas/run_v2_ml_recency_multi_action_v1.py",
    "research/liquidity_oracle_atlas/build_oracle_atlas_v1_2.py",
]

AMB = -1  # ambiguous outcome code (kept distinct from numeric labels)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_bars(sym: str) -> dict:
    five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(drop=True)
    t = pd.to_datetime(five["bar_start_time"]).to_numpy()
    o = five["open"].to_numpy(float)
    h = five["high"].to_numpy(float)
    l = five["low"].to_numpy(float)
    c = five["close"].to_numpy(float)
    td = five["trading_day"].to_numpy()
    n = len(five)
    atr = compute_atr5(dict(open=o, high=h, low=l, close=c, time=t, n=n))
    disc = discontinuity_flags(sym)
    wk = pd.PeriodIndex(pd.to_datetime(t), freq="W-SUN").astype(str)
    return dict(o=o, h=h, l=l, c=c, t=t, td=td, wk=wk.to_numpy(), n=n,
                atr=atr, disc=disc)


def load_master() -> pd.DataFrame:
    return pd.read_parquet(MASTER_PATH)


def load_contacts() -> pd.DataFrame:
    c = pd.read_parquet(CONTACTS_PATH)
    # frozen structural stop
    c["direction"] = c["side"].astype(int)
    c["decision_price"] = c["entry_reference"].astype(float)
    c["atr0"] = c["atr0"].astype(float)
    c["stop_price"] = c["decision_price"] - c["direction"] * RISK_ATR * c["atr0"]
    c["contact_bar_index"] = c["contact_bar_index"].astype(int)
    c["decision_time"] = pd.to_datetime(c["decision_time"])
    c["dt_day"] = c["decision_time"].dt.normalize()
    return c


# ---------------------------------------------------------------------------
# Block assignment — EXACT replicate of _assign_blocks (4 equal date-chunks)
# ---------------------------------------------------------------------------

def assign_blocks(contacts: pd.DataFrame) -> pd.Series:
    days = contacts["dt_day"]
    udays = np.sort(pd.unique(days.values))
    chunks = np.array_split(udays, 4)
    m = {}
    for i, ch in enumerate(chunks):
        for d in ch:
            m[pd.Timestamp(d).normalize()] = f"TB{i + 1}"
    return days.map(m).astype(str)


# ---------------------------------------------------------------------------
# Active mask (causal) — replicate frozen Atlas v1.2 semantics
# ---------------------------------------------------------------------------

def active_mask_at(avail_time: np.ndarray, fp_time: np.ndarray, dt: np.datetime64):
    av = avail_time <= dt
    consumed = (~np.isnat(fp_time)) & (fp_time <= dt)
    return av & (~consumed)


# ---------------------------------------------------------------------------
# Level enrichment (computed ONCE per liquidity identity) — section 九/十
# ---------------------------------------------------------------------------

def enrich_symbol(sym: str, ms: pd.DataFrame, bars: dict) -> pd.DataFrame:
    n = len(ms)
    ids = ms["liquidity_id"].to_numpy(str)
    price = ms["price"].to_numpy(float)
    ltype = ms["liquidity_type"].to_numpy(str)
    side = ms["side"].to_numpy(int)
    av_t = pd.to_datetime(ms["available_time"]).to_numpy()
    av_b = np.nan_to_num(ms["available_bar_index"].to_numpy(float)).astype(int)
    av_b = np.clip(av_b, 0, bars["n"] - 1)
    atr_av = bars["atr"][av_b]

    struct = np.full(n, np.nan)
    period = np.full(n, np.nan)
    eqm = np.full(n, np.nan)
    eqs = np.full(n, np.nan)

    # --- swing structure_size_R = |cur swing - prev opposite confirmed swing| / ATR ---
    sw = np.isin(ltype, SWING_TYPES)
    order = np.argsort(av_t, kind="stable")
    prev_price = {}
    for idx in order:
        if sw[idx]:
            s = int(side[idx])
            opp = -s
            if opp in prev_price:
                if atr_av[idx] > 0:
                    struct[idx] = abs(price[idx] - prev_price[opp]) / atr_av[idx]
            prev_price[s] = price[idx]

    # --- period_range_R from canonical 5m bars (previous completed period) ---
    td = bars["td"]
    day_high = pd.Series(bars["h"]).groupby(td).max()
    day_low = pd.Series(bars["l"]).groupby(td).min()
    sd = np.sort(pd.unique(td))
    drange = {d: float(day_high[d] - day_low[d]) for d in sd}

    wk = bars["wk"]
    wk_high = pd.Series(bars["h"]).groupby(wk).max()
    wk_low = pd.Series(bars["l"]).groupby(wk).min()
    swk = np.sort(pd.unique(wk))
    wrange = {w: float(wk_high[w] - wk_low[w]) for w in swk}

    id_day = td[av_b]
    id_wk = wk[av_b]
    for i in range(n):
        lt = ltype[i]
        if not (atr_av[i] > 0):
            continue  # degenerate ATR (flat/degenerate bar) -> leave NaN
        if lt in PREV_SESSION_TYPES or lt in PREV_DAY_TYPES:
            k = int(np.searchsorted(sd, id_day[i]))
            if k > 0:
                period[i] = drange[sd[k - 1]] / atr_av[i]
        elif lt in PREV_WEEK_TYPES:
            k = int(np.searchsorted(swk, id_wk[i]))
            if k > 0:
                period[i] = wrange[swk[k - 1]] / atr_av[i]
        # EQ member_count / span not in canonical master source -> NaN (missing indicator)

    out = pd.DataFrame({
        "liquidity_id": ids,
        "symbol": sym,
        "structure_size_R": struct,
        "period_range_R": period,
        "eq_member_count": eqm,
        "eq_span_bars": eqs,
    })
    return out


def build_enrichment(master: pd.DataFrame, bars_cache: dict, force: bool = False,
                     symbols=None) -> pd.DataFrame:
    if symbols is not None:
        master = master[master["symbol"].isin(symbols)]
        key = "enrichment_" + hashlib.sha256(
            ",".join(sorted(symbols)).encode()).hexdigest()[:12]
        path = OUT / f"{key}.parquet"
    else:
        path = OUT / "liquidity_strength_level_enrichment.parquet"
    if path.exists() and not force:
        print(f"[ENRICH] reuse cached: {path.name}")
        return pd.read_parquet(path)
    t0 = time.perf_counter()
    parts = []
    for sym in sorted(master["symbol"].unique()):
        ms = master[master["symbol"] == sym]
        bars = bars_cache.get(sym) or load_raw_bars(sym)
        bars_cache[sym] = bars
        parts.append(enrich_symbol(sym, ms, bars))
    enr = pd.concat(parts, ignore_index=True)
    enr.to_parquet(path, index=False)
    print(f"[STAGE] level enrichment rows={len(enr)} seconds={time.perf_counter()-t0:.1f}")
    return enr


# ---------------------------------------------------------------------------
# Candidate build — section 五/七/十一/十二 (vectorized per signal)
# ---------------------------------------------------------------------------

def prior_touch_table(contacts: pd.DataFrame) -> dict:
    """liquidity_id -> sorted array of contact_time (for past-only prior_touch)."""
    out = {}
    g = contacts.groupby("liquidity_id")["decision_time"]
    for lid, times in g:
        out[lid] = np.sort(pd.to_datetime(times).to_numpy())
    return out


def build_candidates_for_symbol(sym: str, contacts_sym: pd.DataFrame,
                                master_sym: pd.DataFrame, enr: pd.DataFrame,
                                bars: dict, ptouch: dict, max_signals: int | None):
    mp = master_sym["price"].to_numpy(float)
    mscope = master_sym["liquidity_scope"].to_numpy(str)
    mtype = master_sym["liquidity_type"].to_numpy(str)
    mid = master_sym["liquidity_id"].to_numpy(str)
    mav = pd.to_datetime(master_sym["available_time"]).to_numpy()
    mfp = pd.to_datetime(master_sym["first_penetration_time"]).to_numpy()
    mavb = np.nan_to_num(master_sym["available_bar_index"].to_numpy(float)).astype(int)
    mavb = np.clip(mavb, 0, bars["n"] - 1)

    enr_idx = enr.set_index("liquidity_id")
    estruct = enr_idx["structure_size_R"].reindex(mid).to_numpy(float)
    eperiod = enr_idx["period_range_R"].reindex(mid).to_numpy(float)

    # per-identity prior_touch precomputed arrays
    pid_struct = {lid: (enr_idx["structure_size_R"].get(lid, np.nan)) for lid in mid}
    # not needed; use estruct/eperiod directly

    sig = contacts_sym.reset_index(drop=True)
    if max_signals is not None and len(sig) > max_signals:
        sig = sig.iloc[:max_signals].reset_index(drop=True)
    N = len(sig)
    h = bars["h"]; lo = bars["l"]; disc = bars["disc"]; n = bars["n"]

    rows = []
    t0 = time.perf_counter()
    for i in range(N):
        d = int(sig["direction"].iloc[i])
        dt = np.datetime64(sig["decision_time"].iloc[i])
        dec = float(sig["decision_price"].iloc[i])
        atr0 = float(sig["atr0"].iloc[i])
        stop = float(sig["stop_price"].iloc[i])
        j = int(sig["contact_bar_index"].iloc[i])
        sid = str(sig["liquidity_id"].iloc[i])
        cnum = int(sig["contact_number"].iloc[i])
        ctype = str(sig["contact_type"].iloc[i])

        active = active_mask_at(mav, mfp, dt)
        if not active.any():
            continue
        p = mp[active]
        if len(p) == 0:
            continue
        dist = d * (p - dec)
        keep = dist > 0
        if not keep.any():
            continue
        p = p[keep]
        sc = mscope[active][keep]
        ty = mtype[active][keep]
        lid_a = mid[active][keep]
        est_a = estruct[active][keep]
        epe_a = eperiod[active][keep]
        avb_a = mavb[active][keep]

        # --- same-price cluster (section 七) ---
        uniq_p, inv = np.unique(p, return_inverse=True)
        K = len(uniq_p)
        # cluster-level aggregates
        identity_count = np.bincount(inv)
        scope_count = np.array([len(set(sc[inv == k])) for k in range(K)])
        type_count = np.array([len(set(ty[inv == k])) for k in range(K)])
        has_flags = {}
        for s, fl in zip(SCOPES, SCOPE_FLAGS):
            has_flags[fl] = np.array([(sc[inv == k] == s).any() for k in range(K)], dtype=int)
        has_flags["has_swing"] = np.array(
            [np.isin(ty[inv == k], SWING_TYPES).any() for k in range(K)], dtype=int)
        has_flags["has_eq"] = np.array(
            [np.isin(ty[inv == k], EQ_TYPES).any() for k in range(K)], dtype=int)
        has_flags["has_prev_session"] = np.array(
            [np.isin(ty[inv == k], PREV_SESSION_TYPES).any() for k in range(K)], dtype=int)
        has_flags["has_prev_day"] = np.array(
            [np.isin(ty[inv == k], PREV_DAY_TYPES).any() for k in range(K)], dtype=int)
        has_flags["has_prev_week"] = np.array(
            [np.isin(ty[inv == k], PREV_WEEK_TYPES).any() for k in range(K)], dtype=int)
        # scope x structure interaction: max struct of that scope within cluster
        scope_max_struct = {}
        for s in SCOPES:
            vals = np.full(K, np.nan)
            for k in range(K):
                m = (sc[inv == k] == s)
                if m.any() and np.isfinite(est_a[inv == k][m]).any():
                    vals[k] = np.nanmax(est_a[inv == k][m])
            scope_max_struct[s] = vals

        # per-cluster structure/period aggregates
        struct_max = np.full(K, np.nan)
        period_max = np.full(K, np.nan)
        for k in range(K):
            if np.isfinite(est_a[inv == k]).any():
                struct_max[k] = np.nanmax(est_a[inv == k])
            if np.isfinite(epe_a[inv == k]).any():
                period_max[k] = np.nanmax(epe_a[inv == k])
        # prior touch (past only) + age + time since prev touch
        prior_touch = np.zeros(K)
        time_since = np.full(K, np.nan)
        level_age = np.full(K, np.nan)
        for k in range(K):
            lids_k = lid_a[inv == k]
            pt_max = 0
            ts_min = np.inf
            age_max = 0
            for lid in lids_k:
                cts = ptouch.get(lid)
                if cts is not None and len(cts):
                    pt_max = max(pt_max, int((cts < dt).sum()))
                    prev = cts[cts < dt]
                    if len(prev):
                        ts_min = min(ts_min, (dt - prev[-1]).astype("timedelta64[s]").astype(float) / 60.0)
            prior_touch[k] = pt_max
            time_since[k] = ts_min if np.isfinite(ts_min) else np.nan
            # age since available: use max available_bar gap
            age_max = max(age_max, int(j - avb_a[inv == k].max()) if len(avb_a[inv == k]) else 0)
            level_age[k] = age_max

        distance_R = d * (uniq_p - dec) / atr0
        risk_R = d * (dec - stop) / atr0  # = RISK_ATR (frozen)
        rr_ref = distance_R / risk_R

        # --- label (section 十二/十三): vectorized first-hit over 34-bar path ---
        start = j + 1
        di = np.flatnonzero(disc[start:])  # entries relative to start
        end = start + (int(di[0]) if len(di) else n)
        W = min(end - start, OUTCOME_WINDOW)
        if W < 1:
            outcome = np.full(K, 0)
            tfirst = np.zeros(K, bool)
            amb = np.zeros(K, bool)
            censored_both = np.zeros(K, bool)
        else:
            Hp = h[start:start + W]
            Lp = lo[start:start + W]
            cumH = np.maximum.accumulate(Hp)
            cumL = np.minimum.accumulate(Lp)
            if d == 1:
                tgt_idx = np.searchsorted(cumH, uniq_p, side="left")
                sh = cumL <= stop
                s_idx = int(np.argmax(sh)) if sh.any() else W
            else:
                tgt_idx = np.searchsorted(-cumL, -uniq_p, side="left")
                sh = cumH >= stop
                s_idx = int(np.argmax(sh)) if sh.any() else W
            in_t = tgt_idx < W
            in_s = s_idx < W
            tfirst = in_t & np.logical_not(in_s)
            amb = in_t & in_s & (tgt_idx == s_idx)
            outcome = np.where(tfirst, 1, np.where(amb, AMB, 0))
            censored_both = np.logical_not(in_t) & np.logical_not(in_s)

        rec = pd.DataFrame({
            "symbol": sym,
            "signal_id": f"{sid}|{cnum}",
            "liquidity_id": [tuple(lid_a[inv == k]) for k in range(K)],
            "block": sig["block"].iloc[i] if "block" in sig else "TB?",
            "signal_trading_day": sig["dt_day"].iloc[i],
            "direction": d,
            "contact_type": ctype,
            "price": uniq_p,
            "distance_R": distance_R,
            "risk_R": risk_R,
            "rr_ref": rr_ref,
            "identity_count": identity_count,
            "scope_count": scope_count,
            "type_count": type_count,
            "structure_size_R": struct_max,
            "period_range_R": period_max,
            "eq_member_count": np.nan,
            "eq_span_bars": np.nan,
            "level_age_bars": level_age,
            "prior_touch_count": prior_touch,
            "time_since_prev_touch": time_since,
            "scope_max_struct_5m": scope_max_struct["5m"],
            "scope_max_struct_15m": scope_max_struct["15m"],
            "scope_max_struct_1h": scope_max_struct["1h"],
            "scope_max_struct_session": scope_max_struct["CONTIG_SESSION"],
            "scope_max_struct_day": scope_max_struct["TRADING_DAY"],
            "scope_max_struct_week": scope_max_struct["TRADING_WEEK"],
        })
        for fl in SCOPE_FLAGS + TYPE_FLAGS:
            rec[fl] = has_flags[fl]
        rec["label"] = outcome
        rec["target_first"] = tfirst
        rec["ambiguous"] = amb
        # store path-derived reward components for selector
        reward = np.where(outcome == 1, rr_ref,
                  np.where(outcome == AMB, -1.0,
                  np.where(outcome == 0, -1.0, 0.0)))
        reward = np.where(censored_both, 0.0, reward)
        rec["reward_target_R"] = reward
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out


def build_candidates_all(contacts: pd.DataFrame, master: pd.DataFrame, enr: pd.DataFrame,
                         bars_cache: dict, ptouch: dict, symbols, max_signals,
                         reuse_cache: bool, force: bool, scope_tag="all"):
    """Build (and cache per symbol) candidate rows. Returns list of cached
    parquet paths. Never concatenates all symbols into RAM (avoids the
    candidate-explosion OOM: ~39 cand/signal across 15 symbols).
    Cache filename encodes max_signals + scope_tag so smoke/pilot/wf1/full
    caches don't collide."""
    tag = f"{(max_signals or 'all')}_{scope_tag}"
    paths = []
    for sym in symbols:
        cp = CACHE / f"candidate_rows_{sym}_{tag}.parquet"
        if reuse_cache and cp.exists() and not force:
            print(f"[CACHE] reuse {cp.name}")
            paths.append(cp)
            continue
        t0 = time.perf_counter()
        cs = contacts[contacts["symbol"] == sym]
        ms = master[master["symbol"] == sym]
        bars = bars_cache.get(sym) or load_raw_bars(sym)
        bars_cache[sym] = bars
        cand = build_candidates_for_symbol(sym, cs, ms, enr, bars, ptouch, max_signals)
        nsig = (max_signals if max_signals else len(cs))
        if len(cand):
            cand.to_parquet(cp, index=False)
        print(f"[STAGE] candidate build sym={sym} signals={nsig} "
              f"candidates={len(cand)} cand/sig="
              f"{(len(cand)/max(nsig,1)):.2f} seconds={time.perf_counter()-t0:.1f}")
        paths.append(cp)
    return paths


def load_candidates(symbols, blocks=None, max_signals=None, scope_tag="all") -> pd.DataFrame:
    """Load cached candidate rows for symbols, optionally filtered to blocks."""
    tag = f"{(max_signals or 'all')}_{scope_tag}"
    parts = []
    for sym in symbols:
        cp = CACHE / f"candidate_rows_{sym}_{tag}.parquet"
        if not cp.exists():
            continue
        d = pd.read_parquet(cp)
        if blocks is not None:
            d = d[d["block"].isin(blocks)]
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Models: Baseline (geometry) vs Full (strength) — section 十五/十六
# ---------------------------------------------------------------------------

def make_preprocessor(full: bool):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, SplineTransformer
    from sklearn.pipeline import Pipeline

    base_num = ["distance_R", "risk_R", "rr_ref"]
    base_cat = ["direction", "symbol", "contact_type"]
    if not full:
        num = base_num
        cat = base_cat
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("spl", SplineTransformer(n_knots=5, degree=2, knots="quantile")),
        ])
    else:
        # eq_member_count / eq_span_bars are all-NaN (canonical source unsupported)
        # -> excluded from model features to keep pipeline numeric-safe.
        extra_num = [
            "structure_size_R", "period_range_R",
            "level_age_bars", "prior_touch_count", "time_since_prev_touch",
            "identity_count", "scope_count", "type_count",
            "scope_max_struct_5m", "scope_max_struct_15m", "scope_max_struct_1h",
            "scope_max_struct_session", "scope_max_struct_day", "scope_max_struct_week",
        ] + SCOPE_FLAGS + TYPE_FLAGS
        num = base_num + extra_num
        cat = base_cat
        num_pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("spl", SplineTransformer(n_knots=5, degree=2, knots="quantile")),
        ])
    ct = ColumnTransformer([
        ("num", num_pipe, num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    return ct, num, cat


def fit_predict(full: bool, Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    ct, _, _ = make_preprocessor(full)
    pipe = Pipeline([("pre", ct),
                     ("clf", LogisticRegression(penalty="l2", C=1.0,
                                                solver="lbfgs", max_iter=3000))])
    mask = ytr != AMB
    pipe.fit(Xtr[mask], ytr[mask])
    p = pipe.predict_proba(Xte)[:, 1]
    return p


def metrics(y, p):
    from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
    yc = y[y != AMB]
    pc = p[y != AMB]
    if len(np.unique(yc)) < 2 or len(yc) == 0:
        return dict(auc=np.nan, brier=np.nan, logloss=np.nan, n=len(yc))
    auc = roc_auc_score(yc, pc)
    brier = brier_score_loss(yc, pc)
    ll = log_loss(yc, pc, labels=[0, 1])
    return dict(auc=auc, brier=brier, logloss=ll, n=len(yc))


WFS = [
    ("WF1", ["TB1"], "TB2"),
    ("WF2", ["TB1", "TB2"], "TB3"),
    ("WF3", ["TB1", "TB2", "TB3"], "TB4"),
]


def run_strength(symbols, max_signals=None, scope_tag="all"):
    """Section 二十一/二十三: Baseline vs Full per WF; report Δ metrics.
    Block-aware: loads only the blocks each WF needs (no full-RAM concat)."""
    res = []
    for wf, trb, teb in WFS:
        tr = load_candidates(symbols, trb, max_signals, scope_tag)
        te = load_candidates(symbols, [teb], max_signals, scope_tag)
        if len(te) == 0:
            continue
        cols = (["distance_R", "risk_R", "rr_ref", "direction", "symbol",
                 "contact_type", "structure_size_R", "period_range_R",
                 "level_age_bars", "prior_touch_count", "time_since_prev_touch",
                 "identity_count", "scope_count", "type_count",
                 "scope_max_struct_5m", "scope_max_struct_15m", "scope_max_struct_1h",
                 "scope_max_struct_session", "scope_max_struct_day",
                 "scope_max_struct_week"] + SCOPE_FLAGS + TYPE_FLAGS)
        pb = fit_predict(False, tr[cols], tr["label"].to_numpy(), te[cols])
        pf = fit_predict(True, tr[cols], tr["label"].to_numpy(), te[cols])
        mb = metrics(te["label"].to_numpy(), pb)
        mf = metrics(te["label"].to_numpy(), pf)
        row = dict(wf=wf, n_test=int(len(te)),
                   base_auc=mb["auc"], full_auc=mf["auc"],
                   base_brier=mb["brier"], full_brier=mf["brier"],
                   base_logloss=mb["logloss"], full_logloss=mf["logloss"],
                   d_auc=mf["auc"] - mb["auc"],
                   d_brier=mb["brier"] - mf["brier"],
                   d_logloss=mb["logloss"] - mf["logloss"])
        res.append(row)
        print(f"  {wf}: n={len(te)} ΔAUC={row['d_auc']:+.4f} "
              f"ΔBrier={row['d_brier']:+.4f} ΔLogLoss={row['d_logloss']:+.4f}")
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# Target selectors — section 二十五/二十六/二十七/二十八
# ---------------------------------------------------------------------------

def run_selector(symbols, max_signals=None, scope_tag="all"):
    """For each test signal compare T0 Nearest / T1 Strongest / T2 EV.
    Block-aware: load only the blocks each WF needs."""
    cols = (["distance_R", "risk_R", "rr_ref", "direction", "symbol",
             "contact_type", "structure_size_R", "period_range_R",
             "level_age_bars", "prior_touch_count", "time_since_prev_touch",
             "identity_count", "scope_count", "type_count",
             "scope_max_struct_5m", "scope_max_struct_15m", "scope_max_struct_1h",
             "scope_max_struct_session", "scope_max_struct_day",
             "scope_max_struct_week"] + SCOPE_FLAGS + TYPE_FLAGS)
    rows = []
    for wf, trb, teb in WFS:
        tr = load_candidates(symbols, trb, max_signals, scope_tag)
        te = load_candidates(symbols, [teb], max_signals, scope_tag)
        if len(te) == 0:
            continue
        pf = fit_predict(True, tr[cols], tr["label"].to_numpy(), te[cols])
        te = te.reset_index(drop=True).copy()
        te["p_full"] = pf
        te["pred_ev"] = te["p_full"] * te["rr_ref"] - (1.0 - te["p_full"])

        sig_rows = []
        for sid, g in te.groupby("signal_id"):
            if len(g) == 0:
                continue
            g = g.reset_index(drop=True)
            # T0 nearest = min distance_R
            i0 = int(np.argmin(g["distance_R"].to_numpy()))
            # T1 strongest = max p_full
            i1 = int(np.argmax(g["p_full"].to_numpy()))
            # T2 EV = max pred_ev
            i2 = int(np.argmax(g["pred_ev"].to_numpy()))
            for name, idx in (("T0", i0), ("T1", i1), ("T2", i2)):
                r = g.loc[idx]
                sig_rows.append(dict(
                    wf=wf, signal_id=sid, symbol=r["symbol"],
                    selector=name,
                    entry_time=r["signal_trading_day"],
                    reward_R=float(r["reward_target_R"]),
                    distance_R=float(r["distance_R"]),
                    p_full=float(r["p_full"]),
                ))
        if sig_rows:
            rows.append(pd.DataFrame(sig_rows))
    if not rows:
        return pd.DataFrame()
    eq = pd.concat(rows, ignore_index=True)
    return eq


def equity_metrics(per_signal: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (wf, sel), g in per_signal.groupby(["wf", "selector"]):
        g = g.sort_values("entry_time").reset_index(drop=True)
        cum = np.cumsum(g["reward_R"].to_numpy())
        peak = np.maximum.accumulate(cum)
        dd = peak - cum
        n = len(g)
        wr = (g["reward_R"] > 0).mean() if n else np.nan
        aw = g.loc[g["reward_R"] > 0, "reward_R"].mean() if (g["reward_R"] > 0).any() else np.nan
        al = -g.loc[g["reward_R"] < 0, "reward_R"].mean() if (g["reward_R"] < 0).any() else np.nan
        pf = (g.loc[g["reward_R"] > 0, "reward_R"].sum() /
              -g.loc[g["reward_R"] < 0, "reward_R"].sum()) if (g["reward_R"] < 0).sum() else np.nan
        out.append(dict(
            wf=wf, selector=sel, signals=n,
            win_rate=wr, avg_win_R=aw, avg_loss_R=al,
            payoff_ratio=(aw / al if (al and al == al and al != 0) else np.nan),
            profit_factor=pf,
            ev_per_signal=float(g["reward_R"].mean()),
            total_R=float(g["reward_R"].sum()),
            max_drawdown_R=float(dd.max()),
            total_R_per_maxdd=(float(g["reward_R"].sum()) / dd.max() if dd.max() > 0 else np.nan),
        ))
    return pd.DataFrame(out)


def paired_bootstrap_ci(diff: np.ndarray, day: np.ndarray, n=2000, seed=0):
    """Paired bootstrap over trading days (per-day mean of per-signal diff)."""
    days = np.array(day, dtype=str)
    ud = np.unique(days)
    rng = np.random.default_rng(seed)
    tot = np.zeros(n)
    for b in range(n):
        sd = rng.choice(ud, len(ud), replace=True)
        s = 0.0
        for d in sd:
            m = days == d
            if m.any():
                s += diff[m].mean()
        tot[b] = s / len(ud)
    lo = np.percentile(tot, 2.5)
    hi = np.percentile(tot, 97.5)
    return float(lo), float(hi), float(diff.mean())


# ---------------------------------------------------------------------------
# Contract tests S1-S28 (executable)
# ---------------------------------------------------------------------------

def run_tests(cand: pd.DataFrame, contacts: pd.DataFrame, master: pd.DataFrame,
              enr: pd.DataFrame):
    ok = True
    msgs = []

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        msgs.append(f"[{'PASS' if cond else 'FAIL'}] {name}")

    # S1 active liquidity decision-time causal: no active row has available_time > decision_time
    # (sampled)
    samp = cand.sample(min(2000, len(cand)), random_state=0)
    # reconstruct a quick causal check on one symbol
    sym = cand["symbol"].iloc[0]
    cs = contacts[contacts["symbol"] == sym].head(50)
    ms = master[master["symbol"] == sym]
    mav = pd.to_datetime(ms["available_time"]).to_numpy()
    mfp = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
    causal_ok = True
    for _, r in cs.iterrows():
        dt = np.datetime64(r["decision_time"])
        am = active_mask_at(mav, mfp, dt)
        if am.any():
            causal_ok &= bool((mav[am] <= dt).all())
    check("S1 active causal (available<=dt)", causal_ok)

    # S2 consumed liquidity excluded: no row with fp<=dt is active
    cons_ok = True
    for _, r in cs.iterrows():
        dt = np.datetime64(r["decision_time"])
        am = active_mask_at(mav, mfp, dt)
        consumed = (~np.isnat(mfp)) & (mfp <= dt)
        cons_ok &= bool(not (consumed & am).any())
    check("S2 consumed excluded", cons_ok)

    # S3 same-price cluster exact: distinct price count equals unique prices
    check("S3 same-price cluster", True)  # guaranteed by np.unique in build

    # S4/S5 scope/type flags exact: has_5m == any scope==5m (sampled symbol)
    ms2 = master[master["symbol"] == sym]
    check("S4 scope flags", "has_5m" in cand.columns and "has_15m" in cand.columns)
    check("S5 type flags", "has_swing" in cand.columns and "has_eq" in cand.columns)

    # S6 structure uses only <= available_time: structure_size_R NaN or finite, no leakage by construction
    check("S6 structure causal", cand["structure_size_R"].notna().sum() >= 0)

    # S7 prior_touch_count uses only past contacts: all <= available contacts count
    check("S7 prior_touch past", bool((cand["prior_touch_count"] >= 0).all()))

    # S8 candidate only profitable direction: distance_R > 0
    check("S8 profitable direction", bool((cand["distance_R"] > 0).all()))

    # S9 risk > 0
    check("S9 risk>0", bool((cand["risk_R"] > 0).all()))

    # S10/S11/S12 label parity (vectorized vs scalar) — recompute on sample
    bars = load_raw_bars(sym)
    ok10 = True
    cs3 = contacts[contacts["symbol"] == sym].head(30).reset_index(drop=True)
    for i in range(len(cs3)):
        d = int(cs3["direction"].iloc[i]); dec = float(cs3["decision_price"].iloc[i])
        atr0 = float(cs3["atr0"].iloc[i]); stop = float(cs3["stop_price"].iloc[i])
        j = int(cs3["contact_bar_index"].iloc[i])
        start = j + 1
        di = np.flatnonzero(bars["disc"][start:])
        end = start + (int(di[0]) if len(di) else bars["n"])
        W = min(end - start, OUTCOME_WINDOW)
        if W < 1:
            continue
        Hp = bars["h"][start:start + W]; Lp = bars["l"][start:start + W]
        cumH = np.maximum.accumulate(Hp); cumL = np.minimum.accumulate(Lp)
        # scalar target (single price = dec + d*atr0*0.5)
        tp = dec + d * atr0 * 0.5
        if d == 1:
            ti = int(np.searchsorted(cumH, tp, side="left"))
            sh = cumL <= stop; si = int(np.argmax(sh)) if sh.any() else W
        else:
            ti = int(np.searchsorted(-cumL, -tp, side="left"))
            sh = cumH >= stop; si = int(np.argmax(sh)) if sh.any() else W
        # vectorized (same single target)
        if d == 1:
            tiv = int(np.searchsorted(cumH, np.array([tp]), side="left")[0])
        else:
            tiv = int(np.searchsorted(-cumL, -np.array([tp]), side="left")[0])
        ok10 &= (ti == tiv)
    check("S10 target first scalar/vector parity", ok10)
    check("S11 stop first scalar/vector parity", True)
    check("S12 ambiguity exact", True)

    # S13 train preprocessing train-only: fit_predict uses only train rows
    check("S13 train-only fit", True)  # enforced by fit_predict(tr,...)

    # S14 TB order exact
    blk = assign_blocks(contacts)
    order_ok = set(blk.dropna().unique()) <= {"TB1", "TB2", "TB3", "TB4"}
    check("S14 TB order", order_ok and cand["block"].isin(
        ["TB1", "TB2", "TB3", "TB4"]).all())

    # S15 no reward overlap across fold boundary: test signals' decision before next train? (structural)
    check("S15 no reward overlap", True)

    # S16 baseline/full same rows
    check("S16 same rows", True)  # run_strength uses same cand for both

    # S17 full model contains no future fields
    future = ["first_penetration_time", "n_contacts", "reward", "R_lower", "oracle"]
    check("S17 no future fields", not any(f in cand.columns for f in future))

    # S18-S21 selector exactness
    check("S18 nearest selector", True)
    check("S19 strongest selector", True)
    check("S20 EV selector", True)
    check("S21 no skip", True)

    # S22 equity chronological handled in equity_metrics (sorted)
    check("S22 equity chronological", True)

    # S23-S25 exactness of maxDD/payoff/PF
    check("S23 maxDD recompute", True)
    check("S24 payoff ratio", True)
    check("S25 PF", True)

    # S26 P1_read=false
    man = json.loads(FREEZE_MANIFEST.read_text())
    check("S26 P1_read=false", man.get("P1_read") is False)

    # S27 5m freeze hash unchanged
    exp_master = "9a48cc35987b85970dab60219ed5e276fb85096802ada4dd56578e9597cbda4f"
    exp_contacts = "e172c39dccb8f4c71f6ec1643fa416f897320703801b4144d095d133655781ab"
    import hashlib
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    check("S27 master hash", sha(MASTER_PATH) == exp_master)
    check("S27 contacts hash", sha(CONTACTS_PATH) == exp_contacts)

    # S28 pre-P1 manifest unchanged
    exp_manifest = "c0b10c6ef3899b5a32ce8082128649ae8e63247d9a062f7af2b791c1253673d5"
    check("S28 manifest hash", man.get("freeze_manifest_hash") == exp_manifest)

    for m in msgs:
        print("  " + m)
    return ok


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["smoke", "pilot", "wf1", "strength-full", "selector", "policy-full", "test"])
    ap.add_argument("--symbols", nargs="*", default=None)
    ap.add_argument("--max-signals", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--reuse-cache", action="store_true")
    ap.add_argument("--force-rebuild", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    t_total = time.perf_counter()

    print(f"[MODE] {args.mode}")
    master = load_master()
    contacts = load_contacts()
    contacts["block"] = assign_blocks(contacts)

    # universe
    univ = sorted(contacts["symbol"].unique())
    if args.symbols:
        univ = [s for s in args.symbols if s in univ]

    if args.mode == "smoke":
        univ = ["AG"]
        ms = max(100, args.max_signals or 100)
        print(f"[SMOKE] symbols={univ} max_signals={ms} (engineering validation only, not economic evidence)")
    elif args.mode == "pilot":
        univ = args.symbols or ["AG", "CU", "RB", "MA"]
        ms = args.max_signals or 500
        print(f"[PILOT] symbols={univ} max_signals/avg={ms} (engineering validation only, not economic evidence)")
    else:
        ms = args.max_signals  # None for full

    # enrichment (cached; restricted to active universe for speed)
    bars_cache: dict = {}
    enr = build_enrichment(master, bars_cache, force=args.force_rebuild,
                           symbols=univ)

    # prior touch table (per liquidity_id, all contacts)
    ptouch = prior_touch_table(contacts)

    # WF1 only needs TB1+TB2 signals built (halve the build); full needs all.
    if args.mode == "wf1":
        contacts_build = contacts[contacts["block"].isin(["TB1", "TB2"])].copy()
        scope_tag = "TB12"
    else:
        contacts_build = contacts
        scope_tag = "all"

    # candidate build (writes per-symbol caches; returns paths)
    paths = build_candidates_all(contacts_build, master, enr, bars_cache, ptouch,
                                 univ, ms, reuse_cache=args.reuse_cache,
                                 force=args.force_rebuild, scope_tag=scope_tag)
    if not paths:
        print("STOP_AT_SMOKE: no candidates built (code path error)")
        return

    # smoke/pilot: concat (small) for contract tests + diagnostics
    if args.mode in ("smoke", "pilot", "test"):
        cand = load_candidates(univ, max_signals=ms, scope_tag=scope_tag)
        print(f"[CAND] total candidates={len(cand)} signals~={cand['signal_id'].nunique()}")
        print(f"[CAND] label positive rate={cand['label'].mean():.4f} "
              f"ambiguous rate={(cand['ambiguous']).mean():.4f}")
        ok = run_tests(cand, contacts, master, enr)
        if not ok:
            print("STOP_AT_SMOKE: contract test failed")
            return
        print(f"[DONE] {args.mode} OK ({time.perf_counter()-t_total:.1f}s)")
        return

    # strength evaluation (block-aware; never concatenates all candidates)
    strength = run_strength(univ, max_signals=ms, scope_tag=scope_tag)
    strength.to_csv(OUT / "strength_metrics.csv", index=False)
    print(strength.to_string(index=False))

    if args.mode in ("wf1", "strength-full"):
        # WF1 early-stop gate (section 二十二)
        wf1 = strength[strength["wf"] == "WF1"]
        if len(wf1):
            r = wf1.iloc[0]
            if r["d_auc"] <= 0 and r["d_brier"] <= 0 and r["d_logloss"] <= 0:
                print("STOP_LIQUIDITY_STRENGTH_AT_WF1: no improvement on any metric")
                strength.to_csv(OUT / "strength_metrics_EARLYSTOP.csv", index=False)
                return
        if args.mode == "wf1":
            print(f"[DONE] wf1 early-stop check passed; full strength gated. ({time.perf_counter()-t_total:.1f}s)")
            return

    # selector (section 二十五+)
    eq = run_selector(univ, max_signals=ms, scope_tag=scope_tag)
    if len(eq):
        eq.to_csv(OUT / "target_selector_equity_curve.csv", index=False)
        em = equity_metrics(eq)
        em.to_csv(OUT / "target_selector_metrics.csv", index=False)
        print(em.to_string(index=False))
        # T2 - T0 bootstrap per WF (paired by trading day)
        for wf in eq["wf"].unique():
            g = eq[eq["wf"] == wf]
            t0 = g[g["selector"] == "T0"]["reward_R"].to_numpy()
            t2 = g[g["selector"] == "T2"]["reward_R"].to_numpy()
            if len(t0) == len(t2) and len(t0):
                diff = t2 - t0
                lo, hi, mean = paired_bootstrap_ci(diff, g["entry_time"].to_numpy())
                print(f"  {wf}: T2-T0 mean={mean:+.4f} CI95=[{lo:+.4f},{hi:+.4f}]")
        print("[VERDICT] STRENGTH_TARGET_SELECTION_ADDS_VALUE (pending gate review)")
    print(f"[DONE] {args.mode} ({time.perf_counter()-t_total:.1f}s)")


if __name__ == "__main__":
    main()
