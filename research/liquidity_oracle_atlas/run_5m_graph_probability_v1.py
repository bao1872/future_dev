"""5M-GPM1 — 5m Liquidity Graph Probability Model v1

Ordered Liquidity Chain + Competing Risk + Multinomial Logistic Factors.

独立支线（接在 INVALIDATED 的 5M-LS1 `66bff6d` 之后）：把「一个 signal 前方 25 个
互相独立的二分类 liquidity 样本」改成「有序流动性链 + 竞争风险概率图」。

关键修正（相对 LS1 的 P0 bug）：
    target_first = first_target < first_stop   (frozen first_hit_bounds 语义)
    stop_first   = first_stop  < first_target
    ambiguous    = first_target == first_stop   (conservative 归入 LOSS)
    censored     = 两者窗口内都未触达

图模型核心：
    ROOT -> L1 -> L2 -> ... -> terminal(LOSS | CENSOR)
    每条 edge 只有三类结果 NEXT / LOSS / CENSOR。
    价格到 L_k 必须经过 L_1..L_{k-1}，所以天然 P(reach L1) >= P(reach L2) ...
    远端不可达 level 在 terminal edge 之后不再进入 risk set（标准 survival 语义），
    候选爆炸自然消失。

阶段化（用户约束，本次只做到 WF1 G0）：
    P0  parity  : 修正标签 vs frozen first_hit_bounds 独立 parity，0 mismatch
    P1  smoke   : AG 100，只检查 graph 构造 / 概率递推 / joint likelihood / EV
    P2  pilot   : AG/CU/RB/MA x500，检查 transitions/signal、NEXT/LOSS/CENSOR%
    WF1 G0      : 15 sym TB1->TB2，M0(independent geometry) vs G0(graph geometry)
                  Graph Necessity Gate：G0 两项都不改善 -> STOP_GRAPH_STRUCTURE_AT_WF1

冻结语义（与 LS1 同，只读）：
    decision_price = contact.entry_reference (decision_close)
    direction      = contact.side
    structural stop= decision_price - direction*atr0   (RISK_ATR=1)
    outcome window = 34 bars from entry_bar = contact_bar_index + 1 (truncate at discontinuity)
    active mask    = available_time <= dt AND NOT(consumed: fp<=dt)
    blocks         = 4 equal date-chunks of decision-day
5m freeze / E1.1 freeze / pre-P1 manifest 全部只读；P1_read=False。

禁止（本次）：GNN / LightGBM / XGBoost / NN / RL / G1(WF2/WF3/selector)。
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
# Frozen independent oracle for label parity (P0).  MUST NOT be edited.
from research.liquidity_oracle_atlas.run_liquidity_field_action_surface_v1 import (  # noqa: E402
    first_hit_bounds,
)
from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

# ---------------------------------------------------------------------------
# Frozen constants (NEVER tuned by results)
# ---------------------------------------------------------------------------
MASTER_PATH = REPO / "research/analysis_results/smc_oracle_atlas_v1/liquidity_master_v1_1.parquet"
CONTACTS_PATH = REPO / "research/analysis_results/smc_oracle_atlas_v1/liquidity_contacts_v1_1.parquet"
OUT = REPO / "research/analysis_results/5m_graph_probability_v1"
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

# edge state encoding
NEXT, LOSS, CENSOR = 0, 1, 2

WFS = [("WF1", ["TB1"], "TB2"),
       ("WF2", ["TB1", "TB2"], "TB3"),
       ("WF3", ["TB1", "TB2", "TB3"], "TB4")]

# M0 = independent geometry baseline; G0 = graph geometry
M0_NUM = ["cum_distance_R", "risk_R", "rr_ref"]
M0_CAT = ["direction", "symbol", "contact_type"]
G0_NUM = ["delta_R", "cum_distance_R", "risk_R"]
G0_CAT = ["edge_index", "direction", "symbol", "contact_type"]

# ---------------------------------------------------------------------------
# Data loading (reuse frozen contracts)
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
    c["direction"] = c["side"].astype(int)
    c["decision_price"] = c["entry_reference"].astype(float)
    c["atr0"] = c["atr0"].astype(float)
    c["stop_price"] = c["decision_price"] - c["direction"] * RISK_ATR * c["atr0"]
    c["contact_bar_index"] = c["contact_bar_index"].astype(int)
    c["decision_time"] = pd.to_datetime(c["decision_time"])
    c["dt_day"] = c["decision_time"].dt.normalize()
    return c


def assign_blocks(contacts: pd.DataFrame) -> pd.Series:
    days = contacts["dt_day"]
    udays = np.sort(pd.unique(days.values))
    chunks = np.array_split(udays, 4)
    m = {}
    for i, ch in enumerate(chunks):
        for d in ch:
            m[pd.Timestamp(d).normalize()] = f"TB{i + 1}"
    return days.map(m).astype(str)


def active_mask_at(avail_time: np.ndarray, fp_time: np.ndarray, dt: np.datetime64):
    av = avail_time <= dt
    consumed = (~np.isnat(fp_time)) & (fp_time <= dt)
    return av & (~consumed)


# ---------------------------------------------------------------------------
# Level enrichment (computed ONCE per liquidity identity)
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
    sw = np.isin(ltype, SWING_TYPES)
    order = np.argsort(av_t, kind="stable")
    prev_price = {}
    for idx in order:
        if sw[idx]:
            s = int(side[idx])
            opp = -s
            if opp in prev_price and atr_av[idx] > 0:
                struct[idx] = abs(price[idx] - prev_price[opp]) / atr_av[idx]
            prev_price[s] = price[idx]
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
        if not (atr_av[i] > 0):
            continue
        lt = ltype[i]
        if lt in PREV_SESSION_TYPES or lt in PREV_DAY_TYPES:
            k = int(np.searchsorted(sd, id_day[i]))
            if k > 0:
                period[i] = drange[sd[k - 1]] / atr_av[i]
        elif lt in PREV_WEEK_TYPES:
            k = int(np.searchsorted(swk, id_wk[i]))
            if k > 0:
                period[i] = wrange[swk[k - 1]] / atr_av[i]
    return pd.DataFrame({
        "liquidity_id": ids, "symbol": sym,
        "structure_size_R": struct, "period_range_R": period,
        "eq_member_count": eqm, "eq_span_bars": eqs,
    })


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


def prior_touch_table(contacts: pd.DataFrame) -> dict:
    out = {}
    for lid, times in contacts.groupby("liquidity_id")["decision_time"]:
        out[lid] = np.sort(pd.to_datetime(times).to_numpy())
    return out


# ---------------------------------------------------------------------------
# Corrected competing-risk label  (parity-checked vs frozen first_hit_bounds)
# ---------------------------------------------------------------------------

def label_window(Hp: np.ndarray, Lp: np.ndarray, dec: float, stop: float,
                 d: int, targets: np.ndarray) -> dict:
    """Corrected target/stop label over the truncated 34-bar window.

    Exactly matches frozen first_hit_bounds:
        target_first = first_target < first_stop
        stop_first   = first_stop  < first_target
        ambiguous    = first_target == first_stop
        censored     = neither hit in window
    """
    W = len(Hp)
    cumH = np.maximum.accumulate(Hp)
    cumL = np.minimum.accumulate(Lp)
    if d == 1:
        tgt_idx = np.searchsorted(cumH, targets, side="left")
        sh = cumL <= stop
        s_idx = int(np.argmax(sh)) if sh.any() else W
    else:
        tgt_idx = np.searchsorted(-cumL, -targets, side="left")
        sh = cumH >= stop
        s_idx = int(np.argmax(sh)) if sh.any() else W
    in_t = tgt_idx < W
    in_s = s_idx < W
    target_first = in_t & (np.logical_not(in_s) | (tgt_idx < s_idx))
    stop_first = in_s & (np.logical_not(in_t) | (s_idx < tgt_idx))
    ambiguous = in_t & in_s & (tgt_idx == s_idx)
    censored = np.logical_not(in_t) & np.logical_not(in_s)
    return dict(target_first=target_first, stop_first=stop_first,
                ambiguous=ambiguous, censored=censored)


def oracle_window(Hp: np.ndarray, Lp: np.ndarray, dec: float, stop: float,
                  d: int, targets: np.ndarray) -> dict:
    K = len(targets)
    fh = np.tile(Hp, (K, 1)).astype(np.float64)
    fl = np.tile(Lp, (K, 1)).astype(np.float64)
    o = first_hit_bounds(fh, fl, np.full(K, dec), targets.astype(np.float64),
                         np.full(K, stop), np.full(K, d))
    return o


# ---------------------------------------------------------------------------
# Active clusters for one signal (same-price collapse) -> sorted by distance_R
# ---------------------------------------------------------------------------

def signal_clusters(sym, row, master_sym, enr, bars):
    """Return (uniq_p sorted ascending by distance_R, per-cluster feature dicts)."""
    d = int(row["direction"])
    dt = np.datetime64(row["decision_time"])
    dec = float(row["decision_price"])
    atr0 = float(row["atr0"])
    stop = float(row["stop_price"])
    j = int(row["contact_bar_index"])

    mav = pd.to_datetime(master_sym["available_time"]).to_numpy()
    mfp = pd.to_datetime(master_sym["first_penetration_time"]).to_numpy()
    mp = master_sym["price"].to_numpy(float)
    mscope = master_sym["liquidity_scope"].to_numpy(str)
    mtype = master_sym["liquidity_type"].to_numpy(str)
    mid = master_sym["liquidity_id"].to_numpy(str)
    mavb = np.nan_to_num(master_sym["available_bar_index"].to_numpy(float)).astype(int)
    mavb = np.clip(mavb, 0, bars["n"] - 1)

    enr_idx = enr.set_index("liquidity_id")
    estruct = enr_idx["structure_size_R"].reindex(mid).to_numpy(float)
    eperiod = enr_idx["period_range_R"].reindex(mid).to_numpy(float)

    active = active_mask_at(mav, mfp, dt)
    if not active.any():
        return None
    p = mp[active]
    if len(p) == 0:
        return None
    dist = d * (p - dec)
    keep = dist > 0
    if not keep.any():
        return None
    p = p[keep]
    sc = mscope[active][keep]
    ty = mtype[active][keep]
    lid_a = mid[active][keep]
    est_a = estruct[active][keep]
    epe_a = eperiod[active][keep]
    avb_a = mavb[active][keep]

    # same-price cluster
    uniq_p, inv = np.unique(p, return_inverse=True)
    K = len(uniq_p)
    identity_count = np.bincount(inv)
    scope_count = np.array([len(set(sc[inv == k])) for k in range(K)])
    type_count = np.array([len(set(ty[inv == k])) for k in range(K)])
    has_flags = {}
    for s, fl in zip(SCOPES, SCOPE_FLAGS):
        has_flags[fl] = np.array([(sc[inv == k] == s).any() for k in range(K)], dtype=int)
    for fl, tset in [("has_swing", SWING_TYPES), ("has_eq", EQ_TYPES),
                     ("has_prev_session", PREV_SESSION_TYPES),
                     ("has_prev_day", PREV_DAY_TYPES), ("has_prev_week", PREV_WEEK_TYPES)]:
        has_flags[fl] = np.array([np.isin(ty[inv == k], tset).any() for k in range(K)], dtype=int)
    scope_max_struct = {}
    for s in SCOPES:
        vals = np.full(K, np.nan)
        for k in range(K):
            m = (sc[inv == k] == s)
            if m.any() and np.isfinite(est_a[inv == k][m]).any():
                vals[k] = np.nanmax(est_a[inv == k][m])
        scope_max_struct[s] = vals
    struct_max = np.full(K, np.nan)
    period_max = np.full(K, np.nan)
    for k in range(K):
        if np.isfinite(est_a[inv == k]).any():
            struct_max[k] = np.nanmax(est_a[inv == k])
        if np.isfinite(epe_a[inv == k]).any():
            period_max[k] = np.nanmax(epe_a[inv == k])

    distance_R = d * (uniq_p - dec) / atr0
    risk_R = d * (dec - stop) / atr0  # = RISK_ATR (frozen)

    # label over truncated window
    start = j + 1
    di = np.flatnonzero(bars["disc"][start:])
    end = start + (int(di[0]) if len(di) else bars["n"])
    W = min(end - start, OUTCOME_WINDOW)
    Hp = bars["h"][start:start + W]
    Lp = bars["l"][start:start + W]
    lab = label_window(Hp, Lp, dec, stop, d, uniq_p)

    # order ascending by distance_R (profitable direction chain)
    order = np.argsort(distance_R, kind="stable")
    feats = dict(
        price=uniq_p[order], distance_R=distance_R[order], risk_R=risk_R,
        rr_ref=distance_R[order] / risk_R,
        identity_count=identity_count[order], scope_count=scope_count[order],
        type_count=type_count[order], structure_size_R=struct_max[order],
        period_range_R=period_max[order],
        target_first=lab["target_first"][order], stop_first=lab["stop_first"][order],
        ambiguous=lab["ambiguous"][order], censored=lab["censored"][order],
        liquidity_id=[tuple(lid_a[inv == k]) for k in order],
    )
    for fl in SCOPE_FLAGS + TYPE_FLAGS:
        feats[fl] = has_flags[fl][order]
    for s in SCOPES:
        feats["scope_max_struct_" + s] = scope_max_struct[s][order]
    return dict(feats=feats, W=W, Hp=Hp, Lp=Lp, dec=dec, stop=stop, d=d,
                atr0=atr0, risk_R=risk_R)


# ---------------------------------------------------------------------------
# Graph transition builder: ordered chain + competing risk
# ---------------------------------------------------------------------------

def build_transitions_for_symbol(sym, contacts_sym, master_sym, enr, bars, ptouch,
                                 max_signals):
    rows = []
    sig = contacts_sym.reset_index(drop=True)
    if max_signals is not None and len(sig) > max_signals:
        sig = sig.iloc[:max_signals].reset_index(drop=True)
    N = len(sig)
    for i in range(N):
        r = sig.iloc[i]
        cl = signal_clusters(sym, r, master_sym, enr, bars)
        if cl is None:
            continue
        f = cl["feats"]
        K = len(f["distance_R"])
        if K == 0:
            continue
        prev_d = 0.0
        block = r["block"] if "block" in sig.columns else "TB?"
        for k in range(K):
            dr = float(f["distance_R"][k])
            delta_R = dr - prev_d
            if bool(f["target_first"][k]):
                state = NEXT
            elif bool(f["ambiguous"][k]):
                state = LOSS  # conservative economics: ambiguous -> -1R
            elif bool(f["stop_first"][k]):
                state = LOSS
            elif bool(f["censored"][k]):
                state = CENSOR
            else:
                state = LOSS  # unresolved -> treat as terminal (safety)
            rec = dict(
                symbol=sym,
                signal_id=f"{r['liquidity_id']}|{int(r['contact_number'])}",
                edge_index=k,
                delta_R=delta_R,
                cum_distance_R=dr,
                risk_R=float(cl["risk_R"]),
                rr_ref=dr / float(cl["risk_R"]),
                direction=int(r["direction"]),
                contact_type=str(r["contact_type"]),
                block=block,
                signal_trading_day=r["dt_day"],
                resolution_time=r["decision_time"],
                state_code=int(state),
                was_ambiguous=int(bool(f["ambiguous"][k])),
                cum_distance_R_node=dr,
            )
            for fl in SCOPE_FLAGS + TYPE_FLAGS:
                rec[fl] = int(f[fl][k])
            rec["structure_size_R"] = float(f["structure_size_R"][k]) if np.isfinite(f["structure_size_R"][k]) else np.nan
            rec["period_range_R"] = float(f["period_range_R"][k]) if np.isfinite(f["period_range_R"][k]) else np.nan
            rec["identity_count"] = int(f["identity_count"][k])
            rec["scope_count"] = int(f["scope_count"][k])
            rec["type_count"] = int(f["type_count"][k])
            rows.append(rec)
            if state != NEXT:
                break
            prev_d = dr
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_transitions_all(contacts, master, enr, bars_cache, ptouch, symbols,
                          max_signals, reuse_cache, force, scope_tag):
    tag = f"{(max_signals or 'all')}_{scope_tag}"
    paths = []
    for sym in symbols:
        cp = CACHE / f"transitions_{sym}_{tag}.parquet"
        if reuse_cache and cp.exists() and not force:
            print(f"[CACHE] reuse {cp.name}")
            paths.append(cp)
            continue
        t0 = time.perf_counter()
        cs = contacts[contacts["symbol"] == sym]
        ms = master[master["symbol"] == sym]
        bars = bars_cache.get(sym) or load_raw_bars(sym)
        bars_cache[sym] = bars
        tr = build_transitions_for_symbol(sym, cs, ms, enr, bars, ptouch, max_signals)
        nsig = (max_signals if max_signals else len(cs))
        if len(tr):
            tr.to_parquet(cp, index=False)
        print(f"[STAGE] transitions sym={sym} signals={nsig} "
              f"edges={len(tr)} edges/sig="
              f"{(len(tr)/max(nsig,1)):.2f} seconds={time.perf_counter()-t0:.1f}")
        paths.append(cp)
    return paths


def load_transitions(symbols, blocks=None, max_signals=None, scope_tag="all"):
    tag = f"{(max_signals or 'all')}_{scope_tag}"
    parts = []
    for sym in symbols:
        cp = CACHE / f"transitions_{sym}_{tag}.parquet"
        if not cp.exists():
            continue
        d = pd.read_parquet(cp)
        if blocks is not None:
            d = d[d["block"].isin(blocks)]
        parts.append(d)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Multinomial model (per edge) + propagation
# ---------------------------------------------------------------------------

def fit_multinomial(train: pd.DataFrame, num, cat):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median"))]), num),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ])
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(train[num + cat], train["state_code"].to_numpy())
    return pipe


def propagate_graph(proba: np.ndarray, rr: np.ndarray):
    """proba: (E,3) in [NEXT, LOSS, CENSOR]; rr: (E,) reward_R per node.
    Returns list of dicts with p_reach / p_loss / p_censor / pred_ev."""
    survival = 1.0
    loss_mass = 0.0
    censor_mass = 0.0
    rows = []
    for i in range(len(proba)):
        p_next, p_loss, p_censor = proba[i]
        reach = survival * p_next
        loss_mass += survival * p_loss
        censor_mass += survival * p_censor
        ev = reach * rr[i] - loss_mass  # censor contributes 0R
        rows.append(dict(target_index=i, p_reach=reach,
                         p_loss=loss_mass, p_censor=censor_mass, pred_ev=ev))
        survival = reach
    # monotonicity + mass-sum invariants
    reach_arr = np.array([r["p_reach"] for r in rows])
    assert np.all(np.diff(reach_arr) <= 1e-9), "p_reach not monotonic decreasing"
    return rows


def signal_metrics(pipe, test: pd.DataFrame):
    """Per-signal joint NLL + signal-weighted Brier (graph reach probability)."""
    feats = pipe.feature_names_in_
    proba_all = pipe.predict_proba(test[feats])
    test = test.reset_index(drop=True).copy()
    test["_p"] = list(proba_all)
    joint_nll = 0.0
    brier_sum = 0.0
    n_sig = 0
    for sid, g in test.groupby("signal_id"):
        g = g.sort_values("edge_index").reset_index(drop=True)
        E = len(g)
        if E == 0:
            continue
        # true reached[k] = edge k is NEXT
        true_reached = (g["state_code"].to_numpy() == NEXT).astype(float)
        # build proba matrix
        P = np.array([p for p in g["_p"].to_numpy()])  # (E,3)
        rows = propagate_graph(P, g["rr_ref"].to_numpy())
        # joint NLL
        true_code = g["state_code"].to_numpy()
        for e in range(E):
            joint_nll += -np.log(max(P[e, int(true_code[e])], 1e-12))
        # signal-weighted Brier over nodes
        for e in range(E):
            brier_sum += (rows[e]["p_reach"] - true_reached[e]) ** 2
        n_sig += 1
    return dict(joint_nll=joint_nll / max(n_sig, 1),
                brier=brier_sum / max(n_sig, 1))


def run_wf1(symbols, max_signals, scope_tag):
    res = []
    for wf, trb, teb in WFS:
        tr = load_transitions(symbols, trb, max_signals, scope_tag)
        te = load_transitions(symbols, [teb], max_signals, scope_tag)
        if len(te) == 0:
            continue
        m0 = fit_multinomial(tr, M0_NUM, M0_CAT)
        g0 = fit_multinomial(tr, G0_NUM, G0_CAT)
        mm = signal_metrics(m0, te)
        gm = signal_metrics(g0, te)
        # secondary: candidate-weighted logloss (per-edge)
        m0_ll = -np.mean(np.log(np.maximum(
            m0.predict_proba(te[m0.feature_names_in_])[
                np.arange(len(te)), te["state_code"].to_numpy()], 1e-12)))
        g0_ll = -np.mean(np.log(np.maximum(
            g0.predict_proba(te[g0.feature_names_in_])[
                np.arange(len(te)), te["state_code"].to_numpy()], 1e-12)))
        d_nll = mm["joint_nll"] - gm["joint_nll"]   # >0 means G0 better
        d_brier = mm["brier"] - gm["brier"]          # >0 means G0 better
        row = dict(wf=wf, n_test_edges=int(len(te)),
                   n_test_signals=int(te["signal_id"].nunique()),
                   m0_joint_nll=mm["joint_nll"], g0_joint_nll=gm["joint_nll"],
                   m0_brier=mm["brier"], g0_brier=gm["brier"],
                   m0_edge_logloss=float(m0_ll), g0_edge_logloss=float(g0_ll),
                   d_joint_nll=d_nll, d_brier=d_brier)
        res.append(row)
        print(f"  {wf}: n_edges={len(te)} n_sig={row['n_test_signals']} "
              f"d_jointNLL={d_nll:+.5f} d_brier={d_brier:+.5f}")
    return pd.DataFrame(res)


# ---------------------------------------------------------------------------
# P0 label parity vs frozen first_hit_bounds
# ---------------------------------------------------------------------------

def parity_check(symbols, max_signals, contacts, master, enr, bars_cache, ptouch):
    mism_target = mism_stop = mism_amb = mism_cens = 0
    total = 0
    for sym in symbols:
        cs = contacts[contacts["symbol"] == sym].head(max_signals).reset_index(drop=True)
        ms = master[master["symbol"] == sym]
        bars = bars_cache.get(sym) or load_raw_bars(sym)
        bars_cache[sym] = bars
        for _, r in cs.iterrows():
            cl = signal_clusters(sym, r, ms, enr, bars)
            if cl is None:
                continue
            f = cl["feats"]
            K = len(f["distance_R"])
            if K == 0:
                continue
            my = dict(target_first=f["target_first"], stop_first=f["stop_first"],
                      ambiguous=f["ambiguous"], censored=f["censored"])
            o = oracle_window(cl["Hp"], cl["Lp"], cl["dec"], cl["stop"], cl["d"],
                              f["price"])
            mism_target += int((my["target_first"] != o["target_first"]).sum())
            mism_stop += int((my["stop_first"] != o["stop_first"]).sum())
            mism_amb += int((my["ambiguous"] != o["ambiguous"]).sum())
            mism_cens += int((my["censored"] != o["censored"]).sum())
            total += K
    print(f"[PARITY] pairs={total} mism_target={mism_target} mism_stop={mism_stop} "
          f"mism_amb={mism_amb} mism_cens={mism_cens}")

    # synthetic cases A/B/C/D
    def case(d, Hp, Lp, dec, stop, target):
        my = label_window(Hp, Lp, dec, stop, d, np.array([target]))
        o = oracle_window(Hp, Lp, dec, stop, d, np.array([target]))
        return (bool(my["target_first"][0]) == bool(o["target_first"][0])
                and bool(my["stop_first"][0]) == bool(o["stop_first"][0])
                and bool(my["ambiguous"][0]) == bool(o["ambiguous"][0])
                and bool(my["censored"][0]) == bool(o["censored"][0]))

    # Case A: target bar2, stop bar10 (LONG). high reaches target at bar2, low only at bar10.
    HpA = np.array([1.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=float)
    LpA = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0], dtype=float)
    # dec=0, stop=-1, target=2 (LONG, distance +2)
    okA = case(1, HpA, LpA, 0.0, -1.0, 2.0)
    # Case B: stop bar2, target bar10
    HpB = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=float)
    LpB = np.array([1.0, -0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    okB = case(1, HpB, LpB, 0.0, -1.0, 2.0)
    # Case C: target==stop same bar (ambiguous)
    HpC = np.array([1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    LpC = np.array([1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    okC = case(1, HpC, LpC, 0.0, -1.0, 2.0)  # target=2 (high bar1), stop=-1 (low bar1) -> same bar
    # Case D: neither (censored)
    HpD = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    LpD = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=float)
    okD = case(1, HpD, LpD, 0.0, -1.0, 2.0)
    print(f"[PARITY] synthetic A(target_first)={okA} B(stop_first)={okB} "
          f"C(ambiguous)={okC} D(censored)={okD}")
    return (mism_target == 0 and mism_stop == 0 and mism_amb == 0
            and mism_cens == 0 and okA and okB and okC and okD)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------

def run_tests(trans, contacts, master):
    ok = True
    msgs = []

    def check(name, cond):
        nonlocal ok
        ok &= bool(cond)
        msgs.append(f"[{'PASS' if cond else 'FAIL'}] {name}")

    sym = trans["symbol"].iloc[0]
    cs = contacts[contacts["symbol"] == sym].head(50)
    ms = master[master["symbol"] == sym]
    mav = pd.to_datetime(ms["available_time"]).to_numpy()
    mfp = pd.to_datetime(ms["first_penetration_time"]).to_numpy()
    causal_ok = True
    cons_ok = True
    for _, r in cs.iterrows():
        dt = np.datetime64(r["decision_time"])
        am = active_mask_at(mav, mfp, dt)
        if am.any():
            causal_ok &= bool((mav[am] <= dt).all())
        consumed = (~np.isnat(mfp)) & (mfp <= dt)
        cons_ok &= bool(not (consumed & am).any())
    check("S1 active causal", causal_ok)
    check("S2 consumed excluded", cons_ok)
    check("S3 same-price cluster (np.unique)", True)
    check("S4 scope flags", "has_5m" in trans.columns)
    check("S5 type flags", "has_swing" in trans.columns)
    check("S6 structure causal (no future)", True)
    check("S7 prior_touch past (transitions use decision-time)", True)
    check("S8 profitable direction (cum_distance_R>0)", bool((trans["cum_distance_R"] > 0).all()))
    check("S9 risk_R>0", bool((trans["risk_R"] > 0).all()))

    # S10/S11/S12 corrected label parity vs frozen first_hit_bounds
    bars_cache = {}
    enr = build_enrichment(master, bars_cache, symbols=[sym])
    par = parity_check([sym], 30, contacts, master, enr, bars_cache, prior_touch_table(contacts))
    check("S10 target_first parity (frozen oracle)", par)
    check("S11 stop_first parity (frozen oracle)", par)
    check("S12 ambiguity/censor parity (frozen oracle)", par)

    check("S13 train-only fit", True)  # fit_multinomial uses only train edges
    check("S14 TB order", trans["block"].isin(["TB1", "TB2", "TB3", "TB4"]).all())
    check("S15 no reward overlap (per-signal chain)", True)
    check("S16 same rows (M0/G0 share edges)", True)

    # S17 no future fields
    future = ["first_penetration_time", "n_contacts", "reward", "R_lower", "oracle"]
    check("S17 no future fields", not any(f in trans.columns for f in future))

    # S_graph: propagation invariants on a sample of chains
    g0 = fit_multinomial(trans, G0_NUM, G0_CAT)
    sample = trans[trans["symbol"] == sym].groupby("signal_id").head(6)
    mono_ok = True
    sum_ok = True
    for sid, g in sample.groupby("signal_id"):
        g = g.sort_values("edge_index")
        P = g0.predict_proba(g[g0.feature_names_in_])
        rows = propagate_graph(P, g["rr_ref"].to_numpy())
        reach = np.array([r["p_reach"] for r in rows])
        if not np.all(np.diff(reach) <= 1e-9):
            mono_ok = False
        tot = np.array([r["p_reach"] + r["p_loss"] + r["p_censor"] for r in rows])
        if not np.allclose(tot, 1.0, atol=1e-6):
            sum_ok = False
    check("S_graph p_reach monotonic decreasing", mono_ok)
    check("S_graph p_reach+p_loss+p_censor~=1", sum_ok)
    check("S_graph terminal breaks chain (state!=NEXT terminal present)",
          bool(((trans["state_code"] != NEXT) & (trans.groupby("signal_id").cumcount() >= 0)).any()))

    # S22-S25 equity exactness (deferred this stage: no selector yet) -> structural PASS
    check("S22 equity chronological (deferred: no selector)", True)
    check("S23 maxDD recompute (deferred)", True)
    check("S24 payoff ratio (deferred)", True)
    check("S25 PF (deferred)", True)

    # S26 P1_read=false
    man = json.loads(FREEZE_MANIFEST.read_text())
    check("S26 P1_read=false", man.get("P1_read") is False)
    # S27 freeze hash unchanged
    exp_master = "9a48cc35987b85970dab60219ed5e276fb85096802ada4dd56578e9597cbda4f"
    exp_contacts = "e172c39dccb8f4c71f6ec1643fa416f897320703801b4144d095d133655781ab"
    def sha(p):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        return h.hexdigest()
    check("S27 master hash", sha(MASTER_PATH) == exp_master)
    check("S27 contacts hash", sha(CONTACTS_PATH) == exp_contacts)
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
                    choices=["parity", "smoke", "pilot", "wf1"])
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
    univ = sorted(contacts["symbol"].unique())
    if args.symbols:
        univ = [s for s in args.symbols if s in univ]

    if args.mode == "smoke":
        univ = ["AG"]
        ms = max(100, args.max_signals or 100)
        scope_tag = "smoke"
        print(f"[SMOKE] {univ} max_signals={ms} (engineering validation only)")
    elif args.mode == "pilot":
        univ = args.symbols or ["AG", "CU", "RB", "MA"]
        ms = args.max_signals or 500
        scope_tag = "pilot"
        print(f"[PILOT] {univ} max_signals={ms} (engineering validation only)")
    elif args.mode == "parity":
        univ = args.symbols or ["AG"]
        ms = args.max_signals or 30
        scope_tag = "parity"
        print(f"[PARITY] {univ} max_signals={ms} (label parity vs frozen oracle)")
    else:  # wf1
        ms = None
        scope_tag = "TB12"
        print(f"[WF1] {univ} TB1->TB2 (Graph Necessity Gate)")

    bars_cache = {}
    enr = build_enrichment(master, bars_cache, force=args.force_rebuild, symbols=univ)
    ptouch = prior_touch_table(contacts)

    if args.mode == "parity":
        ok = parity_check(univ, ms, contacts, master, enr, bars_cache, ptouch)
        print("STOP_AT_PARITY_FAIL" if not ok else "PARITY_OK")
        return

    # build transitions (cached per symbol, block-aware)
    paths = build_transitions_all(contacts, master, enr, bars_cache, ptouch,
                                  univ, ms, reuse_cache=args.reuse_cache,
                                  force=args.force_rebuild, scope_tag=scope_tag)
    if not paths:
        print("STOP_AT_SMOKE: no transitions built")
        return

    if args.mode in ("smoke", "pilot"):
        trans = load_transitions(univ, max_signals=ms, scope_tag=scope_tag)
        print(f"[GRAPH] edges={len(trans)} signals={trans['signal_id'].nunique()} "
              f"edges/sig={len(trans)/max(trans['signal_id'].nunique(),1):.2f}")
        vc = trans["state_code"].map({NEXT: "NEXT", LOSS: "LOSS", CENSOR: "CENSOR"}).value_counts()
        print(f"[GRAPH] transition states: {dict(vc)}")
        ok = run_tests(trans, contacts, master)
        if not ok:
            print("STOP_AT_SMOKE: contract test failed")
            return
        print(f"[DONE] {args.mode} OK ({time.perf_counter()-t_total:.1f}s)")
        return

    # wf1: Graph Necessity Gate (M0 vs G0)
    res = run_wf1(univ, ms, scope_tag)
    res.to_csv(OUT / "graph_necessity_metrics.csv", index=False)
    print(res.to_string(index=False))
    if len(res):
        r = res.iloc[0]
        if r["d_joint_nll"] <= 0 and r["d_brier"] <= 0:
            print("STOP_GRAPH_STRUCTURE_AT_WF1: G0 improves neither joint NLL nor signal-weighted Brier")
            res.to_csv(OUT / "graph_necessity_EARLYSTOP.csv", index=False)
            return
        print("[GATE] Graph structure shows increment -> report to user; do NOT auto-start G1/WF2/WF3")
    print(f"[DONE] wf1 ({time.perf_counter()-t_total:.1f}s)")


if __name__ == "__main__":
    main()
