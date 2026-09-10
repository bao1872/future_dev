"""Step 7-14：outcome、paired response curve、分层、dedup、clean-control、
day-block bootstrap、native-signed falsification。

NO MODEL / NO ML / NO tuning。Matching 已冻结（v1.1）。
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from research.ob_event_value.ev_contract_v1 import (
    ANCHOR_H, HORIZONS, K_CONTROLS, MATCH_COLS, RESULTS, SYMBOLS,
    get_bars,
)
from research.ob_event_value.match_controls import (
    PRE_RANGE_CALIPER, _RANGE_IDX, attach_state, robust_scale_params,
)

BOOT = 500
SEED = 7

PRIMARY = "max_directional_excursion_R"
AUX = ("forward_range_R", "realized_vol_R", "abs_close_move_R")
BINARY = ("hit_abs_1p5R", "hit_abs_2p5R")


def outcome_arrays(bars: dict) -> dict:
    """向量化：返回 {horizon: {key: array}}，索引与 bar 对齐。"""
    c = pd.Series(bars["close"], dtype=float)
    h = pd.Series(bars["high"], dtype=float)
    lo = pd.Series(bars["low"], dtype=float)
    r0 = pd.Series(np.r_[np.nan, bars["atr5"][:-1]], dtype=float)
    out = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        for hz in HORIZONS:
            mx = h.shift(-(hz - 1)).rolling(hz).max().shift(-1) \
                if False else h.shift(-(hz - 1)).rolling(hz).max()
            mn = lo.shift(-(hz - 1)).rolling(hz).min()
            ce = c.shift(-hz)
            d2 = (c.diff() ** 2)
            rv = d2.shift(-(hz - 1)).rolling(hz).sum()
            # 上述在 index i 表示 [i .. i+hz-1]，我们需要 [t0+1 .. t0+hz]
            mx1 = mx.shift(-1)
            mn1 = mn.shift(-1)
            rv1 = rv.shift(-1)
            p0 = c
            up = (mx1 - p0) / r0
            dn = (p0 - mn1) / r0
            mde = np.fmax(up, dn)
            out[hz] = dict(
                up_exc_R=up.to_numpy(float),
                down_exc_R=dn.to_numpy(float),
                max_directional_excursion_R=mde.to_numpy(float),
                forward_range_R=((mx1 - mn1) / r0).to_numpy(float),
                abs_close_move_R=((ce - p0).abs() / r0).to_numpy(float),
                realized_vol_R=(np.sqrt(rv1) / r0).to_numpy(float),
                hit_abs_1p5R=(mde >= 1.5).astype(float).to_numpy(),
                hit_abs_2p5R=(mde >= 2.5).astype(float).to_numpy(),
            )
    return out


def native_signed(bars: dict, hz: int) -> np.ndarray:
    c = pd.Series(bars["close"], dtype=float)
    r0 = pd.Series(np.r_[np.nan, bars["atr5"][:-1]], dtype=float)
    return ((c.shift(-hz) - c) / r0).to_numpy(float)


def collect(pairs, ev, pool, oa_by_sym, key, hz):
    """返回 (treat_vals, control_mean_vals) 按 pair 行对齐。"""
    tv = np.full(len(pairs), np.nan)
    cv = np.full(len(pairs), np.nan)
    ev_idx = ev.set_index("event_bar_id")
    pool_idx = pool.set_index("control_id")
    eid = pairs["event_bar_id"].to_numpy()
    cid = pairs["control_id"].to_numpy()
    esym = ev_idx.loc[eid, "symbol"].to_numpy()
    et0 = ev_idx.loc[eid, "t0"].to_numpy()
    csym = pool_idx.loc[cid, "symbol"].to_numpy()
    ct0 = pool_idx.loc[cid, "t0"].to_numpy()
    assert (esym == csym).all(), "pair 跨品种"
    for sym in SYMBOLS:
        mt = esym == sym
        if not mt.any():
            continue
        arr = oa_by_sym[sym][hz][key]
        tv[mt] = arr[et0[mt]]
        cv[mt] = arr[ct0[mt]]
    return tv, cv


def paired_frame(pairs, ev, pool, oa_by_sym, hz):
    """每个 event_bar_id 一行：treat 与 3 个 control 的均值。"""
    eids = pairs["event_bar_id"].to_numpy()
    df = pd.DataFrame({"event_bar_id": eids})
    for key in (PRIMARY,) + AUX + BINARY + ("up_exc_R", "down_exc_R"):
        tv, cv = collect(pairs, ev, pool, oa_by_sym, key, hz)
        df[f"t_{key}"] = tv
        df[f"c_{key}"] = cv
    g = df.groupby("event_bar_id", sort=False)
    out = g[[f"t_{k}" for k in (PRIMARY,) + AUX + BINARY]].mean()
    out.columns = [c[2:] for c in out.columns]
    out = out.add_prefix("t_")
    cm = g[[f"c_{k}" for k in (PRIMARY,) + AUX + BINARY]].mean()
    cm.columns = [c[2:] for c in cm.columns]
    cm = cm.add_prefix("c_")
    res = pd.concat([out, cm], axis=1).reset_index()
    for k in (PRIMARY,) + AUX + BINARY:
        res[f"d_{k}"] = res[f"t_{k}"] - res[f"c_{k}"]
    return res


def block_boot(frame, day_of, n=BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    days = np.asarray(sorted(set(day_of)))
    idx = {d: np.flatnonzero(day_of == d) for d in days}
    keys = (PRIMARY,) + AUX + BINARY
    vals = {k: frame[f"d_{k}"].to_numpy(float) for k in keys}
    tvals = {k: frame[f"t_{k}"].to_numpy(float) for k in keys}
    cvals = {k: frame[f"c_{k}"].to_numpy(float) for k in keys}
    acc = {k: [] for k in keys}
    rel = {k: [] for k in keys}
    for _ in range(n):
        pick = rng.choice(days, size=len(days), replace=True)
        rows = np.concatenate([idx[d] for d in pick])
        for k in keys:
            acc[k].append(np.nanmean(vals[k][rows]))
            cm = np.nanmean(cvals[k][rows])
            rel[k].append((np.nanmean(tvals[k][rows]) / cm - 1.0)
                          if cm > 0 else np.nan)
    o = {}
    for k in keys:
        o[f"{k}_delta"] = round(float(np.nanmean(vals[k])), 4)
        o[f"{k}_ci"] = [round(float(np.nanquantile(acc[k], .025)), 4),
                        round(float(np.nanquantile(acc[k], .975)), 4)]
        o[f"{k}_rel"] = round(float(
            np.nanmean(tvals[k]) / np.nanmean(cvals[k]) - 1.0), 4)
        o[f"{k}_rel_ci"] = [round(float(np.nanquantile(rel[k], .025)), 4),
                            round(float(np.nanquantile(rel[k], .975)), 4)]
    return o


def main():
    ev = pd.read_parquet(RESULTS / "event_bars.parquet")
    pool = pd.read_parquet(RESULTS / "control_pool.parquet")
    pairs = pd.read_parquet(RESULTS / "matching_pairs.parquet")
    ev = attach_state(ev)

    oa = {s: outcome_arrays(get_bars(s)) for s in SYMBOLS}
    print("[outcomes] arrays built", flush=True)

    # day of treatment（用于 block bootstrap）
    day_map = {}
    for s in SYMBOLS:
        day = get_bars(s)["trading_day"]
        for i, d in enumerate(day):
            day_map[(s, i)] = d
    ev["tday"] = [day_map[(s, int(t))]
                  for s, t in zip(ev["symbol"], ev["t0"])]
    tday = ev.set_index("event_bar_id")["tday"]

    frames = {}
    for hz in HORIZONS:
        pf = paired_frame(pairs, ev, pool, oa, hz)
        pf["tday"] = tday.reindex(pf["event_bar_id"]).to_numpy()
        frames[hz] = pf
        print(f"  H={hz} pairs={len(pf)}", flush=True)

    # ---------- response curve ----------
    rows = []
    for hz in HORIZONS:
        f = frames[hz]
        for k in (PRIMARY,) + AUX:
            tm = float(f[f"t_{k}"].mean())
            cm = float(f[f"c_{k}"].mean())
            rows.append(dict(horizon=hz, outcome=k, n=len(f),
                             ob_mean=round(tm, 4), control_mean=round(cm, 4),
                             delta=round(tm - cm, 4),
                             relative_uplift=round(tm / cm - 1, 4)
                             if cm > 0 else None))
    rc = pd.DataFrame(rows)
    rc.to_csv(RESULTS / "event_response_curve.csv", index=False,
              encoding="utf-8-sig")
    print("\n=== RESPONSE CURVE ===")
    print(rc.pivot_table(index="horizon", columns="outcome",
                         values=["ob_mean", "control_mean", "delta",
                                 "relative_uplift"]).round(4).to_string())

    # ---------- H12 anchor + bootstrap ----------
    anch = frames[ANCHOR_H]
    b = block_boot(anch, anch["tday"].to_numpy())
    h12 = pd.DataFrame([{"horizon": ANCHOR_H, **b}])
    h12.to_csv(RESULTS / "event_effect_h12.csv", index=False,
               encoding="utf-8-sig")
    print(f"\n=== H={ANCHOR_H} ANCHOR (bootstrap {BOOT}) ===")
    for k in (PRIMARY,) + AUX + BINARY:
        print(f"  {k:30s} OB={float(anch[f't_{k}'].mean()):.4f} "
              f"CTRL={float(anch[f'c_{k}'].mean()):.4f} "
              f"delta={b[f'{k}_delta']:+.4f} ci={b[f'{k}_ci']} "
              f"rel={b[f'{k}_rel']:+.4f} rel_ci={b[f'{k}_rel_ci']}")

    # ---------- stratifications ----------
    meta = ev.set_index("event_bar_id")[["symbol", "source_tf",
                                         "bias_composition",
                                         "native_direction", "t0"]]

    def strat(col, name, frame=None):
        f = frames[ANCHOR_H] if frame is None else frame
        f = f.copy()
        f[col] = meta.reindex(f["event_bar_id"])[col].to_numpy()
        rs = []
        for k, g in f.groupby(col):
            tm = float(g[f"t_{PRIMARY}"].mean())
            cm = float(g[f"c_{PRIMARY}"].mean())
            rs.append(dict(**{name: k}, n=len(g),
                           ob=round(tm, 4), control=round(cm, 4),
                           delta=round(tm - cm, 4),
                           rel_uplift=round(tm / cm - 1, 4) if cm > 0 else None,
                           d_range=round(float(g["d_forward_range_R"].mean()), 4),
                           d_rvol=round(float(g["d_realized_vol_R"].mean()), 4),
                           rd_2p5=round(float(
                               g[f"d_hit_abs_2p5R"].mean()), 4)))
        t = pd.DataFrame(rs).sort_values("delta", ascending=False)
        t.to_csv(RESULTS / f"effect_by_{col}.csv", index=False,
                 encoding="utf-8-sig")
        return t

    for col, name in (("source_tf", "source_tf"), ("symbol", "symbol"),
                      ("bias_composition", "bias_composition")):
        t = strat(col, name)
        print(f"\n=== by {name} (H={ANCHOR_H}) ===")
        print(t.to_string(index=False))
        if col == "symbol":
            print(f"  positive symbols: "
                  f"{int((t['delta'] > 0).sum())}/{len(t)}  "
                  f"macro median delta: {t['delta'].median():.4f}")

    # ---------- folds ----------
    import research.m2_nondeep_temporal_v1 as m2
    folds, _ = m2.build_folds(anch["tday"].to_numpy(), len(anch))
    fr = []
    for fi in range(4):
        te_d = folds[fi][2]
        g = anch[anch["tday"].isin(te_d)]
        tm = float(g[f"t_{PRIMARY}"].mean())
        cm = float(g[f"c_{PRIMARY}"].mean())
        fr.append(dict(fold=f"F{fi+1}", n=len(g), ob=round(tm, 4),
                       control=round(cm, 4), delta=round(tm - cm, 4),
                       rel_uplift=round(tm / cm - 1, 4) if cm else None,
                       d_range=round(float(g["d_forward_range_R"].mean()), 4)))
    pd.DataFrame(fr).to_csv(RESULTS / "effect_by_fold.csv", index=False,
                            encoding="utf-8-sig")
    print("\n=== by fold ===")
    print(pd.DataFrame(fr).to_string(index=False))

    # ---------- ANY_TF dedup ----------
    anytf = pd.read_parquet(RESULTS / "event_bars_anytf_dedup.parquet")
    keep = set(anytf["event_bar_id"])
    sub = anch[anch["event_bar_id"].isin(keep)]
    tm, cm = float(sub[f"t_{PRIMARY}"].mean()), float(sub[f"c_{PRIMARY}"].mean())
    anyrow = dict(dataset="ANY_TF_DEDUP", n=len(sub), ob=round(tm, 4),
                  control=round(cm, 4), delta=round(tm - cm, 4),
                  rel_uplift=round(tm / cm - 1, 4),
                  d_range=round(float(sub["d_forward_range_R"].mean()), 4),
                  d_rvol=round(float(sub["d_realized_vol_R"].mean()), 4),
                  rd_2p5=round(float(sub["d_hit_abs_2p5R"].mean()), 4))
    pd.DataFrame([anyrow]).to_csv(
        RESULTS / "any_tf_dedup_sensitivity.csv", index=False,
        encoding="utf-8-sig")
    print("\n=== ANY_TF dedup ===")
    print(pd.DataFrame([anyrow]).to_string(index=False))

    # ---------- native-signed falsification ----------
    ns_rows = []
    for sym in SYMBOLS:
        arr = native_signed(get_bars(sym), ANCHOR_H)
        m = (meta.reindex(anch["event_bar_id"])["symbol"].to_numpy() == sym)
        sub = anch[m]
        if sub.empty:
            continue
        t0s = meta.reindex(sub["event_bar_id"])["t0"].to_numpy()
        nds = meta.reindex(sub["event_bar_id"])["native_direction"].to_numpy()
    # 对 control 也要用相同 native_direction
    pool_idx = pool.set_index("control_id")
    pr = pairs.merge(ev[["event_bar_id", "native_direction"]],
                     on="event_bar_id", how="left", validate="many_to_one")
    pr = pr[pr["event_bar_id"].isin(set(anch["event_bar_id"]))]
    tv = np.full(len(pr), np.nan)
    cv = np.full(len(pr), np.nan)
    nd = pr["native_direction"].to_numpy()
    mix = nd[0] if len(nd) else 0
    for sym in SYMBOLS:
        arr = native_signed(get_bars(sym), ANCHOR_H)
        et = ev.set_index("event_bar_id")
        mt = (et.reindex(pr["event_bar_id"])["symbol"].to_numpy() == sym)
        if not mt.any():
            continue
        t0s = et.reindex(pr["event_bar_id"])["t0"].to_numpy()[mt]
        tv[mt] = arr[t0s] * nd[mt]
        ct0 = pool_idx.reindex(pr["control_id"])["t0"].to_numpy()[mt]
        cv[mt] = arr[ct0] * nd[mt]
    okm = nd != 0          # 排除 MIXED
    ns = pd.DataFrame(dict(event_bar_id=pr["event_bar_id"].to_numpy()[okm],
                           t_ns=tv[okm], c_ns=cv[okm]))
    g = ns.groupby("event_bar_id", sort=False).mean().reset_index()
    ns_out = dict(
        n=len(g),
        ob_native_signed=round(float(g["t_ns"].mean()), 4),
        control_native_signed=round(float(g["c_ns"].mean()), 4),
        delta=round(float((g["t_ns"] - g["c_ns"]).mean()), 4),
    )
    pd.DataFrame([ns_out]).to_csv(
        RESULTS / "native_signed_falsification.csv", index=False,
        encoding="utf-8-sig")
    print("\n=== native-signed falsification (H=12) ===")
    print(pd.DataFrame([ns_out]).to_string(index=False))

    bootstrap_out = {"H12": b}
    pd.DataFrame([dict(dataset="ALL", horizon=ANCHOR_H, **b)]).to_csv(
        RESULTS / "bootstrap.csv", index=False, encoding="utf-8-sig")

    (RESULTS / "audit.json").write_text(json.dumps(dict(
        matching_design="v1.1", pre_range_caliper=PRE_RANGE_CALIPER,
        k=K_CONTROLS, horizons=list(HORIZONS), anchor=ANCHOR_H,
        n_pairs=int(len(anch)), boot=BOOT,
        h12=b, native_signed=ns_out,
    ), indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print("\nOB_EVENT_VALUE_DONE")


if __name__ == "__main__":
    main()
