"""P0 + P1：interaction eligibility 修正 + distance-permuted placebo levels。

同一个 resolver 同时服务 TRUE liquidity 与 PLACEBO，
保证两边事件语义完全一致。

P0.1 activation 时 level 必须仍在价格前方（VALID_AHEAD）
P0.3 GAP_CROSS 与 CONTINUOUS_CROSS 分离（primary = CONTINUOUS_CROSS）
P0.5 post-reclaim 扫描至第一个**相关** competing structural event
P1   距离置换 placebo（3 个预注册 seed，derangement）
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from research.build_ob_candidate_universe_v3 import aggregate_4h_from_1h
from research.build_pytdx_panel import aggregate_15m
from research.dsa_adapter import compute_dsa_canonical
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.ob_trigger_snapshot import (aggregate_1h_from_15m,
                                          build_full_ob_smc_tf)
from research.phase1_tradability.phase1_contract_v1 import discontinuity_flags
from research.rl_62d_simulator_v1 import minute_of_day

RESULTS = Path("research/analysis_results/liquidity_specificity_v2")
RESULTS.mkdir(parents=True, exist_ok=True)
FIVE = pd.Timedelta(minutes=5)
PLACEBO_SEEDS = (20260910, 20260911, 20260912)
MATCH_COLS_NUM = ["level_age_log1p", "atr_rel_pre", "pre_ret_3_R",
                  "pre_ret_12_R", "pre_rv_12_R", "pre_range_12_R",
                  "volume_z_20_pre", "penetration_depth_R",
                  "close_beyond_R", "bar_range_R", "abs_return_R",
                  "gap_R", "volume_z_t0"]
STAGE1_NUM = ["level_age_log1p", "atr_rel_pre", "pre_ret_3_R",
              "pre_ret_12_R", "pre_rv_12_R", "pre_range_12_R",
              "volume_z_20_pre", "gap_R"]


def session_coords(t):
    ts = pd.to_datetime(pd.Series(t))
    brk = ts.diff() > pd.Timedelta(minutes=5)
    seg = brk.cumsum().to_numpy()
    mod = minute_of_day(t)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    seg_start_idx = np.empty(len(t), dtype=np.int64)
    seg_start_mod = np.empty(len(t), dtype=np.int64)
    cur = 0
    for i in range(len(t)):
        while cur + 1 < len(starts) and starts[cur + 1] <= i:
            cur += 1
        seg_start_idx[i] = starts[cur]
        seg_start_mod[i] = mod[starts[cur]]
    dt = ts.to_numpy()
    mfs = (dt - dt[seg_start_idx]) / np.timedelta64(1, "m")
    return seg_start_mod, (mfs.astype(np.int64) // 30)


def pre_state(bars):
    c = pd.Series(bars["close"], dtype=float)
    h = pd.Series(bars["high"], dtype=float)
    lo = pd.Series(bars["low"], dtype=float)
    v = pd.Series(bars["volume"], dtype=float)
    atr = pd.Series(bars["atr5"], dtype=float)
    r0 = atr.shift(1)
    dc = c.diff()
    vm = v.rolling(20).mean()
    vs = v.rolling(20).std(ddof=0).clip(lower=1e-12)
    return dict(
        R_pre=r0.to_numpy(float),
        atr_rel_pre=(r0 / c.shift(1)).to_numpy(float),
        pre_ret_3_R=((c.shift(1) - c.shift(4)) / r0).to_numpy(float),
        pre_ret_12_R=((c.shift(1) - c.shift(13)) / r0).to_numpy(float),
        pre_rv_12_R=(np.sqrt((dc ** 2).rolling(12).sum()) / r0).to_numpy(float),
        pre_range_12_R=((h.rolling(12).max() - lo.rolling(12).min())
                        / r0).to_numpy(float),
        volume_z_20_pre=((v.shift(1) - vm) / vs).to_numpy(float),
        vroll_mean=vm.to_numpy(float), vroll_std=vs.to_numpy(float),
    )


class SymbolData:
    def __init__(self, sym):
        five = load_raw_5m(sym).sort_values("bar_start_time").reset_index(
            drop=True)
        five["volume"] = five["trade"].astype(float)
        five["trading_day"] = five["trading_day"].astype(str)
        fifteen = aggregate_15m(five)
        oneh = aggregate_1h_from_15m(fifteen)
        env4 = aggregate_4h_from_1h(oneh)
        self.sym = sym
        self.five = five
        self.t = pd.to_datetime(five["bar_start_time"]).to_numpy()
        self.hi = five["high"].to_numpy(float)
        self.lo = five["low"].to_numpy(float)
        self.cl = five["close"].to_numpy(float)
        self.op = five["open"].to_numpy(float)
        self.vol = five["volume"].to_numpy(float)
        self.day = five["trading_day"].to_numpy()
        self.n = len(five)
        self.disc = discontinuity_flags(sym)
        self.atr5 = self._atr()
        self.st = pre_state(dict(close=self.cl, high=self.hi, low=self.lo,
                                 volume=self.vol, atr5=self.atr5))
        self.session_type, self.time_bucket = session_coords(self.t)
        smc5 = build_full_ob_smc_tf(five.copy())
        self.ev_at, self.ev_kind, self.ev_bias = self._events(smc5)
        self.trend1h = self._trend(oneh, "1h")
        dsa = compute_dsa_canonical(env4)
        self.env = pd.DataFrame(dict(
            available_time=pd.to_datetime(env4["bar_end_time"]).to_numpy(),
            env_direction_4h=pd.to_numeric(dsa["dsa_direction"],
                                           errors="coerce").to_numpy()))

    def _atr(self):
        from research.phase1_tradability.phase1_contract_v1 import \
            compute_atr5
        return compute_atr5(dict(open=self.op, high=self.hi, low=self.lo,
                                 close=self.cl, time=self.t, n=self.n))

    def _events(self, smc5):
        at, kd, bi = [], [], []
        for e in smc5["events"]:
            ci = int(e["confirmed_index"])
            if ci >= self.n:
                continue
            at.append(self.t[ci] + FIVE)
            kd.append(e["type"])
            bi.append(int(e["bias"]))
        o = np.argsort(np.asarray(at))
        return (np.asarray(at)[o], np.asarray(kd)[o], np.asarray(bi)[o])

    def _trend(self, bars, tf):
        smc = build_full_ob_smc_tf(bars.copy())
        st = pd.DataFrame(smc["state_timeline"])
        bt = pd.to_datetime(bars["bar_start_time"])
        per = pd.Timedelta(hours=1) if tf == "1h" else pd.Timedelta(
            minutes=15)
        return pd.DataFrame(dict(
            available_time=(bt.iloc[st["bar_index"].to_numpy()]
                            + per).to_numpy(),
            trend_struct_1h=st["swing_bias"].to_numpy(int)))


def activation_state(sd: SymbolData, level_price: float, side: int,
                     activation_i: int):
    ref_i = activation_i - 1
    if ref_i < 0:
        return None, None
    c_ref = sd.cl[ref_i]
    r0 = sd.atr5[ref_i]
    if not np.isfinite(r0) or r0 <= 0:
        return None, None
    if side == +1:
        st = ("VALID_AHEAD" if c_ref < level_price else
              "STALE_ALREADY_BEYOND" if c_ref > level_price else "AT_LEVEL")
    else:
        st = ("VALID_AHEAD" if c_ref > level_price else
              "STALE_ALREADY_BEYOND" if c_ref < level_price else "AT_LEVEL")
    d0 = side * (level_price - c_ref) / r0
    return st, dict(ref_i=ref_i, c_ref=c_ref, r0=r0, d0_R=d0)


def resolve_levels(sd: SymbolData, levels: pd.DataFrame,
                   is_placebo: bool) -> pd.DataFrame:
    rows = []
    t5n = sd.t
    for r in levels.itertuples(index=False):
        side = int(r.side)
        lvl = float(r.price)
        act_i = int(np.searchsorted(
            t5n, np.datetime64(r.available_time), side="left"))
        if act_i >= sd.n:
            continue
        stt, ctx = activation_state(sd, lvl, side, act_i)
        if stt is None:
            continue
        rec = dict(
            level_key=getattr(r, "level_key"),
            symbol=sd.sym, liquidity_type=r.liquidity_type,
            liquidity_scope=r.liquidity_scope, side=side,
            price=lvl, available_time=r.available_time,
            activation_i=act_i, activation_state=stt,
            is_placebo=bool(is_placebo),
        )
        if is_placebo:
            rec["replica_id"] = int(r.replica_id)
            rec["template_liquidity_id"] = r.template_liquidity_id
            rec["donor_liquidity_id"] = r.donor_liquidity_id
            rec["true_price"] = float(r.true_price)
        rec["d0_R"] = ctx["d0_R"]
        rec["c0"] = ctx["c_ref"]
        rec["r0_act"] = ctx["r0"]
        if stt != "VALID_AHEAD" or ctx["d0_R"] <= 0:
            rows.append(rec)
            continue
        # fresh interaction
        ti, pre_state_ = None, None
        for i in range(act_i, sd.n):
            if sd.disc[i]:
                pre_state_ = "ROLL_CENSORED_PRE_INTERACTION"
                break
            reached = (sd.hi[i] >= lvl) if side == +1 else (sd.lo[i] <= lvl)
            if reached:
                ti = i
                break
        if pre_state_ is not None:
            rec["pre_touch_state"] = pre_state_
            rows.append(rec)
            continue
        if ti is None:
            rows.append(rec)
            continue
        rec["interaction_i"] = ti
        rec["interaction_time"] = t5n[ti]
        rec["trading_day"] = sd.day[ti]
        prev_close = sd.cl[ti - 1] if ti > 0 else np.nan
        if side == +1:
            gap = (np.isfinite(prev_close) and prev_close < lvl
                   and sd.op[ti] > lvl)
        else:
            gap = (np.isfinite(prev_close) and prev_close > lvl
                   and sd.op[ti] < lvl)
        rec["interaction_path"] = "GAP_CROSS" if gap else "CONTINUOUS_CROSS"
        pen = (sd.hi[ti] > lvl) if side == +1 else (sd.lo[ti] < lvl)
        rec["penetrated"] = bool(pen)
        if not pen:
            rows.append(rec)
            continue
        # Stage 1
        if side == +1:
            sb, cb = sd.cl[ti] < lvl, sd.cl[ti] > lvl
        else:
            sb, cb = sd.cl[ti] > lvl, sd.cl[ti] < lvl
        rec["stage1"] = ("SAME_BAR_RECLAIM" if sb else
                         "CLOSE_BEYOND" if cb else "CLOSE_AT_LEVEL")
        rec["penetration_bar_start"] = t5n[ti]
        rec["penetration_available_time"] = t5n[ti] + FIVE
        # covariates (R = ATR5[t0-1])
        R = sd.atr5[ti - 1] if ti > 0 else np.nan
        rec["R_at_t0"] = R
        if np.isfinite(R) and R > 0:
            ext = sd.hi[ti] if side == +1 else sd.lo[ti]
            st = sd.st
            rec["level_age_log1p"] = float(np.log1p(max(ti - act_i, 0)))
            rec["atr_rel_pre"] = float(st["atr_rel_pre"][ti])
            rec["pre_ret_3_R"] = float(st["pre_ret_3_R"][ti])
            rec["pre_ret_12_R"] = float(st["pre_ret_12_R"][ti])
            rec["pre_rv_12_R"] = float(st["pre_rv_12_R"][ti])
            rec["pre_range_12_R"] = float(st["pre_range_12_R"][ti])
            rec["volume_z_20_pre"] = float(st["volume_z_20_pre"][ti])
            rec["penetration_depth_R"] = side * (ext - lvl) / R
            rec["close_beyond_R"] = side * (sd.cl[ti] - lvl) / R
            rec["bar_range_R"] = (sd.hi[ti] - sd.lo[ti]) / R
            rec["abs_return_R"] = abs(sd.cl[ti] - sd.cl[ti - 1]) / R
            rec["gap_R"] = side * (sd.op[ti] - sd.cl[ti - 1]) / R
            rec["volume_z_t0"] = ((sd.vol[ti] - st["vroll_mean"][ti])
                                  / st["vroll_std"][ti])
            rec["session_type"] = int(sd.session_type[ti])
            rec["time_bucket_30m"] = int(sd.time_bucket[ti])
        # post-reclaim (strict, scan to first RELEVANT event)
        if sb:
            rt = t5n[ti] + FIVE
            rev, pen2 = -side, side
            post, ptime = "NO_RELEVANT_EVENT_BEFORE_CENSOR", None
            k0 = int(np.searchsorted(sd.ev_at, np.datetime64(rt),
                                     side="right"))
            for k in range(k0, len(sd.ev_at)):
                if sd.disc[min(int(np.searchsorted(
                        t5n, np.datetime64(sd.ev_at[k]))) - 1, sd.n - 1)]:
                    post = "ROLL_CENSORED"
                    break
                is_rev = (sd.ev_kind[k] == "CHoCH") and (sd.ev_bias[k] == rev)
                is_res = (sd.ev_kind[k] == "BOS") and (sd.ev_bias[k] == pen2)
                if is_rev and is_res:
                    post, ptime = "AMBIGUOUS_SAME_TIMESTAMP", sd.ev_at[k]
                    break
                if is_rev:
                    post, ptime = "REVERSAL_MSS_CONFIRMED", sd.ev_at[k]
                    break
                if is_res:
                    post, ptime = "REJECTION_FAILED_REACCEPTED", sd.ev_at[k]
                    break
                # irrelevant -> continue
            rec["post_reclaim_state"] = post
            rec["post_reclaim_time"] = ptime
        rows.append(rec)
    return pd.DataFrame(rows)


def build_placebo(true_lv: pd.DataFrame, sd_map) -> pd.DataFrame:
    """Distance-permuted placebo：只使用 activation-time 信息。"""
    out = []
    for seed in PLACEBO_SEEDS:
        for (sym, ty, side), g in true_lv.groupby(
                ["symbol", "liquidity_type", "side"]):
            if len(g) < 3:
                continue
            sd = sd_map[sym]
            rng = np.random.default_rng(seed + hash((sym, ty, side)) % 1000)
            donors = g["d0_R"].to_numpy(float)
            # derangement
            perm = rng.permutation(len(g))
            for i in range(len(perm)):
                if perm[i] == i:
                    j = (i + 1) % len(perm)
                    perm[i], perm[j] = perm[j], perm[i]
            g = g.reset_index(drop=True)
            for i in range(len(g)):
                d = float(donors[perm[i]])
                c0 = float(g.loc[i, "c0"])
                r0 = float(g.loc[i, "r0_act"])
                pseudo = c0 + int(side) * d * r0
                if abs(pseudo - float(g.loc[i, "price"])) < 1e-12:
                    continue
                out.append(dict(
                    level_key=f"PLB|{seed}|{g.loc[i,'level_key']}",
                    symbol=sym, liquidity_type=ty,
                    liquidity_scope=g.loc[i, "liquidity_scope"],
                    side=int(side), price=pseudo,
                    available_time=g.loc[i, "available_time"],
                    replica_id=int(seed),
                    template_liquidity_id=g.loc[i, "level_key"],
                    donor_liquidity_id=g.loc[
                        int(np.flatnonzero(
                            (true_lv['level_key']
                             == g.loc[i, 'level_key']).to_numpy())[0])
                        if False else perm[i], "level_key"],
                    true_price=float(g.loc[i, "price"]),
                    template_d0_R=float(g.loc[i, "d0_R"]),
                    donor_d0_R=d,
                ))
    return pd.DataFrame(out)


def main():
    lv = pd.read_parquet(
        "research/analysis_results/liquidity_state_machine_v1/"
        "liquidity_levels.parquet")
    lv = lv.rename(columns={"liquidity_id": "level_key"})
    lv["available_time"] = pd.to_datetime(lv["available_time"])

    sd_map, trues = {}, []
    t0 = time.perf_counter()
    for sym in sorted(lv["symbol"].unique()):
        sd = SymbolData(sym)
        sd_map[sym] = sd
        sub = lv[lv["symbol"] == sym]
        r = resolve_levels(sd, sub, False)
        trues.append(r)
        print(f"  [true] {sym}: {len(r)} ({time.perf_counter()-t0:.0f}s)",
              flush=True)
    true_res = pd.concat(trues, ignore_index=True)
    true_res.to_parquet(RESULTS / "true_interactions.parquet", index=False)

    # enrichment for placebo generation
    keep = true_res[true_res["activation_state"] == "VALID_AHEAD"].copy()
    plb = build_placebo(keep, sd_map)
    plb.to_parquet(RESULTS / "placebo_levels.parquet", index=False)
    print(f"\nplacebo levels = {len(plb)}")

    plbs = []
    for sym in sorted(plb["symbol"].unique()):
        sd = sd_map[sym]
        sub = plb[plb["symbol"] == sym]
        out = []
        for rep, g in sub.groupby("replica_id"):
            rr = resolve_levels(sd, g, True)
            out.append(rr)
        plbs.append(pd.concat(out, ignore_index=True))
        print(f"  [placebo] {sym}: {time.perf_counter()-t0:.0f}s",
              flush=True)
    plb_res = pd.concat(plbs, ignore_index=True)
    plb_res.to_parquet(RESULTS / "placebo_interactions.parquet",
                       index=False)

    # audits
    a = (true_res.groupby(["liquidity_type", "activation_state"])
         .size().rename("n").reset_index())
    a.to_csv(RESULTS / "interaction_eligibility_audit.csv", index=False,
             encoding="utf-8-sig")
    g = (true_res[true_res["activation_state"] == "VALID_AHEAD"]
         .groupby("interaction_path").size().rename("n").reset_index())
    g.to_csv(RESULTS / "gap_interaction_audit.csv", index=False,
             encoding="utf-8-sig")
    p = true_res[true_res["post_reclaim_state"].notna()]
    pd.DataFrame(p["post_reclaim_state"].value_counts()).reset_index()\
        .to_csv(RESULTS / "post_reclaim_fix_audit.csv", index=False,
                encoding="utf-8-sig")
    gen = plb.groupby(["replica_id", "liquidity_type", "side"]).agg(
        n=("level_key", "size"),
        d0_mean=("donor_d0_R", "mean"),
        d0_mean_true=("template_d0_R", "mean")).reset_index()
    gen.to_csv(RESULTS / "placebo_generation_audit.csv", index=False,
               encoding="utf-8-sig")

    print("\nactivation_state:"); print(
        true_res["activation_state"].value_counts().to_string())
    print("\ninteraction_path (VALID_AHEAD):")
    print(g.to_string(index=False))
    print("\npost_reclaim_state:")
    print(p["post_reclaim_state"].value_counts().to_string())
    print("\nP0P1_DONE")


if __name__ == "__main__":
    main()
