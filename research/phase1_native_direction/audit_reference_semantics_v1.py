"""Reference Semantics Audit（不训练任何模型）。

问题：以 trigger-bar close 作为 Phase 1 reference_price，是否真正代表
"OB entered 后尚未明显走掉行情" 的合理测量时点？

方向坐标：+ = 顺 native OB 方向。
  near_edge = 价格进入 OB 时先触及的那条边（LONG: zone_high / SHORT: zone_low）
  far_edge  = 失效侧那条边（LONG: zone_low  / SHORT: zone_high）
  依据 build_ob_candidate_universe_v3.touch_bar_state：bias=1 时 far edge
  = zone_low，bias=-1 时 far edge = zone_high。

⚠️ Source Owner 未提供 touch_price / first_touch_price / touch_reference
   （已确认），因此精确的 intrabar 首次触碰价与时间不可恢复：
   INTRABAR_TOUCH_PRICE_UNRESOLVED。
   本审计使用 **canonical 进入边界（near_edge）** 作为可无歧义确定的
   参考水平；trigger bar 内的 MFE/MAE 只能给出 OHLC 上界，并显式标注
   同 bar 内方向不可分辨的情况。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from research.phase1_native_direction.nd_contract_v1 import (
    RESULTS, SYMBOLS, compute_atr5, get_bars,
)
from research.phase1_native_direction.nd_label_v1 import BULLISH

V3 = "research/exports/ob_candidate_universe_v3/{sym}_candidates.csv"


def load_candidates():
    parts = []
    for s in SYMBOLS:
        c = pd.read_csv(V3.format(sym=s), low_memory=False)
        c["symbol"] = s
        parts.append(c[[
            "candidate_id", "symbol", "source_tf", "source_ob_bias",
            "source_ob_zone_low", "source_ob_zone_high",
            "touch_5m_bar_index", "touch_ordinal", "is_first_touch",
            "touch_close_beyond_far_edge",
            "touch_intrabar_far_edge_breach", "touch_reclaimed_by_close",
        ]])
    d = pd.concat(parts, ignore_index=True)
    d["candidate_id"] = d["candidate_id"].astype(str)
    return d.rename(columns={"source_ob_bias": "native_direction"})


def build():
    c = load_candidates()
    lab = pd.read_parquet(RESULTS / "native_labels_v1.parquet")
    lab["candidate_id"] = lab["candidate_id"].astype(str)
    c = c.merge(lab[["candidate_id", "native_label", "native_status"]],
                on="candidate_id", how="left", validate="one_to_one")

    rows = []
    for sym, g in c.groupby("symbol"):
        bars = get_bars(sym)
        atr = compute_atr5(bars)
        o, h, l, cl = (bars["open"], bars["high"], bars["low"],
                       bars["close"])
        n = len(cl)
        for (cid, tf, d, zl, zh, di, nat) in zip(
                g["candidate_id"], g["source_tf"], g["native_direction"],
                g["source_ob_zone_low"], g["source_ob_zone_high"],
                g["touch_5m_bar_index"], g["native_label"]):
            di = int(di)
            if di < 0 or di >= n:
                continue
            a = float(atr[di])
            if not np.isfinite(a) or a <= 0:
                continue
            near = float(zh) if d == BULLISH else float(zl)
            far = float(zl) if d == BULLISH else float(zh)
            cpx = float(cl[di])
            fav = float(h[di]) if d == BULLISH else float(l[di])
            adv = float(l[di]) if d == BULLISH else float(h[di])

            d_near = d * (cpx - near) / a          # 进入边 → close 的方向位移
            d_far = d * (cpx - far) / a            # 距失效边剩余空间
            mfe = d * (fav - near) / a             # bar 内顺向上界
            mae = d * (adv - near) / a             # bar 内逆向上界
            lo, hi = min(zl, zh), max(zl, zh)

            rows.append(dict(
                candidate_id=cid, symbol=sym, source_tf=tf,
                native_direction=int(d), touch_ordinal=None,
                ATR5=a, zone_low=float(zl), zone_high=float(zh),
                near_edge=near, far_edge=far, close=cpx,
                close_inside_zone=bool(lo <= cpx <= hi),
                close_beyond_near_edge=bool(d_near > 0),
                close_beyond_far_edge=bool(d_far < 0),
                dist_close_to_near_edge_R=d_near,
                dist_close_to_far_edge_R=d_far,
                touch_to_close_R=d_near,
                native_MFE_R=mfe, native_MAE_R=mae,
                bar_range_R=(float(h[di]) - float(l[di])) / a,
                bar_order_ambiguous=bool(mfe >= 0.5 and mae <= -0.5),
                native_label=nat,
            ))
    return pd.DataFrame(rows)


def _agg(g, key=None):
    d = dict(
        n=len(g),
        close_inside_zone_pct=round(float(g["close_inside_zone"].mean()), 4),
        close_beyond_near_pct=round(
            float(g["close_beyond_near_edge"].mean()), 4),
        close_beyond_far_pct=round(
            float(g["close_beyond_far_edge"].mean()), 4),
        touch_to_close_p10=round(float(g["touch_to_close_R"].quantile(.10)), 4),
        touch_to_close_p25=round(float(g["touch_to_close_R"].quantile(.25)), 4),
        touch_to_close_median=round(float(g["touch_to_close_R"].median()), 4),
        touch_to_close_p75=round(float(g["touch_to_close_R"].quantile(.75)), 4),
        touch_to_close_p90=round(float(g["touch_to_close_R"].quantile(.90)), 4),
        touch_to_close_p95=round(float(g["touch_to_close_R"].quantile(.95)), 4),
        MFE_ge_0_5R_pct=round(float((g["native_MFE_R"] >= 0.5).mean()), 4),
        MFE_ge_1_0R_pct=round(float((g["native_MFE_R"] >= 1.0).mean()), 4),
        MFE_ge_1_5R_pct=round(float((g["native_MFE_R"] >= 1.5).mean()), 4),
        MFE_ge_2_5R_pct=round(float((g["native_MFE_R"] >= 2.5).mean()), 4),
        MAE_ge_0_5R_pct=round(float((g["native_MAE_R"] <= -0.5).mean()), 4),
        MAE_ge_1_0R_pct=round(float((g["native_MAE_R"] <= -1.0).mean()), 4),
        bar_order_ambiguous_pct=round(
            float(g["bar_order_ambiguous"].mean()), 4),
        room_to_far_edge_median_R=round(
            float(g["dist_close_to_far_edge_R"].median()), 4),
        native_base_rate=(round(float(g["native_label"].mean()), 4)
                          if g["native_label"].notna().any() else None),
    )
    if key is not None:
        # 注意：必须是固定列名 "_k"，用 {key: key} 会把分组值变成列名
        d = {"_k": key, **d}
    return d


def profile(t, by=None, keyname=None):
    rows = []
    if by is None:
        rows.append(_agg(t, "ALL" if keyname else None))
    else:
        for k, g in t.groupby(by):
            rows.append(_agg(g, k))
    df = pd.DataFrame(rows)
    if keyname:
        df = df.rename(columns={"_k": keyname})
    return df


def main():
    t = build()
    t.to_csv(RESULTS / "reference_semantics_events.csv", index=False,
             encoding="utf-8-sig")
    print(f"[ref-audit] events={len(t)}", flush=True)

    p = profile(t)
    p.to_csv(RESULTS / "reference_semantics_profile.csv", index=False,
             encoding="utf-8-sig")
    print("\n=== ALL ===")
    print(p.T.to_string(header=False))

    # 诊断：median=0 是否来自恰好等于 near_edge 的质量点
    z = t["touch_to_close_R"]
    print("\n=== touch_to_close_R 分布诊断 ===")
    print(f"  exactly == 0        : {float((z == 0).mean()):.4f}")
    print(f"  < 0                 : {float((z < 0).mean()):.4f}")
    print(f"  > 0                 : {float((z > 0).mean()):.4f}")
    print(f"  |z| <= 0.05R        : {float((z.abs() <= 0.05).mean()):.4f}")
    print("  quantiles:",
          {f"p{int(q*100)}": round(float(z.quantile(q)), 4)
           for q in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)})

    for by, fn, kn in (("symbol", "reference_by_symbol.csv", "symbol"),
                       ("source_tf", "reference_by_source_tf.csv",
                        "source_tf"),
                       ("native_direction", "reference_by_direction.csv",
                        "native_direction")):
        d = profile(t, by, kn)
        d.to_csv(RESULTS / fn, index=False, encoding="utf-8-sig")
        print(f"\n=== by {kn} ===")
        print(d[[kn, "n", "close_inside_zone_pct",
                 "touch_to_close_median", "touch_to_close_p90",
                 "MFE_ge_1_0R_pct", "MAE_ge_1_0R_pct",
                 "native_base_rate"]].to_string(index=False))

    # ---- trigger bar excursion 明细 ----
    ex = []
    for lo, hi, name in ((None, 0.0, "<=0R"),
                         (0.0, 0.25, "0~0.25R"),
                         (0.25, 0.5, "0.25~0.5R"),
                         (0.5, 1.0, "0.5~1R"),
                         (1.0, None, ">1R")):
        m = (t["touch_to_close_R"] <= hi) if lo is None else (
            (t["touch_to_close_R"] > lo)
            if hi is None else
            (t["touch_to_close_R"] > lo) & (t["touch_to_close_R"] <= hi))
        g = t[m]
        ex.append(dict(bucket=name, n=len(g),
                       share=round(len(g) / len(t), 4),
                       native_base_rate=(
                           round(float(g["native_label"].mean()), 4)
                           if g["native_label"].notna().any() else None),
                       median_MFE_R=round(float(g["native_MFE_R"].median()), 4),
                       median_MAE_R=round(float(g["native_MAE_R"].median()), 4)))
    edf = pd.DataFrame(ex)
    edf.to_csv(RESULTS / "trigger_bar_excursion.csv", index=False,
               encoding="utf-8-sig")
    print("\n=== 按 touch_to_close_R 分组（仅描述，不训练）===")
    print(edf.to_string(index=False))

    print("\n=== 与 canonical 布尔字段交叉校验 ===")
    c = load_candidates()
    m = t.merge(c[["candidate_id", "touch_close_beyond_far_edge"]],
                on="candidate_id", how="left", validate="one_to_one")
    agree = (m["close_beyond_far_edge"].astype(bool)
             == m["touch_close_beyond_far_edge"].astype(bool))
    print(f"  close_beyond_far_edge 与 canonical 一致率: "
          f"{agree.mean():.6f}  (n={len(m)})")

    print("\nINTRABAR_TOUCH_PRICE_UNRESOLVED: Source Owner 未提供 "
          "touch_price/first_touch_price；本审计以 canonical near_edge "
          "作为可无歧义确定的进入水平。")
    print("REFERENCE_AUDIT_DONE")


if __name__ == "__main__":
    main()
