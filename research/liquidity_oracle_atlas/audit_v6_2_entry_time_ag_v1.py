"""
audit_v6_2_entry_time_ag_v1.py
==============================

ENTRY-TIME AUDIT for the experimental V6.2 kernel (AG).

WHY THIS EXISTS
---------------
The production Viewer currently displays the V6.1 artifact (951 labels) and
draws geometry at CANDIDATE time, not at the actual ENTRY DECISION. It can
therefore neither validate nor refute V6.2. This audit produces the
frozen-rule evidence picture for V6.2:

    HARD VISUAL INVARIANT
    ---------------------
    Looking at the Entry bar, one must already be able to point at the
    Planned Target among the structures that existed at that moment.
    A structure that forms LATER may never become this trade's target.

What this script does (NO label math change, NO artifact replacement,
NO Viewer change):

  1. runs the LOCAL V6.2 kernel (entry-time target, overlap allowed) on AG;
  2. locates the V6.2 labels whose [entry, exit] window overlaps the OLD
     V6.1 artifact Trade 7 / Trade 8 leg windows (same market segment);
  3. prints the full entry-time evidence for every matched V6.2 label:
       entry_decision_index / entry_fill_index / entry_price
       ALL visible locations at geom_by_decision[entry_decision]
       the selected PLANNED TARGET (price / bounds / member ids)
       target_visible_at_entry            -> must be YES (hard check)
       first_ever_visible_bar per member  -> provenance display only
       old V6.1 target visible at entry?  -> context (expected: NO for the
                                             trades the old logic leaked on)
  4. writes interactive plotly HTML figures per matched label:
       Candidate A (faint) / Entry + Exit markers /
       ENTRY-TIME VISIBLE structures (solid) /
       PLANNED TARGET (highlighted) /
       FUTURE structures at the exit bar (grey, FUTURE -- NOT ELIGIBLE).

Run:
    python research/liquidity_oracle_atlas/audit_v6_2_entry_time_ag_v1.py [SYMBOL]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_2_entry_target import (
    LONG,
    choose_entry_time_target,
    geometry_zones,
    locations_from_geometry_snapshot,
    run_entry_time_target_oracle,
)
from research.liquidity_oracle_atlas.structural_god_oracle_m15_v6_touch_chain import (
    structure_id_from_match,
    intervals_overlap,
)

EPS = 1e-6

# OLD V6.1 artifact reference legs (same market segment as the screenshots)
REF_TRADE_SEQS = (7, 8)
ARTIFACT_TRADES = (
    Path(__file__).resolve().parents[2]
    / "artifacts" / "god_oracle_m15_latest" / "trades.parquet"
)
OUT_DIR = Path(__file__).resolve().parent / "evidence" / "entry_time_audit_v1"

COLOR_SR = "rgba(230, 145, 56, 0.14)"
COLOR_LIQ = "rgba(120, 100, 220, 0.14)"
COLOR_TARGET_LONG = "rgba(38, 166, 91, 0.30)"
COLOR_TARGET_SHORT = "rgba(214, 69, 65, 0.30)"
COLOR_FUTURE = "rgba(140, 140, 140, 0.10)"


def structure_first_visible_map(geom_by_decision) -> dict:
    """structure_id -> first bar index it ever appeared in the geometry."""
    first: dict = {}
    for b, geom in enumerate(geom_by_decision):
        for z in geometry_zones(geom):
            sid = structure_id_from_match(z)
            first.setdefault(sid, b)
    return first


def visible_at_entry(loc_by_decision, d, entry_price, direction, target_price):
    """Strict: a location visible at d whose first-touch edge == target."""
    if d is None or d < 0 or d >= len(loc_by_decision):
        return False
    for L in loc_by_decision[d]:
        if direction == LONG:
            if (abs(float(L.bottom) - float(target_price)) <= EPS
                    and float(L.bottom) > float(entry_price) + EPS):
                return True
        else:
            if (abs(float(L.top) - float(target_price)) <= EPS
                    and float(L.top) < float(entry_price) - EPS):
                return True
    return False


def audit_one_trade(t: dict, geom_by_decision, loc_by_decision, first_vis: dict) -> bool:
    """Print the entry-time evidence for one V6.2 label. Returns all-OK."""
    d = int(t["best_entry_decision_index"])
    f = int(t["best_entry_fill_index"])
    entry_price = float(t["best_entry_price"])
    direction = t["oracle_direction"]
    target_price = float(t["target_price"])

    print("\n" + "=" * 78)
    print(
        f"V6.2 LABEL  source_bar={t['source_bar']}  dir={direction}  "
        f"entry_decision={d}  entry_fill={f}"
    )
    print("=" * 78)
    print(f"  entry_time        : {pd.Timestamp(t['best_entry_fill_time'])}")
    print(f"  entry_price       : {entry_price:.1f}")
    print(f"  planned_target    : {target_price:.1f}  "
          f"(location [{t['target_location_bottom']:.1f}, "
          f"{t['target_location_top']:.1f}])")
    print(f"  target_id         : {t['target_structure_id']}")
    print(f"  exit_bar          : {t['exit_fill_index']}  "
          f"({pd.Timestamp(t['exit_fill_time'])})")
    print(f"  utility           : {t['utility']:.1f}")

    # ---- locations actually visible at the entry decision ----
    geom = geom_by_decision[d]
    locations = locations_from_geometry_snapshot(geom)
    print(f"\n  ALL VISIBLE LOCATIONS AT ENTRY DECISION d={d} "
          f"(geom_by_decision[{d}]):")
    for L in locations:
        sids = ", ".join(m.structure_id for m in L.members)
        print(f"    [{L.bottom:9.1f}, {L.top:9.1f}]  {sids}")

    # ---- hard checks ----
    ok_snapshot = int(t["target_snapshot_index"]) == d
    elig = [
        L for L in locations
        if not intervals_overlap(
            L.bottom, L.top,
            float(t["zone_bottom"]), float(t["zone_top"]), EPS,
        )
    ]
    re_chosen = choose_entry_time_target(elig, entry_price, direction)
    ok_reproduce = (
        re_chosen is not None
        and abs(float(re_chosen[1]) - target_price) <= EPS
    )
    ok_visible = visible_at_entry(
        loc_by_decision, d, entry_price, direction, target_price,
    )
    ok_visible = ok_visible and ok_reproduce

    print("\n  HARD CHECKS:")
    print(f"    target_snapshot_index == entry_decision      : "
          f"{'PASS' if ok_snapshot else 'FAIL'}")
    print(f"    target independently re-derived from geom[{d}]: "
          f"{'PASS' if ok_reproduce else 'FAIL'}")
    print(f"    TARGET VISIBLE AT ENTRY                      : "
          f"{'YES' if ok_visible else 'NO'}")

    # ---- provenance: when did the target members first appear ----
    tgt = re_chosen[0] if re_chosen is not None else None
    if tgt is not None:
        print("\n  TARGET MEMBER PROVENANCE (structure_id -> first ever bar):")
        for m in tgt.members:
            fv = first_vis.get(m.structure_id)
            note = "pre-existing at entry" if (fv is not None and fv <= d) \
                else "APPEARED AFTER ENTRY (LEAK!)"
            print(f"    {m.structure_id}  first_bar={fv}  -> {note}")

    return ok_snapshot and ok_reproduce and ok_visible


def make_figure(t: dict, geom_by_decision, ef, out_dir: Path) -> Path:
    """Entry-Time Geometry figure for one V6.2 label -> HTML."""
    d = int(t["best_entry_decision_index"])
    f = int(t["best_entry_fill_index"])
    src = int(t["source_bar"])
    exit_bar = int(t["exit_fill_index"])
    direction = t["oracle_direction"]

    times = ef["bar_start_time"].to_numpy()
    pad = max(80, (exit_bar - src) // 2)
    lo = max(0, src - pad)
    hi = min(len(ef) - 1, exit_bar + pad)
    seg = ef.iloc[lo:hi + 1]

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=pd.to_datetime(seg["bar_start_time"]),
        open=seg["open"], high=seg["high"],
        low=seg["low"], close=seg["close"],
        name="Kline", increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350", showlegend=False,
    ))

    t0, t1 = pd.Timestamp(times[lo]), pd.Timestamp(times[hi])

    def band(bottom, top, color, text, line_color=None, dash=None):
        fig.add_shape(type="rect", x0=t0, x1=t1, y0=float(bottom),
                      y1=float(top), fillcolor=color, opacity=1.0,
                      line=dict(width=1, color=line_color or color,
                                dash=dash or "solid"))
        fig.add_annotation(
            x=t0, y=float(top), text=text, showarrow=False,
            xanchor="left", yshift=8, font=dict(size=9),
        )

    # 1) Candidate A (faint)
    band(t["zone_bottom"], t["zone_top"], "rgba(90,120,180,0.12)",
         "Candidate A (source)")

    # 2) ENTRY-TIME VISIBLE structures (geom at the entry decision)
    entry_locs = locations_from_geometry_snapshot(geom_by_decision[d])
    tgt_bottom = float(t["target_location_bottom"])
    tgt_top = float(t["target_location_top"])
    for L in entry_locs:
        if abs(float(L.bottom) - tgt_bottom) <= EPS and \
           abs(float(L.top) - tgt_top) <= EPS:
            continue  # drawn separately as the planned target
        fam = {m.family for m in L.members}
        color = COLOR_LIQ if "LIQ" in fam else COLOR_SR
        band(L.bottom, L.top, color, "ENTRY-TIME VISIBLE")

    # 3) PLANNED TARGET (highlighted)
    tcolor = COLOR_TARGET_LONG if direction == LONG else COLOR_TARGET_SHORT
    band(tgt_bottom, tgt_top, tcolor,
         f"PLANNED TARGET AT ENTRY ({float(t['target_price']):.1f})",
         line_color="#26a69a" if direction == LONG else "#d64541",
         dash="dash")

    # 4) FUTURE structures (geometry at the exit bar) -- grey, not eligible
    exit_locs = locations_from_geometry_snapshot(geom_by_decision[exit_bar])
    for L in exit_locs:
        overl = any(
            intervals_overlap(L.bottom, L.top, E.bottom, E.top, EPS)
            for E in entry_locs
        )
        if overl:
            continue
        band(L.bottom, L.top, COLOR_FUTURE, "FUTURE -- NOT ELIGIBLE",
             line_color="rgba(140,140,140,0.6)", dash="dot")

    # 5) Entry / Exit markers
    fig.add_vline(x=pd.Timestamp(times[f]), line_width=1,
                  line_dash="dash", line_color="#26a69a",
                  annotation_text="entry decision fill",
                  annotation_font_size=9)
    fig.add_vline(x=pd.Timestamp(times[exit_bar]), line_width=1,
                  line_dash="dash", line_color="#888888",
                  annotation_text="exit", annotation_font_size=9)

    fig.update_layout(
        title=(
            f"V6.2 ENTRY-TIME AUDIT  {t['oracle_direction']}  "
            f"source={src}  entry_decision={d}  fill={f}  "
            f"entry={float(t['best_entry_price']):.1f}  "
            f"target={float(t['target_price']):.1f}  "
            f"(geom drawn at d={d}, NOT at candidate time)"
        ),
        xaxis_rangeslider_visible=False,
        template="plotly_dark", height=760,
    )
    out = out_dir / (
        f"v62_trade_src{src}_{direction.lower()}_d{d}.html"
    )
    fig.write_html(str(out), include_plotlyjs="cdn")
    return out


def main(symbol: str = "AG") -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---- reference legs from the OLD V6.1 artifact ----
    ref = pd.read_parquet(ARTIFACT_TRADES)
    ref = ref[ref.trade_seq.isin(REF_TRADE_SEQS)]

    print("=" * 78)
    print(f"V6.2 ENTRY-TIME AUDIT  symbol={symbol}")
    print("=" * 78)
    print("reference legs from the OLD V6.1 artifact (951-label view):")
    for _, r in ref.iterrows():
        print(f"  V6.1 Trade {int(r.trade_seq)}: {r.oracle_direction}  "
              f"bars [{int(r.candidate_decision_index)}, "
              f"{int(r.exit_fill_index)}]  entry={r.best_entry_price:.1f} "
              f"target={r.target_price:.1f}")

    res = run_entry_time_target_oracle(symbol)
    trades = res["trades"]
    geom_by_decision = res["geom_by_decision"]
    ef = res["exec_frame"]

    print(f"\nV6.2 canonical labels (local kernel): {len(trades)}")

    first_vis = structure_first_visible_map(geom_by_decision)
    loc_by_decision = [
        locations_from_geometry_snapshot(g) for g in geom_by_decision
    ]

    # ---- old V6.1 targets: were they visible at the OLD entry decision? ----
    print("\n--- OLD V6.1 reference targets vs ENTRY-TIME visibility "
          "(context) ---")
    for _, r in ref.iterrows():
        d_old = int(r.best_entry_decision_index)
        vis = visible_at_entry(
            loc_by_decision, d_old,
            float(r.best_entry_price), r.oracle_direction,
            float(r.target_price),
        )
        print(f"  V6.1 Trade {int(r.trade_seq)}: old target "
              f"{float(r.target_price):.1f} visible at OLD entry decision "
              f"{d_old}: {'YES' if vis else 'NO'}")

    # ---- matched V6.2 labels in the same market segments ----
    all_ok = True
    figure_paths = []
    for _, r in ref.iterrows():
        c, x = int(r.candidate_decision_index), int(r.exit_fill_index)
        matched = [
            t for t in trades
            if int(t["best_entry_fill_index"]) <= x
            and int(t["exit_fill_index"]) >= c
        ]
        print("\n" + "#" * 78)
        print(f"SEGMENT OF OLD V6.1 Trade {int(r.trade_seq)} "
              f"(bars [{c}, {x}])  ->  {len(matched)} V6.2 label(s)")
        print("#" * 78)
        if not matched:
            print("  (no V6.2 label overlaps this leg window)")
        for t in matched:
            ok = audit_one_trade(t, geom_by_decision, loc_by_decision, first_vis)
            all_ok = all_ok and ok
            figure_paths.append(make_figure(t, geom_by_decision, ef, OUT_DIR))

    print("\n" + "=" * 78)
    print("ENTRY-TIME INVARIANT (hard visual invariant, machine-checked):")
    print(f"  every matched V6.2 label has its planned target visible at the "
          f"entry decision: {'PASS' if all_ok else 'FAIL'}")
    print(f"  figures written to: {OUT_DIR}")
    for p in figure_paths:
        print(f"    {p.name}")
    print("=" * 78)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sym = sys.argv[1] if len(sys.argv) > 1 else "AG"
    raise SystemExit(main(sym))
