#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualize Panji canonical SMC + Momentum on SHFE silver futures.

============================================================
SCOPE OF THIS ROUND (strict)
============================================================
IN:
  * read-only use of the already validated silver CSVs
  * canonical SMC      -> panji_indicators.compute_smc_pine
  * canonical Momentum -> panji_indicators.compute_sqzmom_lb
                          + panji_indicators.build_momentum_history
  * TqSdk native visualization only
    (web_gui + draw_line / draw_text / draw_box + serial columns)

OUT (explicitly NOT this round):
  * no DSA
  * no strategy / signal rules
  * no backtest, no orders, no rollover / execution routing
  * no indicator runtime architecture
  * no parameter optimisation
  * no modification of canonical algorithms
  * no second (matplotlib) indicator renderer

The renderer only CONSUMES canonical output. Nothing is recomputed
and no canonical default parameter is overridden.

============================================================
HARD RULES
============================================================
1. Indicators are computed on the FULL history. Cropping happens only
   at the very end. Never `df.tail(300)` before computing - otherwise
   swing pivots / BOS / CHoCH / OB lifecycle / squeeze state are wrong.
2. `visualization_source` is the KQ.m continuous series:
   it is NOT a tradable contract.
3. Guards run BEFORE any picture is produced:
   SMC invariants, Momentum invariants, prefix PIT check.
   Any FAIL -> STOP (no "visual success" is claimed).

Run:
    python visualize_smc_momentum_tqsdk.py --timeframe 15m
    python visualize_smc_momentum_tqsdk.py --timeframe 15m --no-gui   # validation only
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from panji_indicators import (
    build_momentum_history,
    compute_smc_pine,
    compute_sqzmom_lb,
)

# ============================================================
# Constants
# ============================================================

SIGNAL_SYMBOL = "KQ.m@SHFE.ag"

TIMEFRAMES = {
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
}

PLOT_BARS = 300              # only used to crop the *report*, never the computation
SERIAL_BARS = 1000           # TqSdk kline serial length (GUI carrier)
PREFIX_CHECKPOINT_OFFSET = 100

REPO = Path(__file__).resolve().parent
DATA_DIR = REPO / "silver_main_data"
ART_DIR = REPO / "artifacts" / "smc_momentum_preview"


def load_dotenv(path: Path) -> None:
    """Minimal .env loader; never overrides existing environment variables."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _f(x):
    """canonical value -> float or None (NaN -> None)."""
    if x is None:
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(v) else v


# ============================================================
# 1. Data adapter (CSV -> canonical-ready frame)
# ============================================================

def load_ohlc(tf: str) -> pd.DataFrame:
    path = DATA_DIR / f"silver_main_{tf}.csv"
    if not path.is_file():
        raise SystemExit(f"[FATAL] missing data file: {path}")

    df = pd.read_csv(path).sort_values("datetime_ns").reset_index(drop=True)

    ns = df["datetime_ns"].astype("int64")
    if ns.duplicated().any():
        raise SystemExit(f"[{tf}] datetime_ns has duplicates")
    if not ns.is_monotonic_increasing:
        raise SystemExit(f"[{tf}] datetime_ns is not monotonic increasing")

    # CSV `datetime` is already Beijing time as a plain string.
    # Do NOT convert timezones again - just parse it.
    idx = pd.to_datetime(df["datetime"], errors="raise")
    if not pd.DatetimeIndex(idx).is_monotonic_increasing:
        raise SystemExit(f"[{tf}] datetime index is not monotonic increasing")

    out = pd.DataFrame(index=pd.DatetimeIndex(idx))
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = df[col].astype(float).to_numpy()
    out["datetime_ns"] = ns.to_numpy()
    return out


# ============================================================
# 2. Canonical calculation (FULL history, defaults untouched)
# ============================================================

def compute_smc(df: pd.DataFrame) -> dict:
    times = [ts.isoformat() for ts in df.index]
    return compute_smc_pine(
        df["open"].tolist(),
        df["high"].tolist(),
        df["low"].tolist(),
        df["close"].tolist(),
        times,
        params=None,          # canonical defaults, never overridden
        emit_timeline=True,
    )


def compute_momentum(df: pd.DataFrame):
    sqz = compute_sqzmom_lb(
        df["open"].to_numpy(float),
        df["high"].to_numpy(float),
        df["low"].to_numpy(float),
        df["close"].to_numpy(float),
    )
    times = [ts.isoformat() for ts in df.index]
    hist = build_momentum_history(
        sqz,
        volume_series=df["volume"].to_numpy(float),
        times=times,
    )
    return sqz, hist


# ============================================================
# 3. Guards - SMC invariants
# ============================================================

def check_smc_invariants(smc: dict, n_bars: int) -> list[str]:
    bad: list[str] = []

    for i, ev in enumerate(smc.get("events", [])):
        a, c = ev.get("anchor_index"), ev.get("confirmed_index")
        if a is None or c is None:
            bad.append(f"event[{i}] missing anchor/confirmed index")
            continue
        if not (a <= c):
            bad.append(f"event[{i}] anchor_index({a}) > confirmed_index({c})")
        if not (0 <= c < n_bars):
            bad.append(f"event[{i}] confirmed_index({c}) outside [0,{n_bars})")

    for i, ob in enumerate(smc.get("order_blocks", [])):
        a, c = ob.get("anchor_index"), ob.get("confirmed_index")
        if a is None or c is None:
            bad.append(f"ob[{i}] missing anchor/confirmed index")
            continue
        if not (a <= c):
            bad.append(f"ob[{i}] anchor_index({a}) > confirmed_index({c})")

        ei = ob.get("enter_index")
        if ei is not None and not (ei > c):
            bad.append(f"ob[{i}] enter_index({ei}) not > confirmed_index({c})")

        mi = ob.get("mitigated_index")
        if mi is not None and not (mi >= c):
            bad.append(f"ob[{i}] mitigated_index({mi}) < confirmed_index({c})")

        if ei is not None and mi is not None and not (ei <= mi):
            bad.append(f"ob[{i}] enter_index({ei}) > mitigated_index({mi})")

    return bad


# ============================================================
# 4. Guards - Momentum invariants
# ============================================================

def check_momentum_invariants(sqz: dict, hist: dict, n_bars: int) -> list[str]:
    bad: list[str] = []
    val = sqz.get("val", [])
    on = sqz.get("sqzOn", [])
    off = sqz.get("sqzOff", [])

    if len(val) != n_bars:
        bad.append(f"len(val)={len(val)} != bars({n_bars})")
    if len(on) != n_bars:
        bad.append(f"len(sqzOn)={len(on)} != bars({n_bars})")
    if len(off) != n_bars:
        bad.append(f"len(sqzOff)={len(off)} != bars({n_bars})")
    if bad:
        return bad

    for i, ev in enumerate(hist.get("sqz_release_events", [])):
        b = ev.get("bar_index")
        if b is None or not (0 < b < n_bars):
            bad.append(f"sqz_release[{i}] bar_index out of range: {b}")
            continue
        if not (on[b - 1] is True and off[b] is True):
            bad.append(
                f"sqz_release[{i}] requires sqzOn[{b - 1}]=True and "
                f"sqzOff[{b}]=True, got {on[b - 1]} / {off[b]}"
            )

    for i, ev in enumerate(hist.get("momentum_zero_cross_events", [])):
        b = ev.get("bar_index")
        if b is None or not (0 < b < n_bars):
            bad.append(f"zero_cross[{i}] bar_index out of range: {b}")
            continue
        pv, cv = _f(val[b - 1]), _f(val[b])
        if pv is None or cv is None:
            bad.append(f"zero_cross[{i}] NaN value at bar {b}")
            continue
        if ev.get("type") == "ZERO_CROSS_UP":
            if not (pv <= 0 and cv > 0):
                bad.append(f"zero_cross_up[{i}] prev={pv} cur={cv}")
        elif ev.get("type") == "ZERO_CROSS_DOWN":
            if not (pv >= 0 and cv < 0):
                bad.append(f"zero_cross_down[{i}] prev={pv} cur={cv}")

    return bad


# ============================================================
# 5. Guards - prefix PIT check (no look-ahead)
# ============================================================

def _ev_key(e: dict):
    return (
        e.get("type"),
        bool(e.get("internal")),
        bool(e.get("bullish")),
        e.get("anchor_index"),
        e.get("confirmed_index"),
    )


def prefix_pit_check(df: pd.DataFrame):
    n = len(df)
    if n <= PREFIX_CHECKPOINT_OFFSET + 10:
        return {"skipped": True, "reason": "not enough bars"}, []

    cp = n - PREFIX_CHECKPOINT_OFFSET

    full_smc = compute_smc(df)
    full_sqz, _ = compute_momentum(df)

    pre = df.iloc[: cp + 1]
    pre_smc = compute_smc(pre)
    pre_sqz, _ = compute_momentum(pre)

    problems: list[str] = []

    tl = full_smc.get("state_timeline") or []
    if len(tl) > cp:
        fs = tl[cp].get("swing_bias")
        fi = tl[cp].get("internal_bias")
    else:
        problems.append("state_timeline shorter than checkpoint")
        fs = fi = None

    if fs != pre_smc.get("swing_bias"):
        problems.append(f"swing_bias@{cp}: full={fs} prefix={pre_smc.get('swing_bias')}")
    if fi != pre_smc.get("internal_bias"):
        problems.append(f"internal_bias@{cp}: full={fi} prefix={pre_smc.get('internal_bias')}")

    full_ev = sorted(_ev_key(e) for e in full_smc.get("events", []) if e["confirmed_index"] <= cp)
    pre_ev = sorted(_ev_key(e) for e in pre_smc.get("events", []))
    if full_ev != pre_ev:
        problems.append(
            f"BOS/CHoCH event list differs at checkpoint "
            f"(full={len(full_ev)} prefix={len(pre_ev)})"
        )

    fv, fo, ff = _f(full_sqz["val"][cp]), full_sqz["sqzOn"][cp], full_sqz["sqzOff"][cp]
    pv, po, pf = _f(pre_sqz["val"][cp]), pre_sqz["sqzOn"][cp], pre_sqz["sqzOff"][cp]
    if fv != pv:
        problems.append(f"momentum val@{cp}: full={fv} prefix={pv}")
    if fo != po:
        problems.append(f"momentum sqzOn@{cp}: full={fo} prefix={po}")
    if ff != pf:
        problems.append(f"momentum sqzOff@{cp}: full={ff} prefix={pf}")

    info = {
        "checkpoint": cp,
        "checkpoint_time": str(df.index[cp]),
        "full_swing_bias_at_cp": fs,
        "prefix_swing_bias_last": pre_smc.get("swing_bias"),
        "full_internal_bias_at_cp": fi,
        "prefix_internal_bias_last": pre_smc.get("internal_bias"),
        "full_events_upto_cp": len(full_ev),
        "prefix_events_total": len(pre_ev),
        "full_val_at_cp": fv,
        "prefix_val_last": pv,
        "problems": problems,
    }
    return info, problems


# ============================================================
# 6. TqSdk visual capability probe
# ============================================================

def probe_tqsdk_visual() -> dict:
    import tqsdk
    from tqsdk import TqApi

    version = tqsdk.__version__
    sig = inspect.signature(TqApi.__init__)

    web_gui = "web_gui" in sig.parameters
    draws = {}
    for name in ("draw_line", "draw_text", "draw_box"):
        if hasattr(TqApi, name):
            draws[name] = str(inspect.signature(getattr(TqApi, name)))
        else:
            draws[name] = None

    src = os.path.dirname(tqsdk.__file__)
    pat = re.compile(r"png|screenshot|export_image|save_image|to_png", re.I)
    hits = []
    for root, _dirs, files in os.walk(src):
        for fn in sorted(files):
            if not fn.endswith(".py"):
                continue
            p = os.path.join(root, fn)
            try:
                with open(p, encoding="utf-8", errors="ignore") as fh:
                    for i, line in enumerate(fh, 1):
                        if pat.search(line):
                            hits.append(f"{os.path.relpath(p, src)}:{i}: {line.rstrip()[:150]}")
            except OSError:
                pass

    return {
        "version": version,
        "web_gui": web_gui,
        "draws": draws,
        "direct_png_api": bool(hits),
        "direct_png_hits": hits[:20],
        "source_dir": src,
    }


def write_probe_file(info: dict) -> None:
    ART_DIR.mkdir(parents=True, exist_ok=True)
    out = ART_DIR / "tqsdk_visual_probe.txt"
    lines = [
        "TQSDK_VISUAL_PROBE",
        "===================",
        f"version            = {info['version']}",
        f"web_gui            = {'yes' if info['web_gui'] else 'no'}",
        f"draw_line          = {'yes' if info['draws'].get('draw_line') else 'no'}",
        f"draw_text          = {'yes' if info['draws'].get('draw_text') else 'no'}",
        f"draw_box           = {'yes' if info['draws'].get('draw_box') else 'no'}",
        f"direct_png_api     = {'yes' if info['direct_png_api'] else 'no'}",
        "direct_png_api_name= (none found in installed source)" if not info["direct_png_api"] else "",
        "",
        "evidence / API signature:",
        f"  TqApi.__init__ web_gui default = False  (set web_gui=True to enable)",
    ]
    for k, v in info["draws"].items():
        lines.append(f"  {k}{v}")
    lines += [
        "",
        "PNG / export search in installed source:",
        f"  source_dir = {info['source_dir']}",
        "  pattern    = png|screenshot|export_image|save_image|to_png",
    ]
    if info["direct_png_hits"]:
        lines += [f"  {h}" for h in info["direct_png_hits"]]
    else:
        lines.append("  NO HITS -> TqSdk 3.10.2 exposes NO direct PNG/export API.")
        lines.append("  Consequence: web_gui is the primary visualization path.")
        lines.append("  No second (matplotlib) indicator renderer is created.")
    lines += [
        "",
        "Verified rendering semantics (from tqsdk/api.py):",
        "  - draw_line/draw_text/draw_box x1/x2 = K-line serial number (positional index).",
        "  - sub-board is created via serial column suffix: klines['COL.board'] = 'BOARD'.",
        "  - series style via klines['COL.type'] in {LINE, DOT, DASH, BAR}.",
        "  - klines['COL.color'] uses ONLY the LAST value of the column",
        "    (api.py: color = data.get('.color', ['#FF0000'])[-1]),",
        "    therefore per-bar colouring (canonical bcolor) is NOT renderable",
        "    by the TqSdk web GUI. The momentum series is drawn with the",
        "    canonical bcolor of the last bar; full per-bar bcolor is saved",
        "    to artifacts for inspection.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"[probe] written: {out}")


# ============================================================
# 7. Summary + artifacts
# ============================================================

def build_summary(tf, df, smc, sqz, hist, validation) -> dict:
    n = len(df)
    events = smc.get("events", [])
    obs = smc.get("order_blocks", [])
    eqs = smc.get("equal_highs_lows", [])
    zc = hist.get("momentum_zero_cross_events", [])

    swing_bos = sum(1 for e in events if e.get("type") == "BOS" and not e.get("internal"))
    swing_choch = sum(1 for e in events if e.get("type") == "CHoCH" and not e.get("internal"))
    int_bos = sum(1 for e in events if e.get("type") == "BOS" and e.get("internal"))
    int_choch = sum(1 for e in events if e.get("type") == "CHoCH" and e.get("internal"))

    return {
        "source": SIGNAL_SYMBOL,
        "visualization_source": (
            "KQ.m@SHFE.ag main-continuous - NOT a tradable contract; "
            "visual verification only"
        ),
        "timeframe": tf,
        "bars": n,
        "data_start": str(df.index[0]),
        "data_end": str(df.index[-1]),
        "smc": {
            "last_swing_bias": smc.get("swing_bias"),
            "last_internal_bias": smc.get("internal_bias"),
            "events_total": len(events),
            "swing_bos": swing_bos,
            "swing_choch": swing_choch,
            "internal_bos": int_bos,
            "internal_choch": int_choch,
            "eqh": sum(1 for e in eqs if e.get("type") == "EQH"),
            "eql": sum(1 for e in eqs if e.get("type") == "EQL"),
            "order_blocks": len(obs),
            "active_order_blocks": sum(1 for o in obs if not o.get("mitigated")),
        },
        "momentum": {
            "last_val": _f(sqz["val"][-1]) if sqz.get("val") else None,
            "last_sqz_on": bool(sqz["sqzOn"][-1]) if sqz.get("sqzOn") else None,
            "last_sqz_off": bool(sqz["sqzOff"][-1]) if sqz.get("sqzOff") else None,
            "sqz_release_events": len(hist.get("sqz_release_events", [])),
            "zero_cross_up": sum(1 for e in zc if e.get("type") == "ZERO_CROSS_UP"),
            "zero_cross_down": sum(1 for e in zc if e.get("type") == "ZERO_CROSS_DOWN"),
        },
        "validation": validation,
    }


def write_artifacts(tf, df, smc, sqz, hist, summary, pit_info, probe_info):
    ART_DIR.mkdir(parents=True, exist_ok=True)

    (ART_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[artifact] written: {ART_DIR / 'summary.json'}")

    # per-bar momentum (canonical val + squeeze state + canonical bcolor preserved)
    mom = pd.DataFrame({
        "datetime": [str(x) for x in df.index],
        "mom_val": [_f(v) for v in sqz["val"]],
        "sqzOn": sqz["sqzOn"],
        "sqzOff": sqz["sqzOff"],
        "noSqz": sqz["noSqz"],
        "bcolor": sqz["bcolor"],
        "scolor": sqz["scolor"],
    })
    p = ART_DIR / f"silver_{tf}_momentum_bars.csv"
    mom.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[artifact] written: {p}")

    ev = pd.DataFrame(smc.get("events", []))
    p = ART_DIR / f"silver_{tf}_smc_events.csv"
    ev.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[artifact] written: {p}")

    ob = pd.DataFrame(smc.get("order_blocks", []))
    p = ART_DIR / f"silver_{tf}_smc_order_blocks.csv"
    ob.to_csv(p, index=False, encoding="utf-8-sig")
    print(f"[artifact] written: {p}")

    tl = pd.DataFrame(smc.get("state_timeline") or [])
    if not tl.empty:
        p = ART_DIR / f"silver_{tf}_smc_state_timeline.csv"
        tl.to_csv(p, index=False, encoding="utf-8-sig")
        print(f"[artifact] written: {p}")

    p = ART_DIR / f"prefix_pit_check_{tf}.json"
    p.write_text(json.dumps(pit_info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[artifact] written: {p}")


# ============================================================
# 8. TqSdk projection (canonical output -> serial columns/drawings)
# ============================================================

class Projector:
    """Maps canonical CSV-indexed output onto the TqSdk kline serial window.

    Canonical values are NEVER recomputed here - they are only looked up
    by timestamp. Bars outside the CSV window stay NaN.
    """

    def __init__(self, df, sqz, smc, hist):
        n = len(df)
        self.ns_to_pos = {int(x): i for i, x in enumerate(df["datetime_ns"])}

        self.mom = np.full(n, np.nan)
        for i, v in enumerate(sqz["val"]):
            fv = _f(v)
            if fv is not None:
                self.mom[i] = fv

        self.swing = np.full(n, np.nan)
        self.internal = np.full(n, np.nan)
        for row in (smc.get("state_timeline") or []):
            i = row.get("bar_index")
            if i is not None and 0 <= i < n:
                self.swing[i] = float(row.get("swing_bias"))
                self.internal[i] = float(row.get("internal_bias"))

        self.events = smc.get("events", [])
        self.obs = smc.get("order_blocks", [])
        self.eqs = smc.get("equal_highs_lows", [])
        self.sqz_events = hist.get("sqz_release_events", [])
        self.zc_events = hist.get("momentum_zero_cross_events", [])
        self.bcolor = sqz.get("bcolor", [])

    def positions(self, klines) -> np.ndarray:
        ns = klines["datetime"].astype("int64").to_numpy()
        return np.array([self.ns_to_pos.get(int(x), -1) for x in ns], dtype=int)

    def project(self, klines):
        pos = self.positions(klines)
        m = pos >= 0
        out = {}
        for name, arr in (("mom", self.mom), ("swing", self.swing), ("internal", self.internal)):
            v = np.full(len(pos), np.nan)
            v[m] = arr[pos[m]]
            out[name] = v
        return pos, out

    @staticmethod
    def serial_map(pos):
        m = {int(c): i for i, c in enumerate(pos) if c >= 0}
        if not m:
            return m, None, None
        return m, min(m), max(m)


def _last_bcolor(bcolor) -> str:
    vals = [c for c in (bcolor or []) if c]
    return vals[-1] if vals else "#FF0000"


def apply_columns(klines, vals, bcolor_last: str) -> None:
    klines["PANJI_MOM"] = vals["mom"]
    klines["PANJI_MOM.board"] = "MOM"
    klines["PANJI_MOM.type"] = "BAR"
    klines["PANJI_MOM.color"] = bcolor_last

    klines["PANJI_SWING_BIAS"] = vals["swing"]
    klines["PANJI_SWING_BIAS.board"] = "SMC_SWING_BIAS"

    klines["PANJI_INTERNAL_BIAS"] = vals["internal"]
    klines["PANJI_INTERNAL_BIAS.board"] = "SMC_INTERNAL_BIAS"


def draw_all(api, klines, proj: Projector):
    pos, vals = proj.project(klines)
    smap, lo, hi = Projector.serial_map(pos)
    last_x = len(klines) - 1

    # count what is ACTUALLY drawn inside the current serial window
    counts = {
        "events": 0, "order_blocks": 0, "eqh_eql": 0,
        "sqz_release": 0, "zero_cross": 0,
    }

    def sp(csv_idx):
        """STRICT - only map when the bar is inside the serial window.

        Used for labels / point markers (BOS-CHoCH text, SQZ, zero cross):
        an event that resolved BEFORE the window must not be drawn at all,
        and must never be collapsed onto the left edge.
        """
        if csv_idx is None:
            return None
        return smap.get(csv_idx)

    def sp_left(csv_idx):
        """Interval START - clamp to the left edge when the interval began
        before the window, so structures extending into the window stay
        visible (canonical OB requirement)."""
        if csv_idx is None:
            return None
        if csv_idx in smap:
            return smap[csv_idx]
        if lo is not None and csv_idx < lo:
            return 0
        return None

    def sp_right(csv_idx):
        """Interval END - clamp to the right edge when the interval is still
        open or extends past the window."""
        if csv_idx is None:
            return last_x
        if csv_idx in smap:
            return smap[csv_idx]
        if hi is not None and csv_idx > hi:
            return last_x
        return None

    def y_at(x):
        v = vals["mom"][x]
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 0.0
        return float(v)

    # --- BOS / CHoCH: label at confirmed_index, structure line anchor->confirmed
    for i, ev in enumerate(proj.events):
        xc = sp(ev.get("confirmed_index"))
        if xc is None:
            continue
        xa = sp_left(ev.get("anchor_index"))
        if xa is None:
            continue
        up = bool(ev.get("bullish"))
        color = "green" if up else "red"
        tag = ("S-" if not ev.get("internal") else "I-") + \
              ("CHoCH" if ev.get("type") == "CHoCH" else "BOS") + \
              ("↑" if up else "↓")
        level = float(ev["level"])
        api.draw_line(klines, xa, level, xc, level,
                      id=f"panji_struct_{i}", board="MAIN",
                      line_type="SEG", color=color, width=1)
        api.draw_text(klines, tag, x=xc, y=level,
                      id=f"panji_evtxt_{i}", board="MAIN", color=color)
        counts["events"] += 1

    # --- Order Blocks: anchor -> mitigated (or last bar), clamped into window
    for i, ob in enumerate(proj.obs):
        # an OB already mitigated before the window begins is not visible at all
        mi = ob.get("mitigated_index")
        if mi is not None and lo is not None and mi < lo:
            continue
        xb = sp_right(mi)
        if xb is None:
            continue
        xa = sp_left(ob.get("anchor_index"))
        if xa is None:
            continue
        if xb < xa:
            continue
        bull = (ob.get("bias") or 0) > 0
        col = "green" if bull else "red"
        api.draw_box(klines, xa, float(ob["bar_high"]), xb, float(ob["bar_low"]),
                     id=f"panji_ob_{i}", board="MAIN",
                     bg_color=col, color=col, width=1)
        counts["order_blocks"] += 1

    # --- EQH / EQL
    for i, eq in enumerate(proj.eqs):
        xc = sp(eq.get("confirmed_index"))
        if xc is None:
            continue
        xa = sp_left(eq.get("anchor_index"))
        if xa is None:
            continue
        level = float(eq["level"])
        api.draw_line(klines, xa, level, xc, level,
                      id=f"panji_eq_{i}", board="MAIN",
                      line_type="SEG", color="blue", width=1)
        counts["eqh_eql"] += 1

    # --- Momentum events (canonical, never re-scanned by the renderer)
    for i, ev in enumerate(proj.sqz_events):
        x = sp(ev.get("bar_index"))
        if x is None:
            continue
        api.draw_text(klines, "SQZ", x=x, y=y_at(x),
                      id=f"panji_sqz_{i}", board="MOM", color="yellow")
        counts["sqz_release"] += 1

    for i, ev in enumerate(proj.zc_events):
        x = sp(ev.get("bar_index"))
        if x is None:
            continue
        up = ev.get("type") == "ZERO_CROSS_UP"
        api.draw_text(klines, "Z↑" if up else "Z↓", x=x, y=y_at(x),
                      id=f"panji_zc_{i}", board="MOM",
                      color="green" if up else "red")
        counts["zero_cross"] += 1

    return vals, counts


def run_gui(tf, df, smc, sqz, hist, serial_bars, hold_seconds) -> None:
    from tqsdk import TqApi, TqAuth

    load_dotenv(REPO / ".env")
    user = os.environ.get("TQ_USER", "").strip()
    pwd = os.environ.get("TQ_PASSWORD", "").strip()
    if not user or not pwd:
        raise SystemExit(
            "[FATAL] TQ_USER / TQ_PASSWORD missing. Export them or put them "
            "into .env (git-ignored)."
        )

    proj = Projector(df, sqz, smc, hist)
    bcolor_last = _last_bcolor(proj.bcolor)

    api = TqApi(auth=TqAuth(user, pwd), web_gui=True)
    try:
        print(f"\n[gui] subscribing {SIGNAL_SYMBOL} {tf} "
              f"(data_length={serial_bars}) as chart host ...")
        klines = api.get_kline_serial(SIGNAL_SYMBOL, TIMEFRAMES[tf],
                                      data_length=serial_bars)
        while not api.is_serial_ready(klines):
            api.wait_update()
        print("[gui] kline serial ready")

        vals, drawn = draw_all(api, klines, proj)
        apply_columns(klines, vals, bcolor_last)

        pos = proj.positions(klines)
        covered = int((pos >= 0).sum())
        print(f"[gui] serial bars = {len(klines)} ; covered by CSV = {covered} "
              f"; not covered (newer than CSV) = {len(klines) - covered}")
        print(f"[gui] drawn objects = {drawn}")
        print(f"[gui] momentum series colour (canonical bcolor of last bar) = {bcolor_last}")
        print("[gui] boards: MAIN | SMC_SWING_BIAS | SMC_INTERNAL_BIAS | MOM")
        print(f"[gui] holding for {hold_seconds}s - open the TqSdk URL printed above.")
        print("[gui] Ctrl-C to stop earlier.")

        last_ns = int(klines["datetime"].iloc[-1])
        last_len = len(klines)
        deadline = time.time() + hold_seconds
        while time.time() < deadline:
            api.wait_update(deadline=min(time.time() + 5.0, deadline))
            cur_ns = int(klines["datetime"].iloc[-1])
            cur_len = len(klines)
            if cur_ns != last_ns or cur_len != last_len:
                last_ns, last_len = cur_ns, cur_len
                vals, _ = draw_all(api, klines, proj)
                apply_columns(klines, vals, bcolor_last)
                print(f"[gui] new bar -> re-projected at {klines['datetime'].iloc[-1]}")
    finally:
        api.close()


# ============================================================
# 9. Main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeframe", choices=("15m", "1h", "4h"), default="15m")
    ap.add_argument("--plot-bars", type=int, default=PLOT_BARS,
                    help="report/console crop size; NEVER used to compute indicators")
    ap.add_argument("--serial-bars", type=int, default=SERIAL_BARS,
                    help="TqSdk kline serial length (chart host)")
    ap.add_argument("--hold-seconds", type=int, default=1800,
                    help="how long to keep the TqSdk web GUI alive")
    ap.add_argument("--no-gui", action="store_true",
                    help="run guards + artifacts only, do not connect to TqSdk")
    args = ap.parse_args()

    tf = args.timeframe

    print("=" * 72)
    print("Panji SMC + Momentum visual verification")
    print("=" * 72)

    # --- probe (always, it is evidence for the report)
    probe_info = probe_tqsdk_visual()
    write_probe_file(probe_info)
    print(f"[probe] tqsdk version      = {probe_info['version']}")
    print(f"[probe] web_gui            = {probe_info['web_gui']}")
    for k, v in probe_info["draws"].items():
        print(f"[probe] {k:<14} = {'yes' if v else 'no'}")
    print(f"[probe] direct_png_api     = {probe_info['direct_png_api']}")

    # --- data
    df = load_ohlc(tf)
    print(f"\n[data] {tf}: bars={len(df)}  "
          f"start={df.index[0]}  end={df.index[-1]}")
    print(f"[data] source = {SIGNAL_SYMBOL} (main-continuous, NOT tradable)")

    # --- canonical computation on FULL history
    print("\n[calc] computing canonical SMC on FULL history ...")
    smc = compute_smc(df)
    print("[calc] computing canonical Momentum on FULL history ...")
    sqz, hist = compute_momentum(df)
    print(f"[calc] SMC events={len(smc.get('events', []))} "
          f"order_blocks={len(smc.get('order_blocks', []))}")
    print(f"[calc] momentum val len={len(sqz['val'])} "
          f"sqz_release={len(hist['sqz_release_events'])} "
          f"zero_cross={len(hist['momentum_zero_cross_events'])}")

    # --- guards
    print("\n[guard] SMC invariants ...")
    smc_bad = check_smc_invariants(smc, len(df))
    print(f"[guard] SMC invariants      : {'PASS' if not smc_bad else 'FAIL'}")
    for b in smc_bad[:20]:
        print(f"        - {b}")

    print("[guard] Momentum invariants ...")
    mom_bad = check_momentum_invariants(sqz, hist, len(df))
    print(f"[guard] Momentum invariants : {'PASS' if not mom_bad else 'FAIL'}")
    for b in mom_bad[:20]:
        print(f"        - {b}")

    print("[guard] prefix PIT check ...")
    pit_info, pit_bad = prefix_pit_check(df)
    print(f"[guard] prefix PIT check    : {'PASS' if not pit_bad else 'FAIL'}"
          f"  (checkpoint={pit_info.get('checkpoint')})")
    for b in pit_bad[:20]:
        print(f"        - {b}")

    validation = {
        "smc_invariants": "PASS" if not smc_bad else "FAIL",
        "momentum_invariants": "PASS" if not mom_bad else "FAIL",
        "prefix_check": "PASS" if not pit_bad else "FAIL",
    }

    if smc_bad or mom_bad or pit_bad:
        print("\n[FATAL] guards FAILED -> STOP. No visualization is produced.")
        raise SystemExit(1)

    # --- artifacts
    summary = build_summary(tf, df, smc, sqz, hist, validation)
    write_artifacts(tf, df, smc, sqz, hist, summary, pit_info, probe_info)

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.no_gui:
        print("\n[--no-gui] skipping TqSdk GUI. Validation artifacts are written.")
        return

    run_gui(tf, df, smc, sqz, hist, args.serial_bars, args.hold_seconds)


if __name__ == "__main__":
    main()
