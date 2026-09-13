"""LOCAL-0.1 — Local Geometry Attribution (closure check)

===========================================================================
为什么需要这一轮
===========================================================================
534de62e 的 B1_LOCAL_GEOMETRY 实际特征是

    geometry (4 个数值) + categorical symbol

而 B0_PRIOR 是 **全市场统一 prior**。因此

    B1 - B0

严格测到的是

    geometry increment  +  symbol base-rate increment

本轮把两者拆开，做 2x2 attribution：

    M0_GLOBAL_PRIOR      无特征
    M1_SYMBOL_ONLY       symbol
    M2_GEOMETRY_ONLY     geometry
    M3_GEOMETRY_SYMBOL   geometry + symbol   (= 534de62e 的 B1，参数不变)

PRIMARY   : M3 - M1  （控制 symbol base rate 后 geometry 是否仍有 OOS 增量）
SECONDARY : M2 - M0  （不用 symbol 时 geometry 是否独立有增量）

===========================================================================
冻结边界
===========================================================================
本轮 **不重建** corrected lifecycle / local pair / labels，
不改 discontinuity / censor 语义。直接复用 534de62e 生成的
gitignored sample cache。

    必须复现
        TB1 resolved before purge = 114898
        TB1 resolved after  purge = 114095
        TB2 resolved              = 124612
    否则 STOP_LOCAL01_SAMPLE_DRIFT

禁止使用 TB3 / TB4 / PnL / Sharpe / SMC / path / liquidity type /
event sequence / HMM / PGM / RL。

===========================================================================
输出
===========================================================================
    local01_model_attribution.csv
    local01_by_symbol_attribution.csv
    local01_bootstrap.csv
    local01_summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

# 复用 LOCAL-0 的冻结常量与度量（不重新实现，避免口径漂移）
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, UP, DOWN, RESOLVED_CODES, NUM_FEATURES, CAT_FEATURES,
    SEED, N_BOOT, TRAIN_BLOCK, TEST_BLOCK, OUT, CACHE,
    to_ns_int, binary_logloss, binary_brier, ece_binary, day_paired_bootstrap,
)

from sklearn.compose import ColumnTransformer  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder  # noqa: E402

# 534de62e 的硬基线（sample drift guard）
EXPECT = dict(tb1_resolved_before_purge=114898,
              tb1_resolved_after_purge=114095,
              tb2_resolved=124612)

# 四个冻结模型规格： (name, num_features, cat_features)
SPECS = [
    ("M0_GLOBAL_PRIOR", [], []),
    ("M1_SYMBOL_ONLY", [], list(CAT_FEATURES)),
    ("M2_GEOMETRY_ONLY", list(NUM_FEATURES), []),
    ("M3_GEOMETRY_SYMBOL", list(NUM_FEATURES), list(CAT_FEATURES)),
]
MODEL_NAMES = [s[0] for s in SPECS]


def make_pipeline(num, cat) -> Pipeline:
    """固定的 sklearn Pipeline；不允许调参。"""
    if num and cat:
        pre = ColumnTransformer([
            ("num", SimpleImputer(strategy="median"), num),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ])
    elif num:
        pre = SimpleImputer(strategy="median")
    elif cat:
        pre = OneHotEncoder(handle_unknown="ignore")
    else:
        raise ValueError("at least one of num / cat required")
    clf = LogisticRegression(penalty="l2", C=1.0, solver="lbfgs", max_iter=3000)
    return Pipeline([("pre", pre), ("clf", clf)])


def fit_spec(spec, train: pd.DataFrame):
    name, num, cat = spec
    if not num and not cat:
        p_up = float(np.mean(train["label"].to_numpy() == UP))
        return dict(name=name, kind="prior", p_up=p_up, num=num, cat=cat)
    cols = num + cat
    pipe = make_pipeline(num, cat)
    pipe.fit(train[cols], train["label"].to_numpy())
    classes = np.asarray(pipe.named_steps["clf"].classes_, dtype=int)
    assert np.array_equal(np.sort(classes), np.array([UP, DOWN])), \
        f"CLASSES_NOT_BINARY_UP_DOWN: {classes}"
    return dict(name=name, kind="pipe", pipe=pipe, num=num, cat=cat,
                cols=cols)


def proba_up(model, df: pd.DataFrame) -> np.ndarray:
    """返回 (n, 2)，列序固定 [P(UP), P(DOWN)]。"""
    if model["kind"] == "prior":
        p = float(model["p_up"])
        return np.tile(np.array([p, 1.0 - p], dtype=float), (len(df), 1))
    P = model["pipe"].predict_proba(df[model["cols"]])
    classes = np.asarray(model["pipe"].named_steps["clf"].classes_, dtype=int)
    out = np.zeros((len(df), 2), dtype=float)
    out[:, classes] = P
    return out


def metrics(P: np.ndarray, y: np.ndarray, name: str) -> dict:
    return dict(
        model=name,
        n=int(len(y)),
        logloss=binary_logloss(P, y),
        brier=binary_brier(P, y),
        ece_up=ece_binary(P[:, UP], (y == UP).astype(float)),
        ece_down=ece_binary(P[:, DOWN], (y == DOWN).astype(float)),
        up_pct=float(np.mean(y == UP)),
        down_pct=float(np.mean(y == DOWN)),
    )


def load_cached_samples(symbols):
    """只复用 cache；缺失即 STOP，不静默重建。"""
    frames = []
    missing = []
    for s in symbols:
        p = CACHE / f"local0_samples_{s}.parquet"
        if not p.exists():
            missing.append(str(p))
            continue
        frames.append(pd.read_parquet(p))
    if missing:
        raise SystemExit(
            "STOP_LOCAL01_CACHE_MISSING: sample cache not found, refusing to "
            "rebuild the frozen LOCAL-0 state. missing="
            + ";".join(missing[:5]))
    return pd.concat(frames, ignore_index=True)


def row_key_hash(df: pd.DataFrame) -> str:
    import hashlib
    key = (df["symbol"].astype(str) + "|"
           + df["decision_bar_index"].astype(str)).to_numpy()
    h = hashlib.sha256()
    h.update("\n".join(sorted(key.tolist())).encode())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="*", default=None)
    args = ap.parse_args()
    symbols = args.symbols or list(FULL_UNIV)

    t_total = time.perf_counter()
    timing = {}

    t0 = time.perf_counter()
    samples = load_cached_samples(symbols)
    timing["cache_load_seconds"] = round(time.perf_counter() - t0, 2)

    needed = {"symbol", "decision_bar_index", "decision_time", "block",
              "trading_day", "label", "resolution_time",
              "up_distance_R", "down_distance_R", "width_R",
              "log_distance_ratio"}
    lack = needed - set(samples.columns)
    if lack:
        raise SystemExit(f"STOP_LOCAL01_CACHE_SCHEMA: missing {sorted(lack)}")

    # ---------------- split / purge（与 534de62e 完全同口径） -------------
    tb2_start_ns = int(np.min(
        to_ns_int(samples.loc[samples["block"] == TEST_BLOCK,
                              "decision_time"])))
    train_all = samples[(samples["block"] == TRAIN_BLOCK)
                        & samples["label"].isin(RESOLVED_CODES)].copy()
    test = samples[(samples["block"] == TEST_BLOCK)
                   & samples["label"].isin(RESOLVED_CODES)].copy().reset_index(
        drop=True)

    n_before = int(len(train_all))
    keep = to_ns_int(train_all["resolution_time"]) < tb2_start_ns
    train = train_all[keep].copy().reset_index(drop=True)

    n_tb2 = int(len(test))
    got = dict(tb1_resolved_before_purge=n_before,
               tb1_resolved_after_purge=int(len(train)),
               tb2_resolved=n_tb2)
    print(f"[SAMPLE] {got}")
    if got != EXPECT:
        raise SystemExit(
            "STOP_LOCAL01_SAMPLE_DRIFT: cached sample counts differ from the "
            f"frozen 534de62e baseline {EXPECT} vs {got}")

    y_tr = train["label"].to_numpy()
    y_te = test["label"].to_numpy()

    # ---------------- fit + metrics ----------------
    t0 = time.perf_counter()
    models = {}
    probas = {}
    rows = []
    for spec in SPECS:
        mdl = fit_spec(spec, train)
        models[mdl["name"]] = mdl
        P = proba_up(mdl, test)
        probas[mdl["name"]] = P
        rows.append(metrics(P, y_te, mdl["name"]))
    timing["fit_seconds"] = round(time.perf_counter() - t0, 2)

    att = pd.DataFrame(rows)
    att.to_csv(OUT / "local01_model_attribution.csv", index=False)
    print("[ATTRIBUTION]")
    print(att.to_string(index=False))

    lls = {k: -np.log(np.maximum(probas[k][np.arange(n_tb2), y_te], 1e-12))
           for k in MODEL_NAMES}

    # ---------------- bootstraps ----------------
    t0 = time.perf_counter()
    days = test["trading_day"].to_numpy()
    pairs = [
        ("PRIMARY", "M3_GEOMETRY_SYMBOL", "M1_SYMBOL_ONLY"),
        ("SECONDARY", "M2_GEOMETRY_ONLY", "M0_GLOBAL_PRIOR"),
        ("DIAGNOSTIC", "M1_SYMBOL_ONLY", "M0_GLOBAL_PRIOR"),
        ("DIAGNOSTIC", "M3_GEOMETRY_SYMBOL", "M2_GEOMETRY_ONLY"),
    ]
    boots = []
    for role, hi, lo in pairs:
        b = day_paired_bootstrap(days, lls[lo], lls[hi], seed=SEED,
                                 n_boot=N_BOOT)
        b["role"] = role
        b["model_a"] = lo
        b["model_b"] = hi
        b["delta"] = f"{hi} - {lo}"
        boots.append(b)
        print(f"[BOOT] {role} {hi} - {lo}: mean={b['mean_delta_logloss']:+.6f} "
              f"CI=[{b['ci_lo']:+.6f},{b['ci_hi']:+.6f}]")
    timing["bootstrap_seconds"] = round(time.perf_counter() - t0, 2)

    bt = pd.DataFrame(boots)[
        ["role", "delta", "model_a", "model_b", "n_days", "n_boot", "seed",
         "mean_delta_logloss", "ci_lo", "ci_hi", "verdict"]]
    bt.to_csv(OUT / "local01_bootstrap.csv", index=False)

    primary = boots[0]
    if primary["ci_hi"] < 0:
        primary_verdict = "LOCAL_GEOMETRY_CONDITIONAL_INCREMENT_SUPPORTED"
        closed = True
    elif primary["ci_lo"] > 0:
        primary_verdict = "LOCAL_GEOMETRY_CONDITIONAL_INCREMENT_NEGATIVE"
        closed = False
    else:
        primary_verdict = "LOCAL_GEOMETRY_CONDITIONAL_INCREMENT_AMBIGUOUS"
        closed = False
    print(f"[PRIMARY] {primary_verdict}  LOCAL0_CLOSED={closed}")

    # ---------------- by-symbol robustness ----------------
    bs_rows = []
    for s, g in test.groupby("symbol"):
        idx = g.index.to_numpy()
        y = y_te[idx]
        l1 = float(np.mean(-np.log(np.maximum(
            probas["M1_SYMBOL_ONLY"][idx, y], 1e-12))))
        l3 = float(np.mean(-np.log(np.maximum(
            probas["M3_GEOMETRY_SYMBOL"][idx, y], 1e-12))))
        bs_rows.append(dict(symbol=s, n=int(len(g)),
                            ll_M1_SYMBOL_ONLY=l1,
                            ll_M3_GEOMETRY_SYMBOL=l3,
                            delta_geometry_given_symbol=l3 - l1))
    bs = pd.DataFrame(bs_rows)
    n_neg = int((bs["delta_geometry_given_symbol"] < 0).sum())
    n_pos = int((bs["delta_geometry_given_symbol"] > 0).sum())
    print(f"[BY-SYMBOL] delta<0: {n_neg}/15   delta>0: {n_pos}/15")
    bs.to_csv(OUT / "local01_by_symbol_attribution.csv", index=False)

    timing["total_seconds"] = round(time.perf_counter() - t_total, 2)

    # ---------------- summary ----------------
    summary = dict(
        experiment="LOCAL-0.1 local geometry attribution closure",
        base_commit="534de62e0182bf6da9410d6cee2375b4393d57b2",
        question=("Does the frozen B1-B0 improvement come from geometry, or "
                  "from symbol base-rate differences?"),
        data=dict(source="gitignored LOCAL-0 sample cache (no rebuild)",
                  sample_counts=got, expected=EXPECT,
                  tb2_start_time=str(pd.Timestamp(tb2_start_ns, unit="ns")),
                  train_row_key_sha256=row_key_hash(train),
                  test_row_key_sha256=row_key_hash(test)),
        model_specs={n: dict(num=list(nm), cat=list(ct))
                     for n, nm, ct in SPECS},
        metrics={r["model"]: r for r in rows},
        bootstrap={b["delta"]: b for b in boots},
        primary=dict(delta="M3_GEOMETRY_SYMBOL - M1_SYMBOL_ONLY",
                     mean_delta_logloss=primary["mean_delta_logloss"],
                     ci_lo=primary["ci_lo"], ci_hi=primary["ci_hi"],
                     verdict=primary_verdict),
        secondary=dict(delta="M2_GEOMETRY_ONLY - M0_GLOBAL_PRIOR",
                       mean_delta_logloss=boots[1]["mean_delta_logloss"],
                       ci_lo=boots[1]["ci_lo"], ci_hi=boots[1]["ci_hi"]),
        by_symbol=dict(n_symbols=int(len(bs)), n_delta_negative=n_neg,
                       n_delta_positive=n_pos,
                       min=float(bs["delta_geometry_given_symbol"].min()),
                       max=float(bs["delta_geometry_given_symbol"].max())),
        calibration_note=(
            "geometry improves proper scoring rules (logloss/Brier) but has "
            "NOT been shown to improve calibration: on TB2 resolved, "
            "ECE_UP M0=%.6f, M3=%.6f. These probabilities must not be called "
            "a calibrated belief state." % (
                rows[0]["ece_up"], rows[3]["ece_up"])),
        interpretation=(
            "Conditional on symbol-specific base-rate information, local "
            "geometry of the nearest frozen upper/lower liquidity boundary "
            "carries stable TB1->TB2 OOS probability information about which "
            "frozen boundary is penetrated first."
            if closed else
            "Primary M3-M1 CI crosses 0: previous B1-B0 result may contain a "
            "material symbol base-rate contribution."),
        forbidden_claims=["stable across all regimes", "long-term stable",
                          "SMC works", "path memory proven",
                          "belief calibrated", "PGM/RL validated"],
        local0_closed=bool(closed),
        verdict=primary_verdict,
        timing=timing,
    )
    (OUT / "local01_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))

    # 在已冻结的 LOCAL-0 summary 中只追加 closure reference
    p0 = OUT / "local0_summary.json"
    if p0.exists():
        d0 = json.loads(p0.read_text())
        d0["attribution_closure"] = dict(
            local01_summary="local01_summary.json",
            primary=summary["primary"],
            local0_closed=bool(closed),
            verdict=primary_verdict,
        )
        p0.write_text(json.dumps(d0, indent=2, default=str))

    print(json.dumps({k: summary[k] for k in
                      ("primary", "secondary", "by_symbol", "timing",
                       "local0_closed", "verdict")},
                     indent=2, default=str))
    print(f"[DONE] {timing['total_seconds']}s -> {OUT}")


if __name__ == "__main__":
    main()
