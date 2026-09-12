"""V3-A1 -- Execution Oracle / Regret Map

唯一目标（value-of-information audit）：
  在 frozen 四动作集合 [SKIP, MARKET, LIMIT_RR3, REASSESS_RR3] 下，
  完美事后动作选择相对固定 RR3 Strict Reassess 还有多少理论收益空间？

  H = EV_oracle - EV_reassess >= 0  （Reassess 本身是候选动作之一）

Governance（HARD）：
  不训练模型、不选特征、不调参、不做 time decay、不改 execution semantics、
  不改 action set、不做 symbol/region 调优、不读 P1。
  Primary reward = frozen censor-worst R。

Authoritative base: a33d4eba5dbe1a476d1b69b530547a04aa277b74
Current verdict: STOP_CURRENT_MULTI_ACTION_ML
"""
from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.liquidity_oracle_atlas.run_v2_ml_recency_multi_action_v1 import (
    OUT_DIR as V2_OUT_DIR,
    build_multi_action_dataset,
)
from research.liquidity_oracle_atlas.run_fixed_execution_baseline_v1 import load_env

# ---------------------------------------------------------------------------
# Frozen definitions
# ---------------------------------------------------------------------------
ACTIONS = ["SKIP", "MARKET", "LIMIT_RR3", "REASSESS_RR3"]
REWARD_COLS = {
    "SKIP": "reward_SKIP",
    "MARKET": "reward_MARKET",
    "LIMIT_RR3": "reward_LIMIT_RR3",
    "REASSESS_RR3": "reward_REASSESS_RR3",
}
RLOWER_COLS = {
    "SKIP": "R_lower_SKIP",
    "MARKET": "R_lower_MARKET",
    "LIMIT_RR3": "R_lower_LIMIT_RR3",
    "REASSESS_RR3": "R_lower_REASSESS_RR3",
}

TOL = 1e-12
MARGINS = [0.00, 0.05, 0.10, 0.25, 0.50, 1.00]
BOOTSTRAP_N = 2000
SEED = 20260912
P1_CUTOFF = pd.Timestamp("2026-09-04 14:55:00")

WF_LIST = ["WF1", "WF2", "WF3"]
WF_N = {"WF1": 3160, "WF2": 2847, "WF3": 3008}
N_TEST = 9015

FROZEN_EV = {
    "MARKET": {"WF1": 0.010376171415, "WF2": 0.033117784921, "WF3": 0.043795014084},
    "LIMIT_RR3": {"WF1": -0.005963576795, "WF2": -0.016045940367,
                  "WF3": 0.058922314077},
    "REASSESS_RR3": {"WF1": 0.034226296623, "WF2": 0.044368531007,
                     "WF3": 0.104135080035},
}
PARITY_ATOL = 1e-9

OUT = REPO_ROOT / "research/analysis_results/v3_execution_oracle_regret_v1"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = V2_OUT_DIR / "multi_action_signals_features.parquet"
TRADES = (REPO_ROOT / "research/analysis_results/execution_frontier_v1"
          / "execution_lag1_trades.parquet")


# ---------------------------------------------------------------------------
# Oracle construction
# ---------------------------------------------------------------------------
def build_oracle_rows(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    reward_matrix = np.column_stack(
        [x[REWARD_COLS[a]].to_numpy(np.float64) for a in ACTIONS])
    assert np.isfinite(reward_matrix).all(), "NON_FINITE_REWARD_MATRIX"

    oracle = reward_matrix.max(axis=1)
    is_best = np.isclose(reward_matrix, oracle[:, None], rtol=0.0, atol=TOL)
    n_best = is_best.sum(axis=1)

    skip, market, limit3, reassess = (
        reward_matrix[:, 0], reward_matrix[:, 1],
        reward_matrix[:, 2], reward_matrix[:, 3])

    x["oracle_reward"] = oracle
    x["oracle_headroom_vs_reassess"] = oracle - reassess
    x["delta_market_vs_reassess"] = market - reassess
    x["delta_limit_vs_reassess"] = limit3 - reassess
    x["delta_skip_vs_reassess"] = skip - reassess

    x["oracle_n_best"] = n_best
    x["oracle_unique"] = n_best == 1
    x["oracle_tied"] = n_best > 1
    for j, action in enumerate(ACTIONS):
        x[f"oracle_contains_{action}"] = is_best[:, j]
        x[f"oracle_unique_{action}"] = is_best[:, j] & (n_best == 1)

    alt = np.column_stack([skip - reassess, market - reassess,
                           limit3 - reassess])
    alt_names = np.array(["SKIP", "MARKET", "LIMIT_RR3"])
    best_alt_idx = np.argmax(alt, axis=1)
    x["best_alt_action"] = alt_names[best_alt_idx]
    x["best_alt_advantage"] = alt[np.arange(len(x)), best_alt_idx]

    # Secondary (mechanism reference only): R_lower oracle. NOT primary.
    rlower_matrix = np.column_stack(
        [x[RLOWER_COLS[a]].to_numpy(np.float64) for a in ACTIONS])
    x["oracle_R_lower"] = rlower_matrix.max(axis=1)

    assert (x["oracle_headroom_vs_reassess"] >= -TOL).all(), \
        "ORACLE_HEADROOM_NEGATIVE"
    return x


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------
def summarize_wf(g: pd.DataFrame) -> dict:
    n = len(g)
    reassess = g["reward_REASSESS_RR3"].to_numpy(float)
    oracle = g["oracle_reward"].to_numpy(float)
    headroom = oracle - reassess
    ev_reassess = float(np.mean(reassess))
    ev_oracle = float(np.mean(oracle))
    out = {
        "n_signals": n,
        "EV_market": float(g["reward_MARKET"].mean()),
        "EV_limit": float(g["reward_LIMIT_RR3"].mean()),
        "EV_reassess": ev_reassess,
        "EV_oracle": ev_oracle,
        "oracle_headroom_per_signal": float(np.mean(headroom)),
        "oracle_total_extra_R": float(np.sum(headroom)),
        "reassess_capture_ratio": (ev_reassess / ev_oracle
                                   if ev_oracle > 0 else np.nan),
        "share_headroom_positive": float(np.mean(headroom > TOL)),
        "median_headroom": float(np.median(headroom)),
        "p90_headroom": float(np.quantile(headroom, 0.90)),
        "p95_headroom": float(np.quantile(headroom, 0.95)),
        "oracle_tie_rate": float(g["oracle_tied"].mean()),
        # secondary R_lower view
        "EV_oracle_R_lower": float(g["oracle_R_lower"].mean()),
        "headroom_R_lower_vs_reassess": float(
            (g["oracle_R_lower"] - g["R_lower_REASSESS_RR3"]).mean()),
    }
    return out


def action_membership(g: pd.DataFrame) -> dict:
    out = {"n_signals": len(g),
           "oracle_tie_rate": float(g["oracle_tied"].mean()),
           "oracle_unique_rate": float(g["oracle_unique"].mean())}
    for action in ACTIONS:
        out[f"oracle_contains_{action}_share"] = float(
            g[f"oracle_contains_{action}"].mean())
        out[f"oracle_unique_{action}_share"] = float(
            g[f"oracle_unique_{action}"].mean())
    for k, v in g["oracle_n_best"].value_counts().sort_index().items():
        out[f"n_best_eq_{int(k)}_share"] = float(v / len(g))
    return out


def margin_distribution(g: pd.DataFrame) -> pd.DataFrame:
    adv = g["best_alt_advantage"].to_numpy(float)
    return pd.DataFrame([
        {"margin_R": m,
         "share_best_alt_gt_margin": float(np.mean(adv > m + TOL)),
         "n_best_alt_gt_margin": int(np.sum(adv > m + TOL))}
        for m in MARGINS])


def advantage_summary(g: pd.DataFrame) -> pd.DataFrame:
    mapping = {"MARKET": "delta_market_vs_reassess",
               "LIMIT_RR3": "delta_limit_vs_reassess",
               "SKIP": "delta_skip_vs_reassess"}
    rows = []
    for action, col in mapping.items():
        z = g[col].to_numpy(float)
        rows.append({
            "action": action,
            "mean_delta": float(np.mean(z)),
            "median_delta": float(np.median(z)),
            "share_delta_gt_0": float(np.mean(z > TOL)),
            "share_delta_gt_0p05": float(np.mean(z > 0.05)),
            "share_delta_gt_0p10": float(np.mean(z > 0.10)),
            "share_delta_gt_0p25": float(np.mean(z > 0.25)),
            "share_delta_gt_0p50": float(np.mean(z > 0.50)),
        })
    return pd.DataFrame(rows)


def bootstrap_headroom(reassess, oracle, n_boot=BOOTSTRAP_N, seed=SEED):
    reassess = np.asarray(reassess, dtype=np.float64)
    oracle = np.asarray(oracle, dtype=np.float64)
    delta = oracle - reassess
    rng = np.random.default_rng(seed)
    n = len(delta)
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        vals[i] = delta[idx].mean()
    return {"mean": float(delta.mean()),
            "median_bootstrap": float(np.median(vals)),
            "ci_2p5": float(np.quantile(vals, 0.025)),
            "ci_97p5": float(np.quantile(vals, 0.975))}


def group_diag(g: pd.DataFrame, key: str) -> pd.DataFrame:
    rows = []
    for k, sub in g.groupby(key, sort=True):
        r = sub["reward_REASSESS_RR3"].to_numpy(float)
        o = sub["oracle_reward"].to_numpy(float)
        d = {"group": str(k), "n": len(sub),
             "EV_reassess": float(r.mean()), "EV_oracle": float(o.mean()),
             "headroom": float((o - r).mean()),
             "share_headroom_positive": float(np.mean((o - r) > TOL))}
        for action in ACTIONS:
            d[f"oracle_contains_{action}_share"] = float(
                sub[f"oracle_contains_{action}"].mean())
        rows.append(d)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
def _fmt(df):
    return "```\n" + df.to_string(index=False) + "\n```\n"


def main():
    t0 = time.perf_counter()
    print("=" * 70)
    print("V3-A1 Execution Oracle / Regret Map (no model training)")
    print("=" * 70)

    # Force the hardened dataset construction (and its mandatory parity
    # bridges) to re-run: the cached parquet would bypass them.
    if CACHE.exists():
        CACHE.unlink()
        print(f"[CACHE] removed stale cache to force hardened rebuild: {CACHE.name}")

    D, master_by_sym, bars_by_sym = load_env()
    trades = pd.read_parquet(TRADES)
    df_comb, _feature_blocks = build_multi_action_dataset(
        D, master_by_sym, bars_by_sym, trades)
    print(f"[DATA] df_comb rows={len(df_comb)} ({time.perf_counter()-t0:.1f}s)")

    df = df_comb[df_comb["wf"].isin(WF_LIST)].copy().reset_index(drop=True)

    # ---------------- T1-T12 executable verification ----------------
    t1 = bool(len(df) == N_TEST
              and all(int((df.wf == w).sum()) == WF_N[w] for w in WF_LIST))
    t2 = bool((df["reward_SKIP"] == 0.0).all())

    parity = {}
    for action, col in [("MARKET", "reward_MARKET"),
                        ("LIMIT_RR3", "reward_LIMIT_RR3"),
                        ("REASSESS_RR3", "reward_REASSESS_RR3")]:
        parity[action] = {}
        for w in WF_LIST:
            actual = float(df.loc[df.wf == w, col].mean())
            exp = FROZEN_EV[action][w]
            parity[action][w] = dict(actual=actual, expected=exp,
                                     abs_diff=abs(actual - exp))
    t3 = all(v["abs_diff"] < PARITY_ATOL for v in parity["MARKET"].values())
    t4 = all(v["abs_diff"] < PARITY_ATOL for v in parity["LIMIT_RR3"].values())
    t5 = all(v["abs_diff"] < PARITY_ATOL
             for v in parity["REASSESS_RR3"].values())
    if not (t1 and t2 and t3 and t4 and t5):
        raise SystemExit(f"STOP_V3_ORACLE_BASELINE_PARITY_FAIL: "
                         f"t1={t1} t2={t2} t3={t3} t4={t4} t5={t5} parity={parity}")

    oracle = build_oracle_rows(df)
    rm = np.column_stack([oracle[REWARD_COLS[a]].to_numpy(float)
                          for a in ACTIONS])
    orc = oracle["oracle_reward"].to_numpy(float)
    t6 = bool((orc[:, None] >= rm - TOL).all())
    t7 = bool((oracle["oracle_headroom_vs_reassess"] >= -TOL).all())
    t8 = bool((oracle["oracle_n_best"] >= 1).all())
    t9 = bool(np.isclose(oracle["oracle_unique"].astype(float)
                         + oracle["oracle_tied"].astype(float), 1.0).all())
    cont = np.column_stack([oracle[f"oracle_contains_{a}"].to_numpy(float)
                            for a in ACTIONS]).sum(axis=1)
    t10 = bool((cont == oracle["oracle_n_best"].to_numpy(float)).all())
    entry_max = pd.to_datetime(oracle["entry_time"]).max()
    src = Path(__file__).read_text()
    # token assembled at runtime so this check's own source cannot self-trip it
    p1_ref = "prospective_selection" + "_p1"
    t11 = bool(entry_max <= P1_CUTOFF and p1_ref not in src)
    # T12: no sklearn import / no estimator instantiation in THIS runner
    bad = []
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            bad += [a.name for a in node.names
                    if a.name.split(".")[0] == "sklearn"]
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] == "sklearn":
                bad.append(node.module)
        elif isinstance(node, ast.Call):
            fn = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            if fn in {"LogisticRegression", "Ridge", "HistGradientBoostingRegressor",
                      "HistGradientBoostingClassifier"}:
                bad.append(fn)
    t12 = len(bad) == 0

    tests = [
        dict(id="T1", desc="signal count 9015 / WF counts exact", passed=t1),
        dict(id="T2", desc="SKIP reward all zero", passed=t2),
        dict(id="T3", desc="Market frozen parity", passed=t3),
        dict(id="T4", desc="Single Limit frozen parity", passed=t4),
        dict(id="T5", desc="Reassess frozen parity", passed=t5),
        dict(id="T6", desc="oracle >= every candidate reward", passed=t6),
        dict(id="T7", desc="oracle headroom >= 0", passed=t7),
        dict(id="T8", desc="oracle_n_best >= 1", passed=t8),
        dict(id="T9", desc="unique XOR tied covers 100%", passed=t9),
        dict(id="T10", desc="n_best == sum(contains)", passed=t10),
        dict(id="T11", desc="P1 not read (entry_time <= P1 cutoff)", passed=t11),
        dict(id="T12", desc="no sklearn estimator / no fitted model",
             passed=t12),
    ]
    all_pass = all(t["passed"] for t in tests)
    (OUT / "v3_tests.json").write_text(json.dumps(dict(all_passed=all_pass,
                                                      tests=tests), indent=2))
    assert all_pass, f"STOP_V3_ORACLE_AUDIT_FAIL: {[t for t in tests if not t['passed']]}"
    print("[TESTS] T1-T12 all PASS")

    # ---------------- summaries ----------------
    summary = pd.DataFrame([dict(wf=w, **summarize_wf(oracle[oracle.wf == w]))
                            for w in WF_LIST])
    membership = pd.DataFrame([dict(wf=w, **action_membership(oracle[oracle.wf == w]))
                               for w in WF_LIST])
    margins = pd.concat([margin_distribution(oracle[oracle.wf == w]).assign(wf=w)
                         for w in WF_LIST], ignore_index=True)
    margins = margins[["wf", "margin_R", "share_best_alt_gt_margin",
                       "n_best_alt_gt_margin"]]
    adv = pd.concat([advantage_summary(oracle[oracle.wf == w]).assign(wf=w)
                     for w in WF_LIST], ignore_index=True)
    adv = adv[["wf", "action", "mean_delta", "median_delta", "share_delta_gt_0",
               "share_delta_gt_0p05", "share_delta_gt_0p10",
               "share_delta_gt_0p25", "share_delta_gt_0p50"]]
    region = group_diag(oracle, "region")
    sym_counts = oracle["symbol"].value_counts()
    sym_ok = sym_counts[sym_counts >= 100].index
    symbol = group_diag(oracle[oracle.symbol.isin(sym_ok)], "symbol")
    boot = pd.DataFrame([
        dict(wf=w, **bootstrap_headroom(
            oracle.loc[oracle.wf == w, "reward_REASSESS_RR3"],
            oracle.loc[oracle.wf == w, "oracle_reward"]))
        for w in WF_LIST])

    summary.to_csv(OUT / "oracle_summary_by_wf.csv", index=False)
    membership.to_csv(OUT / "oracle_action_membership_by_wf.csv", index=False)
    margins.to_csv(OUT / "oracle_advantage_margin_by_wf.csv", index=False)
    adv.to_csv(OUT / "oracle_action_advantage_by_wf.csv", index=False)
    region.to_csv(OUT / "oracle_region_diagnostic.csv", index=False)
    symbol.to_csv(OUT / "oracle_symbol_diagnostic.csv", index=False)
    boot.to_csv(OUT / "oracle_bootstrap.csv", index=False)

    keep_cols = (["gid", "wf", "symbol", "region", "oracle_reward"]
                 + list(REWARD_COLS.values())
                 + ["oracle_headroom_vs_reassess", "delta_market_vs_reassess",
                    "delta_limit_vs_reassess", "delta_skip_vs_reassess",
                    "best_alt_action", "best_alt_advantage", "oracle_n_best",
                    "oracle_unique", "oracle_tied"]
                 + [f"oracle_contains_{a}" for a in ACTIONS]
                 + [f"oracle_unique_{a}" for a in ACTIONS]
                 + ["oracle_R_lower"])
    oracle[keep_cols].to_parquet(OUT / "oracle_signal_level.parquet", index=False)

    # ---------------- verdict (declared rules, no tuned thresholds) -------
    hr = summary.set_index("wf")["oracle_headroom_per_signal"]
    sh10 = margins[margins.margin_R == 0.10].set_index("wf")["share_best_alt_gt_margin"]
    spread = float(hr.max() / hr.min()) if hr.min() > 0 else float("inf")
    if hr.min() > 0 and spread >= 2.0:
        verdict = "ORACLE_HEADROOM_INCONSISTENT"
    elif float(sh10.max()) < 0.10:
        verdict = "ORACLE_HEADROOM_LOW"
    else:
        verdict = "ORACLE_HEADROOM_MATERIAL"
    rules = {
        "ORACLE_HEADROOM_INCONSISTENT":
            "max_wf_headroom / min_wf_headroom >= 2.0 (WFs disagree in magnitude)",
        "ORACLE_HEADROOM_LOW":
            "max over WF of share(best_alt_advantage > 0.10R) < 0.10",
        "ORACLE_HEADROOM_MATERIAL": "otherwise (all WF headroom > 0 and material share >= 0.10)",
        "note": "thresholds declared a priori; no economic tuning performed",
        "headroom_spread_ratio": spread,
    }

    audit = dict(
        experiment="V3-A1 Execution Oracle / Regret Map",
        base_commit="a33d4eba5dbe1a476d1b69b530547a04aa277b74",
        current_verdict="STOP_CURRENT_MULTI_ACTION_ML",
        actions=ACTIONS, primary_reward="censor-worst R",
        n_test_signals=int(len(df)), wf_n=WF_N,
        frozen_parity=parity, tests=tests, all_tests_pass=all_pass,
        verdict=verdict, verdict_rules=rules,
        no_modeling=dict(sklearn_used=False, p1_read=False,
                         feature_selection=False, tuning=False),
        bootstrap=dict(n=BOOTSTRAP_N, seed=SEED))
    (OUT / "V3_EXECUTION_ORACLE_REGRET_AUDIT.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, default=str))

    # ---------------- report ----------------
    L = []
    A = L.append
    A("# V3-A1 — Execution Oracle / Regret Map\n")
    A("**No model is trained in this experiment.** 目的：测量在 frozen 四动作集合下，"
      "完美事后动作选择相对固定 RR3 Strict Reassess 还有多少理论收益空间。\n")
    A(f"- signals = `{len(df)}` (WF1/WF2/WF3 = {WF_N['WF1']}/{WF_N['WF2']}/{WF_N['WF3']})")
    A(f"- primary reward = frozen **censor-worst** R")
    A(f"- T1–T12 全部 PASS；P1_read = `False`；sklearn = `False`\n")

    A("## 1. Headroom（核心表）\n")
    A(_fmt(summary[["wf", "n_signals", "EV_reassess", "EV_oracle",
                    "oracle_headroom_per_signal", "oracle_total_extra_R",
                    "reassess_capture_ratio", "share_headroom_positive",
                    "median_headroom", "p90_headroom", "oracle_tie_rate"]]))
    A("`reassess_capture_ratio` = EV_reassess / EV_oracle：固定 Reassess 已经拿走"
      "多少 theoretical action-selection value。\n")

    A("## 2. Reassess 相对 Oracle 的差距\n")
    gap = summary[["wf", "EV_reassess", "EV_oracle"]].copy()
    gap["headroom"] = gap["EV_oracle"] - gap["EV_reassess"]
    gap["captured_pct"] = 100.0 * summary["reassess_capture_ratio"].to_numpy()
    A(_fmt(gap))

    A("## 3. Best-alternative advantage margin\n")
    A("`share(best_alt_advantage > margin)`：至少有一个 alternative 比 Reassess 好多少。\n")
    A(_fmt(margins))
    A("margin=0 即「至少有一个 alternative 严格优于 Reassess」的比例。\n")

    A("## 4. Per-action advantage vs Reassess\n")
    A(_fmt(adv))

    A("## 5. Oracle action membership（tie-aware）\n")
    A(_fmt(membership))
    A("`contains_*` = 属于最优动作集合；`unique_*` = 唯一严格最优。"
      "未做 tie-break。\n")

    A("## 6. Bootstrap（paired, Oracle − Reassess）\n")
    A(_fmt(boot))

    A("## 7. Region diagnostic（仅解释，不得据此造 region policy）\n")
    A(_fmt(region))

    A("## 8. Symbol diagnostic（n ≥ 100，仅解释）\n")
    A(_fmt(symbol))

    A("## 9. Secondary：R_lower oracle（机制参考，非 Primary）\n")
    A(_fmt(summary[["wf", "EV_reassess", "EV_oracle_R_lower",
                    "headroom_R_lower_vs_reassess"]]))

    A("## 10. Verdict\n")
    A(f"**{verdict}**\n")
    A(_fmt(pd.DataFrame([{"rule": k, "meaning": str(v)}
                         for k, v in rules.items()])))
    A("## 11. 结论要点\n")
    q = [
        f"Q1/Q2/Q3 每 WF Reassess / Oracle / Headroom：见 §1、§2。",
        f"Q4 Reassess 捕获比例："
        + "；".join(f"{r.wf}={100*r.reassess_capture_ratio:.1f}%"
                    for r in summary.itertuples()),
        f"Q5 alternative > Reassess 的比例（margin=0）："
        + "；".join(f"{r.wf}={r.share_best_alt_gt_margin:.3f}"
                    for r in margins[margins.margin_R == 0].itertuples()),
        f"Q6 > 0.05 / 0.10 / 0.25 / 0.50R：见 §3 表。",
        f"Q7 经常属于 oracle 的动作：见 §5。",
        f"Q8 unique winner 分布：见 §5 的 `oracle_unique_*_share`。",
        f"Q9 region / symbol：仅解释性异质性，禁止据此形成 policy。",
        f"Q10 是否值得继续 Relative Advantage ML：由 §10 verdict 决定。",
    ]
    for x in q:
        A(f"- {x}")
    A("\n**STOP**：不自动进入 Relative Advantage ML。\n")
    (OUT / "V3_EXECUTION_ORACLE_REGRET_V1.md").write_text("\n".join(L),
                                                          encoding="utf-8")

    print(f"\n[VERDICT] {verdict}")
    for w in WF_LIST:
        r = summary[summary.wf == w].iloc[0]
        print(f"  {w}: EV_reassess={r.EV_reassess:+.6f} EV_oracle={r.EV_oracle:+.6f} "
              f"headroom={r.oracle_headroom_per_signal:+.6f} "
              f"capture={r.reassess_capture_ratio:.4f} "
              f"pos_share={r.share_headroom_positive:.4f}")
    print(f"[DONE] {time.perf_counter()-t0:.1f}s -> {OUT}")


if __name__ == "__main__":
    main()
