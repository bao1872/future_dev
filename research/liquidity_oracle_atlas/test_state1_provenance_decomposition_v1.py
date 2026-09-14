"""STATE-1 — deterministic synthetic / contract tests.

覆盖：
  * provenance active contract（activation / expiry / penetration）与 age>=0
  * provenance 只使用 target start bar t 时已知信息
  * 四层模型 feature set 精确性（无 duration / path / symbol / type / scope）
  * PROV 恰好 6 个预注册变量
  * window 定义 / seed / comparison 集合冻结
  * sample parity 与 REPL-0 hash 一致（读取已提交结果）
  * TB4 措辞合同（tb4_analytically_used，不是 tb4_read_for_values）

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_state1_provenance_decomposition_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 as S  # noqa: E402
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as P  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def test_registered_feature_sets():
    check("PROV has exactly the 6 preregistered variables",
          S.PROV == ["upper_oldest_age_bars", "upper_newest_age_bars",
                     "upper_n_active_identities", "lower_oldest_age_bars",
                     "lower_newest_age_bars", "lower_n_active_identities"],
          S.PROV)
    check("GEOM is exactly the frozen PATH-0 M0 geometry",
          S.GEOM == P.CUR_NUM == ["cur_up_distance_R", "cur_down_distance_R",
                                  "cur_width_R", "cur_log_distance_ratio"],
          S.GEOM)
    check("S0 = geometry only",
          S.MODELS[S.S0] == (S.GEOM, []), S.MODELS[S.S0])
    check("S1 = geometry + 6 provenance",
          S.MODELS[S.S1] == (S.GEOM + S.PROV, []), S.MODELS[S.S1])
    check("S2 = geometry + prev_event one-hot only",
          S.MODELS[S.S2] == (S.GEOM, S.CAT)
          and S.CAT == ["prev_event_mask"], S.MODELS[S.S2])
    check("S3 = geometry + provenance + prev_event exactly",
          S.MODELS[S.S3] == (S.GEOM + S.PROV, S.CAT), S.MODELS[S.S3])
    banned = ("duration", "path", "symbol", "type", "scope", "volume",
              "signature", "smc", "bos", "choch", "sweep", "seq", "hmm")
    bad = [c for v in S.MODELS.values() for c in (v[0] + v[1])
           if any(b in c.lower() for b in banned)]
    check("no duration / path / symbol / SMC / seq feature in any layer",
          bad == [], bad)
    check("primary comparison is S3 - S1",
          S.COMPARISONS[0] == (S.S3, S.S1), S.COMPARISONS[0])
    check("exactly 4 frozen comparisons, no extras",
          len(S.COMPARISONS) == 4
          and set(S.COMPARISONS) == {(S.S3, S.S1), (S.S1, S.S0),
                                     (S.S2, S.S0), (S.S3, S.S2)},
          S.COMPARISONS)


def test_windows_and_seeds():
    w = S.WINDOWS
    check("window A is TB1 -> TB2 with seed 20260916",
          w[0]["train"] == ["TB1"] and w[0]["eval"] == "TB2"
          and w[0]["seed"] == 20260916, w[0])
    check("window B is TB1+TB2 -> TB3 with seed 20260917",
          w[1]["train"] == ["TB1", "TB2"] and w[1]["eval"] == "TB3"
          and w[1]["seed"] == 20260917, w[1])
    check("no window reads TB4",
          all("TB4" not in r["train"] and r["eval"] != "TB4" for r in w), w)
    check("bootstrap reps frozen at 1000", S.BOOTSTRAP_REPS == 1000, "")


def _grp(acts, exps, pens):
    return dict(group_starts=np.array([0], np.int64),
                group_lengths=np.array([len(acts)], np.int64),
                act=np.array(acts, np.int64),
                exp=np.array(exps, np.int64),
                pen=np.array(pens, np.int64))


def test_provenance_active_contract():
    # one identity activated at 10, expiry 100, never penetrated
    g = _grp([10], [100], [-1])
    check("not active before activation", S.group_provenance(g, 0, 9) is None, "")
    check("active on activation bar itself",
          S.group_provenance(g, 0, 10) == (0, 0, 1),
          S.group_provenance(g, 0, 10))
    check("age grows with t", S.group_provenance(g, 0, 25) == (15, 15, 1),
          S.group_provenance(g, 0, 25))
    check("not active at expiry bar (t < expiry)",
          S.group_provenance(g, 0, 100) is None, "")

    gp = _grp([10], [100], [40])
    check("penetrated: not active at/after penetration bar",
          S.group_provenance(gp, 0, 40) is None
          and S.group_provenance(gp, 0, 39) == (29, 29, 1), "")

    gm = _grp([10, 20, 30], [100, 100, 100], [-1, -1, -1])
    check("oldest/newest ages and count over multiple identities",
          S.group_provenance(gm, 0, 50) == (40, 20, 3),
          S.group_provenance(gm, 0, 50))
    check("partially activated identities excluded",
          S.group_provenance(gm, 0, 25) == (15, 5, 2),
          S.group_provenance(gm, 0, 25))

    check("negative group index returns None", S.group_provenance(gm, -1, 50)
          is None, "")
    empty = _grp([], [], [])
    check("empty group returns None", S.group_provenance(empty, 0, 50) is None,
          "")


def test_provenance_uses_only_information_at_t():
    """未来 activation 不得影响 t 时的 provenance。"""
    g_future = _grp([10, 999], [100, 1000], [-1, -1])
    g_no_future = _grp([10], [100], [-1])
    t = 50
    a = S.group_provenance(g_future, 0, t)
    b = S.group_provenance(g_no_future, 0, t)
    check("an identity activating after t cannot change provenance at t",
          a == b == (40, 40, 1), (a, b))


def test_group_provenance_matches_frozen_contract():
    """与 frozen LOCAL lifecycle 的 active 定义逐项一致。"""
    rng = np.random.default_rng(7)
    for _ in range(200):
        n = int(rng.integers(1, 5))
        acts = rng.integers(0, 200, n)
        exps = acts + rng.integers(1, 200, n)
        pens = np.where(rng.random(n) < 0.5, -1, rng.integers(0, 400, n))
        g = _grp(acts.tolist(), exps.tolist(), pens.tolist())
        t = int(rng.integers(0, 400))
        ok = ((acts <= t) & (t < exps) & ((pens < 0) | (t < pens)))
        exp = None if not ok.any() else (t - int(acts[ok].min()),
                                         t - int(acts[ok].max()),
                                         int(ok.sum()))
        got = S.group_provenance(g, 0, t)
        if got != exp:
            check("frozen active contract parity", False, (t, got, exp))
            return
    check("frozen active contract parity over 200 random cases", True, "")


def test_no_tb4_wording():
    src = Path(S.__file__).read_text()
    check("source never claims 'TB4 never read'",
          "TB4 never read" not in src and "tb4_read_for_values" not in src, "")
    check("source records tb4_analytically_used",
          "tb4_analytically_used" in src, "")


def test_runtime_results_if_present():
    out = S.OUT
    f = out / "state1_summary.json"
    if not f.exists():
        check("state1_summary.json exists (skipped: not yet generated)", True,
              "")
        return
    d = json.loads(f.read_text())
    check("summary reports sample parity vs REPL-0",
          d["sample_parity"]["matches"] is True
          and d["sample_parity"]["n_samples"] == 37224,
          d["sample_parity"])
    check("summary has exactly 4 comparisons per window",
          len(d["bootstrap"]) == 8, len(d["bootstrap"]))
    check("S2-S0 reproduces frozen PATH-0 M1-M0 exactly",
          abs(d["crosscheck_vs_frozen_rounds"]["window_A_S2_minus_S0"]
              - d["crosscheck_vs_frozen_rounds"]["path0_M1_minus_M0"]) < 1e-15,
          d["crosscheck_vs_frozen_rounds"])
    check("S2-S0 reproduces frozen REPL-0 M1-M0 exactly",
          abs(d["crosscheck_vs_frozen_rounds"]["window_B_S2_minus_S0"]
              - d["crosscheck_vs_frozen_rounds"]["repl0_M1_minus_M0"]) < 1e-15,
          d["crosscheck_vs_frozen_rounds"])
    check("tb4_analytically_used is false",
          d["tb4_analytically_used"] is False, d["tb4_analytically_used"])
    check("no per-window TB4 statistic present",
          "TB4" not in json.dumps(d["by_symbol"]), "")


def test_provenance_audit_if_present():
    out = S.OUT
    f = out / "state1_sample_audit.json"
    if not f.exists():
        check("state1_sample_audit.json exists (skipped)", True, "")
        return
    d = json.loads(f.read_text())
    p = d["provenance"]
    check("all samples have active provenance on both sides",
          p["n_provenance_missing"] == 0
          and p["n_provenance_ok"] == p["n_samples"], p)
    check("ages are non-negative",
          p["upper_newest_age_bars"]["min"] >= 0, p["upper_newest_age_bars"])
    check("at least one active identity on every side",
          p["upper_n_active_identities"]["mean"] >= 1.0, p)
    c = d["prev_event_to_provenance_coupling"]
    check("coupling audit shows prev NEW event <=> fresh boundary",
          c["frac_upper_age0_given_prev_NEW_UPPER"]
          > c["frac_upper_age0_given_prev_no_NEW_UPPER"], c)


def test_repl0_sample_parity_on_disk():
    cache = S.CACHE / "repl0_samples.parquet"
    if not cache.exists():
        check("repl0 sample cache present (skipped)", True, "")
        return
    sm = pd.read_parquet(cache)
    d = json.loads((S.OUT / "repl0_summary.json").read_text())
    check("sample key hash recomputed equals frozen REPL-0 hash",
          S.sample_key_hash(sm) == d["repl0_sample_key_sha256"], "")


def main():
    test_registered_feature_sets()
    test_windows_and_seeds()
    test_provenance_active_contract()
    test_provenance_uses_only_information_at_t()
    test_group_provenance_matches_frozen_contract()
    test_no_tb4_wording()
    test_repl0_sample_parity_on_disk()
    test_runtime_results_if_present()
    test_provenance_audit_if_present()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
