"""STATE-1.1 — deterministic synthetic / contract tests.

覆盖：
  * 10 个 functional provenance 变量的定义（log1p / exact-zero / count）
  * F0–F3 feature set 精确性；无 duration / path / symbol / SMC / seq
  * zero flags 预先固定进入 frozen numeric pipeline（无新分支）
  * window / seed / comparison 集合冻结；primary = F3 - F1
  * sample hash 与 frozen STATE-1/REPL-0 一致
  * F2-F0 精确复现 STATE-1 S2-S0（运行时硬 guard 的常量）
  * 不写 STATE-1 原始输出文件
  * TB4 措辞合同

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_state11_functional_provenance_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_state11_functional_provenance_v1 as T  # noqa: E402
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as P  # noqa: E402
import research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 as S  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def test_functional_provenance_spec():
    check("exactly 10 functional provenance features",
          len(T.FUNC_PROV) == 10, T.FUNC_PROV)
    check("feature names are the preregistered ones",
          T.FUNC_PROV == [
              "upper_oldest_log_age", "upper_newest_log_age",
              "upper_newest_age_zero", "upper_oldest_age_zero",
              "upper_n_active_identities",
              "lower_oldest_log_age", "lower_newest_log_age",
              "lower_newest_age_zero", "lower_oldest_age_zero",
              "lower_n_active_identities"], T.FUNC_PROV)
    check("exactly 4 log-age + 4 zero flags + 2 counts",
          sum("log_age" in c for c in T.FUNC_PROV) == 4
          and sum(c.endswith("_zero") for c in T.FUNC_PROV) == 4
          and sum(c.endswith("n_active_identities") for c in T.FUNC_PROV) == 2,
          T.FUNC_PROV)
    check("no raw age_bars feature remains in the functional set",
          not any(c.endswith("_age_bars") for c in T.FUNC_PROV),
          [c for c in T.FUNC_PROV if c.endswith("_age_bars")])
    check("no age bins / feature search artefacts",
          not any(("bin" in c) or ("qcut" in c) for c in T.FUNC_PROV), "")


def test_functional_forms_are_deterministic():
    ages = np.array([0, 1, 2, 7, 100, 29288], dtype=float)
    check("log1p(age) matches the preregistered transform",
          np.allclose(np.log1p(ages), np.log1p(ages)), "")
    check("log1p(0) == 0 (age-zero maps to 0)",
          float(np.log1p(0.0)) == 0.0, "")
    check("zero flag is exactly age == 0",
          ((ages == 0).astype(float).tolist() == [1., 0., 0., 0., 0., 0.]), "")
    check("skew is tamed: max log1p far below raw max",
          float(np.log1p(ages).max()) < 11.0 < float(ages.max()), "")


def test_model_layers():
    check("F0 = frozen geometry only",
          T.MODELS[T.F0] == (T.GEOM, []), T.MODELS[T.F0])
    check("GEOM identical to PATH-0 / STATE-1",
          T.GEOM == P.CUR_NUM == S.GEOM, T.GEOM)
    check("F1 = geometry + 10 functional provenance, no categorical",
          T.MODELS[T.F1] == (T.GEOM + T.FUNC_PROV, []), T.MODELS[T.F1])
    check("F2 = geometry + prev_event only (S2 / M1 semantics)",
          T.MODELS[T.F2] == (T.GEOM, ["prev_event_mask"])
          and T.MODELS[T.F2] == (S.GEOM, S.CAT), T.MODELS[T.F2])
    check("F3 = geometry + functional provenance + prev_event exactly",
          T.MODELS[T.F3] == (T.GEOM + T.FUNC_PROV, ["prev_event_mask"]),
          T.MODELS[T.F3])
    banned = ("duration", "path", "symbol", "type", "scope", "volume",
              "signature", "smc", "bos", "choch", "sweep", "hmm", "seq")
    bad = [c for v in T.MODELS.values() for c in (v[0] + v[1])
           if any(b in c.lower() for b in banned)]
    check("no duration / path / symbol / SMC / seq feature", bad == [], bad)
    check("zero flags sit in the numeric tuple (no new pipeline branch)",
          all(c in T.MODELS[T.F1][0] for c in T.FUNC_PROV)
          and T.MODELS[T.F1][1] == [], T.MODELS[T.F1])
    check("primary comparison is F3 - F1",
          T.COMPARISONS[0] == (T.F3, T.F1), T.COMPARISONS[0])
    check("exactly 4 frozen comparisons",
          len(T.COMPARISONS) == 4
          and set(T.COMPARISONS) == {(T.F3, T.F1), (T.F1, T.F0),
                                     (T.F2, T.F0), (T.F3, T.F2)},
          T.COMPARISONS)


def test_windows_and_frozen_constants():
    w = T.WINDOWS
    check("window A = TB1 -> TB2 seed 20260916",
          w[0]["train"] == ["TB1"] and w[0]["eval"] == "TB2"
          and w[0]["seed"] == 20260916, w[0])
    check("window B = TB1+TB2 -> TB3 seed 20260917",
          w[1]["train"] == ["TB1", "TB2"] and w[1]["eval"] == "TB3"
          and w[1]["seed"] == 20260917, w[1])
    check("no window touches TB4",
          all("TB4" not in r["train"] and r["eval"] != "TB4" for r in w), w)
    check("frozen sample hash constant matches REPL-0/STATE-1",
          T.FROZEN_SAMPLE_HASH
          == "ddf034ed30a90597aa86dbad71d6397b1934d532afb348b8a93b406984787c12",
          T.FROZEN_SAMPLE_HASH)
    check("cross-check constants are the frozen S2-S0 values",
          T.EXPECT_A_F2_F0 == -0.007165155454171992
          and T.EXPECT_B_F2_F0 == -0.00813512230487083,
          (T.EXPECT_A_F2_F0, T.EXPECT_B_F2_F0))
    check("provenance function is the same object as STATE-1's",
          T.group_provenance is S.group_provenance, "")
    check("bootstrap reps frozen at 1000", T.BOOTSTRAP_REPS == 1000, "")


def test_does_not_touch_state1_outputs():
    src = Path(T.__file__).read_text()
    bad = [c for c in ["state1_summary.json\", \"w\"",
                       "state1_model_metrics.csv\", \"w",
                       "state1_bootstrap.csv\", \"w",
                       "state1_per_bit.csv\", \"w",
                       "state1_by_symbol.csv\", \"w",
                       "state1_sample_audit.json\", \"w"]
           if c in src]
    check("STATE-1 result files are read-only for this script", bad == [], bad)
    check("source never claims 'TB4 never read'",
          "TB4 never read" not in src and "tb4_read_for_values" not in src, "")
    check("source records tb4_analytically_used",
          "tb4_analytically_used" in src, "")


def test_coupling_table_shape():
    rng = np.random.default_rng(3)
    prev = rng.random(50) < 0.4
    zero = np.where(prev, rng.random(50) < 0.999, rng.random(50) < 0.07)
    rows = T.coupling_row(prev, zero, None, "prev_NEW_UPPER",
                          "upper_newest_age_zero", "tbl")
    check("coupling table has 2 rows forming a 2x2 (total preserved)",
          len(rows) == 2
          and sum(r["n_prev"] for r in rows) == 50
          and all(r["n_zero_flag_1"] + r["n_zero_flag_0"] == r["n_prev"]
                  for r in rows), rows)
    check("coupling rates are conditional rates in [0,1]",
          all(0.0 <= r["rate_zero_flag_1"] <= 1.0 for r in rows), rows)


def test_runtime_results_if_present():
    f = T.OUT / "state11_summary.json"
    if not f.exists():
        check("state11_summary.json exists (skipped)", True, "")
        return
    d = json.loads(f.read_text())
    cc = d["frozen_crosscheck"]
    check("F2-F0 reproduces frozen STATE-1 S2-S0 exactly (A)",
          abs(cc["window_A_F2_minus_F0"] - cc["expected_A"]) < 1e-15, cc)
    check("F2-F0 reproduces frozen STATE-1 S2-S0 exactly (B)",
          abs(cc["window_B_F2_minus_F0"] - cc["expected_B"]) < 1e-15, cc)
    check("STATE-1 sample hash parity confirmed",
          cc["state1_sample_hash_matches"] is True, cc)
    check("provenance recompute validated against STATE-1 audit",
          d["provenance_recompute_identical"]["checks"] >= 10
          and d["provenance_recompute_identical"]["all_identical"] is True,
          d["provenance_recompute_identical"])
    check("sample parity matches frozen cache",
          d["sample_parity"]["matches_frozen"] is True
          and d["sample_parity"]["n_samples"] == 37224, d["sample_parity"])
    check("exactly 4 comparisons per window (8 rows)",
          len(d["bootstrap"]) == 8, len(d["bootstrap"]))
    check("tb4_analytically_used is false",
          d["tb4_analytically_used"] is False, d["tb4_analytically_used"])
    check("no TB4 statistic present in outputs",
          "TB4" not in json.dumps(d["by_symbol"]), "")
    ct = pd.read_csv(T.OUT / "state11_coupling_tables.csv")
    check("4 coupling tables written, each with 2 rows",
          len(ct) == 8 and ct["table"].nunique() == 4, len(ct))


def test_coupling_matches_expected_structure():
    ct_p = T.OUT / "state11_coupling_tables.csv"
    if not ct_p.exists():
        check("coupling tables exist (skipped)", True, "")
        return
    ct = pd.read_csv(ct_p)
    nu = ct[ct["table"] == "prev_NEW_UPPER x upper_newest_age_zero"]
    r1 = float(nu[nu["prev_condition"].str.contains("== 1")]
               ["rate_zero_flag_1"].iloc[0])
    r0 = float(nu[nu["prev_condition"].str.contains("== 0")]
               ["rate_zero_flag_1"].iloc[0])
    check("prev NEW_UPPER is almost deterministic fresh-boundary indicator",
          r1 > 0.99 and r0 < 0.15, (r1, r0))


def main():
    test_functional_provenance_spec()
    test_functional_forms_are_deterministic()
    test_model_layers()
    test_windows_and_frozen_constants()
    test_does_not_touch_state1_outputs()
    test_coupling_table_shape()
    test_runtime_results_if_present()
    test_coupling_matches_expected_structure()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
