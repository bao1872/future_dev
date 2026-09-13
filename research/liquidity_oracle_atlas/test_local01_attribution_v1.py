"""LOCAL-0.1 — geometry attribution 的确定性契约测试。

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_local01_attribution_v1.py

依赖 534de62e 生成的 gitignored sample cache；cache 缺失时按约定 SKIP 并
明确报告（不静默重建 frozen LOCAL-0 state）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 as L0  # noqa: E402
import research.liquidity_oracle_atlas.experiment_local01_geometry_attribution_v1 as m  # noqa: E402

FAILS = []
SKIPPED = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def skip(name, why):
    print(f"[SKIP] {name} :: {why}")
    SKIPPED.append(name)


# ---------------------------------------------------------------- 1–4
def test_feature_sets():
    want = {
        "M0_GLOBAL_PRIOR": ([], []),
        "M1_SYMBOL_ONLY": ([], ["symbol"]),
        "M2_GEOMETRY_ONLY": (list(L0.NUM_FEATURES), []),
        "M3_GEOMETRY_SYMBOL": (list(L0.NUM_FEATURES), ["symbol"]),
    }
    got = {n: (list(nm), list(ct)) for n, nm, ct in m.SPECS}
    check("1 four model feature sets exactly as defined", got == want,
          f"got={got} want={want}")

    g = dict(got)
    check("2 M1_SYMBOL_ONLY contains no geometry",
          not (set(g["M1_SYMBOL_ONLY"][0]) & set(L0.NUM_FEATURES)),
          str(g["M1_SYMBOL_ONLY"]))
    check("3 M2_GEOMETRY_ONLY contains no symbol",
          "symbol" not in g["M2_GEOMETRY_ONLY"][1],
          str(g["M2_GEOMETRY_ONLY"]))
    check("4 M3 feature set == frozen B1 (NUM + CAT)",
          g["M3_GEOMETRY_SYMBOL"] == (list(L0.NUM_FEATURES),
                                      list(L0.CAT_FEATURES)),
          str(g["M3_GEOMETRY_SYMBOL"]))
    check("4b frozen B1 feature list unchanged",
          list(L0.NUM_FEATURES) == ["up_distance_R", "down_distance_R",
                                    "width_R", "log_distance_ratio"]
          and list(L0.CAT_FEATURES) == ["symbol"],
          f"{L0.NUM_FEATURES} {L0.CAT_FEATURES}")


def test_pipeline_params():
    pipe = m.make_pipeline(list(L0.NUM_FEATURES), list(L0.CAT_FEATURES))
    clf = pipe.named_steps["clf"]
    check("clf params frozen (L2 / C=1 / lbfgs / 3000)",
          clf.penalty == "l2" and clf.C == 1.0 and clf.solver == "lbfgs"
          and clf.max_iter == 3000,
          f"{clf.penalty} {clf.C} {clf.solver} {clf.max_iter}")


# ---------------------------------------------------------------- 5–6
def test_sample_keys_and_blocks():
    missing = [s for s in L0.FULL_UNIV
               if not (m.CACHE / f"local0_samples_{s}.parquet").exists()]
    if missing:
        skip("5 train/test row keys vs 534de62e", f"cache missing {missing}")
        skip("6 TB3/TB4 excluded from fit/metric/bootstrap", "cache missing")
        return

    samples = m.load_cached_samples(L0.FULL_UNIV)
    tb2_start = int(np.min(L0.to_ns_int(
        samples.loc[samples["block"] == L0.TEST_BLOCK, "decision_time"])))
    train_all = samples[(samples["block"] == L0.TRAIN_BLOCK)
                        & samples["label"].isin(L0.RESOLVED_CODES)].copy()
    test = samples[(samples["block"] == L0.TEST_BLOCK)
                   & samples["label"].isin(L0.RESOLVED_CODES)].copy()
    train = train_all[L0.to_ns_int(train_all["resolution_time"])
                      < tb2_start].copy().reset_index(drop=True)
    test = test.reset_index(drop=True)

    got = dict(tb1_resolved_before_purge=int(len(train_all)),
               tb1_resolved_after_purge=int(len(train)),
               tb2_resolved=int(len(test)))
    check("5a sample counts == frozen 534de62e baseline", got == m.EXPECT,
          f"got={got} want={m.EXPECT}")

    p = m.OUT / "local01_summary.json"
    if p.exists():
        d = json.loads(p.read_text())["data"]
        check("5b train_row_key_sha256 matches committed summary",
              d["train_row_key_sha256"] == m.row_key_hash(train),
              "train keys drifted")
        check("5c test_row_key_sha256 matches committed summary",
              d["test_row_key_sha256"] == m.row_key_hash(test),
              "test keys drifted")
    else:
        skip("5b/5c row-key hashes", "local01_summary.json not present yet")

    check("6a no TB3/TB4 in train",
          not train["block"].isin(["TB3", "TB4"]).any(),
          str(sorted(train["block"].unique())))
    check("6b no TB3/TB4 in test",
          not test["block"].isin(["TB3", "TB4"]).any(),
          str(sorted(test["block"].unique())))
    check("6c TB3/TB4 still present in cache (guard is meaningful)",
          int(samples["block"].isin(["TB3", "TB4"]).sum()) > 0,
          str(int(samples["block"].isin(["TB3", "TB4"]).sum())))
    check("6d no censored/ambiguous rows in train or test",
          not np.isin(train["label"].to_numpy(),
                      list(L0.RIGHT_CENSOR_CODES) + [L0.AMBIGUOUS]).any()
          and not np.isin(test["label"].to_numpy(),
                          list(L0.RIGHT_CENSOR_CODES) + [L0.AMBIGUOUS]).any())


# ---------------------------------------------------------------- synthetic
def test_prior_and_proba_contract():
    tr = pd.DataFrame(dict(
        label=np.array([L0.UP, L0.UP, L0.DOWN, L0.DOWN]),
        symbol=np.array(["AG", "CU", "AG", "CU"]),
        up_distance_R=np.array([1.0, 2.0, 3.0, 4.0]),
        down_distance_R=np.array([2.0, 1.0, 4.0, 3.0]),
        width_R=np.array([3.0, 3.0, 7.0, 7.0]),
        log_distance_ratio=np.array([-0.7, 0.7, -0.3, 0.3]),
    ))
    mdl = m.fit_spec(("M0_GLOBAL_PRIOR", [], []), tr)
    P = m.proba_up(mdl, tr)
    check("M0 prior = train P(UP) and sums to 1",
          np.allclose(P[:, 0], 0.5) and np.allclose(P.sum(axis=1), 1.0),
          str(P.tolist()))
    m1 = m.fit_spec(("M1_SYMBOL_ONLY", [], ["symbol"]), tr)
    check("M1 fitted without geometry columns",
          m1["cols"] == ["symbol"], str(m1["cols"]))
    m2 = m.fit_spec(("M2_GEOMETRY_ONLY", list(L0.NUM_FEATURES), []), tr)
    check("M2 fitted without symbol column",
          "symbol" not in m2["cols"], str(m2["cols"]))


def main():
    test_feature_sets()
    test_pipeline_params()
    test_sample_keys_and_blocks()
    test_prior_and_proba_contract()
    print(f"\n==== {len(FAILS)} FAIL / {len(SKIPPED)} SKIP ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
