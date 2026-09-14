"""PGM-BAR-0 — 硬测试（§20）。

覆盖：
  * frozen episode / sample hash 与现有结果一致
  * PGM-0 CRF helper parity（15-mask pairwise CRF 公式逐元素一致）
  * 每个 episode 恰好 1 个 terminal hazard=1
  * terminal mask 与 frozen episode mask 完全一致
  * 所有 observation feature 只能使用 <= t 的数据（因果时序）
  * 75 个 SMC prefix causality checks 全部 exact
  * TB4 在进入 SMC 前已截断（不出现 TB4 bar）
  * M0⊂M1⊂M2⊂M3⊂M4 严格 nested
  * preprocessing 只 fit train
  * Window A/B 无 overlap
  * existing PGM-0 / SEQ / STATE 输出零修改（本测试不读取/不修改它们；只验证新输出自洽）

运行（必须用带 sklearn 的解释器）：
  python3.11 research/liquidity_oracle_atlas/test_pgm_bar0_path_smc_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas import experiment_pgm_bar0_path_smc_v1 as M  # noqa: E402
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, CACHE, OUT,
)
from research.liquidity_oracle_atlas.run_5m_graph_probability_v1 import (  # noqa: E402
    load_raw_bars,
)
from research.liquidity_oracle_atlas.experiment_pgm0_endpoint_coupling_v1 import (  # noqa: E402
    fit_conditional_crf, predict_mask_prob, mask_index, unpack,
    MASK_BITS, MASK_PAIRS, MASK_VALUES, PAIR_INDEX,
)

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def _load_artifacts():
    obs = pd.read_parquet(CACHE / "pgm_bar0_samples.parquet")
    summary = json.loads((OUT / "pgm_bar0_summary.json").read_text())
    sample_audit = json.loads((OUT / "pgm_bar0_sample_audit.json").read_text())
    causality = pd.read_csv(OUT / "pgm_bar0_causality_audit.csv")
    bootstrap = pd.read_csv(OUT / "pgm_bar0_bootstrap.csv")
    return obs, summary, sample_audit, causality, bootstrap


# ---------------------------------------------------------------- §20.1 hash
def test_frozen_episode_hash_parity():
    _, summary, sample_audit, _, _ = _load_artifacts()
    h = sample_audit["episode_identity_sha256"]
    check("episode identity hash == frozen cf840a7",
          h == M.REVIEWER_FROZEN_EPISODE_HASH,
          f"{h} vs {M.REVIEWER_FROZEN_EPISODE_HASH}")
    check("episode hash matches_frozen flag",
          sample_audit["episode_hash_matches_frozen"] is True)
    check("tb4 analytically NOT used",
          sample_audit["tb4_analytically_used"] is False
          and summary["tb4_analytically_used"] is False)
    check("sample audit reports n_episodes>0",
          sample_audit["n_episodes"] > 0, str(sample_audit["n_episodes"]))


# ---------------------------------------------------------------- §20.2 CRF
def test_pgm0_crf_helper_parity():
    rng = np.random.default_rng(0)
    d = 4
    n = 30
    X = rng.standard_normal((n, d))
    mask = rng.integers(1, 16, size=n).astype(np.int64)
    theta, _ = fit_conditional_crf(X, mask, with_pairs=True)
    p = predict_mask_prob(theta, X[:1], with_pairs=True)[0]
    # brute-force 15-mask pairwise CRF (theta is a flat parameter array)
    intercept, W, beta = unpack(theta, X.shape[1], True)
    x = X[0]
    eta = np.zeros(15)
    for yi in range(15):
        bits = [int(bool((MASK_VALUES[yi] >> k) & 1)) for k in range(4)]
        e = 0.0
        for k in range(4):
            e += (intercept[k] + W[k] @ x) * bits[k]
        for pidx, (i, j) in enumerate(PAIR_INDEX):
            e += beta[pidx] * bits[i] * bits[j]
        eta[yi] = e
    p2 = np.exp(eta - np.max(eta))
    p2 /= p2.sum()
    check("CRF rows sum to 1", abs(p.sum() - 1.0) < 1e-9, str(p.sum()))
    check("predict_mask_prob matches brute-force pairwise CRF",
          np.allclose(p, p2, atol=1e-9), f"maxdiff={np.abs(p - p2).max()}")
    # determinism
    theta2, _ = fit_conditional_crf(X, mask, with_pairs=True)
    p3 = predict_mask_prob(theta2, X[:1], with_pairs=True)[0]
    check("fit_conditional_crf deterministic",
          np.allclose(p, p3, atol=0.0))
    # mask_index roundtrip
    mi = mask_index(np.array([1, 5, 15]))
    check("mask_index roundtrip",
          list(mi) == [0, 4, 14], str(list(mi)))


# ---------------------------------------------------------------- §20.3 term
def test_terminal_hazard_unique():
    obs, _, _, _, _ = _load_artifacts()
    g = obs.groupby("episode_id")["hazard"].sum()
    check("every episode has exactly 1 terminal hazard",
          bool((g == 1).all()),
          f"min={g.min()} max={g.max()} nbad={(g != 1).sum()}")


def test_terminal_mask_matches_frozen():
    obs, _, _, _, _ = _load_artifacts()
    ep0 = pd.read_parquet(CACHE / "episode0_episodes.parquet")
    key = {(r.symbol, int(r.start_bar)): int(r.event_mask)
           for r in ep0.itertuples()}
    # TB3 episodes have no frozen parquet entry; check TB1/TB2 only
    term = obs[(obs["hazard"] == 1) & (obs["block"] != "TB3")]
    mism = 0
    checked = 0
    for r in term.itertuples():
        exp = key.get((r.symbol, int(r.start_bar)))
        if exp is None:
            continue
        checked += 1
        if int(r.target_mask) != exp:
            mism += 1
    check("terminal target_mask == frozen episode mask (TB1/TB2)",
          mism == 0, f"{mism} mismatches / {checked} terminals")


# ---------------------------------------------------------------- §20.4 timing
def test_feature_timing_causal():
    obs, _, _, _, _ = _load_artifacts()
    ep0 = pd.read_parquet(CACHE / "episode0_episodes.parquet")
    up_price = {(r.symbol, int(r.start_bar)): float(r.start_upper_price)
                for r in ep0.itertuples()}
    valid = set(up_price.keys())
    atr0_map = {}
    # pick 6 episodes present in the frozen TB1/TB2 parquet
    cand_ep = obs.drop_duplicates("episode_id")
    cand_ep = cand_ep[cand_ep.apply(
        lambda r: (r.symbol, int(r.start_bar)) in valid, axis=1)]
    cand = cand_ep.sample(min(6, len(cand_ep)), random_state=1)
    max_diff = 0.0
    checked = 0
    for r in cand.itertuples():
        sym = r.symbol; s = int(r.start_bar)
        if sym not in atr0_map:
            atr0_map[sym] = load_raw_bars(sym)["atr"]
        atr0 = float(atr0_map[sym][s])
        if atr0 <= 0:
            continue
        U = up_price[(sym, s)]
        sub = obs[(obs["episode_id"] == r.episode_id)]
        bars = load_raw_bars(sym)
        c = bars["c"]
        # current close must be the observation bar's own close (<= t, causal)
        got = sub["cur_up_distance_R"].to_numpy()
        exp = (U - c[sub["bar_t"].to_numpy()].astype(float)) / atr0
        md = float(np.max(np.abs(got - exp)))
        max_diff = max(max_diff, md)
        checked += 1
        # current close must NOT be the episode-end close (future leak guard)
        end_c = c[int(sub["bar_t"].max())].astype(float)
        if len(sub) > 1:
            assert not np.allclose(got, (U - end_c) / atr0)
    check("cur_up_distance_R uses causal current close (<=t)",
          checked > 0 and max_diff < 1e-6, f"max_diff={max_diff} n={checked}")


# ---------------------------------------------------------------- §20.5 SMC
def test_smc_causality_all_pass():
    _, _, _, causality, _ = _load_artifacts()
    check("exactly 75 SMC prefix causality checks",
          len(causality) == 75, str(len(causality)))
    n_pass = int(causality["pass_"].sum())
    check("all 75 SMC prefix checks PASS",
          n_pass == 75, f"{n_pass} pass")


# ---------------------------------------------------------------- §20.6 TB4
def test_tb4_truncated():
    from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (
        build_blocks,
    )
    obs, _, _, _, _ = _load_artifacts()
    bars_by_sym = {s: load_raw_bars(s) for s in FULL_UNIV}
    all_days, day_block_code, _ = build_blocks(bars_by_sym)
    block_names = np.array(["TB1", "TB2", "TB3", "TB4"], dtype=object)
    ok = True
    detail = ""
    for s in FULL_UNIV:
        b = bars_by_sym[s]
        td = np.asarray(b["td"]).astype("datetime64[D]")
        code = day_block_code[np.searchsorted(all_days, td)]
        tb3_end = int(np.flatnonzero(code <= 2)[-1])
        sub = obs[obs["symbol"] == s]
        if len(sub) == 0:
            continue
        max_bar = int(sub["bar_t"].max())
        if max_bar > tb3_end:
            ok = False
            detail += f"{s}: max_bar={max_bar} > tb3_end={tb3_end}; "
    check("no observation uses a TB4 bar (all <= TB3 end)", ok, detail)
    check("all blocks in {TB1,TB2,TB3}",
          set(obs["block"].unique()) <= {"TB1", "TB2", "TB3"},
          str(set(obs["block"].unique())))


# ---------------------------------------------------------------- §20.7 nested
def test_nested_models():
    M0, M1, M2, M3, M4 = M.MODELS.keys()
    c0, c1, c2, c3, c4 = (set(M.MODELS[k][0]) for k in (M0, M1, M2, M3, M4))
    check("M0⊂M1⊂M2⊂M3⊂M4 strictly nested (numeric)",
          c0 < c1 < c2 < c3 < c4, f"{len(c0)}{len(c1)}{len(c2)}{len(c3)}{len(c4)}")
    check("M4 numeric == full NUM_ALL (42)",
          set(M.M4_NUM) == set(M.NUM_ALL) and len(M.NUM_ALL) == 42,
          str(len(M.NUM_ALL)))
    check("categorical == [prev_event_mask] for all models",
          all(v[1] == ["prev_event_mask"] for v in M.MODELS.values()))
    check("M0 adds current-5m snapshot beyond start (M1-M0 non-empty)",
          len(c1 - c0) > 0)
    check("M2 adds path beyond M1 (M2-M1 non-empty)",
          len(c2 - c1) > 0)
    check("M3 adds SMC beyond M2 (M3-M2 non-empty)",
          len(c3 - c2) > 0)
    check("M4 adds OB beyond M3 (M4-M3 non-empty)",
          len(c4 - c3) > 0)


# ---------------------------------------------------------------- §20.8 preproc
def test_preprocessing_fit_train_only():
    rng = np.random.default_rng(3)
    tr = pd.DataFrame({"x": rng.normal(10, 2, 50), "g": rng.integers(0, 3, 50)})
    ev = pd.DataFrame({"x": [12.0, 7.0], "g": [1, 2]})
    pipe = M.make_pipeline(["x"], ["g"])
    pre = pipe.named_steps["pre"]  # ColumnTransformer (preprocessing only)
    pre.fit(tr)  # no label needed for preprocessing
    # median imputer + standardscaler on train x
    mu = float(tr["x"].mean()); sigma = float(tr["x"].std(ddof=0))
    got = pre.transform(ev[["x", "g"]])
    exp0 = (12.0 - mu) / sigma
    check("preprocessing scaled eval by TRAIN statistics only",
          abs(got[0, 0] - exp0) < 1e-9, f"got={got[0,0]} exp={exp0}")


# ---------------------------------------------------------------- §20.9 window
def test_window_no_overlap():
    obs, summary, _, _, _ = _load_artifacts()
    blocks = set(obs["block"].unique())
    check("no TB4 block present in sample", "TB4" not in blocks, str(blocks))
    check("TB1/TB2/TB3 all represented",
          {"TB1", "TB2", "TB3"}.issubset(blocks), str(blocks))
    wins = summary["windows"]
    # Window A: train TB1, eval TB2 ; Window B: train TB1+TB2, eval TB3
    wA = next(w for w in wins if w["eval"] == "TB2")
    wB = next(w for w in wins if w["eval"] == "TB3")
    check("Window A eval(TB2) not in train",
          wA["eval"] not in wA["train"], str(wA))
    check("Window B eval(TB3) not in train",
          wB["eval"] not in wB["train"], str(wB))
    check("Windows A and B disjoint eval blocks",
          wA["eval"] != wB["eval"])


# ---------------------------------------------------------------- verdict
def test_verdict_consistency():
    _, summary, _, _, bootstrap = _load_artifacts()
    for gate, info in summary["verdict"].items():
        rows = bootstrap[bootstrap["gate"] == gate]
        expected = bool((rows["ci_hi"] < 0).all())
        check(f"verdict[{gate}] consistent with bootstrap CI",
              info["supported"] == expected,
              f"supported={info['supported']} expected={expected}")


def main():
    test_frozen_episode_hash_parity()
    test_pgm0_crf_helper_parity()
    test_terminal_hazard_unique()
    test_terminal_mask_matches_frozen()
    test_feature_timing_causal()
    test_smc_causality_all_pass()
    test_tb4_truncated()
    test_nested_models()
    test_preprocessing_fit_train_only()
    test_window_no_overlap()
    test_verdict_consistency()
    print("\n" + ("=" * 60))
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)} failed): {FAILS}")
        sys.exit(1)
    print("RESULT: ALL PGM-BAR-0 HARD TESTS PASS")


if __name__ == "__main__":
    main()
