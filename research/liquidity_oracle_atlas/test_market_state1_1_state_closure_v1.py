"""MARKET-STATE-1.1 — State Representation Closure 硬测试。

收口两轮审计缺口：
  * 1.0 的 identity parity 只查 event_mask —— 本轮补 end_bar/start_block/
    event_mask + provenance group-id 的 exact closure。
  * tempo 必须被归类为 B0 Location+Path 的 derived representation（非新信息）。
  * compact provenance（6）必须保留绝大多数 full provenance 增益。

运行：python3.11 research/liquidity_oracle_atlas/test_market_state1_1_state_closure_v1.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from research.liquidity_oracle_atlas.experiment_market_state1_1_state_closure_v1 import (  # noqa: E402
    M2_NUM, COMPACT_PROV, DROPPED_PROV, TEMPO, WINDOWS,
)
from research.liquidity_oracle_atlas.experiment_pgm_bar0_path_smc_v1 import (  # noqa: E402
    make_pipeline,
)
from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
    group_provenance,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, CACHE, OUT,
)

FAILS = []
PARENT = "31e651c36d65d0b395f4c29a7addfbd239d4f7fb"


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def _load():
    base = pd.read_parquet(CACHE / "pgm_bar0_samples.parquet")
    enriched = pd.read_parquet(CACHE / "market_state1_samples.parquet")
    summary = json.loads((OUT / "market_state1_1_summary.json").read_text())
    identity = json.loads(
        (OUT / "market_state1_1_identity_audit.json").read_text())
    return base, enriched, summary, identity


# -------------------------------------------------- 1. built on 31e651c
def test_parent_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=REPO,
                         capture_output=True, text=True)
    head_parent = out.stdout.strip()
    check("MARKET-STATE-1.1 is exactly 1 commit on top of 31e651c",
          head_parent == PARENT, f"{head_parent}")


# -------------------------------------------------- 2. sample parity
def test_sample_parity():
    base, _, _, _ = _load()
    n_ep = int(base["episode_id"].nunique())
    n_rows = int(len(base))
    n_term = int(base["hazard"].sum())
    check("sample parity 37987/359714/37987",
          (n_ep, n_rows, n_term) == (37987, 359714, 37987),
          f"{n_ep}/{n_rows}/{n_term}")


# -------------------------------------------------- 3. EXACT identity closure
def test_exact_identity_closure():
    _, _, _, identity = _load()
    ei = identity["exact_episode_identity"]
    pg = identity["provenance_group_id_closure"]
    check("exact identity: 37987 terminals, 0 mismatch",
          ei["checked"] == 37987 and ei["mism_end"] == 0
          and ei["mism_blk"] == 0 and ei["mism_ev"] == 0
          and ei["missing"] == 0, str(ei))
    check("provenance group-id closure max_err == 0",
          pg["checked"] >= 100 and pg["max_err"] == 0.0, str(pg))


# -------------------------------------------------- 4. provenance parity
def test_provenance_parity():
    par = pd.read_csv(OUT / "market_state1_1_provenance_parity.csv")
    check("provenance canonical parity recorded (>=1000 points)",
          int(par["n_points"].iloc[0]) >= 1000, str(par.to_dict("records")))


# -------------------------------------------------- 5. first-bar residual
def test_provenance_timing():
    _, enriched, _, _ = _load()
    RESID = [c for c in enriched.columns
             if "age_residual" in c or "count_delta" in c]
    first = (enriched["bar_t"].to_numpy(np.int64)
             - enriched["start_bar"].to_numpy(np.int64)) == 0
    mx = float(max(np.abs(enriched.loc[first, c].to_numpy()).max()
                   for c in RESID))
    check("residual features == 0 at first bar (no future leak)", mx < 1e-9,
          f"max={mx}")


# -------------------------------------------------- 6. tempo derived repr
def test_tempo_derived_representation():
    _, enriched, _, _ = _load()
    k = (enriched["bar_t"].to_numpy(np.int64)
         - enriched["start_bar"].to_numpy(np.int64))
    net = (enriched["start_up_distance_R"].to_numpy(float)
           - enriched["cur_up_distance_R"].to_numpy(float))
    tv = enriched["path_total_variation_R"].to_numpy(float)
    ok = (np.allclose(enriched["tempo_signed_speed"], net / np.maximum(k, 1))
          and np.allclose(enriched["tempo_signed_efficiency"],
                          net / np.maximum(tv, 1e-12)))
    check("tempo is deterministic function of B0 columns (DERIVED)", bool(ok))


# -------------------------------------------------- 7. B0 == M2
def test_b0_parity():
    summary = json.loads((OUT / "market_state1_1_summary.json").read_text())
    pgm0 = json.loads((OUT / "pgm_bar0_summary.json").read_text())
    b0 = summary["model_metrics"]
    m2 = pgm0["model_metrics"]
    ok = True
    det = {}
    for w in ["A_TB1_to_TB2", "B_TB1TB2_to_TB3"]:
        ref = m2[w]["M2_INTRA_EPISODE_PATH"]
        got = b0[w]["B0_VALIDATED_STATE"]
        for key in ("mean_episode_nll", "hazard_nll", "endpoint_joint_nll"):
            d = abs(float(got[key]) - float(ref[key]))
            det[f"{w}.{key}"] = d
            ok = ok and d < 1e-6
    check("B0 reproduces PGM-BAR-0 M2 within 1e-6", ok, str(det))


# -------------------------------------------------- 8. compact retains gain
def test_compact_retains_gain():
    _, _, summary, _ = _load()
    gr = summary["gain_retention"]["retained_fraction"]
    ok = all(gr[w] is not None and gr[w] >= 0.9 for w in gr)
    check("compact provenance (6) retains >=90% of full provenance gain",
          ok, str(gr))


# -------------------------------------------------- 9. same sample
def test_same_sample():
    mt = pd.read_csv(OUT / "market_state1_1_model_metrics.csv")
    for w in mt["window"].unique():
        ok_w = len(mt[mt["window"] == w]["n_obs"].unique()) == 1
        check(f"all 5 models identical n_obs in {w}", ok_w)
    # compact + dropped partition matches design
    check("COMPACT_PROV has 6 features", len(COMPACT_PROV) == 6,
          str(COMPACT_PROV))
    check("DROPPED_PROV has 4 features", len(DROPPED_PROV) == 4,
          str(DROPPED_PROV))
    # dropped features are the always-0 / derived ones
    dropped_ok = (set(DROPPED_PROV) ==
                  {"upper_oldest_log_age_residual",
                   "lower_oldest_log_age_residual",
                   "upper_current_oldest_age_zero",
                   "lower_current_oldest_age_zero"})
    check("DROPPED_PROV = 2 always-zero oldest residuals + 2 oldest age-zero",
          dropped_ok, str(DROPPED_PROV))


# -------------------------------------------------- 10. preprocess train-only
def test_preprocessing_fit_train_only():
    rng = np.random.default_rng(7)
    tr = pd.DataFrame({"x": rng.normal(10, 2, 50), "g": rng.integers(0, 3, 50)})
    ev = pd.DataFrame({"x": [12.0, 7.0], "g": [1, 2]})
    pipe = make_pipeline(["x"], ["g"])
    pre = pipe.named_steps["pre"]
    pre.fit(tr)
    mu = float(tr["x"].mean()); sigma = float(tr["x"].std(ddof=0))
    got = pre.transform(ev[["x", "g"]])
    check("preprocessing scaled eval by TRAIN stats only",
          abs(got[0, 0] - (12.0 - mu) / sigma) < 1e-9)


# -------------------------------------------------- 11. window no overlap
def test_window_no_overlap():
    _, _, summary, _ = _load()
    wins = summary["windows"]
    wA = next(w for w in wins if w["eval"] == "TB2")
    wB = next(w for w in wins if w["eval"] == "TB3")
    check("Window A eval not in train", wA["eval"] not in wA["train"])
    check("Window B eval not in train", wB["eval"] not in wB["train"])


# -------------------------------------------------- 12. no TB4
def test_no_tb4():
    base, _, _, _ = _load()
    check("no TB4 block in sample", "TB4" not in set(base["block"].unique()))


# -------------------------------------------------- 13. one terminal
def test_one_terminal():
    base, _, _, _ = _load()
    g = base.groupby("episode_id")["hazard"].sum()
    check("each episode exactly 1 terminal hazard", bool((g == 1).all()),
          f"min={g.min()} max={g.max()}")


# -------------------------------------------------- 14. CRF success
def test_crf_success():
    opt = pd.read_csv(OUT / "market_state1_1_optimizer_audit.csv")
    check("all CRF optimizer success", bool(opt["success"].all()),
          str(opt[~opt["success"]].to_dict("records")))


# -------------------------------------------------- 15. frozen unmodified
def test_frozen_unmodified():
    out = subprocess.run(["git", "diff", "--name-only", PARENT], cwd=REPO,
                         capture_output=True, text=True)
    modified = [p for p in out.stdout.splitlines() if p.strip()]
    allowed_prefixes = (
        "research/liquidity_oracle_atlas/experiment_market_state1_1",
        "research/liquidity_oracle_atlas/test_market_state1_1",
        "research/analysis_results/local_liquidity_transition_v0/market_state1_1_",
    )
    forbidden = (
        "episode0_episodes.parquet", "episode_repl0_through_tb3.parquet",
        "pgm_bar0_samples.parquet", "pgm_bar0_", "episode_repl0",
        "episode0_",
    )
    bad_forbidden = [p for p in modified if any(f in p for f in forbidden)]
    bad_unallowed = [p for p in modified
                     if not any(p.startswith(a) for a in allowed_prefixes)]
    check("no historical frozen input modified (tracked diff vs 31e651c)",
          len(bad_forbidden) == 0 and len(bad_unallowed) == 0,
          f"forbidden={bad_forbidden} unallowed={bad_unallowed}")


def main():
    test_parent_commit()
    test_sample_parity()
    test_exact_identity_closure()
    test_provenance_parity()
    test_provenance_timing()
    test_tempo_derived_representation()
    test_b0_parity()
    test_compact_retains_gain()
    test_same_sample()
    test_preprocessing_fit_train_only()
    test_window_no_overlap()
    test_no_tb4()
    test_one_terminal()
    test_crf_success()
    test_frozen_unmodified()
    print("\n" + "=" * 60)
    if FAILS:
        print(f"RESULT: FAIL ({len(FAILS)} failed): {FAILS}")
        sys.exit(1)
    print("RESULT: ALL MARKET-STATE-1.1 HARD TESTS PASS")


if __name__ == "__main__":
    main()
