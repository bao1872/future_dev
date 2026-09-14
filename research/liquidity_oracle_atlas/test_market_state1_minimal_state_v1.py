"""MARKET-STATE-1 — 硬测试（§20）。

覆盖：
  * base commit 正确（099fbb87）
  * pgm_bar0 sample parity（37987 / 359714 / 37987）
  * TB1/TB2 terminal 与 frozen episode0 完全一致
  * TB3 terminal 与 episode_repl0_through_tb3 完全一致
  * vectorized provenance 与 canonical group_provenance 1000+ 点 exact parity
  * provenance 残差只使用 <= t 信息（首 bar 残差=0）
  * tempo 数学恒等式检查
  * B0 精确复现 PGM-BAR-0 M2（hazard / endpoint / episode NLL）
  * B0/P/T/PT 使用同一份样本
  * preprocessing 只 fit train
  * Window A/B 无 overlap
  * TB4 无 observation
  * 每个 episode 恰好一个 terminal
  * 所有 CRF optimizer success
  * 历史 frozen 文件零修改
  * 既有测试继续 0 FAIL（本文件不自动跑，单独执行验证）

运行（必须用带 sklearn 的解释器）：
  python3.11 research/liquidity_oracle_atlas/test_market_state1_minimal_state_v1.py
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

from research.liquidity_oracle_atlas import (  # noqa: E402
    experiment_market_state1_minimal_state_v1 as M,
)
from research.liquidity_oracle_atlas.experiment_pgm_bar0_path_smc_v1 import (  # noqa: E402
    make_pipeline,
)
from research.liquidity_oracle_atlas.experiment_local_liquidity_transition_v0 import (  # noqa: E402
    FULL_UNIV, CACHE, OUT,
)
from research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 import (  # noqa: E402
    group_provenance,
)
from research.liquidity_oracle_atlas.experiment_step01_cause_decomposition_v1 import (  # noqa: E402
    load_seq,
)

FAILS = []
BASE = "099fbb87b0fa5db5cce6e03342decb4838cb7c07"


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


def _load():
    obs = pd.read_parquet(CACHE / "market_state1_samples.parquet")
    base = pd.read_parquet(CACHE / "pgm_bar0_samples.parquet")
    summary = json.loads((OUT / "market_state1_summary.json").read_text())
    sample = json.loads((OUT / "market_state1_sample_audit.json").read_text())
    return obs, base, summary, sample


# ---------------------------------------------------------------- §20.1 base
def test_base_commit():
    out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                         capture_output=True, text=True)
    head = out.stdout.strip()
    check("base commit == 099fbb87", head == BASE, f"{head}")


# ---------------------------------------------------------- §20.2 sample parity
def test_sample_parity():
    base, _, _, _ = _load()
    n_ep = int(base["episode_id"].nunique())
    n_rows = int(len(base))
    n_term = int(base["hazard"].sum())
    check("sample parity 37987/359714/37987",
          (n_ep, n_rows, n_term) == (37987, 359714, 37987),
          f"{n_ep}/{n_rows}/{n_term}")


# ---------------------------------------------------------- §20.3 terminal id
def test_terminal_identity():
    base, _, _, _ = _load()
    ep0 = pd.read_parquet(CACHE / "episode0_episodes.parquet")
    ep3 = pd.read_parquet(CACHE / "episode_repl0_through_tb3.parquet")
    k0 = {(r.symbol, int(r.start_bar)): int(r.event_mask) for r in ep0.itertuples()}
    k3 = {(r.symbol, int(r.start_bar)): int(r.event_mask) for r in ep3.itertuples()}
    term = base[base["hazard"] == 1]
    mism = 0
    for r in term.itertuples():
        mp = k0 if r.block in ("TB1", "TB2") else k3
        exp = mp.get((r.symbol, int(r.start_bar)))
        if exp is None or int(r.target_mask) != exp:
            mism += 1
    check("TB1/TB2/TB3 terminal mask == frozen episode mask", mism == 0,
          f"{mism} mismatches")


# -------------------------------------------------- §20.4 provenance parity
def test_provenance_parity():
    par = pd.read_csv(OUT / "market_state1_provenance_parity.csv")
    check("provenance parity >=1000 points, all ok",
          len(par) >= 1000 and bool(par["ok"].all()),
          f"n={len(par)} all_ok={par['ok'].all() if len(par) else None}")


# ------------------------------------------------ §20.5 provenance <= t only
def test_provenance_timing():
    obs, _, _, _ = _load()
    RESID = [c for c in M.PROV_DELTA if "age_residual" in c or "count_delta" in c]
    first = (obs["bar_t"].to_numpy(np.int64)
             - obs["start_bar"].to_numpy(np.int64)) == 0
    mx = float(max(np.abs(obs.loc[first, c].to_numpy()).max() for c in RESID))
    check("residual features == 0 at first bar (no future leak)", mx < 1e-9,
          f"max={mx}")


# ------------------------------------------------------- §20.6 tempo identity
def test_tempo_identity():
    obs, _, _, _ = _load()
    k = (obs["bar_t"].to_numpy(np.int64)
         - obs["start_bar"].to_numpy(np.int64))
    net = (obs["start_up_distance_R"].to_numpy(float)
           - obs["cur_up_distance_R"].to_numpy(float))
    tv = obs["path_total_variation_R"].to_numpy(float)
    ss = obs["tempo_signed_speed"].to_numpy(float)
    as_ = obs["tempo_abs_speed"].to_numpy(float)
    se = obs["tempo_signed_efficiency"].to_numpy(float)
    ae = obs["tempo_abs_efficiency"].to_numpy(float)
    # identity 1: at k==0 all zero
    first = k == 0
    ok_first = np.allclose(ss[first], 0) and np.allclose(se[first], 0)
    # identity 2: abs == |signed|
    ok_abs = (np.allclose(as_, np.abs(ss)) and np.allclose(ae, np.abs(se)))
    # identity 3: signed_speed == net / max(k,1)
    ok_speed = np.allclose(ss, net / np.maximum(k, 1))
    # identity 4: signed_eff == net / max(tv,eps)
    ok_eff = np.allclose(se, net / np.maximum(tv, 1e-12))
    check("tempo first-bar == 0", bool(ok_first))
    check("tempo abs == |signed|", bool(ok_abs))
    check("tempo signed_speed == net/max(k,1)", bool(ok_speed))
    check("tempo signed_eff == net/max(tv,eps)", bool(ok_eff))


# ------------------------------------------------ §20.7 B0 == PGM-BAR-0 M2
def test_b0_parity():
    summary = json.loads((OUT / "market_state1_summary.json").read_text())
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
    check("B0 reproduces PGM-BAR-0 M2 (hazard/endpoint/episode NLL)", ok,
          str(det))


# ------------------------------------------------- §20.8 same sample
def test_same_sample():
    obs, _, summary, _ = _load()
    mt = pd.read_csv(OUT / "market_state1_model_metrics.csv")
    ok = True
    for w in mt["window"].unique():
        n_obs = mt[mt["window"] == w]["n_obs"].unique()
        ok = ok and len(n_obs) == 1
    check("all 4 models use identical n_obs (per window)", ok,
          str(mt.groupby("window")["n_obs"].unique().to_dict()))
    # M2 columns unchanged vs base; new columns present
    for c in M.M2_NUM:
        check(f"M2 column present in enriched sample: {c}", c in obs.columns)
    for c in M.PROV_DELTA + M.TEMPO:
        check(f"new feature present: {c}", c in obs.columns)
    # no TB4
    check("no TB4 block in sample", "TB4" not in set(obs["block"].unique()))


# ------------------------------------------------- §20.9 preprocess train-only
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


# ------------------------------------------------- §20.10 window no overlap
def test_window_no_overlap():
    _, _, summary, _ = _load()
    wins = summary["windows"]
    wA = next(w for w in wins if w["eval"] == "TB2")
    wB = next(w for w in wins if w["eval"] == "TB3")
    check("Window A eval not in train", wA["eval"] not in wA["train"])
    check("Window B eval not in train", wB["eval"] not in wB["train"])


# ------------------------------------------------- §20.11 TB4 none
def test_no_tb4():
    obs, _, _, _ = _load()
    check("no TB4 block", "TB4" not in set(obs["block"].unique()),
          str(set(obs["block"].unique())))


# ------------------------------------------------- §20.12 one terminal
def test_one_terminal():
    obs, _, _, _ = _load()
    g = obs.groupby("episode_id")["hazard"].sum()
    check("each episode exactly 1 terminal hazard", bool((g == 1).all()),
          f"min={g.min()} max={g.max()}")


# ------------------------------------------------- §20.13 CRF success
def test_crf_success():
    opt = pd.read_csv(OUT / "market_state1_optimizer_audit.csv")
    check("all CRF optimizer success", bool(opt["success"].all()),
          str(opt[~opt["success"]].to_dict("records")))


# ------------------------------------------------- §20.14 frozen zero modify
def test_frozen_unmodified():
    out = subprocess.run(
        ["git", "diff", "--name-only", BASE], cwd=REPO,
        capture_output=True, text=True)
    modified = [p for p in out.stdout.splitlines() if p.strip()]
    # only the two new MARKET-STATE-1 files may be tracked-changed; everything
    # else must be untracked (??) which git diff --name-only does NOT list.
    allowed = {
        "research/liquidity_oracle_atlas/experiment_market_state1_minimal_state_v1.py",
        "research/liquidity_oracle_atlas/test_market_state1_minimal_state_v1.py",
    }
    bad = [p for p in modified if p not in allowed]
    check("no historical frozen file modified (tracked diff)", len(bad) == 0,
          str(bad))


def main():
    test_base_commit()
    test_sample_parity()
    test_terminal_identity()
    test_provenance_parity()
    test_provenance_timing()
    test_tempo_identity()
    test_b0_parity()
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
    print("RESULT: ALL MARKET-STATE-1 HARD TESTS PASS")


if __name__ == "__main__":
    main()
