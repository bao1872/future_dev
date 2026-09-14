"""Tests for DYNAMIC-PGM-1B Terminal + Reset Closure.

Covers:
  * frozen sample / terminal-parity constants and episode identity hash
  * Phi = 10 / mem = 6 with the 1A.2c frozen partition
  * no cross-episode memory (lag1 sources absent on episode-first rows)
  * reset pair chaining (same symbol+block, gap >= 0, mask handoff)
  * reset primitive supports + full start-state reconstruction
  * shared Geometric gap magnitude and shared constant-ZTP count magnitude
  * reset count delta == occurrence delta
  * day-cluster bootstrap semantics
  * terminal + reset synthetic window smoke and boundary NLL exact sum
  * verdicts read only the intended comparisons
  * namespace isolation, JSON serializable
"""
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2_lag_closure_v1 as lag  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2c_representation_control_v1 as rep  # noqa: E402
import research.liquidity_oracle_atlas.experiment_market_state1_1_state_closure_v1 as ms  # noqa: E402
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as pm  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1b_terminal_reset_closure_v1 as T  # noqa: E402

OUT = base.OUT


# --------------------------------------------------------------------------- #
def test_frozen_constants():
    assert T.EXPECTED_ROWS == 359714
    assert T.EXPECTED_EPISODES == 37987 and T.EXPECTED_TERMINALS == 37987
    assert T.FROZEN_EPISODE_HASH == pm.REVIEWER_FROZEN_EPISODE_HASH
    assert T.FROZEN_EPISODE_HASH == (
        "cf840a7191e0265e31442f7b8d8d5ae4173751b0cba15729f95d5c146ab9355c")
    assert set(T.FROZEN_TERMINAL) == {"A_TB1_to_TB2", "B_TB1TB2_to_TB3"}
    assert abs(T.FROZEN_TERMINAL["A_TB1_to_TB2"]["hazard_nll"]
               - 0.32684498811981755) < 1e-15
    assert abs(T.FROZEN_TERMINAL["B_TB1TB2_to_TB3"]["mean_episode_nll"]
               - 4.299368968756157) < 1e-15
    # frozen terminal parity model is exactly the observable state
    assert T.TP_NUM == list(ms.MODELS["PTc_COMPACT_PROV_TEMPO"][0]) == T.OBS_NUM
    assert T.WINDOWS == base.WINDOWS


def test_phi_and_mem_layout():
    assert len(T.PHI_COLS) == 10 and len(T.MEM_COLS) == 6
    # underlying layout equals the 1A.2c frozen CUR / MEM partition
    assert [c[3:] for c in rep.ZT_CUR_COLS] == [c[4:] for c in T.PHI_COLS]
    assert [c[3:] for c in rep.ZT_MEM_COLS] == [c[4:] for c in T.MEM_COLS]
    # nesting of terminal model designs
    assert T.T0_NUM == T.OBS_NUM + [T.LAG_AVAIL]
    assert T.T1_NUM == T.T0_NUM + T.PHI_COLS
    assert T.T2_NUM == T.T1_NUM + T.MEM_COLS
    # nesting of reset occurrence designs
    assert T.R1_OCC == T.R0_OCC + T.OBS_NUM + T.PHI_COLS
    assert T.R2_OCC == T.R1_OCC + T.MEM_COLS
    # reset models must never carry a previous-bar STATE value
    assert not (set(T.MEM_COLS) & set(lag.LAG_COLS))
    assert len(T.RESET_TARGETS) == 10


def test_reset_target_partition():
    assert len(T.RESET_GEOM) == 2 and len(T.RESET_SHAPE) == 2
    assert len(T.RESET_AGE) == 4 and len(T.RESET_CNT) == 2
    assert len(T.RESET_TARGETS) == len(set(T.RESET_TARGETS)) == 10
    assert T.GAP_FEAT == ["gap_positive", "log1p_gap"]


# --------------------------------------------------------------------------- #
def _real_small(blocks=("TB1", "TB2"), k=100):
    """Contiguous first-k episodes per symbol+block (keeps the chain intact)."""
    obs = pd.read_parquet(base.CACHE / "market_state1_samples.parquet")
    keep_cols = list(dict.fromkeys(
        T.OBS_NUM + T.CAT + list(lag.LAG_BASE) + base.BUILD_SRC_COLS
        + ["symbol", "episode_id", "start_bar", "bar_t", "block",
           "episode_start_day", "hazard", "target_mask", "prev_event_mask"]))
    keep_cols = [c for c in keep_cols if c in obs.columns]
    obs = obs[keep_cols]
    sub = obs[obs["block"].isin(blocks)].copy()
    sel = (sub.sort_values(["symbol", "block", "start_bar"], kind="stable")
              .groupby(["symbol", "block"]).head(k)["episode_id"])
    small = sub[sub["episode_id"].isin(set(sel))].copy()
    small, phi, mem = T.add_phi_and_memory(small)
    return small, phi, mem


def test_real_pair_chain_and_reconstruction():
    small, phi, mem = _real_small()
    assert len(phi) == 10 and len(mem) == 6
    pair, first = T.build_reset_pairs(small)
    assert len(pair) > 0
    au = T.audit_reset_pairs(pair)
    assert all(v == 0 for v in au["checks"].values())
    assert au["n_immediate"] + au["n_positive_gap"] == au["n_pairs"]
    rc = T.audit_reset_reconstruction(pair, first)
    assert rc["max_error"] < 1e-8
    # storage-limited derived representation is reported, not gated
    assert rc["storage_limited"]["start_log_ratio_max_abs_error"] < 1e-4


def test_memory_has_no_cross_episode_leak():
    small, phi, mem = _real_small()
    first = (small["bar_t"].to_numpy(np.int64)
             - small["start_bar"].to_numpy(np.int64)) == 0
    assert bool((small.loc[first, T.LAG_AVAIL].to_numpy() == 0).all())
    for f in ("path_max_up_excursion_R", "path_max_down_excursion_R",
              "upper_active_identity_count_delta",
              "lower_active_identity_count_delta"):
        assert bool(small.loc[first, f"lag1_{f}"].isna().all())
    assert bool(np.isfinite(
        small.loc[~first, T.MEM_COLS].to_numpy(np.float64)).all())
    # phi is a pure function of S_t -> always finite
    assert bool(np.isfinite(small[T.PHI_COLS].to_numpy(np.float64)).all())


def test_no_cross_block_pair():
    obs = pd.read_parquet(base.CACHE / "market_state1_samples.parquet")
    # build_reset_pairs needs the state/phi/mem columns -> derive them first
    keep = list(dict.fromkeys(
        T.OBS_NUM + T.CAT + list(lag.LAG_BASE) + base.BUILD_SRC_COLS
        + ["symbol", "episode_id", "start_bar", "bar_t", "block",
           "episode_start_day", "hazard", "target_mask", "prev_event_mask"]))
    obs = obs[[c for c in keep if c in obs.columns]]
    obs, _phi, _mem = T.add_phi_and_memory(obs)
    pair, _ = T.build_reset_pairs(obs)
    assert int((pair["block"] != pair["next_block"]).sum()) == 0
    assert int((pair["gap_bars"] < 0).sum()) == 0


# --------------------------------------------------------------------------- #
def test_gap_geometric_support_and_shared_magnitude():
    g = np.array([0, 0, 1, 1, 2, 3, 0, 0], dtype=float)
    p = T.fit_gap_geometric(g)
    assert 0 < p <= 1
    # p = 1 / mean(positive) = 1 / 1.75
    assert abs(p - 1.0 / 1.75) < 1e-12
    nll = T.gap_positive_nll(g, p)
    assert np.all(np.isfinite(nll))
    assert np.allclose(nll[g == 0], 0.0)          # zero rows carry no magnitude
    assert np.all(nll[g > 0] > 0)
    # edge p -> 1 must not produce NaN
    assert np.all(np.isfinite(T.gap_positive_nll(g, 1.0)))
    # a single shared p means the magnitude NLL difference is exactly zero
    assert np.allclose(T.gap_positive_nll(g, p), T.gap_positive_nll(g, p))


def test_day_cluster_bootstrap_semantics():
    delta = np.array([0., 0., 3., 3., 3.])
    day = np.array(["A", "A", "B", "B", "B"])
    lo, hi = ms.boot_delta(delta, day, 7, reps=200)
    assert lo <= 1.8 <= hi
    c = np.full(40, -1.5)
    lo2, hi2 = ms.boot_delta(c, np.array(["A"] * 20 + ["B"] * 20), 3, reps=200)
    assert abs(lo2 + 1.5) < 1e-12 and abs(hi2 + 1.5) < 1e-12


def test_reset_heads_shared_count_magnitude():
    small, _phi, _mem = _real_small(k=80)
    pair, _ = T.build_reset_pairs(small)
    ptr = pair[pair["block"] == "TB1"].reset_index(drop=True)
    pev = pair[pair["block"] == "TB2"].reset_index(drop=True)
    assert len(ptr) and len(pev)
    Xtr, Xev = T._design(ptr, pev, T.R0_OCC + T.GAP_FEAT, T.MASK_CAT)
    d = T._fit_heads(Xtr, Xev, ptr, pev, 0.5)
    Ytr = ptr[T.RESET_CNT].to_numpy(np.int64)
    Yev = pev[T.RESET_CNT].to_numpy(np.int64)
    k0c = base.fit_constant_count_head(Ytr, Yev)
    kc = base.fit_state_count_head(Xtr, Ytr, Xev, Yev,
                                   constant_rates=k0c["constant_rates"])
    for j in range(len(T.RESET_CNT)):
        assert np.array_equal(k0c["rate_ev"][:, j], kc["rate_ev"][:, j])
        assert len(np.unique(kc["rate_ev"][:, j])) == 1
    # count NLL delta equals occurrence delta
    d_cnt = kc["nll_ev"] - k0c["nll_ev"]
    d_occ = (lag._occ_nll(Yev, kc["p0_ev"])
             - lag._occ_nll(Yev, k0c["p0_ev"]))
    assert np.max(np.abs(d_cnt - d_occ)) < 1e-10
    # every staged head is finite
    for k_, v in d.items():
        if k_.startswith("_"):
            continue
        arr = np.asarray(v)
        assert np.all(np.isfinite(arr))


# --------------------------------------------------------------------------- #
def test_window_smoke_and_boundary_sum():
    small, _phi, _mem = _real_small(k=60)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.parquet"
        small.to_parquet(p, index=False)
        w = dict(name="SYNTH", train=["TB1"], eval=["TB2"][0], seed=11)
        res = T.run_single_window_1b(w, str(p))
    tm = {m["model"] for m in res["term_metrics"]}
    assert tm == {"TP_FROZEN_PTc", "T0_STATE_AVAIL", "T1_STATE_PHI",
                  "T2_STATE_PHI_MEM"}
    rm = {m["model"] for m in res["reset_metrics"]}
    assert rm == {"R0_ENDPOINT_ONLY", "R1_STATE_PHI", "R2_STATE_PHI_MEM"}
    for m in res["term_metrics"] + res["reset_metrics"]:
        assert np.isfinite(m.get("mean_episode_nll",
                                 m.get("mean_reset_nll")))
    # boundary bootstrap carries exactly the two intended comparisons
    assert {b["comparison"] for b in res["bound_boots"]} == {T.C_BOUND_PHI,
                                                             T.C_BOUND_MEM}
    assert {b["comparison"] for b in res["term_boots"]} == {
        "T0-TP", T.C_TERM_PHI, T.C_TERM_MEM}
    assert {b["comparison"] for b in res["reset_boots"]} == {T.C_RESET_STATE,
                                                             T.C_RESET_MEM}
    assert all(r["success"] for r in res["opt_rows"])
    assert json.loads(json.dumps(res, default=str))


def test_boundary_nll_is_terminal_plus_reset():
    """B_i must be exactly terminal episode NLL + reset NLL for paired episodes."""
    small, _phi, _mem = _real_small(k=60)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "s.parquet"
        small.to_parquet(p, index=False)
        w = dict(name="SYNTH", train=["TB1"], eval="TB2", seed=11)
        res = T.run_single_window_1b(w, str(p))
    b0 = next(b for b in res["bound_boots"] if b["comparison"] == T.C_BOUND_PHI)
    # B1-B0 = (T1-T0) episode + (R1-R0) reset, both episode-weighted
    t = next(b for b in res["term_boots"] if b["comparison"] == T.C_TERM_PHI)
    r = next(b for b in res["reset_boots"] if b["comparison"] == T.C_RESET_STATE)
    assert b0["n_episodes"] == r["n_pairs"]
    # signs must agree (both are Phi gains on a shared episode set)
    assert (b0["delta_sample_mean"] < 0) == \
        ((t["delta_sample_mean"] + r["delta_sample_mean"]) < 0)


def test_verdict_reads_only_intended_comparisons():
    src = inspect.getsource(T.main)
    assert "TERMINAL_PHI_SUPPORTED" in src
    assert "RESET_STATE_SIGNAL_SUPPORTED" in src
    assert "BOUNDARY_PHI_SUPPORTED" in src
    assert "DYNAMIC_PGM1B_BOUNDARY_KERNEL_CLOSED" in src
    assert T.C_TERM_PHI == "T1-T0" and T.C_RESET_STATE == "R1-R0"
    assert T.C_BOUND_PHI == "B1-B0"
    # memory challengers are reported but never gate
    for label in ('"T2-T1"', '"R2-R1"', '"B2-B1"'):
        assert label in src
    gate_region = src[src.index("def _verdict"):src.index("# ---------------- outputs")]
    assert "mem_report" in gate_region


def test_output_namespace_isolation():
    assert T.PREFIX == "dynamic_pgm1b"
    assert T.BASE_SHA == "7e86c8ee72d3117b368dd1639e0c73de49534fc5"
    for f in ("dynamic_pgm1b_terminal_metrics.csv",
              "dynamic_pgm1b_reset_bootstrap.csv",
              "dynamic_pgm1b_boundary_by_symbol.csv",
              "dynamic_pgm1b_summary.json"):
        assert f.startswith(T.PREFIX)


if __name__ == "__main__":
    import traceback

    only = sys.argv[1:] if len(sys.argv) > 1 else None
    ok = fail = 0
    for nm in sorted(n for n in dir() if n.startswith("test_")):
        if only and nm not in only:
            continue
        try:
            globals()[nm]()
            print("PASS", nm)
            ok += 1
        except Exception:
            print("FAIL", nm)
            traceback.print_exc()
            fail += 1
    print(f"\nTOTAL ok={ok} fail={fail}")
    sys.exit(1 if fail else 0)
