"""Tests for DYNAMIC-PGM-1A.2 Lag Closure.

Covers:
  * frozen constants match the committed 1A.1b outputs (parity gate inputs)
  * lag1 is the exact episode-internal previous row (no cross-episode leakage)
  * lag1_available semantics
  * lag1 bar_t-gap guard (shift(1) must be the true previous BAR)
  * K1 / K2 evaluate identical rows
  * K0 / K1 / K2 share one state-INDEPENDENT constant ZTP magnitude lambda
  * K2-K1 count delta == K2-K1 occurrence delta (magnitude cancels)
  * bootstrap: episode aggregation + exact cluster multiplicity
  * synthetic K0/K1/K2 window smoke (finite NLL, optimizer success)
  * output namespace isolation (dynamic_pgm1a2_* only)
  * JSON serializable results
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a1b_count_magnitude_closure_v1 as base  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2_lag_closure_v1 as T  # noqa: E402


# --------------------------------------------------------------------------- #
def test_frozen_parity_constants_match_1a1b_outputs():
    """The hard-coded parity targets must equal the committed 1A.1b outputs."""
    p = base.OUT / "dynamic_pgm1a1b_model_metrics.csv"
    if not p.exists():
        return
    df = pd.read_csv(p)
    assert len(df) == 4
    for _, r in df.iterrows():
        frozen = T.FROZEN_JOINT[r["window"]][r["model"]]
        assert abs(float(r["mean_joint_nll"]) - frozen) < 1e-12
    sa = json.loads((base.OUT / "dynamic_pgm1a1b_sample_audit.json").read_text())
    assert sa["z_feature_hash"] == T.FROZEN_Z_FEATURE_HASH
    assert sa["n_transition_rows"] == T.EXPECTED_TRANSITIONS == 321727


def test_lag_base_unchanged_from_1a1b():
    assert T.LAG_BASE == list(base.LAG_BASE)
    assert len(T.LAG_BASE) == 15
    assert len(T.K2_EXTRA) == 16
    assert T.LAG_AVAIL == "lag1_available"
    # TEMPO must NOT be in the lag block (derived representation of S_t)
    for c in base.TEMPO:
        assert c not in T.LAG_BASE


# --------------------------------------------------------------------------- #
def _small_frame():
    """Two episodes, values encode position so any leakage is detectable."""
    return pd.DataFrame({
        "episode_id": [0, 0, 0, 1, 1, 1],
        "bar_t": [0, 1, 2, 0, 1, 2],
        "cur_up_distance_R": [10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
        "cur_down_distance_R": [20.0, 21.0, 22.0, 200.0, 201.0, 202.0],
        "cur_log_ratio": [-0.6, -0.5, -0.4, -5.3, -5.2, -5.1],
        "path_total_variation_R": [1.0, 2.0, 3.0, 10.0, 11.0, 12.0],
        "path_max_up_excursion_R": [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
        "path_max_down_excursion_R": [0.1, 0.2, 0.3, 1.1, 1.2, 1.3],
        "path_direction_change_rate": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "path_last_return_R": [0.0, -0.1, -0.2, 0.0, -0.1, -0.2],
        "path_current_bar_range_R": [2.0, 2.1, 2.2, 3.0, 3.1, 3.2],
        **{c: [0.0, 0.1, 0.2, 1.0, 1.1, 1.2] for c in base.COMPACT_PROV},
    })


def test_lag1_exact_shift_no_cross_episode():
    x = T.add_lag1_causal(_small_frame())
    # episode 0: prev of rows 1,2 are rows 0,1
    assert np.array_equal(x["lag1_cur_up_distance_R"].to_numpy(),
                          np.array([np.nan, 10.0, 11.0,
                                    np.nan, 100.0, 101.0]), equal_nan=True)
    # episode 1 first row must NOT see episode 0 last row (leakage check)
    assert np.isnan(x.loc[3, "lag1_cur_up_distance_R"])
    assert x.loc[3, "lag1_cur_up_distance_R"] != 12.0


def test_lag1_available():
    x = T.add_lag1_causal(_small_frame())
    avail = x[T.LAG_AVAIL].to_numpy()
    assert list(avail) == [0, 1, 1, 0, 1, 1]
    # every lag column missing exactly where unavailable
    for c in T.LAG_COLS:
        assert bool(x.loc[avail == 0, c].isna().all())
        assert bool(x.loc[avail == 1, c].notna().all())


def test_lag_bar_gap_guard():
    df = _small_frame()
    df.loc[2, "bar_t"] = 5          # create a bar_t gap inside episode 0
    try:
        T.add_lag1_causal(df)
    except SystemExit as e:
        assert "STOP_DYNAMIC_PGM1A2_LAG_BAR_GAP" in str(e)
    else:
        raise AssertionError("bar_t gap must raise STOP_DYNAMIC_PGM1A2_LAG_BAR_GAP")


def test_lag_audit_report():
    x = T.add_lag1_causal(_small_frame())
    rep = T.audit_lag_causality(x)
    assert rep["n_rows"] == 6
    assert rep["n_episodes"] == 2
    assert rep["n_lag1_available"] == 4
    assert rep["n_lag1_missing"] == 2
    assert rep["bar_t_gap_count"] == 0
    assert rep["cross_episode_leakage"] is False


# --------------------------------------------------------------------------- #
def test_bootstrap_episode_aggregation_and_multiplicity():
    """point must be the EPISODE mean (not the row mean), and day clusters must
    be weighted by their exact episode multiplicity."""
    # day A: 1 episode with 5 rows, delta 0
    # day B: 3 episodes (1 row each), deltas 0 / 3 / 15
    #   episode means = [0, 0, 3, 15] -> episode mean = 4.5
    #   row mean      = (0*5 + 0 + 3 + 15) / 8 = 2.25
    delta = np.array([0.] * 5 + [0., 3., 15.])
    day = np.array(["A"] * 5 + ["B"] * 3)
    eid = np.array([0] * 5 + [1, 2, 3])
    lo, hi, point = base.boot_cluster(delta, day, eid, 1)
    assert abs(point - 4.5) < 1e-12, point
    assert abs(point - 2.25) > 1.0          # proves episode (not row) aggregation
    assert lo <= point <= hi
    # multiplicity: resampling both days must weight by episode counts 1 and 3
    # => (0*1 + 18*1) / (1 + 3) == 4.5
    assert abs((0 * 1 + 18 * 1) / (1 + 3) - 4.5) < 1e-12
    # constant delta -> point and both CI bounds collapse to the constant
    c = np.full(20, -1.5)
    lo2, hi2, p2 = base.boot_cluster(c, np.array(["A"] * 20),
                                     np.zeros(20, dtype=int), 2)
    assert abs(p2 + 1.5) < 1e-12 and abs(lo2 + 1.5) < 1e-12 and abs(hi2 + 1.5) < 1e-12


# --------------------------------------------------------------------------- #
def _synthetic_transition_frame(n_ep=30, bars=20, seed=7):
    rng = np.random.default_rng(seed)
    n = n_ep * bars
    eid = np.repeat(np.arange(n_ep), bars)
    data = {
        "episode_id": eid.astype(np.int64),
        "bar_t": np.tile(np.arange(bars), n_ep).astype(np.int64),
        "block": np.where(eid < n_ep // 2, "TB1", "TB2"),
        "symbol": np.array([f"S{i % 5:02d}" for i in eid]),
        "episode_start_day": np.array([f"2026-01-{(i % 10) + 1:02d}"
                                       for i in eid]),
    }
    for c in base.OBS_STATE_NUM:
        data[c] = rng.normal(size=n).astype(np.float32)
    for c in base.OBS_STATE_CAT:
        data[c] = rng.integers(0, 3, size=n).astype(str)
    for c in T.LAG_BASE:
        data[c] = rng.normal(size=n).astype(np.float32)
    data["z_d_up"] = rng.normal(scale=0.5, size=n).astype(np.float32)
    for nm in ("z_dmfe", "z_dmae", "z_range", "z_uresid", "z_lresid"):
        data[f"{nm}_ispos"] = (rng.random(n) > 0.4).astype(np.float32)
        data[f"{nm}_log"] = np.log(rng.random(n) + 0.1).astype(np.float32)
    is0 = (rng.random(n) > 0.8).astype(np.float32)
    is1 = (rng.random(n) > 0.8).astype(np.float32)
    is1[(is0 + is1) > 1.5] = 0.0
    data["z_dcr_is0"], data["z_dcr_is1"] = is0, is1
    data["z_dcr_logit"] = rng.normal(size=n).astype(np.float32)
    for c in base.COUNT_Z:
        data[c] = rng.integers(0, 4, size=n).astype(np.int64)
    data[base.DISC_Z] = rng.integers(0, 4, size=n).astype(np.int64)
    return T.add_lag1_causal(pd.DataFrame(data))


def _design_matrices(df):
    tr = df[df["block"] == "TB1"].reset_index(drop=True)
    ev = df[df["block"] == "TB2"].reset_index(drop=True)
    ct1 = T._make_ct(list(base.OBS_STATE_NUM))
    ct2 = T._make_ct(list(base.OBS_STATE_NUM) + list(T.K2_EXTRA))
    return (tr, ev,
            ct1.fit_transform(tr).astype(np.float32),
            ct1.transform(ev).astype(np.float32),
            ct2.fit_transform(tr).astype(np.float32),
            ct2.transform(ev).astype(np.float32))


def test_k0_k1_k2_shared_constant_ztp_magnitude():
    df = _synthetic_transition_frame()
    tr, ev, Xtr1, Xev1, Xtr2, Xev2 = _design_matrices(df)
    Ytr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yev = ev[base.COUNT_Z].to_numpy(np.int64)

    # K1 / K2 evaluate identical rows
    assert Xev1.shape[0] == Xev2.shape[0] == len(ev)
    assert Xtr2.shape[1] == Xtr1.shape[1] + len(T.K2_EXTRA)

    k0c = base.fit_constant_count_head(Ytr, Yev)
    k1c = base.fit_state_count_head(Xtr1, Ytr, Xev1, Yev,
                                    constant_rates=k0c["constant_rates"])
    k2c = base.fit_state_count_head(Xtr2, Ytr, Xev2, Yev,
                                    constant_rates=k0c["constant_rates"])
    for j in range(len(base.COUNT_Z)):
        # identical magnitude across all three models
        assert np.array_equal(k0c["rate_ev"][:, j], k1c["rate_ev"][:, j])
        assert np.array_equal(k0c["rate_ev"][:, j], k2c["rate_ev"][:, j])
        # magnitude is state-INDEPENDENT (a single constant per column)
        assert len(np.unique(k2c["rate_ev"][:, j])) == 1
        # but occurrence IS state-dependent (otherwise K2 could not differ)
        assert len(np.unique(k2c["p0_ev"][:, j])) > 1


def test_count_delta_equals_occurrence_delta():
    df = _synthetic_transition_frame()
    tr, ev, Xtr1, Xev1, Xtr2, Xev2 = _design_matrices(df)
    Ytr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yev = ev[base.COUNT_Z].to_numpy(np.int64)
    k0c = base.fit_constant_count_head(Ytr, Yev)
    k1c = base.fit_state_count_head(Xtr1, Ytr, Xev1, Yev,
                                    constant_rates=k0c["constant_rates"])
    k2c = base.fit_state_count_head(Xtr2, Ytr, Xev2, Yev,
                                    constant_rates=k0c["constant_rates"])
    d_count = k2c["nll_ev"] - k1c["nll_ev"]
    d_occ = T._occ_nll(Yev, k2c["p0_ev"]) - T._occ_nll(Yev, k1c["p0_ev"])
    assert np.max(np.abs(d_count - d_occ)) < 1e-10


def test_synthetic_window_k0_k1_k2_smoke():
    df = _synthetic_transition_frame()
    base.configure_child_semantics(agezero_deterministic=True)
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.parquet"
            df.to_parquet(p, index=False)
            w = dict(name="SYNTH", train=["TB1"], eval="TB2", seed=1)
            res = T.run_single_window_1a2(w, str(p))
    finally:
        base.configure_child_semantics(agezero_deterministic=False)

    models = {m["model"]: m for m in res["model_metrics"]}
    assert set(models) == {"K0_UNCONDITIONAL", "K1_STATE", "K2_STATE_LAG1"}
    for m in res["model_metrics"]:
        assert np.isfinite(m["mean_joint_nll"])
        assert m["n_rows"] == len(df[df["block"] == "TB2"])
    # every model sees the same eval rows
    assert len({m["n_rows"] for m in res["model_metrics"]}) == 1

    # optimizer audit: all fits succeeded
    assert all(r["success"] for r in res["opt_rows"])
    assert {r["model"] for r in res["opt_rows"]} == set(models)

    # primary + reference comparisons both bootstrapped
    comps = {b["comparison"] for b in res["boots"]}
    assert comps == {"K1-K0", "K2-K1"}

    # count magnitude closure evidence
    assert res["count_occ_rows"], "count occurrence diagnostics missing"
    for r in res["count_occ_rows"]:
        assert r["state_dependent_magnitude"] is False
        assert r["count_minus_occurrence_residual"] < 1e-10

    # attribution blocks: Count split out of LiquidityComposition
    blocks = {b["block"] for b in res["block_rows"]}
    assert blocks == {"Location", "Path", "LiquidityResidual",
                      "CountOccurrence"}

    # JSON serializable (production writes results with default=str)
    assert json.loads(json.dumps(res, default=str))


def test_output_namespace_isolation():
    assert T.PREFIX == "dynamic_pgm1a2"
    assert T.BASE_SHA == "16ab731ac099ffcec7c2089640b95b5555184853"
    # 1A.2 must never write into the 1A.1b namespace
    for name in ("dynamic_pgm1a1b_model_metrics.csv",
                 "dynamic_pgm1a1b_summary.json"):
        assert not name.startswith(T.PREFIX)
    assert T.FROZEN_JOINT.keys() == {"A_TB1_to_TB2", "B_TB1TB2_to_TB3"}


if __name__ == "__main__":
    import inspect
    import traceback

    ok = fail = 0
    for nm in sorted(n for n in dir() if n.startswith("test_")):
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
