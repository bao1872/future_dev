"""Tests for DYNAMIC-PGM-1A.2c Representation Control.

Covers:
  * deterministic Z_t recoverability audit: 16 = 10 re-encoding + 6 innovation
  * re-encoding columns are exactly the frozen encoding of an S_t field
  * z_d_up is re-encoding via the frozen last_return = -z_d_up identity
  * MFE / MAE / count increments are TRUE_HISTORY_INNOVATION (need S_{t-1})
  * classification is a partition of ZT_COLS (no overlap, no omission)
  * MC uses only re-encoding columns; MMEM only innovation columns
  * all models share one state-independent constant ZTP magnitude, same rows
  * MZ-MC count delta == occurrence delta
  * bootstrap emits all six comparisons; verdict reads only MZ-MC
  * six-model synthetic smoke
  * frozen parity targets match committed 1A.1b / 1A.2 / 1A.2b outputs
"""
import inspect
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
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2_lag_closure_v1 as lag  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2b_innovation_attribution_v1 as prev  # noqa: E402
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2c_representation_control_v1 as T  # noqa: E402

OUT = base.OUT


# --------------------------------------------------------------------------- #
def _consistent_small_frame():
    """6 rows / 2 episodes where every frozen identity HOLDS, so the
    deterministic recoverability audit can be exercised."""
    n = 6
    ep = [0, 0, 0, 1, 1, 1]
    bar = [0, 1, 2, 0, 1, 2]
    dcr = np.array([0.0, 0.25, 1.0, 0.5, 0.0, 0.75])
    rng = np.array([0.0, 2.0, 3.0, 1.5, 0.0, 4.0])
    ures = np.array([-0.5, 0.0, -1.25, -0.2, -0.7, 0.0])
    lres = np.array([0.0, -0.3, -0.8, -1.1, 0.0, -0.4])
    mfe = np.array([1.0, 1.5, 1.5, 2.0, 2.5, 3.0])
    mae = np.array([0.5, 0.5, 0.9, 1.1, 1.1, 1.4])
    cu = np.array([5., 5., 7., 3., 3., 4.])
    cl = np.array([2., 4., 4., 1., 1., 1.])
    zdup = np.array([0.1, -0.2, 0.3, 0.4, -0.5, 0.6])

    # frozen invariant: S_t.last_return == -z_d_up(t-1 -> t)
    last_ret = np.zeros(n)
    last_ret[1], last_ret[2] = -zdup[0], -zdup[1]
    last_ret[4], last_ret[5] = -zdup[3], -zdup[4]

    def nxt_arr(a):
        b = np.zeros(n)
        b[0], b[1], b[3], b[4] = a[1], a[2], a[4], a[5]
        return b

    def dlt(a):
        b = np.zeros(n)
        b[0], b[1] = a[1] - a[0], a[2] - a[1]
        b[3], b[4] = a[4] - a[3], a[5] - a[4]
        return b

    data = dict(episode_id=ep, bar_t=bar,
                path_last_return_R=last_ret,
                path_direction_change_rate=dcr,
                path_current_bar_range_R=rng,
                upper_newest_log_age_residual=ures,
                lower_newest_log_age_residual=lres,
                path_max_up_excursion_R=mfe,
                path_max_down_excursion_R=mae,
                upper_active_identity_count_delta=cu,
                lower_active_identity_count_delta=cl)
    for node, raw in (("z_d_up", zdup), ("z_dcr", nxt_arr(dcr)),
                      ("z_range", nxt_arr(rng)), ("z_uresid", nxt_arr(ures)),
                      ("z_lresid", nxt_arr(lres)), ("z_dmfe", dlt(mfe)),
                      ("z_dmae", dlt(mae))):
        for k, v in T._encode_with_frozen(node, raw).items():
            data[k] = v
    # count increments are NOT in NODE_SPECS and are stored unencoded
    data["z_delta_upper_count"] = dlt(cu)
    data["z_delta_lower_count"] = dlt(cl)
    for c in base.OBS_STATE_NUM:
        data.setdefault(c, np.zeros(n))
    for c in base.OBS_STATE_CAT:
        data[c] = ["a"] * n
    for c in lag.LAG_BASE:
        data.setdefault(c, np.zeros(n))
    data[base.DISC_Z] = np.zeros(n, dtype=np.int64)
    data["block"] = ["TB1"] * 3 + ["TB2"] * 3
    data["symbol"] = ["S0"] * 6
    data["episode_start_day"] = ["2026-01-01"] * 6
    df = lag.add_lag1_causal(pd.DataFrame(data))
    return prev.add_zt_causal(df)


def test_zt_recoverability_audit():
    df = _consistent_small_frame()
    cls, rep = T.audit_zt_recoverability(df)
    assert rep["n_zt_columns"] == 16
    assert rep["n_current_state_reencoding"] == 10
    assert rep["n_true_history_innovation"] == 6
    assert len(T.ZT_CUR_COLS) == 10 and len(T.ZT_MEM_COLS) == 6
    # partition
    assert sorted(T.ZT_CUR_COLS + T.ZT_MEM_COLS) == sorted(T.ZT_COLS)
    assert not (set(T.ZT_CUR_COLS) & set(T.ZT_MEM_COLS))


def test_reenencoding_and_innovation_members():
    # z_d_up is re-encoding: frozen last_return = -z_d_up identity
    assert "zt_z_d_up" in T.ZT_CUR_COLS
    # value nodes are re-encodings of S_t fields
    for c in ("zt_z_dcr_is0", "zt_z_dcr_logit", "zt_z_range_ispos",
              "zt_z_uresid_log", "zt_z_lresid_log"):
        assert c in T.ZT_CUR_COLS
    # cumulative-max increments and count increments are genuine memory
    for c in ("zt_z_dmfe_ispos", "zt_z_dmfe_log", "zt_z_dmae_log",
              "zt_z_delta_upper_count", "zt_z_delta_lower_count"):
        assert c in T.ZT_MEM_COLS


def test_control_model_design_matrices():
    """MC carries only re-encoding cols; MMEM only innovation cols."""
    assert T.MC_EXTRA == [T.LAG_AVAIL] + T.ZT_CUR_COLS
    assert T.MMEM_EXTRA == [T.LAG_AVAIL] + T.ZT_MEM_COLS
    assert len(T.MC_EXTRA) == 11 and len(T.MMEM_EXTRA) == 7
    assert not (set(T.MC_EXTRA) & set(T.ZT_MEM_COLS))
    assert not (set(T.MMEM_EXTRA) & set(T.ZT_CUR_COLS))
    # neither may carry a previous-bar STATE value
    for extra in (T.MC_EXTRA, T.MMEM_EXTRA):
        assert not (set(extra) & set(lag.LAG_COLS))
    # MZ is the union of the two halves
    assert set(T.MZ_EXTRA) == set(T.MC_EXTRA) | set(T.MMEM_EXTRA)


# --------------------------------------------------------------------------- #
def _random_frame(n_ep=30, bars=20, seed=13):
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
    for c in lag.LAG_BASE:
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
    df = lag.add_lag1_causal(pd.DataFrame(data))
    return prev.add_zt_causal(df)


def test_six_models_shared_ztp_and_same_rows():
    df = _random_frame()
    tr = df[df["block"] == "TB1"].reset_index(drop=True)
    ev = df[df["block"] == "TB2"].reset_index(drop=True)
    Ytr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yev = ev[base.COUNT_Z].to_numpy(np.int64)
    obs = list(base.OBS_STATE_NUM)
    k0c = base.fit_constant_count_head(Ytr, Yev)
    seen = []
    for tag, extra in [("M0", T.M0_EXTRA), ("MF", T.MF_EXTRA),
                       ("MZ", T.MZ_EXTRA), ("MC", T.MC_EXTRA),
                       ("MMEM", T.MMEM_EXTRA)]:
        ct = lag._make_ct(obs + list(extra))
        Xtr = ct.fit_transform(tr).astype(np.float32)
        Xev = ct.transform(ev).astype(np.float32)
        assert Xev.shape[0] == len(ev)
        kc = base.fit_state_count_head(Xtr, Ytr, Xev, Yev,
                                       constant_rates=k0c["constant_rates"])
        seen.append(kc["rate_ev"])
        for j in range(len(base.COUNT_Z)):
            assert np.array_equal(k0c["rate_ev"][:, j], kc["rate_ev"][:, j])
            assert len(np.unique(kc["rate_ev"][:, j])) == 1
            assert len(np.unique(kc["p0_ev"][:, j])) > 1
    for r in seen[1:]:
        assert np.array_equal(seen[0], r)


def test_mz_minus_mc_count_equals_occurrence():
    df = _random_frame()
    tr = df[df["block"] == "TB1"].reset_index(drop=True)
    ev = df[df["block"] == "TB2"].reset_index(drop=True)
    Ytr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yev = ev[base.COUNT_Z].to_numpy(np.int64)
    obs = list(base.OBS_STATE_NUM)
    k0c = base.fit_constant_count_head(Ytr, Yev)
    out = {}
    for tag, extra in [("M0", T.M0_EXTRA), ("MZ", T.MZ_EXTRA),
                       ("MC", T.MC_EXTRA)]:
        ct = lag._make_ct(obs + list(extra))
        Xtr = ct.fit_transform(tr).astype(np.float32)
        Xev = ct.transform(ev).astype(np.float32)
        out[tag] = base.fit_state_count_head(
            Xtr, Ytr, Xev, Yev, constant_rates=k0c["constant_rates"])
    for a, b in (("MZ", "M0"), ("MZ", "MC")):
        d_cnt = out[a]["nll_ev"] - out[b]["nll_ev"]
        d_occ = (lag._occ_nll(Yev, out[a]["p0_ev"])
                 - lag._occ_nll(Yev, out[b]["p0_ev"]))
        assert np.max(np.abs(d_cnt - d_occ)) < 1e-10


def test_synthetic_six_model_smoke():
    df = _random_frame()
    base.configure_child_semantics(agezero_deterministic=True)
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.parquet"
            df.to_parquet(p, index=False)
            w = dict(name="SYNTH", train=["TB1"], eval="TB2", seed=5)
            res = T.run_single_window_1a2c(w, str(p))
    finally:
        base.configure_child_semantics(agezero_deterministic=False)

    models = {m["model"] for m in res["model_metrics"]}
    assert models == {T.MODEL_K0, T.MODEL_K1, T.MODEL_M0, T.MODEL_MF,
                      T.MODEL_MZ, T.MODEL_MC, T.MODEL_MMEM}
    for m in res["model_metrics"]:
        assert np.isfinite(m["mean_joint_nll"])
    assert len({m["n_rows"] for m in res["model_metrics"]}) == 1
    assert all(r["success"] for r in res["opt_rows"])
    comps = {b["comparison"] for b in res["boots"]}
    assert comps == {T.C_REF, T.C_FULL, T.C_INNOV, T.C_RESID, T.C_REENC,
                     T.C_MEM}
    for r in res["count_occ_rows"]:
        assert r["state_dependent_magnitude"] is False
        assert r["count_minus_occurrence_residual"] < 1e-10
    assert len(res["decom_rows"]) == 1
    assert json.loads(json.dumps(res, default=str))


def test_verdict_reads_only_primary_mz_minus_mc():
    assert T.C_MEM == "MZ-MC" and T.C_REENC == "MC-M0"
    src = inspect.getsource(T.main)
    v = src[src.index("# ---------------- verdict"):]
    assert '_passes(w["name"], C_MEM)' in v
    gate_lines = [l for l in v.splitlines() if "verdict =" in l]
    assert gate_lines
    assert all("C_REENC" not in l for l in gate_lines)
    assert all("C_INNOV" not in l for l in gate_lines)


def test_frozen_parity_targets_match_committed_outputs():
    d = OUT
    p1 = d / "dynamic_pgm1a1b_model_metrics.csv"
    if p1.exists():
        for _, r in pd.read_csv(p1).iterrows():
            f = T.FROZEN_1A1B_JOINT[r["window"]][r["model"]]
            assert abs(float(r["mean_joint_nll"]) - f) < 1e-12
    p2 = d / "dynamic_pgm1a2_model_metrics.csv"
    if p2.exists():
        for _, r in pd.read_csv(p2).iterrows():
            if r["model"] in ("K1A_STATE_AVAIL", "K2_STATE_LAG1"):
                f = T.FROZEN_1A2_JOINT[r["window"]][r["model"]]
                assert abs(float(r["mean_joint_nll"]) - f) < 1e-12
    p3 = d / "dynamic_pgm1a2b_model_metrics.csv"
    if p3.exists():
        for _, r in pd.read_csv(p3).iterrows():
            if r["model"] == "MZ_STATE_INNOV":
                f = T.FROZEN_1A2B_JOINT[r["window"]][r["model"]]
                assert abs(float(r["mean_joint_nll"]) - f) < 1e-12


def test_output_namespace_isolation():
    assert T.PREFIX == "dynamic_pgm1a2c"
    assert T.BASE_SHA == "1089fd1622b4b9de7743a8180a40f5d89176c8c4"
    assert T.ZT_COLS == prev.ZT_COLS
    assert T.FROZEN_Z_FEATURE_HASH == lag.FROZEN_Z_FEATURE_HASH


if __name__ == "__main__":
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
