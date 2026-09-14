"""Tests for DYNAMIC-PGM-1A.2b Innovation Attribution.

Covers:
  * Z_t is the exact episode-internal previous row's z vector (no leakage)
  * Z_t availability agrees with lag1_available
  * Z_t bar_t-gap guard
  * MZ block is 14 innovation columns (+availability); MF is 15 lag (+avail)
  * M0 / MF / MZ evaluate identical rows
  * all models share one state-independent constant ZTP magnitude
  * MZ-M0 count delta == occurrence delta
  * bootstrap produces K1-K0 / MF-M0 / MZ-M0 / MF-MZ
  * verdict reads only the two gates (MZ-M0, MF-MZ)
  * frozen parity targets equal the committed 1A.1b / 1A.2 outputs
  * six-model synthetic smoke (finite NLL, all fits succeed)
  * output namespace isolation (dynamic_pgm1a2b_*)
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
import research.liquidity_oracle_atlas.experiment_dynamic_pgm1a2b_innovation_attribution_v1 as T  # noqa: E402

OUT = base.OUT


# --------------------------------------------------------------------------- #
def test_frozen_parity_targets_match_committed_outputs():
    """Hard-coded parity targets must equal the committed 1A.1b / 1A.2 CSVs."""
    p1 = OUT / "dynamic_pgm1a1b_model_metrics.csv"
    if p1.exists():
        for _, r in pd.read_csv(p1).iterrows():
            frozen = T.FROZEN_1A1B_JOINT[r["window"]][r["model"]]
            assert abs(float(r["mean_joint_nll"]) - frozen) < 1e-12
    p2 = OUT / "dynamic_pgm1a2_model_metrics.csv"
    if p2.exists():
        d = pd.read_csv(p2)
        for _, r in d.iterrows():
            if r["model"] in ("K1A_STATE_AVAIL", "K2_STATE_LAG1"):
                frozen = T.FROZEN_1A2_JOINT[r["window"]][r["model"]]
                assert abs(float(r["mean_joint_nll"]) - frozen) < 1e-12


def test_zt_block_definition():
    assert len(T.ZT_COLS) == len(base.ALL_Z_COLS) == 14
    assert all(c.startswith("zt_") for c in T.ZT_COLS)
    # M0 / MF / MZ nesting: MF = M0 + 15 lag values, MZ = M0 + 14 innovations
    assert T.M0_EXTRA == [T.LAG_AVAIL]
    assert len(T.MF_EXTRA) == 1 + len(lag.LAG_COLS) == 16
    assert len(T.MZ_EXTRA) == 1 + len(T.ZT_COLS) == 15
    # MZ is more compact than the full lag block
    assert len(T.MZ_EXTRA) < len(T.MF_EXTRA)
    # explanatory split covers every innovation column exactly once
    assert sorted(T.PATH_ZT_COLS + T.NONPATH_ZT_COLS) == sorted(T.ZT_COLS)
    assert not (set(T.PATH_ZT_COLS) & set(T.NONPATH_ZT_COLS))
    # MZ must never carry a previous-bar STATE value
    assert not (set(T.MZ_EXTRA) & set(lag.LAG_COLS))


# --------------------------------------------------------------------------- #
def _small_frame():
    # 6 rows, 2 episodes. Includes the LAG_BASE columns so that
    # lag.add_lag1_causal() can also run on this frame.
    d = {
        "episode_id": [0, 0, 0, 1, 1, 1],
        "bar_t": [0, 1, 2, 0, 1, 2],
    }
    for c in lag.LAG_BASE:
        d[c] = [10.0, 11.0, 12.0, 100.0, 101.0, 102.0]
    d.update({
        "z_d_up": [1.0, 2.0, 3.0, 7.0, 8.0, 9.0],
        "z_dmfe_ispos": [0., 1., 1., 0., 1., 0.],
        "z_dmfe_log": [0.0, 0.5, 0.6, 0.0, 0.7, 0.0],
        "z_dmae_ispos": [1., 0., 1., 1., 0., 1.],
        "z_dmae_log": [0.2, 0.0, 0.3, 0.4, 0.0, 0.5],
        "z_dcr_is0": [1., 0., 0., 1., 0., 0.],
        "z_dcr_is1": [0., 0., 1., 0., 0., 1.],
        "z_dcr_logit": [0.0, 0.1, 0.2, 0.0, 0.3, 0.4],
        "z_range_ispos": [1., 1., 1., 1., 1., 1.],
        "z_range_log": [0.7, 0.8, 0.9, 1.1, 1.2, 1.3],
        "z_uresid_ispos": [0., 0., 1., 0., 1., 0.],
        "z_uresid_log": [0.0, 0.0, 0.4, 0.0, 0.6, 0.0],
        "z_lresid_ispos": [0., 1., 0., 1., 0., 1.],
        "z_lresid_log": [0.0, 0.9, 0.0, 0.8, 0.0, 0.7],
    })
    return pd.DataFrame(d)


def test_zt_is_previous_row_no_leakage():
    x = T.add_zt_causal(_small_frame())
    # row1's Z_t is row0's z vector; row3 (episode start) must have none
    assert x.loc[1, "zt_z_d_up"] == 1.0
    assert x.loc[2, "zt_z_d_up"] == 2.0
    assert np.isnan(x.loc[3, "zt_z_d_up"])
    # episode 1 first row must NOT see episode 0 last row
    assert x.loc[3, "zt_z_d_up"] != 3.0
    assert x.loc[4, "zt_z_d_up"] == 7.0


def test_zt_availability_matches_lag1():
    x = lag.add_lag1_causal(_small_frame())
    x = T.add_zt_causal(x)
    avail = x[T.LAG_AVAIL].to_numpy().astype(bool)
    assert list(avail) == [False, True, True, False, True, True]
    for c in T.ZT_COLS:
        assert bool(x.loc[avail, c].notna().all())
        assert bool(x.loc[~avail, c].isna().all())
    rep = T.audit_zt(x)
    assert rep["n_zt_available"] == 4 and rep["n_zt_missing"] == 2
    assert rep["bar_t_gap_count"] == 0
    assert rep["cross_episode_leakage"] is False


def test_zt_bar_gap_guard():
    df = _small_frame()
    df.loc[2, "bar_t"] = 7
    try:
        T.add_zt_causal(df)
    except SystemExit as e:
        assert "STOP_DYNAMIC_PGM1A2B_ZT_BAR_GAP" in str(e)
    else:
        raise AssertionError("bar_t gap must be rejected")


# --------------------------------------------------------------------------- #
def _synthetic_frame(n_ep=30, bars=20, seed=11):
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
    return T.add_zt_causal(df)


def test_six_models_shared_ztp_and_same_rows():
    df = _synthetic_frame()
    tr = df[df["block"] == "TB1"].reset_index(drop=True)
    ev = df[df["block"] == "TB2"].reset_index(drop=True)
    Ytr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yev = ev[base.COUNT_Z].to_numpy(np.int64)
    obs = list(base.OBS_STATE_NUM)
    k0c = base.fit_constant_count_head(Ytr, Yev)
    rates, ncols = [], {}
    for tag, extra in [("M0", T.M0_EXTRA), ("MF", T.MF_EXTRA),
                       ("MZ", T.MZ_EXTRA), ("MZP", T.MZP_EXTRA),
                       ("MZN", T.MZN_EXTRA)]:
        ct = lag._make_ct(obs + list(extra))
        Xtr = ct.fit_transform(tr).astype(np.float32)
        Xev = ct.transform(ev).astype(np.float32)
        # identical eval rows across every model
        assert Xev.shape[0] == len(ev)
        ncols[tag] = Xtr.shape[1]
        kc = base.fit_state_count_head(Xtr, Ytr, Xev, Yev,
                                       constant_rates=k0c["constant_rates"])
        rates.append(kc["rate_ev"])
        for j in range(len(base.COUNT_Z)):
            assert np.array_equal(k0c["rate_ev"][:, j], kc["rate_ev"][:, j])
            assert len(np.unique(kc["rate_ev"][:, j])) == 1
            assert len(np.unique(kc["p0_ev"][:, j])) > 1
    # all share one identical magnitude
    for r in rates[1:]:
        assert np.array_equal(rates[0], r)
    # nested dimensions
    assert ncols["MF"] == ncols["M0"] + len(lag.LAG_COLS)
    assert ncols["MZ"] == ncols["M0"] + len(T.ZT_COLS)
    # the two explanatory halves add back to the full innovation block
    assert ((ncols["MZP"] - ncols["M0"]) + (ncols["MZN"] - ncols["M0"])
            == len(T.ZT_COLS))


def test_mz_minus_m0_count_equals_occurrence():
    df = _synthetic_frame()
    tr = df[df["block"] == "TB1"].reset_index(drop=True)
    ev = df[df["block"] == "TB2"].reset_index(drop=True)
    Ytr = tr[base.COUNT_Z].to_numpy(np.int64)
    Yev = ev[base.COUNT_Z].to_numpy(np.int64)
    obs = list(base.OBS_STATE_NUM)
    k0c = base.fit_constant_count_head(Ytr, Yev)
    out = {}
    for tag, extra in [("M0", T.M0_EXTRA), ("MZ", T.MZ_EXTRA)]:
        ct = lag._make_ct(obs + list(extra))
        Xtr = ct.fit_transform(tr).astype(np.float32)
        Xev = ct.transform(ev).astype(np.float32)
        out[tag] = base.fit_state_count_head(
            Xtr, Ytr, Xev, Yev, constant_rates=k0c["constant_rates"])
    d_count = out["MZ"]["nll_ev"] - out["M0"]["nll_ev"]
    d_occ = (lag._occ_nll(Yev, out["MZ"]["p0_ev"])
             - lag._occ_nll(Yev, out["M0"]["p0_ev"]))
    assert np.max(np.abs(d_count - d_occ)) < 1e-10


def test_synthetic_six_model_smoke():
    df = _synthetic_frame()
    base.configure_child_semantics(agezero_deterministic=True)
    try:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "t.parquet"
            df.to_parquet(p, index=False)
            w = dict(name="SYNTH", train=["TB1"], eval="TB2", seed=3)
            res = T.run_single_window_1a2b(w, str(p))
    finally:
        base.configure_child_semantics(agezero_deterministic=False)

    models = {m["model"] for m in res["model_metrics"]}
    assert models == {T.MODEL_K0, T.MODEL_K1, T.MODEL_M0, T.MODEL_MF,
                      T.MODEL_MZ, T.MODEL_MZP, T.MODEL_MZN}
    for m in res["model_metrics"]:
        assert np.isfinite(m["mean_joint_nll"])
        assert m["n_rows"] == len(df[df["block"] == "TB2"])
    assert len({m["n_rows"] for m in res["model_metrics"]}) == 1
    assert all(r["success"] for r in res["opt_rows"])

    # all four comparisons bootstrapped
    assert {b["comparison"] for b in res["boots"]} == {T.C_REF, T.C_FULL,
                                                       T.C_INNOV, T.C_RESID}
    assert {b["comparison"] for b in res["bysym"]} == {T.C_REF, T.C_FULL,
                                                       T.C_INNOV, T.C_RESID}
    # count closure evidence
    for r in res["count_occ_rows"]:
        assert r["state_dependent_magnitude"] is False
        assert r["count_minus_occurrence_residual"] < 1e-10
    # explanatory innovation split exists and is finite
    assert len(res["innov_rows"]) == 1
    assert np.isfinite(res["innov_rows"][0]["retention_abs_ratio"])
    assert json.loads(json.dumps(res, default=str))


def test_verdict_reads_only_two_gates():
    assert T.C_INNOV == "MZ-M0" and T.C_RESID == "MF-MZ"
    src = inspect.getsource(T.main)
    v = src[src.index("# ---------------- verdict"):]
    assert "_passes(w[\"name\"], C_INNOV)" in v
    assert "_passes(w[\"name\"], C_RESID)" in v
    # the explanatory ratio must never gate
    gate_lines = [l for l in v.splitlines() if "verdict =" in l]
    assert gate_lines and all("retention" not in l for l in gate_lines)


def test_output_namespace_isolation():
    assert T.PREFIX == "dynamic_pgm1a2b"
    assert T.BASE_SHA == "b12778e1159e322fc9f2de6937a00eb0b1bf6260"
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
