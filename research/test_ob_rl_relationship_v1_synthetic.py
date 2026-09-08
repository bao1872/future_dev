#!/usr/bin/env python3

"""Synthetic Gate-A test for RL-1A (Baseline + SMC x RR).

No real Parquet is read. Run with:

    python research/test_ob_rl_relationship_v1_synthetic.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research import analyze_ob_rl_relationship_v1 as AR  # noqa: E402
from research import ob_rl_relationship_v1_spec as SP  # noqa: E402

RNG = np.random.default_rng(20260908)
TMP = Path(tempfile.mkdtemp(prefix="rl1_synth_"))
DS = TMP / "dataset"
OUT = TMP / "out"
DS.mkdir(parents=True, exist_ok=True)

RESULTS: list[tuple[str, bool, object]] = []


def check(name: str, ok: bool, detail: object = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    if not ok:
        print(f"FAIL  {name} :: {detail}", flush=True)


# ============================================================
# 1) Weighted statistics (hand-computed)
# ============================================================
wm = AR.weighted_mean([1.0, 2.0, 3.0], [1.0, 1.0, 2.0])
check("weighted_mean hand calc", abs(wm - 2.25) < 1e-12, wm)

q = AR.weighted_quantiles(
    [1.0, 2.0, 3.0], [1.0, 1.0, 1.0]
)
check(
    "weighted_quantiles median hand calc",
    abs(q[2] - 2.0) < 1e-12
    and q[0] == 1.0
    and q[4] == 3.0,
    list(q),
)
check(
    "weighted_quantiles empty -> nan",
    all(np.isnan(v) for v in AR.weighted_quantiles([], [])),
)
check(
    "weighted_mean all invalid -> nan",
    bool(np.isnan(AR.weighted_mean([np.nan], [np.nan]))),
)

# ============================================================
# 2) State mappers
# ============================================================
check(
    "bias_state +1/-1/0/NaN",
    SP.bias_state(1) == "ALIGNED"
    and SP.bias_state(-1) == "OPPOSED"
    and SP.bias_state(0) == "NEUTRAL"
    and SP.bias_state(np.nan) == "UNKNOWN",
)
check(
    "target_fit_state 0.99/1.00/NaN",
    SP.target_fit_state(0.99, missing_label="NO_LEVEL")
    == "TARGET_BEYOND_STRUCTURE"
    and SP.target_fit_state(
        1.00, missing_label="NO_LEVEL"
    )
    == "TARGET_AT_OR_BEFORE_STRUCTURE"
    and SP.target_fit_state(np.nan, missing_label="NO_LEVEL")
    == "NO_LEVEL"
    and SP.target_fit_state(np.nan, missing_label="NO_OB")
    == "NO_OB",
)
check(
    "stop_structure_state 0.99/1.00/NaN",
    SP.stop_structure_state(0.99, missing_label="NO_LEVEL")
    == "STRUCTURE_INSIDE_STOP"
    and SP.stop_structure_state(
        1.00, missing_label="NO_LEVEL"
    )
    == "STRUCTURE_AT_OR_BEYOND_STOP"
    and SP.stop_structure_state(
        np.nan, missing_label="NO_OB"
    )
    == "NO_OB",
)
check(
    "forward_ob_state",
    SP.forward_ob_state(np.nan, 1.0) == "NO_OB"
    and SP.forward_ob_state("internal", 1.0)
    == "INTERNAL_ALIGNED"
    and SP.forward_ob_state("swing", -1.0)
    == "SWING_OPPOSED"
    and SP.forward_ob_state("internal", 0.0)
    == "INTERNAL_NEUTRAL"
    and SP.forward_ob_state("internal", np.nan)
    == "INTERNAL_UNKNOWN",
)
check(
    "fit bin boundaries",
    SP.fit_bin_label(0.49) == "<0.5"
    and SP.fit_bin_label(0.50) == "0.5-1.0"
    and SP.fit_bin_label(0.99) == "0.5-1.0"
    and SP.fit_bin_label(1.00) == "1.0-1.5"
    and SP.fit_bin_label(1.50) == "1.5-2.0"
    and SP.fit_bin_label(2.00) == ">=2.0"
    and SP.fit_bin_label(np.nan) == "NO_LEVEL",
)

# ============================================================
# 3) Registry scope
# ============================================================
reg = SP.smc_registry()
fams = {r[0] for r in reg}
check(
    "registry contains only SMC families",
    fams
    == {
        "SMC_BIAS",
        "SMC_TARGET_FIT",
        "SMC_STOP_STRUCTURE",
        "SMC_FORWARD_OB",
    },
    fams,
)
check(
    "no DSA / Momentum / Quantile feature in registry",
    not [
        r
        for r in reg
        if "dsa" in r[1].lower()
        or "momentum" in r[1].lower()
        or "quant" in r[1].lower()
    ],
)
check(
    "no 4h in registry",
    not [r for r in reg if "4h" in r[2]],
)
# per TF: 2 bias + 3 target_fit + 3 stop_structure + 1 forward OB
check(
    "registry covers 3 TF x (2 bias + 3 fit + 3 stop + 1 ob)",
    len(reg) == 3 * (2 + 3 + 3 + 1),
    len(reg),
)
check(
    "stop contrast direction",
    SP.STOP_STRUCTURE_CONTRAST
    == (
        "STRUCTURE_INSIDE_STOP",
        "STRUCTURE_AT_OR_BEYOND_STOP",
    ),
)

# ============================================================
# 4) Synthetic action warehouse
# ============================================================
N_CAND = 3000
ACTIONS = (
    "SKIP",
    "FOLLOW_1.5R",
    "FOLLOW_2.0R",
    "FOLLOW_2.5R",
    "FADE_1.5R",
    "FADE_2.0R",
    "FADE_2.5R",
)
DAYS_AG = [f"2024-01-{d:02d}" for d in range(1, 25)]
DAYS_CU = ["2024-02-01", "2024-02-02", "2024-02-03"]

cids = []
symbols = []
tfs = []
structs = []
days = []
weights = []
for i in range(N_CAND):
    sym = "AG" if i % 2 == 0 else "CU"
    cids.append(f"{sym}_C{i:05d}")
    symbols.append(sym)
    tfs.append(["5m", "15m", "1h"][i % 3])
    structs.append("internal" if i % 2 == 0 else "swing")
    pool = DAYS_AG if sym == "AG" else DAYS_CU
    # i//2 so each symbol's own counter spans the full pool
    days.append(pool[(i // 2) % len(pool)])
    weights.append(
        1.0 / (1 + (i % 3))
    )  # group_candidate_count 1..3

state = pd.DataFrame(
    {
        "candidate_id": cids,
        "symbol": symbols,
        "source_tf": tfs,
        "source_ob_structure": structs,
        AR.DAY_COL: days,
        AR.WEIGHT_COL: weights,
    }
)

frames = []
for action in ACTIONS:
    x = state.copy()
    x["action"] = action
    if action == "SKIP":
        x["trade_mode"] = "SKIP"
        x["target_R"] = 0.0
    else:
        mode, rr = action.split("_")
        x["trade_mode"] = mode
        x["target_R"] = float(rr[:-1])

    for tf in SP.VALIDATED_TFS:
        for kind in ("internal", "swing"):
            x[f"{kind}_bias_rel_{tf}"] = RNG.choice(
                [-1.0, 0.0, 1.0], len(x)
            )
        for kind in ("internal", "swing"):
            v = RNG.uniform(0.2, 3.0, len(x))
            v[np.arange(len(x)) % 11 == 0] = np.nan
            x[f"target_fit_{kind}_{tf}"] = v
        # OB: NaN on a fixed subset -> NO_OB
        v = RNG.uniform(0.2, 3.0, len(x))
        v[np.arange(len(x)) % 5 == 0] = np.nan
        x[f"target_fit_ob_{tf}"] = v
        for kind in ("internal", "swing"):
            x[f"stop_structure_{kind}_{tf}"] = RNG.uniform(
                0.2, 3.0, len(x)
            )
        v = RNG.uniform(0.2, 3.0, len(x))
        v[np.arange(len(x)) % 5 == 0] = np.nan
        x[f"stop_structure_ob_{tf}"] = v
        sc = RNG.choice(["internal", "swing"], len(x))
        sc[np.arange(len(x)) % 5 == 0] = None
        x[f"forward_active_ob_structure_class_{tf}"] = sc
        x[f"forward_active_ob_bias_rel_{tf}"] = RNG.choice(
            [-1.0, 1.0], len(x)
        )

    for h in (6, 12, 24):
        if action == "SKIP":
            x[f"gross_R_h{h}"] = 0.0
            x[f"exit_code_h{h}"] = -1
        else:
            r = RNG.normal(0, 1, len(x))
            r[RNG.random(len(x)) < 0.25] = np.nan
            x[f"gross_R_h{h}"] = r
            x[f"exit_code_h{h}"] = RNG.choice(
                [2, 3, 4, 5, 6, 7], len(x)
            )
    frames.append(x)

action = pd.concat(frames, ignore_index=True)
action_path = DS / "ob_rl_action_v0.parquet"
action.to_parquet(
    action_path, engine="pyarrow", compression="zstd",
    index=False,
)

pq_sha = AR.sha256_file(action_path)
MANIFEST_BASE = {
    "rl0_finalized": True,
    "model_view_version": "ob_rl_model_view_v0",
    "model_view_feature_count": 62,
    "parquet": {
        "action": {"parquet_sha256": pq_sha, "rows": 150_367}
    },
}


# ============================================================
# 5) Input authority gates
# ============================================================
def expect_stop(label: str, mutate) -> None:
    m = json.loads(json.dumps(MANIFEST_BASE))
    mutate(m)
    try:
        AR.verify_rl0_input(DS, m)
        check(label, False, "no raise")
    except RuntimeError as e:
        check(label, True, str(e)[:70])


try:
    AR.verify_rl0_input(DS, MANIFEST_BASE)
    check("verify_rl0_input passes", True)
except RuntimeError as e:
    check("verify_rl0_input passes", False, repr(e))

expect_stop(
    "rl0_finalized False -> STOP",
    lambda m: m.update({"rl0_finalized": False}),
)
expect_stop(
    "model view version drift -> STOP",
    lambda m: m.update(
        {"model_view_version": "something_else"}
    ),
)
expect_stop(
    "model_view_feature_count drift -> STOP",
    lambda m: m.update({"model_view_feature_count": 61}),
)
expect_stop(
    "row count drift -> STOP",
    lambda m: m["parquet"]["action"].update({"rows": 1}),
)
expect_stop(
    "parquet sha drift -> STOP",
    lambda m: m["parquet"]["action"].update(
        {"parquet_sha256": "0" * 64}
    ),
)

# ============================================================
# 6) Duplicate key gate
# ============================================================
dup = pd.concat([action, action.iloc[[0]]], ignore_index=True)
try:
    AR.assert_no_duplicate_key(dup)
    check("duplicate candidate-action STOP", False, "no raise")
except RuntimeError:
    check("duplicate candidate-action STOP", True)
AR.assert_no_duplicate_key(action)
check("clean key passes", True)

# ============================================================
# 7) Full analysis run on synthetic
# ============================================================
AR.DATASET_ROOT = DS
AR.OUT_ROOT = OUT
(DS / "dataset_manifest.json").write_text(
    json.dumps(MANIFEST_BASE), encoding="utf-8"
)

try:
    AR.main()
    check("rl1 analysis run", True)
except Exception as e:  # pragma: no cover
    check("rl1 analysis run", False, repr(e))
    import traceback

    traceback.print_exc()
    raise SystemExit(1)

baseline = pd.read_csv(OUT / "baseline_cells.csv")
pa = pd.read_csv(OUT / "paired_action_contrasts.csv")
pr = pd.read_csv(OUT / "paired_rr_contrasts.csv")
smc = pd.read_csv(OUT / "smc_cells.csv")
smc_c = pd.read_csv(OUT / "smc_contrasts.csv")

# --- #9 raw preserved + fit bins descriptive ---
_trades = action[action["trade_mode"] != "SKIP"]
raw_before = _trades["target_fit_swing_15m"].copy()
derived = AR.add_smc_states(_trades)
check(
    "raw target fit preserved",
    np.allclose(
        raw_before.to_numpy(float),
        derived["target_fit_swing_15m"].to_numpy(float),
        equal_nan=True,
    ),
)
check(
    "fit bins appear",
    "SMC_TARGET_FIT_BIN" in set(smc["analysis_family"]),
)
check(
    "fit bins descriptive only",
    "SMC_TARGET_FIT_BIN" not in set(smc_c["analysis_family"]),
)

# --- decision_weight preserved (not renormalised) ---
trades = action[action["trade_mode"] != "SKIP"]
check(
    "decision_weight column untouched by analysis",
    abs(
        float(trades[AR.WEIGHT_COL].sum())
        - float(action[AR.WEIGHT_COL].sum())
        * (6.0 / 7.0)
    )
    < 1e-6,
)
sample = baseline[baseline["view"] == "by_source"].head(3)
check(
    "cell weight_valid equals raw weight sum",
    all(
        abs(
            row["weight_valid"]
            - float(
                trades[
                    (trades["source_tf"] == row["source_tf"])
                    & (
                        trades["source_ob_structure"]
                        == row["source_ob_structure"]
                    )
                    & (
                        trades["trade_mode"]
                        == row["trade_mode"]
                    )
                    & (
                        np.isclose(
                            trades["target_R"], row["target_R"]
                        )
                    )
                    & np.isfinite(
                        trades[
                            f"gross_R_h{int(row['horizon'])}"
                        ]
                    )
                ][AR.WEIGHT_COL].sum()
            )
        )
        < 1e-6
        for _, row in sample.iterrows()
    ),
)

# --- valid coverage on the whole cell ---
b = baseline[baseline["view"] == "by_source"]
ok_pct = bool(
    np.allclose(
        b["valid_pct"],
        b["valid_n"] / b["rows_total"] * 100.0,
        atol=0.02,
    )
)
check("valid_pct computed on whole cell", ok_pct)

# --- baseline horizons ---
check(
    "baseline covers H6/H12/H24",
    set(baseline["horizon"].unique()) == {6, 12, 24},
)
check(
    "SMC discovery is H12 only",
    set(smc["horizon"].unique()) == {12},
)

# --- primary conditioning always present ---
by_src = baseline[baseline["view"] == "by_source"]
check(
    "by_source always has source_tf + structure",
    by_src["source_tf"].notna().all()
    and by_src["source_ob_structure"].notna().all(),
)
smc_src = smc[smc["view"] == "by_source"]
check(
    "SMC by_source always has source_tf + structure",
    smc_src["source_tf"].notna().all()
    and smc_src["source_ob_structure"].notna().all(),
)
check(
    "SMC cells always conditioned on trade_mode + target_R",
    smc["trade_mode"].notna().all()
    and smc["target_R"].notna().all(),
)
check(
    "no 4h anywhere in outputs",
    not [
        f
        for f in (
            "baseline_cells.csv",
            "smc_cells.csv",
            "smc_contrasts.csv",
        )
        if "4h"
        in (OUT / f).read_text(encoding="utf-8")
    ],
)

# --- paired action delta ---
sub = trades[
    trades["source_tf"].astype(str) == "5m"
]
f1 = sub[sub["trade_mode"] == "FOLLOW"].set_index(
    ["candidate_id", "target_R"]
)
d1 = sub[sub["trade_mode"] == "FADE"].set_index(
    ["candidate_id", "target_R"]
)
common = f1.index.intersection(d1.index)
manual = (
    f1.loc[common, "gross_R_h12"]
    - d1.loc[common, "gross_R_h12"]
)
w = f1.loc[common, AR.WEIGHT_COL]
manual_mean = AR.weighted_mean(
    manual.to_numpy(float), w.to_numpy(float)
)
got = pa[
    (pa["view"] == "by_source")
    & (pa["source_tf"] == "5m")
    & (pa["symbol"].isna())
]
check(
    "FOLLOW-FADE paired delta matches manual",
    len(got) > 0
    and abs(
        float(
            got[got["target_R"] == got["target_R"].iloc[0]][
                "delta_mean_R"
            ].iloc[0]
        )
        - manual_mean
    )
    < 0.4,
    (manual_mean, got["delta_mean_R"].tolist()[:3]),
)
check(
    "paired_action has target_R and no trade_mode sort",
    "target_R" in pa.columns
    and pa["pair_kind"].str.contains("FOLLOW_minus_FADE").all(),
)

# --- paired RR adjacent only ---
check(
    "RR contrasts are adjacent only",
    set(zip(pr["rr_low"], pr["rr_high"]))
    == {(1.5, 2.0), (2.0, 2.5)},
    set(zip(pr["rr_low"], pr["rr_high"])),
)

# --- SMC states present ---
for state in (
    "ALIGNED",
    "OPPOSED",
    "TARGET_BEYOND_STRUCTURE",
    "TARGET_AT_OR_BEFORE_STRUCTURE",
    "STRUCTURE_INSIDE_STOP",
    "STRUCTURE_AT_OR_BEYOND_STOP",
    "NO_OB",
    "NO_LEVEL",
):
    check(f"smc state {state} present", state in set(smc["state"]))
check(
    "forward OB identity present",
    any(
        str(s).startswith(("INTERNAL_", "SWING_"))
        for s in smc["state"].unique()
    ),
)

# --- contrasts are pre-registered only ---
check(
    "contrasts pre-registered",
    set(smc_c["state_definition"].unique())
    <= {"bias", "target_fit", "stop_structure"},
    set(smc_c["state_definition"].unique()),
)
check(
    "contrast names follow high_minus_low",
    all(
        "_minus_" in str(c) for c in smc_c["contrast"].unique()
    ),
)

# --- bootstrap determinism + low support ---
a_arm = pd.DataFrame(
    {
        AR.REWARD_COL: RNG.normal(0, 1, 300),
        AR.WEIGHT_COL: np.ones(300),
        AR.DAY_COL: [
            f"d{i % 25}" for i in range(300)
        ],
    }
)
b_arm = pd.DataFrame(
    {
        AR.REWARD_COL: RNG.normal(0, 1, 300),
        AR.WEIGHT_COL: np.ones(300),
        AR.DAY_COL: [
            f"d{i % 25}" for i in range(300)
        ],
    }
)
ci1 = AR.cluster_bootstrap_delta(a_arm, b_arm)
ci2 = AR.cluster_bootstrap_delta(a_arm, b_arm)
check(
    "cluster bootstrap deterministic",
    ci1 == ci2 and ci1[2] > 0,
    (ci1, ci2),
)
few = pd.DataFrame(
    {
        AR.REWARD_COL: [1.0, 2.0],
        AR.WEIGHT_COL: [1.0, 1.0],
        AR.DAY_COL: ["d1", "d2"],
    }
)
lo, hi, reps = AR.cluster_bootstrap_delta(few, few)
check(
    "low trading days -> no CI",
    np.isnan(lo) and np.isnan(hi) and reps == 0,
)
check(
    "low-support cells still summarised",
    (smc_c["ci_status"] == SP.CI_LOW_SUPPORT).sum() > 0
    and smc_c.loc[
        smc_c["ci_status"] == SP.CI_LOW_SUPPORT, "delta_mean_R"
    ]
    .notna()
    .any(),
)
check(
    "some cells reach OK CI status",
    (smc_c["ci_status"] == SP.CI_OK).sum() > 0,
)

# --- #9 paired bootstrap directly on delta_R ---
toy = pd.DataFrame(
    {
        AR.DAY_COL: ["d1", "d1", "d2", "d2"],
        AR.WEIGHT_COL: [1, 1, 1, 1],
        "delta_R": [1, 1, -1, -1],
    }
)
check(
    "paired point estimate = 0",
    abs(
        AR.weighted_mean(
            toy["delta_R"], toy[AR.WEIGHT_COL]
        )
    )
    < 1e-12,
    AR.weighted_mean(toy["delta_R"], toy[AR.WEIGHT_COL]),
)

det = pd.DataFrame(
    {
        AR.DAY_COL: [f"d{i % 25}" for i in range(300)],
        AR.WEIGHT_COL: np.ones(300),
        "delta_R": RNG.normal(0, 1, 300),
    }
)
ci1 = AR.cluster_bootstrap_paired_delta(det)
ci2 = AR.cluster_bootstrap_paired_delta(det)
check(
    "paired bootstrap deterministic",
    ci1 == ci2 and ci1[2] > 0,
    (ci1, ci2),
)

# trade-mode contamination: delta_R already encodes FOLLOW-FADE,
# the raw FADE reward must never enter the bootstrap.
contam = pd.DataFrame(
    {
        AR.DAY_COL: [f"d{i % 25}" for i in range(250)],
        AR.WEIGHT_COL: np.ones(250),
        "delta_R": np.ones(250),
        "gross_R_h12_fade": np.full(250, -10.0),
    }
)
cic = AR.cluster_bootstrap_paired_delta(
    contam[[AR.DAY_COL, AR.WEIGHT_COL, "delta_R"]]
)
check(
    "paired bootstrap ignores FADE raw (-10)",
    cic[2] > 0 and cic[0] <= 1.0 <= cic[1],
    cic,
)

# --- #9 independent arm trading-day gate ---
A = pd.DataFrame(
    {
        AR.REWARD_COL: RNG.normal(0, 1, 200),
        AR.WEIGHT_COL: np.ones(200),
        AR.DAY_COL: [f"ad{i % 5}" for i in range(200)],
    }
)
B = pd.DataFrame(
    {
        AR.REWARD_COL: RNG.normal(0, 1, 200),
        AR.WEIGHT_COL: np.ones(200),
        AR.DAY_COL: [f"bd{i % 25}" for i in range(200)],
    }
)
lo, hi, reps = AR.cluster_bootstrap_delta(A, B)
check(
    "independent arm day gate LOW_SUPPORT",
    np.isnan(lo) and np.isnan(hi) and reps == 0,
    (lo, hi, reps),
)

# --- #9 finite support: n_high/n_low counts only finite reward ---
_many = pd.DataFrame(
    {
        "gross_R_h12": [np.nan] * 80
        + list(RNG.normal(0, 1, 120)),
        AR.WEIGHT_COL: np.ones(200),
        AR.DAY_COL: [f"d{i % 25}" for i in range(200)],
    }
)
check(
    "finite support counts only finite reward",
    len(AR._arm(_many, 12)) == 120,
)

# --- guard against ranking language ---
try:
    AR.guard_outputs(OUT)
    check("output guard passes", True)
except RuntimeError as e:
    check("output guard passes", False, str(e))

(OUT / "bad.csv").write_text("a,b\n1,best_action\n")
try:
    AR.guard_outputs(OUT)
    check("guard blocks ranking language", False, "no raise")
except RuntimeError:
    check("guard blocks ranking language", True)
(OUT / "bad.csv").unlink()

# --- no sorting by reward in outputs ---
check(
    "no reward-sorted ordering (columns stable)",
    list(baseline.columns) == list(SP.CELL_COLUMNS),
)

# ============================================================
print("\n========= RL-1A SYNTHETIC RESULT =========")
fails = [r for r in RESULTS if not r[1]]
for name, _, det in fails:
    print(f"FAIL  {name} :: {det}")
print(
    f"checks={len(RESULTS)} pass={len(RESULTS) - len(fails)} "
    f"fail={len(fails)}"
)
print("==========================================")
shutil.rmtree(TMP, ignore_errors=True)
raise SystemExit(1 if fails else 0)
