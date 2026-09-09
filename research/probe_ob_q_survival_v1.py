#!/usr/bin/env python3

"""
OB Q Survival Probe V1

Reuses the frozen Q-probe functions from explore_ob_q_relationship_v1
(identical CatBoost config, policy rule, weighting, candidate universe).
No new research framework.

Question being tested:
    Is the FULL_NO_MOMENTUM +0.056R signal real, or a one-time-slice
    artifact of the single 70/30 split?

Method (user-specified, minimal):
    2 models:  EVENT_ACTION, FULL_NO_MOMENTUM
    3 chronological train/test windows by candidate percentile:
        A: train 0-55%,   test 55-70%
        B: train 0-70%,   test 70-85%
        C: train 0-85%,   test 85-100%
    No feature redesign, no CatBoost tuning, no bootstrap.

Only reports: q_corr, selected_R, trade_rate, per-symbol selected_R.
"""

import pandas as pd

from research import explore_ob_q_relationship_v1 as E

REWARD = E.REWARD
WEIGHT = E.WEIGHT

# Frozen complete-H12 candidate action rows from the same dataset.
work = E.work

cand_meta = (
    work[
        [
            "candidate_id",
            "trading_day",
            "symbol",
            "source_tf",
            WEIGHT,
        ]
    ]
    .drop_duplicates("candidate_id")
    .copy()
)

cand_meta["_day"] = pd.to_datetime(
    cand_meta["trading_day"],
    errors="raise",
)

cand_meta = cand_meta.sort_values(
    ["_day", "candidate_id"]
).reset_index(drop=True)

ids = cand_meta["candidate_id"].tolist()
n = len(ids)

# (train_start_frac, train_end_frac, test_start_frac, test_end_frac)
WINDOWS = {
    "A": (0.00, 0.55, 0.55, 0.70),
    "B": (0.00, 0.70, 0.70, 0.85),
    "C": (0.00, 0.85, 0.85, 1.00),
}

MODELS = {
    "EVENT_ACTION": E.MODEL_SPECS["EVENT_ACTION"],
    "FULL_NO_MOMENTUM": E.MODEL_SPECS["FULL_NO_MOMENTUM"],
}


def frac_idx(frac: float) -> int:
    return int(round(n * frac))


results = []

for wname, (tr0, tr1, te0, te1) in WINDOWS.items():

    train_ids = set(ids[frac_idx(tr0):frac_idx(tr1)])
    test_ids = set(ids[frac_idx(te0):frac_idx(te1)])

    train = (
        work[work["candidate_id"].isin(train_ids)]
        .sort_values(
            ["trading_day", "candidate_id", "trade_mode", "target_R"]
        )
        .reset_index(drop=True)
    )

    test = (
        work[work["candidate_id"].isin(test_ids)]
        .sort_values(
            ["trading_day", "candidate_id", "trade_mode", "target_R"]
        )
        .reset_index(drop=True)
    )

    # Reuse frozen probe functions by swapping module globals.
    E.train = train
    E.test = test

    for mname, feats in MODELS.items():

        _, pred, metrics = E.fit_probe(mname, feats)
        selected = E.policy_from_predictions(pred, mname)

        all_sum = E.summarize_policy(
            selected,
            model_name=mname,
            scope="ALL",
            scope_value="ALL",
        )

        per_symbol = {
            str(sym): E.summarize_policy(
                g,
                model_name=mname,
                scope="SYMBOL",
                scope_value=str(sym),
            )["weighted_selected_R"]
            for sym, g in selected.groupby("symbol", observed=True)
        }

        results.append(
            {
                "window": wname,
                "model": mname,
                "train_candidates": int(len(train_ids)),
                "test_candidates": int(len(test_ids)),
                "q_corr": metrics["weighted_q_corr"],
                "q_rmse": metrics["weighted_q_rmse"],
                "selected_R": all_sum["weighted_selected_R"],
                "trade_rate": all_sum["trade_rate"],
                "per_symbol_R": per_symbol,
            }
        )


print()
print("SURVIVAL_PROBE_DONE")

for r in results:
    print(
        f"{r['window']} {r['model']:18s} "
        f"tr={r['train_candidates']:5d} te={r['test_candidates']:5d} "
        f"qcorr={r['q_corr']:.5f} selR={r['selected_R']:.5f} "
        f"trade={r['trade_rate']:.4f} "
        f"sym={r['per_symbol_R']}"
    )

print()
for mname in MODELS:
    vals = [
        r["selected_R"]
        for r in results
        if r["model"] == mname
    ]
    n_pos = sum(1 for v in vals if v > 0)
    print(
        f"{mname}: windows_selected_R>0 = {n_pos}/3 "
        f"values={[round(v, 5) for v in vals]}"
    )

print()
print("JUDGEMENT_LEFT_TO_USER_PER_STATED_RULE")
