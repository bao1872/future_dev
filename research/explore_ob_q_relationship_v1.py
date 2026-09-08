#!/usr/bin/env python3

"""
OB Q-Relationship Probe V1

Purpose
-------
Use the frozen OB RL Dataset V0 as:

    State + Action -> H12 Reward

This is NOT a production strategy and NOT final RL training.

It is an exploratory probe answering:

1. Does the full simultaneous 5m + 15m + 1h environment contain
   out-of-sample information about action-specific reward?

2. Does the full environment improve action selection over:
       action only
       OB-event + action

3. Which environment family contributes incrementally when all
   timeframes are considered together?

Fixed:
- chronological 70/30 trading-day split
- H12 only
- no hyperparameter search
- no threshold tuning
- no random split
- no symbol as model feature
- no future/outcome fields as model input
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import catboost
    from catboost import CatBoostRegressor
except ImportError as exc:
    raise RuntimeError(
        "CatBoost is required for this exploratory probe. "
        "Install locally with: "
        "python -m pip install 'catboost>=1.2,<2'"
    ) from exc


from research.ob_rl_dataset_v0_spec import (
    VALIDATED_TFS,
)

from research.ob_rl_model_view_v0_spec import (
    MODEL_VIEW_VERSION,
    MODEL_FEATURES_V0,
    CATEGORICAL_FEATURES_V0,
    EVENT_FEATURES_V0,
    ACTION_FEATURES_V0,
    QUANTILE_FEATURES_V0,
    smc_features,
    dsa_features,
    momentum_features,
)


# ============================================================
# Fixed experiment contract
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_rl_dataset_v0"
)

PARQUET = (
    DATA_ROOT
    / "ob_rl_action_v0.parquet"
)

OUT = (
    ROOT
    / "research"
    / "analysis_results"
    / "ob_q_relationship_v1"
)

OUT.mkdir(
    parents=True,
    exist_ok=True,
)

REWARD = "gross_R_h12"
WEIGHT = "decision_weight"

SEED = 20260908
TRAIN_FRACTION = 0.70

NON_SKIP_MODES = (
    "FOLLOW",
    "FADE",
)

EXPECTED_RR = (
    1.5,
    2.0,
    2.5,
)

EXPECTED_ACTION_ROWS = 6


# ============================================================
# Feature families
# ============================================================

SMC_FEATURES = tuple(
    c
    for tf in VALIDATED_TFS
    for c in smc_features(tf)
)

DSA_FEATURES = tuple(
    c
    for tf in VALIDATED_TFS
    for c in dsa_features(tf)
)

MOMENTUM_FEATURES = tuple(
    c
    for tf in VALIDATED_TFS
    for c in momentum_features(tf)
)

EVENT_FEATURES = tuple(
    EVENT_FEATURES_V0
)

ACTION_FEATURES = tuple(
    ACTION_FEATURES_V0
)

QUANTILE_FEATURES = tuple(
    QUANTILE_FEATURES_V0
)

FULL_FEATURES = tuple(
    MODEL_FEATURES_V0
)


def remove_features(
    base: tuple[str, ...],
    removed: tuple[str, ...],
) -> tuple[str, ...]:

    removed_set = set(removed)

    return tuple(
        c
        for c in base
        if c not in removed_set
    )


MODEL_SPECS = {
    # Trivial action baseline.
    # trade_direction is deliberately excluded here because it already
    # contains source-state information.
    "ACTION_LABEL_ONLY": (
        "trade_mode",
        "target_R",
    ),

    # Knows the OB event itself, but not surrounding environment.
    "EVENT_ACTION": tuple(
        dict.fromkeys(
            (
                *EVENT_FEATURES,
                *ACTION_FEATURES,
            )
        )
    ),

    # Primary model: all simultaneous multi-TF environment.
    "FULL": FULL_FEATURES,

    # Family ablations: ALL timeframes removed together.
    "FULL_NO_SMC": remove_features(
        FULL_FEATURES,
        SMC_FEATURES,
    ),

    "FULL_NO_DSA": remove_features(
        FULL_FEATURES,
        DSA_FEATURES,
    ),

    "FULL_NO_MOMENTUM": remove_features(
        FULL_FEATURES,
        MOMENTUM_FEATURES,
    ),

    "FULL_NO_QUANTILE": remove_features(
        FULL_FEATURES,
        QUANTILE_FEATURES,
    ),
}


# ============================================================
# Utilities
# ============================================================

def git_head() -> str:
    p = subprocess.run(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    return p.stdout.strip()


def weighted_mean(
    values,
    weights,
) -> float:

    v = np.asarray(
        values,
        dtype=float,
    )

    w = np.asarray(
        weights,
        dtype=float,
    )

    ok = (
        np.isfinite(v)
        & np.isfinite(w)
        & (w > 0)
    )

    if not ok.any():
        return np.nan

    return float(
        np.sum(
            v[ok] * w[ok]
        )
        / np.sum(
            w[ok]
        )
    )


def weighted_rate(
    mask,
    weights,
) -> float:

    x = np.asarray(
        mask,
        dtype=float,
    )

    w = np.asarray(
        weights,
        dtype=float,
    )

    ok = (
        np.isfinite(x)
        & np.isfinite(w)
        & (w > 0)
    )

    if not ok.any():
        return np.nan

    return float(
        np.sum(
            x[ok] * w[ok]
        )
        / np.sum(
            w[ok]
        )
    )


def weighted_rmse(
    actual,
    pred,
    weights,
) -> float:

    y = np.asarray(
        actual,
        dtype=float,
    )

    p = np.asarray(
        pred,
        dtype=float,
    )

    w = np.asarray(
        weights,
        dtype=float,
    )

    ok = (
        np.isfinite(y)
        & np.isfinite(p)
        & np.isfinite(w)
        & (w > 0)
    )

    if not ok.any():
        return np.nan

    return float(
        np.sqrt(
            np.sum(
                w[ok]
                * (
                    y[ok]
                    - p[ok]
                ) ** 2
            )
            / np.sum(
                w[ok]
            )
        )
    )


def weighted_mae(
    actual,
    pred,
    weights,
) -> float:

    y = np.asarray(
        actual,
        dtype=float,
    )

    p = np.asarray(
        pred,
        dtype=float,
    )

    w = np.asarray(
        weights,
        dtype=float,
    )

    ok = (
        np.isfinite(y)
        & np.isfinite(p)
        & np.isfinite(w)
        & (w > 0)
    )

    if not ok.any():
        return np.nan

    return float(
        np.sum(
            w[ok]
            * np.abs(
                y[ok]
                - p[ok]
            )
        )
        / np.sum(
            w[ok]
        )
    )


def weighted_corr(
    actual,
    pred,
    weights,
) -> float:

    y = np.asarray(
        actual,
        dtype=float,
    )

    p = np.asarray(
        pred,
        dtype=float,
    )

    w = np.asarray(
        weights,
        dtype=float,
    )

    ok = (
        np.isfinite(y)
        & np.isfinite(p)
        & np.isfinite(w)
        & (w > 0)
    )

    if ok.sum() < 2:
        return np.nan

    y = y[ok]
    p = p[ok]
    w = w[ok]

    w = w / w.sum()

    my = np.sum(
        w * y
    )

    mp = np.sum(
        w * p
    )

    cov = np.sum(
        w
        * (y - my)
        * (p - mp)
    )

    vy = np.sum(
        w
        * (y - my) ** 2
    )

    vp = np.sum(
        w
        * (p - mp) ** 2
    )

    if (
        vy <= 0
        or vp <= 0
    ):
        return np.nan

    return float(
        cov
        / np.sqrt(
            vy * vp
        )
    )


def action_label(
    trade_mode,
    target_r,
) -> str:

    return (
        f"{str(trade_mode)}_"
        f"{float(target_r):g}R"
    )


# ============================================================
# Load frozen model view
# ============================================================

needed_columns = set(
    FULL_FEATURES
)

needed_columns.update(
    {
        "candidate_id",
        "symbol",
        "trading_day",
        "source_tf",
        WEIGHT,
        REWARD,
    }
)

df = pd.read_parquet(
    PARQUET,
    columns=sorted(
        needed_columns
    ),
)


# ============================================================
# Hard model-view checks
# ============================================================

missing_features = (
    set(FULL_FEATURES)
    - set(df.columns)
)

if missing_features:
    raise RuntimeError(
        "missing Model View fields: "
        f"{sorted(missing_features)}"
    )

if len(FULL_FEATURES) != 62:
    raise RuntimeError(
        "Model View V0 feature count changed: "
        f"{len(FULL_FEATURES)} != 62"
    )


# ============================================================
# Six non-SKIP counterfactual actions
# ============================================================

df = df[
    df["trade_mode"]
    .astype(str)
    .isin(
        NON_SKIP_MODES
    )
].copy()

df["target_R"] = pd.to_numeric(
    df["target_R"],
    errors="coerce",
)

df[REWARD] = pd.to_numeric(
    df[REWARD],
    errors="coerce",
)

df[WEIGHT] = pd.to_numeric(
    df[WEIGHT],
    errors="coerce",
)


actions = (
    df[
        [
            "trade_mode",
            "target_R",
        ]
    ]
    .drop_duplicates()
    .sort_values(
        [
            "trade_mode",
            "target_R",
        ]
    )
)

if len(actions) != EXPECTED_ACTION_ROWS:
    raise RuntimeError(
        f"expected 6 non-SKIP actions, "
        f"got {len(actions)}"
    )

for rr in EXPECTED_RR:
    for mode in NON_SKIP_MODES:

        hit = (
            (
                actions["trade_mode"]
                .astype(str)
                == mode
            )
            &
            np.isclose(
                actions["target_R"],
                rr,
            )
        )

        if hit.sum() != 1:
            raise RuntimeError(
                "missing action: "
                f"{mode} {rr}R"
            )


# ============================================================
# Candidate cardinality / metadata invariance
# ============================================================

candidate_rows = (
    df.groupby(
        "candidate_id",
        observed=True,
    )
    .size()
)

if not (
    candidate_rows
    == EXPECTED_ACTION_ROWS
).all():
    raise RuntimeError(
        "each candidate must have "
        "exactly six non-SKIP actions"
    )


for col in (
    "symbol",
    "trading_day",
    "source_tf",
    WEIGHT,
):

    varying = (
        df.groupby(
            "candidate_id",
            observed=True,
        )[col]
        .nunique(
            dropna=False
        )
    )

    if (
        varying > 1
    ).any():

        raise RuntimeError(
            f"{col} varies across actions "
            "within candidate"
        )


# ============================================================
# Complete H12 counterfactual candidates only
# ============================================================

df["_reward_ok"] = np.isfinite(
    df[REWARD]
)

reward_support = (
    df.groupby(
        "candidate_id",
        observed=True,
    )["_reward_ok"]
    .sum()
)


complete_ids = (
    reward_support[
        reward_support
        == EXPECTED_ACTION_ROWS
    ]
    .index
)

partial_candidates = int(
    (
        (
            reward_support > 0
        )
        &
        (
            reward_support
            < EXPECTED_ACTION_ROWS
        )
    ).sum()
)

zero_reward_candidates = int(
    (
        reward_support
        == 0
    ).sum()
)


work = df[
    df["candidate_id"]
    .isin(
        complete_ids
    )
].copy()

if work.empty:
    raise RuntimeError(
        "no complete H12 candidates"
    )

if work[REWARD].isna().any():
    raise RuntimeError(
        "complete reward subset contains NaN"
    )


# ============================================================
# Chronological candidate-level split
# ============================================================

candidate_meta = (
    work[
        [
            "candidate_id",
            "symbol",
            "trading_day",
            "source_tf",
            WEIGHT,
        ]
    ]
    .drop_duplicates(
        "candidate_id"
    )
    .copy()
)

candidate_meta[
    "_day"
] = pd.to_datetime(
    candidate_meta[
        "trading_day"
    ],
    errors="raise",
)

days = np.array(
    sorted(
        candidate_meta[
            "_day"
        ]
        .unique()
    )
)

if len(days) < 10:
    raise RuntimeError(
        "too few trading days"
    )

cut = int(
    np.floor(
        len(days)
        * TRAIN_FRACTION
    )
)

cut = min(
    max(
        cut,
        1,
    ),
    len(days) - 1,
)

split_day = pd.Timestamp(
    days[cut]
)


candidate_meta[
    "split"
] = np.where(
    candidate_meta[
        "_day"
    ]
    < split_day,
    "TRAIN",
    "TEST",
)


split_map = (
    candidate_meta
    .set_index(
        "candidate_id"
    )["split"]
)

work["split"] = (
    work["candidate_id"]
    .map(
        split_map
    )
)


train = (
    work[
        work["split"]
        == "TRAIN"
    ]
    .sort_values(
        [
            "trading_day",
            "candidate_id",
            "trade_mode",
            "target_R",
        ]
    )
    .reset_index(
        drop=True
    )
)

test = (
    work[
        work["split"]
        == "TEST"
    ]
    .sort_values(
        [
            "trading_day",
            "candidate_id",
            "trade_mode",
            "target_R",
        ]
    )
    .reset_index(
        drop=True
    )
)


train_ids = set(
    train[
        "candidate_id"
    ]
)

test_ids = set(
    test[
        "candidate_id"
    ]
)

if (
    train_ids
    & test_ids
):
    raise RuntimeError(
        "candidate leakage across split"
    )


# ============================================================
# CatBoost input preparation
# ============================================================

CATEGORICAL_SET = set(
    CATEGORICAL_FEATURES_V0
)


def prepare_X(
    frame: pd.DataFrame,
    features: tuple[str, ...],
):

    x = frame[
        list(
            features
        )
    ].copy()

    cat_cols = [
        c
        for c in features
        if c in CATEGORICAL_SET
    ]

    for c in cat_cols:

        x[c] = (
            x[c]
            .astype("object")
            .where(
                x[c].notna(),
                "__MISSING__",
            )
            .astype(str)
        )

    for c in features:

        if c in cat_cols:
            continue

        x[c] = pd.to_numeric(
            x[c],
            errors="coerce",
        )

    return (
        x,
        cat_cols,
    )


# ============================================================
# Fixed probe model
#
# No tuning.
# No early stopping.
# No stochastic bootstrap.
# ============================================================

def fit_probe(
    name: str,
    features: tuple[str, ...],
):

    x_train, cat_cols = (
        prepare_X(
            train,
            features,
        )
    )

    x_test, _ = (
        prepare_X(
            test,
            features,
        )
    )

    model = CatBoostRegressor(
        iterations=400,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5.0,
        loss_function="RMSE",
        random_seed=SEED,
        random_strength=0.0,
        bootstrap_type="No",
        one_hot_max_size=32,
        allow_writing_files=False,
        verbose=False,
        thread_count=-1,
    )

    model.fit(
        x_train,
        train[REWARD],
        cat_features=cat_cols,
        sample_weight=train[
            WEIGHT
        ],
    )

    pred = model.predict(
        x_test
    )

    result = test[
        [
            "candidate_id",
            "symbol",
            "source_tf",
            "trading_day",
            "trade_mode",
            "target_R",
            WEIGHT,
            REWARD,
        ]
    ].copy()

    result[
        "pred_q"
    ] = pred

    result[
        "action_label"
    ] = [
        action_label(
            m,
            r,
        )
        for m, r in zip(
            result[
                "trade_mode"
            ],
            result[
                "target_R"
            ],
        )
    ]

    metrics = {
        "model":
            name,

        "feature_count":
            len(features),

        "train_action_rows":
            len(train),

        "test_action_rows":
            len(test),

        "test_candidates":
            test[
                "candidate_id"
            ].nunique(),

        "weighted_q_rmse":
            weighted_rmse(
                result[
                    REWARD
                ],
                result[
                    "pred_q"
                ],
                result[
                    WEIGHT
                ],
            ),

        "weighted_q_mae":
            weighted_mae(
                result[
                    REWARD
                ],
                result[
                    "pred_q"
                ],
                result[
                    WEIGHT
                ],
            ),

        "weighted_q_corr":
            weighted_corr(
                result[
                    REWARD
                ],
                result[
                    "pred_q"
                ],
                result[
                    WEIGHT
                ],
            ),
    }

    return (
        model,
        result,
        metrics,
    )


# ============================================================
# Candidate-level policy evaluation
#
# Highest predicted non-SKIP Q.
#
# If max predicted Q <= 0:
#     SKIP
#     realized reward = 0
#
# This is fixed before seeing test reward.
# ============================================================

def policy_from_predictions(
    pred: pd.DataFrame,
    model_name: str,
):

    best_idx = (
        pred.groupby(
            "candidate_id",
            observed=True,
        )["pred_q"]
        .idxmax()
    )

    selected = (
        pred.loc[
            best_idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    selected[
        "traded"
    ] = (
        selected[
            "pred_q"
        ]
        > 0
    )

    selected[
        "selected_R"
    ] = np.where(
        selected[
            "traded"
        ],
        selected[
            REWARD
        ],
        0.0,
    )

    selected[
        "selected_action"
    ] = np.where(
        selected[
            "traded"
        ],
        selected[
            "action_label"
        ],
        "SKIP",
    )


    oracle = (
        pred.groupby(
            "candidate_id",
            observed=True,
        )[REWARD]
        .max()
        .clip(
            lower=0.0
        )
    )

    selected[
        "oracle_R"
    ] = (
        selected[
            "candidate_id"
        ]
        .map(
            oracle
        )
    )

    selected[
        "regret_R"
    ] = (
        selected[
            "oracle_R"
        ]
        - selected[
            "selected_R"
        ]
    )

    selected[
        "model"
    ] = model_name

    return selected


def summarize_policy(
    selected: pd.DataFrame,
    *,
    model_name: str,
    scope: str,
    scope_value: str,
):

    w = selected[
        WEIGHT
    ]

    traded = selected[
        "traded"
    ].astype(bool)

    traded_rows = (
        selected[
            traded
        ]
    )

    follow_share = np.nan
    fade_share = np.nan

    if len(
        traded_rows
    ):

        follow_share = (
            weighted_rate(
                traded_rows[
                    "trade_mode"
                ]
                .astype(str)
                .eq(
                    "FOLLOW"
                ),
                traded_rows[
                    WEIGHT
                ],
            )
        )

        fade_share = (
            weighted_rate(
                traded_rows[
                    "trade_mode"
                ]
                .astype(str)
                .eq(
                    "FADE"
                ),
                traded_rows[
                    WEIGHT
                ],
            )
        )

    return {
        "model":
            model_name,

        "scope":
            scope,

        "scope_value":
            scope_value,

        "candidate_n":
            len(selected),

        "weighted_selected_R":
            weighted_mean(
                selected[
                    "selected_R"
                ],
                w,
            ),

        "weighted_oracle_R":
            weighted_mean(
                selected[
                    "oracle_R"
                ],
                w,
            ),

        "weighted_regret_R":
            weighted_mean(
                selected[
                    "regret_R"
                ],
                w,
            ),

        "trade_rate":
            weighted_rate(
                traded,
                w,
            ),

        "follow_share_among_trades":
            follow_share,

        "fade_share_among_trades":
            fade_share,

        "positive_realized_R_rate":
            weighted_rate(
                selected[
                    "selected_R"
                ]
                > 0,
                w,
            ),
    }


# ============================================================
# Global train-action-mean baseline
# ============================================================

train_action_means = []

for (
    mode,
    rr,
), g in train.groupby(
    [
        "trade_mode",
        "target_R",
    ],
    observed=True,
):

    train_action_means.append(
        {
            "trade_mode":
                str(mode),

            "target_R":
                float(rr),

            "train_mean_R":
                weighted_mean(
                    g[
                        REWARD
                    ],
                    g[
                        WEIGHT
                    ],
                ),
        }
    )


train_action_means = (
    pd.DataFrame(
        train_action_means
    )
    .sort_values(
        "train_mean_R",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)

best_global = (
    train_action_means.iloc[0]
)

global_mode = str(
    best_global[
        "trade_mode"
    ]
)

global_rr = float(
    best_global[
        "target_R"
    ]
)

global_mean_q = float(
    best_global[
        "train_mean_R"
    ]
)


# Build global baseline test selection.
test_candidates = (
    test[
        [
            "candidate_id",
            "symbol",
            "source_tf",
            "trading_day",
            WEIGHT,
        ]
    ]
    .drop_duplicates(
        "candidate_id"
    )
)


global_action_rows = test[
    (
        test[
            "trade_mode"
        ]
        .astype(str)
        == global_mode
    )
    &
    np.isclose(
        test[
            "target_R"
        ],
        global_rr,
    )
][
    [
        "candidate_id",
        "trade_mode",
        "target_R",
        REWARD,
    ]
]


global_selected = (
    test_candidates
    .merge(
        global_action_rows,
        on="candidate_id",
        validate="one_to_one",
    )
)


global_selected[
    "traded"
] = (
    global_mean_q
    > 0
)

global_selected[
    "selected_R"
] = np.where(
    global_selected[
        "traded"
    ],
    global_selected[
        REWARD
    ],
    0.0,
)

global_selected[
    "selected_action"
] = np.where(
    global_selected[
        "traded"
    ],
    action_label(
        global_mode,
        global_rr,
    ),
    "SKIP",
)


test_oracle = (
    test.groupby(
        "candidate_id",
        observed=True,
    )[REWARD]
    .max()
    .clip(
        lower=0.0
    )
)


global_selected[
    "oracle_R"
] = (
    global_selected[
        "candidate_id"
    ]
    .map(
        test_oracle
    )
)

global_selected[
    "regret_R"
] = (
    global_selected[
        "oracle_R"
    ]
    - global_selected[
        "selected_R"
    ]
)

global_selected[
    "model"
] = (
    "GLOBAL_ACTION_MEAN"
)


# ============================================================
# Fit all probes
# ============================================================

models = {}
predictions = {}
model_metric_rows = []
selection_frames = {}


for (
    name,
    features,
) in MODEL_SPECS.items():

    print(
        "FIT",
        name,
        "features=",
        len(features),
    )

    model, pred, metrics = (
        fit_probe(
            name,
            features,
        )
    )

    models[
        name
    ] = model

    predictions[
        name
    ] = pred

    model_metric_rows.append(
        metrics
    )

    selection_frames[
        name
    ] = (
        policy_from_predictions(
            pred,
            name,
        )
    )


selection_frames[
    "GLOBAL_ACTION_MEAN"
] = global_selected


# ============================================================
# Policy summaries
# ============================================================

policy_rows = []


for (
    name,
    selected,
) in selection_frames.items():

    policy_rows.append(
        summarize_policy(
            selected,
            model_name=name,
            scope="ALL",
            scope_value="ALL",
        )
    )

    for (
        symbol,
        g,
    ) in selected.groupby(
        "symbol",
        observed=True,
    ):

        policy_rows.append(
            summarize_policy(
                g,
                model_name=name,
                scope="SYMBOL",
                scope_value=str(
                    symbol
                ),
            )
        )

    for (
        source_tf,
        g,
    ) in selected.groupby(
        "source_tf",
        observed=True,
    ):

        policy_rows.append(
            summarize_policy(
                g,
                model_name=name,
                scope="SOURCE_TF",
                scope_value=str(
                    source_tf
                ),
            )
        )


policy = pd.DataFrame(
    policy_rows
)


# ============================================================
# Ablation summary
# ============================================================

model_metrics = pd.DataFrame(
    model_metric_rows
)


all_policy = policy[
    (
        policy[
            "scope"
        ]
        == "ALL"
    )
].copy()


full_q = float(
    model_metrics.loc[
        model_metrics[
            "model"
        ]
        == "FULL",
        "weighted_q_rmse",
    ].iloc[0]
)


full_policy_r = float(
    all_policy.loc[
        all_policy[
            "model"
        ]
        == "FULL",
        "weighted_selected_R",
    ].iloc[0]
)


ablation_rows = []


for name in MODEL_SPECS:

    row_q = (
        model_metrics[
            model_metrics[
                "model"
            ]
            == name
        ]
        .iloc[0]
    )

    row_p = (
        all_policy[
            all_policy[
                "model"
            ]
            == name
        ]
        .iloc[0]
    )

    ablation_rows.append(
        {
            "model":
                name,

            "weighted_q_rmse":
                float(
                    row_q[
                        "weighted_q_rmse"
                    ]
                ),

            "q_rmse_minus_full":
                float(
                    row_q[
                        "weighted_q_rmse"
                    ]
                    - full_q
                ),

            "weighted_selected_R":
                float(
                    row_p[
                        "weighted_selected_R"
                    ]
                ),

            "selected_R_minus_full":
                float(
                    row_p[
                        "weighted_selected_R"
                    ]
                    - full_policy_r
                ),
        }
    )


ablation = pd.DataFrame(
    ablation_rows
)


# ============================================================
# Full-model feature importance
#
# Exploratory only.
# Ablation is the more important evidence.
# ============================================================

def feature_family(
    feature: str,
) -> str:

    if feature in EVENT_FEATURES:
        return "EVENT"

    if feature in SMC_FEATURES:
        return "SMC"

    if feature in DSA_FEATURES:
        return "DSA"

    if feature in MOMENTUM_FEATURES:
        return "MOMENTUM"

    if feature in QUANTILE_FEATURES:
        return "QUANTILE"

    if feature in ACTION_FEATURES:
        return "ACTION"

    return "OTHER"


def feature_tf(
    feature: str,
) -> str:

    for tf in VALIDATED_TFS:

        if feature.endswith(
            f"_{tf}"
        ):
            return tf

    return "NA"


full_model = models[
    "FULL"
]


importance = (
    pd.DataFrame(
        {
            "feature":
                list(
                    MODEL_SPECS[
                        "FULL"
                    ]
                ),

            "importance":
                full_model
                .get_feature_importance(),
        }
    )
)


importance[
    "family"
] = (
    importance[
        "feature"
    ]
    .map(
        feature_family
    )
)

importance[
    "timeframe"
] = (
    importance[
        "feature"
    ]
    .map(
        feature_tf
    )
)


importance = (
    importance
    .sort_values(
        "importance",
        ascending=False,
    )
    .reset_index(
        drop=True
    )
)


importance_family_tf = (
    importance.groupby(
        [
            "family",
            "timeframe",
        ],
        as_index=False,
        observed=True,
    )["importance"]
    .sum()
    .sort_values(
        "importance",
        ascending=False,
    )
)


# ============================================================
# Full-model Q calibration
#
# If predicted Q is informative, realized R should generally
# increase as predicted-Q bucket increases.
# ============================================================

full_pred = predictions[
    "FULL"
].copy()


full_pred[
    "q_bucket"
] = pd.qcut(
    full_pred[
        "pred_q"
    ],
    q=10,
    labels=False,
    duplicates="drop",
)


q_rows = []


for (
    bucket,
    g,
) in full_pred.groupby(
    "q_bucket",
    observed=True,
):

    q_rows.append(
        {
            "q_bucket":
                int(bucket),

            "action_rows":
                len(g),

            "weighted_mean_pred_q":
                weighted_mean(
                    g[
                        "pred_q"
                    ],
                    g[
                        WEIGHT
                    ],
                ),

            "weighted_mean_realized_R":
                weighted_mean(
                    g[
                        REWARD
                    ],
                    g[
                        WEIGHT
                    ],
                ),
        }
    )


q_calibration = (
    pd.DataFrame(
        q_rows
    )
    .sort_values(
        "q_bucket"
    )
)


# ============================================================
# Persist small results only
# ============================================================

model_metrics.to_csv(
    OUT
    / "model_metrics.csv",
    index=False,
)

policy.to_csv(
    OUT
    / "policy_metrics.csv",
    index=False,
)

ablation.to_csv(
    OUT
    / "ablation_metrics.csv",
    index=False,
)

importance.to_csv(
    OUT
    / "full_feature_importance.csv",
    index=False,
)

importance_family_tf.to_csv(
    OUT
    / "full_importance_family_tf.csv",
    index=False,
)

q_calibration.to_csv(
    OUT
    / "full_q_calibration.csv",
    index=False,
)

train_action_means.to_csv(
    OUT
    / "train_action_means.csv",
    index=False,
)


audit = {
    "experiment":
        "ob_q_relationship_v1",

    "git_head":
        git_head(),

    "model_view_version":
        MODEL_VIEW_VERSION,

    "model_feature_count":
        len(
            MODEL_FEATURES_V0
        ),

    "validated_timeframes":
        list(
            VALIDATED_TFS
        ),

    "reward":
        REWARD,

    "split_method":
        "chronological_trading_day_70_30",

    "split_day":
        split_day.isoformat(),

    "all_candidate_count":
        int(
            reward_support.size
        ),

    "complete_h12_candidate_count":
        int(
            len(
                complete_ids
            )
        ),

    "partial_h12_candidate_count":
        partial_candidates,

    "zero_h12_candidate_count":
        zero_reward_candidates,

    "train_candidate_count":
        int(
            train[
                "candidate_id"
            ]
            .nunique()
        ),

    "test_candidate_count":
        int(
            test[
                "candidate_id"
            ]
            .nunique()
        ),

    "train_action_rows":
        int(
            len(
                train
            )
        ),

    "test_action_rows":
        int(
            len(
                test
            )
        ),

    "catboost_version":
        catboost.__version__,

    "catboost_params": {
        "iterations":
            400,

        "depth":
            6,

        "learning_rate":
            0.05,

        "l2_leaf_reg":
            5.0,

        "random_seed":
            SEED,

        "random_strength":
            0.0,

        "bootstrap_type":
            "No",

        "one_hot_max_size":
            32,
    },

    "model_feature_counts": {
        name:
            len(features)

        for (
            name,
            features,
        ) in MODEL_SPECS.items()
    },

    "global_train_best_action": {
        "trade_mode":
            global_mode,

        "target_R":
            global_rr,

        "train_mean_R":
            global_mean_q,
    },
}


with (
    OUT
    / "audit.json"
).open(
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        audit,
        f,
        indent=2,
        ensure_ascii=False,
    )


print()
print(
    "OB_Q_RELATIONSHIP_V1_DONE"
)

print(
    json.dumps(
        audit,
        indent=2,
        ensure_ascii=False,
    )
)
