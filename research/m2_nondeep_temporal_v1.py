"""
M2: 非深度模型 × 时间连续信息实验
================================

目标：
  1. 判断 CatBoost 的筛选能力能否被 LightGBM / XGBoost 复现；
  2. 判断排序目标是否改善同一事件内六动作选择；
  3. 判断加入市场状态时间变化后，滚动历史测试表现是否稳定改善。

设计（冻结项全部保持）：
  - 数据合同 / 62 维快照特征 / 6 动作 / 未来 12 根观察期 / 交易模拟语义全部冻结。
  - 时间增强只来自决策时点之前已有的底层市场状态（momentum + DSA + 基础波动/参与），
    由冻结的 build_momentum_frame / build_dsa_frame 逐根产出，因果安全（无未来函数）。
  - 不新增技术指标；不修改止损/目标/手续费；不做深度学习/强化学习。
  - 滚动向前验证：4 折，训练区只向过去扩张，每折模型选择在 40 天选择段完成、测试段冻结。
  - 覆盖率仅 [0.10, 0.20, 0.30, 0.50]，在选择段按每日等权夏普率选择。

时间增强范围说明（诚实声明）：
  internal_bias / swing_bias / active_ob_structure_class / quant_state 等字段来自冻结的
  V3 context/levels 单点快照，无逐根（per-bar）函数可用；若强行逐根需重跑重型 V3 SMC 管线，
  本轮不做。因此时间增强只覆盖 momentum + DSA + 基础波动/参与 维度。这已足以回答
  “时间连续信息是否有稳定增量”。

运行：python -m research.m2_nondeep_temporal_v1
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from research.ob_rl_model_view_v0_spec import MODEL_FEATURES_V0
from research.rl_62d_core_v1 import ACTION_NAMES
from research.train_rl_62d_v1 import ROLL_GAP_ATR_THRESHOLD, curve_metrics
from research.export_ob_trigger_execution_v21 import load_raw_5m
from research.build_ob_rl_dataset_v0 import build_momentum_frame, build_dsa_frame
from research.ob_rl_dataset_v0_spec import VALIDATED_TFS

import catboost as cb
import lightgbm as lgb
import xgboost as xgb
from sklearn.linear_model import Ridge

SEED = 20240901
COVERAGES = [0.10, 0.20, 0.30, 0.50]
ACTIONS = ACTION_NAMES[1:]  # 6 个可交易动作
RESULTS = Path("research/analysis_results/rl_62d_m2")
RESULTS.mkdir(parents=True, exist_ok=True)

# 滚动折的累计分界比例（相对总交易日 N），对应 ~200/240/280/320/360 天（N=406 时）。
FOLD_FRACS = [200 / 406, 240 / 406, 280 / 406, 320 / 406, 360 / 406]

# 连续底层状态字段（用于时间变化/slope）
CONT_FIELDS = [
    "sqzmom_val", "sqzmom_delta", "dsa_raw_dsa_vwap_dev_pct", "vol20", "vol_part"
]
# 离散底层状态字段（用于状态切换）
DISC_FIELDS = ["momentum_direction_code", "dsa_direction"]

# 时间增强特征列名
TEMP_COLS = []
for f in CONT_FIELDS:
    TEMP_COLS += [f"{f}__d1", f"{f}__d3", f"{f}__d6", f"{f}__slope6"]
for f in DISC_FIELDS:
    TEMP_COLS += [f"{f}__changed3", f"{f}__since_change"]

MODELS = {
    "CatBoost回归": ("catboost", "reg"),
    "LightGBM回归": ("lgbm", "reg"),
    "XGBoost回归": ("xgb", "reg"),
    "Ridge回归": ("ridge", "reg"),
    "LightGBM回归+排序": ("lgbm", "reg+rank"),
    "XGBoost回归+排序": ("xgb", "reg+rank"),
}
VIEWS = ["SNAPSHOT", "TEMPORAL"]


# --------------------------------------------------------------------------- #
# 数据加载
# --------------------------------------------------------------------------- #
def load_base():
    idx = pd.read_csv("research/analysis_results/rl_62d_v1/event_index_v2.csv")
    idx["candidate_id"] = idx["candidate_id"].astype(str)
    rewards = np.load("research/analysis_results/rl_62d_v1/reward_matrix_v2.npy")
    gaps = np.load("research/analysis_results/rl_62d_v1/max_gap_atr_v2.npy")
    keep = np.nan_to_num(gaps[:, 1:], nan=0.0).max(axis=1) <= ROLL_GAP_ATR_THRESHOLD
    print(f"[load_base] idx/rewards/gaps loaded; n_kept={int(keep.sum())}", flush=True)

    order = idx["candidate_id"].to_numpy()
    keep_ids = order[keep]
    n_kept = int(keep.sum())

    # 元信息（按 candidate_id）
    state = pd.read_parquet(
        "research/analysis_results/ob_rl_dataset_v0/ob_rl_state_v0.parquet"
    )
    state["candidate_id"] = state["candidate_id"].astype(str)
    state = state.set_index("candidate_id")
    meta = state.loc[keep_ids, [
        "symbol", "trading_day", "touch_5m_bar_index",
        "candidate_group_id", "decision_weight",
    ]].copy()
    meta["trading_day"] = meta["trading_day"].astype(str)
    print("[load_base] state parquet loaded + meta built", flush=True)

    # 62 维快照特征（按动作展开，与 rewards 列 1..6 对齐）
    act = pd.read_parquet(
        "research/analysis_results/ob_rl_dataset_v0/ob_rl_action_v0.parquet"
    )
    act["candidate_id"] = act["candidate_id"].astype(str)
    act = act[act["action"].isin(ACTIONS)].copy()
    act["action"] = pd.Categorical(act["action"], categories=ACTIONS, ordered=True)
    act = act.sort_values(["candidate_id", "action"])

    # 兼容性修复：parquet 中大量数值特征被存为字符串（ArrowStringArray），
    # 且新版 pandas/pyarrow 的 dtype 不再以 'object' 表示，原 cat_cols 推导会得到
    # 空集，导致数值字符串列无法转 float（如 source_tf='5m'）而崩溃。
    # 这里显式区分：数值型字符串 -> 转 float；真·离散字符串 -> 留待编码。
    feats = list(MODEL_FEATURES_V0)
    cat_cols = []
    for c in feats:
        s = act[c]
        if pd.api.types.is_numeric_dtype(s):
            continue
        conv = pd.to_numeric(s, errors="coerce")
        if s.notna().sum() > 0 and conv.notna().sum() == s.notna().sum():
            act[c] = conv.astype(float)  # 数值型字符串 -> float
        else:
            cat_cols.append(c)  # 真·离散字符串，交给编码

    feat_map = {}
    for cid, g in act.groupby("candidate_id"):
        feat_map[cid] = g[feats].to_numpy()  # (6, 62)
    feat = np.array([feat_map[c] for c in keep_ids])  # (n_kept, 6, 62)
    snap_rows = feat.reshape(n_kept * 6, 62)
    print(f"[load_base] action parquet + 62d features built; snap_rows={snap_rows.shape}", flush=True)

    R = rewards[keep]  # (n_kept, 7)
    y_rows = R[:, 1:].reshape(-1)  # (n_kept*6,)

    ev_sym = meta["symbol"].to_numpy()
    ev_day = meta["trading_day"].to_numpy()
    ev_bar = meta["touch_5m_bar_index"].to_numpy().astype(int)
    ev_cg = meta["candidate_group_id"].to_numpy()
    ev_dw = meta["decision_weight"].to_numpy().astype(float)
    return dict(
        n_kept=n_kept, R=R, y_rows=y_rows, snap_rows=snap_rows,
        ev_sym=ev_sym, ev_day=ev_day, ev_bar=ev_bar, ev_cg=ev_cg, ev_dw=ev_dw,
        cat_cols=cat_cols, keep_ids=keep_ids,
    )


# --------------------------------------------------------------------------- #
# 时间增强：逐根市场状态面板 + 时间特征
# --------------------------------------------------------------------------- #
# 时间面板已在步骤 9（build_m2_temporal_panel.py）一次性构建，含全部底层市场状态
# 原始字段（sqzmom_val / sqzmom_delta / dsa_raw_dsa_vwap_dev_pct / vol20 / vol_part /
# momentum_direction / dsa_direction），与 build_temporal 所需的原始字段完全一致。
# 直接加载，避免每个 run() 重复调用 build_momentum_frame / build_dsa_frame（环境升级后极慢）。
PANEL_PATH = Path("research/analysis_results/m2/temporal_market_panel.parquet")
_PANEL_CACHE: dict = {}


def build_panel(sym):
    global _PANEL_CACHE
    if not _PANEL_CACHE:
        print(f"[build_panel] loading precomputed panel {PANEL_PATH}", flush=True)
        p = pd.read_parquet(PANEL_PATH)
        for s, g in p.groupby("symbol"):
            gg = g.set_index("bar_index").copy()
            gg["momentum_direction_code"] = pd.factorize(gg["momentum_direction"])[0]
            _PANEL_CACHE[s] = gg
        print(f"[build_panel] cached panels for {sorted(_PANEL_CACHE)}", flush=True)
    panel = _PANEL_CACHE[sym]
    # 防御：确保 momentum_direction_code 存在
    if "momentum_direction_code" not in panel.columns:
        panel = panel.copy()
        panel["momentum_direction_code"] = pd.factorize(panel["momentum_direction"])[0]
    return panel


def build_temporal(ev_sym, ev_bar):
    syms = pd.unique(ev_sym)
    print(f"[build_temporal] start; syms={list(syms)}", flush=True)
    out = np.full((len(ev_sym), len(TEMP_COLS)), np.nan, dtype=float)
    for sym in syms:
        panel = build_panel(sym)
        print(f"[build_temporal] {sym}: panel built; building features", flush=True)
        arr = {c: panel[c].to_numpy() for c in CONT_FIELDS + ["momentum_direction_code"]}
        arr["dsa_direction"] = panel["dsa_direction"].to_numpy().astype(float)
        mask = ev_sym == sym
        B = ev_bar[mask]
        n = len(B)
        base = np.where(mask)[0]
        for j, f in enumerate(CONT_FIELDS):
            a = arr[f]
            cur = a[B]
            for k, lag in enumerate((1, 3, 6)):
                prev = a[np.clip(B - lag, 0, None)]
                d = cur - prev
                d[B < lag] = np.nan
                out[base, j * 4 + k] = d
            # slope over last 6 bars [B-5..B]
            sl = np.full(n, np.nan)
            ok = B >= 5
            idxb = np.where(ok)[0]
            for ii in idxb:
                b = B[ii]
                win = a[b - 5:b + 1]
                if np.any(np.isnan(win)):
                    continue
                sl[ii] = np.polyfit(np.arange(6), win, 1)[0]
            out[base, j * 4 + 3] = sl
        # 离散字段
        disc_off = len(CONT_FIELDS) * 4
        for di, f in enumerate(DISC_FIELDS):
            a = arr[f]
            cur = a[B]
            changed3 = (cur != a[np.clip(B - 3, 0, None)]).astype(float)
            changed3[B < 3] = np.nan
            out[base, disc_off + di * 2] = changed3
            since = np.full(n, np.nan)
            for ii in range(n):
                b = B[ii]
                c0 = a[b]
                cnt = 0
                for step in range(1, min(b, 20) + 1):
                    if a[b - step] == c0:
                        cnt += 1
                    else:
                        break
                since[ii] = cnt
            out[base, disc_off + di * 2 + 1] = since
    return out


# --------------------------------------------------------------------------- #
# 模型
# --------------------------------------------------------------------------- #
def _split_val(Xtr, ytr):
    n = len(ytr)
    n_val = max(1, int(n * 0.15))
    return Xtr[:-n_val], ytr[:-n_val], Xtr[-n_val:], ytr[-n_val:]


def train_reg(kind, Xtr, ytr, Xte, cat_cols, seed):
    if kind == "catboost":
        Xtr_f, ytr_f, Xv_f, yv_f = _split_val(Xtr, ytr)
        # CatBoost 要求类别列为整数类型。numpy 数组为同质 dtype 无法混类型，
        # 故改用 DataFrame（按列保持 int 类型）传入，cat_features 用列索引。
        cat_idx = [MODEL_FEATURES_V0.index(c) for c in cat_cols]

        def _df_int(X):
            d = pd.DataFrame(X)
            for i in cat_idx:
                d[i] = d[i].astype("int64")
            return d

        dtr = _df_int(Xtr_f)
        dv = _df_int(Xv_f)
        dte = _df_int(Xte)
        model = cb.CatBoostRegressor(
            iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=5.0,
            loss_function="RMSE", random_seed=seed, thread_count=1,
            early_stopping_rounds=50, verbose=False,
        )
        model.fit(
            dtr, ytr_f,
            eval_set=(dv, yv_f),
            cat_features=cat_idx,
        )
        return model.predict(dte)
    if kind == "lgbm":
        Xtr_f, ytr_f, Xv_f, yv_f = _split_val(Xtr, ytr)
        dtr = lgb.Dataset(Xtr_f, ytr_f)
        dval = lgb.Dataset(Xv_f, yv_f, reference=dtr)
        bst = lgb.train(
            {"objective": "regression", "metric": "rmse", "learning_rate": 0.05,
             "num_leaves": 31, "lambda_l2": 1.0, "seed": seed, "verbose": -1},
            dtr, num_boost_round=800, valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return bst.predict(Xte)
    if kind == "xgb":
        Xtr_f, ytr_f, Xv_f, yv_f = _split_val(Xtr, ytr)
        model = xgb.XGBRegressor(
            n_estimators=800, learning_rate=0.05, max_depth=6, reg_lambda=1.0,
            random_state=seed, early_stopping_rounds=50,
        )
        model.fit(Xtr_f, ytr_f, eval_set=[(Xv_f, yv_f)], verbose=False)
        return model.predict(Xte)
    if kind == "ridge":
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(Xtr, ytr)
        return model.predict(Xte)
    raise ValueError(kind)


def train_rank(kind, Xtr, ytr_rank, groups_tr, Xte, seed):
    """组对齐切分（按事件而非按行），保证 LightGBM / XGBoost 的 group 边界完整。
    groups_tr: 长度 = 事件数，元素 = 每组行数（本项目恒为 6）。
    ytr_rank: 已展平为 (事件数*6,)，与 Xtr 每行一一对应。"""
    n_groups = len(groups_tr)
    n_val_g = max(1, int(n_groups * 0.15))
    g_tr = n_groups - n_val_g
    tr_rows = int(groups_tr[:g_tr].sum())
    va_rows = int(groups_tr[g_tr:g_tr + n_val_g].sum())
    Xtr_f = Xtr[:tr_rows]
    ytr_f = ytr_rank[:tr_rows]
    Xv_f = Xtr[tr_rows:tr_rows + va_rows]
    yv_f = ytr_rank[tr_rows:tr_rows + va_rows]
    g_f = groups_tr[:g_tr]
    g_v = groups_tr[g_tr:g_tr + n_val_g]
    if kind == "lgbm":
        dtr = lgb.Dataset(Xtr_f, ytr_f, group=g_f)
        dval = lgb.Dataset(Xv_f, yv_f, group=g_v, reference=dtr)
        bst = lgb.train(
            {"objective": "lambdarank", "metric": "ndcg", "ndcg_eval_at": [1, 2, 3],
             "learning_rate": 0.05, "num_leaves": 31, "lambda_l2": 1.0,
             "seed": seed, "verbose": -1},
            dtr, num_boost_round=800, valid_sets=[dval],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        return bst.predict(Xte)
    if kind == "xgb":
        model = xgb.XGBRanker(
            n_estimators=800, learning_rate=0.05, max_depth=6, reg_lambda=1.0,
            random_state=seed, early_stopping_rounds=50,
        )
        model.fit(Xtr_f, ytr_f, group=g_f,
                  eval_set=[(Xv_f, yv_f)], eval_group=[g_v], verbose=False)
        return model.predict(Xte)
    raise ValueError(kind)


def build_group_rank_labels(rewards):
    """组内（每事件 6 动作）按真实收益排序，相同收益并列。"""
    out = np.empty_like(rewards, dtype=np.int32)
    for e in range(rewards.shape[0]):
        vals = rewards[e]
        uniq = np.sort(np.unique(vals))
        mp = {v: i for i, v in enumerate(uniq)}
        out[e] = np.array([mp[v] for v in vals], dtype=np.int32)
    return out


# --------------------------------------------------------------------------- #
# 特征矩阵构造（含编码）
# --------------------------------------------------------------------------- #
def validate_temporal_join(Xtemp, temp_cols, n_kept, ev_ids):
    """硬断言：事件级时间特征必须一致地广播到该事件的全部 6 个动作行。

    历史 bug：曾使用 pd.concat 按行位置对齐，导致只有 1/6 动作行拿到
    时间特征、其余 5/6 全 NaN。这里用断言彻底禁止该模式再次出现。
    """
    arr = Xtemp[temp_cols].to_numpy(dtype=float)
    n_rows, n_col = arr.shape

    uniq = pd.unique(ev_ids)
    assert len(uniq) == n_kept, (
        f"candidate_id 不唯一: unique={len(uniq)} != n_kept={n_kept}"
    )
    assert n_rows == n_kept * 6, (
        f"动作行数错误: {n_rows} != {n_kept} * 6"
    )

    nan = np.isnan(arr)
    rows_all_nan = int(nan.all(axis=1).sum())
    rows_with_data = n_rows - rows_all_nan

    # 布局为「事件主序、动作次序」：reshape 后沿动作轴比较
    arr3 = arr.reshape(n_kept, 6, n_col)
    nan3 = nan.reshape(n_kept, 6, n_col)

    # 1) 缺失结构一致
    assert (nan3 == nan3[:, :1, :]).all(), (
        "同事件 6 个动作行的时间特征缺失位置不一致"
    )
    # 2) 数值完全一致
    filled = np.where(nan3, 0.0, arr3)
    assert (filled == filled[:, :1, :]).all(), (
        "同事件 6 个动作行的时间特征值不一致"
    )
    # 3) 禁止「只有部分动作行拥有时间特征」
    have3 = (~nan.all(axis=1)).reshape(n_kept, 6)
    assert (have3 == have3[:, :1]).all(), (
        "出现仅部分动作行拥有时间特征（1/6 现象）"
    )
    assert rows_all_nan % 6 == 0, (
        f"全 NaN 行数 {rows_all_nan} 不是 6 的倍数，说明未按事件整组"
    )

    rep = dict(
        action_rows=int(n_rows),
        temporal_event_rows=int(n_kept),
        unique_candidate_id_count=int(len(uniq)),
        rows_with_temporal_data=int(rows_with_data),
        rows_all_temporal_nan=int(rows_all_nan),
        events_with_temporal_data=int((~nan3.all(axis=2)).any(axis=1).sum()),
        events_all_temporal_nan=int(nan3.all(axis=2).all(axis=1).sum()),
        temporal_cols=int(n_col),
    )
    print(f"[temporal-join] {rep}", flush=True)
    return rep


def build_feature_matrices(D, temporal):
    """构造 62 维快照矩阵与时间增强矩阵。

    时间特征是【事件级】，特征矩阵是【动作级】(事件 × 6 动作)。
    两者必须用 candidate_id 做显式 many-to-one 关联，
    严禁依赖行位置（位置对齐曾造成 83.3% 行时间特征为 NaN）。
    """
    n_kept = D["n_kept"]
    snap = pd.DataFrame(D["snap_rows"], columns=MODEL_FEATURES_V0)
    cat = D["cat_cols"]
    Xsnap = snap.copy()
    for c in cat:
        Xsnap[c] = Xsnap[c].astype(str)

    if temporal is None:
        return Xsnap, Xsnap, cat

    ev_ids = np.asarray(D["keep_ids"])
    act_ids = np.repeat(ev_ids, 6)

    tdf = pd.DataFrame(temporal, columns=TEMP_COLS)
    tdf.insert(0, "candidate_id", ev_ids)

    left = Xsnap.copy()
    left.insert(0, "candidate_id", act_ids)

    Xtemp = left.merge(
        tdf, on="candidate_id", how="left", validate="many_to_one"
    )
    Xtemp = Xtemp.drop(columns=["candidate_id"])

    validate_temporal_join(Xtemp, TEMP_COLS, n_kept, ev_ids)
    return Xsnap, Xtemp, cat


def encode_for_fold(Xsnap, Xtemp, cat, tr_row):
    """返回 (Xsnap_num, Xtemp_num, cat_cols_kept) 用训练行拟合编码器。"""
    enc = {}
    for c in cat:
        vals = Xsnap[c].to_numpy()
        mp = {v: i + 1 for i, v in enumerate(pd.unique(vals[tr_row]))}
        enc[c] = mp
    # 训练行中位数用于所有数值列（快照 + 时间特征）的 NaN 填补，
    # 否则 Ridge / XGBoost 在含 NaN 的视图上会崩溃。离散（cat）列由上方映射处理。
    medians = {}
    for col in Xtemp.columns:
        if col in cat:
            continue
        try:
            arr = Xtemp[col].to_numpy(dtype=float)
        except (ValueError, TypeError):
            continue
        medians[col] = np.nanmedian(arr[tr_row])
    def to_num(X):
        Xn = X.copy()
        for c in cat:
            Xn[c] = Xn[c].map(lambda v: enc[c].get(v, 0)).astype(float)
        for col, m in medians.items():
            if col in Xn.columns:
                Xn[col] = Xn[col].fillna(m)
        return Xn
    Xsnap_num = to_num(Xsnap)
    Xtemp_num = to_num(Xtemp)
    return Xsnap_num, Xtemp_num


# --------------------------------------------------------------------------- #
# 折叠与评估
# --------------------------------------------------------------------------- #
def build_folds(ev_day, n_kept):
    uniq = np.array(sorted(pd.unique(ev_day)))
    N = len(uniq)
    bounds = [min(N - 1, max(1, int(N * f))) for f in FOLD_FRACS]
    for i in range(1, len(bounds)):
        if bounds[i] <= bounds[i - 1]:
            bounds[i] = bounds[i - 1] + 1
    bounds[-1] = min(bounds[-1], N - 1)
    folds = []
    for f in range(4):
        b0, b1, b2 = bounds[f], bounds[f + 1], bounds[f + 2] if f < 3 else N
        tr_days = set(uniq[:b0].tolist())
        se_days = set(uniq[b0:b1].tolist())
        te_days = set(uniq[b1:b2].tolist())
        folds.append((tr_days, se_days, te_days))
    return folds, uniq


def day_mask(ev_day, days):
    return np.isin(ev_day, list(days))


def select_coverage(yhat_sel, R_sel, day_sel, ev_mask_sel):
    """在选择段按每日等权夏普率选最佳覆盖率，返回 cutoff。"""
    yhat_e = yhat_sel.reshape(-1, 6)
    ev_score = yhat_e.max(axis=1)
    chosen = yhat_e.argmax(axis=1)
    best = None
    for c in COVERAGES:
        cutoff = np.quantile(ev_score, 1 - c)
        traded = ev_score > cutoff
        if traded.sum() == 0:
            continue
        realized = np.where(traded, R_sel[np.arange(len(R_sel)), chosen], 0.0)
        daily = pd.Series(realized, index=day_sel).groupby(level=0).mean()
        sh = curve_metrics(daily)["夏普率"]
        if best is None or sh > best[0] or (sh == best[0] and traded.sum() > best[1]):
            best = (sh, int(traded.sum()), float(cutoff), c)
    if best is None:
        return 0.0, 0.0
    return best[2], best[3]


def evaluate(yhat_test, R_test, day_test, cutoff, action_src):
    """action_src: 'reg' 用 yhat_test argmax；'rank' 用 rank_scores argmax。"""
    yhat_e = yhat_test.reshape(-1, 6)
    ev_score = yhat_e.max(axis=1)
    traded = ev_score > cutoff
    if action_src == "rank":
        raise RuntimeError("rank action handled separately")
    chosen = yhat_e.argmax(axis=1)
    realized = np.where(traded, R_test[np.arange(len(R_test)), chosen], 0.0)
    daily = pd.Series(realized, index=day_test).groupby(level=0).mean()
    met = curve_metrics(daily)
    pf = _pf(realized[traded]) if traded.sum() else 0.0
    return daily, realized, traded, chosen, dict(
        cum=met["累计收益"], sharpe=met["夏普率"], mdd=met["最大回撤"],
        pf=pf, trades=int(traded.sum()),
        no_trade=float(1 - traded.sum() / len(traded)),
        sel_avg=float(realized[traded].mean()) if traded.sum() else None,
    )


def _pf(r):
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return 0.0
    win = r[r > 0].sum()
    loss = -r[r < 0].sum()
    if loss <= 0:
        return float("inf") if win > 0 else 0.0
    return float(win / loss)


def ranking_ability(yhat_test, R_test):
    yhat_e = yhat_test.reshape(-1, 6)
    yhat_e = np.where(np.isnan(yhat_e), -1e9, yhat_e)
    chosen = yhat_e.argmax(axis=1)
    true_best = R_test.argmax(axis=1)
    top1 = float((chosen == true_best).mean())
    order = np.argsort(-R_test, axis=1)
    top2 = float(np.mean([chosen[i] in order[i, :2] for i in range(len(R_test))]))
    dir_acc = float((R_test[np.arange(len(R_test)), chosen] > 0).mean())
    regret = float((R_test.max(axis=1) - R_test[np.arange(len(R_test)), chosen]).mean())
    sp = []
    for i in range(len(R_test)):
        if np.std(yhat_e[i]) > 0 and np.std(R_test[i]) > 0:
            sp.append(np.corrcoef(yhat_e[i], R_test[i])[0, 1])
    sp = float(np.nanmean(sp))
    return dict(top1=top1, top2=top2, spearman=sp, dir_acc=dir_acc, regret=regret)


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def run():
    D = load_base()
    n_kept = D["n_kept"]
    R = D["R"][:, 1:]  # (n_kept, 6)
    ev_sym = D["ev_sym"]
    ev_day = D["ev_day"]
    ek2 = np.repeat(np.arange(n_kept), 6)
    act_k = np.tile(np.arange(6), n_kept)
    y_rows = D["y_rows"]

    temporal = build_temporal(ev_sym, D["ev_bar"])
    print(f"temporal features built: shape={temporal.shape}, nan%="
          f"{np.isnan(temporal).mean():.2%}")

    Xsnap, Xtemp, cat = build_feature_matrices(D, temporal)
    folds, uniq = build_folds(ev_day, n_kept)
    print(f"folds built: N_days={len(uniq)}, bounds="
          f"{[len(f[0]) for f in folds]} train / "
          f"{[len(f[1]) for f in folds]} sel / {[len(f[2]) for f in folds]} test")

    # 结果容器
    results = []          # 主表
    temp_delta = []       # 快照 vs 时间增强
    rank_tab = []         # 排序能力
    by_sym = []           # 分品种
    by_fold = []          # 分折

    # 累积测试事件：(day, realized, sym) 每个 (model, view)
    accum = {(m, v): {"day": [], "r": [], "sym": [], "fold": []}
             for m in MODELS for v in VIEWS}

    for mname, (mkind, mrole) in MODELS.items():
        for vname in VIEWS:
            Xall = Xsnap if vname == "SNAPSHOT" else Xtemp
            fold_dailies = []
            fold_realized = []
            fold_acc = {"day": [], "r": [], "sym": [], "fold": []}
            rank_rec_reg = None
            rank_rec_rank = None
            for fi, (tr_days, se_days, te_days) in enumerate(folds):
                tr_ev = day_mask(ev_day, tr_days)
                se_ev = day_mask(ev_day, se_days)
                te_ev = day_mask(ev_day, te_days)
                tr_row = np.repeat(tr_ev, 6)
                se_row = np.repeat(se_ev, 6)
                te_row = np.repeat(te_ev, 6)

                Xsnap_num, Xtemp_num = encode_for_fold(Xsnap, Xtemp, cat, tr_row)
                Xn = Xsnap_num if vname == "SNAPSHOT" else Xtemp_num

                Xtr = Xn.iloc[tr_row].to_numpy(float)
                ytr = y_rows[tr_row]
                Xse = Xn.iloc[se_row].to_numpy(float)
                Xte = Xn.iloc[te_row].to_numpy(float)

                yhat_se = train_reg(mkind, Xtr, ytr, Xse, cat, SEED)
                yhat_te = train_reg(mkind, Xtr, ytr, Xte, cat, SEED)

                cutoff, _ = select_coverage(
                    yhat_se, R[se_ev], ev_day[se_ev], se_ev)

                # 纯回归动作选择评估
                daily, realized, traded, chosen, met = evaluate(
                    yhat_te, R[te_ev], ev_day[te_ev], cutoff, "reg")
                fold_dailies.append(daily)
                fold_realized.append(realized)
                fold_acc["day"].append(ev_day[te_ev])
                fold_acc["r"].append(realized)
                fold_acc["sym"].append(ev_sym[te_ev])
                fold_acc["fold"].append(np.full(len(realized), fi))

                # 排序能力（回归 argmax）
                if rank_rec_reg is None:
                    rank_rec_reg = ranking_ability(yhat_te, R[te_ev])
                else:
                    a = ranking_ability(yhat_te, R[te_ev])
                    for k in rank_rec_reg:
                        rank_rec_reg[k] = (rank_rec_reg[k] + a[k]) / 2

                # 排序组合：需要排序模型
                if mrole == "reg+rank":
                    ytr_rank = build_group_rank_labels(R[tr_ev]).reshape(-1)
                    groups_tr = np.full(int(tr_ev.sum()), 6, dtype=int)
                    yhat_rank_se = train_rank(mkind, Xtr, ytr_rank, groups_tr, Xse, SEED)
                    yhat_rank_te = train_rank(mkind, Xtr, ytr_rank, groups_tr, Xte, SEED)
                    # 用排序 argmax 作为动作，回归 event_score 做筛选
                    yhat_e = yhat_te.reshape(-1, 6)
                    ev_score = yhat_e.max(axis=1)
                    tr_ev_score = ev_score
                    se_ev_score = yhat_se.reshape(-1, 6).max(axis=1)
                    cutoff_r = np.quantile(se_ev_score, 1 - 0.30)  # 默认 30%
                    # 重新以 30% 为折中（与回归一致选最佳）
                    bestc = None
                    for c in COVERAGES:
                        co = np.quantile(se_ev_score, 1 - c)
                        td = se_ev_score > co
                        if td.sum() == 0:
                            continue
                        rs = np.where(td, R[se_ev][np.arange(len(R[se_ev])),
                                            yhat_rank_se.reshape(-1, 6).argmax(axis=1)], 0.0)
                        dd = pd.Series(rs, index=ev_day[se_ev]).groupby(level=0).mean()
                        sh = curve_metrics(dd)["夏普率"]
                        if bestc is None or sh > bestc[0]:
                            bestc = (sh, co)
                    cutoff_rank = bestc[1]
                    traded_r = tr_ev_score > cutoff_rank
                    ch = yhat_rank_te.reshape(-1, 6).argmax(axis=1)
                    rs = np.where(traded_r, R[te_ev][np.arange(len(R[te_ev])), ch], 0.0)
                    dd = pd.Series(rs, index=ev_day[te_ev]).groupby(level=0).mean()
                    met_r = curve_metrics(dd)
                    # 用排序版覆盖 met（组合模型以排序动作为准）
                    met = dict(
                        cum=met_r["累计收益"], sharpe=met_r["夏普率"],
                        mdd=met_r["最大回撤"], pf=_pf(rs[traded_r]) if traded_r.sum() else 0.0,
                        trades=int(traded_r.sum()),
                        no_trade=float(1 - traded_r.sum() / len(traded_r)),
                        sel_avg=float(rs[traded_r].mean()) if traded_r.sum() else None,
                    )
                    if rank_rec_rank is None:
                        rank_rec_rank = ranking_ability(yhat_rank_te, R[te_ev])
                    else:
                        a = ranking_ability(yhat_rank_te, R[te_ev])
                        for k in rank_rec_rank:
                            rank_rec_rank[k] = (rank_rec_rank[k] + a[k]) / 2
                    # 组合模型以排序动作为准：用排序动作的 per-event realized 覆盖
                    # 前面追加的回归动作 realized（day/sym/fold 长度一致，仅动作不同）。
                    fold_acc["r"][-1] = rs

            # 汇总该 (model, view)
            all_day = np.concatenate(fold_acc["day"])
            all_r = np.concatenate(fold_acc["r"])
            all_sym = np.concatenate(fold_acc["sym"])
            all_fold = np.concatenate(fold_acc["fold"])
            daily_all = pd.Series(all_r, index=all_day).groupby(level=0).mean().sort_index()
            cm = curve_metrics(daily_all)
            pf = _pf(all_r[all_r != 0])
            results.append(dict(
                模型=mname, 特征视图=vname,
                累计收益=round(float(cm["累计收益"]), 4),
                夏普率=round(float(cm["夏普率"]), 4),
                最大回撤=round(float(cm["最大回撤"]), 4),
                pf=round(pf, 4) if np.isfinite(pf) else (1e9 if pf > 0 else 0.0),
                交易次数=int((all_r != 0).sum()),
                不交易比例=round(float(1 - (all_r != 0).mean()), 4),
                选中平均收益=round(float(all_r[all_r != 0].mean()), 6) if (all_r != 0).sum() else None,
            ))
            accum[(mname, vname)]["day"] = all_day.tolist()
            accum[(mname, vname)]["r"] = all_r.tolist()
            accum[(mname, vname)]["sym"] = all_sym.tolist()
            accum[(mname, vname)]["fold"] = all_fold.tolist()

            # 分折
            for fi in range(4):
                m = all_fold == fi
                if m.sum() == 0:
                    continue
                dm = pd.Series(all_r[m], index=all_day[m]).groupby(level=0).mean()
                cf = curve_metrics(dm)
                by_fold.append(dict(
                    模型=mname, 特征视图=vname, 折=f"F{fi+1}",
                    累计收益=round(float(cf["累计收益"]), 4),
                    夏普率=round(float(cf["夏普率"]), 4),
                    交易次数=int((all_r[m] != 0).sum()),
                ))
            # 分品种
            for s in pd.unique(all_sym):
                m = all_sym == s
                ds = pd.Series(all_r[m], index=all_day[m]).groupby(level=0).mean()
                cs = curve_metrics(ds)
                by_sym.append(dict(
                    模型=mname, 特征视图=vname, 品种=s,
                    累计收益=round(float(cs["累计收益"]), 4),
                    夏普率=round(float(cs["夏普率"]), 4),
                    交易次数=int((all_r[m] != 0).sum()),
                ))
            # 排序能力记录
            if mrole == "reg+rank":
                rec = rank_rec_rank
                rec_src = "排序模型"
            else:
                rec = rank_rec_reg
                rec_src = "回归模型"
            rank_tab.append(dict(
                模型=mname, 特征视图=vname, 动作选择来源=rec_src,
                第一名命中率=round(rec["top1"], 4),
                前二名命中率=round(rec["top2"], 4),
                事件内排序相关=round(rec["spearman"], 4),
                方向正确率=round(rec["dir_acc"], 4),
                平均动作遗憾=round(rec["regret"], 6),
            ))
            print(f"  done {mname} / {vname}: sharpe={cm['夏普率']:.3f} "
                  f"cum={cm['累计收益']:.3f} pf={pf:.3f} trades={(all_r!=0).sum()}")

    # 快照 vs 时间增强
    res_idx = {(r["模型"], r["特征视图"]): r for r in results}
    for mname in MODELS:
        sk, tk = (mname, "SNAPSHOT"), (mname, "TEMPORAL")
        if sk in res_idx and tk in res_idx:
            s = res_idx[sk]
            t = res_idx[tk]
            temp_delta.append(dict(
                模型=mname, 快照夏普=s["夏普率"], 时间增强夏普=t["夏普率"],
                差值=round(t["夏普率"] - s["夏普率"], 4),
                快照收益=s["累计收益"], 时间增强收益=t["累计收益"],
            ))

    # 写出
    pd.DataFrame(results).to_csv(RESULTS / "m2_main.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(temp_delta).to_csv(RESULTS / "m2_temporal.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(rank_tab).to_csv(RESULTS / "m2_ranking.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(by_sym).to_csv(RESULTS / "m2_by_symbol.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(by_fold).to_csv(RESULTS / "m2_by_fold.csv", index=False, encoding="utf-8-sig")

    # 审计
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    audit = dict(
        script="research/m2_nondeep_temporal_v1.py",
        base_commit="9cfcb0218927a01b699bbab3ea55d5eaec4ca098",
        current_head=head,
        seed=SEED, coverages=COVERAGES, n_kept=int(n_kept),
        n_days=int(len(uniq)), fold_bounds=[int(len(f[0])) for f in folds],
        temporal_features=TEMP_COLS,
        temporal_scope_note=(
            "时间增强仅覆盖 momentum+DSA+基础波动/参与的逐根状态；"
            "internal_bias/swing_bias/active_ob_structure_class/quant_state "
            "为冻结单点快照，无逐根函数，本轮不纳入时间维度。"
        ),
        models=list(MODELS.keys()),
    )
    (RESULTS / "m2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== M2 主表 ===")
    print(pd.DataFrame(results).to_string(index=False))
    print("\n=== 快照 vs 时间增强 ===")
    print(pd.DataFrame(temp_delta).to_string(index=False))
    print("\n=== 排序能力（回归 vs 排序动作选择）===")
    print(pd.DataFrame(rank_tab).to_string(index=False))
    print(f"\nRL_62D_M2_DONE head={head}")


if __name__ == "__main__":
    run()
