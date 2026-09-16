"""
test_pgm_fixed_rr_v1.py

PGM-FIXED-RR-1 测试：
- 冻结常量：止损 1R / 止盈 2R / 成本 0.01 / 持仓 6-12-24 / 随机 seed
- 方向构建：PGM / REVERSE / RANDOM（确定性）
- 归一化相对路径数学正确性
- 触发顺序：同根 STOP FIRST、timeout 取第 hold-1 根收盘
- 跳空规则：IDEAL_FILL 与 CONSERVATIVE_GAP_THROUGH
- 成本：每笔只扣一次
- 向量化 vs 标量参照 parity（atol <= 1e-12）
- 未来函数：持仓窗口之外的未来数据不得影响结果
- 效率：张量只构建一次（N x 24），三个持仓窗口共用切片
- 治理：不复用 EXEC-1 的 selection / surface / verdict；禁止 TB4；热路径无逐笔循环
"""
from __future__ import annotations

import pathlib
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

import research.liquidity_oracle_atlas.experiment_pgm_exec1_entry_stop_target_v1 as x1
import research.liquidity_oracle_atlas.experiment_pgm_fixed_rr_v1 as frr


# ===========================================================================
# 合成数据
# ===========================================================================
def make_paths(
    n: int = 120,
    hmax: int = frr.HMAX,
    seed: int = 20260916,
) -> Dict[str, Any]:
    """生成合成未来路径与归一化相对路径。"""
    rng = np.random.default_rng(seed)
    direction = rng.choice([-1, 1], size=n).astype(np.int8)
    atr0 = rng.uniform(5.0, 20.0, size=n)
    entry = rng.uniform(100.0, 500.0, size=n)

    steps = rng.normal(0.0, 0.6, size=(n, hmax))
    close = entry[:, None] + np.cumsum(steps, axis=1) * atr0[:, None] * 0.3
    open_ = close - rng.normal(0.0, 0.2, size=(n, hmax)) * atr0[:, None] * 0.3
    high = np.maximum(open_, close) + np.abs(rng.normal(0.0, 0.5, size=(n, hmax))) * atr0[:, None] * 0.3
    low = np.minimum(open_, close) - np.abs(rng.normal(0.0, 0.5, size=(n, hmax))) * atr0[:, None] * 0.3

    # 入场价必须等于次根开盘：令 O[:, 0] == entry（与真实实验语义一致）
    open_[:, 0] = entry
    high[:, 0] = np.maximum(open_[:, 0], close[:, 0])
    low[:, 0] = np.minimum(open_[:, 0], close[:, 0])

    o_rel, h_rel, l_rel, c_rel = frr.build_relative_path(
        open_, high, low, close, entry, direction, atr0
    )
    return dict(
        n=n, hmax=hmax, direction=direction, atr0=atr0, entry=entry,
        O=open_, H=high, L=low, C=close,
        o_rel=o_rel, h_rel=h_rel, l_rel=l_rel, c_rel=c_rel,
    )


def make_synthetic_block_frame(n_bars: int = 60, n_rows: int = 24) -> Dict[str, Any]:
    """构造可供 build_block_frame 使用的合成 scored / bars。"""
    sym = "FIXSYM"
    rng = np.random.default_rng(11)
    close = 100.0 + np.cumsum(rng.normal(0.0, 0.6, size=n_bars))
    open_ = close - rng.normal(0.0, 0.2, size=n_bars)
    bars_by_sym = {
        sym: dict(
            o=open_,
            h=np.maximum(open_, close) + 0.5,
            l=np.minimum(open_, close) - 0.5,
            c=close,
            t=pd.date_range("2024-01-02 09:00:00", periods=n_bars, freq="5min").to_numpy(dtype="datetime64[ns]"),
            day=np.full(n_bars, np.datetime64("2024-01-02", "ns"), dtype="datetime64[ns]"),
            disc=np.zeros(n_bars, dtype=bool),
            n=n_bars,
        )
    }
    rows = []
    for i in range(n_rows):
        e = 3 + (i % 30)
        atr = float(rng.uniform(1.5, 3.0))
        d = 1.0 if i % 2 == 0 else -1.0
        rows.append(
            dict(
                block=frr.TB2_BLOCK,
                symbol=sym,
                entry_bar=int(e),
                base_action=float(d),
                atr0=atr,
                score_mu=float(d) * float(rng.uniform(0.2, 1.0)),
                pi=float(d * (close[e] - open_[e]) / atr),
                same_block_entry_valid=True,
                hazard=int(i % 2),
                entry_day=f"2024-01-{2 + (i % 5):02d}",
            )
        )
    return dict(scored=pd.DataFrame(rows), bars_by_sym=bars_by_sym)


# ===========================================================================
# 1. 冻结常量
# ===========================================================================
def test_frozen_rr_constants_are_fixed_2_to_1():
    """Reward/Risk 必须严格冻结 2:1，且 target 不得由 stop 推导。"""
    assert frr.STOP_R == 1.0
    assert frr.TARGET_R == 2.0
    assert frr.TARGET_R / frr.STOP_R == pytest.approx(2.0)
    assert frr.COST_ATR0 == 0.01
    assert frr.HOLD_GRID == [6, 12, 24]
    assert frr.HMAX == 24
    assert frr.RANDOM_SEED == 20260916
    assert frr.PRIMARY_GAP_RULE in frr.GAP_RULES
    # 源码层面：TARGET_R 必须是字面常量，而不是由 stop 推导出来的隐藏优化
    # （剥离注释行后再扫描，避免文档里的反例措辞误伤）
    src = pathlib.Path(frr.__file__).read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines() if not ln.lstrip().startswith("#"))
    assert "TARGET_R = 2.0" in code
    assert "STOP_R = 1.0" in code
    assert "TARGET_R = STOP_R" not in code
    assert "stop_r *" not in code
    assert "m * stop" not in code


# ===========================================================================
# 2. 方向
# ===========================================================================
def test_direction_builders_and_random_determinism():
    score = np.array([1.5, -0.2, 0.0, 3.0], dtype=float)
    assert frr.build_direction(score).tolist() == [1, -1, 0, 1]

    d = np.array([1, -1, 1], dtype=np.int8)
    assert frr.reverse_direction(d).tolist() == [-1, 1, -1]

    r1 = frr.random_direction(500, frr.RANDOM_SEED)
    r2 = frr.random_direction(500, frr.RANDOM_SEED)
    assert np.array_equal(r1, r2)
    assert set(np.unique(r1).tolist()) == {-1, 1}

    built = frr.build_all_directions(d)
    assert built[frr.DIRECTION_PGM].tolist() == [1, -1, 1]
    assert built[frr.DIRECTION_REVERSE].tolist() == [-1, 1, -1]
    assert len(built[frr.DIRECTION_RANDOM]) == 3


# ===========================================================================
# 3. 相对路径数学
# ===========================================================================
def test_relative_path_normalization_math():
    entry = np.array([100.0, 200.0])
    direction = np.array([1, -1], dtype=np.int8)
    atr0 = np.array([10.0, 10.0])
    O = np.array([[100.0, 110.0], [200.0, 190.0]])
    H = np.array([[105.0, 115.0], [205.0, 195.0]])
    L = np.array([[95.0, 105.0], [195.0, 185.0]])
    C = np.array([[102.0, 112.0], [198.0, 188.0]])

    o_rel, h_rel, l_rel, c_rel = frr.build_relative_path(O, H, L, C, entry, direction, atr0)

    # 多头：d=+1，scale = 1/10
    np.testing.assert_allclose(o_rel[0], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(h_rel[0], [0.5, 1.5], atol=1e-12)
    np.testing.assert_allclose(l_rel[0], [-0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(c_rel[0], [0.2, 1.2], atol=1e-12)
    # 空头：d=-1，方向化后与多头镜像
    np.testing.assert_allclose(o_rel[1], [0.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(h_rel[1], [-0.5, 0.5], atol=1e-12)
    np.testing.assert_allclose(l_rel[1], [0.5, 1.5], atol=1e-12)
    np.testing.assert_allclose(c_rel[1], [0.2, 1.2], atol=1e-12)
    # 入场根的相对开盘必须为 0（入场价即次根开盘）
    assert np.allclose(o_rel[:, 0], 0.0, atol=1e-15)
    # 合成路径中高低点相对入场对称 -> 多头有利极值应与多头不利极值互为反号
    assert h_rel[0][0] == pytest.approx(-l_rel[0][0])


# ===========================================================================
# 4. 触发顺序
# ===========================================================================
def _flat(n: int, val: float = 0.0) -> np.ndarray:
    return np.full((n, frr.HMAX), val, dtype=float)


def test_same_bar_stop_and_target_stop_first():
    """同根同时触发止损与止盈 -> 必须判定 STOP。"""
    h_rel = _flat(1)
    l_rel = _flat(1)
    o_rel = _flat(1)
    c_rel = _flat(1, 0.5)
    h_rel[0, 2] = 5.0   # 触及 +2R 止盈
    l_rel[0, 2] = -5.0  # 同根触及 -1R 止损

    sim = frr.simulate_fixed_rr(h_rel, l_rel, c_rel, o_rel, 6, gap_rule=frr.GAP_RULE_IDEAL)
    assert sim["exit_code"][0] == frr.STOP_EXIT
    assert sim["exit_idx"][0] == 2
    assert sim["gross_return"][0] == pytest.approx(-1.0)


def test_target_when_no_stop_and_timeout_close():
    h_rel = _flat(2)
    l_rel = _flat(2)
    o_rel = _flat(2)
    c_rel = _flat(2, 0.3)

    h_rel[0, 3] = 3.0  # row0 止盈
    # row1 无任何触发 -> timeout，取第 hold-1 根收盘
    c_rel[1, 5] = 0.75

    sim = frr.simulate_fixed_rr(h_rel, l_rel, c_rel, o_rel, 6, gap_rule=frr.GAP_RULE_IDEAL)
    assert sim["exit_code"][0] == frr.TARGET_EXIT
    assert sim["exit_idx"][0] == 3
    assert sim["gross_return"][0] == pytest.approx(2.0)

    assert sim["exit_code"][1] == frr.TIMEOUT_EXIT
    assert sim["exit_idx"][1] == 5
    assert sim["gross_return"][1] == pytest.approx(0.75)


def test_stop_is_detected_via_open_gap_as_well():
    """开盘已越过止损位也必须触发止损（不是只靠 low）。"""
    h_rel = _flat(1)
    l_rel = _flat(1)
    o_rel = _flat(1)
    c_rel = _flat(1, 0.1)
    o_rel[0, 1] = -1.5

    sim = frr.simulate_fixed_rr(h_rel, l_rel, c_rel, o_rel, 6, gap_rule=frr.GAP_RULE_IDEAL)
    assert sim["exit_code"][0] == frr.STOP_EXIT
    assert sim["exit_idx"][0] == 1


# ===========================================================================
# 5. 跳空规则
# ===========================================================================
def test_gap_rule_ideal_vs_conservative():
    """跳空越过止损位：IDEAL 固定 -1R，CONSERVATIVE 承担 gap loss。"""
    h_rel = _flat(1)
    l_rel = _flat(1)
    o_rel = _flat(1)
    c_rel = _flat(1, 0.0)
    o_rel[0, 2] = -1.75
    l_rel[0, 2] = -1.9

    ideal = frr.simulate_fixed_rr(h_rel, l_rel, c_rel, o_rel, 6, gap_rule=frr.GAP_RULE_IDEAL)
    cons = frr.simulate_fixed_rr(h_rel, l_rel, c_rel, o_rel, 6, gap_rule=frr.GAP_RULE_CONSERVATIVE)

    assert ideal["gross_return"][0] == pytest.approx(-1.0)
    assert cons["gross_return"][0] == pytest.approx(-1.75)
    # 两条规则的触发时点必须一致
    assert ideal["exit_idx"][0] == cons["exit_idx"][0] == 2
    assert ideal["exit_code"][0] == cons["exit_code"][0] == frr.STOP_EXIT

    # 普通盘中止损（开盘未越过）两种规则应当一致
    o_rel2 = _flat(1)
    l_rel2 = _flat(1)
    l_rel2[0, 1] = -1.4
    a = frr.simulate_fixed_rr(h_rel, l_rel2, c_rel, o_rel2, 6, gap_rule=frr.GAP_RULE_IDEAL)
    b = frr.simulate_fixed_rr(h_rel, l_rel2, c_rel, o_rel2, 6, gap_rule=frr.GAP_RULE_CONSERVATIVE)
    assert a["gross_return"][0] == pytest.approx(-1.0)
    assert b["gross_return"][0] == pytest.approx(-1.0)


def test_unknown_gap_rule_fails_closed():
    with pytest.raises(SystemExit) as exc:
        frr.simulate_fixed_rr(_flat(1), _flat(1), _flat(1), _flat(1), 6, gap_rule="SOMETHING")
    assert "STOP_PGM_FIXEDRR1_UNKNOWN_GAP_RULE" in str(exc.value)


# ===========================================================================
# 6. 成本
# ===========================================================================
def test_cost_is_deducted_exactly_once_per_decision():
    paths = make_paths(n=50)
    sim = frr.simulate_fixed_rr(paths["h_rel"], paths["l_rel"], paths["c_rel"], paths["o_rel"], 12)
    np.testing.assert_allclose(sim["net_return"], sim["gross_return"] - frr.COST_ATR0, atol=1e-15)
    # 本设计每笔决策必成交（entry = next open），故所有行都付费一次
    assert np.allclose(sim["net_return"], sim["gross_return"] - 0.01)


# ===========================================================================
# 7. 向量化 vs 标量参照
# ===========================================================================
def test_vectorized_vs_scalar_reference_parity():
    paths = make_paths(n=140, seed=777)
    for gap_rule in frr.GAP_RULES:
        for hold in frr.HOLD_GRID:
            sim = frr.simulate_fixed_rr(
                paths["h_rel"], paths["l_rel"], paths["c_rel"], paths["o_rel"], hold, gap_rule=gap_rule
            )
            for i in range(paths["n"]):
                ref = frr.simulate_fixed_rr_scalar_reference(
                    paths["h_rel"], paths["l_rel"], paths["c_rel"], paths["o_rel"], i, hold, gap_rule=gap_rule
                )
                assert sim["exit_code"][i] == ref["exit_code"], f"exit_code row {i} hold {hold} {gap_rule}"
                assert sim["exit_idx"][i] == ref["exit_idx"], f"exit_idx row {i} hold {hold} {gap_rule}"
                assert abs(sim["gross_return"][i] - ref["gross_return"]) <= 1e-12
                assert abs(sim["net_return"][i] - ref["net_return"]) <= 1e-12


# ===========================================================================
# 8. 未来函数 / 泄漏
# ===========================================================================
def test_future_bars_beyond_hold_do_not_affect_result():
    """持仓窗口之外的未来数据不得影响任何结果（无未来泄漏）。"""
    paths = make_paths(n=80, seed=4242)
    base = frr.simulate_fixed_rr(paths["h_rel"], paths["l_rel"], paths["c_rel"], paths["o_rel"], 6)

    tampered = {k: paths[f"{k}_rel"].copy() for k in ["h", "l", "c", "o"]}
    for k in tampered:
        tampered[k][:, 6:] += 50.0  # 粗暴改写第 7 根之后的全部未来数据
    after = frr.simulate_fixed_rr(tampered["h"], tampered["l"], tampered["c"], tampered["o"], 6)

    np.testing.assert_array_equal(base["exit_code"], after["exit_code"])
    np.testing.assert_array_equal(base["exit_idx"], after["exit_idx"])
    np.testing.assert_allclose(base["gross_return"], after["gross_return"], atol=1e-15)


def test_entry_is_next_bar_open():
    """入场价必须严格等于次根开盘 O[:, 0]（不得用任何未来价格）。"""
    paths = make_paths(n=40, seed=99)
    # 归一化后入场根的相对开盘必须为 0
    assert np.allclose(paths["o_rel"][:, 0], 0.0, atol=1e-15)

    frame = frr.build_block_frame(*(lambda b: (b["scored"], frr.TB2_BLOCK, b["bars_by_sym"]))(make_synthetic_block_frame()))
    np.testing.assert_allclose(frame["entry"], frame["tensor"]["O"][:, 0], atol=1e-15)
    # 相对路径的入场根同样为 0
    d = frame["direction"]
    o_rel, _, _, _ = frr.build_relative_path(
        frame["tensor"]["O"], frame["tensor"]["H"], frame["tensor"]["L"], frame["tensor"]["C"],
        frame["entry"], d, frame["atr0"],
    )
    assert np.allclose(o_rel[:, 0], 0.0, atol=1e-15)


def test_direction_owner_mismatch_fails_closed(monkeypatch):
    """score_mu 与 base_action 不一致时必须硬停（方向必须来自冻结所有者）。"""
    bundle = make_synthetic_block_frame()
    scored = bundle["scored"].copy()
    scored.loc[0, "score_mu"] = -scored.loc[0, "score_mu"]  # 人为破坏一致性
    with pytest.raises(SystemExit) as exc:
        frr.build_block_frame(scored, frr.TB2_BLOCK, bundle["bars_by_sym"])
    assert "STOP_PGM_FIXEDRR1_DIRECTION_OWNER_MISMATCH" in str(exc.value)


# ===========================================================================
# 9. 效率：张量只构建一次，三个持仓共用
# ===========================================================================
def test_future_tensor_built_once_and_shared_across_holds(monkeypatch):
    """N x 24 张量必须只构建一次，6/12/24 三个持仓窗口共享切片。"""
    bundle = make_synthetic_block_frame()
    calls = {"n": 0}
    real_build = x1.build_future_tensor

    def counting_build(eval_df, bars_by_sym, n_future=x1.FUTURE_BARS):
        calls["n"] += 1
        calls["n_future"] = n_future
        return real_build(eval_df, bars_by_sym, n_future=n_future)

    monkeypatch.setattr(x1, "build_future_tensor", counting_build)

    frame = frr.build_block_frame(bundle["scored"], frr.TB2_BLOCK, bundle["bars_by_sym"])
    assert calls["n"] == 1
    assert calls["n_future"] == frr.HMAX == 24

    N = frame["n_valid"]
    assert N > 0

    df_res, net_by_kind = frr.evaluate_block(frame, frr.PRIMARY_GAP_RULE)
    assert calls["n"] == 1, "三个持仓窗口不得触发额外的张量构建"
    assert len({int(r) for r in df_res["n_decisions"]}) == 1, "三个持仓必须共享同一决策宇宙"
    assert set(df_res["hold_bars"].tolist()) == set(frr.HOLD_GRID)
    for kind in frr.DIRECTION_KINDS:
        assert set(net_by_kind[kind].keys()) == set(frr.HOLD_GRID)
        for hold in frr.HOLD_GRID:
            assert len(net_by_kind[kind][hold]) == N


def test_evaluate_block_uses_single_universe_and_three_directions():
    bundle = make_synthetic_block_frame()
    frame = frr.build_block_frame(bundle["scored"], frr.TB2_BLOCK, bundle["bars_by_sym"])
    df_res, _ = frr.evaluate_block(frame, frr.PRIMARY_GAP_RULE)
    assert len(df_res) == len(frr.DIRECTION_KINDS) * len(frr.HOLD_GRID)
    assert set(df_res["direction"].tolist()) == set(frr.DIRECTION_KINDS)
    # REVERSE 与 PGM 必须来自同一批 |d|，随机方向独立于两者
    assert df_res["n_decisions"].nunique() == 1


# ===========================================================================
# 10. 治理
# ===========================================================================
def test_no_exec1_selection_or_surface_reuse():
    """本实验不得复用 EXEC-1 的 selection / surface / verdict。"""
    src = pathlib.Path(frr.__file__).read_text(encoding="utf-8")
    for forbidden in ["select_policies", "evaluate_surface", "determine_verdict", "evaluate_tb3_frozen"]:
        assert forbidden not in src, f"PGM-FIXED-RR-1 不得引用 {forbidden}"
    # 必须复用冻结所有者
    for required in ["x1.load_and_score", "x1.build_future_tensor", "e0.paired_day_mean_bootstrap"]:
        assert required in src


def test_assert_allowed_blocks_forbids_tb4():
    frr.assert_allowed_blocks(pd.DataFrame({"block": ["TB1", "TB2", "TB3"]}))
    with pytest.raises(SystemExit) as exc:
        frr.assert_allowed_blocks(pd.DataFrame({"block": ["TB2", "TB4"]}))
    assert "STOP_PGM_FIXEDRR1_FORBIDDEN_TB4" in str(exc.value)


def test_run_authorization_gate(monkeypatch):
    monkeypatch.delenv("AUTHORIZE_PGM_FIXED_RR1_RUN", raising=False)
    with pytest.raises(SystemExit) as exc:
        frr.require_run_authorization()
    assert "STOP_PGM_FIXEDRR1_RUN_NOT_AUTHORIZED" in str(exc.value)

    monkeypatch.setenv("AUTHORIZE_PGM_FIXED_RR1_RUN", "0")
    with pytest.raises(SystemExit):
        frr.require_run_authorization()

    monkeypatch.setenv("AUTHORIZE_PGM_FIXED_RR1_RUN", "1")
    frr.require_run_authorization()


def test_hot_loop_source_audit():
    frr.audit_forbidden_patterns_in_hot_loops()


def test_bootstrap_contrast_wiring():
    """bootstrap 必须复用 e0 所有者，并输出 PGM / PGM-RANDOM / PGM-REVERSE。"""
    bundle = make_synthetic_block_frame()
    frame = frr.build_block_frame(bundle["scored"], frr.TB2_BLOCK, bundle["bars_by_sym"])
    _, net_by_kind = frr.evaluate_block(frame, frr.PRIMARY_GAP_RULE)
    df_boot = frr.run_bootstrap_contrasts(
        frame["entry_day"], net_by_kind, frr.TB2_BLOCK, frr.PRIMARY_GAP_RULE, n_boot=200
    )
    assert len(df_boot) == len(frr.HOLD_GRID) * 5
    expected = {"PGM_NET", "PGM_MINUS_RANDOM", "PGM_MINUS_REVERSE", "REVERSE_NET", "RANDOM_NET"}
    assert set(df_boot["contrast"].tolist()) == expected
    for col in ["point", "ci95_lower", "ci95_upper", "p_pos"]:
        assert col in df_boot.columns
        assert np.isfinite(df_boot[col].to_numpy(float)).all()
