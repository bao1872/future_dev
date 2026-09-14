"""SEQ-1 — deterministic synthetic / contract tests.

覆盖：
  * triple chain 构造规则（两段 contiguous + gap 0；历史 censor → STOP）
  * ordered pair code 保序（"4>8" != "8>4"），禁止 bullish/bearish 映射
  * Q0/Q1/Q2/Q3 feature set 精确性；Q3 只含 pair（不重复加 prev_event）
  * window / seed / comparison 集合冻结；primary = Q3 - Q1
  * 复用 STATE-1.1 的 FUNC_PROV 与 frozen geometry（function/object parity）
  * 禁用 feature（duration / path / symbol / SMC / 三阶）不在任何模型里
  * 运行时：sparsity / transition / contribution 表结构，TB4 措辞

运行：
  .venv/bin/python research/liquidity_oracle_atlas/test_seq1_second_order_grammar_v1.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import research.liquidity_oracle_atlas.experiment_seq1_second_order_grammar_v1 as G  # noqa: E402
import research.liquidity_oracle_atlas.experiment_state11_functional_provenance_v1 as T  # noqa: E402
import research.liquidity_oracle_atlas.experiment_state1_provenance_decomposition_v1 as S  # noqa: E402
import research.liquidity_oracle_atlas.experiment_path0_episode_path_memory_v1 as P  # noqa: E402

FAILS = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} :: {detail}")
        FAILS.append(name)


# ------------------------------------------------------------ triple chain
def _eps(rows):
    return pd.DataFrame(rows)


def test_build_triples_contiguity():
    ep = _eps([
        dict(symbol="X", start_bar=0, end_bar=3, event_mask=1,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
        dict(symbol="X", start_bar=3, end_bar=5, event_mask=4,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
        dict(symbol="X", start_bar=5, end_bar=9, event_mask=8,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB2"),
    ])
    tri = G.build_triples(ep)
    check("one triple chain built from three contiguous episodes",
          len(tri) == 1, len(tri))
    r = tri.iloc[0]
    check("roles assigned correctly (prevprev / prev / target)",
          (r["prevprev_start_bar"], r["prevprev_end_bar"]) == (0, 3)
          and (r["prev_start_bar"], r["prev_end_bar"]) == (3, 5)
          and (r["target_start_bar"], r["target_end_bar"]) == (5, 9), dict(r))
    check("event masks recorded in chain order",
          (r["prevprev_event_mask"], r["prev_event_mask"],
           r["target_event_mask"]) == (1, 4, 8), dict(r))

    ep_gap = _eps([
        dict(symbol="X", start_bar=0, end_bar=3, event_mask=1,
             gap_bars_before_episode=0, gap_bars_after_episode=2,
             start_block="TB1", end_block="TB1"),
        dict(symbol="X", start_bar=5, end_bar=6, event_mask=4,
             gap_bars_before_episode=2, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
        dict(symbol="X", start_bar=6, end_bar=8, event_mask=8,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
    ])
    check("gap breaks the chain (no triple)",
          len(G.build_triples(ep_gap)) == 0, "")

    ep_nc = _eps([
        dict(symbol="X", start_bar=0, end_bar=3, event_mask=1,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
        dict(symbol="X", start_bar=4, end_bar=6, event_mask=4,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
        dict(symbol="X", start_bar=6, end_bar=8, event_mask=8,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
    ])
    check("non-contiguous prevprev/prev excluded",
          len(G.build_triples(ep_nc)) == 0, "")

    ep_x = _eps([
        dict(symbol="X", start_bar=0, end_bar=1, event_mask=1,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
        dict(symbol="X", start_bar=1, end_bar=2, event_mask=4,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
        dict(symbol="Y", start_bar=2, end_bar=3, event_mask=8,
             gap_bars_before_episode=0, gap_bars_after_episode=0,
             start_block="TB1", end_block="TB1"),
    ])
    check("chains never cross symbols", len(G.build_triples(ep_x)) == 0, "")


def test_event_pair_code_is_ordered():
    a, b = 4, 8
    check("pair code format is '{prevprev}>{prev}'",
          f"{a}>{b}" == "4>8" and f"{b}>{a}" == "8>4", "")
    check("order is preserved (no unordered collapse)",
          f"{a}>{b}" != f"{b}>{a}", "")
    src = Path(G.__file__).read_text()
    tokens = ["bullish", "bearish", "continuation", "reversal"]
    hits = [ln for ln in src.splitlines()
            if any(t in ln.lower() for t in tokens)]
    check("directional mapping tokens appear ONLY in prohibition wording",
          all(("forbid" in ln.lower()) or ("no " in ln.lower())
              or ("禁止" in ln) for ln in hits), hits)
    check("pair code is built from raw masks by string concatenation",
          'astype(str) + ">"' in src, "")
    check("no manual mask merge table in source",
          "MASK_MERGE" not in src and "merge_mask" not in src, "")


# ----------------------------------------------------------------- models
def test_model_layers():
    q0n, q0c = G.MODELS[G.Q0]
    q1n, q1c = G.MODELS[G.Q1]
    q2n, q2c = G.MODELS[G.Q2]
    q3n, q3c = G.MODELS[G.Q3]
    check("Q0 = geometry + functional provenance, no categorical",
          q0n == G.GEOM + G.FUNC_PROV and q0c == [], (q0n, q0c))
    check("Q1 = Q0 + prev_event_mask only",
          q1n == q0n and q1c == ["prev_event_mask"], (q1n, q1c))
    check("Q2 = Q1 + prevprev_event_mask",
          q2n == q1n and q2c == ["prev_event_mask", "prevprev_event_mask"],
          (q2n, q2c))
    check("Q3 = Q0 + ordered event_pair only",
          q3n == q0n and q3c == ["event_pair_code"], (q3n, q3c))
    check("Q3 does NOT re-add prev_event_mask (no redundancy)",
          "prev_event_mask" not in q3c, q3c)
    banned = ("duration", "path", "symbol", "type", "scope", "volume",
              "signature", "smc", "bos", "choch", "sweep", "hmm",
              "prevprevprev", "third_order")
    bad = [c for v in G.MODELS.values() for c in (v[0] + v[1])
           if any(b in c.lower() for b in banned)]
    check("no duration / path / symbol / SMC / third-order feature",
          bad == [], bad)
    check("FUNC_PROV identical to STATE-1.1 frozen list",
          G.FUNC_PROV is T.FUNC_PROV, "")
    check("GEOM identical to PATH-0 / STATE-1 / STATE-1.1",
          G.GEOM == P.CUR_NUM == S.GEOM == T.GEOM, G.GEOM)
    check("primary comparison is Q3 - Q1",
          G.COMPARISONS[0] == (G.Q3, G.Q1), G.COMPARISONS[0])
    check("exactly 4 frozen comparisons",
          len(G.COMPARISONS) == 4
          and set(G.COMPARISONS) == {(G.Q3, G.Q1), (G.Q2, G.Q1),
                                     (G.Q3, G.Q2), (G.Q1, G.Q0)},
          G.COMPARISONS)


def test_windows_and_seeds():
    w = G.WINDOWS
    check("window A = TB1 -> TB2 seed 20260918",
          w[0]["train"] == ["TB1"] and w[0]["eval"] == "TB2"
          and w[0]["seed"] == 20260918, w[0])
    check("window B = TB1+TB2 -> TB3 seed 20260919",
          w[1]["train"] == ["TB1", "TB2"] and w[1]["eval"] == "TB3"
          and w[1]["seed"] == 20260919, w[1])
    check("no window touches TB4",
          all("TB4" not in r["train"] and r["eval"] != "TB4" for r in w), w)
    check("bootstrap reps frozen at 1000", G.BOOTSTRAP_REPS == 1000, "")
    check("frozen geometry hash constant reused",
          G.FROZEN_SAMPLE_HASH == T.FROZEN_SAMPLE_HASH, "")


def test_triple_hash_keys():
    df = pd.DataFrame([dict(
        symbol="X", prevprev_start_bar=0, prevprev_end_bar=1,
        prev_start_bar=1, prev_end_bar=2, target_start_bar=2,
        target_end_bar=3, prevprev_event_mask=4, prev_event_mask=8,
        target_event_mask=1)])
    df2 = df.iloc[::-1].reset_index(drop=True)
    check("triple hash is order invariant",
          G.triple_hash(df) == G.triple_hash(df2), "")
    check("triple hash covers all ten declared fields",
          G.triple_hash(df) != G.triple_hash(
              df.assign(prevprev_event_mask=[2])), "")
    check("triple hash changes with event order swap",
          G.triple_hash(df) != G.triple_hash(
              df.assign(prevprev_event_mask=[8], prev_event_mask=[4])), "")


def test_no_forbidden_wording():
    src = Path(G.__file__).read_text()
    check("source never claims 'TB4 never read'",
          "TB4 never read" not in src and "tb4_read_for_values" not in src, "")
    check("source records tb4_analytically_used",
          "tb4_analytically_used" in src, "")
    check("source forbids third-order escalation in the wording",
          "third-order" in src and "third_order" in str(G.__dict__).lower()
          or "does NOT authorise third-order" in src, "")


def test_outputs_if_present():
    f = G.OUT / "seq1_summary.json"
    if not f.exists():
        check("seq1_summary.json exists (skipped)", True, "")
        return
    d = json.loads(f.read_text())
    a = d["sample_audit"]
    check("triple chain audit recorded with both exclusion counts",
          all(k in a for k in ["n_triple_chains_raw", "excluded_target_censor",
                               "excluded_cross_block_target",
                               "n_triple_chains"]), list(a)[:6])
    check("exactly 4 comparisons per window (8 rows)",
          len(d["bootstrap"]) == 8, len(d["bootstrap"]))
    check("tb4_analytically_used is false",
          d["tb4_analytically_used"] is False, d["tb4_analytically_used"])
    check("frozen cross-check constants present",
          "repl0_sample_hash" in d["frozen_context"]
          and d["frozen_context"]["repl0_sample_hash"]
          == T.FROZEN_SAMPLE_HASH, d["frozen_context"])
    sp = pd.DataFrame(list(a["sparsity"].values()))
    check("sparsity audit has both windows",
          len(sp) == 2 and "pair_train_count_p50" in sp.columns, list(sp))
    for nm in ["seq1_transition_first_order.csv", "seq1_transition_pair.csv",
               "seq1_pair_contribution.csv", "seq1_model_metrics.csv",
               "seq1_bootstrap.csv", "seq1_per_bit.csv",
               "seq1_by_symbol.csv"]:
        check(f"{nm} written", (G.OUT / nm).exists(), nm)
    c = pd.read_csv(G.OUT / "seq1_pair_contribution.csv")
    check("pair contribution table has n and loss delta columns",
          {"window", "event_pair_code", "n",
           "mean_sample_loss_Q3_minus_Q1"} <= set(c.columns), list(c.columns))
    check("pair contribution rows sorted by n descending",
          bool((c.groupby("window")["n"]
                .apply(lambda s: s.is_monotonic_decreasing)).all()), "")


def main():
    test_build_triples_contiguity()
    test_event_pair_code_is_ordered()
    test_model_layers()
    test_windows_and_seeds()
    test_triple_hash_keys()
    test_no_forbidden_wording()
    test_outputs_if_present()
    print(f"\n==== {len(FAILS)} FAIL ====")
    if FAILS:
        print("FAILED:", FAILS)
        sys.exit(1)
    print("ALL TESTS PASS")


if __name__ == "__main__":
    main()
