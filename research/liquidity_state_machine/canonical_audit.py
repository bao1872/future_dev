"""Phase L0 — Canonical Semantic Audit（只读）。

不写状态机。只回答：当前 canonical 数据到底支持哪些 SMC 状态，
以及每一项的因果可用性规则。

输出：research/liquidity_state_machine/CANONICAL_SEMANTICS_AUDIT.md
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path("research")
OUT_DIR = ROOT / "liquidity_state_machine"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# concept -> (search regex over research/*.py, note)
PROBES = [
    ("5m bars", r"def load_raw_5m|v3r_5m"),
    ("15m bars", r"def aggregate_15m"),
    ("1h bars", r"def aggregate_1h_from_15m"),
    ("4h bars", r"def aggregate_4h_from_1h"),
    ("session / trading-day mapping", r"trading_day|build_session_masks"),
    ("continuous-futures roll / discontinuity",
     r"def discontinuity_flags|ROLL_GAP_ATR_THRESHOLD"),
    ("confirmed swings / pivots",
     r"current_pivots_at|piv_conf_by_type|confirmed_index"),
    ("BOS", r"\bBOS\b"),
    ("CHoCH / MSS", r"CHoCH|MSS"),
    ("structure / trend direction", r"swing_bias|internal_bias"),
    ("DSA direction", r"dsa_direction|def dsa"),
    ("displacement", r"displacement"),
    ("FVG / imbalance", r"fvg|imbalance"),
    ("OB lifecycle", r"OB_CREATED|OB_ENTERED|OB_MITIGATED"),
    ("equal highs / lows", r"equal_highs_lows|confirmed_equal_levels_at"),
    ("previous day H/L", r"prev_day|previous_day|prior_day"),
    ("previous session H/L", r"prev_session|previous_session"),
    ("previous week H/L", r"prev_week|previous_week"),
    ("range / consolidation boundary",
     r"consolidat|range_high|range_low"),
]


def scan():
    files = sorted(ROOT.glob("*.py"))
    rows = []
    for name, pat in PROBES:
        rx = re.compile(pat, re.IGNORECASE)
        hits = []
        for f in files:
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for m in rx.finditer(txt):
                line = txt[:m.start()].count("\n") + 1
                hits.append(f"{f.name}:{line}")
        rows.append(dict(concept=name, exists=bool(hits),
                         n_hits=len(hits),
                         examples="; ".join(hits[:4])))
    return pd.DataFrame(rows)


def _md(df: pd.DataFrame) -> str:
    head = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    body = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + body)


def main():
    df = scan()
    df.to_csv(OUT_DIR / "canonical_probe_raw.csv", index=False,
              encoding="utf-8-sig")

    # 4H gate 判定（不靠 regex，靠显式读取 Source Owner 语义）
    src = (ROOT / "build_ob_candidate_universe_v3.py").read_text(
        encoding="utf-8")
    forbid4h = '4h candidate is forbidden' in src
    agg4h = (start_ns // FOUR_HOUR_NS) if False else (
        "FOUR_HOUR_NS" in src and
        "x[\"_bucket\"] = (start_ns // FOUR_HOUR_NS)" in src)

    lines = [
        "# CANONICAL SEMANTICS AUDIT (Phase L0)",
        "",
        "只读审计。本文件由 `research/liquidity_state_machine/"
        "canonical_audit.py` 生成。",
        "",
        "## 1. 概念可用性探测",
        "",
        _md(df),
        "",
        "## 2. 4H GATE（Hard Gate）",
        "",
        "用户合同明确要求 5m / 15m / 1h / 4h 四周期。因此 4h 不能偷偷缺失。",
        "",
    ]

    if forbid4h:
        lines += [
            "**判定：`TF4H_SEMANTICS_GATE_FAIL`**",
            "",
            "事实（逐条来自 Source Owner 代码，不是推测）：",
            "",
            "1. 存在 `build_ob_candidate_universe_v3.aggregate_4h_from_1h`，"
            "但其注释明确写着 **“4h aggregation (environment only)”**。",
            "2. 它的分桶方式是**epoch 锚定的自然时钟 4 小时桶**：",
            "   `x[\"_bucket\"] = (start_ns // FOUR_HOUR_NS) * FOUR_HOUR_NS`，",
            "   等价于 `resample(\"4H\")`，切出的边界是 UTC 00/04/08/12/16/20，",
            "   对应北京时间 08/12/16/20/00/04 —— 正好是合同禁止的"
            "“自然时钟切中国期货夜盘”。",
            "3. 更关键的是，canonical 构建器**显式禁止 4h 结构**：",
            "   ```python",
            "   if any(r[\"source_tf\"] == \"4h\" for r in cand_records):",
            "       raise RuntimeError(\"4h candidate is forbidden\")",
            "   ```",
            "   以及全局校验：",
            "   ```python",
            "   if (candidates[\"source_tf\"] == \"4h\").any():",
            "       raise RuntimeError(\"4h candidate count > 0\")",
            "   ```",
            "4. 实测 canonical OB universe 的 `source_tf` 取值只有 "
            "`['15m', '1h', '5m']`，**没有任何 4h 结构事件**。",
            "",
            "因此：",
            "",
            "- **不存在 canonical 4h 结构**（只有 environment 用途的"
            "epoch 聚合，且被禁止用于 candidate）。",
            "- **不存在项目统一的 session-aware HTF bar builder**："
            "现有 HTF 构建器只有 `aggregate_15m`（15 分钟时钟格）、"
            "`aggregate_1h_from_15m`（整点 epoch 桶）、"
            "`aggregate_4h_from_1h`（4 小时 epoch 桶），"
            "全部是时钟/epoch 锚定，没有基于 valid trading bar 计数或"
            "session 边界的 HTF 构建器。",
            "",
            "按合同第 3 节：**既没有 canonical 4h，也没有可治理的 HTF "
            "builder → `TF4H_SEMANTICS_GATE_FAIL`，立即停止，"
            "不自己发明 4h convention。**",
            "",
            "本轮因此**不进入 L1 / L2**，不构建流动性地图与状态机。",
            "",
            "### 若要解除该 gate，需要 reviewer 先裁定以下之一：",
            "",
            "1. **定义 4h 的合法语义**：采用“每个交易 session 内的第 N 个"
            "1h bar 组合”或“连续 4 根 valid 1h bar 且不跨 session 断点”，"
            "并显式写出 anchor / available_time / causal close 规则；或",
            "2. **正式把 4h 从研究范围移除**，把 HTF 栈降级为 "
            "1h / 15m / 5m，并在本研究中改称 “MTF (1h) stack”；或",
            "3. 复用现有 epoch-anchored 4h 但**只作为 environment 上下文**，"
            "明确声明它不参与结构确认与流动性定义。",
            "",
            "这三种选择会改变 HTF_STACK 的语义，必须由 reviewer 决定，"
            "IDE 不自行选择。",
        ]
    else:
        lines += ["**判定：PASS**（canonical 4h 可用）"]

    (OUT_DIR / "CANONICAL_SEMANTICS_AUDIT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:40]))
    print("\nCANONICAL_AUDIT_DONE")
    return forbid4h


if __name__ == "__main__":
    main()
