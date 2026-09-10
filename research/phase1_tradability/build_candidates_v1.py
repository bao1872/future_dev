"""Phase 1 候选事件画像：确认 canonical trigger 是否已覆盖全部 source_tf。

候选定义：canonical entered == True（沿用已确认的 entered 语义）。
不做任何 source_tf / OB宽度 / 历史收益 / 方向 / 未来结果 的提前过滤。

输出：
    candidate_profile.csv
    candidates_v1.parquet（可再生，不入库）
"""
from __future__ import annotations

import pandas as pd

from research.phase1_tradability.phase1_contract_v1 import (
    EVENT_INDEX, RESULTS, STATE_PARQUET,
)


def main():
    st = pd.read_parquet(STATE_PARQUET)
    st["candidate_id"] = st["candidate_id"].astype(str)
    assert st["candidate_id"].is_unique, "candidate_id 不唯一"

    idx = pd.read_csv(EVENT_INDEX)
    idx["candidate_id"] = idx["candidate_id"].astype(str)

    rows = [dict(维度="总计", 取值="ALL", 候选数=len(st))]
    for k, v in st["source_tf"].value_counts().items():
        rows.append(dict(维度="source_tf", 取值=str(k), 候选数=int(v)))
    for k, v in st["symbol"].value_counts().items():
        rows.append(dict(维度="symbol", 取值=str(k), 候选数=int(v)))
    for (s, tf), v in st.groupby(["symbol", "source_tf"]).size().items():
        rows.append(dict(维度="symbol×source_tf", 取值=f"{s}|{tf}",
                         候选数=int(v)))

    inter = len(set(st["candidate_id"]) & set(idx["candidate_id"]))
    rows.append(dict(维度="与event_index交集",
                     取值=f"state={len(st)} index={len(idx)}",
                     候选数=int(inter)))
    prof = pd.DataFrame(rows)
    prof.to_csv(RESULTS / "candidate_profile.csv", index=False,
                encoding="utf-8-sig")

    keep = st[[
        "candidate_id", "candidate_group_id", "symbol", "source_tf",
        "touch_5m_bar_index", "trading_day", "touch_time",
    ]].copy()
    keep.to_parquet(RESULTS / "candidates_v1.parquet", index=False)

    print("=== 候选事件画像 ===")
    print(prof.to_string(index=False))
    print(f"\nstate 候选 {len(st)} 已覆盖 source_tf = "
          f"{sorted(st['source_tf'].unique().tolist())}")
    print(f"与 event_index({len(idx)}) 交集 = {inter}")
    print("CANDIDATES_DONE")


if __name__ == "__main__":
    main()
