# Pre-P1 Freeze Closure

本文件由 `run_pre_p1_freeze_closure_v1.py` 生成（governance only）。
RR3 的数字 / `region_hash` / `policy_hash` **未被改动**；此处只补 E1.1 对称 freeze、
`benchmark_roles`、dual-conflict scope 与 root manifest。

- source_commit: `7691ff21f3d9552151cf3df3ae48e2420aa3ae92`
- `freeze_manifest_hash` = `c0b10c6ef3899b5a32ce8082128649ae8e63247d9a062f7af2b791c1253673d5`

## 1. Benchmark roles

| benchmark | role |
|---|---|
| `E1.1_MARKET` | `P1_ADJUDICATION_BASELINE` |
| `V2_RR3_STRICT_REASSESS` | `P1_ADJUDICATION_CANDIDATE` |
| `E2_RR3_STRICT_SINGLE_LIMIT` | `HISTORICAL_SEMANTIC_REFERENCE` |

> P1 只允许取 `P1_ADJUDICATION_BASELINE` 与 `P1_ADJUDICATION_CANDIDATE`。
> `HISTORICAL_SEMANTIC_REFERENCE` 作为语义回归哨兵保留（例如防止
> `AMBIGUOUS_FILL_STOP_ORDER` 的 -1R 被再次抹成 0R），不得作为 P1 比较对象。

## 2. E1.1 Hardened Market（对称 freeze）

- `policy_hash` = `ed431df02d6b55d3810fe84866e1e03a02e2217b8def4f162113e8f7c127df42`
- `execution_hash` = `04e116706853cfa860ba4083d83f0f16f4c3d42f95d06ac7778a3cee98d58bf2`
- `reward_hash` = `77fc1a6e3f6fbe8f64e16292138d16a299d921fb23898445cb01cd95c400e855`
- `timing_hash` = `0423f1e2b635bfa47fc984f5349f7422b4e58f12a514698cccc03b7d3cf281bb`
- `region_hash` = `None`（E1.1 是 market-execution baseline，不存在 R1–R4 opportunity-policy regions，因此没有 region_hash；其身份由 execution/reward/timing + function hashes 构成的 policy_hash 定义。）

### executable development replay

replay_ok = **True** (n_signals=9015)

```
 wf  n_signals  EV_cw_expected  EV_cw_actual  EV_cw_abs_diff  EV_lower_expected  EV_lower_actual  total_R_expected  total_R_actual  pass_
WF1       3160        0.010376      0.010376    1.904952e-13           0.042022         0.042022        132.788702      132.788702   True
WF2       2847        0.033118      0.033118    4.599307e-13           0.047519         0.047519        135.286334      135.286334   True
WF3       3008        0.043795      0.043795    4.064180e-13           0.076375         0.076375        229.735402      229.735402   True
```

## 3. Dual-conflict scope（原样冻结，禁止统一）

| 函数 | 语义 |
|---|---|
| `build_policy` | TB1 / development opportunity-policy construction semantics. R2+R3 both matched -> SKIP_DUAL_ACTION_CONFLICT and the elif-chain ends, so R4 is NOT reachable in that branch. |
| `route_reassess` | RR3 Strict Reassess execution lifecycle semantics. R2+R3 both available -> neither attempt is made (transition R2_R3_DUAL_CONFLICT recorded), but R4 IS still routed afterwards (separate `if R4 in avail` block). |

- `unification_forbidden` = `True`
- P1 使用：`route_reassess`
- 理由：两种语义服务不同阶段：build_policy 决定开发期机会策略构造；route_reassess 决定 P1 候选策略的执行生命周期。在 P1 前统一二者等同于修改策略，禁止。

## 4. Root manifest hash

```
freeze_manifest_hash = c0b10c6ef3899b5a32ce8082128649ae8e63247d9a062f7af2b791c1253673d5
inputs: strategy_hashes, code_hashes, input_hashes, benchmark_roles, benchmark_numbers, source_commit
```

## 5. Verification

只读校验（不写任何文件，mismatch 即 fail closed）：

```bash
.venv/bin/python research/liquidity_oracle_atlas/verify_frozen_specs_v1.py
```

## 6. 尚未完成（不在本轮范围）

- P1 adjudication contract（预注册）**未编写**。
- P1 **未读取**（`P1_read=false`）。
- `P1_read=false` 仍是**声明**；把「禁止读取 P1」升级为可执行限制属于 P1 contract 的职责。
