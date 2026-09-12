# V2 Strategy Specification Freeze

- candidate: **RR3_STRICT_REASSESS** — `DEVELOPMENT_FROZEN_PENDING_P1`
- development candidate only; NOT 'optimal', 'final', 'validated' or 'live' strategy
- source_commit: `7691ff21f3d9552151cf3df3ae48e2420aa3ae92`
- `region_hash` = `7d488897be3e699410ebe6340d033f22d2431c5d89074c7efd0fbe076a3c23c7`
- `policy_hash` = `21adee821eb3c5278b35d1100e4ec071d9054e3e476fd06ca476073e205c4e42`
- P1_read = `False`

## 1. Opportunity policy (R1–R4, frozen)

```
R1: action=OUTWARD h=2 scale=1.2 target_atr=[3.0,5.0) risk_atr=[1.0,2.0)
R2: action=OUTWARD h=5 scale=1.2 target_atr=[0.5,1.0) risk_atr=[1.0,2.0)
R3: action=INWARD h=5 scale=1.2 target_atr=[0.5,1.0) risk_atr=[1.0,2.0)
R4: action=OUTWARD h=8 scale=1.2 target_atr=[0.5,1.0) risk_atr=[1.0,2.0)
```
Precedence（first-match, at most one trade per gid）:

- R1 -> ENTER_R1
- R2 & R3 both matched -> SKIP_DUAL_ACTION_CONFLICT (no trade)
- R2 -> ENTER_R2
- R3 -> ENTER_R3
- R4 -> ENTER_R4
- no region -> SKIP_NO_GEOMETRY_EDGE
- at most one trade per gid (asserted MULTIPLE_ENTRY)

## 2. Execution policy

```
{
  "rr_target": 3.0,
  "fill_model": "STRICT_TRADE_THROUGH",
  "execution_mode": "REASSESS_NEXT_H",
  "ttl_active_bars": 2,
  "limit_price": "(target + rr*stop) / (1 + rr)",
  "limit_formula_symmetric": true,
  "fill_statuses": [
    "FILLED",
    "FILLED_AT_OPEN"
  ],
  "terminal_statuses": [
    "DATA_END",
    "DISCONTINUITY_BEFORE_LIMIT_ACTIVATION"
  ],
  "route": "R1 -> (if not filled and not terminal) R2 or R3 -> R4; stop at first FILL; stop at TERMINAL",
  "dual_conflict": "if both R2 and R3 available, neither is attempted; transition R2_R3_DUAL_CONFLICT is recorded; R4 still allowed",
  "strict_trade_through": "fill requires strict inequality (LONG: L < limit, SHORT: H > limit)",
  "marketable_open": "FILLED_AT_OPEN when bar-0 open crosses the limit (no prefill / gap / discontinuity)"
}
```

## 3. Reward / outcome semantics

```
{
  "target": "frozen target_price",
  "stop": "frozen stop_price",
  "outcome_window_bars": 34,
  "censor": "both bounds unseen inside window -> censored",
  "ambiguous_fill_target_order": {
    "upper": "rr",
    "lower": 0.0
  },
  "ambiguous_fill_stop_order": {
    "upper": 0.0,
    "lower": -1.0
  },
  "prefill_statuses": [
    "BOTH_BOUNDARIES_TOUCHED_BEFORE_ENTRY",
    "STOP_INVALIDATED_BEFORE_ENTRY",
    "TARGET_CONSUMED_BEFORE_ENTRY"
  ],
  "censor_worst": "per signal: mean of R_lower where filled & resolved, -1 for filled & censored, 0 for not filled",
  "single_limit_censor_worst": "frozen E2: R_lower kept verbatim for every non-NaN row (incl. AMBIGUOUS_FILL_STOP_ORDER = -1R); NaN only for filled & censored -> -1"
}
```

> 特别记录（a33d4eb 修复）：`AMBIGUOUS_FILL_STOP_ORDER → lower = -1R`，且**不得**被 `np.where(filled, R_lower, 0)` 抹成 0。

## 4. Timing contract

```
{
  "t0": "contact decision_time (liquidity field frozen at t0)",
  "signal_bar_index": "contact_bar_index + 1 + h",
  "validation_bar": "signal_bar_index (prefill gate bar)",
  "entry_bar_index": "signal_bar_index + 1",
  "actionable_entry": "open of entry_bar_index",
  "reward_end_bar_index": "max(entry_bar+33 MARKET, entry_bar+34 LIMIT_RR3, contact_bar+44 REASSESS_RR3)",
  "reward_end_time": "authoritative BAR END of reward_end_bar_index (= bar_start_time + 5min; availability_time == bar_end_time)",
  "purge": "train reward_end_time < validation_start",
  "bar_label": "INTERVAL END; bar_start_time = bar_end - 5min"
}
```

> 最易犯错处：某 geometry 在 next open 才可知，不可假设同一 open 成交。frozen `entry_bar_index = signal_bar_index + 1`，成交价 = 该 bar 的 open。

## 5. Hashes

### code files (sha256)
```
5dd7eae5dd96c2199c58051960a2ea1a2b01916bb68b7f471eb5f6d40b35d8d2  research/liquidity_oracle_atlas/run_enter_skip_selection_v1.py
226f6860e3828f3a1592f167acec4d7952a830eddcbb5671b15edd43a92218c8  research/liquidity_oracle_atlas/run_execution_limit_frontier_v1.py
7da37dbeae42137a50051c9c09e39ce80e18f5ad049166e7fcd1810769d43200  research/liquidity_oracle_atlas/run_execution_limit_frontier_v1_closure.py
2df9448559af63fc5ad0dd2216c36a8f71f694d093018c2262298c3ddac9e00e  research/liquidity_oracle_atlas/run_liquidity_field_action_surface_v1.py
f2258fb85c1bef0ddf6c71e00318d027121eb009b7397d8af8736710c2038110  research/liquidity_oracle_atlas/run_latent_state_compression_v1.py
b03f609e593d483c14272d156516923ceb785b9425dd635f24a8077732055446  research/liquidity_oracle_atlas/run_fixed_execution_baseline_v1.py
30e59733c26abcf19ba1ad5544cb1525a4f4cc6c8b0d3dd88938d71d1b53e203  research/liquidity_oracle_atlas/run_v2_ml_recency_multi_action_v1.py
9597b7517bad69ee763575ccc0f433728d929fee4076dbb8b473d4342ca1354d  research/liquidity_oracle_atlas/build_oracle_atlas_v1_2.py
```
### input artifacts (sha256)
```
9a48cc35987b85970dab60219ed5e276fb85096802ada4dd56578e9597cbda4f  research/analysis_results/smc_oracle_atlas_v1/liquidity_master_v1_1.parquet
e172c39dccb8f4c71f6ec1643fa416f897320703801b4144d095d133655781ab  research/analysis_results/smc_oracle_atlas_v1/liquidity_contacts_v1_1.parquet
b8a1189025da44f9820808ea17c2baa4ca6ac8a519715aeb5fbd9dbf863331c6  research/analysis_results/execution_frontier_v1/execution_lag1_trades.parquet
```
### critical function hashes
```
818cbc7093e6ccbe8e799151faf3a005c96926f70a755fdff90fe6e2be0ab9e6  build_policy
cf8126f51b822b8f3a151e737370029ad7c8dfbe6e1c52e4f3089ac6fe93b718  primary_matches
fdbe0a3e2474e9b0629a54ec800126ade540967fb1954792ab9b1a6bc05b2920  run_block
0d4ccca7df59891c024a1eebf4a4fd479789965b50d5aa2ae8b5cd514b2b90b2  prefill
09b14de8a4678ca18de8e123f49c370710736a12419bf506cbdb7348dc18dee6  route_reassess
6a6a0476592e54b637712cef031da2b11fad12ea1ed8c7228e768f735abd812e  scalar_outcome
ae37061720aa537a0a22aa40df2ece50d6dd54403fd0f878e19c96ce0fdb8c58  first_hit_bounds
b181bcabed93f400a1f1e635eab3a1704c19e7265d3fa03e8d916c0556072ce8  surviving_field_and_target
4dc1d3c74cf3fcaf1822e967524d0f5faf2728bcbed7c6d0b2e54e12be00a020  latest_confirmed_extreme
5b25c0d6fd90b7f5323eededb837c8d401b1f0e97e281e1f80a377fae5f8d3eb  attach_exact_reward_end
7725e1d40d7416ab18d4f36d356424287876f4a27747b5e7e318499a8dc687fb  conservative_reward
0b7a5b5c939ff0a3688d1fb1846e08ccad4ec06186870969207fc0f0243eb70b  e2_single_limit_censor_worst
```

## 6. Frozen benchmark + §9 executable replay

replay_ok = **True** (n_signals=9015, cache rebuilt from frozen pipeline)

```
                    metric  wf  expected    actual     abs_diff  pass_
               E1.1_MARKET WF1  0.010376  0.010376 1.904952e-13   True
               E1.1_MARKET WF2  0.033118  0.033118 4.599307e-13   True
               E1.1_MARKET WF3  0.043795  0.043795 4.064180e-13   True
    V2_RR3_STRICT_REASSESS WF1  0.034226  0.034226 3.469239e-13   True
    V2_RR3_STRICT_REASSESS WF2  0.044369  0.044369 1.414563e-13   True
    V2_RR3_STRICT_REASSESS WF3  0.104135  0.104135 4.162365e-13   True
E2_RR3_STRICT_SINGLE_LIMIT WF1 -0.005964 -0.005964 6.844178e-14   True
E2_RR3_STRICT_SINGLE_LIMIT WF2 -0.016046 -0.016046 4.830615e-13   True
E2_RR3_STRICT_SINGLE_LIMIT WF3  0.058922  0.058922 1.369460e-13   True
```

## 7. Superseded / closed lines

- V2 absolute multi-action Q → `STOP_CURRENT_MULTI_ACTION_ML` (a33d4eb)
- V3-A1 oracle headroom (descriptive) → `ORACLE_HEADROOM_MATERIAL` (7691ff2)
- V3-A2 relative advantage learnability → `NO_LEARNABLE_RELATIVE_ADVANTAGE` (ce1cb26)

## 8. Freeze rules

- 本文件之后**不得**再改动策略自由度（geometry / execution / reward / timing）。
- 唯一允许的后续动作：P1 adjudication contract（预注册），然后一次性开封 P1。
- P1 只比较两个**预先冻结**对象：`E1.1 Hardened Market` vs `RR3 Strict Reassess`；不得带入 Oracle / ML / S2 / time-decay / best-symbol / region diagnostics。
- 本 spec 的 `status` 在 P1 完成前不得升级，名称不得脱离 **development candidate**。
