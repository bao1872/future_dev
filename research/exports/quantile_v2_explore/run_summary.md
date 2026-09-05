# Quantile V2 State Exploration

No predictive model.
No backtest.
No p-value claim.

All state thresholds were learned from past training X only and applied to later purged OOS blocks.

## H=4 (60m)

| feature | median Q50 high-low | sign consistency | Q50 monotonicity | width high-low |
|---|---:|---:|---:|---:|
| feat_15m_volume_ratio_32 | 0.001886 | 0.800 | 0.500 | 0.000325 |
| feat_15m_ret_16 | -0.001703 | 0.800 | -0.300 | -0.001393 |
| feat_5m_4h_ret_sum | -0.001703 | 0.800 | -0.300 | -0.001393 |
| feat_5m_rv_rate_ratio_1h_4h | 0.001625 | 0.800 | 0.300 | 0.003041 |
| feat_5m_1h_max_abs_return | 0.001586 | 0.800 | 0.700 | 0.005212 |
| feat_time_bars_since_segment_start | -0.001420 | 0.800 | -0.500 | 0.001056 |
| feat_5m_4h_gap_sq | 0.001039 | 1.000 | 0.300 | -0.003491 |
| feat_5m_1h_neg_semivar_share | 0.000936 | 0.800 | 0.300 | -0.004173 |

## H=8 (120m)

| feature | median Q50 high-low | sign consistency | Q50 monotonicity | width high-low |
|---|---:|---:|---:|---:|
| feat_time_bars_since_segment_start | -0.003654 | 0.800 | -0.800 | 0.003851 |
| feat_5m_1h_positive_return_share | 0.002223 | 0.800 | 0.300 | 0.004995 |
| feat_15m_ret_8 | -0.001928 | 0.800 | -0.400 | 0.000554 |
| feat_5m_rv_rate_ratio_1h_4h | 0.001849 | 0.800 | 0.300 | 0.007147 |
| feat_5m_4h_gap_sq | 0.001464 | 1.000 | 0.600 | -0.009661 |
| feat_15m_ret_16 | -0.001761 | 0.800 | -0.300 | -0.007543 |
| feat_5m_4h_ret_sum | -0.001761 | 0.800 | -0.300 | -0.007543 |
| feat_5m_4h_neg_semivar_share | 0.001737 | 0.800 | 0.400 | 0.015969 |

## H=16 (240m)

| feature | median Q50 high-low | sign consistency | Q50 monotonicity | width high-low |
|---|---:|---:|---:|---:|
| feat_time_bars_since_segment_start | -0.005137 | 0.800 | -0.900 | -0.003896 |
| feat_5m_4h_gap_sq | 0.004739 | 0.800 | 0.300 | 0.004231 |
| feat_5m_rv_rate_ratio_4h_8h | -0.005621 | 0.600 | -0.100 | -0.008985 |
| feat_5m_4h_rv | -0.005563 | 0.600 | -0.100 | 0.006960 |
| feat_5m_4h_ret_sum | 0.004691 | 0.600 | 0.300 | -0.005700 |
| feat_15m_ret_16 | 0.004691 | 0.600 | 0.300 | -0.005700 |
| feat_5m_4h_neg_semivar_share | 0.004166 | 0.600 | 0.100 | 0.022278 |
| feat_5m_4h_jump_share | 0.003304 | 0.600 | 0.700 | 0.019408 |
