# Quantile V2 Cross-Instrument Robustness

AG = discovery instrument.
CU/AL/SN/I/SC/M/CF = holdout instruments.

No tuning. No backtest. No pooled model.

## Primary hypotheses on HOLDOUT_7

| hypothesis | median | positive share | min | max |
|---|---:|---:|---:|---:|
| H1_H4_GBR_F1_Q10_SKILL | 0.00247 | 0.857 | -0.00359 | 0.02937 |
| H2_H4_GBR_F1_Q90_SKILL | 0.01602 | 0.714 | -0.00245 | 0.04847 |
| H3_H4_GBR_F1_INTERVAL_SKILL | 0.01287 | 0.857 | -0.00035 | 0.03393 |
| H4_H4_GBR_F1_WIDTH_PATH_MONO | 0.96364 | 1.000 | 0.91515 | 0.97576 |
| H5_H8_GBR_F1_Q10_SKILL | -0.01009 | 0.143 | -0.01909 | 0.03833 |
| H6_H4_GBR_F2_TO_F3_TAIL_DELTA | 0.00240 | 0.714 | -0.00359 | 0.01753 |