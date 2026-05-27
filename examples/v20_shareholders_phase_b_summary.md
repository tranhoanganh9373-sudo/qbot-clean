# Phase B v20 shareholders Sidecar — 结果

**Run date:** 2026-05-26 03:10
**Base pred cache:** v17_dens_train24_predictions.parquet (Phase 2 v3)
**factor:** count_change_12m  sign=-1  horizon=5d
**λ candidates:** [0.1, 0.2, 0.3]
**IS:** 2017-01 ~ 2020-12 (48 months)
**OOS:** 2021-05 ~ 2026-04 (60 months)

## IS sweep (sorted desc by Calmar)

| phase   |   lam |   cum_% |   ann_% |   sharpe |   mdd_% |   calmar |   win_% |   avg_picks |   n_months |
|:--------|------:|--------:|--------:|---------:|--------:|---------:|--------:|------------:|-----------:|
| IS      |   0.3 | 1236.96 |   91.22 |     1.74 |  -14.98 |     6.09 |   66.67 |        3.72 |         48 |
| IS      |   0.2 | 1126.64 |   87.15 |     1.68 |  -14.98 |     5.82 |   66.67 |        3.67 |         48 |
| IS      |   0.1 | 1087.92 |   85.65 |     1.64 |  -14.98 |     5.72 |   66.67 |        3.66 |         48 |

**Locked λ = 0.3**  (best IS Calmar = 6.09)

## OOS single run

| metric | value |
|---|---|
| Calmar | 0.39 |
| Sharpe | 0.5 |
| ann %  | 13.15 |
| MDD %  | -33.74 |
| cum %  | 85.46 |
| win %  | 48.33 |
| n_months | 60 |
| avg_picks | 3.14 |

## Comparison (vs v2-era runs)

| version | OOS Calmar |
|---|---|
| Phase2_v2_baseline | 0.86 |
| v19.4_margin_v2 | 1.28 |
| v19.6_amplitude_v2 | 0.58 |
| v20_industry_60d_v2 | 0.84 |
| v20_vol_z_5d_v2 | 0.54 |
| v20_super_big_net_v2 | -0.07 |
| **v20_shareholders** | **0.39** |

**Verdict:** abort (OOS Calmar < 0.50)