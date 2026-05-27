# Phase B v20 super_big_net Sidecar — 结果

**Run date:** 2026-05-26 02:40
**Base pred cache:** v17_dens_train24_predictions.pre_phase2_v2.bak (Phase 2 v2)
**factor:** net_super_big_5d_chg  sign=-1  horizon=5d
**λ candidates:** [0.1, 0.2, 0.3]
**IS:** 2017-01 ~ 2020-12 (48 months)
**OOS:** 2021-05 ~ 2026-04 (60 months)

## IS sweep (sorted desc by Calmar)

| phase   |   lam |   cum_% |   ann_% |   sharpe |   mdd_% |   calmar |   win_% |   avg_picks |   n_months |
|:--------|------:|--------:|--------:|---------:|--------:|---------:|--------:|------------:|-----------:|
| IS      |   0.3 |  248.32 |   36.61 |     0.88 |  -18.21 |     2.01 |   43.75 |        1.64 |         48 |
| IS      |   0.1 |  171.02 |   28.31 |     0.94 |  -16.15 |     1.75 |   43.75 |        1.78 |         48 |
| IS      |   0.2 |  194.96 |   31.05 |     0.77 |  -26.51 |     1.17 |   41.67 |        1.7  |         48 |

**Locked λ = 0.3**  (best IS Calmar = 2.01)

## OOS single run

| metric | value |
|---|---|
| Calmar | -0.07 |
| Sharpe | 0.04 |
| ann %  | -2.62 |
| MDD %  | -37.71 |
| cum %  | -12.43 |
| win %  | 43.33 |
| n_months | 60 |
| avg_picks | 2.93 |

## Comparison

| version | Calmar | Sharpe | ann % | MDD % | cum % |
|---|---|---|---|---|---|
| Phase2_v2_baseline | 0.42 | 0.68 | 12.45 | -29.86 | 79.84 |
| v19.4_margin_sidecar | 0.61 | 0.76 | 12.86 | -21.23 | 83.07 |
| v19.6_amplitude | 0.79 | — | 14.54 | -18.51 | — |
| **v20** | **-0.07** | 0.04 | -2.62 | -37.71 | -12.43 |

**Verdict:** abort (OOS Calmar < 0.50, sidecar 不生效或负贡献)