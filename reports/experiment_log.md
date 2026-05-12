# Experiment Log

Competition: `stock-market-signal-predict-next-day-returns`

## Summary

Best recorded public leaderboard score: `0.52029`.

Current strongest family:

- Prior best rank anchor plus a TabICL v2 single-estimator signal.
- Useful TabICL seed `123` rank-blend weights: `3%`, `5%`, `8%`, `10%`, and `12%` all scored `0.52028`.
- Fresh TabICL seed `314159` reached `0.52029` at `8%` over the `rs123_w050` anchor.
- `rs123` at `15%` dropped to `0.52027`, so that expansion was stopped.

## Public Baselines Reviewed

- Emanuel Lazaro: LGBM/XGB/MLP/LR blend with adversarial validation and feature diagnostics.
- LightningV08: blind blend and single TabICL public baseline.
- Ravi Ramakrishnan: public baselines and blind blend.
- Yihan Guo: XGB/KFold seed ensemble.
- Alana Monks, Junaid512, Nina2025, Himanshu Dhiman, Swas06, Minn Minn Cherry, and other public notebooks.

Key finding: random cross-validation was not reliable for this competition. Public leaderboard probes and time-aware holdouts were more informative than random folds.

## Submission History

### 2026-04-26

| File | Public LB | Readout |
|---|---:|---|
| `public_rank_blend_v1.csv` | `0.51784` | Broad public rank blend. |
| `public_rank_blend_plus_contrarian_v1.csv` | `0.51838` | Public blend plus time signal. |
| `emanuellcs_stock_market_signal_predict_next_day_returns/submission.csv` | `0.51886` | Strongest single public anchor at the time. |
| `contrarian_signal.csv` | `0.51288` | Pure time-holdout contrarian signal. |
| `emanuellcs70_contrarian30_rank.csv` | `0.51953` | Narrow Emanuel plus contrarian blend. |

### 2026-04-27

| File | Public LB | Readout |
|---|---:|---|
| `narrow_em_contr_light_alan_rank.csv` | `0.51931` | Four-way blend hurt versus anchor. |
| `em675_contr324_rank.csv` | `0.51953` | Tied best. |
| `em725_contr275_rank.csv` | `0.51950` | Slightly worse. |
| `em650_contr350_rank.csv` | `0.51949` | Slightly worse. |
| `lb_corr_opt_1.csv` | `0.51785` | Correlation optimizer overfit. |

### 2026-05-05

| File | Public LB | Readout |
|---|---:|---|
| `bestv1_tabiclv2_est2_rs777_w350.csv` | `0.51837` | Higher-estimator replacement hurt badly. |
| `bestv1_tabiclv2_est1_w350_ravi_pub_v1_w005.csv` | `0.52027` | Public-output microblend tied old best. |
| `bestv1_tabiclv2_est1_w350_ravi_pub_v1_w010.csv` | `0.52026` | More public-output weight hurt. |
| `bestv1_tabiclv2_est1_w350_yihan_xgb_w005.csv` | `0.52026` | XGB public-output microblend hurt. |

### 2026-05-06

| File | Public LB | Readout |
|---|---:|---|
| `bestv1_tabiclv2_est1_stockid_rs42_w010.csv` | `0.52025` | Stock-id feature hurt. |
| `bestv1_tabiclv2_est1_plateau_w340_350_360_380_avg.csv` | `0.52027` | Plateau averaging tied old best. |
| `bestv1_tabiclv2_est1_plateau_w340_350_360_380_median.csv` | `0.52027` | Plateau median tied old best. |
| `bestv1_tabiclv2_est1_rs123_w030.csv` | `0.52028` | New seed at 3% improved by `0.00001`. |
| `bestv1_tabiclv2_est1_rs123_w050.csv` | `0.52028` | 5% tied the new best. |

### 2026-05-07

| File | Public LB | Readout |
|---|---:|---|
| `bestv1_tabiclv2_est1_rs123_w080.csv` | `0.52028` | 8% tied current best. |
| `bestv1_tabiclv2_est1_rs123_w100.csv` | `0.52028` | 10% tied current best. |
| `bestv1_tabiclv2_est1_rs123_w120.csv` | `0.52028` | 12% tied current best. |
| `best20260506_rs123w050_emanuel_lgbm_w0025.csv` | `0.52027` | Emanuel LGBM 0.25% microblend hurt. |
| `bestv1_tabiclv2_est1_rs123_w150.csv` | `0.52027` | 15% rs123 weight hurt. |

### 2026-05-12

| File | Public LB | Readout |
|---|---:|---|
| `best20260512_rs123w050_ankush_xgb_w0025.csv` | `0.52028` | New public XGB signal at 0.25% tied the anchor. |
| `best20260512_rs123w050_ankush_xgb_w0050.csv` | `0.52027` | 0.5% XGB weight hurt. |
| `best20260512_rs123w050_tabicl_rs314159_w030.csv` | `0.52028` | Fresh TabICL seed at 3% tied. |
| `best20260512_rs123w050_tabicl_rs314159_w050.csv` | `0.52028` | 5% tied. |
| `best20260512_rs123w050_tabicl_rs314159_w080.csv` | `0.52029` | 8% produced a new best. |

## Current Rules

- Keep `0.52029` as the public anchor.
- Do not continue rs123 expansion above `12%`.
- Do not continue stock-id TabICL blends without new evidence.
- Do not continue direct higher-estimator replacement blends without new evidence.
- Avoid broad public-output microblends; tested variants mostly tied or hurt.
- The next useful direction is probing the fresh `314159` seed around `10-12%`, combining useful seeds at small total weight, or generating another genuinely different TabICL seed/configuration.
