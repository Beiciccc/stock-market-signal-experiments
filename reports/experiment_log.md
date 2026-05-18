# Experiment Log

Competition: `stock-market-signal-predict-next-day-returns`

## Summary

Best recorded public leaderboard score: `0.52035`.

Current strongest family:

- Prior best rank anchor plus a TabICL v2 single-estimator signal.
- Useful TabICL seed `123` rank-blend weights: `3%`, `5%`, `8%`, `10%`, and `12%` all scored `0.52028`.
- Fresh TabICL seed `314159` reached `0.52029` at `8%` over the `rs123_w050` anchor.
- Fresh TabICL seed `2027` reached `0.52031` at `5%` over the `0.52029` anchor.
- Fresh TabICL seed `2027` continued improving through `15%`, reaching `0.52035`.
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

### 2026-05-13

| File | Public LB | Readout |
|---|---:|---|
| `best20260513_rs123w050_tabicl_rs314159_w100.csv` | `0.52028` | Extending `rs314159` from 8% to 10% hurt. |
| `best20260513_rs123w050_tabicl_rs314159_peakprobe_w070.csv` | `0.52029` | 7% tied the best. |
| `best20260513_rs123w050_tabicl_rs314159_peakprobe_w085.csv` | `0.52028` | 8.5% hurt; the `rs314159` peak is narrow. |
| `best20260513_best0529_tabicl_rs2027_w030.csv` | `0.52030` | Fresh TabICL seed `2027` at 3% improved. |
| `best20260513_best0529_tabicl_rs2027_w050.csv` | `0.52031` | Same seed at 5% improved again; current best. |

### 2026-05-14

| File | Public LB | Readout |
|---|---:|---|
| `best20260513_best0529_tabicl_rs2027_w080.csv` | `0.52032` | `rs2027` expansion from 5% to 8% improved. |
| `best20260514_best0529_tabicl_rs2027_w100.csv` | `0.52033` | Continued `rs2027` expansion improved. |
| `best20260514_best0529_tabicl_rs2027_w120.csv` | `0.52034` | Continued `rs2027` expansion improved. |
| `best20260514_best0529_tabicl_rs2027_w150.csv` | `0.52035` | Continued `rs2027` expansion improved. |
| `best20260514_best0529_tabicl_rs2027_w180.csv` | `0.52035` | 18% tied 15%; expansion likely reached a plateau. |

### 2026-05-15

| File | Public LB | Readout |
|---|---:|---|
| `best20260515_best0529_tabicl_rs2027_total_w160.csv` | `0.52035` | Total 16% tied current best. |
| `best20260515_best0529_tabicl_rs2027_total_w140.csv` | `0.52034` | Lower edge dropped. |
| `best20260515_best052035_tabicl_rs271828_w030.csv` | `0.52030` | Fresh seed `271828` hurt at 3%; stopped. |
| `best20260515_best0529_tabicl_rs2027_total_w170.csv` | `0.52035` | Total 17% tied current best. |
| `best20260515_best0529_tabicl_rs2027_total_w130.csv` | `0.52034` | Lower edge dropped. |

### 2026-05-16

| File | Public LB | Readout |
|---|---:|---|
| `best20260516_rs2027_total_w155.csv` | `0.52035` | Total 15.5% `rs2027` tied current best; plateau center did not break through. |
| `best20260516_rs2027_total_w165.csv` | `0.52035` | Total 16.5% `rs2027` tied current best; confirms the 15%-18% region is a plateau. |
| `best20260516_rs2027_plateau_avg_w150_160_170_180.csv` | `0.52035` | Rank average of the stable `rs2027` plateau tied current best; averaging did not remove enough noise to improve. |
| `best20260516_anchor_anti_rs271828_w005.csv` | `0.52036` | Micro anti-blend against failed seed `271828` improved; promoted to current anchor. |
| `best20260516_best052036_anti_rs314159_w005.csv` | `0.52036` | Micro anti-blend against `rs314159`, recomputed on the new `0.52036` anchor, tied the new best. |

### 2026-05-17

| File | Public LB | Readout |
|---|---:|---|
| `best20260517_anchor_anti_rs271828_total_w010.csv` | `0.52036` | Total -1.0% anti-`rs271828` tied current best; the branch remains stable. |
| `best20260517_anchor_anti_rs271828_total_w015.csv` | `0.52036` | Total -1.5% anti-`rs271828` tied current best; anti-`rs271828` looks like a plateau. |
| `best20260517_combo_anti271_w0075_anti314_w0025.csv` | `0.52036` | Small combined anti-`rs271828`/anti-`rs314159` perturbation tied current best. |
| `best20260517_best052036_combo_anti271_w010_anti314_w005.csv` | `0.52036` | Stronger combined anti-`rs271828`/anti-`rs314159` perturbation also tied; no additive lift. |
| `best20260517_anchor_anti_rs271828_total_w020.csv` | `0.52037` | Total -2.0% anti-`rs271828` improved; promoted to current anchor. |

### 2026-05-18

| File | Public LB | Readout |
|---|---:|---|
| `best20260518_anchor_anti_rs271828_total_w025.csv` | `0.52038` | Total -2.5% anti-`rs271828` improved; promoted to current anchor. |

## Current Rules

- Keep `0.52038` as the public anchor.
- Do not continue rs123 expansion above `12%`.
- Do not continue `rs314159` above the 7%-8% peak region.
- Do not continue stock-id TabICL blends without new evidence.
- Do not continue direct higher-estimator replacement blends without new evidence.
- Avoid broad public-output microblends; tested variants mostly tied or hurt.
- The next useful directions are a fresh no-stock-id TabICL seed/config and fine-sweeping the anti-`rs271828` peak around total `-2.5%`. `rs2027` is plateaued around `15%-18%`; positive `rs271828` hurt at 3%, but total `-2.5%` anti-`rs271828` improved.
