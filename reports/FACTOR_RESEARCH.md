> 版本归档：本文描述原V1.0—V1.5及其当时局限。V1.6的新测试、执行模型和结论请看[最新研究报告](V16_RESEARCH_REPORT.md)。

> 本次复核：下文 Out-of-Sample 仅指滚动时序预测，不代表未触碰测试集。图像已在[首页](../README.md)展示；原Generated Figures清单是运行输出名称。

# Equity Alpha Lab 2.0

## V1.2 Factor Research Report

### Scope

CSI300 historical constituent stocks; monthly post-close features and forward open-to-open returns.

Panel records: 29,611
Unique signal dates: 102
Missing outcome rows: 491

## Single-factor Rank IC

Factors are evaluated using monthly cross-sectional Spearman correlation.

An ICIR here is mean monthly IC divided by the sample standard deviation of monthly IC.

| factor          |   n_months |   mean_ic |     icir |   positive_month_ratio |   hac_p |
|:----------------|-----------:|----------:|---------:|-----------------------:|--------:|
| volume_surprise |         42 |   0.02739 |  0.21397 |                0.57143 | 0.11366 |
| mom_60          |         42 |   0.02163 |  0.11738 |                0.59524 | 0.35378 |
| ma_gap_20       |         42 |   0.01095 |  0.06277 |                0.61905 | 0.61733 |
| vol_20          |         42 |  -0.00659 | -0.02879 |                0.52381 | 0.78594 |
| mom_20          |         42 |  -0.00999 | -0.05945 |                0.47619 | 0.68745 |
| reversal_5      |         42 |  -0.01223 | -0.07759 |                0.35714 | 0.5006  |
| range_20        |         42 |  -0.01382 | -0.06077 |                0.47619 | 0.61001 |
| vol_60          |         42 |  -0.01429 | -0.05941 |                0.5     | 0.6116  |

## Decile Analysis

Stocks are ranked by each factor using available forward-return observations. Portfolio returns are equally weighted diagnostic averages, not executed portfolios.

| period                | factor          |   months |   mean_top_minus_bottom |   positive_spread_fraction |    hac_t |   hac_p |   hac_se |
|:----------------------|:----------------|---------:|------------------------:|---------------------------:|---------:|--------:|---------:|
| DEVELOPMENT_2012_2016 | ma_gap_20       |       60 |                -0.01139 |                    0.4     | -1.30369 | 0.19234 |  0.00874 |
| DEVELOPMENT_2012_2016 | mom_20          |       59 |                -0.01311 |                    0.44068 | -1.23478 | 0.21691 |  0.01062 |
| DEVELOPMENT_2012_2016 | mom_60          |       57 |                -0.01428 |                    0.38596 | -1.2691  | 0.20441 |  0.01125 |
| DEVELOPMENT_2012_2016 | range_20        |       60 |                -0.00846 |                    0.41667 | -1.01315 | 0.31099 |  0.00835 |
| DEVELOPMENT_2012_2016 | reversal_5      |       60 |                 0.01384 |                    0.7     |  1.5638  | 0.11786 |  0.00885 |
| DEVELOPMENT_2012_2016 | vol_20          |       60 |                -0.00642 |                    0.43333 | -0.83516 | 0.40363 |  0.00769 |
| DEVELOPMENT_2012_2016 | vol_60          |       60 |                -0.00624 |                    0.45    | -1.12704 | 0.25972 |  0.00553 |
| DEVELOPMENT_2012_2016 | volume_surprise |       60 |                 0.00418 |                    0.58333 |  0.79277 | 0.42791 |  0.00527 |
| EVALUATION_2017_2020  | ma_gap_20       |       42 |                 0.00598 |                    0.71429 |  0.85358 | 0.39334 |  0.007   |
| EVALUATION_2017_2020  | mom_20          |       42 |                 0.00126 |                    0.54762 |  0.20323 | 0.83896 |  0.0062  |
| EVALUATION_2017_2020  | mom_60          |       42 |                 0.00501 |                    0.59524 |  0.69491 | 0.48711 |  0.00721 |
| EVALUATION_2017_2020  | range_20        |       42 |                 0.00701 |                    0.54762 |  0.69747 | 0.48551 |  0.01005 |
| EVALUATION_2017_2020  | reversal_5      |       42 |                -0.00752 |                    0.33333 | -1.49147 | 0.13584 |  0.00504 |
| EVALUATION_2017_2020  | vol_20          |       42 |                 0.00614 |                    0.57143 |  0.75808 | 0.44841 |  0.0081  |
| EVALUATION_2017_2020  | vol_60          |       42 |                 0.00793 |                    0.52381 |  0.76062 | 0.44688 |  0.01042 |
| EVALUATION_2017_2020  | volume_surprise |       42 |                 0.00521 |                    0.52381 |  1.20063 | 0.2299  |  0.00434 |

## Factor Redundancy

Correlations are computed cross-sectionally within each signal month and then averaged.

| factor_a        | factor_b        |   mean_rank_correlation |   absolute_correlation |
|:----------------|:----------------|------------------------:|-----------------------:|
| vol_20          | range_20        |                 0.89209 |                0.89209 |
| vol_60          | range_20        |                 0.8266  |                0.8266  |
| vol_20          | vol_60          |                 0.82404 |                0.82404 |
| mom_20          | ma_gap_20       |                 0.81276 |                0.81276 |
| reversal_5      | ma_gap_20       |                -0.71955 |                0.71955 |
| mom_20          | mom_60          |                 0.50248 |                0.50248 |
| mom_20          | reversal_5      |                -0.46022 |                0.46022 |
| mom_60          | ma_gap_20       |                 0.4388  |                0.4388  |
| reversal_5      | volume_surprise |                -0.32395 |                0.32395 |
| volume_surprise | ma_gap_20       |                 0.32374 |                0.32374 |
| mom_60          | range_20        |                 0.31745 |                0.31745 |
| mom_60          | reversal_5      |                -0.26586 |                0.26586 |

## Out-of-Sample Model Diagnostics

| model       |   n_months |   mean_ic |   std_ic |    icir |   positive_month_ratio |   hac_t |   hac_p |   hac_se |
|:------------|-----------:|----------:|---------:|--------:|-----------------------:|--------:|--------:|---------:|
| EqualFactor |         42 |   0.02541 |  0.22271 | 0.11409 |                0.5     | 0.84602 | 0.39754 |  0.03003 |
| LightGBM    |         42 |   0.03033 |  0.11838 | 0.25626 |                0.61905 | 1.80088 | 0.07172 |  0.01684 |
| Ridge       |         42 |   0.03672 |  0.15482 | 0.23717 |                0.66667 | 1.72603 | 0.08434 |  0.02127 |

### Paired Monthly Rank IC Differences

| comparison             |   paired_months |   mean_delta_rank_ic |    hac_t |   hac_p |   hac_se |
|:-----------------------|----------------:|---------------------:|---------:|--------:|---------:|
| Ridge - EqualFactor    |              42 |              0.01131 |  0.37358 | 0.70872 |  0.03028 |
| LightGBM - EqualFactor |              42 |              0.00493 |  0.18203 | 0.85556 |  0.02706 |
| LightGBM - Ridge       |              42 |             -0.00638 | -0.46851 | 0.63942 |  0.01363 |

## Interpretation and Limitations

1. These are exploratory, previously inspected historical periods, not a pristine holdout.

2. Future returns are used only to assess predictions after the fact.

3. Missing future outcomes are excluded from IC and decile computations, never from the historical inference universe.

4. Decile portfolios are not a realizable long-short backtest and exclude trading costs.

5. Eight factors and multiple comparisons were inspected; individual p-values are exploratory and are not multiplicity adjusted.

6. Actual multi-horizon factor decay was not calculated because independent forward 5/10/20-day labels are not present.

7. Company-level fundamental characteristics and industry risk data are not available in this research panel.

8. Qlib price adjustment, suspended holdings and market execution assumptions remain separate implementation limitations.

## Generated Figures

- factor_ic_ranking.png
- factor_decile_profiles.png
- factor_correlation_heatmap.png
- model_cumulative_rank_ic.png
- model_rolling_rank_ic.png
