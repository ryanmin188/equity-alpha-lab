> 本次复核：所有组合累计收益低于同期开盘到开盘沪深300价格指数39.66%；回撤是月频观测。详见[方法审核](METHODOLOGY_REVIEW.md)。

# Equity Alpha Lab 2.0

## From Predictive Signals to Investable Performance

### Abstract

This study investigates cross-sectional stock-return prediction
among historical CSI300 constituents, comparing a pre-specified
equal-factor signal, Ridge regression and LightGBM.

The research separates predictive ranking quality from portfolio
implementation. Monthly stock-selection simulations incorporate
cash accounting, order constraints and a baseline transaction
cost assumption. Factor diagnostics, account checks and market
risk analysis are reported separately.

The evidence does not establish a statistically significant,
persistent investment alpha. Results instead highlight factor
instability, overlapping price-volume signals, market exposure
and implementation frictions.

### 1. Research Questions

1. Do economically motivated price-volume factors predict
   the relative future returns of CSI300 constituent stocks?
2. Does machine learning improve ranking ability relative
   to a simple factor aggregation?
3. Does predictive ability translate into benchmark-relative
   and market-risk-adjusted investment performance?
4. How sensitive are simulated results to turnover, costs,
   stale marks and blocked orders?

### 2. Research Design

- Historical research panel: 29,611 stock-month observations.
- Eight price-volume factors.
- Signals formed after month-end close.
- Returns evaluated over the following open-to-open interval.
- Rolling historical model training without current-period
  labels entering the training set.
- Three signal models: EqualFactor, Ridge and LightGBM.
- Two construction rules: Top30 and Buffer30/60.
- Baseline single-side transaction cost: 5 basis points.
- Six paper portfolios assessed over 42 months.

The 2017–2020 evaluation period had been examined during
development, so it is not an untouched final holdout.

### 3. Single-Factor Predictability

| Factor | Mean Rank IC | ICIR | HAC p |
| --- | --- | --- | --- |
| volume_surprise | 0.027 | 0.214 | 0.1137 |
| mom_60 | 0.022 | 0.117 | 0.3538 |
| ma_gap_20 | 0.011 | 0.063 | 0.6173 |
| vol_20 | -0.007 | -0.029 | 0.7859 |
| mom_20 | -0.010 | -0.059 | 0.6874 |
| reversal_5 | -0.012 | -0.078 | 0.5006 |
| range_20 | -0.014 | -0.061 | 0.6100 |
| vol_60 | -0.014 | -0.059 | 0.6116 |

The earlier development period displayed stronger negative
Rank IC for volatility and range-related factors and positive
Rank IC for short-term reversal. Several relationships weakened
or changed direction in the later evaluation period.

A change in estimated sign alone does not demonstrate
a statistically significant structural break.

### 4. Model Ranking Performance

| Model | Months | Mean IC | ICIR | HAC p |
| --- | --- | --- | --- | --- |
| Ridge | 42 | 0.037 | 0.237 | 0.0843 |
| LightGBM | 42 | 0.030 | 0.256 | 0.0717 |
| EqualFactor | 42 | 0.025 | 0.114 | 0.3975 |

Paired tests of monthly Rank IC differences:

| Comparison | Mean Delta IC | HAC p |
| --- | --- | --- |
| Ridge - EqualFactor | 0.011 | 0.7087 |
| LightGBM - EqualFactor | 0.005 | 0.8556 |
| LightGBM - Ridge | -0.006 | 0.6394 |

These comparisons do not provide convincing evidence of
one model consistently outperforming another in Rank IC.

### 5. Portfolio Performance and Market Risk

| Model | Portfolio | Total Return | Beta | Annualized Regression Alpha | Alpha HAC p | Maximum Drawdown | Information Ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| EqualFactor | Buffer30_60 | 19.90% | 0.721 | -1.78% | 0.6501 | -22.75% | -0.552 |
| EqualFactor | Top30 | 16.36% | 0.736 | -2.80% | 0.4017 | -22.69% | -0.678 |
| LightGBM | Buffer30_60 | 27.33% | 1.271 | -4.08% | 0.4723 | -29.01% | -0.074 |
| LightGBM | Top30 | 27.98% | 1.248 | -3.82% | 0.4774 | -30.13% | -0.078 |
| Ridge | Buffer30_60 | 26.85% | 1.081 | -2.93% | 0.5852 | -24.63% | -0.197 |
| Ridge | Top30 | 18.52% | 1.091 | -5.02% | 0.3255 | -27.71% | -0.419 |

Regression specification:

r(portfolio,t) = alpha + beta * r(CSI300,t) + error(t)

The reported annualized regression alpha is the monthly
intercept multiplied by 12. The zero-risk-free-rate market
model is descriptive, not a causal identification strategy.

The models' higher absolute returns should not be equated
with a stable benchmark-relative investment alpha.

### 6. Factor Redundancy

| Factor A | Factor B | Mean Rank Correlation |
| --- | --- | --- |
| vol_20 | range_20 | 0.892 |
| vol_60 | range_20 | 0.827 |
| vol_20 | vol_60 | 0.824 |
| mom_20 | ma_gap_20 | 0.813 |
| reversal_5 | ma_gap_20 | -0.720 |
| mom_20 | mom_60 | 0.502 |
| mom_20 | reversal_5 | -0.460 |
| mom_60 | ma_gap_20 | 0.439 |

Several volatility/range and momentum/trend indicators
are highly correlated, reducing the number of economically
independent information sources.

### 7. Cost Sensitivity

The following results are first-order, fixed-path sensitivity
estimates. They are NOT re-executed cash/position simulations.

| Model | Portfolio | Single-Side bps | Approximate Total Return |
| --- | --- | --- | --- |
| EqualFactor | Buffer30_60 | 0 | 21.58% |
| EqualFactor | Buffer30_60 | 5 | 19.90% |
| EqualFactor | Buffer30_60 | 10 | 18.24% |
| EqualFactor | Buffer30_60 | 20 | 15.00% |
| EqualFactor | Buffer30_60 | 30 | 11.84% |
| EqualFactor | Top30 | 0 | 19.17% |
| EqualFactor | Top30 | 5 | 16.36% |
| EqualFactor | Top30 | 10 | 13.61% |
| EqualFactor | Top30 | 20 | 8.31% |
| EqualFactor | Top30 | 30 | 3.24% |
| LightGBM | Buffer30_60 | 0 | 31.33% |
| LightGBM | Buffer30_60 | 5 | 27.33% |
| LightGBM | Buffer30_60 | 10 | 23.45% |
| LightGBM | Buffer30_60 | 20 | 16.03% |
| LightGBM | Buffer30_60 | 30 | 9.05% |
| LightGBM | Top30 | 0 | 32.54% |
| LightGBM | Top30 | 5 | 27.98% |
| LightGBM | Top30 | 10 | 23.58% |
| LightGBM | Top30 | 20 | 15.22% |
| LightGBM | Top30 | 30 | 7.40% |
| Ridge | Buffer30_60 | 0 | 30.62% |
| Ridge | Buffer30_60 | 5 | 26.85% |
| Ridge | Buffer30_60 | 10 | 23.19% |
| Ridge | Buffer30_60 | 20 | 16.17% |
| Ridge | Buffer30_60 | 30 | 9.54% |
| Ridge | Top30 | 0 | 22.68% |
| Ridge | Top30 | 5 | 18.52% |
| Ridge | Top30 | 10 | 14.50% |
| Ridge | Top30 | 20 | 6.86% |
| Ridge | Top30 | 30 | -0.28% |

Costs particularly affect high-turnover portfolios.
Fixed-path assumptions mean results should not be labeled
as exact counterfactual account performance.

### 8. Execution and Accounting Controls

- V1.1 status: PASS_INTERNAL_IDENTITIES_ONLY
- Internal account inconsistencies reported: zero.
- External execution realism has NOT been verified.
- Missing open prices caused blocked transactions.
- Stale historical marks were used for valuation.
- Qlib adjusted prices require separate verification for
  cash/share consistency and corporate action treatment.

| Model | Portfolio | Blocked Orders | Blocked Buys | Blocked Sells | Stale-Mark Months |
| --- | --- | --- | --- | --- | --- |
| EqualFactor | Buffer30_60 | 41 | 0 | 41 | 21 |
| EqualFactor | Top30 | 39 | 0 | 39 | 18 |
| LightGBM | Buffer30_60 | 61 | 4 | 57 | 28 |
| LightGBM | Top30 | 61 | 4 | 57 | 28 |
| Ridge | Buffer30_60 | 40 | 1 | 39 | 21 |
| Ridge | Top30 | 44 | 1 | 43 | 21 |

Blocked transaction records may include repeated attempts
on the same instrument and are not unique suspension events.

### 9. Robustness

The analysis compares calendar-period performance,
paired model net returns, buffered portfolio construction,
cost scenarios and market risk loadings.

Statistical inference remains exploratory, and the
42-month period is too short to establish stability.

### 10. Limitations

1. The evaluation period is not pristine out-of-sample.
2. The study uses price-volume factors, not a full
   fundamental equity factor library.
3. Market-beta adjustment does not control for industry,
   size, value and other systematic exposures.
4. Individual hypothesis tests are exploratory and
   not adjusted for multiple comparisons.
5. Some historical future returns are unavailable;
   prediction eligibility and outcome evaluation differ.
6. Execution lacks exchange price-limit, lot-size and
   market-impact modeling.
7. Suspended or unquoted holdings may require stale marks.
8. Adjusted price series do not independently prove
   tradable-price and dividend accounting correctness.
9. Cost scenario estimates do not replay the trading ledger.
10. No live or paper-broker deployment was performed.

### 11. Conclusion

The project demonstrates a reproducible historical research
pipeline, not an independently verified profitable strategy.

Its central finding is that predictive rankings, portfolio
construction choices, market exposures and trading frictions
must be assessed jointly when evaluating investable alpha.

---
Generated from cached V1–V1.3 research outputs.
