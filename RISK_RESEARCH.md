# Equity Alpha Lab 2.0

## V1.3 Portfolio Risk Attribution and Robustness Report

### Research setting

Monthly CSI300 constituent-stock selection using EqualFactor, Ridge and LightGBM.

The evaluation period has been explored previously and is not a pristine holdout.

### Risk attribution

```
      model    strategy  months  total_return  benchmark_total_return  annualized_return  annualized_volatility  sharpe_rf0  max_drawdown max_dd_peak max_dd_trough  tracking_error  information_ratio  mean_monthly_excess  excess_hac_p  positive_month_fraction  worst_month  best_month  longest_loss_streak_months  avg_monthly_turnover  avg_cash_weight    beta  alpha_monthly  alpha_annualized_linear  alpha_hac_p  beta_hac_p  r_squared
EqualFactor Buffer30_60      42       0.19899                 0.39664            0.05322                0.15573     0.41102      -0.22755  2018-02-01    2019-01-02         0.08972           -0.55177             -0.00413       0.30640                  0.61905     -0.13860     0.09619                           3               0.66804          0.02381 0.72085       -0.00148                 -0.01782      0.65013         0.0    0.78595
EqualFactor       Top30      42       0.16357                 0.39664            0.04423                0.15692     0.35381      -0.22689  2018-02-01    2019-01-02         0.08553           -0.67801             -0.00483       0.20526                  0.54762     -0.12961     0.09695                           3               1.14338          0.02381 0.73595       -0.00233                 -0.02802      0.40174         0.0    0.80680
   LightGBM Buffer30_60      42       0.27329                 0.39664            0.07147                0.27325     0.37900      -0.29009  2017-10-09    2018-11-01         0.13445           -0.07400             -0.00083       0.88226                  0.47619     -0.16954     0.30162                           5               1.48244          0.02535 1.27137       -0.00340                 -0.04075      0.47231         0.0    0.79409
   LightGBM       Top30      42       0.27981                 0.39664            0.07303                0.26728     0.38730      -0.30129  2017-10-09    2018-11-01         0.12866           -0.07768             -0.00083       0.87969                  0.42857     -0.15808     0.29441                           7               1.67492          0.02535 1.24814       -0.00318                 -0.03816      0.47739         0.0    0.79989
      Ridge Buffer30_60      42       0.26849                 0.39664            0.07031                0.23038     0.40529      -0.24628  2017-09-01    2018-11-01         0.10220           -0.19708             -0.00168       0.71392                  0.50000     -0.15639     0.20573                           4               1.40146          0.02454 1.08108       -0.00245                 -0.02934      0.58519         0.0    0.80776
      Ridge       Top30      42       0.18521                 0.39664            0.04975                0.22893     0.32202      -0.27713  2017-10-09    2018-11-01         0.09502           -0.41874             -0.00332       0.43961                  0.52381     -0.15639     0.19070                           3               1.64669          0.02454 1.09131       -0.00418                 -0.05016      0.32549         0.0    0.83355
```

Monthly market-model regression:

`strategy_return = alpha + beta * CSI300_return + error`

Alpha is annualized by multiplying the monthly intercept by 12. Risk-free rate is assumed zero for this diagnostic.

Alpha and beta are associations, not proof of causal investment skill.

### Transaction cost sensitivity

Original simulated single-side cost: 5.0 bp.

```
      model    strategy  single_side_bps  approx_total_return  approx_annualized_return  approx_max_drawdown  avg_monthly_turnover  baseline_fee_fraction_mean  fixed_trade_path  exact_resimulation
EqualFactor Buffer30_60                0              0.21577                   0.05741             -0.22544               0.66804                     0.00033              True               False
EqualFactor Buffer30_60                5              0.19899                   0.05322             -0.22755               0.66804                     0.00033              True               False
EqualFactor Buffer30_60               10              0.18243                   0.04904             -0.22965               0.66804                     0.00033              True               False
EqualFactor Buffer30_60               20              0.14999                   0.04074             -0.23385               0.66804                     0.00033              True               False
EqualFactor Buffer30_60               30              0.11840                   0.03249             -0.23803               0.66804                     0.00033              True               False
EqualFactor       Top30                0              0.19168                   0.05138             -0.22261               1.14338                     0.00057              True               False
EqualFactor       Top30                5              0.16357                   0.04423             -0.22689               1.14338                     0.00057              True               False
EqualFactor       Top30               10              0.13611                   0.03713             -0.23114               1.14338                     0.00057              True               False
EqualFactor       Top30               20              0.08307                   0.02306             -0.23958               1.14338                     0.00057              True               False
EqualFactor       Top30               30              0.03245                   0.00917             -0.24793               1.14338                     0.00057              True               False
   LightGBM Buffer30_60                0              0.31329                   0.08098             -0.28310               1.48244                     0.00074              True               False
   LightGBM Buffer30_60                5              0.27329                   0.07147             -0.29009               1.48244                     0.00074              True               False
   LightGBM Buffer30_60               10              0.23448                   0.06203             -0.29702               1.48244                     0.00074              True               False
   LightGBM Buffer30_60               20              0.16030                   0.04339             -0.31068               1.48244                     0.00074              True               False
   LightGBM Buffer30_60               30              0.09047                   0.02505             -0.32410               1.48244                     0.00074              True               False
   LightGBM       Top30                0              0.32535                   0.08381             -0.29363               1.67492                     0.00084              True               False
   LightGBM       Top30                5              0.27981                   0.07303             -0.30129               1.67492                     0.00084              True               False
   LightGBM       Top30               10              0.23579                   0.06236             -0.30888               1.67492                     0.00084              True               False
   LightGBM       Top30               20              0.15215                   0.04130             -0.32382               1.67492                     0.00084              True               False
   LightGBM       Top30               30              0.07404                   0.02062             -0.33847               1.67492                     0.00084              True               False
      Ridge Buffer30_60                0              0.30619                   0.07931             -0.24010               1.40146                     0.00070              True               False
      Ridge Buffer30_60                5              0.26849                   0.07031             -0.24628               1.40146                     0.00070              True               False
      Ridge Buffer30_60               10              0.23186                   0.06139             -0.25271               1.40146                     0.00070              True               False
      Ridge Buffer30_60               20              0.16165                   0.04374             -0.26541               1.40146                     0.00070              True               False
      Ridge Buffer30_60               30              0.09536                   0.02636             -0.27791               1.40146                     0.00070              True               False
      Ridge       Top30                0              0.22677                   0.06013             -0.26988               1.64669                     0.00082              True               False
      Ridge       Top30                5              0.18521                   0.04975             -0.27713               1.64669                     0.00082              True               False
      Ridge       Top30               10              0.14503                   0.03945             -0.28434               1.64669                     0.00082              True               False
      Ridge       Top30               20              0.06862                   0.01914             -0.29855               1.64669                     0.00082              True               False
      Ridge       Top30               30             -0.00281                  -0.00080             -0.31781               1.64669                     0.00082              True               False
```

IMPORTANT: Alternative costs are evaluated using the saved monthly turnover and returns. These are approximate fixed-trade-path counterfactuals, NOT exact account resimulations.

### Execution and liquidity

```
      model    strategy  execution_events  filled_orders  blocked_orders  blocked_buys  blocked_sells  blocked_sell_fraction  unique_blocked_stocks  months_with_stale_marks  stale_holding_records  max_stale_positions_at_exit  mean_cash_weight  max_cash_weight  terminal_unliquidated
EqualFactor Buffer30_60              1682           1641              41             0             41                1.00000                      8                       21                     41                            5           0.02381              1.0                      0
EqualFactor       Top30              1973           1934              39             0             39                1.00000                      8                       18                     39                            6           0.02381              1.0                      0
   LightGBM Buffer30_60              2208           2147              61             4             57                0.93443                     16                       28                     57                            4           0.02535              1.0                      0
   LightGBM       Top30              2330           2269              61             4             57                0.93443                     17                       28                     57                            4           0.02535              1.0                      0
      Ridge Buffer30_60              2139           2099              40             1             39                0.97500                      9                       21                     39                            4           0.02454              1.0                      0
      Ridge       Top30              2296           2252              44             1             43                0.97727                     12                       21                     43                            4           0.02454              1.0                      0
```

Blocked orders are not necessarily unique suspension events. Stale marks are historical adjusted opens used solely for valuation.

### Calendar-year performance

```
      model    strategy  year  months  period_return  benchmark_return  mean_monthly_excess  positive_excess_months
EqualFactor Buffer30_60  2017      10        0.17413           0.18067             -0.00056                       4
EqualFactor Buffer30_60  2018      12       -0.12993          -0.18555              0.00536                       9
EqualFactor Buffer30_60  2019      12        0.13372           0.17629             -0.00373                       8
EqualFactor Buffer30_60  2020       8        0.03524           0.23475             -0.02340                       1
EqualFactor       Top30  2017      10        0.17383           0.18067             -0.00057                       5
EqualFactor       Top30  2018      12       -0.12700          -0.18555              0.00566                       9
EqualFactor       Top30  2019      12        0.08077           0.17629             -0.00769                       7
EqualFactor       Top30  2020       8        0.05061           0.23475             -0.02161                       2
   LightGBM Buffer30_60  2017      10        0.02851           0.18067             -0.01353                       2
   LightGBM Buffer30_60  2018      12       -0.19775          -0.18555             -0.00102                       6
   LightGBM Buffer30_60  2019      12        0.17358           0.17629              0.00007                       6
   LightGBM Buffer30_60  2020       8        0.31492           0.23475              0.01399                       6
   LightGBM       Top30  2017      10        0.04678           0.18067             -0.01180                       2
   LightGBM       Top30  2018      12       -0.21740          -0.18555             -0.00315                       5
   LightGBM       Top30  2019      12        0.16548           0.17629             -0.00062                       6
   LightGBM       Top30  2020       8        0.34043           0.23475              0.01603                       6
      Ridge Buffer30_60  2017      10        0.03819           0.18067             -0.01291                       2
      Ridge Buffer30_60  2018      12       -0.18954          -0.18555             -0.00025                       7
      Ridge Buffer30_60  2019      12        0.18425           0.17629              0.00049                       7
      Ridge Buffer30_60  2020       8        0.27302           0.23475              0.00697                       4
      Ridge       Top30  2017      10        0.04953           0.18067             -0.01181                       2
      Ridge       Top30  2018      12       -0.21919          -0.18555             -0.00332                       6
      Ridge       Top30  2019      12        0.17191           0.17629             -0.00015                       5
      Ridge       Top30  2020       8        0.23414           0.23475              0.00255                       4
```

Partial first/last years must not be interpreted as full-year returns.

### Direct vs buffered portfolios

```
      model          comparison  paired_months  mean_monthly_net_difference   hac_p   hac_t
EqualFactor Buffer30_60 - Top30             42                      0.00071 0.44059 0.77120
      Ridge Buffer30_60 - Top30             42                      0.00164 0.06879 1.81979
   LightGBM Buffer30_60 - Top30             42                      0.00000 0.99703 0.00372
```

### Model net return comparisons

```
   strategy             comparison  paired_months  mean_monthly_net_difference   hac_p
      Top30       LightGBM - Ridge             42                      0.00248 0.47302
      Top30 LightGBM - EqualFactor             42                      0.00400 0.62038
      Top30    Ridge - EqualFactor             42                      0.00152 0.81059
Buffer30_60       LightGBM - Ridge             42                      0.00085 0.80703
Buffer30_60 LightGBM - EqualFactor             42                      0.00330 0.68980
Buffer30_60    Ridge - EqualFactor             42                      0.00245 0.71272
```

### Limitations

1. Adjusted Qlib OHLC prices do not independently establish executable trade prices.

2. The simulator lacks exchange limit-up/limit-down execution, lot sizes and impact costs.

3. Stale marks may understate or overstate risk.

4. Regression alpha does not control for size, value, momentum or industry exposures.

5. Samples are relatively short, and all tests are exploratory with no correction for multiple comparisons.

6. Cost sensitivity uses fixed observed trading amounts rather than re-executing cash and stock positions at new fee levels.

7. V1.1 verified internal accounting identities but not market execution realism.

### Conclusion

Interpret risk, turnover and factor evidence together. Positive prediction IC and positive absolute return do not imply positive risk-adjusted benchmark alpha.