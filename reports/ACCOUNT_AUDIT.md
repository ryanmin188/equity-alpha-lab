> 本次复核：已重跑通过；仍仅是内部恒等式检查，不是交易所成交或真实现金股数核验。

# Equity Alpha Lab V1.1 — Account and Execution Audit

Status: **PASS_INTERNAL_IDENTITIES_ONLY**
Internal failures: 0; warnings: 0
Strategy configurations: 6; account-months: 252

## Summary
```
      model    strategy  months first_entry final_exit  final_nav  cumulative_net_return  mean_net_return  mean_benchmark_return  mean_monthly_excess  mean_turnover  total_blocked_orders  months_with_stale_marks  terminal_unliquidated
EqualFactor Buffer30_60      42  2017-02-03 2020-08-03   1.198987               0.198987         0.005334               0.009459            -0.004125       0.668042                    41                       21                      0
EqualFactor       Top30      42  2017-02-03 2020-08-03   1.163573               0.163573         0.004627               0.009459            -0.004832       1.143383                    39                       18                      0
   LightGBM Buffer30_60      42  2017-02-03 2020-08-03   1.273290               0.273290         0.008630               0.009459            -0.000829       1.482436                    61                       28                      0
   LightGBM       Top30      42  2017-02-03 2020-08-03   1.279808               0.279808         0.008626               0.009459            -0.000833       1.674922                    61                       28                      0
      Ridge Buffer30_60      42  2017-02-03 2020-08-03   1.268492               0.268492         0.007781               0.009459            -0.001678       1.401463                    40                       21                      0
      Ridge       Top30      42  2017-02-03 2020-08-03   1.185212               0.185212         0.006143               0.009459            -0.003316       1.646693                    44                       21                      0
```

## Key limitations
- Trade execution uses same-day adjusted open in simulator, not an independent auction fill feed.
- Missing opens are modeled as unfillable; this does not test limit-up/limit-down or order-book liquidity.
- Stale marking uses previous adjusted open and may misstate NAV through halts/corporate actions.
- Qlib adjusted prices may not map directly to executable cash/share quantities through corporate actions.
- No independent per-trade cash/share ledger snapshots: this checks output identities, not complete transaction-by-transaction self-financing.
- Future-label availability and research-period reuse still limit independent inference.
- Holding values and reported cash fraction are jointly simulator-generated, so agreement alone cannot prove market realism.
- The reported gross field is holding-period NAV change after entry trading, not fully fee-free strategy return.

## Interpretation
A passed audit establishes internal consistency of existing CSV outputs, not exchange-level execution validity.
Cross-check adjusted-price execution against raw prices/corporate actions before reporting investable P&L.
