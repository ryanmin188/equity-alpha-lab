# Equity Alpha Lab 2.0

### From Predictive Signals to Investable Performance

A reproducible, Python-based empirical study of
CSI300 constituent-stock selection, multi-factor
prediction, portfolio implementation and risk.

**Status:** Historical exploratory research.
Not a live trading system or investment recommendation.

## Research Highlights

- Historical CSI300 constituent universe via Microsoft Qlib.
- Eight interpretable price-volume characteristics.
- EqualFactor, Ridge and LightGBM signal comparison.
- Rolling monthly model training and evaluation.
- Top30 and turnover-buffered portfolio construction.
- Cash/position ledger and blocked-order diagnostics.
- Cross-sectional Rank IC and decile return diagnostics.
- HAC tests for average IC and paired differences.
- Market beta, regression alpha and drawdown analysis.
- Approximate fixed-path transaction cost stress tests.
- Accounting integrity and methodology limitations.

## Selected Historical Results

| Model | Portfolio | Cumulative Return | Market Beta | Max DD |
| --- | --- | --- | --- | --- |
| EqualFactor | Buffer30_60 | 19.90% | 0.721 | -22.75% |
| EqualFactor | Top30 | 16.36% | 0.736 | -22.69% |
| LightGBM | Buffer30_60 | 27.33% | 1.271 | -29.01% |
| LightGBM | Top30 | 27.98% | 1.248 | -30.13% |
| Ridge | Buffer30_60 | 26.85% | 1.081 | -24.63% |
| Ridge | Top30 | 18.52% | 1.091 | -27.71% |

The observed 2017–2020 period has previously been
used in development and is not a pristine final holdout.

The absolute portfolio returns do not establish
statistically significant market-adjusted alpha.

## Research Workflow

1. Historical universe, calendar and price data
2. Point-in-time signal construction
3. Eight price-volume features
4. Rolling model training and predictions
5. Monthly stock selection and account simulation
6. Internal accounting and execution diagnostics
7. Factor diagnostics and predictive tests
8. Portfolio risk and robustness analysis
9. Publication dashboard and research report

## Python Environment

Main packages used across the project:

- Python 3.12
- qlib
- pandas / numpy
- scipy / statsmodels
- scikit-learn
- lightgbm
- matplotlib

Use the original research scripts to reproduce the
pipeline. Qlib CN historical data must be available
locally; this repository does not bundle market data.

## Reproduction Order

From `scripts/equity_alpha_v2/`:

```text
1. run_research.py
2. audit_v11_accounting.py
3. research_v12_factors.py
4. research_v13_risk.py
5. build_v14_report.py
```

Each script can be executed through PyCharm Run using
the configured project Python environment.

## Output Artifacts

See:

`outputs/v1_csi300/v14_publication/`

- `research_dashboard.html`
- `RESEARCH_REPORT.md`
- `README_GITHUB.md`
- `integrity_manifest.json`

The dashboard is self-contained and opens in a browser.

## Methodology and Caveats

- Prediction metrics are not investment returns.
- Historical long/short decile spreads are diagnostics,
  not executable profit estimates.
- Missing future returns do not automatically exclude
  securities from the live-time prediction universe.
- Accounting identities were verified internally,
  but exchange-level executability remains unverified.
- Adjusted prices, corporate actions and stale marks
  require further reconciliation.
- The evaluation period is previously explored and
  no statistically established positive alpha is claimed.

## Research Question

**When does predictive skill translate into
investable portfolio performance?**

The project compares ranking ability with portfolio
returns while accounting for market exposure,
turnover, transaction costs and execution frictions.

## License and Data

Check the licensing conditions of upstream tools
and any externally reused code before redistributing.
No proprietary price dataset is included.

## Disclaimer

For academic research and demonstration only.
Not financial advice.
