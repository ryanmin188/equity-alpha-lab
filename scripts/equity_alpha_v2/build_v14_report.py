
"""
EQUITY ALPHA LAB 2.0
V1.4 - FINAL RESEARCH DASHBOARD & REPORT

One-click reporting on existing research outputs.

Reads:
  V1   monthly portfolio and run metadata
  V1.1 accounting audit
  V1.2 factor research
  V1.3 portfolio risk research

Writes:
  outputs/v1_csi300/v14_publication/
    research_dashboard.html
    RESEARCH_REPORT.md
    README_GITHUB.md
    integrity_manifest.json

No Qlib access.
No model fitting.
No strategy re-execution.
No additional pip dependencies.
"""

from __future__ import annotations

import os

for variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
):
    os.environ[variable] = "1"

from pathlib import Path
from datetime import datetime, timezone
from html import escape
import base64
import hashlib
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "outputs" / "v1_csi300"

V11 = SOURCE / "v11_audit"
V12 = SOURCE / "v12_factor_research"
V13 = SOURCE / "v13_risk_research"

OUT = SOURCE / "v14_publication"

EXPECTED_MODELS = {
    "EqualFactor",
    "Ridge",
    "LightGBM",
}

EXPECTED_STRATEGIES = {
    "Top30",
    "Buffer30_60",
}

PERIODS_PER_YEAR = 12


# ============================================================
# UTILITIES
# ============================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_csv(path, required):
    require(
        path.is_file(),
        f"Missing required input: {path}",
    )

    df = pd.read_csv(path)

    missing = set(required) - set(df.columns)

    require(
        not missing,
        f"{path.name}: missing columns {sorted(missing)}",
    )

    return df


def read_json(path):
    require(
        path.is_file(),
        f"Missing required input: {path}",
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def finite_number(value):
    try:
        x = float(value)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def fmt(value, style="decimal"):
    x = finite_number(value)

    if x is None:
        return "N/A"

    if style == "percent":
        return f"{x * 100:.2f}%"

    if style == "number":
        return f"{x:.3f}"

    if style == "pvalue":
        return f"{x:.4f}"

    return f"{x:.4f}"


def markdown_table(df, columns, labels=None, formats=None):
    labels = labels or columns
    formats = formats or {}

    header = "| " + " | ".join(labels) + " |"
    divider = "| " + " | ".join(
        ["---"] * len(columns)
    ) + " |"

    lines = [header, divider]

    for _, row in df.iterrows():
        cells = []

        for col in columns:
            value = row[col]

            if col in formats:
                cells.append(
                    fmt(value, formats[col])
                )
            else:
                cells.append(str(value))

        lines.append(
            "| " + " | ".join(cells) + " |"
        )

    return "\n".join(lines)


def html_table(df, columns, labels=None, formats=None):
    labels = labels or columns
    formats = formats or {}

    parts = [
        "<div class='table-wrap'><table><thead><tr>"
    ]

    for label in labels:
        parts.append(
            f"<th>{escape(str(label))}</th>"
        )

    parts.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        parts.append("<tr>")

        for col in columns:
            value = row[col]

            if col in formats:
                rendered = fmt(
                    value, formats[col]
                )
            else:
                rendered = (
                    "N/A"
                    if pd.isna(value)
                    else str(value)
                )

            parts.append(
                f"<td>{escape(rendered)}</td>"
            )

        parts.append("</tr>")

    parts.append("</tbody></table></div>")

    return "".join(parts)


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)

            if not chunk:
                break

            h.update(chunk)

    return h.hexdigest()


def image_data(path):
    if not path.is_file():
        return ""

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return "data:image/png;base64," + encoded


def safe_json(obj):
    return json.dumps(
        obj,
        ensure_ascii=False,
        allow_nan=False,
    ).replace("<", "\\u003c")


def derive_key(df):
    return (
        df["model"].astype(str)
        + " | "
        + df["strategy"].astype(str)
    )


# ============================================================
# INPUT COLLECTION
# ============================================================

def load_research():
    print("\n[1/5] Loading research results")

    inputs = {
        "v1_metadata": SOURCE / "run_metadata.json",
        "v11_audit": V11 / "audit_summary.json",
        "v12_factor_ic": V12 / "factor_ic_summary.csv",
        "v12_spreads": V12 / "factor_spread_summary.csv",
        "v12_model_ic": V12 / "model_ic_summary.csv",
        "v12_model_tests": V12 / "model_paired_tests.csv",
        "v12_redundancy": V12 / "factor_redundancy_pairs.csv",
        "v13_risk": V13 / "risk_attribution.csv",
        "v13_monthly": V13 / "monthly_risk_panel.csv",
        "v13_costs": V13 / "cost_sensitivity.csv",
        "v13_execution": V13 / "execution_risk_summary.csv",
        "v13_yearly": V13 / "annual_performance.csv",
        "v13_buffer_tests": V13 / "strategy_paired_tests.csv",
        "v13_model_tests": V13 / "model_net_paired_tests.csv",
    }

    for path in inputs.values():
        require(
            path.is_file(),
            f"Missing: {path}",
        )

    metadata = read_json(inputs["v1_metadata"])
    audit = read_json(inputs["v11_audit"])

    require(
        audit.get("status")
        == "PASS_INTERNAL_IDENTITIES_ONLY",
        "Accounting audit was not confirmed as passed.",
    )

    factor_ic = read_csv(
        inputs["v12_factor_ic"],
        ["period", "factor", "mean_ic", "hac_p"],
    )

    spreads = read_csv(
        inputs["v12_spreads"],
        [
            "period", "factor",
            "mean_top_minus_bottom", "hac_p",
        ],
    )

    model_ic = read_csv(
        inputs["v12_model_ic"],
        [
            "model", "mean_ic", "icir",
            "n_months", "hac_p",
        ],
    )

    model_tests = read_csv(
        inputs["v12_model_tests"],
        [
            "comparison",
            "mean_delta_rank_ic",
            "hac_p",
        ],
    )

    redundancy = read_csv(
        inputs["v12_redundancy"],
        [
            "factor_a", "factor_b",
            "mean_rank_correlation",
        ],
    )

    risk = read_csv(
        inputs["v13_risk"],
        [
            "model", "strategy",
            "total_return",
            "annualized_return",
            "beta",
            "alpha_annualized_linear",
            "alpha_hac_p",
            "max_drawdown",
            "information_ratio",
            "mean_monthly_excess",
        ],
    )

    monthly = read_csv(
        inputs["v13_monthly"],
        [
            "model", "strategy",
            "exit_date", "net", "benchmark",
            "wealth", "benchmark_wealth",
            "underwater",
        ],
    )

    costs = read_csv(
        inputs["v13_costs"],
        [
            "model", "strategy",
            "single_side_bps",
            "approx_total_return",
        ],
    )

    execution = read_csv(
        inputs["v13_execution"],
        [
            "model", "strategy",
            "blocked_orders",
            "blocked_buys",
            "blocked_sells",
            "months_with_stale_marks",
        ],
    )

    yearly = read_csv(
        inputs["v13_yearly"],
        [
            "model", "strategy",
            "year", "months",
            "period_return",
            "benchmark_return",
        ],
    )

    buffer_tests = read_csv(
        inputs["v13_buffer_tests"],
        ["model", "mean_monthly_net_difference", "hac_p"],
    )

    net_tests = read_csv(
        inputs["v13_model_tests"],
        ["strategy", "comparison", "hac_p"],
    )

    require(
        set(risk["model"]) == EXPECTED_MODELS,
        "Unexpected models in risk summary.",
    )

    require(
        set(risk["strategy"]) == EXPECTED_STRATEGIES,
        "Unexpected portfolio strategies.",
    )

    require(
        len(risk) == 6,
        "Expected exactly six strategy summaries.",
    )

    require(
        not monthly.duplicated(
            ["model", "strategy", "exit_date"]
        ).any(),
        "Duplicate monthly portfolio observations.",
    )

    monthly["exit_date"] = pd.to_datetime(
        monthly["exit_date"],
        errors="raise",
    )

    counts = monthly.groupby(
        ["model", "strategy"]
    ).size()

    require(
        counts.nunique() == 1,
        "Different evaluation lengths among strategies.",
    )

    monthly = monthly.sort_values(
        ["model", "strategy", "exit_date"]
    )

    # Verify cached V1.3 wealth directly against returns.
    for (model, strategy), group in monthly.groupby(
        ["model", "strategy"]
    ):
        expected = (
            1 + group["net"].to_numpy(float)
        ).cumprod()

        observed = group["wealth"].to_numpy(float)

        require(
            np.allclose(
                expected,
                observed,
                atol=1e-7,
                rtol=1e-7,
            ),
            f"Monthly NAV mismatch: {model}/{strategy}",
        )

        terminal_nav = float(expected[-1])

        official = risk.loc[
            (risk["model"] == model)
            & (risk["strategy"] == strategy),
            "total_return",
        ].iloc[0]

        require(
            abs(
                terminal_nav - 1 - float(official)
            ) < 1e-6,
            f"Risk summary NAV mismatch: "
            f"{model}/{strategy}",
        )

    data = {
        "metadata": metadata,
        "audit": audit,
        "factor_ic": factor_ic,
        "spreads": spreads,
        "model_ic": model_ic,
        "model_tests": model_tests,
        "redundancy": redundancy,
        "risk": risk,
        "monthly": monthly,
        "costs": costs,
        "execution": execution,
        "yearly": yearly,
        "buffer_tests": buffer_tests,
        "net_tests": net_tests,
    }

    return data, inputs


# ============================================================
# INTERACTIVE CHART DATA
# ============================================================

def build_chart_data(monthly):
    chart = {}

    for (model, strategy), group in monthly.groupby(
        ["model", "strategy"]
    ):
        group = group.sort_values("exit_date")

        key = f"{model} | {strategy}"

        chart[key] = {
            "dates": [
                d.strftime("%Y-%m-%d")
                for d in group["exit_date"]
            ],
            "nav": [
                float(v)
                for v in group["wealth"]
            ],
            "benchmark": [
                float(v)
                for v in group["benchmark_wealth"]
            ],
            "drawdown": [
                float(v)
                for v in group["underwater"]
            ],
            "monthly_excess": [
                float(x)
                for x in (
                    group["net"] - group["benchmark"]
                )
            ],
        }

    return chart


# ============================================================
# ENGLISH REPORT
# ============================================================

def build_report(d):
    print("[2/5] Writing research report")

    risk = d["risk"].sort_values(
        ["model", "strategy"]
    )

    evaluation = d["factor_ic"].loc[
        d["factor_ic"]["period"].astype(str)
        .str.contains("EVALUATION")
    ].sort_values(
        "mean_ic",
        ascending=False,
    )

    model_ic = d["model_ic"].sort_values(
        "mean_ic",
        ascending=False,
    )

    columns_risk = [
        "model", "strategy", "total_return",
        "beta", "alpha_annualized_linear",
        "alpha_hac_p", "max_drawdown",
        "information_ratio",
    ]

    risk_formats = {
        "total_return": "percent",
        "alpha_annualized_linear": "percent",
        "alpha_hac_p": "pvalue",
        "max_drawdown": "percent",
        "beta": "number",
        "information_ratio": "number",
    }

    report = f"""# Equity Alpha Lab 2.0

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

{markdown_table(
    evaluation,
    ["factor", "mean_ic", "icir", "hac_p"],
    ["Factor", "Mean Rank IC", "ICIR", "HAC p"],
    {"mean_ic": "number", "icir": "number",
     "hac_p": "pvalue"}
)}

The earlier development period displayed stronger negative
Rank IC for volatility and range-related factors and positive
Rank IC for short-term reversal. Several relationships weakened
or changed direction in the later evaluation period.

A change in estimated sign alone does not demonstrate
a statistically significant structural break.

### 4. Model Ranking Performance

{markdown_table(
    model_ic,
    ["model", "n_months", "mean_ic", "icir", "hac_p"],
    ["Model", "Months", "Mean IC", "ICIR", "HAC p"],
    {"mean_ic": "number", "icir": "number",
     "hac_p": "pvalue"}
)}

Paired tests of monthly Rank IC differences:

{markdown_table(
    d["model_tests"],
    ["comparison", "mean_delta_rank_ic", "hac_p"],
    ["Comparison", "Mean Delta IC", "HAC p"],
    {"mean_delta_rank_ic": "number",
     "hac_p": "pvalue"}
)}

These comparisons do not provide convincing evidence of
one model consistently outperforming another in Rank IC.

### 5. Portfolio Performance and Market Risk

{markdown_table(
    risk, columns_risk,
    ["Model", "Portfolio", "Total Return",
     "Beta", "Annualized Regression Alpha",
     "Alpha HAC p", "Maximum Drawdown",
     "Information Ratio"],
    risk_formats
)}

Regression specification:

r(portfolio,t) = alpha + beta * r(CSI300,t) + error(t)

The reported annualized regression alpha is the monthly
intercept multiplied by 12. The zero-risk-free-rate market
model is descriptive, not a causal identification strategy.

The models' higher absolute returns should not be equated
with a stable benchmark-relative investment alpha.

### 6. Factor Redundancy

{markdown_table(
    d["redundancy"].head(8),
    ["factor_a", "factor_b", "mean_rank_correlation"],
    ["Factor A", "Factor B", "Mean Rank Correlation"],
    {"mean_rank_correlation": "number"}
)}

Several volatility/range and momentum/trend indicators
are highly correlated, reducing the number of economically
independent information sources.

### 7. Cost Sensitivity

The following results are first-order, fixed-path sensitivity
estimates. They are NOT re-executed cash/position simulations.

{markdown_table(
    d["costs"].loc[
        d["costs"]["single_side_bps"].isin(
            [0, 5, 10, 20, 30]
        )
    ].sort_values(
        ["model", "strategy", "single_side_bps"]
    ),
    ["model", "strategy", "single_side_bps",
     "approx_total_return"],
    ["Model", "Portfolio", "Single-Side bps",
     "Approximate Total Return"],
    {"approx_total_return": "percent"}
)}

Costs particularly affect high-turnover portfolios.
Fixed-path assumptions mean results should not be labeled
as exact counterfactual account performance.

### 8. Execution and Accounting Controls

- V1.1 status: {d["audit"].get("status")}
- Internal account inconsistencies reported: zero.
- External execution realism has NOT been verified.
- Missing open prices caused blocked transactions.
- Stale historical marks were used for valuation.
- Qlib adjusted prices require separate verification for
  cash/share consistency and corporate action treatment.

{markdown_table(
    d["execution"],
    ["model", "strategy",
     "blocked_orders", "blocked_buys",
     "blocked_sells", "months_with_stale_marks"],
    ["Model", "Portfolio", "Blocked Orders",
     "Blocked Buys", "Blocked Sells",
     "Stale-Mark Months"]
)}

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
"""

    path = OUT / "RESEARCH_REPORT.md"

    path.write_text(
        report,
        encoding="utf-8",
    )

    return path


# ============================================================
# README
# ============================================================

def build_readme(d):
    print("[3/5] Writing GitHub README")

    risk = d["risk"].sort_values(
        ["model", "strategy"]
    )

    display = markdown_table(
        risk,
        [
            "model", "strategy",
            "total_return", "beta",
            "max_drawdown",
        ],
        [
            "Model", "Portfolio",
            "Cumulative Return",
            "Market Beta", "Max DD",
        ],
        {
            "total_return": "percent",
            "beta": "number",
            "max_drawdown": "percent",
        },
    )

    readme = f"""# Equity Alpha Lab 2.0

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

{display}

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
"""

    path = OUT / "README_GITHUB.md"

    path.write_text(
        readme,
        encoding="utf-8",
    )

    return path


# ============================================================
# HTML DASHBOARD
# ============================================================

HTML_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport"
      content="width=device-width, initial-scale=1">
<title>Equity Alpha Lab 2.0 | Research Dashboard</title>

<style>
:root {
  color-scheme: light;
  --bg: #f4f6fa;
  --paper: #ffffff;
  --ink: #172235;
  --muted: #66758c;
  --line: #e1e7f0;
  --blue: #2563eb;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  background: var(--bg);
  color: var(--ink);
  font-family:
    Inter, "Segoe UI", Arial, sans-serif;
  line-height: 1.55;
}

header {
  background: #15243b;
  color: white;
  padding: 44px 24px;
}

header .inner, main {
  width: min(1200px, calc(100% - 36px));
  margin: auto;
}

header h1 {
  font-size: clamp(26px, 4vw, 42px);
  margin: 0 0 6px;
}

header p {
  margin: 0;
  color: #cad6e7;
}

main { padding: 24px 0 50px; }

.section-title {
  margin: 32px 0 14px;
  font-size: 22px;
}

.cards {
  display: grid;
  grid-template-columns:
    repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}

.card, .panel {
  background: var(--paper);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 19px;
  box-shadow: 0 3px 12px rgba(10,30,60,.035);
}

.kpi-title {
  color: var(--muted);
  font-size: 12px;
}

.kpi-value {
  font-weight: 720;
  font-size: 27px;
  margin-top: 5px;
}

.note {
  color: var(--muted);
  font-size: 13px;
  margin-top: 10px;
}

.warning {
  border-left: 4px solid #c28330;
  padding: 12px 16px;
  background: #fff9ed;
  border-radius: 6px;
  margin: 18px 0;
}

.table-wrap {
  overflow-x: auto;
  max-width: 100%;
}

table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

th, td {
  text-align: left;
  padding: 10px 11px;
  border-bottom: 1px solid var(--line);
  white-space: nowrap;
}

th {
  color: #586b83;
  font-size: 12px;
  background: #f7f9fc;
}

tr:hover td { background: #f8fbff; }

.chart-controls {
  display: flex;
  gap: 12px;
  flex-wrap: wrap;
  align-items: center;
  margin: 12px 0;
}

select {
  padding: 9px 12px;
  border: 1px solid #bfcadb;
  border-radius: 7px;
  background: white;
  font-size: 14px;
}

.plot {
  width: 100%;
  background: #fff;
}

.chart-title {
  font-weight: 650;
  margin: 12px 0 0;
}

.fig-grid {
  display: grid;
  grid-template-columns:
    repeat(auto-fit, minmax(330px, 1fr));
  gap: 15px;
}

figure {
  margin: 0;
  padding: 8px;
  background: white;
  border: 1px solid var(--line);
  border-radius: 9px;
}

figure img {
  width: 100%;
  height: auto;
}

figcaption {
  font-size: 12px;
  color: var(--muted);
  padding: 4px 8px 9px;
}

footer {
  margin-top: 30px;
  color: var(--muted);
  font-size: 12px;
  text-align: center;
}

svg text {
  font-family: "Segoe UI", Arial, sans-serif;
  fill: #62728b;
  font-size: 11px;
}
</style>
</head>

<body>
<header>
  <div class="inner">
    <h1>Equity Alpha Lab 2.0</h1>
    <p>From Predictive Signals to
       Investable Performance</p>
    <p style="margin-top:12px">
      CSI300 · Multi-Factor · Ridge · LightGBM ·
      Cost-Aware Portfolio Research
    </p>
  </div>
</header>

<main>

<div class="warning">
  <strong>Research status:</strong>
  Exploratory historical simulation, not verified live alpha.
  The evaluation period has been examined previously.
  Accounting reconciliation is internal only.
  Adjusted prices and market execution assumptions
  require additional validation.
</div>

<section>
  <h2 class="section-title">Research Overview</h2>
  <div class="cards">
    __KPI_CARDS__
  </div>
</section>

<section>
  <h2 class="section-title">Portfolio Explorer</h2>

  <div class="panel">
    <div class="chart-controls">
      <label for="portfolio">
        <strong>Portfolio:</strong>
      </label>
      <select id="portfolio"
              onchange="updateCharts()"></select>
      <span class="note">
        Hover on chart points for exact values.
      </span>
    </div>

    <p class="chart-title">
      Cumulative Wealth vs CSI300
    </p>
    <div id="navchart" class="plot"></div>

    <p class="chart-title">
      Portfolio Drawdown
    </p>
    <div id="ddchart" class="plot"></div>
  </div>
</section>

<section>
  <h2 class="section-title">
    Portfolio Performance & Market Exposure
  </h2>
  <div class="panel">
    __RISK_TABLE__
    <p class="note">
      Regression alpha is a descriptive monthly
      market-model intercept multiplied by 12;
      it is not proof of causal stock-selection skill.
    </p>
  </div>
</section>

<section>
  <h2 class="section-title">
    Factor Predictability
  </h2>
  <div class="panel">
    __FACTOR_TABLE__
    <p class="note">
      Mean monthly cross-sectional Spearman Rank IC.
      The listed p-values are exploratory HAC estimates
      without multiple-testing adjustment.
    </p>
  </div>
</section>

<section>
  <h2 class="section-title">
    Model Prediction Comparison
  </h2>
  <div class="panel">
    __MODEL_TABLE__
  </div>
</section>

<section>
  <h2 class="section-title">
    Trading Costs & Execution Risks
  </h2>
  <div class="panel">
    <h3>Fixed-Path Cost Sensitivity</h3>
    __COST_TABLE__
    <p class="note">
      Alternative fee assumptions are approximations
      using previously observed turnover, not
      exact cash/share account re-executions.
    </p>
    <h3>Blocked Orders & Valuation</h3>
    __EXECUTION_TABLE__
  </div>
</section>

<section>
  <h2 class="section-title">
    Research Figures
  </h2>
  <div class="fig-grid">
    __FIGURES__
  </div>
</section>

<section>
  <h2 class="section-title">
    Study Limitations
  </h2>

  <div class="panel">
    <ol>
      <li>Previously explored historical evaluation period.</li>
      <li>No statistically established positive alpha.</li>
      <li>Possible correlated and regime-sensitive signals.</li>
      <li>Adjusted-price / corporate-action accounting
          requires separate validation.</li>
      <li>Suspensions, stale marks and blocked order attempts.</li>
      <li>No full exchange price-limit, lot-size or
          market-impact simulation.</li>
      <li>Limited factor-risk and industry exposure controls.</li>
      <li>Stress-test transaction costs are approximate.</li>
    </ol>
  </div>
</section>

<footer>
  Generated from cached V1–V1.3 research artifacts.
  No model training or strategy execution occurs
  when viewing this page.
</footer>

</main>

<script>
const chartData = __CHART_DATA__;

const keys = Object.keys(chartData);
const selector = document.getElementById("portfolio");

keys.forEach(key => {
  const option = document.createElement("option");
  option.value = key;
  option.textContent = key;
  selector.appendChild(option);
});

if (keys.includes("LightGBM | Top30")) {
  selector.value = "LightGBM | Top30";
}

function makeSVG(containerId, dates, series,
                 yPercent=false) {
  const target = document.getElementById(containerId);

  const W = 1020;
  const H = 325;
  const left = 65;
  const right = 25;
  const top = 24;
  const bottom = 52;
  const plotW = W - left - right;
  const plotH = H - top - bottom;

  const allValues = series.flatMap(s => s.values);

  let ymin = Math.min(...allValues);
  let ymax = Math.max(...allValues);

  if (Math.abs(ymax - ymin) < 1e-10) {
    ymax = ymin + 1;
  }

  const padding = (ymax - ymin) * 0.07;
  ymin -= padding;
  ymax += padding;

  function xAt(i) {
    return left + i / Math.max(1, dates.length - 1)
      * plotW;
  }

  function yAt(v) {
    return top + (ymax - v) / (ymax - ymin)
      * plotH;
  }

  let svg = `<svg viewBox="0 0 ${W} ${H}"
    xmlns="http://www.w3.org/2000/svg"
    role="img" aria-label="Interactive chart"
    style="width:100%;height:auto">`;

  // Horizontal grid and y labels.
  for (let i = 0; i <= 5; i++) {
    const value = ymin + (ymax - ymin) * i / 5;
    const py = yAt(value);
    const label = yPercent
      ? `${(value * 100).toFixed(0)}%`
      : value.toFixed(2);

    svg += `<line x1="${left}" y1="${py}"
      x2="${W - right}" y2="${py}"
      stroke="#e7ecf3"/>`;

    svg += `<text x="${left - 9}" y="${py + 4}"
      text-anchor="end">${label}</text>`;
  }

  // Date labels.
  for (let i = 0; i < dates.length; i += 7) {
    const px = xAt(i);

    svg += `<text x="${px}" y="${H - 22}"
      text-anchor="middle">${dates[i].slice(0,7)}</text>`;
  }

  series.forEach(s => {
    const points = s.values.map(
      (v, i) => `${xAt(i)},${yAt(v)}`
    ).join(" ");

    svg += `<polyline points="${points}"
      fill="none"
      stroke="${s.color}"
      stroke-width="2.4"
      stroke-linejoin="round"/>`;

    s.values.forEach((v, i) => {
      const text = yPercent
        ? `${(v * 100).toFixed(2)}%`
        : v.toFixed(4);

      svg += `<circle cx="${xAt(i)}"
        cy="${yAt(v)}" r="5.2"
        fill="${s.color}" fill-opacity="0.025"
        stroke="transparent">
        <title>${s.name} | ${dates[i]} | ${text}</title>
      </circle>`;
    });
  });

  let legendX = left + 10;

  series.forEach(s => {
    svg += `<line x1="${legendX}" y1="13"
      x2="${legendX + 18}" y2="13"
      stroke="${s.color}" stroke-width="3"/>`;

    svg += `<text x="${legendX + 24}"
      y="17">${s.name}</text>`;

    legendX += Math.max(170, s.name.length * 9 + 45);
  });

  svg += "</svg>";

  target.innerHTML = svg;
}

function updateCharts() {
  const selected = chartData[selector.value];

  if (!selected) return;

  makeSVG(
    "navchart",
    selected.dates,
    [
      {name: "Portfolio", values: selected.nav,
       color: "#2563eb"},
      {name: "CSI300", values: selected.benchmark,
       color: "#d97706"}
    ]
  );

  makeSVG(
    "ddchart",
    selected.dates,
    [
      {name: "Drawdown", values: selected.drawdown,
       color: "#b3263b"}
    ],
    true
  );
}

updateCharts();
</script>
</body>
</html>
"""


def create_dashboard(d):
    print("[4/5] Building offline HTML dashboard")

    risk = d["risk"].sort_values(
        ["model", "strategy"]
    )

    factor = d["factor_ic"].loc[
        d["factor_ic"]["period"]
        .astype(str).str.contains("EVALUATION")
    ].sort_values(
        "mean_ic",
        ascending=False,
    )

    model = d["model_ic"].sort_values(
        "mean_ic",
        ascending=False,
    )

    best = risk.sort_values(
        "total_return",
        ascending=False,
    ).iloc[0]

    months = int(
        d["monthly"].groupby(
            ["model", "strategy"]
        ).size().iloc[0]
    )

    total_blocked = int(
        d["execution"]["blocked_orders"].sum()
    )

    cards_data = [
        ("Monthly Periods", str(months)),
        ("Portfolios", str(len(risk))),
        ("Best Simulated Total Return",
         fmt(best["total_return"], "percent")),
        ("Accounting Audit", "Internal PASS"),
        ("Blocked Order Records", str(total_blocked)),
    ]

    cards = "".join(
        "<div class='card'>"
        f"<div class='kpi-title'>{escape(title)}</div>"
        f"<div class='kpi-value'>{escape(value)}</div>"
        "</div>"
        for title, value in cards_data
    )

    risk_table = html_table(
        risk,
        [
            "model", "strategy",
            "total_return", "beta",
            "alpha_annualized_linear",
            "alpha_hac_p", "max_drawdown",
            "information_ratio",
        ],
        [
            "Model", "Portfolio", "Return",
            "Beta", "Reg. Alpha (ann.)",
            "Alpha p", "Max DD", "IR",
        ],
        {
            "total_return": "percent",
            "beta": "number",
            "alpha_annualized_linear": "percent",
            "alpha_hac_p": "pvalue",
            "max_drawdown": "percent",
            "information_ratio": "number",
        },
    )

    factor_table = html_table(
        factor,
        ["factor", "mean_ic", "icir", "hac_p"],
        ["Factor", "Mean Rank IC", "ICIR", "HAC p"],
        {
            "mean_ic": "number",
            "icir": "number",
            "hac_p": "pvalue",
        },
    )

    model_table = html_table(
        model,
        ["model", "n_months", "mean_ic", "icir", "hac_p"],
        ["Model", "Months", "Mean IC", "ICIR", "HAC p"],
        {
            "mean_ic": "number",
            "icir": "number",
            "hac_p": "pvalue",
        },
    )

    cost_table = html_table(
        d["costs"].sort_values(
            ["model", "strategy", "single_side_bps"]
        ),
        [
            "model", "strategy", "single_side_bps",
            "approx_total_return",
        ],
        ["Model", "Portfolio", "Cost bps",
         "Approx. Cumulative Return"],
        {
            "approx_total_return": "percent",
        },
    )

    execution_table = html_table(
        d["execution"],
        [
            "model", "strategy", "blocked_orders",
            "blocked_buys", "blocked_sells",
            "months_with_stale_marks",
        ],
        [
            "Model", "Portfolio", "Blocked",
            "Buys", "Sells", "Stale-Mark Months",
        ],
    )

    figure_paths = [
        (
            V12 / "factor_ic_ranking.png",
            "Individual factor Rank IC",
        ),
        (
            V12 / "factor_decile_profiles.png",
            "Factor decile return profiles",
        ),
        (
            V12 / "factor_correlation_heatmap.png",
            "Factor redundancy",
        ),
        (
            V12 / "model_rolling_rank_ic.png",
            "Rolling model Rank IC",
        ),
        (
            V13 / "03_market_exposure.png",
            "Market beta and regression alpha",
        ),
        (
            V13 / "04_cost_sensitivity.png",
            "Fixed-path transaction cost sensitivity",
        ),
        (
            V13 / "05_execution_risk.png",
            "Blocked order records",
        ),
        (
            V13 / "06_yearly_returns.png",
            "Yearly and partial-year performance",
        ),
    ]

    figure_html = []

    for path, caption in figure_paths:
        src = image_data(path)

        if not src:
            print(
                "WARNING: missing optional image:",
                path.name,
            )
            continue

        figure_html.append(
            "<figure>"
            f"<img src='{src}' "
            f"alt='{escape(caption)}'>"
            f"<figcaption>{escape(caption)}</figcaption>"
            "</figure>"
        )

    html = HTML_TEMPLATE

    replacements = {
        "__KPI_CARDS__": cards,
        "__RISK_TABLE__": risk_table,
        "__FACTOR_TABLE__": factor_table,
        "__MODEL_TABLE__": model_table,
        "__COST_TABLE__": cost_table,
        "__EXECUTION_TABLE__": execution_table,
        "__FIGURES__": "".join(figure_html),
        "__CHART_DATA__": safe_json(
            build_chart_data(d["monthly"])
        ),
    }

    for marker, content in replacements.items():
        require(
            marker in html,
            f"HTML template missing {marker}",
        )

        html = html.replace(marker, content)

    require(
        "__CHART_DATA__" not in html,
        "Chart data was not inserted.",
    )

    output = OUT / "research_dashboard.html"

    output.write_text(
        html,
        encoding="utf-8",
    )

    return output


# ============================================================
# REPRODUCIBILITY MANIFEST
# ============================================================

def create_manifest(inputs):
    print("[5/5] Creating integrity manifest")

    paths = dict(inputs)

    for key, path in [
        ("v14_dashboard", OUT / "research_dashboard.html"),
        ("v14_report", OUT / "RESEARCH_REPORT.md"),
        ("v14_readme", OUT / "README_GITHUB.md"),
    ]:
        paths[key] = path

    manifest = {
        "project": "Equity Alpha Lab 2.0",
        "report_version": "V1.4",
        "generated_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "no_retraining": True,
        "no_rebacktesting": True,
        "evaluation_is_pristine_holdout": False,
        "accounting_scope": "Internal identities only",
        "execution_reality_verified": False,
        "files": {},
    }

    for name, path in paths.items():
        manifest["files"][name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

    output = OUT / "integrity_manifest.json"

    output.write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("EQUITY ALPHA LAB 2.0")
    print("V1.4 - RESEARCH PUBLICATION SUITE")
    print("=" * 72)

    data, inputs = load_research()

    report = build_report(data)
    readme = build_readme(data)
    dashboard = create_dashboard(data)
    manifest = create_manifest(inputs)

    print("\n" + "=" * 72)
    print("V1.4 COMPLETED")
    print("=" * 72)

    for label, path in [
        ("HTML Dashboard", dashboard),
        ("Research Report", report),
        ("GitHub README", readme),
        ("SHA256 Manifest", manifest),
    ]:
        print(f"{label}: {path}")

    print("\nNEXT:")
    print("Open research_dashboard.html in Edge/Chrome.")
    print("Review the report before publishing.")
    print(
        "README_GITHUB.md is a draft; "
        "do not overwrite an existing repository README "
        "without reviewing it."
    )


if __name__ == "__main__":
    main()
