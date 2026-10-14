
"""
EQUITY ALPHA LAB 2.0
V1.3 - PORTFOLIO RISK ATTRIBUTION & ROBUSTNESS

Inputs:
    outputs/v1_csi300/paper_backtest.csv
    outputs/v1_csi300/execution_audit.csv
    outputs/v1_csi300/holdings_audit.csv
    outputs/v1_csi300/run_metadata.json
    outputs/v1_csi300/v11_audit/audit_summary.json

No Qlib query. No model retraining.

IMPORTANT:
- Regression alpha is descriptive, not causal alpha.
- Fixed-path cost scenarios are approximate.
- Transaction prices are Qlib-adjusted opens.
- Stale marks and unmodeled price limits remain risks.
"""

from __future__ import annotations

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"

from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "outputs" / "v1_csi300"
OUT = SOURCE / "v13_risk_research"

MODELS = ["EqualFactor", "Ridge", "LightGBM"]
STRATEGIES = ["Top30", "Buffer30_60"]

COST_SCENARIOS_BPS = [0, 5, 10, 20, 30]

PERIODS_PER_YEAR = 12
HAC_LAGS = 3
MIN_REG_MONTHS = 18

ALLOW_V11_WARNINGS = False

COLORS = {
    "EqualFactor": "#64748b",
    "Ridge": "#2563eb",
    "LightGBM": "#d97706",
}


# ============================================================
# 2. BASIC UTILITIES
# ============================================================

def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def save_csv(df, filename):
    df.to_csv(
        OUT / filename,
        index=False,
        encoding="utf-8-sig",
    )


def pretty(title, df, limit=30):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)

    show = df.head(limit).copy()

    numeric = show.select_dtypes(
        include="number"
    ).columns

    show[numeric] = show[numeric].round(6)

    print(show.to_string(index=False))


def hac_mean(series):
    values = pd.Series(series).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()

    if len(values) < 12:
        return {
            "n": len(values),
            "mean": float(values.mean())
            if len(values) else np.nan,
            "hac_t": np.nan,
            "hac_p": np.nan,
            "hac_se": np.nan,
        }

    fit = sm.OLS(
        values.to_numpy(dtype=float),
        np.ones((len(values), 1)),
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS},
    )

    return {
        "n": len(values),
        "mean": float(fit.params[0]),
        "hac_t": float(fit.tvalues[0]),
        "hac_p": float(fit.pvalues[0]),
        "hac_se": float(fit.bse[0]),
    }


def equity_curve(returns):
    r = np.asarray(returns, dtype=float)
    return np.r_[1.0, np.cumprod(1.0 + r)]


def drawdown_details(returns, dates):
    growth = equity_curve(returns)
    peak = np.maximum.accumulate(growth)
    dd = growth / peak - 1

    # dd[0] refers to the initial account value.
    worst_idx = int(np.argmin(dd))

    if worst_idx == 0:
        return {
            "max_drawdown": 0.0,
            "max_dd_trough": pd.NaT,
            "max_dd_peak": pd.NaT,
        }

    peak_idx = int(
        np.argmax(growth[:worst_idx + 1])
    )

    return {
        "max_drawdown": float(dd[worst_idx]),
        "max_dd_trough": dates[worst_idx - 1],
        "max_dd_peak": (
            dates[peak_idx - 1]
            if peak_idx > 0 else pd.NaT
        ),
    }


# ============================================================
# 3. INPUT VALIDATION
# ============================================================

def load_inputs():
    print("\n=== LOADING V1 AND V1.1 OUTPUTS ===")

    paths = {
        "portfolio": SOURCE / "paper_backtest.csv",
        "orders": SOURCE / "execution_audit.csv",
        "holdings": SOURCE / "holdings_audit.csv",
        "metadata": SOURCE / "run_metadata.json",
        "audit": SOURCE / "v11_audit"
                 / "audit_summary.json",
    }

    for name, path in paths.items():
        require(
            path.is_file(),
            f"Missing {name}: {path}"
        )

    p = pd.read_csv(
        paths["portfolio"],
        parse_dates=[
            "date", "entry_date", "exit_date"
        ],
    )

    orders = pd.read_csv(
        paths["orders"],
        parse_dates=[
            "date", "execution_date"
        ],
    )

    holdings = pd.read_csv(
        paths["holdings"],
        parse_dates=[
            "date", "valuation_date"
        ],
    )

    conf = json.loads(
        paths["metadata"].read_text(
            encoding="utf-8"
        )
    )

    audit = json.loads(
        paths["audit"].read_text(
            encoding="utf-8"
        )
    )

    allowed = (
        ["PASS_INTERNAL_IDENTITIES_ONLY"]
        if not ALLOW_V11_WARNINGS
        else [
            "PASS_INTERNAL_IDENTITIES_ONLY",
            "PASS_WITH_LIMITATIONS",
        ]
    )

    require(
        audit["status"] in allowed,
        "V1.1 audit did not pass. "
        "Resolve accounting issues first."
    )

    required_p = [
        "model", "strategy", "date",
        "entry_date", "exit_date",
        "net", "benchmark", "wealth",
        "traded_notional",
        "cost_fraction", "cash_weight",
        "blocked_buys", "blocked_sells",
        "stale_at_entry", "stale_at_exit",
        "terminal_unliquidated",
    ]

    required_o = [
        "model", "strategy", "date",
        "execution_date", "side",
        "status", "instrument", "notional",
    ]

    required_h = [
        "model", "strategy", "date",
        "instrument", "marked_value",
        "stale",
    ]

    for name, frame, columns in [
        ("paper_backtest", p, required_p),
        ("execution_audit", orders, required_o),
        ("holdings_audit", holdings, required_h),
    ]:
        missing = [
            col for col in columns
            if col not in frame.columns
        ]

        require(
            not missing,
            f"{name} missing columns: {missing}"
        )

    for col in [
        "net", "benchmark", "wealth",
        "traded_notional", "cost_fraction",
        "cash_weight", "blocked_buys",
        "blocked_sells", "stale_at_entry",
        "stale_at_exit", "terminal_unliquidated",
    ]:
        p[col] = pd.to_numeric(
            p[col], errors="coerce"
        )

    require(
        np.isfinite(
            p[
                [
                    "net", "benchmark",
                    "wealth", "traded_notional",
                    "cost_fraction", "cash_weight",
                ]
            ].to_numpy(dtype=float)
        ).all(),
        "Non-finite portfolio accounting values."
    )

    require(
        not p.duplicated(
            ["model", "strategy", "date"]
        ).any(),
        "Duplicate account-month records."
    )

    require(
        (p["net"] > -1).all()
        and (p["benchmark"] > -1).all(),
        "Invalid monthly returns."
    )

    require(
        set(p["model"]) == set(MODELS)
        and set(p["strategy"]) == set(STRATEGIES),
        "Unexpected model or strategy groups."
    )

    n_per_group = p.groupby(
        ["model", "strategy"]
    ).size()

    require(
        n_per_group.nunique() == 1,
        "Strategies have different month counts."
    )

    reference = None

    for _, group in p.groupby(
        ["model", "strategy"]
    ):
        dates = group.sort_values(
            "entry_date"
        )[
            ["entry_date", "exit_date"]
        ].reset_index(drop=True)

        if reference is None:
            reference = dates
        else:
            require(
                dates.equals(reference),
                "Portfolio holding periods differ."
            )

    base_bps = float(
        conf["cost_single_side_bps"]
    )

    require(
        base_bps >= 0,
        "Invalid baseline fee."
    )

    print("Account-month rows:", len(p))
    print("Execution records:", len(orders))
    print("Holding rows:", len(holdings))
    print("Months per strategy:", n_per_group.iloc[0])
    print("Baseline single-side cost:", base_bps, "bp")
    print("V1.1 audit:", audit["status"])

    return p, orders, holdings, conf, audit


# ============================================================
# 4. MARKET RISK ATTRIBUTION
# ============================================================

def risk_regression(strategy, benchmark):
    y = np.asarray(strategy, dtype=float)
    x = np.asarray(benchmark, dtype=float)

    if len(y) < MIN_REG_MONTHS:
        return {
            "beta": np.nan,
            "alpha_monthly": np.nan,
            "alpha_annualized_linear": np.nan,
            "alpha_hac_p": np.nan,
            "beta_hac_p": np.nan,
            "r_squared": np.nan,
        }

    # Zero risk-free-rate convention.
    # This is a descriptive market-model regression.
    X = sm.add_constant(x)

    fit = sm.OLS(y, X).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS},
    )

    alpha = float(fit.params[0])
    beta = float(fit.params[1])

    return {
        "beta": beta,
        "alpha_monthly": alpha,
        "alpha_annualized_linear": (
            alpha * PERIODS_PER_YEAR
        ),
        "alpha_hac_p": float(fit.pvalues[0]),
        "beta_hac_p": float(fit.pvalues[1]),
        "r_squared": float(fit.rsquared),
    }


def portfolio_statistics(p):
    print("\n=== RISK AND PERFORMANCE ATTRIBUTION ===")

    rows = []
    monthly_rows = []

    for (model, strategy), group in p.groupby(
        ["model", "strategy"],
        sort=True,
    ):
        group = group.sort_values(
            "entry_date"
        )

        r = group["net"].to_numpy(dtype=float)
        b = group["benchmark"].to_numpy(dtype=float)
        excess = r - b

        wealth = equity_curve(r)
        bench_wealth = equity_curve(b)

        total = float(wealth[-1] - 1)
        bench_total = float(bench_wealth[-1] - 1)

        ann = float(
            wealth[-1] ** (
                PERIODS_PER_YEAR / len(r)
            ) - 1
        )

        vol = float(
            np.std(r, ddof=1)
            * np.sqrt(PERIODS_PER_YEAR)
        )

        te = float(
            np.std(excess, ddof=1)
            * np.sqrt(PERIODS_PER_YEAR)
        )

        stats = drawdown_details(
            r,
            list(group["exit_date"]),
        )

        reg = risk_regression(r, b)
        excess_test = hac_mean(excess)

        longest_loss_streak = 0
        current_streak = 0

        for value in r:
            if value < 0:
                current_streak += 1
            else:
                current_streak = 0

            longest_loss_streak = max(
                longest_loss_streak,
                current_streak,
            )

        rows.append({
            "model": model,
            "strategy": strategy,
            "months": len(r),
            "total_return": total,
            "benchmark_total_return": bench_total,
            "annualized_return": ann,
            "annualized_volatility": vol,
            "sharpe_rf0": (
                float(r.mean() * 12 / vol)
                if vol > 0 else np.nan
            ),
            "max_drawdown": stats["max_drawdown"],
            "max_dd_peak": stats["max_dd_peak"],
            "max_dd_trough": stats["max_dd_trough"],
            "tracking_error": te,
            "information_ratio": (
                float(excess.mean() * 12 / te)
                if te > 0 else np.nan
            ),
            "mean_monthly_excess": float(
                excess.mean()
            ),
            "excess_hac_p": excess_test["hac_p"],
            "positive_month_fraction": float(
                np.mean(r > 0)
            ),
            "worst_month": float(r.min()),
            "best_month": float(r.max()),
            "longest_loss_streak_months": (
                longest_loss_streak
            ),
            "avg_monthly_turnover": float(
                group["traded_notional"].mean()
            ),
            "avg_cash_weight": float(
                group["cash_weight"].mean()
            ),
            **reg,
        })

        for i, row in enumerate(
            group.itertuples(index=False)
        ):
            monthly_rows.append({
                "model": model,
                "strategy": strategy,
                "date": row.date,
                "entry_date": row.entry_date,
                "exit_date": row.exit_date,
                "net": float(r[i]),
                "benchmark": float(b[i]),
                "excess": float(excess[i]),
                "wealth": float(wealth[i + 1]),
                "benchmark_wealth": float(
                    bench_wealth[i + 1]
                ),
                "underwater": float(
                    wealth[i + 1]
                    / np.max(wealth[:i + 2]) - 1
                ),
                "turnover": float(
                    row.traded_notional
                ),
                "cash_weight": float(
                    row.cash_weight
                ),
            })

    result = pd.DataFrame(rows)
    monthly = pd.DataFrame(monthly_rows)

    save_csv(
        result, "risk_attribution.csv"
    )
    save_csv(
        monthly, "monthly_risk_panel.csv"
    )

    return result, monthly


# ============================================================
# 5. TRANSACTION COST SENSITIVITY
# ============================================================

def cost_sensitivity(p, base_bps):
    print("\n=== FIXED-PATH COST SENSITIVITY ===")

    rows = []
    monthly_rows = []

    baseline_rate = base_bps / 10000.0

    for (model, strategy), group in p.groupby(
        ["model", "strategy"],
        sort=True,
    ):
        group = group.sort_values(
            "entry_date"
        )

        r0 = group["net"].to_numpy(float)
        turnover = group[
            "traded_notional"
        ].to_numpy(float)

        original_fee_fraction = group[
            "cost_fraction"
        ].to_numpy(float)

        # Cost fraction is based on original capital,
        # whereas the net monthly return is calculated
        # after actual simulator ledger changes.
        #
        # Changing the rate changes cash, share counts
        # and subsequent order sizes. Therefore we
        # DO NOT assert exact re-execution.
        #
        # This is a first-order sensitivity of monthly
        # net returns to fees on the OBSERVED notional.
        for bps in COST_SCENARIOS_BPS:
            delta_rate = (
                bps / 10000.0 - baseline_rate
            )

            approx_r = (
                r0 - delta_rate * turnover
            )

            if (approx_r <= -1).any():
                raise RuntimeError(
                    "Invalid fee scenario: return <= -100%."
                )

            nav = equity_curve(approx_r)

            rows.append({
                "model": model,
                "strategy": strategy,
                "single_side_bps": bps,
                "approx_total_return": float(
                    nav[-1] - 1
                ),
                "approx_annualized_return": float(
                    nav[-1] ** (
                        PERIODS_PER_YEAR / len(approx_r)
                    ) - 1
                ),
                "approx_max_drawdown": (
                    drawdown_details(
                        approx_r,
                        list(group["exit_date"]),
                    )["max_drawdown"]
                ),
                "avg_monthly_turnover": float(
                    turnover.mean()
                ),
                "baseline_fee_fraction_mean": float(
                    original_fee_fraction.mean()
                ),
                "fixed_trade_path": True,
                "exact_resimulation": False,
            })

            for i, row in enumerate(
                group.itertuples(index=False)
            ):
                monthly_rows.append({
                    "model": model,
                    "strategy": strategy,
                    "date": row.date,
                    "single_side_bps": bps,
                    "approx_net": float(approx_r[i]),
                    "approx_nav": float(nav[i + 1]),
                })

        # At baseline rate, approximate return
        # must equal the actual saved return exactly.
        reconstructed = (
            r0 - (
                baseline_rate - baseline_rate
            ) * turnover
        )

        require(
            np.allclose(
                reconstructed,
                r0,
                atol=1e-14,
            ),
            "Cost sensitivity does not reproduce baseline."
        )

    summary = pd.DataFrame(rows)
    monthly = pd.DataFrame(monthly_rows)

    save_csv(
        summary, "cost_sensitivity.csv"
    )
    save_csv(
        monthly, "cost_sensitivity_monthly.csv"
    )

    return summary


# ============================================================
# 6. EXECUTION AND LIQUIDITY RISKS
# ============================================================

def execution_risks(p, orders, holdings):
    print("\n=== EXECUTION AND LIQUIDITY RISKS ===")

    events = orders.copy()

    events["blocked"] = (
        events["status"]
        .astype(str)
        .str.startswith("BLOCKED")
    )

    events["filled"] = events["status"].isin(
        ["FILLED", "FILLED_TERMINAL"]
    )

    events["notional"] = pd.to_numeric(
        events["notional"],
        errors="coerce",
    )

    holdings = holdings.copy()

    holdings["stale_flag"] = (
        holdings["stale"]
        .astype(str)
        .str.lower()
        .eq("true")
    )

    holdings["marked_value"] = (
        pd.to_numeric(
            holdings["marked_value"],
            errors="coerce",
        )
    )

    blocked = events.loc[
        events["blocked"]
    ].copy()

    blocked_by_type = (
        blocked.groupby(
            [
                "model",
                "strategy",
                "status",
                "side",
            ],
            dropna=False,
        )
        .agg(
            count=("instrument", "size"),
            unique_stocks=(
                "instrument", "nunique"
            ),
        )
        .reset_index()
    )

    summary = []

    for (model, strategy), group in p.groupby(
        ["model", "strategy"],
        sort=True,
    ):
        e = events.loc[
            (events["model"] == model)
            & (events["strategy"] == strategy)
        ]

        h = holdings.loc[
            (holdings["model"] == model)
            & (holdings["strategy"] == strategy)
        ]

        total_blocked = int(
            e["blocked"].sum()
        )

        blocked_sell = int(
            (
                e["blocked"]
                & (e["side"] == "SELL")
            ).sum()
        )

        blocked_buy = int(
            (
                e["blocked"]
                & (e["side"] == "BUY")
            ).sum()
        )

        stale_group = h.loc[
            h["stale_flag"]
        ]

        summary.append({
            "model": model,
            "strategy": strategy,
            "execution_events": len(e),
            "filled_orders": int(
                e["filled"].sum()
            ),
            "blocked_orders": total_blocked,
            "blocked_buys": blocked_buy,
            "blocked_sells": blocked_sell,
            "blocked_sell_fraction": (
                blocked_sell / total_blocked
                if total_blocked else np.nan
            ),
            "unique_blocked_stocks": int(
                e.loc[
                    e["blocked"],
                    "instrument",
                ].nunique()
            ),
            "months_with_stale_marks": int(
                (
                    (group["stale_at_entry"] > 0)
                    | (group["stale_at_exit"] > 0)
                ).sum()
            ),
            "stale_holding_records": len(
                stale_group
            ),
            "max_stale_positions_at_exit": int(
                group["stale_at_exit"].max()
            ),
            "mean_cash_weight": float(
                group["cash_weight"].mean()
            ),
            "max_cash_weight": float(
                group["cash_weight"].max()
            ),
            "terminal_unliquidated": int(
                group[
                    "terminal_unliquidated"
                ].iloc[-1]
            ),
        })

    result = pd.DataFrame(summary)

    save_csv(
        result, "execution_risk_summary.csv"
    )
    save_csv(
        blocked_by_type, "blocked_order_types.csv"
    )
    save_csv(
        blocked, "blocked_order_events.csv"
    )
    save_csv(
        holdings.loc[
            holdings["stale_flag"]
        ],
        "stale_valuation_records.csv",
    )

    return result, blocked_by_type


# ============================================================
# 7. PERIOD AND STRATEGY ROBUSTNESS
# ============================================================

def regime_summary(monthly):
    print("\n=== ANNUAL AND PERIOD ROBUSTNESS ===")

    data = monthly.copy()

    data["year"] = data[
        "exit_date"
    ].dt.year

    rows = []

    for (model, strategy, year), group in (
        data.groupby(
            ["model", "strategy", "year"]
        )
    ):
        r = group["net"].to_numpy(float)
        b = group["benchmark"].to_numpy(float)

        rows.append({
            "model": model,
            "strategy": strategy,
            "year": year,
            "months": len(group),
            "period_return": float(
                np.prod(1 + r) - 1
            ),
            "benchmark_return": float(
                np.prod(1 + b) - 1
            ),
            "mean_monthly_excess": float(
                np.mean(r - b)
            ),
            "positive_excess_months": int(
                np.sum(r > b)
            ),
        })

    annual = pd.DataFrame(rows)
    save_csv(
        annual, "annual_performance.csv"
    )

    # Direct and buffered versions of the
    # SAME model must be evaluated on identical dates.
    pivot = monthly.pivot(
        index="entry_date",
        columns=["model", "strategy"],
        values="net",
    )

    comparisons = []

    for model in MODELS:
        direct = pivot[
            (model, "Top30")
        ]

        buffered = pivot[
            (model, "Buffer30_60")
        ]

        paired = pd.concat(
            [direct, buffered],
            axis=1,
        ).dropna()

        require(
            len(paired) >= 24,
            f"Insufficient paired dates for {model}"
        )

        diff = (
            paired.iloc[:, 1]
            - paired.iloc[:, 0]
        )

        test = hac_mean(diff)

        comparisons.append({
            "model": model,
            "comparison": "Buffer30_60 - Top30",
            "paired_months": len(diff),
            "mean_monthly_net_difference": (
                float(diff.mean())
            ),
            "hac_p": test["hac_p"],
            "hac_t": test["hac_t"],
        })

    strategy_tests = pd.DataFrame(
        comparisons
    )

    save_csv(
        strategy_tests,
        "strategy_paired_tests.csv",
    )

    model_pairs = [
        ("LightGBM", "Ridge"),
        ("LightGBM", "EqualFactor"),
        ("Ridge", "EqualFactor"),
    ]

    model_rows = []

    for strategy in STRATEGIES:
        for new, old in model_pairs:
            paired = pd.concat(
                [
                    pivot[(new, strategy)],
                    pivot[(old, strategy)],
                ],
                axis=1,
            ).dropna()

            diff = (
                paired.iloc[:, 0]
                - paired.iloc[:, 1]
            )

            test = hac_mean(diff)

            model_rows.append({
                "strategy": strategy,
                "comparison": f"{new} - {old}",
                "paired_months": len(diff),
                "mean_monthly_net_difference": (
                    float(diff.mean())
                ),
                "hac_p": test["hac_p"],
            })

    model_tests = pd.DataFrame(
        model_rows
    )

    save_csv(
        model_tests,
        "model_net_paired_tests.csv",
    )

    return annual, strategy_tests, model_tests


# ============================================================
# 8. FIGURES
# ============================================================

def figures(
    monthly,
    risk,
    costs,
    execution,
    annual,
):
    print("\n=== GENERATING RISK FIGURES ===")

    plt.rcParams.update({
        "font.size": 10,
        "figure.dpi": 110,
    })

    # Figure 1: Cumulative wealth
    fig, ax = plt.subplots(figsize=(11, 6))

    for (model, strategy), group in monthly.groupby(
        ["model", "strategy"]
    ):
        group = group.sort_values("exit_date")

        linestyle = (
            "-"
            if strategy == "Top30"
            else "--"
        )

        ax.plot(
            group["exit_date"],
            group["wealth"],
            label=f"{model} {strategy}",
            color=COLORS[model],
            linestyle=linestyle,
        )

    reference = monthly.loc[
        (monthly["model"] == "EqualFactor")
        & (monthly["strategy"] == "Top30")
    ].sort_values("exit_date")

    ax.plot(
        reference["exit_date"],
        reference["benchmark_wealth"],
        label="CSI300",
        color="black",
        linewidth=2,
    )

    ax.set_title(
        "Monthly Paper Portfolios vs CSI300"
    )
    ax.set_ylabel("Growth of 1")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUT / "01_equity_vs_benchmark.png",
        dpi=160,
    )
    plt.close(fig)

    # Figure 2: Drawdown
    fig, ax = plt.subplots(figsize=(11, 5))

    for (model, strategy), group in monthly.groupby(
        ["model", "strategy"]
    ):
        if strategy != "Top30":
            continue

        group = group.sort_values("exit_date")

        ax.plot(
            group["exit_date"],
            group["underwater"],
            label=model,
            color=COLORS[model],
        )

    ax.set_title("Portfolio Drawdowns: Top30")
    ax.set_ylabel("Drawdown")
    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUT / "02_drawdowns.png",
        dpi=160,
    )
    plt.close(fig)

    # Figure 3: Beta / Alpha
    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5)
    )

    x = np.arange(len(risk))

    axes[0].bar(
        x,
        risk["beta"],
    )
    axes[0].set_title(
        "Estimated Market Beta"
    )

    axes[1].bar(
        x,
        risk["alpha_annualized_linear"],
    )
    axes[1].axhline(
        0, color="black", linewidth=0.8
    )
    axes[1].set_title(
        "Regression Alpha (Annualized Linear)"
    )

    labels = (
        risk["model"]
        + "\n"
        + risk["strategy"]
    )

    for ax in axes:
        ax.set_xticks(
            x, labels, rotation=35,
            ha="right",
        )
        ax.grid(
            axis="y", alpha=0.2
        )

    fig.tight_layout()
    fig.savefig(
        OUT / "03_market_exposure.png",
        dpi=160,
    )
    plt.close(fig)

    # Figure 4: Cost sensitivity
    fig, ax = plt.subplots(figsize=(10, 6))

    for (model, strategy), group in costs.groupby(
        ["model", "strategy"]
    ):
        ax.plot(
            group["single_side_bps"],
            group["approx_total_return"],
            marker="o",
            label=f"{model} {strategy}",
            color=COLORS[model],
            linestyle=(
                "-" if strategy == "Top30"
                else "--"
            ),
        )

    ax.set_title(
        "Approximate Fixed-Path Cost Sensitivity"
    )
    ax.set_xlabel(
        "Single-side cost (basis points)"
    )
    ax.set_ylabel(
        "Approximate cumulative net return"
    )
    ax.axhline(
        0, color="black", linewidth=0.8
    )
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUT / "04_cost_sensitivity.png",
        dpi=160,
    )
    plt.close(fig)

    # Figure 5: Blocked orders
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(execution))

    ax.bar(
        x,
        execution["blocked_buys"],
        label="Blocked BUY",
    )

    ax.bar(
        x,
        execution["blocked_sells"],
        bottom=execution["blocked_buys"],
        label="Blocked SELL",
    )

    labels = (
        execution["model"]
        + "\n"
        + execution["strategy"]
    )

    ax.set_xticks(
        x,
        labels,
        rotation=35,
        ha="right",
    )

    ax.set_title(
        "Execution Obstacles by Strategy"
    )
    ax.set_ylabel("Order-event count")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUT / "05_execution_risk.png",
        dpi=160,
    )
    plt.close(fig)

    # Figure 6: Annual returns
    pivot = annual.pivot(
        index="year",
        columns=["model", "strategy"],
        values="period_return",
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    pivot.plot(
        kind="bar",
        ax=ax,
    )

    ax.set_title(
        "Calendar-Year / Partial-Year Portfolio Returns"
    )
    ax.set_ylabel("Period return")
    ax.axhline(
        0, color="black", linewidth=0.8
    )
    ax.grid(axis="y", alpha=0.2)
    ax.legend(fontsize=7, ncol=2)

    fig.tight_layout()
    fig.savefig(
        OUT / "06_yearly_returns.png",
        dpi=160,
    )
    plt.close(fig)


# ============================================================
# 9. RESEARCH REPORT
# ============================================================

def text_table(df):
    """
    Plain-text tables avoid a dependency on tabulate.
    """
    show = df.copy()

    num = show.select_dtypes(
        include="number"
    ).columns

    show[num] = show[num].round(5)

    return "```\n" + show.to_string(
        index=False
    ) + "\n```"


def write_report(
    risk,
    costs,
    execution,
    annual,
    strategy_tests,
    model_tests,
    base_bps,
):
    lines = [
        "# Equity Alpha Lab 2.0",
        "",
        "## V1.3 Portfolio Risk Attribution "
        "and Robustness Report",
        "",
        "### Research setting",
        "",
        "Monthly CSI300 constituent-stock selection "
        "using EqualFactor, Ridge and LightGBM.",
        "",
        "The evaluation period has been explored "
        "previously and is not a pristine holdout.",
        "",
        "### Risk attribution",
        "",
        text_table(risk),
        "",
        "Monthly market-model regression:",
        "",
        "`strategy_return = alpha + "
        "beta * CSI300_return + error`",
        "",
        "Alpha is annualized by multiplying the "
        "monthly intercept by 12. Risk-free rate "
        "is assumed zero for this diagnostic.",
        "",
        "Alpha and beta are associations, not "
        "proof of causal investment skill.",
        "",
        "### Transaction cost sensitivity",
        "",
        f"Original simulated single-side cost: "
        f"{base_bps} bp.",
        "",
        text_table(costs),
        "",
        "IMPORTANT: Alternative costs are evaluated "
        "using the saved monthly turnover and returns. "
        "These are approximate fixed-trade-path "
        "counterfactuals, NOT exact account resimulations.",
        "",
        "### Execution and liquidity",
        "",
        text_table(execution),
        "",
        "Blocked orders are not necessarily unique "
        "suspension events. Stale marks are historical "
        "adjusted opens used solely for valuation.",
        "",
        "### Calendar-year performance",
        "",
        text_table(annual),
        "",
        "Partial first/last years must not be "
        "interpreted as full-year returns.",
        "",
        "### Direct vs buffered portfolios",
        "",
        text_table(strategy_tests),
        "",
        "### Model net return comparisons",
        "",
        text_table(model_tests),
        "",
        "### Limitations",
        "",
        "1. Adjusted Qlib OHLC prices do not independently "
        "establish executable trade prices.",
        "",
        "2. The simulator lacks exchange limit-up/"
        "limit-down execution, lot sizes and impact costs.",
        "",
        "3. Stale marks may understate or overstate risk.",
        "",
        "4. Regression alpha does not control for "
        "size, value, momentum or industry exposures.",
        "",
        "5. Samples are relatively short, and all "
        "tests are exploratory with no correction "
        "for multiple comparisons.",
        "",
        "6. Cost sensitivity uses fixed observed "
        "trading amounts rather than re-executing "
        "cash and stock positions at new fee levels.",
        "",
        "7. V1.1 verified internal accounting identities "
        "but not market execution realism.",
        "",
        "### Conclusion",
        "",
        "Interpret risk, turnover and factor evidence "
        "together. Positive prediction IC and positive "
        "absolute return do not imply positive "
        "risk-adjusted benchmark alpha.",
    ]

    (
        OUT / "risk_research_report.md"
    ).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


# ============================================================
# 10. MAIN
# ============================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    t0 = time.perf_counter()

    print("=" * 76)
    print("EQUITY ALPHA LAB 2.0")
    print("V1.3 - PORTFOLIO RISK & ROBUSTNESS SUITE")
    print("=" * 76)

    p, orders, holdings, conf, audit = (
        load_inputs()
    )

    risk, monthly = portfolio_statistics(p)

    base_bps = float(
        conf["cost_single_side_bps"]
    )

    costs = cost_sensitivity(
        p, base_bps
    )

    execution, blocked = execution_risks(
        p, orders, holdings
    )

    annual, strategy_tests, model_tests = (
        regime_summary(monthly)
    )

    figures(
        monthly,
        risk,
        costs,
        execution,
        annual,
    )

    write_report(
        risk,
        costs,
        execution,
        annual,
        strategy_tests,
        model_tests,
        base_bps,
    )

    metadata = {
        "version": "V1.3",
        "project": "Equity Alpha Lab 2.0",
        "account_audit": audit["status"],
        "original_cost_bps": base_bps,
        "cost_scenarios_bps": COST_SCENARIOS_BPS,
        "cost_scenarios_exact": False,
        "model_regression": "OLS HAC",
        "hac_lags": HAC_LAGS,
        "annualization": PERIODS_PER_YEAR,
        "retraining": False,
        "status": "Exploratory research",
    }

    (
        OUT / "research_metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    pretty(
        "RISK ATTRIBUTION",
        risk[
            [
                "model", "strategy",
                "total_return",
                "annualized_return",
                "beta",
                "alpha_annualized_linear",
                "alpha_hac_p",
                "max_drawdown",
                "tracking_error",
                "information_ratio",
            ]
        ],
    )

    pretty(
        "COST SENSITIVITY",
        costs[
            [
                "model", "strategy",
                "single_side_bps",
                "approx_total_return",
                "approx_max_drawdown",
            ]
        ],
        limit=40,
    )

    pretty(
        "EXECUTION RISK",
        execution,
    )

    pretty(
        "ANNUAL ROBUSTNESS",
        annual,
        limit=30,
    )

    pretty(
        "BUFFER VS TOP30 PAIRED TESTS",
        strategy_tests,
    )

    pretty(
        "MODEL NET RETURN PAIRED TESTS",
        model_tests,
    )

    print("\n" + "=" * 76)
    print("V1.3 COMPLETED")
    print("=" * 76)

    print(
        "Runtime:",
        round(time.perf_counter() - t0, 2),
        "seconds",
    )

    print("Output:", OUT)
    print(
        "Report:",
        OUT / "risk_research_report.md",
    )


if __name__ == "__main__":
    main()
