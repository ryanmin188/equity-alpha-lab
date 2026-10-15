
"""
EQUITY ALPHA LAB 2.0
V1.2 - FACTOR RESEARCH & MODEL DIAGNOSTICS

Reads existing V1 results; no retraining or Qlib query.

Outputs:
    outputs/v1_csi300/v12_factor_research/

Research principles:
- Monthly cross-sectional Spearman Rank IC.
- Future returns used only for ex-post evaluation.
- Ten portfolios are diagnostic, not executable P&L.
- Paired model comparisons use the same dates.
- HAC tests operate on monthly observations.
- No fabricated factor decay across other horizons.
"""

from __future__ import annotations

import os

# Keep numerical libraries from overusing threads.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import spearmanr
import statsmodels.api as sm


# ===========================================================
# 1. CONFIG
# ===========================================================

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "outputs" / "v1_csi300"

OUT = SOURCE / "v12_factor_research"

PANEL_FILE = SOURCE / "monthly_asof_panel.pkl"
PRED_FILE = SOURCE / "predictions.csv"

FACTORS = [
    "mom_20",
    "mom_60",
    "reversal_5",
    "vol_20",
    "vol_60",
    "volume_surprise",
    "range_20",
    "ma_gap_20",
]

MODELS = [
    "EqualFactor",
    "Ridge",
    "LightGBM",
]

DEV_END = pd.Timestamp("2016-12-31")
EVAL_START = pd.Timestamp("2017-01-01")

MIN_STOCKS_IC = 30
MIN_STOCKS_DECILE = 100

N_DECILES = 10

HAC_LAGS = 3

# These are exploratory directions, not estimates.
FACTOR_DESCRIPTIONS = {
    "mom_20": "20-day momentum",
    "mom_60": "60-day momentum",
    "reversal_5": "5-day reversal",
    "vol_20": "20-day return volatility",
    "vol_60": "60-day return volatility",
    "volume_surprise": "Volume shock",
    "range_20": "20-day intraday range",
    "ma_gap_20": "Price relative to 20-day mean",
}


# ===========================================================
# 2. HELPERS
# ===========================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def clean_values(frame, columns):
    out = frame.copy()

    for col in columns:
        out[col] = pd.to_numeric(
            out[col],
            errors="coerce",
        )

    out[columns] = out[columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )

    return out


def assign_period(date):
    return (
        "DEVELOPMENT_2012_2016"
        if date <= DEV_END
        else "EVALUATION_2017_2020"
    )


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    good = np.isfinite(x) & np.isfinite(y)

    if good.sum() < MIN_STOCKS_IC:
        return np.nan

    x = x[good]
    y = y[good]

    if (
        np.unique(x).size < 2
        or np.unique(y).size < 2
    ):
        return np.nan

    result = spearmanr(x, y)

    return float(result.statistic)


def hac_test(values):
    s = (
        pd.Series(values)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if len(s) < 12:
        return {
            "hac_t": np.nan,
            "hac_p": np.nan,
            "hac_se": np.nan,
        }

    result = sm.OLS(
        s.to_numpy(dtype=float),
        np.ones((len(s), 1)),
    ).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": HAC_LAGS},
    )

    return {
        "hac_t": float(result.tvalues[0]),
        "hac_p": float(result.pvalues[0]),
        "hac_se": float(result.bse[0]),
    }


def describe_ic(series):
    valid = (
        pd.Series(series)
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    if valid.empty:
        return {
            "n_months": 0,
            "mean_ic": np.nan,
            "std_ic": np.nan,
            "icir": np.nan,
            "positive_month_ratio": np.nan,
            "hac_t": np.nan,
            "hac_p": np.nan,
            "hac_se": np.nan,
        }

    mean = float(valid.mean())
    std = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan

    stats = {
        "n_months": len(valid),
        "mean_ic": mean,
        "std_ic": std,
        "icir": (
            mean / std
            if np.isfinite(std) and std > 0
            else np.nan
        ),
        "positive_month_ratio": float(
            (valid > 0).mean()
        ),
    }

    stats.update(hac_test(valid))

    return stats


def save_table(frame, name):
    frame.to_csv(
        OUT / name,
        index=False,
        encoding="utf-8-sig",
    )


def print_table(title, frame, limit=30):
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)

    display = frame.head(limit).copy()

    numeric = display.select_dtypes(
        include="number"
    ).columns

    display[numeric] = display[numeric].round(6)

    print(display.to_string(index=False))


# ===========================================================
# 3. LOAD AND VALIDATE DATA
# ===========================================================

def load_panel():
    print("\n=== LOADING MONTHLY FACTOR PANEL ===")

    require(
        PANEL_FILE.is_file(),
        f"Missing {PANEL_FILE}",
    )

    raw = pd.read_pickle(PANEL_FILE)

    require(
        isinstance(raw.index, pd.MultiIndex),
        "Expected a MultiIndex factor panel.",
    )

    require(
        set(raw.index.names) == {
            "date", "instrument"
        },
        f"Unexpected panel index: {raw.index.names}",
    )

    panel = (
        raw.reset_index()
        .copy()
    )

    required = (
        FACTORS
        + [
            "date",
            "instrument",
            "forward_return",
            "entry_date",
            "exit_date",
        ]
    )

    missing = [
        col for col in required
        if col not in panel.columns
    ]

    require(
        not missing,
        f"Panel missing columns: {missing}",
    )

    panel["date"] = pd.to_datetime(
        panel["date"]
    ).dt.normalize()

    panel["entry_date"] = pd.to_datetime(
        panel["entry_date"]
    ).dt.normalize()

    panel["exit_date"] = pd.to_datetime(
        panel["exit_date"]
    ).dt.normalize()

    panel["instrument"] = (
        panel["instrument"]
        .astype(str)
        .str.upper()
    )

    require(
        not panel.duplicated(
            ["date", "instrument"]
        ).any(),
        "Duplicate date/instrument in panel.",
    )

    require(
        (
            (panel["date"] < panel["entry_date"])
            & (
                panel["entry_date"]
                < panel["exit_date"]
            )
        ).all(),
        "Invalid signal/entry/exit chronology.",
    )

    panel = clean_values(
        panel,
        FACTORS + ["forward_return"],
    )

    panel["period"] = panel[
        "date"
    ].map(assign_period)

    panel = panel.sort_values(
        ["date", "instrument"]
    ).reset_index(drop=True)

    print("Stock-month observations:", len(panel))
    print("Signal months:", panel["date"].nunique())
    print("Distinct stocks:", panel["instrument"].nunique())
    print(
        "Missing outcome observations:",
        int(panel["forward_return"].isna().sum()),
    )

    return panel


def load_predictions():
    print("\n=== LOADING EXISTING OOS PREDICTIONS ===")

    require(
        PRED_FILE.is_file(),
        f"Missing {PRED_FILE}",
    )

    preds = pd.read_csv(
        PRED_FILE,
        parse_dates=[
            "date", "entry_date", "exit_date"
        ],
    )

    required = [
        "date",
        "instrument",
        "model",
        "score",
        "forward_return",
        "entry_date",
        "exit_date",
    ]

    missing = [
        col for col in required
        if col not in preds.columns
    ]

    require(
        not missing,
        f"Prediction file missing: {missing}",
    )

    for col in [
        "date",
        "entry_date",
        "exit_date",
    ]:
        preds[col] = (
            preds[col].dt.normalize()
        )

    preds["instrument"] = (
        preds["instrument"]
        .astype(str)
        .str.upper()
    )

    preds = clean_values(
        preds,
        ["score", "forward_return"],
    )

    require(
        not preds.duplicated(
            ["date", "instrument", "model"]
        ).any(),
        "Duplicate prediction keys.",
    )

    found_models = set(
        preds["model"].unique()
    )

    require(
        found_models == set(MODELS),
        f"Unexpected models: {found_models}",
    )

    print("Prediction rows:", len(preds))
    print("Evaluation months:", preds["date"].nunique())

    return preds


# ===========================================================
# 4. MONTHLY SINGLE-FACTOR IC
# ===========================================================

def calculate_factor_ic(panel):
    print("\n=== CALCULATING SINGLE-FACTOR RANK IC ===")

    rows = []

    for date, group in panel.groupby(
        "date",
        sort=True,
    ):
        outcomes = group[
            "forward_return"
        ].to_numpy(dtype=float)

        for factor in FACTORS:
            signal = group[
                factor
            ].to_numpy(dtype=float)

            good = (
                np.isfinite(signal)
                & np.isfinite(outcomes)
            )

            rows.append({
                "date": date,
                "period": assign_period(date),
                "factor": factor,
                "n_candidates": len(group),
                "n_valid": int(good.sum()),
                "rank_ic": safe_spearman(
                    signal, outcomes
                ),
            })

    daily = pd.DataFrame(rows)

    summaries = []

    for (period, factor), g in daily.groupby(
        ["period", "factor"],
        sort=True,
    ):
        summary = describe_ic(
            g["rank_ic"]
        )

        summary.update({
            "period": period,
            "factor": factor,
        })

        summaries.append(summary)

    summary = pd.DataFrame(summaries)

    summary = summary[
        [
            "period",
            "factor",
            "n_months",
            "mean_ic",
            "std_ic",
            "icir",
            "positive_month_ratio",
            "hac_t",
            "hac_p",
            "hac_se",
        ]
    ]

    save_table(
        daily,
        "factor_monthly_ic.csv",
    )

    save_table(
        summary,
        "factor_ic_summary.csv",
    )

    return daily, summary


# ===========================================================
# 5. DECILE ANALYSIS
# ===========================================================

def calculate_deciles(panel):
    print("\n=== CALCULATING FACTOR DECILES ===")

    portfolio_rows = []
    spread_rows = []

    for date, group in panel.groupby(
        "date",
        sort=True,
    ):
        for factor in FACTORS:
            clean = group[
                ["instrument", factor, "forward_return"]
            ].dropna().copy()

            if len(clean) < MIN_STOCKS_DECILE:
                continue

            if clean[factor].nunique() < N_DECILES:
                continue

            # Stable tie break, based on stock code.
            # Never tie-break using future returns.
            clean = clean.sort_values(
                ["instrument"]
            )

            clean["factor_rank"] = clean[
                factor
            ].rank(
                method="first",
                ascending=True,
            )

            clean["decile"] = pd.qcut(
                clean["factor_rank"],
                q=N_DECILES,
                labels=False,
            ).astype(int) + 1

            means = (
                clean.groupby("decile")[
                    "forward_return"
                ]
                .mean()
            )

            if len(means) != N_DECILES:
                continue

            counts = (
                clean.groupby("decile")
                .size()
            )

            for decile in range(
                1, N_DECILES + 1
            ):
                portfolio_rows.append({
                    "date": date,
                    "period": assign_period(date),
                    "factor": factor,
                    "decile": decile,
                    "n_stocks": int(
                        counts.loc[decile]
                    ),
                    "mean_forward_return": float(
                        means.loc[decile]
                    ),
                })

            spread_rows.append({
                "date": date,
                "period": assign_period(date),
                "factor": factor,
                "top_decile_return": float(
                    means.loc[N_DECILES]
                ),
                "bottom_decile_return": float(
                    means.loc[1]
                ),
                "top_minus_bottom": float(
                    means.loc[N_DECILES]
                    - means.loc[1]
                ),
            })

    decile_monthly = pd.DataFrame(
        portfolio_rows
    )

    spreads = pd.DataFrame(
        spread_rows
    )

    require(
        not decile_monthly.empty,
        "No valid decile observations.",
    )

    avg_decile = (
        decile_monthly.groupby(
            ["period", "factor", "decile"],
            as_index=False,
        )["mean_forward_return"]
        .mean()
    )

    spread_summary = []

    for (period, factor), group in spreads.groupby(
        ["period", "factor"]
    ):
        stats = hac_test(
            group["top_minus_bottom"]
        )

        spread_summary.append({
            "period": period,
            "factor": factor,
            "months": len(group),
            "mean_top_minus_bottom": float(
                group["top_minus_bottom"].mean()
            ),
            "positive_spread_fraction": float(
                (
                    group["top_minus_bottom"] > 0
                ).mean()
            ),
            **stats,
        })

    spread_summary = pd.DataFrame(
        spread_summary
    )

    save_table(
        decile_monthly,
        "decile_monthly_returns.csv",
    )

    save_table(
        avg_decile,
        "decile_average_returns.csv",
    )

    save_table(
        spreads,
        "factor_monthly_spreads.csv",
    )

    save_table(
        spread_summary,
        "factor_spread_summary.csv",
    )

    return (
        avg_decile,
        spread_summary,
    )


# ===========================================================
# 6. CROSS-SECTIONAL FACTOR CORRELATION
# ===========================================================

def calculate_factor_correlations(panel):
    print("\n=== CALCULATING FACTOR CORRELATIONS ===")

    records = []

    for date, group in panel.groupby(
        "date",
        sort=True,
    ):
        corr = group[FACTORS].corr(
            method="spearman",
            min_periods=MIN_STOCKS_IC,
        )

        for a in FACTORS:
            for b in FACTORS:
                records.append({
                    "date": date,
                    "factor_a": a,
                    "factor_b": b,
                    "correlation": corr.loc[a, b],
                })

    long = pd.DataFrame(records)

    avg = (
        long.groupby(
            ["factor_a", "factor_b"]
        )["correlation"]
        .mean()
        .unstack("factor_b")
        .reindex(
            index=FACTORS,
            columns=FACTORS,
        )
    )

    pairs = []

    for i, a in enumerate(FACTORS):
        for b in FACTORS[i+1:]:
            r = avg.loc[a, b]

            pairs.append({
                "factor_a": a,
                "factor_b": b,
                "mean_rank_correlation": r,
                "absolute_correlation": abs(r),
            })

    pairs = (
        pd.DataFrame(pairs)
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
    )

    avg.to_csv(
        OUT / "factor_correlation_matrix.csv",
        encoding="utf-8-sig",
    )

    save_table(
        pairs,
        "factor_redundancy_pairs.csv",
    )

    return avg, pairs


# ===========================================================
# 7. MODEL RANK IC AND PAIRED TESTS
# ===========================================================

def calculate_model_ic(preds):
    print("\n=== MODEL PREDICTIVE COMPARISON ===")

    rows = []

    for (date, model), group in preds.groupby(
        ["date", "model"],
        sort=True,
    ):
        rows.append({
            "date": date,
            "model": model,
            "n_candidates": len(group),
            "n_valid": int(
                (
                    group["score"].notna()
                    & group["forward_return"].notna()
                ).sum()
            ),
            "rank_ic": safe_spearman(
                group["score"],
                group["forward_return"],
            ),
        })

    monthly = pd.DataFrame(rows)

    summary = []

    for model, group in monthly.groupby(
        "model"
    ):
        summary.append({
            "model": model,
            **describe_ic(group["rank_ic"]),
        })

    summary = pd.DataFrame(summary)

    pivot = monthly.pivot(
        index="date",
        columns="model",
        values="rank_ic",
    ).sort_index()

    comparisons = [
        ("Ridge", "EqualFactor"),
        ("LightGBM", "EqualFactor"),
        ("LightGBM", "Ridge"),
    ]

    comparison_rows = []
    paired_monthly = []

    for new, baseline in comparisons:
        paired = pivot[
            [new, baseline]
        ].dropna()

        diff = (
            paired[new] - paired[baseline]
        )

        stats = hac_test(diff)

        comparison_rows.append({
            "comparison": (
                f"{new} - {baseline}"
            ),
            "paired_months": len(diff),
            "mean_delta_rank_ic": float(
                diff.mean()
            ),
            **stats,
        })

        for date, value in diff.items():
            paired_monthly.append({
                "date": date,
                "comparison": f"{new} - {baseline}",
                "delta_rank_ic": value,
            })

    comparisons_df = pd.DataFrame(
        comparison_rows
    )

    paired_df = pd.DataFrame(
        paired_monthly
    )

    save_table(
        monthly,
        "model_monthly_ic.csv",
    )

    save_table(
        summary,
        "model_ic_summary.csv",
    )

    save_table(
        comparisons_df,
        "model_paired_tests.csv",
    )

    save_table(
        paired_df,
        "model_paired_monthly.csv",
    )

    return (
        monthly,
        summary,
        comparisons_df,
    )


# ===========================================================
# 8. PLOTS
# ===========================================================

def create_plots(
    factor_ic,
    avg_decile,
    corr,
    model_monthly,
):
    print("\n=== GENERATING RESEARCH FIGURES ===")

    plt.rcParams.update({
        "font.size": 10,
        "figure.dpi": 110,
    })

    # -------------------------------------------------------
    # 8.1 Factor IC ranking
    # -------------------------------------------------------

    evaluation = factor_ic.loc[
        factor_ic["period"]
        == "EVALUATION_2017_2020"
    ]

    ordered = evaluation.sort_values(
        "mean_ic"
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    ax.barh(
        ordered["factor"],
        ordered["mean_ic"],
    )

    ax.axvline(
        0,
        color="black",
        linewidth=0.8,
    )

    ax.set_title(
        "Single-Factor Rank IC | 2017-2020"
    )
    ax.set_xlabel(
        "Mean monthly Spearman Rank IC"
    )

    fig.tight_layout()
    fig.savefig(
        OUT / "factor_ic_ranking.png",
        dpi=160,
    )
    plt.close(fig)

    # -------------------------------------------------------
    # 8.2 Decile returns
    # -------------------------------------------------------

    evaluation_deciles = avg_decile.loc[
        avg_decile["period"]
        == "EVALUATION_2017_2020"
    ]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(15, 8),
        sharex=True,
    )

    for ax, factor in zip(
        axes.flat,
        FACTORS,
    ):
        one = evaluation_deciles.loc[
            evaluation_deciles["factor"]
            == factor
        ].sort_values("decile")

        ax.plot(
            one["decile"],
            one["mean_forward_return"],
            marker="o",
        )

        ax.set_title(factor)

        ax.axhline(
            0,
            color="gray",
            linewidth=0.7,
        )

        ax.grid(
            alpha=0.2
        )

        ax.set_xticks(
            [1, 3, 5, 7, 10]
        )

    fig.suptitle(
        "Monthly Forward Returns by Factor Decile",
        fontsize=13,
    )

    fig.supxlabel(
        "Decile (10 = highest factor value)"
    )

    fig.supylabel(
        "Average next-month return"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "factor_decile_profiles.png",
        dpi=160,
    )

    plt.close(fig)

    # -------------------------------------------------------
    # 8.3 Correlation heatmap
    # -------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    im = ax.imshow(
        corr.to_numpy(dtype=float),
        vmin=-1,
        vmax=1,
        cmap="coolwarm",
    )

    ax.set_xticks(
        range(len(FACTORS)),
        FACTORS,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(FACTORS)),
        FACTORS,
    )

    for i in range(len(FACTORS)):
        for j in range(len(FACTORS)):
            value = corr.iloc[i, j]

            if np.isfinite(value):
                ax.text(
                    j,
                    i,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    ax.set_title(
        "Average Monthly Cross-Sectional "
        "Factor Rank Correlation"
    )

    fig.colorbar(
        im,
        ax=ax,
        shrink=0.8,
    )

    fig.tight_layout()

    fig.savefig(
        OUT / "factor_correlation_heatmap.png",
        dpi=160,
    )

    plt.close(fig)

    # -------------------------------------------------------
    # 8.4 Cumulative monthly IC
    # -------------------------------------------------------

    pivot = model_monthly.pivot(
        index="date",
        columns="model",
        values="rank_ic",
    ).sort_index()

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    for model in MODELS:
        series = pivot[model]

        # Sum of observed monthly IC values,
        # NOT an investment NAV.
        ax.plot(
            series.index,
            series.fillna(0).cumsum(),
            label=model,
        )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.8,
    )

    ax.set_title(
        "Cumulative Rank IC | Diagnostic Only"
    )

    ax.set_ylabel(
        "Sum of monthly Rank IC"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUT / "model_cumulative_rank_ic.png",
        dpi=160,
    )
    plt.close(fig)

    # -------------------------------------------------------
    # 8.5 Rolling Rank IC
    # -------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(11, 5)
    )

    for model in MODELS:
        rolling = pivot[model].rolling(
            window=6,
            min_periods=6,
        ).mean()

        ax.plot(
            rolling.index,
            rolling,
            label=model,
        )

    ax.axhline(
        0,
        color="gray",
        linewidth=0.8,
    )

    ax.set_title(
        "Six-Month Rolling Mean Rank IC"
    )
    ax.set_ylabel(
        "Rolling Mean Rank IC"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    fig.tight_layout()
    fig.savefig(
        OUT / "model_rolling_rank_ic.png",
        dpi=160,
    )
    plt.close(fig)


# ===========================================================
# 9. RESEARCH REPORT
# ===========================================================

def build_report(
    panel,
    factor_ic,
    spreads,
    redundancy,
    model_summary,
    comparisons,
):
    evaluation = factor_ic.loc[
        factor_ic["period"]
        == "EVALUATION_2017_2020"
    ].sort_values(
        "mean_ic",
        ascending=False,
    )

    lines = [
        "# Equity Alpha Lab 2.0",
        "",
        "## V1.2 Factor Research Report",
        "",
        "### Scope",
        "",
        "CSI300 historical constituent stocks; "
        "monthly post-close features and forward "
        "open-to-open returns.",
        "",
        f"Panel records: {len(panel):,}",
        f"Unique signal dates: {panel['date'].nunique()}",
        f"Missing outcome rows: "
        f"{panel['forward_return'].isna().sum()}",
        "",
        "## Single-factor Rank IC",
        "",
        "Factors are evaluated using monthly "
        "cross-sectional Spearman correlation.",
        "",
        "An ICIR here is mean monthly IC divided "
        "by the sample standard deviation of monthly IC.",
        "",
        evaluation[
            [
                "factor",
                "n_months",
                "mean_ic",
                "icir",
                "positive_month_ratio",
                "hac_p",
            ]
        ].round(5).to_markdown(index=False),
        "",
        "## Decile Analysis",
        "",
        "Stocks are ranked by each factor using "
        "available forward-return observations. "
        "Portfolio returns are equally weighted "
        "diagnostic averages, not executed portfolios.",
        "",
        spreads.round(5).to_markdown(
            index=False
        ),
        "",
        "## Factor Redundancy",
        "",
        "Correlations are computed cross-sectionally "
        "within each signal month and then averaged.",
        "",
        redundancy.head(12).round(5).to_markdown(
            index=False
        ),
        "",
        "## Out-of-Sample Model Diagnostics",
        "",
        model_summary.round(5).to_markdown(
            index=False
        ),
        "",
        "### Paired Monthly Rank IC Differences",
        "",
        comparisons.round(5).to_markdown(
            index=False
        ),
        "",
        "## Interpretation and Limitations",
        "",
        "1. These are exploratory, previously inspected "
        "historical periods, not a pristine holdout.",
        "",
        "2. Future returns are used only to assess "
        "predictions after the fact.",
        "",
        "3. Missing future outcomes are excluded "
        "from IC and decile computations, never from "
        "the historical inference universe.",
        "",
        "4. Decile portfolios are not a realizable "
        "long-short backtest and exclude trading costs.",
        "",
        "5. Eight factors and multiple comparisons "
        "were inspected; individual p-values are "
        "exploratory and are not multiplicity adjusted.",
        "",
        "6. Actual multi-horizon factor decay was "
        "not calculated because independent forward "
        "5/10/20-day labels are not present.",
        "",
        "7. Company-level fundamental characteristics "
        "and industry risk data are not available "
        "in this research panel.",
        "",
        "8. Qlib price adjustment, suspended holdings "
        "and market execution assumptions remain "
        "separate implementation limitations.",
        "",
        "## Generated Figures",
        "",
        "- factor_ic_ranking.png",
        "- factor_decile_profiles.png",
        "- factor_correlation_heatmap.png",
        "- model_cumulative_rank_ic.png",
        "- model_rolling_rank_ic.png",
    ]

    report = "\n".join(lines)

    (
        OUT / "factor_research_report.md"
    ).write_text(
        report,
        encoding="utf-8",
    )


# ===========================================================
# 10. MAIN
# ===========================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 76)
    print("EQUITY ALPHA LAB 2.0")
    print("V1.2 - FACTOR RESEARCH SUITE")
    print("=" * 76)
    print("Source:", SOURCE)
    print("Output:", OUT)

    panel = load_panel()
    preds = load_predictions()

    factor_monthly, factor_ic = (
        calculate_factor_ic(panel)
    )

    avg_decile, spreads = (
        calculate_deciles(panel)
    )

    corr, redundancy = (
        calculate_factor_correlations(panel)
    )

    model_monthly, model_summary, comparisons = (
        calculate_model_ic(preds)
    )

    create_plots(
        factor_ic,
        avg_decile,
        corr,
        model_monthly,
    )

    build_report(
        panel,
        factor_ic,
        spreads,
        redundancy,
        model_summary,
        comparisons,
    )

    metadata = {
        "version": "V1.2",
        "project": "Equity Alpha Lab 2.0",
        "research_type": "Factor research",
        "number_of_factors": len(FACTORS),
        "factors": FACTOR_DESCRIPTIONS,
        "panel_rows": len(panel),
        "signal_months": panel["date"].nunique(),
        "evaluation_models": MODELS,
        "rank_ic_method": "Spearman",
        "number_of_deciles": N_DECILES,
        "hac_lags": HAC_LAGS,
        "factor_decay_calculated": False,
        "retraining_required": False,
        "status": "Exploratory development research",
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

    print_table(
        "FACTOR IC SUMMARY",
        factor_ic.sort_values(
            ["period", "mean_ic"],
            ascending=[True, False],
        ),
    )

    print_table(
        "DECILE SPREAD SUMMARY",
        spreads.sort_values(
            ["period", "factor"]
        ),
    )

    print_table(
        "HIGHEST FACTOR CORRELATIONS",
        redundancy,
        limit=12,
    )

    print_table(
        "MODEL IC SUMMARY",
        model_summary,
    )

    print_table(
        "PAIRED MODEL TESTS",
        comparisons,
    )

    print("\n" + "=" * 76)
    print("V1.2 COMPLETED")
    print("=" * 76)
    print("Output:", OUT)
    print(
        "Report:",
        OUT / "factor_research_report.md",
    )


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        category=FutureWarning,
    )

    main()
