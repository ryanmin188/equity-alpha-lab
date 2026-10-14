
"""
EQUITY ALPHA LAB 2.0
V1.5 - FINAL VALIDATION & GITHUB RELEASE

Goals:
1. Verify accounting / report consistency.
2. Diagnose 100% cash observations.
3. Analyze blocked orders and stale valuations.
4. Check fees against executed notional.
5. Produce a clean GitHub release candidate.
6. Explicitly document unverified execution assumptions.

No model fitting.
No new strategy backtest.
No internet access.
No GitHub push.

Run directly in PyCharm.
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
import hashlib
import json
import shutil
import sys

import numpy as np
import pandas as pd


# ===========================================================
# CONFIGURATION
# ===========================================================

ROOT = Path(__file__).resolve().parent

SOURCE = ROOT / "outputs" / "v1_csi300"

V11 = SOURCE / "v11_audit"
V12 = SOURCE / "v12_factor_research"
V13 = SOURCE / "v13_risk_research"
V14 = SOURCE / "v14_publication"

OUT = SOURCE / "v15_final_validation"

PACKAGE = OUT / "github_release_candidate"

EXPECTED_MODELS = {
    "EqualFactor",
    "Ridge",
    "LightGBM",
}

EXPECTED_STRATEGIES = {
    "Top30",
    "Buffer30_60",
}

SOURCE_SCRIPTS = [
    "run_research.py",
    "audit_v11_accounting.py",
    "research_v12_factors.py",
    "research_v13_risk.py",
    "build_v14_report.py",
    "finalize_v15_release.py",
]

ABS_TOL = 2e-7
REL_TOL = 2e-6


# ===========================================================
# UTILITIES
# ===========================================================

def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def load_csv(path, dates=()):
    require(
        path.is_file(),
        f"Required file not found: {path}",
    )

    data = pd.read_csv(
        path,
        parse_dates=list(dates),
        low_memory=False,
    )

    require(
        not data.empty,
        f"Input file is empty: {path}",
    )

    return data


def load_json(path):
    require(
        path.is_file(),
        f"Required file not found: {path}",
    )

    return json.loads(
        path.read_text(encoding="utf-8")
    )


def check_columns(df, columns, name):
    missing = sorted(set(columns) - set(df.columns))

    require(
        not missing,
        f"{name} missing columns: {missing}",
    )


def is_close(a, b):
    return bool(np.isclose(
        float(a),
        float(b),
        atol=ABS_TOL,
        rtol=REL_TOL,
    ))


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for chunk in iter(
            lambda: f.read(1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def write_csv(df, name):
    df.to_csv(
        OUT / name,
        index=False,
        encoding="utf-8-sig",
    )


def print_table(title, df, limit=20):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)

    show = df.head(limit).copy()

    numbers = show.select_dtypes(
        include="number"
    ).columns

    show[numbers] = show[numbers].round(6)

    print(show.to_string(index=False))


# ===========================================================
# 1. LOAD EXISTING OUTPUTS
# ===========================================================

def load_inputs():
    print("\n[1/6] Loading completed research artifacts")

    paths = {
        "account": SOURCE / "paper_backtest.csv",
        "orders": SOURCE / "execution_audit.csv",
        "holdings": SOURCE / "holdings_audit.csv",
        "predictions": SOURCE / "predictions.csv",
        "metadata": SOURCE / "run_metadata.json",
        "audit": V11 / "audit_summary.json",
        "v13_risk": V13 / "risk_attribution.csv",
        "v14_manifest": V14 / "integrity_manifest.json",
        "dashboard": V14 / "research_dashboard.html",
        "report": V14 / "RESEARCH_REPORT.md",
        "readme": V14 / "README_GITHUB.md",
    }

    for name, path in paths.items():
        require(path.is_file(), f"Missing {name}: {path}")

    p = load_csv(
        paths["account"],
        ("date", "entry_date", "exit_date"),
    )

    o = load_csv(
        paths["orders"],
        ("date", "execution_date"),
    )

    h = load_csv(
        paths["holdings"],
        ("date", "valuation_date"),
    )

    pred = load_csv(
        paths["predictions"],
        ("date", "entry_date", "exit_date"),
    )

    risk = load_csv(paths["v13_risk"])

    metadata = load_json(paths["metadata"])
    audit = load_json(paths["audit"])
    manifest = load_json(paths["v14_manifest"])

    check_columns(
        p,
        [
            "model", "strategy", "date",
            "entry_date", "exit_date",
            "wealth", "net", "benchmark",
            "cash_weight", "n_holdings",
            "traded_notional", "cost_fraction",
            "terminal_unliquidated",
        ],
        "paper_backtest",
    )

    check_columns(
        o,
        [
            "model", "strategy", "date",
            "execution_date", "status",
            "side", "instrument",
            "shares", "notional",
        ],
        "execution_audit",
    )

    check_columns(
        h,
        [
            "model", "strategy", "date",
            "valuation_date", "instrument",
            "shares", "marked_price",
            "marked_value", "stale",
        ],
        "holdings_audit",
    )

    check_columns(
        risk,
        [
            "model", "strategy",
            "total_return", "beta",
            "alpha_hac_p",
        ],
        "risk_attribution",
    )

    require(
        audit.get("status")
        == "PASS_INTERNAL_IDENTITIES_ONLY",
        "V1.1 internal accounting audit did not pass.",
    )

    require(
        set(p["model"]) == EXPECTED_MODELS,
        "Unexpected models.",
    )

    require(
        set(p["strategy"]) == EXPECTED_STRATEGIES,
        "Unexpected strategies.",
    )

    require(
        not p.duplicated(
            ["model", "strategy", "date"]
        ).any(),
        "Duplicate account-month rows.",
    )

    return {
        "account": p,
        "orders": o,
        "holdings": h,
        "predictions": pred,
        "risk": risk,
        "metadata": metadata,
        "audit": audit,
        "manifest": manifest,
        "paths": paths,
    }


# ===========================================================
# 2. CASH AND TERMINAL LIQUIDATION DIAGNOSTICS
# ===========================================================

def cash_diagnostics(d):
    print("\n[2/6] Checking cash weights and liquidation")

    p = d["account"].sort_values(
        ["model", "strategy", "entry_date"]
    ).copy()

    h = d["holdings"]

    p["cash_weight"] = pd.to_numeric(
        p["cash_weight"], errors="raise"
    )

    p["n_holdings"] = pd.to_numeric(
        p["n_holdings"], errors="raise"
    )

    p["is_terminal_month"] = (
        p.groupby(
            ["model", "strategy"]
        ).cumcount()
        ==
        p.groupby(
            ["model", "strategy"]
        )["date"].transform("size") - 1
    )

    p["near_100pct_cash"] = (
        p["cash_weight"] >= 0.999999
    )

    p["nonterminal_full_cash"] = (
        p["near_100pct_cash"]
        & ~p["is_terminal_month"]
    )

    p["terminal_full_cash"] = (
        p["near_100pct_cash"]
        & p["is_terminal_month"]
    )

    p["nonterminal_cash_weight"] = (
        p["cash_weight"].where(
            ~p["is_terminal_month"]
        )
    )

    rows = []

    for (model, strategy), g in p.groupby(
        ["model", "strategy"]
    ):
        terminal = g.loc[
            g["is_terminal_month"]
        ].iloc[0]

        normal = g.loc[
            ~g["is_terminal_month"]
        ]

        rows.append({
            "model": model,
            "strategy": strategy,
            "months": len(g),
            "terminal_cash_weight": float(
                terminal["cash_weight"]
            ),
            "terminal_n_holdings": int(
                terminal["n_holdings"]
            ),
            "full_cash_months": int(
                g["near_100pct_cash"].sum()
            ),
            "full_cash_nonterminal_months": int(
                g["nonterminal_full_cash"].sum()
            ),
            "mean_cash_including_terminal": float(
                g["cash_weight"].mean()
            ),
            "mean_cash_excluding_terminal": float(
                normal["cash_weight"].mean()
            ) if len(normal) else np.nan,
            "max_cash_excluding_terminal": float(
                normal["cash_weight"].max()
            ) if len(normal) else np.nan,
        })

    summary = pd.DataFrame(rows)

    fully_cash = p.loc[
        p["near_100pct_cash"],
        [
            "model", "strategy", "date",
            "entry_date", "exit_date",
            "cash_weight", "n_holdings",
            "is_terminal_month",
        ],
    ].copy()

    # Check whether the final holdings snapshot has
    # zero remaining securities for liquidated accounts.
    for row in summary.itertuples(index=False):
        account = p.loc[
            (p["model"] == row.model)
            & (p["strategy"] == row.strategy)
        ].sort_values("entry_date")

        last_date = account.iloc[-1]["date"]

        last_holdings = h.loc[
            (h["model"] == row.model)
            & (h["strategy"] == row.strategy)
            & (h["date"] == last_date)
        ]

        reported_n = int(
            account.iloc[-1]["n_holdings"]
        )

        require(
            len(last_holdings) == reported_n,
            f"Terminal inventory mismatch: "
            f"{row.model}/{row.strategy}",
        )

    write_csv(summary, "cash_diagnostics.csv")
    write_csv(fully_cash, "full_cash_months.csv")

    return summary


# ===========================================================
# 3. TRADE, FEE AND NAV CHECK
# ===========================================================

def ledger_consistency(d):
    print("\n[3/6] Cross-checking trading and account values")

    p = d["account"]
    o = d["orders"]
    h = d["holdings"]
    risk = d["risk"]

    rate = (
        float(d["metadata"]["cost_single_side_bps"])
        / 10000.0
    )

    order_groups = {
        key: group
        for key, group in o.groupby(
            ["model", "strategy", "date"],
            sort=False,
        )
    }

    holding_groups = {
        key: group
        for key, group in h.groupby(
            ["model", "strategy", "date"],
            sort=False,
        )
    }

    problems = []
    reconciled = []

    for (model, strategy), g in p.groupby(
        ["model", "strategy"]
    ):
        g = g.sort_values("entry_date")

        prev_nav = 1.0

        for row in g.itertuples(index=False):
            key = (model, strategy, row.date)

            orders = order_groups.get(
                key, pd.DataFrame(columns=o.columns)
            )

            holdings = holding_groups.get(
                key, pd.DataFrame(columns=h.columns)
            )

            filled = orders.loc[
                orders["status"].isin(
                    ["FILLED", "FILLED_TERMINAL"]
                )
            ]

            notional = float(
                filled["notional"].sum()
            )

            expected_turnover = (
                notional / prev_nav
            )

            expected_fee_fraction = (
                notional * rate / prev_nav
            )

            expected_holdings = (
                float(row.wealth)
                * (1 - float(row.cash_weight))
            )

            actual_holdings = float(
                holdings["marked_value"].sum()
            )

            checks = {
                "NAV": is_close(
                    row.wealth,
                    prev_nav * (1 + row.net),
                ),
                "turnover": is_close(
                    row.traded_notional,
                    expected_turnover,
                ),
                "fees": is_close(
                    row.cost_fraction,
                    expected_fee_fraction,
                ),
                "positions": is_close(
                    actual_holdings,
                    expected_holdings,
                ),
                "holding_count": (
                    int(row.n_holdings)
                    == len(holdings)
                ),
            }

            for name, passed in checks.items():
                if not passed:
                    problems.append({
                        "model": model,
                        "strategy": strategy,
                        "date": row.date,
                        "check": name,
                    })

            reconciled.append({
                "model": model,
                "strategy": strategy,
                "date": row.date,
                "nav": row.wealth,
                "filled_notional": notional,
                "inferred_fees": notional * rate,
                "holdings_value": actual_holdings,
                "implied_cash": (
                    float(row.wealth)
                    - actual_holdings
                ),
                "all_checks_passed": all(
                    checks.values()
                ),
            })

            prev_nav = float(row.wealth)

        final_return = prev_nav - 1.0

        found = risk.loc[
            (risk["model"] == model)
            & (risk["strategy"] == strategy),
            "total_return",
        ]

        if (
            len(found) != 1
            or not is_close(
                final_return,
                found.iloc[0],
            )
        ):
            problems.append({
                "model": model,
                "strategy": strategy,
                "date": pd.NaT,
                "check": "V1.3_total_return",
            })

    problems_df = pd.DataFrame(
        problems,
        columns=[
            "model", "strategy", "date", "check"
        ],
    )

    ledger_df = pd.DataFrame(reconciled)

    write_csv(
        problems_df,
        "ledger_failures.csv",
    )

    write_csv(
        ledger_df,
        "ledger_reconciliation.csv",
    )

    return problems_df


# ===========================================================
# 4. EXECUTION RISK SUMMARY
# ===========================================================

def execution_diagnostics(d):
    print("\n[4/6] Summarizing execution limitations")

    o = d["orders"].copy()
    h = d["holdings"].copy()

    o["blocked"] = (
        o["status"].astype(str)
        .str.startswith("BLOCKED")
    )

    h["stale_flag"] = (
        h["stale"].astype(str)
        .str.lower()
        .eq("true")
    )

    blocked = o.loc[o["blocked"]]
    stale = h.loc[h["stale_flag"]]

    blocked_types = (
        blocked.groupby(
            ["status", "side"],
            dropna=False,
        )
        .agg(
            order_records=("instrument", "size"),
            unique_stocks=("instrument", "nunique"),
        )
        .reset_index()
    )

    by_strategy = (
        blocked.groupby(
            ["model", "strategy"]
        ).agg(
            blocked_orders=("instrument", "size"),
            blocked_unique_stocks=(
                "instrument", "nunique"
            ),
        )
        .reset_index()
    )

    stale_summary = (
        stale.groupby(
            ["model", "strategy"]
        ).agg(
            stale_holding_records=(
                "instrument", "size"
            ),
            unique_stale_stocks=(
                "instrument", "nunique"
            ),
        )
        .reset_index()
    )

    summary = by_strategy.merge(
        stale_summary,
        on=["model", "strategy"],
        how="outer",
    ).fillna(0)

    write_csv(
        blocked_types,
        "blocked_order_categories.csv",
    )

    write_csv(
        summary,
        "execution_limitations.csv",
    )

    # Diagnostic evidence only.
    # A stale adjusted price cannot establish true
    # mark-to-market value or exchange executability.
    return summary


# ===========================================================
# 5. DOCUMENT AND REPRODUCIBILITY CHECKS
# ===========================================================

def publication_checks(d):
    print("\n[5/6] Checking research publication artifacts")

    paths = d["paths"]
    manifest = d["manifest"]

    checks = []

    for name in (
        "dashboard", "report", "readme"
    ):
        path = paths[name]

        checks.append({
            "check": f"{name}_exists",
            "passed": path.is_file()
            and path.stat().st_size > 100,
            "detail": str(path),
        })

    manifest_files = manifest.get("files", {})

    # V1.4 recorded hashes of publication inputs
    # and final output files. Check whether they
    # still correspond to the current filesystem.
    mismatches = []

    for label, item in manifest_files.items():
        file_path = Path(item["path"])

        if not file_path.is_file():
            mismatches.append(
                f"{label}: missing"
            )
            continue

        expected_hash = item.get("sha256")

        if expected_hash != sha256(file_path):
            mismatches.append(
                f"{label}: SHA256 changed"
            )

    checks.append({
        "check": "v14_manifest_integrity",
        "passed": len(mismatches) == 0,
        "detail": "; ".join(mismatches)
        if mismatches else "All recorded files match",
    })

    source_missing = [
        name for name in SOURCE_SCRIPTS
        if not (ROOT / name).is_file()
    ]

    checks.append({
        "check": "research_scripts_exist",
        "passed": not source_missing,
        "detail": (
            ", ".join(source_missing)
            if source_missing else
            "All expected source scripts found"
        ),
    })

    # Check existence of source research artifacts.
    for name, path in (
        (
            "factor_report",
            V12 / "factor_research_report.md",
        ),
        (
            "risk_report",
            V13 / "risk_research_report.md",
        ),
        (
            "v11_audit",
            V11 / "audit_report.md",
        ),
    ):
        checks.append({
            "check": name,
            "passed": path.is_file(),
            "detail": str(path),
        })

    df = pd.DataFrame(checks)

    write_csv(
        df,
        "publication_checks.csv",
    )

    return df, source_missing


# ===========================================================
# 6. RELEASE CANDIDATE
# ===========================================================

def create_release_package(source_missing):
    print("\n[6/6] Building local release candidate")

    scripts_dir = PACKAGE / "scripts" / "equity_alpha_v2"
    reports_dir = PACKAGE / "reports"

    scripts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    reports_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for name in SOURCE_SCRIPTS:
        src = ROOT / name

        if src.is_file():
            shutil.copy2(
                src,
                scripts_dir / name,
            )

    for src, dest in (
        (
            V14 / "RESEARCH_REPORT.md",
            reports_dir / "RESEARCH_REPORT.md",
        ),
        (
            V14 / "research_dashboard.html",
            reports_dir / "research_dashboard.html",
        ),
        (
            V11 / "audit_report.md",
            reports_dir / "ACCOUNT_AUDIT.md",
        ),
        (
            V12 / "factor_research_report.md",
            reports_dir / "FACTOR_RESEARCH.md",
        ),
        (
            V13 / "risk_research_report.md",
            reports_dir / "RISK_RESEARCH.md",
        ),
    ):
        if src.is_file():
            shutil.copy2(src, dest)

    readme_src = V14 / "README_GITHUB.md"

    if readme_src.is_file():
        shutil.copy2(
            readme_src,
            PACKAGE / "README.md",
        )

    gitignore = """
# Python
__pycache__/
*.py[cod]
.venv/
venv/
.env
.env.*
!.env.example

# IDE
.idea/
.vscode/

# Local datasets and model artifacts
.qlib/
qlib_data/
*.pkl
*.pickle
*.parquet
*.npy
*.npz

# Research caches and bulk outputs
scripts/equity_alpha_v2/outputs/
outputs/
data/
datasets/
cache/

# Operating system
.DS_Store
Thumbs.db

# Secrets and credentials
*.key
credentials.json
secrets.json
""".lstrip()

    (
        PACKAGE / ".gitignore"
    ).write_text(
        gitignore,
        encoding="utf-8",
    )

    # Document the environment without enforcing
    # guessed dependency versions.
    environment = {
        "python": sys.version.split()[0],
        "packages": {},
    }

    from importlib import metadata

    for package in (
        "pyqlib",
        "pandas",
        "numpy",
        "scipy",
        "statsmodels",
        "scikit-learn",
        "lightgbm",
        "matplotlib",
        "tabulate",
    ):
        try:
            environment["packages"][package] = (
                metadata.version(package)
            )
        except metadata.PackageNotFoundError:
            environment["packages"][package] = None

    (
        PACKAGE / "environment_versions.json"
    ).write_text(
        json.dumps(
            environment,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    requirements = [
        f"{package}=={version}"
        for package, version
        in environment["packages"].items()
        if version is not None
    ]

    (
        PACKAGE / "requirements.txt"
    ).write_text(
        "\n".join(requirements) + "\n",
        encoding="utf-8",
    )

    return environment


# ===========================================================
# 7. FINAL REPORT
# ===========================================================

def final_report(
    cash,
    ledger_failures,
    execution,
    publication,
    environment,
):
    internal_pass = ledger_failures.empty

    publication_pass = bool(
        publication["passed"].all()
    )

    nonterminal_cash = int(
        cash["full_cash_nonterminal_months"].sum()
    )

    if not internal_pass:
        status = "HOLD_ACCOUNTING_FAILURE"
    elif not publication_pass:
        status = "HOLD_PUBLICATION_INCOMPLETE"
    elif nonterminal_cash:
        status = "REVIEW_NONTERMINAL_FULL_CASH"
    else:
        status = "RESEARCH_RELEASE_CANDIDATE"

    limitations = [
        "No independent unadjusted execution-price "
        "and corporate-action reconciliation.",
        "No exchange-level limit-up/limit-down, "
        "lot-size or market-impact verification.",
        "Stale adjusted prices may misstate the "
        "economic value of suspended positions.",
        "The evaluation period has previously "
        "been explored during research.",
        "No demonstrated statistically significant "
        "positive market-adjusted alpha.",
        "Cost scenarios are first-order fixed-path "
        "approximations, not full trade replays.",
        "Internal accounting checks reuse outputs "
        "generated by the same simulator.",
    ]

    report = [
        "# Equity Alpha Lab 2.0 — V1.5 Final Validation",
        "",
        f"**Release status:** `{status}`",
        "",
        "## Internal Accounting",
        "",
        f"- Ledger failures: {len(ledger_failures)}",
        "- V1.1 internal accounting audit: passed",
        "",
        "## Cash and Liquidation",
        "",
        "The final month includes terminal "
        "liquidation, which may create a "
        "100% cash-weight observation.",
        "",
        "Cash weights after terminal liquidation "
        "should not be interpreted as normal "
        "portfolio allocations.",
        "",
        f"- Nonterminal 100% cash observations: "
        f"{nonterminal_cash}",
        "",
        "```",
        cash.to_string(index=False),
        "```",
        "",
        "## Execution Limitations",
        "",
        "```",
        execution.to_string(index=False),
        "```",
        "",
        "## Publication Checks",
        "",
        "```",
        publication.to_string(index=False),
        "```",
        "",
        "## Unresolved Market-Realism Requirements",
        "",
        *[f"- {x}" for x in limitations],
        "",
        "## Interpretation",
        "",
        "RESEARCH_RELEASE_CANDIDATE authorizes "
        "consideration of this project as "
        "historical academic/research work only. "
        "It does not establish tradable alpha "
        "or authorize claims of production readiness.",
        "",
        "Before uploading, inspect the package "
        "for private information and check the "
        "licenses of any third-party source code.",
        "",
        "## Environment",
        "",
        "```json",
        json.dumps(
            environment,
            indent=2,
            ensure_ascii=False,
        ),
        "```",
    ]

    (
        OUT / "FINAL_VALIDATION_REPORT.md"
    ).write_text(
        "\n".join(report),
        encoding="utf-8",
    )

    summary = {
        "status": status,
        "accounting_failures": len(
            ledger_failures
        ),
        "publication_checks_passed": (
            publication_pass
        ),
        "nonterminal_full_cash_months": (
            nonterminal_cash
        ),
        "research_publication_only": True,
        "live_execution_verified": False,
        "adjusted_price_accounting_verified": False,
        "evaluation_pristine_holdout": False,
        "limitations": limitations,
    }

    (
        OUT / "release_status.json"
    ).write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return summary


# ===========================================================
# MAIN
# ===========================================================

def main():
    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 72)
    print("EQUITY ALPHA LAB 2.0")
    print("V1.5 - FINAL VALIDATION & GITHUB RELEASE")
    print("=" * 72)

    data = load_inputs()

    cash = cash_diagnostics(data)

    ledger_failures = ledger_consistency(data)

    execution = execution_diagnostics(data)

    publication, source_missing = (
        publication_checks(data)
    )

    environment = create_release_package(
        source_missing
    )

    status = final_report(
        cash,
        ledger_failures,
        execution,
        publication,
        environment,
    )

    print_table(
        "CASH / TERMINAL LIQUIDATION",
        cash,
    )

    print_table(
        "EXECUTION RISKS",
        execution,
    )

    print_table(
        "PUBLICATION CHECKS",
        publication,
    )

    print("\n" + "=" * 72)
    print("FINAL RELEASE STATUS")
    print("=" * 72)

    print("Status:", status["status"])
    print(
        "Accounting failures:",
        status["accounting_failures"],
    )
    print(
        "Nonterminal full-cash months:",
        status["nonterminal_full_cash_months"],
    )
    print(
        "Publication checks passed:",
        status["publication_checks_passed"],
    )

    print("\nValidation report:")
    print(OUT / "FINAL_VALIDATION_REPORT.md")

    print("\nGitHub release candidate:")
    print(PACKAGE)

    print(
        "\nIMPORTANT: Research release does not "
        "mean exchange-level execution was verified."
    )


if __name__ == "__main__":
    main()
