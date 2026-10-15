"""Export a small, read-only public dashboard dataset from saved summaries.

This is a publication transform, not a rerun of the frozen strategy. It reads
only previously generated aggregate CSVs and rejects unexpected account data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUTS = (
    "market_attribution.csv",
    "v18_yearly_active_decomposition.csv",
    "v18_top_security_drivers.csv",
    "v18_ch4_regression.csv",
    "v191_factor_ic_summary.csv",
    "v191_quantile_summary.csv",
    "v191_mean_abs_factor_correlation.csv",
    "v16_execution_stress.csv",
)


def _finite(value):
    if pd.isna(value):
        return None
    return round(float(value), 10)


def build(summary_dir: Path) -> dict:
    summary_dir = Path(summary_dir)
    missing = [name for name in INPUTS if not (summary_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing saved aggregate input(s): {', '.join(missing)}")
    sources = {name: hashlib.sha256((summary_dir / name).read_bytes()).hexdigest() for name in INPUTS}
    read = lambda name: pd.read_csv(summary_dir / name)

    market = read("market_attribution.csv")
    expected = {"date", "convention", "portfolio_wealth", "benchmark_wealth"}
    if not expected.issubset(market) or len(market) != 667 or market.date.duplicated().any():
        raise ValueError("Unexpected frozen daily summary schema or row count")
    if market.date.iloc[0] != "2024-01-02" or market.date.iloc[-1] != "2026-10-08":
        raise ValueError("Unexpected frozen period")
    if abs(float(market.portfolio_wealth.iloc[-1]) - 1.3771236079981812) > 1e-8:
        raise ValueError("Frozen portfolio endpoint changed")
    if abs(float(market.benchmark_wealth.iloc[-1]) - 1.2710434) > 1e-8:
        raise ValueError("Frozen benchmark endpoint changed")
    daily = [
        [str(r.date), _finite(r.portfolio_wealth), _finite(r.benchmark_wealth), str(r.convention)]
        for r in market.itertuples(index=False)
    ]

    yearly = read("v18_yearly_active_decomposition.csv")
    yearly_rows = [
        {"year": int(r.year), "portfolio_log_relative": _finite(r.portfolio_log_relative),
         "security_log_relative": _finite(r.security_log_relative),
         "cash_log_relative": _finite(r.cash_log_relative)}
        for r in yearly.itertuples(index=False)
    ]
    drivers = read("v18_top_security_drivers.csv")
    driver_rows = [
        {"year": int(r.year), "direction": str(r.direction), "rank": int(r.rank),
         "instrument": str(r.instrument), "net_pnl_cny": _finite(r.net_pnl_cny)}
        for r in drivers.itertuples(index=False)
        if int(r.year) in (2024, 2025) and int(r.rank) <= 10
    ]

    reg = read("v18_ch4_regression.csv")
    reg = reg.loc[reg.period.eq("2024-2025_primary") & reg.model.isin(["CH_market", "CH4"])]
    if len(reg) != 2 or set(reg.observations) != {484}:
        raise ValueError("The fixed CH-4 comparison must contain two 484-day models")
    regressions = [
        {"model": str(r.model), "observations": int(r.observations),
         "alpha_linear_annual": _finite(r.alpha_linear_annual),
         "alpha_hac_p": _finite(r.alpha_hac_p), "r_squared": _finite(r.r_squared),
         "beta_vmg": _finite(r.beta_VMG)} for r in reg.itertuples(index=False)
    ]

    ic = read("v191_factor_ic_summary.csv")
    ic = ic.loc[ic.period.eq("opened_final_2024_2026")].sort_values("factor")
    if len(ic) != 28 or ic.factor.duplicated().any():
        raise ValueError("Expected 28 frozen factor rows")
    groups = read("v191_quantile_summary.csv")
    groups = groups.loc[groups.period.eq("opened_final_2024_2026")]
    group_lookup = {(r.factor, int(r.n_groups)): r for r in groups.itertuples(index=False)}
    factors = []
    for r in ic.itertuples(index=False):
        item = {"name": str(r.factor), "months": int(r.n_months),
                "mean_rank_ic": _finite(r.mean_rank_ic), "icir": _finite(r.icir_monthly_unannualized),
                "hac_p": _finite(r.hac_p), "bh_q": _finite(r.bh_q_28),
                "coverage_stocks": _finite(r.mean_stocks)}
        for n in (5, 10):
            g = group_lookup.get((r.factor, n))
            item[f"spread_{n}"] = _finite(g.mean_high_minus_low) if g is not None else None
            item[f"group_months_{n}"] = int(g.n_months) if g is not None else 0
        factors.append(item)

    corr = read("v191_mean_abs_factor_correlation.csv").set_index("Unnamed: 0")
    names = [f["name"] for f in factors]
    if set(corr.index) != set(names) or set(corr.columns) != set(names):
        raise ValueError("Factor correlation labels do not match the frozen factor set")
    corr_rows = [[_finite(corr.loc[a, b]) for b in names] for a in names]

    stress = read("v16_execution_stress.csv")
    stress_rows = [
        {"capital_cny": int(r.capital), "slippage_bps": _finite(r.slippage_bps),
         "total_return": _finite(r.total_return), "max_drawdown": _finite(r.max_drawdown)}
        for r in stress.itertuples(index=False)
    ]
    return {
        "schema": "equity-alpha-lab-showcase-v1",
        "status": "historical_simulation_read_only",
        "source_sha256": sources,
        "daily_columns": ["date", "portfolio_wealth", "benchmark_wealth", "convention"],
        "daily": daily,
        "yearly_relative": yearly_rows,
        "top_drivers": driver_rows,
        "regressions": regressions,
        "factors": factors,
        "correlation": {"names": names, "values": corr_rows},
        "execution_stress": stress_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-dir", type=Path, default=ROOT / "reports" / "summary")
    parser.add_argument("--output", type=Path, default=ROOT / "showcase" / "data" / "research.json")
    args = parser.parse_args()
    result = build(args.summary_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes((json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    print(f"Wrote {args.output}: {len(result['daily'])} account days, {len(result['factors'])} factors")


if __name__ == "__main__":
    main()
