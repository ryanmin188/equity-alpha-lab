"""Deterministic synthetic-data walkthrough of the existing execution engine.

The prices and scores here are invented for software demonstration. They do
not reproduce the CSI300 research or its published performance statistics.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "equity_alpha_v2"))
from execution_v16 import Market, run_account  # noqa: E402
from execution_v22 import run_account as run_v22  # noqa: E402


def build_example() -> tuple[pd.DataFrame, Market, dict, dict]:
    """Create a tiny, point-in-time synthetic market and monthly ranks."""
    dates = pd.bdate_range("2023-01-02", periods=110)
    symbols = [f"SZ000{i:03d}" for i in range(1, 11)]
    t = np.arange(len(dates), dtype=float)[:, None]
    s = np.arange(len(symbols), dtype=float)[None, :]
    close = 10 + .013 * t * (1 + .13 * s) + .34 * np.sin(t / 6 + s * .8)
    open_px = close * (1 + .0015 * np.cos(t / 5 + s))
    volume = np.broadcast_to(1_200_000 + 50_000 * s, close.shape).copy()
    market = Market(
        dates=dates,
        symbols=symbols,
        fields={
            "open": open_px,
            "close": close,
            "high": np.maximum(open_px, close) * 1.005,
            "low": np.minimum(open_px, close) * .995,
            "volume": volume,
            "factor": np.ones_like(close),
        },
        benchmark_open=12 + .006 * np.arange(len(dates)),
        benchmark_close=12 + .006 * np.arange(len(dates)) + .004,
    )
    rows = []
    for entry_i in (25, 45, 65, 85):
        signal_i = entry_i - 1
        # Only prices available at the preceding signal close enter the rank.
        past_return = close[signal_i] / close[signal_i - 10] - 1
        scores = past_return + .003 * np.cos(s.ravel() + entry_i / 8)
        for j, symbol in enumerate(symbols):
            rows.append({
                "date": dates[signal_i],
                "entry_date": dates[entry_i],
                "exit_date": dates[-1],
                "instrument": symbol,
                "score": float(scores[j]),
                "raw_vol_60": .02,
            })
    portfolio = {"name": "SyntheticBuffer3_6", "top_n": 3, "buffer_n": 6, "weighting": "equal"}
    config = {
        "capital": 100_000,
        "max_stock_weight": .4,
        "max_rebalance_gross_turnover": .8,
        "minimum_commission_cny": 5.,
        "commission_bps": 3.,
        "sell_tax_bps_before_2023_08_28": 10.,
        "sell_tax_bps_from_2023_08_28": 5.,
        "transfer_fee_bps_assumption": .2,
        "slippage_bps": 5.,
        "impact_coefficient": .1,
        "max_participation_of_lagged_adv": .01,
    }
    return pd.DataFrame(rows), market, portfolio, config


def run_demo(regime: str = 'E0'):
    inputs=build_example()
    if regime=='E0':
        return run_account(*inputs)
    return run_v22(*inputs,regime=regime)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a synthetic Equal8-style turnover and execution demo")
    parser.add_argument("--output", type=Path, default=ROOT / "examples" / "demo_output")
    parser.add_argument("--regime", choices=('E0','E1','E2'), default='E0')
    args = parser.parse_args()
    result=run_demo(args.regime)
    summary,daily,orders,targets=result[:4]
    args.output.mkdir(parents=True, exist_ok=True)
    daily.to_csv(args.output / "synthetic_daily.csv", index=False)
    orders.to_csv(args.output / "synthetic_orders.csv", index=False)
    targets.to_csv(args.output / "synthetic_rebalances.csv", index=False)
    if len(result)==5:
        result[4].to_csv(args.output / "synthetic_exposures.csv",index=False)
    print("SYNTHETIC EXAMPLE — not the published CSI300 performance")
    print(f"Target decisions: {len(targets)} | order events: {len(orders)} | fees: CNY {summary['fees_cny']:.2f}")
    print(f"Terminal NAV: CNY {daily.nav.iloc[-1]:,.2f} | return: {summary['total_return']:.2%}")
    print(f"CSV output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
