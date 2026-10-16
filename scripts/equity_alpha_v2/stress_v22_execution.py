"""Prespecified cost/capital sensitivity; no parameter choice is made here."""
from __future__ import annotations

import pandas as pd

from research_v22_execution import (CONFIG,OUT,PORTFOLIO,SOURCE,load_predictions)
from execution_v16 import run_account as run_e0
from execution_v22 import run_account as run_e2
import pickle


def main():
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    b0=load_predictions()['B0_Equal8']
    intervals={'reused_validation_2021_2023':('2021-01-01','2023-12-01'),
               'opened_2024_2026':('2024-01-01','2026-10-08')}
    settings=[('high_slippage',1_000_000.,15.),('ten_million_capacity',10_000_000.,5.)]
    rows=[]
    for period,(start,end) in intervals.items():
        subset=b0[(b0.entry_date>=start)&(b0.exit_date<=end)]
        for name,capital,slip in settings:
            for regime in ('E0','E2'):
                if regime=='E0':
                    summary,daily,orders,_=run_e0(subset,market,PORTFOLIO,CONFIG,
                                                   capital=capital,slippage_bps=slip)
                else:
                    summary,daily,orders,_,_=run_e2(subset,market,PORTFOLIO,CONFIG,
                                                    regime='E2',capital=capital,slippage_bps=slip)
                rows.append({'period':period,'scenario':name,'regime':regime,
                             'capital':capital,'slippage_bps':slip,
                             'total_return':summary['total_return'],
                             'benchmark_return':summary['benchmark_return'],
                             'highest_weight':daily.max_weight.max(),
                             'median_holdings':daily.n_holdings.median(),
                             'blocked_orders':summary['blocked_orders']})
    output=pd.DataFrame(rows)
    output.to_csv(OUT/'cost_capacity_stress.csv',index=False)
    print(output.to_string(index=False))


if __name__=='__main__':main()
