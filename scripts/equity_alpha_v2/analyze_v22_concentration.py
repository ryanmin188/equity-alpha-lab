"""Private arithmetic security-contribution sensitivity, not a new strategy."""
from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd

from attribute_v18_holdings import replay

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/v22_execution_reality'
SOURCE=HERE/'outputs/v16_improvements'


def main():
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    rows=[]
    for folder in sorted(OUT.glob('*/*/E?')):
        daily=pd.read_csv(folder/'daily_account.csv',parse_dates=['date'])
        orders=pd.read_csv(folder/'orders.csv',parse_dates=['date','signal_date'])
        ledger,checks,_=replay(daily,orders,market)
        # Direct NAV arithmetic; removing a winner is not a tradable counterfactual.
        net=float(daily.nav.iloc[-1]-1_000_000)
        security=ledger.groupby('instrument',as_index=False).net_pnl.sum().sort_values('net_pnl',ascending=False)
        yearly=ledger.groupby(['year','instrument'],as_index=False).net_pnl.sum()
        security.to_csv(folder/'security_pnl_aggregate_private.csv',index=False)
        yearly.to_csv(folder/'yearly_security_pnl_aggregate_private.csv',index=False)
        top=security.iloc[0]
        top2025=yearly[yearly.year.eq(2025)].sort_values('net_pnl',ascending=False)
        item={'interval':folder.parents[1].name,'candidate':folder.parent.name,'regime':folder.name,
              'net_pnl_cny':net,'largest_winner':top.instrument,
              'largest_winner_pnl_cny':float(top.net_pnl),
              'net_pnl_minus_largest_winner_cny_arithmetic_only':net-float(top.net_pnl),
              'largest_winner_share_of_net_pnl':float(top.net_pnl/net) if net else float('nan'),
              'top_2025_winner':top2025.iloc[0].instrument if len(top2025) else '',
              'top_2025_winner_pnl_cny':float(top2025.iloc[0].net_pnl) if len(top2025) else float('nan'),
              'max_replay_residual_cny':float(checks.nav_residual.abs().max())}
        rows.append(item)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'largest_winner_arithmetic_sensitivity.csv',index=False)
    print(result.to_string(index=False))


if __name__=='__main__':main()
