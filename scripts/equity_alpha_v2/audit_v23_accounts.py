"""Independent cash/quantity replay of all V2.3 accounts, including stress sizes."""
import pickle

import numpy as np
import pandas as pd

from research_v23_signals import OUT,SOURCE


def replay(daily,orders,market):
    initial=float(daily.nav.iloc[0]/daily.wealth.iloc[0])
    cash=initial
    quantity=np.zeros(len(market.symbols))
    previous=initial
    by_date={d:g for d,g in orders.groupby('date')}
    checks=[]
    for row in daily.itertuples(index=False):
        for trade in by_date.get(row.date,pd.DataFrame()).itertuples(index=False):
            if trade.status.startswith('BLOCKED'):continue
            j=market.symbol_index[trade.instrument]
            direction=1 if trade.side=='BUY' else -1
            quantity[j]+=direction*trade.filled_adjusted_units
            cash-=direction*trade.notional+trade.fees
        if quantity.min() < -1e-7 or cash < -1e-5:
            raise AssertionError('Negative quantity/cash')
        i=market.date_index[row.date]
        observed=market.fields[row.valuation][i]
        fallback=market.marks[i-1] if row.valuation=='open' else market.marks[i]
        marks=np.where(np.isfinite(observed)&(observed>0),observed,fallback)
        held=quantity>1e-8
        if not np.isfinite(marks[held]).all():raise AssertionError('Unvalued holding')
        nav=cash+quantity[held]@marks[held]
        checks.append({'date':row.date,'nav_residual':nav-row.nav,'cash_residual':cash-row.cash,
                       'return_residual':nav/previous-1-row.net})
        previous=nav
    result=pd.DataFrame(checks)
    if result[['nav_residual','cash_residual']].abs().max().max()>1e-5:
        raise AssertionError('Unreconciled account')
    return result


def main():
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    rows=[]
    for p in sorted(OUT.rglob('daily_account.csv')):
        daily=pd.read_csv(p,parse_dates=['date'])
        orders=pd.read_csv(p.with_name('orders.csv'),parse_dates=['date'])
        checks=replay(daily,orders,market)
        checks.to_csv(p.with_name('independent_audit.csv'),index=False)
        rows.append({'account':str(p.parent.relative_to(OUT)), 'days':len(daily),
                     'max_nav_residual_cny':checks.nav_residual.abs().max(),
                     'max_cash_residual_cny':checks.cash_residual.abs().max(),
                     'max_return_residual':checks.return_residual.abs().max()})
    result=pd.DataFrame(rows)
    result.to_csv(OUT/'independent_account_checks.csv',index=False)
    print('Reconciled',len(result),'accounts; max residual CNY',result.max_nav_residual_cny.max())


if __name__=='__main__':main()
