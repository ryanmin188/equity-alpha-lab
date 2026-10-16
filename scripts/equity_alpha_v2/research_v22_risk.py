"""Fixed, causal covariance forecast comparison on deterministic stock baskets."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'outputs/v16_improvements'
OUT=HERE/'outputs/v22_execution_reality'
LOOKBACK=252
FORWARD=20
N_STOCKS=10


def covariance_forecasts(history):
    x=np.asarray(history,float)
    if x.shape[0]<LOOKBACK or x.shape[1]!=N_STOCKS or not np.isfinite(x).all():
        raise ValueError('Expected 252 finite observations for ten stocks')
    sample=np.cov(x,rowvar=False)
    diagonal=np.diag(np.diag(sample))
    shrink=LedoitWolf().fit(x).covariance_
    estimates={'sample':sample,'diagonal':diagonal,'ledoit_wolf':shrink}
    for matrix in estimates.values():
        if not np.allclose(matrix,matrix.T,atol=1e-12) or np.linalg.eigvalsh(matrix).min()<-1e-12:
            raise AssertionError('Invalid forecast covariance')
    return estimates


def evaluate(panel,market):
    close=market.fields['close']
    returns=close[1:]/close[:-1]-1
    rows=[];skipped=0
    for date,group in panel.groupby('date',sort=True):
        if date.year<2017:continue
        i=market.date_index[date]
        if i<LOOKBACK or i+FORWARD>=len(market.dates):continue
        candidates=sorted(group.instrument)
        selected=[]
        for symbol in candidates:
            s=market.symbol_index[symbol]
            history=returns[i-LOOKBACK:i,s]
            if np.isfinite(history).all():
                selected.append(s)
            if len(selected)==N_STOCKS:break
        if len(selected)<N_STOCKS:
            skipped+=1;continue
        # The basket is selected entirely at signal close. Missing future marks
        # make its evaluation unavailable; replacement by a future-known name is forbidden.
        future=returns[i:i+FORWARD,selected]
        if not np.isfinite(future).all():
            skipped+=1;continue
        past=returns[i-LOOKBACK:i,selected]
        w=np.full(N_STOCKS,1/N_STOCKS)
        realized=float(np.mean((future@w)**2))
        period=('development_2017_2020' if date.year<=2020 else
                'reused_validation_2021_2023' if date.year<=2023 else 'opened_2024_2026')
        for name,cov in covariance_forecasts(past).items():
            predicted=float(w@cov@w)
            rows.append({'date':date,'period':period,'model':name,
                         'predicted_daily_variance':predicted,
                         'realized_next20_daily_variance':realized,
                         'squared_error':(predicted-realized)**2,
                         'absolute_error':abs(predicted-realized),
                         'selected_symbols':','.join(market.symbols[s] for s in selected)})
    return pd.DataFrame(rows),skipped


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    panel=pd.read_pickle(SOURCE/'panel_final.pkl')
    panel.date=pd.to_datetime(panel.date)
    result,skipped=evaluate(panel,market)
    result.to_csv(OUT/'risk_forecast_monthly.csv',index=False)
    summary=result.groupby(['period','model']).agg(n_months=('date','count'),
        mean_squared_error=('squared_error','mean'),mean_absolute_error=('absolute_error','mean'),
        mean_predicted_variance=('predicted_daily_variance','mean'),
        mean_realized_variance=('realized_next20_daily_variance','mean')).reset_index()
    summary['calibration_ratio']=summary.mean_realized_variance/summary.mean_predicted_variance
    summary.to_csv(OUT/'risk_forecast_summary.csv',index=False)
    print('skipped_dates',skipped)
    print(summary.to_string(index=False))


if __name__=='__main__':main()
