"""Observable market gates and an unconditional cyclical risk-premium control."""
import json
import pickle

import numpy as np
import pandas as pd

from execution_v22 import run_account
from research_v23_signals import ROOT,SOURCE,OUT,PORTFOLIO,annual_metrics,centered_rank


def past_beta(stock,benchmark):
    stock,benchmark=np.asarray(stock,float),np.asarray(benchmark,float)
    if stock.shape!=(252,) or benchmark.shape!=(252,):raise ValueError('252 past returns required')
    ok=np.isfinite(stock)&np.isfinite(benchmark)
    if ok.sum()<202:return np.nan
    x=benchmark[ok]-benchmark[ok].mean()
    if x@x<1e-15:return np.nan
    return float(x@(stock[ok]-stock[ok].mean())/(x@x))


def market_gates(past_close):
    x=np.asarray(past_close,float)
    if len(x)!=253 or not np.isfinite(x).all() or (x<=0).any():
        raise ValueError('253 valid observed benchmark closes required')
    return bool(x[-1]>x[-200:].mean()),bool(x[-1]>x[0])


def predict(frame,market):
    ret=pd.DataFrame(market.fields['close']).pct_change(fill_method=None).to_numpy()
    bm=pd.Series(market.benchmark_close).pct_change(fill_method=None).to_numpy()
    rows,audits=[],[]
    for date,g in frame.groupby('date',sort=True):
        if date<pd.Timestamp('2016-12-01'):continue
        i=market.date_index[date]
        betas=[past_beta(ret[i-251:i+1,market.symbol_index[s]],bm[i-251:i+1]) for s in g.instrument]
        cyclical=.5*(centered_rank(betas)+centered_rank(g.reversal_5))
        ma,roc=market_gates(market.benchmark_close[i-252:i+1])
        scores={'R0_Cyclical':cyclical,'R1_MA200':cyclical if ma else g.equal8.to_numpy(),
                'R2_ROC252':cyclical if roc else g.equal8.to_numpy()}
        audits.append({'date':date,'known_through':market.dates[i],
                       'ma200_risk_on':ma,'roc252_risk_on':roc,'beta_coverage':np.isfinite(betas).mean()})
        for model,score in scores.items():
            sub=g[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
            sub['model'],sub['score']=model,score
            rows.append(sub)
    return pd.concat(rows,ignore_index=True),pd.DataFrame(audits)


def main():
    folder=OUT/'market_gate';folder.mkdir(parents=True,exist_ok=True)
    frame=pd.read_pickle(OUT/'family_panel_private.pkl')
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    predictions,audit=predict(frame,market)
    predictions.to_pickle(folder/'predictions_private.pkl')
    audit.to_csv(folder/'gate_audit.csv',index=False)
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    rows=[]
    for period,start,end in [('development_2017_2020','2017-01-01','2021-02-01'),
                             ('reused_validation_2021_2023','2021-01-01','2024-02-01'),
                             ('opened_2024_2026','2024-01-01','2026-10-08')]:
        for model in predictions.model.unique():
            sub=predictions[(predictions.model==model)&(predictions.entry_date>=start)&(predictions.exit_date<=end)]
            scenarios=[('base',5.,1e6)]+([('slip15',15.,1e6),('capital10m',5.,1e7)] if period=='opened_2024_2026' else [])
            for scenario,slip,capital in scenarios:
                dest=folder/period/model/scenario;dest.mkdir(parents=True,exist_ok=True)
                _,daily,orders,targets,exposures=run_account(sub,market,PORTFOLIO,config,regime='E2',slippage_bps=slip,capital=capital)
                for name,data in [('daily_account',daily),('orders',orders),('targets',targets),('exposures',exposures)]:
                    data.to_csv(dest/f'{name}.csv',index=False)
                annual=annual_metrics(daily)
                annual['model'],annual['period'],annual['scenario']=model,period,scenario
                rows.append(annual)
                print(period,model,scenario,annual[['year','strategy_return','active_pp']].round(4).to_dict('records'),flush=True)
    pd.concat(rows,ignore_index=True).to_csv(folder/'annual.csv',index=False)


if __name__=='__main__':main()
