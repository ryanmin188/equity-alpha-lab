"""Explicitly exploratory search; every trial saved, causality tested separately."""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from execution_v24 import run_account
from research_v23_signals import ROOT,SOURCE,centered_rank,annual_metrics
from audit_v23_accounts import replay

OUT=Path(__file__).resolve().parent/'outputs/v24_causal_search'
PORTFOLIOS=[{'name':'Buffer30_60','top_n':30,'buffer_n':60,'weighting':'equal'},
            {'name':'Buffer20_40','top_n':20,'buffer_n':40,'weighting':'equal'}]


def blend_scores(frame):
    parts=[]
    for date,g in frame.groupby('date',sort=True):
        eq=centered_rank(g.equal8)
        for factor in ('mom_120','mom_252_skip20'):
            trend=centered_rank(g[factor])
            for weight in (.25,.5,.75):
                p=g[['date','instrument','entry_date','exit_date','raw_vol_60','forward_return']].copy()
                p['model']=f'{factor}_eq{int(weight*100)}'
                p['score']=weight*eq+(1-weight)*trend
                parts.append(p)
    return pd.concat(parts,ignore_index=True)


def evaluate(predictions,market,portfolios,out,stress=False):
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    rows=[]
    periods=[('development_2017_2020','2017-01-01','2021-02-01'),
             ('reused_validation_2021_2023','2021-01-01','2024-02-01'),
             ('opened_2024_2026','2024-01-01','2026-10-08')]
    for period,start,end in periods:
        if stress and period!='opened_2024_2026':continue
        for model in predictions.model.unique():
            sub=predictions[(predictions.model==model)&(predictions.entry_date>=start)&(predictions.exit_date<=end)]
            for portfolio in portfolios:
                for scenario,slip,capital in ([('slip15',15,1e6),('capital10m',5,1e7)] if stress else [('base',5,1e6)]):
                    dest=out/period/model/portfolio['name']/scenario;dest.mkdir(parents=True,exist_ok=True)
                    summary,daily,orders,targets,exposures=run_account(sub,market,portfolio,config,slippage_bps=slip,capital=capital)
                    checks=replay(daily,orders,market)
                    for name,data in [('daily_account',daily),('orders',orders),('targets',targets),('exposures',exposures),('audit',checks)]:
                        data.to_csv(dest/f'{name}.csv',index=False)
                    (dest/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()),encoding='utf-8')
                    annual=annual_metrics(daily)
                    annual['model'],annual['portfolio'],annual['period'],annual['scenario']=model,portfolio['name'],period,scenario
                    rows.append(annual)
                    print(period,model,portfolio['name'],scenario,annual[['year','strategy_return','active_pp']].round(4).to_dict('records'),flush=True)
    result=pd.concat(rows,ignore_index=True)
    result.to_csv(out/('stress_annual.csv' if stress else 'base_annual.csv'),index=False)
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frame=pd.read_pickle(SOURCE.parent/'v23_signal_comparison/family_panel_private.pkl')
    predictions=blend_scores(frame)
    predictions.to_pickle(OUT/'monthly_predictions_private.pkl')
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    annual=evaluate(predictions,market,PORTFOLIOS,OUT)
    target=annual[(annual.period=='opened_2024_2026')&annual.year.isin([2024,2025])]
    passes=target.groupby(['model','portfolio']).active_pp.min()
    passes=passes[passes>0]
    passes.to_csv(OUT/'first_batch_passes.csv')
    print('DUAL YEAR PASSES',passes.to_dict(),flush=True)
    for model,portfolio in passes.index:
        evaluate(predictions[predictions.model==model],market,[p for p in PORTFOLIOS if p['name']==portfolio],
                 OUT/'stress'/model/portfolio,stress=True)


if __name__=='__main__':main()
