"""Posthoc component ablation with fixed economic directions, all trials saved."""
import json
import pickle

import pandas as pd

from execution_v22 import run_account
from research_v23_signals import ROOT,SOURCE,OUT,PORTFOLIO,annual_metrics,centered_rank


def main():
    folder=OUT/'trend_ablation';folder.mkdir(parents=True,exist_ok=True)
    frame=pd.read_pickle(OUT/'family_panel_private.pkl')
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    predictions=[]
    for date,g in frame.groupby('date',sort=True):
        if date<pd.Timestamp('2016-12-01'):continue
        eq=centered_rank(g.equal8)
        short=centered_rank(g.mom_120);long=centered_rank(g.mom_252_skip20)
        scores={'T1_Mom120':short,'T2_Mom252Skip20':long,
                'T3_Equal8Mom120':(eq+short)/2,'T4_Equal8Mom252':(eq+long)/2}
        for model,score in scores.items():
            sub=g[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
            sub['model'],sub['score']=model,score
            predictions.append(sub)
    predictions=pd.concat(predictions,ignore_index=True)
    predictions.to_pickle(folder/'predictions_private.pkl')
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
