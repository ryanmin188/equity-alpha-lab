"""Recorded posthoc portfolio sensitivity, without changing the E2 engine."""
import json
import pickle

import pandas as pd

from execution_v22 import run_account
from research_v23_signals import ROOT, SOURCE, OUT, MODELS, annual_metrics


def main():
    with (SOURCE/'market_final.pkl').open('rb') as f:
        market,_=pickle.load(f)
    predictions=pd.read_pickle(OUT/'predictions_private.pkl')
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    protocol=json.loads((ROOT/'configs/v23_portfolio_followup.json').read_text(encoding='utf-8'))
    rows=[]
    for period,start,end in [('reused_validation_2021_2023','2021-01-01','2024-02-01'),
                             ('opened_2024_2026','2024-01-01','2026-10-08')]:
        for model in MODELS:
            subset=predictions[(predictions.model==model)&(predictions.entry_date>=start)&(predictions.exit_date<=end)]
            for portfolio in protocol['portfolios']:
                folder=OUT/'portfolio_followup'/period/model/portfolio['name']
                folder.mkdir(parents=True,exist_ok=True)
                summary,daily,orders,targets,exposures=run_account(subset,market,portfolio,config,regime='E2')
                for name,data in [('daily_account',daily),('orders',orders),('targets',targets),('exposures',exposures)]:
                    data.to_csv(folder/f'{name}.csv',index=False)
                annual=annual_metrics(daily)
                annual['model'],annual['period'],annual['portfolio']=model,period,portfolio['name']
                rows.append(annual)
                print(period,model,portfolio['name'],annual[['year','strategy_return','active_pp']].round(4).to_dict('records'),flush=True)
    pd.concat(rows,ignore_index=True).to_csv(OUT/'portfolio_followup_annual.csv',index=False)


if __name__=='__main__':main()
