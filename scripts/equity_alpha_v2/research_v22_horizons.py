"""Read-only 5/20/60 trading-day factor diagnostics; never train on these labels."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from factor_research_v191 import bh_fdr
from research_v16_improvements import FEATURES

HERE=Path(__file__).resolve().parent
SOURCE=HERE/'outputs/v16_improvements'
OUT=HERE/'outputs/v22_execution_reality'
HORIZONS=(5,20,60)


def attach_horizon_return(panel,market,horizon):
    """Entry is the next open; exit is exactly N observed trading sessions later."""
    frame=panel.copy()
    entry=frame.entry_date.map(market.date_index).to_numpy()
    symbol=frame.instrument.map(market.symbol_index).to_numpy()
    if pd.isna(entry).any() or pd.isna(symbol).any():
        raise ValueError('Market index missing an input key')
    entry=entry.astype(int); symbol=symbol.astype(int)
    exit_index=entry+horizon
    valid=exit_index<len(market.dates)
    start=np.full(len(frame),np.nan)
    finish=np.full(len(frame),np.nan)
    start[valid]=market.fields['open'][entry[valid],symbol[valid]]
    finish[valid]=market.fields['open'][exit_index[valid],symbol[valid]]
    frame['horizon_return']=np.where((start>0)&(finish>0),finish/start-1,np.nan)
    frame['horizon_exit']=pd.NaT
    frame.loc[valid,'horizon_exit']=market.dates[exit_index[valid]]
    return frame


def monthly_rows(frame,horizon):
    rows=[]
    for date,group in frame.groupby('date',sort=True):
        year=date.year
        if year<2017:continue
        period=('development_2017_2020' if year<=2020 else
                'reused_validation_2021_2023' if year<=2023 else 'opened_2024_2026')
        for factor in FEATURES:
            x=pd.to_numeric(group[factor],errors='coerce').replace([np.inf,-np.inf],np.nan)
            y=group.horizon_return
            valid=x.notna()&y.notna()
            xx=x[valid];yy=y[valid]
            rec={'date':date,'period':period,'horizon_days':horizon,'factor':factor,
                 'n_stocks':len(xx),'rank_ic':np.nan,'top_bottom_gross':np.nan,
                 'five_groups_complete':False}
            if len(xx)>=30 and xx.nunique()>1 and yy.nunique()>1:
                rec['rank_ic']=float(xx.corr(yy,method='spearman'))
                buckets=pd.qcut(xx.rank(method='first'),5,labels=False)
                means=yy.groupby(buckets).mean()
                counts=yy.groupby(buckets).size()
                if len(means)==5 and counts.min()>=5:
                    rec['top_bottom_gross']=float(means.iloc[-1]-means.iloc[0])
                    rec['five_groups_complete']=True
            rows.append(rec)
    return pd.DataFrame(rows)


def summarize(monthly):
    rows=[]
    for (period,horizon,factor),group in monthly.groupby(['period','horizon_days','factor']):
        x=group.rank_ic.dropna().to_numpy(float)
        row={'period':period,'horizon_days':horizon,'factor':factor,'valid_months':len(x),
             'mean_stocks':float(group.n_stocks.mean()),
             'mean_rank_ic':float(x.mean()) if len(x) else np.nan,
             'icir_unannualized':float(x.mean()/x.std(ddof=1)) if len(x)>1 and x.std(ddof=1)>0 else np.nan,
             'positive_ic_share':float((x>0).mean()) if len(x) else np.nan,
             'mean_top_bottom_gross':float(group.top_bottom_gross.mean()),
             'complete_quintile_months':int(group.five_groups_complete.sum()),
             'hac_p':np.nan,'ci_low':np.nan,'ci_high':np.nan}
        if len(x)>=12:
            # 60-day monthly labels overlap; lag 3 is predeclared for all horizons.
            fit=sm.OLS(x,np.ones((len(x),1))).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            row.update(hac_p=float(fit.pvalues[0]),ci_low=float(fit.conf_int()[0,0]),
                       ci_high=float(fit.conf_int()[0,1]))
        rows.append(row)
    result=pd.DataFrame(rows)
    result['bh_q_28']=np.nan
    for _,part in result.groupby(['period','horizon_days']):
        result.loc[part.index,'bh_q_28']=bh_fdr(part.hac_p.to_numpy(float))
    return result


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    panel=pd.read_pickle(SOURCE/'panel_final.pkl')
    for col in ('date','entry_date','exit_date'):panel[col]=pd.to_datetime(panel[col])
    all_rows=[]
    for horizon in HORIZONS:
        labelled=attach_horizon_return(panel,market,horizon)
        monthly=monthly_rows(labelled,horizon)
        all_rows.append(monthly)
        print('horizon',horizon,'months',monthly.date.nunique(),flush=True)
    monthly=pd.concat(all_rows,ignore_index=True)
    monthly.to_csv(OUT/'horizon_monthly_factor_diagnostics.csv',index=False)
    results=summarize(monthly)
    results.to_csv(OUT/'horizon_factor_summary.csv',index=False)
    print(results.loc[results.factor.isin(('mom_20','mom_60','reversal_5','vol_20')),
                      ['period','horizon_days','factor','valid_months','mean_rank_ic',
                       'mean_top_bottom_gross','hac_p','bh_q_28']].to_string(index=False))


if __name__=='__main__':main()
