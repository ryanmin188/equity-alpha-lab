"""Selected Qlib-style OHLCV families, causal monthly models, fixed parameters."""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge

from execution_v22 import run_account
from research_v16_improvements import FEATURES
from research_v23_signals import ROOT, SOURCE, OUT, PORTFOLIO, annual_metrics

WINDOWS=(5,10,20,60)


def price_volume_features(close, op, high, low, volume):
    """Inputs end at the signal close. No filling across absent observations."""
    c,o,h,l,v=[pd.DataFrame(a,dtype=float) for a in (close,op,high,low,volume)]
    if len(c)<61 or any(a.shape!=c.shape for a in (o,h,l,v)):
        raise ValueError('At least 61 aligned past bars required')
    last=c.iloc[-1].replace(0,np.nan)
    spread=(h.iloc[-1]-l.iloc[-1]).replace(0,np.nan)
    raw={
        'k_body':(last-o.iloc[-1])/o.iloc[-1],
        'k_range':(h.iloc[-1]-l.iloc[-1])/o.iloc[-1],
        'k_body_ratio':(last-o.iloc[-1])/spread,
        'k_upper':(h.iloc[-1]-pd.concat([o.iloc[-1],last],axis=1).max(axis=1))/spread,
        'k_lower':(pd.concat([o.iloc[-1],last],axis=1).min(axis=1)-l.iloc[-1])/spread,
        'k_position':(2*last-h.iloc[-1]-l.iloc[-1])/spread,
    }
    ret=c.pct_change(fill_method=None)
    vchange=np.log(v.where(v>0)).diff()
    diff=c.diff()
    for w in WINDOWS:
        cw,vw=c.iloc[-w:],v.iloc[-w:]
        valid=cw.notna()
        t=np.arange(w,dtype=float)[:,None]
        x=cw.to_numpy(); mask=np.isfinite(x)
        count=mask.sum(axis=0)
        tm=(t*mask).sum(axis=0)/np.maximum(count,1)
        xm=np.nansum(x,axis=0)/np.maximum(count,1)
        tx=t-tm; xx=x-xm
        cross=np.nansum(tx*xx,axis=0)
        tt=(tx**2*mask).sum(axis=0)
        ss=np.nansum(xx**2,axis=0)
        slope=np.divide(cross,tt,out=np.zeros_like(cross),where=tt>0)
        rsq=np.divide(cross**2,tt*ss,out=np.full_like(cross,np.nan),where=tt*ss>1e-20)
        prefix=f'w{w}_'
        block={
            'roc':c.iloc[-w-1]/last,
            'ma':cw.mean()/last,
            'std':cw.std()/last,
            'slope':slope/last,
            'rsq':pd.Series(rsq),
            'resid':(last-(xm+slope*(w-1-tm)))/last,
            'rank':cw.le(last).sum()/valid.sum(),
            'rsv':(last-l.iloc[-w:].min())/(h.iloc[-w:].max()-l.iloc[-w:].min()),
            'corr':cw.corrwith(np.log(vw.where(vw>0))),
            'cord':ret.iloc[-w:].corrwith(vchange.iloc[-w:]),
            'positive':ret.iloc[-w:].gt(0).where(ret.iloc[-w:].notna()).mean(),
            'sum_up':diff.iloc[-w:].clip(lower=0).sum(min_count=max(2,int(.8*w)))/diff.iloc[-w:].abs().sum(),
            'vma':vw.mean()/v.iloc[-1],
            'vstd':vw.std()/v.iloc[-1],
            'weighted_vol':(ret.iloc[-w:].abs()*vw).std()/(ret.iloc[-w:].abs()*vw).mean(),
        }
        for name,value in block.items():
            raw[prefix+name]=pd.Series(value).where(count>=max(3,int(.8*w)))
    result=pd.DataFrame(raw).replace([np.inf,-np.inf],np.nan)
    return result


def build_features(panel,market):
    frames=[]
    for date,g in panel.groupby('date',sort=True):
        i=market.date_index[pd.Timestamp(date)]
        if i<60:raise ValueError('Insufficient history')
        js=[market.symbol_index[s] for s in g.instrument]
        arrays=[market.fields[k][i-60:i+1,js] for k in ('close','open','high','low','volume')]
        values=price_volume_features(*arrays)
        # Cross-sectional transforms contain neither future labels nor future dates.
        ranks=values.rank(pct=True)*2-1
        frame=g.reset_index(drop=True).copy()
        for name in ranks:frame['ms_'+name]=ranks[name].to_numpy()
        frames.append(frame)
    return pd.concat(frames,ignore_index=True)


def causal_train(frame,date):
    past=frame[(frame.date<date)&(frame.exit_date<=date)]
    mature_dates=past.groupby('date').exit_date.max()
    # Reject partial months if some rows have not yet matured.
    all_exits=frame.groupby('date').exit_date.max()
    dates=mature_dates.index[all_exits.reindex(mature_dates.index)<=date][-60:]
    if len(dates)<36:raise ValueError('Insufficient mature training months')
    return past[past.date.isin(dates)].dropna(subset=['forward_return'])


def predict(frame,config):
    features=FEATURES+[c for c in frame if c.startswith('ms_')]
    rows,audit=[],[]
    for date,today in frame.groupby('date',sort=True):
        if date<pd.Timestamp('2016-12-01'):continue
        train=causal_train(frame,date)
        x=train[features].fillna(0)
        y=train.groupby('date').forward_return.rank(pct=True)-.5
        xt=today[features].fillna(0)
        ridge=Ridge(alpha=100.).fit(x,y)
        tree=lgb.LGBMRegressor(**config['tree_parameters'],verbosity=-1,deterministic=True,force_col_wise=True).fit(x,y)
        for name,estimator in [('M1_MultiscaleRidge',ridge),('M2_MultiscaleTree',tree)]:
            part=today[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
            part['model']=name
            part['score']=estimator.predict(xt)
            rows.append(part)
        audit.append({'date':date,'max_train_exit':train.exit_date.max(),
                      'train_months':train.date.nunique(),'n_features':len(features)})
        print('multiscale',date.date(),len(features),'features',flush=True)
    return pd.concat(rows,ignore_index=True),pd.DataFrame(audit)


def main():
    folder=OUT/'multiscale'
    folder.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'market_final.pkl').open('rb') as f:market,_=pickle.load(f)
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    panel=pd.read_pickle(SOURCE/'panel_final.pkl')
    frame=build_features(panel,market)
    frame.to_pickle(folder/'panel_private.pkl')
    predictions,audit=predict(frame,config)
    predictions.to_pickle(folder/'predictions_private.pkl')
    audit.to_csv(folder/'training_audit.csv',index=False)
    rows=[]
    for period,start,end in [('development_2017_2020','2017-01-01','2021-02-01'),
                             ('reused_validation_2021_2023','2021-01-01','2024-02-01'),
                             ('opened_2024_2026','2024-01-01','2026-10-08')]:
        for model in predictions.model.unique():
            sub=predictions[(predictions.model==model)&(predictions.entry_date>=start)&(predictions.exit_date<=end)]
            scenarios=[('base',5.,1e6)]+([('slip15',15.,1e6),('capital10m',5.,1e7)] if period=='opened_2024_2026' else [])
            for scenario,slip,capital in scenarios:
                dest=folder/period/model/scenario
                dest.mkdir(parents=True,exist_ok=True)
                summary,daily,orders,targets,exposures=run_account(sub,market,PORTFOLIO,config,regime='E2',slippage_bps=slip,capital=capital)
                for name,data in [('daily_account',daily),('orders',orders),('targets',targets),('exposures',exposures)]:
                    data.to_csv(dest/f'{name}.csv',index=False)
                annual=annual_metrics(daily)
                annual['model'],annual['period'],annual['scenario']=model,period,scenario
                rows.append(annual)
                print(period,model,scenario,annual[['year','strategy_return','active_pp']].round(4).to_dict('records'),flush=True)
    pd.concat(rows,ignore_index=True).to_csv(folder/'annual.csv',index=False)


if __name__=='__main__':main()
