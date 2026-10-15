"""Frozen, staged CSI300 research. Read README and configs/v16_protocol.json.

Stages: develop (<=2023 data only), final (requires frozen choice).
No direct source copying: technical feature families and turnover buffering
draw on Qlib's public documentation; ranking uses LightGBM's public API.
"""
from __future__ import annotations
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
from pathlib import Path
import argparse
import hashlib
import json
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import qlib
from qlib.data import D
from execution_v16 import Market, run_account

ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/v16_improvements'
PROTOCOL=ROOT/'configs/v16_protocol.json'
BASE=['mom_20','mom_60','reversal_5','vol_20','vol_60','volume_surprise','range_20','ma_gap_20']
EXTRA=['mom_120','mom_252_skip20','mom_60_skip5','vol_ratio','downside_vol60','max_up20',
       'max_down20','drawdown120','price_position60','skew60','volume_ratio5_20',
       'log_dollar_volume','intraday20','overnight20','return_volume_corr20',
       'positive_days60','trend60','ma_gap60','vol5','amihud60']
FEATURES=BASE+EXTRA


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path,value):
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding='utf-8')


def load_market(end,stage):
    cache=OUT/f'market_{stage}.pkl'
    if cache.exists():
        with cache.open('rb') as f:
            return pickle.load(f)
    qlib.init(provider_uri=str(Path.home()/'.qlib/qlib_data/cn_data'),region='cn',kernels=1)
    dates=pd.DatetimeIndex(D.calendar(start_time='2010-01-01',end_time=end,freq='day'))
    intervals=D.list_instruments(D.instruments('csi300'),start_time=dates[0],end_time=dates[-1],as_list=False)
    intervals={str(s).upper():[(pd.Timestamp(a),pd.Timestamp(b)) for a,b in spans] for s,spans in intervals.items()}
    symbols=sorted(intervals)
    fields=['open','close','high','low','volume','factor']
    raw=D.features(symbols,['$'+f for f in fields],start_time=dates[0],end_time=dates[-1],freq='day')
    raw.index=raw.index.set_levels(raw.index.levels[0].str.upper(),level='instrument')
    arrays={f:raw['$'+f].unstack('instrument').reindex(index=dates,columns=symbols).to_numpy(dtype=float,copy=True) for f in fields}
    for f,a in arrays.items():
        a[~np.isfinite(a)|(a<=0)]=np.nan
    bench=D.features(['SH000300'],['$open','$close'],start_time=dates[0],end_time=dates[-1],freq='day')
    bench=bench.droplevel('instrument').reindex(dates)
    market=Market(dates,symbols,arrays,bench['$open'].to_numpy(),bench['$close'].to_numpy())
    result=(market,intervals)
    with cache.open('wb') as f:
        pickle.dump(result,f,protocol=pickle.HIGHEST_PROTOCOL)
    print(f'Market {stage}: {len(dates)} days, {len(symbols)} historical stocks; max date {dates[-1].date()}',flush=True)
    return result


def make_panel(market,intervals,stage):
    cache=OUT/f'panel_{stage}.pkl'
    if cache.exists():
        return pd.read_pickle(cache)
    ends=pd.Series(market.dates,index=market.dates).groupby(market.dates.to_period('M')).last()
    signal_dates=[d for d in ends if market.date_index[d]+1<len(market.dates)]
    blocks=[]
    for date,next_date in zip(signal_dates[:-1],signal_dates[1:]):
        i=market.date_index[date]
        if i<252 or date<pd.Timestamp('2012-01-01'):
            continue
        members=[s for s,spans in intervals.items() if any(a<=date<=b for a,b in spans)]
        js=[market.symbol_index[s] for s in members]
        hist={f:pd.DataFrame(a[i-252:i+1,js],columns=members) for f,a in market.fields.items()}
        c,o,h,l,v=[hist[f] for f in ('close','open','high','low','volume')]
        ret=c.pct_change(fill_method=None)
        elig=(c.iloc[-1]>0)&(v.iloc[-1]>0)&(c.iloc[-252:].notna().sum()>=202)&(hist['factor'].iloc[-1]>0)
        x=pd.DataFrame(index=pd.Index(members,name='instrument'))
        for window in (20,60,120):
            x[f'mom_{window}']=c.iloc[-1]/c.iloc[-window-1]-1
        x['reversal_5']=1-c.iloc[-1]/c.iloc[-6]
        for window in (20,60):
            x[f'vol_{window}']=ret.iloc[-window:].std()
        x['volume_surprise']=np.log(v.iloc[-1]/v.iloc[-21:-1].mean())
        x['range_20']=((h-l)/c).iloc[-20:].mean()
        x['ma_gap_20']=c.iloc[-1]/c.iloc[-20:].mean()-1
        x['mom_252_skip20']=c.iloc[-21]/c.iloc[-253]-1
        x['mom_60_skip5']=c.iloc[-6]/c.iloc[-61]-1
        x['vol_ratio']=x.vol_20/x.vol_60
        x['downside_vol60']=ret.clip(upper=0).iloc[-60:].pow(2).mean().pow(.5)
        x['max_up20']=ret.iloc[-20:].max()
        x['max_down20']=ret.iloc[-20:].min()
        x['drawdown120']=c.iloc[-1]/c.iloc[-120:].max()-1
        x['price_position60']=(c.iloc[-1]-l.iloc[-60:].min())/(h.iloc[-60:].max()-l.iloc[-60:].min())
        x['skew60']=ret.iloc[-60:].skew()
        x['volume_ratio5_20']=v.iloc[-5:].mean()/v.iloc[-20:].mean()
        x['log_dollar_volume']=np.log((c*v).iloc[-20:].mean())
        x['intraday20']=np.log(c/o).iloc[-20:].sum(min_count=16)
        x['overnight20']=np.log(o/c.shift(1)).iloc[-20:].sum(min_count=16)
        x['return_volume_corr20']=ret.iloc[-20:].corrwith(np.log(v/v.shift(1)).iloc[-20:])
        x['positive_days60']=ret.iloc[-60:].gt(0).where(ret.iloc[-60:].notna()).mean()
        times=np.arange(60,dtype=float); times-=times.mean()
        logs=np.log(c.iloc[-60:])
        x['trend60']=logs.mul(times,axis=0).sum(min_count=48)/(times@times)
        x['ma_gap60']=c.iloc[-1]/c.iloc[-60:].mean()-1
        x['vol5']=ret.iloc[-5:].std()
        x['amihud60']=np.log((ret.abs()/(c*v)).iloc[-60:].mean())
        x=x.replace([np.inf,-np.inf],np.nan)
        elig &= x.notna().mean(axis=1)>=.8
        x=x.loc[elig].copy()
        if len(x)<150:
            continue
        raw_vol=x.vol_60.copy()
        # Cross-sectional ranks are observable at signal time and robust to scale.
        x=(x.rank(pct=True)-.5)*2
        x['raw_vol_60']=raw_vol
        entry_i=i+1
        exit_i=market.date_index[next_date]+1
        instrument_indices=[market.symbol_index[s] for s in x.index]
        start=market.fields['open'][entry_i,instrument_indices]
        finish=market.fields['open'][exit_i,instrument_indices]
        x['forward_return']=finish/start-1
        x['date']=date
        x['entry_date']=market.dates[entry_i]
        x['exit_date']=market.dates[exit_i]
        blocks.append(x.reset_index())
    panel=pd.concat(blocks,ignore_index=True)
    panel.to_pickle(cache)
    return panel


def ridge_scores(train,today,features,config):
    model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),StandardScaler(),Ridge(alpha=config['ridge_alpha']))
    model.fit(train[features],train.target)
    return model.predict(today[features])


def score_month(train,today,config,needed):
    scores={}
    if 'Equal8' in needed:
        scores['Equal8']=today[BASE].fillna(0).to_numpy()@np.array([1,1,1,-1,-1,-1,-1,1])/8
    if 'MomentumQuality' in needed:
        names=['mom_120','mom_252_skip20','mom_60_skip5','vol_60','amihud60']
        scores['MomentumQuality']=today[names].fillna(0).to_numpy()@np.array([1,1,1,-1,-1])/5
    train=train.dropna(subset=['forward_return']).sort_values(['date','instrument']).copy()
    train['target']=train.groupby('date').forward_return.rank(pct=True)-.5
    if 'Ridge8Rank' in needed:
        scores['Ridge8Rank']=ridge_scores(train,today,BASE,config)
    if {'RidgeExtendedRank','EnsembleRank'} & needed:
        scores['RidgeExtendedRank']=ridge_scores(train,today,FEATURES,config)
    if 'PrunedRidgeRank' in needed:
        corr=train[FEATURES].corr().abs()
        selected=[]
        for name in FEATURES:
            if all(corr.loc[name,old]<.85 or pd.isna(corr.loc[name,old]) for old in selected):
                selected.append(name)
        scores['PrunedRidgeRank']=ridge_scores(train,today,selected,config)
    parameters={**config['tree_parameters'],'verbosity':-1,'deterministic':True,'force_col_wise':True}
    if {'LGBRankTarget','EnsembleRank'} & needed:
        model=lgb.LGBMRegressor(**parameters)
        model.fit(train[FEATURES],train.target)
        scores['LGBRankTarget']=model.predict(today[FEATURES])
    if 'LambdaRank' in needed:
        groups=train.groupby('date',sort=True).size().to_numpy()
        labels=np.minimum(((train.target+.5)*5).astype(int),4)
        model=lgb.LGBMRanker(**parameters,objective='lambdarank',label_gain=[0,1,2,3,4])
        model.fit(train[FEATURES],labels,group=groups)
        scores['LambdaRank']=model.predict(today[FEATURES])
    if 'EnsembleRank' in needed:
        scores['EnsembleRank']=(pd.Series(scores['RidgeExtendedRank']).rank(pct=True).to_numpy()+pd.Series(scores['LGBRankTarget']).rank(pct=True).to_numpy())/2
    return {k:np.asarray(v,float) for k,v in scores.items() if k in needed}


def predict(panel,config,needed,entry_start,stage):
    blocks=[]
    timings=[]
    for date,today in panel.groupby('date',sort=True):
        if today.entry_date.iloc[0]<pd.Timestamp(entry_start):
            continue
        earliest=date-pd.DateOffset(months=config['training_lookback_months'])
        train=panel[(panel.date>=earliest)&(panel.date<date)&(panel.exit_date<=date)]
        if train.date.nunique()<config['minimum_training_months']:
            continue
        assert train.exit_date.max()<=date
        now=time.time()
        scores=score_month(train,today,config,set(needed))
        for model,values in scores.items():
            part=today[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
            part['score']=values
            part['model']=model
            if not np.isfinite(values).all():
                raise ValueError(f'Non-finite prediction: {model}/{date}')
            blocks.append(part)
        timings.append({'date':date,'max_training_exit':train.exit_date.max(),'train_months':train.date.nunique(),'seconds':time.time()-now})
        print(f'{stage} {date.date()} trained={train.date.nunique()} months; {len(today)} stocks',flush=True)
    result=pd.concat(blocks,ignore_index=True)
    result.to_csv(OUT/f'predictions_{stage}.csv',index=False)
    pd.DataFrame(timings).to_csv(OUT/f'training_audit_{stage}.csv',index=False)
    return result


def interval(predictions,start,end):
    return predictions[(predictions.entry_date>=pd.Timestamp(start))&(predictions.exit_date<=pd.Timestamp(end))]


def evaluate(predictions,market,config,models,portfolios,start,end,stage):
    rows=[]
    for model in models:
        p=interval(predictions[predictions.model==model],start,end)
        for portfolio in portfolios:
            summary,daily,orders,targets=run_account(p,market,portfolio,config)
            name=model+'__'+portfolio['name']
            summary.update(candidate=name,model=model,portfolio=portfolio['name'],stage=stage)
            rows.append(summary)
            folder=OUT/stage/name
            folder.mkdir(parents=True,exist_ok=True)
            daily.to_csv(folder/'daily_account.csv',index=False)
            orders.to_csv(folder/'orders.csv',index=False)
            targets.to_csv(folder/'targets.csv',index=False)
            print(stage,name,'return',round(summary['total_return'],4),'benchmark',round(summary['benchmark_return'],4),flush=True)
    table=pd.DataFrame(rows)
    table.to_csv(OUT/f'{stage}_summary.csv',index=False)
    return table


def develop(config):
    frozen_path=OUT/'frozen_selection.json'
    opened=frozen_path.exists() and json.loads(frozen_path.read_text(encoding='utf-8')).get('final_opened',False)
    if opened or (OUT/'final_summary.csv').exists():
        raise RuntimeError('Final test already opened: create a new protocol/version to make changes')
    market,membership=load_market(config['validation_exit_end'],'develop')
    panel=make_panel(market,membership,'develop')
    preds=predict(panel,config,config['models'],config['development_entry_start'],'develop')
    evaluate(preds,market,config,config['models'],config['portfolios'],config['development_entry_start'],config['development_exit_end'],'development')
    validation=evaluate(preds,market,config,config['models'],config['portfolios'],config['validation_entry_start'],config['validation_exit_end'],'validation')
    feasible=validation[validation.max_drawdown>=-.4]
    if feasible.empty:
        feasible=validation
    winner=feasible.sort_values(['information_ratio','average_monthly_turnover','candidate'],ascending=[False,True,True]).iloc[0]
    frozen={'protocol_sha256':digest(PROTOCOL),'research_code_sha256':digest(__file__),
            'execution_code_sha256':digest(HERE/'execution_v16.py'),'selected':winner.to_dict(),
            'final_opened':False,'selection_rule':config['selection_rule'],
            'created_utc':pd.Timestamp.now(tz='UTC').isoformat(),
            'predeclared_baselines':['Equal8','Ridge8Rank']}
    write_json(OUT/'frozen_selection.json',frozen)
    print('FROZEN',winner.candidate,flush=True)


def final(config):
    frozen=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    for key,path in [('protocol_sha256',PROTOCOL),('research_code_sha256',__file__),('execution_code_sha256',HERE/'execution_v16.py')]:
        if digest(path)!=frozen[key]:
            raise RuntimeError('Code or protocol changed since selection: '+key)
    if frozen['final_opened'] or (OUT/'final_summary.csv').exists():
        raise RuntimeError('Final already opened; results cannot be used for another selection')
    frozen['final_opened']=True
    frozen['final_opened_utc']=pd.Timestamp.now(tz='UTC').isoformat()
    write_json(OUT/'frozen_selection.json',frozen)
    market,membership=load_market(config['final_data_end'],'final')
    panel=make_panel(market,membership,'final')
    model=frozen['selected']['model']
    models=list(dict.fromkeys([model,*frozen['predeclared_baselines']]))
    preds=predict(panel,config,models,config['final_entry_start'],'final')
    portfolio=next(p for p in config['portfolios'] if p['name']==frozen['selected']['portfolio'])
    evaluate(preds,market,config,models,[portfolio],config['final_entry_start'],config['final_data_end'],'final')
    p=interval(preds[preds.model==model],config['final_entry_start'],config['final_data_end'])
    stresses=[]
    for capital in config['stress_capital_cny']:
        for slip in config['stress_slippage_bps']:
            stats,_,_,_=run_account(p,market,portfolio,config,capital=capital,slippage_bps=slip)
            stresses.append(stats)
    pd.DataFrame(stresses).to_csv(OUT/'final_execution_stress.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['develop','final'],required=True)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    (develop if args.stage=='develop' else final)(config)


if __name__=='__main__':
    main()
