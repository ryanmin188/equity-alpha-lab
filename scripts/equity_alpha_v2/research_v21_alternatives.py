"""Predeclared V2.1 alternative-alpha research from saved, already-opened data.

All historical outputs are exploratory. No frozen V1.6 signal, account, or
protocol is modified. Missing point-in-time data is a hard gate, not imputed.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from execution_v16 import run_account

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'outputs' / 'v16_improvements'
OUT = HERE / 'outputs' / 'v21_alternatives'
FACTOR_FILE = HERE / 'outputs' / 'v18_attribution' / 'CH4_factors_daily_202608_source.csv'
BASE = ['mom_20','mom_60','reversal_5','vol_20','vol_60','volume_surprise','range_20','ma_gap_20']
SIGNS = np.array([1,1,1,-1,-1,-1,-1,1]) / 8
STATE = ['market_trend_60','market_vol_20','mom_trend_interaction','lowvol_vol_interaction']
CONFIG = json.loads((ROOT / 'configs' / 'v16_protocol.json').read_text(encoding='utf-8'))
PROTOCOL = json.loads((ROOT / 'configs' / 'v21_research_protocol.json').read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal8_scores(frame):
    return frame[BASE].fillna(0).to_numpy(float) @ SIGNS


def state_at_signal(market, date):
    i = market.date_index[pd.Timestamp(date)]
    if i < 60:
        raise ValueError('Insufficient prior benchmark observations')
    p = market.benchmark_close
    if not np.isfinite(p[i-60:i+1]).all() or np.min(p[i-60:i+1]) <= 0:
        raise ValueError('Incomplete prior benchmark prices')
    r = p[i-19:i+1] / p[i-20:i] - 1
    return float(p[i]/p[i-60]-1), float(np.std(r,ddof=1)*np.sqrt(252))


def attach_states(panel, market):
    out = panel.copy()
    states = {date:state_at_signal(market,date) for date in out.date.unique()}
    out['market_trend_60'] = out.date.map(lambda date:states[date][0])
    out['market_vol_20'] = out.date.map(lambda date:states[date][1])
    out['mom_trend_interaction'] = out.mom_60 * out.market_trend_60
    out['lowvol_vol_interaction'] = -out.vol_60 * out.market_vol_20
    return out


def mature_train(panel, date):
    earliest = date - pd.DateOffset(months=PROTOCOL['common']['training_lookback_months'])
    train = panel[(panel.date >= earliest) & (panel.date < date) & (panel.exit_date <= date)]
    if train.date.nunique() < PROTOCOL['common']['minimum_mature_months']:
        return None
    if not (train.exit_date <= date).all():
        raise RuntimeError('Future label in training sample')
    return train.copy()


def rank_target(frame, label):
    return frame.groupby('date')[label].rank(pct=True) - .5


def fit_b_models(train, today):
    train = train.dropna(subset=['forward_return']).copy()
    target = rank_target(train,'forward_return')
    baseline_train = train.groupby('date',sort=False).apply(
        lambda g:pd.Series(equal8_scores(g),index=g.index).rank(pct=True)-.5,
        include_groups=False).droplevel(0).reindex(train.index)
    baseline_today = pd.Series(equal8_scores(today),index=today.index).rank(pct=True)-.5
    features = ['mom_trend_interaction','lowvol_vol_interaction']
    linear = make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                           StandardScaler(),Ridge(alpha=100.))
    linear.fit(train[features],target-baseline_train)
    b1 = baseline_today.to_numpy() + linear.predict(today[features])
    tree_features = BASE + STATE
    parameters = {**CONFIG['tree_parameters'],'verbosity':-1,'deterministic':True,
                  'force_col_wise':True}
    tree = lgb.LGBMRegressor(**parameters)
    tree.fit(train[tree_features],target)
    b2 = tree.predict(today[tree_features])
    return b1,b2


def predict_b(panel):
    blocks = []
    for date,today in panel.groupby('date',sort=True):
        if date < pd.Timestamp('2017-01-01'):
            continue
        train = mature_train(panel,date)
        if train is None:
            continue
        b1,b2 = fit_b_models(train,today)
        for model,score in [('B1_state_linear',b1),('B2_state_tree',b2)]:
            block = today[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
            block['score'] = score
            block['model'] = model
            blocks.append(block)
    return pd.concat(blocks,ignore_index=True)


def monthly_ic(predictions,label='forward_return'):
    rows=[]
    for (model,date),group in predictions.groupby(['model','date']):
        good=group[['score',label]].dropna()
        ic=float(spearmanr(good.score,good[label]).statistic) if len(good)>=20 else np.nan
        rows.append({'model':model,'date':date,'n':len(good),'rank_ic':ic})
    return pd.DataFrame(rows)


def period_of(date):
    year=pd.Timestamp(date).year
    if 2017<=year<=2020:return 'development_2017_2020'
    if 2021<=year<=2023:return 'reused_validation_2021_2023'
    if 2024<=year<=2025:return 'opened_2024_2025'
    if year==2026:return 'opened_2026_partial'
    return 'historical_training'


def ic_summary(monthly):
    x=monthly.copy()
    x['period']=x.date.map(period_of)
    x['year']=x.date.dt.year
    rows=[]
    for (model,period),group in x.groupby(['model','period']):
        v=group.rank_ic.dropna()
        rows.append({'model':model,'period':period,'months':len(v),'mean_rank_ic':float(v.mean()),
                     'icir_unannualized':float(v.mean()/v.std(ddof=1)) if len(v)>1 and v.std(ddof=1)>0 else np.nan,
                     'positive_fraction':float((v>0).mean()) if len(v) else np.nan,
                     'yearly_mean_ic':json.dumps({str(y):round(float(g.rank_ic.mean()),6)
                                                   for y,g in group.groupby('year')},ensure_ascii=False)})
    return pd.DataFrame(rows)


def load_inputs():
    with (SOURCE/'market_final.pkl').open('rb') as file:
        market,_=pickle.load(file)
    panel=pd.read_pickle(SOURCE/'panel_final.pkl')
    for col in ('date','entry_date','exit_date'):
        panel[col]=pd.to_datetime(panel[col])
    if panel.duplicated(['date','instrument']).any() or not ((panel.date<panel.entry_date)&
                                                            (panel.entry_date<panel.exit_date)).all():
        raise ValueError('Invalid panel key or time boundary')
    return panel,market


def load_ch4(market):
    required=['rf_dly','mktrf','SMB','VMG','PMO']
    factor=pd.read_csv(FACTOR_FILE,dtype={'date':str})
    if set(required)-set(factor) or factor.date.duplicated().any():
        raise ValueError('CH-4 source schema or unique dates failed')
    factor.date=pd.to_datetime(factor.date,format='%Y%m%d')
    factor=factor.set_index('date').sort_index().reindex(market.dates)
    return factor[required].to_numpy(float)


def residual_labels(panel,market,factors):
    """Daily close-to-close residual labels; factor exposure freezes at signal."""
    output=panel[['date','instrument','entry_date','exit_date']].copy()
    output['residual_label']=np.nan
    output['beta_market']=np.nan
    output['beta_smb']=np.nan
    output['beta_vmg']=np.nan
    output['beta_pmo']=np.nan
    output['paired_history_days']=0
    prices=market.fields['close']
    stock_returns=np.full_like(prices,np.nan)
    stock_returns[1:]=prices[1:]/prices[:-1]-1
    stock_returns[~np.isfinite(stock_returns)]=np.nan
    for k,(date,today) in enumerate(panel.groupby('date',sort=True)):
        if date>pd.Timestamp('2025-12-31'):
            continue
        i=market.date_index[date]
        end=market.date_index[pd.Timestamp(today.exit_date.iloc[0])]-1
        if end<=i:
            raise ValueError('Residual label must follow signal close')
        future=factors[i+1:end+1]
        if not np.isfinite(future).all():
            continue  # no interpolation of absent CH-4 days
        start=max(1,i-251)
        exposure_factors=factors[start:i+1]
        for row in today.itertuples():
            j=market.symbol_index[row.instrument]
            y=stock_returns[start:i+1,j]
            good=np.isfinite(y)&np.isfinite(exposure_factors).all(axis=1)
            output.at[row.Index,'paired_history_days']=int(good.sum())
            if good.sum()<126:
                continue
            design=np.column_stack((np.ones(good.sum()),exposure_factors[good,1:]))
            beta=np.linalg.lstsq(design,y[good]-exposure_factors[good,0],rcond=None)[0][1:]
            output.loc[row.Index,['beta_market','beta_smb','beta_vmg','beta_pmo']]=beta
            realized=stock_returns[i+1:end+1,j]
            if not np.isfinite(realized).all():
                continue
            modeled=future[:,0]+future[:,1:]@beta
            output.at[row.Index,'residual_label']=float(np.sum(realized-modeled))
        if k%24==0:
            print('A2 residual labels through',date.date(),flush=True)
    return output


def predict_a2(panel,residual):
    joined=panel.merge(residual[['date','instrument','residual_label']],
                       on=['date','instrument'],validate='one_to_one')
    blocks=[]
    for date,today in joined.groupby('date',sort=True):
        if not (pd.Timestamp('2017-01-01')<=date<=pd.Timestamp('2025-12-31')):
            continue
        train=mature_train(joined,date)
        if train is None:
            continue
        train=train.dropna(subset=['residual_label'])
        if train.date.nunique()<PROTOCOL['common']['minimum_mature_months']:
            continue
        target=rank_target(train,'residual_label')
        model=make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                            StandardScaler(),Ridge(alpha=100.))
        model.fit(train[BASE],target)
        block=today[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60','residual_label']].copy()
        block['score']=model.predict(today[BASE])
        block['model']='A2_CH4_residual_ridge'
        blocks.append(block)
    return pd.concat(blocks,ignore_index=True)


def run_a(panel,market):
    factors=load_ch4(market)
    residual=residual_labels(panel,market,factors)
    OUT.mkdir(parents=True,exist_ok=True)
    residual.to_csv(OUT/'A_residual_labels_private.csv',index=False)
    a2=predict_a2(panel,residual)
    a2.to_csv(OUT/'A_predictions_private.csv',index=False)
    a0=panel[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
    a0['score']=equal8_scores(panel)
    a0['model']='A0_Equal8_reference'
    a0=a0.merge(residual[['date','instrument','residual_label']],on=['date','instrument'],validate='one_to_one')
    combined=pd.concat([a0,a2],ignore_index=True)
    residual_ic=monthly_ic(combined,'residual_label')
    raw_ic=monthly_ic(combined,'forward_return')
    residual_ic.to_csv(OUT/'A_residual_monthly_ic.csv',index=False)
    raw_ic.to_csv(OUT/'A_raw_monthly_ic.csv',index=False)
    ic_summary(residual_ic).to_csv(OUT/'A_residual_ic_summary.csv',index=False)
    ic_summary(raw_ic).to_csv(OUT/'A_raw_ic_summary.csv',index=False)
    portfolio={'name':'Buffer30_60','top_n':30,'buffer_n':60,'weighting':'equal'}
    comparisons=[]
    for name,start,end in [('development_2017_2020','2017-01-01','2020-08-03'),
                           ('reused_validation_2021_2023','2021-01-01','2023-12-01'),
                           ('opened_2024_2025_signals','2024-01-01','2026-02-02')]:
        subset=a2[(a2.entry_date>=start)&(a2.exit_date<=end)]
        if subset.empty:continue
        summary,daily,orders,targets=run_account(subset,market,portfolio,CONFIG)
        detail=OUT/'A2_CH4_residual_ridge'/name
        detail.mkdir(parents=True,exist_ok=True)
        daily.to_csv(detail/'daily_account.csv',index=False)
        orders.to_csv(detail/'orders.csv',index=False)
        targets.to_csv(detail/'targets.csv',index=False)
        comparisons.append({'model':'A2_CH4_residual_ridge','period':name,**summary})
    pd.DataFrame(comparisons).to_csv(OUT/'A_account_comparison.csv',index=False)
    return ic_summary(residual_ic),ic_summary(raw_ic),pd.DataFrame(comparisons)


def run_b(panel,market):
    scored=attach_states(panel,market)
    predictions=predict_b(scored)
    reference=scored[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
    reference['score']=equal8_scores(scored)
    reference['model']='B0_Equal8_reference'
    ic=monthly_ic(pd.concat([predictions,reference],ignore_index=True))
    OUT.mkdir(parents=True,exist_ok=True)
    ic.to_csv(OUT/'B_monthly_ic.csv',index=False)
    ic_summary(ic).to_csv(OUT/'B_ic_summary.csv',index=False)
    predictions.to_csv(OUT/'B_predictions_private.csv',index=False)
    portfolio={'name':'Buffer30_60','top_n':30,'buffer_n':60,'weighting':'equal'}
    comparisons=[]
    for model,group in predictions.groupby('model'):
        for name,start,end in [('development_2017_2020','2017-01-01','2020-08-03'),
                               ('reused_validation_2021_2023','2021-01-01','2023-12-01'),
                               ('opened_2024_2026','2024-01-01','2026-10-08')]:
            subset=group[(group.entry_date>=start)&(group.exit_date<=end)]
            if subset.empty:continue
            summary,daily,orders,targets=run_account(subset,market,portfolio,CONFIG)
            detail=OUT/model/name
            detail.mkdir(parents=True,exist_ok=True)
            daily.to_csv(detail/'daily_account.csv',index=False)
            orders.to_csv(detail/'orders.csv',index=False)
            targets.to_csv(detail/'targets.csv',index=False)
            comparisons.append({'model':model,'period':name,**summary})
    pd.DataFrame(comparisons).to_csv(OUT/'B_account_comparison.csv',index=False)
    return ic_summary(ic),pd.DataFrame(comparisons)


def main():
    panel,market=load_inputs()
    a_residual_ic,a_raw_ic,a_accounts=run_a(panel,market)
    b_ic,b_accounts=run_b(panel,market)
    (OUT/'provenance.json').write_text(json.dumps({'status':'OPENED_HISTORICAL_EXPLORATION',
        'input_sha256':{str(p.relative_to(ROOT)):digest(p) for p in
            (SOURCE/'panel_final.pkl',SOURCE/'market_final.pkl',FACTOR_FILE,
             ROOT/'configs/v21_research_protocol.json')},
        'research_code_sha256':digest(__file__),
        'execution_v16_sha256':digest(HERE/'execution_v16.py'),
        'independent_future_months':0},ensure_ascii=False,indent=2),encoding='utf-8')
    print('A residual label IC:',a_residual_ic.to_string(index=False),flush=True)
    print('A raw return IC:',a_raw_ic.to_string(index=False),flush=True)
    print(a_accounts[['model','period','total_return','information_ratio']].to_string(index=False),flush=True)
    print(b_ic.to_string(index=False),flush=True)
    print(b_accounts[['model','period','total_return','information_ratio']].to_string(index=False),flush=True)


if __name__=='__main__':
    main()
