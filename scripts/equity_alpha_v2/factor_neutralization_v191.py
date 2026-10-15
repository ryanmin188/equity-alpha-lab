"""Point-in-time cross-sectional neutralization framework; no vendor data bundled.

Pure functions may be tested with synthetic inputs. Real evaluation is gated
on dated industry and market-cap observations with explicit provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
PRIVATE=HERE/'outputs/v191_factor_research'
PUBLIC=ROOT/'reports/summary'


def neutralize_cross_section(factor:pd.Series, *, market_cap:pd.Series|None=None,
                             industry:pd.Series|None=None, min_stocks:int=5,
                             missing:str='raise') -> tuple[pd.Series,dict]:
    """OLS residual, standardized in-sample, with deterministic rank handling.

    Uses no outcome labels. Missing exposures either raise or are excluded with
    a reported count; a singular column is dropped and named in diagnostics.
    """
    if missing not in ('raise','drop'):
        raise ValueError('missing must be raise or drop')
    y=pd.to_numeric(factor,errors='coerce').replace([np.inf,-np.inf],np.nan).astype(float)
    frame=pd.DataFrame({'factor':y})
    if market_cap is not None:
        frame['market_cap']=pd.to_numeric(market_cap.reindex(y.index),errors='coerce')
        frame['market_cap']=frame.market_cap.where(frame.market_cap.gt(0))
    if industry is not None:
        frame['industry']=industry.reindex(y.index).astype('string').replace('',pd.NA)
    if market_cap is None and industry is None:
        raise ValueError('At least one risk exposure is required')
    incomplete=frame.isna().any(axis=1)
    if incomplete.any() and missing=='raise':
        raise ValueError(f'{int(incomplete.sum())} rows have missing factors or risk exposures')
    usable=frame.loc[~incomplete].copy()
    if len(usable)<min_stocks:
        raise ValueError(f'Insufficient complete observations: {len(usable)}')
    x=pd.DataFrame({'intercept':np.ones(len(usable))},index=usable.index)
    notes=[]
    if market_cap is not None:
        logcap=np.log(usable.market_cap.to_numpy(float))
        if np.std(logcap)>1e-12:
            x['log_market_cap']=logcap
        else:
            notes.append('constant_log_market_cap_dropped')
    if industry is not None:
        categories=sorted(usable.industry.unique())
        if len(categories)==1:
            notes.append('single_industry_no_dummy')
        else:
            for category in categories[1:]:
                x['industry:'+str(category)]=usable.industry.eq(category).astype(float).to_numpy()
    # Keep independent columns in declared order. Do not use y or future data
    # to decide which exposure to retain.
    keep=[]; dropped=[]
    for column in x.columns:
        candidate=keep+[column]
        if np.linalg.matrix_rank(x[candidate].to_numpy(float))==len(candidate):
            keep.append(column)
        else:
            dropped.append(column)
    if dropped:
        notes.append('rank_deficient_columns_dropped')
    design=x[keep].to_numpy(float)
    beta=np.linalg.lstsq(design,usable.factor.to_numpy(float),rcond=None)[0]
    residual=usable.factor.to_numpy(float)-design@beta
    sd=float(np.std(residual,ddof=0))
    standardized=residual/sd if sd>1e-12 else np.zeros_like(residual)
    result=pd.Series(np.nan,index=y.index,name='neutralized_z',dtype=float)
    result.loc[usable.index]=standardized
    ortho=np.max(np.abs(design.T@residual)) if len(residual) else np.nan
    info={'n_input':len(frame),'n_used':len(usable),'n_missing_dropped':int(incomplete.sum()),
          'design_columns':keep,'dropped_columns':dropped,'notes':notes,
          'condition_number':float(np.linalg.cond(design)),
          'max_normal_equation_residual':float(ortho),
          'residual_std_before_z':sd,'status':'ok' if sd>1e-12 else 'constant_residual'}
    return result,info


def validate_historical_exposures(frame:pd.DataFrame, signal_dates:pd.DataFrame) -> pd.DataFrame:
    required={'date','instrument','industry','market_cap','industry_valid_from',
              'industry_valid_to','source_id','market_cap_asof','market_cap_unit'}
    if required-set(frame):
        raise ValueError(f'Missing exposure fields: {sorted(required-set(frame))}')
    x=frame.copy()
    for column in ('date','industry_valid_from','industry_valid_to'):
        x[column]=pd.to_datetime(x[column],errors='raise')
    if x[['date','industry_valid_from','industry_valid_to']].isna().any().any():
        raise ValueError('Missing exposure dates')
    if x.duplicated(['date','instrument']).any():
        raise ValueError('Duplicate exposure security-month')
    if x[['instrument','industry','source_id']].isna().any().any() or x.source_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing industry, instrument or source provenance')
    if x.market_cap_unit.isna().any() or x.market_cap_unit.astype(str).str.strip().eq('').any():
        raise ValueError('Missing market-cap unit')
    if x.market_cap_unit.nunique()!=1:
        raise ValueError('Mixed market-cap units require explicit conversion before regression')
    if not x.date.between(x.industry_valid_from,x.industry_valid_to).all():
        raise ValueError('Industry label is not effective at signal date')
    x['market_cap']=pd.to_numeric(x.market_cap,errors='raise')
    if not np.isfinite(x.market_cap).all() or x.market_cap.le(0).any():
        raise ValueError('Market cap must be positive finite in documented units')
    if 'market_cap_asof' not in x:
        raise ValueError('Additional market_cap_asof date is required to prove timing')
    x['market_cap_asof']=pd.to_datetime(x.market_cap_asof,errors='raise')
    if x.market_cap_asof.isna().any() or x.market_cap_asof.gt(x.date).any():
        raise ValueError('Market cap observed after signal date')
    expected=signal_dates[['date','instrument']].drop_duplicates()
    expected['date']=pd.to_datetime(expected.date)
    merged=expected.merge(x,on=['date','instrument'],how='left',validate='one_to_one',indicator=True)
    if merged._merge.ne('both').any():
        raise ValueError(f'Incomplete point-in-time exposure coverage: {int(merged._merge.ne("both").sum())} rows')
    return merged.drop(columns='_merge')


def synthetic_barra_risk_example() -> dict:
    """Illustrate XFX' + D and active risk; NOT a CNE5/CNE6 estimate."""
    x=np.array([[1.,-1.],[1.,0.],[1.,1.]])
    f=np.array([[.0004,.0001],[.0001,.0009]])
    d=np.diag([.0001,.00015,.0002])
    sigma=x@f@x.T+d
    portfolio=np.array([.5,.3,.2])
    benchmark=np.array([.3,.3,.4])
    active=portfolio-benchmark
    return {'X':x,'F':f,'D':d,'sigma':sigma,
            'portfolio_daily_vol':float(np.sqrt(portfolio@sigma@portfolio)),
            'active_daily_tracking_error':float(np.sqrt(active@sigma@active)),
            'symmetric':bool(np.allclose(sigma,sigma.T,atol=1e-12)),
            'minimum_eigenvalue':float(np.linalg.eigvalsh(sigma).min())}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--historical-exposures',type=Path,
                        help='Licensed dated industry/market-cap CSV; omit if unavailable')
    args=parser.parse_args()
    PUBLIC.mkdir(parents=True,exist_ok=True)
    if args.historical_exposures is None:
        state={'status':'DATA_NOT_AVAILABLE','real_neutralized_ic':'DATA_NOT_AVAILABLE',
               'real_neutralized_quantiles':'DATA_NOT_AVAILABLE',
               'required_columns':['date','instrument','industry','market_cap',
                                   'industry_valid_from','industry_valid_to','source_id',
                                   'market_cap_asof','market_cap_unit'],
               'note':'No historical industry or cap feed supplied; synthetic functions tested only'}
        (PUBLIC/'v191_neutralization_status.json').write_bytes(json.dumps(
            state,ensure_ascii=False,indent=2).encode())
        print(json.dumps(state,ensure_ascii=False))
        return
    from factor_research_v191 import (PANEL,FEATURES,period_of,validate_panel,
                                     monthly_factor_metrics,quantile_rows,
                                     summarize_ic,summarize_quantiles,save_csv)
    panel=validate_panel(pd.read_pickle(PANEL))
    exposures=validate_historical_exposures(pd.read_csv(args.historical_exposures,
                                                        dtype={'instrument':str}),panel)
    joined=panel.merge(exposures,on=['date','instrument'],how='inner',validate='one_to_one')
    if len(joined)!=len(panel):
        raise ValueError('Exposure merge changed the frozen panel row count')
    monthly=[];spreads=[];diagnostics=[]
    for date,group in joined.groupby('date',sort=True):
        group=group.set_index('instrument').sort_index()
        period=period_of(group.entry_date.iloc[0])
        for factor in FEATURES:
            original=group[factor]
            modes={'stored_rank':original}
            for label,industry in [('size_neutralized',None),
                                   ('industry_size_neutralized',group.industry)]:
                residual,meta=neutralize_cross_section(
                    original,market_cap=group.market_cap,industry=industry,
                    min_stocks=100,missing='drop')
                modes[label]=residual
                diagnostics.append({'date':date.date(),'period':period,'factor':factor,
                                    'mode':label,'n_used':meta['n_used'],
                                    'n_missing_dropped':meta['n_missing_dropped'],
                                    'condition_number':meta['condition_number'],
                                    'max_normal_equation_residual':meta['max_normal_equation_residual'],
                                    'dropped_columns':';'.join(meta['dropped_columns']),
                                    'status':meta['status']})
            for mode,x in modes.items():
                metric=monthly_factor_metrics(x,group.forward_return)
                monthly.append({'date':date.date(),'period':period,'factor':factor,
                                'mode':mode,**metric})
                for buckets in (5,10):
                    _,info=quantile_rows(x,group.forward_return,buckets)
                    spreads.append({'date':date.date(),'period':period,'factor':factor,
                                    'mode':mode,'n_groups':buckets,**info})
    PRIVATE.mkdir(parents=True,exist_ok=True)
    monthly=pd.DataFrame(monthly);spreads=pd.DataFrame(spreads)
    save_csv(monthly,PRIVATE/'neutralization_monthly_ic_private.csv')
    save_csv(spreads,PRIVATE/'neutralization_monthly_quantiles_private.csv')
    save_csv(pd.DataFrame(diagnostics),PRIVATE/'neutralization_diagnostics_private.csv')
    results=[];qresults=[]
    for mode,g in monthly.groupby('mode'):
        out=summarize_ic(g)
        out.insert(0,'mode',mode)
        results.append(out)
        q=summarize_quantiles(spreads.loc[spreads['mode']==mode])
        q.insert(0,'mode',mode)
        qresults.append(q)
    save_csv(pd.concat(results,ignore_index=True),PUBLIC/'v191_neutralized_ic_summary.csv')
    save_csv(pd.concat(qresults,ignore_index=True),PUBLIC/'v191_neutralized_quantile_summary.csv')
    state={'status':'EMPIRICALLY_EVALUATED',
           'exposure_file_sha256':hashlib.sha256(args.historical_exposures.read_bytes()).hexdigest(),
           'source_ids':sorted(exposures.source_id.unique().tolist()),
           'market_cap_units':sorted(exposures.market_cap_unit.unique().tolist()),
           'rows':len(exposures),'months':panel.date.nunique(),
           'caveat':'Data license, original effective dates, and accounting vintage still require independent source review'}
    (PUBLIC/'v191_neutralization_status.json').write_bytes(json.dumps(
        state,ensure_ascii=False,indent=2).encode())
    print(json.dumps(state,ensure_ascii=False))


if __name__=='__main__':
    main()
