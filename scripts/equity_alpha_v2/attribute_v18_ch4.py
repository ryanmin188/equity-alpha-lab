"""Post-test A-share CH-4 return regression, requiring validated external CSV.

Input is the original Mingshi daily CH4 workbook, or its untouched CSV export.
Workbook itself is never committed to the repository.
Only overlapping ordinary close-to-close account intervals enter estimation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACCOUNT = HERE/'outputs/v16_improvements/final/Equal8__Buffer30_60/daily_account.csv'
PRIVATE = HERE/'outputs/v18_attribution'
PUBLIC = ROOT/'reports/summary'
FIELDS = ['rf_dly','mktrf','SMB','VMG','PMO']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_aligned(path):
    f=pd.read_excel(path,sheet_name='Returnseries') if path.suffix.lower()=='.xlsx' else pd.read_csv(path)
    if set(['date',*FIELDS]) - set(f.columns):
        raise ValueError('CH4 file must have date, rf_dly, mktrf, SMB, VMG, PMO')
    if f.date.astype(str).str.fullmatch(r'\d{8}').all():
        f.date=pd.to_datetime(f.date.astype(str),format='%Y%m%d')
    else:
        f.date=pd.to_datetime(f.date,errors='raise')
    assert f.date.is_unique and f.date.is_monotonic_increasing
    assert f[FIELDS].notna().all().all()
    assert (f[FIELDS].abs()<.5).all().all(), 'factor values should be decimal daily returns, not percentages'
    daily=pd.read_csv(ACCOUNT,parse_dates=['date'])
    assert daily.date.is_unique and daily.date.is_monotonic_increasing
    # Drop the first open-to-close construction day and terminal open mark.
    ordinary=daily.iloc[1:].loc[daily.iloc[1:].valuation.eq('close'),['date','net','benchmark_wealth']]
    start,end=ordinary.date.min(),min(ordinary.date.max(),f.date.max())
    ordinary=ordinary.loc[ordinary.date.between(start,end)]
    factor=f.loc[f.date.between(start,end),['date',*FIELDS]]
    joined=ordinary.merge(f,on='date',how='left',validate='one_to_one',indicator=True)
    missing=joined.loc[joined._merge.ne('both'),'date'].dt.strftime('%Y-%m-%d').tolist()
    if any(d<'2026-01-01' for d in missing):
        raise ValueError(f'CH4 missing primary 2024-2025 account dates: {missing}')
    joined.attrs['missing_factor_dates']=missing
    joined=joined.loc[joined._merge.eq('both')].drop(columns='_merge').copy()
    joined['portfolio_excess']=joined.net-joined.rf_dly
    joined['year']=joined.date.dt.year
    return joined


def fit_row(frame,period,model,fields):
    y=frame.portfolio_excess.to_numpy()
    x=sm.add_constant(frame[fields].to_numpy(),has_constant='add')
    fit=sm.OLS(y,x).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    ci=fit.conf_int(alpha=.05)
    row=dict(period=period,model=model,start=frame.date.min().date(),end=frame.date.max().date(),
             observations=len(frame),alpha_daily=fit.params[0],alpha_linear_annual=fit.params[0]*252,
             alpha_annual_ci_low=ci[0,0]*252,alpha_annual_ci_high=ci[0,1]*252,
             alpha_hac_p=fit.pvalues[0],r_squared=fit.rsquared,
             mean_portfolio_excess_daily=y.mean())
    for name,beta,p in zip(fields,fit.params[1:],fit.pvalues[1:]):
        row['beta_'+name]=beta
        row['p_'+name]=p
        row['mean_'+name+'_component_daily']=beta*frame[name].mean()
    return row


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ch4-file','--ch4-csv',dest='ch4_file',required=True,type=Path)
    parser.add_argument('--source-xlsx-sha256',help='Required for converted CSV to preserve original-workbook provenance')
    args=parser.parse_args()
    if not args.ch4_file.exists():
        parser.error('external CH4 file not found')
    if args.ch4_file.suffix.lower() not in ('.xlsx','.csv'):
        parser.error('CH4 source must be original .xlsx or unchanged .csv export')
    source_sha=digest(args.ch4_file) if args.ch4_file.suffix.lower()=='.xlsx' else args.source_xlsx_sha256
    if not source_sha:
        parser.error('--source-xlsx-sha256 is required for a converted CSV')
    aligned=load_aligned(args.ch4_file)
    rows=[]
    samples=[('2024-2025_primary',aligned.loc[aligned.year.isin([2024,2025])]),
             ('through_2026-08_available',aligned)]
    samples += [(str(y),g) for y,g in aligned.groupby('year')]
    for period,frame in samples:
        for model,fields in [('CH_market',['mktrf']),('CH4',['mktrf','SMB','VMG','PMO'])]:
            rows.append(fit_row(frame,period,model,fields))
    result=pd.DataFrame(rows)
    PRIVATE.mkdir(parents=True,exist_ok=True)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    aligned.to_csv(PRIVATE/'ch4_aligned_private.csv',index=False,float_format='%.12g',lineterminator='\n')
    result.to_csv(PUBLIC/'v18_ch4_regression.csv',index=False,float_format='%.12g',lineterminator='\n')
    manifest={'original_workbook_sha256':source_sha.lower(),
              'input_file_sha256':digest(args.ch4_file),
              'account_sha256':digest(ACCOUNT),
              'source_page':'https://en.mingshiim.com/database',
              'input_first_date':str((pd.read_excel(args.ch4_file,sheet_name='Returnseries',usecols=['date'],nrows=1) if args.ch4_file.suffix.lower()=='.xlsx' else pd.read_csv(args.ch4_file,usecols=['date'],nrows=1)).date.iloc[0]),
              'first_aligned_date':str(aligned.date.min().date()),
              'last_aligned_date':str(aligned.date.max().date()),
              'overlap_days':len(aligned),'excluded_boundary':'initial open-close and terminal close-open',
              'missing_factor_dates':aligned.attrs.get('missing_factor_dates',[]),
              'risk_free_and_factors_unit':'daily decimal returns, four decimal places in original workbook',
              'models':'same aligned sample, CH market-only and CH4, HAC(5); descriptive post-test'}
    (PRIVATE/'ch4_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(result.to_string(index=False))
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
