"""Read-only factor diagnostics on the saved V1.6 monthly panel.

The 28 stored features are ALREADY within-month percentile ranks from V1.6.
This module never rebuilds a trading account or changes the frozen strategy.
All per-security observations stay in ignored outputs; public files aggregate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PANEL = HERE / 'outputs/v16_improvements/panel_final.pkl'
PRIVATE = HERE / 'outputs/v191_factor_research'
PUBLIC = ROOT / 'reports/summary'
FEATURES = ['mom_20','mom_60','reversal_5','vol_20','vol_60','volume_surprise',
            'range_20','ma_gap_20','mom_120','mom_252_skip20','mom_60_skip5',
            'vol_ratio','downside_vol60','max_up20','max_down20','drawdown120',
            'price_position60','skew60','volume_ratio5_20','log_dollar_volume',
            'intraday20','overnight20','return_volume_corr20','positive_days60',
            'trend60','ma_gap60','vol5','amihud60']
BASE_DIRECTION = dict(zip(FEATURES[:8], [1,1,1,-1,-1,-1,-1,1]))
MIN_STOCKS = 100
MIN_MONTHS_HAC = 6


def period_of(date: pd.Timestamp) -> str:
    if date < pd.Timestamp('2021-01-01'):
        return 'development_2012_2020'
    if date < pd.Timestamp('2024-01-01'):
        return 'validation_2021_2023'
    return 'opened_final_2024_2026'


def validate_panel(panel: pd.DataFrame, decision_date: pd.Timestamp | None = None,
                   training: bool = False) -> pd.DataFrame:
    need = {'date','entry_date','exit_date','instrument','forward_return',*FEATURES}
    if need - set(panel):
        raise ValueError(f'Missing panel columns: {sorted(need-set(panel))}')
    frame = panel.copy()
    for col in ('date','entry_date','exit_date'):
        frame[col] = pd.to_datetime(frame[col], errors='raise')
    if frame[['date','entry_date','exit_date']].isna().any().any():
        raise ValueError('Missing date boundary')
    if frame.duplicated(['date','instrument']).any():
        raise ValueError('Duplicate security and signal month')
    if not (frame.date.lt(frame.entry_date) & frame.entry_date.lt(frame.exit_date)).all():
        raise ValueError('Labels must enter after signal and exit after entry')
    if decision_date is not None:
        decision_date = pd.Timestamp(decision_date)
        if training:
            if not (frame.date.lt(decision_date) & frame.exit_date.le(decision_date)).all():
                raise ValueError('Unmatured label or overlapping training/evaluation boundary')
        elif not frame.date.ge(decision_date).all():
            raise ValueError('Evaluation includes pre-boundary training rows')
    return frame


def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1:
        raise ValueError('pvalues must be 1D')
    answer = np.full(p.shape, np.nan)
    valid = np.isfinite(p) & (p >= 0) & (p <= 1)
    order = np.argsort(p[valid], kind='stable')
    ordered = p[valid][order]
    if len(ordered):
        # Preserve the predeclared family size even if an untestable factor
        # has no p-value; do not make the remaining tests easier to pass.
        adjusted = np.minimum.accumulate((ordered * len(p) /
                                          np.arange(1, len(ordered)+1))[::-1])[::-1]
        block = np.empty(len(ordered))
        block[order] = np.minimum(adjusted, 1.)
        answer[valid] = block
    return answer


def preprocess_cross_section(values: pd.Series, *, mad_multiplier: float = 3.0,
                             min_stocks: int = MIN_STOCKS) -> tuple[pd.DataFrame, dict]:
    """MAD, population-Z and percentile rank using one date's stored ranks."""
    x = pd.to_numeric(values, errors='coerce').replace([np.inf,-np.inf], np.nan).astype(float)
    ok = x.notna()
    n, total = int(ok.sum()), len(x)
    result = pd.DataFrame(index=x.index, data={'stored_raw': x, 'winsorized': np.nan,
                                                'zscore': np.nan, 'rank': np.nan})
    info = {'n_total': total, 'n_valid': n, 'coverage': n/total if total else 0.,
            'missing_rate': 1-n/total if total else 1.,
            'pre_mean': float(x.mean()) if n else np.nan,
            'pre_std': float(x.std(ddof=0)) if n else np.nan,
            'pre_min': float(x.min()) if n else np.nan,
            'pre_max': float(x.max()) if n else np.nan,
            'median': np.nan, 'mad': np.nan, 'lower': np.nan, 'upper': np.nan,
            'post_mean': np.nan, 'post_std': np.nan, 'post_min': np.nan,
            'post_max': np.nan, 'clipped_count': 0, 'status': 'ok'}
    if n == 0:
        info['status'] = 'all_missing'
        return result, info
    med = float(x.median())
    mad = float((x-med).abs().median())
    info.update(median=med, mad=mad)
    if n < min_stocks:
        info['status'] = 'insufficient_stocks'
        return result, info
    if x[ok].nunique() < 2 or mad == 0:
        info['status'] = 'constant_or_zero_mad'
        return result, info
    width = mad_multiplier * 1.4826 * mad
    lower, upper = med-width, med+width
    clipped = x.clip(lower, upper)
    stdev = float(clipped.std(ddof=0))
    if not np.isfinite(stdev) or stdev <= 1e-12:
        info['status'] = 'constant_after_clip'
        return result, info
    result['winsorized'] = clipped
    result['zscore'] = (clipped-clipped.mean())/stdev
    result['rank'] = clipped.rank(pct=True, method='average')
    info.update(lower=lower, upper=upper,
                clipped_count=int(((x<lower)|(x>upper)).sum()),
                post_mean=float(clipped.mean()), post_std=stdev,
                post_min=float(clipped.min()), post_max=float(clipped.max()))
    return result, info


def monthly_factor_metrics(x: pd.Series, y: pd.Series, *, min_stocks=MIN_STOCKS) -> dict:
    pair = pd.concat([x.rename('x'), y.rename('y')], axis=1).replace([np.inf,-np.inf], np.nan).dropna()
    n = len(pair)
    if n < min_stocks or pair.x.nunique() < 2 or pair.y.nunique() < 2:
        return {'n_stocks': n, 'pearson_ic': np.nan, 'rank_ic': np.nan,
                'status': 'insufficient_or_constant'}
    return {'n_stocks': n, 'pearson_ic': float(pair.x.corr(pair.y)),
            'rank_ic': float(spearmanr(pair.x, pair.y).statistic), 'status': 'ok'}


def quantile_rows(x: pd.Series, y: pd.Series, n_groups: int, *, min_stocks=MIN_STOCKS) -> tuple[pd.DataFrame, dict]:
    pair = pd.concat([x.rename('x'), y.rename('y')], axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    n = len(pair)
    if n < max(min_stocks, n_groups*5) or pair.x.nunique() < n_groups:
        return pd.DataFrame(), {'n_stocks':n, 'complete':False, 'status':'insufficient_or_tied'}
    # Stable input index is instrument; ties are resolved by instrument, never return.
    pair = pair.sort_index().sort_values('x', kind='stable')
    pair['bucket'] = np.floor(np.arange(n)*n_groups/n).astype(int)+1
    groups = pair.groupby('bucket', as_index=False).agg(mean_forward_return=('y','mean'),
                                                        n_stocks=('y','size'))
    complete = len(groups)==n_groups and groups.n_stocks.min() >= 5
    means = groups.mean_forward_return.to_numpy()
    summary = {'n_stocks': n, 'complete': bool(complete), 'status': 'ok' if complete else 'incomplete',
               'high_minus_low': float(means[-1]-means[0]) if complete else np.nan,
               'monotonic_spearman': float(spearmanr(np.arange(1,n_groups+1), means).statistic)
               if complete and np.unique(means).size > 1 else np.nan}
    return groups, summary


def summarize_ic(monthly: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for period in ['development_2012_2020','validation_2021_2023','opened_final_2024_2026']:
        for factor in FEATURES:
            sample = monthly.loc[(monthly.period==period)&(monthly.factor==factor)&monthly.rank_ic.notna()]
            vals = sample.rank_ic.to_numpy(float)
            row = {'period':period,'factor':factor,'n_months':len(vals),
                   'mean_stocks':sample.n_stocks.mean(),
                   'mean_rank_ic':np.mean(vals) if len(vals) else np.nan,
                   'sd_rank_ic':np.std(vals,ddof=1) if len(vals)>1 else np.nan,
                   'icir_monthly_unannualized':np.mean(vals)/np.std(vals,ddof=1)
                   if len(vals)>1 and np.std(vals,ddof=1)>0 else np.nan,
                   'positive_month_share':np.mean(vals>0) if len(vals) else np.nan,
                   'mean_pearson_ic':sample.pearson_ic.mean(),
                   'hac_p':np.nan,'hac_ci_low':np.nan,'hac_ci_high':np.nan}
            if len(vals)>=MIN_MONTHS_HAC:
                fit=sm.OLS(vals,np.ones((len(vals),1))).fit(cov_type='HAC',cov_kwds={'maxlags':3})
                row.update(hac_p=float(fit.pvalues[0]),hac_ci_low=float(fit.conf_int()[0,0]),
                           hac_ci_high=float(fit.conf_int()[0,1]))
            rows.append(row)
    out=pd.DataFrame(rows)
    out['bh_q_28']=np.nan
    for period, part in out.groupby('period'):
        out.loc[part.index,'bh_q_28']=bh_fdr(part.hac_p.to_numpy())
    return out


def summarize_quantiles(spreads: pd.DataFrame) -> pd.DataFrame:
    rows=[]
    for (period,factor,groups),g in spreads.groupby(['period','factor','n_groups']):
        valid=g.loc[g.complete]
        direction=BASE_DIRECTION.get(factor)
        rows.append({'period':period,'factor':factor,'n_groups':groups,'n_months':len(valid),
                     'missing_months':len(g)-len(valid),
                     'mean_high_minus_low':valid.high_minus_low.mean(),
                     'positive_spread_share':valid.high_minus_low.gt(0).mean() if len(valid) else np.nan,
                     'mean_monotonic_spearman':valid.monotonic_spearman.mean(),
                     'frozen_equal8_direction':direction,
                     'direction_adjusted_mean_spread':valid.high_minus_low.mean()*direction
                     if direction is not None else np.nan})
    return pd.DataFrame(rows)


def redundancy(panel: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,list[list[str]],pd.DataFrame]:
    matrices=[];records=[]
    for date,g in panel.groupby('date',sort=True):
        matrix=g[FEATURES].corr(method='spearman',min_periods=MIN_STOCKS)
        matrices.append(matrix)
        long=matrix.stack().rename('spearman').reset_index()
        long.columns=['factor_a','factor_b','spearman']
        long.insert(0,'date',date.date())
        records.append(long)
    stacked=pd.concat(matrices,keys=range(len(matrices)))
    mean_signed=stacked.groupby(level=1).mean()
    mean_abs=stacked.abs().groupby(level=1).mean()
    coverage=panel[FEATURES].notna().mean()
    pairs=[]
    adjacency={f:set() for f in FEATURES}
    for i,a in enumerate(FEATURES):
        for b in FEATURES[i+1:]:
            value=float(mean_abs.loc[a,b])
            if value>=.85:
                pairs.append({'factor_a':a,'factor_b':b,'mean_monthly_abs_spearman':value,
                              'coverage_a':coverage[a],'coverage_b':coverage[b]})
                adjacency[a].add(b);adjacency[b].add(a)
    groups=[]; seen=set()
    for f in FEATURES:
        if f in seen or not adjacency[f]:
            continue
        stack=[f]; component=[]
        while stack:
            item=stack.pop()
            if item in seen: continue
            seen.add(item);component.append(item);stack.extend(adjacency[item]-seen)
        groups.append(sorted(component))
    return (mean_signed,mean_abs,
            pd.DataFrame(pairs,columns=['factor_a','factor_b','mean_monthly_abs_spearman','coverage_a','coverage_b']),
            groups,pd.concat(records,ignore_index=True))


def save_csv(df:pd.DataFrame,path:Path,index=False) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(df.to_csv(index=index,float_format='%.12g',lineterminator='\n').encode())


def main() -> None:
    panel=validate_panel(pd.read_pickle(PANEL)) # trusted local cache created by V1.6
    # Panel already has only fully matured next-period labels through 2026-10-08.
    if panel.exit_date.max()>pd.Timestamp('2026-10-08'):
        raise ValueError('Panel has labels beyond saved account boundary')
    PRIVATE.mkdir(parents=True,exist_ok=True);PUBLIC.mkdir(parents=True,exist_ok=True)
    save_csv(panel[['date','entry_date','exit_date','instrument','forward_return',*FEATURES]],
             PRIVATE/'stored_rank_features_and_labels_private.csv')
    processed=[];parameters=[];ics=[];qrows=[];spreads=[]
    for date,group in panel.groupby('date',sort=True):
        group=group.set_index('instrument').sort_index()
        period=period_of(group.entry_date.iloc[0])
        block=pd.DataFrame(index=group.index)
        block['date']=date
        block['forward_return']=group.forward_return
        for factor in FEATURES:
            p,meta=preprocess_cross_section(group[factor])
            meta.update(date=date.date(),period=period,factor=factor)
            parameters.append(meta)
            block[factor]=p['rank']
            ic=monthly_factor_metrics(p['rank'],group.forward_return)
            ics.append({'date':date.date(),'period':period,'factor':factor,**ic})
            for buckets in (5,10):
                q,info=quantile_rows(p['rank'],group.forward_return,buckets)
                spreads.append({'date':date.date(),'period':period,'factor':factor,
                                'n_groups':buckets,**info})
                if len(q):
                    q.insert(0,'n_groups',buckets);q.insert(0,'factor',factor)
                    q.insert(0,'period',period);q.insert(0,'date',date.date())
                    qrows.append(q)
        processed.append(block.reset_index())
    save_csv(pd.concat(processed,ignore_index=True),PRIVATE/'processed_factor_ranks_private.csv')
    params=pd.DataFrame(parameters);monthly=pd.DataFrame(ics)
    save_csv(params,PRIVATE/'preprocessing_parameters_private.csv')
    save_csv(monthly,PRIVATE/'monthly_ic_private.csv')
    save_csv(pd.concat(qrows,ignore_index=True),PRIVATE/'monthly_quantile_groups_private.csv')
    spreads=pd.DataFrame(spreads)
    save_csv(spreads,PRIVATE/'monthly_quantile_spreads_private.csv')
    summary=summarize_ic(monthly)
    save_csv(summary,PUBLIC/'v191_factor_ic_summary.csv')
    save_csv(summarize_quantiles(spreads),PUBLIC/'v191_quantile_summary.csv')
    distribution=params.groupby(['period','factor'],as_index=False).agg(
        months=('date','size'),mean_coverage=('coverage','mean'),minimum_coverage=('coverage','min'),
        total_clipped=('clipped_count','sum'),failed_months=('status',lambda x:int(x.ne('ok').sum())))
    save_csv(distribution,PUBLIC/'v191_preprocessing_summary.csv')
    signed,corr,pairs,groups,monthly_corr=redundancy(panel)
    save_csv(monthly_corr,PRIVATE/'monthly_factor_correlation_private.csv')
    save_csv(signed,PUBLIC/'v191_mean_signed_factor_correlation.csv',index=True)
    save_csv(corr,PUBLIC/'v191_mean_abs_factor_correlation.csv',index=True)
    save_csv(pairs,PUBLIC/'v191_high_correlation_pairs.csv')
    (PUBLIC/'v191_redundancy_groups.json').write_bytes(json.dumps(
        {'threshold_mean_monthly_abs_spearman':.85,'connected_components':groups},
        indent=2,ensure_ascii=False).encode())
    manifest={'input_file_sha256':hashlib.sha256(PANEL.read_bytes()).hexdigest(),
              'input_rows':len(panel),'signal_months':panel.date.nunique(),
              'feature_count':len(FEATURES),'stored_input_units':'V1.6 within-month percentile ranks [-1,1]',
              'label':'next holding interval open-to-open, gross security return',
              'last_label_exit':str(panel.exit_date.max().date()),
              'preprocess':'within-month median +/- 3 * 1.4826 * MAD, population Z, percentile rank',
              'minimum_stocks':MIN_STOCKS,'hac_lag_months':3,
              'fdr':'BH across 28 features within each fixed period'}
    (PUBLIC/'v191_factor_manifest.json').write_bytes(json.dumps(manifest,indent=2,ensure_ascii=False).encode())
    print(json.dumps({'months':manifest['signal_months'],'rows':manifest['input_rows'],
                      'final_q_below_005':int(((summary.period=='opened_final_2024_2026')&
                                                (summary.bh_q_28<.05)).sum()),
                      'high_corr_pairs':len(pairs)},ensure_ascii=False))


if __name__=='__main__':
    main()
