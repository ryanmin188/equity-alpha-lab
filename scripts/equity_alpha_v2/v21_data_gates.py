"""Point-in-time gates and synthetic-only benchmark-relative optimizer.

No financial, sector or benchmark-weight rows are distributed with the project.
Passing unit tests proves contracts/math, not empirical availability.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd
from pathlib import Path
import json

FINANCIAL_COLUMNS={'instrument','period_end','announcement_date','available_at','source_id',
                   'EP','BP','ROE','profit_growth'}
BENCHMARK_COLUMNS={'date','instrument','weight','industry','industry_valid_from',
                   'industry_valid_to','available_at','source_id'}


def asof_financial(signals:pd.DataFrame, financials:pd.DataFrame):
    """Attach only filings known by each signal, rejecting undated or revised leaks."""
    if set(signals.columns)<{'date','instrument'} or set(financials.columns)<FINANCIAL_COLUMNS:
        raise ValueError('PIT financial source contract not met')
    if signals.duplicated(['date','instrument']).any():
        raise ValueError('Duplicate signal key')
    f=financials.copy()
    for col in ('period_end','announcement_date','available_at'):
        f[col]=pd.to_datetime(f[col],errors='raise')
    if f[list(FINANCIAL_COLUMNS-{'instrument','source_id','EP','BP','ROE','profit_growth'})].isna().any().any():
        raise ValueError('Missing required source time')
    if f.source_id.isna().any() or f.source_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing financial source id')
    if f.duplicated(['instrument','period_end','available_at']).any():
        raise ValueError('Ambiguous same-time financial revision')
    if not ((f.period_end<=f.announcement_date)&(f.announcement_date<=f.available_at)).all():
        raise ValueError('Financial observation/release chronology invalid')
    s=signals.copy()
    s['date']=pd.to_datetime(s.date)
    if s.date.isna().any():raise ValueError('Missing signal date')
    joined=s.merge(f,on='instrument',how='left',validate='many_to_many')
    joined=joined[joined.available_at.le(joined.date) | joined.available_at.isna()]
    joined=joined.sort_values(['date','instrument','available_at','period_end'])
    latest=joined.drop_duplicates(['date','instrument'],keep='last')
    result=s.merge(latest[['date','instrument']+[col for col in f if col!='instrument']],
                   on=['date','instrument'],how='left',validate='one_to_one')
    if len(result)!=len(s):raise RuntimeError('PIT join changed signal row count')
    return result


def validate_benchmark_snapshot(frame:pd.DataFrame,signal_date,*,weight_tolerance=.001):
    """Validate one fully sourced index-weight snapshot, never renormalize gaps."""
    if set(frame.columns)<BENCHMARK_COLUMNS:
        raise ValueError('Historical benchmark/industry source contract not met')
    x=frame.copy()
    signal=pd.Timestamp(signal_date)
    for col in ('date','industry_valid_from','industry_valid_to','available_at'):
        x[col]=pd.to_datetime(x[col],errors='raise')
    if x.empty or x.instrument.duplicated().any() or x.isna().any().any():
        raise ValueError('Empty, duplicate or incomplete benchmark snapshot')
    if not (x.date.dt.normalize().eq(signal.normalize())&x.available_at.le(signal)&x.industry_valid_from.le(signal)&
            x.industry_valid_to.ge(signal)).all():
        raise ValueError('Future benchmark or industry information at signal date')
    if (x.weight<0).any() or abs(x.weight.sum()-1)>weight_tolerance:
        raise ValueError('Benchmark weights incomplete or invalid')
    if x.source_id.astype(str).str.strip().eq('').any():
        raise ValueError('Missing benchmark source id')
    return x


def benchmark_active_optimizer(scores,benchmark_weights,industries,*,old_weights=None,
                               covariance=None,max_stock_deviation=.01,
                               max_industry_deviation=.05,score_coefficient=.01,
                               risk_penalty=10.,turnover_penalty=.002):
    """D1 if covariance is absent; D2 if a valid ex-ante covariance is supplied."""
    score=np.asarray(scores,float)
    base=np.asarray(benchmark_weights,float)
    sector=np.asarray(industries)
    if score.ndim!=1 or not (len(score)==len(base)==len(sector)) or not np.isfinite(np.r_[score,base]).all():
        raise ValueError('Invalid active optimizer input')
    if np.any(base<0) or abs(base.sum()-1)>1e-8:
        raise ValueError('Benchmark weights must sum to one')
    old=base.copy() if old_weights is None else np.asarray(old_weights,float)
    if old.shape!=base.shape or not np.isfinite(old).all():
        raise ValueError('Invalid prior weights')
    w=cp.Variable(len(base))
    active=w-base
    objective=-score_coefficient*score@active+turnover_penalty*cp.norm1(w-old)
    if covariance is not None:
        cov=np.asarray(covariance,float)
        if cov.shape!=(len(base),len(base)) or not np.isfinite(cov).all() or not np.allclose(cov,cov.T,atol=1e-10):
            raise ValueError('Invalid covariance matrix')
        if np.linalg.eigvalsh(cov).min() < -1e-9:
            raise ValueError('Covariance must be positive semidefinite')
        objective+=risk_penalty*cp.quad_form(active,cov)
    constraints=[w>=0,cp.sum(w)==1,cp.abs(active)<=max_stock_deviation]
    for name in np.unique(sector):
        mask=(sector==name).astype(float)
        constraints.append(cp.abs(mask@active)<=max_industry_deviation)
    problem=cp.Problem(cp.Minimize(objective),constraints)
    problem.solve(solver='CLARABEL')
    if problem.status not in ('optimal','optimal_inaccurate'):
        raise RuntimeError(f'Active optimization failed: {problem.status}')
    answer=np.asarray(w.value,float).reshape(-1)
    if abs(answer.sum()-1)>1e-6 or answer.min() < -1e-7:
        raise RuntimeError('Active optimization numerical breach')
    return answer


def audit_local_sources(root:Path):
    """Explicitly gate C/D; never use prices as a substitute for PIT inputs."""
    root=Path(root)
    data=root/'data'/'v21'
    financial=data/'financial_pit.csv'
    benchmark=data/'csi300_historical_weights_and_industry.csv'
    result={}
    if financial.exists():
        f=pd.read_csv(financial)
        if set(f)<FINANCIAL_COLUMNS:
            raise ValueError('Financial file exists but fails PIT contract')
        result['C']={'status':'INPUT_PRESENT_REQUIRES_COVERAGE_AUDIT','path':str(financial),'rows':len(f)}
    else:
        result['C']={'status':'DATA_NOT_AVAILABLE','required_path':str(financial),
                     'missing_columns':sorted(FINANCIAL_COLUMNS)}
    if benchmark.exists():
        b=pd.read_csv(benchmark)
        if set(b)<BENCHMARK_COLUMNS:
            raise ValueError('Benchmark file exists but fails dated-weight contract')
        result['D']={'status':'INPUT_PRESENT_REQUIRES_COVERAGE_AUDIT','path':str(benchmark),'rows':len(b)}
    else:
        result['D']={'status':'DATA_NOT_AVAILABLE','required_path':str(benchmark),
                     'missing_columns':sorted(BENCHMARK_COLUMNS)}
    return result


if __name__=='__main__':
    repo=Path(__file__).resolve().parents[2]
    output=Path(__file__).resolve().parent/'outputs'/'v21_alternatives'
    output.mkdir(parents=True,exist_ok=True)
    status=audit_local_sources(repo)
    (output/'CD_data_gates.json').write_text(json.dumps(status,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(status,ensure_ascii=False,indent=2))
