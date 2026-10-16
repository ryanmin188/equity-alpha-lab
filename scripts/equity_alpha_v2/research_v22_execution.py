"""Frozen V2.2 E0/E1/E2 replay. All outputs are private, exploratory history."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from attribute_v18_holdings import replay
from execution_v16 import run_account as run_e0
from execution_v22 import run_account as run_v22

ROOT = Path(__file__).resolve().parents[2]
SOURCE = Path(__file__).resolve().parent / 'outputs' / 'v16_improvements'
V21 = Path(__file__).resolve().parent / 'outputs' / 'v21_alternatives'
OUT = Path(__file__).resolve().parent / 'outputs' / 'v22_execution_reality'
CONFIG = json.loads((ROOT / 'configs' / 'v16_protocol.json').read_text(encoding='utf-8'))
PORTFOLIO = {'name':'Buffer30_60','top_n':30,'buffer_n':60,'weighting':'equal'}


def load_predictions():
    b0 = pd.read_csv(SOURCE/'predictions_final.csv', parse_dates=['date','entry_date','exit_date'])
    b0 = b0[b0.model.eq('Equal8')]
    earlier=pd.read_csv(SOURCE/'predictions_develop.csv',parse_dates=['date','entry_date','exit_date'])
    b0=pd.concat([earlier[earlier.model.eq('Equal8')],b0],ignore_index=True)
    a = pd.read_csv(V21/'A_predictions_private.csv',parse_dates=['date','entry_date','exit_date'])
    b = pd.read_csv(V21/'B_predictions_private.csv',parse_dates=['date','entry_date','exit_date'])
    a = a[a.model.eq('A2_CH4_residual_ridge')]
    b = b[b.model.eq('B2_state_tree')]
    return {'B0_Equal8':b0,'A2_CH4_residual_ridge':a,'B2_state_tree':b}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'market_final.pkl').open('rb') as f:
        market,_=pickle.load(f)
    predictions=load_predictions()
    intervals={'reused_validation_2021_2023':('2021-01-01','2023-12-01'),
               'common_2024_2025':('2024-01-01','2026-02-02'),
               'extended_2024_2026':('2024-01-01','2026-10-08')}
    comparisons=[]
    for label,(start,end) in intervals.items():
        for candidate,frame in predictions.items():
            if label=='extended_2024_2026' and candidate.startswith('A2'):
                continue
            subset=frame[(frame.entry_date>=start)&(frame.exit_date<=end)].copy()
            if subset.empty:
                raise ValueError(f'No input for {candidate} {label}')
            assert not subset.duplicated(['date','instrument']).any()
            assert (subset.date<subset.entry_date).all()
            assert (subset.entry_date<subset.exit_date).all()
            for regime in ('E0','E1','E2'):
                if regime=='E0':
                    summary,daily,orders,targets=run_e0(subset,market,PORTFOLIO,CONFIG)
                    exposures=pd.DataFrame()
                else:
                    summary,daily,orders,targets,exposures=run_v22(
                        subset,market,PORTFOLIO,CONFIG,regime=regime)
                ledger,checks,stale=replay(daily,orders,market)
                detail=OUT/label/candidate/regime
                detail.mkdir(parents=True,exist_ok=True)
                for name,data in [('daily_account',daily),('orders',orders),('targets',targets),
                                  ('exposures',exposures),('audit_checks',checks),('stale_marks',stale)]:
                    if not data.empty:
                        data.to_csv(detail/f'{name}.csv',index=False)
                max_residual=float(checks[['nav_residual','pnl_residual','cash_residual']].abs().max().max())
                breaches=int((daily.max_weight>CONFIG['max_stock_weight']+1e-10).sum())
                rejected_lot=int(orders.status.eq('BLOCKED_LOT_OR_CAPACITY').sum())
                comparison={'interval':label,'candidate':candidate,'regime':regime,
                            'max_audit_residual_cny':max_residual,
                            'days_above_5pct':breaches,'highest_marked_weight':daily.max_weight.max(),
                            'median_holdings':daily.n_holdings.median(),
                            'lot_or_capacity_rejects':rejected_lot,
                            'risk_only_orders':int(orders.kind.eq('RISK_ONLY').sum()) if 'kind' in orders else 0,
                            **summary}
                comparisons.append(comparison)
                print(label,candidate,regime,'return=',round(summary['total_return'],4),
                      'max_weight=',round(comparison['highest_marked_weight'],4),
                      'lot_rejects=',rejected_lot,'audit=',max_residual,flush=True)
    result=pd.DataFrame(comparisons)
    result.to_csv(OUT/'e0_e1_e2_comparison.csv',index=False)
    original=pd.read_csv(SOURCE/'final/Equal8__Buffer30_60/daily_account.csv')
    recreated=pd.read_csv(OUT/'extended_2024_2026/B0_Equal8/E0/daily_account.csv')
    if len(original)!=len(recreated) or not np.allclose(original.nav,recreated.nav,atol=1e-6,rtol=0):
        raise AssertionError('E0 does not reproduce saved frozen B0 account')
    (OUT/'e0_saved_baseline_reproduction.json').write_text(json.dumps(
        {'identical_rows':len(original),'max_abs_nav_difference_cny':float(np.max(np.abs(original.nav-recreated.nav)))},
        indent=2),encoding='utf-8')
    print(result[['interval','candidate','regime','total_return','benchmark_return',
                  'days_above_5pct','highest_marked_weight','lot_or_capacity_rejects',
                  'max_audit_residual_cny']].to_string(index=False))


if __name__=='__main__':
    main()
