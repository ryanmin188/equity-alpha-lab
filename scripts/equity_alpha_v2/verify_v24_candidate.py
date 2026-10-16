"""Validate a historical candidate without claiming independent prediction proof."""
import hashlib
import json
import pickle
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pandas as pd

import research_v16_improvements as feature_source
from execution_v24 import run_account
from execution_v22 import run_account as old_execution
from attribute_v18_holdings import replay as security_replay
from research_v24_search import OUT,ROOT,SOURCE

MODEL='mom_252_skip20_eq25'
PORTFOLIO={'name':'Buffer20_40','top_n':20,'buffer_n':40,'weighting':'equal'}


def main():
    with (SOURCE/'market_final.pkl').open('rb') as f:market,intervals=pickle.load(f)
    predictions=pd.read_pickle(OUT/'monthly_predictions_private.pkl')
    p=predictions[(predictions.model==MODEL)&(predictions.entry_date>='2024-01-01')&(predictions.exit_date<='2026-10-08')].copy()
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    folder=OUT/'opened_2024_2026'/MODEL/PORTFOLIO['name']/'base'
    original=pd.read_csv(folder/'daily_account.csv',parse_dates=['date'])
    orders=pd.read_csv(folder/'orders.csv',parse_dates=['date','signal_date'])
    ledger,checks,stale=security_replay(original,orders,market)
    ledger.groupby(['year','instrument']).net_pnl.sum().reset_index().to_csv(OUT/'candidate_yearly_security_pnl_private.csv',index=False)
    checks.to_csv(OUT/'candidate_security_replay.csv',index=False)
    stale.to_csv(OUT/'candidate_stale_marks_private.csv',index=False)
    # Run the old E2 only for this NEW signal, never overwrite old accounts.
    old=old_execution(p,market,PORTFOLIO,config,regime='E2')
    pd.DataFrame({'date':original.date,'v24_nav':original.nav,'old_gate_nav':old[1].nav}).to_csv(OUT/'candidate_volume_gate_control.csv',index=False)
    cutoff=market.dates[market.dates.searchsorted(pd.Timestamp('2025-06-02'))]
    altered=deepcopy(market)
    i=altered.date_index[cutoff]
    for name in ('open','close','high','low'):altered.fields[name][i:]*=1.07
    altered.fields['volume'][i:]*=7
    altered.benchmark_open[i:]*=1.03;altered.benchmark_close[i:]*=1.03
    altered.__post_init__()
    changed_p=p.copy();changed_p['forward_return']=-999.
    result=run_account(changed_p,altered,PORTFOLIO,config)
    cols=['date','signal_date','instrument','side','requested_adjusted_units','filled_adjusted_units','notional','fees','status']
    pd.testing.assert_frame_equal(orders.loc[orders.date<cutoff,cols].reset_index(drop=True),
                                  result[2].loc[result[2].date<cutoff,cols].reset_index(drop=True),check_exact=False,atol=1e-8,rtol=1e-12)
    np.testing.assert_allclose(original.loc[original.date<cutoff,'nav'],result[1].loc[result[1].date<cutoff,'nav'],atol=1e-7,rtol=0)
    # Rebuild original monthly features from bars; future label prices are changed,
    # but cannot change any historical feature value or universe eligibility.
    feature_dir=OUT/'feature_time_audit';feature_dir.mkdir(parents=True,exist_ok=True)
    with patch.object(feature_source,'OUT',feature_dir):
        first=feature_source.make_panel(market,intervals,'original')
        changed=feature_source.make_panel(altered,intervals,'future_shock')
    cols=['date','instrument']+feature_source.FEATURES+['raw_vol_60']
    a=first[first.date<cutoff][cols].reset_index(drop=True)
    b=changed[changed.date<cutoff][cols].reset_index(drop=True)
    pd.testing.assert_frame_equal(a,b)
    cached=pd.read_pickle(SOURCE/'panel_final.pkl')
    pd.testing.assert_frame_equal(first[cols].reset_index(drop=True),cached[cols].reset_index(drop=True))
    family=pd.read_pickle(SOURCE.parent/'v23_signal_comparison/family_panel_private.pkl')
    score_checks=[]
    for date,g in cached.groupby('date',sort=True):
        eq=g[feature_source.BASE].fillna(0).to_numpy()@np.array([1,1,1,-1,-1,-1,-1,1])/8
        rank_eq=pd.Series(eq).rank(pct=True).to_numpy()*2-1
        rank_mom=g.mom_252_skip20.rank(pct=True).fillna(.5).to_numpy()*2-1
        expected=.25*rank_eq+.75*rank_mom
        actual=predictions[(predictions.model==MODEL)&(predictions.date==date)].set_index('instrument').reindex(g.instrument).score.to_numpy()
        np.testing.assert_allclose(expected,actual,atol=1e-14)
        score_checks.append({'date':date,'max_score_difference':float(np.max(abs(expected-actual)))})
    pd.DataFrame(score_checks).to_csv(OUT/'candidate_independent_score_checks.csv',index=False)
    result={'feature_rows_checked':len(a),'independent_score_months':len(score_checks),
            'future_cutoff':str(cutoff.date()),'future_mutation_changes_historical_features':False,
            'future_mutation_changes_historical_orders':False,
            'max_security_pnl_replay_residual_cny':float(checks[['nav_residual','pnl_residual','cash_residual']].abs().max().max()),
            'old_current_volume_gate_max_nav_difference_cny':float(np.max(abs(original.nav-old[1].nav))),
            'classification':'Causal computation tests passed; research selection remains posthoc'}
    (OUT/'candidate_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    hashes={str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in [SOURCE/'market_final.pkl',SOURCE/'panel_final.pkl',ROOT/'configs/v24_research_protocol.json',ROOT/'scripts/equity_alpha_v2/execution_v24.py']}
    (OUT/'candidate_input_hashes.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
