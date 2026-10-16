"""Independent account invariants and prospective risk-order behavior."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/equity_alpha_v2'))
import execution_v16 as old
import execution_v22 as new
from attribute_v18_holdings import replay


def market_case():
    dates=pd.bdate_range('2024-01-01',periods=70)
    symbols=['SH688256','SH600001','SH600002']
    fields={name:np.full((70,3),10.,float) for name in ('open','close','high','low')}
    fields['factor']=np.ones((70,3))
    fields['volume']=np.full((70,3),1e7)
    fields['close'][30,0]=10.5
    fields['open'][31:,0]=10.5
    fields['close'][31:,0]=10.5
    market=old.Market(dates,symbols,fields,np.full(70,100.),np.full(70,100.))
    rows=[]
    for signal,entry,exit_,order in [(21,22,43,symbols),(42,43,69,symbols[1:]+symbols[:1])]:
        for rank,symbol in enumerate(order):
            rows.append({'date':dates[signal],'entry_date':dates[entry],'exit_date':dates[exit_],
                         'instrument':symbol,'score':3-rank,'raw_vol_60':.02,'forward_return':0.})
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    config.update(max_stock_weight=.5,max_rebalance_gross_turnover=2)
    return pd.DataFrame(rows),market,{'name':'test','top_n':2,'buffer_n':2,'weighting':'equal'},config


class ExecutionV22Tests(unittest.TestCase):
    def test_e0_unchanged_and_e1_e2_replay(self):
        p,m,portfolio,c=market_case()
        before=old.run_account(p,m,portfolio,c)
        for regime in ('E1','E2'):
            summary,daily,orders,targets,exposures=new.run_account(p,m,portfolio,c,regime=regime)
            _,checks,_=replay(daily,orders,m)
            self.assertLess(checks.nav_residual.abs().max(),1e-5)
            self.assertLess(checks.pnl_residual.abs().max(),1e-5)
            self.assertTrue((daily.cash>=-1e-6).all())
            self.assertTrue((orders.loc[orders.status.str.startswith('BLOCKED'),'filled_adjusted_units']==0).all())
            self.assertFalse(exposures.empty)
            self.assertTrue((exposures.marked_weight>=0).all())
            self.assertEqual(summary['regime'],regime)
        after=old.run_account(p,m,portfolio,c)
        pd.testing.assert_frame_equal(before[1],after[1])
        pd.testing.assert_frame_equal(before[2],after[2])

    def test_risk_uses_prior_close_and_ignores_future_labels(self):
        p,m,portfolio,c=market_case()
        c['max_stock_weight']=.5
        c['max_rebalance_gross_turnover']=2
        m.fields['close'][30,0]=20.
        m.fields['open'][31:,0]=20.
        m.fields['close'][31:,0]=20.
        m.__post_init__()
        first=new.run_account(p,m,portfolio,c,regime='E2')
        risk=first[2][first[2].kind.eq('RISK_ONLY')]
        self.assertTrue((risk.side=='SELL').all())
        self.assertGreater(len(risk),0)
        self.assertGreaterEqual(risk.date.min(),m.dates[31])
        p['forward_return']=999.
        second=new.run_account(p,m,portfolio,c,regime='E2')
        pd.testing.assert_frame_equal(first[1],second[1])
        pd.testing.assert_frame_equal(first[2],second[2])

    def test_invalid_regime_rejected(self):
        p,m,portfolio,c=market_case()
        with self.assertRaises(ValueError):
            new.run_account(p,m,portfolio,c,regime='E0')

    def test_failed_monthly_exit_is_retried_without_new_signal(self):
        p,m,portfolio,c=market_case()
        m.fields['open'][43,0]=np.nan
        _,_,orders,_,_=new.run_account(p,m,portfolio,c,regime='E2')
        first=orders[(orders.date==m.dates[43])&(orders.instrument=='SH688256')&(orders.side=='SELL')]
        self.assertEqual(first.iloc[0].status,'BLOCKED_PRICE_OR_FACTOR')
        later=orders[(orders.date>m.dates[43])&(orders.instrument=='SH688256')&
                     (orders.side=='SELL')&(orders.kind=='RISK_ONLY')]
        self.assertGreater(len(later),0)
        self.assertGreater(later.filled_adjusted_units.sum(),0)

    def test_future_price_change_cannot_rewrite_prior_orders(self):
        p,m,portfolio,c=market_case()
        first=new.run_account(p,m,portfolio,c,regime='E2')
        cutoff=m.dates[50]
        m.fields['close'][50:]*=1.05
        m.__post_init__()
        second=new.run_account(p,m,portfolio,c,regime='E2')
        pd.testing.assert_frame_equal(first[1][first[1].date<cutoff].reset_index(drop=True),
                                      second[1][second[1].date<cutoff].reset_index(drop=True))
        pd.testing.assert_frame_equal(first[2][first[2].date<cutoff].reset_index(drop=True),
                                      second[2][second[2].date<cutoff].reset_index(drop=True))

    def test_post_projection_rebalance_budget(self):
        p,m,portfolio,c=market_case()
        c['max_rebalance_gross_turnover']=.2
        _,_,_,targets,_=new.run_account(p,m,portfolio,c,regime='E2')
        monthly=targets[targets.kind.eq('REBALANCE')].iloc[1:]
        self.assertTrue((monthly.planned_legal_turnover<=.2+1e-8).all())


if __name__=='__main__':unittest.main()
