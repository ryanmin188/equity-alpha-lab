"""Economic invariants with synthetic bars. No real market data needed."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/equity_alpha_v2'))
import execution_v16 as ex
import research_v16_improvements as research


def fixture():
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    config.update(max_stock_weight=.5,max_rebalance_gross_turnover=2)
    dates=pd.bdate_range('2022-01-03',periods=65)
    symbols=['SH600001','SH600002','SH600003','SH600004']
    fields={f:np.full((65,4),10.) for f in ['open','close','high','low']}
    fields['factor']=np.ones((65,4))
    fields['volume']=np.full((65,4),1e7)
    fields['open'][43,0]=np.nan  # a held name cannot be sold
    market=ex.Market(dates,symbols,fields,np.full(65,100.),np.full(65,100.))
    rows=[]
    for signal,entry,exit_,ranking in [(21,22,43,symbols),(42,43,64,symbols[2:]+symbols[:2])]:
        for rank,symbol in enumerate(ranking):
            rows.append(dict(date=dates[signal],entry_date=dates[entry],exit_date=dates[exit_],
                             instrument=symbol,score=4-rank,raw_vol_60=.02,forward_return=0.))
    portfolio=dict(name='fixture',top_n=2,buffer_n=2,weighting='equal')
    return pd.DataFrame(rows),market,portfolio,config


class ExecutionV16Tests(unittest.TestCase):
    def test_independent_ledger_replay_blocked_sale_and_lots(self):
        p,m,portfolio,c=fixture()
        summary,daily,orders,_=ex.run_account(p,m,portfolio,c)
        cash=c['capital']; held={s:0. for s in m.symbols}
        for row in daily.itertuples():
            for order in orders[orders.date==row.date].itertuples():
                if order.status.startswith('BLOCKED'):
                    self.assertEqual(order.filled_adjusted_units,0.)
                    continue
                sign=1 if order.side=='BUY' else -1
                cash-=sign*order.notional+order.fees
                held[order.instrument]+=sign*order.filled_adjusted_units
                self.assertAlmostEqual(order.raw_share_proxy/100,round(order.raw_share_proxy/100))
                self.assertLessEqual(order.participation,c['max_participation_of_lagged_adv']+1e-12)
                self.assertGreaterEqual(cash,0)
                self.assertGreaterEqual(held[order.instrument],0)
            self.assertAlmostEqual(cash,row.cash,places=6)
            self.assertAlmostEqual(cash+sum(held.values())*10,row.nav,places=6)
        blocked=orders[(orders.instrument=='SH600001')&(orders.side=='SELL')]
        self.assertEqual(blocked.iloc[0].status,'BLOCKED_PRICE_OR_FACTOR')
        self.assertGreater(held['SH600001'],0)
        self.assertAlmostEqual(summary['total_return']*c['capital'],
                               -summary['fees_cny']-summary['slippage_impact_cny'],places=6)

    def test_future_labels_do_not_affect_account(self):
        p,m,portfolio,c=fixture()
        before=ex.run_account(p,m,portfolio,c)
        p['forward_return']=np.nan
        after=ex.run_account(p,m,portfolio,c)
        for a,b in zip(before[1:],after[1:]): pd.testing.assert_frame_equal(a,b)

    def test_full_exit_does_not_strand_odd_lot(self):
        p,m,portfolio,c=fixture()
        m.fields['factor'][43:,1]=1.001
        _,_,orders,_=ex.run_account(p,m,portfolio,c)
        buys=orders[(orders.instrument=='SH600002')&(orders.side=='BUY')]
        sells=orders[(orders.instrument=='SH600002')&(orders.side=='SELL')]
        self.assertAlmostEqual(buys.filled_adjusted_units.sum(),sells.filled_adjusted_units.sum())

    def test_sizing_precedes_open_and_capacity_uses_lag(self):
        p,m,portfolio,c=fixture()
        _,_,before,_=ex.run_account(p,m,portfolio,c)
        m.fields['open'][22,:]=10.5
        # Today's volume cannot inflate the lagged capacity estimate.
        m.fields['volume'][22,:]*=100
        m.__post_init__()
        _,_,after,_=ex.run_account(p,m,portfolio,c)
        a=before[before.date==m.dates[22]]
        b=after[after.date==m.dates[22]]
        np.testing.assert_allclose(a.requested_adjusted_units,b.requested_adjusted_units)
        self.assertAlmostEqual(m.lagged_adv[22,0],1e8)

    def test_tax_change_and_minimum_commission(self):
        _,_,_,c=fixture()
        early=ex.charges(100000,'SELL',pd.Timestamp('2023-08-25'),c)
        late=ex.charges(100000,'SELL',pd.Timestamp('2023-08-28'),c)
        self.assertAlmostEqual(early-late,50.)
        self.assertAlmostEqual(ex.charges(100,'BUY',pd.Timestamp('2024-01-02'),c),5.002)

    def test_training_matures_and_final_guard_survives_incomplete_run(self):
        root=ROOT/'scripts/equity_alpha_v2/outputs/test_v16'
        root.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=root) as temporary:
            output=Path(temporary)
            self.assertEqual(output.resolve().parent,root.resolve())
            dates=pd.date_range('2020-01-31',periods=5,freq='ME')
            panel=pd.DataFrame(dict(date=dates,instrument='FAKE',entry_date=dates+pd.Timedelta(days=1),
                                    exit_date=dates+pd.DateOffset(months=2),forward_return=.01,raw_vol_60=.02))
            calls=[]
            def score(train,today,config,needed):
                self.assertTrue((train.exit_date<=today.date.iloc[0]).all())
                calls.append(len(train))
                return {'Equal8':np.ones(len(today))}
            with patch.object(research,'OUT',output),patch.object(research,'score_month',score),contextlib.redirect_stdout(io.StringIO()):
                research.predict(panel,dict(training_lookback_months=60,minimum_training_months=1),['Equal8'],'2020-01-01','test')
                self.assertTrue(calls)
                (output/'frozen_selection.json').write_text('{"final_opened":true}',encoding='utf-8')
                with self.assertRaisesRegex(RuntimeError,'already opened'): research.develop({})


if __name__=='__main__': unittest.main()
