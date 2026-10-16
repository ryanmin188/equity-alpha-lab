"""Time-boundary and numerical tests for exploratory signal research."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/equity_alpha_v2'))
from research_v23_signals import FAMILIES, annual_metrics, mature_weights, residual_signal, predict
from research_v23_multiscale import price_volume_features, causal_train
from research_v23_market_gate import past_beta, market_gates, predict as gate_predict
from execution_v16 import Market
from audit_v23_accounts import replay


class V23Tests(unittest.TestCase):
    def test_market_gate_predictions_ignore_future_prices_and_labels(self):
        rng=np.random.default_rng(3)
        dates=pd.bdate_range('2016-01-01',periods=400)
        prices=np.cumprod(1+rng.normal(.0003,.01,(400,40)),axis=0)*20
        benchmark=np.cumprod(1+rng.normal(.0003,.01,400))*100
        fields={k:prices.copy() for k in ('close','open','high','low')}
        fields['volume']=np.full((400,40),1e6)
        fields['factor']=np.ones((400,40))
        market=Market(dates,[str(j) for j in range(40)],fields,benchmark.copy(),benchmark.copy())
        decision=dates[300]
        frame=pd.DataFrame({'date':decision,'entry_date':dates[301],'exit_date':dates[322],
                            'instrument':market.symbols,'forward_return':rng.normal(size=40),
                            'raw_vol_60':.02,'reversal_5':rng.normal(size=40),'equal8':rng.normal(size=40)})
        before,_=gate_predict(frame,market)
        market.fields['close'][301:]*=1000
        market.benchmark_close[301:]*=.01
        frame.forward_return*=-100
        after,_=gate_predict(frame,market)
        np.testing.assert_allclose(before.score,after.score)
        self.assertEqual(market_gates(np.arange(1,254.)),(True,True))
        self.assertEqual(market_gates(np.arange(253,0,-1.)),(False,False))
        x=np.linspace(-.1,.1,252)
        self.assertAlmostEqual(past_beta(2*x+.005,x),2.)

    def test_independent_cash_replay_handles_capital_and_fees(self):
        dates=pd.bdate_range('2024-01-01',periods=3)
        prices=np.array([[10.],[11.],[12.]])
        fields={k:prices.copy() for k in ('close','open','high','low')}
        fields['volume']=np.full((3,1),1e6);fields['factor']=np.ones((3,1))
        m=Market(dates,['S'],fields,np.ones(3),np.ones(3))
        orders=pd.DataFrame([{'date':dates[1],'status':'FILLED','instrument':'S','side':'BUY',
                              'filled_adjusted_units':100.,'notional':1100.,'fees':5.}])
        d=pd.DataFrame({'date':dates[1:],'valuation':['close','open'],'nav':[9999995.,10000095.],
                        'wealth':[.9999995,1.0000095],'cash':[9998895.,9998895.],
                        'net':[-.0000005,100/9999995.]})
        result=replay(d,orders,m)
        self.assertLess(result.nav_residual.abs().max(),1e-7)
        d.loc[1,'cash']+=1
        with self.assertRaises(AssertionError):replay(d,orders,m)

    def test_multiscale_linear_trend_and_scale_invariance(self):
        c=np.column_stack([np.arange(1,62,dtype=float),np.arange(1,62,dtype=float)**1.1])
        v=np.column_stack([np.arange(100,161,dtype=float),np.arange(300,361,dtype=float)])
        base=price_volume_features(c,c-.1,c+.2,c-.2,v)
        scaled=price_volume_features(c*7,(c-.1)*7,(c+.2)*7,(c-.2)*7,v*3)
        np.testing.assert_allclose(base,scaled,atol=1e-10,rtol=1e-8,equal_nan=True)
        self.assertAlmostEqual(base.loc[0,'w60_rsq'],1.)
        self.assertAlmostEqual(base.loc[0,'w60_resid'],0.,places=12)
        self.assertAlmostEqual(base.loc[0,'w20_roc'],41/61)
        self.assertEqual(base.loc[0,'w20_rank'],1.)

    def test_multiscale_rejects_partial_mature_month(self):
        dates=pd.date_range('2010-01-31',periods=50,freq='ME')
        frame=pd.DataFrame({'date':np.repeat(dates,2),
                            'exit_date':np.repeat(dates+pd.Timedelta(days=35),2),
                            'forward_return':1.})
        decision=dates[-1]
        partial=dates[-3]
        frame.loc[frame[frame.date.eq(partial)].index[-1],'exit_date']=decision+pd.Timedelta(days=10)
        train=causal_train(frame,decision)
        self.assertNotIn(partial,set(train.date))
        self.assertTrue((train.exit_date<=decision).all())

    def test_residual_removes_market_and_intercept(self):
        rng = np.random.default_rng(42)
        market = rng.normal(0,.01,252)
        residual = rng.normal(0,.005,252)
        base = residual_signal(residual,market)
        shifted = residual_signal(residual+2.3*market+.007,market)
        self.assertAlmostEqual(base,shifted,places=10)
        self.assertTrue(np.isnan(residual_signal(np.ones(252),market)))

    def test_maturity_excludes_future_and_immature_labels(self):
        dates = pd.date_range('2018-01-31',periods=30,freq='ME')
        h = pd.DataFrame({'date':dates,'exit_date':dates+pd.Timedelta(days=35)})
        for j,name in enumerate(FAMILIES):
            h[name] = np.arange(30)/100+j/100
        decision = dates[-3]
        before, train = mature_weights(h,decision,24)
        future = h.copy()
        future.loc[future.exit_date>decision,FAMILIES] = -100000
        after,_ = mature_weights(future,decision,24)
        np.testing.assert_allclose(before,after)
        self.assertLessEqual(train.exit_date.max(),decision)
        self.assertAlmostEqual(np.abs(before).sum(),1.)

    def test_current_labels_do_not_change_current_scores(self):
        rng = np.random.default_rng(8)
        dates = pd.date_range('2013-01-31',periods=55,freq='ME')
        rows=[]
        for d in dates:
            for j in range(40):
                rows.append({'date':d,'entry_date':d+pd.Timedelta(days=1),
                             'exit_date':d+pd.Timedelta(days=35),'instrument':str(j),
                             'forward_return':rng.normal(),'raw_vol_60':.02,
                             'equal8':rng.normal(),**{f:rng.normal() for f in FAMILIES}})
        frame=pd.DataFrame(rows)
        before,_,_=predict(frame)
        decision=dates[-3]
        altered=frame.copy()
        altered.loc[altered.exit_date>decision,'forward_return'] *= -1000
        after,_,_=predict(altered)
        cols=['date','instrument','model','score']
        pd.testing.assert_frame_equal(before.loc[before.date<=decision,cols].reset_index(drop=True),
                                      after.loc[after.date<=decision,cols].reset_index(drop=True))

    def test_annual_return_keeps_prior_year_denominator(self):
        daily=pd.DataFrame({'date':['2024-12-30','2024-12-31','2025-01-02','2025-12-31'],
                            'net':[.1,.1,-.1,.2], 'wealth':[1.1,1.21,1.089,1.3068],
                            'benchmark_wealth':[1.05,1.1,1.2,1.32],
                            'max_weight':[.04]*4,'cash_weight':[.03]*4})
        a=annual_metrics(daily).set_index('year')
        self.assertAlmostEqual(a.loc[2024,'strategy_return'],.21)
        self.assertAlmostEqual(a.loc[2025,'strategy_return'],.08)
        self.assertAlmostEqual(a.loc[2025,'benchmark_return'],.2)
        self.assertAlmostEqual(a.loc[2025,'active_pp'],-.12)


if __name__=='__main__':unittest.main()
