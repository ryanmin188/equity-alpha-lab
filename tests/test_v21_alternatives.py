import json
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'/'equity_alpha_v2'))
from execution_v16 import Market
from research_v21_alternatives import state_at_signal, mature_train, residual_labels
from v21_data_gates import asof_financial, validate_benchmark_snapshot, benchmark_active_optimizer
from lot_aware_target_v21 import project_sell_target


class V21AlternativesTests(unittest.TestCase):
    def test_market_state_ignores_future_prices(self):
        dates=pd.bdate_range('2020-01-01',periods=120)
        benchmark=100*np.cumprod(1+.001+.006*np.sin(np.arange(120)))
        stock=np.tile(np.linspace(10,12,120)[:,None],(1,2))
        fields={name:stock.copy() for name in ('open','close','high','low')}
        fields['factor']=np.ones_like(stock)
        fields['volume']=np.full_like(stock,1e6)
        market=Market(dates,['A','B'],fields,benchmark.copy(),benchmark.copy())
        first=state_at_signal(market,dates[80])
        market.benchmark_close[81:]*=10
        self.assertEqual(first,state_at_signal(market,dates[80]))

    def test_training_requires_mature_exit(self):
        dates=pd.date_range('2016-01-31',periods=40,freq='ME')
        panel=pd.DataFrame({'date':dates,'exit_date':dates+pd.Timedelta(days=25),
                            'instrument':'A','forward_return':np.arange(40)/100})
        today=dates[-1]
        train=mature_train(panel,today)
        self.assertIsNotNone(train)
        self.assertTrue((train.exit_date<=today).all())
        self.assertNotIn(today,set(train.date))

    def test_residual_exposure_and_label_time_boundaries(self):
        dates=pd.bdate_range('2020-01-01',periods=180)
        factor=np.zeros((180,5))
        factor[:,1]=.008*np.sin(np.arange(180)*.2)
        factor[:,2]=.004*np.cos(np.arange(180)*.3)
        factor[:,3]=.003*np.sin(np.arange(180)*.11)
        factor[:,4]=.002*np.cos(np.arange(180)*.17)
        factor[:,0]=.0001
        stock_return=factor[:,0]+1.3*factor[:,1]-.5*factor[:,2]+.2*factor[:,3]+.1*factor[:,4]+.0003
        prices=10*np.cumprod(1+stock_return)
        stock=prices[:,None]
        fields={name:stock.copy() for name in ('open','close','high','low')}
        fields['factor']=np.ones_like(stock)
        fields['volume']=np.full_like(stock,1e6)
        market=Market(dates,['A'],fields,np.full(180,100.),np.full(180,100.))
        panel=pd.DataFrame({'date':[dates[150]],'instrument':['A'],
                            'entry_date':[dates[151]],'exit_date':[dates[161]]})
        first=residual_labels(panel,market,factor).iloc[0]
        self.assertEqual(first.paired_history_days,150)
        self.assertAlmostEqual(first.beta_market,1.3,places=4)
        self.assertTrue(np.isfinite(first.residual_label))
        market.fields['close'][161:]*=100
        factor[161:]*=50
        second=residual_labels(panel,market,factor).iloc[0]
        self.assertAlmostEqual(first.beta_market,second.beta_market)
        self.assertAlmostEqual(first.residual_label,second.residual_label)

    def test_financial_join_uses_actual_release_date(self):
        signals=pd.DataFrame({'date':pd.to_datetime(['2024-12-31','2025-05-01']),
                              'instrument':['A','A']})
        financial=pd.DataFrame({'instrument':['A'],'period_end':['2024-12-31'],
            'announcement_date':['2025-04-20'],'available_at':['2025-04-21'],
            'source_id':['vendor:filing:1'],'EP':[.08],'BP':[.7],
            'ROE':[.12],'profit_growth':[.05]})
        result=asof_financial(signals,financial)
        self.assertTrue(pd.isna(result.EP.iloc[0]))
        self.assertEqual(result.EP.iloc[1],.08)
        bad=financial.copy()
        bad.loc[0,'available_at']='2025-04-19'
        with self.assertRaises(ValueError):asof_financial(signals,bad)

    def test_benchmark_timing_and_weights(self):
        snapshot=pd.DataFrame({'date':['2025-01-31']*2,'instrument':['A','B'],
            'weight':[.6,.4],'industry':['Bank','Tech'],
            'industry_valid_from':['2020-01-01']*2,
            'industry_valid_to':['2026-12-31']*2,
            'available_at':['2025-01-30']*2,'source_id':['index-vendor']*2})
        validate_benchmark_snapshot(snapshot,'2025-01-31')
        late=snapshot.copy();late.loc[0,'available_at']='2025-02-01'
        with self.assertRaises(ValueError):validate_benchmark_snapshot(late,'2025-01-31')
        incomplete=snapshot.copy();incomplete.loc[0,'weight']=.3
        with self.assertRaises(ValueError):validate_benchmark_snapshot(incomplete,'2025-01-31')

    def test_active_optimizer_conservation_sector_and_psd(self):
        base=np.array([.3,.2,.3,.2])
        score=np.array([1.,-1.,.5,-.5])
        sector=np.array(['A','A','B','B'])
        covariance=np.diag([.02,.03,.015,.025])
        for cov in (None,covariance):
            w=benchmark_active_optimizer(score,base,sector,covariance=cov)
            self.assertAlmostEqual(w.sum(),1.,places=6)
            self.assertGreaterEqual(w.min(),-1e-7)
            self.assertLessEqual(np.abs(w-base).max(),.010001)
            self.assertLessEqual(abs((w-base)[:2].sum()),.050001)
        invalid=covariance.copy();invalid[0,0]=-.1
        with self.assertRaises(ValueError):
            benchmark_active_optimizer(score,base,sector,covariance=invalid)

    def test_candidate_budget(self):
        p=json.loads((ROOT/'configs'/'v21_research_protocol.json').read_text(encoding='utf-8'))
        self.assertEqual(sum(p['candidate_budget'][key] for key in 'ABCD'),10)
        self.assertEqual(p['C']['status_without_data'].split(';')[0],'DATA_NOT_AVAILABLE')

    def test_lot_aware_projection_prevents_unfillable_partial_sale(self):
        projected,reason=project_sell_target(current_raw_shares=200,desired_raw_shares=100,
            current_value=50000,desired_value=25000,max_position_value=60000,star_board=True)
        self.assertEqual((projected,reason),(0.,'FULL_EXIT_NEAREST_LEGAL'))
        projected,reason=project_sell_target(current_raw_shares=200,desired_raw_shares=180,
            current_value=90000,desired_value=81000,max_position_value=50000,star_board=True)
        self.assertEqual((projected,reason),(0.,'FULL_EXIT_CAP_BREACH'))
        projected,reason=project_sell_target(current_raw_shares=1000,desired_raw_shares=950,
            current_value=40000,desired_value=38000,max_position_value=50000,star_board=True)
        self.assertEqual((projected,reason),(1000.,'KEEP_NEAREST_LEGAL'))


if __name__=='__main__':unittest.main()
