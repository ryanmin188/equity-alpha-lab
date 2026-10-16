"""Numeric and time-boundary tests for prespecified V2.2 diagnostics."""
from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/equity_alpha_v2'))
from execution_v16 import Market
from research_v16_improvements import BASE,FEATURES
from research_v22_horizons import attach_horizon_return,monthly_rows
from research_v22_pairwise import make_block
from research_v22_risk import covariance_forecasts,LOOKBACK,N_STOCKS


class V22DiagnosticTests(unittest.TestCase):
    def test_horizon_label_exact_and_future_mutation(self):
        dates=pd.bdate_range('2024-01-01',periods=100)
        symbols=['SH600001']
        values=np.arange(1,101,dtype=float)[:,None]
        fields={name:values.copy() for name in ('open','close','high','low')}
        fields['volume']=np.full((100,1),1e6)
        fields['factor']=np.ones((100,1))
        market=Market(dates,symbols,fields,np.full(100,100.),np.full(100,100.))
        panel=pd.DataFrame({'date':[dates[30]],'entry_date':[dates[31]],
                            'exit_date':[dates[51]],'instrument':[symbols[0]]})
        before=attach_horizon_return(panel,market,5)
        self.assertEqual(before.horizon_exit.iloc[0],dates[36])
        self.assertAlmostEqual(before.horizon_return.iloc[0],37/32-1)
        market.fields['open'][37:]*=100
        after=attach_horizon_return(panel,market,5)
        self.assertAlmostEqual(before.horizon_return.iloc[0],after.horizon_return.iloc[0])

    def test_quintile_counts_and_ic_direction(self):
        n=50; date=pd.Timestamp('2020-01-31')
        frame=pd.DataFrame({'date':[date]*n,'horizon_return':np.arange(n,dtype=float)/1000})
        for factor in FEATURES:frame[factor]=np.arange(n,dtype=float)
        rows=monthly_rows(frame,20)
        self.assertEqual(len(rows),len(FEATURES))
        self.assertTrue(rows.five_groups_complete.all())
        self.assertTrue((rows.n_stocks==n).all())
        self.assertTrue(np.allclose(rows.rank_ic,1.))
        self.assertTrue((rows.top_bottom_gross>0).all())

    def test_pairs_are_same_month_symmetric_and_deterministic(self):
        n=100;date=pd.Timestamp('2020-01-31')
        group=pd.DataFrame({'exit_date':[pd.Timestamp('2020-03-01')]*n,
                            'forward_return':np.linspace(-1,1,n)})
        for j,name in enumerate(BASE):group[name]=np.arange(n)+j
        a=make_block(group,date);b=make_block(group,date)
        half=len(a['pair_y'])//2
        np.testing.assert_allclose(a['pair_x'][:half],-a['pair_x'][half:])
        np.testing.assert_array_equal(a['pair_y'][:half],1-a['pair_y'][half:])
        np.testing.assert_array_equal(a['pair_y'],b['pair_y'])
        self.assertEqual(a['exit_date'],pd.Timestamp('2020-03-01'))

    def test_covariance_symmetry_psd_and_no_future_access(self):
        rng=np.random.default_rng(7)
        past=rng.normal(size=(LOOKBACK,N_STOCKS))*.01
        before=covariance_forecasts(past)
        future=rng.normal(size=(20,N_STOCKS))
        future*=1000
        after=covariance_forecasts(past)
        for key in before:
            np.testing.assert_allclose(before[key],after[key])
            np.testing.assert_allclose(before[key],before[key].T)
            self.assertGreaterEqual(np.linalg.eigvalsh(before[key]).min(),-1e-12)
        with self.assertRaises(ValueError):covariance_forecasts(past[:251])


if __name__=='__main__':unittest.main()
