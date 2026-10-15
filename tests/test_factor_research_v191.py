import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/equity_alpha_v2'))
from factor_research_v191 import (FEATURES,bh_fdr,monthly_factor_metrics,
                                  preprocess_cross_section,quantile_rows,validate_panel)


class FactorResearchTests(unittest.TestCase):
    def panel(self):
        d={'date':['2023-12-29'],'entry_date':['2024-01-02'],
           'exit_date':['2024-02-01'],'instrument':['SH600000'],
           'forward_return':[.05]}
        d.update({f:[.1] for f in FEATURES})
        return pd.DataFrame(d)

    def test_preprocessing_ignores_future_returns(self):
        x=pd.Series(np.arange(120,dtype=float))
        y=pd.Series(np.arange(120,dtype=float))
        a,params_a=preprocess_cross_section(x)
        y.iloc[:]=10000 # deliberately perturb every future outcome
        b,params_b=preprocess_cross_section(x)
        pd.testing.assert_frame_equal(a,b)
        self.assertEqual(params_a,params_b)

    def test_future_month_does_not_change_fixed_history(self):
        first=pd.Series(np.arange(120,dtype=float),index=[f'S{i}' for i in range(120)])
        later=pd.Series(np.arange(120,dtype=float)*100)
        before,meta_before=preprocess_cross_section(first)
        _=preprocess_cross_section(later)
        after,meta_after=preprocess_cross_section(first)
        pd.testing.assert_frame_equal(before,after)
        self.assertEqual(meta_before,meta_after)

    def test_training_and_evaluation_boundaries(self):
        frame=self.panel()
        validate_panel(frame,decision_date=pd.Timestamp('2024-02-01'),training=True)
        with self.assertRaises(ValueError):
            validate_panel(frame,decision_date=pd.Timestamp('2024-01-31'),training=True)
        validate_panel(frame,decision_date=pd.Timestamp('2023-12-29'),training=False)
        with self.assertRaises(ValueError):
            validate_panel(frame,decision_date=pd.Timestamp('2024-01-02'),training=False)

    def test_duplicate_stock_month_and_invalid_label_dates(self):
        frame=self.panel()
        with self.assertRaises(ValueError):
            validate_panel(pd.concat([frame,frame],ignore_index=True))
        frame.loc[0,'exit_date']='2023-12-28'
        with self.assertRaises(ValueError):
            validate_panel(frame)

    def test_all_missing_constant_outlier_and_small_sample(self):
        _,missing=preprocess_cross_section(pd.Series([np.nan]*120))
        self.assertEqual(missing['status'],'all_missing')
        _,constant=preprocess_cross_section(pd.Series([3.]*120))
        self.assertEqual(constant['status'],'constant_or_zero_mad')
        _,small=preprocess_cross_section(pd.Series(np.arange(20,dtype=float)))
        self.assertEqual(small['status'],'insufficient_stocks')
        x=pd.Series(np.arange(119,dtype=float).tolist()+[100000.])
        processed,info=preprocess_cross_section(x)
        self.assertGreater(info['clipped_count'],0)
        self.assertLess(processed.winsorized.max(),100000)
        self.assertAlmostEqual(processed.zscore.mean(),0.,places=12)

    def test_ic_direction_and_quantile_arithmetic(self):
        x=pd.Series(np.arange(100,dtype=float),index=[f'S{i:03d}' for i in range(100)])
        y=x/100
        metric=monthly_factor_metrics(x,y)
        self.assertAlmostEqual(metric['pearson_ic'],1.)
        self.assertAlmostEqual(metric['rank_ic'],1.)
        groups,s=quantile_rows(x,y,5)
        self.assertEqual(groups.n_stocks.tolist(),[20]*5)
        self.assertAlmostEqual(s['high_minus_low'],.8)
        self.assertAlmostEqual(s['monotonic_spearman'],1.)
        groups2,s2=quantile_rows(x,-y,10)
        self.assertEqual(sum(groups2.n_stocks),100)
        self.assertAlmostEqual(s2['high_minus_low'],-.9)

    def test_bh_fdr_reference_and_invariants(self):
        p=np.array([.01,.04,.03,.2,np.nan])
        q=bh_fdr(p)
        np.testing.assert_allclose(q[:4],[.05,.06666666666666667,.06666666666666667,.25])
        self.assertTrue(np.isnan(q[4]))
        self.assertEqual(len(q),len(p))
        self.assertTrue(((q[np.isfinite(q)]>=0)&(q[np.isfinite(q)]<=1)).all())
        order=np.argsort(p[:4])
        self.assertTrue(np.diff(q[:4][order]).min()>=-1e-12)


if __name__=='__main__':
    unittest.main()
