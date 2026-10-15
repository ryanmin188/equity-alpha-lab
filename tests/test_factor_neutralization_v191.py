import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/equity_alpha_v2'))
from factor_neutralization_v191 import (neutralize_cross_section,
                                        validate_historical_exposures,
                                        synthetic_barra_risk_example)


class NeutralizationTests(unittest.TestCase):
    def synthetic(self):
        rng=np.random.default_rng(42)
        n=120
        cap=pd.Series(np.exp(np.linspace(10,20,n)),index=[f'S{i:03d}' for i in range(n)])
        industry=pd.Series(['bank']*60+['utility']*60,index=cap.index)
        noise=rng.normal(0,1,n)
        factor=pd.Series(2+1.7*np.log(cap.to_numpy())+
                         np.where(industry.eq('utility'),3.,0.)+noise,index=cap.index)
        return factor,cap,industry

    def test_orthogonality_and_industry_conditional_mean(self):
        factor,cap,industry=self.synthetic()
        z,info=neutralize_cross_section(factor,market_cap=cap,industry=industry)
        self.assertEqual(info['n_used'],120)
        self.assertLess(info['max_normal_equation_residual'],1e-8)
        self.assertAlmostEqual(z.mean(),0.,places=12)
        self.assertAlmostEqual(z.std(ddof=0),1.,places=12)
        self.assertAlmostEqual(z.loc[industry.eq('bank')].mean(),0.,places=12)
        self.assertAlmostEqual(z.loc[industry.eq('utility')].mean(),0.,places=12)
        self.assertAlmostEqual(np.dot(z.to_numpy(),np.log(cap.to_numpy())),0.,places=9)

    def test_missing_exposure_rejected_or_reported(self):
        factor,cap,industry=self.synthetic()
        cap.iloc[3]=np.nan
        with self.assertRaisesRegex(ValueError,'missing'):
            neutralize_cross_section(factor,market_cap=cap,industry=industry)
        _,info=neutralize_cross_section(factor,market_cap=cap,industry=industry,missing='drop')
        self.assertEqual(info['n_missing_dropped'],1)

    def test_single_industry_constant_cap_and_rank_deficiency(self):
        factor,cap,industry=self.synthetic()
        same_cap=pd.Series([100.]*len(cap),index=cap.index)
        same_industry=pd.Series(['one']*len(cap),index=cap.index)
        _,info=neutralize_cross_section(factor,market_cap=same_cap,industry=same_industry)
        self.assertIn('constant_log_market_cap_dropped',info['notes'])
        self.assertIn('single_industry_no_dummy',info['notes'])
        # log cap is exactly a linear function of two-industry dummy.
        cap2=pd.Series(np.where(industry.eq('bank'),100.,200.),index=cap.index)
        _,singular=neutralize_cross_section(factor,market_cap=cap2,industry=industry)
        self.assertEqual(len(singular['dropped_columns']),1)
        self.assertTrue(np.isfinite(singular['condition_number']))

    def test_insufficient_sample(self):
        with self.assertRaisesRegex(ValueError,'Insufficient'):
            neutralize_cross_section(pd.Series([1.,2.]),market_cap=pd.Series([10.,20.]))

    def test_industry_timing_and_market_cap_timing(self):
        frame=pd.DataFrame({'date':['2024-01-31'],'instrument':['SH600000'],
                            'industry':['bank'],'market_cap':[1e10],
                            'industry_valid_from':['2024-02-01'],
                            'industry_valid_to':['2025-12-31'],
                            'source_id':['vendor_snapshot_1'],
                            'market_cap_asof':['2024-01-31'],
                            'market_cap_unit':['CNY']})
        signal=frame[['date','instrument']]
        with self.assertRaisesRegex(ValueError,'not effective'):
            validate_historical_exposures(frame,signal)
        frame.loc[0,'industry_valid_from']='2024-01-01'
        frame.loc[0,'market_cap_asof']='2024-02-01'
        with self.assertRaisesRegex(ValueError,'after signal'):
            validate_historical_exposures(frame,signal)
        frame.loc[0,'market_cap_asof']='2024-01-31'
        result=validate_historical_exposures(frame,signal)
        self.assertEqual(len(result),1)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            validate_historical_exposures(pd.concat([frame,frame]),signal)
        mixed=pd.concat([frame,frame.assign(instrument='SH600001',market_cap_unit='CNY_10000')],ignore_index=True)
        with self.assertRaisesRegex(ValueError,'Mixed market-cap units'):
            validate_historical_exposures(mixed,mixed[['date','instrument']])

    def test_synthetic_risk_covariance(self):
        result=synthetic_barra_risk_example()
        x,f,d,sigma=(result[k] for k in ('X','F','D','sigma'))
        np.testing.assert_allclose(sigma,x@f@x.T+d)
        self.assertTrue(result['symmetric'])
        self.assertGreater(result['minimum_eigenvalue'],0.)
        self.assertGreater(result['portfolio_daily_vol'],0.)
        self.assertGreater(result['active_daily_tracking_error'],0.)


if __name__=='__main__':
    unittest.main()
