import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/equity_alpha_v2'))
from execution_v24 import run_account
from research_v24_search import blend_scores
from test_execution_v22 import market_case


class V24Tests(unittest.TestCase):
    def test_open_orders_cannot_see_same_day_volume_or_close(self):
        p,m,portfolio,c=market_case()
        before=run_account(p,m,portfolio,c)
        day=m.dates[22]
        m.fields['volume'][22]=0
        m.fields['close'][22]*=3
        m.__post_init__()
        after=run_account(p,m,portfolio,c)
        pd.testing.assert_frame_equal(before[2][before[2].date==day].reset_index(drop=True),
                                      after[2][after[2].date==day].reset_index(drop=True))
        self.assertGreater(before[2][before[2].date==day].notional.sum(),0)

    def test_prior_volume_gate_and_capacity_not_double_shifted(self):
        p,m,portfolio,c=market_case()
        m.fields['volume'][21]=0
        m.__post_init__()
        _,_,orders,_,_=run_account(p,m,portfolio,c)
        g=orders[orders.date==m.dates[22]]
        self.assertTrue(g.status.eq('BLOCKED_NO_SESSION_VOLUME').all())
        self.assertEqual(g.notional.sum(),0)

    def test_blend_ignores_labels_and_future_cross_sections(self):
        g=pd.DataFrame({'date':pd.Timestamp('2024-01-31'),'instrument':['A','B','C'],
                        'entry_date':pd.Timestamp('2024-02-01'),'exit_date':pd.Timestamp('2024-03-01'),
                        'raw_vol_60':.02,'forward_return':[.1,-.5,.2],
                        'equal8':[1,2,3],'mom_120':[3,2,1],'mom_252_skip20':[2,3,1]})
        before=blend_scores(g)
        g.forward_return*=100
        future=g.copy();future.date=pd.Timestamp('2025-01-31');future.equal8*=-100
        after=blend_scores(pd.concat([g,future],ignore_index=True))
        np.testing.assert_array_equal(before.score,after[after.date==g.date.iloc[0]].score)


if __name__=='__main__':unittest.main()
