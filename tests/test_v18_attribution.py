"""Synthetic accounting identities for the read-only V1.8 replay."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'equity_alpha_v2'))
from execution_v16 import Market
from attribute_v18_holdings import replay


class AttributionReplayTest(unittest.TestCase):
    def test_buy_mark_sell_and_fee_basis(self):
        dates = pd.to_datetime(['2024-01-02','2024-01-03','2024-01-04'])
        fields = {'open':np.array([[10.],[11.],[13.]]),
                  'close':np.array([[11.],[12.],[13.]]),
                  'volume':np.full((3,1),10000.),
                  'factor':np.ones((3,1))}
        market = Market(dates,['SH600000'],fields,
                        np.array([100.,101.,102.]),np.array([101.,102.,103.]))
        orders = pd.DataFrame([
            {'date':dates[0],'instrument':'SH600000','side':'BUY','status':'FILLED',
             'filled_adjusted_units':100.,'notional':1000.,'fees':5.,'slippage_impact':0.},
            {'date':dates[1],'instrument':'SH600000','side':'SELL','status':'FILLED',
             'filled_adjusted_units':40.,'notional':440.,'fees':3.,'slippage_impact':0.}])
        # Day 1: cash 998995, marked shares 1100 => 1000095.
        # Day 2: cash 999432, marked shares 720 => 1000152.
        # Day 3: final open 13, 60 shares => 1000212.
        nav=[1000095.,1000152.,1000212.]
        daily=pd.DataFrame({'date':dates,'nav':nav,'cash':[998995.,999432.,999432.],
                            'stale_positions':[0,0,0],
                            'valuation':['close','close','open'],
                            'benchmark_wealth':[1.01,1.02,1.02]})
        ledger,checks,stale=replay(daily,orders,market)
        self.assertAlmostEqual(ledger.gross_pnl.sum(),220.)
        self.assertAlmostEqual(ledger.order_fees.sum(),8.)
        self.assertAlmostEqual(ledger.realized_pnl.sum(),40.)
        self.assertAlmostEqual(ledger.change_unrealized_pnl.sum(),180.)
        self.assertAlmostEqual(ledger.net_pnl.sum(),212.)
        self.assertAlmostEqual(checks.active_log.sum(),np.log(1000212/1_000_000/1.02))
        self.assertTrue(stale.empty)


if __name__ == '__main__':
    unittest.main()
