import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts/equity_alpha_v2'))
from diagnose_v19_ch4 import diagnostics
from record_forward_signal import make_signal
from validate_v19_external import benchmark, events, sector_gate


class ExternalDataGateTests(unittest.TestCase):
    def test_total_return_boundary_excludes_first_open_and_terminal_open(self):
        account = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-03', '2024-01-04']),
                                'valuation': ['close', 'close', 'open'], 'net': [.02, .03, -.01]})
        index = pd.DataFrame({'date': pd.to_datetime(['2024-01-02', '2024-01-03']),
                              'price_close': [100, 102], 'total_return_close': [100, 103]})
        result = benchmark(account, index)
        self.assertAlmostEqual(result['portfolio_return_close_to_close'], .03)
        self.assertAlmostEqual(result['total_return_close_to_close'], .03)
        self.assertAlmostEqual(result['dividend_return_gap_pp'], 1.)

    def test_event_rejects_missing_prior_independent_quote(self):
        date = pd.Timestamp('2024-07-17')
        raw = pd.DataFrame({'date': [date], 'instrument': ['SH601988'], 'raw_close': [4.66]})
        actions = pd.DataFrame({'date': [date], 'instrument': ['SH601988'],
                                'cash_dividend_per_share': [.2364]})
        held = pd.DataFrame({'date': [date], 'instrument': ['SH601988'],
                             'ending_adjusted_units': [100.]})
        result = events(raw, actions, held)
        self.assertEqual(result.quote_coverage.iloc[0], 'missing')

    def test_sector_gate_requires_historical_months(self):
        weights = pd.DataFrame({'date': pd.to_datetime(['2024-01-31']),
                                'instrument': ['A'], 'weight_pct': [100.]})
        industry = pd.DataFrame({'instrument': ['A'], 'industry': ['bank'],
                                 'valid_from': pd.to_datetime(['2020-01-01']),
                                 'valid_to': pd.to_datetime(['2025-12-31'])})
        result = sector_gate(weights, industry, pd.Series(pd.to_datetime(['2024-01-31', '2024-02-29'])))
        self.assertEqual(result['status'], 'coverage_failed')
        self.assertEqual(result['missing_months'], ['2024-02'])

    def test_forward_rejects_old_snapshot(self):
        frame = pd.DataFrame({'date': ['2026-10-08'], 'instrument': ['A']})
        with self.assertRaises(ValueError):
            make_signal(frame, pd.Timestamp('2026-10-08'))

    def test_forward_scores_use_frozen_equal8_signs(self):
        names = ['mom_20', 'mom_60', 'reversal_5', 'vol_20', 'vol_60',
                 'volume_surprise', 'range_20', 'ma_gap_20']
        data = {'date': ['2026-10-30'] * 150,
                'instrument': [f'S{i:03d}' for i in range(150)],
                'point_in_time_member': [True] * 150,
                'valid_closes_252': [252] * 150,
                'raw_close': [10.] * 150,
                'raw_volume': [1000.] * 150,
                'adjustment_factor': [1.] * 150,
                'feature_coverage_28': [1.] * 150}
        for name in names:
            data[name] = list(range(149, -1, -1)) if name in {
                'vol_20', 'vol_60', 'volume_surprise', 'range_20'
            } else list(range(150))
        result = make_signal(pd.DataFrame(data), pd.Timestamp('2026-10-30'))
        self.assertEqual(len(result), 150)
        self.assertEqual(result.instrument.iloc[0], 'S149')

    def test_published_independent_event_arithmetic(self):
        path = Path(__file__).resolve().parent / 'fixtures/v19_independent_event_case.csv'
        row = pd.read_csv(path).iloc[0]
        gross = (row.independent_raw_ex_close + row.issuer_gross_dividend_per_share) / row.independent_raw_prev_close - 1
        self.assertAlmostEqual(gross, row.raw_plus_gross_dividend_return)
        self.assertAlmostEqual((row.qlib_adjusted_return - gross) * 10000,
                               row.qlib_minus_external_gross_return_bp)

    def test_ch4_fixed_sample_rejects_short_input(self):
        with self.assertRaises(ValueError):
            diagnostics(pd.DataFrame({'date': pd.to_datetime(['2024-01-02']),
                                      'mktrf': [0.], 'SMB': [0.], 'VMG': [0.],
                                      'PMO': [0.], 'portfolio_excess': [0.]}))


if __name__ == '__main__':
    unittest.main()
