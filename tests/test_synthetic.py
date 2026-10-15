"""Deterministic fake prices; no Qlib dataset, network or broker needed.

Run: python -m unittest discover -s tests -v
These tests check timing and ledger mechanics, not trading profitability.
"""
from pathlib import Path
import contextlib
import importlib.util
import io
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/equity_alpha_v2/run_research.py'
spec = importlib.util.spec_from_file_location('research', SCRIPT)
research = importlib.util.module_from_spec(spec)
spec.loader.exec_module(research)


@contextlib.contextmanager
def temporary_output():
    # Keep generated files inside the repository's ignored output directory.
    root = (SCRIPT.parent / 'outputs/synthetic_tests').resolve()
    root.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.TemporaryDirectory(dir=root)
    target = Path(temporary.name).resolve()
    if target.parent != root:
        raise RuntimeError('Temporary directory escaped the test output root')
    try:
        yield str(target)
    finally:
        temporary.cleanup()


def fake_market():
    symbols = [f'FAKE{i:03d}' for i in range(60)]
    dates = pd.to_datetime(['2021-02-01', '2021-03-01', '2021-04-01'])
    prices = pd.DataFrame({'open': 10.0}, index=pd.MultiIndex.from_product(
        [dates, symbols], names=['datetime', 'instrument']))
    # First-month holding cannot be sold at the second rebalance.
    prices.loc[(dates[1], symbols[0]), 'open'] = np.nan
    rows = []
    for month, signal in enumerate(pd.to_datetime(['2021-01-29', '2021-02-26'])):
        ranking = symbols if month == 0 else symbols[30:] + symbols[:30]
        for rank, symbol in enumerate(ranking):
            rows.append(dict(date=signal, instrument=symbol, model='EqualFactor',
                             entry_date=dates[month], exit_date=dates[month + 1],
                             score=60-rank, forward_return=0.0))
    return pd.DataFrame(rows), prices, pd.Series(100.0, index=dates)


class SyntheticPipelineTests(unittest.TestCase):
    def run_account(self, folder, predictions, prices, benchmark):
        with patch.object(research, 'SAVE_DIR', Path(folder)), \
             patch.object(research, 'MODELS', ['EqualFactor']), \
             contextlib.redirect_stdout(io.StringIO()):
            return research.simulate(predictions, prices, benchmark)

    def test_blocked_sale_and_independent_cash_replay(self):
        predictions, prices, benchmark = fake_market()
        with temporary_output() as folder:
            result = self.run_account(folder, predictions, prices, benchmark)
            orders = pd.read_csv(Path(folder) / 'execution_audit.csv')
            for strategy, monthly in result.groupby('strategy'):
                cash = 1.0
                positions = {}
                events = orders[orders.strategy == strategy]
                for row in events.itertuples():
                    if row.status.startswith('BLOCKED'):
                        self.assertEqual(row.notional, 0.0)
                        if row.side == 'SELL':
                            self.assertGreater(positions[row.instrument], 0)
                        continue
                    fee = row.notional * research.SINGLE_SIDE_COST_BPS / 10000
                    sign = 1 if row.side == 'BUY' else -1
                    cash -= sign * row.notional + fee
                    positions[row.instrument] = positions.get(row.instrument, 0) + sign * row.shares
                    self.assertGreaterEqual(cash, -1e-10)
                    self.assertGreaterEqual(positions[row.instrument], -1e-10)
                self.assertTrue(all(abs(q) < 1e-10 for q in positions.values()))
                self.assertAlmostEqual(cash, monthly.iloc[-1].wealth, places=10)
                self.assertLess(cash, 1.0)  # flat prices lose the fees
                self.assertEqual(monthly.iloc[-1].cash_weight, 1.0)
                blocked = events[events.status == 'BLOCKED_NO_OPEN']
                self.assertTrue(((blocked.side == 'SELL') & (blocked.instrument == 'FAKE000')).any())

    def test_future_outcomes_do_not_change_trades(self):
        predictions, prices, benchmark = fake_market()
        with temporary_output() as first, temporary_output() as second:
            baseline = self.run_account(first, predictions, prices, benchmark)
            predictions['forward_return'] = np.nan
            changed = self.run_account(second, predictions, prices, benchmark)
            pd.testing.assert_frame_equal(baseline, changed)
            self.assertEqual((Path(first) / 'execution_audit.csv').read_bytes(),
                             (Path(second) / 'execution_audit.csv').read_bytes())

    def test_training_only_receives_mature_labels(self):
        dates = pd.date_range('2017-01-31', periods=6, freq='ME')
        index = pd.MultiIndex.from_product([dates, ['FAKE_A', 'FAKE_B']],
                                         names=['date', 'instrument'])
        panel = pd.DataFrame(0.0, index=index, columns=research.FACTOR_NAMES)
        panel['forward_return'] = 0.01
        panel['entry_date'] = index.get_level_values('date') + pd.Timedelta(days=1)
        panel['exit_date'] = index.get_level_values('date') + pd.DateOffset(months=2)
        calls = []
        def scores(training, today):
            decision = today.index.get_level_values('date')[0]
            self.assertTrue((training.exit_date <= decision).all())
            self.assertTrue((training.index.get_level_values('date') < decision).all())
            calls.append(decision)
            return {'EqualFactor': np.arange(len(today), dtype=float)}
        with temporary_output() as folder, \
             patch.object(research, 'SAVE_DIR', Path(folder)), \
             patch.object(research, 'MIN_TRAIN_MONTHS', 1), \
             patch.object(research, 'model_scores', scores), \
             contextlib.redirect_stdout(io.StringIO()):
            research.generate_oos(panel)
        self.assertGreater(len(calls), 0)


if __name__ == '__main__':
    unittest.main()
