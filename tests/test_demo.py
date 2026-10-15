import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from demo import build_example, run_demo


class SyntheticDemoTests(unittest.TestCase):
    def test_signal_precedes_execution_and_demo_is_deterministic(self):
        predictions, _, _, _ = build_example()
        self.assertTrue((predictions.date < predictions.entry_date).all())
        a, daily_a, orders_a, targets_a = run_demo()
        b, daily_b, orders_b, targets_b = run_demo()
        self.assertEqual(len(targets_a), 4)
        self.assertEqual(len(targets_b), 4)
        self.assertGreater((orders_a.filled_adjusted_units > 0).sum(), 0)
        self.assertAlmostEqual(a["total_return"], b["total_return"])
        self.assertTrue(orders_a.equals(orders_b))
        self.assertTrue(daily_a.equals(daily_b))
        self.assertTrue((daily_a.nav > 0).all())
        self.assertTrue((daily_a.cash >= -1e-6).all())
        self.assertAlmostEqual(daily_a.nav.iloc[-1], a["capital"] * (1 + a["total_return"]))
        self.assertGreater(a["fees_cny"], 0)


if __name__ == "__main__":
    unittest.main()
