import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ShowcaseDatasetTests(unittest.TestCase):
    def test_public_dataset_is_frozen_and_contains_only_aggregates(self):
        data = json.loads((ROOT / "showcase/data/research.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "historical_simulation_read_only")
        self.assertEqual(len(data["daily"]), 667)
        self.assertEqual(len(data["factors"]), 28)
        self.assertEqual(data["daily"][0][0], "2024-01-02")
        self.assertEqual(data["daily"][-1][0], "2026-10-08")
        self.assertAlmostEqual(data["daily"][-1][1] - 1, .377123608, places=8)
        self.assertAlmostEqual(data["daily"][-1][2] - 1, .2710434, places=8)
        self.assertFalse(any(f["bh_q"] < .05 for f in data["factors"]))
        self.assertEqual({r["observations"] for r in data["regressions"]}, {484})
        self.assertEqual(len(data["correlation"]["values"]), 28)
        self.assertEqual(len(data["correlation"]["values"][0]), 28)
        forbidden = ("orders", "holdings", "raw_open", "raw_close", "api_key", "credentials")
        self.assertFalse(any(word in data for word in forbidden))


if __name__ == "__main__":
    unittest.main()
