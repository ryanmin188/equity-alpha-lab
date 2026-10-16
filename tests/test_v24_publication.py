import json
import unittest
from pathlib import Path
class CandidatePublicationTests(unittest.TestCase):
    def test_aggregate_results_match_daily_wealth(self):
        data=json.loads((Path(__file__).resolve().parents[1]/'showcase/data/candidate.json').read_text())
        for name,s in data['scenarios'].items():
            self.assertEqual(len(s['daily']),667)
            self.assertTrue(all(len(r)==3 for r in s['daily']))
            self.assertAlmostEqual(s['daily'][-1][1]-1,s['total_return'],places=7)
            previous=[1.,1.]
            for annual in s['annual']:
                rows=[r for r in s['daily'] if int(r[0][:4])==annual['year']]
                self.assertAlmostEqual(rows[-1][1]/previous[0]-1,annual['strategy_return'],places=7)
                self.assertAlmostEqual(rows[-1][2]/previous[1]-1,annual['benchmark_return'],places=7)
                previous=rows[-1][1:]
        for name in ['base','slip15']:
            self.assertTrue(all(r['active_pp']>0 for r in data['scenarios'][name]['annual'][:2]))
        self.assertTrue(all(r['active_pp']<0 for r in data['scenarios']['capital10m']['annual'][:2]))
