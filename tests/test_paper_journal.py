import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/equity_alpha_v2'))
from paper_journal import new_state,apply_event


class PaperJournalTests(unittest.TestCase):
    def event(self,id,kind,**kwargs):
        return dict(id=id,kind=kind,date='2026-10-15',quote_date='2026-10-15',quotes={'FAKE':10.},**kwargs)

    def test_no_automatic_fill_and_t_plus_one(self):
        s=apply_event(new_state(),self.event('order1','order',symbol='FAKE',side='BUY',quantity=100))
        self.assertEqual(s['cash'],1e6)
        self.assertEqual(s['positions'],{})
        fill=self.event('fill1','fill',order_id='order1',quantity=100,price=10.,fee=5.)
        s=apply_event(s,fill)
        self.assertEqual(s['cash'],998995.)
        with self.assertRaisesRegex(ValueError,'Duplicate'): apply_event(s,fill)
        with self.assertRaisesRegex(ValueError,r'T\+1'):
            apply_event(s,self.event('sell','order',symbol='FAKE',side='SELL',quantity=100))
        sell=self.event('sell','order',symbol='FAKE',side='SELL',quantity=100)
        sell.update(date='2026-10-16',quote_date='2026-10-16')
        self.assertIn('sell',apply_event(s,sell)['pending'])

    def test_limits_staleness_and_drawdown_stop(self):
        with self.assertRaisesRegex(ValueError,'Single-stock'):
            apply_event(new_state(),self.event('large','order',symbol='FAKE',side='BUY',quantity=5100))
        event=self.event('stale','mark'); event['quote_date']='2026-10-14'
        with self.assertRaisesRegex(ValueError,'Stale'): apply_event(new_state(),event)
        s=new_state(); s['peak_nav']=2e6
        with self.assertRaisesRegex(ValueError,'Drawdown'):
            apply_event(s,self.event('stop','order',symbol='FAKE',side='BUY',quantity=100))
