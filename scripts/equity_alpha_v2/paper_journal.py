"""Local forward paper journal. No broker connection and no assumed fills.

Input JSON contains a dated quote snapshot and explicit raw-share orders.
Every accepted fill must be supplied separately. Corporate actions require
external reconciliation; this small ledger must not be used for live routing.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


def positive(value):
    return isinstance(value,(int,float)) and math.isfinite(value) and value>0


def apply_event(state,event):
    state=copy.deepcopy(state)
    if event['id'] in state['seen_ids']:
        raise ValueError('Duplicate event id')
    day=event['date']
    datetime.strptime(day,'%Y-%m-%d')
    if day<state['last_date']: raise ValueError('Cannot backdate the journal')
    quotes=event['quotes']
    if event['quote_date']!=day: raise ValueError('Stale quote snapshot')
    for s,q in quotes.items():
        if not positive(q): raise ValueError('Invalid raw price: '+s)
    if any(s not in quotes for s,q in state['positions'].items() if q>0):
        raise ValueError('Missing held-stock mark')
    nav=state['cash']+sum(q*quotes[s] for s,q in state['positions'].items() if q>0)
    state['peak_nav']=max(state['peak_nav'],nav)
    drawdown=nav/state['peak_nav']-1
    kind=event['kind']
    if kind=='order':
        symbol,side,quantity=event['symbol'],event['side'],event['quantity']
        if side not in ('BUY','SELL') or not positive(quantity) or quantity%100:
            raise ValueError('Orders require BUY/SELL and positive 100-share lots')
        if symbol not in quotes: raise ValueError('Missing order quote')
        if day>state['last_date']: state['bought_today']={}
        pending=list(state['pending'].values())
        if side=='BUY':
            if drawdown<=-state['drawdown_stop']: raise ValueError('Drawdown buy stop')
            reserved=sum(o['quantity']*o['reference_price']*1.02 for o in pending if o['side']=='BUY')
            if quantity*quotes[symbol]*1.02>state['cash']-reserved:
                raise ValueError('Cash reserve exceeded')
            committed=state['positions'].get(symbol,0)+sum(o['quantity'] for o in pending if o['symbol']==symbol and o['side']=='BUY')
            if (committed+quantity)*quotes[symbol]>nav*state['max_stock_weight']:
                raise ValueError('Single-stock limit exceeded')
        else:
            reserved=sum(o['quantity'] for o in pending if o['symbol']==symbol and o['side']=='SELL')
            available=state['positions'].get(symbol,0)-state['bought_today'].get(symbol,0)-reserved
            if quantity>available: raise ValueError('Position/T+1 limit exceeded')
        state['pending'][event['id']]={k:event[k] for k in ['symbol','side','quantity','date']}
        state['pending'][event['id']]['reference_price']=quotes[symbol]
    elif kind=='fill':
        order=state['pending'][event['order_id']]
        if day!=order['date']: raise ValueError('Order expired; cancel and resubmit')
        quantity,price,fee=event['quantity'],event['price'],event['fee']
        if not positive(quantity) or quantity%100 or quantity>order['quantity'] or not positive(price):
            raise ValueError('Invalid fill')
        if not math.isfinite(fee) or fee<0: raise ValueError('Invalid fee')
        if abs(price/order['reference_price']-1)>.02: raise ValueError('Fill exceeds 2% price collar')
        symbol=order['symbol']; sign=1 if order['side']=='BUY' else -1
        state['cash']-=sign*quantity*price+fee
        state['positions'][symbol]=state['positions'].get(symbol,0)+sign*quantity
        if state['cash']<0 or state['positions'][symbol]<0: raise ValueError('Invalid cash or position')
        if sign==1: state['bought_today'][symbol]=state['bought_today'].get(symbol,0)+quantity
        order['quantity']-=quantity
        if order['quantity']==0: del state['pending'][event['order_id']]
    elif kind=='cancel':
        del state['pending'][event['order_id']]
    elif kind!='mark':
        raise ValueError('Unknown event kind')
    if day>state['last_date'] and kind!='order': state['bought_today']={}
    state['last_date']=day
    state['seen_ids'].append(event['id'])
    state['last_nav']=state['cash']+sum(q*quotes[s] for s,q in state['positions'].items() if q>0)
    return state


def new_state(capital=1000000.):
    if not positive(capital): raise ValueError('Capital must be positive')
    return dict(cash=capital,positions={},pending={},bought_today={},seen_ids=[],last_date='1900-01-01',
                peak_nav=capital,last_nav=capital,max_stock_weight=.05,drawdown_stop=.15)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--journal',type=Path,required=True)
    parser.add_argument('--event',type=Path)
    parser.add_argument('--capital',type=float,default=1e6)
    args=parser.parse_args()
    # Rebuild from immutable event records; rejected events never append.
    state=new_state(args.capital)
    previous='0'*64
    if args.journal.exists():
        for line in args.journal.read_text(encoding='utf-8').splitlines():
            record=json.loads(line)
            if record['initial_capital']!=args.capital: raise ValueError('Initial capital cannot change during replay')
            payload=json.dumps(record['event'],sort_keys=True,separators=(',',':'))
            expected=hashlib.sha256((previous+payload).encode()).hexdigest()
            if expected!=record['hash']: raise ValueError('Journal chain mismatch')
            state=apply_event(state,record['event']); previous=expected
    if args.event:
        event=json.loads(args.event.read_text(encoding='utf-8'))
        state=apply_event(state,event)
        payload=json.dumps(event,sort_keys=True,separators=(',',':'))
        record=dict(event=event,hash=hashlib.sha256((previous+payload).encode()).hexdigest(),
                    initial_capital=args.capital,
                    received_utc=datetime.now(timezone.utc).isoformat())
        args.journal.parent.mkdir(parents=True,exist_ok=True)
        with args.journal.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps(state,indent=2,ensure_ascii=False))


if __name__=='__main__': main()
