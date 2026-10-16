"""Publish aggregate candidate NAV/annual metrics only; keep private books local."""
import hashlib
import json

import numpy as np
import pandas as pd

from research_v24_search import OUT,ROOT
from research_v23_signals import annual_metrics


def main():
    verification=json.loads((OUT/'candidate_verification.json').read_text(encoding='utf-8'))
    if verification['future_mutation_changes_historical_features'] or verification['future_mutation_changes_historical_orders']:
        raise ValueError('Causal verification failed')
    model='mom_252_skip20_eq25';portfolio='Buffer20_40'
    paths={'base':OUT/'opened_2024_2026'/model/portfolio/'base',
           'slip15':OUT/'stress'/model/portfolio/'opened_2024_2026'/model/portfolio/'slip15',
           'capital10m':OUT/'stress'/model/portfolio/'opened_2024_2026'/model/portfolio/'capital10m'}
    scenarios={}
    for scenario,path in paths.items():
        d=pd.read_csv(path/'daily_account.csv')
        a=annual_metrics(d)
        scenarios[scenario]={'capital':1e7 if scenario=='capital10m' else 1e6,
                            'slippage_bps':15 if scenario=='slip15' else 5,
                            'daily':[[r.date,round(r.wealth,8),round(r.benchmark_wealth,8)] for r in d.itertuples()],
                            'annual':a[['year','strategy_return','benchmark_return','active_pp']].to_dict('records'),
                            'total_return':float(d.wealth.iloc[-1]-1),
                            'max_drawdown':float((np.r_[1,d.wealth]/np.maximum.accumulate(np.r_[1,d.wealth])-1).min()),
                            'max_marked_weight':float(d.max_weight.max()),
                            'source_sha256':hashlib.sha256((path/'daily_account.csv').read_bytes()).hexdigest()}
    public={'name':'TrendBlend20','status':'posthoc_historical_candidate','selected_on':'2026-10-16',
            'formula':'25% Equal8 rank + 75% 252-day momentum excluding latest20 days rank',
            'portfolio':'20 target stocks / retention rank40 / equal weights',
            'base_trials_this_round':12,'previous_round_configurations':30,
            'scenarios':scenarios,'causal_verification':verification,
            'note':'Signals use then-observable information; selecting this candidate used opened historical results. Not independent out-of-sample evidence.'}
    dest=ROOT/'showcase/data/candidate.json'
    dest.write_text(json.dumps(public,ensure_ascii=False,separators=(',',':')),encoding='utf-8')
    base=scenarios['base']['daily']; old=json.loads((ROOT/'showcase/data/research.json').read_text(encoding='utf-8'))['daily']
    assert [r[0] for r in base]==[r[0] for r in old]
    width,height=920,360;left,right,top,bottom=65,30,55,45
    series=[[r[1] for r in base],[r[1] for r in old],[r[2] for r in base]]
    lo=.7;hi=max(max(s) for s in series)+.04
    x=lambda i:left+(width-left-right)*i/(len(base)-1)
    y=lambda v:top+(height-top-bottom)*(hi-v)/(hi-lo)
    svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="Historical candidate, frozen baseline and price benchmark NAV">',
         '<rect width="100%" height="100%" rx="16" fill="#101b2c"/>',
         '<text x="65" y="30" font-family="Arial" font-size="17" fill="#eef6fa">Historical simulation | 2024-01-02 — 2026-10-08</text>']
    for value in np.linspace(.8,1.6,5):
        svg.append(f'<path d="M{left},{y(value):.1f}H{width-right}" stroke="#2c3d52"/><text x="48" y="{y(value)+4:.1f}" text-anchor="end" fill="#a7b7c9" font-family="Arial" font-size="12">{value:.1f}</text>')
    colors=['#70e1c0','#90a9ff','#e9bd72']
    for s,color in zip(series,colors):
        points=' '.join(f'{x(i):.1f},{y(v):.1f}' for i,v in enumerate(s))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2"/>')
    for xpos,label,color in [(65,'TrendBlend20 (posthoc)',colors[0]),(365,'Equal8 (frozen)',colors[1]),(650,'CSI300 price index',colors[2])]:
        svg.append(f'<text x="{xpos}" y="342" fill="{color}" font-family="Arial" font-size="13">{label}</text>')
    svg.append('</svg>')
    (ROOT/'assets/v24_comparison.svg').write_text('\n'.join(svg),encoding='utf-8')
    print('Published aggregate candidate data and NAV chart; private accounts excluded')


if __name__=='__main__':main()
