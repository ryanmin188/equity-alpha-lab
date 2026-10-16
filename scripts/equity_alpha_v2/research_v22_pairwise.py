"""One prespecified pairwise logistic ranking diagnostic, no trading account."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression, Ridge

from research_v16_improvements import BASE
from research_v21_alternatives import equal8_scores

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs/v22_execution_reality'
PAIRS_PER_MONTH=150
LOOKBACK_MONTHS=60
MIN_MONTHS=36


def make_block(group,date):
    """Ranks are fitted within the signal-date cross-section only."""
    g=group.reset_index(drop=True)
    x=g[BASE].rank(pct=True).fillna(.5).to_numpy(float)-.5
    y=g.forward_return.rank(pct=True).to_numpy(float)-.5
    valid=np.flatnonzero(np.isfinite(y))
    rng=np.random.default_rng(int(pd.Timestamp(date).strftime('%Y%m%d')))
    if len(valid)<30:
        pair_x=np.empty((0,len(BASE)));pair_y=np.empty(0,int)
    else:
        a=rng.choice(valid,PAIRS_PER_MONTH)
        b=rng.choice(valid,PAIRS_PER_MONTH)
        ok=(a!=b)&(y[a]!=y[b])
        differences=x[a[ok]]-x[b[ok]]
        labels=(y[a[ok]]>y[b[ok]]).astype(int)
        pair_x=np.vstack([differences,-differences])
        pair_y=np.r_[labels,1-labels]
    return {'date':pd.Timestamp(date),'exit_date':g.exit_date.max(),
            'x':x,'y':y,'pair_x':pair_x,'pair_y':pair_y,
            'equal8':equal8_scores(g)}


def evaluate(blocks):
    rows=[]
    for i,block in enumerate(blocks):
        date=block['date']
        if date.year<2017:continue
        mature=[b for b in blocks[:i] if b['exit_date']<=date and len(b['pair_y'])]
        train=mature[-LOOKBACK_MONTHS:]
        if len(train)<MIN_MONTHS:continue
        train_x=np.vstack([b['pair_x'] for b in train])
        train_y=np.concatenate([b['pair_y'] for b in train])
        model=LogisticRegression(C=1.,fit_intercept=False,solver='liblinear',max_iter=300)
        model.fit(train_x,train_y)
        ridge_x=np.vstack([b['x'][np.isfinite(b['y'])] for b in train])
        ridge_y=np.concatenate([b['y'][np.isfinite(b['y'])] for b in train])
        ridge=Ridge(alpha=100.)
        ridge.fit(ridge_x,ridge_y)
        valid=np.isfinite(block['y'])
        if valid.sum()<30:continue
        observed=block['y'][valid]
        period=('development_2017_2020' if date.year<=2020 else
                'reused_validation_2021_2023' if date.year<=2023 else 'opened_2024_2026')
        scores={'PairwiseLogistic':block['x']@model.coef_[0],
                'DirectRidge':ridge.predict(block['x']),
                'Equal8':block['equal8']}
        for name,score in scores.items():
            ic=float(spearmanr(score[valid],observed).statistic) if np.std(score[valid])>0 else np.nan
            rows.append({'date':date,'period':period,'model':name,'rank_ic':ic,
                         'n_stocks':int(valid.sum()),'mature_train_months':len(train),
                         'latest_train_exit':max(b['exit_date'] for b in train)})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    panel=pd.read_pickle(HERE/'outputs/v16_improvements/panel_final.pkl')
    for col in ('date','entry_date','exit_date'):panel[col]=pd.to_datetime(panel[col])
    blocks=[make_block(g,d) for d,g in panel.groupby('date',sort=True)]
    result=evaluate(blocks)
    if not (result.latest_train_exit<=result.date).all():
        raise AssertionError('Immature label leaked into training')
    result.to_csv(OUT/'pairwise_monthly_rank_ic.csv',index=False)
    summary=result.groupby(['period','model']).agg(n_months=('rank_ic','count'),
        mean_rank_ic=('rank_ic','mean'),positive_share=('rank_ic',lambda v:(v>0).mean()),
        mean_stocks=('n_stocks','mean')).reset_index()
    summary.to_csv(OUT/'pairwise_summary.csv',index=False)
    print(summary.to_string(index=False))


if __name__=='__main__':main()
