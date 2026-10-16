"""Causal signal comparisons on opened history; never selects a frozen strategy.

Only the local ignored output directory receives market-dependent artifacts.
GitHub references motivate feature families/evaluation, not copied returns/code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from execution_v22 import run_account
from research_v16_improvements import BASE

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'outputs/v16_improvements'
OUT = HERE / 'outputs/v23_signal_comparison'
FAMILIES = ['trend', 'reversal', 'defensive', 'liquidity', 'price_volume', 'residual']
MODELS = ['B0_Equal8', 'S1_Trend', 'S2_BalancedTrend', 'S3_ResidualTrend',
          'S4_BalancedResidual', 'S5_Adaptive24', 'S6_Adaptive60']
PORTFOLIO = {'name': 'Buffer30_60', 'top_n': 30, 'buffer_n': 60, 'weighting': 'equal'}


def centered_rank(x):
    return (pd.Series(x).rank(pct=True) * 2 - 1).fillna(0).to_numpy()


def residual_signal(stock_returns, market_returns):
    """Past-only daily market residual analogue, not a replication of FF3 paper."""
    r = np.asarray(stock_returns, float)
    m = np.asarray(market_returns, float)
    if r.shape != (252,) or m.shape != (252,):
        raise ValueError('Exactly 252 prior daily observations required')
    ok = np.isfinite(r) & np.isfinite(m)
    if ok.sum() < 202 or np.std(m[ok]) < 1e-10:
        return np.nan
    design = np.column_stack([np.ones(252), m])
    coef = np.linalg.lstsq(design[ok], r[ok], rcond=None)[0]
    residual = r - design @ coef
    window = residual[-120:-20]
    good = window[np.isfinite(window)]
    if len(good) < 80 or good.std(ddof=1) < 1e-10:
        return np.nan
    return float(good.sum() / good.std(ddof=1))


def build_families(panel, market):
    if panel.duplicated(['date', 'instrument']).any():
        raise ValueError('Duplicate signal/security')
    if not ((panel.date < panel.entry_date) & (panel.entry_date < panel.exit_date)).all():
        raise ValueError('Invalid time boundaries')
    close = pd.DataFrame(market.fields['close'])
    returns = close.pct_change(fill_method=None).to_numpy()
    benchmark = pd.Series(market.benchmark_close).pct_change(fill_method=None).to_numpy()
    blocks = []
    for date, frame in panel.groupby('date', sort=True):
        g = frame.copy()
        i = market.date_index[pd.Timestamp(date)]
        if i < 252:
            raise ValueError('Insufficient past data')
        values = [residual_signal(returns[i-251:i+1, market.symbol_index[s]],
                                  benchmark[i-251:i+1]) for s in g.instrument]
        g['residual_available'] = np.isfinite(values)
        g['residual'] = centered_rank(values)
        g['trend'] = centered_rank(g[['mom_120', 'mom_252_skip20']].mean(axis=1))
        g['reversal'] = centered_rank((g.reversal_5 - g.mom_20) / 2)
        g['defensive'] = centered_rank(-g[['vol_60', 'downside_vol60']].mean(axis=1))
        g['liquidity'] = centered_rank((g.log_dollar_volume - g.amihud60) / 2)
        g['price_volume'] = centered_rank((g.overnight20 - g.intraday20 - g.return_volume_corr20) / 3)
        g['equal8'] = g[BASE].fillna(0).to_numpy() @ np.array([1,1,1,-1,-1,-1,-1,1]) / 8
        blocks.append(g)
    return pd.concat(blocks, ignore_index=True)


def mature_weights(history, date, lookback):
    """No current or immature outcomes may affect a decision."""
    valid = history[(history.date < date) & (history.exit_date <= date)].tail(lookback)
    if len(valid) < 18:
        return None, valid
    values = valid[FAMILIES]
    weights = (values.mean() / np.sqrt(values.var(ddof=1) + .01)).fillna(0).to_numpy()
    norm = np.abs(weights).sum()
    return (weights / norm if norm > 1e-12 else np.zeros(len(FAMILIES))), valid


def predict(families):
    ic_rows = []
    for date, g in families.groupby('date', sort=True):
        row = {'date': date, 'exit_date': g.exit_date.max()}
        for name in FAMILIES:
            ok = g[[name, 'forward_return']].dropna()
            row[name] = ok[name].corr(ok.forward_return, method='spearman') if len(ok) >= 30 else np.nan
        ic_rows.append(row)
    history = pd.DataFrame(ic_rows)
    output, audits = [], []
    for date, g in families.groupby('date', sort=True):
        if date < pd.Timestamp('2016-12-01'):
            continue
        eq = centered_rank(g.equal8)
        scores = {'B0_Equal8': g.equal8.to_numpy(), 'S1_Trend': g.trend.to_numpy(),
                  'S2_BalancedTrend': .5 * (eq + g.trend.to_numpy()),
                  'S3_ResidualTrend': g.residual.to_numpy(),
                  'S4_BalancedResidual': .5 * (eq + g.residual.to_numpy())}
        for model, window in [('S5_Adaptive24', 24), ('S6_Adaptive60', 60)]:
            weights, train = mature_weights(history, date, window)
            if weights is None:
                raise ValueError(f'Insufficient mature history: {date}')
            scores[model] = g[FAMILIES].fillna(0).to_numpy() @ weights
            audits.append({'date': date, 'model': model, 'n_train_months': len(train),
                           'latest_train_exit': train.exit_date.max(),
                           **dict(zip(FAMILIES, weights))})
        for model, score in scores.items():
            part = g[['date','instrument','entry_date','exit_date','forward_return','raw_vol_60']].copy()
            part['model'] = model
            part['score'] = score
            output.append(part)
    return pd.concat(output, ignore_index=True), pd.DataFrame(audits), history


def annual_metrics(daily):
    """Calendar returns from a continuous account; never restart at Jan 1."""
    d = daily.copy().sort_values('date')
    if d.date.duplicated().any():
        raise ValueError('Duplicate account date')
    d['benchmark_net'] = d.benchmark_wealth / d.benchmark_wealth.shift(1).fillna(1) - 1
    if not np.allclose((1+d.net).cumprod(), d.wealth, atol=1e-10, rtol=1e-10):
        raise ValueError('Daily returns do not reconstruct wealth')
    rows = []
    for year, g in d.groupby(pd.to_datetime(d.date).dt.year):
        net = float((1+g.net).prod()-1)
        benchmark = float((1+g.benchmark_net).prod()-1)
        rows.append({'year': int(year), 'strategy_return': net, 'benchmark_return': benchmark,
                     'active_pp': net-benchmark, 'n_days': len(g),
                     'start': g.date.iloc[0], 'end': g.date.iloc[-1],
                     'max_marked_weight': g.max_weight.max(),
                     'mean_cash_weight': g.cash_weight.mean()})
    return pd.DataFrame(rows)


def evaluate(predictions, market, out, stress=False):
    config = json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    periods = {'development_2017_2020': ('2017-01-01','2021-02-01'),
               'reused_validation_2021_2023': ('2021-01-01','2024-02-01'),
               'opened_2024_2026': ('2024-01-01','2026-10-08')}
    settings = [('base', 5., 1e6)] if not stress else [('slip15', 15., 1e6), ('capital10m', 5., 1e7)]
    summaries, years = [], []
    for period, (start, end) in periods.items():
        if stress and period != 'opened_2024_2026':
            continue
        for model in MODELS:
            sub = predictions[(predictions.model == model) & (predictions.entry_date >= start)
                              & (predictions.exit_date <= end)].copy()
            for scenario, slip, capital in settings:
                folder = out/period/model/scenario
                folder.mkdir(parents=True, exist_ok=True)
                summary, daily, orders, targets, exposures = run_account(
                    sub, market, PORTFOLIO, config, regime='E2', slippage_bps=slip, capital=capital)
                for name, data in [('daily_account',daily),('orders',orders),('targets',targets),('exposures',exposures)]:
                    data.to_csv(folder/f'{name}.csv',index=False)
                annual = annual_metrics(daily)
                annual['model'], annual['period'], annual['scenario'] = model, period, scenario
                years.append(annual)
                summaries.append({'model':model,'period':period,'scenario':scenario,**summary})
                print(period, model, scenario, annual[['year','strategy_return','active_pp']].round(4).to_dict('records'),flush=True)
    suffix = 'stress' if stress else 'base'
    pd.DataFrame(summaries).to_csv(out/f'{suffix}_summary.csv',index=False)
    pd.concat(years,ignore_index=True).to_csv(out/f'{suffix}_annual.csv',index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stress',action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    with (SOURCE/'market_final.pkl').open('rb') as f:
        market,_ = pickle.load(f)
    if args.stress:
        predictions = pd.read_pickle(OUT/'predictions_private.pkl')
    else:
        panel = pd.read_pickle(SOURCE/'panel_final.pkl')
        for c in ('date','entry_date','exit_date'):
            panel[c] = pd.to_datetime(panel[c])
        families = build_families(panel, market)
        predictions, audit, ics = predict(families)
        families.to_pickle(OUT/'family_panel_private.pkl')
        predictions.to_pickle(OUT/'predictions_private.pkl')
        audit.to_csv(OUT/'training_audit.csv',index=False)
        ics.to_csv(OUT/'family_monthly_ic.csv',index=False)
        files = [SOURCE/'panel_final.pkl', SOURCE/'market_final.pkl', ROOT/'configs/v23_signal_protocol.json', Path(__file__)]
        manifest = {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        (OUT/'input_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    evaluate(predictions,market,OUT,stress=args.stress)


if __name__ == '__main__':
    main()
