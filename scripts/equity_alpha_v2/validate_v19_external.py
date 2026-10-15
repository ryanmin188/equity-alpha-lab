"""Validate independent economic and point-in-time attribution inputs.

Never silently fill missing quotes, weights, dividends, or industry histories.
Accepted inputs remain local; only a compact coverage report is published.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACCOUNT = HERE / 'outputs/v16_improvements/final/Equal8__Buffer30_60/daily_account.csv'
PRIVATE = HERE / 'outputs/v18_attribution'
PUBLIC = ROOT / 'reports/summary'


def read(path: Path, fields: set[str], keys: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={'instrument': str})
    if fields - set(frame.columns):
        raise ValueError(f'{path.name}: missing {sorted(fields - set(frame.columns))}')
    frame['date'] = pd.to_datetime(frame.date, errors='raise')
    if frame.duplicated(keys).any():
        raise ValueError(f'{path.name}: duplicate {keys}')
    if frame.date.isna().any():
        raise ValueError(f'{path.name}: invalid dates')
    return frame


def benchmark(account: pd.DataFrame, index: pd.DataFrame) -> dict:
    for field in ('price_close', 'total_return_close'):
        if not np.isfinite(index[field]).all() or (index[field] <= 0).any():
            raise ValueError(f'Invalid {field}')
    ordinary = account.loc[account.valuation.eq('close')].copy()
    # The saved first date starts at the open; daily close-to-close analysis
    # begins at the second account date. The final account mark is an open.
    aligned = ordinary[['date']].merge(index, on='date', how='left', validate='one_to_one')
    missing = aligned.loc[aligned.total_return_close.isna(), 'date'].dt.strftime('%Y-%m-%d').tolist()
    if missing:
        return {'status': 'coverage_failed', 'missing_dates': missing, 'matched': len(ordinary) - len(missing)}
    first, last = aligned.iloc[0], aligned.iloc[-1]
    ordinary_net = ordinary.loc[ordinary.date > first.date, 'net']
    portfolio_return = float(np.prod(1 + ordinary_net.to_numpy()) - 1)
    benchmark_total_return = last.total_return_close / first.total_return_close - 1
    return {'status': 'ordinary_close_only', 'matched': len(aligned),
            'first_date': str(first.date.date()), 'last_date': str(last.date.date()),
            'portfolio_return_close_to_close': portfolio_return,
            'price_return_close_to_close': last.price_close / first.price_close - 1,
            'total_return_close_to_close': benchmark_total_return,
            'portfolio_minus_total_return_pp': 100 * (portfolio_return - benchmark_total_return),
            'dividend_return_gap_pp': 100 * (last.total_return_close / first.total_return_close
                                             - last.price_close / first.price_close)}


def events(raw: pd.DataFrame, actions: pd.DataFrame, held: pd.DataFrame) -> pd.DataFrame:
    if (raw.raw_close <= 0).any() or (actions.cash_dividend_per_share < 0).any():
        raise ValueError('Nonpositive raw close or negative cash dividend')
    if (held.ending_adjusted_units <= 0).any():
        raise ValueError('Held units must be positive')
    raw = raw.sort_values(['instrument', 'date']).copy()
    raw['previous_raw_close'] = raw.groupby('instrument').raw_close.shift()
    raw['previous_quote_date'] = raw.groupby('instrument').date.shift()
    # Event inputs must specify independent raw quotes on both adjacent
    # trading dates; no Qlib-implied raw price is accepted here.
    out = actions.merge(raw, on=['instrument', 'date'], how='left', validate='one_to_one')
    out = out.merge(held[['date', 'instrument', 'ending_adjusted_units']],
                    on=['date', 'instrument'], how='inner', validate='one_to_one')
    out['raw_plus_gross_dividend_return'] = (out.raw_close + out.cash_dividend_per_share) / out.previous_raw_close - 1
    out['quote_coverage'] = np.where(out[['raw_close', 'previous_raw_close']].notna().all(axis=1), 'present_unverified_adjacent', 'missing')
    # Adjusted units cannot be called actual shares. Real cash is therefore
    # deliberately not calculated from this table.
    return out


def sector_gate(weights: pd.DataFrame, industry: pd.DataFrame, calendar: pd.Series) -> dict:
    if (weights.weight_pct < 0).any() or (weights.weight_pct > 100).any():
        raise ValueError('Invalid benchmark constituent weights')
    sums = weights.groupby('date').weight_pct.sum()
    bad = sums.loc[(sums - 100).abs() > 0.5]
    if len(bad):
        raise ValueError(f'Index weights do not sum near 100% on {len(bad)} dates')
    if (industry.valid_from > industry.valid_to).any():
        raise ValueError('Invalid industry validity interval')
    overlaps = 0
    for _, group in industry.sort_values('valid_from').groupby('instrument'):
        if (group.valid_from.iloc[1:].to_numpy() <= group.valid_to.iloc[:-1].to_numpy()).any():
            overlaps += 1
    if overlaps:
        raise ValueError(f'Overlapping industry histories for {overlaps} securities')
    required_months = set(calendar.dt.to_period('M').astype(str))
    supplied_months = set(weights.date.dt.to_period('M').astype(str))
    classifications = weights[['date', 'instrument']].merge(
        industry[['instrument', 'industry', 'valid_from', 'valid_to']],
        on='instrument', how='left')
    valid = classifications.date.between(classifications.valid_from, classifications.valid_to)
    unmatched = weights[['date', 'instrument']].merge(
        classifications.loc[valid, ['date', 'instrument']].drop_duplicates(),
        on=['date', 'instrument'], how='left', indicator=True)
    missing_classifications = int(unmatched._merge.ne('both').sum())
    return {'status': 'input_coverage_partial' if not required_months - supplied_months and not missing_classifications else 'coverage_failed',
            'monthly_weight_snapshots': int(len(sums)),
            'missing_months': sorted(required_months - supplied_months),
            'constituents_without_dated_industry': missing_classifications,
            'industry_intervals': int(len(industry))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-quotes', type=Path)
    parser.add_argument('--actions', type=Path)
    parser.add_argument('--csi300-index', type=Path)
    parser.add_argument('--historical-weights', type=Path)
    parser.add_argument('--historical-industries', type=Path)
    args = parser.parse_args()
    if bool(args.raw_quotes) != bool(args.actions):
        parser.error('--raw-quotes and --actions must be supplied together')
    if bool(args.historical_weights) != bool(args.historical_industries):
        parser.error('--historical-weights and --historical-industries must be supplied together')
    account = pd.read_csv(ACCOUNT, parse_dates=['date'])
    result: dict = {'account_end': str(account.date.max().date()), 'datasets': {},
                    'benchmark': {'status': 'missing_external_daily_total_return_levels'},
                    'corporate_actions': {'status': 'one_independent_quote_event_only',
                                          'actual_share_cash_reconciliation': 'unavailable_without_raw_share_ledger'},
                    'sector': {'status': 'missing_point_in_time_weights_and_industries',
                               'brinson_status': 'not_computed'},
                    'style': {'status': 'missing_historical_market_cap_and_point_in_time_accounting'}}
    inputs = {key: value for key, value in vars(args).items() if value is not None}
    for name, path in inputs.items():
        result['datasets'][name] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                                    'path_name': path.name}
    if args.csi300_index:
        index = read(args.csi300_index, {'date', 'price_close', 'total_return_close'}, ['date'])
        result['benchmark'] = benchmark(account, index)
    if args.raw_quotes and args.actions:
        raw = read(args.raw_quotes, {'date', 'instrument', 'raw_close'}, ['date', 'instrument'])
        actions = read(args.actions, {'date', 'instrument', 'cash_dividend_per_share'}, ['date', 'instrument'])
        held = pd.read_csv(PRIVATE / 'daily_security_pnl_private.csv', parse_dates=['date'])
        cases = events(raw, actions, held)
        PRIVATE.mkdir(parents=True, exist_ok=True)
        cases.to_csv(PRIVATE / 'independent_event_checks_private.csv', index=False)
        result['corporate_actions'] = {'held_events': len(cases),
                                       'present_quote_pairs': int(cases.quote_coverage.eq('present_unverified_adjacent').sum()),
                                       'actual_share_cash_reconciliation': 'unavailable_without_raw_share_ledger'}
    if args.historical_weights and args.historical_industries:
        weights = read(args.historical_weights, {'date', 'instrument', 'weight_pct'}, ['date', 'instrument'])
        industry = pd.read_csv(args.historical_industries, dtype={'instrument': str}, parse_dates=['valid_from', 'valid_to'])
        if {'instrument', 'industry', 'valid_from', 'valid_to'} - set(industry.columns):
            raise ValueError('Industry history columns are incomplete')
        result['sector'] = sector_gate(weights, industry, account.date)
        result['sector']['brinson_status'] = 'not_computed_without_aligned_benchmark_security_returns'
    PUBLIC.mkdir(parents=True, exist_ok=True)
    (PUBLIC / 'v19_external_coverage.json').write_bytes(
        json.dumps(result, indent=2, ensure_ascii=False, default=str).encode('utf-8'))
    print(json.dumps(result, ensure_ascii=False, default=str))


if __name__ == '__main__':
    main()
