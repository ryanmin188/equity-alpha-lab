"""Append a frozen Equal8 signal before any future return is observed.

The input is an independently prepared point-in-time feature snapshot. This
program does not load V1.6 outcome labels, choose a model, or simulate fills.
Full scores/holdings remain in ignored local outputs; public log has hashes.
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
PRIVATE = HERE / 'outputs/forward_signals'
PUBLIC = ROOT / 'reports/summary/v19_forward_signal_registry.csv'
BASE = ['mom_20', 'mom_60', 'reversal_5', 'vol_20', 'vol_60',
        'volume_surprise', 'range_20', 'ma_gap_20']
SIGNS = np.array([1, 1, 1, -1, -1, -1, -1, 1])
LAST_TEST = pd.Timestamp('2026-10-08')


def make_signal(frame: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    required = {'date', 'instrument', 'point_in_time_member', 'valid_closes_252', 'raw_close',
                'raw_volume', 'adjustment_factor', 'feature_coverage_28', *BASE}
    if required - set(frame):
        raise ValueError(f'Missing fields: {sorted(required - set(frame))}')
    frame = frame.copy()
    frame.date = pd.to_datetime(frame.date, errors='raise')
    if frame.date.isna().any() or not frame.date.eq(asof).all() or frame.instrument.duplicated().any():
        raise ValueError('Snapshot needs one as-of date and unique securities')
    if asof <= LAST_TEST or len(frame) < 150:
        raise ValueError('Only future as-of dates and at least 150 securities are accepted')
    features = frame[BASE].apply(pd.to_numeric, errors='raise').replace([np.inf, -np.inf], np.nan)
    # These are the frozen V1.6 eligibility checks. The upstream snapshot
    # builder must additionally use the same 252-day/202 valid-close rule.
    eligible = frame.point_in_time_member.eq(True)
    eligible &= frame.raw_close.gt(0) & frame.raw_volume.gt(0)
    eligible &= frame.adjustment_factor.gt(0) & frame.feature_coverage_28.ge(.8)
    eligible &= frame.valid_closes_252.ge(202)
    x = frame.loc[eligible, ['instrument']].copy()
    features = features.loc[eligible]
    if len(x) < 150:
        raise ValueError('Fewer than 150 eligible securities')
    ranks = (features.rank(pct=True) - .5) * 2
    x['score'] = ranks.fillna(0).to_numpy() @ SIGNS / 8
    if not np.isfinite(x.score).all():
        raise ValueError('Nonfinite score')
    return x.sort_values(['score', 'instrument'], ascending=[False, True]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--asof', type=pd.Timestamp, required=True)
    parser.add_argument('--source-id', required=True,
                        help='Immutable vendor/version or provider snapshot identifier')
    args = parser.parse_args()
    if not args.source_id.strip():
        parser.error('A nonempty source-id is required')
    signal = make_signal(pd.read_csv(args.features, dtype={'instrument': str}), args.asof)
    source_hash = hashlib.sha256(args.features.read_bytes()).hexdigest()
    payload = signal.to_csv(index=False, float_format='%.12g', lineterminator='\n').encode()
    signal_hash = hashlib.sha256(payload).hexdigest()
    PRIVATE.mkdir(parents=True, exist_ok=True)
    snapshot = PRIVATE / f'{args.asof.date()}_{signal_hash[:16]}.csv'
    if any(PRIVATE.glob(f'{args.asof.date()}_*.csv')):
        raise FileExistsError('Date already frozen; do not overwrite or revise an observed signal')
    snapshot.write_bytes(payload)
    registry = pd.DataFrame([{'asof_date': args.asof.date(), 'source_id': args.source_id,
                              'source_sha256': source_hash, 'signal_sha256': signal_hash,
                              'eligible_securities': len(signal), 'model': 'Equal8_score_only',
                              'status': 'score_only_no_portfolio_or_future_outcome'}])
    if PUBLIC.exists():
        old = pd.read_csv(PUBLIC)
        if str(args.asof.date()) in old.asof_date.astype(str).tolist():
            snapshot.unlink()
            raise FileExistsError('Public registry already has this as-of date')
        registry = pd.concat([old, registry], ignore_index=True)
    registry.to_csv(PUBLIC, index=False, lineterminator='\n')
    print(json.dumps({'asof': str(args.asof.date()), 'signal_sha256': signal_hash,
                      'n': len(signal), 'status': 'archived_before_outcome'}))


if __name__ == '__main__':
    main()
