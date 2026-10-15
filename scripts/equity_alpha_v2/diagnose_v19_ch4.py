"""Fixed, descriptive diagnostics for the already selected V1.8 CH-4 model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / 'outputs/v18_attribution/ch4_aligned_private.csv'
OUT = ROOT / 'reports/summary'
FACTORS = ['mktrf', 'SMB', 'VMG', 'PMO']


def diagnostics(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = frame.loc[(frame.date >= '2024-01-01') & (frame.date < '2026-01-01')].copy()
    if len(primary) != 484 or primary.date.duplicated().any() or primary[FACTORS].isna().any().any():
        raise ValueError('The fixed 484-day complete primary sample is required')
    corr = primary[FACTORS].corr()
    matrix = sm.add_constant(primary[FACTORS].to_numpy(), has_constant='add')
    vif = {}
    for j, name in enumerate(FACTORS):
        other = np.delete(matrix, j + 1, axis=1)
        r2 = sm.OLS(matrix[:, j + 1], other).fit().rsquared
        vif[name] = np.inf if r2 >= 1 else 1 / (1 - r2)
    rows = []
    periods = [('2024-2025_primary', primary)] + [
        (str(year), group) for year, group in primary.groupby(primary.date.dt.year)
    ]
    for label, sample in periods:
        fit = sm.OLS(sample.portfolio_excess.to_numpy(),
                     sm.add_constant(sample[FACTORS].to_numpy(), has_constant='add')).fit(
                         cov_type='HAC', cov_kwds={'maxlags': 5})
        for j, name in enumerate(['alpha', *FACTORS]):
            rows.append({'period': label, 'term': name, 'n': len(sample),
                         'coefficient_daily': fit.params[j], 'hac_p': fit.pvalues[j],
                         'ci_low': fit.conf_int()[j, 0], 'ci_high': fit.conf_int()[j, 1],
                         'vif_primary': vif.get(name, np.nan)})
    return corr, pd.DataFrame(rows)


def main() -> None:
    frame = pd.read_csv(SOURCE, parse_dates=['date'])
    corr, coefficients = diagnostics(frame)
    OUT.mkdir(exist_ok=True, parents=True)
    (OUT / 'v19_ch4_factor_correlation.csv').write_bytes(
        corr.to_csv(float_format='%.8g', lineterminator='\n').encode())
    (OUT / 'v19_ch4_stability.csv').write_bytes(
        coefficients.to_csv(index=False, float_format='%.12g', lineterminator='\n').encode())
    print(json.dumps({'primary_days': 484, 'maximum_vif': float(coefficients.vif_primary.max()),
                      'outputs': ['v19_ch4_factor_correlation.csv', 'v19_ch4_stability.csv']}))


if __name__ == '__main__':
    main()
