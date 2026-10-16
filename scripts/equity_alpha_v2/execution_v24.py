"""Causal open execution adapter; preserves the frozen V1.6 and V2.2 engines.

V2.2's session-volume availability gate observes a full-day bar at the open.
Replace ONLY that gate with previous-session volume. Keep lagged ADV and
volatility computed from the original past bars, and actual current open/factor
for execution. A valid daily open is still only a simulated fill proxy.
"""
from copy import copy

import numpy as np

from execution_v22 import run_account as run_e2


def run_account(predictions, market, portfolio, config, **kwargs):
    causal = copy(market)
    causal.fields = dict(market.fields)
    volume = np.full_like(market.fields['volume'], np.nan)
    volume[1:] = market.fields['volume'][:-1]
    causal.fields['volume'] = volume
    # Deliberately do not call __post_init__: ADV/vol are already shifted once.
    result = run_e2(predictions, causal, portfolio, config, regime='E2', **kwargs)
    result[0]['regime'] = 'E24_prior_volume_gate'
    return result
