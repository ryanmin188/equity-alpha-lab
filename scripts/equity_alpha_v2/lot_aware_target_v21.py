"""Prospective, synthetic-tested lot-aware target projection.

This is an execution-design response to a post-hoc finding. It is not used in
the V2.1 historical performance figures or the frozen V1.6 account.
"""
from __future__ import annotations

import math


def project_sell_target(*, current_raw_shares, desired_raw_shares, current_value,
                        desired_value, max_position_value, star_board):
    """Choose a legal no-trade, partial sale, or complete exit before order sizing.

    STAR partial-sale orders need at least 200 raw shares and then 1-share
    increments. Other A-share partial sales use 100-share increments here.
    A whole-position exit is allowed when it is a valid 200/100-share sale or
    when an odd-lot residual must be sold in one order. Liquidity and price
    limits are separate later-stage checks; this function cannot promise fills.
    """
    current=float(current_raw_shares)
    desired=float(desired_raw_shares)
    if any(not math.isfinite(x) for x in (current,desired,current_value,desired_value,max_position_value)):
        raise ValueError('Non-finite target input')
    if min(current,desired,current_value,desired_value,max_position_value)<0:
        raise ValueError('Negative target input')
    if current<=0 or desired>=current:
        return desired,'NO_SELL_PROJECTION'
    minimum=200 if star_board else 100
    increment=1 if star_board else 100
    requested=current-desired
    if requested>=minimum:
        tradable=math.floor(requested/increment)*increment
        if tradable>=minimum:
            if current-tradable<=1e-8:
                return 0.,'FULL_EXIT'
            return current-tradable,'PARTIAL_SELL'
    # A sub-minimum partial sale is not sent as an impossible order. If the
    # existing position already breaches its value cap, full exit is mandatory.
    if current_value>max_position_value+1e-8:
        return 0.,'FULL_EXIT_CAP_BREACH'
    keep_error=abs(current_value-desired_value)
    exit_error=abs(desired_value)
    if exit_error<=keep_error:
        return 0.,'FULL_EXIT_NEAREST_LEGAL'
    return current,'KEEP_NEAREST_LEGAL'
