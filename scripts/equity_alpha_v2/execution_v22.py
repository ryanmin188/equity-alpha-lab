"""V2.2 execution experiments on unchanged historical signals.

Decisions and target quantities use the signal close. Orders fill at the next
open plus signed slippage and a square-root impact assumption. A lagged-ADV cap
limits participation. Qlib factor maps adjusted units to raw-share lot proxies;
corporate actions remain adjusted-return accounting, not cash dividend records.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from lot_aware_target_v21 import project_sell_target


@dataclass
class Market:
    dates: pd.DatetimeIndex
    symbols: list[str]
    fields: dict[str, np.ndarray]
    benchmark_open: np.ndarray
    benchmark_close: np.ndarray

    def __post_init__(self):
        self.date_index = {pd.Timestamp(d): i for i, d in enumerate(self.dates)}
        self.symbol_index = {s: i for i, s in enumerate(self.symbols)}
        close = pd.DataFrame(self.fields['close'])
        self.marks = close.where(close > 0).ffill().to_numpy()
        returns = close.pct_change(fill_method=None)
        self.lagged_vol = returns.rolling(20, min_periods=15).std().shift(1).to_numpy()
        amount = close * self.fields['volume']
        self.lagged_adv = amount.rolling(20, min_periods=15).mean().shift(1).to_numpy()


def capped_weights(values, cap):
    values = np.asarray(values, float)
    if len(values) * cap < 1 - 1e-10:
        raise ValueError('Stock cap cannot support a fully allocated target')
    values = np.where(np.isfinite(values) & (values > 0), values, 1.0)
    result = np.zeros(len(values))
    remaining = np.ones(len(values), dtype=bool)
    for _ in range(len(values) + 1):
        if not remaining.any():
            break
        allocation = values[remaining] / values[remaining].sum() * (1-result.sum())
        too_large = allocation > cap + 1e-12
        if not too_large.any():
            result[remaining] = allocation
            break
        indices = np.flatnonzero(remaining)[too_large]
        result[indices] = cap
        remaining[indices] = False
    assert abs(result.sum()-1) < 1e-8 and result.max() <= cap+1e-8
    return result


def limit_band(symbol, date):
    # Proxy only: historical ST/IPO status and exchange limit references absent.
    if symbol.startswith('SH688') or (symbol.startswith(('SZ300','SZ301')) and date >= pd.Timestamp('2020-08-24')):
        return .195
    return .095


def charges(notional, side, date, config):
    if notional <= 0:
        return 0.0
    commission = max(config['minimum_commission_cny'], notional*config['commission_bps']/1e4)
    tax = 0.0
    if side == 'SELL':
        key = 'sell_tax_bps_from_2023_08_28' if date >= pd.Timestamp('2023-08-28') else 'sell_tax_bps_before_2023_08_28'
        tax = notional * config[key]/1e4
    return commission + tax + notional*config['transfer_fee_bps_assumption']/1e4


def run_account(predictions, market, portfolio, config, *, regime, slippage_bps=None, capital=None):
    """One candidate, one period; E2 retries exit orders from prior-close state.

    Metrics end at the final open for comparability with the original account.
    Normal-day valuation is at close. Terminal open is a mark, not forced
    liquidation: residual inventory is reported and exit charges are not assumed.
    """
    if regime not in ('E1', 'E2'):
        raise ValueError('E0 uses the untouched V1.6 engine')
    predictions = predictions.sort_values(['date','score','instrument'], ascending=[True,False,True])
    if predictions.empty:
        raise ValueError('Empty prediction interval')
    groups = {pd.Timestamp(g.entry_date.iloc[0]): g for _,g in predictions.groupby('date')}
    begin = min(groups)
    end = pd.Timestamp(predictions.exit_date.max())
    i0, i1 = market.date_index[begin], market.date_index[end]
    cfg = config
    initial = float(capital or cfg['capital'])
    cash = initial
    shares = np.zeros(len(market.symbols))
    bought_today = np.zeros(len(market.symbols))
    slip = cfg['slippage_bps'] if slippage_bps is None else slippage_bps
    events, daily, targets, exposures = [], [], [], []
    total_fees = total_slippage = total_notional = 0.0
    previous_nav = initial
    selected_set = set()
    bench_initial = market.benchmark_open[i0]
    if not np.isfinite(bench_initial) or bench_initial <= 0:
        raise ValueError('Missing benchmark start open')

    for i in range(i0,i1+1):
        date = market.dates[i]
        bought_today[:] = 0
        open_px = market.fields['open'][i]
        prior = market.marks[i-1]
        factor = market.fields['factor'][i]
        fee_day = slip_day = traded_day = 0.0
        rebalance = date in groups and i < i1
        risk_only = regime == 'E2' and i > i0 and i < i1 and not rebalance
        strategic_weights = {}
        legal_weights = {}
        if rebalance or risk_only:
            group = groups[date] if rebalance else None
            signal = pd.Timestamp(group.date.iloc[0]) if rebalance else market.dates[i-1]
            signal_i = market.date_index[signal]
            if signal_i >= i:
                raise ValueError('Signal must precede execution')
            known_px = market.marks[signal_i]
            held = np.flatnonzero(shares > 1e-10)
            if held.size and not np.isfinite(known_px[held]).all():
                raise ValueError('Cannot value held positions as of signal')
            signal_nav = cash + np.sum(shares[held]*known_px[held])
            current_values = np.zeros(len(shares))
            current_values[held] = shares[held]*known_px[held]
            target_quantity = shares.copy()
            rank = {}
            if rebalance:
                ordered = [market.symbol_index[s] for s in group.instrument]
                rank = {s:j for j,s in enumerate(ordered)}
                retained = sorted([s for s in held if rank.get(s,10**9)<portfolio['buffer_n']], key=lambda s:rank[s])
                selected = retained[:portfolio['top_n']]
                selected += [s for s in ordered if s not in selected][:portfolio['top_n']-len(selected)]
                if len(selected) != portfolio['top_n']:
                    raise ValueError('Insufficient selected securities')
                selected_set = set(selected)
                raw_vol = group.set_index('instrument')['raw_vol_60']
                if portfolio['weighting'] == 'inverse_vol':
                    values = [1/max(float(raw_vol.get(market.symbols[s],.02)),.005) for s in selected]
                else:
                    values = np.ones(len(selected))
                weights = capped_weights(values,cfg['max_stock_weight'])
                desired_values = np.zeros(len(shares))
                desired_values[selected] = signal_nav * .98 * weights
                change = desired_values-current_values
                proposed_turnover = np.abs(change).sum()/signal_nav
                if held.size and proposed_turnover > cfg['max_rebalance_gross_turnover']:
                    change *= cfg['max_rebalance_gross_turnover']/proposed_turnover
                    desired_values = current_values+change
                active = np.flatnonzero((desired_values>1e-10)|(shares>1e-10))
                target_quantity[active] = desired_values[active]/known_px[active]
                targets.append({'signal_date':signal,'execution_date':date,'signal_nav':signal_nav,
                                'proposed_turnover':proposed_turnover,'target_positions':len(selected),'kind':'REBALANCE'})
            else:
                desired_values = current_values.copy()
                overweight=held[current_values[held]/signal_nav>cfg['max_stock_weight']+1e-10]
                orphaned=np.array([s for s in held if s not in selected_set],dtype=int)
                active=np.union1d(overweight,orphaned).astype(int)
                for s in active:
                    desired_values[s]=(0. if s not in selected_set else
                                       .9*cfg['max_stock_weight']*signal_nav)
                    target_quantity[s]=desired_values[s]/known_px[s]
                if len(active):
                    targets.append({'signal_date':signal,'execution_date':date,'signal_nav':signal_nav,
                                    'proposed_turnover':float(np.sum(current_values[active]-desired_values[active])/signal_nav),
                                    'target_positions':len(held),'kind':'RISK_ONLY'})
            signal_factor = market.fields['factor'][signal_i]
            projection_reason = {}
            for s in active:
                strategic_weights[s]=float(desired_values[s]/signal_nav)
                if target_quantity[s]<shares[s]-1e-9:
                    if not np.isfinite(signal_factor[s]) or signal_factor[s]<=0:
                        projection_reason[s]='NO_SIGNAL_FACTOR'
                    else:
                        raw_target,why=project_sell_target(
                            current_raw_shares=shares[s]*signal_factor[s],
                            desired_raw_shares=target_quantity[s]*signal_factor[s],
                            current_value=current_values[s],desired_value=desired_values[s],
                            max_position_value=(cfg['max_stock_weight'] if regime=='E2' else 100.)*signal_nav,
                            star_board=market.symbols[s].startswith('SH688'))
                        target_quantity[s]=raw_target/signal_factor[s]
                        projection_reason[s]=why
                elif regime=='E2' and target_quantity[s]>shares[s]+1e-9:
                    minimum=(200 if market.symbols[s].startswith('SH688') else 100)*known_px[s]/signal_factor[s] if np.isfinite(signal_factor[s]) and signal_factor[s]>0 else np.inf
                    adv=market.lagged_adv[i,s]
                    if not np.isfinite(adv) or minimum>cfg['max_participation_of_lagged_adv']*adv:
                        target_quantity[s]=shares[s]
                        projection_reason[s]='BUY_CAPACITY_GATE'
                legal_weights[s]=float(target_quantity[s]*known_px[s]/signal_nav)
            if rebalance:
                budget=cfg['max_rebalance_gross_turnover']
                planned=float(np.sum(np.abs(target_quantity[active]-shares[active])*known_px[active])/signal_nav)
                if held.size and planned>budget+1e-10:
                    # A projected whole-position exit can enlarge the original
                    # turnover-scaled request. Keep valid existing holdings and
                    # cancel low-ranked buys before sacrificing risk-reducing sells.
                    cancellable=sorted(active,key=lambda s:(
                        target_quantity[s]<shares[s]-1e-9,
                        current_values[s]/signal_nav>cfg['max_stock_weight'],
                        -rank.get(s,10**9),s))
                    for s in cancellable:
                        if planned<=budget+1e-10:break
                        if abs(target_quantity[s]-shares[s])<1e-9:continue
                        planned-=abs(target_quantity[s]-shares[s])*known_px[s]/signal_nav
                        target_quantity[s]=shares[s]
                        legal_weights[s]=float(current_values[s]/signal_nav)
                        projection_reason[s]='TURNOVER_BUDGET_SKIP'
                if held.size and planned>budget+1e-8:
                    raise AssertionError('Projected rebalance turnover exceeds budget')
                targets[-1]['planned_legal_turnover']=planned
            # Quantities are fixed using prior information; actual open cannot
            # improve target sizing. Cash and liquidity can only reduce a fill.
            pending = [(s, target_quantity[s]-shares[s]) for s in active]
            pending.sort(key=lambda x:(x[1]>=0,
                -current_values[x[0]]/signal_nav if regime=='E2' and x[1]<0 else 0.,
                rank.get(x[0],10**9),x[0]))
            for s,delta in pending:
                if abs(delta)<1e-9:
                    continue
                side = 'BUY' if delta>0 else 'SELL'
                direction = 1 if delta>0 else -1
                px = open_px[s]
                record = {'date':date,'signal_date':signal,'instrument':market.symbols[s],
                          'side':side,'requested_adjusted_units':abs(delta),'filled_adjusted_units':0.0,
                          'notional':0.0,'fees':0.0,'slippage_impact':0.0,'status':'FILLED',
                          'projection':projection_reason.get(s,'NONE'),
                          'kind':'REBALANCE' if rebalance else 'RISK_ONLY'}
                reason = None
                if not np.isfinite(px) or px<=0 or not np.isfinite(factor[s]) or factor[s]<=0:
                    reason='BLOCKED_PRICE_OR_FACTOR'
                elif not np.isfinite(prior[s]) or prior[s]<=0:
                    reason='BLOCKED_REFERENCE'
                elif not np.isfinite(market.fields['volume'][i,s]) or market.fields['volume'][i,s]<=0:
                    reason='BLOCKED_NO_SESSION_VOLUME'
                elif abs(px/prior[s]-1)>=limit_band(market.symbols[s],date):
                    reason='BLOCKED_LIMIT_PROXY'
                adv=market.lagged_adv[i,s]
                sigma=market.lagged_vol[i,s]
                if reason is None and (not np.isfinite(adv) or adv<=0 or not np.isfinite(sigma)):
                    reason='BLOCKED_HISTORY'
                if reason:
                    record['status']=reason
                    events.append(record)
                    continue
                wanted=abs(delta)
                quantity=min(wanted,cfg['max_participation_of_lagged_adv']*adv/px)
                if side=='SELL':
                    quantity=min(quantity,max(0.,shares[s]-bought_today[s]))
                # 100 raw-share lots mapped via the provider adjustment factor.
                star=market.symbols[s].startswith('SH688')
                lot=(1 if star else 100)/factor[s]
                minimum=(200 if star else 100)/factor[s]
                # A full exit may sell the entire odd-lot remainder. Partial
                # reductions still use whole lots; otherwise small holdings
                # would become permanently unsellable after corporate actions.
                full_exit=side=='SELL' and target_quantity[s]<=1e-10 and quantity>=shares[s]-1e-8
                quantity=shares[s] if full_exit else np.floor(quantity/lot)*lot
                if not full_exit and quantity<minimum-1e-8:
                    quantity=0.
                if quantity<=0:
                    record['status']='BLOCKED_LOT_OR_CAPACITY'
                    events.append(record)
                    continue
                impact=cfg['impact_coefficient']*sigma*np.sqrt(quantity*px/adv)
                execution_px=px*(1+direction*(slip/1e4+impact))
                if side=='BUY':
                    # Conservative affordability estimate; recompute exact fees
                    # after lot rounding, without borrowing cash.
                    affordable=max(0.,cash-cfg['minimum_commission_cny'])/(execution_px*(1+(cfg['commission_bps']+cfg['transfer_fee_bps_assumption'])/1e4))
                    quantity=min(quantity,np.floor(affordable/lot)*lot)
                    if quantity<minimum-1e-8: quantity=0.
                if quantity<=0:
                    record['status']='BLOCKED_CASH'
                    events.append(record)
                    continue
                # Recalculate impact when a cash constraint reduces the fill.
                impact=cfg['impact_coefficient']*sigma*np.sqrt(quantity*px/adv)
                execution_px=px*(1+direction*(slip/1e4+impact))
                notional=quantity*execution_px
                fee=charges(notional,side,date,cfg)
                cash-=direction*notional+fee
                shares[s]+=direction*quantity
                if side=='BUY':
                    bought_today[s]+=quantity
                if cash < -1e-6 or shares[s]<-1e-8:
                    raise RuntimeError('Negative cash/position')
                cost=quantity*abs(execution_px-px)
                record.update(filled_adjusted_units=quantity,notional=notional,fees=fee,
                              slippage_impact=cost,status='PARTIAL' if quantity<wanted-1e-8 else 'FILLED',
                              factor=float(factor[s]),raw_share_proxy=quantity*factor[s],
                              participation=quantity*px/adv,execution_price=execution_px)
                events.append(record)
                fee_day+=fee
                slip_day+=cost
                traded_day+=notional
        held=np.flatnonzero(shares>1e-10)
        # Last observation is an open mark, consistent with the return labels.
        observed=open_px if i==i1 else market.fields['close'][i]
        mark=np.where(np.isfinite(observed)&(observed>0),observed,market.marks[i-1] if i==i1 else market.marks[i])
        if held.size and not np.isfinite(mark[held]).all():
            raise ValueError('Missing historical valuation')
        stock_value=float(np.sum(shares[held]*mark[held]))
        nav=cash+stock_value
        bmark=market.benchmark_open[i] if i==i1 else market.benchmark_close[i]
        if not np.isfinite(bmark) or bmark<=0 or nav<=0:
            raise ValueError('Invalid account or benchmark mark')
        weights=shares[held]*mark[held]/nav
        if regime=='E2' or rebalance:
            open_mark=np.where(np.isfinite(open_px)&(open_px>0),open_px,prior)
            open_nav=cash+float(np.sum(shares[held]*open_mark[held]))
            for s,w in zip(held,weights):
                exposures.append({'date':date,'instrument':market.symbols[s],
                    'target_weight':strategic_weights.get(s,np.nan),
                    'legal_target_weight':legal_weights.get(s,np.nan),
                    'executed_open_weight':shares[s]*open_mark[s]/open_nav,
                    'marked_weight':w,'risk_breach':w>cfg['max_stock_weight']+1e-10})
        total_fees+=fee_day
        total_slippage+=slip_day
        total_notional+=traded_day
        daily.append({'date':date,'wealth':nav/initial,'net':nav/previous_nav-1,
                      'benchmark_wealth':bmark/bench_initial,'cash':cash,'holdings_value':stock_value,
                      'nav':nav,'fees':fee_day,'slippage_impact':slip_day,'traded_notional':traded_day,
                      'n_holdings':len(held),'cash_weight':cash/nav,'max_weight':weights.max() if len(weights) else 0.,
                      'stale_positions':int((~np.isfinite(observed[held])|(observed[held]<=0)).sum()),
                      'valuation':'open' if i==i1 else 'close'})
        previous_nav=nav
    daily=pd.DataFrame(daily)
    orders=pd.DataFrame(events)
    targets=pd.DataFrame(targets)
    r=daily.net.to_numpy()
    bnav=np.r_[1.,daily.benchmark_wealth.to_numpy()]
    br=bnav[1:]/bnav[:-1]-1
    excess=r-br
    years=(end-begin).days/365.25
    growth=np.r_[1.,daily.wealth.to_numpy()]
    dd=(growth/np.maximum.accumulate(growth)-1).min()
    te=np.std(excess,ddof=1)*np.sqrt(252)
    summary={'first_entry':str(begin.date()),'final_mark':str(end.date()),'days':len(daily),
             'total_return':growth[-1]-1,'benchmark_return':bnav[-1]-1,
             'cumulative_excess_points':growth[-1]-bnav[-1],
             'relative_wealth_return':growth[-1]/bnav[-1]-1,
             'annualized_return':growth[-1]**(1/years)-1,'max_drawdown':dd,
             'information_ratio':float(excess.mean()*252/te) if te>0 else np.nan,
             'average_monthly_turnover':total_notional/initial/(years*12),
             'fees_cny':total_fees,'slippage_impact_cny':total_slippage,
             'blocked_orders':int(orders.status.str.startswith('BLOCKED').sum()) if len(orders) else 0,
             'terminal_positions':len(held),'capital':initial,'slippage_bps':slip,
             'terminal_liquidation':False}
    summary['regime']=regime
    return summary,daily,orders,targets,pd.DataFrame(exposures)
