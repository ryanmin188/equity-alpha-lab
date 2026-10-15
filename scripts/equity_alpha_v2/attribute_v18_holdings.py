"""Read-only attribution of the saved V1.6 final account.

Private security/day/order outputs stay under the git-ignored outputs tree.
Only aggregate, non-trade-level summaries are copied to reports/summary.
This is adjusted-unit simulation accounting, not a raw-share dividend ledger.
"""
from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = HERE / 'outputs' / 'v16_improvements'
PRIVATE = HERE / 'outputs' / 'v18_attribution'
PUBLIC = ROOT / 'reports' / 'summary'
ACCOUNT = BASE / 'final' / 'Equal8__Buffer30_60'
FEATURES = ['mom_20','mom_60','reversal_5','vol_20','vol_60','volume_surprise',
            'range_20','ma_gap_20','mom_120','mom_252_skip20','mom_60_skip5',
            'vol_ratio','downside_vol60','max_up20','max_down20','drawdown120',
            'price_position60','skew60','volume_ratio5_20','log_dollar_volume',
            'intraday20','overnight20','return_volume_corr20','positive_days60',
            'trend60','ma_gap60','vol5','amihud60']
INITIAL = 1_000_000.0
TOL = 1e-5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(df, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format='%.12g', lineterminator='\n')


def load_inputs():
    daily = pd.read_csv(ACCOUNT / 'daily_account.csv', parse_dates=['date'])
    orders = pd.read_csv(ACCOUNT / 'orders.csv', parse_dates=['date'])
    # Pickle is trusted because it was produced by this repository on this host.
    with (BASE / 'market_final.pkl').open('rb') as stream:
        market, _ = pickle.load(stream)
    assert daily.date.is_unique and daily.date.is_monotonic_increasing
    assert set(orders.date).issubset(set(daily.date))
    assert daily.valuation.iloc[-1] == 'open'
    assert daily.valuation.iloc[:-1].eq('close').all()
    return daily, orders, market


def replay(daily, orders, market):
    n = len(market.symbols)
    q = np.zeros(n)
    basis = np.zeros(n)
    unrealized = np.zeros(n)
    cash = INITIAL
    previous_marks = np.full(n, np.nan)
    previous_benchmark_wealth = 1.0
    ledger, daily_checks, stale = [], [], []
    by_date = {d: frame for d, frame in orders.groupby('date')}
    for j, row in daily.iterrows():
        date = row.date
        i = market.date_index[date]
        opening_q = q.copy()
        opening_value = np.where(opening_q > 1e-8, opening_q * np.nan_to_num(previous_marks), 0.)
        prior_cash = cash
        buy = np.zeros(n)
        sell = np.zeros(n)
        fees = np.zeros(n)
        realized = np.zeros(n)
        slip = np.zeros(n)
        for order in by_date.get(date, pd.DataFrame()).itertuples(index=False):
            if order.status.startswith('BLOCKED'):
                continue
            s = market.symbol_index[order.instrument]
            qty, notional, fee = float(order.filled_adjusted_units), float(order.notional), float(order.fees)
            assert qty > 0 and np.isfinite(notional) and np.isfinite(fee)
            price = notional / qty
            fees[s] += fee
            slip[s] += float(order.slippage_impact)
            if order.side == 'BUY':
                old_cost = q[s] * basis[s]
                buy[s] += notional
                cash -= notional + fee
                q[s] += qty
                basis[s] = (old_cost + notional) / q[s]
            elif order.side == 'SELL':
                assert qty <= q[s] + 1e-7
                sell[s] += notional
                cash += notional - fee
                realized[s] += qty * (price - basis[s])
                q[s] -= qty
                if q[s] < 1e-7:
                    q[s] = 0.
                    basis[s] = 0.
            else:
                raise ValueError(f'Unrecognized order side {order.side}')
        observed = market.fields['open'][i] if row.valuation == 'open' else market.fields['close'][i]
        fallback = market.marks[i-1] if row.valuation == 'open' else market.marks[i]
        mark = np.where(np.isfinite(observed) & (observed > 0), observed, fallback)
        held = np.flatnonzero(q > 1e-8)
        assert np.isfinite(mark[held]).all()
        ending_value = np.where(q > 1e-8, q * np.nan_to_num(mark), 0.)
        gross = ending_value - opening_value + sell - buy
        new_unrealized = np.where(q > 1e-8, q * (np.nan_to_num(mark) - basis), 0.)
        unrealized_delta = new_unrealized - unrealized
        net = gross - fees
        active = np.flatnonzero((opening_q > 1e-8) | (q > 1e-8) | (buy > 0) | (sell > 0))
        for s in active:
            ledger.append(dict(date=date.date(), year=date.year, instrument=market.symbols[s],
                               opening_adjusted_units=opening_q[s], ending_adjusted_units=q[s],
                               opening_value=opening_value[s], ending_value=ending_value[s],
                               purchase_notional=buy[s], sale_notional=sell[s],
                               gross_pnl=gross[s], realized_pnl=realized[s],
                               change_unrealized_pnl=unrealized_delta[s],
                               unrealized_pnl_end=new_unrealized[s], order_fees=fees[s],
                               slippage_impact_in_notional=slip[s], net_pnl=net[s]))
        nav = cash + ending_value.sum()
        nav_change = nav - (INITIAL if j == 0 else daily.nav.iloc[j-1])
        prior_nav = INITIAL if j == 0 else daily.nav.iloc[j-1]
        benchmark_return = row.benchmark_wealth / previous_benchmark_wealth - 1
        portfolio_return = nav_change / prior_nav
        active_simple = portfolio_return - benchmark_return
        active_log = np.log1p(portfolio_return) - np.log1p(benchmark_return)
        log_scale = active_log / active_simple if abs(active_simple) > 1e-12 else 1/(1+benchmark_return)
        security_active_simple = net/prior_nav - opening_value/prior_nav*benchmark_return
        security_active_log = security_active_simple*log_scale
        cash_active_simple = -prior_cash/prior_nav*benchmark_return
        for rec, s in zip(ledger[-len(active):] if len(active) else [], active):
            rec['active_simple_contribution'] = security_active_simple[s]
            rec['active_log_contribution'] = security_active_log[s]
        daily_checks.append(dict(date=date.date(), account_nav=row.nav, replay_nav=nav,
                                 replay_cash=cash, replay_holdings=ending_value.sum(),
                                 security_gross_pnl=gross.sum(), order_fees=fees.sum(),
                                 security_net_pnl=net.sum(), nav_change=nav_change,
                                 nav_residual=nav-row.nav,
                                 pnl_residual=nav_change-net.sum(),
                                 cash_residual=cash-row.cash,
                                 basis_residual=gross.sum()-realized.sum()-unrealized_delta.sum(),
                                 benchmark_return=benchmark_return, portfolio_return=portfolio_return,
                                 active_log=active_log, security_active_log=security_active_log.sum(),
                                 cash_active_log=cash_active_simple*log_scale,
                                 active_log_residual=active_log-(security_active_simple.sum()+cash_active_simple)*log_scale))
        for s in held:
            if not np.isfinite(observed[s]) or observed[s] <= 0:
                stale.append(dict(date=date.date(), instrument=market.symbols[s],
                                  adjusted_units=q[s], fallback_mark=mark[s],
                                  marked_exposure=ending_value[s], nav=nav,
                                  fraction_nav=ending_value[s]/nav,
                                  valuation=row.valuation))
        previous_marks = mark
        unrealized = new_unrealized
        previous_benchmark_wealth = row.benchmark_wealth
        assert abs((cash-prior_cash) - (sell.sum()-buy.sum()-fees.sum())) < TOL
    ledger = pd.DataFrame(ledger)
    checks = pd.DataFrame(daily_checks)
    stale = pd.DataFrame(stale, columns=['date','instrument','adjusted_units','fallback_mark',
                                         'marked_exposure','nav','fraction_nav','valuation'])
    for field in ['nav_residual','pnl_residual','cash_residual','basis_residual','active_log_residual']:
        if checks[field].abs().max() > TOL:
            raise AssertionError(f'{field} exceeds {TOL}: {checks[field].abs().max()}')
    assert abs(checks.security_net_pnl.sum() - (daily.nav.iloc[-1]-INITIAL)) < TOL
    assert len(stale) == int(daily.stale_positions.sum())
    return ledger, checks, stale


def security_summaries(ledger, daily, stale):
    security = ledger.groupby('instrument', as_index=False).agg(
        gross_pnl=('gross_pnl','sum'), realized_pnl=('realized_pnl','sum'),
        change_unrealized_pnl=('change_unrealized_pnl','sum'),
        fees=('order_fees','sum'), slippage_impact_in_notional=('slippage_impact_in_notional','sum'),
        net_pnl=('net_pnl','sum'), active_log_contribution=('active_log_contribution','sum'),
        holding_days=('date','nunique'))
    yearly = ledger.groupby(['year','instrument'], as_index=False).agg(
        gross_pnl=('gross_pnl','sum'), realized_pnl=('realized_pnl','sum'),
        change_unrealized_pnl=('change_unrealized_pnl','sum'), fees=('order_fees','sum'),
        net_pnl=('net_pnl','sum'), active_log_contribution=('active_log_contribution','sum'),
        holding_days=('date','nunique'))
    rows, top_rows = [], []
    for label, group in [('all',security), *[(str(y),g) for y,g in yearly.groupby('year')]]:
        pos = group.loc[group.net_pnl > 0].sort_values('net_pnl',ascending=False)
        neg = group.loc[group.net_pnl < 0].sort_values('net_pnl')
        total_positive = pos.net_pnl.sum()
        top5 = pos.head(5).net_pnl.sum()
        top10 = pos.head(10).net_pnl.sum()
        rows.append(dict(period=label, securities=len(group), net_pnl=group.net_pnl.sum(),
                         positive_pnl=total_positive, negative_pnl=neg.net_pnl.sum(),
                         top5_positive_pnl=top5, top10_positive_pnl=top10,
                         top5_share_positive=top5/total_positive if total_positive else np.nan,
                         top10_share_positive=top10/total_positive if total_positive else np.nan,
                         top10_share_net=top10/group.net_pnl.sum() if group.net_pnl.sum() else np.nan,
                         n_positive=len(pos), n_negative=len(neg)))
        # Public release is intentionally ranking aggregates only; no full holdings/trade history.
        if label in ('2024','2025'):
            for direction, subset in [('gainer',pos.head(10)),('detractor',neg.head(10))]:
                for rank, item in enumerate(subset.itertuples(),1):
                    top_rows.append(dict(year=label, direction=direction, rank=rank,
                                         instrument=item.instrument, net_pnl_cny=item.net_pnl))
    concentration = pd.DataFrame(rows)
    top = pd.DataFrame(top_rows)
    stale_yearly = (stale.assign(year=pd.to_datetime(stale.date).dt.year)
                     .groupby('year',as_index=False).agg(stale_position_days=('date','size'),
                         stale_dates=('date','nunique'), peak_exposure_fraction=('fraction_nav','max'),
                         sum_marked_exposure_cny=('marked_exposure','sum')))
    save(security, PRIVATE/'security_pnl_attribution.csv')
    save(yearly, PRIVATE/'yearly_security_drivers.csv')
    save(concentration, PRIVATE/'contribution_concentration.csv')
    save(stale, PRIVATE/'stale_mark_exposure.csv')
    save(concentration, PUBLIC/'v18_contribution_concentration.csv')
    save(top, PUBLIC/'v18_top_security_drivers.csv')
    save(stale_yearly, PUBLIC/'v18_stale_mark_summary.csv')
    return security, yearly, concentration, top, stale_yearly


def factor_significance():
    panel = pd.read_pickle(BASE/'panel_final.pkl')
    panel = panel.loc[(panel.date >= '2023-12-01') & (panel.date <= '2026-08-31')]
    rows = []
    for date, group in panel.groupby('date'):
        if pd.Timestamp(group.entry_date.iloc[0]) < pd.Timestamp('2024-01-01'):
            continue
        y = group.forward_return.to_numpy(dtype=float)
        for factor in FEATURES:
            x = group[factor].to_numpy(dtype=float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() < 100 or np.unique(x[ok]).size < 4:
                continue
            ic = spearmanr(x[ok], y[ok]).statistic
            rows.append(dict(signal_date=date.date(),entry_year=pd.Timestamp(group.entry_date.iloc[0]).year,
                             factor=factor, n_stocks=int(ok.sum()), ic=ic))
    monthly = pd.DataFrame(rows)
    stats = []
    for factor in FEATURES:
        group = monthly.loc[monthly.factor == factor].sort_values('signal_date')
        for period, sample in [('all',group), *[(str(y),g) for y,g in group.groupby('entry_year')]]:
            vals = sample.ic.to_numpy(dtype=float)
            if len(vals) < 6:
                continue
            fit = sm.OLS(vals,np.ones((len(vals),1))).fit(cov_type='HAC',cov_kwds={'maxlags':3})
            stats.append(dict(period=period,factor=factor,n_months=len(vals),mean_ic=vals.mean(),
                              median_ic=np.median(vals),positive_month_share=(vals>0).mean(),
                              hac_t=float(fit.tvalues[0]),hac_p=float(fit.pvalues[0])))
    summary = pd.DataFrame(stats)
    all_idx = summary.index[summary.period.eq('all')]
    p = summary.loc[all_idx,'hac_p'].to_numpy()
    order = np.argsort(p)
    ranked = p[order]
    qsorted = np.minimum.accumulate((ranked*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    q = np.empty_like(p)
    q[order] = np.minimum(qsorted,1)
    summary['bh_q_28'] = np.nan
    summary.loc[all_idx,'bh_q_28'] = q
    save(monthly, PRIVATE/'monthly_factor_ic.csv')
    save(summary, PRIVATE/'factor_significance.csv')
    save(summary, PUBLIC/'v18_factor_significance.csv')
    return summary


def technical_style_exposures(ledger, daily, market):
    """Signal-date descriptive exposures; size/value need other data."""
    panel = pd.read_pickle(BASE/'panel_final.pkl')
    signal_dates = sorted(set(pd.to_datetime(panel.date)) & set(daily.date))
    market_close = market.fields['close']
    benchmark = market.benchmark_close
    rows = []
    by_date = {date: group for date, group in ledger.groupby('date')}
    nav_by_date = daily.set_index(daily.date.dt.date).nav
    for date in signal_dates:
        if date.year < 2024 or date > daily.date.iloc[-1]:
            continue
        i = market.date_index[date]
        if i < 253:
            continue
        b = benchmark[i-252:i+1]
        rb = b[1:]/b[:-1]-1
        positions = by_date.get(date.date())
        if positions is None:
            continue
        for item in positions.itertuples(index=False):
            if item.ending_adjusted_units < 1e-8:
                continue
            s = market.symbol_index[item.instrument]
            c = market_close[i-252:i+1,s]
            with np.errstate(divide='ignore',invalid='ignore'):
                r = c[1:]/c[:-1]-1
                mom = c[-21]/c[0]-1
            valid = np.isfinite(r) & np.isfinite(rb)
            beta = np.cov(r[valid],rb[valid],ddof=1)[0,1]/np.var(rb[valid],ddof=1) if valid.sum()>=160 else np.nan
            vol60 = np.nanstd(r[-60:],ddof=1)*np.sqrt(252) if np.isfinite(r[-60:]).sum()>=48 else np.nan
            rows.append(dict(signal_date=date.date(),year=date.year,instrument=item.instrument,
                             weight_nav=item.ending_value/nav_by_date[date.date()],
                             trailing_beta_252=beta, annualized_vol_60=vol60,
                             momentum_252_skip20=mom,
                             valid_beta_days=int(valid.sum())))
    detail = pd.DataFrame(rows)
    agg=[]
    for date,g in detail.groupby('signal_date'):
        rec={'signal_date':date,'year':pd.Timestamp(date).year,'n_holdings':len(g),
             'invested_weight':g.weight_nav.sum()}
        for metric in ['trailing_beta_252','annualized_vol_60','momentum_252_skip20']:
            valid=g[metric].notna()
            rec[metric+'_weighted']=float((g.loc[valid,metric]*g.loc[valid,'weight_nav']).sum())
            rec[metric+'_covered_weight']=float(g.loc[valid,'weight_nav'].sum())
        agg.append(rec)
    aggregate=pd.DataFrame(agg)
    save(detail,PRIVATE/'technical_style_security_exposures.csv')
    save(aggregate,PUBLIC/'v18_technical_style_exposures.csv')
    return aggregate


def dividend_case(ledger, market):
    """One issuer-announcement check; raw quotes are inferred, not independent."""
    symbol = 'SH601988'
    date = pd.Timestamp('2024-07-17')
    i = market.date_index[date]
    s = market.symbol_index[symbol]
    dividend = .2364  # SSE 2024-07-10 implementation notice, pre-tax CNY/share.
    prev_adjusted = float(market.fields['close'][i-1,s])
    ex_adjusted = float(market.fields['close'][i,s])
    prev_factor = float(market.fields['factor'][i-1,s])
    ex_factor = float(market.fields['factor'][i,s])
    prev_raw_inferred = prev_adjusted/prev_factor
    ex_raw_inferred = ex_adjusted/ex_factor
    actual = ledger.loc[(ledger.date==date.date()) & (ledger.instrument==symbol)].iloc[0]
    assert actual.opening_adjusted_units>0 and actual.purchase_notional==0 and actual.sale_notional==0
    adjusted_return=ex_adjusted/prev_adjusted-1
    cash_return=(ex_raw_inferred+dividend)/prev_raw_inferred-1
    case=dict(instrument=symbol,record_date='2024-07-16',ex_date=str(date.date()),
              declared_cash_dividend_cny_per_raw_share=dividend,
              qlib_implied_raw_prev_close=prev_raw_inferred,
              qlib_implied_raw_ex_close=ex_raw_inferred,
              qlib_adjusted_return=adjusted_return,
              hypothetical_raw_plus_gross_dividend_return=cash_return,
              adjusted_minus_hypothetical_return_bp=(adjusted_return-cash_return)*1e4,
              source='https://www.sse.com.cn/disclosure/listedinfo/announcement/c/new/2024-07-10/601988_20240710_CPYP.pdf',
              verification='Issuer action verified; raw quotes inferred from Qlib, not independent raw-price confirmation')
    save(pd.DataFrame([case]),PUBLIC/'v18_corporate_action_case.csv')
    private={**case,'opening_adjusted_units':float(actual.opening_adjusted_units),
             'raw_share_proxy_from_prev_factor':float(actual.opening_adjusted_units*prev_factor),
             'qlib_adjusted_pnl_cny':float(actual.gross_pnl),
             'hypothetical_raw_plus_gross_dividend_pnl_cny':float(actual.opening_adjusted_units*prev_factor*(ex_raw_inferred-prev_raw_inferred+dividend))}
    private['pnl_gap_cny']=private['qlib_adjusted_pnl_cny']-private['hypothetical_raw_plus_gross_dividend_pnl_cny']
    save(pd.DataFrame([private]),PRIVATE/'corporate_action_case_private.csv')
    return case


def main():
    daily, orders, market = load_inputs()
    ledger, checks, stale = replay(daily, orders, market)
    save(ledger, PRIVATE/'daily_security_pnl_private.csv')
    save(checks, PRIVATE/'daily_reconciliation_private.csv')
    security, yearly, concentration, top, stale_yearly = security_summaries(ledger,daily,stale)
    factors = factor_significance()
    technical = technical_style_exposures(ledger,daily,market)
    action = dividend_case(ledger,market)
    checks['year'] = pd.to_datetime(checks.date).dt.year
    active_yearly = checks.groupby('year',as_index=False).agg(
        portfolio_log_relative=('active_log','sum'),security_log_relative=('security_active_log','sum'),
        cash_log_relative=('cash_active_log','sum'))
    assert (active_yearly.portfolio_log_relative-active_yearly.security_log_relative-active_yearly.cash_log_relative).abs().max()<1e-10
    save(active_yearly,PUBLIC/'v18_yearly_active_decomposition.csv')
    manifest = {'input_sha256': {name:sha(path) for name,path in {
        'daily_account.csv':ACCOUNT/'daily_account.csv','orders.csv':ACCOUNT/'orders.csv',
        'market_final.pkl':BASE/'market_final.pkl','panel_final.pkl':BASE/'panel_final.pkl'}.items()},
        'days':len(daily),'orders':len(orders),'security_day_rows':len(ledger),
        'security_count':len(security),'stale_position_days':len(stale),
        'max_abs_nav_residual_cny':float(checks.nav_residual.abs().max()),
        'max_abs_pnl_residual_cny':float(checks.pnl_residual.abs().max()),
        'max_abs_basis_residual_cny':float(checks.basis_residual.abs().max()),
        'sum_security_net_pnl_cny':float(security.net_pnl.sum()),
        'final_nav_change_cny':float(daily.nav.iloc[-1]-INITIAL),
        'all_period_concentration':concentration.iloc[0].to_dict(),
        'yearly_net_pnl_cny':{str(y):float(g.net_pnl.sum()) for y,g in yearly.groupby('year')},
        'yearly_log_relative':active_yearly.to_dict('records'),
        'factor_full_period_bh_q_lt_0_05':factors.loc[(factors.period=='all') & (factors.bh_q_28<.05),'factor'].tolist(),
        'monthly_technical_exposure_dates':len(technical),
        'corporate_action_case_adjusted_minus_hypothetical_bp':action['adjusted_minus_hypothetical_return_bp'],
        'external_data_status':{'historical_industry':'unavailable','historical_csi300_security_weights':'unavailable',
          'historical_market_cap_and_book_value':'unavailable','independent_a_share_factor_returns':'separate CH4 module; see ch4_manifest.json',
          'independent_corporate_actions':'one issuer notice checked; independent raw quotes and complete ledger unavailable',
          'csi300_total_return_levels':'unavailable'}}
    PRIVATE.mkdir(parents=True,exist_ok=True)
    (PRIVATE/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False,default=str),encoding='utf-8')
    print(json.dumps(manifest,indent=2,ensure_ascii=False,default=str))
    print('2024 gainers/detractors:', top.loc[top.year=='2024'].head(3).to_dict('records'))
    print('2025 gainers/detractors:', top.loc[top.year=='2025'].head(3).to_dict('records'))


if __name__ == '__main__':
    main()
