"""Equity Alpha Lab 2.0 | CSI300 monthly ML factor research, V1.

Run via PyCharm green Run. Uses local Qlib daily CN data.
No future outcome is used for stock eligibility; labels used for training
are restricted to periods whose exit date is <= decision date.
Execution uses a cash-and-shares monthly paper account. Missing execution
opens become unfilled orders; positions are marked using last known open.
This is an exploratory research pipeline, not a production trading system.
"""
from __future__ import annotations

# IMPORTANT on Windows: set native numerical-library thread limits BEFORE
# importing NumPy, SciPy, scikit-learn or LightGBM. These do not replace
# Qlib's own process-count limit, which is configured in main().
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

from pathlib import Path
import json
import time
import warnings
import multiprocessing as mp

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import qlib
from qlib.data import D

# ------------------------- USER CONFIG -------------------------
QLIB_DATA = Path.home() / ".qlib" / "qlib_data" / "cn_data"
START = "2012-01-01"
END = "2020-08-31"
TEST_START = "2017-01-01"   # explored previously; development, NOT pristine holdout
MIN_TRAIN_MONTHS = 24
TRAIN_LOOKBACK_MONTHS = 48
TOP_N = 30
BUFFER_N = 60
SINGLE_SIDE_COST_BPS = 5
RIDGE_ALPHA = 10.0
MIN_CANDIDATES = 150
SEED = 42
SAVE_DIR = Path(__file__).resolve().parent / "outputs" / "v1_csi300"
# ---------------------------------------------------------------

FACTOR_NAMES = [
    "mom_20", "mom_60", "reversal_5", "vol_20",
    "vol_60", "volume_surprise", "range_20", "ma_gap_20",
]
MODELS = ["EqualFactor", "Ridge", "LightGBM"]


def check(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def market_days():
    idx = pd.DatetimeIndex(D.calendar(start_time=START, end_time=END, freq="day"))
    check(len(idx) > 300 and idx.is_unique, "Invalid Qlib trading calendar")
    return idx.sort_values()


def month_ends(calendar):
    # calendar month-end observation; no calendar-day execution assumptions
    grouped = pd.Series(calendar, index=calendar).groupby(calendar.to_period("M")).last()
    return list(pd.DatetimeIndex(grouped.values))


def historical_membership():
    raw = D.list_instruments(D.instruments(market="csi300"),
        start_time=START, end_time=END, freq="day", as_list=False)
    check(isinstance(raw, dict) and raw, "Historical CSI300 membership intervals unavailable")
    return {str(k).upper(): [(pd.Timestamp(a).normalize(), pd.Timestamp(b).normalize())
            for a, b in intervals] for k, intervals in raw.items()}


def fetch_market_panel(codes):
    raw = D.features(instruments=sorted(codes), fields=["$close", "$open", "$volume", "$high", "$low"],
        start_time=START, end_time=END, freq="day")
    check(isinstance(raw.index, pd.MultiIndex), "Unexpected Qlib market index")
    raw = raw.reorder_levels(["datetime", "instrument"]).sort_index()
    raw.columns = ["close", "open", "volume", "high", "low"]
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw.index = pd.MultiIndex.from_arrays([
        pd.to_datetime(raw.index.get_level_values("datetime")).normalize(),
        raw.index.get_level_values("instrument").astype(str).str.upper()],
        names=["datetime", "instrument"])
    check(raw.index.is_unique, "Duplicate Qlib prices")
    return raw


def get_index_open():
    frame = D.features(instruments=["SH000300"], fields=["$open"],
        start_time=START, end_time=END, freq="day")
    check(not frame.empty, "Missing SH000300 benchmark open")
    s = frame.iloc[:, 0].droplevel("instrument")
    s.index = pd.to_datetime(s.index).normalize()
    return s.sort_index()


def snapshot(date, intervals, panel, calendar):
    dt = pd.Timestamp(date)
    members = [s for s, spans in intervals.items()
               if any(a <= dt <= b for a, b in spans)]
    check(200 <= len(members) <= 400, f"Suspicious PIT membership {dt}: {len(members)}")
    # All rolling features use data through the signal close(t) only.
    t_index = calendar.get_loc(dt)
    history_dates = calendar[max(0, t_index - 95):t_index + 1]
    history = panel.loc[panel.index.get_level_values("datetime").isin(history_dates)]
    # Individual stock price observations are aligned to the calendar (no silent ffill).
    close = history["close"].unstack("instrument").reindex(index=history_dates, columns=members)
    vol = history["volume"].unstack("instrument").reindex(index=history_dates, columns=members)
    high = history["high"].unstack("instrument").reindex(index=history_dates, columns=members)
    low = history["low"].unstack("instrument").reindex(index=history_dates, columns=members)
    ret = close.pct_change(fill_method=None)
    f = pd.DataFrame(index=pd.Index(members, name="instrument"))
    f["mom_20"] = (close.iloc[-1] / close.iloc[-21] - 1).values if len(close) >= 21 else np.nan
    f["mom_60"] = (close.iloc[-1] / close.iloc[-61] - 1).values if len(close) >= 61 else np.nan
    f["reversal_5"] = (1 - close.iloc[-1] / close.iloc[-6]).values if len(close) >= 6 else np.nan
    f["vol_20"] = ret.iloc[-20:].std(ddof=1).values
    f["vol_60"] = ret.iloc[-60:].std(ddof=1).values
    vbase = vol.shift(1).iloc[-21:-1].mean()
    f["volume_surprise"] = np.log((vol.iloc[-1] / vbase).where(vbase > 0)).values
    f["range_20"] = ((high.iloc[-20:] - low.iloc[-20:]) / close.iloc[-20:]).mean().values
    f["ma_gap_20"] = (close.iloc[-1] / close.iloc[-20:].mean() - 1).values
    # Current-day availability is observable when forming the post-close signal.
    eligible = (close.iloc[-1].notna() & (close.iloc[-1] > 0) &
                vol.iloc[-1].notna() & (vol.iloc[-1] > 0))
    f = f.loc[eligible.reindex(f.index).fillna(False).to_numpy()]
    check(len(f) >= MIN_CANDIDATES, f"Too few as-of candidates on {dt}: {len(f)}")
    # NaNs are allowed in factors, but not infinite values.
    f = f.replace([np.inf, -np.inf], np.nan)
    # Pure as-of cross-sectional normalization (no future labels).
    for col in FACTOR_NAMES:
        s = f[col]
        if s.notna().sum() < 20:
            continue
        lower, upper = s.quantile([.01, .99])
        s = s.clip(lower, upper)
        sd = s.std(ddof=0)
        f[col] = (s - s.mean()) / sd if pd.notna(sd) and sd > 1e-12 else 0.0
    f["date"] = dt
    return f.reset_index().set_index(["date", "instrument"])[FACTOR_NAMES]


def create_monthly_dataset(calendar, intervals, prices):
    ends = month_ends(calendar)
    # t signal at close(month end); entry next trading day's open.
    dates = [d for d in ends if calendar.get_loc(d) + 1 < len(calendar)]
    entry_by_signal = {d: calendar[calendar.get_loc(d) + 1] for d in dates}
    # Last signal without a following monthly exit is excluded.
    decisions = dates[:-1]
    open_px = prices["open"].unstack("instrument").reindex(calendar)
    blocks = []
    for n, date in enumerate(decisions, 1):
        x = snapshot(date, intervals, prices, calendar)
        entry = entry_by_signal[date]
        next_entry = entry_by_signal[dates[dates.index(date)+1]]
        # The outcome is stored separately; NEVER filter stock eligibility by it.
        start_open = open_px.reindex(index=[entry], columns=x.index.get_level_values("instrument")).iloc[0]
        end_open = open_px.reindex(index=[next_entry], columns=x.index.get_level_values("instrument")).iloc[0]
        actual_ret = (end_open / start_open - 1).where((start_open > 0) & (end_open > 0))
        x["forward_return"] = actual_ret.reindex(x.index.get_level_values("instrument")).to_numpy()
        x["entry_date"] = entry
        x["exit_date"] = next_entry
        blocks.append(x)
        if n % 24 == 0:
            print(f"  built {n}/{len(decisions)} monthly snapshots")
    result = pd.concat(blocks).sort_index()
    check(result.index.is_unique, "Duplicate snapshot stock-dates")
    result.to_pickle(SAVE_DIR / "monthly_asof_panel.pkl")
    return result, entry_by_signal


def model_scores(training, predict_x):
    # Only labels that have already fully materialized before signal date.
    tr = training.dropna(subset=["forward_return"])
    check(len(tr) >= 1000, "Insufficient label-complete training rows")
    X, y = tr[FACTOR_NAMES], tr["forward_return"].to_numpy()
    Xnow = predict_x[FACTOR_NAMES]
    # Economic factor signs predefined from conventional priors; not optimized on test.
    weights = np.array([1, 1, 1, -1, -1, -1, -1, 1], dtype=float)
    equal = Xnow.fillna(0).to_numpy() @ weights / np.abs(weights).sum()
    ridge = make_pipeline(SimpleImputer(strategy="median", add_indicator=False),
                          StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    ridge.fit(X, y)
    gbm = lgb.LGBMRegressor(n_estimators=120, learning_rate=.035, num_leaves=15,
        max_depth=5, min_child_samples=80, reg_lambda=10, verbosity=-1,
        random_state=SEED, n_jobs=1)
    gbm.fit(X, y)
    return {"EqualFactor": equal, "Ridge": ridge.predict(Xnow),
            "LightGBM": gbm.predict(Xnow)}


def generate_oos(panel):
    dates = pd.DatetimeIndex(panel.index.get_level_values("date").unique()).sort_values()
    chunks = []
    monthly_diagnostics = []
    for i, dt in enumerate(dates):
        if dt < pd.Timestamp(TEST_START) or i < MIN_TRAIN_MONTHS + 2:
            continue
        history_start = dates[max(0, i - TRAIN_LOOKBACK_MONTHS)]
        # Crucial: train rows must satisfy exit_date <= current signal date.
        hist = panel.loc[(panel.index.get_level_values("date") >= history_start) &
                         (panel.index.get_level_values("date") < dt)]
        train = hist.loc[hist["exit_date"] <= dt]
        months = train.index.get_level_values("date").nunique()
        if months < MIN_TRAIN_MONTHS:
            continue
        today = panel.xs(dt, level="date", drop_level=False)
        scores = model_scores(train, today)
        for name, yhat in scores.items():
            part = today[["forward_return", "entry_date", "exit_date"]].copy()
            part["score"] = yhat
            part["model"] = name
            part = part.reset_index()
            chunks.append(part)
            clean = part.dropna(subset=["forward_return", "score"])
            ic = clean["score"].corr(clean["forward_return"], method="spearman") if len(clean) >= 30 else np.nan
            monthly_diagnostics.append({"date": dt, "model": name, "rank_ic": ic,
                "candidates": len(part), "label_available": len(clean),
                "train_months": months})
        print(f"  OOS {dt.date()} | train={months} months | candidates={len(today)}")
    check(chunks, "No OOS predictions; inspect time window and data coverage")
    preds = pd.concat(chunks, ignore_index=True)
    check(not preds.duplicated(["date", "instrument", "model"]).any(), "Duplicate OOS predictions")
    preds.to_csv(SAVE_DIR / "predictions.csv", index=False)
    diagnostics = pd.DataFrame(monthly_diagnostics)
    diagnostics.to_csv(SAVE_DIR / "prediction_diagnostics.csv", index=False)
    return preds, diagnostics


def simulate(preds, all_prices, benchmark):
    """Cash-and-shares monthly paper ledger (fractional shares, not live execution).

    Orders at an entry date use that day's OPEN only. No future exit price is
    checked when deciding which assets to buy. When an open is missing, no
    transaction occurs. Existing shares remain held; cash is not force-invested.

    For mark-to-market only, a missing open is valued using the most recent
    positive OPEN at or before the valuation date. Such marks are stale and
    MUST NOT be interpreted as executable fills. Adjusted Qlib OHLC values are
    used consistently, but this remains a simplified research simulator.
    """
    opens = all_prices["open"].unstack("instrument").sort_index()
    opens = opens.where(np.isfinite(opens) & (opens > 0))
    benchmark = benchmark.sort_index().where(lambda x: np.isfinite(x) & (x > 0))
    dates_available = opens.index
    last_open = opens.ffill()  # strictly historical as-of prices for valuation ONLY
    fee = SINGLE_SIDE_COST_BPS / 10000.0
    records = []
    order_events = []
    holdings_records = []

    def mark(date, symbol):
        if date not in last_open.index or symbol not in last_open.columns:
            raise RuntimeError(f"No price series for {symbol} on {date}")
        price = float(last_open.at[date, symbol])
        check(np.isfinite(price) and price > 0,
              f"No historical price for held {symbol} as of {date}; cannot value portfolio")
        today = opens.at[date, symbol]
        stale = not bool(np.isfinite(today) and today > 0)
        return price, stale

    def nav(date, shares, cash):
        total = float(cash)
        stale_positions = 0
        for sym, qty in shares.items():
            if qty <= 1e-12:
                continue
            price, stale = mark(date, sym)
            total += qty * price
            stale_positions += int(stale)
        check(np.isfinite(total) and total > 0,
              f"Invalid portfolio NAV on {date}")
        return total, stale_positions

    for model in MODELS:
        group = preds.loc[preds["model"] == model].copy()
        group["date"] = pd.to_datetime(group["date"])
        signal_dates = sorted(group["date"].unique())
        for rule in ["Top30", "Buffer30_60"]:
            cash = 1.0
            shares = {}
            prior_exit = None
            for j, date in enumerate(signal_dates):
                day = (group.loc[group["date"] == date]
                       .sort_values(["score", "instrument"], ascending=[False, True]))
                entry = pd.Timestamp(day["entry_date"].iloc[0])
                exit_day = pd.Timestamp(day["exit_date"].iloc[0])
                check(entry in dates_available and exit_day in dates_available,
                      f"Missing trading calendar entry/exit: {entry} / {exit_day}")
                if prior_exit is not None:
                    check(entry == prior_exit,
                          f"Noncontiguous periods: {model}, {rule}, {date}")
                start_nav, start_stale = nav(entry, shares, cash)
                scores = day["instrument"].tolist()
                check(len(scores) >= BUFFER_N and len(set(scores)) == len(scores),
                      f"Bad prediction universe on {date}")
                if rule == "Top30":
                    selected = scores[:TOP_N]
                else:
                    buffer_set = set(scores[:BUFFER_N])
                    retained = [s for s in shares if shares[s] > 1e-12 and s in buffer_set]
                    # Previous positions that cannot be sold may remain anyway.
                    selected = retained[:TOP_N] + [s for s in scores if s not in retained][:max(0, TOP_N-len(retained))]
                check(len(selected) == TOP_N and len(set(selected)) == TOP_N,
                      "Incorrect target stock count")
                target_values = {s: start_nav / TOP_N for s in selected}
                fees_paid = 0.0
                trade_notional = 0.0
                blocked_buys = 0
                blocked_sells = 0
                # SELL reductions first. Non-tradable holdings are NOT removed.
                for sym in list(shares):
                    held = shares.get(sym, 0.0)
                    if held <= 1e-12:
                        shares.pop(sym, None)
                        continue
                    value_price, _ = mark(entry, sym)
                    desired_qty = target_values.get(sym, 0.0) / value_price
                    excess = max(0.0, held - desired_qty)
                    if excess <= 1e-12:
                        continue
                    trade_px = opens.at[entry, sym]
                    if not np.isfinite(trade_px) or trade_px <= 0:
                        blocked_sells += 1
                        order_events.append(dict(date=date, execution_date=entry,
                            model=model, strategy=rule, instrument=sym,
                            side="SELL", status="BLOCKED_NO_OPEN", shares=excess,
                            notional=0.0))
                        continue
                    proceeds = excess * float(trade_px)
                    cost = proceeds * fee
                    cash += proceeds - cost
                    shares[sym] -= excess
                    if shares[sym] <= 1e-12:
                        del shares[sym]
                    fees_paid += cost
                    trade_notional += proceeds
                    order_events.append(dict(date=date, execution_date=entry,
                        model=model, strategy=rule, instrument=sym,
                        side="SELL", status="FILLED", shares=excess,
                        notional=proceeds))
                # BUY at the *same* entry open using available cash.
                # A price-less symbol stays unfilled; we DO NOT redistribute its
                # allocation to the other stocks to chase a perfect backtest.
                for sym in selected:
                    trade_px = opens.at[entry, sym]
                    if not np.isfinite(trade_px) or trade_px <= 0:
                        blocked_buys += 1
                        order_events.append(dict(date=date, execution_date=entry,
                            model=model, strategy=rule, instrument=sym,
                            side="BUY", status="BLOCKED_NO_OPEN", shares=0.0,
                            notional=0.0))
                        continue
                    trade_px = float(trade_px)
                    existing = shares.get(sym, 0.0) * trade_px
                    needed = max(0.0, target_values[sym] - existing)
                    spend = min(needed, max(0.0, cash) / (1.0 + fee))
                    if spend <= 1e-12:
                        continue
                    qty = spend / trade_px
                    cash -= spend * (1.0 + fee)
                    shares[sym] = shares.get(sym, 0.0) + qty
                    fees_paid += spend * fee
                    trade_notional += spend
                    order_events.append(dict(date=date, execution_date=entry,
                        model=model, strategy=rule, instrument=sym,
                        side="BUY", status="FILLED", shares=qty,
                        notional=spend))
                check(cash >= -1e-8, f"Negative cash: {cash}")
                cash = max(cash, 0.0)
                after_trade, _ = nav(entry, shares, cash)
                pre_exit_nav, stale_end = nav(exit_day, shares, cash)
                liquidation_fee = 0.0
                terminal_unliquidated = 0
                if j == len(signal_dates) - 1:
                    # Terminal inventory must be visible, not assumed liquidated.
                    for sym in list(shares):
                        px = opens.at[exit_day, sym]
                        if not np.isfinite(px) or px <= 0:
                            terminal_unliquidated += 1
                            order_events.append(dict(date=date, execution_date=exit_day,
                                model=model, strategy=rule, instrument=sym,
                                side="SELL", status="BLOCKED_TERMINAL_NO_OPEN",
                                shares=shares[sym], notional=0.0))
                            continue
                        proceeds = shares[sym] * float(px)
                        cost = proceeds * fee
                        cash += proceeds - cost
                        liquidation_fee += cost
                        trade_notional += proceeds
                        order_events.append(dict(date=date, execution_date=exit_day,
                            model=model, strategy=rule, instrument=sym,
                            side="SELL", status="FILLED_TERMINAL",
                            shares=shares[sym], notional=proceeds))
                        del shares[sym]
                end_nav, _ = nav(exit_day, shares, cash)
                gross_holding_return = pre_exit_nav / after_trade - 1.0
                net_return = end_nav / start_nav - 1.0
                b_entry = float(benchmark.loc[entry])
                b_exit = float(benchmark.loc[exit_day])
                check(np.isfinite(b_entry) and b_entry > 0 and
                      np.isfinite(b_exit) and b_exit > 0,
                      f"Missing CSI300 benchmark open on {entry} or {exit_day}")
                records.append(dict(date=pd.Timestamp(date), entry_date=entry,
                    exit_date=exit_day, model=model, strategy=rule,
                    gross=gross_holding_return, net=net_return,
                    benchmark=b_exit / b_entry - 1.0,
                    traded_notional=trade_notional / start_nav,
                    cost_fraction=(fees_paid + liquidation_fee) / start_nav,
                    blocked_buys=blocked_buys, blocked_sells=blocked_sells,
                    stale_at_entry=start_stale, stale_at_exit=stale_end,
                    terminal_unliquidated=terminal_unliquidated,
                    cash_weight=cash / end_nav, n_holdings=len(shares),
                    wealth=end_nav))
                for sym, qty in shares.items():
                    px, stale = mark(exit_day, sym)
                    holdings_records.append(dict(date=pd.Timestamp(date),
                        valuation_date=exit_day, model=model, strategy=rule,
                        instrument=sym, shares=qty, marked_price=px,
                        marked_value=qty*px, stale=stale))
                prior_exit = exit_day
            print(f"  simulated {model:12s} {rule:12s} "
                  f"final NAV={end_nav:.5f} | blocked orders="
                  f"{sum(x['status'].startswith('BLOCKED') for x in order_events if x['model']==model and x['strategy']==rule)}")
    result = pd.DataFrame(records)
    result.to_csv(SAVE_DIR / "paper_backtest.csv", index=False)
    pd.DataFrame(order_events).to_csv(SAVE_DIR / "execution_audit.csv", index=False)
    pd.DataFrame(holdings_records).to_csv(SAVE_DIR / "holdings_audit.csv", index=False)
    return result


def performance(returns, bench):
    r = np.asarray(returns, float)
    b = np.asarray(bench, float)
    growth = np.r_[1., np.cumprod(1 + r)]
    drawdown = growth / np.maximum.accumulate(growth) - 1
    vol = float(np.std(r, ddof=1) * np.sqrt(12)) if len(r) > 1 else np.nan
    ann = float(growth[-1] ** (12 / len(r)) - 1)
    excess = r - b
    tracking = float(np.std(excess, ddof=1) * np.sqrt(12)) if len(r)>1 else np.nan
    return {"months": len(r), "total_return": growth[-1]-1,
        "annualized_return": ann, "annualized_volatility": vol,
        "sharpe_rf0": float((np.mean(r)*12)/vol) if vol > 0 else np.nan,
        "max_drawdown": float(drawdown.min()), "mean_monthly_excess": float(excess.mean()),
        "information_ratio": float(excess.mean()*12/tracking) if tracking > 0 else np.nan}


def export_reports(preds, ic, backtest):
    rows = []
    for (m, s), g in backtest.groupby(["model", "strategy"]):
        stats = performance(g["net"], g["benchmark"])
        stats.update({"model": m, "strategy": s, "avg_monthly_turnover": g["traded_notional"].mean(),
            "mean_rank_ic": float(ic.loc[ic["model"] == m, "rank_ic"].mean())})
        rows.append(stats)
    table = pd.DataFrame(rows).sort_values(["strategy", "model"])
    table.to_csv(SAVE_DIR / "performance_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(11, 6))
    for (m, s), g in backtest.groupby(["model", "strategy"]):
        g = g.sort_values("exit_date")
        ax.plot(g["exit_date"], g["wealth"], label=f"{m} | {s}")
    first = backtest.loc[(backtest["model"] == MODELS[0]) &
                          (backtest["strategy"] == "Top30")].sort_values("exit_date")
    bench_curve = (1 + first["benchmark"]).cumprod()
    ax.plot(first["exit_date"], bench_curve, label="CSI300", linestyle="--", color="black")
    ax.set_title("CSI300 monthly paper portfolios | not executable P&L")
    ax.set_ylabel("Growth of 1")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(SAVE_DIR / "equity_curves.png", dpi=160)
    plt.close(fig)
    (SAVE_DIR / "run_metadata.json").write_text(json.dumps({
        "start": START, "end": END, "test_start": TEST_START, "market": "csi300",
        "factor_names": FACTOR_NAMES, "models": MODELS,
        "cost_single_side_bps": SINGLE_SIDE_COST_BPS,
        "note": "Exploratory adjusted-price fractional-share paper account; missing opens block orders; last historical open used for stale marks; no lot/auction/limit, and not executable P&L."}, indent=2), encoding="utf-8")
    print("\n=== PERFORMANCE SUMMARY ===\n", table.round(4).to_string(index=False))
    print("\n=== MEAN MONTHLY RANKIC ===\n", ic.groupby("model")["rank_ic"].agg(["mean", "std", "count"]).round(5))


def main():
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    check(QLIB_DATA.is_dir(), f"Qlib data path missing: {QLIB_DATA}")
    # Qlib's default kernel count can spawn many Python worker processes.
    # On Windows each spawned interpreter reimports large ML libraries;
    # this previously exhausted the system's RAM/page-file commit limit.
    print("Resource-safe mode: Qlib kernels=1, LightGBM n_jobs=1, BLAS threads=1", flush=True)
    qlib.init(provider_uri=str(QLIB_DATA), region="cn", kernels=1)
    calendar = market_days()
    intervals = historical_membership()
    print(f"Historical CSI300 stock codes: {len(intervals)}; calendar days: {len(calendar)}")
    prices = fetch_market_panel(intervals)
    benchmark = get_index_open()
    panel, _ = create_monthly_dataset(calendar, intervals, prices)
    print("As-of panel:", panel.shape, "missing forward labels:", panel["forward_return"].isna().sum())
    preds, ic = generate_oos(panel)
    backtest = simulate(preds, prices, benchmark)
    export_reports(preds, ic, backtest)
    print(f"\nV1 completed in {time.perf_counter()-t0:.1f}s; output: {SAVE_DIR}")


if __name__ == "__main__":
    mp.freeze_support()
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
