"""Equity Alpha Lab 2.0 — V1.1 independent accounting & execution audit.

Place beside run_research.py; run directly in PyCharm. Reads saved V1 outputs,
NO retraining and NO Qlib/ML imports. Does not rewrite original outputs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "outputs" / "v1_csi300"
OUT = SOURCE / "v11_audit"
ABS_TOL = 2e-7
REL_TOL = 2e-6


def require(condition, message):
    if not bool(condition):
        raise RuntimeError(message)


def load(name, dates=()):
    path = SOURCE / name
    require(path.is_file(), f"Missing required file: {path}")
    frame = pd.read_csv(path, parse_dates=list(dates), low_memory=False)
    require(not frame.empty, f"Empty file: {path}")
    return frame


def numeric(df, cols, table):
    for col in cols:
        require(col in df, f"{table}: missing column '{col}'")
        df[col] = pd.to_numeric(df[col], errors="coerce")
        require(np.isfinite(df[col].to_numpy(dtype=float)).all(),
                f"{table}: nonfinite {col}")


def near(a, b, label, anomalies, meta=None, tol=ABS_TOL):
    if not np.isfinite(a) or not np.isfinite(b) or abs(float(a)-float(b)) > tol + REL_TOL * max(abs(a),abs(b)):
        anomalies.append(dict(level="FAIL", check=label, expected=b, actual=a,
                              difference=float(a-b), **(meta or {})))
        return False
    return True


def audit():
    require(SOURCE.is_dir(), f"V1 outputs do not exist: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    perf = load("paper_backtest.csv", ("date", "entry_date", "exit_date"))
    orders = load("execution_audit.csv", ("date", "execution_date"))
    holdings = load("holdings_audit.csv", ("date", "valuation_date"))
    predictions = load("predictions.csv", ("date", "entry_date", "exit_date"))
    meta_path = SOURCE / "run_metadata.json"
    require(meta_path.is_file(), f"Missing run_metadata.json: {meta_path}")
    conf = json.loads(meta_path.read_text(encoding="utf-8"))
    bps = float(conf["cost_single_side_bps"])
    rate = bps / 10000.0
    key = ["model", "strategy"]
    period_key = key + ["date"]
    required = ["gross","net","benchmark","traded_notional","cost_fraction",
                "blocked_buys","blocked_sells","stale_at_entry","stale_at_exit",
                "terminal_unliquidated","cash_weight","n_holdings","wealth"]
    numeric(perf,required,"paper_backtest")
    numeric(orders,["shares","notional"],"execution_audit")
    numeric(holdings,["shares","marked_price","marked_value"],"holdings_audit")
    numeric(predictions,["score"],"predictions")
    require(not perf.duplicated(period_key).any(), "Duplicate monthly account rows")
    require(not predictions.duplicated(["date","instrument","model"]).any(),
            "Duplicate stock predictions")
    require(orders["side"].isin(["BUY","SELL"]).all(),"Unexpected order side")
    require((orders[["shares","notional"]]>=0).all().all(),"Negative order quantities")
    require((holdings[["shares","marked_price","marked_value"]]>=0).all().all(),"Negative holdings")

    anomalies = []
    account_rows = []
    strategy_rows = []
    order_rollups = []
    blocked_records = []
    holdings_idx = {tuple(k): g for k,g in holdings.groupby(period_key,sort=False)}
    order_idx = {tuple(k): g for k,g in orders.groupby(period_key,sort=False)}
    pred_idx = {tuple(k): g for k,g in predictions.groupby(["model","date"],sort=False)}

    for (model, strategy), grp in perf.groupby(key,sort=True):
        grp = grp.sort_values("entry_date")
        prev_wealth = 1.0
        prev_exit = None
        cumulative = 1.0
        sum_trades = 0.0
        stale_months = 0
        blocked_total = 0
        for j, p in enumerate(grp.itertuples(index=False)):
            idx = (model,strategy,p.date)
            m = {"model":model,"strategy":strategy,"date":p.date}
            require(p.exit_date > p.entry_date, f"Invalid monthly period: {idx}")
            if prev_exit is not None and prev_exit != p.entry_date:
                anomalies.append(dict(level="FAIL",check="noncontiguous_periods",**m,
                                      expected=prev_exit,actual=p.entry_date))
            near(p.wealth,prev_wealth*(1+p.net),"wealth_return_identity",anomalies,m)
            cumulative *= 1+p.net
            near(p.wealth,cumulative,"compounded_return_identity",anomalies,m)
            require(0 <= p.cash_weight <= 1 + 1e-7, f"Invalid cash weight {idx}")
            require(p.wealth > 0, f"Invalid account wealth {idx}")
            h = holdings_idx.get(idx,pd.DataFrame(columns=holdings.columns))
            o = order_idx.get(idx,pd.DataFrame(columns=orders.columns))
            signal = pred_idx.get((model,p.date))
            if signal is None:
                anomalies.append(dict(level="FAIL",check="missing_prediction_date",**m))
            else:
                dates = signal[["entry_date","exit_date"]].drop_duplicates()
                if len(dates)!=1 or pd.Timestamp(dates.iloc[0]["entry_date"])!=p.entry_date or pd.Timestamp(dates.iloc[0]["exit_date"])!=p.exit_date:
                    anomalies.append(dict(level="FAIL",check="prediction_execution_date_mismatch",**m))

            near(h.marked_value.sum(),p.wealth*(1-p.cash_weight),
                 "holdings_plus_cash_equals_nav",anomalies,m)
            for r in h.itertuples(index=False):
                near(r.marked_value,r.shares*r.marked_price,
                     "stock_shares_times_mark",anomalies,{**m,"instrument":r.instrument})
                if r.valuation_date!=p.exit_date:
                    anomalies.append(dict(level="FAIL",check="wrong_holdings_valuation_date",**m,
                                          instrument=r.instrument))
            if int(p.n_holdings)!=len(h):
                anomalies.append(dict(level="FAIL",check="holdings_count",**m,
                                      expected=int(p.n_holdings),actual=len(h)))
            # The published simulator has both filled opening orders and
            # filled terminal liquidation orders on the LAST period.
            filled = o.loc[o.status.isin(["FILLED","FILLED_TERMINAL"])]
            blocked = o.loc[o.status.astype(str).str.startswith("BLOCKED")]
            notional = float(filled.notional.sum())
            fee = notional * rate
            near(p.traded_notional,notional/prev_wealth,
                 "turnover_from_fills",anomalies,m)
            near(p.cost_fraction,fee/prev_wealth,
                 "fees_from_fills",anomalies,m)
            if (o.loc[~o.status.isin(["FILLED","FILLED_TERMINAL"]),"notional"] > 1e-10).any():
                anomalies.append(dict(level="FAIL",check="nonfilled_order_has_notional",**m))
            bad_buy=int(((o.side=="BUY") & (o.status=="BLOCKED_NO_OPEN")).sum())
            bad_sell=int(((o.side=="SELL") & (o.status=="BLOCKED_NO_OPEN")).sum())
            bad_terminal=int((o.status=="BLOCKED_TERMINAL_NO_OPEN").sum())
            if bad_buy!=int(p.blocked_buys) or bad_sell!=int(p.blocked_sells):
                anomalies.append(dict(level="FAIL",check="blocked_order_count",**m,
                                      expected=f"{bad_buy}/{bad_sell}",actual=f"{p.blocked_buys}/{p.blocked_sells}"))
            if bad_terminal!=int(p.terminal_unliquidated):
                anomalies.append(dict(level="FAIL",check="terminal_blocked_count",**m,
                                      expected=bad_terminal,actual=p.terminal_unliquidated))
            # After the last rebalance, all liquidatable positions should be sold.
            if j==len(grp)-1 and len(h)!=bad_terminal:
                anomalies.append(dict(level="WARN",check="terminal_inventory_mismatch",**m,
                                      expected=bad_terminal,actual=len(h)))
            stale_count=int(h.stale.fillna(False).astype(str).str.lower().eq("true").sum()) if len(h) else 0
            if int(p.stale_at_exit)!=stale_count:
                anomalies.append(dict(level="FAIL",check="stale_marks_count",**m,
                                      expected=stale_count,actual=p.stale_at_exit))
            if p.stale_at_entry>0 or p.stale_at_exit>0: stale_months+=1
            blocked_total+=len(blocked)
            sum_trades+=notional
            for r in blocked.itertuples(index=False):
                blocked_records.append(dict(model=model,strategy=strategy,date=p.date,
                                    execution_date=r.execution_date,instrument=r.instrument,
                                    side=r.side,status=r.status,shares=r.shares))
            account_rows.append(dict(**m,entry_date=p.entry_date,exit_date=p.exit_date,
                initial_equity=prev_wealth,end_equity=p.wealth,reported_net=p.net,
                computed_net=p.wealth/prev_wealth-1,holdings_value=h.marked_value.sum(),
                inferred_cash=p.cash_weight*p.wealth,fees_recomputed=fee,
                trades_notional=notional,filled_orders=len(filled),blocked_orders=len(blocked),
                stale_exit_positions=stale_count,terminal_unsold=bad_terminal))
            order_rollups.append(dict(**m,buys=int(((filled.side=="BUY")).sum()),
                sells=int(((filled.side=="SELL")).sum()),blocked_buy=bad_buy,
                blocked_sell=bad_sell,blocked_terminal=bad_terminal))
            prev_wealth=p.wealth
            prev_exit=p.exit_date
        final=grp.iloc[-1]
        strategy_rows.append(dict(model=model,strategy=strategy,months=len(grp),
            first_entry=grp.entry_date.min(),final_exit=grp.exit_date.max(),
            final_nav=final.wealth,cumulative_net_return=final.wealth-1,
            mean_net_return=grp.net.mean(),mean_benchmark_return=grp.benchmark.mean(),
            mean_monthly_excess=(grp.net-grp.benchmark).mean(),
            mean_turnover=grp.traded_notional.mean(),
            total_blocked_orders=blocked_total,months_with_stale_marks=stale_months,
            terminal_unliquidated=int(final.terminal_unliquidated)))

    # Evaluate date-level benchmark agreement between all strategies.
    b=perf.pivot_table(index=["entry_date","exit_date"],columns=key,values="benchmark")
    benchmark_spread=(b.max(axis=1)-b.min(axis=1)).max()
    if benchmark_spread>ABS_TOL:
        anomalies.append(dict(level="FAIL",check="benchmark_inconsistent_between_strategies",
                              actual=float(benchmark_spread)))
    # Only internal checks: no independent exchange price series is provided.
    blocked_df=pd.DataFrame(blocked_records)
    accounts_df=pd.DataFrame(account_rows)
    summary_df=pd.DataFrame(strategy_rows)
    event_df=pd.DataFrame(order_rollups)
    anomalies_df=pd.DataFrame(anomalies,columns=["level","check","model","strategy","date","instrument","expected","actual","difference"])
    accounts_df.to_csv(OUT/"account_reconciliation.csv",index=False,encoding="utf-8-sig")
    summary_df.to_csv(OUT/"strategy_audit_summary.csv",index=False,encoding="utf-8-sig")
    event_df.to_csv(OUT/"order_counts_by_month.csv",index=False,encoding="utf-8-sig")
    blocked_df.to_csv(OUT/"blocked_orders.csv",index=False,encoding="utf-8-sig")
    anomalies_df.to_csv(OUT/"audit_anomalies.csv",index=False,encoding="utf-8-sig")

    nfail=int((anomalies_df.level=="FAIL").sum())
    nwarn=int((anomalies_df.level=="WARN").sum())
    summary={"status":"FAIL" if nfail else ("PASS_WITH_LIMITATIONS" if nwarn else "PASS_INTERNAL_IDENTITIES_ONLY"),
      "failed_checks":nfail,"warnings":nwarn,"strategies":len(summary_df),
      "account_months":len(accounts_df),"benchmark_max_spread":float(benchmark_spread),
      "cost_single_side_bps":bps,"source_folder":str(SOURCE),
      "limitations":[
        "Trade execution uses same-day adjusted open in simulator, not an independent auction fill feed.",
        "Missing opens are modeled as unfillable; this does not test limit-up/limit-down or order-book liquidity.",
        "Stale marking uses previous adjusted open and may misstate NAV through halts/corporate actions.",
        "Qlib adjusted prices may not map directly to executable cash/share quantities through corporate actions.",
        "No independent per-trade cash/share ledger snapshots: this checks output identities, not complete transaction-by-transaction self-financing.",
        "Future-label availability and research-period reuse still limit independent inference.",
        "Holding values and reported cash fraction are jointly simulator-generated, so agreement alone cannot prove market realism.",
        "The reported gross field is holding-period NAV change after entry trading, not fully fee-free strategy return.",
      ]}
    (OUT/"audit_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    report=["# Equity Alpha Lab V1.1 — Account and Execution Audit","",
        f"Status: **{summary['status']}**",f"Internal failures: {nfail}; warnings: {nwarn}",
        f"Strategy configurations: {len(summary_df)}; account-months: {len(accounts_df)}","",
        "## Summary","```\n"+summary_df.to_string(index=False)+"\n```","",
        "## Key limitations",*(f"- {x}" for x in summary["limitations"]),"",
        "## Interpretation",
        "A passed audit establishes internal consistency of existing CSV outputs, not exchange-level execution validity.",
        "Cross-check adjusted-price execution against raw prices/corporate actions before reporting investable P&L."]
    try:
        text="\n".join(report)
    except Exception:
        text="Audit completed"
    (OUT/"audit_report.md").write_text(text,encoding="utf-8")
    print("="*78)
    print("EQUITY ALPHA LAB 2.0 — V1.1 ACCOUNT / EXECUTION AUDIT")
    print("="*78)
    print("Source:",SOURCE)
    print("\nSTRATEGY SUMMARY")
    print(summary_df.round(6).to_string(index=False))
    print("\nBLOCKED ORDER CATEGORIES")
    if len(blocked_df):
        print(blocked_df.groupby(["status","side"]).size().to_string())
    else:
        print("No blocked orders")
    print("\nINTERNAL AUDIT STATUS:",summary["status"])
    print("Failures:",nfail,"Warnings:",nwarn)
    print("Max benchmark spread between strategies:",benchmark_spread)
    if len(anomalies_df):
        print("First anomalies:\n",anomalies_df.head(15).to_string(index=False))
    print("\nOutputs:",OUT)
    print("\nNOTE: Internal consistency != real-market executability.")
    if nfail:
        print("FAIL: inspect audit_anomalies.csv before using performance claims")


if __name__=="__main__":
    audit()
