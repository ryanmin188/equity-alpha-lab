"""Read-only V1.7 return-convention audit and market attribution from saved CSVs.

Does not import Qlib or call the V1.6 account, model or training scripts.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm


ROOT=Path(__file__).resolve().parents[2]
PROTOCOL=ROOT/'configs/v17_attribution_protocol.json'
REPORT=ROOT/'reports'
SUMMARY=REPORT/'summary'
FIGURES=REPORT/'figures'


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def regression(frame: pd.DataFrame, name: str, *, group: str='full') -> dict:
    if len(frame)<30:
        raise ValueError(f'Insufficient observations for {name}: {len(frame)}')
    y=frame['portfolio_return'].to_numpy(float)
    x=frame['benchmark_return'].to_numpy(float)
    fit=sm.OLS(y,sm.add_constant(x)).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    interval=fit.conf_int(alpha=.05)
    active=y-x
    tracking=float(active.std(ddof=1)*np.sqrt(252))
    return dict(scope=name,segment=group,start=str(frame.date.iloc[0].date()),end=str(frame.date.iloc[-1].date()),
                observations=len(frame),beta=float(fit.params[1]),beta_hac_pvalue=float(fit.pvalues[1]),
                alpha_daily=float(fit.params[0]),alpha_annual_linear=float(fit.params[0]*252),
                alpha_hac_pvalue=float(fit.pvalues[0]),alpha_ci95_low_annual=float(interval[0,0]*252),
                alpha_ci95_high_annual=float(interval[0,1]*252),tracking_error_annual=tracking,
                arithmetic_information_ratio=float(active.mean()*252/tracking) if tracking>0 else np.nan)


def main() -> None:
    protocol=json.loads(PROTOCOL.read_text(encoding='utf-8'))
    source=ROOT/protocol['input']
    summary_source=ROOT/protocol['summary_input']
    public_summary=SUMMARY/'v16_final.csv'
    if summary_source.exists() and public_summary.exists() and sha256(summary_source)!=sha256(public_summary):
        raise ValueError('Published summary differs from frozen local CSV')
    if not summary_source.exists(): summary_source=public_summary
    if not source.exists() or not summary_source.exists():
        raise FileNotFoundError('Saved V1.6 final CSVs are required; V1.7 never reruns the account')
    daily=pd.read_csv(source,parse_dates=['date'])
    prior=pd.read_csv(summary_source).set_index('candidate').loc[protocol['primary_candidate']]
    if len(daily)<252 or not daily.date.is_monotonic_increasing or daily.date.duplicated().any():
        raise ValueError('Daily observations missing or dates unsorted/duplicated')
    if daily.valuation.tolist()!=['close']*(len(daily)-1)+['open']:
        raise ValueError('Unexpected valuation boundary pattern')
    initial=float(prior.capital)
    base_p=np.r_[1.,daily.wealth.to_numpy(float)[:-1]]
    base_b=np.r_[1.,daily.benchmark_wealth.to_numpy(float)[:-1]]
    derived_p=daily.wealth.to_numpy(float)/base_p-1
    derived_b=daily.benchmark_wealth.to_numpy(float)/base_b-1
    reported_p=daily.net.to_numpy(float)
    nav=daily.nav.to_numpy(float)
    nav_error=nav-daily.cash.to_numpy(float)-daily.holdings_value.to_numpy(float)
    wealth_error=nav/initial-daily.wealth.to_numpy(float)
    p_return_error=reported_p-derived_p
    if not np.isfinite(derived_p).all() or not np.isfinite(derived_b).all():
        raise ValueError('Non-finite return')
    assert np.max(np.abs(p_return_error))<protocol['audit_tolerances']['wealth_ratio_absolute']
    assert np.max(np.abs(wealth_error))<protocol['audit_tolerances']['wealth_ratio_absolute']
    assert np.max(np.abs(nav_error))<protocol['audit_tolerances']['nav_identity_cny']
    assert abs(float(daily.fees.sum())-float(prior.fees_cny))<protocol['audit_tolerances']['cost_sum_cny']
    assert abs(float(daily.slippage_impact.sum())-float(prior.slippage_impact_cny))<protocol['audit_tolerances']['cost_sum_cny']
    assert abs(daily.wealth.iloc[-1]-1-float(prior.total_return))<1e-10
    # Qlib index marks in the saved account are float32, so the published
    # summary and daily CSV may differ by a few 1e-8 after serialization.
    benchmark_roundoff=abs(daily.benchmark_wealth.iloc[-1]-1-float(prior.benchmark_return))
    assert benchmark_roundoff<1e-6

    date=daily.date
    interval_start=[f'{date.iloc[0].date()} open']
    interval_start += [f'{d.date()} close' for d in date.iloc[:-1]]
    interval_end=[f'{d.date()} '+v for d,v in zip(date,daily.valuation)]
    convention=['first_open_to_close']+['close_to_close']*(len(daily)-2)+['terminal_close_to_open']
    audit=pd.DataFrame(dict(date=date,interval_start=interval_start,interval_end=interval_end,
                            convention=convention,portfolio_start_wealth=base_p,
                            portfolio_end_wealth=daily.wealth,
                            benchmark_start_wealth=base_b,
                            benchmark_end_wealth=daily.benchmark_wealth,
                            portfolio_return=derived_p,reported_portfolio_return=reported_p,
                            benchmark_return=derived_b,portfolio_return_error=p_return_error,
                            nav_identity_error_cny=nav_error,wealth_identity_error=wealth_error,
                            nav_cny=nav,cash_cny=daily.cash,holdings_value_cny=daily.holdings_value,
                            fees_cny=daily.fees,slippage_impact_cny=daily.slippage_impact,
                            stale_positions=daily.stale_positions,calendar_gap_days=date.diff().dt.days))
    audit.loc[0,'interval_start']=f'{date.iloc[0].date()} open'
    SUMMARY.mkdir(exist_ok=True)
    # Keep cash, position values, NAV and daily execution costs in the local
    # ignored audit. The public audit contains the timing and return identities.
    internal=ROOT/'scripts/equity_alpha_v2/outputs/v17_attribution'
    internal.mkdir(parents=True,exist_ok=True)
    audit.to_csv(internal/'return_convention_audit_full.csv',index=False)
    public_columns=['date','interval_start','interval_end','convention',
                    'portfolio_start_wealth','portfolio_end_wealth',
                    'benchmark_start_wealth','benchmark_end_wealth',
                    'portfolio_return','reported_portfolio_return','benchmark_return',
                    'portfolio_return_error','nav_identity_error_cny',
                    'wealth_identity_error','calendar_gap_days']
    audit[public_columns].to_csv(SUMMARY/'return_convention_audit.csv',index=False)
    anomaly=[]
    def note(idx:int,severity:str,category:str,detail:str)->None:
        anomaly.append(dict(date=str(date.iloc[idx].date()),severity=severity,category=category,detail=detail))
    note(0,'INFO','OPENING_BOUNDARY','Both series: initial open to first close; portfolio includes initial orders and costs')
    note(len(daily)-1,'INFO','TERMINAL_BOUNDARY','Both series: prior close to final open; portfolio is not liquidated')
    note(0,'UNVERIFIED','BENCHMARK_SOURCE_PRICE','Index open/close levels and independent provider are absent from saved daily CSV; only normalized wealth can be checked')
    for idx in np.flatnonzero(daily.stale_positions.to_numpy(int)>0):
        note(int(idx),'WARNING','STALE_PORTFOLIO_MARK',f'{int(daily.stale_positions.iloc[idx])} held positions used a stale mark')
    for idx in np.flatnonzero(audit.calendar_gap_days.fillna(0).to_numpy(float)>4):
        note(int(idx),'INFO','CALENDAR_GAP_UNVERIFIED',f'{int(audit.calendar_gap_days.iloc[idx])} calendar-day gap; no exchange calendar in this CSV')
    for key,errors,tolerance in [('RETURN_IDENTITY',p_return_error,1e-10),('NAV_IDENTITY',nav_error,1e-5)]:
        for idx in np.flatnonzero(np.abs(errors)>tolerance):
            note(int(idx),'ERROR',key,f'difference={errors[idx]:.12g}')
    pd.DataFrame(anomaly).to_csv(SUMMARY/'return_convention_anomalies.csv',index=False)

    interior=audit.iloc[1:-1].copy()
    full=regression(audit,'all_intervals_including_boundaries')
    primary=regression(interior,'interior_close_to_close')
    rows=[primary,full]
    annual=[]
    for year,part in audit.groupby(audit.date.dt.year):
        regular=part[part.convention=='close_to_close']
        est=regression(regular,f'year_{year}_interior',group=str(year))
        rows.append(est)
        annual.append(dict(year=int(year),start=str(part.date.iloc[0].date()),end=str(part.date.iloc[-1].date()),
                           intervals=len(part),portfolio_return=float(np.prod(1+part.portfolio_return)-1),
                           benchmark_return=float(np.prod(1+part.benchmark_return)-1),
                           log_relative=float(np.log1p(part.portfolio_return).sum()-np.log1p(part.benchmark_return).sum()),
                           beta=est['beta'],alpha_annual_linear=est['alpha_annual_linear'],
                           alpha_hac_pvalue=est['alpha_hac_pvalue'],tracking_error_annual=est['tracking_error_annual']))
    pd.DataFrame(rows).to_csv(SUMMARY/'v17_market_regressions.csv',index=False)
    annual_frame=pd.DataFrame(annual)
    annual_frame.to_csv(SUMMARY/'v17_calendar_segments.csv',index=False)

    rolling=[]
    window=int(protocol['rolling_window'])
    for last in range(window-1,len(interior)):
        sample=interior.iloc[last-window+1:last+1]
        x=sample.benchmark_return.to_numpy(float)
        y=sample.portfolio_return.to_numpy(float)
        design=np.column_stack([np.ones(window),x])
        alpha,beta=np.linalg.lstsq(design,y,rcond=None)[0]
        active=y-x
        rolling.append(dict(date=sample.date.iloc[-1],start=sample.date.iloc[0],observations=window,
                            beta=float(beta),alpha_annual_linear=float(alpha*252),
                            tracking_error_annual=float(active.std(ddof=1)*np.sqrt(252))))
    rolling_frame=pd.DataFrame(rolling)
    rolling_frame.to_csv(SUMMARY/'v17_rolling_market_risk.csv',index=False)

    beta=primary['beta']
    assert np.all(1+beta*derived_b>0)
    market_relative=np.log1p(beta*derived_b)-np.log1p(derived_b)
    remainder=np.log1p(derived_p)-np.log1p(beta*derived_b)
    logrelative=np.log1p(derived_p)-np.log1p(derived_b)
    assert np.max(np.abs(logrelative-market_relative-remainder))<1e-12
    attribution=pd.DataFrame(dict(date=date,convention=convention,portfolio_return=derived_p,
                                  benchmark_return=derived_b,portfolio_wealth=daily.wealth,
                                  benchmark_wealth=daily.benchmark_wealth,
                                  simple_cumulative_excess_points=daily.wealth-daily.benchmark_wealth,
                                  relative_wealth=daily.wealth/daily.benchmark_wealth,
                                  log_relative_interval=logrelative,
                                  cumulative_log_relative=np.cumsum(logrelative),
                                  fixed_beta_market_relative_log=market_relative,
                                  cumulative_fixed_beta_market_relative_log=np.cumsum(market_relative),
                                  unassigned_remainder_log=remainder,
                                  cumulative_unassigned_remainder_log=np.cumsum(remainder)))
    attribution.to_csv(SUMMARY/'market_attribution.csv',index=False)
    final_log=float(attribution.cumulative_log_relative.iloc[-1])
    opening_log=float(logrelative[0])
    terminal_log=float(logrelative[-1])
    interior_log=float(logrelative[1:-1].sum())
    wealth_log=math.log(float(daily.wealth.iloc[-1]/daily.benchmark_wealth.iloc[-1]))
    assert abs(final_log-wealth_log)<1e-12
    assert abs(float(attribution.cumulative_fixed_beta_market_relative_log.iloc[-1]
                     +attribution.cumulative_unassigned_remainder_log.iloc[-1])-final_log)<1e-12

    FIGURES.mkdir(exist_ok=True)
    fig,axes=plt.subplots(3,1,figsize=(11,9),sharex=True)
    axes[0].plot(rolling_frame.date,rolling_frame.beta,color='#1976a3')
    axes[0].axhline(1,color='#777',ls='--',lw=.8)
    axes[0].set(ylabel='126-day beta',title='V1.7 | Rolling market risk, interior close-to-close intervals')
    axes[1].plot(rolling_frame.date,100*rolling_frame.alpha_annual_linear,color='#bd6c27')
    axes[1].axhline(0,color='#777',lw=.8)
    axes[1].set(ylabel='Annualized OLS intercept (%)')
    axes[2].plot(rolling_frame.date,100*rolling_frame.tracking_error_annual,color='#68883a')
    axes[2].set(ylabel='Annualized tracking error (%)')
    for ax in axes: ax.grid(alpha=.25)
    fig.tight_layout();fig.savefig(FIGURES/'v17_rolling_market_risk.png',dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,5))
    ax.plot(date,attribution.cumulative_log_relative,label='Observed log relative wealth',color='#1976a3')
    ax.plot(date,attribution.cumulative_fixed_beta_market_relative_log,label='Fixed-beta market-relative component',color='#bd6c27')
    ax.plot(date,attribution.cumulative_unassigned_remainder_log,label='Unassigned remainder',color='#68883a')
    ax.axhline(0,color='#777',lw=.8);ax.grid(alpha=.25)
    ax.set(ylabel='Cumulative log contribution',title='Exact log identity; components are descriptive, not causal')
    ax.legend();fig.tight_layout();fig.savefig(FIGURES/'v17_log_relative_decomposition.png',dpi=160);plt.close(fig)

    def pct(value): return f'{100*value:.2f}%'
    def brief_reg(est):
        return f"Beta {est['beta']:.3f}；线性年化截距 {pct(est['alpha_annual_linear'])}；HAC p={est['alpha_hac_pvalue']:.3f}；95%区间 [{pct(est['alpha_ci95_low_annual'])}, {pct(est['alpha_ci95_high_annual'])}]；年化跟踪误差 {pct(est['tracking_error_annual'])}。"
    detail=annual_frame.copy()
    for column in ['portfolio_return','benchmark_return','alpha_annual_linear','tracking_error_annual']:
        detail[column]=detail[column].map(pct)
    detail['log_relative']=detail.log_relative.map(lambda v:f'{v:.4f}')
    detail['beta']=detail.beta.map(lambda v:f'{v:.3f}')
    detail['alpha_hac_pvalue']=detail.alpha_hac_pvalue.map(lambda v:f'{v:.3f}')
    detail.columns=['年份','起点','终点','区间数','组合收益','基准收益','对数相对收益','Beta','年化截距','截距p值','跟踪误差']
    report=f'''# V1.7 收益口径与市场归因审计

本报告仅解释已经打开过的V1.6最终区间。**Equal8 + Buffer30/60信号、组合、成交与成本均未改变；没有重新训练或重跑交易账户。** 本报告是事后解释，不是一次新的最终测试。[事先固定的分析协议](../configs/v17_attribution_protocol.json)在计算前单独提交，输入是本地保存的最终账户CSV。完整现金与持仓核对表留在被Git忽略的本地输出；公开审计表仅包含收益区间、净值与身份式误差。

## 1. 收益口径审计

账户从{date.iloc[0].date()}开盘前100万元现金起算。首行策略含当天开盘建仓、费用与收盘估值，基准以同日开盘作1的起点、收盘估值。中间{len(interior)}个区间为上个交易日收盘至本日收盘，策略在区间内可于开盘交易。末行{date.iloc[-1].date()}为上个交易日收盘至当天开盘，双方都以开盘估值；策略仍持有{int(daily.n_holdings.iloc[-1])}只股票，未强制清仓。**两条序列的边界时间戳一致，但首末行并非标准收盘到收盘收益。**

首日组合收益{pct(derived_p[0])}、基准收益{pct(derived_b[0])}，简单日差为{100*(derived_p[0]-derived_b[0]):.2f}个百分点，对数相对贡献{opening_log:.4f}。末日两者分别为{pct(derived_p[-1])}、{pct(derived_b[-1])}，对数相对贡献{terminal_log:.4f}。纯内部665个区间的对数相对贡献为{interior_log:.4f}；它与完整期的{final_log:.4f}起止时间不同，不能直接拿其回报率冒充完整期成绩。

逐行由账户净值、现金加持仓和基准归一化财富重建收益；策略日收益最大误差{np.max(np.abs(p_return_error)):.3g}，NAV身份式最大误差{np.max(np.abs(nav_error)):.3g}元。费用与滑点逐日求和与V1.6汇总一致；基准末值与汇总差{benchmark_roundoff:.3g}，属于保存浮点精度范围。CSV没有原始指数点位或独立供应商价格，故只能验证**内部归一化收益**，不能独立证明基准开收盘原始报价或每个交易日都齐全。

[逐日口径核对CSV](summary/return_convention_audit.csv) · [边界、陈旧估值和未核对事项](summary/return_convention_anomalies.csv)。异常文件中的INFO是设计边界，WARNING为陈旧估值，UNVERIFIED指缺失外部核据；数值身份式没有ERROR。{int((daily.stale_positions>0).sum())}个估值日存在至少1只持仓使用陈旧价格，合计{int(daily.stale_positions.sum())}个持仓日，收益仍受这些估值假设影响。

## 2. 累计收益、相对财富与对数路径

期末组合财富{daily.wealth.iloc[-1]:.6f}，基准财富{daily.benchmark_wealth.iloc[-1]:.6f}。它们的简单累计收益差为 **{100*(daily.wealth.iloc[-1]-daily.benchmark_wealth.iloc[-1]):.2f}个百分点**；相对财富收益为 **{pct(daily.wealth.iloc[-1]/daily.benchmark_wealth.iloc[-1]-1)}**；对数相对财富为 **{final_log:.4f}**。三个量有不同单位与含义，不能互换。逐日对数相对收益求和与最终财富比的对数逐位一致。

使用下节预先指定的内部区间Beta={beta:.3f}，将每天的对数相对收益恒等分为“固定Beta市场相对项”与“未分配剩余项”：全期分别为 **{float(market_relative.sum()):.4f}** 和 **{float(remainder.sum()):.4f}**，相加等于{final_log:.4f}。由于全期基准上涨，低市场Beta的市场相对项在此定义下{'为负，形成拖累' if market_relative.sum()<0 else '为正'}。剩余项混有个股选择、风格、行业、交易成本、估值及模型误差，**不能命名为纯选股Alpha**。Beta为事后估计、这种代数拆分也不是可交易的对冲组合或因果归因。

![对数相对净值拆分](figures/v17_log_relative_decomposition.png)

## 3. 市场回归：固定主口径与边界敏感性

主口径使用**内部{len(interior)}个收盘到收盘区间**，普通OLS加HAC(5)标准误、零无风险利率、单沪深300价格指数因子：{brief_reg(primary)}

作为事先规定的边界敏感性，完整{len(audit)}区间包含首日开盘到收盘及末日收盘到开盘：{brief_reg(full)} 这重现V1.6原回归的约9.00%线性年化截距。两种口径一并报告；没有因为显著性不同而换主结果。

单因子截距是**未解释的统计回归项**，含缺失风格、行业和公司行动误差；短样本、日收益相关性与已打开测试期的事后分析限制推断。即使某个p值小于阈值，也不能据此宣称已识别经济Alpha。[完整回归CSV](summary/v17_market_regressions.csv)

### 分年度区间

{detail.to_markdown(index=False)}

年度收益按该年观测区间复利；2024从首个开盘起、2026仅到10月8日开盘，不是完整自然年。年度回归仍只用内部标准收盘到收盘区间；各年样本短，p值为描述而非重新择模依据。[年度原始汇总](summary/v17_calendar_segments.csv)

可见全期相对优势主要形成于2024年：该段对数相对收益{annual_frame.log_relative.iloc[0]:.4f}；2025年为{annual_frame.log_relative.iloc[1]:.4f}，抵消了不少优势；2026年截至10月为{annual_frame.log_relative.iloc[2]:.4f}。2024年单独回归的p值不能视为多次分段考察后得到的独立发现。全期账户平均现金权重为{pct(float((daily.cash/daily.nav).mean()))}，因此低Beta也不能仅凭现金仓位解释；需要股票层面风险暴露数据进一步核对。

### 126个交易日滚动估计

Beta、线性年化截距和年化跟踪误差按固定126个内部区间逐日滚动估计；图仅用于识别时变风险，没有逐点做显著性筛选。[滚动CSV](summary/v17_rolling_market_risk.csv)

![滚动市场风险](figures/v17_rolling_market_risk.png)

## 4. 经济解释的边界

这一步能确认计量边界没有把策略与基准错位，并量化市场暴露与仍未解释的相对财富。**不能据此区分个股选择、风格或行业配置。** 仓库没有经独立核准的A股规模/价值/动量因子收益序列和逐日行业映射；股票特征排名不能充当因子收益。也没有外部公司行动现金流源，不能宣称已核对分红送转。

下一步是否取得外部数据，应先看可靠来源、授权与日期对齐成本。若要作真正的多因子或公司行动核对，应另定协议并完整保留所有结果；2024—2026年已用于本次分析，任何后续改动也不能再称作对此期间的全新最终测试。

输入CSV的SHA-256：`{sha256(source)}`；原汇总CSV的SHA-256：`{sha256(summary_source)}`。这些指纹便于核对本次读入文件，无法替代原始行情和券商账单。
'''
    (REPORT/'V17_RETURN_AND_MARKET_ATTRIBUTION.md').write_text(report,encoding='utf-8')
    result=dict(input_sha256=sha256(source),rows=len(daily),interior_rows=len(interior),
                errors=int(sum(x['severity']=='ERROR' for x in anomaly)),
                full_alpha=full['alpha_annual_linear'],primary_alpha=primary['alpha_annual_linear'],
                primary_beta=beta,market_log=float(market_relative.sum()),remainder_log=float(remainder.sum()),
                final_log=final_log)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
