"""Publish aggregates and figures; never changes the frozen model or selection."""
from pathlib import Path
import hashlib
import json
import pickle
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from research_v16_improvements import OUT,ROOT,FEATURES,digest,final


def pct(x): return f'{x*100:.2f}%'


def table(frame):
    return frame.to_markdown(index=False,floatfmt='.4f')


def audit_account(folder,market,capital):
    daily=pd.read_csv(folder/'daily_account.csv',parse_dates=['date'])
    orders=pd.read_csv(folder/'orders.csv',parse_dates=['date','signal_date'])
    positions=np.zeros(len(market.symbols)); cash=capital
    max_error=0.
    for row in daily.itertuples():
        for order in orders[orders.date==row.date].itertuples():
            assert order.signal_date<order.date
            if order.status.startswith('BLOCKED'):
                assert order.filled_adjusted_units==0 and order.notional==0
                continue
            sign=1 if order.side=='BUY' else -1
            cash-=sign*order.notional+order.fees
            positions[market.symbol_index[order.instrument]]+=sign*order.filled_adjusted_units
            assert cash>=-1e-6 and positions.min()>=-1e-6
            assert order.participation<=.01+1e-10
        i=market.date_index[row.date]
        observed=market.fields['open' if row.valuation=='open' else 'close'][i]
        # Independent backward search for each held stock's last usable mark.
        value=0.
        for j in np.flatnonzero(positions>1e-10):
            price=observed[j]
            if not np.isfinite(price) or price<=0:
                hist=market.fields['close'][:i if row.valuation=='open' else i+1,j]
                price=hist[np.isfinite(hist)&(hist>0)][-1]
            value+=positions[j]*price
        max_error=max(max_error,abs(cash-row.cash),abs(cash+value-row.nav))
    assert max_error<1e-5,max_error
    return dict(max_cash_nav_error_cny=max_error,days=len(daily),orders=len(orders),
                end_positions=int((positions>1e-10).sum()),minimum_cash=float(daily.cash.min()),
                maximum_realized_weight=float(daily.max_weight.max()),stale_position_days=int(daily.stale_positions.sum()))


def main():
    report=ROOT/'reports'; summary=report/'summary'; figures=report/'figures'
    frozen=json.loads((OUT/'frozen_selection.json').read_text(encoding='utf-8'))
    config=json.loads((ROOT/'configs/v16_protocol.json').read_text(encoding='utf-8'))
    selected=frozen['selected']['candidate']; model=frozen['selected']['model']
    development=pd.read_csv(OUT/'development_summary.csv')
    validation=pd.read_csv(OUT/'validation_summary.csv')
    final_results=pd.read_csv(OUT/'final_summary.csv')
    stresses=pd.read_csv(OUT/'final_execution_stress.csv')
    for stage in ['development','validation','final']:
        shutil.copy2(OUT/f'{stage}_summary.csv',summary/f'v16_{stage}.csv')
    shutil.copy2(OUT/'final_execution_stress.csv',summary/'v16_execution_stress.csv')
    shutil.copy2(OUT/'frozen_selection.json',report/'V16_FINAL_RUN.json')
    with (OUT/'market_final.pkl').open('rb') as stream: market,_=pickle.load(stream)
    checks={}
    for candidate in final_results.candidate:
        checks[candidate]=audit_account(OUT/'final'/candidate,market,config['capital'])
    for stage in ['develop','final']:
        training=pd.read_csv(OUT/f'training_audit_{stage}.csv',parse_dates=['date','max_training_exit'])
        assert (training.max_training_exit<=training.date).all()
    a=pd.read_pickle(OUT/'panel_develop.pkl').set_index(['date','instrument'])
    b=pd.read_pickle(OUT/'panel_final.pkl').set_index(['date','instrument'])
    shared=a.index.intersection(b.index)
    pd.testing.assert_frame_equal(a.loc[shared,FEATURES+['raw_vol_60']],b.loc[shared,FEATURES+['raw_vol_60']])
    for key,path in [('protocol_sha256',ROOT/'configs/v16_protocol.json'),('research_code_sha256',Path(__file__).with_name('research_v16_improvements.py')),('execution_code_sha256',Path(__file__).with_name('execution_v16.py'))]:
        assert digest(path)==frozen[key]
    try: final(config)
    except RuntimeError as error: assert 'already opened' in str(error)
    else: raise AssertionError('Final guard failed')
    audit=dict(scope='Internal mechanics and temporal consistency; not independent prices or broker validation',
               source_code_frozen=True,final_reopen_rejected=True,training_labels_mature=True,
               historical_feature_prefix_rows=len(shared),historical_feature_prefix_exact=True,
               final_accounts=checks,local_cache_hashes={name:digest(OUT/name) for name in ['market_develop.pkl','market_final.pkl','panel_develop.pkl','panel_final.pkl']})
    (report/'V16_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')

    chosen=final_results.set_index('candidate').loc[selected]
    daily=pd.read_csv(OUT/'final'/selected/'daily_account.csv',parse_dates=['date'])
    r=daily.net.to_numpy(); bnav=np.r_[1.,daily.benchmark_wealth.to_numpy()]; br=bnav[1:]/bnav[:-1]-1
    fit=sm.OLS(r,sm.add_constant(br)).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    mean=sm.OLS(r-br,np.ones((len(r),1))).fit(cov_type='HAC',cov_kwds={'maxlags':5})
    ci=fit.conf_int()[0]*252
    risk=dict(observations=len(r),beta=float(fit.params[1]),annualized_alpha_linear=float(fit.params[0]*252),
              alpha_hac_pvalue=float(fit.pvalues[0]),alpha_ci95_low=float(ci[0]),alpha_ci95_high=float(ci[1]),
              mean_daily_excess_hac_pvalue=float(mean.pvalues[0]),
              benchmark_max_drawdown=float((bnav/np.maximum.accumulate(bnav)-1).min()),
              assumptions='Zero risk-free rate; single price-index market factor; HAC lag5; post-selection inference not multiplicity adjusted')
    (report/'V16_RISK_STATISTICS.json').write_text(json.dumps(risk,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True,gridspec_kw={'height_ratios':[2,1]})
    for candidate in final_results.candidate:
        d=pd.read_csv(OUT/'final'/candidate/'daily_account.csv',parse_dates=['date'])
        axes[0].plot(d.date,d.wealth,label=candidate)
    axes[0].plot(daily.date,daily.benchmark_wealth,label='CSI300 price index',color='#555555',linestyle='--')
    axes[0].set(ylabel='Net wealth (initial = 1)',title='V1.6 frozen final evaluation | modeled execution costs')
    axes[0].legend(loc='upper left',fontsize=9); axes[0].grid(alpha=.2)
    wealth=np.r_[1.,daily.wealth.to_numpy()]; dd=(wealth/np.maximum.accumulate(wealth)-1)[1:]*100
    axes[1].fill_between(daily.date,dd,0,color='#247c9b',alpha=.35)
    axes[1].set(ylabel='Selected drawdown (%)'); axes[1].grid(alpha=.2)
    fig.tight_layout(); fig.savefig(figures/'v16_final_equity.png',dpi=160); plt.close(fig)
    joined=development[['candidate','total_return']].merge(validation[['candidate','total_return']],on='candidate',suffixes=('_development','_validation'))
    fig,ax=plt.subplots(figsize=(12,8)); y=np.arange(len(joined))
    ax.barh(y-.18,joined.total_return_development*100,.36,label='Development 2017-2020',color='#5599bb')
    ax.barh(y+.18,joined.total_return_validation*100,.36,label='Validation 2021-2023',color='#df9b54')
    ax.set_yticks(y,joined.candidate,fontsize=8); ax.invert_yaxis(); ax.axvline(0,color='#444444',lw=.7)
    ax.set(xlabel='Cumulative simulated return (%)',title='All 16 candidates | selected by validation IR, not development return')
    ax.legend(); ax.grid(axis='x',alpha=.2); fig.tight_layout(); fig.savefig(figures/'v16_all_candidates.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5))
    for capital,rows in stresses.groupby('capital'):
        ax.plot(rows.slippage_bps,rows.total_return*100,marker='o',label=f'Capital CNY {capital/1e6:g}m')
    ax.axhline(chosen.benchmark_return*100,color='#555',ls='--',label='CSI300 price index')
    ax.set(xlabel='One-way slippage (bp), plus modeled impact and fees',ylabel='Cumulative return (%)',title='Full account replay at each execution assumption')
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(figures/'v16_execution_stress.png',dpi=160); plt.close(fig)
    def view(frame):
        result=frame[['candidate','total_return','benchmark_return','max_drawdown','information_ratio']].copy()
        for col in ['total_return','benchmark_return','max_drawdown']: result[col]=result[col].map(pct)
        result.columns=['候选','累计收益','同期基准','每日最大回撤','信息比率']
        return table(result)
    report_text=f'''# V1.6 分期研究与执行改进

## 结论

冻结组合 **{selected}** 在最终时间测试 {chosen.first_entry} 至 {chosen.final_mark} 累计模拟收益 **{pct(chosen.total_return)}**，同期沪深300价格指数 **{pct(chosen.benchmark_return)}**，领先 **{chosen.cumulative_excess_points*100:.2f}个百分点**。年化收益{pct(chosen.annualized_return)}，每日最大回撤{pct(chosen.max_drawdown)}；日频信息比率仅{chosen.information_ratio:.3f}，不是稳定alpha已经成立的证明。

冻结发生在最终测试之前，见[GitHub冻结提交](https://github.com/ryanmin188/equity-alpha-lab/commit/3a8663e50bb69dca774f9feff837bf70f46aa551)、[原冻结记录](V16_FROZEN_SELECTION.json)和[最终运行状态](V16_FINAL_RUN.json)。时期未曾用于此前开发来自项目使用者声明，不是第三方审计保证。最终数据打开后未改策略、模型、成本或选择规则。

## 关于39.66%的目标

39.66%对应2017-02-03至2020-08-03的旧基准。新版本该区间的MomentumQuality/Buffer30_60为{pct(development.set_index('candidate').loc['MomentumQuality__Buffer30_60','total_return'])}，LGBRankTarget/Buffer30_60为{pct(development.set_index('candidate').loc['LGBRankTarget__Buffer30_60','total_return'])}，都超过旧基准。但开发期已被使用，且新旧执行口径不同；这些结果仅是开发证据。

正式冻结的Equal8/Buffer30_60在旧区间只有{pct(development.set_index('candidate').loc[selected,'total_return'])}，**没有跑赢旧基准**。它因验证期信息比率最高而入选。不能把开发期最好模型的收益、验证期另一模型的稳健性和最终测试成绩拼成同一策略，也不能拿2024年后的37.71%直接与旧区间39.66%比较。

## 数据、特征与选择

保留CSI300历史成分股、月末信号、次月开盘执行和滚动训练框架。8因子基线扩展为28因子，比较收益排名Ridge、正则树模型、按月分组LambdaRank、排名集成、训练样本内相关性筛选及固定动量信号。最多60个月训练，至少36个月，训练标签必须已经到期；完整252日特征预热、至少80%覆盖。训练中位数填补；固定信号缺失排名用中性值0。

开发期为2017-02-03至2020-08-03；验证期实际为2021-01-04至2023-12-01；最终为2024-01-02至2026-10-08。只计算完整月度持有区间，故不覆盖数据截至2026-10-12的最后几天。2020-08之后至年底的结果未用于模型选择，但其成熟标签可进入以后训练。验证结束至最终开始的标签同理。最终阶段每月更新训练集属于冻结的滚动规则。

模型选择只用验证期净收益信息比率，并优先满足回撤不超过40%的方案；规则与全部参数见[协议](../configs/v16_protocol.json)。8×2共16个候选均保留；没有在最终测试后增补候选。复杂模型在此次验证中未胜过简单基线。

![全部候选](figures/v16_all_candidates.png)

### 开发期：不得当作独立成绩

{view(development)}

### 验证期：用于选择，不能再当独立测试

{view(validation)}

### 最终期：冻结后只评估选定方案和预先指定基线

{view(final_results)}

选定方案本身就是Equal8，因此与Equal8基线去重，仅另评估Ridge8Rank，没有选择最终表现更好的模型。

![最终净值与回撤](figures/v16_final_equity.png)

## 执行模型

本金100万元、98%目标配置、目标单股5%上限、初始建仓后单次目标买卖总额不超过信号日NAV的80%；等权Top30保留至排名60。数量按信号日收盘计算，次日开盘成交，卖出先于买入，无借款。实际持仓可因部分成交、拒单、换手约束超过30只；期末为{int(chosen.terminal_positions)}只。

佣金3bp、每笔最低5元；卖出印花税2023-08-28前10bp、此后5bp；过户费单边0.2bp假设；滑点5bp，额外冲击为0.1×滞后20日波动率×平方根(交易额/滞后20日平均成交额)。单次交易额最多为滞后平均成交额的1%，并按股数代理取整。卖出完整仓位允许退出剩余零股；科创板200股起、1股递增。价格限制使用板块代理，未提供ST/IPO正式限价资料。

最终主情景费用{chosen.fees_cny:,.2f}元，滑点与冲击损耗{chosen.slippage_impact_cny:,.2f}元；拒单{int(chosen.blocked_orders)}笔，其中可能包括手数、容量或资金不足的小额调整。这些均是模拟器结果。日成交量只判断是否存在交易，滞后ADV不能证明开盘集合竞价容量。T+1虽有约束，但月频策略通常不会触发同日反向交易。

**期末仍持仓，以开盘估值，未扣未来清仓费用；公司行动采用复权单位连续收益，未独立重建真实股数与现金分红。** 基准为价格指数，未计投资成本，未确认总回报口径；因此尚不能称为可交易指数增强的真实超额业绩。

## 压力测试：改变条件后完整重放

{table(stresses[['capital','slippage_bps','total_return','benchmark_return','max_drawdown','information_ratio']].assign(total_return=lambda d:d.total_return.map(pct),benchmark_return=lambda d:d.benchmark_return.map(pct),max_drawdown=lambda d:d.max_drawdown.map(pct)))}

![成本与容量压力测试](figures/v16_execution_stress.png)

放大本金会改变手数、最低费用、容量和成交路径，不是原收益简单乘10。0bp滑点也仍计冲击和税费。以上是预先规定的情景，未从中选一个更漂亮的作为主结果。

## 统计与会计证据

使用667个估值观测、零无风险利率、单市场因子回归、HAC滞后5：市场beta={risk['beta']:.3f}，线性年化alpha={pct(risk['annualized_alpha_linear'])}，95%区间[{pct(risk['alpha_ci95_low'])}, {pct(risk['alpha_ci95_high'])}]，p={risk['alpha_hac_pvalue']:.4f}。平均日超额收益HAC检验p={risk['mean_daily_excess_hac_pvalue']:.4f}。这些检验未对16候选选择、多情景和残余风格暴露充分校正，短样本不能支持长期稳定获利。[完整统计](V16_RISK_STATISTICS.json)

独立读取成交事件，逐日重建现金和持仓估值，两组最终账户误差均小于0.00001元。开发数据和延长数据中{len(shared):,}条共同历史股票月记录的全部特征逐值相同；所有训练标签在信号日已成熟；源代码哈希与冻结版本相同；重复打开最终测试被拒绝。[核对记录](V16_AUDIT.json)

模拟样本单元测试覆盖拒单、账本、先决策后开盘、滞后容量、零股退出、税费、成熟标签和向前账本。内部通过不代表外部行情正确或实际成交可行。曾修复的实现问题和修正前结果已保留在[协议说明](V16_PROTOCOL_NOTES.md)及summary目录。

## 实盘与面试定位

这是具有时序隔离和更细执行假设的离线研究作品，适合展示研究判断、负结果、模型比较与工程核对。未接券商、没有未来实际日期的交易业绩；[向前模拟账本与计划](FORWARD_PAPER_PLAN.md)只是下一阶段基础。模型选择、数据修订、公司行动、风格暴露与未建模执行风险仍需后续验证。
'''
    (report/'V16_RESEARCH_REPORT.md').write_text(report_text,encoding='utf-8')
    print(json.dumps(dict(selected=selected,final_return=chosen.total_return,risk=risk,audit=checks),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
