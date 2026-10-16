"""Consolidate every opened-history trial into a local-only research decision."""
import hashlib
import json

import pandas as pd

from research_v23_signals import OUT,ROOT


def main():
    parts=[]
    files=['base_annual.csv','stress_annual.csv','portfolio_followup_annual.csv',
           'multiscale/annual.csv','market_gate/annual.csv','trend_ablation/annual.csv']
    for name in files:
        d=pd.read_csv(OUT/name)
        if 'scenario' not in d:d['scenario']='base'
        if 'portfolio' not in d:d['portfolio']='Buffer30_60'
        parts.append(d)
    annual=pd.concat(parts,ignore_index=True)
    annual.to_csv(OUT/'all_candidate_annual.csv',index=False)
    target=annual[(annual.period=='opened_2024_2026')&annual.year.isin([2024,2025])]
    rows=[]
    for (model,portfolio,scenario),g in target.groupby(['model','portfolio','scenario']):
        g=g.set_index('year')
        if not g.index.is_unique or set(g.index)!={2024,2025}:raise ValueError('Incomplete or duplicate target years')
        rows.append({'model':model,'portfolio':portfolio,'scenario':scenario,
                     'return_2024':g.loc[2024,'strategy_return'],'return_2025':g.loc[2025,'strategy_return'],
                     'active_2024':g.loc[2024,'active_pp'],'active_2025':g.loc[2025,'active_pp'],
                     'min_active':g.active_pp.min(),'both_years_pass':bool((g.active_pp>0).all())})
    board=pd.DataFrame(rows).sort_values(['scenario','min_active'],ascending=[True,False])
    board.to_csv(OUT/'dual_year_scoreboard.csv',index=False)
    audit=pd.read_csv(OUT/'independent_account_checks.csv')
    base=board[board.scenario=='base']
    paths=list((ROOT/'configs').glob('v23*.json'))+list((ROOT/'scripts/equity_alpha_v2').glob('*v23*.py'))
    manifest={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    (OUT/'implementation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Base candidates',len(base),'dual-year passes',base.both_years_pass.sum(),'audited accounts',len(audit))


if __name__=='__main__':main()
