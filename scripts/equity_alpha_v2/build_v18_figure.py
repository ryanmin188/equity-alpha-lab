"""Draw the published yearly relative-wealth decomposition from aggregate CSV."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
data=pd.read_csv(ROOT/'reports/summary/v18_yearly_active_decomposition.csv')
labels=[str(y)+(' (to Oct 8)' if int(y)==2026 else '') for y in data.year]
x=np.arange(len(data))
stock=data.security_log_relative.to_numpy()
cash=data.cash_log_relative.to_numpy()
fig,ax=plt.subplots(figsize=(8.2,4.6),dpi=160)
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#fbfcfe')
ax.axhline(0,color='#485366',linewidth=1)
for j,(a,b) in enumerate(zip(stock,cash)):
    ax.bar(j,a,width=.55,color='#205a85',label='Security relative' if j==0 else None)
    ax.bar(j,b,bottom=a,width=.55,color='#e0a23b',label='Cash relative' if j==0 else None)
    total=a+b
    ax.text(j,total+(0.008 if total>=0 else -0.012),f'{total:+.3f}',
            ha='center',va='bottom' if total>=0 else 'top',fontsize=10,fontweight='bold')
ax.set_xticks(x,labels)
ax.set_ylim(-.19,.23)
ax.set_ylabel('Log relative wealth contribution')
ax.set_title('Where relative performance changed',loc='left',fontsize=14,fontweight='bold',pad=14)
ax.legend(loc='upper right',frameon=False)
ax.grid(axis='y',alpha=.16)
ax.spines[['top','right']].set_visible(False)
fig.text(.11,.015,'Frozen V1.6 simulation versus CSI 300 price index; 2026 ends at Oct 8 open.',
         fontsize=8,color='#687586')
fig.tight_layout(rect=[0,.04,1,1])
out=ROOT/'reports/figures/v18_yearly_active_decomposition.png'
fig.savefig(out,bbox_inches='tight')
print(out)
