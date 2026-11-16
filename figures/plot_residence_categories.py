"""SI S11: O-pair intervals categorized by the two bounding rotation events."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _common import out,ref
import pandas as pd
import matplotlib.pyplot as plt


def plot(path):
    d=pd.read_csv(path)
    fig,ax=plt.subplots(figsize=(7,5),layout='constrained')
    styles=[('NP-NP',dict(histtype='bar',alpha=.7)),
            ('NP-P',dict(histtype='step',lw=3)),
            ('P-P',dict(histtype='step',lw=2,ls='-.',color='black'))]
    for category,style in styles:
        values=d.loc[d.category==category,'tau_ps']
        if len(values): ax.hist(values,bins='fd',density=True,label=category,**style)
    ax.set(xlabel=r'Residence time $\tau$ [ps]',ylabel=r'$\rho(\tau)$',
           yscale='log',ylim=(5e-5,1e-1),xlim=(-25,800))
    ax.legend(frameon=False)
    fig.savefig(out(path.stem+'.png'),dpi=300);plt.close(fig)

if __name__=='__main__':
    paths=sorted(ref('residence_categories').glob('*K_categories.csv'))
    if not paths: raise SystemExit('No categorized-residence inputs')
    for path in paths: plot(path)
