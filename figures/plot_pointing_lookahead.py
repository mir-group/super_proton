"""SI S20: fraction of fresh episodes that cross versus post-episode lookahead."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _common import out,ref
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

if __name__=='__main__':
    paths=sorted(ref('pointing').glob('*K_lookahead.csv'))
    if not paths: raise SystemExit('No pointing-lookahead inputs')
    data=pd.concat([pd.read_csv(p) for p in paths],ignore_index=True)
    norm=Normalize(data.temperature_K.min(),data.temperature_K.max())
    for material,group in data.groupby('material'):
        fig,ax=plt.subplots(figsize=(6.5,4.5),layout='constrained')
        for temp,d in group.groupby('temperature_K'):
            d=d.sort_values('lookahead_ps')
            ax.plot(d.lookahead_ps,d.fraction,'o-',ms=3,color=plt.cm.viridis(norm(temp)))
        ax.set(xlabel='Crossing lookahead after episode [ps]',ylabel='Fraction crossing',ylim=(0,1))
        fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap='viridis'),ax=ax,label='Temperature [K]')
        ax.grid(alpha=.25)
        fig.savefig(out(f'{material}_pointing_lookahead.png'),dpi=300);plt.close(fig)
