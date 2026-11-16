"""SI S8-S9: G_event heatmap and cuts at the first three neighbor shells."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from _common import each_material,out,ref
import numpy as np
import matplotlib.pyplot as plt


def plot(material,temperature):
    with np.load(ref('correlation')/f'{material}_{temperature}K_g_event.npz') as d:
        mask=(d['shells_A']>=3)&(d['shells_A']<=15)
        shells,G=d['shells_A'][mask],d['G'][mask]
        times,edges=d['dt_centers_ps'],d['dt_edges_ps']
        fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
        dev=min(float(np.nanmax(np.abs(G-1))),.5)
        dev=max(dev,1e-6)
        im=ax.pcolormesh(edges,np.arange(len(shells)+1)-.5,np.ma.masked_invalid(G),
                         cmap='RdBu_r',vmin=1-dev,vmax=1+dev)
        ax.set_yticks(np.arange(len(shells)),labels=[f'{r:.1f}' for r in shells])
        ax.set(xlabel=r'$\Delta t$ [ps]',ylabel=r'Shell distance [$\mathrm{\AA}$]')
        ax.set_facecolor('gainsboro')
        fig.colorbar(im,ax=ax,label=r'$G_{\mathrm{event}}(r,\Delta t)$')
        fig.savefig(out(f'G_event_2D_{material}.png'),dpi=300);plt.close(fig)
        fig,ax=plt.subplots(figsize=(8,6),layout='constrained')
        for i in range(min(3,len(shells))):
            ax.plot(times,G[i],ls=['-','--','-.'][i],color=plt.cm.Blues(1-.25*i),
                    label=f'Shell {i+1} ({shells[i]:.1f} Å)',lw=1.5)
        ax.axhline(1,color='gray',ls=':',lw=1)
        ax.set(xlabel=r'$\Delta t$ [ps]',ylabel=r'$G_{\mathrm{event}}$')
        ax.legend(frameon=False)
        fig.savefig(out(f'G_event_dt_cuts_{material}.png'),dpi=300);plt.close(fig)

if __name__=='__main__':
    each_material({'CDP':525,'CHS':450},plot)
