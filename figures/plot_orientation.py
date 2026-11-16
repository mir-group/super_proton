"""Main Fig. 8b,c,e,f: one-bond and all-polyanion angular densities at 550 K.

Inputs include binned densities and initial/experimental markers. The fixed
initial-sublattice SO4 mapping is in superproton.geometry, not the plotter.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot(material):
    for scope in ('single','all'):
        with np.load(ref('orientation')/f'{material}_550K_{scope}.npz') as d:
            fig, ax = plt.subplots(figsize=(6,5), layout='constrained')
            im = ax.pcolormesh(d['phi_edges_deg'],d['theta_edges_deg'],
                              np.ma.masked_less_equal(d['density'].T,0),
                              norm=LogNorm(),cmap='viridis',shading='flat')
            ax.scatter(d['marker_phi_deg'],d['marker_theta_deg'],c='red',
                       marker='o' if scope=='single' else 'X',s=60)
            ax.set(xlim=(-180,180),ylim=(0,180),xlabel=r'$\phi$ [degrees]',
                   ylabel=r'$\theta$ [degrees]')
            fig.colorbar(im,ax=ax,label=r'Angular density [degree$^{-2}$]')
            fig.savefig(out(f'{material}_{scope}_XO.png'),dpi=300)
            plt.close(fig)

if __name__=='__main__':
    each_material(('CDP','CHS'),plot)
