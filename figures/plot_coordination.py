"""Main Figs. 4d and 7b: mean bound-proton count versus residence, plus marginal."""
import sys
from pathlib import Path
sys.path[:0] = [str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1])]
from _common import each_material, out, ref
from superproton.coordination import joint_density_Np_tau
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def plot(material, temperature):
    d = pd.read_csv(ref('coordination')/f'{material}_{temperature}K_coordination.csv')
    tau, number = d.tau_ps.to_numpy(), d.mean_Np.to_numpy()
    density, te, ne = joint_density_Np_tau(tau,number,tau_bins=140,np_bins=70,
                                         np_range=(number.min(),number.max()))
    fig,(ax,marginal)=plt.subplots(1,2,figsize=(7,4.5),sharey=True,
                                  gridspec_kw={'width_ratios':[5,1]},layout='constrained')
    im=ax.pcolormesh(te,ne,np.ma.masked_less_equal(density.T,0),norm=LogNorm(),cmap='viridis')
    fig.colorbar(im,ax=ax,label='Joint probability density')
    marginal.hist(number,bins='auto',density=True,orientation='horizontal',alpha=.7)
    marginal.tick_params(labelleft=False,labelright=True,right=True,left=False)
    marginal.set_xlabel(r'$\rho(\bar N_p)$')
    ax.set(xlim=(0,600),xlabel=r'Residence time $\tau$ [ps]',ylabel=r'$\bar N_p(\tau)$')
    tc=72 if material=='CDP' else 68
    for mask,x0,x1,color,style,label in [
        (tau<tc,0,tc,'black','--',r'$\tau<\tau_c$'),
        (tau>tc,tc,600,'tab:orange','-.',r'$\tau>\tau_c$')]:
        if mask.any(): ax.hlines(number[mask].mean(),x0,x1,color=color,ls=style,label=label)
    ax.legend(frameon=False,loc='lower right')
    fig.savefig(out(f'{material}_coordination.png'),dpi=300)
    plt.close(fig)

if __name__=='__main__':
    each_material({'CDP':525,'CHS':450},plot)
