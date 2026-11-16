"""
Effect of thermal expansion on proton diffusion in CsH2PO4.

Arrhenius plot of the H diffusion coefficient from 4 ns MLMD in cells with the
thermally expanded lattice at each temperature (orange circles) vs. the
fixed-cell production runs of the main paper (blue squares), 525-560 K.
Dashdot lines are per-dataset linear fits of log10(D) vs 1000/T. The two
datasets agree within a few percent, so neglecting thermal expansion does not
change the transport results.

Reads CDP_thermal_expansion_diffusion.csv with columns:
    dataset (thermal_expansion | fixed_cell), T_K, D_cm2_s, D_err_cm2_s
The CSV holds all simulated temperatures (525-650 K); the figure shows the
T < 570 K window used in the paper.

Run:
    python figures/plot_thermal_expansion.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from _common import ref, out  # noqa: E402
from superproton.residence import arrhenius_fit  # noqa: E402

DATA = ref("thermal_expansion")


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

UPPER_TEMP = 570  # K, plot window mask


def oneThousand_over(x):
    x = np.array(x).astype(float)
    near_zero = np.isclose(x, 0)
    x[near_zero] = np.inf
    x[~near_zero] = 1e3 / x[~near_zero]
    return x


d = pd.read_csv(DATA / "CDP_thermal_expansion_diffusion.csv")
d = d[d["T_K"] < UPPER_TEMP]
te = d[d["dataset"] == "thermal_expansion"].sort_values("T_K")
fc = d[d["dataset"] == "fixed_cell"].sort_values("T_K")

plt.rcParams["font.size"] = 18
fig, ax = plt.subplots(figsize=(7, 5))

for grp, color, marker, size, label, zorder in (
    (te, "orange", "o", None, r"CsH$_2$PO$_4$, thermal expansion", 10),
    (fc, "dodgerblue", "s", 60, r"CsH$_2$PO$_4$", None),
):
    T, D = grp["T_K"].to_numpy(), grp["D_cm2_s"].to_numpy()
    ax.scatter(1e3 / T, D, c=color, marker=marker, s=size, label=label, zorder=zorder)
    fit = stats.linregress(1e3 / T, np.log10(D))
    ax.plot(1e3 / T, 10 ** (fit.slope * 1e3 / T + fit.intercept), color,
            linewidth=1, linestyle="dashdot", zorder=zorder)
    # superproton.residence.arrhenius_fit is the same ln(rate) vs 1/T slope the
    # rate figures use, and is regression-tested against the paper's E_a values.
    ea = arrhenius_fit(T, D)
    print(f"{grp['dataset'].iloc[0]}: Ea = {ea.E_a_eV:.3f} +/- {ea.stderr_E_a_eV:.3f} eV")

rel_diff = np.abs(te["D_cm2_s"].to_numpy() - fc["D_cm2_s"].to_numpy()) / te["D_cm2_s"].to_numpy()
print(f"average |D_thermal - D_fixed| / D_thermal: {rel_diff.mean() * 100:.2f}%")

ax.set_xlabel(r"1000/T (K$^{-1}$)")
ax.set_ylabel(r"D (cm$^2$/s)")
ax.set_yscale("log")
ax.legend(loc="lower left", fontsize=14, frameon=False)
secax = ax.secondary_xaxis("top", functions=(oneThousand_over, oneThousand_over))
secax.set_xlabel("T (K)")
ax.tick_params(which="both", direction="in", length=4, width=1.2)
secax.tick_params(which="both", direction="in", length=4, width=1.2)
plt.tight_layout()
plt.savefig(out("thermal_expansion_CDP.png"), dpi=300, transparent=True)
print("wrote thermal_expansion_CDP.png")
