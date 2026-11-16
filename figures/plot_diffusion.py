"""
Proton diffusivity D vs. temperature: MLMD (computation) vs. PFG-NMR (experiment),
for both CDP (CsH2PO4) and CHS (CsHSO4).

Reproduces sim_vs_experiment.png. Reads four CSVs (all in this folder):
  CDP_MLMD_diffusion.csv     T_K, D_cm2_s, D_err_cm2_s   (averaged + pruned to 525-650 K)
  CHS_MLMD_diffusion.csv     T_K, D_cm2_s, D_err_cm2_s   (averaged, sorted, T=440 K dropped)
  CDP_PFG_NMR_Ishikawa.csv   T_C, T_K, D_cm2_s           (experiment)
  CHS_PFG_NMR_Blinc.csv      inv1000_T_perK, T_K, D_cm2_s (experiment)

Dashed/dash-dot lines are Arrhenius fits to the MLMD points.

Run:
    python plot_diffusion.py
Produces: sim_vs_experiment.png
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import load_csv_columns as load, ref, out  # noqa: E402

DATA = ref("diffusion")


import numpy as np
import matplotlib.pyplot as plt


def inv1000(x):
    """1000/T, safe at T=0 (for the secondary axis transform)."""
    x = np.asarray(x, dtype=float)
    near0 = np.isclose(x, 0)
    out = np.full_like(x, np.inf)
    out[~near0] = 1e3 / x[~near0]
    return out


def arrhenius_fit(T, D):
    """log10(D) linear in 1000/T; return T-sorted fit curve (x=1000/T, y=D)."""
    slope, intercept = np.polyfit(1e3 / T, np.log10(D), 1)
    Tline = np.linspace(T.min(), T.max(), 100)
    return 1e3 / Tline, 10 ** (slope * 1e3 / Tline + intercept)


if __name__ == "__main__":
    cdp = load(DATA / "CDP_MLMD_diffusion.csv")
    chs = load(DATA / "CHS_MLMD_diffusion.csv")
    cdp_nmr = load(DATA / "CDP_PFG_NMR_Ishikawa.csv")
    chs_nmr = load(DATA / "CHS_PFG_NMR_Blinc.csv")

    plt.rcParams["font.size"] = 16
    fig, ax = plt.subplots(figsize=(7, 5))

    # --- CDP MLMD (+ Arrhenius fit) ---
    ax.scatter(inv1000(cdp["T_K"]), cdp["D_cm2_s"], c="dodgerblue", marker="o",
               label=r"CsH$_2$PO$_4$(MLMD)")
    xf, yf = arrhenius_fit(cdp["T_K"], cdp["D_cm2_s"])
    ax.plot(xf, yf, color="dodgerblue", lw=1, ls="dashdot")

    # --- CDP PFG-NMR (Ishikawa) ---
    ax.scatter(inv1000(cdp_nmr["T_K"]), cdp_nmr["D_cm2_s"], marker="*", c="blue",
               label=r"CsH$_2$PO$_4$(PFG-NMR)")

    # --- CHS MLMD (+ Arrhenius fit) ---
    ax.scatter(inv1000(chs["T_K"]), chs["D_cm2_s"], c="darkviolet", marker="o",
               label=r"CsHSO$_4$(MLMD)")
    xf, yf = arrhenius_fit(chs["T_K"], chs["D_cm2_s"])
    ax.plot(xf, yf, color="darkviolet", lw=1, ls="--")

    # --- CHS PFG-NMR (Blinc) ---
    ax.scatter(inv1000(chs_nmr["T_K"]), chs_nmr["D_cm2_s"], marker="<", c="purple",
               label=r"CsHSO$_4$(PFG-NMR)")

    ax.set_xlabel(r"1000/T [K$^{-1}$]")
    ax.set_ylabel(r"D [cm$^2$/s]")
    ax.set_yscale("log")
    ax.legend(frameon=False, loc="best", fontsize=12)

    secax = ax.secondary_xaxis("top", functions=(inv1000, inv1000))
    secax.set_xlabel("T [K]")
    ax.tick_params(which="both", direction="in", length=4, width=1.2)
    secax.tick_params(which="both", direction="in", length=4, width=1.2)

    fig.tight_layout()
    fig.savefig(out("sim_vs_experiment.png"), dpi=300, transparent=True)
    plt.close()
    print("wrote sim_vs_experiment.png")
