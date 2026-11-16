"""
PBE vs. Allegro equation-of-state validation for CDP and CHS, with the training-data
volume distribution overlaid.

Top panel: energy per atom vs. volume per formula unit, DFT/PBE (blue x, dashed)
vs. Allegro (red open circles), for a set of isotropically scaled CsH2PO4 cells.
An orange histogram (right axis) shows how many training MD snapshots fall at each
volume -- the model is most accurate where training data is densest.
Bottom panel: Allegro - PBE energy error (meV/atom); it stays within ~3 meV/atom
across the scanned volumes.

Reads {CDP,CHS}_eos.csv (volume_per_formula_A3, E_PBE_eV_atom, E_Allegro_eV_atom,
error_meV_atom) and {CDP,CHS}_train_volumes.csv (train_volume_per_formula_A3).

Run:
    python plot_eos.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("model_validation")


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter



def plot(material):
    d = pd.read_csv(DATA / f"{material}_eos.csv")
    tv = pd.read_csv(DATA / f"{material}_train_volumes.csv")["train_volume_per_formula_A3"]

    fig = plt.figure(figsize=(6, 5))
    plt.rcParams["font.size"] = 12
    gs = fig.add_gridspec(2, 1, height_ratios=(2.2, 1), left=0.14, right=0.86,
                          bottom=0.12, top=0.9, wspace=0.05, hspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1_err = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax2 = ax1.twinx()

    ax1.plot(d["volume_per_formula_A3"], d["E_PBE_eV_atom"], c="dodgerblue",
             linestyle="--", marker="x", label="PBE")
    ax1.scatter(d["volume_per_formula_A3"], d["E_Allegro_eV_atom"], s=80,
                facecolors="none", edgecolors="r", label="Allegro")
    ax1.set_ylabel("Energy [eV/atom]")
    ax1.legend(frameon=False, loc="best")
    ax1.tick_params(axis="both", direction="in", labelbottom=False)

    ax1_err.plot(d["volume_per_formula_A3"], d["error_meV_atom"], c="dodgerblue",
                 linestyle="--", marker="^")
    ax1_err.set_ylabel(r"Error [meV/atom]")
    ax1_err.set_xlabel(r"Volume per formula [Å$^3$]")
    ax1_err.tick_params(axis="both", direction="in")
    ax1_err.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.2f}"))

    ax2.hist(tv, bins="auto", density=False, alpha=0.5, color="orange")
    ax2.set_ylabel("Number of training MD snapshots", color="orange")
    ax2.tick_params(axis="y", labelcolor="orange", direction="in")

    fig.savefig(out(f"PBE_vs_Allegro_{material}.png"), dpi=300)
    plt.close()
    print(f"wrote PBE_vs_Allegro_{material}.png")


if __name__ == "__main__":
    each_material(("CDP", "CHS"), plot)
