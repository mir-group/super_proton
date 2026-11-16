"""
rho(r): probability density of the polyanion (O) rotation distance during a
proton-transfer / reorientation event, for CDP (525 K) and CHS (450 K), 4 ns MLMD.

Two panels:
  top  = nonproductive O-pair switch  -> host (filled), exchanger (step)
  bottom = productive P-pair switch   -> courier (filled), recipient (step),
                                         provider (dashed step)
where, per event, the shared O carrying the proton is the "host" (nonproductive)
or "courier" (productive); the recipient/provider/exchanger are the other O's
involved in the O-pair swap.

Reads raw per-event O displacements in Angstrom from data/reference/rotation_distance/
and writes PNG figures to figures/output/. See docs/paper_mapping.md for analysis
functions.

Run from the repository root::

    python figures/plot_rotation_distance.py

To use your own inputs under results/reference/rotation_distance/::

    SUPERPROTON_REFERENCE_DIR=results/reference python figures/plot_rotation_distance.py

Inputs: CDP_525K_4ns_O_displacement_rot_periodps.npz and the CHS_450K counterpart.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("rotation_distance")


import numpy as np
import matplotlib.pyplot as plt

BLUE, ORANGE, GREEN = "#1f77b4", "#ff7f0e", "#2ca02c"

# (panel, role, npz key) in draw order
ROLES = [
    ("nonproductive", "host",      "NP_dr_O_all_carrier"),
    ("nonproductive", "exchanger", "NP_dr_O_all_acceptor"),
    ("productive",    "courier",   "P_dr_O_all_carrier"),
    ("productive",    "recipient", "P_dr_O_all_acceptor"),
    ("productive",    "provider",  "P_dr_O_all_donor"),
]

MATERIALS = {
    "CDP_525K": DATA / "CDP_525K_4ns_O_displacement_rot_periodps.npz",
    "CHS_450K": DATA / "CHS_450K_4ns_O_displacement_rot_periodps.npz",
}

# figure-name prefix (PNG only): CDP -> a, CHS -> b
PREFIX = {"CDP_525K": "a", "CHS_450K": "b"}


def load(npz):
    d = np.load(npz, allow_pickle=True)
    # -1 marks events with no carrier O; drop them (matches original analysis)
    return {k: (lambda a: a[a != -1])(np.asarray(d[k], float)) for k in d.files}


def plot(material, d, out_png):
    plt.rcParams.update({"font.size": 15})
    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    # top: nonproductive O-pair switch
    axes[0].hist(d["NP_dr_O_all_carrier"], bins="auto", density=True,
                 color=BLUE, alpha=0.5, label="host")
    axes[0].hist(d["NP_dr_O_all_acceptor"], bins="auto", density=True,
                 color=BLUE, histtype="step", linewidth=2, zorder=2, label="exchanger")

    # bottom: productive P-pair switch
    axes[1].hist(d["P_dr_O_all_carrier"], bins="auto", density=True,
                 color=ORANGE, alpha=0.5, label="courier")
    axes[1].hist(d["P_dr_O_all_acceptor"], bins="auto", density=True,
                 color=ORANGE, histtype="step", linewidth=2.5, label="recipient")
    axes[1].hist(d["P_dr_O_all_donor"], bins="auto", density=True,
                 color=GREEN, histtype="step", linestyle="dashed", linewidth=2, label="provider")

    for ax in axes:
        ax.legend(frameon=False, fontsize=14)
        ax.tick_params(which="both", direction="in", length=4, width=1.2)
        ax.set_ylabel(r"$\rho(r)$")
    axes[1].set_xlabel(r"Polyanion rotation distance [Å]")

    fig.tight_layout()
    fig.savefig(out(out_png), dpi=300, transparent=True)
    plt.close(fig)
    print(f"wrote {out_png}")




if __name__ == "__main__":
    for material, npz in MATERIALS.items():
        d = load(npz)
        plot(material, d, f"{PREFIX[material]}_{material}_4ns_rot_dist_rot_periodps.png")
        print(f"  {material}: productive courier median = "
              f"{np.median(d['P_dr_O_all_carrier']):.2f} Å")
