"""
Time-averaged radial distribution function g(r) of the polyanion-polyanion pair
(P-P in CDP, S-S in CHS), overlaid for both materials.

  CDP: P-P at 525 K,  CHS: S-S at 500 K   (4 ns MLMD, partial RDF, time-averaged)

The first peak (~4.5-4.8 Å) is the nearest polyanion-polyanion distance; the
sublattice stays well ordered while protons/polyanions reorient.

Reads rdf_type3-3.csv (columns: r_A, g_CDP_PP_525K, g_CHS_SS_500K).

Run:
    python plot_rdf.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("rdf")


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    d = pd.read_csv(DATA / "rdf_type3-3.csv")

    plt.rcParams["font.size"] = 14
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(d["r_A"], d["g_CDP_PP_525K"], label=r"CDP 525 K (P$-$P)", color="#1f77b4", linewidth=2)
    ax.plot(d["r_A"], d["g_CHS_SS_500K"], label=r"CHS 500 K (S$-$S)", color="#ff7f0e", linewidth=2)
    ax.set_xlabel(r"$r$ [Å]")
    ax.set_ylabel(r"$g(r)$")
    ax.set_title("Time-averaged RDF")
    ax.set_xlim(0, 10)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=1)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out("rdf_type3-3_CDP_vs_CHS_4ns.png"), dpi=200)
    plt.close()
    print("wrote rdf_type3-3_CDP_vs_CHS_4ns.png")
