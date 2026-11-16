"""
VCF (vector correlation function) vs. Delta t for the polyanion X-O bond:
CDP (P-O) and CHS (S-O), 550 K, 4 ns MLMD.

The VCF is the autocorrelation of the unit X-O bond direction; it decays as the
polyanion reorients. P-O (CDP) relaxes far more slowly than S-O (CHS).

Reads vcf.csv (columns: dt_ps, VCF_PO, VCF_SO). S-O is tabulated to 300 ps and
P-O to 800 ps, matching the plotted ranges (blank S-O cells beyond 300 ps).

Run:
    python plot_vcf.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("vcf")


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    d = pd.read_csv(DATA / "vcf.csv")

    plt.rcParams["font.size"] = 24
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(d["dt_ps"], d["VCF_PO"], ".", label="P-O bond")
    ax.plot(d["dt_ps"], d["VCF_SO"], ".", label="S-O bond")   # NaN beyond 300 ps is skipped
    ax.set_xlabel(r"$\Delta t$ [ps]")
    ax.set_ylabel("VCF")
    ax.legend(frameon=False)
    ax.tick_params(which="both", direction="in", length=4, width=1.2)

    fig.tight_layout()
    fig.savefig(out("XO_VCF_CDP_CHS.png"), dpi=300)
    plt.close(fig)
    print("wrote XO_VCF_CDP_CHS.png")
