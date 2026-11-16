"""
Committed vs. productive proton-hop rate vs. temperature, for CDP and CHS.

Two rate estimates of the same quantity, per material (units /proton/ps):
  committed  - event-bounded attempt-success rate: how often a pointing attempt
               actually commits (crosses and stays), from the pointing analysis
  productive - the productive polyanion pair-switch rate P_rate_pdf from the
               residence-time analysis (folder 3's lambda_P)

They track each other closely (the committed rate slightly exceeds the productive
rate), confirming the two independent definitions of a successful proton transfer
agree. CHS rates are ~5x higher than CDP.

Reads committed_vs_productive.csv (columns: material, T_K, inv1000_T,
committed_rate, productive_rate, productive_rate_err).

Run:
    python plot_committed_vs_productive.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("pointing")


import pandas as pd
import matplotlib.pyplot as plt

COLORS = {"CDP": "#1f77b4", "CHS": "darkviolet"}
MAT_LABEL = {"CDP": r"$\mathrm{CsH_2PO_4}$", "CHS": r"$\mathrm{CsHSO_4}$"}

if __name__ == "__main__":
    df = pd.read_csv(DATA / "committed_vs_productive.csv")

    fig, ax = plt.subplots(figsize=(6, 4.4))
    # committed first (both materials), then productive, to match the grouped legend
    for mat in ("CDP", "CHS"):
        s = df[df["material"] == mat].sort_values("T_K")
        ax.plot(s["inv1000_T"], s["committed_rate"], "o-", color=COLORS[mat],
                label=f"{MAT_LABEL[mat]} committed")
    for mat in ("CDP", "CHS"):
        s = df[df["material"] == mat].sort_values("T_K")
        ax.errorbar(s["inv1000_T"], s["productive_rate"], yerr=s["productive_rate_err"],
                    fmt="s--", color=COLORS[mat], mfc="white", capsize=2,
                    label=f"{MAT_LABEL[mat]} productive")

    ax.set_yscale("log")
    ax.set_xlabel(r"1000/T [K$^{-1}$]")
    ax.set_ylabel("rate [/proton$\\cdot$ps]")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out("committed_vs_productive.png"), dpi=300)
    plt.close()
    print("wrote committed_vs_productive.png")
