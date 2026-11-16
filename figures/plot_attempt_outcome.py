"""
Attempt-outcome fractions vs. temperature, one panel per material (CDP, CHS).

Once a proton "points" toward a new acceptor (a transfer attempt), that attempt
ends in one of three ways, which sum to 1:
  P(commit)     - the proton crosses and stays with the new acceptor
  P(revert)     - it falls back to the original pair
  P(superseded) - a different new pointing target takes over before it commits

CDP commits ~84% of attempts across T; CHS reverts most attempts (commit ~20-25%),
i.e. CHS makes far more nonproductive attempts.

Reads attempt_outcome.csv (columns: material, T_K, n_decided, n_commit, n_revert,
n_superseded, P_commit, P_revert, P_superseded).

Run:
    python plot_attempt_outcome.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("pointing")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAT_LABEL = {"CDP": r"$\mathrm{CsH_2PO_4}$", "CHS": r"$\mathrm{CsHSO_4}$"}
COMMIT, REVERT, SUPER = "#2a9d8f", "tab:orange", "0.78"

if __name__ == "__main__":
    df = pd.read_csv(DATA / "attempt_outcome.csv")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for ax, mat in zip(axes, ("CDP", "CHS")):
        d = df[df["material"] == mat].sort_values("T_K")
        Ts = d["T_K"].to_numpy(float)
        p_com = d["P_commit"].to_numpy(float)
        p_rev = d["P_revert"].to_numpy(float)
        p_sup = d["P_superseded"].to_numpy(float)
        w = 0.6 * np.diff(Ts).min()
        ax.bar(Ts, p_com, width=w, color=COMMIT, label=r"$P(\mathrm{commit})$")
        ax.bar(Ts, p_rev, width=w, bottom=p_com, color=REVERT, label=r"$P(\mathrm{revert})$")
        ax.bar(Ts, p_sup, width=w, bottom=p_com + p_rev, color=SUPER,
               label=r"$P(\mathrm{superseded})$")
        ax.set_xlabel("T [K]")
        ax.set_title(MAT_LABEL[mat])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("attempt outcome fraction")

    # flat 1-row legend centred in the gap between the two panels, along the top
    plt.tight_layout()
    handles, labels = axes[0].get_legend_handles_labels()
    p0, p1 = axes[0].get_position(), axes[1].get_position()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               fontsize=8, bbox_to_anchor=((p0.x1 + p1.x0) / 2, p0.y1),
               handlelength=1.3, columnspacing=1.0)
    plt.savefig(out("attempt_outcome_stacked.png"), dpi=300, bbox_inches="tight")
    plt.close()
    print("wrote attempt_outcome_stacked.png")
