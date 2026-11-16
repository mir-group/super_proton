"""
Probability density vs. cos(angle) between the proton and oxygen displacement
vectors, cos(dr_H, dr_O), overlaid for CDP and CHS (1.5 ns MLMD).

For each proton, in the ~50 frames preceding a polyanion pair switch (while the H
stays bonded to the same O), the per-frame cosine between the single-frame O
displacement and H displacement is recorded:
    cos = (dr_O . dr_H) / (|dr_O| |dr_H|)
cos ~ +1 means H and O move together (rigid O-H, the O carries the H); cos ~ 0/neg
means the H moves independently of its oxygen (a transfer-like motion). Both
distributions peak toward +1 but have a substantial tail below 0.

Reads cos_HO_perH_{CDP,CHS}_1.5ns.npy from data/reference/cos_HO/ and writes
figures/output/cos_HO_perH.png. See docs/paper_mapping.md for analysis functions.

Run from the repository root::

    python figures/plot_cos_angle.py

To use your own inputs under results/reference/cos_HO/::

    SUPERPROTON_REFERENCE_DIR=results/reference python figures/plot_cos_angle.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("cos_HO")


import numpy as np
import matplotlib.pyplot as plt

# material -> (npy, color)  (CHS drawn first, matching the source)
SERIES = [("CHS", "darkviolet"), ("CDP", "dodgerblue")]

if __name__ == "__main__":
    plt.rcParams.update({"font.size": 16})
    plt.figure(figsize=(7, 5))

    rows = []
    for material, color in SERIES:
        cos = np.concatenate(np.load(DATA / f"cos_HO_perH_{material}_1.5ns.npy", allow_pickle=True))
        plt.hist(cos, bins="auto", density=True, alpha=0.7, label=material,
                 color=color, histtype="step", linewidth=2.5)
        rho, edges = np.histogram(cos, bins="auto", density=True)
        for i, r in enumerate(rho):
            rows.append({"material": material, "bin_left": edges[i], "bin_right": edges[i + 1],
                         "bin_center": 0.5 * (edges[i] + edges[i + 1]), "prob_density": r})
        print(f"{material}: n={cos.size}, mean cos={cos.mean():.3f}")

    plt.xlabel(r"cos($\Delta r_H$, $\Delta r_O$)")
    plt.ylabel("probability density")
    plt.legend(frameon=False, loc="upper left")
    plt.tick_params(which="both", direction="in", length=4, width=1.2)
    plt.tight_layout()
    plt.savefig(out("cos_HO_perH.png"), dpi=300, transparent=True)
    plt.close()

    print("wrote cos_HO_perH.png")
