"""
P-O bond vector autocorrelation of CsH2PO4 at 525 K: original vs. augmented
Allegro model, in 4x4x4 and 5x5x5 supercells (2 ns ML MD each).

The "augmented" model adds 512-atom (4x4x4) supercell configurations to the
training set; "original" is the model trained on the base set. Solid = augmented,
dashed = original; color encodes the supercell size. The curves overlap, showing
the training-set augmentation does not change the PO4 rotational dynamics.

Reads PO_auto_corr_525K_model_compare.csv with columns:
    supercell_scale, model, tau_ps, PO_autocorr, n_PO4

Run:
    python plot_PO_auto_corr.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("finite_size")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d = pd.read_csv(DATA / "PO_auto_corr_525K_model_compare.csv")

scales = sorted(d["supercell_scale"].unique())
colors = plt.cm.tab10(np.linspace(0, 1, len(scales)))
color_map = {s: colors[i] for i, s in enumerate(scales)}
linestyle_map = {"augmented": "-", "original": "--"}

plt.figure(figsize=(8, 6))
plt.rcParams["font.size"] = 15

for (scale, model), grp in d.groupby(["supercell_scale", "model"]):
    grp = grp.sort_values("tau_ps")
    plt.plot(grp["tau_ps"], grp["PO_autocorr"], lw=2,
             color=color_map[scale], linestyle=linestyle_map[model],
             label=f"{scale}x{scale}x{scale}, {model}")

plt.xlim([0, d["tau_ps"].max()])
plt.ylim([-0.2, 1.05])
plt.legend(frameon=False)
plt.xlabel(r"$\Delta t$ (ps)")
plt.ylabel("P-O vector autocorrelation")
plt.title("CsH2PO4, 525 K, 2 ns ML MD")

plt.savefig(out("PO_auto_corr_525K_model_compare_a100.png"), dpi=300, bbox_inches="tight")
plt.close()
print("wrote PO_auto_corr_525K_model_compare_a100.png")
