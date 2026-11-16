"""
Distribution of the number of non-productive O-pair switches within each
polyanion-pair residence interval tau_{P-P}, per temperature, with a geometric-law
fit. Reproduces the count_10 subplot grids for CDP and CHS.

Each panel: bars = P(N_nonproductive) from the MLMD data; red dashed = geometric
fit P(N) = p (1-p)^N with p the per-attempt productive-rotation probability
(fit to N-values whose occurrence count >= 10). A larger p means a proton is more
likely to make a productive (long-range) rotation before accumulating many
non-productive ones.

Reads (produced by make_nonproductive_csv.py):
  {mat}_nonproductive_distribution.csv   temperature_K, N_nonproductive, count, probability
  {mat}_geometric_fit.csv                temperature_K, p, R2, slope, intercept, count_threshold, n_events

Run:
    python plot_nonproductive.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("nonproductive")


import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MAT_NAME = {"CDP": r"CsH$_2$PO$_4$", "CHS": r"CsHSO$_4$"}
TEMPS = {
    "CDP": [525, 540, 550, 560, 575, 600, 625, 650],
    "CHS": [415, 425, 450, 460, 475, 500, 525, 550, 575, 600],
}
N_COLS = 4


def plot(material):
    dist = pd.read_csv(DATA / f"{material}_nonproductive_distribution.csv")
    fit = pd.read_csv(DATA / f"{material}_geometric_fit.csv").set_index("temperature_K")
    temps = TEMPS[material]

    n_rows = math.ceil(len(temps) / N_COLS)
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(N_COLS * 5, n_rows * 4))
    plt.rcParams.update({"font.size": 16})
    axes_flat = axes.flatten()

    for i, temp in enumerate(temps):
        ax = axes_flat[i]
        d = dist[dist["temperature_K"] == temp].sort_values("N_nonproductive")
        n_values = d["N_nonproductive"].to_numpy()
        prob = d["probability"].to_numpy()
        ax.bar(n_values, prob, alpha=0.6, label="Data")

        p = fit.loc[temp, "p"]
        if not np.isnan(p):
            r2 = fit.loc[temp, "R2"]
            k = np.arange(1, n_values.max() + 1)
            ax.plot(k, p * (1 - p) ** k, "r--", alpha=0.7,
                    label=rf"Fit $p$={p:.3f}," + "\n" + r"R$^2$" + f"={r2:.3f}")
            ax.set_title(f"{MAT_NAME[material]} {temp}K")
        else:
            ax.set_title(f"{temp}K (No fit)")

        ax.set_xlabel(r"$N_{\mathrm{nonproductive}}$")
        ax.set_ylabel("Probability")
        ax.set_ylim(bottom=5e-4)
        ax.set_yscale("log")
        ax.legend(loc="upper right")

    for j in range(len(temps), len(axes_flat)):
        axes_flat[j].axis("off")

    plt.tight_layout()
    fig.savefig(out(f"{material}_num_nonproductive_switches_in_tauP_P_count_10_subplots.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {material}_num_nonproductive_switches_in_tauP_P_count_10_subplots.png")


if __name__ == "__main__":
    each_material(("CDP", "CHS"), plot)
