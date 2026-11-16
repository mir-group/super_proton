"""
Single-temperature version of the non-productive-rotation figure: CDP at 525 K
and CHS at 450 K (the two panels called out for Fig. S12).

Same content as one panel of plot_nonproductive.py: bars = P(N_nonproductive) from
MLMD, red dashed = geometric fit P(N) = p (1-p)^N, one standalone figure each.

Reads (produced by make_nonproductive_csv.py):
  {mat}_nonproductive_distribution.csv   temperature_K, N_nonproductive, count, probability
  {mat}_geometric_fit.csv                temperature_K, p, R2, slope, intercept, count_threshold, n_events

Run:
    python plot_nonproductive_single.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("nonproductive")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# material -> (temperature for the S12 panel, display name)
PANELS = {
    "CDP": (525, r"CsH$_2$PO$_4$"),
    "CHS": (450, r"CsHSO$_4$"),
}


def plot(material):
    temp, mat_name = PANELS[material]
    dist = pd.read_csv(DATA / f"{material}_nonproductive_distribution.csv")
    fit = pd.read_csv(DATA / f"{material}_geometric_fit.csv").set_index("temperature_K")

    d = dist[dist["temperature_K"] == temp].sort_values("N_nonproductive")
    n_values = d["N_nonproductive"].to_numpy()
    prob = d["probability"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.rcParams.update({"font.size": 16})
    plt.bar(n_values, prob, alpha=0.6, label="Data")

    p = fit.loc[temp, "p"]
    if not np.isnan(p):
        r2 = fit.loc[temp, "R2"]
        k = np.arange(1, n_values.max() + 1)
        plt.plot(k, p * (1 - p) ** k, "r--", alpha=0.7,
                 label=rf"Fit $p$={p:.3f}," + "\n" + r"R$^2$" + f"={r2:.3f}")

    plt.title(f"{mat_name} {temp}K")
    plt.xlabel(r"$N_{\mathrm{nonproductive}}$")
    plt.ylabel("Probability")
    plt.ylim(bottom=5e-4)
    plt.yscale("log")
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out(f"{material}_{temp}K_num_nonproductive_switches.png"),
                dpi=300, bbox_inches="tight")
    plt.close()
    print(f"wrote {material}_{temp}K_num_nonproductive_switches.png")


if __name__ == "__main__":
    each_material(("CDP", "CHS"), plot)
