"""
Probability density vs. long-range proton hop distance, for CDP and CHS,
one figure per material, one histogram per temperature (4 ns MLMD).

A "long-range hop" is a proton that leaves its original oxygen and commits to a
new acceptor; the hop distance is how far the proton's position moves. The
distributions are nearly temperature-independent for each material.

Reads {CDP,CHS}_hop_distance.csv (columns: temperature_K, hop_distance_A) -- one
row per hop event. Each CSV holds every temperature in the dataset; the figure
shows the subset used in the paper (set in PLOT_TEMPS).

Run:
    python plot_hop_distance.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import each_material, out, ref  # noqa: E402

DATA = ref("hop_distance")


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# temperatures shown per material (the CSVs may contain more)
PLOT_TEMPS = {"CDP": [525, 550, 575], "CHS": [450, 475, 525]}


def plot(material):
    df = pd.read_csv(DATA / f"{material}_hop_distance.csv")
    plt.figure(figsize=(7, 5))
    plt.rcParams.update({"font.size": 16})
    for temp in PLOT_TEMPS[material]:
        d = df.loc[df["temperature_K"] == temp, "hop_distance_A"].to_numpy()
        if d.size == 0:
            continue
        plt.hist(d, bins="auto", density=True, histtype="step", linewidth=1.5,
                 alpha=0.9, label=f"{temp}K (n={d.size})")
        print(f"{material} {temp}K: median={np.median(d):.2f} Å, mean={d.mean():.2f} Å")
    plt.xlabel("Long-range proton hop distance (Å)")
    plt.ylabel("Probability density")
    plt.legend(loc="upper right", fontsize=14)
    plt.tight_layout()
    plt.savefig(out(f"{material}_jump_distances_all_temps.png"), dpi=300,
                bbox_inches="tight", transparent=True)
    plt.close()
    print(f"  -> {material}_jump_distances_all_temps.png")


if __name__ == "__main__":
    each_material(("CDP", "CHS"), plot)
