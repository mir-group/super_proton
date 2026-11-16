"""
O-sharing occurrence vs. time before the end of a rotation, for CDP (525 K, 4 ns MLMD).

While a proton reorients, its host oxygen can transiently "share" the proton with
a neighboring oxygen (an O...H...O encounter). This plots, as a function of time
before the rotation ends (a pair switch at t=0), the normalized number of such
O-sharing events, separately for nonproductive and productive rotations.

Both rise sharply approaching t=0: O-sharing becomes much more frequent just
before the rotation completes. Nonproductive rotations show more O-sharing overall.

Reads CDP_525K_O_sharing.csv (columns: t_before_end_ps, NP_normalized_O_sharing,
P_normalized_O_sharing). y = (# O-sharing encounters at that time) / (total # rotations).

Run:
    python plot_O_sharing.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("o_sharing")


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    d = pd.read_csv(DATA / "CDP_525K_O_sharing.csv")

    plt.figure(figsize=(7, 5))
    plt.rcParams["font.size"] = 16
    plt.plot(d["t_before_end_ps"], d["NP_normalized_O_sharing"], marker=".",
             label="nonproductive rotation")
    plt.plot(d["t_before_end_ps"], d["P_normalized_O_sharing"], marker=".",
             label="productive rotation")
    plt.xlabel("Time before the end of a rotation [ps]")
    plt.ylabel(r"Normalized number of O-sharing")
    plt.legend(frameon=False)
    plt.tick_params(which="both", direction="in", length=4, width=1.2)
    plt.tight_layout()
    plt.savefig(out("CDP_4ns_O_sharing_50ps.png"), dpi=300, transparent=True)
    plt.close()
    print("wrote CDP_4ns_O_sharing_50ps.png")
