"""
dH/dO displacement ratio vs. time before a rotation ends, for a single proton
(CDP, illustrative example).

This is the time-resolved version of the mean ratio in folder 8: for one proton's
productive pair-switch event, it traces the per-frame dr_H/dr_O ratio over the
2.5 ps leading up to the switch (t_f = 0). Most frames sit near the mean (~1.9,
the O carries the H), but a few sharp spikes -- the "outlier" frames (ratio >
mean + 2 sigma, whose average is ~6.8) -- are the moments the proton briefly
outruns its oxygen, i.e. the actual transfer motion.

Reads CDP_525K_single_proton_dHdO.csv (columns: t_before_tf_ps, dHdO_ratio).

Run:
    python plot_dHdO_vs_time.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ref, out  # noqa: E402

DATA = ref("dhdo_vs_time")


import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    d = pd.read_csv(DATA / "CDP_525K_single_proton_dHdO.csv")
    t = d["t_before_tf_ps"].to_numpy()
    r = d["dHdO_ratio"].to_numpy()

    plt.figure(figsize=(7, 5))
    plt.rcParams["font.size"] = 15
    plt.plot(t, r, "-^")

    mean_val, std_val = r.mean(), r.std()
    outlier = r[r > mean_val + 2 * std_val].mean()   # mean of the >2 sigma frames
    plt.hlines(mean_val, t[0], t[-1], linestyles="--", label="mean")
    plt.hlines(outlier, t[0], t[-1], linestyles="-.", label="outlier", colors="#ff7f0e")

    plt.legend(loc="upper right", ncol=2, frameon=False)
    plt.tick_params(which="both", direction="in", length=4, width=1.2)
    plt.ylim(-0.1, 10.5)
    plt.ylabel(r"dH/dO displacement ratio")
    plt.xlabel(r"Time before $t_f$ [ps]")
    plt.tight_layout()
    plt.savefig(out("HO_ratio_P.png"), dpi=300, transparent=True)
    plt.close()
    print(f"wrote HO_ratio_P.png (mean={mean_val:.2f}, outlier={outlier:.2f})")
